# /workspace/execution/realtime_feature_engine.py
import sys
from concurrent.futures import ThreadPoolExecutor  # [LAG-FIX-3] 6 TF 並列計算用
from pathlib import Path
from collections import deque
import numpy as np
import pandas as pd
import polars as pl  # [Phase 9b] 統合 .select() 用
import logging

# ▼▼▼ 追加: Numpyの無害な計算警告をミュートしてログをクリーンに保つ ▼▼▼
# 【アーキテクチャ設計メモ】
# Numpyのゼロ除算(RuntimeWarning)に関して、全ての割り算にif文等の安全装置をつけて
# 「完全準拠」させると、C言語レベルのベクトル計算の恩恵が失われシステムが重くなる。
# そのため、ここではあえて警告をミュートし、途中でinfやNaNが発生しても最高速で計算を回す。
# 発生した異常値は、最終出口(calculate_feature_vector)で一括洗浄するのがクオンツとしての正解。
np.seterr(divide="ignore", invalid="ignore")
import warnings

warnings.filterwarnings("ignore", category=RuntimeWarning, module="numpy")
# ▲▲▲ ここまで追加 ▲▲▲

from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Optional, Any
import json
import re
import pickle  # ▼追加: スナップショット保存用
import os  # ▼追加: ファイル存在確認用

# --- /workspace をパスに追加してから blueprint をインポート ---
# engine_1_A と同じルールに統一: sys.path.append が blueprint より必ず先
project_root = Path(__file__).resolve().parents[1]
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

import blueprint as config  # noqa: E402
from blueprint import ATR_BASELINE_DAYS  # noqa: E402
from blueprint import SESSION_BASELINE_DAYS, SESSION_BY_UTC_HOUR  # noqa: E402

# --- core_indicators: Single Source of Truth ---
sys.path.append(str(config.CORE_DIR))
from core_indicators import calculate_atr_wilder, calculate_atr_wilder_disc_aware, calculate_barrier_atr, neutralize_ols  # noqa: E402

# ==================================================================
# 外部モジュール (完全カプセル化クラス群) のインポート
# 各モジュールは calculate_features(data) メソッドで一括計算を行います
# ==================================================================
# --- ATR Ratio計算用：時間足ごとの1日あたりバー数 ---
TIMEFRAME_BARS_PER_DAY: Dict[str, int] = {
    "M0.5": 2880,
    "M1": 1440,
    "M3": 480,
    "M5": 288,
    "M8": 180,
    "M15": 96,
    # "M30": 48,
    # "H1": 24,
    # "H4": 6, "H6": 4, "H12": 2, "D1": 1,  # [FIX] 使用されない時間足
}

# [SESSION-RATIO] 学習側 create_proxy_labels の session_atr_ratio を numpy で完全再現する。
#   学習側: atr / rolling_mean_by("timestamp","{N}d", closed=right).over(session)  (現バー含む)
#   本番側: 同一セッションかつ (now-Nd, now] の窓の atr 平均で割る。
#   セッションは UTC hour → SESSION_BY_UTC_HOUR (計算系UTC0、学習側 dt.hour() と一致)。
#   ns 整数の時刻配列 ts_ns_arr は atr_arr と要素対応 (data_buffers の並行 deque)。
_NS_PER_HOUR = 3_600_000_000_000
_NS_PER_DAY = 86_400_000_000_000
# UTC hour → session を高速引きするための配列 (index=hour 0..23)
_SESSION_HOUR_LUT = np.array(
    [SESSION_BY_UTC_HOUR[h] for h in range(24)], dtype=object
)


def compute_session_atr_ratio_last(
    atr_arr: np.ndarray, ts_ns_arr: np.ndarray, baseline_days: int
) -> float:
    """atr_arr の末尾バーの session_atr_ratio を返す (学習側とビット一致)。"""
    if atr_arr.size == 0 or ts_ns_arr.size == 0:
        return 1.0
    atr_now = float(atr_arr[-1])
    now = int(ts_ns_arr[-1])
    hours = ((ts_ns_arr // _NS_PER_HOUR) % 24).astype(np.int64)  # UTC hour-of-day
    cur_sess = _SESSION_HOUR_LUT[hours[-1]]
    sess_of = _SESSION_HOUR_LUT[hours]
    window_ns = baseline_days * _NS_PER_DAY
    mask = (ts_ns_arr > (now - window_ns)) & (sess_of == cur_sess)
    vals = atr_arr[mask]
    baseline = float(vals.mean()) if vals.size > 0 else atr_now
    return atr_now / (baseline + 1e-10)


from execution.realtime_feature_engine_1A_statistics import FeatureModule1A
from execution.realtime_feature_engine_1B_timeseries import FeatureModule1B
from execution.realtime_feature_engine_1C_technical import FeatureModule1C
from execution.realtime_feature_engine_1D_volume import FeatureModule1D
from execution.realtime_feature_engine_1E_signal import FeatureModule1E
from execution.realtime_feature_engine_1F_experimental import FeatureModule1F


@dataclass
class Signal:
    """
    リアルタイムエンジンが main.py に返すシグナルオブジェクト
    """

    features: np.ndarray  # 純化済み特徴量ベクトル (1, 304)
    timestamp: datetime  # シグナル発生時刻 (バーのクローズ時刻)
    timeframe: str  # シグナル発生の時間足 (e.g., "M1", "M15")
    market_info: Dict[str, Any]  # リスクエンジンに渡す市場文脈 (V4 V5ルール)
    atr_value: float  # 動的バリア計算用の定規(ATR)
    close_price: float  # 動的バリア計算用の起点(現在価格)
    # [FIX-3] feature_dict を追加 — main.py で signal.feature_dict にアクセスするために必要
    feature_dict: Dict[str, float] = None  # type: ignore[assignment]

    def __post_init__(self):
        if self.feature_dict is None:
            self.feature_dict = {}


class RealtimeFeatureEngine:
    """
    【Project Chimera V5: オーケストレーター】
    15時間足の独立したNumpyバッファを保持し、M0.5バーを起点とした
    全時間足の同期・リサンプリング・OLS純化・ベクトル生成を司る司令塔。
    特徴量計算そのものは外部のNumbaモジュール(1A〜1F)へ委譲する。
    """

    ALL_TIMEFRAMES = {
        "M0.5": 0.5,  # [FIX] orthogonal特徴量リストに存在するため有効化。30秒足（M1の半分）
        "M1": 1,
        "M3": 3,
        "M5": 5,
        "M8": 8,
        "M15": 15,
        # "M30": 30,
        # "H1": 60,
        # "H4": 240,
        # "H6": 360,   # [FIX] orthogonal全4ファイルで gain 0.006%以下・削除
        # "H12": 720,  # [FIX] orthogonal全4ファイルで gain 0.004%以下・削除
        # "D1": 1440,  # [FIX] orthogonalリストに存在しないためスキップ
        # "W1": 10080,  # [V5] orthogonalリストに存在しないためスキップ
        # "MN": 43200,  # [V5] orthogonalリストに存在しないためスキップ
        # "tick": None, # [V5] orthogonalリストに存在しないためスキップ
    }

    TF_RESAMPLE_RULES = {
        # "M0.5": "30s",  # M0.5はm1_dataframeの起点のためスキップ
        "M1": "1min",
        "M3": "3min",
        "M5": "5min",
        "M8": "8min",
        "M15": "15min",
        # "M30": "30min",
        # "H1": "1h",
        # "H4": "4h",   # [FIX] orthogonalリストに存在しないためスキップ
        # "H6": "6h",   # [FIX] gain 0.006%以下・削除
        # "H12": "12h", # [FIX] gain 0.004%以下・削除
        # "D1": "1D",   # [FIX] orthogonalリストに存在しないためスキップ
        # "W1": "1W",   # [V5] スキップ
        # "MN": "1MS",  # [V5] スキップ
    }

    OHLCV_COLS = ["open", "high", "low", "close", "volume"]
    DEFAULT_LOOKBACK = 200
    ATR_CALC_PERIOD = 13
    OLS_WINDOW_DEFAULT = 2016  # 純化用 OLS 回帰窓のフォールバック値
    # [Phase 9d 発見 #63] TF 毎の OLS 窓は blueprint.NEUTRALIZATION_CONFIG["HF"]
    # ["window_per_tf"] が SSoT (Phase 10 設計、2 日案):
    #     M0.5: 5760  /  M1: 2880  /  M3:  960
    #     M5:    576  /  M8:  360  /  M15: 192
    # 本属性は blueprint に未登録の TF のみフォールバックとして使用される。

    @classmethod
    def _get_ols_window(cls, tf_name: str) -> int:
        """[Phase 9d 発見 #63] TF 毎の OLS 純化窓を blueprint から取得。

        学習側 2_G_alpha_neutralizer.py が
        ``blueprint.NEUTRALIZATION_CONFIG["HF"]["window_per_tf"]`` を SSoT
        として使用する (Phase 10 設計、2 日案)。本番側もここで完全に同じ値を
        引くことで OLS 純化結果の数値的整合性を保証する。

        blueprint に該当 TF のエントリがない場合は ``OLS_WINDOW_DEFAULT``
        (=2016) にフォールバックする。これは Phase 9b 以前の固定窓挙動と等価。

        Args:
            tf_name: 時間足名 (例: "M0.5", "M1", "M3", "M5", "M8", "M15")

        Returns:
            int: OLS 回帰窓のサイズ (deque maxlen として使う本数)
        """
        try:
            hf_config = config.NEUTRALIZATION_CONFIG.get("HF", {})
            per_tf = hf_config.get("window_per_tf", {})
            window = per_tf.get(tf_name)
            if window is not None and int(window) > 0:
                return int(window)
        except Exception:
            # blueprint 未ロード/属性欠落 等の防御的フォールバック
            pass
        return cls.OLS_WINDOW_DEFAULT

    @classmethod
    def _filter_to_closed_buckets_warmup(
        cls,
        resampled_df: pd.DataFrame,
        tf_name: str,
        m05_history_pd: pd.DataFrame,
    ) -> pd.DataFrame:
        """[Phase 9d 発見 #64] warmup resample 結果から「形成中の最後のバー」を除外。

        `_resample_and_update_buffer` (runtime, 発見 #62) と同じ timestamp ベース
        close 判定を、warmup の一括 resample にも適用する。

        判定式:
            bucket_close_ts = m05_history_pd.index[-1] + M0.5_freq_sec  (= +30s)
            bucket [label, label + tf_freq_sec) は次の条件で closed:
                label + tf_freq_sec <= bucket_close_ts

        例: warmup の最後の M0.5 が 2026-04-01 00:00:00 のとき、
            bucket_close_ts = 00:00:30
            M1 [00:00, 00:01): 00:00 + 60s = 00:01 ≤ 00:00:30? NO → 除外
            M15 [00:00, 00:15): 00:00 + 900s = 00:15 ≤ 00:00:30? NO → 除外
            M15 [23:45, 00:00): 23:45 + 900s = 00:00 ≤ 00:00:30? YES → 残る

        Args:
            resampled_df: TF にリサンプル済み (index=timestamp, OHLCV columns)
            tf_name: 当該 TF 名 (M1/M3/M5/M8/M15 等)
            m05_history_pd: warmup の M0.5 履歴 (index=timestamp, sorted)

        Returns:
            形成中の最後のバーを除外した DataFrame
        """
        if resampled_df.empty or m05_history_pd.empty:
            return resampled_df
        m05_latest_ts = m05_history_pd.index[-1]
        m05_freq_sec = cls._TF_FREQ_SECONDS.get("M0.5", 30)
        bucket_close_ts = m05_latest_ts + pd.Timedelta(seconds=m05_freq_sec)

        tf_freq_sec = cls._TF_FREQ_SECONDS.get(tf_name, 0)
        if tf_freq_sec <= 0:
            # 未知 TF: 安全側で旧来の iloc[:-1] 同等 (最後を除外)
            return resampled_df.iloc[:-1] if len(resampled_df) > 0 else resampled_df

        tf_freq_td = pd.Timedelta(seconds=tf_freq_sec)
        closed_mask = (resampled_df.index + tf_freq_td) <= bucket_close_ts
        return resampled_df[closed_mask]

    @staticmethod
    def _ffill_lookup_market_proxy(
        market_proxy_cache: pd.DataFrame,
        search_ts: datetime,
    ) -> float:
        """[Phase 9d 発見 #65] thread-safe な ffill lookup を numpy で実装。

        pandas DatetimeIndex.get_indexer(method="ffill") の代替実装。
        pandas Index 機構 (lazy hashtable initialization) を完全に迂回し、
        並列実行下でも race condition が発生しない。

        セマンティクスは pandas 版と完全に同等:
            旧: proxy_cache_unique = market_proxy_cache[
                    ~market_proxy_cache.index.duplicated(keep="last")
                ].sort_index()
                idx = proxy_cache_unique.index.get_indexer(
                    [search_ts], method="ffill"
                )[0]
                return iloc[idx]["market_proxy"] if idx != -1 else 0.0

            新: numpy snapshot → stable sort → keep_last dedup →
                searchsorted(side="right") - 1 → 該当値 or 0.0

        学習側 2_G の `join_asof(strategy="backward") + fill_null(0.0)` と
        数値完全一致を維持する。

        Args:
            market_proxy_cache: index=DatetimeIndex (tz-aware), columns=["market_proxy"]
            search_ts: 検索対象 timestamp (tz-aware datetime)

        Returns:
            float: search_ts 以前の最新 proxy 値。該当なし or 非有限値の場合 0.0
        """
        if market_proxy_cache.empty:
            return 0.0

        # ── 1. timestamp と value を numpy ndarray にスナップショット ───
        # pandas 2.x で DatetimeIndex の内部単位が "ns" でなく "us" (μs) や
        # "ms" の場合がある (特に tz-aware index)。`asi8` はその単位の int を
        # 返すため、`pd.Timestamp.value` (常に ns) と比較できない。
        # → dtype.unit を判定して ns にスケール統一する。
        # `.values.astype(np.int64)` は tz-aware で object 配列を返して
        # silent に壊れるので使用しない。
        idx = market_proxy_cache.index
        if isinstance(idx, pd.DatetimeIndex) and hasattr(idx, "asi8"):
            asi8 = np.asarray(idx.asi8, dtype=np.int64)
            unit = getattr(idx.dtype, "unit", "ns")
            scale_map = {
                "ns": np.int64(1),
                "us": np.int64(1_000),
                "ms": np.int64(1_000_000),
                "s":  np.int64(1_000_000_000),
            }
            scale = scale_map.get(unit, None)
            if scale is None:
                # 未知の unit → 安全側に per-element 変換
                ts_ns = np.fromiter(
                    (pd.Timestamp(t).value for t in idx),
                    dtype=np.int64,
                    count=len(idx),
                )
            else:
                ts_ns = asi8 * scale
        else:
            # フォールバック (DatetimeIndex 以外)
            ts_ns = np.fromiter(
                (pd.Timestamp(t).value for t in idx),
                dtype=np.int64,
                count=len(idx),
            )
        proxy_arr = market_proxy_cache["market_proxy"].to_numpy(
            dtype=np.float64, copy=False
        )

        if ts_ns.size == 0:
            return 0.0

        # ── 2. timestamp で stable sort ──────────────────────────────────
        # stable sort により、同 timestamp は元の順序が保持される。
        # この性質が次ステップの "keep last" 等価性を保証する。
        sort_idx = np.argsort(ts_ns, kind="stable")
        sorted_ts_ns = ts_ns[sort_idx]
        sorted_proxy = proxy_arr[sort_idx]

        # ── 3. 重複除去: 同 timestamp の "最後" を残す ──────────────────
        # pandas duplicated(keep="last") = 最後の出現以外を duplicate 扱い。
        # stable sort 後、同 timestamp グループの末尾要素 = 元データの最新出現。
        # → "次の要素と timestamp が違う" もしくは "最後の要素" のみ残す mask。
        n = sorted_ts_ns.size
        if n > 1:
            keep_mask = np.empty(n, dtype=np.bool_)
            keep_mask[-1] = True
            keep_mask[:-1] = sorted_ts_ns[:-1] != sorted_ts_ns[1:]
            sorted_ts_ns = sorted_ts_ns[keep_mask]
            sorted_proxy = sorted_proxy[keep_mask]

        # ── 4. binary search (numpy, thread-safe) ────────────────────────
        # search_ts を UTC ns に変換。pd.Timestamp.value は常に UTC ns。
        # tz-naive の場合は UTC として扱う (元コードと同じ挙動)。
        from datetime import timezone as _tz
        if isinstance(search_ts, pd.Timestamp):
            if search_ts.tzinfo is None:
                search_ts_pd = search_ts.tz_localize("UTC")
            else:
                search_ts_pd = search_ts.tz_convert("UTC")
        else:
            # 通常の datetime
            if search_ts.tzinfo is None:
                search_ts_pd = pd.Timestamp(search_ts).tz_localize("UTC")
            else:
                search_ts_pd = pd.Timestamp(search_ts).tz_convert("UTC")
        search_ns = np.int64(search_ts_pd.value)

        # side="right" → search_ns 以下の最後の位置 + 1 を返す
        # -1 で「search_ns 以下の最後の位置」になる (= ffill)
        idx = int(np.searchsorted(sorted_ts_ns, search_ns, side="right")) - 1
        if idx < 0:
            return 0.0
        val = float(sorted_proxy[idx])
        if not np.isfinite(val):
            return 0.0
        return val


    # ─────────────────────────────────────────────────────────────────
    # [DISC-FLAG SSoT] 学習側 s1_1_B_build_ohlcv.py の TIMEFRAME_FREQ_SECONDS
    # / DISC_GAP_MULTIPLIER と完全同一。本番側 disc 計算の唯一の真実源。
    #
    # 発見 #60 (Phase 9d 追加修正): 通常 poll_m3_bar 経路で EA から送られる
    # M0.5 バー dict に disc キーが含まれず、_append_bar_to_buffer 内で
    # bar_dict.get("disc", False) が常に False を返す構造的欠陥があった。
    # これは Phase 5 で潰したバグ A (週末跨ぎ ATR 汚染) の本質的問題が、
    # 短時間ギャップ (45-360秒) という形で生き残っていた状態。
    #
    # 修正: disc 計算を _compute_disc_flag に一元化し、_append_bar_to_buffer を
    # 通る全経路 (M0.5 直接追加 / M3-M15 リサンプル / gap-fill / warmup_only) で
    # 自動的に正しい disc が立つ構造に変更。閾値は 1.5x ルール (学習側と同一)。
    # ─────────────────────────────────────────────────────────────────
    _TF_FREQ_SECONDS = {
        "M0.5": 30,
        "M1": 60,
        "M3": 180,
        "M5": 300,
        "M8": 480,
        "M15": 900,
        "M30": 1800,
        "H1": 3600,
        "H4": 14400,
        "H6": 21600,
        "H12": 43200,
        "D1": 86400,
        "W1": 604800,
        "MN": 2592000,
    }
    _DISC_GAP_MULTIPLIER = 1.5  # 想定間隔の何倍を超えたら不連続とみなすか

    def __init__(
        self,
        feature_list_path: str = str(config.S3_FEATURES_FOR_TRAINING_V5),
        qa_state_artifacts: Optional[Dict[str, Dict]] = None,
    ):
        """
        Args:
            feature_list_path: 特徴量名簿ファイル
            qa_state_artifacts: [Phase 9d 発見 #66 Phase D-3] 学習側で生成された
                QAState 最終 EWM 状態の dict。形式:
                  {engine_id: {(tf, feat): {ewm_mean, ewm_var, ewm_n}}}
                main.py 起動時に S3_QA_STATES_DIR/qa_state_e1{a..f}.pkl を load
                して渡す。None の場合は旧挙動 (warmup loop で seed) で fallback。
        """
        self.logger = logging.getLogger("♾️Chimera♾️.FEAT")

        # [Phase 9d 発見 #66 Phase D-3] artifact を instance var に保持。
        # 後段 (qa_states 初期化箇所) で _extract_artifact が参照する。
        self._qa_state_artifacts = qa_state_artifacts

        # risk_config.json を読み込み (min_atr_threshold等を動的取得)
        try:
            with open(config.CONFIG_RISK, "r") as f:
                self.risk_config = json.load(f)
        except Exception:
            self.logger.warning(
                "risk_config.json の読み込みに失敗しました。デフォルト値を使用します。"
            )
            self.risk_config = {}

        # 1. 特徴量名簿をロード
        try:
            self.feature_list = self._load_feature_list(feature_list_path)
            self.logger.info(
                f"Feature roster loaded ({len(self.feature_list)} items)."
            )
        except Exception as e:
            self.logger.critical(f"Feature roster load failed ({feature_list_path}): {e}")
            raise

        # 2. 名簿から各時間足の最大ルックバック期間を特定
        self.lookbacks_by_tf = self._parse_feature_list_and_get_lookbacks(
            self.feature_list
        )

        # 3. 独立したデータバッファを初期化
        self.data_buffers: Dict[str, Dict[str, deque]] = {}
        self.is_buffer_filled: Dict[str, bool] = {}
        self.last_bar_timestamps: Dict[str, Optional[pd.Timestamp]] = {}
        self.latest_features_cache: Dict[str, Dict[str, float]] = {}

        # [発見#D対応] calculate_feature_vector で「純化済み('_neutralized_'を含む)
        # でもなく、許可リスト(NON_NEUTRALIZED_BASE_NAMES)にも該当しない」特徴量を
        # 検知した際に警告ログを出すが、毎バー出力されるとスパムになるため
        # 一度警告した名前は記録して再警告しない。
        self._warned_unknown_features: set = set()

        for tf_name in self.ALL_TIMEFRAMES.keys():
            if self.ALL_TIMEFRAMES[tf_name] is None:
                continue

            if tf_name not in self.lookbacks_by_tf:
                self.lookbacks_by_tf[tf_name] = self.DEFAULT_LOOKBACK
                self.logger.debug(
                    f"  -> {tf_name:<3} バッファ初期化 (Default: {self.DEFAULT_LOOKBACK})"
                )
            else:
                self.logger.info(
                    f"  -> {tf_name:<5} buffer init (lookback: {self.lookbacks_by_tf[tf_name]})"
                )

            lookback = self.lookbacks_by_tf[tf_name]

            self.data_buffers[tf_name] = {
                col: deque(maxlen=lookback) for col in self.OHLCV_COLS
            }
            # [DISC-FLAG] 不連続フラグバッファ: resampleでNaNになった足はTrue
            # discフラグがTrueの足ではTR計算時に前Closeを使わず H-L のみで計算する
            self.data_buffers[tf_name]["disc"] = deque(maxlen=lookback)
            # [SESSION-RATIO] atr_arr と要素対応する UTC ns 時刻の並行 deque。
            #   同一セッション×N日窓の分母計算に使う。OHLCV/disc と同一タイミングで append。
            self.data_buffers[tf_name]["__bar_ts_ns__"] = deque(maxlen=lookback)
            self.is_buffer_filled[tf_name] = False
            self.last_bar_timestamps[tf_name] = None
            self.latest_features_cache[tf_name] = {}

        # 4. M1データを保持するDeque (リサンプリング元)
        max_lookback_val = (
            max(self.lookbacks_by_tf.values()) if self.lookbacks_by_tf else 1000
        )
        max_m05_bars_needed = max_lookback_val * 2880 + 1000
        self.m05_dataframe: deque[Dict[str, Any]] = deque(maxlen=max_m05_bars_needed)

        # [GAP-DETECT §11.34.16-T 層3] M0.5 バーの連続性監視 (多重防御の最終 backstop)。
        # 層1 (EA 最終バケット flush) と層2 (bridge 完全性照合) で取りこぼしは源で塞ぐが、
        # それらを抜けた残りを process_new_m05_bar の単一入口で検知する。末尾との時刻差が
        # 30s (M0.5 の 1 足) を超えたら欠損と判定し、区間 (start, end) を記録。main が
        # pop_detected_m05_gaps() で受け取り、MT5 (S1 と同源) から pinpoint 再取得して
        # 単調順で流し直す (deque 末尾 append 不変条件を保持)。本物の落ちなら足は必ず
        # MT5 に在るので再取得は成功する。rfe は検知のみ (再取得は bridge を持つ main の責務)。
        self._detected_m05_gaps: List[tuple] = []
        self._M05_FREQ_SEC: int = 30  # M0.5 = 30 秒足
        self._GAP_DETECT_MAX_BARS: int = 20  # [要修正②] これ超の欠落は正規市場ギャップ扱い

        # 5. 純化(OLS)用 状態保持バッファ
        self.proxy_feature_buffers: Dict[str, Dict[str, deque]] = {}
        self.ols_state: Dict[str, Dict[str, Dict[str, float]]] = {}

        PROXY_FEATURES = [
            "atr",
            "log_return",
            "price_momentum",
            "rolling_volatility",
            "volume_ratio",
        ]

        for tf_name in self.data_buffers.keys():
            # [Phase 9d 発見 #63] OLS純化バッファ: TF 毎可変窓を blueprint から取得。
            # 学習側 2_G の Phase 10 設計と一致させる (M0.5=5760, M1=2880, ...)。
            # blueprint 未登録 TF は OLS_WINDOW_DEFAULT (=2016) にフォールバック。
            tf_ols_window = self._get_ols_window(tf_name)
            self.proxy_feature_buffers[tf_name] = {
                feat: deque(maxlen=tf_ols_window) for feat in PROXY_FEATURES
            }
            self.proxy_feature_buffers[tf_name]["market_proxy"] = deque(
                maxlen=tf_ols_window
            )
            # [計測基盤] OLS deque 各要素に対応するバー時刻 (close_ts) を記録する並行 deque。
            #   x_deque/y_deque と同一タイミングで append され length が常に一致する。
            #   OLS 計算には一切関与しない (更新ループは latest_proxy_features で回るため、
            #   buffer キーを総舐めする箇所は存在しない)。exact-timestamp での
            #   train-serve 突合を可能にし、positional 推測アライメントを廃するための土台。
            self.proxy_feature_buffers[tf_name]["__bar_ts__"] = deque(
                maxlen=tf_ols_window
            )

            self.ols_state[tf_name] = {}
            # 各エントリは _update_incremental_ols で特徴量登場時に動的初期化される

        self.logger.info(f"M0.5 deque buffer init (maxlen: {max_m05_bars_needed})")

        # [診断 L1] バッファ容量と特徴量計算要求の整合性を検証
        # 学習側 timeframe_bars_per_day と本番側 lookbacks_by_tf のズレを起動時に検出する
        self._validate_buffer_sizes()

        # 6. JITコンパイルのウォームアップ
        self._warmup_jit()

        # 7. 各時間足・各モジュール(1A〜1F)のQAStateを初期化
        # [乖離①修正] 学習側 apply_quality_assurance_to_group と等価のQA処理を有効化
        # lookback_barsは時間足ごとの1日バー数（M3=480等）を使用
        #
        # [Phase 9d 発見 #66 Phase D-3] artifact load 経路:
        #   self._qa_state_artifacts は __init__ 直前に main.py から渡される
        #   学習側 5 年分の EWM 最終状態 dict。形式:
        #     {engine_id: {(tf, feat): {ewm_mean, ewm_var, ewm_n}}}
        #   QAState 構築時に該当 engine × TF の artifact を抽出して渡す。
        #   artifact 不在の場合 (= self._qa_state_artifacts is None or 該当エントリ無し)
        #   は旧挙動 (warmup loop で 577 update 経由の seed) で fallback。
        self.qa_states: Dict[str, Dict[str, Any]] = {}
        for tf_name in self.ALL_TIMEFRAMES.keys():
            if self.ALL_TIMEFRAMES[tf_name] is None:
                continue
            lb = TIMEFRAME_BARS_PER_DAY.get(tf_name, 1440)

            # 各 engine 別に該当 TF の artifact を抽出 (None なら旧挙動)
            def _extract_artifact(engine_id: str) -> Optional[Dict[str, Dict[str, float]]]:
                if not getattr(self, "_qa_state_artifacts", None):
                    return None
                engine_artifact = self._qa_state_artifacts.get(engine_id)
                if not engine_artifact:
                    return None
                # engine_artifact: {(tf, feat): {ewm_mean, ewm_var, ewm_n}}
                # → 該当 TF の {feat: {...}} に変換
                return {
                    feat: state
                    for (t, feat), state in engine_artifact.items()
                    if t == tf_name
                }

            self.qa_states[tf_name] = {
                "1A": FeatureModule1A.QAState(
                    lookback_bars=lb, artifact=_extract_artifact("e1a")
                ),
                "1B": FeatureModule1B.QAState(
                    lookback_bars=lb, artifact=_extract_artifact("e1b")
                ),
                "1C": FeatureModule1C.QAState(
                    lookback_bars=lb, artifact=_extract_artifact("e1c")
                ),
                "1D": FeatureModule1D.QAState(
                    lookback_bars=lb, artifact=_extract_artifact("e1d")
                ),
                "1E": FeatureModule1E.QAState(
                    lookback_bars=lb, artifact=_extract_artifact("e1e")
                ),
                "1F": FeatureModule1F.QAState(
                    lookback_bars=lb, artifact=_extract_artifact("e1f")
                ),
            }
        self.logger.info("✓ QAState initialized for all timeframes.")

        # Phase D-3 artifact load 状況のサマリーをログ出力
        if getattr(self, "_qa_state_artifacts", None):
            engines_loaded = sorted(self._qa_state_artifacts.keys())
            total_entries = sum(
                len(art) for art in self._qa_state_artifacts.values()
            )
            self.logger.info(
                f"[Phase D-3] QAState artifact load 済: "
                f"engines={engines_loaded}, total_entries={total_entries}"
            )
        else:
            self.logger.warning(
                "[Phase D-3] QAState artifact なし → 旧挙動 "
                "(warmup loop で seed) で fallback します。"
                "Train-Serve Skew が残存する可能性があります。"
            )

        # [LAG-FIX-3] 6 TF 並列計算用の ThreadPoolExecutor を初期化
        # process_new_m05_bar の step3 (全 TF 強制再計算) を並列化することで、
        # 6 TF × 75-110ms 直列 = 547ms を、理論上 ~110ms (最遅 TF 律速) 程度まで短縮可能。
        # 各 TF の処理は独立 (異なる self.data_buffers[tf]/proxy_feature_buffers[tf]/
        # latest_features_cache[tf] にアクセス) のため thread safety 問題なし。
        # save_state の対象 dict には含めないため pickle 化は問題なし。
        self._tf_executor = ThreadPoolExecutor(
            max_workers=6, thread_name_prefix="tf_recalc"
        )

    def _load_feature_list(self, path: str) -> List[str]:
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(f"Feature list not found: {path}")

        if p.suffix.lower() == ".json":
            with open(p, "r", encoding="utf-8") as f:
                features = json.load(f)
            return features
        else:
            with open(p, "r", encoding="utf-8") as f:
                return [line.strip() for line in f if line.strip()]

    def _warmup_jit(self):
        """各モジュールの完全カプセル化メソッドを呼び出し、JITコンパイルを済ませる"""
        self.logger.info("Starting JIT warmup of external modules...")
        try:
            # OHLCVのダミーデータ（辞書）を作成
            dummy_arr = np.cumsum(np.random.randn(300)).astype(np.float64) + 1000.0
            dummy_data = {
                "open": dummy_arr,
                "high": dummy_arr + 10.0,
                "low": dummy_arr - 10.0,
                "close": dummy_arr + np.random.randn(300),
                "volume": np.abs(np.random.randn(300) * 100),
            }

            # 各モジュールのメインメソッドにダミーデータを流し込む
            _ = FeatureModule1A.calculate_features(dummy_data)
            _ = FeatureModule1B.calculate_features(dummy_data)
            _ = FeatureModule1C.calculate_features(dummy_data)
            _ = FeatureModule1D.calculate_features(dummy_data)
            _ = FeatureModule1E.calculate_features(dummy_data)
            _ = FeatureModule1F.calculate_features(dummy_data)

            self.logger.info("✓ JIT warmup done.")
        except Exception as e:
            self.logger.warning(f"JIT warmup warning: {e}")

    def _parse_feature_list_and_get_lookbacks(
        self, feature_list: List[str]
    ) -> Dict[str, int]:
        """名簿に登場する TF を抽出し、各 TF の data_buffers maxlen を決定する。

        【設計】
            data_buffers (OHLCV 用) の maxlen は「特徴量計算に必要な本数」だけで決まる。
            純化用の OLS_WINDOW は別バッファ proxy_feature_buffers で管理されているため、
            ここでは混入させない。3 つの概念は完全に独立:

              A. 特徴量の窓 (各特徴量の rolling_*(N) の N) — モジュール毎に決まる
              B. data_buffers の maxlen      — A の最長窓 + マージン (本メソッドが返す値)
              C. proxy_feature_buffers の maxlen = OLS_WINDOW (別管理、TF 毎可変)
                 - Phase 9d 発見 #63 で blueprint.NEUTRALIZATION_CONFIG["HF"]
                   ["window_per_tf"] を SSoT として参照する設計に変更
                 - 旧設計: 全 TF 共通 2016 (Phase 9b 以前)
                 - 新設計: M0.5=5760, M1=2880, M3=960, M5=576, M8=360, M15=192
                          (Phase 10 の 2 日案、学習側 2_G_alpha_neutralizer と一致)

        【M0.5 バッファ不足バグの修正】
            旧実装: 全 TF 一律 SAFE_MIN_LOOKBACK=2016 (純化窓を data_buffers に混入) + 100 = 2116
                    → M0.5 では 1D vol_ma1440 (= 2880 本必要) が NaN を返す致命バグ
                       (学習側 timeframe_bars_per_day["M0.5"]=2880 と数値乖離)
                    → e1d_obv_rel_M0.5 (gain 7,463) など 5 特徴量が常時 0 で死蔵

            新実装: TF 毎に「特徴量計算に必要な最大本数」を個別決定。OLS_WINDOW は
                    別バッファに任せ、data_buffers から完全に切り離す。

        【PER_TF_FEATURE_MAX の根拠】
            各 TF で必要な最大窓 = 各モジュールの最大窓 max:
              1A:  100, 1B: 100, 1C: 200,
              1D: lookback_bars (= TIMEFRAME_BARS_PER_DAY[tf]),
              1E: spectral_flux の window×2 = 1024,
              1F:  100
            → max を取ると:
              M0.5: max(100, 100, 200, 2880, 1024, 100) = 2880
              M1:   max(100, 100, 200, 1440, 1024, 100) = 1440
              M3:   max(100, 100, 200,  480, 1024, 100) = 1024
              M5:   max(100, 100, 200,  288, 1024, 100) = 1024
              M8:   max(100, 100, 200,  180, 1024, 100) = 1024
              M15:  max(100, 100, 200,   96, 1024, 100) = 1024
        """
        tf_pattern = re.compile(r"_(M[0-9\.]+|H[0-9]+|D[0-9]+|W[0-9]+|MN|tick)$")
        seen_tfs = set()

        for feature_name in feature_list:
            tf_match = tf_pattern.search(feature_name)
            if tf_match:
                seen_tfs.add(tf_match.group(1))

        # TF 毎に必要な特徴量計算用バッファ本数 (上記コメント参照)
        # dict 定義順 (M0.5 → M1 → M3 → M5 → M8 → M15) でログ出力するため、
        # PER_TF_FEATURE_MAX の順序がそのまま出力順になる。
        #
        # 【案 D 網羅監査の結果 (Phase 9b 後)】
        # 全モジュールの最大窓:
        #   1A: 100  (for window in [5,10,20,50,100])
        #   1B: 100  (for window in [50,100])
        #   1C: 100  (window_sizes["general"] = [10,20,50,100])
        #   1D: 1440 (rolling_quantile(0.8/0.6, window_size=1440) ← 固定値)
        #        + lookback_bars (TF 毎可変、M0.5 で 2880)
        #   1E: 128  (rolling_max(128) for spectral_flux)
        #   1F: 100  (window_sizes 全カテゴリの最大)
        # → 数値固定窓の絶対最大は 1440 (1D rolling_quantile)。
        # → 全 TF で最低 1440 本のバッファが必要。
        #
        # 【Phase E+ EMA: recurrence 系の warmup 要件追加】
        # rolling_*(N) の N と「特徴量計算に必要な本数」だけでなく、recurrence
        # 系 (EMA, Wilder smoothing 等) の収束 warmup も考慮が必要。
        # 各 recurrence の α と必要 warmup (rtol=1e-7 通過):
        #   - EMA span=200 (1C: ema_200 / ema_deviation_200):
        #       α=2/201≈0.00995 → warmup ≈ 1620 bar  ← 今回の真因
        #       ※特に ema_deviation_200 = (close-ema)/ema*100 は ema が close に
        #          肉薄する瞬間に分母小特異点で rel_diff が増幅される。これを
        #          shadow_mode rtol=1e-7 で通すには ULP オーダーまで warmup
        #          完全収束させる必要 (3500 bar 程度)。
        #   - EMA span≤100 (MACD/PO/Wilder/KAMA/TRIX/TSI):
        #       warmup ≤ 880 bar → 旧 maxlen 1540 で吸収済
        #   - QA bounds EWM (half_life=bars/day, 最大 M0.5 で 66907 bar 必要):
        #       Phase D-3 で artifact ロード方式で構造的に解決済 (本ファイル
        #       L98-119 の QAState seed artifact、from scratch ではなく学習側
        #       成熟状態からの stream 増分更新)
        #
        # 修正履歴:
        #   旧 (Phase 9b 初期 hotfix): M3-M15 = 1024 (1E spectral_flux のみ考慮)
        #   新 (Phase 9b 案 A): M3-M15 = 1440 (1D rolling_quantile を追加考慮)
        #     → e1d_hv_regime_50 が学習側と整合 (現在 gain=0 で AI 未使用、構造的整合のみ)
        #   新 (Phase E+ EMA, today phase 1):
        #     M0.5: 2880 → 3500, M1-M15: 1440 → 2000
        #     → e1c failing 3,182 → 16 (M1: 14, M3: 2)
        #     → 残 16 cells は ema_deviation_200 の分母小特異点で warmup 残差
        #        (8e-10 オーダー) が rel_diff に増幅される現象
        #   新 (Phase E+ EMA, today phase 2 = 確定):
        #     全 TF: 3500 統一
        #     → warmup 3499 → (1-α)^N ≈ 8e-16 (M0.5 と同等の ULP 完全収束)
        #     → e1c failing 16 → 0 確定 (Phase E+ EMA 完全合格)
        #     → M5-M15 も予防的に 3500 に統一 (将来の test 期間延長や別 EMA-200
        #        系 feature 追加時のリスク排除、設定統一でシンプル化)
        PER_TF_FEATURE_MAX = {
            "M0.5": 3500,   # EMA-200 warmup 3499 → (1-α)^N ≈ 8e-16 (完全 ULP)
            "M1":   3500,   # 同上 (旧 2000 → 3500、14 cells failing を解消)
            "M3":   3500,   # 同上 (旧 2000 → 3500、 2 cells failing を解消)
            "M5":   3500,   # 同上 (現状 failing 0 だが予防的に統一)
            "M8":   3500,   # 同上
            "M15":  3500,   # 同上
        }
        DEFAULT_FEATURE_MAX = 3500  # 未知 TF のフォールバック (EMA-200 ULP 完全収束に必要な値)

        final_lookbacks = {}
        # PER_TF_FEATURE_MAX の dict 定義順で処理 → ログも M0.5 → M15 の順になる
        for tf_name_parsed in PER_TF_FEATURE_MAX.keys():
            if tf_name_parsed not in seen_tfs:
                continue
            req_size = PER_TF_FEATURE_MAX.get(tf_name_parsed, DEFAULT_FEATURE_MAX)
            final_lookbacks[tf_name_parsed] = req_size + 100  # 安全マージン
            tf_ols_window = self._get_ols_window(tf_name_parsed)
            self.logger.info(
                f"  -> {tf_name_parsed:<5} max lookback: {final_lookbacks[tf_name_parsed]} "
                f"(features; purify window {tf_ols_window} separate)"
            )

        # PER_TF_FEATURE_MAX に未登録の TF があれば末尾に追加 (ソート済み、ログ出力)
        for tf_name_parsed in sorted(seen_tfs - set(PER_TF_FEATURE_MAX.keys())):
            req_size = DEFAULT_FEATURE_MAX
            final_lookbacks[tf_name_parsed] = req_size + 100
            tf_ols_window = self._get_ols_window(tf_name_parsed)
            self.logger.info(
                f"  -> {tf_name_parsed:<5} max lookback: {final_lookbacks[tf_name_parsed]} "
                f"(features; purify window {tf_ols_window} は別バッファ、未登録TF)"
            )

        return final_lookbacks

    def _validate_buffer_sizes(self) -> None:
        """
        【診断 L1: 静的設定値検証】
        各 TF の data_buffers maxlen が、各モジュールが要求する最大窓を
        満たすか検証する。学習側 (engine_1_X.timeframe_bars_per_day) との
        ズレや、特徴量変更時の設定漏れを起動時に検出する。

        本番運用で過去発生したバグ:
            M0.5 で deque maxlen=2116 だったが、1D `vol_ma1440` が
            rolling_mean(2880) を要求 → 永遠に NaN を返し、QA で 0 にクリップ。
            学習側で gain 7,463 の `e1d_obv_rel_M0.5` が常時 0 で死蔵していた。

        修正 (Phase 9b 後の hotfix):
            data_buffers maxlen を TF 毎可変に変更
            (M0.5: 2980, M1: 1540, M3-M15: 1124)。
            本メソッドはこの修正が以降のバージョンでも維持されているかを
            起動時に保証する役割を果たす。

        各 TF の必要最大窓 (案 D 網羅監査の結果):
            M0.5: 2880  (1D vol_ma1440 = bars_per_day["M0.5"])
            M1:   1440  (1D vol_ma1440 + 1D rolling_quantile(1440))
            M3:   1440  (1D rolling_quantile(1440) ← Phase 9b 案 A で 1024→1440)
            M5:   1440  (同上)
            M8:   1440  (同上)
            M15:  1440  (同上)
        """
        PER_TF_FEATURE_MAX = {
            "M0.5": 2880,
            "M1":   1440,
            "M3":   1440,  # Phase 9b 案 A: 1D rolling_quantile(1440) を追加考慮
            "M5":   1440,
            "M8":   1440,
            "M15":  1440,
        }

        self.logger.info("--- Buffer capacity check (diag L1) ---")
        all_ok = True
        for tf_name, required in PER_TF_FEATURE_MAX.items():
            if tf_name not in self.lookbacks_by_tf:
                continue
            actual = self.lookbacks_by_tf[tf_name]
            if actual < required:
                self.logger.error(
                    f"  ❌ {tf_name:<5} バッファ容量不足: maxlen={actual} < 必要={required}。"
                    f" 該当 TF の長期 rolling 特徴量が NaN→0 で死蔵します!"
                    f" data_buffers の lookback_bars 設定を確認してください。"
                )
                all_ok = False
            else:
                self.logger.info(
                    f"  ✓  {tf_name:<5} buffer capacity OK: maxlen={actual} >= need={required}"
                )
        if all_ok:
            self.logger.info("✓ All TF buffer capacities meet training requirements.")
        else:
            self.logger.error(
                "❌ バッファ容量不足の TF があります。学習側との特徴量分布が乖離します。"
            )

    def run_smoke_test(self) -> None:
        """
        【診断 L2: 実行時健全性検証】起動時 1 バーでの過去バグ再発検知。

        Layer 1 Shadow Mode が rtol=1e-7 で完全検証済 (発見 #69 §D 参照) の
        ため、偽陽性の多い INFO/WARNING ログは廃止。以下の ERROR のみ出力:
            - M0.5 で HIGH_GAIN_M05_FEATURES のうち 3+ 件が同時 0
            - 全 TF で 0 値比率 > 30% (EXPECTED_ZERO_FEATURES 除外後)

        正常時は無音 (= ログに何も出ない = OK)。

        起動経路 SSoT 化 (Phase 9b 案 V):
            メソッドを public 化し、main.py 起動シーケンスの両経路 (フル
            ウォームアップ / スナップショット復帰) の合流地点で 1 回呼ぶ。
        """
        # 過去にバグで死蔵していた特徴量 (e1d_obv_rel_M0.5 = gain 7,463 等)。
        # 再発検知のため 3+ 件同時 0 を ERROR 判定する。
        HIGH_GAIN_M05_FEATURES = [
            "e1d_obv_rel",
            "e1d_force_index_norm",
            "e1d_accumulation_distribution_rel",
            "e1d_volume_ma20_rel",
            "e1d_volume_price_trend_norm",
        ]

        # 定義上 0 が正常な特徴量 (gain=0、AI 未使用)。zero_pct 集計から除外。
        EXPECTED_ZERO_FEATURES = {
            "e1d_hv_regime_50",
            "e1f_biomechanical_efficiency_10",
            "e1f_linguistic_complexity_15",
            "e1f_rhythm_pattern_12",
        }

        for tf_name in self.lookbacks_by_tf.keys():
            if not self.is_buffer_filled.get(tf_name, False):
                continue
            try:
                data = {
                    col: np.array(self.data_buffers[tf_name][col], dtype=np.float64)
                    for col in self.OHLCV_COLS
                }
                # [Phase 9d 発見 #66 Phase D-3] smoke test は QAState を update しない
                features = self._calculate_base_features(
                    data, tf_name, skip_qa_update=True
                )

                # ERROR 1: M0.5 で過去バグ被害者 (高 gain 特徴量) の 3+ 件同時 0
                if tf_name == "M0.5":
                    zero_critical = [
                        feat for feat in HIGH_GAIN_M05_FEATURES
                        if features.get(feat) == 0.0
                    ]
                    if len(zero_critical) >= 3:
                        self.logger.error(
                            f"❌ {tf_name} 死蔵バグ再発の可能性: {zero_critical}"
                        )

                # ERROR 2: 全 TF で 0 比率 > 30% (EXPECTED_ZERO_FEATURES 除外後)
                zero_count = sum(
                    1 for k, v in features.items()
                    if v == 0.0 and k not in EXPECTED_ZERO_FEATURES
                )
                total_count = len(features) - len(EXPECTED_ZERO_FEATURES)
                if total_count > 0 and zero_count / total_count > 0.30:
                    self.logger.error(
                        f"❌ {tf_name} 特徴量 0 比率異常: "
                        f"{zero_count}/{total_count} ({zero_count/total_count*100:.1f}%)"
                    )
            except Exception as e:
                self.logger.error(f"❌ {tf_name} smoke test 例外: {e}")

    def get_max_lookback_for_all_timeframes(self) -> Dict[str, int]:
        return self.lookbacks_by_tf

    def is_all_buffers_filled(self) -> bool:
        for tf_name in self.lookbacks_by_tf.keys():
            if not self.is_buffer_filled.get(tf_name, False):
                self.logger.warning(f"バッファ {tf_name} はまだ充填されていません。")
                return False
        return True

    def _buffer_to_dataframe(self, tf_name: str) -> pd.DataFrame:
        """
        指定された時間足のDequeバッファをPandas DataFrameに変換する。
        """
        df = pd.DataFrame(self.data_buffers[tf_name])

        last_ts = self.last_bar_timestamps[tf_name]
        if last_ts is None:
            raise ValueError(f"バッファ {tf_name} のタイムスタンプがありません。")

        # [FIX-INFO-2] Pandas 2.2以降の推奨エイリアスに更新 (T→min, H→h)
        freq_map = {
            "M0.5": "30s",  # [FIX] M0.5追加
            "M1": "1min",
            "M3": "3min",
            "M5": "5min",
            "M8": "8min",
            "M15": "15min",
            "M30": "30min",
            # "H1": "1h",
            # "H4": "4h",  # 未使用
            # "H6": "6h",  # [FIX] 削除済み
            # "H12": "12h", # [FIX] 削除済み
            # "D1": "1D",  # 未使用
            # "W1": "1W",  # 未使用
            # "MN": "1MS", # 未使用
        }
        freq = freq_map.get(tf_name, "1T")

        # Dequeの長さに応じてタイムスタンプインデックスを逆算
        timestamps = pd.date_range(
            end=last_ts, periods=len(self.data_buffers[tf_name]["close"]), freq=freq
        )
        df["timestamp"] = timestamps

        return df.set_index("timestamp")

    @staticmethod
    def _add_disc_column(df: pd.DataFrame, freq_seconds: int) -> pd.DataFrame:
        """
        [DISC-FLAG] DataFrame のインデックス(timestamp)から不連続フラグ列 'disc' を追加する。

        学習側 s1_1_B_build_ohlcv.py の DISC-FLAG 付与ロジックと完全一致させるため、
        本メソッドはウォームアップ (vectorized 一括充填) 用の disc 推定器として機能する
        (Train-Serve Skew Free)。リアルタイム単発バーは _compute_disc_flag を使用すること。

        判定ルール:
            disc[i] = (timestamp[i] - timestamp[i-1]).total_seconds() > freq_seconds * 1.5
            先頭バーは便宜上 False (前バーがないため連続扱い)

        [DTYPE-SAFE 修正]
        旧実装は `out.index.astype("int64")` で ns 想定の int を取り出していたが、
        pandas 2.0+ で DatetimeIndex のデフォルト dtype が datetime64[us] となり、
        μs 単位の int 値を ns 単位の threshold と比較する形になり常に disc=False
        になる構造的バグがあった。これを Timedelta 経由の秒数比較に変更し、
        pandas/numpy のバージョンによらず正しく動作するように修正。

        Args:
            df:           timestamp 昇順の DataFrame (index がタイムスタンプ)
            freq_seconds: 当該時間足の想定バー間隔 (秒)。0 のときは disc=False で固定。

        Returns:
            'disc' 列を追加した DataFrame (元 DataFrame は変更しない)。
        """
        out = df.copy()
        if freq_seconds <= 0 or len(out) == 0:
            out["disc"] = False
            return out

        # [DTYPE-SAFE] Timedelta 経由で秒単位の差分を取得 (datetime64[ns] /
        # datetime64[us] / tz-aware / tz-naive すべての環境で正しく動作)。
        # _DISC_GAP_MULTIPLIER をクラス属性経由で参照 (SSoT 統一)。
        ts_index = pd.DatetimeIndex(out.index)
        gaps_sec = np.zeros(len(ts_index), dtype=np.float64)
        if len(ts_index) > 1:
            # diff() は最初の要素を NaT として返すので、先頭は 0 にする
            # .copy() を付けるのは pandas 2.0+ で diff().to_numpy() が
            # read-only な view を返す場合があるため
            diffs = ts_index.to_series().diff().dt.total_seconds().to_numpy().copy()
            diffs[0] = 0.0
            gaps_sec = diffs

        threshold_sec = freq_seconds * RealtimeFeatureEngine._DISC_GAP_MULTIPLIER
        out["disc"] = gaps_sec > threshold_sec
        return out

    def _compute_disc_flag(self, tf_name: str, bar_timestamp) -> bool:
        """
        [DISC-FLAG SSoT] deque 末尾との時刻差から単発バーの disc を計算する。

        _add_disc_column のスカラー版 — 閾値式は完全一致 (1.5x ルール)。
        リアルタイム単発バー追加 (poll_m3_bar / resample / gap-fill / warmup_only) の
        全経路で本メソッドが唯一の disc 計算箇所となる (発見 #60 修正)。

        学習側 s1_1_B_build_ohlcv.py との同値性:
            学習側 (vectorized): disc[i] = (gap_seconds > expected_seconds * 1.5)
            本番側 (scalar):     disc    = (gap_sec     > expected_sec * 1.5)
        両者は数学的に完全一致。検証済み。

        Args:
            tf_name:       時間足名 (M0.5 / M1 / M3 / M5 / M8 / M15 等)
            bar_timestamp: 新しいバーのタイムスタンプ (pd.Timestamp)

        Returns:
            bool: True なら不連続バー (前 close を使わない TR 計算が必要)
                  先頭バー (前バーなし) または expected_sec=0 (tick足等) は False
        """
        prev_ts = self.last_bar_timestamps.get(tf_name)
        if prev_ts is None:
            return False  # 先頭バー (前バーがないので連続扱い)

        expected_sec = self._TF_FREQ_SECONDS.get(tf_name, 0)
        if expected_sec <= 0:
            return False  # tick足など、想定間隔不明の TF

        # pd.Timestamp 同士の差分は Timedelta、total_seconds() で秒に変換
        gap_sec = (bar_timestamp - prev_ts).total_seconds()
        return gap_sec > expected_sec * self._DISC_GAP_MULTIPLIER

    def _replace_buffer_from_dataframe(
        self,
        tf_name: str,
        df: pd.DataFrame,
        market_proxy_cache: Optional[pd.DataFrame] = None,
    ) -> None:
        """
        DataFrameの過去データを使ってOHLCVバッファを充填しつつ、
        全特徴量のOLS状態(純化バッファ)をウォームアップする。

        [V12.0 バグ修正: 成長スライス問題の解消]

        【旧実装の致命的バグ】
        旧実装ではウォームアップのステップ i で `arr[:i+1]` (成長スライス) を使って
        特徴量を計算し、その値でOLSを学習していた。
        しかしリアルタイム推論では常に deque の全データ (buffer_len 本の固定ウィンドウ)
        で特徴量を計算する。

        この不一致が `volume_price_trend` などの累積特徴量で破滅的な分布シフトを引き起こす:
          - OLSが学習した mean_y  ≈ 成長スライス平均 ≈ VPT_full / 2（小さい値）
          - 推論時の実際の特徴量値 ≈ VPT_full（2116本分の累積、はるかに大きい値）
          - 残差 = VPT_full - (beta * proxy + alpha) → 内部クリップ値 ±100,000 に張り付く
          - モデルが OOD 入力を受け取り → M2 が Long/Short 両方向で 1.0 を出力

        【修正内容】
        OLSウォームアップにおいて、特徴量を「成長スライス(arr[:i+1])」ではなく
        「固定スライディングウィンドウ (arr[max(0, i+1-buffer_len) : i+1])」で計算する。
        これによりウォームアップ時の特徴量分布がリアルタイム推論と完全に一致する。

        さらに、渡された df の全行（最大 OLS_WINDOW + buffer_len 行）を使うことで、
        OLS が十分な数のフルウィンドウ特徴量値で学習できるようにする。
        （旧実装は df を buffer_len 行に先頭から切り捨てていたため、累積特徴量が
          フルウィンドウに達する前の値でOLSが汚染されていた。）
        """
        if tf_name not in self.data_buffers:
            self.logger.warning(f"_replace_buffer: {tf_name} は管理対象外です。")
            return

        # [Phase 9d 発見 #63] TF 毎可変 OLS 窓に対応 (Phase 10 設計と整合)。
        # 学習側 2_G_alpha_neutralizer は M0.5=5760, M1=2880, M3=960, M5=576,
        # M8=360, M15=192 (2 日案) を使う。本番もこれと一致させる。
        OLS_WINDOW = self._get_ols_window(tf_name)
        buffer_len = self.lookbacks_by_tf[tf_name]

        # [修正] OLS を十分なフルウィンドウ特徴量値で学習するために必要な行数。
        # - buffer_len 行: 1つのフルウィンドウを形成するために必要な最小データ
        # - OLS_WINDOW 行: OLS が安定した分布を学習するために必要なサンプル数
        # 合計が用意できない場合は利用可能な全行数を使う。
        ols_total_needed = buffer_len + OLS_WINDOW
        df_for_processing = df.iloc[-min(len(df), ols_total_needed) :]

        # [旧実装との互換性確保] OHLCVバッファ充填のフォールバック用スライス
        df_slice_for_no_proxy = df.iloc[-buffer_len:]

        # 1. OHLCVバッファを一旦完全にクリア
        for col in self.OHLCV_COLS:
            self.data_buffers[tf_name][col].clear()
        # [DISC-FLAG] disc deque も同時にクリア (バグA修正の一部)
        self.data_buffers[tf_name]["disc"].clear()
        # [SESSION-RATIO] 時刻 deque も同時にクリア
        self.data_buffers[tf_name]["__bar_ts_ns__"].clear()

        # プロキシがない場合（通常はここには来ない）
        if market_proxy_cache is None or market_proxy_cache.empty:
            self.logger.warning(
                f"  -> {tf_name:<3} OLSバックフィルスキップ (プロキシ未提供)"
            )
            for col in self.OHLCV_COLS:
                self.data_buffers[tf_name][col].extend(
                    df_slice_for_no_proxy[col].values
                )
            # [DISC-FLAG / バグA修正] disc deque も OHLCV と同時に充填する。
            #   旧実装ではこの分岐 (および OLS 経由の通常分岐) で disc deque を
            #   空のまま放置していたため、calculate_barrier_atr が起動直後に
            #   全 disc=False としてパディングし、ギャップ越境TR をシードに
            #   含む異常 ATR を最大20時間出力していた。
            if "disc" in df_slice_for_no_proxy.columns:
                self.data_buffers[tf_name]["disc"].clear()
                self.data_buffers[tf_name]["disc"].extend(
                    df_slice_for_no_proxy["disc"].astype(bool).values
                )
            else:
                # disc 列が無い場合は安全側 (全 False = 連続) で初期化。
                # ただし通常はウォームアップ呼び出し側で _add_disc_column が
                # 既に呼ばれている前提のため、警告ログを残す。
                self.logger.warning(
                    f"  -> {tf_name:<3} disc 列が見つかりません。全 False で初期化します。"
                )
                self.data_buffers[tf_name]["disc"].clear()
                self.data_buffers[tf_name]["disc"].extend(
                    [False] * len(df_slice_for_no_proxy)
                )
            self.last_bar_timestamps[tf_name] = df_slice_for_no_proxy.index[-1]
            # [SESSION-RATIO] 時刻 deque を disc/OHLCV と同一行で充填 (末尾整合)
            self.data_buffers[tf_name]["__bar_ts_ns__"].clear()
            self.data_buffers[tf_name]["__bar_ts_ns__"].extend(
                np.asarray(df_slice_for_no_proxy.index.astype("int64"))
            )
            if len(df_slice_for_no_proxy) > 0:
                self.is_buffer_filled[tf_name] = True
            return

        n_rows = len(df_for_processing)
        self.logger.info(
            f"  -> {tf_name:<3} ウォームアップ開始 (固定ウィンドウ版: {n_rows}行 / "
            f"必要: {ols_total_needed}行 / buffer={buffer_len} / OLS={OLS_WINDOW})..."
        )
        if n_rows < ols_total_needed:
            self.logger.warning(
                f"  -> {tf_name:<3} 利用可能データが不足 ({n_rows} < {ols_total_needed})。"
                f" OLS精度が低下する可能性があります。"
                f" 取得する M0.5(tick→resample) バー数を増やすことを検討してください。"
            )

        # --- Numpy配列として一括抽出 ---
        arr_open = df_for_processing["open"].values.astype(np.float64)
        arr_high = df_for_processing["high"].values.astype(np.float64)
        arr_low = df_for_processing["low"].values.astype(np.float64)
        arr_close = df_for_processing["close"].values.astype(np.float64)
        arr_vol = df_for_processing["volume"].values.astype(np.float64)
        timestamps = df_for_processing.index

        base_features: dict = {}

        # (1) OHLCVバッファを先に一括充填（全行・超高速）
        for col in self.OHLCV_COLS:
            self.data_buffers[tf_name][col].extend(
                df_for_processing[col].values[-buffer_len:]
            )
        # [DISC-FLAG / バグA修正] disc deque も同時に充填する。
        #   旧実装ではここで disc deque が空のまま放置されており、
        #   calculate_barrier_atr が全 disc=False とみなしてギャップ越境TR を
        #   通常 TR として扱い、起動から最大20時間 ATR が異常値を出力していた。
        if "disc" in df_for_processing.columns:
            self.data_buffers[tf_name]["disc"].clear()
            self.data_buffers[tf_name]["disc"].extend(
                df_for_processing["disc"].astype(bool).values[-buffer_len:]
            )
        else:
            # 通常はウォームアップ呼び出し側で _add_disc_column が呼ばれて
            # disc 列が DataFrame に付与されている前提。万一無い場合は警告。
            self.logger.warning(
                f"  -> {tf_name:<3} disc 列が見つかりません。全 False で初期化します。"
            )
            self.data_buffers[tf_name]["disc"].clear()
            self.data_buffers[tf_name]["disc"].extend(
                [False] * min(len(df_for_processing), buffer_len)
            )
        self.last_bar_timestamps[tf_name] = timestamps[-1]
        # [SESSION-RATIO] 時刻 deque を disc/OHLCV と同一行(末尾buffer_len)で充填
        self.data_buffers[tf_name]["__bar_ts_ns__"].clear()
        self.data_buffers[tf_name]["__bar_ts_ns__"].extend(
            np.asarray(timestamps.astype("int64"))[-buffer_len:]
        )

        # (2) OLSウォームアップ：フルウィンドウ分のみ計算（buffer_len未満はスキップ）
        for i in range(n_rows):
            window_start = max(0, i + 1 - buffer_len)
            window_size = i + 1 - window_start

            # フルウィンドウ未満はOLSスキップ（無駄な計算を排除）
            if window_size < buffer_len:
                continue

            data = {
                "open": arr_open[window_start : i + 1],
                "high": arr_high[window_start : i + 1],
                "low": arr_low[window_start : i + 1],
                "close": arr_close[window_start : i + 1],
                "volume": arr_vol[window_start : i + 1],
            }

            # [Phase 9d 発見 #66 Phase D-3] warmup loop は QAState を成熟させる
            # ことが本来の目的だが、artifact から学習側 5 年分の成熟状態を継承
            # 済みの場合は warmup の 577 回 update で artifact 状態を破壊しない
            # よう skip_update=True を渡す。artifact 不在のフォールバック経路では
            # 旧挙動 (warmup loop で 2 半減期分まで成熟させる) を維持する。
            _skip_qa = self._any_artifact_loaded(tf_name)
            base_features = self._calculate_base_features(
                data, tf_name, skip_qa_update=_skip_qa
            )

            # (3) 固定ウィンドウ特徴量値で OLS 状態を更新
            self._update_incremental_ols(
                tf_name, base_features, market_proxy_cache, timestamps[i]
            )
        if n_rows > 0:
            self.is_buffer_filled[tf_name] = True

            # ウォームアップ終了時に最新値をキャッシュに保存する
            try:
                if base_features:
                    neutralized = self._calculate_neutralized_features(
                        base_features, tf_name, timestamps[-1], market_proxy_cache
                    )
                    self.latest_features_cache[tf_name] = neutralized
            except Exception as e:
                self.logger.warning(f"{tf_name} ウォームアップキャッシュ保存失敗: {e}")

        # proxy_feature_buffersのDequeサイズをOLS学習サンプル数として報告
        # （V5でols_stateのWelford変数は廃止済み・Dequeが実体）
        ols_n = 0
        if tf_name in self.proxy_feature_buffers:
            mp_deque = self.proxy_feature_buffers[tf_name].get("market_proxy")
            if mp_deque:
                ols_n = len(mp_deque)
        self.logger.info(
            f"  -> {tf_name:<3} ウォームアップ完了 (固定ウィンドウ版)。"
            f" OLS学習サンプル数: ~{ols_n} / {OLS_WINDOW}"
        )

    def pop_detected_m05_gaps(self) -> List[tuple]:
        """[GAP-DETECT §11.34.16-T 層3] 検知済み M0.5 欠損区間を返し内部をクリアする。

        main のリアルタイムループが毎サイクル呼び、返った各 (start_ts, end_ts) を
        bridge.request_historical_data で pinpoint 再取得 → process_new_m05_bar へ
        再投入する。再投入もこの単一入口を通るため、再取得中に別の欠落があれば再び
        検知される (入口が 1 点なので漏れない)。pop 方式でループ間の二重処理を防ぐ。
        """
        gaps = list(self._detected_m05_gaps)
        self._detected_m05_gaps.clear()
        return gaps

    def fill_all_buffers(
        self,
        history_data_map: Dict[str, pd.DataFrame],
        market_proxy_cache: pd.DataFrame,
    ) -> None:
        """
        1. M0.5データのみを history_data_map から受け取る
        2. M0.5バッファを充填
        3. M0.5データから M1・M3〜MN のすべてをリサンプリングして充填する
        """
        self.logger.info(
            "全時間足の履歴データでNumpyバッファを一括充填中 (V12.0: M0.5起点)..."
        )

        if "M0.5" not in history_data_map:
            raise ValueError(
                "履歴データに M0.5 がありません。リサンプリングできません。"
            )

        m05_history_pd = history_data_map["M0.5"]
        if "timestamp" not in m05_history_pd.columns:
            raise ValueError("M0.5履歴データに 'timestamp' カラムが見つかりません。")
        m05_history_pd = m05_history_pd.set_index("timestamp")

        # [V=0 GUARD] 学習側 s1_1_B_build_ohlcv.py の filter(tick_count > 0) と
        # 完全整合させるため、履歴 M0.5 から V=0 ghost bar を除外する。
        # EA 側 CollectM05Bar の new-bucket 分岐に volume>0 ガードが欠落していた
        # ため、ProcessHistoryRequest 経由で V=0 stub が混入していた。
        # M1 以降のリサンプル時 .dropna() は close=NaN しか拾えず、
        # close=prev_close (finite) の V=0 stub は通過してしまうため、
        # M0.5 起点でフィルタをかける必要がある。
        if "volume" in m05_history_pd.columns:
            _n_before = len(m05_history_pd)
            m05_history_pd = m05_history_pd[m05_history_pd["volume"] > 0]
            _n_after = len(m05_history_pd)
            if _n_before != _n_after:
                self.logger.info(
                    f"[V=0 GUARD] M0.5 履歴から V=0 ghost {_n_before - _n_after} 本を除外 "
                    f"(残: {_n_after} / 元: {_n_before} 本)"
                )

        self.logger.info(f"  -> M0.5 バッファをMT5データから充填中...")
        # [DISC-FLAG] M1以降と同様に _add_disc_column を適用してから充填する。
        # MT5直接取得データには disc 列が存在しないため全 False で初期化されていた。
        # freq=30秒 (M0.5=30秒足) は s1_1_B の TIMEFRAME_FREQ_SECONDS["M0.5"]=30 と完全一致。
        m05_history_pd = self._add_disc_column(m05_history_pd, freq_seconds=30)
        self._replace_buffer_from_dataframe("M0.5", m05_history_pd, market_proxy_cache)

        # M1をM0.5からリサンプリングして生成し、m05_dataframeを構築
        # [SSoT / Phase 9d 発見 #59] closed='left', label='left' を明示。
        #   学習側 s1_1_B_build_ohlcv.py L359 の Polars
        #   `group_by_dynamic("datetime", every=freq, closed="left", label="left")`
        #   と完全一致させる。pandas のデフォルトも分単位 TF では同じだが、
        #   pandas バージョン更新でデフォルトが変わった際の Train-Serve Skew を
        #   未然に防ぐための永続的な保険として明示する。
        self.logger.info(f"  -> M1  をM0.5からリサンプリング中...")
        m1_history_pd = (
            m05_history_pd.resample("1min", closed="left", label="left")
            .agg(
                {
                    "open": "first",
                    "high": "max",
                    "low": "min",
                    "close": "last",
                    "volume": "sum",
                }
            )
            .dropna()  # 学習側 filter(tick_count>0) と完全一致
        )
        # ─────────────────────────────────────────────────────────────
        # [Phase 9d 発見 #64] warmup resample 末尾の「形成中バー」を除外
        # ─────────────────────────────────────────────────────────────
        # 旧実装: dropna() のみ。M0.5 履歴の末端 (warmup_end_ts) に位置する
        #   M1 bucket は、当該 bucket 内の M0.5 が一部しか存在しなくても
        #   resample 結果に含まれてしまう (1 本だけから derived された
        #   M1 OHLCV)。これが production deque の末尾に残り、その後の
        #   M1 close 検知 (発見 #62 の timestamp ベース判定) で
        #   last_known >= 該当 timestamp となり、正しい M1 バーで上書きされない。
        #   結果として、長窓特徴量 (e.g. rolling_20) が 20 本の窓を通過する
        #   間ずっと汚染値を参照する構造的バグ。
        # 新実装: `_resample_and_update_buffer` (runtime, 発見 #62) と同じ
        #   timestamp ベース close 判定を適用して、警報の「形成中バー」を除外。
        # ─────────────────────────────────────────────────────────────
        m1_history_pd = self._filter_to_closed_buckets_warmup(
            m1_history_pd, "M1", m05_history_pd
        )
        # [DISC-FLAG] タイムスタンプ差から不連続フラグを推定
        #   学習側 s1_1_B の DISC-FLAG 付与ロジックと完全一致させる。
        m1_history_pd = self._add_disc_column(m1_history_pd, freq_seconds=60)
        self._replace_buffer_from_dataframe("M1", m1_history_pd, market_proxy_cache)

        self.m05_dataframe.clear()
        m05_records = m05_history_pd.reset_index().to_dict("records")
        self.m05_dataframe.extend(m05_records)

        # [DISC-FLAG SSoT] _freq_seconds_map のローカル定義は撤去 (発見 #60)。
        # TF 名 → 秒 のマッピングはクラス属性 self._TF_FREQ_SECONDS を使用する。

        for tf_name, rule in self.TF_RESAMPLE_RULES.items():
            if tf_name not in self.data_buffers or tf_name in ("M0.5", "M1"):
                continue

            try:
                self.logger.info(f"  -> {tf_name:<3} をM0.5からリサンプリング中...")
                # [SSoT / Phase 9d 発見 #59] closed='left', label='left' を明示
                #   (学習側 s1_1_B と完全一致、pandas デフォルト依存を排除)。
                resampled_df = (
                    m05_history_pd.resample(rule, closed="left", label="left")
                    .agg(
                        {
                            "open": "first",
                            "high": "max",
                            "low": "min",
                            "close": "last",
                            "volume": "sum",
                        }
                    )
                    .dropna()  # 学習側 filter(tick_count>0) と完全一致
                )

                if resampled_df.empty:
                    self.logger.warning(f"{tf_name} のリサンプリング結果が空です。")
                    continue

                # [Phase 9d 発見 #64] warmup resample 末尾の「形成中バー」を除外
                # (M1 の同等処理と同じ理由。詳細は M1 側コメント参照)
                resampled_df = self._filter_to_closed_buckets_warmup(
                    resampled_df, tf_name, m05_history_pd
                )

                if resampled_df.empty:
                    self.logger.warning(
                        f"{tf_name} のリサンプリング結果が空 (incomplete bucket filter 後)。"
                    )
                    continue

                # [DISC-FLAG SSoT] タイムスタンプ差から不連続フラグを推定
                # クラス属性 _TF_FREQ_SECONDS から TF 名で直接秒数を取得
                expected_sec = self._TF_FREQ_SECONDS.get(tf_name, 0)
                resampled_df = self._add_disc_column(
                    resampled_df, freq_seconds=expected_sec
                )

                self._replace_buffer_from_dataframe(
                    tf_name, resampled_df, market_proxy_cache
                )
            except Exception as e:
                self.logger.error(f"{tf_name} のリサンプリング充填に失敗: {e}")

        self.logger.info("✓ 全バッファの初期充填が完了しました。")

        # [Phase 9b 案 V] smoke test (診断 L2) はここでは呼ばない。
        # main.py の起動シーケンス側 (フルウォームアップ / スナップショット復帰の
        # 両経路の合流地点) で `engine.run_smoke_test()` を明示呼び出しすることで、
        # 起動経路によらず必ず 1 回実行される設計に変更。

    def _append_bar_to_buffer(
        self,
        tf_name: str,
        bar_df: pd.DataFrame,
        market_proxy_cache: pd.DataFrame,
    ) -> bool:
        """
        バッファに新しいバー (DataFrame形式) を追加し、
        純化(OLS)状態を逐次更新する。

        Returns:
            True: バーが正常に追加された
            False: 同一タイムスタンプの重複のためスキップした
        """
        if tf_name not in self.data_buffers:
            return False

        try:
            bar_dict = bar_df.iloc[0].to_dict()
            bar_timestamp = bar_df.index[0]

            # [DEDUP] 同一タイムスタンプの二重追加を防止。
            # gap-fill(warmup_only)がバーをバッファに追加後、正規のpoll_m3_barパスが
            # 同じバーを再追加しようとする場合（またはその逆）に発生する。
            # last_bar_timestamps[tf_name] == bar_timestamp なら既に追加済み → スキップ。
            if self.last_bar_timestamps.get(tf_name) == bar_timestamp:
                return False

            # 1. OHLCVバッファを更新
            for col in self.OHLCV_COLS:
                self.data_buffers[tf_name][col].append(bar_dict[col])
            # [DISC-FLAG SSoT] disc は deque 末尾との差分から動的計算する。
            # 旧実装の bar_dict.get("disc", False) フォールバックは廃止。
            # - 通常 poll_m3_bar 経路で M0.5 disc が常に False になっていた
            #   構造的欠陥 (発見 #60、Phase 5 のバグA本質的修正の短時間ギャップ版) を解消
            # - gap-fill / resample 経路で事前計算された disc は無視 (再計算で同値)
            # - 学習側 s1_1_B の disc 判定式と完全一致 (1.5x ルール)
            disc_flag = self._compute_disc_flag(tf_name, bar_timestamp)
            self.data_buffers[tf_name]["disc"].append(disc_flag)
            # [SESSION-RATIO] UTC ns 時刻を並行 deque に記録 (atr_arr と要素対応を維持)
            self.data_buffers[tf_name]["__bar_ts_ns__"].append(
                int(pd.Timestamp(bar_timestamp).value)
            )
            self.last_bar_timestamps[tf_name] = bar_timestamp

            # ★ 2, 3 の古い限定的OLS更新処理を削除 (process_new_m05_bar内で全特徴量を一括更新するため)

            # 4. 充填状態を更新
            if not self.is_buffer_filled[tf_name]:
                self.is_buffer_filled[tf_name] = True
                self.logger.info(f"✅ {tf_name} バッファ計算開始 (Best-Effort)。")

            return True

        except KeyError as e:
            self.logger.error(f"バーデータ {tf_name} にキーがありません: {e}")
            return False
        except Exception as e:
            self.logger.error(f"バー {tf_name} の追加に失敗: {e}")
            return False

    def _resample_and_update_buffer(
        self, tf_name: str, rule: str, market_proxy_cache: pd.DataFrame
    ) -> List[pd.Timestamp]:
        """
        M0.5 DequeをDFに変換してリサンプリングし、新しいバーが生成されていたら
        対象のバッファに追加し、新バーのタイムスタンプを返す。
        """
        try:
            last_known_timestamp = self.last_bar_timestamps.get(tf_name)
            if last_known_timestamp is None:
                self.logger.warning(
                    f"{tf_name} の最終時刻が不明です。リサンプリングをスキップします。"
                )
                return []

            # 1. Dequeから必要なデータ「だけ」を抽出 (メモリコピー地獄を回避)
            # [FIX-WARNING-5] off-by-one 修正: last_known_timestamp 以降のバーのみ収集し
            # リサンプリングのオーバーラップ用に1本前のバーを追加する
            new_m05_bars_for_resampling = []
            found_anchor = False
            for bar in reversed(self.m05_dataframe):
                bar_ts = bar["timestamp"]
                if bar_ts >= last_known_timestamp:
                    new_m05_bars_for_resampling.append(bar)
                else:
                    # 1本前のアンカーバーを追加してリサンプリングの境界を正確にする
                    if not found_anchor:
                        new_m05_bars_for_resampling.append(bar)
                        found_anchor = True
                    break

            new_m05_bars_for_resampling.reverse()

            if len(new_m05_bars_for_resampling) < 2:
                return []

            new_m05_data = pd.DataFrame(new_m05_bars_for_resampling).set_index(
                "timestamp"
            )

            # 2. 抽出したDFのみをリサンプリング
            # [SSoT / Phase 9d 発見 #59] closed='left', label='left' を明示
            #   (学習側 s1_1_B と完全一致、pandas デフォルト依存を排除)。
            resampled_df = (
                new_m05_data.resample(rule, closed="left", label="left")
                .agg(
                    {
                        "open": "first",
                        "high": "max",
                        "low": "min",
                        "close": "last",
                        "volume": "sum",
                    }
                )
            )
            # ─────────────────────────────────────────────────────────────
            # [3者完全統一] 学習側 s1_1_B_build_ohlcv.py の
            #   `.filter(pl.col("tick_count") > 0)` と完全に同じ挙動に揃える。
            #
            # 旧実装 (ffill + disc=isna) は学習側に存在しない「ffillで埋めたバー」を
            # バッファに混入させ、rolling系特徴量の時間幅が学習と本番でズレる
            # 構造的な Train-Serve Skew 源だった。
            #
            # 新実装:
            #   1. resample 結果が NaN の行 (= ティックなしバー) はバッファに追加しない
            #      → 学習側の filter(tick_count>0) と完全等価
            #   2. 残った行に対し、タイムスタンプ差から disc を後付け推定
            #      (前バーとの間隔 > 想定間隔×1.5 なら不連続)
            #      → s1_1_B の DISC-FLAG 付与ロジックと完全等価
            # ─────────────────────────────────────────────────────────────
            # 1. NaN 行を完全に除外 (学習側 filter(tick_count>0) 相当)
            resampled_df = resampled_df.dropna(subset=["close"])

            if len(resampled_df) == 0:
                return []

            # [DISC-FLAG SSoT] resample 後の disc 列付与は撤去 (発見 #60)。
            # 旧実装ではここでタイムスタンプ差から disc を推定して bar_df["disc"]
            # に詰めていたが、_append_bar_to_buffer 内の _compute_disc_flag が
            # deque 末尾との差分から再計算するため重複する。
            # 計算結果は同値 (両者とも 1.5x ルールで学習側 s1_1_B と一致) のため
            # 撤去で副作用ゼロ。disc 計算箇所を 1 箇所に集約することで保守性向上。

            # ─────────────────────────────────────────────────────────────
            # [Phase 9d 発見 #62] iloc[:-1] による 3 分遅延を解消
            # ─────────────────────────────────────────────────────────────
            # 旧実装: `newly_closed_bars = resampled_df.iloc[:-1]`
            #   → 「resample 結果の最後のバーは形成中」という保守的判定。
            #   → 案 X (発見 #61, EA から M3 close 時に直近 6 本の M0.5 を送信) を
            #     導入しても、最後のバー (12:02:30) を含む M3 [12:00, 12:03) bucket
            #     は label=12:00 で resample の最後に来るため、iloc[:-1] で除外され、
            #     M3 close 検知は次の M3 通知 (12:06 の 12:05:30 バー到着時) まで
            #     遅延する。結果: 構造的な 3 分遅延 (シミュレーションで確認済)。
            #
            # 新実装: タイムスタンプベースの bucket close 判定。
            #   bucket_close_ts = m05_dataframe[-1].timestamp + M0.5_freq_sec (= 30s)
            #     - 最新 M0.5 バー [t, t+30s) の END 時刻 (= 次のバー想定開始時刻)
            #   bucket [label, label + tf_freq_sec) は次を満たすとき closed:
            #     label + tf_freq_sec <= bucket_close_ts
            #   = 「次バーが当該 bucket に追加される余地がない」
            #
            # 例: bar 12:02:30 着, M3 検知:
            #   bucket_close_ts = 12:02:30 + 30s = 12:03:00
            #   M3 [12:00, 12:03), label=12:00, tf_freq=180s
            #   12:00 + 180s = 12:03 <= 12:03:00? → YES → closed → 即時 close 検知
            #
            # 案 X (EA 6 本送信) と組み合わせて、M3 close 時に学習側と数学的に
            # 完全等価な OHLCV 集約 + 即時シグナル発火を達成する。
            # ─────────────────────────────────────────────────────────────
            if not self.m05_dataframe:
                return []
            m05_latest_ts = self.m05_dataframe[-1]["timestamp"]
            m05_freq_sec = self._TF_FREQ_SECONDS.get("M0.5", 30)
            bucket_close_ts = m05_latest_ts + pd.Timedelta(seconds=m05_freq_sec)

            tf_freq_sec = self._TF_FREQ_SECONDS.get(tf_name, 0)
            if tf_freq_sec <= 0:
                # tick足等、想定間隔不明 → 旧来の保守的挙動 (最後を除外)
                newly_closed_bars = resampled_df.iloc[:-1]
            else:
                # bucket [label, label + tf_freq_sec) が closed なのは
                # label + tf_freq_sec <= bucket_close_ts のとき
                tf_freq_td = pd.Timedelta(seconds=tf_freq_sec)
                closed_mask = (resampled_df.index + tf_freq_td) <= bucket_close_ts
                newly_closed_bars = resampled_df[closed_mask]

            new_bars = newly_closed_bars[newly_closed_bars.index > last_known_timestamp]

            if new_bars.empty:
                return []

            new_bar_timestamps = []

            # 4. 新しいバーをバッファに追加
            # _append_bar_to_buffer がTrueを返した（実際に追加された）場合のみカウント。
            # Falseは重複排除によるスキップ（gap-fillで既に追加済みのバー）。
            for timestamp, row in new_bars.iterrows():
                bar_df = pd.DataFrame(row).T
                bar_df.index = [timestamp]
                bar_df.index.name = "timestamp"

                added = self._append_bar_to_buffer(tf_name, bar_df, market_proxy_cache)
                if added:
                    new_bar_timestamps.append(timestamp)

            if new_bar_timestamps:
                self.logger.debug(
                    f"  -> {tf_name} バッファに {len(new_bars)} 件の確定バーを追加しました。"
                )
            return new_bar_timestamps

        except Exception as e:
            self.logger.error(
                f"{tf_name} のリサンプリング更新に失敗: {e}", exc_info=True
            )
            return []

    def process_new_m05_bar(
        self,
        m05_bar: Dict[str, Any],
        market_proxy_cache: pd.DataFrame,
        warmup_only: bool = False,
    ) -> List[Signal]:
        """
        [メインループ] main.py から M0.5 バーを受け取り、全バッファを更新し、
        M3確定時のみシグナルをチェックして返す。

        Args:
            warmup_only: Trueのとき、バッファ更新・OLS更新・特徴量キャッシュ更新のみ行い、
                         シグナルチェックと Signal 生成を完全にスキップする。
                         差分追いつきループ（スナップショット復帰時）で使用し、
                         追いつき期間中の意図しない発注を根本から防ぐ。
        """
        signal_list: List[Signal] = []

        try:
            m05_timestamp = m05_bar["timestamp"]

            # [V=0 GUARD] 学習側 s1_1_B_build_ohlcv.py の filter(tick_count > 0) と
            # 物理的に同じ挙動を本番側でも確立させる fail-safe ガード。
            # EA 側 CollectM05Bar の new-bucket 分岐に volume>0 ガードが欠落していた
            # ため、Phase 9 #54 で導入された OnTimer 強制確定の V=0 stub が
            # silent → tick 復帰の境界で g_m05_bars に漏出し、Python の M0.5 buffer
            # に流入することで M3 close を「silent 開始時点の prev_close」で固定
            # → TP_REVERSED_BY_LAG / Execution Failed の連鎖を引き起こしていた。
            # EA 側修正後でも本ガードを残すことで、将来の EA 側退行に対する二重防御。
            _m05_volume = m05_bar.get("volume", 0)
            if _m05_volume is None or _m05_volume <= 0:
                self.logger.warning(
                    f"[V=0 GUARD] V=0 ghost bar を破棄: "
                    f"ts={m05_timestamp} OHLC={m05_bar.get('close')} V={_m05_volume}"
                )
                return signal_list

            # 1. M0.5バッファに新しいバーを追加
            # m05_dataframe はリサンプリング起点として使用
            # [DEDUP-M05DF 単調増加ガード / 支流(保険)] 末尾1本比較 (旧 == 実装) では
            # warmup/live 境界の「バッチ再投入」(実機で 527 本オーバーラップ確認、
            # 重複 1259 行・ユニーク重複時刻 577・出現位置間隔の主要値 527) を防げなかった。
            # 直前 1 本としか比較しないため、527 本前の重複を擦り抜けて同一時刻 2 値が残り、
            # M5 リサンプル close が分裂 → proxy(X) partial 混入 → OLS cov 崩壊を招いた。
            # 新バー時刻が既存最新時刻「以下」なら過去/同一の再投入として拒否する
            # (time series 単調増加原則)。実機の重複は全て既処理範囲の再投入で
            # 新規時刻を含まないため <= は正当な新規バーを取りこぼさない。
            # 復帰系 (clear→extend / load_state) は append を通らずこのガードの影響を
            # 受けず、復帰後の live バーのみ昇順で通過する。
            # 本流の根治は main.py の live ループ (g_last_processed_bar_time 前進) 側で、
            # 本ガードは万一の二重投入再発に対する多層防御 (保険)。
            if (
                len(self.m05_dataframe) > 0
                and m05_timestamp <= self.m05_dataframe[-1]["timestamp"]
            ):
                self.logger.debug(
                    f"[DEDUP-M05DF] 非単調(<=last)の再投入をスキップ: "
                    f"ts={m05_timestamp} last={self.m05_dataframe[-1]['timestamp']}"
                )
            else:
                # [GAP-DETECT §11.34.16-T 層3] append 直前に末尾との連続性を検査。
                # M0.5 は 30 秒刻み。末尾→新バーの差が 30s を超えたら間のバーが欠落。
                # (層1=EA flush・層2=bridge 完全性照合 を抜けた取りこぼしの最終 backstop。
                #  実機で 02:38:30 を 1 本取りこぼし spectral_512 が窓ズレした事象に対応)
                # warmup (fill_all_buffers の clear→extend) はこの append を通らないため
                # 誤検知しない。live/gap-fill の昇順バーのみ通過する。
                if len(self.m05_dataframe) > 0:
                    _last_ts = self.m05_dataframe[-1]["timestamp"]
                    _delta_sec = (m05_timestamp - _last_ts).total_seconds()
                    if _delta_sec > self._M05_FREQ_SEC:
                        _gap_start = _last_ts + pd.Timedelta(seconds=self._M05_FREQ_SEC)
                        _gap_end = m05_timestamp - pd.Timedelta(seconds=self._M05_FREQ_SEC)
                        _n_missing = int(_delta_sec / self._M05_FREQ_SEC) - 1
                        # [GAP-FIX 要修正②] 穴サイズ上限。週末/休場の正規ギャップ
                        # (数千本) は S1 にも足が無く埋め不要なので記録しない。本物の
                        # 落ち (層1・2 後はせいぜい 1〜2 本) だけ main に再取得させる。
                        if _n_missing <= self._GAP_DETECT_MAX_BARS:
                            self._detected_m05_gaps.append((_gap_start, _gap_end))
                            self.logger.warning(
                                f"[GAP-DETECT] M0.5 連続性違反: {_n_missing} 本欠落 "
                                f"({_gap_start} 〜 {_gap_end})。末尾={_last_ts} 新バー={m05_timestamp}。"
                                f"main に pinpoint 再取得を要求。"
                            )
                        else:
                            self.logger.info(
                                f"[GAP-DETECT] {_n_missing} 本の大欠落は上限"
                                f"({self._GAP_DETECT_MAX_BARS})超 = 正規市場ギャップ"
                                f"(週末/休場)とみなし記録せず。"
                            )
                self.m05_dataframe.append(m05_bar)
            m05_bar_df = pd.DataFrame([m05_bar]).set_index("timestamp")
            self._append_bar_to_buffer("M0.5", m05_bar_df, market_proxy_cache)

            # ▼▼▼▼▼▼ 【修正】Tick特徴量のOLS純化とキャッシュ登録 ▼▼▼▼▼▼
            # [V5] Tick特徴量は110件リストに存在しないためコメントアウトして処理をスキップ
            """
            if "tick" not in self.latest_features_cache:
                self.latest_features_cache["tick"] = {}

            raw_tick_vol = m1_bar.get("tick_volume_mean_5", 0.0)
            tick_features = {"e1a_fast_volume_mean_5": raw_tick_vol}

            # TickのOLS状態を更新
            self._update_incremental_ols(
                "tick", tick_features, market_proxy_cache, m1_timestamp
            )

            # 純化(残差計算)してキャッシュに保存
            neutralized_tick = self._calculate_neutralized_features(
                tick_features, "tick", m1_timestamp, market_proxy_cache
            )
            self.latest_features_cache["tick"].update(neutralized_tick)
            """
            # ▲▲▲▲▲▲ 【修正ここまで】 ▲▲▲▲▲▲

            # 2. M1以外の全時間足バッファをリサンプリング更新
            # [DEDUP対応] warmup_only=True でも全TFリサンプリングを実行し、
            # M3/M5/M8/M15 バッファと OLS 状態を正しく更新する。
            # 重複追加の防止は _append_bar_to_buffer の DEDUP チェックで担保する。
            # （旧実装: warmup_only=True でリサンプリングをスキップしていたが
            #   M3/M5 OLS が gap-fill 期間分だけ欠落する問題があった）
            newly_closed_timeframes: Dict[str, List[pd.Timestamp]] = {}
            for tf_name, rule in self.TF_RESAMPLE_RULES.items():
                if tf_name not in self.data_buffers:
                    continue

                new_timestamps = self._resample_and_update_buffer(
                    tf_name, rule, market_proxy_cache
                )
                if new_timestamps:
                    newly_closed_timeframes[tf_name] = new_timestamps

            newly_closed_timeframes["M0.5"] = [m05_timestamp]

            # ─────────────────────────────────────────────────────────────
            # [Phase 9d 発見 #66 Phase B] 学習側 per-TF-bar cadence への揃え
            # (第二段階: M0.5/M1/M5/M8 を学習側と一致させる)
            # ─────────────────────────────────────────────────────────────
            # 旧実装 (Phase A まで):
            #   L1703 の "if 'M3' not in newly_closed_timeframes: return" で
            #   M3 close 以外の経路を早期 return していた。これにより M0.5/M1 の
            #   update_and_clip は M3 cadence (3分毎=340回/17h) でしか呼ばれず、
            #   学習側 M0.5 (2040回/17h) / M1 (1020回/17h) と大幅乖離。
            #   M5/M8 も同様に M3 と LCM 一致時のみ recalc で過少 (68/43)。
            #
            # 新実装 (Phase B):
            #   早期 return を削除し、recalc は「close した TF 全て」に対し
            #   実行する。これで各 TF の update_and_clip 呼び出し頻度が
            #   学習側 per-TF-bar cadence と完全一致:
            #     M0.5: 毎 M0.5 close = 30秒毎 = 2040回/17h ← 一致
            #     M1:   毎 M1 close   = 1分毎  = 1020回/17h ← 一致
            #     M3:   毎 M3 close   = 3分毎  = 340回/17h  ← 一致
            #     M5:   毎 M5 close   = 5分毎  = 204回/17h  ← 一致
            #     M8:   毎 M8 close   = 8分毎  = 127回/17h  ← 一致
            #     M15:  毎 M15 close  = 15分毎 = 68回/17h   ← 一致
            #   シグナル生成は M3 close 時のみに限定 (既存設計を維持)。
            #
            # _recalc_one_tf 引数変更:
            #   旧: closure 経由で m3_timestamp を参照していた (全 TF で M3 ts)
            #   新: 各 TF の close timestamp を明示的に引数として渡す。
            #       これにより _update_incremental_ols と
            #       _calculate_neutralized_features が「その TF の close 時刻」を
            #       受け取り、学習側 per-TF row の timestamp と一致する。
            #
            # TF の close timestamp 計算規約 (Phase B で統一):
            #   M0.5: newly_closed_timeframes["M0.5"][-1] = m05_timestamp
            #         (引数として渡されるのは既に close 時刻)
            #   他 TF (M1/M3/M5/M8/M15):
            #         newly_closed_timeframes[tf][-1] (= resample 結果の bar
            #         開始時刻) + ALL_TIMEFRAMES[tf] 分 = close 時刻
            # ─────────────────────────────────────────────────────────────

            def _close_ts_for(tf_name: str) -> pd.Timestamp:
                """各 TF の close timestamp を統一的に取得"""
                if tf_name == "M0.5":
                    # L1699 で既に close 時刻が格納されている
                    return newly_closed_timeframes["M0.5"][-1]
                # 他 TF: resample 結果 (bar 開始時刻) + TF duration
                tf_minutes = self.ALL_TIMEFRAMES[tf_name]
                return (
                    newly_closed_timeframes[tf_name][-1]
                    + pd.Timedelta(minutes=tf_minutes)
                )

            # [LAG-FIX-3] 全時間足のバッファから強制再計算 (並列実行)
            # 各 TF の処理は独立なので thread safety 問題なし。Polars の
            # rayon/Numba njit は GIL を解放するため CPython でも本物の並列実行。
            def _recalc_one_tf(tf_name: str, close_ts: pd.Timestamp):
                if not self.is_buffer_filled.get(tf_name, False):
                    return None
                try:
                    data = {
                        col: np.array(self.data_buffers[tf_name][col], dtype=np.float64)
                        for col in self.OHLCV_COLS
                    }
                    # ─────────────────────────────────────────────────────
                    # [Phase 9d 発見 #66 Phase D-3] QAState artifact 由来の
                    # 状態を warmup 中に破壊しないよう skip_qa_update を判定。
                    # ─────────────────────────────────────────────────────
                    # warmup_only=True かつ 該当 TF の少なくとも 1 つの QAState が
                    # artifact 由来 (= _artifact_loaded == True) なら、本 recalc
                    # 内の update_and_clip 呼び出しで EWM 状態の追加 update を
                    # 抑止する (clip 自体は適用される)。
                    # これにより learning-side の 5 年分 EWM 成熟状態が warmup の
                    # 30 日分追加 update で破壊されることを防ぎ、Layer 1 で
                    # 数値完全一致 (1e-15) を保証する。
                    skip_qa_update = (
                        warmup_only and self._any_artifact_loaded(tf_name)
                    )

                    base_features = self._calculate_base_features(
                        data, tf_name, skip_qa_update=skip_qa_update
                    )

                    self._update_incremental_ols(
                        tf_name, base_features, market_proxy_cache, close_ts
                    )

                    neutralized = self._calculate_neutralized_features(
                        base_features, tf_name, close_ts, market_proxy_cache
                    )
                    self.latest_features_cache[tf_name] = neutralized

                    return tf_name
                except Exception as e:
                    self.logger.warning(f"{tf_name} 特徴量キャッシュ更新失敗: {e}")
                    return None

            # close した TF を全て recalc (学習側 per-TF-bar cadence と一致)
            tf_names = [
                tf for tf in self.ALL_TIMEFRAMES.keys()
                if tf in newly_closed_timeframes
            ]
            if tf_names:
                futures = [
                    self._tf_executor.submit(_recalc_one_tf, tf, _close_ts_for(tf))
                    for tf in tf_names
                ]
                for future in futures:
                    future.result()

            # ─────────────────────────────────────────────────────────────
            # シグナル生成 — M3 close 時のみ実行 (Phase B で recalc と分離)
            # ─────────────────────────────────────────────────────────────
            # Phase A 以前は L1703 の早期 return が recalc と signal の両方を
            # M3 close 時のみに制限していたが、Phase B で recalc は全 TF close
            # 時に実行されるようになり、signal だけが M3 close 限定で残る。
            # これは設計上正しい: AI 推論 (V5_check) は M3 cadence で行う前提。
            if "M3" not in newly_closed_timeframes:
                return signal_list

            m3_timestamp = newly_closed_timeframes["M3"][-1] + pd.Timedelta(minutes=3)

            # [STALE-GUARD] warmup_only=True（差分追いつき中）はシグナル生成を根本からスキップ
            if warmup_only:
                return signal_list

            V5_check_result = self._check_for_signal("M3", m3_timestamp)

            if V5_check_result["is_V5"]:
                feature_vector = self.calculate_feature_vector(
                    "M3", m3_timestamp, market_proxy_cache
                )

                if feature_vector is not None:
                    combined_features = dict(zip(self.feature_list, feature_vector[0]))

                    signal = Signal(
                        features=feature_vector,
                        timestamp=m3_timestamp,
                        timeframe="M3",
                        market_info=V5_check_result["market_info"],
                        atr_value=V5_check_result["market_info"].get("atr_value", 0.0),
                        close_price=V5_check_result["market_info"].get(
                            "current_price", 0.0
                        ),
                        feature_dict=combined_features,
                    )
                    signal_list.append(signal)

            return signal_list

        except Exception as e:
            self.logger.error(f"process_new_m05_bar でエラー: {e}", exc_info=True)
            return []

    def _check_for_signal(self, tf_name: str, timestamp: datetime) -> Dict[str, Any]:
        """
        指定された時間足のバッファがV5レジーム (ATR比率条件) かを判定する。
        """
        # [設計根拠] create_proxy_labels の TARGET_TIMEFRAMES = ["M3"] に準拠
        # Optunaの結論: M3単体・ATR ratio 0.8・TD 30min
        ALLOWED_TIMEFRAMES = ["M3"]
        if tf_name not in ALLOWED_TIMEFRAMES:
            return {"is_V5": False, "reason": "timeframe_not_allowed"}
        if tf_name not in self.data_buffers:
            return {"is_V5": False, "reason": "timeframe_not_managed"}

        try:
            data = {
                "high":  np.array(self.data_buffers[tf_name]["high"],  dtype=np.float64),
                "low":   np.array(self.data_buffers[tf_name]["low"],   dtype=np.float64),
                "close": np.array(self.data_buffers[tf_name]["close"], dtype=np.float64),
            }

            # [Phase 7 disc乖離修正] ATR Ratio 用 ATR を disc-aware 版で統一。
            #
            # 旧実装の問題:
            #   calculate_atr_wilder() は disc フラグを参照せず、週末ギャップ越境 TR
            #   （金曜 close → 月曜 first bar の大幅ジャンプ）をそのまま TR に含む。
            #   → 本番 ATR が学習側より大きくスパイク
            #   → スパイクが 480 本ローリング分母 (baseline) に混入
            #   → 月曜 24 時間、本番 ATR Ratio が学習側より低めに計算される
            #   → 月曜に本来通過するはずのシグナルが本番で弾かれる (週次 Train-Serve Skew)
            #
            # 修正:
            #   calculate_atr_wilder_disc_aware() を使用。
            #   学習側 create_proxy_labels の TR 計算式
            #     pl.when(disc).then(H-L).otherwise(max(H-L, |H-prev_close|, |L-prev_close|))
            #     .ewm_mean(alpha=1/period, adjust=False)
            #   と完全一致する。seed=TR[0]、返却型 np.ndarray で baseline 計算にも使用可能。
            #
            # 注意: SL/TP バリア幅の計算は引き続き calculate_barrier_atr() を使用（責務分離）。
            high, low, close = data["high"], data["low"], data["close"]
            disc_arr = np.array(self.data_buffers[tf_name]["disc"], dtype=np.bool_)
            if len(close) > 1:
                # [Phase 7 修正] disc-aware Wilder EWM ATR（学習側と完全一致）
                atr_arr = calculate_atr_wilder_disc_aware(
                    high.astype(np.float64),
                    low.astype(np.float64),
                    close.astype(np.float64),
                    disc_arr,
                    self.ATR_CALC_PERIOD,
                )
                atr_value = float(atr_arr[-1]) if len(atr_arr) > 0 and np.isfinite(atr_arr[-1]) else 0.0
            else:
                atr_value = 0.0
                atr_arr = np.array([])

            current_price = data["close"][-1]
            if np.isnan(atr_value):
                return {"is_V5": False, "reason": "atr_is_nan"}

            # ATR Ratioフィルター (risk_config.json で管理)
            # ATR Ratio = 現在のATR(EWM) / 過去ATR_BASELINE_DAYS日のATR(EWM)の平均
            # 学習側: atr_ratio = ATR / ATR.rolling_mean(baseline_period)
            atr_threshold = self.risk_config.get("min_atr_threshold", 0.8)  # Ratio閾値
            baseline_period = (
                TIMEFRAME_BARS_PER_DAY.get(tf_name, 1440) * ATR_BASELINE_DAYS
            )
            # ATR配列全体からbaselineを計算（ATRのrolling mean = 学習側と一致）
            if len(atr_arr) >= baseline_period:
                baseline_atr = float(np.mean(atr_arr[-baseline_period:]))
            elif len(atr_arr) > 0:
                baseline_atr = float(np.mean(atr_arr))
            else:
                baseline_atr = atr_value
            atr_ratio = atr_value / (baseline_atr + 1e-10)

            # [SESSION-RATIO] session_atr_ratio を学習側とビット一致で計算。
            #   時刻 deque (__bar_ts_ns__) は atr_arr と要素対応 (同一 fill 経路で維持)。
            #   安全網: 長さ不一致は buffer 同期バグの証拠。silent skew を防ぐため ERROR ログ。
            ts_ns_arr = np.asarray(
                self.data_buffers[tf_name]["__bar_ts_ns__"], dtype=np.int64
            )
            if ts_ns_arr.size != atr_arr.size:
                self.logger.error(
                    f"[SESSION-RATIO DESYNC] {tf_name}: ts_ns({ts_ns_arr.size}) != "
                    f"atr({atr_arr.size})。buffer 同期バグ。末尾整合で暫定計算する。"
                )
                m = min(ts_ns_arr.size, atr_arr.size)
                session_atr_ratio = (
                    compute_session_atr_ratio_last(
                        atr_arr[-m:], ts_ns_arr[-m:], SESSION_BASELINE_DAYS
                    )
                    if m > 0
                    else atr_ratio
                )
            else:
                session_atr_ratio = compute_session_atr_ratio_last(
                    atr_arr, ts_ns_arr, SESSION_BASELINE_DAYS
                )

            # atr_ratio / session_atr_ratio を latest_features_cache に書き込む
            # → calculate_feature_vector が atr_ratio_M3 / session_atr_ratio_M3 を
            #   処理する際に latest_features_cache[tf_name].get(...) で取得できる
            # 学習側（S6）と同じ計算式・ベースライン期間なので純化不要
            if tf_name in self.latest_features_cache:
                self.latest_features_cache[tf_name]["atr_ratio"] = atr_ratio
                self.latest_features_cache[tf_name][
                    "session_atr_ratio"
                ] = session_atr_ratio

            if atr_ratio >= atr_threshold:
                # [DEBUG] バッファ診断情報を計算
                atr_buffer_len = len(atr_arr)
                if len(high) >= 2 and len(low) >= 2 and len(close) >= 2:
                    last_tr = float(max(
                        high[-1] - low[-1],
                        abs(high[-1] - close[-2]),
                        abs(low[-1] - close[-2]),
                    ))
                else:
                    last_tr = float(high[-1] - low[-1]) if len(high) >= 1 else 0.0

                # [フェーズ3] SL/TP計算専用ATR（calculate_barrier_atr）に切り替え
                # AIモデル入力特徴量用のcalculate_atr_wilder()は一切触らない。
                # discフラグを渡すことでギャップ越境TRを防止し、SMAシードで安定化する。
                disc_raw = np.array(
                    self.data_buffers[tf_name]["disc"], dtype=np.bool_
                )
                # disc_arrはcloseと同じ長さに揃える。
                # 初期充填時はdiscが書き込まれていないバーがあるためFalseでパディング。
                n_close = len(close)
                n_disc = len(disc_raw)
                if n_disc >= n_close:
                    disc_arr = disc_raw[-n_close:]
                else:
                    # 先頭をFalse（連続）でパディング
                    disc_arr = np.concatenate([
                        np.zeros(n_close - n_disc, dtype=np.bool_),
                        disc_raw
                    ])
                barrier_atr = calculate_barrier_atr(
                    high.astype(np.float64),
                    low.astype(np.float64),
                    close.astype(np.float64),
                    disc_arr,
                    self.ATR_CALC_PERIOD,
                )
                # barrier_atrがNaN（バッファ不足）の場合はフォールバックとしてatr_valueを使用
                barrier_atr_value = (
                    float(barrier_atr) if np.isfinite(barrier_atr) else atr_value
                )

                market_info = {
                    "atr_value": barrier_atr_value,  # SL/TP計算用（堅牢版）
                    "atr_value_raw": atr_value,       # 参考値（学習側と同一のWilder EWM）
                    "atr_ratio": atr_ratio,
                    "current_price": current_price,
                    "sl_multiplier_long": self.risk_config.get(
                        "sl_multiplier_long", 5.0
                    ),
                    "pt_multiplier_long": self.risk_config.get(
                        "pt_multiplier_long", 1.0
                    ),
                    "sl_multiplier_short": self.risk_config.get(
                        "sl_multiplier_short", 5.0
                    ),
                    "pt_multiplier_short": self.risk_config.get(
                        "pt_multiplier_short", 1.0
                    ),
                    "direction": None,
                    # [DEBUG] 原因特定用診断情報
                    "atr_buffer_len": atr_buffer_len,
                    "last_tr": last_tr,
                }

                self.logger.info(
                    f"🏄  V5 Signal Check ({tf_name} @ {timestamp.strftime('%H:%M')}): "
                    f"PASSED (ATR Ratio: {atr_ratio:.3f} >= {atr_threshold:.3f})"
                )
                return {"is_V5": True, "market_info": market_info}
            else:
                # ▼▼▼ 追加: ATR不足で見送った時のログ ▼▼▼
                self.logger.info(
                    f"🏄  V5 Signal Check ({tf_name} @ {timestamp.strftime('%H:%M')}): "
                    f"FAILED ⛔ (ATR Ratio: {atr_ratio:.3f} < {atr_threshold:.3f})"
                )
                # ▲▲▲ ここまで追加 ▲▲▲
                return {"is_V5": False, "reason": "below_min_atr_threshold"}

        except Exception as e:
            self.logger.warning(f"_check_for_signal ({tf_name}) でエラー: {e}")
            return {"is_V5": False, "reason": "atr_calculation_error"}

    def _update_incremental_ols(
        self,
        tf_name: str,
        latest_proxy_features: Dict[str, float],
        market_proxy_cache: pd.DataFrame,
        timestamp: datetime,
    ):
        """
        【インクリメンタルOLS版】
        Dequeへの追加と並行して ols_state の sum_x/sum_x2/sum_y/sum_xy/count を
        スライディングウィンドウで逐次更新する。
        x_deque と y_deque は同一タイミングで積み上げられるため、
        満杯判定は x_deque で統一して old_x/old_y のペアを正確に取得する。
        """
        from datetime import timezone

        # [Phase 9d 発見 #63] TF 毎可変 OLS 窓 (Phase 10 設計と整合)。
        # blueprint.NEUTRALIZATION_CONFIG["HF"]["window_per_tf"] が SSoT。
        OLS_WINDOW = self._get_ols_window(tf_name)

        try:
            search_ts = timestamp
            if search_ts.tzinfo is None:
                search_ts = search_ts.replace(tzinfo=timezone.utc)
            else:
                search_ts = search_ts.astimezone(timezone.utc)

            # ─────────────────────────────────────────────────────────────
            # [Phase 9d 発見 #65] pandas DatetimeIndex.get_indexer(method="ffill")
            # の thread-safety 問題を回避するため numpy searchsorted に置換
            # ─────────────────────────────────────────────────────────────
            # 旧実装:
            #     proxy_cache_unique = market_proxy_cache[
            #         ~market_proxy_cache.index.duplicated(keep="last")
            #     ].sort_index()
            #     idx = proxy_cache_unique.index.get_indexer(
            #         [search_ts], method="ffill"
            #     )[0]
            #
            # 問題:
            #     process_new_m05_bar の M3 close 経路で _recalc_one_tf が
            #     6 TF 並列実行される (ThreadPoolExecutor)。6 スレッドが同じ
            #     market_proxy_cache.index に同時アクセスすると、pandas
            #     DatetimeIndex の lazy hashtable 初期化 (.duplicated() /
            #     get_indexer 経由) で race condition が発生し、まれに
            #     "Reindexing only valid with uniquely valued Index objects"
            #     例外が投げられて、当該 TF の OLS 状態更新が skip される。
            #     production runtime では OLS 係数の微小 drift につながり、
            #     将来の Layer 1 v2 (post-OLS 比較) で byte-identical 比較不可。
            #
            # 新実装:
            #     numpy ndarray の read-only スナップショットを取って searchsorted
            #     で binary search する。pandas DatetimeIndex 機構を完全に迂回し、
            #     thread-safety を numpy の documented behavior で保証する。
            #     学習側 2_G の join_asof(strategy="backward") + fill_null(0.0)
            #     と数値完全一致を維持。
            # ─────────────────────────────────────────────────────────────
            latest_x = self._ffill_lookup_market_proxy(
                market_proxy_cache, search_ts
            )
            if not np.isfinite(latest_x):
                latest_x = 0.0

            # バッファ・状態の初期化
            if tf_name not in self.proxy_feature_buffers:
                self.proxy_feature_buffers[tf_name] = {
                    "market_proxy": deque(maxlen=OLS_WINDOW),
                    "__bar_ts__": deque(maxlen=OLS_WINDOW),  # [計測基盤] バー時刻並行記録
                }
            if tf_name not in self.ols_state:
                self.ols_state[tf_name] = {}

            x_deque = self.proxy_feature_buffers[tf_name]["market_proxy"]

            # x_dequeが満杯かどうかをループ前に1回だけ確認する
            # x_dequeとy_dequeは同一タイミングで積み上げられるため、
            # x_dequeの満杯 = y_dequeの満杯 が保証される
            x_is_full = len(x_deque) == OLS_WINDOW
            old_x = float(x_deque[0]) if x_is_full else 0.0

            for feat_name, latest_y in latest_proxy_features.items():
                if not np.isfinite(latest_y):
                    latest_y = 0.0

                if feat_name not in self.proxy_feature_buffers[tf_name]:
                    self.proxy_feature_buffers[tf_name][feat_name] = deque(
                        maxlen=OLS_WINDOW
                    )

                if feat_name not in self.ols_state[tf_name]:
                    self.ols_state[tf_name][feat_name] = {
                        "sum_x": 0.0,
                        "sum_x2": 0.0,
                        "sum_y": 0.0,
                        "sum_xy": 0.0,
                        "count": 0,
                    }

                state = self.ols_state[tf_name][feat_name]
                y_deque = self.proxy_feature_buffers[tf_name][feat_name]

                # ウィンドウ満杯なら最古の値を減算（x_dequeの満杯で統一判定）
                if x_is_full:
                    old_y = float(y_deque[0])
                    state["sum_x"] -= old_x
                    state["sum_x2"] -= old_x * old_x
                    state["sum_y"] -= old_y
                    state["sum_xy"] -= old_x * old_y
                    state["count"] -= 1

                # 新しい値を加算
                state["sum_x"] += latest_x
                state["sum_x2"] += latest_x * latest_x
                state["sum_y"] += latest_y
                state["sum_xy"] += latest_x * latest_y
                state["count"] += 1

                y_deque.append(latest_y)

            # x_dequeはループ後に1回だけ更新
            x_deque.append(latest_x)
            # [計測基盤] このバーの時刻 (tz正規化済 close_ts = search_ts) を並行記録。
            #   x_deque.append と同一地点なので length が常に一致する。
            #   旧 state 復元で欠けている場合は setdefault で生成 (以後 append で同期)。
            self.proxy_feature_buffers[tf_name].setdefault(
                "__bar_ts__", deque(maxlen=OLS_WINDOW)
            ).append(search_ts)

        except Exception as e:
            feat_name_safe = locals().get("feat_name", "<unknown>")
            self.logger.warning(
                f"[{tf_name}] バッファの更新に失敗 ({feat_name_safe}): {e}",
                exc_info=False,
            )

    def _calculate_neutralized_features(
        self,
        base_features_dict: Dict[str, float],
        tf_name: str,
        signal_timestamp: datetime,
        market_proxy_cache_df: pd.DataFrame,
    ) -> Dict[str, float]:
        """
        【V5完全修正版 + core_indicators統一版】
        OLS純化を core_indicators.neutralize_ols に統一し、
        学習側 (2_G_alpha_neutralizer.py) と物理的に同一のロジックを保証する。

        処理フロー:
            1. proxy_feature_buffers からウィンドウ分の x_arr / y_arr を抽出
            2. neutralize_ols 相当の incremental OLS を実行
               (window は TF 毎に blueprint.NEUTRALIZATION_CONFIG から取得、
                Phase 9d 発見 #63 の修正でハードコード 2016 を撤去)
            3. 結果配列の末尾要素 [-1] を最新の純化済み値として採用
        """
        neutralized_features: Dict[str, float] = {}

        try:
            latest_x_deque = self.proxy_feature_buffers.get(tf_name, {}).get(
                "market_proxy"
            )
            if not latest_x_deque:
                return base_features_dict

            # x_latest を x_deque から直接取得（x_arr生成不要）
            x_latest = (
                float(latest_x_deque[-1])
                if latest_x_deque and np.isfinite(latest_x_deque[-1])
                else 0.0
            )

            for base_name, latest_y in base_features_dict.items():
                # ols_stateからインクリメンタル統計量を取得
                state = self.ols_state.get(tf_name, {}).get(base_name)
                if state is None or state["count"] < 30:
                    neutralized_features[base_name] = latest_y
                    continue

                count = state["count"]
                mean_x = state["sum_x"] / count
                mean_x2 = state["sum_x2"] / count
                var_x = max(0.0, mean_x2 - mean_x * mean_x)
                mean_y = state["sum_y"] / count
                mean_xy = state["sum_xy"] / count
                cov_xy = mean_xy - mean_x * mean_y
                beta = cov_xy / (var_x + 1e-10)
                alpha = mean_y - beta * mean_x

                y_latest = float(latest_y) if np.isfinite(latest_y) else 0.0
                # ─────────────────────────────────────────────────────────────
                # [純化撤去 / unpurified §11.34.14] proxy OLS 回帰 (Y − (β·X + α))
                # を撤去し、生 Y をそのまま採用する恒等化 (β=0, α=0 と等価)。
                # 学習側の恒等コピー 2_G_alpha_neutralizer_unpurified.py L327:
                #     pl.col(base_name).cast(pl.Float64).fill_null(0.0)
                # と中身を一致させる。上の y_latest が
                #   float(latest_y) if np.isfinite(latest_y) else 0.0
                # = Float64 化 + 非有限(=欠損相当)→0.0 を担い、学習側
                # cast(Float64).fill_null(0.0) と整合する (engine QA で NaN/inf は
                # 構造的に消滅済のため「非有限→0」と「fill_null(0)」は等価)。
                # beta / x_latest / alpha は上で計算されるが、ここで一切参照しない
                # ため出力に影響しない (proxy 非依存 = proxy 経由の train-serve skew
                # 発生源が構造的に消滅)。proxy_feature_buffers / ols_state の充填は
                # warmup 経路でそのまま走るが死んだ計算であり無害。proxy 機構の
                # 物理削除 (クリーンアップ) は動作確認後の別段階で行う。
                # ─────────────────────────────────────────────────────────────
                val = y_latest

                if np.isfinite(val):
                    neutralized_features[base_name] = val
                else:
                    latest_y_safe = latest_y if np.isfinite(latest_y) else 0.0
                    neutralized_features[base_name] = latest_y_safe
            return neutralized_features

        except Exception as e:
            self.logger.error(f"アルファ純化 ({tf_name}) に失敗: {e}", exc_info=True)
            return base_features_dict

    def _any_artifact_loaded(self, tf_name: str) -> bool:
        """
        [Phase 9d 発見 #66 Phase D-3] 該当 TF の QAState (各モジュール) のうち
        少なくとも 1 つが学習側 artifact から load されているか判定。

        warmup loop / smoke test 等の「QAState を update すべきでない」経路で、
        update_and_clip(skip_update=True) を渡すかどうかの判定に使う。
        """
        tf_qa = self.qa_states.get(tf_name, {})
        return any(getattr(qs, "_artifact_loaded", False) for qs in tf_qa.values())

    def _calculate_base_features(
        self,
        data: Dict[str, np.ndarray],
        tf_name: str,
        skip_qa_update: bool = False,
    ) -> Dict[str, float]:
        """
        【Phase 9b 改修版: 司令塔統合 .select()】

        各モジュール (1A〜1F) の `_build_polars_pieces` から
        (columns, exprs, layer2) を収集し、統合 DataFrame に対する
        単一の `.select()` で全 505 特徴量を一括計算する。

        効果:
            - 旧 (Phase 9 / Step B): 6 モジュール × `df.lazy().select(exprs).tail(1).collect()`
              → FFI overhead × 6 / TF
            - 新 (Phase 9b): 全モジュールの式を 1 つの DataFrame に対して .select()
              → FFI overhead × 1 / TF (期待: 各 TF 75-110ms → 30-50ms)

        AI 分布への影響:
            なし。Polars クエリープランナーは各 alias 式を独立に評価するため、
            統合 .select() でも各特徴量の数値は単独 .select() と完全一致する
            (CSE で重複サブグラフは 1 度しか計算されない)。

        QA 振り分け:
            プレフィックス e1a_/e1b_/.../e1f_ で qa_states[tf_name][module_id] を
            参照。e1d_sample_weight / e1e_sample_weight は QA 対象外
            (学習側 base_columns 扱いと一致、Phase 5 #36)。

        [Phase 9d 発見 #66 Phase D-3] skip_qa_update 引数:
            True の場合、qa_state.update_and_clip(skip_update=True) を渡して
            EWM 状態の更新を抑止する。clip 自体は適用される。
            これは learning-side QAState artifact を本番側で load した後の
            warmup 期間中に追加 update が起きるのを防ぎ、artifact 状態を
            純粋に維持するための仕組み (司令塔 _recalc_one_tf 経由)。
            デフォルト False (旧挙動互換)。
        """
        # === [§B.12.10.X cell-level deque trace] 環境変数で制御される読み取り専用 dump フック ===
        # FORGE_DEQ_TRACE_TARGETS が空 (= default) なら早期 return、production 動作に影響ゼロ。
        # 設定時のみ target (tf, ts) に一致した呼び出しで `data` (= OHLCV dict of np.array) を
        # pickle dump する。monkey-patch ではない (production code 内に追加された静的フック)
        # ので numba JIT cache 汚染リスクなし (§B.12.11.2)。
        #
        # 使い方:
        #   FORGE_DEQ_TRACE_TARGETS="M0.5:2026-04-01T06:48:00,M0.5:2026-04-01T06:48:30" \
        #     python3 run_shadow_test.py ...
        #   → /tmp/forge_deque_dump/deque_M0_5_2026-04-01T06-48-00.pkl が 1 ファイルだけ生成される
        _dump_targets_env = os.environ.get("FORGE_DEQ_TRACE_TARGETS", "")
        if _dump_targets_env:
            _ts_at_call = self.last_bar_timestamps.get(tf_name)
            if _ts_at_call is not None:
                _ts_iso = pd.Timestamp(_ts_at_call).strftime("%Y-%m-%dT%H:%M:%S")
                _target_set = {
                    tuple(tok.strip().split(":", 1))
                    for tok in _dump_targets_env.split(",") if ":" in tok
                }
                if (tf_name, _ts_iso) in _target_set:
                    _dump_dir = Path(
                        os.environ.get("FORGE_DEQ_TRACE_DIR", "/tmp/forge_deque_dump")
                    )
                    _dump_dir.mkdir(parents=True, exist_ok=True)
                    _safe_ts = _ts_iso.replace(":", "-")
                    _safe_tf = tf_name.replace(".", "_")
                    _dump_path = _dump_dir / f"deque_{_safe_tf}_{_safe_ts}.pkl"
                    if not _dump_path.exists():  # 1 cell につき 1 回だけ
                        _snapshot = {
                            "tf": tf_name,
                            "ts": _ts_iso,
                            "skip_qa_update": skip_qa_update,
                            "data": {k: np.asarray(v).copy() for k, v in data.items()},
                            "data_lengths": {k: len(v) for k, v in data.items()},
                        }
                        with open(_dump_path, "wb") as _f:
                            pickle.dump(_snapshot, _f)
                        self.logger.info(
                            f"[deque trace] dumped {_dump_path} "
                            f"(skip_qa_update={skip_qa_update})"
                        )
        # === end deque trace hook ===

        # [乖離①修正] qa_stateとlookback_barsを時間足に合わせて渡す
        tf_qa = self.qa_states.get(tf_name, {})
        lb = TIMEFRAME_BARS_PER_DAY.get(tf_name, 1440)

        features: Dict[str, float] = {}

        # ---------------------------------------------------------------
        # 1. 各モジュールから (columns, exprs, layer2) を収集
        # ---------------------------------------------------------------
        try:
            cols_a, exprs_a, l2_a = FeatureModule1A._build_polars_pieces(data, lb)
            cols_b, exprs_b, l2_b = FeatureModule1B._build_polars_pieces(data, lb)
            cols_c, exprs_c, l2_c = FeatureModule1C._build_polars_pieces(data, lb)
            cols_d, exprs_d, l2_d = FeatureModule1D._build_polars_pieces(data, lb)
            cols_e, exprs_e, l2_e = FeatureModule1E._build_polars_pieces(data, lb)
            cols_f, exprs_f, l2_f = FeatureModule1F._build_polars_pieces(data, lb)
        except Exception as e:
            self.logger.error(
                f"_build_polars_pieces 収集中にエラー ({tf_name}): {e}",
                exc_info=True,
            )
            cols_a = cols_b = cols_c = cols_d = cols_e = cols_f = {}
            exprs_a = exprs_b = exprs_c = exprs_d = exprs_e = exprs_f = []
            l2_a    = l2_b    = l2_c    = l2_d    = l2_e    = l2_f    = {}

        # ---------------------------------------------------------------
        # 2. 統合 columns/exprs/layer2 を構築
        #
        # 列名衝突は dict.update で同名 key 上書き → 同値なので問題なし。
        # 共通列 (close/high/low/open/volume) と __temp_atr_13 は複数モジュールで
        # 同じ値を入れているため、上書きしても影響なし。
        # 1F は columns/exprs が空なので何も寄与しない (layer2 のみマージ)。
        # ---------------------------------------------------------------
        all_columns: Dict[str, np.ndarray] = {
            **cols_a, **cols_b, **cols_c, **cols_d, **cols_e, **cols_f,
        }
        all_exprs: List[pl.Expr] = (
            exprs_a + exprs_b + exprs_c + exprs_d + exprs_e + exprs_f
        )
        all_layer2: Dict[str, float] = {
            **l2_a, **l2_b, **l2_c, **l2_d, **l2_e, **l2_f,
        }

        # === [§B.12.12.X cell-level pieces trace] 2 番目の dump フック ===
        # _build_polars_pieces 実行直後の各モジュールの cols_a/d/e から
        # __temp_atr_13 配列を dump する。これにより:
        #   - production が実際に生成した ATR 配列 (cols_a の値 = rfe_1A の出力)
        #   - dict.update 後勝者となる cols_e の ATR 配列 (rfe_1E の出力)
        #   - all_columns の最終 __temp_atr_13 (= polars DataFrame 内の divisor)
        # を取得できる。trace_one_cell.py 側で「dump された data から独立計算した ATR」
        # との bit 比較が可能になる。
        # production 動作には影響ゼロ (環境変数 FORGE_DEQ_TRACE_TARGETS 設定時のみ動作)。
        if _dump_targets_env:
            _ts_at_call = self.last_bar_timestamps.get(tf_name)
            if _ts_at_call is not None:
                _ts_iso = pd.Timestamp(_ts_at_call).strftime("%Y-%m-%dT%H:%M:%S")
                _target_set = {
                    tuple(tok.strip().split(":", 1))
                    for tok in _dump_targets_env.split(",") if ":" in tok
                }
                if (tf_name, _ts_iso) in _target_set:
                    _dump_dir = Path(
                        os.environ.get("FORGE_DEQ_TRACE_DIR", "/tmp/forge_deque_dump")
                    )
                    _dump_dir.mkdir(parents=True, exist_ok=True)
                    _safe_ts = _ts_iso.replace(":", "-")
                    _safe_tf = tf_name.replace(".", "_")
                    _pieces_path = _dump_dir / f"pieces_{_safe_tf}_{_safe_ts}.pkl"
                    if not _pieces_path.exists():  # 1 cell につき 1 回だけ
                        def _maybe_arr(d, key):
                            v = d.get(key) if isinstance(d, dict) else None
                            return np.asarray(v).copy() if v is not None else None
                        _pieces_snap = {
                            "tf": tf_name,
                            "ts": _ts_iso,
                            # 各モジュールが出した __temp_atr_13 (= raw or +1e-10)
                            "cols_a_temp_atr_13": _maybe_arr(cols_a, "__temp_atr_13"),
                            "cols_d_temp_atr_13": _maybe_arr(cols_d, "__temp_atr_13"),
                            "cols_e_temp_atr_13": _maybe_arr(cols_e, "__temp_atr_13"),
                            # all_columns 内の最終勝者 (= polars 内で divisor に使われる値)
                            "all_columns_temp_atr_13": _maybe_arr(all_columns, "__temp_atr_13"),
                            # 各モジュールの close (input data の確認用、同じはず)
                            "cols_a_close": _maybe_arr(cols_a, "close"),
                            "cols_e_close": _maybe_arr(cols_e, "close"),
                        }
                        with open(_pieces_path, "wb") as _f:
                            pickle.dump(_pieces_snap, _f)
                        self.logger.info(
                            f"[deque trace] pieces dumped {_pieces_path}"
                        )
        # === end pieces trace hook ===

        # ---------------------------------------------------------------
        # 3. 統合 DataFrame で単一 .select() を実行 (FFI overhead 1 回)
        # ---------------------------------------------------------------
        if all_columns and all_exprs:
            try:
                df = pl.DataFrame(all_columns)
                polars_results = (
                    df.lazy().select(all_exprs).tail(1).collect().to_dicts()[0]
                )
                for k, v in polars_results.items():
                    features[k] = float(v) if v is not None else np.nan
            except Exception as e:
                self.logger.error(
                    f"統合 .select() 実行中にエラー ({tf_name}): {e}",
                    exc_info=True,
                )

        # Layer 2 (Numba UDF 直接呼び結果 + 1F の全特徴量) をマージ
        features.update(all_layer2)

        # ---------------------------------------------------------------
        # 4. QA 処理 (プレフィックスでモジュール振り分け)
        #
        # e1a_/e1b_/.../e1f_ 始まりの特徴量を該当モジュールの QAState で処理。
        # sample_weight (e1d_/e1e_) は QA 対象外 (Phase 5 #36)。
        # ---------------------------------------------------------------
        # sample_weight は学習側 base_columns 扱いで QA 対象外
        _SAMPLE_WEIGHT_KEYS = ("e1d_sample_weight", "e1e_sample_weight")

        qa_results: Dict[str, float] = {}
        for k, v in features.items():
            if k in _SAMPLE_WEIGHT_KEYS:
                # QA をスキップ。inf/NaN はそのまま (sample_weight はそもそも有限)
                qa_results[k] = v
                continue

            # プレフィックスから モジュール ID を抽出 ("e1a_..." → "1A")
            prefix = k.split("_", 1)[0]  # "e1a", "e1b", ...
            if len(prefix) == 3 and prefix.startswith("e1"):
                module_id = prefix[1:].upper()  # "1A", "1B", ...
                qa_state = tf_qa.get(module_id)
                if qa_state is not None:
                    # [Phase 9d 発見 #66 Phase D-3] skip_qa_update を伝播。
                    # warmup 中かつ artifact load 済の場合のみ True で
                    # EWM 状態更新を抑止 (clip は適用される)。
                    qa_results[k] = qa_state.update_and_clip(
                        k, v, skip_update=skip_qa_update
                    )
                    continue

            # プレフィックス不一致 / qa_state 不在 → inf/NaN フォールバックのみ
            qa_results[k] = v if np.isfinite(v) else 0.0

        features = qa_results

        # ---------------------------------------------------------------
        # 5. 純化用プロキシ (必須) の計算
        # [TRAIN-SERVE-FIX] 学習側 s1_1_C_enrich.py と完全一致させる:
        #   VOLATILITY_WINDOW = 20, VOLUME_WINDOW = 50, MOMENTUM_WINDOW = 5
        #   log_return         = np.log(close[t] / close[t-1])
        #   rolling_volatility = log_return.rolling(20, min_periods=1).std(ddof=1)
        #   price_momentum     = close[t] / close[t-5] - 1   (5本前比リターン)
        #   rolling_avg_volume = volume.rolling(50, min_periods=1).mean()
        #   volume_ratio       = volume / rolling_avg_volume
        # ---------------------------------------------------------------
        VOLATILITY_WINDOW = 20
        VOLUME_WINDOW = 50
        MOMENTUM_WINDOW = 5

        def _window(arr: np.ndarray, window: int) -> np.ndarray:
            return arr[-window:] if len(arr) >= window else arr

        # atr の計算 (e1c_atr_13は相対値のため使用禁止 → core_indicators.calculate_atr_wilderで統一)
        high, low, close = data["high"], data["low"], data["close"]
        if len(close) > 1:
            atr_arr = calculate_atr_wilder(
                high.astype(np.float64),
                low.astype(np.float64),
                close.astype(np.float64),
                self.ATR_CALC_PERIOD,
            )
            atr_last = float(atr_arr[-1]) if len(atr_arr) > 0 else np.nan
            # nan ガード: バッファが ATR_CALC_PERIOD 未満の極端な起動直後でも 0.0 で安全に逃げる
            features["atr"] = atr_last if np.isfinite(atr_last) else 0.0
        else:
            features["atr"] = 0.0

        # [TRAIN-SERVE-FIX] log_return: np.log(close[t] / close[t-1])
        # 学習側: close_shifted = close.shift(1).replace(0, 1e-12)
        #         log_return = np.log((close / close_shifted).fillna(1.0))
        if len(close) > 1:
            prev_close = close[-2] if close[-2] != 0 else 1e-12
            features["log_return"] = float(np.log(close[-1] / prev_close))
        else:
            features["log_return"] = 0.0

        # [TRAIN-SERVE-FIX] price_momentum: close[t] / close[t-5] - 1（5本前比リターン）
        # 学習側: close_shifted_momentum = close.shift(5).replace(0, 1e-12)
        #         price_momentum = close / close_shifted_momentum - 1
        if len(close) > MOMENTUM_WINDOW:
            prev_close_mom = close[-1 - MOMENTUM_WINDOW] if close[-1 - MOMENTUM_WINDOW] != 0 else 1e-12
            features["price_momentum"] = float(close[-1] / prev_close_mom - 1.0)
        else:
            features["price_momentum"] = np.nan

        # [TRAIN-SERVE-FIX] rolling_volatility: log_returnのrolling(20).std(ddof=1)
        # 学習側: log_return.rolling(VOLATILITY_WINDOW, min_periods=1).std(ddof=1)
        # 本番では直近(VOLATILITY_WINDOW)バー分のlog_returnを計算してstd(ddof=1)
        if len(close) >= 2:
            # 直近 VOLATILITY_WINDOW + 1 本のcloseから VOLATILITY_WINDOW 個のlog_returnを生成
            n_window = min(VOLATILITY_WINDOW + 1, len(close))
            close_window = close[-n_window:]
            # ゼロ保護
            close_safe = np.where(close_window[:-1] == 0, 1e-12, close_window[:-1])
            log_returns_window = np.log(close_window[1:] / close_safe)
            # ddof=1で不偏推定（最低2サンプル必要）
            if len(log_returns_window) >= 2:
                features["rolling_volatility"] = float(np.std(log_returns_window, ddof=1))
            else:
                features["rolling_volatility"] = 0.0
        else:
            features["rolling_volatility"] = np.nan

        # [TRAIN-SERVE-FIX] volume_ratio: volume / rolling_avg_volume(window=50)
        # 学習側: rolling_avg_volume = volume.rolling(50, min_periods=1).mean()
        #         volume_ratio = volume / rolling_avg_volume.replace(0, 1.0)
        if len(data["volume"]) > 0:
            vol_window = _window(data["volume"], VOLUME_WINDOW)
            avg_vol = float(np.mean(vol_window))
            if avg_vol == 0:
                avg_vol = 1.0  # 学習側のreplace(0, 1.0)を再現
            features["volume_ratio"] = float(data["volume"][-1] / avg_vol)
        else:
            features["volume_ratio"] = np.nan

        return features

    # [発見#D対応] 純化対象外として明示的に許可するベース名のセット。
    # create_proxy_labels が S6 出力に追加する非純化カラムが該当する。
    # 学習側 (create_proxy_labels.py L902, L947) で `pl.col("atr_ratio")` として
    # S6 に書き込まれ、特徴量名簿に `atr_ratio_M3` の形で含まれる。
    # ここに無いベース名で `_neutralized_` も含まない名前は、学習側に存在しない
    # カラム名である可能性が高いため、警告ログを出す。
    NON_NEUTRALIZED_BASE_NAMES = frozenset({
        "atr_ratio",
        "session_atr_ratio",  # [SESSION-RATIO] S6 に session_atr_ratio として書き込み・非純化
    })

    def calculate_feature_vector(
        self, tf_name: str, timestamp: datetime, market_proxy_cache: pd.DataFrame
    ) -> Optional[np.ndarray]:
        """
        [ベクトル生成] 304個の精鋭リストに厳密準拠したベクトルを構築。

        [発見#D対応] base_name 抽出ロジックを厳密化:
          - 特徴量名に '_neutralized_' を含む場合: 純化済み特徴量として cache から取得
          - 含まない場合: NON_NEUTRALIZED_BASE_NAMES に含まれる場合のみ非純化値として取得。
            それ以外の名前は学習側との数値不一致リスクがあるため、初回のみ警告ログを出す。
        """
        if not self.is_buffer_filled[tf_name]:
            return None

        try:
            vector = []
            tf_pattern = re.compile(r"_(M[0-9\.]+|H[0-9]+|D[0-9]+|W[0-9]+|MN|tick)$")

            for feat_name in self.feature_list:
                m = tf_pattern.search(feat_name)
                target_tf = m.group(1) if m else None
                if not target_tf or target_tf not in self.latest_features_cache:
                    vector.append(0.0)
                    continue

                # [発見#D対応] base_name 抽出を厳密化
                if "_neutralized_" in feat_name:
                    # 純化済み: 例 "e1c_atr_13_neutralized_M3" → "e1c_atr_13"
                    base_name = feat_name.split("_neutralized_")[0]
                else:
                    # 非純化: 例 "atr_ratio_M3" → "atr_ratio"
                    # 末尾の _<TF> を剥がして base_name を得る
                    base_name = feat_name[: m.start()] if m else feat_name
                    # 学習側に存在しない可能性のある名前を検知して警告 (初回のみ)
                    if (
                        base_name not in self.NON_NEUTRALIZED_BASE_NAMES
                        and feat_name not in self._warned_unknown_features
                    ):
                        self.logger.warning(
                            f"⚠️ 特徴量 '{feat_name}' は純化済み('_neutralized_'を含まず)、"
                            f"かつ非純化許可リスト{set(self.NON_NEUTRALIZED_BASE_NAMES)}に"
                            f"も該当しません。cache から '{base_name}' を取得しますが、"
                            f"学習側 S6 にこのカラムが存在しないため数値が一致しない可能性があります。"
                            f"特徴量名簿の生成元 (split_features_first_orthogonal.py 等) を確認してください。"
                        )
                        self._warned_unknown_features.add(feat_name)

                val = self.latest_features_cache[target_tf].get(base_name, 0.0)
                vector.append(val)

            # 【アーキテクチャ設計メモ: 最強の出口フィルター】
            # ファイル冒頭でミュートしたNumpyのゼロ除算等による異常値(inf, NaN)は、
            # AIモデル(LightGBM)に渡る直前のここで、一括して安全な 0.0 に浄化される。
            final_vector = np.nan_to_num(
                np.array(vector, dtype=np.float64), nan=0.0, posinf=0.0, neginf=0.0
            )
            # ▼修正: 10万でのクリッピングを撤廃 (VPTなどの自然な巨大値が切り捨てられOODになるのを防ぐ)
            return np.array([final_vector])
        except Exception as e:
            self.logger.error(f"Vector calculation error: {e}")
            return None

    def save_state(self, filepath: str) -> bool:
        """
        特徴量エンジンの内部状態をアトミックにPickle保存する。
        Ctrl+Cなどの強制終了時でもファイルの破損を完全に防ぎます。
        """
        temp_filepath = f"{filepath}.tmp"

        try:
            # ▼▼ 保存する中身はあなたの元のコードのまま維持 ▼▼
            state_data = {
                "data_buffers": self.data_buffers,
                "is_buffer_filled": self.is_buffer_filled,
                "last_bar_timestamps": self.last_bar_timestamps,
                "latest_features_cache": self.latest_features_cache,
                "m05_dataframe": self.m05_dataframe,
                "proxy_feature_buffers": self.proxy_feature_buffers,
                "ols_state": self.ols_state,
                "qa_states": self.qa_states,  # [乖離①修正] QAStateをスナップショットに含める
            }

            # 1. まず一時ファイル (.tmp) に書き込む
            with open(temp_filepath, "wb") as f:
                pickle.dump(state_data, f, protocol=pickle.HIGHEST_PROTOCOL)

                # 2. OSのバッファをフラッシュし、物理ディスクへの書き込みを強制する (fsync)
                f.flush()
                import os  # 念のため関数内でインポートしておく

                os.fsync(f.fileno())

            # 3. 書き込みが完璧に完了したら、一瞬で本番ファイルとすり替える（アトミック操作）
            os.replace(temp_filepath, filepath)

            self.logger.info(
                f"✓ Feature-engine state snapshotted (atomic): {filepath}"
            )
            return True

        except Exception as e:
            self.logger.error(f"✗ 特徴量エンジンの状態保存に失敗: {e}", exc_info=True)
            # エラーや強制終了が起きた場合は、書き込み途中のゴミ(.tmp)を削除して元ファイルを保護
            import os

            if os.path.exists(temp_filepath):
                try:
                    os.remove(temp_filepath)
                except OSError:
                    pass
            return False

    def load_state(self, filepath: str) -> bool:
        """保存されたファイルから特徴量バッファとOLS状態を瞬時に復元する"""
        if not os.path.exists(filepath):
            return False

        try:
            with open(filepath, "rb") as f:
                state_data = pickle.load(f)

            self.data_buffers = state_data["data_buffers"]
            self.is_buffer_filled = state_data["is_buffer_filled"]
            self.last_bar_timestamps = state_data["last_bar_timestamps"]
            self.latest_features_cache = state_data["latest_features_cache"]
            self.m05_dataframe = state_data.get("m05_dataframe", self.m05_dataframe)

            # 後方互換性のため get() を使用
            self.proxy_feature_buffers = state_data.get(
                "proxy_feature_buffers", self.proxy_feature_buffers
            )
            # [計測基盤] 旧 state には __bar_ts__ が無い (または長さ不一致)。
            #   market_proxy (= x_deque) と長さを揃えて None パディングし、 位置対応を
            #   壊さない。 以後の _update_incremental_ols append で実時刻が入る。
            for _tf, _bufs in self.proxy_feature_buffers.items():
                _mp = _bufs.get("market_proxy")
                if _mp is not None:
                    _bt = _bufs.get("__bar_ts__")
                    if _bt is None or len(_bt) != len(_mp):
                        _bufs["__bar_ts__"] = deque(
                            [None] * len(_mp), maxlen=_mp.maxlen
                        )
            self.ols_state = state_data.get("ols_state", self.ols_state)
            self.qa_states = state_data.get("qa_states", self.qa_states)  # [乖離①修正]

            self.logger.info(
                f"✓ Feature-engine state restored from snapshot: {filepath}"
            )
            return True
        except Exception as e:
            self.logger.error(f"✗ 特徴量エンジンの状態復元に失敗: {e}", exc_info=True)
            return False

    # ▲▲▲ ここまで追加 ▲▲▲


# =====================================================================
# [Phase4: profiling_patch 撤廃] (2026-04-30 final fix)
# =====================================================================
# 旧コードはここで profiling_patch.process_new_m05_bar / _calculate_base_features
# をモンキーパッチで上書きしていた。しかし profiling_patch.py の中身は古いコピーで:
#
#   1. process_new_m05_bar に warmup_only 引数がない
#      → STALE-GUARD 復帰時に TypeError → 「スナップショット破損」と誤検知され
#         毎回フルウォームアップが走っていた
#
#   2. _calculate_base_features のプロキシ特徴量4つが旧実装のまま
#      → 学習側 s1_1_C_enrich.py と完全に異なる値を返していた:
#         - price_momentum: close[-1]-close[-11] (差分・ドル単位)
#                          ← 学習側は close[t]/close[t-5]-1 (5本前比リターン)
#         - rolling_volatility: pct_change[-20:].std(ddof=0)
#                               ← 学習側は log_return.rolling(20).std(ddof=1)
#         - volume_ratio: window=20
#                        ← 学習側は window=50
#         - log_return: 微小な ε 保護差 (影響軽微)
#      → 監査乖離#5 の修正が本番では効いていなかった (本体側は正しいが上書きされていた)
#
# このパッチの本来の目的は処理時間計測のみ (PROFILING_ENABLED フラグでログ抑制可能)
# だったが、PROFILING_ENABLED に関わらずメソッド本体が無条件に上書きされる実装ミス。
# 今後パフォーマンス計測が必要な場合は、本体クラスに @timer デコレータや
# logger.debug のタイマーを直接仕込む方式に切り替えること。
#
# from execution import profiling_patch
#
# RealtimeFeatureEngine.process_new_m05_bar = profiling_patch.process_new_m05_bar
# RealtimeFeatureEngine._calculate_base_features = (
#     profiling_patch._calculate_base_features
# )
