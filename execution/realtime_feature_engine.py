# /workspace/execution/realtime_feature_engine.py
import sys
from pathlib import Path
from collections import deque
import numpy as np
import pandas as pd
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

# --- core_indicators: Single Source of Truth ---
sys.path.append(str(config.CORE_DIR))
from core_indicators import calculate_atr_wilder, calculate_barrier_atr, neutralize_ols  # noqa: E402

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
    【Project Cimera V5: オーケストレーター】
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

    def __init__(
        self,
        feature_list_path: str = str(config.S3_FEATURES_FOR_TRAINING_V5),
    ):
        self.logger = logging.getLogger("ProjectCimera.FeatureEngine")

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
                f"特徴量名簿 ({len(self.feature_list)}個) をロードしました。"
            )
        except Exception as e:
            self.logger.critical(f"特徴量名簿 {feature_list_path} のロードに失敗: {e}")
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
                    f"  -> {tf_name:<3} バッファ初期化 (Lookback: {self.lookbacks_by_tf[tf_name]})"
                )

            lookback = self.lookbacks_by_tf[tf_name]

            self.data_buffers[tf_name] = {
                col: deque(maxlen=lookback) for col in self.OHLCV_COLS
            }
            # [DISC-FLAG] 不連続フラグバッファ: resampleでNaNになった足はTrue
            # discフラグがTrueの足ではTR計算時に前Closeを使わず H-L のみで計算する
            self.data_buffers[tf_name]["disc"] = deque(maxlen=lookback)
            self.is_buffer_filled[tf_name] = False
            self.last_bar_timestamps[tf_name] = None
            self.latest_features_cache[tf_name] = {}

        # 4. M1データを保持するDeque (リサンプリング元)
        max_lookback_val = (
            max(self.lookbacks_by_tf.values()) if self.lookbacks_by_tf else 1000
        )
        max_m05_bars_needed = max_lookback_val * 2880 + 1000
        self.m05_dataframe: deque[Dict[str, Any]] = deque(maxlen=max_m05_bars_needed)

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
            # ▼修正: OLS純化バッファは、どの時間足でもStage3と完全一致の「2016」で固定
            self.proxy_feature_buffers[tf_name] = {
                feat: deque(maxlen=2016) for feat in PROXY_FEATURES
            }
            self.proxy_feature_buffers[tf_name]["market_proxy"] = deque(maxlen=2016)

            self.ols_state[tf_name] = {}
            # 各エントリは _update_incremental_ols で特徴量登場時に動的初期化される

        self.logger.info(f"M0.5 Dequeバッファを初期化 (maxlen: {max_m05_bars_needed})")

        # 6. JITコンパイルのウォームアップ
        self._warmup_jit()

        # 7. 各時間足・各モジュール(1A〜1F)のQAStateを初期化
        # [乖離①修正] 学習側 apply_quality_assurance_to_group と等価のQA処理を有効化
        # lookback_barsは時間足ごとの1日バー数（M3=480等）を使用
        self.qa_states: Dict[str, Dict[str, Any]] = {}
        for tf_name in self.ALL_TIMEFRAMES.keys():
            if self.ALL_TIMEFRAMES[tf_name] is None:
                continue
            lb = TIMEFRAME_BARS_PER_DAY.get(tf_name, 1440)
            self.qa_states[tf_name] = {
                "1A": FeatureModule1A.QAState(lookback_bars=lb),
                "1B": FeatureModule1B.QAState(lookback_bars=lb),
                "1C": FeatureModule1C.QAState(lookback_bars=lb),
                "1D": FeatureModule1D.QAState(lookback_bars=lb),
                "1E": FeatureModule1E.QAState(lookback_bars=lb),
                "1F": FeatureModule1F.QAState(lookback_bars=lb),
            }
        self.logger.info("✓ 全時間足のQAStateを初期化しました。")

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
        self.logger.info("外部モジュールのJITウォームアップを開始します...")
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

            self.logger.info("✓ JITウォームアップ完了。")
        except Exception as e:
            self.logger.warning(f"JITウォームアップ中に警告: {e}")

    def _parse_feature_list_and_get_lookbacks(
        self, feature_list: List[str]
    ) -> Dict[str, int]:
        """名簿に基づき、必要な最大ルックバック期間を決定（デッドコードのe2aロジックはパージ済）"""
        tf_pattern = re.compile(r"_(M[0-9\.]+|H[0-9]+|D[0-9]+|W[0-9]+|MN|tick)$")
        lookbacks: Dict[str, int] = {}

        for feature_name in feature_list:
            tf_match = tf_pattern.search(feature_name)
            if not tf_match:
                continue
            tf_name_parsed = tf_match.group(1)
            if tf_name_parsed not in lookbacks:
                lookbacks[tf_name_parsed] = 0

        # MFDFA等の特大要求が消えたため、ベース特徴量計算用マージンのみ確保
        # ▼修正: 純化の2016期間を完全にカバーするため1000から2016に変更
        SAFE_MIN_LOOKBACK = 2016

        final_lookbacks = {}
        for tf_name_parsed in lookbacks.keys():
            req_size = max(lookbacks[tf_name_parsed], SAFE_MIN_LOOKBACK)
            final_lookbacks[tf_name_parsed] = req_size + 100
            self.logger.info(
                f"  -> {tf_name_parsed:<3} 最大ルックバック: {final_lookbacks[tf_name_parsed]}"
            )

        return final_lookbacks

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
        本メソッドはウォームアップ・リアルタイム双方から呼ばれる単一の disc 推定器として機能する
        (Train-Serve Skew Free)。

        判定ルール:
            disc[i] = (timestamp[i] - timestamp[i-1]) > freq_seconds * 1.5
            先頭バーは便宜上 False (前バーがないため連続扱い)

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

        threshold_ns = int(freq_seconds * 1.5 * 1_000_000_000)
        ts_int = out.index.astype("int64").to_numpy()
        gaps = np.diff(ts_int, prepend=ts_int[0])
        gaps[0] = 0  # 先頭バーは disc=False
        out["disc"] = gaps > threshold_ns
        return out

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

        OLS_WINDOW = 2016
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

            base_features = self._calculate_base_features(data, tf_name)

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

        self.logger.info(f"  -> M0.5 バッファをMT5データから充填中...")
        self._replace_buffer_from_dataframe("M0.5", m05_history_pd, market_proxy_cache)

        # M1をM0.5からリサンプリングして生成し、m05_dataframeを構築
        self.logger.info(f"  -> M1  をM0.5からリサンプリング中...")
        m1_history_pd = (
            m05_history_pd.resample("1min")
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
        # [DISC-FLAG] タイムスタンプ差から不連続フラグを推定
        #   学習側 s1_1_B の DISC-FLAG 付与ロジックと完全一致させる。
        m1_history_pd = self._add_disc_column(m1_history_pd, freq_seconds=60)
        self._replace_buffer_from_dataframe("M1", m1_history_pd, market_proxy_cache)

        self.m05_dataframe.clear()
        m05_records = m05_history_pd.reset_index().to_dict("records")
        self.m05_dataframe.extend(m05_records)

        # TF_RESAMPLE_RULES → 想定秒数のマッピング (s1_1_B と完全一致)
        _freq_seconds_map = {
            "1min": 60, "3min": 180, "5min": 300, "8min": 480,
            "15min": 900, "30min": 1800,
            "1h": 3600, "1H": 3600, "4h": 14400, "4H": 14400,
            "6h": 21600, "6H": 21600, "12h": 43200, "12H": 43200,
            "1D": 86400, "1d": 86400,
        }

        for tf_name, rule in self.TF_RESAMPLE_RULES.items():
            if tf_name not in self.data_buffers or tf_name in ("M0.5", "M1"):
                continue

            try:
                self.logger.info(f"  -> {tf_name:<3} をM0.5からリサンプリング中...")
                resampled_df = (
                    m05_history_pd.resample(rule)
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

                # [DISC-FLAG] タイムスタンプ差から不連続フラグを推定
                expected_sec = _freq_seconds_map.get(rule, 0)
                resampled_df = self._add_disc_column(
                    resampled_df, freq_seconds=expected_sec
                )

                self._replace_buffer_from_dataframe(
                    tf_name, resampled_df, market_proxy_cache
                )
            except Exception as e:
                self.logger.error(f"{tf_name} のリサンプリング充填に失敗: {e}")

        self.logger.info("✓ 全バッファの初期充填が完了しました。")

    def _append_bar_to_buffer(
        self,
        tf_name: str,
        bar_df: pd.DataFrame,
        market_proxy_cache: pd.DataFrame,
    ) -> None:
        """
        バッファに新しいバー (DataFrame形式) を追加し、
        純化(OLS)状態を逐次更新する。
        """
        if tf_name not in self.data_buffers:
            return

        try:
            bar_dict = bar_df.iloc[0].to_dict()
            bar_timestamp = bar_df.index[0]

            # 1. OHLCVバッファを更新
            for col in self.OHLCV_COLS:
                self.data_buffers[tf_name][col].append(bar_dict[col])
            # [DISC-FLAG] 不連続フラグを書き込む（デフォルトはFalse=連続）
            disc_flag = bool(bar_dict.get("disc", False))
            self.data_buffers[tf_name]["disc"].append(disc_flag)
            self.last_bar_timestamps[tf_name] = bar_timestamp

            # ★ 2, 3 の古い限定的OLS更新処理を削除 (process_new_m05_bar内で全特徴量を一括更新するため)

            # 4. 充填状態を更新
            if not self.is_buffer_filled[tf_name]:
                self.is_buffer_filled[tf_name] = True
                self.logger.info(f"✅ {tf_name} バッファ計算開始 (Best-Effort)。")

        except KeyError as e:
            self.logger.error(f"バーデータ {tf_name} にキーがありません: {e}")
        except Exception as e:
            self.logger.error(f"バー {tf_name} の追加に失敗: {e}")

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
            resampled_df = (
                new_m05_data.resample(rule)
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

            # 2. タイムスタンプ差から disc 推定
            #    s1_1_B の TIMEFRAME_FREQ_SECONDS と完全一致させる
            _freq_seconds_map = {
                "1min": 60, "3min": 180, "5min": 300, "8min": 480,
                "15min": 900, "30min": 1800, "1h": 3600, "1H": 3600,
                "4h": 14400, "4H": 14400, "6h": 21600, "6H": 21600,
                "12h": 43200, "12H": 43200, "1D": 86400, "1d": 86400,
            }
            expected_sec = _freq_seconds_map.get(rule, 0)
            if expected_sec > 0:
                threshold_ns = int(expected_sec * 1.5 * 1_000_000_000)
                ts_int = resampled_df.index.astype("int64")
                gaps = np.diff(ts_int, prepend=ts_int[0])
                # 先頭バーは便宜上 disc=False (前バーがないので連続扱い)
                gaps[0] = 0
                resampled_df["disc"] = gaps > threshold_ns
            else:
                resampled_df["disc"] = False

            if len(resampled_df) < 2:
                return []

            # 3. 確定したバーのみを抽出 (形成中 = 最後の行 を除外)
            newly_closed_bars = resampled_df.iloc[:-1]
            new_bars = newly_closed_bars[newly_closed_bars.index > last_known_timestamp]

            if new_bars.empty:
                return []

            new_bar_timestamps = []

            # 4. 新しいバーをバッファに追加
            for timestamp, row in new_bars.iterrows():
                bar_df = pd.DataFrame(row).T
                bar_df.index = [timestamp]
                bar_df.index.name = "timestamp"

                self._append_bar_to_buffer(tf_name, bar_df, market_proxy_cache)
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

            # 1. M0.5バッファに新しいバーを追加
            # m05_dataframe はリサンプリング起点として使用
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

            # 3. M3確定時のみ全時間足を強制再計算・シグナルチェック
            # M3非確定時はSTEP1・2のバッファ更新のみで完結（処理なし）
            if "M3" not in newly_closed_timeframes:
                return signal_list

            m3_timestamp = newly_closed_timeframes["M3"][-1] + pd.Timedelta(minutes=3)

            # M3確定時：全時間足のバッファから強制再計算（学習側と一致）
            for tf_name in self.ALL_TIMEFRAMES.keys():
                if not self.is_buffer_filled.get(tf_name, False):
                    continue
                try:
                    data = {
                        col: np.array(self.data_buffers[tf_name][col], dtype=np.float64)
                        for col in self.OHLCV_COLS
                    }
                    base_features = self._calculate_base_features(data, tf_name)

                    self._update_incremental_ols(
                        tf_name, base_features, market_proxy_cache, m3_timestamp
                    )

                    neutralized = self._calculate_neutralized_features(
                        base_features, tf_name, m3_timestamp, market_proxy_cache
                    )
                    self.latest_features_cache[tf_name] = neutralized
                except Exception as e:
                    self.logger.warning(f"{tf_name} 特徴量キャッシュ更新失敗: {e}")

            # シグナルチェックはM3のみ
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
                "high": np.array(self.data_buffers[tf_name]["high"], dtype=np.float64),
                "low": np.array(self.data_buffers[tf_name]["low"], dtype=np.float64),
                "close": np.array(
                    self.data_buffers[tf_name]["close"], dtype=np.float64
                ),
            }

            # [乖離③修正] 学習側(create_proxy_labels)と計算方式を統一:
            #   学習側: TR.ewm_mean(alpha=1/ATR_PERIOD, adjust=False)  ← Wilder EWM
            #   旧本番: np.mean(tr[-13:])  ← 単純平均
            #   新本番: calculate_atr_wilder()  ← core_indicatorsのWilder EWM実装で統一
            # ATR Ratioのbaselineも学習側に合わせ「ATR(EWM)のrolling mean」を使用
            high, low, close = data["high"], data["low"], data["close"]
            if len(close) > 1:
                # Wilder EWM ATR（学習側と同一）
                atr_arr = calculate_atr_wilder(
                    high.astype(np.float64),
                    low.astype(np.float64),
                    close.astype(np.float64),
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

            # atr_ratio を latest_features_cache に書き込む
            # → calculate_feature_vector が atr_ratio_M3 を処理する際に
            #   latest_features_cache[tf_name].get("atr_ratio", 0.0) で正しい値を取得できる
            # 学習側（S6）と同じ計算式・ベースライン期間なので純化不要
            if tf_name in self.latest_features_cache:
                self.latest_features_cache[tf_name]["atr_ratio"] = atr_ratio

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
                    f"  -> V5 Signal Check ({tf_name} @ {timestamp.strftime('%H:%M')}): "
                    f"PASSED (ATR Ratio: {atr_ratio:.3f} >= {atr_threshold:.3f})"
                )
                return {"is_V5": True, "market_info": market_info}
            else:
                # ▼▼▼ 追加: ATR不足で見送った時のログ ▼▼▼
                self.logger.info(
                    f"  -> V5 Signal Check ({tf_name} @ {timestamp.strftime('%H:%M')}): "
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

        OLS_WINDOW = 2016

        try:
            search_ts = timestamp
            if search_ts.tzinfo is None:
                search_ts = search_ts.replace(tzinfo=timezone.utc)
            else:
                search_ts = search_ts.astimezone(timezone.utc)

            # 重複除去・ソートしてからffill検索
            proxy_cache_unique = market_proxy_cache[
                ~market_proxy_cache.index.duplicated(keep="last")
            ].sort_index()
            idx = proxy_cache_unique.index.get_indexer([search_ts], method="ffill")[0]
            latest_x = (
                float(proxy_cache_unique.iloc[idx]["market_proxy"])
                if idx != -1
                else 0.0
            )
            if not np.isfinite(latest_x):
                latest_x = 0.0

            # バッファ・状態の初期化
            if tf_name not in self.proxy_feature_buffers:
                self.proxy_feature_buffers[tf_name] = {
                    "market_proxy": deque(maxlen=OLS_WINDOW)
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
            2. neutralize_ols(y_arr, x_arr, window=2016, min_periods=30) を呼び出し
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
                val = y_latest - (beta * x_latest + alpha)

                if np.isfinite(val):
                    neutralized_features[base_name] = val
                else:
                    latest_y_safe = latest_y if np.isfinite(latest_y) else 0.0
                    neutralized_features[base_name] = latest_y_safe
            return neutralized_features

        except Exception as e:
            self.logger.error(f"アルファ純化 ({tf_name}) に失敗: {e}", exc_info=True)
            return base_features_dict

    def _calculate_base_features(
        self, data: Dict[str, np.ndarray], tf_name: str
    ) -> Dict[str, float]:
        """
        【特徴量ルーター：完全カプセル化対応版】
        各モジュール（1A〜1F）のクラスメソッドにデータを渡し、
        完成した特徴量辞書を一括で受け取ってマージする。
        """
        features = {}

        # [乖離①修正] qa_stateとlookback_barsを時間足に合わせて渡す
        tf_qa = self.qa_states.get(tf_name, {})
        lb = TIMEFRAME_BARS_PER_DAY.get(tf_name, 1440)

        # 1. 各カテゴリのクラスにデータを渡し、完成した辞書を受け取って結合
        try:
            features.update(FeatureModule1A.calculate_features(data, lookback_bars=lb, qa_state=tf_qa.get("1A")))
            features.update(FeatureModule1B.calculate_features(data, lookback_bars=lb, qa_state=tf_qa.get("1B")))
            features.update(FeatureModule1C.calculate_features(data, lookback_bars=lb, qa_state=tf_qa.get("1C")))
            features.update(FeatureModule1D.calculate_features(data, lookback_bars=lb, qa_state=tf_qa.get("1D")))
            features.update(FeatureModule1E.calculate_features(data, lookback_bars=lb, qa_state=tf_qa.get("1E")))
            features.update(FeatureModule1F.calculate_features(data, lookback_bars=lb, qa_state=tf_qa.get("1F")))
        except Exception as e:
            self.logger.error(
                f"ベース特徴量の計算中にエラーが発生しました ({tf_name}): {e}",
                exc_info=True,
            )

        # 2. 純化用プロキシ (必須) の計算
        # [TRAIN-SERVE-FIX] 学習側 s1_1_C_enrich.py と完全一致させる:
        #   VOLATILITY_WINDOW = 20, VOLUME_WINDOW = 50, MOMENTUM_WINDOW = 5
        #   log_return         = np.log(close[t] / close[t-1])
        #   rolling_volatility = log_return.rolling(20, min_periods=1).std(ddof=1)
        #   price_momentum     = close[t] / close[t-5] - 1   (5本前比リターン)
        #   rolling_avg_volume = volume.rolling(50, min_periods=1).mean()
        #   volume_ratio       = volume / rolling_avg_volume
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
                f"✓ 特徴量エンジンの状態をスナップショット保存しました (Atomic): {filepath}"
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
            self.ols_state = state_data.get("ols_state", self.ols_state)
            self.qa_states = state_data.get("qa_states", self.qa_states)  # [乖離①修正]

            self.logger.info(
                f"✓ 特徴量エンジンの状態をスナップショットから復元しました: {filepath}"
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
