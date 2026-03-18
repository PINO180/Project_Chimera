# /workspace/execution/realtime_feature_engine.py
import sys
from pathlib import Path
from collections import deque
import numpy as np
import pandas as pd
import logging

# ▼▼▼ 追加: Numpyの無害な計算警告をミュートしてログをクリーンに保つ ▼▼▼
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
import blueprint as config

# --- プロジェクトのルートディレクトリをPythonの検索パスに追加 ---
project_root = Path(__file__).resolve().parents[1]
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

# ==================================================================
# 外部モジュール (完全カプセル化クラス群) のインポート
# 各モジュールは calculate_features(data) メソッドで一括計算を行います
# ==================================================================
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
    15時間足の独立したNumpyバッファを保持し、M1バーを起点とした
    全時間足の同期・リサンプリング・OLS純化・ベクトル生成を司る司令塔。
    特徴量計算そのものは外部のNumbaモジュール(1A〜1F)へ委譲する。
    """

    ALL_TIMEFRAMES = {
        "M1": 1,
        "M3": 3,
        "M5": 5,
        "M8": 8,
        "M15": 15,
        # "M30": 30,  # [V5] 110件リストに存在しないためスキップ
        "H1": 60,
        "H4": 240,
        "H6": 360,
        "H12": 720,
        "D1": 1440,
        # "W1": 10080, # [V5] 110件リストに存在しないためスキップ
        # "MN": 43200, # [V5] 110件リストに存在しないためスキップ
        # "tick": None, # [V5] 110件リストに存在しないためスキップ
        # "M0.5": None, # [V5] 110件リストに存在しないためスキップ
    }

    TF_RESAMPLE_RULES = {
        "M3": "3min",
        "M5": "5min",
        "M8": "8min",
        "M15": "15min",
        # "M30": "30min", # [V5] スキップ
        "H1": "1h",
        "H4": "4h",
        "H6": "6h",
        "H12": "12h",
        "D1": "1D",
        # "W1": "1W",     # [V5] スキップ
        # "MN": "1MS",    # [V5] スキップ
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

        # 1. 特徴量名簿をロード (304個の精鋭リスト)
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
            self.is_buffer_filled[tf_name] = False
            self.last_bar_timestamps[tf_name] = None
            self.latest_features_cache[tf_name] = {}

        # 4. M1データを保持するDeque (リサンプリング元)
        max_lookback_val = (
            max(self.lookbacks_by_tf.values()) if self.lookbacks_by_tf else 1000
        )
        max_m1_bars_needed = max_lookback_val * 1440 + 1000
        self.m1_dataframe: deque[Dict[str, Any]] = deque(maxlen=max_m1_bars_needed)

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
            for feat in PROXY_FEATURES:
                self.ols_state[tf_name][feat] = {
                    "sum_x": 0.0,
                    "sum_y": 0.0,
                    "sum_xy": 0.0,
                    "sum_x_sq": 0.0,
                    "sum_y_sq": 0.0,
                    "count": 0.0,
                }

        self.logger.info(f"M1 Dequeバッファを初期化 (maxlen: {max_m1_bars_needed})")

        # 6. JITコンパイルのウォームアップ
        self._warmup_jit()

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
            "M1": "1min",
            "M3": "3min",
            "M5": "5min",
            "M8": "8min",
            "M15": "15min",
            "M30": "30min",
            "H1": "1h",
            "H4": "4h",
            "H6": "6h",
            "H12": "12h",
            "D1": "1D",
            "W1": "1W",
            "MN": "1MS",
        }
        freq = freq_map.get(tf_name, "1T")

        # Dequeの長さに応じてタイムスタンプインデックスを逆算
        timestamps = pd.date_range(
            end=last_ts, periods=len(self.data_buffers[tf_name]["close"]), freq=freq
        )
        df["timestamp"] = timestamps

        return df.set_index("timestamp")

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

        # プロキシがない場合（通常はここには来ない）
        if market_proxy_cache is None or market_proxy_cache.empty:
            self.logger.warning(
                f"  -> {tf_name:<3} OLSバックフィルスキップ (プロキシ未提供)"
            )
            for col in self.OHLCV_COLS:
                self.data_buffers[tf_name][col].extend(
                    df_slice_for_no_proxy[col].values
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
                f" 取得する M1 バー数を増やすことを検討してください。"
            )

        # --- Numpy配列として一括抽出 ---
        arr_open = df_for_processing["open"].values.astype(np.float64)
        arr_high = df_for_processing["high"].values.astype(np.float64)
        arr_low = df_for_processing["low"].values.astype(np.float64)
        arr_close = df_for_processing["close"].values.astype(np.float64)
        arr_vol = df_for_processing["volume"].values.astype(np.float64)
        timestamps = df_for_processing.index

        base_features: dict = {}

        for i in range(n_rows):
            # (1) OHLCVバッファに1行追加 (deque の maxlen=buffer_len が古い行を自動で押し出す)
            self.data_buffers[tf_name]["open"].append(arr_open[i])
            self.data_buffers[tf_name]["high"].append(arr_high[i])
            self.data_buffers[tf_name]["low"].append(arr_low[i])
            self.data_buffers[tf_name]["close"].append(arr_close[i])
            self.data_buffers[tf_name]["volume"].append(arr_vol[i])

            ts = timestamps[i]
            self.last_bar_timestamps[tf_name] = ts

            # (2) [修正の核心] 固定スライディングウィンドウでデータを切り出す。
            #
            # 旧実装: arr[:i+1]  ← 成長スライス（バグの根本原因）
            # 新実装: arr[window_start : i+1]  ← 固定ウィンドウ（リアルタイム推論と同一）
            #
            # 例 (buffer_len=2116):
            #   i=   0 → arr[0:1]      (1本)    <- 起動直後は仕方なく短いが…
            #   i=2115 → arr[0:2116]   (2116本) <- ここからフルウィンドウ
            #   i=2116 → arr[1:2117]   (2116本) <- スライドして常に2116本 ✓
            #   i=4131 → arr[2015:4132](2116本) <- 最終ステップ
            #
            # これにより VPT 等の累積特徴量が「フルウィンドウ分」のみ蓄積された
            # 値となり、OLS が学習する分布がリアルタイム推論と完全に一致する。
            window_start = max(0, i + 1 - buffer_len)
            data = {
                "open": arr_open[window_start : i + 1],
                "high": arr_high[window_start : i + 1],
                "low": arr_low[window_start : i + 1],
                "close": arr_close[window_start : i + 1],
                "volume": arr_vol[window_start : i + 1],
            }

            # 最低 30 本未満ではほぼすべての特徴量が NaN になるためスキップ
            if (i + 1 - window_start) < 30:
                continue

            base_features = self._calculate_base_features(data, tf_name)

            # ↓ 一時デバッグ用（M1のi=100のときだけ出力）
            if tf_name == "M1" and i == 100:
                self.logger.info(
                    f"DEBUG: base_features keys count = {len(base_features)}"
                )
                self.logger.info(
                    f"DEBUG: ols_state M1 keys = {list(self.ols_state.get('M1', {}).keys())[:5]}"
                )

            # (3) 固定ウィンドウ特徴量値で OLS 状態を更新
            self._update_incremental_ols(tf_name, base_features, market_proxy_cache, ts)

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

        # OLS の有効サンプル数を報告（十分な精度かを確認するため）
        ols_n = 0
        if tf_name in self.ols_state:
            # いずれかの特徴量の count を代表値として取得
            for feat_state in self.ols_state[tf_name].values():
                ols_n = int(feat_state.get("count", 0))
                break
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
        1. M1データのみを history_data_map から受け取る
        2. M1バッファを充填
        3. M1データから M3〜MN のすべてをリサンプリングして充填する
        """
        self.logger.info("全時間足の履歴データでNumpyバッファを一括充填中 (M1 Only)...")

        if "M1" not in history_data_map:
            raise ValueError("履歴データに M1 がありません。リサンプリングできません。")

        m1_history_pd = history_data_map["M1"]
        if "timestamp" not in m1_history_pd.columns:
            raise ValueError("M1履歴データに 'timestamp' カラムが見つかりません。")
        m1_history_pd = m1_history_pd.set_index("timestamp")

        self.logger.info(f"  -> M1 バッファをMT5データから充填中...")
        self._replace_buffer_from_dataframe("M1", m1_history_pd, market_proxy_cache)

        self.m1_dataframe.clear()
        m1_records = m1_history_pd.reset_index().to_dict("records")
        self.m1_dataframe.extend(m1_records)

        for tf_name, rule in self.TF_RESAMPLE_RULES.items():
            if tf_name not in self.data_buffers or tf_name == "M1":
                continue

            try:
                self.logger.info(f"  -> {tf_name:<3} をM1からリサンプリング中...")
                resampled_df = (
                    m1_history_pd.resample(rule)
                    .agg(
                        {
                            "open": "first",
                            "high": "max",
                            "low": "min",
                            "close": "last",
                            "volume": "sum",
                        }
                    )
                    .dropna()
                )

                if resampled_df.empty:
                    self.logger.warning(f"{tf_name} のリサンプリング結果が空です。")
                    continue

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
            self.last_bar_timestamps[tf_name] = bar_timestamp

            # ★ 2, 3 の古い限定的OLS更新処理を削除 (process_new_m1_bar内で全特徴量を一括更新するため)

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
        M1 DequeをDFに変換してリサンプリングし、新しいバーが生成されていたら
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
            new_m1_bars_for_resampling = []
            found_anchor = False
            for bar in reversed(self.m1_dataframe):
                bar_ts = bar["timestamp"]
                if bar_ts >= last_known_timestamp:
                    new_m1_bars_for_resampling.append(bar)
                else:
                    # 1本前のアンカーバーを追加してリサンプリングの境界を正確にする
                    if not found_anchor:
                        new_m1_bars_for_resampling.append(bar)
                        found_anchor = True
                    break

            new_m1_bars_for_resampling.reverse()

            if len(new_m1_bars_for_resampling) < 2:
                return []

            new_m1_data = pd.DataFrame(new_m1_bars_for_resampling).set_index(
                "timestamp"
            )

            # 2. 抽出したDFのみをリサンプリング
            resampled_df = (
                new_m1_data.resample(rule)
                .agg(
                    {
                        "open": "first",
                        "high": "max",
                        "low": "min",
                        "close": "last",
                        "volume": "sum",
                    }
                )
                .dropna()
            )

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

    def process_new_m1_bar(
        self, m1_bar: Dict[str, Any], market_proxy_cache: pd.DataFrame
    ) -> List[Signal]:
        """
        [メインループ] main.py から M1 バーを受け取り、全バッファを更新し、
        シグナルをチェックして返す。
        """
        signal_list: List[Signal] = []

        try:
            m1_timestamp = m1_bar["timestamp"]

            # 1. M1バッファに新しいバーを追加
            self.m1_dataframe.append(m1_bar)
            m1_bar_df = pd.DataFrame([m1_bar]).set_index("timestamp")
            self._append_bar_to_buffer("M1", m1_bar_df, market_proxy_cache)

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

            newly_closed_timeframes["M1"] = [m1_timestamp]

            # 3. 新しくバーが確定した各時間足について処理
            for tf_name, timestamps in newly_closed_timeframes.items():
                for timestamp in timestamps:
                    # 特徴量キャッシュの更新 (クロス参照対応)
                    if self.is_buffer_filled[tf_name]:
                        try:
                            data = {
                                col: np.array(
                                    self.data_buffers[tf_name][col], dtype=np.float64
                                )
                                for col in self.OHLCV_COLS
                            }
                            base_features = self._calculate_base_features(data, tf_name)

                            # ★ 追加: 算出された全特徴量のOLSバッファを動的更新
                            self._update_incremental_ols(
                                tf_name, base_features, market_proxy_cache, timestamp
                            )

                            neutralized = self._calculate_neutralized_features(
                                base_features, tf_name, timestamp, market_proxy_cache
                            )
                            self.latest_features_cache[tf_name] = neutralized
                        except Exception as e:
                            self.logger.warning(
                                f"{tf_name} 特徴量キャッシュ更新失敗: {e}"
                            )

                    # シグナルチェック (V5レジームフィルター)
                    V5_check_result = self._check_for_signal(tf_name, timestamp)

                    if V5_check_result["is_V5"]:
                        feature_vector = self.calculate_feature_vector(
                            tf_name, timestamp, market_proxy_cache
                        )

                        if feature_vector is not None:
                            # 修正: 上書きを防ぎ、モデルが期待するサフィックス付きの完全な辞書を構築
                            combined_features = dict(
                                zip(self.feature_list, feature_vector[0])
                            )

                            signal = Signal(
                                features=feature_vector,
                                timestamp=timestamp,
                                timeframe=tf_name,
                                market_info=V5_check_result["market_info"],
                                atr_value=V5_check_result["market_info"].get(
                                    "atr_value", 0.0
                                ),
                                close_price=V5_check_result["market_info"].get(
                                    "current_price", 0.0
                                ),
                                feature_dict=combined_features,
                            )
                            signal_list.append(signal)

            return signal_list

        except Exception as e:
            self.logger.error(f"process_new_m1_bar でエラー: {e}", exc_info=True)
            return []

    def _check_for_signal(self, tf_name: str, timestamp: datetime) -> Dict[str, Any]:
        """
        指定された時間足のバッファがV5レジーム (ATR比率条件) かを判定する。
        """
        # ▼▼▼ 修正: シグナル判定を Mixed (M1〜M15) に拡大 ▼▼▼
        ALLOWED_TIMEFRAMES = ["M1", "M3", "M5", "M8", "M15"]
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

            # --- 外部エンジン依存を排除し、Numpyでローカル計算 ---
            high, low, close = data["high"], data["low"], data["close"]
            if len(close) > 1:
                tr = np.maximum(
                    high[1:] - low[1:],
                    np.maximum(
                        np.abs(high[1:] - close[:-1]), np.abs(low[1:] - close[:-1])
                    ),
                )
                atr_value = (
                    float(np.mean(tr[-self.ATR_CALC_PERIOD :]))
                    if len(tr) >= self.ATR_CALC_PERIOD
                    else float(np.mean(tr))
                )
            else:
                atr_value = 0.0

            current_price = data["close"][-1]
            if np.isnan(atr_value):
                return {"is_V5": False, "reason": "atr_is_nan"}

            # ATR閾値フィルター (risk_config.json で管理)
            atr_threshold = self.risk_config.get("min_atr_threshold", 0.0)

            if atr_value > atr_threshold:
                # ▼▼▼ 修正: 古い unified キーの取得と payoff_ratio を廃止し、Long/Short 分割キーを取得 ▼▼▼
                market_info = {
                    "atr_value": atr_value,
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
                    "direction": None,  # Two-Brain推論前のため未確定とする
                }
                # ▲▲▲ ここまで修正 ▲▲▲

                self.logger.info(
                    f"  -> V5 Signal Check ({tf_name} @ {timestamp.strftime('%H:%M')}): "
                    f"PASSED (ATR: {atr_value:.2f} > {atr_threshold:.2f})"
                )
                return {"is_V5": True, "market_info": market_info}
            else:
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
        【V5完全修正版】 脆弱なWelford状態変数(sum_x, sum_x_sq等)の維持を廃止し、
        純粋にDeque（リングバッファ）へ最新値を追加するだけの処理に特化。
        """
        from datetime import timezone
        import numpy as np

        try:
            search_ts = timestamp
            if search_ts.tzinfo is None:
                search_ts = search_ts.replace(tzinfo=timezone.utc)
            else:
                search_ts = search_ts.astimezone(timezone.utc)

            idx = market_proxy_cache.index.get_indexer([search_ts], method="ffill")[0]

            if idx == -1:
                latest_x = 0.0
            else:
                latest_x = float(market_proxy_cache.iloc[idx]["market_proxy"])

            if not np.isfinite(latest_x):
                latest_x = 0.0

            # リングバッファの初期化チェック
            if tf_name not in self.proxy_feature_buffers:
                buf_len = self.lookbacks_by_tf.get(tf_name, 2016)
                self.proxy_feature_buffers[tf_name] = {
                    "market_proxy": deque(maxlen=buf_len)
                }

            x_deque = self.proxy_feature_buffers[tf_name]["market_proxy"]

            # 各特徴量の最新値をバッファへ追加
            for feat_name, latest_y in latest_proxy_features.items():
                if not np.isfinite(latest_y):
                    latest_y = 0.0

                if feat_name not in self.proxy_feature_buffers[tf_name]:
                    buf_len = self.lookbacks_by_tf.get(tf_name, 2016)
                    self.proxy_feature_buffers[tf_name][feat_name] = deque(
                        maxlen=buf_len
                    )

                self.proxy_feature_buffers[tf_name][feat_name].append(latest_y)

            # 市場プロキシの最新値をバッファへ追加
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
        【V5完全修正版】 Welfordの逐次減算による情報落ち(Catastrophic Cancellation)を排除し、
        毎ティックNumpyのC言語バックエンドで2016期間の分散・共分散をO(N)でフル計算する。
        """
        import numpy as np

        neutralized_features: Dict[str, float] = {}

        try:
            latest_x_deque = self.proxy_feature_buffers.get(tf_name, {}).get(
                "market_proxy"
            )
            if not latest_x_deque:
                return base_features_dict

            # 過去最大2016件を抽出 (Polarsの rolling_window=2016 に完全一致させる)
            x_arr = np.array(latest_x_deque, dtype=np.float64)[-2016:]
            n = len(x_arr)

            # Polarsの min_periods=30 に準拠
            if n < 30:
                return base_features_dict

            # Numpyベクトル演算で一括計算 (過去の蓄積値に依存しないためドリフトは物理的に発生しない)
            mean_x = np.mean(x_arr)
            # Polarsの var_x = mean(x^2) - mean(x)^2 と数学的に完全一致
            var_x = np.maximum(0.0, np.mean(x_arr**2) - mean_x**2)
            latest_x = x_arr[-1]

            for base_name, latest_y in base_features_dict.items():
                y_deque = self.proxy_feature_buffers[tf_name].get(base_name)
                if not y_deque:
                    neutralized_features[base_name] = latest_y
                    continue

                y_arr = np.array(y_deque, dtype=np.float64)[-2016:]
                # XとYのデータ数が一致しない場合のフェイルセーフ
                if len(y_arr) != n:
                    neutralized_features[base_name] = latest_y
                    continue

                mean_y = np.mean(y_arr)

                # Polarsの cov_xy = mean(x*y) - mean(x)*mean(y) と数学的に完全一致
                cov_xy = np.mean(x_arr * y_arr) - (mean_x * mean_y)

                # 学習時と全く同じ純化式 (ゼロ除算防止の 1e-10 も同一)
                beta = cov_xy / (var_x + 1e-10)
                alpha = mean_y - beta * mean_x

                latest_y_safe = latest_y if np.isfinite(latest_y) else 0.0
                neutralized_features[base_name] = latest_y_safe - (
                    beta * latest_x + alpha
                )

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

        # 1. 各カテゴリのクラスにデータを渡し、完成した辞書を受け取って結合
        try:
            features.update(FeatureModule1A.calculate_features(data))
            features.update(FeatureModule1B.calculate_features(data))
            features.update(FeatureModule1C.calculate_features(data))
            features.update(FeatureModule1D.calculate_features(data))
            features.update(FeatureModule1E.calculate_features(data))
            features.update(FeatureModule1F.calculate_features(data))
        except Exception as e:
            self.logger.error(
                f"ベース特徴量の計算中にエラーが発生しました ({tf_name}): {e}",
                exc_info=True,
            )

        # 2. 純化用プロキシ (必須) の計算
        # 司令塔側で最低限必要なプロキシ値を計算して追加する
        def _window(arr: np.ndarray, window: int) -> np.ndarray:
            return arr[-window:] if len(arr) >= window else arr

        def _pct(arr: np.ndarray) -> np.ndarray:
            if len(arr) < 2:
                return np.full_like(arr, np.nan)
            arr_safe = arr[:-1].copy()
            arr_safe[arr_safe == 0] = 1e-10
            return np.concatenate(([np.nan], np.diff(arr) / arr_safe))

        close_pct = _pct(data["close"])

        # atr のフォールバック処理 (1Cモジュールから取得できている場合はそれを使用)
        if "e1c_atr_13" in features:
            features["atr"] = features["e1c_atr_13"]
        else:
            high, low, close = data["high"], data["low"], data["close"]
            if len(close) > 1:
                tr = np.maximum(
                    high[1:] - low[1:],
                    np.maximum(
                        np.abs(high[1:] - close[:-1]), np.abs(low[1:] - close[:-1])
                    ),
                )
                features["atr"] = (
                    np.mean(_window(tr, 13)) if len(tr) >= 13 else np.mean(tr)
                )
            else:
                features["atr"] = 0.0

        features["log_return"] = (
            np.log((data["close"][-1] + 1e-10) / (data["close"][-2] + 1e-10))
            if len(data["close"]) > 1
            else 0.0
        )
        features["price_momentum"] = (
            data["close"][-1] - data["close"][-11]
            if len(data["close"]) > 10
            else np.nan
        )
        features["rolling_volatility"] = (
            np.std(_window(close_pct, 20)) if len(close_pct) >= 20 else np.nan
        )
        features["volume_ratio"] = (
            data["volume"][-1] / (np.mean(_window(data["volume"], 20)) + 1e-10)
            if len(data["volume"]) > 0
            else np.nan
        )

        return features

    def calculate_feature_vector(
        self, tf_name: str, timestamp: datetime, market_proxy_cache: pd.DataFrame
    ) -> Optional[np.ndarray]:
        """
        [ベクトル生成] 304個の精鋭リストに厳密準拠したベクトルを構築。
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

                base_name = (
                    feat_name.split("_neutralized_")[0]
                    if "_neutralized_" in feat_name
                    else feat_name.rsplit("_", 1)[0]
                )
                val = self.latest_features_cache[target_tf].get(base_name, 0.0)
                vector.append(val)

            final_vector = np.nan_to_num(
                np.array(vector, dtype=np.float64), nan=0.0, posinf=0.0, neginf=0.0
            )
            # ▼修正: 10万でのクリッピングを撤廃 (VPTなどの自然な巨大値が切り捨てられOODになるのを防ぐ)
            return np.array([final_vector])
        except Exception as e:
            self.logger.error(f"Vector calculation error: {e}")
            return None

    # ▼▼▼ 追加: スナップショット（Pickle）保存とロード機能 ▼▼▼
    def save_state(self, filepath: str) -> bool:
        """現在の特徴量バッファとOLS状態を丸ごとファイルに保存する"""
        try:
            state_data = {
                "data_buffers": self.data_buffers,
                "is_buffer_filled": self.is_buffer_filled,
                "last_bar_timestamps": self.last_bar_timestamps,
                "latest_features_cache": self.latest_features_cache,
                "m1_dataframe": self.m1_dataframe,
                "proxy_feature_buffers": self.proxy_feature_buffers,
                "ols_state": self.ols_state,
            }
            with open(filepath, "wb") as f:
                pickle.dump(state_data, f)
            self.logger.info(
                f"✓ 特徴量エンジンの状態をスナップショット保存しました: {filepath}"
            )
            return True
        except Exception as e:
            self.logger.error(f"✗ 特徴量エンジンの状態保存に失敗: {e}", exc_info=True)
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
            self.m1_dataframe = state_data["m1_dataframe"]

            # 後方互換性のため get() を使用
            self.proxy_feature_buffers = state_data.get(
                "proxy_feature_buffers", self.proxy_feature_buffers
            )
            self.ols_state = state_data.get("ols_state", self.ols_state)

            self.logger.info(
                f"✓ 特徴量エンジンの状態をスナップショットから復元しました: {filepath}"
            )
            return True
        except Exception as e:
            self.logger.error(f"✗ 特徴量エンジンの状態復元に失敗: {e}", exc_info=True)
            return False

    # ▲▲▲ ここまで追加 ▲▲▲
