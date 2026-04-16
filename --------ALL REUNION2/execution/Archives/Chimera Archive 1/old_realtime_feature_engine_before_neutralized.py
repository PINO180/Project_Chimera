# /workspace/execution/realtime_feature_engine.py
import sys
from pathlib import Path
from collections import deque
import numpy as np
import pandas as pd
import logging
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Optional, Any
import json
import re
import blueprint as config

# --- プロジェクトのルートディレクトリをPythonの検索パスに追加 ---
project_root = Path(__file__).resolve().parents[1]
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

# ==================================================================
# 外部モジュール (Numba UDF群) のインポート
# モノリスファイルから切り出された計算エンジン群に処理を委譲します
# ==================================================================
import realtime_feature_engine_1A_statistics as engine_1A
import realtime_feature_engine_1B_timeseries as engine_1B
import realtime_feature_engine_1C_technical as engine_1C
import realtime_feature_engine_1D_volume as engine_1D
import realtime_feature_engine_1E_signal as engine_1E
import realtime_feature_engine_1F_experimental as engine_1F


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
        "M30": 30,
        "H1": 60,
        "H4": 240,
        "H6": 360,
        "H12": 720,
        "D1": 1440,
        "W1": 10080,
        "MN": 43200,
        "tick": None,
        "M0.5": None,
    }

    TF_RESAMPLE_RULES = {
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

    OHLCV_COLS = ["open", "high", "low", "close", "volume"]
    DEFAULT_LOOKBACK = 200
    ATR_CALC_PERIOD = 21

    def __init__(
        self,
        feature_list_path: str = str(
            config.S3_SELECTED_FEATURES_PURIFIED_DIR / "final_feature_set_v3.txt"
        ),
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
                    f"  -> {tf_name} バッファ初期化 (Default: {self.DEFAULT_LOOKBACK})"
                )
            else:
                self.logger.info(
                    f"  -> {tf_name} バッファ初期化 (Lookback: {self.lookbacks_by_tf[tf_name]})"
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
            lookback = self.data_buffers[tf_name]["close"].maxlen

            self.proxy_feature_buffers[tf_name] = {
                feat: deque(maxlen=lookback) for feat in PROXY_FEATURES
            }
            self.proxy_feature_buffers[tf_name]["market_proxy"] = deque(maxlen=lookback)

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
        """外部モジュールのJITコンパイルを起動時に済ませる"""
        self.logger.info("外部NumbaモジュールのJITウォームアップを開始します...")
        try:
            dummy_data = np.cumsum(np.random.randn(100)).astype(np.float64)
            # 各エンジンの軽量な関数を叩いてコンパイルを誘発
            _ = engine_1A.rolling_skew_numba(dummy_data)
            _ = engine_1B.t分布_自由度_udf(dummy_data)
            _ = engine_1C.rolling_mean_numba(dummy_data, 10)
            _ = engine_1C.calculate_schaff_trend_cycle_numba(dummy_data, 12, 26, 9)
            _ = engine_1D.hv_robust_udf(dummy_data)
            _ = engine_1E.spectral_centroid_udf(dummy_data)
            _ = engine_1F.rolling_linguistic_complexity_udf(dummy_data)
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
        SAFE_MIN_LOOKBACK = 1000

        final_lookbacks = {}
        for tf_name_parsed in lookbacks.keys():
            req_size = max(lookbacks[tf_name_parsed], SAFE_MIN_LOOKBACK)
            final_lookbacks[tf_name_parsed] = req_size + 100
            self.logger.info(
                f"  -> {tf_name_parsed} 最大ルックバック: {final_lookbacks[tf_name_parsed]}"
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
        DataFrameからバッファを一括充填する。
        - OHLCVバッファ
        - 純化用特徴量バッファ (OLS用)
        - OLS状態
        """
        if tf_name not in self.data_buffers:
            self.logger.warning(f"_replace_buffer: {tf_name} は管理対象外です。")
            return

        buffer_len = self.lookbacks_by_tf[tf_name]
        df_slice = df.iloc[-buffer_len:]

        # 1. OHLCVバッファを充填
        for col in self.OHLCV_COLS:
            self.data_buffers[tf_name][col].clear()
            self.data_buffers[tf_name][col].extend(df_slice[col].values)
        self.last_bar_timestamps[tf_name] = df_slice.index[-1]

        # データが少しでもあれば計算許可を出す
        if len(df_slice) > 0:
            self.is_buffer_filled[tf_name] = True
            if len(df_slice) < buffer_len:
                self.logger.warning(
                    f"  ⚠️ {tf_name} はデータ不足 ({len(df_slice)}/{buffer_len}) ですが、計算を許可します。"
                )

        self.logger.info(
            f"  -> {tf_name} OHLCVバッファを {len(df_slice)} 行で充填しました。"
        )

        # 2. 純化用バッファとOLS状態をバックフィル
        if market_proxy_cache is None or market_proxy_cache.empty:
            self.logger.warning(
                f"  -> {tf_name} OLSバックフィルスキップ (プロキシ未提供)"
            )
            return

        # 2a. 純化対象5特徴量の「全履歴」を計算 (起動時に1回だけ実行)
        close_arr = df_slice["close"].to_numpy(dtype=np.float64)
        high_arr = df_slice["high"].to_numpy(dtype=np.float64)
        low_arr = df_slice["low"].to_numpy(dtype=np.float64)
        volume_arr = df_slice["volume"].to_numpy(dtype=np.float64)

        proxy_feat_df = pd.DataFrame(index=df_slice.index)

        # --- 外部エンジン(1C) への委譲 ---
        proxy_feat_df["atr"] = engine_1C.calculate_atr_numba(
            high_arr, low_arr, close_arr, 13
        )

        pct = np.full_like(close_arr, np.nan)
        if len(close_arr) >= 2:
            safe_denominator_pct = close_arr[:-1].copy()
            safe_denominator_pct[safe_denominator_pct == 0] = 1e-10
            pct_calc = np.diff(close_arr) / safe_denominator_pct
            pct[1:] = pct_calc
        close_pct = pct

        proxy_feat_df["log_return"] = np.concatenate(
            ([np.nan], np.log(close_arr[1:] / (close_arr[:-1] + 1e-10)))
        )
        proxy_feat_df["price_momentum"] = df_slice["close"].diff(10).to_numpy()
        proxy_feat_df["rolling_volatility"] = (
            pd.Series(close_pct).rolling(window=20).std().to_numpy()
        )
        proxy_feat_df["volume_ratio"] = volume_arr / (
            pd.Series(volume_arr).rolling(window=20).mean().to_numpy() + 1e-10
        )

        # 2b. 市場プロキシ (x) をAsof-Join
        # [FIX-8] インデックスのタイムゾーンをUTCに統一してJoinエラーを防ぐ
        if proxy_feat_df.index.tz is None:
            proxy_feat_df.index = proxy_feat_df.index.tz_localize("UTC")
        else:
            proxy_feat_df.index = proxy_feat_df.index.tz_convert("UTC")

        aligned_df = proxy_feat_df.join(market_proxy_cache, how="left").ffill()
        aligned_df = aligned_df.fillna(0.0)

        # 2c. バッファとOLS状態を充填
        for feat_name in self.proxy_feature_buffers[tf_name].keys():
            if feat_name == "market_proxy":
                continue

            y_history = aligned_df[feat_name].to_numpy()
            x_history = aligned_df["market_proxy"].to_numpy()

            self.proxy_feature_buffers[tf_name][feat_name].clear()
            self.proxy_feature_buffers[tf_name][feat_name].extend(y_history)
            if feat_name == "atr":
                self.proxy_feature_buffers[tf_name]["market_proxy"].clear()
                self.proxy_feature_buffers[tf_name]["market_proxy"].extend(x_history)

            state = self.ols_state[tf_name][feat_name]
            state["sum_x"] = np.sum(x_history)
            state["sum_y"] = np.sum(y_history)
            state["sum_xy"] = np.sum(x_history * y_history)
            state["sum_x_sq"] = np.sum(x_history**2)
            state["sum_y_sq"] = np.sum(y_history**2)
            state["count"] = float(len(x_history))

        self.logger.info(f"  -> {tf_name} 純化用バッファとOLS状態を充填しました。")

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

        self.logger.info(
            "M1データから M3〜MN の全バッファをリサンプリングして充填中..."
        )
        for tf_name, rule in self.TF_RESAMPLE_RULES.items():
            if tf_name not in self.data_buffers or tf_name == "M1":
                continue

            try:
                self.logger.info(f"  -> {tf_name} をM1からリサンプリング中...")
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

            # 2. 純化用5特徴量の「最新値」を計算
            latest_proxy_features = self._calculate_proxy_features_incremental(
                tf_name, bar_df
            )

            # 3. OLS状態を逐次更新
            if latest_proxy_features:
                self._update_incremental_ols(
                    tf_name,
                    latest_proxy_features,
                    market_proxy_cache,
                    bar_timestamp,
                )

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

            # ▼▼▼▼▼▼ 【ここから追加】最強Tick特徴量のキャッシュ直接登録 ▼▼▼▼▼▼
            if "tick" not in self.latest_features_cache:
                self.latest_features_cache["tick"] = {}

            # 生の値を取得し、学習時と同じベース名で登録（無相関化はベクトル抽出時に自動適用）
            raw_tick_vol = m1_bar.get("tick_volume_mean_5", 0.0)
            self.latest_features_cache["tick"]["e1a_fast_volume_mean_5"] = raw_tick_vol
            # ▲▲▲▲▲▲ 【追加ここまで】 ▲▲▲▲▲▲

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
                            # [FIX-3] feature_dict を latest_features_cache から取得して Signal に詰める
                            combined_features: Dict[str, float] = {}
                            for _tf, _cache in self.latest_features_cache.items():
                                combined_features.update(_cache)
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
        # シグナル発生を許可する下位足に限定
        ALLOWED_TIMEFRAMES = ["M1", "M3", "M5", "M8", "M15", "H1"]
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

            # --- 外部エンジン(1C) への委譲 ---
            atr_21_arr = engine_1C.calculate_atr_numba(
                data["high"], data["low"], data["close"], self.ATR_CALC_PERIOD
            )
            atr_value = atr_21_arr[-1]
            current_price = data["close"][-1]

            if np.isnan(atr_value):
                return {"is_V5": False, "reason": "atr_is_nan"}

            # ATR閾値フィルター (risk_config.json で管理)
            atr_threshold = self.risk_config.get("min_atr_threshold", 0.0)

            if atr_value > atr_threshold:
                pt_mult = self.risk_config.get("pt_multiplier", 1.0)
                sl_mult = self.risk_config.get("sl_multiplier", 5.0)

                market_info = {
                    "atr_value": atr_value,
                    "current_price": current_price,
                    "pt_multiplier": pt_mult,
                    "sl_multiplier": sl_mult,
                    "payoff_ratio": pt_mult / sl_mult if sl_mult > 0 else 0.2,
                    "direction": None,  # Two-Brain推論前のため未確定とする
                }

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

    def _calculate_proxy_features_incremental(
        self, tf_name: str, ohlcv_df: pd.DataFrame
    ) -> Dict[str, float]:
        """
        純化対象の5特徴量（プロキシ）の「最新値」のみを計算する。
        """
        if ohlcv_df.empty:
            return {}

        ohlcv_buffer = self.data_buffers[tf_name]

        close_arr = np.array(ohlcv_buffer["close"], dtype=np.float64)
        high_arr = np.array(ohlcv_buffer["high"], dtype=np.float64)
        low_arr = np.array(ohlcv_buffer["low"], dtype=np.float64)
        volume_arr = np.array(ohlcv_buffer["volume"], dtype=np.float64)

        if len(close_arr) < 2:
            return {}

        latest_features = {}

        # 1. atr (13)
        # --- 外部エンジン(1C) への委譲 ---
        atr_13_arr = engine_1C.calculate_atr_numba(high_arr, low_arr, close_arr, 13)
        latest_features["atr"] = atr_13_arr[-1] if len(atr_13_arr) > 0 else np.nan

        # 2. log_return
        safe_close_prev = close_arr[-2]
        if safe_close_prev == 0:
            safe_close_prev = 1e-10
        latest_features["log_return"] = np.log(close_arr[-1] / safe_close_prev)

        # 3. price_momentum (10)
        if len(close_arr) > 10:
            latest_features["price_momentum"] = close_arr[-1] - close_arr[-11]
        else:
            latest_features["price_momentum"] = np.nan

        # 4. rolling_volatility (20)
        if len(close_arr) > 20:
            safe_denominator_pct = close_arr[-21:-1]
            safe_denominator_pct[safe_denominator_pct == 0] = 1e-10
            pct_calc = np.diff(close_arr[-21:]) / safe_denominator_pct
            latest_features["rolling_volatility"] = np.std(pct_calc)
        else:
            latest_features["rolling_volatility"] = np.nan

        # 5. volume_ratio (20)
        if len(volume_arr) > 20:
            vol_mean_20 = np.mean(volume_arr[-20:])
            latest_features["volume_ratio"] = volume_arr[-1] / (vol_mean_20 + 1e-10)
        else:
            latest_features["volume_ratio"] = np.nan

        return latest_features

    def _update_incremental_ols(
        self,
        tf_name: str,
        latest_proxy_features: Dict[str, float],
        market_proxy_cache: pd.DataFrame,
        timestamp: datetime,
    ):
        """
        純化対象の5特徴量について、OLS状態(sum_x, sum_y...)を逐次更新する。
        """
        from datetime import timezone

        try:
            # 1. 最新の市場プロキシ値 (x) を取得 (Asofライク)
            # [FIX-8] timestampのタイムゾーンをUTCに統一してインデクサーエラーを防ぐ
            search_ts = timestamp
            if search_ts.tzinfo is None:
                search_ts = search_ts.replace(tzinfo=timezone.utc)
            else:
                search_ts = search_ts.astimezone(timezone.utc)

            idx = market_proxy_cache.index.get_indexer([search_ts], method="ffill")[0]
            latest_x = market_proxy_cache.iloc[idx]["market_proxy"]
            if np.isnan(latest_x):
                latest_x = 0.0

            for feat_name, latest_y in latest_proxy_features.items():
                if np.isnan(latest_y):
                    latest_y = 0.0

                state = self.ols_state[tf_name][feat_name]
                buffer_len = self.lookbacks_by_tf[tf_name]

                # Welford's online algorithm (バッファ満杯なら古い値を抜く)
                if state["count"] >= buffer_len:
                    old_x_deque = self.proxy_feature_buffers[tf_name]["market_proxy"]
                    old_y_deque = self.proxy_feature_buffers[tf_name][feat_name]
                    if old_x_deque and old_y_deque:
                        old_x = old_x_deque[0]
                        old_y = old_y_deque[0]
                        state["sum_x"] -= old_x
                        state["sum_y"] -= old_y
                        state["sum_xy"] -= old_x * old_y
                        state["sum_x_sq"] -= old_x**2
                        state["sum_y_sq"] -= old_y**2
                        state["count"] -= 1.0

                state["sum_x"] += latest_x
                state["sum_y"] += latest_y
                state["sum_xy"] += latest_x * latest_y
                state["sum_x_sq"] += latest_x**2
                state["sum_y_sq"] += latest_y**2
                state["count"] += 1.0

                # 最新値を保存
                self.proxy_feature_buffers[tf_name][feat_name].append(latest_y)
                if feat_name == "atr":
                    self.proxy_feature_buffers[tf_name]["market_proxy"].append(latest_x)

        except Exception as e:
            feat_name_safe = locals().get("feat_name", "<unknown>")
            self.logger.warning(
                f"[{tf_name}] 逐次OLSの更新に失敗 ({feat_name_safe}): {e}",
                exc_info=False,
            )

    def _calculate_neutralized_features(
        self,
        base_features_dict: Dict[str, float],
        tf_name: str,
        signal_timestamp: datetime,  # noqa: ARG002 – kept for API compatibility
        market_proxy_cache_df: pd.DataFrame,  # noqa: ARG002 – kept for API compatibility
    ) -> Dict[str, float]:
        """
        逐次計算されたOLS状態を使って、瞬時に残差(純化済み特徴量)を計算する。
        """
        neutralized_features: Dict[str, float] = {}

        PROXY_FEATURES = [
            "atr",
            "log_return",
            "price_momentum",
            "rolling_volatility",
            "volume_ratio",
        ]

        try:
            latest_x_deque = self.proxy_feature_buffers[tf_name]["market_proxy"]
            latest_x = latest_x_deque[-1] if latest_x_deque else 0.0

            for base_name, latest_y in base_features_dict.items():
                if base_name not in PROXY_FEATURES:
                    neutralized_features[base_name] = latest_y
                    continue

                state = self.ols_state[tf_name][base_name]
                n = state["count"]

                if n < 20:
                    neutralized_features[base_name] = latest_y
                    continue

                mean_x = state["sum_x"] / n
                mean_y = state["sum_y"] / n
                cov_xy = (state["sum_xy"] / n) - (mean_x * mean_y)
                var_x = (state["sum_x_sq"] / n) - (mean_x**2)

                beta = cov_xy / var_x if var_x >= 1e-10 else 0.0
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
        【特徴量ルーター：最適化版】
        final_feature_set_v3.txt (304個) に必要な特徴量のみを計算。
        不要なループやパラメータを排除し、外部モジュール(1A〜1C)へ委譲。
        """
        features = {}

        # --- 高速化用内部ヘルパー ---
        def _window(arr: np.ndarray, window: int) -> np.ndarray:
            return arr[-window:] if len(arr) >= window else arr

        def _array(arr: np.ndarray) -> np.ndarray:
            return arr

        def _last(arr: np.ndarray) -> float:
            return arr[-1] if len(arr) > 0 else np.nan

        def _pct(arr: np.ndarray) -> np.ndarray:
            if len(arr) < 2:
                return np.full_like(arr, np.nan)
            arr_safe = arr[:-1].copy()
            arr_safe[arr_safe == 0] = 1e-10
            return np.concatenate(([np.nan], np.diff(arr) / arr_safe))

        def _ema(arr: np.ndarray, span: int) -> np.ndarray:
            alpha = 2.0 / (span + 1.0)
            ema = np.zeros_like(arr, dtype=np.float64)
            if len(arr) == 0:
                return ema
            ema[0] = arr[0]
            for i in range(1, len(arr)):
                ema[i] = alpha * arr[i] + (1 - alpha) * ema[i - 1]
            return ema

        close_pct = _pct(data["close"])

        # =======================================================
        # Engine 1A: 統計特徴量 (V3リスト準拠)
        # =======================================================
        features["e1a_anderson_darling_statistic_30"] = (
            engine_1A.anderson_darling_numba(_window(data["close"], 30))
        )
        features["e1a_fast_basic_stabilization"] = _last(
            engine_1A.basic_stabilization_numba(_window(data["close"], 100))
        )

        # 移動平均・標準偏差 (リストにある期間のみ)
        for w in [5, 10, 50]:
            features[f"e1a_fast_rolling_mean_{w}"] = np.mean(_window(data["close"], w))
        for w in [5, 10, 20, 100]:
            features[f"e1a_fast_rolling_std_{w}"] = np.std(_window(data["close"], w))
        for w in [5, 10, 20, 50]:
            features[f"e1a_fast_volume_mean_{w}"] = np.mean(_window(data["volume"], w))

        features["e1a_jarque_bera_statistic_50"] = (
            engine_1A.jarque_bera_statistic_numba(_window(data["close"], 50))
        )

        # ロバスト統計
        for w in [10, 20, 50]:
            q75, q25 = np.percentile(_window(data["close"], w), [75, 25])
            features[f"e1a_robust_iqr_{w}"] = q75 - q25
            if w == 50:
                features["e1a_robust_q75_50"] = q75

        features["e1a_robust_mad_20"] = engine_1A.mad_rolling_numba(
            _window(data["close"], 20)
        )
        features["e1a_robust_median_50"] = np.median(_window(data["close"], 50))
        features["e1a_runs_test_statistic_30"] = engine_1A.runs_test_numba(
            _window(data["close"], 30)
        )

        for w in [10, 20, 50]:
            features[f"e1a_statistical_cv_{w}"] = np.std(_window(data["close"], w)) / (
                np.mean(_window(data["close"], w)) + 1e-10
            )

        for w in [20, 50]:
            features[f"e1a_statistical_kurtosis_{w}"] = (
                engine_1A.rolling_kurtosis_numba(_window(data["close"], w))
            )
            features[f"e1a_statistical_skewness_{w}"] = engine_1A.rolling_skew_numba(
                _window(data["close"], w)
            )

        # 高次モーメント
        for m in [5, 6, 7, 8]:
            for w in [20, 50]:
                # リストにある組み合わせのみ抽出
                if (
                    (m == 5 and w in [20, 50])
                    or (m == 6 and w in [20, 50])
                    or (m == 7 and w in [20, 50])
                    or (m == 8 and w == 50)
                ):
                    features[f"e1a_statistical_moment_{m}_{w}"] = (
                        engine_1A.statistical_moment_numba(_window(data["close"], w), m)
                    )

        features["e1a_statistical_variance_10"] = np.var(_window(data["close"], 10))
        features["e1a_von_neumann_ratio_30"] = engine_1A.von_neumann_ratio_numba(
            _window(data["close"], 30)
        )

        # =======================================================
        # Engine 1B: 時系列モデル (V3リスト準拠)
        # =======================================================
        for w in [50, 100]:
            features[f"e1b_adf_statistic_{w}"] = engine_1B.adf_統計量_udf(
                _window(data["close"], w)
            )
            features[f"e1b_arima_residual_var_{w}"] = engine_1B.arima_残差分散_udf(
                _window(data["close"], w)
            )
            features[f"e1b_kpss_statistic_{w}"] = engine_1B.kpss_統計量_udf(
                _window(data["close"], w)
            )
            features[f"e1b_holt_level_{w}"] = engine_1B.holt_winters_レベル_udf(
                _window(data["close"], w)
            )
            features[f"e1b_holt_trend_{w}"] = engine_1B.holt_winters_トレンド_udf(
                _window(data["close"], w)
            )
            features[f"e1b_lowess_fitted_{w}"] = engine_1B.lowess_適合値_udf(
                _window(data["close"], w)
            )
            features[f"e1b_theil_sen_slope_{w}"] = engine_1B.theil_sen_傾き_udf(
                _window(data["close"], w)
            )

        m50, s50 = (
            np.mean(_window(data["close"], 50)),
            np.std(_window(data["close"], 50)),
        )
        features["e1b_bollinger_lower_50"] = m50 - 2 * s50
        features["e1b_bollinger_upper_50"] = m50 + 2 * s50
        features["e1b_kalman_state_100"] = engine_1B.kalman_状態推定_udf(
            _window(data["close"], 100)
        )
        features["e1b_pp_statistic_100"] = engine_1B.phillips_perron_統計量_udf(
            _window(data["close"], 100)
        )
        features["e1b_price_change"] = _last(close_pct)
        features["e1b_price_range"] = data["high"][-1] - data["low"][-1]
        features["e1b_rolling_mean_100"] = np.mean(_window(data["close"], 100))
        features["e1b_rolling_median_50"] = np.median(_window(data["close"], 50))
        features["e1b_rolling_median_100"] = np.median(_window(data["close"], 100))
        features["e1b_t_dist_dof_50"] = engine_1B.t分布_自由度_udf(
            _window(close_pct, 50)
        )
        features["e1b_t_dist_scale_50"] = engine_1B.t分布_尺度_udf(
            _window(close_pct, 50)
        )
        features["e1b_volatility_20"] = np.std(_window(close_pct, 20))
        features["e1b_zscore_20"] = (
            data["close"][-1] - np.mean(_window(data["close"], 20))
        ) / (np.std(_window(data["close"], 20)) + 1e-10)
        features["e1b_zscore_50"] = (data["close"][-1] - m50) / (s50 + 1e-10)

        # =======================================================
        # Engine 1C: テクニカル指標 (V3リスト準拠)
        # =======================================================
        features["e1c_adx_13"] = _last(
            engine_1C.calculate_adx_numba(
                _array(data["high"]), _array(data["low"]), _array(data["close"]), 13
            )
        )
        features["e1c_aroon_down_14"] = _last(
            engine_1C.calculate_aroon_down_numba(_array(data["low"]), 14)
        )
        features["e1c_aroon_up_14"] = _last(
            engine_1C.calculate_aroon_up_numba(_array(data["high"]), 14)
        )
        features["e1c_aroon_oscillator_14"] = (
            features["e1c_aroon_up_14"] - features["e1c_aroon_down_14"]
        )

        # ATR系
        atr_13_arr = engine_1C.calculate_atr_numba(
            _array(data["high"]), _array(data["low"]), _array(data["close"]), 13
        )
        atr_13 = atr_13_arr[-1]
        features["e1c_atr_13"] = atr_13
        features["e1c_atr_pct_13"] = (atr_13 / data["close"][-1]) * 100
        features["e1c_atr_trend_13"] = (
            atr_13_arr[-1] - atr_13_arr[-2] if len(atr_13_arr) > 1 else 0.0
        )
        for m in [1.5, 2.0, 2.5]:
            if m in [1.5, 2.0]:
                features[f"e1c_atr_lower_13_{m}"] = data["close"][-1] - (atr_13 * m)
            features[f"e1c_atr_upper_13_{m}"] = data["close"][-1] + (atr_13 * m)
        for w in [13, 21, 34, 55]:
            arr = engine_1C.calculate_atr_numba(
                _array(data["high"]), _array(data["low"]), _array(data["close"]), w
            )
            features[f"e1c_atr_volatility_{w}"] = np.std(_window(arr, w))

        # Bollinger Bands (リストにある組み合わせのみ)
        for p, s in [(20, 2.5), (30, 2.5), (50, 2.0), (50, 2.5), (50, 3.0)]:
            m_p, s_p = (
                np.mean(_window(data["close"], p)),
                np.std(_window(data["close"], p)),
            )
            w_p = 2 * s * s_p
            if s == 2.5:
                features[f"e1c_bb_lower_{p}_{s}"] = m_p - s * s_p
                # e1c_bb_upper_30_2.5 は最終リストに存在しないため p==30 を除外
                if p != 30:
                    features[f"e1c_bb_upper_{p}_{s}"] = m_p + s * s_p
                if p == 30:
                    features[f"e1c_bb_width_{p}_{s}"] = w_p
                    features[f"e1c_bb_width_pct_{p}_{s}"] = (w_p / (m_p + 1e-10)) * 100
                    features[f"e1c_bb_position_{p}_{s}"] = (data["close"][-1] - m_p) / (
                        s_p + 1e-10
                    )
                features[f"e1c_bb_percent_{p}_{s}"] = (
                    data["close"][-1] - (m_p - s * s_p)
                ) / (w_p + 1e-10)
            elif s == 2.0 and p == 50:
                features[f"e1c_bb_percent_{p}_2"] = (
                    data["close"][-1] - (m_p - s * s_p)
                ) / (w_p + 1e-10)
            elif s == 3.0 and p == 50:
                features[f"e1c_bb_lower_{p}_3"] = m_p - s * s_p
                features[f"e1c_bb_upper_{p}_3"] = m_p + s * s_p

        features["e1c_coppock_curve"] = (
            np.mean(_window(_pct(data["close"])[-24:] * 100, 10))
            if len(data["close"]) >= 24
            else np.nan
        )
        features["e1c_di_minus_13"] = _last(
            engine_1C.calculate_di_minus_numba(
                _array(data["high"]), _array(data["low"]), _array(data["close"]), 13
            )
        )
        features["e1c_di_plus_13"] = _last(
            engine_1C.calculate_di_plus_numba(
                _array(data["high"]), _array(data["low"]), _array(data["close"]), 13
            )
        )
        features["e1c_di_plus_21"] = _last(
            engine_1C.calculate_di_plus_numba(
                _array(data["high"]), _array(data["low"]), _array(data["close"]), 21
            )
        )

        # DPO / Deviation
        for p in [20, 30, 50]:
            features[f"e1c_dpo_{p}"] = 0.0  # Lookahead回避
        for p in [10, 20, 50, 100, 200]:
            ev = _last(_ema(data["close"], p))
            features[f"e1c_ema_deviation_{p}"] = (
                (data["close"][-1] - ev) / (ev + 1e-10) * 100
            )
            if p in [50, 100, 200]:
                sv = np.mean(_window(data["close"], p))
                features[f"e1c_sma_deviation_{p}"] = (
                    (data["close"][-1] - sv) / (sv + 1e-10) * 100
                )
                if p == 100:
                    features["e1c_sma_100"] = sv

        features["e1c_hma_21"] = _last(
            engine_1C.calculate_hma_numba(_array(data["close"]), 21)
        )
        # KST / MACD / Momentum (リスト準拠)
        roc_10, roc_15, roc_20, roc_30 = (
            _pct(_window(data["close"], 11))[-1],
            _pct(_window(data["close"], 16))[-1],
            _pct(_window(data["close"], 21))[-1],
            _pct(_window(data["close"], 31))[-1],
        )
        kst = (roc_10 * 1 + roc_15 * 2 + roc_20 * 3 + roc_30 * 4) / 10 * 100
        features["e1c_kst"], features["e1c_kst_signal"] = kst, kst
        features["e1c_macd_12_26"] = _last(
            _ema(data["close"], 12) - _ema(data["close"], 26)
        )
        features["e1c_macd_19_39"] = _last(
            _ema(data["close"], 19) - _ema(data["close"], 39)
        )
        features["e1c_macd_histogram_12_26_9"] = features["e1c_macd_12_26"] - _last(
            _ema(_ema(data["close"], 12) - _ema(data["close"], 26), 9)
        )
        features["e1c_macd_histogram_19_39_9"] = features["e1c_macd_19_39"] - _last(
            _ema(_ema(data["close"], 19) - _ema(data["close"], 39), 9)
        )
        features["e1c_macd_histogram_5_35_5"] = _last(
            _ema(data["close"], 5) - _ema(data["close"], 35)
        ) - _last(_ema(_ema(data["close"], 5) - _ema(data["close"], 35), 5))

        for p in [10, 20, 30, 50]:
            features[f"e1c_momentum_{p}"] = data["close"][-1] - data["close"][-1 - p]
            if p in [20, 50]:
                features[f"e1c_rate_of_change_{p}"] = (
                    features[f"e1c_momentum_{p}"] / (data["close"][-1 - p] + 1e-10)
                ) * 100

        # RSI / Stoch / Trend
        rsi_14_arr, rsi_21_arr = (
            engine_1C.calculate_rsi_numba(_array(data["close"]), 14),
            engine_1C.calculate_rsi_numba(_array(data["close"]), 21),
        )
        features["e1c_rsi_14"] = rsi_14_arr[-1]
        features["e1c_rsi_divergence_14"] = (
            (data["close"][-1] - data["close"][-15]) / (data["close"][-15] + 1e-10)
        ) - ((rsi_14_arr[-1] - rsi_14_arr[-15]) / 50 - 1)
        features["e1c_rsi_divergence_21"] = (
            (data["close"][-1] - data["close"][-22]) / (data["close"][-22] + 1e-10)
        ) - ((rsi_21_arr[-1] - rsi_21_arr[-22]) / 50 - 1)
        features["e1c_rsi_momentum_14"], features["e1c_rsi_momentum_21"] = (
            rsi_14_arr[-1] - rsi_14_arr[-2],
            rsi_21_arr[-1] - rsi_21_arr[-2],
        )

        stoch_k_14, stoch_k_21 = (
            engine_1C.calculate_stochastic_numba(
                _array(data["high"]), _array(data["low"]), _array(data["close"]), 14
            ),
            engine_1C.calculate_stochastic_numba(
                _array(data["high"]), _array(data["low"]), _array(data["close"]), 21
            ),
        )
        features["e1c_stoch_k_14"], features["e1c_stoch_d_14_3"] = (
            stoch_k_14[-1],
            np.mean(_window(stoch_k_14, 3)),
        )
        features["e1c_stoch_d_21_5"], features["e1c_stoch_slow_d_21_5_5"] = (
            np.mean(_window(stoch_k_21, 5)),
            np.mean(_window(_ema(stoch_k_21, 5), 5)),
        )
        features["e1c_stochastic_rsi_14"] = (
            (rsi_14_arr[-1] - np.min(_window(rsi_14_arr, 14)))
            / (
                np.max(_window(rsi_14_arr, 14))
                - np.min(_window(rsi_14_arr, 14))
                + 1e-10
            )
            * 100
        )
        features["e1c_stochastic_rsi_21"] = (
            (rsi_21_arr[-1] - np.min(_window(rsi_21_arr, 21)))
            / (
                np.max(_window(rsi_21_arr, 21))
                - np.min(_window(rsi_21_arr, 21))
                + 1e-10
            )
            * 100
        )

        for p in [20, 50, 100]:
            if p in [20, 50]:
                features[f"e1c_trend_strength_{p}"] = 1.0 / (
                    np.std(_window(data["close"], p)) + 1e-10
                )
            features[f"e1c_trend_consistency_{p}"] = (
                engine_1C.rolling_trend_consistency_numba(
                    _window(data["close"], p + 10), p
                )
            )

        features["e1c_trix_14"] = _last(
            engine_1C.calculate_trix_numba(_array(data["close"]), 14)
        )
        features["e1c_tsi_13"] = _last(
            engine_1C.calculate_tsi_numba(_array(data["close"]), 25, 13)
        )
        features["e1c_ultimate_oscillator"] = _last(
            engine_1C.calculate_ultimate_oscillator_numba(
                _array(data["high"]),
                _array(data["low"]),
                _array(data["close"]),
                _array(data["volume"]),
            )
        )
        features["e1c_williams_r_14"] = _last(
            engine_1C.calculate_williams_r_numba(
                _array(data["high"]), _array(data["low"]), _array(data["close"]), 14
            )
        )
        features["e1c_wma_200"] = engine_1C.wma_rolling_numba(
            _window(data["close"], 200), 200
        )

        # RVI (V3リストにあるものに限定)
        for p in [10, 14, 20]:
            rvi_arr = _last(_ema(data["close"] - data["open"], p)) / (
                _last(_ema(data["high"] - data["low"], p)) + 1e-10
            )
            features[f"e1c_relative_vigor_index_{p}"] = rvi_arr
            features[f"e1c_rvi_signal_{p}"] = rvi_arr  # 簡易化

        features["e1c_schaff_trend_cycle_12_26_9"] = _last(
            engine_1C.calculate_schaff_trend_cycle_numba(
                _array(data["close"]), 12, 26, 9
            )
        )
        features["e1c_schaff_trend_cycle_23_50_10"] = _last(
            engine_1C.calculate_schaff_trend_cycle_numba(
                _array(data["close"]), 23, 50, 10
            )
        )

        # =======================================================
        # Engine 1D: ボリューム・プライスアクション (V3リスト準拠)
        # =======================================================
        features["e1d_accumulation_distribution"] = _last(
            engine_1D.accumulation_distribution_udf(
                _array(data["high"]),
                _array(data["low"]),
                _array(data["close"]),
                _array(data["volume"]),
            )
        )
        features["e1d_body_size"] = abs(data["close"][-1] - data["open"][-1])
        features["e1d_candlestick_pattern"] = _last(
            engine_1D.candlestick_patterns_udf(
                _array(data["open"]),
                _array(data["high"]),
                _array(data["low"]),
                _array(data["close"]),
            )
        )
        features["e1d_chaikin_volatility_10"] = _last(
            engine_1D.chaikin_volatility_udf(
                _array(data["high"]), _array(data["low"]), 10
            )
        )
        features["e1d_cmf_13"] = _last(
            engine_1D.cmf_udf(
                _array(data["high"]),
                _array(data["low"]),
                _array(data["close"]),
                _array(data["volume"]),
                13,
            )
        )
        features["e1d_force_index"] = _last(
            engine_1D.force_index_udf(_array(data["close"]), _array(data["volume"]))
        )

        for w in [10, 20, 30, 50]:
            features[f"e1d_hv_robust_{w}"] = engine_1D.hv_robust_udf(
                _window(close_pct, w)
            )
        features["e1d_hv_robust_annual_252"] = engine_1D.hv_robust_udf(
            _window(close_pct, 252)
        ) * np.sqrt(252)

        for w in [10, 30, 50]:
            features[f"e1d_hv_standard_{w}"] = engine_1D.hv_standard_udf(
                _window(close_pct, w)
            )

        features["e1d_hv_regime_50"] = (
            1.0 if features["e1d_hv_robust_50"] > 0.005 else 0.0
        )
        features["e1d_intraday_return"] = (data["close"][-1] - data["open"][-1]) / (
            data["open"][-1] + 1e-10
        )
        features["e1d_lower_wick_ratio"] = (
            min(data["open"][-1], data["close"][-1]) - data["low"][-1]
        ) / (data["high"][-1] - data["low"][-1] + 1e-10)
        features["e1d_mass_index_20"] = _last(
            engine_1D.mass_index_udf(_array(data["high"]), _array(data["low"]), 20)
        )
        features["e1d_mfi_13"] = _last(
            engine_1D.mfi_udf(
                _array(data["high"]),
                _array(data["low"]),
                _array(data["close"]),
                _array(data["volume"]),
                13,
            )
        )
        features["e1d_obv"] = _last(
            engine_1D.obv_udf(_array(data["close"]), _array(data["volume"]))
        )
        features["e1d_overnight_gap"] = (
            (data["open"][-1] - data["close"][-2]) / (data["close"][-2] + 1e-10)
            if len(data["close"]) > 1
            else 0.0
        )
        features["e1d_price_channel_upper_100"] = np.max(_window(data["high"], 100))
        features["e1d_price_location_hl"] = (data["close"][-1] - data["low"][-1]) / (
            data["high"][-1] - data["low"][-1] + 1e-10
        )
        features["e1d_upper_wick_ratio"] = (
            data["high"][-1] - max(data["open"][-1], data["close"][-1])
        ) / (data["high"][-1] - data["low"][-1] + 1e-10)
        features["e1d_volume_price_trend"] = np.mean(
            _window(data["close"] * data["volume"], 10)
        )
        features["e1d_volume_ratio"] = data["volume"][-1] / (
            np.mean(_window(data["volume"], 20)) + 1e-10
        )

        # =======================================================
        # Engine 1E: 信号処理 (V3リスト準拠)
        # =======================================================
        for w in [128, 256]:
            features[f"e1e_acoustic_frequency_{w}"] = engine_1E.acoustic_frequency_udf(
                _window(close_pct, w)
            )
        features["e1e_acoustic_power_128"] = engine_1E.acoustic_power_udf(
            _window(close_pct, 128)
        )
        features["e1e_hilbert_amp_cv_100"] = np.std(_window(data["close"], 100)) / (
            np.mean(_window(data["close"], 100)) + 1e-10
        )
        features["e1e_hilbert_amplitude_100"] = engine_1E.hilbert_amplitude_udf(
            _window(close_pct, 100)
        )
        features["e1e_hilbert_freq_energy_ratio_100"] = np.sum(
            _window(close_pct, 100) ** 2
        ) / (np.sum(_window(data["close"], 100) ** 2) + 1e-10)
        features["e1e_hilbert_freq_mean_100"] = engine_1E.hilbert_freq_mean_udf(
            _window(close_pct, 100)
        )
        features["e1e_hilbert_freq_std_100"] = engine_1E.hilbert_freq_std_udf(
            _window(close_pct, 100)
        )
        features["e1e_hilbert_phase_stability_50"] = (
            engine_1E.hilbert_phase_stability_udf(_window(close_pct, 50))
        )
        features["e1e_hilbert_phase_var_50"] = engine_1E.hilbert_phase_var_udf(
            _window(close_pct, 50)
        )
        features["e1e_signal_crest_factor_50"] = np.max(
            np.abs(_window(data["close"], 50))
        ) / (np.sqrt(np.mean(_window(data["close"], 50) ** 2)) + 1e-10)
        features["e1e_signal_peak_to_peak_100"] = np.max(
            _window(data["close"], 100)
        ) - np.min(_window(data["close"], 100))
        features["e1e_signal_rms_50"] = np.sqrt(np.mean(_window(close_pct, 50) ** 2))
        features["e1e_spectral_bandwidth_128"] = engine_1E.spectral_bandwidth_udf(
            _window(close_pct, 128)
        )
        features["e1e_spectral_centroid_128"] = engine_1E.spectral_centroid_udf(
            _window(close_pct, 128)
        )
        for w in [64, 128, 256, 512]:
            features[f"e1e_spectral_energy_{w}"] = np.sum(_window(close_pct, w) ** 2)
        features["e1e_spectral_entropy_64"] = engine_1E.spectral_entropy_udf(
            _window(close_pct, 64)
        )
        features["e1e_spectral_flatness_128"] = engine_1E.spectral_flatness_udf(
            _window(close_pct, 128)
        )
        features["e1e_spectral_rolloff_128"] = engine_1E.spectral_rolloff_udf(
            _window(close_pct, 128)
        )
        features["e1e_wavelet_entropy_64"] = engine_1E.wavelet_entropy_udf(
            _window(close_pct, 64)
        )
        for w in [64, 128, 256]:
            features[f"e1e_wavelet_mean_{w}"] = np.mean(_window(close_pct, w))
        for w in [32, 64, 128, 256]:
            features[f"e1e_wavelet_std_{w}"] = np.std(_window(close_pct, w))

        # =======================================================
        # Engine 1F: 実験的特徴量 (V3リスト準拠)
        # =======================================================
        for w in [21, 34, 55, 89]:
            features[f"e1f_aesthetic_balance_{w}"] = (
                engine_1F.rolling_aesthetic_balance_udf(_window(data["close"], w))
            )
            features[f"e1f_symmetry_measure_{w}"] = (
                engine_1F.rolling_symmetry_measure_udf(_window(data["close"], w))
            )
        # e1f_golden_ratio_adherence_89 は最終リストに存在しないため w=21,34,55 のみ
        for w in [21, 34, 55]:
            features[f"e1f_golden_ratio_adherence_{w}"] = (
                engine_1F.rolling_golden_ratio_adherence_udf(_window(data["close"], w))
            )

        features["e1f_biomechanical_efficiency_20"] = (
            engine_1F.rolling_biomechanical_efficiency_udf(_window(data["close"], 20))
        )
        for w in [20, 40, 60]:
            features[f"e1f_energy_expenditure_{w}"] = (
                engine_1F.rolling_energy_expenditure_udf(_window(data["close"], w))
            )
        for w in [48, 96]:
            features[f"e1f_harmony_{w}"] = engine_1F.rolling_harmony_udf(
                _window(data["close"], w)
            )
        for w in [10, 20, 40]:
            features[f"e1f_kinetic_energy_{w}"] = engine_1F.rolling_kinetic_energy_udf(
                _window(data["close"], w)
            )
        for w in [25, 40]:
            features[f"e1f_linguistic_complexity_{w}"] = (
                engine_1F.rolling_linguistic_complexity_udf(_window(data["close"], w))
            )

        features["e1f_muscle_force_20"] = engine_1F.rolling_muscle_force_udf(
            _window(data["close"], 20)
        )
        for w in [24, 48, 96]:
            features[f"e1f_musical_tension_{w}"] = (
                engine_1F.rolling_musical_tension_udf(_window(data["close"], w))
            )
            features[f"e1f_rhythm_pattern_{w}"] = engine_1F.rolling_rhythm_pattern_udf(
                _window(data["close"], w)
            )

        for w in [20, 30, 50, 100]:
            features[f"e1f_network_clustering_{w}"] = (
                engine_1F.rolling_network_clustering_udf(_window(data["close"], w))
            )
            features[f"e1f_network_density_{w}"] = (
                engine_1F.rolling_network_density_udf(_window(data["close"], w))
            )

        for w in [15, 25, 40]:
            features[f"e1f_semantic_flow_{w}"] = engine_1F.rolling_semantic_flow_udf(
                _window(data["close"], w)
            )
        for w in [12, 24, 48, 96]:
            features[f"e1f_tonality_{w}"] = engine_1F.rolling_tonality_udf(
                _window(data["close"], w)
            )
        for w in [15, 25, 40, 80]:
            features[f"e1f_vocabulary_diversity_{w}"] = (
                engine_1F.rolling_vocabulary_diversity_udf(_window(data["close"], w))
            )

        # -------------------------------------------------------
        # 純化用プロキシ (必須)
        features["atr"] = atr_13
        features["log_return"] = np.log(
            (data["close"][-1] + 1e-10) / (data["close"][-2] + 1e-10)
        )
        features["price_momentum"] = (
            data["close"][-1] - data["close"][-11]
            if len(data["close"]) > 10
            else np.nan
        )
        features["rolling_volatility"] = np.std(_window(close_pct, 20))
        features["volume_ratio"] = data["volume"][-1] / (
            np.mean(_window(data["volume"], 20)) + 1e-10
        )

        return features

    def calculate_dynamic_context(self, hmm_model: Any) -> Dict[str, float]:
        """
        [環境認識用] HMMレジームと主要統計量の計算。
        """
        if "D1" not in self.data_buffers:
            return {}
        close_arr = np.array(self.data_buffers["D1"]["close"], dtype=np.float64)
        if len(close_arr) < 50:
            return {}

        context = {}
        try:
            # HMMレジーム判定
            if hmm_model is not None:
                log_rets = np.diff(np.log(close_arr[-100:] + 1e-10)).reshape(-1, 1)
                probs = hmm_model.predict_proba(log_rets)
                context["hmm_prob_0"], context["hmm_prob_1"] = (
                    probs[-1][0],
                    probs[-1][1],
                )

            context["e1a_statistical_kurtosis_50"] = engine_1A.rolling_kurtosis_numba(
                close_arr[-50:]
            )
            adx_arr = engine_1C.calculate_adx_numba(
                np.array(self.data_buffers["D1"]["high"]),
                np.array(self.data_buffers["D1"]["low"]),
                close_arr,
                21,
            )
            context["e1c_adx_21"] = adx_arr[-1]
            return {k: (v if np.isfinite(v) else 0.0) for k, v in context.items()}
        except Exception:
            return {}

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
            return np.array([np.clip(final_vector, -1e5, 1e5)])
        except Exception as e:
            self.logger.error(f"Vector calculation error: {e}")
            return None
