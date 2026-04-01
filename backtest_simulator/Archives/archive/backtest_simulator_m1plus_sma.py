# /workspace/models/backtest_simulator.py
# [修正版: 破産閾値 (min_capital_threshold) の導入]
# [修正版: Chunking (Streaming) 導入によるメモリ効率化 + Schemaログ抑制]
# [修正版: Booster API の .predict() を使うように修正]
# [修正版: AttributeError (clip_min) 修正 + レバレッジ調整]
# [修正版: 最大リスク上限導入 + グラフエラー修正]
# [修正版: ロットサイズ (lot_size) の計算とログ出力を追加]
# ★★★ [最終修正版 v2: S6の動的ATR/SL/PT/方向性 を使って正確なロット計算＋ログ出力] ★★★
# ★★★ [最終修正版 v3: 最大ロット数、必要証拠金/動的レバレッジ制限、スプレッドコストを実装] ★★★
# ★★★ [V4修正: M2文脈特徴量をIn-Sampleモード用に統合] ★★★
# ★★★ [V4修正(B): L187-188 構文エラーを修正] ★★★
# ★★★ [V5修正: 日足データ読み込み + 環境認識フィルタ(Trend/Crash) + マルチシナリオ比較レポート] ★★★

import sys
from pathlib import Path
import logging
import argparse
from dataclasses import dataclass, field
from typing import Dict, Any, Tuple, List
import json
import datetime as dt
import zoneinfo

import polars as pl
import numpy as np
import joblib
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import seaborn as sns
from tqdm import tqdm
import gc

from decimal import Decimal, getcontext, ROUND_HALF_UP

# --- Decimal の精度を設定 (5000桁) ---
getcontext().prec = 5000

# --- プロジェクトのルートディレクトリをPythonの検索パスに追加 ---
project_root = Path(__file__).resolve().parents[1]
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

# --- blueprintから必要なパスをインポート ---
from blueprint import (
    S6_WEIGHTED_DATASET,
    S7_M2_OOF_PREDICTIONS,
    S3_FEATURES_FOR_TRAINING,
    S7_M1_MODEL_PKL,
    S7_M2_MODEL_PKL,
    S7_M1_CALIBRATED,
    S7_M2_CALIBRATED,
    S7_MODELS,
    S7_CONTEXT_FEATURES,
)

# --- 出力ファイルパス ---
FINAL_REPORT_PATH = S7_MODELS / "final_backtest_report.json"
EQUITY_CURVE_PATH = S7_MODELS / "equity_curve_comparison.png"  # ファイル名を変更


# --- ロギング設定 ---
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

# ★★★ 定数: XAUUSDの契約サイズ ★★★
CONTRACT_SIZE = Decimal("100")  # 1 lot = 100 oz

# ★★★ ヘルパー関数: JSTタイムゾーン ★★★
JST = zoneinfo.ZoneInfo("Asia/Tokyo")


# ================================================================
# フェーズ 0: 作戦司令室 (パラメータ設定)
# ================================================================
@dataclass
class BacktestConfig:
    """シミュレーションの全パラメータをここで一元管理します"""

    initial_capital: float = 1000.0
    simulation_data_path: Path = S6_WEIGHTED_DATASET
    oof_predictions_path: Path = S7_M2_OOF_PREDICTIONS
    feature_list_path: Path = S3_FEATURES_FOR_TRAINING
    context_data_path: Path = S7_CONTEXT_FEATURES

    # --- ★★★ 追加: 日足データのパス (ユーザー指定パスに修正) ★★★ ---
    daily_data_path: Path = Path(
        "/workspace/data/XAUUSD/stratum_6_training/1A_2B/daily_ohlc.parquet"
    )
    # -------------------------------------------------------------

    m1_model_path: Path = S7_M1_MODEL_PKL
    m2_model_path: Path = S7_M2_MODEL_PKL
    m1_calibrator_path: Path = S7_M1_CALIBRATED
    m2_calibrator_path: Path = S7_M2_CALIBRATED
    kelly_fraction: float = 1.0
    max_risk_per_trade: float = 0.02
    f_star_threshold: float = 0.0
    m2_proba_threshold: float = 0.6
    test_limit_partitions: int = 0
    oof_mode: bool = False
    min_capital_threshold: float = 1.0
    min_lot_size: float = 0.01

    max_positions: int = 50

    base_leverage: float = 2000.0
    spread_pips: float = 16.0
    value_per_pip: float = 1.0


class BacktestSimulator:
    def __init__(self, config: BacktestConfig):
        self.config = config
        self.m1_model = self._load_model(config.m1_model_path, "M1 Model")
        self.m2_model = self._load_model(config.m2_model_path, "M2 Model")
        self.m1_calibrator = self._load_model(
            config.m1_calibrator_path, "M1 Calibrator (Isotonic)"
        )
        self.m2_calibrator = self._load_model(
            config.m2_calibrator_path, "M2 Calibrator (Isotonic)"
        )
        self.features_base = self._load_features(config.feature_list_path)

        self.context_features_m2 = [
            "hmm_prob_0",
            "hmm_prob_1",
            "atr_ratio",
            "trend_bias_25",
            "e1a_statistical_kurtosis_50",
            "e1c_adx_21",
            "e2a_mfdfa_hurst_mean_250",
            "e2a_kolmogorov_complexity_60",
        ]
        self.features_m2 = (
            self.features_base + ["m1_pred_proba"] + self.context_features_m2
        )
        logging.info(f"Loading M2 context features from {config.context_data_path}...")
        try:
            context_df = pl.read_parquet(config.context_data_path)
            self.context_df = (
                context_df.with_columns(pl.col("timestamp").dt.date().alias("date"))
                .select(["date"] + self.context_features_m2)
                .sort("date")
                .lazy()
            )
            logging.info(
                f"  -> M2 features redefined with {len(self.context_features_m2)} context features."
            )
        except Exception as e:
            logging.error(
                f"CRITICAL: Failed to load M2 context features from {config.context_data_path}: {e}",
                exc_info=True,
            )
            raise

        # --- ★★★ 追加: 日足データの読み込みと準備 ★★★ ---
        self.daily_features_lf = self._prepare_daily_features()
        # ----------------------------------------------------

        self._current_capital = Decimal(str(self.config.initial_capital))

    def _prepare_daily_features(self) -> pl.LazyFrame:
        """日足データを読み込み、SMA200と前日変動率を計算してLazyFrameを返す"""
        logging.info(f"Loading Daily OHLC data from {self.config.daily_data_path}...")
        if not self.config.daily_data_path.exists():
            logging.warning(
                f"Daily data file not found at {self.config.daily_data_path}. Filters will be ineffective (NaN)."
            )
            # ダミーを返す
            return pl.DataFrame(
                {"date": [], "daily_sma_200": [], "prev_day_change": []},
                schema={
                    "date": pl.Date,
                    "daily_sma_200": pl.Float64,
                    "prev_day_change": pl.Float64,
                },
            ).lazy()

        try:
            df = pl.read_parquet(self.config.daily_data_path)

            # timestamp を date に変換（必要な場合）
            if "date" not in df.columns and "timestamp" in df.columns:
                df = df.with_columns(pl.col("timestamp").dt.date().alias("date"))

            # 指標計算
            # 1. SMA 200
            # 2. 前日変動率 ((Close - Open) / Open) の1日シフト (当日判断に使うのは「前日」の値のため)
            df = df.sort("date").with_columns(
                [
                    pl.col("close")
                    .rolling_mean(window_size=200)
                    .alias("daily_sma_200"),
                    (
                        ((pl.col("close") - pl.col("open")) / pl.col("open")).shift(1)
                    ).alias("prev_day_change"),
                ]
            )

            # 必要な列のみ選択
            lf = df.select(["date", "daily_sma_200", "prev_day_change"]).lazy()
            logging.info("  -> Daily features (SMA200, PrevChange) prepared.")
            return lf

        except Exception as e:
            logging.error(f"Error preparing daily features: {e}", exc_info=True)
            raise

    def _load_model(self, path: Path, name: str):
        logging.info(f"Loading {name} from {path}...")
        if not path.exists():
            raise FileNotFoundError(f"{name} file not found: {path}")
        try:
            model = joblib.load(path)
            logging.info(f"  -> {name} loaded successfully.")
            return model
        except Exception as e:
            logging.error(f"Error loading {name} from {path}: {e}", exc_info=True)
            raise

    def _load_features(self, path: Path) -> list[str]:
        logging.info(f"Loading feature list from {path}...")
        if not path.exists():
            raise FileNotFoundError(f"Feature list not found: {path}")
        with open(path, "r") as f:
            features = [line.strip() for line in f if line.strip()]
        logging.info(f"   -> Loaded {len(features)} base features.")
        return features

    def _get_effective_leverage(self, equity: Decimal) -> Decimal:
        base_leverage_dec = Decimal(str(self.config.base_leverage))
        if equity < Decimal("5000"):
            limit_leverage = base_leverage_dec
        elif equity < Decimal("30000"):
            limit_leverage = Decimal("2000")
        elif equity < Decimal("100000"):
            limit_leverage = Decimal("1000")
        else:
            limit_leverage = Decimal("500")

        return base_leverage_dec.min(limit_leverage)

    def _get_max_lot_allowed(self, timestamp_utc: dt.datetime) -> Decimal:
        timestamp_jst = timestamp_utc.astimezone(JST)
        hour_jst = timestamp_jst.hour

        if 6 <= hour_jst < 16:
            return Decimal("20")
        else:
            return Decimal("200")

    def run(self):
        logging.info("### Backtest Simulator: START (Multi-Scenario) ###")
        logging.info(
            f"Strategy: Probabilistic Betting with Kelly Fraction = {self.config.kelly_fraction}, "
            f"Base Leverage = {self.config.base_leverage}, Max Risk/Trade = {self.config.max_risk_per_trade * 100:.1f}%, "
            f"Spread = {self.config.spread_pips} pips"
        )
        logging.info(
            f"Bankruptcy Threshold (Min Capital): {self.config.min_capital_threshold:,.2f}"
        )

        lf, partitions_df = self._prepare_data()

        # --- ★★★ マルチシナリオ定義 ★★★ ---
        scenarios = {
            "Original": {"filter_mode": "none"},
            "Trend_Filter": {"filter_mode": "trend"},
            "Crash_Filter": {"filter_mode": "crash"},
            "Combined": {"filter_mode": "both"},
        }

        # 各シナリオの状態管理用辞書
        scenario_states = {}
        for name in scenarios.keys():
            scenario_states[name] = {
                "current_capital": Decimal(str(self.config.initial_capital)),
                "results_dfs": [],
                "trade_logs": [],
                "active_exit_times": [],
            }
        # ------------------------------------

        logging.info(
            f"Processing {len(partitions_df)} partitions sequentially for {len(scenarios)} scenarios..."
        )

        partitions_to_process = partitions_df
        if self.config.test_limit_partitions > 0:
            logging.warning(
                f"--- TEST MODE: Limiting to first {self.config.test_limit_partitions} partitions. ---"
            )
            partitions_to_process = partitions_df.head(
                self.config.test_limit_partitions
            )

        DECIMAL_MIN_CAPITAL = Decimal(str(self.config.min_capital_threshold))

        for row in tqdm(
            partitions_to_process.iter_rows(named=True),
            total=len(partitions_to_process),
            desc="Simulating Partitions",
        ):
            current_date = row["date"]

            logging.debug(f"Processing partition: {current_date}")
            try:
                # データを収集 (シナリオ共通)
                df_chunk = lf.filter(
                    pl.col("timestamp").dt.date() == current_date
                ).collect()
                if df_chunk.is_empty():
                    continue

                # 予測 (シナリオ共通)
                df_chunk_predicted = self._run_ai_predictions(df_chunk)

                # --- ★★★ 各シナリオのシミュレーションを実行 ★★★ ---
                for name, params in scenarios.items():
                    state = scenario_states[name]

                    # 破産している場合はスキップ
                    if state["current_capital"] < DECIMAL_MIN_CAPITAL:
                        continue

                    # シミュレーションループ実行
                    # 引数に現在の資本、フィルタモード、保有ポジションリストを渡す
                    results_chunk, trade_log_chunk, next_capital, next_positions = (
                        self._run_simulation_loop(
                            df_chunk_predicted,
                            start_capital=state["current_capital"],
                            filter_mode=params["filter_mode"],
                            active_exit_times=state["active_exit_times"],
                        )
                    )

                    # 状態更新
                    state["current_capital"] = next_capital
                    state["active_exit_times"] = next_positions
                    state["results_dfs"].append(results_chunk)
                    state["trade_logs"].append(trade_log_chunk)
                # ----------------------------------------------------

                del df_chunk, df_chunk_predicted, results_chunk, trade_log_chunk
                gc.collect()

            except Exception as e:
                logging.error(
                    f"Error processing partition {current_date}: {e}", exc_info=True
                )
                continue

        # --- 結果の集約とレポート作成 ---
        final_results_dict = {}

        for name in scenarios.keys():
            state = scenario_states[name]
            if not state["results_dfs"]:
                logging.warning(f"No results for scenario {name}.")
                continue

            try:
                full_results = pl.concat(state["results_dfs"]).sort("timestamp")

                # Trade Logs
                if any(not df.is_empty() for df in state["trade_logs"]):
                    full_trade_log = pl.concat(
                        [df for df in state["trade_logs"] if not df.is_empty()]
                    ).sort("timestamp")
                else:
                    full_trade_log = pl.DataFrame()

                final_results_dict[name] = {
                    "equity_df": full_results,
                    "trade_log": full_trade_log,
                    "final_capital": state["current_capital"],
                }
            except Exception as e:
                logging.error(f"Error concatenating results for {name}: {e}")

        if not final_results_dict:
            logging.error("No simulation results generated for any scenario.")
            return

        # 比較レポートの作成 (既存のレポート機能の置き換え)
        self._analyze_and_report_comparison(final_results_dict)
        logging.info("### Backtest Simulator: FINISHED ###")

    def _prepare_data(self) -> Tuple[pl.LazyFrame, pl.DataFrame]:
        # S6から読み込む基本カラムリスト
        base_cols = [
            "timestamp",
            "close",
            "label",
            "payoff_ratio",
            "atr_value",
            "sl_multiplier",
            "pt_multiplier",
            "direction",
            "t1",
        ]

        if not self.config.oof_mode:  # In-Sample Mode
            logging.info(
                f"Preparing data from {self.config.simulation_data_path} (In-Sample Mode)..."
            )
            required_cols_set = set(self.features_base)
            required_cols_set.update(base_cols)
            required_cols = list(required_cols_set)

            try:
                glob_path = str(self.config.simulation_data_path / "**/*.parquet")
                lf = pl.scan_parquet(glob_path)
                schema = lf.collect_schema()
            except Exception as e:
                logging.error(f"Failed to scan Parquet files: {e}", exc_info=True)
                raise

            missing_cols = [col for col in required_cols if col not in schema]
            if missing_cols:
                raise ValueError(
                    f"CRITICAL: Required columns not found in S6 dataset schema! Missing: {missing_cols}. "
                )

            lf = lf.select(required_cols).sort("timestamp")

            logging.info("  -> [In-Sample] Joining M2 context features...")
            lf = lf.with_columns(pl.col("timestamp").dt.date().alias("date"))
            lf = lf.join_asof(self.context_df, on="date")

        else:  # OOF Mode
            logging.info(
                f"Preparing data from {self.config.simulation_data_path} and {self.config.oof_predictions_path} (OOF Mode)..."
            )
            try:
                base_glob_path = str(self.config.simulation_data_path / "**/*.parquet")
                base_lf = pl.scan_parquet(base_glob_path).select(base_cols)
            except Exception as e:
                logging.error(f"Failed to scan base data (S6): {e}", exc_info=True)
                raise

            oof_cols = ["timestamp", "prediction"]
            try:
                oof_lf = pl.scan_parquet(self.config.oof_predictions_path).select(
                    oof_cols
                )
                oof_lf = oof_lf.rename({"prediction": "m2_raw_proba"})
            except Exception as e:
                logging.error(
                    f"Failed to scan OOF predictions (S7): {e}", exc_info=True
                )
                raise

            logging.info("Joining base data (S6) with OOF predictions (S7)...")
            lf = base_lf.join(oof_lf, on="timestamp", how="inner").sort("timestamp")

            # 日足結合のためにdate列作成
            lf = lf.with_columns(pl.col("timestamp").dt.date().alias("date"))

        # --- ★★★ 追加: 日足特徴量 (SMA200, PrevChange) の結合 ★★★ ---
        logging.info("Joining Daily features (SMA200, PrevChange)...")
        # 日足データは date をキーにして結合 (left join)
        lf = lf.join(self.daily_features_lf, on="date", how="left")

        # 結合後、不要になった date 列を削除
        lf = lf.drop("date")
        # -------------------------------------------------------------

        logging.info("Discovering partitions...")
        partitions_df = (
            lf.select(pl.col("timestamp").dt.date().alias("date"))
            .unique()
            .collect()
            .sort("date")
        )

        if partitions_df.is_empty():
            raise ValueError(
                "No partitions found. Check data paths and OOF/S6 alignment."
            )

        logging.info(f"Found {len(partitions_df)} partitions.")
        logging.info(
            "LazyFrame prepared. Data will be collected partition by partition."
        )
        return lf, partitions_df

    def _run_ai_predictions(self, df_chunk: pl.DataFrame) -> pl.DataFrame:
        logging.debug(f"Running AI processing for chunk (size: {len(df_chunk)})...")
        if not self.config.oof_mode:
            logging.debug("  -> Mode: In-Sample (Re-predicting)...")
            logging.debug("  -> Step 1/2: M1 Prediction & Calibration...")
            try:
                missing_m1_features = [
                    f for f in self.features_base if f not in df_chunk.columns
                ]
                if missing_m1_features:
                    raise ValueError(f"Missing M1 features: {missing_m1_features}")
                X_m1 = df_chunk.select(self.features_base).fill_null(0).to_numpy()
                raw_m1_proba = self.m1_model.predict(X_m1)
                calibrated_m1_proba = self.m1_calibrator.predict(raw_m1_proba)
                calibrated_m1_proba = np.clip(calibrated_m1_proba, 0.0, 1.0)
                df_chunk = df_chunk.with_columns(
                    pl.Series("m1_pred_proba", calibrated_m1_proba)
                )
            except Exception as e:
                logging.error(f"Error M1 (In-Sample): {e}", exc_info=True)
                raise
            logging.debug("  -> Step 2/2: M2 Prediction & Calibration...")
            try:
                missing_m2_features = [
                    f for f in self.features_m2 if f not in df_chunk.columns
                ]
                if missing_m2_features:
                    raise ValueError(f"Missing M2 features: {missing_m2_features}")
                X_m2 = df_chunk.select(self.features_m2).fill_null(0).to_numpy()
                raw_m2_proba = self.m2_model.predict(X_m2)
                calibrated_m2_proba = self.m2_calibrator.predict(raw_m2_proba)
                calibrated_m2_proba = np.clip(calibrated_m2_proba, 0.0, 1.0)
                df_chunk = df_chunk.with_columns(
                    pl.Series("m2_calibrated_proba", calibrated_m2_proba)
                )
                logging.debug("AI predictions (In-Sample) completed.")
                return df_chunk
            except Exception as e:
                logging.error(f"Error M2 (In-Sample): {e}", exc_info=True)
                raise
        else:  # OOF Mode
            logging.debug("  -> Mode: OOF (Loading pre-calculated)...")
            try:
                if "m2_raw_proba" not in df_chunk.columns:
                    raise ValueError("Missing 'm2_raw_proba' column.")
                raw_m2_proba_oof = df_chunk["m2_raw_proba"].to_numpy()
                logging.debug("  -> Step 1/1: M2 Calibration (OOF)...")
                calibrated_m2_proba = self.m2_calibrator.predict(raw_m2_proba_oof)
                calibrated_m2_proba = np.clip(calibrated_m2_proba, 0.0, 1.0)
                df_chunk = df_chunk.with_columns(
                    pl.Series("m2_calibrated_proba", calibrated_m2_proba)
                )
                logging.debug("AI processing (OOF) completed.")
                return df_chunk
            except Exception as e:
                logging.error(f"Error M2 (OOF): {e}", exc_info=True)
                raise

    def _run_simulation_loop(
        self,
        df_chunk: pl.DataFrame,
        start_capital: Decimal,  # ★ 追加
        filter_mode: str,  # ★ 追加 ('none', 'trend', 'crash', 'both')
        active_exit_times: List[int],  # ★ 追加 (状態引継ぎ用)
    ) -> Tuple[pl.DataFrame, pl.DataFrame, Decimal, List[int]]:  # ★ 戻り値変更
        trade_log_chunk = []
        equity_values_chunk = []
        current_capital = start_capital

        DECIMAL_ZERO = Decimal("0.0")
        DECIMAL_ONE = Decimal("1.0")
        DECIMAL_MAX_RISK = Decimal(str(self.config.max_risk_per_trade))
        DECIMAL_KELLY_FRACTION = Decimal(str(self.config.kelly_fraction))
        DECIMAL_F_STAR_THRESHOLD = Decimal(str(self.config.f_star_threshold))
        DECIMAL_MIN_CAPITAL = Decimal(str(self.config.min_capital_threshold))
        DECIMAL_VALUE_PER_PIP = Decimal(str(self.config.value_per_pip))
        DECIMAL_MIN_LOT_SIZE = Decimal(str(self.config.min_lot_size))
        DECIMAL_CONTRACT_SIZE = CONTRACT_SIZE
        MAX_POSITIONS = self.config.max_positions

        timestamps_chunk = df_chunk["timestamp"].to_list()
        t1_chunk = df_chunk["t1"].to_list()
        close_prices_chunk = df_chunk["close"].to_numpy()
        p_m2_calibrated = df_chunk["m2_calibrated_proba"].to_numpy()
        labels_chunk = df_chunk["label"].to_numpy()
        atr_values_chunk = df_chunk["atr_value"].to_numpy()
        sl_multipliers_chunk = df_chunk["sl_multiplier"].to_numpy()
        pt_multipliers_chunk = df_chunk["pt_multiplier"].to_numpy()
        directions_chunk = df_chunk["direction"].to_numpy()

        # --- ★★★ 日足指標の取得 ★★★ ---
        has_daily_data = (
            "daily_sma_200" in df_chunk.columns
            and "prev_day_change" in df_chunk.columns
        )
        if has_daily_data:
            sma_200_chunk = df_chunk["daily_sma_200"].to_numpy()
            prev_change_chunk = df_chunk["prev_day_change"].to_numpy()
        else:
            sma_200_chunk = [None] * len(df_chunk)
            prev_change_chunk = [None] * len(df_chunk)
        # -------------------------------

        payoff_ratios_chunk = [
            Decimal(str(b))
            if b is not None and np.isfinite(b) and b > 0
            else Decimal("2.0")  # Fallback
            for b in df_chunk["payoff_ratio"].to_list()
        ]

        for i in range(len(df_chunk)):
            lot_size_float = 0.0
            margin_required_decimal = DECIMAL_ZERO
            spread_cost_decimal = DECIMAL_ZERO
            effective_leverage_decimal = DECIMAL_ZERO

            current_timestamp = timestamps_chunk[i]
            try:
                current_timestamp_dt = current_timestamp.replace(tzinfo=dt.timezone.utc)
                current_timestamp_int = int(
                    current_timestamp_dt.timestamp() * 1_000_000
                )
            except Exception:
                current_timestamp_int = int(current_timestamp.timestamp() * 1_000_000)

            current_price_float = close_prices_chunk[i]
            atr_value_float = atr_values_chunk[i]
            sl_multiplier_float = sl_multipliers_chunk[i]
            pt_multiplier_float = pt_multipliers_chunk[i]
            direction_int = directions_chunk[i]

            # --- 日足データ ---
            current_sma_200 = sma_200_chunk[i] if has_daily_data else None
            current_prev_change = prev_change_chunk[i] if has_daily_data else None

            if (
                current_price_float is None
                or not np.isfinite(current_price_float)
                or current_price_float <= 0
            ):
                equity_values_chunk.append(current_capital)
                continue
            current_price_decimal = Decimal(str(current_price_float))

            if current_capital < DECIMAL_MIN_CAPITAL:
                equity_values_chunk.append(DECIMAL_ZERO)
                continue

            # ポジション整理
            active_exit_times = [
                t for t in active_exit_times if t > current_timestamp_int
            ]
            if len(active_exit_times) >= MAX_POSITIONS:
                equity_values_chunk.append(current_capital)
                continue

            # --- ★★★ ハードフィルタリング判定 ★★★ ---
            is_filtered_out = False

            # Filter A: Trend (Price < SMA200 -> Skip)
            if filter_mode in ["trend", "both"]:
                if current_sma_200 is not None and np.isfinite(current_sma_200):
                    if current_price_float < current_sma_200:
                        is_filtered_out = True

            # Filter B: Crash (Prev Change < -2% -> Skip)
            if not is_filtered_out and filter_mode in ["crash", "both"]:
                if current_prev_change is not None and np.isfinite(current_prev_change):
                    if current_prev_change < -0.02:  # -2%
                        is_filtered_out = True
            # -----------------------------------------

            p_float = p_m2_calibrated[i]
            b = payoff_ratios_chunk[i]
            p_decimal = Decimal(str(p_float))
            q = DECIMAL_ONE - p_decimal
            kelly_f_star = (b * p_decimal - q) / b if b > DECIMAL_ZERO else DECIMAL_ZERO
            f_star = (
                kelly_f_star.copy_abs().max(DECIMAL_ZERO)
                if kelly_f_star.is_finite()
                else DECIMAL_ZERO
            )

            # ★ 修正: should_trade 判定に is_filtered_out を追加
            should_trade = (
                (f_star > DECIMAL_F_STAR_THRESHOLD)
                and (p_float > self.config.m2_proba_threshold)
                and (not is_filtered_out)
            )

            if should_trade:
                base_bet_fraction = f_star * DECIMAL_KELLY_FRACTION
                effective_bet_fraction = (
                    base_bet_fraction.copy_abs()
                    .min(DECIMAL_MAX_RISK)
                    .copy_abs()
                    .min(DECIMAL_ONE)
                )
                capped_bet_fraction = effective_bet_fraction

                if effective_bet_fraction > DECIMAL_ZERO:
                    risk_amount_decimal = current_capital * effective_bet_fraction

                    if (
                        atr_value_float is None
                        or not np.isfinite(atr_value_float)
                        or atr_value_float <= 0
                        or sl_multiplier_float is None
                        or not np.isfinite(sl_multiplier_float)
                        or sl_multiplier_float <= 0
                    ):
                        desired_lot_size_decimal = DECIMAL_ZERO
                        stop_loss_currency_per_lot = DECIMAL_ZERO
                    else:
                        dynamic_sl_PRICE_decimal = Decimal(
                            str(atr_value_float)
                        ) * Decimal(str(sl_multiplier_float))

                        stop_loss_currency_per_lot = (
                            dynamic_sl_PRICE_decimal * DECIMAL_CONTRACT_SIZE
                        )

                        if stop_loss_currency_per_lot > DECIMAL_ZERO:
                            desired_lot_size_decimal = (
                                risk_amount_decimal / stop_loss_currency_per_lot
                            )
                        else:
                            desired_lot_size_decimal = DECIMAL_ZERO

                    effective_leverage_decimal = self._get_effective_leverage(
                        current_capital
                    )

                    if effective_leverage_decimal > DECIMAL_ZERO:
                        max_lot_by_margin = (
                            current_capital * effective_leverage_decimal
                        ) / (current_price_decimal * DECIMAL_CONTRACT_SIZE)
                        max_lot_by_margin = max_lot_by_margin.copy_abs().max(
                            DECIMAL_ZERO
                        )
                    else:
                        max_lot_by_margin = DECIMAL_ZERO

                    max_lot_allowed_by_broker = self._get_max_lot_allowed(
                        current_timestamp
                    )

                    final_lot_size_decimal = desired_lot_size_decimal.min(
                        max_lot_by_margin
                    ).min(max_lot_allowed_by_broker)
                    final_lot_size_decimal = final_lot_size_decimal.copy_abs().max(
                        DECIMAL_ZERO
                    )

                    if (final_lot_size_decimal > DECIMAL_ZERO) and (
                        final_lot_size_decimal < DECIMAL_MIN_LOT_SIZE
                    ):
                        final_lot_size_decimal = DECIMAL_MIN_LOT_SIZE

                    if final_lot_size_decimal > DECIMAL_ZERO:
                        margin_required_decimal = (
                            current_price_decimal
                            * final_lot_size_decimal
                            * DECIMAL_CONTRACT_SIZE
                        ) / effective_leverage_decimal

                        spread_pips_decimal = Decimal(str(self.config.spread_pips))
                        spread_cost_decimal = (
                            final_lot_size_decimal
                            * spread_pips_decimal
                            * DECIMAL_VALUE_PER_PIP
                        )

                        if spread_cost_decimal >= current_capital:
                            should_trade = False
                            pnl = DECIMAL_ZERO
                            actual_label = labels_chunk[i]
                        else:
                            capital_before_pnl = current_capital - spread_cost_decimal

                            if stop_loss_currency_per_lot > DECIMAL_ZERO:
                                risk_amount_final = (
                                    final_lot_size_decimal * stop_loss_currency_per_lot
                                )
                            else:
                                risk_amount_final = DECIMAL_ZERO

                            pnl = DECIMAL_ZERO
                            actual_label = labels_chunk[i]
                            if actual_label == 1:
                                pnl = risk_amount_final * b
                            elif actual_label == -1:
                                pnl = risk_amount_final.copy_negate()

                            next_capital = capital_before_pnl + pnl
                            lot_size_float = (
                                float(final_lot_size_decimal)
                                if final_lot_size_decimal.is_finite()
                                else 0.0
                            )

                            if not next_capital.is_finite():
                                current_capital = DECIMAL_ZERO
                            else:
                                current_capital = next_capital

                            current_t1_dt = t1_chunk[i].replace(tzinfo=dt.timezone.utc)
                            new_exit_time = int(current_t1_dt.timestamp() * 1_000_000)
                            active_exit_times.append(new_exit_time)

                    else:
                        should_trade = False
                        pnl = DECIMAL_ZERO
                        actual_label = labels_chunk[i]

                else:
                    should_trade = False
                    pnl = DECIMAL_ZERO
                    actual_label = labels_chunk[i]

            else:
                should_trade = False
                pnl = DECIMAL_ZERO
                actual_label = labels_chunk[i]

            equity_values_chunk.append(current_capital)

            if should_trade and final_lot_size_decimal > DECIMAL_ZERO:
                trade_log_chunk.append(
                    {
                        "timestamp": current_timestamp,
                        "pnl": pnl,
                        "capital_after_trade": current_capital,
                        "m2_calibrated_proba": p_float,
                        "payoff_ratio": b,
                        "kelly_f_star": kelly_f_star,
                        "f_star": f_star,
                        "base_bet_fraction": base_bet_fraction,
                        "capped_bet_fraction": capped_bet_fraction,
                        "effective_bet_fraction": effective_bet_fraction,
                        "label": actual_label,
                        "lot_size": lot_size_float,
                        "atr_value": atr_value_float
                        if np.isfinite(atr_value_float)
                        else None,
                        "sl_multiplier": sl_multiplier_float
                        if np.isfinite(sl_multiplier_float)
                        else None,
                        "pt_multiplier": pt_multiplier_float
                        if np.isfinite(pt_multiplier_float)
                        else None,
                        "direction": direction_int,
                        "effective_leverage": float(effective_leverage_decimal),
                        "margin_required": margin_required_decimal,
                        "spread_cost": spread_cost_decimal,
                        "close_price": current_price_decimal,
                    }
                )

        results_chunk_df = pl.DataFrame(
            {
                "timestamp": timestamps_chunk,
                "equity": pl.Series("equity", equity_values_chunk, dtype=pl.Object),
            }
        )

        trade_log_schema = {
            "timestamp": pl.Datetime,
            "pnl": pl.Object,
            "capital_after_trade": pl.Object,
            "m2_calibrated_proba": pl.Float64,
            "payoff_ratio": pl.Object,
            "kelly_f_star": pl.Object,
            "f_star": pl.Object,
            "base_bet_fraction": pl.Object,
            "capped_bet_fraction": pl.Object,
            "effective_bet_fraction": pl.Object,
            "label": pl.Int64,
            "lot_size": pl.Float64,
            "atr_value": pl.Float64,
            "sl_multiplier": pl.Float32,
            "pt_multiplier": pl.Float32,
            "direction": pl.Int8,
            "effective_leverage": pl.Float32,
            "margin_required": pl.Object,
            "spread_cost": pl.Object,
            "close_price": pl.Object,
        }

        if trade_log_chunk:
            try:
                trade_log_data = {
                    key: [d.get(key) for d in trade_log_chunk]
                    for key in trade_log_schema.keys()
                }
            except Exception as e:
                logging.error(f"Error creating trade_log_data dict: {e}")
                trade_log_chunk_df = pl.DataFrame(schema=trade_log_schema)
                return (
                    results_chunk_df,
                    trade_log_chunk_df,
                    current_capital,
                    active_exit_times,
                )

            series_dict = {}
            for key, dtype in trade_log_schema.items():
                series_dict[key] = pl.Series(key, trade_log_data.get(key), dtype=dtype)
            trade_log_chunk_df = pl.DataFrame(series_dict)
        else:
            trade_log_chunk_df = pl.DataFrame(schema=trade_log_schema)

        return results_chunk_df, trade_log_chunk_df, current_capital, active_exit_times

    def _analyze_and_report_comparison(self, results_dict: Dict[str, Any]):
        logging.info("Analyzing Multi-Scenario results...")

        # 色定義
        colors = {
            "Original": "black",
            "Trend_Filter": "blue",
            "Crash_Filter": "orange",
            "Combined": "green",
        }

        summary_stats = []
        DECIMAL_ONE = Decimal("1.0")
        DECIMAL_ZERO = Decimal("0.0")

        # グラフ準備
        sns.set_style("darkgrid")
        fig, ax = plt.subplots(figsize=(15, 8))

        # 全てのシナリオをループ
        for name, data in results_dict.items():
            equity_df = data["equity_df"]
            trade_log = data["trade_log"]
            final_cap = data["final_capital"]

            # --- 基本統計計算 ---
            initial_cap = Decimal(str(self.config.initial_capital))
            total_return = (
                (final_cap / initial_cap - DECIMAL_ONE)
                if initial_cap > 0
                else DECIMAL_ZERO
            )

            # Profit Factor
            profit_factor = 0.0
            if not trade_log.is_empty():
                pnl_float = trade_log["pnl"].map_elements(
                    lambda x: float(x) if x is not None else 0.0,
                    return_dtype=pl.Float64,
                )
                win_sum = pnl_float.filter(pnl_float > 0).sum()
                lose_sum = pnl_float.filter(pnl_float < 0).sum()
                if lose_sum != 0:
                    profit_factor = abs(win_sum / lose_sum)
                elif win_sum > 0:
                    profit_factor = float("inf")

            # Drawdown (Polars optimized)
            max_drawdown = 0.0
            if not equity_df.is_empty():
                eq_float = (
                    equity_df["equity"]
                    .map_elements(
                        lambda x: float(x) if x is not None else np.nan,
                        return_dtype=pl.Float64,
                    )
                    .fill_null(strategy="forward")
                )
                roll_max = eq_float.cum_max()

                # 0除算回避
                dd = (eq_float - roll_max) / roll_max.fill_nan(1.0).replace(0.0, 1.0)

                # --- ★★★ [修正箇所] all_null() エラーの修正 ★★★ ---
                # null_count() と len() を比較して、すべてがNullでない場合のみ計算
                if dd.null_count() < dd.len():
                    max_drawdown = dd.min()  # ドローダウンは負の値なので最小値をとる
                    if max_drawdown is None or np.isnan(max_drawdown):
                        max_drawdown = 0.0
                else:
                    max_drawdown = 0.0
                # -----------------------------------------------

                # Plot
                timestamps = equity_df["timestamp"].to_list()
                eq_list = eq_float.to_list()
                ax.plot(
                    timestamps,
                    eq_list,
                    label=f"{name} (Ret: {total_return * 100:.1f}%)",
                    color=colors.get(name, "gray"),
                    alpha=0.8,
                )

            summary_stats.append(
                {
                    "Scenario": name,
                    "Trades": len(trade_log),
                    "Final Capital": float(final_cap),
                    "Total Return": float(total_return) * 100,
                    "Profit Factor": profit_factor,
                    "Max DD": max_drawdown * 100,
                }
            )

        # グラフ仕上げ
        ax.set_title("Equity Curve Comparison: Hard Filters", fontsize=16)
        ax.set_ylabel("Equity")
        ax.legend()
        ax.grid(True)
        try:
            ax.ticklabel_format(style="plain", axis="y")
        except:
            pass

        plt.tight_layout()
        EQUITY_CURVE_PATH.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(EQUITY_CURVE_PATH)  # 上書き保存
        logging.info(f"Comparison chart saved to {EQUITY_CURVE_PATH}")
        plt.close(fig)

        # テーブル出力
        print("\n" + "=" * 80)
        print(f"    Scenario Comparison Report")
        print("=" * 80)
        # ヘッダー
        print(
            f"{'Scenario':<15} | {'Trades':<8} | {'Final Cap':<12} | {'Return %':<10} | {'PF':<6} | {'Max DD %':<10}"
        )
        print("-" * 80)
        for stat in summary_stats:
            print(
                f"{stat['Scenario']:<15} | {stat['Trades']:<8} | {stat['Final Capital']:,.2f} | {stat['Total Return']:>8.2f} % | {stat['Profit Factor']:>6.2f} | {stat['Max DD']:>8.2f} %"
            )
        print("=" * 80 + "\n")


if __name__ == "__main__":
    default_config = BacktestConfig()

    parser = argparse.ArgumentParser(
        description="Project Forge Backtest Simulator (Multi-Scenario: Hard Filters)"
    )

    parser.add_argument(
        "--kelly",
        type=float,
        default=default_config.kelly_fraction,
        help=f"Kelly fraction. Default: {default_config.kelly_fraction}",
    )
    parser.add_argument(
        "--base-leverage",
        type=float,
        default=default_config.base_leverage,
        dest="base_leverage",
        help=f"Base leverage setting. Default: {default_config.base_leverage}",
    )
    parser.add_argument(
        "--max-risk",
        type=float,
        default=default_config.max_risk_per_trade,
        help=f"Max risk per trade. Default: {default_config.max_risk_per_trade}",
    )
    parser.add_argument(
        "--fstar-th",
        type=float,
        default=default_config.f_star_threshold,
        dest="fstar_th",
        help=f"Min f_star threshold. Default: {default_config.f_star_threshold}",
    )
    parser.add_argument(
        "--m2-th",
        type=float,
        default=default_config.m2_proba_threshold,
        dest="m2_th",
        help=f"Min M2 prob threshold. Default: {default_config.m2_proba_threshold}",
    )
    parser.add_argument("--oof", action="store_true", help="Run in OOF mode.")
    parser.add_argument(
        "--min-capital",
        type=float,
        default=default_config.min_capital_threshold,
        dest="min_capital",
        help=f"Min capital threshold. Default: {default_config.min_capital_threshold}",
    )
    parser.add_argument(
        "--value-per-pip",
        type=float,
        default=default_config.value_per_pip,
        dest="value_per_pip",
        help=f"(ASSUMPTION) Value per lot per pip. Default: {default_config.value_per_pip}",
    )
    parser.add_argument(
        "--spread-pips",
        type=float,
        default=default_config.spread_pips,
        dest="spread_pips",
        help=f"Spread in pips for cost calculation. Default: {default_config.spread_pips}",
    )
    parser.add_argument(
        "--test",
        type=int,
        default=default_config.test_limit_partitions,
        metavar="N",
        dest="test_limit_partitions",
        help=f"Limit to first N partitions. Default: {default_config.test_limit_partitions} (all)",
    )
    parser.add_argument(
        "--max-positions",
        type=int,
        default=default_config.max_positions,
        dest="max_positions",
        help=f"Max concurrent positions. Default: {default_config.max_positions}",
    )
    # --- ★★★ 追加: 日足データのパス引数 ★★★ ---
    parser.add_argument(
        "--daily-data",
        type=Path,
        default=default_config.daily_data_path,
        dest="daily_data_path",
        help=f"Path to daily OHLC parquet. Default: {default_config.daily_data_path}",
    )
    # ---------------------------------------------

    args = parser.parse_args()

    config = BacktestConfig(
        kelly_fraction=args.kelly,
        base_leverage=args.base_leverage,
        max_risk_per_trade=args.max_risk,
        f_star_threshold=args.fstar_th,
        m2_proba_threshold=args.m2_th,
        test_limit_partitions=args.test_limit_partitions,
        oof_mode=args.oof,
        min_capital_threshold=args.min_capital,
        value_per_pip=args.value_per_pip,
        spread_pips=args.spread_pips,
        max_positions=args.max_positions,
        daily_data_path=args.daily_data_path,  # ★ 追加
    )

    if not (0 < config.max_risk_per_trade <= 1.0):
        parser.error("--max-risk must be between 0 (exclusive) and 1.0 (inclusive).")
    if config.base_leverage < 1.0:
        parser.error("--base-leverage must be >= 1.0.")
    if config.value_per_pip <= 0:
        parser.error("--value-per-pip must be greater than 0.")
    if config.spread_pips < 0:
        parser.error("--spread-pips cannot be negative.")

    simulator = BacktestSimulator(config)
    simulator.run()
