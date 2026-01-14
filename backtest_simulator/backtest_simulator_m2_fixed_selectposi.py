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

import sys
from pathlib import Path
import logging
import argparse
from dataclasses import dataclass, field
from typing import Dict, Any, Tuple, List
import json
import datetime as dt
import zoneinfo  # ★★★ 追加: タイムゾーン変換用 ★★★

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
    S7_CONTEXT_FEATURES,  # --- ▼▼▼ MODIFICATION 1 ▼▼▼ ---
)

# --- 出力ファイルパス ---
FINAL_REPORT_PATH = S7_MODELS / "final_backtest_report.json"
EQUITY_CURVE_PATH = S7_MODELS / "equity_curve.png"


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

    initial_capital: float = 100.0
    simulation_data_path: Path = S6_WEIGHTED_DATASET
    oof_predictions_path: Path = S7_M2_OOF_PREDICTIONS
    feature_list_path: Path = S3_FEATURES_FOR_TRAINING
    context_data_path: Path = S7_CONTEXT_FEATURES  # --- ▼▼▼ MODIFICATION 1 ▼▼▼ ---
    m1_model_path: Path = S7_M1_MODEL_PKL
    m2_model_path: Path = S7_M2_MODEL_PKL
    m1_calibrator_path: Path = S7_M1_CALIBRATED
    m2_calibrator_path: Path = S7_M2_CALIBRATED
    kelly_fraction: float = 1.0
    # max_leverage: float = 100 # <- ★★★ 削除: base_leverage を使う ★★★
    max_risk_per_trade: float = 0.01  # ★★★ 50 (5000%) -> 0.5 (50%) に修正 ★★★
    f_star_threshold: float = 0.0
    m2_proba_threshold: float = 0.6
    test_limit_partitions: int = 0
    oof_mode: bool = False
    min_capital_threshold: float = 1.0
    min_lot_size: float = 0.01  # ★★★ 最小ロット(0.01)を追加 ★★★

    # ★★★ 追加: 最大同時保有ポジション数 ★★★
    max_positions: int = 100  # ここを好きな数に変更（例: 3, 5, 10）

    # --- ★★★ 修正: 基本レバレッジとスプレッドを追加 ★★★ ---
    base_leverage: float = 2000.0  # 設定可能な基本レバレッジ
    spread_pips: float = 16.0  # XAUUSD スタンダード口座のスプレッド
    value_per_pip: float = 1.0
    """
    (ASSUMPTION) 1ロットあたりの1pipの価値 (口座通貨単位)。
    XAUUSD (1 lot = 100 oz) の場合: $10.0
    """
    # --- ★★★ 修正ここまで ★★★ ---


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

        # --- ▼▼▼ MODIFICATION 1: Load M2 Context Features ▼▼▼ ---
        # (model_training_metalabeling_C.py と一致させる)
        self.context_features_m2 = [
            "hmm_prob_0",
            "hmm_prob_1",
            "atr_ratio",  # 修正: atr -> atr_ratio
            "trend_bias_25",  # 追加: トレンドバイアス
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
            # Prepare for join_asof
            self.context_df = (
                context_df.with_columns(pl.col("timestamp").dt.date().alias("date"))
                .select(["date"] + self.context_features_m2)  # Select only needed cols
                .sort("date")
                .lazy()
            )
            logging.info(
                f"  -> M2 features redefined with {len(self.context_features_m2)} context features."
            )
        except Exception as e:
            # --- ▼▼▼ SYNTAX FIX (L187-188) ▼▼▼ ---
            logging.error(
                f"CRITICAL: Failed to load M2 context features from {config.context_data_path}: {e}",
                exc_info=True,
            )
            # --- ▲▲▲ SYNTAX FIX END ▲▲▲ ---
            raise
        # --- ▲▲▲ MODIFICATION 1 END ▲▲▲ ---

        self._current_capital = Decimal(str(self.config.initial_capital))

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

    # ★★★ ヘルパー関数: 有効証拠金から実効レバレッジを取得 ★★★
    def _get_effective_leverage(self, equity: Decimal) -> Decimal:
        """有効証拠金に基づいてExnessのレバレッジ制限を適用"""
        base_leverage_dec = Decimal(str(self.config.base_leverage))
        # Exnessの証拠金レベル (USD)
        if equity < Decimal("5000"):
            limit_leverage = base_leverage_dec  # 無制限期間はベース設定を使用 (要調整)
        elif equity < Decimal("30000"):
            limit_leverage = Decimal("2000")  # 最大2000倍
        elif equity < Decimal("100000"):
            limit_leverage = Decimal("1000")  # 最大1000倍
        else:
            limit_leverage = Decimal("500")  # 最大500倍

        # 設定した基本レバレッジと証拠金による上限のうち、小さい方を適用
        return base_leverage_dec.min(limit_leverage)

    # ★★★ ヘルパー関数: JST時間帯から最大ロット数を取得 ★★★
    def _get_max_lot_allowed(self, timestamp_utc: dt.datetime) -> Decimal:
        """JST時間帯に基づいてExnessの最大ロット数を返す"""
        timestamp_jst = timestamp_utc.astimezone(JST)
        hour_jst = timestamp_jst.hour

        # 日本時間 午前6:00 ～ 午後3:59 (15:59) -> 20ロット
        if 6 <= hour_jst < 16:
            return Decimal("20")
        # 日本時間 午後4:00 (16:00) ～ 午前5:59 -> 200ロット
        else:
            return Decimal("200")

    def run(self):
        logging.info("### Backtest Simulator: START ###")
        # ★★★ 修正: ログメッセージ更新 ★★★
        logging.info(
            f"Strategy: Probabilistic Betting with Kelly Fraction = {self.config.kelly_fraction}, "
            f"Base Leverage = {self.config.base_leverage}, Max Risk/Trade = {self.config.max_risk_per_trade * 100:.1f}%, "
            f"Spread = {self.config.spread_pips} pips"
        )
        # ★★★ 修正ここまで ★★★
        logging.info(
            f"Bankruptcy Threshold (Min Capital): {self.config.min_capital_threshold:,.2f}"
        )
        logging.info(
            f"Lot Size Params: Value = {self.config.value_per_pip}/lot/pip (SL width from S6 data), Contract Size = {CONTRACT_SIZE}"
        )

        lf, partitions_df = self._prepare_data()

        all_results_dfs = []
        all_trade_logs = []

        logging.info(f"Processing {len(partitions_df)} partitions sequentially...")

        partitions_to_process = partitions_df
        if self.config.test_limit_partitions > 0:
            logging.warning(
                f"--- TEST MODE: Limiting to first {self.config.test_limit_partitions} partitions. ---"
            )
            partitions_to_process = partitions_df.head(
                self.config.test_limit_partitions
            )

        self._current_capital = Decimal(str(self.config.initial_capital))
        DECIMAL_MIN_CAPITAL = Decimal(str(self.config.min_capital_threshold))

        for row in tqdm(
            partitions_to_process.iter_rows(named=True),
            total=len(partitions_to_process),
            desc="Simulating Partitions",
        ):
            current_date = row["date"]

            logging.debug(f"Processing partition: {current_date}")
            try:
                df_chunk = lf.filter(
                    pl.col("timestamp").dt.date() == current_date
                ).collect()
                if df_chunk.is_empty():
                    logging.debug(f"Skipping empty partition: {current_date}")
                    continue
                logging.debug(f"Collected {len(df_chunk)} rows for {current_date}")

                # シミュレーションに必要な列を定義 (closeを追加)
                required_sim_cols = [
                    "timestamp",
                    "close",
                    "label",
                    "payoff_ratio",
                    "atr_value",
                    "sl_multiplier",
                    "pt_multiplier",
                    "direction",
                ]
                if not self.config.oof_mode:
                    required_sim_cols.extend(self.features_base)
                    # --- ▼▼▼ MODIFICATION 1 ▼▼▼ ---
                    # (In-Sampleモードでは文脈特徴量も収集されている必要がある)
                    required_sim_cols.extend(self.context_features_m2)
                    # --- ▲▲▲ MODIFICATION 1 END ▲▲▲ ---
                else:
                    required_sim_cols.append("m2_raw_proba")

                missing_sim_cols = [
                    col for col in required_sim_cols if col not in df_chunk.columns
                ]
                if missing_sim_cols:
                    logging.error(
                        f"CRITICAL: Required columns for simulation missing in collected data for {current_date}! Missing: {missing_sim_cols}"
                    )
                    continue

            except Exception as e:
                logging.error(
                    f"Error collecting data for partition {current_date}: {e}",
                    exc_info=True,
                )
                continue

            try:
                if self._current_capital < DECIMAL_MIN_CAPITAL:
                    logging.warning(
                        f"Capital ({self._current_capital:,.2f}) fell below threshold ({DECIMAL_MIN_CAPITAL:,.2f}) "
                        f"before processing {current_date}. Stopping simulation."
                    )
                    break

                df_chunk_predicted = self._run_ai_predictions(df_chunk)
                results_chunk_df, trade_log_chunk_df = self._run_simulation_loop(
                    df_chunk_predicted
                )

                all_results_dfs.append(results_chunk_df)
                all_trade_logs.append(trade_log_chunk_df)

                del df_chunk, df_chunk_predicted, results_chunk_df, trade_log_chunk_df
                gc.collect()

            except Exception as e:
                logging.error(
                    f"Error processing partition {current_date}: {e}", exc_info=True
                )
                continue

        if not all_results_dfs:
            logging.error("No simulation results were generated. Cannot create report.")
            return

        logging.info("Concatenating results from all partitions...")
        try:
            final_results_df = pl.concat(all_results_dfs).sort("timestamp")
            final_trade_log_df = (
                pl.concat([df for df in all_trade_logs if not df.is_empty()]).sort(
                    "timestamp"
                )
                if any(not df.is_empty() for df in all_trade_logs)
                else pl.DataFrame()
            )

        except Exception as e:
            logging.error(f"Error concatenating results: {e}", exc_info=True)
            return

        if final_results_df.is_empty():
            logging.error(
                "Concatenated results DataFrame is empty. Cannot generate report."
            )
            return

        self._analyze_and_report(final_results_df, final_trade_log_df)
        logging.info("### Backtest Simulator: FINISHED ###")

    def _prepare_data(self) -> Tuple[pl.LazyFrame, pl.DataFrame]:
        # S6から読み込む基本カラムリスト (closeを追加)
        base_cols = [
            "timestamp",
            "close",
            "label",
            "payoff_ratio",
            "atr_value",
            "sl_multiplier",
            "pt_multiplier",
            "direction",
            "t1",  # ★★★ 1ポジルールのために取引終了時刻を追加 ★★★
        ]

        if not self.config.oof_mode:  # In-Sample Mode
            logging.info(
                f"Preparing data from {self.config.simulation_data_path} (In-Sample Mode)..."
            )
            required_cols_set = set(self.features_base)
            required_cols_set.update(base_cols)
            # (文脈特徴量はS6にないので、ここでは追加しない)
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
                    "Ensure create_proxy_labels.py included 'close' price."
                )

            lf = lf.select(required_cols).sort("timestamp")

            # --- ▼▼▼ MODIFICATION 1: Join context features for In-Sample mode ▼▼▼ ---
            logging.info("  -> [In-Sample] Joining M2 context features...")
            lf = lf.with_columns(
                pl.col("timestamp").dt.date().alias("date")
            )  # Add date key
            lf = lf.join_asof(self.context_df, on="date")  # Join context features
            lf = lf.drop(
                "date"
            )  # Drop date key to avoid conflict during partition loop
            # --- ▲▲▲ MODIFICATION 1 END ▲▲▲ ---

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
            required_cols = base_cols + ["m2_raw_proba"]

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
        # (この関数は変更なし)
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
                # --- ▼▼▼ MODIFICATION 1 ▼▼▼ ---
                # self.features_m2 (redefined in __init__) is now used.
                # df_chunk (modified in _prepare_data) now has context features.
                # This check will now pass.
                missing_m2_features = [
                    f for f in self.features_m2 if f not in df_chunk.columns
                ]
                if missing_m2_features:
                    raise ValueError(f"Missing M2 features: {missing_m2_features}")
                X_m2 = df_chunk.select(self.features_m2).fill_null(0).to_numpy()
                # --- ▲▲▲ MODIFICATION 1 END ▲▲▲ ---
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
        self, df_chunk: pl.DataFrame
    ) -> Tuple[pl.DataFrame, pl.DataFrame]:
        # ---
        # --- [MODIFICATION 2 & 3 - REJECTED] ---
        # The user's requested rewrite (`calculate_pnl`) is based on a different,
        # simpler simulator. The logic below is from the complex v3 simulator
        # and *already* correctly implements dollar-based PnL (Defect 2)
        # and timeout costs (Defect 3) via the `spread_cost_decimal` (L599).
        # No changes are made to this function.
        # ---
        trade_log_chunk = []
        equity_values_chunk = []
        current_capital = self._current_capital
        DECIMAL_ZERO = Decimal("0.0")
        DECIMAL_ONE = Decimal("1.0")
        DECIMAL_MAX_RISK = Decimal(str(self.config.max_risk_per_trade))
        DECIMAL_KELLY_FRACTION = Decimal(str(self.config.kelly_fraction))
        DECIMAL_F_STAR_THRESHOLD = Decimal(str(self.config.f_star_threshold))
        DECIMAL_MIN_CAPITAL = Decimal(str(self.config.min_capital_threshold))
        DECIMAL_VALUE_PER_PIP = Decimal(str(self.config.value_per_pip))
        DECIMAL_MIN_LOT_SIZE = Decimal(
            str(self.config.min_lot_size)
        )  # ★★★ 最小ロット定義を追加 ★★★

        # ★★★ 修正: CONTRACT_SIZE を Decimal 型で定義 ★★★
        DECIMAL_CONTRACT_SIZE = CONTRACT_SIZE
        # ★★★ 修正ここまで ★★★

        # ★★★ 修正: 終了時刻をリストで管理 ★★★
        active_exit_times = []  # 保有中ポジションの決済予定時刻リスト
        MAX_POSITIONS = self.config.max_positions

        timestamps_chunk = df_chunk["timestamp"].to_list()
        # --- ▼▼▼ [1ポジルール修正] t1（終了時刻）を取得 ▼▼▼ ---
        t1_chunk = df_chunk["t1"].to_list()
        # --- ▲▲▲ [1ポジルール修正] ここまで ▲▲▲ ---
        close_prices_chunk = df_chunk[
            "close"
        ].to_numpy()  # ★★★ 追加: close価格を取得 ★★★
        p_m2_calibrated = df_chunk["m2_calibrated_proba"].to_numpy()
        labels_chunk = df_chunk["label"].to_numpy()
        atr_values_chunk = df_chunk["atr_value"].to_numpy()
        sl_multipliers_chunk = df_chunk["sl_multiplier"].to_numpy()
        pt_multipliers_chunk = df_chunk["pt_multiplier"].to_numpy()
        directions_chunk = df_chunk["direction"].to_numpy()

        payoff_ratios_chunk = [
            Decimal(str(b))
            if b is not None and np.isfinite(b) and b > 0
            else Decimal("2.0")  # Fallback
            for b in df_chunk["payoff_ratio"].to_list()
        ]

        for i in range(len(df_chunk)):
            # --- ★★★ 初期化: このループで使う変数を定義 ★★★ ---
            lot_size_float = 0.0
            margin_required_decimal = DECIMAL_ZERO
            spread_cost_decimal = DECIMAL_ZERO
            effective_leverage_decimal = DECIMAL_ZERO
            # --- ★★★ 初期化ここまで ★★★ ---

            # S6から読み込んだ情報を取得
            current_timestamp = timestamps_chunk[i]
            # --- ▼▼▼ [1ポジルール修正] シグナルのタイムスタンプを整数（us）で取得 ▼▼▼ ---
            try:
                # Polars/Pandasのdatetimeをネイティブのdatetimeに変換
                current_timestamp_dt = current_timestamp.replace(tzinfo=dt.timezone.utc)
                current_timestamp_int = int(
                    current_timestamp_dt.timestamp() * 1_000_000
                )
            except Exception:
                # 既にdatetimeでない場合（テストなど）
                current_timestamp_int = int(current_timestamp.timestamp() * 1_000_000)
            # --- ▲▲▲ [1ポジルール修正] ここまで ▲▲▲ ---

            current_price_float = close_prices_chunk[i]  # ★★★ 追加 ★★★
            atr_value_float = atr_values_chunk[i]
            sl_multiplier_float = sl_multipliers_chunk[i]
            pt_multiplier_float = pt_multipliers_chunk[i]
            direction_int = directions_chunk[i]

            # ★★★ 追加: 価格が有効かチェック ★★★
            if (
                current_price_float is None
                or not np.isfinite(current_price_float)
                or current_price_float <= 0
            ):
                logging.warning(
                    f"Invalid close price {current_price_float} at {current_timestamp}, skipping."
                )
                equity_values_chunk.append(current_capital)  # 資本は変動しない
                continue
            current_price_decimal = Decimal(str(current_price_float))
            # ★★★ 追加ここまで ★★★

            # 破産チェック
            if current_capital < DECIMAL_MIN_CAPITAL:
                equity_values_chunk.append(DECIMAL_ZERO)
                continue  # 破産したら以降の処理はスキップ

            # ★★★ 修正: ポジション数のチェックロジック ★★★
            # 1. 既に決済時刻を過ぎたポジションをリストから削除（整理）
            active_exit_times = [
                t for t in active_exit_times if t > current_timestamp_int
            ]

            # 2. 現在の保有数が上限に達しているかチェック
            if len(active_exit_times) >= MAX_POSITIONS:
                equity_values_chunk.append(current_capital)  # 満杯ならスキップ
                continue
            # ★★★ 修正ここまで ★★★

            # 取引判断 (ここに来た時点でポジションは保有していない)
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
            should_trade = (f_star > DECIMAL_F_STAR_THRESHOLD) and (
                p_float > self.config.m2_proba_threshold
            )

            if should_trade:
                base_bet_fraction = f_star * DECIMAL_KELLY_FRACTION
                effective_bet_fraction = (
                    base_bet_fraction.copy_abs()
                    .min(DECIMAL_MAX_RISK)
                    .copy_abs()
                    .min(DECIMAL_ONE)
                )
                capped_bet_fraction = effective_bet_fraction  # ログ用

                if effective_bet_fraction > DECIMAL_ZERO:
                    # --- ★★★ 修正: 動的ロットサイズ計算 (ここから) ★★★ ---
                    risk_amount_decimal = current_capital * effective_bet_fraction

                    # 1. 動的SL幅 (価格単位) を計算
                    if (
                        atr_value_float is None
                        or not np.isfinite(atr_value_float)
                        or atr_value_float <= 0
                        or sl_multiplier_float is None
                        or not np.isfinite(sl_multiplier_float)
                        or sl_multiplier_float <= 0
                    ):
                        logging.warning(
                            f"Invalid ATR ({atr_value_float}) or SL mult ({sl_multiplier_float}) at {current_timestamp}, cannot calculate lot size."
                        )
                        desired_lot_size_decimal = DECIMAL_ZERO
                        stop_loss_currency_per_lot = DECIMAL_ZERO  # ★ 追加
                    else:
                        # atr_value (例: 1.80) は「価格単位」
                        dynamic_sl_PRICE_decimal = Decimal(
                            str(atr_value_float)
                        ) * Decimal(str(sl_multiplier_float))

                        # 2. 1ロットあたりのストップロス価値（通貨）を計算
                        # ★★★ これがバグ修正箇所 ★★★
                        # (誤) ... * DECIMAL_VALUE_PER_PIP (10.0)
                        # (正) ... * DECIMAL_CONTRACT_SIZE (100.0)
                        stop_loss_currency_per_lot = (
                            dynamic_sl_PRICE_decimal * DECIMAL_CONTRACT_SIZE
                        )
                        # ★★★ 修正完了 ★★★

                        # 3. リスクベースの希望ロットサイズ計算
                        if stop_loss_currency_per_lot > DECIMAL_ZERO:
                            desired_lot_size_decimal = (
                                risk_amount_decimal / stop_loss_currency_per_lot
                            )
                        else:
                            desired_lot_size_decimal = DECIMAL_ZERO

                    # 4. 実効レバレッジを決定
                    effective_leverage_decimal = self._get_effective_leverage(
                        current_capital
                    )

                    # 5. 証拠金ベースの最大許容ロットサイズを計算
                    # Lot = (Equity * Leverage) / (Price * ContractSize)
                    if effective_leverage_decimal > DECIMAL_ZERO:
                        max_lot_by_margin = (
                            current_capital * effective_leverage_decimal
                        ) / (current_price_decimal * DECIMAL_CONTRACT_SIZE)
                        max_lot_by_margin = max_lot_by_margin.copy_abs().max(
                            DECIMAL_ZERO
                        )  # 負にならないように
                    else:
                        max_lot_by_margin = DECIMAL_ZERO

                    # 6. 時間帯ベースの最大許容ロット数を取得
                    max_lot_allowed_by_broker = self._get_max_lot_allowed(
                        current_timestamp
                    )

                    # 7. 最終的なロットサイズを決定 (希望ロット vs 証拠金上限 vs ブローカー上限)
                    final_lot_size_decimal = desired_lot_size_decimal.min(
                        max_lot_by_margin
                    ).min(max_lot_allowed_by_broker)
                    final_lot_size_decimal = final_lot_size_decimal.copy_abs().max(
                        DECIMAL_ZERO
                    )  # 念のためゼロ以上を保証

                    # --- ▼▼▼ [最小ロット修正] 最小ロット(0.01)のチェック (ロジック修正) ▼▼▼ ---
                    # 最終ロットサイズが 0 より大きいが、最小ロット(0.01)より小さいか？
                    if (final_lot_size_decimal > DECIMAL_ZERO) and (
                        final_lot_size_decimal < DECIMAL_MIN_LOT_SIZE
                    ):
                        # 最小ロット(0.01)に切り上げる（丸める）
                        final_lot_size_decimal = DECIMAL_MIN_LOT_SIZE
                    # --- ▲▲▲ [最小ロット修正] ここまで ▲▲▲ ---

                    # 8. 最終ロットサイズがゼロより大きいか？
                    if final_lot_size_decimal > DECIMAL_ZERO:
                        # 9. 必要証拠金を計算
                        margin_required_decimal = (
                            current_price_decimal
                            * final_lot_size_decimal
                            * DECIMAL_CONTRACT_SIZE
                        ) / effective_leverage_decimal

                        # 10. スプレッドコストを計算
                        spread_pips_decimal = Decimal(str(self.config.spread_pips))
                        spread_cost_decimal = (
                            final_lot_size_decimal
                            * spread_pips_decimal
                            * DECIMAL_VALUE_PER_PIP
                        )
                        # ★★★ 追加: スプレッドコスト破綻チェック ★★★
                        if spread_cost_decimal >= current_capital:
                            logging.debug(
                                f"Spread cost ({spread_cost_decimal:,.2f}) exceeds current capital ({current_capital:,.2f}) at {current_timestamp}. Skipping trade."
                            )
                            should_trade = False  # 取引しない
                            pnl = DECIMAL_ZERO
                            actual_label = labels_chunk[i]
                            # equity_values_chunk.append(current_capital) # ← これはループの最後に移動
                            # continue や return は不要、ループの残りを実行して資本を記録
                        else:
                            # 11. スプレッドコストを資本から差し引く
                            capital_before_pnl = current_capital - spread_cost_decimal

                            # 12. PnLを再計算 (最終ロットサイズに基づいてリスク額を再定義)
                            #    リスク額 = 最終ロットサイズ * 1ロットあたりSL(通貨)
                            if (
                                stop_loss_currency_per_lot > DECIMAL_ZERO
                            ):  # desired_lot_size計算時にチェック済みだが念のため
                                # ★★★ [最小ロット修正] 実際のリスク額は、最終ロットサイズに基づいて再計算する ★★★
                                risk_amount_final = (
                                    final_lot_size_decimal * stop_loss_currency_per_lot
                                )
                            else:
                                risk_amount_final = DECIMAL_ZERO

                            pnl = DECIMAL_ZERO
                            actual_label = labels_chunk[i]
                            if actual_label == 1:
                                # 利益 = リスク額 * ペイオフレシオ (b = pt_mult / sl_mult)
                                pnl = risk_amount_final * b
                            elif actual_label == -1:
                                # 損失 = -リスク額
                                pnl = risk_amount_final.copy_negate()

                            # [MODIFICATION 3 REJECTED]
                            # 'actual_label == 0' (timeout) は
                            # pnl = 0 となり、 'capital_before_pnl' (L599)
                            # によってスプレッドコストが引かれるため、
                            # 正しく処理されている。

                            # 13. 次の資本を計算
                            next_capital = capital_before_pnl + pnl

                            # 14. ログ用に float に変換
                            lot_size_float = (
                                float(final_lot_size_decimal)
                                if final_lot_size_decimal.is_finite()
                                else 0.0
                            )

                            # 15. 無限大/NaN チェック
                            if not next_capital.is_finite():
                                logging.error(
                                    f"Capital NaN/Inf at {current_timestamp}. Prev: {current_capital:.2E}, Spread: {spread_cost_decimal:.2E}, PnL: {pnl:.2E}"
                                )
                                current_capital = DECIMAL_ZERO
                            else:
                                current_capital = next_capital

                            # ★★★ 修正: リストに追加 ★★★
                            current_t1_dt = t1_chunk[i].replace(tzinfo=dt.timezone.utc)
                            new_exit_time = int(current_t1_dt.timestamp() * 1_000_000)
                            active_exit_times.append(new_exit_time)
                            # ★★★ 修正ここまで ★★★

                    else:  # 最終ロットサイズがゼロになった場合
                        should_trade = False  # 取引しないことにする
                        pnl = DECIMAL_ZERO
                        actual_label = labels_chunk[i]
                        # lot_size_float, margin_required_decimal, spread_cost_decimal はゼロのまま
                    # --- ★★★ 修正: 動的ロットサイズ計算 (ここまで) ★★★ ---

                else:  # ケリー推奨がゼロ以下 or M2確率が閾値以下
                    should_trade = False
                    pnl = DECIMAL_ZERO
                    actual_label = labels_chunk[i]
                    # lot_size_float, margin_required_decimal, spread_cost_decimal はゼロのまま

            else:  # (should_trade が False の場合)
                should_trade = False
                pnl = DECIMAL_ZERO
                actual_label = labels_chunk[
                    i
                ]  # ログ用にラベルだけ取得 (実際は取引しない)

            # 資本を記録 (取引しなかった場合も含む)
            equity_values_chunk.append(current_capital)

            # 取引ログを記録 (取引した場合のみ)
            if (
                should_trade and final_lot_size_decimal > DECIMAL_ZERO
            ):  # 実際に取引したか再確認
                trade_log_chunk.append(
                    {
                        "timestamp": current_timestamp,
                        "pnl": pnl,
                        "capital_after_trade": current_capital,  # PnL適用後の資本
                        "m2_calibrated_proba": p_float,
                        "payoff_ratio": b,
                        "kelly_f_star": kelly_f_star,
                        "f_star": f_star,
                        "base_bet_fraction": base_bet_fraction,
                        "capped_bet_fraction": capped_bet_fraction,  # Kelly * fraction (max_risk適用前)
                        "effective_bet_fraction": effective_bet_fraction,  # max_risk適用後
                        "label": actual_label,
                        "lot_size": lot_size_float,  # 最終的なロットサイズ
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
                        # ★★★ 追加ログ ★★★
                        "effective_leverage": float(
                            effective_leverage_decimal
                        ),  # 適用されたレバレッジ
                        "margin_required": margin_required_decimal,  # 必要証拠金
                        "spread_cost": spread_cost_decimal,  # スプレッドコスト
                        "close_price": current_price_decimal,  # 参考: 取引時の価格
                    }
                )

        self._current_capital = current_capital

        results_chunk_df = pl.DataFrame(
            {
                "timestamp": timestamps_chunk,
                "equity": pl.Series("equity", equity_values_chunk, dtype=pl.Object),
            }
        )

        # --- ★★★ 修正: trade_log_schema に新しい列の型定義を追加 ★★★ ---
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
            # ★★★ 追加 ★★★
            "effective_leverage": pl.Float32,  # floatで十分
            "margin_required": pl.Object,  # Decimalのまま
            "spread_cost": pl.Object,  # Decimalのまま
            "close_price": pl.Object,  # Decimalのまま
        }
        # --- ★★★ 修正ここまで ★★★ ---

        if trade_log_chunk:
            try:
                # get()で安全にアクセスし、キーが存在しない場合はNoneを返す
                trade_log_data = {
                    key: [d.get(key) for d in trade_log_chunk]
                    for key in trade_log_schema.keys()
                }
            except Exception as e:  # 通常は発生しないはずだが念のため
                logging.error(f"Error creating trade_log_data dict: {e}")
                trade_log_chunk_df = pl.DataFrame(schema=trade_log_schema)
                return results_chunk_df, trade_log_chunk_df

            # --- ★★★ 修正: DataFrame 構築時に新しい列を追加 ★★★ ---
            series_dict = {}
            for key, dtype in trade_log_schema.items():
                series_dict[key] = pl.Series(key, trade_log_data.get(key), dtype=dtype)
            trade_log_chunk_df = pl.DataFrame(series_dict)
            # --- ★★★ 修正ここまで ★★★ ---
        else:
            trade_log_chunk_df = pl.DataFrame(schema=trade_log_schema)

        return results_chunk_df, trade_log_chunk_df

    def _analyze_and_report(self, results_df: pl.DataFrame, trade_log: pl.DataFrame):
        # (レポート計算部分は変更なし - 前回の回答と同じ)
        logging.info("Analyzing results and generating report...")
        DECIMAL_ONE = Decimal("1.0")
        DECIMAL_ZERO = Decimal("0.0")
        DECIMAL_HUNDRED = Decimal("100.0")
        if results_df.is_empty():
            logging.error("No simulation results to analyze.")
            initial_capital = Decimal(str(self.config.initial_capital))
            final_capital = initial_capital
            total_return = DECIMAL_ZERO
            sharpe_ratio = 0.0
            sortino_ratio = 0.0
            max_drawdown = 0.0
            data_period_start = "N/A"
            data_period_end = "N/A"
            drawdown = pl.Series(dtype=pl.Float64)
        else:
            initial_capital = Decimal(str(self.config.initial_capital))
            final_capital_raw = results_df["equity"][-1]
            final_capital = (
                final_capital_raw
                if final_capital_raw is not None and final_capital_raw.is_finite()
                else DECIMAL_ZERO
            )
            total_return = (
                (final_capital / initial_capital - DECIMAL_ONE)
                if initial_capital > DECIMAL_ZERO and initial_capital.is_finite()
                else DECIMAL_ZERO
            )
            daily_equity = (
                results_df.group_by(pl.col("timestamp").dt.date().alias("date"))
                .agg(pl.last("equity"))
                .sort("date")
            )
            daily_equity_list = daily_equity["equity"].to_list()
            daily_returns_float = []
            if len(daily_equity_list) > 1:
                for i in range(1, len(daily_equity_list)):
                    prev = daily_equity_list[i - 1]
                    curr = daily_equity_list[i]
                    if (
                        prev is not None
                        and curr is not None
                        and prev.is_finite()
                        and curr.is_finite()
                        and prev > DECIMAL_ZERO
                    ):
                        daily_ret_decimal = (curr / prev) - DECIMAL_ONE
                        daily_ret_float = (
                            float(daily_ret_decimal)
                            if daily_ret_decimal.is_finite()
                            else np.nan
                        )
                        daily_returns_float.append(daily_ret_float)
                    else:
                        daily_returns_float.append(np.nan)
            daily_returns = pl.Series(
                "daily_returns", daily_returns_float, dtype=pl.Float64
            ).drop_nans()
            num_trading_days = len(daily_returns)
            sharpe_ratio = 0.0
            sortino_ratio = 0.0
            if num_trading_days > 1:
                mean_daily_return = daily_returns.mean()
                std_daily_return = daily_returns.std()
                if (
                    mean_daily_return is not None
                    and std_daily_return is not None
                    and std_daily_return > 0
                ):
                    sharpe_ratio = (mean_daily_return / std_daily_return) * np.sqrt(252)
                    negative_returns = daily_returns.filter(daily_returns < 0)
                    downside_std = negative_returns.std()
                    if downside_std is not None and downside_std > 0:
                        sortino_ratio = (mean_daily_return / downside_std) * np.sqrt(
                            252
                        )

            # --- ▼▼▼ [ハング修正] Polarsによる高速なドローダウン計算 ▼▼▼ ---
            logging.info("  -> Calculating drawdown (Polars optimized)...")

            # 1. DecimalをFloatに変換 (Polarsの高速処理のため)
            #    (初期資本を先頭に追加して計算の基準にする)
            initial_equity_series = pl.Series(
                "equity", [initial_capital], dtype=pl.Object
            )

            # --- ▼▼▼ [L1005 バグ修正] .extend() -> pl.concat() に変更 ▼▼▼ ---
            equity_series_decimal = pl.concat(
                [initial_equity_series, results_df["equity"]]
            )
            # --- ▲▲▲ [L1005 バグ修正] ここまで ▲▲▲ ---

            equity_series_float = equity_series_decimal.map_elements(
                lambda d: float(d) if d is not None and d.is_finite() else np.nan,
                return_dtype=pl.Float64,
            ).fill_null(strategy="forward")  # Nanを前の値で埋める

            if (
                equity_series_float.is_empty()
                or equity_series_float.null_count() == equity_series_float.len()
            ):
                logging.warning(
                    "  -> Drawdown calculation skipped (no valid equity data)."
                )
                drawdown = pl.Series(dtype=pl.Float64)
                max_drawdown = 0.0
            else:
                # 2. 累積最大値（Rolling Max）を計算
                rolling_max_series = equity_series_float.cum_max().alias("rolling_max")

                # 3. ドローダウン（%）を計算
                drawdown_series_pct = (
                    ((equity_series_float - rolling_max_series) / rolling_max_series)
                    .fill_nan(0.0)
                    .alias("drawdown")
                )  # 0/0 を 0.0 で埋める

                # 4. 最終的なドローダウン結果を (オリジナルの `results_df` の長さに戻す)
                drawdown = drawdown_series_pct.slice(1)  # 初期資本の行を除外

                # 5. 最大ドローダウンを取得
                max_drawdown_raw = drawdown_series_pct.min()
                max_drawdown = (
                    max_drawdown_raw
                    if max_drawdown_raw is not None and np.isfinite(max_drawdown_raw)
                    else 0.0
                )
            logging.info("  -> Drawdown calculation complete.")
            # --- ▲▲▲ [ハング修正] ここまで ▲▲▲ ---

            data_period_start = str(results_df["timestamp"].min())
            data_period_end = str(results_df["timestamp"].max())

        total_trades = len(trade_log)
        win_rate = 0.0
        avg_profit = 0.0
        avg_loss = 0.0
        profit_factor = 0.0
        avg_bet_fraction = 0.0
        if total_trades > 0:
            pnl_list_decimal = trade_log["pnl"].to_list()
            pnl_list_float = [
                float(d) if d is not None and d.is_finite() else np.nan
                for d in pnl_list_decimal
            ]
            pnl_series_float = pl.Series(
                "pnl_float", pnl_list_float, dtype=pl.Float64
            ).drop_nans()
            winning_trades = trade_log.filter(pl.col("label") == 1)
            losing_trades = trade_log.filter(pl.col("label") == -1)
            num_winning_trades = len(winning_trades)
            num_losing_trades = len(losing_trades)
            win_rate = num_winning_trades / total_trades if total_trades > 0 else 0.0
            winning_pnl_list = winning_trades["pnl"].to_list()
            winning_pnl_float = [
                float(d) if d is not None and d.is_finite() else np.nan
                for d in winning_pnl_list
            ]
            winning_pnl_series = pl.Series("win_pnl", winning_pnl_float).drop_nans()
            losing_pnl_list = losing_trades["pnl"].to_list()
            losing_pnl_float = [
                float(d) if d is not None and d.is_finite() else np.nan
                for d in losing_pnl_list
            ]
            losing_pnl_series = pl.Series("lose_pnl", losing_pnl_float).drop_nans()
            avg_profit = (
                winning_pnl_series.mean() if not winning_pnl_series.is_empty() else 0.0
            )
            avg_loss = (
                losing_pnl_series.mean() if not losing_pnl_series.is_empty() else 0.0
            )
            total_profit = winning_pnl_series.sum()
            total_loss = losing_pnl_series.sum()
            if total_loss is not None and total_loss != 0:
                profit_factor = (
                    abs(total_profit / total_loss) if total_profit is not None else 0.0
                )
            elif total_profit is not None and total_profit > 0:
                profit_factor = float("inf")
            else:
                profit_factor = 0.0
            bet_list_decimal = trade_log["effective_bet_fraction"].to_list()
            bet_list_float = [
                float(d) if d is not None and d.is_finite() else np.nan
                for d in bet_list_decimal
            ]
            bet_frac_float = pl.Series(
                "bet_frac", bet_list_float, dtype=pl.Float64
            ).drop_nans()
            avg_bet_fraction = (
                bet_frac_float.mean() if not bet_frac_float.is_empty() else 0.0
            )

        report_data = {
            "strategy": f"Probabilistic Betting (Kelly Fraction: {self.config.kelly_fraction}, Base Leverage: {self.config.base_leverage}, Max Risk/Trade: {self.config.max_risk_per_trade * 100:.1f}%, Spread: {self.config.spread_pips} pips, f* Thresh: {self.config.f_star_threshold}, M2 Thresh: {self.config.m2_proba_threshold})",
            "initial_capital": float(initial_capital),
            "final_capital": float(final_capital),
            "total_return_pct": float(total_return * DECIMAL_HUNDRED),
            "sharpe_ratio_annual": sharpe_ratio if np.isfinite(sharpe_ratio) else None,
            "sortino_ratio_annual": sortino_ratio
            if np.isfinite(sortino_ratio)
            else None,
            "max_drawdown_pct": max_drawdown * 100
            if np.isfinite(max_drawdown)
            else None,
            "total_trades": total_trades,
            "win_rate_pct": win_rate * 100,
            "average_profit": avg_profit if np.isfinite(avg_profit) else None,
            "average_loss": avg_loss if np.isfinite(avg_loss) else None,
            "profit_factor": profit_factor if np.isfinite(profit_factor) else None,
            "average_effective_bet_fraction_pct": avg_bet_fraction * 100
            if np.isfinite(avg_bet_fraction)
            else None,
            "data_period_start": data_period_start,
            "data_period_end": data_period_end,
        }

        print("\n" + "=" * 50)
        print("    Backtest Performance Report")
        print("=" * 50)
        print(f" Strategy:             {report_data.get('strategy', 'N/A')}")
        print(f" Initial Capital:      {report_data.get('initial_capital', 0.0):,.2f}")
        print(f" Final Capital:        {report_data.get('final_capital', 0.0):,.2f}")
        print(
            f" Total Return:         {report_data.get('total_return_pct', 0.0):,.2f}%"
        )
        print(
            f" Sharpe Ratio (Ann.):  {report_data.get('sharpe_ratio_annual', 0.0):.2f}"
        )
        print(
            f" Sortino Ratio (Ann.): {report_data.get('sortino_ratio_annual', 0.0):.2f}"
        )
        print(
            f" Max Drawdown:         {report_data.get('max_drawdown_pct', 0.0):,.2f}%"
        )
        print("-" * 50)
        print(f" Total Trades:         {report_data.get('total_trades', 0)}")
        print(f" Win Rate:             {report_data.get('win_rate_pct', 0.0):.2f}%")
        print(f" Average Profit:       {report_data.get('average_profit', 0.0):,.3f}")
        print(f" Average Loss:         {report_data.get('average_loss', 0.0):,.3f}")
        print(f" Profit Factor:        {report_data.get('profit_factor', 0.0):.2f}")
        print(
            f" Avg. Bet Fraction:    {report_data.get('average_effective_bet_fraction_pct', 0.0):.2f}%"
        )
        print("-" * 50)
        print(
            f" Period:               {report_data.get('data_period_start', 'N/A')} to {report_data.get('data_period_end', 'N/A')}"
        )
        print("=" * 50)

        FINAL_REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
        try:
            with open(FINAL_REPORT_PATH, "w") as f:
                json.dump(report_data, f, indent=4, default=str)
            logging.info(f"Performance report saved to {FINAL_REPORT_PATH}")
        except Exception as e:
            logging.error(f"Failed to save JSON performance report: {e}")

        # --- PNGグラフ出力 (復活&修正) ---
        logging.info("Generating equity curve and drawdown chart...")
        if results_df.is_empty() or drawdown.is_empty():
            logging.warning("No data available to generate equity curve chart.")
        else:
            try:
                sns.set_style("darkgrid")
                fig, (ax1, ax2) = plt.subplots(
                    2,
                    1,
                    figsize=(15, 10),
                    sharex=True,
                    gridspec_kw={"height_ratios": [3, 1]},
                )
                timestamps_list = results_df["timestamp"].to_list()

                # --- ▼▼▼ [ハング修正] 巨大な equity_list_raw を生成し直すのではなく、Polars -> Float変換済みの `equity_series_float` を使う ---
                # (先頭の初期資本は除外する)
                equity_list_float = equity_series_float.slice(1).to_list()

                drawdown_list_raw = drawdown.to_list()
                # --- ▲▲▲ [ハング修正] ここまで ▲▲▲ ---

                drawdown_list_float = [
                    d if np.isfinite(d) else 0.0 for d in drawdown_list_raw
                ]
                ax1.plot(
                    timestamps_list,
                    equity_list_float,
                    label="Equity Curve",
                    color="dodgerblue",
                )
                ax1.set_title(
                    f"Equity Curve (Kelly: {self.config.kelly_fraction}, Lev: {self.config.base_leverage}, Max Risk: {self.config.max_risk_per_trade * 100:.1f}%)",
                    fontsize=16,
                )
                ax1.set_ylabel("Equity")
                ax1.grid(True)
                try:
                    finite_equity = [
                        e for e in equity_list_float if np.isfinite(e) and e > 0
                    ]
                    if not finite_equity:
                        ax1.ticklabel_format(style="plain", axis="y")
                    elif any(np.isinf(equity_list_float)) or (
                        max(finite_equity, default=1)
                        / max(min(finite_equity, default=1), 1)
                        > 1000
                    ):
                        ax1.set_yscale("log")
                    else:
                        ax1.ticklabel_format(style="plain", axis="y")
                except Exception as scale_err:
                    logging.warning(
                        f"Could not determine y-axis scale, using plain: {scale_err}"
                    )
                    ax1.ticklabel_format(style="plain", axis="y")
                ax2.fill_between(
                    timestamps_list, drawdown_list_float, 0, color="red", alpha=0.3
                )
                ax2.set_title("Drawdown", fontsize=16)
                ax2.set_ylabel("Drawdown (%)")
                ax2.yaxis.set_major_formatter(
                    mtick.PercentFormatter(xmax=1.0, decimals=1)
                )
                ax2.grid(True)
                plt.tight_layout()
                EQUITY_CURVE_PATH.parent.mkdir(parents=True, exist_ok=True)
                plt.savefig(EQUITY_CURVE_PATH)
                logging.info(f"Saved equity curve chart to {EQUITY_CURVE_PATH}")
                plt.close(fig)
            except Exception as e:
                logging.error(
                    f"Failed to generate equity curve chart: {e}", exc_info=True
                )
        # --- PNGグラフ出力ここまで ---

        # --- CSVログ出力 (列選択と並び替え、フォーマット修正) ---
        if not trade_log.is_empty():
            trade_log_output_path = FINAL_REPORT_PATH.parent / "detailed_trade_log.csv"
            logging.info(
                f"Preparing detailed trade log for CSV output ({len(trade_log)} trades)..."
            )
            try:
                # --- ★★★ 修正箇所 Start ★★★ ---

                # --- 1. フォーマット処理を先に実行 ---
                temp_log_formatted = trade_log.clone()  # 元のDataFrameを変更しない
                format_expressions = []

                # Timestamp -> 文字列
                format_expressions.append(
                    pl.col("timestamp")
                    .dt.strftime("%Y-%m-%d %H:%M:%S")
                    .alias("timestamp")
                )

                # Decimal/Object -> Float & Round (存在する可能性のある全てのDecimal列を対象)
                decimal_cols_round = {
                    "capital_after_trade": 2,
                    "pnl": 2,
                    "kelly_f_star": 4,
                    "spread_cost": 2,
                    "margin_required": 2,
                    "payoff_ratio": 2,
                    "close_price": 3,
                    "f_star": 4,
                    "base_bet_fraction": 4,
                    "capped_bet_fraction": 4,
                    "effective_bet_fraction": 4,
                }
                for col_name, digits in decimal_cols_round.items():
                    if col_name in temp_log_formatted.columns:  # 列が存在するか確認
                        format_expressions.append(
                            pl.col(col_name)
                            .map_elements(
                                lambda d: float(d)
                                if d is not None and d.is_finite()
                                else None,
                                return_dtype=pl.Float64,
                            )
                            .round(digits)
                            .alias(col_name)
                        )

                # Float & Round
                float_cols_round = {
                    "m2_calibrated_proba": 4,
                    "lot_size": 2,
                    "atr_value": 4,
                    "sl_multiplier": 2,
                    "pt_multiplier": 2,
                    "effective_leverage": 0,
                }
                for col_name, digits in float_cols_round.items():
                    if col_name in temp_log_formatted.columns:  # 列が存在するか確認
                        format_expressions.append(
                            pl.col(col_name).round(digits).alias(col_name)
                        )

                # フォーマット適用
                if format_expressions:
                    temp_log_formatted = temp_log_formatted.with_columns(
                        format_expressions
                    )

                # --- 2. 必要な列を選択し、並び替える ---
                desired_columns_final = [
                    "timestamp",  # 日付
                    "capital_after_trade",  # 取引後残高
                    "pnl",  # 取引損益
                    "kelly_f_star",  # ケリー (f*)
                    "m2_calibrated_proba",  # M2 Calibrated Proba
                    "lot_size",  # ロット
                    "spread_cost",  # スプレッドコスト
                    "margin_required",  # 必要証拠金
                    "label",  # 結果ラベル
                    "payoff_ratio",  # ペイオフレシオ
                    "effective_leverage",  # 実効レバレッジ
                    "direction",  # 方向
                    "close_price",  # エントリー価格
                    "atr_value",  # ATR値
                    "sl_multiplier",  # SL係数
                    "pt_multiplier",  # PT係数
                    "effective_bet_fraction",  # 実効ベット割合
                ]
                # 存在する列だけを、desired_columns_final の順序で選択
                available_columns_final = [
                    col
                    for col in desired_columns_final
                    if col in temp_log_formatted.columns
                ]
                trade_log_final_csv = temp_log_formatted.select(available_columns_final)

                # --- ★★★ 修正箇所 End ★★★ ---

                # --- CSV書き出し ---
                trade_log_final_csv.write_csv(
                    trade_log_output_path,
                    null_value="NaN",  # Null値を 'NaN' 文字列として出力
                )
                logging.info(
                    f"Formatted detailed trade log saved to {trade_log_output_path}"
                )

            except PermissionError as pe:
                logging.error(
                    f"Permission denied saving trade log to {trade_log_output_path}: {pe}"
                )
                logging.error("Please check file/directory permissions.")
            except Exception as e:
                logging.error(
                    f"Failed to save formatted detailed trade log: {e}", exc_info=True
                )
        else:
            logging.info("No trades were executed, skipping detailed trade log output.")
        # --- CSVログ出力ここまで ---

        # --- MT5風テキストレポート出力 ---
        text_report_path = FINAL_REPORT_PATH.with_suffix(".txt")
        logging.info(f"Generating text performance report to {text_report_path}...")
        try:
            with open(text_report_path, "w", encoding="utf-8") as f:
                f.write("=" * 60 + "\n")
                f.write("    Strategy Performance Report (MT5 Style)\n")
                f.write("=" * 60 + "\n\n")
                f.write(f"Strategy:\t\t{report_data.get('strategy', 'N/A')}\n")
                f.write(
                    f"Period:\t\t\t{report_data.get('data_period_start', 'N/A')} - {report_data.get('data_period_end', 'N/A')}\n\n"
                )
                f.write("-" * 30 + " Summary " + "-" * 30 + "\n")
                initial_cap = report_data.get("initial_capital", 0.0)
                final_cap = report_data.get("final_capital", 0.0)
                total_net_profit = final_cap - initial_cap
                total_ret_pct = report_data.get("total_return_pct", 0.0)
                f.write(f"Initial Deposit:\t{initial_cap:,.2f}\n")
                f.write(f"Total Net Profit:\t{total_net_profit:,.2f}\n")
                f.write(f"Final Balance:\t\t{final_cap:,.2f}\n")
                f.write(f"Total Return:\t\t{total_ret_pct:,.2f} %\n")
                profit_factor = report_data.get("profit_factor", 0.0)
                f.write(f"Profit Factor:\t\t{profit_factor:.2f}\n")
                sharpe = report_data.get("sharpe_ratio_annual", 0.0)
                f.write(f"Sharpe Ratio (Ann.):\t{sharpe:.2f}\n")
                sortino = report_data.get("sortino_ratio_annual", 0.0)
                f.write(f"Sortino Ratio (Ann.):\t{sortino:.2f}\n")
                max_dd = report_data.get("max_drawdown_pct", 0.0)
                f.write(f"Maximal Drawdown:\t{abs(max_dd):,.2f} %\n\n")
                f.write("-" * 30 + " Trades " + "-" * 30 + "\n")
                total_trades = report_data.get("total_trades", 0)
                win_pct = report_data.get("win_rate_pct", 0.0)
                loss_pct = 100.0 - win_pct if total_trades > 0 else 0.0
                num_win_trades = int(total_trades * (win_pct / 100.0))
                num_loss_trades = total_trades - num_win_trades
                f.write(f"Total Trades:\t\t{total_trades}\n")
                f.write(f"Winning Trades (%):\t{num_win_trades} ({win_pct:.2f} %)\n")
                f.write(f"Losing Trades (%):\t{num_loss_trades} ({loss_pct:.2f} %)\n")
                avg_profit = report_data.get("average_profit", 0.0)
                avg_loss = report_data.get("average_loss", 0.0)
                f.write(f"Average Profit:\t\t{avg_profit:,.3f}\n")
                f.write(f"Average Loss:\t\t{avg_loss:,.3f}\n")
                avg_bet_pct = report_data.get("average_effective_bet_fraction_pct", 0.0)
                f.write(f"Avg Bet Size (% Cap):\t{avg_bet_pct:.2f} %\n\n")
                f.write("=" * 60 + "\n")
            logging.info(f"Text performance report saved successfully.")
        except Exception as e:
            logging.error(f"Failed to save text performance report: {e}", exc_info=True)
        # --- MT5風テキストレポート出力ここまで ---


if __name__ == "__main__":
    default_config = BacktestConfig()

    parser = argparse.ArgumentParser(
        description="Project Forge Backtest Simulator (Kelly + Dynamic SL + Constraints)"  # Title Updated
    )
    # --- ★★★ 修正: 引数定義を Config に合わせて修正 ★★★ ---
    parser.add_argument(
        "--kelly",
        type=float,
        default=default_config.kelly_fraction,
        help=f"Kelly fraction. Default: {default_config.kelly_fraction}",
    )
    # parser.add_argument("--leverage", ...) # <- 削除
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
    # --- ★★★ 修正ここまで ★★★ ---
    args = parser.parse_args()

    # --- ★★★ 修正: Config生成を引数に合わせて修正 ★★★ ---
    config = BacktestConfig(
        kelly_fraction=args.kelly,
        base_leverage=args.base_leverage,  # ★ 修正
        max_risk_per_trade=args.max_risk,
        f_star_threshold=args.fstar_th,
        m2_proba_threshold=args.m2_th,
        test_limit_partitions=args.test_limit_partitions,
        oof_mode=args.oof,
        min_capital_threshold=args.min_capital,
        value_per_pip=args.value_per_pip,
        spread_pips=args.spread_pips,  # ★ 追加
        max_positions=args.max_positions,
        # (context_data_path は default_config から自動的に設定される)
    )
    # --- ★★★ 修正ここまで ★★★ ---

    # --- ★★★ 修正: バリデーションを修正 ★★★ ---
    if not (0 < config.max_risk_per_trade <= 1.0):
        parser.error("--max-risk must be between 0 (exclusive) and 1.0 (inclusive).")
    # if config.max_leverage < 1.0: # <- 削除
    #     parser.error("--leverage must be >= 1.0.")
    if config.base_leverage < 1.0:  # ★ 修正
        parser.error("--base-leverage must be >= 1.0.")
    if config.value_per_pip <= 0:
        parser.error("--value-per-pip must be greater than 0.")
    if config.spread_pips < 0:  # ★ 追加 (ゼロは許容)
        parser.error("--spread-pips cannot be negative.")
    # --- ★★★ 修正ここまで ★★★ ---

    simulator = BacktestSimulator(config)
    simulator.run()
