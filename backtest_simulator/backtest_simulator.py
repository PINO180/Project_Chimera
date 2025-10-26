# /workspace/models/backtest_simulator.py
# [修正版: 破産閾値 (min_capital_threshold) の導入]
# [修正版: Chunking (Streaming) 導入によるメモリ効率化 + Schemaログ抑制]
# [修正版: Booster API の .predict() を使うように修正]
# [修正版: AttributeError (clip_min) 修正 + レバレッジ調整]
# [修正版: 最大リスク上限導入 + グラフエラー修正]
# [修正版: ロットサイズ (lot_size) の計算とログ出力を追加]
# ★★★ [最終修正版 v2: S6の動的ATR/SL/PT/方向性 を使って正確なロット計算＋ログ出力] ★★★
# ★★★ [最終修正版 v3: 最大ロット数、必要証拠金/動的レバレッジ制限、スプレッドコストを実装] ★★★

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
    m1_model_path: Path = S7_M1_MODEL_PKL
    m2_model_path: Path = S7_M2_MODEL_PKL
    m1_calibrator_path: Path = S7_M1_CALIBRATED
    m2_calibrator_path: Path = S7_M2_CALIBRATED
    kelly_fraction: float = 0.5
    # max_leverage: float = 100 # <- ★★★ 削除: base_leverage を使う ★★★
    max_risk_per_trade: float = 0.02
    f_star_threshold: float = 0.0
    m2_proba_threshold: float = 0.3
    test_limit_partitions: int = 0
    oof_mode: bool = False
    min_capital_threshold: float = 1.0

    # --- ★★★ 修正: 基本レバレッジとスプレッドを追加 ★★★ ---
    base_leverage: float = 2000.0  # 設定可能な基本レバレッジ
    spread_pips: float = 16.0  # XAUUSD スタンダード口座のスプレッド
    value_per_pip: float = 10.0
    """
    (ASSUMPTION) 1ロットあたりの1pipの価値 (口座通貨単位)。
    XAUUSD (1 lot = 100 oz) の場合: $10.0 [cite: 31-35]
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
        self.features_m2 = self.features_base + ["m1_pred_proba"]
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
            limit_leverage = Decimal("2000")  # 最大2000倍 [cite: 11, 105]
        elif equity < Decimal("100000"):
            limit_leverage = Decimal("1000")  # 最大1000倍 [cite: 11, 106]
        else:
            limit_leverage = Decimal("500")  # 最大500倍 [cite: 11, 107]

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
                    "Ensure create_proxy_labels.py included 'close' price."
                )

            lf = lf.select(required_cols).sort("timestamp")

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
        self, df_chunk: pl.DataFrame
    ) -> Tuple[pl.DataFrame, pl.DataFrame]:
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

        timestamps_chunk = df_chunk["timestamp"].to_list()
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

            # 取引判断
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
                    # --- ★★★ 修正: 動的ロットサイズ計算 + 制約適用 (ここから) ★★★ ---
                    risk_amount_decimal = current_capital * effective_bet_fraction

                    # 1. 動的SL幅 (pips) を計算
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
                    else:
                        dynamic_sl_pips_decimal = Decimal(
                            str(atr_value_float)
                        ) * Decimal(str(sl_multiplier_float))
                        stop_loss_currency_per_lot = (
                            dynamic_sl_pips_decimal * DECIMAL_VALUE_PER_PIP
                        )

                        # 2. リスクベースの希望ロットサイズ計算
                        if stop_loss_currency_per_lot > DECIMAL_ZERO:
                            desired_lot_size_decimal = (
                                risk_amount_decimal / stop_loss_currency_per_lot
                            )
                        else:
                            desired_lot_size_decimal = DECIMAL_ZERO

                    # 3. 実効レバレッジを決定
                    effective_leverage_decimal = self._get_effective_leverage(
                        current_capital
                    )

                    # 4. 証拠金ベースの最大許容ロットサイズを計算
                    # Lot = (Equity * Leverage) / (Price * ContractSize)
                    if effective_leverage_decimal > DECIMAL_ZERO:
                        max_lot_by_margin = (
                            current_capital * effective_leverage_decimal
                        ) / (current_price_decimal * CONTRACT_SIZE)
                        max_lot_by_margin = max_lot_by_margin.copy_abs().max(
                            DECIMAL_ZERO
                        )  # 負にならないように
                    else:
                        max_lot_by_margin = DECIMAL_ZERO

                    # 5. 時間帯ベースの最大許容ロット数を取得
                    max_lot_allowed_by_broker = self._get_max_lot_allowed(
                        current_timestamp
                    )

                    # 6. 最終的なロットサイズを決定 (希望ロット vs 証拠金上限 vs ブローカー上限)
                    final_lot_size_decimal = desired_lot_size_decimal.min(
                        max_lot_by_margin
                    ).min(max_lot_allowed_by_broker)
                    final_lot_size_decimal = final_lot_size_decimal.copy_abs().max(
                        DECIMAL_ZERO
                    )  # 念のためゼロ以上を保証

                    # 7. 最終ロットサイズがゼロより大きいか？
                    if final_lot_size_decimal > DECIMAL_ZERO:
                        # 8. 必要証拠金を計算 [cite: 45]
                        margin_required_decimal = (
                            current_price_decimal
                            * final_lot_size_decimal
                            * CONTRACT_SIZE
                        ) / effective_leverage_decimal

                        # 9. スプレッドコストを計算
                        spread_pips_decimal = Decimal(str(self.config.spread_pips))
                        spread_cost_decimal = (
                            final_lot_size_decimal
                            * spread_pips_decimal
                            * DECIMAL_VALUE_PER_PIP
                        )

                        # 10. スプレッドコストを資本から差し引く
                        capital_before_pnl = current_capital - spread_cost_decimal

                        # 11. PnLを再計算 (最終ロットサイズに基づいてリスク額を再定義)
                        #    リスク額 = 最終ロットサイズ * 1ロットあたりSL(通貨)
                        if (
                            stop_loss_currency_per_lot > DECIMAL_ZERO
                        ):  # desired_lot_size計算時にチェック済みだが念のため
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

                        # 12. 次の資本を計算
                        next_capital = capital_before_pnl + pnl

                        # 13. ログ用に float に変換
                        lot_size_float = (
                            float(final_lot_size_decimal)
                            if final_lot_size_decimal.is_finite()
                            else 0.0
                        )

                        # 14. 無限大/NaN チェック
                        if not next_capital.is_finite():
                            logging.error(
                                f"Capital NaN/Inf at {current_timestamp}. Prev: {current_capital:.2E}, Spread: {spread_cost_decimal:.2E}, PnL: {pnl:.2E}"
                            )
                            current_capital = DECIMAL_ZERO
                        else:
                            current_capital = next_capital

                    else:  # 最終ロットサイズがゼロになった場合
                        should_trade = False  # 取引しないことにする
                        pnl = DECIMAL_ZERO
                        actual_label = labels_chunk[i]
                        # lot_size_float, margin_required_decimal, spread_cost_decimal はゼロのまま
                    # --- ★★★ 修正: 動的ロットサイズ計算 + 制約適用 (ここまで) ★★★ ---

                else:  # ケリー推奨がゼロ以下 or M2確率が閾値以下
                    should_trade = False
                    pnl = DECIMAL_ZERO
                    actual_label = labels_chunk[i]
                    # lot_size_float, margin_required_decimal, spread_cost_decimal はゼロのまま

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
        # (レポート生成部分は変更なし、CSV出力部分のみ修正)
        logging.info("Analyzing results and generating report...")
        # ... (既存のレポート計算ロジックは省略) ...

        # ★★★ 修正: CSV出力部分 ★★★
        if not trade_log.is_empty():
            trade_log_output_path = FINAL_REPORT_PATH.parent / "detailed_trade_log.csv"
            try:
                trade_log_for_csv = trade_log.clone()

                # --- 型変換とフォーマット指定 ---
                format_expressions = []

                # Timestamp
                format_expressions.append(
                    pl.col("timestamp")
                    .dt.strftime("%Y-%m-%d %H:%M:%S")
                    .alias("timestamp")
                )

                # Object (Decimal) -> Float & Round
                decimal_cols = [
                    ("pnl", 0),
                    ("capital_after_trade", 0),
                    ("payoff_ratio", 2),
                    ("kelly_f_star", 4),
                    ("f_star", 4),
                    ("base_bet_fraction", 4),
                    ("capped_bet_fraction", 4),
                    ("effective_bet_fraction", 4),
                    ("margin_required", 2),
                    ("spread_cost", 2),
                    ("close_price", 3),  # closeは小数点以下3桁程度
                ]
                for col_name, digits in decimal_cols:
                    if col_name in trade_log_for_csv.columns:
                        # Nullを許容しつつfloatに変換し、丸める
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

                # Float & Round (既存のFloat列 + Objectから変換されたFloat列も含む可能性)
                float_cols_for_round = [
                    ("m2_calibrated_proba", 4),
                    ("lot_size", 2),
                    ("atr_value", 4),
                    ("sl_multiplier", 2),
                    ("pt_multiplier", 2),
                    ("effective_leverage", 0),  # レバレッジは整数
                ]
                for col_name, digits in float_cols_for_round:
                    if col_name in trade_log_for_csv.columns:
                        # 既存のfloat列もNullを考慮して丸める
                        format_expressions.append(
                            pl.col(col_name).round(digits).alias(col_name)
                        )

                # Integer (フォーマット不要だが選択はしておく)
                int_cols = ["label", "direction"]
                # format_expressions.extend([pl.col(c) for c in int_cols if c in trade_log_for_csv.columns]) # そのまま選択

                # フォーマット適用 (エラーハンドリング強化)
                try:
                    if format_expressions:
                        # 必要な列だけを選択しつつフォーマットを適用
                        select_cols = [
                            e.meta.output_name() for e in format_expressions
                        ] + [c for c in int_cols if c in trade_log_for_csv.columns]
                        trade_log_formatted = trade_log_for_csv.select(
                            select_cols
                        ).with_columns(format_expressions)
                    else:
                        trade_log_formatted = trade_log_for_csv
                except Exception as fmt_e:
                    logging.error(
                        f"Error applying formatting: {fmt_e}. Saving raw data instead."
                    )
                    trade_log_formatted = trade_log_for_csv  # エラー時は生データを出力

                # CSV 書き出し
                trade_log_formatted.write_csv(
                    trade_log_output_path,
                    null_value="NaN",
                )
                logging.info(f"Detailed trade log saved to {trade_log_output_path}")

            except Exception as e:
                logging.error(f"Failed to save detailed trade log: {e}", exc_info=True)
        else:
            logging.info("No trades were executed, skipping detailed trade log output.")
        # ★★★ CSV出力修正ここまで ★★★

        # ... (既存のグラフ生成ロジックは省略) ...


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
