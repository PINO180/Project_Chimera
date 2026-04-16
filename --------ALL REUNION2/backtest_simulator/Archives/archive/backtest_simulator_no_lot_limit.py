# /workspace/models/backtest_simulator.py
# [修正版: 破産閾値 (min_capital_threshold) の導入]
# [修正版: Chunking (Streaming) 導入によるメモリ効率化 + Schemaログ抑制]
# [修正版: Booster API の .predict() を使うように修正]
# [修正版: AttributeError (clip_min) 修正 + レバレッジ調整]
# [修正版: 最大リスク上限導入 + グラフエラー修正]
# [修正版: ロットサイズ (lot_size) の計算とログ出力を追加]
# ★★★ [最終修正版: S6の動的ATR/SL/PT/方向性 を使って正確なロット計算＋ログ出力] ★★★

import sys
from pathlib import Path
import logging
import argparse
from dataclasses import dataclass, field
from typing import Dict, Any, Tuple, List
import json
import datetime as dt

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
    kelly_fraction: float = 0.1
    max_leverage: float = 2000
    max_risk_per_trade: float = 0.02
    f_star_threshold: float = 0.0
    m2_proba_threshold: float = 0.5
    test_limit_partitions: int = 0
    oof_mode: bool = False
    min_capital_threshold: float = 1.0

    # --- ★★★ 修正: 不要な仮定値を削除し、pip価値のみ残す ★★★ ---
    # assumed_sl_pips: float = 20.0 # <- 削除 (S6から動的に取得するため)
    value_per_pip: float = 10.0  # ★★★ 引数名を変更 ★★★
    """
    (ASSUMPTION) 1ロットあたりの1pipの価値 (口座通貨単位)。
    XAUUSD (1 lot = 100 oz) の場合:
    1 pip (0.1) の価格変動 = $0.1/oz * 100 oz = $10.0
    この値はシンボルによって異なるため、実行時に指定が必要。
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

    def run(self):
        logging.info("### Backtest Simulator: START ###")
        logging.info(
            f"Strategy: Probabilistic Betting with Kelly Fraction = {self.config.kelly_fraction}, "
            f"Max Leverage = {self.config.max_leverage}, Max Risk/Trade = {self.config.max_risk_per_trade * 100:.1f}%"
        )
        logging.info(
            f"Bankruptcy Threshold (Min Capital): {self.config.min_capital_threshold:,.2f}"
        )
        # --- ★★★ 修正: ログ出力を pip 価値のみに修正 ★★★ ---
        logging.info(
            f"Lot Size Assumption: Value = {self.config.value_per_pip}/lot/pip (SL width from S6 data)"
        )
        # --- ★★★ 修正ここまで ★★★ ---

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
                # ★★★ 修正: _prepare_data で必要な列を確実に含めるようにしたので collect 前の select は不要 ★★★
                df_chunk = lf.filter(
                    pl.col("timestamp").dt.date() == current_date
                ).collect()
                if df_chunk.is_empty():
                    logging.debug(f"Skipping empty partition: {current_date}")
                    continue
                logging.debug(f"Collected {len(df_chunk)} rows for {current_date}")

                # ★★★ 追加: 収集したデータに必要な列があるか最終確認 ★★★
                required_sim_cols = [
                    "timestamp",
                    "label",
                    "payoff_ratio",
                    "atr_value",
                    "sl_multiplier",
                    "pt_multiplier",
                    "direction",
                ]
                if not self.config.oof_mode:  # In-Sample モードでは特徴量も必要
                    required_sim_cols.extend(self.features_base)
                else:  # OOF モードでは m2_raw_proba が必要
                    required_sim_cols.append("m2_raw_proba")

                missing_sim_cols = [
                    col for col in required_sim_cols if col not in df_chunk.columns
                ]
                if missing_sim_cols:
                    logging.error(
                        f"CRITICAL: Required columns for simulation missing in collected data for {current_date}! Missing: {missing_sim_cols}"
                    )
                    logging.error(
                        "   -> This might indicate an issue with create_proxy_labels.py or data preparation."
                    )
                    continue  # 問題のあるパーティションをスキップ

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
        # ★★★ 修正: S6から読み込む基本カラムリストに必要な列を追加 ★★★
        base_cols = [
            "timestamp",
            "label",
            "payoff_ratio",
            "atr_value",
            "sl_multiplier",
            "pt_multiplier",
            "direction",
        ]
        # ★★★ 修正ここまで ★★★

        if not self.config.oof_mode:
            logging.info(
                f"Preparing data from {self.config.simulation_data_path} (In-Sample Mode)..."
            )
            # ★★★ 修正: required_cols に基本カラムと特徴量リストを結合 ★★★
            required_cols_set = set(self.features_base)
            required_cols_set.update(base_cols)  # base_cols を追加
            required_cols = list(required_cols_set)
            # ★★★ 修正ここまで ★★★

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
                    "Ensure create_proxy_labels.py was run correctly with the latest changes."
                )

            lf = lf.select(required_cols).sort("timestamp")

        else:  # OOF Mode
            logging.info(
                f"Preparing data from {self.config.simulation_data_path} and {self.config.oof_predictions_path} (OOF Mode)..."
            )
            try:
                base_glob_path = str(self.config.simulation_data_path / "**/*.parquet")
                # ★★★ 修正: select するカラムリストを base_cols に変更 ★★★
                base_lf = pl.scan_parquet(base_glob_path).select(base_cols)
            # ★★★ 修正ここまで ★★★
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
            lf = base_lf.join(oof_lf, on="timestamp", how="inner").sort(
                "timestamp"
            )  # Use inner join
            # ★★★ 修正: required_cols は base_cols と m2_raw_proba を合わせたもの ★★★
            required_cols = base_cols + ["m2_raw_proba"]
            # ★★★ 修正ここまで ★★★

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
                # ★★★ 追加: X_m1 に必要な特徴量があるか確認 ★★★
                missing_m1_features = [
                    f for f in self.features_base if f not in df_chunk.columns
                ]
                if missing_m1_features:
                    raise ValueError(
                        f"Missing M1 features in data chunk: {missing_m1_features}"
                    )
                # ★★★ 追加ここまで ★★★
                X_m1 = df_chunk.select(self.features_base).fill_null(0).to_numpy()
                raw_m1_proba = self.m1_model.predict(X_m1)
                calibrated_m1_proba = self.m1_calibrator.predict(raw_m1_proba)
                calibrated_m1_proba = np.clip(calibrated_m1_proba, 0.0, 1.0)
                df_chunk = df_chunk.with_columns(
                    pl.Series("m1_pred_proba", calibrated_m1_proba)
                )
            except Exception as e:
                logging.error(
                    f"Error during M1 prediction/calibration (In-Sample): {e}",
                    exc_info=True,
                )
                raise

            logging.debug("  -> Step 2/2: M2 Prediction & Calibration...")
            try:
                # ★★★ 追加: X_m2 に必要な特徴量があるか確認 ★★★
                missing_m2_features = [
                    f for f in self.features_m2 if f not in df_chunk.columns
                ]
                if missing_m2_features:
                    raise ValueError(
                        f"Missing M2 features in data chunk: {missing_m2_features}"
                    )
                # ★★★ 追加ここまで ★★★
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
                logging.error(
                    f"Error during M2 prediction/calibration (In-Sample): {e}",
                    exc_info=True,
                )
                raise

        else:  # OOF Mode
            logging.debug("  -> Mode: OOF (Loading pre-calculated)...")
            try:
                # ★★★ 追加: 必要な 'm2_raw_proba' があるか確認 ★★★
                if "m2_raw_proba" not in df_chunk.columns:
                    raise ValueError("Missing 'm2_raw_proba' column in OOF data chunk.")
                # ★★★ 追加ここまで ★★★
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
                logging.error(f"Error during M2 calibration (OOF): {e}", exc_info=True)
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

        # --- ★★★ 修正: 不要な仮定値を削除し、pip価値のみDecimal化 ★★★ ---
        DECIMAL_VALUE_PER_PIP = Decimal(str(self.config.value_per_pip))
        # --- ★★★ 修正ここまで ★★★ ---

        timestamps_chunk = df_chunk["timestamp"].to_list()
        p_m2_calibrated = df_chunk["m2_calibrated_proba"].to_numpy()
        labels_chunk = df_chunk["label"].to_numpy()
        # ★★★ 追加: S6から読み込んだ新しい列を取得 ★★★
        atr_values_chunk = df_chunk["atr_value"].to_numpy()
        sl_multipliers_chunk = df_chunk["sl_multiplier"].to_numpy()
        pt_multipliers_chunk = df_chunk["pt_multiplier"].to_numpy()
        directions_chunk = df_chunk["direction"].to_numpy()
        # ★★★ 追加ここまで ★★★

        payoff_ratios_chunk = []
        for b_float in df_chunk["payoff_ratio"].to_list():
            if b_float is None or not np.isfinite(b_float) or b_float <= 0:
                logging.warning(
                    f"Invalid or non-positive payoff_ratio {b_float} encountered, using Decimal('2.0')"
                )
                payoff_ratios_chunk.append(Decimal("2.0"))  # Fallback
            else:
                payoff_ratios_chunk.append(Decimal(str(b_float)))

        for i in range(len(df_chunk)):
            lot_size_float = 0.0
            # ★★★ 追加: ループ内で使う新しい変数を初期化 ★★★
            atr_value_float = atr_values_chunk[i]
            sl_multiplier_float = sl_multipliers_chunk[i]
            pt_multiplier_float = pt_multipliers_chunk[i]
            direction_int = directions_chunk[i]
            # ★★★ 追加ここまで ★★★

            if current_capital < DECIMAL_MIN_CAPITAL:
                equity_values_chunk.append(DECIMAL_ZERO)
                pnl, base_bet_fraction, capped_bet_fraction, effective_bet_fraction = (
                    DECIMAL_ZERO,
                ) * 4
                kelly_f_star, f_star = (DECIMAL_ZERO,) * 2
                p = p_m2_calibrated[i]
                b = payoff_ratios_chunk[i]
                actual_label = labels_chunk[i]
                should_trade = False
                continue
            else:
                p_float = p_m2_calibrated[i]
                b = payoff_ratios_chunk[i]
                p_decimal = Decimal(str(p_float))
                q = DECIMAL_ONE - p_decimal
                if b <= DECIMAL_ZERO:
                    kelly_f_star = DECIMAL_ZERO
                else:
                    kelly_f_star = (b * p_decimal - q) / b

                f_star = (
                    kelly_f_star.copy_abs().max(DECIMAL_ZERO)
                    if kelly_f_star.is_finite()
                    else DECIMAL_ZERO
                )

                logging.debug(f"Calculated f_star: {float(f_star):.4f}")

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
                    capped_bet_fraction = effective_bet_fraction

                    if effective_bet_fraction > DECIMAL_ZERO:
                        # --- ★★★ 修正: 動的ロットサイズ計算 (ここから) ★★★ ---
                        risk_amount_decimal = current_capital * effective_bet_fraction

                        # 動的SL幅 (pips) を計算 (float -> Decimal)
                        # Null や非有限値でないことを確認
                        if (
                            atr_value_float is None
                            or not np.isfinite(atr_value_float)
                            or atr_value_float <= 0
                        ):
                            logging.warning(
                                f"Invalid atr_value {atr_value_float} at {timestamps_chunk[i]}, skipping lot calculation."
                            )
                            lot_size_decimal = DECIMAL_ZERO
                        elif (
                            sl_multiplier_float is None
                            or not np.isfinite(sl_multiplier_float)
                            or sl_multiplier_float <= 0
                        ):
                            logging.warning(
                                f"Invalid sl_multiplier {sl_multiplier_float} at {timestamps_chunk[i]}, skipping lot calculation."
                            )
                            lot_size_decimal = DECIMAL_ZERO
                        else:
                            dynamic_sl_pips_decimal = Decimal(
                                str(atr_value_float)
                            ) * Decimal(str(sl_multiplier_float))

                            # 1ロットあたりのSL（通貨）を計算
                            stop_loss_currency_per_lot = (
                                dynamic_sl_pips_decimal * DECIMAL_VALUE_PER_PIP
                            )

                            # ロットサイズ計算 (ゼロ除算チェック)
                            if stop_loss_currency_per_lot > DECIMAL_ZERO:
                                lot_size_decimal = (
                                    risk_amount_decimal / stop_loss_currency_per_lot
                                )
                            else:
                                logging.warning(
                                    f"Stop loss per lot is zero or negative ({stop_loss_currency_per_lot}) at {timestamps_chunk[i]}, cannot calculate lot size."
                                )
                                lot_size_decimal = DECIMAL_ZERO

                        # ログ用に float に変換
                        lot_size_float = (
                            float(lot_size_decimal)
                            if lot_size_decimal.is_finite()
                            else 0.0
                        )
                        # --- ★★★ 修正: 動的ロットサイズ計算 (ここまで) ★★★ ---

                        pnl = DECIMAL_ZERO
                        actual_label = labels_chunk[i]
                        if actual_label == 1:
                            pnl = current_capital * effective_bet_fraction * b
                        elif actual_label == -1:
                            loss_amount = (
                                current_capital * effective_bet_fraction * DECIMAL_ONE
                            )
                            pnl = loss_amount.copy_negate()

                        next_capital = current_capital + pnl

                        if not next_capital.is_finite():
                            logging.error(
                                f"Capital became non-finite (Inf/NaN) at {timestamps_chunk[i]}. "
                                f"Prev Capital: {current_capital:,.2E}, PnL: {pnl:,.2E}, Label: {actual_label}"
                            )
                            current_capital = DECIMAL_ZERO
                            equity_values_chunk.append(current_capital)
                            continue

                        # --- ★★★ 修正: ログ辞書に新しい列を追加 ★★★ ---
                        trade_log_chunk.append(
                            {
                                "timestamp": timestamps_chunk[i],
                                "pnl": pnl,
                                "capital_after_trade": next_capital,
                                "m2_calibrated_proba": p_float,
                                "payoff_ratio": b,
                                "kelly_f_star": kelly_f_star,
                                "f_star": f_star,
                                "base_bet_fraction": base_bet_fraction,
                                "capped_bet_fraction": capped_bet_fraction,
                                "effective_bet_fraction": effective_bet_fraction,
                                "label": actual_label,
                                "lot_size": lot_size_float,
                                # ★★★ 追加 ★★★
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
                            }
                        )
                        # --- ★★★ 修正ここまで ★★★ ---
                        current_capital = next_capital
                    else:  # effective_bet_fraction が 0 以下の場合
                        should_trade = False
                        pnl = DECIMAL_ZERO
                        actual_label = labels_chunk[i]
                        # lot_size_float は 0.0 のまま
                else:  # 取引しなかった場合 (should_trade is False)
                    pnl = DECIMAL_ZERO
                    base_bet_fraction = DECIMAL_ZERO
                    capped_bet_fraction = DECIMAL_ZERO
                    effective_bet_fraction = DECIMAL_ZERO
                    actual_label = labels_chunk[i]
                    # lot_size_float は 0.0 のまま

            equity_values_chunk.append(current_capital)

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
            # ★★★ 追加 ★★★
            "atr_value": pl.Float64,
            "sl_multiplier": pl.Float32,  # 元のデータ型に合わせる
            "pt_multiplier": pl.Float32,  # 元のデータ型に合わせる
            "direction": pl.Int8,  # 元のデータ型に合わせる
        }
        # --- ★★★ 修正ここまで ★★★ ---

        if trade_log_chunk:
            try:
                # ★★★ 修正: trade_log_schema.keys() を使う ★★★
                trade_log_data = {
                    key: [d.get(key) for d in trade_log_chunk]
                    for key in trade_log_schema.keys()
                }
            except KeyError as e:
                logging.error(f"Inconsistent keys in trade_log_chunk: {e}")
                trade_log_chunk_df = pl.DataFrame(schema=trade_log_schema)
                return results_chunk_df, trade_log_chunk_df

            # --- ★★★ 修正: DataFrame 構築時に新しい列を追加 ★★★ ---
            # 各列を pl.Series で作成し、型を指定する
            series_dict = {
                "timestamp": pl.Series(
                    "timestamp", trade_log_data["timestamp"], dtype=pl.Datetime
                ),
                "pnl": pl.Series("pnl", trade_log_data["pnl"], dtype=pl.Object),
                "capital_after_trade": pl.Series(
                    "capital_after_trade",
                    trade_log_data["capital_after_trade"],
                    dtype=pl.Object,
                ),
                "m2_calibrated_proba": pl.Series(
                    "m2_calibrated_proba",
                    trade_log_data["m2_calibrated_proba"],
                    dtype=pl.Float64,
                ),
                "payoff_ratio": pl.Series(
                    "payoff_ratio",
                    trade_log_data["payoff_ratio"],
                    dtype=pl.Object,
                ),
                "kelly_f_star": pl.Series(
                    "kelly_f_star",
                    trade_log_data["kelly_f_star"],
                    dtype=pl.Object,
                ),
                "f_star": pl.Series(
                    "f_star", trade_log_data["f_star"], dtype=pl.Object
                ),
                "base_bet_fraction": pl.Series(
                    "base_bet_fraction",
                    trade_log_data["base_bet_fraction"],
                    dtype=pl.Object,
                ),
                "capped_bet_fraction": pl.Series(
                    "capped_bet_fraction",
                    trade_log_data["capped_bet_fraction"],
                    dtype=pl.Object,
                ),
                "effective_bet_fraction": pl.Series(
                    "effective_bet_fraction",
                    trade_log_data["effective_bet_fraction"],
                    dtype=pl.Object,
                ),
                "label": pl.Series("label", trade_log_data["label"], dtype=pl.Int64),
                "lot_size": pl.Series(
                    "lot_size", trade_log_data["lot_size"], dtype=pl.Float64
                ),
                # ★★★ 追加 ★★★
                "atr_value": pl.Series(
                    "atr_value", trade_log_data["atr_value"], dtype=pl.Float64
                ),
                "sl_multiplier": pl.Series(
                    "sl_multiplier", trade_log_data["sl_multiplier"], dtype=pl.Float32
                ),
                "pt_multiplier": pl.Series(
                    "pt_multiplier", trade_log_data["pt_multiplier"], dtype=pl.Float32
                ),
                "direction": pl.Series(
                    "direction", trade_log_data["direction"], dtype=pl.Int8
                ),
            }
            trade_log_chunk_df = pl.DataFrame(series_dict)
            # --- ★★★ 修正ここまで ★★★ ---
        else:
            trade_log_chunk_df = pl.DataFrame(schema=trade_log_schema)

        return results_chunk_df, trade_log_chunk_df

    def _analyze_and_report(self, results_df: pl.DataFrame, trade_log: pl.DataFrame):
        logging.info("Analyzing results and generating report...")

        DECIMAL_ONE = Decimal("1.0")
        DECIMAL_ZERO = Decimal("0.0")

        if results_df.is_empty():
            logging.error("No simulation results to analyze.")
            final_capital = Decimal(str(self.config.initial_capital))
            initial_capital = Decimal(str(self.config.initial_capital))
            total_return = DECIMAL_ZERO
            sharpe_ratio = 0.0
            sortino_ratio = 0.0
            max_drawdown = 0.0
            daily_returns = pl.Series(dtype=pl.Float64)
            drawdown = pl.Series(dtype=pl.Float64)
        else:
            initial_capital = Decimal(str(self.config.initial_capital))
            final_capital = (
                results_df["equity"][-1]
                if not results_df.is_empty()
                else initial_capital
            )
            if final_capital is None or not final_capital.is_finite():
                final_capital = DECIMAL_ZERO

            total_return = (
                (final_capital / initial_capital - DECIMAL_ONE)
                if initial_capital > DECIMAL_ZERO
                else DECIMAL_ZERO
            )

            daily_equity_series = (
                results_df.group_by(pl.col("timestamp").dt.date().alias("date"))
                .agg(pl.first("equity"))
                .sort("date")["equity"]
            )
            daily_equity_list = daily_equity_series.to_list()
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
                        daily_returns_float.append(
                            float(daily_ret_decimal)
                            if daily_ret_decimal.is_finite()
                            else np.nan
                        )
                    elif prev is not None and not prev.is_finite():
                        daily_returns_float.append(np.nan)
                    else:
                        daily_returns_float.append(0.0)
            daily_returns = pl.Series(daily_returns_float, dtype=pl.Float64)

            num_trading_days = len(daily_returns)
            if num_trading_days > 1:
                std_daily_return = (
                    daily_returns.drop_nans().std()
                    if not daily_returns.is_empty()
                    else 0
                )
                if std_daily_return is not None and std_daily_return > 0:
                    mean_daily_return = daily_returns.drop_nans().mean()
                    sharpe_ratio = (
                        (mean_daily_return / std_daily_return) * np.sqrt(252)
                        if mean_daily_return is not None
                        else 0.0
                    )
                    negative_returns = daily_returns.filter(daily_returns < 0)
                    downside_std = (
                        negative_returns.drop_nans().std()
                        if not negative_returns.is_empty()
                        else 0
                    )
                    sortino_ratio = (
                        (mean_daily_return / downside_std) * np.sqrt(252)
                        if mean_daily_return is not None
                        and downside_std is not None
                        and downside_std > 0
                        else 0.0
                    )
                else:
                    sharpe_ratio = 0.0
                    sortino_ratio = 0.0
            else:
                sharpe_ratio = 0.0
                sortino_ratio = 0.0

            equity_list = results_df["equity"].to_list()
            rolling_max_list = []
            current_max = Decimal("-Inf")
            for e in equity_list:
                if e is None:
                    e = Decimal("-Inf")
                elif not e.is_finite():
                    if e.is_nan():
                        e = Decimal("-Inf")

                if e > current_max:
                    current_max = e
                rolling_max_list.append(current_max)

            drawdown_list_float = []
            for i in range(len(equity_list)):
                r_max = rolling_max_list[i]
                e_curr = equity_list[i]

                if e_curr is None or not e_curr.is_finite():
                    e_curr = Decimal("-Inf")

                if r_max > DECIMAL_ZERO and r_max.is_finite():
                    dd_decimal = (e_curr - r_max) / r_max
                    drawdown_list_float.append(
                        float(dd_decimal) if dd_decimal.is_finite() else np.nan
                    )
                elif r_max is not None and not r_max.is_finite():
                    drawdown_list_float.append(float(e_curr - r_max))
                else:
                    drawdown_list_float.append(0.0)

            drawdown = pl.Series(drawdown_list_float, dtype=pl.Float64)
            max_drawdown = (
                drawdown.drop_nans().min() if not drawdown.is_empty() else 0.0
            )
            if max_drawdown is None or np.isnan(max_drawdown) or np.isinf(max_drawdown):
                max_drawdown = -1.0

        total_trades = len(trade_log)
        if total_trades > 0:
            pnl_list_decimal = trade_log["pnl"].to_list()
            pnl_list_float = [
                float(d) if d is not None and d.is_finite() else np.nan
                for d in pnl_list_decimal
            ]
            pnl_series_float = pl.Series("pnl_float", pnl_list_float, dtype=pl.Float64)

            winning_pnl_float = pnl_series_float.filter(trade_log["label"] == 1)
            losing_pnl_float = pnl_series_float.filter(trade_log["label"] == -1)

            win_rate = (
                len(winning_pnl_float) / total_trades if total_trades > 0 else 0.0
            )
            avg_profit = (
                winning_pnl_float.drop_nans().mean()
                if not winning_pnl_float.is_empty()
                else 0.0
            )
            avg_loss = (
                losing_pnl_float.drop_nans().mean()
                if not losing_pnl_float.is_empty()
                else 0.0
            )
            profit_factor_num = winning_pnl_float.drop_nans().sum()
            profit_factor_den = losing_pnl_float.drop_nans().sum()
            profit_factor = (
                abs(profit_factor_num / profit_factor_den)
                if profit_factor_den is not None and profit_factor_den != 0
                else float("inf")
                if profit_factor_num is not None and profit_factor_num > 0
                else 0.0
            )

            bet_list_decimal = trade_log["effective_bet_fraction"].to_list()
            bet_list_float = [
                float(d) if d is not None and d.is_finite() else np.nan
                for d in bet_list_decimal
            ]
            bet_frac_float = pl.Series(
                "bet_frac_float", bet_list_float, dtype=pl.Float64
            )
            avg_bet_fraction = (
                bet_frac_float.drop_nans().mean()
                if not bet_frac_float.is_empty()
                else 0.0
            )
        else:
            win_rate = 0.0
            avg_profit = 0.0
            avg_loss = 0.0
            profit_factor = 0.0
            avg_bet_fraction = 0.0

        report_data = {
            "strategy": f"Probabilistic Betting ("
            f"Kelly Fraction: {self.config.kelly_fraction}, "
            f"Max Leverage: {self.config.max_leverage}, "
            f"Max Risk/Trade: {self.config.max_risk_per_trade * 100:.1f}%, "
            f"f* Thresh: {self.config.f_star_threshold}, "
            f"M2 Thresh: {self.config.m2_proba_threshold}"
            f")",
            "initial_capital": initial_capital,
            "final_capital": final_capital,
            "total_return_pct": total_return * Decimal("100.0"),
            "sharpe_ratio_annual": sharpe_ratio,
            "sortino_ratio_annual": sortino_ratio,
            "max_drawdown_pct": max_drawdown * 100,
            "total_trades": total_trades,
            "win_rate_pct": win_rate * 100,
            "average_profit": avg_profit,
            "average_loss": avg_loss,
            "profit_factor": profit_factor,
            "average_effective_bet_fraction_pct": avg_bet_fraction * 100,
            "data_period_start": str(results_df["timestamp"].min())
            if not results_df.is_empty()
            else "N/A",
            "data_period_end": str(results_df["timestamp"].max())
            if not results_df.is_empty()
            else "N/A",
        }

        print("\n" + "=" * 50)
        print("    Backtest Performance Report")
        print("=" * 50)
        print(f" Strategy:             {report_data['strategy']}")
        print(f" Initial Capital:      {report_data['initial_capital']:,.2f}")
        if (
            isinstance(report_data["final_capital"], Decimal)
            and report_data["final_capital"].is_finite()
        ):
            print(f" Final Capital:        {report_data['final_capital']:,.2f}")
        else:
            print(f" Final Capital:        {report_data['final_capital']}")

        if (
            isinstance(report_data["total_return_pct"], Decimal)
            and report_data["total_return_pct"].is_finite()
        ):
            print(f" Total Return:         {report_data['total_return_pct']:,.2f}%")
        else:
            print(f" Total Return:         {report_data['total_return_pct']}%")
        print(f" Sharpe Ratio (Ann.):  {report_data['sharpe_ratio_annual']:.2f}")
        print(f" Sortino Ratio (Ann.): {report_data['sortino_ratio_annual']:.2f}")
        print(f" Max Drawdown:         {report_data['max_drawdown_pct']:.2f}%")
        print("-" * 50)
        print(f" Total Trades:         {report_data['total_trades']}")
        print(f" Win Rate:             {report_data['win_rate_pct']:.2f}%")
        print(f" Average Profit:       {report_data['average_profit']:,.3f}")
        print(f" Average Loss:         {report_data['average_loss']:,.3f}")
        print(f" Profit Factor:        {report_data['profit_factor']:.2f}")
        print(
            f" Avg. Bet Fraction:    {report_data['average_effective_bet_fraction_pct']:.2f}%"
        )
        print("-" * 50)
        print(
            f" Period:               {report_data['data_period_start']} to {report_data['data_period_end']}"
        )
        print("=" * 50)

        FINAL_REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(FINAL_REPORT_PATH, "w") as f:
            json.dump(report_data, f, indent=4, default=str)
        logging.info(f"Performance report saved to {FINAL_REPORT_PATH}")

        logging.info("Generating equity curve and drawdown chart...")
        if results_df.is_empty():
            logging.warning("No data available to generate equity curve chart.")
            return
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
            equity_list_float = [
                float(e) if e is not None and e.is_finite() else np.nan
                for e in results_df["equity"].to_list()
            ]
            drawdown_list = (
                drawdown.to_list()
                if not drawdown.is_empty()
                else [0] * len(timestamps_list)
            )

            ax1.plot(
                timestamps_list,
                equity_list_float,
                label="Equity Curve",
                color="dodgerblue",
            )
            ax1.set_title(
                f"Equity Curve (Kelly Fraction: {self.config.kelly_fraction}, Max Lev: {self.config.max_leverage}, Max Risk: {self.config.max_risk_per_trade * 100:.1f}%)",
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
            except Exception:
                ax1.ticklabel_format(style="plain", axis="y")

            ax2.fill_between(timestamps_list, drawdown_list, 0, color="red", alpha=0.3)
            ax2.set_title("Drawdown", fontsize=16)
            ax2.set_ylabel("Drawdown (%)")
            ax2.yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1.0, decimals=1))
            ax2.grid(True)
            plt.tight_layout()
            plt.savefig(EQUITY_CURVE_PATH)
            logging.info(f"Saved results chart to {EQUITY_CURVE_PATH}")
            plt.close(fig)
        except Exception as e:
            logging.error(f"Failed to generate equity curve chart: {e}", exc_info=True)

        if not trade_log.is_empty():
            trade_log_output_path = FINAL_REPORT_PATH.parent / "detailed_trade_log.csv"
            try:
                trade_log_for_csv = trade_log.clone()

                object_cols = [
                    col
                    for col in trade_log_for_csv.columns
                    if trade_log_for_csv[col].dtype == pl.Object
                ]
                # ★★★ 修正: float_cols に新しい float 列も含める ★★★
                float_cols = [
                    col
                    for col in trade_log_for_csv.columns
                    if trade_log_for_csv[col].dtype in [pl.Float64, pl.Float32]
                ]
                # ★★★ 修正ここまで ★★★
                # ★★★ 追加: int_cols も定義 ★★★
                int_cols = [
                    col
                    for col in trade_log_for_csv.columns
                    if trade_log_for_csv[col].dtype in [pl.Int64, pl.Int8]
                ]
                # ★★★ 追加ここまで ★★★

                converted_series_list = []
                for col_name in object_cols:
                    decimal_list = trade_log_for_csv[col_name].to_list()
                    float_list = [
                        float(d) if d is not None and d.is_finite() else None
                        for d in decimal_list
                    ]
                    converted_series_list.append(
                        pl.Series(col_name, float_list, dtype=pl.Float64)
                    )

                if converted_series_list:
                    trade_log_for_csv = trade_log_for_csv.with_columns(
                        converted_series_list
                    )

                format_expressions = []

                format_expressions.append(
                    pl.col("timestamp")
                    .dt.strftime("%Y-%m-%d %H:%M:%S")
                    .alias("timestamp")
                )

                if "pnl" in trade_log_for_csv.columns:
                    format_expressions.append(pl.col("pnl").round(0).alias("pnl"))
                if "capital_after_trade" in trade_log_for_csv.columns:
                    format_expressions.append(
                        pl.col("capital_after_trade")
                        .round(0)
                        .alias("capital_after_trade")
                    )

                other_object_cols = [
                    c for c in object_cols if c not in ["pnl", "capital_after_trade"]
                ]
                for col_name in other_object_cols:
                    if col_name in trade_log_for_csv.columns:
                        format_expressions.append(
                            pl.col(col_name).round(2).alias(col_name)
                        )

                # ★★★ 修正: このループが lot_size, atr_value, sl_multiplier, pt_multiplier を自動的に処理 ★★★
                for col_name in float_cols:
                    # atr_value は少し桁数を多く残す (例: 4桁)
                    round_digits = 4 if col_name == "atr_value" else 2
                    format_expressions.append(
                        pl.col(col_name).round(round_digits).alias(col_name)
                    )
                # ★★★ 修正ここまで ★★★
                # ★★★ 追加: 整数列はそのまま (フォーマット不要) ★★★
                # for col_name in int_cols:
                #     pass # No formatting needed for integers like label, direction
                # ★★★ 追加ここまで ★★★

                if format_expressions:
                    trade_log_formatted = trade_log_for_csv.with_columns(
                        format_expressions
                    )
                else:
                    trade_log_formatted = trade_log_for_csv

                trade_log_formatted.write_csv(
                    trade_log_output_path,
                    null_value="NaN",
                )
                logging.info(f"Detailed trade log saved to {trade_log_output_path}")

            except Exception as e:
                logging.error(f"Failed to save detailed trade log: {e}", exc_info=True)
        else:
            logging.info("No trades were executed, skipping detailed trade log output.")


if __name__ == "__main__":
    default_config = BacktestConfig()

    parser = argparse.ArgumentParser(
        description="Project Forge Backtest Simulator (Kelly Version with Dynamic SL)"  # Title Updated
    )
    parser.add_argument(
        "--kelly",
        type=float,
        default=default_config.kelly_fraction,
        help=f"Kelly fraction to use (e.g., 0.5 for half-kelly). Default: {default_config.kelly_fraction}",
    )
    parser.add_argument(
        "--leverage",
        type=float,
        default=default_config.max_leverage,
        help=f"Maximum leverage to apply. Default: {default_config.max_leverage}",
    )
    parser.add_argument(
        "--max-risk",
        type=float,
        default=default_config.max_risk_per_trade,
        help=f"Maximum fraction of capital to risk per trade. Default: {default_config.max_risk_per_trade}",
    )
    parser.add_argument(
        "--fstar-th",
        type=float,
        default=default_config.f_star_threshold,
        dest="fstar_th",
        help=f"Minimum f_star value to initiate a trade. Default: {default_config.f_star_threshold}",
    )
    parser.add_argument(
        "--m2-th",
        type=float,
        default=default_config.m2_proba_threshold,
        dest="m2_th",
        help=f"Minimum M2 calibrated probability to initiate a trade. Default: {default_config.m2_proba_threshold}",
    )
    parser.add_argument(
        "--oof",
        action="store_true",
        help="Run in Out-of-Fold (OOF) mode using pre-calculated predictions (S7_M2_OOF_PREDICTIONS).",
    )
    parser.add_argument(
        "--min-capital",
        type=float,
        default=default_config.min_capital_threshold,
        dest="min_capital",
        help=f"Minimum capital threshold to stop simulation (bankruptcy). Default: {default_config.min_capital_threshold}",
    )
    # --- ★★★ 修正: 不要な引数を削除し、pip価値引数を修正 ★★★ ---
    # parser.add_argument(
    #     "--sl-pips", ... ) # <- 削除
    parser.add_argument(
        "--value-per-pip",  # ★★★ 引数名変更 ★★★
        type=float,
        default=default_config.value_per_pip,  # ★★★ 参照先変更 ★★★
        dest="value_per_pip",  # ★★★ dest名変更 ★★★
        help=f"(ASSUMPTION) Assumed currency value per 1 lot per 1 pip. Default: {default_config.value_per_pip}",
    )
    # --- ★★★ 修正ここまで ★★★ ---
    parser.add_argument(
        "--test",
        type=int,
        default=default_config.test_limit_partitions,
        metavar="N",
        dest="test_limit_partitions",
        help=f"Run in test mode, limiting to the first N partitions. Default: {default_config.test_limit_partitions} (all)",
    )
    args = parser.parse_args()

    # --- ★★★ 修正: config に渡す引数を修正 ★★★ ---
    config = BacktestConfig(
        kelly_fraction=args.kelly,
        max_leverage=args.leverage,
        max_risk_per_trade=args.max_risk,
        f_star_threshold=args.fstar_th,
        m2_proba_threshold=args.m2_th,
        test_limit_partitions=args.test_limit_partitions,
        oof_mode=args.oof,
        min_capital_threshold=args.min_capital,
        # assumed_sl_pips=args.sl_pips,  # <- 削除
        value_per_pip=args.value_per_pip,  # ★★★ 修正 ★★★
    )
    # --- ★★★ 修正ここまで ★★★ ---

    if not (0 < config.max_risk_per_trade <= 1.0):
        parser.error("--max-risk must be between 0 (exclusive) and 1.0 (inclusive).")
    if config.max_leverage < 1.0:
        parser.error("--leverage must be >= 1.0.")
    # --- ★★★ 修正: 不要なバリデーションを削除し、pip価値バリデーションを修正 ★★★ ---
    # if config.assumed_sl_pips <= 0: # <- 削除
    #     parser.error("--sl-pips must be greater than 0.")
    if config.value_per_pip <= 0:  # ★★★ 修正 ★★★
        parser.error("--value-per-pip must be greater than 0.")
    # --- ★★★ 修正ここまで ★★★ ---

    simulator = BacktestSimulator(config)
    simulator.run()
