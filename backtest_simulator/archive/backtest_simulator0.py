# /workspace/models/backtest_simulator.py

import sys
from pathlib import Path
import logging
import argparse
from dataclasses import dataclass, field
from typing import Dict, Any

import polars as pl
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm  # ▼▼▼ 修正点: tqdmをインポート ▼▼▼

# --- プロジェクトのルートディレクトリをPythonの検索パスに追加 ---
project_root = Path(__file__).resolve().parents[1]
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

# --- blueprintから必要なパスをインポート ---
from blueprint import (
    S6_WEIGHTED_DATASET,
    S3_FEATURES_FOR_TRAINING,
    S7_M1_MODEL_PKL,
    S7_M2_MODEL_PKL,
    S7_M1_CALIBRATED,
    S7_M2_CALIBRATED,
)

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

    initial_capital: float = 10000.0
    simulation_data_path: Path = S6_WEIGHTED_DATASET
    feature_list_path: Path = S3_FEATURES_FOR_TRAINING
    m1_model_path: Path = S7_M1_MODEL_PKL
    m2_model_path: Path = S7_M2_MODEL_PKL
    m1_calibrator_path: Path = S7_M1_CALIBRATED
    m2_calibrator_path: Path = S7_M2_CALIBRATED
    m1_entry_threshold: float = 0.5
    m2_entry_threshold: float = 0.65
    kelly_fraction: float = 1.0
    leverage: float = 2000.0
    profit_take_multiplier: float = 2.0
    stop_loss_multiplier: float = 1.0
    atr_column_name: str = "e1c_atr_21"
    test_limit_rows: int = 0


class BacktestSimulator:
    def __init__(self, config: BacktestConfig):
        self.config = config
        self.m1_model = self._load_model(config.m1_model_path, "M1 Model")
        self.m2_model = self._load_model(config.m2_model_path, "M2 Model")
        self.m1_calibrator = self._load_model(
            config.m1_calibrator_path, "M1 Calibrator"
        )
        self.m2_calibrator = self._load_model(
            config.m2_calibrator_path, "M2 Calibrator"
        )
        self.features = self._load_features(config.feature_list_path)
        self.payoff_ratio = config.profit_take_multiplier / config.stop_loss_multiplier

    def _load_model(self, path: Path, name: str):
        logging.info(f"Loading {name} from {path}...")
        if not path.exists():
            raise FileNotFoundError(f"{name} file not found: {path}")
        return joblib.load(path)

    def _load_features(self, path: Path) -> list[str]:
        logging.info(f"Loading feature list from {path}...")
        if not path.exists():
            raise FileNotFoundError(f"Feature list not found: {path}")
        with open(path, "r") as f:
            return [line.strip() for line in f if line.strip()]

    def run(self):
        logging.info("### Backtest Simulator: START ###")
        logging.info(
            f"Strategy: Kelly Fraction = {self.config.kelly_fraction}, M2 Threshold = {self.config.m2_entry_threshold}"
        )
        df = self._prepare_data()
        df = self._run_ai_predictions(df)
        results_df, trade_log = self._run_simulation_loop(df)
        self._analyze_and_report(results_df, trade_log)
        logging.info("### Backtest Simulator: FINISHED ###")

    def _prepare_data(self) -> pl.DataFrame:
        logging.info(f"Preparing data from {self.config.simulation_data_path}...")
        required_cols = self.features + [
            "timestamp",
            "label",
            self.config.atr_column_name,
            "close",
        ]
        lf = pl.scan_parquet(str(self.config.simulation_data_path / "**/*.parquet"))
        schema = lf.collect_schema()
        for col in required_cols:
            if col not in schema:
                raise ValueError(
                    f"CRITICAL: Required column '{col}' not found in the dataset schema!"
                )
        lf = lf.select(required_cols).sort("timestamp")
        if self.config.test_limit_rows > 0:
            logging.warning(
                f"--- TEST MODE: Limiting to first {self.config.test_limit_rows} rows. ---"
            )
            lf = lf.head(self.config.test_limit_rows)
        return lf.collect(streaming=True)

    def _run_ai_predictions(self, df: pl.DataFrame) -> pl.DataFrame:
        logging.info("Running AI predictions (M1 -> Calibrator -> M2 -> Calibrator)...")
        logging.info("  -> Step 1/2: Predicting with M1 and calibrating...")
        raw_m1_proba = self.m1_model.predict_proba(df.select(self.features).to_numpy())[
            :, 1
        ]
        calibrated_m1_proba = self.m1_calibrator.predict(raw_m1_proba)
        df = df.with_columns(pl.Series("m1_pred_proba", calibrated_m1_proba))
        logging.info("  -> Step 2/2: Predicting with M2 and calibrating...")
        m2_features = self.features + ["m1_pred_proba"]
        raw_m2_proba = self.m2_model.predict_proba(df.select(m2_features).to_numpy())[
            :, 1
        ]
        calibrated_m2_proba = self.m2_calibrator.predict(raw_m2_proba)
        return df.with_columns(pl.Series("m2_pred_proba", calibrated_m2_proba))

    def _run_simulation_loop(self, df: pl.DataFrame):
        logging.info("Running event-driven simulation loop...")
        capital = self.config.initial_capital
        equity_curve = [capital]
        trade_log = []
        for row in tqdm(
            df.iter_rows(named=True), total=len(df), desc="  Simulating trades"
        ):
            p_m1 = row["m1_pred_proba"]
            p_m2 = row["m2_pred_proba"]
            if (
                p_m1 > self.config.m1_entry_threshold
                and p_m2 > self.config.m2_entry_threshold
            ):
                p = p_m2
                b = self.payoff_ratio
                q = 1 - p
                kelly_f_star = (b * p - q) / b if b != 0 else 0
                if kelly_f_star > 0:
                    bet_fraction = kelly_f_star * self.config.kelly_fraction
                    bet_amount = capital * bet_fraction
                    pnl = 0
                    if row["label"] == 1:
                        pnl = bet_amount * self.payoff_ratio * self.config.leverage
                    elif row["label"] == -1:
                        pnl = -bet_amount * self.config.leverage

                    capital += pnl

                    if capital <= 0:
                        logging.warning(
                            f"ACCOUNT BLOWN at {row['timestamp']}. Capital reset to 0."
                        )
                        capital = 0
                        equity_curve.append(capital)
                        trade_log.append(
                            {
                                "timestamp": row["timestamp"],
                                "pnl": pnl,
                                "capital": capital,
                                "p_m2": p_m2,
                                "bet_fraction": bet_fraction,
                            }
                        )
                        break

                    trade_log.append(
                        {
                            "timestamp": row["timestamp"],
                            "pnl": pnl,
                            "capital": capital,
                            "p_m2": p_m2,
                            "bet_fraction": bet_fraction,
                        }
                    )
            equity_curve.append(capital)

        num_equity_points_needed = len(df) + 1
        if len(equity_curve) < num_equity_points_needed:
            equity_curve.extend(
                [equity_curve[-1]] * (num_equity_points_needed - len(equity_curve))
            )

        return pl.DataFrame(
            {"timestamp": df["timestamp"], "equity": equity_curve[1:]}
        ), pl.DataFrame(trade_log)

    def _analyze_and_report(self, results_df: pl.DataFrame, trade_log: pl.DataFrame):
        logging.info("Analyzing results and generating report...")
        if results_df.is_empty():
            logging.error("No simulation results to analyze.")
            return

        returns = results_df["equity"].pct_change().drop_nans()
        final_capital = (
            results_df["equity"][-1]
            if not results_df.is_empty()
            else self.config.initial_capital
        )
        total_return = (
            (final_capital / self.config.initial_capital) - 1
            if self.config.initial_capital > 0
            else 0
        )

        # 15分足を年率換算 (365日 * 24時間 * 4 (15分/時間))
        annualization_factor = 365 * 24 * 4
        sharpe_ratio = (
            (returns.mean() / returns.std()) * np.sqrt(annualization_factor)
            if returns.std() > 0
            else 0
        )

        rolling_max = results_df["equity"].cum_max()
        drawdown = (results_df["equity"] - rolling_max) / rolling_max
        max_drawdown = drawdown.min() if not drawdown.is_empty() else 0

        print("\n" + "=" * 50)
        print("    Backtest Performance Report")
        print("=" * 50)
        print(f" Initial Capital:    {self.config.initial_capital:,.2f}")
        print(f" Final Capital:        {final_capital:,.2f}")
        print(f" Total Return:         {total_return:.2%}")
        print(f" Sharpe Ratio (Ann.):  {sharpe_ratio:.2f}")
        print(f" Max Drawdown:         {max_drawdown:.2%}")
        print(f" Total Trades:         {len(trade_log)}")
        print("=" * 50)

        # グラフ描画
        sns.set_style("darkgrid")
        fig, (ax1, ax2) = plt.subplots(
            2, 1, figsize=(15, 10), sharex=True, gridspec_kw={"height_ratios": [3, 1]}
        )
        ax1.plot(
            results_df["timestamp"],
            results_df["equity"],
            label="Equity Curve",
            color="dodgerblue",
        )
        ax1.set_title(
            f"Equity Curve (Kelly Fraction: {self.config.kelly_fraction}, M2 Thresh: {self.config.m2_entry_threshold})",
            fontsize=16,
        )
        ax1.set_ylabel("Equity")
        ax1.grid(True)
        ax2.fill_between(results_df["timestamp"], drawdown, 0, color="red", alpha=0.3)
        ax2.set_title("Drawdown", fontsize=16)
        ax2.set_ylabel("Drawdown")
        ax2.grid(True)
        plt.tight_layout()
        plt.savefig("backtest_results.png")
        logging.info("Saved results chart to backtest_results.png")
        plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Project Forge Backtest Simulator")
    parser.add_argument(
        "--kelly",
        type=float,
        default=1.0,
        help="Kelly fraction to use (e.g., 0.5 for half-kelly).",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.65,
        help="M2 probability threshold for entry.",
    )
    parser.add_argument(
        "--test-rows",
        type=int,
        default=0,
        help="Limit simulation to the first N rows for testing.",
    )
    args = parser.parse_args()
    config = BacktestConfig(
        kelly_fraction=args.kelly,
        m2_entry_threshold=args.threshold,
        test_limit_rows=args.test_rows,
    )
    simulator = BacktestSimulator(config)
    simulator.run()
