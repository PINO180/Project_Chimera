# /workspace/models/backtest_simulator.py
# [修正版: Chunking (Streaming) 導入によるメモリ効率化 + Schemaログ抑制]
# [修正版: Booster API の .predict() を使うように修正]
# [修正版: AttributeError (clip_min) 修正 + レバレッジ調整]
# [修正版: 最大リスク上限導入 + グラフエラー修正]

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
import matplotlib.ticker as mtick  # ★ インポート追加
import seaborn as sns
from tqdm import tqdm
import gc

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
    feature_list_path: Path = S3_FEATURES_FOR_TRAINING
    m1_model_path: Path = S7_M1_MODEL_PKL
    m2_model_path: Path = S7_M2_MODEL_PKL
    m1_calibrator_path: Path = S7_M1_CALIBRATED
    m2_calibrator_path: Path = S7_M2_CALIBRATED
    kelly_fraction: float = 0.5
    max_leverage: float = 100  # レバレッジ (前回修正済み)
    # --- ★ 追加: 1取引あたりの最大リスク割合 ---
    max_risk_per_trade: float = 0.02  # 例: 資金の10%0.1を上限とする

    f_star_threshold: float = 0.0  # 例: f_star 閾値 (0.0 なら実質無効)
    m2_proba_threshold: float = 0.6  # 例: M2 確率閾値

    test_limit_partitions: int = 0


class BacktestSimulator:
    def __init__(self, config: BacktestConfig):
        self.config = config
        # --- モデル/較正器/特徴量のロード (変更なし) ---
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
        self._current_capital = self.config.initial_capital

    def _load_model(self, path: Path, name: str):
        # (変更なし)
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
        # (変更なし)
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
            f"Max Leverage = {self.config.max_leverage}, Max Risk/Trade = {self.config.max_risk_per_trade * 100:.1f}%"  # リスク上限もログ出力
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

        self._current_capital = self.config.initial_capital

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
            except Exception as e:
                logging.error(
                    f"Error collecting data for partition {current_date}: {e}",
                    exc_info=True,
                )
                continue

            try:
                if self._current_capital <= 0:
                    logging.warning(
                        f"Capital depleted before processing {current_date}. Skipping remaining partitions."
                    )
                    # 破産した場合、以降のパーティション処理をスキップ
                    break  # ループを抜ける

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
            # 結果がない場合はレポート生成前に終了
            return

        logging.info("Concatenating results from all partitions...")
        # 結合時にエラーが発生する可能性も考慮
        try:
            final_results_df = pl.concat(all_results_dfs).sort("timestamp")
            # trade_log は空の DataFrame が含まれる可能性があるので、空でないものだけ結合
            final_trade_log_df = (
                pl.concat([df for df in all_trade_logs if not df.is_empty()]).sort(
                    "timestamp"
                )
                if any(not df.is_empty() for df in all_trade_logs)
                else pl.DataFrame()
            )

        except Exception as e:
            logging.error(f"Error concatenating results: {e}", exc_info=True)
            return  # 結合エラー時も終了

        # 結合後の DataFrame が空でないか最終確認
        if final_results_df.is_empty():
            logging.error(
                "Concatenated results DataFrame is empty. Cannot generate report."
            )
            return

        self._analyze_and_report(final_results_df, final_trade_log_df)
        logging.info("### Backtest Simulator: FINISHED ###")

    def _prepare_data(self) -> Tuple[pl.LazyFrame, pl.DataFrame]:
        # (変更なし)
        logging.info(f"Preparing data from {self.config.simulation_data_path}...")

        required_cols_set = set(self.features_base)
        required_cols_set.update(["timestamp", "label", "payoff_ratio"])
        required_cols = list(required_cols_set)

        try:
            glob_path = str(self.config.simulation_data_path / "**/*.parquet")
            logging.info(f"Scanning Parquet files using glob: {glob_path}")
            lf = pl.scan_parquet(glob_path)
            schema = lf.collect_schema()
        except Exception as e:
            logging.error(f"Failed to scan Parquet files: {e}", exc_info=True)
            raise

        missing_cols = [col for col in required_cols if col not in schema]
        if missing_cols:
            raise ValueError(
                f"CRITICAL: Required columns not found in the dataset schema! Missing: {missing_cols}"
            )

        lf = lf.select(required_cols).sort("timestamp")

        logging.info("Discovering partitions...")
        partitions_df = (
            lf.select(pl.col("timestamp").dt.date().alias("date"))
            .unique()
            .collect()
            .sort("date")
        )
        logging.info(f"Found {len(partitions_df)} partitions.")

        logging.info(
            "LazyFrame prepared. Data will be collected partition by partition."
        )
        return lf, partitions_df

    def _run_ai_predictions(self, df_chunk: pl.DataFrame) -> pl.DataFrame:
        # (変更なし)
        logging.debug(f"Running AI predictions for chunk (size: {len(df_chunk)})...")

        logging.debug("  -> Step 1/2: M1 Prediction & Calibration...")
        try:
            X_m1 = df_chunk.select(self.features_base).fill_null(0).to_numpy()
            raw_m1_proba = self.m1_model.predict(X_m1)
            calibrated_m1_proba = self.m1_calibrator.predict(raw_m1_proba)
            calibrated_m1_proba = np.clip(calibrated_m1_proba, 0.0, 1.0)
            df_chunk = df_chunk.with_columns(
                pl.Series("m1_pred_proba", calibrated_m1_proba)
            )
        except Exception as e:
            logging.error(
                f"Error during M1 prediction/calibration on chunk: {e}", exc_info=True
            )
            raise

        logging.debug("  -> Step 2/2: M2 Prediction & Calibration...")
        try:
            X_m2 = df_chunk.select(self.features_m2).fill_null(0).to_numpy()
            raw_m2_proba = self.m2_model.predict(X_m2)
            calibrated_m2_proba = self.m2_calibrator.predict(raw_m2_proba)
            calibrated_m2_proba = np.clip(calibrated_m2_proba, 0.0, 1.0)
            df_chunk = df_chunk.with_columns(
                pl.Series("m2_calibrated_proba", calibrated_m2_proba)
            )
            logging.debug("AI predictions for chunk completed.")
            return df_chunk
        except Exception as e:
            logging.error(
                f"Error during M2 prediction/calibration on chunk: {e}", exc_info=True
            )
            raise

    def _run_simulation_loop(
        self, df_chunk: pl.DataFrame
    ) -> Tuple[pl.DataFrame, pl.DataFrame]:
        trade_log_chunk = []
        equity_values_chunk = []
        current_capital = self._current_capital

        timestamps_chunk = df_chunk["timestamp"].to_list()
        p_m2_calibrated = df_chunk["m2_calibrated_proba"].to_numpy()
        labels_chunk = df_chunk["label"].to_numpy()
        payoff_ratios_chunk = df_chunk["payoff_ratio"].to_numpy()

        for i in range(len(df_chunk)):
            if current_capital <= 0:
                # (破産時の処理)
                # ... (変更なし) ...
                # 変数初期化 (ログ記録のため)
                pnl = 0.0
                base_bet_fraction = 0.0  # kelly計算結果
                capped_bet_fraction = 0.0  # リスク上限後 (ログ用)
                effective_bet_fraction = 0.0  # 最終リスク割合
                kelly_f_star = 0.0  # ケリー推奨値
                f_star = 0.0  # max(0, kelly)
                p = p_m2_calibrated[i]
                b = payoff_ratios_chunk[i]
                actual_label = labels_chunk[i]
                should_trade = False  # 取引しないフラグ
            else:
                p = p_m2_calibrated[i]
                b = payoff_ratios_chunk[i]
                q = 1.0 - p
                kelly_f_star = (b * p - q) / b if b > 0 else 0.0
                f_star = max(0.0, kelly_f_star) if np.isfinite(kelly_f_star) else 0.0

                # --- ★★★ ここから修正 ★★★ ---
                logging.debug(f"Calculated f_star: {f_star:.4f}")

                # 1. 取引すべきかどうかの判断
                should_trade = (f_star > self.config.f_star_threshold) and (
                    p > self.config.m2_proba_threshold
                )

                if should_trade:
                    # 2. 基本ベット割合 (ケリー推奨値 * 分数)
                    base_bet_fraction = f_star * self.config.kelly_fraction

                    # 3. 最終的なリスク割合の決定 (修正)
                    #    PnL計算(pnl = -capital * risk_fraction * 1.0)において、
                    #    「失うことを許容する資金の割合」が effective_bet_fraction となる。
                    #    max_leverage は、このリスク割合を実現する「手段」であり、
                    #    リスク割合の「計算」には含めない。
                    #
                    #    最終的なリスク割合は、
                    #    A) ケリー推奨値 (base_bet_fraction)
                    #    B) 自己設定上限 (max_risk_per_trade)
                    #    C) 絶対上限 (1.0 = 100% of capital)
                    #    の最小値となる。
                    effective_bet_fraction = min(
                        base_bet_fraction,
                        self.config.max_risk_per_trade,
                        1.0,  # 念のため100%キャップも残す
                    )

                    # (デバッグ/ログ用に、旧 capped_bet_fraction 相当の値を設定)
                    capped_bet_fraction = effective_bet_fraction

                    # 4. 最終的なベットサイズが (計算誤差等で) 0以下になっていないか確認 (旧ステップ5)
                    if effective_bet_fraction > 0:
                        pnl = 0.0
                        actual_label = labels_chunk[i]
                        if actual_label == 1:
                            pnl = current_capital * effective_bet_fraction * b
                        elif actual_label == -1:
                            pnl = -current_capital * effective_bet_fraction * 1.0

                        next_capital = current_capital + pnl

                        trade_log_chunk.append(
                            {
                                "timestamp": timestamps_chunk[i],
                                "pnl": pnl,
                                "capital_after_trade": next_capital,
                                "m2_calibrated_proba": p,
                                "payoff_ratio": b,
                                "kelly_f_star": kelly_f_star,  # ケリー推奨値
                                "f_star": f_star,  # max(0, kelly)
                                "base_bet_fraction": base_bet_fraction,  # kelly * fraction
                                "capped_bet_fraction": capped_bet_fraction,  # リスク上限後 (ログ用)
                                "effective_bet_fraction": effective_bet_fraction,  # 最終リスク割合
                                "label": actual_label,
                            }
                        )
                        current_capital = next_capital  # 資本を更新
                    else:  # ベットサイズ計算の結果 0 になった場合 (記録しない)
                        should_trade = False  # 取引しなかったことにする
                        pnl = 0.0
                        actual_label = labels_chunk[i]
                        # logging.debug(f"Trade skipped at {timestamps_chunk[i]} due to zero effective bet size.")

                # --- ★★★ 修正ここまで ★★★ ---

                # --- 取引しなかった場合の変数を初期化 (ログや次のステップのため) ---
                if not should_trade:
                    pnl = 0.0
                    base_bet_fraction = 0.0  # ケリー*割合 (取引しないので0)
                    capped_bet_fraction = 0.0  # リスク上限後 (取引しないので0)
                    effective_bet_fraction = 0.0  # レバレッジ後 (取引しないので0)
                    actual_label = labels_chunk[i]  # ラベルはそのまま

            # ★★★ 修正: この位置（ループの最後）に追加 ★★★
            # これにより、取引が実行された後（または実行されなかった後）の
            # 確定した資本額がそのtimestampの記録として残る
            equity_values_chunk.append(current_capital)

        self._current_capital = current_capital

        results_chunk_df = pl.DataFrame(
            {"timestamp": df_chunk["timestamp"], "equity": equity_values_chunk}
        )
        trade_log_chunk_df = pl.DataFrame(trade_log_chunk)

        return results_chunk_df, trade_log_chunk_df

    def _analyze_and_report(self, results_df: pl.DataFrame, trade_log: pl.DataFrame):
        # (レポート計算部分は前回修正済み)
        logging.info("Analyzing results and generating report...")
        if results_df.is_empty():
            logging.error("No simulation results to analyze.")
            final_capital = self.config.initial_capital
            initial_capital = self.config.initial_capital
            total_return = 0.0
            sharpe_ratio = 0.0
            sortino_ratio = 0.0
            max_drawdown = 0.0
            daily_returns = pl.Series(dtype=pl.Float64)
            drawdown = pl.Series(dtype=pl.Float64)
        else:
            # 最終資本は results_df の最後の equity 値
            final_capital = (
                results_df["equity"][-1]
                if not results_df.is_empty()
                else self.config.initial_capital
            )
            initial_capital = self.config.initial_capital
            total_return = (
                (final_capital / initial_capital - 1.0) if initial_capital > 0 else 0.0
            )
            daily_returns = (
                results_df.group_by(pl.col("timestamp").dt.date().alias("date"))
                .agg(pl.first("equity"))  # 各日の開始時資本を使う
                .sort("date")["equity"]
                .pct_change()
                .drop_nulls()
            )
            num_trading_days = len(daily_returns)
            if num_trading_days > 1:
                std_daily_return = (
                    daily_returns.std() if not daily_returns.is_empty() else 0
                )
                if std_daily_return is not None and std_daily_return > 0:
                    mean_daily_return = daily_returns.mean()
                    sharpe_ratio = (
                        (mean_daily_return / std_daily_return) * np.sqrt(252)
                        if mean_daily_return is not None
                        else 0.0
                    )
                    negative_returns = daily_returns.filter(daily_returns < 0)
                    downside_std = (
                        negative_returns.std() if not negative_returns.is_empty() else 0
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

            rolling_max = results_df["equity"].cum_max()
            drawdown = (results_df["equity"] - rolling_max) / rolling_max.clip(
                lower_bound=1e-9
            )
            max_drawdown = drawdown.min() if not drawdown.is_empty() else 0.0

        total_trades = len(trade_log)
        if total_trades > 0:
            winning_trades = trade_log.filter(pl.col("label") == 1)
            losing_trades = trade_log.filter(pl.col("label") == -1)
            win_rate = len(winning_trades) / total_trades
            avg_profit = (
                winning_trades["pnl"].mean() if not winning_trades.is_empty() else 0.0
            )
            avg_loss = (
                losing_trades["pnl"].mean() if not losing_trades.is_empty() else 0.0
            )
            profit_factor_num = winning_trades["pnl"].sum()
            profit_factor_den = losing_trades["pnl"].sum()
            profit_factor = (
                abs(profit_factor_num / profit_factor_den)
                if profit_factor_den is not None and profit_factor_den != 0
                else float("inf")
                if profit_factor_num is not None and profit_factor_num > 0
                else 0.0
            )
            avg_bet_fraction = (
                trade_log["effective_bet_fraction"].mean()
                if not trade_log.is_empty()
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
            f"f* Thresh: {self.config.f_star_threshold}, "  # ★ 追加
            f"M2 Thresh: {self.config.m2_proba_threshold}"  # ★ 追加
            f")",
            "initial_capital": initial_capital,
            "final_capital": final_capital,
            "total_return_pct": total_return * 100,
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
        print(f" Final Capital:        {report_data['final_capital']:,.2f}")
        print(f" Total Return:         {report_data['total_return_pct']:.2f}%")
        print(f" Sharpe Ratio (Ann.):  {report_data['sharpe_ratio_annual']:.2f}")
        print(f" Sortino Ratio (Ann.): {report_data['sortino_ratio_annual']:.2f}")
        print(f" Max Drawdown:         {report_data['max_drawdown_pct']:.2f}%")
        print("-" * 50)
        print(f" Total Trades:         {report_data['total_trades']}")
        print(f" Win Rate:             {report_data['win_rate_pct']:.2f}%")
        print(f" Average Profit:       {report_data['average_profit']:,.2f}")
        print(f" Average Loss:         {report_data['average_loss']:,.2f}")
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
            equity_list = results_df["equity"].to_list()
            drawdown_list = (
                drawdown.to_list()
                if not drawdown.is_empty()
                else [0] * len(timestamps_list)
            )

            ax1.plot(
                timestamps_list, equity_list, label="Equity Curve", color="dodgerblue"
            )
            ax1.set_title(
                f"Equity Curve (Kelly Fraction: {self.config.kelly_fraction}, Max Lev: {self.config.max_leverage}, Max Risk: {self.config.max_risk_per_trade * 100:.1f}%)",  # リスク上限追加
                fontsize=16,
            )
            ax1.set_ylabel("Equity")
            ax1.grid(True)
            ax1.ticklabel_format(style="plain", axis="y")

            ax2.fill_between(timestamps_list, drawdown_list, 0, color="red", alpha=0.3)
            ax2.set_title("Drawdown", fontsize=16)
            ax2.set_ylabel("Drawdown (%)")
            # --- ★ 修正: PercentFormatter を使用 ---
            ax2.yaxis.set_major_formatter(
                mtick.PercentFormatter(xmax=1.0, decimals=1)
            )  # xmax=1.0 で割合表示
            # --- ★ 修正ここまで ---
            ax2.grid(True)
            plt.tight_layout()
            plt.savefig(EQUITY_CURVE_PATH)
            logging.info(f"Saved results chart to {EQUITY_CURVE_PATH}")
            plt.close(fig)
        except Exception as e:
            logging.error(f"Failed to generate equity curve chart: {e}", exc_info=True)


if __name__ == "__main__":
    # --- ★ 追加: まずデフォルト設定のConfigインスタンスを作成 ---
    default_config = BacktestConfig()
    # --- ★ 追加ここまで ---

    parser = argparse.ArgumentParser(
        description="Project Forge Backtest Simulator (Kelly Version)"
    )
    parser.add_argument(
        "--kelly",
        type=float,
        # --- ★ 修正: Configからデフォルト値を取得 ---
        default=default_config.kelly_fraction,
        # --- ★ 修正ここまで ---
        help=f"Kelly fraction to use (e.g., 0.5 for half-kelly). Default: {default_config.kelly_fraction}",  # ヘルプ表示も動的に
    )
    parser.add_argument(
        "--leverage",
        type=float,
        # --- ★ 修正: Configからデフォルト値を取得 ---
        default=default_config.max_leverage,
        # --- ★ 修正ここまで ---
        help=f"Maximum leverage to apply. Default: {default_config.max_leverage}",  # ヘルプ表示も動的に
    )
    parser.add_argument(
        "--max-risk",
        type=float,
        # --- ★ 修正: Configからデフォルト値を取得 ---
        default=default_config.max_risk_per_trade,
        # --- ★ 修正ここまで ---
        help=f"Maximum fraction of capital to risk per trade. Default: {default_config.max_risk_per_trade}",  # ヘルプ表示も動的に
    )

    # --- ★★★ ここに不足している引数を追加 ★★★ ---
    parser.add_argument(
        "--fstar-th",
        type=float,
        default=default_config.f_star_threshold,
        dest="fstar_th",  # 'dest' は config で参照する属性名と一致させる
        help=f"Minimum f_star value to initiate a trade. Default: {default_config.f_star_threshold}",
    )
    parser.add_argument(
        "--m2-th",
        type=float,
        default=default_config.m2_proba_threshold,
        dest="m2_th",  # 'dest' は config で参照する属性名と一致させる
        help=f"Minimum M2 calibrated probability to initiate a trade. Default: {default_config.m2_proba_threshold}",
    )
    # --- ★★★ 追加ここまで ★★★ ---

    parser.add_argument(
        "--test",
        type=int,
        # --- ★ 修正: Configからデフォルト値を取得 ---
        default=default_config.test_limit_partitions,  # 0 がデフォルト
        # --- ★ 修正ここまで ---
        metavar="N",
        dest="test_limit_partitions",
        help=f"Run in test mode, limiting to the first N partitions. Default: {default_config.test_limit_partitions} (all)",  # ヘルプ表示も動的に
    )
    args = parser.parse_args()

    # ★ このConfig作成部分は変更なし (引数が正しく渡されるようになる)
    config = BacktestConfig(
        kelly_fraction=args.kelly,
        max_leverage=args.leverage,
        max_risk_per_trade=args.max_risk,
        f_star_threshold=args.fstar_th,
        m2_proba_threshold=args.m2_th,
        test_limit_partitions=args.test_limit_partitions,
    )

    # --- (以降の検証ロジック、シミュレータ実行は同じ) ---
    if not (0 < config.max_risk_per_trade <= 1.0):
        parser.error("--max-risk must be between 0 (exclusive) and 1.0 (inclusive).")
    if config.max_leverage < 1.0:
        parser.error("--leverage must be >= 1.0.")

    simulator = BacktestSimulator(config)
    simulator.run()
