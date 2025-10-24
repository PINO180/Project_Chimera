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

from decimal import Decimal, getcontext, ROUND_HALF_UP

# --- Decimal の精度を設定 (50桁あれば十分のはず) ---
getcontext().prec = 5000  # <--- これを追加

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
    # ★ 追加: OOF予測ファイルのパス
    oof_predictions_path: Path = S7_M2_OOF_PREDICTIONS
    feature_list_path: Path = S3_FEATURES_FOR_TRAINING
    m1_model_path: Path = S7_M1_MODEL_PKL
    m2_model_path: Path = S7_M2_MODEL_PKL
    m1_calibrator_path: Path = S7_M1_CALIBRATED
    m2_calibrator_path: Path = S7_M2_CALIBRATED
    kelly_fraction: float = 1.0
    max_leverage: float = 100  # レバレッジ (前回修正済み)
    # --- ★ 追加: 1取引あたりの最大リスク割合 ---
    max_risk_per_trade: float = 0.50  # 例: 資金の10%0.1を上限とする

    f_star_threshold: float = 0.0  # 例: f_star 閾値 (0.0 なら実質無効)
    m2_proba_threshold: float = 0.5  # 例: M2 確率閾値

    test_limit_partitions: int = 0
    # ★★★ ここに oof_mode を追加 ★★★
    oof_mode: bool = False


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
        # --- ★ 修正: 初期資本を Decimal に変換 ---
        self._current_capital = Decimal(
            str(self.config.initial_capital)
        )  # 文字列経由で精度を保証

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

        # self._current_capital = self.config.initial_capital # ← (float) で上書きしてしまう
        self._current_capital = Decimal(
            str(self.config.initial_capital)
        )  # ← 正しく Decimal で初期化

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

                # --- ★★★ ここから修正 ★★★ ---
                # OOFモードかどうかに関わらず、常に _run_ai_predictions を呼ぶ
                # (この関数内で OOF/In-Sample を分岐させる)
                df_chunk_predicted = self._run_ai_predictions(df_chunk)
                # --- ★★★ 修正ここまで ★★★ ---

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

    # /workspace/models/backtest_simulator.py (修正箇所: _prepare_data)

    def _prepare_data(self) -> Tuple[pl.LazyFrame, pl.DataFrame]:
        if not self.config.oof_mode:
            # --- 従来（イン・サンプル）のロジック ---
            logging.info(
                f"Preparing data from {self.config.simulation_data_path} (In-Sample Mode)..."
            )
            required_cols_set = set(self.features_base)
            required_cols_set.update(["timestamp", "label", "payoff_ratio"])
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
                    f"CRITICAL: Required columns not found in the dataset schema! Missing: {missing_cols}"
                )

            lf = lf.select(required_cols).sort("timestamp")

        else:
            # --- ★★★ OOFモードのロジック (修正) ★★★ ---
            logging.info(
                f"Preparing data from {self.config.simulation_data_path} and {self.config.oof_predictions_path} (OOF Mode)..."
            )

            # 1. S6からシミュレーションに必要な基本情報をロード
            base_cols = ["timestamp", "label", "payoff_ratio"]
            try:
                base_glob_path = str(self.config.simulation_data_path / "**/*.parquet")
                base_lf = pl.scan_parquet(base_glob_path).select(base_cols)
            except Exception as e:
                logging.error(f"Failed to scan base data (S6): {e}", exc_info=True)
                raise

            # 2. S7からOOF予測結果をロード
            #    ★ 変更: 'm2_raw_proba' カラムも必要
            oof_cols = ["timestamp", "prediction"]
            try:
                oof_lf = pl.scan_parquet(self.config.oof_predictions_path).select(
                    oof_cols
                )

                # ★ 変更: 'm2_raw_proba' (未較正) としてリネームする
                oof_lf = oof_lf.rename({"prediction": "m2_raw_proba"})

            except Exception as e:
                logging.error(
                    f"Failed to scan OOF predictions (S7): {e}", exc_info=True
                )
                raise

            # 3. 2つの LazyFrame を timestamp で結合
            logging.info("Joining base data (S6) with OOF predictions (S7)...")
            lf = base_lf.join(oof_lf, on="timestamp").sort("timestamp")

            # ★ 変更: OOFモードで必要なカラムリスト
            required_cols = base_cols + ["m2_raw_proba"]
            # ★★★ OOFロジックここまで ★★★

        logging.info("Discovering partitions...")
        partitions_df = (
            lf.select(pl.col("timestamp").dt.date().alias("date"))
            .unique()
            .collect()
            .sort("date")
        )

        # パーティションが空でないか確認
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
            # --- 従来（イン・サンプル）のロジック ---
            logging.debug("  -> Mode: In-Sample (Re-predicting)...")
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
                    f"Error during M1 prediction/calibration (In-Sample): {e}",
                    exc_info=True,
                )
                raise

            logging.debug("  -> Step 2/2: M2 Prediction & Calibration...")
            try:
                X_m2 = df_chunk.select(self.features_m2).fill_null(0).to_numpy()
                raw_m2_proba = self.m2_model.predict(X_m2)
                calibrated_m2_proba = self.m2_calibrator.predict(raw_m2_proba)  # ★ 較正
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

        else:
            # --- ★★★ 新規（OOFモード）のロジック ★★★ ---
            logging.debug("  -> Mode: OOF (Loading pre-calculated)...")
            try:
                # 1. 未較正のOOF予測確率をNumpy配列として取得
                raw_m2_proba_oof = df_chunk["m2_raw_proba"].to_numpy()

                # 2. 予測をスキップし、較正のみを実行
                logging.debug("  -> Step 1/1: M2 Calibration (OOF)...")
                calibrated_m2_proba = self.m2_calibrator.predict(
                    raw_m2_proba_oof
                )  # ★ 較正
                calibrated_m2_proba = np.clip(calibrated_m2_proba, 0.0, 1.0)

                # 3. 較正済みの確率をカラムに追加
                df_chunk = df_chunk.with_columns(
                    pl.Series("m2_calibrated_proba", calibrated_m2_proba)
                )
                logging.debug("AI processing (OOF) completed.")
                return df_chunk
            except Exception as e:
                logging.error(f"Error during M2 calibration (OOF): {e}", exc_info=True)
                raise

    # /workspace/models/backtest_simulator.py (修正版: _run_simulation_loop)
    def _run_simulation_loop(
        self, df_chunk: pl.DataFrame
    ) -> Tuple[pl.DataFrame, pl.DataFrame]:
        trade_log_chunk = []
        equity_values_chunk = []
        # --- ★ 修正: 現在の資本を Decimal で受け取る ---
        current_capital = self._current_capital
        # --- ★ 修正: Decimal 定数を定義 ---
        DECIMAL_ZERO = Decimal("0.0")
        DECIMAL_ONE = Decimal("1.0")
        DECIMAL_MAX_RISK = Decimal(str(self.config.max_risk_per_trade))
        DECIMAL_KELLY_FRACTION = Decimal(str(self.config.kelly_fraction))
        DECIMAL_F_STAR_THRESHOLD = Decimal(str(self.config.f_star_threshold))

        timestamps_chunk = df_chunk["timestamp"].to_list()
        p_m2_calibrated = df_chunk["m2_calibrated_proba"].to_numpy()
        labels_chunk = df_chunk["label"].to_numpy()

        # --- ★ 修正: payoff_ratio を Decimal に変換 (None や NaN に対応) ---
        payoff_ratios_chunk = []
        for b_float in df_chunk["payoff_ratio"].to_list():
            if b_float is None or not np.isfinite(b_float) or b_float <= 0:
                logging.warning(
                    f"Invalid or non-positive payoff_ratio {b_float} encountered, using Decimal('2.0')"
                )
                payoff_ratios_chunk.append(Decimal("2.0"))
            else:
                payoff_ratios_chunk.append(Decimal(str(b_float)))
        # --- ★ 修正ここまで ---

        for i in range(len(df_chunk)):
            # --- ★ 修正: 資本 <= 0 のチェック (Decimal) ---
            if current_capital <= DECIMAL_ZERO:
                equity_values_chunk.append(DECIMAL_ZERO)
                # ログ用変数は初期化 (Decimal)
                pnl, base_bet_fraction, capped_bet_fraction, effective_bet_fraction = (
                    DECIMAL_ZERO,
                ) * 4
                kelly_f_star, f_star = (DECIMAL_ZERO,) * 2
                p = p_m2_calibrated[i]  # float
                b = payoff_ratios_chunk[i]  # Decimal
                actual_label = labels_chunk[i]
                should_trade = False
                continue
            else:
                p_float = p_m2_calibrated[i]  # float
                b = payoff_ratios_chunk[i]  # Decimal
                # --- ★ 修正: 計算はすべて Decimal で ---
                p_decimal = Decimal(str(p_float))
                q = DECIMAL_ONE - p_decimal
                # --- ★ 修正: kelly_f_star 計算時のゼロ除算チェック強化 ---
                if b <= DECIMAL_ZERO:
                    kelly_f_star = DECIMAL_ZERO
                else:
                    kelly_f_star = (b * p_decimal - q) / b

                # --- ★ 修正: max を Decimal.copy_abs().max() で確実に Decimal 化 ---
                f_star = (
                    kelly_f_star.copy_abs().max(DECIMAL_ZERO)
                    if kelly_f_star.is_finite()
                    else DECIMAL_ZERO
                )
                # --- ★ 修正ここまで ---

                # デバッグログの float 化 (任意)
                logging.debug(f"Calculated f_star: {float(f_star):.4f}")

                # --- ★ 修正: 取引判断 (p は float のまま比較, f_star は Decimal で比較) ---
                should_trade = (f_star > DECIMAL_F_STAR_THRESHOLD) and (
                    p_float > self.config.m2_proba_threshold
                )

                if should_trade:
                    # --- ★ 修正: 計算はすべて Decimal で ---
                    base_bet_fraction = f_star * DECIMAL_KELLY_FRACTION
                    # --- ★ 修正: min を Decimal.copy_abs().min() で確実に Decimal 化 ---
                    effective_bet_fraction = (
                        base_bet_fraction.copy_abs()
                        .min(DECIMAL_MAX_RISK)
                        .copy_abs()
                        .min(DECIMAL_ONE)
                    )
                    # --- ★ 修正ここまで ---
                    capped_bet_fraction = effective_bet_fraction  # ログ用

                    if effective_bet_fraction > DECIMAL_ZERO:
                        pnl = DECIMAL_ZERO
                        actual_label = labels_chunk[i]
                        if actual_label == 1:
                            # --- ★ 修正なし: PnL 計算 (Decimal * Decimal * Decimal) ---
                            pnl = current_capital * effective_bet_fraction * b
                        elif actual_label == -1:
                            # --- ★★★ 再々修正: 単項マイナスではなく copy_negate() を使用 ★★★ ---
                            loss_amount = (
                                current_capital * effective_bet_fraction * DECIMAL_ONE
                            )
                            pnl = (
                                loss_amount.copy_negate()
                            )  # Decimal の negate メソッドを使う
                            # --- ★★★ 再々修正ここまで ★★★ ---

                        # --- ★ 修正なし: 次の資本 (Decimal + Decimal) ---
                        next_capital = current_capital + pnl

                        # --- ★ 修正: ログには float ではなく Decimal のまま記録 ---
                        trade_log_chunk.append(
                            {
                                "timestamp": timestamps_chunk[i],
                                "pnl": pnl,  # float() を削除
                                "capital_after_trade": next_capital,  # float() を削除
                                "m2_calibrated_proba": p_float,  # これは元から float
                                "payoff_ratio": b,  # float() を削除
                                "kelly_f_star": kelly_f_star,  # float() を削除
                                "f_star": f_star,  # float() を削除
                                "base_bet_fraction": base_bet_fraction,  # float() を削除
                                "capped_bet_fraction": capped_bet_fraction,  # float() を削除
                                "effective_bet_fraction": effective_bet_fraction,  # float() を削除
                                "label": actual_label,
                            }
                        )
                        current_capital = next_capital
                    else:
                        should_trade = False
                        pnl = DECIMAL_ZERO
                        actual_label = labels_chunk[i]
                else:  # 取引しなかった場合
                    pnl = DECIMAL_ZERO
                    base_bet_fraction = DECIMAL_ZERO
                    capped_bet_fraction = DECIMAL_ZERO
                    effective_bet_fraction = DECIMAL_ZERO
                    actual_label = labels_chunk[i]

            equity_values_chunk.append(current_capital)  # Decimal のまま追加

        self._current_capital = current_capital  # Decimal のまま保持

        # ★★★ ここから修正 ★★★
        # Polars に渡す前に、Decimal を float に変換する。
        # Inf/NaN になる可能性があるが、それは後続の _analyze_and_report で処理する。

        # 1. equity を float リストに変換
        equity_values_float = [
            float(e) if e is not None and e.is_finite() else np.nan
            for e in equity_values_chunk
        ]

        results_chunk_df = pl.DataFrame(
            {
                "timestamp": timestamps_chunk,
                "equity": pl.Series("equity", equity_values_float, dtype=pl.Float64),
            }
        )

        # 2. trade_log_chunk の Decimal も float に変換
        trade_log_chunk_float = []
        for row in trade_log_chunk:
            trade_log_chunk_float.append(
                {
                    "timestamp": row["timestamp"],
                    "pnl": float(row["pnl"]) if row["pnl"].is_finite() else np.nan,
                    "capital_after_trade": float(row["capital_after_trade"])
                    if row["capital_after_trade"].is_finite()
                    else np.nan,
                    "m2_calibrated_proba": row["m2_calibrated_proba"],  # float
                    "payoff_ratio": float(row["payoff_ratio"])
                    if row["payoff_ratio"].is_finite()
                    else np.nan,
                    "kelly_f_star": float(row["kelly_f_star"])
                    if row["kelly_f_star"].is_finite()
                    else np.nan,
                    "f_star": float(row["f_star"])
                    if row["f_star"].is_finite()
                    else np.nan,
                    "base_bet_fraction": float(row["base_bet_fraction"])
                    if row["base_bet_fraction"].is_finite()
                    else np.nan,
                    "capped_bet_fraction": float(row["capped_bet_fraction"])
                    if row["capped_bet_fraction"].is_finite()
                    else np.nan,
                    "effective_bet_fraction": float(row["effective_bet_fraction"])
                    if row["effective_bet_fraction"].is_finite()
                    else np.nan,
                    "label": row["label"],  # int
                }
            )

        # 3. DataFrame 作成 (dtype=pl.Object は不要になった)
        trade_log_schema = {
            "timestamp": pl.Datetime,
            "pnl": pl.Float64,
            "capital_after_trade": pl.Float64,
            "m2_calibrated_proba": pl.Float64,
            "payoff_ratio": pl.Float64,
            "kelly_f_star": pl.Float64,
            "f_star": pl.Float64,
            "base_bet_fraction": pl.Float64,
            "capped_bet_fraction": pl.Float64,
            "effective_bet_fraction": pl.Float64,
            "label": pl.Int64,
        }
        trade_log_chunk_df = pl.DataFrame(
            trade_log_chunk_float, schema=trade_log_schema
        )
        # ★★★ 修正ここまで ★★★

        return results_chunk_df, trade_log_chunk_df

    # /workspace/models/backtest_simulator.py (修正版: _analyze_and_report)
    def _analyze_and_report(self, results_df: pl.DataFrame, trade_log: pl.DataFrame):
        logging.info("Analyzing results and generating report...")

        # --- ★ 修正: Decimal定数を定義 (レポート表示用) ---
        DECIMAL_ONE = Decimal("1.0")
        DECIMAL_ZERO = Decimal("0.0")
        # ★ 修正: initial_capital は config から float で取得
        initial_capital_float = self.config.initial_capital

        if results_df.is_empty():
            logging.error("No simulation results to analyze.")
            final_capital_float = initial_capital_float
            total_return_float = 0.0
            sharpe_ratio = 0.0
            sortino_ratio = 0.0
            max_drawdown = 0.0
            daily_returns = pl.Series(dtype=pl.Float64)
            drawdown = pl.Series(dtype=pl.Float64)
        else:
            # --- ★ 修正: final_capital は results_df (Float64) から取得 ---
            final_capital_float = (
                results_df["equity"][-1]
                if not results_df.is_empty() and results_df["equity"][-1] is not None
                else initial_capital_float
            )
            total_return_float = (
                (final_capital_float / initial_capital_float - 1.0)
                if initial_capital_float > 0
                else 0.0
            )

            # --- ★ 修正: daily_returns を Float64 で計算 ---
            daily_equity_series = (
                results_df.group_by(pl.col("timestamp").dt.date().alias("date"))
                .agg(pl.first("equity"))  # 'equity' は Float64
                .sort("date")["equity"]
            )

            # pct_change() は自動的に NaN を処理する
            daily_returns = daily_equity_series.pct_change().fill_null(0.0)
            # --- ★ 修正ここまで ---

            num_trading_days = len(daily_returns)
            if num_trading_days > 1:
                # --- ★ 修正: drop_nans() で Inf/NaN を無視 ---
                std_daily_return = (
                    daily_returns.drop_nans().std()
                    if not daily_returns.is_empty()
                    else 0
                )
                if std_daily_return is not None and std_daily_return > 0:
                    mean_daily_return = daily_returns.drop_nans().mean()
                    # --- ★ 修正ここまで ---
                    sharpe_ratio = (
                        (mean_daily_return / std_daily_return) * np.sqrt(252)
                        if mean_daily_return is not None
                        else 0.0
                    )
                    negative_returns = daily_returns.filter(daily_returns < 0)
                    # --- ★ 修正: drop_nans() で Inf/NaN を無視 ---
                    downside_std = (
                        negative_returns.drop_nans().std()
                        if not negative_returns.is_empty()
                        else 0
                    )
                    # --- ★ 修正ここまで ---
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

            # --- ★ 修正: drawdown を Float64 で計算 ---
            equity_series = results_df["equity"]  # Float64
            rolling_max = equity_series.cum_max(reverse=False)

            # (equity - rolling_max) / rolling_max
            drawdown = (equity_series / rolling_max - 1.0).fill_null(0.0)

            max_drawdown = (
                drawdown.drop_nans().min() if not drawdown.is_empty() else 0.0
            )
            # --- ★ 修正ここまで ---

        total_trades = len(trade_log)
        if total_trades > 0:
            # --- ★ 修正: trade_log の pnl (Float64) を使用 ---
            pnl_series_float = trade_log["pnl"]
            winning_pnl_float = pnl_series_float.filter(trade_log["label"] == 1)
            losing_pnl_float = pnl_series_float.filter(trade_log["label"] == -1)

            win_rate = (
                len(winning_pnl_float) / total_trades if total_trades > 0 else 0.0
            )
            # --- ★ 修正: drop_nans() で Inf/NaN を無視 ---
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
            # --- ★ 修正ここまで ---
            profit_factor = (
                abs(profit_factor_num / profit_factor_den)
                if profit_factor_den is not None and profit_factor_den != 0
                else float("inf")
                if profit_factor_num is not None and profit_factor_num > 0
                else 0.0
            )

            # --- ★ 修正: bet_fraction (Float64) を使用 ---
            bet_frac_float = trade_log["effective_bet_fraction"]
            # --- ★ 修正: drop_nans() で Inf/NaN を無視 ---
            avg_bet_fraction = (
                bet_frac_float.drop_nans().mean()
                if not bet_frac_float.is_empty()
                else 0.0
            )
            # --- ★ 修正ここまで ---
        else:
            win_rate = 0.0
            avg_profit = 0.0
            avg_loss = 0.0
            profit_factor = 0.0
            avg_bet_fraction = 0.0

        # ★★★ ここからインデントを修正 (if/else の外) ★★★
        report_data = {
            "strategy": f"Probabilistic Betting ("
            f"Kelly Fraction: {self.config.kelly_fraction}, "
            f"Max Leverage: {self.config.max_leverage}, "
            f"Max Risk/Trade: {self.config.max_risk_per_trade * 100:.1f}%, "
            f"f* Thresh: {self.config.f_star_threshold}, "
            f"M2 Thresh: {self.config.m2_proba_threshold}"
            f")",
            # ★ 修正: レポート用に Decimal に変換
            "initial_capital": Decimal(str(initial_capital_float)),
            "final_capital": Decimal(str(final_capital_float))
            if np.isfinite(final_capital_float)
            else "Inf/NaN",
            "total_return_pct": Decimal(str(total_return_float)) * Decimal("100.0")
            if np.isfinite(total_return_float)
            else "Inf/NaN",
            "sharpe_ratio_annual": sharpe_ratio,  # (float)
            "sortino_ratio_annual": sortino_ratio,  # (float)
            "max_drawdown_pct": max_drawdown * 100,  # (float)
            "total_trades": total_trades,
            "win_rate_pct": win_rate * 100,
            # ★ 修正: inf になる可能性がある float
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
        # ★★★ インデント修正ここまで ★★★

        print("\n" + "=" * 50)
        print("    Backtest Performance Report")
        print("=" * 50)
        print(f" Strategy:             {report_data['strategy']}")
        # --- ★ 修正: 科学的表記 (E) を使用 (Decimal または str) ---
        if isinstance(report_data["initial_capital"], Decimal):
            print(f" Initial Capital:      {report_data['initial_capital']:,.2f}")
        else:
            print(f" Initial Capital:      {report_data['initial_capital']}")

        if isinstance(report_data["final_capital"], Decimal):
            print(f" Final Capital:        {report_data['final_capital']:,.2E}")
        else:
            print(f" Final Capital:        {report_data['final_capital']}")

        if isinstance(report_data["total_return_pct"], Decimal):
            print(f" Total Return:         {report_data['total_return_pct']:,.2E}%")
        else:
            print(f" Total Return:         {report_data['total_return_pct']}%")
        # --- ★ 修正ここまで ---
        print(f" Sharpe Ratio (Ann.):  {report_data['sharpe_ratio_annual']:.2f}")
        print(f" Sortino Ratio (Ann.): {report_data['sortino_ratio_annual']:.2f}")
        print(f" Max Drawdown:         {report_data['max_drawdown_pct']:.2f}%")
        print("-" * 50)
        print(f" Total Trades:         {report_data['total_trades']}")
        print(f" Win Rate:             {report_data['win_rate_pct']:.2f}%")
        # --- ★ 修正: 科学的表記 (E) を使用 ---
        print(f" Average Profit:       {report_data['average_profit']:,.2E}")
        print(f" Average Loss:         {report_data['average_loss']:,.2E}")
        # --- ★ 修正ここまで ---
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
        # ★ 修正: json.dump は default=str を使う
        with open(FINAL_REPORT_PATH, "w") as f:
            json.dump(
                report_data, f, indent=4, default=str
            )  # default=str で Decimal/Inf/NaN を文字列に
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
            # --- ★ 修正: equity (Float64) を使用 ---
            equity_list_float = results_df["equity"].to_list()
            # --- ★ 修正ここまで ---
            drawdown_list = (
                drawdown.to_list()  # これは既に float のリスト
                if not drawdown.is_empty()
                else [0] * len(timestamps_list)
            )

            ax1.plot(
                timestamps_list,
                equity_list_float,  # float リストを使用
                label="Equity Curve",
                color="dodgerblue",
            )
            ax1.set_title(
                f"Equity Curve (Kelly Fraction: {self.config.kelly_fraction}, Max Lev: {self.config.max_leverage}, Max Risk: {self.config.max_risk_per_trade * 100:.1f}%)",
                fontsize=16,
            )
            ax1.set_ylabel("Equity")
            ax1.grid(True)
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
    # ★★★ ここに OOF モードの引数を追加 ★★★
    parser.add_argument(
        "--oof",
        action="store_true",  # この引数があれば True になる
        help="Run in Out-of-Fold (OOF) mode using pre-calculated predictions (S7_M2_OOF_PREDICTIONS).",
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
        oof_mode=args.oof,
    )

    # --- (以降の検証ロジック、シミュレータ実行は同じ) ---
    if not (0 < config.max_risk_per_trade <= 1.0):
        parser.error("--max-risk must be between 0 (exclusive) and 1.0 (inclusive).")
    if config.max_leverage < 1.0:
        parser.error("--leverage must be >= 1.0.")

    simulator = BacktestSimulator(config)
    simulator.run()
