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

    initial_capital: float = 1000.0
    simulation_data_path: Path = S6_WEIGHTED_DATASET
    oof_predictions_path: Path = S7_M2_OOF_PREDICTIONS

    feature_list_path: Path = S3_FEATURES_FOR_TRAINING

    # ★★★ [修正] 必要なUniverse (A, C, E) だけを定義 ★★★
    # Universe C (Price & ATR source)
    s2_path_c: Path = Path(
        "/workspace/data/XAUUSD/stratum_2_features_fixed/feature_value_a_vast_universeC/features_e1c_tick"
    )
    # Universe A (Skew/Kurt for V5 Gatekeeper)
    s2_path_a: Path = Path(
        "/workspace/data/XAUUSD/stratum_2_features_fixed/feature_value_a_vast_universeA/features_e1a_tick"
    )
    # Universe E (Hilbert for V5 Gatekeeper)
    s2_path_e: Path = Path(
        "/workspace/data/XAUUSD/stratum_2_features_fixed/feature_value_a_vast_universeE/features_e1e_tick"
    )

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

    max_positions: int = 1000
    base_leverage: float = 2000.0
    spread_pips: float = 16.0
    value_per_pip: float = 1.0
    is_cent_mode: bool = False


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

        # ★★★ [修正] 共通の特徴量リストを読み込み、鉄壁のフィルターを適用 ★★★
        self.features_base = self._load_features_from_txt(config.feature_list_path)
        self.features_m1 = self.features_base.copy()

        # M2特徴量には m1_pred_proba を追加
        self.features_m2 = ["m1_pred_proba"] + self.features_base

        # In-Sampleモードで参照される変数を初期化（エラー回避）
        self.context_features_m2 = []

        # ★★★ 追加: 前日のデータを持ち越すためのバッファ ★★★
        self.prev_chunk_buffer = None

        # --- [セントモード設定] ---
        if self.config.is_cent_mode:
            logging.info("--- 🛡️ RUNNING IN CENT ACCOUNT MODE 🛡️ ---")
            logging.info(
                "Converting Initial Capital and Pip Value to Cents (x100) for internal calculation."
            )
            self._current_capital = Decimal(str(self.config.initial_capital)) * Decimal(
                "100"
            )
            self.config.value_per_pip *= 100.0
            self.config.min_capital_threshold *= 100.0
        else:
            self._current_capital = Decimal(str(self.config.initial_capital))

    # ★ テキスト形式の特徴量リスト読み込み用メソッド
    def _load_features_from_txt(self, path: Path) -> List[str]:
        logging.info(f"Loading features from {path}...")
        if not path.exists():
            raise FileNotFoundError(f"Feature list file not found: {path}")

        with open(path, "r", encoding="utf-8") as f:
            raw_features = [line.strip() for line in f if line.strip()]

        # ★追加: V5ラベリングエンジンが生成する全メタデータ・未来情報の完全除外
        exclude_exact = {
            "timestamp",
            "timeframe",
            "t1",
            "label",
            "uniqueness",
            "payoff_ratio",
            "pt_multiplier",
            "sl_multiplier",
            "direction",
            "exit_type",
            "first_ex_reason_int",
            "atr_value",
            "calculated_body_ratio",
            "fallback_vol",
            "open",
            "high",
            "low",
            "close",
            "meta_label",
            "m1_pred_proba",
        }

        features = []
        for col in raw_features:
            if col in exclude_exact or col.startswith("is_trigger_on"):
                continue
            features.append(col)

        logging.info(
            f"  -> Loaded {len(features)} valid features (filtered out metadata)."
        )
        return features

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

    # ★ JSON特徴量リスト読み込み専用メソッド
    def _load_features_from_json(self, path: Path) -> List[str]:
        logging.info(f"Loading features from {path}...")
        if not path.exists():
            raise FileNotFoundError(f"Feature file not found: {path}")

        with open(path, "r", encoding="utf-8") as f:
            features = json.load(f)

        logging.info(f"  -> Loaded {len(features)} features.")
        return features

    # ★★★ ヘルパー関数: 有効証拠金から実効レバレッジを取得 ★★★
    def _get_effective_leverage(self, equity: Decimal) -> Decimal:
        """有効証拠金に基づいてExnessのレバレッジ制限を適用"""
        base_leverage_dec = Decimal(str(self.config.base_leverage))

        # ★★★ [修正] セントモードの場合、判定用証拠金をドルに換算する ★★★
        # (これをしないと 1000ドル=100000セント が 10万ドルの大金と判定され、レバレッジ規制を食らう)
        equity_for_check = equity
        if self.config.is_cent_mode:
            equity_for_check = equity / Decimal("100")

        # Exnessの証拠金レベル (USD基準)
        # $0 - $4,999: 無制限 (ここではBase設定)
        if equity_for_check < Decimal("5000"):
            limit_leverage = base_leverage_dec
        # $5,000 - $29,999: 最大2000倍
        elif equity_for_check < Decimal("30000"):
            limit_leverage = Decimal("2000")
        # $30,000 - $99,999: 最大1000倍
        elif equity_for_check < Decimal("100000"):
            limit_leverage = Decimal("1000")
        # $100,000+: 最大500倍
        else:
            limit_leverage = Decimal("500")

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
        # ★★★ 修正: ログメッセージ更新 (Base Lev, Spread, f* 削除 / Max Pos 追加) ★★★
        logging.info(
            f"Strategy: Probabilistic Betting (Kelly Fraction: {self.config.kelly_fraction}, "
            f"Max Risk/Trade: {self.config.max_risk_per_trade * 100:.1f}%, "
            f"M2 Thresh: {self.config.m2_proba_threshold}, Max Pos: {self.config.max_positions})"
        )
        logging.info(
            f"Bankruptcy Threshold (Min Capital): {self.config.min_capital_threshold:,.2f}"
        )
        logging.info(
            f"Lot Size Params: Value = {self.config.value_per_pip}/lot/pip (SL width from S6 data), Contract Size = {CONTRACT_SIZE}"
        )

        lf, partitions_df = self._prepare_data()

        all_results_dfs = []
        all_trade_logs = []
        all_killed_logs = []  # ★追加

        logging.info(f"Processing {len(partitions_df)} partitions sequentially...")

        partitions_to_process = partitions_df
        if self.config.test_limit_partitions > 0:
            logging.warning(
                f"--- TEST MODE: Limiting to first {self.config.test_limit_partitions} partitions. ---"
            )
            partitions_to_process = partitions_df.head(
                self.config.test_limit_partitions
            )

        # --- [修正 4: run実行時にもセントモードを適用する] ---
        if self.config.is_cent_mode:
            self._current_capital = Decimal(str(self.config.initial_capital)) * Decimal(
                "100"
            )
        else:
            self._current_capital = Decimal(str(self.config.initial_capital))
        # ----------------------------------------------------

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
                    "exit_type",  # ★追加: イグジット理由
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
                # ★★★ [修正] 3つの戻り値を受け取る ★★★
                results_chunk_df, trade_log_chunk_df, killed_log_chunk_df = (
                    self._run_simulation_loop(df_chunk_predicted)
                )

                all_results_dfs.append(results_chunk_df)
                all_trade_logs.append(trade_log_chunk_df)

                # Killed Logも追加 (空でなければ)
                if not killed_log_chunk_df.is_empty():
                    all_killed_logs.append(killed_log_chunk_df)

                del (
                    df_chunk,
                    df_chunk_predicted,
                    results_chunk_df,
                    trade_log_chunk_df,
                    killed_log_chunk_df,
                )
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

            # Trade Log結合
            final_trade_log_df = (
                pl.concat([df for df in all_trade_logs if not df.is_empty()]).sort(
                    "timestamp"
                )
                if any(not df.is_empty() for df in all_trade_logs)
                else pl.DataFrame()
            )

            # ★★★ [追加] Killed Log結合 ★★★
            final_killed_log_df = (
                pl.concat(all_killed_logs).sort("timestamp")
                if all_killed_logs
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

        # 引数を追加して呼び出し
        self._analyze_and_report(
            final_results_df, final_trade_log_df, final_killed_log_df
        )
        logging.info("### Backtest Simulator: FINISHED ###")

    def _prepare_data(self) -> Tuple[pl.LazyFrame, pl.DataFrame]:
        logging.info("Preparing data (Merging S6 Predictions + Fixed V5 Indicators)...")

        # 1. S6 (予測データ / 純化済み) [UTC]
        lf_s6 = pl.scan_parquet(str(self.config.simulation_data_path / "**/*.parquet"))

        # 2. V5 (門番データ / 生データ) [UTC]
        # models/prepare_v5_indicators_from_monolithic.py でUTC化済み
        V5_DATA_PATH = Path(
            "/workspace/data/XAUUSD/stratum_2_features_fixed/v5_gatekeeper_ready"
        )

        if not V5_DATA_PATH.exists():
            raise FileNotFoundError(
                f"V5 Gatekeeper data not found at: {V5_DATA_PATH}\nPlease run 'models/prepare_v5_indicators_from_monolithic.py' first."
            )

        lf_v5 = pl.scan_parquet(str(V5_DATA_PATH / "**/*.parquet"))

        # 3. 結合 (Inner Join)
        # S6(UTC) + V5(UTC) -> OK
        lf_merged = lf_s6.join(lf_v5, on="timestamp", how="inner")

        # 4. OOFデータの結合 (Configで有効な場合のみ)
        if self.config.oof_mode and self.config.oof_predictions_path:
            try:
                logging.info(
                    f"OOF Mode: Joining predictions from {self.config.oof_predictions_path}..."
                )

                # ★★★ 復活させた修正: OOFデータも強制的にUTC扱いにする ★★★
                # 以前のスクリプトが行っていた処理をここで再現します
                lf_oof = (
                    pl.scan_parquet(str(self.config.oof_predictions_path))
                    .select(["timestamp", pl.col("prediction").alias("m2_raw_proba")])
                    .with_columns(
                        pl.col("timestamp")
                        .cast(pl.Datetime("us"))  # まずマイクロ秒に統一
                        .dt.replace_time_zone(None)  # 念のためNaiveに戻す(誤変換防止)
                        .dt.replace_time_zone("UTC")  # 明示的にUTCスタンプを押す
                    )
                )

                # UTC同士なので結合OK
                lf_merged = lf_merged.join(lf_oof, on="timestamp", how="left")
            except Exception as e:
                logging.warning(f"Failed to join OOF predictions: {e}")

        # 5. パーティション情報の取得
        logging.info("Discovering partitions from merged data...")
        partitions_df = (
            lf_merged.select(pl.col("timestamp").dt.date().alias("date"))
            .unique()
            .collect()
            .sort("date")
        )

        return lf_merged, partitions_df

    def _run_ai_predictions(self, df_chunk: pl.DataFrame) -> pl.DataFrame:
        """
        V5ロジック + AI推論を実行
        バッファ消失時でも min_periods=1 で計算を強制し、Null死（機会損失）を防ぐ
        """
        logging.debug(
            f"Running V5 Logic & AI Inference for chunk (size: {len(df_chunk)})..."
        )

        # デバッグ: バッファ状態の確認
        # buffer_status = (
        #     "None"
        #     if getattr(self, "prev_chunk_buffer", None) is None
        #     else f"Exists (len={len(self.prev_chunk_buffer)})"
        # )
        # logging.info(f"Buffer Status: {buffer_status}")

        # 入力時のカラムリストを保存
        original_cols = df_chunk.columns

        # --- [修正 1] バッファ結合の堅牢化 ---
        if self.prev_chunk_buffer is not None:
            try:
                # 垂直結合
                calc_df = pl.concat([self.prev_chunk_buffer, df_chunk], how="vertical")
                is_buffered = True
            except Exception as e:
                logging.error(f"Buffer concat failed! Resetting buffer. Error: {e}")
                calc_df = df_chunk
                is_buffered = False
                self.prev_chunk_buffer = None  # 破損したバッファを破棄
        else:
            calc_df = df_chunk
            is_buffered = False

        # --- [修正 4] V5 指標計算（強制計算モード） ---
        # バッファが無い場合でも、min_periods=1 により初手から値を生成する。
        # これにより、バッファ消失時の「先頭50行のNull化」を回避し、シグナルを殺さない。
        calc_df = calc_df.with_columns(
            pl.col("e1e_hilbert_amplitude_50")
            .rolling_mean(50, min_samples=1)  # <--- ★ここが決定打です★
            .alias("v5_hilbert_mean")
        )

        calc_df = calc_df.with_columns(
            [
                (pl.col("e1e_hilbert_amplitude_50") > pl.col("v5_hilbert_mean") * 0.00)
                .fill_null(False)
                .alias("v5_trigger"),
                (pl.col("e1a_statistical_skewness_20") >= -1.52)
                .fill_null(False)
                .alias("v5_pass_skew"),
                (pl.col("e1a_statistical_kurtosis_20") <= 100.0)
                .fill_null(False)
                .alias("v5_pass_kurt"),
                (pl.col("e1e_hilbert_phase_stability_50") >= 0.25)
                .fill_null(False)
                .alias("v5_pass_phase"),
                (pl.col("v5_body_ratio") <= 0.95)
                .fill_null(False)
                .alias("v5_pass_body"),
            ]
        )

        # 総合判定
        calc_df = calc_df.with_columns(
            (
                pl.col("v5_trigger")
                & pl.col("v5_pass_skew")
                & pl.col("v5_pass_kurt")
                & pl.col("v5_pass_phase")
                & pl.col("v5_pass_body")
            ).alias("v5_gate_passed")
        )

        # --- [修正 2] バッファ分を除外して当日データのみを抽出 ---
        if is_buffered:
            buffer_len = len(self.prev_chunk_buffer)
            # バッファ分を除外
            df_chunk = calc_df.slice(buffer_len, len(calc_df) - buffer_len)
        else:
            df_chunk = calc_df

        # --- [修正 3] メモリ安全なバッファ保存 (.clone() の追加) ---
        try:
            # .clone() により親DataFrameへの参照を切断し、独立したメモリ領域を確保
            next_buffer = df_chunk.select(original_cols).tail(50).clone()

            # バッファが空でないか確認
            if len(next_buffer) > 0:
                self.prev_chunk_buffer = next_buffer
            else:
                logging.warning(
                    "Insufficient data for buffer update, keeping previous buffer"
                )

        except Exception as e:
            logging.error(f"Failed to update buffer: {e}")
            # エラー時も処理は続行

        # --- ML 推論 (M1 -> M2) ---
        try:
            X_m1 = df_chunk.select(self.features_m1).fill_null(0).to_numpy()
            raw_m1_proba = self.m1_model.predict(X_m1)
            calibrated_m1_proba = self.m1_calibrator.predict(raw_m1_proba)
            calibrated_m1_proba = np.clip(calibrated_m1_proba, 0.0, 1.0)

            df_chunk = df_chunk.with_columns(
                pl.Series("m1_pred_proba", calibrated_m1_proba)
            )
        except Exception as e:
            logging.error(f"Error in M1 Inference: {e}", exc_info=True)
            raise

        try:
            m2_feats_only = [f for f in self.features_m2 if f != "m1_pred_proba"]
            X_m2_base = df_chunk.select(m2_feats_only).fill_null(0).to_numpy()
            X_m2 = np.hstack([calibrated_m1_proba.reshape(-1, 1), X_m2_base])
            raw_m2_proba = self.m2_model.predict(X_m2)
            calibrated_m2_proba = self.m2_calibrator.predict(raw_m2_proba)
            calibrated_m2_proba = np.clip(calibrated_m2_proba, 0.0, 1.0)

            df_chunk = df_chunk.with_columns(
                pl.Series("m2_calibrated_proba", calibrated_m2_proba)
            )
        except Exception as e:
            logging.error(f"Error in M2 Inference: {e}", exc_info=True)
            raise

        return df_chunk

    def _run_simulation_loop(
        self, df_chunk: pl.DataFrame
    ) -> Tuple[pl.DataFrame, pl.DataFrame]:
        trade_log_chunk = []
        equity_values_chunk = []
        current_capital = self._current_capital

        # 定数の準備
        DECIMAL_ZERO = Decimal("0.0")
        DECIMAL_ONE = Decimal("1.0")
        DECIMAL_MAX_RISK = Decimal(str(self.config.max_risk_per_trade))
        DECIMAL_KELLY_FRACTION = Decimal(str(self.config.kelly_fraction))
        DECIMAL_F_STAR_THRESHOLD = Decimal(str(self.config.f_star_threshold))
        DECIMAL_MIN_CAPITAL = Decimal(str(self.config.min_capital_threshold))
        DECIMAL_VALUE_PER_PIP = Decimal(str(self.config.value_per_pip))
        DECIMAL_MIN_LOT_SIZE = Decimal(str(self.config.min_lot_size))

        DECIMAL_CONTRACT_SIZE = CONTRACT_SIZE
        if self.config.is_cent_mode:
            DECIMAL_CONTRACT_SIZE *= Decimal("100")

        # --- ATR(21) の取得 ---
        # ★★★ 修正: 生のOHLCから再計算せず、データセットにある 'atr_value' を使用する ★★★
        # S6データセット作成時に計算済み、かつ欠損時は _prepare_data で補完済みであるため安全
        if "atr_value" in df_chunk.columns:
            atr_values_chunk = df_chunk["atr_value"].to_numpy()
        else:
            # 万が一のフォールバック (Closeの0.2%)
            logging.warning(
                "atr_value column missing in simulation loop. Using fallback."
            )
            atr_values_chunk = (df_chunk["close"] * 0.002).to_numpy()

        # データ展開
        timestamps_chunk = df_chunk["timestamp"].to_list()
        close_prices_chunk = df_chunk["close"].to_numpy()
        labels_chunk = df_chunk["label"].to_numpy()
        t1_chunk = df_chunk["t1"].to_list()
        timeframes_chunk = df_chunk["timeframe"].to_list()
        exit_types_chunk = df_chunk["exit_type"].to_list()  # ★追加

        # 予測スコア
        m1_proba_chunk = df_chunk["m1_pred_proba"].to_numpy()
        m2_proba_chunk = df_chunk["m2_calibrated_proba"].to_numpy()

        # --- ★★★ [修正] CSV出力用にV5の「生の数値」を取得 ★★★ ---
        # (Booleanではなく、実際の値を分析するために抽出)
        raw_amp_chunk = df_chunk["e1e_hilbert_amplitude_50"].to_numpy()
        raw_mean_chunk = df_chunk["v5_hilbert_mean"].to_numpy()  # 追加したカラム
        raw_skew_chunk = df_chunk["e1a_statistical_skewness_20"].to_numpy()
        raw_kurt_chunk = df_chunk["e1a_statistical_kurtosis_20"].to_numpy()
        raw_phase_chunk = df_chunk["e1e_hilbert_phase_stability_50"].to_numpy()
        raw_body_chunk = df_chunk["v5_body_ratio"].to_numpy()

        # Booleanフラグ（内訳分析用）
        v5_trigger_chunk = df_chunk["v5_trigger"].to_numpy()
        v5_skew_chunk = df_chunk["v5_pass_skew"].to_numpy()
        v5_kurt_chunk = df_chunk["v5_pass_kurt"].to_numpy()
        v5_phase_chunk = df_chunk["v5_pass_phase"].to_numpy()
        v5_body_chunk = df_chunk["v5_pass_body"].to_numpy()

        # V5 Gatekeeper Result
        v5_passed_chunk = df_chunk["v5_gate_passed"].to_numpy()

        # Payoff Ratio (Fallback: 2.0)
        payoff_ratios_chunk = [
            Decimal(str(b))
            if b is not None and np.isfinite(b) and b > 0
            else Decimal("2.0")
            for b in df_chunk["payoff_ratio"].to_list()
        ]

        active_exit_times = []
        MAX_POSITIONS = self.config.max_positions
        ALLOWED_TIMEFRAMES = ["M1", "M3", "M5", "M8", "M15", "H1"]
        last_trade_timestamp = None

        # ★★★ [追加] 殺されたトレードを記録するリスト ★★★
        killed_log_chunk = []

        for i in range(len(df_chunk)):
            # --- スキップ条件 ---
            current_tf = timeframes_chunk[i]
            if current_tf not in ALLOWED_TIMEFRAMES:
                equity_values_chunk.append(current_capital)
                continue

            current_timestamp = timestamps_chunk[i]
            try:
                # タイムスタンプ変換
                if isinstance(current_timestamp, dt.datetime):
                    if current_timestamp.tzinfo is None:
                        current_timestamp = current_timestamp.replace(
                            tzinfo=dt.timezone.utc
                        )
                    current_timestamp_int = int(
                        current_timestamp.timestamp() * 1_000_000
                    )
                else:
                    # まれにDate型などが来る場合のガード
                    equity_values_chunk.append(current_capital)
                    continue
            except Exception:
                equity_values_chunk.append(current_capital)
                continue

            current_price_float = close_prices_chunk[i]
            if (
                current_price_float is None
                or not np.isfinite(current_price_float)
                or current_price_float <= 0
            ):
                equity_values_chunk.append(current_capital)
                continue
            current_price_decimal = Decimal(str(current_price_float))

            # 破産チェック
            if current_capital < DECIMAL_MIN_CAPITAL:
                equity_values_chunk.append(DECIMAL_ZERO)
                continue

            # ポジション管理（終了したものを削除）
            active_exit_times = [
                t for t in active_exit_times if t > current_timestamp_int
            ]
            if len(active_exit_times) >= MAX_POSITIONS:
                equity_values_chunk.append(current_capital)
                continue

            # --- エントリー判定 (V5 + ML) ---

            # 1. V5 Gatekeeper Check
            is_v5_passed = v5_passed_chunk[i]

            # 2. ML Score Check
            # Buy Entry: M1 > 0.5 AND M2 > Threshold
            m1_score = m1_proba_chunk[i]
            m2_score = m2_proba_chunk[i]
            is_ml_passed = (m1_score > 0.5) and (
                m2_score > self.config.m2_proba_threshold
            )

            # ★★★ [修正] V5による「機会損失」を記録するためのロジック分岐 ★★★
            should_trade = False

            if is_ml_passed:
                if is_v5_passed:
                    should_trade = True
                else:
                    # AIはGoと言ったが、V5が止めた -> Killed Logへ記録
                    # (ポジション数制限などは考慮せず、あくまでV5単体の影響を見る)
                    killed_log_chunk.append(
                        {
                            "timestamp": current_timestamp,
                            "m1_proba": float(m1_score),
                            "m2_proba": float(m2_score),
                            "v5_trigger_val": float(raw_amp_chunk[i]),
                            "v5_trigger_mean": float(raw_mean_chunk[i]),
                            "v5_skew_val": float(raw_skew_chunk[i]),
                            "v5_kurt_val": float(raw_kurt_chunk[i]),
                            "v5_phase_val": float(raw_phase_chunk[i]),
                            "v5_body_val": float(raw_body_chunk[i]),
                            # どのルールに抵触したか
                            "fail_trigger": not v5_trigger_chunk[i],
                            "fail_skew": not v5_skew_chunk[i],
                            "fail_kurt": not v5_kurt_chunk[i],
                            "fail_phase": not v5_phase_chunk[i],
                            "fail_body": not v5_body_chunk[i],
                        }
                    )

            # 同時刻エントリー制限
            if should_trade and (last_trade_timestamp == current_timestamp):
                should_trade = False

            # --- ロット計算と約定 ---
            final_lot_size_decimal = DECIMAL_ZERO
            pnl = DECIMAL_ZERO
            margin_required_decimal = DECIMAL_ZERO
            spread_cost_decimal = DECIMAL_ZERO

            # ログ用変数
            kelly_f_star = DECIMAL_ZERO
            f_star = DECIMAL_ZERO
            base_bet_fraction = DECIMAL_ZERO
            effective_bet_fraction = DECIMAL_ZERO
            capped_bet_fraction = DECIMAL_ZERO
            atr_val_log = 0.0

            if should_trade:
                # ケリー基準などの資金管理ロジック
                # 今回は簡略化のため、M2スコアを確率(p)として使用
                p_decimal = Decimal(str(m2_score))
                b_decimal = payoff_ratios_chunk[i]
                q_decimal = DECIMAL_ONE - p_decimal

                if b_decimal > DECIMAL_ZERO:
                    kelly_f_star = (b_decimal * p_decimal - q_decimal) / b_decimal

                f_star = (
                    kelly_f_star.copy_abs().max(DECIMAL_ZERO)
                    if kelly_f_star.is_finite()
                    else DECIMAL_ZERO
                )

                if f_star > DECIMAL_F_STAR_THRESHOLD:
                    base_bet_fraction = f_star * DECIMAL_KELLY_FRACTION
                    effective_bet_fraction = (
                        base_bet_fraction.copy_abs()
                        .min(DECIMAL_MAX_RISK)
                        .min(DECIMAL_ONE)
                    )
                    capped_bet_fraction = effective_bet_fraction

                    if effective_bet_fraction > DECIMAL_ZERO:
                        risk_amount_decimal = current_capital * effective_bet_fraction

                        # ATRベースのSL計算 (R4ルール: SL = 5.0 * ATR)
                        atr_val = atr_values_chunk[i]
                        atr_val_log = float(atr_val)

                        if atr_val > 0:
                            # SL幅（価格）
                            sl_width_price = Decimal(str(atr_val)) * Decimal("5.0")
                            # 1ロットあたりのSL損失額（通貨）
                            stop_loss_currency_per_lot = (
                                sl_width_price * DECIMAL_CONTRACT_SIZE
                            )

                            if stop_loss_currency_per_lot > DECIMAL_ZERO:
                                desired_lot = (
                                    risk_amount_decimal / stop_loss_currency_per_lot
                                )
                            else:
                                desired_lot = DECIMAL_ZERO
                        else:
                            desired_lot = DECIMAL_ZERO

                        # レバレッジ・最大ロット制限
                        eff_lev = self._get_effective_leverage(current_capital)
                        if eff_lev > DECIMAL_ZERO:
                            max_lot_margin = (current_capital * eff_lev) / (
                                current_price_decimal * DECIMAL_CONTRACT_SIZE
                            )
                        else:
                            max_lot_margin = DECIMAL_ZERO

                        max_lot_broker = self._get_max_lot_allowed(current_timestamp)

                        final_lot_size_decimal = (
                            desired_lot.min(max_lot_margin)
                            .min(max_lot_broker)
                            .max(DECIMAL_ZERO)
                        )

                        # 最小ロットチェック
                        if (
                            final_lot_size_decimal > DECIMAL_ZERO
                            and final_lot_size_decimal < DECIMAL_MIN_LOT_SIZE
                        ):
                            final_lot_size_decimal = DECIMAL_MIN_LOT_SIZE

            # --- 取引実行 ---
            if final_lot_size_decimal > DECIMAL_ZERO:
                last_trade_timestamp = current_timestamp

                # 必要証拠金
                eff_lev = self._get_effective_leverage(current_capital)
                margin_required_decimal = (
                    current_price_decimal
                    * final_lot_size_decimal
                    * DECIMAL_CONTRACT_SIZE
                ) / eff_lev

                # スプレッドコスト
                spread_pips_decimal = Decimal(str(self.config.spread_pips))
                spread_cost_decimal = (
                    final_lot_size_decimal * spread_pips_decimal * DECIMAL_VALUE_PER_PIP
                )

                if spread_cost_decimal >= current_capital:
                    # コスト倒れならキャンセル
                    pnl = DECIMAL_ZERO
                    final_lot_size_decimal = DECIMAL_ZERO
                else:
                    capital_before_pnl = current_capital - spread_cost_decimal

                    # PnL計算
                    # SL幅 = 5.0 * ATR
                    atr_val = atr_values_chunk[i]
                    sl_dist_price = Decimal(str(atr_val)) * Decimal("5.0")
                    risk_amount_final = (
                        final_lot_size_decimal * sl_dist_price * DECIMAL_CONTRACT_SIZE
                    )

                    label = labels_chunk[i]  # 1: Win, -1: Loss, 0: Timeout

                    if label == 1:
                        # Win: 利益 = リスク額 * PayoffRatio
                        pnl = risk_amount_final * b_decimal
                    elif label == -1:
                        # Loss: 損失 = -リスク額
                        pnl = risk_amount_final.copy_negate()
                    else:
                        # Timeout (0): PnL = 0 (スプレッド分のみ減少)
                        pnl = DECIMAL_ZERO

                    current_capital = capital_before_pnl + pnl

                    if not current_capital.is_finite():
                        current_capital = DECIMAL_ZERO

                    # 終了時刻登録
                    if t1_chunk[i] is not None:
                        try:
                            t1_dt = t1_chunk[i]
                            if t1_dt.tzinfo is None:
                                t1_dt = t1_dt.replace(tzinfo=dt.timezone.utc)
                            new_exit_time = int(t1_dt.timestamp() * 1_000_000)
                            active_exit_times.append(new_exit_time)
                        except:
                            pass

                # ログ記録
                trade_log_chunk.append(
                    {
                        "timestamp": current_timestamp,
                        "timeframe": current_tf,
                        "pnl": pnl,
                        "capital_after_trade": current_capital,
                        "m2_calibrated_proba": float(m2_score),
                        "payoff_ratio": b_decimal,
                        "kelly_f_star": kelly_f_star,
                        "f_star": f_star,
                        "base_bet_fraction": base_bet_fraction,
                        "capped_bet_fraction": capped_bet_fraction,
                        "effective_bet_fraction": effective_bet_fraction,
                        "label": labels_chunk[i],
                        "lot_size": float(final_lot_size_decimal),
                        "atr_value": atr_val_log,
                        "sl_multiplier": 5.0,  # R4固定
                        "pt_multiplier": 1.0,  # R4固定
                        "direction": 1,  # Buy Only
                        "effective_leverage": float(eff_lev)
                        if "eff_lev" in locals()
                        else 0.0,
                        "margin_required": margin_required_decimal,
                        "spread_cost": spread_cost_decimal,
                        "close_price": current_price_decimal,
                        # --- ★★★ [修正] Booleanではなく「生の数値」を記録 ★★★ ---
                        "v5_amp": float(raw_amp_chunk[i]),
                        "v5_mean": float(raw_mean_chunk[i]),
                        "v5_skew": float(raw_skew_chunk[i]),
                        "v5_kurt": float(raw_kurt_chunk[i]),
                        "v5_phase": float(raw_phase_chunk[i]),
                        "v5_body": float(raw_body_chunk[i]),
                        "exit_type": exit_types_chunk[i],  # ★追加
                    }
                )

            equity_values_chunk.append(current_capital)

        self._current_capital = current_capital

        # 結果のDataFrame化
        results_chunk_df = pl.DataFrame(
            {
                "timestamp": timestamps_chunk,
                "equity": pl.Series("equity", equity_values_chunk, dtype=pl.Object),
            }
        )

        # trade_logのスキーマ定義とDF化
        trade_log_schema = {
            "timestamp": pl.Datetime,
            "timeframe": pl.Utf8,
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
            # ★★★ [修正] スキーマを生の数値に変更 ★★★
            "v5_amp": pl.Float64,
            "v5_mean": pl.Float64,
            "v5_skew": pl.Float64,
            "v5_kurt": pl.Float64,
            "v5_phase": pl.Float64,
            "v5_body": pl.Float64,
            "exit_type": pl.Utf8,  # ★追加
        }

        if trade_log_chunk:
            trade_log_data = {
                key: [d.get(key) for d in trade_log_chunk]
                for key in trade_log_schema.keys()
            }
            trade_log_chunk_df = pl.DataFrame(trade_log_data, schema=trade_log_schema)
        else:
            trade_log_chunk_df = pl.DataFrame(schema=trade_log_schema)

        # ★★★ [追加] Killed LogのDataFrame化 ★★★
        killed_log_chunk_df = pl.DataFrame(killed_log_chunk)

        # 戻り値を3つにする
        return results_chunk_df, trade_log_chunk_df, killed_log_chunk_df

    # ★★★ [修正] 引数に killed_log: pl.DataFrame を追加してください ★★★
    def _analyze_and_report(
        self,
        results_df: pl.DataFrame,
        trade_log: pl.DataFrame,
        killed_log: pl.DataFrame,
    ):
        # --- [修正 3] 表示用にデータをドルに戻す (Deep Copyして変換) ---
        if self.config.is_cent_mode:
            logging.info("Converting simulation results back to USD for reporting...")
            DECIMAL_100 = Decimal("100")

            # 1. 資産推移 (Equity) を 1/100
            results_df = results_df.with_columns(
                pl.col("equity").map_elements(
                    lambda x: x / DECIMAL_100, return_dtype=pl.Object
                )
            )

            # 2. トレード履歴 (PnL, Capital, Cost) を 1/100
            if not trade_log.is_empty():
                cols_to_convert = [
                    "pnl",
                    "capital_after_trade",
                    "spread_cost",
                    "margin_required",
                ]
                valid_cols = [c for c in cols_to_convert if c in trade_log.columns]

                if valid_cols:
                    trade_log = trade_log.with_columns(
                        [
                            pl.col(c).map_elements(
                                lambda x: x / DECIMAL_100 if x is not None else None,
                                return_dtype=pl.Object,
                            )
                            for c in valid_cols
                        ]
                    )

            # # ※注意: initial_capital もレポート表示用に書き換える
            # self.config.initial_capital /= 100.0  # 元に戻す
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

        # ★★★ 修正: レポートヘッダー更新 (Base Lev, Spread, f* 削除 / Max Pos 追加) ★★★
        report_data = {
            "strategy": f"Probabilistic Betting (Kelly Fraction: {self.config.kelly_fraction}, Max Risk/Trade: {self.config.max_risk_per_trade * 100:.1f}%, M2 Thresh: {self.config.m2_proba_threshold}, Max Pos: {self.config.max_positions})",
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
        # --- [修正 3: None安全対策] ---
        pf = report_data.get("profit_factor")
        if pf is not None:
            print(f" Profit Factor:        {pf:.2f}")
        else:
            print(f" Profit Factor:        N/A")

        avg_bet = report_data.get("average_effective_bet_fraction_pct")
        if avg_bet is not None:
            print(f" Avg. Bet Fraction:    {avg_bet:.2f}%")
        else:
            print(f" Avg. Bet Fraction:    N/A")
        # -----------------------------
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

        # ==============================================================================
        # [修正 1] CSVログ出力をグラフ描画より「先」に実行 (メモリ不足で死んでもログは残す)
        # ==============================================================================
        if not trade_log.is_empty():
            trade_log_output_path = FINAL_REPORT_PATH.parent / "detailed_trade_log.csv"
            logging.info(
                f"Preparing detailed trade log for CSV output ({len(trade_log)} trades)..."
            )
            try:
                # --- フォーマット処理 ---
                temp_log_formatted = trade_log.clone()
                format_expressions = []

                # Timestamp -> String
                if "timestamp" in temp_log_formatted.columns:
                    format_expressions.append(
                        pl.col("timestamp")
                        .dt.strftime("%Y-%m-%d %H:%M:%S")
                        .alias("timestamp")
                    )

                # Decimal/Object -> Float & Round
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
                    if col_name in temp_log_formatted.columns:
                        format_expressions.append(
                            pl.col(col_name)
                            .map_elements(
                                lambda d: (
                                    float(d)
                                    if d is not None and d.is_finite()
                                    else None
                                ),
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
                    if col_name in temp_log_formatted.columns:
                        format_expressions.append(
                            pl.col(col_name).round(digits).alias(col_name)
                        )

                if format_expressions:
                    temp_log_formatted = temp_log_formatted.with_columns(
                        format_expressions
                    )

                # --- 列選択と並び替え ---
                desired_columns_final = [
                    "timestamp",
                    "timeframe",
                    "capital_after_trade",
                    "pnl",
                    "kelly_f_star",
                    "m2_calibrated_proba",
                    "lot_size",
                    "spread_cost",
                    "margin_required",
                    "label",
                    "exit_type",  # ★追加
                    # --- ★★★ [修正] 生の数値を出力 ---
                    "v5_amp",
                    "v5_mean",
                    "v5_skew",
                    "v5_kurt",
                    "v5_phase",
                    "v5_body",
                    # "payoff_ratio",
                    "effective_leverage",
                    # "direction",
                    "close_price",
                    "atr_value",
                    # "sl_multiplier",
                    # "pt_multiplier",
                    # "effective_bet_fraction",
                ]
                available_columns_final = [
                    c for c in desired_columns_final if c in temp_log_formatted.columns
                ]
                trade_log_final_csv = temp_log_formatted.select(available_columns_final)

                # --- CSV書き出し ---
                trade_log_final_csv.write_csv(trade_log_output_path, null_value="NaN")
                logging.info(
                    f"Formatted detailed trade log saved to {trade_log_output_path}"
                )

            except Exception as e:
                logging.error(
                    f"Failed to save formatted detailed trade log: {e}", exc_info=True
                )
        else:
            logging.info("No trades were executed, skipping detailed trade log output.")

        # ==============================================================================
        # ★★★ [追加] V5ゲートキーパー専用レポート & Killed Log CSV出力 ★★★
        # ==============================================================================
        v5_report_path = FINAL_REPORT_PATH.parent / "v5_gatekeeper_report.txt"
        killed_csv_path = FINAL_REPORT_PATH.parent / "killed_trades_log.csv"

        logging.info("Generating V5 Gatekeeper analysis report...")

        total_executed = len(trade_log)
        total_killed = len(killed_log)
        total_signals = total_executed + total_killed

        # Killed Log CSV出力
        if not killed_log.is_empty():
            killed_log.write_csv(killed_csv_path)
            logging.info(f"Killed trades log saved to {killed_csv_path}")

        # 集計
        kill_counts = {
            "Trigger (Low Volatility)": 0,
            "Skewness (Crash Risk)": 0,
            "Kurtosis (Fat Tail)": 0,
            "Phase (Unstable)": 0,
            "Body Ratio (Falling Knife)": 0,
        }

        if not killed_log.is_empty():
            kill_counts["Trigger (Low Volatility)"] = killed_log["fail_trigger"].sum()
            kill_counts["Skewness (Crash Risk)"] = killed_log["fail_skew"].sum()
            kill_counts["Kurtosis (Fat Tail)"] = killed_log["fail_kurt"].sum()
            kill_counts["Phase (Unstable)"] = killed_log["fail_phase"].sum()
            kill_counts["Body Ratio (Falling Knife)"] = killed_log["fail_body"].sum()

        # --- ★★★ [追加] トリガー倍率(Amp/Mean)の分布解析 ★★★ ---
        # import numpy as np  <-- 削除

        # 1. 通過したトレードの倍率
        ratios_passed = []
        if not trade_log.is_empty() and "v5_amp" in trade_log.columns:
            # 0除算回避のため numpy で計算
            amp = trade_log["v5_amp"].to_numpy()
            mean = trade_log["v5_mean"].to_numpy()
            with np.errstate(divide="ignore", invalid="ignore"):
                ratios = amp / mean
                ratios_passed = ratios[np.isfinite(ratios)].tolist()

        # 2. 弾かれたトレードの倍率
        ratios_killed = []
        if not killed_log.is_empty() and "v5_trigger_val" in killed_log.columns:
            amp = killed_log["v5_trigger_val"].to_numpy()
            mean = killed_log["v5_trigger_mean"].to_numpy()
            with np.errstate(divide="ignore", invalid="ignore"):
                ratios = amp / mean
                ratios_killed = ratios[np.isfinite(ratios)].tolist()

        # 3. 統合と統計計算
        all_ratios = np.array(ratios_passed + ratios_killed)

        dist_stats = {}
        if len(all_ratios) > 0:
            dist_stats["min"] = np.min(all_ratios)
            dist_stats["max"] = np.max(all_ratios)
            dist_stats["mean"] = np.mean(all_ratios)
            dist_stats["median"] = np.median(all_ratios)
            dist_stats["p10"] = np.percentile(all_ratios, 10)
            dist_stats["p25"] = np.percentile(all_ratios, 25)
            dist_stats["p75"] = np.percentile(all_ratios, 75)
            dist_stats["p90"] = np.percentile(all_ratios, 90)

        # --- ★★★ [追加] 実行されたトレードのイグジット理由集計 ★★★ ---
        exit_type_summary = {}
        if not trade_log.is_empty() and "exit_type" in trade_log.columns:
            try:
                # Polarsでグループ化してカウント
                exit_counts = trade_log.group_by("exit_type").len().to_dicts()
                exit_type_summary = {
                    row["exit_type"]: row["len"] for row in exit_counts
                }
            except Exception as e:
                logging.warning(f"Could not group exit_type: {e}")

        # 大文字小文字のブレを吸収してカウント
        em_skew_count = exit_type_summary.get("EM_SKEW", 0) + exit_type_summary.get(
            "em_skew", 0
        )
        em_kurt_count = exit_type_summary.get("EM_KURT", 0) + exit_type_summary.get(
            "em_kurt", 0
        )
        em_phase_count = exit_type_summary.get("EM_PHASE", 0) + exit_type_summary.get(
            "em_phase", 0
        )
        emergency_total = (
            exit_type_summary.get("EMERGENCY", 0)
            + exit_type_summary.get("emergency", 0)
            + em_skew_count
            + em_kurt_count
            + em_phase_count
        )
        pt_count = exit_type_summary.get("PT", 0) + exit_type_summary.get("pt", 0)
        sl_count = exit_type_summary.get("SL", 0) + exit_type_summary.get("sl", 0)
        to_count = exit_type_summary.get("TIMEOUT", 0) + exit_type_summary.get(
            "timeout", 0
        )
        # --- [ここまで追加] ---

        try:
            with open(v5_report_path, "w", encoding="utf-8") as f:
                f.write("=" * 60 + "\n")
                f.write("    V5 Gatekeeper Analysis Report\n")
                f.write("=" * 60 + "\n\n")

                f.write(f"Total AI Signals (M1>0.5 & M2>Thresh):  {total_signals}\n")
                f.write(
                    f"  - Passed & Executed:                  {total_executed} ({total_executed / total_signals * 100:.1f}%)\n"
                )
                f.write(
                    f"  - Rejected (Killed):                  {total_killed} ({total_killed / total_signals * 100:.1f}%)\n\n"
                )

                # ★★★ [追加] 分布統計の出力 ★★★
                if len(all_ratios) > 0:
                    f.write(
                        "-" * 30
                        + " Trigger Ratio (Amp/Mean) Distribution "
                        + "-" * 30
                        + "\n"
                    )
                    f.write(
                        f"Based on {len(all_ratios)} total signals (Passed + Killed)\n\n"
                    )
                    f.write(f"  Min:    {dist_stats['min']:.4f}\n")
                    f.write(f"  Max:    {dist_stats['max']:.4f}\n")
                    f.write(f"  Mean:   {dist_stats['mean']:.4f}\n")
                    f.write(f"  Median: {dist_stats['median']:.4f}\n")
                    f.write(
                        f"  10%ile: {dist_stats['p10']:.4f}  (If threshold was here, you'd catch 90%)\n"
                    )
                    f.write(f"  25%ile: {dist_stats['p25']:.4f}\n")
                    f.write(f"  75%ile: {dist_stats['p75']:.4f}\n")
                    f.write(f"  90%ile: {dist_stats['p90']:.4f}\n\n")

                f.write("-" * 30 + " Rejection Breakdown " + "-" * 30 + "\n")
                f.write("(Note: A trade can be rejected by multiple rules)\n\n")

                for reason, count in kill_counts.items():
                    pct = (count / total_killed * 100) if total_killed > 0 else 0.0
                    f.write(f"{reason:<30}: {count:>5} ({pct:>5.1f}% of killed)\n")

                # --- ★★★ [追加] 実行トレードのイグジット内訳表示 ★★★ ---
                f.write(
                    "\n"
                    + "-" * 30
                    + " Executed Trades Exit Breakdown "
                    + "-" * 30
                    + "\n"
                )
                f.write(f"Total Executed Trades: {total_executed}\n\n")
                f.write(f"  Profit Take (PT):       {pt_count:>5}\n")
                f.write(f"  Normal Stop Loss (SL):  {sl_count:>5}\n")
                f.write(f"  Timeout (0):            {to_count:>5}\n")
                f.write(f"  Emergency Exits:        {emergency_total:>5}\n")
                if emergency_total > 0:
                    f.write(f"    ├ Skewness:           {em_skew_count:>5}\n")
                    f.write(f"    ├ Kurtosis:           {em_kurt_count:>5}\n")
                    f.write(f"    └ Phase:              {em_phase_count:>5}\n")
                # --- [ここまで追加] ---

                f.write("\n" + "=" * 60 + "\n")

            logging.info(f"V5 Gatekeeper report saved to {v5_report_path}")

        except Exception as e:
            logging.error(f"Failed to generate V5 report: {e}")

        # ==============================================================================
        # [修正 2 & 3] グラフ描画 (メモリ対策の間引き + 初期横ばい期間の除外)
        # ==============================================================================
        logging.info("Generating equity curve and drawdown chart...")
        if results_df.is_empty() or drawdown.is_empty():
            logging.warning("No data available to generate equity curve chart.")
        else:
            try:
                # データの準備 (Polars -> List)
                # equity_series_float は既に計算済み (初期資金 + results_df['equity'])
                full_equity_list = equity_series_float.to_list()

                # timestampは初期資金分の1行が足りないので、先頭にダミー(または最初の時刻)を足して合わせる
                # ただし描画には使うため、seriesの長さを合わせる
                first_ts = results_df["timestamp"][0]
                full_timestamps_list = [first_ts] + results_df["timestamp"].to_list()

                full_drawdown_list = [
                    0.0
                ] + drawdown.to_list()  # 初期値0を追加して長さを合わせる

                # --- [修正 3] 初期横ばい期間の除外ロジック ---
                start_plot_index = 0
                initial_val = full_equity_list[0]

                # 資産が初期値から変動した最初のポイントを探す
                for i, val in enumerate(full_equity_list):
                    if val != initial_val:
                        # 変動が始まった地点の少し前(例えば20期間前)から描画を開始して、
                        # 「動き出し」が見えるようにする
                        start_plot_index = max(0, i - 20)
                        break

                if start_plot_index > 0:
                    logging.info(
                        f"Skipping initial {start_plot_index} periods (flat equity) for plotting."
                    )

                # スライス適用
                sliced_equity = full_equity_list[start_plot_index:]
                sliced_timestamps = full_timestamps_list[start_plot_index:]
                sliced_drawdown = full_drawdown_list[start_plot_index:]

                # --- [修正 2] メモリ対策の間引き (Downsampling) ---
                # データ点数が多すぎる(例: 10万点以上)場合、描画負荷を下げるために間引く
                MAX_PLOT_POINTS = 5000  # グラフ上の点は5000個もあれば十分綺麗に見える
                data_len = len(sliced_equity)
                step = max(1, data_len // MAX_PLOT_POINTS)

                if step > 1:
                    logging.info(
                        f"Downsampling chart data by factor of {step} (Original: {data_len} points)."
                    )
                    plot_equity = sliced_equity[::step]
                    plot_timestamps = sliced_timestamps[::step]
                    plot_drawdown = sliced_drawdown[::step]
                else:
                    plot_equity = sliced_equity
                    plot_timestamps = sliced_timestamps
                    plot_drawdown = sliced_drawdown

                # --- 描画実行 ---
                sns.set_style("darkgrid")
                fig, (ax1, ax2) = plt.subplots(
                    2,
                    1,
                    figsize=(15, 10),
                    sharex=True,
                    gridspec_kw={"height_ratios": [3, 1]},
                )

                # NaNを0に置換 (描画エラー回避)
                plot_drawdown = [d if np.isfinite(d) else 0.0 for d in plot_drawdown]

                ax1.plot(
                    plot_timestamps,
                    plot_equity,
                    label="Equity Curve",
                    color="dodgerblue",
                    linewidth=1.5,
                )
                ax1.set_title(
                    f"Equity Curve (Kelly: {self.config.kelly_fraction}, Lev: {self.config.base_leverage}, Max Risk: {self.config.max_risk_per_trade * 100:.1f}%)",
                    fontsize=16,
                )
                ax1.set_ylabel("Equity")
                ax1.grid(True)

                # Y軸スケール設定 (ログスケール判定)
                try:
                    finite_equity = [e for e in plot_equity if np.isfinite(e) and e > 0]
                    if not finite_equity:
                        ax1.ticklabel_format(style="plain", axis="y")
                    elif any(np.isinf(plot_equity)) or (
                        max(finite_equity) / min(finite_equity) > 1000
                    ):
                        ax1.set_yscale("log")
                    else:
                        ax1.ticklabel_format(style="plain", axis="y")
                except Exception:
                    ax1.ticklabel_format(style="plain", axis="y")

                ax2.fill_between(
                    plot_timestamps, plot_drawdown, 0, color="red", alpha=0.3
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

                # None安全対策
                pf = report_data.get("profit_factor")
                f.write(f"Profit Factor:\t\t{pf:.2f}\n") if pf is not None else f.write(
                    "Profit Factor:\t\tN/A\n"
                )

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

                avg_bet_pct = report_data.get("average_effective_bet_fraction_pct")
                if avg_bet_pct is not None:
                    f.write(f"Avg Bet Size (% Cap):\t{avg_bet_pct:.2f} %\n\n")
                else:
                    f.write("Avg Bet Size (% Cap):\tN/A\n\n")

                f.write("=" * 60 + "\n")
            logging.info(f"Text performance report saved successfully.")
        except Exception as e:
            logging.error(f"Failed to save text performance report: {e}", exc_info=True)


if __name__ == "__main__":
    default_config = BacktestConfig()

    parser = argparse.ArgumentParser(
        description="Project Forge Backtest Simulator (V5 Gatekeeper + M1/M2 Pipeline)"
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
    parser.add_argument(
        "--cent-mode",
        action="store_true",
        help="Run in Exness Cent Account mode (Input/Output in USD, Internal in USC).",
    )
    args = parser.parse_args()

    # --- Config生成 ---
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
        is_cent_mode=args.cent_mode,
    )

    # バリデーション
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
