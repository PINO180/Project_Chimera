# /workspace/models/backtest_simulator_cimera_purified
# [V5改修版: Project Cimera 双方向ラベリング仕様 (Part 1)]

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

# --- blueprintから必要なパスをインポート (V5仕様に合わせてパスを調整) ---
from blueprint import (
    S6_WEIGHTED_DATASET,
    S3_FEATURES_FOR_TRAINING,
    S7_M1_MODEL_PKL,
    S7_M2_MODEL_PKL,
    S7_M1_CALIBRATED,
    S7_M2_CALIBRATED,
    S7_MODELS,
    S7_M2_LONG_OOF_PREDICTIONS,
    S7_M2_SHORT_OOF_PREDICTIONS,
)

# --- 出力ファイルパス ---
FINAL_REPORT_PATH = S7_MODELS / "final_backtest_report_v5.json"
EQUITY_CURVE_PATH = S7_MODELS / "equity_curve_v5.png"


# --- ロギング設定 ---
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

# XAUUSDの契約サイズ
CONTRACT_SIZE = Decimal("100")  # 1 lot = 100 oz

# タイムゾーン変換用
JST = zoneinfo.ZoneInfo("Asia/Tokyo")


# ================================================================
# フェーズ 0: 作戦司令室 (パラメータ設定 - V5仕様)
# ================================================================
@dataclass
class BacktestConfig:
    """シミュレーションの全パラメータを一元管理 (V5 Two-Brain Architecture)"""

    initial_capital: float = 1000.0
    simulation_data_path: Path = S6_WEIGHTED_DATASET

    # V5: Long/Short独立のOOF予測パス
    oof_long_path: Path = S7_M2_LONG_OOF_PREDICTIONS
    oof_short_path: Path = S7_M2_SHORT_OOF_PREDICTIONS

    feature_list_path: Path = S3_FEATURES_FOR_TRAINING
    top_50_features_path: Path = project_root / "models" / "TOP_50_FEATURES.json"
    m1_model_path: Path = S7_M1_MODEL_PKL
    m2_model_path: Path = S7_M2_MODEL_PKL
    m1_calibrator_path: Path = S7_M1_CALIBRATED
    m2_calibrator_path: Path = S7_M2_CALIBRATED

    kelly_fraction: float = 0.5
    max_risk_per_trade: float = 0.005  # 0.5%
    f_star_threshold: float = 0.0
    m2_proba_threshold: float = 0.6
    test_limit_partitions: int = 0
    oof_mode: bool = True  # V5はOOF必須のためデフォルトをTrueに変更
    min_capital_threshold: float = 1.0
    min_lot_size: float = 0.01

    max_positions: int = 50

    base_leverage: float = 2000.0
    spread_pips: float = 16.0
    value_per_pip: float = 1.0

    # ==========================================
    # V5 新規追加: 取引ロジックの固定パラメータ
    # ==========================================
    sl_multiplier: float = 5.0
    pt_multiplier: float = 1.0
    payoff_ratio: float = 0.2  # PT_MULTIPLIER / SL_MULTIPLIER


class BacktestSimulator:
    def __init__(self, config: BacktestConfig):
        self.config = config

        # V5はOOFモード専用（Two-Brainアーキテクチャ）のため、旧V4モデルのロード処理を完全に削除
        if not self.config.oof_mode:
            raise ValueError(
                "V5 Project Cimera requires OOF mode (--oof). In-Sample mode is disabled."
            )

        self._current_capital = Decimal(str(self.config.initial_capital))

    def _load_top_50_features(self) -> List[str]:
        logging.info(
            f"Loading Top 50 features list from {self.config.top_50_features_path}..."
        )
        if not self.config.top_50_features_path.exists():
            raise FileNotFoundError(
                f"Top 50 features file not found at: {self.config.top_50_features_path}"
            )
        with open(self.config.top_50_features_path, "r", encoding="utf-8") as f:
            features = json.load(f)
        return features

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
            features = [line.strip() for line in f if line.strip()]
        return features

    def _get_effective_leverage(self, equity: Decimal) -> Decimal:
        """有効証拠金に基づいてExnessのレバレッジ制限を適用"""
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
        """JST時間帯に基づいてExnessの最大ロット数を返す"""
        timestamp_jst = timestamp_utc.astimezone(JST)
        hour_jst = timestamp_jst.hour
        if 6 <= hour_jst < 16:
            return Decimal("20")
        else:
            return Decimal("200")

    def _prepare_data(self) -> Tuple[pl.LazyFrame, pl.DataFrame]:
        # V5仕様: S6から読み込む基本カラム (direction等は削除)
        base_cols = [
            "timestamp",
            "close",
            "atr_value",
            "t1",  # 取引終了時刻
            "duration_long",  # V5追加: Long経過時間
            "duration_short",  # V5追加: Short経過時間
        ]

        if not self.config.oof_mode:  # In-Sample Mode
            logging.info(
                f"Preparing data from {self.config.simulation_data_path} (In-Sample Mode)..."
            )
            required_cols_set = set(self.features_base)
            required_cols_set.update(base_cols)
            required_cols_set.update(self.top_50_features)
            required_cols = list(required_cols_set)

            lf = pl.scan_parquet(str(self.config.simulation_data_path / "**/*.parquet"))
            schema = lf.collect_schema()

            missing_cols = [col for col in required_cols if col not in schema]
            if missing_cols:
                raise ValueError(
                    f"CRITICAL: Required columns not found in S6 schema! Missing: {missing_cols}"
                )

            lf = lf.select(required_cols).sort("timestamp")

        else:  # OOF Mode (V5 Bidirectional)
            logging.info(
                f"Preparing base data (S6) and merging Bidirectional OOF (Long/Short)..."
            )

            # 1. Base Data (S6)
            base_lf = pl.scan_parquet(
                str(self.config.simulation_data_path / "**/*.parquet")
            ).select(base_cols)

            oof_cols = ["timestamp", "prediction", "true_label", "uniqueness"]

            # 2. Long OOF
            long_lf = (
                pl.scan_parquet(self.config.oof_long_path)
                .select(oof_cols)
                .rename(
                    {
                        "prediction": "m2_proba_long",
                        "true_label": "label_long",
                        "uniqueness": "uniqueness_long",
                    }
                )
            )

            # 3. Short OOF
            short_lf = (
                pl.scan_parquet(self.config.oof_short_path)
                .select(oof_cols)
                .rename(
                    {
                        "prediction": "m2_proba_short",
                        "true_label": "label_short",
                        "uniqueness": "uniqueness_short",
                    }
                )
            )

            # 4. Merge (Two-Brain)
            # base_lf を軸にして、timestampでLeft Join。両建てが発生する行は両方に確率が入る。
            lf = (
                base_lf.join(long_lf, on="timestamp", how="left")
                .join(short_lf, on="timestamp", how="left")
                .sort("timestamp")
            )

            # V5追加: タイムアウト決済用の未来価格を asof join (forward) で事前結合
            price_lf_long = base_lf.select(
                [
                    pl.col("timestamp").alias("ts_future"),
                    pl.col("close").alias("close_future_long"),
                ]
            ).sort("ts_future")
            price_lf_short = base_lf.select(
                [
                    pl.col("timestamp").alias("ts_future"),
                    pl.col("close").alias("close_future_short"),
                ]
            ).sort("ts_future")

            lf = lf.with_columns(
                (pl.col("timestamp") + pl.duration(minutes=15)).alias("ts_plus_15m")
            )
            lf = lf.join_asof(
                price_lf_long,
                left_on="ts_plus_15m",
                right_on="ts_future",
                strategy="forward",
            ).drop(["ts_plus_15m", "ts_future"])

            lf = lf.with_columns(
                (pl.col("timestamp") + pl.duration(minutes=5)).alias("ts_plus_5m")
            )
            lf = lf.join_asof(
                price_lf_short,
                left_on="ts_plus_5m",
                right_on="ts_future",
                strategy="forward",
            ).drop(["ts_plus_5m", "ts_future"])

            # Null埋め (予測がない場合は確率0として扱う)
            lf = lf.with_columns(
                [
                    pl.col("m2_proba_long").fill_null(0.0),
                    pl.col("m2_proba_short").fill_null(0.0),
                ]
            )

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
        return lf, partitions_df

    def _run_ai_predictions(self, df_chunk: pl.DataFrame) -> pl.DataFrame:
        """
        V5仕様: OOFモードを主軸とするため、In-Sample(動的推論)モードは
        Two-Brainモデルのロードが別途必要になります。
        今回はOOFデータ(結合済み)をそのまま返すパスをデフォルトとします。
        """
        logging.debug(f"Running AI processing for chunk (size: {len(df_chunk)})...")
        if not self.config.oof_mode:
            logging.error(
                "In-Sample mode requires V5 Two-Brain model architecture. Please use OOF mode (--oof) for V5."
            )
            raise NotImplementedError(
                "In-Sample mode for V5 is not fully implemented yet."
            )
        else:
            # OOF Mode: _prepare_data で既に結合・確率マッピング済み
            return df_chunk

    def _run_simulation_loop(
        self, df_chunk: pl.DataFrame
    ) -> Tuple[pl.DataFrame, pl.DataFrame]:
        trade_log_chunk = []
        equity_values_chunk = []
        current_capital = self._current_capital

        # --- 定数の初期化 ---
        DECIMAL_ZERO = Decimal("0.0")
        DECIMAL_ONE = Decimal("1.0")
        DECIMAL_MAX_RISK = Decimal(str(self.config.max_risk_per_trade))
        DECIMAL_KELLY_FRACTION = Decimal(str(self.config.kelly_fraction))
        DECIMAL_F_STAR_THRESHOLD = Decimal(str(self.config.f_star_threshold))
        DECIMAL_MIN_CAPITAL = Decimal(str(self.config.min_capital_threshold))
        DECIMAL_VALUE_PER_PIP = Decimal(str(self.config.value_per_pip))
        DECIMAL_MIN_LOT_SIZE = Decimal(str(self.config.min_lot_size))
        DECIMAL_CONTRACT_SIZE = CONTRACT_SIZE

        # V5 固定パラメータ
        DECIMAL_PAYOFF_RATIO = Decimal(str(self.config.payoff_ratio))
        DECIMAL_SL_MULT = Decimal(str(self.config.sl_multiplier))
        DECIMAL_PT_MULT = Decimal(
            str(self.config.pt_multiplier)
        )  # 追加: 利確マルチプライヤー

        active_exit_times = []
        MAX_POSITIONS = self.config.max_positions

        # --- DataFrameからのデータ抽出 (高速化のためリスト/Numpy配列化) ---
        timestamps_chunk = df_chunk["timestamp"].to_list()
        t1_chunk = df_chunk["t1"].to_list()
        close_prices_chunk = df_chunk["close"].to_numpy()
        atr_values_chunk = df_chunk["atr_value"].to_numpy()

        # V5 Two-Brain の確率とラベル
        p_long_chunk = df_chunk["m2_proba_long"].to_numpy()
        p_short_chunk = df_chunk["m2_proba_short"].to_numpy()
        labels_long_chunk = df_chunk["label_long"].to_numpy()
        labels_short_chunk = df_chunk["label_short"].to_numpy()

        # V5 追加: TO計算用の経過時間と未来価格
        duration_long_chunk = df_chunk["duration_long"].to_numpy()
        duration_short_chunk = df_chunk["duration_short"].to_numpy()
        close_future_long_chunk = df_chunk["close_future_long"].to_numpy()
        close_future_short_chunk = df_chunk["close_future_short"].to_numpy()

        for i in range(len(df_chunk)):
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

            # 決済時刻を過ぎたポジションをクリア
            active_exit_times = [
                t for t in active_exit_times if t > current_timestamp_int
            ]

            # =========================================================
            # V5 両建て評価ロジック (Long -> Short の順で独立評価)
            # =========================================================
            directions_to_evaluate = [
                (
                    1,
                    p_long_chunk[i],
                    labels_long_chunk[i],
                    duration_long_chunk[i],
                    close_future_long_chunk[i],
                ),  # Long評価
                (
                    -1,
                    p_short_chunk[i],
                    labels_short_chunk[i],
                    duration_short_chunk[i],
                    close_future_short_chunk[i],
                ),  # Short評価
            ]

            # 同一timestampで複数取引が発生する可能性があるため、ループごとに資本を更新
            traded_in_this_step = False

            for (
                direction_int,
                p_float,
                actual_label,
                duration_float,
                close_future_float,
            ) in directions_to_evaluate:
                # NoneやNaNの回避
                if p_float is None or not np.isfinite(p_float):
                    continue

                # ポジション数上限チェック (Long/Short それぞれ1枠消費)
                if len(active_exit_times) >= MAX_POSITIONS:
                    continue

                p_decimal = Decimal(str(p_float))
                q_decimal = DECIMAL_ONE - p_decimal
                b_decimal = DECIMAL_PAYOFF_RATIO

                # ケリー基準 f* の計算
                kelly_f_star = (
                    (b_decimal * p_decimal - q_decimal) / b_decimal
                    if b_decimal > DECIMAL_ZERO
                    else DECIMAL_ZERO
                )
                f_star = (
                    kelly_f_star.copy_abs().max(DECIMAL_ZERO)
                    if kelly_f_star.is_finite()
                    else DECIMAL_ZERO
                )

                # エントリー判定
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
                        risk_amount_decimal = current_capital * effective_bet_fraction

                        if (
                            atr_value_float is None
                            or not np.isfinite(atr_value_float)
                            or atr_value_float <= 0
                        ):
                            continue

                        # 動的SL幅 (価格単位) * 固定SLマルチプライヤー
                        dynamic_sl_PRICE_decimal = (
                            Decimal(str(atr_value_float)) * DECIMAL_SL_MULT
                        )

                        # 1ロットあたりのストップロス価値（通貨）
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

                        # 最小ロット補正
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
                                continue  # スプレッドコスト破綻

                            capital_before_pnl = current_capital - spread_cost_decimal

                            pnl = DECIMAL_ZERO
                            valid_label = (
                                actual_label
                                if (
                                    actual_label is not None
                                    and np.isfinite(actual_label)
                                )
                                else 0
                            )
                            duration_val = (
                                duration_float
                                if (
                                    duration_float is not None
                                    and np.isfinite(duration_float)
                                )
                                else 0.0
                            )

                            # V5 追加: Exit Price ベースの厳密な PnL 計算 (スプレッド二重取り回避)
                            exit_price_decimal = current_price_decimal
                            if direction_int == 1:  # Long
                                if valid_label == 1:
                                    exit_price_decimal = current_price_decimal + (
                                        Decimal(str(atr_value_float)) * DECIMAL_PT_MULT
                                    )
                                elif valid_label == 0 and duration_val < 15.0:
                                    exit_price_decimal = current_price_decimal - (
                                        Decimal(str(atr_value_float)) * DECIMAL_SL_MULT
                                    )
                                else:  # タイムアウト
                                    future_p = (
                                        close_future_float
                                        if (
                                            close_future_float is not None
                                            and np.isfinite(close_future_float)
                                        )
                                        else current_price_float
                                    )
                                    exit_price_decimal = Decimal(str(future_p))
                                pnl = (
                                    (exit_price_decimal - current_price_decimal)
                                    * final_lot_size_decimal
                                    * DECIMAL_CONTRACT_SIZE
                                )

                            else:  # Short
                                if valid_label == 1:
                                    exit_price_decimal = current_price_decimal - (
                                        Decimal(str(atr_value_float)) * DECIMAL_PT_MULT
                                    )
                                elif valid_label == 0 and duration_val < 5.0:
                                    exit_price_decimal = current_price_decimal + (
                                        Decimal(str(atr_value_float)) * DECIMAL_SL_MULT
                                    )
                                else:  # タイムアウト
                                    future_p = (
                                        close_future_float
                                        if (
                                            close_future_float is not None
                                            and np.isfinite(close_future_float)
                                        )
                                        else current_price_float
                                    )
                                    exit_price_decimal = Decimal(str(future_p))
                                pnl = (
                                    (current_price_decimal - exit_price_decimal)
                                    * final_lot_size_decimal
                                    * DECIMAL_CONTRACT_SIZE
                                )

                            next_capital = capital_before_pnl + pnl
                            current_capital = (
                                next_capital
                                if next_capital.is_finite()
                                else DECIMAL_ZERO
                            )

                            # T1終了時刻を登録
                            if t1_chunk[i] is not None:
                                current_t1_dt = t1_chunk[i].replace(
                                    tzinfo=dt.timezone.utc
                                )
                                new_exit_time = int(
                                    current_t1_dt.timestamp() * 1_000_000
                                )
                                active_exit_times.append(new_exit_time)

                            traded_in_this_step = True

                            # ログへの記録
                            trade_log_chunk.append(
                                {
                                    "timestamp": current_timestamp,
                                    "pnl": pnl,
                                    "capital_after_trade": current_capital,
                                    "m2_proba": p_float,
                                    "direction": direction_int,
                                    "label": valid_label,
                                    "kelly_f_star": kelly_f_star,
                                    "f_star": f_star,
                                    "base_bet_fraction": base_bet_fraction,
                                    "capped_bet_fraction": capped_bet_fraction,
                                    "effective_bet_fraction": effective_bet_fraction,
                                    "lot_size": float(final_lot_size_decimal),
                                    "atr_value": atr_value_float,
                                    "effective_leverage": float(
                                        effective_leverage_decimal
                                    ),
                                    "margin_required": margin_required_decimal,
                                    "spread_cost": spread_cost_decimal,
                                    "close_price": current_price_decimal,
                                }
                            )

            # 資本の時系列記録
            equity_values_chunk.append(current_capital)

        self._current_capital = current_capital

        results_chunk_df = pl.DataFrame(
            {
                "timestamp": timestamps_chunk,
                "equity": pl.Series("equity", equity_values_chunk, dtype=pl.Object),
            }
        )

        # V5仕様のトレードログスキーマ
        trade_log_schema = {
            "timestamp": pl.Datetime,
            "pnl": pl.Object,
            "capital_after_trade": pl.Object,
            "m2_proba": pl.Float64,
            "direction": pl.Int8,
            "label": pl.Int64,
            "kelly_f_star": pl.Object,
            "f_star": pl.Object,
            "base_bet_fraction": pl.Object,
            "capped_bet_fraction": pl.Object,
            "effective_bet_fraction": pl.Object,
            "lot_size": pl.Float64,
            "atr_value": pl.Float64,
            "effective_leverage": pl.Float32,
            "margin_required": pl.Object,
            "spread_cost": pl.Object,
            "close_price": pl.Object,
        }

        if trade_log_chunk:
            trade_log_data = {
                key: [d.get(key) for d in trade_log_chunk]
                for key in trade_log_schema.keys()
            }
            series_dict = {
                key: pl.Series(key, trade_log_data.get(key), dtype=dtype)
                for key, dtype in trade_log_schema.items()
            }
            trade_log_chunk_df = pl.DataFrame(series_dict)
        else:
            trade_log_chunk_df = pl.DataFrame(schema=trade_log_schema)

        return results_chunk_df, trade_log_chunk_df

    def _analyze_and_report(self, results_df: pl.DataFrame, trade_log: pl.DataFrame):
        logging.info("Analyzing results and generating V5 report...")
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

            logging.info("  -> Calculating drawdown (Polars optimized)...")
            initial_equity_series = pl.Series(
                "equity", [initial_capital], dtype=pl.Object
            )

            equity_series_decimal = pl.concat(
                [initial_equity_series, results_df["equity"]]
            )

            equity_series_float = equity_series_decimal.map_elements(
                lambda d: float(d) if d is not None and d.is_finite() else np.nan,
                return_dtype=pl.Float64,
            ).fill_null(strategy="forward")

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
                rolling_max_series = equity_series_float.cum_max().alias("rolling_max")
                drawdown_series_pct = (
                    ((equity_series_float - rolling_max_series) / rolling_max_series)
                    .fill_nan(0.0)
                    .alias("drawdown")
                )
                drawdown = drawdown_series_pct.slice(1)
                max_drawdown_raw = drawdown_series_pct.min()
                max_drawdown = (
                    max_drawdown_raw
                    if max_drawdown_raw is not None and np.isfinite(max_drawdown_raw)
                    else 0.0
                )
            logging.info("  -> Drawdown calculation complete.")

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

            # V5 仕様: 1=Win, 0=Lose
            winning_trades = trade_log.filter(pl.col("label") == 1)
            losing_trades = trade_log.filter(pl.col("label") == 0)

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
            "strategy": f"V5 Two-Brain (Kelly: {self.config.kelly_fraction}, Base Lev: {self.config.base_leverage}, Max Risk: {self.config.max_risk_per_trade * 100:.1f}%, SL Mult: {self.config.sl_multiplier}, PT Mult: {self.config.pt_multiplier})",
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
        print("    Project Forge V5 Backtest Performance Report")
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
                equity_list_float = equity_series_float.slice(1).to_list()
                drawdown_list_raw = drawdown.to_list()

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
                    f"V5 Equity Curve (Kelly: {self.config.kelly_fraction}, Lev: {self.config.base_leverage}, Max Risk: {self.config.max_risk_per_trade * 100:.1f}%)",
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

        if not trade_log.is_empty():
            trade_log_output_path = (
                FINAL_REPORT_PATH.parent / "detailed_trade_log_v5.csv"
            )
            logging.info(
                f"Preparing detailed trade log for CSV output ({len(trade_log)} trades)..."
            )
            try:
                temp_log_formatted = trade_log.clone()
                format_expressions = []

                format_expressions.append(
                    pl.col("timestamp")
                    .dt.strftime("%Y-%m-%d %H:%M:%S")
                    .alias("timestamp")
                )

                decimal_cols_round = {
                    "capital_after_trade": 2,
                    "pnl": 2,
                    "kelly_f_star": 4,
                    "spread_cost": 2,
                    "margin_required": 2,
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

                float_cols_round = {
                    "m2_proba": 4,
                    "lot_size": 2,
                    "atr_value": 4,
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

                desired_columns_final = [
                    "timestamp",
                    "direction",
                    "label",
                    "m2_proba",
                    "pnl",
                    "capital_after_trade",
                    "lot_size",
                    "spread_cost",
                    "margin_required",
                    "kelly_f_star",
                    "effective_bet_fraction",
                    "effective_leverage",
                    "close_price",
                    "atr_value",
                ]

                available_columns_final = [
                    col
                    for col in desired_columns_final
                    if col in temp_log_formatted.columns
                ]
                trade_log_final_csv = temp_log_formatted.select(available_columns_final)

                trade_log_final_csv.write_csv(
                    trade_log_output_path,
                    null_value="NaN",
                )
                logging.info(
                    f"Formatted detailed trade log saved to {trade_log_output_path}"
                )

            except PermissionError as pe:
                logging.error(
                    f"Permission denied saving trade log to {trade_log_output_path}: {pe}"
                )
            except Exception as e:
                logging.error(
                    f"Failed to save formatted detailed trade log: {e}", exc_info=True
                )
        else:
            logging.info("No trades were executed, skipping detailed trade log output.")

        text_report_path = FINAL_REPORT_PATH.with_suffix(".txt")
        logging.info(f"Generating text performance report to {text_report_path}...")
        try:
            with open(text_report_path, "w", encoding="utf-8") as f:
                f.write("=" * 60 + "\n")
                f.write("    V5 Two-Brain Strategy Performance Report (MT5 Style)\n")
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


if __name__ == "__main__":
    default_config = BacktestConfig()

    parser = argparse.ArgumentParser(
        description="Project Forge V5 Backtest Simulator (Two-Brain, Kelly + Dynamic SL + Constraints)"
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
    parser.add_argument(
        "--oof", action="store_true", help="Run in OOF mode (Required for V5)."
    )
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
        help=f"Value per lot per pip. Default: {default_config.value_per_pip}",
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

    # V5 新規追加パラメータ
    parser.add_argument(
        "--sl-multiplier",
        type=float,
        default=default_config.sl_multiplier,
        dest="sl_multiplier",
        help=f"Stop Loss multiplier for ATR. Default: {default_config.sl_multiplier}",
    )
    parser.add_argument(
        "--pt-multiplier",
        type=float,
        default=default_config.pt_multiplier,
        dest="pt_multiplier",
        help=f"Profit Target multiplier for ATR. Default: {default_config.pt_multiplier}",
    )
    parser.add_argument(
        "--oof-long",
        type=str,
        default=str(default_config.oof_long_path),
        dest="oof_long_path",
        help=f"Path to Long OOF predictions. Default: {default_config.oof_long_path}",
    )
    parser.add_argument(
        "--oof-short",
        type=str,
        default=str(default_config.oof_short_path),
        dest="oof_short_path",
        help=f"Path to Short OOF predictions. Default: {default_config.oof_short_path}",
    )

    args = parser.parse_args()

    # payoff_ratio を動的計算 (PT / SL)
    calculated_payoff_ratio = args.pt_multiplier / args.sl_multiplier

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
        sl_multiplier=args.sl_multiplier,
        pt_multiplier=args.pt_multiplier,
        payoff_ratio=calculated_payoff_ratio,
        oof_long_path=Path(args.oof_long_path),
        oof_short_path=Path(args.oof_short_path),
    )

    if not (0 < config.max_risk_per_trade <= 1.0):
        parser.error("--max-risk must be between 0 (exclusive) and 1.0 (inclusive).")
    if config.base_leverage < 1.0:
        parser.error("--base-leverage must be >= 1.0.")
    if config.value_per_pip <= 0:
        parser.error("--value-per-pip must be greater than 0.")
    if config.spread_pips < 0:
        parser.error("--spread-pips cannot be negative.")
    if config.sl_multiplier <= 0:
        parser.error("--sl-multiplier must be greater than 0.")

    simulator = BacktestSimulator(config)
    simulator.run()
