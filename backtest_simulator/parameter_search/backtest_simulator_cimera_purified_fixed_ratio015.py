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
# import zoneinfo

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
project_root = Path(__file__).resolve().parents[2]
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

# --- blueprintから必要なパスをインポート (V5仕様に合わせてパスを調整) ---
from blueprint import (
    S6_WEIGHTED_DATASET,
    S3_SELECTED_FEATURES_PURIFIED_DIR,
    S7_M2_OOF_PREDICTIONS_LONG,
    S7_M2_OOF_PREDICTIONS_SHORT,
    S7_MODELS,
)

# --- 出力ファイルパス ---
# [一時的] 出力先をハードコード (001の部分をスクリプトごとに書き換える)
OUTPUT_DIR = Path(
    "/workspace/data/XAUUSD/stratum_7_models/1A_2B/Chimera -4- Reunion first stage 110/backtast data/015"
)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)  # フォルダが存在しなければ自動作成

FINAL_REPORT_PATH = OUTPUT_DIR / "final_backtest_report_v5.json"
EQUITY_CURVE_PATH = OUTPUT_DIR / "equity_curve_v5.png"

# --- ロギング設定 ---
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

# XAUUSDの契約サイズ
CONTRACT_SIZE = Decimal("100")  # 1 lot = 100 oz

# # タイムゾーン変換用
# JST = zoneinfo.ZoneInfo("Asia/Tokyo")


# ================================================================
# フェーズ 0: 作戦司令室 (パラメータ設定 - V5仕様)
# ================================================================
@dataclass
class BacktestConfig:
    """シミュレーションの全パラメータを一元管理 (V5 Two-Brain Architecture)"""

    initial_capital: float = 100.0
    simulation_data_path: Path = S6_WEIGHTED_DATASET

    # V5: Long/Short独立のOOF予測パス
    oof_long_path: Path = S7_M2_OOF_PREDICTIONS_LONG
    oof_short_path: Path = S7_M2_OOF_PREDICTIONS_SHORT

    # V5: 純化特徴量ディレクトリ (In-Sample拡張時の布石として保持)
    purified_features_dir: Path = S3_SELECTED_FEATURES_PURIFIED_DIR

    # --- V5 新規: 自動ロット調整と発火閾値 ---
    auto_lot_base_capital: float = 1000.0
    auto_lot_size_per_base: float = 1.0

    # ▼▼▼ 追加: 固定比率資金管理のパラメータ ▼▼▼
    use_fixed_risk: bool = True
    fixed_risk_percent: float = (
        0.20  # 口座残高の何%を1トレードのリスクとするか (0.02 = 2%)
    )
    # ▲▲▲ ここまで追加 ▲▲▲

    m2_proba_threshold: float = 0.50
    test_limit_partitions: int = 0
    oof_mode: bool = True
    min_capital_threshold: float = 1.0
    min_lot_size: float = 0.01

    max_positions: int = 1000

    # --- V5 新規: サーキットブレーカーと同時発注禁止 ---
    prevent_simultaneous_orders: bool = True
    max_consecutive_sl: int = 1
    cooldown_minutes_after_sl: int = 10

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

        # V5仕様: OOFモード専用のため、旧In-Sample用のモデルやTop 50特徴量のロードは完全撤廃
        if not self.config.oof_mode:
            raise NotImplementedError(
                "In-Sample mode is disabled in V5. Please use --oof."
            )

        self._current_capital = Decimal(str(self.config.initial_capital))

    def _get_effective_leverage(self, equity: Decimal) -> Decimal:
        """
        有効証拠金に基づいてExnessのレバレッジ制限を適用

        [FIX-3] extreme_risk_engine._get_exness_leverage() と完全統一:
          equity < $5,000    → base_leverage そのまま
          equity < $30,000   → 2000倍上限
          equity < $100,000  → 1000倍上限
          equity >= $100,000 → 500倍上限
        """
        base_leverage_dec = Decimal(str(self.config.base_leverage))
        if equity < Decimal("5000"):
            limit_leverage = base_leverage_dec  # 上限なし (base_leverage に従う)
        elif equity < Decimal("30000"):
            limit_leverage = Decimal("2000")
        elif equity < Decimal("100000"):
            limit_leverage = Decimal("1000")
        else:
            limit_leverage = Decimal("500")
        return base_leverage_dec.min(limit_leverage)

    def run(self):
        logging.info("### Project Forge V5 Backtest Simulator: START ###")
        logging.info(
            f"Strategy: Auto Lot (Base={self.config.auto_lot_base_capital}, Size={self.config.auto_lot_size_per_base}), "
            f"Base Leverage = {self.config.base_leverage}, "
            f"Spread = {self.config.spread_pips} pips"
        )
        logging.info(
            f"Bankruptcy Threshold (Min Capital): {self.config.min_capital_threshold:,.2f}"
        )
        logging.info(
            f"Lot Size Params: Value = {self.config.value_per_pip}/lot/pip, Contract Size = 100"
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

        # ▼▼▼ 新規追加: サーキットブレーカーの統計と最高資金(HWM)の初期化 ▼▼▼
        self.cb_simultaneous_prevented = 0
        self.cb_cooldown_long = 0
        self.cb_cooldown_short = 0
        self.high_water_mark = self._current_capital
        # ▲▲▲ ここまで追加 ▲▲▲

        for row in tqdm(
            partitions_to_process.iter_rows(named=True),
            total=len(partitions_to_process),
            desc="Simulating Partitions",
        ):
            current_date = row["date"]

            logging.debug(f"Processing partition: {current_date}")
            try:
                # Polars LazyFrameを日付ごとにcollectしてメモリを節約
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
                if self._current_capital < DECIMAL_MIN_CAPITAL:
                    logging.warning(
                        f"Capital ({self._current_capital:,.2f}) fell below threshold ({DECIMAL_MIN_CAPITAL:,.2f}) "
                        f"before processing {current_date}. Stopping simulation."
                    )
                    break

                # V5仕様: OOFデータの事前結合が済んでいるため、AI推論をスキップして直接シミュレーションへ
                results_chunk_df, trade_log_chunk_df = self._run_simulation_loop(
                    df_chunk
                )

                all_results_dfs.append(results_chunk_df)
                all_trade_logs.append(trade_log_chunk_df)

                del df_chunk, results_chunk_df, trade_log_chunk_df
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
        logging.info("### Project Forge V5 Backtest Simulator: FINISHED ###")

    def _prepare_data(self) -> Tuple[pl.LazyFrame, pl.DataFrame]:
        # V5仕様: S6から読み込む基本カラム (direction等は削除)
        base_cols = [
            "timestamp",
            "close",
            "atr_value",
            "duration_long",  # V5追加: Long経過時間
            "duration_short",  # V5追加: Short経過時間
        ]

        if not self.config.oof_mode:  # In-Sample Mode
            raise NotImplementedError(
                "In-Sample mode is not supported in V5. Data preparation requires OOF files."
            )

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
        DECIMAL_MIN_CAPITAL = Decimal(str(self.config.min_capital_threshold))
        DECIMAL_VALUE_PER_PIP = Decimal(str(self.config.value_per_pip))
        DECIMAL_MIN_LOT_SIZE = Decimal(str(self.config.min_lot_size))
        DECIMAL_CONTRACT_SIZE = CONTRACT_SIZE

        # V5 固定パラメータ
        DECIMAL_PAYOFF_RATIO = Decimal(str(self.config.payoff_ratio))
        DECIMAL_SL_MULT = Decimal(str(self.config.sl_multiplier))
        DECIMAL_PT_MULT = Decimal(str(self.config.pt_multiplier))

        # --- サーキットブレーカー用状態管理 ---
        pending_exits = []  # [(exit_time_int, direction_int, is_sl)]
        consecutive_sl_long = 0
        consecutive_sl_short = 0
        cooldown_until_long = 0
        cooldown_until_short = 0

        active_exit_times = []
        MAX_POSITIONS = self.config.max_positions

        # --- DataFrameからのデータ抽出 (高速化のためリスト/Numpy配列化) ---
        timestamps_chunk = df_chunk["timestamp"].to_list()
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

            # =========================================================
            # 完了したポジションの精算（SLカウントとクールダウン管理）
            # =========================================================
            finished_positions = [
                p for p in pending_exits if p[0] <= current_timestamp_int
            ]
            pending_exits = [p for p in pending_exits if p[0] > current_timestamp_int]

            for exit_time, direction, is_sl in sorted(
                finished_positions, key=lambda x: x[0]
            ):
                if direction == 1:
                    if is_sl:
                        consecutive_sl_long += 1
                        if consecutive_sl_long >= self.config.max_consecutive_sl:
                            cooldown_until_long = exit_time + int(
                                self.config.cooldown_minutes_after_sl * 60 * 1_000_000
                            )
                            consecutive_sl_long = 0
                            self.cb_cooldown_long += 1  # ★追加
                    else:
                        consecutive_sl_long = 0
                else:
                    if is_sl:
                        consecutive_sl_short += 1
                        if consecutive_sl_short >= self.config.max_consecutive_sl:
                            cooldown_until_short = exit_time + int(
                                self.config.cooldown_minutes_after_sl * 60 * 1_000_000
                            )
                            consecutive_sl_short = 0
                            self.cb_cooldown_short += 1  # ★追加
                    else:
                        consecutive_sl_short = 0

            # 決済時刻を過ぎたポジションをクリア
            active_exit_times = [
                t for t in active_exit_times if t > current_timestamp_int
            ]

            # =========================================================
            # 同時発注禁止フィルター
            # =========================================================
            should_trade_long = p_long_chunk[i] > self.config.m2_proba_threshold
            should_trade_short = p_short_chunk[i] > self.config.m2_proba_threshold

            if (
                self.config.prevent_simultaneous_orders
                and should_trade_long
                and should_trade_short
            ):
                should_trade_long = False
                should_trade_short = False
                self.cb_simultaneous_prevented += 1  # ★追加

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
                    should_trade_long,
                ),  # Long評価
                (
                    -1,
                    p_short_chunk[i],
                    labels_short_chunk[i],
                    duration_short_chunk[i],
                    close_future_short_chunk[i],
                    should_trade_short,
                ),  # Short評価
            ]

            # 同一timestampで複数取引が発生する可能性があるため、ループごとに資本を更新

            # 同一timestampで複数取引が発生する可能性があるため、ループごとに資本を更新
            traded_in_this_step = False

            for (
                direction_int,
                p_float,
                actual_label,
                duration_float,
                close_future_float,
                base_should_trade,
            ) in directions_to_evaluate:
                # NoneやNaNの回避
                if p_float is None or not np.isfinite(p_float):
                    continue

                # ポジション数上限チェック (Long/Short それぞれ1枠消費)
                if len(active_exit_times) >= MAX_POSITIONS:
                    continue

                # =========================================================
                # エントリー判定とクールダウン判定
                # =========================================================
                should_trade = base_should_trade

                if direction_int == 1 and current_timestamp_int < cooldown_until_long:
                    should_trade = False
                elif (
                    direction_int == -1 and current_timestamp_int < cooldown_until_short
                ):
                    should_trade = False

                if should_trade:
                    if (
                        atr_value_float is None
                        or not np.isfinite(atr_value_float)
                        or atr_value_float <= 0
                    ):
                        continue

                    # [FIX-4] Auto Lot 計算を extreme_risk_engine.calculate_auto_lot() と統一
                    # 旧: ハードコード乗数方式 (0.25倍/0.5倍) → 新: 証拠金上限数式

                    # ▼▼▼ 修正: 固定比率(Fixed Risk)と固定複利(Auto Lot)の分岐 ▼▼▼
                    if self.config.use_fixed_risk:
                        # --- 固定比率（Fixed Risk）---
                        # Lot = (Balance * risk_pct) / (ATR * SL_Mult * ContractSize)
                        risk_pct_dec = Decimal(str(self.config.fixed_risk_percent))
                        max_loss_amount = current_capital * risk_pct_dec
                        sl_price_distance = (
                            Decimal(str(atr_value_float)) * DECIMAL_SL_MULT
                        )

                        if sl_price_distance > DECIMAL_ZERO:
                            base_lot = max_loss_amount / (
                                sl_price_distance * DECIMAL_CONTRACT_SIZE
                            )
                        else:
                            base_lot = DECIMAL_ZERO
                    else:
                        # --- 従来の固定複利（Auto Lot）---
                        base_capital_dec = Decimal(
                            str(self.config.auto_lot_base_capital)
                        )
                        size_per_base_dec = Decimal(
                            str(self.config.auto_lot_size_per_base)
                        )
                        base_lot = (
                            current_capital / base_capital_dec
                        ) * size_per_base_dec
                    # ▲▲▲ ここまで修正 ▲▲▲

                    # Step2: レバレッジに基づく証拠金上限ロット
                    effective_leverage_decimal = self._get_effective_leverage(
                        current_capital
                    )
                    max_lot_margin = (current_capital * effective_leverage_decimal) / (
                        current_price_decimal * DECIMAL_CONTRACT_SIZE
                    )

                    # Step3: 基本ロット vs 証拠金上限 の小さい方、絶対上限 200 でキャップ
                    raw_lot_size = min(base_lot, max_lot_margin, Decimal("200.0"))

                    # Step4: 0.01刻み切り捨て
                    final_lot_size_decimal = Decimal(int(raw_lot_size * 100)) / Decimal(
                        "100"
                    )

                    # ▼▼▼ 追加: ニート化防止！ 最低ロット数の保証 ▼▼▼
                    final_lot_size_decimal = max(
                        final_lot_size_decimal, DECIMAL_MIN_LOT_SIZE
                    )
                    # ▲▲▲ ここまで追加 ▲▲▲

                    if final_lot_size_decimal >= DECIMAL_MIN_LOT_SIZE:
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

                        # 必要証拠金とスプレッドコストが現在の資金を上回る場合は安全に弾く
                        if (
                            margin_required_decimal + spread_cost_decimal
                            > current_capital
                        ):
                            continue

                        capital_before_pnl = current_capital - spread_cost_decimal

                        pnl = DECIMAL_ZERO
                        is_sl_hit = False  # V5: SL判定フラグ
                        valid_label = (
                            actual_label
                            if (actual_label is not None and np.isfinite(actual_label))
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
                                is_sl_hit = True
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
                                is_sl_hit = True
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
                            next_capital if next_capital.is_finite() else DECIMAL_ZERO
                        )

                        # ▼▼▼ 新規追加: HWMの更新と現在のドローダウン率計算 ▼▼▼
                        self.high_water_mark = max(
                            self.high_water_mark, current_capital
                        )
                        current_dd_pct = (
                            (
                                (current_capital - self.high_water_mark)
                                / self.high_water_mark
                                * Decimal("100.0")
                            )
                            if self.high_water_mark > DECIMAL_ZERO
                            else DECIMAL_ZERO
                        )
                        # ▲▲▲ ここまで追加 ▲▲▲

                        # V5修正: duration_float (分) をマイクロ秒に変換して終了時刻を計算
                        if duration_float is not None and np.isfinite(duration_float):
                            new_exit_time = current_timestamp_int + int(
                                duration_float * 60 * 1_000_000
                            )
                            active_exit_times.append(new_exit_time)
                            pending_exits.append(
                                (new_exit_time, direction_int, is_sl_hit)
                            )

                        traded_in_this_step = True

                        current_active_longs = sum(
                            1 for p in pending_exits if p[1] == 1
                        )
                        current_active_shorts = sum(
                            1 for p in pending_exits if p[1] == -1
                        )

                        # ログへの記録 (カラム名を短縮形に変更 & 新規データ追加)
                        trade_log_chunk.append(
                            {
                                "timestamp": current_timestamp,
                                "pnl": pnl,
                                "balance": current_capital,  # ★変更
                                "m2_proba": float(p_float),
                                "direction": int(direction_int),
                                "label": int(valid_label),
                                "lot_size": float(final_lot_size_decimal),
                                "atr_value": float(atr_value_float),
                                "leverage": float(effective_leverage_decimal),  # ★変更
                                "margin": margin_required_decimal,  # ★変更
                                "spread": spread_cost_decimal,  # ★変更済のはずですが確認
                                "close_price": current_price_decimal,
                                "active(L)": int(current_active_longs),  # ★変更
                                "active(S)": int(current_active_shorts),  # ★変更
                                "TD": float(duration_val),
                                "DD(%)": float(current_dd_pct),  # ★変更
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

        # V5仕様のトレードログスキーマ (名前短縮版)
        trade_log_schema = {
            "timestamp": pl.Datetime,
            "pnl": pl.Object,
            "balance": pl.Object,  # ★変更
            "m2_proba": pl.Float64,
            "direction": pl.Int8,
            "label": pl.Int64,
            "lot_size": pl.Float64,
            "atr_value": pl.Float64,
            "leverage": pl.Float32,  # ★変更
            "margin": pl.Object,  # ★変更済のはずですが確認
            "spread": pl.Object,  # ★変更済のはずですが確認
            "close_price": pl.Object,
            "active(L)": pl.Int32,  # ★変更
            "active(S)": pl.Int32,  # ★変更
            "TD": pl.Float64,
            "DD(%)": pl.Float64,  # ★変更
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

        # ▼▼▼ 修正: レポートのStrategy名に資金管理方式を反映 ▼▼▼
        strategy_str = "V5 Two-Brain "
        if self.config.use_fixed_risk:
            strategy_str += f"Fixed Risk ({self.config.fixed_risk_percent * 100:.1f}%)"
        else:
            strategy_str += f"Auto Lot (Base: {self.config.auto_lot_base_capital}, Size: {self.config.auto_lot_size_per_base})"
        strategy_str += f", M2 Thresh: {self.config.m2_proba_threshold}, SL Mult: {self.config.sl_multiplier}, PT Mult: {self.config.pt_multiplier}"

        report_data = {
            "strategy": strategy_str,
            "initial_capital": float(initial_capital),
            # ▲▲▲ ここまで修正 ▲▲▲
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
            "cb_simultaneous_prevented": self.cb_simultaneous_prevented,  # ★追加
            "cb_cooldown_long": self.cb_cooldown_long,  # ★追加
            "cb_cooldown_short": self.cb_cooldown_short,  # ★追加
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
                    f"V5 Equity Curve (Auto Lot: {self.config.auto_lot_size_per_base} per {self.config.auto_lot_base_capital}, M2 Thresh: {self.config.m2_proba_threshold})",
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
                    "balance": 2,  # ★変更
                    "pnl": 2,
                    "spread": 2,  # ★変更
                    "margin": 2,  # ★変更
                    "close_price": 3,
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
                    "leverage": 0,
                    "TD": 1,
                    "DD(%)": 2,  # ★変更
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
                    "balance",
                    "lot_size",
                    "active(L)",  # ★変更
                    "active(S)",  # ★変更
                    "margin",
                    "leverage",
                    "spread",
                    "close_price",
                    "atr_value",
                    "TD",
                    "DD(%)",  # ★変更
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
            # ▼▼▼ 追加計算: 各種統計情報の取得 ▼▼▼
            if not trade_log.is_empty():
                max_active_l = trade_log["active(L)"].max()
                max_active_s = trade_log["active(S)"].max()
                max_active_tot = (trade_log["active(L)"] + trade_log["active(S)"]).max()

                l_trades = trade_log.filter(pl.col("direction") == 1)
                s_trades = trade_log.filter(pl.col("direction") == -1)
                count_l = len(l_trades)
                count_s = len(s_trades)
                avg_td_l = l_trades["TD"].mean() if count_l > 0 else 0.0
                avg_td_s = s_trades["TD"].mean() if count_s > 0 else 0.0

                to_count = len(
                    trade_log.filter(
                        (pl.col("label") == 0)
                        & (
                            ((pl.col("direction") == 1) & (pl.col("TD") >= 14.9))
                            | ((pl.col("direction") == -1) & (pl.col("TD") >= 4.9))
                        )
                    )
                )

                m2_lst = trade_log["m2_proba"].to_list()
                m2_bins = {
                    "<= 0.5": sum(1 for x in m2_lst if x <= 0.5),
                    "0.5 - 0.6": sum(1 for x in m2_lst if 0.5 < x <= 0.6),
                    "0.6 - 0.7": sum(1 for x in m2_lst if 0.6 < x <= 0.7),
                    "0.7 - 0.8": sum(1 for x in m2_lst if 0.7 < x <= 0.8),
                    "0.8 - 0.9": sum(1 for x in m2_lst if 0.8 < x <= 0.9),
                    "> 0.9": sum(1 for x in m2_lst if x > 0.9),
                }
                atr_lst = trade_log["atr_value"].to_list()
                atr_bins = {
                    "<= 0.1": sum(1 for x in atr_lst if x <= 0.1),
                    "0.1 - 0.2": sum(1 for x in atr_lst if 0.1 < x <= 0.2),
                    "0.2 - 0.3": sum(1 for x in atr_lst if 0.2 < x <= 0.3),
                    "0.3 - 0.4": sum(1 for x in atr_lst if 0.3 < x <= 0.4),
                    "0.4 - 0.5": sum(1 for x in atr_lst if 0.4 < x <= 0.5),
                    "> 0.5": sum(1 for x in atr_lst if x > 0.5),
                }
            else:
                max_active_l = max_active_s = max_active_tot = count_l = count_s = (
                    to_count
                ) = 0
                avg_td_l = avg_td_s = 0.0
                m2_bins = atr_bins = {}
            # ▲▲▲ ここまで追加 ▲▲▲

            with open(text_report_path, "w", encoding="utf-8") as f:
                f.write("=" * 60 + "\n")
                f.write("    V5 Two-Brain Strategy Performance Report (MT5 Style)\n")
                f.write("=" * 60 + "\n\n")
                f.write(f"Strategy:\t\t{report_data.get('strategy', 'N/A')}\n")
                f.write(
                    f"Period:\t\t\t{report_data.get('data_period_start', 'N/A')} - {report_data.get('data_period_end', 'N/A')}\n\n"
                )

                # ▼▼▼ 新規追加: BacktestConfigの全パラメーターを出力 ▼▼▼
                f.write("-" * 22 + " Configuration " + "-" * 23 + "\n")
                for key, value in self.config.__dict__.items():
                    f.write(f"{key.ljust(30)}: {value}\n")
                f.write("\n")
                # ▲▲▲ ここまで追加 ▲▲▲

                # ▼▼▼ 新規追加: サーキットブレーカーのサマリーを出力 ▼▼▼
                f.write("-" * 21 + " Circuit Breakers " + "-" * 21 + "\n")
                f.write(
                    f"Simultaneous Orders Prevented:  {report_data.get('cb_simultaneous_prevented', 0)} times\n"
                )
                f.write(
                    f"Cooldown Triggered (Long)    :  {report_data.get('cb_cooldown_long', 0)} times\n"
                )
                f.write(
                    f"Cooldown Triggered (Short)   :  {report_data.get('cb_cooldown_short', 0)} times\n\n"
                )
                # ▲▲▲ ここまで追加 ▲▲▲

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

                # ▼▼▼ 追加出力: 詳細統計と分布 ▼▼▼
                f.write("-" * 23 + " Positions & Durations " + "-" * 14 + "\n")
                f.write(f"Max Concurrent Longs:\t{max_active_l}\n")
                f.write(f"Max Concurrent Shorts:\t{max_active_s}\n")
                f.write(f"Max Concurrent Total:\t{max_active_tot}\n")
                f.write(f"Total Long Trades:\t{count_l}\n")
                f.write(f"Total Short Trades:\t{count_s}\n")
                f.write(f"Avg TD (Long):\t\t{avg_td_l:.1f} mins\n")
                f.write(f"Avg TD (Short):\t\t{avg_td_s:.1f} mins\n")
                f.write(f"Timeout (TO) Count:\t{to_count}\n\n")

                f.write("-" * 23 + " M2 Proba Distribution " + "-" * 14 + "\n")
                for k, v in m2_bins.items():
                    pct = (v / total_trades) * 100 if total_trades > 0 else 0
                    f.write(f"{k.ljust(15)}: {str(v).rjust(6)} ({pct:5.1f} %)\n")
                f.write("\n")

                f.write("-" * 23 + " ATR Value Distribution " + "-" * 13 + "\n")
                for k, v in atr_bins.items():
                    pct = (v / total_trades) * 100 if total_trades > 0 else 0
                    f.write(f"{k.ljust(15)}: {str(v).rjust(6)} ({pct:5.1f} %)\n")
                f.write("\n")
                # ▲▲▲ ここまで追加 ▲▲▲

                f.write("=" * 60 + "\n")
            logging.info(f"Text performance report saved successfully.")
        except Exception as e:
            logging.error(f"Failed to save text performance report: {e}", exc_info=True)


if __name__ == "__main__":
    default_config = BacktestConfig()

    parser = argparse.ArgumentParser(
        description="Project Forge V5 Backtest Simulator (Two-Brain, Auto Lot + Dynamic SL + Timeouts)"
    )

    parser.add_argument(
        "--auto-lot-base",
        type=float,
        default=default_config.auto_lot_base_capital,
        dest="auto_lot_base_capital",
        help=f"Base capital for auto lot calculation. Default: {default_config.auto_lot_base_capital}",
    )
    parser.add_argument(
        "--auto-lot-size",
        type=float,
        default=default_config.auto_lot_size_per_base,
        dest="auto_lot_size_per_base",
        help=f"Lot size per base capital. Default: {default_config.auto_lot_size_per_base}",
    )
    # ▼▼▼ 追加: 引数パーサー ▼▼▼
    parser.add_argument(
        "--use-fixed-risk",
        action="store_true",
        help="Use fixed risk % position sizing instead of auto lot.",
    )
    parser.add_argument(
        "--fixed-risk-pct",
        type=float,
        default=default_config.fixed_risk_percent,
        dest="fixed_risk_pct",
        help=f"Risk percentage for fixed risk sizing (e.g., 0.02 for 2%%). Default: {default_config.fixed_risk_percent}",
    )
    # ▲▲▲ ここまで追加 ▲▲▲
    parser.add_argument(
        "--base-leverage",
        type=float,
        default=default_config.base_leverage,
        dest="base_leverage",
        help=f"Base leverage setting. Default: {default_config.base_leverage}",
    )

    parser.add_argument(
        "--m2-th",
        type=float,
        default=default_config.m2_proba_threshold,
        dest="m2_th",
        help=f"Min M2 prob threshold. Default: {default_config.m2_proba_threshold}",
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

    # --- V5 新規: サーキットブレーカー引数 ---
    parser.add_argument(
        "--allow-simultaneous",
        action="store_true",
        help="Allow simultaneous Long/Short orders (default: prevented)",
    )
    parser.add_argument(
        "--max-consecutive-sl",
        type=int,
        default=default_config.max_consecutive_sl,
        dest="max_consecutive_sl",
        help=f"Max consecutive SLs before cooldown. Default: {default_config.max_consecutive_sl}",
    )
    parser.add_argument(
        "--cooldown-minutes",
        type=int,
        default=default_config.cooldown_minutes_after_sl,
        dest="cooldown_minutes",
        help=f"Cooldown minutes after max SLs. Default: {default_config.cooldown_minutes_after_sl}",
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
        auto_lot_base_capital=args.auto_lot_base_capital,
        auto_lot_size_per_base=args.auto_lot_size_per_base,
        # ▼▼▼ 修正: args ではなく、一番上の設定(default_config)を直接読み込ませる ▼▼▼
        use_fixed_risk=default_config.use_fixed_risk,
        fixed_risk_percent=default_config.fixed_risk_percent,
        # ▲▲▲ ここまで修正 ▲▲▲
        base_leverage=args.base_leverage,
        m2_proba_threshold=args.m2_th,
        test_limit_partitions=args.test_limit_partitions,
        oof_mode=True,
        min_capital_threshold=args.min_capital,
        value_per_pip=args.value_per_pip,
        spread_pips=args.spread_pips,
        max_positions=args.max_positions,
        prevent_simultaneous_orders=not args.allow_simultaneous,
        max_consecutive_sl=args.max_consecutive_sl,
        cooldown_minutes_after_sl=args.cooldown_minutes,
        sl_multiplier=args.sl_multiplier,
        pt_multiplier=args.pt_multiplier,
        payoff_ratio=calculated_payoff_ratio,
        oof_long_path=Path(args.oof_long_path),
        oof_short_path=Path(args.oof_short_path),
    )

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
