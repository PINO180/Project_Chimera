# /workspace/models/backtest_simulator_cimera_purified
# [V5改修版: Project Cimera 双方向ラベリング仕様 (Part 1)]

import sys
import pickle
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
project_root = Path(__file__).resolve().parents[1]
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

# --- blueprintから必要なパスをインポート (V5仕様に合わせてパスを調整) ---
from blueprint import (
    S6_WEIGHTED_DATASET,
    S3_SELECTED_FEATURES_PURIFIED_DIR,
    S7_M1_OOF_PREDICTIONS_LONG,
    S7_M1_OOF_PREDICTIONS_SHORT,
    S7_M2_OOF_PREDICTIONS_LONG,
    S7_M2_OOF_PREDICTIONS_SHORT,
    S7_MODELS,
    S7_BACKTEST_CACHE_M1,
    S7_BACKTEST_CACHE_M2,
    S7_BACKTEST_SIM_RESULTS,
)

# --- 出力ファイルパス（実行時に動的生成）---
FINAL_REPORT_PATH = S7_MODELS / "final_backtest_report_v5.json"  # 起動時に上書き
EQUITY_CURVE_PATH = S7_MODELS / "equity_curve_v5.png"  # 起動時に上書き


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

    initial_capital: float = 1000.0
    simulation_data_path: Path = S6_WEIGHTED_DATASET

    # V5: Long/Short独立のOOF予測パス
    oof_long_path: Path = S7_M2_OOF_PREDICTIONS_LONG
    oof_short_path: Path = S7_M2_OOF_PREDICTIONS_SHORT

    # V5: 純化特徴量ディレクトリ (In-Sample拡張時の布石として保持)
    purified_features_dir: Path = S3_SELECTED_FEATURES_PURIFIED_DIR

    # --- V5 新規: 自動ロット調整と発火閾値 ---
    auto_lot_base_capital: float = 1000.0
    auto_lot_size_per_base: float = 0.1

    # ▼▼▼ 追加: 固定比率資金管理のパラメータ ▼▼▼
    use_fixed_risk: bool = True
    fixed_risk_percent: float = (
        0.02  # 口座残高の何%を1トレードのリスクとするか (0.02 = 2%)
    )
    # ▲▲▲ ここまで追加 ▲▲▲

    m2_proba_threshold: float = 0.70
    m2_delta_threshold: float = 0.50  # ★追加: LongとShortの確率の差分(Delta)閾値

    test_limit_partitions: int = 0
    oof_mode: bool = True
    min_capital_threshold: float = 1.0
    min_lot_size: float = 0.01
    min_atr_threshold: float = (
        0.8  # ★修正: ドル値(2.0) → ATR Ratio閾値(0.8) (プロンプト⑯ 修正②)
    )

    max_positions: int = 100

    # --- V5 新規: サーキットブレーカーと同時発注禁止 ---
    prevent_simultaneous_orders: bool = True
    max_consecutive_sl: int = 2
    cooldown_minutes_after_sl: int = 30

    base_leverage: float = 2000.0
    spread_pips: float = 36.0
    value_per_pip: float = 1.0

    # ==========================================
    # V5 新規追加: 取引ロジックの固定パラメータ
    # ==========================================
    sl_multiplier_long: float = 5.0
    pt_multiplier_long: float = 1.0
    sl_multiplier_short: float = 5.0
    pt_multiplier_short: float = 1.0

    td_minutes_long: float = 30.0
    td_minutes_short: float = 30.0

    # ▼▼▼ ここから追加 ▼▼▼
    # ==========================================
    # V5 Optuna対応: 証拠金維持率とロスカット設定
    # ==========================================
    margin_call_percent: float = 0.0  # 証拠金維持率がこれを下回ると新規エントリー禁止
    stop_out_percent: float = 0.0  # 証拠金維持率がこれを下回ると強制ロスカット
    # ▲▲▲ ここまで追加 ▲▲▲


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

    def preload_data(self) -> Tuple[Dict[dt.date, pl.DataFrame], pl.DataFrame]:
        """
        Optuna超高速化用: 全パーティションのデータを1回だけ読み込み、
        メモリ上の辞書(Dict)にキャッシュして返す。
        """
        logging.info("Pre-loading all data into memory for ultra-fast simulation...")
        lf, partitions_df = self._prepare_data()

        preloaded_dict = {}
        partitions_to_process = partitions_df

        if self.config.test_limit_partitions > 0:
            partitions_to_process = partitions_df.head(
                self.config.test_limit_partitions
            )

        # 全日数をループして、LazyFrame(lf) を DataFrame としてメモリに collect() する
        for row in tqdm(
            partitions_to_process.iter_rows(named=True),
            total=len(partitions_to_process),
            desc="Preloading Partitions to Memory",
        ):
            current_date = row["date"]
            try:
                df_chunk = lf.filter(
                    pl.col("timestamp").dt.date() == current_date
                ).collect()
                if not df_chunk.is_empty():
                    preloaded_dict[current_date] = df_chunk
            except Exception as e:
                logging.error(f"Error preloading partition {current_date}: {e}")

        logging.info(
            f"Successfully preloaded {len(preloaded_dict)} partitions into memory."
        )
        return preloaded_dict, partitions_to_process

    # ▼▼▼ def run(self): を引数付きに変更 ▼▼▼
    def run(
        self, preloaded_data: Tuple[Dict[dt.date, pl.DataFrame], pl.DataFrame] = None
    ):
        logging.info("### Project Forge V5 Backtest Simulator: START ###")
        logging.info(
            f"Strategy: Fixed Risk ({self.config.fixed_risk_percent * 100:.1f}%), "
            f"Base Leverage = {self.config.base_leverage}, "
            f"Spread = {self.config.spread_pips} pips"
        )

        # =========================================================
        # オンメモリデータの受け取り、または単独実行時の自動ロード
        # =========================================================
        if preloaded_data is not None:
            preloaded_dict, partitions_to_process = preloaded_data
            logging.info("Using PRELOADED data from memory (Ultra-fast mode).")
        else:
            # Optunaを使わず、このスクリプトを単独で実行した場合の処理
            preloaded_dict, partitions_to_process = self.preload_data()

        all_results_dfs = []
        all_trade_logs = []

        self._current_capital = Decimal(str(self.config.initial_capital))
        DECIMAL_MIN_CAPITAL = Decimal(str(self.config.min_capital_threshold))

        self.cb_simultaneous_prevented = 0
        self.cb_cooldown_long = 0
        self.cb_cooldown_short = 0
        self.high_water_mark = self._current_capital
        self.min_margin_level_pct = Decimal("inf")
        self.stop_out_count = 0

        # tqdmのプログレスバーはOptuna側で大量に出力されると邪魔なので、
        # オンメモリ(preloaded_dataあり)の場合はバーを非表示(disable=True)にする
        disable_tqdm = preloaded_data is not None

        for row in tqdm(
            partitions_to_process.iter_rows(named=True),
            total=len(partitions_to_process),
            desc="Simulating Partitions",
            disable=disable_tqdm,  # ★追加: Optuna実行時は静かに回す
        ):
            current_date = row["date"]

            # ▼▼▼ 激重だった collect() 処理を廃止し、メモリ(辞書)から一瞬で取り出す ▼▼▼
            df_chunk = preloaded_dict.get(current_date)

            if df_chunk is None or df_chunk.is_empty():
                continue

            try:
                if self._current_capital < DECIMAL_MIN_CAPITAL:
                    # 破産した場合はそれ以降の日付をスキップ
                    break

                # 取得したメモリ上のデータを使ってシミュレーションを実行
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

        # ▼▼▼ 修正前 ▼▼▼
        # self._analyze_and_report(final_results_df, final_trade_log_df)
        # logging.info("### Project Forge V5 Backtest Simulator: FINISHED ###")

        # ▼▼▼ 修正後 ▼▼▼
        report_data = self._analyze_and_report(final_results_df, final_trade_log_df)

        # レポートデータに最低証拠金維持率とストップアウト回数をねじ込む
        report_data["min_margin_level_pct"] = (
            float(self.min_margin_level_pct)
            if self.min_margin_level_pct != Decimal("inf")
            else 9999.0
        )
        report_data["stop_out_count"] = self.stop_out_count

        logging.info("### Project Forge V5 Backtest Simulator: FINISHED ###")
        return report_data

    def _prepare_data(self) -> Tuple[pl.LazyFrame, pl.DataFrame]:
        # V5仕様: timeframe を必須キーとして取得
        base_cols = [
            "timestamp",
            "timeframe",  # ★追加: 行増殖バグを防ぐための必須キー
            "close",
            "atr_value",
            "atr_ratio",  # ★追加: create_proxy_labelsで計算済み・ATR Ratio判定用 (プロンプト⑯ 修正②)
            "duration_long",
            "duration_short",
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

            # ★追加: timeframe を必須キーとして取得
            oof_cols = [
                "timestamp",
                "timeframe",
                "prediction",
                "true_label",
                "uniqueness",
            ]

            # 2. Long OOF (timestampをUTC awareに統一)
            long_lf = (
                pl.scan_parquet(self.config.oof_long_path)
                .select(oof_cols)
                .with_columns(pl.col("timestamp").dt.replace_time_zone("UTC"))
                .rename(
                    {
                        "prediction": "m2_proba_long",
                        "true_label": "label_long",
                        "uniqueness": "uniqueness_long",
                    }
                )
            )

            # 3. Short OOF (timestampをUTC awareに統一)
            short_lf = (
                pl.scan_parquet(self.config.oof_short_path)
                .select(oof_cols)
                .with_columns(pl.col("timestamp").dt.replace_time_zone("UTC"))
                .rename(
                    {
                        "prediction": "m2_proba_short",
                        "true_label": "label_short",
                        "uniqueness": "uniqueness_short",
                    }
                )
            )

            # 4. Merge (Two-Brain)
            # ★修正: timestamp と timeframe の両方で完全一致結合 (NxM増殖を防ぐ)
            lf = (
                base_lf.join(long_lf, on=["timestamp", "timeframe"], how="left")
                .join(short_lf, on=["timestamp", "timeframe"], how="left")
                .sort(["timestamp", "timeframe"])
            )

            # V5追加: タイムアウト決済用の未来価格を asof join (forward) で事前結合
            # ★修正: 未来価格のルックアップテーブルは timestamp で一意にする
            price_lf_long = (
                base_lf.select(
                    [
                        pl.col("timestamp").alias("ts_future"),
                        pl.col("close").alias("close_future_long"),
                    ]
                )
                .unique(subset=["ts_future"], keep="last")
                .sort("ts_future")
            )

            price_lf_short = (
                base_lf.select(
                    [
                        pl.col("timestamp").alias("ts_future"),
                        pl.col("close").alias("close_future_short"),
                    ]
                )
                .unique(subset=["ts_future"], keep="last")
                .sort("ts_future")
            )

            # ★TDのハードコード解除
            lf = lf.with_columns(
                (
                    pl.col("timestamp")
                    + pl.duration(minutes=int(self.config.td_minutes_long))
                ).alias("ts_plus_long")
            )
            lf = lf.join_asof(
                price_lf_long,
                left_on="ts_plus_long",
                right_on="ts_future",
                strategy="forward",
            ).drop(["ts_plus_long", "ts_future"])

            lf = lf.with_columns(
                (
                    pl.col("timestamp")
                    + pl.duration(minutes=int(self.config.td_minutes_short))
                ).alias("ts_plus_short")
            )
            lf = lf.join_asof(
                price_lf_short,
                left_on="ts_plus_short",
                right_on="ts_future",
                strategy="forward",
            ).drop(["ts_plus_short", "ts_future"])

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
        # DECIMAL_PAYOFF_RATIO = Decimal(str(self.config.payoff_ratio))
        # ▼▼▼ 以下の2行を削除またはコメントアウト ▼▼▼
        # DECIMAL_SL_MULT = Decimal(str(self.config.sl_multiplier))
        # DECIMAL_PT_MULT = Decimal(str(self.config.pt_multiplier))
        # ▼▼▼ 修正後 ▼▼▼
        # --- サーキットブレーカー用状態管理 ---
        pending_exits = []  # [(exit_time_int, direction_int, is_sl, margin_used_decimal)] ★変更
        consecutive_sl_long = 0
        consecutive_sl_short = 0
        cooldown_until_long = 0
        cooldown_until_short = 0

        # --- 証拠金トラッキング用 ---
        total_used_margin = DECIMAL_ZERO
        # ▼▼▼ 以下の2行を【削除】してください ▼▼▼
        # min_margin_level_pct = Decimal("inf")
        # stop_out_count = 0
        active_exit_times = []
        MAX_POSITIONS = self.config.max_positions

        # --- DataFrameからのデータ抽出 (高速化のためリスト/Numpy配列化) ---
        timestamps_chunk = df_chunk["timestamp"].to_list()
        close_prices_chunk = df_chunk["close"].to_numpy()
        atr_values_chunk = df_chunk["atr_value"].to_numpy()
        atr_ratios_chunk = df_chunk[
            "atr_ratio"
        ].to_numpy()  # ★追加: ATR Ratio (プロンプト⑯ 修正②)

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
            atr_ratio_float = atr_ratios_chunk[
                i
            ]  # ★追加: ATR Ratio (プロンプト⑯ 修正②)

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
            # 完了したポジションの精算（SLカウントと証拠金の解放）
            # =========================================================
            finished_positions = [
                p for p in pending_exits if p[0] <= current_timestamp_int
            ]
            pending_exits = [p for p in pending_exits if p[0] > current_timestamp_int]

            # ★ 引数に margin_used を追加
            for exit_time, direction, is_sl, margin_used in sorted(
                finished_positions, key=lambda x: x[0]
            ):
                # 証拠金の解放
                total_used_margin -= margin_used
                if total_used_margin < DECIMAL_ZERO:
                    total_used_margin = DECIMAL_ZERO

                if direction == 1:
                    if is_sl:
                        consecutive_sl_long += 1
                        if consecutive_sl_long >= self.config.max_consecutive_sl:
                            cooldown_until_long = exit_time + int(
                                self.config.cooldown_minutes_after_sl * 60 * 1_000_000
                            )
                            consecutive_sl_long = 0
                            self.cb_cooldown_long += 1
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
                            self.cb_cooldown_short += 1
                    else:
                        consecutive_sl_short = 0

            # 決済時刻を過ぎたポジションをクリア
            active_exit_times = [
                t for t in active_exit_times if t > current_timestamp_int
            ]

            # =========================================================
            # リアルタイム証拠金維持率のチェック & 強制ロスカット(Stop Out)
            # =========================================================
            current_margin_level = Decimal("inf")
            if total_used_margin > DECIMAL_ZERO:
                current_margin_level = (current_capital / total_used_margin) * Decimal(
                    "100.0"
                )
                if current_margin_level < self.min_margin_level_pct:  # ★ self. をつける
                    self.min_margin_level_pct = current_margin_level  # ★ self. をつける

                # ストップアウト（強制ロスカット）発動
                if current_margin_level <= Decimal(str(self.config.stop_out_percent)):
                    self.stop_out_count += 1  # ★ self. をつける
                    # 簡易処理: 証拠金の大部分を失い、全ポジションを強制決済する
                    current_capital = total_used_margin * (
                        Decimal(str(self.config.stop_out_percent)) / Decimal("100.0")
                    )
                    total_used_margin = DECIMAL_ZERO
                    pending_exits.clear()
                    active_exit_times.clear()
                    continue

            # =========================================================
            # V5改修: Delta (差分) フィルター & 同時発注禁止ロジック
            # =========================================================
            p_l = p_long_chunk[i]
            p_s = p_short_chunk[i]

            should_trade_long = False
            should_trade_short = False

            # LongとShortの確率の差分（Delta）を計算
            delta = abs(p_l - p_s)

            # 条件1: 差分(Delta)が閾値以上開いていること
            # 条件2: 勝つ方の絶対確率自体も最低限の閾値(m2_proba_threshold)を超えていること
            if delta >= self.config.m2_delta_threshold:
                if p_l > p_s and p_l > self.config.m2_proba_threshold:
                    should_trade_long = True
                elif p_s > p_l and p_s > self.config.m2_proba_threshold:
                    should_trade_short = True

            # 両方Falseのまま（差分が足りない、または絶対確率が足りない）場合はブロックカウント
            if not should_trade_long and not should_trade_short:
                self.cb_simultaneous_prevented += 1

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
                    Decimal(str(self.config.pt_multiplier_long)),  # ★追加
                    Decimal(str(self.config.sl_multiplier_long)),  # ★追加
                ),  # Long評価
                (
                    -1,
                    p_short_chunk[i],
                    labels_short_chunk[i],
                    duration_short_chunk[i],
                    close_future_short_chunk[i],
                    should_trade_short,
                    Decimal(str(self.config.pt_multiplier_short)),  # ★追加
                    Decimal(str(self.config.sl_multiplier_short)),  # ★追加
                ),  # Short評価
            ]

            traded_in_this_step = False

            # ▼▼▼ unpackedする変数を増やす ▼▼▼
            for (
                direction_int,
                p_float,
                actual_label,
                duration_float,
                close_future_float,
                base_should_trade,
                current_pt_mult,  # ★追加
                current_sl_mult,  # ★追加
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
                        atr_ratio_float is None
                        or not np.isfinite(atr_ratio_float)
                        or atr_ratio_float
                        < self.config.min_atr_threshold  # ★修正: ATR Ratio と比較 (プロンプト⑯ 修正②)
                    ):
                        continue

                    # [FIX-4] Auto Lot 計算を extreme_risk_engine.calculate_auto_lot() と統一
                    # 旧: ハードコード乗数方式 (0.25倍/0.5倍) → 新: 証拠金上限数式

                    # ▼▼▼ 修正: 固定比率(Fixed Risk)と固定複利(Auto Lot)の分岐 ▼▼▼
                    if self.config.use_fixed_risk:
                        risk_pct_dec = Decimal(str(self.config.fixed_risk_percent))
                        max_loss_amount = current_capital * risk_pct_dec
                        # ▼▼ DECIMAL_SL_MULT を current_sl_mult に変更 ▼▼
                        sl_price_distance = (
                            Decimal(str(atr_value_float)) * current_sl_mult
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

                        # ▼▼▼ 新規追加: マージンコール（証拠金維持率）チェック ▼▼▼
                        new_total_margin = total_used_margin + margin_required_decimal
                        new_margin_level = (
                            current_capital / new_total_margin
                        ) * Decimal("100.0")

                        if new_margin_level < Decimal(
                            str(self.config.margin_call_percent)
                        ):
                            continue  # 維持率100%を下回るような過剰なエントリーは拒否

                        total_used_margin = new_total_margin
                        # ▲▲▲ ここまで追加 ▲▲▲

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
                                # ★ DECIMAL_PT_MULT を current_pt_mult に変更
                                exit_price_decimal = current_price_decimal + (
                                    Decimal(str(atr_value_float)) * current_pt_mult
                                )
                            elif (
                                # ★ 120.0 などの直書きを td_minutes_long に変更
                                valid_label == 0
                                and duration_val < (self.config.td_minutes_long - 0.1)
                            ):
                                # ★ DECIMAL_SL_MULT を current_sl_mult に変更
                                exit_price_decimal = current_price_decimal - (
                                    Decimal(str(atr_value_float)) * current_sl_mult
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
                                # ★ DECIMAL_PT_MULT を current_pt_mult に変更
                                exit_price_decimal = current_price_decimal - (
                                    Decimal(str(atr_value_float)) * current_pt_mult
                                )
                            elif (
                                # ★ 60.0 などの直書きを td_minutes_short に変更
                                valid_label == 0
                                and duration_val < (self.config.td_minutes_short - 0.1)
                            ):
                                # ★ DECIMAL_SL_MULT を current_sl_mult に変更
                                exit_price_decimal = current_price_decimal + (
                                    Decimal(str(atr_value_float)) * current_sl_mult
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

                        # ▼▼▼ 修正前 ▼▼▼
                        # if duration_float is not None and np.isfinite(duration_float):
                        #     new_exit_time = current_timestamp_int + int(duration_float * 60 * 1_000_000)
                        #     active_exit_times.append(new_exit_time)
                        #     pending_exits.append((new_exit_time, direction_int, is_sl_hit))

                        # ▼▼▼ 修正後 ▼▼▼
                        if duration_float is not None and np.isfinite(duration_float):
                            new_exit_time = current_timestamp_int + int(
                                duration_float * 60 * 1_000_000
                            )
                            active_exit_times.append(new_exit_time)
                            pending_exits.append(
                                (
                                    new_exit_time,
                                    direction_int,
                                    is_sl_hit,
                                    margin_required_decimal,
                                )
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
                                "atr_ratio": float(atr_ratio_float),  # ★追加: 相対ATR
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
            "atr_ratio": pl.Float64,  # ★追加: 相対ATR
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
        # OOFパスからM1/M2モードを判定してStrategy文字列に反映
        _oof_label = "M1" if "m1_oof" in str(self.config.oof_long_path) else "M2"
        strategy_str += f", {_oof_label}: {self.config.m2_proba_threshold}, L(PT{self.config.pt_multiplier_long}/SL{self.config.sl_multiplier_long}), S(PT{self.config.pt_multiplier_short}/SL{self.config.sl_multiplier_short})"

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
            _log_suffix = "M1" if "m1_oof" in str(self.config.oof_long_path) else "M2"
            trade_log_output_path = (
                FINAL_REPORT_PATH.parent / f"detailed_trade_log_v5_{_log_suffix}.csv"
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
                    "atr_ratio": 4,  # ★追加: 相対ATR
                    "leverage": 0,
                    "TD": 1,
                    "DD(%)": 2,
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
                    "active(L)",
                    "active(S)",
                    "margin",
                    "leverage",
                    "spread",
                    "close_price",
                    "atr_value",
                    "atr_ratio",  # ★追加: 相対ATR
                    "TD",
                    "DD(%)",
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

        text_report_path = FINAL_REPORT_PATH.with_suffix(".txt")  # _M1/.txt or _M2/.txt
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

                # ▼▼▼ 修正前 ▼▼▼
                # to_count = len(
                #     trade_log.filter(
                #         (pl.col("label") == 0)
                #         & (
                #             ((pl.col("direction") == 1) & (pl.col("TD") >= 119.9))
                #             | ((pl.col("direction") == -1) & (pl.col("TD") >= 59.9))
                #         )
                #     )
                # )

                # ▼▼▼ 修正後 ▼▼▼
                to_count = len(
                    trade_log.filter(
                        (pl.col("label") == 0)
                        & (
                            (
                                (pl.col("direction") == 1)
                                & (pl.col("TD") >= (self.config.td_minutes_long - 0.1))
                            )
                            | (
                                (pl.col("direction") == -1)
                                & (pl.col("TD") >= (self.config.td_minutes_short - 0.1))
                            )
                        )
                    )
                )
                m2_lst = trade_log["m2_proba"].to_list()
                m2_bins = {
                    "<= 0.50": sum(1 for x in m2_lst if x <= 0.50),
                    "0.50-0.55": sum(1 for x in m2_lst if 0.50 < x <= 0.55),
                    "0.55-0.60": sum(1 for x in m2_lst if 0.55 < x <= 0.60),
                    "0.60-0.65": sum(1 for x in m2_lst if 0.60 < x <= 0.65),
                    "0.65-0.70": sum(1 for x in m2_lst if 0.65 < x <= 0.70),
                    "0.70-0.75": sum(1 for x in m2_lst if 0.70 < x <= 0.75),
                    "0.75-0.80": sum(1 for x in m2_lst if 0.75 < x <= 0.80),
                    "0.80-0.85": sum(1 for x in m2_lst if 0.80 < x <= 0.85),
                    "0.85-0.90": sum(1 for x in m2_lst if 0.85 < x <= 0.90),
                    "0.90-0.95": sum(1 for x in m2_lst if 0.90 < x <= 0.95),
                    "0.95-1.00": sum(1 for x in m2_lst if x > 0.95),
                }
                # ATRは相対値(atr_ratio)で集計 ── min_atr_thresholdと同じ軸で評価するため
                # trade_logにatr_ratioカラムがなければatr_valueで代替（警告付き）
                if "atr_ratio" in trade_log.columns:
                    atr_rel_lst = trade_log["atr_ratio"].to_list()
                    atr_label = "ATR Ratio (Relative)"
                else:
                    atr_rel_lst = trade_log["atr_value"].to_list()
                    atr_label = "ATR Value (Absolute) ※atr_ratio列なし"
                    logging.warning(
                        "atr_ratio column not found in trade_log. Falling back to atr_value."
                    )
                atr_bins = {
                    "< 0.5": sum(1 for x in atr_rel_lst if x is not None and x < 0.5),
                    "0.5-0.8": sum(
                        1 for x in atr_rel_lst if x is not None and 0.5 <= x < 0.8
                    ),
                    "0.8-1.0": sum(
                        1 for x in atr_rel_lst if x is not None and 0.8 <= x < 1.0
                    ),
                    "1.0-1.2": sum(
                        1 for x in atr_rel_lst if x is not None and 1.0 <= x < 1.2
                    ),
                    "1.2-1.5": sum(
                        1 for x in atr_rel_lst if x is not None and 1.2 <= x < 1.5
                    ),
                    ">= 1.5": sum(1 for x in atr_rel_lst if x is not None and x >= 1.5),
                }

                # pnl・labelリストを帯別分析で共通利用するため先に取得
                pnl_lst = trade_log["pnl"].to_list()
                label_lst = trade_log["label"].to_list()

                # ATR絶対値帯別 勝率・PF分析（参考）
                atr_abs_lst = trade_log["atr_value"].to_list()
                atr_abs_band_defs = [
                    ("< 1.0", lambda x: x < 1.0),
                    ("1.0-2.0", lambda x: 1.0 <= x < 2.0),
                    ("2.0-3.0", lambda x: 2.0 <= x < 3.0),
                    ("3.0-5.0", lambda x: 3.0 <= x < 5.0),
                    (">= 5.0", lambda x: x >= 5.0),
                ]
                atr_abs_band_stats = {}
                for band_name, band_fn in atr_abs_band_defs:
                    idxs = [
                        i
                        for i, x in enumerate(atr_abs_lst)
                        if x is not None and band_fn(x)
                    ]
                    if not idxs:
                        atr_abs_band_stats[band_name] = None
                        continue
                    band_labels = [label_lst[i] for i in idxs]
                    band_pnls = [
                        float(pnl_lst[i]) if pnl_lst[i] is not None else 0.0
                        for i in idxs
                    ]
                    wins = [p for p in band_pnls if p > 0]
                    loses = [p for p in band_pnls if p < 0]
                    pf = sum(wins) / abs(sum(loses)) if loses else float("inf")
                    atr_abs_band_stats[band_name] = {
                        "count": len(idxs),
                        "win_rate": sum(1 for l in band_labels if l == 1)
                        / len(idxs)
                        * 100,
                        "pf": pf,
                        "avg_pnl": sum(band_pnls) / len(band_pnls),
                    }

                # ATR Ratio帯別 勝率・PF分析
                atr_band_stats = {}
                atr_band_defs = [
                    ("< 0.5", lambda x: x < 0.5),
                    ("0.5-0.8", lambda x: 0.5 <= x < 0.8),
                    ("0.8-1.0", lambda x: 0.8 <= x < 1.0),
                    ("1.0-1.2", lambda x: 1.0 <= x < 1.2),
                    ("1.2-1.5", lambda x: 1.2 <= x < 1.5),
                    (">= 1.5", lambda x: x >= 1.5),
                ]
                for band_name, band_fn in atr_band_defs:
                    idxs = [
                        i
                        for i, x in enumerate(atr_rel_lst)
                        if x is not None and band_fn(x)
                    ]
                    if not idxs:
                        atr_band_stats[band_name] = None
                        continue
                    band_labels = [label_lst[i] for i in idxs]
                    band_pnls = [
                        float(pnl_lst[i]) if pnl_lst[i] is not None else 0.0
                        for i in idxs
                    ]
                    wins = [p for p in band_pnls if p > 0]
                    loses = [p for p in band_pnls if p < 0]
                    pf = sum(wins) / abs(sum(loses)) if loses else float("inf")
                    atr_band_stats[band_name] = {
                        "count": len(idxs),
                        "win_rate": sum(1 for l in band_labels if l == 1)
                        / len(idxs)
                        * 100,
                        "pf": pf,
                        "avg_pnl": sum(band_pnls) / len(band_pnls),
                    }
            else:
                max_active_l = max_active_s = max_active_tot = count_l = count_s = (
                    to_count
                ) = 0
                avg_td_l = avg_td_s = 0.0
                m2_bins = atr_bins = {}
                atr_band_stats = {}
                atr_abs_band_stats = {}
                atr_label = "ATR Ratio (Relative)"
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

                _proba_label = (
                    "M1" if "m1_oof" in str(self.config.oof_long_path) else "M2"
                )

                # --- 全トリガー分布（OOFファイルから直接計算）---
                try:
                    _oof_long = pl.read_parquet(self.config.oof_long_path)
                    _oof_short = pl.read_parquet(self.config.oof_short_path)
                    _oof_all = pl.concat([_oof_long, _oof_short])[
                        "prediction"
                    ].to_list()
                    _total_triggers = len(_oof_all)
                    _all_bins = {
                        "<= 0.50": sum(1 for x in _oof_all if x <= 0.50),
                        "0.50-0.55": sum(1 for x in _oof_all if 0.50 < x <= 0.55),
                        "0.55-0.60": sum(1 for x in _oof_all if 0.55 < x <= 0.60),
                        "0.60-0.65": sum(1 for x in _oof_all if 0.60 < x <= 0.65),
                        "0.65-0.70": sum(1 for x in _oof_all if 0.65 < x <= 0.70),
                        "0.70-0.75": sum(1 for x in _oof_all if 0.70 < x <= 0.75),
                        "0.75-0.80": sum(1 for x in _oof_all if 0.75 < x <= 0.80),
                        "0.80-0.85": sum(1 for x in _oof_all if 0.80 < x <= 0.85),
                        "0.85-0.90": sum(1 for x in _oof_all if 0.85 < x <= 0.90),
                        "0.90-0.95": sum(1 for x in _oof_all if 0.90 < x <= 0.95),
                        "0.95-1.00": sum(1 for x in _oof_all if x > 0.95),
                    }
                    f.write(
                        "-" * 23
                        + f" {_proba_label} Proba Distribution (全トリガー / OOFベース) "
                        + "-" * 3
                        + "\n"
                    )
                    f.write("  ※ 全シグナル候補に対する生の分布（フィルター前）\n")
                    for k, v in _all_bins.items():
                        pct = (v / _total_triggers) * 100 if _total_triggers > 0 else 0
                        f.write(f"{k.ljust(15)}: {str(v).rjust(8)} ({pct:5.1f} %)\n")
                    f.write("\n")
                except Exception as _e:
                    logging.warning(f"全トリガー分布の計算に失敗しました: {_e}")

                # --- 約定トレード分布（濃縮後）---
                f.write(
                    "-" * 23
                    + f" {_proba_label} Proba Distribution (約定トレードのみ / 濃縮後) "
                    + "-" * 3
                    + "\n"
                )
                f.write("  ※ フィルター（閾値・Delta・ATR）通過後の約定トレードのみ\n")
                for k, v in m2_bins.items():
                    pct = (v / total_trades) * 100 if total_trades > 0 else 0
                    f.write(f"{k.ljust(15)}: {str(v).rjust(6)} ({pct:5.1f} %)\n")
                f.write("\n")

                f.write("-" * 23 + f" {atr_label} Distribution " + "-" * 3 + "\n")
                f.write(f"  (min_atr_threshold = {self.config.min_atr_threshold})\n")
                for k, v in atr_bins.items():
                    pct = (v / total_trades) * 100 if total_trades > 0 else 0
                    f.write(f"{k.ljust(15)}: {str(v).rjust(6)} ({pct:5.1f} %)\n")
                f.write("\n")

                f.write("-" * 23 + " ATR Ratio Band Analysis " + "-" * 12 + "\n")
                f.write(
                    f"  {'Band':<10} {'件数':>7} {'割合%':>7} {'勝率%':>7} {'PF':>7} {'平均PnL':>12}\n"
                )
                f.write("  " + "-" * 56 + "\n")
                for band_name, stats in atr_band_stats.items():
                    if stats is None:
                        f.write(f"  {band_name:<10} {'N/A':>7}\n")
                        continue
                    pct = stats["count"] / total_trades * 100 if total_trades > 0 else 0
                    f.write(
                        f"  {band_name:<10} {stats['count']:>7} {pct:>7.1f} "
                        f"{stats['win_rate']:>7.2f} {stats['pf']:>7.2f} {stats['avg_pnl']:>12.2f}\n"
                    )
                f.write("\n")

                f.write(
                    "-" * 23
                    + " ATR Value Band Analysis (参考: 絶対値) "
                    + "-" * 0
                    + "\n"
                )
                f.write(
                    f"  {'Band':<10} {'件数':>7} {'割合%':>7} {'勝率%':>7} {'PF':>7} {'平均PnL':>12}\n"
                )
                f.write("  " + "-" * 56 + "\n")
                for band_name, stats in atr_abs_band_stats.items():
                    if stats is None:
                        f.write(f"  {band_name:<10} {'N/A':>7}\n")
                        continue
                    pct = stats["count"] / total_trades * 100 if total_trades > 0 else 0
                    f.write(
                        f"  {band_name:<10} {stats['count']:>7} {pct:>7.1f} "
                        f"{stats['win_rate']:>7.2f} {stats['pf']:>7.2f} {stats['avg_pnl']:>12.2f}\n"
                    )
                f.write("\n")
                # ▲▲▲ ここまで追加 ▲▲▲

                f.write("=" * 60 + "\n")
            logging.info(f"Text performance report saved successfully.")
        except Exception as e:
            logging.error(f"Failed to save text performance report: {e}", exc_info=True)

        # ▼▼▼ この行を末尾に追加！ ▼▼▼
        return report_data


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
    # ▼▼▼ 追加: 差分(Delta)閾値用の引数 ▼▼▼
    parser.add_argument(
        "--m2-delta",
        type=float,
        default=default_config.m2_delta_threshold,
        dest="m2_delta",
        help=f"Min M2 probability delta (difference between L and S). Default: {default_config.m2_delta_threshold}",
    )
    # ▲▲▲ ここまで追加 ▲▲▲
    parser.add_argument(
        "--min-capital",
        type=float,
        default=default_config.min_capital_threshold,
        dest="min_capital",
        help=f"Min capital threshold. Default: {default_config.min_capital_threshold}",
    )
    # ★追加
    parser.add_argument(
        "--min-atr",
        type=float,
        default=default_config.min_atr_threshold,
        dest="min_atr",
        help=f"Minimum ATR threshold. Default: {default_config.min_atr_threshold}",
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

    # V5 新規追加パラメータ (Long/Short独立)
    parser.add_argument(
        "--sl-long",
        type=float,
        default=default_config.sl_multiplier_long,
        dest="sl_long",
    )
    parser.add_argument(
        "--pt-long",
        type=float,
        default=default_config.pt_multiplier_long,
        dest="pt_long",
    )
    parser.add_argument(
        "--sl-short",
        type=float,
        default=default_config.sl_multiplier_short,
        dest="sl_short",
    )
    parser.add_argument(
        "--pt-short",
        type=float,
        default=default_config.pt_multiplier_short,
        dest="pt_short",
    )
    parser.add_argument(
        "--td-long", type=float, default=default_config.td_minutes_long, dest="td_long"
    )
    parser.add_argument(
        "--td-short",
        type=float,
        default=default_config.td_minutes_short,
        dest="td_short",
    )

    parser.add_argument(
        "--oof-long",
        type=str,
        default=str(default_config.oof_long_path),
        dest="oof_long_path",
        help=f"Path to Long OOF predictions.",
    )
    parser.add_argument(
        "--oof-short",
        type=str,
        default=str(default_config.oof_short_path),
        dest="oof_short_path",
        help=f"Path to Short OOF predictions.",
    )

    args = parser.parse_args()

    config = BacktestConfig(
        auto_lot_base_capital=args.auto_lot_base_capital,
        auto_lot_size_per_base=args.auto_lot_size_per_base,
        use_fixed_risk=default_config.use_fixed_risk,
        fixed_risk_percent=default_config.fixed_risk_percent,
        base_leverage=args.base_leverage,
        m2_proba_threshold=args.m2_th,
        m2_delta_threshold=args.m2_delta,  # ★これを追加！
        test_limit_partitions=args.test_limit_partitions,
        oof_mode=True,
        min_capital_threshold=args.min_capital,
        min_atr_threshold=args.min_atr,
        value_per_pip=args.value_per_pip,
        spread_pips=args.spread_pips,
        max_positions=args.max_positions,
        prevent_simultaneous_orders=not args.allow_simultaneous,
        max_consecutive_sl=args.max_consecutive_sl,
        cooldown_minutes_after_sl=args.cooldown_minutes,
        sl_multiplier_long=args.sl_long,
        pt_multiplier_long=args.pt_long,
        sl_multiplier_short=args.sl_short,
        pt_multiplier_short=args.pt_short,
        td_minutes_long=args.td_long,
        td_minutes_short=args.td_short,
        oof_long_path=Path(args.oof_long_path),
        oof_short_path=Path(args.oof_short_path),
    )

    if config.base_leverage < 1.0:
        parser.error("--base-leverage must be >= 1.0.")
    if config.value_per_pip <= 0:
        parser.error("--value-per-pip must be greater than 0.")
    if config.spread_pips < 0:
        parser.error("--spread-pips cannot be negative.")
    if config.sl_multiplier_long <= 0 or config.sl_multiplier_short <= 0:
        parser.error("SL multipliers must be greater than 0.")

    simulator = BacktestSimulator(config)

    # =========================================================
    # 推論モード選択: M1単独 or M2 (Two-Brain)
    # =========================================================
    print("\n" + "=" * 50)
    print("  🧠 推論モードを選択してください:")
    print("    [1] M2モード (通常: Two-Brain) [デフォルト]")
    print("    [2] M1モード (実験: M1単独)")
    print("=" * 50)
    mode_ans = input("選択 [1/2, Enterでデフォルト]: ").strip()

    if mode_ans == "2":
        config.oof_long_path = S7_M1_OOF_PREDICTIONS_LONG
        config.oof_short_path = S7_M1_OOF_PREDICTIONS_SHORT
        inference_mode = "M1"
        active_cache_path = S7_BACKTEST_CACHE_M1
        oof_ref_long = S7_M1_OOF_PREDICTIONS_LONG
        oof_ref_short = S7_M1_OOF_PREDICTIONS_SHORT
        logging.info("🔬 [M1モード] M1単独OOFを使用します。")
    else:
        inference_mode = "M2"
        active_cache_path = S7_BACKTEST_CACHE_M2
        oof_ref_long = S7_M2_OOF_PREDICTIONS_LONG
        oof_ref_short = S7_M2_OOF_PREDICTIONS_SHORT
        logging.info("🧠 [M2モード] Two-Brain OOFを使用します。")

    # モードに応じて結果フォルダを生成して出力パスを設定
    import datetime as _dt

    _now_str = _dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    _risk_pct = int(config.fixed_risk_percent * 100)
    _folder_name = (
        f"{inference_mode}_{_now_str}"
        f"_Th{config.m2_proba_threshold}"
        f"_D{config.m2_delta_threshold}"
        f"_R{_risk_pct}"
    )
    _result_dir = S7_BACKTEST_SIM_RESULTS / _folder_name
    _result_dir.mkdir(parents=True, exist_ok=True)

    FINAL_REPORT_PATH = _result_dir / f"final_backtest_report_v5_{inference_mode}.json"
    EQUITY_CURVE_PATH = _result_dir / f"equity_curve_v5_{inference_mode}.png"
    logging.info(f"出力先フォルダ: {_result_dir}")

    # =========================================================
    # キャッシュ管理: モード別キャッシュファイルを使い分け
    # =========================================================
    def load_or_generate_cache() -> tuple:
        if active_cache_path.exists():
            cache_mtime = active_cache_path.stat().st_mtime
            oof_mtime = max(
                oof_ref_long.stat().st_mtime,
                oof_ref_short.stat().st_mtime,
            )
            stale = cache_mtime < oof_mtime

            print(f"\n[{inference_mode}] キャッシュが存在します: {active_cache_path}")
            if stale:
                print(
                    "  ⚠️  キャッシュがOOFより古い可能性があります。再生成を推奨します。"
                )
            print("  [y] このまま使用する")
            print("  [r] 削除して再生成する")
            ans = input("選択 [y/r]: ").strip().lower()

            if ans == "r":
                active_cache_path.unlink()
                logging.info(
                    f"[{inference_mode}] キャッシュを削除しました。再生成します..."
                )
            else:
                logging.info(f"[{inference_mode}] キャッシュを読み込んでいます...")
                with open(active_cache_path, "rb") as f:
                    data = pickle.load(f)
                logging.info(f"[{inference_mode}] キャッシュ読み込み完了。")
                return data

        logging.info(
            f"[{inference_mode}] キャッシュがありません。データを生成します..."
        )
        data = simulator.preload_data()
        logging.info(
            f"[{inference_mode}] データ生成完了。キャッシュに保存しています..."
        )
        with open(active_cache_path, "wb") as f:
            pickle.dump(data, f)
        logging.info(f"[{inference_mode}] キャッシュ保存完了: {active_cache_path}")
        return data

    preloaded_data = load_or_generate_cache()
    simulator.run(preloaded_data=preloaded_data)
