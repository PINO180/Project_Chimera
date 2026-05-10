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
    m2_delta_threshold: float = 0.30  # ★追加: LongとShortの確率の差分(Delta)閾値

    test_limit_partitions: int = 0
    oof_mode: bool = True
    min_capital_threshold: float = 1.0
    min_lot_size: float = 0.01
    min_atr_threshold: float = (
        0.9  # ★修正: ドル値(2.0) → ATR Ratio閾値(0.8) (プロンプト⑯ 修正②)
    )
    # [baseline_ATR床フィルター] 前日24h ATR平均の絶対下限 (0.0 = フィルターなし)
    # baseline_ATR = atr_value / atr_ratio (= 直近480本のATR平均 = 前日の平均ボラ水準)
    # この値未満の場合はエントリーをスキップ (前日が静かすぎる日は入らない)
    # 分析結果: 0.82でPF 19.09→19.62, 0.90でPF 19.97, 1.00でPF 20.27
    # Optunaで最適値を探索すること (backtest_simulator_run_optuna_baseline.py)
    min_baseline_atr: float = 0.0

    # [baseline_ratio相対フィルター] 前日24h baseline / 過去N日 baseline の比率下限
    # baseline_ratio = mean(ATR,1日) / mean(ATR,N日) = 昨日 vs 過去N日平均の相対ボラ比率
    # 価格水準($1800/$4600)に依存しないスケールフリーなフィルター
    # 0.0 = フィルターなし（デフォルト）
    # Optunaで探索: backtest_simulator_run_optuna_baseline_ratio.py
    min_baseline_ratio: float = 0.0
    baseline_ratio_lookback_days: int = 7  # 分母の長期ウィンドウ日数

    # [SAR: 日中季節性調整済み相対ATRフィルター]
    # SAR = 現在ATR / 過去D日間の同時刻ATR平均
    # 「UTC 13:00のATRを過去D日のUTC 13:00平均と比較」
    # → 24h混合ベースラインの問題を根本解決
    # → XAU/USDの時間帯季節性（Tokyo静/London活発）を分離評価
    # → 価格水準非依存・重複ウィンドウ問題なし
    # 0.0 = フィルターなし（デフォルト）
    # Optunaで探索: backtest_simulator_run_optuna_sar.py
    min_sar_threshold: float = 0.0
    sar_lookback_days: int = 10  # 過去何日間の同時刻平均を使うか

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
        logging.info("Pre-loading all data into memory (Optimized Single Scan)...")
        lf, partitions_df = self._prepare_data()

        # 1回のcollect()で全件メモリに乗せる（1382回のディスクスキャンを回避）
        logging.info("Executing single collect() pass on dataset. Please wait...")
        df_all = lf.with_columns(pl.col("timestamp").dt.date().alias("date")).collect()

        # [baseline_ratio相対フィルター用] 全データを時系列ソートしてbaseline_ratioを一括計算
        # baseline_atr  = atr_value / (atr_ratio + 1e-10)  = 過去1日(480本)ATR平均
        # baseline_7d   = baseline_atrのrolling_mean(N日分=N×480本)
        # baseline_ratio = baseline_atr / baseline_7d
        #   = 「昨日のボラ」 vs 「過去N日のボラ平均」の相対比率
        # ※ 全期間データに対して一括計算することでパーティション境界の問題を回避
        bars_per_day = 480  # M3: 1日=480本
        long_window = bars_per_day * self.config.baseline_ratio_lookback_days
        df_all = (
            df_all.sort("timestamp")
            .with_columns(
                [
                    (pl.col("atr_value") / (pl.col("atr_ratio") + 1e-10)).alias(
                        "_baseline_atr"
                    ),
                ]
            )
            .with_columns(
                [
                    pl.col("_baseline_atr")
                    .rolling_mean(window_size=long_window, min_samples=bars_per_day)
                    .alias("_baseline_long"),
                ]
            )
            .with_columns(
                [
                    (
                        pl.col("_baseline_atr") / (pl.col("_baseline_long") + 1e-10)
                    ).alias("baseline_ratio"),
                ]
            )
            .drop(["_baseline_atr", "_baseline_long"])
        )
        logging.info(
            f"baseline_ratio computed: lookback={self.config.baseline_ratio_lookback_days}days "
            f"({long_window}bars), null_count={df_all['baseline_ratio'].null_count()}"
        )

        # [SAR: 日中季節性調整済み相対ATRフィルター]
        # SAR = 現在ATR / 過去D日間の同時刻ATR平均
        # 設計原則（Gemini Deep Research推奨・案C）:
        #   - UTC時刻（時・分）でグループ化し「同時刻」のATR平均をベースラインにする
        #   - shift(1) で当日データを除外 → 重複ウィンドウ問題を完全回避
        #   - Tokyo静/London活発の日中季節性を分離評価
        #   - 前日が祝日で静くても当日London閾値は過去D日のLondon基準
        # min_sar_threshold=0.0 のとき計算をスキップ（後方互換）
        if self.config.min_sar_threshold > 0.0:
            logging.info(
                f"Computing SAR (Seasonality-Adjusted Ratio): "
                f"lookback={self.config.sar_lookback_days}days..."
            )
            df_all = (
                df_all.sort("timestamp")
                .with_columns(
                    [
                        # UTC時刻キー: (hour, minute) でグループ化
                        pl.col("timestamp").dt.hour().alias("_tod_h"),
                        pl.col("timestamp").dt.minute().alias("_tod_m"),
                    ]
                )
                .with_columns(
                    [
                        # 同時刻グループ内でshift(1)して当日を除外し、
                        # 過去D日分（sar_lookback_days本）の移動平均をベースラインに
                        pl.col("atr_value")
                        .shift(1)
                        .rolling_mean(
                            window_size=self.config.sar_lookback_days,
                            min_samples=max(1, self.config.sar_lookback_days // 2),
                        )
                        .over(["_tod_h", "_tod_m"])
                        .alias("_sar_baseline"),
                    ]
                )
                .with_columns(
                    [
                        # SAR = 現在ATR / 同時刻ベースライン
                        (pl.col("atr_value") / (pl.col("_sar_baseline") + 1e-10)).alias(
                            "sar"
                        ),
                    ]
                )
                .drop(["_tod_h", "_tod_m", "_sar_baseline"])
            )
            logging.info(
                f"SAR computed: null_count={df_all['sar'].null_count()}, "
                f"mean={df_all['sar'].drop_nulls().mean():.4f}"
            )
        else:
            # min_sar_threshold=0.0: SAR列を作らない（後方互換）
            pass

        preloaded_dict = {}
        partitions_to_process = partitions_df

        if self.config.test_limit_partitions > 0:
            partitions_to_process = partitions_df.head(
                self.config.test_limit_partitions
            )

        # メモリ上のDataFrameから日付ごとに切り出す（超高速）
        for row in tqdm(
            partitions_to_process.iter_rows(named=True),
            total=len(partitions_to_process),
            desc="Splitting to Dictionary",
        ):
            current_date = row["date"]
            df_chunk = df_all.filter(pl.col("date") == current_date)
            if not df_chunk.is_empty():
                preloaded_dict[current_date] = df_chunk

        del df_all
        gc.collect()

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
        # 連続SL/Loss最大値（チャンク間で引き継ぎ）
        self.max_consec_sl_long = 0
        self.max_consec_sl_short = 0
        self.max_consec_sl_total = 0
        self.max_consec_loss_total = 0

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

        # report_dataに追加統計をねじ込む（ここで初めてjson.dumpする）
        report_data["min_margin_level_pct"] = (
            float(self.min_margin_level_pct)
            if self.min_margin_level_pct != Decimal("inf")
            else 9999.0
        )
        report_data["max_consec_sl_long"] = self.max_consec_sl_long
        report_data["max_consec_sl_short"] = self.max_consec_sl_short
        report_data["max_consec_sl_total"] = self.max_consec_sl_total
        report_data["max_consec_loss_total"] = self.max_consec_loss_total

        try:
            with open(FINAL_REPORT_PATH, "w") as f:
                json.dump(report_data, f, indent=4, default=str)
            logging.info(f"Performance report saved to {FINAL_REPORT_PATH}")
        except Exception as e:
            logging.error(f"Failed to save JSON performance report: {e}")

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
        active_exit_times = []
        MAX_POSITIONS = self.config.max_positions

        # --- 連続SL/Loss・証拠金維持率トラッキング ---
        consec_sl_long_cur = 0  # 現在のLong連続SL数
        consec_sl_short_cur = 0  # 現在のShort連続SL数
        consec_loss_cur = 0  # 現在の全体連続負け数（SL+TO）
        max_consec_sl_long = 0  # Long最大連続SL
        max_consec_sl_short = 0  # Short最大連続SL
        max_consec_sl_total = 0  # 全体最大連続SL
        max_consec_loss_total = 0  # 全体最大連続負け（SL+TO）

        # --- DataFrameからのデータ抽出 (高速化のためリスト/Numpy配列化) ---
        timestamps_chunk = df_chunk["timestamp"].to_list()
        close_prices_chunk = df_chunk["close"].to_numpy()
        atr_values_chunk = df_chunk["atr_value"].to_numpy()
        atr_ratios_chunk = df_chunk[
            "atr_ratio"
        ].to_numpy()  # ★追加: ATR Ratio (プロンプト⑯ 修正②)
        # [baseline_ratio相対フィルター用] preload_dataで計算済みの列を読み込む
        # nullの場合（ウォームアップ期間）はフィルタースキップ用にNaNで埋める
        if "baseline_ratio" in df_chunk.columns:
            baseline_ratios_chunk = (
                df_chunk["baseline_ratio"].fill_null(float("nan")).to_numpy()
            )
        else:
            baseline_ratios_chunk = None

        # [SAR] preload_dataで計算済みのsar列を読み込む
        # min_sar_threshold=0.0 の場合列が存在しないためNoneで安全にスキップ
        if "sar" in df_chunk.columns:
            sar_chunk = df_chunk["sar"].fill_null(float("nan")).to_numpy()
        else:
            sar_chunk = None

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

            for exit_time, direction, is_sl, margin_used, log_entry in sorted(
                finished_positions, key=lambda x: x[0]
            ):
                # 証拠金の解放
                total_used_margin -= margin_used
                if total_used_margin < DECIMAL_ZERO:
                    total_used_margin = DECIMAL_ZERO

                if direction == 1:
                    if is_sl:
                        consecutive_sl_long += 1
                        consec_sl_long_cur += 1
                        consec_sl_short_cur = 0
                        consec_loss_cur += 1
                        if consecutive_sl_long >= self.config.max_consecutive_sl:
                            cooldown_until_long = exit_time + int(
                                self.config.cooldown_minutes_after_sl * 60 * 1_000_000
                            )
                            consecutive_sl_long = 0
                            self.cb_cooldown_long += 1
                    else:  # PT（勝ち）
                        consecutive_sl_long = 0
                        consec_sl_long_cur = 0
                        consec_sl_short_cur = 0
                        consec_loss_cur = 0
                else:
                    if is_sl:
                        consecutive_sl_short += 1
                        consec_sl_short_cur += 1
                        consec_sl_long_cur = 0
                        consec_loss_cur += 1
                        if consecutive_sl_short >= self.config.max_consecutive_sl:
                            cooldown_until_short = exit_time + int(
                                self.config.cooldown_minutes_after_sl * 60 * 1_000_000
                            )
                            consecutive_sl_short = 0
                            self.cb_cooldown_short += 1
                    else:  # PT（勝ち）
                        consecutive_sl_short = 0
                        consec_sl_long_cur = 0
                        consec_sl_short_cur = 0
                        consec_loss_cur = 0

                # 最大値更新
                max_consec_sl_long = max(max_consec_sl_long, consec_sl_long_cur)
                max_consec_sl_short = max(max_consec_sl_short, consec_sl_short_cur)
                max_consec_sl_total = max(
                    max_consec_sl_total, consec_sl_long_cur + consec_sl_short_cur
                )
                max_consec_loss_total = max(max_consec_loss_total, consec_loss_cur)

                # 決済確定後の連続SL値をlog_entryに書き込んでからトレードログに追記
                log_entry["csl_L"] = consec_sl_long_cur
                log_entry["csl_S"] = consec_sl_short_cur
                log_entry["closs"] = consec_loss_cur
                trade_log_chunk.append(log_entry)

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

                    # [baseline_ATR床フィルター] 前日24h ATR平均の絶対下限チェック
                    # baseline_atr = atr_value / atr_ratio (= 直近480本のATR平均)
                    # min_baseline_atr=0.0 のとき無効 (後方互換)
                    if self.config.min_baseline_atr > 0.0:
                        baseline_atr_float = atr_value_float / (atr_ratio_float + 1e-10)
                        if baseline_atr_float < self.config.min_baseline_atr:
                            continue

                    # [baseline_ratio相対フィルター] 昨日ボラ / 過去N日ボラ の比率チェック
                    # min_baseline_ratio=0.0 のとき無効 (後方互換)
                    if (
                        self.config.min_baseline_ratio > 0.0
                        and baseline_ratios_chunk is not None
                    ):
                        br = baseline_ratios_chunk[i]
                        if not np.isfinite(br) or br < self.config.min_baseline_ratio:
                            continue

                    # [SARフィルター] 日中季節性調整済み相対ATR
                    # SAR = 現在ATR / 過去D日の同時刻ATR平均
                    # min_sar_threshold=0.0 のとき無効 (後方互換)
                    if self.config.min_sar_threshold > 0.0 and sar_chunk is not None:
                        sar_val = sar_chunk[i]
                        if (
                            not np.isfinite(sar_val)
                            or sar_val < self.config.min_sar_threshold
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

                        # ▼▼▼ 修正後: log_entryをpending_exitsに同梱し決済時にcsl/clossを書いてからログ追記 ▼▼▼
                        traded_in_this_step = True

                        current_active_longs = sum(
                            1 for p in pending_exits if p[1] == 1
                        )
                        current_active_shorts = sum(
                            1 for p in pending_exits if p[1] == -1
                        )

                        _mg_lv = float(
                            (current_capital / total_used_margin * Decimal("100.0"))
                            if total_used_margin > DECIMAL_ZERO
                            else Decimal("9999.0")
                        )
                        _log_entry = {
                            "timestamp": current_timestamp,
                            "pnl": pnl,
                            "balance": current_capital,
                            "m2_proba": float(p_float),
                            "direction": int(direction_int),
                            "label": int(valid_label),
                            "lot_size": float(final_lot_size_decimal),
                            "atr_value": float(atr_value_float),
                            "atr_ratio": float(atr_ratio_float),
                            "leverage": float(effective_leverage_decimal),
                            "margin": margin_required_decimal,
                            "spread": spread_cost_decimal,
                            "close_price": current_price_decimal,
                            "aL": int(current_active_longs),
                            "aS": int(current_active_shorts),
                            "TD": float(duration_val),
                            "DD(%)": float(current_dd_pct),
                            "mg_lv%": _mg_lv,
                            "csl_L": 0,  # 決済時に上書き
                            "csl_S": 0,  # 決済時に上書き
                            "closs": 0,  # 決済時に上書き
                        }

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
                                    _log_entry,  # ← log_entryを同梱
                                )
                            )
                        else:
                            # duration不明の場合はcsl=0のままログ追記
                            trade_log_chunk.append(_log_entry)

            # 資本の時系列記録
            equity_values_chunk.append(current_capital)

        self._current_capital = current_capital
        # チャンク内の最大値をselfに反映（複数チャンクをまたいで最大値を保持）
        self.max_consec_sl_long = max(self.max_consec_sl_long, max_consec_sl_long)
        self.max_consec_sl_short = max(self.max_consec_sl_short, max_consec_sl_short)
        self.max_consec_sl_total = max(self.max_consec_sl_total, max_consec_sl_total)
        self.max_consec_loss_total = max(
            self.max_consec_loss_total, max_consec_loss_total
        )

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
            "balance": pl.Object,
            "m2_proba": pl.Float64,
            "direction": pl.Int8,
            "label": pl.Int64,
            "lot_size": pl.Float64,
            "atr_value": pl.Float64,
            "atr_ratio": pl.Float64,
            "leverage": pl.Float32,
            "margin": pl.Object,
            "spread": pl.Object,
            "close_price": pl.Object,
            "aL": pl.Int32,
            "aS": pl.Int32,
            "TD": pl.Float64,
            "DD(%)": pl.Float64,
            "mg_lv%": pl.Float64,  # 証拠金維持率(%)
            "csl_L": pl.Int32,  # Long連続SL（決済後）
            "csl_S": pl.Int32,  # Short連続SL（決済後）
            "closs": pl.Int32,  # 全体連続負け（SL+TO）
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
                    "atr_ratio": 4,
                    "leverage": 0,
                    "TD": 1,
                    "DD(%)": 2,
                    "mg_lv%": 1,
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
                    "aL",
                    "aS",
                    "margin",
                    "leverage",
                    "spread",
                    "close_price",
                    "atr_value",
                    "atr_ratio",
                    "TD",
                    "DD(%)",
                    "mg_lv%",
                    "csl_L",
                    "csl_S",
                    "closs",
                ]

                available_columns_final = [
                    col
                    for col in desired_columns_final
                    if col in temp_log_formatted.columns
                ]
                trade_log_final_csv = temp_log_formatted.select(available_columns_final)

                # timestampをUTC→JSTに変換（+9時間）して上書き
                trade_log_final_csv = trade_log_final_csv.with_columns(
                    pl.col("timestamp")
                    .str.to_datetime("%Y-%m-%d %H:%M:%S", strict=False)
                    .dt.offset_by("9h")
                    .alias("timestamp")
                )

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
                max_active_l = trade_log["aL"].max()
                max_active_s = trade_log["aS"].max()
                max_active_tot = (trade_log["aL"] + trade_log["aS"]).max()

                l_trades = trade_log.filter(pl.col("direction") == 1)
                s_trades = trade_log.filter(pl.col("direction") == -1)
                count_l = len(l_trades)
                count_s = len(s_trades)
                avg_td_l = l_trades["TD"].mean() if count_l > 0 else 0.0
                avg_td_s = s_trades["TD"].mean() if count_s > 0 else 0.0

                # 方向別 勝率・PF
                def _win_rate_pf(trades):
                    if len(trades) == 0:
                        return 0.0, 0.0
                    pnls = [float(p) for p in trades["pnl"].to_list() if p is not None]
                    wins = [p for p in pnls if p > 0]
                    losses = [p for p in pnls if p < 0]
                    wr = len(wins) / len(pnls) * 100 if pnls else 0.0
                    pf = sum(wins) / abs(sum(losses)) if losses else float("inf")
                    return wr, pf

                wr_l, pf_l = _win_rate_pf(l_trades)
                wr_s, pf_s = _win_rate_pf(s_trades)

                # 連続SL最大値（selfから取得）
                max_csl_long = self.max_consec_sl_long
                max_csl_short = self.max_consec_sl_short
                max_csl_total = self.max_consec_sl_total
                max_closs_total = self.max_consec_loss_total

                # 証拠金維持率最低値（selfから取得）
                min_mg_lv = (
                    float(self.min_margin_level_pct)
                    if self.min_margin_level_pct != Decimal("inf")
                    else None
                )

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
                wr_l = wr_s = pf_l = pf_s = 0.0
                max_csl_long = max_csl_short = max_csl_total = max_closs_total = 0
                min_mg_lv = None
                m2_bins = atr_bins = {}
                atr_band_stats = {}
                atr_abs_band_stats = {}
                atr_label = "ATR Ratio (Relative)"
            # ▲▲▲ ここまで追加 ▲▲▲

            # --- 時間帯・曜日分析の事前計算（CSVとTXT両方で使用）---
            def _session(h):
                """JST時間帯でセッション分類"""
                if 9 <= h < 16:
                    return "Tokyo"
                elif 16 <= h < 21:
                    return "London"
                elif h >= 21 or h < 1:
                    return "Overlap"
                elif 1 <= h < 6:
                    return "NY"
                else:
                    return "Oceania"  # 6-9 JST

            def _band_stats(indices, pnl_lst, label_lst):
                if not indices:
                    return None
                p = [float(pnl_lst[i]) for i in indices if pnl_lst[i] is not None]
                wins = [x for x in p if x > 0]
                losses = [x for x in p if x < 0]
                wr = len(wins) / len(p) * 100 if p else 0.0
                pf = sum(wins) / abs(sum(losses)) if losses else float("inf")
                avg = sum(p) / len(p) if p else 0.0
                tot = sum(p)
                return {
                    "count": len(p),
                    "win_rate": wr,
                    "pf": pf,
                    "avg_pnl": avg,
                    "total_pnl": tot,
                }

            hourly_stats = {}
            weekday_stats = {}
            hxatr_stats = {}

            if not trade_log.is_empty():
                ts_list2 = trade_log["timestamp"].to_list()
                pnl_lst2 = trade_log["pnl"].to_list()
                lbl_lst2 = trade_log["label"].to_list()
                atr_lst2 = (
                    trade_log["atr_ratio"].to_list()
                    if "atr_ratio" in trade_log.columns
                    else [1.0] * len(ts_list2)
                )

                atr_bands = [
                    ("< 0.5", lambda x: x < 0.5),
                    ("0.5-0.8", lambda x: 0.5 <= x < 0.8),
                    ("0.8-1.0", lambda x: 0.8 <= x < 1.0),
                    ("1.0-1.2", lambda x: 1.0 <= x < 1.2),
                    ("1.2-1.5", lambda x: 1.2 <= x < 1.5),
                    (">= 1.5", lambda x: x >= 1.5),
                ]

                hour_idx = {}
                weekday_idx = {}
                hxatr_idx = {}

                for i, ts in enumerate(ts_list2):
                    try:
                        # UTC→JST変換（+9時間）
                        h_jst = (ts.hour + 9) % 24
                        # 日またぎ考慮: UTC時刻+9が24を超えた場合は翌日
                        day_offset = 1 if (ts.hour + 9) >= 24 else 0
                        wd_jst = (ts.weekday() + day_offset) % 7
                    except Exception:
                        continue
                    hour_idx.setdefault(h_jst, []).append(i)
                    weekday_idx.setdefault(wd_jst, []).append(i)
                    ar = atr_lst2[i]
                    if ar is not None and isinstance(ar, (int, float)):
                        for band_name, band_fn in atr_bands:
                            if band_fn(float(ar)):
                                hxatr_idx.setdefault((h_jst, band_name), []).append(i)
                                break

                for h in range(24):
                    hourly_stats[h] = _band_stats(
                        hour_idx.get(h, []), pnl_lst2, lbl_lst2
                    )
                for wd in range(7):
                    weekday_stats[wd] = _band_stats(
                        weekday_idx.get(wd, []), pnl_lst2, lbl_lst2
                    )
                for h in range(24):
                    for band_name, _ in atr_bands:
                        hxatr_stats[(h, band_name)] = _band_stats(
                            hxatr_idx.get((h, band_name), []), pnl_lst2, lbl_lst2
                        )

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
                f.write(f"Maximal Drawdown:\t{abs(max_dd):,.2f} %\n")
                f.write(
                    f"Min Margin Level:\t{min_mg_lv:,.1f} %"
                    if min_mg_lv is not None
                    else "Min Margin Level:\tN/A (no open positions)"
                )
                f.write("\n\n")
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

                f.write("-" * 25 + " Direction Analysis " + "-" * 25 + "\n")
                f.write(f"{'':20}{'Long':>12}{'Short':>12}\n")
                f.write(f"{'Trade Count':20}{count_l:>12}{count_s:>12}\n")
                f.write(f"{'Win Rate (%)':20}{wr_l:>11.2f}%{wr_s:>11.2f}%\n")
                pf_l_str = f"{pf_l:.2f}" if pf_l != float("inf") else "inf"
                pf_s_str = f"{pf_s:.2f}" if pf_s != float("inf") else "inf"
                f.write(f"{'Profit Factor':20}{pf_l_str:>12}{pf_s_str:>12}\n\n")

                f.write("-" * 25 + " Consecutive Losses " + "-" * 25 + "\n")
                f.write(f"Max Consec SL (Long):\t{max_csl_long}\n")
                f.write(f"Max Consec SL (Short):\t{max_csl_short}\n")
                f.write(f"Max Consec SL (Total):\t{max_csl_total}\n")
                f.write(f"Max Consec Loss (SL+TO):\t{max_closs_total}\n\n")

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

                # --- M1 全トリガー分布（参考値・M2モード時のみ追加表示）---
                if "m1_oof" not in str(self.config.oof_long_path):
                    try:
                        _m1_oof_long = pl.read_parquet(S7_M1_OOF_PREDICTIONS_LONG)
                        _m1_oof_short = pl.read_parquet(S7_M1_OOF_PREDICTIONS_SHORT)
                        _m1_all = pl.concat([_m1_oof_long, _m1_oof_short])[
                            "prediction"
                        ].to_list()
                        _m1_total = len(_m1_all)
                        _m1_bins = {
                            "<= 0.50": sum(1 for x in _m1_all if x <= 0.50),
                            "0.50-0.55": sum(1 for x in _m1_all if 0.50 < x <= 0.55),
                            "0.55-0.60": sum(1 for x in _m1_all if 0.55 < x <= 0.60),
                            "0.60-0.65": sum(1 for x in _m1_all if 0.60 < x <= 0.65),
                            "0.65-0.70": sum(1 for x in _m1_all if 0.65 < x <= 0.70),
                            "0.70-0.75": sum(1 for x in _m1_all if 0.70 < x <= 0.75),
                            "0.75-0.80": sum(1 for x in _m1_all if 0.75 < x <= 0.80),
                            "0.80-0.85": sum(1 for x in _m1_all if 0.80 < x <= 0.85),
                            "0.85-0.90": sum(1 for x in _m1_all if 0.85 < x <= 0.90),
                            "0.90-0.95": sum(1 for x in _m1_all if 0.90 < x <= 0.95),
                            "0.95-1.00": sum(1 for x in _m1_all if x > 0.95),
                        }
                        f.write(
                            "-" * 23
                            + " M1 Proba Distribution (参考 / OOFベース) "
                            + "-" * 3
                            + "\n"
                        )
                        f.write("  ※ M1の生の出力分布（M2への入力前）\n")
                        for k, v in _m1_bins.items():
                            pct = (v / _m1_total) * 100 if _m1_total > 0 else 0
                            f.write(
                                f"{k.ljust(15)}: {str(v).rjust(8)} ({pct:5.1f} %)\n"
                            )
                        f.write("\n")
                    except Exception as _e:
                        logging.warning(f"M1分布の計算に失敗しました: {_e}")

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

                # --- TXTに時間帯・曜日サマリーを追記 ---
                wd_names = [
                    "Monday",
                    "Tuesday",
                    "Wednesday",
                    "Thursday",
                    "Friday",
                    "Saturday",
                    "Sunday",
                ]
                session_order = ["Tokyo", "London", "Overlap", "NY", "Oceania"]

                # セッション別集計
                session_stats = {}
                for h, st in hourly_stats.items():
                    if st is None:
                        continue
                    s = _session(h)
                    if s not in session_stats:
                        session_stats[s] = {
                            "count": 0,
                            "wins": 0,
                            "pnl_wins": 0.0,
                            "pnl_losses": 0.0,
                        }
                    session_stats[s]["count"] += st["count"]
                    w = int(st["count"] * st["win_rate"] / 100)
                    session_stats[s]["wins"] += w
                    if st["pf"] != float("inf"):
                        l_cnt = st["count"] - w
                        if l_cnt > 0:
                            avg_loss = st["avg_pnl"] - st["win_rate"] / 100 * (
                                st["avg_pnl"] * st["pf"] / (1 + st["pf"])
                                if st["pf"] > 0
                                else 0
                            )
                    session_stats[s]["pnl_wins"] += (
                        st["total_pnl"] if st["total_pnl"] > 0 else 0
                    )
                    session_stats[s]["pnl_losses"] += (
                        st["total_pnl"] if st["total_pnl"] < 0 else 0
                    )

                f.write("-" * 22 + " Session Summary " + "-" * 21 + "\n")
                f.write(
                    f"  {'Session':<10}{'Trades':>8}{'WinRate%':>10}{'PF':>8}{'AvgPnL':>14}{'TotalPnL':>16}\n"
                )
                f.write("  " + "-" * 66 + "\n")
                for s in session_order:
                    hs = [
                        st
                        for h, st in hourly_stats.items()
                        if _session(h) == s and st is not None
                    ]
                    if not hs:
                        continue
                    tc = sum(x["count"] for x in hs)
                    tw = sum(int(x["count"] * x["win_rate"] / 100) for x in hs)
                    wr = tw / tc * 100 if tc > 0 else 0
                    all_pnl = [x["total_pnl"] for x in hs]
                    pw = sum(x for x in all_pnl if x > 0)
                    pl_neg = sum(x for x in all_pnl if x < 0)
                    pf_s = pw / abs(pl_neg) if pl_neg != 0 else float("inf")
                    pf_str = f"{pf_s:.2f}" if pf_s != float("inf") else "inf"
                    tot = sum(all_pnl)
                    avg = tot / tc if tc > 0 else 0
                    f.write(
                        f"  {s:<10}{tc:>8}{wr:>9.2f}%{pf_str:>8}{avg:>14,.0f}{tot:>16,.0f}\n"
                    )
                f.write("\n")

                f.write("-" * 22 + " Weekday Summary " + "-" * 21 + "\n")
                f.write(
                    f"  {'Weekday':<12}{'Trades':>8}{'WinRate%':>10}{'PF':>8}{'AvgPnL':>14}{'TotalPnL':>16}\n"
                )
                f.write("  " + "-" * 66 + "\n")
                for wd in range(7):
                    st = weekday_stats.get(wd)
                    if st is None:
                        continue
                    pf_str = f"{st['pf']:.2f}" if st["pf"] != float("inf") else "inf"
                    f.write(
                        f"  {wd_names[wd]:<12}{st['count']:>8}{st['win_rate']:>9.2f}%{pf_str:>8}{st['avg_pnl']:>14,.0f}{st['total_pnl']:>16,.0f}\n"
                    )
                f.write("\n")

                # ベスト・ワースト時間帯
                valid_hours = [
                    (h, st)
                    for h, st in hourly_stats.items()
                    if st and st["count"] >= 10
                ]
                if valid_hours:
                    best_wr = max(valid_hours, key=lambda x: x[1]["win_rate"])
                    worst_wr = min(valid_hours, key=lambda x: x[1]["win_rate"])
                    best_pf = max(
                        valid_hours,
                        key=lambda x: x[1]["pf"] if x[1]["pf"] != float("inf") else 0,
                    )
                    f.write("-" * 22 + " Hourly Highlights " + "-" * 19 + "\n")
                    f.write(
                        f"  Best  Win Rate : {best_wr[0]:02d}:00 UTC ({_session(best_wr[0])}) → {best_wr[1]['win_rate']:.2f}%\n"
                    )
                    f.write(
                        f"  Worst Win Rate : {worst_wr[0]:02d}:00 UTC ({_session(worst_wr[0])}) → {worst_wr[1]['win_rate']:.2f}%\n"
                    )
                    f.write(
                        f"  Best  PF       : {best_pf[0]:02d}:00 UTC ({_session(best_pf[0])}) → PF {best_pf[1]['pf']:.2f}\n\n"
                    )

                f.write("=" * 60 + "\n")
            logging.info(f"Text performance report saved successfully.")
        except Exception as e:
            logging.error(f"Failed to save text performance report: {e}", exc_info=True)

        # --- 月別・年別リターン CSV 出力 ---
        try:
            if not trade_log.is_empty():
                monthly_path = FINAL_REPORT_PATH.parent / (
                    FINAL_REPORT_PATH.stem + "_monthly_breakdown.csv"
                )
                # pnl/balance はObject型のため先にFloat64へ変換
                _tl = trade_log.with_columns(
                    [
                        pl.col("pnl")
                        .map_elements(
                            lambda x: float(x) if x is not None else None,
                            return_dtype=pl.Float64,
                        )
                        .alias("pnl_f"),
                        pl.col("label").cast(pl.Int32).alias("label_i"),
                        pl.col("timestamp").dt.year().alias("year"),
                        pl.col("timestamp").dt.month().alias("month"),
                    ]
                )

                monthly_rows = []
                for (yr, mo), grp in _tl.group_by(
                    ["year", "month"], maintain_order=False
                ):
                    pnls = grp["pnl_f"].to_list()
                    labels = grp["label_i"].to_list()
                    wins = [p for p in pnls if p > 0]
                    losses = [p for p in pnls if p < 0]
                    wr = len(wins) / len(pnls) * 100 if pnls else 0.0
                    pf = sum(wins) / abs(sum(losses)) if losses else float("inf")
                    tot_pnl = sum(pnls)
                    dd_vals = grp["DD(%)"].to_list()
                    max_dd = min(dd_vals) if dd_vals else 0.0
                    monthly_rows.append(
                        {
                            "year": yr,
                            "month": mo,
                            "trades": len(pnls),
                            "win_rate_%": round(wr, 2),
                            "profit_factor": round(pf, 3)
                            if pf != float("inf")
                            else None,
                            "total_pnl": round(tot_pnl, 2),
                            "max_dd_%": round(max_dd, 2),
                        }
                    )

                monthly_rows.sort(key=lambda r: (r["year"], r["month"]))

                # 年計行を挿入
                output_rows = []
                cur_year = None
                year_buf = []
                for row in monthly_rows:
                    if cur_year is not None and row["year"] != cur_year:
                        # 年計
                        yr_pnls_w = [
                            r["total_pnl"] for r in year_buf if r["total_pnl"] > 0
                        ]
                        yr_pnls_l = [
                            r["total_pnl"] for r in year_buf if r["total_pnl"] < 0
                        ]
                        yr_trades = sum(r["trades"] for r in year_buf)
                        yr_pf = (
                            sum(yr_pnls_w) / abs(sum(yr_pnls_l)) if yr_pnls_l else None
                        )
                        output_rows.append(
                            {
                                "year": cur_year,
                                "month": "TOTAL",
                                "trades": yr_trades,
                                "win_rate_%": "",
                                "profit_factor": round(yr_pf, 3) if yr_pf else None,
                                "total_pnl": round(
                                    sum(r["total_pnl"] for r in year_buf), 2
                                ),
                                "max_dd_%": round(
                                    min(r["max_dd_%"] for r in year_buf), 2
                                ),
                            }
                        )
                        output_rows.append({})  # 空行
                        year_buf = []
                    cur_year = row["year"]
                    year_buf.append(row)
                    output_rows.append(row)

                # 最終年の年計
                if year_buf:
                    yr_pnls_w = [r["total_pnl"] for r in year_buf if r["total_pnl"] > 0]
                    yr_pnls_l = [r["total_pnl"] for r in year_buf if r["total_pnl"] < 0]
                    yr_pf = sum(yr_pnls_w) / abs(sum(yr_pnls_l)) if yr_pnls_l else None
                    output_rows.append(
                        {
                            "year": cur_year,
                            "month": "TOTAL",
                            "trades": sum(r["trades"] for r in year_buf),
                            "win_rate_%": "",
                            "profit_factor": round(yr_pf, 3) if yr_pf else None,
                            "total_pnl": round(
                                sum(r["total_pnl"] for r in year_buf), 2
                            ),
                            "max_dd_%": round(min(r["max_dd_%"] for r in year_buf), 2),
                        }
                    )

                import csv as _csv

                fieldnames = [
                    "year",
                    "month",
                    "trades",
                    "win_rate_%",
                    "profit_factor",
                    "total_pnl",
                    "max_dd_%",
                ]
                with open(monthly_path, "w", newline="", encoding="utf-8") as f:
                    writer = _csv.DictWriter(f, fieldnames=fieldnames)
                    writer.writeheader()
                    for row in output_rows:
                        writer.writerow({k: row.get(k, "") for k in fieldnames})
                logging.info(f"Monthly breakdown CSV saved to {monthly_path}")
        except Exception as e:
            logging.error(f"Failed to save monthly breakdown CSV: {e}", exc_info=True)

        # --- 時間帯別分析 CSV ---
        try:
            if not trade_log.is_empty() and hourly_stats:
                import csv as _csv

                hourly_path = FINAL_REPORT_PATH.parent / (
                    FINAL_REPORT_PATH.stem + "_hourly_analysis.csv"
                )
                fields = [
                    "hour_jst",
                    "session",
                    "trades",
                    "win_rate_%",
                    "profit_factor",
                    "avg_pnl",
                    "total_pnl",
                ]
                with open(hourly_path, "w", newline="", encoding="utf-8") as f:
                    w = _csv.DictWriter(f, fieldnames=fields)
                    w.writeheader()
                    for h in range(24):
                        st = hourly_stats.get(h)
                        if st is None:
                            w.writerow(
                                {
                                    "hour_jst": h,
                                    "session": _session(h),
                                    "trades": 0,
                                    "win_rate_%": "",
                                    "profit_factor": "",
                                    "avg_pnl": "",
                                    "total_pnl": "",
                                }
                            )
                        else:
                            pf_val = (
                                round(st["pf"], 3) if st["pf"] != float("inf") else None
                            )
                            w.writerow(
                                {
                                    "hour_jst": h,
                                    "session": _session(h),
                                    "trades": st["count"],
                                    "win_rate_%": round(st["win_rate"], 2),
                                    "profit_factor": pf_val,
                                    "avg_pnl": round(st["avg_pnl"], 2),
                                    "total_pnl": round(st["total_pnl"], 2),
                                }
                            )
                logging.info(f"Hourly analysis CSV saved to {hourly_path}")
        except Exception as e:
            logging.error(f"Failed to save hourly analysis CSV: {e}", exc_info=True)

        # --- 曜日別分析 CSV ---
        try:
            if not trade_log.is_empty() and weekday_stats:
                import csv as _csv

                wd_path = FINAL_REPORT_PATH.parent / (
                    FINAL_REPORT_PATH.stem + "_weekday_analysis.csv"
                )
                wd_names_csv = [
                    "Monday",
                    "Tuesday",
                    "Wednesday",
                    "Thursday",
                    "Friday",
                    "Saturday",
                    "Sunday",
                ]
                fields = [
                    "weekday",
                    "weekday_name",
                    "trades",
                    "win_rate_%",
                    "profit_factor",
                    "avg_pnl",
                    "total_pnl",
                ]
                with open(wd_path, "w", newline="", encoding="utf-8") as f:
                    w = _csv.DictWriter(f, fieldnames=fields)
                    w.writeheader()
                    for wd in range(7):
                        st = weekday_stats.get(wd)
                        if st is None:
                            w.writerow(
                                {
                                    "weekday": wd,
                                    "weekday_name": wd_names_csv[wd],
                                    "trades": 0,
                                    "win_rate_%": "",
                                    "profit_factor": "",
                                    "avg_pnl": "",
                                    "total_pnl": "",
                                }
                            )
                        else:
                            pf_val = (
                                round(st["pf"], 3) if st["pf"] != float("inf") else None
                            )
                            w.writerow(
                                {
                                    "weekday": wd,
                                    "weekday_name": wd_names_csv[wd],
                                    "trades": st["count"],
                                    "win_rate_%": round(st["win_rate"], 2),
                                    "profit_factor": pf_val,
                                    "avg_pnl": round(st["avg_pnl"], 2),
                                    "total_pnl": round(st["total_pnl"], 2),
                                }
                            )
                logging.info(f"Weekday analysis CSV saved to {wd_path}")
        except Exception as e:
            logging.error(f"Failed to save weekday analysis CSV: {e}", exc_info=True)

        # --- 時間帯×ATR帯 分析 CSV ---
        try:
            if not trade_log.is_empty() and hxatr_stats:
                import csv as _csv

                hxatr_path = FINAL_REPORT_PATH.parent / (
                    FINAL_REPORT_PATH.stem + "_hour_x_atr_analysis.csv"
                )
                atr_band_names = [
                    "< 0.5",
                    "0.5-0.8",
                    "0.8-1.0",
                    "1.0-1.2",
                    "1.2-1.5",
                    ">= 1.5",
                ]
                fields = [
                    "hour_jst",
                    "session",
                    "atr_band",
                    "trades",
                    "win_rate_%",
                    "profit_factor",
                    "avg_pnl",
                    "total_pnl",
                ]
                with open(hxatr_path, "w", newline="", encoding="utf-8") as f:
                    w = _csv.DictWriter(f, fieldnames=fields)
                    w.writeheader()
                    for h in range(24):
                        for band_name in atr_band_names:
                            st = hxatr_stats.get((h, band_name))
                            if st is None or st["count"] == 0:
                                continue
                            pf_val = (
                                round(st["pf"], 3) if st["pf"] != float("inf") else None
                            )
                            w.writerow(
                                {
                                    "hour_jst": h,
                                    "session": _session(h),
                                    "atr_band": band_name,
                                    "trades": st["count"],
                                    "win_rate_%": round(st["win_rate"], 2),
                                    "profit_factor": pf_val,
                                    "avg_pnl": round(st["avg_pnl"], 2),
                                    "total_pnl": round(st["total_pnl"], 2),
                                }
                            )
                logging.info(f"Hour x ATR analysis CSV saved to {hxatr_path}")
        except Exception as e:
            logging.error(f"Failed to save hour x ATR analysis CSV: {e}", exc_info=True)

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
        active_cache_path.parent.mkdir(parents=True, exist_ok=True)  # ← これを追加
        with open(active_cache_path, "wb") as f:
            pickle.dump(data, f)
        logging.info(f"[{inference_mode}] キャッシュ保存完了: {active_cache_path}")
        return data

    preloaded_data = load_or_generate_cache()
    simulator.run(preloaded_data=preloaded_data)
