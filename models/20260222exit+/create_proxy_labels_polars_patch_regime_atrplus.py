# /workspace/models/create_proxy_labels_polars_patch_regime.py
# [フェーズ3: 最終ラベリングスクリプト]
# - ATRレジーム（V4ルールブック）をハードコード
# - [修正済み] 変動ATRフィルター (0.28%) を適用
# - [修正済み] 行増殖バグ (64x) を GroupBy + Coalesce で修正

import sys
from pathlib import Path
import warnings
import argparse
import shutil
from dataclasses import dataclass, field
import logging
from typing import List, Dict, Any, Optional, Tuple
import polars as pl
from tqdm import tqdm
import re
import gc
import datetime as dt

# --- ▼▼▼ Numba移行のためのインポート追加 ▼▼▼ ---
import numpy as np

try:
    from numba import njit, prange
    from numba.core.errors import NumbaPerformanceWarning

    warnings.filterwarnings("ignore", category=NumbaPerformanceWarning)
    NUMBA_AVAILABLE = True
except ImportError:
    logging.warning(
        "Numba not found. Labeling performance will be significantly degraded."
    )
    NUMBA_AVAILABLE = False
# --- ▲▲▲ インポート追加ここまで ▲▲▲ ---


# --- Project Path Setup ---
project_root = Path(__file__).resolve().parents[1]
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))


# --- Blueprint Imports ---
from blueprint import S5_NEUTRALIZED_ALPHA_SET, S2_FEATURES_FIXED, S6_LABELED_DATASET

# --- Logging Setup ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(processName)s - %(message)s",
)
warnings.filterwarnings("ignore", category=UserWarning, module="polars")
try:
    from polars.exceptions import PolarsUsePyarrowWarning

    warnings.filterwarnings("ignore", category=PolarsUsePyarrowWarning)
except ImportError:
    pass


# --- ▼▼▼ [フェーズ3 改造] Project Forge V5 Logic (Optimized) ▼▼▼ ---

# 1. Trigger (Accel)
# Engine 1E: window=[50, 100, 200]. Use 50 for max speed.
TRIGGER_HILBERT_COL = "e1e_hilbert_amplitude_50"

# 【修正】期間を「50」に戻す
# 理由: Window 20では平均が追従しすぎて乖離（倍率）が出ないため。
# 50にすることで、急騰時に「平均との差」が出やすくなります。
TRIGGER_ROLLING_WINDOW = 50

# 【修正】倍率を「1.08」に設定
# 理由: 1.0(全通)と1.2(遮断)の間にある「有効なシグナル」を拾うため。
# 平均より8%高いエネルギーは、ノイズではなく「明確な動き」の初動です。
TRIGGER_AMP_MULTIPLIER = 1.08

LOCKOUT_RULES = {
    # 歪度: -5.0 -> -2.0 (通常の調整は許容、クラッシュのみ弾く)
    "skewness": {"col": "e1a_statistical_skewness_20", "thresh": -2.0, "op": "lt"},
    # 尖度: 20.0 -> 12.0 (ファットテールを許容しつつ、異常値は弾く)
    "kurtosis": {"col": "e1a_statistical_kurtosis_20", "thresh": 15.0, "op": "gt"},
    # 位相安定性: 0.1 -> 0.25 (カオスすぎる相場は避ける)
    "phase_stability": {
        "col": "e1e_hilbert_phase_stability_50",
        "thresh": 0.25,
        "op": "lt",
    },
    # 実体比率: 0.98 -> 0.90 (実体が大きすぎる＝落ちるナイフを回避)
    "body_ratio": {"col": None, "thresh": 0.90, "op": "gt"},
}

# ★追加: 動的イグジット（緊急撤退）専用の閾値
# 優先案A: マージン拡大（Whipsaw抑制）
EMERGENCY_EXIT_RULES = {
    "skewness": {"col": "e1a_statistical_skewness_20", "thresh": -3.5, "op": "lt"},
    "kurtosis": {"col": "e1a_statistical_kurtosis_50", "thresh": 30.0, "op": "gt"},
    "phase_stability": {
        "col": "e1e_hilbert_phase_stability_50",
        "thresh": 0.15,
        "op": "lt",
    },
    # body_ratio はティックデータにOpen/Closeが存在しないためNumbaバリアから除外
}
# 3. Barrier Scaling
BARRIER_ATR_COL = "e1c_atr_21"

# R4レジーム設定 (ターゲットのみ継承)
REGIME_RULE_R4 = {
    "pt": 1.0,
    "sl": 5.0,
    "td": "1200m",
    "payoff": 1.0 / 5.0,
}
# --- ▲▲▲ 改造ここまで ▲▲▲ ---


# --- Configuration (V7の柔軟な設定) ---
@dataclass
class ProxyLabelConfig:
    """Config for creating a context-adaptive, labeled subset for proxy model training."""

    input_dir: Path = S5_NEUTRALIZED_ALPHA_SET
    price_data_source_dir: Path = S2_FEATURES_FIXED
    output_dir: Path = S6_LABELED_DATASET

    # Filtering options
    filter_mode: str = "year"  # 'year', 'month', 'all'
    filter_year: Optional[int] = 2023  # Used for 'year' mode and 'month' mode
    filter_month: Optional[int] = (
        None  # Used only for 'month' mode (requires filter_year)
    )
    resume: bool = True
    execution_start_time: str = field(
        default_factory=lambda: dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    )

    def get_filter_description(self) -> str:
        """Returns a human-readable description of the filter being applied."""
        if self.filter_mode == "year" and self.filter_year is not None:
            return f"Year = {self.filter_year}"
        elif (
            self.filter_mode == "month"
            and self.filter_year is not None
            and self.filter_month is not None
        ):
            month_str = f"{self.filter_month:02d}"
            return f"Year/Month = {self.filter_year}/{month_str}"
        elif self.filter_mode == "all":
            return "All Time"
        else:
            return f"Invalid or Incomplete Filter ({self.filter_mode}, Year: {self.filter_year}, Month: {self.filter_month})"


# --- ▼▼▼ Numbaヘルパー関数 (変更なし) ▼▼▼ ---
def _njit_if_available(func):
    """Applies @njit decorator only if Numba is available."""
    if NUMBA_AVAILABLE:
        return njit(func, parallel=True, fastmath=True, cache=True)
    else:
        return func


@_njit_if_available
def _numba_find_hits(
    bets_t0: np.ndarray,
    bets_t1_max: np.ndarray,
    bets_pt_barrier: np.ndarray,
    bets_sl_barrier: np.ndarray,
    ticks_ts: np.ndarray,
    ticks_high: np.ndarray,
    ticks_low: np.ndarray,
    # ★追加: 第4バリア用の特徴量配列
    ticks_skewness: np.ndarray,
    ticks_kurtosis: np.ndarray,
    ticks_phase: np.ndarray,
    ticks_vol_surge: np.ndarray,  # ★追加: ATR急増フラグ
    exit_skew_thresh: float,
    exit_kurt_thresh: float,
    exit_phase_thresh: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Numba JIT compiled function to find barrier hits. (Quadruple Barrier)"""
    n_bets = len(bets_t0)
    n_ticks = len(ticks_ts)
    out_first_pt_time = np.zeros(n_bets, dtype=np.int64)
    out_first_sl_time = np.zeros(n_bets, dtype=np.int64)
    out_first_ex_time = np.zeros(n_bets, dtype=np.int64)  # ★追加: 緊急イグジット時刻
    out_first_ex_reason = np.zeros(n_bets, dtype=np.int8)  # ★追加: イグジット理由
    if n_ticks == 0:
        return (
            out_first_pt_time,
            out_first_sl_time,
            out_first_ex_time,
            out_first_ex_reason,
        )
    for i in prange(n_bets):
        t0 = bets_t0[i]
        t1_max = bets_t1_max[i]
        pt = bets_pt_barrier[i]
        sl = bets_sl_barrier[i]
        start_idx = np.searchsorted(ticks_ts, t0, side="left")
        first_pt_found = np.int64(0)
        first_sl_found = np.int64(0)
        first_ex_found = np.int64(0)  # ★追加
        first_ex_reason_found = np.int8(0)  # ★追加: 理由記録用
        for j in range(start_idx, n_ticks):
            tick_time = ticks_ts[j]
            if tick_time > t1_max:
                break
            tick_high = ticks_high[j]
            tick_low = ticks_low[j]
            if first_pt_found == 0 and tick_high >= pt:
                first_pt_found = tick_time
            if first_sl_found == 0 and tick_low <= sl:
                first_sl_found = tick_time
            # ★変更: 第4バリア（純粋なOR条件 ＋ 厳格閾値 ＋ 理由記録）
            if first_ex_found == 0:
                skew_breach = ticks_skewness[j] <= exit_skew_thresh
                kurt_breach = ticks_kurtosis[j] >= exit_kurt_thresh
                phase_breach = ticks_phase[j] <= exit_phase_thresh

                # ATRフィルターを外し、どれか1つでも超過したら発動
                if skew_breach or kurt_breach or phase_breach:
                    first_ex_found = tick_time
                    if skew_breach:
                        first_ex_reason_found = 1
                    elif kurt_breach:
                        first_ex_reason_found = 2
                    elif phase_breach:
                        first_ex_reason_found = 3
            if first_pt_found != 0 and first_sl_found != 0 and first_ex_found != 0:
                break
        out_first_pt_time[i] = first_pt_found
        out_first_sl_time[i] = first_sl_found
        out_first_ex_time[i] = first_ex_found  # ★追加
        out_first_ex_reason[i] = first_ex_reason_found  # ★追加
    return out_first_pt_time, out_first_sl_time, out_first_ex_time, out_first_ex_reason


# --- ▲▲▲ Numbaヘルパー関数ここまで ▲▲▲ ---


class ProxyLabelingEngine:
    """Engine to create a labeled subset of data using 'Context-Adaptive Labeling'."""

    def __init__(self, config: ProxyLabelConfig):
        self.config = config
        logging.info(f"Using output directory: {self.config.output_dir}")
        self.label_counts: Dict[int, int] = {1: 0, -1: 0, 0: 0}
        self.report_data: List[Dict[str, Any]] = []
        self._validate_paths()
        self._validate_config()

    def _validate_paths(self):
        if not self.config.input_dir.exists():
            raise FileNotFoundError(
                f"Input directory not found: {self.config.input_dir}"
            )
        if not self.config.price_data_source_dir.exists():
            raise FileNotFoundError(
                f"Price data source dir not found: {self.config.price_data_source_dir}"
            )

    def _validate_config(self):
        cfg = self.config
        if cfg.filter_mode == "year" and cfg.filter_year is None:
            raise ValueError(
                "Filter mode 'year' requires a specific year (--year YYYY)."
            )
        if cfg.filter_mode == "month":
            if cfg.filter_year is None:
                raise ValueError(
                    "Filter mode 'month' requires a specific year (--year YYYY or via YYYY/MM)."
                )
            if cfg.filter_month is None or not (1 <= cfg.filter_month <= 12):
                raise ValueError(
                    "Filter mode 'month' requires a valid month number (1-12, via YYYY/MM)."
                )
        if cfg.filter_mode not in ["year", "month", "all"]:
            raise ValueError(
                f"Invalid filter_mode: {cfg.filter_mode}. Choose 'year', 'month', or 'all'."
            )

    def _get_duration_in_minutes(self, duration_str: str) -> float:
        """Converts a duration string (e.g., '300m', '90s') to a float value in minutes."""
        match = re.match(r"^(\d+)([ms])$", duration_str)
        if not match:
            logging.warning(
                f"Unexpected duration format in _get_duration_in_minutes: {duration_str}. Trying to parse as minutes."
            )
            num_match = re.match(r"(\d+)", duration_str)
            if num_match:
                return float(num_match.group(1))
            return 0.0
        value, unit = match.groups()
        value_float = float(value)
        if unit == "m":
            return value_float
        elif unit == "s":
            return value_float / 60.0
        return 0.0

    def run(self):
        # --- [フェーズ3 改造] ログメッセージを更新 ---
        logging.info(
            f"### Phase 3: Final Labeling (Project Forge V5 - Crash Lockout) ###"
        )
        logging.info(f"Applying filter: {self.config.get_filter_description()}")
        logging.info(
            f"Logic: Trigger (Hilbert Amp > {TRIGGER_AMP_MULTIPLIER}x) + Lockouts (Skew, Kurt, Body, Phase)"
        )
        logging.info(
            f"Target Rule (R4): PT={REGIME_RULE_R4['pt']}, SL={REGIME_RULE_R4['sl']}, TD={REGIME_RULE_R4['td']}"
        )
        # --- ▲▲▲ 改造ここまで ▲▲▲ ---

        cfg = self.config

        if not cfg.resume and cfg.output_dir.exists():
            logging.info(
                f"Resume is disabled. Deleting existing output directory: {cfg.output_dir}"
            )
            shutil.rmtree(cfg.output_dir)
        cfg.output_dir.mkdir(parents=True, exist_ok=True)

        try:
            logging.info("Step 1: Discovering all feature paths...")
            all_feature_paths = self._discover_feature_paths()

            logging.info(
                "Step 2: Building unified bet/feature frames (separating Hive vs. Files)..."
            )
            unified_hive_lf, unified_file_lf = self._build_unified_lazyframe(
                all_feature_paths
            )

            if (
                unified_hive_lf.collect_schema().names() == []
                and unified_file_lf.collect_schema().names() == []
            ):
                logging.warning(
                    f"No data found for the specified filter '{cfg.get_filter_description()}'. Exiting."
                )
                self._log_final_summary()
                self._generate_report()
                return

            logging.info(
                "Step 3: Preparing price data blueprints (Scanning ALL S2 Universes)..."
            )
            price_components = self._load_all_price_data()
            base_price_lf = price_components["base_lf"]
            atr_lfs = price_components["atr_lfs"]

            logging.info(
                f"   -> Pre-loading {len(atr_lfs)} S2 feature files into memory & Pre-calculating Triggers..."
            )

            # --- [修正] プレ計算ロジックの挿入 ---
            atr_dfs = []
            try:
                for lf in atr_lfs:
                    # まずメモリに展開
                    df = lf.collect().sort("timestamp")

                    # このDFに含まれるトリガーカラムを探す (例: e1e_hilbert_amplitude_50_15m)
                    trigger_col_candidate = None
                    suffix = ""
                    for col in df.columns:
                        if col.startswith(TRIGGER_HILBERT_COL):
                            trigger_col_candidate = col
                            # suffix抽出 (例: "_15m")
                            suffix = col[len(TRIGGER_HILBERT_COL) :]
                            break

                    # トリガーカラムがあれば、全期間で移動平均と判定を一括計算する
                    if trigger_col_candidate:
                        trigger_flag_col = f"is_trigger_on{suffix}"

                        # 判定: 現在値 > 平均 * 3.0
                        df = (
                            df.with_columns(
                                pl.col(trigger_col_candidate)
                                .rolling_mean(window_size=TRIGGER_ROLLING_WINDOW)
                                .alias("hilbert_mean")
                            )
                            .with_columns(
                                (
                                    pl.col(trigger_col_candidate)
                                    > (pl.col("hilbert_mean") * TRIGGER_AMP_MULTIPLIER)
                                ).alias(trigger_flag_col)
                            )
                            .drop("hilbert_mean")
                        )  # 中間カラム削除

                        # logging.info(f"      [Pre-Calc] Added '{trigger_flag_col}' (Window: {TRIGGER_ROLLING_WINDOW})")

                    atr_dfs.append(df)

                logging.info(
                    f"   -> Successfully pre-loaded {len(atr_dfs)} S2 DataFrames with Pre-Calculated Triggers."
                )
            except Exception as e:
                logging.error(
                    f"Failed to pre-load S2 feature files: {e}", exc_info=True
                )
                raise
            # -----------------------------------

            logging.info("Step 4: Discovering daily partitions for processing...")
            partitions_df_hive = self._discover_partitions(unified_hive_lf)
            partitions_df_file = self._discover_partitions(unified_file_lf)

            partitions_df = (
                pl.concat([partitions_df_hive, partitions_df_file])
                .unique()
                .sort("date")
            )

            logging.info(
                f"   -> Found {len(partitions_df)} daily partitions based on S5 structure and filter '{cfg.get_filter_description()}'."
            )

            if partitions_df.is_empty():
                logging.warning(
                    f"No partitions found after applying filter '{cfg.get_filter_description()}'."
                )
                self._log_final_summary()
                self._generate_report()
                return

            logging.info(
                "Step 5: Starting daily processing loop (S2/S5 Hive data scanned per-loop)..."
            )

            max_lookahead_minutes = self._get_duration_in_minutes(REGIME_RULE_R4["td"])
            max_lookahead_delta = dt.timedelta(minutes=max_lookahead_minutes)

            total_partitions = len(partitions_df)
            for row in tqdm(
                partitions_df.iter_rows(named=True),
                total=total_partitions,
                desc="Processing Partitions",
                unit="partition",
            ):
                current_date = row["date"]
                year, month, day = (
                    current_date.year,
                    current_date.month,
                    current_date.day,
                )

                output_partition_dir = (
                    cfg.output_dir / f"year={year}/month={month}/day={day}"
                )
                if cfg.resume and output_partition_dir.exists():
                    logging.debug(
                        f"Resuming: Output already exists for {current_date}. Skipping partition."
                    )
                    continue

                daily_bets_hive_df = unified_hive_lf.filter(
                    pl.col("timestamp").dt.date() == current_date
                ).collect()

                daily_bets_file_df = unified_file_lf.filter(
                    pl.col("timestamp").dt.date() == current_date
                ).collect()

                daily_bets_df = (
                    pl.concat([daily_bets_hive_df, daily_bets_file_df], how="diagonal")
                    .group_by(["timestamp", "timeframe"])
                    .agg(pl.all().drop_nulls().first())
                    .sort("timestamp")
                )

                if daily_bets_df.is_empty():
                    logging.debug(
                        f"No betting data found for {current_date}. Skipping partition."
                    )
                    continue

                min_ts_req = daily_bets_df["timestamp"].min()
                if min_ts_req is None:
                    continue

                max_ts_req = min_ts_req + max_lookahead_delta + dt.timedelta(days=2)

                price_window_df = (
                    base_price_lf.filter(
                        pl.col("timestamp").is_between(min_ts_req, max_ts_req)
                    )
                    .collect()
                    .sort("timestamp")
                )

                if price_window_df.is_empty():
                    logging.warning(
                        f"No base price data found for {current_date}. Skipping."
                    )
                    continue

                # 結合処理 (プレ計算済みのTriggerカラムもここで結合される)
                for atr_df in atr_dfs:
                    atr_df_small = atr_df.filter(
                        pl.col("timestamp").is_between(min_ts_req, max_ts_req)
                    )
                    if not atr_df_small.is_empty():
                        price_window_df = price_window_df.join_asof(
                            atr_df_small,
                            on="timestamp",
                        )
                price_window_df = price_window_df.fill_null(
                    strategy="forward"
                ).fill_null(strategy="backward")

                daily_labeled_df = self._calculate_labels_for_batch(
                    daily_bets_df, price_window_df
                )

                if daily_labeled_df is not None and not daily_labeled_df.is_empty():
                    self._update_label_counts(daily_labeled_df)
                    self._collect_report_data(daily_labeled_df, current_date)
                    output_partition_dir.mkdir(parents=True, exist_ok=True)
                    daily_labeled_df.write_parquet(
                        output_partition_dir / "data.parquet", compression="zstd"
                    )

                del daily_bets_df, price_window_df, daily_labeled_df
                del daily_bets_hive_df, daily_bets_file_df
                gc.collect()

            self._log_final_summary()
            self._generate_report()

        except Exception as e:
            logging.error(f"A critical error occurred: {e}", exc_info=True)
            raise

    # =========================================================================
    # コアロジック (S5/S2読み込み、パーティション発見など)
    # =========================================================================

    def _discover_feature_paths(self) -> List[Path]:
        logging.info(
            f"Recursively searching for feature paths in {self.config.input_dir}..."
        )
        discovered_paths = list(self.config.input_dir.rglob("features_*_neutralized*"))
        feature_paths = [
            p for p in discovered_paths if p.is_dir() or p.name.endswith(".parquet")
        ]
        if not feature_paths:
            raise FileNotFoundError(
                f"No feature paths found in {self.config.input_dir}."
            )
        logging.info(f"  -> Found {len(feature_paths)} feature paths.")
        return feature_paths

    def _build_unified_lazyframe(
        self, feature_paths: List[Path]
    ) -> Tuple[pl.LazyFrame, pl.LazyFrame]:
        """
        [修正版] S5の特徴量データを、Hiveパーティション(Lazy)と
        単一ファイル(Lazy)に分離する。

        ★メモリ対策: ここでは .collect() をせず、LazyFrameのまま返す。
        行増殖バグの修正(GroupBy)は、メモリ溢れを防ぐため run() の日次ループ内で行う。
        """
        all_lazy_frames_hive = []
        all_lazy_frames_file = []

        timeframe_pattern = re.compile(
            r"features_e\d+[a-z]?_([a-zA-Z0-9\.]+)(?:_neutralized)?"
        )
        cfg = self.config

        logging.info("   -> Separating Hive partitions vs. single files for S5...")
        for path in feature_paths:
            name_to_match = path.stem if path.is_file() else path.name
            match = timeframe_pattern.search(name_to_match)
            if not match:
                logging.warning(f"Could not extract timeframe from: {name_to_match}")
                continue
            timeframe = match.group(1)
            timeframe_suffix = f"_{timeframe}"

            lf: Optional[pl.LazyFrame] = None

            if path.is_dir():
                scan_base_path = path
                if cfg.filter_mode == "year" and cfg.filter_year is not None:
                    scan_base_path = scan_base_path / f"year={cfg.filter_year}"
                elif (
                    cfg.filter_mode == "month"
                    and cfg.filter_year is not None
                    and cfg.filter_month is not None
                ):
                    scan_base_path = (
                        scan_base_path
                        / f"year={cfg.filter_year}/month={cfg.filter_month}"
                    )

                if scan_base_path.exists():
                    try:
                        lf = pl.scan_parquet(str(scan_base_path / "**/*.parquet"))
                    except Exception as e:
                        logging.warning(f"Failed to scan {scan_base_path}: {e}")
                        continue

                if lf is not None:
                    lf_renamed = self._rename_features(lf, timeframe_suffix)
                    if lf_renamed is not None:
                        all_lazy_frames_hive.append(
                            lf_renamed.with_columns(
                                pl.lit(timeframe).alias("timeframe")
                            )
                        )

            elif path.is_file():
                try:
                    lf_full = pl.scan_parquet(str(path))
                    if "timestamp" not in lf_full.collect_schema().names():
                        continue
                except Exception as e:
                    logging.warning(f"Failed to scan file {path}: {e}")
                    continue

                date_filter: Optional[pl.Expr] = None
                if cfg.filter_mode == "year" and cfg.filter_year is not None:
                    date_filter = pl.col("timestamp").dt.year() == cfg.filter_year
                elif (
                    cfg.filter_mode == "month"
                    and cfg.filter_year is not None
                    and cfg.filter_month is not None
                ):
                    year_filter = pl.col("timestamp").dt.year() == cfg.filter_year
                    month_filter = pl.col("timestamp").dt.month() == cfg.filter_month
                    date_filter = year_filter & month_filter

                if date_filter is not None:
                    lf = lf_full.filter(date_filter)
                else:
                    lf = lf_full

                if lf is not None:
                    lf_renamed = self._rename_features(lf, timeframe_suffix)
                    if lf_renamed is not None:
                        all_lazy_frames_file.append(
                            lf_renamed.with_columns(
                                pl.lit(timeframe).alias("timeframe")
                            )
                        )

        unified_hive_lf = pl.LazyFrame()
        if all_lazy_frames_hive:
            try:
                # LazyFrameのまま結合 (ここでは集約しない)
                unified_hive_lf = pl.concat(all_lazy_frames_hive, how="diagonal")
                logging.info(
                    f"   -> Prepared S5 Hive LazyFrame ({len(all_lazy_frames_hive)} sources)."
                )
            except pl.exceptions.ComputeError as e:
                if "cannot concat empty list" in str(e).lower():
                    logging.warning("S5 Hive concatenation resulted in empty data.")
                else:
                    raise e
        else:
            logging.warning("No S5 Hive (tick) data found for the filter.")

        unified_file_lf = pl.LazyFrame()
        if all_lazy_frames_file:
            try:
                # LazyFrameのまま結合 (ここでは集約しない)
                unified_file_lf = pl.concat(all_lazy_frames_file, how="diagonal")
                logging.info(
                    f"   -> Prepared S5 non-tick LazyFrame ({len(all_lazy_frames_file)} sources)."
                )
            except pl.exceptions.ComputeError as e:
                if "cannot concat empty list" in str(e).lower():
                    logging.warning(
                        "S5 non-tick file concatenation resulted in empty data."
                    )
                else:
                    raise e
        else:
            logging.warning("No S5 non-tick (single file) data found for the filter.")

        return unified_hive_lf, unified_file_lf

    def _rename_features(
        self, lf: pl.LazyFrame, timeframe_suffix: str
    ) -> Optional[pl.LazyFrame]:
        try:
            current_schema = lf.collect_schema()
            if not current_schema.names():
                return None
            feature_cols = [col for col in current_schema.names() if col != "timestamp"]
            rename_exprs = [
                pl.col(col).alias(f"{col}{timeframe_suffix}") for col in feature_cols
            ]
            select_exprs = [pl.col("timestamp").cast(pl.Datetime("us", "UTC"))]
            if rename_exprs:
                select_exprs.extend(rename_exprs)
            return lf.select(select_exprs)
        except Exception as e:
            logging.warning(
                f"Failed to rename features with suffix {timeframe_suffix}: {e}"
            )
            return None

    def _load_all_price_data(self) -> Dict[str, Any]:
        """
        [修正版] V5ロジックに必要なS2特徴量（全Engine対象）と
        基本価格データ（Open/High/Low/Close）を読み込む。
        """
        # 価格データ（Tick）の場所（ここはUniverse Cで正解）
        price_source_dir = (
            self.config.price_data_source_dir / "feature_value_a_vast_universeC"
        )
        tick_dir = price_source_dir / "features_e1c_tick"

        if not tick_dir.exists():
            raise FileNotFoundError(f"Master price directory not found: {tick_dir}")

        logging.info(
            f"   -> Lazily scanning '{tick_dir}' as the master price source (S2 Tick)."
        )
        base_lf = (
            pl.scan_parquet(str(tick_dir / "**/*.parquet"))
            .select("timestamp", "open", "close", "high", "low")
            .with_columns(pl.col("timestamp").cast(pl.Datetime("us", "UTC")))
            .unique("timestamp", keep="first", maintain_order=True)
        )

        # --- 修正点A: 必須カラムの構築（Noneを除外） ---
        required_cols = {TRIGGER_HILBERT_COL, BARRIER_ATR_COL}

        # Lockoutルールから外部参照が必要なカラムのみ追加
        for rule in LOCKOUT_RULES.values():
            if rule["col"] is not None:
                required_cols.add(rule["col"])

        # ★追加: 緊急イグジットルールから外部参照が必要なカラムも追加
        for rule in EMERGENCY_EXIT_RULES.values():
            if rule.get("col") is not None:
                required_cols.add(rule["col"])

        # --- 修正点B: S2ルートディレクトリから全Engineを探索 ---
        s2_root_dir = self.config.price_data_source_dir

        # 再帰的にすべての .parquet を探す（Engine A, B, C, D, E... すべて対象）
        feature_files = list(s2_root_dir.rglob("features_*.parquet"))

        if not feature_files:
            logging.warning(
                f"No S2 feature files found in {s2_root_dir}. Lockout logic might fail."
            )

        timeframe_pattern = re.compile(
            r"features_(?:e\d+[a-z]+)_([a-zA-Z0-9\.]+)(?:_.*)?\.parquet"
        )

        all_feature_lfs = []
        logging.info(
            f"   -> Scanning {len(feature_files)} S2 files across ALL engines for V5 features..."
        )

        loaded_features = set()

        for f_path in feature_files:
            # Tickデータフォルダ自体はスキップ（重複ロード防止）
            if "features_e1c_tick" in str(f_path):
                continue

            match = timeframe_pattern.search(f_path.name)
            if not match:
                continue
            timeframe = match.group(1)

            try:
                lf_temp = pl.scan_parquet(str(f_path))
                schema_cols = lf_temp.collect_schema().names()

                time_col = None
                if "timestamp" in schema_cols:
                    time_col = "timestamp"
                elif "datetime" in schema_cols:
                    time_col = "datetime"

                if not time_col:
                    continue

                found_cols = [c for c in schema_cols if c in required_cols]

                if found_cols:
                    select_exprs = [
                        pl.col(time_col)
                        .cast(pl.Datetime("us", "UTC"))
                        .alias("timestamp")
                    ]
                    for col in found_cols:
                        target_col_name = f"{col}_{timeframe}"
                        select_exprs.append(pl.col(col).alias(target_col_name))
                        loaded_features.add(col)

                    all_feature_lfs.append(lf_temp.select(select_exprs))

            except Exception as e:
                logging.warning(f"Failed to inspect {f_path.name}: {e}")

        # 検証
        missing_mandatory = []
        if TRIGGER_HILBERT_COL not in loaded_features:
            missing_mandatory.append("Trigger(Hilbert)")
        if BARRIER_ATR_COL not in loaded_features:
            logging.warning(
                f"WARNING: {BARRIER_ATR_COL} for barrier scaling not found. Will use fixed % fallback."
            )

        if missing_mandatory:
            logging.critical(
                f"CRITICAL: Missing mandatory columns: {missing_mandatory}. Labeling will fail."
            )
        else:
            logging.info(f"   -> V5 Features Ready. Loaded: {list(loaded_features)}")

        return {"base_lf": base_lf, "atr_lfs": all_feature_lfs}

    def _discover_partitions(self, unified_lf: pl.LazyFrame) -> pl.DataFrame:
        if unified_lf.collect_schema().names() == []:
            return pl.DataFrame({"date": []}).select(pl.col("date").cast(pl.Date))
        df_dates = (
            unified_lf.select(pl.col("timestamp").dt.date().alias("date"))
            .unique()
            .collect()
        )
        if df_dates.is_empty():
            return df_dates.select(pl.col("date").cast(pl.Date))
        return df_dates.sort("date")

    def _get_bar_duration_minutes(self, timeframe: str) -> float:
        """Calculates the approximate duration of a bar in minutes for a given timeframe string."""
        if timeframe == "tick":
            return 0.0  # Tick has no fixed duration in this context

        if timeframe == "MN":
            return 1.0 * 43200  # Approximate minutes in a month (value=1)

        value_match = re.search(r"(\d*\.?\d+)", timeframe)
        unit_match = re.search(r"([A-Z])", timeframe)  # Expecting M, H, D, W

        if not value_match or not unit_match:
            logging.warning(
                f"Could not parse value or unit from timeframe: {timeframe}"
            )
            return 0.0  # Return 0 if parsing fails

        try:
            value_str = value_match.group(1)
            value = float(value_str) if value_str else 1.0
        except ValueError:
            logging.warning(
                f"Could not convert value '{value_match.group(1)}' to float for timeframe: {timeframe}"
            )
            return 0.0

        unit = unit_match.group(1)

        if unit == "M":  # Minute
            return value
        if unit == "H":  # Hour
            return value * 60
        if unit == "D":  # Day
            return value * 1440  # 60 * 24
        if unit == "W":  # Week
            return value * 10080  # 60 * 24 * 7

        logging.warning(f"Unhandled timeframe unit '{unit}' in timeframe: {timeframe}")
        return 0.0  # Return 0 for any other unhandled units

    # --- コアロジック (変動ATR対応) ---
    def _calculate_labels_for_batch(
        self, daily_bets_df: pl.DataFrame, price_window_df: pl.DataFrame
    ) -> pl.DataFrame | None:
        """
        [フェーズ3: V5 Logic (Final Fix)]
        ATR結合バグを修正し、通過したベットが確実にラベル生成されるようにしたバージョン。
        """
        if daily_bets_df.is_empty():
            return None

        if not NUMBA_AVAILABLE:
            logging.error("Numba is required for labeling but not found.")
            return None

        # 1. Numba用にTickデータを配列化
        try:
            ticks_df_np = price_window_df.select(
                pl.col("timestamp").cast(pl.Int64).alias("ticks_ts"),
                pl.col("high").alias("ticks_high"),
                pl.col("low").alias("ticks_low"),
            )
            ticks_ts_np = ticks_df_np["ticks_ts"].to_numpy()
            ticks_high_np = ticks_df_np["ticks_high"].to_numpy()
            ticks_low_np = ticks_df_np["ticks_low"].to_numpy()
        except Exception as e:
            logging.error(f"Failed to convert tick data to Numpy arrays: {e}")
            return None

        labeled_chunks = []

        # Timeframeごとにループ処理
        for timeframe_tuple, group_df in daily_bets_df.group_by("timeframe"):
            timeframe = timeframe_tuple[0]
            if timeframe is None or group_df.is_empty():
                continue

            # --- V5 Feature Column Names Construction ---
            tf_suffix = f"_{timeframe}"
            trigger_flag_col = f"is_trigger_on{tf_suffix}"
            atr_col_name = f"{BARRIER_ATR_COL}{tf_suffix}"

            # --- [DEBUG] 1. トリガーカラムの存在確認 ---
            if trigger_flag_col not in price_window_df.columns:
                if timeframe != "tick":
                    cols_sample = price_window_df.columns[:5] + ["..."]
                    logging.warning(
                        f"DEBUG [{timeframe}]: Missing trigger col '{trigger_flag_col}'! Available cols sample: {cols_sample}"
                    )
                continue

            # ロックアウト用カラム名の解決
            lockout_cols_map = {}
            for key, rule in LOCKOUT_RULES.items():
                if rule["col"] is None:
                    continue
                col_name = f"{rule['col']}{tf_suffix}"
                if col_name in price_window_df.columns:
                    lockout_cols_map[key] = col_name

            # --- Step 1: Join Bets with S2 Metrics (INCLUDE ATR HERE!) ---
            # 修正: ここでATRも含めておくことで、Step 4での結合ミス(null)を防ぐ
            cols_to_use = (
                ["timestamp", "open", "high", "low", "close"]
                + [trigger_flag_col]
                + list(lockout_cols_map.values())
            )

            # ATRがあれば追加
            has_atr = False
            if atr_col_name in price_window_df.columns:
                cols_to_use.append(atr_col_name)
                has_atr = True

            # 必要なカラムだけ持ったS2データ
            s2_subset = price_window_df.select(cols_to_use)

            # BetsにS2データを結合 (join_asofなので時刻ズレがあっても直近の値を取れる)
            original_cols = [
                c for c in group_df.columns if c not in ["timestamp", "close"]
            ]
            bets_joined = group_df.join_asof(
                s2_subset, on="timestamp", strategy="backward"
            )

            # --- Step 2: Dynamic Calculation (Body Ratio) ---
            bets_joined = bets_joined.with_columns(
                (
                    (pl.col("close") - pl.col("open")).abs()
                    / ((pl.col("high") - pl.col("low")) + 1e-9)
                ).alias("calculated_body_ratio")
            )

            # --- Step 3: Apply V5 Logic ---

            # 1. Trigger Check
            total_bets = bets_joined.height
            triggered_bets = bets_joined.filter(pl.col(trigger_flag_col) == True)
            trigger_pass_count = triggered_bets.height

            logging.warning(
                f"DEBUG [{timeframe}]: Bets Total {total_bets} -> Triggered {trigger_pass_count}"
            )

            if trigger_pass_count == 0:
                continue

            # 2. Lockout Check
            valid_bets = triggered_bets

            for rule_name in ["skewness", "kurtosis", "body_ratio", "phase_stability"]:
                before_count = valid_bets.height
                if before_count == 0:
                    break
                rule = LOCKOUT_RULES.get(rule_name)
                if not rule:
                    continue

                if rule_name == "body_ratio":
                    target_col = "calculated_body_ratio"
                else:
                    target_col = lockout_cols_map.get(rule_name)

                if not target_col:
                    continue

                op = rule["op"]
                thresh = rule["thresh"]
                if op == "gt":
                    valid_bets = valid_bets.filter(pl.col(target_col) <= thresh)
                elif op == "lt":
                    valid_bets = valid_bets.filter(pl.col(target_col) >= thresh)

                removed = before_count - valid_bets.height
                if removed > 0:
                    logging.warning(
                        f"DEBUG [{timeframe}]: Lockout '{rule_name}' removed {removed} bets."
                    )

            if valid_bets.is_empty():
                logging.warning(
                    f"DEBUG [{timeframe}]: All bets eliminated by Lockouts."
                )
                continue

            logging.warning(
                f"DEBUG [{timeframe}]: >>> FINAL VALID BETS: {valid_bets.height} <<<"
            )

            # --- Step 4: Triple Barrier Method (TBM) ---
            rule_pt = REGIME_RULE_R4["pt"]
            rule_sl = REGIME_RULE_R4["sl"]
            rule_td_str = REGIME_RULE_R4["td"]
            rule_payoff = REGIME_RULE_R4["payoff"]

            target_duration_minutes = self._get_duration_in_minutes(rule_td_str)
            bar_duration_minutes = self._get_bar_duration_minutes(timeframe)

            if timeframe == "tick":
                t1_max_expr = pl.col("timestamp") + pl.duration(
                    minutes=target_duration_minutes
                )
            else:
                if bar_duration_minutes == 0:
                    continue
                lookahead_bars = max(
                    1, int(round(target_duration_minutes / bar_duration_minutes))
                )
                t1_max_expr = pl.col("timestamp") + pl.duration(
                    minutes=lookahead_bars * bar_duration_minutes
                )

            # Scaling (ATR)
            # 修正: Step 1ですでに結合済みのATRカラムを使用する

            if has_atr:
                vol_col = pl.col(atr_col_name)
                tbm_prep_df = valid_bets
            else:
                # Fallback
                vol_col = pl.col("fallback_vol")
                tbm_prep_df = valid_bets.with_columns(
                    (pl.col("close") * 0.002).alias("fallback_vol")
                )

            tbm_df = tbm_prep_df.select(
                pl.col("timestamp").alias(
                    "t0"
                ),  # Step1のjoin_asofで残ったtimestampを使う
                (pl.col("close") + vol_col * rule_pt).alias("pt_barrier"),
                (pl.col("close") - vol_col * rule_sl).alias("sl_barrier"),
                t1_max_expr.alias("t1_max"),
                pl.col("close"),
                vol_col.alias("atr_value"),  # <--- これを追加！
                pl.col(original_cols),
            ).drop_nulls(["pt_barrier", "sl_barrier"])

            if tbm_df.is_empty():
                logging.warning(
                    f"DEBUG [{timeframe}]: TBM DF empty after Barrier Calc! (Possible ATR nulls?)"
                )
                continue

            # Numba実行
            try:
                bets_t0_np = tbm_df["t0"].cast(pl.Int64).to_numpy()
                bets_t1_max_np = tbm_df["t1_max"].cast(pl.Int64).to_numpy()
                bets_pt_np = tbm_df["pt_barrier"].to_numpy(writable=True)
                bets_sl_np = tbm_df["sl_barrier"].to_numpy(writable=True)
            except Exception as e:
                logging.error(f"Numpy conversion error in {timeframe}: {e}")
                continue

            # ★追加: 第4バリア用の特徴量配列をprice_window_dfから取得
            ex_skew_col = f"{EMERGENCY_EXIT_RULES['skewness']['col']}{tf_suffix}"
            ex_kurt_col = f"{EMERGENCY_EXIT_RULES['kurtosis']['col']}{tf_suffix}"
            ex_phase_col = (
                f"{EMERGENCY_EXIT_RULES['phase_stability']['col']}{tf_suffix}"
            )

            # ★修正: ATR自体ではなく「現在の足の変動幅(High-Low)がATRの2倍以上か」を判定
            if atr_col_name in price_window_df.columns:
                atr_s = price_window_df[atr_col_name]
                spread_s = price_window_df["high"] - price_window_df["low"]
                # 現在のボラティリティ(変動幅)が平常時(ATR)の2倍以上あるか
                ticks_vol_surge_np = (
                    (spread_s >= (atr_s * 2.0)).fill_null(False).to_numpy()
                )
            else:
                ticks_vol_surge_np = np.ones(len(ticks_ts_np), dtype=np.bool_)

            _dummy = np.zeros(len(ticks_ts_np), dtype=np.float64)
            ticks_skew_np = (
                price_window_df[ex_skew_col].to_numpy()
                if ex_skew_col in price_window_df.columns
                else _dummy
            )
            ticks_kurt_np = (
                price_window_df[ex_kurt_col].to_numpy()
                if ex_kurt_col in price_window_df.columns
                else _dummy  # ★修正: データがない場合は 0.0 を返し、誤発動を完全に防ぐ
            )
            ticks_phase_np = (
                price_window_df[ex_phase_col].to_numpy()
                if ex_phase_col in price_window_df.columns
                else _dummy + 999.0
            )
            first_pt_time_np, first_sl_time_np, first_ex_time_np, first_ex_reason_np = (
                _numba_find_hits(
                    bets_t0_np,
                    bets_t1_max_np,
                    bets_pt_np,
                    bets_sl_np,
                    ticks_ts_np,
                    ticks_high_np,
                    ticks_low_np,
                    ticks_skew_np,  # ★追加
                    ticks_kurt_np,  # ★追加
                    ticks_phase_np,  # ★追加
                    ticks_vol_surge_np,  # ★追加: ATRコンファメーション配列
                    float(EMERGENCY_EXIT_RULES["skewness"]["thresh"]),  # ★追加
                    float(EMERGENCY_EXIT_RULES["kurtosis"]["thresh"]),  # ★追加
                    float(EMERGENCY_EXIT_RULES["phase_stability"]["thresh"]),  # ★追加
                )
            )

            final_group_df = (
                tbm_df.with_columns(
                    pl.Series("first_pt_time_int", first_pt_time_np),
                    pl.Series("first_sl_time_int", first_sl_time_np),
                    pl.Series("first_ex_time_int", first_ex_time_np),  # ★追加
                    pl.Series(
                        "first_ex_reason_int", first_ex_reason_np
                    ),  # ★追加: イグジット理由
                )
                .with_columns(
                    first_pt_time=pl.when(pl.col("first_pt_time_int") > 0)
                    .then(pl.col("first_pt_time_int"))
                    .otherwise(None)
                    .cast(pl.Datetime("us", "UTC")),
                    first_sl_time=pl.when(pl.col("first_sl_time_int") > 0)
                    .then(pl.col("first_sl_time_int"))
                    .otherwise(None)
                    .cast(pl.Datetime("us", "UTC")),
                    first_ex_time=pl.when(pl.col("first_ex_time_int") > 0)  # ★追加
                    .then(pl.col("first_ex_time_int"))  # ★追加
                    .otherwise(None)  # ★追加
                    .cast(pl.Datetime("us", "UTC")),  # ★追加
                )
                .drop(
                    "first_pt_time_int", "first_sl_time_int", "first_ex_time_int"
                )  # ★変更
            )

            labeled_group = final_group_df.with_columns(
                # ★変更: PT/SL/EX(緊急イグジット)の中で最も早く発火したものを採用
                t1=pl.when(
                    (pl.col("first_pt_time").is_not_null())
                    & (
                        (pl.col("first_sl_time").is_null())
                        | (pl.col("first_pt_time") <= pl.col("first_sl_time"))
                    )
                    & (
                        (pl.col("first_ex_time").is_null())
                        | (pl.col("first_pt_time") <= pl.col("first_ex_time"))
                    )
                )
                .then(pl.col("first_pt_time"))
                .when(
                    (pl.col("first_sl_time").is_not_null())
                    & (
                        (pl.col("first_ex_time").is_null())
                        | (pl.col("first_sl_time") <= pl.col("first_ex_time"))
                    )
                )
                .then(pl.col("first_sl_time"))
                .when(pl.col("first_ex_time").is_not_null())
                .then(pl.col("first_ex_time"))
                .otherwise(pl.col("t1_max")),
                label=pl.when(
                    (pl.col("first_pt_time").is_not_null())
                    & (
                        (pl.col("first_sl_time").is_null())
                        | (pl.col("first_pt_time") <= pl.col("first_sl_time"))
                    )
                    & (
                        (pl.col("first_ex_time").is_null())
                        | (pl.col("first_pt_time") <= pl.col("first_ex_time"))
                    )
                )
                .then(pl.lit(1, dtype=pl.Int8))
                .when(
                    (pl.col("first_sl_time").is_not_null())
                    & (
                        (pl.col("first_ex_time").is_null())
                        | (pl.col("first_sl_time") <= pl.col("first_ex_time"))
                    )
                )
                .then(pl.lit(-1, dtype=pl.Int8))
                .when(pl.col("first_ex_time").is_not_null())
                .then(pl.lit(-1, dtype=pl.Int8))  # 緊急イグジットも損切り扱い
                .otherwise(pl.lit(0, dtype=pl.Int8)),
                payoff_ratio=pl.lit(rule_payoff, dtype=pl.Float32),
                sl_multiplier=pl.lit(rule_sl, dtype=pl.Float32),
                pt_multiplier=pl.lit(rule_pt, dtype=pl.Float32),
                direction=pl.lit(1, dtype=pl.Int8),
                # ★追加: どのバリアで終了したかを記録
                exit_type=pl.when(
                    (pl.col("first_pt_time").is_not_null())
                    & (
                        (pl.col("first_sl_time").is_null())
                        | (pl.col("first_pt_time") <= pl.col("first_sl_time"))
                    )
                    & (
                        (pl.col("first_ex_time").is_null())
                        | (pl.col("first_pt_time") <= pl.col("first_ex_time"))
                    )
                )
                .then(pl.lit("PT"))
                .when(
                    (pl.col("first_sl_time").is_not_null())
                    & (
                        (pl.col("first_ex_time").is_null())
                        | (pl.col("first_sl_time") <= pl.col("first_ex_time"))
                    )
                )
                .then(pl.lit("SL"))
                .when(pl.col("first_ex_time").is_not_null())
                .then(
                    pl.when(pl.col("first_ex_reason_int") == 1)
                    .then(pl.lit("EM_SKEW"))
                    .when(pl.col("first_ex_reason_int") == 2)
                    .then(pl.lit("EM_KURT"))
                    .when(pl.col("first_ex_reason_int") == 3)
                    .then(pl.lit("EM_PHASE"))
                    .otherwise(pl.lit("EMERGENCY"))
                )
                .otherwise(pl.lit("TIMEOUT")),
            ).rename({"t0": "timestamp"})

            labeled_chunks.append(
                labeled_group.drop(
                    [
                        "pt_barrier",
                        "sl_barrier",
                        "t1_max",
                        "first_pt_time",
                        "first_sl_time",
                        "first_ex_time",  # ★追加
                    ]
                )
            )

        if not labeled_chunks:
            return None
        return pl.concat(labeled_chunks).sort("timestamp")

    def _update_label_counts(self, df: pl.DataFrame):
        counts = df.group_by("label").len()
        for row in counts.iter_rows(named=True):
            if row["label"] in self.label_counts:
                self.label_counts[row["label"]] += row["len"]

    def _log_final_summary(self):
        total_samples = sum(self.label_counts.values())
        if total_samples == 0:
            logging.warning("No samples were processed for the selected filter.")
            summary = (
                "\n" + "=" * 60 + "\n"
                f"### ATR Regime Labeling COMPLETED (Filter: {self.config.get_filter_description()}) ###\n"
                f"The 'V4_R4_Only' version of the 'Problem Set' is ready at: {self.config.output_dir}\n"
                "  - No samples matched the specified filter.\n" + "=" * 60
            )
            logging.info(summary)
            return
        pos = self.label_counts.get(1, 0)
        neg = self.label_counts.get(-1, 0)
        scale_pos_weight = neg / pos if pos > 0 else 1.0
        summary = (
            "\n" + "=" * 60 + "\n"
            f"### ATR Regime Labeling COMPLETED (Filter: {self.config.get_filter_description()}) ###\n"
            f"The 'V4_R4_Only' version of the 'Problem Set' is ready at: {self.config.output_dir}\n"
            f"  - Total Labeled Samples (R4 only): {total_samples}\n"
            f"  - (+) Profit-Take: {pos}, (-) Stop-Loss: {neg}, (0) Timed-Out: {self.label_counts.get(0, 0)}\n"
            f"  - Calculated `scale_pos_weight` for next step: {scale_pos_weight:.4f}\n"
            + "="
            * 60
        )
        logging.info(summary)

    def _collect_report_data(self, df: pl.DataFrame, current_date: dt.date):
        try:
            required_cols = ["timestamp", "t1", "timeframe", "label"]
            if not all(col in df.columns for col in required_cols):
                logging.warning(
                    f"Skipping report data collection for {current_date}: Missing required columns."
                )
                return
            report_df = df.with_columns(
                (pl.col("t1") - pl.col("timestamp"))
                .dt.total_seconds()
                .alias("event_duration_seconds"),
                pl.lit(current_date).alias("date"),
                pl.col("exit_type").str.to_lowercase().alias("exit_type_str"),
            ).select(
                [
                    "timeframe",
                    "label",
                    "exit_type_str",
                    "event_duration_seconds",
                    "date",
                ]
            )
            self.report_data.extend(report_df.to_dicts())
        except Exception as e:
            logging.warning(f"Error collecting report data for {current_date}: {e}")

    def _generate_report(self):
        logging.info("Generating detailed execution report...")
        cfg = self.config

        # --- [修正] 書き込み直前にディレクトリ存在を強制的に保証する ---
        if not cfg.output_dir.exists():
            logging.warning(
                f"Output directory missing at report generation. Re-creating: {cfg.output_dir}"
            )
            cfg.output_dir.mkdir(parents=True, exist_ok=True)
        # -------------------------------------------------------------

        report_filename = "execution_report_regime_v5.txt"
        report_path = cfg.output_dir / report_filename

        rule = REGIME_RULE_R4

        # V5用のサマリーテーブル作成
        lockout_desc = ", ".join(LOCKOUT_RULES.keys())

        summary_table = f"""
| Item | Value |
|:---|:---|
| **Execution Timestamp** | `{cfg.execution_start_time}` |
| **Script Path** | `{" / ".join(Path(__file__).parts[-4:])}` |
| **Data Filter Applied** | `{cfg.get_filter_description()}` |
| **Labeling Strategy** | `Project Forge V5 (Trigger + Lockout)` |
| **Trigger Logic** | `Hilbert Amp > {TRIGGER_AMP_MULTIPLIER}x (Rolling {TRIGGER_ROLLING_WINDOW})` |
| **Lockout Logic** | `{lockout_desc}` |
| **Target Duration (R4)** | `{rule["td"]}` |
| **Payoff Ratio (PT/SL) (R4)** | `{rule["payoff"]:.2f}` (`PT mult: {rule["pt"]}`, `SL mult: {rule["sl"]}`) |
"""

        if not self.report_data:
            logging.warning("No data available for report generation.")
            report_content = f"# Proxy Labeling Engine - Execution Report\n\n**Filter Applied: {cfg.get_filter_description()}**\n\n**No samples were processed for duration '{rule['td']}' with the specified filter.**"
            # ここでも書き込みを行うため、ディレクトリ保証が効いてきます
            report_path.write_text(report_content)
            return

        try:
            df = pl.from_dicts(self.report_data)
            total = len(df)
            pt = df.filter(pl.col("label") == 1).height
            sl = df.filter(pl.col("label") == -1).height
            to = df.filter(pl.col("label") == 0).height
            scale_pos_weight = sl / pt if pt > 0 else 1.0

            # ★追加: exit_type_str列による内訳集計（詳細化）
            em_skew = df.filter(pl.col("exit_type_str") == "em_skew").height
            em_kurt = df.filter(pl.col("exit_type_str") == "em_kurt").height
            em_phase = df.filter(pl.col("exit_type_str") == "em_phase").height
            em = em_skew + em_kurt + em_phase
            sl_only = df.filter(pl.col("exit_type_str") == "sl").height

            perf_table = f"""
    | Metric | Value |
    |:---|:---|
    | **Total Labeled Samples (V5 Passed)** | `{total:,}` |
    | **(+) Profit-Take (PT)** | `{pt:,}` (`{pt / total:.2%}`) |
    | **(-) Stop-Loss (SL) 合計** | `{sl:,}` (`{sl / total:.2%}`) |
    | **　└ 通常SL（価格ベース）** | `{sl_only:,}` (`{sl_only / total:.2%}`) |
    | **　└ 緊急イグジット（構造崩壊）** | `{em:,}` (`{em / total:.2%}`) |
    | **　　├ 歪度 (Skewness)** | `{em_skew:,}` (`{em_skew / total:.2%}`) |
    | **　　├ 尖度 (Kurtosis)** | `{em_kurt:,}` (`{em_kurt / total:.2%}`) |
    | **　　└ 位相 (Phase Stability)** | `{em_phase:,}` (`{em_phase / total:.2%}`) |
    | **(0) Timed-Out (TO)** | `{to:,}` (`{to / total:.2%}`) |
    | **`scale_pos_weight` (calc)** | `{scale_pos_weight:.4f}` |
    """
            tf_breakdown = (
                df.group_by("timeframe")
                .agg(
                    pl.len().alias("bets"),
                    pl.col("label").filter(pl.col("label") == 1).len().alias("pt"),
                    pl.col("label").filter(pl.col("label") == -1).len().alias("sl"),
                    pl.col("label").filter(pl.col("label") == 0).len().alias("to"),
                )
                .sort("bets", descending=True)
            )
            tf_table = "| Timeframe | Bet Count | PT (%) | SL (%) | Timeout (%) |\n"
            tf_table += "|:---|---:|---:|---:|---:|\n"
            for row in tf_breakdown.to_dicts():
                total_bets = row["bets"]
                pt_pct = row["pt"] / total_bets if total_bets > 0 else 0
                sl_pct = row["sl"] / total_bets if total_bets > 0 else 0
                to_pct = row["to"] / total_bets if total_bets > 0 else 0
                tf_table += f"| **{row['timeframe']}** | `{total_bets:,}` | `{row['pt']:,}` (`{pt_pct:.2%}`) | `{row['sl']:,}` (`{sl_pct:.2%}`) | `{row['to']:,}` (`{to_pct:.2%}`) |\n"

            duration_stats_seconds = df["event_duration_seconds"]
            valid_durations = duration_stats_seconds.filter(
                duration_stats_seconds.is_finite()
                & duration_stats_seconds.is_not_null()
            )
            if valid_durations.len() > 0:
                duration_stats = (valid_durations / 60).describe()

                def get_stat(df_describe, stat_name):
                    try:
                        value = (
                            df_describe.filter(pl.col("statistic") == stat_name)
                            .select(pl.col("value"))
                            .item()
                        )
                        return f"{value:.2f}"
                    except Exception as e:
                        logging.warning(
                            f"Could not extract statistic '{stat_name}': {e}"
                        )
                        return "N/A"

                duration_table = f"""
| Statistic | Duration (minutes) |
|:---|:---|
| **Mean** | `{get_stat(duration_stats, "mean")}` |
| **Median (50%)** | `{get_stat(duration_stats, "50%")}` |
| **Std Dev** | `{get_stat(duration_stats, "std")}` |
| **Min** | `{get_stat(duration_stats, "min")}` |
| **Max** | `{get_stat(duration_stats, "max")}` |
| **25th Percentile** | `{get_stat(duration_stats, "25%")}` |
| **75th Percentile** | `{get_stat(duration_stats, "75%")}` |
"""
            else:
                duration_table = "| Statistic | Duration (minutes) |\n|:---|:---|\n| **N/A** | No valid duration data |"

            daily_activity = (
                df.group_by("date").len().sort("len", descending=True).limit(10)
            )
            daily_table = "| Date | Labeled Samples |\n|:---|---:|\n"
            for row in daily_activity.to_dicts():
                daily_table += f"| `{row['date']}` | `{row['len']:,}` |\n"

            report_content = f"""
# Proxy Labeling Engine - Execution Report (Project Forge V5)

---

### 1. Execution Summary
{summary_table.strip()}

---

### 2. Overall Performance (Gatekeeper Passed)
{perf_table.strip()}

---

### 3. Timeframe Breakdown
This table shows which timeframes generated the most betting opportunities and their outcomes for the selected filter.

{tf_table.strip()}

---

### 4. Event Duration Analysis
This table analyzes the time it took for an event to conclude (hit a barrier or time out).

{duration_table.strip()}

---

### 5. Top 10 Busiest Days
This table lists the days with the highest number of labeled samples within the filtered period.

{daily_table.strip()}
"""
            report_path.write_text(report_content.strip())
            logging.info(f"Detailed execution report saved to: {report_path}")
        except Exception as e:
            logging.error(f"Failed to generate report: {e}", exc_info=True)


def _get_interactive_config() -> ProxyLabelConfig:
    print("\n[ Interactive Configuration Mode - Project Forge V5 (Crash Lockout) ]")
    print("Logic: Hilbert Trigger + Statistical Lockouts (Skew/Kurt/Body/Phase).")
    print("   [1] Year  - Process data for a specific year.")
    print("   [2] Month - Process data for a specific month within a specific year.")
    print("   [3] All   - Process data for all available data.")

    def get_input(
        prompt: str, default: Any, type_converter: Any, validation: Any = None
    ) -> Any:
        while True:
            val = input(f"{prompt} [{default}]: ").strip()
            if not val:
                return default
            try:
                converted_val = type_converter(val)
                if validation and not validation(converted_val):
                    print(f"  -> Invalid value. Please try again.")
                    continue
                return converted_val
            except ValueError:
                print(
                    f"  -> Invalid input type. Please enter a valid {type_converter.__name__}."
                )

    def get_bool_input(prompt: str, default: bool) -> bool:
        default_str = "yes" if default else "no"
        while True:
            val = input(f"{prompt} (yes/no) [{default_str}]: ").strip().lower()
            if not val:
                return default
            if val in ["y", "yes"]:
                return True
            if val in ["n", "no"]:
                return False
            print("  -> Invalid input. Please enter 'yes' or 'no'.")

    mode_choice = get_input(
        "   Enter choice (1, 2, or 3)", "1", str, lambda m: m in ["1", "2", "3"]
    )
    filter_mode: str = "all"
    filter_year: Optional[int] = None
    filter_month: Optional[int] = None
    current_year = dt.date.today().year

    if mode_choice == "1":
        filter_mode = "year"
        filter_year = get_input(
            f"   Enter year (e.g., {current_year})",
            current_year,
            int,
            lambda y: 1900 < y < 2100,
        )
    elif mode_choice == "2":
        filter_mode = "month"
        while True:
            year_month_str = get_input(
                f"   Enter Year/Month (e.g., {current_year}/10)",
                f"{current_year}/{dt.date.today().month}",
                str,
            )
            match = re.match(r"(\d{4})/(\d{1,2})$", year_month_str)
            if match:
                year_part, month_part = map(int, match.groups())
                if 1900 < year_part < 2100 and 1 <= month_part <= 12:
                    filter_year = year_part
                    filter_month = month_part
                    break
                else:
                    print(
                        "  -> Invalid year or month number. Please use YYYY/MM format with valid ranges."
                    )
            else:
                print("  -> Invalid format. Please use YYYY/MM format (e.g., 2023/10).")
    elif mode_choice == "3":
        filter_mode = "all"

    resume = get_bool_input("\nResume from previous run?", True)

    config = ProxyLabelConfig(
        filter_mode=filter_mode,
        filter_year=filter_year,
        filter_month=filter_month,
        resume=resume,
    )
    print("\n" + "-" * 50)
    print("Configuration Summary:")
    print(f"  - Data Filter: {config.get_filter_description()}")
    print(
        f"  - V5 Trigger:  Hilbert Amp > {TRIGGER_AMP_MULTIPLIER}x (Rolling {TRIGGER_ROLLING_WINDOW})"
    )
    print(f"  - V5 Lockouts: {list(LOCKOUT_RULES.keys())}")
    print(
        f"  - Target Rule: PT={REGIME_RULE_R4['pt']}, SL={REGIME_RULE_R4['sl']}, TD={REGIME_RULE_R4['td']}"
    )
    print(f"  - Resume: {config.resume}")
    print("-" * 50)
    if not get_bool_input("Is this configuration correct?", True):
        print("Aborted.")
        sys.exit(0)
    return config


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="[Phase 3] Create Final V5 Gatekeeper Proxy Labels."
    )
    parser.add_argument(
        "--filter-mode",
        type=str,
        default=None,
        choices=["year", "month", "all"],
        help="Filter data by 'year', specific 'month' (YYYY/MM), or process 'all'. If omitted, interactive mode starts.",
    )
    parser.add_argument(
        "--year",
        type=int,
        default=None,
        help="Year to process (used with --filter-mode=year).",
    )
    parser.add_argument(
        "--year-month",
        type=str,
        default=None,
        help="Year and month to process in YYYY/MM format (used with --filter-mode=month).",
    )
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="Disable resume and start from scratch.",
    )

    args = parser.parse_args()
    run_interactive = args.filter_mode is None

    if run_interactive:
        config = _get_interactive_config()
    else:
        filter_mode_arg = args.filter_mode
        filter_year_arg: Optional[int] = None
        filter_month_arg: Optional[int] = None

        if filter_mode_arg == "year":
            if args.year is None:
                parser.error("--year is required when --filter-mode=year")
            filter_year_arg = args.year
        elif filter_mode_arg == "month":
            if args.year_month is None:
                parser.error(
                    "--year-month YYYY/MM is required when --filter-mode=month"
                )
            match = re.match(r"(\d{4})/(\d{1,2})$", args.year_month)
            if not match:
                parser.error(
                    "Invalid format for --year-month. Use YYYY/MM (e.g., 2023/10)."
                )
            year_part, month_part = map(int, match.groups())
            if not (1900 < year_part < 2100 and 1 <= month_part <= 12):
                parser.error("Invalid year or month number in --year-month.")
            filter_year_arg = year_part
            filter_month_arg = month_part

        config = ProxyLabelConfig(
            filter_mode=filter_mode_arg,
            filter_year=filter_year_arg,
            filter_month=filter_month_arg,
            resume=not args.no_resume,
        )
        try:
            temp_engine_for_validation = ProxyLabelingEngine(config)
        except ValueError as e:
            print(f"Configuration Error: {e}")
            parser.print_help()
            sys.exit(1)

    print("\nStarting engine with the specified configuration...")
    engine = ProxyLabelingEngine(config)
    engine.run()
