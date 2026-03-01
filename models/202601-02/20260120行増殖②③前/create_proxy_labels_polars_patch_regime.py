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


# --- ▼▼▼ [フェーズ3 改造] V4ルールブックのハードコード ▼▼▼ ---

# [新規] 変動ATRフィルターの基準比率
# 2021年基準 ($1800) での ATR 5.0 相当 = 5.0 / 1800 ≒ 0.0028 (0.28%)
ATR_RATIO_THRESHOLD = 0.0028

# R4レジームに適用する単一のルール
REGIME_RULE_R4 = {
    "pt": 1.0,
    "sl": 5.0,
    "td": "1200m",
    "payoff": 1.0 / 5.0,  # 0.2
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
) -> Tuple[np.ndarray, np.ndarray]:
    """Numba JIT (Just-In-Time) compiled function to find barrier hits."""
    n_bets = len(bets_t0)
    n_ticks = len(ticks_ts)
    out_first_pt_time = np.zeros(n_bets, dtype=np.int64)
    out_first_sl_time = np.zeros(n_bets, dtype=np.int64)
    if n_ticks == 0:
        return out_first_pt_time, out_first_sl_time
    for i in prange(n_bets):
        t0 = bets_t0[i]
        t1_max = bets_t1_max[i]
        pt = bets_pt_barrier[i]
        sl = bets_sl_barrier[i]
        start_idx = np.searchsorted(ticks_ts, t0, side="left")
        first_pt_found = np.int64(0)
        first_sl_found = np.int64(0)
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
            if first_pt_found != 0 and first_sl_found != 0:
                break
        out_first_pt_time[i] = first_pt_found
        out_first_sl_time[i] = first_sl_found
    return out_first_pt_time, out_first_sl_time


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
        logging.info(f"### Phase 3: Final Labeling (ATR Regime V4 - Dynamic) ###")
        logging.info(f"Applying filter: {self.config.get_filter_description()}")
        logging.info(
            f"ATR Cutoff: Dynamic Ratio >= {ATR_RATIO_THRESHOLD:.4%} of Price (e.g., Price $2000 -> ATR>5.6)"
        )
        logging.info(
            f"R4 Rule: PT={REGIME_RULE_R4['pt']}, SL={REGIME_RULE_R4['sl']}, TD={REGIME_RULE_R4['td']}"
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
            # ★★★ [修正済み] LazyFrameのまま受け取る ★★★
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
                "Step 3: Preparing price data blueprints (separating Hive vs. Files)..."
            )
            price_components = self._load_all_price_data()
            base_price_lf = price_components["base_lf"]
            atr_lfs = price_components["atr_lfs"]

            logging.info(
                f"   -> Pre-loading {len(atr_lfs)} non-tick S2 ATR files into memory..."
            )
            try:
                atr_dfs = [lf.collect().sort("timestamp") for lf in atr_lfs]
                logging.info(
                    f"   -> Successfully pre-loaded {len(atr_dfs)} S2 ATR DataFrames."
                )
            except Exception as e:
                logging.error(f"Failed to pre-load S2 ATR files: {e}", exc_info=True)
                raise

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

            # --- [フェーズ3 改造] ルックアヘッドをハードコード ---
            max_lookahead_minutes = self._get_duration_in_minutes(
                REGIME_RULE_R4["td"]
            )  # "1200m"
            max_lookahead_delta = dt.timedelta(minutes=max_lookahead_minutes)
            # --- ▲▲▲ 改造ここまで ▲▲▲ ---

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

                # --- 修正前 ---
                # daily_bets_hive_df = unified_hive_lf.filter(
                #     pl.col("timestamp").dt.date() == current_date
                # ).collect()

                # --- 修正案: ここで "tick" を除外してしまう ---
                daily_bets_hive_df = unified_hive_lf.filter(
                    (pl.col("timestamp").dt.date() == current_date)
                    & (pl.col("timeframe") != "tick")
                ).collect()

                daily_bets_file_df = unified_file_lf.filter(
                    (pl.col("timestamp").dt.date() == current_date)
                    & (pl.col("timeframe") != "tick")
                ).collect()

                # --- ★★★ [修正] 日次データロード後に「行増殖バグ」を修正する (Coalesce) ★★★ ---
                # concat するとカラム不一致で行が増えるため、ここで GroupBy + First で凝縮する
                # これにより、メモリ使用量は「1日分」に抑えられ、PCが落ちない
                daily_bets_df = (
                    pl.concat([daily_bets_hive_df, daily_bets_file_df], how="diagonal")
                    # ▼▼▼【重要】ナノ秒・マイクロ秒・ミリ秒のズレを全て切り捨てて0にする ▼▼▼
                    .with_columns(pl.col("timestamp").dt.truncate("1s"))
                    .group_by(["timestamp", "timeframe"])
                    .agg(pl.all().drop_nulls().first())  # ここが行増殖バグの修正箇所
                    .sort("timestamp")
                )
                # ---------------------------------------------------------------

                # [検証用ログ] 重複排除前後で行数を比較し、正規表現修正が効いているか確認
                raw_count = daily_bets_hive_df.height + daily_bets_file_df.height
                dedup_count = daily_bets_df.height
                if raw_count != dedup_count:
                    logging.warning(
                        f"Duplicate rows detected and merged! Raw: {raw_count} -> Dedup: {dedup_count}. "
                        f"Diff: {raw_count - dedup_count}. (Regex fix might still be leaking specific patterns)"
                    )
                else:
                    logging.debug(
                        f"Integrity Check Passed: No duplicate rows found for {current_date}."
                    )

                if daily_bets_df.is_empty():
                    logging.debug(
                        f"No betting data found for {current_date}. Skipping partition."
                    )
                    continue

                min_ts_req = daily_bets_df["timestamp"].min()
                if min_ts_req is None:
                    logging.warning(
                        f"Could not determine minimum timestamp for {current_date}. Skipping partition."
                    )
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
                        f"No base price data found for {current_date} in the required window ({min_ts_req} to {max_ts_req}). Skipping partition."
                    )
                    continue

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
        # [Report Fix 8.2] Strict filtering to prevent 'ghost file' ingestion
        # Exclude hidden files, checkpoints, and ensure valid extensions
        discovered_paths = []
        for p in self.config.input_dir.rglob("features_*_neutralized*"):
            # 1. Must be a directory or a parquet file
            if not (p.is_dir() or p.suffix == ".parquet"):
                continue

            # 2. Exclude hidden files/dirs (start with .)
            if p.name.startswith("."):
                continue

            # 3. Exclude checkpoints/backups
            if "checkpoint" in p.name or "copy" in p.name or ".bak" in p.name:
                continue

            discovered_paths.append(p)

        if not discovered_paths:
            raise FileNotFoundError(
                f"No feature paths found in {self.config.input_dir}."
            )
        logging.info(f"  -> Found {len(discovered_paths)} feature paths.")
        return discovered_paths

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

        # [Report Fix 8.1] Remove dot from character class to prevent greedy capture of extensions
        # Old: r"features_e\d+[a-z]?_([a-zA-Z0-9\.]+)(?:_neutralized)?"
        # New: r"features_e\d+[a-z]?_([a-zA-Z0-9]+)"
        timeframe_pattern = re.compile(r"features_e\d+[a-z]?_([a-zA-Z0-9]+)")
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
        """(変更なし) S2の価格データ(Tick)と、各時間足の「価格単位ATR」を読み込む。"""
        price_dir = self.config.price_data_source_dir / "feature_value_a_vast_universeC"
        tick_dir = price_dir / "features_e1c_tick"
        if not tick_dir.exists():
            raise FileNotFoundError(f"Master price directory not found: {tick_dir}")
        logging.info(
            f"   -> Lazily scanning '{tick_dir}' as the master price source (S2 Tick)."
        )
        base_lf = (
            pl.scan_parquet(str(tick_dir / "**/*.parquet"))
            .select("timestamp", "close", "high", "low")
            .with_columns(pl.col("timestamp").cast(pl.Datetime("us", "UTC")))
            .unique("timestamp", keep="first", maintain_order=True)
        )
        atr_files = list(price_dir.glob("features_e1c_*.parquet"))
        if not atr_files:
            raise FileNotFoundError(
                f"No ATR parquet files (standard or dedicated tick) found in {price_dir} or specified path."
            )
        timeframe_pattern = re.compile(
            r"features_e1c_([a-zA-Z0-9\.]+)(?:_atr_only_tick_fixed)?\.parquet"
        )
        all_atr_lfs = []
        logging.info(
            f"   -> Processing {len(atr_files)} potential S2 ATR source files (S2 Non-Tick)."
        )
        SOURCE_ATR_COLUMN_NAME = "e1c_atr_21"
        for f_path in atr_files:
            is_dedicated_tick_file = "_atr_only_tick_fixed" in f_path.name
            match = timeframe_pattern.search(f_path.name)
            if not match:
                logging.warning(
                    f"   -> Could not parse timeframe from {f_path.name}. Skipping."
                )
                continue
            timeframe = match.group(1)
            if is_dedicated_tick_file:
                timeframe = "tick"
            target_atr_name = f"e1c_atr_21_{timeframe}"
            lf_original = pl.scan_parquet(str(f_path))
            schema_names = lf_original.collect_schema().names()
            if is_dedicated_tick_file:
                expected_tick_atr_col = "e1c_tick_atr_only_atr_21"
                time_col_name = None
                if "datetime" in schema_names:
                    time_col_name = "datetime"
                elif "timestamp" in schema_names:
                    time_col_name = "timestamp"
                if time_col_name and expected_tick_atr_col in schema_names:
                    processed_lf = lf_original.select(
                        [
                            pl.col(time_col_name)
                            .cast(pl.Datetime("us", "UTC"))
                            .alias("timestamp"),
                            pl.col(expected_tick_atr_col).alias(target_atr_name),
                        ]
                    )
                    all_atr_lfs.append(processed_lf)
                    logging.info(
                        f"   -> Prepared ATR blueprint for timeframe '{timeframe}' from dedicated file {f_path.name} ('{time_col_name}' processed)"
                    )
                else:
                    logging.warning(
                        f"   -> Dedicated tick ATR file {f_path.name} does not contain required columns ('datetime' or 'timestamp', and '{expected_tick_atr_col}'). Found: {schema_names}"
                    )
            else:
                lf = lf_original.with_columns(
                    pl.col("timestamp").cast(pl.Datetime("us", "UTC"))
                )
                if SOURCE_ATR_COLUMN_NAME in schema_names:
                    all_atr_lfs.append(
                        lf.select(["timestamp", SOURCE_ATR_COLUMN_NAME]).rename(
                            {SOURCE_ATR_COLUMN_NAME: target_atr_name}
                        )
                    )
                    logging.info(
                        f"   -> Prepared ATR blueprint for timeframe '{timeframe}' from {f_path.name} (Loaded '{SOURCE_ATR_COLUMN_NAME}' -> Renamed to '{target_atr_name}')"
                    )
                else:
                    logging.warning(
                        f"   -> Required ATR column '{SOURCE_ATR_COLUMN_NAME}' not found in {f_path.name}. Skipping."
                    )
        if not all_atr_lfs:
            raise ValueError(
                "FATAL: No valid ATR columns were extracted from any price files."
            )
        logging.info(
            f"   -> Prepared {len(all_atr_lfs)} S2 ATR blueprints (as LazyFrames)."
        )
        return {"base_lf": base_lf, "atr_lfs": all_atr_lfs}

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
        [フェーズ3: Numba + 変動ATR版]
        修正済み:
        1. .select で timestamp 重複を回避
        2. 'tick' 時間足をエントリー対象から除外 (Machine Gun Entry 防止)
        """
        if daily_bets_df.is_empty():
            return None

        if not NUMBA_AVAILABLE:
            logging.error(
                "Numba is required for labeling but not found. Skipping labeling."
            )
            return None

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
        for timeframe_tuple, group_df in daily_bets_df.group_by("timeframe"):
            timeframe = timeframe_tuple[0]
            if timeframe is None or group_df.is_empty():
                continue

            # --- ★★★ [修正] Tickデータ自体でのエントリーを禁止する ★★★ ---
            # Tickデータは判定用であり、エントリー足ではないためスキップする
            if timeframe == "tick":
                continue
            # -------------------------------------------------------------

            rule_pt = REGIME_RULE_R4["pt"]
            rule_sl = REGIME_RULE_R4["sl"]
            rule_td_str = REGIME_RULE_R4["td"]
            rule_payoff = REGIME_RULE_R4["payoff"]

            target_duration_minutes = self._get_duration_in_minutes(rule_td_str)
            bar_duration_minutes = self._get_bar_duration_minutes(timeframe)
            t1_max_expr: pl.Expr

            # bar_duration_minutes が 0 (計算不能) の場合はスキップ
            if bar_duration_minutes == 0:
                logging.warning(
                    f"Skipping timeframe {timeframe} due to zero bar duration calculation."
                )
                continue

            lookahead_bars = max(
                1, int(round(target_duration_minutes / bar_duration_minutes))
            )
            t1_max_expr = pl.col("timestamp") + pl.duration(
                minutes=lookahead_bars * bar_duration_minutes
            )

            atr_col_name = f"e1c_atr_21_{timeframe}"
            if atr_col_name not in price_window_df.columns:
                logging.warning(
                    f"Required ATR column '{atr_col_name}' not found in price data. Skipping timeframe '{timeframe}'."
                )
                continue

            # 元の特徴量カラムリストを作成（timestamp, close は手動で扱うため除外）
            original_cols = [
                c for c in group_df.columns if c not in ["timestamp", "close"]
            ]

            bets_with_price_df = group_df.join_asof(
                price_window_df.select(
                    ["timestamp", "close", "high", "low", atr_col_name]
                ),
                on="timestamp",
            ).filter(pl.col(atr_col_name).is_not_null())

            if bets_with_price_df.is_empty():
                continue

            bets_df_with_atr = bets_with_price_df.with_columns(
                pl.col(atr_col_name).alias("atr_value")
            )

            # --- [重要] レジームフィルタリング (変動ATR) ---
            bets_df_filtered = bets_df_with_atr.filter(
                pl.col("atr_value") >= (pl.col("close") * ATR_RATIO_THRESHOLD)
            )

            if bets_df_filtered.is_empty():
                continue

            # R4ルール適用 & カラム整理
            bets_df = bets_df_filtered.select(
                pl.col("timestamp").alias("t0"),
                (pl.col("close") + pl.col("atr_value") * rule_pt).alias("pt_barrier"),
                (pl.col("close") - pl.col("atr_value") * rule_sl).alias("sl_barrier"),
                t1_max_expr.alias("t1_max"),
                pl.col("atr_value"),
                pl.col("close"),
                pl.col(original_cols),  # その他の特徴量を展開
            )

            if bets_df.is_empty():
                continue

            try:
                bets_t0_np = bets_df["t0"].cast(pl.Int64).to_numpy()
                bets_t1_max_np = bets_df["t1_max"].cast(pl.Int64).to_numpy()
                bets_pt_np = bets_df["pt_barrier"].to_numpy(writable=True)
                bets_sl_np = bets_df["sl_barrier"].to_numpy(writable=True)
            except Exception as e:
                logging.error(
                    f"Failed to convert bets data to Numpy arrays for timeframe {timeframe}: {e}"
                )
                continue

            first_pt_time_np, first_sl_time_np = _numba_find_hits(
                bets_t0_np,
                bets_t1_max_np,
                bets_pt_np,
                bets_sl_np,
                ticks_ts_np,
                ticks_high_np,
                ticks_low_np,
            )

            final_group_df = (
                bets_df.with_columns(
                    pl.Series("first_pt_time_int", first_pt_time_np),
                    pl.Series("first_sl_time_int", first_sl_time_np),
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
                )
                .drop("first_pt_time_int", "first_sl_time_int")
            )

            labeled_group = final_group_df.with_columns(
                t1=pl.when(
                    (pl.col("first_pt_time").is_not_null())
                    & (
                        (pl.col("first_sl_time").is_null())
                        | (pl.col("first_pt_time") <= pl.col("first_sl_time"))
                    )
                )
                .then(pl.col("first_pt_time"))
                .when(pl.col("first_sl_time").is_not_null())
                .then(pl.col("first_sl_time"))
                .otherwise(pl.col("t1_max")),
                label=pl.when(
                    (pl.col("first_pt_time").is_not_null())
                    & (
                        (pl.col("first_sl_time").is_null())
                        | (pl.col("first_pt_time") <= pl.col("first_sl_time"))
                    )
                )
                .then(pl.lit(1, dtype=pl.Int8))
                .when(pl.col("first_sl_time").is_not_null())
                .then(pl.lit(-1, dtype=pl.Int8))
                .otherwise(pl.lit(0, dtype=pl.Int8)),
                payoff_ratio=pl.lit(rule_payoff, dtype=pl.Float32),
                sl_multiplier=pl.lit(rule_sl, dtype=pl.Float32),
                pt_multiplier=pl.lit(rule_pt, dtype=pl.Float32),
                direction=pl.lit(1, dtype=pl.Int8),
            ).rename({"t0": "timestamp"})

            labeled_chunks.append(
                labeled_group.drop(
                    [
                        "pt_barrier",
                        "sl_barrier",
                        "t1_max",
                        "first_pt_time",
                        "first_sl_time",
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
            ).select(["timeframe", "label", "event_duration_seconds", "date"])
            self.report_data.extend(report_df.to_dicts())
        except Exception as e:
            logging.warning(f"Error collecting report data for {current_date}: {e}")

    def _generate_report(self):
        logging.info("Generating detailed execution report...")
        cfg = self.config
        report_filename = "execution_report_regime_v4.md"
        report_path = cfg.output_dir / report_filename

        rule = REGIME_RULE_R4

        summary_table = f"""
| Item | Value |
|:---|:---|
| **Execution Timestamp** | `{cfg.execution_start_time}` |
| **Script Path** | `{" / ".join(Path(__file__).parts[-4:])}` |
| **Data Filter Applied** | `{cfg.get_filter_description()}` |
| **Labeling Strategy** | `ATR Regime V4 (R4 Only)` |
| **ATR Cutoff** | `Dynamic: > {ATR_RATIO_THRESHOLD:.4%} of Price` |
| **Target Duration (R4)** | `{rule["td"]}` |
| **Payoff Ratio (PT/SL) (R4)** | `{rule["payoff"]:.2f}` (`PT mult: {rule["pt"]}`, `SL mult: {rule["sl"]}`) |
"""

        if not self.report_data:
            logging.warning("No data available for report generation.")
            report_content = f"# Proxy Labeling Engine - Execution Report\n\n**Filter Applied: {cfg.get_filter_description()}**\n\n**No samples were processed for duration '{rule['td']}' with the specified filter.**"
            report_path.write_text(report_content)
            return

        try:
            df = pl.from_dicts(self.report_data)
            total = len(df)
            pt = df.filter(pl.col("label") == 1).height
            sl = df.filter(pl.col("label") == -1).height
            to = df.filter(pl.col("label") == 0).height
            scale_pos_weight = sl / pt if pt > 0 else 1.0
            perf_table = f"""
| Metric | Value |
|:---|:---|
| **Total Labeled Samples (R4 only)** | `{total:,}` |
| **(+) Profit-Take (PT)** | `{pt:,}` (`{pt / total:.2%}`) |
| **(-) Stop-Loss (SL)** | `{sl:,}` (`{sl / total:.2%}`) |
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
# Proxy Labeling Engine - Execution Report (Regime V4) 統

---

### 1. Execution Summary
{summary_table.strip()}

---

### 2. Overall Performance (R4 Only) 投
{perf_table.strip()}

---

### 3. Timeframe Breakdown (R4 Only) 葡
This table shows which timeframes generated the most betting opportunities and their outcomes for the selected filter.

{tf_table.strip()}

---

### 4. Event Duration Analysis (R4 Only) 竢ｳ
This table analyzes the time it took for an event to conclude (hit a barrier or time out).

{duration_table.strip()}

---

### 5. Top 10 Busiest Days (R4 Only) 欄ｸThis table lists the days with the highest number of labeled samples within the filtered period.

{daily_table.strip()}
"""
            report_path.write_text(report_content.strip())
            logging.info(f"Detailed execution report saved to: {report_path}")
        except Exception as e:
            logging.error(f"Failed to generate report: {e}", exc_info=True)


def _get_interactive_config() -> ProxyLabelConfig:
    print("\n[ Interactive Configuration Mode - ATR Regime V4 (Dynamic Ratio) ]")
    print("ATR rules are based on Price Ratio. Please select data filter mode.")
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
        f"  - ATR Cutoff: Dynamic Ratio < {ATR_RATIO_THRESHOLD:.4%} of Price (Discarded)"
    )
    print(
        f"  - R4 Rule: PT={REGIME_RULE_R4['pt']}, SL={REGIME_RULE_R4['sl']}, TD={REGIME_RULE_R4['td']}"
    )
    print(f"  - Resume: {config.resume}")
    print("-" * 50)
    if not get_bool_input("Is this configuration correct?", True):
        print("Aborted.")
        sys.exit(0)
    return config


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="[Phase 3] Create Final ATR-Regime (V4) Proxy Labels."
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
