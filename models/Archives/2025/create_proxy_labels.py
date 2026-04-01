# /workspace/models/optimal_parameters/create_proxy_labels.py
# [Gemini 修正版: 非Hiveデータ(単一Parquet)を先にメモリロードし、日次ループでのI/Oを削減]
# [Gemini 修正版: Ruff F821 (args.no_resume) および E731 (lambda) エラーを修正]
# [ユーザー修正依頼: drop_nulls("close") を削除し、意図しないデータ削除を回避]

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


# --- Configuration (V7の柔軟な設定) ---
@dataclass
class ProxyLabelConfig:
    """Config for creating a context-adaptive, labeled subset for proxy model training."""

    input_dir: Path = S5_NEUTRALIZED_ALPHA_SET
    price_data_source_dir: Path = S2_FEATURES_FIXED
    output_dir: Path = S6_LABELED_DATASET
    target_duration: str = "1m"
    profit_take_multiplier: float = 0.8
    stop_loss_multiplier: float = 0.2
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
        duration_match = re.match(r"^(\d+)([ms])?$", cfg.target_duration)
        if not duration_match:
            raise ValueError(
                f"Invalid target_duration format: {cfg.target_duration}. Expected format like '300m', '90s', or '60' (assumed as minutes)."
            )
        value_str, unit = duration_match.groups()
        if unit is None:
            cfg.target_duration += "m"

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
        logging.info(
            f"### Phase 1, Script 1: Create Proxy Labels (Context-Adaptive Engine) for {self.config.target_duration} ###"
        )
        logging.info(f"Applying filter: {self.config.get_filter_description()}")
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
            unified_hive_lf, unified_file_df = self._build_unified_lazyframe(
                all_feature_paths
            )

            if (
                unified_hive_lf.collect_schema().names() == []
                and unified_file_df.is_empty()
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
                # ★★★ 修正: ATR DF の collect 時にソートを追加 ★★★
                atr_dfs = [lf.collect().sort("timestamp") for lf in atr_lfs]
                logging.info(
                    f"   -> Successfully pre-loaded {len(atr_dfs)} S2 ATR DataFrames."
                )
            except Exception as e:
                logging.error(f"Failed to pre-load S2 ATR files: {e}", exc_info=True)
                raise

            logging.info("Step 4: Discovering daily partitions for processing...")
            partitions_df_hive = self._discover_partitions(unified_hive_lf)
            partitions_df_file = self._discover_partitions(unified_file_df.lazy())

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

            # # --- MODIFICATION START: Add start date filter ---
            # known_start_date = dt.date(
            #     2021, 8, 1
            # )  # Define the known start date of reliable S2 data
            # logging.info(
            #     f"Applying start date filter: Processing partitions on or after {known_start_date}..."
            # )
            # partitions_df_filtered = partitions_df.filter(
            #     pl.col("date") >= known_start_date
            # )
            # num_original = len(partitions_df)
            # num_filtered = len(
            #     partitions_df_filtered
            # )  # Recalculate based on filtered df

            # if num_filtered < num_original:
            #     logging.info(
            #         f"   -> Filtered out {num_original - num_filtered} partitions before {known_start_date}."
            #     )

            # if partitions_df_filtered.is_empty():
            #     logging.warning(
            #         f"No partitions remaining after applying start date filter ({known_start_date}). Exiting."
            #     )
            #     self._log_final_summary()
            #     self._generate_report()
            #     return
            # # --- MODIFICATION END ---

            logging.info(
                "Step 5: Starting daily processing loop (S2/S5 Hive data scanned per-loop)..."
            )

            max_lookahead_minutes = self._get_duration_in_minutes(cfg.target_duration)
            max_lookahead_delta = dt.timedelta(minutes=max_lookahead_minutes)

            # # --- Use the filtered DataFrame and its length for the loop and tqdm ---
            # total_partitions = len(
            #     partitions_df_filtered
            # )  # ★★★ CORRECT: Use length of filtered DataFrame ★★★
            # # logging.info(
            # #     f"Processing {total_partitions} partitions from {known_start_date} onwards..."
            # # )  # ★★★ CORRECT: Log uses filtered count ★★★

            # Wrap the FILTERED iterator with tqdm, using the CORRECT total
            total_partitions = len(
                partitions_df
            )  # ★★★ フィルター前の total を再定義 ★★★
            for row in tqdm(
                partitions_df.iter_rows(named=True),  # ★★★ _filtered を削除 ★★★
                total=total_partitions,  # ★★★ フィルター前の total を使用 ★★★
                desc="Processing Partitions",
                unit="partition",
            ):
                # --- End loop setup correction ---

                current_date = row["date"]
                year, month, day = (
                    current_date.year,
                    current_date.month,
                    current_date.day,
                )

                output_partition_dir = (
                    cfg.output_dir / f"year={year}/month={month}/day={day}"
                )
                # Check if output already exists for this partition and if resume is enabled
                if cfg.resume and output_partition_dir.exists():
                    # If resuming, simply skip this partition
                    logging.debug(
                        f"Resuming: Output already exists for {current_date}. Skipping partition."
                    )
                    continue  # Skip to the next partition

                # --- Collect data for the current partition ---
                # Collect from Hive LazyFrame
                daily_bets_hive_df = unified_hive_lf.filter(
                    pl.col("timestamp").dt.date() == current_date
                ).collect()

                # Filter from the pre-loaded File DataFrame
                daily_bets_file_df = unified_file_df.filter(
                    pl.col("timestamp").dt.date() == current_date
                )

                # --- ▼▼▼ 修正ブロック 1 ▼▼▼ ---
                # Concatenate Hive and File data for the current day
                daily_bets_df = pl.concat(
                    [daily_bets_hive_df, daily_bets_file_df], how="diagonal"
                ).sort("timestamp")  # ★★★ 修正: .sort("timestamp") を追加 ★★★
                # --- ▲▲▲ 修正ブロック 1 ▲▲▲ ---

                # Skip if no betting data for this day
                if daily_bets_df.is_empty():
                    logging.debug(
                        f"No betting data found for {current_date}. Skipping partition."
                    )
                    continue

                # --- Prepare price window for this partition ---
                # Determine required time range for price data (min timestamp to max lookahead + buffer)
                min_ts_req = daily_bets_df["timestamp"].min()
                if (
                    min_ts_req is None
                ):  # Should not happen if daily_bets_df is not empty
                    logging.warning(
                        f"Could not determine minimum timestamp for {current_date}. Skipping partition."
                    )
                    continue

                # Calculate the maximum timestamp needed for lookahead + a buffer (e.g., 2 days)
                max_ts_req = min_ts_req + max_lookahead_delta + dt.timedelta(days=2)

                # --- ▼▼▼ 修正ブロック 2 ▼▼▼ ---
                # Collect required price data from the base LazyFrame (tick data)
                price_window_df = (
                    base_price_lf.filter(
                        pl.col("timestamp").is_between(min_ts_req, max_ts_req)
                    )
                    .collect()
                    .sort("timestamp")  # ★★★ 修正: .sort("timestamp") を追加 ★★★
                )
                # --- ▲▲▲ 修正ブロック 2 ▲▲▲ ---

                if price_window_df.is_empty():
                    logging.warning(
                        f"No base price data found for {current_date} in the required window ({min_ts_req} to {max_ts_req}). Skipping partition."
                    )
                    continue

                # Join relevant ATR data from pre-loaded non-tick DataFrames
                # Iterate through the pre-loaded atr_dfs and join applicable ones
                for atr_df in atr_dfs:
                    # Filter ATR data for the relevant time window before joining
                    atr_df_small = atr_df.filter(
                        pl.col("timestamp").is_between(min_ts_req, max_ts_req)
                    )
                    if not atr_df_small.is_empty():
                        # Use join_asof to merge ATR data based on the closest preceding timestamp
                        # price_window_df (left) はソート済み
                        # atr_df (right) も collect 時にソート済み
                        price_window_df = price_window_df.join_asof(
                            atr_df_small,
                            on="timestamp",  # .sort("timestamp") は不要
                        )

                # Forward fill missing ATR/price values, then backward fill remaining NaNs at the start
                price_window_df = price_window_df.fill_null(
                    strategy="forward"
                ).fill_null(strategy="backward")

                # --- Calculate labels for the current partition's data ---
                daily_labeled_df = self._calculate_labels_for_batch(
                    daily_bets_df, price_window_df
                )

                # --- Write results if labels were generated ---
                if daily_labeled_df is not None and not daily_labeled_df.is_empty():
                    self._update_label_counts(daily_labeled_df)
                    self._collect_report_data(daily_labeled_df, current_date)

                    output_partition_dir.mkdir(parents=True, exist_ok=True)
                    # Write the labeled data to the partition directory
                    daily_labeled_df.write_parquet(
                        output_partition_dir / "data.parquet", compression="zstd"
                    )

                # --- Clean up memory for the current loop iteration ---
                del daily_bets_df, price_window_df, daily_labeled_df
                del daily_bets_hive_df, daily_bets_file_df  # Ensure these are deleted
                gc.collect()

            # --- Loop finished ---
            self._log_final_summary()  # Log the final summary counts
            self._generate_report()  # Generate the markdown report

        except Exception as e:
            logging.error(f"A critical error occurred: {e}", exc_info=True)
            raise  # Re-raise the exception after logging

    # =========================================================================
    # コアロジック (変更なし)
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
    ) -> Tuple[pl.LazyFrame, pl.DataFrame]:
        """
        [★★★ 修正版 ★★★]
        S5の特徴量データを、Hiveパーティション(Lazy)と
        単一ファイル(Eager, メモリにロード)に分離する。
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
                unified_hive_lf = pl.concat(all_lazy_frames_hive, how="diagonal").sort(
                    "timestamp"
                )
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

        unified_file_df = pl.DataFrame()
        if all_lazy_frames_file:
            try:
                logging.info(
                    f"   -> Pre-loading {len(all_lazy_frames_file)} S5 non-tick files into memory..."
                )
                unified_file_df = (
                    pl.concat(all_lazy_frames_file, how="diagonal")
                    .collect()
                    .sort("timestamp")
                )
                logging.info(
                    f"   -> Successfully pre-loaded {len(unified_file_df):,} S5 non-tick bet records."
                )
            except pl.exceptions.ComputeError as e:
                if "cannot concat empty list" in str(e).lower():
                    logging.warning(
                        "S5 non-tick file concatenation resulted in empty data."
                    )
                else:
                    raise e
            except Exception as e:
                logging.error(
                    f"Failed to pre-load S5 non-tick files (OutOfMemory?): {e}",
                    exc_info=True,
                )
                raise
        else:
            logging.warning("No S5 non-tick (single file) data found for the filter.")

        return unified_hive_lf, unified_file_df

    def _rename_features(
        self, lf: pl.LazyFrame, timeframe_suffix: str
    ) -> Optional[pl.LazyFrame]:
        """Helper to apply feature renaming and timestamp casting."""
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
        [修正版] S2の価格データ(Tick)と、
        各時間足の「価格単位ATR (e1c_atr_21)」のみを明示的に読み込む。
        """
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

        # S2 (engine_1_C) が出力した全 .parquet ファイルを検索
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

        # ★★★ 修正箇所 ★★★
        # 読み込むべき「価格単位」のATRカラム名を明示的に定義
        # engine_1_C はこの名前で価格単位ATRを保存している
        SOURCE_ATR_COLUMN_NAME = "e1c_atr_21"
        # ★★★ 修正箇所 ★★★

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

            # ★★★ 修正箇所 ★★★
            # このスクリプト内で使用する、時間足サフィックス付きの
            # 「ターゲット（最終的な）」カラム名を定義
            target_atr_name = f"e1c_atr_21_{timeframe}"
            # ★★★ 修正箇所 ★★★

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
                # --- ▼▼▼ 修正ブロック ▼▼▼ ---
                # (修正前は "e1c_atr_21" を含むカラムを曖昧に検索していた)

                lf = lf_original.with_columns(
                    pl.col("timestamp").cast(pl.Datetime("us", "UTC"))
                )

                # 読み込むべき「価格単位」のカラム (e1c_atr_21) が
                # ファイル内に存在するかを明示的に確認
                if SOURCE_ATR_COLUMN_NAME in schema_names:
                    # 存在する場合、そのカラム (e1c_atr_21) を選択し、
                    # ターゲット名 (e1c_atr_21_H1など) にリネームする
                    all_atr_lfs.append(
                        lf.select(["timestamp", SOURCE_ATR_COLUMN_NAME]).rename(
                            {SOURCE_ATR_COLUMN_NAME: target_atr_name}
                        )
                    )
                    logging.info(
                        f"   -> Prepared ATR blueprint for timeframe '{timeframe}' from {f_path.name} (Loaded '{SOURCE_ATR_COLUMN_NAME}' -> Renamed to '{target_atr_name}')"
                    )
                else:
                    # 目的の「価格単位」カラムが見つからない場合は警告し、スキップする
                    # (これにより e1c_atr_pct_21 が誤って読み込まれるのを防ぐ)
                    logging.warning(
                        f"   -> Required ATR column '{SOURCE_ATR_COLUMN_NAME}' not found in {f_path.name}. Skipping."
                    )
                # --- ▲▲▲ 修正ブロック ▲▲▲ ---

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
            return pl.DataFrame({"date": []})
        return (
            unified_lf.select(pl.col("timestamp").dt.date().alias("date"))
            .unique()
            .collect()
            .sort("date")
        )

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

    def _calculate_labels_for_batch(
        self, daily_bets_df: pl.DataFrame, price_window_df: pl.DataFrame
    ) -> pl.DataFrame | None:
        cfg = self.config
        if daily_bets_df.is_empty():
            return None

        # --- ▼▼▼ 除外リストを定義 ▼▼▼ ---
        EXCLUDE_TIMEFRAMES = [
            "MN",
            "W1",
            "D1",
            "H12",
            "H6",
            "H4",
            "H30",
            "M0.5",
            "tick",
            # "H1",
            # "M15",
            # "M8",
            # "M5",
            # "M3",
            # "M1",
        ]  # ここに除外したい時間足を追加
        # --- ▲▲▲ 除外リストここまで ▲▲▲ ---

        labeled_chunks = []
        # daily_bets_df は run() でソート済み
        for timeframe_tuple, group_df in daily_bets_df.group_by("timeframe"):
            timeframe = timeframe_tuple[0]
            if timeframe is None or group_df.is_empty():
                continue

            # --- ▼▼▼ ここで除外チェック ▼▼▼ ---
            if timeframe in EXCLUDE_TIMEFRAMES:
                logging.debug(f"Skipping labeling for excluded timeframe: {timeframe}")
                continue  # この時間足の処理をスキップ
            # --- ▲▲▲ 除外チェックここまで ▲▲▲ ---

            target_duration_minutes = self._get_duration_in_minutes(cfg.target_duration)
            bar_duration_minutes = self._get_bar_duration_minutes(timeframe)
            t1_max_expr: pl.Expr
            if timeframe == "tick":
                t1_max_expr = pl.col("timestamp") + pl.duration(
                    minutes=target_duration_minutes
                )
            else:
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
                if timeframe != "tick":
                    logging.warning(
                        f"Required ATR column '{atr_col_name}' not found in price data. Skipping timeframe '{timeframe}'."
                    )
                continue

            original_cols = group_df.columns

            # ATR/Price 結合 (これは join_asof で正しい)
            # group_df (left) はソート済み
            # price_window_df (right) も run() でソート済み
            bets_with_price_df = group_df.join_asof(
                price_window_df.select(
                    ["timestamp", "close", "high", "low", atr_col_name]
                ),  # .sort("timestamp") は不要 (run()でソート済み)
                on="timestamp",
            ).filter(pl.col(atr_col_name).is_not_null())

            if bets_with_price_df.is_empty():
                continue

            # バリアと時間制限 (t1_max) を計算
            bets_df = bets_with_price_df.select(
                pl.col("timestamp").alias("t0"),
                (
                    pl.col("close") + pl.col(atr_col_name) * cfg.profit_take_multiplier
                ).alias("pt_barrier"),
                (
                    pl.col("close") - pl.col(atr_col_name) * cfg.stop_loss_multiplier
                ).alias("sl_barrier"),
                t1_max_expr.alias("t1_max"),
                pl.col(atr_col_name).alias("atr_value"),
                pl.col(original_cols).exclude("timestamp"),
            )

            # --- ▼▼▼ 修正ブロック: join_where を使用した区間結合 ▼▼▼ ---
            # (レポートで推奨された正しい実装)

            # price_window_df (ティックデータ) は run() でソート済み
            ticks_df = price_window_df.select(["timestamp", "high", "low"])

            # hits_df: ベット(bets_df)を起点とし、
            # [t0, t1_max] の区間に入る全てのティックを結合する
            hits_df = (
                bets_df.join_where(
                    ticks_df,
                    # 述語1: ティックはベット開始以降
                    pl.col("timestamp") >= pl.col("t0"),
                    # 述語2: ティックは時間バリア以前
                    pl.col("timestamp") <= pl.col("t1_max"),
                )
                .with_columns(
                    # ヒット判定
                    pt_hit_time=pl.when(pl.col("high") >= pl.col("pt_barrier")).then(
                        pl.col("timestamp")
                    ),
                    sl_hit_time=pl.when(pl.col("low") <= pl.col("sl_barrier")).then(
                        pl.col("timestamp")
                    ),
                )
                .group_by("t0")  # 元のベット(t0)ごとに集計
                .agg(
                    # 最初にヒットした時間を探す
                    pl.col("pt_hit_time").min().alias("first_pt_time"),
                    pl.col("sl_hit_time").min().alias("first_sl_time"),
                )
            )
            # --- ▲▲▲ 修正ブロック ▲▲▲ ---

            # Join hit times back to the original bets_df
            # (hits_dfにはヒットしなかったベット(TO)が含まれないため、left joinが必須)
            final_group_df = bets_df.join(hits_df, on="t0", how="left")

            # Determine final label and t1
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
                payoff_ratio=pl.lit(
                    cfg.profit_take_multiplier / cfg.stop_loss_multiplier,
                    dtype=pl.Float32,
                ),
                sl_multiplier=pl.lit(cfg.stop_loss_multiplier, dtype=pl.Float32),
                pt_multiplier=pl.lit(cfg.profit_take_multiplier, dtype=pl.Float32),
                direction=pl.lit(1, dtype=pl.Int8),
            ).rename({"t0": "timestamp"})

            # Drop intermediate columns
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
                f"### Context-Adaptive Labeling COMPLETED (Filter: {self.config.get_filter_description()}) ###\n"
                f"The '{self.config.target_duration}' version of the 'Problem Set' is ready at: {self.config.output_dir}\n"
                "  - No samples matched the specified filter.\n" + "=" * 60
            )
            logging.info(summary)
            return
        pos = self.label_counts.get(1, 0)
        neg = self.label_counts.get(-1, 0)
        scale_pos_weight = neg / pos if pos > 0 else 1.0
        summary = (
            "\n" + "=" * 60 + "\n"
            f"### Context-Adaptive Labeling COMPLETED (Filter: {self.config.get_filter_description()}) ###\n"
            f"The '{self.config.target_duration}' version of the 'Problem Set' is ready at: {self.config.output_dir}\n"
            f"  - Total Labeled Samples: {total_samples}\n"
            f"  - (+) Profit-Take: {pos}, (-) Stop-Loss: {neg}, (0) Timed-Out: {self.label_counts.get(0, 0)}\n"
            f"  - Calculated `scale_pos_weight` for next step: {scale_pos_weight:.4f}\n"
            + "="
            * 60
        )
        logging.info(summary)

    def _collect_report_data(self, df: pl.DataFrame, current_date: dt.date):
        """Collects data from the daily labeled dataframe for the final report."""
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
        """Generates a detailed markdown report of the execution results."""
        logging.info("Generating detailed execution report...")
        cfg = self.config
        report_filename = "execution_report.md"
        report_path = cfg.output_dir / report_filename
        if not self.report_data:
            logging.warning("No data available for report generation.")
            report_content = f"# Proxy Labeling Engine - Execution Report\n\n**Filter Applied: {cfg.get_filter_description()}**\n\n**No samples were processed for duration '{cfg.target_duration}' with the specified filter. The output is empty.**"
            report_path.write_text(report_content)
            return
        try:
            df = pl.from_dicts(self.report_data)
            summary_table = f"""
| Item | Value |
|:---|:---|
| **Execution Timestamp** | `{cfg.execution_start_time}` |
| **Script Path** | `{" / ".join(Path(__file__).parts[-4:])}` |
| **Data Filter Applied** | `{cfg.get_filter_description()}` |
| **Target Duration** | `{cfg.target_duration}` |
| **Payoff Ratio (PT/SL)** | `{cfg.profit_take_multiplier / cfg.stop_loss_multiplier:.2f}` (`PT mult: {cfg.profit_take_multiplier}`, `SL mult: {cfg.stop_loss_multiplier}`) |
"""
            total = len(df)
            pt = df.filter(pl.col("label") == 1).height
            sl = df.filter(pl.col("label") == -1).height
            to = df.filter(pl.col("label") == 0).height
            scale_pos_weight = sl / pt if pt > 0 else 1.0
            perf_table = f"""
| Metric | Value |
|:---|:---|
| **Total Labeled Samples** | `{total:,}` |
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
# Proxy Labeling Engine - Execution Report 統

---

### 1. Execution Summary
{summary_table.strip()}

---

### 2. Overall Performance 投
{perf_table.strip()}

---

### 3. Timeframe Breakdown 葡
This table shows which timeframes generated the most betting opportunities and their outcomes for the selected filter.

{tf_table.strip()}

---

### 4. Event Duration Analysis 竢ｳ
This table analyzes the time it took for an event to conclude (hit a barrier or time out).

{duration_table.strip()}

---

### 5. Top 10 Busiest Days (within filter) 欄ｸThis table lists the days with the highest number of labeled samples within the filtered period.

{daily_table.strip()}
"""
            report_path.write_text(report_content.strip())
            logging.info(f"Detailed execution report saved to: {report_path}")
        except Exception as e:
            logging.error(f"Failed to generate report: {e}", exc_info=True)


# --- V7のインタラクティブモード（年/月/全期間対応） ---
def _get_interactive_config() -> ProxyLabelConfig:
    """Gets configuration from the user interactively with new filter options."""
    print("\n[ Interactive Configuration Mode ]")
    print("Enter parameters. Press Enter to use the default value shown in [].\n")

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

    print("1. Select data filter mode:")
    print("   [1] Year  - Process data for a specific year.")
    print("   [2] Month - Process data for a specific month within a specific year.")
    print("   [3] All   - Process data for all available data.")
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

    # --- ★★★ ここが修正箇所 (E731) ★★★ ---
    # (変更前) duration_valid = lambda d: bool(re.match(r"^(\d+)([ms])?$", d))

    # (変更後) def を使用
    def duration_valid(d: str) -> bool:
        """Checks if the duration string is valid (e.g., '300m', '90s', '60')."""
        return bool(re.match(r"^(\d+)([ms])?$", d))

    # --- ★★★ 修正ここまで ★★★ ---

    duration = get_input(
        "\n2. Target lookahead duration (e.g., '300m', '90s', or '60' [assumed minutes])",
        "300m",
        str,
        duration_valid,  # validationに関数を渡す
    )
    if duration.isdigit():
        duration += "m"
    pt_mult = get_input("3. Profit-Take ATR multiplier", 2.0, float, lambda p: p > 0)
    sl_mult = get_input("4. Stop-Loss ATR multiplier", 1.0, float, lambda s: s > 0)
    resume = get_bool_input("5. Resume from previous run?", True)
    config = ProxyLabelConfig(
        target_duration=duration,
        filter_mode=filter_mode,
        filter_year=filter_year,
        filter_month=filter_month,
        profit_take_multiplier=pt_mult,
        stop_loss_multiplier=sl_mult,
        resume=resume,
    )
    print("\n" + "-" * 50)
    print("Configuration Summary:")
    print(f"  - Data Filter: {config.get_filter_description()}")
    print(f"  - Target Duration: {config.target_duration}")
    print(f"  - PT Multiplier: {config.profit_take_multiplier}")
    print(f"  - SL Multiplier: {config.stop_loss_multiplier}")
    print(f"  - Resume: {config.resume}")
    print("-" * 50)
    if not get_bool_input("Is this configuration correct?", True):
        print("Aborted.")
        sys.exit(0)
    return config


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Create Context-Adaptive Proxy Labels."
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
        "--duration",
        type=str,
        default="300m",
        help="Target lookahead duration (e.g., '300m', '90s', or '60' [assumed minutes]).",
    )
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="Disable resume and start from scratch.",
    )
    parser.add_argument(
        "--pt-mult", type=float, default=2.0, help="Profit-take multiplier for ATR."
    )
    parser.add_argument(
        "--sl-mult", type=float, default=1.0, help="Stop-loss multiplier for ATR."
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
        duration_arg = args.duration
        if duration_arg.isdigit():
            duration_arg += "m"

        # --- ★★★ ここが修正箇所 (F821) ★★★ ---
        # (変更前) resume=not args.no-resume,
        # (変更後) resume=not args.no_resume,
        config = ProxyLabelConfig(
            target_duration=duration_arg,
            filter_mode=filter_mode_arg,
            filter_year=filter_year_arg,
            filter_month=filter_month_arg,
            resume=not args.no_resume,  # ★★★ 修正 ★★★
            profit_take_multiplier=args.pt_mult,
            stop_loss_multiplier=args.sl_mult,
        )
        # --- ★★★ 修正ここまで ★★★ ---
        try:
            temp_engine_for_validation = ProxyLabelingEngine(config)
        except ValueError as e:
            print(f"Configuration Error: {e}")
            parser.print_help()
            sys.exit(1)
    print("\nStarting engine with the specified configuration...")
    engine = ProxyLabelingEngine(config)
    engine.run()
