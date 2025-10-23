# /workspace/models/optimal_parameters/create_proxy_labels.py
# FINAL ARCHITECTURE V7.1 - V6の効率的なパスフィルターロジック + V7の全期間/月間フィルターとレポート機能
# [Gemini 修正版: target_duration で秒単位 (e.g., "90s") をサポート]

import sys
from pathlib import Path
import warnings
import argparse
import shutil
from dataclasses import dataclass, field
import logging
from typing import List, Dict, Any, Optional
import polars as pl
from tqdm import tqdm
import re
import gc
import datetime as dt

# --- Project Path Setup ---
project_root = Path(__file__).resolve().parents[2]
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

# --- Blueprint Imports ---
from blueprint import S5_NEUTRALIZED_ALPHA_SET, S2_FEATURES_FIXED

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
    output_dir: Path = (
        project_root
        / "data/XAUUSD/stratum_7_models/1A_2B/optimal_parameters/temp_labeled_subset_partitioned"
    )
    target_duration: str = "300m"
    profit_take_multiplier: float = 2.0
    stop_loss_multiplier: float = 1.0
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
            # Format month number with leading zero if needed
            month_str = f"{self.filter_month:02d}"
            return f"Year/Month = {self.filter_year}/{month_str}"
        elif self.filter_mode == "all":
            return "All Time"
        else:
            # Handle cases where year/month might be missing for the selected mode
            return f"Invalid or Incomplete Filter ({self.filter_mode}, Year: {self.filter_year}, Month: {self.filter_month})"


class ProxyLabelingEngine:
    """Engine to create a labeled subset of data using 'Context-Adaptive Labeling'."""

    def __init__(self, config: ProxyLabelConfig):
        self.config = config

        # --- [修正箇所 START] (V7.1の動的パス) ---
        # target_duration と PT/SL比率に基づいて、一意の出力ディレクトリを動的に設定する

        # 1. dataclassからベースパスを取得 (e.g., ".../temp_labeled_subset_partitioned")
        base_output_dir = config.output_dir
        base_name = base_output_dir.name  # "temp_labeled_subset_partitioned"
        base_parent = base_output_dir.parent

        # 2. duration (e.g., "15m" -> "15", "90s" -> "90s")
        # [Gemini 修正]: .replace("m", "") は "90s" を "90s" のままにするため、このロジックは変更不要
        duration_str = config.target_duration.replace("m", "")

        # 3. TP (e.g., 2.0 -> "20", 0.3 -> "03")
        tp_str = f"{int(config.profit_take_multiplier * 10):02d}"

        # 4. SL (e.g., 1.0 -> "10", 0.1 -> "01")
        sl_str = f"{int(config.stop_loss_multiplier * 10):02d}"

        # 5. 結合 (e.g., "temp_labeled_subset_partitioned_15_10_05" or "..._90s_10_05")
        new_dir_name = f"{base_name}_{duration_str}_{tp_str}_{sl_str}"
        self.config.output_dir = (
            base_parent / new_dir_name
        )  # config.output_dir を上書き

        logging.info(f"Dynamic output directory set to: {self.config.output_dir}")
        # --- [修正箇所 END] ---

        self.label_counts: Dict[int, int] = {1: 0, -1: 0, 0: 0}
        self.report_data: List[Dict[str, Any]] = []
        self._validate_paths()
        self._validate_config()  # V7のバリデーションロジック

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
        """Validates the filter configuration (V7のロジック)."""
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

        # --- [Gemini 修正箇所 START] ---
        # Validate duration format
        # 正規表現を 'm' または 's' を受け入れるように変更
        duration_match = re.match(r"^(\d+)([ms])?$", cfg.target_duration)
        if not duration_match:
            raise ValueError(
                f"Invalid target_duration format: {cfg.target_duration}. Expected format like '300m', '90s', or '60' (assumed as minutes)."
            )

        value_str, unit = duration_match.groups()

        # Ensure 'm' is present for consistency internally if no unit provided
        if unit is None:  # e.g., "60"
            cfg.target_duration += "m"  # "60m"
        # --- [Gemini 修正箇所 END] ---

    # --- [Gemini 修正箇所 START] ---
    # 新規ヘルパー関数: duration文字列を浮動小数点数の「分」に変換
    def _get_duration_in_minutes(self, duration_str: str) -> float:
        """Converts a duration string (e.g., '300m', '90s') to a float value in minutes."""
        # _validate_config が実行済みであることを前提とする (e.g., "300m", "90s")
        match = re.match(r"^(\d+)([ms])$", duration_str)
        if not match:
            # _validate_config が通っていれば、ここには到達しないはず
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

        return 0.0  # Should not happen

    # --- [Gemini 修正箇所 END] ---

    def run(self):
        # V7のrunロジック（ログや空データハンドリングが改善されている）
        logging.info(
            f"### Phase 1, Script 1: Create Proxy Labels (Context-Adaptive Engine) for {self.config.target_duration} ###"
        )
        logging.info(
            f"Applying filter: {self.config.get_filter_description()}"
        )  # Log filter
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
                "Step 2: Building unified lazy frame with 'timeframe' column..."
            )
            # ★★★ ここで修正版の関数を呼ぶ ★★★
            unified_lf = self._build_unified_lazyframe(all_feature_paths)

            # Check if filtering resulted in an empty LazyFrame
            if unified_lf.collect_schema().names() == []:  # Check if schema is empty
                logging.warning(
                    f"No data found for the specified filter '{cfg.get_filter_description()}'. Exiting."
                )
                self._log_final_summary()  # Still log summary (will show 0 samples)
                self._generate_report()  # Attempt to generate empty report
                return

            logging.info("Step 3: Preparing price data blueprints (LazyFrames)...")
            price_components = self._load_all_price_data()
            base_price_lf = price_components["base_lf"]
            atr_lfs = price_components["atr_lfs"]

            logging.info("Step 4: Discovering daily partitions for processing...")
            # Discover partitions *after* filtering
            partitions_df = self._discover_partitions(unified_lf)
            logging.info(
                f"   -> Found {len(partitions_df)} daily partitions to process for the filter '{cfg.get_filter_description()}'."
            )

            if partitions_df.is_empty():
                logging.warning(
                    f"No partitions found after applying filter '{cfg.get_filter_description()}'."
                )
                self._log_final_summary()
                self._generate_report()
                return

            logging.info(
                "Step 5: Starting daily processing loop with on-demand data assembly..."
            )

            # --- [Gemini 修正箇所 START] ---
            # duration_match = re.match(r"(\d+)", cfg.target_duration) # OLD
            # max_lookahead_minutes = float(duration_match.group(1)) # OLD

            # Get duration in minutes as a float (e.g., "90s" -> 1.5)
            max_lookahead_minutes = self._get_duration_in_minutes(cfg.target_duration)

            # Convert to a Python timedelta object for datetime arithmetic
            # Pythonのdatetime + pl.duration(minutes=1.5) は TypeError になるため
            max_lookahead_delta = dt.timedelta(minutes=max_lookahead_minutes)
            # --- [Gemini 修正箇所 END] ---

            for row in tqdm(
                partitions_df.iter_rows(named=True),
                total=len(partitions_df),
                desc="Processing Partitions",
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
                    continue

                # Collect only the data for the current day *from the already filtered LazyFrame*
                daily_bets_df = unified_lf.filter(
                    pl.col("timestamp").dt.date() == current_date
                ).collect()
                if daily_bets_df.is_empty():
                    continue  # Should not happen if partition was discovered, but safe check

                min_ts_req = daily_bets_df["timestamp"].min()
                if min_ts_req is None:
                    continue

                # --- [Gemini 修正箇所 START] ---
                # max_ts_req = min_ts_req + pl.duration( # OLD
                #     minutes=max_lookahead_minutes, days=2
                # )

                # Python の datetime オブジェクトには、Python の timedelta を使用する
                max_ts_req = min_ts_req + max_lookahead_delta + dt.timedelta(days=2)
                # --- [Gemini 修正箇所 END] ---

                price_window_df = base_price_lf.filter(
                    pl.col("timestamp").is_between(min_ts_req, max_ts_req)
                ).collect()
                if price_window_df.is_empty():
                    logging.warning(
                        f"No base price data found for {current_date}. Skipping partition."
                    )
                    continue

                for atr_lf in atr_lfs:
                    atr_df_small = atr_lf.filter(
                        pl.col("timestamp").is_between(min_ts_req, max_ts_req)
                    ).collect()
                    if not atr_df_small.is_empty():
                        price_window_df = price_window_df.join_asof(
                            atr_df_small.sort("timestamp"), on="timestamp"
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
                gc.collect()

            self._log_final_summary()
            self._generate_report()

        except Exception as e:
            logging.error(f"A critical error occurred: {e}", exc_info=True)
            raise

    # =========================================================================
    # 修正されたコアロジック
    # =========================================================================

    def _discover_feature_paths(self) -> List[Path]:
        # V6/V7で同一
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

    def _build_unified_lazyframe(self, feature_paths: List[Path]) -> pl.LazyFrame:
        """
        [★★★ 修正版 ★★★]
        V7のフィルター(year/month/all)をサポートしつつ、
        V6(動作する方)の効率的なファイルパスベースのスキャンロジックを融合する。
        ★★★ さらに、各特徴量カラム名にタイムフレーム接尾辞を付与するロジックを追加 ★★★
        """
        all_lazy_frames = []
        timeframe_pattern = re.compile(
            # [修正] .parquet 拡張子を考慮しないようにパターンを調整
            r"features_e\d+[a-z]?_([a-zA-Z0-9\.]+)(?:_neutralized)?"
        )
        cfg = self.config

        for path in feature_paths:
            # --- [修正] ファイル名またはディレクトリ名からタイムフレームを抽出 ---
            name_to_match = path.stem if path.is_file() else path.name
            match = timeframe_pattern.search(name_to_match)
            if not match:
                logging.warning(f"Could not extract timeframe from: {name_to_match}")
                continue
            timeframe = match.group(1)
            timeframe_suffix = f"_{timeframe}"  # 接尾辞を作成 (例: _M1, _D1)
            # --- [修正ここまで] ---

            lf: Optional[pl.LazyFrame] = None

            # --- V6/V7 融合ロジック (ディレクトリ用) ---
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
            # --- ディレクトリ用ロジックここまで ---

            # --- V6/V7 共通ロジック (単一ファイル用) ---
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
                else:  # 'all' モード
                    lf = lf_full
            # --- 単一ファイルロジックここまで ---

            # --- ▼▼▼ ここから接尾辞付与ロジックを追加 ▼▼▼ ---
            if lf is not None:
                # スキャン結果が空でないか確認
                current_schema = lf.collect_schema()
                if not current_schema.names():
                    continue

                # 'timestamp' 以外の特徴量カラムを取得
                feature_cols = [
                    col for col in current_schema.names() if col != "timestamp"
                ]

                # 各特徴量カラムにタイムフレーム接尾辞を付与する式を作成
                rename_exprs = [
                    pl.col(col).alias(f"{col}{timeframe_suffix}")
                    for col in feature_cols
                ]

                # timestamp列と、名前変更された特徴量列を選択
                if rename_exprs:
                    # [修正] with_columns ではなく select を使用して列を再構築
                    lf_renamed = lf.select(
                        pl.col("timestamp").cast(
                            pl.Datetime("us", "UTC")
                        ),  # timestamp の型変換もここで行う
                        *rename_exprs,
                    )
                else:
                    # 特徴量カラムがない場合 (timestamp のみなど)
                    lf_renamed = lf.select(
                        pl.col("timestamp").cast(pl.Datetime("us", "UTC"))
                    )

                # 元のタイムフレームを示す 'timeframe' 列を追加 (これは接尾辞とは別)
                # [修正] lf_renamed に対して with_columns を適用
                all_lazy_frames.append(
                    lf_renamed.with_columns(pl.lit(timeframe).alias("timeframe"))
                )
            # --- ▲▲▲ 接尾辞付与ロジックここまで ▲▲▲ ---

        if not all_lazy_frames:
            logging.warning(
                f"No feature data found for the filter: {cfg.get_filter_description()}."
            )
            return pl.LazyFrame()

        try:
            # how="diagonal" で結合。異なる接尾辞を持つカラムは別の列として保持される
            return pl.concat(all_lazy_frames, how="diagonal").sort("timestamp")
        except pl.exceptions.ComputeError as e:
            if "cannot concat empty list" in str(e).lower():
                logging.warning(
                    f"Concatenation resulted in empty data for filter: {cfg.get_filter_description()}."
                )
                return pl.LazyFrame()
            else:
                raise e

    # =========================================================================
    # V6/V7で同一、またはV7で改善済みのヘルパー関数
    # =========================================================================

    def _load_all_price_data(self) -> Dict[str, Any]:
        """
        Prepares blueprints for base price and ATRs, adding dedicated Tick ATR
        while preserving the original logic for standard ATR files and renaming 'datetime' column.
        """
        price_dir = self.config.price_data_source_dir / "feature_value_a_vast_universeC"
        tick_dir = price_dir / "features_e1c_tick"
        if not tick_dir.exists():
            raise FileNotFoundError(f"Master price directory not found: {tick_dir}")

        logging.info(f"   -> Lazily scanning '{tick_dir}' as the master price source.")
        base_lf = (
            pl.scan_parquet(str(tick_dir / "**/*.parquet"))
            .select("timestamp", "close", "high", "low")
            .with_columns(pl.col("timestamp").cast(pl.Datetime("us", "UTC")))
            .unique("timestamp", keep="first", maintain_order=True)
        )

        # Find standard ATR files first
        atr_files = list(price_dir.glob("features_e1c_*.parquet"))

        # # --- [ 追加 ] ---
        # # Explicitly add the path to the newly created and fixed tick ATR file
        # tick_atr_file_path = Path(
        #     "/workspace/data/XAUUSD/stratum_2_features_fixed/features_e1c_tick_atr_only_tick_fixed.parquet"
        # )
        # tick_atr_file_found = False
        # if tick_atr_file_path.exists():
        #     atr_files.append(
        #         tick_atr_file_path
        #     )  # Add the dedicated file path to the list
        #     tick_atr_file_found = True
        #     logging.info(
        #         f"   -> Found dedicated tick ATR file: {tick_atr_file_path.name}"
        #     )
        # else:
        #     logging.warning(
        #         f"   -> Dedicated tick ATR file not found at: {tick_atr_file_path}."
        #     )
        # # --- [ 追加ここまで ] ---

        if not atr_files:
            raise FileNotFoundError(
                f"No ATR parquet files (standard or dedicated tick) found in {price_dir} or specified path."
            )

        # Adjusted regex slightly
        timeframe_pattern = re.compile(
            r"features_e1c_([a-zA-Z0-9\.]+)(?:_atr_only_tick_fixed)?\.parquet"
        )
        all_atr_lfs = []
        logging.info(f"   -> Processing {len(atr_files)} potential ATR source files.")

        for f_path in atr_files:
            is_dedicated_tick_file = "_atr_only_tick_fixed" in f_path.name
            match = timeframe_pattern.search(f_path.name)

            if not match:
                logging.warning(
                    f"   -> Could not parse timeframe from {f_path.name}. Skipping."
                )
                continue

            # Correctly identify the timeframe
            timeframe = match.group(1)
            if is_dedicated_tick_file:
                timeframe = "tick"

            target_atr_name = f"e1c_atr_21_{timeframe}"

            # Scan the parquet file (Do not cast timestamp here yet)
            lf_original = pl.scan_parquet(str(f_path))
            schema_names = lf_original.collect_schema().names()

            # --- Logic branching: Dedicated Tick File vs Standard Files ---
            if is_dedicated_tick_file:
                # --- Handle Dedicated Tick ATR File ---
                expected_tick_atr_col = (
                    "e1c_tick_atr_only_atr_21"  # <- ★★★ Correct this if needed ★★★
                )
                time_col_name = None
                if "datetime" in schema_names:
                    time_col_name = "datetime"
                elif "timestamp" in schema_names:
                    time_col_name = "timestamp"

                if time_col_name and expected_tick_atr_col in schema_names:
                    # Select, rename time column to 'timestamp', cast, select ATR and rename
                    processed_lf = lf_original.select(
                        [
                            pl.col(time_col_name)
                            .cast(pl.Datetime("us", "UTC"))
                            .alias("timestamp"),  # Cast and Rename time column
                            pl.col(expected_tick_atr_col).alias(
                                target_atr_name
                            ),  # Select and Rename ATR column
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
                # --- End Tick File Handling ---

            else:
                # --- Handle Standard ATR Files (Using ORIGINAL User Logic, adding UTC cast) ---
                # Cast timestamp for standard files here
                lf = lf_original.with_columns(
                    pl.col("timestamp").cast(pl.Datetime("us", "UTC"))
                )
                # Re-fetch schema names after potential column operations if needed, but not strictly necessary here
                # schema_names = lf.collect_schema().names() # Re-fetch if needed

                atr_col_to_rename = next(
                    (
                        col
                        for col in schema_names
                        if "e1c_atr_21" in col and col != target_atr_name
                    ),
                    None,
                )

                if target_atr_name in schema_names:
                    all_atr_lfs.append(lf.select(["timestamp", target_atr_name]))
                    logging.info(
                        f"   -> Prepared ATR blueprint for timeframe '{timeframe}' from standard file {f_path.name} (Exact match)"
                    )
                elif atr_col_to_rename:
                    all_atr_lfs.append(
                        lf.select(["timestamp", atr_col_to_rename]).rename(
                            {atr_col_to_rename: target_atr_name}
                        )
                    )
                    logging.info(
                        f"   -> Prepared ATR blueprint for timeframe '{timeframe}' from standard file {f_path.name} (Renamed '{atr_col_to_rename}')"
                    )
                else:
                    logging.warning(
                        f"   -> No ATR column found in {f_path.name}. Skipping."
                    )
                # --- End Standard File Handling ---

        if not all_atr_lfs:
            raise ValueError(
                "FATAL: No valid ATR columns were extracted from any price files."
            )

        logging.info(f"   -> Prepared {len(all_atr_lfs)} ATR blueprints in total.")
        return {"base_lf": base_lf, "atr_lfs": all_atr_lfs}

    def _discover_partitions(self, unified_lf: pl.LazyFrame) -> pl.DataFrame:
        # V7の改善版（空スキーマチェック）
        if unified_lf.collect_schema().names() == []:
            return pl.DataFrame(
                {"date": []}
            )  # Return empty DataFrame with correct column
        return (
            unified_lf.select(pl.col("timestamp").dt.date().alias("date"))
            .unique()
            .collect()
            .sort("date")
        )

    def _get_bar_duration_minutes(self, timeframe: str) -> float:
        # V6/V7で同一
        if timeframe == "tick":
            return 0.0
        value_match = re.search(r"(\d*\.?\d+)", timeframe)
        unit_match = re.search(r"([A-Z])", timeframe)
        if not value_match or not unit_match:
            return 0.0
        value = float(value_match.group(1))
        unit = unit_match.group(1)
        if unit == "M":
            return value
        if unit == "H":
            return value * 60
        if unit == "D":
            return value * 1440
        if unit == "W":
            return value * 10080
        if unit == "N":
            return value * 43200
        return 0.0

    def _calculate_labels_for_batch(
        self, daily_bets_df: pl.DataFrame, price_window_df: pl.DataFrame
    ) -> pl.DataFrame | None:
        # V6/V7で同一
        cfg = self.config
        if daily_bets_df.is_empty():
            return None

        labeled_chunks = []
        for timeframe_tuple, group_df in daily_bets_df.group_by("timeframe"):
            timeframe = timeframe_tuple[0]
            if timeframe is None or group_df.is_empty():
                continue

            # --- [Gemini 修正箇所 START] ---
            # duration_match = re.match(r"(\d+)", cfg.target_duration) # OLD
            # if not duration_match: # OLD
            #     continue  # Should be validated earlier # OLD
            # target_duration_minutes = float(duration_match.group(1)) # OLD

            # Get duration in minutes as a float (e.g., "90s" -> 1.5)
            # この変数は 1) Polars expression 内での pl.duration (float OK)
            #            2) lookahead_bars の計算 (float OK)
            # の2箇所で使われるため、float のままで問題ない
            target_duration_minutes = self._get_duration_in_minutes(cfg.target_duration)
            # --- [Gemini 修正箇所 END] ---

            bar_duration_minutes = self._get_bar_duration_minutes(timeframe)

            t1_max_expr: pl.Expr
            if timeframe == "tick":
                # Polars expression 内では float minutes が使える
                t1_max_expr = pl.col("timestamp") + pl.duration(
                    minutes=target_duration_minutes
                )
            else:
                if bar_duration_minutes == 0:
                    continue
                lookahead_bars = max(
                    1, int(round(target_duration_minutes / bar_duration_minutes))
                )
                # Polars expression 内では float minutes が使える
                t1_max_expr = pl.col("timestamp") + pl.duration(
                    minutes=lookahead_bars * bar_duration_minutes
                )

            atr_col_name = f"e1c_atr_21_{timeframe}"
            if atr_col_name not in price_window_df.columns:
                continue

            original_cols = group_df.columns
            bets_with_price_df = group_df.join_asof(
                price_window_df, on="timestamp"
            ).filter(pl.col(atr_col_name).is_not_null())
            if bets_with_price_df.is_empty():
                continue

            bets_df = bets_with_price_df.select(
                pl.col("timestamp").alias("t0"),
                (
                    pl.col("close") + pl.col(atr_col_name) * cfg.profit_take_multiplier
                ).alias("pt_barrier"),
                (
                    pl.col("close") - pl.col(atr_col_name) * cfg.stop_loss_multiplier
                ).alias("sl_barrier"),
                t1_max_expr.alias("t1_max"),
                pl.col(original_cols).exclude("timestamp"),
            )

            hits_df = (
                price_window_df.join_asof(
                    bets_df.select(["t0", "pt_barrier", "sl_barrier", "t1_max"]),
                    left_on="timestamp",
                    right_on="t0",
                )
                .filter(pl.col("timestamp") <= pl.col("t1_max"))
                .with_columns(
                    pt_hit_time=pl.when(pl.col("high") >= pl.col("pt_barrier")).then(
                        pl.col("timestamp")
                    ),
                    sl_hit_time=pl.when(pl.col("low") <= pl.col("sl_barrier")).then(
                        pl.col("timestamp")
                    ),
                )
                .group_by("t0")
                .agg(
                    pl.col("pt_hit_time").min().alias("first_pt_time"),
                    pl.col("sl_hit_time").min().alias("first_sl_time"),
                )
            )

            final_group_df = bets_df.join(hits_df, on="t0", how="left")
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
                # --- ▼▼▼ デバッグ用に一時的に追加 ▼▼▼ ---
                debug_atr_col_used=pl.lit(atr_col_name),
                # --- ▲▲▲ デバッグ用に追加ここまで ▲▲▲ ---
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
        # V6/V7で同一
        counts = df.group_by("label").len()
        for row in counts.iter_rows(named=True):
            if row["label"] in self.label_counts:
                self.label_counts[row["label"]] += row["len"]

    def _log_final_summary(self):
        # V7の改善版（フィルター情報をログに出力）
        total_samples = sum(self.label_counts.values())
        if total_samples == 0:
            logging.warning("No samples were processed for the selected filter.")
            # Added more context to the message
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
            f"### Context-Adaptive Labeling COMPLETED (Filter: {self.config.get_filter_description()}) ###\n"  # Added filter desc
            f"The '{self.config.target_duration}' version of the 'Problem Set' is ready at: {self.config.output_dir}\n"
            f"  - Total Labeled Samples: {total_samples}\n"
            f"  - (+) Profit-Take: {pos}, (-) Stop-Loss: {neg}, (0) Timed-Out: {self.label_counts.get(0, 0)}\n"
            f"  - Calculated `scale_pos_weight` for next step: {scale_pos_weight:.4f}\n"
            + "="
            * 60
        )
        logging.info(summary)

    # --- V7のレポート機能（V6のバグ修正済み） ---
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

        # --- [修正箇所 START] (V7.1の動的パス) ---
        # 1. duration (e.g., "15m" -> "15", "90s" -> "90s")
        # [Gemini 修正]: .replace("m", "") は "90s" を "90s" のままにするため、このロジックは変更不要
        duration_str = cfg.target_duration.replace("m", "")

        # 2. TP (e.g., 2.0 -> "20", 0.3 -> "03")
        tp_str = f"{int(cfg.profit_take_multiplier * 10):02d}"

        # 3. SL (e.g., 1.0 -> "10", 0.1 -> "01")
        sl_str = f"{int(cfg.stop_loss_multiplier * 10):02d}"

        # 4. 結合 (e.g., "execution_report_15_10_05.md" or "..._90s_10_05.md")
        # フィルター情報 (Year-Monthなど) はファイル名から除外
        report_filename = f"execution_report_{duration_str}_{tp_str}_{sl_str}.md"
        report_path = cfg.output_dir / report_filename
        # --- [修正箇所 END] ---

        if not self.report_data:
            logging.warning("No data available for report generation.")
            # レポート内容にもフィルター情報を記載 (ファイル名からは除外したが、中身には残す)
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

            # --- [V7のバグ修正済みロジック] ---
            duration_stats_seconds = df["event_duration_seconds"]
            valid_durations = duration_stats_seconds.filter(
                duration_stats_seconds.is_finite()
                & duration_stats_seconds.is_not_null()
            )

            # --- [ ★★★ ここを修正 ★★★ ] ---
            # .height ではなく .len() を使う
            if valid_durations.len() > 0:
                # --- [ ★★★ 修正完了 ★★★ ] ---
                duration_stats = (valid_durations / 60).describe()  # Convert to minutes

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
            # --- バグ修正ロジックここまで ---

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

    # --- Helper Functions (Identical to V7) ---
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

    # --- End Helper Functions ---

    # --- New Filter Mode Selection ---
    print("1. Select data filter mode:")
    print("   [1] Year  - Process data for a specific year.")
    print("   [2] Month - Process data for a specific month within a specific year.")
    print("   [3] All   - Process all available data.")
    mode_choice = get_input(
        "   Enter choice (1, 2, or 3)", "1", str, lambda m: m in ["1", "2", "3"]
    )

    filter_mode: str = "all"  # Default to all
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
    # --- End Filter Mode Selection ---

    # --- [Gemini 修正箇所 START] ---
    # duration_valid = ( # OLD
    #     lambda d: (isinstance(d, str) and (d.endswith("m") and d[:-1].isdigit()))
    #     or d.isdigit()
    # )
    duration_valid = lambda d: bool(re.match(r"^(\d+)([ms])?$", d))

    duration = get_input(
        # "\n2. Target lookahead duration (e.g., '300m', '60')", # OLD
        "\n2. Target lookahead duration (e.g., '300m', '90s', or '60' [assumed minutes])",  # NEW
        "300m",
        str,
        duration_valid,
    )
    if duration.isdigit():
        duration += "m"  # Ensure 'm' suffix
    # --- [Gemini 修正箇所 END] ---

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
    print(f"  - Data Filter: {config.get_filter_description()}")  # Uses updated method
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
    # V7のコマンドライン引数パーサー（年/月/全期間対応）
    parser = argparse.ArgumentParser(
        description="Create Context-Adaptive Proxy Labels."
    )
    # --- Command Line Arguments for Filtering ---
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
    # --- End Filter Arguments ---
    parser.add_argument(
        "--duration",
        type=str,
        default="300m",
        # --- [Gemini 修正箇所 START] ---
        # help="Target lookahead duration (e.g., '300m' or '60').", # OLD
        help="Target lookahead duration (e.g., '300m', '90s', or '60' [assumed minutes]).",  # NEW
        # --- [Gemini 修正箇所 END] ---
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

    # Determine if running interactively based on --filter-mode presence
    run_interactive = args.filter_mode is None

    if run_interactive:
        # No filter mode provided, run interactive setup
        config = _get_interactive_config()
    else:
        # Filter mode provided via command line
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
        # 'all' mode requires no further args

        duration_arg = args.duration
        # [Gemini 修正]: duration_arg が "90s" の場合は isdigit() が False になる
        #               duration_arg が "60" の場合は isdigit() が True になり "60m" になる
        # このロジックは _validate_config で処理されるため、ここで `isdigit()` チェックを
        # 削除しても良いが、互換性のために残しておく
        if duration_arg.isdigit():
            duration_arg += "m"

        config = ProxyLabelConfig(
            target_duration=duration_arg,
            filter_mode=filter_mode_arg,
            filter_year=filter_year_arg,
            filter_month=filter_month_arg,
            resume=not args.no_resume,
            profit_take_multiplier=args.pt_mult,
            stop_loss_multiplier=args.sl_mult,
        )
        # Validate config immediately after parsing args
        try:
            # Use the validation logic within the Engine's init
            # [Gemini 修正]: _validate_config がここで呼び出されるため、
            # duration_arg が "90s" や "60" でも正しく処理される
            temp_engine_for_validation = ProxyLabelingEngine(config)
        except ValueError as e:
            print(f"Configuration Error: {e}")
            parser.print_help()  # Show help on config error
            sys.exit(1)

    print("\nStarting engine with the specified configuration...")
    # [Gemini 修正]: config オブジェクトは _validate_config によって
    # (例: "60" -> "60m") 正規化されている
    engine = ProxyLabelingEngine(config)
    engine.run()
