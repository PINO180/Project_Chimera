# /workspace/models/proxy_optimization/create_proxy_labels.py

import sys
from pathlib import Path
import warnings
import argparse
import shutil
from dataclasses import dataclass
import logging
from typing import List, Dict
import polars as pl
from tqdm import tqdm

# --- Project Path Setup ---
project_root = Path(__file__).resolve().parents[2]
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

# --- Blueprint Imports ---
from blueprint import S5_NEUTRALIZED_ALPHA_SET, S2_FEATURES_AFTER_AV

# --- Logging Setup ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(processName)s - %(message)s",
)
# Ignore common deprecation warnings from Polars
warnings.filterwarnings("ignore", category=DeprecationWarning)


# --- Configuration ---
@dataclass
class ProxyLabelConfig:
    """Config for creating a lightweight, labeled subset for proxy model training."""

    input_dir: Path = S5_NEUTRALIZED_ALPHA_SET
    price_data_source: Path = (
        S2_FEATURES_AFTER_AV
        / "feature_value_a_vast_universeC"
        / "features_e1c_M5.parquet"
    )
    output_dir: Path = (
        project_root
        / "data/XAUUSD/stratum_7_models/1A_2B/optimal_parameters/temp_labeled_subset_partitioned"
    )
    lookahead_periods: int = 60
    profit_take_multiplier: float = 2.0
    stop_loss_multiplier: float = 1.0
    atr_col_name: str = "e1c_atr_21"
    subset_year: int = 2023
    resume: bool = True


class ProxyLabelingEngine:
    """
    Engine to create a labeled subset of data for proxy model training.
    """

    def __init__(self, config: ProxyLabelConfig):
        self.config = config
        self.label_counts: Dict[int, int] = {1: 0, -1: 0, 0: 0}
        self._validate_paths()

    def _validate_paths(self):
        if not self.config.input_dir.exists():
            raise FileNotFoundError(
                f"Input directory not found: {self.config.input_dir}"
            )
        if not self.config.price_data_source.exists():
            raise FileNotFoundError(
                f"Price data source not found: {self.config.price_data_source}"
            )

    def run(self):
        logging.info(
            "### Phase 1, Script 1: Create Proxy Labels (Lightweight Problem Set) ###"
        )
        cfg = self.config

        if not cfg.resume and cfg.output_dir.exists():
            logging.warning(
                f"Output directory {cfg.output_dir} exists and not resuming. Removing it."
            )
            shutil.rmtree(cfg.output_dir)
        cfg.output_dir.mkdir(parents=True, exist_ok=True)

        try:
            logging.info(
                f"Step 1: Discovering feature files for year {cfg.subset_year}..."
            )
            all_feature_files = self._discover_feature_files()

            logging.info(
                "Step 2: Building a unified lazy frame with timeframe suffixes..."
            )
            unified_lf = self._build_unified_lazyframe(all_feature_files)

            logging.info(f"Step 3: Loading price data from {cfg.price_data_source}...")
            price_df = self._load_price_data()

            logging.info(
                "Step 4: Discovering daily partitions via lightweight reconnaissance..."
            )
            partitions_df = self._discover_partitions(all_feature_files)
            logging.info(
                f"   -> Found {len(partitions_df)} daily partitions to process."
            )

            logging.info(
                "Step 5: Starting daily processing and direct-to-disk writing loop..."
            )
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

                daily_bets_lf = unified_lf.filter(
                    pl.col("timestamp").dt.date() == current_date
                )
                daily_labeled_df = self._calculate_labels_for_batch(
                    daily_bets_lf, price_df
                )

                if daily_labeled_df is not None and not daily_labeled_df.is_empty():
                    self._update_label_counts(daily_labeled_df)
                    output_partition_dir.mkdir(parents=True, exist_ok=True)
                    daily_labeled_df.write_parquet(
                        output_partition_dir / "data.parquet", compression="zstd"
                    )

            self._log_final_summary()

        except Exception as e:
            logging.error(
                f"An error occurred during the labeling process: {e}", exc_info=True
            )
            raise

    def _discover_feature_files(self) -> List[str]:
        glob_pattern = (
            f"**/year={self.config.subset_year}/**/*.parquet"
            if self.config.subset_year > 0
            else "**/*.parquet"
        )
        logging.info(f"   -> Using glob pattern: {glob_pattern}")
        all_feature_files = [str(p) for p in self.config.input_dir.rglob(glob_pattern)]
        if not all_feature_files:
            raise ValueError(
                f"No feature files found for year {self.config.subset_year}."
            )
        return all_feature_files

    def _build_unified_lazyframe(self, file_paths: List[str]) -> pl.LazyFrame:
        modified_lazy_frames = []
        for f_path in file_paths:
            path_obj = Path(f_path)
            timeframe_suffix = ""

            # --- ★★★ CORRECTED: Robust timeframe extraction logic ★★★ ---
            for part in path_obj.parts:
                if "features_e" in part and "_neutralized" in part:
                    clean_part = part.replace(".parquet", "")
                    split_parts = clean_part.split("_")
                    if len(split_parts) >= 3:
                        timeframe_suffix = f"_{split_parts[2]}"
                        break

            if not timeframe_suffix:
                logging.warning(
                    f"Could not extract timeframe from path {f_path}. Suffix will be empty."
                )

            lf = pl.scan_parquet(f_path)
            try:
                feature_cols = [
                    col for col in lf.collect_schema().names() if col != "timestamp"
                ]
            except Exception:
                feature_cols = [
                    col
                    for col in pl.read_parquet(f_path, n_rows=1).columns
                    if col != "timestamp"
                ]

            renamed_lf = lf.select(
                "timestamp",
                *[pl.col(c).alias(f"{c}{timeframe_suffix}") for c in feature_cols],
            )
            modified_lazy_frames.append(renamed_lf)

        return pl.concat(modified_lazy_frames, how="diagonal").sort("timestamp")

    def _load_price_data(self) -> pl.DataFrame:
        return (
            pl.read_parquet(self.config.price_data_source)
            .select(["timestamp", "high", "low", "close", self.config.atr_col_name])
            .sort("timestamp")
        )

    def _discover_partitions(self, file_paths: List[str]) -> pl.DataFrame:
        scan_plans = (pl.scan_parquet(f).select("timestamp") for f in file_paths)
        recon_plan_lf = pl.concat(scan_plans, how="diagonal")
        return (
            recon_plan_lf.select(pl.col("timestamp").dt.date().alias("date"))
            .unique()
            .collect()
            .sort("date")
        )

    def _calculate_labels_for_batch(
        self, bets_lf: pl.LazyFrame, price_df: pl.DataFrame
    ) -> pl.DataFrame | None:
        cfg = self.config
        try:
            bets_with_price_lf = bets_lf.join_asof(
                price_df.lazy(), on="timestamp"
            ).filter(pl.col(cfg.atr_col_name).is_not_null())

            bets_df = bets_with_price_lf.select(
                pl.col("timestamp").alias("t0"),
                (
                    pl.col("close")
                    + pl.col(cfg.atr_col_name) * cfg.profit_take_multiplier
                ).alias("pt_barrier"),
                (
                    pl.col("close")
                    - pl.col(cfg.atr_col_name) * cfg.stop_loss_multiplier
                ).alias("sl_barrier"),
                (
                    pl.col("timestamp") + pl.duration(minutes=cfg.lookahead_periods * 5)
                ).alias("t1_max"),
                pl.all().exclude(
                    ["timestamp", "close", cfg.atr_col_name, "high", "low"]
                ),
            ).collect()

            if bets_df.is_empty():
                return None

            min_ts, max_ts = bets_df["t0"].min(), bets_df["t1_max"].max()
            if min_ts is None or max_ts is None:
                return None

            price_window_df = price_df.filter(
                pl.col("timestamp").is_between(min_ts, max_ts)
            )

            hits_df = (
                price_window_df.lazy()
                .join_asof(
                    bets_df.lazy().select(["t0", "pt_barrier", "sl_barrier", "t1_max"]),
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
                .collect()
            )

            final_df = (
                bets_df.join(hits_df, on="t0", how="left")
                .with_columns(
                    t1=pl.when(
                        (pl.col("first_pt_time").is_not_null())
                        & (
                            pl.col("first_sl_time").is_null()
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
                            pl.col("first_sl_time").is_null()
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
                )
                .rename({"t0": "timestamp"})
            )
            return final_df.drop(
                ["pt_barrier", "sl_barrier", "t1_max", "first_pt_time", "first_sl_time"]
            )
        except Exception as e:
            logging.error(f"Error in _calculate_labels_for_batch: {e}", exc_info=False)
            return None

    def _update_label_counts(self, df: pl.DataFrame):
        # --- ★★★ CORRECTED: No .collect() on a DataFrame ★★★ ---
        # --- ★★★ CORRECTED: Use .len() instead of .count() ★★★ ---
        counts = df.group_by("label").len()
        for row in counts.iter_rows(named=True):
            label = row["label"]
            if label in self.label_counts:
                self.label_counts[label] += row["len"]

    def _log_final_summary(self):
        total_samples = sum(self.label_counts.values())
        if total_samples == 0:
            logging.warning("No samples were processed. The output is empty.")
            return

        pos_count = self.label_counts.get(1, 0)
        neg_count = self.label_counts.get(-1, 0)
        scale_pos_weight = neg_count / pos_count if pos_count > 0 else 1.0

        logging.info("\n" + "=" * 60)
        logging.info("### Proxy Labeling COMPLETED! ###")
        logging.info(
            f"The lightweight 'Problem Set' for our Proxy AI is ready at: {self.config.output_dir}"
        )
        logging.info(f"  - Total Labeled Samples: {total_samples}")
        logging.info(f"  - Profit-Take (1): {self.label_counts.get(1, 0)}")
        logging.info(f"  - Stop-Loss (-1): {self.label_counts.get(-1, 0)}")
        logging.info(f"  - Timed-Out (0): {self.label_counts.get(0, 0)}")
        logging.info(
            f"  - Calculated `scale_pos_weight` for next step: {scale_pos_weight:.4f}"
        )
        logging.info("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Create a labeled subset for proxy model training."
    )
    parser.add_argument(
        "--year",
        type=int,
        default=2023,
        help="The year of data to process. Set to 0 to process all available data. Default is 2023.",
    )
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="Disable resume feature and start from scratch.",
    )
    args = parser.parse_args()

    config = ProxyLabelConfig(subset_year=args.year, resume=not args.no_resume)
    engine = ProxyLabelingEngine(config)
    engine.run()
