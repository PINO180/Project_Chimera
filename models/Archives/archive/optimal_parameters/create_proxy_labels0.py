# /workspace/models/optimal_parameters/create_proxy_labels.py
# FINAL ARCHITECTURE V6 - User's Original Core Logic + Interactive/Reporting Layers

import sys
from pathlib import Path
import warnings
import argparse
import shutil
from dataclasses import dataclass, field
import logging
from typing import List, Dict, Any
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


# --- Configuration ---
@dataclass
class ProxyLabelConfig:
    """Config for creating a context-adaptive, labeled subset for proxy model training."""

    input_dir: Path = S5_NEUTRALIZED_ALPHA_SET
    price_data_source_dir: Path = S2_FEATURES_FIXED
    # NOTE: This is the exact default output_dir from the user's original script.
    output_dir: Path = (
        project_root
        / "data/XAUUSD/stratum_7_models/1A_2B/optimal_parameters/temp_labeled_subset_partitioned"
    )
    target_duration: str = "300m"
    profit_take_multiplier: float = 2.0
    stop_loss_multiplier: float = 1.0
    subset_year: int = 2023
    resume: bool = True
    # [ADDED] Field to store the start time of the execution for reporting
    execution_start_time: str = field(
        default_factory=lambda: dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    )


class ProxyLabelingEngine:
    """Engine to create a labeled subset of data using 'Context-Adaptive Labeling'."""

    def __init__(self, config: ProxyLabelConfig):
        self.config = config
        self.label_counts: Dict[int, int] = {1: 0, -1: 0, 0: 0}
        # [ADDED] List to store detailed statistics for the final report
        self.report_data: List[Dict[str, Any]] = []
        self._validate_paths()

    def _validate_paths(self):
        if not self.config.input_dir.exists():
            raise FileNotFoundError(
                f"Input directory not found: {self.config.input_dir}"
            )
        if not self.config.price_data_source_dir.exists():
            raise FileNotFoundError(
                f"Price data source dir not found: {self.config.price_data_source_dir}"
            )

    def run(self):
        logging.info(
            f"### Phase 1, Script 1: Create Proxy Labels (Context-Adaptive Engine) for {self.config.target_duration} ###"
        )
        cfg = self.config

        # NOTE: Using the exact output logic from the user's original script.
        # No subdirectories for duration are created automatically.
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
            unified_lf = self._build_unified_lazyframe(all_feature_paths)

            logging.info("Step 3: Preparing price data blueprints (LazyFrames)...")
            price_components = self._load_all_price_data()
            base_price_lf = price_components["base_lf"]
            atr_lfs = price_components["atr_lfs"]

            logging.info("Step 4: Discovering daily partitions for processing...")
            partitions_df = self._discover_partitions(unified_lf)
            logging.info(
                f"   -> Found {len(partitions_df)} daily partitions to process."
            )

            if partitions_df.is_empty():
                self._log_final_summary()
                return

            logging.info(
                "Step 5: Starting daily processing loop with on-demand data assembly..."
            )
            max_lookahead_minutes = float(cfg.target_duration.replace("m", ""))

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

                daily_bets_df = unified_lf.filter(
                    pl.col("timestamp").dt.date() == current_date
                ).collect()
                if daily_bets_df.is_empty():
                    continue

                min_ts_req = daily_bets_df["timestamp"].min()
                if min_ts_req is None:
                    continue
                max_ts_req = min_ts_req + pl.duration(
                    minutes=max_lookahead_minutes, days=2
                )

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
                    # NOTE: The original label count logic is preserved.
                    self._update_label_counts(daily_labeled_df)
                    # [ADDED] Report data is collected in a separate, non-intrusive step.
                    self._collect_report_data(daily_labeled_df, current_date)

                    output_partition_dir.mkdir(parents=True, exist_ok=True)
                    daily_labeled_df.write_parquet(
                        output_partition_dir / "data.parquet", compression="zstd"
                    )

                del daily_bets_df, price_window_df, daily_labeled_df
                gc.collect()

            self._log_final_summary()
            # [ADDED] Generate the detailed report at the end of the process.
            self._generate_report()

        except Exception as e:
            logging.error(f"A critical error occurred: {e}", exc_info=True)
            raise

    # =========================================================================
    # ALL METHODS BELOW ARE IDENTICAL TO THE USER'S ORIGINAL SCRIPT
    # (except for the added reporting methods at the end)
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

    def _build_unified_lazyframe(self, feature_paths: List[Path]) -> pl.LazyFrame:
        all_lazy_frames = []
        timeframe_pattern = re.compile(
            r"features_e\d+[a-z]?_([a-zA-Z0-9\.]+)_neutralized"
        )
        for path in feature_paths:
            name_to_match = path.stem if path.is_file() else path.name
            match = timeframe_pattern.search(name_to_match)
            if not match:
                continue
            timeframe = match.group(1)
            lf = None
            if path.is_dir():
                glob_path = path / f"year={self.config.subset_year}"
                if glob_path.exists():
                    lf = pl.scan_parquet(str(glob_path / "**/*.parquet"))
            elif path.is_file():
                lf_full = pl.scan_parquet(str(path))
                if "timestamp" in lf_full.collect_schema().names():
                    lf = lf_full.filter(
                        pl.col("timestamp").dt.year() == self.config.subset_year
                    )

            if lf is not None:
                lf = lf.with_columns(pl.col("timestamp").cast(pl.Datetime("us", "UTC")))
                all_lazy_frames.append(
                    lf.with_columns(pl.lit(timeframe).alias("timeframe"))
                )

        if not all_lazy_frames:
            raise ValueError(
                f"No feature data found for year {self.config.subset_year}."
            )
        return pl.concat(all_lazy_frames, how="diagonal").sort("timestamp")

    def _load_all_price_data(self) -> Dict[str, Any]:
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

        atr_files = list(price_dir.glob("features_e1c_*.parquet"))
        if not atr_files:
            raise FileNotFoundError(f"No ATR parquet files found in {price_dir}")

        timeframe_pattern = re.compile(r"features_e1c_([a-zA-Z0-9\.]+)\.parquet")
        all_atr_lfs = []
        logging.info(f"   -> Found {len(atr_files)} ATR source files in {price_dir}.")

        for f in atr_files:
            match = timeframe_pattern.search(f.name)
            if not match:
                continue
            timeframe = match.group(1)
            target_atr_name = f"e1c_atr_21_{timeframe}"
            lf = pl.scan_parquet(f).with_columns(
                pl.col("timestamp").cast(pl.Datetime("us", "UTC"))
            )
            atr_col_to_rename = next(
                (col for col in lf.collect_schema().names() if "e1c_atr_21" in col),
                None,
            )

            if target_atr_name in lf.columns:
                all_atr_lfs.append(lf.select(["timestamp", target_atr_name]))
            elif atr_col_to_rename:
                all_atr_lfs.append(
                    lf.select(["timestamp", atr_col_to_rename]).rename(
                        {atr_col_to_rename: target_atr_name}
                    )
                )
            else:
                logging.warning(f"   -> No ATR column found in {f.name}. Skipping.")

        if not all_atr_lfs:
            raise ValueError(
                "FATAL: No valid ATR columns were extracted from any price files."
            )
        logging.info(f"   -> Prepared {len(all_atr_lfs)} ATR blueprints.")
        return {"base_lf": base_lf, "atr_lfs": all_atr_lfs}

    def _discover_partitions(self, unified_lf: pl.LazyFrame) -> pl.DataFrame:
        return (
            unified_lf.select(pl.col("timestamp").dt.date().alias("date"))
            .unique()
            .collect()
            .sort("date")
        )

    def _get_bar_duration_minutes(self, timeframe: str) -> float:
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
        cfg = self.config
        if daily_bets_df.is_empty():
            return None

        labeled_chunks = []
        for timeframe_tuple, group_df in daily_bets_df.group_by("timeframe"):
            timeframe = timeframe_tuple[0]
            if timeframe is None or group_df.is_empty():
                continue

            target_duration_minutes = float(cfg.target_duration.replace("m", ""))
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

            atr_col_name = f"e1c_atr_21_{timeframe}"
            # NOTE: This is the original logic. If ATR for a timeframe (like 'tick') is not found,
            # it silently skips, which is the correct, intended behavior. No warning is logged.
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
            logging.warning("No samples were processed. The output is empty.")
            return
        pos = self.label_counts.get(1, 0)
        neg = self.label_counts.get(-1, 0)
        scale_pos_weight = neg / pos if pos > 0 else 1.0
        summary = (
            "\n" + "=" * 60 + "\n"
            "### Context-Adaptive Labeling COMPLETED! ###\n"
            f"The '{self.config.target_duration}' version of the 'Problem Set' is ready at: {self.config.output_dir}\n"
            f"  - Total Labeled Samples: {total_samples}\n"
            f"  - (+) Profit-Take: {pos}, (-) Stop-Loss: {neg}, (0) Timed-Out: {self.label_counts.get(0, 0)}\n"
            f"  - Calculated `scale_pos_weight` for next step: {scale_pos_weight:.4f}\n"
            + "="
            * 60
        )
        logging.info(summary)

    # --- [ADDED] New Methods for Reporting ---

    def _collect_report_data(self, df: pl.DataFrame, current_date: dt.date):
        """Collects data from the daily labeled dataframe for the final report."""
        report_df = df.with_columns(
            (pl.col("t1") - pl.col("timestamp"))
            .dt.total_seconds()
            .alias("event_duration_seconds"),
            pl.lit(current_date).alias("date"),
        ).select(["timeframe", "label", "event_duration_seconds", "date"])
        self.report_data.extend(report_df.to_dicts())

    def _generate_report(self):
        """Generates a detailed markdown report of the execution results."""
        logging.info("Generating detailed execution report...")
        cfg = self.config
        report_path = cfg.output_dir / f"execution_report_{cfg.target_duration}.md"

        if not self.report_data:
            logging.warning("No data available for report generation.")
            report_content = f"# Proxy Labeling Engine - Execution Report\n\n**No samples were processed for duration '{cfg.target_duration}'. The output is empty.**"
            report_path.write_text(report_content)
            return

        df = pl.from_dicts(self.report_data)

        summary_table = f"""
| Item | Value |
|:---|:---|
| **Execution Timestamp** | `{cfg.execution_start_time}` |
| **Script Path** | `{" / ".join(Path(__file__).parts[-4:])}` |
| **Target Year** | `{cfg.subset_year}` |
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

        duration_stats = (df["event_duration_seconds"] / 60).describe()
        duration_table = f"""
| Statistic | Duration (minutes) |
|:---|:---|
| **Mean** | `{duration_stats.get_row(3).to_numpy()[0, 1]:.2f}` |
| **Median (50%)** | `{duration_stats.get_row(5).to_numpy()[0, 1]:.2f}` |
| **Std Dev** | `{duration_stats.get_row(4).to_numpy()[0, 1]:.2f}` |
| **Min** | `{duration_stats.get_row(7).to_numpy()[0, 1]:.2f}` |
| **Max** | `{duration_stats.get_row(8).to_numpy()[0, 1]:.2f}` |
| **25th Percentile** | `{duration_stats.get_row(9).to_numpy()[0, 1]:.2f}` |
| **75th Percentile** | `{duration_stats.get_row(10).to_numpy()[0, 1]:.2f}` |
"""
        daily_activity = (
            df.group_by("date").len().sort("len", descending=True).limit(10)
        )
        daily_table = "| Date | Labeled Samples |\n|:---|---:|\n"
        for row in daily_activity.to_dicts():
            daily_table += f"| `{row['date']}` | `{row['len']:,}` |\n"

        report_content = f"""
# Proxy Labeling Engine - Execution Report 📝

---

### 1. Execution Summary
{summary_table.strip()}

---

### 2. Overall Performance 📊
{perf_table.strip()}

---

### 3. Timeframe Breakdown 🕒
This table shows which timeframes generated the most betting opportunities and their outcomes.

{tf_table.strip()}

---

### 4. Event Duration Analysis ⏳
This table analyzes the time it took for an event to conclude (hit a barrier or time out).

{duration_table.strip()}

---

### 5. Top 10 Busiest Days 🗓️
This table lists the days with the highest number of labeled samples, indicating periods of high activity.

{daily_table.strip()}
"""
        report_path.write_text(report_content.strip())
        logging.info(f"Detailed execution report saved to: {report_path}")


# --- [ADDED] Interactive Mode Functionality ---
def _get_interactive_config() -> ProxyLabelConfig:
    """Gets configuration from the user interactively."""
    print("\n[ Interactive Configuration Mode ]")
    print("Enter parameters. Press Enter to use the default value shown in [].\n")

    def get_input(prompt: str, default: Any, type_converter: Any) -> Any:
        while True:
            val = input(f"{prompt} [{default}]: ").strip()
            if not val:
                return default
            try:
                return type_converter(val)
            except ValueError:
                print(
                    f"  -> Invalid input. Please enter a valid {type_converter.__name__}."
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

    duration = get_input(
        "1. Target lookahead duration (e.g., '300m', '60m')", "300m", str
    )
    year = get_input("2. Year of data to process (e.g., 2023)", 2023, int)
    pt_mult = get_input("3. Profit-Take ATR multiplier", 2.0, float)
    sl_mult = get_input("4. Stop-Loss ATR multiplier", 1.0, float)
    resume = get_bool_input("5. Resume from previous run?", True)

    config = ProxyLabelConfig(
        target_duration=duration,
        subset_year=year,
        profit_take_multiplier=pt_mult,
        stop_loss_multiplier=sl_mult,
        resume=resume,
    )

    print("\n" + "-" * 50)
    print("Configuration Summary:")
    print(f"  - Target Duration: {config.target_duration}")
    print(f"  - Subset Year: {config.subset_year}")
    print(f"  - PT Multiplier: {config.profit_take_multiplier}")
    print(f"  - SL Multiplier: {config.stop_loss_multiplier}")
    print(f"  - Resume: {config.resume}")
    print("-" * 50)

    if not get_bool_input("Is this configuration correct?", True):
        print("Aborted.")
        sys.exit(0)

    return config


if __name__ == "__main__":
    if len(sys.argv) > 1:
        parser = argparse.ArgumentParser(
            description="Create Context-Adaptive Proxy Labels."
        )
        parser.add_argument(
            "--duration",
            type=str,
            default="300m",
            help="Target lookahead duration (e.g., '300m').",
        )
        parser.add_argument(
            "--year", type=int, default=2023, help="Year of data to process."
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

        config = ProxyLabelConfig(
            target_duration=args.duration,
            subset_year=args.year,
            resume=not args.no_resume,
            profit_take_multiplier=args.pt_mult,
            stop_loss_multiplier=args.sl_mult,
        )
    else:
        config = _get_interactive_config()

    print("\nStarting engine with the specified configuration...")
    engine = ProxyLabelingEngine(config)
    engine.run()
