# /workspace/models/optimal_parameters/join_proxy_uniqueness.py

import sys
import polars as pl
import pyarrow.parquet as pq
from pathlib import Path
import logging
import shutil
from typing import Dict
from tqdm import tqdm

# --- Project Path Setup ---
project_root = Path(__file__).resolve().parents[2]
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

# --- Logging Setup ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(processName)s - %(message)s",
)

# --- ★★★ Path Definitions (Proxy Optimization Phase) ★★★ ---
OPTIMIZATION_DIR = (
    project_root / "data/XAUUSD/stratum_7_models/1A_2B/optimal_parameters"
)

# Input 1: The daily partitioned subset
INPUT_LABELED_PATH = OPTIMIZATION_DIR / "temp_labeled_subset_partitioned"

# Input 2: The concurrency results we just created
INPUT_CONCURRENCY_PATH = OPTIMIZATION_DIR / "temp_concurrency_results.parquet"

# Output: The final weighted subset for this phase
OUTPUT_PATH = OPTIMIZATION_DIR / "temp_weighted_subset_partitioned"


def get_partition_info(input_dir: Path) -> Dict[Path, Dict[str, int]]:
    """Calculates the row count and running offset for each daily partition file."""
    logging.info("Step 1: Calculating row counts and offsets for each partition...")
    partitions = sorted(input_dir.glob("year=*/month=*/day=*/*.parquet"))
    if not partitions:
        raise FileNotFoundError(f"No daily parquet files found in {input_dir}")

    partition_info = {}
    total_rows = 0
    for path in tqdm(partitions, desc="Scanning Partitions"):
        try:
            row_count = pq.ParquetFile(path).metadata.num_rows
            partition_info[path] = {"offset": total_rows, "rows": row_count}
            total_rows += row_count
        except Exception as e:
            logging.error(f"Failed to read metadata from {path}: {e}")
            raise

    logging.info(
        f"   -> Found {len(partitions)} daily partitions. Total rows: {total_rows}."
    )
    return partition_info


def main():
    """
    Main orchestration function. Joins concurrency results to the daily partitioned
    dataset in a memory-safe, sequential manner.
    """
    logging.info("### Phase 1, Script 4: Join Uniqueness (Join Step) ###")

    if not INPUT_CONCURRENCY_PATH.exists():
        logging.error(
            f"CRITICAL: Concurrency results not found at {INPUT_CONCURRENCY_PATH}"
        )
        logging.error("Please run calculate_proxy_uniqueness.py first.")
        return
    if not INPUT_LABELED_PATH.exists():
        logging.error(f"CRITICAL: Labeled subset not found at {INPUT_LABELED_PATH}")
        return

    # Clean up previous runs
    if OUTPUT_PATH.exists():
        logging.warning(
            f"Output directory {OUTPUT_PATH} exists. Removing it for a clean run."
        )
        shutil.rmtree(OUTPUT_PATH)
    OUTPUT_PATH.mkdir(parents=True)

    # --- Stage 1: Get info of each partition ---
    partition_info_map = get_partition_info(INPUT_LABELED_PATH)

    # --- Stage 2: Sequential processing loop ---
    logging.info(
        f"Stage 2: Starting sequential join for {len(partition_info_map)} daily partitions."
    )

    # Load the entire (but small) concurrency results into memory once
    concurrency_df = pl.read_parquet(INPUT_CONCURRENCY_PATH)

    for path, info in tqdm(partition_info_map.items(), desc="Joining Uniqueness"):
        try:
            offset = info["offset"]
            row_count = info["rows"]
            if row_count == 0:
                continue

            start_id = offset + 1
            end_id = offset + row_count

            # Filter the concurrency data for the current partition's range
            concurrency_slice = concurrency_df.filter(
                pl.col("event_id").is_between(start_id, end_id, closed="both")
            )

            # Read the daily data, add event_id, and join the slice
            labeled_df = pl.read_parquet(path)

            final_df = (
                labeled_df.sort("timestamp", "t1")
                .with_row_count(name="row_num", offset=offset)
                .with_columns((pl.col("row_num") + 1).alias("event_id"))
                .join(concurrency_slice, on="event_id", how="left")
                .with_columns(
                    (1.0 / pl.max_horizontal(pl.col("concurrency"), 1)).alias(
                        "uniqueness"
                    )
                )
                .select(pl.all().exclude(["row_num", "concurrency"]))
            )

            # Write the result to the new partitioned structure
            year = int(path.parent.parent.parent.name.split("=")[1])
            month = int(path.parent.parent.name.split("=")[1])
            day = int(path.parent.name.split("=")[1])

            output_dir = OUTPUT_PATH / f"year={year}/month={month}/day={day}"
            output_dir.mkdir(parents=True, exist_ok=True)
            final_df.write_parquet(output_dir / "data.parquet", compression="zstd")

        except Exception as e:
            partition_name = path.relative_to(INPUT_LABELED_PATH)
            logging.error(f"Processing for {partition_name} failed: {e}", exc_info=True)
            raise

    logging.info("\n" + "=" * 60)
    logging.info("### Uniqueness Join COMPLETED! ###")
    logging.info(f"The final weighted subset is ready at: {OUTPUT_PATH}")
    logging.info("Ready for the proxy model training.")
    logging.info("=" * 60)


if __name__ == "__main__":
    main()
