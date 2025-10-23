# /workspace/models/proxy_optimization/aggregate_proxy_subset.py

import sys
import duckdb
import logging
from pathlib import Path
import os
import shutil

# --- Project Path Setup ---
project_root = Path(__file__).resolve().parents[2]
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

# --- Logging Setup ---
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

# --- ★★★ Path Definitions (Proxy Optimization Phase) ★★★ ---
# The central directory for this optimization phase
OPTIMIZATION_DIR = (
    project_root / "data/XAUUSD/stratum_7_models/1A_2B/optimal_parameters"
)

# Input: The daily partitioned subset we just created
INPUT_PATH = OPTIMIZATION_DIR / "temp_labeled_subset_partitioned_5_08_02"

# Output: The monthly aggregated version of the subset
OUTPUT_PATH = OPTIMIZATION_DIR / "temp_labeled_subset_monthly"


def main():
    """
    Aggregates the lightweight daily partitioned data into a monthly format
    to dramatically speed up the subsequent uniqueness calculation step.
    """
    logging.info("### Phase 1, Script 2: Aggregate Daily Subset to Monthly ###")

    if not INPUT_PATH.exists():
        logging.error(f"CRITICAL: Input directory not found at {INPUT_PATH}")
        logging.error("Please run create_proxy_labels.py first.")
        return

    # Clean up previous runs
    if OUTPUT_PATH.exists():
        logging.warning(
            f"Output directory {OUTPUT_PATH} exists. Removing it for a clean aggregation."
        )
        shutil.rmtree(OUTPUT_PATH)

    # Ensure the parent directory exists
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    logging.info(f"Input data path: {INPUT_PATH}")
    logging.info(f"Output path for monthly aggregation: {OUTPUT_PATH}")

    # DuckDB is extremely efficient for this kind of large-scale aggregation.
    # It reads all daily files, re-partitions them by month, and writes the output.
    query = f"""
    SET enable_progress_bar = true;
    SET threads TO 12;

    COPY (
        SELECT *
        FROM read_parquet('{str(INPUT_PATH)}/**/*.parquet')
    ) TO '{str(OUTPUT_PATH)}' (
        FORMAT PARQUET,
        PARTITION_BY (year, month),
        OVERWRITE_OR_IGNORE 1
    );
    """

    try:
        con = duckdb.connect()
        logging.info("Executing aggregation query with DuckDB...")
        con.execute(query)
        logging.info("Aggregation query executed successfully.")
    except Exception as e:
        logging.error(f"An error occurred during DuckDB execution: {e}", exc_info=True)
        raise
    finally:
        if "con" in locals():
            con.close()
            logging.info("DuckDB connection closed.")

    logging.info("\n" + "=" * 60)
    logging.info("### Aggregation COMPLETED! ###")
    logging.info(f"Monthly aggregated subset is ready at: {OUTPUT_PATH}")
    logging.info("Ready for the uniqueness weighting calculation.")
    logging.info("=" * 60)


if __name__ == "__main__":
    main()
