# /workspace/models/proxy_optimization/calculate_proxy_uniqueness.py

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
OPTIMIZATION_DIR = (
    project_root / "data/XAUUSD/stratum_7_models/1A_2B/optimal_parameters"
)

# Input: The monthly aggregated subset
INPUT_PATH = OPTIMIZATION_DIR / "temp_labeled_subset_monthly"

# Output: The intermediate concurrency results
OUTPUT_PATH = OPTIMIZATION_DIR / "temp_concurrency_results.parquet"

# DuckDB temporary directory
DUCKDB_TEMP_DIR_CONTAINER = "/duckdb_temp"


def main():
    """
    Calculates event concurrency using the highly efficient window function algorithm on the
    lightweight, monthly-aggregated dataset.
    """
    logging.info("### Phase 1, Script 3: Calculate Uniqueness (Calculate Step) ###")

    if not INPUT_PATH.exists():
        logging.error(f"CRITICAL: Input directory not found at {INPUT_PATH}")
        logging.error("Please run aggregate_proxy_subset.py first.")
        return

    # Clean up previous runs
    if OUTPUT_PATH.exists():
        logging.warning(
            f"Output file {OUTPUT_PATH} exists. Removing it for a clean run."
        )
        os.remove(OUTPUT_PATH)
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    # Ensure DuckDB temp directory exists
    os.makedirs(DUCKDB_TEMP_DIR_CONTAINER, exist_ok=True)

    logging.info(f"Input data path: {INPUT_PATH}")
    logging.info(f"Temporary directory for DuckDB: {DUCKDB_TEMP_DIR_CONTAINER}")
    logging.info(f"Output path for concurrency results: {OUTPUT_PATH}")

    # The battle-tested, final SQL query that avoids self-joins and OOM errors.
    # This query implements the "event sourcing" paradigm.
    query = f"""
    -- Critical settings for stability and performance
    SET temp_directory = '{DUCKDB_TEMP_DIR_CONTAINER}';
    SET memory_limit = '20GB';
    SET preserve_insertion_order = false;
    SET enable_progress_bar = true;
    SET threads = 12;

    -- Directly copy the final result to the output file
    COPY (
        WITH event_stream AS (
            -- Step 1: Decompose each event [t0, t1] into two point-in-time events:
            -- a +1 "start" event at t0 and a -1 "end" event at t1.
            -- This also generates a unique event_id on the fly.
            SELECT
                row_number() OVER (ORDER BY timestamp, t1) AS event_id,
                timestamp AS event_time,
                1 AS type
            FROM read_parquet('{str(INPUT_PATH)}/**/*.parquet', hive_partitioning=true)

            UNION ALL

            SELECT
                row_number() OVER (ORDER BY timestamp, t1) AS event_id,
                t1 AS event_time,
                -1 AS type
            FROM read_parquet('{str(INPUT_PATH)}/**/*.parquet', hive_partitioning=true)
        ),
        concurrency_calc AS (
            -- Step 2: Calculate the running cumulative sum of the 'type' column.
            -- This sum at any given point in time is the exact number of concurrent events.
            SELECT
                event_id,
                event_time,
                SUM(type) OVER (ORDER BY event_time ASC, type DESC) AS concurrency
            FROM event_stream
        ),
        final_concurrency AS (
            -- Step 3: Filter for the "start" events (where concurrency is calculated)
            -- and select only the event_id and its calculated concurrency.
            SELECT
                event_id,
                concurrency
            FROM concurrency_calc
            -- We group by event_id to ensure we only get one value per original event.
            -- The MAX is just to satisfy the aggregation, the value will be the same.
            GROUP BY event_id, concurrency
        )
        -- Final output selection
        SELECT * FROM final_concurrency
        ORDER BY event_id

    ) TO '{str(OUTPUT_PATH)}' (FORMAT PARQUET);
    """

    try:
        con = duckdb.connect()
        logging.info("Executing the proven, optimized concurrency query...")
        con.execute(query)
        logging.info("Concurrency calculation query executed successfully.")
    except Exception as e:
        logging.error(f"An error occurred during DuckDB execution: {e}", exc_info=True)
        raise
    finally:
        if "con" in locals():
            con.close()
            logging.info("DuckDB connection closed.")

    logging.info("\n" + "=" * 60)
    logging.info("### Uniqueness Calculation COMPLETED! ###")
    logging.info(f"The 'engine block' of uniqueness is forged at: {OUTPUT_PATH}")
    logging.info("Ready for the final join step.")
    logging.info("=" * 60)


if __name__ == "__main__":
    main()
