# /workspace/models/optimal_parameters/create_proxy_feature_list.py

import sys
from pathlib import Path
import logging
import polars as pl
from typing import Set

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

# Input: The weighted subset we just created
SOURCE_DATA_DIR = OPTIMIZATION_DIR / "temp_weighted_subset_partitioned"

# Output: The feature list for the proxy model
OUTPUT_FEATURE_LIST = OPTIMIZATION_DIR / "proxy_feature_list.txt"


def find_first_valid_file(base_dir: Path) -> Path | None:
    """Finds the first available parquet file in the partitioned directory."""
    logging.info(f"Searching for a sample data file in {base_dir}...")
    if not base_dir.exists():
        return None

    files = list(base_dir.glob("year=*/month=*/day=*/*.parquet"))
    if not files:
        return None

    logging.info(f"  -> Found sample file: {files[0]}")
    return files[0]


def create_updated_feature_list(source_file: Path, output_file: Path):
    """
    Extracts feature column names from the weighted dataset to create a new list file,
    excluding all non-feature columns.
    """
    logging.info("--- Creating Updated Feature List for Proxy Model ---")

    if not source_file or not source_file.exists():
        logging.error(f"❌ ERROR: Source file not found: {source_file}")
        return

    try:
        # 1. Get all column names from the data schema
        df_columns = pl.read_parquet_schema(source_file).keys()

        # 2. Define all columns that are NOT features
        # This now includes columns added during labeling and weighting
        non_feature_cols: Set[str] = {
            "timestamp",
            "t1",
            "label",
            "year",
            "month",
            "day",
            "event_id",
            "uniqueness",
            "payoff_ratio",
        }

        # 3. Filter out the non-feature columns
        feature_cols = [col for col in df_columns if col not in non_feature_cols]

        # 4. Sort alphabetically for consistency
        feature_cols.sort()

        # 5. Write the clean list to the new file
        output_file.parent.mkdir(parents=True, exist_ok=True)
        with open(output_file, "w") as f:
            for feature in feature_cols:
                f.write(f"{feature}\n")

        logging.info(
            f"✅ Successfully created new feature list with {len(feature_cols)} features."
        )
        logging.info(f"   -> Saved to: {output_file}")

    except Exception as e:
        logging.error(f"❌ An error occurred: {e}", exc_info=True)


if __name__ == "__main__":
    # Find a representative file automatically instead of hardcoding the path
    sample_file = find_first_valid_file(SOURCE_DATA_DIR)

    if sample_file:
        create_updated_feature_list(sample_file, OUTPUT_FEATURE_LIST)
    else:
        logging.error(f"Could not find any data files in {SOURCE_DATA_DIR}. Aborting.")
