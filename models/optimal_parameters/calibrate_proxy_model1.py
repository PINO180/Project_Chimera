# /workspace/models/optimal_parameters/calibrate_proxy_model.py

import sys
from pathlib import Path
import logging
import warnings
from dataclasses import dataclass

import polars as pl
import numpy as np
import joblib
from sklearn.isotonic import IsotonicRegression
from typing import List, Tuple
from tqdm import tqdm
import gc

# --- Project Path Setup ---
project_root = Path(__file__).resolve().parents[2]
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

# --- Logging Setup ---
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
warnings.filterwarnings("ignore", category=UserWarning)

# --- Path Definitions (Proxy Optimization Phase) ---
OPTIMIZATION_DIR = (
    project_root / "data/XAUUSD/stratum_7_models/1A_2B/optimal_parameters"
)

# --- Inputs ---
# The data used to train the proxy model
INPUT_DATA_DIR = OPTIMIZATION_DIR / "temp_weighted_subset_partitioned"
# The proxy model itself (the 'scout')
INPUT_PROXY_MODEL_PATH = OPTIMIZATION_DIR / "proxy_model.pkl"
# The feature list used by the proxy model
FEATURE_LIST_PATH = OPTIMIZATION_DIR / "proxy_feature_list.txt"

# --- Output ---
# The calibrator (the 'megaphone')
OUTPUT_CALIBRATOR_PATH = OPTIMIZATION_DIR / "proxy_model_calibrator.pkl"


@dataclass
class ProxyCalibrationConfig:
    """Configuration for calibrating the proxy model's probabilities."""

    input_data_dir: Path = INPUT_DATA_DIR
    input_model_path: Path = INPUT_PROXY_MODEL_PATH
    feature_list_path: Path = FEATURE_LIST_PATH
    output_calibrator_path: Path = OUTPUT_CALIBRATOR_PATH


def load_all_data_for_calibration(
    partitions: List[Path], features: List[str]
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Loads all necessary data from the partitioned directory into memory.
    This mimics the exact data loading process used for model training to ensure consistency.
    """
    logging.info("Loading all training data into memory for calibration...")
    X_list, y_list = [], []

    for path in tqdm(partitions, desc="Loading Data Partitions"):
        try:
            df_chunk = pl.read_parquet(path)
        except Exception:
            continue

        if df_chunk.is_empty():
            continue

        # Use the same robust feature handling as in the training script
        current_cols = set(df_chunk.columns)
        missing_cols = [f for f in features if f not in current_cols]
        if missing_cols:
            df_chunk = df_chunk.with_columns(
                [pl.lit(None, dtype=pl.Float64).alias(col) for col in missing_cols]
            )

        X_list.append(df_chunk.select(features).to_numpy())
        y_list.append(np.where(df_chunk["label"] == 1, 1, 0))

    logging.info("Concatenating all data chunks...")
    X_all = np.vstack(X_list)
    y_all = np.concatenate(y_list)

    del X_list, y_list
    gc.collect()

    logging.info(f"  -> Data loaded. Total samples: {len(X_all)}")
    return X_all, y_all


def calibrate_proxy_model(config: ProxyCalibrationConfig):
    """
    Trains an Isotonic Regression model to calibrate the proxy model's outputs.
    """
    logging.info("### Phase 1, New Script: Calibrate Proxy Model (Build Megaphone) ###")

    # --- 1. Validation and Setup ---
    if not config.input_data_dir.exists():
        logging.error(
            f"CRITICAL: Input data directory not found at {config.input_data_dir}"
        )
        return
    if not config.input_model_path.exists():
        logging.error(f"CRITICAL: Proxy model not found at {config.input_model_path}")
        return
    if not config.feature_list_path.exists():
        logging.error(f"CRITICAL: Feature list not found at {config.feature_list_path}")
        return

    config.output_calibrator_path.parent.mkdir(parents=True, exist_ok=True)

    # --- 2. Load Features and Data Partitions ---
    logging.info(f"Loading feature list from {config.feature_list_path}...")
    with open(config.feature_list_path, "r") as f:
        features = [line.strip() for line in f if line.strip()]
    logging.info(f"  -> Loaded {len(features)} features.")

    partitions = sorted(config.input_data_dir.glob("year=*/month=*/day=*/*.parquet"))
    if not partitions:
        raise FileNotFoundError(f"No data partitions found in {config.input_data_dir}")

    # --- 3. Load All Data into Memory ---
    X_all, y_all = load_all_data_for_calibration(partitions, features)

    # --- 4. Get Raw Predictions from Proxy Model (the 'Whispers') ---
    logging.info(f"Loading proxy model from {config.input_model_path}...")
    proxy_model = joblib.load(config.input_model_path)

    logging.info("Generating raw probability predictions (the 'whispers')...")
    raw_probabilities = proxy_model.predict_proba(X_all)[:, 1]

    # Free up memory from the full feature matrix
    del X_all
    gc.collect()

    # --- 5. Train the Calibrator (the 'Megaphone') ---
    logging.info("Training the Isotonic Regression calibrator (the 'megaphone')...")

    # IsotonicRegression is perfect for converting a model's biased outputs
    # into a properly calibrated probability scale.
    calibrator = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip")

    # We fit the calibrator on the raw probabilities and the true outcomes.
    # It learns the mapping: "When the scout whispers '0.002%', the actual win rate was 2%".
    calibrator.fit(raw_probabilities, y_all)

    logging.info("  -> Calibrator training complete.")

    # --- 6. Save the Calibrator ---
    joblib.dump(calibrator, config.output_calibrator_path)

    logging.info("\n" + "=" * 60)
    logging.info("### Proxy Model Calibration COMPLETED! ###")
    logging.info(
        f"The 'Megaphone' (Calibrator) is ready at: {config.output_calibrator_path}"
    )
    logging.info(
        "The high-speed optimization in Phase 2 can now begin with calibrated probabilities."
    )
    logging.info("=" * 60)


if __name__ == "__main__":
    config = ProxyCalibrationConfig()
    calibrate_proxy_model(config)
