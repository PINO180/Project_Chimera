# /workspace/models/optimal_parameters/calibrate_proxy_model.py
# FINAL VERSION - Multi-class calibration

import sys
from pathlib import Path
import logging
import warnings
from dataclasses import dataclass

import polars as pl
import numpy as np
import joblib
from sklearn.isotonic import IsotonicRegression
from typing import List, Tuple, Dict
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
    Ensures consistency with the multi-class training data preparation.
    """
    logging.info("Loading all training data into memory for calibration...")
    X_list, y_list = [], []

    # This label map must be identical to the one in the training script
    label_map = {-1: 0, 0: 1, 1: 2}  # SL -> 0, TO -> 1, PT -> 2

    for path in tqdm(partitions, desc="Loading Data Partitions"):
        try:
            df_chunk = pl.read_parquet(path)
        except Exception as e:
            logging.warning(f"Could not read {path}: {e}. Skipping.")
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

        X_chunk = (
            df_chunk.select(features)
            .cast(pl.Float64, strict=False)
            .fill_null(0)
            .to_numpy()
        )
        X_list.append(X_chunk)

        # --- ★★★ CORE CHANGE: Load labels for multi-class ★★★ ---
        y_chunk = df_chunk["label"].replace_strict(label_map, default=1).to_numpy()
        y_list.append(y_chunk)
        # --------------------------------------------------------

    logging.info("Concatenating all data chunks...")
    X_all = np.vstack(X_list)
    y_all = np.concatenate(y_list)

    del X_list, y_list, X_chunk, y_chunk
    gc.collect()

    logging.info(f"  -> Data loaded. Total samples: {len(X_all)}")
    return X_all, y_all


def calibrate_proxy_model(config: ProxyCalibrationConfig):
    """
    Trains an Isotonic Regression model for each class (SL, TO, PT) to calibrate
    the multi-class proxy model's outputs.
    """
    logging.info("### Phase 1, Script 6: Calibrate Multi-Class Proxy Model ###")

    # --- 1. Validation and Setup ---
    if not all(
        [
            config.input_data_dir.exists(),
            config.input_model_path.exists(),
            config.feature_list_path.exists(),
        ]
    ):
        logging.error("CRITICAL: One or more input paths not found. Aborting.")
        return
    config.output_calibrator_path.parent.mkdir(parents=True, exist_ok=True)

    # --- 2. Load Features and Data Partitions ---
    logging.info(f"Loading feature list from {config.feature_list_path}...")
    with open(config.feature_list_path, "r") as f:
        features = [line.strip() for line in f if line.strip()]

    partitions = sorted(config.input_data_dir.glob("year=*/month=*/day=*/*.parquet"))
    if not partitions:
        raise FileNotFoundError(f"No data partitions found in {config.input_data_dir}")

    # --- 3. Load All Data into Memory ---
    X_all, y_all = load_all_data_for_calibration(partitions, features)

    # --- 4. Get Raw Predictions from Proxy Model ---
    logging.info(f"Loading proxy model from {config.input_model_path}...")
    proxy_model = joblib.load(config.input_model_path)

    logging.info("Generating raw probability predictions for all 3 classes...")
    # This will be an (n_samples, 3) array
    raw_probabilities = proxy_model.predict_proba(X_all)
    del X_all
    gc.collect()

    # --- ★★★ CORE CHANGE: Train one calibrator per class ★★★ ---
    logging.info("Training one Isotonic Regression calibrator per class...")

    # We will store the 3 calibrators in a dictionary
    calibrators: Dict[int, IsotonicRegression] = {}
    class_map = {0: "SL", 1: "TO", 2: "PT"}

    for i in range(raw_probabilities.shape[1]):  # Should be 3 classes
        class_name = class_map.get(i, f"Class {i}")
        logging.info(f"  -> Calibrating for class {i} ({class_name})...")

        # Get the raw probabilities for the current class
        raw_proba_class = raw_probabilities[:, i]

        # Create a binary target for the current class (1 if this class, 0 otherwise)
        y_true_class = (y_all == i).astype(int)

        # Train a separate calibrator for this class
        calibrator = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip")
        calibrator.fit(raw_proba_class, y_true_class)

        calibrators[i] = calibrator
        logging.info(f"     -> Done.")

    logging.info("  -> All calibrators trained.")
    # -------------------------------------------------------------

    # --- 6. Save the Collection of Calibrators ---
    # Save the dictionary containing all 3 calibrator models
    joblib.dump(calibrators, config.output_calibrator_path)

    logging.info("\n" + "=" * 60)
    logging.info("### Multi-Class Proxy Model Calibration COMPLETED! ###")
    logging.info(f"The multi-class calibrator (dictionary of 3 models) is ready at:")
    logging.info(f"  -> {config.output_calibrator_path}")
    logging.info(
        "The high-speed optimization can now use these calibrated probabilities."
    )
    logging.info("=" * 60)


if __name__ == "__main__":
    config = ProxyCalibrationConfig()
    calibrate_proxy_model(config)
