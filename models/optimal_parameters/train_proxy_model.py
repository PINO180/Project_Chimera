# /workspace/models/optimal_parameters/train_proxy_model.py
# FINAL VERSION - Multi-class classification with class weighting

import sys
from pathlib import Path
import logging
import argparse
import warnings
from dataclasses import dataclass, field
import gc

import polars as pl
import numpy as np
import lightgbm as lgb
import joblib
from typing import List, Dict, Any, Tuple
from tqdm import tqdm
from collections import Counter

# --- Project Path Setup ---
project_root = Path(__file__).resolve().parents[2]
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

# --- Blueprint Imports ---
# NOTE: S3_FEATURES_FOR_TRAINING might not be strictly needed here if
# proxy_feature_list.txt is used, but kept for potential future reference.
from blueprint import S3_FEATURES_FOR_TRAINING

# --- Logging Setup ---
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
warnings.filterwarnings("ignore", category=UserWarning)

# --- Path Definitions (Proxy Optimization Phase) ---
OPTIMIZATION_DIR = (
    project_root / "data/XAUUSD/stratum_7_models/1A_2B/optimal_parameters"
)
INPUT_PATH = OPTIMIZATION_DIR / "temp_weighted_subset_partitioned"
OUTPUT_PATH = OPTIMIZATION_DIR / "proxy_model.pkl"
FEATURE_LIST_PATH = OPTIMIZATION_DIR / "proxy_feature_list.txt"


@dataclass
class ProxyTrainingConfig:
    """Configuration for training the multi-class proxy model."""

    input_dir: Path = INPUT_PATH
    feature_list_path: Path = FEATURE_LIST_PATH
    output_model_path: Path = OUTPUT_PATH
    lgbm_params: Dict[str, Any] = field(
        default_factory=lambda: {
            # --- ★★★ CORE CHANGE: Multi-class objective ★★★ ---
            "objective": "multiclass",
            "metric": "multi_logloss",  # Common metric for multi-class
            "num_class": 3,  # PT=1, SL=-1, TO=0 -> mapped to 0, 1, 2
            # ----------------------------------------------------
            "boosting_type": "gbdt",
            "n_estimators": 500,  # Kept relatively low for proxy model speed
            "learning_rate": 0.05,
            "num_leaves": 31,
            "max_depth": 7,
            "seed": 42,
            "n_jobs": -1,
            "verbose": -1,
            "colsample_bytree": 0.8,
            "subsample": 0.8,
            # 'class_weight' will be added dynamically based on data
        }
    )
    test_limit: int = 0  # Number of partitions to process for testing


def calculate_class_weights(partitions: List[Path]) -> Dict[int, float]:
    """
    Calculates class weights based on inverse frequency for multi-class objective.
    Maps labels {-1 (SL), 0 (TO), 1 (PT)} to {0, 1, 2}.
    """
    logging.info("Calculating class weights for multi-class objective...")
    label_counts = Counter()
    total_samples = 0

    label_map = {-1: 0, 0: 1, 1: 2}  # SL -> 0, TO -> 1, PT -> 2

    for path in tqdm(partitions, desc="Scanning for Labels"):
        try:
            # Only read the 'label' column
            df_labels = pl.read_parquet(path, columns=["label"])

            # --- ★★★ FIX 2 (DeprecationWarning) ★★★ ---
            # Map labels and count occurrences using replace_strict
            mapped_labels = (
                df_labels["label"].replace_strict(label_map, default=1).drop_nulls()
            )  # Drop any unexpected nulls
            # -------------------------------------------

            counts_in_partition = mapped_labels.value_counts()

            for i in range(3):  # Iterate through classes 0, 1, 2
                count = (
                    counts_in_partition.filter(pl.col("label") == i)
                    .select(pl.col("count"))
                    .item()
                )
                if count:
                    label_counts[i] += count
                    total_samples += count

        except Exception as e:
            logging.warning(f"Could not read labels from {path}: {e}")
            continue

    if total_samples == 0:
        logging.warning(
            "No labels found. Using default weights {0: 1.0, 1: 1.0, 2: 1.0}"
        )
        return {0: 1.0, 1: 1.0, 2: 1.0}

    # Calculate weights: weight = total_samples / (num_classes * count_for_class)
    num_classes = 3
    class_weights = {
        i: total_samples / (num_classes * label_counts[i])
        if label_counts[i] > 0
        else 1.0
        for i in range(num_classes)
    }

    logging.info(f"  -> Total samples: {total_samples}")
    logging.info(f"  -> Label counts (mapped): {dict(label_counts)}")
    logging.info(f"  -> Calculated class weights: {class_weights}")
    return class_weights


def train_proxy_model(config: ProxyTrainingConfig):
    """
    Trains a single, simple LightGBM multi-class model.
    """
    logging.info("### Phase 1, Script 5: Train Multi-Class Proxy Model ###")

    # --- Validation and Setup ---
    if not config.input_dir.exists():
        logging.error(f"CRITICAL: Input directory not found at {config.input_dir}")
        return
    config.output_model_path.parent.mkdir(parents=True, exist_ok=True)

    # --- Load Features ---
    logging.info(f"Loading feature list from {config.feature_list_path}...")
    with open(config.feature_list_path, "r") as f:
        features = [line.strip() for line in f if line.strip()]
    logging.info(f"  -> Loaded {len(features)} features.")

    partitions = sorted(config.input_dir.glob("year=*/month=*/day=*/*.parquet"))
    if not partitions:
        raise FileNotFoundError(f"No data partitions found in {config.input_dir}")

    if config.test_limit > 0:
        logging.warning(
            f"--- TEST MODE: Training on the first {config.test_limit} partitions only. ---"
        )
        partitions = partitions[: config.test_limit]

    # --- Calculate Class Weights ---
    class_weights = calculate_class_weights(partitions)
    config.lgbm_params["class_weight"] = class_weights  # Add weights to LGBM params

    # --- Sequential Training ---
    logging.info("Starting sequential training of the multi-class proxy model...")
    model = lgb.LGBMClassifier(**config.lgbm_params)
    is_first_chunk = True

    for path in tqdm(partitions, desc="Training Proxy Model"):
        try:
            df_chunk = pl.read_parquet(path)
        except Exception as e:
            logging.warning(f"Could not read partition {path}: {e}. Skipping.")
            continue

        if df_chunk.is_empty():
            continue

        # --- Robust Feature Handling ---
        current_cols = set(df_chunk.columns)
        missing_cols = [f for f in features if f not in current_cols]
        if missing_cols:
            df_chunk = df_chunk.with_columns(
                [pl.lit(None, dtype=pl.Float64).alias(col) for col in missing_cols]
            )

        # --- ★★★ FIX 1 (ValueError) ★★★ ---
        # Cast all feature columns to float, converting strings like 'M1' to null.
        # Then, fill all nulls (original nulls + new ones from cast) with 0.
        X_chunk = (
            df_chunk.select(features)
            .cast(pl.Float64, strict=False)
            .fill_null(0)
            .to_numpy()
        )
        # --- End Robust Feature Handling ---

        # --- ★★★ CORE CHANGE: Prepare labels for multi-class ★★★ ---
        # Map original labels {-1, 0, 1} to {0, 1, 2} for multiclass objective
        label_map = {-1: 0, 0: 1, 1: 2}  # SL -> 0, TO -> 1, PT -> 2

        # --- ★★★ FIX 2 (DeprecationWarning) ★★★ ---
        y_chunk = (
            df_chunk["label"].replace_strict(label_map, default=1).to_numpy()
        )  # default to TO if unexpected label occurs
        # ----------------------------------------------------------

        w_chunk = df_chunk["uniqueness"].to_numpy()

        # Fit the model sequentially
        try:
            model.fit(
                X_chunk,
                y_chunk,
                sample_weight=w_chunk,
                init_model=None if is_first_chunk else model.booster_,
            )
            is_first_chunk = False
        except Exception as fit_error:
            logging.error(
                f"Error during model fitting for partition {path}: {fit_error}",
                exc_info=True,
            )
            # Decide whether to continue or stop
            # For now, let's log and continue to see if it happens frequently
            continue  # Skip this chunk

    # --- Save the Model ---
    if is_first_chunk:
        logging.error(
            "No data chunks were successfully processed. Model was not trained."
        )
    else:
        logging.info("Proxy model training completed.")
        joblib.dump(model, config.output_model_path)

        logging.info("\n" + "=" * 60)
        logging.info("### Multi-Class Proxy Model Training COMPLETED! ###")
        logging.info(
            f"The 'Multi-Class Proxy AI' is ready at: {config.output_model_path}"
        )
        logging.info("Ready for calibration and high-speed optimization.")
        logging.info("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train the multi-class proxy model.")
    parser.add_argument(
        "--test",
        type=int,
        default=0,
        help="Run in test mode, limiting training to the first N partitions (e.g., --test 5). Default is 0 (all partitions).",
    )
    args = parser.parse_args()

    config = ProxyTrainingConfig(test_limit=args.test)
    train_proxy_model(config)
