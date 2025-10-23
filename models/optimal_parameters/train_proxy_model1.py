# /workspace/models/optimal_parameters/train_proxy_model.py

import sys
from pathlib import Path
import logging
import argparse
import warnings
from dataclasses import dataclass, field

import polars as pl
import numpy as np
import lightgbm as lgb
import joblib
from typing import List, Dict, Any
from tqdm import tqdm

# --- Project Path Setup ---
project_root = Path(__file__).resolve().parents[2]
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

# --- Blueprint Imports ---
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
# The new feature list path for the proxy model
FEATURE_LIST_PATH = OPTIMIZATION_DIR / "proxy_feature_list.txt"


@dataclass
class ProxyTrainingConfig:
    """Configuration for training the lightweight proxy model."""

    input_dir: Path = INPUT_PATH
    feature_list_path: Path = FEATURE_LIST_PATH
    output_model_path: Path = OUTPUT_PATH
    lgbm_params: Dict[str, Any] = field(
        default_factory=lambda: {
            "objective": "binary",
            "metric": "auc",
            "boosting_type": "gbdt",
            "n_estimators": 500,
            "learning_rate": 0.05,
            "num_leaves": 31,
            "max_depth": 7,
            "seed": 42,
            "n_jobs": -1,
            "verbose": -1,
            "colsample_bytree": 0.8,
            "subsample": 0.8,
        }
    )
    test_limit: int = 0


def calculate_scale_pos_weight(partitions: List[Path]) -> float:
    """
    Calculates the scale_pos_weight by scanning all partitions to count labels.
    """
    logging.info("Calculating scale_pos_weight from all partitions...")
    pos_count = 0
    neg_count = 0

    for path in tqdm(partitions, desc="Scanning for Labels"):
        try:
            labels = pl.read_parquet(path, columns=["label"])["label"]
            pos_count += labels.filter(labels == 1).len()
            neg_count += labels.filter(labels == -1).len()
        except Exception as e:
            logging.warning(f"Could not read labels from {path}: {e}")
            continue

    if pos_count == 0:
        logging.warning("No positive samples found. Defaulting scale_pos_weight to 1.0")
        return 1.0

    scale_pos_weight = neg_count / pos_count
    logging.info(f"  -> Positive samples: {pos_count}, Negative samples: {neg_count}")
    logging.info(f"  -> Calculated scale_pos_weight: {scale_pos_weight:.4f}")
    return scale_pos_weight


def train_proxy_model(config: ProxyTrainingConfig):
    """
    Trains a single, simple LightGBM model on the entire weighted subset.
    """
    logging.info("### Phase 1, Script 5: Train Proxy Model ###")

    if not config.input_dir.exists():
        logging.error(f"CRITICAL: Input directory not found at {config.input_dir}")
        return

    config.output_model_path.parent.mkdir(parents=True, exist_ok=True)

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

    scale_pos_weight = calculate_scale_pos_weight(partitions)
    config.lgbm_params["scale_pos_weight"] = scale_pos_weight

    logging.info("Starting sequential training of the proxy model...")
    model = lgb.LGBMClassifier(**config.lgbm_params)
    is_first_chunk = True

    for path in tqdm(partitions, desc="Training Proxy Model"):
        try:
            df_chunk = pl.read_parquet(path)
        except Exception:
            continue

        if df_chunk.is_empty():
            continue

        # --- ★★★ THE FINAL, ROBUST FIX ★★★ ---
        # 1. Check which features from the master list are missing in this chunk.
        current_cols = set(df_chunk.columns)
        missing_cols = [f for f in features if f not in current_cols]

        # 2. If any are missing, add them as null columns.
        if missing_cols:
            df_chunk = df_chunk.with_columns(
                [pl.lit(None, dtype=pl.Float64).alias(col) for col in missing_cols]
            )

        # 3. Now, select all features in the correct, consistent order. This will no longer fail.
        X_chunk = df_chunk.select(features).to_numpy()
        # --- ★★★ FIX ENDS ★★★ ---

        y_chunk = np.where(df_chunk["label"] == 1, 1, 0)
        w_chunk = df_chunk["uniqueness"].to_numpy()

        model.fit(
            X_chunk,
            y_chunk,
            sample_weight=w_chunk,
            init_model=None if is_first_chunk else model.booster_,
        )
        is_first_chunk = False

    logging.info("Proxy model training completed.")
    joblib.dump(model, config.output_model_path)

    logging.info("\n" + "=" * 60)
    logging.info("### Proxy Model Training COMPLETED! ###")
    logging.info(f"The 'Proxy AI' is ready at: {config.output_model_path}")
    logging.info("We are now ready to begin the high-speed optimization in Phase 2.")
    logging.info("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train the proxy model for optimization."
    )
    parser.add_argument(
        "--test",
        type=int,
        default=0,
        help="Run in test mode, limiting training to the first N partitions.",
    )
    args = parser.parse_args()

    config = ProxyTrainingConfig(test_limit=args.test)
    train_proxy_model(config)
