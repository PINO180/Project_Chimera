# /workspace/models/optimal_parameters/optimizer.py
# FINAL VERSION - Expected Value Optimization (with decision logic)

import sys
from pathlib import Path
import logging
import json
import warnings
from dataclasses import dataclass
import gc

import polars as pl
import numpy as np
import lightgbm as lgb
import joblib
import optuna
from optuna.samplers import TPESampler
from typing import List, Tuple, Dict, Any
from tqdm import tqdm

# --- Project Path Setup ---
project_root = Path(__file__).resolve().parents[2]
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

# --- Logging Setup ---
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=optuna.exceptions.ExperimentalWarning)

# --- Path Definitions (FINAL) ---
OPTIMIZATION_DIR = (
    project_root / "data/XAUUSD/stratum_7_models/1A_2B/optimal_parameters"
)
# Input 1: The lightweight data used for training
DATA_SOURCE_PATH = OPTIMIZATION_DIR / "temp_weighted_subset_partitioned"
# Input 2: The 'scout' model
PROXY_MODEL_PATH = OPTIMIZATION_DIR / "proxy_model.pkl"
# Input 3: The 'megaphone' calibrator (now a dictionary of 3 calibrators)
PROXY_CALIBRATOR_PATH = OPTIMIZATION_DIR / "proxy_model_calibrator.pkl"
# Input 4: The feature list
FEATURE_LIST_PATH = OPTIMIZATION_DIR / "proxy_feature_list.txt"
# Output: The final instruction sheet
OUTPUT_PARAMS_PATH = OPTIMIZATION_DIR / "optimal_parameters.json"


@dataclass
class OptimizerConfig:
    """Configuration for the Bayesian optimization process."""

    n_trials: int = 100
    n_startup_trials: int = 20
    payoff_ratio_range: Tuple[float, float] = (0.01, 100.0)


class Objective:
    """
    Optuna objective function that maximizes the expected value of the trading strategy.
    """

    def __init__(self, config: OptimizerConfig):
        self.config = config
        logging.info("Initializing Objective function for high-speed optimization...")

        # --- 1. Load all necessary components ---
        logging.info("  -> Loading 'scout' (proxy_model.pkl)...")
        self.proxy_model: lgb.LGBMClassifier = joblib.load(PROXY_MODEL_PATH)

        logging.info(
            "  -> Loading multi-class 'megaphone' (proxy_model_calibrator.pkl)..."
        )
        self.calibrators: Dict[int, Any] = joblib.load(PROXY_CALIBRATOR_PATH)

        with open(FEATURE_LIST_PATH, "r") as f:
            self.features = [line.strip() for line in f if line.strip()]

        # --- 2. Load data and generate calibrated probabilities ONCE ---
        logging.info("  -> Loading feature data into memory...")
        X_all = self._load_features_into_numpy(DATA_SOURCE_PATH, self.features)

        logging.info("  -> Generating raw multi-class 'whispers' from the scout...")
        raw_probs = self.proxy_model.predict_proba(X_all)
        del X_all
        gc.collect()

        logging.info(
            "  -> Amplifying whispers into 'clear calls' with the multi-class megaphone..."
        )
        calibrated_probs = np.zeros_like(raw_probs)
        for i in range(raw_probs.shape[1]):
            if i in self.calibrators:
                calibrated_probs[:, i] = self.calibrators[i].predict(raw_probs[:, i])
            else:
                logging.warning(
                    f"Calibrator for class {i} not found. Using raw probabilities."
                )
                calibrated_probs[:, i] = raw_probs[:, i]

        prob_sum = np.sum(calibrated_probs, axis=1, keepdims=True)
        prob_sum[prob_sum == 0] = 1.0
        self.final_probs = calibrated_probs / prob_sum

        self.P_sl = self.final_probs[:, 0]
        self.P_to = self.final_probs[:, 1]
        self.P_pt = self.final_probs[:, 2]

        logging.info(
            f"  -> First 5 final probabilities (SL, TO, PT): \n{self.final_probs[:5]}"
        )
        logging.info("Initialization complete. Ready for optimization.")

    def __call__(self, trial: optuna.Trial) -> float:
        """
        Calculates the mean expected value FOR TRADES THAT ARE ACTUALLY TAKEN.
        """
        try:
            payoff_ratio = trial.suggest_float(
                "payoff_ratio", *self.config.payoff_ratio_range
            )

            # --- ★★★ LOGIC FIX: Introduce trade decision making ★★★ ---
            R_pt = payoff_ratio
            R_sl = -1.0
            R_to = 0.0

            # 1. Calculate expected return for ALL potential trades
            all_expected_returns = (
                (self.P_pt * R_pt) + (self.P_sl * R_sl) + (self.P_to * R_to)
            )

            # 2. DECISION: Only consider trades where the expected value is positive
            profitable_trades_mask = all_expected_returns > 0

            # If no trades are deemed profitable for this payoff_ratio, return a very low value.
            if not np.any(profitable_trades_mask):
                return -1.0

            # 3. OBJECTIVE: Maximize the mean expected value of ONLY the trades taken
            mean_of_profitable_trades = np.mean(
                all_expected_returns[profitable_trades_mask]
            )

            return mean_of_profitable_trades
            # -----------------------------------------------------------

        except Exception as e:
            logging.error(f"  -> Trial {trial.number} failed: {e}", exc_info=False)
            return -999.0

    def _load_features_into_numpy(
        self, data_dir: Path, features: List[str]
    ) -> np.ndarray:
        """Loads only the feature data from the partitioned directory into a NumPy array."""
        partitions = sorted(data_dir.glob("year=*/month=*/day=*/*.parquet"))
        if not partitions:
            raise FileNotFoundError(f"No data partitions found in {data_dir}")

        all_dfs = [
            pl.read_parquet(p, columns=features)
            for p in tqdm(partitions, desc="Loading Feature Data")
        ]
        full_df = pl.concat(all_dfs, how="vertical")

        current_cols = set(full_df.columns)
        missing_cols = [f for f in features if f not in current_cols]
        if missing_cols:
            full_df = full_df.with_columns(
                [pl.lit(None, dtype=pl.Float64).alias(col) for col in missing_cols]
            )

        X = (
            full_df.select(features)
            .cast(pl.Float64, strict=False)
            .fill_null(0)
            .to_numpy()
        )

        del full_df, all_dfs
        gc.collect()
        return X


def main():
    logging.info(
        "### Phase 2: Bayesian Optimization based on Expected Value (with decision logic) ###"
    )
    config = OptimizerConfig()
    sampler = TPESampler(
        seed=42, n_startup_trials=config.n_startup_trials, multivariate=True
    )
    study = optuna.create_study(
        direction="maximize",
        sampler=sampler,
        study_name="project_forge_optimizer_v4_ev_decision",
    )
    try:
        objective = Objective(config)
        logging.info(f"Starting optimization for {config.n_trials} trials...")
        study.optimize(objective, n_trials=config.n_trials, show_progress_bar=True)

        logging.info("Optimization finished.")
        best = study.best_trial
        logging.info(f"  -> Number of finished trials: {len(study.trials)}")
        logging.info(f"  -> Best trial (Trial {best.number}):")
        logging.info(f"    -> Value (Mean Profitable Expected Value): {best.value:.6f}")
        logging.info("    -> Params: ")
        for key, value in best.params.items():
            logging.info(f"      - {key}: {value:.4f}")

        with open(OUTPUT_PARAMS_PATH, "w") as f:
            json.dump(best.params, f, indent=4)

        logging.info("\n" + "=" * 60)
        logging.info("### Optimization COMPLETED! ###")
        logging.info(
            f"The 'Ultimate Instruction Sheet' is ready at: {OUTPUT_PARAMS_PATH}"
        )
        logging.info("We are now ready to build the final AI in Phase 3.")
        logging.info("=" * 60)

    except Exception as e:
        logging.error(f"A critical error occurred: {e}", exc_info=True)


if __name__ == "__main__":
    main()
