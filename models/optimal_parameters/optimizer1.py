# /workspace/models/optimal_parameters/optimizer.py

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

# --- ★★★ Path Definitions (FINAL) ★★★ ---
OPTIMIZATION_DIR = (
    project_root / "data/XAUUSD/stratum_7_models/1A_2B/optimal_parameters"
)
# Input 1: The lightweight data used for training
DATA_SOURCE_PATH = OPTIMIZATION_DIR / "temp_weighted_subset_partitioned"
# Input 2: The 'scout' model
PROXY_MODEL_PATH = OPTIMIZATION_DIR / "proxy_model.pkl"
# Input 3: The 'megaphone' calibrator
PROXY_CALIBRATOR_PATH = OPTIMIZATION_DIR / "proxy_model_calibrator.pkl"
# Input 4: The feature list
FEATURE_LIST_PATH = OPTIMIZATION_DIR / "proxy_feature_list.txt"
# Output: The final instruction sheet
OUTPUT_PARAMS_PATH = OPTIMIZATION_DIR / "optimal_parameters.json"


@dataclass
class OptimizerConfig:
    n_trials: int = 100
    n_startup_trials: int = 20
    # ★★★ REMOVED: lookahead_bars_range is no longer needed ★★★
    payoff_ratio_range: Tuple[float, float] = (1.5, 5.0)
    risk_free_rate: float = 0.0
    kelly_fraction: float = 0.5


class Objective:
    def __init__(self, config: OptimizerConfig):
        self.config = config
        logging.info("Initializing Objective function for high-speed optimization...")

        # --- 1. Load all necessary components ---
        logging.info("  -> Loading 'scout' (proxy_model.pkl)...")
        self.proxy_model: lgb.LGBMClassifier = joblib.load(PROXY_MODEL_PATH)

        logging.info("  -> Loading 'megaphone' (proxy_model_calibrator.pkl)...")
        self.calibrator = joblib.load(PROXY_CALIBRATOR_PATH)

        with open(FEATURE_LIST_PATH, "r") as f:
            self.features = [line.strip() for line in f if line.strip()]

        # --- 2. Load pre-labeled data and generate probabilities ONCE ---
        logging.info("  -> Loading pre-labeled data into memory...")
        X_all, self.labels_all = self._load_prelabeled_data_into_numpy(
            DATA_SOURCE_PATH, self.features
        )

        logging.info("  -> Generating raw 'whispers' from the scout...")
        raw_probs = self.proxy_model.predict_proba(X_all)[:, 1]

        logging.info(
            "  -> Amplifying whispers into 'clear calls' with the megaphone..."
        )
        self.calibrated_probs = self.calibrator.predict(raw_probs)

        logging.info(
            f"  -> First 5 calibrated probabilities: {self.calibrated_probs[:5]}"
        )
        logging.info("Initialization complete. Ready for optimization.")
        del X_all
        gc.collect()

    def __call__(self, trial: optuna.Trial) -> float:
        try:
            # ★★★ CRITICAL CHANGE: We ONLY optimize payoff_ratio now ★★★
            payoff_ratio = trial.suggest_float(
                "payoff_ratio", *self.config.payoff_ratio_range
            )

            # We use the pre-calculated, calibrated probabilities and fixed labels.
            # No more dynamic labeling. This is why it's extremely fast.
            simulated_returns = self._simulate_strategy_numpy(
                self.calibrated_probs, self.labels_all, payoff_ratio
            )

            if simulated_returns is None or len(simulated_returns) < 20:
                return -1.0

            sortino_ratio = self._calculate_sortino_ratio(simulated_returns)
            return sortino_ratio if not np.isnan(sortino_ratio) else -1.0

        except Exception as e:
            logging.error(f"  -> Trial {trial.number} failed: {e}", exc_info=False)
            return -999.0

    def _load_prelabeled_data_into_numpy(
        self, data_dir: Path, features: List[str]
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Loads all data from the partitioned directory into NumPy arrays."""
        partitions = sorted(data_dir.glob("year=*/month=*/day=*/*.parquet"))
        if not partitions:
            raise FileNotFoundError(f"No data partitions found in {data_dir}")

        all_dfs = [pl.read_parquet(p) for p in tqdm(partitions, desc="Loading Data")]
        full_df = pl.concat(all_dfs, how="vertical")

        # Ensure all feature columns exist, same as in training script
        current_cols = set(full_df.columns)
        missing_cols = [f for f in features if f not in current_cols]
        if missing_cols:
            full_df = full_df.with_columns(
                [pl.lit(None, dtype=pl.Float64).alias(col) for col in missing_cols]
            )

        # Fill nulls and convert to NumPy
        X = full_df.select(features).fill_null(0).to_numpy()
        labels = full_df["label"].to_numpy()

        del full_df, all_dfs
        gc.collect()
        return X, labels

    def _simulate_strategy_numpy(
        self, probabilities: np.ndarray, labels: np.ndarray, payoff_ratio: float
    ) -> np.ndarray | None:
        p, b, q = probabilities, payoff_ratio, 1 - probabilities
        # Prevent division by zero if b is somehow zero
        b = np.maximum(b, 1e-12)
        f_star = (p * b - q) / b
        bet_size = np.clip(f_star * self.config.kelly_fraction, 0, 1)

        # We only care about trades where the label is win (1) or loss (-1)
        entry_signal = (bet_size > 1e-5) & (labels != 0)

        trade_returns = np.select(
            [(entry_signal) & (labels == 1), (entry_signal) & (labels == -1)],
            [bet_size * payoff_ratio, bet_size * -1.0],
            default=0.0,
        )
        # Return only the returns from actual trades that were taken
        return trade_returns[trade_returns != 0]

    def _calculate_sortino_ratio(self, returns: np.ndarray) -> float:
        """Calculates the annualized Sortino ratio correctly from raw trade returns."""
        if len(returns) < 20:
            return -1.0

        mean_return_per_trade = np.mean(returns)
        target_return = self.config.risk_free_rate
        downside_returns = returns[returns < target_return]

        if len(downside_returns) < 2:
            return 100.0 if mean_return_per_trade > 0 else 0.0

        downside_deviation = np.sqrt(np.mean(np.square(downside_returns)))

        if downside_deviation == 0:
            return 100.0 if mean_return_per_trade > 0 else 0.0

        sortino_ratio = (mean_return_per_trade - target_return) / downside_deviation
        annualization_factor = np.sqrt(252)
        return sortino_ratio * annualization_factor


def main():
    logging.info(
        "### Phase 2: Bayesian Optimization for Optimal Parameters (Calibrated Edition) ###"
    )
    config = OptimizerConfig()
    sampler = TPESampler(
        seed=42, n_startup_trials=config.n_startup_trials, multivariate=True
    )
    study = optuna.create_study(
        direction="maximize", sampler=sampler, study_name="project_forge_optimizer_v2"
    )
    try:
        objective = Objective(config)
        logging.info(f"Starting optimization for {config.n_trials} trials...")
        study.optimize(objective, n_trials=config.n_trials, show_progress_bar=True)
        logging.info("Optimization finished.")
        best = study.best_trial
        logging.info(f"  -> Number of finished trials: {len(study.trials)}")
        logging.info(f"  -> Best trial (Trial {best.number}):")
        logging.info(f"    -> Value (Sortino Ratio): {best.value:.4f}")
        logging.info("    -> Params: ")
        for key, value in best.params.items():
            logging.info(f"      - {key}: {value}")
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
