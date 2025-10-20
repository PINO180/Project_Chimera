# /workspace/models/optimal_parameters/optimizer_diagnostic_runner.py
# This is the FINAL, corrected version of the diagnostic script.

import sys
from pathlib import Path
import logging
import warnings
import numpy as np

# Import the corrected Objective class and its helper functions
from optimizer import Objective, OptimizerConfig, _fast_label_numba_loop

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
warnings.filterwarnings("ignore", category=UserWarning)


def run_diagnostic():
    logging.info("### Starting Emergency Triage Diagnostic (FINAL v3) ###")
    config = OptimizerConfig()

    try:
        objective = Objective(config)
    except Exception as e:
        logging.error(f"CRITICAL ERROR during initialization: {e}", exc_info=True)
        return

    # Define the parameters for our single test run
    test_params = {"payoff_ratio": 2.35, "lookahead_bars": 69}

    # --- Test Case 1: Run with the REAL proxy model ---
    logging.info("\n" + "=" * 60)
    logging.info("### Test Case 1: Using the REAL Proxy Model ###")
    logging.info("=" * 60)

    try:
        # --- ★★★ FIX #1: DYNAMIC RE-LABELING IS NOW APPLIED ★★★ ---
        labels_real = _fast_label_numba_loop(
            objective.close_all,
            objective.high_all,
            objective.low_all,
            objective.atr_all,
            test_params["payoff_ratio"],
            test_params["lookahead_bars"],
        )

        probs_real = objective.proxy_model.predict_proba(objective.X_all)[:, 1]

        logging.info("--- Intermediate values for REAL model ---")
        logging.info(f"  -> First 5 Probabilities: {probs_real[:5]}")

        returns_real = objective._simulate_strategy_numpy(
            probs_real, labels_real, test_params["payoff_ratio"]
        )

        # --- ★★★ FIX #2: CORRECT SORTINO CALCULATION IS NOW USED ★★★ ---
        score_real = (
            objective._calculate_sortino_ratio(returns_real)
            if returns_real is not None and len(returns_real) > 20
            else -1.0
        )

        logging.info(
            f"--- Result for Test Case 1 (REAL Model): Sortino Ratio = {score_real:.4f} ---"
        )

    except Exception as e:
        logging.error(f"An error occurred during Test Case 1: {e}", exc_info=True)
        score_real = -999.0

    # --- Test Case 2: Run with a "Perfect" AI ---
    logging.info("\n" + "=" * 60)
    logging.info(
        "### Test Case 2: Using a FAKE 'Perfect' Model ###"
    )  # Title changed for clarity
    logging.info("=" * 60)

    try:
        # We use the same dynamically generated labels
        labels_perfect = labels_real

        # --- ★★★ FIX #4: CREATE A TRULY 'PERFECT' AI ★★★ ---
        # Instead of being blindly optimistic, this AI knows the future.
        # It assigns a high probability (0.9) ONLY when the label is a win (1),
        # and a low probability (0.1) when the label is a loss (-1) or neutral (0).
        # This correctly simulates a model with high predictive accuracy.
        probs_perfect = np.select(
            [labels_perfect == 1, labels_perfect == -1],
            [0.9, 0.1],  # p=0.9 for wins, p=0.1 for losses
            default=0.5,  # p=0.5 for neutral labels
        )

        logging.info("--- Intermediate values for 'PERFECT' model ---")
        logging.info(f"  -> First 5 Probabilities: {probs_perfect[:5]}")
        # (This will now vary depending on the first 5 labels)

        returns_perfect = objective._simulate_strategy_numpy(
            probs_perfect, labels_perfect, test_params["payoff_ratio"]
        )

        # Use the corrected Sortino calculation here as well
        score_perfect = (
            objective._calculate_sortino_ratio(returns_perfect)
            if returns_perfect is not None and len(returns_perfect) > 20
            else -1.0
        )

        logging.info(
            f"--- Result for Test Case 2 ('Perfect' Model): Sortino Ratio = {score_perfect:.4f} ---"
        )

    except Exception as e:
        logging.error(f"An error occurred during Test Case 2: {e}", exc_info=True)
        score_perfect = -999.0

    logging.info("\n" + "=" * 60)
    logging.info("### Diagnostic Complete ###")
    if score_perfect > 0:
        logging.info("✅ SUCCESS: Simulation logic is working correctly!")
        if score_real > 0:
            logging.info(
                "✅ EXCELLENT: The REAL proxy model is also generating positive returns!"
            )
        else:
            logging.info(
                "⚠️ ATTENTION: The proxy model's predictions are not yet profitable, but the system works."
            )
    else:
        logging.info("❌ FAILURE: Simulation logic still has a fundamental bug.")
        logging.info(
            "Even with a perfect signal, no positive Sortino Ratio was achieved."
        )


if __name__ == "__main__":
    run_diagnostic()
