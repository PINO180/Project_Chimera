# /workspace/models/analyze_importance_purified.py
import sys
from pathlib import Path
import joblib
import pandas as pd
import lightgbm as lgb
import logging

# --- パス設定 ---
PROJECT_ROOT = Path("/workspace")
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from blueprint import (
    S7_MODELS,
    S7_M1_MODEL_LONG_PKL,
    S7_M1_MODEL_SHORT_PKL,
    S7_M2_MODEL_LONG_PKL,
    S7_M2_MODEL_SHORT_PKL,
    S3_SELECTED_FEATURES_DIR,
    S3_SELECTED_FEATURES_PURIFIED_DIR,
)

S3_SELECTED_FEATURES_PURIFIED_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


def load_purified_features(filename: str) -> list[str]:
    filepath = S3_SELECTED_FEATURES_DIR / filename  #
    if not filepath.exists():
        logging.error(f"Feature list not found: {filepath}")
        return []
    with open(filepath, "r") as f:
        return [line.strip() for line in f if line.strip()]


def analyze_and_export(
    model_path: Path, model_name: str, feature_names: list[str], output_filename: str
):
    print(f"\n{'=' * 40}\n {model_name} (Purification Phase) \n{'=' * 40}")
    if not model_path.exists():
        logging.error(f"Model not found: {model_path}")
        return

    model = joblib.load(model_path)
    booster = model if isinstance(model, lgb.Booster) else model.booster_
    importance_gain = booster.feature_importance(importance_type="gain")

    if len(importance_gain) != len(feature_names):
        logging.warning(
            f"Feature counts MISMATCH! Model: {len(importance_gain)}, List: {len(feature_names)}"
        )
        final_names = [f"Column_{i}" for i in range(len(importance_gain))]
    else:
        final_names = feature_names

    df = pd.DataFrame({"feature": final_names, "gain": importance_gain})
    df_active = df[df["gain"] > 0].sort_values(by="gain", ascending=False)

    # --- ログ出力の復元 ---
    print(f"✨ Purified features (Gain > 0): {len(df_active)} / {len(feature_names)}")
    print(f"{'Rank':<5} {'Feature Name':<50} {'Gain':>15}")
    print("-" * 500)
    for i, row in enumerate(df_active.head(500).itertuples()):
        print(f"{i + 1:<5} {str(row.feature):<50} {row.gain:>15.2f}")

    out_path = S3_SELECTED_FEATURES_PURIFIED_DIR / output_filename  #
    with open(out_path, "w") as f:
        for feature in df_active["feature"]:
            f.write(f"{feature}\n")
    print(f"💾 Saved purified list to: {out_path.parent.name}/{out_path.name}")


def main():
    class DualLogger:
        def __init__(self, filepath):
            self.terminal = sys.stdout
            self.log = open(filepath, "w", encoding="utf-8")

        def write(self, message):
            self.terminal.write(message)
            self.log.write(message)

        def flush(self):
            self.terminal.flush()
            self.log.flush()

    sys.stdout = DualLogger(
        S3_SELECTED_FEATURES_PURIFIED_DIR / "console_log_purified.txt"
    )

    configs = [
        (S7_M1_MODEL_LONG_PKL, "M1 (LONG)", "m1_long_features.txt"),
        (S7_M1_MODEL_SHORT_PKL, "M1 (SHORT)", "m1_short_features.txt"),
        (S7_M2_MODEL_LONG_PKL, "M2 (LONG)", "m2_long_features.txt"),
        (S7_M2_MODEL_SHORT_PKL, "M2 (SHORT)", "m2_short_features.txt"),
    ]

    for model_path, label, fname in configs:
        features = load_purified_features(fname)
        if features:
            analyze_and_export(model_path, label, features, fname)


if __name__ == "__main__":
    main()
