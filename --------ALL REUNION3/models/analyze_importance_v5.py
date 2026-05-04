# /workspace/models/analyze_importance_v5.py
import sys
from pathlib import Path
import joblib
import pandas as pd
import lightgbm as lgb
import logging

# --- パス設定 ---
project_root = Path(__file__).resolve().parents[1]
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

from blueprint import (
    S7_MODELS,
    S7_M1_MODEL_LONG_PKL,  #
    S7_M1_MODEL_SHORT_PKL,  #
    S7_M2_MODEL_LONG_PKL,  #
    S7_M2_MODEL_SHORT_PKL,  #
    # S3_FEATURES_FOR_TRAINING,  # ← 削除
    S3_FEATURES_FOR_TRAINING_V5,  # ★ 追加
    S3_SELECTED_FEATURES_DIR,  #
)

S3_SELECTED_FEATURES_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


def get_base_features() -> list[str]:
    if not S3_FEATURES_FOR_TRAINING_V5.exists():
        logging.error(f"Feature list not found: {S3_FEATURES_FOR_TRAINING_V5}")
        return []
    with open(S3_FEATURES_FOR_TRAINING_V5, "r") as f:
        raw_features = [line.strip() for line in f if line.strip()]

    exclude_exact = {
        "timestamp",
        # "timeframe",  # ★削除: 特徴量として使うため除外してはいけない
        "t1",
        "label",
        "label_long",
        "label_short",
        "uniqueness",
        "uniqueness_long",
        "uniqueness_short",
        "payoff_ratio",
        "payoff_ratio_long",
        "payoff_ratio_short",
        "pt_multiplier",
        "sl_multiplier",
        "direction",
        "exit_type",
        "first_ex_reason_int",
        "atr_value",
        "calculated_body_ratio",
        "fallback_vol",
        "open",
        "high",
        "low",
        "close",
        "meta_label",
        "m1_pred_proba",
        "is_trigger",
        "concurrency_long",
        "concurrency_short",
    }

    # ★追加: Script A/C と完全に同じロジックで特徴量リストを構築する
    features = ["timeframe"]
    for col in raw_features:
        if (
            col in exclude_exact
            or col == "timeframe"
            or col.startswith("is_trigger_on")
        ):
            continue
        features.append(col)

    return features


def analyze_and_export(
    model_path: Path, model_name: str, feature_names: list[str], output_filename: str
):
    print(f"\n{'=' * 40}\n {model_name} \n{'=' * 40}")
    if not model_path.exists():
        logging.error(f"Model not found: {model_path}")
        return

    model = joblib.load(model_path)
    booster = model if isinstance(model, lgb.Booster) else model.booster_
    importance_gain = booster.feature_importance(importance_type="gain")

    if len(importance_gain) != len(feature_names):
        logging.warning("Feature counts MISMATCH! Using Column_X names.")
        final_names = [f"Column_{i}" for i in range(len(importance_gain))]
    else:
        final_names = feature_names

    df = pd.DataFrame({"feature": final_names, "gain": importance_gain})
    df_active = df[df["gain"] > 0].sort_values(by="gain", ascending=False)

    # --- ログ出力の復元 ---
    print(f"✅ Active features (Gain > 0): {len(df_active)} / {len(feature_names)}")
    print(f"{'Rank':<5} {'Feature Name':<50} {'Gain':>15}")
    print("-" * 75)
    for i, row in enumerate(df_active.head(2000).itertuples()):
        print(f"{i + 1:<5} {str(row.feature):<50} {row.gain:>15.2f}")

    if len(df_active) > 2000:
        print(f"   ... and {len(df_active) - 2000} more features.")

    out_path = S3_SELECTED_FEATURES_DIR / output_filename
    with open(out_path, "w") as f:
        for feature in df_active["feature"]:
            f.write(f"{feature}\n")
    print(f"💾 Saved {len(df_active)} features to: {out_path.name}")


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

    sys.stdout = DualLogger(S3_SELECTED_FEATURES_DIR / "console_log_v5.txt")

    base_features = get_base_features()
    if not base_features:
        return

    analyze_and_export(
        S7_M1_MODEL_LONG_PKL, "M1 (LONG)", base_features, "m1_long_features.txt"
    )
    analyze_and_export(
        S7_M1_MODEL_SHORT_PKL, "M1 (SHORT)", base_features, "m1_short_features.txt"
    )

    m2_features = ["m1_pred_proba"] + base_features
    analyze_and_export(
        S7_M2_MODEL_LONG_PKL, "M2 (LONG)", m2_features, "m2_long_features.txt"
    )
    analyze_and_export(
        S7_M2_MODEL_SHORT_PKL, "M2 (SHORT)", m2_features, "m2_short_features.txt"
    )


if __name__ == "__main__":
    main()
