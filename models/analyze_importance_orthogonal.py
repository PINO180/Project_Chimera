# /workspace/models/analyze_importance_orthogonal.py
# 直交分割版（selected_features_orthogonal_v5）のGain分析スクリプト
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
    S7_M1_MODEL_LONG_PKL,
    S7_M1_MODEL_SHORT_PKL,
    S7_M2_MODEL_LONG_PKL,
    S7_M2_MODEL_SHORT_PKL,
    S3_SELECTED_FEATURES_ORTHOGONAL_DIR,
)

S3_SELECTED_FEATURES_ORTHOGONAL_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


def load_orthogonal_features(filename: str) -> list[str]:
    """直交分割版の特徴量リストを読み込む（Cx_purified.pyの_load_features()と同一ロジック）"""
    filepath = S3_SELECTED_FEATURES_ORTHOGONAL_DIR / filename
    if not filepath.exists():
        logging.error(f"Feature list not found: {filepath}")
        return []

    exclude_exact = {
        "timestamp", "t1", "label", "label_long", "label_short",
        "uniqueness", "uniqueness_long", "uniqueness_short",
        "payoff_ratio", "payoff_ratio_long", "payoff_ratio_short",
        "pt_multiplier", "sl_multiplier", "direction", "exit_type",
        "first_ex_reason_int", "atr_value", "calculated_body_ratio",
        "fallback_vol", "open", "high", "low", "close",
        "meta_label", "m1_pred_proba", "is_trigger",
    }

    with open(filepath, "r") as f:
        raw_features = [line.strip() for line in f if line.strip()]

    # [FIX] "timeframe"を除外する。
    # かつてMixedモード（M1/M3/M5/M15複合）時代にAIへのコンテキストタグとして使っていたが、
    # 現在はM3専用スナイパーに純化済みのため分散ゼロの無意味な特徴量。
    # ファイルに混入するとmain.pyの推論時にクラッシュの原因になる。
    features = []
    for col in raw_features:
        if col in exclude_exact or col == "timeframe":
            continue
        if col.startswith("is_trigger_on"):
            continue
        features.append(col)

    return features


def analyze_and_export(
    model_path: Path, model_name: str, feature_names: list[str], output_filename: str
):
    print(f"\n{'=' * 40}\n {model_name} (Orthogonal Phase) \n{'=' * 40}")
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

    # --- ログ出力 ---
    print(f"✅ Active features (Gain > 0): {len(df_active)} / {len(feature_names)}")
    print(f"{'Rank':<5} {'Feature Name':<50} {'Gain':>15}")
    print("-" * 75)
    for i, row in enumerate(df_active.head(2000).itertuples()):
        print(f"{i + 1:<5} {str(row.feature):<50} {row.gain:>15.2f}")

    # [FIX] 4ファイルへの上書き保存を廃止。ログ出力のみ。
    # Gain=0の特徴量が見つかった場合は別途手動で4ファイルを修正すること。
    gain_zero = df[df["gain"] == 0]
    if len(gain_zero) > 0:
        print(f"⚠️  Gain=0 の特徴量: {len(gain_zero)}件 → 4ファイルからの手動削除を検討してください")
        for row in gain_zero.itertuples():
            print(f"   Gain=0: {row.feature}")


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
        S3_SELECTED_FEATURES_ORTHOGONAL_DIR / "console_log_orthogonal_v5.txt"
    )

    configs = [
        (S7_M1_MODEL_LONG_PKL,  "M1 (LONG)",  "m1_long_features.txt"),
        (S7_M1_MODEL_SHORT_PKL, "M1 (SHORT)", "m1_short_features.txt"),
        (S7_M2_MODEL_LONG_PKL,  "M2 (LONG)",  "m2_long_features.txt"),
        (S7_M2_MODEL_SHORT_PKL, "M2 (SHORT)", "m2_short_features.txt"),
    ]

    for model_path, label, fname in configs:
        features = load_orthogonal_features(fname)
        # M2モデルはm1_pred_probaが末尾に追加されている（Cx版と同一処理）
        if "M2" in label and "m1_pred_proba" not in features:
            features.append("m1_pred_proba")
        if features:
            analyze_and_export(model_path, label, features, fname)


if __name__ == "__main__":
    main()
