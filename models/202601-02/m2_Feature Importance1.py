import joblib
import pandas as pd
from pathlib import Path
import sys

# --- 設定 ---
MODEL_PATH = Path("/workspace/data/XAUUSD/stratum_7_models/1A_2B/m2_model_v2.pkl")
FEATURE_LIST_PATH = Path(
    "/workspace/data/XAUUSD/stratum_3_artifacts/1A_2B/train_12m_val_6m/final_feature_set_v3.txt"
)


def show_all_feature_importance():
    if not MODEL_PATH.exists():
        print(f"❌ モデルが見つかりません: {MODEL_PATH}")
        return

    print(f"🔄 Loading model from {MODEL_PATH}...")
    model = joblib.load(MODEL_PATH)

    with open(FEATURE_LIST_PATH, "r") as f:
        base_features = [line.strip() for line in f if line.strip()]

    # M2特徴量定義
    m2_specific_features = [
        "m1_pred_proba",
        "hmm_prob_0",
        "hmm_prob_1",
        "atr_ratio",
        "trend_bias_25",
        "e1a_statistical_kurtosis_50",
        "e1c_adx_21",
        "e2a_mfdfa_hurst_mean_250",
        "e2a_kolmogorov_complexity_60",
    ]
    full_feature_names = base_features + m2_specific_features

    importance = model.feature_importance(importance_type="gain")

    # 長さ調整
    min_len = min(len(importance), len(full_feature_names))
    full_feature_names = full_feature_names[:min_len]
    importance = importance[:min_len]

    # DataFrame作成
    df_imp = pd.DataFrame(
        {"feature_name": full_feature_names, "importance": importance}
    )

    # ソートしてランク付け
    df_imp = df_imp.sort_values("importance", ascending=False).reset_index(drop=True)
    df_imp["rank"] = df_imp.index + 1

    # --- 全件表示設定 ---
    pd.set_option("display.max_rows", None)  # 行数の制限を解除
    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 1000)

    print("\n" + "=" * 80)
    print(f"       🏆 ALL FEATURES RANKING ({len(df_imp)} Features)       ")
    print("=" * 80)
    print(df_imp.to_string(index=False))

    # --- M2特徴量の位置確認 ---
    print("\n" + "=" * 80)
    print("       🤖 M2 CONTEXT FEATURES POSITION       ")
    print("=" * 80)
    df_m2 = df_imp[df_imp["feature_name"].isin(m2_specific_features)]
    print(df_m2.to_string(index=False))


if __name__ == "__main__":
    show_all_feature_importance()
