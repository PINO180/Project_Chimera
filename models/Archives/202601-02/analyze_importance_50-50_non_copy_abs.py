import sys
from pathlib import Path
import joblib
import pandas as pd
import lightgbm as lgb
import json
import logging

# --- パス設定 ---
PROJECT_ROOT = Path("/workspace")
SYMBOL = "XAUUSD"
FEATURE_SET_ID = "1A_2B"
RUN_ID = "train_12m_val_6m"

# モデルファイルのパス
M1_MODEL_PATH = (
    PROJECT_ROOT
    / "data"
    / SYMBOL
    / "stratum_7_models"
    / FEATURE_SET_ID
    / "m1_model_v2.pkl"
)
M2_MODEL_PATH = (
    PROJECT_ROOT
    / "data"
    / SYMBOL
    / "stratum_7_models"
    / FEATURE_SET_ID
    / "m2_model_v2.pkl"
)

# 特徴量リストのパス
M1_FEATURE_LIST_PATH = (
    PROJECT_ROOT
    / "data"
    / SYMBOL
    / "stratum_3_artifacts"
    / FEATURE_SET_ID
    / RUN_ID
    / "final_feature_set_v3.txt"
)
M2_FEATURE_LIST_PATH = PROJECT_ROOT / "models" / "TOP_50_FEATURES.json"


def load_feature_names_txt(path: Path):
    if not path.exists():
        print(f"⚠️ Feature list not found: {path}")
        return None
    with open(path, "r") as f:
        return [line.strip() for line in f if line.strip()]


def load_feature_names_json(path: Path):
    if not path.exists():
        print(f"⚠️ Feature list not found: {path}")
        return None
    with open(path, "r") as f:
        return json.load(f)


def analyze_model_full(
    model_path: Path, feature_names_path: Path, model_name: str, file_type="txt"
):
    print(f"\n{'=' * 30} {model_name} {'=' * 30}")

    # 1. モデル読み込み
    if not model_path.exists():
        print(f"❌ Model not found: {model_path}")
        return

    try:
        model = joblib.load(model_path)
        if isinstance(model, lgb.Booster):
            booster = model
        elif hasattr(model, "booster_"):
            booster = model.booster_
        else:
            print(f"❌ Unsupported model type: {type(model)}")
            return
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return

    # 2. 特徴量リスト読み込み & 補正
    print(f"📂 Loading feature names from: {feature_names_path}")

    if file_type == "json":
        feature_names = load_feature_names_json(feature_names_path)
        if feature_names and "M2" in model_name:
            print("ℹ️  Injecting 'm1_pred_proba' for M2")
            feature_names = ["m1_pred_proba"] + feature_names

    elif file_type == "txt_m2":  # ★パターン②M2用の新モード
        feature_names = load_feature_names_txt(feature_names_path)
        if feature_names:
            print("ℹ️  Injecting 'm1_pred_proba' for M2 (Full Mode)")
            feature_names = ["m1_pred_proba"] + feature_names

    else:  # 通常の "txt"
        feature_names = load_feature_names_txt(feature_names_path)

    # 3. 重要度取得
    importance_gain = booster.feature_importance(importance_type="gain")
    importance_split = booster.feature_importance(importance_type="split")

    n_features_model = len(importance_gain)
    n_features_list = len(feature_names)

    print(f"📊 Model Features: {n_features_model}")
    print(f"📝 List Features : {n_features_list}")

    final_names = []

    if n_features_model == n_features_list:
        print("✅ Feature counts match! Mapping names correctly.")
        final_names = feature_names
    else:
        print("⚠️ Feature counts MISMATCH!")
        print(f"   Using model's internal names (Column_X) for safety.")
        final_names = [f"Column_{i}" for i in range(n_features_model)]

    # 4. DataFrame作成
    df = pd.DataFrame(
        {"feature": final_names, "gain": importance_gain, "split": importance_split}
    )

    # 5. フィルタリング (Gain > 0 のみ)
    df_active = df[df["gain"] > 0].copy()

    # ソート
    df_sorted = df_active.sort_values(by="gain", ascending=False)

    # 表示
    print(f"\n✅ Showing ALL {len(df_sorted)} features with Gain > 0:\n")
    print(f"{'Rank':<5} {'Feature Name':<60} {'Gain':>15} {'Split':>8}")
    print("-" * 90)

    for i, row in enumerate(df_sorted.itertuples()):
        print(f"{i + 1:<5} {str(row.feature):<60} {row.gain:>15.2f} {row.split:>8}")

    if len(df_sorted) < len(df):
        print(f"\nℹ️  (Omitted {len(df) - len(df_sorted)} features with 0 gain)")


if __name__ == "__main__":
    # ---------------------------------------------------------
    # 【実験パターン①】 M1もM2も Top 50 (JSON) を使用
    # ---------------------------------------------------------

    # 1. M1 分析 (Top 50 JSONを使用)
    analyze_model_full(
        model_path=M1_MODEL_PATH,
        feature_names_path=M2_FEATURE_LIST_PATH,  # ★ここをJSONのパス(M2_FEATURE_LIST_PATH)に変更
        model_name="M1 Model (Top 50 Only)",
        file_type="json",  # ★JSONとして読み込む (ただしM1は m1_pred_proba を含まないので注意が必要)
    )
    # ※ 注: analyze_model_full内の "json" モードは Script C のロジック(先頭にm1_pred_proba追加)が入っています。
    # M1は m1_pred_proba を使わない純粋な Top 50 なので、
    # もしM1のfeature数がJSONの数と一致しない(1つズレる)場合は、
    # file_type="json_raw" のような処理が必要ですが、
    # 下記の「修正版関数」を使うことでより確実にマッピングできます。

    # 2. M2 分析 (Top 50 JSON + m1_pred_proba)
    analyze_model_full(
        model_path=M2_MODEL_PATH,
        feature_names_path=M2_FEATURE_LIST_PATH,
        model_name="M2 Model (Top 50 + Score)",
        file_type="json",
    )
