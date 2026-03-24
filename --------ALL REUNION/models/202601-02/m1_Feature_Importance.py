import sys
import joblib
import pandas as pd
from pathlib import Path

# --- プロジェクトルートをパスに追加 (blueprint読み込み用) ---
project_root = Path(__file__).resolve().parents[1]
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

import blueprint as config


def show_m1_feature_importance():
    # =================================================================
    # 設定: M1モデル (Full Feature Set) の分析
    # =================================================================
    model_path = config.S7_M1_MODEL_PKL
    # M1は通常 S3_FEATURES_FOR_TRAINING (final_feature_set.txt) を使用
    feature_list_path = config.S3_FEATURES_FOR_TRAINING

    print("=" * 80)
    print("📊 M1 Feature Importance Analysis (The Primary Judge)")
    print("=" * 80)

    # 1. ファイル存在チェック
    if not model_path.exists():
        print(f"❌ M1モデルが見つかりません: {model_path}")
        return

    if not feature_list_path.exists():
        print(f"❌ 特徴量リストが見つかりません: {feature_list_path}")
        return

    # 2. モデルと特徴量リストのロード
    print(f"🔄 Loading M1 Model: {model_path.name} ...")
    try:
        model = joblib.load(model_path)
    except Exception as e:
        print(f"❌ モデルロードエラー: {e}")
        return

    print(f"🔄 Loading Feature List: {feature_list_path.name} ...")
    with open(feature_list_path, "r", encoding="utf-8") as f:
        # 空行を除いてリスト化
        feature_names = [line.strip() for line in f if line.strip()]

    print(f"   -> Total Features in M1: {len(feature_names)}")

    # 3. 重要度 (Gain) の取得
    try:
        if hasattr(model, "booster_"):
            # sklearn API (LGBMClassifier)
            importance = model.booster_.feature_importance(importance_type="gain")
        else:
            # Native Booster API
            importance = model.feature_importance(importance_type="gain")
    except Exception as e:
        print(f"❌ 重要度の取得に失敗しました: {e}")
        return

    # 4. 整合性チェック
    # LightGBMは特徴量名を保存しないため、長さが一致することが絶対条件
    if len(importance) != len(feature_names):
        print(f"⚠️ CRITICAL WARNING: Feature count mismatch!")
        print(f"   Model expects: {len(importance)}")
        print(f"   List contains: {len(feature_names)}")
        print(
            "   -> ランキングがズレる可能性があります。学習時と同じリストを使用してください。"
        )
        # 一致する範囲で続行（または中断）
        min_len = min(len(importance), len(feature_names))
        feature_names = feature_names[:min_len]
        importance = importance[:min_len]

    # 5. DataFrame作成
    df = pd.DataFrame({"feature_name": feature_names, "importance": importance})

    # 6. ランキング作成
    df = df.sort_values("importance", ascending=False).reset_index(drop=True)
    df["rank"] = df.index + 1

    # 7. 表示設定
    pd.set_option("display.max_rows", None)
    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 1000)
    pd.options.display.float_format = "{:.4f}".format

    # 8. 結果表示
    print("\n" + "=" * 80)
    print(f"🏆 M1 Feature Ranking (All {len(df)} Features)")
    print("=" * 80)
    print(df.to_string(index=False))

    # 9. 下位の特徴量を確認 (リストラ候補)
    print("\n" + "=" * 80)
    print("🗑️ Bottom 20 Features (Candidates for Removal)")
    print("=" * 80)
    print(df.tail(20).to_string(index=False))

    print("\n" + "=" * 80)


if __name__ == "__main__":
    show_m1_feature_importance()
