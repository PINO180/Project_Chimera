import sys
import json
import joblib
import pandas as pd
from pathlib import Path

# --- プロジェクトルートをパスに追加 (blueprint読み込み用) ---
project_root = Path(__file__).resolve().parents[1]
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

import blueprint as config


def show_m2_feature_importance():
    # =================================================================
    # 設定: B案改 (Ace Injection) 用 - blueprintからパスを取得
    # =================================================================
    model_path = config.S7_M2_MODEL_PKL
    top_50_path = project_root / "models" / "TOP_50_FEATURES.json"

    print("=" * 80)
    print("📊 M2 Feature Importance Analysis (Plan B Revised: Ace Injection)")
    print("=" * 80)

    # 1. ファイル存在チェック
    if not model_path.exists():
        print(f"❌ モデルが見つかりません: {model_path}")
        print("   -> Script C (M2 Training) が正常に完了しているか確認してください。")
        return

    if not top_50_path.exists():
        print(f"❌ 特徴量リストが見つかりません: {top_50_path}")
        print("   -> Script 1 (Create Top 50 List) を実行してください。")
        return

    # 2. モデルと特徴量リストのロード
    print(f"🔄 Loading M2 Model: {model_path.name} ...")
    try:
        model = joblib.load(model_path)
    except Exception as e:
        print(f"❌ モデルロードエラー: {e}")
        return

    print(f"🔄 Loading Top 50 List: {top_50_path.name} ...")
    with open(top_50_path, "r", encoding="utf-8") as f:
        top_50_features = json.load(f)

    # 3. M2の特徴量定義を再構築
    # Training Script (C) と完全に一致させる: ["m1_pred_proba"] + [Top 50 Features]
    m2_feature_names = ["m1_pred_proba"] + top_50_features

    print(
        f"   -> Total Features in M2: {len(m2_feature_names)} (1 Confidence + 50 Context)"
    )

    # 4. 重要度 (Gain) の取得 - オブジェクトタイプ判定付き
    try:
        if hasattr(model, "booster_"):
            # sklearn API (LGBMClassifier) の場合
            importance = model.booster_.feature_importance(importance_type="gain")
        else:
            # Native Booster API の場合
            importance = model.feature_importance(importance_type="gain")
    except Exception as e:
        print(f"❌ 重要度の取得に失敗しました: {e}")
        return

    # 5. DataFrame作成と整合性チェック
    min_len = min(len(importance), len(m2_feature_names))
    if len(importance) != len(m2_feature_names):
        print(
            f"⚠️ 警告: モデルの特徴量数({len(importance)}) と リスト数({len(m2_feature_names)}) が一致しません。"
        )
        print(
            "   -> 学習時とリスト定義が異なる可能性があります。一致する範囲で表示します。"
        )

    df = pd.DataFrame(
        {"feature_name": m2_feature_names[:min_len], "importance": importance[:min_len]}
    )

    # 6. ランキング作成
    df = df.sort_values("importance", ascending=False).reset_index(drop=True)
    df["rank"] = df.index + 1

    # 7. 表示設定
    pd.set_option("display.max_rows", None)
    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 1000)
    pd.options.display.float_format = "{:.4f}".format

    # 8. 結果表示: 全ランキング
    print("\n" + "=" * 80)
    print(f"🏆 M2 Feature Ranking (All {len(df)} Features)")
    print("=" * 80)
    print(df.to_string(index=False))

    # 9. 結果表示: m1_pred_proba の位置 (エースの確認)
    print("\n" + "=" * 80)
    print("🎯 Spotlight: m1_pred_proba (The Ace)")
    print("=" * 80)
    m1_row = df[df["feature_name"] == "m1_pred_proba"]
    if not m1_row.empty:
        print(m1_row.to_string(index=False))
    else:
        print(
            "⚠️ 'm1_pred_proba' がランキングに見つかりません。学習データに含まれていない可能性があります。"
        )

    print("\n" + "=" * 80)


if __name__ == "__main__":
    show_m2_feature_importance()
