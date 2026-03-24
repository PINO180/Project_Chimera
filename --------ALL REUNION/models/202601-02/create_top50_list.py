import sys
import json
import joblib
import pandas as pd
from pathlib import Path

# プロジェクトルートをパスに追加
sys.path.append(str(Path(__file__).resolve().parents[1]))
import blueprint as config


def create_top50_list():
    # =================================================================
    # 【修正】参照先を「実績のあるM2モデル」に変更
    # =================================================================
    # ユーザー様が提示されたパスを使用
    model_path = Path("/workspace/data/XAUUSD/stratum_7_models/1A_2B/m2_model_v2.pkl")
    feature_list_path = Path(
        "/workspace/data/XAUUSD/stratum_3_artifacts/1A_2B/train_12m_val_6m/final_feature_set_v3.txt"
    )
    output_path = Path("models/TOP_50_FEATURES.json")

    print("=" * 60)
    print("🚀 Top 50 Feature List Generator (Fixed Source)")
    print("=" * 60)
    print(f"🔄 Loading Reference Model: {model_path}")

    if not model_path.exists():
        print(f"❌ モデルが見つかりません: {model_path}")
        print("パスが正しいか確認してください。")
        return

    # =================================================================
    # 1. データロード & 特徴量名リストの再構築
    # =================================================================
    model = joblib.load(model_path)

    # 基本特徴量リストのロード
    with open(feature_list_path, "r") as f:
        base_features = [line.strip() for line in f if line.strip()]

    # 【重要】M2モデル学習時と同じ「追加特徴量」を定義してリストを結合する
    # これをしないと特徴量名と重要度の並びがズレてしまいます
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

    # Feature Importance (Gain) の取得
    try:
        if hasattr(model, "booster_"):
            importance = model.booster_.feature_importance(importance_type="gain")
        else:
            importance = model.feature_importance(importance_type="gain")
    except Exception as e:
        print(f"❌ 重要度の取得エラー: {e}")
        return

    # データ長合わせ
    min_len = min(len(importance), len(full_feature_names))
    df = pd.DataFrame(
        {
            "feature_name": full_feature_names[:min_len],
            "importance": importance[:min_len],
        }
    )

    # =================================================================
    # 2. フィルタリング (B案改ルール)
    # =================================================================
    # m1_pred_proba や 旧文脈特徴量(HMM等) は、
    # 今回の「Top 50入力リスト」としては除外します（システム側で別途扱うため）

    ignore_names = set(m2_specific_features)

    def get_timeframe(name):
        parts = name.split("_")
        known_tfs = ["M15", "H1", "H4", "H6", "H12", "M8", "D1", "W1", "MN"]
        if parts[-1] in known_tfs:
            return parts[-1]
        for tf in known_tfs:
            if name.endswith(f"_{tf}"):
                return tf
        return "Other"

    df["timeframe"] = df["feature_name"].apply(get_timeframe)

    # 除外対象タイムフレーム
    exclude_tfs = ["M8", "D1", "W1", "MN", "Other"]

    # フィルタリング実行
    # 1. M2固有の特徴量（m1_proba等）を除外（これらは純粋な特徴量ではないため）
    # 2. 不要な時間足を除外
    df_filtered = df[
        (~df["feature_name"].isin(ignore_names)) & (~df["timeframe"].isin(exclude_tfs))
    ].copy()

    # =================================================================
    # 3. ランキング順にソート & 抽出
    # =================================================================
    df_sorted = df_filtered.sort_values("importance", ascending=False).reset_index(
        drop=True
    )
    top_50_features = df_sorted.head(50)["feature_name"].tolist()

    # =================================================================
    # 4. JSON保存
    # =================================================================
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(top_50_features, f, indent=4)

    print(f"✅ JSON Saved to: {output_path}")
    print("-" * 60)
    print("🏆 Top 10 Features (Reference from M2 Model):")
    for i, name in enumerate(top_50_features[:10]):
        imp_val = df_sorted.iloc[i]["importance"]
        print(f"   {i + 1:2d}. {name:<60} (Gain: {imp_val:.2f})")
    print("..." + "\n" + "=" * 60)


if __name__ == "__main__":
    create_top50_list()
