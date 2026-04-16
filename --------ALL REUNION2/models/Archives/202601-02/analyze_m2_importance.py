import sys
from pathlib import Path
import joblib
import pandas as pd
import lightgbm as lgb
import logging

# --- プロジェクトのルートディレクトリをPythonの検索パスに追加 ---
project_root = Path(__file__).resolve().parents[1]
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

from blueprint import S7_M2_MODEL_PKL


def analyze_m2_importance_corrected():
    # ログ設定
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    logger = logging.getLogger()

    print("=" * 80)
    print("### M2 Model Feature Importance Analysis (Corrected Mapping) ###")
    print("=" * 80)

    # 1. モデルのロード
    model_path = S7_M2_MODEL_PKL
    if not model_path.exists():
        logger.error(f"❌ モデルが見つかりません: {model_path}")
        return

    logger.info(f"🔄 Loading M2 model from: {model_path}")
    try:
        model = joblib.load(model_path)
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        return

    # Boosterオブジェクトの取得
    if isinstance(model, lgb.Booster):
        booster = model
    elif hasattr(model, "booster_"):
        booster = model.booster_
    else:
        logger.error("Unknown model format.")
        return

    # 2. 特徴量リストの定義
    # Two-Stage Stacking構成で使用した、正確な特徴量リスト（8個）を定義
    # ※ここが「Column_0, Column_1...」の正体です
    m2_features = [
        "hmm_prob_0",
        "hmm_prob_1",
        "atr_ratio",
        "trend_bias_25",
        "e1a_statistical_kurtosis_50",
        "e1c_adx_21",
        "e2a_mfdfa_hurst_mean_250",
        "e2a_kolmogorov_complexity_60",
    ]

    # モデルが学習に使用した特徴量の数を確認
    num_model_features = booster.num_feature()
    print(f"\n[Model Input Features]: {num_model_features}")
    print(f"[Defined Feature Names]: {len(m2_features)}")

    if num_model_features != len(m2_features):
        logger.warning(
            f"⚠️ 注意: モデルの特徴量数({num_model_features})とリストの数({len(m2_features)})が一致しません。"
        )
        logger.warning("正しいモデルファイル(v2)がロードされているか確認してください。")
        # 一致しない場合は、安全のためリストの長さをモデルに合わせるか、モデルの長さに合わせる
        min_len = min(num_model_features, len(m2_features))
        m2_features = m2_features[:min_len]

    # 3. 重要度の抽出 (Gain)
    importance = booster.feature_importance(importance_type="gain")

    # 4. DataFrame作成とマッピング
    df_imp = pd.DataFrame({"feature_name": m2_features, "importance_gain": importance})

    # 5. ソートと表示
    df_imp = df_imp.sort_values("importance_gain", ascending=False).reset_index(
        drop=True
    )

    # 貢献度（%）の計算
    total_gain = df_imp["importance_gain"].sum()
    if total_gain > 0:
        df_imp["contribution_pct"] = (df_imp["importance_gain"] / total_gain) * 100
    else:
        df_imp["contribution_pct"] = 0.0

    # 表示設定
    pd.set_option("display.max_rows", None)
    pd.set_option("display.width", 1000)

    print("\n" + "=" * 80)
    print(f"       🏆 M2 FEATURE RANKING (Total Gain: {total_gain:.4f})       ")
    print("=" * 80)

    # フォーマットして表示
    print(f"{'Rank':<5} | {'Feature Name':<35} | {'Gain':<12} | {'Contribution':<10}")
    print("-" * 70)

    for i, row in df_imp.iterrows():
        print(
            f"{i + 1:<5} | {row['feature_name']:<35} | {row['importance_gain']:<12.4f} | {row['contribution_pct']:.2f}%"
        )

    print("=" * 80)


if __name__ == "__main__":
    analyze_m2_importance_corrected()
