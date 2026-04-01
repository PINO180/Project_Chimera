import sys
from pathlib import Path
import joblib
import pandas as pd
import lightgbm as lgb
import shutil
import logging

# --- プロジェクトパス設定 ---
# ユーザーの環境に合わせてパスを指定
MODEL_PATH = Path("/workspace/data/XAUUSD/stratum_7_models/1A_2B/m1_model_v2.pkl")
FEATURE_LIST_PATH = Path(
    "/workspace/data/XAUUSD/stratum_3_artifacts/1A_2B/train_12m_val_6m/final_feature_set_v2.txt"
)


def optimize_feature_list():
    # ログ設定
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    logger = logging.getLogger()

    print("=" * 60)
    print("### Feature List Optimization: Removing Unused Features ###")
    print("=" * 60)

    # 1. モデルの存在確認とロード
    if not MODEL_PATH.exists():
        logger.error(f"❌ M1 Model not found at: {MODEL_PATH}")
        return

    logger.info(f"🔄 Loading M1 model from: {MODEL_PATH}")
    try:
        model = joblib.load(MODEL_PATH)
    except Exception as e:
        logger.error(f"❌ Failed to load model: {e}")
        return

    # Boosterオブジェクトの取得
    if isinstance(model, lgb.Booster):
        booster = model
    elif hasattr(model, "booster_"):
        booster = model.booster_
    else:
        logger.error("❌ Unknown model format.")
        return

    # 2. 特徴量重要度(Gain)の取得
    importance = booster.feature_importance(importance_type="gain")
    feature_names = booster.feature_name()

    # DataFrame化
    df_imp = pd.DataFrame(
        {"feature_name": feature_names, "importance_gain": importance}
    )

    # 3. 選抜: Gain > 0 の特徴量のみ抽出
    # Gainが0の特徴量は、モデルの決定木において一度も使われていない（無視されている）
    active_features = df_imp[df_imp["importance_gain"] > 0].copy()

    # 重要度順にソート（確認用）
    active_features = active_features.sort_values("importance_gain", ascending=False)

    original_count = len(feature_names)
    new_count = len(active_features)
    removed_count = original_count - new_count

    print(f"\n📊 Optimization Stats:")
    print(f"  - Original Feature Count : {original_count}")
    print(f"  - Selected Feature Count : {new_count}")
    print(f"  - Removed (Unused)       : {removed_count}")

    if new_count == 0:
        logger.error(
            "❌ Error: No features have positive gain. Something is wrong with the model."
        )
        return

    # 4. バックアップ作成 (安全のため)
    backup_path = FEATURE_LIST_PATH.with_suffix(".txt.bak_full")
    if FEATURE_LIST_PATH.exists():
        shutil.copy(FEATURE_LIST_PATH, backup_path)
        logger.info(f"\n📦 Backup of original list created at: {backup_path}")

    # 5. ファイルの上書き保存
    # 注意: モデルから取得した特徴量名は、元のテキストファイルの順番とは異なる可能性があるが、
    # 重要なのは「名前が合っていること」なので、リスト順で保存しても問題ない。
    # ただし、lightgbmは列の順番に敏感な場合があるため、
    # できれば「元のファイルの順番を維持しつつ、不要なものを消す」のがベストだが、
    # 次回の学習(Script C)でこの新しいリストを使ってデータセットを作り直すため、
    # ここでの順番が「新しい正」となる。したがって、重要度順あるいは元の順序で保存すればよい。
    # ここでは、元のリストの順序を尊重してフィルタリングする方式をとる。

    # 元のファイルからリストを読み込む（順序維持のため）
    with open(backup_path, "r") as f:
        original_lines = [line.strip() for line in f if line.strip()]

    # 有効な特徴量名のセット
    active_feature_set = set(active_features["feature_name"].values)

    # 元の順序を保ちつつ、有効なものだけ残す
    final_list = [f for f in original_lines if f in active_feature_set]

    # ※念のため、モデルにはあるがファイルにない（あり得ないが）ケースの救済
    # 通常はないはずだが、整合性チェック
    for f in active_feature_set:
        if f not in final_list:
            final_list.append(f)

    # 保存
    with open(FEATURE_LIST_PATH, "w") as f:
        f.write("\n".join(final_list))

    logger.info(f"✅ Feature list updated successfully at: {FEATURE_LIST_PATH}")
    logger.info(f"📝 New list contains {len(final_list)} features.")

    print("\n" + "=" * 60)
    print("🚀 NEXT STEP: Re-run 'Script C' (model_training_metalabeling_C.py)")
    print("   This will train a new, lightweight M1 model using only these features.")
    print("=" * 60)


if __name__ == "__main__":
    optimize_feature_list()
