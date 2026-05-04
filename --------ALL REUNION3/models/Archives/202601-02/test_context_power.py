import polars as pl
import lightgbm as lgb
import numpy as np
from pathlib import Path
from sklearn.metrics import roc_auc_score
import logging

# --- 設定 ---
# データディレクトリ（Script Bの出力先）
DATA_DIR = Path(
    "/workspace/data/XAUUSD/stratum_7_models/1A_2B/meta_labeled_oof_partitioned_v2"
)

# テストする「新特徴量」のリスト
CONTEXT_FEATURES = [
    "trend_bias_25",
    "atr_ratio",
    "hmm_prob_0",
    "hmm_prob_1",
    "e2a_mfdfa_hurst_mean_250",
    "e1c_adx_21",
]


def run_solitary_test():
    print("🚀 Starting Solitary Confinement Test (Context Features Only)...")

    # 1. データをロード（全期間は重いので、直近1年分などをサンプリングしても良いが、一旦全部読む）
    #    ※メモリ不足になる場合は limit=100000 などで制限してください
    files = sorted(list(DATA_DIR.glob("**/*.parquet")))
    if not files:
        print("❌ データファイルが見つかりません。Script Bを実行しましたか？")
        return

    print(f"📊 Loading data from {len(files)} partitions...")
    try:
        # 高速化のため、必要なカラムだけ読む
        cols_to_read = CONTEXT_FEATURES + ["meta_label"]

        # 簡易的に最初の100ファイル（約3ヶ月分）だけでテストする（高速化）
        # 全期間見たい場合は [:100] を削除してください
        target_files = files[:200]

        lf = pl.scan_parquet(target_files)
        df = lf.select(cols_to_read).collect()

        print(f"✅ Data Loaded: {df.shape}")
    except Exception as e:
        print(f"❌ データ読み込みエラー: {e}")
        return

    # 2. データセット作成
    X = df.select(CONTEXT_FEATURES).to_pandas()
    y = df["meta_label"].to_numpy()

    # 3. LightGBMデータセット
    train_data = lgb.Dataset(X, label=y)

    # 4. パラメータ（かなり緩める）
    params = {
        "objective": "binary",
        "metric": "auc",
        "boosting_type": "gbdt",
        "n_estimators": 100,  # 少なくても傾向は出る
        "learning_rate": 0.05,
        "num_leaves": 15,  # シンプルに
        "colsample_bytree": 1.0,  # 全力で使わせる
        "feature_pre_filter": False,  # フィルタリング無効化
        "min_data_in_leaf": 20,  # かなり緩く
        "verbose": -1,
        "seed": 42,
    }

    # 5. 学習
    print("\n🥊 Training LightGBM on ONLY context features...")
    model = lgb.train(params, train_data)

    # 6. 評価 (In-Sampleですが、重要度がつくかの確認なのでOK)
    y_pred = model.predict(X)
    auc = roc_auc_score(y, y_pred)

    print("\n" + "=" * 40)
    print(f"🎯 Solitary Test Result (AUC): {auc:.5f}")
    print("=" * 40)

    # 7. 重要度確認
    importance = model.feature_importance(importance_type="gain")
    imp_df = pl.DataFrame({"feature": CONTEXT_FEATURES, "importance": importance}).sort(
        "importance", descending=True
    )

    print(imp_df)

    if auc < 0.52:
        print("\n💀 判定: 死んでいます。")
        print("   これだけお膳立てしてもAUCが0.5に近い場合、")
        print("   『結合データが予測対象(meta_label)と全く相関していない』")
        print("   あるいは『データの質（計算ロジック）』に致命的な問題があります。")
    else:
        print("\n✨ 判定: 生きています！")
        print("   単体では予測能力があります。")
        print("   本番で使われないのは、M1等の他特徴量が強すぎる(Overshadow)せいです。")
        print(
            "   -> 対策: 『Interaction Features (M1 * Trend)』を作成して渡しましょう。"
        )


if __name__ == "__main__":
    run_solitary_test()
