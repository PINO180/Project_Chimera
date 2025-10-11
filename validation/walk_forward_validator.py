# =================================================================
#
# walk_forward_validator_v5_final.py
#
# Project Forge - Second Line of Defense
#
# -----------------------------------------------------------------
#
# **設計思想とアーキテクチャ**
#
# 過去の失敗の歴史 (v3.x, v4.x) から得られた最大の教訓は、
# Daskの遅延評価が我々の環境では不安定要因である、という一点に尽きる。
# 特に、.shift() と .dropna() を組み合わせた目的変数生成は、
# Daskの計算グラフ内でインデックスの不整合を頻繁に引き起こし、
# 致命的な "Length of labels differs" エラーの原因となった [cite: 135-136, 144]。
#
# 本スクリプト (v5) は、その問題を根本から解決するため、
# 以下の「脱Dask・物理データ確定」アーキテクチャを全面的に採用する。
#
# 1. **脱Dask**: 不安定要因であったDaskを完全に排除。データ操作はPolarsに一本化。
#
# 2. **ストリーミング・ウォークフォワード**:
#    マスターテーブル全体を一度に読み込むのではなく、ウォークフォワードの
#    各分割（Split）で必要となるパーティションファイル群のみを
#    `polars.scan_parquet`でスキャンする。
#
# 3. **物理データ確定 (最重要)**:
#    各分割において、目的変数を生成した後、`collect(streaming=True)`を実行する。
#    これにより、特徴量(X)と目的変数(y)が完全に整合した状態でメモリ上に
#    物理的に展開される。これにより、インデックス不整合の問題は原理的に発生しない。
#
# 4. **標準ライブラリによる学習**:
#    物理的に確定したクリーンなデータを、オリジナルの`lightgbm`および`shap`
#    ライブラリに渡す。これにより、dask-lightgbm等の複雑なラッパーは不要となり、
#    最も安定した方法で学習と評価を実行できる。
#
# このアーキテクチャは、AVスクリプトやbuild_master_table.pyで得た
# 「小さく分割して処理し、結果を物理的に確定させる」という
# 成功体験の集大成である [cite: 24, 66]。
#
# =================================================================

import gc
from pathlib import Path

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
import polars as pl
import shap
from tqdm import tqdm

# --- Project Forge Blueprint ---
# 他のスクリプトと設定を共有するため、blueprintからパスをインポート
# (blueprint.pyが同じディレクトリにあると仮定)
try:
    from blueprint import (
        S3_FINAL_FEATURE_TEAM,
        S3_SHAP_SCORES,
        S4_MASTER_TABLE_PARTITIONED,
    )
except ImportError:
    # blueprint.py がない場合のフォールバック
    print("WARNING: blueprint.py not found. Using hardcoded paths.")
    S4_MASTER_TABLE_PARTITIONED = Path(
        "/workspace/data/XAUUSD/stratum_4_master/1A_2B/master_table_partitioned"
    )
    S3_ARTIFACTS = Path("/workspace/data/XAUUSD/stratum_3_artifacts/1A_2B")
    S3_FINAL_FEATURE_TEAM = S3_ARTIFACTS / "final_feature_team.joblib"
    S3_SHAP_SCORES = S3_ARTIFACTS / "shap_scores.joblib"
    S3_ARTIFACTS.mkdir(parents=True, exist_ok=True)


# --- 設定項目 ---
class Config:
    """スクリプトの振る舞いを制御する設定クラス"""

    # 入出力パス
    MASTER_TABLE_PATH = S4_MASTER_TABLE_PARTITIONED
    OUTPUT_FEATURE_LIST_PATH = S3_FINAL_FEATURE_TEAM
    OUTPUT_SHAP_SCORES_PATH = S3_SHAP_SCORES

    # ウォークフォワード検証の設定
    N_SPLITS = 5  # データセットをいくつのフォールドに分割するか

    # 目的変数の定義
    TARGET_SHIFT = -30  # 30ステップ未来のリターンを予測
    TARGET_THRESHOLD = 0.0005  # 0.05%以上の価格上昇を「買い」シグナル (ラベル=1) とする

    # LightGBMモデルのパラメータ (軽量な設定)
    LGBM_PARAMS = {
        "objective": "binary",
        "metric": "auc",
        "n_estimators": 500,
        "learning_rate": 0.05,
        "feature_fraction": 0.8,
        "bagging_fraction": 0.8,
        "bagging_freq": 1,
        "lambda_l1": 0.1,
        "lambda_l2": 0.1,
        "num_leaves": 31,
        "verbose": -1,
        "n_jobs": -1,
        "seed": 42,
        "boosting_type": "gbdt",
    }

    # 特徴量選抜
    TOP_N_FEATURES = 100  # SHAPスコア上位何個の特徴量を選抜するか


def get_partition_paths(master_path: Path) -> list[Path]:
    """マスターテーブルの全パーティションパスを時系列順に取得する"""
    if not master_path.is_dir():
        raise FileNotFoundError(f"Master table directory not found: {master_path}")

    # year/month/day の構造を想定
    paths = sorted(list(master_path.glob("year=*/month=*/day=*/*.parquet")))
    if not paths:
        raise FileNotFoundError(
            f"No parquet files found in master table directory: {master_path}"
        )
    print(f"Found {len(paths)} daily partitions.")
    return paths


def define_walk_forward_splits(
    partition_paths: list[Path], n_splits: int
) -> list[tuple[list[Path], list[Path]]]:
    """ウォークフォワード検証のための分割を定義する"""
    splits = []
    total_partitions = len(partition_paths)

    # 拡張ウィンドウ方式 (Expanding Window)
    # 最初の訓練期間を確保するため、全期間を (n_splits + 1) で分割
    initial_train_size = total_partitions // (n_splits + 1)
    validation_size = initial_train_size

    for i in range(n_splits):
        train_end_idx = initial_train_size + i * validation_size
        val_end_idx = train_end_idx + validation_size

        # 最後のスプリットで残りの全データを含める
        if i == n_splits - 1:
            val_end_idx = total_partitions

        train_paths = partition_paths[:train_end_idx]
        val_paths = partition_paths[train_end_idx:val_end_idx]

        if not train_paths or not val_paths:
            print(f"Skipping split {i + 1} due to empty train/validation set.")
            continue

        splits.append((train_paths, val_paths))

    print(f"Defined {len(splits)} walk-forward splits.")
    return splits


def load_and_prepare_data_for_split(
    train_paths: list[Path], val_paths: list[Path]
) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series, list[str]]:
    """
    指定されたパーティションを読み込み、目的変数を生成し、
    物理的に確定された訓練・検証データセットを返す。
    """
    print(
        f"  Loading {len(train_paths)} train partitions and {len(val_paths)} validation partitions..."
    )

    # Polars LazyFrameで必要なファイルのみをスキャン
    lazy_df = pl.scan_parquet(train_paths + val_paths)

    # --- 目的変数 (ターゲット) の生成 ---
    # .shift() を使って未来のリターンを計算
    future_return = lazy_df.select(
        pl.col("timestamp"),
        (pl.col("close").shift(Config.TARGET_SHIFT) / pl.col("close") - 1).alias(
            "future_return"
        ),
    )

    # LazyFrameを結合
    lazy_df = lazy_df.join(future_return, on="timestamp", how="left")

    # ラベルを定義 (1: 上昇, 0: その他)
    lazy_df = lazy_df.with_columns(
        pl.when(pl.col("future_return") > Config.TARGET_THRESHOLD)
        .then(1)
        .otherwise(0)
        .alias("target")
    )

    # 目的変数生成に伴うNaNを持つ行を削除
    # これがDaskでインデックス不整合を引き起こした元凶
    lazy_df = lazy_df.drop_nulls(subset=["target"])

    # --- 物理データ確定 ---
    # ここで .collect() を呼ぶことで、計算が実行され、
    # 完全に整合性の取れたデータがメモリ上に展開される。
    print("  Materializing data... (This may take a moment)")
    df = lazy_df.collect(streaming=True)
    print(f"  Materialized dataframe shape: {df.shape}")

    # Pandas DataFrameに変換して後続処理へ
    df = df.to_pandas()

    # メモリ解放
    del lazy_df
    gc.collect()

    # 特徴量カラムとターゲットを分離
    target_col = "target"
    # タイムスタンプと目的変数そのものは特徴量から除外
    feature_cols = [
        c for c in df.columns if c not in ["timestamp", "future_return", target_col]
    ]

    X = df[feature_cols]
    y = df[target_col]

    # 訓練データと検証データを再度分割
    # 境界となるタイムスタンプを取得
    split_timestamp = pd.to_datetime(
        pl.scan_parquet(train_paths[-1]).select(pl.max("timestamp")).collect()[0, 0]
    )

    train_mask = df["timestamp"] <= split_timestamp
    val_mask = df["timestamp"] > split_timestamp

    X_train, y_train = X[train_mask], y[train_mask]
    X_val, y_val = X[val_mask], y[val_mask]

    print(f"  Train set shape: {X_train.shape}, Validation set shape: {X_val.shape}")

    return X_train, y_train, X_val, y_val, feature_cols


def main():
    """スクリプトのメイン実行関数"""
    print("--- Project Forge: Second Line of Defense ---")
    print("--- Walk-Forward Validator (v5 - Dask-Free) ---")

    # 1. パーティションパスの取得
    all_paths = get_partition_paths(Config.MASTER_TABLE_PATH)

    # 2. ウォークフォワード分割の定義
    splits = define_walk_forward_splits(all_paths, Config.N_SPLITS)

    all_shap_values = []
    feature_names = None

    # 3. 各分割で学習と評価を実行
    for i, (train_paths, val_paths) in enumerate(splits):
        print(f"\n--- Processing Split {i + 1}/{len(splits)} ---")

        try:
            # 3a. データの読み込みと物理的確定
            X_train, y_train, X_val, y_val, current_features = (
                load_and_prepare_data_for_split(train_paths, val_paths)
            )

            if feature_names is None:
                feature_names = current_features

            if X_train.empty or X_val.empty:
                print("  Skipping split due to empty data after preparation.")
                continue

            # 3b. モデルの学習
            print("  Training LightGBM model...")
            model = lgb.LGBMClassifier(**Config.LGBM_PARAMS)
            model.fit(
                X_train,
                y_train,
                eval_set=[(X_val, y_val)],
                callbacks=[lgb.early_stopping(100, verbose=False)],
            )

            # 3c. SHAP値の計算
            print("  Calculating SHAP values...")
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(X_val)

            # shap_values[1] がクラス1（買いシグナル）に対するSHAP値
            all_shap_values.append(pd.DataFrame(shap_values[1], columns=feature_names))

            # メモリクリーンアップ
            del X_train, y_train, X_val, y_val, model, explainer, shap_values
            gc.collect()

        except Exception as e:
            print(f"  !! ERROR in Split {i + 1}: {e}")
            print("  Skipping this split and continuing...")
            continue

    if not all_shap_values:
        print("\n--- No splits were successfully processed. Exiting. ---")
        return

    # 4. 全分割の結果を集計
    print("\n--- Aggregating results from all splits ---")
    combined_shap_df = pd.concat(all_shap_values)

    # 平均絶対SHAP値を計算して特徴量をランク付け
    mean_abs_shap = combined_shap_df.abs().mean().sort_values(ascending=False)

    shap_scores_df = pd.DataFrame(
        {"feature": mean_abs_shap.index, "mean_abs_shap": mean_abs_shap.values}
    )

    # 5. 結果の保存
    print(f"Saving SHAP scores to: {Config.OUTPUT_SHAP_SCORES_PATH}")
    joblib.dump(shap_scores_df, Config.OUTPUT_SHAP_SCORES_PATH)

    final_feature_team = shap_scores_df.head(Config.TOP_N_FEATURES)["feature"].tolist()

    print(
        f"Saving final feature team (top {Config.TOP_N_FEATURES}) to: {Config.OUTPUT_FEATURE_LIST_PATH}"
    )
    joblib.dump(final_feature_team, Config.OUTPUT_FEATURE_LIST_PATH)

    # 6. 結果の表示
    print("\n--- Top 20 Features by Mean Absolute SHAP Value ---")
    print(shap_scores_df.head(20).to_string(index=False))

    print(
        f"\n✅ Successfully completed. Found {len(final_feature_team)} elite features."
    )
    print("--- End of Process ---")


if __name__ == "__main__":
    main()
