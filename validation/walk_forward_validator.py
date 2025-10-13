# =================================================================
#
# walk_forward_validator_v7.2_final_fix.py
#
# Project Forge - Second Line of Defense
#
# -----------------------------------------------------------------
#
# **設計思想とアーキテクチャ (v7.2 - 最終修正版)**
#
# v7.1で'No booster found'エラーを解決したが、今度は
# 'pandas dtypes must be int, float or bool'という型エラーに直面した。
# これは、特徴量リストに文字列型である'timeframe'列が
# 誤って含まれていたことが原因である。
#
# 本スクリプト(v7.2)は、特徴量リストから'timeframe'列を
# 明示的に除外することで、この型エラーを解決する。
# これにより、LightGBMには純粋な数値データのみが渡されるようになり、
# 正常に学習が実行されるはずである。
#
# =================================================================

import polars as pl
import lightgbm as lgb
import shap
import pandas as pd
from pathlib import Path
import joblib
import gc
from typing import List, Tuple, Optional

# --- Project Forge Blueprint ---
try:
    from blueprint import (
        S4_MASTER_TABLE_PARTITIONED,
        S3_FINAL_FEATURE_TEAM,
        S3_SHAP_SCORES,
    )
except ImportError:
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
    TRAIN_BATCH_SIZE_DAYS = 5
    VALIDATION_BATCH_SIZE_DAYS = 5
    N_SPLITS = 5
    TARGET_SHIFT = -30
    TARGET_THRESHOLD = 0.0005
    TOP_N_FEATURES = 100
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


# --- ヘルパー関数 ---


def get_partition_paths(master_path: Path) -> list[Path]:
    paths = sorted(list(master_path.glob("year=*/month=*/day=*/*.parquet")))
    if not paths:
        raise FileNotFoundError(f"No parquet files found: {master_path}")
    print(f"Found {len(paths)} daily partitions.")
    return paths


def define_walk_forward_splits(
    paths: list[Path], n_splits: int
) -> list[tuple[list[Path], list[Path]]]:
    splits = []
    total = len(paths)
    fold_size = total // (n_splits + 1)
    for i in range(n_splits):
        train_end = fold_size * (i + 1)
        val_end = train_end + fold_size if i < n_splits - 1 else total
        splits.append((paths[:train_end], paths[train_end:val_end]))
    print(f"Defined {len(splits)} walk-forward splits.")
    return splits


def prepare_data_from_paths(
    paths: List[Path], features: List[str]
) -> Tuple[pd.DataFrame, pd.Series]:
    if not paths:
        return pd.DataFrame(), pd.Series()
    lazy_df = pl.scan_parquet(paths)
    lazy_df = (
        lazy_df.with_columns(
            (pl.col("close").shift(Config.TARGET_SHIFT) / pl.col("close") - 1).alias(
                "future_return"
            )
        )
        .with_columns(
            pl.when(pl.col("future_return") > Config.TARGET_THRESHOLD)
            .then(1)
            .otherwise(0)
            .alias("target")
        )
        .drop_nulls(subset=["target"])
    )

    df = lazy_df.collect(engine="streaming").to_pandas()

    # 渡されたfeatureリストに存在しない列がある場合のエラーを防ぐ
    valid_features = [f for f in features if f in df.columns]

    return df[valid_features], df["target"]


def train_model_for_split(
    train_paths: List[Path], features: List[str]
) -> Optional[lgb.LGBMClassifier]:
    model = None
    batches = [
        train_paths[j : j + Config.TRAIN_BATCH_SIZE_DAYS]
        for j in range(0, len(train_paths), Config.TRAIN_BATCH_SIZE_DAYS)
    ]
    print(f"  Training data is split into {len(batches)} batches.")

    for i, batch_paths in enumerate(batches):
        print(f"    Training on batch {i + 1}/{len(batches)}...")
        X_batch, y_batch = prepare_data_from_paths(batch_paths, features)
        if X_batch.empty:
            continue

        if model is None:
            model = lgb.LGBMClassifier(**Config.LGBM_PARAMS)
            model.fit(X_batch, y_batch)
        else:
            model.fit(X_batch, y_batch, init_model=model)

        del X_batch, y_batch
        gc.collect()

    return model


def evaluate_model_for_split(
    model: lgb.LGBMClassifier, val_paths: List[Path], features: List[str]
) -> Optional[pd.DataFrame]:
    explainer = shap.TreeExplainer(model)
    shap_dfs = []
    batches = [
        val_paths[j : j + Config.VALIDATION_BATCH_SIZE_DAYS]
        for j in range(0, len(val_paths), Config.VALIDATION_BATCH_SIZE_DAYS)
    ]
    print(
        f"  Validation data is split into {len(batches)} batches for SHAP calculation."
    )

    for i, batch_paths in enumerate(batches):
        print(f"    Calculating SHAP for validation batch {i + 1}/{len(batches)}...")
        X_batch, _ = prepare_data_from_paths(batch_paths, features)
        if X_batch.empty:
            continue

        shap_values = explainer.shap_values(X_batch)
        shap_dfs.append(pd.DataFrame(shap_values[1], columns=X_batch.columns))
        del X_batch, shap_values
        gc.collect()

    return pd.concat(shap_dfs) if shap_dfs else None


# --- メイン実行部 ---


def main():
    print("--- Project Forge: Second Line of Defense ---")
    print("--- Walk-Forward Validator (v7.2 - Final Fix) ---")

    all_paths = get_partition_paths(S4_MASTER_TABLE_PARTITIONED)
    splits = define_walk_forward_splits(all_paths, Config.N_SPLITS)
    all_shap_dfs = []

    try:
        schema = pl.read_parquet_schema(all_paths[0])
        # --- 【バグ修正箇所】 ---
        # timeframe列は学習に使わないメタデータなので、特徴量リストから除外する
        base_cols = {
            "timestamp",
            "future_return",
            "target",
            "open",
            "high",
            "low",
            "close",
            "volume",
            "timeframe",
        }
        features = [col for col in schema.keys() if col not in base_cols]
        print(f"Found {len(features)} feature columns to use for training.")
    except Exception as e:
        print(f"Could not read schema from first partition: {e}")
        return

    for i, (train_paths, val_paths) in enumerate(splits):
        print(f"\n--- Processing Split {i + 1}/{len(splits)} ---")
        try:
            model = train_model_for_split(train_paths, features)
            if model is None:
                print("  Skipping split: No model was trained.")
                continue

            shap_df = evaluate_model_for_split(model, val_paths, features)
            if shap_df is None:
                print("  Skipping split: No SHAP values were calculated.")
                continue

            all_shap_dfs.append(shap_df)
            del model, shap_df
            gc.collect()
        except Exception as e:
            print(f"  !! ERROR in Split {i + 1}: {e}\n  Skipping...")
            continue

    if not all_shap_dfs:
        print("\n--- No splits were successfully processed. Exiting. ---")
        return

    print("\n--- Aggregating results from all splits ---")
    final_shap_df = (
        pd.concat(all_shap_dfs).abs().mean().sort_values(ascending=False).reset_index()
    )
    final_shap_df.columns = ["feature", "mean_abs_shap"]

    print(f"Saving SHAP scores to: {S3_SHAP_SCORES}")
    joblib.dump(final_shap_df, S3_SHAP_SCORES)

    final_team = final_shap_df.head(Config.TOP_N_FEATURES)["feature"].tolist()
    print(
        f"Saving final feature team (top {Config.TOP_N_FEATURES}) to: {S3_FINAL_FEATURE_TEAM}"
    )
    joblib.dump(final_team, S3_FINAL_FEATURE_TEAM)

    print("\n--- Top 20 Features by Mean Absolute SHAP Value ---")
    print(final_shap_df.head(20).to_string(index=False))

    print(f"\n✅ Successfully completed. Found {len(final_team)} elite features.")
    print("--- End of Process ---")


if __name__ == "__main__":
    main()
