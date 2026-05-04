import sys
from pathlib import Path

# プロジェクトのルートディレクトリをPythonの検索パスに追加
sys.path.append(str(Path(__file__).resolve().parents[1]))

import polars as pl
import pandas as pd
import numpy as np
import lightgbm as lgb
import shap
import warnings
import argparse
from datetime import timedelta
import re

# blueprintから一元管理された設定を読み込む
from blueprint import (
    S2_FEATURES_AFTER_AV,
    S3_PRESELECTION,
    S3_ELITE_LF_FEATURES,
    S3_ARTIFACTS,
    S3_FINAL_FEATURE_TEAM,
    S3_SHAP_SCORES,
)

# --- ウォークフォワード設定 ---
# --- 修正点: 訓練期間を短縮し、フォールド数を増やす ---
TRAIN_PERIOD_MONTHS = 12  # 24から12へ変更
VALIDATION_PERIOD_MONTHS = 6
ANALYSIS_START_DATE = "2021-07-02"
ANALYSIS_END_DATE = "2024-12-31"


def parse_suffixed_feature_name(suffixed_name: str) -> tuple[str, str]:
    """'e1a_rsi_14_D1' のような名前から、元の名前とタイムフレームを抽出する。"""
    parts = suffixed_name.split("_")
    timeframe_pattern = re.compile(r"^(M[0-9\.]+|H[0-9]+|D[0-9]+|W[0-9]+|MN|tick)$")
    if len(parts) > 1 and timeframe_pattern.match(parts[-1]):
        base_name = "_".join(parts[:-1])
        timeframe = parts[-1]
        return base_name, timeframe
    return suffixed_name, None


def find_feature_files(elite_features: list[str]) -> dict[str, Path]:
    """精鋭特徴量リスト(サフィックス付き)から、元のファイルへのマッピングを作成する。"""
    print("Mapping elite features to their source files...")
    feature_map = {}

    source_file_schema_map = {}
    all_files = list(S2_FEATURES_AFTER_AV.rglob("*.parquet"))
    for file_path in all_files:
        try:
            schema = pl.read_parquet_schema(file_path)
            source_file_schema_map[file_path] = set(schema.keys())
        except Exception as e:
            print(f"Warning: Could not read schema from {file_path}: {e}")

    for suffixed_feature in elite_features:
        base_name, timeframe = parse_suffixed_feature_name(suffixed_feature)

        found = False
        for file_path, schema_cols in source_file_schema_map.items():
            if base_name in schema_cols:
                feature_map[suffixed_feature] = file_path
                found = True
                break
        if not found:
            print(
                f"Warning: Could not find source file for feature '{base_name}' (from '{suffixed_feature}')"
            )

    missing_features = set(elite_features) - set(feature_map.keys())
    if missing_features:
        print(
            f"Warning: {len(missing_features)} features could not be mapped to any file."
        )

    print(
        f"Successfully mapped {len(feature_map)} features to {len(set(feature_map.values()))} unique files."
    )
    return feature_map


def build_virtual_lf(
    elite_features: list[str], feature_map: dict[str, Path]
) -> pl.LazyFrame:
    """with_columnsとaliasを使い、重複参照問題を解決して仮想LFを構築する。"""
    print("Building virtual LazyFrame for walk-forward analysis...")

    unique_files = sorted(list(set(feature_map.values())))
    if not unique_files:
        raise ValueError("Cannot build LazyFrame: No source files were found.")

    features_by_file = {}
    for suffixed_feature, path in feature_map.items():
        if path not in features_by_file:
            features_by_file[path] = []
        features_by_file[path].append(suffixed_feature)

    processed_lfs = []
    for file_path in unique_files:
        suffixed_features_in_file = features_by_file[file_path]

        base_name_map = {}
        for sf in suffixed_features_in_file:
            base_name, _ = parse_suffixed_feature_name(sf)
            if base_name not in base_name_map:
                base_name_map[base_name] = []
            base_name_map[base_name].append(sf)

        original_cols_needed = list(base_name_map.keys())
        lf = pl.scan_parquet(file_path).select(["timestamp"] + original_cols_needed)

        alias_expressions = []
        for base_name, suffixed_names in base_name_map.items():
            for sf_name in suffixed_names:
                alias_expressions.append(pl.col(base_name).alias(sf_name))

        lf = lf.with_columns(alias_expressions)

        final_cols_for_lf = ["timestamp"] + suffixed_features_in_file
        processed_lfs.append(lf.select(final_cols_for_lf))

    final_lf = processed_lfs[0]
    for lf_to_join in processed_lfs[1:]:
        final_lf = final_lf.join_asof(lf_to_join, on="timestamp", strategy="backward")

    return final_lf.sort("timestamp")


def get_walk_forward_splits(start_date_str, end_date_str, train_months, val_months):
    """ウォークフォワードの期間を生成するジェネレータ"""
    start_date = pd.to_datetime(start_date_str)
    end_date = pd.to_datetime(end_date_str)

    current_start = start_date
    while True:
        train_end = current_start + pd.DateOffset(months=train_months)
        val_end = train_end + pd.DateOffset(months=val_months)

        if val_end > end_date:
            break

        yield {
            "train_start": current_start,
            "train_end": train_end,
            "val_start": train_end,
            "val_end": val_end,
        }
        current_start = current_start + pd.DateOffset(months=val_months)


def main(test_mode: bool):
    """メイン実行関数"""
    print("Starting Phase 1: M1 Walk-Forward Validation...")
    S3_ARTIFACTS.mkdir(parents=True, exist_ok=True)

    print(f"Loading elite LF features from: {S3_ELITE_LF_FEATURES}")
    if not S3_ELITE_LF_FEATURES.exists():
        raise FileNotFoundError(
            f"Elite feature list not found. Please run 01_LF_pre_selector.py first."
        )

    with open(S3_ELITE_LF_FEATURES, "r") as f:
        elite_lf_features = [line.strip() for line in f if line.strip()]
    print(f"Loaded {len(elite_lf_features)} elite LF features.")

    feature_file_map = find_feature_files(elite_lf_features)
    virtual_lf = build_virtual_lf(elite_lf_features, feature_file_map)

    # 修正: _M5.parquet を _M1.parquet に変更
    price_path = (
        S2_FEATURES_AFTER_AV / "feature_value_a_vast_universeA/features_e1a_M1.parquet"
    )
    if not price_path.exists():
        raise FileNotFoundError(
            f"M1 price data for target generation not found at {price_path}"
        )

    target_lf = (
        pl.scan_parquet(price_path)
        .select(["timestamp", "close"])
        .sort("timestamp")
        # M1の shift(-5) なので「5分後の収益率」をターゲットとする
        .with_columns((pl.col("close").shift(-5) / pl.col("close") - 1).alias("target"))
        .select(["timestamp", "target"])
    )

    data_lf = virtual_lf.join_asof(target_lf, on="timestamp", strategy="backward")

    splits = list(
        get_walk_forward_splits(
            ANALYSIS_START_DATE,
            ANALYSIS_END_DATE,
            TRAIN_PERIOD_MONTHS,
            VALIDATION_PERIOD_MONTHS,
        )
    )

    if test_mode:
        print("\n--- TEST MODE ACTIVE: Running only the first 2 folds. ---")
        splits = splits[:2]

    all_shap_scores, all_predictions = [], []

    for i, split in enumerate(splits):
        print(f"\n--- Processing Fold {i + 1}/{len(splits)} ---")
        print(
            f"Train: {split['train_start'].date()} to {split['train_end'].date()} | Validation: {split['val_start'].date()} to {split['val_end'].date()}"
        )

        print("  Loading and preparing training data...")
        train_df = (
            data_lf.filter(
                pl.col("timestamp").is_between(split["train_start"], split["train_end"])
            )
            .collect(streaming=True)
            .to_pandas()
            .dropna(subset=["target"])
            .fillna(0)
        )

        X_train = train_df[elite_lf_features]
        y_train = train_df["target"]

        if X_train.empty:
            print("  Warning: Training data is empty for this fold. Skipping.")
            continue

        print(f"  Training model on {len(X_train)} samples...")
        model = lgb.LGBMRegressor(random_state=42, verbosity=-1, n_jobs=-1)
        model.fit(X_train, y_train)

        print("  Calculating SHAP values...")
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_train)

        mean_abs_shap = np.abs(shap_values).mean(axis=0)
        shap_df = pd.DataFrame(
            {"feature": X_train.columns, "shap_value": mean_abs_shap, "fold": i + 1}
        )
        all_shap_scores.append(shap_df)

        print("  Loading and preparing validation data...")
        val_df = (
            data_lf.filter(
                pl.col("timestamp").is_between(split["val_start"], split["val_end"])
            )
            .collect(streaming=True)
            .to_pandas()
            .fillna(0)
        )

        X_val = val_df[elite_lf_features]

        if not X_val.empty:
            print(f"  Generating predictions on {len(X_val)} samples...")
            predictions = model.predict(X_val)

            pred_df = val_df[["timestamp"]].copy()
            pred_df["signal"] = predictions
            all_predictions.append(pred_df)

    print("\n--- Finalizing and saving results ---")
    if not all_shap_scores:
        print("No SHAP scores were generated. Exiting.")
        return

    final_shap_scores = pd.concat(all_shap_scores)

    # ★ 修正: 将来フォールドの情報を完全に遮断するための厳密なアプローチ
    # 最も直近の学習データ（最後のFold）におけるSHAP値のみを基準に特徴量を選定する。
    last_fold_idx = len(splits)
    last_fold_shap = final_shap_scores[
        final_shap_scores["fold"] == last_fold_idx
    ].copy()
    last_fold_shap = last_fold_shap.sort_values(by="shap_value", ascending=False)

    # 最後のFoldでSHAP値がプラスだった特徴量を最終チームとして採用
    final_feature_team = last_fold_shap[last_fold_shap["shap_value"] > 0][
        "feature"
    ].tolist()

    print(
        f"Selected {len(final_feature_team)} features as the final team based on the LAST training fold."
    )

    pd.Series(final_feature_team).to_csv(
        S3_FINAL_FEATURE_TEAM, index=False, header=False
    )
    print(f"Final feature team saved to: {S3_FINAL_FEATURE_TEAM}")

    # 参考記録として全Foldの平均スコアは保存しておく
    avg_shap_scores = (
        final_shap_scores.groupby("feature")["shap_value"]
        .mean()
        .sort_values(ascending=False)
        .reset_index()
    )
    avg_shap_scores.to_csv(S3_SHAP_SCORES, index=False)
    print(f"Average SHAP scores saved to: {S3_SHAP_SCORES}")

    if all_predictions:
        final_predictions = (
            pd.concat(all_predictions).sort_values("timestamp").reset_index(drop=True)
        )
        signal_path = S3_ARTIFACTS / "primary_model_signals.csv"
        final_predictions.to_csv(signal_path, index=False)
        print(f"Primary model signals saved to: {signal_path}")

    print("Phase 1 completed successfully.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="M1 Walk-Forward Validation Script")
    parser.add_argument(
        "--test-mode",
        action="store_true",
        help="Run in test mode using only the first 2 folds of the walk-forward analysis.",
    )
    args = parser.parse_args()

    warnings.filterwarnings("ignore", category=UserWarning)
    main(args.test_mode)
