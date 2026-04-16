import sys
from pathlib import Path

# --- プロジェクトのルートディレクトリをPythonの検索パスに追加 ---
sys.path.append(str(Path(__file__).resolve().parents[1]))

import polars as pl
import numpy as np
import lightgbm as lgb
import shap
import pandas as pd
import re
import warnings
import argparse
from typing import List, Dict, Optional

# blueprintから一元管理された設定を読み込む
from blueprint import S2_FEATURES_AFTER_AV, S3_PRESELECTION, S3_ELITE_LF_FEATURES

# --- 特徴量グループの定義 ---
HF_TIMEFRAMES = {"tick", "M0.5", "M1"}
LF_MICRO_TIMEFRAMES = {"M3", "M5", "M8"}
LF_MESO_TIMEFRAMES = {"M15", "M30", "H1", "H4", "H6"}
LF_MACRO_TIMEFRAMES = {"H12", "D1", "W1", "MN"}

# --- 事前選抜のパラメータ ---
CORRELATION_THRESHOLD = 0.95
SHAP_SAMPLE_SIZE = 50000
TOP_N_FEATURES_PER_TIER = 2000

# --- スクリプト本体 ---


def get_timeframe_from_path(path: Path) -> Optional[str]:
    """ファイルパスから時間枠を抽出する。"""
    if path.is_dir() and "tick" in path.name:
        return "tick"
    if path.suffix == ".parquet":
        match = re.search(r"_([A-Z0-9\.]+)$", path.stem)
        if match:
            return match.group(1)
    return None


def classify_paths(base_path: Path, test_mode: bool = False) -> Dict[str, List[Path]]:
    """ファイルパスをHF, LFの各階層に分類する"""
    print("Discovering and classifying feature files...")
    all_paths = list(base_path.rglob("*.parquet")) + [
        p for p in base_path.iterdir() if p.is_dir()
    ]

    classified = {"hf": [], "lf_micro": [], "lf_meso": [], "lf_macro": []}
    tier_map = {
        "hf": HF_TIMEFRAMES,
        "lf_micro": LF_MICRO_TIMEFRAMES,
        "lf_meso": LF_MESO_TIMEFRAMES,
        "lf_macro": LF_MACRO_TIMEFRAMES,
    }

    path_by_timeframe = {}
    for path in all_paths:
        timeframe = get_timeframe_from_path(path)
        if timeframe:
            if timeframe not in path_by_timeframe:
                path_by_timeframe[timeframe] = []
            path_by_timeframe[timeframe].append(path)

    for tier, timeframes in tier_map.items():
        for tf in sorted(list(timeframes)):
            if tf in path_by_timeframe:
                classified[tier].extend(path_by_timeframe[tf])

    if test_mode:
        print("\n--- TEST MODE ACTIVE ---")
        print("Using only a subset of files for each tier.")
        for tier in classified:
            if len(classified[tier]) > 2:
                classified[tier] = classified[tier][:2]

    print(f"  - Found {len(classified['hf'])} High-Frequency (HF) sources.")
    print(f"  - Found {len(classified['lf_micro'])} LF-Micro sources.")
    print(f"  - Found {len(classified['lf_meso'])} LF-Meso sources.")
    print(f"  - Found {len(classified['lf_macro'])} LF-Macro sources.")
    return classified


# --- 修正点: 根本原因を解決する、最終版の結合ロジック ---
def build_virtual_table_for_tier(file_paths: List[Path]) -> Optional[pl.LazyFrame]:
    """
    「ベースは一度だけ、追加は特徴量のみ」の原則に基づき、
    DuplicateErrorを完全に回避する仮想テーブルを構築する。
    """
    if not file_paths:
        return None

    print(f"Building virtual table from {len(file_paths)} sources...")

    base_cols = {
        "open",
        "high",
        "low",
        "close",
        "volume",
        "timeframe",
        "year",
        "month",
        "day",
    }

    # 1. 最初のファイルをベースとしてロードし、その特徴量にサフィックスを付与
    base_lf_path = file_paths[0]
    base_timeframe = get_timeframe_from_path(base_lf_path)
    base_lf = pl.scan_parquet(base_lf_path)

    base_rename_dict = {
        col: f"{col}_{base_timeframe}"
        for col in base_lf.collect_schema().names()
        if col not in base_cols and col != "timestamp"
    }
    final_lf = base_lf.rename(base_rename_dict)

    # 2. 2つ目以降のファイルをループで結合
    for path in file_paths[1:]:
        timeframe = get_timeframe_from_path(path)
        if not timeframe:
            continue

        lf_to_join = pl.scan_parquet(path)
        original_cols = lf_to_join.collect_schema().names()

        # 結合するカラムを「timestamp」と「サフィックス付き特徴量」のみに厳格に限定
        cols_to_select = ["timestamp"]
        rename_dict = {}
        for col in original_cols:
            if col not in base_cols and col != "timestamp":
                new_name = f"{col}_{timeframe}"
                cols_to_select.append(col)
                rename_dict[col] = new_name

        if len(cols_to_select) > 1:
            lf_processed = lf_to_join.select(cols_to_select).rename(rename_dict)
            final_lf = final_lf.join_asof(
                lf_processed, on="timestamp", strategy="backward"
            )

    return final_lf


def select_features_for_tier(
    tier_name: str, lf_tier: pl.LazyFrame, price_lf: pl.LazyFrame
) -> List[str]:
    """指定された階層の仮想テーブルに対して、冗長性削減とSHAPスクリーニングを行う"""
    print(f"\n--- Processing LF Tier: {tier_name} ---")

    base_cols = {
        "timestamp",
        "open",
        "high",
        "low",
        "close",
        "volume",
        "timeframe",
        "year",
        "month",
        "day",
    }

    feature_cols = sorted(
        [col for col in lf_tier.collect_schema().names() if col not in base_cols]
    )

    if not feature_cols:
        print("No features found in this tier. Skipping.")
        return []

    print(f"Found {len(feature_cols)} total unique features in this tier.")

    print("Preparing sample data for analysis...")
    lf_with_target = lf_tier.join_asof(price_lf, on="timestamp", strategy="backward")

    df_all = lf_with_target.collect(streaming=True).to_pandas()
    df_all = df_all.sort_values("timestamp").reset_index(drop=True)
    df_all.dropna(subset=["target"], inplace=True)
    df_all[feature_cols] = df_all[feature_cols].fillna(0)

    # ★ 修正: ランダムサンプリングを廃止し、時系列前半のみをSHAP学習に使用する。
    # 後半の将来ターゲット値が特徴量選択に影響するlookhead leakを排除。
    split_idx = int(len(df_all) * 0.5)
    df_sample_pd = df_all.iloc[:split_idx].copy()

    if df_sample_pd.shape[0] > SHAP_SAMPLE_SIZE:
        # 時系列を保持したままサンプリング（末尾からではなく等間隔で間引く）
        step = df_sample_pd.shape[0] // SHAP_SAMPLE_SIZE
        df_sample_pd = df_sample_pd.iloc[::step].head(SHAP_SAMPLE_SIZE)

    if df_sample_pd.empty:
        print("Sample is empty. Cannot perform analysis. Skipping.")
        return []

    print(f"Using a sample of {df_sample_pd.shape[0]} rows.")

    print("Step 1: Reducing redundancy based on correlation...")
    corr_matrix = df_sample_pd[feature_cols].corr().abs()
    upper_tri = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    to_drop = {
        column
        for column in upper_tri.columns
        if any(upper_tri[column] > CORRELATION_THRESHOLD)
    }
    features_after_corr = [f for f in feature_cols if f not in to_drop]
    print(
        f"Removed {len(to_drop)} highly correlated features. Remaining: {len(features_after_corr)}"
    )

    if not features_after_corr:
        print("No features remaining after correlation filtering. Skipping.")
        return []

    print("Step 2: Performing lightweight SHAP screening...")
    X_sample = df_sample_pd[features_after_corr]
    y_sample = df_sample_pd["target"]

    model = lgb.LGBMRegressor(random_state=42, verbosity=-1, n_jobs=-1)
    model.fit(X_sample, y_sample)

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_sample)

    mean_abs_shap = np.abs(shap_values).mean(axis=0)
    shap_importance = pd.DataFrame(
        list(zip(X_sample.columns, mean_abs_shap)), columns=["feature", "shap_value"]
    ).sort_values(by="shap_value", ascending=False)

    elite_features = shap_importance.head(TOP_N_FEATURES_PER_TIER)["feature"].tolist()
    print(f"Selected top {len(elite_features)} features based on SHAP values.")

    return elite_features


def main(test_mode: bool):
    """メインの実行関数"""
    print("Starting Phase 0: Low-Frequency Feature Pre-selection...")
    if not S2_FEATURES_AFTER_AV.exists():
        raise FileNotFoundError(f"Base data path not found: {S2_FEATURES_AFTER_AV}")

    S3_PRESELECTION.mkdir(parents=True, exist_ok=True)

    classified_paths = classify_paths(S2_FEATURES_AFTER_AV, test_mode)

    print("\nPreparing a temporary target variable for screening...")
    # 修正: lf_micro から hf へ変更し、_M5 を _M1 に変更
    price_path = next((p for p in classified_paths["hf"] if "_M1" in p.name), None)
    if not price_path:
        raise FileNotFoundError(
            "Could not find M1 price data to generate a target variable."
        )

    print(f"Using {price_path.name} for target generation.")
    price_lf = (
        pl.scan_parquet(price_path)
        .select(["timestamp", "close"])
        .sort("timestamp")
        # M1の shift(-5) なので「5分後の収益率」をターゲットとする
        .with_columns((pl.col("close").shift(-5) / pl.col("close") - 1).alias("target"))
        .select(["timestamp", "target"])
    )

    elite_features_all = []
    lf_tiers = {
        "macro": classified_paths["lf_macro"],
        "meso": classified_paths["lf_meso"],
        "micro": classified_paths["lf_micro"],
    }

    for tier_name, file_paths in lf_tiers.items():
        if not file_paths:
            print(f"\n--- No files for LF Tier: {tier_name}. Skipping. ---")
            continue

        lf_tier_virtual = build_virtual_table_for_tier(file_paths)
        if lf_tier_virtual is None:
            continue

        selected = select_features_for_tier(tier_name, lf_tier_virtual, price_lf)
        if selected:
            elite_features_all.extend(selected)

    final_feature_set = sorted(list(set(elite_features_all)))
    print(f"\n--- Finalizing ---")
    print(f"Total unique elite LF features selected: {len(final_feature_set)}")

    with open(S3_ELITE_LF_FEATURES, "w") as f:
        for feature_name in final_feature_set:
            f.write(f"{feature_name}\n")

    print(f"Elite LF feature list saved to: {S3_ELITE_LF_FEATURES}")
    print("Phase 0 completed successfully.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Low-Frequency Feature Pre-selection Script"
    )
    parser.add_argument(
        "--test-mode",
        action="store_true",
        help="Run in test mode using a small subset of data.",
    )
    args = parser.parse_args()

    warnings.filterwarnings("ignore", category=UserWarning)
    main(args.test_mode)
