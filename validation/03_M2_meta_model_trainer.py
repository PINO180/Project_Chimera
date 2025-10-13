import sys
from pathlib import Path
from datetime import timedelta

# --- プロジェクトのルートディレクトリをPythonの検索パスに追加 ---
sys.path.append(str(Path(__file__).resolve().parents[1]))

import polars as pl
import pandas as pd
import numpy as np
import lightgbm as lgb
import shap
import warnings
import argparse
from typing import List, Dict, Optional
import re
from tqdm import tqdm

# blueprintから一元管理された設定を読み込む
from blueprint import S2_FEATURES_AFTER_AV, S2_FEATURES_FIXED, S3_ARTIFACTS

# --- 特徴量グループの定義 ---
HF_TIMEFRAMES = {"tick", "M0.5", "M1"}

# --- メタモデル設定 ---
TRIPLE_BARRIER_LOOKAHEAD = 60
PROFIT_TAKE_MULTIPLIER = 1.5
STOP_LOSS_MULTIPLIER = 1.0

# --- ヘルパー関数群 (変更なし) ---


def get_unique_identifier_from_path(path: Path) -> Optional[str]:
    name = path.name.replace(".parquet", "")
    parts = name.split("_")
    if len(parts) > 2 and parts[0] == "features":
        return "_".join(parts[1:])
    return None


def classify_paths(base_path: Path) -> Dict[str, List[Path]]:
    print("Discovering and classifying feature files for HF group...")
    hf_sources = []
    engine_dirs = [d for d in base_path.iterdir() if d.is_dir()]
    for engine_dir in engine_dirs:
        for path in engine_dir.iterdir():
            timeframe_match = re.search(
                r"_(tick|M0\.5|M1)$", path.name.replace(".parquet", "")
            )
            if timeframe_match and timeframe_match.group(1) in HF_TIMEFRAMES:
                hf_sources.append(path)
    classified = {"hf": sorted(list(set(hf_sources)))}
    print(f"  - Found {len(classified['hf'])} High-Frequency (HF) sources.")
    return classified


def create_meta_labels(
    signals_df: pd.DataFrame, price_df: pd.DataFrame
) -> pd.DataFrame:
    print("Creating meta-labels...")
    signals_df["timestamp"] = pd.to_datetime(signals_df["timestamp"]).dt.tz_localize(
        None
    )
    price_df["timestamp"] = pd.to_datetime(price_df["timestamp"]).dt.tz_localize(None)
    signals_pl = pl.from_pandas(signals_df.dropna(subset=["timestamp"]))
    price_pl = pl.from_pandas(price_df.dropna(subset=["timestamp"]))
    data = signals_pl.join_asof(
        price_pl, on="timestamp", strategy="backward"
    ).drop_nulls()
    if data.is_empty():
        return pd.DataFrame(columns=["timestamp", "meta_label"])
    data_pd = data.to_pandas().set_index("timestamp")
    price_pd_indexed = price_df.set_index("timestamp")
    meta_labels = []
    for index, row in tqdm(
        data_pd.iterrows(), total=len(data_pd), desc="  - Applying Triple-Barrier"
    ):
        end_time = index + pd.Timedelta(minutes=5 * TRIPLE_BARRIER_LOOKAHEAD)
        future_prices = price_pd_indexed.loc[index:end_time]
        if future_prices.empty:
            meta_labels.append(0)
            continue
        pt_barrier, sl_barrier = (
            row["close"] + row["atr"] * PROFIT_TAKE_MULTIPLIER,
            row["close"] - row["atr"] * STOP_LOSS_MULTIPLIER,
        )
        pt_hits, sl_hits = (
            future_prices[future_prices["high"] >= pt_barrier],
            future_prices[future_prices["low"] <= sl_barrier],
        )
        first_pt_hit, first_sl_hit = (
            (pt_hits.index.min() if not pt_hits.empty else pd.NaT),
            (sl_hits.index.min() if not sl_hits.empty else pd.NaT),
        )
        if pd.notna(first_pt_hit) and (
            pd.isna(first_sl_hit) or first_pt_hit <= first_sl_hit
        ):
            meta_labels.append(1)
        else:
            meta_labels.append(0)
    data_pd["meta_label"] = meta_labels
    result_df = data_pd.reset_index()[["timestamp", "meta_label"]]
    print(f"  - Meta-labels created.")
    return result_df


# --- メインロジック ---


def main(test_mode: bool):
    """メイン実行関数"""
    print("Starting Phase 2: M2 Meta-Model Training for HF Feature Validation...")
    run_id = "train_12m_val_6m"
    run_output_dir = S3_ARTIFACTS / run_id
    run_output_dir.mkdir(parents=True, exist_ok=True)

    signal_path = S3_ARTIFACTS / "primary_model_signals.csv"
    signals_df = pd.read_csv(signal_path, parse_dates=["timestamp"])
    price_path = (
        S2_FEATURES_FIXED / "feature_value_a_vast_universeC/features_e1c_M5.parquet"
    )
    price_df = (
        pl.read_parquet(price_path)
        .select(["timestamp", "open", "high", "low", "close", "atr"])
        .to_pandas()
    )
    meta_labels_df = create_meta_labels(signals_df, price_df)

    if test_mode:
        print("\n--- TEST MODE ACTIVE: Processing only the first 3 signals. ---")
        meta_labels_df = meta_labels_df.head(3)

    classified_paths = classify_paths(S2_FEATURES_AFTER_AV)
    all_hf_paths = classified_paths["hf"]
    tick_hf_paths = [
        p
        for p in all_hf_paths
        if get_unique_identifier_from_path(p)
        and get_unique_identifier_from_path(p).endswith("tick")
    ]
    nontick_hf_paths = [
        p
        for p in all_hf_paths
        if get_unique_identifier_from_path(p)
        and not get_unique_identifier_from_path(p).endswith("tick")
    ]
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

    print("\nProcessing non-tick HF features with join_asof...")
    base_lf = pl.from_pandas(meta_labels_df).lazy()
    nontick_lfs_to_join = []
    for path in nontick_hf_paths:
        unique_id = get_unique_identifier_from_path(path)
        if not unique_id:
            continue
        lf = pl.scan_parquet(path)
        original_cols = lf.collect_schema().names()
        cols_to_select, rename_dict = ["timestamp"], {}
        for col in original_cols:
            if col not in base_cols:
                suffixed_name = f"{col}_{unique_id}"
                cols_to_select.append(col)
                rename_dict[col] = suffixed_name
        if len(cols_to_select) > 1:
            nontick_lfs_to_join.append(lf.select(cols_to_select).rename(rename_dict))
    for lf_to_join in nontick_lfs_to_join:
        base_lf = base_lf.join_asof(lf_to_join, on="timestamp", strategy="backward")
    processed_df = base_lf.collect().to_pandas()

    print("\nProcessing tick HF features with manual cross-day partition lookup...")
    final_rows = []
    for _, row in tqdm(
        processed_df.iterrows(), total=len(processed_df), desc="  - Manual tick lookup"
    ):
        signal_timestamp = row["timestamp"]
        final_row = row.to_dict()

        for tick_path in tick_hf_paths:
            # 調査対象の日付リスト (当日と前日)
            target_date = signal_timestamp.date()
            previous_date = target_date - timedelta(days=1)
            dates_to_check = [target_date, previous_date]

            found_in_this_source = False
            for date_obj in dates_to_check:
                if found_in_this_source:
                    break

                year, month, day = date_obj.year, date_obj.month, date_obj.day
                partition_file = (
                    tick_path
                    / f"year={year}"
                    / f"month={month}"
                    / f"day={day}"
                    / "0.parquet"
                )

                if partition_file.exists():
                    try:
                        daily_tick_df = pl.read_parquet(partition_file).with_columns(
                            pl.col("timestamp").dt.replace_time_zone(None)
                        )

                        tick_feature_row_df = (
                            daily_tick_df.filter(
                                pl.col("timestamp") <= signal_timestamp
                            )
                            .sort("timestamp", descending=True)
                            .head(1)
                        )

                        if not tick_feature_row_df.is_empty():
                            tick_feature_row = tick_feature_row_df.to_dicts()
                            unique_id = get_unique_identifier_from_path(tick_path)
                            for col, val in tick_feature_row[0].items():
                                if col not in base_cols:
                                    final_row[f"{col}_{unique_id}"] = val
                            found_in_this_source = True
                    except Exception as e:
                        print(
                            f"Warning: Failed to process {partition_file}. Error: {e}"
                        )

        final_rows.append(final_row)

    print("\nConcatenating all processed rows...")
    if not final_rows:
        print("No data was processed. Exiting.")
        return

    meta_model_data_df = pd.DataFrame(final_rows).fillna(0)

    hf_feature_cols = [
        col
        for col in meta_model_data_df.columns
        if col not in ["timestamp", "meta_label"]
    ]
    if meta_model_data_df.empty or not hf_feature_cols:
        print("No valid data or HF features to train on after final join. Exiting.")
        return
    print(
        f"Created final dataset with {len(meta_model_data_df)} samples and {len(hf_feature_cols)} features."
    )

    print("Training M2 meta-model and performing SHAP analysis...")
    X, y = meta_model_data_df[hf_feature_cols], meta_model_data_df["meta_label"]
    model = lgb.LGBMClassifier(
        random_state=42, verbosity=-1, n_jobs=-1, is_unbalance=True
    )
    model.fit(X, y)
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)

    if isinstance(shap_values, list):
        shap_values_for_class_1 = shap_values[1]
    else:
        shap_values_for_class_1 = shap_values

    mean_abs_shap = np.abs(shap_values_for_class_1).mean(axis=0)
    shap_importance = pd.DataFrame(
        list(zip(X.columns, mean_abs_shap)), columns=["feature", "shap_value"]
    ).sort_values(by="shap_value", ascending=False)
    survived_hf_features = shap_importance[shap_importance["shap_value"] > 0][
        "feature"
    ].tolist()

    print(f"\n--- Finalizing and saving results ---")
    print(
        f"Selected {len(survived_hf_features)} HF features based on positive SHAP values."
    )
    hf_list_path = run_output_dir / "survived_hf_features.txt"
    pd.Series(survived_hf_features).to_csv(hf_list_path, index=False, header=False)
    print(f"Survived HF feature list saved to: {hf_list_path}")
    hf_shap_path = run_output_dir / "shap_scores_hf.csv"
    shap_importance.to_csv(hf_shap_path, index=False)
    print(f"HF feature SHAP scores saved to: {hf_shap_path}")
    print("Phase 2 completed successfully.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="M2 Meta-Model Trainer (Final v4.0)")
    parser.add_argument(
        "--test-mode",
        action="store_true",
        help="Run in test mode using only the first 3 signals.",
    )
    args = parser.parse_args()
    warnings.filterwarnings("ignore", category=UserWarning)
    from polars.exceptions import PolarsInefficientMapWarning

    warnings.filterwarnings("ignore", category=PolarsInefficientMapWarning)

    main(args.test_mode)
