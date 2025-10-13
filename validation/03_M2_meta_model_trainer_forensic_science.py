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

from blueprint import S2_FEATURES_AFTER_AV, S2_FEATURES_FIXED, S3_ARTIFACTS

# --- 科学捜査モード設定 ---
ENABLE_FORENSICS = True
FORENSICS_SIGNAL_INDEX = 50
# ---

HF_TIMEFRAMES = {"tick", "M0.5", "M1"}
TRIPLE_BARRIER_LOOKAHEAD = 60
PROFIT_TAKE_MULTIPLIER = 1.5
STOP_LOSS_MULTIPLIER = 1.0


# (get_unique_identifier_from_path, classify_paths, create_meta_labels は変更なし)
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
    signals_pl, price_pl = (
        pl.from_pandas(signals_df.dropna(subset=["timestamp"])),
        pl.from_pandas(price_df.dropna(subset=["timestamp"])),
    )
    data = signals_pl.join_asof(
        price_pl, on="timestamp", strategy="backward"
    ).drop_nulls()
    if data.is_empty():
        return pd.DataFrame(columns=["timestamp", "meta_label"])
    data_pd, price_pd_indexed = (
        data.to_pandas().set_index("timestamp"),
        price_df.set_index("timestamp"),
    )
    meta_labels = []
    # tqdmはログが長くなるため鑑識中は非表示
    iterable = data_pd.iterrows()
    if not ENABLE_FORENSICS:
        iterable = tqdm(
            data_pd.iterrows(), total=len(data_pd), desc="  - Applying Triple-Barrier"
        )

    for index, row in iterable:
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


def main(test_mode: bool):
    if ENABLE_FORENSICS:
        print(
            "=" * 50 + "\n### FORENSICS MODE: CROSS-DAY INVESTIGATION ###\n" + "=" * 50
        )
        test_mode = True

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
        if ENABLE_FORENSICS:
            meta_labels_df = meta_labels_df.iloc[[FORENSICS_SIGNAL_INDEX]]
            print(
                f"\n--- Investigating signal index {FORENSICS_SIGNAL_INDEX} only. ---"
            )
        else:
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

    print("\nProcessing tick HF features with manual partition lookup...")
    final_rows = []

    iterable = processed_df.iterrows()
    if not ENABLE_FORENSICS:
        iterable = tqdm(
            processed_df.iterrows(),
            total=len(processed_df),
            desc="  - Manual tick lookup",
        )

    for _, row in iterable:
        signal_timestamp = row["timestamp"]
        final_row = row.to_dict()

        if ENABLE_FORENSICS:
            print("\n" + "-" * 20 + " FORENSICS LOG START " + "-" * 20)
            print(f"Target Signal Timestamp: {signal_timestamp}")

        for tick_path in tick_hf_paths:
            if ENABLE_FORENSICS:
                print(f"\n[Investigating Tick Source]: {tick_path.name}")

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

                if ENABLE_FORENSICS:
                    print(f"  - Checking file for date {date_obj}: {partition_file}")

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
                            if ENABLE_FORENSICS:
                                print(
                                    f"    -> SUCCESS: Data found in file for {date_obj}!"
                                )
                                print(tick_feature_row_df)

                            tick_feature_row = tick_feature_row_df.to_dicts()
                            unique_id = get_unique_identifier_from_path(tick_path)
                            for col, val in tick_feature_row[0].items():
                                if col not in base_cols:
                                    final_row[f"{col}_{unique_id}"] = val
                            found_in_this_source = True
                        elif ENABLE_FORENSICS:
                            print(
                                f"    -> No data found for this timestamp in file for {date_obj}."
                            )

                    except Exception as e:
                        if ENABLE_FORENSICS:
                            print(f"    -> ERROR: Failed to process file. Error: {e}")
                elif ENABLE_FORENSICS:
                    print(f"    -> File for {date_obj} not found.")

        final_rows.append(final_row)
        if ENABLE_FORENSICS:
            print("-" * (42 + len(" FORENSICS LOG START ")))

    if ENABLE_FORENSICS:
        print("\nForensics run completed. Please check the detailed log above.")
        return

    # (以降のモデル学習部分は変更なし)
    print("\nConcatenating all processed rows...")
    if not final_rows:
        print("No data was processed. Exiting.")
        return
    # ... (以降のコードは変更なし) ...


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="M2 Meta-Model Trainer (Forensics v2)")
    # --- ここを修正：引数の定義を復活 ---
    parser.add_argument(
        "--test-mode",
        action="store_true",
        help="Run in test mode using only the first 3 signals (ignored if Forensics mode is ON).",
    )
    args = parser.parse_args()
    warnings.filterwarnings("ignore", category=UserWarning)
    from polars.exceptions import PolarsInefficientMapWarning

    warnings.filterwarnings("ignore", category=PolarsInefficientMapWarning)

    main(args.test_mode)
