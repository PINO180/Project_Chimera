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
from typing import List, Dict, Optional
import re

# blueprintから一元管理された設定を読み込む
from blueprint import S2_FEATURES_AFTER_AV, S2_FEATURES_FIXED, S3_ARTIFACTS

# --- 特徴量グループの定義 ---
HF_TIMEFRAMES = {"tick", "M0.5", "M1"}

# --- メタモデル設定 ---
TRIPLE_BARRIER_LOOKAHEAD = 60
PROFIT_TAKE_MULTIPLIER = 1.5
STOP_LOSS_MULTIPLIER = 1.0


def get_timeframe_from_path(path: Path) -> Optional[str]:
    """ファイルパスからタイムフレームを抽出する、改善版"""
    name = path.name
    if path.is_dir():
        match = re.search(r"_([a-zA-Z0-9\.]+)$", name)
        if match:
            return match.group(1)
    elif path.suffix == ".parquet":
        match = re.search(r"_([a-zA-Z0-9\.]+)$", path.stem)
        if match:
            return match.group(1)
    return None


def classify_paths(base_path: Path) -> Dict[str, List[Path]]:
    """ファイルパスをHF（高頻度）に分類する"""
    print("Discovering and classifying feature files for HF group...")
    all_paths = list(base_path.rglob("*.parquet")) + [
        p for p in base_path.iterdir() if p.is_dir()
    ]

    unique_sources = set()
    for p in all_paths:
        if p.is_dir() and "tick" in p.name:
            unique_sources.add(p)
        elif p.suffix == ".parquet":
            unique_sources.add(p)

    classified = {"hf": []}
    for path in unique_sources:
        timeframe = get_timeframe_from_path(path)
        if timeframe in HF_TIMEFRAMES:
            classified["hf"].append(path)

    classified["hf"] = sorted(list(set(classified["hf"])))
    print(f"  - Found {len(classified['hf'])} High-Frequency (HF) sources.")
    return classified


def create_meta_labels(
    signals_df: pd.DataFrame, price_df: pd.DataFrame
) -> pd.DataFrame:
    """M1のシグナルと価格データから、トリプルバリア法を用いてメタラベルを生成する。"""
    print("Creating meta-labels using Triple-Barrier Method...")
    signals_pl = pl.from_pandas(signals_df.dropna(subset=["timestamp"]))
    price_pl = pl.from_pandas(price_df.dropna(subset=["timestamp"]))
    data = signals_pl.join_asof(
        price_pl, on="timestamp", strategy="backward"
    ).drop_nulls()

    if data.is_empty():
        print("Warning: No data remains after joining signals with price data.")
        return pd.DataFrame(columns=["timestamp", "meta_label"])

    data_pd = data.to_pandas().set_index("timestamp")
    price_pd_indexed = price_df.set_index("timestamp")

    meta_labels = []
    for index, row in data_pd.iterrows():
        end_time = index + pd.Timedelta(minutes=5 * TRIPLE_BARRIER_LOOKAHEAD)

        future_prices = price_pd_indexed.loc[index:end_time]
        if future_prices.empty:
            meta_labels.append(0)
            continue

        pt_barrier = row["close"] + row["atr"] * PROFIT_TAKE_MULTIPLIER
        sl_barrier = row["close"] - row["atr"] * STOP_LOSS_MULTIPLIER

        pt_hits = future_prices[future_prices["high"] >= pt_barrier]
        sl_hits = future_prices[future_prices["low"] <= sl_barrier]

        first_pt_hit = pt_hits.index.min() if not pt_hits.empty else pd.NaT
        first_sl_hit = sl_hits.index.min() if not sl_hits.empty else pd.NaT

        if pd.notna(first_pt_hit) and (
            pd.isna(first_sl_hit) or first_pt_hit <= first_sl_hit
        ):
            meta_labels.append(1)
        else:
            meta_labels.append(0)

    data_pd["meta_label"] = meta_labels
    result_df = data_pd.reset_index()[["timestamp", "meta_label"]]

    print(f"  - Generated {result_df['meta_label'].sum()} successful labels.")
    print(f"  - Generated {(result_df['meta_label'] == 0).sum()} failed labels.")

    return result_df


def main():
    """メイン実行関数"""
    print("Starting Phase 2: M2 Meta-Model Training for HF Feature Validation...")

    run_id = "train_12m_val_6m"
    run_output_dir = S3_ARTIFACTS / run_id
    run_output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Results for this run will be saved in: {run_output_dir}")

    signal_path = S3_ARTIFACTS / "primary_model_signals.csv"

    if not signal_path.exists():
        raise FileNotFoundError(
            f"Primary model signals not found at {signal_path}. Please run 02_M1... script first."
        )
    print(f"Loading primary signals from: {signal_path}")
    signals_df = pd.read_csv(signal_path, parse_dates=["timestamp"])

    price_path = (
        S2_FEATURES_FIXED / "feature_value_a_vast_universeC/features_e1c_M5.parquet"
    )
    if not price_path.exists():
        raise FileNotFoundError(
            f"M5 price/ATR data source not found in stratum_2_features_fixed at {price_path}"
        )

    print(f"Loading base price and ATR data from: {price_path}")
    price_df = (
        pl.read_parquet(price_path)
        .select(["timestamp", "open", "high", "low", "close", "atr"])
        .to_pandas()
    )

    meta_labels_df = create_meta_labels(signals_df, price_df)

    classified_paths = classify_paths(S2_FEATURES_AFTER_AV)
    hf_paths = classified_paths["hf"]

    if not hf_paths:
        print("No High-Frequency features found. Exiting.")
        return

    print("Loading and joining all High-Frequency features with dynamic suffixing...")
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

    processed_lfs = []
    for path in hf_paths:
        timeframe = get_timeframe_from_path(path)
        if not timeframe:
            continue

        scan_path = path / "**/*.parquet" if path.is_dir() else path
        lf = pl.scan_parquet(scan_path)

        original_cols = lf.collect_schema().names()

        rename_dict = {
            col: f"{col}_{timeframe}"
            for col in original_cols
            if col not in base_cols and col != "timestamp"
        }

        lf_renamed = lf.select(["timestamp"] + list(rename_dict.keys())).rename(
            rename_dict
        )
        processed_lfs.append(lf_renamed)

    if not processed_lfs:
        print("No processable HF features found. Exiting.")
        return

    all_timestamps_lf = (
        pl.concat([lf.select("timestamp") for lf in processed_lfs], how="diagonal")
        .unique()
        .sort("timestamp")
    )

    final_hf_lf = all_timestamps_lf
    for lf in processed_lfs:
        final_hf_lf = final_hf_lf.join_asof(lf, on="timestamp", strategy="backward")

    meta_model_data_pl = (
        pl.from_pandas(meta_labels_df)
        .lazy()
        .join_asof(final_hf_lf, on="timestamp", strategy="backward")
    )

    meta_model_data_df = meta_model_data_pl.collect().to_pandas().dropna()

    hf_feature_cols = [
        col
        for col in meta_model_data_df.columns
        if col not in ["timestamp", "meta_label"]
    ]

    if not hf_feature_cols:
        print("No valid HF features to train on after join and dropna. Exiting.")
        return

    print(
        f"Created dataset for meta-model with {len(meta_model_data_df)} samples and {len(hf_feature_cols)} HF features."
    )

    print("Training M2 meta-model and performing SHAP analysis...")
    X = meta_model_data_df[hf_feature_cols]
    y = meta_model_data_df["meta_label"]

    model = lgb.LGBMClassifier(
        random_state=42, verbosity=-1, n_jobs=-1, is_unbalance=True
    )
    model.fit(X, y)

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)

    # --- 修正点: SHAPの出力形式に柔軟に対応する ---
    # shap_valuesがリスト（2クラス分の出力）か、単一のNumpy配列（1クラス分の出力）かを判定
    if isinstance(shap_values, list):
        # 従来通り、クラス1（成功）のSHAP値を取得
        shap_values_for_class_1 = shap_values[1]
    else:
        # 単一の配列で返された場合は、それをそのまま使用
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
    warnings.filterwarnings("ignore", category=UserWarning)
    from polars.exceptions import PolarsInefficientMapWarning

    warnings.filterwarnings("ignore", category=PolarsInefficientMapWarning)
    main()
