import sys
from pathlib import Path

# プロジェクトのルートディレクトリをPythonの検索パスに追加
sys.path.append(str(Path(__file__).resolve().parents[1]))

import polars as pl
import argparse
import warnings
import re
from tqdm import tqdm
import numpy as np
import shutil

# blueprintから一元管理された設定を読み込む
from blueprint import (
    S2_FEATURES_AFTER_AV,
    S3_FEATURES_FOR_ALPHA_DECAY,
    S5_ALPHA,
    S5_NEUTRALIZED_ALPHA_SET,
)

# --- スクリプト設定 ---
MARKET_PROXY_TIMEFRAME = "M5"
MARKET_PROXY_LOOKBACK = 5


# --- ヘルパー関数群 (変更なし) ---
def parse_suffixed_feature_name(suffixed_name: str) -> tuple[str, str, str]:
    parts = suffixed_name.split("_")
    timeframe_pattern = re.compile(r"^(M[0-9\.]+|H[0-9]+|D[0-9]+|W[0-9]+|MN|tick)$")
    if (
        parts[0].startswith("e")
        and len(parts) > 2
        and timeframe_pattern.match(parts[-1])
    ):
        engine_id = parts[0]
        base_name = "_".join(parts[:-1])
        timeframe = parts[-1]
        return engine_id, base_name, timeframe
    if len(parts) > 1 and timeframe_pattern.match(parts[-1]):
        base_name = "_".join(parts[:-1])
        timeframe = parts[-1]
        return "base", base_name, timeframe
    return suffixed_name, suffixed_name, "unknown"


def find_features_by_file(final_features: list[str]) -> dict[Path, list[str]]:
    print("Mapping final features to their source files with high precision...")
    features_by_file = {}
    all_sources_with_schema = {}
    all_sources = list(S2_FEATURES_AFTER_AV.rglob("*"))
    for source_path in tqdm(all_sources, desc="Caching all source schemas"):
        try:
            schema = set()
            if source_path.is_file() and source_path.name.endswith(".parquet"):
                schema = set(pl.read_parquet_schema(source_path).keys())
            elif source_path.is_dir() and "tick" in source_path.name:
                first_parquet = next(source_path.rglob("*.parquet"), None)
                if first_parquet:
                    schema = set(pl.read_parquet_schema(first_parquet).keys())
            if schema:
                all_sources_with_schema[source_path] = schema
        except Exception:
            continue
    print(f"Searching {len(final_features)} features...")
    for suffixed_feature in tqdm(final_features, desc="Mapping features"):
        engine_id, base_name, timeframe = parse_suffixed_feature_name(suffixed_feature)
        if timeframe == "unknown":
            print(
                f"Warning: Could not determine timeframe for '{suffixed_feature}'. Skipping."
            )
            continue
        found_path = None
        for source_path, schema in all_sources_with_schema.items():
            path_name = source_path.name
            if base_name not in schema:
                continue
            if timeframe not in path_name:
                continue
            if engine_id != "base" and engine_id not in path_name:
                continue
            found_path = source_path
            break
        if found_path:
            if found_path not in features_by_file:
                features_by_file[found_path] = []
            features_by_file[found_path].append(suffixed_feature)
        else:
            print(
                f"Warning: Could not find a precise source file for '{suffixed_feature}'"
            )
    print(
        f"Grouped {len(final_features)} features into {len(features_by_file)} unique source files."
    )
    return features_by_file


def get_output_path(input_path: Path) -> Path:
    relative_path = input_path.relative_to(S2_FEATURES_AFTER_AV)
    output_name = (
        input_path.name.replace(".parquet", "_neutralized.parquet")
        if input_path.is_file()
        else f"{input_path.name}_neutralized"
    )
    return S5_NEUTRALIZED_ALPHA_SET / relative_path.parent / output_name


def neutralize_lazyframe(
    lf: pl.LazyFrame,
    feature_names: list[str],
    market_proxy_col: str,
    rolling_window: int = 2016,  # 約1週間分(M5×2016=約7日)
) -> pl.LazyFrame:
    """ローリング共分散の数学的展開を用いた正しい純化処理"""
    neutralization_exprs = []

    # プロキシ(X)の準備と、Xのローリング分散( E[X^2] - E[X]^2 )
    x = pl.col(market_proxy_col).fill_null(0)
    x_mean = x.rolling_mean(window_size=rolling_window, min_periods=30)
    var_x = (x * x).rolling_mean(window_size=rolling_window, min_periods=30) - (
        x_mean * x_mean
    )

    for suffixed_feature in feature_names:
        _, base_name, _ = parse_suffixed_feature_name(suffixed_feature)
        y = pl.col(base_name).fill_null(0)
        y_mean = y.rolling_mean(window_size=rolling_window, min_periods=30)

        # XとYのローリング共分散( E[XY] - E[X]E[Y] )
        cov_xy = (x * y).rolling_mean(window_size=rolling_window, min_periods=30) - (
            x_mean * y_mean
        )

        # ベータとアルファ
        rolling_beta = cov_xy / (var_x + 1e-10)
        rolling_alpha = y_mean - rolling_beta * x_mean

        # 残差計算
        residual_expr = (
            y - (rolling_beta.fill_null(0) * x + rolling_alpha.fill_null(0))
        ).alias(f"{base_name}_neutralized")
        neutralization_exprs.append(residual_expr)

    return lf.with_columns(neutralization_exprs)


def main(test_mode: bool):
    print("=" * 60)
    print("### Stage 3: Alpha Neutralization (The Grand Finale) ###")
    print("=" * 60)

    S5_ALPHA.mkdir(parents=True, exist_ok=True)
    if S5_NEUTRALIZED_ALPHA_SET.exists():
        print(
            f"Output directory {S5_NEUTRALIZED_ALPHA_SET} already exists. Removing it."
        )
        shutil.rmtree(S5_NEUTRALIZED_ALPHA_SET)
    S5_NEUTRALIZED_ALPHA_SET.mkdir(parents=True, exist_ok=True)

    with open(S3_FEATURES_FOR_ALPHA_DECAY, "r") as f:
        final_features = [line.strip() for line in f if line.strip()]

    features_by_source_file = find_features_by_file(final_features)

    print(f"\nPreparing market proxy using {MARKET_PROXY_TIMEFRAME} data...")
    proxy_source_path = (
        S2_FEATURES_AFTER_AV
        / "feature_value_a_vast_universeA"
        / f"features_e1a_{MARKET_PROXY_TIMEFRAME}.parquet"
    )
    market_proxy_lf = (
        pl.scan_parquet(proxy_source_path)
        .select(["timestamp", "close"])
        .sort("timestamp")
        .with_columns(
            # 修正: shift(-5) を廃止し、過去への参照（正の数）に変更
            (pl.col("close") / pl.col("close").shift(MARKET_PROXY_LOOKBACK) - 1).alias(
                "market_proxy"
            )
        )
        .select(["timestamp", "market_proxy"])
    )

    source_files_to_process = list(features_by_source_file.keys())
    if test_mode:
        print(f"\n--- TEST MODE ACTIVE: Processing only the first 2 source files. ---")
        source_files_to_process = source_files_to_process[:2]

    print(
        f"\nStarting neutralization process for {len(source_files_to_process)} source files..."
    )
    for i, (input_path, features_in_file) in enumerate(features_by_source_file.items()):
        if input_path not in source_files_to_process:
            continue
        print(
            f"\n--- Processing file {i + 1}/{len(source_files_to_process)}: {input_path.name} ---"
        )
        base_feature_names = list(
            set([parse_suffixed_feature_name(f)[1] for f in features_in_file])
        )
        lf = (
            pl.scan_parquet(input_path)
            .select(["timestamp"] + base_feature_names)
            .sort("timestamp")
        )
        lf_with_proxy = lf.join_asof(
            market_proxy_lf, on="timestamp", strategy="backward"
        ).drop_nulls("market_proxy")
        neutralized_lf = neutralize_lazyframe(
            lf_with_proxy, features_in_file, "market_proxy"
        )
        output_cols = ["timestamp"] + [
            f"{parse_suffixed_feature_name(f)[1]}_neutralized" for f in features_in_file
        ]
        final_lf_to_save = neutralized_lf.select(output_cols)
        output_path = get_output_path(input_path)
        print(f"Saving to: {output_path}")

        if input_path.is_dir():
            print("Collecting data for manual partitioning...")
            df_to_save = final_lf_to_save.collect(streaming=True)
            print("Grouping data by date...")
            grouped = df_to_save.group_by(
                pl.col("timestamp").dt.year().alias("year"),
                pl.col("timestamp").dt.month().alias("month"),
                pl.col("timestamp").dt.day().alias("day"),
                maintain_order=True,
            )
            output_path.mkdir(parents=True, exist_ok=True)
            print("Writing partitions manually...")
            for (year, month, day), data_group in tqdm(
                grouped, desc="Writing partitions"
            ):
                partition_dir = output_path / f"year={year}/month={month}/day={day}"
                partition_dir.mkdir(parents=True, exist_ok=True)

                # --- ★★★ これが最後の、最後の、最後の修正 ★★★ ---
                # `data_group`は既に純粋なデータなので、そのまま書き込む
                data_group.write_parquet(partition_dir / "0.parquet", use_pyarrow=True)
        else:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            final_lf_to_save.sink_parquet(output_path)

    print("\n" + "=" * 60)
    print("### Third Defense Line CLEARED! ###")
    print(f"Individually neutralized alpha files are ready for Chapter 3.")
    print("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Stage 3: Feature Neutralization (The Grand Finale)"
    )
    parser.add_argument("--test-mode", action="store_true", help="Run in test mode.")
    args = parser.parse_args()
    warnings.filterwarnings("ignore", category=UserWarning)
    try:
        from polars.exceptions import PolarsInefficientMapWarning

        warnings.filterwarnings("ignore", category=PolarsInefficientMapWarning)
    except ImportError:
        pass
    main(args.test_mode)
