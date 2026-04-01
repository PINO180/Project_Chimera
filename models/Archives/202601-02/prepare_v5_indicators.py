# /workspace/models/prepare_v5_indicators_from_monolithic.py

import sys
from pathlib import Path
import logging
import argparse
import polars as pl
from tqdm import tqdm

# --- Project Path Setup ---
project_root = Path(__file__).resolve().parents[1]
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

from blueprint import DATA_DIR, SYMBOL

# --- Configuration ---
# 保存先
OUTPUT_DIR = DATA_DIR / SYMBOL / "stratum_2_features_fixed" / "v5_gatekeeper_ready"

# 入力元ルート
INPUT_ROOT = DATA_DIR / SYMBOL / "stratum_2_features_fixed"

# 抽出対象カラム
TARGET_FEATURES = {
    "A": ["e1a_statistical_skewness_20", "e1a_statistical_kurtosis_20"],
    "E": ["e1e_hilbert_amplitude_50", "e1e_hilbert_phase_stability_50"],
    "C": ["open", "high", "low", "close"],
}

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)


def find_file_for_timeframe(root_dir: Path, universe_key: str, timeframe: str) -> Path:
    """指定されたUniverseと時間足のファイルを探す"""
    patterns = []
    if universe_key == "C":
        patterns = ["*universeC*", "*Price*"]
    elif universe_key == "A":
        patterns = ["*universeA*", "*Stat*"]
    elif universe_key == "E":
        patterns = ["*universeE*", "*Hilbert*"]

    target_dir = None
    for p in patterns:
        candidates = list(root_dir.glob(p))
        if candidates:
            target_dir = candidates[0]
            break

    if not target_dir:
        return None

    file_patterns = [f"*{timeframe}.parquet", f"*{timeframe.lower()}.parquet"]
    for fp in file_patterns:
        files = list(target_dir.glob(fp))
        if files:
            return files[0]

    return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--timeframe", type=str, default="M5", help="Target timeframe (e.g., M5, M15)"
    )
    args = parser.parse_args()

    tf = args.timeframe
    logging.info(f"### Starting V5 ETL (Monolithic -> Hive) for {tf} [UTC FIX] ###")
    logging.info(f"Input Root: {INPUT_ROOT}")
    logging.info(f"Output:     {OUTPUT_DIR}")

    # 1. ファイル探索
    path_c = find_file_for_timeframe(INPUT_ROOT, "C", tf)
    path_a = find_file_for_timeframe(INPUT_ROOT, "A", tf)
    path_e = find_file_for_timeframe(INPUT_ROOT, "E", tf)

    if not (path_c and path_a and path_e):
        logging.error(
            "CRITICAL: Could not find parquet files for one or more universes."
        )
        sys.exit(1)

    logging.info(f"  Found Price:   {path_c.name}")
    logging.info(f"  Found Stats:   {path_a.name}")
    logging.info(f"  Found Hilbert: {path_e.name}")

    # 2. 読み込み & メモリ上での結合
    logging.info("Loading data into memory...")

    def load_lf(path, cols):
        return pl.scan_parquet(path).select(["timestamp"] + cols)

    lf_c = load_lf(path_c, TARGET_FEATURES["C"])
    lf_a = load_lf(path_a, TARGET_FEATURES["A"])
    lf_e = load_lf(path_e, TARGET_FEATURES["E"])

    # 結合 (Inner Join)
    lf_merged = lf_c.join(lf_a, on="timestamp", how="inner").join(
        lf_e, on="timestamp", how="inner"
    )

    # 3. 前処理 & 指標計算
    logging.info("Processing features and normalizing timestamps to UTC...")
    lf_processed = lf_merged.with_columns(
        [
            # ★★★ 修正: UTCタイムゾーンを明示的に付与 ★★★
            # 元データにタイムゾーンが付いていれば削除してからUTCを付ける（誤変換防止）
            pl.col("timestamp")
            .cast(pl.Datetime("us"))
            .dt.replace_time_zone(None)  # 一度Naiveにする
            .dt.replace_time_zone("UTC"),  # 明示的にUTCスタンプを押す
            # V5 Body Ratio
            (
                (pl.col("close") - pl.col("open")).abs()
                / ((pl.col("high") - pl.col("low")) + 1e-9)
            ).alias("v5_body_ratio"),
            # パーティション用カラム生成
            pl.col("timestamp").dt.year().alias("year"),
            pl.col("timestamp").dt.month().alias("month"),
            pl.col("timestamp").dt.day().alias("day"),
        ]
    ).drop(["open", "high", "low", "close"])

    # 4. パーティション書き出し
    logging.info("Writing Hive-partitioned dataset (UTCized)...")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    try:
        lf_processed.sink_parquet(
            OUTPUT_DIR, hive_partitioning=True, compression="zstd"
        )
        logging.info("SUCCESS: Data successfully partitioned and saved.")

    except Exception as e:
        logging.error(f"Error during sink_parquet: {e}")
        logging.info("Falling back to manual partition write...")

        df = lf_processed.collect()
        years = df["year"].unique().to_list()

        for y in tqdm(years, desc="Writing Years"):
            df_y = df.filter(pl.col("year") == y)
            months = df_y["month"].unique().to_list()
            for m in months:
                df_m = df_y.filter(pl.col("month") == m)
                days = df_m["day"].unique().to_list()
                for d in days:
                    part_dir = OUTPUT_DIR / f"year={y}" / f"month={m}" / f"day={d}"
                    part_dir.mkdir(parents=True, exist_ok=True)
                    df_m.filter(pl.col("day") == d).write_parquet(
                        part_dir / "v5_indicators.parquet"
                    )


if __name__ == "__main__":
    main()
