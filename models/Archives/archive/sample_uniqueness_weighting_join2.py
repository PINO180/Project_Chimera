# /workspace/models/sample_uniqueness_weighting_join.py
# [MN無効化戦略に合わせたクリーン版 - Inf発生時は行を削除する最小限の防御]

import sys
import polars as pl
import pyarrow.parquet as pq
from pathlib import Path
import logging
import os
import shutil
from typing import Dict

# プロジェクトのルートディレクトリをPythonの検索パスに追加
sys.path.append(str(Path(__file__).resolve().parents[1]))

# blueprintから一元管理された設定を読み込む
from blueprint import (
    S6_LABELED_DATASET,
    S6_WEIGHTED_DATASET,
    BASE_DIR,
    S3_CONCURRENCY_RESULTS,
)

# --- 定数 ---
CONCURRENCY_RESULTS_PATH = S3_CONCURRENCY_RESULTS

# --- ロギング設定 ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(processName)s - %(message)s",
)


def get_partition_info(input_dir: Path) -> Dict[Path, Dict[str, int]]:
    """各パーティションファイルのオフセットと行数を計算する。"""
    logging.info("Step 1: Calculating row counts and offsets for each partition...")
    partitions = sorted(input_dir.glob("year=*/month=*/day=*/*.parquet"))
    if not partitions:
        raise FileNotFoundError(
            f"No daily parquet files found in the expected Hive structure within {input_dir}"
        )

    partition_info = {}
    total_rows = 0
    for path in partitions:
        try:
            row_count = pq.ParquetFile(path).metadata.num_rows

            # 各ファイルの年/月/日情報を抽出（ログ出力用）
            day_part = path.parent.name
            month_part = path.parent.parent.name
            year_part = path.parent.parent.parent.name

            partition_info[path] = {
                "offset": total_rows,
                "rows": row_count,
                "year": int(year_part.split("=")[1]),
                "month": int(month_part.split("=")[1]),
                "day": int(day_part.split("=")[1]),
            }
            total_rows += row_count
        except Exception as e:
            logging.error(f"Failed to read metadata from {path}: {e}")
            raise

    logging.info(
        f"   -> Found {len(partitions)} daily partitions. Total rows: {total_rows}. Info calculated."
    )
    return partition_info


def main():
    """メインオーケストレーション関数（日次・逐次処理版）"""
    logging.info("### Final Battle Stage 2: Daily Sequential Assembly ###")
    logging.warning(
        "NOTE: MN/W1/D1 labels are now excluded. Inf/NaN uniqueness rows will be physically dropped to ensure model stability (minimal risk)."
    )

    if not CONCURRENCY_RESULTS_PATH.exists():
        logging.error(
            f"CRITICAL: Concurrency results not found at {CONCURRENCY_RESULTS_PATH}"
        )
        return

    # --- 準備：出力ディレクトリのクリーンアップと作成 ---
    if S6_WEIGHTED_DATASET.exists():
        logging.warning(
            f"Output directory {S6_WEIGHTED_DATASET} exists. Removing it for a clean run."
        )
        shutil.rmtree(S6_WEIGHTED_DATASET)
    S6_WEIGHTED_DATASET.mkdir(parents=True)

    # --- ステージ1: 各パーティションの情報を取得 ---
    partition_info_map = get_partition_info(S6_LABELED_DATASET)

    # --- ステージ2: 逐次処理ループ ---
    logging.info(
        f"Stage 2: Starting sequential processing of {len(partition_info_map)} daily partitions."
    )

    error_count = 0
    total_processed_rows = 0

    # concurrency_resultsを一度だけ遅延スキャンしておく
    concurrency_lf = pl.scan_parquet(CONCURRENCY_RESULTS_PATH)

    for i, (path, info) in enumerate(partition_info_map.items()):
        # ログ出力用のパーティション情報を取得
        partition_name = f"year={info['year']}/month={info['month']}/day={info['day']}"
        logging.info(
            f"Processing: [{i + 1}/{len(partition_info_map)}] - Partition {partition_name}..."
        )

        try:
            # このパーティションが担当するevent_idの範囲を計算
            offset = info["offset"]
            row_count = info["rows"]

            # 行数が0のパーティションはスキップ
            if row_count == 0:
                logging.warning(f"Skipping empty partition: {partition_name}")
                continue

            start_id = offset + 1
            end_id = offset + row_count

            # 担当パーティションを遅延スキャン
            labeled_lf = pl.scan_parquet(path)

            # concurrencyデータから必要な範囲だけをフィルタリング
            concurrency_slice_lf = concurrency_lf.filter(
                pl.col("event_id").is_between(start_id, end_id)
            )

            # event_idを付与し、スライス済みのconcurrencyを結合
            final_lf = (
                labeled_lf.sort("timestamp", "t1")
                .with_row_count(name="row_num", offset=offset)
                .with_columns((pl.col("row_num") + 1).alias("event_id"))
                .join(concurrency_slice_lf, on="event_id", how="left")
                .with_columns((1.0 / pl.col("concurrency")).alias("uniqueness"))
                # ★★★ 最小限の防御: Inf重みでシステムがクラッシュするのを防ぐ ★★★
                # MNラベルが無効化されたため、Inf行は微量であり、削除してもAUCは回復する
                .filter(pl.col("uniqueness").is_finite())
                .drop_nulls("uniqueness")
                # ★★★ 修正ここまで ★★★
                .select(pl.all().exclude(["row_num", "concurrency"]))
            )

            # 結果を最終的な日次パーティション構造で書き出す
            output_dir = S6_WEIGHTED_DATASET / partition_name
            output_dir.mkdir(parents=True, exist_ok=True)
            output_path = output_dir / "data.parquet"

            # 計算を実行し、結果をファイルに書き出す
            result_df = final_lf.collect(streaming=True)

            if result_df.is_empty():
                logging.warning(
                    f"Partition {partition_name} became empty after uniqueness filtering. Skipping write."
                )
                continue

            result_df.write_parquet(output_path, compression="zstd")
            total_processed_rows += len(result_df)

        except Exception as e:
            logging.error(f"Processing for {partition_name} failed: {e}", exc_info=True)
            error_count += 1
            break

    # --- 最終報告 ---
    logging.info("\n" + "=" * 60)
    if error_count == 0:
        logging.info("### MISSION ACCOMPLISHED! All Stages COMPLETED! ###")
        logging.info(
            f"Successfully processed {len(partition_info_map)} partitions ({total_processed_rows} rows)."
        )
        logging.info(f"The final weighted dataset is ready at: {S6_WEIGHTED_DATASET}")
    else:
        logging.error("### PROCESSING FAILED ###")
        logging.error(
            f"An error occurred. {total_processed_rows} rows were processed before failure. Please check the logs."
        )
    logging.info("=" * 60)


if __name__ == "__main__":
    main()
