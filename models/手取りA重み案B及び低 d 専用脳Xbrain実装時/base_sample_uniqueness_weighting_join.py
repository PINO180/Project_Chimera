# /workspace/models/sample_uniqueness_weighting_join.py

import sys
import polars as pl
from pathlib import Path
import logging
import shutil
from typing import List

# プロジェクトのルートディレクトリをPythonの検索パスに追加
sys.path.append(str(Path(__file__).resolve().parents[1]))

# blueprintから一元管理された設定を読み込む
from blueprint import (
    S6_LABELED_DATASET,
    S6_WEIGHTED_DATASET,
    S3_CONCURRENCY_RESULTS,
)

# --- 定数 ---
CONCURRENCY_RESULTS_PATH = S3_CONCURRENCY_RESULTS

# --- ロギング設定 ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(processName)s - %(message)s",
)


def get_partitions(input_dir: Path) -> List[Path]:
    """日次パーティションのファイルパス一覧を取得する。"""
    logging.info("Step 1: Finding daily partitions...")
    partitions = sorted(input_dir.glob("year=*/month=*/day=*/*.parquet"))
    if not partitions:
        raise FileNotFoundError(
            f"No daily parquet files found in the expected Hive structure within {input_dir}"
        )

    logging.info(f"   -> Found {len(partitions)} daily partitions.")
    return partitions


def main():
    """メインオーケストレーション関数（日次・逐次処理版・V5仕様）"""
    logging.info("### Final Battle Stage 2: Daily Sequential Assembly (V5) ###")

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

    # --- ステージ1: パーティション情報の取得 ---
    partitions = get_partitions(S6_LABELED_DATASET)

    # --- ステージ2: 逐次処理ループ ---
    logging.info(
        f"Stage 2: Starting sequential processing of {len(partitions)} daily partitions."
    )

    error_count = 0
    total_processed_rows = 0

    # [TAKEHOME §36 / 案B] 手取り(pt−d, ATR単位)で正例に濃淡をつける最終重み w_final を別列で作る。
    #   uniqueness(=1/concurrency) は純粋なまま保持し、w_final = uniqueness × 手取り係数 を新設。
    #   手取り係数 = 正例: max(EPS, 1 + GAIN*(σ(K*(np−C0)) − 0.5))、負例: 1.0。
    #   np=C0 で係数=1、厚利>1・薄利<1、正例平均≈1（class balance を崩さない）。σ飽和で外れ値の濃淡発散なし。
    #   全体平均を要さない局所計算＝日次パーティション処理を崩さない。学習は weight_col=w_final を読む。
    TAKEHOME_EPS = 0.05
    TAKEHOME_K = 2.0
    TAKEHOME_C0 = 0.3  # PT=1.5(NP=1.5−d)に合わせ 0.5→0.3。PT変更時は pt_mult に連動させて調整
    TAKEHOME_GAIN = 1.5

    # concurrency_resultsを一度だけ遅延スキャンしておく
    concurrency_lf = pl.scan_parquet(CONCURRENCY_RESULTS_PATH)

    for i, path in enumerate(partitions):
        partition_name = f"{path.parent.parent.parent.name}/{path.parent.parent.name}/{path.parent.name}"
        logging.info(
            f"Processing: [{i + 1}/{len(partitions)}] - Partition {partition_name}..."
        )

        try:
            # 日次パーティションを遅延スキャン
            labeled_lf = pl.scan_parquet(path)

            # --- 修正前 ---
            # final_lf = (
            #     labeled_lf.join(concurrency_lf, on="timestamp", how="left")
            #     .with_columns(

            # --- 修正後 ---
            final_lf = (
                labeled_lf.join(
                    concurrency_lf, on=["timestamp", "timeframe"], how="left"
                )  # ★ timeframeを追加
                .with_columns(
                    [
                        # concurrency_long が存在し、かつ0より大きい場合に uniqueness_long を計算
                        pl.when(
                            pl.col("concurrency_long").is_not_null()
                            & (pl.col("concurrency_long") > 0)
                        )
                        .then(1.0 / pl.col("concurrency_long"))
                        .otherwise(0.0)
                        .alias("uniqueness_long"),
                        # concurrency_short が存在し、かつ0より大きい場合に uniqueness_short を計算
                        pl.when(
                            pl.col("concurrency_short").is_not_null()
                            & (pl.col("concurrency_short") > 0)
                        )
                        .then(1.0 / pl.col("concurrency_short"))
                        .otherwise(0.0)
                        .alias("uniqueness_short"),
                    ]
                )
                # [TAKEHOME §36 / 案B] uniqueness は純粋なまま。別列 w_final = uniqueness × 手取り係数 を新設。
                .with_columns(
                    [
                        (
                            pl.col("uniqueness_long")
                            * pl.when(
                                (pl.col("label_long") == 1)
                                & pl.col("np_realized_long").is_finite()
                            )
                            .then(
                                pl.max_horizontal(
                                    pl.lit(TAKEHOME_EPS),
                                    1.0
                                    + TAKEHOME_GAIN
                                    * (
                                        (
                                            1.0
                                            / (
                                                1.0
                                                + (
                                                    -TAKEHOME_K
                                                    * (pl.col("np_realized_long") - TAKEHOME_C0)
                                                ).exp()
                                            )
                                        )
                                        - 0.5
                                    ),
                                )
                            )
                            .otherwise(1.0)
                        ).alias("w_final_long"),
                        (
                            pl.col("uniqueness_short")
                            * pl.when(
                                (pl.col("label_short") == 1)
                                & pl.col("np_realized_short").is_finite()
                            )
                            .then(
                                pl.max_horizontal(
                                    pl.lit(TAKEHOME_EPS),
                                    1.0
                                    + TAKEHOME_GAIN
                                    * (
                                        (
                                            1.0
                                            / (
                                                1.0
                                                + (
                                                    -TAKEHOME_K
                                                    * (pl.col("np_realized_short") - TAKEHOME_C0)
                                                ).exp()
                                            )
                                        )
                                        - 0.5
                                    ),
                                )
                            )
                            .otherwise(1.0)
                        ).alias("w_final_short"),
                    ]
                )
                # concurrency と np_realized を削除（np_realized は w_final へ畳み込み済み＝特徴量漏洩防止）
                .select(
                    pl.all().exclude(
                        [
                            "concurrency_long",
                            "concurrency_short",
                            "np_realized_long",
                            "np_realized_short",
                        ]
                    )
                )
            )

            # 日次パーティションのパスから年/月/日を抽出
            day_part = path.parent.name
            month_part = path.parent.parent.name
            year_part = path.parent.parent.parent.name
            year = int(year_part.split("=")[1])
            month = int(month_part.split("=")[1])
            day = int(day_part.split("=")[1])

            # 結果を最終的な日次パーティション構造で書き出す
            output_dir = S6_WEIGHTED_DATASET / f"year={year}/month={month}/day={day}"
            output_dir.mkdir(parents=True, exist_ok=True)
            output_path = output_dir / "data.parquet"

            # 計算を実行し、結果をファイルに書き出す
            result_df = final_lf.collect(engine="streaming")

            if len(result_df) == 0:
                logging.warning(
                    f"Empty dataframe after processing partition: {partition_name}"
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
            f"Successfully processed {len(partitions)} partitions ({total_processed_rows} rows)."
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
