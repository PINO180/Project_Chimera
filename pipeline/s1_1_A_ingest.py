#!/usr/bin/env python3
"""
Project Forge - Stratum 1 Ingestion Script (s1_1_A_ingest.py)
生ティックデータ(CSV)をPolarsでチャンク処理し、ParquetおよびHiveパーティション形式へ変換する。
"""

import sys
import os
import time
import gc
import logging
import tempfile
from pathlib import Path
import polars as pl

# ==========================================
# 1. Path Resolution & Config Import
# ==========================================
sys.path.append(str(Path(__file__).resolve().parents[1]))
import blueprint as config

# ==========================================
# 2. Logging Setup
# ==========================================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


# ==========================================
# 3. Processing Configuration
# ==========================================
class ProcessingConfig:
    # チャンクサイズ（メモリ使用量と処理速度のトレードオフで調整）
    CHUNK_SIZE = 50000


# ==========================================
# 4. Core Processing Functions
# ==========================================
def process_chunk_data(chunk_data: list, chunk_num: int) -> pl.DataFrame:
    """
    1チャンク分の文字列リストをPolarsでパースし、型変換・派生カラム計算を行う。
    """
    if not chunk_data:
        return pl.DataFrame()

    logger.info(f"  チャンク {chunk_num}: {len(chunk_data):,} 行を処理中...")

    # チャンクをPolarsデータフレームに変換
    df = pl.DataFrame({"raw_line": chunk_data})

    # タブ区切りデータを分割 (DATE TIME BID ASK LAST VOLUME FLAGS)
    df = (
        df.with_columns([pl.col("raw_line").str.split("\t").alias("split_data")])
        .filter(pl.col("split_data").list.len() >= 6)
        .with_columns(
            [
                pl.col("split_data").list.get(0).alias("date_str"),
                pl.col("split_data").list.get(1).alias("time_str"),
                pl.col("split_data").list.get(2).alias("bid_str"),
                pl.col("split_data").list.get(3).alias("ask_str"),
                pl.col("split_data").list.get(4).alias("last_str"),
                pl.col("split_data").list.get(5).alias("volume_str"),
                pl.col("split_data").list.get(6).alias("flags_str"),
            ]
        )
    )

    # 日時の結合と各種型のキャスト
    df = df.with_columns(
        [(pl.col("date_str") + " " + pl.col("time_str")).alias("datetime_str")]
    ).with_columns(
        [
            # 【タイムスタンプ精度方針】
            # 元のCSVデータのミリ秒精度(%3f)を保持し、後続エンジンでの不要なキャストを防ぐため、
            # Polarsデフォルトのマイクロ秒(us)ではなく、明示的にミリ秒(ms)としてパース・統一します。
            pl.col("datetime_str")
            .str.to_datetime(
                format="%Y.%m.%d %H:%M:%S.%3f", time_unit="ms", strict=False
            )
            .alias("datetime"),
            pl.when(pl.col("bid_str").str.len_chars() > 0)
            .then(pl.col("bid_str").cast(pl.Float64, strict=False))
            .otherwise(None)
            .alias("bid"),
            pl.when(pl.col("ask_str").str.len_chars() > 0)
            .then(pl.col("ask_str").cast(pl.Float64, strict=False))
            .otherwise(None)
            .alias("ask"),
            pl.when(pl.col("last_str").str.len_chars() > 0)
            .then(pl.col("last_str").cast(pl.Float64, strict=False))
            .otherwise(None)
            .alias("last"),
            pl.when(pl.col("volume_str").str.len_chars() > 0)
            .then(pl.col("volume_str").cast(pl.Int64, strict=False))
            .otherwise(None)
            .alias("volume"),
            pl.when(pl.col("flags_str").str.len_chars() > 0)
            .then(pl.col("flags_str").cast(pl.Int32, strict=False))
            .otherwise(None)
            .alias("flags"),
        ]
    )

    # 有効なデータのフィルタリング
    df = df.filter(
        (pl.col("datetime").is_not_null())
        & (pl.col("bid").is_not_null())
        & (pl.col("ask").is_not_null())
        & (pl.col("bid") > 0)
        & (pl.col("ask") > 0)
    ).select(["datetime", "bid", "ask", "last", "volume", "flags"])

    if df.shape[0] == 0:
        return df

    # 派生カラム計算と型最適化 (float32化によるメモリ節約)
    # 併せてHiveパーティション用のyear/month/dayをint32で作成（後続エンジンのdtype問題回避）
    df = df.with_columns(
        [
            (pl.col("ask") - pl.col("bid")).alias("spread"),
            ((pl.col("ask") + pl.col("bid")) / 2.0).alias("mid_price"),
            pl.col("datetime").dt.year().cast(pl.Int32).alias("year"),
            pl.col("datetime").dt.month().cast(pl.Int32).alias("month"),
            pl.col("datetime").dt.day().cast(pl.Int32).alias("day"),
        ]
    ).with_columns(
        [
            pl.col("bid").cast(pl.Float32),
            pl.col("ask").cast(pl.Float32),
            pl.col("last").cast(pl.Float32),
            pl.col("spread").cast(pl.Float32),
            pl.col("mid_price").cast(pl.Float32),
        ]
    )

    return df


# ==========================================
# 5. Main Ingestion Pipeline
# ==========================================
def ingest_s0_to_s1():
    """
    S0のCSVを読み込み、S1の単一ParquetおよびパーティションParquetへ変換・出力する。
    """
    input_file = Path(config.S0_RAW_CSV)
    output_single = Path(config.S1_RAW_TICK_PARQUET)
    output_partitioned = Path(config.S1_RAW_TICK_PARTITIONED)

    logger.info("=== S1 Ingestion (Raw CSV -> Parquet) 開始 ===")
    logger.info(f"対象シンボル: {config.SYMBOL}")
    logger.info(f"入力ファイル: {input_file}")
    logger.info(f"出力(単一): {output_single}")
    logger.info(f"出力(分割): {output_partitioned}")

    if not input_file.exists():
        logger.error(f"入力ファイルが見つかりません: {input_file}")
        sys.exit(1)

    # 既存ファイルの確認と上書きプロンプト
    if output_single.exists() or output_partitioned.exists():
        ans = input(
            f"\n警告: 出力先のParquetファイルまたはディレクトリが既に存在します。\n上書きして続行しますか？ [y/N]: "
        )
        if ans.lower() != "y":
            logger.info("ユーザーキャンセルにより処理を中断しました。")
            sys.exit(0)

    # 出力先ディレクトリの確保
    output_single.parent.mkdir(parents=True, exist_ok=True)
    output_partitioned.mkdir(parents=True, exist_ok=True)

    start_time = time.time()
    total_rows = 0
    temp_files = []

    try:
        file_size_mb = input_file.stat().st_size / (1024**2)
        logger.info(f"入力ファイルサイズ: {file_size_mb:.1f} MB")

        # tempfileモジュールを使用して一時ディレクトリを作成（エラー時もwith句抜けで自動削除）
        with tempfile.TemporaryDirectory(prefix="forge_s1_") as temp_dir:
            logger.info(f"一時ディレクトリを作成しました: {temp_dir}")

            with open(input_file, "r", encoding="utf-8") as f:
                header_line = f.readline().strip()
                logger.debug(f"ヘッダースキップ: {header_line}")

                chunk_num = 0
                chunk_data = []

                for line_num, line in enumerate(f, 1):
                    line = line.strip()
                    if line:
                        chunk_data.append(line)

                    if len(chunk_data) >= ProcessingConfig.CHUNK_SIZE:
                        chunk_num += 1
                        df_chunk = process_chunk_data(chunk_data, chunk_num)

                        if df_chunk.shape[0] > 0:
                            temp_file_path = (
                                Path(temp_dir) / f"chunk_{chunk_num:04d}.parquet"
                            )
                            df_chunk.write_parquet(
                                temp_file_path, compression="zstd", use_pyarrow=True
                            )
                            temp_files.append(temp_file_path)
                            total_rows += df_chunk.shape[0]

                        del df_chunk
                        chunk_data = []
                        gc.collect()

                        if chunk_num % 10 == 0:
                            elapsed = time.time() - start_time
                            rate = line_num / elapsed if elapsed > 0 else 0
                            logger.info(
                                f"  [進捗] {line_num:,} 行読み込み完了 (速度: {rate:,.0f} 行/秒)"
                            )

                # 最終チャンクの処理
                if chunk_data:
                    chunk_num += 1
                    df_chunk = process_chunk_data(chunk_data, chunk_num)
                    if df_chunk.shape[0] > 0:
                        temp_file_path = (
                            Path(temp_dir) / f"chunk_{chunk_num:04d}.parquet"
                        )
                        df_chunk.write_parquet(
                            temp_file_path, compression="zstd", use_pyarrow=True
                        )
                        temp_files.append(temp_file_path)
                        total_rows += df_chunk.shape[0]

            if total_rows == 0:
                logger.warning("有効なデータが見つかりませんでした。処理を終了します。")
                return

            # ==========================================
            # 中間ファイルの結合と出力
            # ==========================================
            logger.info(
                f"チャンク処理完了。中間ファイル({len(temp_files)}個)をストリーミングで結合します..."
            )

            temp_pattern = [str(f) for f in temp_files]

            # 1. 単一Parquetファイルの出力（sink_parquetでメモリに乗せずストリーミング書き出し）
            logger.info(f"単一Parquetファイルを出力中: {output_single}")
            (
                pl.scan_parquet(temp_pattern)
                .sort("datetime")
                .sink_parquet(
                    output_single,
                    compression="zstd",
                )
            )

            # 2. Hiveパーティション形式のParquet出力
            logger.info(f"Hiveパーティション形式で出力中: {output_partitioned}")
            import pyarrow.parquet as pq

            for i, temp_file in enumerate(temp_files):
                chunk_table = pl.read_parquet(temp_file).to_arrow()
                pq.write_to_dataset(
                    chunk_table,
                    root_path=str(output_partitioned),
                    partition_cols=["year", "month", "day"],
                    compression="zstd",
                )
                del chunk_table
                if (i + 1) % 100 == 0:
                    logger.info(
                        f"  Hive出力進捗: {i + 1}/{len(temp_files)} ファイル完了"
                    )

            logger.info("一時ファイルの自動クリーンアップを実行します。")

        # ==========================================
        # サマリー情報の出力
        # ==========================================
        processing_time = time.time() - start_time
        out_single_size_mb = output_single.stat().st_size / (1024**2)

        # サマリー用にメタデータのみ取得（全データをメモリに乗せない）
        pf = pl.read_parquet(output_single, columns=["datetime"])
        dt_min = pf["datetime"].min()
        dt_max = pf["datetime"].max()
        total_rows_final = len(pf)
        del pf

        logger.info("\n" + "=" * 40)
        logger.info("=== S1 Ingestion 処理完了サマリー ===")
        logger.info("=" * 40)
        logger.info(f"総処理行数 : {total_rows_final:,} 行")
        logger.info(f"日時範囲   : {dt_min} 〜 {dt_max}")
        logger.info(f"出力サイズ : {out_single_size_mb:.1f} MB (単一Parquet)")
        logger.info(f"圧縮率     : {file_size_mb / out_single_size_mb:.1f}x")
        logger.info(
            f"処理時間   : {processing_time / 60:.1f} 分 ({processing_time:.1f} 秒)"
        )
        logger.info(f"処理速度   : {total_rows_final / processing_time:,.0f} 行/秒")
        logger.info("=" * 40)

    except Exception as e:
        logger.error(f"パイプライン処理中にエラーが発生しました: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    ingest_s0_to_s1()
