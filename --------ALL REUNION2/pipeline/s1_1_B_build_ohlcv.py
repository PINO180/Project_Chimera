#!/usr/bin/env python3
"""
Project Forge - Pipeline Step 2: Build OHLCV Bars
S1_RAW_TICK_PARQUETのティックデータから15種類のマルチタイムフレームバーを生成し、
S1_MULTITIMEFRAMEへHiveパーティション形式で出力する。

※当スクリプトは「生のOHLCVバーを生成する素材工場」としての役割に専念する。
※日足/週足などのカレンダー依存のパディングや、フォワードフィルによる特徴量化は、
※後続のengineスクリプト側で処理する。
"""

import sys
import os
import time
import shutil
import logging
import gc
from pathlib import Path
import pandas as pd

# -------------------------------------------------------------------
# 設定の読み込み
# -------------------------------------------------------------------
sys.path.append(str(Path(__file__).resolve().parents[1]))
import blueprint as config

# ロギング設定
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# -------------------------------------------------------------------
# タイムフレーム定義
# -------------------------------------------------------------------
# CPU (Polars) 用の頻度文字列
TIMEFRAMES_CPU = {
    "tick": None,
    "M0.5": "30s",
    "M1": "1m",
    "M3": "3m",
    "M5": "5m",
    "M8": "8m",
    "M15": "15m",
    "M30": "30m",
    "H1": "1h",
    "H4": "4h",
    "H6": "6h",
    "H12": "12h",
    "D1": "1d",
    "W1": "1w",
    "MN": "1mo",
}

# GPU (Pandas/cuDF) 用の頻度文字列
TIMEFRAMES_GPU = {
    "tick": None,
    "M0.5": "30s",
    "M1": "1min",
    "M3": "3min",
    "M5": "5min",
    "M8": "8min",
    "M15": "15min",
    "M30": "30min",
    "H1": "1H",
    "H4": "4H",
    "H6": "6H",
    "H12": "12H",
    "D1": "1D",
    "W1": "7D",
    "MN": "30D",
}

# -------------------------------------------------------------------
# 環境検出
# -------------------------------------------------------------------
try:
    import cudf
    import dask_cudf
    from dask_cuda import LocalCUDACluster
    from dask.distributed import Client

    USE_GPU = True
except ImportError:
    import polars as pl

    USE_GPU = False


def prepare_output_directory(output_dir: Path):
    """出力先ディレクトリの初期化"""
    if output_dir.exists():
        logger.warning(f"既存の出力ディレクトリを削除します: {output_dir}")
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"出力ディレクトリを作成しました: {output_dir}")


def process_with_gpu():
    """Dask-cuDFを使用したGPU処理パス"""
    logger.info("=== 処理エンジン: GPU (Dask-cuDF) ===")

    input_file = Path(config.S1_RAW_TICK_PARQUET)
    output_base_dir = Path(config.S1_MULTITIMEFRAME)

    logger.info(f"ティックデータを読み込み中: {input_file}")
    ddf = dask_cudf.read_parquet(input_file)

    ts_col = "datetime"
    price_col = "mid_price"

    # タイムスタンプと基本列の準備
    ddf[ts_col] = ddf[ts_col].astype("datetime64[ns]")
    if "tick_count" not in ddf.columns:
        ddf["tick_count"] = 1
    if "volume" not in ddf.columns:
        ddf["volume"] = ddf["tick_count"]

    # 数値計算用タイムスタンプ（ナノ秒）
    ddf_int_ts = ddf[ts_col].astype("int64")

    # 集計ルールの定義
    agg_rules = {
        price_col: ["first", "max", "min", "last"],
        "volume": ["sum"],
        "tick_count": ["sum"],
        "bid": ["mean"],
        "ask": ["mean"],
        "spread": ["mean"],
    }

    summary_stats = {}

    for name, freq in TIMEFRAMES_GPU.items():
        loop_start = time.time()
        logger.info(f"[{name}] バー生成中...")

        if name == "tick":
            resampled_ddf = ddf[
                [ts_col, price_col, "volume", "tick_count", "bid", "ask", "spread"]
            ].copy()
            resampled_ddf = resampled_ddf.rename(
                columns={ts_col: "timestamp", price_col: "open"}
            )
            resampled_ddf["high"] = resampled_ddf["open"]
            resampled_ddf["low"] = resampled_ddf["open"]
            resampled_ddf["close"] = resampled_ddf["open"]
        else:
            # 数値計算によるグルーピング
            bucket_size_ns = pd.to_timedelta(freq).total_seconds() * 1_000_000_000
            ddf["group_key"] = (ddf_int_ts // bucket_size_ns) * bucket_size_ns

            resampled = ddf.groupby("group_key").agg(agg_rules)
            # カラム名の平滑化 (MultiIndex対応)
            resampled.columns = [
                "open",
                "high",
                "low",
                "close",
                "volume",
                "tick_count",
                "bid",
                "ask",
                "spread",
            ]

            resampled = resampled.reset_index().rename(
                columns={"group_key": "timestamp"}
            )
            resampled["timestamp"] = resampled["timestamp"].astype("datetime64[ns]")
            resampled_ddf = resampled.drop(columns=["group_key"], errors="ignore")

        # スキーマの厳密なキャスト
        # timestamp: 前工程(s1_1_A)の出力であるマイクロ秒(us)からミリ秒(ms)へ明示的にダウンキャストして統一
        resampled_ddf["timestamp"] = resampled_ddf["timestamp"].astype("datetime64[ms]")

        for col in ["open", "high", "low", "close", "bid", "ask", "spread"]:
            resampled_ddf[col] = resampled_ddf[col].astype("float32")
        resampled_ddf["volume"] = resampled_ddf["volume"].astype("int64")
        resampled_ddf["tick_count"] = resampled_ddf["tick_count"].astype("int32")

        # timeframe列の型保証: CategoricalやObject型への意図せぬ変換を避け、純粋なString型を強制する
        resampled_ddf["timeframe"] = str(name)
        # resampled_ddf["timeframe"] = resampled_ddf["timeframe"].astype("string")
        # 安全な修正案
        resampled_ddf["timeframe"] = resampled_ddf["timeframe"].astype(
            str
        )  # objectとして保持

        # カラム順序の整理
        cols_order = [
            "timeframe",
            "timestamp",
            "open",
            "high",
            "low",
            "close",
            "volume",
            "tick_count",
            "bid",
            "ask",
            "spread",
        ]
        resampled_ddf = resampled_ddf[cols_order]

        # 出力
        output_path = output_base_dir / f"timeframe={name}"
        resampled_ddf.to_parquet(output_path, write_index=False)

        # サマリー用の件数カウント
        row_count = len(resampled_ddf)
        summary_stats[name] = {"count": row_count, "time": time.time() - loop_start}
        logger.info(
            f"[{name}] 完了: {row_count:,} 行 ({summary_stats[name]['time']:.2f}秒)"
        )

    return summary_stats


def process_with_cpu():
    """Polarsを使用したCPU処理パス"""
    logger.info("=== 処理エンジン: CPU (Polars) ===")

    input_file = Path(config.S1_RAW_TICK_PARQUET)
    output_base_dir = Path(config.S1_MULTITIMEFRAME)

    logger.info(f"ティックデータを読み込み中(Lazy): {input_file}")
    tick_lf = pl.scan_parquet(input_file)  # read_parquet から scan_parquet へ変更

    # LazyFrame用のスキーマ確認
    if "volume" not in tick_lf.collect_schema().names():
        tick_lf = tick_lf.with_columns(pl.lit(1, dtype=pl.Int64).alias("volume"))
    else:
        tick_lf = tick_lf.with_columns(
            pl.when(pl.col("volume").is_not_null())
            .then(pl.col("volume").cast(pl.Int64))
            .otherwise(pl.lit(0, dtype=pl.Int64))
            .alias("volume")
        )

    summary_stats = {}

    for name, freq in TIMEFRAMES_CPU.items():
        loop_start = time.time()
        logger.info(f"[{name}] バー生成中...")

        # --- bars を bars_lf (LazyFrame) として構築 ---
        if name == "tick":
            bars_lf = tick_lf.with_columns(
                [
                    pl.col("datetime").alias("timestamp"),
                    pl.col("mid_price").alias("open"),
                    pl.col("mid_price").alias("high"),
                    pl.col("mid_price").alias("low"),
                    pl.col("mid_price").alias("close"),
                    pl.lit(1).cast(pl.Int32).alias("tick_count"),
                ]
            )
        else:
            bars_lf = (
                tick_lf.sort("datetime")
                .group_by_dynamic("datetime", every=freq, closed="left", label="left")
                .agg(
                    [
                        pl.col("mid_price").first().alias("open"),
                        pl.col("mid_price").max().alias("high"),
                        pl.col("mid_price").min().alias("low"),
                        pl.col("mid_price").last().alias("close"),
                        pl.col("volume").sum().alias("volume"),
                        pl.col("mid_price").len().alias("tick_count"),
                        pl.col("bid").mean().alias("bid"),
                        pl.col("ask").mean().alias("ask"),
                        pl.col("spread").mean().alias("spread"),
                    ]
                )
                .filter(pl.col("tick_count") > 0)
            )

            bars_lf = bars_lf.rename({"datetime": "timestamp"})

        # スキーマの厳密なキャスト
        bars_lf = bars_lf.with_columns(
            [
                pl.lit(name).cast(pl.Utf8).alias("timeframe"),
                pl.col("timestamp").cast(pl.Datetime("ms")),
                pl.col("open").cast(pl.Float32),
                pl.col("high").cast(pl.Float32),
                pl.col("low").cast(pl.Float32),
                pl.col("close").cast(pl.Float32),
                pl.col("volume").cast(pl.Int64),
                pl.col("tick_count").cast(pl.Int32),
                pl.col("bid").cast(pl.Float32),
                pl.col("ask").cast(pl.Float32),
                pl.col("spread").cast(pl.Float32),
            ]
        ).select(
            [
                "timeframe",
                "timestamp",
                "open",
                "high",
                "low",
                "close",
                "volume",
                "tick_count",
                "bid",
                "ask",
                "spread",
            ]
        )

        # ここで初めて評価・実体化（ストリーミングが有効な処理はメモリを大幅に節約）
        bars = bars_lf.collect(engine="streaming")

        # 出力
        output_dir = output_base_dir / f"timeframe={name}"
        output_dir.mkdir(parents=True, exist_ok=True)
        bars.write_parquet(
            output_dir / "part-00000.parquet", compression="zstd", use_pyarrow=True
        )

        # サマリー用の件数カウント
        row_count = bars.height
        summary_stats[name] = {"count": row_count, "time": time.time() - loop_start}
        logger.info(
            f"[{name}] 完了: {row_count:,} 行 ({summary_stats[name]['time']:.2f}秒)"
        )

        # ★重要: ループごとに確実にメモリを解放
        del bars, bars_lf
        gc.collect()

    return summary_stats


def main():
    start_time = time.time()
    logger.info("--- マルチタイムフレームOHLCV生成を開始します ---")

    output_dir = Path(config.S1_MULTITIMEFRAME)
    prepare_output_directory(output_dir)

    if USE_GPU:
        summary_stats = process_with_gpu()
    else:
        summary_stats = process_with_cpu()

    total_time = time.time() - start_time

    # 最終サマリーの出力
    logger.info("=== 処理完了サマリー ===")
    logger.info(f"総処理時間: {total_time / 60:.2f} 分 ({total_time:.1f} 秒)")
    logger.info("時間足別 行数:")

    total_rows = 0
    for name, stats in summary_stats.items():
        logger.info(
            f"  - {name:>4}: {stats['count']:>12,} 行 ({stats['time']:>6.2f}秒)"
        )
        total_rows += stats["count"]

    logger.info(f"合計生成行数: {total_rows:,} 行")
    logger.info(f"出力先: {output_dir}")
    logger.info("--------------------------------------------------")


if __name__ == "__main__":
    if USE_GPU:
        # GPUが利用可能な場合のみローカルクラスタを起動
        with (
            LocalCUDACluster(
                n_workers=1, device_memory_limit="10GB", rmm_pool_size="8GB"
            ) as cluster,
            Client(cluster) as client,
        ):
            logger.info("Dask-CUDA Client を初期化しました。")
            logger.info(f"Dashboard link: {client.dashboard_link}")
            main()
    else:
        # CPU単体実行
        main()
