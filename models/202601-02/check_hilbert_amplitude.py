import polars as pl
from pathlib import Path

output_dir = Path("/workspace/data/XAUUSD/stratum_2_features_fixed/v5_gatekeeper_ready")

# 1ファイルではなく月全体を読む
glob_pattern = "year=2024/month=11/**/*.parquet"

try:
    print(f"--- チェック対象: {glob_pattern} ---")

    # next()を使わず、globパターンに一致する全ファイルをLazy読み込みして結合
    df = pl.scan_parquet(str(output_dir / glob_pattern)).collect().sort("timestamp")

    col = "e1e_hilbert_amplitude_50"

    # 50期間の移動平均を計算し、「現在の値 / 移動平均」の倍率（Ratio）を出す
    df = df.with_columns(
        pl.col(col).rolling_mean(window_size=50, min_samples=1).alias("hilbert_mean")
    ).with_columns((pl.col(col) / pl.col("hilbert_mean")).alias("trigger_ratio"))

    # NaNや無限大を除外
    df_valid = df.filter(pl.col("trigger_ratio").is_finite())

    print(f"--- '移動平均に対する倍率 (trigger_ratio)' の統計情報 ---")
    print(df_valid.select("trigger_ratio").describe())

    print(f"\n--- 倍率のパーセンタイル分布 ---")
    print(
        df_valid.select(
            [
                pl.col("trigger_ratio").quantile(0.50).alias("p50 (通常時)"),
                pl.col("trigger_ratio").quantile(0.75).alias("p75 (やや活発)"),
                pl.col("trigger_ratio").quantile(0.85).alias("p85 (初動候補)"),
                pl.col("trigger_ratio").quantile(0.90).alias("p90 (明確な初動)"),
                pl.col("trigger_ratio").quantile(0.95).alias("p95 (急騰/急落)"),
                pl.col("trigger_ratio").quantile(0.99).alias("p99 (異常事態)"),
            ]
        )
    )

except StopIteration:
    print("エラー: データが見つかりません。")
except pl.exceptions.ColumnNotFoundError as e:
    print(f"エラー: カラムが見つかりません: {e}")
