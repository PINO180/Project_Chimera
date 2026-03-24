import polars as pl
from pathlib import Path

# --- 設定 ---
# M1の学習に使った分足データのパス
SOURCE_PATH = Path(
    "/workspace/data/XAUUSD/stratum_6_training/1A_2B/weighted_dataset_partitioned_v2"
)
# 保存先
OUTPUT_PATH = Path("/workspace/data/daily_ohlc.parquet")


def create_daily_data_from_close():
    print(f"🔄 Reading minute data from: {SOURCE_PATH}")

    # closeのみ読み込む（open, high, lowがないため）
    try:
        q = pl.scan_parquet(str(SOURCE_PATH / "**/*.parquet"))

        # 日足にリサンプリング (1日1行に集約)
        # open/high/low がないので、close の推移から近似する
        daily_df = (
            q.select(["timestamp", "close"])  # 必要な列のみ選択
            .sort("timestamp")
            .group_by_dynamic("timestamp", every="1d")
            .agg(
                [
                    pl.col("close").first().alias("open"),  # 始値 ≈ その日の最初の終値
                    pl.col("close").max().alias("high"),  # 高値 ≈ その日の終値の最大
                    pl.col("close").min().alias("low"),  # 安値 ≈ その日の終値の最小
                    pl.col("close").last().alias("close"),  # 終値 = その日の最後の終値
                ]
            )
            .collect()
        )

        # 欠損値の除去（データがない日など）
        daily_df = daily_df.drop_nulls()

        print(f"✅ Generated {len(daily_df)} daily records (Approximated from Close).")
        print(daily_df.head())

        # 保存
        OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
        daily_df.write_parquet(OUTPUT_PATH)
        print(f"💾 Saved daily data to: {OUTPUT_PATH}")
        print("🚀 You can now run the backtest simulator!")

    except Exception as e:
        print(f"❌ Error: {e}")


if __name__ == "__main__":
    create_daily_data_from_close()
