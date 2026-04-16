import polars as pl
from pathlib import Path

# 例: tickデータの一つのファイルを調べる
s6_tick_file = Path(
    "/workspace/data/XAUUSD/stratum_6_training/1A_2B/labeled_dataset_partitioned/year=2021/month=8/day=1/data.parquet"
)  # 適切なパスに変更

if s6_tick_file.exists():
    df = pl.read_parquet(s6_tick_file)
    print(f"Schema for {s6_tick_file.name}:")
    print(df.schema)
    print("\nFirst 5 timestamps:")
    print(df.head(5)["timestamp"])
    # 秒やミリ秒が表示されるか確認
else:
    print(f"File not found: {s6_tick_file}")
