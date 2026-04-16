import polars as pl
from pathlib import Path
import random

s6_dir = Path(
    "/workspace/data/XAUUSD/stratum_6_training/1A_2B/labeled_dataset_partitioned"
)
parquet_files = list(s6_dir.glob("year=*/month=*/day=*/*.parquet"))

if not parquet_files:
    print(f"❌ No parquet files found in {s6_dir}")
else:
    # ランダムに5ファイルほどスキーマを確認
    num_to_check = min(5, len(parquet_files))
    print(f"Checking schema for {num_to_check} random files in {s6_dir}...")
    for f in random.sample(parquet_files, num_to_check):
        try:
            schema = pl.read_parquet_schema(f)
            if "close" in schema:
                print(f"✅ Found 'close': {schema['close']} in {f.relative_to(s6_dir)}")
            else:
                print(f"❌ MISSING 'close' in {f.relative_to(s6_dir)}")
                print(f"   Schema: {schema}")
        except Exception as e:
            print(f"⚠️ Error reading schema for {f.relative_to(s6_dir)}: {e}")
