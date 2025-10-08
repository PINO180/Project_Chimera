"""
check_timestamp_dtype.py
全特徴量ファイルのtimestamp型を診断
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))

import blueprint as config
import polars as pl
from collections import defaultdict

feature_universe_path = Path(config.S2_FEATURES2)

results = defaultdict(list)

print("timestamp型を診断中...")

for engine_dir in feature_universe_path.iterdir():
    if not engine_dir.is_dir():
        continue
    
    for item in engine_dir.iterdir():
        try:
            if item.is_file() and item.suffix == '.parquet':
                df = pl.read_parquet(item, n_rows=1)
            elif item.is_dir():
                # パーティション内の最初のファイル
                first_file = next(item.rglob("*.parquet"))
                df = pl.read_parquet(first_file, n_rows=1)
            else:
                continue
            
            if 'timestamp' in df.columns:
                dtype = str(df['timestamp'].dtype)
                results[dtype].append(item.name)
        except:
            results['ERROR'].append(item.name)

# 結果表示
print("\n" + "=" * 60)
print("診断結果")
print("=" * 60)

for dtype, files in sorted(results.items()):
    print(f"\n[{dtype}] ({len(files)}個)")
    for f in sorted(files)[:10]:  # 最初の10個のみ表示
        print(f"  - {f}")
    if len(files) > 10:
        print(f"  ... 他{len(files) - 10}個")

print("\n" + "=" * 60)
if 'Datetime(time_unit=\'us\', time_zone=None)' in results:
    us_count = len(results['Datetime(time_unit=\'us\', time_zone=None)'])
    print(f"⚠️  {us_count}個のファイルがus型（修正が必要）")
else:
    print("✅ 全てns型（修正不要）")
print("=" * 60)