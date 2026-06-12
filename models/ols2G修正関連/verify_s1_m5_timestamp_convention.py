#!/usr/bin/env python3
"""
verify_s1_m5_timestamp_convention.py — S1 M5 timestamp 規約の実データ検証

仮説 A: label="left" (= bar 開始時刻) → close は未来の bar の close (= look-ahead)
仮説 B: label="right" (= bar 終了時刻) → close は同 timestamp の close (= OK)

S1 M5 の close と、 同じ tick stream の M0.5 close を直接比較して判定する。
"""

import sys
from pathlib import Path
import polars as pl
import pandas as pd

sys.path.insert(0, "/workspace")
import blueprint as config

# 1. S1 M5 全体を読む
s1_m5_dir = Path(config.S1_PROCESSED) / "timeframe=M5"
m5 = pl.scan_parquet(str(s1_m5_dir / "*.parquet")).select(["timestamp", "close"]).collect()
m5 = m5.with_columns(pl.col("timestamp").cast(pl.Datetime("us", "UTC"))).sort("timestamp").to_pandas()
m5["timestamp"] = pd.to_datetime(m5["timestamp"], utc=True)

# 2. S1 M0.5 全体を読む
s1_m05_dir = Path(config.S1_PROCESSED) / "timeframe=M0.5"
m05 = pl.scan_parquet(str(s1_m05_dir / "*.parquet")).select(["timestamp", "close"]).collect()
m05 = m05.with_columns(pl.col("timestamp").cast(pl.Datetime("us", "UTC"))).sort("timestamp").to_pandas()
m05["timestamp"] = pd.to_datetime(m05["timestamp"], utc=True)

# 3. テスト対象の timestamp を 5 個選ぶ (M5 の中盤辺り)
test_ts_list = m5["timestamp"].iloc[len(m5) // 2 : len(m5) // 2 + 5].tolist()

print("=" * 80)
print("S1 M5 timestamp 規約 実データ検証")
print("=" * 80)
print(f"  S1 M5: {len(m5):,} rows, range {m5['timestamp'].min()} 〜 {m5['timestamp'].max()}")
print(f"  S1 M0.5: {len(m05):,} rows, range {m05['timestamp'].min()} 〜 {m05['timestamp'].max()}")
print()

for ts in test_ts_list:
    print("-" * 80)
    m5_row = m5[m5["timestamp"] == ts]
    if len(m5_row) == 0:
        continue
    m5_close = m5_row["close"].iloc[0]
    print(f"S1 M5 timestamp = {ts}, close = {m5_close:.5f}")
    print()
    print(f"  この timestamp 前後 5分間の M0.5 close:")
    # ts と ts + 5min の間
    m05_range = m05[
        (m05["timestamp"] >= ts - pd.Timedelta(minutes=1))
        & (m05["timestamp"] <= ts + pd.Timedelta(minutes=6))
    ]
    for _, row in m05_range.iterrows():
        marker = ""
        if abs(row["close"] - m5_close) < 1e-6:
            marker = "  <-- これが S1 M5 close と一致"
        offset_sec = (row["timestamp"] - ts).total_seconds()
        print(f"    {row['timestamp']} (ts+{offset_sec:+.0f}sec) close={row['close']:.5f}{marker}")
    print()

print("=" * 80)
print("判定:")
print("  S1 M5[T].close が T+30sec 〜 T+60sec のいずれかの M0.5 close と一致")
print("    → label='right' (= bar 終了時刻仕様、 リーク無し)")
print("  S1 M5[T].close が T+4:30sec 付近の M0.5 close と一致")
print("    → label='left' (= bar 開始時刻仕様、 close は未来の bar)")
print("=" * 80)
