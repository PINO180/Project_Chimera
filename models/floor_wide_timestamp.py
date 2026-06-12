#!/usr/bin/env python3
"""feature_snapshots_wide.csv の Timestamp を分単位 floor して training の
M3 グリッド時刻 (秒=00) に合わせた一時 CSV を出力する応急処置。"""
import sys
import pandas as pd

src = sys.argv[1]
dst = sys.argv[2]

df = pd.read_csv(src, low_memory=False)
ts_col = "Timestamp" if "Timestamp" in df.columns else "timestamp"
ts = pd.to_datetime(df[ts_col], utc=True, errors="coerce")
df[ts_col] = ts.dt.floor("min").dt.strftime("%Y-%m-%d %H:%M:%S")
df.to_csv(dst, index=False)
print(f"floor 完了: {src} -> {dst}")
print(f"  行数: {len(df)}")
print(f"  Timestamp 例: {df[ts_col].iloc[0]} ... {df[ts_col].iloc[-1]}")
print(f"  ユニーク時刻数: {df[ts_col].nunique()}")
