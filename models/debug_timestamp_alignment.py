#!/usr/bin/env python3
"""
debug_timestamp_alignment.py — dump deque 末尾と学習側 S2 timestamp 対応を目視確認

1 つの dump を選んで、 dump x_deque 末尾 5 個 と S2 e1a 同 TF の signal_ts 前後の
値を直接表示。 「何 bar 分ズレているか」 を一発で確認できる。
"""

import pickle
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import polars as pl

sys.path.insert(0, "/workspace")
import blueprint as config

# 確認したい dump (1 件)
DUMP_PATH = Path("/workspace/logs/ols_state_dumps/ols_20260526_143000_M3.pkl")

with open(DUMP_PATH, "rb") as f:
    dump = pickle.load(f)

signal_ts = pd.Timestamp(dump["signal_timestamp"]).tz_convert("UTC")
print(f"signal_timestamp: {signal_ts}")
print(f"signal_timeframe: {dump['signal_timeframe']}")
print()

# 各 TF で dump x_deque 末尾 vs S2 timestamp 周辺
for tf in ["M0.5", "M1", "M3", "M5", "M8", "M15"]:
    x_deque = dump["proxy_feature_buffers"].get(tf, {}).get("market_proxy", [])
    if not x_deque:
        continue

    # 学習側 S2 e1a 同 TF をロード
    s2_path = (
        Path(config.S2_FEATURES_VALIDATED)
        / "feature_value_a_vast_universeA"
        / f"features_e1a_{tf}.parquet"
    )
    s2 = pl.read_parquet(s2_path).select("timestamp").to_pandas()
    s2["timestamp"] = pd.to_datetime(s2["timestamp"], utc=True)

    # signal_ts ピッタリの S2 timestamp の前後を表示
    s2_around = s2[
        (s2["timestamp"] >= signal_ts - pd.Timedelta(seconds=600))
        & (s2["timestamp"] <= signal_ts + pd.Timedelta(seconds=600))
    ].sort_values("timestamp")

    print("=" * 70)
    print(f"TF={tf} (deque length={len(x_deque)})")
    print("=" * 70)
    print(f"dump x_deque 末尾 5 個:")
    for v in x_deque[-5:]:
        print(f"  {v:+.10e}")
    print()
    print(f"S2 e1a {tf} timestamp signal_ts ±10min:")
    for ts in s2_around["timestamp"]:
        marker = "  <-- signal_ts" if ts == signal_ts else (
                 "  <-- signal_ts - 1bar" if ts < signal_ts else "")
        print(f"  {ts}{marker}")
    print()
