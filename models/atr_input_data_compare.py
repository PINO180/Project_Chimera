#!/usr/bin/env python3
"""atr_input_data_compare.py — 本番dump OHLCV と 学習S1 M0.5 の同時刻OHLCVをbit比較。"""
from __future__ import annotations
import argparse, pickle
from pathlib import Path
import numpy as np
import pandas as pd

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dump", type=Path, required=True)
    p.add_argument("--s1-m05", type=Path,
                   default=Path("/workspace/data/XAUUSD/stratum_1_base/master_multitimeframe/timeframe=M0.5"))
    p.add_argument("--n-tail", type=int, default=20)
    args = p.parse_args()
    snap = pickle.load(open(args.dump, "rb"))
    ts_target = pd.Timestamp(snap["ts"]).tz_localize("UTC")
    print(f"dump ts(末尾バー) = {ts_target}, deque長={len(snap['data']['close'])}\n")
    dump_close = snap["data"]["close"].astype(np.float64)
    dump_high = snap["data"]["high"].astype(np.float64)
    dump_low = snap["data"]["low"].astype(np.float64)
    nN = args.n_tail
    dump_ts = [ts_target - pd.Timedelta(seconds=30*(nN-1-i)) for i in range(nN)]
    dump_tail = pd.DataFrame({"timestamp": dump_ts,
        "high_dump": dump_high[-nN:], "low_dump": dump_low[-nN:], "close_dump": dump_close[-nN:]})
    files = sorted(args.s1_m05.rglob("*.parquet"))
    if not files:
        print(f"ERROR: S1 M0.5 parquet 無し: {args.s1_m05}"); return 1
    s1 = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    if s1["timestamp"].dt.tz is None:
        s1["timestamp"] = s1["timestamp"].dt.tz_localize("UTC")
    else:
        s1["timestamp"] = s1["timestamp"].dt.tz_convert("UTC")
    s1 = s1[["timestamp","high","low","close"]].rename(columns={"high":"high_s1","low":"low_s1","close":"close_s1"})
    m = dump_tail.merge(s1, on="timestamp", how="left")
    print(f"末尾{nN}本の OHLCV bit比較 (dump vs 学習S1):\n")
    for _, r in m.iterrows():
        ch = r["close_dump"]-r["close_s1"] if pd.notna(r["close_s1"]) else float('nan')
        hh = r["high_dump"]-r["high_s1"] if pd.notna(r["high_s1"]) else float('nan')
        lh = r["low_dump"]-r["low_s1"] if pd.notna(r["low_s1"]) else float('nan')
        flag = "" if (abs(ch)<1e-9 and abs(hh)<1e-9 and abs(lh)<1e-9) else "  <<< DIFF"
        print(f"  {r['timestamp']}: Δclose={ch:+.2e} Δhigh={hh:+.2e} Δlow={lh:+.2e}{flag}")
    matched = m.dropna(subset=["close_s1"])
    if len(matched)>0:
        dc=(matched["close_dump"]-matched["close_s1"]).abs()
        dh=(matched["high_dump"]-matched["high_s1"]).abs()
        dl=(matched["low_dump"]-matched["low_s1"]).abs()
        print(f"\nマッチ {len(matched)}/{nN} 本:")
        print(f"  close max|Δ|={dc.max():.2e}  high max|Δ|={dh.max():.2e}  low max|Δ|={dl.max():.2e}")
        print(f"\n解釈: 全部Δ<1e-9→入力一致(ATR差は別経路) / Δ有意→入力OHLCVズレが真因")
    else:
        print("\n⚠ 時刻マッチ無し。30秒刻み仮定が学習側とズレ。値ベース照合に切替要")

if __name__ == "__main__":
    main()
