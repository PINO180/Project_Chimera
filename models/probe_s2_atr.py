#!/usr/bin/env python3
"""probe_s2_atr.py — S2の複数ATR正規化特徴量から学習側ATRを多経路逆算し一致を見る。"""
from __future__ import annotations
import argparse, pickle
from pathlib import Path
import numpy as np
import pandas as pd
import sys
sys.path.insert(0, "/workspace"); sys.path.insert(0, "/workspace/core")

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dump", type=Path, required=True)
    p.add_argument("--s2-root", type=Path, default=Path("/workspace/data/XAUUSD/stratum_2_features"))
    args = p.parse_args()
    from stable_rolling import stable_rolling_var, stable_rolling_std, stable_rolling_mean
    snap = pickle.load(open(args.dump,"rb"))
    close = snap["data"]["close"].astype(np.float64)
    ts = pd.Timestamp(snap["ts"]).tz_localize("UTC")
    tf = snap["tf"]
    sv10 = stable_rolling_var(close,10,1)[-1]
    ss10 = stable_rolling_std(close,10,1)[-1]
    srm10 = stable_rolling_mean(close,10)[-1]
    close_last = close[-1]
    print(f"本番分子: stable_var(10)={sv10:.10f}  stable_std(10)={ss10:.10f}  (close-srm(10))={close_last-srm10:.10f}\n")
    s2_file = args.s2_root/"feature_value_a_vast_universeA"/f"features_e1a_{tf}.parquet"
    df = pd.read_parquet(s2_file)
    if df["timestamp"].dt.tz is None: df["timestamp"]=df["timestamp"].dt.tz_localize("UTC")
    else: df["timestamp"]=df["timestamp"].dt.tz_convert("UTC")
    row = df[df["timestamp"]==ts]
    if len(row)==0:
        print("S2に該当ts無し"); return 1
    r = row.iloc[0]
    print("S2保存値 → 逆算ATR:")
    results = {}
    if "e1a_statistical_variance_10" in r:
        v=r["e1a_statistical_variance_10"]; atr=np.sqrt(sv10/v) if v>0 else np.nan
        results["variance_10"]=atr; print(f"  variance_10 = {v:.10f}  → ATR= {atr:.10f}")
    if "e1a_statistical_std_10" in r:
        v=r["e1a_statistical_std_10"]; atr=ss10/v if v!=0 else np.nan
        results["std_10"]=atr; print(f"  std_10      = {v:.10f}  → ATR= {atr:.10f}")
    if "e1a_statistical_mean_10" in r:
        v=r["e1a_statistical_mean_10"]; atr=(close_last-srm10)/v if v!=0 else np.nan
        results["mean_10"]=atr; print(f"  mean_10     = {v:.10f}  → ATR= {atr:.10f}")
    print(f"\n本番ATR(dump 3600本) = 1.7905396191278222")
    vals=[x for x in results.values() if np.isfinite(x)]
    if len(vals)>=2:
        spread=max(vals)-min(vals)
        print(f"逆算ATR群の散らばり: {spread:.2e}")
        if spread < 1e-6:
            mean=np.mean(vals)
            print(f"→ ✅ 全経路が同一ATR={mean:.10f} に収束 = 学習側ATR")
            print(f"   本番との差: {mean-1.7905396191278222:+.4e} ({abs(mean-1.7905396191278222)/1.7905*100:.2f}%)")
        else:
            print(f"→ ❌ 経路で逆算ATRがバラつく = 式の不一致が真因")
            for k,vv in results.items(): print(f"     {k}: {vv:.10f}")

if __name__=="__main__":
    main()
