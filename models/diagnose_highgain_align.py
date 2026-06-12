#!/usr/bin/env python3
"""
diagnose_highgain_align.py — 予測を動かす高gain特徴が prod/S6 でどう違うか外科的に見る

予測 (M1 long) を支配する高 gain の post-OLS 特徴について、 signal バーで
  prod (snapshot)  vs  S6[T-1=10:27]  vs  S6[T=10:30]
を並べ、 各特徴が lag-1/lag-0 どちらに近いか・どれだけ違うかを出す。
高 gain 特徴が lag-1 で揃う → timing 整合で改善余地。
高 gain 特徴がどちらとも違う → magnitude (volume/moment) の実差。

使い方:
  python diagnose_highgain_align.py \
    --snapshot /workspace/.../snapshot_20260528_103001_L0.413_S0.000.csv
"""
from __future__ import annotations
import sys, argparse, glob
from pathlib import Path
import numpy as np, pandas as pd
try:
    import polars as pl
except Exception:
    pl = None
import joblib
sys.path.insert(0, "/workspace")
import blueprint as config


def parse_snapshot(p):
    df = pd.read_csv(p); df.columns=[c.strip() for c in df.columns]
    meta, feats = {}, {}
    for _, r in df.iterrows():
        k=str(r["feature_name"]).strip(); v=r["value"]
        if k.startswith("_"): meta[k]=v
        else: feats[k]=pd.to_numeric(v, errors="coerce")
    return meta, feats


def load_s6_window(s6_dir, t0, t1):
    files = glob.glob(str(s6_dir/"**"/"*.parquet"), recursive=True) or glob.glob(str(s6_dir/"*.parquet"))
    if not files: return None, None
    sample = pl.read_parquet(files[0], n_rows=5) if pl else pd.read_parquet(files[0]).head(5)
    tsc = next((c for c in sample.columns if "time" in c.lower() or "date" in c.lower()), None)
    frames=[]
    for f in files:
        try: d = pl.read_parquet(f).to_pandas() if pl else pd.read_parquet(f)
        except Exception: continue
        d[tsc]=pd.to_datetime(d[tsc],utc=True,errors="coerce")
        d=d[(d[tsc]>=t0)&(d[tsc]<=t1)]
        if len(d): frames.append(d)
    if not frames: return None, tsc
    return pd.concat(frames,ignore_index=True), tsc


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--snapshot", type=Path, required=True)
    ap.add_argument("--s6-dir", type=Path, default=Path(config.S6_WEIGHTED_DATASET))
    ap.add_argument("--tf", default="M3")
    ap.add_argument("--top", type=int, default=25)
    args=ap.parse_args()
    dur=pd.Timedelta(minutes=float(args.tf.replace("M","")))

    meta, feats = parse_snapshot(args.snapshot)
    bar=pd.to_datetime(meta.get("_timestamp_utc"),utc=True).floor("min")

    m=joblib.load(config.S7_M1_MODEL_LONG_PKL)
    b=m.booster_ if hasattr(m,"booster_") else m
    names=b.feature_name(); g=b.feature_importance(importance_type="gain")
    tot=g.sum() or 1
    gain={n:100*gi/tot for n,gi in zip(names,g)}
    top=sorted(gain.items(), key=lambda kv: kv[1], reverse=True)[:args.top]

    s6,tsc=load_s6_window(args.s6_dir, bar-2*dur, bar+1*dur)
    if s6 is None: print("S6 読込失敗"); return
    s6=s6.drop_duplicates(tsc).set_index(tsc).sort_index()
    r27 = s6.loc[bar-dur] if (bar-dur) in s6.index else None
    r30 = s6.loc[bar] if bar in s6.index else None

    print("="*100)
    print(f"  diagnose_highgain_align — signal bar {bar}  (prod _m1_long={meta.get('_m1_long')})")
    print(f"  高 gain post-OLS 特徴: prod vs S6[10:27](lag-1) vs S6[10:30](lag-0)")
    print("="*100)
    print(f"  {'feature':<46}{'gain%':>6}{'prod':>9}{'S6@27':>9}{'S6@30':>9}{'近い':>6}")
    n27=n30=0
    for feat, gn in top:
        pv = feats.get(feat, np.nan)
        v27 = float(pd.to_numeric(r27[feat],errors='coerce')) if (r27 is not None and feat in r27.index) else np.nan
        v30 = float(pd.to_numeric(r30[feat],errors='coerce')) if (r30 is not None and feat in r30.index) else np.nan
        d27=abs(pv-v27); d30=abs(pv-v30)
        near = "27" if d27<d30 else "30"
        if d27<d30: n27+=1
        else: n30+=1
        print(f"  {feat:<46}{gn:>6.2f}{pv:>9.3f}{v27:>9.3f}{v30:>9.3f}{near:>6}")
    print("="*100)
    print(f"  高gain TOP{args.top} のうち lag-1(10:27)に近い {n27} / lag-0(10:30)に近い {n30}")
    print(f"  lag-1 多数 → timing 整合で改善余地 / 割れる・どちらも遠い → magnitude(volume/moment)実差")
    print("="*100)


if __name__=="__main__":
    main()
