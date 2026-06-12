#!/usr/bin/env python3
"""
diagnose_mean10_detail.py — mean_10 の 31% 不一致が「未カバー」か「本物の skew」か確定

e1a_statistical_mean_10 (M5) について:
  - S2 の timestamp 範囲 vs deque の範囲
  - k=0 で S2 に相手がいるバー数 (= covered)、そのうち一致/不一致
  - 不一致バーの時間分布 (日別ヒスト) → 直近偏在なら未カバー、全域散在なら本物
  - 不一致バーの値が S2 のどこかに在るか (= timing か真の値差か)
  - 例を数件表示

使い方:
  python diagnose_mean10_detail.py
"""
from __future__ import annotations
import sys, pickle
from pathlib import Path
from collections import Counter
import numpy as np, pandas as pd, polars as pl
sys.path.insert(0, "/workspace")
import blueprint as config

TF = "M5"; TF_DUR = 300; TOL = 1e-6
FEATS = [("e1a", "e1a_statistical_mean_10"), ("e1d", "e1d_hv_annual_252")]


def s2_series(engine, feat):
    u = {"e1a":"A","e1b":"B","e1c":"C","e1d":"D","e1e":"E","e1f":"F"}[engine]
    p = (Path(config.S2_FEATURES_VALIDATED)/f"feature_value_a_vast_universe{u}"
         /f"features_{engine}_{TF}.parquet")
    df = pl.read_parquet(p).to_pandas()
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    return df[["timestamp", feat]].rename(columns={feat: "v"})


def main():
    st = pickle.load(open(config.STATE_CHECKPOINT_DIR/"feature_engine_state.pkl","rb"))
    pfb = st["proxy_feature_buffers"][TF]
    bts = [pd.Timestamp(t).tz_convert("UTC") if pd.Timestamp(t).tzinfo
           else pd.Timestamp(t).tz_localize("UTC") for t in pfb["__bar_ts__"]]

    for eng, feat in FEATS:
        if feat not in pfb:
            print(f"  {feat} は deque に無し"); continue
        dv = np.asarray(list(pfb[feat]), float)[-len(bts):]
        bts_f = bts[-len(dv):]
        s = s2_series(eng, feat)
        ts_to_v = dict(zip(s["timestamp"], s["v"].astype(float)))
        col_sorted = np.sort(s["v"].to_numpy(float))

        print("="*78)
        print(f"  {feat}")
        print("="*78)
        print(f"  deque __bar_ts__ 範囲 : {bts_f[0]} 〜 {bts_f[-1]}  ({len(dv)}本)")
        print(f"  S2 timestamp 範囲     : {s['timestamp'].min()} 〜 {s['timestamp'].max()}  ({len(s)}本)")

        covered = match = differ = 0
        differ_ts = []
        examples = []
        for i, b in enumerate(bts_f):
            tv = ts_to_v.get(b)          # k=0 (label = __bar_ts__)
            if tv is None or not np.isfinite(dv[i]):
                continue
            covered += 1
            if abs(dv[i]-tv) <= TOL:
                match += 1
            else:
                differ += 1
                differ_ts.append(b)
                if len(examples) < 8:
                    # この prod 値は S2 のどこかに在るか?
                    j = np.searchsorted(col_sorted, dv[i])
                    near = min(abs(col_sorted[min(j,len(col_sorted)-1)]-dv[i]),
                               abs(col_sorted[max(j-1,0)]-dv[i]))
                    examples.append((b, dv[i], tv, dv[i]-tv, near))

        print(f"  covered (k=0 で S2 に相手有) : {covered} / {len(dv)}")
        print(f"    一致(bit-identical)        : {match}  ({100*match/covered:.1f}%)" if covered else "")
        print(f"    不一致                      : {differ}  ({100*differ/covered:.1f}%)" if covered else "")

        if differ_ts:
            days = Counter(t.date() for t in differ_ts)
            print(f"  不一致バーの日別分布:")
            for d in sorted(days):
                print(f"    {d}: {days[d]} 本")
            print(f"  → 直近1-2日に偏る=未カバー疑い / 全域に散る=本物の skew")
            print(f"  不一致の例 (ts, prod, train, diff, prod値とS2全体の最近傍距離):")
            for b,pv,tv,df_,near in examples:
                tag = "(S2のどこかに在り=時刻問題)" if near <= TOL else "(S2に存在せず=真の値差)"
                print(f"    {b}  prod={pv:.6g}  train={tv:.6g}  diff={df_:+.4g}  nearest={near:.2e} {tag}")
        print()


if __name__ == "__main__":
    main()
