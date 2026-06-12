#!/usr/bin/env python3
"""
diagnose_feature_correspondence.py — 53-68% feature が「真の値差」か「時刻ズレ」かを確定

各代表特徴量 (M5) について:
  (A) 値メンバーシップ: production deque の各値が training 列の値集合に tol 内で存在する割合。
       ~100% → 値は本物 (= 時刻/位相の問題)。 低い → 値が本当に違う (= 真の skew)。
  (B) k カーブ: k=-5..+5 の一致率。 クリーンな単一ピーク → 一定オフセット。
       平坦 → 自己相関で複数 k が当たるだけ (best_k は無意味)。 どこも低い → 真の差。

使い方:
  python diagnose_feature_correspondence.py
"""
from __future__ import annotations
import sys, pickle
from pathlib import Path
import numpy as np, pandas as pd, polars as pl
sys.path.insert(0, "/workspace")
import blueprint as config

ENGINE_TO_UNIVERSE = {"e1a":"A","e1b":"B","e1c":"C","e1d":"D","e1e":"E","e1f":"F"}
TF = "M5"; TF_DUR = 300; TOL = 1e-6


def s2_series(engine, feat):
    u = ENGINE_TO_UNIVERSE.get(engine)
    if u is None: return None
    p = (Path(config.S2_FEATURES_VALIDATED)/f"feature_value_a_vast_universe{u}"
         /f"features_{engine}_{TF}.parquet")
    if not p.exists(): return None
    df = pl.read_parquet(p).to_pandas()
    if feat not in df.columns: return None
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    return df[["timestamp", feat]].rename(columns={feat: "v"})


def main():
    st = pickle.load(open(config.STATE_CHECKPOINT_DIR/"feature_engine_state.pkl","rb"))
    pfb = st["proxy_feature_buffers"][TF]
    bts = [pd.Timestamp(t).tz_convert("UTC") if pd.Timestamp(t).tzinfo
           else pd.Timestamp(t).tz_localize("UTC") for t in pfb["__bar_ts__"]]

    # engine 別に代表 1 特徴量
    picked = {}
    for feat in pfb:
        if feat in ("market_proxy","__bar_ts__"): continue
        eng = feat.split("_")[0]
        if eng not in ENGINE_TO_UNIVERSE or eng in picked: continue
        s = s2_series(eng, feat)
        if s is not None: picked[eng] = (feat, s)

    print("="*80)
    print("  diagnose_feature_correspondence — 値メンバーシップ & k カーブ (TF=M5)")
    print("="*80)

    for eng,(feat,s) in sorted(picked.items()):
        dv = np.asarray(list(pfb[feat]), float)[-len(bts):]
        bts_f = bts[-len(dv):]
        ts_to_v = dict(zip(s["timestamp"], s["v"].astype(float)))
        col_sorted = np.sort(s["v"].to_numpy(float))

        # (A) 値メンバーシップ
        fin = dv[np.isfinite(dv)]
        idx = np.searchsorted(col_sorted, fin)
        present = np.zeros(len(fin), bool)
        for sh in (0, -1):
            j = np.clip(idx+sh, 0, len(col_sorted)-1)
            present |= np.abs(col_sorted[j]-fin) <= TOL
        memb = 100*present.mean() if len(fin) else float("nan")

        # (B) k カーブ
        curve = {}
        for k in range(-5,6):
            shift = pd.Timedelta(seconds=TF_DUR*k)
            m=t=0
            for i,b in enumerate(bts_f):
                tv = ts_to_v.get(b+shift)
                if tv is None or not np.isfinite(dv[i]) or not np.isfinite(tv): continue
                t+=1
                if abs(dv[i]-tv) <= TOL: m+=1
            curve[k] = 100*m/t if t else float("nan")

        print(f"\n  {feat}  (engine {eng})")
        print(f"    (A) 値メンバーシップ: {memb:.2f}%  "
              f"{'→ 値は本物・時刻問題' if memb>99 else ('→ 値が本当に違う' if memb<50 else '→ 一部別値')}")
        print(f"    (B) k カーブ:  " + "  ".join(
            f"{k:+d}:{curve[k]:.0f}%" if curve[k]==curve[k] else f"{k:+d}:—" for k in range(-5,6)))

    print("\n" + "="*80)
    print("  メンバーシップ~100%+kカーブにクリーンなピーク → 計測(位相)問題、値は正しい。")
    print("  メンバーシップ低 → その特徴量は production と training で本当に値が違う(真の skew)。")
    print("="*80)


if __name__ == "__main__":
    main()
