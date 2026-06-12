#!/usr/bin/env python3
"""
diagnose_liveedge_offset.py — ライブ端だけで offset を全特徴掃引

§11.30 の offset 掃引は窓全体平均で k=0 (平坦域がラグをマスク)。 ここでは
末尾 N バー (ライブ端) に限定し、 全 M3 特徴で prod[i] vs S2[bts[i]+k·dur] を
掃引する。 lag が露呈する領域なので、 真の整合 (k=0 か k=−1 か) が分かる。

  k=-1 が圧勝 → ライブ端で production が S2 より 1 バー遅れ (= ラベル規約 or stale)。
  k=0  が最良 → ラグではなく真の最新バー計算差。

使い方:
  python diagnose_liveedge_offset.py --tf M3 --last 30
"""
from __future__ import annotations
import sys, argparse, pickle
from pathlib import Path
import numpy as np, pandas as pd, polars as pl
sys.path.insert(0, "/workspace")
import blueprint as config

ENGINE_TO_UNIVERSE = {"e1a":"A","e1b":"B","e1c":"C","e1d":"D","e1e":"E","e1f":"F"}


def s2_maps(tf, needed):
    out = {}
    need = set(needed)
    for eng, u in ENGINE_TO_UNIVERSE.items():
        p = (Path(config.S2_FEATURES_VALIDATED)/f"feature_value_a_vast_universe{u}"
             /f"features_{eng}_{tf}.parquet")
        if not p.exists(): continue
        df = pl.read_parquet(p).to_pandas()
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
        for c in df.columns:
            if c == "timestamp" or c not in need: continue
            if not pd.api.types.is_numeric_dtype(df[c]): continue
            out[c] = dict(zip(df["timestamp"], df[c].astype(float)))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--state", type=Path,
                    default=config.STATE_CHECKPOINT_DIR/"feature_engine_state.pkl")
    ap.add_argument("--tf", default="M3")
    ap.add_argument("--last", type=int, default=30)
    args = ap.parse_args()
    tf = args.tf
    dur = pd.Timedelta(minutes=float(tf.replace("M","")))

    st = pickle.load(open(args.state,"rb"))
    pfb = st["proxy_feature_buffers"][tf]
    bts = [pd.Timestamp(t).tz_convert("UTC") if pd.Timestamp(t).tzinfo
           else pd.Timestamp(t).tz_localize("UTC") for t in pfb["__bar_ts__"]]
    base_feats = [f for f in pfb if f not in ("market_proxy","__bar_ts__")]
    s2 = s2_maps(tf, base_feats)

    N = args.last
    idxs = range(max(0, len(bts)-N), len(bts))

    print("="*78)
    print(f"  diagnose_liveedge_offset — TF={tf}  末尾{N}バーで全特徴 offset 掃引")
    print(f"  対象バー: {bts[max(0,len(bts)-N)]} 〜 {bts[-1]}")
    print("="*78)
    print(f"  {'shift(bar)':>11}{'<1e-6':>9}{'<1e-2':>9}{'中央|Δ|':>11}{'セル数':>9}")

    best_k, best_rate = None, -1
    for k in (-2,-1,0,1,2):
        tot = bit = w2 = 0
        ds = []
        for feat in base_feats:
            if feat not in s2: continue
            y = np.asarray(list(pfb[feat]), float)
            bf = bts[-len(y):]
            base = len(bts)-len(y)
            for i in idxs:
                j = i - base
                if j < 0 or j >= len(y): continue
                sv = s2[feat].get(bf[j] + k*dur, np.nan)
                if np.isfinite(y[j]) and np.isfinite(sv):
                    d = abs(y[j]-sv); tot += 1; ds.append(d)
                    if d < 1e-6: bit += 1
                    if d < 1e-2: w2 += 1
        if tot == 0:
            print(f"  {k:>11}        -"); continue
        r6 = 100*bit/tot; r2 = 100*w2/tot; med = np.median(ds)
        print(f"  {k:>11}{r6:>8.1f}%{r2:>8.1f}%{med:>11.4g}{tot:>9}")
        if r6 > best_rate: best_rate, best_k = r6, k

    print("="*78)
    print(f"  ライブ端の最良 shift = {best_k} bar (bit一致 {best_rate:.1f}%)")
    if best_k == -1:
        print("  → production[T] = S2[T−1]。 ライブ端で 1 バー遅れ確定。")
        print("    次: これが (a) ラベル規約のズレ(実害なし・突合を直す) か")
        print("        (b) production が stale バーで判断(実害あり) かを、 BT/production の")
        print("        判断タイミング規約で確定する。")
    elif best_k == 0:
        print("  → k=0 が最良。 ラグではなく真の最新バー計算差(不完全バー/feed)。")
    print("="*78)


if __name__ == "__main__":
    main()
