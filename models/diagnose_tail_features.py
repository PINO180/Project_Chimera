#!/usr/bin/env python3
"""
diagnose_tail_features.py — Y の ~4% 巨大差テールがどの特徴量か特定する

各 TF で feature ごとに (k=0, label=__bar_ts__):
  - 最悪|diff|
  - tol=1e-2 を超えるセルの割合 (= 巨大差率)
を出し、 最悪 diff 降順で TOP を表示。 高次モーメント等に集中しているか確認する。
あわせて feature ファミリー (名前の接頭) 別に巨大差率を集計。

使い方:
  python diagnose_tail_features.py            # 既定 M0.5, M5
  python diagnose_tail_features.py --tf M5
"""
from __future__ import annotations
import sys, argparse, pickle, re
from pathlib import Path
from collections import defaultdict
import numpy as np, pandas as pd, polars as pl
sys.path.insert(0, "/workspace")
import blueprint as config

ENGINE_TO_UNIVERSE = {"e1a":"A","e1b":"B","e1c":"C","e1d":"D","e1e":"E","e1f":"F"}
SKIP = {"market_proxy","__bar_ts__"}
BIG = 1e-2


def get_s2(engine, tf):
    u = ENGINE_TO_UNIVERSE.get(engine)
    if u is None: return None
    p = (Path(config.S2_FEATURES_VALIDATED)/f"feature_value_a_vast_universe{u}"
         /f"features_{engine}_{tf}.parquet")
    if not p.exists(): return None
    df = pl.read_parquet(p).to_pandas()
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    return {t:i for i,t in enumerate(df["timestamp"])}, df


def family(feat):
    # 末尾の数字を畳んでファミリー名に (e1a_statistical_moment_8_50 -> e1a_statistical_moment)
    base = feat.split("_")
    out = []
    for tok in base:
        if re.fullmatch(r"[0-9.]+", tok): break
        out.append(tok)
    return "_".join(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tf", nargs="*", default=["M0.5","M5"])
    ap.add_argument("--top", type=int, default=20)
    args = ap.parse_args()

    st = pickle.load(open(config.STATE_CHECKPOINT_DIR/"feature_engine_state.pkl","rb"))

    for tf in args.tf:
        pfb = st["proxy_feature_buffers"].get(tf)
        if not pfb or "__bar_ts__" not in pfb: continue
        bts = [pd.Timestamp(t).tz_convert("UTC") if pd.Timestamp(t).tzinfo
               else pd.Timestamp(t).tz_localize("UTC") for t in pfb["__bar_ts__"]]
        rows = []
        fam_big = defaultdict(lambda: [0,0])  # family -> [big_cells, total_cells]
        s2cache = {}
        for feat in pfb:
            if feat in SKIP: continue
            eng = feat.split("_")[0]
            if eng not in ENGINE_TO_UNIVERSE: continue
            if eng not in s2cache: s2cache[eng] = get_s2(eng, tf)
            if s2cache[eng] is None: continue
            ts_to_idx, df = s2cache[eng]
            if feat not in df.columns: continue
            col = df[feat].to_numpy(float)
            dv = np.asarray(list(pfb[feat]), float)[-len(bts):]
            bf = bts[-len(dv):]
            diffs = []
            for i,b in enumerate(bf):
                pos = ts_to_idx.get(b)
                if pos is None: continue
                tv = col[pos]
                if np.isfinite(dv[i]) and np.isfinite(tv):
                    diffs.append(abs(dv[i]-tv))
            if not diffs: continue
            d = np.asarray(diffs)
            nbig = int((d>=BIG).sum())
            rows.append((feat, d.max(), 100*nbig/len(d), len(d)))
            fam = family(feat)
            fam_big[fam][0] += nbig; fam_big[fam][1] += len(d)

        rows.sort(key=lambda r: r[1], reverse=True)
        print("="*84)
        print(f"  TF={tf}  最悪|diff| 降順 TOP{args.top} (k=0)")
        print("="*84)
        print(f"  {'feature':<42}{'最悪|diff|':>13}{'>1e-2率':>10}{'セル':>8}")
        for feat,mx,bigpct,n in rows[:args.top]:
            print(f"  {feat:<42}{mx:>13.4g}{bigpct:>9.1f}%{n:>8}")

        print(f"\n  ファミリー別 >1e-2 率 (巨大差が集中している族, TOP10):")
        fam_sorted = sorted(fam_big.items(),
                            key=lambda kv: (kv[1][0]/kv[1][1] if kv[1][1] else 0), reverse=True)
        for fam,(big,tot) in fam_sorted[:10]:
            if tot and big:
                print(f"    {fam:<40} {100*big/tot:>5.1f}%  ({big}/{tot})")
        print()


if __name__ == "__main__":
    main()
