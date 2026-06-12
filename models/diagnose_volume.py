#!/usr/bin/env python3
"""
diagnose_volume.py — volume 族の系統差が「スケール差」か「別データ」か特定

production の volume 系特徴量 vs S2 を exact-ts (k=0) で比較し:
  - 差のあるバーで prod/train の比率が ~一定か (= スケール差: tick vs real volume) を見る
  - 価格系特徴量 (atr) と並べて「価格は合うが volume だけズレる」を確認
  - 例を表示

使い方:
  python diagnose_volume.py            # 既定 M5
  python diagnose_volume.py --tf M0.5
"""
from __future__ import annotations
import sys, argparse, pickle
from pathlib import Path
import numpy as np, pandas as pd, polars as pl
sys.path.insert(0, "/workspace")
import blueprint as config

ENGINE_TO_UNIVERSE = {"e1a":"A","e1b":"B","e1c":"C","e1d":"D","e1e":"E","e1f":"F"}
# 確認したい代表 (volume 系 + 対照の価格系)
TARGETS = ["e1b_volume_ma20", "e1d_volume_ratio", "e1d_obv_rel",
           "e1a_fast_volume_mean_20", "e1c_atr_13"]


def s2_map(engine, tf, feat):
    u = ENGINE_TO_UNIVERSE.get(engine)
    if u is None: return None
    p = (Path(config.S2_FEATURES_VALIDATED)/f"feature_value_a_vast_universe{u}"
         /f"features_{engine}_{tf}.parquet")
    if not p.exists(): return None
    df = pl.read_parquet(p).to_pandas()
    if feat not in df.columns: return None
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    return dict(zip(df["timestamp"], df[feat].astype(float)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tf", default="M5")
    args = ap.parse_args()
    tf = args.tf

    st = pickle.load(open(config.STATE_CHECKPOINT_DIR/"feature_engine_state.pkl","rb"))
    pfb = st["proxy_feature_buffers"][tf]
    bts = [pd.Timestamp(t).tz_convert("UTC") if pd.Timestamp(t).tzinfo
           else pd.Timestamp(t).tz_localize("UTC") for t in pfb["__bar_ts__"]]

    # deque にある volume 系の実名を拾う (TARGETS は候補、実在するものだけ)
    avail = [f for f in pfb if f not in ("market_proxy","__bar_ts__")]
    targets = [t for t in TARGETS if t in avail]
    # 候補が無ければ volume を含む名前を数件
    if len([t for t in targets if "volume" in t or "obv" in t]) < 2:
        targets += [f for f in avail if ("volume" in f or "obv" in f)][:3]
    targets = list(dict.fromkeys(targets))

    print("="*84)
    print(f"  diagnose_volume — TF={tf}  volume 系の系統差を特定 (k=0)")
    print("="*84)

    for feat in targets:
        eng = feat.split("_")[0]
        ref = s2_map(eng, tf, feat)
        if ref is None:
            print(f"\n  {feat}: S2 に無し (skip)"); continue
        dv = np.asarray(list(pfb[feat]), float)[-len(bts):]
        bf = bts[-len(dv):]
        prod, train = [], []
        for i,b in enumerate(bf):
            tv = ref.get(b)
            if tv is None or not np.isfinite(dv[i]) or not np.isfinite(tv): continue
            prod.append(dv[i]); train.append(tv)
        prod, train = np.asarray(prod), np.asarray(train)
        if len(prod)==0:
            print(f"\n  {feat}: 共通時刻なし"); continue
        d = np.abs(prod-train)
        match = 100*np.mean(d < 1e-6)
        # 比率 (train!=0 のみ)
        nz = np.abs(train) > 1e-9
        ratio = prod[nz]/train[nz] if nz.any() else np.array([])
        kind = "価格系(対照)" if ("volume" not in feat and "obv" not in feat) else "volume系"
        print(f"\n  [{kind}] {feat}")
        print(f"    bit-identical率: {match:.1f}%   共通: {len(prod)}")
        if len(ratio):
            print(f"    prod/train 比: median={np.median(ratio):.4f}  "
                  f"mean={np.mean(ratio):.4f}  std={np.std(ratio):.4f}  "
                  f"[{np.percentile(ratio,5):.3f}, {np.percentile(ratio,95):.3f}]")
            print(f"    → 比が ~一定なら スケール差 / バラつけば 別データ")
        # 差のある例
        bigidx = np.where(d >= 1e-2)[0][:5]
        for j in bigidx:
            r = prod[j]/train[j] if abs(train[j])>1e-9 else float('nan')
            print(f"      prod={prod[j]:.5g}  train={train[j]:.5g}  ratio={r:.4f}  diff={prod[j]-train[j]:+.4g}")

    print("\n" + "="*84)
    print("  比が全 volume 特徴で ~同じ定数 → tick vs real volume のスケール差 (理解可能・対処容易)")
    print("  比がバラバラ → 生 volume のパターン自体が feed 間で違う (#68 feed divergence)")
    print("  価格系(atr) が bit-identical なら『価格は合うが volume だけ別』が確定")
    print("="*84)


if __name__ == "__main__":
    main()
