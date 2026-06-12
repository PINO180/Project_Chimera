#!/usr/bin/env python3
"""
diagnose_pred_baroffset.py — 1バーズレを予測レベルで確定

production の snapshot 予測 (_m1_long=0.5803) を、 S6 の各バー (10:18..10:33) で
M1 long モデルを回した予測と突合。 どの S6 バーが 0.5803 を再現するかを見る。

  S6[T-1]=10:27 が prod の 0.5803 を再現 → 予測乖離は 1 バー整合ズレ (物差し)。
                                           実 skew は lag-1 整合後の残差 (~2% volume) のみ。
  どの S6 バーも再現しない → 1 バーでは説明できない真の乖離。

使い方:
  python diagnose_pred_baroffset.py \
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
        (meta if k.startswith("_") else feats).__setitem__(k, v)
    feats = {k: pd.to_numeric(v, errors="coerce") for k,v in feats.items()}
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
        d[tsc] = pd.to_datetime(d[tsc], utc=True, errors="coerce")
        d = d[(d[tsc]>=t0)&(d[tsc]<=t1)]
        if len(d): frames.append(d)
    if not frames: return None, tsc
    return pd.concat(frames, ignore_index=True), tsc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--snapshot", type=Path, required=True)
    ap.add_argument("--s6-dir", type=Path, default=Path(config.S6_WEIGHTED_DATASET))
    ap.add_argument("--tf", default="M3")
    args = ap.parse_args()
    dur = pd.Timedelta(minutes=float(args.tf.replace("M","")))

    meta, feats = parse_snapshot(args.snapshot)
    ts = pd.to_datetime(meta.get("_timestamp_utc"), utc=True)
    bar = ts.floor("min")
    prod_pred = float(meta.get("_m1_long"))

    m = joblib.load(config.S7_M1_MODEL_LONG_PKL)
    b = m.booster_ if hasattr(m,"booster_") else m
    order = b.feature_name()

    # prod ベクトル (snapshot) で sanity
    Xp = np.array([[float(feats.get(f,0.0)) for f in order]], dtype=np.float32)
    pp = float(b.predict(Xp)[0])

    print("="*78)
    print(f"  diagnose_pred_baroffset — TF={args.tf}  snapshot bar≈{bar}")
    print(f"  production _m1_long (log)        = {prod_pred:.4f}")
    print(f"  production features → M1再現      = {pp:.4f}  (snapshot 再構成 sanity)")
    print("="*78)

    s6, tsc = load_s6_window(args.s6_dir, bar-5*dur, bar+3*dur)
    if s6 is None:
        print(f"  S6 読込失敗 ({args.s6_dir})"); return
    s6 = s6.drop_duplicates(tsc).set_index(tsc).sort_index()

    print(f"  S6 各バーで M1 long を推論 → prod {prod_pred:.4f} を再現するバーを探す:")
    print(f"  {'S6 bar':<22}{'shift':>7}{'M1 long':>10}{'|Δ vs prod|':>13}")
    best=None
    for k in (-3,-2,-1,0,1):
        tb = bar + k*dur
        if tb not in s6.index: 
            print(f"  {str(tb):<22}{k:>7}   (S6に無)"); continue
        row = s6.loc[tb]
        Xs = np.array([[float(pd.to_numeric(row[f], errors='coerce')) if f in row.index else 0.0
                        for f in order]], dtype=np.float32)
        ps = float(b.predict(Xs)[0])
        d = abs(ps - prod_pred)
        mark = "  ←最近" if (best is None or d < best[1]) else ""
        if best is None or d < best[1]: best=(k,d,ps)
        print(f"  {str(tb):<22}{k:>7}{ps:>10.4f}{d:>13.4f}{mark}")

    print("="*78)
    if best:
        k,d,ps = best
        print(f"  最も prod を再現する S6 バー: shift {k} (M1={ps:.4f}, |Δ|={d:.4f})")
        if k == -1:
            print("  → prod[T] = S6[T−1] を予測レベルで確認。 0.58↔0.25 は 1 バー整合ズレ。")
            print("    実 skew は lag-1 整合後の残差 (~2% volume) のみ = compare_predictions を")
            print("    1 バー直して再評価すべき。")
        elif k == 0:
            print("  → S6[T] が最良。 1 バーでは説明できず、 最新バー計算差が残る。")
    print("="*78)


if __name__ == "__main__":
    main()
