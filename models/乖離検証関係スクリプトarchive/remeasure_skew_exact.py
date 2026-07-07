#!/usr/bin/env python3
"""
remeasure_skew_exact.py — __bar_ts__ exact-timestamp 計測 (正しいオフセット + 複数 tol)

[確定したオフセット規約 (diagnose_offset_sweep / feature_correspondence より)]
  - Y (pre-OLS 特徴量): label = __bar_ts__ (k=0)。 __bar_ts__ は bar start ラベル = S2 label=left と一致。
  - X (market_proxy):   label = __bar_ts__ + 300s (= M5 1本)。 proxy は M5 proxy で 1 本位相がある。
                         M3/M8/M15 は 300s がバー境界に乗らないため部分的にしか合わない(ffill マッピング差)。

[複数 tol]
  1e-6 (bit厳格) / 1e-4 / 1e-2 で一致率を出す。 tol を緩めて跳ねれば残差は極小 (warmup/丸め)、
  緩めても低いままなら真の値差。

使い方:
  python remeasure_skew_exact.py --state <state.pkl> --train-proxy-dir <dir>
"""
from __future__ import annotations
import sys, argparse, pickle, warnings
from pathlib import Path
from typing import Dict, Optional, Tuple
import numpy as np, pandas as pd, polars as pl
warnings.filterwarnings("ignore", category=FutureWarning)
sys.path.insert(0, "/workspace")
import blueprint as config

ENGINE_TO_UNIVERSE = {"e1a":"A","e1b":"B","e1c":"C","e1d":"D","e1e":"E","e1f":"F"}
TF_DUR_SEC = {"M0.5":30,"M1":60,"M3":180,"M5":300,"M8":480,"M15":900}
SKIP_KEYS = {"market_proxy","__bar_ts__"}
PROXY_OFFSET_SEC = 300   # X: market_proxy は M5 proxy → +1 M5 bar
TOLS = [1e-6, 1e-4, 1e-2]
_S2: Dict[Tuple[str,str], Optional[Tuple[dict,pd.DataFrame]]] = {}


def get_s2(engine, tf):
    key = (engine, tf)
    if key in _S2: return _S2[key]
    u = ENGINE_TO_UNIVERSE.get(engine)
    p = (Path(config.S2_FEATURES_VALIDATED)/f"feature_value_a_vast_universe{u}"
         /f"features_{engine}_{tf}.parquet") if u else None
    if p is None or not p.exists():
        _S2[key] = None; return None
    df = pl.read_parquet(p).to_pandas()
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    _S2[key] = ({t:i for i,t in enumerate(df["timestamp"])}, df)
    return _S2[key]


def train_proxy_map(tp_dir, tf):
    for name in (f"train_proxy_{tf}.parquet", f"train_proxy_{tf.replace('.','')}.parquet"):
        p = tp_dir / name
        if p.exists():
            df = pl.read_parquet(p).to_pandas()
            df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
            vcol = [c for c in df.columns if c != "timestamp"][0]
            return dict(zip(df["timestamp"], df[vcol].astype(float)))
    return None


def labels(bar_ts_full, n, offset_sec):
    bts = list(bar_ts_full)[-n:]
    off = pd.Timedelta(seconds=offset_sec)
    out = []
    for t in bts:
        if t is None: out.append(None); continue
        ts = pd.Timestamp(t)
        ts = ts.tz_localize("UTC") if ts.tzinfo is None else ts.tz_convert("UTC")
        out.append(ts + off)
    return out


def compare_idxcol(dvals, bar_ts, offset_sec, ts_to_idx, col):
    dv = np.asarray(list(dvals), float)
    labs = labels(bar_ts, len(dv), offset_sec)
    diffs = []
    for i, lab in enumerate(labs):
        if lab is None: continue
        pos = ts_to_idx.get(lab)
        if pos is None: continue
        tv = float(col[pos])
        if np.isfinite(dv[i]) and np.isfinite(tv):
            diffs.append(abs(dv[i]-tv))
    return np.asarray(diffs)


def compare_map(dvals, bar_ts, offset_sec, ref):
    dv = np.asarray(list(dvals), float)
    labs = labels(bar_ts, len(dv), offset_sec)
    diffs = []
    for i, lab in enumerate(labs):
        if lab is None: continue
        tv = ref.get(lab)
        if tv is not None and np.isfinite(dv[i]) and np.isfinite(tv):
            diffs.append(abs(dv[i]-tv))
    return np.asarray(diffs)


def rates(diffs):
    if len(diffs) == 0: return None
    return [100*np.mean(diffs < t) for t in TOLS], len(diffs), float(diffs.max())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--state", type=Path, default=config.STATE_CHECKPOINT_DIR/"feature_engine_state.pkl")
    ap.add_argument("--train-proxy-dir", type=Path, default=config.DATA_DIR/"diagnostics"/"train_proxy")
    args = ap.parse_args()

    dump = pickle.load(open(args.state, "rb"))
    pfb = dump["proxy_feature_buffers"]

    print("="*92)
    print("  remeasure_skew_exact — 正しいオフセット (Y:k=0, X:+300s) + 複数 tol")
    print(f"  source: {Path(args.state).name}")
    print("="*92)
    th = "".join(f"<{t:g}".rjust(10) for t in TOLS)
    print(f"\n  【Y: pre-OLS 特徴量 (label=__bar_ts__)】")
    print(f"  {'TF':<6}{'feat':>6}{'S2欠':>5}{'セル数':>9}{th}{'最悪|diff|':>13}")
    for tf in sorted(pfb.keys()):
        bts = pfb[tf].get("__bar_ts__")
        if not bts: continue
        all_d = []; nfeat = nmiss = 0
        for feat in pfb[tf]:
            if feat in SKIP_KEYS: continue
            eng = feat.split("_")[0]
            s2 = get_s2(eng, tf)
            if s2 is None: continue
            ts_to_idx, df = s2
            if feat not in df.columns: nmiss += 1; continue
            d = compare_idxcol(pfb[tf][feat], bts, 0, ts_to_idx, df[feat].to_numpy(float))
            if len(d): nfeat += 1; all_d.append(d)
        if all_d:
            D = np.concatenate(all_d); r = rates(D)
            rr = "".join(f"{x:8.1f}%" for x in r[0])
            print(f"  {tf:<6}{nfeat:>6}{nmiss:>5}{r[1]:>9}{rr}{r[2]:>13.4g}")

    print(f"\n  【X: market_proxy vs train_proxy (label=__bar_ts__ +300s)】")
    print(f"  {'TF':<6}{'セル数':>9}{th}{'最悪|diff|':>13}")
    if Path(args.train_proxy_dir).exists():
        for tf in sorted(pfb.keys()):
            bts = pfb[tf].get("__bar_ts__")
            if not bts or "market_proxy" not in pfb[tf]: continue
            ref = train_proxy_map(args.train_proxy_dir, tf)
            if ref is None: print(f"  {tf:<6} train_proxy 無し"); continue
            d = compare_map(pfb[tf]["market_proxy"], bts, PROXY_OFFSET_SEC, ref)
            r = rates(d)
            if r:
                rr = "".join(f"{x:8.1f}%" for x in r[0])
                print(f"  {tf:<6}{r[1]:>9}{rr}{r[2]:>13.4g}")
            else:
                print(f"  {tf:<6}{0:>9}   (時刻被りなし: +300s が境界に乗らない TF)")

    print("\n" + "="*92)
    print("  tol を緩めて率が跳ねる → 残差は極小 (warmup収束/丸め)。 緩めても低い → 真の値差。")
    print("  M3/M8/M15 の X は +300s がバー境界に乗らず低い(ffillマッピング差。値自体は ~0.002-0.006)。")
    print("="*92)


if __name__ == "__main__":
    main()
