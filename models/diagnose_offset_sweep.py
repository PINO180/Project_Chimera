#!/usr/bin/env python3
"""
diagnose_offset_sweep.py — label オフセットを掃引し「規約ズレ」か「真の値差」かを切り分ける

remeasure_skew_exact が X=0%・Y~10% を出した原因を確定する。
  各 deque 要素 i の照合時刻を label = __bar_ts__[i] + k*tf_dur (k=-3..+3) と振り、
  X(market_proxy vs train_proxy) と 代表 Y 特徴量の一致率がどの k で最大化するか見る。

判定:
  - ある k で X が ~100% に跳ねる → 私の ruler の label 規約が k 本ズレていただけ (= アーティファクト)。
  - どの k でも低い → 値が本当に違う (emit_train_proxy が別物 か 真の skew)。

使い方:
  python diagnose_offset_sweep.py
"""
from __future__ import annotations
import sys, pickle
from pathlib import Path
import numpy as np, pandas as pd, polars as pl
sys.path.insert(0, "/workspace")
import blueprint as config

ENGINE_TO_UNIVERSE = {"e1a":"A","e1b":"B","e1c":"C","e1d":"D","e1e":"E","e1f":"F"}
TF_DUR = {"M0.5":30,"M1":60,"M3":180,"M5":300,"M8":480,"M15":900}
TP_DIR = config.DATA_DIR / "diagnostics" / "train_proxy"
TOL = 1e-6


def proxy_map(tf):
    p = TP_DIR / f"train_proxy_{tf}.parquet"
    if not p.exists(): return None
    df = pl.read_parquet(p).to_pandas()
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    vcol = [c for c in df.columns if c != "timestamp"][0]
    return dict(zip(df["timestamp"], df[vcol].astype(float)))


def s2_col(engine, tf, feat):
    u = ENGINE_TO_UNIVERSE.get(engine)
    if u is None:
        return None  # PROXY_FEATURES (atr/log_return/...) は engine 特徴量でない → skip
    p = (Path(config.S2_FEATURES_VALIDATED)/f"feature_value_a_vast_universe{u}"
         /f"features_{engine}_{tf}.parquet")
    if not p.exists(): return None
    df = pl.read_parquet(p).to_pandas()
    if feat not in df.columns: return None
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    return dict(zip(df["timestamp"], df[feat].astype(float)))


def sweep(dvals, bts, tf_dur, ref, tol=TOL, ks=range(-3,16)):
    dv = np.asarray(list(dvals), float)
    bts = list(bts)[-len(dv):]
    base = [pd.Timestamp(t).tz_convert("UTC") if pd.Timestamp(t).tzinfo
            else pd.Timestamp(t).tz_localize("UTC") for t in bts]
    out = {}
    for k in ks:
        shift = pd.Timedelta(seconds=tf_dur*k)
        m = t = 0; worst = 0.0
        for i, b in enumerate(base):
            tv = ref.get(b + shift)
            if tv is None or not np.isfinite(dv[i]) or not np.isfinite(tv): continue
            t += 1; d = abs(dv[i]-tv)
            if d < tol: m += 1
            if d > worst: worst = d
        out[k] = (100*m/t if t else float("nan"), t, worst)
    return out


def main():
    st = pickle.load(open(config.STATE_CHECKPOINT_DIR/"feature_engine_state.pkl","rb"))
    pfb = st["proxy_feature_buffers"]

    print("="*78)
    print("  diagnose_offset_sweep — label を k 本ずらした時の一致率 (k: 照合相手の世代)")
    print("  注: label = __bar_ts__ + k*tf_dur。 k=-1 は現規約(=__bar_ts__-tf_dur)からさらに -1 本ではなく")
    print("      __bar_ts__ そのもの基準で k 本ずらした絶対位置。 ~100% の k が真の規約。")
    print("="*78)

    print("\n【X: market_proxy vs train_proxy】 最良オフセット k と一致率 (label=__bar_ts__ + k*tf_dur)")
    print(f"  {'TF':<6}{'best_k':>8}{'一致率':>10}{'セル数':>9}{'最悪|diff|':>12}   (k=+1 の率も併記)")
    for tf in ["M0.5","M1","M3","M5","M8","M15"]:
        if tf not in pfb or "market_proxy" not in pfb[tf]: continue
        ref = proxy_map(tf)
        if ref is None:
            print(f"  {tf:<6} train_proxy 無し"); continue
        res = sweep(pfb[tf]["market_proxy"], pfb[tf]["__bar_ts__"], TF_DUR[tf], ref)
        best_k = max(res, key=lambda k: (res[k][0] if res[k][0]==res[k][0] else -1))
        br, bn, bw = res[best_k]
        k1 = res.get(1, (float('nan'),0,0))[0]
        print(f"  {tf:<6}{best_k:>+8d}{br:>9.1f}%{bn:>9}{bw:>12.4g}   (k=+1: {k1:.1f}%)")

    print("\n【Y: 代表特徴量 (TF=M5 固定, engine 別に1つ)】 最良オフセット k と一致率")
    tf = "M5"
    bts = pfb[tf]["__bar_ts__"]
    picked = {}
    for feat in pfb[tf]:
        if feat in ("market_proxy","__bar_ts__"): continue
        eng = feat.split("_")[0]
        if eng not in ENGINE_TO_UNIVERSE or eng in picked: continue
        ref = s2_col(eng, tf, feat)
        if ref: picked[eng] = (feat, ref)
    print(f"  {'feat':<36}{'best_k':>8}{'一致率':>10}{'セル数':>9}{'最悪|diff|':>12}")
    for eng,(feat,ref) in sorted(picked.items()):
        res = sweep(pfb[tf][feat], bts, TF_DUR[tf], ref)
        best_k = max(res, key=lambda k: (res[k][0] if res[k][0]==res[k][0] else -1))
        br, bn, bw = res[best_k]
        print(f"  {feat:<36}{best_k:>+8d}{br:>9.1f}%{bn:>9}{bw:>12.4g}")

    print("\n" + "="*78)
    print("  読み: ある k で ~100% → 私の ruler が k 本ズレていただけ (アーティファクト、即修正)。")
    print("        どの k でも低い → 値が本当に違う (emit_train_proxy 別物 or 真の skew)。")
    print("="*78)


if __name__ == "__main__":
    main()
