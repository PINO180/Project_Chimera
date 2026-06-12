#!/usr/bin/env python3
"""
diagnose_lastbar.py — 乖離はライブ端の最新バーに局所化しているか確定

amp≈1.0 (diagnose_ols_amplification) は「窓の古いバーは一致・最新バーだけ乖離」を示唆。
これを per-bar で直接確認する:
  - 代表 feature について production buffer Y vs S2 Y を __bar_ts__ で全バー突合
  - 窓全体の一致率 と 末尾 N バーの (ts, prod, S2, |Δ|) を表示
  - 最新バーを S2 の ts-1/ts/ts+1 と比較 (off-by-one か、 真の最新バー計算差か)

使い方:
  python diagnose_lastbar.py --tf M3
"""
from __future__ import annotations
import sys, argparse, pickle
from pathlib import Path
import numpy as np, pandas as pd, polars as pl
sys.path.insert(0, "/workspace")
import blueprint as config

ENGINE_TO_UNIVERSE = {"e1a":"A","e1b":"B","e1c":"C","e1d":"D","e1e":"E","e1f":"F"}
TARGETS = ["e1a_statistical_moment_8_20", "e1c_aroon_oscillator_14",
           "e1d_force_index_norm", "e1a_statistical_mean_10",
           "e1c_williams_r_14", "e1d_obv_rel"]


def s2_map_one(tf, feat):
    eng = feat.split("_")[0]; u = ENGINE_TO_UNIVERSE.get(eng)
    if u is None: return None
    p = (Path(config.S2_FEATURES_VALIDATED)/f"feature_value_a_vast_universe{u}"
         /f"features_{eng}_{tf}.parquet")
    if not p.exists() or feat is None: return None
    df = pl.read_parquet(p).to_pandas()
    if feat not in df.columns: return None
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    return dict(zip(df["timestamp"], df[feat].astype(float)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--state", type=Path,
                    default=config.STATE_CHECKPOINT_DIR/"feature_engine_state.pkl")
    ap.add_argument("--tf", default="M3")
    ap.add_argument("--tail", type=int, default=8)
    args = ap.parse_args()
    tf = args.tf

    st = pickle.load(open(args.state,"rb"))
    pfb = st["proxy_feature_buffers"][tf]
    bts = [pd.Timestamp(t).tz_convert("UTC") if pd.Timestamp(t).tzinfo
           else pd.Timestamp(t).tz_localize("UTC") for t in pfb["__bar_ts__"]]
    dur = pd.Timedelta(minutes=float(tf.replace("M","")))

    print("="*88)
    print(f"  diagnose_lastbar — TF={tf}  乖離のライブ端局所化を確認")
    print(f"  窓長 {len(bts)}  最新バー __bar_ts__[-1] = {bts[-1]}")
    print("="*88)

    avail = [f for f in TARGETS if f in pfb]
    avail += [f for f in pfb if f not in ("market_proxy","__bar_ts__") and f not in avail][:0]

    for feat in avail:
        ref = s2_map_one(tf, feat)
        if ref is None:
            print(f"\n  {feat}: S2 無し"); continue
        y = np.asarray(list(pfb[feat]), float)
        bf = bts[-len(y):]
        diffs = []
        for i,t in enumerate(bf):
            sv = ref.get(t, np.nan)
            if np.isfinite(y[i]) and np.isfinite(sv):
                diffs.append(abs(y[i]-sv))
        diffs = np.asarray(diffs)
        # 窓全体(末尾Nを除く)の一致率
        if len(diffs) > args.tail:
            older = diffs[:-args.tail]
            rate_old = 100*np.mean(older < 1e-6)
            med_old = np.median(older)
        else:
            rate_old, med_old = float('nan'), float('nan')
        print(f"\n  [{feat}]")
        print(f"    古いバー(末尾{args.tail}除く {len(diffs)-args.tail}本): bit一致 {rate_old:.1f}% / 中央|Δ| {med_old:.3g}")
        print(f"    末尾{args.tail}バー:")
        print(f"      {'ts':<22}{'prod':>12}{'S2':>12}{'|Δ|':>11}")
        for i in range(max(0,len(bf)-args.tail), len(bf)):
            sv = ref.get(bf[i], np.nan)
            dd = abs(y[i]-sv) if (np.isfinite(y[i]) and np.isfinite(sv)) else float('nan')
            print(f"      {str(bf[i]):<22}{y[i]:>12.4f}{sv:>12.4f}{dd:>11.4g}")
        # off-by-one チェック: 最新 prod を S2 の ts-1/ts/ts+1 と比較
        tlast = bf[-1]; ylast = y[-1]
        s_m1 = ref.get(tlast-dur, np.nan); s_0 = ref.get(tlast, np.nan); s_p1 = ref.get(tlast+dur, np.nan)
        print(f"    最新バー off-by-one チェック (prod[-1]={ylast:.4f}):")
        print(f"      S2[{(tlast-dur).time()}]={s_m1:.4f}(|Δ|{abs(ylast-s_m1):.3g}) "
              f"S2[{tlast.time()}]={s_0:.4f}(|Δ|{abs(ylast-s_0):.3g}) "
              f"S2[{(tlast+dur).time()}]={s_p1:.4f}(|Δ|{abs(ylast-s_p1):.3g})")

    print("\n" + "="*88)
    print("  古いバーが bit一致 ~100% & 末尾(特に最新)だけ乖離 → ライブ端の最新バー計算差で確定。")
    print("  最新 prod が S2 の隣 ts と一致 → off-by-one。 どの ts とも不一致 → 不完全バー/feed 差。")
    print("="*88)


if __name__ == "__main__":
    main()
