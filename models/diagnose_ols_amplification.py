#!/usr/bin/env python3
"""
diagnose_ols_amplification.py — post-OLS 乖離 = 入力差 × 桁落ち増幅 を確定 (仮説 IV)

production と batch 2_G は β 計算式・窓・proxy が同一 (コード確認済)。 ならば
post-OLS の乖離は pre-OLS 入力差 (Y) の桁落ち増幅のはず。 これを確定する:

  最新バーで、 同じ式・同じ proxy(X) を使い、
    residual_prod  = neutralize(Y=production buffer)   → snapshot を再現するはず
    residual_batch = neutralize(Y=S2 batch 値)          → S6 を再現するはず
  両者の差が pre-OLS Y 差だけで説明できれば、 機構 = 入力差の増幅で確定。
  さらに 増幅率 = post-OLS|Δ| / pre-OLS|Δ| を feature 毎に出す。

X(proxy) は production buffer の market_proxy を両方に流用 (proxy はクリーン: 最悪0.002)。

使い方:
  python diagnose_ols_amplification.py \
    --state <feature_engine_state.pkl> \
    --snapshot /workspace/.../snapshot_20260528_103001_L0.413_S0.000.csv \
    --tf M3
"""
from __future__ import annotations
import sys, argparse, pickle
from pathlib import Path
import numpy as np, pandas as pd, polars as pl
sys.path.insert(0, "/workspace")
import blueprint as config

ENGINE_TO_UNIVERSE = {"e1a":"A","e1b":"B","e1c":"C","e1d":"D","e1e":"E","e1f":"F"}
EPS = 1e-10


def neutralize_last(x_arr, y_arr):
    """production / 2_G と同一式で最新バーの残差を返す + 診断量。"""
    x = np.asarray(x_arr, float); y = np.asarray(y_arr, float)
    m = np.isfinite(x) & np.isfinite(y)
    x, y = x[m], y[m]
    n = len(x)
    if n < 30: return None
    mean_x = x.mean(); var_x = max(0.0, (x*x).mean() - mean_x*mean_x)
    mean_y = y.mean(); cov_xy = (x*y).mean() - mean_x*mean_y
    beta = cov_xy / (var_x + EPS); alpha = mean_y - beta*mean_x
    x_last, y_last = x[-1], y[-1]
    fitted = beta*x_last + alpha
    resid = y_last - fitted
    return dict(beta=beta, alpha=alpha, resid=resid, y_last=y_last,
                fitted=fitted, n=n)


def s2_maps(tf, needed):
    """S2 の各 base feature について ts->value の dict を返す。 needed のみ・数値列のみ。"""
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
    ap.add_argument("--snapshot", type=Path, default=None)
    ap.add_argument("--tf", default="M3")
    ap.add_argument("--top", type=int, default=25)
    args = ap.parse_args()
    tf = args.tf

    st = pickle.load(open(args.state, "rb"))
    pfb = st["proxy_feature_buffers"][tf]
    x_arr = list(pfb["market_proxy"])
    bts = [pd.Timestamp(t).tz_convert("UTC") if pd.Timestamp(t).tzinfo
           else pd.Timestamp(t).tz_localize("UTC") for t in pfb["__bar_ts__"]]

    snap = {}
    if args.snapshot and args.snapshot.exists():
        sdf = pd.read_csv(args.snapshot)
        snap = {str(r["feature_name"]).strip(): pd.to_numeric(r["value"], errors="coerce")
                for _, r in sdf.iterrows()}

    print("="*92)
    print(f"  diagnose_ols_amplification — TF={tf}  入力差×桁落ち増幅の確定")
    print(f"  窓長 {len(x_arr)}  最新バー {bts[-1] if bts else '?'}")
    print("="*92)

    s2 = None  # 後で base_feats 確定後にロード
    base_feats = [f for f in pfb if f not in ("market_proxy","__bar_ts__")]
    s2 = s2_maps(tf, base_feats)

    rows = []
    sanity = []
    for feat in base_feats:
        y_prod = list(pfb[feat])
        if feat not in s2:  # S2 に base が無い
            continue
        # 同窓の S2 値を __bar_ts__ で取得
        y_batch = [s2[feat].get(t, np.nan) for t in bts[-len(y_prod):]]
        xw = x_arr[-len(y_prod):]
        rp = neutralize_last(xw, y_prod)
        rb = neutralize_last(xw, y_batch)
        if rp is None or rb is None: continue
        pre = abs(rp["y_last"] - rb["y_last"])          # pre-OLS 入力差
        post = abs(rp["resid"] - rb["resid"])           # post-OLS 残差差
        amp = post/pre if pre > 1e-12 else float('nan') # 増幅率
        canc = abs(rp["resid"]) / (abs(rp["y_last"]) + EPS)  # 桁落ち比 (小=激しい)
        rows.append((feat, post, pre, amp, canc, rp["resid"], rb["resid"]))
        # sanity: snapshot の neutralized 値と residual_prod を照合
        nm = f"{feat}_neutralized_{tf}"
        if nm in snap and np.isfinite(snap[nm]):
            sanity.append((nm, rp["resid"], snap[nm]))

    rows.sort(key=lambda r: r[1], reverse=True)

    # sanity 表示
    if sanity:
        print("  【sanity: residual_prod vs snapshot (再構成の正しさ)】")
        ok = 0
        for nm, rp, sv in sanity[:6]:
            mark = "✓" if abs(rp-sv) < 1e-3 else "✗"
            if abs(rp-sv) < 1e-3: ok += 1
            print(f"    {mark} {nm:<46} 再構成={rp:>10.4f}  snapshot={sv:>10.4f}")
        print(f"    一致 {sum(1 for nm,rp,sv in sanity if abs(rp-sv)<1e-3)}/{len(sanity)} "
              f"(これが揃えば私の再構成が production と一致 = 以降の議論が有効)\n")

    print(f"  【post-OLS|Δ| 降順 TOP{args.top}: 入力Y差(pre) → 残差差(post) と増幅率】")
    print(f"  {'feature':<44}{'post|Δ|':>9}{'pre|Δ|':>9}{'増幅':>7}{'桁落ち比':>8}")
    for feat, post, pre, amp, canc, rpv, rbv in rows[:args.top]:
        print(f"  {feat:<44}{post:>9.3g}{pre:>9.3g}{amp:>7.1f}{canc:>8.3g}")

    if rows:
        amps = [r[3] for r in rows if np.isfinite(r[3])]
        print(f"\n  増幅率 中央値 {np.median(amps):.1f}  最大 {np.max(amps):.0f}  "
              f"(>1 = 桁落ちで pre-OLS 差が拡大)")
        print(f"  → residual_prod が snapshot を再現し、 post|Δ| ≈ pre|Δ|×増幅 で説明できれば、")
        print(f"     機構 = 入力差の桁落ち増幅で確定。 入力(pre-OLS)を詰めれば post も減る。")
    print("="*92)


if __name__ == "__main__":
    main()
