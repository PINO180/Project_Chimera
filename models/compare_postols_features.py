#!/usr/bin/env python3
"""
compare_postols_features.py — post-OLS(純化後)特徴量の train-serve 突合 (局所化)

production snapshot (縦持ち feature_name,value / 全て _neutralized_ = post-OLS)
  vs  S6_WEIGHTED_DATASET の同バー行 (= 学習側バッチの post-OLS)

手順:
  1. snapshot を読み、bar 時刻と production 予測 (_m1_long 等) を取得。
  2. S6 から該当日の行を読む。
  3. 【物差し確認】 snapshot vs S6[bar ± N×M3] を比較し、 一番揃う shift を探す
     (隣バーで揃う→offset アーティファクト / bar=0 が最良でも乖離→本物)。
  4. 最良 shift で feature 別 |diff| ランキング + M1 gain 注釈。
  5. 【決定打】 M1 long モデルに snapshot ベクトル / S6 ベクトルを食わせ、 予測を再現
     (snapshot→~0.58, S6→~0.25 が出れば「post-OLS 特徴差が予測を動かす」を offline で証明)。

使い方:
  python compare_postols_features.py \
    --snapshot /workspace/data/diagnostics/feature_snapshots/snapshot_20260528_103001_L0.413_S0.000.csv
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
try:
    import lightgbm as lgb
except Exception:
    lgb = None

M3 = pd.Timedelta(minutes=3)


def parse_snapshot(p: Path):
    df = pd.read_csv(p)
    df.columns = [c.strip() for c in df.columns]
    meta, feats = {}, {}
    for _, r in df.iterrows():
        k = str(r["feature_name"]).strip(); v = r["value"]
        if k.startswith("_"):
            meta[k] = v
        else:
            feats[k] = pd.to_numeric(v, errors="coerce")
    return meta, pd.Series(feats, dtype=float)


def load_s6_window(s6_dir: Path, t0, t1):
    files = glob.glob(str(s6_dir/"**"/"*.parquet"), recursive=True)
    if not files:
        files = glob.glob(str(s6_dir/"*.parquet"))
    if not files:
        return None, None
    # ts 列検出のため1ファイルだけ schema を見る
    sample = (pl.read_parquet(files[0], n_rows=5) if pl else pd.read_parquet(files[0]).head(5))
    cols = sample.columns
    tsc = next((c for c in cols if "time" in c.lower() or "date" in c.lower()), None)
    if tsc is None: return None, None
    frames = []
    for f in files:
        try:
            d = (pl.read_parquet(f).to_pandas() if pl else pd.read_parquet(f))
        except Exception:
            continue
        d[tsc] = pd.to_datetime(d[tsc], utc=True, errors="coerce")
        d = d[(d[tsc] >= t0) & (d[tsc] <= t1)]
        if len(d): frames.append(d)
    if not frames: return None, tsc
    return pd.concat(frames, ignore_index=True), tsc


def m1_gain_map():
    p = getattr(config, "S7_M1_MODEL_LONG_PKL", None)
    if p is None or not Path(p).exists(): return {}, None
    m = joblib.load(p)
    b = m.booster_ if hasattr(m, "booster_") else m
    try:
        names = b.feature_name(); g = b.feature_importance(importance_type="gain")
    except Exception:
        return {}, m
    tot = g.sum() or 1
    return {n: 100*gi/tot for n, gi in zip(names, g)}, m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--snapshot", type=Path, required=True)
    ap.add_argument("--s6-dir", type=Path, default=Path(config.S6_WEIGHTED_DATASET))
    args = ap.parse_args()

    meta, snap = parse_snapshot(args.snapshot)
    ts = pd.to_datetime(meta.get("_timestamp_utc"), utc=True)
    bar = ts.floor("min")
    print("="*86)
    print(f"  compare_postols_features — post-OLS 突合 (局所化)")
    print(f"  snapshot bar≈{bar}  prod予測 _m1_long={meta.get('_m1_long')} _m2_long={meta.get('_m2_long')}")
    print(f"  snapshot 特徴量 {len(snap)} 個 (post-OLS)")
    print("="*86)

    s6, tsc = load_s6_window(args.s6_dir, bar - 10*M3, bar + 10*M3)
    if s6 is None:
        print(f"  ⚠ S6 読み込み失敗 (dir={args.s6_dir}, ts列={tsc})"); return
    s6 = s6.drop_duplicates(tsc).set_index(tsc).sort_index()
    print(f"  S6 窓 {len(s6)}行  列{len(s6.columns)}  ts範囲 {s6.index.min()}〜{s6.index.max()}\n")

    common = [f for f in snap.index if f in s6.columns]
    print(f"  共通 post-OLS 特徴量: {len(common)} 個")

    # 【物差し確認】 bar shift 掃引
    print("\n  【物差し確認】 snapshot vs S6[bar + N×M3] の一致 (どの bar が真か)")
    print(f"  {'shift(分)':>9}{'<1e-6':>9}{'<1e-2':>9}{'中央|Δ|':>11}{'最悪|Δ|':>11}")
    best, best_rate = 0, -1
    rows_by_shift = {}
    for n in (-2,-1,0,1,2):
        tb = bar + n*M3
        if tb not in s6.index:
            print(f"  {3*n:>9}{'(S6に無)':>9}"); continue
        s6row = s6.loc[tb]
        d = np.abs(snap[common].values - pd.to_numeric(s6row[common], errors="coerce").values)
        d = d[np.isfinite(d)]
        r6 = 100*np.mean(d < 1e-6); r2 = 100*np.mean(d < 1e-2)
        print(f"  {3*n:>9}{r6:>8.1f}%{r2:>8.1f}%{np.median(d):>11.4g}{np.max(d):>11.4g}")
        rows_by_shift[n] = s6row
        if r6 > best_rate: best_rate, best = r6, n

    print(f"\n  → 最良 shift = {3*best}分 (bit一致率 {best_rate:.1f}%)")
    if best != 0:
        print(f"  ⚠ bar=0 でなく {3*best}分ズレが最良 → timestamp offset アーティファクトの疑い")
    s6best = rows_by_shift.get(best)
    if s6best is None:
        print("  最良 shift 行が取れず終了"); return

    # feature 別 |diff| ランキング + M1 gain
    gain, m1model = m1_gain_map()
    diffs = []
    for f in common:
        a = snap[f]; b = pd.to_numeric(s6best[f], errors="coerce")
        if np.isfinite(a) and np.isfinite(b):
            diffs.append((f, abs(a-b), a, b, gain.get(f, 0.0)))
    diffs.sort(key=lambda x: x[1], reverse=True)
    print(f"\n  【最良 shift({3*best}分)での post-OLS |diff| ランキング TOP25】")
    print(f"  {'feature':<46}{'|Δ|':>10}{'snap':>9}{'S6':>9}{'M1gain%':>8}")
    for f, dd, a, b, gn in diffs[:25]:
        print(f"  {f:<46}{dd:>10.4g}{a:>9.3f}{b:>9.3f}{gn:>7.2f}%")

    n_big = sum(1 for _,dd,_,_,_ in diffs if dd >= 1e-2)
    gain_big = sum(gn for _,dd,_,_,gn in diffs if dd >= 1e-2)
    print(f"\n  |Δ|≥1e-2 の特徴量: {n_big}/{len(diffs)} 個、 それらの M1 gain 合計 {gain_big:.1f}%")

    # 【決定打】 M1 long モデルで snapshot vs S6 を推論再現
    if m1model is not None and lgb is not None:
        b = m1model.booster_ if hasattr(m1model,"booster_") else m1model
        order = b.feature_name()
        def vec(src):
            return np.array([[float(src.get(f, 0.0)) if hasattr(src,"get")
                              else float(src[f]) if f in src else 0.0
                              for f in order]], dtype=np.float32)
        Xs = np.array([[float(snap.get(f, 0.0)) for f in order]], dtype=np.float32)
        Xt = np.array([[float(pd.to_numeric(s6best[f], errors="coerce")) if f in s6best.index else 0.0
                        for f in order]], dtype=np.float32)
        ps = float(b.predict(Xs)[0]); pt = float(b.predict(Xt)[0])
        print("\n  【決定打: M1 long モデルで再現】")
        print(f"    snapshot 特徴量 → M1 long = {ps:.4f}  (本番ログ _m1_long={meta.get('_m1_long')})")
        print(f"    S6       特徴量 → M1 long = {pt:.4f}  (infer 期待値 ≈ 0.25)")
        print(f"    → 同一モデル・同一バーで予測差 {abs(ps-pt):.4f} を offline 再現。"
              f" 差は {'post-OLS 特徴量' if abs(ps-pt)>0.1 else '極小'} 由来。")
    print("="*86)


if __name__ == "__main__":
    main()
