#!/usr/bin/env python3
"""
compare_predictions.py — 予測レベル突合 (案 A): production vs training-side 予測

  production 側: dry-run の m1_m2_predictions_log.csv (LOGS_DIR)
  training 側 : infer_period の m1/m2 × long/short OOF parquet (infer_20260503_20260528)

同一時刻で M1L/M1S/M2L/M2S の確率を突合し、 「~2% の volume 差が予測を動かすか」 を測る。
一致 → 小差は immaterial = deploy 可。 ズレ → force_index 経由で効く = 要対処。

まずスキーマを表示し、 列を自動検出してから突合する (列名が想定と違っても schema で分かる)。

使い方:
  python compare_predictions.py
  python compare_predictions.py --infer-dir /workspace/data/XAUUSD/stratum_7_models/infer_20260503_20260528
  python compare_predictions.py --prod-log /workspace/logs/m1_m2_predictions_log.csv
"""
from __future__ import annotations
import sys, argparse
from pathlib import Path
import numpy as np, pandas as pd
try:
    import polars as pl
except Exception:
    pl = None
sys.path.insert(0, "/workspace")
import blueprint as config


def read_any(p: Path) -> pd.DataFrame:
    if p.suffix == ".parquet":
        return pl.read_parquet(p).to_pandas() if pl else pd.read_parquet(p)
    return pd.read_csv(p)


def find_ts_col(df):
    for c in df.columns:
        lc = c.lower()
        if "time" in lc or "date" in lc or lc in ("ts","t"):
            return c
    # datetime dtype フォールバック
    for c in df.columns:
        if np.issubdtype(df[c].dtype, np.datetime64):
            return c
    return None


def find_prob_col(df, ts_col):
    """予測確率列を選ぶ。 prediction/proba 系を優先し、 label/true/target 系は除外。"""
    EXCL = ("label","true","target","uniqueness","weight","timeframe")
    # 1. prediction/proba 系の名前を最優先
    for c in df.columns:
        lc = c.lower()
        if c == ts_col or any(e in lc for e in EXCL): continue
        if any(k in lc for k in ("predict","pred","proba","prob","score")):
            return c
    # 2. フォールバック: [0,1] float (label/true 等は除外) で最分散
    best, bestvar = None, -1
    for c in df.columns:
        lc = c.lower()
        if c == ts_col or any(e in lc for e in EXCL): continue
        s = pd.to_numeric(df[c], errors="coerce")
        if s.notna().mean() < 0.5: continue
        v = s.dropna()
        if len(v) and v.min() >= -0.01 and v.max() <= 1.01 and v.var() > bestvar:
            bestvar, best = v.var(), c
    return best


def norm_ts(s):
    t = pd.to_datetime(s, utc=True, errors="coerce")
    return t


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--infer-dir", type=Path,
                    default=Path(config.S7_MODELS_DIR)/"infer_20260503_20260528"
                    if hasattr(config,"S7_MODELS_DIR") else
                    Path("/workspace/data/XAUUSD/stratum_7_models/infer_20260503_20260528"))
    ap.add_argument("--prod-log", type=Path,
                    default=config.LOGS_DIR/"m1_m2_predictions_log.csv")
    ap.add_argument("--tol", type=float, default=0.05, help="一致とみなす確率差")
    args = ap.parse_args()

    print("="*84)
    print("  compare_predictions — 予測レベル突合 (案 A)")
    print("="*84)

    # ── training 側 infer parquet 4本 ──
    infer = {}
    for key, fname in [("m1_long","m1_oof_predictions_long.parquet"),
                       ("m1_short","m1_oof_predictions_short.parquet"),
                       ("m2_long","m2_oof_predictions_long.parquet"),
                       ("m2_short","m2_oof_predictions_short.parquet")]:
        p = args.infer_dir / fname
        if not p.exists():
            print(f"  [infer] 無し: {p}"); continue
        df = read_any(p)
        tsc = find_ts_col(df); pc = "prediction" if "prediction" in df.columns else find_prob_col(df, tsc)
        print(f"\n  [infer/{key}] {p.name}")
        print(f"    columns: {list(df.columns)}")
        if "timeframe" in df.columns:
            tfs = list(pd.unique(df["timeframe"]))
            print(f"    timeframe 種別: {tfs}")
            if len(tfs) > 1:
                df = df[df["timeframe"] == "M3"].copy()
                print(f"    → production シグナルは M3 のため timeframe==M3 に絞り込み ({len(df)}行)")
        print(f"    rows={len(df)}  ts_col={tsc}  prob_col={pc}")
        if tsc and pc:
            df = df[[tsc, pc]].copy()
            df.columns = ["timestamp","p"]
            df["timestamp"] = norm_ts(df["timestamp"])
            df = df.dropna(subset=["timestamp"]).drop_duplicates("timestamp")
            infer[key] = df
            print(f"    ts範囲: {df['timestamp'].min()} 〜 {df['timestamp'].max()}")

    # ── production 側 log ──
    if not args.prod_log.exists():
        print(f"\n  [prod] ログ無し: {args.prod_log}"); return
    prod = read_any(args.prod_log)
    ptsc = find_ts_col(prod)
    print(f"\n  [prod] {args.prod_log.name}")
    print(f"    columns: {list(prod.columns)}")
    print(f"    rows={len(prod)}  ts_col={ptsc}")
    print(f"    先頭2行:\n{prod.head(2).to_string()}")
    if ptsc is None:
        print("\n  ⚠ production ログの timestamp 列が自動検出できません。 columns を見て手動指定が必要。")
        return
    prod["_ts"] = norm_ts(prod[ptsc])

    # production 側の M1L/M1S/M2L/M2S 列を名前で推定
    def guess(colpat):
        for c in prod.columns:
            lc = c.lower().replace("_","").replace(" ","")
            if all(t in lc for t in colpat): return c
        return None
    pcols = {
        "m1_long":  guess(["m1","long"]) or guess(["m1","l"]),
        "m1_short": guess(["m1","short"]) or guess(["m1","s"]),
        "m2_long":  guess(["m2","long"]) or guess(["m2","l"]),
        "m2_short": guess(["m2","short"]) or guess(["m2","s"]),
    }
    print(f"\n  production 確率列の推定: {pcols}")

    # ── 突合 ──
    keys = [k for k in ["m1_long","m1_short","m2_long","m2_short"]
            if k in infer and pcols[k] is not None]

    # (a) 6点の実ペア値を表示 (offset=0)
    print("\n  【(a) 各シグナル時刻の実ペア値 (prod vs infer prediction)】")
    base = pd.DataFrame({"timestamp": prod["_ts"]})
    for k in keys:
        base[f"prod_{k}"] = pd.to_numeric(prod[pcols[k]], errors="coerce").values
    tbl = base.dropna(subset=["timestamp"]).copy()
    for k in keys:
        tbl = tbl.merge(infer[k].rename(columns={"p":f"inf_{k}"}), on="timestamp", how="left")
    with pd.option_context("display.width", 200, "display.max_columns", 30):
        cols = ["timestamp"] + sum([[f"prod_{k}", f"inf_{k}"] for k in keys], [])
        print(tbl[cols].to_string(index=False))

    # (b) timestamp shift 掃引 (production を ±2 M3バー=±3分 ずらす)
    print("\n  【(b) timestamp shift 掃引 (production を N×3分 ずらして m1_long 一致を見る)】")
    print(f"  {'shift(分)':>9}{'共通':>6}{'平均|Δ|':>10}{'方向一致':>9}")
    if "m1_long" in keys:
        pser = pd.to_numeric(prod[pcols["m1_long"]], errors="coerce")
        for nbar in (-2,-1,0,1,2):
            shifted = prod["_ts"] + pd.Timedelta(minutes=3*nbar)
            pdf = pd.DataFrame({"timestamp": shifted, "prod": pser}).dropna()
            mg = pdf.merge(infer["m1_long"], on="timestamp", how="inner")
            if len(mg)==0:
                print(f"  {3*nbar:>9}{0:>6}        -        -"); continue
            d=(mg["prod"]-mg["p"]).abs()
            dm=100*np.mean((mg["prod"]>=0.5)==(mg["p"]>=0.5))
            print(f"  {3*nbar:>9}{len(mg):>6}{d.mean():>10.4f}{dm:>8.0f}%")

    # (c) offset=0 のサマリ (従来)
    print("\n  【(c) offset=0 サマリ】")
    print(f"  {'モデル':<10}{'共通点数':>8}{'平均|Δ|':>10}{'最大|Δ|':>10}{'方向一致':>10}")
    for key in keys:
        pser = pd.to_numeric(prod[pcols[key]], errors="coerce")
        pdf = pd.DataFrame({"timestamp": prod["_ts"], "prod": pser}).dropna()
        merged = pdf.merge(infer[key], on="timestamp", how="inner")
        if len(merged) == 0:
            print(f"  {key:<10}  共通時刻なし"); continue
        d = (merged["prod"] - merged["p"]).abs()
        dir_match = 100*np.mean((merged["prod"]>=0.5) == (merged["p"]>=0.5))
        print(f"  {key:<10}{len(merged):>8}{d.mean():>10.4f}{d.max():>10.4f}{dir_match:>9.0f}%")

    print("\n  読み: (a) で infer 値が prod と桁違い/逆 → OOF か long/short 取り違え。")
    print("        (b) で非ゼロ shift が一致を跳ね上げ → timestamp 規約の 1バー offset。")
    print("        どれも該当せず Δ 大 → 真に予測が動く (force_index 経由)。")
    print("="*84)


if __name__ == "__main__":
    main()
