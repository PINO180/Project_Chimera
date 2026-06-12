#!/usr/bin/env python3
"""
diagnose_volume_skew.py — リード B (volume tick 集約差) の系統的解明

複数の pre-gate snapshot × S6 同バーで、volume 関連特徴の乖離を集計:
  1. どの volume 特徴がどれだけ乖離しているか (per-feature |Δ|, Δ%)
  2. 系統的か散発的か (中央値・分散・サンプル分布)
  3. gain 重み付き「予測への効き度合い」順
  4. M3 のみ / M8 境界時 / M15 境界時で層別

使い方:
  python diagnose_volume_skew.py \
    --snapshots-dir /workspace/data/diagnostics/feature_snapshots \
    --date-pattern '20260528' \
    --top 30
"""
from __future__ import annotations
import sys, argparse, glob, re
from pathlib import Path
import numpy as np, pandas as pd
try:
    import polars as pl
except Exception:
    pl = None
import joblib
sys.path.insert(0, "/workspace")
import blueprint as config

_TF_SUFFIX_RE = re.compile(r"_M([0-9]+(?:\.[0-9]+)?)$")
# volume 関連の特徴量 base name パターン (S2 build と S6 features で共通する命名)
_VOLUME_FEATURE_PATTERN = re.compile(
    r"force_index|obv|volume_ratio|volume_mean|volume_std|"
    r"volume_change|mfi|cmf|chaikin|ad_line|vwap|volume_z",
    re.IGNORECASE,
)


def parse_snapshot(p):
    df = pd.read_csv(p); df.columns = [c.strip() for c in df.columns]
    meta, feats = {}, {}
    for _, r in df.iterrows():
        k = str(r["feature_name"]).strip(); v = r["value"]
        if k.startswith("_"): meta[k] = v
        else: feats[k] = pd.to_numeric(v, errors="coerce")
    return meta, feats


def is_higher_tf_boundary(minute_idx, name):
    """特徴量名から TF を抽出し、minute_idx がその TF の境界かを返す。
    M0.5/M1/M3 は常に境界扱い (= 境界 True)。"""
    m = _TF_SUFFIX_RE.search(name)
    if not m: return True  # TF サフィックス無しは常に有効
    try: tf_min = float(m.group(1))
    except (TypeError, ValueError): return True
    if tf_min < 5: return True
    tf_int = int(tf_min)
    return tf_int > 0 and (minute_idx % tf_int == 0)


def load_s6_window(s6_dir, t0, t1):
    files = glob.glob(str(s6_dir/"**"/"*.parquet"), recursive=True) \
        or glob.glob(str(s6_dir/"*.parquet"))
    frames = []
    tsc = None
    for f in files:
        try: d = pl.read_parquet(f).to_pandas() if pl else pd.read_parquet(f)
        except Exception: continue
        if tsc is None:
            tsc = next((c for c in d.columns if "time" in c.lower() or "date" in c.lower()), None)
            if not tsc: continue
        d[tsc] = pd.to_datetime(d[tsc], utc=True, errors="coerce")
        d = d[(d[tsc] >= t0) & (d[tsc] <= t1)]
        if len(d): frames.append(d)
    if not frames: return None, tsc
    return pd.concat(frames, ignore_index=True), tsc


def get_gain_map():
    """M1 long の gain map (feature -> gain%) を返す。"""
    m = joblib.load(config.S7_M1_MODEL_LONG_PKL)
    b = m.booster_ if hasattr(m, "booster_") else m
    names = b.feature_name()
    g = b.feature_importance(importance_type="gain")
    tot = g.sum() or 1
    return {n: 100 * gi / tot for n, gi in zip(names, g)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--snapshots-dir", type=Path,
                    default=Path("/workspace/data/diagnostics/feature_snapshots"))
    ap.add_argument("--date-pattern", default="20260528",
                    help="snapshot ファイル名にこの文字列を含むもののみ対象")
    ap.add_argument("--s6-dir", type=Path, default=Path(config.S6_WEIGHTED_DATASET))
    ap.add_argument("--top", type=int, default=30,
                    help="表示する top N 特徴量")
    args = ap.parse_args()

    snap_files = sorted(args.snapshots_dir.glob(f"snapshot_*{args.date_pattern}*.csv"))
    if not snap_files:
        print(f"snapshot 見つからず: {args.snapshots_dir} / pattern {args.date_pattern}")
        return
    print(f"対象 snapshot: {len(snap_files)} 件")

    # メタ + bar 一覧
    snapshots = []
    for f in snap_files:
        try:
            meta, feats = parse_snapshot(f)
            bar = pd.to_datetime(meta.get("_timestamp_utc"), utc=True).floor("min")
            snapshots.append((bar, feats, f))
        except Exception as e:
            print(f"  読込失敗 {f.name}: {e}")
    snapshots.sort(key=lambda x: x[0])
    if not snapshots:
        print("有効 snapshot 無し"); return

    t0, t1 = snapshots[0][0], snapshots[-1][0]
    print(f"期間: {t0} → {t1}")

    s6, tsc = load_s6_window(args.s6_dir, t0, t1)
    if s6 is None:
        print("S6 読込失敗"); return
    s6 = s6.drop_duplicates(tsc).set_index(tsc).sort_index()
    print(f"S6 window: {len(s6)} 行")

    gain_map = get_gain_map()

    # 集計用: feature -> list of (bar, prod_val, s6_val, abs_diff, rel_diff%, gain%)
    records = []
    layer_stats = {"M3only": 0, "M8boundary": 0, "M15boundary": 0, "AllBoundary": 0}

    for bar, feats, _ in snapshots:
        if bar not in s6.index: continue
        s6_row = s6.loc[bar]
        minute_idx = int(bar.timestamp() // 60)

        # bar の境界状況を分類
        m8b = (minute_idx % 8 == 0)
        m15b = (minute_idx % 15 == 0)
        if m8b and m15b: layer_stats["AllBoundary"] += 1
        elif m8b: layer_stats["M8boundary"] += 1
        elif m15b: layer_stats["M15boundary"] += 1
        else: layer_stats["M3only"] += 1

        for name, prod_val in feats.items():
            if not _VOLUME_FEATURE_PATTERN.search(name): continue
            # この feature が gate で 0 化されるなら S6 とも 0 同士で比較になる、 skip
            if not is_higher_tf_boundary(minute_idx, name): continue
            if name not in s6_row.index: continue
            try:
                pv = float(prod_val)
                sv = float(pd.to_numeric(s6_row[name], errors="coerce"))
            except (TypeError, ValueError):
                continue
            if not (np.isfinite(pv) and np.isfinite(sv)): continue
            d = pv - sv
            abs_d = abs(d)
            denom = max(abs(pv), abs(sv), 1e-9)
            rel_d = 100.0 * abs_d / denom
            gn = gain_map.get(name, 0.0)
            records.append((name, bar, pv, sv, d, abs_d, rel_d, gn))

    if not records:
        print("対象レコード無し"); return

    df = pd.DataFrame(records, columns=["feature", "bar", "prod", "s6", "diff", "absdiff", "reldiff_pct", "gain_pct"])
    print(f"対象レコード: {len(df)} 件 ({df['feature'].nunique()} 特徴量 × {df['bar'].nunique()} バー)")
    print()

    print("=" * 110)
    print("  バー層別カウント (snapshot × S6 がマッチした bar のみ)")
    print("=" * 110)
    for k, v in layer_stats.items():
        print(f"    {k:<14}: {v}")
    print()

    # 特徴量別集計
    agg = df.groupby("feature").agg(
        n=("absdiff", "count"),
        median_absdiff=("absdiff", "median"),
        median_reldiff=("reldiff_pct", "median"),
        mean_diff=("diff", "mean"),  # 符号付き = 系統的か
        std_diff=("diff", "std"),
        gain_pct=("gain_pct", "first"),
    ).reset_index()
    agg["impact"] = agg["median_absdiff"] * agg["gain_pct"]
    agg = agg.sort_values("impact", ascending=False).head(args.top)

    print("=" * 110)
    print(f"  TOP {args.top} volume 特徴量 (M1 gain × 中央値 |Δ| で sort = 予測への効き度合い)")
    print("=" * 110)
    print(f"  {'feature':<50}{'gain%':>7}{'n':>5}{'med|Δ|':>9}{'med%Δ':>8}{'mean(Δ)':>10}{'std(Δ)':>9}{'impact':>9}")
    for _, r in agg.iterrows():
        print(f"  {r['feature']:<50}"
              f"{r['gain_pct']:>7.2f}{int(r['n']):>5}"
              f"{r['median_absdiff']:>9.4f}{r['median_reldiff']:>8.2f}"
              f"{r['mean_diff']:>10.4f}{r['std_diff']:>9.4f}{r['impact']:>9.3f}")
    print("=" * 110)

    # 系統性の判定
    print()
    print("=" * 110)
    print("  系統性指標 (mean_diff の符号が揃っていれば系統的、 std が小さいほど一貫)")
    print("=" * 110)
    sign_consistency = (agg["mean_diff"] > 0).sum(), (agg["mean_diff"] < 0).sum()
    print(f"  mean_diff > 0 の特徴量数: {sign_consistency[0]} / {len(agg)}")
    print(f"  mean_diff < 0 の特徴量数: {sign_consistency[1]} / {len(agg)}")
    # mean|Δ| と std の比率 (Coefficient of Variation 的)
    agg["cv"] = agg["std_diff"] / (agg["median_absdiff"] + 1e-9)
    print(f"  CV (std/median|Δ|) 中央値: {agg['cv'].median():.2f}  "
          f"(< 1 → 一貫した乖離、 >> 1 → 散発的)")
    print("=" * 110)


if __name__ == "__main__":
    main()
