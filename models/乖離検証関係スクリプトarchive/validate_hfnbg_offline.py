#!/usr/bin/env python3
"""
validate_hfnbg_offline.py — pre-gate snapshot に HF-NB-GATE を offline 適用し、
M1 long 予測が
  (1) snapshot 生 (pre-gate, forward-fill 規約)
  (2) snapshot に gate 適用 (post-gate 期待値)
  (3) S6 同バー (学習側 0-fill 規約)
の 3 点でどう動くかを確認する。 目標: (2) ≈ (3)。
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


def parse_snapshot(p):
    df = pd.read_csv(p); df.columns = [c.strip() for c in df.columns]
    meta, feats = {}, {}
    for _, r in df.iterrows():
        k = str(r["feature_name"]).strip(); v = r["value"]
        if k.startswith("_"): meta[k] = v
        else: feats[k] = pd.to_numeric(v, errors="coerce")
    return meta, feats


def apply_gate(feature_dict: dict, ts: pd.Timestamp) -> dict:
    """main.py L1173-1207 の HF-NB-GATE を offline で再現"""
    minute_idx = int(ts.timestamp() // 60)
    out = dict(feature_dict)
    for name in list(out.keys()):
        m = _TF_SUFFIX_RE.search(name)
        if not m: continue
        try: tf_min = float(m.group(1))
        except (TypeError, ValueError): continue
        if tf_min < 5: continue
        tf_int = int(tf_min)
        if tf_int > 0 and minute_idx % tf_int != 0:
            out[name] = 0.0
    return out


def load_s6_row(s6_dir, bar):
    files = glob.glob(str(s6_dir / "**" / "*.parquet"), recursive=True) \
        or glob.glob(str(s6_dir / "*.parquet"))
    for f in files:
        try: d = pl.read_parquet(f).to_pandas() if pl else pd.read_parquet(f)
        except Exception: continue
        tsc = next((c for c in d.columns if "time" in c.lower() or "date" in c.lower()), None)
        if not tsc: continue
        d[tsc] = pd.to_datetime(d[tsc], utc=True, errors="coerce")
        match = d[d[tsc] == bar]
        if len(match): return match.iloc[0]
    return None


def get_feature_names(model):
    if hasattr(model, "feature_name"): return model.feature_name()
    b = model.booster_ if hasattr(model, "booster_") else model
    return b.feature_name()


def vector_from(model, source):
    names = get_feature_names(model)
    vec, miss = [], 0
    for n in names:
        v = source.get(n, 0.0) if isinstance(source, dict) \
            else (source[n] if n in source.index else 0.0)
        try:
            v = float(v)
            if not np.isfinite(v): v = 0.0; miss += 1
        except (TypeError, ValueError):
            v = 0.0; miss += 1
        vec.append(v)
    return np.array([vec], dtype=np.float32), miss


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--snapshot", type=Path, required=True)
    ap.add_argument("--s6-dir", type=Path, default=Path(config.S6_WEIGHTED_DATASET))
    args = ap.parse_args()

    meta, feats = parse_snapshot(args.snapshot)
    bar = pd.to_datetime(meta.get("_timestamp_utc"), utc=True).floor("min")
    minute_idx = int(bar.timestamp() // 60)
    snap_m1_long = meta.get("_m1_long")

    m = joblib.load(config.S7_M1_MODEL_LONG_PKL)

    # (1) raw (gate 未適用 = 修正前本番状態の再現)
    X_raw, _ = vector_from(m, feats)
    p_raw = float(m.predict(X_raw)[0])

    # (2) gated (HF-NB-GATE 適用 = 修正後本番期待値)
    feats_gated = apply_gate(feats, bar)
    X_gated, _ = vector_from(m, feats_gated)
    p_gated = float(m.predict(X_gated)[0])

    # (3) S6 (学習側、 = infer 期待値)
    s6_row = load_s6_row(args.s6_dir, bar)
    p_s6 = None; miss_s6 = -1
    if s6_row is not None:
        X_s6, miss_s6 = vector_from(m, s6_row)
        p_s6 = float(m.predict(X_s6)[0])

    diff_count = sum(
        1 for k in feats
        if not np.isclose(
            feats.get(k, 0.0) if pd.notna(feats.get(k, 0.0)) else 0.0,
            feats_gated.get(k, 0.0)
        )
    )

    print("=" * 92)
    print(f"  validate_hfnbg_offline — bar {bar}")
    print(f"  境界判定 (minute_idx={minute_idx}):  "
          f"%5={minute_idx%5}  %8={minute_idx%8}  %15={minute_idx%15}  "
          f"(0=境界)")
    print(f"  snapshot meta _m1_long = {snap_m1_long}  (本番ログ値)")
    print("=" * 92)
    print(f"  M1 long 予測:")
    print(f"    (1) snapshot 生   (= 修正前本番状態)         : {p_raw:.4f}")
    print(f"    (2) snapshot + gate 適用 (= 修正後本番期待値): {p_gated:.4f}")
    if p_s6 is not None:
        print(f"    (3) S6 同バー    (= 学習/infer 期待値)        : {p_s6:.4f}  "
              f"(S6 欠損 {miss_s6} 件)")
        print()
        d_raw_s6 = abs(p_raw - p_s6)
        d_gated_s6 = abs(p_gated - p_s6)
        print(f"  距離 |(1)−(3)| = {d_raw_s6:.4f}  (修正前の乖離)")
        print(f"  距離 |(2)−(3)| = {d_gated_s6:.4f}  (修正後の残差)")
        if d_raw_s6 > 1e-6:
            ratio = d_gated_s6 / d_raw_s6
            print(f"  縮小率 = {ratio:.2%}")
            verdict = "✓ gate で乖離 80%+ 縮小、 残差は B 系統由来と推定" if ratio < 0.2 \
                else "✓ gate で乖離縮小、 残差あり (リード B 等)" if ratio < 0.5 \
                else "△ 縮小は限定的、 別機構が残っている可能性"
            print(f"  判定: {verdict}")
    else:
        print(f"    (3) S6 同バー → S6 に該当バー無し ({bar})")
    print()
    print(f"  gate で 0 化された特徴数: {diff_count}")
    print("=" * 92)


if __name__ == "__main__":
    main()
