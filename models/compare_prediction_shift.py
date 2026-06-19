"""compare_prediction_shift.py — 本番予測 vs infer 予測の突合 (§11.34.16 改修版)

旧 compare_prediction_live_vs_infer.py の後継。変更点:
  1. [SHIFT] LIVE(T) ↔ INFER(L = T - 180s) で突合 (§11.34.16 B 節の 1 本ズレ補正)。
     LIVE は本番シグナル時刻 T (= M3 close)、INFER は OOF ラベル L グリッド。
     旧版はシフトなしで突合し corr ≈ 0 (でたらめ) だった。
  2. [GATE 層別] 各行を「ゲート整合 / 不整合」 に分けて metric を出す。
     ゲート: T の分が高 TF (M5/M8/M15) 境界か = L の分が境界か。
     旧 main.py (T 基準) と新 main.py (L 基準) では実値の乗る行が互い違いに
     なるため、修正前データでは不整合行が崩れる (corr 0.32-0.58) はず。
     修正後データでは全行が整合行化し corr ≈ 0.99 に揃う見込み (§11.34.16 N.8)。

判定: SHIFT 後・整合行で corr >= 0.99 / diff_median <= 0.02。

使い方:
  python compare_prediction_shift.py \\
      --live /workspace/logs/m1_m2_predictions_log.csv \\
      --infer /workspace/data/XAUUSD/stratum_7_models/infer_XXXX \\
      --start "2026-06-XX 12:00:00" --end "2026-06-XX 16:30:00" \\
      --out-dir /workspace/data/diagnostics/prediction_compare_XXXX
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pandas as pd
import polars as pl

import compare_common as cc


def load_live(csv_path: Path, start: datetime, end: datetime) -> pd.DataFrame:
    df = pd.read_csv(csv_path, low_memory=False, on_bad_lines="skip")
    ts_col = "Timestamp" if "Timestamp" in df.columns else "timestamp"
    df[ts_col] = pd.to_datetime(df[ts_col], utc=True, errors="coerce")
    df = df.rename(columns={ts_col: "timestamp"})
    present = {
        "Long_M1_Raw": "live_m1_long",
        "Short_M1_Raw": "live_m1_short",
        "Long_M2_Raw": "live_m2_long",
        "Short_M2_Raw": "live_m2_short",
    }
    present = {k: v for k, v in present.items() if k in df.columns}
    out = df[["timestamp"] + list(present.keys())].rename(columns=present)
    out = out[(out["timestamp"] >= start) & (out["timestamp"] <= end)].copy()
    return out


def load_infer(infer_dir: Path, start: datetime, end: datetime) -> pd.DataFrame:
    files = {
        "infer_m1_long": infer_dir / "m1_oof_predictions_long.parquet",
        "infer_m1_short": infer_dir / "m1_oof_predictions_short.parquet",
        "infer_m2_long": infer_dir / "m2_oof_predictions_long.parquet",
        "infer_m2_short": infer_dir / "m2_oof_predictions_short.parquet",
    }
    merged: Optional[pd.DataFrame] = None
    for col, path in files.items():
        if not path.exists():
            print(f"    ⚠ {path.name} 無し → {col} スキップ")
            continue
        d = pl.read_parquet(str(path)).to_pandas()
        d["timestamp"] = pd.to_datetime(d["timestamp"], utc=True)
        pred_col = "prediction" if "prediction" in d.columns else None
        if pred_col is None:
            for cand in ["m2_pred_proba_raw", "m1_pred_proba_raw", "proba"]:
                if cand in d.columns:
                    pred_col = cand
                    break
        if pred_col is None:
            print(f"    ⚠ {path.name} 予測列なし → スキップ")
            continue
        d = d[["timestamp", pred_col]].rename(columns={pred_col: col})
        d = d.drop_duplicates("timestamp", keep="last")
        merged = d if merged is None else merged.merge(d, on="timestamp", how="outer")
    if merged is None:
        return pd.DataFrame(columns=["timestamp"])
    merged = merged[(merged["timestamp"] >= start) & (merged["timestamp"] <= end)].copy()
    return merged


def gate_consistent_mask(live_ts: pd.Series, shift_sec: int) -> np.ndarray:
    """各行が「ゲート整合」 か。

    T = live_ts、L = T - shift_sec。高 TF (5/8/15) のいずれかについて
    本番ゲートが見る分 (T 基準: 旧実装) と学習側実値行 (L 基準) が
    食い違う行を「不整合」 とする。
    旧 main.py は T 基準で 0 化していたので、T%tf==0 で本番実値・
    L%tf==0 で学習実値。両者が一致しない行が不整合。
    """
    # pandas datetime の int64 表現は us (マイクロ秒)。//10**6 で秒、//60 で分。
    T = pd.to_datetime(pd.Series(np.asarray(live_ts)), utc=True)
    L = T - pd.Timedelta(seconds=shift_sec)
    T_min = (T.astype("int64") // 10**6 // 60).to_numpy()
    L_min = (L.astype("int64") // 10**6 // 60).to_numpy()
    mism = np.zeros(len(live_ts), dtype=bool)
    for tf in (5, 8, 15):
        prod_open = (T_min % tf == 0)   # 旧本番ゲートが通す
        train_has = (L_min % tf == 0)   # 学習に実値がある
        mism |= (prod_open != train_has)
    return ~mism  # 整合 = 不一致なし


def compute_metrics(j: pd.DataFrame) -> Dict[str, Dict[str, float]]:
    pairs = {
        "M1_long": ("live_m1_long", "infer_m1_long"),
        "M1_short": ("live_m1_short", "infer_m1_short"),
        "M2_long": ("live_m2_long", "infer_m2_long"),
        "M2_short": ("live_m2_short", "infer_m2_short"),
    }
    out: Dict[str, Dict[str, float]] = {}
    for name, (lc, ic) in pairs.items():
        if lc not in j.columns or ic not in j.columns:
            continue
        sub = j[[lc, ic]].dropna()
        if len(sub) < 2:
            continue
        a = sub[lc].to_numpy(float)
        b = sub[ic].to_numpy(float)
        m = cc.pair_metrics(a, b)  # corr/diff_med/diff_max/bit_rate/rel_med/n
        out[name] = m
    return out


def print_block(title: str, metrics: Dict[str, Dict[str, float]]):
    print(f"  [{title}]")
    if not metrics:
        print("    (該当行なし)")
        return
    for name, m in metrics.items():
        corr_disp = "nan" if pd.isna(m["corr"]) else f"{m['corr']:.4f}"
        print(
            f"    {name:9s} n={m['n']:>3} corr={corr_disp:>8} "
            f"diff_med={m['diff_med']:.5f} bit={m['bit_rate']:.0f}%"
        )


def main():
    p = argparse.ArgumentParser(
        description="prediction LIVE vs INFER (SHIFT + GATE 層別) §11.34.16"
    )
    p.add_argument("--live", type=Path, default=Path("/workspace/logs/m1_m2_predictions_log.csv"))
    p.add_argument("--infer", type=Path, required=True)
    p.add_argument("--start", required=True)
    p.add_argument("--end", required=True)
    p.add_argument(
        "--out-dir", type=Path,
        default=Path("/workspace/data/diagnostics/prediction_compare_shift"),
    )
    p.add_argument("--shift-sec", type=int, default=cc.SHIFT_SEC)
    args = p.parse_args()
    cc.SHIFT_SEC = args.shift_sec

    args.out_dir.mkdir(parents=True, exist_ok=True)
    start_dt = cc.parse_dt(args.start)
    end_dt = cc.parse_dt(args.end)

    print("=" * 72)
    print("  予測 LIVE vs INFER 突合 (SHIFT + GATE 層別) §11.34.16")
    print("=" * 72)
    print(f"  期間: {start_dt} 〜 {end_dt}   shift: -{cc.SHIFT_SEC}s")
    print()

    live = load_live(args.live, start_dt, end_dt)
    infer = load_infer(args.infer, start_dt, end_dt)
    print(f"  LIVE: {len(live)} 行 / INFER: {len(infer)} 行")
    if len(live) == 0 or len(infer) == 0:
        print("  ❌ どちらかが空")
        sys.exit(1)

    # [SHIFT] LIVE(T) を L = T - shift にずらして INFER(L) と突合
    live_s = live.copy()
    live_s["ts_join"] = pd.to_datetime(live_s["timestamp"], utc=True) - pd.Timedelta(
        seconds=cc.SHIFT_SEC
    )
    infer = infer.copy()
    infer["timestamp"] = pd.to_datetime(infer["timestamp"], utc=True)
    j = live_s.merge(infer, left_on="ts_join", right_on="timestamp",
                     how="inner", suffixes=("", "_inf"))
    print(f"  [SHIFT -{cc.SHIFT_SEC}s] 突合: {len(j)} 行")
    if len(j) == 0:
        print("  ❌ シフト後の共通 timestamp 0")
        sys.exit(1)

    # [GATE 層別] 整合 / 不整合
    consistent = gate_consistent_mask(
        pd.to_datetime(j["timestamp"], utc=True), cc.SHIFT_SEC
    )
    j_con = j[consistent]
    j_mis = j[~consistent]
    print(f"  ゲート整合 {len(j_con)} 行 / 不整合 {len(j_mis)} 行")
    print()

    print("--- 予測一致 (SHIFT 後) ---")
    m_all = compute_metrics(j)
    m_con = compute_metrics(j_con)
    m_mis = compute_metrics(j_mis)
    print_block("全行", m_all)
    print()
    print_block("ゲート整合行 (← 修正後はこれが全行になる)", m_con)
    print()
    print_block("ゲート不整合行 (← 修正後は消える / 修正前は崩れる)", m_mis)

    # 保存
    j.drop(columns=[c for c in ["ts_join"] if c in j.columns]).to_parquet(
        args.out_dir / "per_signal.parquet", index=False
    )
    rows = []
    for grp, md in [("all", m_all), ("consistent", m_con), ("mismatch", m_mis)]:
        for name, m in md.items():
            rows.append({"group": grp, "pair": name, **m})
    pd.DataFrame(rows).to_parquet(args.out_dir / "metrics.parquet", index=False)

    # 判定 (整合行ベース)
    print()
    con_corrs = [m["corr"] for m in m_con.values() if not pd.isna(m["corr"])]
    con_diffs = [m["diff_med"] for m in m_con.values()]
    ok = (len(con_corrs) > 0 and min(con_corrs) >= 0.99
          and max(con_diffs) <= 0.02)
    verdict = "PASS (整合行 corr>=0.99)" if ok else "未達 (整合行 corr<0.99 → 残差あり)"
    print(f"  判定: {verdict}")
    with open(args.out_dir / "verdict.txt", "w") as f:
        f.write(f"consistent_min_corr="
                f"{min(con_corrs) if con_corrs else float('nan'):.4f} "
                f"verdict={verdict}\n")
    print(f"\n  出力一式: {args.out_dir}")


if __name__ == "__main__":
    main()
