#!/usr/bin/env python3
"""
resample_gap_compare.py — gap バーでの M5 close 窓の差を dry-run なしで特定

[確定済み]
  - gap バー (disc=True, 2026-05-24 22:00) で raw=S2=7680、 production=115
  - クリップ/pkl は無罪 (seed 境界広い) → production の生値が 115 = M1 (入力窓の差)
  - リサンプル規約は両側一致 (closed=left, label=left)
  - 残る差: gap 跨ぎでの実バケット内容 = pandas.resample(本番) vs polars.group_by_dynamic(S1)
    の anchoring 差

[本スクリプト]
  S1 の M0.5 を「本番と同じ pandas.resample(closed=left,label=left)+V=0 guard」 で
  M5 化し、 S1 M5 (= s1_1_B / polars 出力) と gap バー周辺で:
    - timestamp / close を 1 本ずつ突合
    - 各々の close 列に stable_moment_k を当て、 gap バーの moment_8 を比較
  これで:
    両者 moment ≈ 7680 で一致 → リサンプルは同一 → 115 は runtime buffer 由来 (要計装)
    両者 close/moment が gap で食い違う → anchoring 差が M1 真因 (offline で確定)

[使い方]
  python resample_gap_compare.py
  python resample_gap_compare.py --tf M5 --window 50 --moment 8 --gap-ts "2026-05-24 22:00:00"
"""

from __future__ import annotations

import sys
import argparse
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl

warnings.filterwarnings("ignore", category=FutureWarning)

sys.path.insert(0, "/workspace")
import blueprint as config  # noqa: E402

sys.path.insert(0, str(config.CORE_DIR))
from stable_rolling import stable_moment_k_engine_formula  # noqa: E402

TF_FREQ = {"M0.5": "30s", "M1": "1min", "M3": "3min", "M5": "5min",
           "M8": "8min", "M15": "15min"}


def load_s1(tf: str) -> pd.DataFrame:
    tf_dir = Path(config.S1_PROCESSED) / f"timeframe={tf}"
    cols = ["timestamp", "open", "high", "low", "close", "volume"]
    lf = pl.scan_parquet(str(tf_dir / "*.parquet"))
    have = [c for c in cols if c in lf.collect_schema().names()]
    df = (
        lf.select(have)
        .with_columns(pl.col("timestamp").cast(pl.Datetime("us", "UTC")))
        .sort("timestamp").collect().to_pandas()
    )
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    return df.set_index("timestamp")


def production_resample(m05: pd.DataFrame, tf: str) -> pd.DataFrame:
    """本番 rfe と同じ pandas.resample(closed=left,label=left)+V=0 guard を再現。"""
    rule = TF_FREQ[tf]
    agg = {"open": "first", "high": "max", "low": "min", "close": "last",
           "volume": "sum"}
    agg = {k: v for k, v in agg.items() if k in m05.columns}
    r = m05.resample(rule, closed="left", label="left").agg(agg)
    # tick_count>0 = volume>0 のバーだけ残す (V=0 GUARD 相当 = s1_1_B filter(tick_count>0))
    if "volume" in r.columns:
        r = r[r["volume"] > 0]
    else:
        r = r.dropna(subset=["close"])
    return r


def moment_at(close: np.ndarray, ts_index: pd.DatetimeIndex,
              gap_ts: pd.Timestamp, w: int, m: int):
    arr = stable_moment_k_engine_formula(close.astype(np.float64), w, m)
    locs = np.where(ts_index == gap_ts)[0]
    if len(locs) == 0:
        return None, None
    i = locs[0]
    return float(arr[i]), i


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tf", default="M5")
    parser.add_argument("--window", type=int, default=50)
    parser.add_argument("--moment", type=int, default=8)
    parser.add_argument("--gap-ts", default="2026-05-24 22:00:00")
    parser.add_argument("--context", type=int, default=4)
    args = parser.parse_args()

    gap_ts = pd.Timestamp(args.gap_ts, tz="UTC")
    print("=" * 72)
    print("  resample_gap_compare.py — gap バー M5 close 窓の差 (offline)")
    print("=" * 72)
    print(f"  tf={args.tf} window={args.window} moment={args.moment} gap_ts={gap_ts}")

    s1_tf = load_s1(args.tf)              # 学習側 (polars group_by_dynamic)
    m05 = load_s1("M0.5")                 # 共通の素データ
    prod_tf = production_resample(m05, args.tf)  # 本番 (pandas resample) 再現

    print(f"\n  S1_{args.tf} (polars)         : {len(s1_tf)} 行")
    print(f"  prod再現_{args.tf} (pandas)    : {len(prod_tf)} 行  (M0.5={len(m05)}行から)")

    # gap バー周辺の timestamp / close を 1 本ずつ突合
    print(f"\n  --- gap バー周辺の close 突合 (S1 vs 本番再現) ---")
    print(f"     {'timestamp':<28}{'S1 close':>14}{'prod close':>14}{'Δ':>12}")
    around = pd.date_range(gap_ts - pd.Timedelta(minutes=5 * args.context),
                           gap_ts + pd.Timedelta(minutes=5 * args.context),
                           freq=TF_FREQ[args.tf], tz="UTC")
    for ts in around:
        s1c = s1_tf["close"].get(ts, np.nan)
        pdc = prod_tf["close"].get(ts, np.nan)
        d = (s1c - pdc) if (np.isfinite(s1c) and np.isfinite(pdc)) else np.nan
        mark = "  <== gap" if ts == gap_ts else ""
        s1s = f"{s1c:.5f}" if np.isfinite(s1c) else "(無)"
        pds = f"{pdc:.5f}" if np.isfinite(pdc) else "(無)"
        ds = f"{d:+.5f}" if np.isfinite(d) else "-"
        print(f"     {str(ts):<28}{s1s:>14}{pds:>14}{ds:>12}{mark}")

    # 各 close 列に stable_moment_k を当てて gap バーの値を比較
    s1_mom, s1_i = moment_at(s1_tf["close"].to_numpy(), s1_tf.index, gap_ts,
                             args.window, args.moment)
    pd_mom, pd_i = moment_at(prod_tf["close"].to_numpy(), prod_tf.index, gap_ts,
                             args.window, args.moment)

    print(f"\n  --- gap バーでの moment_{args.moment}_{args.window} ---")
    print(f"     S1 (polars)   : {s1_mom if s1_mom is None else f'{s1_mom:+.6g}'}  (idx={s1_i})")
    print(f"     本番再現(pandas): {pd_mom if pd_mom is None else f'{pd_mom:+.6g}'}  (idx={pd_i})")

    print("\n  --- 判定 ---")
    if s1_mom is not None and pd_mom is not None:
        rel = abs(s1_mom - pd_mom) / max(1.0, abs(s1_mom))
        if rel < 1e-6:
            print(f"     両者一致 ({s1_mom:+.6g}) → リサンプルは同一。")
            print("     → 本番 dry-run の 115 は runtime buffer 由来 (warmup/extent)。")
            print("       次は計装 dry-run で gap バーの生値+rollstd を直接 capture。")
        else:
            print(f"     食い違い (S1={s1_mom:+.6g} vs 本番再現={pd_mom:+.6g})")
            print("     → gap 跨ぎ anchoring 差が M1 真因。 offline で確定。")
            print("       上の close 突合で「どの M0.5 がどの M5 バケットに入るか」 の差を確認。")
    else:
        print("     gap_ts が一方のインデックスに無い → 上の close 突合 (無) 行を確認。")
        print("     これ自体が anchoring 差 (バケット境界が違い gap バーの label がズレる)。")

    print("\n" + "=" * 72)


if __name__ == "__main__":
    main()
