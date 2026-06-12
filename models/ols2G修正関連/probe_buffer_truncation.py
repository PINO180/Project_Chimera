#!/usr/bin/env python3
"""
probe_buffer_truncation.py — production の 115 が「バッファ長(warmup境界)」で出るか

[確定済み]
  - リサンプル同一、 連続系列での moment_8@gap = 7680 (= 学習) を offline 再現
  - production live = 115 → runtime buffer 由来
  - stable_moment_k が文脈独立なら、 gap で終わる ≥(2W-1)=99 本の窓は 7680 を再現するはず
  - production の data_buffers が gap バー処理時に短ければ (warmup 途中) → 別値 = 115?

[本スクリプト]
  S1 連続 close から「gap バーで終わる長さ L のバッファ」を切り出し、
  stable_moment_k を当てて末尾 (= gap バー) の値を L 毎に出す。
    どれかの L で ≈115 → バッファ長/warmup 境界が真因 (offline 確定、 計装不要)
    どの L でも 7680 (≥99) / NaN・0 (<99) で 115 が出ない
        → 115 は buffer 長でなく「中身が違う」 → runtime capture が必要

[使い方]
  python probe_buffer_truncation.py
  python probe_buffer_truncation.py --tf M5 --window 50 --moment 8
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

LENGTHS = [52, 60, 70, 80, 90, 99, 100, 110, 120, 150, 200, 300, 400, 576, 800]


def load_s1_close(tf: str) -> pd.DataFrame:
    tf_dir = Path(config.S1_PROCESSED) / f"timeframe={tf}"
    df = (
        pl.scan_parquet(str(tf_dir / "*.parquet"))
        .select(["timestamp", "close"])
        .with_columns(pl.col("timestamp").cast(pl.Datetime("us", "UTC")))
        .sort("timestamp").collect().to_pandas()
    )
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    return df


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tf", default="M5")
    ap.add_argument("--window", type=int, default=50)
    ap.add_argument("--moment", type=int, default=8)
    ap.add_argument("--gap-ts", default="2026-05-24 22:00:00")
    args = ap.parse_args()

    gap_ts = pd.Timestamp(args.gap_ts, tz="UTC")
    W, M = args.window, args.moment

    print("=" * 72)
    print("  probe_buffer_truncation.py — buffer 長で 115 が出るか")
    print("=" * 72)
    print(f"  tf={args.tf} W={W} moment={M} gap_ts={gap_ts}")
    print(f"  (連続=7680.31, production live=115.20 が参照)")

    s1 = load_s1_close(args.tf)
    close = s1["close"].to_numpy(np.float64)
    gi = s1.index[s1["timestamp"] == gap_ts]
    if len(gi) == 0:
        print("  [FATAL] gap_ts が S1 に無い"); return
    gi = int(gi[0])

    # 参照: 連続全系列での gap 値
    full = stable_moment_k_engine_formula(close, W, M)[gi]
    print(f"\n  連続全系列 moment@gap = {full:+.6g}")

    print(f"\n  {'buffer長 L':>10}{'gap値 (末尾)':>18}{'有効?':>8}")
    print("  " + "-" * 38)
    for L in LENGTHS:
        lo = gi - L + 1
        if lo < 0:
            continue
        buf = close[lo:gi + 1]               # gap バーで終わる長さ L の窓
        arr = stable_moment_k_engine_formula(buf, W, M)
        val = arr[-1]
        ok = "finite" if np.isfinite(val) else "NaN/inf"
        flag = ""
        if np.isfinite(val):
            if abs(val - 115.203) / 115.203 < 0.05:
                flag = "  ★≈115 (production 一致!)"
            elif abs(val - full) / max(1.0, abs(full)) < 1e-6:
                flag = "  = 連続(7680)"
        print(f"  {L:>10}{val:>18.6g}{ok:>8}{flag}")

    print("\n  --- 判定 ---")
    print("   どこかで ★≈115 が出れば → buffer 長/warmup 境界が真因 (offline 確定)")
    print("   ≥99 で常に 7680、 <99 で NaN/0 のみ (115 出ない)")
    print("     → 115 は buffer 長でなく『中身が違う』 → 計装 dry-run が必要")
    print("=" * 72)


if __name__ == "__main__":
    main()
