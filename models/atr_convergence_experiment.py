#!/usr/bin/env python3
"""atr_convergence_experiment.py — ATR context長依存の収束カーブを実測する。
dump deque で ATR(13) を直近N本で計算し、N を変えて last 値の収束を見る。"""
from __future__ import annotations
import argparse, pickle, sys
from pathlib import Path
import numpy as np

def atr_wilder(high, low, close, period):
    n = len(high)
    atr = np.full(n, np.nan, dtype=np.float64)
    if n == 0:
        return atr
    tr = np.zeros(n, dtype=np.float64)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        hl = high[i] - low[i]
        hc = abs(high[i] - close[i-1])
        lc = abs(low[i] - close[i-1])
        tr[i] = max(hl, hc, lc)
    atr[0] = tr[0]
    for i in range(1, n):
        atr[i] = atr[i-1]*(period-1.0)/period + tr[i]/period
    return atr

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dump", type=Path, required=True)
    p.add_argument("--period", type=int, default=13)
    p.add_argument("--s2-atr", type=float, default=None)
    args = p.parse_args()
    with open(args.dump, "rb") as f:
        snap = pickle.load(f)
    high = snap["data"]["high"].astype(np.float64)
    low  = snap["data"]["low"].astype(np.float64)
    close = snap["data"]["close"].astype(np.float64)
    full_n = len(close)
    print(f"dump: tf={snap['tf']} ts={snap['ts']} deque長={full_n}")
    print(f"ATR period={args.period}, seed=TR[0]\n")
    atr_full = atr_wilder(high, low, close, args.period)
    last_full = atr_full[-1]
    print(f"【基準】全{full_n}本での ATR last = {last_full!r}")
    print(f"        (これが本番 rfe が出した値)\n")
    print("N本(直近)で計算した ATR last の収束カーブ:")
    print(f"{'N':>6} | {'ATR_last':>20} | {'全長との差':>14} | {'前Nとの差':>14}")
    print("-"*64)
    Ns = [n for n in [13,26,50,100,200,300,500,800,1200,1800,2500,3000,3600] if n <= full_n]
    prev = None
    for N in Ns:
        h, l, c = high[-N:], low[-N:], close[-N:]
        last_n = atr_wilder(h, l, c, args.period)[-1]
        d_full = last_n - last_full
        d_prev = (last_n - prev) if prev is not None else float('nan')
        print(f"{N:>6} | {last_n:>20.12f} | {d_full:>+14.2e} | {d_prev:>+14.2e}")
        prev = last_n
    print()
    print("解釈:")
    print("  - 前Nとの差が早期(N=200-500)に 1e-10未満 → ATR収束済み → 学習との差は【窓長以外】が真因")
    print("  - N=3600でもまだ大きい → 収束途上 → 窓長が本質 → 案Bで根治、再学習で一致")
    if args.s2_atr is not None:
        print(f"\n  学習側ATR真値: {args.s2_atr}")
        print(f"  全長ATRとの差: {last_full - args.s2_atr:+.4e} ({abs(last_full-args.s2_atr)/args.s2_atr*100:.2f}%)")

if __name__ == "__main__":
    main()
