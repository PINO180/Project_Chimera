#!/usr/bin/env python3
"""atr_disc_hypothesis.py — 本番ATR(1.79)と学習ATR(1.86)の差がdisc(週末ギャップ)由来か検証。"""
from __future__ import annotations
import argparse, pickle
from pathlib import Path
import numpy as np

def tr_series(high, low, close, disc=None):
    n = len(high)
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        hl = high[i] - low[i]
        if disc is not None and disc[i]:
            tr[i] = hl
        else:
            hc = abs(high[i] - close[i-1]); lc = abs(low[i] - close[i-1])
            tr[i] = max(hl, hc, lc)
    return tr

def atr_from_tr(tr, period):
    n = len(tr); atr = np.full(n, np.nan); atr[0] = tr[0]
    for i in range(1, n):
        atr[i] = atr[i-1]*(period-1.0)/period + tr[i]/period
    return atr

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dump", type=Path, required=True)
    p.add_argument("--period", type=int, default=13)
    p.add_argument("--s2-atr", type=float, default=1.8556592220303743)
    args = p.parse_args()
    snap = pickle.load(open(args.dump, "rb"))
    high = snap["data"]["high"].astype(np.float64)
    low  = snap["data"]["low"].astype(np.float64)
    close = snap["data"]["close"].astype(np.float64)
    n = len(close)
    tr = tr_series(high, low, close)
    atr_normal = atr_from_tr(tr, args.period)
    print(f"通常TR ATR last = {atr_normal[-1]:.12f}  (本番値)")
    print(f"学習ATR真値      = {args.s2_atr:.12f}")
    print(f"差               = {atr_normal[-1]-args.s2_atr:+.4e} ({abs(atr_normal[-1]-args.s2_atr)/args.s2_atr*100:.2f}%)\n")
    print("直近500本での close-to-close ジャンプ TOP10 (ギャップ候補):")
    jumps = np.abs(np.diff(close[-500:]))
    idx_sorted = np.argsort(jumps)[::-1][:10]
    bar_len = np.abs(high[-500:] - low[-500:])
    for rank, j in enumerate(idx_sorted):
        gi = n - 500 + j + 1
        print(f"  #{rank+1}: idx={gi} jump=|Δclose|={jumps[j]:.4f}  H-L={bar_len[j+1]:.4f}  TR={tr[gi]:.4f}")
    print("\nTRスパイク足を H-L のみに置換した場合の ATR last:")
    for thr_mult in [3.0, 2.0, 1.5]:
        tr_test = tr.copy(); n_replaced = 0
        for i in range(max(1, n-500), n):
            hl = high[i]-low[i]
            if tr[i] > thr_mult * hl and hl > 0:
                tr_test[i] = hl; n_replaced += 1
        atr_test = atr_from_tr(tr_test, args.period)
        diff = atr_test[-1] - args.s2_atr
        print(f"  TR>{thr_mult}×(H-L)置換 ({n_replaced}足): ATR={atr_test[-1]:.12f}  学習との差={diff:+.4e} ({abs(diff)/args.s2_atr*100:.2f}%)")
    print("\n解釈: 置換でATRが学習値(1.8557)に近づく→disc真因 / 1.79のまま→別の経路差")

if __name__ == "__main__":
    main()
