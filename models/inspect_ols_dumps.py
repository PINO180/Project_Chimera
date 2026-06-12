#!/usr/bin/env python3
"""
inspect_ols_dumps.py — OLS state dump の sanity check

dump ファイル群を読み、 構造と統計をざっくり表示する。 本格比較スクリプトの前段。

確認項目:
  1. dump ファイル一覧と timestamp
  2. 各 dump の TF 別 OLS state サイズ
  3. 各 TF の x_deque (market_proxy) 長と統計値
  4. ols_state[tf][feat] の count 分布 (= warmup 充足度の指標)
  5. dump 内 feature_dict と ols_state の整合性
"""

from __future__ import annotations

import argparse
import pickle
from collections import Counter
from pathlib import Path

import numpy as np


def inspect_one(path: Path) -> dict:
    """1 dump file の構造とサイズを表示。 返り値は要約 dict"""
    print("=" * 72)
    print(f"  {path.name} ({path.stat().st_size:,} bytes)")
    print("=" * 72)

    with open(path, "rb") as f:
        d = pickle.load(f)

    ts = d["signal_timestamp"]
    sig_tf = d["signal_timeframe"]
    print(f"  signal_timestamp: {ts}")
    print(f"  signal_timeframe: {sig_tf}")
    print(f"  feature_dict size: {len(d['feature_dict']):,}")

    # market_proxy_tail
    mp = d.get("market_proxy_tail_200")
    if mp:
        print(f"  market_proxy_tail_200: {len(mp)} rows")
        ts_first = mp[0].get("timestamp", "?")
        ts_last  = mp[-1].get("timestamp", "?")
        proxy_vals = [r.get("market_proxy") for r in mp if r.get("market_proxy") is not None]
        print(f"    range: {ts_first} 〜 {ts_last}")
        if proxy_vals:
            arr = np.asarray(proxy_vals, dtype=np.float64)
            print(f"    proxy 値: mean={arr.mean():+.6f}, std={arr.std():.6f}, "
                  f"min={arr.min():+.6f}, max={arr.max():+.6f}")
    else:
        print("  market_proxy_tail_200: None")

    # OLS state per TF
    print(f"\n  --- OLS state (= incremental running sums) ---")
    ols_state = d["ols_state"]
    for tf in sorted(ols_state.keys()):
        feats = ols_state[tf]
        if not feats:
            continue
        counts = [s["count"] for s in feats.values() if isinstance(s, dict) and "count" in s]
        sum_x = [s["sum_x"] for s in feats.values() if isinstance(s, dict) and "sum_x" in s]
        # count 分布 (warmup 充足度)
        if counts:
            print(f"    {tf:6s}: {len(feats)} features, "
                  f"count min/median/max = {min(counts)}/{int(np.median(counts))}/{max(counts)}, "
                  f"sum_x stats: mean={np.mean(sum_x):+.4e}, std={np.std(sum_x):.4e}")
        else:
            print(f"    {tf:6s}: {len(feats)} features (no count)")

    # proxy_feature_buffers per TF
    print(f"\n  --- proxy_feature_buffers (= deque 中身) ---")
    buffers = d["proxy_feature_buffers"]
    for tf in sorted(buffers.keys()):
        bufs = buffers[tf]
        if not bufs:
            continue
        x_deq = bufs.get("market_proxy", [])
        feat_count = sum(1 for k in bufs if k != "market_proxy")
        if x_deq:
            x_arr = np.asarray(x_deq, dtype=np.float64)
            x_finite = x_arr[np.isfinite(x_arr)]
            if len(x_finite) > 0:
                print(f"    {tf:6s}: x_deque len={len(x_deq):,}, "
                      f"x stats: mean={x_finite.mean():+.4e}, "
                      f"std={x_finite.std():.4e}, "
                      f"var={x_finite.var():.4e}, "
                      f"y_deques count={feat_count}")
            else:
                print(f"    {tf:6s}: x_deque len={len(x_deq):,} (all non-finite!)")
        else:
            print(f"    {tf:6s}: x_deque empty!  y_deques count={feat_count}")

    return {
        "path": path,
        "signal_timestamp": ts,
        "signal_timeframe": sig_tf,
        "feature_dict_size": len(d["feature_dict"]),
        "tfs_in_ols_state": list(ols_state.keys()),
        "tfs_in_buffers": list(buffers.keys()),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dump-dir", type=Path,
        default="/workspace/logs/ols_state_dumps",
    )
    args = parser.parse_args()

    dumps = sorted(args.dump_dir.glob("ols_*.pkl"))
    print(f"Found {len(dumps)} dump files in {args.dump_dir}")
    print()
    if not dumps:
        print("❌ dump ファイル無し")
        return

    summaries = []
    for p in dumps:
        try:
            summaries.append(inspect_one(p))
            print()
        except Exception as e:
            print(f"❌ {p.name}: {e}")
            import traceback; traceback.print_exc()

    # 全体サマリー
    print("=" * 72)
    print("  全体サマリー")
    print("=" * 72)
    print(f"  dump files: {len(summaries)}")
    if summaries:
        sigs_per_tf = Counter(s["signal_timeframe"] for s in summaries)
        print(f"  signal TF 分布: {dict(sigs_per_tf)}")
        ts_range = f"{summaries[0]['signal_timestamp']} 〜 {summaries[-1]['signal_timestamp']}"
        print(f"  timestamp range: {ts_range}")


if __name__ == "__main__":
    main()
