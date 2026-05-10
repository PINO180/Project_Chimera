#!/usr/bin/env python3
"""
[Shadow Mode] 仮説 α 検証用診断スクリプト

production の warmup 後 deque 状態 (M0.5/M1/M3/M5/M8/M15) と
master_multitimeframe の対応する parquet を bar-by-bar で突合する。

結果から以下が判別できる:
  - OHLCV のどの列が違うか (open/high/low/close/volume)
  - 全 TF で違うのか、特定 TF だけか
  - 違いの大きさ (絶対 / 相対)
  - 1 本だけ違うのか、複数本違うのか
  - timestamp が ずれているのか、値がずれているのか

使い方:
    cd /workspace
    python -m shadow_mode.diagnostic_compare \\
        --warmup-end 2026-04-01 \\
        --warmup-days 30

オプション:
    --warmup-end : warmup 境界 (= 前回 run と同じ値で cache hit)
    --warmup-days: warmup 期間 (default 30)
    --n-tail     : 各 TF で deque 末尾 N 本を比較 (default 5)
    --no-cache   : フルウォームアップ強制
"""

import argparse
import logging
import sys
import time
from pathlib import Path
from typing import List, Optional, Dict

import numpy as np
import pandas as pd

# Allow imports from /workspace
_WORKSPACE = Path(__file__).resolve().parents[1]
if str(_WORKSPACE) not in sys.path:
    sys.path.insert(0, str(_WORKSPACE))

import blueprint as config  # noqa: E402

from shadow_mode.replay_bridge import ReplayBridge  # noqa: E402
from shadow_mode.feature_capturer import ShadowEngine  # noqa: E402
from shadow_mode.run_shadow_test import (  # noqa: E402
    build_initial_market_proxy,
    _compute_warmup_cache_key,
    _load_warmup_cache,
)

logger = logging.getLogger("shadow_mode.diagnostic")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--warmup-end", default="2026-04-01",
                   help="Warmup boundary (default 2026-04-01)")
    p.add_argument("--warmup-days", type=int, default=30)
    p.add_argument("--n-tail", type=int, default=5,
                   help="Compare last N bars of each TF deque")
    p.add_argument("--no-cache", action="store_true")
    p.add_argument("--m05-path", type=Path, default=None)
    p.add_argument("--mtf-root", type=Path, default=None,
                   help="master_multitimeframe root")
    p.add_argument("--feature-list", type=Path, default=None)
    return p.parse_args()


def setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )


def load_reference_bar(
    mtf_root: Path, tf: str, ts: pd.Timestamp
) -> Optional[Dict[str, float]]:
    """master_multitimeframe / timeframe=<TF> から指定 timestamp の bar を読む。"""
    tf_dir = mtf_root / f"timeframe={tf}"
    if not tf_dir.exists():
        logger.warning(f"  [{tf}] reference dir not found: {tf_dir}")
        return None

    files = sorted(tf_dir.rglob("*.parquet"))
    if not files:
        logger.warning(f"  [{tf}] no parquet under {tf_dir}")
        return None

    # 全件ロードは重いので filter してから collect
    try:
        import polars as pl
        # pyarrow ベースで複数ファイル読み
        scan = pl.scan_parquet([str(f) for f in files])
        # timestamp 比較のため文字列に変換 (polars の datetime 比較は慎重に)
        target_naive = ts.tz_convert("UTC").tz_localize(None) \
            if ts.tz is not None else ts
        # まず timestamp dtype を取得
        schema = scan.collect_schema()
        ts_dtype = schema.get("timestamp")
        if ts_dtype is None:
            logger.warning(f"  [{tf}] no timestamp column")
            return None

        # filter を polars で適用
        try:
            df = scan.filter(
                pl.col("timestamp") == pl.lit(target_naive)
            ).collect()
        except Exception:
            # tz-aware として再試行
            df = scan.filter(
                pl.col("timestamp") == pl.lit(ts.tz_convert("UTC"))
            ).collect()

        if df.is_empty():
            return None
        row = df.to_dicts()[0]
        return {
            "open": float(row.get("open", 0.0)),
            "high": float(row.get("high", 0.0)),
            "low": float(row.get("low", 0.0)),
            "close": float(row.get("close", 0.0)),
            "volume": float(row.get("volume", 0.0)),
            "disc": bool(row.get("disc", False)) if "disc" in row else None,
            "_raw": row,
        }
    except Exception as e:
        logger.error(f"  [{tf}] reference load error: {e}")
        return None


def get_prod_bar_at(
    engine: ShadowEngine, tf: str, idx: int = -1
) -> Optional[Dict[str, float]]:
    """production deque から idx 番目 (default 末尾) の bar を取得。"""
    if tf not in engine.data_buffers:
        return None
    db = engine.data_buffers[tf]
    if "close" not in db or len(db["close"]) == 0:
        return None
    n = len(db["close"])
    if abs(idx) > n:
        return None

    def _at(col: str) -> float:
        return float(list(db[col])[idx])

    out = {
        "open": _at("open"),
        "high": _at("high"),
        "low": _at("low"),
        "close": _at("close"),
        "volume": _at("volume"),
    }
    if "disc" in db:
        try:
            out["disc"] = bool(list(db["disc"])[idx])
        except Exception:
            out["disc"] = None
    # last_bar_timestamps は最末尾しか保持しないので、idx=-1 のみ ts 取得
    if idx == -1:
        ts = engine.last_bar_timestamps.get(tf)
        if ts is not None:
            out["timestamp"] = ts
    return out


def fmt_bar(b: Dict[str, float]) -> str:
    return (
        f"O={b['open']:.5f} H={b['high']:.5f} L={b['low']:.5f} "
        f"C={b['close']:.5f} V={b['volume']:.0f}"
    )


def compare_bars(prod: dict, ref: dict, label: str) -> List[str]:
    """OHLCV を行毎に比較、差分があれば返す。"""
    diffs = []
    for col in ["open", "high", "low", "close", "volume"]:
        p = prod[col]
        r = ref[col]
        if abs(p - r) > 1e-7 * max(abs(r), 1.0):
            rel = (p - r) / r if r != 0 else float("inf")
            diffs.append(
                f"    {col:<6}: prod={p:.6f}  ref={r:.6f}  "
                f"diff={p - r:+.6f}  rel={rel:+.4%}"
            )
    if "disc" in prod and "disc" in ref and ref["disc"] is not None:
        if prod["disc"] != ref["disc"]:
            diffs.append(
                f"    disc  : prod={prod['disc']}  ref={ref['disc']}"
            )
    return diffs


def diagnose_tf(
    engine: ShadowEngine, mtf_root: Path, tf: str, n_tail: int
) -> None:
    """1 つの TF について deque 末尾 n_tail 本を比較。"""
    print(f"\n{'=' * 70}")
    print(f"  [{tf}]  deque 末尾 {n_tail} 本 を master_multitimeframe と比較")
    print(f"{'=' * 70}")

    if tf not in engine.data_buffers:
        print(f"  ❌ TF {tf} not in data_buffers — skip")
        return

    n_buf = len(engine.data_buffers[tf]["close"])
    if n_buf == 0:
        print(f"  ❌ deque empty — skip")
        return

    last_ts = engine.last_bar_timestamps.get(tf)
    print(f"  deque size: {n_buf} bars")
    print(f"  last_bar_timestamps[{tf}]: {last_ts}")

    if last_ts is None:
        print(f"  ❌ last_bar_timestamps unknown — skip")
        return

    # TF 毎の bar 間隔
    tf_freq_min = {
        "M0.5": 0.5, "M1": 1, "M3": 3, "M5": 5, "M8": 8, "M15": 15,
    }.get(tf)
    if tf_freq_min is None:
        print(f"  ❌ unknown TF freq — skip")
        return

    # 末尾 n_tail 本の timestamp を逆算で得る
    # (production deque は OHLCV のみで timestamp 列がないので、
    #  last_ts から step で計算する)
    timestamps = []
    for i in range(n_tail):
        ts = last_ts - pd.Timedelta(minutes=tf_freq_min * i)
        timestamps.append(ts)
    timestamps.reverse()  # 古い順

    n_ok = 0
    n_diff = 0
    n_missing_ref = 0
    for offset, ts in enumerate(timestamps):
        idx = -(n_tail - offset)  # -n_tail .. -1
        prod = get_prod_bar_at(engine, tf, idx=idx)
        if prod is None:
            print(f"  [{ts}] prod bar at idx={idx} missing")
            continue

        ref = load_reference_bar(mtf_root, tf, ts)
        if ref is None:
            print(f"  [{ts}] ref bar missing in master_multitimeframe")
            n_missing_ref += 1
            continue

        diffs = compare_bars(prod, ref, label=f"{tf} @ {ts}")
        if not diffs:
            n_ok += 1
            print(f"  [{ts}] ✅ match: {fmt_bar(prod)}")
        else:
            n_diff += 1
            print(f"  [{ts}] ❌ DIFFER:")
            print(f"    prod: {fmt_bar(prod)}")
            print(f"    ref : {fmt_bar(ref)}")
            for d in diffs:
                print(d)

    print(f"\n  {tf} 集計: match={n_ok}, differ={n_diff}, "
          f"ref_missing={n_missing_ref} / total={len(timestamps)}")


def main() -> int:
    args = parse_args()
    setup_logging()

    warmup_end = pd.Timestamp(args.warmup_end).tz_localize("UTC")
    warmup_start = warmup_end - pd.Timedelta(days=args.warmup_days)

    m05_path = args.m05_path or (
        Path(config.S1_MULTITIMEFRAME) / "timeframe=M0.5"
    )
    mtf_root = args.mtf_root or Path(config.S1_MULTITIMEFRAME)
    feature_list = args.feature_list or Path(config.S3_FEATURES_FOR_TRAINING_V5)

    logger.info("=" * 70)
    logger.info("Shadow Mode Diagnostic — production deque vs master_multitimeframe")
    logger.info("=" * 70)
    logger.info(f"  warmup_end:    {warmup_end}")
    logger.info(f"  warmup_start:  {warmup_start}")
    logger.info(f"  m05_path:      {m05_path}")
    logger.info(f"  mtf_root:      {mtf_root}")
    logger.info(f"  feature_list:  {feature_list}")
    logger.info("=" * 70)

    # Step 1: ReplayBridge (warmup-end までの履歴 + 先 1 日テスト想定)
    logger.info("[1/3] Loading M0.5 history...")
    t0 = time.time()
    bridge = ReplayBridge(
        m05_parquet_path=m05_path,
        warmup_end_ts=warmup_end,
        test_end_ts=warmup_end + pd.Timedelta(days=1),
    )
    logger.info(f"  loaded ({time.time() - t0:.1f}s)")

    # Step 2: ShadowEngine warmup (cache hit 想定)
    logger.info("[2/3] Initializing engine (cache hit expected)...")
    t0 = time.time()
    engine = ShadowEngine(feature_list_path=str(feature_list))

    engine_code_path = _WORKSPACE / "execution" / "realtime_feature_engine.py"
    cache_key = _compute_warmup_cache_key(
        warmup_end_ts=warmup_end,
        feature_list_path=feature_list,
        engine_code_path=engine_code_path,
    )
    cache_file = (
        Path(__file__).parent / ".warmup_cache" / f"warmup_{cache_key}.pkl"
    )

    market_proxy = None
    if not args.no_cache and cache_file.exists():
        logger.info(f"  loading cache: {cache_file}")
        market_proxy = _load_warmup_cache(cache_file, engine)

    if market_proxy is None:
        logger.info("  cache miss — running full warmup (will take minutes)")
        warmup_history = bridge.get_warmup_history()
        market_proxy = build_initial_market_proxy(warmup_history)
        engine.fill_all_buffers({"M0.5": warmup_history}, market_proxy)
    logger.info(f"  ready ({time.time() - t0:.1f}s)")

    # Step 3: 全 TF を比較
    logger.info("[3/3] Comparing deque vs master_multitimeframe...")
    timeframes = ["M0.5", "M1", "M3", "M5", "M8", "M15"]
    for tf in timeframes:
        diagnose_tf(engine, mtf_root, tf, n_tail=args.n_tail)

    print(f"\n{'=' * 70}")
    print("Diagnostic complete.")
    print(f"{'=' * 70}")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as e:
        logger.exception(f"Fatal: {e}")
        sys.exit(2)
