#!/usr/bin/env python3
"""
[Layer 1] Shadow Mode 差分テスト — エントリーポイント

使い方:
    python -m shadow_mode.run_shadow_test \\
        --period 2026-04-01:2026-05-01 \\
        --scenario continuous \\
        --output /workspace/shadow_mode/reports/run_$(date +%Y%m%d_%H%M%S) \\
        --rtol 1e-7 --atol 1e-12

引数:
    --period       : 比較期間 (test) 'YYYY-MM-DD:YYYY-MM-DD'
    --warmup-days  : warmup 日数 (default 30; 全 TF の OLS_WINDOW_PER_TF を満たす)
    --scenario     : 'continuous' のみ (v1)
    --output       : レポート出力先ディレクトリ (default: ./reports/run_<ts>/)
    --rtol         : 相対許容誤差 (default 1e-7)
    --atol         : 絶対許容誤差 (default 1e-12)
    --m05-path     : M0.5 parquet (default: blueprint.S1_MULTITIMEFRAME/timeframe=M0.5)
    --s2-path      : S2 reference (default: blueprint.S2_FEATURES_VALIDATED)
    --feature-list : feature_list.txt (default: blueprint.S3_FEATURES_FOR_TRAINING_V5)
    --timeframes   : 比較対象 TF (default: M0.5,M1,M3,M5,M8,M15)
    --cache-dir    : ウォームアップ snapshot キャッシュ先 (default: ./.warmup_cache/)
    --no-cache     : キャッシュ無効化 (常にフルウォームアップ + 保存しない)

ウォームアップキャッシュ:
    同じ warmup_end_ts (= --period の開始日) で繰り返し実行する際、
    2 回目以降は warmup を skip してキャッシュから状態復元する。
    キャッシュキーは (warmup_end_ts, feature_list mtime, production engine
    code mtime, NEUTRALIZATION_CONFIG window_per_tf) から生成され、いずれが
    変わっても自動的に無効化される。

    例:
        # 初回: 1 日テスト → フルウォームアップ + キャッシュ作成
        python -m shadow_mode.run_shadow_test --period 2026-04-01:2026-04-02

        # 2 回目: 同じ 04-01 起点で 1 ヶ月テスト → キャッシュヒットで即時開始
        python -m shadow_mode.run_shadow_test --period 2026-04-01:2026-05-01

        # 別の起点に変えると新たにフルウォームアップ
        python -m shadow_mode.run_shadow_test --period 2026-04-15:2026-05-15

終了コード:
    0 = 全特徴量一致 (PASS)
    1 = 一部以上不一致 (FAIL)
    2 = エラー (例外、データ欠損等)
"""

import argparse
import hashlib
import json
import logging
import os
import pickle
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

# Allow imports from /workspace
_WORKSPACE = Path(__file__).resolve().parents[1]
if str(_WORKSPACE) not in sys.path:
    sys.path.insert(0, str(_WORKSPACE))

import blueprint as config  # noqa: E402

from shadow_mode.replay_bridge import ReplayBridge  # noqa: E402
from shadow_mode.feature_capturer import ShadowEngine  # noqa: E402
from shadow_mode.reference_builder import ReferenceBuilder  # noqa: E402
from shadow_mode.diff_aggregator import DiffAggregator  # noqa: E402
from shadow_mode.diff_report import DiffReport  # noqa: E402
from shadow_mode.stress_injector import StressInjector  # noqa: E402

logger = logging.getLogger("shadow_mode.runner")


# ─────────────────────────────────────────────────────────────────────
# Helper: market_proxy initialization (main.py L325-365 を要約)
# ─────────────────────────────────────────────────────────────────────


def build_initial_market_proxy(m05_warmup_df: pd.DataFrame) -> pd.DataFrame:
    """warmup M0.5 から M5 close-based market_proxy を生成。

    main.py の initialize_data_buffer() 内の処理と数式上等価:
        m5_close = m05_close.resample("5min", label="left", closed="left").last().dropna()
        proxy = m5_close.pct_change(1).to_frame("market_proxy").dropna()
    """
    if m05_warmup_df.empty:
        return pd.DataFrame(
            columns=["market_proxy"],
            index=pd.DatetimeIndex([], tz="UTC", name="timestamp"),
        )
    df = m05_warmup_df[["timestamp", "close"]].copy()
    df = df.set_index("timestamp")
    if df.index.tz is None:
        df.index = df.index.tz_localize("UTC")
    else:
        df.index = df.index.tz_convert("UTC")

    m5_close = (
        df["close"]
        .resample("5min", label="left", closed="left")
        .last()
        .dropna()
    )
    proxy = m5_close.pct_change(1).to_frame(name="market_proxy").dropna()
    if proxy.index.tz is None:
        proxy = proxy.tz_localize("UTC")
    else:
        proxy = proxy.tz_convert("UTC")
    return proxy


def update_market_proxy(
    market_proxy: pd.DataFrame, engine: ShadowEngine
) -> pd.DataFrame:
    """test loop で M3 通知到着相当のタイミングで market_proxy を更新。

    main.py L968-995 と数式上等価。engine.m05_dataframe[-20:] から
    最新 M5 close を計算して proxy にアペンド。
    """
    if len(engine.m05_dataframe) < 20:
        return market_proxy
    recent = pd.DataFrame(list(engine.m05_dataframe)[-20:]).set_index(
        "timestamp"
    )
    m5 = (
        recent["close"]
        .resample("5min", label="left", closed="left")
        .last()
        .dropna()
    )
    if len(m5) < 2:
        return market_proxy

    new_proxy_val = (float(m5.iloc[-1]) - float(m5.iloc[-2])) / (
        float(m5.iloc[-2]) + 1e-10
    )
    new_proxy_df = pd.DataFrame(
        {"market_proxy": [new_proxy_val]},
        index=pd.DatetimeIndex([m5.index[-1]], tz="UTC"),
    )
    if market_proxy.empty:
        return new_proxy_df
    if m5.index[-1] not in market_proxy.index:
        market_proxy = pd.concat([market_proxy, new_proxy_df])
    if len(market_proxy) > 10000:
        market_proxy = market_proxy.iloc[-5000:]
    return market_proxy


# ─────────────────────────────────────────────────────────────────────
# Helper: warmup snapshot cache
# ─────────────────────────────────────────────────────────────────────
#
# 同じ warmup_end_ts でテスト期間だけ変える反復実行のとき、ウォームアップを
# 再実行せずキャッシュから復元する。
#
# キャッシュキーは以下から生成:
#   - warmup_end_ts (ISO 文字列)
#   - feature_list ファイルの mtime (内容変更検出のプロキシ)
#   - production realtime_feature_engine.py の mtime
#     (Layer 1 が読み込む production コードが変わったらキャッシュ無効化)
#   - blueprint.NEUTRALIZATION_CONFIG["HF"]["window_per_tf"] の値
#     (発見 #63 のような OLS 窓設定変更で自動的に無効化)
#
# キャッシュは安全側のみ作用: ヒットしたら warmup スキップ、
# ミスや例外なら通常のフルウォームアップを実行 (機能は壊さない)。


def _compute_warmup_cache_key(
    warmup_end_ts: pd.Timestamp,
    feature_list_path: Path,
    engine_code_path: Path,
) -> str:
    """ウォームアップキャッシュのキーを計算する。"""
    parts: List[str] = []
    parts.append(f"end:{warmup_end_ts.isoformat()}")

    if feature_list_path.exists():
        try:
            parts.append(f"flist:{feature_list_path.stat().st_mtime:.3f}")
        except OSError:
            parts.append(f"flist:nostat")
    else:
        parts.append(f"flist:missing")

    if engine_code_path.exists():
        try:
            parts.append(f"engine:{engine_code_path.stat().st_mtime:.3f}")
        except OSError:
            parts.append(f"engine:nostat")
    else:
        parts.append(f"engine:missing")

    # blueprint NEUTRALIZATION_CONFIG (発見 #63 関連)
    try:
        ols_cfg = config.NEUTRALIZATION_CONFIG.get("HF", {}).get(
            "window_per_tf", {}
        )
        ols_str = json.dumps(ols_cfg, sort_keys=True)
        parts.append(f"ols:{ols_str}")
    except Exception:
        parts.append("ols:default")

    raw = "|".join(parts)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]


def _load_warmup_cache(
    cache_file: Path, engine: ShadowEngine
) -> Optional[pd.DataFrame]:
    """キャッシュからウォームアップ状態を復元。失敗時は None。"""
    if not cache_file.exists():
        return None
    try:
        with open(cache_file, "rb") as f:
            data = pickle.load(f)
    except Exception as e:
        logger.warning(f"  Cache file load failed ({e}) — will full-warmup")
        return None

    try:
        # production engine の state 変数を直接書き戻す
        # (production の load_state と同等の操作)
        engine.data_buffers = data["data_buffers"]
        engine.is_buffer_filled = data["is_buffer_filled"]
        engine.last_bar_timestamps = data["last_bar_timestamps"]
        engine.latest_features_cache = data["latest_features_cache"]
        engine.m05_dataframe = data["m05_dataframe"]
        engine.proxy_feature_buffers = data["proxy_feature_buffers"]
        engine.ols_state = data["ols_state"]
        engine.qa_states = data["qa_states"]
        market_proxy = data["market_proxy"]
        return market_proxy
    except KeyError as e:
        logger.warning(
            f"  Cache file missing key {e} — will full-warmup "
            f"(cache format changed?)"
        )
        return None
    except Exception as e:
        logger.warning(f"  Cache restore failed ({e}) — will full-warmup")
        return None


def _save_warmup_cache(
    cache_file: Path,
    engine: ShadowEngine,
    market_proxy: pd.DataFrame,
    metadata: Dict[str, Any],
) -> None:
    """ウォームアップ状態をキャッシュに保存 (atomic write)。"""
    cache_file.parent.mkdir(parents=True, exist_ok=True)
    tmp_file = cache_file.with_suffix(cache_file.suffix + ".tmp")

    data = {
        "data_buffers": engine.data_buffers,
        "is_buffer_filled": engine.is_buffer_filled,
        "last_bar_timestamps": engine.last_bar_timestamps,
        "latest_features_cache": engine.latest_features_cache,
        "m05_dataframe": engine.m05_dataframe,
        "proxy_feature_buffers": engine.proxy_feature_buffers,
        "ols_state": engine.ols_state,
        "qa_states": engine.qa_states,
        "market_proxy": market_proxy,
        "metadata": metadata,
    }

    try:
        with open(tmp_file, "wb") as f:
            pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_file, cache_file)
    except Exception as e:
        logger.error(f"  Cache save failed: {e}")
        if tmp_file.exists():
            try:
                tmp_file.unlink()
            except OSError:
                pass


# ─────────────────────────────────────────────────────────────────────
# Main runner
# ─────────────────────────────────────────────────────────────────────


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Shadow Mode Layer 1 — diff test against learning side S2"
    )
    p.add_argument(
        "--period", required=True,
        help="Test period 'YYYY-MM-DD:YYYY-MM-DD' (UTC)"
    )
    p.add_argument(
        "--warmup-days", type=int, default=30,
        help="Warmup history days before test period (default 30)"
    )
    p.add_argument(
        "--scenario", default="continuous",
        help="Stress scenario name (v1: continuous only)"
    )
    p.add_argument(
        "--output", type=Path, default=None,
        help="Report output directory (default: ./reports/run_<timestamp>/)"
    )
    p.add_argument("--rtol", type=float, default=1e-7)
    p.add_argument("--atol", type=float, default=1e-12)
    p.add_argument(
        "--m05-path", type=Path, default=None,
        help="M0.5 parquet directory (default: S1_MULTITIMEFRAME/timeframe=M0.5)"
    )
    p.add_argument(
        "--s2-path", type=Path, default=None,
        help="S2 reference root (default: S2_FEATURES_VALIDATED)"
    )
    p.add_argument(
        "--feature-list", type=Path, default=None,
        help="Feature list (default: S3_FEATURES_FOR_TRAINING_V5)"
    )
    p.add_argument(
        "--timeframes", default="M0.5,M1,M3,M5,M8,M15",
        help="Comma-separated TFs to compare"
    )
    p.add_argument(
        "--cache-dir", type=Path, default=None,
        help="Warmup snapshot cache directory "
             "(default: <shadow_mode>/.warmup_cache/)"
    )
    p.add_argument(
        "--no-cache", action="store_true",
        help="Skip warmup cache (force full warmup, do not save)"
    )
    p.add_argument(
        "--log-level", default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"]
    )
    return p.parse_args()


def setup_logging(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )


def parse_period(period_str: str) -> tuple:
    parts = period_str.split(":")
    if len(parts) != 2:
        raise ValueError(
            f"--period must be 'YYYY-MM-DD:YYYY-MM-DD', got '{period_str}'"
        )
    start = pd.Timestamp(parts[0]).tz_localize("UTC")
    end = pd.Timestamp(parts[1]).tz_localize("UTC")
    if start >= end:
        raise ValueError(f"Period start ({start}) >= end ({end})")
    return start, end


def resolve_paths(args: argparse.Namespace) -> dict:
    m05_path = args.m05_path
    if m05_path is None:
        m05_path = Path(config.S1_MULTITIMEFRAME) / "timeframe=M0.5"
    s2_path = args.s2_path
    if s2_path is None:
        s2_path = Path(config.S2_FEATURES_VALIDATED)
    feature_list = args.feature_list
    if feature_list is None:
        feature_list = Path(config.S3_FEATURES_FOR_TRAINING_V5)

    output = args.output
    if output is None:
        ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        output = Path(__file__).parent / "reports" / f"run_{ts}"

    cache_dir = args.cache_dir
    if cache_dir is None:
        cache_dir = Path(__file__).parent / ".warmup_cache"

    return {
        "m05_path": m05_path,
        "s2_path": s2_path,
        "feature_list": feature_list,
        "output": Path(output),
        "cache_dir": Path(cache_dir),
    }


def main() -> int:
    args = parse_args()
    setup_logging(args.log_level)

    test_start, test_end = parse_period(args.period)
    warmup_end = test_start  # warmup includes all bars up to test_start (exclusive)
    warmup_start = warmup_end - pd.Timedelta(days=args.warmup_days)

    paths = resolve_paths(args)
    timeframes = [tf.strip() for tf in args.timeframes.split(",")]

    logger.info("=" * 70)
    logger.info("Shadow Mode Layer 1 — diff test")
    logger.info("=" * 70)
    logger.info(f"  Test period:    {test_start} → {test_end}")
    logger.info(f"  Warmup period:  {warmup_start} → {warmup_end} "
                f"({args.warmup_days} days)")
    logger.info(f"  Scenario:       {args.scenario}")
    logger.info(f"  M0.5 source:    {paths['m05_path']}")
    logger.info(f"  S2 reference:   {paths['s2_path']}")
    logger.info(f"  Feature list:   {paths['feature_list']}")
    logger.info(f"  Output dir:     {paths['output']}")
    logger.info(f"  Cache dir:      {paths['cache_dir']}"
                + (" (DISABLED)" if args.no_cache else ""))
    logger.info(f"  Timeframes:     {timeframes}")
    logger.info(f"  Tolerance:      rtol={args.rtol}, atol={args.atol}")
    logger.info("=" * 70)

    # ─── Step 1: Load M0.5 parquet via ReplayBridge ────────────────────
    logger.info("[Step 1/6] Loading M0.5 history via ReplayBridge")
    t0 = time.time()
    bridge = ReplayBridge(
        m05_parquet_path=paths["m05_path"],
        warmup_end_ts=warmup_end,
        test_end_ts=test_end,
    )
    summary = bridge.test_period_summary
    if summary["n_bars"] == 0:
        logger.error("Test period contains 0 M0.5 bars — abort")
        return 2
    logger.info(
        f"  warmup={bridge.total_warmup_bars()} bars, "
        f"test={bridge.total_test_bars()} bars "
        f"({time.time() - t0:.1f}s)"
    )

    # ─── Step 2: Initialize ShadowEngine + warmup (with cache) ─────────
    logger.info("[Step 2/6] Initializing ShadowEngine + warmup")
    t0 = time.time()
    engine = ShadowEngine(feature_list_path=str(paths["feature_list"]))

    # Cache key
    engine_code_path = _WORKSPACE / "execution" / "realtime_feature_engine.py"
    cache_key = _compute_warmup_cache_key(
        warmup_end_ts=warmup_end,
        feature_list_path=paths["feature_list"],
        engine_code_path=engine_code_path,
    )
    cache_file = paths["cache_dir"] / f"warmup_{cache_key}.pkl"

    market_proxy: Optional[pd.DataFrame] = None
    cache_hit = False
    if not args.no_cache:
        logger.info(f"  Cache key: {cache_key}")
        logger.info(f"  Cache file: {cache_file}")
        if cache_file.exists():
            market_proxy = _load_warmup_cache(cache_file, engine)
            if market_proxy is not None:
                cache_hit = True
                logger.info(
                    f"  ✓ Cache hit — warmup skipped "
                    f"({time.time() - t0:.1f}s)"
                )
        else:
            logger.info(f"  Cache miss — running full warmup")

    if not cache_hit:
        warmup_history = bridge.get_warmup_history()
        if warmup_history.empty:
            logger.error(
                "Warmup history is empty (warmup period has no M0.5 bars). "
                "Check --warmup-days and M0.5 parquet coverage."
            )
            return 2
        history_data_map = {"M0.5": warmup_history}
        market_proxy = build_initial_market_proxy(warmup_history)
        engine.fill_all_buffers(history_data_map, market_proxy)
        logger.info(
            f"  Full warmup done ({time.time() - t0:.1f}s)"
        )

        if not args.no_cache:
            t_save = time.time()
            metadata = {
                "warmup_end_ts": warmup_end.isoformat(),
                "feature_list": str(paths["feature_list"]),
                "engine_code_path": str(engine_code_path),
                "ols_windows": dict(
                    config.NEUTRALIZATION_CONFIG.get("HF", {}).get(
                        "window_per_tf", {}
                    )
                ),
                "warmup_bars": bridge.total_warmup_bars(),
            }
            _save_warmup_cache(cache_file, engine, market_proxy, metadata)
            logger.info(
                f"  Cache saved ({time.time() - t_save:.1f}s) — "
                f"future runs with same warmup_end_ts skip warmup"
            )

    logger.info(
        f"  Engine ready: market_proxy={len(market_proxy)} rows"
    )

    # ─── Step 3: Run test period (capture enabled) ─────────────────────
    logger.info("[Step 3/6] Running test period replay (capture enabled)")
    engine.enable_capture()
    injector = StressInjector(scenario=args.scenario)

    t0 = time.time()
    n_processed = 0
    n_total = bridge.total_test_bars()
    progress_step = max(1, n_total // 20)

    for bar in injector.transform(bridge.iter_test_bars()):
        # Update market_proxy at boundaries
        # (production main.py does this on every M3 notify; we do per bar
        #  but DEDUP inside update prevents redundant entries)
        market_proxy = update_market_proxy(market_proxy, engine)

        # Process bar with warmup_only=True (no signal generation needed
        # for shadow mode; we only care about feature values)
        engine.process_new_m05_bar(bar, market_proxy, warmup_only=True)

        n_processed += 1
        if n_processed % progress_step == 0:
            pct = n_processed / n_total * 100
            logger.info(
                f"  progress: {n_processed}/{n_total} ({pct:.1f}%), "
                f"captured={engine.captured_count()}"
            )
    elapsed = time.time() - t0
    logger.info(
        f"  test loop done: {n_processed} bars processed, "
        f"{engine.captured_count()} captures ({elapsed:.1f}s, "
        f"{n_processed / max(elapsed, 0.001):.0f} bars/sec)"
    )

    # ─── Step 4: Build long-format captured + reference ────────────────
    logger.info("[Step 4/6] Building long-format DataFrames")
    t0 = time.time()
    captured_long = engine.to_long_format()
    logger.info(
        f"  captured: {len(captured_long):,} rows, "
        f"{captured_long['feature_name'].nunique() if not captured_long.empty else 0} features"
    )

    ref_builder = ReferenceBuilder(
        s2_root=paths["s2_path"],
        timeframes=timeframes,
        test_start_ts=test_start,
        test_end_ts=test_end,
    )
    reference_long = ref_builder.build()
    logger.info(
        f"  reference: {len(reference_long):,} rows "
        f"({time.time() - t0:.1f}s)"
    )

    # ─── Step 5: Diff aggregation ──────────────────────────────────────
    logger.info("[Step 5/6] Diff aggregation")
    t0 = time.time()
    aggregator = DiffAggregator(rtol=args.rtol, atol=args.atol)
    result = aggregator.compare(captured_long, reference_long)
    logger.info(f"  done ({time.time() - t0:.1f}s)")

    # ─── Step 6: Write reports ─────────────────────────────────────────
    logger.info("[Step 6/6] Writing reports")
    report = DiffReport(output_dir=paths["output"])
    context = {
        "test_period": f"{test_start} → {test_end}",
        "warmup_days": str(args.warmup_days),
        "scenario": args.scenario,
        "n_bars_processed": str(n_processed),
        "n_captures": str(engine.captured_count()),
        "rtol": str(args.rtol),
        "atol": str(args.atol),
        "timeframes": ",".join(timeframes),
        "warmup_cache": "hit" if cache_hit else (
            "saved" if not args.no_cache else "disabled"
        ),
        "git_status": "(not captured)",
    }
    report.write_all(result, context)

    # ─── Verdict ───────────────────────────────────────────────────────
    stats = result["stats"]
    logger.info("=" * 70)
    if stats.is_pass():
        logger.info(
            f"✅ PASS — all {stats.total:,} compared rows match within tolerance"
        )
        return 0
    else:
        logger.error(
            f"❌ FAIL — {stats.failed:,}/{stats.total:,} rows out of tolerance "
            f"({stats.fail_rate * 100:.4f}%)"
        )
        logger.error(f"   See {paths['output']}/summary.md for details")
        return 1


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as e:
        logger.exception(f"Fatal error: {e}")
        sys.exit(2)
