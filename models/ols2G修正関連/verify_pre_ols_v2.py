#!/usr/bin/env python3
"""
verify_pre_ols_v2.py — 学習側 engine_1_X と 本番側 rfe_1X の pre-OLS bit-identical 検証 v2

[v1 からの改修]
  1. --timeframe オプション追加 — TF を指定可能。 本番側 lookback_bars を TF から
     自動算出して学習側と整合させる (v1 では本番 lookback_bars=1440 固定、 学習側
     timeframe="M3" → 480 で 1.09% 乖離していた artifact を解消)。
  2. engine 別 interface 分岐:
       e1a/e1b/e1d: _get_all_feature_expressions(timeframe=tf) → Dict[str, pl.Expr]
       e1c:         _get_all_feature_expressions(lazy_frame, timeframe=tf) → pl.LazyFrame
       e1e/e1f:     _get_all_feature_expressions() → Dict[str, pl.Expr] (引数なし)

[呼び出し例]
  python verify_pre_ols_v2.py                          # 全 engine, M3
  python verify_pre_ols_v2.py --timeframe M1           # 全 engine, M1
  python verify_pre_ols_v2.py --engine e1c             # e1c のみ
  python verify_pre_ols_v2.py --timeframe M0.5 --n 3000  # M0.5, データ長 3000
"""

from __future__ import annotations

import argparse
import importlib
import shutil
import sys
import traceback
from typing import Dict, Optional

import numpy as np
import polars as pl

# ════════════════════════════════════════════════════════════════
# Path setup
# ════════════════════════════════════════════════════════════════
sys.path.insert(0, "/workspace")
sys.path.insert(0, "/workspace/features")
sys.path.insert(0, "/workspace/execution")

# ════════════════════════════════════════════════════════════════
# Engine 別 module / class / interface pattern
# ════════════════════════════════════════════════════════════════
ENGINE_INFO = {
    "e1a": {"suffix": "statistics",   "pattern": "timeframe"},
    "e1b": {"suffix": "timeseries",   "pattern": "timeframe"},
    "e1c": {"suffix": "technical",    "pattern": "lazy_frame"},
    "e1d": {"suffix": "volume",       "pattern": "timeframe"},
    "e1e": {"suffix": "signal",       "pattern": "noargs"},
    "e1f": {"suffix": "experimental", "pattern": "noargs"},
}

# TIMEFRAME_BARS_PER_DAY (本番側 realtime_feature_engine.py L49 と一致)
TF_BARS_PER_DAY = {
    "M0.5": 2880, "M1": 1440, "M3": 480,
    "M5":   288,  "M8":  180,  "M15": 96,
}


# ════════════════════════════════════════════════════════════════
# 合成データ生成 (v1 と同一)
# ════════════════════════════════════════════════════════════════
def generate_synthetic_data(n: int, seed: int = 42) -> Dict[str, np.ndarray]:
    np.random.seed(seed)
    returns = np.random.randn(n) * 0.001
    close = 100.0 * np.exp(np.cumsum(returns))
    spread = np.abs(np.random.randn(n) * 0.05)
    high = close + spread
    low = close - spread
    open_arr = np.empty_like(close)
    open_arr[0] = close[0]
    open_arr[1:] = close[:-1] + np.random.randn(n - 1) * 0.01
    volume = np.random.randint(1, 100, n).astype(np.float64)

    return {
        "open":   open_arr.astype(np.float64),
        "high":   high.astype(np.float64),
        "low":    low.astype(np.float64),
        "close":  close.astype(np.float64),
        "volume": volume,
    }


# ════════════════════════════════════════════════════════════════
# 学習側計算 (engine 別パターン分岐)
# ════════════════════════════════════════════════════════════════
def compute_learning_features(
    engine_id: str,
    timeframe: str,
    data: Dict[str, np.ndarray],
    prod_columns: Dict[str, np.ndarray],
    learning_mod,
):
    """engine 別の interface に応じて学習側を実行し、最新 1 行を Dict として返す。"""
    info = ENGINE_INFO[engine_id]
    pattern = info["pattern"]

    ProcessingConfig  = learning_mod.ProcessingConfig
    CalculationEngine = learning_mod.CalculationEngine

    config = ProcessingConfig(engine_id=engine_id)
    ce = CalculationEngine(config)

    # DataFrame 構築 (本番側 _build_polars_pieces で必要な列を全部含める)
    df_cols = {**prod_columns}
    # data の OHLCV も追加 (重複しなければ)
    for k, v in data.items():
        if k not in df_cols:
            df_cols[k] = v

    try:
        if pattern == "timeframe":
            # e1a, e1b, e1d: Dict[str, pl.Expr] を返す
            expressions = ce._get_all_feature_expressions(timeframe=timeframe)
            df_input = pl.DataFrame(df_cols)
            result_df = df_input.lazy().select(list(expressions.values())).collect()

        elif pattern == "lazy_frame":
            # e1c: lazy_frame を引数に取り、 LazyFrame を返す
            lazy_input = pl.LazyFrame(df_cols)
            result_lazy = ce._get_all_feature_expressions(
                lazy_frame=lazy_input, timeframe=timeframe
            )
            result_df = result_lazy.collect()

        elif pattern == "noargs":
            # e1e, e1f: 引数なし、 Dict[str, pl.Expr] を返す
            expressions = ce._get_all_feature_expressions()
            df_input = pl.DataFrame(df_cols)
            result_df = df_input.lazy().select(list(expressions.values())).collect()

        else:
            raise ValueError(f"unknown pattern: {pattern}")

    finally:
        # 一時ディレクトリ片付け
        try:
            if hasattr(ce, "temp_dir") and ce.temp_dir.exists():
                shutil.rmtree(ce.temp_dir)
        except Exception:
            pass

    # 末尾 1 行 → Dict[str, float]
    tail_row = result_df.tail(1).to_dicts()[0]
    out = {}
    for k, v in tail_row.items():
        if v is None:
            out[k] = np.nan
        else:
            try:
                fv = float(v)
                out[k] = fv if np.isfinite(fv) else np.nan
            except (TypeError, ValueError):
                out[k] = np.nan
    return out


# ════════════════════════════════════════════════════════════════
# Verification (1 engine)
# ════════════════════════════════════════════════════════════════
def run_verification(
    engine_id: str,
    timeframe: str = "M3",
    n_synth: int = 2000,
    seed: int = 42,
    show_fails: int = 15,
    verbose: bool = True,
) -> Optional[Dict]:

    if engine_id not in ENGINE_INFO:
        print(f"  ❌ unknown engine_id: {engine_id}")
        return None

    info = ENGINE_INFO[engine_id]
    letter = engine_id[-1].upper()
    lookback_bars = TF_BARS_PER_DAY[timeframe]

    if verbose:
        print("=" * 72)
        print(f"  Engine {engine_id.upper()} pre-OLS Bit-Identical Verification (v2)")
        print("=" * 72)
        print(f"  timeframe={timeframe}, lookback_bars={lookback_bars}, "
              f"N={n_synth}, seed={seed}, pattern={info['pattern']}")

    # ─── 動的 import ─────────────────────────────────────
    try:
        learning_mod = importlib.import_module(
            f"engine_1_{letter}_a_vast_universe_of_features"
        )
        production_mod = importlib.import_module(
            f"realtime_feature_engine_1{letter}_{info['suffix']}"
        )
        FeatureModule = getattr(production_mod, f"FeatureModule1{letter}")
    except Exception as e:
        print(f"  ❌ import 失敗: {e}")
        traceback.print_exc()
        return None

    # ─── 合成データ ────────────────────────────────────
    data = generate_synthetic_data(n_synth, seed)

    # ─── 本番側計算 (TF 別 lookback_bars を渡す!) ────────
    if verbose:
        print(f"  --- Production (rfe) 計算中... lookback_bars={lookback_bars} ---")
    try:
        production_features = FeatureModule.calculate_features(
            data, lookback_bars=lookback_bars
        )
    except Exception as e:
        print(f"  ❌ Production calculation 失敗: {e}")
        traceback.print_exc()
        return None
    if verbose:
        print(f"     production features: {len(production_features)} keys")

    # ─── 本番側 _build_polars_pieces で columns 取得 ──────
    try:
        prod_columns, _, _ = FeatureModule._build_polars_pieces(
            data, lookback_bars=lookback_bars
        )
    except Exception as e:
        if verbose:
            print(f"     (info) _build_polars_pieces 失敗、 columns 注入なし: {e}")
        prod_columns = {}

    # ATR13 を学習側 __temp_atr_13 用に注入
    if "__temp_atr_13" not in prod_columns:
        try:
            from realtime_feature_engine_1A_statistics import calculate_atr_wilder
            atr_arr = calculate_atr_wilder(
                data["high"], data["low"], data["close"], 13
            ) + 1e-10
            prod_columns["__temp_atr_13"] = atr_arr
        except Exception as e:
            print(f"  ❌ ATR13 計算失敗: {e}")
            return None

    # ─── 学習側計算 ────────────────────────────────────
    if verbose:
        print(f"  --- Learning (engine) 計算中... timeframe={timeframe} ---")
    try:
        learning_features = compute_learning_features(
            engine_id, timeframe, data, prod_columns, learning_mod
        )
    except Exception as e:
        print(f"  ❌ Learning calculation 失敗: {e}")
        traceback.print_exc()
        return None
    if verbose:
        print(f"     learning features: {len(learning_features)} keys")

    # ─── Alias 集合比較 ─────────────────────────────────
    prod_set  = set(production_features.keys())
    learn_set = set(learning_features.keys())
    common      = prod_set & learn_set
    prod_only   = prod_set - learn_set
    learn_only  = learn_set - prod_set

    # learn_only は OHLCV や中間列が含まれる可能性があるので、
    # alias prefix がエンジン規約 "e1{letter}_" に合致するもののみカウント
    expected_prefix = f"e1{engine_id[-1]}_"
    learn_only_features = {k for k in learn_only if k.startswith(expected_prefix)}
    learn_only_other    = learn_only - learn_only_features

    if verbose:
        print()
        print("  === Alias 集合比較 ===")
        print(f"     共通:                    {len(common)}")
        print(f"     本番のみ:                {len(prod_only)}")
        if prod_only:
            print(f"       例: {sorted(prod_only)[:5]}")
        print(f"     学習側のみ (特徴量):     {len(learn_only_features)}")
        if learn_only_features:
            print(f"       例: {sorted(learn_only_features)[:5]}")
        if learn_only_other:
            print(f"     学習側のみ (中間列等):   {len(learn_only_other)} (比較対象外)")

    # ─── 数値比較 ────────────────────────────────────────
    pass_count = 0
    fail_count = 0
    nan_match  = 0
    max_diff = 0.0
    max_diff_key = None
    fails = []

    for k in sorted(common):
        v_p = production_features[k]
        v_l = learning_features[k]

        try:
            v_p_f = float(v_p) if v_p is not None else np.nan
            if not np.isfinite(v_p_f):
                v_p_f = np.nan
        except (TypeError, ValueError):
            v_p_f = np.nan
        v_l_f = v_l if isinstance(v_l, float) else np.nan

        if np.isnan(v_p_f) and np.isnan(v_l_f):
            nan_match += 1
            pass_count += 1
            continue
        if np.isnan(v_p_f) or np.isnan(v_l_f):
            fails.append((k, v_p_f, v_l_f, "NaN mismatch"))
            fail_count += 1
            continue

        diff = abs(v_p_f - v_l_f)
        rel  = diff / (abs(v_l_f) + 1e-12)

        if diff > max_diff:
            max_diff = diff
            max_diff_key = k

        if rel < 1e-7 or diff < 1e-12:
            pass_count += 1
        else:
            fails.append((k, v_p_f, v_l_f, f"rel={rel:.2e}, abs={diff:.2e}"))
            fail_count += 1

    if verbose:
        print()
        print("  === 数値比較 (rtol=1e-7 OR atol=1e-12) ===")
        print(f"     PASS: {pass_count} (NaN-match: {nan_match})")
        print(f"     FAIL: {fail_count}")
        print(f"     max_abs_diff: {max_diff:.2e}  ({max_diff_key})")

        if fails:
            print()
            print(f"  === FAIL 先頭 {show_fails} 件 ===")
            for k, v_p, v_l, info_str in fails[:show_fails]:
                print(f"     {k}")
                print(f"       prod={v_p}, learn={v_l}  ({info_str})")

        # 結論
        print()
        if fail_count == 0 and len(prod_only) == 0 and len(learn_only_features) == 0:
            print(f"  ✅ {engine_id.upper()}: 完全 bit-identical")
        elif fail_count == 0:
            print(f"  ⚠️  {engine_id.upper()}: 数値一致だが alias 集合差あり "
                  f"(prod_only={len(prod_only)}, learn_only={len(learn_only_features)})")
        else:
            print(f"  ❌ {engine_id.upper()}: 数値乖離 {fail_count} FAIL")
        print()

    return {
        "engine":              engine_id,
        "timeframe":           timeframe,
        "common":              len(common),
        "prod_only":           len(prod_only),
        "learn_only_features": len(learn_only_features),
        "pass":                pass_count,
        "fail":                fail_count,
        "max_diff":            max_diff,
        "max_diff_key":        max_diff_key,
    }


# ════════════════════════════════════════════════════════════════
# main
# ════════════════════════════════════════════════════════════════
def main():
    parser = argparse.ArgumentParser(description="pre-OLS bit-identical 検証 v2")
    parser.add_argument("--engine", default="all", help="e1a/e1b/e1c/e1d/e1e/e1f or 'all'")
    parser.add_argument("--timeframe", default="M3",
                        help="検証する timeframe (M0.5/M1/M3/M5/M8/M15, default M3)")
    parser.add_argument("--n", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--show-fails", type=int, default=15)
    args = parser.parse_args()

    if args.timeframe not in TF_BARS_PER_DAY:
        print(f"❌ unknown timeframe: {args.timeframe} (valid: {list(TF_BARS_PER_DAY.keys())})")
        sys.exit(1)

    engines = (
        ["e1a", "e1b", "e1c", "e1d", "e1e", "e1f"]
        if args.engine == "all" else [args.engine]
    )

    summary = []
    for eng in engines:
        try:
            r = run_verification(eng, args.timeframe, args.n, args.seed, args.show_fails)
            summary.append(r)
        except Exception as e:
            print(f"  ❌ {eng}: Exception: {e}")
            traceback.print_exc()
            summary.append(None)

    # 最終サマリー
    print()
    print("=" * 72)
    print(f"  全 Engine 検証サマリー (timeframe={args.timeframe})")
    print("=" * 72)
    print(f"  {'engine':8s} {'common':>8s} {'prod_only':>10s} {'learn_only':>11s} "
          f"{'PASS':>6s} {'FAIL':>6s} {'max_diff':>12s}")
    print("  " + "─" * 70)
    for r in summary:
        if r is None:
            continue
        ok_set    = (r["prod_only"] == 0 and r["learn_only_features"] == 0)
        ok_num    = (r["fail"] == 0)
        status = "✅" if (ok_set and ok_num) else ("⚠️" if ok_num else "❌")
        print(f"  {status} {r['engine']:6s} {r['common']:>8d} {r['prod_only']:>10d} "
              f"{r['learn_only_features']:>11d} {r['pass']:>6d} {r['fail']:>6d} "
              f"{r['max_diff']:>12.2e}")
    print()


if __name__ == "__main__":
    main()
