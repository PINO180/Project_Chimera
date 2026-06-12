#!/usr/bin/env python3
"""
verify_pre_ols.py — 学習側 engine_1_X と 本番側 realtime_feature_engine_1X の
                    pre-OLS 数値 bit-identical 検証 (全 engine 対応)

[目的]
  Phase 11 で確定した「post-OLS 99.6% 重度乖離」 の真因が:
    A. pre-OLS の差 (= engine_1_X と realtime_feature_engine_1X の計算差) から来るのか
    B. OLS 純化計算経路の差から来るのか
  を切り分けるため、 合成データで両側の pre-OLS 計算が bit-identical かを検証する。

  e1a で既に別セッションで PASS (max 1.7e-12) の結果がある。 本スクリプトで e1a も
  含めて再現性を確認し、 e1b〜e1f にも展開する。

[検証構造]
  1. 合成 M0.5 bars 生成 (N=2000、 random walk + 適度な volatility)
  2. 本番側: FeatureModule1X.calculate_features(data) → Dict[str, float] (最新 1 行)
  3. 学習側: CalculationEngine._get_all_feature_expressions() で Polars 式取得
            同じ data から DataFrame 作成 → select → tail(1) → Dict[str, float]
  4. alias 集合比較 + 数値比較 (rtol=1e-7 or atol=1e-12)

[呼び出し例]
  python verify_pre_ols.py                # 全 engine (e1a〜e1f)
  python verify_pre_ols.py --engine e1c   # e1c のみ
  python verify_pre_ols.py --n 3000       # 合成データ長を変更
"""

from __future__ import annotations

import argparse
import importlib
import shutil
import sys
import traceback
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import polars as pl

# ════════════════════════════════════════════════════════════════
# Path setup
# ════════════════════════════════════════════════════════════════
sys.path.insert(0, "/workspace")
sys.path.insert(0, "/workspace/features")   # 学習側 engine_1_X
sys.path.insert(0, "/workspace/execution")  # 本番側 realtime_feature_engine_1X

# ════════════════════════════════════════════════════════════════
# Engine 別の module / class 命名規約
# ════════════════════════════════════════════════════════════════
ENGINE_SUFFIXES = {
    "e1a": "statistics",
    "e1b": "timeseries",
    "e1c": "technical",
    "e1d": "volume",
    "e1e": "signal",
    "e1f": "experimental",
}


# ════════════════════════════════════════════════════════════════
# 合成データ生成
# ════════════════════════════════════════════════════════════════
def generate_synthetic_data(n: int, seed: int = 42) -> Dict[str, np.ndarray]:
    """random walk ベースの合成 OHLCV を生成"""
    np.random.seed(seed)
    returns = np.random.randn(n) * 0.001
    close = 100.0 * np.exp(np.cumsum(returns))
    # high/low は close からの spread (常に high >= close >= low)
    spread = np.abs(np.random.randn(n) * 0.05)
    high = close + spread
    low = close - spread
    # open は前 bar の close + 小さなギャップ
    open_arr = np.empty_like(close)
    open_arr[0] = close[0]
    open_arr[1:] = close[:-1] + np.random.randn(n - 1) * 0.01
    # volume は正の整数
    volume = np.random.randint(1, 100, n).astype(np.float64)

    return {
        "open":   open_arr.astype(np.float64),
        "high":   high.astype(np.float64),
        "low":    low.astype(np.float64),
        "close":  close.astype(np.float64),
        "volume": volume,
    }


# ════════════════════════════════════════════════════════════════
# Verification
# ════════════════════════════════════════════════════════════════
def run_verification(
    engine_id: str,
    n_synth: int = 2000,
    lookback_bars: int = 1440,
    seed: int = 42,
    show_fails: int = 15,
    verbose: bool = True,
) -> Optional[Dict]:
    """1 engine 分の検証を実行"""
    if verbose:
        print("=" * 72)
        print(f"  Engine {engine_id.upper()} pre-OLS Bit-Identical Verification")
        print("=" * 72)
        print(f"  合成データ N={n_synth}, lookback={lookback_bars}, seed={seed}")

    suffix = ENGINE_SUFFIXES.get(engine_id)
    if suffix is None:
        print(f"  ❌ unknown engine_id: {engine_id}")
        return None
    letter = engine_id[-1].upper()  # 'a' → 'A'

    # ─── 動的 import ─────────────────────────────────────────
    try:
        learning_mod = importlib.import_module(
            f"engine_1_{letter}_a_vast_universe_of_features"
        )
        production_mod = importlib.import_module(
            f"realtime_feature_engine_1{letter}_{suffix}"
        )
    except Exception as e:
        print(f"  ❌ import 失敗: {e}")
        traceback.print_exc()
        return None

    try:
        ProcessingConfig    = learning_mod.ProcessingConfig
        CalculationEngine   = learning_mod.CalculationEngine
        FeatureModule       = getattr(production_mod, f"FeatureModule1{letter}")
    except AttributeError as e:
        print(f"  ❌ class lookup 失敗: {e}")
        return None

    if verbose:
        print(f"  Learning:   engine_1_{letter}_a_vast_universe_of_features (CalculationEngine)")
        print(f"  Production: realtime_feature_engine_1{letter}_{suffix} (FeatureModule1{letter})")

    # ─── 合成データ ───────────────────────────────────────
    data = generate_synthetic_data(n_synth, seed)

    # ─── 本番側計算 ───────────────────────────────────────
    if verbose:
        print()
        print("  --- Production (rfe) 計算中... ---")
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

    # ─── 学習側計算 (Polars 式を直接適用) ──────────────
    if verbose:
        print("  --- Learning (engine) 計算中... ---")
    try:
        config = ProcessingConfig(engine_id=engine_id)
        ce = CalculationEngine(config)
        expressions = ce._get_all_feature_expressions(timeframe="M3")
    except Exception as e:
        print(f"  ❌ Learning init/exprs 失敗: {e}")
        traceback.print_exc()
        return None
    if verbose:
        print(f"     learning expressions: {len(expressions)} keys")

    # ATR13 を事前計算 (学習側 __temp_atr_13 と本番側 __temp_atr_13 で同じ)
    # 本番側 rfe_1A の calculate_atr_wilder を使用 (全 engine で共通のはず)
    try:
        from realtime_feature_engine_1A_statistics import calculate_atr_wilder
        atr_arr = calculate_atr_wilder(
            data["high"], data["low"], data["close"], 13
        ) + 1e-10
    except Exception as e:
        print(f"  ❌ ATR13 計算失敗: {e}")
        return None

    # DataFrame 構築 (本番側 _build_polars_pieces と同じ key 群)
    df_cols = {
        "close":         data["close"],
        "high":          data["high"],
        "low":           data["low"],
        "open":          data["open"],
        "volume":        data["volume"],
        "__temp_atr_13": atr_arr,
    }

    # ─── 学習側式が必要とする __num_* 列を事前注入 ─────────
    # 本番側は _build_polars_pieces で __num_srm_*, __num_srv_* 等を numpy で先計算 →
    # 注入している。 学習側は map_batches 経由なので注入不要のはずだが、 念のため
    # 本番側の columns と同じものを学習側 select 時に追加注入する。
    try:
        # 本番側 _build_polars_pieces を呼んで columns を取得
        prod_columns, _, _ = FeatureModule._build_polars_pieces(data, lookback_bars=lookback_bars)
        # 既に df_cols にあるもの以外を追加
        for k, v in prod_columns.items():
            if k not in df_cols:
                df_cols[k] = v
    except Exception as e:
        if verbose:
            print(f"     (info) _build_polars_pieces columns 注入失敗 (続行): {e}")

    df_input = pl.DataFrame(df_cols)

    try:
        # Polars 式を select で適用 (engine_1_X の _get_all_feature_expressions の戻り値)
        learning_result_df = df_input.lazy().select(list(expressions.values())).collect()
    except Exception as e:
        print(f"  ❌ Learning Polars select 失敗: {e}")
        traceback.print_exc()
        return None

    # 末尾 1 行 → Dict[str, float]
    learning_features_row = learning_result_df.tail(1).to_dicts()[0]
    learning_features = {}
    for k, v in learning_features_row.items():
        if v is None:
            learning_features[k] = np.nan
        else:
            try:
                fv = float(v)
                learning_features[k] = fv if np.isfinite(fv) else np.nan
            except (TypeError, ValueError):
                learning_features[k] = np.nan
    if verbose:
        print(f"     learning features: {len(learning_features)} keys")

    # ─── Alias 集合比較 ─────────────────────────────────
    prod_set  = set(production_features.keys())
    learn_set = set(learning_features.keys())
    common      = prod_set & learn_set
    prod_only   = prod_set - learn_set
    learn_only  = learn_set - prod_set

    if verbose:
        print()
        print("  === Alias 集合比較 ===")
        print(f"     共通:        {len(common)}")
        print(f"     本番のみ:    {len(prod_only)}")
        if prod_only:
            print(f"       例: {sorted(prod_only)[:5]}")
        print(f"     学習側のみ:  {len(learn_only)}")
        if learn_only:
            print(f"       例: {sorted(learn_only)[:5]}")

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

        # NaN 化
        try:
            v_p_f = float(v_p) if v_p is not None else np.nan
            if not np.isfinite(v_p_f):
                v_p_f = np.nan
        except (TypeError, ValueError):
            v_p_f = np.nan
        v_l_f = v_l if isinstance(v_l, float) else np.nan

        # 両方 NaN → match
        if np.isnan(v_p_f) and np.isnan(v_l_f):
            nan_match += 1
            pass_count += 1
            continue

        # 片方だけ NaN → fail
        if np.isnan(v_p_f) or np.isnan(v_l_f):
            fails.append((k, v_p_f, v_l_f, "NaN mismatch"))
            fail_count += 1
            continue

        diff = abs(v_p_f - v_l_f)
        rel  = diff / (abs(v_l_f) + 1e-12)

        if diff > max_diff:
            max_diff = diff
            max_diff_key = k

        # rtol=1e-7 OR atol=1e-12 → PASS
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
            for k, v_p, v_l, info in fails[:show_fails]:
                print(f"     {k}")
                print(f"       prod={v_p}, learn={v_l}  ({info})")

    # ─── 結論 ────────────────────────────────────────────
    if verbose:
        print()
        if fail_count == 0 and len(prod_only) == 0 and len(learn_only) == 0:
            print(f"  ✅ {engine_id.upper()}: 完全 bit-identical (alias 集合差 0、 全 alias PASS)")
        elif fail_count == 0:
            print(f"  ⚠️  {engine_id.upper()}: 数値は一致するが alias 集合差あり "
                  f"(prod_only={len(prod_only)}, learn_only={len(learn_only)})")
        else:
            print(f"  ❌ {engine_id.upper()}: 数値乖離 {fail_count} FAIL")
        print()

    # 一時ディレクトリ片付け (CalculationEngine.__init__ で作られた tempfile)
    try:
        if hasattr(ce, "temp_dir") and ce.temp_dir.exists():
            shutil.rmtree(ce.temp_dir)
    except Exception:
        pass

    return {
        "engine":     engine_id,
        "common":     len(common),
        "prod_only":  len(prod_only),
        "learn_only": len(learn_only),
        "pass":       pass_count,
        "fail":       fail_count,
        "max_diff":   max_diff,
        "max_diff_key": max_diff_key,
        "fails":      fails,
    }


# ════════════════════════════════════════════════════════════════
# main
# ════════════════════════════════════════════════════════════════
def main():
    parser = argparse.ArgumentParser(description="pre-OLS bit-identical 検証")
    parser.add_argument(
        "--engine", default="all",
        help="e1a/e1b/e1c/e1d/e1e/e1f or 'all'",
    )
    parser.add_argument("--n", type=int, default=2000,
                        help="合成データ長 (default: 2000)")
    parser.add_argument("--lookback", type=int, default=1440,
                        help="lookback_bars (default: 1440)")
    parser.add_argument("--seed", type=int, default=42,
                        help="random seed (default: 42)")
    parser.add_argument("--show-fails", type=int, default=15,
                        help="表示する FAIL 件数 (default: 15)")
    args = parser.parse_args()

    engines = (
        ["e1a", "e1b", "e1c", "e1d", "e1e", "e1f"]
        if args.engine == "all" else [args.engine]
    )

    summary = []
    for eng in engines:
        try:
            r = run_verification(eng, args.n, args.lookback, args.seed, args.show_fails)
            summary.append(r)
        except Exception as e:
            print(f"  ❌ {eng}: Exception: {e}")
            traceback.print_exc()
            summary.append(None)

    # 最終サマリー
    print()
    print("=" * 72)
    print("  全 Engine 検証サマリー")
    print("=" * 72)
    print(f"  {'engine':8s} {'common':>8s} {'prod_only':>10s} {'learn_only':>11s} "
          f"{'PASS':>6s} {'FAIL':>6s} {'max_diff':>12s}")
    print("  " + "─" * 70)
    for r in summary:
        if r is None:
            continue
        status = "✅" if (r["fail"] == 0 and r["prod_only"] == 0 and r["learn_only"] == 0) else (
                 "⚠️" if r["fail"] == 0 else "❌")
        print(f"  {status} {r['engine']:6s} {r['common']:>8d} {r['prod_only']:>10d} "
              f"{r['learn_only']:>11d} {r['pass']:>6d} {r['fail']:>6d} "
              f"{r['max_diff']:>12.2e}")
    print()


if __name__ == "__main__":
    main()
