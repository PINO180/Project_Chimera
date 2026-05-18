#!/usr/bin/env python3
"""[§B.12.10.X cell-level trace] dump された production deque を engine_1_A 経路で計算

目的:
  e1a Cluster A 70 cells の真因仮説 4 (ATR の context 長依存 + stable_rolling 経路の差異)
  を実機で決着させる。

入力:
  /tmp/forge_deque_dump/deque_M0_5_2026-04-01T06-48-00.pkl
    (production の _calculate_base_features 入口で dump された OHLCV deque snapshot)

出力 (stdout):
  - シンプル numpy var(close[-10:], ddof=1) の結果
  - stable_rolling_var(close, 10, ddof=1) の last 値 (production deque だけで)
  - calculate_atr_wilder(high, low, close, 13) の last 値 (production deque だけで)
  - 各組み合わせで variance_10 final 値を計算
  - S2_FEATURES (= 新 engine_A の出力) の同じ ts の値
  - production が出した値
  - 比較判定

実行:
  python3 trace_one_cell.py --dump /tmp/forge_deque_dump/deque_M0_5_2026-04-01T06-48-00.pkl
"""

from __future__ import annotations
import argparse
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dump", type=Path, required=True,
                   help="production の dump フックが生成した data pickle ファイル "
                        "(deque_*.pkl)")
    p.add_argument("--pieces", type=Path, default=None,
                   help="production の 2 番目の dump フックが生成した pieces pickle "
                        "(pieces_*.pkl)。production が _build_polars_pieces 内で実際に "
                        "生成した __temp_atr_13 配列を含む。指定すれば bit 比較できる")
    p.add_argument("--feature", default="e1a_statistical_variance_10",
                   help="比較する feature 名 (default: e1a_statistical_variance_10)")
    p.add_argument("--s2-root", type=Path,
                   default=Path("/workspace/data/XAUUSD/stratum_2_features"),
                   help="S2_FEATURES (新 engine_A 出力) のルート")
    p.add_argument("--workspace", type=Path, default=Path("/workspace"),
                   help="workspace ルート (core_indicators / core/stable_rolling import 用)")
    args = p.parse_args()

    # --- 1. dump 読み込み ---
    if not args.dump.exists():
        print(f"ERROR: {args.dump} not found", file=sys.stderr)
        return 1
    with open(args.dump, "rb") as f:
        snap = pickle.load(f)

    print("=" * 78)
    print("Dumped deque snapshot")
    print("=" * 78)
    print(f"  tf: {snap['tf']}")
    print(f"  ts: {snap['ts']}")
    print(f"  skip_qa_update: {snap.get('skip_qa_update', 'N/A')}")
    print(f"  data keys: {list(snap['data'].keys())}")
    for k, l in snap["data_lengths"].items():
        print(f"    {k}: len={l}")
    print()

    close = snap["data"]["close"].astype(np.float64)
    high = snap["data"]["high"].astype(np.float64)
    low = snap["data"]["low"].astype(np.float64)

    # --- 2. variance_10 numerator: 3 経路で計算 ---
    print("=" * 78)
    print("【A】variance_10 の分子 = stable_rolling_var(close, 10, ddof=1) の last")
    print("=" * 78)

    # 経路 1: シンプル numpy
    last10 = close[-10:]
    var_np = float(np.var(last10, ddof=1))
    print(f"  経路 1 (numpy var(close[-10:], ddof=1)):     {var_np!r}")
    print(f"          hex: {var_np.hex()}")

    # 経路 2: stable_rolling_var (deque 全体) — production と engine_1_A が共通で使う
    try:
        sys.path.insert(0, str(args.workspace))
        sys.path.insert(0, str(args.workspace / "core"))
        from stable_rolling import stable_rolling_var
        var_sr_full = stable_rolling_var(close, 10, 1)
        var_sr_last = float(var_sr_full[-1])
        print(f"  経路 2 (stable_rolling_var, deque 全体):    {var_sr_last!r}")
        print(f"          hex: {var_sr_last.hex()}")
    except ImportError as e:
        print(f"  経路 2: stable_rolling import 失敗 ({e})")
        var_sr_last = None

    # 経路 3: stable_rolling_var (last 10 だけ) — context 非依存なら経路 2 と同じはず
    if var_sr_last is not None:
        var_sr10 = stable_rolling_var(last10, 10, 1)
        var_sr10_last = float(var_sr10[-1])
        print(f"  経路 3 (stable_rolling_var, last 10 だけ):  {var_sr10_last!r}")
        print(f"          hex: {var_sr10_last.hex()}")
        if abs(var_sr10_last - var_sr_last) < 1e-15:
            print(f"  → ✅ stable_rolling_var は context 非依存 (経路 2 = 経路 3)")
        else:
            print(f"  → ❌ stable_rolling_var が context 依存! 差={var_sr10_last - var_sr_last:.3e}")

    # 経路 4: polars rolling_var (deque 全体)
    df_full = pl.DataFrame({"close": close})
    var_pl_full = df_full.with_columns(
        pl.col("close").rolling_var(window_size=10).alias("v")
    )["v"].to_list()
    var_pl_last = var_pl_full[-1] if var_pl_full[-1] is not None else float("nan")
    print(f"  経路 4 (polars rolling_var, deque 全体):    {var_pl_last!r}")
    if isinstance(var_pl_last, float) and not np.isnan(var_pl_last):
        print(f"          hex: {var_pl_last.hex()}")

    # --- 3. ATR(13) を deque だけで計算 (= production の最終 atr 値の再現) ---
    print()
    print("=" * 78)
    print("【B】__temp_atr_13 = calculate_atr_wilder(high, low, close, 13) + 1e-10")
    print("=" * 78)
    try:
        from core_indicators import calculate_atr_wilder
        atr_arr = calculate_atr_wilder(high, low, close, 13)
        atr_last_raw = float(atr_arr[-1])
        atr_last = atr_last_raw + 1e-10
        print(f"  経路 1 (deque 全体, raw):                   {atr_last_raw!r}")
        print(f"          hex: {atr_last_raw.hex()}")
        print(f"  経路 1 (deque 全体, + 1e-10):               {atr_last!r}")
        print(f"          hex: {atr_last.hex()}")
        print(f"  経路 1 deque 長: {len(close)} bars  (ATR seed = TR[0] from 最古 bar)")
    except ImportError as e:
        print(f"  ERROR: core_indicators import 失敗 ({e})")
        return 2

    # --- 3.5 pieces (production の 2 番目 dump フックの出力) と bit 比較 ---
    prod_atr_last = None
    if args.pieces is not None and args.pieces.exists():
        print()
        print("=" * 78)
        print("【B-prime】production が _build_polars_pieces 内で実際に生成した __temp_atr_13")
        print("=" * 78)
        with open(args.pieces, "rb") as f:
            pieces = pickle.load(f)

        # 各モジュールが出した __temp_atr_13 の last 値
        for key in ["cols_a_temp_atr_13", "cols_d_temp_atr_13", "cols_e_temp_atr_13",
                    "all_columns_temp_atr_13"]:
            arr = pieces.get(key)
            if arr is None:
                print(f"  {key:30s}: None (モジュール出力なし)")
                continue
            last_v = float(arr[-1])
            print(f"  {key:30s}: last={last_v!r}")
            print(f"  {'':30s}  hex: {last_v.hex()}  len={len(arr)}")

        # 私の trace で計算した atr (経路 1) と比較
        prod_atr_a = pieces.get("cols_a_temp_atr_13")
        prod_atr_e = pieces.get("cols_e_temp_atr_13")
        prod_atr_all = pieces.get("all_columns_temp_atr_13")
        print()
        print("  ↓ 私の trace との bit 比較:")
        print(f"    私の trace (経路 1, deque + 1e-10):  {atr_last!r}")
        print(f"                                          hex: {atr_last.hex()}")
        if prod_atr_a is not None:
            diff_a = float(prod_atr_a[-1]) - atr_last
            print(f"    cols_a last - trace:               {diff_a:+.3e}")
        if prod_atr_e is not None:
            # cols_e は raw (epsilon 無し)
            diff_e_raw = float(prod_atr_e[-1]) - atr_last_raw
            diff_e_eps = float(prod_atr_e[-1]) - atr_last
            print(f"    cols_e last - trace_raw:           {diff_e_raw:+.3e}")
            print(f"    cols_e last - trace_eps:           {diff_e_eps:+.3e}")
        if prod_atr_all is not None:
            diff_all_eps = float(prod_atr_all[-1]) - atr_last
            print(f"    all_columns last - trace_eps:      {diff_all_eps:+.3e}")
            prod_atr_last = float(prod_atr_all[-1])

        # 各モジュールの close も比較 (input data が同じか確認)
        print()
        print("  ↓ input close の bit 比較 (snap['data']['close'] vs cols_a/e):")
        for key, arr in [("cols_a_close", pieces.get("cols_a_close")),
                         ("cols_e_close", pieces.get("cols_e_close"))]:
            if arr is None:
                continue
            diff_c = np.abs(close - arr).max()
            print(f"    max abs diff (snap close vs {key}): {diff_c:.3e}")

    # --- 4. variance_10 final = numerator / atr_last^2 ---
    print()
    print("=" * 78)
    print("【C】variance_10 final = stable_rolling_var / __temp_atr_13^2")
    print("=" * 78)
    if var_sr_last is not None:
        var10_final = var_sr_last / (atr_last ** 2)
        print(f"  C-1 (経路 2 numerator / 私の trace atr^2):  {var10_final!r}")
        print(f"          hex: {var10_final.hex()}")
        # production が実際に生成した atr で計算 (= prod 値に近づくか確認)
        if prod_atr_last is not None:
            var10_final_prod = var_sr_last / (prod_atr_last ** 2)
            print(f"  C-2 (経路 2 numerator / production atr^2):  {var10_final_prod!r}")
            print(f"          hex: {var10_final_prod.hex()}")

    # --- 5. S2_FEATURES (= 新 engine_A の出力) と比較 ---
    print()
    print("=" * 78)
    print("【D】S2_FEATURES (新 engine_A の出力、context full 3.4M bars) と比較")
    print("=" * 78)
    s2_file = args.s2_root / "feature_value_a_vast_universeA" / f"features_e1a_{snap['tf']}.parquet"
    if not s2_file.exists():
        print(f"  S2 file not found: {s2_file}")
    else:
        df_s2 = pd.read_parquet(s2_file)
        if df_s2["timestamp"].dt.tz is None:
            df_s2["timestamp"] = df_s2["timestamp"].dt.tz_localize("UTC")
        else:
            df_s2["timestamp"] = df_s2["timestamp"].dt.tz_convert("UTC")
        ts_target = pd.Timestamp(snap["ts"]).tz_localize("UTC")
        row = df_s2[df_s2["timestamp"] == ts_target]
        if len(row) == 0:
            print(f"  ts {snap['ts']} not in S2 file")
        else:
            s2_val = float(row[args.feature].iloc[0])
            print(f"  S2 ({args.feature} @ {snap['ts']}): {s2_val!r}")
            print(f"          hex: {s2_val.hex()}")

            # 比較表
            print()
            print("=" * 78)
            print("【E】最終比較")
            print("=" * 78)
            print(f"  S2 ref (engine_1_A full context):           {s2_val:.10f}")
            if var_sr_last is not None:
                print(f"  trace (deque-only stable_var + deque-atr):  {var10_final:.10f}")
                print(f"    → diff: {var10_final - s2_val:+.4e}")
            print()
            print("  解釈:")
            print("  trace ≈ S2 ref (diff ~ 1e-10 オーダー)")
            print("    → engine_1_A と production rfe_1A は数値的に完全等価、")
            print("       Cluster A の真因は別 (deque 構築や別経路)")
            print()
            print("  trace ≠ S2 ref (diff ~ 1e-4 オーダー、production prod とほぼ一致)")
            print("    → ATR context 長依存が真因。production deque (~2880 bars) で計算した ATR と")
            print("       engine_1_A full series (~3.4M bars) で計算した ATR が違うため、")
            print("       variance_10 = var/atr^2 が異なる値になる。")
            print("       修正方針: rfe_1A の ATR を engine_1_A と同じ context で計算する仕組み導入。")
            print()
            print("  trace ≠ S2 ref かつ production prod とも違う")
            print("    → さらに別の経路差 (stable_rolling 経路の引数差、numba JIT cache 等) ")

    print("=" * 78)
    return 0


if __name__ == "__main__":
    sys.exit(main())
