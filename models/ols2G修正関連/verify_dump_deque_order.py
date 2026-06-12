#!/usr/bin/env python3
"""
verify_dump_deque_order.py — 取引時間ベースで dump deque を順序比較

[v1 (compare_dump_deque_vs_s1s2.py) の問題]
  reconstruct_deque_timestamps が「カレンダー時間で bar_sec 逆算」 する仮定だったが、
  実際の production deque は「取引時間」 で N 本遡る (= 土日 gap は飛ばす)。
  そのため merge_asof で違う時刻の学習側値を引いて巨大な見かけ上の乖離が出ていた。

[本スクリプトの修正]
  学習側 S2 (engine=e1a を代表) の timestamp 列が、 production deque と同じ
  「取引時間で並んだ TF close 時刻の系列」 を持っている。 これを直接使って:
    - signal_ts までの末尾 N 個の timestamp を取得 (= production deque と同じ刻み)
    - 各 timestamp について学習側 値を取得 (X: S1 backward join、 Y: S2 直接)
    - dump deque と順序通り 1:1 比較

[出力]
  X 比較: dump x_deque (= TF 別 market_proxy 系列) vs 学習側 backward join 値
  Y 比較: dump y_deque (= 各 feature 純化前系列)   vs 学習側 S2 純化前値
"""

from __future__ import annotations

import argparse
import pickle
import sys
import warnings
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import polars as pl

warnings.filterwarnings("ignore", category=FutureWarning)

sys.path.insert(0, "/workspace")
import blueprint as config

ENGINE_TO_UNIVERSE = {"e1a": "A", "e1b": "B", "e1c": "C",
                     "e1d": "D", "e1e": "E", "e1f": "F"}


# ════════════════════════════════════════════════════════════════
# 学習側 proxy / y ロード
# ════════════════════════════════════════════════════════════════
def load_learning_proxy() -> pd.DataFrame:
    s1_m5_dir = Path(config.S1_PROCESSED) / "timeframe=M5"
    df = (
        pl.scan_parquet(str(s1_m5_dir / "*.parquet"))
        .select(["timestamp", "close"])
        .collect()
        .with_columns(pl.col("timestamp").cast(pl.Datetime("us", "UTC")))
        .sort("timestamp")
        .with_columns(
            (pl.col("close") / pl.col("close").shift(1) - 1).alias("proxy_train")
        )
        .to_pandas()
    )
    df["timestamp"] = (
        pd.to_datetime(df["timestamp"], utc=True)
        .astype("datetime64[ns, UTC]")
    )
    return df


def load_learning_y(engine: str, tf: str) -> Optional[pd.DataFrame]:
    universe = ENGINE_TO_UNIVERSE[engine]
    path = (
        Path(config.S2_FEATURES_VALIDATED)
        / f"feature_value_a_vast_universe{universe}"
        / f"features_{engine}_{tf}.parquet"
    )
    if not path.exists():
        return None
    df = (
        pl.read_parquet(path)
        .with_columns(pl.col("timestamp").cast(pl.Datetime("us", "UTC")))
        .sort("timestamp")
        .to_pandas()
    )
    df["timestamp"] = (
        pd.to_datetime(df["timestamp"], utc=True)
        .astype("datetime64[ns, UTC]")
    )
    return df


# ════════════════════════════════════════════════════════════════
# X 比較 (取引時間ベース): dump x_deque vs 学習側 backward join された M5 proxy
# ════════════════════════════════════════════════════════════════
def compare_x_strict(
    dump: Dict, s1_proxy: pd.DataFrame, s2_caches: Dict, dump_name: str
) -> pd.DataFrame:
    signal_ts = pd.Timestamp(dump["signal_timestamp"]).tz_convert("UTC")
    s1_sorted = s1_proxy[["timestamp", "proxy_train"]].sort_values("timestamp")

    rows = []
    for tf in sorted(dump["proxy_feature_buffers"].keys()):
        x_deque = dump["proxy_feature_buffers"][tf].get("market_proxy", [])
        if not x_deque:
            continue
        n = len(x_deque)

        # 学習側 S2 の timestamp を取引時間ベースで取得 (e1a を代表)
        s2 = s2_caches.get(("e1a", tf))
        if s2 is None:
            rows.append({"dump": dump_name, "tf": tf, "n_deque": n,
                        "n_used": 0, "note": "S2 e1a 不在"})
            continue
        s2_until_sig = (
            s2[s2["timestamp"] <= signal_ts]
            .sort_values("timestamp")
        )
        n_avail = len(s2_until_sig)
        n_use = min(n, n_avail)
        if n_use == 0:
            rows.append({"dump": dump_name, "tf": tf, "n_deque": n,
                        "n_used": 0, "note": "S2 期間内データなし"})
            continue

        # dump deque の末尾 n_use 個と、 S2 timestamp の末尾 n_use 個を対応
        dump_arr = np.asarray(x_deque[-n_use:], dtype=np.float64)
        ts_used = s2_until_sig["timestamp"].tail(n_use).to_numpy()

        # S1 M5 proxy を backward join (production の M5 proxy 取得 spec を再現:
        # allow_exact_matches=False で「同 timestamp の M5 close は反映前」 を再現)
        ts_df = pd.DataFrame({"timestamp": ts_used}).sort_values("timestamp")
        merged = pd.merge_asof(
            ts_df, s1_sorted, on="timestamp", direction="backward",
            allow_exact_matches=False,
        )
        learning_arr = merged["proxy_train"].to_numpy()

        diff = np.abs(dump_arr - learning_arr)
        finite = np.isfinite(diff)
        diff_f = diff[finite]
        if len(diff_f) == 0:
            continue

        rows.append({
            "dump":             dump_name,
            "tf":               tf,
            "signal_ts":        signal_ts,
            "n_deque":          n,
            "n_used":           n_use,
            "n_finite":         int(finite.sum()),
            "max_abs_diff":     float(diff_f.max()),
            "mean_abs_diff":    float(diff_f.mean()),
            "median_abs_diff":  float(np.median(diff_f)),
            "bit_identical_lt_1e-9": int((diff_f < 1e-9).sum()),
            "small_1e-9_to_1e-6":    int(((diff_f >= 1e-9) & (diff_f < 1e-6)).sum()),
            "moderate_ge_1e-6":      int((diff_f >= 1e-6).sum()),
            "large_ge_1e-4":         int((diff_f >= 1e-4).sum()),
        })
    return pd.DataFrame(rows)


# ════════════════════════════════════════════════════════════════
# Y 比較 (取引時間ベース): dump y_deque vs S2 純化前
# ════════════════════════════════════════════════════════════════
def compare_y_strict(
    dump: Dict, sample_features: Dict[str, List[str]],
    s2_caches: Dict, dump_name: str
) -> pd.DataFrame:
    signal_ts = pd.Timestamp(dump["signal_timestamp"]).tz_convert("UTC")

    rows = []
    for tf, feats_list in sample_features.items():
        buffers = dump["proxy_feature_buffers"].get(tf, {})
        if not buffers:
            continue
        for feat_name in feats_list:
            y_deque = buffers.get(feat_name)
            if y_deque is None:
                continue
            n = len(y_deque)
            engine = feat_name.split("_")[0]
            if engine not in ENGINE_TO_UNIVERSE:
                continue
            s2 = s2_caches.get((engine, tf))
            if s2 is None or feat_name not in s2.columns:
                continue

            s2_until_sig = (
                s2[s2["timestamp"] <= signal_ts]
                .sort_values("timestamp")
            )
            n_avail = len(s2_until_sig)
            n_use = min(n, n_avail)
            if n_use == 0:
                continue

            y_prod = np.asarray(y_deque[-n_use:], dtype=np.float64)
            y_train = (
                s2_until_sig[feat_name].tail(n_use).to_numpy(dtype=np.float64)
            )

            diff = np.abs(y_prod - y_train)
            finite = np.isfinite(diff)
            diff_f = diff[finite]
            if len(diff_f) == 0:
                continue

            rel = diff_f / (np.abs(y_train[finite]) + 1e-10)

            rows.append({
                "dump":            dump_name,
                "tf":              tf,
                "engine":          engine,
                "feature":         feat_name,
                "signal_ts":       signal_ts,
                "n_deque":         n,
                "n_used":          n_use,
                "n_finite":        int(finite.sum()),
                "max_abs_diff":    float(diff_f.max()),
                "mean_abs_diff":   float(diff_f.mean()),
                "median_abs_diff": float(np.median(diff_f)),
                "max_rel_diff":    float(rel.max()),
                "bit_identical":   int((diff_f < 1e-9).sum()),
                "small":           int(((diff_f >= 1e-9) & (diff_f < 1e-6)).sum()),
                "moderate":        int((diff_f >= 1e-6).sum()),
            })
    return pd.DataFrame(rows)


# ════════════════════════════════════════════════════════════════
# main
# ════════════════════════════════════════════════════════════════
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dump-dir", type=Path,
                        default="/workspace/logs/ols_state_dumps")
    parser.add_argument("--out-dir", type=Path,
                        default="/workspace/data/diagnostics/dump_strict")
    parser.add_argument("--y-sample-per-engine", type=int, default=5)
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 72)
    print("  dump vs S1/S2 取引時間ベース 順序比較 (strict)")
    print("=" * 72)

    dump_files = sorted(args.dump_dir.glob("ols_*.pkl"))
    print(f"  対象 dump 数: {len(dump_files)}")
    if not dump_files:
        return

    # 学習側データロード
    print()
    print("--- 学習側データロード ---")
    s1_proxy = load_learning_proxy()
    print(f"  S1 M5 proxy: {len(s1_proxy):,} rows, "
          f"{s1_proxy['timestamp'].min()} 〜 {s1_proxy['timestamp'].max()}")

    # S2 キャッシュ + sample 選定 (e1a〜e1f に限定)
    print()
    print("--- S2 ロード + y feature サンプル ---")
    np.random.seed(42)
    sample_features: Dict[str, List[str]] = {}
    s2_caches: Dict = {}

    first_dump = pickle.load(open(dump_files[0], "rb"))
    for tf, bufs in first_dump["proxy_feature_buffers"].items():
        feat_names = [k for k in bufs if k != "market_proxy"]
        if not feat_names:
            continue
        engines = sorted({
            f.split("_")[0] for f in feat_names
            if f.split("_")[0] in ENGINE_TO_UNIVERSE
        })
        # X 比較用に e1a の S2 は必須
        for engine in (engines + (["e1a"] if "e1a" not in engines else [])):
            if (engine, tf) not in s2_caches:
                s2 = load_learning_y(engine, tf)
                if s2 is not None:
                    s2_caches[(engine, tf)] = s2

        tf_sample = []
        for engine in engines:
            s2 = s2_caches.get((engine, tf))
            if s2 is None:
                continue
            feats_eng = [f for f in feat_names
                        if f.startswith(f"{engine}_") and f in s2.columns]
            if not feats_eng:
                continue
            n_take = min(args.y_sample_per_engine, len(feats_eng))
            tf_sample.extend(list(np.random.choice(feats_eng, n_take, replace=False)))
        sample_features[tf] = tf_sample
    print(f"  S2 caches: {len(s2_caches)} (engine,tf) pairs")
    print(f"  y サンプル: {sum(len(v) for v in sample_features.values())} features")

    # 全 dump 処理
    print()
    print("--- 全 dump 処理 ---")
    all_x, all_y = [], []
    for i, dp in enumerate(dump_files):
        print(f"  [{i+1}/{len(dump_files)}] {dp.name}", end="", flush=True)
        try:
            with open(dp, "rb") as f:
                dump = pickle.load(f)
            xr = compare_x_strict(dump, s1_proxy, s2_caches, dp.name)
            yr = compare_y_strict(dump, sample_features, s2_caches, dp.name)
            all_x.append(xr)
            all_y.append(yr)
            print(f"  ✓ x={len(xr)}, y={len(yr)}")
        except Exception as e:
            print(f"  ❌ {e}")
            import traceback; traceback.print_exc()

    all_x = pd.concat(all_x, ignore_index=True) if all_x else pd.DataFrame()
    all_y = pd.concat(all_y, ignore_index=True) if all_y else pd.DataFrame()
    if len(all_x) > 0:
        all_x.to_parquet(args.out_dir / "all_x_strict.parquet")
    if len(all_y) > 0:
        all_y.to_parquet(args.out_dir / "all_y_strict.parquet")

    # ─── X 集計 ─────────────────────────────────────────────
    print()
    print("=" * 72)
    print("  X 比較 集計 (= dump x_deque vs 学習側 backward-joined M5 proxy)")
    print("=" * 72)
    if len(all_x) > 0:
        x_agg = all_x.groupby("tf").agg(
            n_dumps=("dump", "nunique"),
            n_total=("n_used", "sum"),
            max_abs_diff_max=("max_abs_diff", "max"),
            max_abs_diff_median=("max_abs_diff", "median"),
            bit_identical_total=("bit_identical_lt_1e-9", "sum"),
            small_total=("small_1e-9_to_1e-6", "sum"),
            moderate_total=("moderate_ge_1e-6", "sum"),
            large_total=("large_ge_1e-4", "sum"),
        )
        print(x_agg.to_string())
        x_agg.to_csv(args.out_dir / "x_agg_strict.csv")

    # ─── Y 集計 (TF × engine) ──────────────────────────────
    print()
    print("=" * 72)
    print("  Y 比較 集計 (TF × engine)")
    print("=" * 72)
    if len(all_y) > 0:
        y_agg_te = all_y.groupby(["tf", "engine"]).agg(
            n_features=("feature", "nunique"),
            n_dumps=("dump", "nunique"),
            n_total=("n_used", "sum"),
            max_abs_diff_max=("max_abs_diff", "max"),
            max_abs_diff_median=("max_abs_diff", "median"),
            max_rel_diff_max=("max_rel_diff", "max"),
            bit_identical_total=("bit_identical", "sum"),
            moderate_total=("moderate", "sum"),
        )
        print(y_agg_te.to_string())
        y_agg_te.to_csv(args.out_dir / "y_agg_strict_by_tf_engine.csv")

    # ─── Y TOP 20 ─────────────────────────────────────────
    print()
    print("=" * 72)
    print("  Y 比較 TOP 20 乖離 features")
    print("=" * 72)
    if len(all_y) > 0:
        y_agg_feat = all_y.groupby(["tf", "feature"]).agg(
            n_dumps=("dump", "nunique"),
            n_total=("n_used", "sum"),
            max_abs_diff_max=("max_abs_diff", "max"),
            max_abs_diff_median=("max_abs_diff", "median"),
            max_rel_diff_max=("max_rel_diff", "max"),
            bit_identical_total=("bit_identical", "sum"),
            moderate_total=("moderate", "sum"),
        )
        print(y_agg_feat.nlargest(20, "max_abs_diff_max").to_string())
        y_agg_feat.to_csv(args.out_dir / "y_agg_strict_by_feature.csv")

    print()
    print("=" * 72)
    print(f"✅ 完了 — 出力: {args.out_dir}")
    print("=" * 72)
    print()
    print("【解釈ガイド】")
    print("  X 比較 bit_identical 〜 100%:")
    print("    → market_proxy 値系列は学習側と完全一致 → 真因は OLS 累積 or β/α 計算")
    print("  X 比較 乖離あり:")
    print("    → M5 close 値が学習側と本番側で違う = broker feed 差 / resample 境界差")
    print("  Y 比較 bit_identical 〜 100%:")
    print("    → engine pre-OLS 計算は完璧 (X-2 合成データ検証と整合)")
    print("  Y 比較 乖離あり:")
    print("    → 実 tick で engine 計算が systematic に違う")
    print("       = 入力 OHLCV (= s1_1_B 出力) が学習側と本番側で違う可能性")


if __name__ == "__main__":
    main()
