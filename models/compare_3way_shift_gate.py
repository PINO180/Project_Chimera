"""compare_3way_shift_gate.py — train-serve 特徴量一致検証 (§11.34.16 改修版)

旧 compare_3way_unpurified.py の後継。変更点:
  1. [SHIFT]  prod(T) ↔ training(T - 180s) で突合 (§11.34.16 B 節の 1 本ズレ補正)。
  2. [GATE]   高 TF (>=5分) 列は「両側とも非ゼロ」 の行のみ比較 (HF-NB-GATE 対応)。
  3. [TF 別]  TF ごとに一致率を集計してレポート。
  4. [悉皆]   全列の diff を出し、残差が残る列を悪い順にリストアップ
             (N.8 の最終残差 0.01 級の犯人特定)。

純化は §11.34.14 で撤去済み (A=B) のため、本検証は B(学習) と C(本番) の
一致に集約する。training 側の列 = 本番側の同名列を直接突合する。

判定:
  - 全 TF で「比較対象行」 の corr ≈ 1.0 / diff_med ≈ 0 → train-serve 一致成立。
  - 特定 TF / 特定列だけ崩れる → そこが残差。悉皆リストで列を特定。

使い方:
  python compare_3way_shift_gate.py \\
      --training /workspace/data/diagnostics/training_snapshot_with_pre_ols_XXXX.parquet \\
      --production /workspace/logs/triggered_features_log.csv \\
      --start "2026-06-XX 12:00:00" --end "2026-06-XX 16:30:00" \\
      --out-dir /workspace/data/diagnostics/compare_3way_shift_gate_XXXX
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import polars as pl

import compare_common as cc


def load_training_parquet(
    parquet_path: Path, start: datetime, end: datetime
) -> pd.DataFrame:
    df_pl = pl.read_parquet(parquet_path)
    df_pl = df_pl.with_columns(pl.col("timestamp").cast(pl.Datetime("us", "UTC")))
    df_pl = df_pl.filter((pl.col("timestamp") >= start) & (pl.col("timestamp") <= end))
    df_pd = df_pl.to_pandas()
    df_pd["timestamp"] = pd.to_datetime(df_pd["timestamp"], utc=True)
    return df_pd


def load_production_csv(
    csv_path: Path, start: datetime, end: datetime
) -> pd.DataFrame:
    df = pd.read_csv(csv_path, low_memory=False, on_bad_lines="skip")
    ts_col = "Timestamp" if "Timestamp" in df.columns else "timestamp"
    df[ts_col] = pd.to_datetime(df[ts_col], utc=True, errors="coerce")
    df = df.rename(columns={ts_col: "timestamp"})
    df = df[(df["timestamp"] >= start) & (df["timestamp"] <= end)].copy()
    return df


def neutralized_feature_cols(prod_cols: List[str], train_cols: List[str]) -> List[str]:
    """突合対象 = 本番と学習の両方に存在する _neutralized_M{n} 列。

    純化撤去後は B (学習恒等) と C (本番恒等) を直接比べる。
    A/B (pre/post) ペアの概念は不要 (A=B) なので、本番に出ている neutralized
    特徴量列をそのまま対象にする。
    """
    train_set = set(train_cols)
    out = []
    for c in prod_cols:
        if "_neutralized_M" not in c:
            continue
        if c in train_set:
            out.append(c)
    return sorted(out)


def compute(
    df_train: pd.DataFrame,
    df_prod: pd.DataFrame,
    feat_cols: List[str],
) -> pd.DataFrame:
    """[SHIFT] + [GATE] を適用して列ごとの一致メトリクスを計算。"""
    # [SHIFT] training を T 基準へ
    df_train_s = cc.shift_training_to_signal(df_train)
    merged = cc.merge_prod_training(df_train_s, df_prod, suffixes=("", "_prod"))
    print(f"    → [SHIFT -{cc.SHIFT_SEC}s] merged: {len(merged)} 行")
    if len(merged) == 0:
        print("  ❌ 共通 timestamp 0 (シフト後)。期間 / ログを確認")
        return pd.DataFrame()

    rows = []
    for col in feat_cols:
        prod_name = f"{col}_prod" if f"{col}_prod" in merged.columns else col
        if col not in merged.columns or prod_name not in merged.columns:
            continue
        B = pd.to_numeric(merged[col], errors="coerce").to_numpy(np.float64)  # 学習
        C = pd.to_numeric(merged[prod_name], errors="coerce").to_numpy(np.float64)  # 本番
        # [GATE] 高TFは両側非ゼロ行のみ。lowTFは有限行すべて。
        mask = cc.gate_mask_for_col(col, B, C)
        if mask.sum() < 2:
            continue
        m = cc.pair_metrics(B[mask], C[mask])
        tfm = cc.tf_minutes_of(col)
        rows.append(
            {
                "feature": col,
                "tf_min": tfm if tfm is not None else -1.0,
                "is_high_tf": cc.is_high_tf_col(col),
                **m,
            }
        )
    return pd.DataFrame(rows)


def report_by_tf(metrics: pd.DataFrame) -> pd.DataFrame:
    """TF 別の集計レポートを作る。"""
    if metrics.empty:
        return pd.DataFrame()
    g = metrics.groupby("tf_min")
    rep = g.agg(
        n_features=("feature", "size"),
        corr_med=("corr", "median"),
        diff_med=("diff_med", "median"),
        bit_rate_med=("bit_rate", "median"),
        n_rows_med=("n", "median"),
    ).reset_index().sort_values("tf_min")
    return rep


def main():
    p = argparse.ArgumentParser(
        description="train-serve feature consistency (SHIFT+GATE+TF別+悉皆) §11.34.16"
    )
    p.add_argument("--training", type=Path, required=True)
    p.add_argument(
        "--production", type=Path, default=Path("/workspace/logs/triggered_features_log.csv")
    )
    p.add_argument("--start", required=True, help="YYYY-MM-DD HH:MM:SS UTC")
    p.add_argument("--end", required=True, help="YYYY-MM-DD HH:MM:SS UTC")
    p.add_argument(
        "--out-dir", type=Path,
        default=Path("/workspace/data/diagnostics/compare_3way_shift_gate"),
    )
    p.add_argument("--top-n", type=int, default=40, help="悉皆: 悪い順に出す列数")
    p.add_argument(
        "--shift-sec", type=int, default=cc.SHIFT_SEC,
        help="prod(T) ↔ training(T - shift) の shift 秒 (既定 180 = M3)",
    )
    args = p.parse_args()
    cc.SHIFT_SEC = args.shift_sec  # 上書き可能に

    args.out_dir.mkdir(parents=True, exist_ok=True)
    start_dt = cc.parse_dt(args.start)
    end_dt = cc.parse_dt(args.end)

    print("=" * 72)
    print("  train-serve 特徴量一致検証 (SHIFT + GATE + TF別 + 悉皆) §11.34.16")
    print("=" * 72)
    print(f"  期間: {start_dt} 〜 {end_dt}   shift: -{cc.SHIFT_SEC}s")
    print()

    print("--- 1. 読み込み ---")
    df_train = load_training_parquet(args.training, start_dt, end_dt)
    df_prod = load_production_csv(args.production, start_dt, end_dt)
    print(f"  training: {len(df_train)} 行 / {df_train.shape[1]} 列")
    print(f"  production: {len(df_prod)} 行 / {df_prod.shape[1]} 列")

    print()
    print("--- 2. 突合対象列 (neutralized, 両側存在) ---")
    feat_cols = neutralized_feature_cols(list(df_prod.columns), list(df_train.columns))
    print(f"  対象列: {len(feat_cols)}")
    if not feat_cols:
        print("  ❌ 突合対象 0")
        sys.exit(1)

    print()
    print("--- 3. [SHIFT]+[GATE] メトリクス計算 ---")
    metrics = compute(df_train, df_prod, feat_cols)
    if metrics.empty:
        sys.exit(1)
    metrics_path = args.out_dir / "metrics_per_feature.parquet"
    metrics.to_parquet(metrics_path, index=False)
    print(f"  列メトリクス: {len(metrics)} → {metrics_path}")

    print()
    print("--- 4. TF 別レポート ---")
    rep = report_by_tf(metrics)
    print(f"  {'TF(分)':>7} {'列数':>5} {'corr中央':>9} {'diff中央':>11} {'bit率中央%':>10} {'行数中央':>8}")
    for _, r in rep.iterrows():
        tf_disp = "low" if r["tf_min"] < 0 else f"M{r['tf_min']:g}"
        print(
            f"  {tf_disp:>7} {int(r['n_features']):>5} {r['corr_med']:>9.4f} "
            f"{r['diff_med']:>11.2e} {r['bit_rate_med']:>10.1f} {int(r['n_rows_med']):>8}"
        )
    rep.to_parquet(args.out_dir / "report_by_tf.parquet", index=False)

    print()
    print("--- 5. [悉皆] 残差が残る列 (corr 昇順 / diff 降順) ---")
    # corr が低い or diff が大きい列を犯人候補として上位表示
    bad = metrics.copy()
    bad["corr_fill"] = bad["corr"].fillna(-1.0)
    bad = bad.sort_values(["corr_fill", "diff_med"], ascending=[True, False])
    head = bad.head(args.top_n)
    print(f"  {'feature':50s} {'TF':>5} {'n':>4} {'corr':>8} {'diff_med':>11} {'bit%':>6}")
    for _, r in head.iterrows():
        tf_disp = "low" if r["tf_min"] < 0 else f"M{r['tf_min']:g}"
        corr_disp = "nan" if pd.isna(r["corr"]) else f"{r['corr']:.4f}"
        print(
            f"  {r['feature'][:50]:50s} {tf_disp:>5} {int(r['n']):>4} "
            f"{corr_disp:>8} {r['diff_med']:>11.2e} {r['bit_rate']:>6.1f}"
        )
    bad.drop(columns=["corr_fill"]).to_parquet(
        args.out_dir / "columns_worst_first.parquet", index=False
    )

    # サマリ
    print()
    print("--- サマリ ---")
    overall_corr = metrics["corr"].median()
    overall_diff = metrics["diff_med"].median()
    n_bad = int((metrics["corr"].fillna(-1) < 0.95).sum())
    print(f"  全列 corr 中央値: {overall_corr:.4f}")
    print(f"  全列 diff 中央値: {overall_diff:.3e}")
    print(f"  corr < 0.95 の列: {n_bad} / {len(metrics)}")
    verdict = "PASS (一致成立)" if overall_corr >= 0.99 and n_bad == 0 else (
        "残差あり (悉皆リスト先頭が犯人)" if overall_corr >= 0.9 else "FAIL")
    print(f"  判定: {verdict}")
    with open(args.out_dir / "verdict.txt", "w") as f:
        f.write(f"corr_med={overall_corr:.4f} diff_med={overall_diff:.3e} "
                f"n_bad={n_bad}/{len(metrics)} verdict={verdict}\n")
    print(f"\n  出力一式: {args.out_dir}")


if __name__ == "__main__":
    main()
