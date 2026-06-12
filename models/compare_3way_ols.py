#!/usr/bin/env python3
"""
compare_3way_ols.py — 学習純化前 / 学習純化後 / 本番純化後 の 3 way 比較

[目的]
  Phase 11 pre-OLS bit-identical 検証で、 engine_1_X と realtime_feature_engine_1X
  の特徴量計算は両側で同値であることが確定した。 残る真因は OLS 純化レイヤー。

  このスクリプトでは、 1 つのデータセット内で:
    A: 学習側 純化前 (S2 由来, training_snapshot の TF サフィックス列)
    B: 学習側 純化後 (S6 由来, training_snapshot の _neutralized_TF サフィックス列)
    C: 本番側 純化後 (production triggered_features_log.csv の同 alias 列)
  を同一 timestamp で突き合わせ、 各特徴量について以下を観察する:

  [A vs B] 学習側 OLS が pre から post にどれだけ値を動かしているか (= β*X + α 等価)
  [B vs C] 学習側 OLS と 本番側 OLS の最終結果差 (← 真因の発現箇所)
  [A vs C] 本番側 OLS が pre から post にどれだけ値を動かしているか

  AB ≈ AC かつ BC ≈ 0 なら両者一致 → 別の真因。
  BC が大きい → OLS 計算経路に systematic な差。
  AB と AC で量・符号が違う → 本番側 β/α が学習側と違う方向にズレている。

[入力]
  --training:   /workspace/data/diagnostics/training_snapshot_with_pre_ols_20260525.parquet
  --production: /workspace/logs/triggered_features_log.csv
  --start/end:  比較対象期間 (UTC)

[出力]
  --out-dir 配下:
    report.md                            # サマリー
    metrics_per_feature.parquet          # feature 別の全メトリクス
    top_diverged_features.parquet        # BC_rel_diff TOP-N
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import polars as pl


# ════════════════════════════════════════════════════════════════
# Data loading
# ════════════════════════════════════════════════════════════════
def load_production_csv(csv_path: Path, start: datetime, end: datetime) -> pd.DataFrame:
    """production triggered_features_log.csv を読み、 timestamp 範囲でフィルタ"""
    print(f"  reading production CSV: {csv_path}")
    df = pd.read_csv(csv_path, low_memory=False, on_bad_lines="skip")
    # production CSV のカラム名は "Timestamp" (大文字)
    ts_col = "Timestamp" if "Timestamp" in df.columns else "timestamp"
    df[ts_col] = pd.to_datetime(df[ts_col], utc=True, errors="coerce")
    df = df.dropna(subset=[ts_col])
    df = df[(df[ts_col] >= start) & (df[ts_col] <= end)].copy()
    df = df.rename(columns={ts_col: "timestamp"})
    print(f"    → {len(df)} 行, {len(df.columns)} cols")
    return df


def load_training_parquet(
    parquet_path: Path, start: datetime, end: datetime
) -> pd.DataFrame:
    """training_snapshot_with_pre_ols.parquet を読み、 期間フィルタ。 polars→pandas で返す。"""
    print(f"  reading training parquet: {parquet_path}")
    df_pl = pl.read_parquet(parquet_path)
    df_pl = df_pl.with_columns(pl.col("timestamp").cast(pl.Datetime("us", "UTC")))
    df_pl = df_pl.filter((pl.col("timestamp") >= start) & (pl.col("timestamp") <= end))
    df_pd = df_pl.to_pandas()
    df_pd["timestamp"] = pd.to_datetime(df_pd["timestamp"], utc=True)
    print(f"    → {len(df_pd)} 行, {len(df_pd.columns)} cols")
    return df_pd


# ════════════════════════════════════════════════════════════════
# Pair identification
# ════════════════════════════════════════════════════════════════
def identify_pre_post_pairs(training_cols: List[str]) -> Dict[str, str]:
    """training snapshot の列名から 純化後 → 純化前 のマップを作る。

    純化後の規約: e1a_<base>_neutralized_<TF>  (例: e1a_statistical_mean_10_neutralized_M3)
    純化前の規約: e1a_<base>_<TF>             (例: e1a_statistical_mean_10_M3)

    対応関係: 純化後列名から `_neutralized_` を `_` に置換 → 純化前列名
    """
    post_cols = [c for c in training_cols if "_neutralized_" in c]
    training_set = set(training_cols)

    pairs: Dict[str, str] = {}
    for post_col in post_cols:
        pre_col = post_col.replace("_neutralized_", "_")
        if pre_col in training_set:
            pairs[post_col] = pre_col
    return pairs


# ════════════════════════════════════════════════════════════════
# 3 way metrics computation
# ════════════════════════════════════════════════════════════════
def compute_3way_metrics(
    df_train: pd.DataFrame,
    df_prod: pd.DataFrame,
    pre_post_pairs: Dict[str, str],
    dump_raw_features: Optional[List[str]] = None,
    dump_raw_path: Optional[Path] = None,
) -> pd.DataFrame:
    """各 feature の 3 way 比較メトリクスを計算。

    A = training pre   (純化前)
    B = training post  (学習側 純化後)
    C = production     (本番側 純化後)

    各 feature について全行で:
      AB_diff = B - A   (学習側 OLS が動かした量)
      BC_diff = C - B   (学習側 vs 本番側の最終差)
      AC_diff = C - A   (本番側 OLS が動かした量)
    """
    print("  merging training/production by timestamp...")
    # timestamp inner join
    df_train["timestamp"] = pd.to_datetime(df_train["timestamp"], utc=True)
    df_prod["timestamp"] = pd.to_datetime(df_prod["timestamp"], utc=True)
    merged = df_train.merge(
        df_prod, on="timestamp", how="inner", suffixes=("", "_prod")
    )
    print(f"    → merged: {len(merged)} 行")

    if len(merged) == 0:
        print("  ❌ 共通 timestamp 0、 比較不能")
        return pd.DataFrame()

    rows = []
    # [§11.34 raw dump] 指定 feature の各時点 timestamp/A/B/C を縦持ちで蓄積
    _dump_set = set(dump_raw_features) if dump_raw_features else set()
    _dump_rows: List[dict] = []
    available = [
        (post, pre) for post, pre in pre_post_pairs.items() if post in df_prod.columns
    ]
    print(f"  processing {len(available)} features...")

    for post_col, pre_col in available:
        # training 列名はそのまま、 prod 列名は suffix なしで存在
        if pre_col not in merged.columns:
            continue
        if post_col not in merged.columns:
            continue
        # post_col は training 由来。 prod 同名は `{post_col}_prod` (training と重複時のみ)
        # training と prod で同名なので merge は重複する → suffixes=("", "_prod") で
        # training 側が post_col、 prod 側が {post_col}_prod になる
        prod_col_name = (
            f"{post_col}_prod" if f"{post_col}_prod" in merged.columns else post_col
        )
        # 念のため: merge で suffix が付かなかった (= 重複なし or only one side) ケース
        if prod_col_name not in merged.columns:
            continue

        A = pd.to_numeric(merged[pre_col], errors="coerce").to_numpy(dtype=np.float64)
        B = pd.to_numeric(merged[post_col], errors="coerce").to_numpy(dtype=np.float64)
        C = pd.to_numeric(merged[prod_col_name], errors="coerce").to_numpy(
            dtype=np.float64
        )

        # 三者とも有限値のみ
        mask = np.isfinite(A) & np.isfinite(B) & np.isfinite(C)
        if mask.sum() < 2:
            continue
        A, B, C = A[mask], B[mask], C[mask]

        # [§11.34 raw dump] 指定 feature なら timestamp 付きで各時点を残す
        if post_col in _dump_set:
            _ts = pd.to_datetime(merged["timestamp"], utc=True).to_numpy()[mask]
            for _i in range(len(A)):
                _dump_rows.append(
                    {
                        "feature": post_col,
                        "timestamp": _ts[_i],
                        "A_pre": float(A[_i]),
                        "B_train_post": float(B[_i]),
                        "C_prod_post": float(C[_i]),
                        "AB_diff": float(B[_i] - A[_i]),
                        "BC_diff": float(C[_i] - B[_i]),
                        "AC_diff": float(C[_i] - A[_i]),
                    }
                )

        AB_diff = B - A
        BC_diff = C - B
        AC_diff = C - A

        # 基本統計
        A_mean, A_std = float(A.mean()), float(A.std())
        B_mean, B_std = float(B.mean()), float(B.std())
        C_mean, C_std = float(C.mean()), float(C.std())

        # AB: 学習側 OLS が動かした量
        AB_abs_mean = float(np.abs(AB_diff).mean())
        AB_signed_mean = float(AB_diff.mean())
        AB_rel = AB_abs_mean / (np.abs(A).mean() + 1e-10)

        # BC: 学習側 vs 本番側 純化後の差 (← これが真因の発現箇所)
        BC_abs_mean = float(np.abs(BC_diff).mean())
        BC_signed_mean = float(BC_diff.mean())
        BC_rel = BC_abs_mean / (np.abs(B).mean() + 1e-10)
        # 相関 (B と C の方向性)
        BC_corr = (
            float(np.corrcoef(B, C)[0, 1]) if B.std() > 0 and C.std() > 0 else np.nan
        )
        # 符号反転率
        sign_flip = float(np.mean(np.sign(B) != np.sign(C))) if len(B) > 0 else np.nan
        # B と C の比 (= 本番側純化が学習側の何倍か)
        ratio = float(np.abs(C).mean() / (np.abs(B).mean() + 1e-10))

        # AC: 本番側 OLS が動かした量
        AC_abs_mean = float(np.abs(AC_diff).mean())
        AC_signed_mean = float(AC_diff.mean())
        AC_rel = AC_abs_mean / (np.abs(A).mean() + 1e-10)

        rows.append(
            {
                "feature": post_col,
                "pre_col": pre_col,
                "n": int(mask.sum()),
                # 基本統計
                "A_mean": A_mean,
                "A_std": A_std,
                "B_mean": B_mean,
                "B_std": B_std,
                "C_mean": C_mean,
                "C_std": C_std,
                # AB: 学習側 OLS の動かし量
                "AB_signed_mean": AB_signed_mean,
                "AB_abs_mean": AB_abs_mean,
                "AB_rel": AB_rel,
                # BC: 学習側 vs 本番側 純化後の差
                "BC_signed_mean": BC_signed_mean,
                "BC_abs_mean": BC_abs_mean,
                "BC_rel": BC_rel,
                "BC_corr": BC_corr,
                "BC_sign_flip_rate": sign_flip,
                "BC_ratio_C_over_B": ratio,
                # AC: 本番側 OLS の動かし量
                "AC_signed_mean": AC_signed_mean,
                "AC_abs_mean": AC_abs_mean,
                "AC_rel": AC_rel,
            }
        )

    # [§11.34 raw dump] 指定 feature の各時点生値を縦持ちで出力
    if _dump_set and dump_raw_path is not None:
        if _dump_rows:
            _dump_df = pd.DataFrame(_dump_rows).sort_values(
                ["feature", "timestamp"]
            )
            _dump_df.to_parquet(dump_raw_path, index=False)
            print(
                f"  [raw dump] {len(_dump_df)} 行 ({_dump_df['feature'].nunique()} feature) → {dump_raw_path}"
            )
        else:
            print(
                f"  [raw dump] ⚠ 指定 feature が merged に見つからず 0 行: {sorted(_dump_set)}"
            )

    return pd.DataFrame(rows)


# ════════════════════════════════════════════════════════════════
# Report generation
# ════════════════════════════════════════════════════════════════
def generate_report(
    metrics: pd.DataFrame,
    args,
    n_train_rows: int,
    n_prod_rows: int,
    n_pairs: int,
    n_compared: int,
) -> str:
    md = []
    md.append("# OLS 純化 3 way 比較レポート")
    md.append("")
    md.append(f"- 期間: {args.start} 〜 {args.end}")
    md.append(f"- training rows: {n_train_rows}")
    md.append(f"- production rows: {n_prod_rows}")
    md.append(f"- 純化前後ペア (training): {n_pairs}")
    md.append(f"- 3 way 比較対象 (production にも存在): {n_compared}")
    md.append("")

    if len(metrics) == 0:
        md.append("⚠️ 比較対象 0、 共通 timestamp が無い可能性")
        return "\n".join(md)

    # 区別: A=純化前, B=学習純化後, C=本番純化後
    md.append("## 凡例")
    md.append("- **A** = 学習側 純化前 (S2 由来)")
    md.append("- **B** = 学習側 純化後 (S6 由来)")
    md.append("- **C** = 本番側 純化後 (triggered_features_log.csv)")
    md.append("")
    md.append("- **AB_rel** = `|B-A|/|A|` → 学習側 OLS が値を動かした相対量")
    md.append("- **BC_rel** = `|C-B|/|B|` → 学習側 vs 本番側 純化後の差")
    md.append("- **AC_rel** = `|C-A|/|A|` → 本番側 OLS が値を動かした相対量")
    md.append("")

    # ────────────────────────────────────
    # [A vs B] 学習側 OLS の影響度
    # ────────────────────────────────────
    md.append(
        "## [A vs B] 学習側 OLS の影響度 (= 学習側純化が pre をどれだけ動かしたか)"
    )
    md.append("")
    md.append("AB_rel の分布:")
    md.append("")
    md.append("| 統計 | 値 |")
    md.append("|---|---|")
    md.append(f"| median | {metrics['AB_rel'].median():.4f} |")
    md.append(f"| mean   | {metrics['AB_rel'].mean():.4f} |")
    md.append(f"| p25    | {metrics['AB_rel'].quantile(0.25):.4f} |")
    md.append(f"| p75    | {metrics['AB_rel'].quantile(0.75):.4f} |")
    md.append(f"| p90    | {metrics['AB_rel'].quantile(0.90):.4f} |")
    md.append(f"| p99    | {metrics['AB_rel'].quantile(0.99):.4f} |")
    md.append(f"| max    | {metrics['AB_rel'].max():.4f} |")
    md.append("")

    # ────────────────────────────────────
    # [B vs C] 学習側 vs 本番側 純化後 ← 真因の発現箇所
    # ────────────────────────────────────
    md.append("## [B vs C] 学習側純化後 vs 本番側純化後 ← 真因の発現箇所")
    md.append("")

    md.append("### BC_rel (相対差) の分布")
    md.append("")
    md.append("| 統計 | 値 |")
    md.append("|---|---|")
    md.append(f"| median | {metrics['BC_rel'].median():.4f} |")
    md.append(f"| mean   | {metrics['BC_rel'].mean():.4f} |")
    md.append(f"| p25    | {metrics['BC_rel'].quantile(0.25):.4f} |")
    md.append(f"| p75    | {metrics['BC_rel'].quantile(0.75):.4f} |")
    md.append(f"| p90    | {metrics['BC_rel'].quantile(0.90):.4f} |")
    md.append(f"| p99    | {metrics['BC_rel'].quantile(0.99):.4f} |")
    md.append(f"| max    | {metrics['BC_rel'].max():.4f} |")
    md.append("")

    bit_identical = int((metrics["BC_rel"] < 1e-7).sum())
    minor = int(((metrics["BC_rel"] >= 1e-7) & (metrics["BC_rel"] < 1e-3)).sum())
    moderate = int(((metrics["BC_rel"] >= 1e-3) & (metrics["BC_rel"] < 0.1)).sum())
    severe = int((metrics["BC_rel"] >= 0.1).sum())
    md.append(f"- **bit-identical (BC_rel < 1e-7)**:  {bit_identical}")
    md.append(f"- 軽微     (1e-7 ≤ BC_rel < 1e-3): {minor}")
    md.append(f"- 中程度   (1e-3 ≤ BC_rel < 0.1):  {moderate}")
    md.append(f"- **重度乖離 (BC_rel ≥ 0.1)**:      {severe}")
    md.append("")

    md.append("### BC_corr (B と C の相関) の分布")
    md.append("")
    md.append("| 統計 | 値 |")
    md.append("|---|---|")
    bc_corr = metrics["BC_corr"].dropna()
    if len(bc_corr) > 0:
        md.append(f"| median | {bc_corr.median():.4f} |")
        md.append(f"| mean   | {bc_corr.mean():.4f} |")
        md.append(f"| p10    | {bc_corr.quantile(0.10):.4f} |")
        md.append(f"| p25    | {bc_corr.quantile(0.25):.4f} |")
        md.append(f"| min    | {bc_corr.min():.4f} |")
    md.append("")
    md.append(f"- 高相関 (corr > 0.95):    {int((bc_corr > 0.95).sum())}")
    md.append(
        f"- 中相関 (0.5 < corr ≤ 0.95): {int(((bc_corr > 0.5) & (bc_corr <= 0.95)).sum())}"
    )
    md.append(
        f"- 低相関 (-0.5 ≤ corr ≤ 0.5): {int(((bc_corr >= -0.5) & (bc_corr <= 0.5)).sum())}"
    )
    md.append(f"- **負相関 (corr < -0.5)**:    {int((bc_corr < -0.5).sum())}")
    md.append("")

    md.append("### 符号反転率の分布")
    md.append("")
    md.append("| 統計 | 値 |")
    md.append("|---|---|")
    md.append(f"| median | {metrics['BC_sign_flip_rate'].median():.4f} |")
    md.append(f"| mean   | {metrics['BC_sign_flip_rate'].mean():.4f} |")
    md.append(f"| max    | {metrics['BC_sign_flip_rate'].max():.4f} |")
    md.append("")
    md.append(
        f"- 符号一致 (sign_flip < 0.1): {int((metrics['BC_sign_flip_rate'] < 0.1).sum())}"
    )
    md.append(
        f"- 部分反転 (0.1 ≤ sign_flip < 0.5): {int(((metrics['BC_sign_flip_rate'] >= 0.1) & (metrics['BC_sign_flip_rate'] < 0.5)).sum())}"
    )
    md.append(
        f"- **多数反転 (sign_flip ≥ 0.5)**: {int((metrics['BC_sign_flip_rate'] >= 0.5).sum())}"
    )
    md.append("")

    # ────────────────────────────────────
    # [A vs C] 本番側 OLS の影響度
    # ────────────────────────────────────
    md.append("## [A vs C] 本番側 OLS の影響度")
    md.append("")
    md.append("AC_rel の分布:")
    md.append("")
    md.append("| 統計 | 値 |")
    md.append("|---|---|")
    md.append(f"| median | {metrics['AC_rel'].median():.4f} |")
    md.append(f"| mean   | {metrics['AC_rel'].mean():.4f} |")
    md.append(f"| max    | {metrics['AC_rel'].max():.4f} |")
    md.append("")

    # ────────────────────────────────────
    # [AB vs AC 比較] 学習側 OLS と本番側 OLS の動かし量の対比
    # ────────────────────────────────────
    md.append("## [AB vs AC] 学習側 OLS と本番側 OLS の動かし量比較")
    md.append("")
    md.append("両者が同じ量だけ動かしていれば AB≈AC、 違う方向に動かしていれば不一致。")
    md.append("")
    metrics["AB_AC_ratio"] = metrics["AC_abs_mean"] / (metrics["AB_abs_mean"] + 1e-10)
    md.append("AC_abs_mean / AB_abs_mean の比 (= 本番側純化が学習側の何倍動かしたか):")
    md.append("")
    md.append("| 統計 | 値 |")
    md.append("|---|---|")
    md.append(f"| median | {metrics['AB_AC_ratio'].median():.4f} |")
    md.append(f"| mean   | {metrics['AB_AC_ratio'].mean():.4f} |")
    md.append(f"| p10    | {metrics['AB_AC_ratio'].quantile(0.10):.4f} |")
    md.append(f"| p90    | {metrics['AB_AC_ratio'].quantile(0.90):.4f} |")
    md.append("")
    md.append("→ 1.0 ≈ 同量、 << 1.0 ≈ 本番が動かなさすぎ、 >> 1.0 ≈ 本番が動かしすぎ")
    md.append("")

    # ────────────────────────────────────
    # TOP-N
    # ────────────────────────────────────
    md.append(f"## TOP-{args.top_n} 乖離 features (BC_rel 降順)")
    md.append("")
    top = metrics.nlargest(args.top_n, "BC_rel")[
        [
            "feature",
            "n",
            "BC_rel",
            "BC_corr",
            "BC_sign_flip_rate",
            "B_mean",
            "C_mean",
            "AB_abs_mean",
            "AC_abs_mean",
        ]
    ]
    md.append(
        "| feature | n | BC_rel | BC_corr | sign_flip | B_mean | C_mean | AB_abs | AC_abs |"
    )
    md.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|")
    for _, r in top.iterrows():
        md.append(
            f"| {r['feature'][:60]} | {int(r['n'])} | "
            f"{r['BC_rel']:.3f} | {r['BC_corr']:.3f} | "
            f"{r['BC_sign_flip_rate']:.2f} | "
            f"{r['B_mean']:.4f} | {r['C_mean']:.4f} | "
            f"{r['AB_abs_mean']:.4f} | {r['AC_abs_mean']:.4f} |"
        )
    md.append("")

    # ────────────────────────────────────
    # 結論
    # ────────────────────────────────────
    md.append("## 結論")
    md.append("")
    if bit_identical / len(metrics) > 0.95:
        md.append(
            f"- ✅ {bit_identical}/{len(metrics)} ({100 * bit_identical / len(metrics):.1f}%) が bit-identical"
        )
        md.append("- 学習側 OLS と 本番側 OLS は 数値的に一致 → 真因は別の場所")
    elif severe / len(metrics) > 0.5:
        md.append(
            f"- ❌ {severe}/{len(metrics)} ({100 * severe / len(metrics):.1f}%) が重度乖離"
        )
        md.append("- 学習側 OLS と 本番側 OLS は systematic に違う計算をしている")
        if bc_corr.median() < -0.5:
            md.append(
                "- **特に corr 中央値が負** → 符号反転が支配的、 β の符号が反転している可能性"
            )
        elif bc_corr.median() < 0.5:
            md.append(
                "- corr 中央値が低い → ランダムに近い乖離、 OLS state 自体が壊れている可能性"
            )
        else:
            md.append(
                "- corr は高め → 方向性は一致するがスケールが違う、 β/α のスケール差の可能性"
            )
    else:
        md.append(
            f"- ⚠️  {severe}/{len(metrics)} ({100 * severe / len(metrics):.1f}%) 重度乖離、 {bit_identical} bit-identical"
        )
        md.append("- 一部の feature でのみ OLS 純化が壊れている")

    return "\n".join(md)


# ════════════════════════════════════════════════════════════════
# main
# ════════════════════════════════════════════════════════════════
def main():
    parser = argparse.ArgumentParser(description="3 way OLS comparison")
    parser.add_argument(
        "--training",
        type=Path,
        default="/workspace/data/diagnostics/training_snapshot_with_pre_ols_20260525.parquet",
    )
    parser.add_argument(
        "--production",
        type=Path,
        default="/workspace/logs/triggered_features_log.csv",
    )
    parser.add_argument(
        "--start", default="2026-05-25 12:00:00", help="YYYY-MM-DD HH:MM:SS UTC"
    )
    parser.add_argument(
        "--end", default="2026-05-25 13:30:00", help="YYYY-MM-DD HH:MM:SS UTC"
    )
    parser.add_argument(
        "--out-dir", type=Path, default="/workspace/data/diagnostics/compare_3way_ols"
    )
    parser.add_argument("--top-n", type=int, default=30)
    parser.add_argument(
        "--dump-raw",
        type=str,
        default="",
        help="生値を縦持ちで残す feature 名 (post_col、 カンマ区切り)。 例: e1d_..._neutralized_M3,e1f_..._neutralized_M1",
    )
    parser.add_argument(
        "--dump-raw-path",
        type=Path,
        default=None,
        help="生値 dump の出力 parquet パス (未指定時は out-dir/raw_abc_dump.parquet)",
    )
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    start_dt = datetime.strptime(args.start, "%Y-%m-%d %H:%M:%S").replace(
        tzinfo=timezone.utc
    )
    end_dt = datetime.strptime(args.end, "%Y-%m-%d %H:%M:%S").replace(
        tzinfo=timezone.utc
    )

    print("=" * 72)
    print("  OLS 純化 3 way 比較")
    print("=" * 72)
    print(f"  期間: {start_dt} 〜 {end_dt}")
    print(f"  出力: {args.out_dir}")
    print()

    print("--- 1. データ読み込み ---")
    df_train = load_training_parquet(args.training, start_dt, end_dt)
    df_prod = load_production_csv(args.production, start_dt, end_dt)

    print()
    print("--- 2. 純化前後ペア識別 ---")
    pre_post_pairs = identify_pre_post_pairs(list(df_train.columns))
    print(f"  training 内の純化前後ペア: {len(pre_post_pairs)}")
    available = {k: v for k, v in pre_post_pairs.items() if k in df_prod.columns}
    print(f"  production にも存在 (= 比較対象): {len(available)}")

    if len(available) == 0:
        print("  ❌ 比較対象 0、 終了")
        sys.exit(1)

    print()
    print("--- 3. 3 way 比較メトリクス計算 ---")
    _dump_feats = [f.strip() for f in args.dump_raw.split(",") if f.strip()]
    _dump_path = args.dump_raw_path
    if _dump_feats and _dump_path is None:
        _dump_path = args.out_dir / "raw_abc_dump.parquet"
    if _dump_feats:
        print(f"  [raw dump] 対象 {len(_dump_feats)} feature → {_dump_path}")
    metrics = compute_3way_metrics(
        df_train,
        df_prod,
        available,
        dump_raw_features=_dump_feats,
        dump_raw_path=_dump_path,
    )
    print(f"  計算成功: {len(metrics)} features")

    if len(metrics) == 0:
        print("  ❌ メトリクス 0、 終了")
        sys.exit(1)

    # 保存
    metrics_path = args.out_dir / "metrics_per_feature.parquet"
    metrics.to_parquet(metrics_path, index=False)
    print(f"  → {metrics_path}")

    top_n_df = metrics.nlargest(args.top_n, "BC_rel")
    top_n_df.to_parquet(args.out_dir / "top_diverged_features.parquet", index=False)

    print()
    print("--- 4. レポート生成 ---")
    report = generate_report(
        metrics,
        args,
        n_train_rows=len(df_train),
        n_prod_rows=len(df_prod),
        n_pairs=len(pre_post_pairs),
        n_compared=len(available),
    )
    report_path = args.out_dir / "report.md"
    report_path.write_text(report, encoding="utf-8")
    print(f"  → {report_path}")

    print()
    print("=" * 72)
    print("✅ 完了")
    print("=" * 72)
    print(report[:3000])  # report 先頭を stdout に出す
    if len(report) > 3000:
        print(f"... ({len(report) - 3000} 文字省略、 全文は {report_path})")


if __name__ == "__main__":
    main()
