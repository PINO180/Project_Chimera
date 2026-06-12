#!/usr/bin/env python3
"""
analyze_3way_structure.py — compare_3way_ols の結果を構造的に分析

[目的]
  metrics_per_feature.parquet (= compare_3way_ols.py の出力) を読み、
  engine × TF × window の各次元で BC_rel / BC_corr / 符号反転率の分布を見る。
  これにより:
    - 特定 TF だけ重度乖離 → deque 充填の問題 (warmup)
    - 特定 engine だけ → 計算経路の問題
    - 全 TF/engine 均一 → market_proxy 自体の問題
    - 特定 window のみ → OLS_WINDOW_PER_TF の値の問題
  という真因の方向性を絞り込む。

[呼び出し例]
  python analyze_3way_structure.py
  (デフォルトで /workspace/data/diagnostics/compare_3way_ols/metrics_per_feature.parquet を読む)
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pandas as pd

# ════════════════════════════════════════════════════════════════
# Feature name parser
# ════════════════════════════════════════════════════════════════
TF_ORDER = ["M0.5", "M1", "M3", "M5", "M8", "M15"]
ENGINE_ORDER = ["e1a", "e1b", "e1c", "e1d", "e1e", "e1f"]


def parse_feature_name(name: str) -> Tuple[Optional[str], Optional[str], Optional[str], Optional[str]]:
    """
    e1a_statistical_mean_10_neutralized_M3 → ('e1a', 'M3', '10', 'statistical_mean')
    e1c_rsi_14_neutralized_M5             → ('e1c', 'M5', '14', 'rsi')
    e1f_musical_tension_96_neutralized_M15 → ('e1f', 'M15', '96', 'musical_tension')
    """
    m = re.match(r"^(e1[a-f])_(.+?)_neutralized_(M[0-9\.]+)$", name)
    if not m:
        return None, None, None, None
    engine, base, tf = m.groups()
    # 末尾の数字を window として抽出
    win_match = re.search(r"_(\d+)$", base)
    if win_match:
        window = win_match.group(1)
        func = base[: win_match.start()]
    else:
        window = None
        func = base
    return engine, tf, window, func


# ════════════════════════════════════════════════════════════════
# Aggregation utilities
# ════════════════════════════════════════════════════════════════
def agg_metrics(df: pd.DataFrame, by: str) -> pd.DataFrame:
    """指定列でグループ化して BC_rel / BC_corr / 符号反転率を集計"""
    return df.groupby(by, dropna=False).agg(
        n=("feature", "count"),
        bc_rel_median=("BC_rel", "median"),
        bc_rel_p90=("BC_rel", lambda x: x.quantile(0.90)),
        bc_corr_median=("BC_corr", "median"),
        bc_corr_p10=("BC_corr", lambda x: x.quantile(0.10)),
        sign_flip_median=("BC_sign_flip_rate", "median"),
        bit_identical=("BC_rel", lambda x: int((x < 1e-7).sum())),
        severe=("BC_rel", lambda x: int((x >= 0.1).sum())),
    )


def fmt_table(df: pd.DataFrame, name: str) -> str:
    """DataFrame を Markdown テーブルに整形"""
    lines = [f"### {name}", ""]
    cols = df.columns.tolist()
    lines.append("| " + " | ".join([df.index.name or "key"] + cols) + " |")
    lines.append("|" + "|".join(["---"] * (len(cols) + 1)) + "|")
    for idx, row in df.iterrows():
        vals = []
        for c in cols:
            v = row[c]
            if isinstance(v, (int, np.integer)):
                vals.append(f"{int(v)}")
            elif isinstance(v, (float, np.floating)):
                if np.isnan(v):
                    vals.append("nan")
                elif abs(v) > 1e5 or (abs(v) < 1e-3 and v != 0):
                    vals.append(f"{v:.2e}")
                else:
                    vals.append(f"{v:.4f}")
            else:
                vals.append(str(v))
        lines.append("| " + " | ".join([str(idx)] + vals) + " |")
    lines.append("")
    return "\n".join(lines)


# ════════════════════════════════════════════════════════════════
# main
# ════════════════════════════════════════════════════════════════
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--metrics", type=Path,
        default="/workspace/data/diagnostics/compare_3way_ols/metrics_per_feature.parquet",
    )
    parser.add_argument(
        "--out-dir", type=Path,
        default="/workspace/data/diagnostics/compare_3way_ols",
    )
    args = parser.parse_args()

    print("=" * 72)
    print("  3 way OLS 構造分析")
    print("=" * 72)
    df = pd.read_parquet(args.metrics)
    print(f"  metrics 読込: {len(df)} features")

    # ─── parse feature names ──────────────────────────────────
    parsed = df["feature"].apply(parse_feature_name)
    df["engine"] = parsed.apply(lambda x: x[0])
    df["tf"]     = parsed.apply(lambda x: x[1])
    df["window"] = parsed.apply(lambda x: x[2])
    df["func"]   = parsed.apply(lambda x: x[3])
    n_parsed = df["engine"].notna().sum()
    print(f"  parse 成功: {n_parsed}/{len(df)} features")

    df_p = df[df["engine"].notna()].copy()

    # 数値変換 (extreme value を緩和するため、 log scale 用)
    df_p["window_int"] = pd.to_numeric(df_p["window"], errors="coerce")
    df_p["B_abs"] = df_p["B_mean"].abs()
    df_p["C_abs"] = df_p["C_mean"].abs()

    # ============================================================
    md_lines = []
    md_lines.append("# 3 way OLS 構造分析レポート")
    md_lines.append("")
    md_lines.append(f"- 元データ: {args.metrics}")
    md_lines.append(f"- 全 features: {len(df)}, parse 成功: {n_parsed}")
    md_lines.append("")

    # ─── 1. Engine 別 ───────────────────────────────────
    print("\n--- 1. Engine 別 集計 ---")
    g_engine = agg_metrics(df_p, "engine").reindex(
        [e for e in ENGINE_ORDER if e in df_p["engine"].unique()]
    )
    print(g_engine.to_string())
    md_lines.append("## Engine 別 集計")
    md_lines.append("")
    md_lines.append(fmt_table(g_engine, "BC_rel / BC_corr / 符号反転率"))

    # ─── 2. TF 別 ────────────────────────────────────
    print("\n--- 2. TF 別 集計 ---")
    g_tf = agg_metrics(df_p, "tf").reindex(
        [tf for tf in TF_ORDER if tf in df_p["tf"].unique()]
    )
    print(g_tf.to_string())
    md_lines.append("## TF 別 集計")
    md_lines.append("")
    md_lines.append(fmt_table(g_tf, "BC_rel / BC_corr / 符号反転率"))

    # ─── 3. Engine × TF (BC_corr 中央値) ─────────────
    print("\n--- 3. Engine × TF (BC_corr 中央値) ---")
    pivot_corr = df_p.pivot_table(
        index="engine", columns="tf", values="BC_corr", aggfunc="median"
    )
    pivot_corr = pivot_corr.reindex(
        index=[e for e in ENGINE_ORDER if e in pivot_corr.index],
        columns=[tf for tf in TF_ORDER if tf in pivot_corr.columns],
    )
    print(pivot_corr.to_string())
    md_lines.append("## Engine × TF: BC_corr 中央値 heatmap")
    md_lines.append("")
    md_lines.append("（1.0 に近いほど学習側と本番側 純化値が一致、 0 はランダム、 -1.0 は完全反転）")
    md_lines.append("")
    md_lines.append("| engine \\ TF | " + " | ".join(pivot_corr.columns) + " |")
    md_lines.append("|" + "|".join(["---"] * (len(pivot_corr.columns) + 1)) + "|")
    for idx, row in pivot_corr.iterrows():
        vals = [f"{v:.3f}" if not pd.isna(v) else "nan" for v in row]
        md_lines.append("| " + idx + " | " + " | ".join(vals) + " |")
    md_lines.append("")

    # ─── 4. Engine × TF (sign_flip 中央値) ─────────────
    print("\n--- 4. Engine × TF (符号反転率 中央値) ---")
    pivot_sf = df_p.pivot_table(
        index="engine", columns="tf", values="BC_sign_flip_rate", aggfunc="median"
    )
    pivot_sf = pivot_sf.reindex(
        index=[e for e in ENGINE_ORDER if e in pivot_sf.index],
        columns=[tf for tf in TF_ORDER if tf in pivot_sf.columns],
    )
    print(pivot_sf.to_string())
    md_lines.append("## Engine × TF: 符号反転率 中央値")
    md_lines.append("")
    md_lines.append("（0 で完全一致、 0.5 で coin flip）")
    md_lines.append("")
    md_lines.append("| engine \\ TF | " + " | ".join(pivot_sf.columns) + " |")
    md_lines.append("|" + "|".join(["---"] * (len(pivot_sf.columns) + 1)) + "|")
    for idx, row in pivot_sf.iterrows():
        vals = [f"{v:.3f}" if not pd.isna(v) else "nan" for v in row]
        md_lines.append("| " + idx + " | " + " | ".join(vals) + " |")
    md_lines.append("")

    # ─── 5. TF 別の bit-identical 数 / 重度乖離数 ────
    print("\n--- 5. TF 別の判定数 ---")
    counts_by_tf = df_p.groupby("tf").agg(
        n=("feature", "count"),
        bit_identical=("BC_rel", lambda x: int((x < 1e-7).sum())),
        minor=("BC_rel", lambda x: int(((x >= 1e-7) & (x < 1e-3)).sum())),
        moderate=("BC_rel", lambda x: int(((x >= 1e-3) & (x < 0.1)).sum())),
        severe=("BC_rel", lambda x: int((x >= 0.1).sum())),
    ).reindex([tf for tf in TF_ORDER if tf in df_p["tf"].unique()])
    print(counts_by_tf.to_string())
    md_lines.append("## TF 別 判定数")
    md_lines.append("")
    md_lines.append(fmt_table(counts_by_tf, "bit-identical / 軽微 / 中程度 / 重度"))

    # ─── 6. window 別 (主要 window のみ) ─────────────
    print("\n--- 6. Window 別 (主要 Top 12) ---")
    g_window = df_p.groupby("window_int", dropna=True).agg(
        n=("feature", "count"),
        bc_rel_median=("BC_rel", "median"),
        bc_corr_median=("BC_corr", "median"),
        sign_flip_median=("BC_sign_flip_rate", "median"),
        bit_identical=("BC_rel", lambda x: int((x < 1e-7).sum())),
        severe=("BC_rel", lambda x: int((x >= 0.1).sum())),
    ).sort_values("n", ascending=False).head(12)
    g_window.index = g_window.index.astype(int)
    g_window.index.name = "window"
    print(g_window.to_string())
    md_lines.append("## Window 別 (主要 Top 12)")
    md_lines.append("")
    md_lines.append(fmt_table(g_window, "Window サイズ別 (Top 12)"))

    # ─── 7. 異常パターン: 「B が ~0」 「C が ~巨大」 ──
    print("\n--- 7. 異常パターン分析 (B≈0, C 巨大) ---")
    anomaly = df_p[(df_p["B_abs"] < 0.01) & (df_p["C_abs"] > 1.0)]
    pct_anomaly = 100 * len(anomaly) / len(df_p) if len(df_p) > 0 else 0
    print(f"  anomaly count: {len(anomaly)}/{len(df_p)} ({pct_anomaly:.1f}%)")

    md_lines.append("## 異常パターン: 学習側 B≈0、 本番側 C=巨大")
    md_lines.append("")
    md_lines.append(f"- 該当数: {len(anomaly)}/{len(df_p)} ({pct_anomaly:.1f}%)")
    md_lines.append("- 解釈: 学習側 OLS で純化されてほぼ 0 になる feature が、 本番側 OLS では大きな値に。 = 本番側 OLS が機能していないパターン")
    md_lines.append("")
    md_lines.append("### Engine 分布")
    md_lines.append("")
    eng_dist = anomaly["engine"].value_counts().reindex(ENGINE_ORDER, fill_value=0)
    md_lines.append("| engine | count |")
    md_lines.append("|---|---:|")
    for e, c in eng_dist.items():
        md_lines.append(f"| {e} | {int(c)} |")
    md_lines.append("")
    md_lines.append("### TF 分布")
    md_lines.append("")
    tf_dist = anomaly["tf"].value_counts().reindex(TF_ORDER, fill_value=0)
    md_lines.append("| TF | count |")
    md_lines.append("|---|---:|")
    for tf, c in tf_dist.items():
        md_lines.append(f"| {tf} | {int(c)} |")
    md_lines.append("")

    # ─── 8. 全 BC_corr 分布 (重要) ────────────────
    print("\n--- 8. BC_corr 全体分布 ---")
    bins = [-1.01, -0.95, -0.5, -0.1, 0.1, 0.5, 0.95, 1.01]
    labels = ["[-1, -0.95]", "(-0.95, -0.5]", "(-0.5, -0.1]", "(-0.1, 0.1]",
              "(0.1, 0.5]", "(0.5, 0.95]", "(0.95, 1.0]"]
    df_p["corr_bin"] = pd.cut(df_p["BC_corr"], bins=bins, labels=labels)
    corr_dist = df_p["corr_bin"].value_counts().reindex(labels)
    for label, count in corr_dist.items():
        pct = 100 * count / len(df_p) if len(df_p) > 0 else 0
        print(f"  {label:18s}: {int(count):4d} ({pct:5.1f}%)")
    md_lines.append("## BC_corr 全体分布")
    md_lines.append("")
    md_lines.append("| 区間 | count | % |")
    md_lines.append("|---|---:|---:|")
    for label, count in corr_dist.items():
        pct = 100 * count / len(df_p) if len(df_p) > 0 else 0
        md_lines.append(f"| {label} | {int(count)} | {pct:.1f} |")
    md_lines.append("")

    # ─── 9. AC/AB 比 の分布 ─────────────────────────
    df_p["ac_ab_ratio"] = df_p["AC_abs_mean"] / (df_p["AB_abs_mean"] + 1e-10)
    print("\n--- 9. AC/AB 比 (本番側が学習側の何倍動かしたか) ---")
    print(f"  median: {df_p['ac_ab_ratio'].median():.3f}")
    print(f"  mean:   {df_p['ac_ab_ratio'].mean():.3f}")
    print(f"  p10:    {df_p['ac_ab_ratio'].quantile(0.10):.3f}")
    print(f"  p90:    {df_p['ac_ab_ratio'].quantile(0.90):.3f}")
    print(f"  AC/AB >  10 倍: {int((df_p['ac_ab_ratio'] > 10).sum())}")
    print(f"  AC/AB <  0.1倍: {int((df_p['ac_ab_ratio'] < 0.1).sum())}")
    md_lines.append("## AC/AB 比 (本番側が学習側の何倍動かしたか)")
    md_lines.append("")
    md_lines.append("| 統計 | 値 |")
    md_lines.append("|---|---:|")
    md_lines.append(f"| median | {df_p['ac_ab_ratio'].median():.3f} |")
    md_lines.append(f"| mean   | {df_p['ac_ab_ratio'].mean():.3f} |")
    md_lines.append(f"| p10    | {df_p['ac_ab_ratio'].quantile(0.10):.3f} |")
    md_lines.append(f"| p90    | {df_p['ac_ab_ratio'].quantile(0.90):.3f} |")
    md_lines.append(f"| AC/AB > 10倍   | {int((df_p['ac_ab_ratio'] > 10).sum())} |")
    md_lines.append(f"| AC/AB < 0.1倍  | {int((df_p['ac_ab_ratio'] < 0.1).sum())} |")
    md_lines.append("")

    # ─── 出力 ────────────────────────────────────────
    out_path = args.out_dir / "structure_analysis.md"
    out_path.write_text("\n".join(md_lines), encoding="utf-8")
    print(f"\n  → {out_path}")

    df_p.to_parquet(args.out_dir / "metrics_with_parse.parquet")
    print(f"  → {args.out_dir / 'metrics_with_parse.parquet'}")

    print("\n" + "=" * 72)
    print("✅ 完了")
    print("=" * 72)


if __name__ == "__main__":
    main()
