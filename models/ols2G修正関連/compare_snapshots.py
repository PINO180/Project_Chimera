#!/usr/bin/env python3
"""
compare_snapshots.py — Production × Training snapshot 乖離レポート

[目的]
  Production triggered_features_log.csv と、 snapshot_training_inference.py が
  生成した training_snapshot.parquet を突合し、以下のメトリクスを出力する:

    ① シグナル発火集合の比較 (一致率)
    ② 方向一致率 (BUY/SELL/HOLD クロス表 + 反転率)
    ③ 予測値の系統的乖離 (M1/M2 4種)
    ④ 特徴量別の系統的乖離 TOP-N (rel_diff 降順, |mean_diff| 降順)

[設計]
  - 両 snapshot を timestamp ベースで突合
  - 列順序の違いは列名ベース join で吸収 (Phase B 並列化由来の順序ぶれに対応)
  - HOLD 含む全行を比較対象とする (production triggered_features_log.csv は
    HOLD を記録しないため、 production の HOLD 行は「未記録」 として扱う)

[出力]
  - report.md: 人間可読サマリー
  - feature_diff_summary.parquet: 特徴量別乖離テーブル (全特徴量)
  - signal_set_details.parquet: prod_only / train_only / both の timestamp 詳細

[呼び出し例]
  python compare_snapshots.py \\
      --production /workspace/logs/triggered_features_log.csv \\
      --training /workspace/data/diagnostics/training_snapshot_20260525.parquet \\
      --start 2026-05-25 --end 2026-05-25 \\
      --start-time 21:00:00 --end-time 22:30:00 \\
      --out-dir /workspace/data/diagnostics/compare_20260525
"""

from __future__ import annotations

import argparse
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import polars as pl

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


# ════════════════════════════════════════════════════════════════
# 比較対象から除外する meta カラム (双方向に存在し、 比較不要なもの)
# ════════════════════════════════════════════════════════════════
EXCLUDE_FROM_FEATURE_COMPARE = {
    # 共通 meta
    "timestamp", "timeframe",
    "action", "close_price", "Price",
    # production CSV ヘッダー (capital case)
    "Timestamp", "Action",
    "P_Long_M2", "P_Short_M2", "P_Long_M1", "P_Short_M1",
    # 推論結果 (predictions と直接比較するため特徴量ランキングからは外す)
    "p_m1_long_raw", "p_m1_short_raw",
    "p_m1_long_logit", "p_m1_short_logit",
    "p_m2_long_raw", "p_m2_short_raw",
    "delta",
    "passes_atr_filter", "passes_delta_filter",
    "passes_proba_long", "passes_proba_short",
    # ラベル / メタデータ
    "is_trigger", "label", "label_long", "label_short",
    "uniqueness", "uniqueness_long", "uniqueness_short",
    "duration_long", "duration_short",
    "concurrency_long", "concurrency_short",
    "atr_value", "atr_ratio",
    # OHLCV 系
    "open", "high", "low", "close",
    "volume", "tick_count", "bid", "ask", "spread",
    "year", "month", "day", "disc",
    "t1", "direction", "exit_type", "first_ex_reason_int",
    "payoff_ratio", "payoff_ratio_long", "payoff_ratio_short",
    "pt_multiplier", "sl_multiplier",
    "calculated_body_ratio", "fallback_vol",
    "m1_pred_proba", "meta_label",
}


# ════════════════════════════════════════════════════════════════
# 1. データロード
# ════════════════════════════════════════════════════════════════
def load_production(path: Path) -> pl.DataFrame:
    """triggered_features_log.csv を読み込み、 標準カラム名に正規化。

    main.py L1377-1402 のヘッダー:
      Header: Timestamp, Action, Price, P_Long_M2, P_Short_M2, + feature_keys
    """
    logger.info(f"Production snapshot: {path}")
    if not path.exists():
        raise FileNotFoundError(f"production csv が存在しない: {path}")

    df = pl.read_csv(path, try_parse_dates=True, infer_schema_length=10000)
    logger.info(f"  raw cols (先頭10): {df.columns[:10]}")

    # 列名正規化 (capital case → snake_case)
    rename_map = {}
    for old, new in [
        ("Timestamp",  "timestamp"),
        ("Action",     "action"),
        ("Price",      "close_price"),
        ("P_Long_M2",  "p_m2_long_raw"),
        ("P_Short_M2", "p_m2_short_raw"),
        # production 側に M1 raw が含まれる場合のために対応 (将来拡張用)
        ("P_Long_M1",  "p_m1_long_raw"),
        ("P_Short_M1", "p_m1_short_raw"),
    ]:
        if old in df.columns:
            rename_map[old] = new
    if rename_map:
        df = df.rename(rename_map)

    # timestamp を UTC tz-aware に
    if df["timestamp"].dtype == pl.String:
        df = df.with_columns(
            pl.col("timestamp").str.strptime(
                pl.Datetime("us", "UTC"),
                "%Y-%m-%d %H:%M:%S",
                strict=False,
            )
        )
    else:
        df = df.with_columns(
            pl.col("timestamp").cast(pl.Datetime("us", "UTC"))
        )

    logger.info(f"  ✓ loaded: rows={len(df):,}, cols={len(df.columns)}")
    return df


def load_training(path: Path) -> pl.DataFrame:
    """snapshot_training_inference.py の出力 parquet を読み込む。"""
    logger.info(f"Training snapshot: {path}")
    if not path.exists():
        raise FileNotFoundError(f"training parquet が存在しない: {path}")

    df = pl.read_parquet(path)
    df = df.with_columns(pl.col("timestamp").cast(pl.Datetime("us", "UTC")))

    logger.info(f"  ✓ loaded: rows={len(df):,}, cols={len(df.columns)}")
    return df


def filter_period(
    df: pl.DataFrame,
    start: datetime | None,
    end: datetime | None,
    name: str,
) -> pl.DataFrame:
    """期間フィルタを適用。"""
    if start is None and end is None:
        return df
    n_before = len(df)
    if start is not None:
        df = df.filter(pl.col("timestamp") >= start)
    if end is not None:
        df = df.filter(pl.col("timestamp") <= end)
    logger.info(f"  [{name}] 期間フィルタ: {n_before:,} → {len(df):,} 行")
    return df


# ════════════════════════════════════════════════════════════════
# 2. シグナル発火集合の比較
# ════════════════════════════════════════════════════════════════
def compare_signal_sets(
    prod_df: pl.DataFrame, train_df: pl.DataFrame
) -> dict:
    """timestamp で集合演算。

    Production は triggered_features_log.csv 由来なので、 記録されている行は
    全て BUY/SELL のいずれか (HOLD は記録されない仕様)。

    Training は HOLD も含む可能性があるので、 BUY/SELL のみを「発火」とみなす。
    """
    # production: 全行を「発火」 (HOLD があれば一応除外)
    if "action" in prod_df.columns:
        prod_fired = prod_df.filter(pl.col("action").is_in(["BUY", "SELL"]))
    else:
        prod_fired = prod_df  # action 列が無ければ全行発火扱い

    # training: BUY/SELL のみ
    if "action" in train_df.columns:
        train_fired = train_df.filter(pl.col("action").is_in(["BUY", "SELL"]))
    else:
        raise ValueError("training snapshot に action 列がない")

    prod_ts = set(prod_fired["timestamp"].to_list())
    train_ts = set(train_fired["timestamp"].to_list())

    both       = prod_ts & train_ts
    prod_only  = prod_ts - train_ts
    train_only = train_ts - prod_ts

    return {
        "prod_total":    len(prod_ts),
        "train_total":   len(train_ts),
        "both":          len(both),
        "prod_only":     len(prod_only),
        "train_only":    len(train_only),
        "both_ts":       sorted(both),
        "prod_only_ts":  sorted(prod_only),
        "train_only_ts": sorted(train_only),
    }


# ════════════════════════════════════════════════════════════════
# 3. 方向一致率 (両方発火の direction クロス表)
# ════════════════════════════════════════════════════════════════
def compare_directions(
    prod_df: pl.DataFrame,
    train_df: pl.DataFrame,
    both_ts: List,
) -> Dict[Tuple[str, str], int]:
    """両方発火している timestamp で direction クロス表を作る。"""
    if not both_ts:
        return {}

    prod_dict  = {row["timestamp"]: row.get("action", "?")
                  for row in prod_df.iter_rows(named=True)}
    train_dict = {row["timestamp"]: row.get("action", "?")
                  for row in train_df.iter_rows(named=True)}

    matrix: Dict[Tuple[str, str], int] = {}
    for ts in both_ts:
        p = prod_dict.get(ts, "MISSING")
        t = train_dict.get(ts, "MISSING")
        key = (p, t)
        matrix[key] = matrix.get(key, 0) + 1
    return matrix


# ════════════════════════════════════════════════════════════════
# 4. 予測値の系統的乖離
# ════════════════════════════════════════════════════════════════
def compare_predictions(
    prod_df: pl.DataFrame,
    train_df: pl.DataFrame,
    both_ts: List,
) -> List[dict]:
    """両方発火の予測値 (M1/M2 long/short) の乖離を計算。"""
    if not both_ts:
        return []

    # both_ts でフィルタ → timestamp で inner join
    pred_cols = ["p_m1_long_raw", "p_m1_short_raw",
                 "p_m2_long_raw", "p_m2_short_raw"]
    prod_pred_cols  = [c for c in pred_cols if c in prod_df.columns]
    train_pred_cols = [c for c in pred_cols if c in train_df.columns]
    common_pred = [c for c in pred_cols if c in prod_pred_cols and c in train_pred_cols]

    if not common_pred:
        logger.warning("  predictions に共通カラムなし — production CSV に M1 raw が含まれていない可能性")
        # M2 だけでも比較
        common_pred = [c for c in ["p_m2_long_raw", "p_m2_short_raw"]
                       if c in prod_df.columns and c in train_df.columns]

    if not common_pred:
        logger.warning("  予測値比較不能")
        return []

    prod_sub  = prod_df.filter(pl.col("timestamp").is_in(both_ts)).select(
        ["timestamp"] + common_pred
    )
    train_sub = train_df.filter(pl.col("timestamp").is_in(both_ts)).select(
        ["timestamp"] + common_pred
    )

    joined = prod_sub.join(
        train_sub, on="timestamp", how="inner", suffix="_train"
    )

    rows = []
    for col in common_pred:
        train_col = f"{col}_train"
        if train_col not in joined.columns:
            continue
        p = joined[col].cast(pl.Float64, strict=False).to_numpy()
        t = joined[train_col].cast(pl.Float64, strict=False).to_numpy()
        mask = np.isfinite(p) & np.isfinite(t)
        if mask.sum() == 0:
            continue
        p, t = p[mask], t[mask]
        diff = p - t
        abs_diff = np.abs(diff)
        try:
            corr = float(np.corrcoef(p, t)[0, 1]) if len(p) > 1 else None
        except Exception:
            corr = None
        rows.append({
            "metric":          col,
            "n":               int(mask.sum()),
            "mean_diff":       float(diff.mean()),
            "median_abs_diff": float(np.median(abs_diff)),
            "p99_abs_diff":    float(np.percentile(abs_diff, 99)),
            "max_abs_diff":    float(abs_diff.max()),
            "correlation":     corr,
        })
    return rows


# ════════════════════════════════════════════════════════════════
# 5. 特徴量別の系統的乖離 (全特徴量ランキング)
# ════════════════════════════════════════════════════════════════
def compare_features(
    prod_df: pl.DataFrame,
    train_df: pl.DataFrame,
    both_ts: List,
) -> pl.DataFrame:
    """両方発火の各特徴量について、 prod - train の統計を計算。

    各 feature f に対し:
      - n: 有限値ペアの数
      - mean_diff: mean(prod - train)              ← 符号付きバイアス
      - median_abs_diff: median(|prod - train|)
      - max_abs_diff: max(|prod - train|)
      - rel_diff: mean(|prod - train|) / (mean(|train|) + 1e-12)
      - correlation: corrcoef(prod, train)
    """
    if not both_ts:
        return pl.DataFrame()

    # 共通カラム (両方に存在し、 EXCLUDE に入っていない)
    common_cols = [
        c for c in prod_df.columns
        if c in train_df.columns and c not in EXCLUDE_FROM_FEATURE_COMPARE
    ]
    logger.info(f"  比較対象特徴量: {len(common_cols)} 列")

    # production にだけある列 / training にだけある列をログ出力
    prod_only_cols  = [c for c in prod_df.columns
                       if c not in train_df.columns
                       and c not in EXCLUDE_FROM_FEATURE_COMPARE]
    train_only_cols = [c for c in train_df.columns
                       if c not in prod_df.columns
                       and c not in EXCLUDE_FROM_FEATURE_COMPARE]
    if prod_only_cols:
        logger.warning(f"  production のみに存在 ({len(prod_only_cols)}): "
                       f"先頭5={prod_only_cols[:5]}")
    if train_only_cols:
        logger.warning(f"  training のみに存在 ({len(train_only_cols)}): "
                       f"先頭5={train_only_cols[:5]}")

    # both_ts でフィルタ → timestamp で inner join
    prod_sub  = prod_df.filter(pl.col("timestamp").is_in(both_ts)).select(
        ["timestamp"] + common_cols
    )
    train_sub = train_df.filter(pl.col("timestamp").is_in(both_ts)).select(
        ["timestamp"] + common_cols
    )

    joined = prod_sub.join(
        train_sub, on="timestamp", how="inner", suffix="_train"
    )
    logger.info(f"  inner join 後: {len(joined)} 行")

    rows = []
    for col in common_cols:
        train_col = f"{col}_train"
        if train_col not in joined.columns:
            continue
        try:
            p = joined[col].cast(pl.Float64, strict=False).to_numpy()
            t = joined[train_col].cast(pl.Float64, strict=False).to_numpy()
        except Exception:
            continue
        mask = np.isfinite(p) & np.isfinite(t)
        if mask.sum() == 0:
            continue
        p, t = p[mask], t[mask]
        diff = p - t
        abs_diff = np.abs(diff)
        t_abs_mean = float(np.abs(t).mean())
        rel_diff = float(abs_diff.mean()) / (t_abs_mean + 1e-12)
        try:
            corr = float(np.corrcoef(p, t)[0, 1]) if len(p) > 1 and t.std() > 0 else None
        except Exception:
            corr = None

        rows.append({
            "feature":         col,
            "n":               int(mask.sum()),
            "mean_diff":       float(diff.mean()),
            "median_abs_diff": float(np.median(abs_diff)),
            "max_abs_diff":    float(abs_diff.max()),
            "rel_diff":        rel_diff,
            "correlation":     corr,
            "train_abs_mean":  t_abs_mean,
        })

    if not rows:
        return pl.DataFrame()
    return pl.DataFrame(rows)


# ════════════════════════════════════════════════════════════════
# 6. レポート書き出し
# ════════════════════════════════════════════════════════════════
def fmt_float(x, fmt=".4f") -> str:
    if x is None:
        return "N/A"
    try:
        if not np.isfinite(x):
            return "N/A"
        return format(x, fmt)
    except (TypeError, ValueError):
        return "N/A"


def write_report(
    signal_set: dict,
    direction_matrix: Dict[Tuple[str, str], int],
    pred_rows: List[dict],
    feat_df: pl.DataFrame,
    out_dir: Path,
    top_n: int,
) -> None:
    """Markdown レポートと parquet を出力。"""
    report: List[str] = []
    report.append("# Snapshot Comparison Report")
    report.append("")
    report.append(f"Generated: {datetime.now(timezone.utc).isoformat()}")
    report.append("")

    # ─── ① シグナル発火集合 ─────────────────────────────────
    report.append("## ① シグナル発火集合の比較")
    report.append("")
    report.append(f"- Production total signals: **{signal_set['prod_total']}**")
    report.append(f"- Training (BT path) total signals: **{signal_set['train_total']}**")
    report.append(f"- **Both fired** (両方発火): {signal_set['both']}")
    report.append(f"- **Production only** (prod が余分に発火): {signal_set['prod_only']}")
    report.append(f"- **Training only** (prod が見逃し): {signal_set['train_only']}")
    if max(signal_set['prod_total'], signal_set['train_total']) > 0:
        match_rate = signal_set['both'] / max(signal_set['prod_total'], signal_set['train_total'])
        report.append(f"- **発火一致率**: {signal_set['both']}/"
                      f"{max(signal_set['prod_total'], signal_set['train_total'])} "
                      f"= **{match_rate:.1%}**")
    report.append("")

    if signal_set['prod_only_ts']:
        report.append(f"### Production only (先頭 20 件 / {len(signal_set['prod_only_ts'])} 件)")
        report.append("")
        for ts in signal_set['prod_only_ts'][:20]:
            report.append(f"  - `{ts}`")
        report.append("")

    if signal_set['train_only_ts']:
        report.append(f"### Training only (先頭 20 件 / {len(signal_set['train_only_ts'])} 件)")
        report.append("")
        for ts in signal_set['train_only_ts'][:20]:
            report.append(f"  - `{ts}`")
        report.append("")

    # ─── ② 方向一致率 ────────────────────────────────────────
    report.append("## ② 方向一致率 (両方発火のみ)")
    report.append("")
    report.append("| Prod \\ Train | BUY | SELL | HOLD | MISSING |")
    report.append("|---|---|---|---|---|")
    for prod_a in ["BUY", "SELL", "HOLD", "MISSING"]:
        cells = [prod_a]
        for train_a in ["BUY", "SELL", "HOLD", "MISSING"]:
            cnt = direction_matrix.get((prod_a, train_a), 0)
            cells.append(str(cnt))
        report.append("| " + " | ".join(cells) + " |")
    report.append("")

    total = sum(direction_matrix.values())
    if total > 0:
        match = (direction_matrix.get(("BUY", "BUY"), 0)
                 + direction_matrix.get(("SELL", "SELL"), 0)
                 + direction_matrix.get(("HOLD", "HOLD"), 0))
        flip = (direction_matrix.get(("BUY", "SELL"), 0)
                + direction_matrix.get(("SELL", "BUY"), 0))
        report.append(f"- **direction 完全一致率**: {match}/{total} = **{match/total:.1%}**")
        report.append(f"- **direction 反転率**: {flip}/{total} = **{flip/total:.1%}**")
        report.append("")

    # ─── ③ 予測値の系統的乖離 ────────────────────────────────
    report.append("## ③ 予測値の系統的乖離 (両方発火)")
    report.append("")
    if pred_rows:
        report.append("| metric | n | mean_diff | median \\|diff\\| | p99 \\|diff\\| | max \\|diff\\| | correlation |")
        report.append("|---|---|---|---|---|---|---|")
        for r in pred_rows:
            report.append(
                f"| `{r['metric']}` | {r['n']} | "
                f"{fmt_float(r['mean_diff'], '+.4f')} | "
                f"{fmt_float(r['median_abs_diff'])} | "
                f"{fmt_float(r['p99_abs_diff'])} | "
                f"{fmt_float(r['max_abs_diff'])} | "
                f"{fmt_float(r['correlation'])} |"
            )
    else:
        report.append("(比較可能な予測値カラムなし)")
    report.append("")

    # ─── ④ 特徴量別の系統的乖離 ──────────────────────────────
    report.append(f"## ④ 特徴量別の系統的乖離 TOP-{top_n} (rel_diff 降順)")
    report.append("")
    report.append("**rel_diff = mean(|prod - train|) / (mean(|train|) + 1e-12)** "
                  "— 規模感を相対化")
    report.append("")
    if len(feat_df) > 0:
        sorted_by_rel = feat_df.sort("rel_diff", descending=True).head(top_n)
        report.append("| # | feature | rel_diff | mean_diff | median \\|diff\\| | max \\|diff\\| | corr | train\\|mean\\| |")
        report.append("|---|---|---|---|---|---|---|---|")
        for i, row in enumerate(sorted_by_rel.iter_rows(named=True), 1):
            report.append(
                f"| {i} | `{row['feature']}` | "
                f"{fmt_float(row['rel_diff'])} | "
                f"{fmt_float(row['mean_diff'], '+.4f')} | "
                f"{fmt_float(row['median_abs_diff'])} | "
                f"{fmt_float(row['max_abs_diff'])} | "
                f"{fmt_float(row['correlation'])} | "
                f"{fmt_float(row['train_abs_mean'])} |"
            )
    else:
        report.append("(比較可能な特徴量カラムなし)")
    report.append("")

    # ─── ⑤ 符号ズレ ───────────────────────────────────────────
    report.append(f"## ⑤ 特徴量別の系統的乖離 TOP-{top_n} (|mean_diff| 降順 = 符号バイアス)")
    report.append("")
    report.append("**mean_diff** の符号が一様 (+ または −) なら、**production が systematic に**"
                  " **その方向にズレている** ことを示す。")
    report.append("")
    if len(feat_df) > 0:
        sorted_by_abs = feat_df.with_columns(
            pl.col("mean_diff").abs().alias("abs_mean_diff")
        ).sort("abs_mean_diff", descending=True).head(top_n)
        report.append("| # | feature | mean_diff | rel_diff | median \\|diff\\| | corr | train\\|mean\\| |")
        report.append("|---|---|---|---|---|---|---|")
        for i, row in enumerate(sorted_by_abs.iter_rows(named=True), 1):
            report.append(
                f"| {i} | `{row['feature']}` | "
                f"{fmt_float(row['mean_diff'], '+.4f')} | "
                f"{fmt_float(row['rel_diff'])} | "
                f"{fmt_float(row['median_abs_diff'])} | "
                f"{fmt_float(row['correlation'])} | "
                f"{fmt_float(row['train_abs_mean'])} |"
            )
    report.append("")

    # ─── ⑥ 相関の低い特徴量 (= 完全に変な動きをしている) ─────
    report.append(f"## ⑥ 相関が低い特徴量 TOP-{top_n} (correlation 昇順)")
    report.append("")
    report.append("**correlation が低い (≈0 or 負)** = 同じ timestamp でも全く違う値 → "
                  "計算経路が根本的に違う可能性が高い。")
    report.append("")
    if len(feat_df) > 0:
        valid_corr = feat_df.filter(pl.col("correlation").is_not_null())
        if len(valid_corr) > 0:
            sorted_by_corr = valid_corr.sort("correlation", descending=False).head(top_n)
            report.append("| # | feature | correlation | rel_diff | mean_diff | median \\|diff\\| |")
            report.append("|---|---|---|---|---|---|")
            for i, row in enumerate(sorted_by_corr.iter_rows(named=True), 1):
                report.append(
                    f"| {i} | `{row['feature']}` | "
                    f"{fmt_float(row['correlation'])} | "
                    f"{fmt_float(row['rel_diff'])} | "
                    f"{fmt_float(row['mean_diff'], '+.4f')} | "
                    f"{fmt_float(row['median_abs_diff'])} |"
                )
    report.append("")

    # 全体サマリー
    report.append("## ⑦ 全体サマリー")
    report.append("")
    if len(feat_df) > 0:
        n_perfect = int((feat_df["rel_diff"] < 1e-7).sum())
        n_minor   = int(((feat_df["rel_diff"] >= 1e-7) & (feat_df["rel_diff"] < 1e-3)).sum())
        n_moderate = int(((feat_df["rel_diff"] >= 1e-3) & (feat_df["rel_diff"] < 0.1)).sum())
        n_severe  = int((feat_df["rel_diff"] >= 0.1).sum())
        report.append(f"- 比較対象特徴量数: **{len(feat_df)}**")
        report.append(f"- bit-identical 相当 (rel_diff < 1e-7): {n_perfect}")
        report.append(f"- 軽微 (1e-7 ≤ rel_diff < 1e-3): {n_minor}")
        report.append(f"- 中程度 (1e-3 ≤ rel_diff < 0.1): {n_moderate}")
        report.append(f"- **重度 (rel_diff ≥ 0.1)**: **{n_severe}**")
        report.append("")
        if n_severe > 0:
            report.append("→ **重度に乖離している特徴量が真犯人候補**。 ④ ⑤ のランキングを参照。")
        elif n_moderate > 0:
            report.append("→ 中程度の乖離あり。 累積で予測値に効いている可能性。")
        else:
            report.append("→ 特徴量レベルではほぼ一致。 予測値乖離は他要因 (model load? dtype?) の可能性。")
        report.append("")

    # 書き出し
    report_path = out_dir / "report.md"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(report))
    logger.info(f"✅ Report: {report_path}")

    # parquet 群
    if len(feat_df) > 0:
        feat_path = out_dir / "feature_diff_summary.parquet"
        feat_df.write_parquet(feat_path, compression="zstd")
        logger.info(f"✅ Feature diff parquet: {feat_path}")

    # signal set 詳細
    signal_details = []
    for ts in signal_set['prod_only_ts']:
        signal_details.append({"timestamp": ts, "category": "prod_only"})
    for ts in signal_set['train_only_ts']:
        signal_details.append({"timestamp": ts, "category": "train_only"})
    for ts in signal_set['both_ts']:
        signal_details.append({"timestamp": ts, "category": "both"})
    if signal_details:
        sig_df = pl.DataFrame(signal_details).with_columns(
            pl.col("timestamp").cast(pl.Datetime("us", "UTC"))
        )
        sig_path = out_dir / "signal_set_details.parquet"
        sig_df.write_parquet(sig_path, compression="zstd")
        logger.info(f"✅ Signal set details: {sig_path}")


# ════════════════════════════════════════════════════════════════
# 7. main
# ════════════════════════════════════════════════════════════════
def main():
    parser = argparse.ArgumentParser(
        description="Production × Training Snapshot Comparison"
    )
    parser.add_argument("--production", required=True, type=Path,
                        help="triggered_features_log.csv のパス")
    parser.add_argument("--training",   required=True, type=Path,
                        help="snapshot_training_inference.py の出力 parquet")
    parser.add_argument("--start", default=None, help="YYYY-MM-DD (任意)")
    parser.add_argument("--end",   default=None, help="YYYY-MM-DD (任意)")
    parser.add_argument("--start-time", default="00:00:00",
                        help="HH:MM:SS UTC (default 00:00:00)")
    parser.add_argument("--end-time",   default="23:59:59",
                        help="HH:MM:SS UTC (default 23:59:59)")
    parser.add_argument("--out-dir", required=True, type=Path,
                        help="出力 dir (report.md と parquet 群)")
    parser.add_argument("--top-n", type=int, default=30,
                        help="ランキング表示数 (default: 30)")
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 72)
    logger.info("Production × Training Snapshot Comparison")
    logger.info("=" * 72)

    # ─── ロード ─────────────────────────────────────────────
    prod_df  = load_production(args.production)
    train_df = load_training(args.training)

    # ─── 期間フィルタ ────────────────────────────────────────
    start_dt = None
    end_dt = None
    if args.start:
        start_dt = datetime.strptime(
            f"{args.start} {args.start_time}", "%Y-%m-%d %H:%M:%S"
        ).replace(tzinfo=timezone.utc)
    if args.end:
        end_dt = datetime.strptime(
            f"{args.end} {args.end_time}", "%Y-%m-%d %H:%M:%S"
        ).replace(tzinfo=timezone.utc)

    prod_df  = filter_period(prod_df,  start_dt, end_dt, "prod")
    train_df = filter_period(train_df, start_dt, end_dt, "train")

    # ─── 比較 ───────────────────────────────────────────────
    logger.info("")
    logger.info("--- ① シグナル発火集合 ---")
    signal_set = compare_signal_sets(prod_df, train_df)
    logger.info(f"  prod={signal_set['prod_total']}, "
                f"train={signal_set['train_total']}, "
                f"both={signal_set['both']}, "
                f"prod_only={signal_set['prod_only']}, "
                f"train_only={signal_set['train_only']}")

    logger.info("--- ② 方向一致率 ---")
    direction_matrix = compare_directions(prod_df, train_df, signal_set['both_ts'])
    if signal_set['both'] > 0:
        match = (direction_matrix.get(("BUY", "BUY"), 0)
                 + direction_matrix.get(("SELL", "SELL"), 0))
        flip = (direction_matrix.get(("BUY", "SELL"), 0)
                + direction_matrix.get(("SELL", "BUY"), 0))
        logger.info(f"  完全一致: {match}/{signal_set['both']}, 反転: {flip}/{signal_set['both']}")

    logger.info("--- ③ 予測値乖離 ---")
    pred_rows = compare_predictions(prod_df, train_df, signal_set['both_ts'])
    for r in pred_rows:
        logger.info(f"  {r['metric']}: mean_diff={r['mean_diff']:+.4f}, "
                    f"max|diff|={r['max_abs_diff']:.4f}, corr={r['correlation']}")

    logger.info("--- ④ 特徴量乖離 ---")
    feat_df = compare_features(prod_df, train_df, signal_set['both_ts'])
    if len(feat_df) > 0:
        n_severe = int((feat_df["rel_diff"] >= 0.1).sum())
        logger.info(f"  比較特徴量 {len(feat_df)} 列, 重度 (rel_diff≥0.1): {n_severe} 列")

    # ─── レポート出力 ────────────────────────────────────────
    write_report(signal_set, direction_matrix, pred_rows, feat_df, args.out_dir, args.top_n)

    logger.info("")
    logger.info("=" * 72)
    logger.info(f"✅ 完了: {args.out_dir}/report.md を確認")
    logger.info("=" * 72)


if __name__ == "__main__":
    main()
