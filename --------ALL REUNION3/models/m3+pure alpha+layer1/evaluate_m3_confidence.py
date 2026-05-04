# /workspace/models/evaluate_m3_confidence.py
# =====================================================================
# M3 Confidence Layer 性能評価スクリプト
#
# 【目的】
#   train_m3_confidence_layer.py が生成したOOFスコアを用いて
#   M3の信頼度スコアが実際の予測精度と一致しているかを検証する。
#
# 【確認内容】
#   1. 信頼度スコア別の実際の外れ率（スコア高=外れ少ないか）
#   2. キャリブレーション曲線（スコアが確率として正確か）
#   3. 閾値（0.70/0.40）適用時のシグナル数・外れ率変化
#   4. 特徴量別の分析（atr_ratio・hour_utc・weekday別）
#
# 【入力】
#   /workspace/data/XAUUSD/stratum_7_models/
#     m3_oof_scores_long.parquet
#     m3_oof_scores_short.parquet
#
# 【出力】
#   /workspace/data/diagnostics/m3_evaluation/
#     calibration_curve_{direction}.png
#     score_vs_accuracy_{direction}.csv
#     summary_{direction}.txt
#
# 【使い方】
#   python evaluate_m3_confidence.py                   # long・short両方
#   python evaluate_m3_confidence.py --direction long  # longのみ
# =====================================================================

import sys
import argparse
import logging
import warnings
from pathlib import Path

import numpy as np
import polars as pl
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    average_precision_score,
    roc_auc_score,
    log_loss,
)

warnings.filterwarnings("ignore")

project_root = Path(__file__).resolve().parents[1]
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

from blueprint import (
    S7_MODELS,
    M3_OOF_SCORES_LONG,
    M3_OOF_SCORES_SHORT,
    M3_EVALUATION_DIR,
)

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("EvalM3")

INPUT_DIR  = S7_MODELS
OUTPUT_DIR = M3_EVALUATION_DIR

# 本番での閾値設定
THRESHOLD_FULL = 0.70  # これ以上 → 見送り（外れやすい・危険）
THRESHOLD_HALF = 0.40  # 未満 → フルロット / 以上 → ハーフロット


def evaluate(direction: str) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    oof_path = (M3_OOF_SCORES_LONG if direction == "long"
                else M3_OOF_SCORES_SHORT)
    if not oof_path.exists():
        logger.error(
            f"OOFスコアが見つかりません: {oof_path}\n"
            f"先に train_m3_confidence_layer.py を実行してください。"
        )
        return

    logger.info(f"\n{'='*60}")
    logger.info(f"M3評価開始: direction={direction.upper()}")
    logger.info(f"{'='*60}")

    df = pl.read_parquet(oof_path).to_pandas()
    logger.info(f"OOFスコア件数: {len(df)}")

    scores    = df["m3_score"].values
    is_wrong  = df["is_wrong"].values.astype(int)
    true_label = df["true_label"].values.astype(int)

    # =====================================================
    # 1. 全体指標
    # =====================================================
    pr_auc  = average_precision_score(is_wrong, scores)
    roc_auc = roc_auc_score(is_wrong, scores)
    ll      = log_loss(is_wrong, scores)
    base_wrong_rate = is_wrong.mean()

    # =====================================================
    # 2. 信頼度スコア別の外れ率
    #    スコアを10分位に分けて各分位の実際の外れ率を確認
    # =====================================================
    df["score_decile"] = pd.qcut(
        df["m3_score"], q=10,
        labels=[f"D{i+1}" for i in range(10)],
        duplicates="drop",
    )
    decile_stats = df.groupby("score_decile", observed=True).agg(
        count=("is_wrong", "count"),
        wrong_rate=("is_wrong", "mean"),
        avg_score=("m3_score", "mean"),
        avg_p_m2=("p_m2", "mean"),
    ).reset_index()

    # CSV保存
    csv_path = OUTPUT_DIR / f"score_vs_accuracy_{direction}.csv"
    decile_stats.to_csv(csv_path, index=False)
    logger.info(f"スコア別精度CSV保存: {csv_path}")

    # =====================================================
    # 3. 閾値適用時の効果
    # =====================================================
    mask_full = scores < THRESHOLD_HALF                              # フルロット（安全）
    mask_half = (scores >= THRESHOLD_HALF) & (scores < THRESHOLD_FULL)  # ハーフロット
    mask_skip = scores >= THRESHOLD_FULL                             # 見送り（危険）

    def stats(mask):
        if mask.sum() == 0:
            return {"count": 0, "wrong_rate": 0.0, "pct": 0.0}
        return {
            "count":      int(mask.sum()),
            "wrong_rate": float(is_wrong[mask].mean()),
            "pct":        float(mask.mean() * 100),
        }

    s_full = stats(mask_full)
    s_half = stats(mask_half)
    s_skip = stats(mask_skip)

    # =====================================================
    # 4. キャリブレーション曲線
    # =====================================================
    try:
        n_bins = min(10, int(len(df) / 50))
        n_bins = max(5, n_bins)
        prob_true, prob_pred = calibration_curve(
            is_wrong, scores, n_bins=n_bins, strategy="quantile"
        )

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # キャリブレーション曲線
        ax1 = axes[0]
        ax1.plot([0, 1], [0, 1], "k--", label="完璧なキャリブレーション")
        ax1.plot(prob_pred, prob_true, "bo-", label="M3スコア")
        ax1.set_xlabel("M3信頼度スコア（予測確率）")
        ax1.set_ylabel("実際の外れ率")
        ax1.set_title(
            f"M3 キャリブレーション曲線 ({direction.upper()})\n"
            f"PR-AUC={pr_auc:.3f} / ROC-AUC={roc_auc:.3f}"
        )
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # スコア分布
        ax2 = axes[1]
        ax2.hist(scores[is_wrong == 0], bins=30, alpha=0.6,
                 color="steelblue", label="正解（is_wrong=0）", density=True)
        ax2.hist(scores[is_wrong == 1], bins=30, alpha=0.6,
                 color="tomato", label="外れ（is_wrong=1）", density=True)
        ax2.axvline(THRESHOLD_FULL, color="red", linestyle="--",
                    label=f"見送り閾値 ({THRESHOLD_FULL})")
        ax2.axvline(THRESHOLD_HALF, color="green", linestyle="--",
                    label=f"フルロット閾値 ({THRESHOLD_HALF})")
        ax2.set_xlabel("M3信頼度スコア")
        ax2.set_ylabel("密度")
        ax2.set_title(f"スコア分布 ({direction.upper()})")
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        fig_path = OUTPUT_DIR / f"calibration_curve_{direction}.png"
        plt.savefig(fig_path, dpi=120)
        plt.close(fig)
        logger.info(f"キャリブレーション曲線保存: {fig_path}")
    except Exception as e:
        logger.warning(f"キャリブレーション曲線の生成に失敗: {e}")

    # =====================================================
    # 5. 特徴量別分析
    # =====================================================
    # 時間帯別
    if "hour_utc" in df.columns:
        hour_stats = df.groupby("hour_utc").agg(
            count=("is_wrong", "count"),
            wrong_rate=("is_wrong", "mean"),
            avg_score=("m3_score", "mean"),
        ).sort_values("hour_utc")
    else:
        hour_stats = pd.DataFrame()

    # 曜日別
    if "weekday" in df.columns:
        wd_names = {0:"Mon", 1:"Tue", 2:"Wed", 3:"Thu",
                    4:"Fri", 5:"Sat", 6:"Sun"}
        weekday_stats = df.groupby("weekday").agg(
            count=("is_wrong", "count"),
            wrong_rate=("is_wrong", "mean"),
            avg_score=("m3_score", "mean"),
        ).sort_values("weekday")
        weekday_stats["weekday_name"] = weekday_stats.index.map(wd_names)
    else:
        weekday_stats = pd.DataFrame()

    # =====================================================
    # 6. レポート出力
    # =====================================================
    lines = [
        "=" * 65,
        f"  M3 Confidence Layer 評価レポート ({direction.upper()})",
        "=" * 65,
        f"  評価件数: {len(df)}件（Fold5）",
        f"  ベースライン外れ率: {base_wrong_rate*100:.2f}%",
        "",
        "--- 全体指標 ---",
        f"  PR-AUC  : {pr_auc:.4f}  ← 主要指標（不均衡対応）",
        f"  ROC-AUC : {roc_auc:.4f}",
        f"  Logloss : {ll:.4f}",
        "",
        "--- 閾値適用時の効果 ---",
        f"  {'区分':<20} {'件数':>8} {'割合%':>8} {'外れ率%':>10}",
        "  " + "-" * 50,
        f"  {'フルロット (<0.40)':<20} "
        f"{s_full['count']:>8} {s_full['pct']:>7.1f}% "
        f"{s_full['wrong_rate']*100:>9.2f}%",
        f"  {'ハーフ (0.40〜0.70)':<20} "
        f"{s_half['count']:>8} {s_half['pct']:>7.1f}% "
        f"{s_half['wrong_rate']*100:>9.2f}%",
        f"  {'見送り (>=0.70)':<20} "
        f"{s_skip['count']:>8} {s_skip['pct']:>7.1f}% "
        f"{s_skip['wrong_rate']*100:>9.2f}%",
        f"  {'全体（フィルターなし）':<20} "
        f"{len(df):>8} {'100.0':>7}% "
        f"{base_wrong_rate*100:>9.2f}%",
        "",
        "  【判定基準】",
        "  フルロット(<0.40)の外れ率 < ベースラインの外れ率 → M3が機能している",
        "  見送り(>=0.70)の外れ率 > ベースラインの外れ率  → M3が危険を正しく検知",
        "",
        "--- スコア10分位別 外れ率 ---",
        f"  {'分位':>6} {'件数':>6} {'avg_score':>10} {'外れ率%':>10}",
        "  " + "-" * 38,
    ]
    for _, row in decile_stats.iterrows():
        lines.append(
            f"  {str(row['score_decile']):>6}  "
            f"{int(row['count']):>6}  "
            f"{row['avg_score']:>10.4f}  "
            f"{row['wrong_rate']*100:>9.2f}%"
        )

    if not weekday_stats.empty:
        lines += ["", "--- 曜日別 平均外れ率 ---"]
        for _, row in weekday_stats.iterrows():
            lines.append(
                f"  {row['weekday_name']:<5}: "
                f"外れ率={row['wrong_rate']*100:.2f}%  "
                f"avg_score={row['avg_score']:.4f}  "
                f"件数={int(row['count'])}"
            )

    if not hour_stats.empty:
        lines += ["", "--- 時間帯別 平均外れ率（上位5・下位5） ---"]
        h_sorted_asc = hour_stats.sort_values("wrong_rate")
        h_sorted_desc = hour_stats.sort_values("wrong_rate", ascending=False)
        lines.append("  外れ率が低い時間帯（M3が有効な時間帯）:")
        for _, row in h_sorted_asc.head(5).iterrows():
            lines.append(
                f"    {int(row.name):02d}:00 JST: "
                f"外れ率={row['wrong_rate']*100:.2f}%  "
                f"件数={int(row['count'])}"
            )
        lines.append("  外れ率が高い時間帯（注意が必要な時間帯）:")
        for _, row in h_sorted_desc.head(5).iterrows():
            lines.append(
                f"    {int(row.name):02d}:00 JST: "
                f"外れ率={row['wrong_rate']*100:.2f}%  "
                f"件数={int(row['count'])}"
            )

    lines += [
        "",
        "--- 出力ファイル ---",
        f"  キャリブレーション曲線: calibration_curve_{direction}.png",
        f"  スコア別精度CSV:        score_vs_accuracy_{direction}.csv",
        "=" * 65,
    ]

    report = "\n".join(lines)
    print("\n" + report)

    txt_path = OUTPUT_DIR / f"summary_{direction}.txt"
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(report + "\n")
    logger.info(f"サマリー保存: {txt_path}")


def main():
    parser = argparse.ArgumentParser(
        description="M3 Confidence Layer 性能評価スクリプト"
    )
    parser.add_argument("--direction", default="both",
                        choices=["long", "short", "both"])
    args = parser.parse_args()

    directions = (["long", "short"] if args.direction == "both"
                  else [args.direction])
    for d in directions:
        evaluate(d)


if __name__ == "__main__":
    main()
