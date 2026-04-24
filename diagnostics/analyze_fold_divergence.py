# analyze_fold_divergence.py
# =====================================================================
# Fold時系列位置 vs OOF乖離 相関分析スクリプト
#
# 【目的】
#   OOFファイルのタイムスタンプからfoldインデックスを逆算し、
#   「foldの時系列位置」と「OOF予測値 vs 最終モデル再現予測値の乖離」の
#   相関を定量的に検証する。
#
#   これにより「古いfoldほど乖離大・新しいfoldほど乖離小」という
#   Geminiレポートの仮説が成立するかどうかを実証する。
#
# 【使い方】
#   python analyze_fold_divergence.py
#   python analyze_fold_divergence.py --direction long
#   python analyze_fold_divergence.py --n_samples 500
#
# 【出力】
#   コンソールにfold別・分位別の乖離統計を表示
#   /workspace/data/diagnostics/fold_divergence/
#     fold_divergence_{direction}.csv
#     summary_{direction}.txt
# =====================================================================

import sys
import argparse
import warnings
import logging
import numpy as np
import pandas as pd
import polars as pl
import joblib
from pathlib import Path
from typing import List, Dict

warnings.filterwarnings("ignore")

logging.root.setLevel(logging.ERROR)
_handler = logging.StreamHandler()
_handler.setLevel(logging.INFO)
_handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
logger = logging.getLogger("FoldDivergence")
logger.setLevel(logging.INFO)
logger.addHandler(_handler)
logger.propagate = False

project_root = Path(__file__).resolve().parents[1]
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

from blueprint import (
    S6_WEIGHTED_DATASET,
    S7_M1_OOF_PREDICTIONS_LONG,
    S7_M1_OOF_PREDICTIONS_SHORT,
    S7_M2_OOF_PREDICTIONS_LONG,
    S7_M2_OOF_PREDICTIONS_SHORT,
    S7_M1_MODEL_LONG_PKL,
    S7_M1_MODEL_SHORT_PKL,
    S7_M2_MODEL_LONG_PKL,
    S7_M2_MODEL_SHORT_PKL,
    S3_SELECTED_FEATURES_ORTHOGONAL_DIR,
)

OUTPUT_DIR = Path("/workspace/data/diagnostics/fold_divergence")

# Ax2と同じfold設定
K_FOLDS = 5
PURGE_DAYS = 3
EMBARGO_DAYS = 2


def load_feature_list(filepath: Path) -> List[str]:
    with open(filepath) as f:
        return [l.strip() for l in f if l.strip() and not l.startswith("is_trigger")]


def assign_fold_index(timestamps: pd.Series, k_folds: int = 5) -> pd.Series:
    """
    タイムスタンプからfoldインデックスを逆算する。
    Ax2のPartitionPurgedKFoldと同じロジック：
    全パーティション（日付）を時系列順にk_folds等分して割り当てる。

    戻り値：
        fold_index (1始まり): どのfoldのtest setに属するか
        fold_position (0.0〜1.0): 全期間における時系列的な相対位置
    """
    dates = pd.to_datetime(timestamps).dt.date
    all_dates = sorted(dates.unique())
    n_partitions = len(all_dates)
    fold_size = n_partitions // k_folds

    date_to_fold = {}
    for i in range(k_folds):
        start = i * fold_size
        end = start + fold_size if i < k_folds - 1 else n_partitions
        for d in all_dates[start:end]:
            date_to_fold[d] = i + 1  # 1始まり

    fold_indices = dates.map(date_to_fold).fillna(-1).astype(int)

    # fold_position: そのfoldの中央日付の全期間に対する相対位置 (0.0〜1.0)
    min_date = pd.Timestamp(all_dates[0])
    max_date = pd.Timestamp(all_dates[-1])
    total_days = (max_date - min_date).days or 1
    fold_positions = (pd.to_datetime(dates).apply(
        lambda d: (pd.Timestamp(d) - min_date).days
    ) / total_days)

    return fold_indices, fold_positions


def compute_logit(p: np.ndarray, clip_min: float = 1e-7, clip_max: float = None) -> np.ndarray:
    if clip_max is None:
        clip_max = 1 - clip_min
    p_clipped = np.clip(p, clip_min, clip_max)
    logits = np.log(p_clipped / (1 - p_clipped))
    return np.clip(logits, -10.0, 10.0)


def run_analysis(
    direction: str = "long",
    n_samples: int = 1000,
    oof_level: str = "m2",  # "m1" or "m2"
) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    logger.info(f"=== Fold乖離分析開始: direction={direction}, n={n_samples}, oof={oof_level} ===")

    # --- モデル・特徴量リスト読み込み ---
    if oof_level == "m2":
        oof_path = S7_M2_OOF_PREDICTIONS_LONG if direction == "long" else S7_M2_OOF_PREDICTIONS_SHORT
        m1_pkl = S7_M1_MODEL_LONG_PKL if direction == "long" else S7_M1_MODEL_SHORT_PKL
        m2_pkl = S7_M2_MODEL_LONG_PKL if direction == "long" else S7_M2_MODEL_SHORT_PKL
    else:
        oof_path = S7_M1_OOF_PREDICTIONS_LONG if direction == "long" else S7_M1_OOF_PREDICTIONS_SHORT
        m1_pkl = S7_M1_MODEL_LONG_PKL if direction == "long" else S7_M1_MODEL_SHORT_PKL
        m2_pkl = None

    logger.info("モデル読み込み中...")
    m1_model = joblib.load(m1_pkl)
    m2_model = joblib.load(m2_pkl) if m2_pkl else None

    m1_features = load_feature_list(
        S3_SELECTED_FEATURES_ORTHOGONAL_DIR / f"m1_{direction}_features.txt"
    )
    m2_features_raw = load_feature_list(
        S3_SELECTED_FEATURES_ORTHOGONAL_DIR / f"m2_{direction}_features.txt"
    )
    m2_features = [f for f in m2_features_raw if f != "m1_pred_proba"]
    m2_features.append("m1_pred_proba")

    # --- OOFファイル読み込み ---
    logger.info(f"OOFファイル読み込み: {oof_path.name}")
    oof_df = pl.read_parquet(oof_path)
    logger.info(f"OOF総件数: {len(oof_df)}")

    # サンプリング（全件 or n_samples件）
    if n_samples > 0 and n_samples < len(oof_df):
        # 時系列順にサンプリング（均等間隔）
        step = len(oof_df) // n_samples
        oof_sample = oof_df[::step].head(n_samples)
    else:
        oof_sample = oof_df

    logger.info(f"分析サンプル数: {len(oof_sample)}")

    sample_timestamps = oof_sample["timestamp"].to_list()
    oof_predictions = oof_sample["prediction"].to_list()

    # --- foldインデックス逆算 ---
    ts_series = pd.Series([
        pd.Timestamp(ts) if not isinstance(ts, pd.Timestamp) else ts
        for ts in sample_timestamps
    ])
    fold_indices, fold_positions = assign_fold_index(ts_series, K_FOLDS)

    # --- S6から特徴量取得 ---
    logger.info("S6データ読み込み中...")
    all_features_needed = list(set(m1_features + m2_features) - {"m1_pred_proba"})
    ts_set = set(sample_timestamps)

    s6_lf = pl.scan_parquet(str(S6_WEIGHTED_DATASET / "**/*.parquet"))
    s6_df = (
        s6_lf
        .filter(pl.col("timestamp").is_in(list(ts_set)))
        .select(["timestamp"] + [f for f in all_features_needed
                                   if f != "atr_ratio_M3"])
        .collect()
    )
    logger.info(f"S6データ取得: {len(s6_df)}件")

    s6_dict = {row["timestamp"]: row for row in s6_df.iter_rows(named=True)}

    # --- 各サンプルで再現推論 ---
    results = []
    skipped = 0

    for i, (ts, oof_pred) in enumerate(zip(sample_timestamps, oof_predictions)):
        if ts not in s6_dict:
            skipped += 1
            continue

        row = s6_dict[ts]

        # M1推論
        X_m1 = np.array([[row.get(f, 0.0) or 0.0 for f in m1_features]])
        p_m1 = float(m1_model.predict(X_m1)[0])

        if oof_level == "m2" and m2_model is not None:
            if p_m1 >= 0.50:
                p_clipped = np.clip(p_m1, 1e-7, 1 - 1e-7)
                logit_m1 = float(np.clip(np.log(p_clipped / (1 - p_clipped)), -10.0, 10.0))
                fd_m2 = {f: (row.get(f, 0.0) or 0.0) for f in m2_features if f != "m1_pred_proba"}
                fd_m2["m1_pred_proba"] = logit_m1
                X_m2 = np.array([[fd_m2.get(f, 0.0) for f in m2_features]])
                p_reprod = float(m2_model.predict(X_m2)[0])
            else:
                p_reprod = 0.0
        else:
            p_reprod = p_m1

        # ロジット差分計算
        logit_oof   = float(compute_logit(np.array([oof_pred]))[0])
        logit_reprod = float(compute_logit(np.array([p_reprod]))[0])
        logit_diff = logit_reprod - logit_oof  # 正=本番が過信、負=OOFが過信

        abs_diff = abs(oof_pred - p_reprod)
        rel_diff = abs_diff / (abs(oof_pred) + 1e-10)

        results.append({
            "timestamp":      str(ts),
            "fold_index":     int(fold_indices.iloc[i]),
            "fold_position":  round(float(fold_positions.iloc[i]), 4),
            "oof_pred":       round(oof_pred, 6),
            "reprod_pred":    round(p_reprod, 6),
            "abs_diff":       round(abs_diff, 6),
            "rel_diff_%":     round(rel_diff * 100, 2),
            "logit_oof":      round(logit_oof, 4),
            "logit_reprod":   round(logit_reprod, 4),
            "logit_diff":     round(logit_diff, 4),  # M3の目的変数候補
        })

    logger.info(f"有効サンプル: {len(results)}件 / スキップ: {skipped}件")

    if not results:
        logger.error("有効結果が0件です。")
        return

    df = pd.DataFrame(results)

    # --- CSV保存 ---
    csv_path = OUTPUT_DIR / f"fold_divergence_{direction}.csv"
    df.to_csv(csv_path, index=False)
    logger.info(f"CSV保存: {csv_path}")

    # --- 統計分析 ---
    abs_diffs = df["abs_diff"].values
    logit_diffs = df["logit_diff"].values
    fold_positions_arr = df["fold_position"].values
    fold_indices_arr = df["fold_index"].values

    # fold位置 vs 乖離の相関
    corr_pos_abs   = np.corrcoef(fold_positions_arr, abs_diffs)[0, 1]
    corr_pos_logit = np.corrcoef(fold_positions_arr, logit_diffs)[0, 1]
    corr_pos_abs_logit = np.corrcoef(fold_positions_arr, np.abs(logit_diffs))[0, 1]

    # fold別統計
    fold_stats = df.groupby("fold_index").agg(
        count=("abs_diff", "count"),
        mean_abs_diff=("abs_diff", "mean"),
        mean_logit_diff=("logit_diff", "mean"),
        std_logit_diff=("logit_diff", "std"),
        mean_fold_position=("fold_position", "mean"),
    ).reset_index()

    # 全体統計
    mae = np.mean(abs_diffs)
    rmse = np.sqrt(np.mean((df["oof_pred"].values - df["reprod_pred"].values) ** 2))
    corr_oof_reprod = np.corrcoef(df["oof_pred"].values, df["reprod_pred"].values)[0, 1]

    # --- レポート出力 ---
    lines = [
        "=" * 70,
        f"  Fold乖離分析レポート ({direction.upper()} / {oof_level.upper()})",
        "=" * 70,
        f"  分析サンプル数: {len(df)}件（スキップ: {skipped}件）",
        "",
        "--- 全体統計 ---",
        f"  MAE              : {mae:.6f}",
        f"  RMSE             : {rmse:.6f}",
        f"  OOF vs 再現 相関 : {corr_oof_reprod:.6f}",
        "",
        "--- Fold位置 vs 乖離 相関 ---",
        f"  fold_position vs abs_diff        : {corr_pos_abs:+.4f}",
        f"  fold_position vs logit_diff      : {corr_pos_logit:+.4f}  ← 正=新しいfoldほど本番が過信",
        f"  fold_position vs |logit_diff|    : {corr_pos_abs_logit:+.4f}  ← 正=新しいfoldほど乖離大",
        "",
        "  【解釈】",
        "  相関が負（古いfold=乖離大）→ Gemini仮説を支持",
        "  相関が正（新しいfold=乖離大）→ 逆仮説を支持",
        "  相関が0付近 → Fold位置は乖離と無関係",
        "",
        "--- Fold別 平均乖離 ---",
        f"  {'Fold':>5} {'件数':>6} {'avg abs_diff':>14} {'avg logit_diff':>16} {'fold中央位置':>14}",
        "  " + "-" * 60,
    ]

    for _, row in fold_stats.iterrows():
        lines.append(
            f"  Fold{int(row['fold_index']):>1}  "
            f"{int(row['count']):>6}  "
            f"{row['mean_abs_diff']:>14.6f}  "
            f"{row['mean_logit_diff']:>+16.4f}  "
            f"{row['mean_fold_position']:>14.3f}"
        )

    lines += [
        "",
        "--- 分位別 平均乖離（fold_position 四分位） ---",
    ]

    df["position_quartile"] = pd.qcut(
        df["fold_position"], q=4,
        labels=["Q1(古)", "Q2", "Q3", "Q4(新)"]
    )
    quartile_stats = df.groupby("position_quartile", observed=True).agg(
        count=("abs_diff", "count"),
        mean_abs_diff=("abs_diff", "mean"),
        mean_logit_diff=("logit_diff", "mean"),
    )
    for q, row in quartile_stats.iterrows():
        lines.append(
            f"  {str(q):<10}  件数={int(row['count']):>5}  "
            f"avg abs_diff={row['mean_abs_diff']:.6f}  "
            f"avg logit_diff={row['mean_logit_diff']:+.4f}"
        )

    lines.append("=" * 70)

    report = "\n".join(lines)
    print("\n" + report)

    txt_path = OUTPUT_DIR / f"summary_{direction}.txt"
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(report + "\n")
    logger.info(f"サマリー保存: {txt_path}")


def main():
    parser = argparse.ArgumentParser(description="Fold位置 vs OOF乖離 相関分析")
    parser.add_argument("--direction", default="long", choices=["long", "short", "both"])
    parser.add_argument("--n_samples", default=1000, type=int,
                        help="分析サンプル数（0=全件）")
    parser.add_argument("--oof_level", default="m2", choices=["m1", "m2"])
    args = parser.parse_args()

    directions = ["long", "short"] if args.direction == "both" else [args.direction]
    for d in directions:
        run_analysis(
            direction=d,
            n_samples=args.n_samples,
            oof_level=args.oof_level,
        )


if __name__ == "__main__":
    main()
