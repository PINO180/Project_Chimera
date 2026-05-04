# reproducibility_verifier.py
# =====================================================================
# ツール③: 再現性検証ツール（S6データ直接読み方式）
#
# 【目的】
#   S6データ（学習時の特徴量付きデータ）をM1/M2モデルに直接入力して
#   予測値を再現し、OOFファイルのM2予測値と比較する。
#   「モデルへの入力順序・欠損値処理・logit変換が正しいか」を数値で確認する。
#
# 【注意】
#   このツールはrealtime_feature_engineを使わない。
#   「学習時の特徴量→モデル→予測値」の再現性を検証するものであり
#   「本番の特徴量生成が正しいか」はツール①（スナップショット）で確認する。
#
# 【使い方】
#   python reproducibility_verifier.py                        # long・short両方（デフォルト）
#   python reproducibility_verifier.py --direction long       # longのみ
#   python reproducibility_verifier.py --direction short      # shortのみ
#   python reproducibility_verifier.py --n_samples 100        # サンプル数変更
#   python reproducibility_verifier.py --min_proba 0.60       # 閾値変更
#
# 【出力】
#   /workspace/data/diagnostics/reproducibility/
#     comparison_{direction}.csv
#     summary_report_{direction}.txt
# =====================================================================

import sys
import logging
import argparse
import warnings
import numpy as np
import pandas as pd
import polars as pl
import joblib
from pathlib import Path
from typing import Dict, List

warnings.filterwarnings("ignore")

# Verifier専用ロガー（他のロガーは全て抑制）
logging.root.setLevel(logging.ERROR)
_handler = logging.StreamHandler()
_handler.setLevel(logging.INFO)
_handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
logger = logging.getLogger("Verifier")
logger.setLevel(logging.INFO)
logger.addHandler(_handler)
logger.propagate = False

project_root = Path(__file__).resolve().parents[1]
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

import blueprint as config
from blueprint import (
    S6_WEIGHTED_DATASET,
    S7_M1_OOF_PREDICTIONS_LONG, S7_M1_OOF_PREDICTIONS_SHORT,
    S7_M2_OOF_PREDICTIONS_LONG, S7_M2_OOF_PREDICTIONS_SHORT,
    S7_M1_MODEL_LONG_PKL, S7_M1_MODEL_SHORT_PKL,
    S7_M2_MODEL_LONG_PKL, S7_M2_MODEL_SHORT_PKL,
    S3_SELECTED_FEATURES_ORTHOGONAL_DIR,
    S7_MODELS,
)


def load_feature_list(filepath: Path) -> List[str]:
    with open(filepath) as f:
        return [l.strip() for l in f if l.strip() and not l.startswith("is_trigger")]


def verify(
    direction: str = "long",
    n_samples: int = 50,
    min_proba: float = 0.70,
    output_dir: Path = None,
) -> None:
    if output_dir is None:
        output_dir = Path("/workspace/data/diagnostics/reproducibility")
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"=== 再現性検証開始: direction={direction}, n={n_samples}, min_proba={min_proba} ===")

    # --- モデル・特徴量リスト読み込み ---
    logger.info("モデル読み込み中...")
    models = {
        "long_m1":  joblib.load(S7_M1_MODEL_LONG_PKL),
        "long_m2":  joblib.load(S7_M2_MODEL_LONG_PKL),
        "short_m1": joblib.load(S7_M1_MODEL_SHORT_PKL),
        "short_m2": joblib.load(S7_M2_MODEL_SHORT_PKL),
    }

    # main.pyと同じ：m1_pred_probaをM2リストから除去して末尾にappend
    feature_lists = {
        "long_m1":  load_feature_list(S3_SELECTED_FEATURES_ORTHOGONAL_DIR / "m1_long_features.txt"),
        "long_m2":  load_feature_list(S3_SELECTED_FEATURES_ORTHOGONAL_DIR / "m2_long_features.txt"),
        "short_m1": load_feature_list(S3_SELECTED_FEATURES_ORTHOGONAL_DIR / "m1_short_features.txt"),
        "short_m2": load_feature_list(S3_SELECTED_FEATURES_ORTHOGONAL_DIR / "m2_short_features.txt"),
    }
    for key in ["long_m2", "short_m2"]:
        fl = feature_lists[key]
        if "m1_pred_proba" in fl:
            fl.remove("m1_pred_proba")
        fl.append("m1_pred_proba")

    # --- OOFファイルからサンプル選択 ---
    oof_path = S7_M2_OOF_PREDICTIONS_LONG if direction == "long" else S7_M2_OOF_PREDICTIONS_SHORT
    logger.info(f"OOFファイル読み込み: {oof_path.name}")
    oof_df = pl.read_parquet(oof_path)
    logger.info(f"OOF総件数: {len(oof_df)}")

    oof_high = oof_df.filter(pl.col("prediction") >= min_proba)
    logger.info(f"prediction >= {min_proba}: {len(oof_high)}件")

    if len(oof_high) == 0:
        logger.error("サンプルが0件です。min_probaを下げてください。")
        return

    n_actual = min(n_samples, len(oof_high))
    oof_sample = oof_high.sample(n_actual, seed=42, shuffle=True)
    logger.info(f"検証サンプル: {n_actual}件")

    sample_timestamps = oof_sample["timestamp"].to_list()
    oof_predictions = oof_sample["prediction"].to_list()

    # --- S6データから該当タイムスタンプの特徴量を取得 ---
    logger.info("S6データ読み込み中...")

    m1_features = feature_lists[f"{direction}_m1"]
    m2_features_no_m1 = [f for f in feature_lists[f"{direction}_m2"] if f != "m1_pred_proba"]
    all_features_needed = list(set(m1_features + m2_features_no_m1))

    # S6データをスキャン（timestampでフィルター）
    ts_set = set(sample_timestamps)
    try:
        s6_lf = pl.scan_parquet(str(S6_WEIGHTED_DATASET / "**/*.parquet"))
        s6_df = (
            s6_lf
            .filter(pl.col("timestamp").is_in(list(ts_set)))
            .select(["timestamp"] + [f for f in all_features_needed if f != "atr_ratio_M3"])
            .collect()
        )
    except Exception as e:
        logger.error(f"S6データ読み込み失敗: {e}")
        return

    logger.info(f"S6データ取得: {len(s6_df)}件 / {len(all_features_needed)}特徴量")

    # timestampをキーにした辞書に変換
    s6_dict = {}
    for row in s6_df.iter_rows(named=True):
        s6_dict[row["timestamp"]] = row

    # --- 各サンプルで推論 ---
    results = []
    skipped = 0

    for i, (ts, oof_pred) in enumerate(zip(sample_timestamps, oof_predictions)):
        logger.info(f"[{i+1}/{n_actual}] 検証中: {ts} (OOF={oof_pred:.4f})")

        if ts not in s6_dict:
            logger.warning(f"  S6データなし → スキップ")
            skipped += 1
            continue

        row = s6_dict[ts]

        # M1推論
        X_m1 = np.array([[row.get(f, 0.0) or 0.0 for f in feature_lists[f"{direction}_m1"]]])
        p_m1 = float(models[f"{direction}_m1"].predict(X_m1)[0])

        # M2推論（M1 >= 0.50のみ）
        if p_m1 >= 0.50:
            p_clipped = np.clip(p_m1, 1e-7, 1 - 1e-7)
            logit = float(np.clip(np.log(p_clipped / (1 - p_clipped)), -10.0, 10.0))
            fd_m2 = {f: (row.get(f, 0.0) or 0.0) for f in m2_features_no_m1}
            fd_m2["m1_pred_proba"] = logit
            X_m2 = np.array([[fd_m2.get(f, 0.0) for f in feature_lists[f"{direction}_m2"]]])
            p_m2 = float(models[f"{direction}_m2"].predict(X_m2)[0])
        else:
            p_m2 = 0.0

        diff = p_m2 - oof_pred
        rel_diff = abs(diff) / (abs(oof_pred) + 1e-10)

        logger.info(f"  M1={p_m1:.4f} → M2={p_m2:.4f} (OOF={oof_pred:.4f}, diff={diff:+.4f})")

        results.append({
            "timestamp":   str(ts),
            "oof_pred":    round(oof_pred, 6),
            "reprod_pred": round(p_m2, 6),
            "m1_pred":     round(p_m1, 6),
            "diff":        round(diff, 6),
            "abs_diff":    round(abs(diff), 6),
            "rel_diff_%":  round(rel_diff * 100, 2),
        })

    if not results:
        logger.error(f"有効な結果が0件でした（{skipped}件スキップ）。")
        return

    # --- CSV出力 ---
    results_df = pd.DataFrame(results).sort_values("abs_diff", ascending=False)
    csv_path = output_dir / f"comparison_{direction}.csv"
    results_df.to_csv(csv_path, index=False)
    logger.info(f"比較CSV保存: {csv_path}")

    # --- サマリーレポート ---
    abs_diffs = results_df["abs_diff"].values
    oof_vals  = results_df["oof_pred"].values
    rep_vals  = results_df["reprod_pred"].values

    mae = np.mean(abs_diffs)
    rmse = np.sqrt(np.mean((oof_vals - rep_vals) ** 2))
    corr = np.corrcoef(oof_vals, rep_vals)[0, 1] if len(results) > 1 else 0.0
    within_001 = np.mean(abs_diffs < 0.01) * 100
    within_005 = np.mean(abs_diffs < 0.05) * 100
    within_010 = np.mean(abs_diffs < 0.10) * 100

    report_lines = [
        "=" * 60,
        f"  再現性検証レポート ({direction.upper()})",
        "=" * 60,
        f"  検証サンプル数    : {len(results)}件（{skipped}件スキップ）",
        f"  min_proba閾値    : {min_proba}",
        "",
        "--- 一致精度 ---",
        f"  MAE              : {mae:.6f}",
        f"  RMSE             : {rmse:.6f}",
        f"  相関係数         : {corr:.6f}",
        f"  差分 < 0.01      : {within_001:.1f}%",
        f"  差分 < 0.05      : {within_005:.1f}%",
        f"  差分 < 0.10      : {within_010:.1f}%",
        "",
        "--- OOF予測値統計 ---",
        f"  mean={oof_vals.mean():.4f}  std={oof_vals.std():.4f}  "
        f"min={oof_vals.min():.4f}  max={oof_vals.max():.4f}",
        "",
        "--- 再現予測値統計 ---",
        f"  mean={rep_vals.mean():.4f}  std={rep_vals.std():.4f}  "
        f"min={rep_vals.min():.4f}  max={rep_vals.max():.4f}",
        "",
        "--- 乖離大 TOP10 ---",
    ]
    for _, row in results_df.head(10).iterrows():
        report_lines.append(
            f"  {str(row['timestamp'])[:19]}  OOF={row['oof_pred']:.4f}"
            f"  再現={row['reprod_pred']:.4f}"
            f"  diff={row['diff']:+.4f}  ({row['rel_diff_%']:.1f}%)"
        )
    report_lines.append("=" * 60)

    report_text = "\n".join(report_lines)
    print("\n" + report_text)

    report_path = output_dir / f"summary_report_{direction}.txt"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report_text + "\n")
    logger.info(f"サマリーレポート保存: {report_path}")


def main():
    parser = argparse.ArgumentParser(description="再現性検証ツール（S6直接読み方式）")
    parser.add_argument("--direction",  default="both",  choices=["long", "short", "both"],
                        help="検証方向（デフォルト: both = long・short両方）")
    parser.add_argument("--n_samples",  default=50,  type=int)
    parser.add_argument("--min_proba",  default=0.70, type=float)
    parser.add_argument("--output_dir", default=None, type=Path)
    args = parser.parse_args()

    directions = ["long", "short"] if args.direction == "both" else [args.direction]
    for direction in directions:
        verify(
            direction=direction,
            n_samples=args.n_samples,
            min_proba=args.min_proba,
            output_dir=args.output_dir,
        )


if __name__ == "__main__":
    main()
