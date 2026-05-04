# /workspace/models/train_layer1_filter.py
# =====================================================================
# Layer 1 超軽量脳 学習スクリプト
#
# 【目的】
#   generate_layer1_training_data.py が生成した学習データを用いて
#   Layer 1 防御フィルター（超軽量脳）を学習する。
#
# 【設計】
#   特徴量  : atr_ratio / hour_utc / weekday（3変数のみ）
#   モデル  : ロジスティック回帰（L2正則化）← デフォルト
#             単一決定木（Depth=2）← --model tree で選択
#   CV      : WF Step1〜3で学習・WF Step4（最新期間）でテスト
#   閾値    : 動的パーセンタイル（スコア上位20%を見送り）
#   評価指標: PR-AUC（不均衡データ対応）・ROC-AUC
#
# 【入力】
#   LAYER1_TRAINING_DATA_DIR/
#     layer1_train_long.parquet
#     layer1_train_short.parquet
#
# 【出力】
#   LAYER1_MODEL_LONG_PKL / LAYER1_MODEL_SHORT_PKL
#   LAYER1_RESULTS_DIR/
#     layer1_report_long.txt
#     layer1_report_short.txt
#     layer1_coefficients_long.csv
#     layer1_coefficients_short.csv
#
# 【使い方】
#   python train_layer1_filter.py                   # long・short両方
#   python train_layer1_filter.py --direction long  # longのみ
#   python train_layer1_filter.py --model tree      # 決定木で学習
# =====================================================================

import sys
import argparse
import logging
import warnings
import csv
from pathlib import Path

import numpy as np
import polars as pl
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
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
    LAYER1_TRAINING_DATA_DIR,
    LAYER1_MODEL_LONG_PKL,
    LAYER1_MODEL_SHORT_PKL,
    LAYER1_RESULTS_DIR,
)

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("TrainLayer1")

# Layer 1の入力特徴量（3変数固定）
L1_FEATURES = ["atr_ratio", "hour_utc", "weekday"]

# 見送り閾値：スコア上位20%（is_bad_trade=1の確率が高い上位20%）を見送り
FILTER_PERCENTILE = 80  # 80パーセンタイル以上 → 見送り


# =====================================================================
# データ読み込み・分割
# =====================================================================

def load_data(direction: str):
    path = LAYER1_TRAINING_DATA_DIR / f"layer1_train_{direction}.parquet"
    if not path.exists():
        raise FileNotFoundError(
            f"学習データが見つかりません: {path}\n"
            f"先に generate_layer1_training_data.py を実行してください。"
        )
    df = pl.read_parquet(path)
    logger.info(f"学習データ読み込み: {len(df)}件 ({direction})")

    # WF Step1〜3を学習・Step4（最新・fold_index最大値）をテスト
    max_fold = df["fold_index"].max()
    train_df = df.filter(pl.col("fold_index") < max_fold)
    test_df  = df.filter(pl.col("fold_index") == max_fold)

    X_train = train_df.select(L1_FEATURES).fill_null(0).to_numpy()
    y_train = train_df["is_bad_trade"].to_numpy().astype(int)
    X_test  = test_df.select(L1_FEATURES).fill_null(0).to_numpy()
    y_test  = test_df["is_bad_trade"].to_numpy().astype(int)

    logger.info(
        f"Train: {len(X_train)}件 "
        f"(is_bad=1: {y_train.sum()}件 / {y_train.mean()*100:.1f}%)"
    )
    logger.info(
        f"Test:  {len(X_test)}件 "
        f"(is_bad=1: {y_test.sum()}件 / {y_test.mean()*100:.1f}%)"
    )
    return X_train, y_train, X_test, y_test


# =====================================================================
# モデル学習
# =====================================================================

def train_logistic(X_train, y_train) -> LogisticRegression:
    model = LogisticRegression(
        C=1.0,
        penalty="l2",
        class_weight="balanced",
        max_iter=1000,
        random_state=42,
    )
    model.fit(X_train, y_train)
    return model


def train_tree(X_train, y_train) -> DecisionTreeClassifier:
    model = DecisionTreeClassifier(
        max_depth=2,
        min_samples_leaf=max(100, len(X_train) // 100),
        class_weight="balanced",
        random_state=42,
    )
    model.fit(X_train, y_train)
    return model


# =====================================================================
# 評価
# =====================================================================

def evaluate(model, X_test, y_test):
    scores = model.predict_proba(X_test)[:, 1]  # is_bad_trade=1の確率

    pr_auc  = average_precision_score(y_test, scores)
    roc_auc = roc_auc_score(y_test, scores)
    ll      = log_loss(y_test, scores)

    # 動的パーセンタイル閾値（スコア上位20%を見送り）
    threshold = float(np.percentile(scores, FILTER_PERCENTILE))

    mask_skip = scores >= threshold  # 見送り（スコアが高い＝危険）
    mask_keep = scores < threshold   # エントリー継続

    base_bad_rate = float(y_test.mean())

    def stats(mask):
        if mask.sum() == 0:
            return {"count": 0, "bad_rate": 0.0, "pct": 0.0}
        return {
            "count":    int(mask.sum()),
            "bad_rate": float(y_test[mask].mean()),
            "pct":      float(mask.mean() * 100),
        }

    return {
        "pr_auc":        pr_auc,
        "roc_auc":       roc_auc,
        "log_loss":      ll,
        "threshold":     threshold,
        "base_bad_rate": base_bad_rate,
        "skip":          stats(mask_skip),
        "keep":          stats(mask_keep),
        "scores":        scores,
    }


# =====================================================================
# メイン
# =====================================================================

def train(direction: str, model_type: str = "logistic") -> None:
    LAYER1_RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    logger.info(f"\n{'='*60}")
    logger.info(
        f"Layer 1学習開始: direction={direction.upper()} / model={model_type}"
    )
    logger.info(f"{'='*60}")

    X_train, y_train, X_test, y_test = load_data(direction)

    # 学習
    if model_type == "tree":
        model = train_tree(X_train, y_train)
        logger.info("決定木学習完了 (depth=2)")
    else:
        model = train_logistic(X_train, y_train)
        logger.info("ロジスティック回帰学習完了")

    # 評価
    result = evaluate(model, X_test, y_test)

    logger.info(f"PR-AUC: {result['pr_auc']:.4f}")
    logger.info(f"ROC-AUC: {result['roc_auc']:.4f}")
    logger.info(
        f"見送り内の負け率: {result['skip']['bad_rate']*100:.2f}% "
        f"(ベース: {result['base_bad_rate']*100:.2f}%)"
    )

    # モデルと閾値を一緒に保存
    model_bundle = {
        "model":      model,
        "threshold":  result["threshold"],
        "features":   L1_FEATURES,
        "direction":  direction,
        "model_type": model_type,
    }
    model_path = (LAYER1_MODEL_LONG_PKL if direction == "long"
                  else LAYER1_MODEL_SHORT_PKL)
    joblib.dump(model_bundle, model_path)
    logger.info(f"モデル保存: {model_path}")

    # 係数・重要度をCSV出力
    coef_path = LAYER1_RESULTS_DIR / f"layer1_coefficients_{direction}.csv"
    with open(coef_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f, fieldnames=["feature", "value", "interpretation"]
        )
        writer.writeheader()
        if model_type == "logistic":
            for feat, coef in zip(L1_FEATURES, model.coef_[0]):
                writer.writerow({
                    "feature":        feat,
                    "value":          round(float(coef), 6),
                    "interpretation": "正=負けやすい / 負=勝ちやすい",
                })
        else:
            for feat, imp in zip(L1_FEATURES, model.feature_importances_):
                writer.writerow({
                    "feature":        feat,
                    "value":          round(float(imp), 6),
                    "interpretation": "重要度（高いほど判定に影響）",
                })
    logger.info(f"係数CSV保存: {coef_path}")

    # レポート
    lines = [
        "=" * 65,
        f"  Layer 1 超軽量脳 学習レポート ({direction.upper()})",
        "=" * 65,
        f"  モデル     : {model_type}",
        f"  特徴量     : {' / '.join(L1_FEATURES)}",
        f"  時刻基準   : UTC（MT5・本番と一致）",
        f"  Train      : {len(X_train)}件（WF Step1〜3）",
        f"  Test       : {len(X_test)}件（WF Step4・最新期間）",
        f"  ベースライン負け率: {result['base_bad_rate']*100:.2f}%",
        "",
        "--- 評価指標 ---",
        f"  PR-AUC  : {result['pr_auc']:.4f}  ← 主要指標",
        f"  ROC-AUC : {result['roc_auc']:.4f}",
        f"  Logloss : {result['log_loss']:.4f}",
        "",
        f"--- フィルター効果（上位{100-FILTER_PERCENTILE}%を見送り）---",
        f"  動的閾値 : {result['threshold']:.4f}"
        f"（スコアの{FILTER_PERCENTILE}パーセンタイル）",
        f"  {'区分':<16} {'件数':>8} {'割合%':>8} {'負け率%':>10}",
        "  " + "-" * 46,
        f"  {'見送り (危険)':<16} "
        f"{result['skip']['count']:>8} "
        f"{result['skip']['pct']:>7.1f}% "
        f"{result['skip']['bad_rate']*100:>9.2f}%",
        f"  {'通過 (安全)':<16} "
        f"{result['keep']['count']:>8} "
        f"{result['keep']['pct']:>7.1f}% "
        f"{result['keep']['bad_rate']*100:>9.2f}%",
        f"  {'全体':<16} "
        f"{len(X_test):>8} "
        f"{'100.0':>7}% "
        f"{result['base_bad_rate']*100:>9.2f}%",
        "",
        "  【判定基準】",
        "  見送り内の負け率 > ベースライン → Layer 1が危険を正しく検知",
        "  通過後の負け率 < ベースライン  → Layer 1が機能している",
        "",
    ]

    if model_type == "logistic":
        lines += ["--- ロジスティック回帰 係数 ---"]
        for feat, coef in zip(L1_FEATURES, model.coef_[0]):
            sign = "→ 負けやすい" if coef > 0 else "→ 勝ちやすい"
            lines.append(f"  {feat:<20}: {coef:>+.4f}  {sign}")
        lines.append(f"  {'intercept':<20}: {model.intercept_[0]:>+.4f}")
    else:
        lines += ["--- 決定木 特徴量重要度 ---"]
        for feat, imp in zip(L1_FEATURES, model.feature_importances_):
            lines.append(f"  {feat:<20}: {imp:.4f}")

    lines += ["", "=" * 65]
    report = "\n".join(lines)
    print("\n" + report)

    report_path = LAYER1_RESULTS_DIR / f"layer1_report_{direction}.txt"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report + "\n")
    logger.info(f"レポート保存: {report_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Layer 1 超軽量脳 学習スクリプト"
    )
    parser.add_argument(
        "--direction", default="both",
        choices=["long", "short", "both"],
    )
    parser.add_argument(
        "--model", default="logistic",
        choices=["logistic", "tree"],
        help="logistic: ロジスティック回帰（デフォルト）/ tree: 決定木Depth=2",
    )
    args = parser.parse_args()

    directions = (["long", "short"] if args.direction == "both"
                  else [args.direction])
    for d in directions:
        train(d, model_type=args.model)


if __name__ == "__main__":
    main()
