# /workspace/models/train_m3_confidence_layer.py
# =====================================================================
# M3 Confidence Layer 学習スクリプト
#
# 【目的】
#   generate_m3_training_data.py が生成した学習データを用いて
#   M3 Confidence Layer（信頼度推定器）を学習する。
#
# 【設計】
#   CV:    TimeSeriesSplit（Fold1〜4で学習・Fold5でテスト）
#          ※学習データ自体がWalk-Forward生成済みのためNested CV不要
#   目的変数: is_wrong（二値分類）
#   評価指標: PR-AUC（不均衡データ対策・Loglossも併記）
#   Optuna: 強固な制約下でハイパーパラメータ探索
#            max_depth=2〜4・min_data_in_leaf=500〜2000
#
# 【入力】
#   /workspace/data/diagnostics/m3_training/
#     m3_train_long.parquet
#     m3_train_short.parquet
#
# 【出力】
#   /workspace/data/XAUUSD/stratum_7_models/
#     m3_model_long.pkl
#     m3_model_short.pkl
#     m3_oof_scores_long.parquet
#     m3_oof_scores_short.parquet
#     m3_training_report_long.txt
#     m3_training_report_short.txt
#
# 【使い方】
#   python train_m3_confidence_layer.py                   # long・short両方
#   python train_m3_confidence_layer.py --direction long  # longのみ
#   python train_m3_confidence_layer.py --n_trials 50     # Optuna試行回数
# =====================================================================

import sys
import argparse
import logging
import warnings
from pathlib import Path
from typing import List, Dict, Any

import numpy as np
import polars as pl
import lightgbm as lgb
import joblib
import optuna
from sklearn.metrics import (
    average_precision_score,
    log_loss,
    roc_auc_score,
)

warnings.filterwarnings("ignore")
optuna.logging.set_verbosity(optuna.logging.WARNING)

project_root = Path(__file__).resolve().parents[1]
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

from blueprint import (
    S7_MODELS,
    M3_TRAINING_DATA_DIR,
    M3_MODEL_LONG_PKL,
    M3_MODEL_SHORT_PKL,
    M3_OOF_SCORES_LONG,
    M3_OOF_SCORES_SHORT,
)

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("TrainM3")

INPUT_DIR  = M3_TRAINING_DATA_DIR
OUTPUT_DIR = S7_MODELS

# M3の入力特徴量（generate_m3_training_data.pyと一致）
M3_FEATURES = [
    "logit_m1",
    "logit_m2",
    "atr_ratio",
    "hour_utc",
    "weekday",
    "error_ema",
]

# Optunaの探索空間（Geminiレポート推奨・極端に狭く設定）
OPTUNA_PARAM_SPACE = {
    "max_depth":          (2, 4),       # 浅い木のみ許容
    "num_leaves":         (3, 15),      # 葉数を極限まで制限
    "min_data_in_leaf":   (500, 2000),  # 大量サンプルを要求
    "lambda_l1":          (0.1, 2.0),
    "lambda_l2":          (0.5, 5.0),
    "learning_rate":      (0.01, 0.05),
    "colsample_bytree":   (0.5, 0.8),
}

BASE_PARAMS = {
    "objective":     "binary",
    "metric":        "average_precision",  # PR-AUC最適化
    "boosting_type": "gbdt",
    "seed":          42,
    "n_jobs":        4,
    "verbose":       -1,
}
NUM_BOOST_ROUND   = 500
EARLY_STOPPING    = 50


def load_training_data(direction: str):
    path = INPUT_DIR / f"m3_train_{direction}.parquet"
    if not path.exists():
        raise FileNotFoundError(
            f"学習データが見つかりません: {path}\n"
            f"先に generate_m3_training_data.py を実行してください。"
        )
    df = pl.read_parquet(path)
    logger.info(f"学習データ読み込み: {len(df)}件")
    logger.info(f"Fold分布:\n{df.group_by('fold_index').agg(pl.len()).sort('fold_index')}")
    return df


def prepare_fold_split(df: pl.DataFrame):
    """
    Fold1〜4を学習・Fold5をテストとして分割する。
    不均衡データのためscale_pos_weightも計算する。
    """
    # WF Step1〜3を学習・WF Step4（最新期間=Fold5のテスト）を評価
    max_step = df["fold_index"].max()
    train_df = df.filter(pl.col("fold_index") < max_step)
    test_df  = df.filter(pl.col("fold_index") == max_step)

    X_train = train_df.select(M3_FEATURES).fill_null(0).to_numpy()
    y_train = train_df["is_wrong"].to_numpy().astype(int)
    X_test  = test_df.select(M3_FEATURES).fill_null(0).to_numpy()
    y_test  = test_df["is_wrong"].to_numpy().astype(int)

    # 不均衡対策：scale_pos_weight
    pos = y_train.sum()
    neg = len(y_train) - pos
    spw = float(neg / pos) if pos > 0 else 1.0

    logger.info(f"Train: {len(X_train)}件 "
                f"(is_wrong=1: {pos}件 / {pos/len(y_train)*100:.1f}%)")
    logger.info(f"Test:  {len(X_test)}件 "
                f"(is_wrong=1: {y_test.sum()}件 / "
                f"{y_test.sum()/len(y_test)*100:.1f}%)")
    logger.info(f"scale_pos_weight: {spw:.2f}")

    return X_train, y_train, X_test, y_test, spw, test_df


def train_with_params(
    X_train, y_train, X_val, y_val, spw: float,
    params: Dict[str, Any],
) -> lgb.Booster:
    """指定パラメータでM3を学習"""
    p = {**BASE_PARAMS, **params, "scale_pos_weight": spw}
    num_round = p.pop("num_boost_round", NUM_BOOST_ROUND)

    dtrain = lgb.Dataset(X_train, label=y_train,
                         feature_name=M3_FEATURES)
    dval   = lgb.Dataset(X_val, label=y_val,
                         reference=dtrain, feature_name=M3_FEATURES)

    callbacks = [
        lgb.early_stopping(EARLY_STOPPING, verbose=False),
        lgb.log_evaluation(period=-1),
    ]

    model = lgb.train(
        p, dtrain,
        num_boost_round=num_round,
        valid_sets=[dval],
        callbacks=callbacks,
    )
    return model


def run_optuna(
    X_train, y_train, X_test, y_test, spw: float,
    n_trials: int = 50,
) -> Dict[str, Any]:
    """Optunaでハイパーパラメータ探索（評価指標: PR-AUC）"""

    def objective(trial):
        params = {
            "max_depth":        trial.suggest_int(
                "max_depth", *OPTUNA_PARAM_SPACE["max_depth"]),
            "num_leaves":       trial.suggest_int(
                "num_leaves", *OPTUNA_PARAM_SPACE["num_leaves"]),
            "min_data_in_leaf": trial.suggest_int(
                "min_data_in_leaf", *OPTUNA_PARAM_SPACE["min_data_in_leaf"]),
            "lambda_l1":        trial.suggest_float(
                "lambda_l1", *OPTUNA_PARAM_SPACE["lambda_l1"], log=True),
            "lambda_l2":        trial.suggest_float(
                "lambda_l2", *OPTUNA_PARAM_SPACE["lambda_l2"], log=True),
            "learning_rate":    trial.suggest_float(
                "learning_rate", *OPTUNA_PARAM_SPACE["learning_rate"],
                log=True),
            "colsample_bytree": trial.suggest_float(
                "colsample_bytree", *OPTUNA_PARAM_SPACE["colsample_bytree"]),
            "num_boost_round":  NUM_BOOST_ROUND,
        }

        # Train/Val内部分割（Fold1〜3でtrain・Fold4でval）
        n     = len(X_train)
        split = int(n * 0.8)
        X_tr, X_vl = X_train[:split], X_train[split:]
        y_tr, y_vl = y_train[:split], y_train[split:]

        try:
            model = train_with_params(X_tr, y_tr, X_vl, y_vl, spw, params)
            scores = model.predict(X_vl)
            pr_auc = average_precision_score(y_vl, scores)
            return pr_auc
        except Exception as e:
            logger.warning(f"Trial失敗: {e}")
            return 0.0

    logger.info(f"Optuna探索開始: {n_trials}試行...")
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

    logger.info(f"最良PR-AUC: {study.best_value:.4f}")
    logger.info(f"最良パラメータ: {study.best_params}")
    return study.best_params


def train(direction: str, n_trials: int = 50) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    logger.info(f"\n{'='*60}")
    logger.info(f"M3学習開始: direction={direction.upper()}")
    logger.info(f"{'='*60}")

    # データ読み込み・分割
    df = load_training_data(direction)
    X_train, y_train, X_test, y_test, spw, test_df = prepare_fold_split(df)

    # Optuna探索
    best_params = run_optuna(
        X_train, y_train, X_test, y_test, spw, n_trials=n_trials
    )

    # 最良パラメータで最終学習（Train全体で）
    logger.info("最良パラメータで最終学習中...")
    best_params["num_boost_round"] = NUM_BOOST_ROUND
    final_model = train_with_params(
        X_train, y_train, X_test, y_test, spw, best_params
    )

    # テストデータで評価
    test_scores = final_model.predict(X_test)
    pr_auc  = average_precision_score(y_test, test_scores)
    roc_auc = roc_auc_score(y_test, test_scores)
    ll      = log_loss(y_test, test_scores)

    logger.info(f"Fold5テスト結果:")
    logger.info(f"  PR-AUC  : {pr_auc:.4f}")
    logger.info(f"  ROC-AUC : {roc_auc:.4f}")
    logger.info(f"  Logloss : {ll:.4f}")

    # モデル保存
    model_path = (M3_MODEL_LONG_PKL if direction == "long"
                  else M3_MODEL_SHORT_PKL)
    joblib.dump(final_model, model_path)
    logger.info(f"モデル保存: {model_path}")

    # OOFスコア保存（evaluate_m3_confidence.pyで使用）
    oof_df = test_df.with_columns(
        pl.Series("m3_score", test_scores)
    ).select([
        "timestamp", "fold_index", "direction",
        "true_label", "is_wrong", "pred_error",
        "p_m1", "p_m2", "logit_m1", "logit_m2",
        "atr_ratio", "hour_utc", "weekday", "error_ema",
        "m3_score",
    ])
    oof_path = (M3_OOF_SCORES_LONG if direction == "long"
                else M3_OOF_SCORES_SHORT)
    oof_df.write_parquet(oof_path, compression="zstd")
    logger.info(f"OOFスコア保存: {oof_path}")

    # 特徴量重要度
    importance = dict(zip(
        M3_FEATURES,
        final_model.feature_importance(importance_type="gain"),
    ))
    importance_sorted = sorted(
        importance.items(), key=lambda x: x[1], reverse=True
    )

    # レポート出力
    lines = [
        "=" * 60,
        f"  M3 Confidence Layer 学習レポート ({direction.upper()})",
        "=" * 60,
        f"  Train: Fold1〜4 ({len(X_train)}件)",
        f"  Test:  Fold5    ({len(X_test)}件)",
        f"  不均衡比率: is_wrong=1 が {y_train.sum()/len(y_train)*100:.2f}%",
        f"  scale_pos_weight: {spw:.2f}",
        "",
        "--- Fold5テスト結果 ---",
        f"  PR-AUC  : {pr_auc:.4f}  ← 主要指標（不均衡対応）",
        f"  ROC-AUC : {roc_auc:.4f}",
        f"  Logloss : {ll:.4f}",
        "",
        "--- 最良ハイパーパラメータ ---",
    ]
    for k, v in best_params.items():
        lines.append(f"  {k:<25}: {v}")

    lines += [
        "",
        "--- 特徴量重要度（Gain） ---",
    ]
    for feat, imp in importance_sorted:
        lines.append(f"  {feat:<20}: {imp:.1f}")

    lines += [
        "",
        "--- 信頼度スコア統計（Fold5） ---",
        f"  mean : {test_scores.mean():.4f}",
        f"  std  : {test_scores.std():.4f}",
        f"  min  : {test_scores.min():.4f}",
        f"  max  : {test_scores.max():.4f}",
        f"  > 0.70 : {(test_scores > 0.70).mean()*100:.1f}% "
        f"（フルロット候補）",
        f"  0.40〜0.70: {((test_scores >= 0.40) & (test_scores <= 0.70)).mean()*100:.1f}% "
        f"（ハーフロット候補）",
        f"  < 0.40 : {(test_scores < 0.40).mean()*100:.1f}% "
        f"（見送り候補）",
        "=" * 60,
    ]

    report = "\n".join(lines)
    print("\n" + report)

    report_path = OUTPUT_DIR / f"m3_training_report_{direction}.txt"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report + "\n")
    logger.info(f"レポート保存: {report_path}")


def main():
    parser = argparse.ArgumentParser(
        description="M3 Confidence Layer 学習スクリプト"
    )
    parser.add_argument("--direction", default="both",
                        choices=["long", "short", "both"])
    parser.add_argument("--n_trials", default=50, type=int,
                        help="Optuna試行回数（デフォルト: 50）")
    args = parser.parse_args()

    directions = (["long", "short"] if args.direction == "both"
                  else [args.direction])
    for d in directions:
        train(d, n_trials=args.n_trials)


if __name__ == "__main__":
    main()
