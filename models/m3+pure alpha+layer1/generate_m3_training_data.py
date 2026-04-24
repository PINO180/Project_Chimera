# /workspace/models/generate_m3_training_data.py
# =====================================================================
# M3 Confidence Layer 学習データ生成スクリプト（Walk-Forward方式）
#
# 【目的】
#   M3 Confidence Layerの学習データを、情報リークなしで生成する。
#   ターゲット変数は「Walk-Forward予測値と実際の正解ラベルとの予測誤差」。
#   最終モデルのインサンプル情報を完全に排除する。
#
# 【設計思想】
#   Fold N までのデータで学習したM1/M2モデルで Fold N+1 を予測し、
#   その予測の正否をM3のターゲットとして記録する。
#
# 【M3入力特徴量（確定版）】
#   logit_m1   : M1の生の自信度
#   logit_m2   : M2の生の自信度（m1_pred_probaに相当）
#   atr_ratio  : ボラティリティ水準
#   hour_utc   : 時間帯（UTC・0〜23）
#   weekday    : 曜日（UTC基準・0=月〜6=日）
#   error_ema  : 直近トレードの予測外れ率Wilder平滑化（period=50）
#                初期値=train_partsの平均外れ率・fold間リセット
#                推論時点（更新前）の値を記録
#
# 【除外特徴量】
#   days_since_training : このシステムでは逆効果（時間経過で成績向上）
#   fold_position       : 致命的バイアスの温床
#
# 【M3ターゲット変数】
#   is_wrong   : predicted_class != true_label（二値・まず分類で実験）
#   pred_error : |p_m2 - true_label|（連続値・安定後に回帰で実験）
#
# 【使い方】
#   python generate_m3_training_data.py                   # long・short両方
#   python generate_m3_training_data.py --direction long  # longのみ
#   python generate_m3_training_data.py --test            # 最初の2foldのみ
#
# 【出力】
#   /workspace/data/diagnostics/m3_training/
#     m3_train_{direction}.parquet
#     summary_{direction}.txt
# =====================================================================

import sys
import gc
import argparse
import datetime
import logging
import warnings
from pathlib import Path
from typing import List

import numpy as np
import polars as pl
import lightgbm as lgb
from tqdm import tqdm

warnings.filterwarnings("ignore")

project_root = Path(__file__).resolve().parents[1]
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

from blueprint import (
    S6_WEIGHTED_DATASET,
    S3_SELECTED_FEATURES_ORTHOGONAL_DIR,
    M3_TRAINING_DATA_DIR,
)

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("M3DataGen")

OUTPUT_DIR = M3_TRAINING_DATA_DIR

K_FOLDS          = 5
PURGE_DAYS       = 3
EMBARGO_DAYS     = 2
ERROR_EMA_PERIOD = 50
ERROR_EMA_ALPHA  = 1.0 / ERROR_EMA_PERIOD  # Wilder alpha = 0.02

BASE_LGBM_PARAMS = {
    "objective":        "binary",
    "metric":           "auc",
    "boosting_type":    "gbdt",
    "learning_rate":    0.05,
    "num_leaves":       63,
    "max_depth":        -1,
    "min_data_in_leaf": 100,
    "lambda_l1":        0.1,
    "lambda_l2":        0.1,
    "seed":             42,
    "n_jobs":           4,
    "verbose":          -1,
    "colsample_bytree": 0.8,
    "subsample":        0.8,
}
NUM_BOOST_ROUND = 500


def walk_forward_split(
    partitions: List[datetime.date],
    n_splits: int = 5,
    purge_days: int = 3,
):
    """
    真のWalk-Forward分割（過去→未来の順方向のみ）。

    Step1: Fold1で学習 → Fold2を検証  (wf_step=1)
    Step2: Fold1+2で学習 → Fold3を検証 (wf_step=2)
    Step3: Fold1+2+3で学習 → Fold4を検証 (wf_step=3)
    Step4: Fold1+2+3+4で学習 → Fold5を検証 (wf_step=4)

    Fold1は検証データがないためスキップ。実質4ステップ。
    fold_index=4が最新期間（Fold5）のテスト。

    yield: (wf_step, train_parts, test_parts)
    """
    n   = len(partitions)
    fsz = n // n_splits

    # 全foldの境界を決める
    boundaries = []
    for i in range(n_splits):
        start = i * fsz
        end   = start + fsz if i < n_splits - 1 else n
        boundaries.append((start, end))

    # WFステップ: test_fold_idx=1〜4（0番目はtrainのみ）
    for test_fold_idx in range(1, n_splits):
        wf_step = test_fold_idx  # 1〜4

        # テスト期間
        ts_start, ts_end = boundaries[test_fold_idx]
        test_parts = partitions[ts_start:ts_end]
        if not test_parts:
            continue

        # 訓練期間: fold_idx=0〜test_fold_idx-1（過去のみ）
        # テスト開始日の直前までをpurge
        purge_boundary = test_parts[0] - datetime.timedelta(days=purge_days)
        train_parts = []
        for fold_idx in range(test_fold_idx):
            tr_start, tr_end = boundaries[fold_idx]
            for p in partitions[tr_start:tr_end]:
                if p < purge_boundary:
                    train_parts.append(p)

        if not train_parts:
            logger.warning(f"WF Step {wf_step}: 訓練データなし → スキップ")
            continue

        yield wf_step, train_parts, test_parts


def load_feature_list(path: Path) -> List[str]:
    with open(path) as f:
        lines = [l.strip() for l in f if l.strip()]
    exclude = {
        "timestamp", "timeframe", "t1", "label", "label_long", "label_short",
        "uniqueness", "uniqueness_long", "uniqueness_short",
        "payoff_ratio", "payoff_ratio_long", "payoff_ratio_short",
        "pt_multiplier", "sl_multiplier", "direction", "exit_type",
        "first_ex_reason_int", "atr_value", "calculated_body_ratio",
        "fallback_vol", "open", "high", "low", "close",
        "meta_label", "m1_pred_proba",
    }
    return [c for c in lines if c not in exclude
            and not c.startswith("is_trigger_on")]


def discover_partitions(base_dir: Path) -> List[datetime.date]:
    paths = sorted(base_dir.glob("year=*/month=*/day=*"))
    return sorted(set(
        datetime.date(
            int(p.parent.parent.name[5:]),
            int(p.parent.name[6:]),
            int(p.name[4:]),
        )
        for p in paths
    ))


def load_partition(base_dir: Path, date: datetime.date,
                   cols: List[str]) -> pl.DataFrame:
    path = (base_dir
            / f"year={date.year}/month={date.month}/day={date.day}/data.parquet")
    if not path.exists():
        return pl.DataFrame()
    try:
        df = pl.read_parquet(path)
        if "timeframe" in df.columns:
            df = df.unique(subset=["timestamp", "timeframe"],
                           keep="last", maintain_order=True)
        available = [c for c in cols if c in df.columns]
        return df.select(available)
    except Exception as e:
        logger.warning(f"読み込みエラー {date}: {e}")
        return pl.DataFrame()


def compute_logit(p: float, clip: float = 1e-7) -> float:
    p = max(clip, min(1.0 - clip, float(p)))
    return float(np.clip(np.log(p / (1.0 - p)), -10.0, 10.0))


def wilder_update(prev_ema: float, new_val: float, alpha: float) -> float:
    return prev_ema * (1.0 - alpha) + new_val * alpha


def generate_for_direction(direction: str, test_folds: int = 0) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    logger.info(f"\n{'='*60}")
    logger.info(f"M3学習データ生成開始: direction={direction.upper()}")
    logger.info(f"{'='*60}")

    m1_features = load_feature_list(
        S3_SELECTED_FEATURES_ORTHOGONAL_DIR / f"m1_{direction}_features.txt"
    )
    m2_features_raw = load_feature_list(
        S3_SELECTED_FEATURES_ORTHOGONAL_DIR / f"m2_{direction}_features.txt"
    )
    m2_features    = [f for f in m2_features_raw if f != "m1_pred_proba"]
    label_col      = f"label_{direction}"
    uniqueness_col = f"uniqueness_{direction}"

    all_cols = list(set(
        ["timestamp", "timeframe", label_col, uniqueness_col,
         "atr_value", "atr_ratio"]
        + m1_features + m2_features
    ))

    partitions = discover_partitions(S6_WEIGHTED_DATASET)
    logger.info(f"パーティション数: {len(partitions)}日")
    if test_folds > 0:
        logger.warning(f"--- TEST MODE: 最初の{test_folds}foldのみ ---")

    all_records = []
    # 真のWalk-Forward: 過去→未来の順方向のみ
    # WF Step1: Fold1学習→Fold2検証
    # WF Step2: Fold1+2学習→Fold3検証
    # WF Step3: Fold1+2+3学習→Fold4検証
    # WF Step4: Fold1+2+3+4学習→Fold5検証（最新期間）
    WF_STEPS = K_FOLDS - 1  # = 4

    for wf_step, train_parts, test_parts in walk_forward_split(
        partitions, n_splits=K_FOLDS, purge_days=PURGE_DAYS
    ):
        fold_num = wf_step  # 1〜4
        if test_folds > 0 and fold_num > test_folds:
            continue

        logger.info(f"\nWF Step {fold_num}/{WF_STEPS}")
        logger.info(f"  訓練: {train_parts[0]} 〜 {train_parts[-1]} ({len(train_parts)}日)")
        logger.info(f"  検証: {test_parts[0]} 〜 {test_parts[-1]} ({len(test_parts)}日)")

        # --- Step1: M1学習 ---
        X_m1_list, y_m1_list, w_m1_list = [], [], []
        for date in tqdm(train_parts, desc=f"  Fold{fold_num} M1 Train",
                         leave=False):
            df = load_partition(S6_WEIGHTED_DATASET, date, all_cols)
            if df.is_empty() or label_col not in df.columns:
                continue
            if "is_trigger" in df.columns:
                df = df.filter(pl.col("is_trigger") == 1)
            feats = [f for f in m1_features if f in df.columns]
            if not feats:
                continue
            X_m1_list.append(df.select(feats).fill_null(0).to_numpy())
            y_m1_list.append((df[label_col].to_numpy() == 1).astype(int))
            w_m1_list.append(
                df[uniqueness_col].fill_null(1.0).to_numpy()
                if uniqueness_col in df.columns else np.ones(len(df))
            )

        if not X_m1_list:
            logger.warning(f"  Fold {fold_num}: M1訓練データ空 → スキップ")
            continue

        X_m1_tr = np.concatenate(X_m1_list)
        y_m1_tr = np.concatenate(y_m1_list)
        w_m1_tr = np.concatenate(w_m1_list)
        del X_m1_list, y_m1_list, w_m1_list
        gc.collect()

        pos = y_m1_tr.sum()
        spw_m1 = float((len(y_m1_tr) - pos) / pos) if pos > 0 else 1.0
        params_m1 = {**BASE_LGBM_PARAMS, "scale_pos_weight": spw_m1}
        logger.info(f"  M1学習中 ({len(X_m1_tr)}件, spw={spw_m1:.2f})...")
        m1_model = lgb.train(params_m1,
                             lgb.Dataset(X_m1_tr, label=y_m1_tr,
                                         weight=w_m1_tr),
                             num_boost_round=NUM_BOOST_ROUND)
        del X_m1_tr, y_m1_tr, w_m1_tr
        gc.collect()

        # --- Step2: M2学習 ---
        X_m2_list, y_m2_list, w_m2_list = [], [], []
        for date in tqdm(train_parts, desc=f"  Fold{fold_num} M2 Train",
                         leave=False):
            df = load_partition(S6_WEIGHTED_DATASET, date, all_cols)
            if df.is_empty() or label_col not in df.columns:
                continue
            if "is_trigger" in df.columns:
                df = df.filter(pl.col("is_trigger") == 1)
            feats_m1 = [f for f in m1_features if f in df.columns]
            feats_m2 = [f for f in m2_features if f in df.columns]
            if not feats_m1 or not feats_m2:
                continue

            p_m1 = m1_model.predict(
                df.select(feats_m1).fill_null(0).to_numpy()
            )
            mask = p_m1 >= 0.50
            if mask.sum() == 0:
                continue

            logit_m1 = np.clip(
                np.log(np.clip(p_m1[mask], 1e-7, 1-1e-7)
                       / (1 - np.clip(p_m1[mask], 1e-7, 1-1e-7))),
                -10.0, 10.0,
            )
            df_f = df.filter(pl.Series(mask))
            X_m2 = np.column_stack([
                df_f.select(feats_m2).fill_null(0).to_numpy(), logit_m1
            ])
            y_m2 = (df_f[label_col].to_numpy() == 1).astype(int)
            w_m2 = (df_f[uniqueness_col].fill_null(1.0).to_numpy()
                    if uniqueness_col in df_f.columns
                    else np.ones(len(df_f)))
            X_m2_list.append(X_m2)
            y_m2_list.append(y_m2)
            w_m2_list.append(w_m2)

        if not X_m2_list:
            logger.warning(f"  Fold {fold_num}: M2訓練データ空 → スキップ")
            del m1_model
            gc.collect()
            continue

        X_m2_tr = np.concatenate(X_m2_list)
        y_m2_tr = np.concatenate(y_m2_list)
        w_m2_tr = np.concatenate(w_m2_list)
        del X_m2_list, y_m2_list, w_m2_list
        gc.collect()

        pos_m2 = y_m2_tr.sum()
        spw_m2 = float((len(y_m2_tr) - pos_m2) / pos_m2) if pos_m2 > 0 else 1.0
        params_m2 = {**BASE_LGBM_PARAMS, "scale_pos_weight": spw_m2}
        logger.info(f"  M2学習中 ({len(X_m2_tr)}件, spw={spw_m2:.2f})...")
        m2_model = lgb.train(params_m2,
                             lgb.Dataset(X_m2_tr, label=y_m2_tr,
                                         weight=w_m2_tr),
                             num_boost_round=NUM_BOOST_ROUND)
        del X_m2_tr, y_m2_tr, w_m2_tr
        gc.collect()

        # --- Step3: train_partsの平均外れ率計算 → error_ema初期値 ---
        logger.info(f"  error_ema初期値計算中（train_parts平均外れ率）...")
        train_wrongs = []
        for date in tqdm(train_parts, desc=f"  Fold{fold_num} ErrorRate",
                         leave=False):
            df = load_partition(S6_WEIGHTED_DATASET, date, all_cols)
            if df.is_empty() or label_col not in df.columns:
                continue
            if "is_trigger" in df.columns:
                df = df.filter(pl.col("is_trigger") == 1)
            feats_m1 = [f for f in m1_features if f in df.columns]
            feats_m2 = [f for f in m2_features if f in df.columns]
            if not feats_m1:
                continue

            true_labels = (df[label_col].to_numpy() == 1).astype(int)
            p_m1_arr    = m1_model.predict(
                df.select(feats_m1).fill_null(0).to_numpy()
            )

            for j in range(len(df)):
                if p_m1_arr[j] >= 0.50 and feats_m2:
                    row = df.row(j, named=True)
                    x_m2 = np.array([[
                        row.get(f, 0.0) or 0.0 for f in feats_m2
                    ] + [compute_logit(p_m1_arr[j])]])
                    p_m2_j = float(m2_model.predict(x_m2)[0])
                else:
                    p_m2_j = 0.0
                pred_cls = 1 if p_m2_j >= 0.5 else 0
                train_wrongs.append(int(pred_cls != int(true_labels[j])))

        error_ema_init = float(np.mean(train_wrongs)) if train_wrongs else 0.05
        logger.info(
            f"  train_parts平均外れ率: {error_ema_init*100:.2f}% "
            f"→ error_ema初期値"
        )

        # --- Step4: test_partsで予測・M3ターゲット生成 ---
        logger.info(f"  検証期間で予測・M3ターゲット生成中...")
        error_ema    = error_ema_init
        fold_records = []

        for date in tqdm(test_parts, desc=f"  Fold{fold_num} Test",
                         leave=False):
            df = load_partition(S6_WEIGHTED_DATASET, date, all_cols)
            if df.is_empty() or label_col not in df.columns:
                continue
            if "is_trigger" in df.columns:
                df = df.filter(pl.col("is_trigger") == 1)
            if df.is_empty():
                continue

            feats_m1   = [f for f in m1_features if f in df.columns]
            feats_m2   = [f for f in m2_features if f in df.columns]
            if not feats_m1:
                continue

            timestamps  = df["timestamp"].to_list()
            true_labels = (df[label_col].to_numpy() == 1).astype(int)
            atr_ratios  = (df["atr_ratio"].to_numpy()
                           if "atr_ratio" in df.columns
                           else np.ones(len(df)))
            p_m1_all    = m1_model.predict(
                df.select(feats_m1).fill_null(0).to_numpy()
            )

            for j in range(len(df)):
                p_m1       = float(p_m1_all[j])
                true_label = int(true_labels[j])

                if p_m1 >= 0.50 and feats_m2:
                    row  = df.row(j, named=True)
                    x_m2 = np.array([[
                        row.get(f, 0.0) or 0.0 for f in feats_m2
                    ] + [compute_logit(p_m1)]])
                    p_m2 = float(m2_model.predict(x_m2)[0])
                else:
                    p_m2 = 0.0

                ts = timestamps[j]
                try:
                    hour_utc = ts.hour
                    weekday  = ts.weekday()
                except Exception:
                    hour_utc = weekday = 0

                pred_class = 1 if p_m2 >= 0.5 else 0
                is_wrong   = int(pred_class != true_label)
                pred_error = abs(p_m2 - true_label)

                # 推論時点（更新前）のerror_emaを記録
                fold_records.append({
                    "timestamp":  timestamps[j],
                    "fold_index": fold_num,
                    "direction":  direction,
                    "true_label": true_label,
                    "logit_m1":   round(compute_logit(p_m1), 4),
                    "logit_m2":   round(compute_logit(p_m2), 4),
                    "atr_ratio":  round(float(atr_ratios[j]), 4),
                    "hour_utc":   hour_utc,
                    "weekday":    weekday,
                    "error_ema":  round(error_ema, 6),  # 更新前
                    "is_wrong":   is_wrong,
                    "pred_error": round(pred_error, 6),
                    "p_m1":       round(p_m1, 6),
                    "p_m2":       round(p_m2, 6),
                })

                # Wilder平滑化で更新（次のトレードから反映）
                error_ema = wilder_update(error_ema, is_wrong,
                                          ERROR_EMA_ALPHA)

        logger.info(
            f"  Fold {fold_num} 完了: {len(fold_records)}件 "
            f"(最終error_ema={error_ema:.4f})"
        )
        all_records.extend(fold_records)
        del m1_model, m2_model
        gc.collect()

    if not all_records:
        logger.error("有効なレコードが0件です。")
        return

    df_out   = pl.DataFrame(all_records).sort("timestamp")
    out_path = OUTPUT_DIR / f"m3_train_{direction}.parquet"
    df_out.write_parquet(out_path, compression="zstd")
    logger.info(f"\nParquet保存: {out_path} ({len(df_out)}件)")

    n_total    = len(df_out)
    n_wrong    = int(df_out["is_wrong"].sum())
    avg_error  = float(df_out["pred_error"].mean())
    wrong_rate = n_wrong / n_total * 100

    fold_stats = df_out.group_by("fold_index").agg([
        pl.len().alias("count"),
        pl.col("pred_error").mean().alias("avg_pred_error"),
        pl.col("is_wrong").mean().alias("wrong_rate"),
        pl.col("p_m2").mean().alias("avg_p_m2"),
        pl.col("error_ema").mean().alias("avg_error_ema"),
    ]).sort("fold_index")

    lines = [
        "=" * 65,
        f"  M3学習データ生成サマリー ({direction.upper()})",
        "=" * 65,
        f"  総レコード数            : {n_total}件",
        f"  平均予測誤差            : {avg_error:.4f}",
        f"  予測外れ率 (is_wrong=1) : {wrong_rate:.2f}%",
        f"  Wilder EMA period       : {ERROR_EMA_PERIOD}",
        "",
        "--- Fold別統計 ---",
        f"  {'Fold':>5} {'件数':>8} {'外れ率%':>8} "
        f"{'avg_pred_error':>15} {'avg_p_m2':>10} {'avg_error_ema':>14}",
        "  " + "-" * 62,
    ]
    for row in fold_stats.iter_rows(named=True):
        lines.append(
            f"  Fold{row['fold_index']:>1}  "
            f"{row['count']:>8}  "
            f"{row['wrong_rate']*100:>7.2f}%  "
            f"{row['avg_pred_error']:>15.4f}  "
            f"{row['avg_p_m2']:>10.4f}  "
            f"{row['avg_error_ema']:>14.6f}"
        )
    lines += [
        "",
        "--- M3特徴量 ---",
        "  logit_m1  : M1の生の自信度",
        "  logit_m2  : M2の生の自信度",
        "  atr_ratio : ボラティリティ水準",
        "  hour_utc  : UTC時間帯（0〜23）",
        "  weekday   : 曜日（UTC基準・0=月〜6=日）",
        f"  error_ema : 外れ率Wilder平滑化（period={ERROR_EMA_PERIOD}）",
        "              初期値=train_parts平均外れ率・fold間リセット・更新前記録",
        "",
        "--- M3ターゲット ---",
        "  is_wrong   : 二値（まず分類で実験）",
        "  pred_error : 連続値（安定後に回帰で実験）",
        "",
        "--- 次のステップ ---",
        "  train_m3_confidence_layer.py でM3を学習",
        "  scale_pos_weight = (is_wrong=0件数) / (is_wrong=1件数)",
        "  max_depth=2〜4 / min_data_in_leaf=500〜2000 / Walk-Forward CV",
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
        description="M3 Confidence Layer 学習データ生成（Walk-Forward方式）"
    )
    parser.add_argument("--direction", default="both",
                        choices=["long", "short", "both"])
    parser.add_argument("--test_folds", default=0, type=int,
                        help="テスト実行するfold数（0=全fold・1=1foldのみ）")
    args = parser.parse_args()

    directions = (["long", "short"] if args.direction == "both"
                  else [args.direction])
    for d in directions:
        generate_for_direction(d, test_folds=args.test_folds)


if __name__ == "__main__":
    main()
