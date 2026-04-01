# /workspace/models/model_training_metalabeling_C.py
# [修正版: M2/M1訓練を lgb.train (Booster API) + n_estimators 分配に修正]
# [修正版: フェーズ7 (V4) - M2の特徴量に市場文脈カラムを追加]

import sys
from pathlib import Path
import logging
import argparse
import json
import warnings
import datetime
import shutil
from dataclasses import dataclass, field
import gc

import polars as pl
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import BaseCrossValidator
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score
import joblib
from typing import List, Tuple, Dict, Any, Generator
from tqdm import tqdm
from collections import Counter

# --- プロジェクトのルートディレクトリをPythonの検索パスに追加 ---
project_root = Path(__file__).resolve().parents[1]
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

from blueprint import (
    S6_WEIGHTED_DATASET,
    S3_FEATURES_FOR_TRAINING,
    S7_M1_OOF_PREDICTIONS,
    S7_META_LABELED_OOF_PARTITIONED,
    S7_M1_MODEL_PKL,
    S7_M2_MODEL_PKL,
    S7_M1_CALIBRATED,
    S7_M2_CALIBRATED,
    S7_MODEL_PERFORMANCE_REPORT,
    S7_MODELS,
)

S7_M2_OOF_PREDICTIONS_TMP = S7_MODELS / "tmp_m2_oof_predictions"
S7_M2_OOF_PREDICTIONS = S7_MODELS / "m2_oof_predictions_v2.parquet"

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
warnings.filterwarnings("ignore", category=UserWarning)
try:
    from polars.exceptions import PolarsInefficientMapWarning

    warnings.filterwarnings("ignore", category=DeprecationWarning)
    warnings.filterwarnings("ignore", category=PolarsInefficientMapWarning)
except ImportError:
    pass


@dataclass
class FinalTrainingConfig:
    weighted_dataset_path: Path = S6_WEIGHTED_DATASET
    meta_labeled_oof_path: Path = S7_META_LABELED_OOF_PARTITIONED
    m1_oof_path: Path = S7_M1_OOF_PREDICTIONS
    feature_list_path: Path = S3_FEATURES_FOR_TRAINING
    top_50_features_path: Path = project_root / "models" / "TOP_50_FEATURES.json"
    n_splits: int = 5
    purge_days: int = 3
    embargo_days: int = 2
    lgbm_params_m1: Dict[str, Any] = field(
        default_factory=lambda: {
            "objective": "binary",
            "metric": "auc",
            "boosting_type": "gbdt",
            "n_estimators": 2000,  # ★推奨: バッチ学習用に2000に戻す
            "learning_rate": 0.01,
            "num_leaves": 31,
            "max_depth": -1,
            "seed": 42,
            "n_jobs": -1,
            "verbose": -1,
            "colsample_bytree": 0.8,
            "subsample": 0.8,
        }
    )
    lgbm_params_m2: Dict[str, Any] = field(
        default_factory=lambda: {
            "objective": "binary",
            "metric": "auc",
            "boosting_type": "gbdt",
            "n_estimators": 2000,  # ★推奨: バッチ学習用に2000に戻す
            "learning_rate": 0.01,
            "num_leaves": 31,  # ★元に戻す: 表現力を確保
            "max_depth": -1,
            "seed": 42,
            "n_jobs": -1,
            "verbose": -1,
            "colsample_bytree": 0.8,  # ★元に戻す: 過学習抑制
            "subsample": 0.8,  # ★元に戻す: 過学習抑制
            # min_child_samples はデフォルト(20)でOKなので削除（または明示的に20）
        }
    )
    test: bool = False


class PartitionPurgedKFold(BaseCrossValidator):
    def __init__(self, n_splits: int = 5, purge_days: int = 3, embargo_days: int = 2):
        self.n_splits, self.purge_days, self.embargo_days = (
            n_splits,
            purge_days,
            embargo_days,
        )

    def split(
        self, partitions: List[datetime.date]
    ) -> Generator[Tuple[List[datetime.date], List[datetime.date]], None, None]:
        n_partitions = len(partitions)
        fold_size = n_partitions // self.n_splits
        for i in range(self.n_splits):
            start, end = (
                i * fold_size,
                (i + 1) * fold_size if i < self.n_splits - 1 else n_partitions,
            )
            test_partitions = partitions[start:end]
            if not test_partitions:
                continue
            test_start_date, test_end_date = test_partitions[0], test_partitions[-1]
            purge_start = test_start_date - datetime.timedelta(days=self.purge_days)
            embargo_end = test_end_date + datetime.timedelta(days=self.embargo_days)
            train_partitions = [
                p for p in partitions if not (purge_start <= p <= embargo_end)
            ]
            yield train_partitions, test_partitions

    def get_n_splits(self, X=None, y=None, groups=None) -> int:
        return self.n_splits


class FinalAssembler:
    def __init__(self, config: FinalTrainingConfig):
        self.config = config

        # [修正] Top 50 (JSON) 読み込みを廃止し、既存の _load_features (txt) を使用する
        self.all_features = self._load_features()

        # M1用の特徴量 (Base) -> 全特徴量を使用
        self.features_base: List[str] = self.all_features

        # M2用の特徴量 -> 全特徴量 + m1_pred_proba
        self.features_m2: List[str] = ["m1_pred_proba"] + self.all_features

        # ... (以下変更なし) ...
        # -----------------------------------------------------------------------

        # M1 と M2 で使用するパーティションリストを個別に検出
        self.partitions_m1_final = self._discover_partitions_for_m1_final_train()
        self.partitions_m2 = self._discover_partitions()  # M2 CV/Final/Calib 用

        # --- M1 用の scale_pos_weight を計算 (S6_WEIGHTED_DATASET 全体から) ---
        self.scale_pos_weight_m1 = self._calculate_scale_pos_weight_m1()
        self.config.lgbm_params_m1["scale_pos_weight"] = self.scale_pos_weight_m1
        logging.info(f"Using scale_pos_weight for M1: {self.scale_pos_weight_m1:.4f}")

        # --- M2 用の scale_pos_weight を計算 (S7_META_LABELED_OOF_PARTITIONED から) ---
        # ★★★ 修正適用済み: 空フレーム防御ロジック搭載 ★★★
        self.scale_pos_weight_m2 = self._calculate_scale_pos_weight_m2()
        self.config.lgbm_params_m2["scale_pos_weight"] = self.scale_pos_weight_m2
        logging.info(f"Using scale_pos_weight for M2: {self.scale_pos_weight_m2:.4f}")

    def _load_features(self) -> List[str]:
        logging.info(
            f"Loading base feature list from {self.config.feature_list_path}..."
        )
        with open(self.config.feature_list_path, "r") as f:
            features = [line.strip() for line in f if line.strip()]
        logging.info(f"   -> Loaded {len(features)} base features.")
        return features

    # def _load_top_50_features(self) -> List[str]:
    #     logging.info(
    #         f"Loading Top 50 features list from {self.config.top_50_features_path}..."
    #     )
    #     if not self.config.top_50_features_path.exists():
    #         raise FileNotFoundError(
    #             f"Top 50 features file not found at: {self.config.top_50_features_path}"
    #         )

    #     with open(self.config.top_50_features_path, "r", encoding="utf-8") as f:
    #         features = json.load(f)

    #     logging.info(f"   -> Loaded {len(features)} Top 50 features.")
    #     return features

    def _discover_partitions(self) -> List[datetime.date]:
        # M2 の学習データ (S7_META_LABELED_OOF_PARTITIONED) からパーティションを見つける
        logging.info(
            f"Discovering partitions from M2 meta-labeled data: {self.config.meta_labeled_oof_path}..."
        )
        paths = self.config.meta_labeled_oof_path.glob("year=*/month=*/day=*")
        dates = sorted(
            list(
                set(
                    datetime.date(
                        int(p.parent.parent.name[5:]),
                        int(p.parent.name[6:]),
                        int(p.name[4:]),
                    )
                    for p in paths
                    if p.is_dir()
                )
            )
        )
        if not dates:
            logging.warning(
                f"No partitions found in {self.config.meta_labeled_oof_path}. Ensure Script B ran successfully."
            )
        else:
            logging.info(
                f"  -> Discovered {len(dates)} daily partitions for M2 processes."
            )
        return dates

    # --- M1 用の scale_pos_weight 計算関数 ---
    def _calculate_scale_pos_weight_m1(self) -> float:
        logging.info(
            "Calculating scale_pos_weight for M1 (label == 1 vs label != 1) from S6_WEIGHTED_DATASET..."
        )
        counts = Counter({0: 0, 1: 0})
        total_samples = 0
        partitions_to_scan = self.partitions_m1_final
        if not partitions_to_scan:
            logging.warning("No M1 partitions (S6) found. Returning 1.0.")
            return 1.0

        for partition_date in tqdm(
            partitions_to_scan, desc="Scanning labels for scale_pos_weight (M1)"
        ):
            p_path_glob = str(
                self.config.weighted_dataset_path
                / f"year={partition_date.year}/month={partition_date.month}/day={partition_date.day}/*.parquet"
            )
            try:
                # 注: weighted_dataset_path は uniqueness 列を持つため、空フレームになる可能性は低い
                df_labels = pl.read_parquet(p_path_glob, columns=["label"])
                if df_labels.is_empty():
                    continue
                binary_labels = df_labels.select(
                    pl.when(pl.col("label") == 1)
                    .then(1)
                    .otherwise(0)
                    .alias("binary_label")
                )
                counts_in_partition = binary_labels["binary_label"].value_counts()

                pos_count_df = counts_in_partition.filter(
                    pl.col("binary_label") == 1
                ).select(pl.col("count"))
                pos_count = pos_count_df.item() if not pos_count_df.is_empty() else 0

                neg_count_df = counts_in_partition.filter(
                    pl.col("binary_label") == 0
                ).select(pl.col("count"))
                neg_count = neg_count_df.item() if not neg_count_df.is_empty() else 0

                counts[1] += pos_count
                counts[0] += neg_count
                total_samples += pos_count + neg_count
            except Exception as e:
                logging.warning(
                    f"Could not read labels from {partition_date} (M1): {e}"
                )
                continue

        count_neg, count_pos = counts[0], counts[1]
        if count_pos == 0 or count_neg == 0:
            logging.warning(
                "M1: One of the classes has zero samples. Using scale_pos_weight = 1.0"
            )
            return 1.0
        scale_pos_weight = count_neg / count_pos
        logging.info(
            f"  -> M1 Total samples: {total_samples}, Pos: {count_pos}, Neg: {count_neg}"
        )
        return scale_pos_weight

    # --- M2 用の scale_pos_weight 計算関数 (空フレーム防御適用) ---
    def _calculate_scale_pos_weight_m2(self) -> float:
        logging.info(
            "Calculating scale_pos_weight for M2 (meta_label == 1 vs meta_label == 0)..."
        )
        counts = Counter({0: 0, 1: 0})
        total_samples = 0
        partitions_to_scan = self.partitions_m2
        if not partitions_to_scan:
            logging.warning(
                "No M2 partitions found to calculate scale_pos_weight. Returning 1.0."
            )
            return 1.0

        for partition_date in tqdm(
            partitions_to_scan, desc="Scanning meta-labels for scale_pos_weight (M2)"
        ):
            p_path_glob = str(
                self.config.meta_labeled_oof_path
                / f"year={partition_date.year}/month={partition_date.month}/day={partition_date.day}/*.parquet"
            )
            try:
                df_labels = pl.read_parquet(p_path_glob, columns=["meta_label"])
                df_labels = df_labels.filter(pl.col("meta_label").is_not_null())
                if df_labels.is_empty():
                    continue

                counts_in_partition = df_labels["meta_label"].value_counts()

                # ★★★ 修正適用済み: Polars の防御的プログラミング (item() 呼び出しの防御) ★★★
                # Positive Count
                pos_count_df = counts_in_partition.filter(
                    pl.col("meta_label") == 1
                ).select(pl.col("count"))
                pos_count = pos_count_df.item() if not pos_count_df.is_empty() else 0

                # Negative Count
                neg_count_df = counts_in_partition.filter(
                    pl.col("meta_label") == 0
                ).select(pl.col("count"))
                neg_count = neg_count_df.item() if not neg_count_df.is_empty() else 0
                # ★★★ 修正ここまで ★★★

                counts[1] += pos_count
                counts[0] += neg_count
                total_samples += pos_count + neg_count
            except Exception as e:
                logging.warning(
                    f"Could not read meta-labels from {partition_date}: {e}"
                )
                continue

        count_neg, count_pos = counts[0], counts[1]
        if count_pos == 0 or count_neg == 0:
            logging.warning(
                "M2: One of the meta-label classes has zero samples. Using scale_pos_weight = 1.0"
            )
            return 1.0
        scale_pos_weight = count_neg / count_pos
        logging.info(
            f"  -> M2 Total samples: {total_samples}, Pos: {count_pos}, Neg: {count_neg}"
        )
        return scale_pos_weight

    def run(self):
        logging.info(
            "### Script 3/3: M2 CV, Final Training, Calibration, and Reporting ###"
        )
        if not S7_M2_OOF_PREDICTIONS_TMP.exists() or not any(
            S7_M2_OOF_PREDICTIONS_TMP.iterdir()
        ):
            if S7_M2_OOF_PREDICTIONS_TMP.exists():
                logging.info(
                    f"Cleaning existing M2 OOF temp directory: {S7_M2_OOF_PREDICTIONS_TMP}"
                )
                shutil.rmtree(S7_M2_OOF_PREDICTIONS_TMP)
            S7_M2_OOF_PREDICTIONS_TMP.mkdir(parents=True)
            self._train_m2_cv_and_write_to_disk()
        else:
            logging.warning(
                "--- SKIPPING M2 Cross-Validation: Temporary OOF files already exist. ---"
            )

        self._train_and_calibrate_final_models()
        self._aggregate_m2_oof_predictions()
        # ★★★ 修正適用済み: AUCロバスト化ロジック搭載 ★★★
        self._generate_performance_report()

        if S7_M2_OOF_PREDICTIONS_TMP.exists():
            logging.info(
                f"Cleaning up temporary directory: {S7_M2_OOF_PREDICTIONS_TMP}"
            )
            shutil.rmtree(S7_M2_OOF_PREDICTIONS_TMP)

        logging.info(
            "\n"
            + "=" * 60
            + "\n### PROJECT FORGE: ALL STAGES COMPLETED! ###\n"
            + "=" * 60
        )

    # --- M2 CV の訓練 (lgb.train/Booster API) ---
    def _train_m2_cv_and_write_to_disk(self):
        logging.info(f"--- Starting BATCH Training for M2 (Meta) CV ---")
        kfold = PartitionPurgedKFold(
            self.config.n_splits, self.config.purge_days, self.config.embargo_days
        )
        partitions_to_use = self.partitions_m2
        if not partitions_to_use:
            logging.error("No partitions available for M2 CV. Aborting CV.")
            return

        for i, (train_dates, val_dates) in enumerate(kfold.split(partitions_to_use)):
            logging.info(f"  [M2 (Meta)] Fold {i + 1}/{self.config.n_splits}...")

            if self.config.test:
                logging.warning(
                    f"--- TEST MODE: Limiting M2 CV Fold {i + 1} partitions. ---"
                )
                train_dates = train_dates[:5]
                val_dates = val_dates[:5]

            # --- データ収集 (Batch) ---
            X_train_list, y_train_list, w_train_list = [], [], []

            if len(train_dates) > 0:
                for p_date in tqdm(
                    train_dates, desc=f"  Loading M2 Train Fold {i + 1}"
                ):
                    p_path_glob = str(
                        self.config.meta_labeled_oof_path
                        / f"year={p_date.year}/month={p_date.month}/day={p_date.day}/*.parquet"
                    )
                    try:
                        df_chunk = pl.read_parquet(p_path_glob)
                    except Exception:
                        continue
                    # M2ターゲットがある行のみ
                    df_chunk = df_chunk.filter(pl.col("meta_label").is_not_null())
                    if df_chunk.is_empty():
                        continue

                    features_to_use = self.features_m2
                    missing_features = [
                        f for f in features_to_use if f not in df_chunk.columns
                    ]
                    if missing_features:
                        continue

                    X_train_list.append(
                        df_chunk.select(features_to_use).fill_null(0).to_numpy()
                    )
                    y_train_list.append(df_chunk["meta_label"].to_numpy())
                    w_train_list.append(df_chunk["uniqueness"].to_numpy())

            # --- 学習実行 (Batch) ---
            model: lgb.Booster = None
            if len(X_train_list) > 0:
                try:
                    X_train = np.concatenate(X_train_list)
                    y_train = np.concatenate(y_train_list)
                    w_train = np.concatenate(w_train_list)

                    train_params = self.config.lgbm_params_m2.copy()
                    n_estimators = train_params.pop("n_estimators", 1000)

                    logging.info(
                        f"    -> Training M2 model on {len(X_train)} samples..."
                    )
                    model = lgb.train(
                        train_params,
                        lgb.Dataset(X_train, label=y_train, weight=w_train),
                        num_boost_round=n_estimators,
                    )
                except Exception as fit_error:
                    logging.error(f"Error fitting M2 model: {fit_error}")

            if model is None:
                logging.warning(
                    f"M2 model for Fold {i + 1} was not trained. Skipping prediction."
                )
                continue

            # --- 予測 (Validation) ---
            for p_date in tqdm(val_dates, desc=f"  Predicting M2 Fold {i + 1}"):
                p_path_glob = str(
                    self.config.meta_labeled_oof_path
                    / f"year={p_date.year}/month={p_date.month}/day={p_date.day}/*.parquet"
                )
                try:
                    df_chunk = pl.read_parquet(p_path_glob)
                except Exception:
                    continue
                df_chunk = df_chunk.filter(pl.col("meta_label").is_not_null())
                if df_chunk.is_empty():
                    continue

                features_to_use = self.features_m2
                try:
                    X_val = df_chunk.select(features_to_use).fill_null(0).to_numpy()
                    predictions = model.predict(X_val)
                except Exception as pred_error:
                    logging.error(f"Error predicting M2 model: {pred_error}")
                    predictions = np.full(len(df_chunk), np.nan)

                oof_df = pl.DataFrame(
                    {
                        "timestamp": df_chunk["timestamp"],
                        "prediction": predictions,
                        "true_label": df_chunk["meta_label"],
                        "uniqueness": df_chunk["uniqueness"],
                        "payoff_ratio": df_chunk["payoff_ratio"],
                    }
                )
                output_partition_dir = (
                    S7_M2_OOF_PREDICTIONS_TMP
                    / f"year={p_date.year}/month={p_date.month}/day={p_date.day}"
                )
                output_partition_dir.mkdir(parents=True, exist_ok=True)
                oof_df.write_parquet(
                    output_partition_dir / "data.parquet", compression="zstd"
                )

            del model
            gc.collect()

    def _train_and_calibrate_final_models(self):
        logging.info("--- Training and Calibrating Final Models ---")

        # --- M1 最終訓練 ---
        m1_model = self._ensure_model_trained(
            "M1",
            S7_M1_MODEL_PKL,
            is_m2=False,
            partitions_to_train=self.partitions_m1_final,  # M1用パーティション
            lgbm_params=self.config.lgbm_params_m1,
        )
        # --- M2 最終訓練 ---
        m2_model = self._ensure_model_trained(
            "M2",
            S7_M2_MODEL_PKL,
            is_m2=True,
            partitions_to_train=self.partitions_m2,  # M2用パーティション
            lgbm_params=self.config.lgbm_params_m2,
        )

        logging.info("  - Calibrating models (Manual, Memory-Safe Mode)...")
        # 較正にはM2データセットの最新部分を使う
        calib_dates = self.partitions_m2[
            -len(self.partitions_m2) // self.config.n_splits :
        ]
        if calib_dates:
            self._manual_calibrate(
                "M1", m1_model, S7_M1_CALIBRATED, calib_dates, is_m2=False
            )
            self._manual_calibrate(
                "M2", m2_model, S7_M2_CALIBRATED, calib_dates, is_m2=True
            )
        else:
            logging.warning(
                "No partitions available for calibration. Skipping calibration."
            )

    def _manual_calibrate(
        self,
        model_name: str,
        model: lgb.Booster,
        save_path: Path,
        dates: List[datetime.date],
        is_m2: bool,
    ):
        if not save_path.exists():
            logging.info(f"    -> Calibrating {model_name} model...")
            y_pred, y_true, weights = self._gather_predictions_for_calibration(
                model_name, dates, model, is_m2
            )
            if y_pred is not None and len(y_pred) > 0:
                try:
                    calibrator = IsotonicRegression(
                        y_min=0.0, y_max=1.0, out_of_bounds="clip"
                    )
                    calibrator.fit(y_pred, y_true, sample_weight=weights)
                    joblib.dump(calibrator, save_path)
                    logging.info(
                        f"  -> Calibrated {model_name} (IsotonicRegressor) saved to {save_path}."
                    )
                except Exception as calib_error:
                    logging.error(
                        f"Failed to fit calibrator for {model_name}: {calib_error}",
                        exc_info=False,
                    )
            else:
                logging.warning(
                    f"No valid predictions gathered for {model_name} calibration. Skipping."
                )
        else:
            logging.warning(
                f"--- SKIPPING {model_name} Calibration: Calibrated model already exists. ---"
            )

    def _gather_predictions_for_calibration(
        self,
        model_name: str,
        dates: List[datetime.date],
        model: lgb.Booster,
        is_m2: bool,
    ) -> Tuple[np.ndarray | None, np.ndarray | None, np.ndarray | None]:
        all_preds, all_labels, all_weights = [], [], []

        # [修正] M1の較正ならS6(全特徴量あり)、M2ならS7(Top50あり)を参照する
        base_path = (
            self.config.meta_labeled_oof_path
            if is_m2
            else self.config.weighted_dataset_path
        )

        for p_date in tqdm(dates, desc=f"  Gathering {model_name} calib predictions"):
            p_path_glob = str(
                base_path
                / f"year={p_date.year}/month={p_date.month}/day={p_date.day}/*.parquet"
            )
            try:
                df_chunk = pl.read_parquet(p_path_glob)
                # M1較正時、S6には 'label' があるのでチェック不要だが念のため
                if not is_m2 and "label" not in df_chunk.columns:
                    logging.warning(
                        f"Original 'label' column not found in {p_date} for M1 calibration. Skipping chunk."
                    )
                    continue
            except Exception:
                continue

            if is_m2:
                # M2の場合は meta_label がある行だけに絞る
                df_chunk = df_chunk.filter(pl.col("meta_label").is_not_null())

            if df_chunk.is_empty():
                continue

            features_to_use = self.features_m2 if is_m2 else self.features_base
            actual_target_col = "meta_label" if is_m2 else "label"

            missing_features = [f for f in features_to_use if f not in df_chunk.columns]
            if missing_features:
                logging.warning(
                    f"Calibration partition {p_date} missing features for {model_name}: {missing_features}. Skipping chunk."
                )
                continue

            X = df_chunk.select(features_to_use).fill_null(0).to_numpy()

            try:
                predictions = model.predict(X)
            except Exception as pred_error:
                logging.error(
                    f"Error predicting for calibration ({model_name}, {p_date}): {pred_error}",
                    exc_info=False,
                )
                continue

            labels = df_chunk[actual_target_col].to_numpy()
            if not is_m2:
                labels = np.where(labels == 1, 1, 0)  # M1ラベルを 0/1 に

            all_preds.append(predictions)
            all_labels.append(labels)
            all_weights.append(df_chunk["uniqueness"].to_numpy())

        if not all_preds:
            return None, None, None
        try:
            if not all_preds or not all_labels or not all_weights:
                return None, None, None
            return (
                np.concatenate(all_preds),
                np.concatenate(all_labels),
                np.concatenate(all_weights),
            )
        except ValueError as concat_error:
            logging.error(
                f"Error concatenating calibration data for {model_name}: {concat_error}"
            )
            return None, None, None

    def _ensure_model_trained(
        self,
        model_name: str,
        model_path: Path,
        is_m2: bool,
        partitions_to_train: List[datetime.date],
        lgbm_params: Dict[str, Any],
    ) -> lgb.Booster:
        if not model_path.exists():
            if not partitions_to_train:
                raise ValueError(
                    f"No partitions available to train the final {model_name} model."
                )
            model = self._train_single_model(
                f"{model_name} (Final)",
                "meta_label" if is_m2 else "label",
                is_m2,
                partitions_to_train,
                lgbm_params,
            )
            joblib.dump(model, model_path)
            logging.info(f"  -> {model_name} model saved to {model_path}.")
        else:
            logging.warning(
                f"--- SKIPPING {model_name} Final Training: Model file already exists at {model_path}. ---"
            )
            model = joblib.load(model_path)
        return model

    # --- 最終モデル訓練を lgb.train (Booster API) で実行 ---
    def _train_single_model(
        self,
        model_name: str,
        target_col: str,
        is_m2: bool,
        partitions_to_train: List[datetime.date],
        lgbm_params: Dict[str, Any],
    ) -> lgb.Booster:
        logging.info(
            f"  - Training {model_name} on {len(partitions_to_train)} partitions (BATCH Mode)..."
        )

        model: lgb.Booster = None
        input_path = (
            self.config.meta_labeled_oof_path
            if is_m2
            else self.config.weighted_dataset_path
        )

        # --- データ収集 ---
        X_list, y_list, w_list = [], [], []

        for p_date in tqdm(
            partitions_to_train, desc=f"  Loading Data for {model_name}"
        ):
            p_path_glob = str(
                input_path
                / f"year={p_date.year}/month={p_date.month}/day={p_date.day}/*.parquet"
            )
            try:
                df_chunk = pl.read_parquet(p_path_glob)
            except Exception:
                continue

            if is_m2:
                df_chunk = df_chunk.filter(pl.col(target_col).is_not_null())
            if df_chunk.is_empty():
                continue

            features_to_use = self.features_m2 if is_m2 else self.features_base
            missing_features = [f for f in features_to_use if f not in df_chunk.columns]
            if missing_features:
                continue

            X_chunk = df_chunk.select(features_to_use).fill_null(0).to_numpy()
            y_chunk = df_chunk[target_col].to_numpy()
            w_chunk = df_chunk["uniqueness"].to_numpy()

            if not is_m2:
                y_chunk = np.where(y_chunk == 1, 1, 0)

            X_list.append(X_chunk)
            y_list.append(y_chunk)
            w_list.append(w_chunk)

        # --- 学習実行 ---
        if not X_list:
            raise RuntimeError(f"No data found for {model_name}.")

        try:
            X_train = np.concatenate(X_list)
            y_train = np.concatenate(y_list)
            w_train = np.concatenate(w_list)

            del X_list, y_list, w_list
            gc.collect()

            train_params = lgbm_params.copy()
            n_estimators = train_params.pop("n_estimators", 1000)

            logging.info(f"    -> fitting {model_name} on {len(X_train)} samples...")

            model = lgb.train(
                train_params,
                lgb.Dataset(X_train, label=y_train, weight=w_train),
                num_boost_round=n_estimators,
            )

            del X_train, y_train, w_train
            gc.collect()

        except Exception as fit_error:
            raise RuntimeError(f"Failed to train {model_name}: {fit_error}")

        return model

    # --- (維持) M1 最終訓練用のパーティション検出 ---
    def _discover_partitions_for_m1_final_train(self) -> List[datetime.date]:
        logging.info(
            f"Discovering partitions for M1 final training from: {self.config.weighted_dataset_path}..."
        )
        paths = self.config.weighted_dataset_path.glob("year=*/month=*/day=*")
        dates = sorted(
            list(
                set(
                    datetime.date(
                        int(p.parent.parent.name[5:]),
                        int(p.parent.name[6:]),
                        int(p.name[4:]),
                    )
                    for p in paths
                    if p.is_dir()
                )
            )
        )
        if not dates:
            logging.warning(
                f"No partitions found in {self.config.weighted_dataset_path} for M1 final training."
            )
        else:
            logging.info(f"  -> Found {len(dates)} partitions for M1 final training.")
        return dates

    # --- (維持) M2 OOF 予測を集約する関数 ---
    def _aggregate_m2_oof_predictions(self):
        logging.info(
            f"Aggregating M2 OOF predictions from {S7_M2_OOF_PREDICTIONS_TMP}..."
        )
        if not S7_M2_OOF_PREDICTIONS_TMP.exists() or not any(
            S7_M2_OOF_PREDICTIONS_TMP.iterdir()
        ):
            logging.warning("No temporary M2 OOF prediction files found to aggregate.")
            pl.DataFrame().write_parquet(S7_M2_OOF_PREDICTIONS, compression="zstd")
            logging.warning(
                f"Created empty M2 OOF prediction file at: {S7_M2_OOF_PREDICTIONS}"
            )
            return
        try:
            m2_oof_df = pl.scan_parquet(
                str(S7_M2_OOF_PREDICTIONS_TMP / "**/*.parquet")
            ).collect(streaming=True)
            m2_oof_df.sort("timestamp").write_parquet(
                S7_M2_OOF_PREDICTIONS, compression="zstd"
            )
            logging.info(
                f"Successfully aggregated {len(m2_oof_df)} M2 OOF predictions into: {S7_M2_OOF_PREDICTIONS}"
            )
        except Exception as e:
            logging.error(f"Failed to aggregate M2 OOF predictions: {e}", exc_info=True)
            pl.DataFrame().write_parquet(S7_M2_OOF_PREDICTIONS, compression="zstd")
            logging.error(
                f"Created empty M2 OOF prediction file due to aggregation error: {S7_M2_OOF_PREDICTIONS}"
            )

    # --- パフォーマンスレポート生成 (AUCロバスト化適用) ---
    def _generate_performance_report(self):
        logging.info(
            "--- Generating Final Performance Report (M1 optional, M2 primary) ---"
        )
        report = {}

        # M1 Performance (Optional)
        if self.config.m1_oof_path.exists():
            try:
                logging.info(
                    f"  -> Loading M1 OOF predictions from {self.config.m1_oof_path}..."
                )
                m1_oof_df = pl.read_parquet(self.config.m1_oof_path)
                if not m1_oof_df.is_empty():
                    logging.info("  -> Calculating M1 performance metrics...")
                    y_true_m1 = np.where(m1_oof_df["true_label"] == 1, 1, 0)
                    y_pred_m1 = m1_oof_df["prediction"].fill_nan(0.5).to_numpy()
                    w_m1 = m1_oof_df["uniqueness"].fill_nan(1.0).to_numpy()
                    del m1_oof_df
                    valid_indices_m1 = ~np.isnan(y_pred_m1) & ~np.isnan(w_m1)

                    if np.any(valid_indices_m1):
                        y_true_m1 = y_true_m1[valid_indices_m1]
                        y_pred_m1 = y_pred_m1[valid_indices_m1]
                        w_m1 = w_m1[valid_indices_m1]

                        # ★★★ 修正適用済み: AUC要件チェック ★★★
                        num_unique_classes = len(np.unique(y_true_m1))
                        if num_unique_classes < 2:
                            auc_score = float("nan")
                            logging.warning(
                                f"  -> M1 AUC SKIPPED: Only {num_unique_classes} class found in y_true. Data too sparse."
                            )
                        else:
                            # NaN/Infが含まれるとroc_auc_scoreがエラーになるため、事前にチェック
                            if not np.all(np.isfinite(w_m1)):
                                raise ValueError(
                                    "M1 sample_weight contains non-finite values during final AUC calculation."
                                )

                            auc_score = roc_auc_score(
                                y_true_m1, y_pred_m1, sample_weight=w_m1
                            )
                        # ★★★ 修正ここまで ★★★

                        report["m1_performance"] = {"auc": auc_score}
                        logging.info("  -> M1 AUC calculated.")
                    else:
                        logging.warning(
                            "No valid M1 OOF predictions found after handling NaNs."
                        )
                else:
                    logging.warning("M1 OOF prediction file is empty.")
            except Exception as e:
                logging.error(
                    f"Could not process M1 OOF predictions for report: {e}. Outputting NaN."
                )
                report["m1_performance"] = {"auc": float("nan")}

        # M2 Performance (Primary)
        if S7_M2_OOF_PREDICTIONS.exists():
            try:
                logging.info(
                    f"  -> Loading aggregated M2 OOF predictions from {S7_M2_OOF_PREDICTIONS}..."
                )
                m2_oof_df = pl.read_parquet(S7_M2_OOF_PREDICTIONS)
                m2_oof_df = m2_oof_df.filter(pl.col("true_label").is_not_null())
                if not m2_oof_df.is_empty():
                    logging.info("  -> Calculating M2 performance metrics...")
                    y_true_m2 = m2_oof_df["true_label"].to_numpy()
                    y_pred_m2 = m2_oof_df["prediction"].fill_nan(0.5).to_numpy()
                    w_m2 = m2_oof_df["uniqueness"].fill_nan(1.0).to_numpy()
                    del m2_oof_df
                    valid_indices_m2 = ~np.isnan(y_pred_m2) & ~np.isnan(w_m2)
                    if np.any(valid_indices_m2):
                        y_true_m2 = y_true_m2[valid_indices_m2]
                        y_pred_m2 = y_pred_m2[valid_indices_m2]
                        w_m2 = w_m2[valid_indices_m2]

                        # ★★★ 修正適用済み: AUC要件チェック ★★★
                        num_unique_classes = len(np.unique(y_true_m2))
                        if num_unique_classes < 2:
                            auc_score = float("nan")
                            logging.warning(
                                f"  -> M2 AUC SKIPPED: Only {num_unique_classes} class found in y_true. Data too sparse."
                            )
                        else:
                            # NaN/Infが含まれるとroc_auc_scoreがエラーになるため、事前にチェック
                            if not np.all(np.isfinite(w_m2)):
                                raise ValueError(
                                    "M2 sample_weight contains non-finite values during final AUC calculation."
                                )

                            auc_score = roc_auc_score(
                                y_true_m2, y_pred_m2, sample_weight=w_m2
                            )
                        # ★★★ 修正ここまで ★★★

                        report["m2_performance"] = {"auc": auc_score}
                        logging.info("  -> M2 AUC calculated.")
                    else:
                        logging.warning(
                            "No valid M2 OOF predictions found after handling NaNs."
                        )
                else:
                    logging.warning(
                        "Aggregated M2 OOF data was empty after filtering NaNs."
                    )
            except Exception as e:
                logging.error(
                    f"Could not process aggregated M2 OOF predictions for report: {e}. Outputting NaN."
                )
                report["m2_performance"] = {"auc": float("nan")}

        S7_MODEL_PERFORMANCE_REPORT.parent.mkdir(parents=True, exist_ok=True)
        # float('nan')をJSONセーフな文字列に変換して保存
        json_report = json.dumps(report, indent=4)
        json_report = json_report.replace(
            "NaN", "null"
        )  # float('nan')をJSONのnullとして出力
        with open(S7_MODEL_PERFORMANCE_REPORT, "w") as f:
            f.write(json_report)

        logging.info(f"Performance report saved to {S7_MODEL_PERFORMANCE_REPORT}")
        print("\nPerformance Report:")
        print(json_report)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Script 3/3: M2 CV, Final Training, Calibration, and Reporting"
    )
    parser.add_argument("--test", action="store_true", help="Run in quick test mode.")
    args = parser.parse_args()

    config = FinalTrainingConfig(test=args.test)
    if args.test:
        config.lgbm_params_m1["n_estimators"] = 10
        config.lgbm_params_m2["n_estimators"] = 10

    assembler = FinalAssembler(config)
    assembler.run()
