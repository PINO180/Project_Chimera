# /workspace/models/model_training_metalabeling_C.py
# [修正版: V5 Project Cimera 双方向ラベリング (Long/Short独立) 対応]

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
from sklearn.metrics import roc_auc_score
import joblib
from typing import List, Tuple, Dict, Any, Generator
from tqdm import tqdm
from collections import Counter

# --- プロジェクトのルートディレクトリをPythonの検索パスに追加 ---
project_root = Path(__file__).resolve().parents[1]
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

# --- 修正後のインポート文 ---
from blueprint import (
    S6_WEIGHTED_DATASET,
    S3_FEATURES_FOR_TRAINING,
    S3_SELECTED_FEATURES_DIR,
    S7_M1_OOF_PREDICTIONS_LONG,
    S7_M1_OOF_PREDICTIONS_SHORT,
    S7_META_LABELED_OOF_LONG,
    S7_META_LABELED_OOF_SHORT,
    S7_M2_OOF_PREDICTIONS_LONG,
    S7_M2_OOF_PREDICTIONS_SHORT,
    S7_M1_MODEL_LONG_PKL,
    S7_M1_MODEL_SHORT_PKL,
    S7_M2_MODEL_LONG_PKL,
    S7_M2_MODEL_SHORT_PKL,
    S7_M1_CALIBRATED_LONG,
    S7_M1_CALIBRATED_SHORT,
    S7_M2_CALIBRATED_LONG,
    S7_M2_CALIBRATED_SHORT,
    S7_MODEL_PERFORMANCE_REPORT_LONG,
    S7_MODEL_PERFORMANCE_REPORT_SHORT,
    S7_M2_OOF_PREDICTIONS_TMP_LONG,
    S7_M2_OOF_PREDICTIONS_TMP_SHORT,
)

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
    direction: str = "long"  # 'long' or 'short'
    weighted_dataset_path: Path = S6_WEIGHTED_DATASET

    # 内部で自動割り当てするため、初期値は None (init=False) に設定
    meta_labeled_oof_path: Path = field(default=None, init=False)
    m1_oof_path_directed: Path = field(default=None, init=False)

    feature_list_path: Path = S3_FEATURES_FOR_TRAINING
    n_splits: int = 5
    purge_days: int = 3
    embargo_days: int = 2
    lgbm_params_m1: Dict[str, Any] = field(
        default_factory=lambda: {
            "objective": "binary",
            "metric": "auc",
            "boosting_type": "gbdt",
            "n_estimators": 2000,
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
            "n_estimators": 2000,
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
    test: bool = False

    def __post_init__(self):
        # --- 読み込み元と出力先をBlueprintの新仕様に合わせる ---
        if self.direction == "long":
            self.meta_labeled_oof_path = S7_META_LABELED_OOF_LONG
            self.m1_oof_path_directed = S7_M1_OOF_PREDICTIONS_LONG
            self.m2_oof_predictions_tmp = S7_M2_OOF_PREDICTIONS_TMP_LONG
            self.m2_oof_predictions = S7_M2_OOF_PREDICTIONS_LONG
            self.m1_model_pkl = S7_M1_MODEL_LONG_PKL
            self.m2_model_pkl = S7_M2_MODEL_LONG_PKL
            self.m1_calibrated = S7_M1_CALIBRATED_LONG
            self.m2_calibrated = S7_M2_CALIBRATED_LONG
            self.performance_report = S7_MODEL_PERFORMANCE_REPORT_LONG
        else:
            self.meta_labeled_oof_path = S7_META_LABELED_OOF_SHORT
            self.m1_oof_path_directed = S7_M1_OOF_PREDICTIONS_SHORT
            self.m2_oof_predictions_tmp = S7_M2_OOF_PREDICTIONS_TMP_SHORT
            self.m2_oof_predictions = S7_M2_OOF_PREDICTIONS_SHORT
            self.m1_model_pkl = S7_M1_MODEL_SHORT_PKL
            self.m2_model_pkl = S7_M2_MODEL_SHORT_PKL
            self.m1_calibrated = S7_M1_CALIBRATED_SHORT
            self.m2_calibrated = S7_M2_CALIBRATED_SHORT
            self.performance_report = S7_MODEL_PERFORMANCE_REPORT_SHORT


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
        # Blueprintで追加したディレクトリをインポート済みであることを確認 (from blueprint import S3_SELECTED_FEATURES_DIR)

        self.config = config
        self.direction = self.config.direction

        # --- パターンB: M1とM2それぞれの専用特徴量リストを動的に読み込む ---
        m1_feature_path = S3_SELECTED_FEATURES_DIR / f"m1_{self.direction}_features.txt"
        m2_feature_path = S3_SELECTED_FEATURES_DIR / f"m2_{self.direction}_features.txt"

        self.features_base: List[str] = self._load_features(m1_feature_path)
        self.features_m2: List[str] = self._load_features(m2_feature_path)

        self.partitions_m1_final = self._discover_partitions_for_m1_final_train()
        self.partitions_m2 = self._discover_partitions()

        self.scale_pos_weight_m1 = self._calculate_scale_pos_weight_m1()
        self.config.lgbm_params_m1["scale_pos_weight"] = self.scale_pos_weight_m1
        logging.info(
            f"[{self.direction.upper()}] Using scale_pos_weight for M1: {self.scale_pos_weight_m1:.4f}"
        )

        self.scale_pos_weight_m2 = self._calculate_scale_pos_weight_m2()
        self.config.lgbm_params_m2["scale_pos_weight"] = self.scale_pos_weight_m2
        logging.info(
            f"[{self.direction.upper()}] Using scale_pos_weight for M2: {self.scale_pos_weight_m2:.4f}"
        )

    def _load_features(self, feature_path: Path) -> List[str]:
        logging.info(
            f"[{self.direction.upper()}] Loading dedicated feature list from {feature_path.name}..."
        )
        with open(feature_path, "r") as f:
            raw_features = [line.strip() for line in f if line.strip()]

        exclude_exact = {
            "timestamp",
            "timeframe",
            "t1",
            "label",
            "uniqueness",
            "payoff_ratio",
            "pt_multiplier",
            "sl_multiplier",
            "direction",
            "exit_type",
            "first_ex_reason_int",
            "atr_value",
            "calculated_body_ratio",
            "fallback_vol",
            "open",
            "high",
            "low",
            "close",
            "meta_label",
            "is_trigger",
        }

        features = []
        for col in raw_features:
            if col in exclude_exact or col.startswith("is_trigger_on"):
                continue
            features.append(col)

        logging.info(f"   -> Loaded {len(features)} valid base features.")
        return features

    def _discover_partitions(self) -> List[datetime.date]:
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
        return dates

    def _discover_partitions_for_m1_final_train(self) -> List[datetime.date]:
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
        return dates

    def _filter_dataframe(self, df: pl.DataFrame, is_m1: bool = False) -> pl.DataFrame:
        """V5仕様: M1学習時のみ、トリガー(is_trigger=1)発火行に絞り込む"""
        if is_m1 and "is_trigger" in df.columns:
            df = df.filter(pl.col("is_trigger") == 1)
        return df

    def _calculate_scale_pos_weight_m1(self) -> float:
        counts = Counter({0: 0, 1: 0})
        if not self.partitions_m1_final:
            return 1.0

        for partition_date in tqdm(
            self.partitions_m1_final,
            desc=f"[{self.direction.upper()}] Scale weight (M1)",
        ):
            p_path_glob = str(
                self.config.weighted_dataset_path
                / f"year={partition_date.year}/month={partition_date.month}/day={partition_date.day}/*.parquet"
            )
            try:
                target_col = f"label_{self.direction}"

                # 実際のスキーマに依存するため、エラーを避けるためにscanを使用するか、全列読んでから絞る
                df_lazy = pl.scan_parquet(p_path_glob)
                df = df_lazy.collect()

                df = self._filter_dataframe(df, is_m1=True)
                if df.is_empty():
                    continue

                binary_labels = df.select(
                    pl.when(pl.col(target_col) == 1)
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
            except Exception:
                continue

        count_neg, count_pos = counts[0], counts[1]
        if count_pos == 0 or count_neg == 0:
            return 1.0
        return count_neg / count_pos

    def _calculate_scale_pos_weight_m2(self) -> float:
        counts = Counter({0: 0, 1: 0})
        if not self.partitions_m2:
            return 1.0

        for partition_date in tqdm(
            self.partitions_m2, desc=f"[{self.direction.upper()}] Scale weight (M2)"
        ):
            p_path_glob = str(
                self.config.meta_labeled_oof_path
                / f"year={partition_date.year}/month={partition_date.month}/day={partition_date.day}/*.parquet"
            )
            try:
                df = pl.read_parquet(p_path_glob)
                df = self._filter_dataframe(df, is_m1=False)
                df = df.filter(pl.col("meta_label").is_not_null())
                if df.is_empty():
                    continue

                counts_in_partition = df["meta_label"].value_counts()

                pos_count_df = counts_in_partition.filter(
                    pl.col("meta_label") == 1
                ).select(pl.col("count"))
                pos_count = pos_count_df.item() if not pos_count_df.is_empty() else 0

                neg_count_df = counts_in_partition.filter(
                    pl.col("meta_label") == 0
                ).select(pl.col("count"))
                neg_count = neg_count_df.item() if not neg_count_df.is_empty() else 0

                counts[1] += pos_count
                counts[0] += neg_count
            except Exception:
                continue

        count_neg, count_pos = counts[0], counts[1]
        if count_pos == 0 or count_neg == 0:
            return 1.0
        return count_neg / count_pos

    def run(self):
        logging.info(
            f"[{self.direction.upper()}] ### Script 3/3: M2 CV, Final Training, Calibration, and Reporting ###"
        )

        # M2 CV OOF用の一時ディレクトリ作成/クリーンアップ
        if not self.config.m2_oof_predictions_tmp.exists() or not any(
            self.config.m2_oof_predictions_tmp.iterdir()
        ):
            if self.config.m2_oof_predictions_tmp.exists():
                logging.info(
                    f"[{self.direction.upper()}] Cleaning existing M2 OOF temp directory: {self.config.m2_oof_predictions_tmp}"
                )
                shutil.rmtree(self.config.m2_oof_predictions_tmp)
            self.config.m2_oof_predictions_tmp.mkdir(parents=True)
            self._train_m2_cv_and_write_to_disk()
        else:
            logging.warning(
                f"[{self.direction.upper()}] --- SKIPPING M2 Cross-Validation: Temporary OOF files already exist. ---"
            )

        # 最終モデル訓練と確率較正
        self._train_and_calibrate_final_models()

        # OOF予測結果の集約とレポート作成 (パート3で定義)
        self._aggregate_m2_oof_predictions()
        self._generate_performance_report()

        if self.config.m2_oof_predictions_tmp.exists():
            logging.info(
                f"[{self.direction.upper()}] Cleaning up temporary directory: {self.config.m2_oof_predictions_tmp}"
            )
            shutil.rmtree(self.config.m2_oof_predictions_tmp)

    def _train_m2_cv_and_write_to_disk(self):
        logging.info(
            f"[{self.direction.upper()}] --- Starting BATCH Training for M2 (Meta) CV ---"
        )
        kfold = PartitionPurgedKFold(
            self.config.n_splits, self.config.purge_days, self.config.embargo_days
        )
        partitions_to_use = self.partitions_m2
        if not partitions_to_use:
            logging.error(
                f"[{self.direction.upper()}] No partitions available for M2 CV. Aborting CV."
            )
            return

        for i, (train_dates, val_dates) in enumerate(kfold.split(partitions_to_use)):
            logging.info(
                f"[{self.direction.upper()}]  [M2 (Meta)] Fold {i + 1}/{self.config.n_splits}..."
            )

            if self.config.test:
                logging.warning(
                    f"[{self.direction.upper()}] --- TEST MODE: Limiting M2 CV Fold {i + 1} partitions. ---"
                )
                train_dates = train_dates[:5]
                val_dates = val_dates[:5]

            X_train_list, y_train_list, w_train_list = [], [], []

            if len(train_dates) > 0:
                for p_date in tqdm(
                    train_dates,
                    desc=f"[{self.direction.upper()}]  Loading M2 Train Fold {i + 1}",
                ):
                    p_path_glob = str(
                        self.config.meta_labeled_oof_path
                        / f"year={p_date.year}/month={p_date.month}/day={p_date.day}/*.parquet"
                    )
                    try:
                        df_chunk = pl.read_parquet(p_path_glob)
                        df_chunk = self._filter_dataframe(df_chunk, is_m1=False)
                    except Exception:
                        continue

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

            model: lgb.Booster = None
            if len(X_train_list) > 0:
                try:
                    X_train = np.concatenate(X_train_list)
                    y_train = np.concatenate(y_train_list)
                    w_train = np.concatenate(w_train_list)

                    train_params = self.config.lgbm_params_m2.copy()
                    n_estimators = train_params.pop("n_estimators", 1000)

                    logging.info(
                        f"[{self.direction.upper()}]    -> Training M2 model on {len(X_train)} samples..."
                    )
                    model = lgb.train(
                        train_params,
                        lgb.Dataset(X_train, label=y_train, weight=w_train),
                        num_boost_round=n_estimators,
                    )
                except Exception as fit_error:
                    logging.error(
                        f"[{self.direction.upper()}] Error fitting M2 model: {fit_error}"
                    )

            if model is None:
                logging.warning(
                    f"[{self.direction.upper()}] M2 model for Fold {i + 1} was not trained. Skipping prediction."
                )
                continue

            for p_date in tqdm(
                val_dates,
                desc=f"[{self.direction.upper()}]  Predicting M2 Fold {i + 1}",
            ):
                p_path_glob = str(
                    self.config.meta_labeled_oof_path
                    / f"year={p_date.year}/month={p_date.month}/day={p_date.day}/*.parquet"
                )
                try:
                    df_chunk = pl.read_parquet(p_path_glob)
                    df_chunk = self._filter_dataframe(df_chunk, is_m1=False)
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
                    logging.error(
                        f"[{self.direction.upper()}] Error predicting M2 model: {pred_error}"
                    )
                    predictions = np.full(len(df_chunk), np.nan)

                oof_df = pl.DataFrame(
                    {
                        "timestamp": df_chunk["timestamp"],
                        "prediction": predictions,
                        "true_label": df_chunk["meta_label"],
                        "uniqueness": df_chunk["uniqueness"],
                    }
                )

                output_partition_dir = (
                    self.config.m2_oof_predictions_tmp
                    / f"year={p_date.year}/month={p_date.month}/day={p_date.day}"
                )
                output_partition_dir.mkdir(parents=True, exist_ok=True)
                oof_df.write_parquet(
                    output_partition_dir / "data.parquet", compression="zstd"
                )

            del model
            gc.collect()

    def _train_and_calibrate_final_models(self):
        logging.info(
            f"[{self.direction.upper()}] --- Training and Calibrating Final Models ---"
        )

        m1_model = self._ensure_model_trained(
            "M1",
            self.config.m1_model_pkl,
            is_m2=False,
            partitions_to_train=self.partitions_m1_final,
            lgbm_params=self.config.lgbm_params_m1,
        )

        m2_model = self._ensure_model_trained(
            "M2",
            self.config.m2_model_pkl,
            is_m2=True,
            partitions_to_train=self.partitions_m2,
            lgbm_params=self.config.lgbm_params_m2,
        )

        logging.info(
            f"[{self.direction.upper()}]   - Calibrating models (Manual, Memory-Safe Mode)..."
        )
        calib_dates = self.partitions_m2[
            -len(self.partitions_m2) // self.config.n_splits :
        ]
        if calib_dates:
            self._manual_calibrate(
                "M1", m1_model, self.config.m1_calibrated, calib_dates, is_m2=False
            )
            self._manual_calibrate(
                "M2", m2_model, self.config.m2_calibrated, calib_dates, is_m2=True
            )
        else:
            logging.warning(
                f"[{self.direction.upper()}] No partitions available for calibration. Skipping calibration."
            )

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
                    f"[{self.direction.upper()}] No partitions available to train the final {model_name} model."
                )
            model = self._train_single_model(
                f"{model_name} (Final)",
                "meta_label" if is_m2 else f"label_{self.direction}",
                is_m2,
                partitions_to_train,
                lgbm_params,
            )
            joblib.dump(model, model_path)
            logging.info(
                f"[{self.direction.upper()}]   -> {model_name} model saved to {model_path}."
            )
        else:
            logging.warning(
                f"[{self.direction.upper()}] --- SKIPPING {model_name} Final Training: Model file already exists at {model_path}. ---"
            )
            model = joblib.load(model_path)
        return model

    def _train_single_model(
        self,
        model_name: str,
        target_col: str,
        is_m2: bool,
        partitions_to_train: List[datetime.date],
        lgbm_params: Dict[str, Any],
    ) -> lgb.Booster:
        logging.info(
            f"[{self.direction.upper()}]   - Training {model_name} on {len(partitions_to_train)} partitions (BATCH Mode)..."
        )

        model: lgb.Booster = None
        input_path = (
            self.config.meta_labeled_oof_path
            if is_m2
            else self.config.weighted_dataset_path
        )

        X_list, y_list, w_list = [], [], []

        for p_date in tqdm(
            partitions_to_train,
            desc=f"[{self.direction.upper()}]   Loading Data for {model_name}",
        ):
            p_path_glob = str(
                input_path
                / f"year={p_date.year}/month={p_date.month}/day={p_date.day}/*.parquet"
            )
            try:
                df_chunk = pl.read_parquet(p_path_glob)
                df_chunk = self._filter_dataframe(df_chunk, is_m1=not is_m2)
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

            weight_col = "uniqueness" if is_m2 else f"uniqueness_{self.direction}"

            X_chunk = df_chunk.select(features_to_use).fill_null(0).to_numpy()
            y_chunk = df_chunk[target_col].to_numpy()
            w_chunk = df_chunk[weight_col].to_numpy()

            if not is_m2:
                y_chunk = np.where(y_chunk == 1, 1, 0)

            X_list.append(X_chunk)
            y_list.append(y_chunk)
            w_list.append(w_chunk)

        if not X_list:
            raise RuntimeError(
                f"[{self.direction.upper()}] No data found for {model_name}."
            )

        try:
            X_train = np.concatenate(X_list)
            y_train = np.concatenate(y_list)
            w_train = np.concatenate(w_list)

            del X_list, y_list, w_list
            gc.collect()

            train_params = lgbm_params.copy()
            n_estimators = train_params.pop("n_estimators", 1000)

            logging.info(
                f"[{self.direction.upper()}]     -> fitting {model_name} on {len(X_train)} samples..."
            )

            model = lgb.train(
                train_params,
                lgb.Dataset(X_train, label=y_train, weight=w_train),
                num_boost_round=n_estimators,
            )

            del X_train, y_train, w_train
            gc.collect()

        except Exception as fit_error:
            raise RuntimeError(
                f"[{self.direction.upper()}] Failed to train {model_name}: {fit_error}"
            )

        return model

    def _manual_calibrate(
        self,
        model_name: str,
        model: lgb.Booster,
        save_path: Path,
        dates: List[datetime.date],
        is_m2: bool,
    ):
        if not save_path.exists():
            logging.info(
                f"[{self.direction.upper()}]     -> Calibrating {model_name} model..."
            )
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
                        f"[{self.direction.upper()}]   -> Calibrated {model_name} (IsotonicRegressor) saved to {save_path}."
                    )
                except Exception as calib_error:
                    logging.error(
                        f"[{self.direction.upper()}] Failed to fit calibrator for {model_name}: {calib_error}",
                        exc_info=False,
                    )
            else:
                logging.warning(
                    f"[{self.direction.upper()}] No valid predictions gathered for {model_name} calibration. Skipping."
                )
        else:
            logging.warning(
                f"[{self.direction.upper()}] --- SKIPPING {model_name} Calibration: Calibrated model already exists. ---"
            )

    def _gather_predictions_for_calibration(
        self,
        model_name: str,
        dates: List[datetime.date],
        model: lgb.Booster,
        is_m2: bool,
    ) -> Tuple[np.ndarray | None, np.ndarray | None, np.ndarray | None]:
        all_preds, all_labels, all_weights = [], [], []

        base_path = (
            self.config.meta_labeled_oof_path
            if is_m2
            else self.config.weighted_dataset_path
        )

        for p_date in tqdm(
            dates,
            desc=f"[{self.direction.upper()}]   Gathering {model_name} calib predictions",
        ):
            p_path_glob = str(
                base_path
                / f"year={p_date.year}/month={p_date.month}/day={p_date.day}/*.parquet"
            )

            # ここで動的にカラム名を決定
            actual_target_col = "meta_label" if is_m2 else f"label_{self.direction}"
            weight_col = "uniqueness" if is_m2 else f"uniqueness_{self.direction}"

            try:
                df_chunk = pl.read_parquet(p_path_glob)
                df_chunk = self._filter_dataframe(df_chunk, is_m1=not is_m2)

                if not is_m2 and actual_target_col not in df_chunk.columns:
                    logging.warning(
                        f"[{self.direction.upper()}] '{actual_target_col}' column not found in {p_date} for M1 calibration. Skipping chunk."
                    )
                    continue
            except Exception:
                continue

            if is_m2:
                df_chunk = df_chunk.filter(pl.col("meta_label").is_not_null())

            if df_chunk.is_empty():
                continue

            features_to_use = self.features_m2 if is_m2 else self.features_base

            missing_features = [f for f in features_to_use if f not in df_chunk.columns]
            if missing_features:
                logging.warning(
                    f"[{self.direction.upper()}] Calibration partition {p_date} missing features for {model_name}. Skipping chunk."
                )
                continue

            X = df_chunk.select(features_to_use).fill_null(0).to_numpy()

            try:
                predictions = model.predict(X)
            except Exception as pred_error:
                logging.error(
                    f"[{self.direction.upper()}] Error predicting for calibration ({model_name}, {p_date}): {pred_error}",
                    exc_info=False,
                )
                continue

            labels = df_chunk[actual_target_col].to_numpy()
            if not is_m2:
                labels = np.where(labels == 1, 1, 0)

            all_preds.append(predictions)
            all_labels.append(labels)
            all_weights.append(df_chunk[weight_col].to_numpy())

        if not all_preds:
            return None, None, None
        try:
            return (
                np.concatenate(all_preds),
                np.concatenate(all_labels),
                np.concatenate(all_weights),
            )
        except ValueError as concat_error:
            logging.error(
                f"[{self.direction.upper()}] Error concatenating calibration data for {model_name}: {concat_error}"
            )
            return None, None, None

    def _aggregate_m2_oof_predictions(self):
        logging.info(
            f"[{self.direction.upper()}] Aggregating M2 OOF predictions from {self.config.m2_oof_predictions_tmp}..."
        )
        if not self.config.m2_oof_predictions_tmp.exists() or not any(
            self.config.m2_oof_predictions_tmp.iterdir()
        ):
            logging.warning(
                f"[{self.direction.upper()}] No temporary M2 OOF prediction files found to aggregate."
            )
            pl.DataFrame().write_parquet(
                self.config.m2_oof_predictions, compression="zstd"
            )
            return
        try:
            m2_oof_df = pl.scan_parquet(
                str(self.config.m2_oof_predictions_tmp / "**/*.parquet")
            ).collect(streaming=True)
            m2_oof_df.sort("timestamp").write_parquet(
                self.config.m2_oof_predictions, compression="zstd"
            )
            logging.info(
                f"[{self.direction.upper()}] Successfully aggregated {len(m2_oof_df)} M2 OOF predictions into: {self.config.m2_oof_predictions}"
            )
        except Exception as e:
            logging.error(
                f"[{self.direction.upper()}] Failed to aggregate M2 OOF predictions: {e}",
                exc_info=True,
            )
            pl.DataFrame().write_parquet(
                self.config.m2_oof_predictions, compression="zstd"
            )

    def _generate_performance_report(self):
        logging.info(
            f"[{self.direction.upper()}] --- Generating Final Performance Report ---"
        )
        report = {}

        # M1 OOFのパフォーマンス (もし該当方向のM1 OOFファイルが存在すれば評価)
        m1_oof_path_directed = self.config.m1_oof_path_directed
        if m1_oof_path_directed.exists():
            try:
                logging.info(
                    f"[{self.direction.upper()}]   -> Loading M1 OOF predictions from {m1_oof_path_directed}..."
                )
                m1_oof_df = pl.read_parquet(m1_oof_path_directed)
                if not m1_oof_df.is_empty():
                    y_true_m1 = np.where(m1_oof_df["true_label"] == 1, 1, 0)
                    y_pred_m1 = m1_oof_df["prediction"].fill_nan(0.5).to_numpy()
                    w_m1 = m1_oof_df["uniqueness"].fill_nan(1.0).to_numpy()
                    del m1_oof_df
                    valid_indices_m1 = ~np.isnan(y_pred_m1) & ~np.isnan(w_m1)

                    if np.any(valid_indices_m1):
                        y_true_m1 = y_true_m1[valid_indices_m1]
                        y_pred_m1 = y_pred_m1[valid_indices_m1]
                        w_m1 = w_m1[valid_indices_m1]

                        if len(np.unique(y_true_m1)) < 2:
                            report["m1_performance"] = {"auc": float("nan")}
                        else:
                            report["m1_performance"] = {
                                "auc": roc_auc_score(
                                    y_true_m1, y_pred_m1, sample_weight=w_m1
                                )
                            }
                            logging.info(
                                f"[{self.direction.upper()}]   -> M1 AUC calculated."
                            )
            except Exception as e:
                logging.error(
                    f"[{self.direction.upper()}] Could not process M1 OOF predictions: {e}"
                )
                report["m1_performance"] = {"auc": float("nan")}

        # M2 Performance
        if self.config.m2_oof_predictions.exists():
            try:
                logging.info(
                    f"[{self.direction.upper()}]   -> Loading aggregated M2 OOF predictions from {self.config.m2_oof_predictions}..."
                )
                m2_oof_df = pl.read_parquet(self.config.m2_oof_predictions)
                m2_oof_df = m2_oof_df.filter(pl.col("true_label").is_not_null())
                if not m2_oof_df.is_empty():
                    y_true_m2 = m2_oof_df["true_label"].to_numpy()
                    y_pred_m2 = m2_oof_df["prediction"].fill_nan(0.5).to_numpy()
                    w_m2 = m2_oof_df["uniqueness"].fill_nan(1.0).to_numpy()
                    del m2_oof_df
                    valid_indices_m2 = ~np.isnan(y_pred_m2) & ~np.isnan(w_m2)

                    if np.any(valid_indices_m2):
                        y_true_m2 = y_true_m2[valid_indices_m2]
                        y_pred_m2 = y_pred_m2[valid_indices_m2]
                        w_m2 = w_m2[valid_indices_m2]

                        if len(np.unique(y_true_m2)) < 2:
                            report["m2_performance"] = {"auc": float("nan")}
                        else:
                            report["m2_performance"] = {
                                "auc": roc_auc_score(
                                    y_true_m2, y_pred_m2, sample_weight=w_m2
                                )
                            }
                            logging.info(
                                f"[{self.direction.upper()}]   -> M2 AUC calculated."
                            )
            except Exception as e:
                logging.error(
                    f"[{self.direction.upper()}] Could not process aggregated M2 OOF predictions: {e}"
                )
                report["m2_performance"] = {"auc": float("nan")}

        self.config.performance_report.parent.mkdir(parents=True, exist_ok=True)
        json_report = json.dumps(report, indent=4).replace("NaN", "null")
        with open(self.config.performance_report, "w") as f:
            f.write(json_report)

        logging.info(
            f"[{self.direction.upper()}] Performance report saved to {self.config.performance_report}"
        )
        print(f"\n[{self.direction.upper()}] Performance Report:")
        print(json_report)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Script 3/3: M2 CV, Final Training, Calibration, and Reporting (Project Cimera V5)"
    )
    parser.add_argument("--test", action="store_true", help="Run in quick test mode.")
    args = parser.parse_args()

    for direction in ["long", "short"]:
        print("\n" + "=" * 60)
        print(
            f"### PROJECT FORGE: STARTING PROCESSING FOR DIRECTION: {direction.upper()} ###"
        )
        print("=" * 60 + "\n")

        config = FinalTrainingConfig(direction=direction, test=args.test)
        if args.test:
            config.lgbm_params_m1["n_estimators"] = 10
            config.lgbm_params_m2["n_estimators"] = 10

        assembler = FinalAssembler(config)
        assembler.run()

    logging.info(
        "\n"
        + "=" * 60
        + "\n### PROJECT FORGE: ALL STAGES (LONG & SHORT) COMPLETED! ###\n"
        + "=" * 60
    )
