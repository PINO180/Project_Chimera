# /workspace/models/model_training_metalabeling_C.py

import sys
from pathlib import Path
import logging
import argparse
import json
import warnings
import datetime
import shutil
from dataclasses import dataclass, field

import polars as pl
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import BaseCrossValidator
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score
import joblib
from typing import List, Tuple, Dict, Any, Generator
from tqdm import tqdm

# (インポート、パス設定、Configクラス、CVクラスは前回と同じ)
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
S7_M2_OOF_PREDICTIONS = S7_MODELS / "m2_oof_predictions.parquet"
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
    n_splits: int = 5
    purge_days: int = 3
    embargo_days: int = 2
    lgbm_params: Dict[str, Any] = field(
        default_factory=lambda: {
            "objective": "binary",
            "metric": "auc",
            "boosting_type": "gbdt",
            "n_estimators": 1000,
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
    # (init, _load_features, _discover_partitions は変更なし)
    def __init__(self, config: FinalTrainingConfig):
        self.config = config
        self.features: List[str] = self._load_features()
        self.partitions = self._discover_partitions()

    def _load_features(self) -> List[str]:
        logging.info(f"Loading feature list from {self.config.feature_list_path}...")
        with open(self.config.feature_list_path, "r") as f:
            features = [line.strip() for line in f if line.strip()]
        return features

    def _discover_partitions(self) -> List[datetime.date]:
        logging.info(
            "Discovering all physical partitions from the meta-labeled dataset..."
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
                )
            )
        )
        logging.info(f"  -> Discovered {len(dates)} daily partitions for M2 training.")
        return dates

    # (run, _train_m2_cv_and_write_to_disk は変更なし)
    def run(self):
        logging.info("### Script 3/3: M2 CV, Final Training, and Reporting ###")
        if not S7_M2_OOF_PREDICTIONS_TMP.exists() or not any(
            S7_M2_OOF_PREDICTIONS_TMP.iterdir()
        ):
            if S7_M2_OOF_PREDICTIONS_TMP.exists():
                shutil.rmtree(S7_M2_OOF_PREDICTIONS_TMP)
            S7_M2_OOF_PREDICTIONS_TMP.mkdir(parents=True)
            self._train_m2_cv_and_write_to_disk()
        else:
            logging.warning(
                "--- SKIPPING M2 Cross-Validation: Temporary OOF files already exist. ---"
            )
        self._train_and_calibrate_final_models()
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

    def _train_m2_cv_and_write_to_disk(self):
        logging.info(
            f"--- Starting Disk-Based Sequential Partition Training for M2 (Meta) ---"
        )
        kfold = PartitionPurgedKFold(
            self.config.n_splits, self.config.purge_days, self.config.embargo_days
        )
        for i, (train_dates, val_dates) in enumerate(kfold.split(self.partitions)):
            logging.info(f"  [M2 (Meta)] Fold {i + 1}/{self.config.n_splits}...")
            model = lgb.LGBMClassifier(**self.config.lgbm_params)
            is_first_chunk = True
            for p_date in tqdm(train_dates, desc=f"  Training Fold {i + 1}"):
                p_path = str(
                    self.config.meta_labeled_oof_path
                    / f"year={p_date.year}/month={p_date.month}/day={p_date.day}/*.parquet"
                )
                try:
                    df_chunk = pl.read_parquet(p_path)
                except Exception:
                    continue
                # ▼▼▼ 門番をここにも配置（CV学習時） ▼▼▼
                df_chunk = df_chunk.filter(pl.col("meta_label").is_not_null())
                if df_chunk.is_empty():
                    continue
                features_to_use = self.features + ["m1_pred_proba"]
                X_chunk, y_chunk, w_chunk = (
                    df_chunk.select(features_to_use).to_numpy(),
                    df_chunk["meta_label"].to_numpy(),
                    df_chunk["uniqueness"].to_numpy(),
                )
                model.fit(
                    X_chunk,
                    y_chunk,
                    sample_weight=w_chunk,
                    init_model=None if is_first_chunk else model.booster_,
                )
                is_first_chunk = False
            if is_first_chunk:
                continue
            for p_date in tqdm(val_dates, desc=f"  Predicting Fold {i + 1}"):
                p_path = str(
                    self.config.meta_labeled_oof_path
                    / f"year={p_date.year}/month={p_date.month}/day={p_date.day}/*.parquet"
                )
                try:
                    df_chunk = pl.read_parquet(p_path)
                except Exception:
                    continue
                if df_chunk.is_empty():
                    continue
                predictions = model.predict_proba(
                    df_chunk.select(features_to_use).to_numpy()
                )[:, 1]
                oof_df = pl.DataFrame(
                    {
                        "timestamp": df_chunk["timestamp"],
                        "prediction": predictions,
                        "true_label": df_chunk["meta_label"],
                        "uniqueness": df_chunk["uniqueness"],
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

    # (_train_and_calibrate_final_models, _manual_calibrate, _gather_predictions... は変更なし)
    def _train_and_calibrate_final_models(self):
        logging.info("--- Training and Calibrating Final Models ---")
        m1_model = self._ensure_model_trained("M1", S7_M1_MODEL_PKL, is_m2=False)
        m2_model = self._ensure_model_trained("M2", S7_M2_MODEL_PKL, is_m2=True)
        logging.info("  - Calibrating models (Manual, Memory-Safe Mode)...")
        calib_dates = self.partitions[-len(self.partitions) // self.config.n_splits :]
        self._manual_calibrate(
            "M1", m1_model, S7_M1_CALIBRATED, calib_dates, is_m2=False
        )
        self._manual_calibrate(
            "M2", m2_model, S7_M2_CALIBRATED, calib_dates, is_m2=True
        )

    def _manual_calibrate(
        self,
        model_name: str,
        model: lgb.LGBMClassifier,
        save_path: Path,
        dates: List[datetime.date],
        is_m2: bool,
    ):
        if not save_path.exists():
            logging.info(f"    -> Calibrating {model_name} model...")
            y_pred, y_true, weights = self._gather_predictions_for_calibration(
                model_name, dates, model, is_m2
            )
            if y_pred is not None:
                calibrator = IsotonicRegression(out_of_bounds="clip")
                calibrator.fit(y_pred, y_true, sample_weight=weights)
                joblib.dump(calibrator, save_path)
                logging.info(
                    f"  -> Calibrated {model_name} (IsotonicRegressor) saved to {save_path}."
                )
        else:
            logging.warning(
                f"--- SKIPPING {model_name} Calibration: Calibrated model already exists. ---"
            )

    def _gather_predictions_for_calibration(
        self,
        model_name: str,
        dates: List[datetime.date],
        model: lgb.LGBMClassifier,
        is_m2: bool,
    ):
        all_preds, all_labels, all_weights = [], [], []
        base_path = (
            self.config.meta_labeled_oof_path
            if is_m2
            else self.config.weighted_dataset_path
        )
        for p_date in tqdm(dates, desc=f"  Gathering {model_name} calib predictions"):
            p_path = str(
                base_path
                / f"year={p_date.year}/month={p_date.month}/day={p_date.day}/*.parquet"
            )
            try:
                df_chunk = pl.read_parquet(p_path)
            except Exception:
                continue
            if df_chunk.is_empty():
                continue
            features_to_use = (
                self.features + ["m1_pred_proba"] if is_m2 else self.features
            )
            target_col = "meta_label" if is_m2 else "label"
            if target_col not in df_chunk.columns:
                continue
            if is_m2:
                df_chunk = df_chunk.filter(pl.col(target_col).is_not_null())
            if df_chunk.is_empty():
                continue
            X = df_chunk.select(features_to_use)
            predictions = model.predict_proba(X.to_numpy())[:, 1]
            labels = df_chunk[target_col].to_numpy()
            if not is_m2:
                labels = np.where(labels == 1, 1, 0)
            all_preds.append(predictions)
            all_labels.append(labels)
            all_weights.append(df_chunk["uniqueness"].to_numpy())
        if not all_preds:
            return None, None, None
        return (
            np.concatenate(all_preds),
            np.concatenate(all_labels),
            np.concatenate(all_weights),
        )

    def _ensure_model_trained(
        self, model_name: str, model_path: Path, is_m2: bool
    ) -> lgb.LGBMClassifier:
        if not model_path.exists():
            partitions_to_train = (
                self._discover_partitions_for_m1_final_train()
                if not is_m2
                else self.partitions
            )
            if self.config.test:
                partitions_to_train = partitions_to_train[:5]
            model = self._train_single_model(
                f"{model_name} (Final)",
                "meta_label" if is_m2 else "label",
                is_m2,
                partitions_to_train,
            )
            joblib.dump(model, model_path)
            logging.info(f"  -> {model_name} model saved.")
        else:
            logging.warning(
                f"--- SKIPPING {model_name} Final Training: Model file already exists. ---"
            )
            model = joblib.load(model_path)
        return model

    # ▼▼▼ 変更点: 門番をここに配置（最終学習時） ▼▼▼
    def _train_single_model(
        self,
        model_name: str,
        target_col: str,
        is_m2: bool,
        partitions_to_train: List[datetime.date],
    ) -> lgb.LGBMClassifier:
        logging.info(
            f"  - Training {model_name} on {len(partitions_to_train)} partitions..."
        )
        model = lgb.LGBMClassifier(**self.config.lgbm_params)
        is_first_chunk = True
        input_path = (
            self.config.meta_labeled_oof_path
            if is_m2
            else self.config.weighted_dataset_path
        )

        for p_date in tqdm(partitions_to_train, desc=f"  Training {model_name}"):
            p_path = str(
                input_path
                / f"year={p_date.year}/month={p_date.month}/day={p_date.day}/*.parquet"
            )
            try:
                df_chunk = pl.read_parquet(p_path)
            except Exception:
                continue

            # --- ここが門番です！ ---
            if is_m2:
                df_chunk = df_chunk.filter(pl.col(target_col).is_not_null())

            if df_chunk.is_empty():
                continue

            features_to_use = (
                self.features + ["m1_pred_proba"] if is_m2 else self.features
            )
            X_chunk, y_chunk, w_chunk = (
                df_chunk.select(features_to_use).to_numpy(),
                df_chunk[target_col].to_numpy(),
                df_chunk["uniqueness"].to_numpy(),
            )
            if not is_m2:
                y_chunk = np.where(y_chunk == 1, 1, 0)
            model.fit(
                X_chunk,
                y_chunk,
                sample_weight=w_chunk,
                init_model=None if is_first_chunk else model.booster_,
            )
            is_first_chunk = False
        return model

    # ... (_discover_partitions_for_m1_final_train, _generate_performance_report, mainブロック は変更なし) ...
    def _discover_partitions_for_m1_final_train(self) -> List[datetime.date]:
        paths = self.config.weighted_dataset_path.glob("year=*/month=*/day=*")
        return sorted(
            list(
                set(
                    datetime.date(
                        int(p.parent.parent.name[5:]),
                        int(p.parent.name[6:]),
                        int(p.name[4:]),
                    )
                    for p in paths
                )
            )
        )

    def _generate_performance_report(self):
        logging.info(
            "--- Generating Final Performance Report (Memory-Safe Direct Load Mode) ---"
        )
        report = {}

        # --- M1 Performance ---
        if self.config.m1_oof_path.exists():
            try:
                logging.info("  -> Loading M1 OOF predictions...")
                m1_oof_df = pl.read_parquet(self.config.m1_oof_path)

                logging.info("  -> Converting M1 data to NumPy and freeing memory...")
                y_true_m1 = np.where(m1_oof_df["true_label"] == 1, 1, 0)
                y_pred_m1 = m1_oof_df["prediction"].to_numpy()
                w_m1 = m1_oof_df["uniqueness"].to_numpy()

                # ここが最重要：巨大なDataFrameをメモリから完全に解放
                del m1_oof_df

                logging.info("  -> Calculating M1 performance metrics...")
                report["m1_performance"] = {
                    "auc": roc_auc_score(y_true_m1, y_pred_m1, sample_weight=w_m1),
                    "precision": precision_score(
                        y_true_m1, y_pred_m1 > 0.5, sample_weight=w_m1
                    ),
                    "recall": recall_score(
                        y_true_m1, y_pred_m1 > 0.5, sample_weight=w_m1
                    ),
                    "f1": f1_score(y_true_m1, y_pred_m1 > 0.5, sample_weight=w_m1),
                }
                logging.info("  -> M1 performance calculated successfully.")

            except Exception as e:
                logging.error(
                    f"Could not process M1 OOF predictions: {e}", exc_info=True
                )

        # --- M2 Performance ---
        if S7_M2_OOF_PREDICTIONS_TMP.exists() and any(
            S7_M2_OOF_PREDICTIONS_TMP.iterdir()
        ):
            try:
                logging.info("  -> Loading M2 OOF predictions...")
                # M2はパーティション化されているのでscan + collectが適切
                m2_oof_df = pl.scan_parquet(
                    str(S7_M2_OOF_PREDICTIONS_TMP / "**/*.parquet")
                ).collect(streaming=True)

                # M2のレポート生成にも門番を配置
                m2_oof_df = m2_oof_df.filter(pl.col("true_label").is_not_null())

                if not m2_oof_df.is_empty():
                    logging.info(
                        "  -> Converting M2 data to NumPy and freeing memory..."
                    )
                    y_true_m2 = m2_oof_df["true_label"].to_numpy()
                    y_pred_m2 = m2_oof_df["prediction"].to_numpy()
                    w_m2 = m2_oof_df["uniqueness"].to_numpy()

                    del m2_oof_df

                    logging.info("  -> Calculating M2 performance metrics...")
                    report["m2_performance"] = {
                        "auc": roc_auc_score(y_true_m2, y_pred_m2, sample_weight=w_m2),
                        "precision": precision_score(
                            y_true_m2, y_pred_m2 > 0.5, sample_weight=w_m2
                        ),
                        "recall": recall_score(
                            y_true_m2, y_pred_m2 > 0.5, sample_weight=w_m2
                        ),
                        "f1": f1_score(y_true_m2, y_pred_m2 > 0.5, sample_weight=w_m2),
                    }
                    logging.info("  -> M2 performance calculated successfully.")
                else:
                    logging.warning("M2 OOF data was empty after filtering NaNs.")

            except Exception as e:
                logging.error(
                    f"Could not process M2 OOF predictions: {e}", exc_info=True
                )

        S7_MODEL_PERFORMANCE_REPORT.parent.mkdir(parents=True, exist_ok=True)
        with open(S7_MODEL_PERFORMANCE_REPORT, "w") as f:
            json.dump(report, f, indent=4)
        logging.info(f"Performance report saved to {S7_MODEL_PERFORMANCE_REPORT}")
        print("\n" + json.dumps(report, indent=4))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Script 3/3: M2 CV, Final Training, and Reporting"
    )
    parser.add_argument("--test", action="store_true", help="Run in quick test mode.")
    args = parser.parse_args()

    config = FinalTrainingConfig(test=args.test)
    if args.test:
        config.lgbm_params["n_estimators"] = 10

    assembler = FinalAssembler(config)

    if args.test:
        logging.warning("--- TEST MODE: Forcing partition limit for final stages. ---")
        # Test mode now correctly handled within _ensure_model_trained

    assembler.run()
