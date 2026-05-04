# /workspace/models/model_training_metalabeling_A.py
# [修正版: n_estimators 分配 + ★非数値カラムの自動除外機能を追加★]

import sys
from pathlib import Path
import logging
import argparse
from dataclasses import dataclass, field
import datetime
import warnings
import gc

import polars as pl
import numpy as np
import lightgbm as lgb
from sklearn.model_selection._split import BaseCrossValidator
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
class TrainingConfig:
    input_dir: Path = S6_WEIGHTED_DATASET
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
    test_limit: int = 0
    test_fold_limit: int = 0


class PartitionPurgedKFold(BaseCrossValidator):
    def __init__(self, n_splits: int = 5, purge_days: int = 3, embargo_days: int = 2):
        self.n_splits = n_splits
        self.purge_days = purge_days
        self.embargo_days = embargo_days

    def split(
        self, partitions: List[datetime.date]
    ) -> Generator[Tuple[List[datetime.date], List[datetime.date]], None, None]:
        n_partitions = len(partitions)
        fold_size = n_partitions // self.n_splits
        for i in range(self.n_splits):
            start = i * fold_size
            end = start + fold_size if i < self.n_splits - 1 else n_partitions
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


class M1CrossValidator:
    def __init__(self, config: TrainingConfig):
        self.config = config
        self.features: List[str] = self._load_features()
        self.partitions = self._discover_partitions()
        if self.config.test_limit > 0:
            logging.warning(
                f"--- TEST MODE: Using only first {self.config.test_limit} partitions. ---"
            )
            self.partitions = self.partitions[: self.config.test_limit]

        self.scale_pos_weight = self._calculate_scale_pos_weight()
        self.config.lgbm_params["scale_pos_weight"] = self.scale_pos_weight
        logging.info(f"Using scale_pos_weight: {self.scale_pos_weight:.4f}")

    def _load_features(self) -> List[str]:
        logging.info(f"Loading feature list from {self.config.feature_list_path}...")
        with open(self.config.feature_list_path, "r") as f:
            features = [line.strip() for line in f if line.strip()]
        logging.info(f"   -> Loaded {len(features)} features.")
        return features

    def _discover_partitions(self) -> List[datetime.date]:
        logging.info("Discovering all physical partitions...")
        paths = self.config.input_dir.glob("year=*/month=*/day=*")
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
        logging.info(f"  -> Discovered and sorted {len(dates)} daily partitions.")
        return dates

    def _calculate_scale_pos_weight(self) -> float:
        logging.info(
            "Calculating scale_pos_weight for M1 (label == 1 vs label != 1)..."
        )
        counts = Counter({0: 0, 1: 0})
        total_samples = 0
        partitions_to_scan = self.partitions

        for partition_date in tqdm(
            partitions_to_scan, desc="Scanning labels for scale_pos_weight"
        ):
            # [修正] data.parquet を直接指定して重複読み込みを防止
            p_path = (
                self.config.input_dir
                / f"year={partition_date.year}/month={partition_date.month}/day={partition_date.day}/data.parquet"
            )
            try:
                df_labels = pl.read_parquet(p_path, columns=["label"])
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
                logging.warning(f"Could not read labels from {partition_date}: {e}")
                continue

        count_neg = counts[0]
        count_pos = counts[1]

        if count_pos == 0 or count_neg == 0:
            logging.warning(
                "One of the classes has zero samples. Using scale_pos_weight = 1.0"
            )
            return 1.0

        scale_pos_weight = count_neg / count_pos
        logging.info(f"  -> Total samples scanned: {total_samples}")
        logging.info(f"  -> Positive (label=1) count: {count_pos}")
        logging.info(f"  -> Negative (label!=1) count: {count_neg}")
        logging.info(f"  -> Calculated scale_pos_weight: {scale_pos_weight:.4f}")
        return scale_pos_weight

    def run(self) -> None:
        logging.info("### Script 1/3: M1 Cross-Validation ###")
        S7_M1_OOF_PREDICTIONS.parent.mkdir(parents=True, exist_ok=True)

        m1_oof_results = self._train_model_partition_based()

        logging.info("--- M1 Cross-Validation Process Completed ---")

        if m1_oof_results["timestamp"].size > 0:
            oof_df = pl.DataFrame(m1_oof_results).sort("timestamp")
            oof_df.write_parquet(S7_M1_OOF_PREDICTIONS, compression="zstd")
            logging.info(
                f"Successfully saved M1 OOF predictions to: {S7_M1_OOF_PREDICTIONS}"
            )
            logging.info(f"  - Total predictions saved: {len(oof_df)}")
        else:
            logging.warning(
                "No OOF predictions were generated. Output file was not created."
            )

        logging.info("\n" + "=" * 60)
        logging.info("### Script 1/3 FINISHED! You can now run Script B. ###")
        logging.info("=" * 60)

    def _train_model_partition_based(self) -> Dict[str, np.ndarray]:
        logging.info(
            "--- Starting True Sequential Partition-Based Training for M1 (Primary) ---"
        )
        kfold = PartitionPurgedKFold(
            self.config.n_splits, self.config.purge_days, self.config.embargo_days
        )
        # ★★★ [修正] timeframe を保存対象に追加 ★★★
        oof_results = {
            "timestamp": [],
            "timeframe": [],  # 追加
            "prediction": [],
            "true_label": [],
            "uniqueness": [],
        }

        dropped_features_log = set()

        for i, (train_partitions, val_partitions) in enumerate(
            kfold.split(self.partitions)
        ):
            logging.info(f"  [M1 (Primary)] Fold {i + 1}/{self.config.n_splits}...")
            model: lgb.Booster = None

            if self.config.test_fold_limit > 0:
                logging.warning(
                    f"--- TEST FOLD MODE: Limiting partitions to {self.config.test_fold_limit} for train/predict. ---"
                )
                train_partitions = train_partitions[: self.config.test_fold_limit]
                val_partitions = val_partitions[: self.config.test_fold_limit]

            train_params = self.config.lgbm_params.copy()
            n_estimators_total = train_params.pop("n_estimators", 1000)
            n_partitions_train = len(train_partitions)
            num_boost_round_per_partition = 0

            if n_partitions_train == 0:
                logging.warning("    -> No train partitions found. Skipping fold.")
            else:
                num_boost_round_per_partition = max(
                    1, int(np.ceil(n_estimators_total / n_partitions_train))
                )
                logging.info(
                    f"    -> Total {n_estimators_total} estimators distributed over {n_partitions_train} partitions."
                )
                logging.info(
                    f"    -> Setting num_boost_round = {num_boost_round_per_partition} per partition."
                )

            if n_partitions_train > 0:
                logging.info(
                    f"    -> Training on {len(train_partitions)} partitions sequentially..."
                )
                for partition_date in tqdm(
                    train_partitions, desc=f"  Training Fold {i + 1}"
                ):
                    p_path_glob = str(
                        self.config.input_dir
                        / f"year={partition_date.year}/month={partition_date.month}/day={partition_date.day}/data.parquet"
                    )
                    try:
                        df_chunk = pl.read_parquet(p_path_glob)
                    except Exception:
                        continue
                    if df_chunk.is_empty():
                        continue

                    valid_numeric_features = []
                    for col in self.features:
                        if col not in df_chunk.columns:
                            continue
                        dtype = df_chunk[col].dtype
                        if dtype in [
                            pl.Float32,
                            pl.Float64,
                            pl.Int8,
                            pl.Int16,
                            pl.Int32,
                            pl.Int64,
                            pl.UInt8,
                            pl.UInt16,
                            pl.UInt32,
                            pl.UInt64,
                            pl.Boolean,
                        ]:
                            valid_numeric_features.append(col)

                    dropped = set(self.features) - set(valid_numeric_features)
                    dropped_key = f"fold_{i}_dropped"
                    if dropped and dropped_key not in dropped_features_log:
                        logging.warning(
                            f"  [Auto-Fix] Dropped {len(dropped)} non-numeric features to prevent errors."
                        )
                        dropped_features_log.add(dropped_key)

                    if not valid_numeric_features:
                        continue

                    X_chunk = df_chunk.select(valid_numeric_features).to_numpy()
                    y_chunk = np.where(df_chunk["label"] == 1, 1, 0)
                    w_chunk = df_chunk["uniqueness"].to_numpy()

                    try:
                        model = lgb.train(
                            train_params,
                            lgb.Dataset(X_chunk, label=y_chunk, weight=w_chunk),
                            num_boost_round=num_boost_round_per_partition,
                            init_model=model,
                            keep_training_booster=True,
                        )
                    except Exception as fit_error:
                        logging.error(
                            f"Error during model training for partition {partition_date}: {fit_error}",
                            exc_info=False,
                        )
                        continue

            if model is None:
                logging.warning(
                    f"    -> Model for Fold {i + 1} was not trained. Skipping."
                )
                continue

            logging.info(
                f"    -> Predicting on {len(val_partitions)} partitions sequentially..."
            )
            for partition_date in tqdm(
                val_partitions, desc=f"  Predicting Fold {i + 1}"
            ):
                p_path_glob = str(
                    self.config.input_dir
                    / f"year={partition_date.year}/month={partition_date.month}/day={partition_date.day}/data.parquet"
                )
                try:
                    df_chunk = pl.read_parquet(p_path_glob)
                except Exception:
                    continue
                if df_chunk.is_empty():
                    continue

                # 予測時も同じ特徴量を使用
                pred_features = [
                    col
                    for col in self.features
                    if col in df_chunk.columns
                    and df_chunk[col].dtype
                    in [
                        pl.Float32,
                        pl.Float64,
                        pl.Int8,
                        pl.Int16,
                        pl.Int32,
                        pl.Int64,
                        pl.UInt8,
                        pl.UInt16,
                        pl.UInt32,
                        pl.UInt64,
                        pl.Boolean,
                    ]
                ]

                X_val = df_chunk.select(pred_features).to_numpy()
                try:
                    predictions = model.predict(X_val)
                except Exception as pred_error:
                    logging.error(
                        f"Error during prediction for partition {partition_date}: {pred_error}",
                        exc_info=False,
                    )
                    predictions = np.full(len(df_chunk), np.nan)

                oof_results["timestamp"].append(df_chunk["timestamp"].to_numpy())
                # ★★★ [修正] timeframe を保存 ★★★
                oof_results["timeframe"].append(df_chunk["timeframe"].to_numpy())
                oof_results["prediction"].append(predictions)
                oof_results["true_label"].append(df_chunk["label"].to_numpy())
                oof_results["uniqueness"].append(df_chunk["uniqueness"].to_numpy())

            logging.info(f"    -> Fold {i + 1} prediction complete.")
            del model
            gc.collect()

        logging.info("Concatenating OOF results...")
        for key in oof_results:
            if oof_results[key]:
                try:
                    oof_results[key] = np.concatenate(oof_results[key])
                except ValueError as concat_error:
                    logging.error(
                        f"Error concatenating results for key '{key}': {concat_error}"
                    )
                    oof_results[key] = np.array([])
            else:
                oof_results[key] = np.array([])
        logging.info("OOF results concatenated.")

        return oof_results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Script 1/3: M1 Cross-Validation")

    parser.add_argument(
        "--test-limit",
        type=int,
        default=0,
        help="Limit total partitions discovered for a very small test. Default is 0 (no limit).",
    )
    parser.add_argument(
        "--test",
        action="store_true",
        help="Run in quick test mode, limiting each fold to 5 partitions for training and prediction.",
    )

    args = parser.parse_args()
    fold_limit = 5 if args.test else 0
    config = TrainingConfig(test_limit=args.test_limit, test_fold_limit=fold_limit)

    validator = M1CrossValidator(config)
    validator.run()
