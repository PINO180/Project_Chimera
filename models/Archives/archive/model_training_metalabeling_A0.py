# /workspace/models/model_training_metalabeling_A.py

import sys
from pathlib import Path
import logging
import argparse
from dataclasses import dataclass, field
import datetime
import warnings

import polars as pl
import numpy as np
import lightgbm as lgb
from sklearn.model_selection._split import BaseCrossValidator
from typing import List, Tuple, Dict, Any, Generator

# プロジェクトのルートディレクトリをPythonの検索パスに追加
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
    test_fold_limit: int = 0  # --- ★★★ 新しいテストモード用の設定 ★★★ ---


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

    def run(self) -> None:
        logging.info("### Script 1/3: M1 Cross-Validation ###")
        S7_M1_OOF_PREDICTIONS.parent.mkdir(parents=True, exist_ok=True)

        m1_oof_results = self._train_model_partition_based()

        logging.info("--- M1 Cross-Validation Process Completed ---")

        # --- ★★★ ここが致命的なエラーの修正箇所 ★★★ ---
        # NumPy配列が空でないことを .size > 0 で正しくチェックする
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
        oof_results = {
            "timestamp": [],
            "prediction": [],
            "true_label": [],
            "uniqueness": [],
        }

        for i, (train_partitions, val_partitions) in enumerate(
            kfold.split(self.partitions)
        ):
            logging.info(f"  [M1 (Primary)] Fold {i + 1}/{self.config.n_splits}...")
            model: lgb.Booster = None

            # --- ★★★ 新しいテストモードの実装箇所 ★★★ ---
            if self.config.test_fold_limit > 0:
                logging.warning(
                    f"--- TEST FOLD MODE: Limiting partitions to {self.config.test_fold_limit} for train/predict. ---"
                )
                train_partitions = train_partitions[: self.config.test_fold_limit]
                val_partitions = val_partitions[: self.config.test_fold_limit]

            logging.info(
                f"    -> Training on {len(train_partitions)} partitions sequentially..."
            )
            for partition_date in train_partitions:
                p_path = str(
                    self.config.input_dir
                    / f"year={partition_date.year}/month={partition_date.month}/day={partition_date.day}/*.parquet"
                )
                try:
                    df_chunk = pl.read_parquet(p_path)
                except Exception:
                    continue
                if df_chunk.is_empty():
                    continue

                X_chunk = df_chunk.select(self.features).to_numpy()
                y_chunk = np.where(df_chunk["label"] == 1, 1, 0)
                w_chunk = df_chunk["uniqueness"].to_numpy()
                model = lgb.train(
                    self.config.lgbm_params,
                    lgb.Dataset(X_chunk, label=y_chunk, weight=w_chunk),
                    init_model=model,
                    keep_training_booster=True,
                )

            if model is None:
                logging.warning(
                    f"    -> Model for Fold {i + 1} was not trained. Skipping."
                )
                continue

            logging.info(
                f"    -> Predicting on {len(val_partitions)} partitions sequentially..."
            )
            for partition_date in val_partitions:
                p_path = str(
                    self.config.input_dir
                    / f"year={partition_date.year}/month={partition_date.month}/day={partition_date.day}/*.parquet"
                )
                try:
                    df_chunk = pl.read_parquet(p_path)
                except Exception:
                    continue
                if df_chunk.is_empty():
                    continue

                X_val = df_chunk.select(self.features).to_numpy()
                predictions = model.predict(X_val)

                oof_results["timestamp"].append(df_chunk["timestamp"].to_numpy())
                oof_results["prediction"].append(predictions)
                oof_results["true_label"].append(df_chunk["label"].to_numpy())
                oof_results["uniqueness"].append(df_chunk["uniqueness"].to_numpy())

            logging.info(f"    -> Fold {i + 1} prediction complete.")

        for key in oof_results:
            if oof_results[key]:
                oof_results[key] = np.concatenate(oof_results[key])
        return oof_results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Script 1/3: M1 Cross-Validation")

    # This argument remains for limiting the total number of partitions discovered
    parser.add_argument(
        "--test-limit",
        type=int,
        default=0,
        help="Limit total partitions discovered for a very small test. Default is 0 (no limit).",
    )

    # --- ★★★ This is the corrected test mode implementation ★★★ ---
    # The presence of this flag activates test mode; its absence means production mode.
    parser.add_argument(
        "--test",
        action="store_true",  # This makes it a boolean flag
        help="Run in quick test mode, limiting each fold to 5 partitions for training and prediction.",
    )

    args = parser.parse_args()

    # Determine the test_fold_limit based on the --test flag
    # If --test is used, limit is 5. If not, limit is 0 (which means no limit).
    fold_limit = 5 if args.test else 0

    # Pass the correctly determined limits to the configuration
    config = TrainingConfig(test_limit=args.test_limit, test_fold_limit=fold_limit)

    validator = M1CrossValidator(config)
    validator.run()
