# /workspace/models/model_training_metalabeling_A_v5.py
# [V5仕様: 双方向ラベリング (Two-Brain) 対応版]

import sys
from pathlib import Path
import logging
import argparse
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
    S3_SELECTED_FEATURES_DIR,
    S7_M1_OOF_PREDICTIONS_LONG,
    S7_M1_OOF_PREDICTIONS_SHORT,
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


from dataclasses import dataclass, field


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
        self.partitions = self._discover_partitions()
        if self.config.test_limit > 0:
            logging.warning(
                f"--- TEST MODE: Using only first {self.config.test_limit} partitions. ---"
            )
            self.partitions = self.partitions[: self.config.test_limit]

    def _load_features(self, feature_path: Path) -> List[str]:
        logging.info(f"Loading features from {feature_path}...")

        if not feature_path.exists():
            raise FileNotFoundError(f"Feature list file not found: {feature_path}")

        # ※以下の with open(...) のパスも self.config... から feature_path に変更してください
        with open(feature_path, "r") as f:
            raw_features = [line.strip() for line in f if line.strip()]

        # V5仕様のメタデータ・未来情報の完全除外
        exclude_exact = {
            "timestamp",
            "timeframe",
            "is_trigger",
            "label_long",
            "label_short",
            "uniqueness_long",
            "uniqueness_short",
            "duration_long",
            "duration_short",
            "concurrency_long",
            "concurrency_short",
            # 念のため旧仕様のカラムも除外
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
        }

        features = []
        for col in raw_features:
            if col in exclude_exact:
                continue
            if col.startswith("is_trigger_on"):
                continue
            features.append(col)

        logging.info(
            f"   -> Loaded {len(features)} valid features (filtered out V5 metadata)."
        )
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

    def _calculate_scale_pos_weight(self, direction: str) -> float:
        """
        V5: 指定された方向 (long/short) のラベル比率を計算する
        対象は is_trigger == 1 の行のみ。
        """
        label_col = f"label_{direction}"
        logging.info(
            f"Calculating scale_pos_weight for {direction.upper()} (is_trigger=1)..."
        )
        counts = Counter({0: 0, 1: 0})
        total_samples = 0

        for partition_date in tqdm(
            self.partitions, desc=f"Scanning {direction} labels"
        ):
            p_path_glob = str(
                self.config.input_dir
                / f"year={partition_date.year}/month={partition_date.month}/day={partition_date.day}/*.parquet"
            )
            try:
                # is_triggerフラグと対象ラベルのみ読み込み
                df_labels = pl.read_parquet(
                    p_path_glob, columns=["is_trigger", label_col]
                )
                # is_trigger == 1 でフィルタリング
                df_labels = df_labels.filter(pl.col("is_trigger") == 1)

                if df_labels.is_empty():
                    continue

                binary_labels = df_labels.select(
                    pl.when(pl.col(label_col) == 1)
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
                f"One of the classes has zero samples for {direction}. Using scale_pos_weight = 1.0"
            )
            return 1.0

        scale_pos_weight = count_neg / count_pos
        logging.info(f"  -> Total samples (is_trigger=1): {total_samples}")
        logging.info(
            f"  -> Calculated scale_pos_weight ({direction}): {scale_pos_weight:.4f}"
        )
        return scale_pos_weight

    def _train_model_partition_based(self, direction: str) -> Dict[str, np.ndarray]:
        logging.info(f"--- Starting BATCH Training for M1 ({direction.upper()}) ---")
        kfold = PartitionPurgedKFold(
            self.config.n_splits, self.config.purge_days, self.config.embargo_days
        )

        label_col = f"label_{direction}"
        weight_col = f"uniqueness_{direction}"

        oof_results = {
            "timestamp": [],
            "prediction": [],
            "true_label": [],
            "uniqueness": [],
        }

        for i, (train_partitions, val_partitions) in enumerate(
            kfold.split(self.partitions)
        ):
            logging.info(
                f"  [{direction.upper()}] Fold {i + 1}/{self.config.n_splits}..."
            )

            if self.config.test_fold_limit > 0:
                train_partitions = train_partitions[: self.config.test_fold_limit]
                val_partitions = val_partitions[: self.config.test_fold_limit]

            # --- バッチ学習用データの収集 ---
            X_train_list, y_train_list, w_train_list = [], [], []

            if len(train_partitions) > 0:
                for partition_date in tqdm(
                    train_partitions, desc=f"  Loading Train Fold {i + 1}"
                ):
                    p_path_glob = str(
                        self.config.input_dir
                        / f"year={partition_date.year}/month={partition_date.month}/day={partition_date.day}/*.parquet"
                    )
                    try:
                        df_chunk = pl.read_parquet(p_path_glob)
                        # V5: トリガー箇所のみを学習対象にする
                        if "is_trigger" in df_chunk.columns:
                            df_chunk = df_chunk.filter(pl.col("is_trigger") == 1)
                    except Exception:
                        continue

                    if df_chunk.is_empty():
                        continue

                    X_train_list.append(df_chunk.select(self.features).to_numpy())
                    y_train_list.append(np.where(df_chunk[label_col] == 1, 1, 0))
                    w_train_list.append(df_chunk[weight_col].to_numpy())

            # --- バッチ学習実行 ---
            model: lgb.Booster = None
            if len(X_train_list) > 0:
                try:
                    X_train = np.concatenate(X_train_list)
                    y_train = np.concatenate(y_train_list)
                    w_train = np.concatenate(w_train_list)

                    del X_train_list, y_train_list, w_train_list
                    gc.collect()

                    train_params = self.config.lgbm_params.copy()
                    n_estimators = train_params.pop("n_estimators", 1000)

                    model = lgb.train(
                        train_params,
                        lgb.Dataset(X_train, label=y_train, weight=w_train),
                        num_boost_round=n_estimators,
                    )

                    del X_train, y_train, w_train
                    gc.collect()

                except Exception as fit_error:
                    logging.error(
                        f"Error during batch training ({direction}): {fit_error}",
                        exc_info=True,
                    )
                    model = None
            else:
                logging.warning(
                    f"    -> No training data found for this fold ({direction})."
                )

            # --- 予測 (Validation) ---
            if model is not None:
                for partition_date in tqdm(
                    val_partitions, desc=f"  Predicting Fold {i + 1}"
                ):
                    p_path_glob = str(
                        self.config.input_dir
                        / f"year={partition_date.year}/month={partition_date.month}/day={partition_date.day}/*.parquet"
                    )
                    try:
                        df_chunk = pl.read_parquet(p_path_glob)
                        # Validationもトリガー箇所のみ
                        if "is_trigger" in df_chunk.columns:
                            df_chunk = df_chunk.filter(pl.col("is_trigger") == 1)
                    except Exception:
                        continue

                    if df_chunk.is_empty():
                        continue

                    X_val = df_chunk.select(self.features).to_numpy()
                    try:
                        predictions = model.predict(X_val)
                    except Exception as pred_error:
                        logging.error(
                            f"Error during prediction ({direction}): {pred_error}"
                        )
                        predictions = np.full(len(df_chunk), np.nan)

                    oof_results["timestamp"].append(df_chunk["timestamp"].to_numpy())
                    oof_results["prediction"].append(predictions)
                    oof_results["true_label"].append(df_chunk[label_col].to_numpy())
                    oof_results["uniqueness"].append(df_chunk[weight_col].to_numpy())

            del model
            gc.collect()

        logging.info(f"Concatenating OOF results for {direction}...")
        for key in oof_results:
            if oof_results[key]:
                try:
                    oof_results[key] = np.concatenate(oof_results[key])
                except ValueError:
                    oof_results[key] = np.array([])
            else:
                oof_results[key] = np.array([])

        return oof_results

    def run(self) -> None:
        logging.info("### Script 1/3: M1 Two-Brain Cross-Validation (V5) ###")
        S7_M1_OOF_PREDICTIONS_LONG.parent.mkdir(parents=True, exist_ok=True)
        directions = ["long", "short"]
        output_paths = {
            "long": S7_M1_OOF_PREDICTIONS_LONG,
            "short": S7_M1_OOF_PREDICTIONS_SHORT,
        }

        # Blueprintで追加したディレクトリをインポート済みであることを確認 (from blueprint import S3_SELECTED_FEATURES_DIR)

        # 双方向のループ処理 (Two-Brain Training)
        for direction in directions:
            logging.info("\n" + "=" * 50)
            logging.info(f"=== Starting Pipeline for: {direction.upper()} ===")
            logging.info("=" * 50)

            # --- パターンB: 専用特徴量リストを動的に読み込む ---
            dedicated_feature_path = (
                S3_SELECTED_FEATURES_DIR / f"m1_{direction}_features.txt"
            )
            self.features = self._load_features(dedicated_feature_path)

            # scale_pos_weightを方向ごとに動的計算
            scale_pos_weight = self._calculate_scale_pos_weight(direction)
            self.config.lgbm_params["scale_pos_weight"] = scale_pos_weight

            # 学習とOOF予測
            oof_results = self._train_model_partition_based(direction)

            # 結果の保存
            if oof_results["timestamp"].size > 0:
                oof_df = pl.DataFrame(
                    {
                        "timestamp": oof_results["timestamp"],
                        "timeframe": oof_results["timeframe"],  # ★追加
                        "prediction": oof_results["prediction"],
                        "true_label": oof_results["true_label"],
                        "uniqueness": oof_results["uniqueness"],
                    }
                )

                # ▼▼▼ 追加: 下流でのJoinエラーを防ぐため、Intを文字列に戻す ▼▼▼
                reverse_map = {0: "M1", 1: "M3", 2: "M5", 3: "M8", 4: "M15"}
                oof_df = oof_df.with_columns(
                    pl.col("timeframe")
                    .replace_strict(reverse_map, default=None)
                    .cast(pl.Utf8)
                ).sort(["timestamp", "timeframe"])
                # ▲▲▲ 追加ここまで ▲▲▲

                out_path = output_paths[direction]
                oof_df.write_parquet(out_path, compression="zstd")

                logging.info(
                    f"Successfully saved M1 {direction.upper()} OOF predictions to: {out_path}"
                )
                logging.info(f"  - Total predictions saved: {len(oof_df)}")
            else:
                logging.warning(f"No OOF predictions generated for {direction}.")

        logging.info("\n" + "=" * 60)
        logging.info("### V5 Two-Brain Training COMPLETED! ###")
        logging.info("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Script 1/3: M1 Two-Brain Cross-Validation (V5)"
    )
    parser.add_argument(
        "--test-limit",
        type=int,
        default=0,
        help="Limit total partitions discovered for a very small test.",
    )
    parser.add_argument("--test", action="store_true", help="Run in quick test mode.")
    args = parser.parse_args()

    fold_limit = 5 if args.test else 0
    config = TrainingConfig(test_limit=args.test_limit, test_fold_limit=fold_limit)

    validator = M1CrossValidator(config)
    validator.run()
