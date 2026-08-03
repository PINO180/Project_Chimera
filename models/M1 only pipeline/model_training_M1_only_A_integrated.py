# /workspace/models/model_training_M1_only_A_integrated.py
# =====================================================================
# [(あ) M1 単独統合スクリプト]  ABC 3本 → A 1本
#
# 目的:
#   直交分割・メタラベリング(M2)を廃止し、全特徴量を単一の脳(M1)に
#   フル投下する構成に統合する。旧 Ax2(CV/OOF) + Cx2 の M1 向け後処理
#   (全データ再学習・Isotonic較正・M1レポート) を 1 本にまとめたもの。
#
# 旧構成との対応:
#   - 旧 A (Ax2): M1 Two-Brain CV → OOF          … 本スクリプトに継承 (特徴量のみ全量化)
#   - 旧 B (Bx2): メタラベル生成                 … 廃止 (M2 が無いので不要)
#   - 旧 C (Cx2): M2 CV + M1/M2 最終学習・較正    … M1 部分のみ本スクリプトに継承、M2 は全廃
#   - split_features_first_orthogonal.py         … 実行不要 (直交分割しない)
#
# 入力:  S6_WEIGHTED_DATASET, S3_FEATURES_FOR_TRAINING_V5 (全特徴量リスト)
# 出力(方向別):
#   - S7_M1_OOF_PREDICTIONS_{LONG,SHORT}          (OOF 予測)
#   - S7_M1_MODEL_{LONG,SHORT}_PKL                (全データ最終モデル)
#   - S7_M1_CALIBRATED_{LONG,SHORT}               (Isotonic 較正器)
#   - S7_MODEL_PERFORMANCE_REPORT_{LONG,SHORT}    (M1 のみの性能レポート json)
#
# 実行順:
#   direction ごとに CV → OOF 保存 → OOF AUC 表示 → 全データ最終学習 →
#   較正 → M1 レポート、を Long 完了・Short 完了で各々出力する。
#
# 注意:
#   M1 の学習仕様は旧 Ax2 と Cx2 で完全一致していることを確認済み
#   (is_trigger==1 フィルタ / label_{dir} を 0/1 / uniqueness_{dir} を weight /
#    同一 lgbm_params / 同一 _load_features exclude_exact)。
#   したがって本統合による数値挙動の変化は「特徴量が直交分割版 →
#   全特徴量に変わる」点のみに由来する。
# =====================================================================

import sys
from pathlib import Path
import logging
import argparse
import datetime
import warnings
import gc
import json

import polars as pl
import numpy as np
import lightgbm as lgb
from sklearn.model_selection._split import BaseCrossValidator
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import roc_auc_score
import joblib
from typing import List, Tuple, Dict, Any, Generator
from tqdm import tqdm
from collections import Counter
from dataclasses import dataclass, field

# --- プロジェクトのルートディレクトリを Python の検索パスに追加 ---
project_root = Path(__file__).resolve().parents[1]
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

from blueprint import (
    S6_WEIGHTED_DATASET,
    S3_FEATURES_FOR_TRAINING_V5,
    S7_M1_OOF_PREDICTIONS_LONG,
    S7_M1_OOF_PREDICTIONS_SHORT,
    S7_M1_MODEL_LONG_PKL,
    S7_M1_MODEL_SHORT_PKL,
    S7_M1_CALIBRATED_LONG,
    S7_M1_CALIBRATED_SHORT,
    S7_MODEL_PERFORMANCE_REPORT_LONG,
    S7_MODEL_PERFORMANCE_REPORT_SHORT,
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
class M1OnlyConfig:
    input_dir: Path = S6_WEIGHTED_DATASET
    feature_list_path: Path = S3_FEATURES_FOR_TRAINING_V5  # ★全特徴量 (直交分割しない)
    n_splits: int = 5
    purge_days: int = 3
    embargo_days: int = 2
    # 最終学習の train/calib 分割比 (旧 Cx2 と同一: 8:2)
    final_train_frac: float = 0.8
    lgbm_params: Dict[str, Any] = field(
        default_factory=lambda: {
            "objective": "binary",
            "metric": "auc",
            "boosting_type": "gbdt",
            "n_estimators": 2000,
            "learning_rate": 0.01,
            "num_leaves": 127,
            "max_depth": -1,
            "min_data_in_leaf": 100,
            "lambda_l1": 0.1,
            "lambda_l2": 0.1,
            "seed": 42,
            "n_jobs": 1,
            "verbose": 1,
            "colsample_bytree": 0.8,
            "subsample": 0.8,
        }
    )
    test_limit: int = 0
    test_fold_limit: int = 0


class PartitionPurgedKFold(BaseCrossValidator):
    """旧 Ax2/Cx2 と同一の Purged K-Fold (日次パーティション境界で purge/embargo)。"""

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


class M1OnlyAssembler:
    def __init__(self, config: M1OnlyConfig):
        self.config = config
        self.partitions = self._discover_partitions()
        if self.config.test_limit > 0:
            logging.warning(
                f"--- TEST MODE: Using only first {self.config.test_limit} partitions. ---"
            )
            self.partitions = self.partitions[: self.config.test_limit]

        # 全特徴量リストを一度だけ読み込む (方向で共通、直交分割しない)
        self.features = self._load_features(self.config.feature_list_path)

    # =================================================================
    # 特徴量リスト (旧 Ax2/Bx2/Cx2 で統一済みの exclude_exact をそのまま踏襲)
    # =================================================================
    def _load_features(self, feature_path: Path) -> List[str]:
        logging.info(f"Loading FULL feature list from {feature_path}...")

        if not feature_path.exists():
            raise FileNotFoundError(f"Feature list file not found: {feature_path}")

        with open(feature_path, "r") as f:
            raw_features = [line.strip() for line in f if line.strip()]

        # メタデータ・未来情報の完全除外 (旧 3 スクリプトの union と一致)
        exclude_exact = {
            # --- 基本メタ ---
            "timestamp",
            "timeframe",  # 学習特徴量から除外（データ管理用カラムとしては保持）
            "is_trigger",
            "t1",
            "direction",
            "exit_type",
            "first_ex_reason_int",
            # --- ラベル系 (双方向) ---
            "label",
            "label_long",
            "label_short",
            # --- uniqueness 系 (双方向) ---
            "uniqueness",
            "uniqueness_long",
            "uniqueness_short",
            # --- duration 系 (双方向) ---
            "duration_long",
            "duration_short",
            # --- concurrency 系 (双方向、未来情報リーク防止) ---
            "concurrency_long",
            "concurrency_short",
            # --- payoff/multiplier 系 (双方向) ---
            "payoff_ratio",
            "payoff_ratio_long",
            "payoff_ratio_short",
            "pt_multiplier",
            "sl_multiplier",
            # --- ATR / 補助計算 ---
            "atr_value",
            "calculated_body_ratio",
            "fallback_vol",
            # --- 価格データ ---
            "open",
            "high",
            "low",
            "close",
            # --- メタラベリング系 (M2 廃止だが念のため残置) ---
            "m1_pred_proba",
            "meta_label",
            # --- 学習対象外メタデータ ---
            "disc",  # 週末跨ぎギャップ判定 bool 列 (最終防御線)
        }

        features = []
        for col in raw_features:
            if col in exclude_exact:
                continue
            if col.startswith("is_trigger_on"):
                continue
            features.append(col)

        logging.info(f"   -> Loaded {len(features)} valid features (M1 full投下).")
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

    # =================================================================
    # M1 データ整形 (旧 Cx2._filter_dataframe の is_m1=True 相当)
    #   is_trigger==1 に絞り、timeframe を Int32 化して (timestamp,timeframe) で unique
    # =================================================================
    def _filter_m1(self, df: pl.DataFrame) -> pl.DataFrame:
        if "is_trigger" in df.columns:
            df = df.filter(pl.col("is_trigger") == 1)

        if "timeframe" in df.columns:
            if df.schema["timeframe"] in [pl.Utf8, pl.String]:
                df = df.with_columns(
                    pl.col("timeframe")
                    .replace({"M0.5": 0, "M1": 1, "M3": 2, "M5": 3, "M8": 4, "M15": 5})
                    .cast(pl.Int32)
                )
            else:
                df = df.with_columns(pl.col("timeframe").cast(pl.Int32))

            df = df.unique(
                subset=["timestamp", "timeframe"], keep="last", maintain_order=True
            )
        return df

    # =================================================================
    # scale_pos_weight (旧 Ax2._calculate_scale_pos_weight と同一ロジック)
    # =================================================================
    def _calculate_scale_pos_weight(self, direction: str) -> float:
        label_col = f"label_{direction}"
        logging.info(
            f"[{direction.upper()}] Calculating scale_pos_weight over all partitions..."
        )
        pos, neg = 0, 0
        for partition_date in tqdm(
            self.partitions, desc=f"  [{direction.upper()}] scale_pos_weight"
        ):
            p_path_glob = str(
                self.config.input_dir
                / f"year={partition_date.year}/month={partition_date.month}/day={partition_date.day}/*.parquet"
            )
            try:
                df_chunk = pl.read_parquet(
                    p_path_glob, columns=["is_trigger", label_col]
                )
            except Exception:
                continue
            if "is_trigger" in df_chunk.columns:
                df_chunk = df_chunk.filter(pl.col("is_trigger") == 1)
            if df_chunk.is_empty():
                continue
            counts = (
                df_chunk.select(
                    pl.when(pl.col(label_col) == 1)
                    .then(pl.lit("pos"))
                    .otherwise(pl.lit("neg"))
                    .alias("cls")
                )["cls"]
                .to_list()
            )
            c = Counter(counts)
            pos += c.get("pos", 0)
            neg += c.get("neg", 0)

        total_samples = pos + neg
        logging.info(f"  -> Total samples (is_trigger=1): {total_samples:,}")
        if pos == 0:
            logging.warning(
                f"  -> No positive samples for {direction}; scale_pos_weight=1.0"
            )
            return 1.0
        spw = neg / pos
        logging.info(
            f"  -> [{direction.upper()}] pos={pos:,} / neg={neg:,} / "
            f"scale_pos_weight={spw:.4f}"
        )
        return spw

    # =================================================================
    # CV → OOF 生成 (旧 Ax2._train_model_partition_based と同一)
    # =================================================================
    def _train_cv_oof(self, direction: str) -> Dict[str, np.ndarray]:
        logging.info(f"--- Starting BATCH CV/OOF for M1 ({direction.upper()}) ---")
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
            "timeframe": [],
        }

        for i, (train_partitions, val_partitions) in enumerate(
            kfold.split(self.partitions)
        ):
            logging.info(f"  [{direction.upper()}] Fold {i + 1}/{self.config.n_splits}...")

            if self.config.test_fold_limit > 0:
                train_partitions = train_partitions[: self.config.test_fold_limit]
                val_partitions = val_partitions[: self.config.test_fold_limit]

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
                        if "is_trigger" in df_chunk.columns:
                            df_chunk = df_chunk.filter(pl.col("is_trigger") == 1)
                        if "timeframe" in df_chunk.columns:
                            df_chunk = df_chunk.unique(
                                subset=["timestamp", "timeframe"],
                                keep="last",
                                maintain_order=True,
                            )
                    except Exception:
                        continue

                    if df_chunk.is_empty():
                        continue

                    X_train_list.append(
                        df_chunk.select(self.features).fill_null(0).to_numpy()
                    )
                    y_train_list.append(np.where(df_chunk[label_col] == 1, 1, 0))
                    w_train_list.append(df_chunk[weight_col].to_numpy())

            model = None
            if len(X_train_list) > 0:
                try:
                    X_train = np.concatenate(X_train_list)
                    y_train = np.concatenate(y_train_list)
                    w_train = np.concatenate(w_train_list)

                    del X_train_list, y_train_list, w_train_list
                    gc.collect()

                    train_params = self.config.lgbm_params.copy()
                    n_estimators = train_params.pop("n_estimators", 1000)

                    logging.info(f"    -> fitting model on {len(X_train):,} samples...")
                    model = lgb.train(
                        train_params,
                        lgb.Dataset(
                            X_train,
                            label=y_train,
                            weight=w_train,
                            feature_name=self.features,
                        ),
                        num_boost_round=n_estimators,
                    )

                    del X_train, y_train, w_train
                    gc.collect()
                except Exception as fit_error:
                    logging.error(
                        f"Error during fold {i + 1} training ({direction}): {fit_error}",
                        exc_info=True,
                    )
                    model = None

            if model is None:
                logging.warning(
                    f"  [{direction.upper()}] Fold {i + 1}: no model trained, skipping val predictions."
                )
                continue

            # --- 検証 fold の予測 (OOF) ---
            for partition_date in tqdm(
                val_partitions, desc=f"  Predicting Val Fold {i + 1}"
            ):
                p_path_glob = str(
                    self.config.input_dir
                    / f"year={partition_date.year}/month={partition_date.month}/day={partition_date.day}/*.parquet"
                )
                try:
                    df_chunk = pl.read_parquet(p_path_glob)
                    if "is_trigger" in df_chunk.columns:
                        df_chunk = df_chunk.filter(pl.col("is_trigger") == 1)
                    if "timeframe" in df_chunk.columns:
                        df_chunk = df_chunk.unique(
                            subset=["timestamp", "timeframe"],
                            keep="last",
                            maintain_order=True,
                        )
                except Exception:
                    continue

                if df_chunk.is_empty():
                    continue

                X_val = df_chunk.select(self.features).fill_null(0).to_numpy()
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
                oof_results["timeframe"].append(df_chunk["timeframe"].to_numpy())

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

    # =================================================================
    # 全データ最終学習 (旧 Cx2._train_single_model の M1 相当)
    # =================================================================
    def _train_final_model(
        self, direction: str, partitions_to_train: List[datetime.date]
    ) -> lgb.Booster:
        logging.info(
            f"[{direction.upper()}]   - Training FINAL M1 on {len(partitions_to_train)} partitions..."
        )
        label_col = f"label_{direction}"
        weight_col = f"uniqueness_{direction}"

        X_list, y_list, w_list = [], [], []
        for p_date in tqdm(
            partitions_to_train, desc=f"[{direction.upper()}]   Loading final-train data"
        ):
            p_path_glob = str(
                self.config.input_dir
                / f"year={p_date.year}/month={p_date.month}/day={p_date.day}/*.parquet"
            )
            try:
                df_chunk = pl.read_parquet(p_path_glob)
                df_chunk = self._filter_m1(df_chunk)
            except Exception:
                continue
            if df_chunk.is_empty():
                continue

            missing = [f for f in self.features if f not in df_chunk.columns]
            if missing:
                continue

            X_list.append(df_chunk.select(self.features).fill_null(0).to_numpy())
            y_list.append(np.where(df_chunk[label_col].to_numpy() == 1, 1, 0))
            w_list.append(df_chunk[weight_col].to_numpy())

        if not X_list:
            raise RuntimeError(
                f"[{direction.upper()}] No data found for final M1 training."
            )

        X_train = np.concatenate(X_list)
        y_train = np.concatenate(y_list)
        w_train = np.concatenate(w_list)
        del X_list, y_list, w_list
        gc.collect()

        train_params = self.config.lgbm_params.copy()
        n_estimators = train_params.pop("n_estimators", 1000)

        logging.info(
            f"[{direction.upper()}]     -> fitting FINAL M1 on {len(X_train):,} samples..."
        )
        model = lgb.train(
            train_params,
            lgb.Dataset(
                X_train, label=y_train, weight=w_train, feature_name=self.features
            ),
            num_boost_round=n_estimators,
        )
        del X_train, y_train, w_train
        gc.collect()
        return model

    def _ensure_final_model(
        self, direction: str, model_path: Path, partitions_to_train: List[datetime.date]
    ) -> lgb.Booster:
        if not model_path.exists():
            if not partitions_to_train:
                raise ValueError(
                    f"[{direction.upper()}] No partitions available to train final M1."
                )
            model = self._train_final_model(direction, partitions_to_train)
            model_path.parent.mkdir(parents=True, exist_ok=True)
            joblib.dump(model, model_path)
            logging.info(
                f"[{direction.upper()}]   -> Final M1 model saved to {model_path}."
            )
        else:
            logging.warning(
                f"[{direction.upper()}] --- SKIPPING M1 Final Training: model exists at {model_path}. ---"
            )
            model = joblib.load(model_path)
        return model

    # =================================================================
    # Isotonic 較正 (旧 Cx2._manual_calibrate + _gather_predictions の M1 相当)
    # =================================================================
    def _gather_predictions_for_calibration(
        self, direction: str, dates: List[datetime.date], model: lgb.Booster
    ):
        label_col = f"label_{direction}"
        weight_col = f"uniqueness_{direction}"
        all_preds, all_labels, all_weights = [], [], []

        for p_date in tqdm(
            dates, desc=f"[{direction.upper()}]   Gathering calib predictions"
        ):
            p_path_glob = str(
                self.config.input_dir
                / f"year={p_date.year}/month={p_date.month}/day={p_date.day}/*.parquet"
            )
            try:
                df_chunk = pl.read_parquet(p_path_glob)
                df_chunk = self._filter_m1(df_chunk)
                if label_col not in df_chunk.columns:
                    logging.warning(
                        f"[{direction.upper()}] '{label_col}' not found in {p_date}. Skipping chunk."
                    )
                    continue
            except Exception:
                continue

            if df_chunk.is_empty():
                continue

            missing = [f for f in self.features if f not in df_chunk.columns]
            if missing:
                logging.warning(
                    f"[{direction.upper()}] calib partition {p_date} missing features. Skipping chunk."
                )
                continue

            X = df_chunk.select(self.features).fill_null(0).to_numpy()
            try:
                predictions = model.predict(X)
            except Exception as pred_error:
                logging.error(
                    f"[{direction.upper()}] Error predicting for calibration ({p_date}): {pred_error}"
                )
                continue

            labels = np.where(df_chunk[label_col].to_numpy() == 1, 1, 0)
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
                f"[{direction.upper()}] Error concatenating calibration data: {concat_error}"
            )
            return None, None, None

    def _manual_calibrate(
        self,
        direction: str,
        model: lgb.Booster,
        save_path: Path,
        dates: List[datetime.date],
    ):
        if save_path.exists():
            logging.warning(
                f"[{direction.upper()}] --- SKIPPING M1 Calibration: calibrated model exists. ---"
            )
            return
        logging.info(f"[{direction.upper()}]     -> Calibrating M1 model (Isotonic)...")
        y_pred, y_true, weights = self._gather_predictions_for_calibration(
            direction, dates, model
        )
        if y_pred is not None and len(y_pred) > 0:
            try:
                calibrator = IsotonicRegression(
                    y_min=0.0, y_max=1.0, out_of_bounds="clip"
                )
                calibrator.fit(y_pred, y_true, sample_weight=weights)
                save_path.parent.mkdir(parents=True, exist_ok=True)
                joblib.dump(calibrator, save_path)
                logging.info(
                    f"[{direction.upper()}]   -> Calibrated M1 (IsotonicRegressor) saved to {save_path}."
                )
            except Exception as calib_error:
                logging.error(
                    f"[{direction.upper()}] Failed to fit calibrator: {calib_error}",
                    exc_info=False,
                )
        else:
            logging.warning(
                f"[{direction.upper()}] No valid predictions gathered for calibration. Skipping."
            )

    # =================================================================
    # OOF AUC 表示 (前ターンで Ax2 に追加したものと同一仕様)
    # =================================================================
    def _report_oof_auc(self, direction: str, oof_df: pl.DataFrame) -> Dict[str, Any]:
        result = {"auc": float("nan"), "auc_weighted": float("nan")}
        try:
            _y_true = oof_df["true_label"].to_numpy().astype(float)
            _y_pred = oof_df["prediction"].to_numpy().astype(float)
            _w = oof_df["uniqueness"].to_numpy().astype(float)

            _valid = ~np.isnan(_y_true) & ~np.isnan(_y_pred) & ~np.isinf(_y_pred)
            _n_valid = int(_valid.sum())
            _n_drop = int(_y_true.shape[0] - _n_valid)

            _yt, _yp, _ww = _y_true[_valid], _y_pred[_valid], _w[_valid]
            _n_pos = int((_yt == 1).sum())
            _n_neg = int((_yt == 0).sum())

            logging.info("-" * 50)
            logging.info(f"### M1 {direction.upper()} OOF AUC REPORT ###")
            logging.info(
                f"  - Valid samples: {_n_valid:,} (dropped NaN/inf: {_n_drop:,})"
            )
            logging.info(
                f"  - Class balance: pos(label=1)={_n_pos:,} / neg(label=0)={_n_neg:,}"
            )

            if _n_valid == 0 or _n_pos == 0 or _n_neg == 0:
                logging.warning(
                    f"  - AUC skipped: one class only or no valid rows (pos={_n_pos}, neg={_n_neg})."
                )
            else:
                _auc = roc_auc_score(_yt, _yp)
                result["auc"] = float(_auc)
                logging.info(f"  - OOF AUC (unweighted): {_auc:.4f}")
                try:
                    _auc_w = roc_auc_score(_yt, _yp, sample_weight=_ww)
                    result["auc_weighted"] = float(_auc_w)
                    logging.info(f"  - OOF AUC (uniqueness-weighted): {_auc_w:.4f}")
                except Exception as _we:
                    logging.warning(f"  - Weighted AUC skipped: {_we}")
                logging.info(
                    f"  - Pred proba: min={_yp.min():.4f} / mean={_yp.mean():.4f} / max={_yp.max():.4f}"
                )
            logging.info("-" * 50)
        except Exception as _auc_err:
            logging.warning(
                f"AUC report failed for {direction}: {_auc_err}", exc_info=True
            )
        return result

    # =================================================================
    # メイン: direction ごとに CV→OOF→AUC→最終学習→較正→レポート
    # =================================================================
    def run(self) -> None:
        logging.info("### M1-ONLY Integrated Trainer (ABC→A / 全特徴量フル投下) ###")

        oof_paths = {
            "long": S7_M1_OOF_PREDICTIONS_LONG,
            "short": S7_M1_OOF_PREDICTIONS_SHORT,
        }
        model_paths = {
            "long": S7_M1_MODEL_LONG_PKL,
            "short": S7_M1_MODEL_SHORT_PKL,
        }
        calib_paths = {
            "long": S7_M1_CALIBRATED_LONG,
            "short": S7_M1_CALIBRATED_SHORT,
        }
        report_paths = {
            "long": S7_MODEL_PERFORMANCE_REPORT_LONG,
            "short": S7_MODEL_PERFORMANCE_REPORT_SHORT,
        }

        S7_M1_OOF_PREDICTIONS_LONG.parent.mkdir(parents=True, exist_ok=True)

        for direction in ["long", "short"]:
            logging.info("\n" + "=" * 60)
            logging.info(f"=== Starting M1-only Pipeline for: {direction.upper()} ===")
            logging.info("=" * 60)

            report: Dict[str, Any] = {"direction": direction}

            # scale_pos_weight
            spw = self._calculate_scale_pos_weight(direction)
            self.config.lgbm_params["scale_pos_weight"] = spw
            report["scale_pos_weight"] = spw

            # --- 1) CV → OOF ---
            oof_results = self._train_cv_oof(direction)

            if oof_results["timestamp"].size > 0:
                oof_df = pl.DataFrame(
                    {
                        "timestamp": oof_results["timestamp"],
                        "timeframe": oof_results["timeframe"],
                        "prediction": oof_results["prediction"],
                        "true_label": oof_results["true_label"],
                        "uniqueness": oof_results["uniqueness"],
                    }
                ).with_columns(
                    pl.col("timestamp").cast(pl.Datetime("us", "UTC")),
                    pl.col("timeframe").cast(pl.Utf8),
                ).sort(["timestamp", "timeframe"])

                out_path = oof_paths[direction]
                oof_df.write_parquet(out_path, compression="zstd")
                logging.info(
                    f"Saved M1 {direction.upper()} OOF predictions to: {out_path} "
                    f"({len(oof_df):,} rows)"
                )

                # --- 2) OOF AUC (Long 完了時点 / Short 完了時点) ---
                report["m1_oof_performance"] = self._report_oof_auc(direction, oof_df)
                del oof_df
                gc.collect()
            else:
                logging.warning(f"No OOF predictions generated for {direction}.")
                report["m1_oof_performance"] = {
                    "auc": float("nan"),
                    "auc_weighted": float("nan"),
                }

            # --- 3) 全データ最終学習 (train/calib = 8:2) ---
            split_idx = int(len(self.partitions) * self.config.final_train_frac)
            train_partitions = self.partitions[:split_idx]
            calib_partitions = self.partitions[split_idx:]

            final_model = self._ensure_final_model(
                direction, model_paths[direction], train_partitions
            )

            # --- 4) Isotonic 較正 ---
            if calib_partitions:
                self._manual_calibrate(
                    direction, final_model, calib_paths[direction], calib_partitions
                )
            else:
                logging.warning(
                    f"[{direction.upper()}] No partitions available for M1 calibration."
                )

            # --- 5) M1 レポート (json) ---
            report_paths[direction].parent.mkdir(parents=True, exist_ok=True)
            json_report = json.dumps(report, indent=4).replace("NaN", "null")
            with open(report_paths[direction], "w") as f:
                f.write(json_report)
            logging.info(
                f"[{direction.upper()}] M1 performance report saved to {report_paths[direction]}"
            )
            print(json_report)

            del final_model
            gc.collect()

            logging.info("\n" + "#" * 60)
            logging.info(f"### {direction.upper()} DONE (OOF + final model + calibration + report) ###")
            logging.info("#" * 60)

        logging.info("\n" + "=" * 60)
        logging.info("### M1-ONLY Integrated Training COMPLETED! ###")
        logging.info("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="(あ) M1 単独統合トレーナー (ABC→A / 全特徴量フル投下)"
    )
    parser.add_argument(
        "--test",
        action="store_true",
        help="Quick test: first few partitions / folds only.",
    )
    parser.add_argument(
        "--test-limit",
        type=int,
        default=0,
        help="Use only first N daily partitions (0 = all).",
    )
    parser.add_argument(
        "--test-fold-limit",
        type=int,
        default=0,
        help="Use only first N partitions per fold (0 = all).",
    )
    args = parser.parse_args()

    config = M1OnlyConfig()
    if args.test:
        config.test_limit = args.test_limit if args.test_limit > 0 else 20
        config.test_fold_limit = args.test_fold_limit if args.test_fold_limit > 0 else 5
    else:
        config.test_limit = args.test_limit
        config.test_fold_limit = args.test_fold_limit

    assembler = M1OnlyAssembler(config)
    assembler.run()
