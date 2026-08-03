# /workspace/models/model_training_metalabeling_Cx2.py
# [1周目: M2 CV・最終学習・較正・レポート生成]
# [MFE/MAE 回帰版: 全載せ (final_feature_set_v5)、直交分割・確率キャリブレーション廃止]

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
import joblib
from typing import List, Tuple, Dict, Any, Generator
from tqdm import tqdm

# --- プロジェクトのルートディレクトリをPythonの検索パスに追加 ---
project_root = Path(__file__).resolve().parents[1]
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

from blueprint import (
    S6_LABELED_DATASET,
    S3_FEATURES_FOR_TRAINING_V5,
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
    direction: str = "long"
    # [重み付けスキップ 2026-07-12] S6_LABELED を直読み(重み付けを飛ばす)。
    weighted_dataset_path: Path = S6_LABELED_DATASET
    meta_labeled_oof_path: Path = field(default=None, init=False)
    m1_oof_path_directed: Path = field(default=None, init=False)
    feature_list_path: Path = S3_FEATURES_FOR_TRAINING_V5
    n_splits: int = 5
    purge_days: int = 3
    embargo_days: int = 2
    lgbm_params_m1: Dict[str, Any] = field(
        default_factory=lambda: {
            "objective": "regression",
            "metric": "l1",
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
    lgbm_params_m2: Dict[str, Any] = field(
        default_factory=lambda: {
            "objective": "regression",
            "metric": "l1",
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
    test: bool = False

    def __post_init__(self):
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
        self.config = config
        self.direction = self.config.direction
        # [REGRESSION 2026-07-12] direction "long"/"short" を回帰ターゲット mfe/mae に対応。
        #   long → mfe (mfe_atr), short → mae (mae_atr)。パスは既存 LONG/SHORT 枠を流用。
        self.target = "mfe" if self.direction == "long" else "mae"
        self.target_col = f"{self.target}_atr"

        # [全載せ] M1/M2 とも final_feature_set_v5 を使用 (直交分割は廃止)。
        all_features = self._load_features(self.config.feature_list_path)
        self.features_base: List[str] = [
            f for f in all_features if f not in ("m1_pred", "m1_pred_proba")
        ]
        # M2 は M1 の回帰予測 m1_pred を特徴に加える (Bx2 が join 時に付与)。
        self.features_m2: List[str] = list(self.features_base)
        if "m1_pred" not in self.features_m2:
            self.features_m2.append("m1_pred")

        self.partitions_m1_final = self._discover_partitions_for_m1_final_train()
        self.partitions_m2 = self._discover_partitions()

        # [REGRESSION] scale_pos_weight(二値専用)は撤去。回帰は重み無し。
        logging.info(
            f"[{self.direction.upper()}→{self.target.upper()}] Regression (target={self.target_col}), "
            f"features(全載せ) M1={len(self.features_base)} / M2={len(self.features_m2)}"
        )

    def _load_features(self, feature_path: Path) -> List[str]:
        logging.info(
            f"[{self.direction.upper()}] Loading dedicated feature list from {feature_path.name}..."
        )
        with open(feature_path, "r") as f:
            raw_features = [line.strip() for line in f if line.strip()]

        # 【Phase 5 修正 (#35)】 Ax2/Bx2/Cx2 の exclude_exact を統一 (union)
        # 各ファイル間で不整合があった項目 (concurrency_long/short, duration_long/short,
        # payoff_ratio_long/short 等) を全て含めることで、3 ファイル間の挙動を一致させる。
        # disc は学習対象外メタデータ (週末跨ぎギャップ判定 bool 列) — 最終防御線。
        exclude_exact = {
            # --- 基本メタ ---
            "timestamp",
            "timeframe",  # 学習特徴量から除外（_filter_dataframe()でのデータ管理用途は継続）
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
            # --- メタラベリング系 ---
            "m1_pred_proba",
            "m1_pred",  # 後段でjoinされるため特徴量リストから除外
            "meta_label",
            # --- 【Phase 5 修正】学習対象外メタデータ ---
            "disc",
            # --- [REGRESSION 2026-07-12] MFE/MAE 回帰ターゲット群 ---
            "mfe_usd",
            "mae_usd",
            "mfe_atr",
            "mae_atr",
            "mfe_direction",
            "mfe_dominance_atr",
            "session_atr_ratio",  # サフィックス無し=シミュレーター用 (｜_M3 は特徴量)
        }

        features = []
        for col in raw_features:
            if col in exclude_exact:
                continue
            if col.startswith("is_trigger_on"):
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

        if "timeframe" in df.columns:
            # 文字列型の場合のみreplaceを実行し、それ以外はただInt32にキャストする
            if df.schema["timeframe"] in [pl.Utf8, pl.String]:
                df = df.with_columns(
                    pl.col("timeframe")
                    .replace({"M0.5": 0, "M1": 1, "M3": 2, "M5": 3, "M8": 4, "M15": 5})
                    .cast(pl.Int32)
                )
            else:
                df = df.with_columns(pl.col("timeframe").cast(pl.Int32))

            # 行増殖バグを最後の行で完全鎮圧
            df = df.unique(
                subset=["timestamp", "timeframe"], keep="last", maintain_order=True
            )

        return df

    def run(self):
        logging.info(
            f"[{self.direction.upper()}→{self.target.upper()}] ### Script 3/3: M2 CV + Final Regression Training + Report ###"
        )

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

        self._train_and_calibrate_final_models()
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

            X_train_list, y_train_list = [], []

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

                    df_chunk = df_chunk.filter(
                    pl.col("meta_label").is_not_null() & pl.col("meta_label").is_not_nan()
                )
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
                    y_train_list.append(
                        df_chunk["meta_label"].cast(pl.Float64).to_numpy()
                    )

            model = None
            if len(X_train_list) > 0:
                try:
                    X_train = np.concatenate(X_train_list)
                    y_train = np.concatenate(y_train_list)

                    del X_train_list, y_train_list
                    gc.collect()

                    train_params = self.config.lgbm_params_m2.copy()
                    n_estimators = train_params.pop("n_estimators", 1000)

                    logging.info(
                        f"[{self.direction.upper()}]    -> Training M2 model on {len(X_train)} samples..."
                    )
                    model = lgb.train(
                        train_params,
                        lgb.Dataset(
                            X_train,
                            label=y_train,
                            feature_name=self.features_m2,
                        ),
                        num_boost_round=n_estimators,
                    )

                    del X_train, y_train
                    gc.collect()

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

                df_chunk = df_chunk.filter(
                    pl.col("meta_label").is_not_null() & pl.col("meta_label").is_not_nan()
                )
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
                        "timeframe": df_chunk["timeframe"],
                        "prediction": predictions,
                        "true_label": df_chunk["meta_label"],
                    }
                )

                # timeframeをInt32→文字列に復元（下流のシミュレーター向け）
                reverse_map = {0: "M0.5", 1: "M1", 2: "M3", 3: "M5", 4: "M8", 5: "M15"}
                oof_df = oof_df.with_columns(
                    pl.col("timeframe")
                    .replace_strict(reverse_map, default=None)
                    .cast(pl.Utf8)
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
        # [REGRESSION 2026-07-12] 確率キャリブレーション(IsotonicRegression)を撤去。
        #   回帰出力は確率ではないため較正の概念が無い。最終 M1/M2 モデルを全期間で
        #   学習・保存するのみ。較正の代わりに残差/MAE は _generate_performance_report で確認。
        logging.info(
            f"[{self.direction.upper()}→{self.target.upper()}] --- Training Final Regression Models (no calibration) ---"
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

        # 較正済みパスにも生モデルを保存しておく (下流のパス契約を壊さないため。
        # 中身は生の回帰モデル = 較正なし)。本番のモデル読込は生モデルを指すこと。
        try:
            if m1_model is not None:
                joblib.dump(m1_model, self.config.m1_calibrated)
            if m2_model is not None:
                joblib.dump(m2_model, self.config.m2_calibrated)
            logging.info(
                f"[{self.direction.upper()}]   - Saved raw regression models to calibrated paths (no isotonic)."
            )
        except Exception as _e:
            logging.warning(f"[{self.direction.upper()}] Failed saving models: {_e}")

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
                "meta_label" if is_m2 else self.target_col,
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

        model = None
        input_path = (
            self.config.meta_labeled_oof_path
            if is_m2
            else self.config.weighted_dataset_path
        )

        X_list, y_list = [], []

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

            # [REGRESSION] M1/M2 とも回帰ターゲット。null 行は除外。
            if target_col in df_chunk.columns:
                df_chunk = df_chunk.filter(
                    pl.col(target_col).is_not_null() & pl.col(target_col).is_not_nan()
                )
            if df_chunk.is_empty():
                continue

            features_to_use = self.features_m2 if is_m2 else self.features_base
            missing_features = [f for f in features_to_use if f not in df_chunk.columns]
            if missing_features:
                continue

            X_chunk = df_chunk.select(features_to_use).fill_null(0).to_numpy()
            y_chunk = df_chunk[target_col].cast(pl.Float64).to_numpy()

            X_list.append(X_chunk)
            y_list.append(y_chunk)

        if not X_list:
            raise RuntimeError(
                f"[{self.direction.upper()}] No data found for {model_name}."
            )

        try:
            X_train = np.concatenate(X_list)
            y_train = np.concatenate(y_list)

            del X_list, y_list
            gc.collect()

            train_params = lgbm_params.copy()
            n_estimators = train_params.pop("n_estimators", 1000)

            logging.info(
                f"[{self.direction.upper()}]     -> fitting {model_name} on {len(X_train)} samples..."
            )

            # [REGRESSION] 重み無し (uniqueness の生成源 triple-barrier を撤去したため)。
            model = lgb.train(
                train_params,
                lgb.Dataset(
                    X_train,
                    label=y_train,
                    feature_name=features_to_use,
                ),
                num_boost_round=n_estimators,
            )

            del X_train, y_train
            gc.collect()

        except Exception as fit_error:
            raise RuntimeError(
                f"[{self.direction.upper()}] Failed to train {model_name}: {fit_error}"
            )

        return model

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
            ).collect(engine="streaming")
            m2_oof_df.sort(["timestamp", "timeframe"]).write_parquet(
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
        # [REGRESSION 2026-07-12] AUC を廃止し、M1/M2 の OOF 回帰メトリクスを出力。
        #   MAE / RMSE / Bias / Pearson / Spearman を M1(direct OOF) と M2(aggregated OOF) で。
        logging.info(
            f"[{self.direction.upper()}→{self.target.upper()}] --- Generating Final Regression Report ---"
        )
        report = {"target": self.target_col}

        def _reg_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
            valid = (
                ~np.isnan(y_true)
                & ~np.isnan(y_pred)
                & ~np.isinf(y_true)
                & ~np.isinf(y_pred)
            )
            n = int(valid.sum())
            if n < 2:
                return {"n": n, "mae": float("nan"), "corr": float("nan")}
            yt, yp = y_true[valid], y_pred[valid]
            mae = float(np.mean(np.abs(yt - yp)))
            rmse = float(np.sqrt(np.mean((yt - yp) ** 2)))
            bias = float(np.mean(yp - yt))
            pear = (
                float(np.corrcoef(yt, yp)[0, 1])
                if yt.std() > 0 and yp.std() > 0
                else float("nan")
            )
            rt = np.argsort(np.argsort(yt)).astype(float)
            rp = np.argsort(np.argsort(yp)).astype(float)
            spear = (
                float(np.corrcoef(rt, rp)[0, 1])
                if rt.std() > 0 and rp.std() > 0
                else float("nan")
            )
            return {
                "n": n, "mae": mae, "rmse": rmse, "bias": bias,
                "pearson": pear, "spearman": spear,
                "pred_mean": float(yp.mean()), "true_mean": float(yt.mean()),
            }

        m1_oof_path_directed = self.config.m1_oof_path_directed
        if m1_oof_path_directed.exists():
            try:
                m1_oof_df = pl.read_parquet(m1_oof_path_directed)
                if not m1_oof_df.is_empty():
                    y_true_m1 = m1_oof_df["true_label"].cast(pl.Float64).to_numpy()
                    y_pred_m1 = m1_oof_df["prediction"].cast(pl.Float64).to_numpy()
                    del m1_oof_df
                    report["m1_performance"] = _reg_metrics(y_true_m1, y_pred_m1)
                    logging.info(
                        f"[{self.direction.upper()}]   -> M1 regression metrics: {report['m1_performance']}"
                    )
            except Exception as e:
                logging.error(
                    f"[{self.direction.upper()}] Could not process M1 OOF predictions: {e}"
                )
                report["m1_performance"] = {"mae": float("nan")}

        if self.config.m2_oof_predictions.exists():
            try:
                m2_oof_df = pl.read_parquet(self.config.m2_oof_predictions)
                m2_oof_df = m2_oof_df.filter(
                    pl.col("true_label").is_not_null() & pl.col("true_label").is_not_nan()
                )
                if not m2_oof_df.is_empty():
                    y_true_m2 = m2_oof_df["true_label"].cast(pl.Float64).to_numpy()
                    y_pred_m2 = m2_oof_df["prediction"].cast(pl.Float64).to_numpy()
                    del m2_oof_df
                    report["m2_performance"] = _reg_metrics(y_true_m2, y_pred_m2)
                    logging.info(
                        f"[{self.direction.upper()}]   -> M2 regression metrics: {report['m2_performance']}"
                    )
            except Exception as e:
                logging.error(
                    f"[{self.direction.upper()}] Could not process aggregated M2 OOF predictions: {e}"
                )
                report["m2_performance"] = {"mae": float("nan")}

        self.config.performance_report.parent.mkdir(parents=True, exist_ok=True)
        json_report = json.dumps(report, indent=4).replace("NaN", "null")
        with open(self.config.performance_report, "w") as f:
            f.write(json_report)

        logging.info(
            f"[{self.direction.upper()}] Performance report saved to {self.config.performance_report}"
        )
        print(f"\n[{self.direction.upper()}→{self.target.upper()}] Regression Report:")
        print(json_report)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Script 3/3: M2 CV + Final Regression Training + Report (MFE/MAE回帰・全載せ)"
    )
    parser.add_argument("--test", action="store_true", help="Run in quick test mode.")
    args = parser.parse_args()

    for direction in ["long", "short"]:
        print("\n" + "=" * 60)
        print(f"### STARTING PROCESSING FOR DIRECTION: {direction.upper()} ###")
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
        + "\n### ALL STAGES (MFE & MAE) COMPLETED! (回帰版) ###\n"
        + "=" * 60
    )
