# /workspace/models/model_training_metalabeling_Ax2.py
# [1周目: M1 Two-Brain Cross-Validation]
# [直交分割版: S3_SELECTED_FEATURES_ORTHOGONAL_DIR から方向別特徴量を読み込む]

import os
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

# --- プロジェクトのルートディレクトリをPythonの検索パスに追加 ---
project_root = Path(__file__).resolve().parents[1]
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

from blueprint import (
    S6_LABELED_DATASET,
    S3_FEATURES_FOR_TRAINING_V5,
    S7_M1_OOF_PREDICTIONS_LONG,
    S7_M1_OOF_PREDICTIONS_SHORT,
)

# [B1 2026-07-15] 方向ターゲット (net_dominance = mfe_atr − mae_atr) の M1 OOF 出力先。
#   blueprint 非改変で既存 OOF と同じフォルダ (stratum_7_models) に置く。
S7_M1_OOF_PREDICTIONS_DOMINANCE = (
    S7_M1_OOF_PREDICTIONS_LONG.parent / "m1_oof_predictions_dominance.parquet"
)


# ============================================================================
# [母集団フィルタ 2026-07-16] 旧トリプルバリアの ATR_RATIO ゲート復活用。
# ----------------------------------------------------------------------------
#   経緯: 旧ラベリングは is_trigger = (atr_ratio >= ATR_RATIO_THRESHOLD=0.8) で低ボラを
#   学習から除外していたが、回帰化の際に「脳に委ねる」思想で撤廃され、全バー
#   (is_trigger=1) が学習対象になった。その結果:
#     - 低ボラ帯は ATR 建て目標値が 3-4 倍大きく (ATR が分母のため)、L1 損失を支配する
#     - = 実際には取引しないバーに学習容量が食われている可能性
#   ここで母集団を絞れば、ラベリング側で絞ったのと学習集合は同一になる (再ラベリング不要)。
#
#   TRAIN_MIN_ATR_RATIO : この値未満の atr_ratio を学習/OOF から除外。0.0 = 無効(全バー)。
#                         0.8 = 旧 ATR_RATIO_THRESHOLD 相当。
#   TRAIN_MAX_ATR_RATIO : この値以上を除外。inf = 上限なし。
#   ※ フィルタが有効なとき、OOF 出力名に _atrXXX サフィックスが付く (無濾過 OOF を上書きしない)。
#   ※ 特徴量選択 (Chapter2) は全母集団で実施済み。厳密には絞った母集団で再選択すべきだが、
#     本実験は「損失の配分が変わると学習が改善するか」の検定なので特徴量は据え置く。
# ============================================================================
TRAIN_MIN_ATR_RATIO = 0.8
TRAIN_MAX_ATR_RATIO = float("inf")


def _population_filter_tag() -> str:
    """フィルタ有効時に OOF 出力名へ付けるサフィックス (無効なら空文字)。"""
    if TRAIN_MIN_ATR_RATIO <= 0.0 and not np.isfinite(TRAIN_MAX_ATR_RATIO):
        return ""
    lo = f"{int(round(TRAIN_MIN_ATR_RATIO * 100)):03d}"
    hi = (
        "inf"
        if not np.isfinite(TRAIN_MAX_ATR_RATIO)
        else f"{int(round(TRAIN_MAX_ATR_RATIO * 100)):03d}"
    )
    return f"_atr{lo}-{hi}"


def _apply_population_filter(df: "pl.DataFrame") -> "pl.DataFrame":
    """is_trigger と atr_ratio 帯で学習母集団を絞る。学習fold/検証foldの両方で使う。"""
    if "is_trigger" in df.columns:
        df = df.filter(pl.col("is_trigger") == 1)
    if "atr_ratio" in df.columns:
        if TRAIN_MIN_ATR_RATIO > 0.0:
            df = df.filter(
                pl.col("atr_ratio").is_not_null()
                & (pl.col("atr_ratio") >= TRAIN_MIN_ATR_RATIO)
            )
        if np.isfinite(TRAIN_MAX_ATR_RATIO):
            df = df.filter(
                pl.col("atr_ratio").is_not_null()
                & (pl.col("atr_ratio") < TRAIN_MAX_ATR_RATIO)
            )
    return df


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
    # [重み付けスキップ 2026-07-12] uniqueness を使わない回帰学習のため、
    # 重み付け(triple-barrier依存)を飛ばして S6_LABELED を直読みする。
    input_dir: Path = S6_LABELED_DATASET
    n_splits: int = 5
    purge_days: int = 3
    embargo_days: int = 2
    lgbm_params: Dict[str, Any] = field(
        default_factory=lambda: {
            # [REGRESSION 2026-07-12] MFE/MAE(ATR建て)を回帰。分位ではなく素のL2で
            # 条件付き期待値を当てる(①点予測=取引ごとに動的)。metric=l1(平均絶対誤差)。
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
            "n_jobs": os.cpu_count(),  # [PERF 2026-07-12] 全コア使用 (旧:1=1コア固定で激遅)
            "verbose": 1,
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

    def _resolve_all_features(self) -> List[str]:
        # [REGRESSION 2026-07-12 全載せ] final_feature_set_v5.txt を全載せリストとして使用。
        # 直交/方向別リストは廃止(旧バリア基準 L 向けチューニングのため今は前提が崩れる)。
        # 念のため下段の exclude_exact でメタ/ラベル/ターゲット混入も二重に防ぐ。
        feature_path = S3_FEATURES_FOR_TRAINING_V5
        logging.info(f"Loading ALL features (全載せ) from {feature_path}...")
        if not feature_path.exists():
            raise FileNotFoundError(f"Feature list file not found: {feature_path}")
        with open(feature_path, "r") as f:
            raw_features = [line.strip() for line in f if line.strip()]

        # メタデータ・未来情報の完全除外
        # 【Phase 5 修正 (#35)】 Ax2/Bx2/Cx2 の exclude_exact を統一 (union)
        # 各ファイル間で不整合があった項目 (concurrency_long/short, duration_long/short,
        # payoff_ratio_long/short 等) を全て含めることで、3 ファイル間の挙動を一致させる。
        # disc は学習対象外メタデータ (週末跨ぎギャップ判定 bool 列) — 最終防御線。
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
            # --- メタラベリング系 ---
            "m1_pred_proba",
            "meta_label",
            # --- 【Phase 5 修正】学習対象外メタデータ ---
            "disc",  # 週末跨ぎギャップ判定 bool 列 (engine_1_C 経由で漏れる可能性に対する最終防御線)
            # --- [REGRESSION 2026-07-12] MFE/MAE 回帰ターゲット群 (答えのリーク防止で除外) ---
            "mfe_usd",
            "mae_usd",
            "mfe_atr",
            "mae_atr",
            "mfe_direction",
            "mfe_dominance_atr",
            # session_atr_ratio(サフィックス無し=シミュレーター用)は除外。
            # session_atr_ratio_M3 は特徴量として残す(全載せに含める)。
            "session_atr_ratio",
        }

        features = []
        for col in raw_features:
            if col in exclude_exact:
                continue
            if col.startswith("is_trigger_on"):
                continue
            features.append(col)

        logging.info(f"   -> Loaded {len(features)} valid features.")
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

    def _train_model_partition_based(self, target: str) -> Dict[str, np.ndarray]:
        # [REGRESSION] target ∈ {"mfe","mae","mfe_dominance"} → ターゲット列 = {target}_atr。
        logging.info(f"--- Starting BATCH Training for M1 ({target.upper()}) ---")
        kfold = PartitionPurgedKFold(
            self.config.n_splits, self.config.purge_days, self.config.embargo_days
        )

        target_col = f"{target}_atr"  # mfe_atr / mae_atr (ATR建て回帰ターゲット)

        oof_results = {
            "timestamp": [],
            "prediction": [],
            "true_label": [],  # 実測ターゲット値(float)。キー名は下流互換のため据え置き。
            "timeframe": [],
        }

        for i, (train_partitions, val_partitions) in enumerate(
            kfold.split(self.partitions)
        ):
            logging.info(f"  [{target.upper()}] Fold {i + 1}/{self.config.n_splits}...")

            if self.config.test_fold_limit > 0:
                train_partitions = train_partitions[: self.config.test_fold_limit]
                val_partitions = val_partitions[: self.config.test_fold_limit]

            X_train_list, y_train_list = [], []

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
                        df_chunk = _apply_population_filter(df_chunk)

                        # timeframeは学習特徴量から除外、データ管理用としてunique処理のみ
                        if "timeframe" in df_chunk.columns:
                            df_chunk = df_chunk.unique(
                                subset=["timestamp", "timeframe"],
                                keep="last",
                                maintain_order=True,
                            )
                        # [REGRESSION] ターゲット(mfe_atr/mae_atr)が欠損の行は学習から除外。
                        # null だけでなく NaN も落とす (polars では NaN≠null、is_not_null では
                        # NaN が捕まらないため。窓内tick無しは労ベリングで null 化済みだが二重防御)。
                        df_chunk = df_chunk.filter(
                            pl.col(target_col).is_not_null()
                            & pl.col(target_col).is_not_nan()
                        )
                    except Exception:
                        continue

                    if df_chunk.is_empty():
                        continue

                    X_train_list.append(
                        df_chunk.select(self.features).fill_null(0).to_numpy()
                    )
                    y_train_list.append(
                        df_chunk[target_col].cast(pl.Float64).to_numpy()
                    )

            model = None
            if len(X_train_list) > 0:
                try:
                    X_train = np.concatenate(X_train_list)
                    y_train = np.concatenate(y_train_list)

                    del X_train_list, y_train_list
                    gc.collect()

                    train_params = self.config.lgbm_params.copy()
                    n_estimators = train_params.pop("n_estimators", 1000)

                    logging.info(f"    -> fitting model on {len(X_train)} samples...")

                    # [REGRESSION] サンプル重みは付けない (unweighted)。uniqueness の
                    # 生成源 = triple-barrier を撤去したため。必要なら後で TO 窓ベースの
                    # 新 uniqueness を weight= に渡す形で再導入できる。
                    model = lgb.train(
                        train_params,
                        lgb.Dataset(
                            X_train,
                            label=y_train,
                            feature_name=self.features,
                        ),
                        num_boost_round=n_estimators,
                    )

                    del X_train, y_train
                    gc.collect()

                except Exception as fit_error:
                    logging.error(
                        f"Error during batch training ({target}): {fit_error}",
                        exc_info=True,
                    )
                    model = None
            else:
                logging.warning(
                    f"    -> No training data found for this fold ({target})."
                )

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
                        df_chunk = _apply_population_filter(df_chunk)

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
                            f"Error during prediction ({target}): {pred_error}"
                        )
                        predictions = np.full(len(df_chunk), np.nan)

                    oof_results["timestamp"].append(df_chunk["timestamp"].to_numpy())
                    oof_results["prediction"].append(predictions)
                    # 実測ターゲット値(float)。null は後段のメトリクスで除外される。
                    oof_results["true_label"].append(
                        df_chunk[target_col].cast(pl.Float64).to_numpy()
                    )
                    oof_results["timeframe"].append(df_chunk["timeframe"].to_numpy())

            del model
            gc.collect()

        logging.info(f"Concatenating OOF results for {target}...")
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
        logging.info("### Script 1/3: M1 MFE/MAE Regression CV (全載せ・直交廃止) ###")
        S7_M1_OOF_PREDICTIONS_LONG.parent.mkdir(parents=True, exist_ok=True)
        # [REGRESSION 2026-07-12] 2モデル枠を mfe/mae に割り当て。
        #   mfe → M1-mfe (mfe_atr を回帰) → 既存 LONG 出力枠を流用
        #   mae → M1-mae (mae_atr を回帰) → 既存 SHORT 出力枠を流用
        # [B1] mfe_dominance (= mfe_atr − mae_atr、符号=方向・大きさ=確信度) を第3ターゲットに。
        #   target_col は f"{target}_atr" 規約でそのまま mfe_dominance_atr が効く。
        #   スプレッド控除は mfe/mae 両方から引かれ差では打ち消すため dominance はスプレッド不変。
        targets = ["mfe", "mae", "mfe_dominance"]
        _tag = _population_filter_tag()

        def _tagged(pth: Path) -> Path:
            # フィルタ有効時のみ _atrXXX-YYY を付ける (無濾過 OOF を上書きしない)
            return pth if not _tag else pth.with_name(f"{pth.stem}{_tag}{pth.suffix}")

        output_paths = {
            "mfe": _tagged(S7_M1_OOF_PREDICTIONS_LONG),
            "mae": _tagged(S7_M1_OOF_PREDICTIONS_SHORT),
            "mfe_dominance": _tagged(S7_M1_OOF_PREDICTIONS_DOMINANCE),
        }
        if _tag:
            logging.info(
                f"### [母集団フィルタ有効] atr_ratio in "
                f"[{TRAIN_MIN_ATR_RATIO}, {TRAIN_MAX_ATR_RATIO}) "
                f"→ OOF 出力: *{_tag}.parquet ###"
            )

        # 全載せ: 特徴量は S6 全列 − メタ/ラベル/ターゲット。両ターゲットで共通。
        self.features = self._resolve_all_features()
        logging.info(f"   -> ALL-features count (全載せ): {len(self.features)}")

        for target in targets:
            logging.info("\n" + "=" * 50)
            logging.info(
                f"=== Starting Pipeline for: M1-{target.upper()} ({target}_atr) ==="
            )
            logging.info("=" * 50)

            oof_results = self._train_model_partition_based(target)

            if oof_results["timestamp"].size > 0:
                oof_df = pl.DataFrame(
                    {
                        "timestamp": oof_results["timestamp"],
                        "timeframe": oof_results["timeframe"],
                        "prediction": oof_results["prediction"],
                        "true_label": oof_results["true_label"],
                    }
                )

                oof_df = oof_df.with_columns(
                    pl.col("timestamp").cast(pl.Datetime("us", "UTC")),
                    pl.col("timeframe").cast(pl.Utf8),
                ).sort(["timestamp", "timeframe"])

                out_path = output_paths[target]
                oof_df.write_parquet(out_path, compression="zstd")

                logging.info(
                    f"Successfully saved M1-{target.upper()} OOF predictions to: {out_path}"
                )
                logging.info(f"  - Total predictions saved: {len(oof_df)}")

                # =====================================================
                # [REGRESSION-REPORT 2026-07-12] AUC を廃止し、回帰の効き目を表示。
                #   - MAE (平均絶対誤差): 予測が平均どれだけ外したか (ATR建て、小=良)。
                #   - Pearson / Spearman 相関 (実測 vs 予測): 大小の順序を当てているか
                #     (1に近いほど良、分類の AUC に相当する "効いてる感" の指標)。
                #   材料は oof_df の prediction と true_label(実測ターゲット値)。null/inf は除外。
                # =====================================================
                try:
                    _y_true = oof_df["true_label"].to_numpy().astype(float)
                    _y_pred = oof_df["prediction"].to_numpy().astype(float)

                    _valid = (
                        ~np.isnan(_y_true)
                        & ~np.isnan(_y_pred)
                        & ~np.isinf(_y_pred)
                        & ~np.isinf(_y_true)
                    )
                    _n_valid = int(_valid.sum())
                    _n_drop = int(_y_true.shape[0] - _n_valid)

                    _yt = _y_true[_valid]
                    _yp = _y_pred[_valid]

                    logging.info("-" * 50)
                    logging.info(
                        f"### M1-{target.upper()} OOF REGRESSION REPORT ({target}_atr) ###"
                    )
                    logging.info(
                        f"  - Valid samples: {_n_valid:,} (dropped NaN/inf: {_n_drop:,})"
                    )

                    if _n_valid < 2:
                        logging.warning("  - Metrics skipped: <2 valid rows.")
                    else:
                        _mae = float(np.mean(np.abs(_yt - _yp)))
                        _rmse = float(np.sqrt(np.mean((_yt - _yp) ** 2)))
                        _bias = float(np.mean(_yp - _yt))
                        # Pearson 相関 (定数列だと nan になるのでガード)
                        if _yt.std() > 0 and _yp.std() > 0:
                            _pearson = float(np.corrcoef(_yt, _yp)[0, 1])
                        else:
                            _pearson = float("nan")
                        # Spearman (順位相関) = 順位化して Pearson
                        _rt = np.argsort(np.argsort(_yt)).astype(float)
                        _rp = np.argsort(np.argsort(_yp)).astype(float)
                        if _rt.std() > 0 and _rp.std() > 0:
                            _spearman = float(np.corrcoef(_rt, _rp)[0, 1])
                        else:
                            _spearman = float("nan")

                        logging.info(f"  - MAE  (平均絶対誤差, ATR): {_mae:.4f}")
                        logging.info(f"  - RMSE (二乗平均平方根, ATR): {_rmse:.4f}")
                        logging.info(f"  - Bias (予測−実測の平均): {_bias:+.4f}")
                        logging.info(f"  - Pearson  corr (実測vs予測): {_pearson:.4f}")
                        logging.info(f"  - Spearman corr (順位相関)  : {_spearman:.4f}")
                        logging.info(
                            f"  - Pred : min={_yp.min():.4f} / mean={_yp.mean():.4f} / max={_yp.max():.4f}"
                        )
                        logging.info(
                            f"  - True : min={_yt.min():.4f} / mean={_yt.mean():.4f} / max={_yt.max():.4f}"
                        )
                    logging.info("-" * 50)
                except Exception as _rep_err:
                    logging.warning(
                        f"Regression report failed for {target}: {_rep_err}",
                        exc_info=True,
                    )
            else:
                logging.warning(f"No OOF predictions generated for M1-{target}.")

        logging.info("\n" + "=" * 60)
        logging.info("### M1 Two-Brain Training COMPLETED! (直交分割版) ###")
        logging.info("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Script 1/3: M1 Two-Brain Cross-Validation (直交分割版)"
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
