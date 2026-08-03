# /workspace/models/model_training_metalabeling_Bx2.py
# [2周目: メタラベル生成 — MFE/MAE 回帰版]
# [全載せ: final_feature_set_v5 を使用 (直交分割は廃止)]
# [logit変換は撤去: M1 の回帰予測 (mfe_atr/mae_atr) をそのまま m1_pred として M2 へ渡す]

import sys
from pathlib import Path
import logging
import argparse
import datetime
import warnings
import shutil
from dataclasses import dataclass
from typing import List

import numpy as np
import polars as pl
from tqdm import tqdm

# --- プロジェクトのルートディレクトリをPythonの検索パスに追加 ---
project_root = Path(__file__).resolve().parents[1]
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

from blueprint import (
    S6_LABELED_DATASET,
    S7_M1_OOF_PREDICTIONS_LONG,
    S7_M1_OOF_PREDICTIONS_SHORT,
    S7_META_LABELED_OOF_LONG,
    S7_META_LABELED_OOF_SHORT,
    S3_FEATURES_FOR_TRAINING_V5,
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
class MetaLabelingConfig:
    # [重み付けスキップ 2026-07-12] S6_LABELED を直読み(重み付けを飛ばす)。
    weighted_dataset_path: Path = S6_LABELED_DATASET
    m1_long_oof_path: Path = S7_M1_OOF_PREDICTIONS_LONG
    m1_short_oof_path: Path = S7_M1_OOF_PREDICTIONS_SHORT
    output_long_dir: Path = S7_META_LABELED_OOF_LONG
    output_short_dir: Path = S7_META_LABELED_OOF_SHORT
    feature_list_path: Path = S3_FEATURES_FOR_TRAINING_V5
    test: bool = False


class MetaLabelGenerator:
    def __init__(self, config: MetaLabelingConfig):
        self.config = config
        self.partitions = self._discover_partitions()

        if self.config.test:
            logging.warning(
                "--- TEST MODE: Processing only the first 5 partitions. ---"
            )
            self.partitions = self.partitions[:5]

    def _discover_partitions(self) -> List[datetime.date]:
        logging.info("Discovering all physical partitions from the weighted dataset...")
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
                )
            )
        )
        logging.info(f"  -> Discovered {len(dates)} daily partitions.")
        return dates

    def _load_features(self, feature_path: Path) -> List[str]:
        logging.info(f"Loading dedicated feature list from {feature_path.name}...")
        if not feature_path.exists():
            raise FileNotFoundError(f"Feature list file not found at: {feature_path}")

        with open(feature_path, "r") as f:
            raw_features = [line.strip() for line in f if line.strip()]

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
            "m1_pred",
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

        logging.info(f"  -> Loaded {len(features)} valid features.")
        return features

    def run(self) -> None:
        logging.info(
            "### Script 2/3: Meta-Label Generation (MFE/MAE回帰・全載せ・logit撤去) ###"
        )

        # [REGRESSION 2026-07-12] 全載せ: final_feature_set_v5 を M2 特徴量に。直交廃止。
        all_features = self._load_features(self.config.feature_list_path)
        # m1_pred は S6 に無く後段 join されるためリストから除外
        self.features = [
            f for f in all_features if f not in ("m1_pred", "m1_pred_proba")
        ]

        # 2モデル枠を mfe/mae に割り当て (mfe→LONG枠 / mae→SHORT枠)
        for target in ["mfe", "mae"]:
            logging.info(
                f"\n{'=' * 60}\n=== Meta-Label Generation for M2-{target.upper()} ({target}_atr) ===\n{'=' * 60}"
            )

            oof_path = (
                self.config.m1_long_oof_path
                if target == "mfe"
                else self.config.m1_short_oof_path
            )
            output_dir = (
                self.config.output_long_dir
                if target == "mfe"
                else self.config.output_short_dir
            )
            target_col = f"{target}_atr"  # meta_label = M2 の回帰ターゲット

            if not oof_path.exists():
                logging.error(
                    f"M1-{target.upper()} OOF prediction file not found at: {oof_path}. Skipping."
                )
                continue

            if output_dir.exists():
                logging.warning(
                    f"Output directory {output_dir} exists. Removing it for a clean run."
                )
                shutil.rmtree(output_dir)
            output_dir.mkdir(parents=True)

            # --- OOF予測のロード (回帰予測。logit変換は行わない) ---
            logging.info(
                f"Loading M1-{target.upper()} OOF predictions from {oof_path}..."
            )
            try:
                m1_oof_df = pl.read_parquet(oof_path)
                m1_oof_df = m1_oof_df.with_columns(
                    pl.col("timestamp").cast(pl.Datetime("us", "UTC"))
                )
                # [REGRESSION] M1 の回帰予測をそのまま m1_pred として渡す (logit変換撤去)。
                if "prediction" in m1_oof_df.columns:
                    m1_oof_df = m1_oof_df.rename({"prediction": "m1_pred"})
                m1_oof_df = m1_oof_df.with_columns(
                    pl.col("timestamp").dt.date().alias("date")
                )
            except Exception as e:
                logging.error(
                    f"Failed to load or process M1-{target.upper()} OOF predictions: {e}",
                    exc_info=True,
                )
                continue

            # --- 特徴量カラムの選定 ---
            columns_to_select = [
                "timestamp",
                "timeframe",
                "atr_value",
                target_col,
            ] + self.features

            total_records_processed = 0

            for partition_date in tqdm(
                self.partitions, desc=f"Generating M2-{target} Meta-Labels"
            ):
                s6_partition_path = (
                    self.config.weighted_dataset_path
                    / f"year={partition_date.year}/month={partition_date.month}/day={partition_date.day}/data.parquet"
                )

                try:
                    if not s6_partition_path.exists():
                        continue
                    df_chunk = pl.read_parquet(s6_partition_path)

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

                try:
                    daily_m1_oof = m1_oof_df.filter(pl.col("date") == partition_date)
                    if daily_m1_oof.is_empty():
                        continue

                    # [REGRESSION] 確率閾値(proba>=0.5)フィルタは撤去。M2 は全シグナルを
                    # 学習する (M1 の回帰予測を特徴に、実測 mfe_atr/mae_atr を refine)。
                    keys = daily_m1_oof.select(["timestamp", "timeframe"]).unique()
                    if keys.is_empty():
                        continue

                    sampled_chunk_lf = (
                        df_chunk.lazy()
                        .join(
                            keys.lazy(),
                            on=["timestamp", "timeframe"],
                            how="inner",
                        )
                        .select(
                            [
                                col
                                for col in columns_to_select
                                if col in df_chunk.columns
                            ]
                        )
                    )

                    # m1_pred (M1回帰予測) を結合
                    merged_chunk_lf = sampled_chunk_lf.join(
                        daily_m1_oof.lazy().select(
                            ["timestamp", "timeframe", "m1_pred"]
                        ),
                        on=["timestamp", "timeframe"],
                        how="inner",
                        coalesce=True,
                    )

                    merged_chunk_df = merged_chunk_lf.collect()
                    if merged_chunk_df.is_empty():
                        continue

                    # メタラベル = M2 の回帰ターゲット (mfe_atr / mae_atr)。null は除外。
                    final_chunk_df = merged_chunk_df.with_columns(
                        pl.col(target_col).alias("meta_label"),
                    ).filter(
                        pl.col("meta_label").is_not_null()
                        & pl.col("meta_label").is_not_nan()
                    )
                    if final_chunk_df.is_empty():
                        continue

                    output_partition_dir = (
                        output_dir
                        / f"year={partition_date.year}/month={partition_date.month}/day={partition_date.day}"
                    )
                    output_partition_dir.mkdir(parents=True, exist_ok=True)

                    final_chunk_df.write_parquet(
                        output_partition_dir / "data.parquet", compression="zstd"
                    )
                    total_records_processed += len(final_chunk_df)

                except Exception as e:
                    logging.error(
                        f"Error processing partition {partition_date} for {target}: {e}",
                        exc_info=True,
                    )
                    continue

            logging.info(
                f"Finished M2-{target.upper()} -> Total samples generated: {total_records_processed}"
            )

        logging.info("\n" + "=" * 60)
        logging.info("### Script 2/3 FINISHED! (MFE/MAE回帰・全載せ・logit撤去) ###")
        logging.info("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Script 2/3: Meta-Label Generation (直交分割版)"
    )
    parser.add_argument(
        "--test",
        action="store_true",
        help="Run in quick test mode, processing only the first 5 partitions.",
    )

    args = parser.parse_args()
    config = MetaLabelingConfig(test=args.test)
    generator = MetaLabelGenerator(config)
    generator.run()
