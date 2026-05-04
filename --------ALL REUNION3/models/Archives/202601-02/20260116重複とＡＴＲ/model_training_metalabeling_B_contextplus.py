# models/model_training_metalabeling_B_contextplus.py
# [B案改: エース投入型M2 - Top50特徴量を利用]
# [修正版: 外部文脈廃止 + uniqueness/payoff_ratioの維持(Script C用)]

import sys
from pathlib import Path
import logging
import argparse
import datetime
import warnings
import json
import shutil
from dataclasses import dataclass
from typing import List

import polars as pl
from tqdm import tqdm

# --- プロジェクトのルートディレクトリをPythonの検索パスに追加 ---
project_root = Path(__file__).resolve().parents[1]
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

from blueprint import (
    S6_WEIGHTED_DATASET,
    S7_M1_OOF_PREDICTIONS,
    S7_META_LABELED_OOF_PARTITIONED,
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
    m1_oof_path: Path = S7_M1_OOF_PREDICTIONS
    weighted_dataset_path: Path = S6_WEIGHTED_DATASET
    output_dir: Path = S7_META_LABELED_OOF_PARTITIONED
    top_50_features_path: Path = project_root / "models" / "TOP_50_FEATURES.json"
    top_n_per_day: int = 20  # 各日のM1予測確率上位N件をサンプリングする
    test: bool = False


class MetaLabelGenerator:
    def __init__(self, config: MetaLabelingConfig):
        self.config = config
        self.partitions = self._discover_partitions()
        self.top_50_features = self._load_top_50_features()

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

    def _load_top_50_features(self) -> List[str]:
        logging.info(
            f"Loading Top 50 features list from {self.config.top_50_features_path}..."
        )
        if not self.config.top_50_features_path.exists():
            raise FileNotFoundError(
                f"Top 50 features file not found at: {self.config.top_50_features_path}"
            )

        with open(self.config.top_50_features_path, "r", encoding="utf-8") as f:
            features = json.load(f)

        logging.info(f"  -> Loaded {len(features)} features.")
        return features

    def run(self) -> None:
        logging.info(
            "### Script 2/3: Meta-Label Generation (Plan B Revised: Ace Injection) ###"
        )
        logging.info(
            f"Using dynamic sampling: Top {self.config.top_n_per_day} M1 predictions per day."
        )

        if not self.config.m1_oof_path.exists():
            raise FileNotFoundError(
                f"M1 OOF prediction file not found at: {self.config.m1_oof_path}. Please run Script A first."
            )

        if self.config.output_dir.exists():
            logging.warning(
                f"Output directory {self.config.output_dir} exists. Removing it for a clean run."
            )
            shutil.rmtree(self.config.output_dir)
        self.config.output_dir.mkdir(parents=True)

        logging.info(f"Loading M1 OOF predictions from {self.config.m1_oof_path}...")
        try:
            m1_oof_df = pl.read_parquet(self.config.m1_oof_path)

            # タイムゾーンの調整 (S6はUTC, S7はTZなしの場合があるためUTCに統一)
            m1_oof_df = m1_oof_df.with_columns(
                pl.col("timestamp").dt.replace_time_zone("UTC")
            )

            # 予測確率列をリネーム
            m1_oof_df = m1_oof_df.rename({"prediction": "m1_pred_proba"})
            # 日付列を追加 (サンプリング処理のため)
            m1_oof_df = m1_oof_df.with_columns(
                pl.col("timestamp").dt.date().alias("date")
            )
        except Exception as e:
            logging.error(
                f"Failed to load or process M1 OOF predictions: {e}", exc_info=True
            )
            return

        total_records_processed = 0
        logging.info(
            f"Processing {len(self.partitions)} partitions to generate and write meta-labels..."
        )

        # 抽出するカラムの定義: 基本情報 + Top 50特徴量 + [必須] uniqueness, payoff_ratio
        # ※ uniquenessはScript Cでの重み付けに必須
        columns_to_select = [
            "timestamp",
            "label",
            "uniqueness",
            "payoff_ratio",
        ] + self.top_50_features

        for partition_date in tqdm(self.partitions, desc="Generating Meta-Labels"):
            partition_path_glob = str(
                self.config.weighted_dataset_path
                / f"year={partition_date.year}/month={partition_date.month}/day={partition_date.day}/*.parquet"
            )

            try:
                # S6データセットの読み込み (必要なカラムが存在するかは後の工程で暗黙的にチェックされる)
                df_chunk = pl.read_parquet(partition_path_glob)
            except Exception:
                continue

            if df_chunk.is_empty():
                continue

            # --- (動的サンプリングとメタラベル生成ロジック) ---
            try:
                # 1. その日のM1 OOF予測データを取得
                daily_m1_oof = m1_oof_df.filter(pl.col("date") == partition_date)
                if daily_m1_oof.is_empty():
                    continue

                # 2. その日の予測確率上位N件のタイムスタンプを取得
                top_n_timestamps = (
                    daily_m1_oof.sort("m1_pred_proba", descending=True)
                    .head(self.config.top_n_per_day)
                    .select("timestamp")
                )
                if top_n_timestamps.is_empty():
                    continue

                # 3. 元のデータチャンク (df_chunk) を上位N件でフィルタリング
                #    同時に、必要なカラム (Top 50 + label + uniqueness) のみを選択して軽量化
                sampled_chunk_lf = (
                    df_chunk.lazy()
                    .join(top_n_timestamps.lazy(), on="timestamp", how="inner")
                    .select(
                        [col for col in columns_to_select if col in df_chunk.columns]
                    )
                )

                # 4. M1予測確率 (m1_pred_proba) を結合
                #    これにより、特徴量セットは [Top50 Features] + [m1_pred_proba] となる
                merged_chunk_lf = sampled_chunk_lf.join(
                    daily_m1_oof.lazy().select(["timestamp", "m1_pred_proba"]),
                    on="timestamp",
                    how="inner",
                )

                # 5. メタラベルを生成
                #    M1が正解(label=1)ならメタラベル1(Go)、不正解なら0(No-Go)
                final_chunk_lf = merged_chunk_lf.with_columns(
                    pl.when(pl.col("label") == 1)
                    .then(1)
                    .otherwise(0)
                    .alias("meta_label")
                )

                result_chunk = final_chunk_lf.collect(streaming=True)

            except Exception as e:
                logging.error(
                    f"Error processing partition {partition_date}: {e}", exc_info=False
                )
                continue
            # --- (ロジックここまで) ---

            if not result_chunk.is_empty():
                output_partition_dir = (
                    self.config.output_dir
                    / f"year={partition_date.year}/month={partition_date.month}/day={partition_date.day}"
                )
                output_partition_dir.mkdir(parents=True, exist_ok=True)

                # 結果の書き出し
                result_chunk.write_parquet(
                    output_partition_dir / "data.parquet", compression="zstd"
                )
                total_records_processed += len(result_chunk)

        logging.info("\n" + "=" * 60)
        if total_records_processed > 0:
            logging.info("### Script 2/3 FINISHED! (Plan B Revised) ###")
            logging.info(
                f"  - Total M2 training samples generated: {total_records_processed}"
            )
            logging.info(
                f"  - Meta-labeled partitioned output is ready at: {self.config.output_dir}"
            )
        else:
            logging.error(
                "No meta-labeled data was generated. Please check M1 predictions and logs."
            )
        logging.info("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Script 2/3: Meta-Label Generation (Plan B Revised)"
    )
    parser.add_argument(
        "--test",
        action="store_true",
        help="Run in quick test mode, processing only the first 5 partitions.",
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=20,
        help="Number of top M1 predictions per day to sample for M2 training.",
    )

    args = parser.parse_args()
    config = MetaLabelingConfig(test=args.test, top_n_per_day=args.top_n)

    generator = MetaLabelGenerator(config)
    generator.run()
