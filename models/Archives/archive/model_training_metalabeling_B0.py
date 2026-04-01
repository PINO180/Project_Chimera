# /workspace/models/model_training_metalabeling_B.py

import sys
from pathlib import Path
import logging
import argparse
import datetime
import warnings
from dataclasses import dataclass
from typing import List
import shutil
from tqdm import tqdm

import polars as pl

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

    def run(self) -> None:
        logging.info(
            "### Script 2/3: Meta-Label Generation (Final Cleanup Version) ###"
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
        m1_oof_lf = pl.scan_parquet(self.config.m1_oof_path)

        total_records_processed = 0
        logging.info(
            f"Processing {len(self.partitions)} partitions to generate and write meta-labels..."
        )

        for partition_date in tqdm(self.partitions, desc="Generating Meta-Labels"):
            partition_path = str(
                self.config.weighted_dataset_path
                / f"year={partition_date.year}/month={partition_date.month}/day={partition_date.day}/*.parquet"
            )

            try:
                df_chunk = pl.read_parquet(partition_path)
            except Exception:
                continue

            if df_chunk.is_empty():
                continue

            # M1のOOF予測結果を結合 (ここで `uniqueness_right` が生成される)
            merged_chunk_lf = df_chunk.lazy().join(
                m1_oof_lf, on="timestamp", how="inner"
            )

            # メタラベルを計算
            final_chunk_lf = merged_chunk_lf.with_columns(
                pl.when((pl.col("prediction") > 0.5) & (pl.col("label") == 1))
                .then(1)
                .when(pl.col("prediction") > 0.5)
                .then(0)
                .otherwise(None)
                .alias("meta_label")
            ).rename({"prediction": "m1_pred_proba"})

            # --- ★★★ ここが最終修正箇所 ★★★ ---
            # 不要な副産物カラムを、存在しない場合でもエラーにならないように安全に削除
            columns_to_drop = ["uniqueness_right", "event_id"]
            # ★★★ Polarsが推奨する、警告の出ない書き方に修正 ★★★
            schema_columns = final_chunk_lf.collect_schema().names()
            existing_columns_to_drop = [
                col for col in columns_to_drop if col in schema_columns
            ]

            if existing_columns_to_drop:
                result_chunk = final_chunk_lf.drop(existing_columns_to_drop).collect()
            else:
                result_chunk = final_chunk_lf.collect()

            if not result_chunk.is_empty():
                output_partition_dir = (
                    self.config.output_dir
                    / f"year={partition_date.year}/month={partition_date.month}/day={partition_date.day}"
                )
                output_partition_dir.mkdir(parents=True, exist_ok=True)
                result_chunk.write_parquet(
                    output_partition_dir / "data.parquet", compression="zstd"
                )
                total_records_processed += len(result_chunk)

        logging.info("\n" + "=" * 60)
        if total_records_processed > 0:
            logging.info("### Script 2/3 FINISHED! You can now run Script C. ###")
            logging.info(
                f"  - Total records processed and saved: {total_records_processed}"
            )
            logging.info(
                f"  - Cleaned partitioned output is ready at: {self.config.output_dir}"
            )
        else:
            logging.error(
                "No data was processed. Please check the input files and logs."
            )
        logging.info("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Script 2/3: Meta-Label Generation")
    parser.add_argument(
        "--test",
        action="store_true",
        help="Run in quick test mode, processing only the first 5 partitions.",
    )
    args = parser.parse_args()
    config = MetaLabelingConfig(test=args.test)
    generator = MetaLabelGenerator(config)
    generator.run()
