# /workspace/models/model_training_metalabeling_B.py
# [修正版: タイムゾーン不一致エラーを修正]
# [修正版: フェーズ6 (V4) - 市場文脈特徴量(S7_CONTEXT_FEATURES)を結合]

import sys
from pathlib import Path
import logging
import argparse
import datetime
import warnings
from dataclasses import dataclass, field
from typing import List
import shutil
from tqdm import tqdm

import polars as pl

# --- プロジェクトのルートディレクトリをPythonの検索パスに追加 ---
project_root = Path(__file__).resolve().parents[1]
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

from blueprint import (
    S6_WEIGHTED_DATASET,
    S7_M1_OOF_PREDICTIONS,
    S7_META_LABELED_OOF_PARTITIONED,
    S7_CONTEXT_FEATURES,  # --- ★★★ [修正] インポート追加 ★★★ ---
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
    top_n_per_day: int = 20  # 各日のM1予測確率上位N件をサンプリングする
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
            "### Script 2/3: Meta-Label Generation (Dynamic Sampling Version) ###"
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

            # --- ★★★ (維持) ここがエラー修正箇所 ★★★ ---
            # S6 (左側) は UTC (datetime[us, UTC]) を持つ
            # S7 (右側) はタイムゾーンを失っている (datetime[us])
            # S7 (m1_oof_df) に UTC タイムゾーンを明示的に付与して型を一致させる
            m1_oof_df = m1_oof_df.with_columns(
                pl.col("timestamp").dt.replace_time_zone("UTC")
            )
            # --- ★★★ 修正ここまで ★★★ ---

            # 予測確率列をリネームしておく
            m1_oof_df = m1_oof_df.rename({"prediction": "m1_pred_proba"})
            # 日付列を追加しておく (サンプリング処理のため)
            m1_oof_df = m1_oof_df.with_columns(
                pl.col("timestamp").dt.date().alias("date")
            )
        except Exception as e:
            logging.error(
                f"Failed to load or process M1 OOF predictions: {e}", exc_info=True
            )
            return

        # --- ★★★ [修正] S7_CONTEXT_FEATURES を読み込む ★★★ ---
        logging.info(f"Loading context features from {S7_CONTEXT_FEATURES}...")
        try:
            context_df = pl.read_parquet(S7_CONTEXT_FEATURES)
            # join_asof のために 'timestamp' を 'date' にキャストして準備
            context_df = context_df.with_columns(
                pl.col("timestamp").dt.date().alias("date")
            ).sort("date")
        except Exception as e:
            logging.error(
                f"Failed to load or process context features: {e}", exc_info=True
            )
            return
        # --- ★★★ [修正] ここまで ★★★ ---

        total_records_processed = 0
        logging.info(
            f"Processing {len(self.partitions)} partitions to generate and write meta-labels..."
        )

        for partition_date in tqdm(self.partitions, desc="Generating Meta-Labels"):
            partition_path_glob = str(
                self.config.weighted_dataset_path
                / f"year={partition_date.year}/month={partition_date.month}/day={partition_date.day}/*.parquet"
            )

            try:
                df_chunk = pl.read_parquet(partition_path_glob)
            except Exception:
                continue

            if df_chunk.is_empty():
                continue

            # --- (動的サンプリングとメタラベル生成ロジック) ---
            try:
                # 1. その日のM1 OOF予測データを取得 (m1_oof_df は既にUTCタイムゾーン持ち)
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

                # 3. 元のデータチャンク (df_chunk, UTCタイムゾーン持ち) を上位N件 (UTCタイムゾーン持ち) でフィルタリング
                sampled_chunk_lf = df_chunk.lazy().join(
                    top_n_timestamps.lazy(), on="timestamp", how="inner"
                )

                # 4. フィルタリングされたデータに、対応するM1予測確率を結合
                merged_chunk_lf = sampled_chunk_lf.join(
                    daily_m1_oof.lazy().select(["timestamp", "m1_pred_proba"]),
                    on="timestamp",
                    how="inner",
                )

                # --- ★★★ [修正] 市場文脈(S7_CONTEXT_FEATURES)を結合 ★★★ ---
                # 5. [新規]
                #    merged_chunk_lf の timestamp (us) から date を作成し、
                #    context_df (daily) に join_asof する
                context_joined_lf = merged_chunk_lf.with_columns(
                    pl.col("timestamp").dt.date().alias("date")
                ).join_asof(context_df.lazy(), on="date")
                # --- ★★★ [修正] ここまで ★★★ ---

                # 6. メタラベルを生成 (元 5)
                final_chunk_lf = (
                    context_joined_lf.with_columns(  # ★ [修正] context_joined_lf を使用
                        pl.when(pl.col("label") == 1)
                        .then(1)  # Yes -> meta_label = 1 (True Positive)
                        .otherwise(0)  # No -> meta_label = 0 (False Positive)
                        .alias("meta_label")
                    )
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
                result_chunk.write_parquet(
                    output_partition_dir / "data.parquet", compression="zstd"
                )
                total_records_processed += len(result_chunk)

        logging.info("\n" + "=" * 60)
        if total_records_processed > 0:
            logging.info("### Script 2/3 FINISHED! You can now run Script C. ###")
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
    parser = argparse.ArgumentParser(description="Script 2/3: Meta-Label Generation")
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
