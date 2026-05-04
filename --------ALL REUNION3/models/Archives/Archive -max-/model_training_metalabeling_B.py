# models/model_training_metalabeling_B.py
# [B案改: エース投入型M2 - Top N特徴量を利用]
# [V5対応: 双方向ラベリング (Long/Short独立生成) + 外部文脈特徴量(S7_CONTEXT_FEATURES)の結合]

import sys
from pathlib import Path
import logging
import argparse
import datetime
import warnings
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
    S7_M1_OOF_PREDICTIONS_LONG,
    S7_M1_OOF_PREDICTIONS_SHORT,
    S7_META_LABELED_OOF_LONG,
    S7_META_LABELED_OOF_SHORT,
    # S3_FEATURES_FOR_TRAINING,  # ← 削除
    S3_FEATURES_FOR_TRAINING_V5,  # ★ 追加
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
    weighted_dataset_path: Path = S6_WEIGHTED_DATASET
    m1_long_oof_path: Path = S7_M1_OOF_PREDICTIONS_LONG
    m1_short_oof_path: Path = S7_M1_OOF_PREDICTIONS_SHORT
    output_long_dir: Path = S7_META_LABELED_OOF_LONG
    output_short_dir: Path = S7_META_LABELED_OOF_SHORT
    # feature_list_path: Path = S3_FEATURES_FOR_TRAINING  # ← 削除
    feature_list_path: Path = S3_FEATURES_FOR_TRAINING_V5  # ★ 追加
    test: bool = False


class MetaLabelGenerator:
    def __init__(self, config: MetaLabelingConfig):
        self.config = config
        self.partitions = self._discover_partitions()
        self.features = self._load_features()

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

    def _load_features(self) -> List[str]:
        logging.info(f"Loading feature list from {self.config.feature_list_path}...")
        if not self.config.feature_list_path.exists():
            raise FileNotFoundError(
                f"Feature list file not found at: {self.config.feature_list_path}"
            )

        with open(self.config.feature_list_path, "r") as f:
            raw_features = [line.strip() for line in f if line.strip()]

        # V5ラベリングエンジンが生成する全メタデータ・未来情報の完全除外
        exclude_exact = {
            "timestamp",
            # "timeframe",  # ★削除: 特徴量として次へ渡すため除外しない
            "t1",
            "label",
            "label_long",
            "label_short",
            "uniqueness",
            "uniqueness_long",
            "uniqueness_short",
            "payoff_ratio",
            "payoff_ratio_long",
            "payoff_ratio_short",
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
            "meta_label",
            "m1_pred_proba",
        }

        features = ["timeframe"]  # ★変更: 特徴量リストの先頭に明示的にセット
        for col in raw_features:
            if col in exclude_exact or col == "timeframe":  # ★変更: 重複防止
                continue
            if col.startswith("is_trigger_on"):
                continue
            features.append(col)

        logging.info(f"  -> Loaded {len(features)} valid features.")
        return features

    # 修正後
    def run(self) -> None:
        logging.info(
            "### Script 2/3: Meta-Label Generation (V5 Bidirectional / Context Plus) ###"
        )
        logging.info("Using dynamic sampling: Top 50% of M1 predictions per day.")

        # 双方向（Long/Short）でループ処理
        for direction in ["long", "short"]:
            logging.info(
                f"\n{'=' * 60}\n=== Starting Meta-Label Generation for {direction.upper()} ===\n{'=' * 60}"
            )

            # 方向ごとの設定切り替え
            oof_path = (
                self.config.m1_long_oof_path
                if direction == "long"
                else self.config.m1_short_oof_path
            )
            output_dir = (
                self.config.output_long_dir
                if direction == "long"
                else self.config.output_short_dir
            )
            label_col = f"label_{direction}"
            uniqueness_col = f"uniqueness_{direction}"

            if not oof_path.exists():
                logging.error(
                    f"{direction.upper()} OOF prediction file not found at: {oof_path}. Skipping."
                )
                continue

            if output_dir.exists():
                logging.warning(
                    f"Output directory {output_dir} exists. Removing it for a clean run."
                )
                shutil.rmtree(output_dir)
            output_dir.mkdir(parents=True)

            # --- OOF予測のロード ---
            logging.info(
                f"Loading {direction.upper()} M1 OOF predictions from {oof_path}..."
            )
            try:
                m1_oof_df = pl.read_parquet(oof_path)
                # --- 修正前 ---
                # m1_oof_df = m1_oof_df.with_columns(
                #     pl.col("timestamp").dt.replace_time_zone("UTC")
                # )

                # --- 修正後 ---
                m1_oof_df = m1_oof_df.with_columns(
                    pl.col("timestamp").cast(pl.Datetime("us", "UTC"))
                )
                if "prediction" in m1_oof_df.columns:
                    m1_oof_df = m1_oof_df.rename({"prediction": "m1_pred_proba"})
                m1_oof_df = m1_oof_df.with_columns(
                    pl.col("timestamp").dt.date().alias("date")
                )
            except Exception as e:
                logging.error(
                    f"Failed to load or process {direction.upper()} M1 OOF predictions: {e}",
                    exc_info=True,
                )
                continue

            # --- 特徴量カラムの選定 ---
            columns_to_select = [
                "timestamp",
                # "timeframe",  # ★削除: self.features の中に既に含まれているため外す
                "atr_value",
                label_col,
                uniqueness_col,
            ] + self.features

            total_records_processed = 0

            for partition_date in tqdm(
                self.partitions, desc=f"Generating {direction.capitalize()} Meta-Labels"
            ):
                s6_partition_path = (
                    self.config.weighted_dataset_path
                    / f"year={partition_date.year}/month={partition_date.month}/day={partition_date.day}/data.parquet"
                )

                try:
                    if not s6_partition_path.exists():
                        continue
                    df_chunk = pl.read_parquet(s6_partition_path)

                    # ▼▼▼ 修正: 以下の Int32 へのキャスト処理を完全に削除する ▼▼▼
                    # if "timeframe" in df_chunk.columns:
                    #     if df_chunk.schema["timeframe"] != pl.Int32:
                    #         df_chunk = df_chunk.with_columns(
                    #             pl.col("timeframe")
                    #             .replace({"M1": 0, "M3": 1, "M5": 2, "M8": 3, "M15": 4})
                    #             .cast(pl.Int32)
                    #         )
                    # ▲▲▲ 削除ここまで ▲▲▲

                    # unique処理だけ残す
                    if "timeframe" in df_chunk.columns:
                        df_chunk = df_chunk.unique(
                            subset=["timestamp", "timeframe"],
                            keep="last",
                            maintain_order=True,
                        )
                except Exception:
                    continue

                # --- サンプリング・結合ロジック ---
                try:
                    # 1. その日のM1 OOF予測データを取得
                    daily_m1_oof = m1_oof_df.filter(pl.col("date") == partition_date)
                    if daily_m1_oof.is_empty():
                        continue

                    # --- 修正前 ---
                    # その日の上位50%を抽出する処理は機械学習トレードにおいて非常に危険な未来情報リーク（分布のシフト）を引き起こします
                    # 2. その日の予測確率上位50%のタイムスタンプを取得
                    # n_samples = max(1, len(daily_m1_oof) // 2)
                    # top_n_keys = (
                    #     daily_m1_oof.sort("m1_pred_proba", descending=True)
                    #     .head(n_samples)
                    #     .select(["timestamp", "timeframe"])
                    #     .unique()
                    # )

                    # --- 修正後 ---
                    # 2. 一定の確率（例：0.5以上）を超えた「自信のあるシグナル」だけをメタモデルに渡す
                    THRESHOLD = (
                        0.50  # ※ベースモデルの強さに応じて 0.55 等に調整してください
                    )
                    top_n_keys = (
                        daily_m1_oof.filter(pl.col("m1_pred_proba") >= THRESHOLD)
                        .select(["timestamp", "timeframe"])
                        .unique()
                    )
                    if top_n_keys.is_empty():
                        continue

                    # 3. S6データを上位N件でフィルタリング
                    sampled_chunk_lf = (
                        df_chunk.lazy()
                        .join(
                            top_n_keys.lazy(),
                            on=["timestamp", "timeframe"],
                            how="inner",
                        )  # ★timeframeを追加
                        .select(
                            [
                                col
                                for col in columns_to_select
                                if col in df_chunk.columns
                            ]
                        )
                    )

                    # 5. M1予測確率 (m1_pred_proba) を結合
                    merged_chunk_lf = sampled_chunk_lf.join(
                        daily_m1_oof.lazy().select(
                            [
                                "timestamp",
                                "timeframe",
                                "m1_pred_proba",
                            ]  # ★timeframeを追加
                        ),
                        on=["timestamp", "timeframe"],  # ★timeframeを追加
                        how="inner",
                    )

                    #  結合処理（.join）のコメントに # timeframeを削除 と書いてありますが、
                    #  これは「スクリプトAの出力（右側のテーブル）には timeframe が存在しないため、timestamp だけで結合する」という意味でコード自体は正しく書かれています。
                    #  左側のテーブル（S6の元データ）からしっかりと timeframe を引っ張ってきているので、結合後の完成データにはちゃんと timeframe が刻み込まれます。

                    merged_chunk_df = merged_chunk_lf.collect()
                    if merged_chunk_df.is_empty():
                        continue

                    # 6. メタラベルを生成 (label_long / label_short と同一値)
                    final_chunk_df = merged_chunk_df.with_columns(
                        pl.col(label_col).alias("meta_label"),
                        pl.col(uniqueness_col).alias("uniqueness"),  # ← この1行を追加
                    )

                    # --- 出力 ---
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
                        f"Error processing partition {partition_date} for {direction}: {e}",
                        exc_info=True,
                    )
                    continue

            logging.info(
                f"Finished {direction.upper()} -> Total samples generated: {total_records_processed}"
            )

        logging.info("\n" + "=" * 60)
        logging.info("### Script 2/3 FINISHED! (V5 Bidirectional Context Plus) ###")
        logging.info("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Script 2/3: Meta-Label Generation (V5 Bidirectional)"
    )
    parser.add_argument(
        "--test",
        action="store_true",
        help="Run in quick test mode, processing only the first 5 partitions.",
    )

    # 修正後
    args = parser.parse_args()
    config = MetaLabelingConfig(test=args.test)

    generator = MetaLabelGenerator(config)
    generator.run()
