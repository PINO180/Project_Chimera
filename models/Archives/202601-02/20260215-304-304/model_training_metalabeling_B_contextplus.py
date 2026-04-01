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
    S3_FEATURES_FOR_TRAINING,  # <--- 追加
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
    # [修正] JSONパスを廃止し、全特徴量リストのパスに変更
    feature_list_path: Path = S3_FEATURES_FOR_TRAINING
    top_n_per_day: int = 20
    test: bool = False


class MetaLabelGenerator:
    def __init__(self, config: MetaLabelingConfig):
        self.config = config
        self.partitions = self._discover_partitions()
        # [修正] 全特徴量をロード
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

    # [修正] メソッド名とロジックを変更
    def _load_features(self) -> List[str]:
        logging.info(f"Loading feature list from {self.config.feature_list_path}...")
        if not self.config.feature_list_path.exists():
            raise FileNotFoundError(
                f"Feature list file not found at: {self.config.feature_list_path}"
            )

        with open(self.config.feature_list_path, "r") as f:
            features = [line.strip() for line in f if line.strip()]

        logging.info(f"  -> Loaded {len(features)} features.")
        return features

    # ==============================================================================
    # ★★★ [修正] M2用: 全時間足の状況を横結合するヘルパーメソッド (Horizontal Context) ★★★
    # ==============================================================================
    def _inject_horizontal_context(self, df: pl.DataFrame) -> pl.DataFrame:
        """
        現在の行（例: M15）に対して、同じ時刻の『他の時間足のM1予測値』と『ATR』を
        カラムとして横付けする。これによりM2は相場全体の状況を把握できる。
        """
        # 必要なカラムだけ切り出してピボット（回転）する
        # timestamp を軸に、timeframe を列に変換

        # 1. M1予測値 (m1_pred_proba) の横展開
        # 結果のカラム名例: ctx_m1_H1, ctx_m1_H4 ...
        m1_pivot = (
            df.select(["timestamp", "timeframe", "m1_pred_proba"])
            .pivot(
                values="m1_pred_proba",
                index="timestamp",
                columns="timeframe",
                aggregate_function="first",  # 重複排除済みなので first でOK
            )
            .fill_null(0.5)  # データがない時間足は「どっちつかず(0.5)」で埋める
        )
        # カラム名にプレフィックスをつける
        rename_map_m1 = {c: f"ctx_m1_{c}" for c in m1_pivot.columns if c != "timestamp"}
        m1_pivot = m1_pivot.rename(rename_map_m1)

        # 2. ATR (atr_value) の横展開
        # 結果のカラム名例: ctx_atr_H1, ctx_atr_H4 ...
        # ※ atr_value が存在する場合のみ実行
        if "atr_value" in df.columns:
            atr_pivot = (
                df.select(["timestamp", "timeframe", "atr_value"])
                .pivot(
                    values="atr_value",
                    index="timestamp",
                    columns="timeframe",
                    aggregate_function="first",
                )
                .fill_null(0.0)
            )
            rename_map_atr = {
                c: f"ctx_atr_{c}" for c in atr_pivot.columns if c != "timestamp"
            }
            atr_pivot = atr_pivot.rename(rename_map_atr)

            # M1ピボットとATRピボットを結合
            context_df = m1_pivot.join(atr_pivot, on="timestamp", how="left")
        else:
            context_df = m1_pivot

        # 3. 元のデータフレームに結合
        # これにより、どの時間足の行にも「全時間足のスコア」が付与される
        df_with_context = df.join(context_df, on="timestamp", how="left")

        return df_with_context

    # ==============================================================================

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

        # [修正] self.top_50_features ではなく self.features (全特徴量) を使用
        columns_to_select = [
            "timestamp",
            "timeframe",
            "atr_value",
            "label",
            "uniqueness",
            "payoff_ratio",
        ] + self.features

        for partition_date in tqdm(self.partitions, desc="Generating Meta-Labels"):
            # ★★★ 修正: ワイルドカードをやめ、data.parquet を厳密に指定 ★★★
            partition_path_glob = str(
                self.config.weighted_dataset_path
                / f"year={partition_date.year}/month={partition_date.month}/day={partition_date.day}/data.parquet"
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
                # ★★★ 修正: unique() を追加してタイムスタンプの重複を排除 ★★★
                top_n_timestamps = (
                    daily_m1_oof.sort("m1_pred_proba", descending=True)
                    .head(self.config.top_n_per_day)
                    .select("timestamp")
                    .unique()
                )
                if top_n_timestamps.is_empty():
                    continue

                # 3. 元のデータチャンク (df_chunk) を上位N件でフィルタリング
                #    必要なカラム (Top 50 + context用 + label/uniqueness) を選択
                sampled_chunk_lf = (
                    df_chunk.lazy()
                    .join(top_n_timestamps.lazy(), on="timestamp", how="inner")
                    .select(
                        [col for col in columns_to_select if col in df_chunk.columns]
                    )
                )

                # 4. M1予測確率 (m1_pred_proba) を結合
                # ★★★ 修正: 結合キーを ["timestamp", "timeframe"] に変更し、右側のカラムを厳選 ★★★
                merged_chunk_lf = sampled_chunk_lf.join(
                    daily_m1_oof.lazy().select(
                        ["timestamp", "timeframe", "m1_pred_proba"]
                    ),
                    on=["timestamp", "timeframe"],
                    how="inner",
                )

                # --- ★★★ [修正] ここでデータを実体化し、水平コンテキストを注入する ★★★ ---
                # Pivot操作を行うため、LazyFrameからDataFrameへ変換(collect)します
                merged_chunk_df = merged_chunk_lf.collect()

                if merged_chunk_df.is_empty():
                    continue

                # 水平コンテキスト注入 (ctx_m1_H4, ctx_atr_H4 等を追加)
                merged_chunk_df = self._inject_horizontal_context(merged_chunk_df)
                # -----------------------------------------------------------------------

                # 5. メタラベルを生成 (DataFrame操作)
                #    M1が正解(label=1)ならメタラベル1(Go)、不正解なら0(No-Go)
                final_chunk_df = merged_chunk_df.with_columns(
                    pl.when(pl.col("label") == 1)
                    .then(1)
                    .otherwise(0)
                    .alias("meta_label")
                )

                result_chunk = final_chunk_df

            except Exception as e:
                logging.error(
                    f"Error processing partition {partition_date}: {e}", exc_info=True
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
