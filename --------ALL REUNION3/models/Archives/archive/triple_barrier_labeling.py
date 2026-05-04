# /workspace/models/triple_barrier_labeling.py
import sys
from pathlib import Path
import warnings
import argparse
import shutil
from dataclasses import dataclass
import logging
from typing import List

# プロジェクトのルートディレクトリをPythonの検索パスに追加
sys.path.append(str(Path(__file__).resolve().parents[1]))

import polars as pl

# blueprintから一元管理された設定を読み込む
from blueprint import S5_NEUTRALIZED_ALPHA_SET, S2_FEATURES_AFTER_AV, S6_LABELED_DATASET

# --- ロギング設定 ---
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


# --- 設定クラス ---
@dataclass
class TripleBarrierConfig:
    input_dir: Path = S5_NEUTRALIZED_ALPHA_SET
    price_data_source: Path = (
        S2_FEATURES_AFTER_AV
        / "feature_value_a_vast_universeC"
        / "features_e1c_M5.parquet"
    )
    output_dir: Path = S6_LABELED_DATASET
    lookahead_periods: int = 60
    profit_take_multiplier: float = 2.0
    stop_loss_multiplier: float = 1.0
    atr_col_name: str = "e1c_atr_21"
    test_limit: int = 0
    resume: bool = True


# --- 実行エンジン ---
class PolarsLabelingEngine:
    def __init__(self, config: TripleBarrierConfig):
        self.config = config
        warnings.filterwarnings("ignore", category=UserWarning, module="polars")
        self._validate_paths()

    def _validate_paths(self):
        if not self.config.input_dir.exists():
            raise FileNotFoundError(
                f"Input directory not found: {self.config.input_dir}"
            )
        if not self.config.price_data_source.exists():
            raise FileNotFoundError(
                f"Price data source not found: {self.config.price_data_source}"
            )

    def run(self):
        logging.info(
            "### Chapter 3, Script 1: Triple Barrier Labeling (Polars Edition v3.2.6 - Final Path Fix) ###"
        )

        if not self.config.resume and self.config.output_dir.exists():
            logging.warning(
                f"Output directory {self.config.output_dir} exists and not resuming. Removing it."
            )
            shutil.rmtree(self.config.output_dir)
        self.config.output_dir.mkdir(parents=True, exist_ok=True)

        try:
            logging.info(
                f"Step 1: Lazily scanning all feature files from {self.config.input_dir}..."
            )
            all_feature_files = [
                str(p) for p in self.config.input_dir.rglob("*.parquet")
            ]
            if not all_feature_files:
                raise ValueError("No feature files found.")

            # --- ここからが核心的な修正箇所 ---
            logging.info(
                "Step 1.5: Renaming columns with timeframe suffixes before concatenation..."
            )
            modified_lazy_frames = []
            for f_path in all_feature_files:
                path_obj = Path(f_path)
                timeframe_suffix = ""

                # パスを構成する全部分（ディレクトリ名、ファイル名）を探索
                for part in path_obj.parts:
                    if part.startswith("features_e1") and "_neutralized" in part:
                        # 例: 'features_e1a_tick_neutralized' or 'features_e1a_D1_neutralized.parquet'

                        # .parquet拡張子がある場合は除去
                        clean_part = part.replace(".parquet", "")

                        split_parts = clean_part.split("_")
                        if len(split_parts) >= 4:
                            # 'features', 'e1a', 'tick', 'neutralized' -> 'tick'
                            timeframe = split_parts[2]
                            timeframe_suffix = f"_{timeframe}"
                            break  # timeframeが見つかったのでループを抜ける

                if not timeframe_suffix:
                    logging.warning(
                        f"Could not extract timeframe from path {f_path}. Skipping suffix."
                    )

                lf = pl.scan_parquet(f_path)

                # 'timestamp' 以外の全ての列を取得
                # [FIX] スキーマ解決を避けるため、collect().columns を使用
                try:
                    feature_cols = [
                        col for col in lf.collect_schema().names() if col != "timestamp"
                    ]
                except Exception:
                    # スキーマ解決が困難な場合でも処理を継続
                    logging.warning(
                        f"Could not reliably determine schema for {f_path}. Proceeding with caution."
                    )
                    # 暫定的に空リストとして扱うことでエラーを回避
                    feature_cols = []

                # 新しい列名を作成するための式(Expression)を生成
                rename_exprs = [
                    pl.col(col).alias(f"{col}{timeframe_suffix}")
                    for col in feature_cols
                ]

                # 列名をリネームしてリストに追加
                if rename_exprs:
                    # selectで明示的に列を選択し直すことで堅牢性を高める
                    renamed_lf = lf.select(
                        "timestamp",
                        *[
                            pl.col(c).alias(f"{c}{timeframe_suffix}")
                            for c in feature_cols
                        ],
                    )
                    modified_lazy_frames.append(renamed_lf)
                else:
                    modified_lazy_frames.append(lf)

            # タイムフレームサフィックスが付与されたLazyFrameを結合
            original_lf = pl.concat(modified_lazy_frames, how="diagonal").sort(
                "timestamp"
            )
            # --- 修正箇所ここまで ---

            logging.info(
                f"   -> Successfully created a lazy plan for {len(all_feature_files)} files with unique column names."
            )

            logging.info(
                f"Step 2: Loading price data from {self.config.price_data_source}..."
            )
            price_df = (
                pl.read_parquet(self.config.price_data_source)
                .select(["timestamp", "high", "low", "close", self.config.atr_col_name])
                .sort("timestamp")
            )

            logging.info(
                "Step 3: Discovering daily partitions via lightweight reconnaissance..."
            )
            recon_plan_lf = pl.concat(
                [pl.scan_parquet(f).select("timestamp") for f in all_feature_files],
                how="diagonal",
            )

            partitions_df = (
                recon_plan_lf.select(pl.col("timestamp").dt.date().alias("date"))
                .unique()
                .collect()
                .sort("date")
            )

            if self.config.test_limit > 0:
                logging.warning(
                    f"--- TEST MODE ENABLED: Processing only the first {self.config.test_limit} partitions. ---"
                )
                partitions_df = partitions_df.head(self.config.test_limit)

            logging.info(
                f"   -> Reconnaissance complete. Found {len(partitions_df)} daily partitions to process."
            )

            logging.info(
                "Step 4: Starting daily processing and direct-to-disk writing loop..."
            )
            for i, row in enumerate(partitions_df.iter_rows(named=True)):
                current_date = row["date"]
                year, month, day = (
                    current_date.year,
                    current_date.month,
                    current_date.day,
                )

                output_partition_dir = (
                    self.config.output_dir / f"year={year}/month={month}/day={day}"
                )
                if self.config.resume and output_partition_dir.exists():
                    logging.info(
                        f"  [{i + 1}/{len(partitions_df)}] SKIPPING date: {current_date} (already processed)."
                    )
                    continue

                logging.info(
                    f"  [{i + 1}/{len(partitions_df)}] Processing date: {current_date}..."
                )

                daily_bets_lf = original_lf.filter(
                    pl.col("timestamp").dt.date() == current_date
                )
                daily_labeled_df = self._calculate_labels_for_batch(
                    daily_bets_lf, price_df
                )

                if not daily_labeled_df.is_empty():
                    output_partition_dir.mkdir(parents=True, exist_ok=True)
                    daily_labeled_df.write_parquet(
                        output_partition_dir / "data.parquet"
                    )

            logging.info("\n" + "=" * 60)
            logging.info("### Triple Barrier Labeling COMPLETED! ###")
            logging.info("The 'Answer Key' for our AI is now ready in Stratum 6.")
            logging.info("=" * 60)

        except Exception as e:
            logging.error(
                f"An error occurred during the labeling process: {e}", exc_info=True
            )
            raise

    def _calculate_labels_for_batch(
        self, bets_lf: pl.LazyFrame, price_df: pl.DataFrame
    ) -> pl.DataFrame:
        cfg = self.config

        bets_with_price_lf = bets_lf.join_asof(price_df.lazy(), on="timestamp").filter(
            pl.col(cfg.atr_col_name).is_not_null()
        )

        bets_df = bets_with_price_lf.select(
            pl.col("timestamp").alias("t0"),
            (
                pl.col("close") + pl.col(cfg.atr_col_name) * cfg.profit_take_multiplier
            ).alias("pt_barrier"),
            (
                pl.col("close") - pl.col(cfg.atr_col_name) * cfg.stop_loss_multiplier
            ).alias("sl_barrier"),
            (
                pl.col("timestamp") + pl.duration(minutes=cfg.lookahead_periods * 5)
            ).alias("t1_max"),
            pl.all().exclude(["timestamp", "close", cfg.atr_col_name, "high", "low"]),
        ).collect()

        if bets_df.is_empty():
            return pl.DataFrame()

        min_ts = bets_df["t0"].min()
        max_ts = bets_df["t1_max"].max()
        if min_ts is None or max_ts is None:
            return pl.DataFrame()
        price_window_df = price_df.filter(
            (pl.col("timestamp") >= min_ts) & (pl.col("timestamp") <= max_ts)
        )

        hits_df = (
            price_window_df.join_asof(
                bets_df.select(["t0", "pt_barrier", "sl_barrier", "t1_max"]),
                left_on="timestamp",
                right_on="t0",
            )
            .filter(pl.col("timestamp") <= pl.col("t1_max"))
            .with_columns(
                pl.when(pl.col("high") >= pl.col("pt_barrier"))
                .then(pl.col("timestamp"))
                .alias("pt_hit_time"),
                pl.when(pl.col("low") <= pl.col("sl_barrier"))
                .then(pl.col("timestamp"))
                .alias("sl_hit_time"),
            )
            .group_by("t0")
            .agg(
                pl.col("pt_hit_time").min().alias("first_pt_time"),
                pl.col("sl_hit_time").min().alias("first_sl_time"),
            )
        )

        final_df = (
            bets_df.join(hits_df, on="t0", how="left")
            .with_columns(
                pl.when(
                    (pl.col("first_pt_time").is_not_null())
                    & (
                        pl.col("first_sl_time").is_null()
                        | (pl.col("first_pt_time") <= pl.col("first_sl_time"))
                    )
                )
                .then(pl.col("first_pt_time"))
                .when(pl.col("first_sl_time").is_not_null())
                .then(pl.col("first_sl_time"))
                .otherwise(pl.col("t1_max"))
                .alias("t1"),
                pl.when(
                    (pl.col("first_pt_time").is_not_null())
                    & (
                        pl.col("first_sl_time").is_null()
                        | (pl.col("first_pt_time") <= pl.col("first_sl_time"))
                    )
                )
                .then(pl.lit(1, dtype=pl.Int8))
                .when(pl.col("first_sl_time").is_not_null())
                .then(pl.lit(-1, dtype=pl.Int8))
                .otherwise(pl.lit(0, dtype=pl.Int8))
                .alias("label"),
            )
            .with_columns(
                pl.col("t0").dt.year().alias("year"),
                pl.col("t0").dt.month().alias("month"),
                pl.col("t0").dt.day().alias("day"),
            )
            .rename({"t0": "timestamp"})
        )

        return final_df.drop(
            ["pt_barrier", "sl_barrier", "t1_max", "first_pt_time", "first_sl_time"]
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Triple Barrier Labeling with Polars")
    parser.add_argument(
        "--test-limit",
        type=int,
        default=0,
        help="Limit processing to the first N partitions for testing.",
    )
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="Disable resume feature and start from scratch.",
    )
    args = parser.parse_args()

    config = TripleBarrierConfig(test_limit=args.test_limit, resume=not args.no_resume)
    engine = PolarsLabelingEngine(config)
    engine.run()
