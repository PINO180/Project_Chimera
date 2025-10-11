# /workspace/validation/build_master_table.py
"""
build_master_table.py v1.6 - レジューム機能・ソート修正版

v1.5からの改善点：
- レジューム機能の追加：マスターテーブルの土台が既に存在する場合、
  3時間かかる初期化プロセスをスキップし、マージ処理から再開する。
- join_asofのエラーを修正：結合前に両方のLazyFrameを'timestamp'で
  明示的にソートする処理を追加し、「not sorted」エラーを回避する。
"""
import sys
from pathlib import Path
import shutil
import logging
import time
from typing import List, Tuple

sys.path.append(str(Path(__file__).resolve().parents[1]))

import polars as pl
from tqdm import tqdm
import blueprint as config

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - [%(levelname)s] - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)


class MasterTableBuilder:
    input_dir: Path
    output_dir: Path
    temp_dir: Path
    base_columns: List[str]

    def __init__(self) -> None:
        self.input_dir = Path(config.S2_FEATURES_AFTER_AV)
        self.output_dir = Path(config.S4_MASTER_TABLE_PARTITIONED)
        self.temp_dir = self.output_dir.parent / f"_temp_{self.output_dir.name}"
        self.base_columns = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
        
        logger.info("="*50)
        logger.info("🚀 MasterTableBuilder initialized (v1.6)")
        logger.info(f"Input Directory: {self.input_dir}")
        logger.info(f"Output Directory: {self.output_dir}")
        logger.info("="*50)

    def _get_feature_sources(self) -> Tuple[Path, List[Path]]:
        logger.info("🔍 Scanning for feature sources correctly...")
        all_tick_dirs = [p for p in self.input_dir.rglob('features_*_tick') if p.is_dir()]
        all_parquet_files_found = list(self.input_dir.rglob('*.parquet'))
        single_parquet_files = []
        for pq_file in all_parquet_files_found:
            is_in_tick_dir = any(tick_dir in pq_file.parents for tick_dir in all_tick_dirs)
            if not is_in_tick_dir:
                single_parquet_files.append(pq_file)

        all_sources = single_parquet_files + all_tick_dirs
        if not all_sources:
            raise FileNotFoundError(f"No feature sources found in {self.input_dir}")

        base_source = next((s for s in all_sources if 'e1a_tick' in s.name and s.is_dir()), None)
        if not base_source:
            base_source = next((s for s in all_sources if s.is_dir() and 'tick' in s.name), None)
        if not base_source:
            raise FileNotFoundError("Base tick data source could not be found.")

        merge_sources = [s for s in all_sources if s.resolve() != base_source.resolve()]
        logger.info(f"🛡️  Base source identified: {base_source.name}")
        logger.info(f"➕ Found {len(merge_sources)} sources to merge. (Corrected)")
        return base_source, merge_sources

    def _write_partitioned(self, lf: pl.LazyFrame, target_dir: Path) -> None:
        if 'year' not in lf.columns:
             lf = lf.with_columns(
                pl.col("timestamp").dt.year().alias("year").cast(pl.Int32),
                pl.col("timestamp").dt.month().alias("month").cast(pl.Int32),
                pl.col("timestamp").dt.day().alias("day").cast(pl.Int32)
            )

        dates_df = lf.select(["year", "month", "day"]).unique().collect()
        logger.info(f"  -> Writing {len(dates_df)} daily partitions...")

        for row in tqdm(dates_df.iter_rows(named=True), total=len(dates_df), desc=f"Writing to {target_dir.name}"):
            y, m, d = row['year'], row['month'], row['day']
            partition_lf = lf.filter((pl.col('year') == y) & (pl.col('month') == m) & (pl.col('day') == d))
            output_path = target_dir / f"year={y}/month={m}/day={d}"
            output_path.mkdir(parents=True, exist_ok=True)
            partition_lf.sink_parquet(str(output_path / "0.parquet"), compression='snappy')

    def _initialize_master_table(self, base_source: Path) -> None:
        logger.info("🏗️  Initializing master table skeleton...")
        if self.output_dir.exists():
            shutil.rmtree(self.output_dir)
        self.output_dir.mkdir(parents=True)

        try:
            lf = pl.scan_parquet(str(base_source))
            skeleton_lf = lf.select(self.base_columns)
            self._write_partitioned(skeleton_lf, self.output_dir)
            logger.info("✅ Master table skeleton created successfully.")
        except Exception as e:
            logger.error(f"❌ Failed to initialize master table: {e}", exc_info=True)
            if self.output_dir.exists():
                shutil.rmtree(self.output_dir)
            raise

    def run(self) -> None:
        start_time = time.time()
        try:
            base_source, merge_sources = self._get_feature_sources()
            
            # --- NEW v1.6: レジューム機能 ---
            if self.output_dir.exists():
                logger.info("✅ Skeleton directory already exists. Skipping initialization and resuming merge process.")
            else:
                self._initialize_master_table(base_source)
            # --- レジューム機能ここまで ---
            
            logger.info("\n" + "="*50)
            logger.info("🔄 Starting iterative merge process...")
            logger.info("="*50)

            # メインの結合ループ
            for i, feature_source in enumerate(tqdm(merge_sources, desc="Merging Features")):
                logger.info(f"\n[{i+1}/{len(merge_sources)}] Merging source: {feature_source.name}")
                
                if self.temp_dir.exists():
                    shutil.rmtree(self.temp_dir)
                self.temp_dir.mkdir(parents=True)

                lf_master = pl.scan_parquet(str(self.output_dir))
                lf_feature = pl.scan_parquet(str(feature_source))
                
                # --- SORTING FIX v1.6: join_asofの前に両方のDFをソート ---
                logger.info("  -> Sorting dataframes by 'timestamp'...")
                lf_master_sorted = lf_master.sort("timestamp")
                lf_feature_sorted = lf_feature.sort("timestamp")
                # --- FIXここまで ---
                
                new_columns_to_add = [col for col in lf_feature_sorted.columns if col not in lf_master_sorted.columns]
                if not new_columns_to_add:
                    logger.warning(f"  -> No new columns to add from {feature_source.name}. Skipping.")
                    continue
                
                lf_feature_filtered = lf_feature_sorted.select(['timestamp'] + new_columns_to_add)
                
                # ソート済みのLazyFrameを使用して結合
                lf_joined = lf_master_sorted.join_asof(lf_feature_filtered, on="timestamp")
                
                self._write_partitioned(lf_joined, self.temp_dir)

                logger.info("  -> Atomically swapping master table...")
                shutil.rmtree(self.output_dir)
                self.temp_dir.rename(self.output_dir)
                logger.info(f"✅ Successfully merged {feature_source.name}")

            total_time = time.time() - start_time
            logger.info("\n" + "="*50)
            logger.info(f"🎉 All features merged successfully! Total time: {total_time:.2f} seconds")
            logger.info(f"Final master table located at: {self.output_dir}")
            logger.info("="*50)

        except Exception as e:
            logger.critical(f"❌ A critical error occurred: {e}", exc_info=True)
        finally:
            if self.temp_dir.exists():
                shutil.rmtree(self.temp_dir)
            logger.info("🧹 Cleanup complete.")


if __name__ == '__main__':
    builder = MasterTableBuilder()
    builder.run()