# create_standardized_dataset_float64.py
import os
import logging
from pathlib import Path
import shutil
import dask
import dask.dataframe as dd
import time
import gc

# --- 基本設定 ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Dask設定: メモリ不足を防ぐため、安全なシングルスレッドモードで実行
dask.config.set({'scheduler': 'single-threaded'})

# --- 設定 ---
class Config:
    PROJECT_ROOT = Path("/workspaces/project_forge")
    
    # 入力元: price_directionをパッチした、スキーマが不統一のデータセット
    SOURCE_PATH = PROJECT_ROOT / "data/1_XAUUSD_base_data/XAUUSD_tick_master_FINAL_PATCHED"
    
    # 出力先: 全てのスキーマが統一された、最終的なクリーンデータセット
    DESTINATION_PATH = PROJECT_ROOT / "data/1_XAUUSD_base_data/XAUUSD_tick_master_GOLDEN"
    
    # ★★★ データセット全体の「あるべき姿」をここで厳密に定義 ★★★
    # 全ての価格関連データをfloat64に統一し、精度を最大化
    GOLDEN_SCHEMA = {
        'timestamp': 'datetime64[ns]',
        'open': 'float64',
        'high': 'float64',
        'low': 'float64',
        'close': 'float64',
        'volume': 'int64',
        'log_return': 'float64',
        'rolling_volatility': 'float64',
        'rolling_avg_volume': 'float64',
        'atr': 'float64',
        'price_direction': 'int8',
        'price_momentum': 'float64',
        'volume_ratio': 'float64',
        'timeframe': 'string' # dictionary型やcategory型を単純なstringに統一
    }

# --- メイン実行パイプライン ---
def main():
    config = Config()
    start_time = time.time()
    logger.info("=== 【float64統一版】データセット標準化処理を開始します ===")

    if not config.SOURCE_PATH.is_dir():
        logger.error(f"入力ディレクトリが見つかりません: {config.SOURCE_PATH}")
        return

    if config.DESTINATION_PATH.exists():
        logger.warning(f"出力先が既に存在するため、一度削除します: {config.DESTINATION_PATH}")
        shutil.rmtree(config.DESTINATION_PATH, ignore_errors=True)

    logger.info("Daskでソースデータセットを読み込んでいます...")
    ddf = dd.read_parquet(config.SOURCE_PATH)
    
    logger.info(f"GOLDEN_SCHEMAに基づいて、全パーティションのデータ型を統一します...")
    # .astype()メソッドで、データセット全体のデータ型をGOLDEN_SCHEMAに強制的に変換
    standardized_ddf = ddf.astype(config.GOLDEN_SCHEMA)
    
    logger.info("スキーマ統一済みの新しいデータセットをHive形式で書き出します...")
    # 統一されたデータフレームを書き出すことで、データ型の不整合は発生しない
    # メタデータも自動で正しく生成される
    standardized_ddf.to_parquet(
        config.DESTINATION_PATH,
        partition_on=['timeframe'],
        write_index=False,
        engine='pyarrow'
    )
    
    total_duration = time.time() - start_time
    logger.info(f"✅ データセットの標準化が正常に完了しました。総処理時間: {total_duration / 60:.1f} 分")
    logger.info(f"分析に使用できるクリーンなデータが以下に出力されました:\n{config.DESTINATION_PATH}")

if __name__ == "__main__":
    main()