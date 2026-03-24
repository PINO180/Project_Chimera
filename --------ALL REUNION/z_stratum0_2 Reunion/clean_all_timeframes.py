import os
import logging
from pathlib import Path
import shutil
import dask
import dask.dataframe as dd
import pandas as pd
import time

# ロギング設定
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- 設定 ---
SOURCE_PATH = "/workspaces/project_forge/data/1_XAUUSD_base_data/XAUUSD_tick_master.parquet"
DESTINATION_PATH = "/workspaces/project_forge/data/1_XAUUSD_base_data/XAUUSD_tick_master_cleaned.parquet"
ROWS_TO_DROP = 5

def main():
    start_time = time.time()

    if os.path.exists(DESTINATION_PATH):
        logger.warning(f"出力先ディレクトリが既に存在するため、一度削除します: {DESTINATION_PATH}")
        shutil.rmtree(DESTINATION_PATH, ignore_errors=True)

    ddf = dd.read_parquet(SOURCE_PATH)
    
    # Daskのインデックスをリセットし、0からの連番にする
    # これにより、パーティションの物理的な順序に関わらず、論理的な先頭行を特定できる
    ddf = ddf.reset_index(drop=True)

    # .locを使い、論理的なインデックスが5以上の行（つまり6行目以降）を全て選択
    logger.info(f"全パーティションを通じて、データ全体の先頭 {ROWS_TO_DROP} 行を削除します...")
    cleaned_ddf = ddf.loc[ROWS_TO_DROP:]
    
    logger.info("--- 最終的な保存処理を実行します ---")
    
    # Hiveパーティション構造を維持したまま、新しいディレクトリに保存
    cleaned_ddf.to_parquet(
        DESTINATION_PATH,
        partition_on=['timeframe'],
        write_index=False,
        write_metadata_file=True
    )

    total_duration = time.time() - start_time
    logger.info("="*50)
    logger.info(f"✅ 全ての処理が完了しました。総処理時間: {total_duration:.1f} 秒")
    logger.info(f"クリーンアップされたデータが以下に出力されました:\n{DESTINATION_PATH}")
    logger.info("="*50)

if __name__ == "__main__":
    dask.config.set({'scheduler': 'single-threaded'})
    main()