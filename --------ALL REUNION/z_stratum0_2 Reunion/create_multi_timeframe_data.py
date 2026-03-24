# create_multi_timeframe_data_gpu_final_loop.py (分割統治アプローチ)

import dask_cudf
import cudf
import pandas as pd
from pathlib import Path
import time
import logging
from dask.distributed import Client
from dask_cuda import LocalCUDACluster
import shutil

# ロギング設定
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- 設定項目 ---
class Config:
    PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
    
    INPUT_FILE = PROJECT_ROOT / "data/1_XAUUSD_base_data/XAUUSDtick_exness.parquet"
    # 出力先はHiveパーティションの親ディレクトリを指定
    OUTPUT_DIR = PROJECT_ROOT / "data/1_XAUUSD_base_data/XAUUSDmulti_15timeframe_bars_CORRECT_GPU.parquet"

    TIMESTAMP_COLUMN = 'datetime' 
    PRICE_COLUMN = 'mid_price' 

    TIMEFRAMES = {
        'tick': None, 'M0.5': '30s', 'M1': '1min', 'M3': '3min', 'M5': '5min', 'M8': '8min',
        'M15': '15min', 'M30': '30min', 'H1': '1H', 'H4': '4H', 'H6': '6H',
        'H12': '12H', 'D1': '1D', 'W1': '7D', 'MN': '30D',
    }
    
def main():
    config = Config()
    start_time = time.time()
    
    # 出力先ディレクトリをクリーンアップ
    if config.OUTPUT_DIR.exists():
        logger.warning(f"Output directory {config.OUTPUT_DIR} exists. Removing it.")
        shutil.rmtree(config.OUTPUT_DIR)
    config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    logger.info(f"--- Starting FINAL GPU-based aggregation from: {config.INPUT_FILE} ---")
    
    ddf = dask_cudf.read_parquet(config.INPUT_FILE)
    
    ts_col = config.TIMESTAMP_COLUMN
    ddf[ts_col] = ddf[ts_col].astype('datetime64[ns]')

    if 'tick_count' not in ddf.columns:
        logger.info("'tick_count' column not found. Creating it with value 1 for each row.")
        ddf['tick_count'] = 1
        
    agg_rules = {
        config.PRICE_COLUMN: ['first', 'max', 'min', 'last'],
        'tick_count': ['sum']
    }
    
    # タイムスタンプを数値に変換（一度だけ実行）
    ddf_int_ts = ddf[ts_col].astype('int64')

    # --- 時間足ごとにループ処理 ---
    for name, freq in config.TIMEFRAMES.items():
        loop_start_time = time.time()
        logger.info(f"--- Processing timeframe: {name} ---")

        if name == 'tick':
            # tickデータはリネームと列選択のみ
            resampled_ddf = ddf[[ts_col, config.PRICE_COLUMN, 'tick_count']].copy()
            resampled_ddf = resampled_ddf.rename(columns={ts_col: 'timestamp', config.PRICE_COLUMN: 'open'})
            resampled_ddf['high'] = resampled_ddf['open']
            resampled_ddf['low'] = resampled_ddf['open']
            resampled_ddf['close'] = resampled_ddf['open']
            resampled_ddf['volume'] = resampled_ddf['tick_count']
            
        else:
            # 数値計算によるグルーピング
            bucket_size_ns = pd.to_timedelta(freq).total_seconds() * 1_000_000_000
            grouper = (ddf_int_ts // bucket_size_ns) * bucket_size_ns
            
            # Daskの曖昧さをなくすため、キーを列として追加
            ddf['group_key'] = grouper
            
            resampled = ddf.groupby('group_key').agg(agg_rules)
            resampled.columns = ['open', 'high', 'low', 'close', 'volume']
            
            resampled = resampled.reset_index().rename(columns={'group_key': 'timestamp'})
            resampled['timestamp'] = resampled['timestamp'].astype('datetime64[ns]')
            resampled_ddf = resampled.drop(columns=['group_key'], errors='ignore')

        # Hiveパーティション形式で直接保存
        # to_parquetが計算をトリガーする
        output_path = config.OUTPUT_DIR / f"timeframe={name}"
        logger.info(f"Saving to {output_path}...")
        resampled_ddf.to_parquet(output_path, write_index=False)
        
        logger.info(f"--- Timeframe {name} completed in {time.time() - loop_start_time:.2f}s ---")

    duration = time.time() - start_time
    logger.info("--- All timeframes aggregated successfully! ---")
    logger.info(f"Total time: {duration / 60:.2f} minutes")

if __name__ == "__main__":
    with LocalCUDACluster(
        n_workers=1, 
        device_memory_limit='10GB',
        rmm_pool_size='8GB'
    ) as cluster, Client(cluster) as client:
        logger.info("Dask-CUDA Client initialized for GPU processing.")
        logger.info(f"Dashboard link: {client.dashboard_link}")
        main()