# preprocess_base_data_cpu_final_stable_v2.py

import os
import logging
from pathlib import Path
import shutil
import dask
import dask.dataframe as dd
import pandas as pd
import time
from typing import List, Dict
import gc
import traceback
import numpy as np

# Daskのライブラリをインポート（Clientは不要）

# ロギング設定
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
)
logger = logging.getLogger(__name__)


# --- 設定パラメータ ---
class ProcessingConfig:
    """前処理パラメータの設定クラス"""
    
    # ファイルパス
    PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
    INPUT_FILE = PROJECT_ROOT / "data/1_XAUUSD_base_data/XAUUSDmulti_15timeframe_bars_CORRECT_GPU.parquet"
    OUTPUT_FILE = PROJECT_ROOT / "data/1_XAUUSD_base_data/XAUUSDmulti_15timeframe_bars_exness_enriched_cpu.parquet"
    
    # 計算パラメータ
    VOLATILITY_WINDOW = 20
    VOLUME_WINDOW = 50
    ATR_WINDOW = 14
    
    # 除外カラム
    COLUMNS_TO_DROP = ['volatility', 'avg_volume']
    
    # 特別扱いする巨大な時間軸
    LARGE_TIMEFRAME = 'tick'


def calculate_enhanced_features(df: pd.DataFrame, global_volume_mean: float = 0.0) -> pd.DataFrame:
    """
    データ型変換を分離し、計算に集中させたCPU処理版
    """
    if len(df) == 0:
        return df
    
    df = df.sort_values('timestamp').reset_index(drop=True)
    
    # 1. 対数リターン
    ratio = df['close'] / df['close'].shift(1)
    df['log_return'] = np.log(ratio.fillna(1.0))
    
    # 2. ローリングボラティリティ
    df['rolling_volatility'] = df['log_return'].rolling(
        window=ProcessingConfig.VOLATILITY_WINDOW, min_periods=1
    ).std()
    
    # 3. ローリング平均出来高
    fallback_mean = global_volume_mean if global_volume_mean > 0 else df['volume'].mean()
    df['rolling_avg_volume'] = df['volume'].rolling(
        window=ProcessingConfig.VOLUME_WINDOW, min_periods=1
    ).mean().fillna(fallback_mean)
    
    # 4. ATR
    if len(df) > 1:
        close_prev = df['close'].shift(1)
        high_low = df['high'] - df['low']
        high_prev_close = (df['high'] - close_prev).abs()
        low_prev_close = (df['low'] - close_prev).abs()
        true_range = pd.concat([high_low, high_prev_close, low_prev_close], axis=1).max(axis=1, skipna=True)
        df['atr'] = true_range.ewm(span=ProcessingConfig.ATR_WINDOW, adjust=False).mean()
    else:
        df['atr'] = 0.0
    
    # 5. 価格方向性
    price_diff = df['close'].diff()
    df['price_direction'] = np.sign(price_diff.fillna(0.0))
    
    # 6. 価格モメンタム
    ratio_5 = df['close'] / df['close'].shift(5)
    df['price_momentum'] = ratio_5 - 1
    
    # 7. 出来高比率
    safe_avg_vol = df['rolling_avg_volume'].replace(0, 1.0)
    df['volume_ratio'] = df['volume'] / safe_avg_vol
    
    # ▼▼▼▼▼▼【ここから修正】▼▼▼▼▼▼
    # Daskのメタデータと実際のデータ型を確実に一致させるための型変換
    
    # 'timeframe'カラムがカテゴリ型に変換されるのを防ぎ、string型に強制
    if 'timeframe' in df.columns:
        df['timeframe'] = df['timeframe'].astype('string')
    
    # 'timestamp'カラムの精度をナノ秒(ns)からミリ秒(ms)に変換
    if 'timestamp' in df.columns:
        df['timestamp'] = df['timestamp'].astype('datetime64[ms]')
    # ▲▲▲▲▲▲【修正ここまで】▲▲▲▲▲▲

    return df
        


def get_final_schema(base_columns: List[str]) -> Dict[str, str]:
    """処理後の完全なスキーマ（型情報）を返す"""
    
    schema = {
        'timeframe': 'string',
        'timestamp': 'datetime64[ms]',
        'open': 'float32',
        'high': 'float32',
        'low': 'float32',
        'close': 'float32',
        'volume': 'int64',
        'tick_count': 'int32',
        'bid': 'float32',
        'ask': 'float32',
        'spread': 'float32',
        'price_change_pct': 'float32',
        'range': 'float32',
        'body_size': 'float32',
    }
    for col in ProcessingConfig.COLUMNS_TO_DROP:
        if col in schema:
            del schema[col]
            
    schema.update({
        'log_return': 'float32',
        'rolling_volatility': 'float32',
        'rolling_avg_volume': 'float64',
        'atr': 'float32',
        'price_direction': 'int8',
        'price_momentum': 'float32',
        'volume_ratio': 'float32'
    })

    final_cols = [col for col in base_columns if col in schema]
    final_cols.extend(get_new_column_meta().keys())
    
    return {col: schema[col] for col in final_cols if col in schema}

def get_new_column_meta() -> Dict[str, str]:
    """新規追加されるカラムの型情報だけを返す"""
    return {
        'log_return': 'float32',
        'rolling_volatility': 'float32',
        'rolling_avg_volume': 'float64',
        'atr': 'float32',
        'price_direction': 'int8',
        'price_momentum': 'float32',
        'volume_ratio': 'float32'
    }

def main() -> bool:
    """メイン処理関数 - CPUでの処理に最適化"""
    config = ProcessingConfig()
    start_time = time.time()

    logger.info("=== 純粋なDaskワークフローによるCPU前処理パイプライン開始 ===")

    if not config.INPUT_FILE.exists():
        logger.error(f"Input file not found: {config.INPUT_FILE}")
        return False
    if config.OUTPUT_FILE.exists():
        logger.warning(f"Output path {config.OUTPUT_FILE} already exists. Removing it.")
        shutil.rmtree(config.OUTPUT_FILE, ignore_errors=True)
    config.OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

    try:
        ddf = dd.read_parquet(config.INPUT_FILE, blocksize="128MB")

        all_timeframes_series = ddf['timeframe'].unique().compute()
        all_timeframes = all_timeframes_series.tolist() if isinstance(all_timeframes_series, pd.Series) else [all_timeframes_series]
        
        in_memory_timeframes = [tf for tf in all_timeframes if tf != config.LARGE_TIMEFRAME]
        
        processed_ddfs: List[dd.DataFrame] = []
        final_schema = get_final_schema(ddf.columns)
        meta_df = pd.DataFrame({k: [] for k in final_schema.keys()}).astype(final_schema)

        if in_memory_timeframes:
            logger.info(f"--- Creating lazy task for {len(in_memory_timeframes)} small timeframes ---")
            in_memory_ddf = ddf[ddf['timeframe'].isin(in_memory_timeframes)]
            cols_to_drop = [col for col in config.COLUMNS_TO_DROP if col in in_memory_ddf.columns]
            if cols_to_drop:
                in_memory_ddf = in_memory_ddf.drop(columns=cols_to_drop)
            
            enhanced_in_memory_ddf = in_memory_ddf.groupby('timeframe', group_keys=False).apply(
                lambda df: calculate_enhanced_features(df, global_volume_mean=0.0), meta=meta_df
            )
            processed_ddfs.append(enhanced_in_memory_ddf)
            logger.info("Task for small timeframes created.")

        if config.LARGE_TIMEFRAME in all_timeframes:
            logger.info(f"--- Creating lazy task for '{config.LARGE_TIMEFRAME}' timeframe ---")
            
            logger.info("Calculating global volume mean for 'tick' in a memory-efficient way...")
            tick_volume_ddf = dd.read_parquet(
                config.INPUT_FILE,
                filters=[('timeframe', '==', config.LARGE_TIMEFRAME)],
                columns=['volume'],
                blocksize="128MB"
            )
            global_volume_mean = tick_volume_ddf['volume'].mean().compute()
            logger.info(f"Global volume mean for 'tick' is {global_volume_mean}")
            del tick_volume_ddf
            gc.collect()

            tick_ddf = ddf[ddf['timeframe'] == config.LARGE_TIMEFRAME]
            cols_to_drop = [col for col in config.COLUMNS_TO_DROP if col in tick_ddf.columns]
            if cols_to_drop:
                tick_ddf = tick_ddf.drop(columns=cols_to_drop)
            
            enhanced_tick_ddf = tick_ddf.map_partitions(
                calculate_enhanced_features, global_volume_mean=float(global_volume_mean), meta=meta_df
            )
            processed_ddfs.append(enhanced_tick_ddf)
            logger.info(f"Task for '{config.LARGE_TIMEFRAME}' created.")

        logger.info("--- Finalizing: Concatenating tasks and executing graph ---")
        final_start = time.time()
        
        if not processed_ddfs:
            logger.warning("No data to process. Exiting.")
            return True

        final_ddf = dd.concat(processed_ddfs, ignore_unknown_divisions=True)
        
        logger.info("Filtering out empty partitions before saving...")
        parts = final_ddf.to_delayed()
        lengths = [dask.delayed(len)(p) for p in parts]
        computed_lengths = dask.compute(*lengths)
        parts_with_data = [p for p, length in zip(parts, computed_lengths) if length > 0]

        if not parts_with_data:
            logger.warning("All partitions resulted in empty data. No output generated.")
            return True
            
        final_ddf = dd.from_delayed(parts_with_data, meta=meta_df)
        logger.info(f"Filtered down to {final_ddf.npartitions} non-empty partitions.")
        
        final_ddf.to_parquet(
            config.OUTPUT_FILE, compression='snappy', write_metadata_file=True, partition_on=['timeframe']
        )
        
        logger.info(f"Final save completed in {time.time() - final_start:.2f}s")
        total_duration = time.time() - start_time
        logger.info("=== 処理完了 ===")
        logger.info(f"総処理時間: {total_duration / 60:.1f} minutes")
        logger.info(f"Enhanced dataset (Hive Partitioned) saved to: {config.OUTPUT_FILE}")
        
        return True
        
    except Exception as e:
        logger.error(f"Processing failed: {e}")
        logger.error(traceback.format_exc())
        return False


if __name__ == "__main__":
    logger.info("Running with Dask's single-threaded scheduler for maximum stability.")
    
    dask.config.set({
        'dataframe.shuffle.method': 'disk',
        'scheduler': 'single-threaded'
    })
    
    success = main()

    if success:
        print("\nCPUでのデータ前処理が正常に完了しました。")
    else:
        print("\nデータ前処理でエラーが発生しました。詳細はログを確認してください。")