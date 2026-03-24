# preprocess_base_data.py

import os
import logging
from pathlib import Path
import shutil
import dask_cudf
import cudf
import time
from typing import List, Dict
import gc
import traceback
import cupy as cp

# DaskとcuDFのライブラリを最初にインポート
from dask_cuda import LocalCUDACluster
from dask.distributed import Client

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
    OUTPUT_FILE = PROJECT_ROOT / "data/1_XAUUSD_base_data/XAUUSDmulti_15timeframe_bars_exness_enriched.parquet"
    
    # 計算パラメータ
    VOLATILITY_WINDOW = 20
    VOLUME_WINDOW = 50
    ATR_WINDOW = 14
    
    # 除外カラム
    COLUMNS_TO_DROP = ['volatility', 'avg_volume']
    
    # 特別扱いする巨大な時間軸
    LARGE_TIMEFRAME = 'tick'


def calculate_enhanced_features(df: cudf.DataFrame, global_volume_mean: float = 0.0) -> cudf.DataFrame:
    """
    データ型変換を分離し、計算に集中させた最終版
    """
    # Daskワーカー上でcupyが利用可能であることを保証するために、関数内でインポート
    # import cupy as cp

    
    if len(df) == 0:
        return df
    
    df = df.sort_values('timestamp').reset_index(drop=True)
    
    # 1. 対数リターン
    ratio = df['close'] / df['close'].shift(1)
    df['log_return'] = cudf.Series(cp.log(ratio.fillna(1.0).values))
    
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
        true_range = cudf.concat([high_low, high_prev_close, low_prev_close], axis=1).max(axis=1, skipna=True)
        df['atr'] = true_range.ewm(span=ProcessingConfig.ATR_WINDOW, adjust=False).mean()
    else:
        df['atr'] = 0.0
    
    # 5. 価格方向性
    price_diff = df['close'].diff()
    df['price_direction'] = cudf.Series(cp.sign(price_diff.fillna(0.0).values))
    
    # 6. 価格モメンタム
    ratio_5 = df['close'] / df['close'].shift(5)
    df['price_momentum'] = ratio_5 - 1
    
    # 7. 出来高比率
    safe_avg_vol = df['rolling_avg_volume'].replace(0, 1.0)
    df['volume_ratio'] = df['volume'] / safe_avg_vol
    
    return df
        


def get_final_schema(base_columns: List[str]) -> Dict[str, str]:
    """処理後の完全なスキーマ（型情報）を返す"""
    
    # 元のカラムの型を定義
    schema = {
        'timeframe': 'object',
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
    # 不要なカラムをスキーマから削除
    for col in ProcessingConfig.COLUMNS_TO_DROP:
        if col in schema:
            del schema[col]
            
    # 新しいカラムの型を追加
    schema.update({
        'log_return': 'float32',
        'rolling_volatility': 'float32',
        'rolling_avg_volume': 'float64',
        'atr': 'float32',
        'price_direction': 'int8',
        'price_momentum': 'float32',
        'volume_ratio': 'float32'
    })

    # 元のカラムリストに含まれるものだけを返す
    final_cols = [col for col in base_columns if col in schema]
    # 新しいカラムを追加
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
    """メイン処理関数 - 全てのデバッグを反映した最終決定版"""
    config = ProcessingConfig()
    start_time = time.time()

    logger.info("=== 純粋なDaskワークフローによるGPU前処理パイプライン開始 ===")

    if not config.INPUT_FILE.exists():
        logger.error(f"Input file not found: {config.INPUT_FILE}")
        return False
    if config.OUTPUT_FILE.exists():
        logger.warning(f"Output path {config.OUTPUT_FILE} already exists. Removing it.")
        shutil.rmtree(config.OUTPUT_FILE, ignore_errors=True)
    config.OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

    try:
        # Daskの低レベルAPIをインポート
        import dask

        ddf = dask_cudf.read_parquet(config.INPUT_FILE)
        all_timeframes = ddf['timeframe'].unique().compute().to_pandas().tolist()
        in_memory_timeframes = [tf for tf in all_timeframes if tf != config.LARGE_TIMEFRAME]
        
        processed_ddfs: List[dask_cudf.DataFrame] = []
        final_schema = get_final_schema(ddf.columns)
        meta_df = cudf.DataFrame({k: [] for k in final_schema.keys()}).astype(final_schema)

        # --- Daskタスクグラフの構築 ---
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
            tick_ddf = ddf[ddf['timeframe'] == config.LARGE_TIMEFRAME]
            global_volume_mean = tick_ddf['volume'].mean().compute()
            logger.info(f"Global volume mean for 'tick' is {global_volume_mean}")
            cols_to_drop = [col for col in config.COLUMNS_TO_DROP if col in tick_ddf.columns]
            if cols_to_drop:
                tick_ddf = tick_ddf.drop(columns=cols_to_drop)
            
            enhanced_tick_ddf = tick_ddf.map_partitions(
                calculate_enhanced_features, global_volume_mean=float(global_volume_mean), meta=meta_df
            )
            processed_ddfs.append(enhanced_tick_ddf)
            logger.info(f"Task for '{config.LARGE_TIMEFRAME}' created.")

        # --- 全タスクの実行と保存 ---
        logger.info("--- Finalizing: Concatenating tasks and executing graph ---")
        final_start = time.time()
        
        if not processed_ddfs:
            logger.warning("No data to process. Exiting.")
            return True

        final_ddf = dask_cudf.concat(processed_ddfs, ignore_unknown_divisions=True)
        
        # ▽▼▽▼▽▼【dask.delayedを使った正しい空パーティション除去】▼▼▼▼▽▼
        logger.info("Filtering out empty partitions before saving...")
        parts = final_ddf.to_delayed()
        
        # 各パーティションの長さを計算する遅延タスクのリストを作成
        lengths = [dask.delayed(len)(p) for p in parts]
        # 全ての長さを一度に計算
        computed_lengths = dask.compute(*lengths)

        # 長さが0より大きいパーティションだけを再結合
        parts_with_data = [p for p, length in zip(parts, computed_lengths) if length > 0]

        if not parts_with_data:
            logger.warning("All partitions resulted in empty data. No output generated.")
            return True
            
        final_ddf = dask_cudf.from_delayed(parts_with_data, meta=meta_df)
        logger.info(f"Filtered down to {final_ddf.npartitions} non-empty partitions.")
        # △▲△▲△▲【ここまで】△▲△▲△▲
        
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
    with LocalCUDACluster(
        n_workers=1, 
        device_memory_limit='10GB',
        rmm_pool_size='8GB'
    ) as cluster, Client(cluster) as client:
        logger.info("Dask-CUDA Client initialized for the entire script.")
        logger.info(f"Dashboard link: {client.dashboard_link}")
        
        success = main()

    if success:
        print("\nハイブリッドアーキテクチャによるデータ前処理が正常に完了しました。")
    else:
        print("\nデータ前処理でエラーが発生しました。詳細はログを確認してください。")