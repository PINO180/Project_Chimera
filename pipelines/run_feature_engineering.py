import pandas as pd
import os
import gc
import argparse
import multiprocessing
import itertools
from tqdm import tqdm
from common.logger_setup import logger

from features.feature_logic import FeatureUniverseGenerator

INPUT_FILE = 'data/XAUUSD_1m.csv'
TEMP_DIR = 'data/temp_chunks'

def run_sequential_for_mfdfa(chunk_size, window_size):
    """MFDFAの依存性を考慮した順次処理版"""
    logger.info(f"--- 🔄 MFDFA順次処理モード開始 ---")
    logger.info(f"設定: チャンクサイズ={chunk_size:,}行, ウィンドウサイズ={window_size}")
    
    os.makedirs(TEMP_DIR, exist_ok=True)
    
    # 既存チャンクの確認
    existing_chunks = {int(f.split('_')[-1].split('.')[0]) for f in os.listdir(TEMP_DIR) 
                      if f.startswith('feature_chunk_')}
    
    if existing_chunks:
        logger.info(f"既存チャンク: {len(existing_chunks)}個")
    
    # 順次処理でチャンクを作成
    chunk_iterator = pd.read_csv(INPUT_FILE, chunksize=chunk_size)
    
    for i, chunk_df in enumerate(chunk_iterator):
        if i in existing_chunks:
            logger.info(f"チャンク {i}: スキップ（既存）")
            continue
            
        logger.info(f"チャンク {i}: 処理開始 - 形状: {chunk_df.shape}")
        
        try:
            # データ前処理
            chunk_df['datetime'] = pd.to_datetime(chunk_df['datetime'])
            chunk_df.set_index('datetime', inplace=True)
            
            # 特徴量生成
            generator = FeatureUniverseGenerator()
            feature_chunk = generator.generate(chunk_df)
            
            logger.info(f"チャンク {i}: 特徴量生成完了 - 形状: {feature_chunk.shape}")
            
            # 保存
            chunk_output_path = os.path.join(TEMP_DIR, f'feature_chunk_{i}.parquet')
            feature_chunk.to_parquet(chunk_output_path)
            
            logger.info(f"チャンク {i}: 保存完了 - {chunk_output_path}")
            
            # メモリクリーンアップ
            del chunk_df, feature_chunk
            gc.collect()
            
        except Exception as e:
            logger.error(f"チャンク {i} でエラー: {e}", exc_info=True)
            return False
    
    logger.info("--- ✅ 順次処理完了 ---")
    return True

def process_single_chunk(task_data):
    """単一のチャンクデータフレームに対して特徴量生成のみを行うワーカー関数"""
    i, chunk_df = task_data
    try:
        logger.info(f"チャンク {i}: 処理開始。")
        chunk_df['datetime'] = pd.to_datetime(chunk_df['datetime'])
        chunk_df.set_index('datetime', inplace=True)

        generator = FeatureUniverseGenerator()
        feature_chunk = generator.generate(chunk_df)
        
        logger.info(f"チャンク {i}: 特徴量生成完了。")

        chunk_output_path = os.path.join(TEMP_DIR, f'feature_chunk_{i}.parquet')
        feature_chunk.to_parquet(chunk_output_path)
        
        del chunk_df, feature_chunk
        gc.collect()
        logger.info(f"チャンク {i}: ファイル保存完了。")
        return f"Chunk {i} completed successfully."
    except Exception as e:
        logger.error(f"Chunk {i} でエラー発生: {e}", exc_info=True)
        return f"Chunk {i} failed."

def run_parallel_feature_generation(num_workers, chunk_size):
    """特徴量生成を並列で実行する"""
    logger.info(f"--- 🚀 特徴量生成（並列処理モード: {num_workers}ワーカー, チャンクサイズ: {chunk_size}）を開始します 🚀 ---")
    os.makedirs(TEMP_DIR, exist_ok=True)
    
    existing_chunks = {int(f.split('_')[-1].split('.')[0]) for f in os.listdir(TEMP_DIR) if f.startswith('feature_chunk_')}
    
    total_rows = sum(1 for row in open(INPUT_FILE)) - 1
    total_chunks = (total_rows // chunk_size) + 1
    
    tasks = []
    chunk_iterator = pd.read_csv(INPUT_FILE, chunksize=chunk_size)
    for i, chunk_df in enumerate(chunk_iterator):
        if i not in existing_chunks:
            tasks.append((i, chunk_df))
            
    if not tasks:
        logger.info("全てのチャンクは既に処理済みです。スキップします。")
        return True

    logger.info(f"合計{total_chunks}チャンクのうち、未処理の{len(tasks)}チャンクを処理します。")
    
    with multiprocessing.Pool(processes=num_workers) as pool:
        results = list(tqdm(pool.imap(process_single_chunk, tasks), total=len(tasks), desc="Processing Chunks in Parallel"))
        
    logger.info("--- ✅ 全てのチャンクの並列処理が完了しました ---")
    if any("failed" in res for res in results):
        logger.error("一部のチャンクの処理に失敗しました。ログを確認してください。")
        return False
    return True

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Project Forge - Feature Generation")
    parser.add_argument('--workers', type=int, default=8, help='並列処理で使用するCPUコア数')
    parser.add_argument('--chunk_size', type=int, default=250000, help='一度に処理する行数（チャンクサイズ）')
    parser.add_argument('--window_size', type=int, default=500, help='MFDFAウィンドウサイズ')
    parser.add_argument('--mode', type=str, default='parallel', choices=['parallel', 'sequential'],
                       help='処理モード: parallel（並列）またはsequential（順次）')
    
    args = parser.parse_args()

    logger.info("--- 💎 特徴量チャンクの生成を開始します 💎 ---")
    
    if args.mode == 'sequential':
        logger.info("順次処理モードで実行します")
        success = run_sequential_for_mfdfa(args.chunk_size, args.window_size)
    else:
        logger.info("並列処理モードで実行します")
        success = run_parallel_feature_generation(args.workers, args.chunk_size)
    
    if success:
        logger.info("--- 🎉🎉🎉 全ての特徴量チャンクの生成が完了しました！ 🎉🎉🎉 ---")
    else:
        logger.error("--- ❌ 特徴量チャンクの生成に失敗しました。 ---")