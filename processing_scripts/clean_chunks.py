import pandas as pd
import os
import gc
from tqdm import tqdm
from common.logger_setup import logger

def clean_individual_chunks():
    """
    temp_chunks内の各Parquetファイルを個別にクレンジング（NaN行を削除）し、
    クリーンな状態で上書き保存する。
    """
    temp_dir = 'data/temp_chunks'
    logger.info(f"--- 🚀 チャンク個別クレンジング処理を開始します ---")

    if not os.path.exists(temp_dir):
        logger.error(f"ディレクトリが見つかりません: {temp_dir}")
        return

    chunk_files = sorted([f for f in os.listdir(temp_dir) if f.endswith('.parquet')])
    
    total_rows_before = 0
    total_rows_after = 0

    for filename in tqdm(chunk_files, desc="Cleaning individual chunks"):
        file_path = os.path.join(temp_dir, filename)
        
        # チャンクを読み込み
        chunk_df = pd.read_parquet(file_path)
        total_rows_before += len(chunk_df)
        
        # NaNを含む行を削除
        chunk_df.dropna(inplace=True)
        total_rows_after += len(chunk_df)
        
        # クリーンな状態でファイルを上書き
        chunk_df.to_parquet(file_path)
        
        del chunk_df
        gc.collect()

    logger.info("--- ✅ 全てのチャンクの個別クレンジングが完了しました ---")
    logger.info(f"合計 {total_rows_before - total_rows_after} 行が除去されました。")

if __name__ == '__main__':
    clean_individual_chunks()