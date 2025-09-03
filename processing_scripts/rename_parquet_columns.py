import pandas as pd
import os
import re
from tqdm import tqdm
from common.logger_setup import logger

def sanitize_column_name(col_name: str) -> str:
    """単一の列名を浄化する"""
    return re.sub(r'[^a-zA-Z0-9_]', '_', str(col_name))

def rename_columns_in_chunks():
    """
    temp_chunks内の全Parquetファイルの列名を、HDF5互換の形式に修正（上書き）する。
    """
    temp_dir = 'data/temp_chunks'
    if not os.path.exists(temp_dir):
        logger.error(f"ディレクトリが見つかりません: {temp_dir}")
        return

    chunk_files = sorted([f for f in os.listdir(temp_dir) if f.endswith('.parquet')])
    if not chunk_files:
        logger.error(f"ディレクトリ内にParquetファイルが見つかりません。")
        return

    logger.info(f"--- 🚀 {len(chunk_files)}個のParquetファイルの列名修正を開始します 🚀 ---")

    for filename in tqdm(chunk_files, desc="Sanitizing columns in chunks"):
        file_path = os.path.join(temp_dir, filename)
        
        # チャンクを読み込み
        chunk_df = pd.read_parquet(file_path)
        
        # 列名を修正
        original_columns = chunk_df.columns
        new_columns = {col: sanitize_column_name(col) for col in original_columns}
        chunk_df.rename(columns=new_columns, inplace=True)
        
        # 修正したデータでファイルを上書き
        chunk_df.to_parquet(file_path)

    logger.info("--- ✅ 全てのチャンクの列名修正が完了しました ---")

if __name__ == '__main__':
    rename_columns_in_chunks()