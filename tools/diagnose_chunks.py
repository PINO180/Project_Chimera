import pandas as pd
import os
from common.logger_setup import logger

def diagnose_all_nan_columns():
    """
    temp_chunks内の各Parquetファイルをスキャンし、
    全要素がNaNである「毒の列」を特定して報告する。
    """
    temp_dir = 'data/temp_chunks'
    logger.info(f"--- 🚀 チャンク診断ツールを開始：全NaN列（毒の列）を探索します ---")

    if not os.path.exists(temp_dir):
        logger.error(f"ディレクトリが見つかりません: {temp_dir}")
        return

    chunk_files = sorted([f for f in os.listdir(temp_dir) if f.endswith('.parquet')])
    if not chunk_files:
        logger.error(f"チャンクファイルが見つかりません。")
        return

    all_poison_columns = set()
    found_any = False

    for filename in chunk_files:
        file_path = os.path.join(temp_dir, filename)
        logger.info(f"--- スキャン中: {filename} ---")
        
        chunk_df = pd.read_parquet(file_path)
        
        # 全要素がNaNの列を特定
        is_all_nan = chunk_df.isna().all()
        poison_columns_in_chunk = is_all_nan[is_all_nan].index.tolist()
        
        if poison_columns_in_chunk:
            logger.warning(f"  -> ❌ 毒の列を発見: {poison_columns_in_chunk}")
            all_poison_columns.update(poison_columns_in_chunk)
            found_any = True
        else:
            logger.info("  -> ✅ このチャンクに毒の列はありませんでした。")

    logger.info("-" * 50)
    if found_any:
        logger.critical("--- ☠️ 診断完了：以下の「毒の列」が根本原因であると特定しました ☠️ ---")
        for col in sorted(list(all_poison_columns)):
            logger.critical(f"  - {col}")
    else:
        logger.info("--- ✅ 診断完了：全てのチャンクに「毒の列」は見つかりませんでした。---")
    logger.info("-" * 50)

if __name__ == '__main__':
    diagnose_all_nan_columns()