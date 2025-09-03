import pandas as pd
import os
from common.logger_setup import logger

def verify_schema_consistency():
    """
    temp_chunks内の全Parquetファイルの列構成が完全に一致しているか検証する。
    """
    temp_dir = 'data/temp_chunks'
    logger.info(f"--- 🚀 チャンク設計図（スキーマ）の一貫性検証を開始します ---")

    if not os.path.exists(temp_dir) or not os.listdir(temp_dir):
        logger.error(f"検証対象のチャンクファイルが見つかりません: {temp_dir}")
        return

    chunk_files = sorted([f for f in os.listdir(temp_dir) if f.endswith('.parquet')])
    
    # 基準となるスキーマ（最初のチャンクの列構成）を取得
    base_schema = pd.read_parquet(os.path.join(temp_dir, chunk_files[0])).columns
    logger.info(f"基準スキーマを '{chunk_files[0]}' から取得しました（{len(base_schema)}列）。")

    all_consistent = True
    # 2番目以降のチャンクを基準スキーマと比較
    for filename in chunk_files[1:]:
        file_path = os.path.join(temp_dir, filename)
        current_schema = pd.read_parquet(file_path).columns
        
        if not base_schema.equals(current_schema):
            logger.error(f"❌ スキーマ不一致: '{filename}' の設計図が基準と異なります。")
            
            # 差分を具体的に表示
            missing_in_current = base_schema.difference(current_schema)
            extra_in_current = current_schema.difference(base_schema)
            if len(missing_in_current) > 0:
                logger.error(f"  -> 基準にあるべき列がありません: {missing_in_current.tolist()}")
            if len(extra_in_current) > 0:
                logger.error(f"  -> 基準にない余分な列があります: {extra_in_current.tolist()}")
            all_consistent = False

    if all_consistent:
        logger.info("--- ✅ 全てのチャンクの設計図は完全に一致しています！ ---")
    else:
        logger.error("--- ❌ 設計図の不一致が検出されました。特徴量生成をやり直してください。 ---")

if __name__ == '__main__':
    verify_schema_consistency()