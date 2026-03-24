import dask.dataframe as dd
import os
from pathlib import Path
import logging

# --- 基本設定 ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__) # ▼▼▼ この一行が抜けていました ▼▼▼

# --- 設定 ---
# パッチ修正後の、正常に完了したデータセットのパス
DATASET_PATH = Path("/workspaces/project_forge/data/1_XAUUSD_base_data/XAUUSD_tick_master_FINAL_PATCHED")

def main():
    logger.info(f"--- メタデータファイルの再生成を開始します ---")
    logger.info(f"対象ディレクトリ: {DATASET_PATH}")

    if not DATASET_PATH.is_dir():
        logger.error("指定されたパスが見つからないか、ディレクトリではありません。")
        return

    # サブディレクトリ内にある全ての`.parquet`ファイルをリストアップ
    try:
        file_list = [
            os.path.join(root, f)
            for root, _, files in os.walk(DATASET_PATH)
            for f in files if f.endswith('.parquet')
        ]

        if not file_list:
            logger.warning("Parquetファイルが見つかりませんでした。")
            return

        logger.info(f"{len(file_list)}個のParquetファイルを検出しました。メタデータを生成します...")
        
        # Daskの機能を使ってメタデータファイルを正しく作成
        dd.io.parquet.create_metadata_file(file_list)
        
        logger.info("✅ メタデータファイルが正常に作成されました。")

    except Exception as e:
        logger.error(f"メタデータファイルの生成中にエラーが発生しました: {e}")

if __name__ == "__main__":
    main()