# fix_and_repartition_definitive.py
import os
import logging
from pathlib import Path
import shutil
import dask
import dask.dataframe as dd
import pandas as pd
import time
import gc

# --- 基本設定 ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Dask設定: メモリ不足を防ぐため、安全なシングルスレッドモードで実行
dask.config.set({'scheduler': 'single-threaded'})


# --- 設定 ---
class Config:
    PROJECT_ROOT = Path("/workspaces/project_forge")
    
    # 元の、正しいデータソースのパス
    SOURCE_PATH = PROJECT_ROOT / "data/1_XAUUSD_base_data/XAUUSD_tick_master.parquet.BACKUP"
    
    # 修正後のデータを出力する新しいディレクトリのパス
    DESTINATION_PATH = PROJECT_ROOT / "data/1_XAUUSD_base_data/XAUUSD_tick_master_FINAL_PATCHED"
    
    # 前方補完の対象カラム
    COLUMN_TO_PATCH = 'price_direction'

# --- パッチ修正用関数 ---
def patch_price_direction_robust(df: pd.DataFrame) -> pd.DataFrame:
    """
    データフレームを受け取り、price_directionを修正し、
    timeframeカラムのデータ型を'string'に統一する関数
    """
    if df.empty:
        return df
    
    # 1. price_directionの前方補完
    col = Config.COLUMN_TO_PATCH
    if col in df.columns:
        df[col] = df[col].replace(0, pd.NA).ffill().fillna(-1).astype('int8')
        
    # 2. ★★★ timeframeカラムのデータ型をstringに強制的に統一 ★★★
    #    これがデータ型不整合エラーを防ぐための最も重要な処理
    if 'timeframe' in df.columns:
        df['timeframe'] = df['timeframe'].astype(str)
        
    return df

# --- メイン実行パイプライン ---
def main():
    config = Config()
    start_time = time.time()
    logger.info("=== 【最終確定版】データ修正・再パーティション処理を開始します ===")

    if not config.SOURCE_PATH.is_dir():
        logger.error(f"入力ディレクトリが見つかりません: {config.SOURCE_PATH}")
        return

    if config.DESTINATION_PATH.exists():
        logger.warning(f"出力先が既に存在するため、一度削除します: {config.DESTINATION_PATH}")
        shutil.rmtree(config.DESTINATION_PATH, ignore_errors=True)
    # 出力ディレクトリはDaskが自動生成するため、ここでは作成しない

    logger.info("Daskでデータセットの構造を読み込んでいます...")
    # engine='pyarrow-dataset' を指定して、より堅牢な読み込みを行う
    ddf = dd.read_parquet(config.SOURCE_PATH)
    
    # --- Daskのmap_partitionsで全データを統一的に処理 ---
    logger.info("全パーティションに対して、パッチ適用とデータ型統一処理を開始します...")
    
    # Daskデータフレームのメタ情報（各カラムの型）を定義する
    # これにより、処理後のデータ構造が保証される
    meta = ddf._meta.copy()
    if config.COLUMN_TO_PATCH in meta.columns:
        meta[config.COLUMN_TO_PATCH] = meta[config.COLUMN_TO_PATCH].astype('int8')
    if 'timeframe' in meta.columns:
        # メタ情報でもtimeframeをカテゴリや辞書型ではなく、単純なオブジェクト(string)として定義
        meta['timeframe'] = meta['timeframe'].astype(object)

    # 全てのパーティション（チャンク）に対して、パッチ適用関数を並列実行
    patched_ddf = ddf.map_partitions(patch_price_direction_robust, meta=meta)
    
    # --- 統一された方法で全データを書き出し ---
    logger.info("処理済みデータをHive形式で書き出します（メタデータも自動生成されます）...")
    patched_ddf.to_parquet(
        config.DESTINATION_PATH,
        partition_on=['timeframe'], # timeframeカラムの値に基づいてディレクトリを自動生成
        write_index=False,
        engine='pyarrow'
    )

    logger.info(f"✅ 全ての処理が正常に完了しました。総処理時間: {(time.time() - start_time) / 60:.1f} 分")
    logger.info(f"修正後のデータが以下に出力されました:\n{config.DESTINATION_PATH}")

if __name__ == "__main__":
    main()