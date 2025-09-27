import polars as pl
import pandas as pd
from pathlib import Path
import logging

# --- 基本設定 ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# 入力と出力のパスを設定
SOURCE_PATH = "/workspaces/project_forge/data/1_XAUUSD_base_data/XAUUSD_tick_master_GOLDEN/timeframe=tick/*.parquet"
OUTPUT_DIR = Path("/workspaces/project_forge/data/0_tick_partitioned/")

# 出力ディレクトリを作成
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

def repartition_tick_data():
    """
    モノリシックなtickデータを日次のHiveパーティションに変換する一度限りのスクリプト。
    """
    logging.info("パーティショニング処理を開始します...")
    
    # 1. ソースとなる巨大なParquetファイルをLazyFrameとしてスキャン
    lf = pl.scan_parquet(SOURCE_PATH)
    
    # 2. 処理対象の期間を特定
    min_max_date_df = lf.select([
        pl.col("timestamp").min().alias("min_date"),
        pl.col("timestamp").max().alias("max_date")
    ]).collect()
    
    start_date = min_max_date_df["min_date"][0].date()
    end_date = min_max_date_df["max_date"][0].date()
    
    logging.info(f"処理対象期間: {start_date} から {end_date} まで")
    
    # 3. 処理を月単位のチャンクに分割してループ
    #    (1ヶ月分なら64GBメモリに確実に収まるという想定)
    date_range = pd.date_range(start=start_date, end=end_date, freq='MS') # MS: Month Start
    
    for i, month_start in enumerate(date_range):
        month_end = month_start + pd.offsets.MonthEnd(1)
        
        logging.info(f"--- チャンク処理中 ({i+1}/{len(date_range)}): {month_start.strftime('%Y-%m')} ---")
        
        # 4. LazyFrame.filter() を用いて1ヶ月分のデータを抽出
        month_lf = lf.filter(
            pl.col("timestamp").dt.date().is_between(month_start.date(), month_end.date())
        )
        
        # 5. パーティションキーとなる year, month, day 列を追加
        month_lf_with_keys = month_lf.with_columns([
            pl.col("timestamp").dt.year().alias("year"),
            pl.col("timestamp").dt.month().alias("month"),
            pl.col("timestamp").dt.day().alias("day")
        ])
        
        # 6. 1ヶ月分のLazyFrameを.collect()してメモリ上に実体化
        logging.info("データをメモリに収集しています...")
        month_df = month_lf_with_keys.collect()
        logging.info(f"収集完了: {len(month_df)}行")
        
        # 7. DataFrame.write_parquet() を用いてパーティション分割しながら書き出し
        # 修正: ファイル名のパターンを指定し、partition_byではなく手動でパーティション処理
        logging.info(f"パーティションを書き出しています: {OUTPUT_DIR}")
        
        # 日付ごとにグループ化して個別にファイルを保存
        for partition_keys, partition_data in month_df.group_by(['year', 'month', 'day']):
            year, month, day = partition_keys
            
            # パーティション用のディレクトリ構造を作成
            partition_dir = OUTPUT_DIR / f"year={year}" / f"month={month:02d}" / f"day={day:02d}"
            partition_dir.mkdir(parents=True, exist_ok=True)
            
            # パーティションキー列を除去してデータを保存
            data_to_save = partition_data.drop(['year', 'month', 'day'])
            
            # ファイル名を指定して保存
            output_file = partition_dir / "data.parquet"
            data_to_save.write_parquet(
                output_file,
                use_pyarrow=True
            )
            
        logging.info("書き出し完了。")
        
    logging.info("全てのパーティショニング処理が完了しました。")

if __name__ == "__main__":
    repartition_tick_data()