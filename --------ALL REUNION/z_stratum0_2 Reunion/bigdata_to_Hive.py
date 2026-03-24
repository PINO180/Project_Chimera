# repartition_safe_v1.py
import dask
import dask.dataframe as dd
from pathlib import Path
import shutil
import time

# --- 設定 ---
source_file = Path("/workspaces/project_forge/data/1_XAUUSD_base_data/XAUUSD_tick_master_FINAL_PATCHED")
output_dir = Path("/workspaces/project_forge/data/1_XAUUSD_base_data/XAUUSD_tick_master_repartitioned")

# --- 実行 ---
if __name__ == "__main__":
    if not source_file.exists():
        print(f"エラー: ソースファイルが見つかりません: {source_file}")
    else:
        if output_dir.exists():
            print(f"既存の出力ディレクトリ {output_dir} を削除します...")
            shutil.rmtree(output_dir)
        
        print(f"ソースファイル: {source_file}")
        print(f"出力先: {output_dir}")
        
        start_time = time.time()
        
        ddf = dd.read_parquet(source_file)
        
        print("パーティション分割処理を開始します（シングルスレッドモード）。時間はかかりますが安全です...")
        
        # このブロック内では、Daskは並列処理をせず、1つずつタスクを処理します
        with dask.config.set(scheduler='single-threaded'):
            ddf.to_parquet(
                output_dir,
                partition_on='timeframe',
                write_index=False,
                engine='pyarrow'
            )
        
        end_time = time.time()
        duration = end_time - start_time
        
        print("\n" + "="*50)
        print("✅ Hiveパーティションの再作成が完了しました！")
        print(f"出力先ディレクトリ: {output_dir}")
        print(f"合計処理時間: {duration / 60:.2f} 分")
        print("="*50)