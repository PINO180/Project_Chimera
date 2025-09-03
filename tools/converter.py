import pandas as pd
import pyarrow.parquet as pq
import time
import os

def convert_parquet_to_jsonl(parquet_path, json_path, chunk_size=100000):
    """巨大なParquetファイルを、メモリ効率の良い方法でJSONLに変換する"""
    try:
        print(f"\n=== ParquetからJSONLへの変換開始 ===")
        print(f"入力: {parquet_path}")
        print(f"出力: {json_path}")
        
        parquet_file = pq.ParquetFile(parquet_path)
        start_time = time.time()
        
        if os.path.exists(json_path):
            os.remove(json_path)
        
        total_rows = 0
        for i, batch in enumerate(parquet_file.iter_batches(batch_size=chunk_size)):
            chunk_df = batch.to_pandas()
            chunk_df.to_json(json_path, orient='records', lines=True, mode='a')
            total_rows += len(chunk_df)
            print(f"  進捗: {total_rows:,} / {parquet_file.num_rows:,} 行を処理完了")
            
        print(f"処理時間: {time.time() - start_time:.2f} 秒")
        print(f"正常に出力されました: {json_path}")
        
    except Exception as e:
        print(f"エラーが発生しました: {e}")

def convert_parquet_to_csv(parquet_path, csv_path, chunk_size=100000):
    """巨大なParquetファイルを、メモリ効率の良い方法でCSVに変換する"""
    try:
        print(f"\n=== ParquetからCSVへの変換開始 ===")
        print(f"入力: {parquet_path}")
        print(f"出力: {csv_path}")
        
        parquet_file = pq.ParquetFile(parquet_path)
        start_time = time.time()
        
        if os.path.exists(csv_path):
            os.remove(csv_path)

        total_rows = 0
        header_written = False
        for i, batch in enumerate(parquet_file.iter_batches(batch_size=chunk_size)):
            chunk_df = batch.to_pandas()
            
            # 最初のチャンクのみヘッダーを書き込み、以降は追記モードでデータのみ書き込む
            if not header_written:
                chunk_df.to_csv(csv_path, mode='w', index=False, header=True)
                header_written = True
            else:
                chunk_df.to_csv(csv_path, mode='a', index=False, header=False)

            total_rows += len(chunk_df)
            print(f"  進捗: {total_rows:,} / {parquet_file.num_rows:,} 行を処理完了")

        print(f"処理時間: {time.time() - start_time:.2f} 秒")
        print(f"正常に出力されました: {csv_path}")

    except Exception as e:
        print(f"エラーが発生しました: {e}")

if __name__ == "__main__":
    # --- ここから設定 ---
    # マスターとなるParquetファイルのパス
    input_parquet = r"C:\project_forge\data\temp_chunks\parquet\02_combined_563features.parquet"

    # --- 実行したい変換を選択 ---
    # CSVに変換したい場合
    output_csv = r"C:\project_forge\data\temp_chunks\csv\02_combined_563features.csv"
    convert_parquet_to_csv(input_parquet, output_csv)

    # JSONに変換したい場合
    output_json = r"C:\project_forge\data\temp_chunks\json\02_combined_563features.json"
    convert_parquet_to_jsonl(input_parquet, output_json)