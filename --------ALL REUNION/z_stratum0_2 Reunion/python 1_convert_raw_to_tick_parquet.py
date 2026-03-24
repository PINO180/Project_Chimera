#!/usr/bin/env python3
"""
修正版 ティックデータ変換スクリプト
正しいフォーマット対応: DATE TIME BID ASK LAST VOLUME FLAGS
"""

import polars as pl
import os
from pathlib import Path
import time
import gc

def process_chunk_data(chunk_data, chunk_num):
    """
    チャンクデータを処理する関数
    """
    if len(chunk_data) == 0:
        return pl.DataFrame()
        
    print(f"  チャンク {chunk_num}: {len(chunk_data)} 行処理中...")
    
    # チャンクをPolarsデータフレームに変換
    df = pl.DataFrame({
        'raw_line': chunk_data
    })
    
    # タブ区切りデータを分割
    df = df.with_columns([
        pl.col('raw_line').str.split('\t').alias('split_data')
    ]).filter(
        pl.col('split_data').list.len() >= 6  # 最低6列あるデータのみ
    ).with_columns([
        pl.col('split_data').list.get(0).alias('date_str'),      # DATE
        pl.col('split_data').list.get(1).alias('time_str'),      # TIME  
        pl.col('split_data').list.get(2).alias('bid_str'),       # BID
        pl.col('split_data').list.get(3).alias('ask_str'),       # ASK
        pl.col('split_data').list.get(4).alias('last_str'),      # LAST
        pl.col('split_data').list.get(5).alias('volume_str'),    # VOLUME
        pl.col('split_data').list.get(6).alias('flags_str')      # FLAGS
    ])
    
    print(f"    分割後: {df.shape[0]} 行")
    
    # データ型変換とクリーニング
    df = df.with_columns([
        # 日時の結合と変換
        (pl.col('date_str') + ' ' + pl.col('time_str')).alias('datetime_str')
    ]).with_columns([
        # 日時変換（複数フォーマット対応）
        pl.col('datetime_str')
        .str.to_datetime(format="%Y.%m.%d %H:%M:%S.%3f", strict=False)
        .alias("datetime"),
        
        # 数値列の変換
        pl.when(pl.col('bid_str').str.len_chars() > 0)
        .then(pl.col('bid_str').cast(pl.Float64, strict=False))
        .otherwise(None)
        .alias("bid"),
        
        pl.when(pl.col('ask_str').str.len_chars() > 0)
        .then(pl.col('ask_str').cast(pl.Float64, strict=False))
        .otherwise(None)
        .alias("ask"),
        
        pl.when(pl.col('last_str').str.len_chars() > 0)
        .then(pl.col('last_str').cast(pl.Float64, strict=False))
        .otherwise(None)
        .alias("last"),
        
        pl.when(pl.col('volume_str').str.len_chars() > 0)
        .then(pl.col('volume_str').cast(pl.Int64, strict=False))
        .otherwise(None)
        .alias("volume"),
        
        pl.when(pl.col('flags_str').str.len_chars() > 0)
        .then(pl.col('flags_str').cast(pl.Int32, strict=False))
        .otherwise(None)
        .alias("flags")
    ])
    
    # 有効なデータのフィルタリング
    df = df.filter(
        (pl.col("datetime").is_not_null()) & 
        (pl.col("bid").is_not_null()) & 
        (pl.col("ask").is_not_null()) &
        (pl.col("bid") > 0) &  # 正の値のみ
        (pl.col("ask") > 0)
    ).select([
        "datetime", "bid", "ask", "last", "volume", "flags"
    ])
    
    print(f"    有効データ: {df.shape[0]} 行")
    
    if df.shape[0] == 0:
        return df
    
    # スプレッドと中値価格を計算
    df = df.with_columns([
        (pl.col("ask") - pl.col("bid")).alias("spread"),
        ((pl.col("ask") + pl.col("bid")) / 2).alias("mid_price")
    ])
    
    # データ型を最適化（メモリ節約）
    df = df.with_columns([
        pl.col("bid").cast(pl.Float32),
        pl.col("ask").cast(pl.Float32),
        pl.col("last").cast(pl.Float32),
        pl.col("spread").cast(pl.Float32),
        pl.col("mid_price").cast(pl.Float32)
    ])
    
    return df

def convert_raw_to_tick_parquet():
    """
    大容量ティックCSVデータをParquet形式に変換
    """
    # パス設定
    input_file = r"C:\clonecloneclone\data\XAUUSDm_row.csv"
    output_file = r"C:\clonecloneclone\data\XAUUSDtick_exness.parquet"
    
    print(f"=== 修正版 ティックデータ変換開始 ===")
    print(f"入力: {input_file}")
    print(f"出力: {output_file}")
    
    start_time = time.time()
    chunk_size = 50000  # 5万行ずつ処理（メモリ節約）
    total_rows = 0
    processed_chunks = []
    temp_files = []
    
    try:
        # ファイルサイズ確認
        file_size_mb = os.path.getsize(input_file) / (1024**2)
        print(f"ファイルサイズ: {file_size_mb:.1f} MB")
        
        # チャンク処理でファイルを読み込み
        with open(input_file, 'r', encoding='utf-8') as f:
            header_line = f.readline().strip()  # ヘッダーをスキップ
            print(f"ヘッダー: {header_line}")
            
            chunk_num = 0
            chunk_data = []
            
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if line:  # 空行をスキップ
                    chunk_data.append(line)
                    
                # チャンクサイズに達した場合、処理実行
                if len(chunk_data) >= chunk_size:
                    chunk_num += 1
                    
                    # チャンクを処理
                    df_chunk = process_chunk_data(chunk_data, chunk_num)
                    
                    if df_chunk.shape[0] > 0:
                        # 中間ファイルに直接保存（メモリ節約）
                        temp_file = output_file.replace('.parquet', f'_temp_{chunk_num:04d}.parquet')
                        df_chunk.write_parquet(
                            temp_file,
                            compression="zstd",
                            use_pyarrow=True
                        )
                        temp_files.append(temp_file)
                        total_rows += df_chunk.shape[0]
                        
                        print(f"    → 中間ファイル保存: {df_chunk.shape[0]} 行")
                    
                    # メモリクリーンアップ
                    del df_chunk
                    chunk_data = []
                    gc.collect()
                    
                    print(f"  累計処理済み: {total_rows:,} 行 (進捗: {line_num:,} 行読み込み)")
                    
                    # 進捗表示（10チャンクごと）
                    if chunk_num % 10 == 0:
                        elapsed = time.time() - start_time
                        rate = line_num / elapsed if elapsed > 0 else 0
                        print(f"  処理速度: {rate:,.0f} 行/秒")
            
            # 残りのデータを処理
            if chunk_data:
                chunk_num += 1
                print(f"最終チャンク処理中...")
                df_chunk = process_chunk_data(chunk_data, chunk_num)
                if df_chunk.shape[0] > 0:
                    temp_file = output_file.replace('.parquet', f'_temp_{chunk_num:04d}.parquet')
                    df_chunk.write_parquet(temp_file, compression="zstd", use_pyarrow=True)
                    temp_files.append(temp_file)
                    total_rows += df_chunk.shape[0]
        
        print(f"\n=== チャンク処理完了 ===")
        print(f"総処理行数: {total_rows:,}")
        print(f"中間ファイル数: {len(temp_files)}")
        
        if total_rows == 0:
            print("警告: 有効なデータが見つかりませんでした")
            return
        
        # 中間ファイルを結合
        print("中間ファイル結合中...")
        
        # 最初のファイルでスキーマ確認
        first_df = pl.read_parquet(temp_files[0])
        print(f"データスキーマ: {first_df.dtypes}")
        print(f"サンプルデータ:")
        print(first_df.head(3))
        
        # 全中間ファイルを読み込んで結合
        all_chunks = []
        for i, temp_file in enumerate(temp_files):
            df_temp = pl.read_parquet(temp_file)
            all_chunks.append(df_temp)
            if (i + 1) % 100 == 0:
                print(f"  読み込み進捗: {i+1}/{len(temp_files)}")
        
        # 最終結合
        print("最終データ結合中...")
        final_df = pl.concat(all_chunks, how="vertical")
        
        # 日時でソート
        print("データソート中...")
        final_df = final_df.sort("datetime")
        
        # 統計情報表示
        print(f"\n=== 処理結果 ===")
        print(f"最終行数: {final_df.shape[0]:,}")
        print(f"列数: {final_df.shape[1]}")
        
        if final_df.shape[0] > 0:
            print(f"日時範囲: {final_df.select(pl.col('datetime').min()).item()} - {final_df.select(pl.col('datetime').max()).item()}")
            print(f"BID範囲: {final_df.select(pl.col('bid').min()).item():.2f} - {final_df.select(pl.col('bid').max()).item():.2f}")
            print(f"ASK範囲: {final_df.select(pl.col('ask').min()).item():.2f} - {final_df.select(pl.col('ask').max()).item():.2f}")
        
        # NULL値確認
        null_counts = final_df.null_count()
        print(f"NULL値:")
        for col, count in zip(final_df.columns, null_counts.row(0)):
            print(f"  {col}: {count:,}")
        
        # 出力ディレクトリ作成
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        
        # 最終Parquet出力
        print("最終Parquetファイル出力中...")
        final_df.write_parquet(
            output_file,
            compression="zstd",
            use_pyarrow=True,
            statistics=True,
            row_group_size=100000
        )
        
        # 中間ファイルクリーンアップ
        print("中間ファイル削除中...")
        for temp_file in temp_files:
            try:
                os.remove(temp_file)
            except:
                pass
        
        # 処理完了情報
        end_time = time.time()
        processing_time = end_time - start_time
        output_file_size = os.path.getsize(output_file) / (1024**2)
        
        print(f"\n=== 変換完了 ===")
        print(f"処理時間: {processing_time/60:.1f}分 ({processing_time:.1f}秒)")
        print(f"出力ファイル: {output_file}")
        print(f"出力ファイルサイズ: {output_file_size:.1f} MB")
        print(f"圧縮率: {file_size_mb/output_file_size:.1f}x")
        print(f"処理速度: {final_df.shape[0]/processing_time:,.0f} 行/秒")
        
        # サンプルデータ表示
        print(f"\n最終データサンプル:")
        print(final_df.head(5))
        
    except Exception as e:
        print(f"エラーが発生しました: {e}")
        import traceback
        traceback.print_exc()
        
        # エラー時も中間ファイルクリーンアップ
        for temp_file in temp_files:
            try:
                os.remove(temp_file)
            except:
                pass
        raise

if __name__ == "__main__":
    convert_raw_to_tick_parquet()