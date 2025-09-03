#!/usr/bin/env python3
"""
Parquetファイルの読み込み可能性を簡単にテストするスクリプト
"""

import pandas as pd
import pyarrow.parquet as pq
from pathlib import Path
import time
import psutil
import os

def test_parquet_readability(file_path: str):
    """Parquetファイルの読み込み可能性をテスト"""
    
    file_path = Path(file_path)
    
    print(f"テスト対象: {file_path.name}")
    print(f"パス: {file_path}")
    print("-" * 50)
    
    # 1. ファイル存在確認
    if not file_path.exists():
        print("❌ ファイルが存在しません")
        return False
    
    # 2. ファイルサイズ確認
    file_size = file_path.stat().st_size
    file_size_gb = file_size / (1024**3)
    print(f"ファイルサイズ: {file_size_gb:.2f} GB")
    
    # 3. 利用可能メモリ確認
    memory = psutil.virtual_memory()
    available_gb = memory.available / (1024**3)
    print(f"利用可能メモリ: {available_gb:.2f} GB")
    
    if file_size_gb > available_gb * 0.8:
        print("⚠️  警告: ファイルサイズが利用可能メモリの80%を超えています")
    
    # 4. Parquetメタデータ読み込みテスト
    print("\n1. メタデータ読み込みテスト...")
    try:
        start_time = time.time()
        parquet_file = pq.ParquetFile(file_path)
        metadata_time = time.time() - start_time
        
        print(f"✅ メタデータ読み込み成功 ({metadata_time:.2f}秒)")
        print(f"   行数: {parquet_file.metadata.num_rows:,}")
        print(f"   列数: {len(parquet_file.schema):,}")
        print(f"   行グループ数: {parquet_file.metadata.num_row_groups}")
        
    except Exception as e:
        print(f"❌ メタデータ読み込み失敗: {e}")
        return False
    
    # 5. 小サンプル読み込みテスト
    print("\n2. 小サンプル読み込みテスト (最初の1000行)...")
    try:
        start_time = time.time()
        df_sample = pd.read_parquet(file_path, engine='pyarrow')
        df_sample = df_sample.head(1000)
        sample_time = time.time() - start_time
        
        print(f"✅ サンプル読み込み成功 ({sample_time:.2f}秒)")
        print(f"   サンプル形状: {df_sample.shape}")
        print(f"   メモリ使用量: {df_sample.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
        
        # メモリ解放
        del df_sample
        
    except Exception as e:
        print(f"❌ サンプル読み込み失敗: {e}")
        return False
    
    # 6. 全量読み込み可能性の判断
    print("\n3. 全量読み込み可能性評価...")
    
    if file_size_gb < available_gb * 0.3:
        print("✅ 全量読み込み: 安全に実行可能")
        recommendation = "full"
    elif file_size_gb < available_gb * 0.6:
        print("⚠️  全量読み込み: 可能だが注意が必要")
        recommendation = "careful"
    else:
        print("❌ 全量読み込み: 推奨されません (チャンク処理を使用)")
        recommendation = "chunked"
    
    # 7. オプション: 全量読み込みテスト
    if recommendation in ["full", "careful"]:
        test_full = input("\n全量読み込みテストを実行しますか？ (y/N): ").strip().lower()
        if test_full == 'y':
            print("\n4. 全量読み込みテスト実行中...")
            try:
                start_time = time.time()
                
                # メモリ使用量監視
                process = psutil.Process(os.getpid())
                initial_memory = process.memory_info().rss / 1024**2
                
                df_full = pd.read_parquet(file_path)
                load_time = time.time() - start_time
                
                final_memory = process.memory_info().rss / 1024**2
                memory_used = final_memory - initial_memory
                
                print(f"✅ 全量読み込み成功!")
                print(f"   読み込み時間: {load_time:.2f}秒")
                print(f"   データ形状: {df_full.shape}")
                print(f"   メモリ使用量増加: {memory_used:.2f} MB")
                print(f"   DataFrameサイズ: {df_full.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
                
                del df_full
                
            except Exception as e:
                print(f"❌ 全量読み込み失敗: {e}")
                return False
    
    print("\n" + "="*50)
    print("結果: ファイルは正常に読み込み可能です")
    return True

def main():
    """メイン関数"""
    
    # デフォルトファイルパス
    default_file = r"C:\project_forge\data\temp_chunks\parquet\01_mfdfa_45features.parquet"
    
    print("Parquet読み込み可能性テスト")
    print("=" * 50)
    
    file_path = input(f"ファイルパス (Enter でデフォルト使用): ").strip()
    if not file_path:
        file_path = default_file
    
    print()
    success = test_parquet_readability(file_path)
    
    if success:
        print("\n✅ テスト完了: ファイルは正常です")
    else:
        print("\n❌ テスト失敗: ファイルに問題があります")

if __name__ == "__main__":
    main()