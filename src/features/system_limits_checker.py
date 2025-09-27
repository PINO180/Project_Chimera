#!/usr/bin/env python3
"""
システム制限調査スクリプト
Killedの真の原因を特定
"""

import os
import subprocess
import resource
import psutil
import time
from pathlib import Path

def check_system_limits():
    """システム制限の詳細調査"""
    
    print("=== システム制限調査 ===")
    
    # 1. ulimit制限確認
    print("\n📊 ulimit制限:")
    try:
        result = subprocess.run(['bash', '-c', 'ulimit -a'], 
                              capture_output=True, text=True)
        print(result.stdout)
    except Exception as e:
        print(f"ulimit確認エラー: {e}")
    
    # 2. プロセス制限確認
    print("\n🔒 プロセスリソース制限:")
    limits = [
        ('RLIMIT_AS', 'アドレス空間'),
        ('RLIMIT_DATA', 'データセグメント'),
        ('RLIMIT_STACK', 'スタックサイズ'),
        ('RLIMIT_RSS', '物理メモリ'),
        ('RLIMIT_NPROC', 'プロセス数'),
        ('RLIMIT_NOFILE', 'ファイルディスクリプタ'),
        ('RLIMIT_MEMLOCK', 'ロックメモリ'),
        ('RLIMIT_VMEM', '仮想メモリ'),
        ('RLIMIT_CPU', 'CPU時間')
    ]
    
    for limit_name, description in limits:
        if hasattr(resource, limit_name):
            try:
                soft, hard = resource.getrlimit(getattr(resource, limit_name))
                soft_str = f"{soft/1024/1024:.1f}MB" if soft != resource.RLIM_INFINITY and soft > 1024*1024 else str(soft)
                hard_str = f"{hard/1024/1024:.1f}MB" if hard != resource.RLIM_INFINITY and hard > 1024*1024 else str(hard)
                print(f"  {description}: soft={soft_str}, hard={hard_str}")
            except Exception as e:
                print(f"  {description}: 取得エラー - {e}")
    
    # 3. ディスク容量確認
    print("\n💾 ディスク容量:")
    try:
        disk_usage = psutil.disk_usage('/')
        print(f"  合計: {disk_usage.total/1024**3:.1f}GB")
        print(f"  使用済み: {disk_usage.used/1024**3:.1f}GB")
        print(f"  利用可能: {disk_usage.free/1024**3:.1f}GB")
        print(f"  使用率: {disk_usage.used/disk_usage.total*100:.1f}%")
    except Exception as e:
        print(f"ディスク容量確認エラー: {e}")
    
    # 4. スワップ確認
    print("\n🔄 スワップ:")
    try:
        swap = psutil.swap_memory()
        print(f"  合計: {swap.total/1024**3:.1f}GB")
        print(f"  使用済み: {swap.used/1024**3:.1f}GB")
        print(f"  利用可能: {swap.free/1024**3:.1f}GB")
        print(f"  使用率: {swap.percent:.1f}%")
    except Exception as e:
        print(f"スワップ確認エラー: {e}")
    
    # 5. systemd制限確認
    print("\n⚙️ systemd制限:")
    try:
        result = subprocess.run(['systemctl', 'show', '--property=MemoryLimit', '--user'], 
                              capture_output=True, text=True)
        print(f"  MemoryLimit: {result.stdout.strip()}")
        
        result = subprocess.run(['systemctl', 'show', '--property=CPUQuota', '--user'], 
                              capture_output=True, text=True)
        print(f"  CPUQuota: {result.stdout.strip()}")
    except Exception as e:
        print(f"systemd制限確認エラー: {e}")
    
    # 6. OOM killer履歴確認
    print("\n💀 OOM Killer履歴:")
    try:
        result = subprocess.run(['dmesg', '|', 'grep', '-i', 'killed'], 
                              shell=True, capture_output=True, text=True)
        if result.stdout:
            print("  最近のOOM Killer履歴:")
            lines = result.stdout.strip().split('\n')[-5:]  # 最新5件
            for line in lines:
                print(f"    {line}")
        else:
            print("  OOM Killer履歴なし")
    except Exception as e:
        print(f"OOM履歴確認エラー: {e}")
    
    # 7. プロセス情報
    print("\n🔍 現在のプロセス情報:")
    try:
        process = psutil.Process()
        print(f"  PID: {process.pid}")
        print(f"  親PID: {process.ppid()}")
        print(f"  メモリ使用量: {process.memory_info().rss/1024**3:.2f}GB")
        print(f"  仮想メモリ: {process.memory_info().vms/1024**3:.2f}GB")
        print(f"  オープンファイル数: {len(process.open_files())}")
        print(f"  スレッド数: {process.num_threads()}")
    except Exception as e:
        print(f"プロセス情報確認エラー: {e}")

def test_polars_limits():
    """Polars固有の制限テスト"""
    
    print("\n=== Polars制限テスト ===")
    
    import polars as pl
    
    # 1. 大きなLazyFrame作成テスト
    print("\n🧪 大きなLazyFrame作成テスト:")
    try:
        input_path = "/workspaces/project_forge/data/1_XAUUSD_base_data/XAUUSD_tick_master_GOLDEN/timeframe=tick/*.parquet"
        
        print("  LazyFrame作成中...")
        lazy_df = pl.scan_parquet(input_path)
        print("  ✅ LazyFrame作成成功")
        
        print("  スキーマ取得中...")
        schema = lazy_df.collect_schema()
        print(f"  ✅ スキーマ取得成功: {len(schema)}列")
        
        print("  slice操作テスト中...")
        slice_df = lazy_df.slice(0, 1000)
        print("  ✅ slice操作成功")
        
        print("  head操作テスト中...")
        head_df = slice_df.head(10)
        print("  ✅ head操作成功")
        
        print("  小規模collect操作テスト中...")
        result = head_df.collect()
        print(f"  ✅ collect操作成功: {result.shape}")
        
    except Exception as e:
        print(f"  ❌ Polarsテストエラー: {e}")

def alternative_approach_test():
    """代替アプローチのテスト"""
    
    print("\n=== 代替アプローチテスト ===")
    
    # PyArrow直接使用テスト
    print("\n🏹 PyArrow直接テスト:")
    try:
        import pyarrow.parquet as pq
        input_path = "/workspaces/project_forge/data/1_XAUUSD_base_data/XAUUSD_tick_master_GOLDEN/timeframe=tick"
        
        # パーケットファイル一覧
        parquet_files = list(Path(input_path).glob("*.parquet"))
        print(f"  パーケットファイル数: {len(parquet_files)}")
        
        if parquet_files:
            # 最初のファイルのメタデータのみ読み込み
            table_meta = pq.read_metadata(str(parquet_files[0]))
            print(f"  ✅ メタデータ読み込み成功")
            print(f"  行数: {table_meta.num_rows:,}")
            print(f"  列数: {table_meta.schema.names}")
        
    except Exception as e:
        print(f"  ❌ PyArrowテストエラー: {e}")

def main():
    """メイン実行"""
    
    print("🔍 Killedエラー原因調査開始")
    print("="*50)
    
    # システム制限確認
    check_system_limits()
    
    # Polars制限テスト  
    test_polars_limits()
    
    # 代替アプローチテスト
    alternative_approach_test()
    
    print("\n" + "="*50)
    print("🎯 推奨対処法:")
    print("1. ulimit -v unlimited  # 仮想メモリ制限解除")
    print("2. ulimit -m unlimited  # 物理メモリ制限解除") 
    print("3. 別の分割アプローチ (PyArrow直接使用)")
    print("4. Polarsバージョン確認・更新")

if __name__ == "__main__":
    main()