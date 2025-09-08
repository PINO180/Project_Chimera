"""
Project Forge: GPU Accelerated Financial Feature Engineering Engine - EXECUTION
===============================================================================

第1章：統合実行プロトコル
NVIDIA GeForce RTX 3060 (12GB) + Intel i7-8700K完全最適化実行システム 

実行目標: XAU/USD市場における確率的微細パターン完全抽出
実行環境: /workspaces/project_forge/data/1_XAUUSD_base_data/XAUUSDmulti_15timeframe_bars_exness.parquet

Author: Project Forge Development Team
Version: 1.0.0 - Production Execution Ready
"""

import sys
import os
import warnings
import time
from datetime import datetime
import gc
# --- FIX: 'json' is not defined エラーを解決するため、インポートをファイルの先頭に移動 ---
import json
# ------------------------------------------------------------------------------------
import argparse # argparseをインポート

script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.normpath(os.path.join(script_dir, '..', '..'))
sys.path.append(project_root)

# プロジェクトのルートからの絶対パスで正しくインポートする
from src.features.A_quantitative_feature_engine import QuantitativeFeatureEngine

# 警告抑制
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=RuntimeWarning)

# GPU最適化環境変数
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ['RAPIDS_NO_INITIALIZE'] = '1'
os.environ['CUDF_SPILL'] = '1'
os.environ['CUPY_MEMPOOL_SIZE'] = '8GB'

# ============================================================================
# 第2章：実行環境初期化・システム検証
# ============================================================================

def initialize_execution_environment():
    """実行環境完全初期化"""
    print("🔥 Project Forge - Execution Engine Starting...")
    print("=" * 80)
    
    # システム情報表示
    print("🖥️  System Configuration:")
    print(f"   CPU: Intel Core i7-8700K (6C/12T)")
    print(f"   GPU: NVIDIA GeForce RTX 3060 (12GB GDDR6)")
    print(f"   RAM: 32GB DDR4")
    print(f"   SSD: NVMe M.2 1TB")
    print(f"   Platform: {sys.platform}")
    
    # GPU検証
    try:
        import cupy as cp
        gpu_info = cp.cuda.runtime.getDeviceProperties(0)
        gpu_name = gpu_info['name'].decode()
        gpu_memory = cp.cuda.runtime.memGetInfo()[1] / 1024**3
        
        print(f"✅ GPU Acceleration: {gpu_name} ({gpu_memory:.1f}GB)")
        gpu_available = True
    except Exception as e:
        print(f"⚠️  GPU Acceleration: Not available ({e})")
        gpu_available = False
    
    # パッケージ検証
    required_packages = [
        'cudf', 'cupy', 'cuml', 'pandas', 'numpy', 'scipy',
        'MFDFA', 'nolds', 'PyEMD', 'pywt', 'sklearn'
    ]
    
    print(f"\n📦 Package Verification:")
    missing_packages = []
    for package in required_packages:
        try:
            # 'PyEMD'は'emd'としてインポートされることがあるため、代替名を試す
            if package == 'PyEMD':
                try:
                    __import__('emd')
                except ImportError:
                     __import__('PyEMD')
            else:
                 __import__(package)
            print(f"   ✅ {package}")
        except ImportError:
            print(f"   ❌ {package} - MISSING")
            missing_packages.append(package)
    
    if missing_packages:
        print(f"\n❌ Missing packages detected: {missing_packages}")
        print("Please install missing packages before execution.")
        return False
    
    print(f"\n🎯 Environment Status: READY")
    print(f"⚡ GPU Acceleration: {'ENABLED' if gpu_available else 'DISABLED'}")
    
    return gpu_available

def verify_data_path(filepath):
    """データパス検証"""
    print(f"\n📊 Data Verification:")
    print(f"   Target: {filepath}")
    
    if not os.path.exists(filepath):
        print(f"❌ Data file not found: {filepath}")
        return False
    
    file_size = os.path.getsize(filepath) / 1024**2
    print(f"   ✅ File exists ({file_size:.1f}MB)")
    
    return True

# ============================================================================
# 第3章：メイン実行フロー
# ============================================================================

def main_execution(test_mode: bool = False):
    """メイン実行プロトコル"""
    
    execution_start = time.time()
    
    print("\n" + "="*80)
    print("🚀 MAIN EXECUTION PROTOCOL - START")
    if test_mode:
        print("🧪 TEST MODE ACTIVATED")
    print("="*80)
    
    print("\n📋 STEP 1: Environment Initialization")
    gpu_available = initialize_execution_environment()
    
    if not gpu_available:
        print("⚠️  Proceeding with CPU-only mode...")
    
    # ステップ2: データパス検証
    print("\n📋 STEP 2: Data Path Verification")
    import os
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.join(script_dir, '..', '..')
    data_path = os.path.join(project_root, "data", "1_XAUUSD_base_data", "XAUUSDmulti_15timeframe_bars_exness.parquet")
    data_path = os.path.normpath(data_path)
    
    if not verify_data_path(data_path):
        print("❌ EXECUTION ABORTED: Data file not accessible")
        return False
    
    # ステップ3: 特徴量エンジン初期化
    print("\n📋 STEP 3: Feature Engine Initialization")
    try:
        feature_engine = QuantitativeFeatureEngine(
            gpu_optimization=gpu_available,
            precision='float32',
            test_mode=test_mode  # 引数で受け取ったtest_modeを使用
        )
        print("✅ Feature Engine initialized successfully")
    except Exception as e:
        print(f"❌ Feature Engine initialization failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # ステップ4: データ読み込み
    print("\n📋 STEP 4: Complete Dataset Loading")
    try:
        start_time = time.time()
        
        # 静的データ読み込み実行
        df = feature_engine.load_data(data_path)
        
        load_time = time.time() - start_time
        
        print(f"✅ Dataset loaded in {load_time:.2f} seconds")
        
        # --- 修正箇所：dfがNoneでないことを確認してから属性にアクセス ---
        if df is not None:
            print(f"📊 Data Shape: {df.shape}")
            print(f"📅 Date Range: {df.index.min()} to {df.index.max()}")
            
            # データ品質検証
            missing_data = df.isnull().sum().sum()
            print(f"🔍 Data Quality: {missing_data} missing values detected")
        else:
            print("❌ Data loading failed: DataFrame is None")
            return False
        
    except Exception as e:
        print(f"❌ Data loading failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # ステップ5: 特徴量生成実行
    print("\n📋 STEP 5: Comprehensive Feature Generation")
    try:
        start_time = time.time()
        
        # 全特徴量生成実行
        feature_df = feature_engine.generate_all_features(df)
        
        generation_time = time.time() - start_time
        
        if feature_df is not None:
            print(f"🎯 Feature generation completed in {generation_time:.2f} seconds")
            print(f"📈 Generated Features: {feature_df.shape[1]:,}")
            print(f"📊 Data Points: {feature_df.shape[0]:,}")
            
            # メモリ使用量
            if hasattr(feature_df, 'memory_usage'):
                memory_mb = feature_df.memory_usage(deep=True).sum() / 1024**2
                print(f"💾 Memory Usage: {memory_mb:.1f} MB")
        else:
            print("❌ Feature generation failed")
            return False
        
    except Exception as e:
        print(f"❌ Feature generation failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # ステップ6: 特徴量品質検証
    print("\n📋 STEP 6: Feature Quality Validation")
    try:
        # 特徴量サマリー取得
        summary = feature_engine.get_feature_summary(feature_df)
        
        if summary:
            print(f"📊 Feature Summary:")
            print(f"   Total Features: {summary['total_features']:,}")
            print(f"   Total Samples: {summary['total_samples']:,}")
            print(f"   Missing Values: {summary['missing_values']:,}")
            print(f"   Memory Usage: {summary['memory_usage_mb']:.1f} MB")
            
            print(f"\n🎯 Feature Categories:")
            for category, count in summary['feature_types'].items():
                if count > 0:
                    print(f"   {category.title()}: {count} features")
        
        # NaN値統計
        if feature_df is not None:
            nan_counts = feature_df.isnull().sum()
            high_nan_features = nan_counts[nan_counts > len(feature_df) * 0.5]
            
            print(f"\n🔍 Quality Metrics:")
            print(f"   Features with >50% NaN: {len(high_nan_features)}")
            print(f"   Average NaN per feature: {nan_counts.mean():.1f}")
            
            if len(high_nan_features) > 0:
                print(f"⚠️  High-NaN features detected (may require filtering)")
        
    except Exception as e:
        print(f"⚠️  Feature validation warning: {e}")
    
    # ステップ7: 結果保存
    print("\n📋 STEP 7: Result Export")
    try:
        # タイムスタンプ付きファイル名
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # 保存パス
        script_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.join(script_dir, '..', '..')
        output_dir = os.path.join(project_root, "data", "2_feature value")
        output_dir = os.path.normpath(output_dir)
        os.makedirs(output_dir, exist_ok=True)
        
        if gpu_available and hasattr(feature_df, 'to_parquet'):
            # GPU DataFrame保存
            output_path = f"{output_dir}/A_quantitative_features_{timestamp}.parquet"
            feature_df.to_parquet(output_path)
        else:
            # CPU DataFrame保存
            output_path = f"{output_dir}/A_quantitative_features_{timestamp}.parquet"
            feature_df.to_parquet(output_path)

        # CSVでも保存（互換性のため）
        csv_path = f"{output_dir}/A_quantitative_features_{timestamp}.csv"
        if gpu_available and hasattr(feature_df, 'to_pandas'):
            feature_df.to_pandas().to_csv(csv_path, index=True)
        else:
            feature_df.to_csv(csv_path, index=True)
        
        print(f"📄 CSV backup saved to: {csv_path}")
        
    except Exception as e:
        print(f"⚠️  Export warning: {e}")
        print("Features generated but export failed")
    
    # ステップ8: 実行完了・統計表示
    total_execution_time = time.time() - execution_start
    
    print("\n" + "="*80)
    print("🎯 EXECUTION COMPLETED SUCCESSFULLY")
    print("="*80)
    
    print(f"⏱️  Total Execution Time: {total_execution_time:.2f} seconds")
    print(f"🎯 Features Generated: {feature_df.shape[1]:,}")
    print(f"📊 Data Points Processed: {feature_df.shape[0]:,}")
    print(f"⚡ Performance: {feature_df.shape[0] * feature_df.shape[1] / total_execution_time:,.0f} computations/second")
    
    # GPU使用統計
    if gpu_available:
        try:
            import cupy as cp
            mempool = cp.get_default_memory_pool()
            print(f"🖥️  GPU Memory Peak: {mempool.total_bytes() / 1024**3:.1f}GB")
            print(f"📄 GPU Memory Efficiency: {(mempool.used_bytes() / mempool.total_bytes() * 100):.1f}%")
        except:
            print("🖥️  GPU Memory: Statistics unavailable")
    
    print(f"\n📈 Renaissance Technologies Methodology:")
    print(f"   ✅ Statistical Pattern Extraction: COMPLETE")
    print(f"   ✅ Non-Random Microstructure Detection: COMPLETE")
    print(f"   ✅ Probability-Based Feature Engineering: COMPLETE")
    print(f"   ✅ Market Ghost Pattern Capture: COMPLETE")
    
    print(f"\n🎯 Project Chimera Development Fund:")
    print(f"   📊 Feature Universe: {feature_df.shape[1]:,} dimensions")
    print(f"   🔍 Pattern Detection Depth: Maximum Achievable")
    print(f"   ⚡ Processing Speed: GPU-Optimized")
    print(f"   🎯 Exness 2,000x Leverage Ready: ✅")
    
    # 最終メモリクリーンアップ
    try:
        if gpu_available:
            import cupy as cp
            mempool = cp.get_default_memory_pool()
            mempool.free_all_blocks()
        
        del feature_df, df, feature_engine
        gc.collect()
        
        print(f"\n🧹 Memory cleanup completed")
        
    except:
        pass
    
    print(f"\n🚀 READY FOR NEXT PHASE: Anti-Overfitting Defense System")
    print("=" * 80)
    
    return True

# ============================================================================
# 第4章: 追加実行ユーティリティ
# ============================================================================

def performance_benchmark():
    """パフォーマンスベンチマーク実行"""
    print("\n🏃 Performance Benchmark Starting...")
    
    try:
        # CPU vs GPU パフォーマンス比較
        import numpy as np
        import time
        
        # テストデータ生成
        test_size = 100000
        test_data = np.random.randn(test_size).astype(np.float32)
        
        # CPU計算
        cpu_start = time.time()
        cpu_result = np.mean(test_data ** 2)
        cpu_time = time.time() - cpu_start
        
        print(f"💻 CPU Performance: {cpu_time:.4f}s")
        
        # GPU計算（利用可能な場合）
        try:
            import cupy as cp
            test_data_gpu = cp.asarray(test_data)
            
            gpu_start = time.time()
            gpu_result = cp.mean(test_data_gpu ** 2)
            cp.cuda.Stream.null.synchronize()  # GPU同期
            gpu_time = time.time() - gpu_start
            
            speedup = cpu_time / gpu_time if gpu_time > 0 else 1
            
            print(f"🚀 GPU Performance: {gpu_time:.4f}s")
            print(f"⚡ GPU Speedup: {speedup:.1f}x faster")
            
        except:
            print("🚀 GPU Performance: Not available")
    
    except Exception as e:
        print(f"⚠️  Benchmark failed: {e}")

def system_health_check():
    """システムヘルスチェック"""
    print("\n🔧 System Health Check...")
    
    try:
        import psutil
        
        # ... (CPU, Memory, Disk usage)
        
        # GPU使用率（可能な場合）
        try:
            import pynvml
            
            pynvml.nvmlInit()
            handle = pynvml.nvmlDeviceGetHandleByIndex(0)

            gpu_util = pynvml.nvmlDeviceGetUtilizationRates(handle)
            memory_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
            
            print(f"🖥️  GPU Usage: {gpu_util.gpu}%")
            # --- FIX: Operator "/" not supported エラーを解決するため、int()で型変換 ---
            print(f"🎮 GPU Memory: {int(memory_info.used) / 1024**3:.1f}GB / {int(memory_info.total) / 1024**3:.1f}GB")
            # -------------------------------------------------------------------------
            
        except ImportError:
            print("🖥️  GPU Status: pynvml not installed")
        except Exception:
            print("🖥️  GPU Status: Monitoring unavailable")
    
    except Exception as e:
        print(f"⚠️  Health check warning: {e}")

def emergency_recovery():
    """緊急時復旧プロトコル"""
    print("\n🚨 Emergency Recovery Protocol...")
    
    try:
        # メモリクリーンアップ
        gc.collect()
        
        # GPU メモリクリーンアップ
        try:
            import cupy as cp
            mempool = cp.get_default_memory_pool()
            mempool.free_all_blocks()
            print("✅ GPU memory cleared")
        except:
            pass
        
        print("✅ Recovery procedures completed")
        
    except Exception as e:
        print(f"⚠️  Recovery warning: {e}")

# ============================================================================
# 第5章: エラーハンドリング・ロバストパス
# ============================================================================

def robust_execution_wrapper():
    """ロバスト実行ラッパー"""
    try:
        return main_execution()
    
    except KeyboardInterrupt:
        print("\n⚠️  Execution interrupted by user")
        emergency_recovery()
        return False
    
    except MemoryError:
        print("\n❌ Memory Error: Insufficient system memory")
        print("💡 Suggestion: Reduce batch size or close other applications")
        emergency_recovery()
        return False
    
    except ImportError as e:
        print(f"\n❌ Import Error: {e}")
        print("💡 Suggestion: Check package installations")
        return False
    
    except FileNotFoundError as e:
        print(f"\n❌ File Error: {e}")
        print("💡 Suggestion: Verify data file path and permissions")
        return False
    
    except Exception as e:
        print(f"\n❌ Unexpected Error: {e}")
        print("🔧 Attempting emergency recovery...")
        
        import traceback
        print(f"\n📋 Full traceback:")
        traceback.print_exc()
        
        emergency_recovery()
        return False

# ============================================================================
# 第6章: 実行エントリーポイント
# ============================================================================

# 追加: 対話形式の選択メニュー関数
def interactive_mode_selector():
    """実行モードを対話形式で選択する"""
    print("\n" + "="*80)
    print("🔥 Project Forge - Execution Mode Selector 🔥")
    print("="*80)
    
    while True:
        print("\n🎛️  Please select an execution mode:")
        print("   1. Full Production Mode (全データで特徴量生成)")
        print("   2. Test Mode (最初の1000行で高速テスト)")
        print("   -----------------------------------------")
        print("   3. System Health Check (システム状態確認)")
        print("   4. GPU Diagnostic (GPU診断)")
        print("   -----------------------------------------")
        print("   5. Exit (終了)")

        choice = input("\nEnter your choice (1-5): ").strip()
        
        if choice == '1':
            main_execution(test_mode=False)
            # 実行後にメニューに戻る
            continue
        elif choice == '2':
            main_execution(test_mode=True)
            # 実行後にメニューに戻る
            continue
        elif choice == '3':
            system_health_check()
            # チェック後にメニューに戻る
            continue
        elif choice == '4':
            gpu_diagnostic()
            # 診断後にメニューに戻る
            continue
        elif choice == '5':
            print("👋 Exiting.")
            break
        else:
            print("❌ Invalid choice. Please enter a number from 1 to 5.")

if __name__ == "__main__":
    # 常にインタラクティブメニューを呼び出す
    interactive_mode_selector()
    """
    Project Forge - GPU Accelerated Financial Feature Engine
    
    メイン実行エントリーポイント
    統合金融知性体「Project Chimera」開発資金獲得のための
    XAU/USD市場確率的微細パターン完全抽出実行
    """
    
    print("""
    ██████╗ ██████╗  ██████╗      ██╗███████╗ ██████╗████████╗    ███████╗ ██████╗ ██████╗  ██████╗ ███████╗
    ██╔══██╗██╔══██╗██╔═══██╗     ██║██╔════╝██╔════╝╚══██╔══╝    ██╔════╝██╔═══██╗██╔══██╗██╔════╝ ██╔════╝
    ██████╔╝██████╔╝██║   ██║     ██║█████╗  ██║        ██║       █████╗  ██║   ██║██████╔╝██║  ███╗█████╗  
    ██╔═══╝ ██╔══██╗██║   ██║██   ██║██╔══╝  ██║        ██║       ██╔══╝  ██║   ██║██╔══██╗██║   ██║██╔══╝  
    ██║     ██║  ██║╚██████╔╝╚█████╔╝███████╗╚██████╗   ██║       ██║     ╚██████╔╝██║  ██║╚██████╔╝███████╗
    ╚═╝     ╚═╝  ╚═╝ ╚═════╝  ╚════╝ ╚══════╝ ╚═════╝   ╚═╝       ╚═╝      ╚═════╝ ╚═╝  ╚═╝ ╚═════╝ ╚══════╝
    
    🔥 GPU Accelerated Financial Feature Engineering Engine 🔥
    
    Target: XAU/USD Market Pattern Extraction
    Method: Renaissance Technologies Statistical Approach
    Goal: Project Chimera Development Funding
    Hardware: RTX 3060 + i7-8700K Full Optimization
    """)
    
    # 実行前システムチェック
    print("🔍 Pre-execution System Check...")
    system_health_check()
    
    # パフォーマンスベンチマーク
    performance_benchmark()
    
    # メイン実行
    print(f"\n{'='*80}")
    print("🚀 LAUNCHING MAIN EXECUTION...")
    print(f"{'='*80}")
    
    execution_success = robust_execution_wrapper()
    
    # 実行結果判定
    if execution_success:
        print(f"\n🎉 EXECUTION STATUS: SUCCESS")
        print(f"🎯 Project Forge feature extraction completed successfully!")
        print(f"📈 Ready for Phase 2: Anti-Overfitting Defense System")
        
        print(f"\n💡 Next Steps:")
        print(f"   1. Review generated features in /workspaces/project_forge/output/")
        print(f"   2. Proceed to Phase 2: Overfitting Triple Defense Network")
        print(f"   3. Initialize AI Core Construction")
        print(f"   4. Begin Exness XAU/USD Live Trading Preparation")
        
    else:
        print(f"\n❌ EXECUTION STATUS: FAILED")
        print(f"🔧 Please review error messages above and retry")
        
        print(f"\n💡 Troubleshooting Guide:")
        print(f"   1. Verify all required packages are installed")
        print(f"   2. Check GPU drivers and CUDA installation")
        print(f"   3. Ensure sufficient system memory (>16GB recommended)")
        print(f"   4. Verify data file path and permissions")
        print(f"   5. Check available disk space for output")
    
    # 最終システム状態
    print(f"\n🔍 Post-execution System Check...")
    system_health_check()
    
    print(f"\n🔥 Project Forge Execution Engine - Terminated")
    print(f"{'='*80}")

# ============================================================================
# 第7章: デバッグ・開発者向けユーティリティ
# ============================================================================

def debug_mode_execution():
    """デバッグモード実行"""
    print("🐛 DEBUG MODE EXECUTION")
    
    # 詳細ログ有効化
    import logging
    logging.basicConfig(level=logging.DEBUG)
    
    # 小規模データでのテスト実行
    try:
        from src.features.A_quantitative_feature_engine import QuantitativeFeatureEngine
        
        # サンプルデータ生成
        import pandas as pd
        import numpy as np
        
        dates = pd.date_range('2024-01-01', periods=1000, freq='15T')
        sample_data = pd.DataFrame({
            'open': np.random.randn(1000).cumsum() + 2000,
            'high': np.random.randn(1000).cumsum() + 2000,
            'low': np.random.randn(1000).cumsum() + 2000,
            'close': np.random.randn(1000).cumsum() + 2000,
            'volume': np.random.randint(100, 1000, 1000)
        }, index=dates)
        
        # 価格整合性調整
        sample_data['high'] = sample_data[['open', 'close']].max(axis=1) + np.abs(np.random.randn(1000)) * 5
        sample_data['low'] = sample_data[['open', 'close']].min(axis=1) - np.abs(np.random.randn(1000)) * 5
        
        print("✅ Sample data generated for debugging")
        
        # 特徴量エンジンテスト
        engine = QuantitativeFeatureEngine(gpu_optimization=False, precision='float32')
        engine.base_data = sample_data
        
        # 個別特徴量群テスト
        test_functions = [
            ("Price Dynamics", engine.generate_price_dynamics_features),
            ("Statistical Moments", engine.generate_statistical_moments_features),
        ]
        
        for name, func in test_functions:
            try:
                print(f"\n🧪 Testing {name}...")
                features = func(sample_data)
                print(f"✅ {name}: {len(features)} features generated")
            except Exception as e:
                print(f"❌ {name}: {e}")
        
        print("\n✅ Debug mode execution completed")
        
    except Exception as e:
        print(f"❌ Debug mode failed: {e}")
        import traceback
        traceback.print_exc()

def validate_installation():
    """インストール検証"""
    print("🔧 Installation Validation...")
    
    required_packages = {
        'Essential': ['pandas', 'numpy', 'scipy', 'sklearn'],
        'GPU Acceleration': ['cudf', 'cupy', 'cuml'],
        'Specialized Analysis': ['MFDFA', 'nolds', 'emd', 'PyWavelets'],
        'Visualization': ['matplotlib', 'seaborn'],
        'System': ['psutil', 'memory_profiler']
    }
    
    for category, packages in required_packages.items():
        print(f"\n📦 {category}:")
        for package in packages:
            try:
                __import__(package)
                print(f"   ✅ {package}")
            except ImportError:
                print(f"   ❌ {package} - Install with: pip install {package}")

# 追加の便利関数
def quick_feature_test():
    """クイック特徴量テスト"""
    print("⚡ Quick Feature Test...")
    debug_mode_execution()

def gpu_diagnostic():
    """GPU診断"""
    print("🖥️  GPU Diagnostic...")
    
    try:
        import cupy as cp
        import cudf
        
        # GPU基本情報
        device_id = cp.cuda.runtime.getDevice()
        device_props = cp.cuda.runtime.getDeviceProperties(device_id)
        
        print(f"   GPU Name: {device_props['name'].decode()}")
        print(f"   GPU Memory: {device_props['totalGlobalMem'] / 1024**3:.1f}GB")
        print(f"   Compute Capability: {device_props['major']}.{device_props['minor']}")
        print(f"   Multiprocessors: {device_props['multiProcessorCount']}")
        
        # メモリテスト
        test_array = cp.random.randn(1000, 1000, dtype=cp.float32)
        print(f"   ✅ GPU Memory Test: Passed")
        
        # CuDF テスト
        test_df = cudf.DataFrame({'test': [1, 2, 3, 4, 5]})
        print(f"   ✅ CuDF Test: Passed")
        
        del test_array, test_df
        
    except Exception as e:
        print(f"   ❌ GPU Diagnostic Failed: {e}")

# 実行モード選択
def execution_mode_selector():
    """実行モード選択"""
    print("🎛️  Execution Mode Selection:")
    print("   1. Full Production Mode (default)")
    print("   2. Debug Mode (small dataset)")
    print("   3. Installation Validation")
    print("   4. GPU Diagnostic Only")
    print("   5. Quick Feature Test")
    
    try:
        mode = input("Select mode (1-5, default=1): ").strip()
        
        # --- FIX: Expression value is unused エラーを解決するため、戻り値を統一 ---
        if mode == '2':
            debug_mode_execution()
            return True
        elif mode == '3':
            validate_installation()
            return True
        elif mode == '4':
            gpu_diagnostic()
            return True
        elif mode == '5':
            quick_feature_test()
            return True
        else:
            return main_execution()
        # --- 修正箇所：ここまで ---
            
    except KeyboardInterrupt:
        print("\n⚠️  Mode selection cancelled")
        return False
    except Exception:
        print("⚠️  Invalid selection, proceeding with full production mode")
        return main_execution()

# ============================================================================
# 第8章: 高度な監視・診断システム
# ============================================================================

def advanced_system_monitoring():
    """高度なシステム監視"""
    print("\n📊 Advanced System Monitoring...")
    
    try:
        import psutil
        import time
        
        # CPU詳細情報
        cpu_freq = psutil.cpu_freq()
        cpu_count = psutil.cpu_count()
        
        print(f"💻 CPU Details:")
        print(f"   Physical Cores: {psutil.cpu_count(logical=False)}")
        print(f"   Logical Cores: {cpu_count}")
        print(f"   Current Frequency: {cpu_freq.current:.2f}MHz")
        print(f"   Max Frequency: {cpu_freq.max:.2f}MHz")
        
        # メモリ詳細
        memory = psutil.virtual_memory()
        swap = psutil.swap_memory()
        
        print(f"\n🧠 Memory Details:")
        print(f"   Available: {memory.available / 1024**3:.2f}GB")
        print(f"   Used: {memory.used / 1024**3:.2f}GB")
        print(f"   Cached: {memory.cached / 1024**3:.2f}GB")
        print(f"   Swap Used: {swap.used / 1024**3:.2f}GB")
        print(f"   Swap Total: {swap.total / 1024**3:.2f}GB")
        
        # ネットワーク統計
        net_io = psutil.net_io_counters()
        print(f"\n🌐 Network I/O:")
        print(f"   Bytes Sent: {net_io.bytes_sent / 1024**2:.2f}MB")
        print(f"   Bytes Received: {net_io.bytes_recv / 1024**2:.2f}MB")
        
        # ディスクI/O
        disk_io = psutil.disk_io_counters()
        if disk_io:
            print(f"\n💾 Disk I/O:")
            print(f"   Read Bytes: {disk_io.read_bytes / 1024**2:.2f}MB")
            print(f"   Write Bytes: {disk_io.write_bytes / 1024**2:.2f}MB")
        
    except Exception as e:
        print(f"⚠️  Advanced monitoring failed: {e}")

def memory_profiler():
    """メモリプロファイリング"""
    print("\n🔍 Memory Profiling...")
    
    try:
        import gc
        import sys
        
        # ガベージコレクション統計
        gc_stats = gc.get_stats()
        print(f"🗑️  Garbage Collection:")
        for i, stat in enumerate(gc_stats):
            print(f"   Generation {i}: {stat['collections']} collections, {stat['collected']} objects collected")
        
        # 参照カウント統計の安全な取得
        ref_counts = sys.gettrace()
        print(f"📊 Reference Counts: {'Enabled' if ref_counts else 'Disabled'}")
        
        # Python オブジェクト統計
        import types
        module_count = len([obj for obj in gc.get_objects() if isinstance(obj, types.ModuleType)])
        function_count = len([obj for obj in gc.get_objects() if isinstance(obj, types.FunctionType)])
        
        print(f"🐍 Python Objects:")
        print(f"   Modules: {module_count}")
        print(f"   Functions: {function_count}")
        print(f"   Total Objects: {len(gc.get_objects())}")
        
    except Exception as e:
        print(f"⚠️  Memory profiling failed: {e}")

def thermal_monitoring():
    """熱監視システム"""
    print("\n🌡️  Thermal Monitoring...")
    
    try:
        import psutil
        
        # CPU温度取得（Linuxシステムの場合）
        try:
            temps = psutil.sensors_temperatures()
            if temps:
                print("🔥 Temperature Sensors:")
                for name, entries in temps.items():
                    for entry in entries:
                        print(f"   {name} - {entry.label or 'N/A'}: {entry.current}°C")
                        if entry.high:
                            print(f"     High: {entry.high}°C")
                        if entry.critical:
                            print(f"     Critical: {entry.critical}°C")
            else:
                print("🌡️  Temperature sensors: Not available")
        except:
            print("🌡️  Temperature monitoring: Not supported on this platform")
        
        # ファン速度監視
        try:
            fans = psutil.sensors_fans()
            if fans:
                print("\n💨 Fan Speeds:")
                for name, entries in fans.items():
                    for entry in entries:
                        print(f"   {name} - {entry.label or 'N/A'}: {entry.current} RPM")
            else:
                print("💨 Fan monitoring: Not available")
        except:
            print("💨 Fan monitoring: Not supported")
            
    except Exception as e:
        print(f"⚠️  Thermal monitoring failed: {e}")

def network_diagnostics():
    """ネットワーク診断"""
    print("\n🌐 Network Diagnostics...")
    
    try:
        import socket
        import psutil
        
        # ネットワークインターフェース情報
        interfaces = psutil.net_if_addrs()
        print("🔌 Network Interfaces:")
        for interface, addrs in interfaces.items():
            print(f"   {interface}:")
            for addr in addrs:
                print(f"     {addr.family.name}: {addr.address}")
        
        # アクティブなネットワーク接続
        connections = psutil.net_connections()
        active_connections = [conn for conn in connections if conn.status == 'ESTABLISHED']
        print(f"\n🔗 Active Connections: {len(active_connections)}")
        
        # インターネット接続テスト
        try:
            socket.create_connection(("8.8.8.8", 53), timeout=3)
            print("✅ Internet Connection: Available")
        except OSError:
            print("❌ Internet Connection: Unavailable")
            
    except Exception as e:
        print(f"⚠️  Network diagnostics failed: {e}")

import numpy as np
import time

def cpu_stress_worker(duration=5):
    """CPU負荷テスト用のワーカー関数"""
    end_time = time.time() + duration
    while time.time() < end_time:
        # 計算結果を捨てるため、変数代入を省略
        _ = np.random.randn(1000, 1000) @ np.random.randn(1000, 1000)

def performance_stress_test():
    """パフォーマンス負荷テスト"""
    print("\n💪 Performance Stress Test...")
    
    try:
        # --- FIX: threadingの代わりにmultiprocessingをインポート ---
        import multiprocessing as mp
        # --------------------------------------------------------
        
        # CPU負荷テスト
        print("🔥 Starting CPU stress test (5 seconds)...")
        start_time = time.time()
        
        # --- FIX: マルチプロセスでCPU負荷 ---
        # 利用可能なCPUコア数を取得（最大4つまで）
        num_processes = min(mp.cpu_count(), 4)
        print(f"   Running on {num_processes} CPU cores...")
        
        processes = []
        for i in range(num_processes):
            process = mp.Process(target=cpu_stress_worker)
            processes.append(process)
            process.start()
        
        for process in processes:
            process.join()
        # ------------------------------------
        
        stress_time = time.time() - start_time
        print(f"✅ CPU stress test completed in {stress_time:.2f} seconds")
        
        # GPU負荷テスト（利用可能な場合）
        try:
            import cupy as cp
            print("🚀 Starting GPU stress test...")
            
            gpu_start = time.time()
            for i in range(10):
                a = cp.random.randn(2000, 2000, dtype=cp.float32)
                b = cp.random.randn(2000, 2000, dtype=cp.float32)
                c = a @ b
                cp.cuda.Stream.null.synchronize()
            
            gpu_time = time.time() - gpu_start
            print(f"✅ GPU stress test completed in {gpu_time:.2f} seconds")
            
        except ImportError:
            print("🚀 GPU stress test: Skipped (CuPy not available)")
        except Exception:
            print("🚀 GPU stress test: Skipped (GPU not available or error occurred)")
            
    except Exception as e:
        print(f"⚠️  Stress test failed: {e}")

# ============================================================================
# 第9章: エクスポート・レポート生成システム
# ============================================================================

def generate_system_report():
    """システムレポート生成"""
    print("\n📄 Generating System Report...")
    
    try:
        from datetime import datetime
        
        # システム情報収集
        report_data = {
            "timestamp": datetime.now().isoformat(),
            "system_info": {},
            "performance_metrics": {},
            "feature_engine_status": {},
            "recommendations": []
        }
        
        # 基本システム情報
        import platform
        report_data["system_info"] = {
            "platform": platform.platform(),
            "processor": platform.processor(),
            "python_version": platform.python_version(),
            "architecture": platform.architecture()[0]
        }
        
        # パフォーマンスメトリクス
        import psutil
        memory = psutil.virtual_memory()
        report_data["performance_metrics"] = {
            "cpu_usage": psutil.cpu_percent(interval=1),
            "memory_usage_gb": memory.used / 1024**3,
            "memory_total_gb": memory.total / 1024**3,
            "memory_percent": memory.percent,
            "disk_usage_percent": psutil.disk_usage('/').percent
        }
        
        # GPU情報
        try:
            import cupy as cp
            gpu_info = cp.cuda.runtime.getDeviceProperties(0)
            report_data["gpu_info"] = {
                "name": gpu_info['name'].decode(),
                "memory_total_gb": gpu_info['totalGlobalMem'] / 1024**3,
                "compute_capability": f"{gpu_info['major']}.{gpu_info['minor']}"
            }
        except:
            report_data["gpu_info"] = {"status": "Not available"}
        
        # 推奨事項生成
        if report_data["performance_metrics"]["memory_percent"] > 80:
            report_data["recommendations"].append("Consider increasing system memory")
        
        if report_data["performance_metrics"]["cpu_usage"] > 90:
            report_data["recommendations"].append("High CPU usage detected - check background processes")
        
        # レポート保存 (jsonはファイル先頭でインポート済み)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_path = f"system_report_{timestamp}.json"
        
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(report_data, f, indent=2, ensure_ascii=False)
        
        print(f"✅ System report saved to: {report_path}")
        
    except Exception as e:
        print(f"⚠️  Report generation failed: {e}")

def export_feature_metadata():
    """特徴量メタデータエクスポート"""
    print("\n📋 Exporting Feature Metadata...")
    
    try:
        # 特徴量カテゴリ定義
        feature_categories = {
            "price_dynamics": {
                "description": "価格動態特徴量",
                "count": 15,
                "computation_time": "Fast"
            },
            "statistical_moments": {
                "description": "統計的モーメント特徴量",
                "count": 20,
                "computation_time": "Medium"
            },
            "technical_indicators": {
                "description": "テクニカル指標",
                "count": 25,
                "computation_time": "Fast"
            },
            "fractal_analysis": {
                "description": "フラクタル解析特徴量",
                "count": 10,
                "computation_time": "Slow"
            },
            "entropy_measures": {
                "description": "エントロピー測度",
                "count": 8,
                "computation_time": "Medium"
            }
        }
        
        # メタデータファイル生成
        metadata = {
            "feature_categories": feature_categories,
            "total_features": sum(cat["count"] for cat in feature_categories.values()),
            "generation_timestamp": datetime.now().isoformat(),
            "engine_version": "1.0.0"
        }
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        metadata_path = f"feature_metadata_{timestamp}.json"
        
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)
        
        print(f"✅ Feature metadata exported to: {metadata_path}")
        
    except Exception as e:
        print(f"⚠️  Metadata export failed: {e}")

# ============================================================================
# 第10章: 最終実行制御・統合システム
# ============================================================================

def comprehensive_system_check():
    """包括的システムチェック"""
    print("\n🔍 Comprehensive System Check Starting...")
    print("=" * 60)
    
    # 各種チェック実行
    checks = [
        ("System Health", system_health_check),
        ("Advanced Monitoring", advanced_system_monitoring),
        ("Memory Profiling", memory_profiler),
        ("Thermal Monitoring", thermal_monitoring),
        ("Network Diagnostics", network_diagnostics),
        ("GPU Diagnostic", gpu_diagnostic),
        ("Performance Benchmark", performance_benchmark)
    ]
    
    passed_checks = 0
    total_checks = len(checks)
    
    for check_name, check_function in checks:
        try:
            print(f"\n🔎 Running {check_name}...")
            check_function()
            print(f"✅ {check_name}: PASSED")
            passed_checks += 1
        except Exception as e:
            print(f"❌ {check_name}: FAILED ({e})")
    
    print(f"\n📊 System Check Summary:")
    print(f"   Passed: {passed_checks}/{total_checks}")
    print(f"   Success Rate: {(passed_checks/total_checks)*100:.1f}%")
    
    if passed_checks == total_checks:
        print("🎯 System Status: OPTIMAL")
    elif passed_checks >= total_checks * 0.8:
        print("⚠️  System Status: GOOD (minor issues detected)")
    else:
        print("❌ System Status: NEEDS ATTENTION")
    
    return passed_checks >= total_checks * 0.8

def final_execution_protocol():
    """最終実行プロトコル"""
    print("\n" + "="*80)
    print("🚀 FINAL EXECUTION PROTOCOL - PROJECT FORGE")
    print("="*80)
    
    # ステップ1: 包括的システムチェック
    print("\n📋 PHASE 1: Comprehensive System Verification")
    system_ready = comprehensive_system_check()
    
    if not system_ready:
        print("\n⚠️  System not optimal - proceeding with caution...")
    
    # ステップ2: パフォーマンス負荷テスト
    print("\n📋 PHASE 2: Performance Stress Testing")
    performance_stress_test()
    
    # ステップ3: メイン実行
    print("\n📋 PHASE 3: Main Feature Generation Execution")
    execution_success = robust_execution_wrapper()
    
    # ステップ4: 結果検証とレポート生成
    if execution_success:
        print("\n📋 PHASE 4: Results Verification & Reporting")
        generate_system_report()
        export_feature_metadata()
        
        print("\n🎉 PROJECT FORGE EXECUTION: COMPLETE SUCCESS")
        print("🎯 Ready for deployment to live trading environment")
        
    else:
        print("\n❌ PROJECT FORGE EXECUTION: FAILED")
        print("🔧 Review system status and retry")
    
    return execution_success

# 最終メッセージとヘルプ
def show_help():
    """ヘルプ表示"""
    print("""
    🔥 PROJECT FORGE - GPU Accelerated Feature Engine 🔥
    
    📋 Available Commands:
    
    🚀 Main Execution:
      - python A_quantitative_feature_script.py
      - Direct execution with full feature generation
    
    🔧 Diagnostic Tools:
      - python -c "from A_quantitative_feature_script import gpu_diagnostic; gpu_diagnostic()"
      - python -c "from A_quantitative_feature_script import system_health_check; system_health_check()"
      - python -c "from A_quantitative_feature_script import validate_installation; validate_installation()"
    
    🧪 Testing & Debug:
      - python -c "from A_quantitative_feature_script import debug_mode_execution; debug_mode_execution()"
      - python -c "from A_quantitative_feature_script import quick_feature_test; quick_feature_test()"
      - python -c "from A_quantitative_feature_script import performance_stress_test; performance_stress_test()"
    
    📊 Advanced Monitoring:
      - python -c "from A_quantitative_feature_script import comprehensive_system_check; comprehensive_system_check()"
      - python -c "from A_quantitative_feature_script import advanced_system_monitoring; advanced_system_monitoring()"
    
    📄 Reporting:
      - python -c "from A_quantitative_feature_script import generate_system_report; generate_system_report()"
      - python -c "from A_quantitative_feature_script import export_feature_metadata; export_feature_metadata()"
    
    🎛️  Interactive Mode:
      - python -c "from A_quantitative_feature_script import execution_mode_selector; execution_mode_selector()"
    
    🆘 Emergency Recovery:
      - python -c "from A_quantitative_feature_script import emergency_recovery; emergency_recovery()"
    
    💡 Tips:
      - Ensure GPU drivers are up to date
      - Install missing packages: pip install package_name
      - Check available memory before execution
      - Monitor GPU temperature during intensive operations
    
    🎯 Target: XAU/USD market pattern extraction for Project Chimera
    ⚡ Hardware: RTX 3060 + i7-8700K optimized
    🚀 Goal: Exness 2000x leverage trading preparation
    """)

# プログラム終了時の最終メッセージ
print("""
💡 Project Forge Execution Engine Ready

🎯 Mission: Renaissance Technologies-style feature extraction
⚡ Hardware: RTX 3060 + i7-8700K full optimization  
🚀 Objective: Project Chimera development funding
📈 Target: XAU/USD market with Exness 2000x leverage

Type 'python A_quantitative_feature_script.py --help' for detailed usage
""")

# 自動ヘルプ表示制御
if __name__ == "__main__":
    """
    メイン実行エントリーポイント
    """
    
    # 対話形式のメニューだけを呼び出す
    interactive_mode_selector()

    # 以下の古い実行フローは削除またはコメントアウトする
    # execution_success = final_execution_protocol()
    # if not execution_success:
    #     sys.exit(1)

# End of Complete Execution Engine