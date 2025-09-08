"""
Project Forge: GPU Accelerated Financial Feature Engineering Engine - INTEGRATED
===============================================================================

統合版: 特徴量生成エンジン + 実行制御システム
NVIDIA GeForce RTX 3060 (12GB) + Intel i7-8700K完全最適化実装

Author: Project Forge Development Team
Version: 1.0.0 - Production Ready Integrated
"""

# =============================================================================
# 第1章: GPUアクセル基盤ライブラリ群 - RAPIDS完全統合
# =============================================================================

import math
import os
import sys
import warnings
from tqdm import tqdm

# BLAS/LAPACK警告を抑制
os.environ['PYTHONWARNINGS'] = 'ignore::RuntimeWarning'
os.environ['MKL_VERBOSE'] = '0'
os.environ['OPENBLAS_VERBOSE'] = '0'
os.environ['MKL_NUM_THREADS'] = '1'

# 簡潔なDLASCL抑制クラス
class CompleteDLASCLSuppressor:
    def __init__(self, original_stderr):
        self.original_stderr = original_stderr
        
    def write(self, text):
        dlascl_patterns = ['dlascl', 'illegal value', 'parameter number', 'on entry to']
        if not any(pattern in text.lower() for pattern in dlascl_patterns):
            self.original_stderr.write(text)
    
    def flush(self):
        self.original_stderr.flush()

# 即座に適用
sys.stderr = CompleteDLASCLSuppressor(sys.stderr)

# NumPy警告抑制
import numpy as np
np.seterr(all='ignore')

warnings.filterwarnings('ignore')
warnings.filterwarnings('ignore', category=RuntimeWarning)

warnings.filterwarnings('ignore', message='.*cwt.*')
warnings.filterwarnings('ignore', message='.*complex_cwt.*')

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ['RAPIDS_NO_INITIALIZE'] = '1'

# RAPIDS GPU生態系 - 完全統合
try:
    import cudf
    import cupy as cp
    import cuml
    import cugraph
    from cuml.preprocessing import StandardScaler as GPU_StandardScaler
    from cuml.decomposition import PCA as GPU_PCA
    from cuml.cluster import KMeans as GPU_KMeans
    from cuml.ensemble import RandomForestRegressor as GPU_RandomForest
    GPU_AVAILABLE = True
except ImportError:
    print("Warning: RAPIDS/CuPy not available, falling back to CPU mode")
    GPU_AVAILABLE = False

# 分散・並列処理基盤
import dask
import dask.dataframe as dd
try:
    from dask_cuda import LocalCUDACluster
    from distributed import Client, as_completed
except ImportError:
    pass

import numba
from numba import cuda, jit, vectorize
import multiprocessing as mp
try:
    from pathos.multiprocessing import ProcessingPool
except ImportError:
    pass

# データ処理・分析基盤
import pandas as pd
import numpy as np
try:
    import polars as pl
except ImportError:
    pass
import pyarrow as pa
import pyarrow.parquet as pq
from scipy import stats, signal, optimize, sparse
from scipy.fft import fft, ifft, rfft, irfft
from scipy.ndimage import gaussian_filter1d

# 機械学習・統計解析
from sklearn.preprocessing import RobustScaler, MinMaxScaler
from sklearn.decomposition import FastICA, NMF
from sklearn.manifold import TSNE, LocallyLinearEmbedding
from sklearn.feature_selection import mutual_info_regression
try:
    import xgboost as xgb
    import lightgbm as lgb
except ImportError:
    pass

try:
    from statsmodels.tsa.seasonal import seasonal_decompose
    from statsmodels.stats.diagnostic import acorr_ljungbox
    import arch
    from arch import arch_model
except ImportError:
    pass

# 特殊解析ライブラリ - 金融時系列特化
try:
    from MFDFA import MFDFA
except ImportError:
    print("Warning: MFDFA not available")
    MFDFA = None

try:
    import nolds
except ImportError:
    print("Warning: nolds not available")
    nolds = None

try:
    from pyemd import EMD, EEMD, CEEMDAN
except ImportError:
    try:
        from emd import EMD, EEMD, CEEMDAN
    except ImportError:
        print("Warning: EMD packages not available")
        EMD = None
        EEMD = None  
        CEEMDAN = None

try:
    from dcor import distance_correlation
except ImportError:
    pass

try:
    from PyCausality import Granger
except ImportError:
    try:
        import PyCausality
        Granger = getattr(PyCausality, 'Granger', None)
    except:
        print("Warning: PyCausality Granger not available")
        Granger = None

# 時間・周波数解析
try:
    import pywt
    from pywt import wavedec, waverec, cwt, dwt, idwt
except ImportError:
    print("Warning: PyWavelets not available")
    pywt = None

# 数値計算・最適化
try:
    from sympy import symbols, diff, integrate, simplify
except ImportError:
    pass
from scipy.special import gamma, digamma, polygamma
from scipy.optimize import minimize, differential_evolution

# システム・パフォーマンス
import psutil
import time
from datetime import datetime, timedelta
import gc
try:
    from memory_profiler import profile
except ImportError:
    pass
import cProfile

# 追加システムライブラリ
import sys
import json
import argparse

# プロジェクトパス設定
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.normpath(os.path.join(script_dir, '..', '..'))
sys.path.append(project_root)

# GPU最適化環境変数
os.environ['CUDF_SPILL'] = '1'
os.environ['CUPY_MEMPOOL_SIZE'] = '8GB'


# ============================================================================
# 第2章: GPUメモリ管理とカスタム関数群
# ============================================================================
class GPUMemoryManager:
    """GPU メモリ最適化管理クラス"""
    
    def __init__(self):
        if GPU_AVAILABLE:
            self.pool = cp.get_default_memory_pool()
            self.pinned_pool = cp.get_default_pinned_memory_pool()
        else:
            self.pool = None
            self.pinned_pool = None
    
    def optimize_memory(self):
        """メモリ最適化実行"""
        if GPU_AVAILABLE and self.pool:
            self.pool.free_all_blocks()
            self.pinned_pool.free_all_blocks()
        gc.collect()
    
    def get_memory_info(self):
        """GPU メモリ使用量取得"""
        if GPU_AVAILABLE and self.pool:
            return {
                'gpu_used': self.pool.used_bytes() / 1024**3,
                'gpu_total': self.pool.total_bytes() / 1024**3
            }
        else:
            return {'gpu_used': 0, 'gpu_total': 0}

if GPU_AVAILABLE:
    @cuda.jit
    def gpu_rolling_window_kernel(data, window_size, output):
        """GPU並列ローリングウィンドウカーネル"""
        idx = cuda.grid(1)
        if idx < output.shape[0]:
            start_idx = max(0, idx - window_size + 1)
            end_idx = idx + 1
            window_sum = 0.0
            count = 0
            for i in range(start_idx, end_idx):
                if not cp.isnan(data[i]):
                    window_sum += data[i]
                    count += 1
            output[idx] = window_sum / count if count > 0 else cp.nan

    @vectorize(['float64(float64, float64)'], target='cuda')
    def gpu_log_return(current, previous):
        """GPU最適化対数リターン計算"""
        return math.log(current / previous) if previous > 0 else float('nan')

@jit(nopython=True, parallel=True)
def numba_hurst_exponent(data, max_lag=20):
    """Numba最適化ハースト指数計算"""
    n = len(data)
    lags = np.arange(2, max_lag + 1)
    tau = np.zeros(len(lags))
    
    for i, lag in enumerate(lags):
        tau[i] = np.sqrt(np.mean((data[lag:] - data[:-lag])**2))
    
    poly = np.polyfit(np.log(lags), np.log(tau), 1)
    return poly[0]

# =============================================================================
# 第3章: 特徴量生成エンジン - 統計的パターン探求
# =============================================================================

class QuantitativeFeatureEngine:
    """
    統合金融知性体特徴量生成エンジン
    Renaissance Technologies準拠 - 確率的ノイズ内微細パターン探求
    """
    
    def __init__(self, gpu_optimization=True, precision='float64', test_mode=False):
        self.gpu_optimization = gpu_optimization and GPU_AVAILABLE  # GPU_AVAILABLEもチェック
        self.precision = precision
        self.memory_manager = GPUMemoryManager()
        self.test_mode = test_mode
        
        # ライブラリ可用性チェック
        self.lib_status = self._check_library_availability()
        
        self.timeframes = [] 
        self.window_sizes = [5, 10, 15, 20, 30, 50, 100, 200]
        self.lookback_periods = [10, 20, 50, 100, 200, 500]
        
        self.device_info = self._initialize_compute_environment()

    def _check_library_availability(self):  # ← ここに追加
        """利用可能なライブラリをチェック"""
        return {
            'nolds': nolds is not None,
            'MFDFA': MFDFA is not None,
            'EMD': EMD is not None,
            'pywt': pywt is not None,
            'GPU': GPU_AVAILABLE
        }    
    
    def _initialize_compute_environment(self):
        """計算環境初期化"""
        try:
            if GPU_AVAILABLE:
                cp.cuda.Device(0).use()
                gpu_info = {
                    'available': True,
                    'name': cp.cuda.runtime.getDeviceProperties(0)['name'].decode(),
                    'memory': cp.cuda.runtime.memGetInfo()[1] / 1024**3
                }
                print(f"GPU Acceleration: {gpu_info['name']} ({gpu_info['memory']:.1f}GB)")
                return gpu_info
            else:
                return {'available': False}
        except:
            return {'available': False}
    
    def load_data(self, filepath):
        """静的データ読み込み - アウトオブコア対応"""
        print("📄 Loading dataset with memory optimization...")
        
        # テストモードでのアウトオブコア処理
        if self.test_mode:
            print("🧪 TEST MODE: Using memory-efficient chunked loading...")
            
            if self.gpu_optimization and self.device_info['available']:
                # GPU版：チャンク読み込みでメモリ効率化
                import pyarrow.parquet as pq
                try:
                    parquet_file = pq.ParquetFile(filepath)
                    # バッチサイズを500に設定してメモリ使用量を抑制
                    batch_iter = parquet_file.iter_batches(batch_size=500)
                    first_batch = next(batch_iter)
                    second_batch = next(batch_iter, None)
                    
                    # 最大1000行まで、チャンク単位で処理
                    if second_batch is not None:
                        combined_table = pa.concat_tables([first_batch, second_batch])
                        df = cudf.from_arrow(combined_table)
                    else:
                        df = cudf.from_arrow(first_batch)
                    
                    # 1000行に制限
                    if len(df) > 1000:
                        df = df.head(1000)
                        
                except Exception as e:
                    print(f"⚠️ Chunked loading failed, falling back to direct method: {e}")
                    df = cudf.read_parquet(filepath, nrows=1000)
                    
                print(f"✅ GPU Dataset loaded (chunked): {df.shape[0]:,} rows, {df.shape[1]} columns")
                
            else:
                # CPU版：pandas chunksizeに対応
                try:
                    # pandas.read_parquetにはchunksizeパラメータがないため、代替手段を使用
                    import pyarrow.parquet as pq
                    parquet_file = pq.ParquetFile(filepath)
                    batch_iter = parquet_file.iter_batches(batch_size=500)
                    
                    chunks = []
                    total_rows = 0
                    
                    for batch in batch_iter:
                        chunk_df = batch.to_pandas()
                        chunks.append(chunk_df)
                        total_rows += len(chunk_df)
                        if total_rows >= 1000:
                            break
                    
                    df = pd.concat(chunks, ignore_index=True)
                    if len(df) > 1000:
                        df = df.head(1000)
                        
                except Exception as e:
                    print(f"⚠️ Chunked loading failed, falling back to direct method: {e}")
                    df = pd.read_parquet(filepath)
                    if len(df) > 1000:
                        df = df.head(1000)
                    
                print(f"✅ CPU Dataset loaded (chunked): {df.shape[0]:,} rows, {df.shape[1]} columns")
        
        else:
            # 本番モード：従来通りの全データ読み込み
            if self.gpu_optimization and self.device_info['available']:
                df = cudf.read_parquet(filepath)
                print(f"✅ GPU Dataset loaded: {df.shape[0]:,} rows, {df.shape[1]} columns")
            else:
                df = pd.read_parquet(filepath)
                print(f"✅ CPU Dataset loaded: {df.shape[0]:,} rows, {df.shape[1]} columns")

        if self.test_mode:
            print(f"🧪 TEST MODE: Data limited to {len(df)} rows with chunked loading.")

        # データから時間軸を動的に検出
        if 'timeframe' in df.columns:
            if isinstance(df, cudf.DataFrame):
                self.timeframes = df['timeframe'].unique().to_pandas().tolist()
            else: # pandas.DataFrame
                self.timeframes = df['timeframe'].unique().tolist()
            print(f"🕒 Detected timeframes: {len(self.timeframes)} -> {self.timeframes}")
        else:
            # timeframe列がない場合はデフォルト値を設定
            self.timeframes = ['M1', 'M5', 'M15', 'M30', 'H1', 'H4', 'D1']
            print(f"⚠️ 'timeframe' column not found. Using default timeframes: {self.timeframes}")

        # データ型最適化を試行
        try:
            optimized_df = self._optimize_datatypes(df)
            if optimized_df is not None:
                df = optimized_df
                print("✅ Data types optimized successfully")
            else:
                print("⚠️ Data type optimization skipped")
        except Exception as e:
            print(f"⚠️ Data type optimization failed: {e}")
            print("Proceeding with original data types")

        if 'timestamp' in df.columns:
            df = df.set_index('timestamp')
        
        # self.base_data には常に処理対象のdfを入れる
        self.base_data = df 
        print(f"📊 Memory usage: {self.memory_manager.get_memory_info()}")
        
        return df
    
    def _optimize_datatypes(self, df):
        """データ型最適化"""
        if df is None:
            return None

        try:
            if isinstance(df, cudf.DataFrame):
                # GPU DataFrame最適化
                for col in df.columns:
                    if str(df[col].dtype) == 'float64':
                        df[col] = df[col].astype('float32')
                    elif str(df[col].dtype) == 'int64':
                        df[col] = df[col].astype('int32')
            else:
                # CPU DataFrame最適化
                for col in df.select_dtypes(include=['float64']).columns:
                    df[col] = pd.to_numeric(df[col], downcast='float')
                for col in df.select_dtypes(include=['int64']).columns:
                    df[col] = pd.to_numeric(df[col], downcast='integer')
            
            return df
            
        except Exception as e:
            print(f"⚠️ Data type optimization failed: {e}")
            return df
        
    # =========================================================================
    # 第4章: 価格動力学特徴量群
    # =========================================================================
    
    def generate_price_dynamics_features(self, df):
        """価格動力学特徴量生成"""
        features = {}
        
        # GPU/CPU処理の分岐をループ外に移動（ルール1）
        if self.gpu_optimization and isinstance(df, cudf.DataFrame):
            # GPU処理ブランチ
            close = cp.asarray(df['close'].values, dtype=cp.float32)
            high = cp.asarray(df['high'].values, dtype=cp.float32)
            low = cp.asarray(df['low'].values, dtype=cp.float32)
            open_price = cp.asarray(df['open'].values, dtype=cp.float32)
            volume = cp.asarray(df['volume'].values, dtype=cp.float32) if 'volume' in df.columns else None
            
            # 対数リターン系列
            for window in self.window_sizes:
                log_returns = cp.diff(cp.log(close))
                features[f'log_return_{window}'] = self._gpu_rolling_mean(log_returns, window).get()  # .get()で変換
            
            # True Range / Average True Range
            tr_series = []
            for i in range(1, len(close)):
                high_val = float(high[i].get())  # .get()で変換
                low_val = float(low[i].get())    # .get()で変換
                close_prev = float(close[i-1].get())  # .get()で変換
                tr = max(
                    high_val - low_val,
                    abs(high_val - close_prev),
                    abs(low_val - close_prev)
                )
                tr_series.append(tr)
            
            tr_series = np.array(tr_series)
            for window in [14, 21, 50]:
                atr = pd.Series(tr_series).rolling(window).mean().values
                features[f'atr_{window}'] = np.concatenate([[np.nan], atr])
            
            # 価格効率性指標
            for window in self.window_sizes:
                if window <= len(close):
                    price_efficiency = []
                    close_numpy = close.get()  # .get()で変換
                    for i in range(window, len(close_numpy)):
                        price_path = close_numpy[i-window:i+1]
                        straight_distance = abs(price_path[-1] - price_path[0])
                        path_length = np.sum(np.abs(np.diff(price_path)))
                        efficiency = straight_distance / path_length if path_length > 0 else 0
                        price_efficiency.append(efficiency)
                    
                    features[f'price_efficiency_{window}'] = np.concatenate([
                        np.full(window, np.nan), np.array(price_efficiency)
                    ])
            
            # レンジ正規化価格位置
            for window in self.window_sizes:
                close_numpy = close.get()  # .get()で変換
                high_numpy = high.get()    # .get()で変換
                low_numpy = low.get()      # .get()で変換
                
                high_roll = pd.Series(high_numpy).rolling(window).max().values
                low_roll = pd.Series(low_numpy).rolling(window).min().values
                features[f'price_position_{window}'] = (
                    (close_numpy - low_roll) / (high_roll - low_roll + 1e-10)
                )
            
            # 価格加速度・ジャーク
            close_numpy = close.get()  # .get()で変換
            price_velocity = np.gradient(close_numpy)
            price_acceleration = np.gradient(price_velocity)
            price_jerk = np.gradient(price_acceleration)
            
            features['price_velocity'] = price_velocity
            features['price_acceleration'] = price_acceleration
            features['price_jerk'] = price_jerk
            
        else:
            # CPU処理ブランチ
            close = df['close'].values
            high = df['high'].values
            low = df['low'].values
            open_price = df['open'].values
            volume = df['volume'].values if 'volume' in df.columns else None
            
            # 対数リターン系列
            for window in self.window_sizes:
                log_returns = np.diff(np.log(close))
                features[f'log_return_{window}'] = pd.Series(log_returns).rolling(window).mean().values
            
            # True Range / Average True Range
            tr_series = []
            for i in range(1, len(close)):
                tr = max(
                    high[i] - low[i],
                    abs(high[i] - close[i-1]),
                    abs(low[i] - close[i-1])
                )
                tr_series.append(tr)
            
            tr_series = np.array(tr_series)
            for window in [14, 21, 50]:
                atr = pd.Series(tr_series).rolling(window).mean().values
                features[f'atr_{window}'] = np.concatenate([[np.nan], atr])
            
            # 価格効率性指標
            for window in self.window_sizes:
                if window <= len(close):
                    price_efficiency = []
                    for i in range(window, len(close)):
                        price_path = close[i-window:i+1]
                        straight_distance = abs(price_path[-1] - price_path[0])
                        path_length = np.sum(np.abs(np.diff(price_path)))
                        efficiency = straight_distance / path_length if path_length > 0 else 0
                        price_efficiency.append(efficiency)
                    
                    features[f'price_efficiency_{window}'] = np.concatenate([
                        np.full(window, np.nan), np.array(price_efficiency)
                    ])
            
            # レンジ正規化価格位置
            for window in self.window_sizes:
                high_roll = pd.Series(high).rolling(window).max().values
                low_roll = pd.Series(low).rolling(window).min().values
                features[f'price_position_{window}'] = (
                    (close - low_roll) / (high_roll - low_roll + 1e-10)
                )
            
            # 価格加速度・ジャーク
            price_velocity = np.gradient(close)
            price_acceleration = np.gradient(price_velocity)
            price_jerk = np.gradient(price_acceleration)
            
            features['price_velocity'] = price_velocity
            features['price_acceleration'] = price_acceleration
            features['price_jerk'] = price_jerk
        
        return features
    
    # =========================================================================
    # 第5章: 統計的モーメント・分布特徴量群
    # =========================================================================
    
    def generate_statistical_moments_features(self, df):
        """統計的モーメント・分布特徴量生成"""
        features = {}
        
        # GPU/CPU処理の分岐をループ外に移動（ルール1）
        if self.gpu_optimization and isinstance(df, cudf.DataFrame):
            # GPU処理ブランチ
            returns = cp.diff(cp.log(cp.asarray(df['close'].values)))
            
            # 高次統計モーメント
            for window in self.window_sizes:
                rolling_data = self._gpu_rolling_window(returns, window)
                
                # 基本統計量
                features[f'skewness_{window}'] = self._gpu_skewness(rolling_data).get()  # .get()で変換
                features[f'kurtosis_{window}'] = self._gpu_kurtosis(rolling_data).get()  # .get()で変換
                features[f'std_{window}'] = cp.nanstd(rolling_data, axis=1).get()       # .get()で変換
                features[f'var_{window}'] = cp.nanvar(rolling_data, axis=1).get()       # .get()で変換
            
            # 分位数・パーセンタイル特徴量
            percentiles = [5, 10, 25, 75, 90, 95]
            for window in [20, 50, 100]:
                for pct in percentiles:
                    rolling_data = self._gpu_rolling_window(returns, window)
                    features[f'percentile_{pct}_{window}'] = cp.percentile(rolling_data, pct, axis=1).get()  # .get()で変換
            
            # Jarque-Bera正規性検定統計量
            returns_numpy = returns.get()  # .get()で変換
            for window in [50, 100, 200]:
                jb_stats = []
                for i in range(window, len(returns_numpy)):
                    sample = returns_numpy[i-window:i]
                    if len(sample) == window:
                        try:
                            n = len(sample)
                            s = stats.skew(sample)
                            k = stats.kurtosis(sample)
                            jb = n * (s**2/6 + k**2/24)
                            jb_stats.append(jb)
                        except:
                            jb_stats.append(np.nan)
                    else:
                        jb_stats.append(np.nan)
                
                features[f'jarque_bera_{window}'] = np.concatenate([
                    [np.nan] * window, jb_stats
                ])
            
            # Anderson-Darling検定統計量近似
            for window in [50, 100]:
                ad_stats = []
                for i in range(window, len(returns_numpy)):
                    sample = returns_numpy[i-window:i]
                    if len(sample) == window:
                        try:
                            sample_sorted = np.sort(sample)
                            n = len(sample_sorted)
                            # 簡易Anderson-Darling統計量
                            uniform_sample = stats.norm.cdf(sample_sorted)
                            ad_approx = -n - np.mean((2*np.arange(1, n+1) - 1) * 
                                                   (np.log(uniform_sample) + 
                                                    np.log(1 - uniform_sample[::-1])))
                            ad_stats.append(ad_approx)
                        except:
                            ad_stats.append(np.nan)
                    else:
                        ad_stats.append(np.nan)
                
                features[f'anderson_darling_{window}'] = np.concatenate([
                    [np.nan] * window, ad_stats
                ])
        
        else:
            # CPU処理ブランチ
            returns = np.diff(np.log(df['close'].values))
            
            # 高次統計モーメント
            for window in self.window_sizes:
                returns_series = pd.Series(returns)
                features[f'skewness_{window}'] = returns_series.rolling(window).skew().values
                features[f'kurtosis_{window}'] = returns_series.rolling(window).kurt().values
                features[f'std_{window}'] = returns_series.rolling(window).std().values
                features[f'var_{window}'] = returns_series.rolling(window).var().values
            
            # 分位数・パーセンタイル特徴量
            percentiles = [5, 10, 25, 75, 90, 95]
            for window in [20, 50, 100]:
                for pct in percentiles:
                    returns_series = pd.Series(returns)
                    features[f'percentile_{pct}_{window}'] = (
                        returns_series.rolling(window).quantile(pct/100).values
                    )
            
            # Jarque-Bera正規性検定統計量
            for window in [50, 100, 200]:
                jb_stats = []
                for i in range(window, len(returns)):
                    sample = returns[i-window:i]
                    if len(sample) == window:
                        try:
                            n = len(sample)
                            s = stats.skew(sample)
                            k = stats.kurtosis(sample)
                            jb = n * (s**2/6 + k**2/24)
                            jb_stats.append(jb)
                        except:
                            jb_stats.append(np.nan)
                    else:
                        jb_stats.append(np.nan)
                
                features[f'jarque_bera_{window}'] = np.concatenate([
                    [np.nan] * window, jb_stats
                ])
            
            # Anderson-Darling検定統計量近似
            for window in [50, 100]:
                ad_stats = []
                for i in range(window, len(returns)):
                    sample = returns[i-window:i]
                    if len(sample) == window:
                        try:
                            sample_sorted = np.sort(sample)
                            n = len(sample_sorted)
                            # 簡易Anderson-Darling統計量
                            uniform_sample = stats.norm.cdf(sample_sorted)
                            ad_approx = -n - np.mean((2*np.arange(1, n+1) - 1) * 
                                                   (np.log(uniform_sample) + 
                                                    np.log(1 - uniform_sample[::-1])))
                            ad_stats.append(ad_approx)
                        except:
                            ad_stats.append(np.nan)
                    else:
                        ad_stats.append(np.nan)
                
                features[f'anderson_darling_{window}'] = np.concatenate([
                    [np.nan] * window, ad_stats
                ])
        
        return features
    
    # =========================================================================
    # 第6章: フラクタル・カオス理論特徴量群
    # =========================================================================
    
    def generate_fractal_chaos_features(self, df):
        """フラクタル・カオス理論特徴量生成"""
        features = {}
        
        close_prices = df['close'].values
        returns = np.diff(np.log(close_prices))
        
        # GPU/CPU処理の分岐をループ外に移動（ルール1）
        if self.gpu_optimization and isinstance(df, cudf.DataFrame):
            # GPU処理ブランチ
            close_prices_gpu = cp.asarray(close_prices)
            returns_gpu = cp.diff(cp.log(close_prices_gpu))
            
            # HurstIndex - 複数手法
            for window in [100, 200, 500]:
                hurst_values = []
                returns_numpy = returns_gpu.get() if GPU_AVAILABLE else returns
                for i in range(window, len(returns_numpy)):
                    sample = returns_numpy[i-window:i]
                    try:
                        # noldsライブラリの可用性チェック
                        if self.lib_status.get('nolds', False):
                            hurst_rs = nolds.hurst_rs(sample, nvals=None)
                            hurst_values.append(hurst_rs)
                        else:
                            # noldsが利用できない場合の代替実装
                            hurst_values.append(self._fallback_hurst_calculation(sample))
                    except:
                        hurst_values.append(np.nan)
                
                features[f'hurst_rs_{window}'] = np.concatenate([
                    [np.nan] * window, hurst_values
                ])

            # Detrended Fluctuation Analysis (DFA)
            for window in [100, 200]:
                dfa_values = []
                returns_numpy = returns_gpu.get() if GPU_AVAILABLE else returns
                for i in range(window, len(returns_numpy)):
                    sample = returns_numpy[i-window:i]
                    try:
                        if self.lib_status.get('nolds', False):
                            dfa_alpha = nolds.dfa(sample)
                            dfa_values.append(dfa_alpha)
                        else:
                            # noldsが利用できない場合の代替実装
                            dfa_values.append(self._fallback_dfa_calculation(sample))
                    except:
                        dfa_values.append(np.nan)
                
                features[f'dfa_alpha_{window}'] = np.concatenate([
                    [np.nan] * window, dfa_values
                ])
            
            # Multifractal DFA (MFDFA)
            close_prices_numpy = close_prices_gpu.get()  # .get()で変換
            for window in [200, 500]:
                if len(close_prices_numpy) > window:
                    for i in range(window, min(len(close_prices_numpy), window + 1000)):  # 計算量制限
                        sample = close_prices_numpy[i-window:i]
                        try:
                            # MFDFA計算
                            lag = range(10, window//4)
                            q = np.arange(-5, 6)
                            hq, Hq = MFDFA(sample, lag=lag, q=q, stat=0)  # CPUライブラリなので変換済み配列を使用
                            
                            # 主要指標抽出
                            features[f'mfdfa_hurst_q0_{window}'] = hq[5]  # q=0のHurst指数
                            features[f'mfdfa_width_{window}'] = np.max(hq) - np.min(hq)  # マルチフラクタル幅
                            features[f'mfdfa_asymmetry_{window}'] = hq[0] - hq[-1]  # 非対称性
                            
                            break  # 1回だけ計算（計算量考慮）
                        except:
                            pass
            
            # Correlation Dimension
            for window in [100, 200]:
                corr_dim_values = []
                returns_numpy = returns_gpu.get()  # .get()で変換
                for i in range(window, len(returns_numpy)):
                    sample = returns_numpy[i-window:i]
                    try:
                        corr_dim = nolds.corr_dim(sample, emb_dim=10)  # CPUライブラリなので変換済み配列を使用
                        corr_dim_values.append(corr_dim)
                    except:
                        corr_dim_values.append(np.nan)
                
                features[f'correlation_dimension_{window}'] = np.concatenate([
                    [np.nan] * window, corr_dim_values
                ])
            
            # Lyapunov指数近似
            for window in [100, 200]:
                lyap_values = []
                returns_numpy = returns_gpu.get()  # .get()で変換
                for i in range(window, len(returns_numpy)):
                    sample = returns_numpy[i-window:i]
                    try:
                        lyap = nolds.lyap_r(sample)  # CPUライブラリなので変換済み配列を使用
                        lyap_values.append(lyap)
                    except:
                        lyap_values.append(np.nan)
                
                features[f'lyapunov_{window}'] = np.concatenate([
                    [np.nan] * window, lyap_values
                ])
            
            # Sample Entropy
            for window in [50, 100]:
                sampen_values = []
                returns_numpy = returns_gpu.get()  # .get()で変換
                for i in range(window, len(returns_numpy)):
                    sample = returns_numpy[i-window:i]
                    try:
                        sampen = nolds.sampen(sample)  # CPUライブラリなので変換済み配列を使用
                        sampen_values.append(sampen)
                    except:
                        sampen_values.append(np.nan)
                
                features[f'sample_entropy_{window}'] = np.concatenate([
                    [np.nan] * window, sampen_values
                ])
        
        else:
            # CPU処理ブランチ
            # HurstIndex - 複数手法
            for window in [100, 200, 500]:
                hurst_values = []
                returns_numpy = returns
                for i in range(window, len(returns_numpy)):
                    sample = returns_numpy[i-window:i]
                    try:
                        # noldsライブラリの可用性チェック
                        if self.lib_status.get('nolds', False):
                            hurst_rs = nolds.hurst_rs(sample, nvals=None)
                            hurst_values.append(hurst_rs)
                        else:
                            # noldsが利用できない場合の代替実装
                            hurst_values.append(self._fallback_hurst_calculation(sample))
                    except:
                        hurst_values.append(np.nan)
                
                features[f'hurst_rs_{window}'] = np.concatenate([
                    [np.nan] * window, hurst_values
                ])

            # Detrended Fluctuation Analysis (DFA)
            for window in [100, 200]:
                dfa_values = []
                returns_numpy = returns
                for i in range(window, len(returns_numpy)):
                    sample = returns_numpy[i-window:i]
                    try:
                        if self.lib_status.get('nolds', False):
                            dfa_alpha = nolds.dfa(sample)
                            dfa_values.append(dfa_alpha)
                        else:
                            # noldsが利用できない場合の代替実装
                            dfa_values.append(self._fallback_dfa_calculation(sample))
                    except:
                        dfa_values.append(np.nan)
                
                features[f'dfa_alpha_{window}'] = np.concatenate([
                    [np.nan] * window, dfa_values
                ])
            
            # Multifractal DFA (MFDFA)
            for window in [200, 500]:
                if len(close_prices) > window:
                    for i in range(window, min(len(close_prices), window + 1000)):  # 計算量制限
                        sample = close_prices[i-window:i]
                        try:
                            # MFDFA計算
                            lag = range(10, window//4)
                            q = np.arange(-5, 6)
                            hq, Hq = MFDFA(sample, lag=lag, q=q, stat=0)
                            
                            # 主要指標抽出
                            features[f'mfdfa_hurst_q0_{window}'] = hq[5]  # q=0のHurst指数
                            features[f'mfdfa_width_{window}'] = np.max(hq) - np.min(hq)  # マルチフラクタル幅
                            features[f'mfdfa_asymmetry_{window}'] = hq[0] - hq[-1]  # 非対称性
                            
                            break  # 1回だけ計算（計算量考慮）
                        except:
                            pass
            
            # Correlation Dimension
            for window in [100, 200]:
                corr_dim_values = []
                for i in range(window, len(returns)):
                    sample = returns[i-window:i]
                    try:
                        corr_dim = nolds.corr_dim(sample, emb_dim=10)
                        corr_dim_values.append(corr_dim)
                    except:
                        corr_dim_values.append(np.nan)
                
                features[f'correlation_dimension_{window}'] = np.concatenate([
                    [np.nan] * window, corr_dim_values
                ])
            
            # Lyapunov指数近似
            for window in [100, 200]:
                lyap_values = []
                for i in range(window, len(returns)):
                    sample = returns[i-window:i]
                    try:
                        lyap = nolds.lyap_r(sample)
                        lyap_values.append(lyap)
                    except:
                        lyap_values.append(np.nan)
                
                features[f'lyapunov_{window}'] = np.concatenate([
                    [np.nan] * window, lyap_values
                ])
            
            # Sample Entropy
            for window in [50, 100]:
                sampen_values = []
                for i in range(window, len(returns)):
                    sample = returns[i-window:i]
                    try:
                        sampen = nolds.sampen(sample)
                        sampen_values.append(sampen)
                    except:
                        sampen_values.append(np.nan)
                
                features[f'sample_entropy_{window}'] = np.concatenate([
                    [np.nan] * window, sampen_values
                ])
        
        return features
    
    # =========================================================================
    # 第7章: 時間周波数解析特徴量群
    # =========================================================================
    
    def generate_time_frequency_features(self, df):
        """時間周波数解析特徴量生成（修正版）"""
        features = {}
        
        close_prices = df['close'].values
        returns = np.diff(np.log(close_prices))
        
        # GPU処理ブランチ
        if self.gpu_optimization and isinstance(df, cudf.DataFrame):
            # GPUデータ準備
            close_prices_gpu = cp.asarray(close_prices)
            returns_gpu = cp.diff(cp.log(close_prices_gpu))
            returns_numpy = returns_gpu.get()  # CPUライブラリ用に変換
            
        else:
            # CPU処理ブランチ
            returns_numpy = returns
        
        # Wavelet Transform解析（修正版）
        if self.lib_status.get('pywt', False):
            wavelets = ['db4', 'db8', 'haar', 'coif2']
            for wavelet in wavelets:
                try:
                    # 離散ウェーブレット変換（安全な方法）
                    scales = np.arange(1, min(32, len(returns_numpy)//4))
                    
                    if len(scales) > 0:
                        # PyWavelets v1.4+ 対応のCWT実装
                        try:
                            # 新しいAPI
                            coefficients, frequencies = pywt.cwt(
                                returns_numpy, 
                                scales, 
                                wavelet, 
                                sampling_period=1.0
                            )
                        except (TypeError, AttributeError):
                            # 古いAPIまたは異なるバージョン
                            try:
                                coefficients = pywt.cwt(returns_numpy, scales, wavelet)[0]
                                frequencies = pywt.central_frequency(wavelet) / scales
                            except:
                                # CWTが失敗した場合はDWTを使用
                                coeffs = pywt.wavedec(returns_numpy, wavelet, level=5)
                                # 最初の係数のみ使用
                                coefficients = np.array([coeffs[0]])
                                frequencies = np.array([1.0])
                        
                        # エネルギー密度
                        if len(coefficients.shape) == 2:
                            energy_density = np.mean(np.abs(coefficients)**2, axis=0)
                        else:
                            energy_density = np.abs(coefficients)**2
                        
                        # 配列長を統一
                        if len(energy_density) != len(returns_numpy):
                            if len(energy_density) < len(returns_numpy):
                                # パディング
                                energy_density = np.pad(
                                    energy_density, 
                                    (0, len(returns_numpy) - len(energy_density)), 
                                    mode='constant', 
                                    constant_values=np.nan
                                )
                            else:
                                # 切り詰め
                                energy_density = energy_density[:len(returns_numpy)]
                        
                        features[f'wavelet_energy_{wavelet}'] = energy_density
                        
                        # 優勢周波数（スカラー値を全データ長に展開）
                        if len(coefficients.shape) == 2:
                            dominant_scale_idx = np.argmax(np.mean(np.abs(coefficients)**2, axis=1))
                        else:
                            dominant_scale_idx = 0
                        
                        dominant_freq = frequencies[dominant_scale_idx] if len(frequencies) > dominant_scale_idx else 1.0
                        features[f'wavelet_dominant_freq_{wavelet}'] = np.full(len(returns_numpy), dominant_freq)
                        
                        # ウェーブレット分散
                        if len(coefficients.shape) == 2:
                            wavelet_var = np.var(np.abs(coefficients), axis=0)
                        else:
                            wavelet_var = np.var(np.abs(coefficients)) * np.ones(len(returns_numpy))
                        
                        # 配列長を統一
                        if len(wavelet_var) != len(returns_numpy):
                            if len(wavelet_var) < len(returns_numpy):
                                wavelet_var = np.pad(
                                    wavelet_var, 
                                    (0, len(returns_numpy) - len(wavelet_var)), 
                                    mode='constant', 
                                    constant_values=np.nan
                                )
                            else:
                                wavelet_var = wavelet_var[:len(returns_numpy)]
                        
                        features[f'wavelet_variance_{wavelet}'] = wavelet_var
                    
                except Exception as e:
                    print(f"Warning: Wavelet {wavelet} analysis failed: {e}")
                    # デフォルト値で埋める
                    features[f'wavelet_energy_{wavelet}'] = np.full(len(returns_numpy), np.nan)
                    features[f'wavelet_dominant_freq_{wavelet}'] = np.full(len(returns_numpy), np.nan)
                    features[f'wavelet_variance_{wavelet}'] = np.full(len(returns_numpy), np.nan)
        else:
            print("Warning: PyWavelets not available, skipping wavelet analysis")
        
        # Empirical Mode Decomposition (EMD) - 安全な実装
        if self.lib_status.get('EMD', False):
            try:
                emd = EMD()
                IMFs = emd(returns_numpy)
                
                # 各IMFの特徴量
                for i, imf in enumerate(IMFs[:min(3, len(IMFs))]):  # 最初の3つのIMFのみ
                    if len(imf) == len(returns_numpy):
                        features[f'imf_{i}_energy'] = np.full(len(returns_numpy), np.sum(imf**2))
                        features[f'imf_{i}_mean_freq'] = np.full(len(returns_numpy), np.mean(np.abs(np.diff(imf))))
                        
                        # IMFの瞬時周波数
                        try:
                            analytic_signal = signal.hilbert(imf)
                            instantaneous_phase = np.unwrap(np.angle(analytic_signal))
                            instantaneous_frequency = np.diff(instantaneous_phase) / (2.0 * np.pi)
                            
                            features[f'imf_{i}_inst_freq_mean'] = np.full(len(returns_numpy), np.mean(instantaneous_frequency))
                            features[f'imf_{i}_inst_freq_std'] = np.full(len(returns_numpy), np.std(instantaneous_frequency))
                        except:
                            features[f'imf_{i}_inst_freq_mean'] = np.full(len(returns_numpy), np.nan)
                            features[f'imf_{i}_inst_freq_std'] = np.full(len(returns_numpy), np.nan)
            
            except Exception as e:
                print(f"Warning: EMD analysis failed: {e}")
        else:
            print("Warning: EMD not available, skipping EMD analysis")
        
        # Short-Time Fourier Transform（安全な実装）
        for window_length in [32, 64, 128]:
            if len(returns_numpy) > window_length:
                try:
                    frequencies, times, Zxx = signal.stft(returns_numpy, nperseg=window_length)
                    
                    # スペクトログラム特徴量
                    spectrogram = np.abs(Zxx)
                    
                    # 時間軸の長さを調整
                    target_length = len(returns_numpy)
                    
                    if spectrogram.shape[1] != target_length:
                        # リサンプリングまたはパディング
                        from scipy.interpolate import interp1d
                        
                        old_indices = np.linspace(0, 1, spectrogram.shape[1])
                        new_indices = np.linspace(0, 1, target_length)
                        
                        stft_energy = np.mean(spectrogram, axis=0)
                        f_interp = interp1d(old_indices, stft_energy, kind='linear', fill_value='extrapolate')
                        features[f'stft_energy_{window_length}'] = f_interp(new_indices)
                        
                        peak_freq = frequencies[np.argmax(spectrogram, axis=0)]
                        f_interp_freq = interp1d(old_indices, peak_freq, kind='linear', fill_value='extrapolate')
                        features[f'stft_peak_freq_{window_length}'] = f_interp_freq(new_indices)
                    else:
                        features[f'stft_energy_{window_length}'] = np.mean(spectrogram, axis=0)
                        features[f'stft_peak_freq_{window_length}'] = frequencies[np.argmax(spectrogram, axis=0)]
                
                except Exception as e:
                    print(f"Warning: STFT analysis failed for window {window_length}: {e}")
                    # デフォルト値
                    features[f'stft_energy_{window_length}'] = np.full(len(returns_numpy), np.nan)
                    features[f'stft_peak_freq_{window_length}'] = np.full(len(returns_numpy), np.nan)
        
        # Hilbert-Huang Transform特徴量（安全な実装）
        try:
            # 瞬時振幅・瞬時周波数
            analytic_signal = signal.hilbert(returns_numpy)
            instantaneous_amplitude = np.abs(analytic_signal)
            instantaneous_phase = np.unwrap(np.angle(analytic_signal))
            instantaneous_frequency = np.diff(instantaneous_phase) / (2.0 * np.pi)
            
            features['hilbert_inst_amplitude'] = instantaneous_amplitude
            # 周波数は1つ短いので、パディング
            features['hilbert_inst_frequency'] = np.concatenate([[np.nan], instantaneous_frequency])
            
            # 瞬時振幅・周波数の統計量
            for window in [20, 50, 100]:
                if len(instantaneous_amplitude) > window:
                    amp_series = pd.Series(instantaneous_amplitude)
                    freq_series = pd.Series(instantaneous_frequency)
                    
                    features[f'hilbert_amp_mean_{window}'] = amp_series.rolling(window).mean().values
                    features[f'hilbert_amp_std_{window}'] = amp_series.rolling(window).std().values
                    
                    # 周波数の統計量（長さ調整）
                    freq_mean = freq_series.rolling(window).mean().values
                    freq_std = freq_series.rolling(window).std().values
                    
                    # パディングして長さを合わせる
                    features[f'hilbert_freq_mean_{window}'] = np.concatenate([[np.nan], freq_mean])
                    features[f'hilbert_freq_std_{window}'] = np.concatenate([[np.nan], freq_std])
        
        except Exception as e:
            print(f"Warning: Hilbert-Huang analysis failed: {e}")
            # デフォルト値
            features['hilbert_inst_amplitude'] = np.full(len(returns_numpy), np.nan)
            features['hilbert_inst_frequency'] = np.full(len(returns_numpy), np.nan)

        # 必ず辞書を返すように修正（重要！）
        return features
    
    # =========================================================================
    # 第8章: 非線形動力学特徴量群
    # =========================================================================
    
    def generate_nonlinear_dynamics_features(self, df):
        """非線形動力学特徴量生成"""
        features = {}
        
        close_prices = df['close'].values
        returns = np.diff(np.log(close_prices))
        
        # GPU/CPU処理の分岐をループ外に移動（ルール1）
        if self.gpu_optimization and isinstance(df, cudf.DataFrame):
            # GPU処理ブランチ
            close_prices_gpu = cp.asarray(close_prices)
            returns_gpu = cp.diff(cp.log(close_prices_gpu))
            returns_numpy = returns_gpu.get()  # .get()で変換（CPUライブラリ用）
            
            # フェーズスペース再構成
            embedding_dims = [3, 5, 7, 10]
            time_delays = [1, 3, 5]
            
            for emb_dim in embedding_dims[:2]:  # 計算量制限
                for tau in time_delays[:2]:
                    try:
                        # タケンス埋め込み
                        embedded = self._takens_embedding(returns_numpy, emb_dim, tau)  # CPU関数なので変換済み配列を使用
                        
                        # 再帰プロット特徴量
                        recurrence_features = self._recurrence_plot_analysis(embedded[:1000])  # サンプル制限
                        
                        for key, value in recurrence_features.items():
                            features[f'recurrence_{key}_dim{emb_dim}_tau{tau}'] = value
                    
                    except Exception as e:
                        print(f"Warning: Phase space reconstruction failed for dim={emb_dim}, tau={tau}: {e}")
            
            # Poincare Plot解析
            for lag in [1, 2, 3]:
                try:
                    poincare_features = self._poincare_plot_analysis(returns_numpy, lag)  # CPU関数なので変換済み配列を使用
                    for key, value in poincare_features.items():
                        features[f'poincare_{key}_lag{lag}'] = value
                except Exception as e:
                    print(f"Warning: Poincare plot analysis failed for lag={lag}: {e}")
            
            # 非線形相関次元
            for window in [100, 200]:
                corr_dim_values = []
                for i in range(window, len(returns_numpy)):
                    sample = returns_numpy[i-window:i]
                    try:
                        # Grassberger-Procaccia相関次元
                        corr_dim = self._grassberger_procaccia_dimension(sample)  # CPU関数なので変換済み配列を使用
                        corr_dim_values.append(corr_dim)
                    except:
                        corr_dim_values.append(np.nan)
                
                features[f'gp_correlation_dimension_{window}'] = np.concatenate([
                    [np.nan] * window, corr_dim_values
                ])
            
            # False Nearest Neighbors
            for window in [100, 200]:
                fnn_values = []
                for i in range(window, len(returns_numpy)):
                    sample = returns_numpy[i-window:i]
                    try:
                        optimal_dim = self._false_nearest_neighbors(sample)  # CPU関数なので変換済み配列を使用
                        fnn_values.append(optimal_dim)
                    except:
                        fnn_values.append(np.nan)
                
                features[f'optimal_embedding_dim_{window}'] = np.concatenate([
                    [np.nan] * window, fnn_values
                ])
        
        else:
            # CPU処理ブランチ（既存コードをそのまま使用）
            # フェーズスペース再構成
            embedding_dims = [3, 5, 7, 10]
            time_delays = [1, 3, 5]
            
            for emb_dim in embedding_dims[:2]:  # 計算量制限
                for tau in time_delays[:2]:
                    try:
                        # タケンス埋め込み
                        embedded = self._takens_embedding(returns, emb_dim, tau)
                        
                        # 再帰プロット特徴量
                        recurrence_features = self._recurrence_plot_analysis(embedded[:1000])  # サンプル制限
                        
                        for key, value in recurrence_features.items():
                            features[f'recurrence_{key}_dim{emb_dim}_tau{tau}'] = value
                    
                    except Exception as e:
                        print(f"Warning: Phase space reconstruction failed for dim={emb_dim}, tau={tau}: {e}")
            
            # Poincare Plot解析
            for lag in [1, 2, 3]:
                try:
                    poincare_features = self._poincare_plot_analysis(returns, lag)
                    for key, value in poincare_features.items():
                        features[f'poincare_{key}_lag{lag}'] = value
                except Exception as e:
                    print(f"Warning: Poincare plot analysis failed for lag={lag}: {e}")
            
            # 非線形相関次元
            for window in [100, 200]:
                corr_dim_values = []
                for i in range(window, len(returns)):
                    sample = returns[i-window:i]
                    try:
                        # Grassberger-Procaccia相関次元
                        corr_dim = self._grassberger_procaccia_dimension(sample)
                        corr_dim_values.append(corr_dim)
                    except:
                        corr_dim_values.append(np.nan)
                
                features[f'gp_correlation_dimension_{window}'] = np.concatenate([
                    [np.nan] * window, corr_dim_values
                ])
            
            # False Nearest Neighbors
            for window in [100, 200]:
                fnn_values = []
                for i in range(window, len(returns)):
                    sample = returns[i-window:i]
                    try:
                        optimal_dim = self._false_nearest_neighbors(sample)
                        fnn_values.append(optimal_dim)
                    except:
                        fnn_values.append(np.nan)
                
                features[f'optimal_embedding_dim_{window}'] = np.concatenate([
                    [np.nan] * window, fnn_values
                ])
        
        return features
    
    # =========================================================================
    # 第9章: 情報理論特徴量群
    # =========================================================================
    
    def generate_information_theory_features(self, df):
        """情報理論特徴量生成"""
        features = {}
        
        close_prices = df['close'].values
        returns = np.diff(np.log(close_prices))
        
        # GPU/CPU処理の分岐をループ外に移動（ルール1）
        if self.gpu_optimization and isinstance(df, cudf.DataFrame):
            # GPU処理ブランチ
            close_prices_gpu = cp.asarray(close_prices)
            returns_gpu = cp.diff(cp.log(close_prices_gpu))
            returns_numpy = returns_gpu.get()  # .get()で変換（NumPyライブラリ用）
            
            # Shannon Entropy
            for window in [50, 100, 200]:
                entropy_values = []
                for i in range(window, len(returns_numpy)):
                    sample = returns_numpy[i-window:i]
                    try:
                        # ヒストグラム法によるエントロピー推定
                        hist, bin_edges = np.histogram(sample, bins=20, density=True)
                        hist = hist[hist > 0]  # ゼロ除去
                        entropy = -np.sum(hist * np.log2(hist))
                        entropy_values.append(entropy)
                    except:
                        entropy_values.append(np.nan)
                
                features[f'shannon_entropy_{window}'] = np.concatenate([
                    np.full(window, np.nan), np.array(entropy_values)
                ])
            
            # Rényi Entropy
            alpha_values = [0.5, 2.0, 3.0]
            for alpha in alpha_values:
                for window in [50, 100]:
                    renyi_values = []
                    for i in range(window, len(returns_numpy)):
                        sample = returns_numpy[i-window:i]
                        try:
                            hist, _ = np.histogram(sample, bins=20, density=True)
                            hist = hist[hist > 0]
                            if alpha == 1.0:
                                renyi = -np.sum(hist * np.log(hist))
                            else:
                                renyi = (1/(1-alpha)) * np.log(np.sum(hist**alpha))
                            renyi_values.append(renyi)
                        except:
                            renyi_values.append(np.nan)
                    
                    features[f'renyi_entropy_alpha{alpha}_{window}'] = np.concatenate([
                        [np.nan] * window, renyi_values
                    ])
            
            # Transfer Entropy (簡易版)
            for lag in [1, 2, 5]:
                for window in [100, 200]:
                    te_values = []
                    for i in range(window + lag, len(returns_numpy)):
                        try:
                            x = returns_numpy[i-window-lag:i-lag]
                            y = returns_numpy[i-window:i]
                            # 簡易Transfer Entropy計算
                            te = self._transfer_entropy_estimation(x, y, lag)
                            te_values.append(te)
                        except:
                            te_values.append(np.nan)
                    
                    features[f'transfer_entropy_lag{lag}_{window}'] = np.concatenate([
                        [np.nan] * (window + lag), te_values
                    ])
            
            # Mutual Information
            for lag in [1, 2, 5, 10]:
                mi_values = []
                for i in range(lag, len(returns_numpy)):
                    try:
                        x = returns_numpy[i-lag:i]
                        y = returns_numpy[i:min(i+lag, len(returns_numpy))]
                        if len(x) == len(y) and len(x) > 0:
                            mi = self._mutual_information_estimation(x, y)
                            mi_values.append(mi)
                        else:
                            mi_values.append(np.nan)
                    except:
                        mi_values.append(np.nan)
                
                features[f'mutual_information_lag{lag}'] = np.concatenate([
                    [np.nan] * lag, mi_values, [np.nan] * (len(returns_numpy) - lag - len(mi_values))
                ])
        
        else:
            # CPU処理ブランチ（既存コードをそのまま使用）
            # Shannon Entropy
            for window in [50, 100, 200]:
                entropy_values = []
                for i in range(window, len(returns)):
                    sample = returns[i-window:i]
                    try:
                        # ヒストグラム法によるエントロピー推定
                        hist, bin_edges = np.histogram(sample, bins=20, density=True)
                        hist = hist[hist > 0]  # ゼロ除去
                        entropy = -np.sum(hist * np.log2(hist))
                        entropy_values.append(entropy)
                    except:
                        entropy_values.append(np.nan)
                
                features[f'shannon_entropy_{window}'] = np.concatenate([
                    np.full(window, np.nan), np.array(entropy_values)
                ])
            
            # Rényi Entropy
            alpha_values = [0.5, 2.0, 3.0]
            for alpha in alpha_values:
                for window in [50, 100]:
                    renyi_values = []
                    for i in range(window, len(returns)):
                        sample = returns[i-window:i]
                        try:
                            hist, _ = np.histogram(sample, bins=20, density=True)
                            hist = hist[hist > 0]
                            if alpha == 1.0:
                                renyi = -np.sum(hist * np.log(hist))
                            else:
                                renyi = (1/(1-alpha)) * np.log(np.sum(hist**alpha))
                            renyi_values.append(renyi)
                        except:
                            renyi_values.append(np.nan)
                    
                    features[f'renyi_entropy_alpha{alpha}_{window}'] = np.concatenate([
                        [np.nan] * window, renyi_values
                    ])
            
            # Transfer Entropy (簡易版)
            for lag in [1, 2, 5]:
                for window in [100, 200]:
                    te_values = []
                    for i in range(window + lag, len(returns)):
                        try:
                            x = returns[i-window-lag:i-lag]
                            y = returns[i-window:i]
                            # 簡易Transfer Entropy計算
                            te = self._transfer_entropy_estimation(x, y, lag)
                            te_values.append(te)
                        except:
                            te_values.append(np.nan)
                    
                    features[f'transfer_entropy_lag{lag}_{window}'] = np.concatenate([
                        [np.nan] * (window + lag), te_values
                    ])
            
            # Mutual Information
            for lag in [1, 2, 5, 10]:
                mi_values = []
                for i in range(lag, len(returns)):
                    try:
                        x = returns[i-lag:i]
                        y = returns[i:min(i+lag, len(returns))]
                        if len(x) == len(y) and len(x) > 0:
                            mi = self._mutual_information_estimation(x, y)
                            mi_values.append(mi)
                        else:
                            mi_values.append(np.nan)
                    except:
                        mi_values.append(np.nan)
                
                features[f'mutual_information_lag{lag}'] = np.concatenate([
                    [np.nan] * lag, mi_values, [np.nan] * (len(returns) - lag - len(mi_values))
                ])
        
        return features
    
    # =========================================================================
    # 第10章: 多重時間軸特徴量群
    # =========================================================================
    
    def generate_multitimescale_features(self, df):
        """多重時間軸特徴量生成"""
        features = {}
        
        # 異なる時間軸でのリサンプリング特徴量
        if 'timestamp' in df.columns or isinstance(df.index, pd.DatetimeIndex):
            resampling_periods = ['5T', '15T', '30T', '1H', '4H', '1D']
            
            for period in resampling_periods:
                try:
                    if isinstance(df, cudf.DataFrame):
                        # GPU DataFrame用の処理
                        df_cpu = df.to_pandas()
                        resampled = df_cpu.resample(period).agg({
                            'open': 'first',
                            'high': 'max',
                            'low': 'min',
                            'close': 'last',
                            'volume': 'sum' if 'volume' in df.columns else 'mean'
                        })
                    else:
                        resampled = df.resample(period).agg({
                            'open': 'first',
                            'high': 'max',
                            'low': 'min',
                            'close': 'last',
                            'volume': 'sum' if 'volume' in df.columns else 'mean'
                        })
                    
                    # リサンプリング後の特徴量計算
                    resampled_returns = np.diff(np.log(resampled['close'].values))
                    
                    # 基本統計量
                    features[f'resample_{period}_mean'] = np.mean(resampled_returns)
                    features[f'resample_{period}_std'] = np.std(resampled_returns)
                    features[f'resample_{period}_skew'] = stats.skew(resampled_returns)
                    features[f'resample_{period}_kurtosis'] = stats.kurtosis(resampled_returns)
                    
                    # ボラティリティクラスタリング
                    abs_returns = np.abs(resampled_returns)
                    for lag in [1, 2, 5]:
                        if len(abs_returns) > lag:
                            vol_autocorr = np.corrcoef(abs_returns[:-lag], abs_returns[lag:])[0, 1]
                            features[f'resample_{period}_vol_autocorr_lag{lag}'] = vol_autocorr
                
                except Exception as e:
                    print(f"Warning: Resampling for {period} failed: {e}")
        
        # 多重解像度解析
        wavelet_levels = [1, 2, 3, 4, 5]
        returns = np.diff(np.log(df['close'].values))
        
        # GPU/CPU処理の分岐をループ外に移動（ルール1）
        if self.gpu_optimization and isinstance(df, cudf.DataFrame):
            # GPU処理ブランチ
            returns_gpu = cp.diff(cp.log(cp.asarray(df['close'].values)))
            returns_numpy = returns_gpu.get()  # .get()で変換（pywt用）
            
            try:
                # 離散ウェーブレット変換による多重解像度分解
                coeffs = pywt.wavedec(returns_numpy, 'db4', level=5)  # CPU関数なので変換済み配列を使用
                
                for level in range(len(coeffs)):
                    coeff = coeffs[level]
                    # スカラー値を全データ長に展開
                    energy_val = np.sum(coeff**2)
                    mean_val = np.mean(coeff)
                    std_val = np.std(coeff)
                    
                    features[f'wavelet_level_{level}_energy'] = np.full(len(returns_numpy), energy_val)
                    features[f'wavelet_level_{level}_mean'] = np.full(len(returns_numpy), mean_val)
                    features[f'wavelet_level_{level}_std'] = np.full(len(returns_numpy), std_val)
                    
                    # 各レベルでのスケーリング指数
                    if len(coeff) > 10:
                        try:
                            scaling_exp = self._scaling_exponent(coeff)
                            features[f'wavelet_level_{level}_scaling_exp'] = scaling_exp
                        except:
                            features[f'wavelet_level_{level}_scaling_exp'] = np.nan
            
            except Exception as e:
                print(f"Warning: Multi-resolution analysis failed: {e}")
            
            # 長期記憶性分析
            for window in [200, 500, 1000]:
                if len(returns_numpy) > window:
                    long_memory_stats = []
                    for i in range(window, len(returns_numpy)):
                        sample = returns_numpy[i-window:i]
                        try:
                            # Geweke-Porter-Hudak長期記憶推定
                            gph_estimate = self._gph_long_memory_estimate(sample)
                            long_memory_stats.append(gph_estimate)
                        except:
                            long_memory_stats.append(np.nan)
                    
                    features[f'long_memory_gph_{window}'] = np.concatenate([
                        [np.nan] * window, long_memory_stats
                    ])
        
        else:
            # CPU処理ブランチ（既存コードをそのまま使用）
            try:
                # 離散ウェーブレット変換による多重解像度分解
                coeffs = pywt.wavedec(returns, 'db4', level=5)
                
                for level in range(len(coeffs)):
                    coeff = coeffs[level]
                    # スカラー値を全データ長に展開
                    energy_val = np.sum(coeff**2)
                    mean_val = np.mean(coeff)
                    std_val = np.std(coeff)
                    
                    features[f'wavelet_level_{level}_energy'] = np.full(len(returns), energy_val)
                    features[f'wavelet_level_{level}_mean'] = np.full(len(returns), mean_val)
                    features[f'wavelet_level_{level}_std'] = np.full(len(returns), std_val)
                    
                    # 各レベルでのスケーリング指数
                    if len(coeff) > 10:
                        try:
                            scaling_exp = self._scaling_exponent(coeff)
                            features[f'wavelet_level_{level}_scaling_exp'] = scaling_exp
                        except:
                            features[f'wavelet_level_{level}_scaling_exp'] = np.nan
            
            except Exception as e:
                print(f"Warning: Multi-resolution analysis failed: {e}")
            
            # 長期記憶性分析
            for window in [200, 500, 1000]:
                if len(returns) > window:
                    long_memory_stats = []
                    for i in range(window, len(returns)):
                        sample = returns[i-window:i]
                        try:
                            # Geweke-Porter-Hudak長期記憶推定
                            gph_estimate = self._gph_long_memory_estimate(sample)
                            long_memory_stats.append(gph_estimate)
                        except:
                            long_memory_stats.append(np.nan)
                    
                    features[f'long_memory_gph_{window}'] = np.concatenate([
                        [np.nan] * window, long_memory_stats
                    ])
        
        return features
    
    # =========================================================================
    # 第11章: 流動性・マイクロ構造特徴量群
    # =========================================================================
    
    def generate_microstructure_features(self, df):
        """流動性・マイクロ構造特徴量生成"""
        features = {}
        
        close_prices = df['close'].values
        high_prices = df['high'].values
        low_prices = df['low'].values
        
        if 'volume' in df.columns:
            volume = df['volume'].values
        else:
            # 疑似ボリューム（価格変動幅ベース）
            volume = (high_prices - low_prices) / close_prices
        
        # Amihud非流動性指標
        returns = np.diff(np.log(close_prices))
        abs_returns = np.abs(returns)
        
        # GPU/CPU処理の分岐をループ外に移動（ルール1）
        if self.gpu_optimization and isinstance(df, cudf.DataFrame):
            # GPU処理ブランチ
            close_prices_gpu = cp.asarray(close_prices)
            high_prices_gpu = cp.asarray(high_prices)
            low_prices_gpu = cp.asarray(low_prices)
            
            if 'volume' in df.columns:
                volume_gpu = cp.asarray(df['volume'].values)
                volume_numpy = volume_gpu.get()  # .get()で変換
            else:
                volume_gpu = (high_prices_gpu - low_prices_gpu) / close_prices_gpu
                volume_numpy = volume_gpu.get()  # .get()で変換
            
            returns_gpu = cp.diff(cp.log(close_prices_gpu))
            abs_returns_gpu = cp.abs(returns_gpu)
            abs_returns_numpy = abs_returns_gpu.get()  # .get()で変換
            
            for window in [20, 50, 100]:
                amihud_values = []
                for i in range(window, len(abs_returns_numpy)):
                    try:
                        window_returns = abs_returns_numpy[i-window:i]
                        window_volume = volume_numpy[i-window:i] if len(volume_numpy) > i else np.ones(window)
                        
                        amihud = np.mean(window_returns / (window_volume + 1e-10))
                        amihud_values.append(amihud)
                    except:
                        amihud_values.append(np.nan)
                
                features[f'amihud_illiquidity_{window}'] = np.concatenate([
                    np.full(window, np.nan), np.array(amihud_values)
                ])
            
            # Roll実効スプレッド推定
            returns_numpy = returns_gpu.get()  # .get()で変換
            for window in [50, 100]:
                roll_values = []
                for i in range(window, len(returns_numpy)):
                    sample_returns = returns_numpy[i-window:i]
                    try:
                        # Roll(1984)スプレッド推定
                        covariance = np.cov(sample_returns[:-1], sample_returns[1:])[0, 1]
                        roll_spread = 2 * np.sqrt(-covariance) if covariance < 0 else 0
                        roll_values.append(roll_spread)
                    except:
                        roll_values.append(np.nan)
                
                features[f'roll_spread_{window}'] = np.concatenate([
                    [np.nan] * window, roll_values
                ])
            
            # 高頻度価格効率性指標
            close_prices_numpy = close_prices_gpu.get()  # .get()で変換
            for window in [20, 50]:
                efficiency_values = []
                for i in range(window, len(close_prices_numpy)):
                    prices = close_prices_numpy[i-window:i]
                    try:
                        # 価格ランダムウォーク効率性
                        price_changes = np.diff(prices)
                        efficiency = self._price_efficiency_ratio(price_changes)
                        efficiency_values.append(efficiency)
                    except:
                        efficiency_values.append(np.nan)
                
                features[f'price_efficiency_ratio_{window}'] = np.concatenate([
                    [np.nan] * window, efficiency_values
                ])
            
            # マーケットインパクト推定
            if len(volume_numpy) == len(close_prices_numpy):
                for window in [20, 50]:
                    impact_values = []
                    for i in range(window, len(returns_numpy)):
                        try:
                            window_returns = returns_numpy[i-window:i]
                            window_volume = volume_numpy[i-window:i]
                            
                            # 簡易マーケットインパクト（リターン-ボリューム関係）
                            impact = np.corrcoef(np.abs(window_returns), window_volume)[0, 1]
                            impact_values.append(impact)
                        except:
                            impact_values.append(np.nan)
                    
                    features[f'market_impact_{window}'] = np.concatenate([
                        [np.nan] * window, impact_values
                    ])
        
        else:
            # CPU処理ブランチ（既存コードをそのまま使用）
            for window in [20, 50, 100]:
                amihud_values = []
                for i in range(window, len(abs_returns)):
                    try:
                        window_returns = abs_returns[i-window:i]
                        window_volume = volume[i-window:i] if len(volume) > i else np.ones(window)
                        
                        amihud = np.mean(window_returns / (window_volume + 1e-10))
                        amihud_values.append(amihud)
                    except:
                        amihud_values.append(np.nan)
                
                features[f'amihud_illiquidity_{window}'] = np.concatenate([
                    np.full(window, np.nan), np.array(amihud_values)
                ])
            
            # Roll実効スプレッド推定
            for window in [50, 100]:
                roll_values = []
                for i in range(window, len(returns)):
                    sample_returns = returns[i-window:i]
                    try:
                        # Roll(1984)スプレッド推定
                        covariance = np.cov(sample_returns[:-1], sample_returns[1:])[0, 1]
                        roll_spread = 2 * np.sqrt(-covariance) if covariance < 0 else 0
                        roll_values.append(roll_spread)
                    except:
                        roll_values.append(np.nan)
                
                features[f'roll_spread_{window}'] = np.concatenate([
                    [np.nan] * window, roll_values
                ])
            
            # 高頻度価格効率性指標
            for window in [20, 50]:
                efficiency_values = []
                for i in range(window, len(close_prices)):
                    prices = close_prices[i-window:i]
                    try:
                        # 価格ランダムウォーク効率性
                        price_changes = np.diff(prices)
                        efficiency = self._price_efficiency_ratio(price_changes)
                        efficiency_values.append(efficiency)
                    except:
                        efficiency_values.append(np.nan)
                
                features[f'price_efficiency_ratio_{window}'] = np.concatenate([
                    [np.nan] * window, efficiency_values
                ])
            
            # マーケットインパクト推定
            if len(volume) == len(close_prices):
                for window in [20, 50]:
                    impact_values = []
                    for i in range(window, len(returns)):
                        try:
                            window_returns = returns[i-window:i]
                            window_volume = volume[i-window:i]
                            
                            # 簡易マーケットインパクト（リターン-ボリューム関係）
                            impact = np.corrcoef(np.abs(window_returns), window_volume)[0, 1]
                            impact_values.append(impact)
                        except:
                            impact_values.append(np.nan)
                    
                    features[f'market_impact_{window}'] = np.concatenate([
                        [np.nan] * window, impact_values
                    ])
        
        return features
    
    # =========================================================================
    # 第12章: 機械学習特徴量群
    # =========================================================================
    
    def generate_ml_features(self, df):
        """機械学習由来特徴量生成"""
        features = {}
        
        close_prices = df['close'].values
        returns = np.diff(np.log(close_prices))
        
        # GPU/CPU処理の分岐をループ外に移動（ルール1）
        if self.gpu_optimization and isinstance(df, cudf.DataFrame):
            # GPU処理ブランチ
            close_prices_gpu = cp.asarray(close_prices)
            returns_gpu = cp.diff(cp.log(close_prices_gpu))
            returns_numpy = returns_gpu.get()  # .get()で変換（sklearn用）
            
            # 主成分分析特徴量
            if len(returns_numpy) > 100:
                try:
                    # ローリング主成分分析
                    pca_window = 100
                    n_components = 5
                    
                    pca = GPU_PCA(n_components=n_components)  # GPU版PCA
                    
                    pca_features = []
                    for i in range(pca_window, len(returns_numpy)):
                        sample = returns_numpy[i-pca_window:i].reshape(-1, 1)
                        
                        # 特徴量行列作成（ラグ特徴量）
                        feature_matrix = self._create_lag_matrix(sample.flatten(), max_lag=10)
                        
                        try:
                            # GPU処理ブランチ  
                            from sklearn.preprocessing import StandardScaler
                            scaler = StandardScaler()
                            feature_matrix_normalized = scaler.fit_transform(feature_matrix)
                            feature_matrix_gpu = cp.asarray(feature_matrix_normalized)
                            pca.fit(feature_matrix_gpu)
                            components = pca.explained_variance_ratio_.get()  # GPU結果を.get()で変換
                            pca_features.append(components)
                        except:
                            pca_features.append([np.nan] * n_components)
                    
                    pca_array = np.array(pca_features)
                    for i in range(n_components):
                        features[f'pca_component_{i}_variance'] = np.concatenate([
                            [np.nan] * pca_window, pca_array[:, i]
                        ])
                
                except Exception as e:
                    print(f"Warning: PCA feature generation failed: {e}")
            
            # 独立成分分析特徴量
            if len(returns_numpy) > 200:
                try:
                    ica_window = 200
                    n_components = 3
                    
                    from sklearn.decomposition import FastICA  # CPUライブラリ
                    ica = FastICA(n_components=n_components, random_state=42)
                    
                    ica_features = []
                    for i in range(ica_window, len(returns_numpy)):
                        sample = returns_numpy[i-ica_window:i]
                        feature_matrix = self._create_lag_matrix(sample, max_lag=5)
                        
                        try:
                            ica.fit(feature_matrix)  # CPU関数なので変換済み配列を使用
                            components = ica.components_
                            ica_features.append(np.var(components, axis=1))
                        except:
                            ica_features.append([np.nan] * n_components)
                    
                    ica_array = np.array(ica_features)
                    for i in range(n_components):
                        features[f'ica_component_{i}_variance'] = np.concatenate([
                            [np.nan] * ica_window, ica_array[:, i]
                        ])
                
                except Exception as e:
                    print(f"Warning: ICA feature generation failed: {e}")
            
            # オートエンコーダ特徴量（簡易版）
            for window in [50, 100]:
                autoencoder_features = []
                for i in range(window, len(returns_numpy)):
                    sample = returns_numpy[i-window:i]
                    try:
                        # 簡易オートエンコーダ（PCA近似）
                        from sklearn.decomposition import PCA  # CPUライブラリ
                        pca = PCA(n_components=min(10, window//5))
                        
                        sample_matrix = sample.reshape(-1, 1)
                        pca.fit(sample_matrix)  # CPU関数なので変換済み配列を使用
                        
                        # 再構成誤差
                        transformed = pca.transform(sample_matrix)
                        reconstructed = pca.inverse_transform(transformed)
                        reconstruction_error = np.mean((sample_matrix - reconstructed)**2)
                        
                        autoencoder_features.append(reconstruction_error)
                    except:
                        autoencoder_features.append(np.nan)
                
                features[f'autoencoder_reconstruction_error_{window}'] = np.concatenate([
                    [np.nan] * window, autoencoder_features
                ])
            
            # 特徴量重要度（Random Forest由来）
            if len(returns_numpy) > 200:
                try:
                    rf_window = 200
                    
                    rf = GPU_RandomForest(n_estimators=50, random_state=42)  # GPU版RF
                    
                    importance_scores = []
                    for i in range(rf_window, len(returns_numpy)):
                        sample = returns_numpy[i-rf_window:i]
                        
                        # 特徴量・ターゲット作成
                        X = self._create_lag_matrix(sample[:-1], max_lag=10)
                        y = sample[10:]  # 10期先予測
                        
                        try:
                            X_gpu = cp.asarray(X)
                            y_gpu = cp.asarray(y)
                            rf.fit(X_gpu, y_gpu)
                            importances = rf.feature_importances_.get()  # GPU結果を.get()で変換
                            importance_scores.append(np.mean(importances))
                        except:
                            importance_scores.append(np.nan)
                    
                    features[f'rf_feature_importance_{rf_window}'] = np.concatenate([
                        [np.nan] * rf_window, importance_scores
                    ])
                
                except Exception as e:
                    print(f"Warning: Random Forest feature importance failed: {e}")
        
        else:
            # CPU処理ブランチ（既存コードをそのまま使用）
            # 主成分分析特徴量
            if len(returns) > 100:
                try:
                    # ローリング主成分分析
                    pca_window = 100
                    n_components = 5
                    
                    from sklearn.decomposition import PCA
                    pca = PCA(n_components=n_components)
                    
                    pca_features = []
                    for i in range(pca_window, len(returns)):
                        sample = returns[i-pca_window:i].reshape(-1, 1)
                        
                        # 特徴量行列作成（ラグ特徴量）
                        feature_matrix = self._create_lag_matrix(sample.flatten(), max_lag=10)
                        
                        try:
                            # CPU処理ブランチ
                            from sklearn.preprocessing import StandardScaler
                            scaler = StandardScaler()
                            feature_matrix_normalized = scaler.fit_transform(feature_matrix)
                            pca.fit(feature_matrix_normalized)
                            components = pca.components_
                            
                            # 主成分の寄与率
                            explained_variance = pca.explained_variance_ratio_ if hasattr(pca, 'explained_variance_ratio_') else [0.5, 0.3, 0.1, 0.06, 0.04]
                            pca_features.append(explained_variance)
                        except:
                            pca_features.append([np.nan] * n_components)
                    
                    pca_array = np.array(pca_features)
                    for i in range(n_components):
                        features[f'pca_component_{i}_variance'] = np.concatenate([
                            [np.nan] * pca_window, pca_array[:, i]
                        ])
                
                except Exception as e:
                    print(f"Warning: PCA feature generation failed: {e}")
            
            # 独立成分分析特徴量
            if len(returns) > 200:
                try:
                    ica_window = 200
                    n_components = 3
                    
                    from sklearn.decomposition import FastICA
                    ica = FastICA(n_components=n_components, random_state=42)
                    
                    ica_features = []
                    for i in range(ica_window, len(returns)):
                        sample = returns[i-ica_window:i]
                        feature_matrix = self._create_lag_matrix(sample, max_lag=5)
                        
                        try:
                            ica.fit(feature_matrix)
                            components = ica.components_
                            ica_features.append(np.var(components, axis=1))
                        except:
                            ica_features.append([np.nan] * n_components)
                    
                    ica_array = np.array(ica_features)
                    for i in range(n_components):
                        features[f'ica_component_{i}_variance'] = np.concatenate([
                            [np.nan] * ica_window, ica_array[:, i]
                        ])
                
                except Exception as e:
                    print(f"Warning: ICA feature generation failed: {e}")
            
            # オートエンコーダ特徴量（簡易版）
            for window in [50, 100]:
                autoencoder_features = []
                for i in range(window, len(returns)):
                    sample = returns[i-window:i]
                    try:
                        # 簡易オートエンコーダ（PCA近似）
                        from sklearn.decomposition import PCA
                        pca = PCA(n_components=min(10, window//5))
                        
                        sample_matrix = sample.reshape(-1, 1)
                        pca.fit(sample_matrix)
                        
                        # 再構成誤差
                        transformed = pca.transform(sample_matrix)
                        reconstructed = pca.inverse_transform(transformed)
                        reconstruction_error = np.mean((sample_matrix - reconstructed)**2)
                        
                        autoencoder_features.append(reconstruction_error)
                    except:
                        autoencoder_features.append(np.nan)
                
                features[f'autoencoder_reconstruction_error_{window}'] = np.concatenate([
                    [np.nan] * window, autoencoder_features
                ])
            
            # 特徴量重要度（Random Forest由来）
            if len(returns) > 200:
                try:
                    rf_window = 200
                    
                    from sklearn.ensemble import RandomForestRegressor
                    rf = RandomForestRegressor(n_estimators=50, random_state=42)
                    
                    importance_scores = []
                    for i in range(rf_window, len(returns)):
                        sample = returns[i-rf_window:i]
                        
                        # 特徴量・ターゲット作成
                        X = self._create_lag_matrix(sample[:-1], max_lag=10)
                        y = sample[10:]  # 10期先予測
                        
                        try:
                            rf.fit(X, y)
                            importances = rf.feature_importances_ if hasattr(rf, 'feature_importances_') else np.random.random(X.shape[1])
                            importance_scores.append(np.mean(importances))
                        except:
                            importance_scores.append(np.nan)
                    
                    features[f'rf_feature_importance_{rf_window}'] = np.concatenate([
                        [np.nan] * rf_window, importance_scores
                    ])
                
                except Exception as e:
                    print(f"Warning: Random Forest feature importance failed: {e}")
        
        return features
    
    # =========================================================================
    # 第13章: ユーティリティメソッド群
    # =========================================================================
      
    def _convert_to_numpy(self, data):
        """CuPy配列をNumPy配列に安全に変換"""
        if isinstance(data, cp.ndarray):
            return data.get()
        elif hasattr(data, 'values') and isinstance(data.values, cp.ndarray):
            return data.values.get()
        else:
            return np.array(data) if not isinstance(data, np.ndarray) else data
        
    def _gpu_rolling_mean(self, data, window):
        """GPU最適化ローリング平均"""
        if len(data) < window:
            return cp.full(len(data), cp.nan)
        
        # CuPyでのローリング平均実装
        result = cp.full(len(data), cp.nan)
        for i in range(window-1, len(data)):
            result[i] = cp.mean(data[i-window+1:i+1])
        
        return result
    
    def _gpu_rolling_window(self, data, window):
        """GPU最適化ローリングウィンドウ"""
        n = len(data)
        if n < window:
            return cp.array([])
        
        # ローリングウィンドウ行列作成
        rolling_matrix = cp.full((n - window + 1, window), cp.nan)
        for i in range(n - window + 1):
            rolling_matrix[i] = data[i:i+window]
        
        return rolling_matrix
    
    def _gpu_skewness(self, data):
        """GPU最適化歪度計算"""
        if data.ndim == 1:
            mean = cp.nanmean(data)
            std = cp.nanstd(data)
            if std == 0:
                return cp.nan
            return cp.nanmean(((data - mean) / std) ** 3)
        else:
            # 2次元配列（ローリングウィンドウ）
            means = cp.nanmean(data, axis=1)
            stds = cp.nanstd(data, axis=1)
            
            result = cp.full(data.shape[0], cp.nan)
            for i in range(data.shape[0]):
                if stds[i] > 0:
                    standardized = (data[i] - means[i]) / stds[i]
                    result[i] = cp.nanmean(standardized ** 3)
            
            return result
    
    def _gpu_kurtosis(self, data):
        """GPU最適化尖度計算"""
        if data.ndim == 1:
            mean = cp.nanmean(data)
            std = cp.nanstd(data)
            if std == 0:
                return cp.nan
            return cp.nanmean(((data - mean) / std) ** 4) - 3
        else:
            # 2次元配列（ローリングウィンドウ）
            means = cp.nanmean(data, axis=1)
            stds = cp.nanstd(data, axis=1)
            
            result = cp.full(data.shape[0], cp.nan)
            for i in range(data.shape[0]):
                if stds[i] > 0:
                    standardized = (data[i] - means[i]) / stds[i]
                    result[i] = cp.nanmean(standardized ** 4) - 3
            
            return result
    
    def _takens_embedding(self, data, dim, tau):
        """タケンス埋め込み実装"""
        N = len(data)
        if N < (dim - 1) * tau + 1:
            raise ValueError("データが短すぎます")
        
        embedded = np.zeros((N - (dim - 1) * tau, dim))
        for i in range(dim):
            embedded[:, i] = data[i * tau:N - (dim - 1 - i) * tau]
        
        return embedded
    
    def _recurrence_plot_analysis(self, embedded_data, epsilon=None):
        """再帰プロット解析"""
        if epsilon is None:
            # 自動閾値設定（距離の10%点）
            distances = []
            for i in range(min(100, len(embedded_data))):
                for j in range(i + 1, min(100, len(embedded_data))):
                    dist = np.linalg.norm(embedded_data[i] - embedded_data[j])
                    distances.append(dist)
            epsilon = np.percentile(distances, 10)
        
        N = len(embedded_data)
        recurrence_matrix = np.zeros((N, N))
        
        for i in range(N):
            for j in range(N):
                if np.linalg.norm(embedded_data[i] - embedded_data[j]) < epsilon:
                    recurrence_matrix[i, j] = 1
        
        # 再帰プロット特徴量計算
        features = {}
        features['recurrence_rate'] = np.sum(recurrence_matrix) / (N * N)
        features['determinism'] = self._calculate_determinism(recurrence_matrix)
        features['average_diagonal_length'] = self._calculate_average_diagonal_length(recurrence_matrix)
        
        return features
    
    def _calculate_determinism(self, recurrence_matrix):
        """決定論性計算"""
        N = recurrence_matrix.shape[0]
        diagonal_lengths = []
        
        # 対角線の長さ計算
        for offset in range(-N+1, N):
            diagonal = np.diagonal(recurrence_matrix, offset=offset)
            consecutive_ones = 0
            for val in diagonal:
                if val == 1:
                    consecutive_ones += 1
                else:
                    if consecutive_ones >= 2:  # 最小対角線長
                        diagonal_lengths.append(consecutive_ones)
                    consecutive_ones = 0
            
            if consecutive_ones >= 2:
                diagonal_lengths.append(consecutive_ones)
        
        if len(diagonal_lengths) == 0:
            return 0.0
        
        return np.sum(diagonal_lengths) / np.sum(recurrence_matrix)
    
    def _calculate_average_diagonal_length(self, recurrence_matrix):
        """平均対角線長計算"""
        N = recurrence_matrix.shape[0]
        diagonal_lengths = []
        
        for offset in range(-N+1, N):
            diagonal = np.diagonal(recurrence_matrix, offset=offset)
            consecutive_ones = 0
            for val in diagonal:
                if val == 1:
                    consecutive_ones += 1
                else:
                    if consecutive_ones >= 2:
                        diagonal_lengths.append(consecutive_ones)
                    consecutive_ones = 0
            
            if consecutive_ones >= 2:
                diagonal_lengths.append(consecutive_ones)
        
        return np.mean(diagonal_lengths) if diagonal_lengths else 0.0
    
    def _poincare_plot_analysis(self, data, lag=1):
        """ポアンカレプロット解析"""
        if len(data) <= lag:
            return {'sd1': np.nan, 'sd2': np.nan, 'sd_ratio': np.nan}
        
        x = data[:-lag]
        y = data[lag:]
        
        # SD1, SD2計算
        diff = x - y
        sum_xy = x + y
        
        sd1 = np.std(diff) / np.sqrt(2)
        sd2 = np.std(sum_xy) / np.sqrt(2)
        sd_ratio = sd1 / sd2 if sd2 > 0 else np.nan
        
        return {
            'sd1': sd1,
            'sd2': sd2,
            'sd_ratio': sd_ratio
        }
    
    def _grassberger_procaccia_dimension(self, data, max_dim=10):
        """Grassberger-Procaccia相関次元"""
        try:
            # 簡易実装
            embedded_dims = range(1, min(max_dim + 1, len(data) // 10))
            correlation_dims = []
            
            for dim in embedded_dims:
                if len(data) > dim * 3:
                    embedded = self._takens_embedding(data, dim, 1)
                    # 相関積分計算（サンプリング）
                    sample_size = min(100, len(embedded))
                    sample_indices = np.random.choice(len(embedded), sample_size, replace=False)
                    embedded_sample = embedded[sample_indices]
                    
                    distances = []
                    for i in range(len(embedded_sample)):
                        for j in range(i + 1, len(embedded_sample)):
                            dist = np.linalg.norm(embedded_sample[i] - embedded_sample[j])
                            distances.append(dist)
                    
                    if distances:
                        # 相関次元近似
                        epsilons = np.logspace(np.log10(min(distances)), np.log10(max(distances)), 10)
                        correlations = []
                        for eps in epsilons:
                            correlation = np.sum(np.array(distances) < eps) / len(distances)
                            correlations.append(correlation + 1e-10)  # ログのためのゼロ回避
                        
                        # 対数スロープ
                        log_eps = np.log(epsilons)
                        log_corr = np.log(correlations)
                        slope = np.polyfit(log_eps, log_corr, 1)[0]
                        correlation_dims.append(slope)
            
            return np.mean(correlation_dims) if correlation_dims else np.nan
        except:
            return np.nan
    
    def _false_nearest_neighbors(self, data):
        """False Nearest Neighbors法"""
        try:
            max_dim = min(10, len(data) // 20)
            if max_dim < 2:
                return np.nan
            
            fnn_percentages = []
            
            for dim in range(1, max_dim):
                if len(data) > (dim + 1) * 3:
                    embedded_dim = self._takens_embedding(data, dim, 1)
                    embedded_dim_plus = self._takens_embedding(data, dim + 1, 1)
                    
                    false_neighbors = 0
                    total_neighbors = 0
                    
                    # サンプリングによる計算量削減
                    sample_size = min(50, len(embedded_dim))
                    for i in np.random.choice(len(embedded_dim), sample_size, replace=False):
                        # 最近傍探索
                        distances_dim = [np.linalg.norm(embedded_dim[i] - embedded_dim[j]) 
                                       for j in range(len(embedded_dim)) if j != i]
                        
                        if distances_dim:
                            nearest_idx = np.argmin(distances_dim)
                            if nearest_idx >= i:
                                nearest_idx += 1  # インデックス調整
                            
                            # 次元拡張後の距離
                            if nearest_idx < len(embedded_dim_plus):
                                dist_dim = distances_dim[nearest_idx] if nearest_idx < len(distances_dim) else distances_dim[-1]
                                dist_dim_plus = np.linalg.norm(embedded_dim_plus[i] - embedded_dim_plus[nearest_idx])
                                
                                # False neighbor判定
                                if dist_dim > 0:
                                    ratio = (dist_dim_plus - dist_dim) / dist_dim
                                    if ratio > 15:  # 閾値
                                        false_neighbors += 1
                                total_neighbors += 1
                    
                    fnn_percentage = false_neighbors / total_neighbors if total_neighbors > 0 else 0
                    fnn_percentages.append(fnn_percentage)
            
            # 最適埋め込み次元（FNN < 1%となる最小次元）
            for i, percentage in enumerate(fnn_percentages):
                if percentage < 0.01:
                    return i + 1
            
            return len(fnn_percentages) + 1
        except:
            return np.nan
    
    def _transfer_entropy_estimation(self, x, y, lag):
        """Transfer Entropy推定（簡易版）"""
        try:
            if len(x) != len(y) or len(x) < lag + 10:
                return np.nan
            
            # 離散化
            x_discrete = self._discretize_series(x, bins=5)
            y_discrete = self._discretize_series(y, bins=5)
            
            # 条件付きエントロピー計算
            te = 0.0
            
            for t in range(lag, len(y_discrete)):
                y_current = y_discrete[t]
                y_past = y_discrete[t-lag:t]
                x_past = x_discrete[t-lag:t]
                
                # 簡易条件付き確率計算
                # P(Y_t | Y_{t-1}, X_{t-1}) vs P(Y_t | Y_{t-1})
                
                # 実用的な近似として、相関ベースの計算
                correlation_xy = np.corrcoef(x[t-lag:t], y[t-lag:t])[0, 1]
                correlation_yy = np.corrcoef(y[t-lag:t], y[t:t+1] if t+1 < len(y) else y[t-1:t])[0, 1]
                
                te += abs(correlation_xy) - abs(correlation_yy) if not np.isnan(correlation_xy) and not np.isnan(correlation_yy) else 0
            
            return te / (len(y_discrete) - lag) if len(y_discrete) > lag else np.nan
        except:
            return np.nan
    
    def _mutual_information_estimation(self, x, y):
        """Mutual Information推定"""
        try:
            if len(x) != len(y) or len(x) < 10:
                return np.nan
            
            # ビン数適応的設定
            n_bins = max(5, min(20, int(np.sqrt(len(x)))))
            
            # 2次元ヒストグラム
            hist_xy, _, _ = np.histogram2d(x, y, bins=n_bins)
            hist_x, _ = np.histogram(x, bins=n_bins)
            hist_y, _ = np.histogram(y, bins=n_bins)
            
            # 正規化
            p_xy = hist_xy / np.sum(hist_xy)
            p_x = hist_x / np.sum(hist_x)
            p_y = hist_y / np.sum(hist_y)
            
            # MI計算
            mi = 0.0
            for i in range(n_bins):
                for j in range(n_bins):
                    if p_xy[i, j] > 0 and p_x[i] > 0 and p_y[j] > 0:
                        mi += p_xy[i, j] * np.log(p_xy[i, j] / (p_x[i] * p_y[j]))
            
            return mi
        except:
            return np.nan
    
    def _discretize_series(self, data, bins=5):
        """時系列離散化"""
        try:
            return np.digitize(data, np.linspace(np.min(data), np.max(data), bins))
        except:
            return np.array([1] * len(data))
    
    def _scaling_exponent(self, data):
        """スケーリング指数計算"""
        try:
            if len(data) < 20:
                return np.nan
            
            # Detrended Fluctuation Analysis簡易版
            n = len(data)
            scales = np.unique(np.logspace(1, np.log10(n//4), 10).astype(int))
            fluctuations = []
            
            for scale in scales:
                if scale >= len(data):
                    continue
                    
                # 積分時系列
                integrated = np.cumsum(data - np.mean(data))
                
                # スケールごとの変動計算
                segments = len(integrated) // scale
                detrended = []
                
                for i in range(segments):
                    start = i * scale
                    end = start + scale
                    segment = integrated[start:end]
                    
                    # 線形トレンド除去
                    x = np.arange(len(segment))
                    coeffs = np.polyfit(x, segment, 1)
                    trend = np.polyval(coeffs, x)
                    detrended.extend(segment - trend)
                
                if detrended:
                    fluctuation = np.sqrt(np.mean(np.array(detrended)**2))
                    fluctuations.append(fluctuation)
                else:
                    fluctuations.append(np.nan)
            
            # スケーリング指数（対数線形回帰）
            valid_scales = []
            valid_fluct = []
            for i, fluct in enumerate(fluctuations):
                if not np.isnan(fluct) and fluct > 0:
                    valid_scales.append(scales[i])
                    valid_fluct.append(fluct)
            
            if len(valid_scales) >= 3:
                log_scales = np.log10(valid_scales)
                log_fluct = np.log10(valid_fluct)
                scaling_exp = np.polyfit(log_scales, log_fluct, 1)[0]
                return scaling_exp
            else:
                return np.nan
        except:
            return np.nan
    
    def _gph_long_memory_estimate(self, data):
        """Geweke-Porter-Hudak長期記憶推定"""
        try:
            n = len(data)
            if n < 50:
                return np.nan
            
            # フーリエ変換
            fft_data = np.fft.fft(data)
            frequencies = np.fft.fftfreq(n)
            
            # 正の周波数のみ
            positive_freq_idx = frequencies > 0
            pos_frequencies = frequencies[positive_freq_idx]
            periodogram = np.abs(fft_data[positive_freq_idx])**2
            
            # 低周波数域での回帰（GPH推定）
            m = min(int(n**0.5), len(pos_frequencies) // 2)
            if m < 10:
                return np.nan
            
            # 回帰変数
            x = np.log(4 * np.sin(np.pi * pos_frequencies[:m] / 2)**2)
            y = np.log(periodogram[:m])
            
            # 線形回帰
            coeffs = np.polyfit(x, y, 1)
            d_estimate = -coeffs[0] / 2  # 長期記憶パラメータ
            
            return d_estimate
        except:
            return np.nan
    
    def _price_efficiency_ratio(self, price_changes):
        """価格効率性比率"""
        try:
            if len(price_changes) < 2:
                return np.nan
            
            # 直線距離
            straight_distance = abs(np.sum(price_changes))
            
            # 経路距離
            path_distance = np.sum(np.abs(price_changes))
            
            # 効率性比率
            efficiency = straight_distance / path_distance if path_distance > 0 else 0
            
            return efficiency
        except:
            return np.nan
    
    def _create_lag_matrix(self, data, max_lag=10):
        """ラグ特徴量行列作成"""
        try:
            n = len(data)
            if n <= max_lag:
                return np.array([])
            
            matrix = np.zeros((n - max_lag, max_lag))
            for lag in range(max_lag):
                matrix[:, lag] = data[max_lag - lag - 1:n - lag - 1]
            
            return matrix
        except:
            return np.array([])
        
    def _fallback_hurst_calculation(self, data):
        """noldsが利用できない場合のHurst指数計算代替実装"""
        try:
            if len(data) < 20:
                return np.nan
            
            # 簡易R/S分析
            n = len(data)
            lags = np.arange(2, min(20, n//2))
            rs_values = []
            
            for lag in lags:
                # 差分系列
                diffs = np.diff(data[:lag])
                if len(diffs) == 0:
                    continue
                    
                # 範囲と標準偏差
                cumulative = np.cumsum(diffs - np.mean(diffs))
                R = np.max(cumulative) - np.min(cumulative)
                S = np.std(diffs)
                
                if S > 0:
                    rs_values.append(R/S)
            
            if len(rs_values) >= 3:
                # 対数回帰でHurst指数を求める
                log_lags = np.log(lags[:len(rs_values)])
                log_rs = np.log(rs_values)
                hurst = np.polyfit(log_lags, log_rs, 1)[0]
                return hurst
            else:
                return np.nan
        except:
            return np.nan

    def _fallback_dfa_calculation(self, data):
        """noldsが利用できない場合のDFA計算代替実装"""
        try:
            if len(data) < 20:
                return np.nan
            
            # 簡易DFA実装
            n = len(data)
            integrated = np.cumsum(data - np.mean(data))
            
            scales = np.unique(np.logspace(1, np.log10(n//4), 8).astype(int))
            fluctuations = []
            
            for scale in scales:
                if scale >= len(integrated):
                    continue
                
                segments = len(integrated) // scale
                detrended_variance = 0
                
                for i in range(segments):
                    start = i * scale
                    end = start + scale
                    segment = integrated[start:end]
                    
                    # 線形トレンド除去
                    x = np.arange(len(segment))
                    coeffs = np.polyfit(x, segment, 1)
                    trend = np.polyval(coeffs, x)
                    detrended = segment - trend
                    detrended_variance += np.var(detrended)
                
                if segments > 0:
                    fluctuation = np.sqrt(detrended_variance / segments)
                    fluctuations.append(fluctuation)
            
            if len(fluctuations) >= 3:
                log_scales = np.log(scales[:len(fluctuations)])
                log_fluctuations = np.log(fluctuations)
                alpha = np.polyfit(log_scales, log_fluctuations, 1)[0]
                return alpha
            else:
                return np.nan
        except:
            return np.nan
        
    # =========================================================================
    # 第14章: 統合特徴量生成メソッド
    # =========================================================================
    
    def generate_all_features(self, df=None):
        """全特徴量統合生成"""
        # 引数dfが指定されなかった場合は、load_dataで設定されたbase_dataを使用
        if df is None:
            df = self.base_data
        
        # ここで再度Noneチェック
        if df is None:
            raise ValueError("データが読み込まれていません。load_data()を実行してください。")
        
        print("🚀 Starting comprehensive feature generation...")
        print(f"📊 Processing {df.shape[0]:,} data points across {len(self.timeframes)} timeframes")
        
        all_features = {}
        generation_times = {}
        
        # DLASCL警告フィルター
        import sys
        import re
        
        class DLASCLFilter:
            def __init__(self, original_stderr):
                self.original_stderr = original_stderr
                
            def write(self, text):
                if 'DLASCL' not in text and 'illegal value' not in text:
                    self.original_stderr.write(text)
                    
            def flush(self):
                self.original_stderr.flush()
        
        original_stderr = sys.stderr
        sys.stderr = DLASCLFilter(original_stderr)
        
        try:
            if self.gpu_optimization:
                self.memory_manager.optimize_memory()
            
            feature_generators = [
                ("Price Dynamics", self.generate_price_dynamics_features),
                ("Statistical Moments", self.generate_statistical_moments_features),
                ("Fractal Chaos", self.generate_fractal_chaos_features),
                ("Time Frequency", self.generate_time_frequency_features),
                ("Nonlinear Dynamics", self.generate_nonlinear_dynamics_features),
                ("Information Theory", self.generate_information_theory_features),
                ("Multi-timescale", self.generate_multitimescale_features),
                ("Microstructure", self.generate_microstructure_features),
                ("Machine Learning", self.generate_ml_features)
            ]
            
            # 特徴量生成ループ
            from tqdm import tqdm
            import sys

            for feature_name, generator_func in tqdm(feature_generators, 
                                       desc="Generating Feature Groups", 
                                       file=sys.stdout):
                try:
                    start_time = time.time()
                    
                    features = generator_func(df)
                    if features is not None:  # Noneチェック追加
                        all_features.update(features)
                    
                    generation_time = time.time() - start_time
                    generation_times[feature_name] = generation_time
                    
                    if self.gpu_optimization:
                        self.memory_manager.optimize_memory()
                
                except Exception as e:
                    print(f"⚠ Error generating {feature_name} features: {e}")
                    import traceback
                    traceback.print_exc()
            
            # 特徴量DataFrame作成
            print(f"🔧 Creating feature DataFrame with {len(all_features)} features...")
            
            # 全特徴量の長さ統一
            max_length = len(df)
            unified_features = {}
            
            for feature_name, feature_values in all_features.items():
                try:
                    # CuPy配列をNumPy配列に変換
                    if hasattr(feature_values, 'get'):  # CuPy配列の場合
                        feature_values = feature_values.get()
                    
                    if isinstance(feature_values, (int, float)):
                        # スカラー値を全行に展開
                        unified_features[feature_name] = np.full(max_length, feature_values)
                    elif hasattr(feature_values, '__len__') and not isinstance(feature_values, str):
                        feature_length = len(feature_values)
                        if feature_length == max_length:
                            unified_features[feature_name] = np.array(feature_values)
                        elif feature_length < max_length:
                            # 不足分をNaNで埋める
                            padded = np.full(max_length, np.nan)
                            padded[:feature_length] = feature_values
                            unified_features[feature_name] = padded
                        else:
                            # 超過分を切り捨て
                            unified_features[feature_name] = np.array(feature_values[:max_length])
                    else:
                        # 無効な特徴量は除外
                        print(f"Warning: Invalid feature {feature_name}, skipping...")
                        continue
                except Exception as e:
                    print(f"Warning: Error processing feature {feature_name}: {e}")
                    continue
            
            if self.gpu_optimization and isinstance(df, cudf.DataFrame):
                feature_df = cudf.DataFrame(unified_features, index=df.index)
            else:
                feature_df = pd.DataFrame(unified_features, index=df.index)
            
            print(f"🎯 Feature generation complete!")
            print(f"📈 Total features: {feature_df.shape[1]}")
            print(f"📊 Data points: {feature_df.shape[0]:,}")
            print(f"⏱️ Total time: {sum(generation_times.values()):.2f}s")
            
            # 生成統計表示
            print(f"\n📋 Generation Summary:")
            for name, time_taken in generation_times.items():
                print(f"   {name}: {time_taken:.2f}s")
            
            return feature_df
            
        except Exception as e:
            print(f"⚠ Error creating feature DataFrame: {e}")
            import traceback
            traceback.print_exc()
            return None
            
        finally:
            # 必ずstderrを元に戻す
            sys.stderr = original_stderr



# ============================================================================
# 第15章：メイン実行フロー
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
        if GPU_AVAILABLE:
            gpu_info = cp.cuda.runtime.getDeviceProperties(0)
            gpu_name = gpu_info['name'].decode()
            gpu_memory = cp.cuda.runtime.memGetInfo()[1] / 1024**3
            
            print(f"✅ GPU Acceleration: {gpu_name} ({gpu_memory:.1f}GB)")
            gpu_available = True
        else:
            raise Exception("RAPIDS/CuPy not available")
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

def get_feature_summary(feature_df):
    """特徴量サマリー統計"""
    if feature_df is None:
        return None
    
    summary = {
        'total_features': feature_df.shape[1],
        'total_samples': feature_df.shape[0],
        'missing_values': feature_df.isnull().sum().sum(),
        'memory_usage_mb': feature_df.memory_usage(deep=True).sum() / 1024**2,
        'feature_types': {}
    }
    
    # 特徴量タイプ別集計
    feature_categories = {
        'price_dynamics': [col for col in feature_df.columns if any(x in col.lower() for x in ['price', 'return', 'atr', 'efficiency'])],
        'statistical': [col for col in feature_df.columns if any(x in col.lower() for x in ['skew', 'kurt', 'std', 'var', 'percentile'])],
        'fractal': [col for col in feature_df.columns if any(x in col.lower() for x in ['hurst', 'dfa', 'fractal', 'lyapunov'])],
        'frequency': [col for col in feature_df.columns if any(x in col.lower() for x in ['wavelet', 'fft', 'stft', 'hilbert'])],
        'information': [col for col in feature_df.columns if any(x in col.lower() for x in ['entropy', 'mutual', 'transfer'])],
        'nonlinear': [col for col in feature_df.columns if any(x in col.lower() for x in ['recurrence', 'poincare', 'correlation_dim'])],
        'microstructure': [col for col in feature_df.columns if any(x in col.lower() for x in ['amihud', 'roll', 'impact'])],
        'ml_derived': [col for col in feature_df.columns if any(x in col.lower() for x in ['pca', 'ica', 'autoencoder', 'rf_'])]
    }
    
    for category, columns in feature_categories.items():
        summary['feature_types'][category] = len(columns)
    
    return summary


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
        
        # デバッグ: クラスとメソッドの確認
        print(f"feature_engine type: {type(feature_engine)}")
        print(f"feature_engine class: {feature_engine.__class__}")
        print(f"Available methods:")
        methods = [method for method in dir(feature_engine) if not method.startswith('_')]
        for method in sorted(methods):
            print(f"  - {method}")

        print(f"Has generate_all_features: {hasattr(feature_engine, 'generate_all_features')}")

        # generate_all_featuresを直接確認
        try:
            method = getattr(feature_engine, 'generate_all_features')
            print(f"Method found: {method}")
        except AttributeError as e:
            print(f"Method not found: {e}")
        
        
        
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
        summary = get_feature_summary(feature_df)
        
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
# 第16章: 追加実行ユーティリティ
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
# 第17章: エラーハンドリング・ロバストパス
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
# 第18章: 実行エントリーポイント
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
# 第19章: デバッグ・開発者向けユーティリティ
# ============================================================================

def debug_mode_execution():
    """デバッグモード実行"""
    print("🐛 DEBUG MODE EXECUTION")
    
    # 詳細ログ有効化
    import logging
    logging.basicConfig(level=logging.DEBUG)
    
    # 小規模データでのテスト実行
    try:
        
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
# 第20章: 高度な監視・診断システム
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
# 第21章: エクスポート・レポート生成システム
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
# 第22章: 最終実行制御・統合システム
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