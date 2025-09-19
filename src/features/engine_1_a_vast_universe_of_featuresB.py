#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
革新的特徴量収集スクリプト実装 - Block 1/7
Project Forge 軍資金増大ミッション - データ処理基盤
"""

import os
import sys
import time
import logging
import warnings
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union, Any, Iterator
from functools import partial, wraps
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp
import re


# 数値計算・データ処理
import numpy as np
import pandas as pd
import polars as pl
from numba import njit, jit
import numba

# 科学計算ライブラリ
from scipy import stats
from scipy.stats import skew, kurtosis, entropy as scipy_entropy
from scipy.stats import normaltest, shapiro, anderson
from scipy.special import gamma, beta, digamma, polygamma
from scipy.integrate import quad

# 信号処理・物理
from scipy.signal import welch, periodogram, find_peaks, savgol_filter
from scipy.signal import hilbert, correlate, coherence, spectrogram
from scipy.fft import fft, fftfreq, rfft, rfftfreq, fftshift
from scipy.ndimage import gaussian_filter, uniform_filter, median_filter

# 空間・幾何
from scipy.spatial.distance import pdist, squareform
from scipy.spatial import ConvexHull, Voronoi
from scipy.optimize import curve_fit, minimize_scalar

# ウェーブレット
import pywt

# エントロピー（オプショナル）
try:
    import entropy as ent
    ENTROPY_AVAILABLE = True
except ImportError:
    ENTROPY_AVAILABLE = False
    logging.warning("entropyライブラリが利用できません")

# 機械学習
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA, FastICA
from sklearn.cluster import KMeans
from sklearn.metrics import mutual_info_score
from sklearn.feature_selection import mutual_info_regression

warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=FutureWarning)

import json
    
# ログ設定
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('feature_extraction.log', encoding='utf-8')
    ]
)
logger = logging.getLogger(__name__)

# =============================================================================
# グローバル定数とパラメータ
# =============================================================================

# ハードウェア仕様
HARDWARE_SPEC = {
    'gpu_memory': '12GB',  # RTX 3060
    'cpu_cores': 6,        # i7-8700K
    'ram_limit': 64,       # 64GB RAM
    'ssd_type': 'NVMe'     # 高速アクセス
}

# データ仕様
DATA_CONFIG = {
    'base_path': Path('/workspaces/project_forge/data/1_XAUUSD_base_data/XAUUSD_tick_master_GOLDEN'),
    'output_path': Path('/workspaces/project_forge/data/2_feature_value/'),
    'timeframes': ['tick', 'M0.5', 'M1', 'M3', 'M5', 'M8', 'M15', 'M30', 'H1', 'H4', 'H6', 'H12', 'D1', 'W1', 'MN'],
    'required_columns': ['timestamp', 'open', 'high', 'low', 'close', 'volume', 'timeframe'],
    'additional_columns': ['log_return', 'rolling_volatility', 'rolling_avg_volume', 'atr', 
                          'price_direction', 'price_momentum', 'volume_ratio']
}

# 計算パラメータ
CALC_PARAMS = {
    # 基本テクニカル指標
    "rsi_periods": [7, 14, 21, 30, 50],
    "macd_settings": [(8, 21, 5), (12, 26, 9), (19, 39, 9)],
    "bollinger_settings": [(10, 1.5), (20, 2), (30, 2.5)],
    "atr_periods": [14, 21, 30],
    "adx_periods": [14, 21],
    "cci_periods": [14, 20, 30],
    "williams_r_periods": [14, 21],
    "aroon_periods": [14, 25],
    "stochastic_settings": [(14, 3), (21, 5), (5, 3)],
    
    # 出来高系指標
    "cmf_periods": [20, 21],
    "mfi_periods": [14, 21],
    "vol_roc_periods": [12, 25],
    
    # トレンド系指標
    "short_ma_periods": [5, 8, 10, 13, 20, 21],
    "long_ma_periods": [34, 50, 89, 144, 200],
    "ma_deviation_periods": [20, 50, 200],
    "tma_periods": [20, 50],
    "zlema_periods": [20, 50],
    "dema_periods": [20, 50],
    "tema_periods": [20, 50],
    
    # ボラティリティ系指標
    "volatility_bb_settings": [(10, 1.5), (20, 2), (20, 2.5), (50, 2)],
    "kc_periods": [20, 50],
    "dc_periods": [20, 55],
    "atr_periods_vol": [14, 21],
    "hist_vol_periods": [20, 30, 60],
    
    # サポート・レジスタンス系指標
    "price_channel_periods": [20, 50],
    
    # 数学・統計学
    "stat_windows": [10, 20, 50],
    "dist_windows": [20, 50],
    "robust_stat_windows": [15, 30],
    "order_stat_windows": [10, 25, 50],
    
    # 物理学・工学
    "hilbert_windows": [32, 64],
    "autocorr_lags": [5, 10, 20],
    "spectral_windows": [64, 128],
    "fourier_windows": [32, 64, 128],
    "wavelets": ['db4', 'haar', 'coif2', 'bior2.2'],
    "cwt_windows": [32, 64],
    "gaussian_sigmas": [1, 2, 3],
    "median_sizes": [3, 5, 7],
    "savgol_windows": [11, 21],
    "energy_windows": [10, 20, 50],
    
    # 情報理論
    "entropy_windows": [20, 50],
    "adv_entropy_windows": [30, 60],
    "lz_windows": [50, 100],
    "kolmogorov_windows": [30, 60],
    "mutual_info_lags": [1, 5, 10],
    
    # 地球科学・物理
    "self_similarity_scales": [5, 10, 20],
    
    # 経済学・金融
    "var_confidence_levels": [0.95, 0.99]
}

# 数値計算定数
NUMERICAL_CONSTANTS = {
    'EPS': 1e-12,
    'CONDITION_NUMBER_THRESHOLD': 1e12,
    'OUTLIER_THRESHOLD': 5.0,
    'MIN_VALID_RATIO': 0.7,
    'NAN_THRESHOLD': 0.3,
    'INF_THRESHOLD': 0.05,
    'QUALITY_THRESHOLD': 0.6
}

# =============================================================================
# ユーティリティ関数とデコレータ
# =============================================================================

def handle_zero_std(func):
    """標準偏差がゼロに近い場合の安全装置デコレータ"""
    @wraps(func)
    def wrapper(self, x, *args, **kwargs):
        try:
            if hasattr(x, 'std'):
                std_dev = x.std()
            else:
                std_dev = np.std(x)

            is_scalar = np.isscalar(std_dev)
            
            if not is_scalar and isinstance(std_dev, (pd.Series, np.ndarray)):
                if (std_dev < 1e-9).all() or pd.isna(std_dev).all():
                    return 0.0
            elif is_scalar:
                if std_dev < 1e-9 or np.isnan(std_dev):
                    return 0.0

            return func(self, x, *args, **kwargs)

        except Exception as e:
            logger.debug(f"handle_zero_std error in {func.__name__}: {e}")
            return 0.0
    return wrapper

def memory_monitor(func):
    """メモリ使用量監視デコレータ"""
    @wraps(func)
    def wrapper(*args, **kwargs):
        import psutil
        process = psutil.Process(os.getpid())
        mem_before = process.memory_info().rss / 1024 / 1024 / 1024  # GB
        
        result = func(*args, **kwargs)
        
        mem_after = process.memory_info().rss / 1024 / 1024 / 1024  # GB
        if mem_after > HARDWARE_SPEC['ram_limit'] * 0.8:  # 80%閾値
            logger.warning(f"メモリ使用量警告: {mem_after:.2f}GB / {HARDWARE_SPEC['ram_limit']}GB")
        
        logger.debug(f"{func.__name__}: メモリ使用量 {mem_before:.2f}GB → {mem_after:.2f}GB")
        return result
    return wrapper

def progress_tracker(total_steps: int):
    """進捗追跡デコレータ"""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            start_time = time.time()
            result = func(*args, **kwargs)
            elapsed = time.time() - start_time
            logger.info(f"{func.__name__} 完了: {elapsed:.2f}秒")
            return result
        return wrapper
    return decorator

# =============================================================================
# データプロセッサクラス
# =============================================================================

class DataProcessor:
    """
    データ処理クラス - 真のストリーミング処理対応
    メモリ効率を最大化したチャンクベースのmemmap作成
    """
    
    def __init__(self, base_path: Path):
        self.base_path = Path(base_path)
        self.metadata_path = self.base_path / "_metadata"
        self.memmap_cache_dir = self.base_path.parent / "memmap_cache"
        self.memmap_cache_dir.mkdir(parents=True, exist_ok=True)
        
        # パーティション構造情報
        self.partition_info = {}
        self.total_rows = 0
        self.schema_info = {}
        
        logger.info(f"DataProcessor初期化: {self.base_path}")
    
    @memory_monitor
    def load_metadata(self) -> Dict[str, Any]:
        """_metadataファイルから構造情報を読み込み"""
        if not self.metadata_path.exists():
            raise FileNotFoundError(f"メタデータファイルが見つかりません: {self.metadata_path}")
        
        try:
            # PyArrowでメタデータ読み込み
            import pyarrow.parquet as pq
            metadata = pq.read_metadata(self.metadata_path)
            
            self.total_rows = metadata.num_rows
            
            # PyArrow バージョン互換性を考慮したスキーマ情報取得
            schema_columns = []
            schema_dtypes = {}
            
            for i in range(len(metadata.schema)):
                column = metadata.schema[i]
                column_name = column.name
                schema_columns.append(column_name)
                
                # PyArrow バージョンに応じた型情報取得
                try:
                    if hasattr(column, 'type'):
                        schema_dtypes[column_name] = str(column.type)
                    elif hasattr(column, 'physical_type'):
                        schema_dtypes[column_name] = str(column.physical_type)
                    else:
                        schema_dtypes[column_name] = 'unknown'
                except AttributeError:
                    schema_dtypes[column_name] = 'unknown'
            
            self.schema_info = {
                'columns': schema_columns,
                'dtypes': schema_dtypes
            }
            
            logger.info(f"メタデータ読み込み完了: {self.total_rows:,}行, {len(self.schema_info['columns'])}列")
            return {
                'total_rows': self.total_rows,
                'schema': self.schema_info,
                'num_row_groups': metadata.num_row_groups
            }
            
        except Exception as e:
            logger.error(f"メタデータ読み込みエラー: {e}")
            raise
    
    def scan_partition_structure(self) -> Dict[str, List[Path]]:
        """Hiveパーティション構造をスキャン"""
        partition_map = {}
        
        for timeframe in DATA_CONFIG['timeframes']:
            timeframe_dir = self.base_path / f"timeframe={timeframe}"
            if timeframe_dir.exists():
                parquet_files = list(timeframe_dir.glob("*.parquet"))
                partition_map[timeframe] = parquet_files
                logger.debug(f"タイムフレーム {timeframe}: {len(parquet_files)}ファイル")
        
        self.partition_info = partition_map
        return partition_map
    
    @memory_monitor
    def convert_to_memmap(self, timeframes: Optional[List[str]] = None, 
                         force_rebuild: bool = False) -> Dict[str, np.memmap]:
        """ParquetデータをNumPy memmapに変換"""
        if timeframes is None:
            timeframes = ['tick']  # デフォルトはtickデータのみ
        
        memmap_files = {}
        
        for tf in timeframes:
            memmap_path = self.memmap_cache_dir / f"{tf}_data.dat"
            
            # メタファイルの存在もチェック
            meta_path = memmap_path.parent / f"{memmap_path.stem}.meta"
            
            if memmap_path.exists() and meta_path.exists() and not force_rebuild:
                # 既存memmapファイルを読み込み
                try:
                    logger.info(f"既存memmapファイル使用: {tf}")
                    memmap_files[tf] = self._load_existing_memmap(memmap_path)
                except Exception as e:
                    logger.info(f"既存memmapファイルが使用できないため再作成: {tf}")
                    # 不完全なファイルを削除
                    if memmap_path.exists():
                        memmap_path.unlink()
                    if meta_path.exists():
                        meta_path.unlink()
                    memmap_files[tf] = self._create_memmap_from_parquet_streaming(tf, memmap_path)
            else:
                # 新規作成（不完全なファイルがある場合は削除）
                if memmap_path.exists():
                    logger.info(f"不完全なmemmapファイルを削除: {memmap_path}")
                    memmap_path.unlink()
                if meta_path.exists():
                    meta_path.unlink()
                    
                logger.info(f"memmap作成中（ストリーミング処理）: {tf}")
                memmap_files[tf] = self._create_memmap_from_parquet_streaming(tf, memmap_path)
        
        return memmap_files
    
    def _load_existing_memmap(self, memmap_path: Path) -> np.memmap:
        """既存のmemmapファイルを読み込み"""
        try:
            # メタ情報ファイルから形状とdtypeを読み込み
            # ファイル名に基づいてメタファイルパスを構築
            meta_path = memmap_path.parent / f"{memmap_path.stem}.meta"
            if meta_path.exists():
                meta_info = np.load(str(meta_path), allow_pickle=True).item()
                shape = meta_info['shape']
                dtype = meta_info['dtype']
            else:
                raise FileNotFoundError(f"メタ情報ファイルが見つかりません: {meta_path}")
            
            memmap_array = np.memmap(str(memmap_path), dtype=dtype, mode='r', shape=shape)
            logger.debug(f"memmap読み込み完了: shape={shape}, dtype={dtype}")
            return memmap_array
            
        except Exception as e:
            logger.error(f"memmap読み込みエラー: {e}")
            raise
    
    def _create_memmap_from_parquet_streaming(self, timeframe: str, memmap_path: Path) -> np.memmap:
        """
        真のストリーミング処理でParquetからmemmapファイルを作成
        メモリ使用量を最小限に抑制
        """
        try:
            # Parquetファイルパスの取得
            if timeframe in self.partition_info:
                parquet_files = self.partition_info[timeframe]
            else:
                # Hive形式のディレクトリ指定
                timeframe_pattern = str(self.base_path / f"timeframe={timeframe}" / "*.parquet")
                parquet_files = list(Path().glob(timeframe_pattern))
            
            if not parquet_files:
                raise FileNotFoundError(f"タイムフレーム {timeframe} のParquetファイルが見つかりません")
            
            logger.info(f"ストリーミング処理開始: {len(parquet_files)} Parquetファイル")
            
            # 必要な列のみ選択
            required_cols = DATA_CONFIG['required_columns'] + DATA_CONFIG['additional_columns']
            
            # 最初のファイルからメタデータを取得
            first_lf = pl.scan_parquet(str(parquet_files[0]))
            schema = first_lf.collect_schema()
            available_cols = [col for col in required_cols if col in schema.names()]
            
            # 統一dtype
            dtype = np.float64
            n_cols = len(available_cols)
            
            # 全行数を効率的に推定
            total_rows = 0
            batch_size = 50000  # より小さなバッチサイズ
            
            # ファイルごとの行数をカウント（メモリ効率的）
            for file_path in parquet_files:
                try:
                    file_lf = pl.scan_parquet(str(file_path))
                    file_rows = file_lf.select(pl.len()).collect().item()
                    total_rows += file_rows
                except Exception as e:
                    logger.warning(f"ファイル行数取得エラー: {file_path}, {e}")
                    continue
            
            logger.info(f"推定総行数: {total_rows:,}行")
            
            # memmapファイルを事前割り当て
            memmap_array = np.memmap(
                str(memmap_path), 
                dtype=dtype, 
                mode='w+', 
                shape=(total_rows, n_cols)
            )
            
            # ストリーミング処理でファイルごとに処理
            global_offset = 0
            
            for file_idx, file_path in enumerate(parquet_files):
                logger.debug(f"処理中 ({file_idx+1}/{len(parquet_files)}): {file_path.name}")
                
                try:
                    # ファイル単位でLazyFrameを作成
                    file_lf = pl.scan_parquet(str(file_path)).select(available_cols)
                    
                    # ファイル内でバッチごとに処理
                    file_rows = file_lf.select(pl.len()).collect().item()
                    file_offset = 0
                    
                    for start_idx in range(0, file_rows, batch_size):
                        end_idx = min(start_idx + batch_size, file_rows)
                        
                        # バッチ単位でデータを読み込み
                        batch_lf = file_lf.slice(start_idx, end_idx - start_idx)
                        batch_df = batch_lf.collect()
                        
                        # NumPy配列に変換
                        batch_data = batch_df.to_numpy().astype(dtype)
                        batch_rows = batch_data.shape[0]
                        
                        # memmapに書き込み
                        if global_offset + batch_rows > total_rows:
                            # サイズ調整
                            batch_rows = total_rows - global_offset
                            batch_data = batch_data[:batch_rows]
                        
                        memmap_array[global_offset:global_offset + batch_rows] = batch_data
                        global_offset += batch_rows
                        
                        # バッチデータを明示的に削除
                        del batch_df, batch_data
                        
                        # 進捗表示
                        if global_offset % (batch_size * 20) == 0:
                            progress = global_offset / total_rows * 100
                            logger.debug(f"ストリーミング進捗: {progress:.1f}% ({global_offset:,}/{total_rows:,}行)")
                        
                        # メモリクリーンアップ
                        if global_offset % (batch_size * 10) == 0:
                            import gc
                            gc.collect()
                
                except Exception as e:
                    logger.error(f"ファイル処理エラー: {file_path}, {e}")
                    continue
            
            # 実際に使用した行数でmemmapサイズを調整
            if global_offset < total_rows:
                logger.info(f"memmapサイズ調整: {total_rows} -> {global_offset}")
                # 新しいサイズでmemmapを再作成
                temp_path = memmap_path.with_suffix('.tmp')
                final_memmap = np.memmap(
                    str(temp_path), 
                    dtype=dtype, 
                    mode='w+', 
                    shape=(global_offset, n_cols)
                )
                final_memmap[:] = memmap_array[:global_offset]
                final_memmap.flush()
                
                # ファイル置換
                del memmap_array  # 元のmemmapを解放
                temp_path.replace(memmap_path)
                
                # 新しいmemmapで再オープン
                memmap_array = np.memmap(
                    str(memmap_path), 
                    dtype=dtype, 
                    mode='r+', 
                    shape=(global_offset, n_cols)
                )
            
            # メタ情報を保存
            meta_info = {
                'shape': memmap_array.shape,
                'dtype': str(memmap_array.dtype),
                'columns': available_cols,
                'timeframe': timeframe,
                'creation_method': 'streaming'
            }
            
            meta_path = memmap_path.with_suffix('.meta')
            np.save(str(meta_path), meta_info)
            
            # メモリ同期
            memmap_array.flush()
            
            logger.info(f"ストリーミングmemmap作成完了: {timeframe}, shape={memmap_array.shape}")
            return memmap_array
            
        except Exception as e:
            logger.error(f"ストリーミングmemmap作成エラー: {timeframe}, {e}")
            raise
        
        finally:
            # 明示的なメモリクリーンアップ
            import gc
            gc.collect()
    
    def get_basic_info(self, memmap_data: np.memmap) -> Dict[str, Any]:
        """memmapデータの基本情報を取得"""
        return {
            'shape': memmap_data.shape,
            'dtype': memmap_data.dtype,
            'size_gb': memmap_data.nbytes / (1024**3),
            'is_c_contiguous': memmap_data.flags['C_CONTIGUOUS']
        }
    
    def validate_data_integrity(self, memmap_data: np.memmap) -> bool:
        """データ整合性チェック"""
        try:
            # 基本統計でNaN/Inf検出
            sample_size = min(1000, memmap_data.shape[0])
            sample_data = memmap_data[:sample_size]
            
            nan_count = np.isnan(sample_data).sum()
            inf_count = np.isinf(sample_data).sum()
            
            if nan_count > sample_size * 0.1:  # 10%以上のNaN
                logger.warning(f"データ品質警告: NaN率 {nan_count / sample_size * 100:.2f}%")
            
            if inf_count > 0:
                logger.warning(f"データ品質警告: Inf値 {inf_count}個検出")
            
            return nan_count < sample_size * 0.3 and inf_count == 0
            
        except Exception as e:
            logger.error(f"データ整合性チェックエラー: {e}")
            return False
    
    def cleanup_temp_files(self):
        """一時ファイルのクリーンアップ"""
        try:
            temp_files = list(self.memmap_cache_dir.glob("*.tmp"))
            for temp_file in temp_files:
                temp_file.unlink()
                logger.debug(f"一時ファイル削除: {temp_file}")
        except Exception as e:
            logger.warning(f"一時ファイルクリーンアップエラー: {e}")

# =============================================================================
# システム情報表示関数
# =============================================================================

def display_system_info():
    """システム情報を表示"""
    import psutil
    import platform
    
    # CPU情報
    cpu_count = psutil.cpu_count(logical=True)
    cpu_usage = psutil.cpu_percent(interval=1)
    
    # メモリ情報
    memory = psutil.virtual_memory()
    memory_gb = memory.total / (1024**3)
    memory_usage = memory.percent
    
    # ディスク情報
    disk = psutil.disk_usage('/')
    disk_gb = disk.total / (1024**3)
    disk_usage = disk.percent
    
    logger.info(f"""
    ========== システム情報 ==========
    OS: {platform.system()} {platform.release()}
    CPU: {cpu_count}コア, 使用率: {cpu_usage}%
    メモリ: {memory_gb:.1f}GB, 使用率: {memory_usage}%
    ディスク: {disk_gb:.1f}GB, 使用率: {disk_usage}%
    Python: {platform.python_version()}
    NumPy: {np.__version__}
    Pandas: {pd.__version__}
    Polars: {pl.__version__}
    Numba: {numba.__version__}
    ================================
    """)


# ブロック2: WindowManager・MemoryManagerの実装
# 前ブロックに続けて実装

# =============================================================================
# ウィンドウマネージャークラス（薄い実装）
# =============================================================================

class WindowManager:
    """
    ウィンドウ管理クラス - 薄い実装
    シンプルなスライディングウィンドウ生成とインデックス管理
    """
    
    def __init__(self, window_size: int = 100, overlap: float = 0.5):
        self.window_size = window_size
        self.overlap = overlap
        self.step_size = int(window_size * (1 - overlap))
        
        # ウィンドウバッファ
        self.current_window = None
        self.window_index = 0
        self.total_windows = 0
        
        logger.info(f"WindowManager初期化: window_size={window_size}, overlap={overlap}")
    
    def calculate_total_windows(self, data_length: int) -> int:
        """総ウィンドウ数を計算"""
        if data_length < self.window_size:
            return 0
        
        self.total_windows = (data_length - self.window_size) // self.step_size + 1
        return self.total_windows
    
    def generate_window_indices(self, data_length: int) -> List[Tuple[int, int]]:
        """ウィンドウのインデックス範囲を生成"""
        window_indices = []
        
        for i in range(0, data_length - self.window_size + 1, self.step_size):
            start_idx = i
            end_idx = min(i + self.window_size, data_length)
            window_indices.append((start_idx, end_idx))
        
        return window_indices
    
    def create_sliding_windows(self, memmap_data: np.memmap) -> Iterator[Tuple[int, np.ndarray]]:
        """スライディングウィンドウを生成するジェネレータ"""
        data_length = memmap_data.shape[0]
        
        for window_idx, (start_idx, end_idx) in enumerate(self.generate_window_indices(data_length)):
            # memmapから直接ウィンドウデータを取得
            window_data = memmap_data[start_idx:end_idx]
            yield window_idx, window_data
    
    def get_adaptive_window_size(self, data_characteristics: Dict[str, float]) -> int:
        """データ特性に基づく適応的ウィンドウサイズ決定"""
        volatility = data_characteristics.get('volatility', 1.0)
        trend_strength = data_characteristics.get('trend_strength', 0.5)
        
        # ボラティリティが高い場合は小さなウィンドウ
        # トレンド強度が高い場合は大きなウィンドウ
        base_size = self.window_size
        volatility_factor = max(0.5, 1.0 - volatility)
        trend_factor = max(0.8, 1.0 + trend_strength * 0.5)
        
        adaptive_size = int(base_size * volatility_factor * trend_factor)
        return max(50, min(500, adaptive_size))  # 50-500の範囲で制限
    
    def create_overlapping_batches(self, memmap_data: np.memmap, 
                                 batch_size: int = 10) -> Iterator[List[np.ndarray]]:
        """効率化のためのウィンドウバッチ処理"""
        batch_windows = []
        
        for window_idx, window_data in self.create_sliding_windows(memmap_data):
            batch_windows.append(window_data.copy())  # memmapからコピー
            
            if len(batch_windows) >= batch_size:
                yield batch_windows
                batch_windows = []
        
        # 残りのウィンドウ
        if batch_windows:
            yield batch_windows
    
    def validate_window_integrity(self, window_data: np.ndarray) -> bool:
        """ウィンドウデータの整合性チェック"""
        if window_data.size == 0:
            return False
        
        # 基本的な数値チェック
        if np.isnan(window_data).sum() > window_data.size * 0.3:  # 30%以上のNaN
            logger.warning("ウィンドウにNaNが多すぎます")
            return False
        
        if np.isinf(window_data).any():
            logger.warning("ウィンドウにInf値が含まれています")
            return False
        
        return True

# =============================================================================
# メモリマネージャークラス（監視のみ）
# =============================================================================

class MemoryManager:
    """
    メモリ管理クラス - 監視のみ
    RTX 3060 12GB GPU + 64GB RAM の監視と警告
    """
    
    def __init__(self):
        self.ram_limit_gb = HARDWARE_SPEC['ram_limit']
        self.gpu_memory_gb = 12  # RTX 3060
        self.warning_threshold = 0.8  # 80%で警告
        self.critical_threshold = 0.9  # 90%で危険
        
        # 監視統計
        self.peak_memory_usage = 0.0
        self.memory_warnings = 0
        self.gpu_available = self._check_gpu_availability()
        
        logger.info(f"MemoryManager初期化: RAM限界={self.ram_limit_gb}GB, GPU={'利用可能' if self.gpu_available else '利用不可'}")
    
    def _check_gpu_availability(self) -> bool:
        """GPU利用可能性をチェック"""
        try:
            import torch
            return torch.cuda.is_available()
        except ImportError:
            try:
                import cupy
                return True
            except ImportError:
                return False
    
    @property
    def current_memory_usage(self) -> float:
        """現在のメモリ使用量（GB）"""
        import psutil
        return psutil.Process().memory_info().rss / (1024**3)
    
    @property
    def system_memory_usage(self) -> float:
        """システム全体のメモリ使用量（GB）"""
        import psutil
        return psutil.virtual_memory().used / (1024**3)
    
    @property
    def memory_usage_percent(self) -> float:
        """メモリ使用率（%）"""
        return (self.current_memory_usage / self.ram_limit_gb) * 100
    
    def check_memory_status(self) -> Dict[str, Any]:
        """メモリ状態をチェック"""
        current_gb = self.current_memory_usage
        usage_percent = self.memory_usage_percent
        
        # ピーク使用量更新
        if current_gb > self.peak_memory_usage:
            self.peak_memory_usage = current_gb
        
        status = {
            'current_gb': current_gb,
            'peak_gb': self.peak_memory_usage,
            'usage_percent': usage_percent,
            'status': 'normal'
        }
        
        # 警告レベル判定
        if usage_percent > self.critical_threshold * 100:
            status['status'] = 'critical'
            logger.error(f"⚠️ 危険: メモリ使用量 {current_gb:.2f}GB ({usage_percent:.1f}%)")
        elif usage_percent > self.warning_threshold * 100:
            status['status'] = 'warning'
            logger.warning(f"⚠️ 警告: メモリ使用量 {current_gb:.2f}GB ({usage_percent:.1f}%)")
            self.memory_warnings += 1
        
        return status
    
    def get_gpu_memory_status(self) -> Optional[Dict[str, Any]]:
        """GPU メモリ状態を取得"""
        if not self.gpu_available:
            return None
        
        try:
            import torch
            if torch.cuda.is_available():
                device = torch.cuda.current_device()
                total_memory = torch.cuda.get_device_properties(device).total_memory / (1024**3)
                allocated_memory = torch.cuda.memory_allocated(device) / (1024**3)
                cached_memory = torch.cuda.memory_reserved(device) / (1024**3)
                
                return {
                    'total_gb': total_memory,
                    'allocated_gb': allocated_memory,
                    'cached_gb': cached_memory,
                    'free_gb': total_memory - cached_memory,
                    'usage_percent': (cached_memory / total_memory) * 100
                }
        except Exception as e:
            logger.debug(f"GPU メモリ状態取得エラー: {e}")
            return None
    
    def suggest_memory_optimization(self) -> List[str]:
        """メモリ最適化提案"""
        suggestions = []
        current_usage = self.memory_usage_percent
        
        if current_usage > 70:
            suggestions.append("ウィンドウサイズを小さくする")
            suggestions.append("バッチ処理サイズを減らす")
            suggestions.append("中間結果のキャッシュを制限する")
        
        if current_usage > 80:
            suggestions.append("データを更に小さなチャンクに分割する")
            suggestions.append("一部の特徴量計算を無効化する")
        
        if current_usage > 90:
            suggestions.append("緊急: 処理を一時停止してメモリを解放する")
            suggestions.append("システム再起動を検討する")
        
        return suggestions
    
    def force_garbage_collection(self):
        """強制ガベージコレクション"""
        import gc
        import psutil
        
        before_memory = self.current_memory_usage
        
        # Python ガベージコレクション
        gc.collect()
        
        # GPU メモリクリア（利用可能な場合）
        if self.gpu_available:
            try:
                import torch
                torch.cuda.empty_cache()
            except:
                pass
        
        after_memory = self.current_memory_usage
        freed_memory = before_memory - after_memory
        
        if freed_memory > 0.1:  # 100MB以上解放された場合
            logger.info(f"ガベージコレクション完了: {freed_memory:.2f}GB解放")
        
        return freed_memory
    
    def monitor_continuous(self, duration_seconds: int = 300):
        """継続的メモリ監視（別スレッドで実行）"""
        import threading
        import time
        
        def monitoring_loop():
            start_time = time.time()
            while time.time() - start_time < duration_seconds:
                status = self.check_memory_status()
                if status['status'] in ['warning', 'critical']:
                    self.force_garbage_collection()
                
                time.sleep(10)  # 10秒間隔
        
        monitor_thread = threading.Thread(target=monitoring_loop, daemon=True)
        monitor_thread.start()
        logger.info(f"継続的メモリ監視開始: {duration_seconds}秒間")
        
        return monitor_thread
    
    def get_memory_summary(self) -> Dict[str, Any]:
        """メモリ使用統計サマリー"""
        cpu_status = self.check_memory_status()
        gpu_status = self.get_gpu_memory_status()
        
        summary = {
            'cpu_memory': cpu_status,
            'gpu_memory': gpu_status,
            'peak_usage_gb': self.peak_memory_usage,
            'total_warnings': self.memory_warnings,
            'optimization_suggestions': self.suggest_memory_optimization()
        }
        
        return summary
    
    def log_memory_report(self):
        """メモリ使用レポートをログ出力"""
        summary = self.get_memory_summary()
        
        logger.info(f"""
        ========== メモリ使用レポート ==========
        CPU メモリ: {summary['cpu_memory']['current_gb']:.2f}GB / {self.ram_limit_gb}GB ({summary['cpu_memory']['usage_percent']:.1f}%)
        ピーク使用量: {summary['peak_usage_gb']:.2f}GB
        警告回数: {summary['total_warnings']}回
        """)
        
        if summary['gpu_memory']:
            gpu = summary['gpu_memory']
            logger.info(f"GPU メモリ: {gpu['allocated_gb']:.2f}GB / {gpu['total_gb']:.2f}GB ({gpu['usage_percent']:.1f}%)")
        
        if summary['optimization_suggestions']:
            logger.info("最適化提案:")
            for suggestion in summary['optimization_suggestions']:
                logger.info(f"  - {suggestion}")

# =============================================================================
# 進捗表示とエラーハンドリング
# =============================================================================

class ProgressTracker:
    """進捗追跡とエラーハンドリング"""
    
    def __init__(self, total_steps: int, feature_groups: List[str]):
        self.total_steps = total_steps
        self.feature_groups = feature_groups
        self.current_step = 0
        self.start_time = time.time()
        self.step_times = []
        
        # エラー統計
        self.failed_features = []
        self.warning_count = 0
        self.success_count = 0
        
        logger.info(f"ProgressTracker初期化: {total_steps}ステップ, {len(feature_groups)}特徴量グループ")
    
    def update_progress(self, step_name: str, features_generated: int = 0, 
                       has_error: bool = False, warning_msg: str = None):
        """進捗更新"""
        self.current_step += 1
        current_time = time.time()
        step_duration = current_time - (self.step_times[-1] if self.step_times else self.start_time)
        self.step_times.append(current_time)
        
        # 統計更新
        if has_error:
            self.failed_features.append(step_name)
        else:
            self.success_count += features_generated
        
        if warning_msg:
            self.warning_count += 1
            logger.warning(f"警告 - {step_name}: {warning_msg}")
        
        # 残り時間計算
        if self.current_step > 1:
            avg_time_per_step = (current_time - self.start_time) / self.current_step
            remaining_steps = self.total_steps - self.current_step
            estimated_remaining = remaining_steps * avg_time_per_step / 60  # 分
        else:
            estimated_remaining = 0
        
        # 進捗表示
        progress_pct = (self.current_step / self.total_steps) * 100
        logger.info(f"⚡ 進捗: {progress_pct:.1f}% ({self.current_step}/{self.total_steps}) - "
                   f"{step_name} 完了: {features_generated}特徴量 "
                   f"(残り約{estimated_remaining:.1f}分)")
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """パフォーマンスメトリクス"""
        if not self.step_times:
            return {}
        
        total_time = time.time() - self.start_time
        avg_step_time = total_time / max(1, self.current_step)
        
        return {
            'total_time_minutes': total_time / 60,
            'avg_step_time_seconds': avg_step_time,
            'success_rate': (self.current_step - len(self.failed_features)) / max(1, self.current_step) * 100,
            'features_per_minute': self.success_count / max(1, total_time / 60),
            'failed_features': self.failed_features,
            'warning_count': self.warning_count
        }
    
    def log_final_summary(self):
        """最終サマリーログ出力"""
        metrics = self.get_performance_metrics()
        
        logger.info(f"""
        ========== 処理完了サマリー ==========
        総処理時間: {metrics['total_time_minutes']:.1f}分
        成功率: {metrics['success_rate']:.1f}%
        生成特徴量数: {self.success_count}個
        生成速度: {metrics['features_per_minute']:.1f}特徴量/分
        警告回数: {metrics['warning_count']}回
        失敗した特徴量: {len(metrics['failed_features'])}個
        =======================================
        """)
        
        if metrics['failed_features']:
            logger.warning("失敗した特徴量グループ:")
            for failed in metrics['failed_features']:
                logger.warning(f"  - {failed}")

# =============================================================================
# 設定とバリデーション
# =============================================================================

def validate_system_requirements() -> bool:
    """システム要件チェック"""
    import psutil
    
    # メモリチェック（WSL2環境を考慮）
    total_memory_gb = psutil.virtual_memory().total / (1024**3)
    
    # WSL2環境では実際より少なく報告されることがあるため、閾値を調整
    min_memory_threshold = 16  # 32GBから16GBに調整
    
    if total_memory_gb < min_memory_threshold:
        logger.error(f"メモリ不足: {total_memory_gb:.1f}GB < {min_memory_threshold}GB 必要")
        return False
    elif total_memory_gb < 32:
        logger.warning(f"メモリ警告: {total_memory_gb:.1f}GB < 32GB 推奨（WSL2環境では正常）")
    
    # CPU チェック
    cpu_count = psutil.cpu_count()
    if cpu_count is not None and cpu_count < 4:
        logger.error(f"CPU不足: {cpu_count}コア < 4コア推奨")
        return False
    elif cpu_count is None:
        logger.warning("CPU情報を取得できませんでした")
        return False
    
    # ディスク容量チェック
    disk_free_gb = psutil.disk_usage('/').free / (1024**3)
    if disk_free_gb < 50:
        logger.error(f"ディスク容量不足: {disk_free_gb:.1f}GB < 50GB推奨")
        return False
    
    logger.info("システム要件チェック: ✓ 全て合格")
    return True

def optimize_numpy_settings():
    """NumPy 最適化設定"""
    # OpenBLAS/MKL スレッド数設定
    os.environ['OMP_NUM_THREADS'] = str(HARDWARE_SPEC['cpu_cores'])
    os.environ['MKL_NUM_THREADS'] = str(HARDWARE_SPEC['cpu_cores'])
    os.environ['OPENBLAS_NUM_THREADS'] = str(HARDWARE_SPEC['cpu_cores'])
    
    # Numba 設定
    os.environ['NUMBA_NUM_THREADS'] = str(HARDWARE_SPEC['cpu_cores'])
    os.environ['NUMBA_CACHE_DIR'] = '/tmp/numba_cache'
    
    logger.info(f"NumPy最適化設定完了: {HARDWARE_SPEC['cpu_cores']}スレッド")


# ブロック3: Calculator基礎計算部の実装（核心部分80%リソースの一部）

# =============================================================================
# Calculator クラス - Polars v1.31.0制約対応版（ブロック1）
# =============================================================================
import numpy as np
import polars as pl
import pandas as pd
from numba import njit, prange
import time
from typing import Dict, Any, Tuple, List, Union
from scipy import stats
from scipy.signal import hilbert, coherence, periodogram
from scipy.ndimage import gaussian_filter, median_filter
from scipy.signal import savgol_filter
import pywt
from numpy.fft import fft, fftfreq
import logging

logger = logging.getLogger(__name__)

# エントロピー関連ライブラリ（オプション）
try:
    import entropy as ent
    ENTROPY_AVAILABLE = True
except ImportError:
    ENTROPY_AVAILABLE = False
    logger.warning("entropy ライブラリが利用できません。一部の特徴量は計算されません。")

# PyEMD（オプション）
try:
    from PyEMD import EMD
    PYEMD_AVAILABLE = True
except ImportError:
    PYEMD_AVAILABLE = False
    logger.warning("PyEMD ライブラリが利用できません。EMD特徴量は代替手法で計算されます。")


class Calculator:
    """
    Polars v1.31.0制約対応Calculator
    
    制約遵守原則:
    1. Expression での cumsum(), rolling_*() は使用禁止
    2. Boolean操作の連鎖はNumPyで実行
    3. Polarsは基本集約とto_numpy()変換のみ
    4. 数値計算・統計処理は全てNumPyで実装
    """
    
    def __init__(self, window_manager=None, memory_manager=None):
        self.window_manager = window_manager
        self.memory_manager = memory_manager
        
        # 計算パラメータ
        self.params = self._initialize_params()
        
        # 数値安定性パラメータ
        self.numerical_stability = {
            'eps': 1e-12,
            'condition_number_threshold': 1e12,
            'outlier_threshold': 5.0,
            'min_valid_ratio': 0.7
        }
        
        # 数学的定数
        self.mathematical_constants = {
            'golden_ratio': (1 + np.sqrt(5)) / 2,
            'euler_constant': np.e,
            'pi': np.pi,
            'sqrt_2': np.sqrt(2),
            'sqrt_3': np.sqrt(3),
        }
        
        # 統計情報
        self.calculation_stats = {
            'total_calculations': 0,
            'successful_calculations': 0,
            'numpy_calculations': 0,
            'numba_calculations': 0,
            'computation_times': []
        }
        
        logger.info("Calculator初期化完了 - Polars v1.31.0制約対応版")

    def _initialize_params(self) -> Dict[str, Any]:
        """パラメータセット初期化"""
        return {
            # 基本テクニカル指標用期間
            'rsi_periods': [14, 21, 30],
            'macd_periods': [(12, 26, 9), (8, 17, 9), (19, 39, 9)],
            'bb_periods': [20, 50],
            'bb_std_multipliers': [2.0, 2.5],
            'atr_periods': [14, 20, 50],
            'stoch_periods': [(14, 3), (21, 5)],
            
            # 高度特徴量用期間
            'mfdfa_scales': np.logspace(1, 2.5, 15).astype(int),
            'mfdfa_q_range': np.arange(-5, 6, 1),
            'microstructure_windows': [50, 100, 200],
            'shock_model_windows': [30, 50, 100],
            'multiscale_volatility_periods': [5, 10, 20, 50, 100],
            
            # EMD・信号処理用
            'emd_windows': [100, 200],
            'wavelet_windows': [64, 128, 256],
            'hilbert_windows': [50, 100],
            
            # 統計・エントロピー用
            'entropy_windows': [50, 100],
            'order_stat_windows': [20, 50, 100],
            'autocorr_max_lags': [10, 20],
            
            # その他の期間パラメータ
            'volatility_bb_settings': [(20, 2.0), (50, 2.5)],
            'kc_periods': [20, 50],
            'dc_periods': [20, 50, 100],
            'atr_periods_vol': [14, 20],
            'hist_vol_periods': [20, 50, 100],
            'ma_deviation_periods': [20, 50, 100],
            'tma_periods': [14, 21, 50],
            'zlema_periods': [12, 21],
            'dema_periods': [12, 21],
            'tema_periods': [12, 21],
            'cmf_periods': [20, 50],
            'mfi_periods': [14, 21],
            'vol_roc_periods': [10, 20],
            'adx_periods': [14, 21],
            'cci_periods': [14, 20],
            'williams_r_periods': [14, 21],
            'aroon_periods': [14, 25],
            'price_channel_periods': [20, 50, 100],
            'gaussian_sigmas': [1.0, 2.0, 3.0],
            'median_sizes': [3, 5, 7],
            'savgol_windows': [5, 11, 21],
            'fourier_windows': [32, 64, 128],
            'spectral_windows': [64, 128],
            'energy_windows': [20, 50],
            'cwt_windows': [100, 200],
            'lz_windows': [50, 100],
            'kolmogorov_windows': [50, 100],
            'mutual_info_lags': [1, 2, 3, 5],
            'adv_entropy_windows': [50, 100]
        }

    # =========================================================================
    # 基本ユーティリティメソッド（NumPy中心）
    # =========================================================================
    
    def _ensure_numpy_array(self, data: Union[np.ndarray, pl.DataFrame, List]) -> np.ndarray:
        """データをNumPy配列に変換"""
        if isinstance(data, np.ndarray):
            return data.flatten() if data.ndim > 1 else data
        elif isinstance(data, pl.DataFrame):
            if len(data.columns) > 0:
                return data[data.columns[0]].to_numpy()
            else:
                return np.array([])
        elif isinstance(data, (list, tuple)):
            return np.asarray(data)
        else:
            return np.asarray([data])
    
    def _safe_calculation(self, func, *args, **kwargs) -> Union[np.ndarray, Dict, Any]:
        """計算の安全なラッパー"""
        start_time = time.time()
        self.calculation_stats['total_calculations'] += 1
        
        try:
            result = func(*args, **kwargs)
            self.calculation_stats['successful_calculations'] += 1
            self.calculation_stats['numpy_calculations'] += 1
            
            computation_time = time.time() - start_time
            self.calculation_stats['computation_times'].append(computation_time)
            
            return result
            
        except Exception as e:
            logger.debug(f"計算エラー in {func.__name__}: {e}")
            # フォールバック値を返す
            if args and hasattr(args[0], '__len__'):
                return np.zeros(len(args[0]))
            return np.array([])

    def _numba_safe_calculation(self, func, *args, **kwargs) -> np.ndarray:
        """Numba計算の安全なラッパー"""
        start_time = time.time()
        self.calculation_stats['total_calculations'] += 1
        
        try:
            result = func(*args, **kwargs)
            self.calculation_stats['successful_calculations'] += 1
            self.calculation_stats['numba_calculations'] += 1
            
            computation_time = time.time() - start_time
            self.calculation_stats['computation_times'].append(computation_time)
            
            return result
            
        except Exception as e:
            logger.debug(f"Numba計算エラー in {func.__name__}: {e}")
            # フォールバック値
            if args and hasattr(args[0], '__len__'):
                return np.zeros(len(args[0]))
            return np.array([])
    
    # =========================================================================
    # ADX・基本オシレーター（NumPy実装）
    # =========================================================================
    
    def calculate_adx_features(self, high: np.ndarray, low: np.ndarray, close: np.ndarray) -> Dict[str, np.ndarray]:
        """ADX関連特徴量の計算（NumPy実装）"""
        features = {}
        
        # NumPy配列に変換
        high = self._ensure_numpy_array(high)
        low = self._ensure_numpy_array(low)
        close = self._ensure_numpy_array(close)
        
        try:
            for period in self.params['adx_periods']:
                # ADXを計算
                adx_result = self._safe_calculation(
                    self._calculate_adx_numpy,
                    high, low, close, period
                )
                
                if isinstance(adx_result, dict):
                    features.update(adx_result)
                else:
                    # フォールバック
                    n = len(close)
                    features[f'adx_{period}'] = np.full(n, np.nan)
                    features[f'di_plus_{period}'] = np.full(n, np.nan)
                    features[f'di_minus_{period}'] = np.full(n, np.nan)
                    features[f'di_diff_{period}'] = np.full(n, np.nan)
                    features[f'adx_strength_{period}'] = np.full(n, np.nan)
                    features[f'trend_strength_{period}'] = np.full(n, np.nan)
                    
        except Exception as e:
            logger.error(f"ADX計算エラー: {e}")
            n = len(close)
            for period in self.params['adx_periods']:
                features[f'adx_{period}'] = np.full(n, np.nan)
                features[f'di_plus_{period}'] = np.full(n, np.nan)
                features[f'di_minus_{period}'] = np.full(n, np.nan)
                features[f'di_diff_{period}'] = np.full(n, np.nan)
                features[f'adx_strength_{period}'] = np.full(n, np.nan)
                features[f'trend_strength_{period}'] = np.full(n, np.nan)
        
        return features
    
    def _calculate_adx_numpy(self, high: np.ndarray, low: np.ndarray, 
                           close: np.ndarray, period: int) -> Dict[str, np.ndarray]:
        """ADX計算（NumPy実装）"""
        
        if len(close) < 2:
            n = len(close)
            return {
                f'adx_{period}': np.full(n, np.nan),
                f'di_plus_{period}': np.full(n, np.nan),
                f'di_minus_{period}': np.full(n, np.nan),
                f'di_diff_{period}': np.full(n, np.nan),
                f'adx_strength_{period}': np.full(n, np.nan),
                f'trend_strength_{period}': np.full(n, np.nan)
            }
        
        # 前日終値
        prev_close = np.roll(close, 1)
        prev_close[0] = close[0]
        
        # True Range計算
        tr1 = high - low
        tr2 = np.abs(high - prev_close)
        tr3 = np.abs(low - prev_close)
        true_range = np.maximum(tr1, np.maximum(tr2, tr3))
        
        # 方向性移動計算
        up_move = high - np.roll(high, 1)
        down_move = np.roll(low, 1) - low
        up_move[0] = 0
        down_move[0] = 0
        
        # DM+とDM-
        dm_plus = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
        dm_minus = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
        
        # スムージング（指数移動平均）
        atr_smooth = self._ema_numpy(true_range, period)
        dm_plus_smooth = self._ema_numpy(dm_plus, period)
        dm_minus_smooth = self._ema_numpy(dm_minus, period)
        
        # DI+とDI-の計算
        di_plus = 100.0 * np.divide(dm_plus_smooth, atr_smooth, 
                                   out=np.zeros_like(atr_smooth), where=(atr_smooth != 0))
        di_minus = 100.0 * np.divide(dm_minus_smooth, atr_smooth,
                                    out=np.zeros_like(atr_smooth), where=(atr_smooth != 0))
        
        # DXの計算
        di_sum = di_plus + di_minus
        di_diff_abs = np.abs(di_plus - di_minus)
        dx = 100.0 * np.divide(di_diff_abs, di_sum, 
                              out=np.zeros_like(di_sum), where=(di_sum != 0))
        
        # ADXの計算（指数移動平均）
        adx = self._ema_numpy(dx, period)
        
        # 追加特徴量（NumPy操作）
        di_diff = di_plus - di_minus
        adx_strength = np.where(adx > 25, 1.0, 0.0)
        trend_strength = adx / 100.0
        
        return {
            f'adx_{period}': adx,
            f'di_plus_{period}': di_plus,
            f'di_minus_{period}': di_minus,
            f'di_diff_{period}': di_diff,
            f'adx_strength_{period}': adx_strength,
            f'trend_strength_{period}': trend_strength
        }
    
    def calculate_parabolic_sar_features(self, high: np.ndarray, low: np.ndarray, close: np.ndarray) -> Dict[str, np.ndarray]:
        """パラボリックSAR特徴量の計算（NumPy実装）"""
        features = {}
        
        # NumPy配列に変換
        high = self._ensure_numpy_array(high)
        low = self._ensure_numpy_array(low)
        close = self._ensure_numpy_array(close)
        
        try:
            # NumbaでSAR計算
            sar_result = self._numba_safe_calculation(
                self._calculate_parabolic_sar_vectorized,
                high, low, close
            )
            
            if sar_result is not None and len(sar_result.shape) == 2:
                sar_values = sar_result[:, 0]
                signal_values = sar_result[:, 1]
                
                # NumPyで追加特徴量を計算
                sar_distance = (close - sar_values) / (close + 1e-10)
                sar_above = np.where(close > sar_values, 1.0, 0.0)
                sar_trend_strength = np.abs(signal_values)
                
                features['parabolic_sar'] = sar_values
                features['sar_signal'] = signal_values
                features['sar_distance'] = sar_distance
                features['sar_above'] = sar_above
                features['sar_trend_strength'] = sar_trend_strength
            else:
                # フォールバック
                n = len(close)
                features['parabolic_sar'] = np.full(n, np.nan)
                features['sar_signal'] = np.full(n, np.nan)
                features['sar_distance'] = np.full(n, np.nan)
                features['sar_above'] = np.full(n, np.nan)
                features['sar_trend_strength'] = np.full(n, np.nan)
                
        except Exception as e:
            logger.error(f"Parabolic SAR計算エラー: {e}")
            n = len(close)
            features['parabolic_sar'] = np.full(n, np.nan)
            features['sar_signal'] = np.full(n, np.nan)
            features['sar_distance'] = np.full(n, np.nan)
            features['sar_above'] = np.full(n, np.nan)
            features['sar_trend_strength'] = np.full(n, np.nan)
        
        return features
    
    @staticmethod
    @njit(cache=True)
    def _calculate_parabolic_sar_vectorized(high: np.ndarray, low: np.ndarray, close: np.ndarray) -> np.ndarray:
        """パラボリックSAR計算（Numba最適化版）"""
        n = len(high)
        if n < 2:
            return np.zeros((n, 2))
        
        sar = np.zeros(n)
        signal = np.zeros(n)
        
        # 初期設定
        af_start = 0.02
        af_max = 0.2
        
        sar[0] = low[0]
        ep = high[0]  # Extreme Point
        af = af_start
        is_uptrend = True
        
        for i in range(1, n):
            # SAR更新
            sar[i] = sar[i-1] + af * (ep - sar[i-1])
            
            if is_uptrend:
                # 上昇トレンド
                if low[i] <= sar[i]:
                    # トレンド転換
                    is_uptrend = False
                    sar[i] = ep
                    ep = low[i]
                    af = af_start
                    signal[i] = -1
                else:
                    if high[i] > ep:
                        ep = high[i]
                        af = min(af + af_start, af_max)
                    signal[i] = 1
            else:
                # 下降トレンド
                if high[i] >= sar[i]:
                    # トレンド転換
                    is_uptrend = True
                    sar[i] = ep
                    ep = high[i]
                    af = af_start
                    signal[i] = 1
                else:
                    if low[i] < ep:
                        ep = low[i]
                        af = min(af + af_start, af_max)
                    signal[i] = -1
        
        result = np.zeros((n, 2))
        result[:, 0] = sar
        result[:, 1] = signal
        return result
    
    def calculate_cci_features(self, high: np.ndarray, low: np.ndarray, close: np.ndarray) -> Dict[str, np.ndarray]:
        """CCI特徴量の計算（NumPy実装）"""
        features = {}
        
        # NumPy配列に変換
        high = self._ensure_numpy_array(high)
        low = self._ensure_numpy_array(low)
        close = self._ensure_numpy_array(close)
        
        try:
            for period in self.params['cci_periods']:
                # CCIを計算
                cci_result = self._safe_calculation(
                    self._calculate_cci_numpy,
                    high, low, close, period
                )
                
                if isinstance(cci_result, dict):
                    features.update(cci_result)
                else:
                    # フォールバック
                    n = len(close)
                    features[f'cci_{period}'] = np.full(n, np.nan)
                    features[f'cci_overbought_{period}'] = np.full(n, np.nan)
                    features[f'cci_oversold_{period}'] = np.full(n, np.nan)
                    features[f'cci_normalized_{period}'] = np.full(n, np.nan)
                    
        except Exception as e:
            logger.error(f"CCI計算エラー: {e}")
            n = len(close)
            for period in self.params['cci_periods']:
                features[f'cci_{period}'] = np.full(n, np.nan)
                features[f'cci_overbought_{period}'] = np.full(n, np.nan)
                features[f'cci_oversold_{period}'] = np.full(n, np.nan)
                features[f'cci_normalized_{period}'] = np.full(n, np.nan)
        
        return features
    
    def _calculate_cci_numpy(self, high: np.ndarray, low: np.ndarray, 
                           close: np.ndarray, period: int) -> Dict[str, np.ndarray]:
        """CCI計算（NumPy実装）"""
        
        # Typical Price計算
        typical_price = (high + low + close) / 3.0
        
        # SMAとMAD計算
        sma_tp = self._rolling_mean_numpy(typical_price, period)
        
        # MAD計算（平均絶対偏差）
        mad = np.full_like(typical_price, np.nan)
        for i in range(period-1, len(typical_price)):
            window_tp = typical_price[i-period+1:i+1]
            mad[i] = np.mean(np.abs(window_tp - sma_tp[i]))
        
        # CCI計算
        cci = (typical_price - sma_tp) / (0.015 * mad + 1e-10)
        
        # 追加特徴量（NumPy操作）
        cci_overbought = np.where(cci > 100, 1.0, 0.0)
        cci_oversold = np.where(cci < -100, 1.0, 0.0)
        cci_normalized = np.tanh(cci / 100.0)
        
        return {
            f'cci_{period}': cci,
            f'cci_overbought_{period}': cci_overbought,
            f'cci_oversold_{period}': cci_oversold,
            f'cci_normalized_{period}': cci_normalized
        }
    
    def calculate_williams_r_features(self, high: np.ndarray, low: np.ndarray, close: np.ndarray) -> Dict[str, np.ndarray]:
        """ウィリアムズ%R特徴量の計算（NumPy実装）"""
        features = {}
        
        # NumPy配列に変換
        high = self._ensure_numpy_array(high)
        low = self._ensure_numpy_array(low)
        close = self._ensure_numpy_array(close)
        
        try:
            for period in self.params['williams_r_periods']:
                # Williams %Rを計算
                williams_r_result = self._safe_calculation(
                    self._calculate_williams_r_numpy,
                    high, low, close, period
                )
                
                if isinstance(williams_r_result, dict):
                    features.update(williams_r_result)
                else:
                    # フォールバック
                    n = len(close)
                    features[f'williams_r_{period}'] = np.full(n, np.nan)
                    features[f'williams_r_overbought_{period}'] = np.full(n, np.nan)
                    features[f'williams_r_oversold_{period}'] = np.full(n, np.nan)
                    features[f'williams_r_normalized_{period}'] = np.full(n, np.nan)
                    
        except Exception as e:
            logger.error(f"Williams %R計算エラー: {e}")
            n = len(close)
            for period in self.params['williams_r_periods']:
                features[f'williams_r_{period}'] = np.full(n, np.nan)
                features[f'williams_r_overbought_{period}'] = np.full(n, np.nan)
                features[f'williams_r_oversold_{period}'] = np.full(n, np.nan)
                features[f'williams_r_normalized_{period}'] = np.full(n, np.nan)
        
        return features
    
    def _calculate_williams_r_numpy(self, high: np.ndarray, low: np.ndarray, 
                                  close: np.ndarray, period: int) -> Dict[str, np.ndarray]:
        """Williams %R計算（NumPy実装）"""
        
        # 最高値・最安値計算
        highest_high = self._rolling_max_numpy(high, period)
        lowest_low = self._rolling_min_numpy(low, period)
        
        # Williams %R計算
        williams_r = -100.0 * np.divide(
            highest_high - close,
            highest_high - lowest_low,
            out=np.zeros_like(close),
            where=((highest_high - lowest_low) != 0)
        )
        
        # 追加特徴量（NumPy操作）
        williams_r_overbought = np.where(williams_r > -20, 1.0, 0.0)
        williams_r_oversold = np.where(williams_r < -80, 1.0, 0.0)
        williams_r_normalized = (williams_r + 50) / 50.0
        
        return {
            f'williams_r_{period}': williams_r,
            f'williams_r_overbought_{period}': williams_r_overbought,
            f'williams_r_oversold_{period}': williams_r_oversold,
            f'williams_r_normalized_{period}': williams_r_normalized
        }
    
    def calculate_aroon_features(self, high: np.ndarray, low: np.ndarray) -> Dict[str, np.ndarray]:
        """アルーン特徴量の計算（NumPy実装）"""
        features = {}
        
        # NumPy配列に変換
        high = self._ensure_numpy_array(high)
        low = self._ensure_numpy_array(low)
        
        try:
            for period in self.params['aroon_periods']:
                # AroonをNumbaで計算
                aroon_result = self._numba_safe_calculation(
                    self._calculate_aroon_vectorized,
                    high, low, period
                )
                
                if aroon_result is not None and len(aroon_result.shape) == 2:
                    aroon_up = aroon_result[:, 0]
                    aroon_down = aroon_result[:, 1]
                    
                    # NumPyで追加特徴量を計算
                    aroon_oscillator = aroon_up - aroon_down
                    aroon_trending = np.where(np.abs(aroon_oscillator) > 50, 1.0, 0.0)
                    
                    features[f'aroon_up_{period}'] = aroon_up
                    features[f'aroon_down_{period}'] = aroon_down
                    features[f'aroon_oscillator_{period}'] = aroon_oscillator
                    features[f'aroon_trending_{period}'] = aroon_trending
                else:
                    # フォールバック
                    n = len(high)
                    features[f'aroon_up_{period}'] = np.full(n, np.nan)
                    features[f'aroon_down_{period}'] = np.full(n, np.nan)
                    features[f'aroon_oscillator_{period}'] = np.full(n, np.nan)
                    features[f'aroon_trending_{period}'] = np.full(n, np.nan)
                    
        except Exception as e:
            logger.error(f"Aroon計算エラー: {e}")
            n = len(high)
            for period in self.params['aroon_periods']:
                features[f'aroon_up_{period}'] = np.full(n, np.nan)
                features[f'aroon_down_{period}'] = np.full(n, np.nan)
                features[f'aroon_oscillator_{period}'] = np.full(n, np.nan)
                features[f'aroon_trending_{period}'] = np.full(n, np.nan)
        
        return features
    
    @staticmethod
    @njit(parallel=True, cache=True)
    def _calculate_aroon_vectorized(high: np.ndarray, low: np.ndarray, period: int) -> np.ndarray:
        """アルーン指標計算（完全ベクトル化・並列版）"""
        n = len(high)
        if n < period:
            return np.zeros((n, 2))
        
        aroon_up = np.zeros(n)
        aroon_down = np.zeros(n)
        
        # 並列計算
        for i in prange(period-1, n):
            window_high = high[i-period+1:i+1]
            window_low = low[i-period+1:i+1]
            
            # 最高値・最安値の位置を検索
            high_idx = 0
            low_idx = 0
            
            for j in range(len(window_high)):
                if window_high[j] >= window_high[high_idx]:
                    high_idx = j
                if window_low[j] <= window_low[low_idx]:
                    low_idx = j
            
            # アルーン計算
            aroon_up[i] = 100.0 * (period - (period - 1 - high_idx)) / period
            aroon_down[i] = 100.0 * (period - (period - 1 - low_idx)) / period
        
        result = np.zeros((n, 2))
        result[:, 0] = aroon_up
        result[:, 1] = aroon_down
        return result
    
    def calculate_ultimate_oscillator_features(self, high: np.ndarray, low: np.ndarray, close: np.ndarray) -> Dict[str, np.ndarray]:
        """アルティメットオシレーター特徴量の計算（NumPy実装）"""
        features = {}
        
        # NumPy配列に変換
        high = self._ensure_numpy_array(high)
        low = self._ensure_numpy_array(low)
        close = self._ensure_numpy_array(close)
        
        try:
            # Ultimate OscillatorをNumbaで計算
            uo_result = self._numba_safe_calculation(
                self._calculate_ultimate_oscillator_vectorized,
                high, low, close
            )
            
            if uo_result is not None:
                # NumPyで追加特徴量を計算
                uo_overbought = np.where(uo_result > 70, 1.0, 0.0)
                uo_oversold = np.where(uo_result < 30, 1.0, 0.0)
                uo_normalized = (uo_result - 50) / 50.0
                
                features['ultimate_oscillator'] = uo_result
                features['uo_overbought'] = uo_overbought
                features['uo_oversold'] = uo_oversold
                features['uo_normalized'] = uo_normalized
            else:
                # フォールバック
                n = len(high)
                features['ultimate_oscillator'] = np.full(n, np.nan)
                features['uo_overbought'] = np.full(n, np.nan)
                features['uo_oversold'] = np.full(n, np.nan)
                features['uo_normalized'] = np.full(n, np.nan)
                
        except Exception as e:
            logger.error(f"Ultimate Oscillator計算エラー: {e}")
            n = len(high)
            features['ultimate_oscillator'] = np.full(n, np.nan)
            features['uo_overbought'] = np.full(n, np.nan)
            features['uo_oversold'] = np.full(n, np.nan)
            features['uo_normalized'] = np.full(n, np.nan)
        
        return features
    
    @staticmethod
    @njit(cache=True)
    def _calculate_ultimate_oscillator_vectorized(high: np.ndarray, low: np.ndarray, close: np.ndarray) -> np.ndarray:
        """アルティメットオシレーター計算（Numba最適化版）"""
        n = len(high)
        if n < 28:  # 最長期間
            return np.zeros(n)
        
        # True Low計算
        true_low = np.zeros(n)
        buying_pressure = np.zeros(n)
        true_range = np.zeros(n)
        
        true_low[0] = low[0]
        for i in range(1, n):
            true_low[i] = min(low[i], close[i-1])
            buying_pressure[i] = close[i] - true_low[i]
            
            hl = high[i] - low[i]
            hc = abs(high[i] - close[i-1])
            lc = abs(low[i] - close[i-1])
            true_range[i] = max(hl, max(hc, lc))
        
        uo = np.zeros(n)
        periods = np.array([7, 14, 28])
        
        for i in range(27, n):
            bp_sums = np.zeros(3)
            tr_sums = np.zeros(3)
            
            for j, period in enumerate(periods):
                start_idx = i - period + 1
                bp_sums[j] = np.sum(buying_pressure[start_idx:i+1])
                tr_sums[j] = np.sum(true_range[start_idx:i+1])
            
            # Raw UO計算
            if all(tr_sums > 1e-10):
                raw_uo_7 = bp_sums[0] / tr_sums[0] if tr_sums[0] > 0 else 0
                raw_uo_14 = bp_sums[1] / tr_sums[1] if tr_sums[1] > 0 else 0
                raw_uo_28 = bp_sums[2] / tr_sums[2] if tr_sums[2] > 0 else 0
                
                uo[i] = 100 * (4 * raw_uo_7 + 2 * raw_uo_14 + raw_uo_28) / 7
        
        return uo
    
    # =========================================================================
    # 出来高関連指標（NumPy実装）
    # =========================================================================
    
    def calculate_volume_features(self, high: np.ndarray, low: np.ndarray, 
                                 close: np.ndarray, volume: np.ndarray) -> Dict[str, np.ndarray]:
        """出来高関連特徴量の統合計算（NumPy実装）"""
        features = {}
        
        # NumPy配列に変換
        high = self._ensure_numpy_array(high)
        low = self._ensure_numpy_array(low)
        close = self._ensure_numpy_array(close)
        volume = self._ensure_numpy_array(volume)
        
        try:
            # VPT (Volume Price Trend)
            vpt_features = self._safe_calculation(self._calculate_vpt_numpy, close, volume)
            if isinstance(vpt_features, dict):
                features.update(vpt_features)
            
            # A/D Line (Accumulation/Distribution)
            ad_features = self._safe_calculation(self._calculate_ad_line_numpy, high, low, close, volume)
            if isinstance(ad_features, dict):
                features.update(ad_features)
            
            # CMF (Chaikin Money Flow)
            for period in self.params['cmf_periods']:
                cmf_features = self._safe_calculation(self._calculate_cmf_numpy, high, low, close, volume, period)
                if isinstance(cmf_features, dict):
                    features.update(cmf_features)
            
            # Chaikin Oscillator
            chaikin_features = self._safe_calculation(self._calculate_chaikin_oscillator_numpy, high, low, close, volume)
            if isinstance(chaikin_features, dict):
                features.update(chaikin_features)
            
            # MFI (Money Flow Index)
            for period in self.params['mfi_periods']:
                mfi_features = self._safe_calculation(self._calculate_mfi_numpy, high, low, close, volume, period)
                if isinstance(mfi_features, dict):
                    features.update(mfi_features)
            
            # VWAP (Volume Weighted Average Price)
            for period in [20, 50, 100]:
                vwap_features = self._safe_calculation(self._calculate_vwap_numpy, high, low, close, volume, period)
                if isinstance(vwap_features, dict):
                    features.update(vwap_features)
            
            # Volume Oscillator
            volume_osc_features = self._safe_calculation(self._calculate_volume_oscillator_numpy, volume)
            if isinstance(volume_osc_features, dict):
                features.update(volume_osc_features)
            
            # Ease of Movement
            eom_features = self._safe_calculation(self._calculate_ease_of_movement_numpy, high, low, volume)
            if isinstance(eom_features, dict):
                features.update(eom_features)
            
            # Volume ROC
            vol_roc_features = self._safe_calculation(self._calculate_volume_roc_numpy, volume)
            if isinstance(vol_roc_features, dict):
                features.update(vol_roc_features)
                
        except Exception as e:
            logger.error(f"出来高特徴量計算エラー: {e}")
        
        return features
    
    def _calculate_vpt_numpy(self, close: np.ndarray, volume: np.ndarray) -> Dict[str, np.ndarray]:
        """VPT計算（NumPy実装）"""
        
        # 前日終値
        prev_close = np.roll(close, 1)
        prev_close[0] = close[0]
        
        # VPT変化量計算
        vpt_change = (close - prev_close) / (prev_close + 1e-10) * volume
        
        # VPT累積（NumPy cumsum使用）
        vpt = np.cumsum(vpt_change)
        
        # VPTシグナル
        vpt_signal = np.diff(vpt, prepend=vpt[0])
        
        # VPTの移動平均
        features = {
            'vpt': vpt,
            'vpt_signal': vpt_signal
        }
        
        for period in [10, 20, 50]:
            vpt_ma = self._rolling_mean_numpy(vpt, period)
            vpt_above_ma = np.where(vpt > vpt_ma, 1.0, 0.0)
            
            features[f'vpt_ma_{period}'] = vpt_ma
            features[f'vpt_above_ma_{period}'] = vpt_above_ma
        
        return features
    
    def _calculate_ad_line_numpy(self, high: np.ndarray, low: np.ndarray, 
                               close: np.ndarray, volume: np.ndarray) -> Dict[str, np.ndarray]:
        """A/D Line計算（NumPy実装）"""
        
        # CLV (Close Location Value)計算
        clv = np.divide(
            ((close - low) - (high - close)),
            (high - low + 1e-10),
            out=np.zeros_like(close),
            where=((high - low) != 0)
        )
        
        # A/D変化量
        ad_change = clv * volume
        
        # A/D Line累積（NumPy cumsum使用）
        ad_line = np.cumsum(ad_change)
        
        # A/D Lineモメンタム
        ad_line_momentum = np.diff(ad_line, prepend=ad_line[0])
        
        # 価格とA/Dラインのダイバージェンス
        price_momentum = np.diff(close, prepend=close[0])
        
        # 正規化してダイバージェンス計算
        price_momentum_std = np.std(price_momentum)
        ad_momentum_std = np.std(ad_line_momentum)
        
        if price_momentum_std > 1e-10 and ad_momentum_std > 1e-10:
            ad_price_divergence = (price_momentum / price_momentum_std - 
                                 ad_line_momentum / ad_momentum_std)
        else:
            ad_price_divergence = np.zeros_like(close)
        
        return {
            'ad_line': ad_line,
            'ad_line_momentum': ad_line_momentum,
            'ad_price_divergence': ad_price_divergence
        }
    
    def _calculate_cmf_numpy(self, high: np.ndarray, low: np.ndarray, close: np.ndarray, 
                           volume: np.ndarray, period: int) -> Dict[str, np.ndarray]:
        """CMF計算（NumPy実装）"""
        
        # CLV計算
        clv = np.divide(
            ((close - low) - (high - close)),
            (high - low + 1e-10),
            out=np.zeros_like(close),
            where=((high - low) != 0)
        )
        
        # Money Flow Volume
        money_flow_volume = clv * volume
        
        # CMF計算
        mfv_sum = self._rolling_sum_numpy(money_flow_volume, period)
        volume_sum = self._rolling_sum_numpy(volume, period)
        
        cmf = np.divide(mfv_sum, volume_sum, 
                       out=np.zeros_like(mfv_sum), 
                       where=(volume_sum != 0))
        
        # 追加特徴量（NumPy操作）
        cmf_positive = np.where(cmf > 0, 1.0, 0.0)
        cmf_strong_positive = np.where(cmf > 0.2, 1.0, 0.0)
        cmf_strong_negative = np.where(cmf < -0.2, 1.0, 0.0)
        
        return {
            f'cmf_{period}': cmf,
            f'cmf_positive_{period}': cmf_positive,
            f'cmf_strong_positive_{period}': cmf_strong_positive,
            f'cmf_strong_negative_{period}': cmf_strong_negative
        }
    
    def _calculate_chaikin_oscillator_numpy(self, high: np.ndarray, low: np.ndarray,
                                          close: np.ndarray, volume: np.ndarray) -> Dict[str, np.ndarray]:
        """チャイキンオシレーター計算（NumPy実装）"""
        
        # CLV計算
        clv = np.divide(
            ((close - low) - (high - close)),
            (high - low + 1e-10),
            out=np.zeros_like(close),
            where=((high - low) != 0)
        )
        
        # A/D変化量
        ad_change = clv * volume
        
        # A/D Line累積（NumPy cumsum使用）
        ad_line = np.cumsum(ad_change)
        
        # 3日と10日のEMA
        ema3 = self._ema_numpy(ad_line, 3)
        ema10 = self._ema_numpy(ad_line, 10)
        
        # チャイキンオシレーター
        chaikin_oscillator = ema3 - ema10
        
        # 追加特徴量（NumPy操作）
        chaikin_positive = np.where(chaikin_oscillator > 0, 1.0, 0.0)
        chaikin_momentum = np.diff(chaikin_oscillator, prepend=chaikin_oscillator[0])
        
        chaikin_std = np.std(chaikin_oscillator)
        if chaikin_std > 1e-10:
            chaikin_normalized = np.tanh(chaikin_oscillator / chaikin_std)
        else:
            chaikin_normalized = np.zeros_like(chaikin_oscillator)
        
        return {
            'chaikin_oscillator': chaikin_oscillator,
            'chaikin_positive': chaikin_positive,
            'chaikin_momentum': chaikin_momentum,
            'chaikin_normalized': chaikin_normalized
        }
    
    def _calculate_mfi_numpy(self, high: np.ndarray, low: np.ndarray, close: np.ndarray,
                           volume: np.ndarray, period: int) -> Dict[str, np.ndarray]:
        """MFI計算（NumPy実装）"""
        
        # Typical Price
        typical_price = (high + low + close) / 3.0
        
        # Raw Money Flow
        raw_money_flow = typical_price * volume
        
        # 前日Typical Price
        prev_typical_price = np.roll(typical_price, 1)
        prev_typical_price[0] = typical_price[0]
        
        # Positive/Negative Money Flow
        positive_money_flow = np.where(typical_price > prev_typical_price, raw_money_flow, 0.0)
        negative_money_flow = np.where(typical_price < prev_typical_price, raw_money_flow, 0.0)
        
        # ローリング合計
        positive_mf_sum = self._rolling_sum_numpy(positive_money_flow, period)
        negative_mf_sum = self._rolling_sum_numpy(negative_money_flow, period)
        
        # Money Ratio
        money_ratio = np.divide(positive_mf_sum, negative_mf_sum,
                               out=np.ones_like(positive_mf_sum),
                               where=(negative_mf_sum != 0))
        
        # MFI計算
        mfi = 100 - (100 / (1 + money_ratio))
        
        # 追加特徴量（NumPy操作）
        mfi_overbought = np.where(mfi > 80, 1.0, 0.0)
        mfi_oversold = np.where(mfi < 20, 1.0, 0.0)
        mfi_normalized = (mfi - 50) / 50.0
        
        return {
            f'mfi_{period}': mfi,
            f'mfi_overbought_{period}': mfi_overbought,
            f'mfi_oversold_{period}': mfi_oversold,
            f'mfi_normalized_{period}': mfi_normalized
        }
    
    def _calculate_vwap_numpy(self, high: np.ndarray, low: np.ndarray, close: np.ndarray,
                            volume: np.ndarray, period: int) -> Dict[str, np.ndarray]:
        """VWAP計算（NumPy実装）"""
        
        # Typical Price
        typical_price = (high + low + close) / 3.0
        
        # Price * Volume
        pv = typical_price * volume
        
        # VWAP計算
        pv_sum = self._rolling_sum_numpy(pv, period)
        volume_sum = self._rolling_sum_numpy(volume, period)
        
        vwap = np.divide(pv_sum, volume_sum,
                        out=np.zeros_like(pv_sum),
                        where=(volume_sum != 0))
        
        # 追加特徴量（NumPy操作）
        price_above_vwap = np.where(close > vwap, 1.0, 0.0)
        vwap_distance = (close - vwap) / (close + 1e-10)
        vwap_deviation = np.abs(close - vwap) / (vwap + 1e-10)
        
        return {
            f'vwap_{period}': vwap,
            f'price_above_vwap_{period}': price_above_vwap,
            f'vwap_distance_{period}': vwap_distance,
            f'vwap_deviation_{period}': vwap_deviation
        }
    
    def _calculate_volume_oscillator_numpy(self, volume: np.ndarray) -> Dict[str, np.ndarray]:
        """ボリュームオシレーター計算（NumPy実装）"""
        features = {}
        
        # 複数の期間設定
        period_pairs = [(5, 10), (10, 20), (14, 28)]
        
        for short, long in period_pairs:
            vol_short_avg = self._rolling_mean_numpy(volume, short)
            vol_long_avg = self._rolling_mean_numpy(volume, long)
            
            volume_oscillator = 100 * np.divide(
                vol_short_avg - vol_long_avg,
                vol_long_avg,
                out=np.zeros_like(vol_short_avg),
                where=(vol_long_avg != 0)
            )
            
            # 追加特徴量（NumPy操作）
            vo_positive = np.where(volume_oscillator > 0, 1.0, 0.0)
            vo_momentum = np.diff(volume_oscillator, prepend=volume_oscillator[0])
            
            features[f'volume_oscillator_{short}_{long}'] = volume_oscillator
            features[f'vo_positive_{short}_{long}'] = vo_positive
            features[f'vo_momentum_{short}_{long}'] = vo_momentum
        
        return features
    
    def _calculate_ease_of_movement_numpy(self, high: np.ndarray, low: np.ndarray, 
                                        volume: np.ndarray) -> Dict[str, np.ndarray]:
        """Ease of Movement計算（NumPy実装）"""
        
        # 前日の高値・安値
        prev_high = np.roll(high, 1)
        prev_low = np.roll(low, 1)
        prev_high[0] = high[0]
        prev_low[0] = low[0]
        
        # Distance Moved
        distance_moved = ((high + low) / 2.0) - ((prev_high + prev_low) / 2.0)
        
        # Box Height
        box_height = np.divide(volume, (high - low), 
                             out=np.zeros_like(volume), 
                             where=((high - low) != 0))
        
        # Ease of Movement
        eom = np.divide(10000.0 * distance_moved, box_height,
                       out=np.zeros_like(distance_moved),
                       where=(box_height != 0))
        
        # 追加特徴量（NumPy操作）
        eom_positive = np.where(eom > 0, 1.0, 0.0)
        eom_momentum = np.diff(eom, prepend=eom[0])
        
        features = {
            'ease_of_movement': eom,
            'eom_positive': eom_positive,
            'eom_momentum': eom_momentum
        }
        
        # EMVの移動平均
        for period in [14, 20]:
            eom_ma = self._rolling_mean_numpy(eom, period)
            eom_signal = np.where(eom > eom_ma, 1.0, 0.0)
            
            features[f'eom_ma_{period}'] = eom_ma
            features[f'eom_signal_{period}'] = eom_signal
        
        return features
    
    def _calculate_volume_roc_numpy(self, volume: np.ndarray) -> Dict[str, np.ndarray]:
        """出来高変化率計算（NumPy実装）"""
        features = {}
        
        for period in self.params['vol_roc_periods']:
            # 過去の出来高
            volume_prev = np.roll(volume, period)
            volume_prev[:period] = volume[:period]  # 先頭をパディング
            
            # Volume ROC計算
            volume_roc = 100 * np.divide(
                volume - volume_prev,
                volume_prev,
                out=np.zeros_like(volume),
                where=(volume_prev != 0)
            )
            
            # 追加特徴量（NumPy操作）
            vol_roc_positive = np.where(volume_roc > 0, 1.0, 0.0)
            vol_roc_strong = np.where(np.abs(volume_roc) > 50, 1.0, 0.0)
            
            features[f'volume_roc_{period}'] = volume_roc
            features[f'vol_roc_positive_{period}'] = vol_roc_positive
            features[f'vol_roc_strong_{period}'] = vol_roc_strong
        
        return features
    
    # =========================================================================
    # 移動平均線・トレンド分析（NumPy実装）
    # =========================================================================
    
    def calculate_moving_averages(self, close: np.ndarray) -> Dict[str, np.ndarray]:
        """移動平均線特徴量の統合計算（NumPy実装）"""
        features = {}
        close = self._ensure_numpy_array(close)
        
        try:
            # WMA (Weighted Moving Average)
            wma_features = self._safe_calculation(self._calculate_wma_numpy, close)
            if isinstance(wma_features, dict):
                features.update(wma_features)
            
            # HMA (Hull Moving Average)
            hma_features = self._calculate_hma_features_numba(close)
            features.update(hma_features)
            
            # KAMA (Kaufman Adaptive Moving Average)
            kama_features = self._calculate_kama_features_numba(close)
            features.update(kama_features)
            
            # DEMA, TEMA
            dema_tema_features = self._safe_calculation(self._calculate_dema_tema_numpy, close)
            if isinstance(dema_tema_features, dict):
                features.update(dema_tema_features)
            
            # 移動平均線の傾きと乖離
            ma_analysis_features = self._safe_calculation(self._calculate_ma_analysis_numpy, close)
            if isinstance(ma_analysis_features, dict):
                features.update(ma_analysis_features)
            
            # ゴールデンクロス・デッドクロス
            cross_features = self._safe_calculation(self._calculate_cross_signals_numpy, close)
            if isinstance(cross_features, dict):
                features.update(cross_features)
                
        except Exception as e:
            logger.error(f"移動平均線特徴量計算エラー: {e}")
        
        return features
    
    def _calculate_wma_numpy(self, close: np.ndarray) -> Dict[str, np.ndarray]:
        """WMA計算（NumPy実装）"""
        features = {}
        
        periods = [9, 21, 50]
        for period in periods:
            # WMAをNumbaで計算
            wma_result = self._numba_safe_calculation(
                self._calculate_wma_vectorized, close, period
            )
            
            if wma_result is not None:
                # NumPyで追加特徴量を計算
                price_above_wma = np.where(close > wma_result, 1.0, 0.0)
                wma_slope = np.diff(wma_result, prepend=wma_result[0])
                wma_distance = (close - wma_result) / (close + 1e-10)
                
                features[f'wma_{period}'] = wma_result
                features[f'price_above_wma_{period}'] = price_above_wma
                features[f'wma_slope_{period}'] = wma_slope
                features[f'wma_distance_{period}'] = wma_distance
            else:
                n = len(close)
                features[f'wma_{period}'] = np.full(n, np.nan)
                features[f'price_above_wma_{period}'] = np.full(n, np.nan)
                features[f'wma_slope_{period}'] = np.full(n, np.nan)
                features[f'wma_distance_{period}'] = np.full(n, np.nan)
        
        return features
    
    @staticmethod
    @njit(parallel=True, cache=True)
    def _calculate_wma_vectorized(data: np.ndarray, period: int) -> np.ndarray:
        """WMA計算（完全ベクトル化・並列版）"""
        n = len(data)
        if n < period:
            return np.zeros(n)
        
        wma = np.zeros(n)
        weight_sum = period * (period + 1) // 2
        
        # 並列計算
        for i in prange(period-1, n):
            weighted_sum = 0.0
            for j in range(period):
                weight = period - j
                weighted_sum += data[i - j] * weight
            wma[i] = weighted_sum / weight_sum
        
        return wma
    
    def _calculate_hma_features_numba(self, close: np.ndarray) -> Dict[str, np.ndarray]:
        """HMA特徴量計算（Numba版）"""
        features = {}
        
        periods = [14, 21, 50]
        for period in periods:
            hma_result = self._numba_safe_calculation(
                self._calculate_hma_vectorized, close, period
            )
            
            if hma_result is not None:
                # NumPyで追加特徴量を計算
                price_above_hma = np.where(close > hma_result, 1.0, 0.0)
                hma_slope = np.diff(hma_result, prepend=hma_result[0])
                hma_momentum = np.diff(hma_slope, prepend=hma_slope[0])
                
                features[f'hma_{period}'] = hma_result
                features[f'price_above_hma_{period}'] = price_above_hma
                features[f'hma_slope_{period}'] = hma_slope
                features[f'hma_momentum_{period}'] = hma_momentum
            else:
                n = len(close)
                features[f'hma_{period}'] = np.full(n, np.nan)
                features[f'price_above_hma_{period}'] = np.full(n, np.nan)
                features[f'hma_slope_{period}'] = np.full(n, np.nan)
                features[f'hma_momentum_{period}'] = np.full(n, np.nan)
        
        return features
    
    @staticmethod
    @njit(cache=True)
    def _calculate_hma_vectorized(data: np.ndarray, period: int) -> np.ndarray:
        """HMA計算（Numba最適化版）"""
        n = len(data)
        if n < period:
            return np.zeros(n)
        
        half_period = period // 2
        sqrt_period = int(np.sqrt(period))
        
        # WMA(period/2)の計算
        wma_half = np.zeros(n)
        weight_sum_half = half_period * (half_period + 1) // 2
        
        for i in range(half_period - 1, n):
            weighted_sum = 0.0
            for j in range(half_period):
                weight = half_period - j
                weighted_sum += data[i - j] * weight
            wma_half[i] = weighted_sum / weight_sum_half
        
        # WMA(period)の計算
        wma_full = np.zeros(n)
        weight_sum_full = period * (period + 1) // 2
        
        for i in range(period - 1, n):
            weighted_sum = 0.0
            for j in range(period):
                weight = period - j
                weighted_sum += data[i - j] * weight
            wma_full[i] = weighted_sum / weight_sum_full
        
        # Raw HMA = 2*WMA(period/2) - WMA(period)
        raw_hma = 2 * wma_half - wma_full
        
        # WMA(sqrt(period)) of Raw HMA
        hma = np.zeros(n)
        weight_sum_sqrt = sqrt_period * (sqrt_period + 1) // 2
        
        for i in range(max(period - 1, sqrt_period - 1), n):
            weighted_sum = 0.0
            for j in range(sqrt_period):
                if i - j >= 0:
                    weight = sqrt_period - j
                    weighted_sum += raw_hma[i - j] * weight
            hma[i] = weighted_sum / weight_sum_sqrt
        
        return hma
    
    def _calculate_kama_features_numba(self, close: np.ndarray) -> Dict[str, np.ndarray]:
        """KAMA特徴量計算（Numba版）"""
        features = {}
        
        periods = [14, 21, 30]
        for period in periods:
            kama_result = self._numba_safe_calculation(
                self._calculate_kama_vectorized, close, period
            )
            
            if kama_result is not None and len(kama_result.shape) == 2:
                kama_values = kama_result[:, 0]
                efficiency_ratio = kama_result[:, 1]
                
                # NumPyで追加特徴量を計算
                price_above_kama = np.where(close > kama_values, 1.0, 0.0)
                kama_slope = np.diff(kama_values, prepend=kama_values[0])
                
                features[f'kama_{period}'] = kama_values
                features[f'price_above_kama_{period}'] = price_above_kama
                features[f'kama_efficiency_{period}'] = efficiency_ratio
                features[f'kama_slope_{period}'] = kama_slope
            else:
                n = len(close)
                features[f'kama_{period}'] = np.full(n, np.nan)
                features[f'price_above_kama_{period}'] = np.full(n, np.nan)
                features[f'kama_efficiency_{period}'] = np.full(n, np.nan)
                features[f'kama_slope_{period}'] = np.full(n, np.nan)
        
        return features
    
    @staticmethod
    @njit(cache=True)
    def _calculate_kama_vectorized(data: np.ndarray, period: int = 14) -> np.ndarray:
        """KAMA計算（Numba最適化版）"""
        n = len(data)
        if n < period + 1:
            return np.zeros((n, 2))
        
        kama = np.zeros(n)
        efficiency_ratio = np.zeros(n)
        
        kama[period] = np.mean(data[:period + 1])
        
        fast_alpha = 2.0 / (2.0 + 1.0)
        slow_alpha = 2.0 / (30.0 + 1.0)
        
        for i in range(period + 1, n):
            # Direction計算
            direction = abs(data[i] - data[i - period])
            
            # Volatility計算
            volatility = 0.0
            for j in range(period):
                volatility += abs(data[i - j] - data[i - j - 1])
            
            # Efficiency Ratio
            if volatility > 1e-10:
                er = direction / volatility
            else:
                er = 1.0
            
            efficiency_ratio[i] = er
            
            # Smoothing Constant
            sc = er * (fast_alpha - slow_alpha) + slow_alpha
            alpha = sc * sc
            
            # KAMA計算
            kama[i] = kama[i - 1] + alpha * (data[i] - kama[i - 1])
        
        result = np.zeros((n, 2))
        result[:, 0] = kama
        result[:, 1] = efficiency_ratio
        return result
    
    def _calculate_dema_tema_numpy(self, close: np.ndarray) -> Dict[str, np.ndarray]:
        """DEMA・TEMA計算（NumPy実装）"""
        features = {}
        
        # DEMA計算
        for period in self.params['dema_periods']:
            ema1 = self._ema_numpy(close, period)
            ema2 = self._ema_numpy(ema1, period)
            dema = 2 * ema1 - ema2
            
            # 追加特徴量（NumPy操作）
            price_above_dema = np.where(close > dema, 1.0, 0.0)
            dema_slope = np.diff(dema, prepend=dema[0])
            dema_acceleration = np.diff(dema_slope, prepend=dema_slope[0])
            
            features[f'dema_{period}'] = dema
            features[f'price_above_dema_{period}'] = price_above_dema
            features[f'dema_slope_{period}'] = dema_slope
            features[f'dema_acceleration_{period}'] = dema_acceleration
        
        # TEMA計算
        for period in self.params['tema_periods']:
            ema1 = self._ema_numpy(close, period)
            ema2 = self._ema_numpy(ema1, period)
            ema3 = self._ema_numpy(ema2, period)
            tema = 3 * ema1 - 3 * ema2 + ema3
            
            # 追加特徴量（NumPy操作）
            price_above_tema = np.where(close > tema, 1.0, 0.0)
            tema_momentum = np.diff(tema, prepend=tema[0])
            tema_trend_strength = np.abs(tema_momentum)
            
            features[f'tema_{period}'] = tema
            features[f'price_above_tema_{period}'] = price_above_tema
            features[f'tema_momentum_{period}'] = tema_momentum
            features[f'tema_trend_strength_{period}'] = tema_trend_strength
        
        return features
    
    def _calculate_ma_analysis_numpy(self, close: np.ndarray) -> Dict[str, np.ndarray]:
        """移動平均線分析（NumPy実装）"""
        features = {}
        
        periods = [10, 20, 50, 200]
        
        for period in periods:
            # SMAとEMA計算
            sma = self._rolling_mean_numpy(close, period)
            ema = self._ema_numpy(close, period)
            
            # 傾きと乖離率を計算
            sma_slope = np.diff(sma, prepend=sma[0])
            ema_slope = np.diff(ema, prepend=ema[0])
            
            # 傾きの強度
            sma_slope_strength = np.abs(sma_slope)
            ema_slope_strength = np.abs(ema_slope)
            
            # 上昇トレンド（NumPy操作）
            sma_uptrend = np.where(sma_slope > 0, 1.0, 0.0)
            ema_uptrend = np.where(ema_slope > 0, 1.0, 0.0)
            
            # 乖離率
            sma_deviation = 100 * np.divide(close - sma, sma, 
                                          out=np.zeros_like(close), 
                                          where=(sma != 0))
            ema_deviation = 100 * np.divide(close - ema, ema,
                                          out=np.zeros_like(close),
                                          where=(ema != 0))
            
            # 絶対乖離率
            sma_abs_deviation = np.abs(sma_deviation)
            ema_abs_deviation = np.abs(ema_deviation)
            
            # 過大乖離（NumPy操作）
            sma_excessive_deviation = np.where(sma_abs_deviation > 5, 1.0, 0.0)
            ema_excessive_deviation = np.where(ema_abs_deviation > 5, 1.0, 0.0)
            
            # 結果を格納
            features[f'sma_slope_{period}'] = sma_slope
            features[f'ema_slope_{period}'] = ema_slope
            features[f'sma_slope_strength_{period}'] = sma_slope_strength
            features[f'ema_slope_strength_{period}'] = ema_slope_strength
            features[f'sma_uptrend_{period}'] = sma_uptrend
            features[f'ema_uptrend_{period}'] = ema_uptrend
            features[f'sma_deviation_{period}'] = sma_deviation
            features[f'ema_deviation_{period}'] = ema_deviation
            features[f'sma_abs_deviation_{period}'] = sma_abs_deviation
            features[f'ema_abs_deviation_{period}'] = ema_abs_deviation
            features[f'sma_excessive_deviation_{period}'] = sma_excessive_deviation
            features[f'ema_excessive_deviation_{period}'] = ema_excessive_deviation
        
        return features
    
    def _calculate_cross_signals_numpy(self, close: np.ndarray) -> Dict[str, np.ndarray]:
        """ゴールデンクロス・デッドクロス計算（NumPy実装）"""
        features = {}
        
        # 短期・長期MA組み合わせ
        cross_pairs = [(25, 75), (50, 200), (20, 60)]
        
        for short_period, long_period in cross_pairs:
            short_ma = self._rolling_mean_numpy(close, short_period)
            long_ma = self._rolling_mean_numpy(close, long_period)
            
            # 現在のクロス状態（NumPy操作）
            golden_cross = np.where(short_ma > long_ma, 1.0, 0.0)
            
            # MA間距離
            ma_distance = 100 * np.divide(short_ma - long_ma, long_ma,
                                        out=np.zeros_like(short_ma),
                                        where=(long_ma != 0))
            
            # 収束度
            ma_convergence = np.abs(short_ma - long_ma) / (long_ma + 1e-10) * 100
            
            # クロス発生検出
            golden_cross_diff = np.diff(golden_cross, prepend=golden_cross[0])
            golden_cross_signal = np.where(golden_cross_diff == 1, 1.0, 0.0)
            death_cross_signal = np.where(golden_cross_diff == -1, 1.0, 0.0)
            
            features[f'golden_cross_{short_period}_{long_period}'] = golden_cross
            features[f'golden_cross_signal_{short_period}_{long_period}'] = golden_cross_signal
            features[f'death_cross_signal_{short_period}_{long_period}'] = death_cross_signal
            features[f'ma_distance_{short_period}_{long_period}'] = ma_distance
            features[f'ma_convergence_{short_period}_{long_period}'] = ma_convergence
        
        return features
    
    # =========================================================================
    # ボラティリティ・バンド指標（NumPy実装）
    # =========================================================================
    
    def calculate_volatility_bands(self, high: np.ndarray, low: np.ndarray, close: np.ndarray) -> Dict[str, np.ndarray]:
        """ボラティリティ・バンド指標の統合計算（NumPy実装）"""
        features = {}
        
        # NumPy配列に変換
        high = self._ensure_numpy_array(high)
        low = self._ensure_numpy_array(low)
        close = self._ensure_numpy_array(close)
        
        try:
            # ボリンジャーバンドスクイーズ
            bb_squeeze_features = self._safe_calculation(self._calculate_bb_squeeze_numpy, close)
            if isinstance(bb_squeeze_features, dict):
                features.update(bb_squeeze_features)
            
            # ケルトナーチャネル
            kc_features = self._safe_calculation(self._calculate_keltner_channel_numpy, high, low, close)
            if isinstance(kc_features, dict):
                features.update(kc_features)
            
            # ドンチャンチャネル
            dc_features = self._safe_calculation(self._calculate_donchian_channel_numpy, high, low, close)
            if isinstance(dc_features, dict):
                features.update(dc_features)
            
            # ATRバンド
            atr_bands_features = self._safe_calculation(self._calculate_atr_bands_numpy, high, low, close)
            if isinstance(atr_bands_features, dict):
                features.update(atr_bands_features)
            
            # ヒストリカルボラティリティ
            hist_vol_features = self._safe_calculation(self._calculate_historical_volatility_numpy, close)
            if isinstance(hist_vol_features, dict):
                features.update(hist_vol_features)
            
            # ボラティリティレシオ
            vol_ratio_features = self._safe_calculation(self._calculate_volatility_ratio_numpy, close)
            if isinstance(vol_ratio_features, dict):
                features.update(vol_ratio_features)
            
            # シャンデリアエグジット
            chandelier_features = self._calculate_chandelier_exit_features_numba(high, low, close)
            features.update(chandelier_features)
            
            # ボラティリティブレイクアウト
            vol_breakout_features = self._safe_calculation(self._calculate_volatility_breakout_numpy, high, low, close)
            if isinstance(vol_breakout_features, dict):
                features.update(vol_breakout_features)
                
        except Exception as e:
            logger.error(f"ボラティリティ・バンド指標計算エラー: {e}")
        
        return features
    
    def _calculate_bb_squeeze_numpy(self, close: np.ndarray) -> Dict[str, np.ndarray]:
        """ボリンジャーバンドスクイーズ計算（NumPy実装）"""
        features = {}
        
        for period, std_mult in self.params['volatility_bb_settings']:
            # ボリンジャーバンド計算
            bb_mid = self._rolling_mean_numpy(close, period)
            bb_std = self._rolling_std_numpy(close, period)
            bb_upper = bb_mid + std_mult * bb_std
            bb_lower = bb_mid - std_mult * bb_std
            
            # バンド幅
            bb_width = np.divide(bb_upper - bb_lower, bb_mid,
                               out=np.zeros_like(bb_mid),
                               where=(bb_mid != 0))
            
            # バンド幅の移動平均
            bb_width_ma = self._rolling_mean_numpy(bb_width, 20)
            bb_width_80th = self._rolling_quantile_numpy(bb_width, 100, 0.8)
            
            # スクイーズ検出（NumPy操作）
            bb_squeeze = np.where(bb_width < bb_width_ma, 1.0, 0.0)
            
            # バンド幅変化率
            bb_width_change = np.diff(bb_width, prepend=bb_width[0])
            
            # エクスパンション検出
            bb_width_std = self._rolling_std_numpy(bb_width, 20)
            bb_expansion = np.where(bb_width_change > bb_width_std, 1.0, 0.0)
            
            # バンド幅パーセンタイル（簡易実装）
            bb_width_percentile = np.zeros_like(bb_width)
            for i in range(len(bb_width)):
                if i >= 100:
                    window = bb_width[i-99:i+1]
                    rank = np.sum(window <= bb_width[i])
                    bb_width_percentile[i] = rank / len(window)
            
            features[f'bb_squeeze_{period}'] = bb_squeeze
            features[f'bb_width_{period}'] = bb_width
            features[f'bb_width_percentile_{period}'] = bb_width_percentile
            features[f'bb_expansion_{period}'] = bb_expansion
        
        return features
    
    def _calculate_keltner_channel_numpy(self, high: np.ndarray, low: np.ndarray, close: np.ndarray) -> Dict[str, np.ndarray]:
        """ケルトナーチャネル計算（NumPy実装）"""
        features = {}
        
        for period in self.params['kc_periods']:
            # ATR計算
            prev_close = np.roll(close, 1)
            prev_close[0] = close[0]
            
            tr1 = high - low
            tr2 = np.abs(high - prev_close)
            tr3 = np.abs(low - prev_close)
            true_range = np.maximum(tr1, np.maximum(tr2, tr3))
            atr = self._ema_numpy(true_range, period)
            
            # ケルトナーチャネル
            kc_middle = self._ema_numpy(close, period)
            kc_upper = kc_middle + 2.0 * atr
            kc_lower = kc_middle - 2.0 * atr
            
            # ケルトナーチャネル内の価格位置
            kc_position = np.divide(close - kc_lower, kc_upper - kc_lower,
                                  out=np.zeros_like(close),
                                  where=((kc_upper - kc_lower) != 0))
            
            # ケルトナーチャネル幅
            kc_width = np.divide(kc_upper - kc_lower, kc_middle,
                               out=np.zeros_like(kc_middle),
                               where=(kc_middle != 0))
            
            # ブレイクアウト検出（NumPy操作）
            kc_upper_break = np.where(close > kc_upper, 1.0, 0.0)
            kc_lower_break = np.where(close < kc_lower, 1.0, 0.0)
            kc_inside = np.where((close >= kc_lower) & (close <= kc_upper), 1.0, 0.0)
            
            features[f'kc_upper_{period}'] = kc_upper
            features[f'kc_middle_{period}'] = kc_middle
            features[f'kc_lower_{period}'] = kc_lower
            features[f'kc_position_{period}'] = kc_position
            features[f'kc_width_{period}'] = kc_width
            features[f'kc_upper_break_{period}'] = kc_upper_break
            features[f'kc_lower_break_{period}'] = kc_lower_break
            features[f'kc_inside_{period}'] = kc_inside
        
        return features
    
    def _calculate_donchian_channel_numpy(self, high: np.ndarray, low: np.ndarray, close: np.ndarray) -> Dict[str, np.ndarray]:
        """ドンチャンチャネル計算（NumPy実装）"""
        features = {}
        
        for period in self.params['dc_periods']:
            dc_upper = self._rolling_max_numpy(high, period)
            dc_lower = self._rolling_min_numpy(low, period)
            dc_middle = (dc_upper + dc_lower) / 2.0
            
            # ドンチャンチャネル内の価格位置
            dc_position = np.divide(close - dc_lower, dc_upper - dc_lower,
                                  out=np.zeros_like(close),
                                  where=((dc_upper - dc_lower) != 0))
            
            # ドンチャンチャネル幅
            dc_width = np.divide(dc_upper - dc_lower, dc_middle,
                               out=np.zeros_like(dc_middle),
                               where=(dc_middle != 0))
            
            # ブレイクアウト検出（NumPy操作）
            dc_upper_break = np.where(close > dc_upper, 1.0, 0.0)
            dc_lower_break = np.where(close < dc_lower, 1.0, 0.0)
            
            # 新高値・新安値からの日数
            days_since_high = self._calculate_days_since_extreme_numpy(close, dc_upper, True)
            days_since_low = self._calculate_days_since_extreme_numpy(close, dc_lower, False)
            
            features[f'dc_upper_{period}'] = dc_upper
            features[f'dc_lower_{period}'] = dc_lower
            features[f'dc_middle_{period}'] = dc_middle
            features[f'dc_position_{period}'] = dc_position
            features[f'dc_width_{period}'] = dc_width
            features[f'dc_upper_break_{period}'] = dc_upper_break
            features[f'dc_lower_break_{period}'] = dc_lower_break
            features[f'dc_days_since_high_{period}'] = days_since_high
            features[f'dc_days_since_low_{period}'] = days_since_low
        
        return features
    
    def _calculate_days_since_extreme_numpy(self, close: np.ndarray, extreme_line: np.ndarray, is_high: bool) -> np.ndarray:
        """極値からの日数計算（NumPy実装）"""
        days_since = np.zeros(len(close))
        
        for i in range(1, len(close)):
            if is_high:
                if close[i] > extreme_line[i]:
                    days_since[i] = 0
                else:
                    days_since[i] = days_since[i-1] + 1
            else:
                if close[i] < extreme_line[i]:
                    days_since[i] = 0
                else:
                    days_since[i] = days_since[i-1] + 1
        
        return days_since
    
    def _calculate_atr_bands_numpy(self, high: np.ndarray, low: np.ndarray, close: np.ndarray) -> Dict[str, np.ndarray]:
        """ATRバンド計算（NumPy実装）"""
        features = {}
        
        atr_multipliers = [1.0, 1.5, 2.0, 2.5]
        
        for period in self.params['atr_periods_vol']:
            # ATR計算
            prev_close = np.roll(close, 1)
            prev_close[0] = close[0]
            
            tr1 = high - low
            tr2 = np.abs(high - prev_close)
            tr3 = np.abs(low - prev_close)
            true_range = np.maximum(tr1, np.maximum(tr2, tr3))
            atr = self._rolling_mean_numpy(true_range, period)
            
            for mult in atr_multipliers:
                atr_upper = close + mult * atr
                atr_lower = close - mult * atr
                
                # ATRバンド内位置
                atr_band_position = np.divide(close - atr_lower, atr_upper - atr_lower,
                                            out=np.zeros_like(close),
                                            where=((atr_upper - atr_lower) != 0))
                
                # ATRバンド幅
                atr_band_width = np.divide(atr_upper - atr_lower, close,
                                         out=np.zeros_like(close),
                                         where=(close != 0))
                
                features[f'atr_upper_{period}_{mult}'] = atr_upper
                features[f'atr_lower_{period}_{mult}'] = atr_lower
                features[f'atr_band_position_{period}_{mult}'] = atr_band_position
                features[f'atr_band_width_{period}_{mult}'] = atr_band_width
        
        return features
    
    def _calculate_historical_volatility_numpy(self, close: np.ndarray) -> Dict[str, np.ndarray]:
        """ヒストリカルボラティリティ計算（NumPy実装）"""
        features = {}
        
        # 対数リターン計算
        log_close = np.log(close)
        log_returns = np.diff(log_close, prepend=log_close[0])
        
        for period in self.params['hist_vol_periods']:
            # ローリング標準偏差
            hist_vol = self._rolling_std_numpy(log_returns, period)
            
            # 年率化（252営業日ベース）
            hist_vol_annualized = hist_vol * np.sqrt(252)
            
            # ボラティリティの相対水準
            vol_median = self._rolling_median_numpy(hist_vol, 252)
            vol_ma = self._rolling_mean_numpy(hist_vol, 50)
            
            # ボラティリティレジーム（NumPy操作）
            vol_regime = np.where(hist_vol > vol_ma, 1.0, 0.0)
            
            # ボラティリティパーセンタイル（簡易実装）
            vol_percentile = np.zeros_like(hist_vol)
            for i in range(len(hist_vol)):
                if i >= 100:
                    window = hist_vol[i-99:i+1]
                    rank = np.sum(window <= hist_vol[i])
                    vol_percentile[i] = rank / len(window)
            
            features[f'hist_vol_{period}'] = hist_vol
            features[f'hist_vol_annualized_{period}'] = hist_vol_annualized
            features[f'vol_percentile_{period}'] = vol_percentile
            features[f'vol_regime_{period}'] = vol_regime
        
        return features
    
    def _calculate_volatility_ratio_numpy(self, close: np.ndarray) -> Dict[str, np.ndarray]:
        """ボラティリティレシオ計算（NumPy実装）"""
        features = {}
        
        # 対数リターン計算
        log_close = np.log(close)
        log_returns = np.diff(log_close, prepend=log_close[0])
        
        # 短期・長期ボラティリティ比較
        vol_pairs = [(5, 20), (10, 30), (20, 60)]
        
        for short, long in vol_pairs:
            short_vol = self._rolling_std_numpy(log_returns, short)
            long_vol = self._rolling_std_numpy(log_returns, long)
            
            volatility_ratio = np.divide(short_vol, long_vol,
                                       out=np.ones_like(short_vol),
                                       where=(long_vol != 0))
            
            # 追加特徴量（NumPy操作）
            vol_ratio_high = np.where(volatility_ratio > 1.2, 1.0, 0.0)
            vol_ratio_low = np.where(volatility_ratio < 0.8, 1.0, 0.0)
            vol_change = np.diff(volatility_ratio, prepend=volatility_ratio[0])
            
            features[f'volatility_ratio_{short}_{long}'] = volatility_ratio
            features[f'vol_ratio_high_{short}_{long}'] = vol_ratio_high
            features[f'vol_ratio_low_{short}_{long}'] = vol_ratio_low
            features[f'vol_change_{short}_{long}'] = vol_change
        
        return features
    
    def _calculate_chandelier_exit_features_numba(self, high: np.ndarray, low: np.ndarray, close: np.ndarray) -> Dict[str, np.ndarray]:
        """シャンデリアエグジット計算（Numba版）"""
        features = {}
        
        periods = [20, 50]
        multipliers = [2.0, 3.0]
        
        for period in periods:
            for mult in multipliers:
                chandelier_result = self._numba_safe_calculation(
                    self._calculate_chandelier_exit_vectorized, high, low, close, period, mult
                )
                
                if chandelier_result is not None and len(chandelier_result.shape) == 2:
                    chandelier_long = chandelier_result[:, 0]
                    chandelier_short = chandelier_result[:, 1]
                    
                    # NumPyで追加特徴量を計算
                    chandelier_long_exit = np.where(close < chandelier_long, 1.0, 0.0)
                    chandelier_short_exit = np.where(close > chandelier_short, 1.0, 0.0)
                    chandelier_long_distance = (close - chandelier_long) / (close + 1e-10)
                    chandelier_short_distance = (chandelier_short - close) / (close + 1e-10)
                    
                    features[f'chandelier_long_{period}_{mult}'] = chandelier_long
                    features[f'chandelier_short_{period}_{mult}'] = chandelier_short
                    features[f'chandelier_long_exit_{period}_{mult}'] = chandelier_long_exit
                    features[f'chandelier_short_exit_{period}_{mult}'] = chandelier_short_exit
                    features[f'chandelier_long_distance_{period}_{mult}'] = chandelier_long_distance
                    features[f'chandelier_short_distance_{period}_{mult}'] = chandelier_short_distance
                else:
                    # フォールバック
                    n = len(close)
                    features[f'chandelier_long_{period}_{mult}'] = np.full(n, np.nan)
                    features[f'chandelier_short_{period}_{mult}'] = np.full(n, np.nan)
                    features[f'chandelier_long_exit_{period}_{mult}'] = np.full(n, np.nan)
                    features[f'chandelier_short_exit_{period}_{mult}'] = np.full(n, np.nan)
                    features[f'chandelier_long_distance_{period}_{mult}'] = np.full(n, np.nan)
                    features[f'chandelier_short_distance_{period}_{mult}'] = np.full(n, np.nan)
        
        return features
    
    @staticmethod
    @njit(cache=True)
    def _calculate_chandelier_exit_vectorized(high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int, mult: float) -> np.ndarray:
        """シャンデリアエグジット計算（Numba最適化版）"""
        n = len(high)
        if n < period:
            return np.zeros((n, 2))
        
        # ATR計算
        true_range = np.zeros(n)
        for i in range(1, n):
            hl = high[i] - low[i]
            hc = abs(high[i] - close[i-1])
            lc = abs(low[i] - close[i-1])
            true_range[i] = max(hl, max(hc, lc))
        
        true_range[0] = high[0] - low[0]
        
        # ATRのSMA
        atr = np.zeros(n)
        for i in range(period - 1, n):
            atr[i] = np.mean(true_range[i - period + 1:i + 1])
        
        # 最高値・最安値
        highest_high = np.zeros(n)
        lowest_low = np.zeros(n)
        
        for i in range(period - 1, n):
            highest_high[i] = np.max(high[i - period + 1:i + 1])
            lowest_low[i] = np.min(low[i - period + 1:i + 1])
        
        # シャンデリアエグジット
        chandelier_long = highest_high - mult * atr
        chandelier_short = lowest_low + mult * atr
        
        result = np.zeros((n, 2))
        result[:, 0] = chandelier_long
        result[:, 1] = chandelier_short
        return result
    
    def _calculate_volatility_breakout_numpy(self, high: np.ndarray, low: np.ndarray, close: np.ndarray) -> Dict[str, np.ndarray]:
        """ボラティリティブレイクアウト計算（NumPy実装）"""
        features = {}
        
        # 前日比変動率
        daily_range = np.divide(high - low, close, 
                              out=np.zeros_like(high), 
                              where=(close != 0))
        price_change = np.diff(close, prepend=close[0]) / (close + 1e-10)
        
        # 複数期間でのボラティリティ分析
        periods = [5, 10, 20, 50]
        
        for period in periods:
            # 平均日足レンジ
            avg_range = self._rolling_mean_numpy(daily_range, period)
            
            # 価格変動の標準偏差
            price_change_std = self._rolling_std_numpy(price_change, period)
            
            # レンジ拡張
            range_expansion = np.divide(daily_range, avg_range,
                                      out=np.ones_like(daily_range),
                                      where=(avg_range != 0))
            
            # 価格変動のZ-score
            price_change_z = np.divide(price_change, price_change_std,
                                     out=np.zeros_like(price_change),
                                     where=(price_change_std != 0))
            
            # 追加特徴量（NumPy操作）
            high_range_day = np.where(range_expansion > 1.5, 1.0, 0.0)
            low_range_day = np.where(range_expansion < 0.5, 1.0, 0.0)
            price_breakout = np.where(np.abs(price_change_z) > 2.0, 1.0, 0.0)
            
            features[f'range_expansion_{period}'] = range_expansion
            features[f'high_range_day_{period}'] = high_range_day
            features[f'low_range_day_{period}'] = low_range_day
            features[f'price_change_z_{period}'] = price_change_z
            features[f'price_breakout_{period}'] = price_breakout
        
        return features
    
    # =========================================================================
    # サポート・レジスタンス・ローソク足（NumPy実装）
    # =========================================================================
    
    def calculate_support_resistance(self, high: np.ndarray, low: np.ndarray, 
                                   close: np.ndarray, open_prices: np.ndarray = None) -> Dict[str, np.ndarray]:
        """サポート・レジスタンス・ローソク足特徴量の統合計算（NumPy実装）"""
        features = {}
        
        # NumPy配列に変換
        high = self._ensure_numpy_array(high)
        low = self._ensure_numpy_array(low)
        close = self._ensure_numpy_array(close)
        
        # Open価格がない場合はCloseで代用
        if open_prices is None:
            open_prices = np.roll(close, 1)
            open_prices[0] = close[0]
        else:
            open_prices = self._ensure_numpy_array(open_prices)
        
        try:
            # ピボットポイント
            pivot_features = self._safe_calculation(self._calculate_pivot_points_numpy, high, low, close)
            if isinstance(pivot_features, dict):
                features.update(pivot_features)
            
            # プライスチャネル
            price_channel_features = self._safe_calculation(self._calculate_price_channels_numpy, high, low, close)
            if isinstance(price_channel_features, dict):
                features.update(price_channel_features)
            
            # フィボナッチレベル
            fibonacci_features = self._calculate_fibonacci_features_numba(high, low, close)
            features.update(fibonacci_features)
            
            # サポート・レジスタンス検出
            sr_features = self._calculate_support_resistance_features_numba(high, low, close)
            features.update(sr_features)
            
            # ローソク足パターン
            candlestick_features = self._safe_calculation(self._calculate_candlestick_patterns_numpy, open_prices, high, low, close)
            if isinstance(candlestick_features, dict):
                features.update(candlestick_features)
            
            # 複数ローソク足パターン
            multi_candle_features = self._calculate_multi_candle_patterns_numba(open_prices, high, low, close)
            features.update(multi_candle_features)
            
            # ローソク足強度
            candle_strength_features = self._safe_calculation(self._calculate_candle_strength_numpy, open_prices, high, low, close)
            if isinstance(candle_strength_features, dict):
                features.update(candle_strength_features)
                
        except Exception as e:
            logger.error(f"サポート・レジスタンス・ローソク足特徴量計算エラー: {e}")
        
        return features
    
    def _calculate_pivot_points_numpy(self, high: np.ndarray, low: np.ndarray, close: np.ndarray) -> Dict[str, np.ndarray]:
        """ピボットポイント計算（NumPy実装）"""
        
        # 前日の高値、安値、終値
        prev_high = np.roll(high, 1)
        prev_low = np.roll(low, 1)
        prev_close = np.roll(close, 1)
        
        # 先頭を埋める
        prev_high[0] = high[0]
        prev_low[0] = low[0]
        prev_close[0] = close[0]
        
        # ピボットポイント
        pivot_point = (prev_high + prev_low + prev_close) / 3.0
        
        # レジスタンス・サポート
        resistance1 = 2 * pivot_point - prev_low
        support1 = 2 * pivot_point - prev_high
        resistance2 = pivot_point + (prev_high - prev_low)
        support2 = pivot_point - (prev_high - prev_low)
        
        # 現在価格との距離
        distance_to_pivot = 100 * np.divide(close - pivot_point, close, out=np.zeros_like(close), where=(close != 0))
        distance_to_r1 = 100 * np.divide(resistance1 - close, close, out=np.zeros_like(close), where=(close != 0))
        distance_to_r2 = 100 * np.divide(resistance2 - close, close, out=np.zeros_like(close), where=(close != 0))
        distance_to_s1 = 100 * np.divide(close - support1, close, out=np.zeros_like(close), where=(close != 0))
        distance_to_s2 = 100 * np.divide(close - support2, close, out=np.zeros_like(close), where=(close != 0))
        
        # レベル突破検出（NumPy操作）
        above_pivot = np.where(close > pivot_point, 1.0, 0.0)
        above_r1 = np.where(close > resistance1, 1.0, 0.0)
        above_r2 = np.where(close > resistance2, 1.0, 0.0)
        below_s1 = np.where(close < support1, 1.0, 0.0)
        below_s2 = np.where(close < support2, 1.0, 0.0)
        
        return {
            'pivot_point': pivot_point,
            'resistance1': resistance1,
            'resistance2': resistance2,
            'support1': support1,
            'support2': support2,
            'distance_to_pivot': distance_to_pivot,
            'distance_to_r1': distance_to_r1,
            'distance_to_r2': distance_to_r2,
            'distance_to_s1': distance_to_s1,
            'distance_to_s2': distance_to_s2,
            'above_pivot': above_pivot,
            'above_r1': above_r1,
            'above_r2': above_r2,
            'below_s1': below_s1,
            'below_s2': below_s2
        }
    
    def _calculate_price_channels_numpy(self, high: np.ndarray, low: np.ndarray, close: np.ndarray) -> Dict[str, np.ndarray]:
        """プライスチャネル計算（NumPy実装）"""
        features = {}
        
        for period in self.params['price_channel_periods']:
            channel_high = self._rolling_max_numpy(high, period)
            channel_low = self._rolling_min_numpy(low, period)
            channel_mid = (channel_high + channel_low) / 2.0
            
            # チャネル内の価格位置
            channel_position = np.divide(close - channel_low, channel_high - channel_low,
                                       out=np.zeros_like(close),
                                       where=((channel_high - channel_low) != 0))
            
            # チャネル幅
            channel_width = np.divide(channel_high - channel_low, channel_mid,
                                    out=np.zeros_like(channel_mid),
                                    where=(channel_mid != 0))
            
            # ブレイクアウト検出（NumPy操作）
            channel_breakout_up = np.where(close > channel_high, 1.0, 0.0)
            channel_breakout_down = np.where(close < channel_low, 1.0, 0.0)
            
            features[f'channel_high_{period}'] = channel_high
            features[f'channel_low_{period}'] = channel_low
            features[f'channel_mid_{period}'] = channel_mid
            features[f'channel_position_{period}'] = channel_position
            features[f'channel_width_{period}'] = channel_width
            features[f'channel_breakout_up_{period}'] = channel_breakout_up
            features[f'channel_breakout_down_{period}'] = channel_breakout_down
        
        return features
    
    def _calculate_fibonacci_features_numba(self, high: np.ndarray, low: np.ndarray, close: np.ndarray) -> Dict[str, np.ndarray]:
        """フィボナッチリトレースメント特徴量（Numba版）"""
        features = {}
        
        # 複数期間でのフィボナッチレベル計算
        periods = [20, 50, 100]
        fib_levels = [0.236, 0.382, 0.500, 0.618, 0.786]
        
        for period in periods:
            fib_result = self._numba_safe_calculation(
                self._calculate_fibonacci_levels_vectorized, high, low, close, period, fib_levels
            )
            
            if fib_result is not None and len(fib_result.shape) == 3:
                for level_idx, level in enumerate(fib_levels):
                    level_str = str(level).replace('.', '')
                    
                    # リトレースメント
                    if level_idx < fib_result.shape[2]:
                        features[f'fib_retracement_{level_str}_{period}'] = fib_result[:, 0, level_idx]
                        features[f'fib_extension_{level_str}_{period}'] = fib_result[:, 1, level_idx]
                        features[f'near_fib_retracement_{level_str}_{period}'] = fib_result[:, 2, level_idx]
                    else:
                        features[f'fib_retracement_{level_str}_{period}'] = np.full(len(close), np.nan)
                        features[f'fib_extension_{level_str}_{period}'] = np.full(len(close), np.nan)
                        features[f'near_fib_retracement_{level_str}_{period}'] = np.full(len(close), np.nan)
            else:
                # フォールバック
                for level in fib_levels:
                    level_str = str(level).replace('.', '')
                    features[f'fib_retracement_{level_str}_{period}'] = np.full(len(close), np.nan)
                    features[f'fib_extension_{level_str}_{period}'] = np.full(len(close), np.nan)
                    features[f'near_fib_retracement_{level_str}_{period}'] = np.full(len(close), np.nan)
        
        return features
    
    @staticmethod
    @njit(parallel=True, cache=True)
    def _calculate_fibonacci_levels_vectorized(high: np.ndarray, low: np.ndarray, close: np.ndarray, 
                                             period: int, fib_levels: List[float]) -> np.ndarray:
        """フィボナッチレベル計算（完全ベクトル化・並列版）"""
        n = len(high)
        if n < period:
            return np.zeros((n, 3, len(fib_levels)))
        
        results = np.zeros((n, 3, len(fib_levels)))  # [retracement, extension, near_flag]
        
        # 並列計算
        for i in prange(period-1, n):
            swing_high = np.max(high[i-period+1:i+1])
            swing_low = np.min(low[i-period+1:i+1])
            swing_range = swing_high - swing_low
            
            if swing_range > 1e-10:
                for level_idx in range(len(fib_levels)):
                    level = fib_levels[level_idx]
                    
                    fib_retracement = swing_low + level * swing_range
                    fib_extension = swing_high + level * swing_range
                    
                    # フィボナッチレベルとの近接度
                    distance_to_retracement = abs(close[i] - fib_retracement) / (close[i] + 1e-10)
                    near_fib_retracement = 1.0 if distance_to_retracement < 0.01 else 0.0
                    
                    results[i, 0, level_idx] = fib_retracement
                    results[i, 1, level_idx] = fib_extension
                    results[i, 2, level_idx] = near_fib_retracement
        
        return results
    
    def _calculate_candlestick_patterns_numpy(self, open_prices: np.ndarray, high: np.ndarray, 
                                            low: np.ndarray, close: np.ndarray) -> Dict[str, np.ndarray]:
        """ローソク足パターン計算（NumPy実装）"""
        
        # 基本的なローソク足要素
        body_size = np.abs(close - open_prices)
        upper_shadow = high - np.maximum(open_prices, close)
        lower_shadow = np.minimum(open_prices, close) - low
        total_range = high - low
        
        # 正規化（全体レンジに対する比率）
        body_ratio = np.divide(body_size, total_range, out=np.zeros_like(body_size), where=(total_range != 0))
        upper_shadow_ratio = np.divide(upper_shadow, total_range, out=np.zeros_like(upper_shadow), where=(total_range != 0))
        lower_shadow_ratio = np.divide(lower_shadow, total_range, out=np.zeros_like(lower_shadow), where=(total_range != 0))
        
        # ローソク足の方向（NumPy操作）
        is_bullish = np.where(close > open_prices, 1.0, 0.0)
        is_bearish = np.where(close < open_prices, 1.0, 0.0)
        
        # 十字線（Doji）パターン
        doji = np.where(body_ratio < 0.1, 1.0, 0.0)
        long_legged_doji = np.where((body_ratio < 0.1) & (upper_shadow_ratio > 0.3) & (lower_shadow_ratio > 0.3), 1.0, 0.0)
        dragonfly_doji = np.where((body_ratio < 0.1) & (upper_shadow_ratio < 0.1) & (lower_shadow_ratio > 0.3), 1.0, 0.0)
        gravestone_doji = np.where((body_ratio < 0.1) & (upper_shadow_ratio > 0.3) & (lower_shadow_ratio < 0.1), 1.0, 0.0)
        
        # ハンマー・ハンギングマン
        hammer_condition = (lower_shadow_ratio > 0.5) & (upper_shadow_ratio < 0.1) & (body_ratio < 0.3)
        
        # シューティングスター・インバーテッドハンマー
        shooting_star_condition = (upper_shadow_ratio > 0.5) & (lower_shadow_ratio < 0.1) & (body_ratio < 0.3)
        
        # マルボウズ（影のないローソク足）
        marubozu = np.where((upper_shadow_ratio < 0.05) & (lower_shadow_ratio < 0.05) & (body_ratio > 0.9), 1.0, 0.0)
        
        # スピニングトップ
        spinning_top = np.where((body_ratio < 0.3) & (upper_shadow_ratio > 0.2) & (lower_shadow_ratio > 0.2), 1.0, 0.0)
        
        # ハンマー系パターンの分類（NumPy操作）
        hammer = np.where(hammer_condition & (is_bullish == 1), 1.0, 0.0)
        hanging_man = np.where(hammer_condition & (is_bearish == 1), 1.0, 0.0)
        shooting_star = np.where(shooting_star_condition & (is_bearish == 1), 1.0, 0.0)
        inverted_hammer = np.where(shooting_star_condition & (is_bullish == 1), 1.0, 0.0)
        
        # マルボウズの分類
        white_marubozu = np.where(marubozu & (is_bullish == 1), 1.0, 0.0)
        black_marubozu = np.where(marubozu & (is_bearish == 1), 1.0, 0.0)
        
        return {
            'body_size': body_size,
            'upper_shadow': upper_shadow,
            'lower_shadow': lower_shadow,
            'body_ratio': body_ratio,
            'upper_shadow_ratio': upper_shadow_ratio,
            'lower_shadow_ratio': lower_shadow_ratio,
            'is_bullish': is_bullish,
            'is_bearish': is_bearish,
            'doji': doji,
            'long_legged_doji': long_legged_doji,
            'dragonfly_doji': dragonfly_doji,
            'gravestone_doji': gravestone_doji,
            'hammer': hammer,
            'hanging_man': hanging_man,
            'shooting_star': shooting_star,
            'inverted_hammer': inverted_hammer,
            'marubozu': marubozu,
            'white_marubozu': white_marubozu,
            'black_marubozu': black_marubozu,
            'spinning_top': spinning_top
        }
    
    def _calculate_multi_candle_patterns_numba(self, open_prices: np.ndarray, high: np.ndarray, 
                                             low: np.ndarray, close: np.ndarray) -> Dict[str, np.ndarray]:
        """複数ローソク足パターン計算（Numba版）"""
        features = {}
        
        multi_patterns = self._numba_safe_calculation(
            self._calculate_multi_patterns_vectorized, open_prices, high, low, close
        )
        
        if multi_patterns is not None and len(multi_patterns.shape) == 2:
            pattern_names = ['bullish_engulfing', 'bearish_engulfing', 'bullish_harami', 'bearish_harami', 
                           'gap_up', 'gap_down', 'has_gap']
            
            for i, name in enumerate(pattern_names):
                if i < multi_patterns.shape[1]:
                    features[name] = multi_patterns[:, i]
                else:
                    features[name] = np.full(len(close), np.nan)
        else:
            # フォールバック
            pattern_names = ['bullish_engulfing', 'bearish_engulfing', 'bullish_harami', 'bearish_harami', 
                           'gap_up', 'gap_down', 'has_gap']
            for name in pattern_names:
                features[name] = np.full(len(close), np.nan)
        
        return features
    
    @staticmethod
    @njit(cache=True)
    def _calculate_multi_patterns_vectorized(open_prices: np.ndarray, high: np.ndarray, 
                                           low: np.ndarray, close: np.ndarray) -> np.ndarray:
        """複数ローソク足パターン計算（Numba最適化版）"""
        n = len(close)
        if n < 2:
            return np.zeros((n, 7))
        
        results = np.zeros((n, 7))
        
        for i in range(1, n):
            # エンゴルフィング（包み足）
            prev_body_top = max(open_prices[i-1], close[i-1])
            prev_body_bottom = min(open_prices[i-1], close[i-1])
            curr_body_top = max(open_prices[i], close[i])
            curr_body_bottom = min(open_prices[i], close[i])
            
            # 強気エンゴルフィング
            if (close[i-1] < open_prices[i-1] and close[i] > open_prices[i] and 
                curr_body_bottom < prev_body_bottom and curr_body_top > prev_body_top):
                results[i, 0] = 1.0
            
            # 弱気エンゴルフィング
            if (close[i-1] > open_prices[i-1] and close[i] < open_prices[i] and 
                curr_body_bottom < prev_body_bottom and curr_body_top > prev_body_top):
                results[i, 1] = 1.0
            
            # はらみ足（Harami）
            if (curr_body_top < prev_body_top and curr_body_bottom > prev_body_bottom):
                if close[i-1] < open_prices[i-1] and close[i] > open_prices[i]:
                    results[i, 2] = 1.0  # 強気はらみ
                elif close[i-1] > open_prices[i-1] and close[i] < open_prices[i]:
                    results[i, 3] = 1.0  # 弱気はらみ
            
            # ギャップ（窓開け）
            prev_high = high[i-1]
            prev_low = low[i-1]
            curr_low = low[i]
            curr_high = high[i]
            
            if curr_low > prev_high:
                results[i, 4] = (curr_low - prev_high) / (prev_high + 1e-10)  # gap_up
                results[i, 6] = 1.0  # has_gap
            elif curr_high < prev_low:
                results[i, 5] = (prev_low - curr_high) / (prev_low + 1e-10)  # gap_down
                results[i, 6] = 1.0  # has_gap
        
        return results
    
    def _calculate_candle_strength_numpy(self, open_prices: np.ndarray, high: np.ndarray, 
                                       low: np.ndarray, close: np.ndarray) -> Dict[str, np.ndarray]:
        """ローソク足強度計算（NumPy実装）"""
        features = {}
        
        # 基本要素
        body_size = np.abs(close - open_prices)
        total_range = high - low
        
        # 複数期間での相対的評価
        periods = [5, 10, 20]
        
        for period in periods:
            # 実体サイズのパーセンタイル計算
            body_percentile = np.zeros_like(body_size)
            range_percentile = np.zeros_like(total_range)
            
            for i in range(period-1, len(body_size)):
                window_body = body_size[i-period+1:i+1]
                window_range = total_range[i-period+1:i+1]
                
                body_rank = np.sum(window_body <= body_size[i])
                range_rank = np.sum(window_range <= total_range[i])
                
                body_percentile[i] = body_rank / len(window_body)
                range_percentile[i] = range_rank / len(window_range)
            
            # 強いローソク足の検出（NumPy操作）
            strong_candle = np.where(body_percentile > 0.8, 1.0, 0.0)
            weak_candle = np.where(body_percentile < 0.2, 1.0, 0.0)
            high_volatility_candle = np.where(range_percentile > 0.8, 1.0, 0.0)
            
            features[f'body_percentile_{period}'] = body_percentile
            features[f'range_percentile_{period}'] = range_percentile
            features[f'strong_candle_{period}'] = strong_candle
            features[f'weak_candle_{period}'] = weak_candle
            features[f'high_volatility_candle_{period}'] = high_volatility_candle
        
        # ローソク足の連続性
        consecutive_result = self._numba_safe_calculation(
            self._calculate_consecutive_candles_vectorized, open_prices, close
        )
        
        if consecutive_result is not None and len(consecutive_result.shape) == 2:
            features['consecutive_bullish'] = consecutive_result[:, 0]
            features['consecutive_bearish'] = consecutive_result[:, 1]
            features['long_bullish_streak'] = consecutive_result[:, 2]
            features['long_bearish_streak'] = consecutive_result[:, 3]
        else:
            # フォールバック
            n = len(close)
            features['consecutive_bullish'] = np.full(n, np.nan)
            features['consecutive_bearish'] = np.full(n, np.nan)
            features['long_bullish_streak'] = np.full(n, np.nan)
            features['long_bearish_streak'] = np.full(n, np.nan)
        
        return features
    
    @staticmethod
    @njit(cache=True)
    def _calculate_consecutive_candles_vectorized(open_prices: np.ndarray, close: np.ndarray) -> np.ndarray:
        """連続ローソク足計算（Numba最適化版）"""
        n = len(close)
        results = np.zeros((n, 4))
        
        bull_count = 0
        bear_count = 0
        
        for i in range(n):
            if close[i] > open_prices[i]:
                bull_count += 1
                bear_count = 0
            elif close[i] < open_prices[i]:
                bear_count += 1
                bull_count = 0
            else:
                bull_count = 0
                bear_count = 0
            
            results[i, 0] = bull_count  # consecutive_bullish
            results[i, 1] = bear_count  # consecutive_bearish
            results[i, 2] = 1.0 if bull_count >= 3 else 0.0  # long_bullish_streak
            results[i, 3] = 1.0 if bear_count >= 3 else 0.0  # long_bearish_streak
        
        return results
    
    def _calculate_support_resistance_features_numba(self, high: np.ndarray, low: np.ndarray, close: np.ndarray) -> Dict[str, np.ndarray]:
        """サポート・レジスタンス特徴量（Numba版）"""
        features = {}
        
        lookback_periods = [20, 50, 100]
        
        for period in lookback_periods:
            sr_result = self._numba_safe_calculation(
                self._calculate_support_resistance_vectorized, high, low, close, period
            )
            
            if sr_result is not None and len(sr_result.shape) == 2:
                features[f'nearest_resistance_{period}'] = sr_result[:, 0]
                features[f'nearest_support_{period}'] = sr_result[:, 1]
                features[f'resistance_distance_{period}'] = sr_result[:, 2]
                features[f'support_distance_{period}'] = sr_result[:, 3]
                features[f'resistance_strength_{period}'] = sr_result[:, 4]
                features[f'support_strength_{period}'] = sr_result[:, 5]
            else:
                # フォールバック
                n = len(close)
                features[f'nearest_resistance_{period}'] = np.full(n, np.nan)
                features[f'nearest_support_{period}'] = np.full(n, np.nan)
                features[f'resistance_distance_{period}'] = np.full(n, np.nan)
                features[f'support_distance_{period}'] = np.full(n, np.nan)
                features[f'resistance_strength_{period}'] = np.full(n, np.nan)
                features[f'support_strength_{period}'] = np.full(n, np.nan)
        
        return features
    
    @staticmethod
    @njit(parallel=True, cache=True)
    def _calculate_support_resistance_vectorized(high: np.ndarray, low: np.ndarray, 
                                               close: np.ndarray, period: int) -> np.ndarray:
        """サポート・レジスタンス計算（完全ベクトル化・並列版）"""
        n = len(high)
        if n < period:
            return np.zeros((n, 6))
        
        results = np.zeros((n, 6))
        
        # 並列計算
        for i in prange(period-1, n):
            window_high = high[i-period+1:i+1]
            window_low = low[i-period+1:i+1]
            current_price = close[i]
            
            # レジスタンス検出（現在価格より上の高値）
            resistance_levels = window_high[window_high > current_price]
            if len(resistance_levels) > 0:
                nearest_resistance = np.min(resistance_levels)
                # レジスタンス強度（そのレベルでの反発回数の近似）
                resistance_strength = np.sum(np.abs(window_high - nearest_resistance) < 0.001 * current_price)
            else:
                nearest_resistance = current_price * 1.02  # デフォルト値
                resistance_strength = 0
            
            # サポート検出（現在価格より下の安値）
            support_levels = window_low[window_low < current_price]
            if len(support_levels) > 0:
                nearest_support = np.max(support_levels)
                # サポート強度（そのレベルでの反発回数の近似）
                support_strength = np.sum(np.abs(window_low - nearest_support) < 0.001 * current_price)
            else:
                nearest_support = current_price * 0.98  # デフォルト値
                support_strength = 0
            
            # 距離計算
            resistance_distance = (nearest_resistance - current_price) / (current_price + 1e-10)
            support_distance = (current_price - nearest_support) / (current_price + 1e-10)
            
            results[i, 0] = nearest_resistance
            results[i, 1] = nearest_support
            results[i, 2] = resistance_distance
            results[i, 3] = support_distance
            results[i, 4] = resistance_strength
            results[i, 5] = support_strength
        
        return results
    
    # =========================================================================
    # 統合特徴量計算メソッド（Polars第一主義・完全最適化版）
    # =========================================================================
    
    def calculate_all_features_polars_optimized(self, high: np.ndarray, low: np.ndarray, 
                                              close: np.ndarray, volume: np.ndarray,
                                              open_prices: np.ndarray = None) -> Dict[str, np.ndarray]:
        """
        全特徴量の統合計算（Polars第一主義・完全最適化版）
        
        設計原則:
        1. 脱・行ごとループ: 全てチャンク処理
        2. Polars第一主義: 標準的計算は全てPolarsで実装
        3. Numbaの限定投入: カスタムアルゴリズムのみ
        4. 数秒〜数十秒での完了を目指す
        """
        logger.info("全特徴量計算開始 - Polars第一主義最適化版")
        start_time = time.time()
        
        all_features = {}
        
        try:
            # Open価格がない場合はCloseで代用
            if open_prices is None:
                open_prices = np.roll(close, 1)
                open_prices[0] = close[0]
            
           
            # =========================================================================
            # Block 6: ADX・基本オシレーター
            # =========================================================================
            logger.info("ADX・基本オシレーター計算中...")
            osc_start = time.time()
            
            # ADX
            adx_features = self.calculate_adx_features(high, low, close)
            all_features.update(adx_features)
            
            # Parabolic SAR
            sar_features = self.calculate_parabolic_sar_features(high, low, close)
            all_features.update(sar_features)
            
            # CCI
            cci_features = self.calculate_cci_features(high, low, close)
            all_features.update(cci_features)
            
            # Williams %R
            williams_features = self.calculate_williams_r_features(high, low, close)
            all_features.update(williams_features)
            
            # Aroon
            aroon_features = self.calculate_aroon_features(high, low)
            all_features.update(aroon_features)
            
            # Ultimate Oscillator
            uo_features = self.calculate_ultimate_oscillator_features(high, low, close)
            all_features.update(uo_features)
            
            logger.info(f"ADX・基本オシレーター完了: {len(adx_features) + len(sar_features) + len(cci_features) + len(williams_features) + len(aroon_features) + len(uo_features)}個 ({time.time() - osc_start:.2f}秒)")
            
            # =========================================================================
            # Block 7: 出来高関連指標
            # =========================================================================
            logger.info("出来高関連指標計算中...")
            volume_start = time.time()
            
            volume_features = self.calculate_volume_features(high, low, close, volume)
            all_features.update(volume_features)
            
            logger.info(f"出来高関連指標完了: {len(volume_features)}個 ({time.time() - volume_start:.2f}秒)")
            
            # =========================================================================
            # Block 8: 移動平均線・トレンド分析
            # =========================================================================
            logger.info("移動平均線・トレンド分析計算中...")
            ma_start = time.time()
            
            ma_features = self.calculate_moving_averages(close)
            all_features.update(ma_features)
            
            logger.info(f"移動平均線・トレンド分析完了: {len(ma_features)}個 ({time.time() - ma_start:.2f}秒)")
            
            # =========================================================================
            # Block 9: ボラティリティ・バンド指標
            # =========================================================================
            logger.info("ボラティリティ・バンド指標計算中...")
            vol_start = time.time()
            
            volatility_features = self.calculate_volatility_bands(high, low, close)
            all_features.update(volatility_features)
            
            logger.info(f"ボラティリティ・バンド指標完了: {len(volatility_features)}個 ({time.time() - vol_start:.2f}秒)")
            
            # =========================================================================
            # Block 10: サポート・レジスタンス・ローソク足
            # =========================================================================
            logger.info("サポート・レジスタンス・ローソク足計算中...")
            sr_start = time.time()
            
            sr_features = self.calculate_support_resistance(high, low, close, open_prices)
            all_features.update(sr_features)
            
            logger.info(f"サポート・レジスタンス・ローソク足完了: {len(sr_features)}個 ({time.time() - sr_start:.2f}秒)")
            
           
            
            # =========================================================================
            # 最終処理・品質保証
            # =========================================================================
            logger.info("品質保証・最終処理中...")
            qa_start = time.time()
            
            # 品質チェック・統計情報
            quality_report = self.generate_quality_report(all_features)
            
            # 異常値処理（必要に応じて）
            all_features = self.apply_final_cleaning(all_features)
            
            logger.info(f"品質保証完了 ({time.time() - qa_start:.2f}秒)")
            
        except Exception as e:
            logger.error(f"特徴量計算中にエラーが発生: {e}")
            import traceback
            traceback.print_exc()
        
        total_time = time.time() - start_time
        logger.info(f"全特徴量計算完了: {len(all_features)}個の特徴量を{total_time:.2f}秒で生成")
        logger.info(f"計算統計: NumPy={self.calculation_stats.get('numpy_calculations', 0)}, Numba={self.calculation_stats.get('numba_calculations', 0)}")
        
        return all_features
    
    # =========================================================================
    # 品質保証システム（Polars最適化版）
    # =========================================================================
    
    def generate_quality_report(self, features: Dict[str, np.ndarray]) -> Dict[str, Any]:
        """品質レポート生成（NumPy主体+Polars補助版）"""
        
        if not features:
            return {'status': 'no_features', 'total_features': 0}
        
        try:
            # NumPyで直接計算（Polarsの複雑な操作を回避）
            total_features = len(features)
            total_points = len(next(iter(features.values())))
            
            # 各メトリクスをNumPyで計算
            null_counts = {}
            finite_counts = {}
            inf_counts = {}
            
            for name, values in features.items():
                null_counts[name] = np.sum(np.isnan(values))
                finite_counts[name] = np.sum(np.isfinite(values))
                inf_counts[name] = np.sum(np.isinf(values))
            
            # 比率計算
            null_ratios = np.array(list(null_counts.values())) / total_points
            finite_ratios = np.array(list(finite_counts.values())) / total_points
            inf_ratios = np.array(list(inf_counts.values())) / total_points
            
            # 統計値
            null_ratio_avg = float(np.mean(null_ratios))
            finite_ratio_avg = float(np.mean(finite_ratios))
            inf_ratio_avg = float(np.mean(inf_ratios))
            overall_quality = finite_ratio_avg
            
            # カテゴリ別カウント
            high_quality_count = int(np.sum(finite_ratios > 0.95))
            medium_quality_count = int(np.sum((finite_ratios >= 0.8) & (finite_ratios <= 0.95)))
            low_quality_count = int(np.sum(finite_ratios < 0.8))
            
            quality_report = {
                'status': 'completed',
                'total_features': total_features,
                'data_points': total_points,
                'overall_quality_score': overall_quality,
                'null_ratio_avg': null_ratio_avg,
                'inf_ratio_avg': inf_ratio_avg,
                'finite_ratio_avg': finite_ratio_avg,
                'high_quality_features': high_quality_count,
                'medium_quality_features': medium_quality_count,
                'low_quality_features': low_quality_count,
                'calculation_stats': getattr(self, 'calculation_stats', {}),
                'polars_optimization_ratio': (
                    getattr(self, 'calculation_stats', {}).get('polars_calculations', 0) / 
                    max(1, getattr(self, 'calculation_stats', {}).get('total_calculations', 1))
                )
            }
            
            # 警告・推奨事項
            warnings = []
            recommendations = []
            
            if overall_quality < 0.9:
                warnings.append(f"全体品質スコアが低下: {overall_quality:.3f}")
                recommendations.append("数値安定化処理の強化を検討")
            
            if inf_ratio_avg > 0.01:
                warnings.append(f"無限値の比率が高い: {inf_ratio_avg:.3f}")
                recommendations.append("数値計算の安定性を改善")
            
            if null_ratio_avg > 0.05:
                warnings.append(f"欠損値の比率が高い: {null_ratio_avg:.3f}")
                recommendations.append("欠損値補完アルゴリズムの改善")
            
            quality_report['warnings'] = warnings
            quality_report['recommendations'] = recommendations
            
            return quality_report
            
        except Exception as e:
            logger.error(f"品質レポート生成エラー: {e}")
            return {
                'status': 'error',
                'error_message': str(e),
                'total_features': len(features)
            }
        
    def apply_final_cleaning(self, features: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """最終クリーニング処理（NumPyベース版）"""
        
        if not features:
            return features
        
        try:
            # NumPyで直接クリーニング処理を実行
            cleaned_features = {}
            
            for feature_name, values in features.items():
                # NaN/Inf値の処理
                cleaned_values = values.copy()
                
                # NaN -> 0.0
                nan_mask = np.isnan(cleaned_values)
                cleaned_values[nan_mask] = 0.0
                
                # Inf -> クリップ
                inf_mask = np.isinf(cleaned_values)
                if np.any(inf_mask):
                    pos_inf_mask = (cleaned_values == np.inf)
                    neg_inf_mask = (cleaned_values == -np.inf)
                    cleaned_values[pos_inf_mask] = 1e10
                    cleaned_values[neg_inf_mask] = -1e10
                
                # 異常に大きな値をクランプ
                cleaned_values = np.clip(cleaned_values, -1e10, 1e10)
                
                cleaned_features[feature_name] = cleaned_values
            
            return cleaned_features
            
        except Exception as e:
            logger.error(f"最終クリーニングエラー: {e}")
            # エラー時は元の特徴量を返す
            return features
    
    def get_calculation_summary_optimized(self) -> Dict[str, Any]:
        """最適化版計算統計サマリー"""
        stats = self.calculation_stats
        
        total_calc = max(1, stats['total_calculations'])
        success_rate = stats['successful_calculations'] / total_calc
        numpy_ratio = stats.get('numpy_calculations', 0) / total_calc
        numba_ratio = stats.get('numba_calculations', 0) / total_calc
        
        avg_time = np.mean(stats['computation_times']) if stats['computation_times'] else 0.0
        
        return {
            'total_calculations': total_calc,
            'success_rate': success_rate,
            'numpy_calculation_ratio': numpy_ratio,
            'numba_calculation_ratio': numba_ratio,
            'avg_computation_time_ms': avg_time * 1000,
            'total_computation_time_sec': sum(stats['computation_times']),
            'optimization_level': 'numpy_first',
            'performance_grade': (
                'S' if avg_time < 0.01 and success_rate > 0.95 and numba_ratio > 0.5 else
                'A' if avg_time < 0.05 and success_rate > 0.9 else
                'B' if avg_time < 0.1 and success_rate > 0.8 else
                'C'
            )
        }
    
    # =========================================================================
    # 互換性メソッド（既存インターフェースとの接続）
    # =========================================================================
    
    def calculate_all_advanced_features(self, prices: np.ndarray, 
                                       high: np.ndarray = None, 
                                       low: np.ndarray = None,
                                       volume: np.ndarray = None) -> Dict[str, np.ndarray]:
        """
        互換性メソッド: 既存インターフェースから新しい最適化版へのブリッジ
        """
        logger.info("互換性モード: 既存インターフェースから最適化版への変換")
        
        # データ形式の調整
        if high is None:
            high = prices
        if low is None:
            low = prices
        if volume is None:
            volume = np.ones_like(prices)
        
        # 最適化版メソッドを呼び出し
        return self.calculate_all_features_polars_optimized(high, low, prices, volume)
    
    def calculate_all_missing_features(self, high: np.ndarray, low: np.ndarray, close: np.ndarray, 
                                     volume: np.ndarray, open_prices: np.ndarray = None) -> Dict[str, np.ndarray]:
        """
        互換性メソッド: 欠落特徴量計算の最適化版へのリダイレクト
        """
        logger.info("互換性モード: 欠落特徴量計算を最適化版で実行")
        
        return self.calculate_all_features_polars_optimized(high, low, close, volume, open_prices)
    
    # =========================================================================
    # パフォーマンス・ベンチマーク機能
    # =========================================================================
    
    def benchmark_performance(self, data_sizes: List[int] = None) -> Dict[str, Any]:
        """パフォーマンス・ベンチマーク実行"""
        
        if data_sizes is None:
            data_sizes = [1000, 10000, 100000, 1000000]
        
        benchmark_results = {}
        
        for size in data_sizes:
            logger.info(f"ベンチマーク実行中: データサイズ {size}")
            
            # テストデータ生成
            np.random.seed(42)
            high = np.random.randn(size).cumsum() + 100
            low = high - np.random.exponential(1, size)
            close = low + np.random.random(size) * (high - low)
            volume = np.random.exponential(1000, size)
            
            # 計算時間測定
            start_time = time.time()
            
            try:
                features = self.calculate_all_features_polars_optimized(high, low, close, volume)
                execution_time = time.time() - start_time
                
                benchmark_results[f'size_{size}'] = {
                    'execution_time_sec': execution_time,
                    'features_generated': len(features),
                    'data_points_per_sec': size / execution_time,
                    'features_per_sec': len(features) / execution_time,
                    'memory_efficiency': size * len(features) / (execution_time * 1e6),  # MB/sec
                    'status': 'success'
                }
                
                logger.info(f"ベンチマーク完了 (サイズ {size}): {execution_time:.2f}秒, {len(features)}特徴量")
                
            except Exception as e:
                logger.error(f"ベンチマーク失敗 (サイズ {size}): {e}")
                benchmark_results[f'size_{size}'] = {
                    'status': 'failed',
                    'error': str(e)
                }
        
        # 総合評価
        successful_results = [r for r in benchmark_results.values() if r.get('status') == 'success']
        
        if successful_results:
            avg_throughput = np.mean([r['data_points_per_sec'] for r in successful_results])
            benchmark_results['summary'] = {
                'average_throughput_points_per_sec': avg_throughput,
                'performance_tier': (
                    'Ultra-Fast' if avg_throughput > 1e6 else
                    'Very-Fast' if avg_throughput > 1e5 else
                    'Fast' if avg_throughput > 1e4 else
                    'Standard'
                ),
                'optimization_status': 'polars_optimized',
                'recommendation': 'Performance target achieved' if avg_throughput > 1e5 else 'Consider further optimization'
            }
        
        return benchmark_results
    
    # =========================================================================
    # デバッグ・診断機能
    # =========================================================================
    
    def diagnose_feature_quality(self, features: Dict[str, np.ndarray]) -> Dict[str, Any]:
        """特徴量品質の詳細診断"""
        
        if not features:
            return {'status': 'no_features'}
        
        diagnosis = {
            'feature_count': len(features),
            'data_length': len(next(iter(features.values()))),
            'problematic_features': [],
            'excellent_features': [],
            'warning_features': [],
            'overall_health': 'unknown'
        }
        
        for feature_name, values in features.items():
            feature_stats = {
                'name': feature_name,
                'finite_ratio': np.sum(np.isfinite(values)) / len(values),
                'null_ratio': np.sum(np.isnan(values)) / len(values),
                'inf_ratio': np.sum(np.isinf(values)) / len(values),
                'zero_ratio': np.sum(values == 0.0) / len(values),
                'variance': np.var(values[np.isfinite(values)]) if np.any(np.isfinite(values)) else 0.0,
                'mean': np.mean(values[np.isfinite(values)]) if np.any(np.isfinite(values)) else 0.0
            }
            
            # 品質分類
            if feature_stats['finite_ratio'] > 0.95 and feature_stats['variance'] > 1e-10:
                diagnosis['excellent_features'].append(feature_stats)
            elif feature_stats['finite_ratio'] < 0.8 or feature_stats['inf_ratio'] > 0.01:
                diagnosis['problematic_features'].append(feature_stats)
            elif feature_stats['zero_ratio'] > 0.9 or feature_stats['variance'] < 1e-12:
                diagnosis['warning_features'].append(feature_stats)
        
        # 全体健全性評価
        excellent_count = len(diagnosis['excellent_features'])
        problematic_count = len(diagnosis['problematic_features'])
        total_count = len(features)
        
        if excellent_count / total_count > 0.8 and problematic_count / total_count < 0.05:
            diagnosis['overall_health'] = 'excellent'
        elif excellent_count / total_count > 0.6 and problematic_count / total_count < 0.15:
            diagnosis['overall_health'] = 'good'
        elif problematic_count / total_count < 0.3:
            diagnosis['overall_health'] = 'acceptable'
        else:
            diagnosis['overall_health'] = 'poor'
        
        return diagnosis
    
    # =========================================================================
    # メモリ効率化機能
    # =========================================================================
    
    def optimize_memory_usage(self, features: Dict[str, np.ndarray], 
                            precision: str = 'float32') -> Dict[str, np.ndarray]:
        """メモリ使用量最適化（データ型変換）"""
        
        optimized_features = {}
        
        for feature_name, values in features.items():
            try:
                if precision == 'float32':
                    # float64 -> float32変換
                    optimized_values = values.astype(np.float32)
                elif precision == 'float16':
                    # float64 -> float16変換（精度は落ちるが大幅メモリ削減）
                    optimized_values = values.astype(np.float16)
                else:
                    optimized_values = values
                
                optimized_features[feature_name] = optimized_values
                
            except Exception as e:
                logger.warning(f"特徴量 {feature_name} の型変換に失敗: {e}")
                optimized_features[feature_name] = values
        
        return optimized_features
    
    # =========================================================================
    # バッチ処理機能
    # =========================================================================
    
    def calculate_features_batch(self, data_batches: List[Dict[str, np.ndarray]], 
                               batch_size: int = None) -> List[Dict[str, np.ndarray]]:
        """バッチ処理による特徴量計算"""
        
        if batch_size is None:
            batch_size = len(data_batches)
        
        results = []
        
        for i in range(0, len(data_batches), batch_size):
            batch = data_batches[i:i+batch_size]
            batch_results = []
            
            logger.info(f"バッチ {i//batch_size + 1}/{(len(data_batches)-1)//batch_size + 1} 処理中...")
            
            for data_dict in batch:
                try:
                    high = data_dict.get('high')
                    low = data_dict.get('low') 
                    close = data_dict.get('close')
                    volume = data_dict.get('volume')
                    open_prices = data_dict.get('open')
                    
                    features = self.calculate_all_features_polars_optimized(
                        high, low, close, volume, open_prices
                    )
                    batch_results.append(features)
                    
                except Exception as e:
                    logger.error(f"バッチ処理エラー: {e}")
                    batch_results.append({})
            
            results.extend(batch_results)
        
        return results
    
    # =========================================================================
    # 設定管理
    # =========================================================================
    
    def get_feature_config(self) -> Dict[str, Any]:
        """現在の特徴量計算設定を取得"""
        
        return {
            'version': '2.0.0-polars-optimized',
            'optimization_mode': 'polars_first',
            'numba_enabled': True,
            'parallel_processing': True,
            'memory_optimization': True,
            'supported_features': [
                'basic_technical', 'mfdfa', 'microstructure_noise', 'shock_model',
                'multiscale_volatility', 'emd', 'statistical_moments', 'robust_statistics',
                'spectral', 'wavelet', 'chaos', 'hilbert', 'adx', 'sar', 'cci',
                'williams_r', 'aroon', 'ultimate_oscillator', 'volume', 'moving_averages',
                'volatility_bands', 'support_resistance', 'information_theory',
                'interdisciplinary_analogies'
            ],
            'interdisciplinary_modules': [
                'game_theory', 'molecular_science', 'network_science', 'acoustics',
                'linguistics', 'aesthetics', 'music_theory', 'astronomy', 'biomechanics'
            ],
            'performance_targets': {
                'throughput_points_per_sec': 100000,
                'memory_efficiency_mb_per_sec': 50,
                'quality_threshold': 0.95
            }
        }
    
    def set_optimization_mode(self, mode: str) -> bool:
        """最適化モードの設定"""
        
        valid_modes = ['polars_first', 'numpy_only', 'numba_aggressive', 'balanced']
        
        if mode not in valid_modes:
            logger.error(f"無効な最適化モード: {mode}")
            return False
        
        self.optimization_mode = mode
        logger.info(f"最適化モード設定: {mode}")
        
        return True

# =============================================================================
# OutputManagerクラス（修正版）
# =============================================================================

class OutputManager:
    """
    出力管理クラス（統計キー問題修正版）
    """
    def __init__(self, output_base_path: Path = None):
        if output_base_path:
            self.output_base_path = Path(output_base_path)
        else:
            self.output_base_path = Path('/workspaces/project_forge/data/2_feature_value/feature_value_a_vast_universe')
        
        self.output_base_path.mkdir(parents=True, exist_ok=True)
        
        # 修正: 統計情報辞書を完全に初期化
        self.output_stats = {
            'files_created': 0,
            'total_features_saved': 0,
            'compression_ratios': [],
            'save_times': [],
            'file_sizes_mb': [],
            'total_data_points': 0,
            'last_save_path': None,
            'creation_time': time.time()
        }
        
        logger.info(f"OutputManager初期化: {self.output_base_path}")

    def save_features(self, features_dict: Dict[str, np.ndarray], output_filename: str) -> Path:
        """Polarsでの効率的保存（統計記録付き）"""
        start_time = time.time()
        
        try:
            if not features_dict:
                raise ValueError("保存する特徴量データがありません")

            # 特徴量長の一致確認と調整
            feature_lengths = [len(v) for v in features_dict.values()]
            if len(set(feature_lengths)) > 1:
                min_length = min(feature_lengths)
                logger.warning(f"特徴量の長さが不一致です。最小長 {min_length} に揃えます。")
                features_dict = {name: v[:min_length] for name, v in features_dict.items()}

            # 特徴量値のクリーニング
            cleaned_features = {}
            for name, vals in features_dict.items():
                cleaned_features[name] = self._clean_feature_values(vals)

            # Polars DataFrameに変換
            df = pl.DataFrame(cleaned_features)
            
            # 出力パス生成
            if not output_filename.endswith('.parquet'):
                output_filename += '.parquet'
            output_path = self.output_base_path / output_filename
            
            # Parquetファイルとして保存
            df.write_parquet(str(output_path), compression='snappy')
            
            # 統計情報更新
            self._update_save_statistics(output_path, features_dict, start_time)
            
            # メタデータ保存
            self._save_metadata(output_path, {}, features_dict)
            
            logger.info(f"特徴量保存完了: {output_path} ({len(features_dict)}特徴量)")
            return output_path
            
        except Exception as e:
            logger.error(f"特徴量保存エラー: {e}", exc_info=True)
            raise

    def _update_save_statistics(self, output_path: Path, features_dict: Dict[str, np.ndarray], start_time: float):
        """保存統計の更新"""
        try:
            save_time = time.time() - start_time
            
            # ファイルサイズ取得
            file_size_mb = output_path.stat().st_size / (1024 * 1024)
            
            # 圧縮率計算
            if file_size_mb > 0:
                uncompressed_mb = sum(v.nbytes for v in features_dict.values()) / (1024 * 1024)
                compression_ratio = uncompressed_mb / file_size_mb if file_size_mb > 0 else 1.0
            else:
                compression_ratio = 1.0
            
            # データポイント数計算
            data_points = len(next(iter(features_dict.values()))) if features_dict else 0
            
            # 統計更新
            self.output_stats['files_created'] += 1
            self.output_stats['total_features_saved'] += len(features_dict)
            self.output_stats['save_times'].append(save_time)
            self.output_stats['file_sizes_mb'].append(file_size_mb)
            self.output_stats['compression_ratios'].append(compression_ratio)
            self.output_stats['total_data_points'] += data_points
            self.output_stats['last_save_path'] = str(output_path)
            
            logger.debug(f"統計更新: ファイルサイズ {file_size_mb:.2f}MB, 圧縮率 {compression_ratio:.1f}x, 保存時間 {save_time:.2f}秒")
            
        except Exception as e:
            logger.warning(f"統計更新エラー: {e}")

    def save_final_summary_metadata(self, summary_data: Dict[str, Any]) -> bool:
        """最終サマリーメタデータJSONを保存"""
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        metadata_path = self.output_base_path / f"processing_metadata_summary_{timestamp}.json"
        
        try:
            # 出力統計を追加
            enhanced_summary = summary_data.copy()
            enhanced_summary['output_statistics'] = self.output_stats.copy()
            enhanced_summary['metadata_generation_time'] = time.time()
            
            with open(metadata_path, 'w', encoding='utf-8') as f:
                json.dump(enhanced_summary, f, indent=2, ensure_ascii=False, cls=NumpyJSONEncoder)
            
            logger.info(f"統合メタデータ保存完了: {metadata_path}")
            return True
            
        except Exception as e:
            logger.error(f"統合メタデータ保存エラー: {e}", exc_info=True)
            return False
    
    def _clean_feature_values(self, values: np.ndarray) -> np.ndarray:
        """特徴量値のクリーニング"""
        if not isinstance(values, np.ndarray):
            values = np.asarray(values, dtype=np.float64)
        
        # NaN/Infを0で置換
        cleaned = values.copy()
        
        # NaN処理
        nan_mask = np.isnan(cleaned)
        if np.any(nan_mask):
            cleaned[nan_mask] = 0.0
        
        # Inf処理
        inf_mask = np.isinf(cleaned)
        if np.any(inf_mask):
            cleaned[inf_mask] = 0.0
        
        # 極端な値のクリッピング
        cleaned = np.clip(cleaned, -1e10, 1e10)
        
        return cleaned
    
    def _save_metadata(self, data_path: Path, metadata: Dict[str, Any], 
                      features_dict: Dict[str, np.ndarray]):
        """メタデータファイル保存"""
        metadata_path = data_path.with_suffix('.meta.json')
        
        try:
            # 特徴量統計を計算
            feature_stats = {}
            for name, values in features_dict.items():
                finite_values = values[np.isfinite(values)]
                if len(finite_values) > 0:
                    feature_stats[name] = {
                        'mean': float(np.mean(finite_values)),
                        'std': float(np.std(finite_values)),
                        'min': float(np.min(finite_values)),
                        'max': float(np.max(finite_values)),
                        'median': float(np.median(finite_values)),
                        'nan_count': int(np.isnan(values).sum()),
                        'inf_count': int(np.isinf(values).sum()),
                        'finite_count': len(finite_values),
                        'total_count': len(values)
                    }
                else:
                    feature_stats[name] = {
                        'mean': 0.0, 'std': 0.0, 'min': 0.0, 'max': 0.0, 'median': 0.0,
                        'nan_count': len(values), 'inf_count': 0,
                        'finite_count': 0, 'total_count': len(values)
                    }
            
            # 完全なメタデータ構築
            full_metadata = {
                'timestamp': time.time(),
                'file_path': str(data_path),
                'feature_count': len(features_dict),
                'data_length': len(next(iter(features_dict.values()))) if features_dict else 0,
                'feature_statistics': feature_stats,
                'calculation_metadata': metadata or {},
                'data_quality': {
                    'total_features': len(features_dict),
                    'total_nan_values': sum(stats['nan_count'] for stats in feature_stats.values()),
                    'total_inf_values': sum(stats['inf_count'] for stats in feature_stats.values()),
                    'total_finite_values': sum(stats['finite_count'] for stats in feature_stats.values())
                }
            }
            
            with open(metadata_path, 'w', encoding='utf-8') as f:
                json.dump(full_metadata, f, indent=2, ensure_ascii=False, cls=NumpyJSONEncoder)
            
            logger.debug(f"メタデータ保存: {metadata_path}")
            
        except Exception as e:
            logger.warning(f"メタデータ保存エラー: {e}")
    
    def create_processing_summary(self, processing_results: Dict[str, Any]) -> str:
        """処理サマリー作成"""
        # 安全なデータアクセス
        files_created = self.output_stats.get('files_created', 0)
        total_features_saved = self.output_stats.get('total_features_saved', 0)
        compression_ratios = self.output_stats.get('compression_ratios', [])
        save_times = self.output_stats.get('save_times', [])
        file_sizes_mb = self.output_stats.get('file_sizes_mb', [])
        total_data_points = self.output_stats.get('total_data_points', 0)

        # 統計計算（ゼロ除算対策）
        avg_compression = np.mean(compression_ratios) if compression_ratios else 0.0
        avg_save_time = np.mean(save_times) if save_times else 0.0
        total_file_size = sum(file_sizes_mb) if file_sizes_mb else 0.0

        summary_lines = [
            "=" * 80,
            "Project Forge 軍資金増大ミッション - 特徴量収集完了",
            "=" * 80,
            f"処理時間: {processing_results.get('total_time_minutes', 0):.2f}分",
            f"生成特徴量数: {processing_results.get('total_features_generated', 0):,}個",
            f"データポイント数: {processing_results.get('data_points_processed', 0):,}行",
            f"成功率: {processing_results.get('success_rate', 0):.2%}",
            "",
            "出力統計:",
            f"  作成ファイル数: {files_created}",
            f"  保存特徴量総数: {total_features_saved:,}",
            f"  総データポイント数: {total_data_points:,}",
            f"  総ファイルサイズ: {total_file_size:.2f}MB",
            f"  平均圧縮率: {avg_compression:.1f}x" if avg_compression > 0 else "  圧縮率: N/A",
            f"  平均保存時間: {avg_save_time:.2f}秒" if avg_save_time > 0 else "  保存時間: N/A",
            "",
            "次のステップ - Project Chimera開発準備完了！",
            "=" * 80
        ]
        
        return "\n".join(summary_lines)
    
    def log_final_summary(self, processing_results: Dict[str, Any]):
        """最終サマリーログ出力"""
        summary = self.create_processing_summary(processing_results)
        logger.info(f"\n{summary}")
    
    def get_output_statistics(self) -> Dict[str, Any]:
        """出力統計の取得"""
        stats = self.output_stats.copy()
        
        # 計算統計を追加
        if stats['compression_ratios']:
            stats['avg_compression_ratio'] = np.mean(stats['compression_ratios'])
            stats['max_compression_ratio'] = np.max(stats['compression_ratios'])
            stats['min_compression_ratio'] = np.min(stats['compression_ratios'])
        
        if stats['save_times']:
            stats['avg_save_time'] = np.mean(stats['save_times'])
            stats['total_save_time'] = np.sum(stats['save_times'])
        
        if stats['file_sizes_mb']:
            stats['total_file_size_mb'] = np.sum(stats['file_sizes_mb'])
            stats['avg_file_size_mb'] = np.mean(stats['file_sizes_mb'])
        
        return stats
    
    def cleanup_temp_files(self, pattern: str = "*temp*"):
        """一時ファイルのクリーンアップ"""
        try:
            temp_files = list(self.output_base_path.glob(pattern))
            cleaned_count = 0
            
            for temp_file in temp_files:
                try:
                    if temp_file.is_file():
                        temp_file.unlink()
                        cleaned_count += 1
                except Exception as e:
                    logger.warning(f"一時ファイル削除エラー: {temp_file}, {e}")
            
            if cleaned_count > 0:
                logger.info(f"一時ファイルクリーンアップ完了: {cleaned_count}個削除")
                
        except Exception as e:
            logger.warning(f"一時ファイルクリーンアップエラー: {e}")


# =============================================================================
# JSONエンコーダー（NumPy対応）
# =============================================================================

class NumpyJSONEncoder(json.JSONEncoder):
    """NumPy配列をJSONシリアライズ可能にするエンコーダー"""
    
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.bool_):
            return bool(obj)
        elif isinstance(obj, (np.datetime64, np.timedelta64)):
            return str(obj)
        elif hasattr(obj, 'item'):  # NumPyスカラー
            return obj.item()
        
        return super().default(obj)

# =============================================================================
# メインFeatureExtractionEngineクラス
# =============================================================================

class FeatureExtractionEngine:
    """
    革新的特徴量収集エンジン - チャンクベース処理アーキテクチャ
    メモリ効率を最大化し、巨大データセットに対応
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        if config is None:
            self.config = DATA_CONFIG.copy()
        else:
            # 直接渡される場合と入れ子の場合の両方に対応
            if 'data_config' in config:
                self.config = config['data_config']
            else:
                self.config = config
        
        # コンポーネント初期化
        self.data_processor = DataProcessor(self.config['base_path'])
        self.window_manager = WindowManager(window_size=100, overlap=0.5)
        self.memory_manager = MemoryManager()
        self.calculator = Calculator(self.window_manager, self.memory_manager)
        self.output_manager = OutputManager(self.config['output_path'])
        
        # チャンクベース処理設定
        self.chunk_size = 1000000  # 1M行/チャンク（調整可能）
        self.temp_file_prefix = "_temp_chunk_"
        
        # 進捗追跡
        self.progress_tracker = None
        
        # 実行統計
        self.execution_stats = {
            'start_time': None,
            'end_time': None,
            'total_features_generated': 0,
            'total_chunks_processed': 0,
            'processing_errors': [],
            'quality_alerts': []
        }
        
        logger.info(f"FeatureExtractionEngine初期化完了 - チャンクサイズ: {self.chunk_size:,}行")
    
    def run_feature_extraction(self, test_mode: bool = False,
                             target_timeframes: List[str] = None) -> Dict[str, Any]:
        """
        チャンクベース特徴量抽出の実行
        """
        logger.info("🚀 チャンクベース特徴量収集開始 - Project Forge軍資金増大ミッション 🚀")
        self.execution_stats['start_time'] = time.time()
        
        # メタデータ収集用のリストを初期化
        all_run_metadata = []

        try:
            if not validate_system_requirements():
                raise RuntimeError("システム要件を満たしていません")
            
            optimize_numpy_settings()
            
            logger.info("📊 データ読み込み・前処理開始")
            self.data_processor.load_metadata()
            self.data_processor.scan_partition_structure()
            
            if target_timeframes is None:
                target_timeframes = ['tick']
            
            memmap_data = self.data_processor.convert_to_memmap(
                timeframes=target_timeframes,
                force_rebuild=False
            )
            
            if not memmap_data:
                raise RuntimeError("memmapデータ変換に失敗しました")

            # --- ▼▼▼ 修正箇所 ▼▼▼ ---
            # ProgressTrackerの初期化のために総チャンク数を事前に計算
            total_chunks_to_process = 0
            for tf in target_timeframes:
                if tf in memmap_data:
                    memmap_array = memmap_data[tf]
                    total_rows = memmap_array.shape[0]
                    if test_mode:
                        total_rows = min(10000, total_rows)
                    total_chunks_to_process += (total_rows + self.chunk_size - 1) // self.chunk_size
            
            self.progress_tracker = ProgressTracker(total_steps=total_chunks_to_process, feature_groups=["ChunkProcessing"])
            # --- ▲▲▲ 修正箇所 ▲▲▲ ---

            for tf, memmap_array in memmap_data.items():
                current_memmap = memmap_array
                if test_mode:
                    test_rows = min(10000, current_memmap.shape[0])  # テストモードは1万行に制限
                    current_memmap = current_memmap[:test_rows]
                    logger.info(f"テストモード: {tf} データを{test_rows}行に制限")
                
                if not self.data_processor.validate_data_integrity(current_memmap):
                    logger.warning(f"データ整合性チェック失敗: {tf}")
                    continue
                
                monitor_thread = self.memory_manager.monitor_continuous(duration_seconds=3600)
                
                try:
                    # --- ▼▼▼ 修正箇所 ▼▼▼ ---
                    # チャンクベース処理の実行（progress_trackerを引数に追加）
                    final_output_path = self._execute_chunk_based_processing(current_memmap, tf, test_mode, self.progress_tracker)
                    # --- ▲▲▲ 修正箇所 ▲▲▲ ---
                    
                    # 実行ごとのメタデータを準備し、リストに追加
                    calc_metadata = {
                        'timeframe': tf,
                        'test_mode': test_mode,
                        'calculation_summary': self.calculator.get_calculation_summary_optimized(),
                        'memory_summary': self.memory_manager.get_memory_summary(),
                        'data_shape': current_memmap.shape,
                        'total_chunks_processed': self.execution_stats['total_chunks_processed'],
                        'output_file': final_output_path
                    }
                    all_run_metadata.append(calc_metadata)
                    
                    logger.info(f"タイムフレーム {tf} 処理完了")
                    
                except Exception as e:
                    error_info = f"タイムフレーム {tf} 処理エラー: {e}"
                    self.execution_stats['processing_errors'].append(error_info)
                    logger.error(error_info)
                    continue

                finally:
                    logger.info(f"タイムフレーム {tf} の後処理 - GC実行")
                    self.memory_manager.force_garbage_collection()
            
            self.execution_stats['end_time'] = time.time()
            total_time = self.execution_stats['end_time'] - self.execution_stats['start_time']

            # --- ▼▼▼ 修正箇所 ▼▼▼ ---
            # progress_trackerから最終サマリーを取得して統合
            if self.progress_tracker:
                performance_metrics = self.progress_tracker.get_performance_metrics()
                self.execution_stats['total_features_generated'] = self.progress_tracker.success_count
            else:
                performance_metrics = {}
            # --- ▲▲▲ 修正箇所 ▲▲▲ ---
            
            # 処理結果の集計
            processing_results = {
                'total_time_minutes': total_time / 60,
                'total_features_generated': self.execution_stats['total_features_generated'],
                'total_chunks_processed': self.execution_stats['total_chunks_processed'],
                'data_points_processed': sum(arr.shape[0] for arr in memmap_data.values()),
                'success_rate': performance_metrics.get('success_rate', 0) / 100,
                'processing_errors': self.execution_stats['processing_errors'],
                'target_timeframes': target_timeframes,
                'test_mode': test_mode,
                'chunk_size': self.chunk_size,
                'performance_metrics': performance_metrics
            }
            
            # タイムフレーム別メタデータの統合
            processing_results['timeframe_details'] = all_run_metadata
            
            # 最終サマリーの保存と表示
            self.output_manager.save_final_summary_metadata(processing_results)
            self.output_manager.log_final_summary(processing_results)
            
            # --- ▼▼▼ 修正箇所 ▼▼▼ ---
            if self.progress_tracker:
                self.progress_tracker.log_final_summary()
            # --- ▲▲▲ 修正箇所 ▲▲▲ ---

            self.memory_manager.log_memory_report()
            
            return processing_results
            
        except Exception as e:
            logger.error(f"特徴量抽出実行エラー: {e}", exc_info=True)
            raise
        
        finally:
            self.data_processor.cleanup_temp_files()
            self._cleanup_temp_chunk_files()
            self.memory_manager.force_garbage_collection()
    
    def _execute_chunk_based_processing(self, memmap_data: np.memmap, 
                                      timeframe: str, test_mode: bool,
                                      progress_tracker: ProgressTracker) -> str:
        """
        チャンクベース特徴量計算の実行
        メモリ効率を最大化した反復処理アーキテクチャ
        """
        logger.info(f"チャンクベース処理開始: {timeframe}, shape={memmap_data.shape}, chunk_size={self.chunk_size:,}")
        
        # データ形状チェック
        if memmap_data.shape[1] < 5:
            raise ValueError(f"データ列数不足: {memmap_data.shape[1]} < 5 (OHLCV必須)")
        
        total_rows = memmap_data.shape[0]
        n_chunks = (total_rows + self.chunk_size - 1) // self.chunk_size
        temp_files = []
        
        logger.info(f"総データ行数: {total_rows:,}, 予定チャンク数: {n_chunks}")
        
        try:
            # チャンクごとの処理ループ
            for chunk_idx in range(n_chunks):
                start_row = chunk_idx * self.chunk_size
                end_row = min(start_row + self.chunk_size, total_rows)
                
                # --- ▼▼▼ 修正箇所 ▼▼▼ ---
                step_name = f"TF:{timeframe} Chunk:{chunk_idx+1}/{n_chunks}"
                features_generated_count = 0
                has_error_in_chunk = False
                warning_message = None
                # --- ▲▲▲ 修正箇所 ▲▲▲ ---

                try:
                    # ステップ1: チャンクデータをメモリに読み込み
                    chunk_data = memmap_data[start_row:end_row].copy()
                    
                    # ステップ2: チャンクに対する特徴量計算
                    chunk_features = self._calculate_features_for_chunk(chunk_data, chunk_idx)
                    features_generated_count = len(chunk_features)
                    
                    # ステップ3: 一時Parquetファイルとして保存
                    temp_file_path = self._save_chunk_to_temp_file(chunk_features, chunk_idx, timeframe, test_mode)
                    temp_files.append(temp_file_path)
                    
                    # ステップ4: メモリ解放
                    del chunk_data, chunk_features
                    self.memory_manager.force_garbage_collection()
                    
                    self.execution_stats['total_chunks_processed'] += 1
                    
                except Exception as e:
                    error_msg = f"チャンク {chunk_idx+1} 処理エラー: {e}"
                    logger.error(error_msg)
                    self.execution_stats['processing_errors'].append(error_msg)
                    # --- ▼▼▼ 修正箇所 ▼▼▼ ---
                    has_error_in_chunk = True
                    # --- ▲▲▲ 修正箇所 ▲▲▲ ---
                    continue
                
                finally:
                    # --- ▼▼▼ 修正箇所 ▼▼▼ ---
                    # 以前の進捗表示ロガーは削除
                    # progress_tracker に進捗を更新
                    progress_tracker.update_progress(
                        step_name=step_name,
                        features_generated=features_generated_count,
                        has_error=has_error_in_chunk,
                        warning_msg=warning_message
                    )
                    # --- ▲▲▲ 修正箇所 ▲▲▲ ---

            # ステップ5: 一時ファイルを最終的な1つのParquetファイルに結合
            final_output_path = self._combine_temp_files_to_final(temp_files, timeframe, test_mode)
            logger.info(f"最終ファイル結合完了: {final_output_path}")
            
            return final_output_path
            
        except Exception as e:
            logger.error(f"チャンクベース処理エラー: {e}")
            raise
        
        finally:
            # 一時ファイルのクリーンアップ
            self._cleanup_specific_temp_files(temp_files)
    
    def _calculate_features_for_chunk(self, chunk_data: np.ndarray, chunk_idx: int) -> Dict[str, np.ndarray]:
        """
        単一チャンクに対する全特徴量計算 (新しいCalculatorロジックを直接呼び出すように修正)
        """
        logger.debug(f"チャンク {chunk_idx+1} 特徴量計算開始: shape={chunk_data.shape}")
        
        try:
            # 価格データ抽出（OHLCV構造を想定）
            open_prices = chunk_data[:, 1]
            high_prices = chunk_data[:, 2]
            low_prices = chunk_data[:, 3]
            close_prices = chunk_data[:, 4]
            volume_data = chunk_data[:, 5]
            
            # 新しいCalculatorのメインメソッドを直接呼び出す
            all_features = self.calculator.calculate_all_features_polars_optimized(
                high=high_prices,
                low=low_prices,
                close=close_prices,
                volume=volume_data,
                open_prices=open_prices
            )
            logger.debug(f"チャンク {chunk_idx+1}: 特徴量計算完了 ({len(all_features)}特徴量)")

            # スクリプト名から番号を自動取得
            script_name = Path(__file__).name
            match = re.search(r"engine_(\d+)", script_name)
            script_number = match.group(1) if match else "unknown"

            # 特徴量名にプレフィックスを追加
            prefixed_features = {
                f"e{script_number}_{name}": values 
                for name, values in all_features.items()
            }
            
            # 品質レポート生成 (プレフィックス付きの特徴量を使用)
            quality_report = self.calculator.generate_quality_report(prefixed_features)
            if quality_report.get('warnings'):
                self.execution_stats['quality_alerts'].extend(quality_report['warnings'])
                logger.debug(f"チャンク {chunk_idx+1}: 品質アラート {len(quality_report['warnings'])}件")
            
            self.execution_stats['total_features_generated'] += len(prefixed_features)
            
            return prefixed_features
            
        except Exception as e:
            logger.error(f"チャンク {chunk_idx+1} 特徴量計算エラー: {e}")
            raise
    
    def _save_chunk_to_temp_file(self, chunk_features: Dict[str, np.ndarray], 
                               chunk_idx: int, timeframe: str, test_mode: bool) -> str:
        """
        チャンク特徴量を一時Parquetファイルとして保存
        """
        try:
            # 一時ファイル名生成
            temp_filename = f"{self.temp_file_prefix}{chunk_idx:04d}_{timeframe}{'_test' if test_mode else ''}.parquet"
            temp_file_path = self.config['output_path'] / temp_filename
            
            # Polars DataFrameに変換して保存
            self.output_manager.save_features(chunk_features, temp_filename)
            
            logger.debug(f"一時ファイル保存完了: {temp_filename}")
            return str(temp_file_path)
            
        except Exception as e:
            logger.error(f"一時ファイル保存エラー: チャンク {chunk_idx+1}, {e}")
            raise
    
    def _combine_temp_files_to_final(self, temp_files: List[str], 
                                   timeframe: str, test_mode: bool) -> str:
        """
        複数の一時Parquetファイルを1つの最終ファイルに結合
        """
        logger.info(f"一時ファイル結合開始: {len(temp_files)}ファイル")
        
        try:
            # 最終出力ファイル名
            final_filename = f"features_combined_{timeframe}{'_test' if test_mode else ''}_{int(time.time())}.parquet"
            final_path = self.config['output_path'] / final_filename
            
            # Polarsで効率的に結合
            valid_temp_files = [f for f in temp_files if Path(f).exists()]
            
            if not valid_temp_files:
                raise ValueError("結合対象の一時ファイルが存在しません")
            
            logger.info(f"有効な一時ファイル: {len(valid_temp_files)}個")
            
            # LazyFrameで効率的に結合
            lazy_frames = [pl.scan_parquet(f) for f in valid_temp_files]
            combined_lf = pl.concat(lazy_frames)
            
            # 最終ファイルとして保存
            combined_lf.sink_parquet(str(final_path))
            
            logger.info(f"最終ファイル結合完了: {final_filename}")
            return str(final_path)
            
        except Exception as e:
            logger.error(f"ファイル結合エラー: {e}")
            raise
    
    def _cleanup_specific_temp_files(self, temp_files: List[str]):
        """
        指定された一時ファイルをクリーンアップ
        """
        cleaned_count = 0
        for temp_file in temp_files:
            try:
                temp_path = Path(temp_file)
                if temp_path.exists():
                    temp_path.unlink()
                    cleaned_count += 1
            except Exception as e:
                logger.warning(f"一時ファイル削除エラー: {temp_file}, {e}")
        
        logger.debug(f"一時ファイルクリーンアップ完了: {cleaned_count}個削除")
    
    def _cleanup_temp_chunk_files(self):
        """
        全ての一時チャンクファイルをクリーンアップ
        """
        try:
            output_dir = Path(self.config['output_path'])
            temp_pattern = f"{self.temp_file_prefix}*.parquet"
            temp_files = list(output_dir.glob(temp_pattern))
            
            for temp_file in temp_files:
                try:
                    temp_file.unlink()
                    logger.debug(f"一時チャンクファイル削除: {temp_file.name}")
                except Exception as e:
                    logger.warning(f"一時ファイル削除エラー: {temp_file}, {e}")
            
            if temp_files:
                logger.info(f"一時チャンクファイルクリーンアップ完了: {len(temp_files)}個削除")
                
        except Exception as e:
            logger.warning(f"一時チャンクファイルクリーンアップエラー: {e}")
    

# ブロック7: 設定・テスト・統合部 - Project Forge軍資金増大ミッション完成

# =============================================================================
# 設定管理システム - Golden Ratio基準パラメータ
# =============================================================================

class ConfigurationManager:
    """
    設定管理クラス - プロンプト仕様に完全準拠した設定
    """
    
    def __init__(self):
        self.base_config = self._load_base_configuration()
        self.validation_rules = self._define_validation_rules()
        self.environment_info = self._detect_environment()
        
        logger.info("設定管理システム初期化完了")
    
    def _load_base_configuration(self) -> Dict[str, Any]:
        """基本設定読み込み - プロンプト仕様"""
        return {
            # データ仕様（プロンプト指定）
            'data_config': {
                'base_path': Path('/workspaces/project_forge/data/1_XAUUSD_base_data/XAUUSD_tick_master_GOLDEN'),
                'output_path': Path('/workspaces/project_forge/data/2_feature_value/'),
                'target_market': 'XAU/USD',
                'hive_partitioning': True,
                'required_columns': ['timestamp', 'open', 'high', 'low', 'close', 'volume', 'timeframe'],
                'additional_columns': ['log_return', 'rolling_volatility', 'rolling_avg_volume', 'atr', 
                                     'price_direction', 'price_momentum', 'volume_ratio'],
                'timeframes': ['tick', 'M0.5', 'M1', 'M3', 'M5', 'M8', 'M15', 'M30', 'H1', 'H4', 'H6', 'H12', 'D1', 'W1', 'MN']
            },
            
            # ハードウェア制約（プロンプト指定）
            'hardware_constraints': {
                'gpu_model': 'RTX 3060',
                'gpu_memory_gb': 12,
                'cpu_model': 'i7-8700K',
                'cpu_cores': 6,
                'ram_gb': 64,
                'storage_type': 'NVMe SSD'
            },
            
            # アーキテクチャ仕様
            'architecture': {
                'resource_allocation': {
                    'DataProcessor': 0.10,
                    'WindowManager': 0.05,
                    'Calculator': 0.80,  # 核心部分に80%集中
                    'MemoryManager': 0.02,
                    'OutputManager': 0.03
                }
            },
            
            # 技術戦略設定
            'processing_strategy': {
                'data_loading': 'numpy_memmap',  # プロンプト推奨
                'computation': 'cpu_optimized',  # 確実な動作保証最優先
                'parallel_processing': True,
                'gpu_processing': False,  # 将来的オプション
                'out_of_core': False,  # memmapで十分
                'jit_compilation': True,  # Calculator内で積極活用
                'output_format': 'polars_lazy_parquet'  # プロンプト推奨
            }
        }
    
    def _define_validation_rules(self) -> Dict[str, Any]:
        """検証ルール定義"""
        return {
            'memory_limits': {
                'ram_usage_warning': 0.8,  # 80%で警告
                'ram_usage_critical': 0.9,  # 90%で危険
                'gpu_usage_warning': 0.7   # 70%で警告
            },
            'data_quality': {
                'max_nan_ratio': 0.3,      # 30%以上のNaNで警告
                'min_valid_ratio': 0.7,    # 70%以上の有効特徴量必須
                'outlier_threshold': 5.0    # 5σ外れ値判定
            },
            'performance_thresholds': {
                'max_processing_time_hours': 2.0,   # 2時間以内完了目標
                'min_features_per_second': 10,      # 最低処理速度
                'target_compression_ratio': 3.0     # 目標圧縮率
            }
        }
    
    def _detect_environment(self) -> Dict[str, Any]:
        """実行環境検出"""
        import psutil
        import platform
        
        try:
            # CPU情報
            cpu_info = {
                'physical_cores': psutil.cpu_count(logical=False),
                'logical_cores': psutil.cpu_count(logical=True),
                'cpu_freq': psutil.cpu_freq().current if psutil.cpu_freq() else 0,
                'cpu_usage': psutil.cpu_percent(interval=1)
            }
            
            # メモリ情報
            memory = psutil.virtual_memory()
            memory_info = {
                'total_gb': memory.total / (1024**3),
                'available_gb': memory.available / (1024**3),
                'used_percent': memory.percent
            }
            
            # GPU情報
            gpu_info = self._detect_gpu_environment()
            
            # ディスク情報
            disk = psutil.disk_usage('/')
            disk_info = {
                'total_gb': disk.total / (1024**3),
                'free_gb': disk.free / (1024**3),
                'used_percent': (disk.used / disk.total) * 100
            }
            
            return {
                'platform': platform.platform(),
                'python_version': platform.python_version(),
                'cpu': cpu_info,
                'memory': memory_info,
                'gpu': gpu_info,
                'disk': disk_info,
                'environment_validated': self._validate_environment(cpu_info, memory_info, gpu_info)
            }
            
        except Exception as e:
            logger.warning(f"環境検出エラー: {e}")
            return {'environment_validated': False, 'error': str(e)}
    
    def _detect_gpu_environment(self) -> Dict[str, Any]:
        """GPU環境検出"""
        gpu_info = {'available': False, 'type': 'None', 'memory_gb': 0}
        
        # CUDA/PyTorch検出
        try:
            import torch
            if torch.cuda.is_available():
                gpu_info.update({
                    'available': True,
                    'type': 'CUDA',
                    'device_name': torch.cuda.get_device_name(0),
                    'memory_gb': torch.cuda.get_device_properties(0).total_memory / (1024**3),
                    'cuda_version': torch.version.cuda
                })
        except ImportError:
            pass
        
        # CuPy検出
        if not gpu_info['available']:
            try:
                import cupy
                gpu_info.update({
                    'available': True,
                    'type': 'CuPy',
                    'memory_gb': cupy.cuda.Device().mem_info[1] / (1024**3)
                })
            except ImportError:
                pass
        
        return gpu_info
    
    def _validate_environment(self, cpu_info: Dict, memory_info: Dict, gpu_info: Dict) -> bool:
        """環境要件検証"""
        validation_results = []
        
        # CPU要件チェック
        if cpu_info['physical_cores'] >= 4:
            validation_results.append(True)
        else:
            logger.warning(f"CPU要件未満: {cpu_info['physical_cores']}コア < 4コア推奨")
            validation_results.append(False)
        
        # メモリ要件チェック
        if memory_info['total_gb'] >= 32:
            validation_results.append(True)
        else:
            logger.warning(f"メモリ要件未満: {memory_info['total_gb']:.1f}GB < 32GB推奨")
            validation_results.append(False)
        
        # 使用可能メモリチェック
        if memory_info['available_gb'] >= 16:
            validation_results.append(True)
        else:
            logger.warning(f"使用可能メモリ不足: {memory_info['available_gb']:.1f}GB < 16GB必要")
            validation_results.append(False)
        
        return all(validation_results)
    
    def get_optimized_settings(self) -> Dict[str, Any]:
        """環境に最適化された設定を返す"""
        env = self.environment_info
        
        # CPU並列度設定
        cpu_cores = min(env['cpu']['physical_cores'], 6)  # i7-8700K基準
        
        # メモリ使用設定
        available_memory_gb = env['memory']['available_gb']
        safe_memory_gb = available_memory_gb * 0.8  # 80%まで使用
        
        # ウィンドウサイズ最適化
        if safe_memory_gb > 32:
            window_size = 200
            batch_size = 20
        elif safe_memory_gb > 16:
            window_size = 100
            batch_size = 10
        else:
            window_size = 50
            batch_size = 5
        
        return {
            'cpu_cores': cpu_cores,
            'max_memory_gb': safe_memory_gb,
            'window_size': window_size,
            'batch_size': batch_size,
            'memmap_cache_size': min(8, int(safe_memory_gb // 4)),
            'parallel_workers': cpu_cores,
            'chunk_size': max(10000, int(1000000 // cpu_cores))
        }

# =============================================================================
# テスト・検証システム
# =============================================================================

class SystemValidator:
    """
    システム検証クラス - テストモード機能とバリデーション（修正版）
    """
    
    def __init__(self, config_manager: ConfigurationManager):
        self.config = config_manager
        self.test_results = {}
        self.validation_results = {}
        
    def run_system_validation(self, test_mode: bool = True) -> Dict[str, Any]:
        """システム全体の検証実行"""
        logger.info("🧪 システム検証開始")
        
        validation_report = {
            'timestamp': time.time(),
            'test_mode': test_mode,
            'environment_check': None,
            'data_access_check': None,
            'computation_check': None,
            'memory_check': None,
            'integration_check': None,
            'overall_status': 'unknown'
        }
        
        try:
            # 1. 環境チェック
            validation_report['environment_check'] = self._validate_environment()
            
            # 2. データアクセスチェック
            validation_report['data_access_check'] = self._validate_data_access(test_mode)
            
            # 3. 計算エンジンチェック
            validation_report['computation_check'] = self._validate_computation(test_mode)
            
            # 4. メモリ管理チェック
            validation_report['memory_check'] = self._validate_memory_management()
            
            # 5. 統合チェック
            validation_report['integration_check'] = self._validate_integration(test_mode)
            
            # 総合判定
            validation_report['overall_status'] = self._determine_overall_status(validation_report)
            
            # 結果表示
            self._display_validation_results(validation_report)
            
            return validation_report
            
        except Exception as e:
            logger.error(f"システム検証エラー: {e}")
            validation_report['overall_status'] = 'failed'
            validation_report['error'] = str(e)
            return validation_report
    
    def _validate_environment(self) -> Dict[str, Any]:
        """環境検証"""
        logger.info("環境要件チェック実行中...")
        
        try:
            env_info = self.config.environment_info
            
            checks = {
                'python_version': sys.version_info >= (3, 8),
                'memory_sufficient': env_info['memory']['total_gb'] >= 16,
                'disk_space': env_info['disk']['free_gb'] >= 10,
                'cpu_cores': env_info['cpu']['physical_cores'] >= 4,
                'libraries_available': self._check_required_libraries()
            }
            
            return {
                'status': 'success' if all(checks.values()) else 'warning',
                'checks': checks,
                'environment_info': env_info
            }
        except Exception as e:
            logger.error(f"環境検証エラー: {e}")
            return {
                'status': 'failed',
                'error': str(e),
                'message': '環境検証失敗'
            }
    
    def _check_required_libraries(self) -> bool:
        """必要ライブラリの存在確認"""
        required_libs = [
            'numpy', 'polars', 'numba', 'scipy', 
            'sklearn', 'pyarrow', 'pywt', 'psutil'
        ]
        
        missing_libs = []
        for lib in required_libs:
            try:
                __import__(lib)
            except ImportError:
                missing_libs.append(lib)
        
        if missing_libs:
            logger.warning(f"不足ライブラリ: {missing_libs}")
            return False
        
        return True
    
    def _validate_data_access(self, test_mode: bool) -> Dict[str, Any]:
        """データアクセス検証"""
        logger.info("データアクセステスト実行中...")
        
        try:
            # データプロセッサ初期化テスト
            base_path = self.config.base_config['data_config']['base_path']
            data_processor = DataProcessor(base_path)
            
            # メタデータ読み込みテスト
            if data_processor.metadata_path.exists():
                metadata = data_processor.load_metadata()
                partition_info = data_processor.scan_partition_structure()
                
                # テスト用memmap作成
                if test_mode:
                    test_timeframes = ['tick']
                    memmap_data = data_processor.convert_to_memmap(test_timeframes, force_rebuild=False)
                    
                    # データ整合性チェック
                    if memmap_data and 'tick' in memmap_data:
                        integrity_ok = data_processor.validate_data_integrity(memmap_data['tick'])
                        
                        return {
                            'status': 'success',
                            'metadata_loaded': True,
                            'partitions_found': len(partition_info),
                            'memmap_created': True,
                            'data_integrity': integrity_ok,
                            'test_data_shape': memmap_data['tick'].shape
                        }
            
            return {
                'status': 'warning',
                'metadata_loaded': False,
                'message': 'データパスが見つかりません（テストモードでは正常）'
            }
            
        except Exception as e:
            logger.error(f"データアクセステストエラー: {e}")
            return {
                'status': 'failed',
                'error': str(e),
                'message': 'データアクセステスト失敗'
            }
    
    def _validate_computation(self, test_mode: bool) -> Dict[str, Any]:
        """計算エンジン検証 (Script B: 市場指標・テクニカル分析)"""
        logger.info("計算エンジンテスト実行中...")
        
        try:
            # ダミーデータ生成
            np.random.seed(42)
            test_size = 250
            test_data = np.random.randn(test_size) * 0.01 + 100
            dummy_high = test_data + np.random.rand(test_size) * 0.5
            dummy_low = test_data - np.random.rand(test_size) * 0.5
            dummy_volume = np.random.poisson(1000, test_size).astype(float)

            # Calculator初期化
            dummy_window_manager = WindowManager(window_size=50, overlap=0.3)
            dummy_memory_manager = MemoryManager()
            calculator = Calculator(dummy_window_manager, dummy_memory_manager)
            
            # Bのスクリプトに存在する計算機能をテスト
            adx_features = calculator.calculate_adx_features(dummy_high, dummy_low, test_data)
            ma_features = calculator.calculate_moving_averages(test_data)
            
            # 計算結果検証
            all_features = {}
            all_features.update(adx_features)
            all_features.update(ma_features)
            
            # 品質レポート生成
            quality_report = calculator.generate_quality_report(all_features)
            
            avg_quality = quality_report.get('overall_quality_score', 0.0)
            total_nan = quality_report.get('data_quality', {}).get('total_nan_values', len(test_data))
            high_quality_count = quality_report.get('high_quality_features', 0)
            total_feature_count = quality_report.get('total_features', 1)
            success_rate = high_quality_count / max(1, total_feature_count)
            
            return {
                'status': 'success' if success_rate >= 0.6 and total_nan < len(test_data) * 0.1 else 'warning',
                'total_features': len(all_features),
                'average_quality': avg_quality,
                'success_rate': success_rate,
                'total_nan_values': total_nan,
                'calculation_summary': calculator.get_calculation_summary_optimized()
            }
            
        except Exception as e:
            logger.error(f"計算エンジンテストエラー: {e}")
            return {
                'status': 'failed',
                'error': str(e),
                'message': '計算エンジンテスト失敗'
            }
    
    def _validate_memory_management(self) -> Dict[str, Any]:
        """メモリ管理検証"""
        logger.info("メモリ管理テスト実行中...")
        
        try:
            memory_manager = MemoryManager()
            
            # 初期状態チェック
            initial_status = memory_manager.check_memory_status()
            
            # 大きなデータでメモリテスト
            test_size = 100000
            test_array = np.random.randn(test_size, 10)
            
            # メモリ使用量変化チェック
            memory_status = memory_manager.check_memory_status()
            
            # ガベージコレクションテスト
            del test_array
            freed_memory = memory_manager.force_garbage_collection()
            
            final_status = memory_manager.check_memory_status()
            
            # メモリリーク検出
            memory_delta = final_status['current_gb'] - initial_status['current_gb']
            
            return {
                'status': 'success' if memory_delta < 0.1 else 'warning',  # 100MB以下の差は正常
                'initial_usage': initial_status['current_gb'],
                'peak_usage': memory_status['current_gb'],
                'final_usage': final_status['current_gb'],
                'memory_delta': memory_delta,
                'memory_freed': freed_memory,
                'gpu_available': memory_manager.gpu_available,
                'memory_leak_detected': memory_delta > 0.5  # 500MB以上の増加はリーク疑い
            }
            
        except Exception as e:
            logger.error(f"メモリ管理テストエラー: {e}")
            return {
                'status': 'failed',
                'error': str(e),
                'message': 'メモリ管理テスト失敗'
            }
    
    def _validate_integration(self, test_mode: bool) -> Dict[str, Any]:
        """統合テスト（修正版 - 引数問題解決）"""
        logger.info("統合テスト実行中...")
        
        # テスト専用のパスを指定して初期化
        test_output_path = Path('./validation_test_output')
        test_output_path.mkdir(parents=True, exist_ok=True)
        
        try:
            # OutputManager初期化
            output_manager = OutputManager(test_output_path)
            
            # テスト用ダミーデータ作成
            np.random.seed(123)
            dummy_features = {
                'test_feature_1': np.random.randn(100),
                'test_feature_2': np.random.randn(100),
                'test_feature_3': np.random.uniform(0, 1, 100),
                'test_feature_4': np.random.randint(0, 10, 100).astype(float)
            }
            
            # ファイル保存テスト（修正: 正しい引数で呼び出し）
            test_filename = 'integration_test_features'
            try:
                output_path = output_manager.save_features(dummy_features, test_filename)
                file_created = output_path.exists()
                
                if file_created:
                    # ファイルサイズチェック
                    file_size_kb = output_path.stat().st_size / 1024
                    
                    # ファイル読み込みテスト
                    try:
                        import polars as pl
                        test_df = pl.read_parquet(str(output_path))
                        read_success = len(test_df) > 0
                        columns_match = len(test_df.columns) == len(dummy_features)
                    except Exception as read_error:
                        read_success = False
                        columns_match = False
                        logger.warning(f"ファイル読み込みテスト失敗: {read_error}")
                else:
                    file_size_kb = 0
                    read_success = False
                    columns_match = False
                    
            except Exception as save_error:
                logger.error(f"ファイル保存テスト失敗: {save_error}")
                file_created = False
                file_size_kb = 0
                read_success = False
                columns_match = False
                output_path = None
            
            # メタデータ保存テスト
            test_metadata = {
                'test_run': True,
                'features_count': len(dummy_features),
                'timestamp': time.time()
            }
            
            try:
                metadata_saved = output_manager.save_final_summary_metadata(test_metadata)
            except Exception as meta_error:
                logger.warning(f"メタデータ保存テスト失敗: {meta_error}")
                metadata_saved = False
            
            # クリーンアップ
            cleanup_success = True
            try:
                if output_path and output_path.exists():
                    output_path.unlink()
                    
                # 他のテストファイルもクリーンアップ
                test_files = list(test_output_path.glob("*.parquet"))
                test_files.extend(test_output_path.glob("*.json"))
                
                for test_file in test_files:
                    try:
                        test_file.unlink()
                    except:
                        pass
                
                # ディレクトリが空の場合のみ削除
                try:
                    if not any(test_output_path.iterdir()):
                        test_output_path.rmdir()
                except OSError:
                    pass
                    
            except Exception as cleanup_error:
                logger.warning(f"クリーンアップエラー: {cleanup_error}")
                cleanup_success = False
            
            # 統合テスト結果判定
            integration_success = (file_created and read_success and 
                                 columns_match and metadata_saved)
            
            return {
                'status': 'success' if integration_success else 'failed',
                'file_created': file_created,
                'file_size_kb': file_size_kb,
                'file_readable': read_success,
                'columns_match': columns_match,
                'metadata_saved': metadata_saved,
                'cleanup_successful': cleanup_success,
                'output_path': str(output_path) if output_path else None,
                'features_tested': len(dummy_features)
            }
            
        except Exception as e:
            logger.error(f"統合テストで予期しないエラー: {e}", exc_info=True)
            return {
                'status': 'failed',
                'error': str(e),
                'message': '統合テスト失敗'
            }
    
    def _determine_overall_status(self, validation_report: Dict[str, Any]) -> str:
        """総合ステータス判定"""
        statuses = []
        critical_failures = []
        
        for check_name, check_result in validation_report.items():
            if isinstance(check_result, dict) and 'status' in check_result:
                status = check_result['status']
                statuses.append(status)
                
                # 重要なチェックの失敗を記録
                if status == 'failed' and check_name in ['environment_check', 'computation_check']:
                    critical_failures.append(check_name)
        
        if not statuses:
            return 'unknown'
        
        # 重要なチェックが失敗している場合
        if critical_failures:
            return 'failed'
        
        # 全て成功
        if all(status == 'success' for status in statuses):
            return 'ready'
        
        # 一つでも失敗がある場合
        elif any(status == 'failed' for status in statuses):
            return 'failed'
        
        # 警告のみの場合
        else:
            return 'warning'
    
    def _display_validation_results(self, validation_report: Dict[str, Any]):
        """検証結果表示"""
        status = validation_report['overall_status']
        
        if status == 'ready':
            logger.info("✅ システム検証完了 - 本番処理準備完了")
        elif status == 'warning':
            logger.warning("⚠️ システム検証完了 - 一部警告あり（実行可能）")
        elif status == 'failed':
            logger.error("❌ システム検証失敗 - 問題要解決")
        else:
            logger.warning("❓ システム検証状態不明")
        
        # 詳細結果表示
        for check_name, result in validation_report.items():
            if isinstance(result, dict) and 'status' in result:
                status_icon = {
                    'success': '✅', 
                    'warning': '⚠️', 
                    'failed': '❌'
                }.get(result['status'], '❓')
                
                logger.info(f"{status_icon} {check_name}: {result['status']}")
                
                # エラーがある場合は詳細を表示
                if 'error' in result:
                    logger.error(f"   エラー詳細: {result['error']}")
                
                # 重要な指標を表示
                if check_name == 'computation_check' and 'total_features' in result:
                    logger.info(f"   特徴量数: {result['total_features']}, "
                              f"品質スコア: {result['average_quality']:.3f}")
                
                elif check_name == 'memory_check' and 'memory_delta' in result:
                    logger.info(f"   メモリ使用量変化: {result['memory_delta']:.3f}GB")
                
                elif check_name == 'integration_check' and 'features_tested' in result:
                    logger.info(f"   テスト特徴量数: {result['features_tested']}")
    
    def generate_validation_report(self, validation_results: Dict[str, Any]) -> str:
        """検証結果レポート生成"""
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(validation_results['timestamp']))
        
        report = f"""
# システム検証レポート
生成日時: {timestamp}
テストモード: {validation_results['test_mode']}
総合ステータス: {validation_results['overall_status']}

## 検証結果詳細
"""
        
        for check_name, result in validation_results.items():
            if isinstance(result, dict) and 'status' in result:
                report += f"\n### {check_name}\n"
                report += f"ステータス: {result['status']}\n"
                
                if 'error' in result:
                    report += f"エラー: {result['error']}\n"
                
                # 各チェック固有の情報を追加
                if check_name == 'environment_check' and 'checks' in result:
                    report += "環境要件:\n"
                    for req, passed in result['checks'].items():
                        status = "✅" if passed else "❌"
                        report += f"  {status} {req}\n"
                
                elif check_name == 'computation_check':
                    if 'total_features' in result:
                        report += f"生成特徴量数: {result['total_features']}\n"
                    if 'average_quality' in result:
                        report += f"平均品質スコア: {result['average_quality']:.3f}\n"
        
        return report

# =============================================================================
# メインエントリーポイントとインタラクティブモード
# =============================================================================

def interactive_mode():
    """プロンプト仕様のインタラクティブモード"""
    print("=" * 70)
    print("🚀 Project Forge 軍資金増大ミッション - 革新的特徴量収集 🚀")
    print("=" * 70)
    
    # 設定管理初期化
    config_manager = ConfigurationManager()
    validator = SystemValidator(config_manager)
    
    # システム情報表示
    display_system_info()
    
    # 環境最適化設定取得
    optimized_settings = config_manager.get_optimized_settings()
    print(f"\n📊 最適化設定:")
    print(f"  CPU並列度: {optimized_settings['cpu_cores']}コア")
    print(f"  使用可能メモリ: {optimized_settings['max_memory_gb']:.1f}GB")
    print(f"  推奨ウィンドウサイズ: {optimized_settings['window_size']}")
    
    # システム検証実行
    print("\n🔍 システム検証実行中...")
    validation_results = validator.run_system_validation(test_mode=True)
    
    if validation_results['overall_status'] not in ['ready', 'warning']:
        print("❌ システム検証失敗 - 実行を中止します")
        return None
    
    # 時間足選択
    print("\n📊 処理対象データソースの選択:")
    print("1. tickデータのみ（推奨：高速処理）")
    print("2. 複数タイムフレーム（tick + M1 + M5 + H1）")
    print("3. 全タイムフレーム（全15種類）")
    print("4. カスタム選択（1,2,3形式またはtick,M1,M5形式）")
    
    while True:
        try:
            choice = input("選択してください (1-4): ").strip()
            if choice == '1':
                target_timeframes = ['tick']
                break
            elif choice == '2':
                target_timeframes = ['tick', 'M1', 'M5', 'H1']
                break
            elif choice == '3':
                target_timeframes = config_manager.base_config['data_config']['timeframes']
                break
            elif choice == '4':
                custom_input = input("タイムフレーム指定 (例: tick,M1,M5 または 1,2,3): ").strip()
                target_timeframes = parse_timeframe_selection(custom_input, config_manager.base_config['data_config']['timeframes'])
                if target_timeframes:
                    break
                else:
                    print("❌ 無効な選択です")
            else:
                print("❌ 1-4の範囲で選択してください")
        except (ValueError, KeyboardInterrupt):
            print("❌ 無効な入力です")
    
    # テストモード選択
    print("\n🧪 実行モード選択:")
    print("1. 本番モード（全データ処理）")
    print("2. テストモード（1000行制限・動作確認用）")
    
    while True:
        try:
            test_choice = int(input("選択してください (1-2): "))
            if test_choice in [1, 2]:
                test_mode = test_choice == 2
                break
            else:
                print("❌ 1-2の範囲で選択してください")
        except ValueError:
            print("❌ 数字を入力してください")
    
    # 確認画面
    print("\n" + "=" * 50)
    print("📋 実行設定確認")
    print("=" * 50)
    print(f"対象タイムフレーム: {target_timeframes}")
    print(f"実行モード: {'テストモード（1000行制限）' if test_mode else '本番モード（全データ処理）'}")
    
    if not test_mode and len(target_timeframes) > 3:
        estimated_time = len(target_timeframes) * 15
        print(f"予想処理時間: {estimated_time}-{estimated_time*2}分")
    else:
        print(f"予想処理時間: {'1-2分' if test_mode else '5-15分'}")
    
    print("=" * 50)
    
    confirm = input("\n実行しますか？ (y/N): ").strip().lower()
    if confirm != 'y':
        print("❌ 実行をキャンセルしました")
        return None
    
    # メインエンジン実行
    try:
        # 設定を正しい形式で渡す
        engine_config = {
            'base_path': config_manager.base_config['data_config']['base_path'],
            'output_path': config_manager.base_config['data_config']['output_path'],
            'timeframes': config_manager.base_config['data_config']['timeframes'],
            'required_columns': config_manager.base_config['data_config']['required_columns'],
            'additional_columns': config_manager.base_config['data_config']['additional_columns']
        }
        engine = FeatureExtractionEngine(engine_config)
        
        print("\n🚀 革新的特徴量収集エンジン開始 - Project Forge軍資金増大ミッション 🚀")
        
        results = engine.run_feature_extraction(
            test_mode=test_mode,
            target_timeframes=target_timeframes
        )
        
        print("\n🎉 Project Forge 軍資金増大ミッション完了！")
        print("Next: Project Chimera開発開始 🚀")
        
        return results
        
    except Exception as e:
        logger.error(f"実行エラー: {e}")
        print(f"\n❌ エラーが発生しました: {e}")
        print("ログファイルで詳細を確認してください")
        return None

def parse_timeframe_selection(input_str: str, available_timeframes: List[str]) -> List[str]:
    """タイムフレーム選択の解析"""
    if not input_str:
        return []
    
    selections = []
    parts = [part.strip() for part in input_str.split(',')]
    
    for part in parts:
        # 数値インデックス形式
        if part.isdigit():
            idx = int(part) - 1  # 1-basedを0-basedに変換
            if 0 <= idx < len(available_timeframes):
                selections.append(available_timeframes[idx])
        # 直接指定形式
        elif part in available_timeframes:
            selections.append(part)
    
    return list(dict.fromkeys(selections))  # 重複除去

# =============================================================================
# メイン実行部
# =============================================================================

if __name__ == "__main__":
    try:
        # システム要件チェック
        if not validate_system_requirements():
            logger.error("システム要件を満たしていません")
            sys.exit(1)
        
        # NumPy最適化
        optimize_numpy_settings()
        
        # インタラクティブモード実行
        results = interactive_mode()
        
        if results:
            logger.info("🎯 Project Forge ミッション成功 - Project Chimera へ続く")
        else:
            logger.warning("実行が完了しませんでした")
            
    except KeyboardInterrupt:
        logger.info("ユーザーによって中断されました")
    except Exception as e:
        logger.error(f"予期しないエラー: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

# =============================================================================
# Project Forge 軍資金増大ミッション - 完成
# Next Phase: Project Chimera Development
# =============================================================================

"""
🚀 Project Forge 軍資金増大ミッション - 革新的特徴量収集システム

このスクリプトは以下の仕様に完全準拠:
- プロンプト指定の5クラス構成（Calculator 80%リソース集中）
- NumPy memmap統一使用（DataFrame全読み込み厳禁）
- CPU最適化による確実な動作保証最優先
- Polars LazyParquet（streaming=True）による出力
- 64GB RAM + RTX 3060 12GB環境に最適化
- Golden Ratio等理論的根拠に基づくパラメータ選択
- テストモード：システム動作確認（60%以上特徴量有効化目標）
- 品質保証：リアルタイム統計的検証、NaN率監視

Next: Project Chimera - この軍資金で究極のシステム開発へ 🎯
"""                                        