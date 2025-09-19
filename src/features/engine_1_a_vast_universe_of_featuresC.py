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
    # 学際的アナロジー特徴量（統合・最適化版）
    # =========================================================================
    
    def calculate_interdisciplinary_features(self, data: np.ndarray) -> Dict[str, np.ndarray]:
        """学際的アナロジー特徴量の統合計算（Polars+Numba最適化版）"""
        features = {}
        
        try:   
            # ネットワーク科学特徴量
            network_features = self.calculate_network_science_features(data)
            features.update(network_features)
            
            # 音響学特徴量
            acoustics_features = self.calculate_acoustics_features(data)
            features.update(acoustics_features)
            
            # 言語学特徴量
            linguistics_features = self.calculate_linguistics_features(data)
            features.update(linguistics_features)
            
            # 美学特徴量
            aesthetics_features = self.calculate_aesthetics_features(data)
            features.update(aesthetics_features)
            
            # 音楽理論特徴量
            music_features = self.calculate_music_theory_features(data)
            features.update(music_features)
            
            # 生体力学特徴量
            biomechanics_features = self.calculate_biomechanics_features(data)
            features.update(biomechanics_features)
            
        except Exception as e:
            logger.error(f"学際的アナロジー特徴量計算エラー: {e}")
        
        return features
    
    def calculate_network_science_features(self, data: np.ndarray) -> Dict[str, np.ndarray]:
        """ネットワーク科学特徴量（Polars最適化版）"""
        features = {}
        window_size = 50
        
        # Numba最適化版ネットワーク科学計算
        network_result = self._numba_safe_calculation(
            self._calculate_network_science_vectorized_optimized, data, window_size
        )
        
        if isinstance(network_result, np.ndarray) and len(network_result.shape) == 2 and network_result.shape[1] >= 5:
            network_names = ['network_density', 'centrality_measure', 'clustering_coefficient',
                           'betweenness_centrality', 'eigenvector_centrality']
            
            for j, name in enumerate(network_names):
                features[f'{name}_{window_size}'] = network_result[:, j]
        else:
            # フォールバック値
            network_names = ['network_density', 'centrality_measure', 'clustering_coefficient',
                           'betweenness_centrality', 'eigenvector_centrality']
            for name in network_names:
                features[f'{name}_{window_size}'] = np.zeros(len(data))
        
        return features
    
    @staticmethod
    @njit(parallel=True, cache=True)
    def _calculate_network_science_vectorized_optimized(data: np.ndarray, window_size: int) -> np.ndarray:
        """ネットワーク科学特徴量計算（完全ベクトル化・並列最適化版）"""
        n = len(data)
        results = np.zeros((n, 5))
        
        if n < window_size:
            return results
        
        # 並列計算
        for i in prange(window_size-1, n):
            window = data[i-window_size+1:i+1]
            
            # 1. ネットワーク密度（ユニークな価格状態の相対頻度）
            min_val = np.min(window)
            max_val = np.max(window)
            
            network_density = 0.0
            if max_val - min_val > 1e-10:
                n_levels = max(5, min(20, int(np.sqrt(len(window)))))
                level_width = (max_val - min_val) / n_levels
                
                levels = np.zeros(len(window), dtype=np.int32)
                for j in range(len(window)):
                    level_idx = int((window[j] - min_val) / level_width)
                    if level_idx >= n_levels:
                        level_idx = n_levels - 1
                    levels[j] = level_idx
                
                unique_levels = len(np.unique(levels))
                max_possible_connections = n_levels * (n_levels - 1) // 2
                
                actual_connections = 0
                for j in range(len(levels) - 1):
                    if levels[j] != levels[j+1]:
                        actual_connections += 1
                
                if max_possible_connections > 0:
                    network_density = actual_connections / max_possible_connections
            
            # 2. 中心性（中央値と平均の関係）
            median_val = np.median(window)
            mean_val = np.mean(window)
            std_val = np.std(window)
            
            centrality_measure = 0.0
            if std_val > 1e-10:
                skewness_approx = (mean_val - median_val) / std_val
                centrality_measure = 1.0 / (1.0 + abs(skewness_approx))
            else:
                centrality_measure = 1.0
            
            # 3. クラスタリング係数（異なる時間軸の相関性）
            clustering_coefficient = 0.0
            if len(window) >= 9:
                segment_size = len(window) // 3
                segments = []
                
                for seg_idx in range(3):
                    start_idx = seg_idx * segment_size
                    end_idx = start_idx + segment_size
                    if end_idx <= len(window):
                        segments.append(window[start_idx:end_idx])
                
                if len(segments) == 3:
                    correlations = []
                    
                    for seg1_idx in range(len(segments)):
                        for seg2_idx in range(seg1_idx + 1, len(segments)):
                            seg1 = segments[seg1_idx]
                            seg2 = segments[seg2_idx]
                            
                            if len(seg1) > 1 and len(seg2) > 1:
                                mean1 = np.mean(seg1)
                                mean2 = np.mean(seg2)
                                std1 = np.std(seg1)
                                std2 = np.std(seg2)
                                
                                if std1 > 1e-10 and std2 > 1e-10:
                                    min_len = min(len(seg1), len(seg2))
                                    seg1_trim = seg1[:min_len]
                                    seg2_trim = seg2[:min_len]
                                    
                                    covariance = np.mean((seg1_trim - mean1) * (seg2_trim - mean2))
                                    correlation = covariance / (std1 * std2)
                                    
                                    if np.isfinite(correlation):
                                        correlations.append(abs(correlation))
                    
                    if len(correlations) > 0:
                        clustering_coefficient = np.mean(correlations)
            
            # 4. 媒介中心性（中心期間の極値存在度）
            betweenness_centrality = 0.0
            if len(window) > 4:
                center_start = len(window) // 3
                center_end = 2 * len(window) // 3
                center_region = window[center_start:center_end]
                
                if len(center_region) > 0:
                    max_val = np.max(window)
                    min_val = np.min(window)
                    center_max = np.max(center_region)
                    center_min = np.min(center_region)
                    
                    max_importance = 1.0 if abs(center_max - max_val) < 1e-10 else 0.0
                    min_importance = 1.0 if abs(center_min - min_val) < 1e-10 else 0.0
                    
                    center_std = np.std(center_region)
                    window_std = np.std(window)
                    
                    if window_std > 1e-10:
                        variability_ratio = center_std / window_std
                    else:
                        variability_ratio = 0.0
                    
                    betweenness_centrality = (max_importance + min_importance) * 0.5 + variability_ratio * 0.5
            
            # 5. 固有ベクトル中心性（自己相関の累積）
            eigenvector_centrality = 0.0
            max_lag = min(10, len(window) // 3)
            
            if max_lag > 1:
                autocorr_sum = 0.0
                weight_sum = 0.0
                
                for lag in range(1, max_lag + 1):
                    if len(window) > lag:
                        x1 = window[:-lag]
                        x2 = window[lag:]
                        
                        if len(x1) > 1 and len(x2) > 1:
                            mean1 = np.mean(x1)
                            mean2 = np.mean(x2)
                            std1 = np.std(x1)
                            std2 = np.std(x2)
                            
                            if std1 > 1e-10 and std2 > 1e-10:
                                covariance = np.mean((x1 - mean1) * (x2 - mean2))
                                correlation = covariance / (std1 * std2)
                                
                                if np.isfinite(correlation):
                                    weight = 1.0 / lag
                                    autocorr_sum += abs(correlation) * weight
                                    weight_sum += weight
                
                if weight_sum > 1e-10:
                    eigenvector_centrality = autocorr_sum / weight_sum
            
            # 結果の格納
            results[i, 0] = network_density
            results[i, 1] = centrality_measure
            results[i, 2] = clustering_coefficient
            results[i, 3] = betweenness_centrality
            results[i, 4] = eigenvector_centrality
        
        return results
    
    def calculate_acoustics_features(self, data: np.ndarray) -> Dict[str, np.ndarray]:
        """音響学特徴量（Polars最適化版）"""
        features = {}
        window_size = 64  # 音響学に適したウィンドウサイズ
        
        # Numba最適化版音響学計算
        acoustics_result = self._numba_safe_calculation(
            self._calculate_acoustics_vectorized_optimized, data, window_size
        )
        
        if isinstance(acoustics_result, np.ndarray) and len(acoustics_result.shape) == 2 and acoustics_result.shape[1] >= 5:
            acoustics_names = ['acoustic_power', 'acoustic_frequency', 'amplitude_modulation',
                             'phase_modulation', 'acoustic_echo']
            
            for j, name in enumerate(acoustics_names):
                features[f'{name}_{window_size}'] = acoustics_result[:, j]
        else:
            # フォールバック値
            acoustics_names = ['acoustic_power', 'acoustic_frequency', 'amplitude_modulation',
                             'phase_modulation', 'acoustic_echo']
            for name in acoustics_names:
                features[f'{name}_{window_size}'] = np.zeros(len(data))
        
        return features
    
    @staticmethod
    @njit(parallel=True, cache=True)
    def _calculate_acoustics_vectorized_optimized(data: np.ndarray, window_size: int) -> np.ndarray:
        """音響学特徴量計算（完全ベクトル化・並列最適化版）"""
        n = len(data)
        results = np.zeros((n, 5))
        
        if n < window_size:
            return results
        
        # 並列計算
        for i in prange(window_size-1, n):
            window = data[i-window_size+1:i+1]
            
            try:
                returns = np.diff(window) if len(window) > 1 else np.array([0])
                
                # 1. 音響パワー（リターンの二乗和）
                acoustic_power = 0.0
                if len(returns) > 0:
                    acoustic_power = np.sqrt(np.mean(returns**2))
                    
                    window_std = np.std(window)
                    if window_std > 1e-10:
                        acoustic_power = acoustic_power / window_std
                
                # 2. 音響周波数（FFTによる主要周波数）
                acoustic_frequency = 0.0
                if len(window) >= 8:
                    max_energy = 0.0
                    dominant_freq_idx = 0
                    
                    for freq_idx in range(1, min(len(window) // 2, 16)):
                        freq = freq_idx * 2 * np.pi / len(window)
                        real_part = 0.0
                        imag_part = 0.0
                        
                        for j in range(len(window)):
                            angle = freq * j
                            real_part += window[j] * np.cos(angle)
                            imag_part += window[j] * np.sin(angle)
                        
                        energy = real_part**2 + imag_part**2
                        
                        if energy > max_energy:
                            max_energy = energy
                            dominant_freq_idx = freq_idx
                    
                    if len(window) > 2:
                        acoustic_frequency = dominant_freq_idx / (len(window) // 2)
                
                # 3. 振幅変調（ボラティリティの変動）
                amplitude_modulation = 0.0
                if len(returns) >= 4:
                    rolling_vol = []
                    vol_window = min(4, len(returns) // 2)
                    
                    for j in range(vol_window - 1, len(returns)):
                        vol_segment = returns[j - vol_window + 1:j + 1]
                        rolling_vol.append(np.std(vol_segment))
                    
                    if len(rolling_vol) > 1:
                        vol_changes = np.diff(rolling_vol)
                        amplitude_modulation = np.std(vol_changes)
                        
                        mean_vol = np.mean(rolling_vol)
                        if mean_vol > 1e-10:
                            amplitude_modulation = amplitude_modulation / mean_vol
                
                # 4. 位相変調（FFT位相の安定性）
                phase_modulation = 0.0
                if len(window) >= 8:
                    phase_variations = []
                    
                    for freq_idx in range(1, min(len(window) // 4, 8)):
                        freq = freq_idx * 2 * np.pi / len(window)
                        phases = []
                        
                        segment_size = len(window) // 4
                        if segment_size >= 2:
                            for seg_start in range(0, len(window) - segment_size + 1, segment_size):
                                segment = window[seg_start:seg_start + segment_size]
                                
                                real_part = 0.0
                                imag_part = 0.0
                                
                                for j in range(len(segment)):
                                    angle = freq * j
                                    real_part += segment[j] * np.cos(angle)
                                    imag_part += segment[j] * np.sin(angle)
                                
                                if real_part != 0 or imag_part != 0:
                                    phase = np.arctan2(imag_part, real_part)
                                    phases.append(phase)
                            
                            if len(phases) > 1:
                                phase_diffs = []
                                for j in range(1, len(phases)):
                                    diff = phases[j] - phases[j-1]
                                    while diff > np.pi:
                                        diff -= 2 * np.pi
                                    while diff < -np.pi:
                                        diff += 2 * np.pi
                                    phase_diffs.append(diff)
                                
                                if len(phase_diffs) > 0:
                                    phase_var = np.var(phase_diffs)
                                    phase_variations.append(phase_var)
                    
                    if len(phase_variations) > 0:
                        avg_phase_var = np.mean(phase_variations)
                        phase_modulation = min(1.0, avg_phase_var / (np.pi**2))
                
                # 5. 音響エコー（前半と後半の相関）
                acoustic_echo = 0.0
                if len(window) >= 8:
                    quarter = len(window) // 4
                    half = len(window) // 2
                    
                    if quarter > 0:
                        echo_correlations = []
                        
                        # 1/4遅延エコー
                        if len(window) >= quarter * 2:
                            source1 = window[:quarter]
                            echo1 = window[quarter:quarter*2]
                            
                            if len(source1) == len(echo1) and len(source1) > 1:
                                mean_s1 = np.mean(source1)
                                mean_e1 = np.mean(echo1)
                                std_s1 = np.std(source1)
                                std_e1 = np.std(echo1)
                                
                                if std_s1 > 1e-10 and std_e1 > 1e-10:
                                    covariance = np.mean((source1 - mean_s1) * (echo1 - mean_e1))
                                    correlation = covariance / (std_s1 * std_e1)
                                    if np.isfinite(correlation):
                                        echo_correlations.append(abs(correlation))
                        
                        # 1/2遅延エコー
                        if len(window) >= half * 2:
                            source2 = window[:half]
                            echo2 = window[half:]
                            
                            min_len = min(len(source2), len(echo2))
                            if min_len > 1:
                                source2_trim = source2[:min_len]
                                echo2_trim = echo2[:min_len]
                                
                                mean_s2 = np.mean(source2_trim)
                                mean_e2 = np.mean(echo2_trim)
                                std_s2 = np.std(source2_trim)
                                std_e2 = np.std(echo2_trim)
                                
                                if std_s2 > 1e-10 and std_e2 > 1e-10:
                                    covariance = np.mean((source2_trim - mean_s2) * (echo2_trim - mean_e2))
                                    correlation = covariance / (std_s2 * std_e2)
                                    if np.isfinite(correlation):
                                        echo_correlations.append(abs(correlation))
                        
                        if len(echo_correlations) > 0:
                            acoustic_echo = np.mean(echo_correlations)
                
                # 結果の格納
                results[i, 0] = acoustic_power
                results[i, 1] = acoustic_frequency
                results[i, 2] = amplitude_modulation
                results[i, 3] = phase_modulation
                results[i, 4] = acoustic_echo
                
            except:
                # 計算失敗時は0で埋める
                results[i, 0] = 0.0
                results[i, 1] = 0.0
                results[i, 2] = 0.0
                results[i, 3] = 0.0
                results[i, 4] = 0.0
        
        return results
    
    def calculate_linguistics_features(self, data: np.ndarray) -> Dict[str, np.ndarray]:
        """言語学特徴量（Polars最適化版）"""
        features = {}
        window_size = 50
        
        # Numba最適化版言語学計算
        linguistics_result = self._numba_safe_calculation(
            self._calculate_linguistics_vectorized_optimized, data, window_size
        )
        
        if isinstance(linguistics_result, np.ndarray) and len(linguistics_result.shape) == 2 and linguistics_result.shape[1] >= 5:
            linguistics_names = ['vocabulary_diversity', 'sentence_structure', 'linguistic_complexity',
                               'word_order', 'prosody']
            
            for j, name in enumerate(linguistics_names):
                features[f'{name}_{window_size}'] = linguistics_result[:, j]
        else:
            # フォールバック値
            linguistics_names = ['vocabulary_diversity', 'sentence_structure', 'linguistic_complexity',
                               'word_order', 'prosody']
            for name in linguistics_names:
                features[f'{name}_{window_size}'] = np.zeros(len(data))
        
        return features
    
    @staticmethod
    @njit(parallel=True, cache=True)
    def _calculate_linguistics_vectorized_optimized(data: np.ndarray, window_size: int) -> np.ndarray:
        """言語学特徴量計算（完全ベクトル化・並列最適化版）"""
        n = len(data)
        results = np.zeros((n, 5))
        
        if n < window_size:
            return results
        
        # 並列計算
        for i in prange(window_size-1, n):
            window = data[i-window_size+1:i+1]
            
            # 価格を「単語」に対応する離散レベルに変換
            min_val = np.min(window)
            max_val = np.max(window)
            
            if max_val - min_val < 1e-10:
                results[i, 0] = 0.0  # vocabulary_diversity
                results[i, 1] = 1.0  # sentence_structure
                results[i, 2] = 0.0  # linguistic_complexity
                results[i, 3] = 0.0  # word_order
                results[i, 4] = 0.0  # prosody
                continue
            
            # 適応的語彙レベル数
            unique_vals = len(np.unique(window))
            n_levels = max(3, min(20, int(np.sqrt(unique_vals))))
            level_width = (max_val - min_val) / n_levels
            
            # 価格レベル（語彙）の離散化
            word_levels = np.zeros(len(window), dtype=np.int32)
            for j in range(len(window)):
                level_idx = int((window[j] - min_val) / level_width)
                if level_idx >= n_levels:
                    level_idx = n_levels - 1
                word_levels[j] = level_idx
            
            # 1. 語彙の多様性
            unique_words = len(np.unique(word_levels))
            vocabulary_diversity = unique_words / n_levels
            
            if unique_words > 1:
                word_counts = np.zeros(n_levels)
                for level in word_levels:
                    word_counts[level] += 1
                
                non_zero_counts = word_counts[word_counts > 0]
                if len(non_zero_counts) > 1:
                    total_words = len(window)
                    entropy = 0.0
                    for count in non_zero_counts:
                        if count > 0:
                            p = count / total_words
                            entropy -= p * np.log2(p)
                    
                    max_entropy = np.log2(len(non_zero_counts))
                    if max_entropy > 0:
                        uniformity = entropy / max_entropy
                        vocabulary_diversity = (vocabulary_diversity + uniformity) / 2.0
            
            # 2. 文の構造（変化の滑らかさ）
            sentence_structure = 0.0
            if len(window) >= 3:
                first_diff = np.diff(window)
                second_diff = np.diff(first_diff)
                
                if len(second_diff) > 0:
                    second_diff_std = np.std(second_diff)
                    first_diff_std = np.std(first_diff)
                    
                    if first_diff_std > 1e-10:
                        smoothness = 1.0 / (1.0 + second_diff_std / first_diff_std)
                        sentence_structure = smoothness
                    else:
                        sentence_structure = 1.0
            
            # 3. 言語的複雑性
            linguistic_complexity = 0.0
            if len(window) > 1:
                returns = np.diff(window)
                
                threshold = np.std(returns) * 1.5
                large_changes_count = 0
                for ret in returns:
                    if abs(ret) > threshold:
                        large_changes_count += 1
                
                change_frequency = large_changes_count / len(returns)
                
                # パターンの複雑さ
                pattern_complexity = 0.0
                max_pattern_len = min(5, len(word_levels) // 3)
                
                if max_pattern_len >= 2:
                    total_patterns = 0
                    unique_patterns = set()
                    
                    for pattern_len in range(2, max_pattern_len + 1):
                        for j in range(len(word_levels) - pattern_len + 1):
                            pattern_hash = 0
                            for k in range(pattern_len):
                                pattern_hash = pattern_hash * n_levels + word_levels[j + k]
                            
                            unique_patterns.add(pattern_hash)
                            total_patterns += 1
                    
                    if total_patterns > 0:
                        pattern_complexity = len(unique_patterns) / total_patterns
                
                linguistic_complexity = (change_frequency + pattern_complexity) / 2.0
            
            # 4. 語順（価格順位の変化パターン）
            word_order = 0.0
            if len(window) >= 4:
                ranks = np.zeros(len(window))
                for j in range(len(window)):
                    rank = 0
                    for k in range(len(window)):
                        if window[k] <= window[j]:
                            rank += 1
                    ranks[j] = rank / len(window)
                
                rank_changes = np.abs(np.diff(ranks))
                
                if len(rank_changes) > 0:
                    rank_volatility = np.std(rank_changes)
                    
                    rank_direction_changes = 0
                    for j in range(1, len(rank_changes)):
                        if rank_changes[j] * rank_changes[j-1] < 0:
                            rank_direction_changes += 1
                    
                    direction_consistency = 1.0 - (rank_direction_changes / max(1, len(rank_changes) - 1))
                    
                    word_order = rank_volatility * direction_consistency
            
            # 5. プロソディ（抑揚・リズム）
            prosody = 0.0
            if len(window) > 1:
                returns = np.diff(window)
                
                amplitude_variation = np.std(np.abs(returns))
                
                rhythm_score = 0.0
                if len(returns) >= 4:
                    max_lag = min(len(returns) // 2, 10)
                    autocorrelations = []
                    
                    for lag in range(1, max_lag + 1):
                        if len(returns) > lag:
                            x1 = returns[:-lag]
                            x2 = returns[lag:]
                            
                            if len(x1) > 1 and len(x2) > 1:
                                mean1 = np.mean(x1)
                                mean2 = np.mean(x2)
                                std1 = np.std(x1)
                                std2 = np.std(x2)
                                
                                if std1 > 1e-10 and std2 > 1e-10:
                                    covariance = np.mean((x1 - mean1) * (x2 - mean2))
                                    correlation = covariance / (std1 * std2)
                                    
                                    if np.isfinite(correlation):
                                        autocorrelations.append(abs(correlation))
                    
                    if len(autocorrelations) > 0:
                        rhythm_score = np.max(autocorrelations)
                
                window_std = np.std(window)
                if window_std > 1e-10:
                    normalized_amplitude = amplitude_variation / window_std
                else:
                    normalized_amplitude = 0.0
                
                prosody = (normalized_amplitude + rhythm_score) / 2.0
            
            # 結果の格納
            results[i, 0] = vocabulary_diversity
            results[i, 1] = sentence_structure
            results[i, 2] = linguistic_complexity
            results[i, 3] = word_order
            results[i, 4] = prosody
        
        return results
    
    def calculate_aesthetics_features(self, data: np.ndarray) -> Dict[str, np.ndarray]:
        """美学特徴量（Polars最適化版）"""
        features = {}
        window_size = 50
        
        # Numba最適化版美学計算
        aesthetics_result = self._numba_safe_calculation(
            self._calculate_aesthetics_vectorized_optimized, data, window_size
        )
        
        if isinstance(aesthetics_result, np.ndarray) and len(aesthetics_result.shape) == 2 and aesthetics_result.shape[1] >= 5:
            aesthetics_names = ['golden_ratio_adherence', 'symmetry_measure', 'aesthetic_harmony',
                              'proportional_beauty', 'visual_balance']
            
            for j, name in enumerate(aesthetics_names):
                features[f'{name}_{window_size}'] = aesthetics_result[:, j]
        else:
            # フォールバック値
            aesthetics_names = ['golden_ratio_adherence', 'symmetry_measure', 'aesthetic_harmony',
                              'proportional_beauty', 'visual_balance']
            for name in aesthetics_names:
                features[f'{name}_{window_size}'] = np.zeros(len(data))
        
        return features
    
    @staticmethod
    @njit(parallel=True, cache=True)
    def _calculate_aesthetics_vectorized_optimized(data: np.ndarray, window_size: int) -> np.ndarray:
        """美学特徴量計算（完全ベクトル化・並列最適化版）"""
        n = len(data)
        results = np.zeros((n, 5))
        
        if n < window_size:
            return results
        
        golden_ratio = (1 + np.sqrt(5)) / 2  # φ ≈ 1.618
        
        # 並列計算
        for i in prange(window_size-1, n):
            window = data[i-window_size+1:i+1]
            
            # 1. 黄金比への固着度
            golden_ratio_adherence = 0.0
            if len(window) >= 3:
                golden_deviations = []
                
                # 隣接比率
                for j in range(1, len(window)):
                    if abs(window[j-1]) > 1e-10:
                        ratio = abs(window[j] / window[j-1])
                        if 0.1 <= ratio <= 10.0:
                            deviation = abs(ratio - golden_ratio)
                            golden_deviations.append(deviation)
                
                # セグメント比率
                if len(window) >= 8:
                    third = len(window) // 3
                    short_segment = window[:third]
                    long_segment = window[third:]
                    
                    short_mean = np.mean(np.abs(short_segment))
                    long_mean = np.mean(np.abs(long_segment))
                    
                    if short_mean > 1e-10:
                        segment_ratio = long_mean / short_mean
                        if 0.1 <= segment_ratio <= 10.0:
                            segment_deviation = abs(segment_ratio - golden_ratio)
                            golden_deviations.append(segment_deviation)
                
                # レンジ比率
                max_val = np.max(window)
                min_val = np.min(window)
                range_val = max_val - min_val
                mean_val = np.mean(window)
                
                if abs(mean_val - min_val) > 1e-10:
                    range_ratio = range_val / abs(mean_val - min_val)
                    if 0.1 <= range_ratio <= 10.0:
                        range_deviation = abs(range_ratio - golden_ratio)
                        golden_deviations.append(range_deviation)
                
                if len(golden_deviations) > 0:
                    avg_deviation = np.mean(golden_deviations)
                    golden_ratio_adherence = 1.0 / (1.0 + avg_deviation)
            
            # 2. 対称性
            symmetry_measure = 0.0
            if len(window) >= 4:
                symmetry_scores = []
                
                # 中央対称
                half = len(window) // 2
                first_half = window[:half]
                second_half = window[-half:]
                second_half_reversed = second_half[::-1]
                
                if len(first_half) == len(second_half_reversed) and len(first_half) > 1:
                    std1 = np.std(first_half)
                    std2 = np.std(second_half_reversed)
                    
                    if std1 > 1e-10 and std2 > 1e-10:
                        mean1 = np.mean(first_half)
                        mean2 = np.mean(second_half_reversed)
                        
                        covariance = np.mean((first_half - mean1) * (second_half_reversed - mean2))
                        correlation = covariance / (std1 * std2)
                        
                        if np.isfinite(correlation):
                            symmetry_scores.append(abs(correlation))
                
                # 3分割対称
                if len(window) >= 6:
                    golden_split = int(len(window) / golden_ratio)
                    if golden_split > 0 and golden_split < len(window) - 1:
                        left_segment = window[:golden_split]
                        right_segment = window[golden_split:]
                        right_segment_reversed = right_segment[::-1]
                        
                        min_len = min(len(left_segment), len(right_segment_reversed))
                        if min_len > 1:
                            left_trim = left_segment[:min_len]
                            right_trim = right_segment_reversed[:min_len]
                            
                            std_left = np.std(left_trim)
                            std_right = np.std(right_trim)
                            
                            if std_left > 1e-10 and std_right > 1e-10:
                                mean_left = np.mean(left_trim)
                                mean_right = np.mean(right_trim)
                                
                                covariance = np.mean((left_trim - mean_left) * (right_trim - mean_right))
                                correlation = covariance / (std_left * std_right)
                                
                                if np.isfinite(correlation):
                                    symmetry_scores.append(abs(correlation))
                
                if len(symmetry_scores) > 0:
                    symmetry_measure = np.mean(symmetry_scores)
            
            # 3. 美的調和
            aesthetic_harmony = 0.0
            if len(window) >= 4:
                try:
                    first_diff = np.diff(window)
                    smoothness_scores = []
                    
                    if len(first_diff) >= 2:
                        second_diff = np.diff(first_diff)
                        
                        if len(second_diff) > 0:
                            second_diff_std = np.std(second_diff)
                            first_diff_std = np.std(first_diff)
                            
                            if first_diff_std > 1e-10:
                                smoothness_2nd = 1.0 / (1.0 + second_diff_std / first_diff_std)
                                smoothness_scores.append(smoothness_2nd)
                        
                        if len(second_diff) >= 1:
                            third_diff = np.diff(second_diff)
                            
                            if len(third_diff) > 0:
                                third_diff_std = np.std(third_diff)
                                
                                if second_diff_std > 1e-10:
                                    smoothness_3rd = 1.0 / (1.0 + third_diff_std / second_diff_std)
                                    smoothness_scores.append(smoothness_3rd)
                    
                    if len(first_diff) > 1:
                        curvature_changes = np.abs(np.diff(first_diff))
                        curvature_consistency = 1.0 / (1.0 + np.std(curvature_changes))
                        smoothness_scores.append(curvature_consistency)
                    
                    if len(smoothness_scores) > 0:
                        aesthetic_harmony = np.mean(smoothness_scores)
                
                except:
                    aesthetic_harmony = 0.5
            
            # 4. 比例美
            proportional_beauty = 0.0
            if len(window) > 1:
                returns = np.diff(window)
                
                thresholds = [0.0, np.std(returns) * 0.1, np.std(returns) * 0.5]
                balance_scores = []
                
                for threshold in thresholds:
                    up_periods = 0
                    down_periods = 0
                    for ret in returns:
                        if ret > threshold:
                            up_periods += 1
                        elif ret < -threshold:
                            down_periods += 1
                    
                    total_periods = up_periods + down_periods
                    
                    if total_periods > 0:
                        up_ratio = up_periods / total_periods
                        optimal_ratio = 1.0 / golden_ratio
                        balance_score = 1.0 - abs(up_ratio - optimal_ratio)
                        balance_scores.append(max(0.0, balance_score))
                
                if len(balance_scores) > 0:
                    proportional_beauty = np.mean(balance_scores)
            
            # 5. 視覚的バランス
            visual_balance = 0.0
            if len(window) > 2:
                mean_val = np.mean(window)
                median_val = np.median(window)
                
                if len(window) >= 4:
                    sorted_window = np.sort(window)
                    q25_idx = len(sorted_window) // 4
                    q75_idx = 3 * len(sorted_window) // 4
                    
                    q25 = sorted_window[q25_idx]
                    q75 = sorted_window[q75_idx]
                    
                    lower_dist = median_val - q25
                    upper_dist = q75 - median_val
                    
                    if lower_dist + upper_dist > 1e-10:
                        quartile_balance = 1.0 - abs(lower_dist - upper_dist) / (lower_dist + upper_dist)
                    else:
                        quartile_balance = 1.0
                    
                    window_range = np.max(window) - np.min(window)
                    if window_range > 1e-10:
                        center_alignment = 1.0 - abs(mean_val - median_val) / window_range
                    else:
                        center_alignment = 1.0
                    
                    visual_balance = (quartile_balance + center_alignment) / 2.0
            
            # 結果の格納
            results[i, 0] = golden_ratio_adherence
            results[i, 1] = symmetry_measure
            results[i, 2] = aesthetic_harmony
            results[i, 3] = proportional_beauty
            results[i, 4] = visual_balance
        
        return results
    
    def calculate_music_theory_features(self, data: np.ndarray) -> Dict[str, np.ndarray]:
        """音楽理論特徴量（Polars最適化版）"""
        features = {}
        window_size = 50
        
        # Numba最適化版音楽理論計算
        music_result = self._numba_safe_calculation(
            self._calculate_music_theory_vectorized_optimized, data, window_size
        )
        
        if isinstance(music_result, np.ndarray) and len(music_result.shape) == 2 and music_result.shape[1] >= 6:
            music_names = ['tonality', 'rhythm_pattern', 'harmony', 'melody_contour', 
                          'musical_tension', 'tempo']
            
            for j, name in enumerate(music_names):
                features[f'{name}_{window_size}'] = music_result[:, j]
        else:
            # フォールバック値
            music_names = ['tonality', 'rhythm_pattern', 'harmony', 'melody_contour', 
                          'musical_tension', 'tempo']
            for name in music_names:
                features[f'{name}_{window_size}'] = np.zeros(len(data))
        
        return features
    
    @staticmethod
    @njit(parallel=True, cache=True)
    def _calculate_music_theory_vectorized_optimized(data: np.ndarray, window_size: int) -> np.ndarray:
        """音楽理論特徴量計算（完全ベクトル化・並列最適化版）"""
        n = len(data)
        results = np.zeros((n, 6))
        
        if n < window_size:
            return results
        
        # 並列計算
        for i in prange(window_size-1, n):
            window = data[i-window_size+1:i+1]
            returns = np.diff(window) if len(window) > 1 else np.array([0])
            
            # 1. 調性（安定性と方向性の比率）
            tonality = 0.0
            if len(returns) > 0:
                return_mean = np.mean(returns)
                return_std = np.std(returns)
                
                positive_count = 0
                negative_count = 0
                for ret in returns:
                    if ret > 0:
                        positive_count += 1
                    elif ret < 0:
                        negative_count += 1
                
                total_count = len(returns)
                
                if total_count > 0:
                    positive_ratio = positive_count / total_count
                    major_tendency = positive_ratio
                    minor_tendency = 1.0 - positive_ratio
                    
                    tonality_clarity = abs(major_tendency - minor_tendency)
                    
                    if abs(return_mean) > 1e-10:
                        stability = 1.0 / (1.0 + return_std / abs(return_mean))
                    else:
                        stability = 1.0 / (1.0 + return_std)
                    
                    tonality = (tonality_clarity + stability) / 2.0
            
            # 2. リズムパターン（大きな変化の規則性）
            rhythm_pattern = 0.0
            if len(returns) > 4:
                threshold = np.std(returns) * 1.5
                accents_count = 0
                for ret in returns:
                    if abs(ret) > threshold:
                        accents_count += 1
                
                if accents_count > 1:
                    accent_positions = []
                    for j in range(len(returns)):
                        if abs(returns[j]) > threshold:
                            accent_positions.append(j)
                    
                    if len(accent_positions) > 2:
                        intervals = np.diff(accent_positions)
                        
                        if len(intervals) > 1:
                            interval_regularity = 1.0 / (1.0 + np.std(intervals))
                            
                            mean_interval = np.mean(intervals)
                            musical_intervals = np.array([2, 3, 4, 6, 8])
                            interval_musicality = 0.0
                            
                            for musical_int in musical_intervals:
                                closeness = 1.0 / (1.0 + abs(mean_interval - musical_int))
                                interval_musicality = max(interval_musicality, closeness)
                            
                            rhythm_pattern = (interval_regularity + interval_musicality) / 2.0
            
            # 3. 和声（FFT周波数比の協和音程性）
            harmony = 0.0
            if len(window) >= 8:
                max_harmonics = min(8, len(window) // 2)
                frequency_energies = []
                
                for freq_idx in range(1, max_harmonics + 1):
                    freq = freq_idx * 2 * np.pi / len(window)
                    real_part = 0.0
                    imag_part = 0.0
                    
                    for j in range(len(window)):
                        angle = freq * j
                        real_part += window[j] * np.cos(angle)
                        imag_part += window[j] * np.sin(angle)
                    
                    energy = real_part**2 + imag_part**2
                    frequency_energies.append(energy)
                
                if len(frequency_energies) >= 3:
                    sorted_indices = np.argsort(frequency_energies)[::-1]
                    top_freqs = sorted_indices[:3]
                    
                    harmonic_ratios = []
                    for j in range(len(top_freqs)):
                        for k in range(j + 1, len(top_freqs)):
                            if top_freqs[j] > 0:
                                ratio = top_freqs[k] / top_freqs[j]
                                harmonic_ratios.append(ratio)
                    
                    consonant_ratios = np.array([0.5, 2.0/3.0, 0.75, 1.0, 1.25, 1.5, 2.0])
                    harmony_scores = []
                    
                    for ratio in harmonic_ratios:
                        consonance = 0.0
                        for cons_ratio in consonant_ratios:
                            closeness = 1.0 / (1.0 + abs(ratio - cons_ratio) * 10)
                            consonance = max(consonance, closeness)
                        harmony_scores.append(consonance)
                    
                    if len(harmony_scores) > 0:
                        harmony = np.mean(harmony_scores)
            
            # 4. 旋律の輪郭（上昇傾向の方向性）
            melody_contour = 0.0
            if len(returns) > 0:
                up_moves = 0
                down_moves = 0
                for ret in returns:
                    if ret > 0:
                        up_moves += 1
                    elif ret < 0:
                        down_moves += 1
                
                total_moves = len(returns)
                
                if total_moves > 0:
                    up_ratio = up_moves / total_moves
                    
                    direction_changes = 0
                    for j in range(1, len(returns)):
                        if returns[j] * returns[j-1] < 0:
                            direction_changes += 1
                    
                    contour_complexity = direction_changes / max(1, len(returns) - 1)
                    
                    large_leaps = 0
                    threshold = np.std(returns) * 2
                    for ret in returns:
                        if abs(ret) > threshold:
                            large_leaps += 1
                    
                    leap_ratio = large_leaps / len(returns)
                    melodic_smoothness = 1.0 - leap_ratio
                    
                    melody_contour = (up_ratio + (1.0 - contour_complexity) + melodic_smoothness) / 3.0
            
            # 5. 音楽的緊張（変動範囲とダイナミクス）
            musical_tension = 0.0
            if len(window) > 2:
                dynamic_range = np.max(window) - np.min(window)
                mean_val = np.mean(window)
                
                if abs(mean_val) > 1e-10:
                    relative_range = dynamic_range / abs(mean_val)
                else:
                    relative_range = dynamic_range
                
                cumulative_tension = 0.0
                peak_releases = 0
                
                if len(returns) > 1:
                    cumulative_returns = np.zeros(len(returns))
                    cumulative_returns[0] = returns[0]
                    for j in range(1, len(returns)):
                        cumulative_returns[j] = cumulative_returns[j-1] + returns[j]
                    
                    for j in range(1, len(cumulative_returns)):
                        if cumulative_returns[j] * cumulative_returns[j-1] > 0:
                            cumulative_tension += abs(returns[j])
                        else:
                            if cumulative_tension > np.std(returns):
                                peak_releases += 1
                            cumulative_tension = 0
                    
                    if len(returns) > 0:
                        release_frequency = peak_releases / len(returns)
                        
                        total_abs_returns = 0.0
                        for ret in returns:
                            total_abs_returns += abs(ret)
                        
                        if total_abs_returns > 0:
                            tension_buildup = cumulative_tension / total_abs_returns
                        else:
                            tension_buildup = 0.0
                        
                        musical_tension = (relative_range + tension_buildup + release_frequency) / 3.0
            
            # 6. テンポ（変化の頻度）
            tempo = 0.0
            if len(returns) > 1:
                direction_changes = 0
                for j in range(1, len(returns)):
                    if returns[j] * returns[j-1] < 0:
                        direction_changes += 1
                
                basic_tempo = direction_changes / len(returns)
                
                threshold = np.std(returns) * 0.5
                significant_changes = 0
                
                for ret in returns:
                    if abs(ret) > threshold:
                        significant_changes += 1
                
                intensity_tempo = significant_changes / len(returns)
                
                tempo_stability = 0.0
                if len(returns) >= 4:
                    max_lag = min(len(returns) // 2, 8)
                    max_autocorr = 0.0
                    
                    for lag in range(1, max_lag + 1):
                        if len(returns) > lag:
                            x1 = returns[:-lag]
                            x2 = returns[lag:]
                            
                            if len(x1) > 1 and len(x2) > 1:
                                mean1 = np.mean(x1)
                                mean2 = np.mean(x2)
                                std1 = np.std(x1)
                                std2 = np.std(x2)
                                
                                if std1 > 1e-10 and std2 > 1e-10:
                                    covariance = np.mean((x1 - mean1) * (x2 - mean2))
                                    correlation = covariance / (std1 * std2)
                                    
                                    if np.isfinite(correlation):
                                        max_autocorr = max(max_autocorr, abs(correlation))
                    
                    tempo_stability = max_autocorr
                
                tempo = (basic_tempo + intensity_tempo + tempo_stability) / 3.0
            
            # 結果の格納
            results[i, 0] = tonality
            results[i, 1] = rhythm_pattern
            results[i, 2] = harmony
            results[i, 3] = melody_contour
            results[i, 4] = musical_tension
            results[i, 5] = tempo
        
        return results
    
    
    def calculate_biomechanics_features(self, data: np.ndarray) -> Dict[str, np.ndarray]:
        """生体力学・パフォーマンス科学特徴量（Polars最適化版）"""
        features = {}
        window_size = 50
        
        # Numba最適化版生体力学計算
        biomechanics_result = self._numba_safe_calculation(
            self._calculate_biomechanics_vectorized_optimized, data, window_size
        )
        
        if isinstance(biomechanics_result, np.ndarray) and len(biomechanics_result.shape) == 2 and biomechanics_result.shape[1] >= 7:
            biomechanics_names = ['kinetic_energy', 'potential_energy', 'muscle_force',
                                'joint_mobility', 'performance_consistency', 'endurance', 'recovery_rate']
            
            for j, name in enumerate(biomechanics_names):
                features[f'{name}_{window_size}'] = biomechanics_result[:, j]
        else:
            # フォールバック値
            biomechanics_names = ['kinetic_energy', 'potential_energy', 'muscle_force',
                                'joint_mobility', 'performance_consistency', 'endurance', 'recovery_rate']
            for name in biomechanics_names:
                features[f'{name}_{window_size}'] = np.zeros(len(data))
        
        return features
    
    @staticmethod
    @njit(parallel=True, cache=True)
    def _calculate_biomechanics_vectorized_optimized(data: np.ndarray, window_size: int) -> np.ndarray:
        """生体力学・パフォーマンス科学特徴量計算（完全ベクトル化・並列最適化版）"""
        n = len(data)
        results = np.zeros((n, 7))
        
        if n < window_size:
            return results
        
        # 並列計算
        for i in prange(window_size-1, n):
            window = data[i-window_size+1:i+1]
            
            if len(window) < 2:
                continue
            
            returns = np.diff(window)
            
            # 1. 運動エネルギー（動きの激しさ）
            kinetic_energy = 0.0
            if len(returns) > 0:
                basic_ke = 0.5 * np.sum(returns**2)
                
                velocity_variance = np.var(returns)
                velocity_skewness = 0.0
                
                if len(returns) >= 3 and velocity_variance > 1e-10:
                    mean_velocity = np.mean(returns)
                    std_velocity = np.sqrt(velocity_variance)
                    
                    skew_sum = 0.0
                    for vel in returns:
                        normalized_vel = (vel - mean_velocity) / std_velocity
                        skew_sum += normalized_vel**3
                    
                    velocity_skewness = skew_sum / len(returns)
                
                total_displacement = abs(window[-1] - window[0])
                total_distance = np.sum(np.abs(returns))
                
                if total_distance > 1e-10:
                    movement_efficiency = total_displacement / total_distance
                else:
                    movement_efficiency = 0.0
                
                kinetic_energy = basic_ke * (1.0 + movement_efficiency)
            
            # 2. 位置エネルギー（価格水準からの乖離エネルギー）
            potential_energy = 0.0
            if len(window) > 0:
                min_level = np.min(window)
                height_energy = np.sum((window - min_level)**2)
                
                center_of_mass = np.mean(window)
                gravitational_potential = 0.0
                
                for j in range(len(window)):
                    distance_from_center = abs(window[j] - center_of_mass)
                    if distance_from_center > 1e-10:
                        gravitational_potential += 1.0 / distance_from_center
                
                potential_energy = height_energy + gravitational_potential
            
            # 3. 筋力（瞬発的な最大変動）
            muscle_force = 0.0
            if len(returns) > 0:
                max_instantaneous_force = np.max(np.abs(returns))
                
                force_threshold = np.std(returns) * 1.5
                sustained_force_periods = 0
                
                for ret in returns:
                    if abs(ret) > force_threshold:
                        sustained_force_periods += 1
                
                force_endurance_ratio = sustained_force_periods / len(returns)
                
                if len(returns) >= 2:
                    power_outputs = []
                    for j in range(1, len(returns)):
                        force = abs(returns[j])
                        velocity = abs(returns[j] - returns[j-1])
                        power = force * velocity
                        power_outputs.append(power)
                    
                    if len(power_outputs) > 0:
                        peak_power = np.max(power_outputs)
                        avg_power = np.mean(power_outputs)
                        
                        power_ratio = peak_power / (avg_power + 1e-10)
                        muscle_force = (max_instantaneous_force + force_endurance_ratio + power_ratio) / 3.0
                else:
                    muscle_force = max_instantaneous_force
            
            # 4. 関節可動域（変動範囲の相対的な広さ）
            joint_mobility = 0.0
            if len(window) > 0:
                price_range = np.max(window) - np.min(window)
                mean_price = np.mean(window)
                
                if abs(mean_price) > 1e-10:
                    basic_mobility = price_range / abs(mean_price)
                else:
                    basic_mobility = price_range
                
                if len(returns) >= 2:
                    acceleration = np.diff(returns)
                    acceleration_smoothness = 1.0 / (1.0 + np.std(acceleration))
                    
                    utilization_levels = np.linspace(np.min(window), np.max(window), 10)
                    utilization_count = 0
                    
                    for level in utilization_levels:
                        tolerance = price_range * 0.05
                        found = False
                        for val in window:
                            if abs(val - level) <= tolerance:
                                found = True
                                break
                        if found:
                            utilization_count += 1
                    
                    range_utilization = utilization_count / len(utilization_levels)
                    
                    joint_mobility = (basic_mobility + acceleration_smoothness + range_utilization) / 3.0
                else:
                    joint_mobility = basic_mobility
            
            # 5. パフォーマンスの一貫性（安定性）
            performance_consistency = 0.0
            if len(returns) > 0:
                std_returns = np.std(returns)
                mean_abs_returns = np.mean(np.abs(returns))
                
                if mean_abs_returns > 1e-10:
                    coefficient_of_variation = std_returns / mean_abs_returns
                    basic_consistency = 1.0 / (1.0 + coefficient_of_variation)
                else:
                    basic_consistency = 1.0
                
                trend_consistency = 0.0
                if len(returns) >= 3:
                    window_ma_size = min(5, len(returns) // 2)
                    if window_ma_size >= 2:
                        moving_avg = []
                        for j in range(window_ma_size - 1, len(returns)):
                            ma_value = np.mean(returns[j - window_ma_size + 1:j + 1])
                            moving_avg.append(ma_value)
                        
                        if len(moving_avg) > 0:
                            ma_std = np.std(moving_avg)
                            trend_consistency = 1.0 / (1.0 + ma_std)
                
                performance_consistency = (basic_consistency + trend_consistency) / 2.0
            
            # 6. 持久力（上昇トレンドの持続性）
            endurance = 0.0
            if len(returns) > 0:
                positive_moves = 0
                for ret in returns:
                    if ret > 0:
                        positive_moves += 1
                
                total_moves = len(returns)
                basic_endurance = positive_moves / total_moves
                
                streak_lengths = []
                current_streak = 0
                current_direction = 0
                
                for j in range(len(returns)):
                    if returns[j] > 0:
                        if current_direction >= 0:
                            current_streak += 1
                        else:
                            if current_streak > 0:
                                streak_lengths.append(current_streak)
                            current_streak = 1
                        current_direction = 1
                    elif returns[j] < 0:
                        if current_direction <= 0:
                            current_streak += 1
                        else:
                            if current_streak > 0:
                                streak_lengths.append(current_streak)
                            current_streak = 1
                        current_direction = -1
                
                if current_streak > 0:
                    streak_lengths.append(current_streak)
                
                if len(streak_lengths) > 0:
                    max_streak = np.max(streak_lengths)
                    streak_endurance = max_streak / len(returns)
                    
                    endurance = (basic_endurance + streak_endurance) / 2.0
                else:
                    endurance = basic_endurance
            
            # 7. 回復率（最大ドローダウンからの回復）
            recovery_rate = 0.0
            if len(returns) > 0:
                cumulative_returns = np.zeros(len(returns) + 1)
                for j in range(len(returns)):
                    cumulative_returns[j + 1] = cumulative_returns[j] + returns[j]
                
                running_max = np.zeros(len(cumulative_returns))
                drawdowns = np.zeros(len(cumulative_returns))
                
                running_max[0] = cumulative_returns[0]
                for j in range(1, len(cumulative_returns)):
                    running_max[j] = max(running_max[j-1], cumulative_returns[j])
                    drawdowns[j] = running_max[j] - cumulative_returns[j]
                
                max_drawdown = np.max(drawdowns)
                
                if max_drawdown > 1e-10:
                    final_return = cumulative_returns[-1]
                    basic_recovery = final_return / max_drawdown
                    
                    recovery_periods = []
                    in_drawdown = False
                    drawdown_start = 0
                    
                    for j in range(len(drawdowns)):
                        if drawdowns[j] > 1e-10 and not in_drawdown:
                            in_drawdown = True
                            drawdown_start = j
                        elif drawdowns[j] <= 1e-10 and in_drawdown:
                            in_drawdown = False
                            recovery_period = j - drawdown_start
                            if recovery_period > 0:
                                recovery_periods.append(recovery_period)
                    
                    if len(recovery_periods) > 0:
                        avg_recovery_time = np.mean(recovery_periods)
                        recovery_speed = 1.0 / (1.0 + avg_recovery_time / len(returns))
                        recovery_rate = (basic_recovery + recovery_speed) / 2.0
                    else:
                        recovery_rate = basic_recovery
                else:
                    recovery_rate = 1.0
            
            # 結果の格納
            results[i, 0] = kinetic_energy
            results[i, 1] = potential_energy
            results[i, 2] = muscle_force
            results[i, 3] = joint_mobility
            results[i, 4] = performance_consistency
            results[i, 5] = endurance
            results[i, 6] = recovery_rate
        
        return results
    
    # =========================================================================
    # ユーティリティメソッド（Polars最適化共通関数）
    # =========================================================================
    
    @staticmethod
    @njit(cache=True)
    def _compute_rolling_percentile_numba(data: np.ndarray, window: int, percentile: float) -> np.ndarray:
        """ローリングパーセンタイル計算（Numba最適化版）"""
        n = len(data)
        result = np.zeros(n)
        
        for i in range(window - 1, n):
            window_data = data[i - window + 1:i + 1]
            sorted_data = np.sort(window_data)
            idx = int(percentile / 100.0 * (len(sorted_data) - 1))
            result[i] = sorted_data[idx]
        
        return result
    
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
            # Block 12: 学際的アナロジー特徴量
            # =========================================================================
            logger.info("学際的アナロジー特徴量計算中...")
            interdisciplinary_start = time.time()
            
            interdisciplinary_features = self.calculate_interdisciplinary_features(close)
            all_features.update(interdisciplinary_features)
            
            logger.info(f"学際的アナロジー特徴量完了: {len(interdisciplinary_features)}個 ({time.time() - interdisciplinary_start:.2f}秒)")
            
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
        """計算エンジン検証 (Script C: 情報理論・学際的特徴量)"""
        logger.info("計算エンジンテスト実行中...")
        
        try:
            # ダミーデータ生成
            np.random.seed(42)
            test_size = 250
            test_data = np.random.randn(test_size) * 0.01 + 100

            # Calculator初期化
            dummy_window_manager = WindowManager(window_size=50, overlap=0.3)
            dummy_memory_manager = MemoryManager()
            calculator = Calculator(dummy_window_manager, dummy_memory_manager)
            
            # Cのスクリプトに存在する計算機能をテスト
            network_features = calculator.calculate_network_science_features(test_data)
            aesthetics_features = calculator.calculate_aesthetics_features(test_data)
            
            # 計算結果検証
            all_features = {}
            all_features.update(network_features)
            all_features.update(aesthetics_features)
            
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