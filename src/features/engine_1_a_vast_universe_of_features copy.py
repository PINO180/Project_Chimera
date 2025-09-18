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
# Calculator クラス - 核心アルゴリズム実装（80%リソース集中）
# =============================================================================
class Calculator:
    def __init__(self, window_manager: WindowManager, memory_manager: MemoryManager):
        self.window_manager = window_manager
        self.memory_manager = memory_manager
        
        # 計算パラメータ
        self.params = CALC_PARAMS.copy()
        
        # 検証統計
        self.calculation_stats = {
            'total_calculations': 0,
            'successful_calculations': 0,
            'nan_count': 0,
            'inf_count': 0,
            'out_of_range_count': 0,
            'computation_times': []
        }
        
        # 数値安定性パラメータ
        self.numerical_stability = {
            'eps': 1e-12,
            'condition_number_threshold': 1e12,
            'outlier_threshold': 5.0,  # 標準偏差の倍数
            'min_valid_ratio': 0.7     # 70%以上の有効値が必要
        }
        
        # Golden ratioなどの数学的定数
        self.mathematical_constants = {
            'golden_ratio': (1 + np.sqrt(5)) / 2,  # φ ≈ 1.618
            'euler_constant': np.e,                  # e ≈ 2.718
            'pi': np.pi,                            # π ≈ 3.14159
            'sqrt_2': np.sqrt(2),                   # √2 ≈ 1.414
            'sqrt_3': np.sqrt(3),                   # √3 ≈ 1.732
        }
        
        # 理論的根拠に基づくパラメータ設定
        self.theoretical_parameters = self._initialize_theoretical_parameters()
        
        # 品質保証システム  
        self.quality_system = {
            'feature_quality_scores': {},
            'theoretical_range_violations': {},
            'statistical_significance_cache': {},
            'numerical_stability_metrics': {}
        }
        
        logger.info("Calculator初期化完了 - 独自アルゴリズム準備完了")

    def _initialize_theoretical_parameters(self) -> Dict[str, Any]:
        """理論的根拠に基づくパラメータ初期化"""
        φ = self.mathematical_constants['golden_ratio']
        
        return {
            # フィボナッチ系列（Golden ratioベース）
            'fibonacci_periods': [int(φ**i) for i in range(1, 8)],  # [1, 2, 3, 5, 8, 13, 21]
            
            # ウィンドウサイズ（Golden ratioの倍数）
            'golden_windows': [int(50 * φ**i) for i in range(0, 5)],  # [50, 81, 131, 212, 343]
            
            # フーリエ解析用（2の冪）
            'fft_windows': [2**i for i in range(4, 10)],  # [16, 32, 64, 128, 256, 512]
            
            # 統計的信頼区間
            'confidence_levels': [0.90, 0.95, 0.99, 0.999],
            
            # ボラティリティ帯域（標準偏差の倍数）
            'volatility_bands': [φ/2, φ, φ*1.5, φ*2],  # Golden ratio基準
            
            # カオス理論用埋め込み次元
            'embedding_dimensions': [2, 3, 5, 8],  # フィボナッチ数列
            
            # 正則化パラメータ（機械学習理論）
            'regularization_alphas': [1e-6, 1e-4, 1e-2, 1e-1]
        }
        
    # =========================================================================
    # 核心アルゴリズム: 数値安定性とロバスト推定
    # =========================================================================

    def robust_calculation_wrapper(self, func, data, *args, **kwargs) -> Union[float, np.ndarray]:
        """
        ロバスト計算ラッパー (データ長チェック機能付き)
        数値安定性とエラーハンドリングを統合
        """
        start_time = time.time()
        self.calculation_stats['total_calculations'] += 1

        try:
            # ▼▼▼ データ長チェックをここに集約 ▼▼▼
            n = len(data)
            period = -1
            
            # args (位置引数) から period/window_size を探す
            if args:
                for arg in args:
                    if isinstance(arg, int) and arg > 1:
                        period = arg
                        break
            # kwargs (キーワード引数) から period/window_size を探す
            if period == -1:
                for key in ['window_size', 'period']:
                    if key in kwargs and isinstance(kwargs[key], int):
                        period = kwargs[key]
                        break

            # periodが見つかり、データ長が不十分な場合は計算をスキップ
            if period != -1 and n < period * 1.5:
                logger.debug(f"データ長不足のため {func.__name__} (period={period}, data_len={n}) をスキップ")
                # 戻り値の型に合わせてフォールバック値を返す
                # Numba関数は特定の型の配列を返すことが多い
                if func.__name__.startswith('_compute_'):
                    # Numbaヘルパー関数の戻り値の形状を推測
                    # この例では、多くが入力と同じ長さの1D配列か、特定の形状の2D配列を返す
                    # 簡単のため、入力と同じ長さのNaN配列を返す
                    return np.full(n, np.nan)
                else:
                    return self._fallback_value(data) # floatを返す
            # ▲▲▲ データ長チェックここまで ▲▲▲

            cleaned_data = self._preprocess_data(data)
            
            if hasattr(cleaned_data, 'shape') and len(cleaned_data.shape) > 1:
                condition_num = np.linalg.cond(cleaned_data)
                if condition_num > self.numerical_stability['condition_number_threshold']:
                    logger.debug(f"高い条件数検出: {condition_num:.2e}")
                    cleaned_data = self._regularize_matrix(cleaned_data)
            
            result = func(cleaned_data, *args, **kwargs)
            
            validated_result = self._validate_result(result)
            
            self.calculation_stats['successful_calculations'] += 1
            computation_time = time.time() - start_time
            self.calculation_stats['computation_times'].append(computation_time)
            
            return validated_result
            
        except Exception as e:
            logger.debug(f"計算エラー in {func.__name__}: {e}")
            return self._fallback_value(data)

    def _preprocess_data(self, data: np.ndarray) -> np.ndarray:
        """データ前処理 - NaN/Inf除去とアウトライア検出"""
        if not isinstance(data, np.ndarray):
            data = np.asarray(data, dtype=np.float64)
        
        # NaN/Inf チェック
        nan_mask = np.isnan(data)
        inf_mask = np.isinf(data)
        
        self.calculation_stats['nan_count'] += np.sum(nan_mask)
        self.calculation_stats['inf_count'] += np.sum(inf_mask)
        
        # 有効データ率チェック
        valid_mask = ~(nan_mask | inf_mask)
        valid_ratio = np.sum(valid_mask) / len(data.flat)
        
        if valid_ratio < self.numerical_stability['min_valid_ratio']:
            logger.warning(f"有効データ率低下: {valid_ratio:.2%}")
        
        # アウトライア検出と処理
        if valid_ratio > 0.5:
            valid_data = data[valid_mask]
            if len(valid_data) > 10:
                outlier_mask = self._detect_outliers(valid_data)
                if np.sum(outlier_mask) > 0:
                    self.calculation_stats['out_of_range_count'] += np.sum(outlier_mask)
                    # アウトライアをメディアンで置換
                    median_val = np.median(valid_data[~outlier_mask])
                    data[outlier_mask] = median_val
        
        # 残りのNaN/Infを安全な値で置換
        data[nan_mask | inf_mask] = np.nanmedian(data[valid_mask]) if np.sum(valid_mask) > 0 else 0.0
        
        return data

    @staticmethod
    @njit(cache=True)
    def _detect_outliers_numba(data: np.ndarray, threshold: float) -> np.ndarray:
        """Numba最適化アウトライア検出"""
        if len(data) < 3:
            return np.zeros(len(data), dtype=numba.boolean)
        
        median = np.median(data)
        mad = np.median(np.abs(data - median))
        
        if mad < 1e-10:  # MADがほぼゼロの場合
            return np.zeros(len(data), dtype=numba.boolean)
        
        modified_z_scores = 0.6745 * (data - median) / mad
        return np.abs(modified_z_scores) > threshold

    def _detect_outliers(self, data: np.ndarray) -> np.ndarray:
        """Modified Z-score によるアウトライア検出"""
        return self._detect_outliers_numba(data, self.numerical_stability['outlier_threshold'])

    def _regularize_matrix(self, matrix: np.ndarray, alpha: float = 1e-6) -> np.ndarray:
        """行列正則化 - 数値安定性向上"""
        if len(matrix.shape) != 2:
            return matrix
        
        # Ridge正則化
        regularized = matrix + alpha * np.eye(min(matrix.shape))
        return regularized

    def _validate_result(self, result) -> Union[float, np.ndarray]:
        """計算結果の妥当性検証"""
        if isinstance(result, (int, float)):
            if np.isnan(result) or np.isinf(result):
                return 0.0
            # 異常に大きな値のクランプ
            if abs(result) > 1e10:
                return np.sign(result) * 1e10
            return float(result)
        
        elif isinstance(result, np.ndarray):
            result = np.asarray(result, dtype=np.float64)
            result[np.isnan(result) | np.isinf(result)] = 0.0
            # 異常値のクランプ
            result = np.clip(result, -1e10, 1e10)
            return result
        
        else:
            return 0.0

    def _fallback_value(self, data) -> float:
        """計算失敗時のフォールバック値"""
        try:
            if hasattr(data, '__len__') and len(data) > 0:
                return float(np.nanmedian(np.asarray(data).flatten()))
            else:
                return 0.0
        except:
            return 0.0

    # =========================================================================
    # 基礎統計・テクニカル指標計算（Numba最適化）
    # =========================================================================

    @staticmethod
    @njit(cache=True)
    def _compute_rsi_numba(prices: np.ndarray, period: int) -> np.ndarray:
        n = len(prices)
        if n < period + 1:
            return np.zeros(n)
        
        # 配列を連続メモリにコピーしてからdiff計算
        prices_copy = np.ascontiguousarray(prices)
        deltas = np.diff(prices_copy)
        gains = np.where(deltas > 0, deltas, 0.0)
        losses = np.where(deltas < 0, -deltas, 0.0)
        
        # 初期平均
        avg_gain = np.mean(gains[:period])
        avg_loss = np.mean(losses[:period])
        
        rsi = np.zeros(n)
        
        # EMA計算
        alpha = 1.0 / period
        for i in range(period, n):
            gain = gains[i-1]
            loss = losses[i-1]
            
            avg_gain = alpha * gain + (1 - alpha) * avg_gain
            avg_loss = alpha * loss + (1 - alpha) * avg_loss
            
            if avg_loss < 1e-10:
                rsi[i] = 100.0 if avg_gain > 0 else 50.0
            else:
                rs = avg_gain / avg_loss
                rsi[i] = 100.0 - (100.0 / (1.0 + rs))
        
        return rsi

    @staticmethod
    @njit(cache=True)
    def _compute_macd_numba(prices: np.ndarray, fast: int, slow: int, signal: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """MACD計算 - Numba最適化版"""
        n = len(prices)
        if n < slow:
            return np.zeros(n), np.zeros(n), np.zeros(n)
        
        # EMA計算
        alpha_fast = 2.0 / (fast + 1)
        alpha_slow = 2.0 / (slow + 1)
        alpha_signal = 2.0 / (signal + 1)
        
        ema_fast = np.zeros(n)
        ema_slow = np.zeros(n)
        macd = np.zeros(n)
        signal_line = np.zeros(n)
        histogram = np.zeros(n)
        
        # 初期値
        ema_fast[0] = prices[0]
        ema_slow[0] = prices[0]
        
        for i in range(1, n):
            ema_fast[i] = alpha_fast * prices[i] + (1 - alpha_fast) * ema_fast[i-1]
            ema_slow[i] = alpha_slow * prices[i] + (1 - alpha_slow) * ema_slow[i-1]
            macd[i] = ema_fast[i] - ema_slow[i]
            
            if i == 1:
                signal_line[i] = macd[i]
            else:
                signal_line[i] = alpha_signal * macd[i] + (1 - alpha_signal) * signal_line[i-1]
            
            histogram[i] = macd[i] - signal_line[i]
        
        return macd, signal_line, histogram

    @staticmethod
    @njit(cache=True)
    def _compute_bollinger_bands_numba(prices: np.ndarray, period: int, std_multiplier: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """ボリンジャーバンド計算 - Numba最適化版"""
        n = len(prices)
        if n < period:
            return prices.copy(), prices.copy(), prices.copy()
        
        sma = np.zeros(n)
        std = np.zeros(n)
        upper = np.zeros(n)
        lower = np.zeros(n)
        
        for i in range(period-1, n):
            window = prices[i-period+1:i+1]
            sma[i] = np.mean(window)
            std[i] = np.std(window)
            upper[i] = sma[i] + std_multiplier * std[i]
            lower[i] = sma[i] - std_multiplier * std[i]
        
        # 初期値の埋め戻し
        for i in range(period-1):
            sma[i] = prices[i]
            upper[i] = prices[i]
            lower[i] = prices[i]
        
        return upper, sma, lower

    @staticmethod
    @njit(cache=True)
    def _compute_atr_numba(high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int) -> np.ndarray:
        """ATR計算 - Numba最適化版"""
        n = len(high)
        if n < 2:
            return np.zeros(n)
        
        true_range = np.zeros(n)
        atr = np.zeros(n)
        
        # True Range 計算
        for i in range(1, n):
            hl = high[i] - low[i]
            hc = abs(high[i] - close[i-1])
            lc = abs(low[i] - close[i-1])
            true_range[i] = max(hl, max(hc, lc))
        
        true_range[0] = high[0] - low[0]
        
        # ATR 計算（EMA）
        alpha = 1.0 / period
        atr[0] = true_range[0]
        
        for i in range(1, n):
            atr[i] = alpha * true_range[i] + (1 - alpha) * atr[i-1]
        
        return atr

    @staticmethod
    @njit(cache=True) 
    def _compute_stochastic_numba(high: np.ndarray, low: np.ndarray, close: np.ndarray, 
                                    k_period: int, d_period: int) -> Tuple[np.ndarray, np.ndarray]:
        """Stochastic Oscillator計算 - Numba最適化版"""
        n = len(high)
        if n < k_period:
            return np.zeros(n), np.zeros(n)
        
        k_values = np.zeros(n)
        d_values = np.zeros(n)
        
        for i in range(k_period-1, n):
            window_high = high[i-k_period+1:i+1]
            window_low = low[i-k_period+1:i+1]
            
            highest_high = np.max(window_high)
            lowest_low = np.min(window_low)
            
            if highest_high - lowest_low > 1e-10:
                k_values[i] = 100.0 * (close[i] - lowest_low) / (highest_high - lowest_low)
            else:
                k_values[i] = 50.0
        
        # %D計算（%Kのd_period移動平均）
        for i in range(k_period + d_period - 2, n):
            d_values[i] = np.mean(k_values[i-d_period+1:i+1])
        
        return k_values, d_values

    # =========================================================================
    # 高度な統計的特徴量（科学的検証付き）
    # =========================================================================

    def calculate_statistical_moments(self, data: np.ndarray, window_size: int) -> Dict[str, np.ndarray]:
        """統計的モーメント計算（高次モーメントまで含む科学的検証付き）"""
        results = {}
        
        def calculate_moments_window(window_data):
            return self.robust_calculation_wrapper(self._calculate_moments_core, window_data)
        
        # ローリングウィンドウで計算
        n = len(data)
        moment_results = np.zeros((n, 8))  # 1-8次モーメント
        
        for i in range(window_size-1, n):
            window = data[i-window_size+1:i+1]
            moments = calculate_moments_window(window)
            if isinstance(moments, np.ndarray) and len(moments) >= 8:
                moment_results[i] = moments
            else:
                moment_results[i] = np.zeros(8)
        
        # 結果を辞書に格納
        moment_names = ['mean', 'variance', 'skewness', 'kurtosis', 'moment_5', 'moment_6', 'moment_7', 'moment_8']
        for j, name in enumerate(moment_names):
            results[f'statistical_{name}_{window_size}'] = moment_results[:, j]
        
        return results

    @staticmethod
    @njit(cache=True)
    def _calculate_moments_core(data: np.ndarray) -> np.ndarray:
        """統計的モーメント計算のコア部分 - Numba最適化"""
        if len(data) < 2:
            return np.zeros(8)
        
        n = len(data)
        mean = np.mean(data)
        
        # 中心化データ
        centered = data - mean
        
        # 各次モーメント計算
        moments = np.zeros(8)
        moments[0] = mean  # 1次モーメント
        moments[1] = np.mean(centered**2)  # 分散
        
        if moments[1] > 1e-10:  # ゼロ除算回避
            std_dev = np.sqrt(moments[1])
            standardized = centered / std_dev
            
            moments[2] = np.mean(standardized**3)  # 歪度
            moments[3] = np.mean(standardized**4) - 3.0  # 超過尖度
            moments[4] = np.mean(standardized**5)  # 5次モーメント
            moments[5] = np.mean(standardized**6)  # 6次モーメント
            moments[6] = np.mean(standardized**7)  # 7次モーメント  
            moments[7] = np.mean(standardized**8)  # 8次モーメント
        
        return moments

    def calculate_robust_statistics(self, data: np.ndarray, window_size: int) -> Dict[str, np.ndarray]:
        """ロバスト統計量計算"""
        results = {}
        n = len(data)
        
        # 各統計量の配列を初期化
        median_vals = np.zeros(n)
        mad_vals = np.zeros(n)  # Median Absolute Deviation
        iqr_vals = np.zeros(n)  # Interquartile Range
        trimmed_mean_vals = np.zeros(n)
        winsorized_mean_vals = np.zeros(n)
        
        for i in range(window_size-1, n):
            window = data[i-window_size+1:i+1]
            
            # ロバスト統計量計算
            robust_stats = self.robust_calculation_wrapper(self._calculate_robust_stats_core, window)
            
            if isinstance(robust_stats, np.ndarray) and len(robust_stats) >= 5:
                median_vals[i] = robust_stats[0]
                mad_vals[i] = robust_stats[1] 
                iqr_vals[i] = robust_stats[2]
                trimmed_mean_vals[i] = robust_stats[3]
                winsorized_mean_vals[i] = robust_stats[4]
            else:
                # フォールバック値
                median_vals[i] = 0.0
                mad_vals[i] = 0.0
                iqr_vals[i] = 0.0
                trimmed_mean_vals[i] = 0.0
                winsorized_mean_vals[i] = 0.0
        
        results.update({
            f'robust_median_{window_size}': median_vals,
            f'robust_mad_{window_size}': mad_vals,
            f'robust_iqr_{window_size}': iqr_vals,
            f'robust_trimmed_mean_{window_size}': trimmed_mean_vals,
            f'robust_winsorized_mean_{window_size}': winsorized_mean_vals
        })
        
        return results

    @staticmethod
    @njit(cache=True)
    def _calculate_robust_stats_core(data: np.ndarray) -> np.ndarray:
        """ロバスト統計量のコア計算 - Numba最適化"""
        if len(data) < 3:
            return np.zeros(5)
        
        sorted_data = np.sort(data)
        n = len(sorted_data)
        
        # メディアン
        median = np.median(sorted_data)
        
        # MAD (Median Absolute Deviation)
        mad = np.median(np.abs(sorted_data - median))
        
        # IQR (Interquartile Range)
        q25_idx = int(n * 0.25)
        q75_idx = int(n * 0.75)
        q25 = sorted_data[q25_idx]
        q75 = sorted_data[q75_idx]
        iqr = q75 - q25
        
        # Trimmed Mean (10%トリミング)
        trim_count = max(1, int(n * 0.1))
        trimmed_data = sorted_data[trim_count:-trim_count] if n > 2 * trim_count else sorted_data
        trimmed_mean = np.mean(trimmed_data)
        
        # Winsorized Mean (5%ウィンソライゼーション)
        wins_count = max(1, int(n * 0.05))
        winsorized_data = sorted_data.copy()
        if n > 2 * wins_count:
            winsorized_data[:wins_count] = sorted_data[wins_count]
            winsorized_data[-wins_count:] = sorted_data[-wins_count-1]
        winsorized_mean = np.mean(winsorized_data)
        
        return np.array([median, mad, iqr, trimmed_mean, winsorized_mean])

    # =========================================================================
    # 信号処理・フーリエ解析（高度な周波数解析）
    # =========================================================================

    def calculate_spectral_features(self, data: np.ndarray, window_size: int) -> Dict[str, np.ndarray]:
        """スペクトル特徴量計算"""
        results = {}
        n = len(data)
        
        spectral_features = np.zeros((n, 6))  # 6つのスペクトル特徴量
        
        for i in range(window_size-1, n):
            window = data[i-window_size+1:i+1]
            features = self.robust_calculation_wrapper(self._calculate_spectral_core, window)
            if isinstance(features, np.ndarray) and len(features) >= 6:
                spectral_features[i] = features
            else:
                spectral_features[i] = np.zeros(6)
        
        feature_names = ['spectral_centroid', 'spectral_bandwidth', 'spectral_rolloff', 
                        'spectral_flux', 'spectral_flatness', 'spectral_entropy']
        
        for j, name in enumerate(feature_names):
            results[f'{name}_{window_size}'] = spectral_features[:, j]
        
        return results

    @staticmethod
    @njit(cache=True) 
    def _calculate_spectral_core(data: np.ndarray) -> np.ndarray:
        """スペクトル特徴量のコア計算 - Numba最適化"""
        if len(data) < 8:
            return np.zeros(6)
        
        # FFT計算（Numba対応）
        n = len(data)
        fft_data = np.fft.fft(data)
        power_spectrum = np.abs(fft_data[:n//2])**2
        freqs = np.arange(n//2) / n
        
        if np.sum(power_spectrum) < 1e-10:
            return np.zeros(6)
        
        # 正規化
        power_spectrum = power_spectrum / np.sum(power_spectrum)
        
        features = np.zeros(6)
        
        # Spectral Centroid（重心周波数）
        features[0] = np.sum(freqs * power_spectrum)
        
        # Spectral Bandwidth（帯域幅）
        centroid = features[0]
        features[1] = np.sqrt(np.sum(((freqs - centroid)**2) * power_spectrum))
        
        # Spectral Rolloff（85%エネルギー点）
        cumsum = np.cumsum(power_spectrum)
        rolloff_idx = np.where(cumsum >= 0.85 * cumsum[-1])[0]
        features[2] = freqs[rolloff_idx[0]] if len(rolloff_idx) > 0 else 0.0
        
        # Spectral Flux（前フレームとの差）
        if n >= 16:
            prev_spectrum = np.abs(np.fft.fft(data[:-n//2])[:n//4])**2
            curr_spectrum = power_spectrum[:len(prev_spectrum)]
            features[3] = np.sum((curr_spectrum - prev_spectrum)**2)
        
        # Spectral Flatness（平坦度）
        if np.all(power_spectrum > 1e-10):
            geo_mean = np.exp(np.mean(np.log(power_spectrum + 1e-10)))
            arith_mean = np.mean(power_spectrum)
            features[4] = geo_mean / (arith_mean + 1e-10)
        
        # Spectral Entropy（エントロピー）
        features[5] = -np.sum(power_spectrum * np.log2(power_spectrum + 1e-10))
        
        return features

    def calculate_wavelet_features(self, data: np.ndarray, wavelet_name: str = 'db4') -> Dict[str, np.ndarray]:
        """
        ウェーブレット特徴量計算（安全なローリング処理版）
        """
        features = {}
        n = len(data)
        window_size = 128 # ウェーブレットに適したウィンドウサイズ
        level = 5 # 分解レベル
        
        # エネルギー, エントロピー, 平均, 標準偏差 の4特徴量 x (level+1)係数
        num_features = 4 * (level + 1)
        wavelet_results = np.zeros((n, num_features))

        for i in range(window_size - 1, n):
            window = data[i - window_size + 1 : i + 1]
            try:
                coeffs = pywt.wavedec(window, wavelet_name, level=level)
                
                for j, coeff in enumerate(coeffs):
                    energy = np.sum(coeff**2)
                    
                    if energy > 1e-10:
                        prob = coeff**2 / energy
                        prob = prob[prob > 1e-10]
                        entropy = -np.sum(prob * np.log2(prob))
                    else:
                        entropy = 0

                    wavelet_results[i, j*4 + 0] = energy
                    wavelet_results[i, j*4 + 1] = entropy
                    wavelet_results[i, j*4 + 2] = np.mean(coeff)
                    wavelet_results[i, j*4 + 3] = np.std(coeff)

            except Exception as e:
                logger.debug(f"Waveletウィンドウ計算エラー (index={i}): {e}")
                continue

        # 辞書に格納
        for j in range(level + 1):
            level_name = 'approx' if j == 0 else f'detail_{j}'
            features[f'wavelet_{level_name}_energy'] = wavelet_results[:, j*4 + 0]
            features[f'wavelet_{level_name}_entropy'] = wavelet_results[:, j*4 + 1]
            features[f'wavelet_{level_name}_mean'] = wavelet_results[:, j*4 + 2]
            features[f'wavelet_{level_name}_std'] = wavelet_results[:, j*4 + 3]

        return features

    # =========================================================================
    # カオス理論・フラクタル解析
    # =========================================================================

    def calculate_chaos_features(self, data: np.ndarray, window_size: int) -> Dict[str, np.ndarray]:
        """カオス理論特徴量計算"""
        results = {}
        n = len(data)
        
        # リアプノフ指数
        lyapunov_values = np.zeros(n)
        # 相関次元
        correlation_dim_values = np.zeros(n) 
        # カオス度
        chaos_degree_values = np.zeros(n)
        
        for i in range(window_size-1, n):
            window = data[i-window_size+1:i+1]
            
            chaos_features = self.robust_calculation_wrapper(self._calculate_chaos_core, window)
            
            if isinstance(chaos_features, np.ndarray) and len(chaos_features) >= 3:
                lyapunov_values[i] = chaos_features[0]
                correlation_dim_values[i] = chaos_features[1] 
                chaos_degree_values[i] = chaos_features[2]
            else:
                # フォールバック値
                lyapunov_values[i] = 0.0
                correlation_dim_values[i] = 0.0
                chaos_degree_values[i] = 0.0
        
        results.update({
            f'lyapunov_exponent_{window_size}': lyapunov_values,
            f'correlation_dimension_{window_size}': correlation_dim_values,
            f'chaos_degree_{window_size}': chaos_degree_values
        })
        
        return results

    @staticmethod
    @njit(cache=True)
    def _calculate_chaos_core(data: np.ndarray) -> np.ndarray:
        """カオス特徴量のコア計算 - Numba最適化"""
        if len(data) < 10:
            return np.zeros(3)
        
        features = np.zeros(3)
        
        # リアプノフ指数の近似計算
        n = len(data)
        total_log_div = 0.0
        count = 0
        
        for i in range(n - 5):
            for j in range(i + 1, min(i + 6, n - 5)):
                initial_dist = abs(data[i] - data[j])
                final_dist = abs(data[i + 5] - data[j + 5])
                if initial_dist > 1e-8:
                    total_log_div += np.log(final_dist / initial_dist)
                    count += 1
        
        features[0] = (total_log_div / count) / 5 if count > 0 else 0.0
        
        # 相関次元の簡易計算
        embedding_dim = min(3, len(data) // 3)
        if embedding_dim >= 2:
            # 位相空間再構成
            points = np.zeros((n - embedding_dim, embedding_dim))
            for i in range(n - embedding_dim):
                for j in range(embedding_dim):
                    points[i, j] = data[i + j]
            
            # 相関積分の計算（簡易版）
            distances = np.zeros(len(points) * (len(points) - 1) // 2)
            idx = 0
            for i in range(len(points)):
                for j in range(i + 1, len(points)):
                    dist = 0.0
                    for k in range(embedding_dim):
                        dist += (points[i, k] - points[j, k])**2
                    distances[idx] = np.sqrt(dist)
                    idx += 1
            
            if len(distances) > 0:
                std_dist = np.std(distances)
                if std_dist > 1e-10:
                    radius = std_dist * 0.1
                    correlation = np.sum(distances < radius) / len(distances)
                    if correlation > 1e-10:
                        features[1] = np.log(correlation) / np.log(radius)
                    else:
                        features[1] = 0.0
        
        # カオス度（標準偏差 vs 平均の比）
        mean_abs = abs(np.mean(data))
        std_dev = np.std(data)
        if mean_abs > 1e-8:
            features[2] = std_dev / mean_abs
        else:
            features[2] = 1.0 if std_dev > 0 else 0.0
        
        return features

    # =========================================================================
    # 計算品質スコア・統計的妥当性チェック
    # =========================================================================

    def calculate_quality_score(self, feature_values: np.ndarray, 
                                theoretical_range: Tuple[float, float] = None) -> float:
        """特徴量の計算品質スコア（0-1）"""
        if len(feature_values) == 0:
            return 0.0
        
        score = 1.0
        
        # NaN率チェック
        nan_ratio = np.isnan(feature_values).sum() / len(feature_values)
        score *= (1.0 - min(nan_ratio, 0.5) * 2)  # 50%以上のNaNで0点
        
        # Inf値チェック
        if np.isinf(feature_values).any():
            score *= 0.5
        
        # 理論的範囲チェック
        if theoretical_range is not None:
            min_val, max_val = theoretical_range
            out_of_range = np.sum((feature_values < min_val) | (feature_values > max_val))
            out_of_range_ratio = out_of_range / len(feature_values)
            score *= (1.0 - min(out_of_range_ratio, 0.3) / 0.3 * 0.5)
        
        # 数値の多様性チェック
        unique_ratio = len(np.unique(feature_values[~np.isnan(feature_values)])) / max(1, np.sum(~np.isnan(feature_values)))
        if unique_ratio < 0.1:  # 10%未満のユニーク値
            score *= 0.7
        
        return max(0.0, min(1.0, score))

    def validate_statistical_significance(self, feature_values: np.ndarray, 
                                        confidence_level: float = 0.95) -> Dict[str, Any]:
        """統計的有意性の即座チェック"""
        clean_values = feature_values[~np.isnan(feature_values) & ~np.isinf(feature_values)]
        
        if len(clean_values) < 10:
            return {'significant': False, 'reason': 'insufficient_data'}
        
        # 正規性テスト
        try:
            _, p_value_normality = stats.normaltest(clean_values)
            is_normal = p_value_normality > (1 - confidence_level)
        except:
            is_normal = False
        
        # ゼロからの有意差テスト
        try:
            if is_normal:
                _, p_value_ttest = stats.ttest_1samp(clean_values, 0)
            else:
                _, p_value_ttest = stats.wilcoxon(clean_values - 0, alternative='two-sided')
        except:
            p_value_ttest = 1.0
        
        is_significant = p_value_ttest < (1 - confidence_level)
        
        return {
            'significant': is_significant,
            'p_value': p_value_ttest,
            'is_normal': is_normal,
            'sample_size': len(clean_values),
            'effect_size': abs(np.mean(clean_values)) / (np.std(clean_values) + 1e-10)
        }

    def get_calculation_summary(self) -> Dict[str, Any]:
        """計算統計サマリー"""
        stats = self.calculation_stats
        
        success_rate = stats['successful_calculations'] / max(1, stats['total_calculations'])
        avg_computation_time = np.mean(stats['computation_times']) if stats['computation_times'] else 0.0
        
        return {
            'total_calculations': stats['total_calculations'],
            'success_rate': success_rate,
            'avg_computation_time_ms': avg_computation_time * 1000,
            'nan_count': stats['nan_count'],
            'inf_count': stats['inf_count'],
            'out_of_range_count': stats['out_of_range_count'],
            'numerical_stability_score': min(1.0, success_rate * (1 - stats['nan_count'] / max(1, stats['total_calculations'])))
        }

    # =========================================================================
    # Tier S特徴量: MFDFA (Multi-Fractal Detrended Fluctuation Analysis)
    # =========================================================================

    def calculate_mfdfa_features(self, data: np.ndarray, q_range: np.ndarray = None, 
                                scales: np.ndarray = None) -> Dict[str, np.ndarray]:
        """
        MFDFA (Multi-Fractal Detrended Fluctuation Analysis) 計算
        市場の長期記憶・フラクタル性を定量化する最優先特徴量
        """
        if q_range is None:
            q_range = np.arange(-5, 6, 1)  # -5 to 5
        
        if scales is None:
            scales = np.logspace(1, np.log10(len(data)//4), 20).astype(int)
            scales = np.unique(scales)
        
        results = {}
        n = len(data)
        
        # ローリングウィンドウでMFDFA計算
        window_size = 200  # MFDFA用ウィンドウサイズ
        mfdfa_results = np.zeros((n, len(q_range) + 3))  # q値ごとのHurst指数 + 3つの追加指標
        
        for i in range(window_size-1, n):
            window = data[i-window_size+1:i+1]
            mfdfa_output = self.robust_calculation_wrapper(
                self._calculate_mfdfa_core, window, q_range, scales
            )
            if isinstance(mfdfa_output, np.ndarray) and len(mfdfa_output) >= len(q_range) + 3:
                mfdfa_results[i] = mfdfa_output
            else:
                mfdfa_results[i] = np.zeros(len(q_range) + 3)
        
        # 結果を辞書に格納
        for j, q in enumerate(q_range):
            results[f'mfdfa_hurst_q{q}'] = mfdfa_results[:, j]
        
        results['mfdfa_multifractal_width'] = mfdfa_results[:, -3]
        results['mfdfa_asymmetry'] = mfdfa_results[:, -2] 
        results['mfdfa_complexity'] = mfdfa_results[:, -1]
        
        return results

    @staticmethod
    @njit(cache=True)
    def _calculate_mfdfa_core(data: np.ndarray, q_range: np.ndarray, 
                            scales: np.ndarray) -> np.ndarray:
        """MFDFA核心計算 - 完全Numba最適化版"""
        N = len(data)
        if N < 20:
            return np.zeros(len(q_range) + 3)
        
        # Step 1: Profile (累積和)
        Y = np.cumsum(data - np.mean(data))
        
        # Step 2: Detrended fluctuation function
        hurst_values = np.zeros(len(q_range))
        
        for qi, q in enumerate(q_range):
            fluctuations = np.zeros(len(scales))
            
            for si, s in enumerate(scales):
                if s < 4 or s >= N//2:
                    continue
                    
                # Divide profile into non-overlapping segments
                Ns = N // s
                
                # Detrend each segment
                F2_segments = np.zeros(Ns)
                
                for v in range(Ns):
                    start_idx = v * s
                    end_idx = (v + 1) * s
                    segment = Y[start_idx:end_idx]
                    
                    # Linear detrending
                    x = np.arange(s, dtype=np.float64)
                    mean_x = np.mean(x)
                    mean_y = np.mean(segment)
                    
                    # Linear regression coefficients
                    numerator = np.sum((x - mean_x) * (segment - mean_y))
                    denominator = np.sum((x - mean_x)**2)
                    
                    if denominator > 1e-10:
                        slope = numerator / denominator
                        intercept = mean_y - slope * mean_x
                        trend = slope * x + intercept
                        detrended = segment - trend
                    else:
                        detrended = segment - mean_y
                    
                    # Variance of detrended segment
                    F2_segments[v] = np.mean(detrended**2)
                
                # q-th order fluctuation function
                if q == 0:
                    # Special case for q=0 (log-average)
                    valid_segments = F2_segments[F2_segments > 1e-10]
                    if len(valid_segments) > 0:
                        fluctuations[si] = np.exp(0.5 * np.mean(np.log(valid_segments)))
                else:
                    if q == 2:
                        # Standard DFA
                        fluctuations[si] = np.sqrt(np.mean(F2_segments))
                    else:
                        # General case
                        positive_segments = F2_segments[F2_segments > 1e-10]
                        if len(positive_segments) > 0:
                            Fq_segments = np.power(positive_segments, q/2.0)
                            fluctuations[si] = np.power(np.mean(Fq_segments), 1.0/q)
            
            # Step 3: Scaling relationship F_q(s) ~ s^h(q)
            valid_mask = fluctuations > 1e-10
            if np.sum(valid_mask) >= 3:
                log_s = np.log(scales[valid_mask].astype(np.float64))
                log_Fq = np.log(fluctuations[valid_mask])
                
                # Linear regression
                mean_log_s = np.mean(log_s)
                mean_log_Fq = np.mean(log_Fq)
                
                numerator = np.sum((log_s - mean_log_s) * (log_Fq - mean_log_Fq))
                denominator = np.sum((log_s - mean_log_s)**2)
                
                if denominator > 1e-10:
                    hurst_values[qi] = numerator / denominator
                else:
                    hurst_values[qi] = 0.5  # Default Hurst value
            else:
                hurst_values[qi] = 0.5
        
        # Step 4: Multifractal spectrum analysis
        multifractal_width = np.max(hurst_values) - np.min(hurst_values)
        
        # Asymmetry of spectrum
        h_mean = np.mean(hurst_values)
        asymmetry = (np.max(hurst_values) - h_mean) - (h_mean - np.min(hurst_values))
        
        # Complexity measure
        complexity = np.std(hurst_values)
        
        # Combine results
        result = np.zeros(len(q_range) + 3)
        result[:len(q_range)] = hurst_values
        result[-3] = multifractal_width
        result[-2] = asymmetry  
        result[-1] = complexity
        
        return result

    # =========================================================================
    # Tier S特徴量: Microstructure Noise Ratio
    # =========================================================================

    def calculate_microstructure_noise_features(self, prices: np.ndarray, 
                                                window_size: int = 100) -> Dict[str, np.ndarray]:
        """
        Microstructure Noise Ratio 計算
        真の価格変動 vs ノイズを分離する重要特徴量
        """
        results = {}
        n = len(prices)
        
        noise_ratios = np.zeros(n)
        signal_strength = np.zeros(n)
        noise_persistence = np.zeros(n)
        
        for i in range(window_size-1, n):
            window_prices = prices[i-window_size+1:i+1]
            noise_features = self.robust_calculation_wrapper(
                self._calculate_microstructure_noise_core, window_prices
            )
            
            if isinstance(noise_features, np.ndarray) and len(noise_features) >= 3:
                noise_ratios[i] = noise_features[0]
                signal_strength[i] = noise_features[1]
                noise_persistence[i] = noise_features[2]
            else:
                noise_ratios[i] = 0.0
                signal_strength[i] = 0.0
                noise_persistence[i] = 0.0
        
        results.update({
            f'microstructure_noise_ratio_{window_size}': noise_ratios,
            f'signal_strength_{window_size}': signal_strength,
            f'noise_persistence_{window_size}': noise_persistence
        })
        
        return results

    @staticmethod
    @njit(cache=True)
    def _calculate_microstructure_noise_core(prices: np.ndarray) -> np.ndarray:
        """Microstructure Noise核心計算 - Numba最適化"""
        if len(prices) < 10:
            return np.zeros(3)
        
        # リターン計算
        returns = np.diff(prices)
        if len(returns) < 5:
            return np.zeros(3)
        
        # 1期リターンと2期リターンの分散比較（Roll 1984）
        returns_1 = returns[1:]  # r_t
        returns_2 = returns[:-1]  # r_{t-1}
        
        # 共分散
        if len(returns_1) > 0 and len(returns_2) > 0:
            covariance = np.mean((returns_1 - np.mean(returns_1)) * 
                                (returns_2 - np.mean(returns_2)))
        else:
            covariance = 0.0
        
        # ノイズ比率推定
        # Var(r_t) = Var(fundamental) + 2*Var(noise)
        # Cov(r_t, r_{t-1}) = -Var(noise)
        returns_variance = np.var(returns)
        
        if returns_variance > 1e-10:
            noise_variance_estimate = -covariance if covariance < 0 else 0.0
            noise_ratio = min(0.5, max(0.0, noise_variance_estimate / returns_variance))
        else:
            noise_ratio = 0.0
        
        # シグナル強度（フラクタル次元に基づく）
        # ハーストインデックス簡易推定
        if len(returns) >= 20:
            # R/S統計
            cumulative = np.cumsum(returns - np.mean(returns))
            R = np.max(cumulative) - np.min(cumulative)
            S = np.std(returns)
            
            if S > 1e-10:
                rs_ratio = R / S
                hurst_estimate = np.log(rs_ratio) / np.log(len(returns))
                signal_strength = abs(hurst_estimate - 0.5) * 2  # 0.5からの乖離度
            else:
                signal_strength = 0.0
        else:
            signal_strength = 0.0
        
        # ノイズ持続性（自己相関による）
        if len(returns) >= 10:
            # ラグ1自己相関
            returns_centered = returns - np.mean(returns)
            autocorr = np.sum(returns_centered[1:] * returns_centered[:-1]) / np.sum(returns_centered**2)
            noise_persistence = abs(autocorr)
        else:
            noise_persistence = 0.0
        
        return np.array([noise_ratio, signal_strength, noise_persistence])

    # =========================================================================
    # Tier 1特徴量: 軽量版ショックモデル
    # =========================================================================

    def calculate_shock_model_features(self, prices: np.ndarray, 
                                        window_size: int = 50) -> Dict[str, np.ndarray]:
        """
        軽量版ショックモデル特徴量
        期待からの乖離とショック検出
        """
        results = {}
        n = len(prices)
        
        shock_intensity = np.zeros(n)
        shock_frequency = np.zeros(n) 
        recovery_speed = np.zeros(n)
        expected_deviation = np.zeros(n)
        
        for i in range(window_size-1, n):
            window_prices = prices[i-window_size+1:i+1]
            shock_features = self.robust_calculation_wrapper(
                self._calculate_shock_model_core, window_prices
            )
            
            if isinstance(shock_features, np.ndarray) and len(shock_features) >= 4:
                shock_intensity[i] = shock_features[0]
                shock_frequency[i] = shock_features[1]
                recovery_speed[i] = shock_features[2] 
                expected_deviation[i] = shock_features[3]
            else:
                shock_intensity[i] = 0.0
                shock_frequency[i] = 0.0
                recovery_speed[i] = 0.0
                expected_deviation[i] = 0.0
        
        results.update({
            f'shock_intensity_{window_size}': shock_intensity,
            f'shock_frequency_{window_size}': shock_frequency,
            f'recovery_speed_{window_size}': recovery_speed,
            f'expected_deviation_{window_size}': expected_deviation
        })
        
        return results

    @staticmethod
    @njit(cache=True)
    def _calculate_shock_model_core(prices: np.ndarray) -> np.ndarray:
        """ショックモデル核心計算 - Numba最適化"""
        if len(prices) < 10:
            return np.zeros(4)
        
        returns = np.diff(prices)
        if len(returns) < 5:
            return np.zeros(4)
        
        # 期待リターン（移動平均）
        expected_return = np.mean(returns)
        return_std = np.std(returns)
        
        if return_std < 1e-10:
            return np.zeros(4)
        
        # ショック検出（2σ閾値）
        shock_threshold = 2.0 * return_std
        shock_events = np.abs(returns - expected_return) > shock_threshold
        
        # ショック強度（平均偏差）
        if np.sum(shock_events) > 0:
            shock_returns = returns[shock_events]
            shock_intensity = np.mean(np.abs(shock_returns - expected_return)) / return_std
        else:
            shock_intensity = 0.0
        
        # ショック頻度
        shock_frequency = np.sum(shock_events) / len(returns)
        
        # 回復速度（ショック後の反転度合い）
        recovery_speeds = []
        for i in range(len(shock_events) - 1):
            if shock_events[i]:  # ショック発生
                # 次の期間での反転を測定
                current_deviation = returns[i] - expected_return
                next_deviation = returns[i + 1] - expected_return
                
                if current_deviation != 0:
                    recovery = -next_deviation / current_deviation  # 反転度合い
                    recovery_speeds.append(recovery)
        
        if len(recovery_speeds) > 0:
            recovery_speed = np.mean(recovery_speeds)
        else:
            recovery_speed = 0.0
        
        # 期待からの乖離度（RMS）
        deviations = returns - expected_return
        expected_deviation = np.sqrt(np.mean(deviations**2)) / (np.abs(expected_return) + 1e-8)
        
        return np.array([shock_intensity, shock_frequency, recovery_speed, expected_deviation])

    # =========================================================================
    # Tier 1特徴量: Multi-Scale Volatility
    # =========================================================================

    def calculate_multiscale_volatility(self, prices: np.ndarray, 
                                        scales: List[int] = None) -> Dict[str, np.ndarray]:
        """
        Multi-Scale Volatility 計算
        複数時間軸でのボラティリティ特性分析
        """
        if scales is None:
            scales = [5, 10, 20, 50, 100]  # 異なるタイムスケール
        
        results = {}
        n = len(prices)
        
        for scale in scales:
            if scale >= n:
                continue
                
            volatility_values = np.zeros(n)
            
            for i in range(scale-1, n):
                window_prices = prices[i-scale+1:i+1]
                vol = self.robust_calculation_wrapper(
                    self._calculate_scale_volatility_core, window_prices, scale
                )
                volatility_values[i] = vol
            
            results[f'multiscale_volatility_{scale}'] = volatility_values
        
        # クロススケール相関
        if len(scales) >= 2:
            cross_correlations = np.zeros((n, len(scales)*(len(scales)-1)//2))
            
            for i in range(50, n):  # 十分なデータがある場合のみ
                corr_idx = 0
                for si, s1 in enumerate(scales[:-1]):
                    for s2 in scales[si+1:]:
                        if f'multiscale_volatility_{s1}' in results and f'multiscale_volatility_{s2}' in results:
                            vol1_window = results[f'multiscale_volatility_{s1}'][max(0,i-49):i+1]
                            vol2_window = results[f'multiscale_volatility_{s2}'][max(0,i-49):i+1]
                            
                            if len(vol1_window) > 10 and len(vol2_window) > 10:
                                corr = self._safe_correlation(vol1_window, vol2_window)
                                cross_correlations[i, corr_idx] = corr
                        
                        corr_idx += 1
            
            # クロススケール相関を結果に追加
            corr_idx = 0
            for si, s1 in enumerate(scales[:-1]):
                for s2 in scales[si+1:]:
                    results[f'cross_scale_corr_{s1}_{s2}'] = cross_correlations[:, corr_idx]
                    corr_idx += 1
        
        return results

    @staticmethod
    @njit(cache=True)
    def _calculate_scale_volatility_core(prices: np.ndarray, scale: int) -> float:
        """スケール別ボラティリティ核心計算"""
        if len(prices) < 2:
            return 0.0
        
        # 対数リターン
        log_prices = np.log(prices)
        returns = np.diff(log_prices)
        
        if len(returns) < 1:
            return 0.0
        
        # Realized Volatility (annualized)
        # 年間252取引日として正規化
        daily_vol = np.std(returns)
        annualized_vol = daily_vol * np.sqrt(252 / scale)
        
        return annualized_vol

    def _safe_correlation(self, x: np.ndarray, y: np.ndarray) -> float:
        """安全な相関係数計算"""
        try:
            if len(x) != len(y) or len(x) < 2:
                return 0.0
            
            # NaN/Inf除去
            valid_mask = np.isfinite(x) & np.isfinite(y)
            if np.sum(valid_mask) < 2:
                return 0.0
            
            x_clean = x[valid_mask]
            y_clean = y[valid_mask]
            
            if np.std(x_clean) < 1e-10 or np.std(y_clean) < 1e-10:
                return 0.0
            
            correlation = np.corrcoef(x_clean, y_clean)[0, 1]
            return correlation if np.isfinite(correlation) else 0.0
            
        except (ValueError, np.linalg.LinAlgError, RuntimeError, IndexError):
            return 0.0

    # =========================================================================
    # Tier 2特徴量: EMD/CEEMDAN (Empirical Mode Decomposition)
    # =========================================================================

    def calculate_emd_features(self, data: np.ndarray) -> Dict[str, np.ndarray]:
        """
        EMD特徴量計算（安全なローリング処理版）
        """
        features = {}
        try:
            from PyEMD import EMD
            emd_available = True
        except ImportError:
            emd_available = False
            logger.warning("PyEMDライブラリが利用できません - EMD特徴量計算をスキップします。")
            return features

        if not emd_available:
            return features

        n = len(data)
        window_size = 200 # EMDに適したウィンドウサイズ
        
        # 結果を格納する配列を初期化
        imf_count = 8 # 最大IMF数
        # エネルギー, 平均周波数, 振幅 の3特徴量 x 8IMF
        emd_results = np.zeros((n, imf_count * 3))

        for i in range(window_size - 1, n):
            window = data[i - window_size + 1 : i + 1]
            try:
                emd = EMD()
                imfs = emd.emd(window, max_imf=imf_count)

                for j in range(imf_count):
                    if j < len(imfs):
                        imf = imfs[j]
                        # エネルギー, 平均周波数, 振幅
                        emd_results[i, j*3 + 0] = np.sum(imf**2)
                        emd_results[i, j*3 + 1] = self._estimate_mean_frequency(imf)
                        emd_results[i, j*3 + 2] = np.std(imf)
            except Exception as e:
                logger.debug(f"EMDウィンドウ計算エラー (index={i}): {e}")
                continue
        
        # 辞書に格納
        for j in range(imf_count):
            features[f'emd_imf_{j}_energy'] = emd_results[:, j*3 + 0]
            features[f'emd_imf_{j}_mean_freq'] = emd_results[:, j*3 + 1]
            features[f'emd_imf_{j}_amplitude'] = emd_results[:, j*3 + 2]
            
        return features

    def _estimate_mean_frequency(self, signal: np.ndarray) -> float:
        """信号の平均周波数推定"""
        try:
            if len(signal) < 8:
                return 0.0
            
            # FFTによる主要周波数検出
            fft_signal = np.fft.fft(signal)
            freqs = np.fft.fftfreq(len(signal))
            power_spectrum = np.abs(fft_signal)**2
            
            # 正の周波数のみ
            positive_freqs = freqs[:len(freqs)//2]
            positive_power = power_spectrum[:len(power_spectrum)//2]
            
            if np.sum(positive_power) > 1e-10:
                mean_freq = np.sum(positive_freqs * positive_power) / np.sum(positive_power)
                return abs(mean_freq)
            else:
                return 0.0
                
        except:
            return 0.0

    def _simple_emd_substitute(self, data: np.ndarray) -> Dict[str, np.ndarray]:
        """EMDの簡易代替実装（ウェーブレットベース）"""
        results = {}
        
        try:
            # 異なるウェーブレットで多重解像度解析
            wavelets = ['db4', 'db8', 'haar']
            
            for wi, wavelet_name in enumerate(wavelets):
                try:
                    # ウェーブレット分解
                    coeffs = pywt.wavedec(data, wavelet_name, level=5)
                    
                    for i, coeff in enumerate(coeffs):
                        level_name = f'approx_{wi}' if i == 0 else f'detail_{wi}_{i}'
                        
                        # エネルギー
                        energy = np.sum(coeff**2)
                        results[f'emd_substitute_{level_name}_energy'] = np.full(len(data), energy)
                        
                        # 周波数推定
                        mean_freq = self._estimate_mean_frequency(coeff)
                        results[f'emd_substitute_{level_name}_freq'] = np.full(len(data), mean_freq)
                        
                except Exception as e:
                    logger.debug(f"ウェーブレット {wavelet_name} エラー: {e}")
                    continue
        
        except Exception as e:
            logger.warning(f"EMD代替計算エラー: {e}")
        
        return results

    # =========================================================================
    # 統合特徴量計算メソッド
    # =========================================================================

    def calculate_all_advanced_features(self, prices: np.ndarray, 
                                        high: np.ndarray = None, 
                                        low: np.ndarray = None,
                                        volume: np.ndarray = None) -> Dict[str, np.ndarray]:
        """
        全高度特徴量の統合計算
        プロンプト要求に従い、すべてのTier S/1/2特徴量を計算
        """
        logger.info("高度特徴量計算開始 - Tier S/1/2 特徴量")
        
        all_features = {}
        
        # Tier S特徴量
        logger.info("Tier S特徴量計算中...")
        
        # MFDFA
        mfdfa_features = self.calculate_mfdfa_features(prices)
        all_features.update(mfdfa_features)
        logger.info(f"MFDFA特徴量: {len(mfdfa_features)}個生成")
        
        # Microstructure Noise Ratio
        noise_features = self.calculate_microstructure_noise_features(prices)
        all_features.update(noise_features)
        logger.info(f"Microstructure Noise特徴量: {len(noise_features)}個生成")
        
        # Tier 1特徴量
        logger.info("Tier 1特徴量計算中...")
        
        # 軽量版ショックモデル
        shock_features = self.calculate_shock_model_features(prices)
        all_features.update(shock_features)
        logger.info(f"ショックモデル特徴量: {len(shock_features)}個生成")
        
        # Multi-Scale Volatility
        multiscale_vol = self.calculate_multiscale_volatility(prices)
        all_features.update(multiscale_vol)
        logger.info(f"Multi-Scale Volatility特徴量: {len(multiscale_vol)}個生成")
        
        # Tier 2特徴量
        logger.info("Tier 2特徴量計算中...")
        
        # EMD/CEEMDAN
        emd_features = self.calculate_emd_features(prices)
        all_features.update(emd_features)
        logger.info(f"EMD特徴量: {len(emd_features)}個生成")
        
        # 統計的モーメント（基礎から高次まで）
        for window in [20, 50, 100]:
            moments = self.calculate_statistical_moments(prices, window)
            all_features.update(moments)
        
        # スペクトル特徴量
        for window in [64, 128]:
            spectral = self.calculate_spectral_features(prices, window)
            all_features.update(spectral)
        
        # カオス理論特徴量
        for window in [50, 100]:
            chaos = self.calculate_chaos_features(prices, window)
            all_features.update(chaos)
        
        logger.info(f"高度特徴量計算完了 - 総計: {len(all_features)}個の特徴量生成")
        
        return all_features

    # =========================================================================
    # 品質保証システム（各ウィンドウでの計算品質スコアリング）
    # =========================================================================
    
    def calculate_window_quality_score(self, window_data: np.ndarray, 
                                     feature_values: np.ndarray,
                                     feature_name: str) -> float:
        """各ウィンドウでの計算品質スコアリング（0-1）"""
        if len(feature_values) == 0:
            return 0.0
        
        # 基本品質スコア
        base_score = 1.0
        
        # 1. データ完全性チェック
        nan_ratio = np.isnan(feature_values).sum() / len(feature_values)
        inf_ratio = np.isinf(feature_values).sum() / len(feature_values)
        
        completeness_score = max(0.0, 1.0 - nan_ratio - inf_ratio)
        
        # 2. 数値安定性チェック
        if len(feature_values[np.isfinite(feature_values)]) > 0:
            finite_values = feature_values[np.isfinite(feature_values)]
            
            # 異常値比率
            if len(finite_values) > 3:
                outlier_threshold = np.percentile(finite_values, [5, 95])
                outlier_ratio = np.sum((finite_values < outlier_threshold[0]) | 
                                     (finite_values > outlier_threshold[1])) / len(finite_values)
                stability_score = max(0.0, 1.0 - outlier_ratio * 2)
            else:
                stability_score = 0.5
        else:
            stability_score = 0.0
        
        # 3. 理論的妥当性チェック
        theoretical_score = self._validate_theoretical_range(feature_values, feature_name)
        
        # 4. 統計的一貫性チェック
        consistency_score = self._check_statistical_consistency(window_data, feature_values)
        
        # 総合品質スコア（重み付き平均）
        weights = {
            'completeness': 0.3,
            'stability': 0.3, 
            'theoretical': 0.25,
            'consistency': 0.15
        }
        
        quality_score = (weights['completeness'] * completeness_score +
                        weights['stability'] * stability_score +
                        weights['theoretical'] * theoretical_score +
                        weights['consistency'] * consistency_score)
        
        # 品質スコアをキャッシュ
        if feature_name not in self.quality_system['feature_quality_scores']:
            self.quality_system['feature_quality_scores'][feature_name] = []
        self.quality_system['feature_quality_scores'][feature_name].append(quality_score)
        
        return quality_score
    
    def _validate_theoretical_range(self, feature_values: np.ndarray, feature_name: str) -> float:
        """理論的範囲外の値に対する自動補正と検証"""
        if len(feature_values) == 0:
            return 0.0
        
        # 特徴量タイプ別の理論的範囲定義
        theoretical_ranges = {
            # 確率・比率系（0-1）
            'ratio': (0.0, 1.0),
            'probability': (0.0, 1.0),
            'correlation': (-1.0, 1.0),
            
            # パーセント系（0-100）
            'percent': (0.0, 100.0),
            'rsi': (0.0, 100.0),
            'stochastic': (0.0, 100.0),
            
            # Hurst指数系（0-1）
            'hurst': (0.0, 1.0),
            'mfdfa_hurst': (0.0, 1.0),
            
            # 無次元指標
            'sharpe': (-5.0, 5.0),
            'sortino': (-5.0, 5.0),
            
            # ボラティリティ系（正値）
            'volatility': (0.0, 10.0),
            'atr': (0.0, np.inf),
            
            # 価格系
            'price': (0.0, np.inf),
            
            # フラクタル次元
            'fractal_dimension': (1.0, 3.0),
            'correlation_dimension': (0.5, 5.0),
        }
        
        # 特徴量名からタイプを推定
        feature_type = self._infer_feature_type(feature_name)
        
        if feature_type in theoretical_ranges:
            min_val, max_val = theoretical_ranges[feature_type]
            
            # 範囲外の値をカウント
            if max_val == np.inf:
                out_of_range = np.sum(feature_values < min_val)
            else:
                out_of_range = np.sum((feature_values < min_val) | (feature_values > max_val))
            
            out_of_range_ratio = out_of_range / len(feature_values)
            
            # 違反をログに記録
            if out_of_range_ratio > 0.1:  # 10%以上の違反
                if feature_name not in self.quality_system['theoretical_range_violations']:
                    self.quality_system['theoretical_range_violations'][feature_name] = []
                self.quality_system['theoretical_range_violations'][feature_name].append(out_of_range_ratio)
                
                logger.warning(f"理論的範囲違反: {feature_name} ({out_of_range_ratio:.2%}が範囲外)")
            
            return max(0.0, 1.0 - out_of_range_ratio * 2)
        else:
            # 未知の特徴量タイプ - 基本的な数値妥当性のみチェック
            finite_ratio = np.sum(np.isfinite(feature_values)) / len(feature_values)
            return finite_ratio
    
    def _infer_feature_type(self, feature_name: str) -> str:
        """特徴量名からタイプを推定"""
        name_lower = feature_name.lower()
        
        # 優先度順でチェック
        if any(keyword in name_lower for keyword in ['ratio', 'proportion']):
            return 'ratio'
        elif any(keyword in name_lower for keyword in ['percent', '%']):
            return 'percent'
        elif any(keyword in name_lower for keyword in ['rsi']):
            return 'rsi' 
        elif any(keyword in name_lower for keyword in ['stoch', 'stochastic']):
            return 'stochastic'
        elif any(keyword in name_lower for keyword in ['hurst', 'mfdfa_hurst']):
            return 'hurst'
        elif any(keyword in name_lower for keyword in ['correlation', 'corr']):
            return 'correlation'
        elif any(keyword in name_lower for keyword in ['sharpe']):
            return 'sharpe'
        elif any(keyword in name_lower for keyword in ['sortino']):
            return 'sortino'
        elif any(keyword in name_lower for keyword in ['volatility', 'vol']):
            return 'volatility'
        elif any(keyword in name_lower for keyword in ['atr']):
            return 'atr'
        elif any(keyword in name_lower for keyword in ['price']):
            return 'price'
        elif any(keyword in name_lower for keyword in ['fractal_dimension']):
            return 'fractal_dimension'
        elif any(keyword in name_lower for keyword in ['correlation_dimension']):
            return 'correlation_dimension'
        else:
            return 'unknown'
    
    def _check_statistical_consistency(self, input_data: np.ndarray, feature_values: np.ndarray) -> float:
        """統計的一貫性チェック"""
        if len(input_data) < 10 or len(feature_values) < 10:
            return 0.5  # 中性スコア
        
        try:
            # 入力データの統計量
            input_mean = np.nanmean(input_data)
            input_std = np.nanstd(input_data)
            input_range = np.nanmax(input_data) - np.nanmin(input_data)
            
            # 特徴量の統計量  
            feature_mean = np.nanmean(feature_values)
            feature_std = np.nanstd(feature_values)
            feature_range = np.nanmax(feature_values) - np.nanmin(feature_values)
            
            # 一貫性スコア
            consistency_checks = []
            
            # 1. スケール一貫性（特徴量が入力データと極端に異なるスケールでないか）
            if input_std > 1e-10 and feature_std > 1e-10:
                scale_ratio = feature_std / input_std
                if 1e-3 <= scale_ratio <= 1e3:  # 3桁以内の差
                    consistency_checks.append(1.0)
                else:
                    consistency_checks.append(0.5)
            else:
                consistency_checks.append(0.5)
            
            # 2. 値域一貫性
            if input_range > 1e-10 and feature_range > 1e-10:
                range_ratio = feature_range / input_range
                if 1e-2 <= range_ratio <= 1e2:  # 2桁以内の差
                    consistency_checks.append(1.0)
                else:
                    consistency_checks.append(0.7)
            else:
                consistency_checks.append(0.5)
            
            # 3. 分布の形状一貫性（歪度・尖度）
            try:
                input_skew = stats.skew(input_data[np.isfinite(input_data)])
                feature_skew = stats.skew(feature_values[np.isfinite(feature_values)])
                
                if abs(input_skew - feature_skew) < 2.0:  # 歪度の差が2以下
                    consistency_checks.append(1.0)
                else:
                    consistency_checks.append(0.8)
            except:
                consistency_checks.append(0.5)
            
            return np.mean(consistency_checks)
            
        except Exception as e:
            logger.debug(f"統計的一貫性チェックエラー: {e}")
            return 0.5
    
    # =========================================================================
    # 統計的妥当性のリアルタイム検証
    # =========================================================================
    
    def validate_feature_significance_realtime(self, feature_values: np.ndarray, 
                                             feature_name: str,
                                             confidence_level: float = 0.95) -> Dict[str, Any]:
        """統計的妥当性のリアルタイム検証"""
        # キャッシュチェック
        cache_key = f"{feature_name}_{confidence_level}_{len(feature_values)}"
        if cache_key in self.quality_system['statistical_significance_cache']:
            return self.quality_system['statistical_significance_cache'][cache_key]
        
        clean_values = feature_values[np.isfinite(feature_values)]
        
        if len(clean_values) < 10:
            result = {
                'significant': False,
                'reason': 'insufficient_data',
                'sample_size': len(clean_values),
                'power': 0.0
            }
        else:
            result = self._perform_significance_tests(clean_values, confidence_level)
        
        # 結果をキャッシュ
        self.quality_system['statistical_significance_cache'][cache_key] = result
        return result
    
    def _perform_significance_tests(self, values: np.ndarray, confidence_level: float) -> Dict[str, Any]:
        """統計的有意性テストの実行"""
        alpha = 1 - confidence_level
        
        try:
            # 1. 正規性テスト
            normality_stat, normality_p = stats.normaltest(values)
            is_normal = normality_p > alpha
            
            # 2. ゼロからの有意差テスト
            if is_normal and len(values) > 30:
                # パラメトリック t-test
                t_stat, t_p_value = stats.ttest_1samp(values, 0)
                test_type = 'parametric_ttest'
            else:
                # ノンパラメトリック Wilcoxon signed-rank test
                try:
                    w_stat, t_p_value = stats.wilcoxon(values, alternative='two-sided')
                    test_type = 'nonparametric_wilcoxon'
                except:
                    # フォールバック: sign test
                    positive_count = np.sum(values > 0)
                    total_count = len(values)
                    t_p_value = 2 * min(
                        stats.binom.cdf(positive_count, total_count, 0.5),
                        1 - stats.binom.cdf(positive_count - 1, total_count, 0.5)
                    )
                    test_type = 'sign_test'
            
            # 3. 効果サイズ（Cohen's d相当）
            effect_size = abs(np.mean(values)) / (np.std(values, ddof=1) + 1e-10)
            
            # 4. 統計的検出力（事後的推定）
            try:
                # 簡易検出力計算
                observed_effect = effect_size
                sample_size = len(values)
                power = self._estimate_statistical_power(observed_effect, sample_size, alpha)
            except:
                power = 0.5  # デフォルト値
            
            # 5. 信頼区間
            confidence_interval = self._calculate_confidence_interval(values, confidence_level)
            
            return {
                'significant': t_p_value < alpha,
                'p_value': t_p_value,
                'effect_size': effect_size,
                'confidence_interval': confidence_interval,
                'test_type': test_type,
                'is_normal': is_normal,
                'sample_size': len(values),
                'power': power,
                'mean': np.mean(values),
                'std': np.std(values, ddof=1)
            }
            
        except Exception as e:
            logger.debug(f"有意性テストエラー: {e}")
            return {
                'significant': False,
                'reason': 'test_failed',
                'error': str(e),
                'sample_size': len(values)
            }
    
    def _estimate_statistical_power(self, effect_size: float, sample_size: int, alpha: float) -> float:
        """統計的検出力の簡易推定"""
        try:
            # Cohen's 効果サイズに基づく検出力推定（近似式）
            z_alpha = stats.norm.ppf(1 - alpha/2)  # 両側検定
            z_beta = effect_size * np.sqrt(sample_size) - z_alpha
            power = stats.norm.cdf(z_beta)
            
            return max(0.0, min(1.0, power))
        except:
            return 0.5
    
    def _calculate_confidence_interval(self, values: np.ndarray, confidence_level: float) -> Tuple[float, float]:
        """信頼区間計算"""
        try:
            alpha = 1 - confidence_level
            n = len(values)
            
            if n < 2:
                return (np.nan, np.nan)
            
            mean = np.mean(values)
            sem = stats.sem(values)  # 標準誤差
            
            # t分布による信頼区間
            t_critical = stats.t.ppf(1 - alpha/2, n - 1)
            margin_error = t_critical * sem
            
            return (mean - margin_error, mean + margin_error)
            
        except:
            return (np.nan, np.nan)
    
    # =========================================================================
    # ロバスト推定によるフォールバック機能
    # =========================================================================
    
    def apply_robust_fallback(self, original_result: np.ndarray, 
                            fallback_method: str = 'median_filter') -> np.ndarray:
        """ロバスト推定によるフォールバック機能"""
        if len(original_result) == 0:
            return original_result
        
        try:
            if fallback_method == 'median_filter':
                # メディアンフィルタ
                from scipy.ndimage import median_filter
                kernel_size = min(5, len(original_result) // 10 * 2 + 1)  # 奇数
                return median_filter(original_result, size=kernel_size)
            
            elif fallback_method == 'hampel_filter':
                # Hampelフィルタ（外れ値除去）
                return self._apply_hampel_filter(original_result)
            
            elif fallback_method == 'winsorized':
                # ウィンソライゼーション（5%トリミング）
                return stats.mstats.winsorize(original_result, limits=[0.05, 0.05])
            
            elif fallback_method == 'robust_scaler':
                # ロバストスケーリング（中央値とMAD使用）
                median = np.median(original_result)
                mad = np.median(np.abs(original_result - median))
                if mad > 1e-10:
                    return (original_result - median) / mad
                else:
                    return original_result - median
            
            else:
                logger.warning(f"未知のフォールバック方法: {fallback_method}")
                return original_result
                
        except Exception as e:
            logger.debug(f"フォールバック処理エラー: {e}")
            return original_result
    
    @staticmethod
    @njit(cache=True)
    def _apply_hampel_filter_numba(data: np.ndarray, window_size: int, 
                                 threshold: float) -> np.ndarray:
        """Hampelフィルタ - Numba最適化版"""
        n = len(data)
        filtered_data = data.copy()
        half_window = window_size // 2
        
        for i in range(half_window, n - half_window):
            window = data[i - half_window:i + half_window + 1]
            median = np.median(window)
            mad = np.median(np.abs(window - median))
            
            if mad > 1e-10:
                modified_z_score = 0.6745 * (data[i] - median) / mad
                if abs(modified_z_score) > threshold:
                    filtered_data[i] = median
        
        return filtered_data
    
    def _apply_hampel_filter(self, data: np.ndarray, window_size: int = 7, 
                           threshold: float = 3.0) -> np.ndarray:
        """Hampelフィルタ適用"""
        if len(data) < window_size:
            return data
        
        return self._apply_hampel_filter_numba(data, window_size, threshold)
    
    # =========================================================================
    # NaN率・外れ値比率の継続監視
    # =========================================================================
    
    def monitor_feature_quality_continuous(self, feature_dict: Dict[str, np.ndarray]) -> Dict[str, Any]:
        """特徴量品質の継続監視"""
        monitoring_report = {
            'timestamp': time.time(),
            'total_features': len(feature_dict),
            'quality_summary': {},
            'alerts': [],
            'recommendations': []
        }
        
        for feature_name, feature_values in feature_dict.items():
            quality_metrics = self._calculate_quality_metrics(feature_values)
            monitoring_report['quality_summary'][feature_name] = quality_metrics
            
            # 品質アラート
            alerts = self._generate_quality_alerts(feature_name, quality_metrics)
            monitoring_report['alerts'].extend(alerts)
        
        # 全体的な品質統計
        all_quality_scores = [metrics['overall_quality'] for metrics in monitoring_report['quality_summary'].values()]
        monitoring_report['average_quality'] = np.mean(all_quality_scores)
        monitoring_report['quality_distribution'] = {
            'high_quality_count': sum(1 for q in all_quality_scores if q >= 0.8),
            'medium_quality_count': sum(1 for q in all_quality_scores if 0.5 <= q < 0.8),
            'low_quality_count': sum(1 for q in all_quality_scores if q < 0.5)
        }
        
        # 推奨アクション
        monitoring_report['recommendations'] = self._generate_quality_recommendations(monitoring_report)
        
        return monitoring_report
    
    def _calculate_quality_metrics(self, feature_values: np.ndarray) -> Dict[str, float]:
        """個別特徴量の品質メトリクス計算"""
        if len(feature_values) == 0:
            return {'overall_quality': 0.0, 'nan_ratio': 1.0, 'inf_ratio': 1.0}
        
        # 基本統計
        nan_ratio = np.isnan(feature_values).sum() / len(feature_values)
        inf_ratio = np.isinf(feature_values).sum() / len(feature_values)
        finite_ratio = 1.0 - nan_ratio - inf_ratio
        
        # 数値範囲チェック
        if finite_ratio > 0:
            finite_values = feature_values[np.isfinite(feature_values)]
            value_range = np.ptp(finite_values)  # peak-to-peak
            outlier_ratio = self._calculate_outlier_ratio(finite_values)
            uniqueness_ratio = len(np.unique(finite_values)) / len(finite_values)
        else:
            value_range = 0.0
            outlier_ratio = 0.0
            uniqueness_ratio = 0.0
        
        # 総合品質スコア
        overall_quality = (finite_ratio * 0.4 +
                          (1.0 - outlier_ratio) * 0.3 +
                          uniqueness_ratio * 0.2 +
                          min(1.0, value_range / (value_range + 1)) * 0.1)
        
        return {
            'overall_quality': overall_quality,
            'nan_ratio': nan_ratio,
            'inf_ratio': inf_ratio,
            'finite_ratio': finite_ratio,
            'outlier_ratio': outlier_ratio,
            'uniqueness_ratio': uniqueness_ratio,
            'value_range': value_range
        }
    
    def _calculate_outlier_ratio(self, values: np.ndarray) -> float:
        """外れ値比率計算"""
        if len(values) < 4:
            return 0.0
        
        try:
            Q1 = np.percentile(values, 25)
            Q3 = np.percentile(values, 75)
            IQR = Q3 - Q1
            
            if IQR > 1e-10:
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                outliers = (values < lower_bound) | (values > upper_bound)
                return np.sum(outliers) / len(values)
            else:
                return 0.0
        except:
            return 0.0
    
    def _generate_quality_alerts(self, feature_name: str, metrics: Dict[str, float]) -> List[Dict[str, Any]]:
        """品質アラート生成"""
        alerts = []
        
        # 高NaN率アラート
        if metrics['nan_ratio'] > 0.2:  # 20%以上
            alerts.append({
                'type': 'high_nan_ratio',
                'feature': feature_name,
                'severity': 'high' if metrics['nan_ratio'] > 0.5 else 'medium',
                'value': metrics['nan_ratio'],
                'message': f"高いNaN率: {metrics['nan_ratio']:.2%}"
            })
        
        # Inf値アラート
        if metrics['inf_ratio'] > 0.05:  # 5%以上
            alerts.append({
                'type': 'inf_values',
                'feature': feature_name,
                'severity': 'high',
                'value': metrics['inf_ratio'],
                'message': f"Inf値検出: {metrics['inf_ratio']:.2%}"
            })
        
        # 低品質アラート
        if metrics['overall_quality'] < 0.5:
            alerts.append({
                'type': 'low_quality',
                'feature': feature_name,
                'severity': 'medium',
                'value': metrics['overall_quality'],
                'message': f"低品質特徴量: スコア {metrics['overall_quality']:.3f}"
            })
        
        # 低多様性アラート
        if metrics['uniqueness_ratio'] < 0.1:  # 10%未満のユニーク値
            alerts.append({
                'type': 'low_diversity',
                'feature': feature_name,
                'severity': 'low',
                'value': metrics['uniqueness_ratio'],
                'message': f"低多様性: ユニーク値率 {metrics['uniqueness_ratio']:.2%}"
            })
        
        return alerts
    
    def _generate_quality_recommendations(self, monitoring_report: Dict[str, Any]) -> List[str]:
        """品質改善推奨アクション"""
        recommendations = []
        
        # アラート数による推奨
        high_severity_alerts = [a for a in monitoring_report['alerts'] if a['severity'] == 'high']
        if len(high_severity_alerts) > monitoring_report['total_features'] * 0.1:
            recommendations.append("計算パラメータの見直しが必要です")
            recommendations.append("数値安定化処理の強化を検討してください")
        
        # 品質分布による推奨
        dist = monitoring_report['quality_distribution']
        if dist['low_quality_count'] > monitoring_report['total_features'] * 0.2:
            recommendations.append("低品質特徴量が多数検出されました - フィルタリング処理の追加を推奨")
            recommendations.append("ロバスト推定手法の適用を検討してください")
        
        # 平均品質による推奨
        if monitoring_report['average_quality'] < 0.7:
            recommendations.append("全体的な特徴量品質が低下しています")
            recommendations.append("データ前処理の見直しを行ってください")
            recommendations.append("ウィンドウサイズの最適化を検討してください")
        
        return recommendations
    
    # =============================================================================
    # 欠落特徴量補完 - Block 1: ADX・基本オシレーター
    # =============================================================================

    @staticmethod
    @njit(cache=True)
    def _compute_adx_numba(high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """ADX（平均方向性指数）とDI+, DI-の計算 - Numba最適化版"""
        n = len(high)
        if n < period + 1:
            return np.zeros(n), np.zeros(n), np.zeros(n)
        
        # True Range計算
        tr = np.zeros(n)
        dm_plus = np.zeros(n)
        dm_minus = np.zeros(n)
        
        for i in range(1, n):
            # True Range
            hl = high[i] - low[i]
            hc = abs(high[i] - close[i-1])
            lc = abs(low[i] - close[i-1])
            tr[i] = max(hl, max(hc, lc))
            
            # Directional Movement
            up_move = high[i] - high[i-1]
            down_move = low[i-1] - low[i]
            
            if up_move > down_move and up_move > 0:
                dm_plus[i] = up_move
            if down_move > up_move and down_move > 0:
                dm_minus[i] = down_move
        
        # スムージング
        alpha = 1.0 / period
        atr_smooth = np.zeros(n)
        dm_plus_smooth = np.zeros(n)
        dm_minus_smooth = np.zeros(n)
        
        # 初期値
        atr_smooth[period] = np.mean(tr[1:period+1])
        dm_plus_smooth[period] = np.mean(dm_plus[1:period+1])
        dm_minus_smooth[period] = np.mean(dm_minus[1:period+1])
        
        # EMA計算
        for i in range(period + 1, n):
            atr_smooth[i] = alpha * tr[i] + (1 - alpha) * atr_smooth[i-1]
            dm_plus_smooth[i] = alpha * dm_plus[i] + (1 - alpha) * dm_plus_smooth[i-1]
            dm_minus_smooth[i] = alpha * dm_minus[i] + (1 - alpha) * dm_minus_smooth[i-1]
        
        # DI+, DI-計算
        di_plus = np.zeros(n)
        di_minus = np.zeros(n)
        adx = np.zeros(n)
        
        for i in range(period, n):
            if atr_smooth[i] > 1e-10:
                di_plus[i] = 100 * dm_plus_smooth[i] / atr_smooth[i]
                di_minus[i] = 100 * dm_minus_smooth[i] / atr_smooth[i]
        
        # ADX計算
        dx = np.zeros(n)
        for i in range(period, n):
            di_sum = di_plus[i] + di_minus[i]
            if di_sum > 1e-10:
                dx[i] = 100 * abs(di_plus[i] - di_minus[i]) / di_sum
        
        # ADXのスムージング
        adx[period] = np.mean(dx[period:period+period])
        for i in range(period + period, n):
            adx[i] = alpha * dx[i] + (1 - alpha) * adx[i-1]
        
        return adx, di_plus, di_minus

    def calculate_adx_features(self, high: np.ndarray, low: np.ndarray, close: np.ndarray) -> Dict[str, np.ndarray]:
        """ADX関連特徴量の計算"""
        features = {}
        
        for period in self.params['adx_periods']:
            adx, di_plus, di_minus = self.robust_calculation_wrapper(
                self._compute_adx_numba, high, low, close, period
            )
            
            features[f'adx_{period}'] = adx
            features[f'di_plus_{period}'] = di_plus
            features[f'di_minus_{period}'] = di_minus
            features[f'di_diff_{period}'] = di_plus - di_minus
            features[f'adx_strength_{period}'] = (adx > 25).astype(float)
            features[f'trend_strength_{period}'] = adx / 100.0
        
        return features

    @staticmethod
    @njit(cache=True)
    def _compute_parabolic_sar_numba(high: np.ndarray, low: np.ndarray, af_start: float = 0.02, af_max: float = 0.2) -> Tuple[np.ndarray, np.ndarray]:
        """パラボリックSAR計算 - Numba最適化版"""
        n = len(high)
        if n < 2:
            return np.zeros(n), np.zeros(n)
        
        sar = np.zeros(n)
        signal = np.zeros(n)
        
        # 初期設定
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
        
        return sar, signal

    def calculate_parabolic_sar_features(self, high: np.ndarray, low: np.ndarray, close: np.ndarray) -> Dict[str, np.ndarray]:
        """パラボリックSAR特徴量の計算"""
        features = {}
        
        sar, signal = self.robust_calculation_wrapper(self._compute_parabolic_sar_numba, high, low)
        
        features['parabolic_sar'] = sar
        features['sar_signal'] = signal
        features['sar_distance'] = (close - sar) / close
        features['sar_above'] = (close > sar).astype(float)
        features['sar_trend_strength'] = np.abs(signal)
        
        return features

    @staticmethod
    @njit(cache=True)
    def _compute_cci_numba(high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int) -> np.ndarray:
        """CCI（Commodity Channel Index）計算 - Numba最適化版"""
        n = len(high)
        if n < period:
            return np.zeros(n)
        
        # Typical Price
        tp = (high + low + close) / 3.0
        cci = np.zeros(n)
        
        for i in range(period - 1, n):
            window_tp = tp[i - period + 1:i + 1]
            sma_tp = np.mean(window_tp)
            
            # Mean Deviation
            mad = np.mean(np.abs(window_tp - sma_tp))
            
            if mad > 1e-10:
                cci[i] = (tp[i] - sma_tp) / (0.015 * mad)
            else:
                cci[i] = 0.0
        
        return cci

    def calculate_cci_features(self, high: np.ndarray, low: np.ndarray, close: np.ndarray) -> Dict[str, np.ndarray]:
        """CCI特徴量の計算"""
        features = {}
        
        for period in self.params['cci_periods']:
            cci = self.robust_calculation_wrapper(self._compute_cci_numba, high, low, close, period)
            
            features[f'cci_{period}'] = cci
            features[f'cci_overbought_{period}'] = (cci > 100).astype(float)
            features[f'cci_oversold_{period}'] = (cci < -100).astype(float)
            features[f'cci_normalized_{period}'] = np.tanh(cci / 100.0)  # -1から1に正規化
            
        return features

    @staticmethod
    @njit(cache=True)
    def _compute_williams_r_numba(high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int) -> np.ndarray:
        """ウィリアムズ%R計算 - Numba最適化版"""
        n = len(high)
        if n < period:
            return np.zeros(n)
        
        williams_r = np.zeros(n)
        
        for i in range(period - 1, n):
            window_high = high[i - period + 1:i + 1]
            window_low = low[i - period + 1:i + 1]
            
            highest_high = np.max(window_high)
            lowest_low = np.min(window_low)
            
            if highest_high - lowest_low > 1e-10:
                williams_r[i] = -100.0 * (highest_high - close[i]) / (highest_high - lowest_low)
            else:
                williams_r[i] = -50.0
        
        return williams_r

    def calculate_williams_r_features(self, high: np.ndarray, low: np.ndarray, close: np.ndarray) -> Dict[str, np.ndarray]:
        """ウィリアムズ%R特徴量の計算"""
        features = {}
        
        for period in self.params['williams_r_periods']:
            williams_r = self.robust_calculation_wrapper(self._compute_williams_r_numba, high, low, close, period)
            
            features[f'williams_r_{period}'] = williams_r
            features[f'williams_r_overbought_{period}'] = (williams_r > -20).astype(float)
            features[f'williams_r_oversold_{period}'] = (williams_r < -80).astype(float)
            features[f'williams_r_normalized_{period}'] = (williams_r + 50) / 50.0  # -1から1に正規化
            
        return features

    @staticmethod
    @njit(cache=True)
    def _compute_aroon_numba(high: np.ndarray, low: np.ndarray, period: int) -> Tuple[np.ndarray, np.ndarray]:
        """アルーン指標計算 - Numba最適化版"""
        n = len(high)
        if n < period:
            return np.zeros(n), np.zeros(n)
        
        aroon_up = np.zeros(n)
        aroon_down = np.zeros(n)
        
        for i in range(period - 1, n):
            window_high = high[i - period + 1:i + 1]
            window_low = low[i - period + 1:i + 1]
            
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
        
        return aroon_up, aroon_down

    def calculate_aroon_features(self, high: np.ndarray, low: np.ndarray) -> Dict[str, np.ndarray]:
        """アルーン特徴量の計算"""
        features = {}
        
        for period in self.params['aroon_periods']:
            aroon_up, aroon_down = self.robust_calculation_wrapper(self._compute_aroon_numba, high, low, period)
            
            features[f'aroon_up_{period}'] = aroon_up
            features[f'aroon_down_{period}'] = aroon_down
            features[f'aroon_oscillator_{period}'] = aroon_up - aroon_down
            features[f'aroon_trending_{period}'] = (np.abs(aroon_up - aroon_down) > 50).astype(float)
            
        return features

    @staticmethod
    @njit(cache=True)
    def _compute_ultimate_oscillator_numba(high: np.ndarray, low: np.ndarray, close: np.ndarray) -> np.ndarray:
        """アルティメットオシレーター計算 - Numba最適化版"""
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

    def calculate_ultimate_oscillator_features(self, high: np.ndarray, low: np.ndarray, close: np.ndarray) -> Dict[str, np.ndarray]:
        """アルティメットオシレーター特徴量の計算"""
        features = {}
        
        uo = self.robust_calculation_wrapper(self._compute_ultimate_oscillator_numba, high, low, close)
        
        features['ultimate_oscillator'] = uo
        features['uo_overbought'] = (uo > 70).astype(float)
        features['uo_oversold'] = (uo < 30).astype(float)
        features['uo_normalized'] = (uo - 50) / 50.0  # -1から1に正規化
        
        return features

    # =============================================================================
    # 欠落特徴量補完 - Block 2: 出来高関連指標
    # =============================================================================

    @staticmethod
    @njit(cache=True)
    def _compute_vpt_numba(close: np.ndarray, volume: np.ndarray) -> np.ndarray:
        """Volume Price Trend（VPT）計算 - Numba最適化版"""
        n = len(close)
        if n < 2:
            return np.zeros(n)
        
        vpt = np.zeros(n)
        vpt[0] = 0
        
        for i in range(1, n):
            if close[i-1] > 1e-10:
                price_change_ratio = (close[i] - close[i-1]) / close[i-1]
                vpt[i] = vpt[i-1] + volume[i] * price_change_ratio
            else:
                vpt[i] = vpt[i-1]
        
        return vpt

    @staticmethod
    @njit(cache=True)
    def _compute_rolling_mean_numba(data: np.ndarray, window: int) -> np.ndarray:
        """ローリング移動平均をNumbaで計算"""
        n = len(data)
        result = np.zeros(n)
        
        for i in range(window - 1, n):
            result[i] = np.mean(data[i - window + 1:i + 1])
        
        return result

    def calculate_vpt_features(self, close: np.ndarray, volume: np.ndarray) -> Dict[str, np.ndarray]:
        """VPT特徴量の計算"""
        features = {}
        
        vpt = self.robust_calculation_wrapper(self._compute_vpt_numba, close, volume)
        
        features['vpt'] = vpt
        features['vpt_signal'] = np.gradient(vpt)
        
        # VPTの移動平均
        for period in [10, 20, 50]:
            vpt_ma = self.robust_calculation_wrapper(self._compute_rolling_mean_numba, vpt, period)
            features[f'vpt_ma_{period}'] = vpt_ma
            features[f'vpt_above_ma_{period}'] = (vpt > vpt_ma).astype(float)
        
        return features

    @staticmethod
    @njit(cache=True)
    def _compute_ad_line_numba(high: np.ndarray, low: np.ndarray, close: np.ndarray, volume: np.ndarray) -> np.ndarray:
        """Accumulation/Distribution Line計算 - Numba最適化版"""
        n = len(high)
        if n < 1:
            return np.zeros(n)
        
        ad_line = np.zeros(n)
        
        for i in range(n):
            if high[i] - low[i] > 1e-10:
                clv = ((close[i] - low[i]) - (high[i] - close[i])) / (high[i] - low[i])
                ad_line[i] = ad_line[i-1] + clv * volume[i] if i > 0 else clv * volume[i]
            else:
                ad_line[i] = ad_line[i-1] if i > 0 else 0
        
        return ad_line

    def calculate_ad_line_features(self, high: np.ndarray, low: np.ndarray, close: np.ndarray, volume: np.ndarray) -> Dict[str, np.ndarray]:
        """A/D Line特徴量の計算"""
        features = {}
        
        ad_line = self.robust_calculation_wrapper(self._compute_ad_line_numba, high, low, close, volume)
        
        features['ad_line'] = ad_line
        features['ad_line_momentum'] = np.gradient(ad_line)
        
        # A/Dラインと価格のダイバージェンス検出
        price_momentum = np.gradient(close)
        ad_momentum = np.gradient(ad_line)
        
        # 正規化してダイバージェンス計算
        price_mom_norm = price_momentum / (np.std(price_momentum) + 1e-10)
        ad_mom_norm = ad_momentum / (np.std(ad_momentum) + 1e-10)
        
        features['ad_price_divergence'] = price_mom_norm - ad_mom_norm
        
        return features

    @staticmethod
    @njit(cache=True)
    def _compute_cmf_numba(high: np.ndarray, low: np.ndarray, close: np.ndarray, volume: np.ndarray, period: int) -> np.ndarray:
        """Chaikin Money Flow（CMF）計算 - Numba最適化版"""
        n = len(high)
        if n < period:
            return np.zeros(n)
        
        cmf = np.zeros(n)
        
        for i in range(period - 1, n):
            money_flow_volume = 0.0
            total_volume = 0.0
            
            for j in range(i - period + 1, i + 1):
                if high[j] - low[j] > 1e-10:
                    clv = ((close[j] - low[j]) - (high[j] - close[j])) / (high[j] - low[j])
                    money_flow_volume += clv * volume[j]
                total_volume += volume[j]
            
            if total_volume > 1e-10:
                cmf[i] = money_flow_volume / total_volume
            else:
                cmf[i] = 0.0
        
        return cmf

    def calculate_cmf_features(self, high: np.ndarray, low: np.ndarray, close: np.ndarray, volume: np.ndarray) -> Dict[str, np.ndarray]:
        """CMF特徴量の計算"""
        features = {}
        
        for period in self.params['cmf_periods']:
            cmf = self.robust_calculation_wrapper(self._compute_cmf_numba, high, low, close, volume, period)
            
            features[f'cmf_{period}'] = cmf
            features[f'cmf_positive_{period}'] = (cmf > 0).astype(float)
            features[f'cmf_strong_positive_{period}'] = (cmf > 0.2).astype(float)
            features[f'cmf_strong_negative_{period}'] = (cmf < -0.2).astype(float)
        
        return features

    @staticmethod
    @njit(cache=True)
    def _compute_chaikin_oscillator_numba(high: np.ndarray, low: np.ndarray, close: np.ndarray, volume: np.ndarray) -> np.ndarray:
        """チャイキンオシレーター計算 - Numba最適化版"""
        n = len(high)
        if n < 10:
            return np.zeros(n)
        
        # A/D Line計算
        ad_line = np.zeros(n)
        for i in range(n):
            if high[i] - low[i] > 1e-10:
                clv = ((close[i] - low[i]) - (high[i] - close[i])) / (high[i] - low[i])
                ad_line[i] = ad_line[i-1] + clv * volume[i] if i > 0 else clv * volume[i]
            else:
                ad_line[i] = ad_line[i-1] if i > 0 else 0
        
        # 3日と10日のEMA
        ema3 = np.zeros(n)
        ema10 = np.zeros(n)
        alpha3 = 2.0 / (3 + 1)
        alpha10 = 2.0 / (10 + 1)
        
        ema3[0] = ad_line[0]
        ema10[0] = ad_line[0]
        
        for i in range(1, n):
            ema3[i] = alpha3 * ad_line[i] + (1 - alpha3) * ema3[i-1]
            ema10[i] = alpha10 * ad_line[i] + (1 - alpha10) * ema10[i-1]
        
        return ema3 - ema10

    def calculate_chaikin_oscillator_features(self, high: np.ndarray, low: np.ndarray, close: np.ndarray, volume: np.ndarray) -> Dict[str, np.ndarray]:
        """チャイキンオシレーター特徴量の計算"""
        features = {}
        
        chaikin_osc = self.robust_calculation_wrapper(self._compute_chaikin_oscillator_numba, high, low, close, volume)
        
        features['chaikin_oscillator'] = chaikin_osc
        features['chaikin_positive'] = (chaikin_osc > 0).astype(float)
        features['chaikin_momentum'] = np.gradient(chaikin_osc)
        features['chaikin_normalized'] = np.tanh(chaikin_osc / (np.std(chaikin_osc) + 1e-10))
        
        return features

    @staticmethod
    @njit(cache=True)
    def _compute_mfi_numba(high: np.ndarray, low: np.ndarray, close: np.ndarray, volume: np.ndarray, period: int) -> np.ndarray:
        """Money Flow Index（MFI）計算 - Numba最適化版"""
        n = len(high)
        if n < period + 1:
            return np.zeros(n)
        
        # Typical Price
        tp = (high + low + close) / 3.0
        raw_money_flow = tp * volume
        
        mfi = np.zeros(n)
        
        for i in range(period, n):
            positive_flow = 0.0
            negative_flow = 0.0
            
            for j in range(i - period + 1, i + 1):
                if j > 0:
                    if tp[j] > tp[j-1]:
                        positive_flow += raw_money_flow[j]
                    elif tp[j] < tp[j-1]:
                        negative_flow += raw_money_flow[j]
            
            if negative_flow > 1e-10:
                money_ratio = positive_flow / negative_flow
                mfi[i] = 100 - (100 / (1 + money_ratio))
            else:
                mfi[i] = 100 if positive_flow > 0 else 50
        
        return mfi

    def calculate_mfi_features(self, high: np.ndarray, low: np.ndarray, close: np.ndarray, volume: np.ndarray) -> Dict[str, np.ndarray]:
        """MFI特徴量の計算"""
        features = {}
        
        for period in self.params['mfi_periods']:
            mfi = self.robust_calculation_wrapper(self._compute_mfi_numba, high, low, close, volume, period)
            
            features[f'mfi_{period}'] = mfi
            features[f'mfi_overbought_{period}'] = (mfi > 80).astype(float)
            features[f'mfi_oversold_{period}'] = (mfi < 20).astype(float)
            features[f'mfi_normalized_{period}'] = (mfi - 50) / 50.0  # -1から1に正規化
        
        return features

    @staticmethod
    @njit(cache=True)
    def _compute_vwap_numba(high: np.ndarray, low: np.ndarray, close: np.ndarray, volume: np.ndarray, period: int) -> np.ndarray:
        """VWAP（Volume Weighted Average Price）計算 - Numba最適化版"""
        n = len(high)
        if n < period:
            return np.zeros(n)
        
        typical_price = (high + low + close) / 3.0
        vwap = np.zeros(n)
        
        for i in range(period - 1, n):
            total_pv = 0.0
            total_volume = 0.0
            
            for j in range(i - period + 1, i + 1):
                total_pv += typical_price[j] * volume[j]
                total_volume += volume[j]
            
            if total_volume > 1e-10:
                vwap[i] = total_pv / total_volume
            else:
                vwap[i] = typical_price[i]
        
        return vwap

    def calculate_vwap_features(self, high: np.ndarray, low: np.ndarray, close: np.ndarray, volume: np.ndarray) -> Dict[str, np.ndarray]:
        """VWAP特徴量の計算"""
        features = {}
        
        for period in [20, 50, 100]:
            vwap = self.robust_calculation_wrapper(self._compute_vwap_numba, high, low, close, volume, period)
            
            features[f'vwap_{period}'] = vwap
            features[f'price_above_vwap_{period}'] = (close > vwap).astype(float)
            features[f'vwap_distance_{period}'] = (close - vwap) / close
            features[f'vwap_deviation_{period}'] = np.abs(close - vwap) / vwap
        
        return features

    @staticmethod
    @njit(cache=True)
    def _compute_volume_oscillator_numba(volume: np.ndarray, short_period: int, long_period: int) -> np.ndarray:
        """ボリュームオシレーター計算 - Numba最適化版"""
        n = len(volume)
        if n < long_period:
            return np.zeros(n)
        
        vo = np.zeros(n)
        
        for i in range(long_period - 1, n):
            short_avg = np.mean(volume[i - short_period + 1:i + 1]) if i >= short_period - 1 else volume[i]
            long_avg = np.mean(volume[i - long_period + 1:i + 1])
            
            if long_avg > 1e-10:
                vo[i] = 100 * (short_avg - long_avg) / long_avg
            else:
                vo[i] = 0.0
        
        return vo

    def calculate_volume_oscillator_features(self, volume: np.ndarray) -> Dict[str, np.ndarray]:
        """ボリュームオシレーター特徴量の計算"""
        features = {}
        
        # 複数の期間設定
        period_pairs = [(5, 10), (10, 20), (14, 28)]
        
        for short, long in period_pairs:
            vo = self.robust_calculation_wrapper(self._compute_volume_oscillator_numba, volume, short, long)
            
            features[f'volume_oscillator_{short}_{long}'] = vo
            features[f'vo_positive_{short}_{long}'] = (vo > 0).astype(float)
            features[f'vo_momentum_{short}_{long}'] = np.gradient(vo)
        
        return features

    @staticmethod
    @njit(cache=True)
    def _compute_ease_of_movement_numba(high: np.ndarray, low: np.ndarray, volume: np.ndarray, scale: float = 10000.0) -> np.ndarray:
        """Ease of Movement計算 - Numba最適化版"""
        n = len(high)
        if n < 2:
            return np.zeros(n)
        
        eom = np.zeros(n)
        
        for i in range(1, n):
            # Distance Moved
            distance_moved = ((high[i] + low[i]) / 2.0) - ((high[i-1] + low[i-1]) / 2.0)
            
            # Box Height
            if volume[i] > 1e-10:
                box_height = volume[i] / (high[i] - low[i]) if high[i] - low[i] > 1e-10 else volume[i]
                eom[i] = scale * distance_moved / box_height
            else:
                eom[i] = 0.0
        
        return eom

    def calculate_ease_of_movement_features(self, high: np.ndarray, low: np.ndarray, volume: np.ndarray) -> Dict[str, np.ndarray]:
        """Ease of Movement特徴量の計算"""
        features = {}
        
        eom = self.robust_calculation_wrapper(self._compute_ease_of_movement_numba, high, low, volume)
        
        features['ease_of_movement'] = eom
        features['eom_positive'] = (eom > 0).astype(float)
        features['eom_momentum'] = np.gradient(eom)
        
        # EMVの移動平均
        for period in [14, 20]:
            eom_ma = self.robust_calculation_wrapper(self._compute_rolling_mean_numba, eom, period)
            features[f'eom_ma_{period}'] = eom_ma
            features[f'eom_signal_{period}'] = (eom > eom_ma).astype(float)
        
        return features

    def calculate_volume_roc_features(self, volume: np.ndarray) -> Dict[str, np.ndarray]:
        """出来高変化率特徴量の計算"""
        features = {}
        
        for period in self.params['vol_roc_periods']:
            if len(volume) > period:
                vol_roc = np.zeros(len(volume))
                for i in range(period, len(volume)):
                    if volume[i - period] > 1e-10:
                        vol_roc[i] = 100 * (volume[i] - volume[i - period]) / volume[i - period]
                    else:
                        vol_roc[i] = 0.0
                
                features[f'volume_roc_{period}'] = vol_roc
                features[f'vol_roc_positive_{period}'] = (vol_roc > 0).astype(float)
                features[f'vol_roc_strong_{period}'] = (np.abs(vol_roc) > 50).astype(float)
        
        return features

    # =============================================================================
    # 欠落特徴量補完 - Block 3: トレンド分析・移動平均線
    # =============================================================================

    @staticmethod
    @njit(cache=True)
    def _compute_wma_numba(data: np.ndarray, period: int) -> np.ndarray:
        """加重移動平均線（WMA）計算 - Numba最適化版"""
        n = len(data)
        if n < period:
            return np.zeros(n)
        
        wma = np.zeros(n)
        weight_sum = period * (period + 1) // 2  # 重みの合計
        
        for i in range(period - 1, n):
            weighted_sum = 0.0
            for j in range(period):
                weight = period - j  # 最新データの重みが最大
                weighted_sum += data[i - j] * weight
            wma[i] = weighted_sum / weight_sum
        
        return wma

    def calculate_wma_features(self, close: np.ndarray) -> Dict[str, np.ndarray]:
        """WMA特徴量の計算"""
        features = {}
        
        periods = [9, 21, 50]
        for period in periods:
            wma = self.robust_calculation_wrapper(self._compute_wma_numba, close, period)
            
            features[f'wma_{period}'] = wma
            features[f'price_above_wma_{period}'] = (close > wma).astype(float)
            features[f'wma_slope_{period}'] = np.gradient(wma)
            features[f'wma_distance_{period}'] = (close - wma) / close
        
        return features

    @staticmethod
    @njit(cache=True)
    def _compute_hma_numba(data: np.ndarray, period: int) -> np.ndarray:
        """ハル移動平均線（HMA）計算 - Numba最適化版"""
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

    def calculate_hma_features(self, close: np.ndarray) -> Dict[str, np.ndarray]:
        """HMA特徴量の計算"""
        features = {}
        
        periods = [14, 21, 50]
        for period in periods:
            hma = self.robust_calculation_wrapper(self._compute_hma_numba, close, period)
            
            features[f'hma_{period}'] = hma
            features[f'price_above_hma_{period}'] = (close > hma).astype(float)
            features[f'hma_slope_{period}'] = np.gradient(hma)
            features[f'hma_momentum_{period}'] = np.gradient(np.gradient(hma))
        
        return features

    @staticmethod
    @njit(cache=True)
    def _compute_tma_numba(data: np.ndarray, period: int) -> np.ndarray:
        """三角移動平均線（TMA）計算 - Numba最適化版"""
        n = len(data)
        if n < period:
            return np.zeros(n)
        
        # 最初にSMAを計算
        sma = np.zeros(n)
        for i in range(period - 1, n):
            sma[i] = np.mean(data[i - period + 1:i + 1])
        
        # SMAのSMAを計算（これがTMA）
        tma = np.zeros(n)
        for i in range(period + period - 2, n):
            if i >= period - 1:
                tma[i] = np.mean(sma[i - period + 1:i + 1])
        
        return tma

    def calculate_tma_features(self, close: np.ndarray) -> Dict[str, np.ndarray]:
        """TMA特徴量の計算"""
        features = {}
        
        for period in self.params['tma_periods']:
            tma = self.robust_calculation_wrapper(self._compute_tma_numba, close, period)
            
            features[f'tma_{period}'] = tma
            features[f'price_above_tma_{period}'] = (close > tma).astype(float)
            features[f'tma_slope_{period}'] = np.gradient(tma)
            features[f'tma_curvature_{period}'] = np.gradient(np.gradient(tma))
        
        return features

    @staticmethod
    @njit(cache=True)
    def _compute_kama_numba(data: np.ndarray, period: int = 14, fast_sc: float = 2.0, slow_sc: float = 30.0) -> np.ndarray:
        """適応型移動平均線（KAMA）計算 - Numba最適化版"""
        n = len(data)
        if n < period + 1:
            return np.zeros(n)
        
        kama = np.zeros(n)
        kama[period] = np.mean(data[:period + 1])
        
        fast_alpha = 2.0 / (fast_sc + 1.0)
        slow_alpha = 2.0 / (slow_sc + 1.0)
        
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
            
            # Smoothing Constant
            sc = er * (fast_alpha - slow_alpha) + slow_alpha
            alpha = sc * sc
            
            # KAMA計算
            kama[i] = kama[i - 1] + alpha * (data[i] - kama[i - 1])
        
        return kama

    def calculate_kama_features(self, close: np.ndarray) -> Dict[str, np.ndarray]:
        """KAMA特徴量の計算"""
        features = {}
        
        periods = [14, 21, 30]
        for period in periods:
            kama = self.robust_calculation_wrapper(self._compute_kama_numba, close, period)
            
            features[f'kama_{period}'] = kama
            features[f'price_above_kama_{period}'] = (close > kama).astype(float)
            features[f'kama_efficiency_{period}'] = self._calculate_efficiency_ratio(close, period)
            features[f'kama_slope_{period}'] = np.gradient(kama)
        
        return features

    @staticmethod
    @njit(cache=True)
    def _calculate_efficiency_ratio_numba(data: np.ndarray, period: int) -> np.ndarray:
        """効率比計算 - Numba最適化版"""
        n = len(data)
        if n < period + 1:
            return np.zeros(n)
        
        er = np.zeros(n)
        
        for i in range(period, n):
            direction = abs(data[i] - data[i - period])
            volatility = 0.0
            
            for j in range(period):
                volatility += abs(data[i - j] - data[i - j - 1])
            
            if volatility > 1e-10:
                er[i] = direction / volatility
            else:
                er[i] = 1.0
        
        return er

    def _calculate_efficiency_ratio(self, data: np.ndarray, period: int) -> np.ndarray:
        """効率比の外部インターフェース"""
        return self.robust_calculation_wrapper(self._calculate_efficiency_ratio_numba, data, period)

    @staticmethod
    @njit(cache=True)
    def _compute_zlema_numba(data: np.ndarray, period: int) -> np.ndarray:
        """ゼロラグ指数移動平均線（ZLEMA）計算 - Numba最適化版"""
        n = len(data)
        if n < period:
            return np.zeros(n)
        
        lag = (period - 1) // 2
        zlema = np.zeros(n)
        alpha = 2.0 / (period + 1.0)
        
        # 修正価格の計算
        adjusted_data = np.zeros(n)
        for i in range(n):
            if i >= lag:
                adjusted_data[i] = data[i] + (data[i] - data[i - lag])
            else:
                adjusted_data[i] = data[i]
        
        # ZLEMA計算
        zlema[0] = adjusted_data[0]
        for i in range(1, n):
            zlema[i] = alpha * adjusted_data[i] + (1 - alpha) * zlema[i - 1]
        
        return zlema

    def calculate_zlema_features(self, close: np.ndarray) -> Dict[str, np.ndarray]:
        """ZLEMA特徴量の計算"""
        features = {}
        
        for period in self.params['zlema_periods']:
            zlema = self.robust_calculation_wrapper(self._compute_zlema_numba, close, period)
            
            features[f'zlema_{period}'] = zlema
            features[f'price_above_zlema_{period}'] = (close > zlema).astype(float)
            features[f'zlema_momentum_{period}'] = np.gradient(zlema)
            features[f'zlema_distance_{period}'] = (close - zlema) / close
        
        return features

    @staticmethod
    @njit(cache=True)
    def _compute_dema_numba(data: np.ndarray, period: int) -> np.ndarray:
        """二重指数移動平均線（DEMA）計算 - Numba最適化版"""
        n = len(data)
        if n < 2:
            return np.zeros(n)
        
        alpha = 2.0 / (period + 1.0)
        
        # 第1段EMA
        ema1 = np.zeros(n)
        ema1[0] = data[0]
        for i in range(1, n):
            ema1[i] = alpha * data[i] + (1 - alpha) * ema1[i - 1]
        
        # 第2段EMA
        ema2 = np.zeros(n)
        ema2[0] = ema1[0]
        for i in range(1, n):
            ema2[i] = alpha * ema1[i] + (1 - alpha) * ema2[i - 1]
        
        # DEMA = 2*EMA1 - EMA2
        dema = 2 * ema1 - ema2
        
        return dema

    def calculate_dema_features(self, close: np.ndarray) -> Dict[str, np.ndarray]:
        """DEMA特徴量の計算"""
        features = {}
        
        for period in self.params['dema_periods']:
            dema = self.robust_calculation_wrapper(self._compute_dema_numba, close, period)
            
            features[f'dema_{period}'] = dema
            features[f'price_above_dema_{period}'] = (close > dema).astype(float)
            features[f'dema_slope_{period}'] = np.gradient(dema)
            features[f'dema_acceleration_{period}'] = np.gradient(np.gradient(dema))
        
        return features

    @staticmethod
    @njit(cache=True)
    def _compute_tema_numba(data: np.ndarray, period: int) -> np.ndarray:
        """三重指数移動平均線（TEMA）計算 - Numba最適化版"""
        n = len(data)
        if n < 3:
            return np.zeros(n)
        
        alpha = 2.0 / (period + 1.0)
        
        # 第1段EMA
        ema1 = np.zeros(n)
        ema1[0] = data[0]
        for i in range(1, n):
            ema1[i] = alpha * data[i] + (1 - alpha) * ema1[i - 1]
        
        # 第2段EMA
        ema2 = np.zeros(n)
        ema2[0] = ema1[0]
        for i in range(1, n):
            ema2[i] = alpha * ema1[i] + (1 - alpha) * ema2[i - 1]
        
        # 第3段EMA
        ema3 = np.zeros(n)
        ema3[0] = ema2[0]
        for i in range(1, n):
            ema3[i] = alpha * ema2[i] + (1 - alpha) * ema3[i - 1]
        
        # TEMA = 3*EMA1 - 3*EMA2 + EMA3
        tema = 3 * ema1 - 3 * ema2 + ema3
        
        return tema

    def calculate_tema_features(self, close: np.ndarray) -> Dict[str, np.ndarray]:
        """TEMA特徴量の計算"""
        features = {}
        
        for period in self.params['tema_periods']:
            tema = self.robust_calculation_wrapper(self._compute_tema_numba, close, period)
            
            features[f'tema_{period}'] = tema
            features[f'price_above_tema_{period}'] = (close > tema).astype(float)
            features[f'tema_momentum_{period}'] = np.gradient(tema)
            features[f'tema_trend_strength_{period}'] = np.abs(np.gradient(tema))
        
        return features

    @staticmethod
    @njit(cache=True)
    def _compute_ema_numba(data: np.ndarray, period: int) -> np.ndarray:
        """指数移動平均をNumbaで計算"""
        n = len(data)
        if n < 1:
            return np.zeros(n)
        
        alpha = 2.0 / (period + 1.0)
        ema = np.zeros(n)
        ema[0] = data[0]
        
        for i in range(1, n):
            ema[i] = alpha * data[i] + (1 - alpha) * ema[i - 1]
        
        return ema

    def calculate_ma_slope_features(self, close: np.ndarray) -> Dict[str, np.ndarray]:
        """移動平均線の傾き特徴量"""
        features = {}
        
        periods = [10, 20, 50, 200]
        for period in periods:
            if len(close) > period:
                sma = self.robust_calculation_wrapper(self._compute_rolling_mean_numba, close, period)
                ema = self.robust_calculation_wrapper(self._compute_ema_numba, close, period)
                
                features[f'sma_slope_{period}'] = np.gradient(sma)
                features[f'ema_slope_{period}'] = np.gradient(ema)
                
                # 傾きの強度
                features[f'sma_slope_strength_{period}'] = np.abs(np.gradient(sma))
                features[f'ema_slope_strength_{period}'] = np.abs(np.gradient(ema))
                
                # 傾きの方向
                features[f'sma_uptrend_{period}'] = (np.gradient(sma) > 0).astype(float)
                features[f'ema_uptrend_{period}'] = (np.gradient(ema) > 0).astype(float)
        
        return features

    def calculate_ma_deviation_features(self, close: np.ndarray) -> Dict[str, np.ndarray]:
        """移動平均線からの乖離率特徴量"""
        features = {}
        
        for period in self.params['ma_deviation_periods']:
            if len(close) > period:
                sma = self.robust_calculation_wrapper(self._compute_rolling_mean_numba, close, period)
                ema = self.robust_calculation_wrapper(self._compute_ema_numba, close, period)
                
                # 乖離率
                sma_deviation = (close - sma) / (sma + 1e-10) * 100
                ema_deviation = (close - ema) / (ema + 1e-10) * 100
                
                features[f'sma_deviation_{period}'] = sma_deviation
                features[f'ema_deviation_{period}'] = ema_deviation
                
                # 絶対乖離率
                features[f'sma_abs_deviation_{period}'] = np.abs(sma_deviation)
                features[f'ema_abs_deviation_{period}'] = np.abs(ema_deviation)
                
                # 過大乖離検出
                features[f'sma_excessive_deviation_{period}'] = (np.abs(sma_deviation) > 5).astype(float)
                features[f'ema_excessive_deviation_{period}'] = (np.abs(ema_deviation) > 5).astype(float)
        
        return features

    def calculate_golden_death_cross_features(self, close: np.ndarray) -> Dict[str, np.ndarray]:
        """ゴールデンクロス・デッドクロス特徴量"""
        features = {}
        
        # 短期・長期MA組み合わせ
        cross_pairs = [(25, 75), (50, 200), (20, 60)]
        
        for short_period, long_period in cross_pairs:
            if len(close) > long_period:
                short_ma = self.robust_calculation_wrapper(self._compute_rolling_mean_numba, close, short_period)
                long_ma = self.robust_calculation_wrapper(self._compute_rolling_mean_numba, close, long_period)
                
                # 現在のクロス状態
                features[f'golden_cross_{short_period}_{long_period}'] = (short_ma > long_ma).astype(float)
                
                # クロス発生検出
                cross_signal = np.diff((short_ma > long_ma).astype(int), prepend=0)
                features[f'golden_cross_signal_{short_period}_{long_period}'] = (cross_signal == 1).astype(float)
                features[f'death_cross_signal_{short_period}_{long_period}'] = (cross_signal == -1).astype(float)
                
                # MA間距離
                ma_distance = (short_ma - long_ma) / (long_ma + 1e-10) * 100
                features[f'ma_distance_{short_period}_{long_period}'] = ma_distance
                features[f'ma_convergence_{short_period}_{long_period}'] = np.abs(ma_distance)
        
        return features

    # =============================================================================
    # 欠落特徴量補完 - Block 4: ボラティリティ・バンド指標
    # =============================================================================

    @staticmethod
    @njit(cache=True)
    def _compute_rolling_std_numba(data: np.ndarray, window: int) -> np.ndarray:
        """ローリング標準偏差をNumbaで計算"""
        n = len(data)
        result = np.zeros(n)
        
        for i in range(window - 1, n):
            window_data = data[i - window + 1:i + 1]
            result[i] = np.std(window_data)
        
        return result

    @staticmethod
    @njit(cache=True)
    def _compute_rolling_rank_numba(data: np.ndarray, window: int) -> np.ndarray:
        """ローリングランク（百分位数）をNumbaで計算"""
        n = len(data)
        result = np.zeros(n)
        
        for i in range(window - 1, n):
            window_data = data[i - window + 1:i + 1]
            current_value = data[i]
            
            # 現在値以下の値の数を数える
            rank = 0
            for val in window_data:
                if val <= current_value:
                    rank += 1
            
            result[i] = rank / len(window_data)
        
        return result

    def calculate_bb_squeeze_features(self, close: np.ndarray, high: np.ndarray, low: np.ndarray) -> Dict[str, np.ndarray]:
        """ボリンジャーバンドスクイーズ特徴量"""
        features = {}
        
        for period, std_mult in self.params['volatility_bb_settings']:
            bb_upper, bb_mid, bb_lower = self.robust_calculation_wrapper(
                self._compute_bollinger_bands_numba, close, period, std_mult
            )
            
            # バンド幅
            bb_width = (bb_upper - bb_lower) / (bb_mid + 1e-10)
            bb_width_ma = self.robust_calculation_wrapper(self._compute_rolling_mean_numba, bb_width, 20)
            
            # スクイーズ検出（バンド幅が平均より小さい）
            features[f'bb_squeeze_{period}'] = (bb_width < bb_width_ma).astype(float)
            features[f'bb_width_{period}'] = bb_width
            features[f'bb_width_percentile_{period}'] = self.robust_calculation_wrapper(self._compute_rolling_rank_numba, bb_width, 100)
            
            # エクスパンション検出（バンド幅急拡大）
            bb_width_change = np.gradient(bb_width)
            features[f'bb_expansion_{period}'] = (bb_width_change > np.std(bb_width_change)).astype(float)
        
        return features

    @staticmethod
    @njit(cache=True)
    def _compute_keltner_channel_numba(high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int, atr_mult: float = 2.0) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """ケルトナーチャネル計算 - Numba最適化版"""
        n = len(high)
        if n < period:
            return np.zeros(n), np.zeros(n), np.zeros(n)
        
        # EMA計算
        alpha = 2.0 / (period + 1.0)
        ema = np.zeros(n)
        ema[0] = close[0]
        
        for i in range(1, n):
            ema[i] = alpha * close[i] + (1 - alpha) * ema[i-1]
        
        # ATR計算
        true_range = np.zeros(n)
        for i in range(1, n):
            hl = high[i] - low[i]
            hc = abs(high[i] - close[i-1])
            lc = abs(low[i] - close[i-1])
            true_range[i] = max(hl, max(hc, lc))
        
        atr = np.zeros(n)
        atr[0] = true_range[0]
        for i in range(1, n):
            atr[i] = alpha * true_range[i] + (1 - alpha) * atr[i-1]
        
        # ケルトナーチャネル
        kc_upper = ema + atr_mult * atr
        kc_middle = ema
        kc_lower = ema - atr_mult * atr
        
        return kc_upper, kc_middle, kc_lower

    def calculate_keltner_channel_features(self, high: np.ndarray, low: np.ndarray, close: np.ndarray) -> Dict[str, np.ndarray]:
        """ケルトナーチャネル特徴量"""
        features = {}
        
        for period in self.params['kc_periods']:
            kc_upper, kc_middle, kc_lower = self.robust_calculation_wrapper(
                self._compute_keltner_channel_numba, high, low, close, period
            )
            
            features[f'kc_upper_{period}'] = kc_upper
            features[f'kc_middle_{period}'] = kc_middle
            features[f'kc_lower_{period}'] = kc_lower
            
            # ケルトナーチャネル内の価格位置
            features[f'kc_position_{period}'] = (close - kc_lower) / (kc_upper - kc_lower + 1e-8)
            features[f'kc_width_{period}'] = (kc_upper - kc_lower) / (kc_middle + 1e-10)
            
            # ブレイクアウト検出
            features[f'kc_upper_break_{period}'] = (close > kc_upper).astype(float)
            features[f'kc_lower_break_{period}'] = (close < kc_lower).astype(float)
            features[f'kc_inside_{period}'] = ((close >= kc_lower) & (close <= kc_upper)).astype(float)
        
        return features

    @staticmethod
    @njit(cache=True)
    def _compute_donchian_channel_numba(high: np.ndarray, low: np.ndarray, period: int) -> Tuple[np.ndarray, np.ndarray]:
        """ドンチャンチャネル計算 - Numba最適化版"""
        n = len(high)
        if n < period:
            return np.zeros(n), np.zeros(n)
        
        dc_upper = np.zeros(n)
        dc_lower = np.zeros(n)
        
        for i in range(period - 1, n):
            dc_upper[i] = np.max(high[i - period + 1:i + 1])
            dc_lower[i] = np.min(low[i - period + 1:i + 1])
        
        return dc_upper, dc_lower

    def calculate_donchian_channel_features(self, high: np.ndarray, low: np.ndarray, close: np.ndarray) -> Dict[str, np.ndarray]:
        """ドンチャンチャネル特徴量"""
        features = {}
        
        for period in self.params['dc_periods']:
            dc_upper, dc_lower = self.robust_calculation_wrapper(
                self._compute_donchian_channel_numba, high, low, period
            )
            
            dc_middle = (dc_upper + dc_lower) / 2.0
            
            features[f'dc_upper_{period}'] = dc_upper
            features[f'dc_lower_{period}'] = dc_lower
            features[f'dc_middle_{period}'] = dc_middle
            
            # ドンチャンチャネル内の価格位置
            features[f'dc_position_{period}'] = (close - dc_lower) / (dc_upper - dc_lower + 1e-8)
            features[f'dc_width_{period}'] = (dc_upper - dc_lower) / (dc_middle + 1e-10)
            
            # ブレイクアウト検出
            features[f'dc_upper_break_{period}'] = (close > dc_upper).astype(float)
            features[f'dc_lower_break_{period}'] = (close < dc_lower).astype(float)
            
            # 新高値・新安値からの日数
            features[f'dc_days_since_high_{period}'] = self._calculate_days_since_extreme(close, dc_upper, True)
            features[f'dc_days_since_low_{period}'] = self._calculate_days_since_extreme(close, dc_lower, False)
        
        return features

    def _calculate_days_since_extreme(self, close: np.ndarray, extreme_line: np.ndarray, is_high: bool) -> np.ndarray:
        """極値からの日数計算"""
        n = len(close)
        days_since = np.zeros(n)
        
        for i in range(1, n):
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

    @staticmethod
    @njit(cache=True)
    def _compute_atr_bands_numba(high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int, atr_mult: float) -> Tuple[np.ndarray, np.ndarray]:
        """ATRバンド計算 - Numba最適化版"""
        n = len(high)
        if n < period:
            return np.zeros(n), np.zeros(n)
        
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
        
        # ATRバンド
        atr_upper = close + atr_mult * atr
        atr_lower = close - atr_mult * atr
        
        return atr_upper, atr_lower

    def calculate_atr_bands_features(self, high: np.ndarray, low: np.ndarray, close: np.ndarray) -> Dict[str, np.ndarray]:
        """ATRバンド特徴量"""
        features = {}
        
        atr_multipliers = [1.0, 1.5, 2.0, 2.5]
        
        for period in self.params['atr_periods_vol']:
            for mult in atr_multipliers:
                atr_upper, atr_lower = self.robust_calculation_wrapper(
                    self._compute_atr_bands_numba, high, low, close, period, mult
                )
                
                features[f'atr_upper_{period}_{mult}'] = atr_upper
                features[f'atr_lower_{period}_{mult}'] = atr_lower
                features[f'atr_band_position_{period}_{mult}'] = (close - atr_lower) / (atr_upper - atr_lower + 1e-8)
                features[f'atr_band_width_{period}_{mult}'] = (atr_upper - atr_lower) / (close + 1e-10)
        
        return features

    def calculate_historical_volatility_features(self, close: np.ndarray) -> Dict[str, np.ndarray]:
        """ヒストリカルボラティリティ特徴量"""
        features = {}
        
        # 対数リターン計算
        log_returns = np.diff(np.log(close + 1e-10), prepend=0)
        log_returns[0] = 0  # 最初の値は0に設定
        
        for period in self.params['hist_vol_periods']:
            if len(log_returns) > period:
                # ローリング標準偏差
                rolling_vol = self.robust_calculation_wrapper(self._compute_rolling_std_numba, log_returns, period)
                
                # 年率化（252営業日ベース）
                annualized_vol = rolling_vol * np.sqrt(252)
                
                features[f'hist_vol_{period}'] = rolling_vol
                features[f'hist_vol_annualized_{period}'] = annualized_vol
                
                # ボラティリティの相対水準
                vol_percentile = self.robust_calculation_wrapper(self._compute_rolling_rank_numba, rolling_vol, 252)
                features[f'vol_percentile_{period}'] = vol_percentile
                
                # ボラティリティレジーム
                vol_ma = self.robust_calculation_wrapper(self._compute_rolling_mean_numba, rolling_vol, 50)
                features[f'vol_regime_{period}'] = (rolling_vol > vol_ma).astype(float)
        
        return features

    def calculate_volatility_ratio_features(self, close: np.ndarray) -> Dict[str, np.ndarray]:
        """ボラティリティレシオ特徴量"""
        features = {}
        
        log_returns = np.diff(np.log(close + 1e-10), prepend=0)
        log_returns[0] = 0
        
        # 短期・長期ボラティリティ比較
        vol_pairs = [(5, 20), (10, 30), (20, 60)]
        
        for short, long in vol_pairs:
            if len(log_returns) > long:
                short_vol = self.robust_calculation_wrapper(self._compute_rolling_std_numba, log_returns, short)
                long_vol = self.robust_calculation_wrapper(self._compute_rolling_std_numba, log_returns, long)
                
                vol_ratio = short_vol / (long_vol + 1e-10)
                
                features[f'volatility_ratio_{short}_{long}'] = vol_ratio
                features[f'vol_ratio_high_{short}_{long}'] = (vol_ratio > 1.2).astype(float)
                features[f'vol_ratio_low_{short}_{long}'] = (vol_ratio < 0.8).astype(float)
                
                # ボラティリティ変化率
                features[f'vol_change_{short}_{long}'] = np.gradient(vol_ratio)
        
        return features

    @staticmethod
    @njit(cache=True)
    def _compute_chandelier_exit_numba(high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int, mult: float) -> Tuple[np.ndarray, np.ndarray]:
        """シャンデリアエグジット計算 - Numba最適化版"""
        n = len(high)
        if n < period:
            return np.zeros(n), np.zeros(n)
        
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
        
        return chandelier_long, chandelier_short

    def calculate_chandelier_exit_features(self, high: np.ndarray, low: np.ndarray, close: np.ndarray) -> Dict[str, np.ndarray]:
        """シャンデリアエグジット特徴量"""
        features = {}
        
        periods = [20, 50]
        multipliers = [2.0, 3.0]
        
        for period in periods:
            for mult in multipliers:
                chandelier_long, chandelier_short = self.robust_calculation_wrapper(
                    self._compute_chandelier_exit_numba, high, low, close, period, mult
                )
                
                features[f'chandelier_long_{period}_{mult}'] = chandelier_long
                features[f'chandelier_short_{period}_{mult}'] = chandelier_short
                
                # エグジットシグナル
                features[f'chandelier_long_exit_{period}_{mult}'] = (close < chandelier_long).astype(float)
                features[f'chandelier_short_exit_{period}_{mult}'] = (close > chandelier_short).astype(float)
                
                # 価格との距離
                features[f'chandelier_long_distance_{period}_{mult}'] = (close - chandelier_long) / (close + 1e-10)
                features[f'chandelier_short_distance_{period}_{mult}'] = (chandelier_short - close) / (close + 1e-10)
        
        return features

    def calculate_volatility_breakout_features(self, high: np.ndarray, low: np.ndarray, close: np.ndarray) -> Dict[str, np.ndarray]:
        """ボラティリティブレイクアウト特徴量"""
        features = {}
        
        # 前日比変動率
        daily_range = (high - low) / (close + 1e-10)
        price_change = np.diff(close, prepend=close[0]) / (close + 1e-10)
        
        # 複数期間でのボラティリティ分析
        periods = [5, 10, 20, 50]
        
        for period in periods:
            if len(daily_range) > period:
                # 平均日足レンジ
                avg_range = self.robust_calculation_wrapper(self._compute_rolling_mean_numba, daily_range, period)
                range_expansion = daily_range / (avg_range + 1e-10)
                
                features[f'range_expansion_{period}'] = range_expansion
                features[f'high_range_day_{period}'] = (range_expansion > 1.5).astype(float)
                features[f'low_range_day_{period}'] = (range_expansion < 0.5).astype(float)
                
                # 価格変動の相対的大きさ
                price_change_std = self.robust_calculation_wrapper(self._compute_rolling_std_numba, price_change, period)
                price_change_z = price_change / (price_change_std + 1e-10)
                
                features[f'price_change_z_{period}'] = price_change_z
                features[f'price_breakout_{period}'] = (np.abs(price_change_z) > 2.0).astype(float)
        
        return features

    # =============================================================================
    # 欠落特徴量補完 - Block 5: サポート・レジスタンス・ローソク足
    # =============================================================================

    @staticmethod
    @njit(cache=True)
    def _compute_pivot_points_numba(high: np.ndarray, low: np.ndarray, close: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """ピボットポイント計算 - Numba最適化版"""
        n = len(high)
        if n < 2:
            return np.zeros(n), np.zeros(n), np.zeros(n), np.zeros(n), np.zeros(n)
        
        pivot = np.zeros(n)
        resistance1 = np.zeros(n)
        resistance2 = np.zeros(n)
        support1 = np.zeros(n)
        support2 = np.zeros(n)
        
        for i in range(1, n):
            # 前日の高値、安値、終値を使用
            prev_high = high[i-1]
            prev_low = low[i-1]
            prev_close = close[i-1]
            
            # ピボットポイント
            pivot[i] = (prev_high + prev_low + prev_close) / 3.0
            
            # レジスタンス・サポート
            resistance1[i] = 2 * pivot[i] - prev_low
            support1[i] = 2 * pivot[i] - prev_high
            resistance2[i] = pivot[i] + (prev_high - prev_low)
            support2[i] = pivot[i] - (prev_high - prev_low)
        
        return pivot, resistance1, resistance2, support1, support2

    def calculate_pivot_points_features(self, high: np.ndarray, low: np.ndarray, close: np.ndarray) -> Dict[str, np.ndarray]:
        """ピボットポイント特徴量"""
        features = {}
        
        pivot, r1, r2, s1, s2 = self.robust_calculation_wrapper(
            self._compute_pivot_points_numba, high, low, close
        )
        
        features['pivot_point'] = pivot
        features['resistance1'] = r1
        features['resistance2'] = r2
        features['support1'] = s1
        features['support2'] = s2
        
        # 現在価格との距離（パーセンテージ）
        features['distance_to_pivot'] = (close - pivot) / (close + 1e-10) * 100
        features['distance_to_r1'] = (r1 - close) / (close + 1e-10) * 100
        features['distance_to_r2'] = (r2 - close) / (close + 1e-10) * 100
        features['distance_to_s1'] = (close - s1) / (close + 1e-10) * 100
        features['distance_to_s2'] = (close - s2) / (close + 1e-10) * 100
        
        # レベル突破検出
        features['above_pivot'] = (close > pivot).astype(float)
        features['above_r1'] = (close > r1).astype(float)
        features['above_r2'] = (close > r2).astype(float)
        features['below_s1'] = (close < s1).astype(float)
        features['below_s2'] = (close < s2).astype(float)
        
        return features

    @staticmethod
    @njit(cache=True)
    def _compute_price_channels_numba(high: np.ndarray, low: np.ndarray, period: int) -> Tuple[np.ndarray, np.ndarray]:
        """プライスチャネル計算 - Numba最適化版"""
        n = len(high)
        if n < period:
            return np.zeros(n), np.zeros(n)
        
        channel_high = np.zeros(n)
        channel_low = np.zeros(n)
        
        for i in range(period - 1, n):
            channel_high[i] = np.max(high[i - period + 1:i + 1])
            channel_low[i] = np.min(low[i - period + 1:i + 1])
        
        return channel_high, channel_low

    def calculate_price_channels_features(self, high: np.ndarray, low: np.ndarray, close: np.ndarray) -> Dict[str, np.ndarray]:
        """プライスチャネル特徴量"""
        features = {}
        
        for period in self.params['price_channel_periods']:
            if len(high) > period:
                channel_high, channel_low = self.robust_calculation_wrapper(
                    self._compute_price_channels_numba, high, low, period
                )
                
                channel_mid = (channel_high + channel_low) / 2.0
                
                features[f'channel_high_{period}'] = channel_high
                features[f'channel_low_{period}'] = channel_low
                features[f'channel_mid_{period}'] = channel_mid
                
                # チャネル内の価格位置
                features[f'channel_position_{period}'] = (close - channel_low) / (channel_high - channel_low + 1e-8)
                features[f'channel_width_{period}'] = (channel_high - channel_low) / (channel_mid + 1e-10)
                
                # ブレイクアウト検出
                features[f'channel_breakout_up_{period}'] = (close > channel_high).astype(float)
                features[f'channel_breakout_down_{period}'] = (close < channel_low).astype(float)
        
        return features

    @staticmethod
    @njit(cache=True)
    def _compute_fibonacci_levels_numba(high: np.ndarray, low: np.ndarray, period: int) -> Tuple[np.ndarray, np.ndarray]:
        """フィボナッチレベル計算 - Numba最適化版"""
        n = len(high)
        if n < period:
            return np.zeros(n), np.zeros(n)
        
        swing_high = np.zeros(n)
        swing_low = np.zeros(n)
        
        for i in range(period - 1, n):
            swing_high[i] = np.max(high[i - period + 1:i + 1])
            swing_low[i] = np.min(low[i - period + 1:i + 1])
        
        return swing_high, swing_low

    def calculate_fibonacci_features(self, high: np.ndarray, low: np.ndarray, close: np.ndarray) -> Dict[str, np.ndarray]:
        """フィボナッチリトレースメント特徴量"""
        features = {}
        
        # 複数期間でのフィボナッチレベル計算
        periods = [20, 50, 100]
        fib_levels = [0.236, 0.382, 0.500, 0.618, 0.786]
        
        for period in periods:
            if len(high) > period:
                swing_high, swing_low = self.robust_calculation_wrapper(
                    self._compute_fibonacci_levels_numba, high, low, period
                )
                
                swing_range = swing_high - swing_low
                
                for level in fib_levels:
                    fib_retracement = swing_low + level * swing_range
                    fib_extension = swing_high + level * swing_range
                    
                    level_str = str(level).replace('.', '')
                    features[f'fib_retracement_{level_str}_{period}'] = fib_retracement
                    features[f'fib_extension_{level_str}_{period}'] = fib_extension
                    
                    # フィボナッチレベルとの近接度
                    distance_to_retracement = np.abs(close - fib_retracement) / (close + 1e-10)
                    features[f'near_fib_retracement_{level_str}_{period}'] = (distance_to_retracement < 0.01).astype(float)
        
        return features

    def calculate_support_resistance_features(self, high: np.ndarray, low: np.ndarray, close: np.ndarray) -> Dict[str, np.ndarray]:
        """サポート・レジスタンス特徴量"""
        features = {}
        
        lookback_periods = [20, 50, 100]
        
        for period in lookback_periods:
            if len(high) > period:
                nearest_resistance = np.zeros(len(high))
                nearest_support = np.zeros(len(high))
                resistance_strength = np.zeros(len(high))
                support_strength = np.zeros(len(high))
                
                for i in range(period - 1, len(high)):
                    window_high = high[i - period + 1:i + 1]
                    window_low = low[i - period + 1:i + 1]
                    current_price = close[i]
                    
                    # レジスタンス検出（現在価格より上の高値）
                    resistance_levels = window_high[window_high > current_price]
                    if len(resistance_levels) > 0:
                        nearest_resistance[i] = np.min(resistance_levels)
                        # レジスタンス強度（そのレベルでの反発回数）
                        resistance_strength[i] = np.sum(np.abs(window_high - nearest_resistance[i]) < 0.001 * current_price)
                    else:
                        nearest_resistance[i] = current_price * 1.02  # デフォルト値
                    
                    # サポート検出（現在価格より下の安値）
                    support_levels = window_low[window_low < current_price]
                    if len(support_levels) > 0:
                        nearest_support[i] = np.max(support_levels)
                        # サポート強度（そのレベルでの反発回数）
                        support_strength[i] = np.sum(np.abs(window_low - nearest_support[i]) < 0.001 * current_price)
                    else:
                        nearest_support[i] = current_price * 0.98  # デフォルト値
                
                features[f'nearest_resistance_{period}'] = nearest_resistance
                features[f'nearest_support_{period}'] = nearest_support
                features[f'resistance_distance_{period}'] = (nearest_resistance - close) / (close + 1e-10)
                features[f'support_distance_{period}'] = (close - nearest_support) / (close + 1e-10)
                features[f'resistance_strength_{period}'] = resistance_strength
                features[f'support_strength_{period}'] = support_strength
        
        return features

    # ローソク足パターン特徴量
    def calculate_candlestick_patterns(self, open_prices: np.ndarray, high: np.ndarray, low: np.ndarray, close: np.ndarray) -> Dict[str, np.ndarray]:
        """ローソク足パターン特徴量"""
        features = {}
        
        # 基本的なローソク足要素
        body_size = np.abs(close - open_prices)
        upper_shadow = high - np.maximum(open_prices, close)
        lower_shadow = np.minimum(open_prices, close) - low
        total_range = high - low
        
        # 正規化（全体レンジに対する比率）
        body_ratio = body_size / (total_range + 1e-8)
        upper_shadow_ratio = upper_shadow / (total_range + 1e-8)
        lower_shadow_ratio = lower_shadow / (total_range + 1e-8)
        
        features['body_size'] = body_size
        features['upper_shadow'] = upper_shadow
        features['lower_shadow'] = lower_shadow
        features['body_ratio'] = body_ratio
        features['upper_shadow_ratio'] = upper_shadow_ratio
        features['lower_shadow_ratio'] = lower_shadow_ratio
        
        # ローソク足の方向
        is_bullish = (close > open_prices).astype(float)
        features['is_bullish'] = is_bullish
        features['is_bearish'] = (close < open_prices).astype(float)
        
        # 十字線（Doji）パターン
        features['doji'] = (body_ratio < 0.1).astype(float)
        features['long_legged_doji'] = ((body_ratio < 0.1) & (upper_shadow_ratio > 0.3) & (lower_shadow_ratio > 0.3)).astype(float)
        features['dragonfly_doji'] = ((body_ratio < 0.1) & (upper_shadow_ratio < 0.1) & (lower_shadow_ratio > 0.3)).astype(float)
        features['gravestone_doji'] = ((body_ratio < 0.1) & (upper_shadow_ratio > 0.3) & (lower_shadow_ratio < 0.1)).astype(float)
        
        # ハンマー・ハンギングマン
        hammer_condition = (lower_shadow_ratio > 0.5) & (upper_shadow_ratio < 0.1) & (body_ratio < 0.3)
        features['hammer'] = (hammer_condition & is_bullish).astype(float)
        features['hanging_man'] = (hammer_condition & (1 - is_bullish)).astype(float)
        
        # シューティングスター・インバーテッドハンマー
        shooting_star_condition = (upper_shadow_ratio > 0.5) & (lower_shadow_ratio < 0.1) & (body_ratio < 0.3)
        features['shooting_star'] = (shooting_star_condition & (1 - is_bullish)).astype(float)
        features['inverted_hammer'] = (shooting_star_condition & is_bullish).astype(float)
        
        # マルボウズ（影のないローソク足）
        features['marubozu'] = ((upper_shadow_ratio < 0.05) & (lower_shadow_ratio < 0.05) & (body_ratio > 0.9)).astype(float)
        features['white_marubozu'] = (features['marubozu'] & is_bullish).astype(float)
        features['black_marubozu'] = (features['marubozu'] & (1 - is_bullish)).astype(float)
        
        # スピニングトップ
        features['spinning_top'] = ((body_ratio < 0.3) & (upper_shadow_ratio > 0.2) & (lower_shadow_ratio > 0.2)).astype(float)
        
        return features

    def calculate_multi_candle_patterns(self, open_prices: np.ndarray, high: np.ndarray, low: np.ndarray, close: np.ndarray) -> Dict[str, np.ndarray]:
        """複数ローソク足パターン特徴量"""
        features = {}
        n = len(close)
        
        if n < 2:
            return features
        
        # エンゴルフィング（包み足）
        bullish_engulfing = np.zeros(n)
        bearish_engulfing = np.zeros(n)
        
        for i in range(1, n):
            # 前日が陰線、当日が陽線で、当日実体が前日実体を完全に包む
            prev_body_top = max(open_prices[i-1], close[i-1])
            prev_body_bottom = min(open_prices[i-1], close[i-1])
            curr_body_top = max(open_prices[i], close[i])
            curr_body_bottom = min(open_prices[i], close[i])
            
            if (close[i-1] < open_prices[i-1] and close[i] > open_prices[i] and 
                curr_body_bottom < prev_body_bottom and curr_body_top > prev_body_top):
                bullish_engulfing[i] = 1.0
            
            if (close[i-1] > open_prices[i-1] and close[i] < open_prices[i] and 
                curr_body_bottom < prev_body_bottom and curr_body_top > prev_body_top):
                bearish_engulfing[i] = 1.0
        
        features['bullish_engulfing'] = bullish_engulfing
        features['bearish_engulfing'] = bearish_engulfing
        
        # はらみ足（Harami）
        bullish_harami = np.zeros(n)
        bearish_harami = np.zeros(n)
        
        for i in range(1, n):
            prev_body_top = max(open_prices[i-1], close[i-1])
            prev_body_bottom = min(open_prices[i-1], close[i-1])
            curr_body_top = max(open_prices[i], close[i])
            curr_body_bottom = min(open_prices[i], close[i])
            
            # 前日の実体内に当日の実体が完全に収まる
            if (curr_body_top < prev_body_top and curr_body_bottom > prev_body_bottom):
                if close[i-1] < open_prices[i-1] and close[i] > open_prices[i]:
                    bullish_harami[i] = 1.0
                elif close[i-1] > open_prices[i-1] and close[i] < open_prices[i]:
                    bearish_harami[i] = 1.0
        
        features['bullish_harami'] = bullish_harami
        features['bearish_harami'] = bearish_harami
        
        # ギャップ（窓開け）
        gap_up = np.zeros(n)
        gap_down = np.zeros(n)
        
        for i in range(1, n):
            prev_high = high[i-1]
            prev_low = low[i-1]
            curr_low = low[i]
            curr_high = high[i]
            
            if curr_low > prev_high:
                gap_up[i] = (curr_low - prev_high) / (prev_high + 1e-10)
            elif curr_high < prev_low:
                gap_down[i] = (prev_low - curr_high) / (prev_low + 1e-10)
        
        features['gap_up'] = gap_up
        features['gap_down'] = gap_down
        features['has_gap'] = ((gap_up > 0) | (gap_down > 0)).astype(float)
        
        return features

    @staticmethod
    @njit(cache=True)
    def _compute_rolling_percentile_numba(data: np.ndarray, window: int, percentile: float) -> np.ndarray:
        """ローリングパーセンタイルをNumbaで計算"""
        n = len(data)
        result = np.zeros(n)
        
        for i in range(window - 1, n):
            window_data = data[i - window + 1:i + 1]
            sorted_data = np.sort(window_data)
            idx = int(percentile / 100.0 * (len(sorted_data) - 1))
            result[i] = sorted_data[idx]
        
        return result

    def calculate_candle_strength_features(self, open_prices: np.ndarray, high: np.ndarray, low: np.ndarray, close: np.ndarray) -> Dict[str, np.ndarray]:
        """ローソク足強度特徴量"""
        features = {}
        
        # 基本要素
        body_size = np.abs(close - open_prices)
        total_range = high - low
        
        # 複数期間での相対的評価
        periods = [5, 10, 20]
        
        for period in periods:
            if len(body_size) > period:
                # 実体サイズの相対的大きさ
                body_percentile = self.robust_calculation_wrapper(self._compute_rolling_rank_numba, body_size, period)
                range_percentile = self.robust_calculation_wrapper(self._compute_rolling_rank_numba, total_range, period)
                
                features[f'body_percentile_{period}'] = body_percentile
                features[f'range_percentile_{period}'] = range_percentile
                
                # 強いローソク足の検出
                features[f'strong_candle_{period}'] = (body_percentile > 0.8).astype(float)
                features[f'weak_candle_{period}'] = (body_percentile < 0.2).astype(float)
                features[f'high_volatility_candle_{period}'] = (range_percentile > 0.8).astype(float)
        
        # ローソク足の連続性
        consecutive_bullish = np.zeros(len(close))
        consecutive_bearish = np.zeros(len(close))
        
        bull_count = 0
        bear_count = 0
        
        for i in range(len(close)):
            if close[i] > open_prices[i]:
                bull_count += 1
                bear_count = 0
            elif close[i] < open_prices[i]:
                bear_count += 1
                bull_count = 0
            else:
                bull_count = 0
                bear_count = 0
            
            consecutive_bullish[i] = bull_count
            consecutive_bearish[i] = bear_count
        
        features['consecutive_bullish'] = consecutive_bullish
        features['consecutive_bearish'] = consecutive_bearish
        features['long_bullish_streak'] = (consecutive_bullish >= 3).astype(float)
        features['long_bearish_streak'] = (consecutive_bearish >= 3).astype(float)
        
        return features

    # =============================================================================
    # 欠落特徴量補完 - Block 6: 数学・統計学応用特徴量
    # =============================================================================

    def calculate_distribution_fitting_features(self, data: np.ndarray, window_size: int = 50) -> Dict[str, np.ndarray]:
        """確率分布フィッティング特徴量"""
        features = {}
        n = len(data)
        
        # 各種分布パラメータ推定結果を格納
        normal_params = np.zeros((n, 2))  # (mu, sigma)
        gamma_params = np.zeros((n, 2))   # (shape, scale)
        beta_params = np.zeros((n, 2))    # (alpha, beta)
        t_params = np.zeros((n, 3))       # (df, loc, scale)
        
        for i in range(window_size-1, n):
            window = data[i-window_size+1:i+1]
            clean_window = window[np.isfinite(window)]
            
            if len(clean_window) < 10:  # 最小サンプル数
                continue
            
            try:
                # 正規分布フィッティング
                mu, sigma = stats.norm.fit(clean_window)
                normal_params[i] = [mu, sigma]
                
                # ガンマ分布フィッティング（正の値のみ）
                if np.all(clean_window > 0):
                    shape, loc, scale = stats.gamma.fit(clean_window, floc=0)
                    gamma_params[i] = [shape, scale]
                
                # ベータ分布フィッティング（0-1範囲に正規化）
                normalized_window = (clean_window - np.min(clean_window)) / (np.max(clean_window) - np.min(clean_window) + 1e-10)
                if np.all((normalized_window >= 0) & (normalized_window <= 1)) and len(np.unique(normalized_window)) > 5:
                    alpha, beta, loc, scale = stats.beta.fit(normalized_window)
                    beta_params[i] = [alpha, beta]
                
                # t分布フィッティング
                df, loc, scale = stats.t.fit(clean_window)
                t_params[i] = [df, loc, scale]
                
            except:
                # フィッティング失敗時はデフォルト値
                continue
        
        features[f'normal_mu_{window_size}'] = normal_params[:, 0]
        features[f'normal_sigma_{window_size}'] = normal_params[:, 1]
        features[f'gamma_shape_{window_size}'] = gamma_params[:, 0]
        features[f'gamma_scale_{window_size}'] = gamma_params[:, 1]
        features[f'beta_alpha_{window_size}'] = beta_params[:, 0]
        features[f'beta_beta_{window_size}'] = beta_params[:, 1]
        features[f't_df_{window_size}'] = t_params[:, 0]
        features[f't_loc_{window_size}'] = t_params[:, 1]
        features[f't_scale_{window_size}'] = t_params[:, 2]
        
        return features

    @staticmethod
    @njit(cache=True)
    def _compute_order_statistics_numba(data: np.ndarray, window_size: int) -> np.ndarray:
        """順序統計量計算 - Numba最適化版"""
        n = len(data)
        if n < window_size:
            return np.zeros((n, 6))
        
        results = np.zeros((n, 6))
        
        for i in range(window_size-1, n):
            window = data[i-window_size+1:i+1]
            sorted_window = np.sort(window)
            
            # 価格の順位（昇順）
            current_price = data[i]
            rank = 0
            for val in sorted_window:
                if val <= current_price:
                    rank += 1
            price_rank = rank / window_size
            
            # 分位数位置
            quantile_25 = sorted_window[int(window_size * 0.25)]
            quantile_50 = sorted_window[int(window_size * 0.50)]
            quantile_75 = sorted_window[int(window_size * 0.75)]
            
            if current_price <= quantile_25:
                quantile_pos = 0.25
            elif current_price <= quantile_50:
                quantile_pos = 0.50
            elif current_price <= quantile_75:
                quantile_pos = 0.75
            else:
                quantile_pos = 1.0
            
            # 極値比率（外れ値の比率）
            iqr = quantile_75 - quantile_25
            lower_bound = quantile_25 - 1.5 * iqr
            upper_bound = quantile_75 + 1.5 * iqr
            
            extreme_count = 0
            for val in window:
                if val < lower_bound or val > upper_bound:
                    extreme_count += 1
            extreme_ratio = extreme_count / window_size
            
            # 最小・最大値
            min_val = sorted_window[0]
            max_val = sorted_window[-1]
            
            # 範囲内での位置
            if max_val - min_val > 1e-10:
                range_position = (current_price - min_val) / (max_val - min_val)
            else:
                range_position = 0.5
            
            results[i] = [price_rank, quantile_pos, extreme_ratio, min_val, max_val, range_position]
        
        return results

    def calculate_order_statistics_features(self, data: np.ndarray) -> Dict[str, np.ndarray]:
        """順序統計量特徴量（修正版）"""
        features = {}
        n = len(data)  # データ長を事前に取得

        for window_size in self.params['order_stat_windows']:
            try:
                order_stats = self.robust_calculation_wrapper(
                    self._compute_order_statistics_numba, data, window_size
                )

                # 修正: robust_calculation_wrapperが失敗してfloat値を返した場合の処理
                if not isinstance(order_stats, np.ndarray):
                    logger.warning(f"順序統計量(window={window_size})の計算に失敗しました。フォールバック値を生成します。")
                    # 期待される形状のNaN配列を生成
                    order_stats = np.full((n, 6), np.nan)
                
                # 配列の形状チェック
                elif order_stats.shape[1] < 6:
                    logger.warning(f"順序統計量(window={window_size})の結果形状が不正です。フォールバック値を生成します。")
                    order_stats = np.full((n, 6), np.nan)
                
                # 基本的な順序統計量特徴量
                features[f'price_rank_{window_size}'] = order_stats[:, 0]
                features[f'quantile_pos_{window_size}'] = order_stats[:, 1]
                features[f'extreme_ratio_{window_size}'] = order_stats[:, 2]
                features[f'window_min_{window_size}'] = order_stats[:, 3]
                features[f'window_max_{window_size}'] = order_stats[:, 4]
                features[f'range_position_{window_size}'] = order_stats[:, 5]

                # 追加の順序統計量（安全なアクセス）
                try:
                    price_ranks = order_stats[:, 0]
                    extreme_ratios = order_stats[:, 2]
                    
                    # NaN/Inf値を考慮した比較
                    features[f'in_top_quartile_{window_size}'] = np.where(
                        np.isfinite(price_ranks), 
                        (price_ranks > 0.75).astype(float), 
                        0.0
                    )
                    features[f'in_bottom_quartile_{window_size}'] = np.where(
                        np.isfinite(price_ranks), 
                        (price_ranks < 0.25).astype(float), 
                        0.0
                    )
                    features[f'is_extreme_{window_size}'] = np.where(
                        np.isfinite(extreme_ratios), 
                        (extreme_ratios > 0.1).astype(float), 
                        0.0
                    )
                except Exception as e:
                    logger.warning(f"追加順序統計量計算エラー(window={window_size}): {e}")
                    # フォールバック値
                    features[f'in_top_quartile_{window_size}'] = np.zeros(n)
                    features[f'in_bottom_quartile_{window_size}'] = np.zeros(n)
                    features[f'is_extreme_{window_size}'] = np.zeros(n)

            except Exception as e:
                logger.error(f"順序統計量計算で予期しないエラー(window={window_size}): {e}")
                # 全特徴量にフォールバック値を設定
                feature_names = [
                    'price_rank', 'quantile_pos', 'extreme_ratio', 'window_min', 
                    'window_max', 'range_position', 'in_top_quartile', 
                    'in_bottom_quartile', 'is_extreme'
                ]
                for name in feature_names:
                    features[f'{name}_{window_size}'] = np.zeros(n)

        return features

    def calculate_normality_tests_features(self, data: np.ndarray, window_size: int = 50) -> Dict[str, np.ndarray]:
        """正規性検定特徴量"""
        features = {}
        n = len(data)
        
        # 検定統計量と p値を格納
        shapiro_stats = np.zeros(n)
        shapiro_pvals = np.zeros(n)
        jarque_bera_stats = np.zeros(n)
        jarque_bera_pvals = np.zeros(n)
        anderson_stats = np.zeros(n)
        
        for i in range(window_size-1, n):
            window = data[i-window_size+1:i+1]
            clean_window = window[np.isfinite(window)]
            
            if len(clean_window) < 10:
                continue
            
            try:
                # Shapiro-Wilk検定
                if len(clean_window) >= 3 and len(clean_window) <= 5000:
                    stat, pval = stats.shapiro(clean_window)
                    shapiro_stats[i] = stat
                    shapiro_pvals[i] = pval
                
                # Jarque-Bera検定
                if len(clean_window) >= 10:
                    stat, pval = stats.jarque_bera(clean_window)
                    jarque_bera_stats[i] = stat
                    jarque_bera_pvals[i] = pval
                
                # Anderson-Darling検定
                if len(clean_window) >= 7:
                    result = stats.anderson(clean_window, dist='norm')
                    anderson_stats[i] = result.statistic
                
            except:
                continue
        
        features[f'shapiro_stat_{window_size}'] = shapiro_stats
        features[f'shapiro_pval_{window_size}'] = shapiro_pvals
        features[f'jarque_bera_stat_{window_size}'] = jarque_bera_stats
        features[f'jarque_bera_pval_{window_size}'] = jarque_bera_pvals
        features[f'anderson_stat_{window_size}'] = anderson_stats
        
        # 正規性の判定（p値ベース）
        features[f'is_normal_shapiro_{window_size}'] = (shapiro_pvals > 0.05).astype(float)
        features[f'is_normal_jb_{window_size}'] = (jarque_bera_pvals > 0.05).astype(float)
        
        return features

    @staticmethod
    @njit(cache=True)
    def _compute_autocorrelation_numba(data: np.ndarray, max_lags: int) -> np.ndarray:
        """自己相関計算 - Numba最適化版"""
        n = len(data)
        if n < max_lags + 10:
            return np.zeros(max_lags)
        
        # データの中心化
        mean_data = np.mean(data)
        centered_data = data - mean_data
        
        autocorr = np.zeros(max_lags)
        
        # ラグ0の分散
        var0 = np.sum(centered_data ** 2) / n
        
        for lag in range(1, max_lags + 1):
            if lag < n:
                # ラグkの自己共分散
                covariance = 0.0
                count = n - lag
                
                for i in range(count):
                    covariance += centered_data[i] * centered_data[i + lag]
                
                covariance /= count
                
                # 自己相関係数
                if var0 > 1e-10:
                    autocorr[lag - 1] = covariance / var0
        
        return autocorr

    def calculate_autocorrelation_features(self, data: np.ndarray, window_size: int = 100) -> Dict[str, np.ndarray]:
        """自己相関特徴量（修正版）"""
        features = {}
        n = len(data)
        
        max_lags = min(20, window_size // 4)  # 最大ラグ数
        if max_lags == 0:
            return {}  # ラグが計算できない場合は空の辞書を返す

        # 修正: ループの前に特徴量配列を無条件で初期化
        for lag in range(1, max_lags + 1):
            features[f'autocorr_lag_{lag}'] = np.zeros(n)
        
        # 要約統計用の特徴量も事前に初期化
        features['autocorr_mean'] = np.zeros(n)
        features['autocorr_max'] = np.zeros(n)
        features['autocorr_sum'] = np.zeros(n)

        try:
            for i in range(window_size-1, n):
                window = data[i-window_size+1:i+1]
                
                try:
                    autocorr = self.robust_calculation_wrapper(
                        self._compute_autocorrelation_numba, window, max_lags
                    )

                    # 修正: autocorrが配列でない場合や長さが不正な場合のフォールバック
                    if not isinstance(autocorr, np.ndarray):
                        logger.debug(f"自己相関計算失敗(index={i}): 戻り値が配列ではありません")
                        continue  # このウィンドウの処理をスキップ
                    
                    if len(autocorr) != max_lags:
                        logger.debug(f"自己相関計算失敗(index={i}): 戻り値の長さが不正です (期待: {max_lags}, 実際: {len(autocorr)})")
                        continue  # このウィンドウの処理をスキップ
                    
                    # NaN/Inf値のチェック
                    if np.any(np.isnan(autocorr)) or np.any(np.isinf(autocorr)):
                        logger.debug(f"自己相関計算結果に無効値が含まれています(index={i})")
                        # NaN/Infを0で置換
                        autocorr = np.where(np.isfinite(autocorr), autocorr, 0.0)
                    
                    # 各ラグの自己相関を格納
                    for lag in range(max_lags):
                        features[f'autocorr_lag_{lag+1}'][i] = autocorr[lag]
                    
                    # 要約統計の計算
                    valid_autocorr = autocorr[np.isfinite(autocorr)]
                    if len(valid_autocorr) > 0:
                        features['autocorr_mean'][i] = np.mean(valid_autocorr)
                        features['autocorr_max'][i] = np.max(np.abs(valid_autocorr))
                        features['autocorr_sum'][i] = np.sum(np.abs(valid_autocorr))
                    else:
                        features['autocorr_mean'][i] = 0.0
                        features['autocorr_max'][i] = 0.0
                        features['autocorr_sum'][i] = 0.0
                        
                except Exception as e:
                    logger.debug(f"ウィンドウ{i}の自己相関計算でエラー: {e}")
                    continue  # このウィンドウの処理をスキップ

            # 最終的な自己相関の要約統計（全体的な計算）
            if max_lags > 0:
                try:
                    # 各ラグの自己相関値を集約
                    lag_features = []
                    for lag in range(1, max_lags + 1):
                        lag_values = features[f'autocorr_lag_{lag}']
                        # 有効値のみを使用
                        valid_values = lag_values[lag_values != 0]
                        if len(valid_values) > 0:
                            lag_features.append(lag_values)
                    
                    if lag_features:
                        autocorr_matrix = np.array(lag_features)
                        
                        # 全体統計を再計算（より精密に）
                        for i in range(n):
                            column_values = autocorr_matrix[:, i]
                            non_zero_values = column_values[column_values != 0]
                            
                            if len(non_zero_values) > 0:
                                features['autocorr_mean'][i] = np.mean(non_zero_values)
                                features['autocorr_max'][i] = np.max(np.abs(non_zero_values))
                                features['autocorr_sum'][i] = np.sum(np.abs(non_zero_values))
                            # else: 既に0で初期化済み
                            
                except Exception as e:
                    logger.warning(f"自己相関要約統計計算エラー: {e}")
                    # 要約統計が失敗してもラグ別の値は保持される

        except Exception as e:
            logger.error(f"自己相関特徴量計算で予期しないエラー: {e}")
            # エラーが発生しても、初期化済みの配列を返す
        
        return features

    @staticmethod
    @njit(cache=True)
    def _compute_cross_correlation_numba(x: np.ndarray, y: np.ndarray, window_size: int) -> float:
        """相互相関計算 - Numba最適化版"""
        if len(x) != len(y) or len(x) < window_size or window_size < 2:
            return 0.0
        
        mean_x = np.mean(x)
        mean_y = np.mean(y)
        
        numerator = 0.0
        sum_x_sq = 0.0
        sum_y_sq = 0.0
        
        for i in range(window_size):
            x_diff = x[i] - mean_x
            y_diff = y[i] - mean_y
            numerator += x_diff * y_diff
            sum_x_sq += x_diff * x_diff
            sum_y_sq += y_diff * y_diff
        
        denominator = np.sqrt(sum_x_sq * sum_y_sq)
        if denominator > 1e-10:
            return numerator / denominator
        else:
            return 0.0

    def calculate_cross_correlation_features(self, data1: np.ndarray, data2: np.ndarray, window_size: int = 50) -> Dict[str, np.ndarray]:
        """相互相関特徴量（価格と出来高など）"""
        features = {}
        n = min(len(data1), len(data2))
        
        max_lags = min(10, window_size // 5)
        
        # 各ラグでの相互相関
        for lag in range(-max_lags, max_lags + 1):
            cross_corr = np.zeros(n)
            
            for i in range(window_size-1, n):
                # ラグを考慮したウィンドウ取得
                if lag >= 0:
                    if i >= window_size - 1 + lag:
                        x = data1[i-window_size+1:i+1]
                        y = data2[i-window_size+1-lag:i+1-lag]
                        if len(x) == len(y) == window_size:
                            cross_corr[i] = self.robust_calculation_wrapper(
                                self._compute_cross_correlation_numba, x, y, window_size
                            )
                else:
                    abs_lag = abs(lag)
                    if i >= window_size - 1 + abs_lag:
                        x = data1[i-window_size+1+abs_lag:i+1+abs_lag]
                        y = data2[i-window_size+1:i+1]
                        if len(x) == len(y) == window_size:
                            cross_corr[i] = self.robust_calculation_wrapper(
                                self._compute_cross_correlation_numba, x, y, window_size
                            )
            
            lag_str = f'pos{lag}' if lag >= 0 else f'neg{abs(lag)}'
            features[f'cross_corr_lag_{lag_str}'] = cross_corr
        
        return features

    def calculate_advanced_moments_features(self, data: np.ndarray, window_size: int = 50) -> Dict[str, np.ndarray]:
        """高次モーメント・キュムラント特徴量"""
        features = {}
        n = len(data)
        
        # 高次モーメントとキュムラント
        moments_results = np.zeros((n, 8))  # 1-8次モーメント
        cumulants_results = np.zeros((n, 4))  # 1-4次キュムラント
        
        for i in range(window_size-1, n):
            window = data[i-window_size+1:i+1]
            clean_window = window[np.isfinite(window)]
            
            if len(clean_window) < 10:
                continue
            
            try:
                mean_val = np.mean(clean_window)
                std_val = np.std(clean_window)
                
                if std_val > 1e-10:
                    # 標準化データ
                    standardized = (clean_window - mean_val) / std_val
                    
                    # モーメント計算
                    moments_results[i, 0] = mean_val
                    moments_results[i, 1] = std_val ** 2  # 分散
                    
                    for moment_order in range(3, 9):
                        moments_results[i, moment_order-1] = np.mean(standardized ** moment_order)
                    
                    # キュムラント計算（近似）
                    cumulants_results[i, 0] = mean_val  # 1次キュムラント = 平均
                    cumulants_results[i, 1] = std_val ** 2  # 2次キュムラント = 分散
                    cumulants_results[i, 2] = stats.skew(clean_window)  # 3次キュムラント = 歪度
                    cumulants_results[i, 3] = stats.kurtosis(clean_window, fisher=True)  # 4次キュムラント = 超過尖度
            
            except:
                continue
        
        # モーメント特徴量
        for i in range(8):
            features[f'moment_{i+1}_{window_size}'] = moments_results[:, i]
        
        # キュムラント特徴量
        for i in range(4):
            features[f'cumulant_{i+1}_{window_size}'] = cumulants_results[:, i]
        
        # 高次モーメントの比率
        features[f'moment_ratio_6_4_{window_size}'] = np.where(
            np.abs(moments_results[:, 3]) > 1e-10,
            moments_results[:, 5] / moments_results[:, 3],
            0
        )
        
        features[f'moment_ratio_8_6_{window_size}'] = np.where(
            np.abs(moments_results[:, 5]) > 1e-10,
            moments_results[:, 7] / moments_results[:, 5],
            0
        )
        
        return features

    def calculate_hypothesis_test_features(self, data: np.ndarray, window_size: int = 50) -> Dict[str, np.ndarray]:
        """統計的仮説検定特徴量"""
        features = {}
        n = len(data)
        
        # 各種検定の結果
        t_test_stats = np.zeros(n)
        t_test_pvals = np.zeros(n)
        ks_test_stats = np.zeros(n)
        ks_test_pvals = np.zeros(n)
        
        for i in range(window_size-1, n):
            window = data[i-window_size+1:i+1]
            clean_window = window[np.isfinite(window)]
            
            if len(clean_window) < 10:
                continue
            
            try:
                # t検定（平均がゼロかどうか）
                t_stat, t_pval = stats.ttest_1samp(clean_window, 0)
                t_test_stats[i] = t_stat
                t_test_pvals[i] = t_pval
                
                # Kolmogorov-Smirnov検定（正規分布との適合）
                ks_stat, ks_pval = stats.kstest(clean_window, 'norm')
                ks_test_stats[i] = ks_stat
                ks_test_pvals[i] = ks_pval
                
            except:
                continue
        
        features[f't_test_stat_{window_size}'] = t_test_stats
        features[f't_test_pval_{window_size}'] = t_test_pvals
        features[f'ks_test_stat_{window_size}'] = ks_test_stats
        features[f'ks_test_pval_{window_size}'] = ks_test_pvals
        
        # 有意性の判定
        features[f'significant_t_test_{window_size}'] = (t_test_pvals < 0.05).astype(float)
        features[f'normal_ks_test_{window_size}'] = (ks_test_pvals > 0.05).astype(float)
        
        return features

    # =============================================================================
    # 欠落特徴量補完 - Block 7: 物理学・信号処理応用特徴量
    # =============================================================================

    def calculate_hilbert_transform_features(self, data: np.ndarray) -> Dict[str, np.ndarray]:
        """ヒルベルト変換特徴量"""
        features = {}
        
        try:
            # ヒルベルト変換
            analytic_signal = hilbert(data)
            amplitude_envelope = np.abs(analytic_signal)
            instantaneous_phase = np.angle(analytic_signal)
            instantaneous_frequency = np.diff(np.unwrap(instantaneous_phase), prepend=0)
            
            features['hilbert_amplitude'] = amplitude_envelope
            features['hilbert_phase'] = instantaneous_phase
            features['hilbert_frequency'] = instantaneous_frequency
            
            # 振幅と位相の統計
            for window in self.params['hilbert_windows']:
                if len(data) > window:
                    amp_rolling = self.robust_calculation_wrapper(self._compute_rolling_mean_numba, amplitude_envelope, window)
                    amp_std = self.robust_calculation_wrapper(self._compute_rolling_std_numba, amplitude_envelope, window)
                    phase_var = self.robust_calculation_wrapper(self._compute_rolling_var_numba, instantaneous_phase, window)
                    
                    features[f'hilbert_amp_mean_{window}'] = amp_rolling
                    features[f'hilbert_amp_std_{window}'] = amp_std
                    features[f'hilbert_phase_var_{window}'] = phase_var
                    
                    # 位相の安定性（位相差の標準偏差）
                    phase_diff = np.diff(instantaneous_phase, prepend=0)
                    phase_stability = self.robust_calculation_wrapper(self._compute_rolling_std_numba, phase_diff, window)
                    features[f'hilbert_phase_stability_{window}'] = phase_stability
        
        except Exception as e:
            logger.warning(f"ヒルベルト変換計算エラー: {e}")
            features['hilbert_amplitude'] = np.zeros(len(data))
            features['hilbert_phase'] = np.zeros(len(data))
            features['hilbert_frequency'] = np.zeros(len(data))
        
        return features

    @staticmethod
    @njit(cache=True)
    def _compute_rolling_var_numba(data: np.ndarray, window: int) -> np.ndarray:
        """ローリング分散をNumbaで計算"""
        n = len(data)
        result = np.zeros(n)
        
        for i in range(window - 1, n):
            window_data = data[i - window + 1:i + 1]
            result[i] = np.var(window_data)
        
        return result

    def calculate_coherence_features(self, data1: np.ndarray, data2: np.ndarray) -> Dict[str, np.ndarray]:
        """コヒーレンス特徴量（価格と出来高間など）"""
        features = {}
        
        try:
            # 長さを合わせる
            min_len = min(len(data1), len(data2))
            data1_trimmed = data1[:min_len]
            data2_trimmed = data2[:min_len]
            
            # ウィンドウサイズでのコヒーレンス計算
            window_sizes = [64, 128, 256]
            
            for window_size in window_sizes:
                if min_len >= window_size * 2:
                    try:
                        # コヒーレンス計算
                        freqs, coherence_vals = coherence(data1_trimmed, data2_trimmed, 
                                                        nperseg=window_size, noverlap=window_size//2)
                        
                        # 各周波数帯域でのコヒーレンス
                        low_freq_coherence = np.mean(coherence_vals[:len(coherence_vals)//4])
                        mid_freq_coherence = np.mean(coherence_vals[len(coherence_vals)//4:3*len(coherence_vals)//4])
                        high_freq_coherence = np.mean(coherence_vals[3*len(coherence_vals)//4:])
                        
                        # 全データ長で結果を拡張
                        features[f'coherence_low_freq_{window_size}'] = np.full(len(data1), low_freq_coherence)
                        features[f'coherence_mid_freq_{window_size}'] = np.full(len(data1), mid_freq_coherence)
                        features[f'coherence_high_freq_{window_size}'] = np.full(len(data1), high_freq_coherence)
                        features[f'coherence_mean_{window_size}'] = np.full(len(data1), np.mean(coherence_vals))
                        features[f'coherence_max_{window_size}'] = np.full(len(data1), np.max(coherence_vals))
                        
                    except Exception as e:
                        logger.debug(f"コヒーレンス計算エラー (window_size={window_size}): {e}")
                        continue
        
        except Exception as e:
            logger.warning(f"コヒーレンス計算全体エラー: {e}")
        
        return features

    @staticmethod
    @njit(cache=True)
    def _compute_zero_crossing_rate_numba(data: np.ndarray, window_size: int) -> np.ndarray:
        """ゼロ交差率計算 - Numba最適化版（修正済み）"""
        n = len(data)
        if n < window_size:
            return np.zeros(n)
        
        zcr = np.zeros(n)
        
        for i in range(window_size-1, n):
            window = data[i-window_size+1:i+1]
            
            mean_val = np.mean(window)
            centered_window = window - mean_val

            # ゼロ交差カウント（生データに対して実行）
            zero_crossings = 0
            for j in range(1, len(centered_window)):
                # 中心化されたデータで符号の変化をチェック
                if centered_window[j] * centered_window[j-1] < 0:
                    zero_crossings += 1
            
            # 正規化: ゼロ交差数を可能な最大交差数で割る
            zcr[i] = zero_crossings / (len(window) - 1) if len(window) > 1 else 0.0
        
        return zcr

    def calculate_zero_crossing_features(self, data: np.ndarray) -> Dict[str, np.ndarray]:
        """ゼロ交差率特徴量（修正版）"""
        features = {}
        n = len(data)

        window_sizes = [20, 50, 100]
        
        for window_size in window_sizes:
            try:
                zcr = self.robust_calculation_wrapper(
                    self._compute_zero_crossing_rate_numba, data, window_size
                )

                # 修正: 計算失敗時のフォールバック処理
                if not isinstance(zcr, np.ndarray):
                    logger.warning(f"ゼロ交差率(window={window_size})の計算に失敗しました。フォールバック値を生成します。")
                    features[f'zero_crossing_rate_{window_size}'] = np.zeros(n)
                    features[f'high_zcr_{window_size}'] = np.zeros(n)
                    continue
                
                # 配列長チェック
                if len(zcr) != n:
                    logger.warning(f"ゼロ交差率(window={window_size})の結果長が不正です。フォールバック値を生成します。")
                    features[f'zero_crossing_rate_{window_size}'] = np.zeros(n)
                    features[f'high_zcr_{window_size}'] = np.zeros(n)
                    continue

                # NaN/Inf値のクリーニング
                zcr_clean = np.where(np.isfinite(zcr), zcr, 0.0)
                features[f'zero_crossing_rate_{window_size}'] = zcr_clean
                
                # 修正: ゼロ交差率の統計的分析（安全な処理）
                try:
                    # 正の値のみを抽出
                    positive_zcr = zcr_clean[zcr_clean > 1e-10]  # 非常に小さな値も除外
                    
                    if len(positive_zcr) > 0:
                        # 中央値を計算
                        median_zcr = np.median(positive_zcr)
                        
                        # 閾値計算（ゼロ除算回避）
                        if median_zcr > 1e-10:
                            threshold = median_zcr * 1.5
                            features[f'high_zcr_{window_size}'] = (zcr_clean > threshold).astype(float)
                        else:
                            # 中央値が極めて小さい場合
                            features[f'high_zcr_{window_size}'] = np.zeros(n)
                            
                    else:
                        # ゼロ交差が一度も発生しなかった場合
                        logger.debug(f"ウィンドウサイズ{window_size}でゼロ交差が検出されませんでした")
                        features[f'high_zcr_{window_size}'] = np.zeros(n)
                        
                except Exception as e:
                    logger.warning(f"ゼロ交差率統計計算エラー(window={window_size}): {e}")
                    features[f'high_zcr_{window_size}'] = np.zeros(n)

            except Exception as e:
                logger.error(f"ゼロ交差率特徴量計算で予期しないエラー(window={window_size}): {e}")
                # 全特徴量にフォールバック値を設定
                features[f'zero_crossing_rate_{window_size}'] = np.zeros(n)
                features[f'high_zcr_{window_size}'] = np.zeros(n)
        
        # 追加: 複数ウィンドウサイズでの統合特徴量
        try:
            if len(window_sizes) > 1:
                # 各ウィンドウサイズのゼロ交差率を集約
                all_zcr_features = []
                for ws in window_sizes:
                    feature_key = f'zero_crossing_rate_{ws}'
                    if feature_key in features:
                        all_zcr_features.append(features[feature_key])
                
                if all_zcr_features:
                    zcr_matrix = np.array(all_zcr_features)
                    
                    # 統合特徴量の計算
                    features['zcr_mean_across_windows'] = np.mean(zcr_matrix, axis=0)
                    features['zcr_std_across_windows'] = np.std(zcr_matrix, axis=0)
                    features['zcr_max_across_windows'] = np.max(zcr_matrix, axis=0)
                    
                    # 一貫性指標（標準偏差が小さい = 一貫している）
                    features['zcr_consistency'] = 1.0 / (1.0 + features['zcr_std_across_windows'])
                    
        except Exception as e:
            logger.warning(f"ゼロ交差率統合特徴量計算エラー: {e}")
            # 統合特徴量が失敗しても、個別ウィンドウの特徴量は保持される
        
        return features

    def calculate_filter_features(self, data: np.ndarray) -> Dict[str, np.ndarray]:
        """フィルタリング特徴量"""
        features = {}
        
        try:
            # ガウシアンフィルタ
            for sigma in self.params['gaussian_sigmas']:
                filtered = gaussian_filter(data, sigma=sigma)
                residual = data - filtered
                
                features[f'gaussian_filtered_{sigma}'] = filtered
                features[f'gaussian_residual_{sigma}'] = residual
                features[f'gaussian_residual_energy_{sigma}'] = np.cumsum(residual**2)
            
            # メディアンフィルタ
            for size in self.params['median_sizes']:
                if size < len(data):
                    filtered = median_filter(data, size=size)
                    residual = data - filtered
                    
                    features[f'median_filtered_{size}'] = filtered
                    features[f'median_residual_{size}'] = residual
                    features[f'median_spike_removal_{size}'] = np.abs(residual)
            
            # Savitzky-Golayフィルタ
            for window in self.params['savgol_windows']:
                if window < len(data) and window % 2 == 1:  # 奇数である必要がある
                    try:
                        filtered = savgol_filter(data, window, polyorder=3)
                        residual = data - filtered
                        
                        features[f'savgol_filtered_{window}'] = filtered
                        features[f'savgol_residual_{window}'] = residual
                        features[f'savgol_smoothness_{window}'] = np.gradient(filtered)
                    except:
                        continue
        
        except Exception as e:
            logger.warning(f"フィルタリング計算エラー: {e}")
        
        return features

    @staticmethod
    @njit(cache=True)
    def _compute_energy_features_numba(data: np.ndarray, window_size: int) -> np.ndarray:
        """エネルギー特徴量計算 - Numba最適化版"""
        n = len(data)
        if n < window_size:
            return np.zeros((n, 4))
        
        results = np.zeros((n, 4))
        
        for i in range(window_size-1, n):
            window = data[i-window_size+1:i+1]
            
            # RMSエネルギー
            rms_energy = np.sqrt(np.mean(window**2))
            
            # 総エネルギー
            total_energy = np.sum(window**2)
            
            # パワー
            power = total_energy / window_size
            
            # クレストファクター（ピーク値 / RMS値）
            peak_val = np.max(np.abs(window))
            crest_factor = peak_val / (rms_energy + 1e-10)
            
            results[i] = [rms_energy, total_energy, power, crest_factor]
        
        return results

    def calculate_energy_features(self, data: np.ndarray) -> Dict[str, np.ndarray]:
        """エネルギー特徴量（修正版）"""
        features = {}
        n = len(data)  # データ長を事前に取得
        
        for window_size in self.params['energy_windows']:
            try:
                energy_results = self.robust_calculation_wrapper(
                    self._compute_energy_features_numba, data, window_size
                )

                # 修正: energy_resultsが配列でない場合の処理
                if not isinstance(energy_results, np.ndarray):
                    logger.warning(f"エネルギー特徴量(window={window_size})の計算に失敗しました。フォールバック値を生成します。")
                    # フォールバック値として0埋めの配列を生成
                    energy_results = np.zeros((n, 4))
                
                # 配列の形状チェック
                elif len(energy_results.shape) != 2 or energy_results.shape[1] < 4:
                    logger.warning(f"エネルギー特徴量(window={window_size})の結果形状が不正です。フォールバック値を生成します。")
                    energy_results = np.zeros((n, 4))
                
                # 基本エネルギー特徴量の抽出
                try:
                    features[f'rms_energy_{window_size}'] = energy_results[:, 0]
                    features[f'total_energy_{window_size}'] = energy_results[:, 1]
                    features[f'power_{window_size}'] = energy_results[:, 2]
                    features[f'crest_factor_{window_size}'] = energy_results[:, 3]
                    
                except IndexError as e:
                    logger.error(f"エネルギー特徴量(window={window_size})のインデックスエラー: {e}")
                    # フォールバック値
                    features[f'rms_energy_{window_size}'] = np.zeros(n)
                    features[f'total_energy_{window_size}'] = np.zeros(n)
                    features[f'power_{window_size}'] = np.zeros(n)
                    features[f'crest_factor_{window_size}'] = np.zeros(n)
                
                # エネルギー変化率の計算（安全な処理）
                try:
                    rms_energy = features[f'rms_energy_{window_size}']
                    # NaN/Inf値をチェックしてから勾配計算
                    if np.any(np.isfinite(rms_energy)):
                        # 有効値のみで勾配を計算
                        clean_energy = np.where(np.isfinite(rms_energy), rms_energy, 0.0)
                        energy_change = np.gradient(clean_energy)
                        features[f'energy_change_rate_{window_size}'] = energy_change
                    else:
                        features[f'energy_change_rate_{window_size}'] = np.zeros(n)
                        
                except Exception as e:
                    logger.warning(f"エネルギー変化率計算エラー(window={window_size}): {e}")
                    features[f'energy_change_rate_{window_size}'] = np.zeros(n)
                
                # エネルギースパイク検出（安全な処理）
                try:
                    crest_factor = features[f'crest_factor_{window_size}']
                    # NaN/Inf値を考慮した比較
                    features[f'energy_spike_{window_size}'] = np.where(
                        np.isfinite(crest_factor), 
                        (crest_factor > 5.0).astype(float), 
                        0.0
                    )
                    
                except Exception as e:
                    logger.warning(f"エネルギースパイク検出エラー(window={window_size}): {e}")
                    features[f'energy_spike_{window_size}'] = np.zeros(n)

            except Exception as e:
                logger.error(f"エネルギー特徴量計算で予期しないエラー(window={window_size}): {e}")
                # 全特徴量にフォールバック値を設定
                feature_names = [
                    'rms_energy', 'total_energy', 'power', 'crest_factor', 
                    'energy_change_rate', 'energy_spike'
                ]
                for name in feature_names:
                    features[f'{name}_{window_size}'] = np.zeros(n)
        
        return features

    @staticmethod
    @njit(cache=True)
    def _compute_spectral_centroid_numba(window: np.ndarray) -> Tuple[float, float, float, float]:
        """スペクトル特徴をNumbaで計算"""
        if len(window) < 8:
            return 0.0, 0.0, 0.0, 0.0
        
        # FFT計算（簡易版）
        n = len(window)
        power_spectrum = np.zeros(n // 2)
        freqs = np.zeros(n // 2)
        
        # 単純なDFTの近似
        for k in range(n // 2):
            real_part = 0.0
            imag_part = 0.0
            for i in range(n):
                angle = -2.0 * np.pi * k * i / n
                real_part += window[i] * np.cos(angle)
                imag_part += window[i] * np.sin(angle)
            power_spectrum[k] = real_part * real_part + imag_part * imag_part
            freqs[k] = k / n
        
        total_power = np.sum(power_spectrum)
        if total_power < 1e-10:
            return 0.0, 0.0, 0.0, 0.0
        
        # スペクトル重心
        centroid = np.sum(freqs * power_spectrum) / total_power
        
        # スペクトル帯域幅
        bandwidth = np.sqrt(np.sum(((freqs - centroid) ** 2) * power_spectrum) / total_power)
        
        # ピーク周波数
        peak_idx = np.argmax(power_spectrum)
        peak_freq = freqs[peak_idx]
        
        # スペクトルフラット度（近似）
        spectral_flatness = np.mean(power_spectrum) / (np.max(power_spectrum) + 1e-10)
        
        return centroid, bandwidth, peak_freq, spectral_flatness

    def calculate_advanced_fft_features(self, data: np.ndarray) -> Dict[str, np.ndarray]:
        """高度なFFT特徴量"""
        features = {}
        
        for window_size in self.params['fourier_windows']:
            if len(data) < window_size:
                continue
                
            # 各ウィンドウでのFFT特徴量
            spectral_centroid = np.zeros(len(data))
            spectral_bandwidth = np.zeros(len(data))
            spectral_peak_freq = np.zeros(len(data))
            spectral_flatness = np.zeros(len(data))
            
            step_size = window_size // 2
            for i in range(window_size - 1, len(data), step_size):
                window = data[max(0, i - window_size + 1):i + 1]
                
                centroid, bandwidth, peak_freq, flatness = self.robust_calculation_wrapper(
                    self._compute_spectral_centroid_numba, window
                )
                
                # ウィンドウの範囲に結果を適用
                start_idx = max(0, i - window_size + 1)
                end_idx = min(len(data), i + step_size)
                
                spectral_centroid[start_idx:end_idx] = centroid
                spectral_bandwidth[start_idx:end_idx] = bandwidth
                spectral_peak_freq[start_idx:end_idx] = peak_freq
                spectral_flatness[start_idx:end_idx] = flatness
            
            features[f'fft_spectral_centroid_{window_size}'] = spectral_centroid
            features[f'fft_spectral_bandwidth_{window_size}'] = spectral_bandwidth
            features[f'fft_peak_frequency_{window_size}'] = spectral_peak_freq
            features[f'fft_spectral_flatness_{window_size}'] = spectral_flatness
        
        return features

    def calculate_cwt_features(self, data: np.ndarray) -> Dict[str, np.ndarray]:
        """連続ウェーブレット変換特徴量"""
        features = {}
        
        try:
            for window_size in self.params['cwt_windows']:
                if len(data) < window_size:
                    continue
                    
                # 複数スケールでのCWT
                scales = np.arange(1, min(window_size//4, 32))
                
                cwt_energy = np.zeros(len(data))
                cwt_entropy = np.zeros(len(data))
                dominant_scale = np.zeros(len(data))
                
                for i in range(window_size-1, len(data)):
                    window = data[i-window_size+1:i+1]
                    
                    try:
                        # CWT計算（Morletウェーブレット使用）
                        coefficients, _ = pywt.cwt(window, scales, 'morl')
                        
                        # エネルギー計算
                        energy = np.sum(np.abs(coefficients)**2)
                        cwt_energy[i] = energy
                        
                        # エントロピー計算
                        if energy > 1e-10:
                            prob = np.abs(coefficients)**2 / energy
                            prob = prob[prob > 1e-10]
                            entropy = -np.sum(prob * np.log2(prob))
                            cwt_entropy[i] = entropy
                        
                        # 支配的スケール
                        scale_energies = np.sum(np.abs(coefficients)**2, axis=1)
                        dominant_scale[i] = scales[np.argmax(scale_energies)]
                        
                    except:
                        continue
                
                features[f'cwt_energy_{window_size}'] = cwt_energy
                features[f'cwt_entropy_{window_size}'] = cwt_entropy
                features[f'cwt_dominant_scale_{window_size}'] = dominant_scale
        
        except Exception as e:
            logger.warning(f"CWT計算エラー: {e}")
        
        return features

    def calculate_spectral_analysis_features(self, data: np.ndarray) -> Dict[str, np.ndarray]:
        """スペクトル解析特徴量"""
        features = {}
        
        for window_size in self.params['spectral_windows']:
            if len(data) < window_size:
                continue
            
            spectral_features = np.zeros((len(data), 8))
            
            for i in range(window_size-1, len(data)):
                window = data[i-window_size+1:i+1]
                
                try:
                    # パワースペクトル密度
                    freqs, psd = periodogram(window)
                    
                    if np.sum(psd) > 1e-10:
                        # 正規化
                        psd_norm = psd / np.sum(psd)
                        
                        # スペクトルモーメント
                        spectral_mean = np.sum(freqs * psd_norm)
                        spectral_variance = np.sum(((freqs - spectral_mean)**2) * psd_norm)
                        spectral_skewness = np.sum(((freqs - spectral_mean)**3) * psd_norm) / (spectral_variance**1.5 + 1e-10)
                        spectral_kurtosis = np.sum(((freqs - spectral_mean)**4) * psd_norm) / (spectral_variance**2 + 1e-10)
                        
                        # スペクトルエッジ周波数（90%エネルギー）
                        cumsum_psd = np.cumsum(psd_norm)
                        edge_freq_idx = np.where(cumsum_psd >= 0.9)[0]
                        spectral_edge = freqs[edge_freq_idx[0]] if len(edge_freq_idx) > 0 else 0
                        
                        # スペクトルフラット度（幾何平均/算術平均）
                        spectral_flatness = np.exp(np.mean(np.log(psd + 1e-10))) / (np.mean(psd) + 1e-10)
                        
                        # スペクトルロールオフ
                        rolloff_idx = np.where(cumsum_psd >= 0.85)[0]
                        spectral_rolloff = freqs[rolloff_idx[0]] if len(rolloff_idx) > 0 else 0
                        
                        # スペクトルフラックス（前回との差）
                        if i > window_size:
                            prev_psd = spectral_features[i-1, 7]  # 前回のPSDを参照
                            spectral_flux = np.sum((psd_norm - prev_psd)**2) if prev_psd != 0 else 0
                        else:
                            spectral_flux = 0
                        
                        spectral_features[i] = [
                            spectral_mean, spectral_variance, spectral_skewness, spectral_kurtosis,
                            spectral_edge, spectral_flatness, spectral_rolloff, spectral_flux
                        ]
                
                except:
                    continue
            
            feature_names = [
                'spectral_mean', 'spectral_variance', 'spectral_skewness', 'spectral_kurtosis',
                'spectral_edge_freq', 'spectral_flatness', 'spectral_rolloff', 'spectral_flux'
            ]
            
            for j, name in enumerate(feature_names):
                features[f'{name}_{window_size}'] = spectral_features[:, j]
        
        return features

    def calculate_signal_complexity_features(self, data: np.ndarray) -> Dict[str, np.ndarray]:
        """信号複雑性特徴量"""
        features = {}
        
        # 複数ウィンドウサイズで複雑性分析
        window_sizes = [50, 100, 200]
        
        for window_size in window_sizes:
            if len(data) < window_size:
                continue
            
            complexity_results = np.zeros((len(data), 5))
            
            for i in range(window_size-1, len(data)):
                window = data[i-window_size+1:i+1]
                
                # 信号のエネルギー分布
                signal_energy = np.sum(window**2)
                
                # 高周波エネルギー比率
                if len(window) >= 8:
                    fft_vals = fft(window)
                    total_power = np.sum(np.abs(fft_vals)**2)
                    high_freq_power = np.sum(np.abs(fft_vals[len(fft_vals)//2:])**2)
                    high_freq_ratio = high_freq_power / (total_power + 1e-10)
                else:
                    high_freq_ratio = 0
                
                # 信号の滑らかさ（2次微分の分散）
                if len(window) >= 3:
                    second_diff = np.diff(window, n=2)
                    smoothness = np.var(second_diff)
                else:
                    smoothness = 0
                
                # 局所極値の数
                local_maxima = 0
                local_minima = 0
                for j in range(1, len(window) - 1):
                    if window[j] > window[j-1] and window[j] > window[j+1]:
                        local_maxima += 1
                    elif window[j] < window[j-1] and window[j] < window[j+1]:
                        local_minima += 1
                
                extrema_density = (local_maxima + local_minima) / len(window)
                
                # 信号の自己類似性（隣接区間の相関）
                if len(window) >= 20:
                    half = len(window) // 2
                    first_half = window[:half]
                    second_half = window[half:half*2]
                    
                    if np.std(first_half) > 1e-10 and np.std(second_half) > 1e-10:
                        self_similarity = np.corrcoef(first_half, second_half)[0, 1]
                        if not np.isfinite(self_similarity):
                            self_similarity = 0
                    else:
                        self_similarity = 0
                else:
                    self_similarity = 0
                
                complexity_results[i] = [
                    signal_energy, high_freq_ratio, smoothness, extrema_density, self_similarity
                ]
            
            complexity_names = [
                'signal_energy', 'high_freq_ratio', 'signal_smoothness', 'extrema_density', 'self_similarity'
            ]
            
            for j, name in enumerate(complexity_names):
                features[f'{name}_{window_size}'] = complexity_results[:, j]
        
        return features

    # =============================================================================
    # 欠落特徴量補完 - Block 8: 情報理論・学際的アナロジー特徴量
    # =============================================================================

    @staticmethod
    @njit(cache=True)
    def _compute_histogram_numba(data: np.ndarray, n_bins: int) -> Tuple[np.ndarray, np.ndarray]:
        """ヒストグラムをNumbaで計算"""
        if len(data) == 0:
            return np.zeros(n_bins), np.zeros(n_bins + 1)
        
        min_val = np.min(data)
        max_val = np.max(data)
        
        if max_val - min_val < 1e-10:
            hist = np.zeros(n_bins)
            hist[n_bins // 2] = len(data)
            edges = np.linspace(min_val - 1e-5, max_val + 1e-5, n_bins + 1)
            return hist, edges
        
        bin_width = (max_val - min_val) / n_bins
        hist = np.zeros(n_bins)
        edges = np.zeros(n_bins + 1)
        
        for i in range(n_bins + 1):
            edges[i] = min_val + i * bin_width
        
        for val in data:
            bin_idx = int((val - min_val) / bin_width)
            if bin_idx >= n_bins:
                bin_idx = n_bins - 1
            if bin_idx < 0:
                bin_idx = 0
            hist[bin_idx] += 1
        
        return hist, edges

    def calculate_advanced_entropy_features(self, data: np.ndarray) -> Dict[str, np.ndarray]:
        """高度なエントロピー特徴量"""
        features = {}
        
        for window_size in self.params['adv_entropy_windows']:
            if len(data) < window_size:
                continue
            
            entropy_results = np.zeros((len(data), 6))
            
            for i in range(window_size-1, len(data)):
                window = data[i-window_size+1:i+1]
                
                # 離散化（適応的ビン数）
                n_bins = max(5, min(20, int(np.sqrt(len(window)))))
                try:
                    hist, bin_edges = self.robust_calculation_wrapper(self._compute_histogram_numba, window, n_bins)
                    prob = hist / np.sum(hist) if np.sum(hist) > 0 else np.zeros(len(hist))
                    prob = prob[prob > 0]  # ゼロ確率を除去
                    
                    # シャノンエントロピー
                    shannon_entropy = -np.sum(prob * np.log2(prob)) if len(prob) > 0 else 0
                    
                    # レニーエントロピー（q=2）
                    renyi_entropy = -np.log2(np.sum(prob**2)) if len(prob) > 0 and np.sum(prob**2) > 0 else 0
                    
                    # Tsallisエントロピー（q=2）
                    tsallis_entropy = (1 - np.sum(prob**2)) / (2 - 1) if len(prob) > 0 else 0
                    
                    # 条件付きエントロピー（近似）
                    if len(window) >= 4:
                        # 1つ前の値に対する条件付きエントロピー
                        conditional_pairs = list(zip(window[:-1], window[1:]))
                        unique_pairs = list(set(conditional_pairs))
                        
                        if len(unique_pairs) > 1:
                            pair_counts = [conditional_pairs.count(pair) for pair in unique_pairs]
                            pair_prob = np.array(pair_counts) / len(conditional_pairs)
                            conditional_entropy = -np.sum(pair_prob * np.log2(pair_prob + 1e-10))
                        else:
                            conditional_entropy = 0
                    else:
                        conditional_entropy = 0
                    
                    entropy_results[i] = [shannon_entropy, renyi_entropy, tsallis_entropy, 
                                        conditional_entropy, len(prob), np.max(prob) if len(prob) > 0 else 0]
                
                except:
                    continue
            
            entropy_names = ['shannon_entropy', 'renyi_entropy', 'tsallis_entropy', 
                            'conditional_entropy', 'effective_bins', 'max_probability']
            
            for j, name in enumerate(entropy_names):
                features[f'{name}_{window_size}'] = entropy_results[:, j]
        
        return features

    def calculate_sample_approximate_entropy(self, data: np.ndarray, window_size: int = 50) -> Dict[str, np.ndarray]:
        """サンプルエントロピー・近似エントロピー特徴量"""
        features = {}
        
        if not ENTROPY_AVAILABLE:
            logger.warning("entropyライブラリが利用できないため、サンプルエントロピー計算をスキップ")
            return features
        
        try:
            sample_entropy_vals = np.zeros(len(data))
            approx_entropy_vals = np.zeros(len(data))
            
            for i in range(window_size-1, len(data)):
                window = data[i-window_size+1:i+1]
                
                try:
                    # サンプルエントロピー
                    sample_ent = ent.sample_entropy(window, sample_length=2, tolerance=0.2*np.std(window))
                    sample_entropy_vals[i] = sample_ent[1] if len(sample_ent) > 1 else 0
                    
                    # 近似エントロピー  
                    approx_ent = ent.app_entropy(window, sample_length=2, tolerance=0.2*np.std(window))
                    approx_entropy_vals[i] = approx_ent
                    
                except:
                    continue
            
            features[f'sample_entropy_{window_size}'] = sample_entropy_vals
            features[f'approximate_entropy_{window_size}'] = approx_entropy_vals
            features[f'entropy_difference_{window_size}'] = sample_entropy_vals - approx_entropy_vals
        
        except Exception as e:
            logger.warning(f"サンプル/近似エントロピー計算エラー: {e}")
        
        return features

    @staticmethod
    @njit(cache=True)
    def _compute_lempel_ziv_complexity_numba(data: np.ndarray, threshold: float) -> float:
        """Lempel-Ziv複雑性計算 - Numba最適化版"""
        if len(data) < 2:
            return 0.0
        
        # バイナリ化
        binary_data = (data > threshold).astype(np.int32)
        
        # Lempel-Ziv複雑性計算
        i = 0
        complexity = 0
        n = len(binary_data)
        
        while i < n:
            max_len = 1
            
            # 最長の既知パターンを検索
            for j in range(i):
                match_len = 0
                while (i + match_len < n and 
                    j + match_len < i and 
                    binary_data[i + match_len] == binary_data[j + match_len]):
                    match_len += 1
                max_len = max(max_len, match_len + 1)
            
            i += max_len
            complexity += 1
        
        # 正規化（理論的最大値で割る）
        normalized_complexity = complexity / (n / np.log2(n + 1))
        return normalized_complexity

    def calculate_lempel_ziv_features(self, data: np.ndarray) -> Dict[str, np.ndarray]:
        """Lempel-Ziv複雑性特徴量"""
        features = {}
        
        for window_size in self.params['lz_windows']:
            if len(data) < window_size:
                continue
            
            lz_complexity = np.zeros(len(data))
            
            for i in range(window_size-1, len(data)):
                window = data[i-window_size+1:i+1]
                threshold = np.median(window)  # 中央値を閾値として使用
                
                complexity = self.robust_calculation_wrapper(
                    self._compute_lempel_ziv_complexity_numba, window, threshold
                )
                lz_complexity[i] = complexity
            
            features[f'lempel_ziv_complexity_{window_size}'] = lz_complexity
            
            # パーセンタイル計算をNumbaループに変更
            lz_75th_percentile = self.robust_calculation_wrapper(
                self._compute_rolling_percentile_numba, lz_complexity, window_size, 75
            )
            features[f'lz_high_complexity_{window_size}'] = (lz_complexity > lz_75th_percentile).astype(float)
        
        return features

    def calculate_kolmogorov_complexity_features(self, data: np.ndarray) -> Dict[str, np.ndarray]:
        """コルモゴロフ複雑性（近似）特徴量"""
        features = {}
        
        for window_size in self.params['kolmogorov_windows']:
            if len(data) < window_size:
                continue
            
            kolmogorov_approx = np.zeros(len(data))
            
            for i in range(window_size-1, len(data)):
                window = data[i-window_size+1:i+1]
                
                # 複雑性の近似指標として複数の手法を組み合わせ
                
                # 1. 圧縮性（ランレングス符号化風）
                compressed_length = 1
                prev_val = window[0]
                
                for val in window[1:]:
                    if abs(val - prev_val) > np.std(window) * 0.1:
                        compressed_length += 1
                    prev_val = val
                
                compression_ratio = compressed_length / len(window)
                
                # 2. パターンの反復性
                pattern_complexity = 0
                pattern_length = min(10, len(window) // 4)
                
                for p_len in range(2, pattern_length + 1):
                    patterns = set()
                    for j in range(len(window) - p_len + 1):
                        pattern = tuple(window[j:j + p_len])
                        patterns.add(pattern)
                    pattern_complexity += len(patterns) / (len(window) - p_len + 1)
                
                # 3. エントロピーベースの複雑性
                n_bins = max(5, int(np.sqrt(len(window))))
                hist, _ = self.robust_calculation_wrapper(self._compute_histogram_numba, window, n_bins)
                prob = hist / np.sum(hist) if np.sum(hist) > 0 else np.zeros(len(hist))
                prob = prob[prob > 0]
                entropy_complexity = -np.sum(prob * np.log2(prob)) / np.log2(n_bins) if len(prob) > 0 else 0
                
                # 複合複雑性スコア
                combined_complexity = (compression_ratio + pattern_complexity + entropy_complexity) / 3
                kolmogorov_approx[i] = combined_complexity
            
            features[f'kolmogorov_complexity_{window_size}'] = kolmogorov_approx
            
            # パーセンタイル計算をNumbaに変更
            kc_80th_percentile = self.robust_calculation_wrapper(
                self._compute_rolling_percentile_numba, kolmogorov_approx, window_size, 80
            )
            features[f'kc_high_complexity_{window_size}'] = (kolmogorov_approx > kc_80th_percentile).astype(float)
        
        return features

    @staticmethod
    @njit(cache=True)
    def _compute_binned_data_numba(data: np.ndarray, n_bins: int) -> np.ndarray:
        """データを離散化 - Numba最適化版"""
        if len(data) == 0:
            return np.array([], dtype=np.int32)
        
        min_val = np.min(data)
        max_val = np.max(data)
        
        if max_val - min_val < 1e-10:
            return np.zeros(len(data), dtype=np.int32)
        
        bin_width = (max_val - min_val) / n_bins
        binned = np.zeros(len(data), dtype=np.int32)
        
        for i in range(len(data)):
            bin_idx = int((data[i] - min_val) / bin_width)
            if bin_idx >= n_bins:
                bin_idx = n_bins - 1
            if bin_idx < 0:
                bin_idx = 0
            binned[i] = bin_idx
        
        return binned

    @staticmethod
    @njit(cache=True)
    def _compute_mutual_information_score_numba(x: np.ndarray, y: np.ndarray, n_bins: int) -> float:
        """相互情報量をNumbaで計算"""
        if len(x) != len(y) or len(x) == 0:
            return 0.0
        
        n = len(x)
        
        # 各変数のヒストグラム
        x_hist = np.zeros(n_bins)
        y_hist = np.zeros(n_bins)
        xy_hist = np.zeros((n_bins, n_bins))
        
        for i in range(n):
            x_bin = x[i]
            y_bin = y[i]
            if 0 <= x_bin < n_bins and 0 <= y_bin < n_bins:
                x_hist[x_bin] += 1
                y_hist[y_bin] += 1
                xy_hist[x_bin, y_bin] += 1
        
        # 確率に変換
        px = x_hist / n
        py = y_hist / n
        pxy = xy_hist / n
        
        # 相互情報量計算
        mi = 0.0
        for i in range(n_bins):
            for j in range(n_bins):
                if pxy[i, j] > 1e-10 and px[i] > 1e-10 and py[j] > 1e-10:
                    mi += pxy[i, j] * np.log2(pxy[i, j] / (px[i] * py[j]))
        
        return mi

    def calculate_mutual_information_features(self, data1: np.ndarray, data2: np.ndarray = None) -> Dict[str, np.ndarray]:
        """相互情報量特徴量"""
        features = {}
        
        # 自己遅延相互情報量
        for lag in self.params['mutual_info_lags']:
            if len(data1) > lag:
                mi_values = np.zeros(len(data1))
                
                for i in range(50, len(data1)):  # 最小ウィンドウサイズ50
                    window_size = min(100, i)
                    if i >= window_size + lag:
                        x = data1[i-window_size:i]
                        y = data1[i-window_size-lag:i-lag]
                        
                        try:
                            # 離散化
                            n_bins = max(5, int(np.sqrt(len(x))))
                            x_binned = self.robust_calculation_wrapper(self._compute_binned_data_numba, x, n_bins)
                            y_binned = self.robust_calculation_wrapper(self._compute_binned_data_numba, y, n_bins)
                            
                            if len(x_binned) > 0 and len(y_binned) > 0:
                                mi = self.robust_calculation_wrapper(
                                    self._compute_mutual_information_score_numba, x_binned, y_binned, n_bins
                                )
                                mi_values[i] = mi
                        except:
                            continue
                
                features[f'mutual_info_lag_{lag}'] = mi_values
                
                # パーセンタイル計算をNumbaに変更
                mi_75th_percentile = self.robust_calculation_wrapper(
                    self._compute_rolling_percentile_numba, mi_values, 50, 75
                )
                features[f'mi_significant_lag_{lag}'] = (mi_values > mi_75th_percentile).astype(float)
        
        # 異なるデータ系列間の相互情報量
        if data2 is not None:
            mi_cross = np.zeros(min(len(data1), len(data2)))
            
            for i in range(50, min(len(data1), len(data2))):
                window_size = min(100, i)
                x = data1[i-window_size:i]
                y = data2[i-window_size:i]
                
                try:
                    n_bins = max(5, int(np.sqrt(len(x))))
                    x_binned = self.robust_calculation_wrapper(self._compute_binned_data_numba, x, n_bins)
                    y_binned = self.robust_calculation_wrapper(self._compute_binned_data_numba, y, n_bins)
                    
                    if len(x_binned) > 0 and len(y_binned) > 0:
                        mi = self.robust_calculation_wrapper(
                            self._compute_mutual_information_score_numba, x_binned, y_binned, n_bins
                        )
                        mi_cross[i] = mi
                except:
                    continue
            
            # 長さを合わせる
            if len(mi_cross) < len(data1):
                mi_cross = np.pad(mi_cross, (0, len(data1) - len(mi_cross)), mode='edge')
            
            features['mutual_info_cross'] = mi_cross[:len(data1)]
            
            # パーセンタイル計算をNumbaに変更
            mi_cross_75th_percentile = self.robust_calculation_wrapper(
                self._compute_rolling_percentile_numba, mi_cross[:len(data1)], 50, 75
            )
            features['mi_cross_significant'] = (mi_cross[:len(data1)] > mi_cross_75th_percentile).astype(float)
        
        return features

    # =============================================================================
    # 学際的アナロジー特徴量 - ゲーム理論
    # =============================================================================

    def calculate_game_theory_features(self, data: np.ndarray, window_size: int = 50) -> Dict[str, np.ndarray]:
        """ゲーム理論特徴量"""
        features = {}
        n = len(data)
        
        game_theory_results = np.zeros((n, 6))
        
        for i in range(window_size-1, n):
            window = data[i-window_size+1:i+1]
            returns = np.diff(window, prepend=window[0])
            
            if len(returns) < 2:
                continue
            
            # 1. ナッシュ均衡（安定性指標）
            return_mean = np.mean(returns)
            return_std = np.std(returns)
            nash_equilibrium = return_std / (abs(return_mean) + 1e-10)  # 安定性/方向性
            
            # 2. 協力指数（連続リターンの相関）
            if len(returns) > 2:
                cooperation_index = np.corrcoef(returns[:-1], returns[1:])[0, 1]
                if not np.isfinite(cooperation_index):
                    cooperation_index = 0
            else:
                cooperation_index = 0
            
            # 3. 戦略多様性（リターンの符号の種類）
            return_signs = np.sign(returns)
            unique_signs = len(np.unique(return_signs[return_signs != 0]))
            strategy_diversity = unique_signs / 3  # 最大3種類（+1, -1, 0）
            
            # 4. ゼロサム指標
            total_return = np.sum(returns)
            total_abs_return = np.sum(np.abs(returns))
            zero_sum_indicator = 1 - abs(total_return) / (total_abs_return + 1e-10)
            
            # 5. 囚人のジレンマ（裏切りvs協調）
            defection_payoff = 0
            cooperation_payoff = 0
            
            for j in range(len(returns) - 1):
                if returns[j] * returns[j+1] < 0:  # 逆方向（裏切り）
                    defection_payoff += abs(returns[j+1])
                else:  # 同方向（協調）
                    cooperation_payoff += abs(returns[j+1])
            
            total_payoff = defection_payoff + cooperation_payoff
            prisoners_dilemma = defection_payoff / (total_payoff + 1e-10)
            
            # 6. ミニマックス戦略（損失最小化）
            max_loss = np.min(returns) if len(returns) > 0 else 0
            max_gain = np.max(returns) if len(returns) > 0 else 0
            minimax_strategy = abs(max_loss) / (abs(max_gain) + abs(max_loss) + 1e-10)
            
            game_theory_results[i] = [nash_equilibrium, cooperation_index, strategy_diversity,
                                    zero_sum_indicator, prisoners_dilemma, minimax_strategy]
        
        game_theory_names = ['nash_equilibrium', 'cooperation_index', 'strategy_diversity',
                            'zero_sum_indicator', 'prisoners_dilemma', 'minimax_strategy']
        
        for j, name in enumerate(game_theory_names):
            features[f'{name}_{window_size}'] = game_theory_results[:, j]
        
        return features

    # =============================================================================
    # 学際的アナロジー特徴量 - 分子科学
    # =============================================================================

    def calculate_molecular_science_features(self, data: np.ndarray, window_size: int = 50) -> Dict[str, np.ndarray]:
        """分子科学特徴量"""
        features = {}
        n = len(data)
        
        molecular_results = np.zeros((n, 6))
        
        for i in range(window_size-1, n):
            window = data[i-window_size+1:i+1]
            
            # 1. 分子振動（高周波成分の強さ）
            if len(window) >= 8:
                fft_vals = fft(window)
                high_freq_power = np.sum(np.abs(fft_vals[len(fft_vals)//2:])**2)
                total_power = np.sum(np.abs(fft_vals)**2)
                molecular_vibration = high_freq_power / (total_power + 1e-10)
            else:
                molecular_vibration = 0
            
            # 2. 結合エネルギー（エントロピー様）
            n_bins = max(3, int(np.sqrt(len(window))))
            hist, _ = self.robust_calculation_wrapper(self._compute_histogram_numba, window, n_bins)
            prob = hist / np.sum(hist) if np.sum(hist) > 0 else np.zeros(len(hist))
            prob = prob[prob > 0]
            bond_energy = -np.sum(prob * np.log(prob + 1e-10)) if len(prob) > 0 else 0
            
            # 3. 電子密度（価格の二乗和）
            electron_density = np.sum(window**2) / len(window)
            
            # 4. 分子軌道（時間加重エネルギー）
            time_weights = np.arange(1, len(window) + 1) / len(window)
            molecular_orbital = np.sum(time_weights * window**2)
            
            # 5. 化学ポテンシャル（平均変化率）
            chemical_potential = np.mean(np.diff(window)) if len(window) > 1 else 0
            
            # 6. 分子間力（低ボラティリティの度合い）
            volatility = np.std(window)
            intermolecular_force = 1 / (1 + volatility)  # ボラティリティが低いほど分子間力が強い
            
            molecular_results[i] = [molecular_vibration, bond_energy, electron_density,
                                molecular_orbital, chemical_potential, intermolecular_force]
        
        molecular_names = ['molecular_vibration', 'bond_energy', 'electron_density',
                        'molecular_orbital', 'chemical_potential', 'intermolecular_force']
        
        for j, name in enumerate(molecular_names):
            features[f'{name}_{window_size}'] = molecular_results[:, j]
        
        return features

    # =============================================================================
    # 学際的アナロジー特徴量 - ネットワーク科学
    # =============================================================================

    def calculate_network_science_features(self, data: np.ndarray, window_size: int = 50) -> Dict[str, np.ndarray]:
        """ネットワーク科学特徴量"""
        features = {}
        n = len(data)
        
        network_results = np.zeros((n, 5))
        
        for i in range(window_size-1, n):
            window = data[i-window_size+1:i+1]
            
            # 1. ネットワーク密度（ユニークな価格状態の数）
            unique_values = len(np.unique(np.round(window, 6)))  # 丸めてノイズ除去
            network_density = unique_values / len(window)
            
            # 2. 中心性（中央値と平均の比率）
            median_val = np.median(window)
            mean_val = np.mean(window)
            centrality_measure = median_val / (mean_val + 1e-10) if abs(mean_val) > 1e-10 else 1
            
            # 3. クラスタリング係数（異なる時間軸の相関）
            if len(window) >= 9:
                short_term = window[-3:]  # 短期
                mid_term = window[-6:-3]  # 中期
                long_term = window[-9:-6]  # 長期
                
                # 各ペア間の相関
                correlations = []
                for x, y in [(short_term, mid_term), (mid_term, long_term), (short_term, long_term)]:
                    if np.std(x) > 1e-10 and np.std(y) > 1e-10:
                        corr = np.corrcoef(x, y)[0, 1]
                        if np.isfinite(corr):
                            correlations.append(abs(corr))
                
                clustering_coefficient = np.mean(correlations) if correlations else 0
            else:
                clustering_coefficient = 0
            
            # 4. 媒介中心性（中心期間の極値存在度）
            center_idx = len(window) // 2
            center_region = window[center_idx-2:center_idx+3] if center_idx >= 2 else window
            max_val = np.max(window)
            min_val = np.min(window)
            
            has_extreme_in_center = (np.max(center_region) == max_val) or (np.min(center_region) == min_val)
            betweenness_centrality = float(has_extreme_in_center)
            
            # 5. 固有ベクトル中心性（自己相関の累積）
            eigenvector_centrality = 0
            for lag in range(1, min(6, len(window)//2)):
                if len(window) > lag:
                    x1 = window[:-lag]
                    x2 = window[lag:]
                    if np.std(x1) > 1e-10 and np.std(x2) > 1e-10:
                        corr = np.corrcoef(x1, x2)[0, 1]
                        if np.isfinite(corr):
                            eigenvector_centrality += abs(corr)
            
            network_results[i] = [network_density, centrality_measure, clustering_coefficient,
                                betweenness_centrality, eigenvector_centrality]
        
        network_names = ['network_density', 'centrality_measure', 'clustering_coefficient',
                        'betweenness_centrality', 'eigenvector_centrality']
        
        for j, name in enumerate(network_names):
            features[f'{name}_{window_size}'] = network_results[:, j]
        
        return features     

    # =============================================================================
    # 欠落特徴量補完 - Block 9: 学際的アナロジー特徴量（音響学・言語学・美学・その他）
    # =============================================================================

    # 音響学特徴量
    def calculate_acoustics_features(self, data: np.ndarray, window_size: int = 64) -> Dict[str, np.ndarray]:
        """音響学特徴量（修正版）"""
        features = {}
        n = len(data)
        
        acoustics_results = np.zeros((n, 5))
        
        for i in range(window_size-1, n):
            window = data[i-window_size+1:i+1]
            
            try:
                returns = np.diff(window) if len(window) > 1 else np.array([0])
                
                # 1. 音響パワー（リターンの二乗和）
                acoustic_power = np.sum(returns**2) / len(returns) if len(returns) > 0 else 0
                
                # 2. 音響周波数（FFTによる主要周波数） - 直流成分除外の修正
                if len(window) >= 8:
                    fft_vals = fft(window)
                    freqs = fftfreq(len(window))
                    power_spectrum = np.abs(fft_vals)**2
                    
                    # 修正: 直流成分(インデックス0)を除外してピークを探す
                    if len(power_spectrum) > 2:
                        # 正の周波数成分のみを考慮（直流成分を除外）
                        positive_freq_range = len(power_spectrum) // 2
                        if positive_freq_range > 1:
                            peak_idx = np.argmax(power_spectrum[1:positive_freq_range]) + 1
                            acoustic_frequency = abs(freqs[peak_idx])
                        else:
                            acoustic_frequency = 0
                    else:
                        acoustic_frequency = 0
                else:
                    acoustic_frequency = 0
                
                # 3. 振幅変調（ボラティリティの変動）
                if len(returns) >= 4:
                    volatility_changes = np.abs(np.diff(returns))
                    amplitude_modulation = np.std(volatility_changes)
                else:
                    amplitude_modulation = 0
                
                # 4. 位相変調（FFT位相の安定性）
                if len(window) >= 8:
                    phases = np.angle(fft(window))
                    # 位相の連続性を考慮した安定性計算
                    phase_diffs = np.diff(phases)
                    # 位相ラッピングを考慮
                    phase_diffs = np.mod(phase_diffs + np.pi, 2*np.pi) - np.pi
                    phase_stability = 1 / (np.var(phase_diffs) + 1e-10)
                    phase_modulation = 1 / (1 + phase_stability)
                else:
                    phase_modulation = 0
                
                # 5. 音響エコー（前半と後半の自己相関） - 配列スライス修正
                if len(window) >= 8:
                    # 修正: 前半と後半の長さを正確に揃える
                    half = len(window) // 2
                    first_half = window[:half]
                    second_half = window[-half:]  # 後半部分を正確に取得
                    
                    # 両方の配列が同じ長さであることを確認
                    if len(first_half) == len(second_half) and len(first_half) > 1:
                        if np.std(first_half) > 1e-10 and np.std(second_half) > 1e-10:
                            correlation_matrix = np.corrcoef(first_half, second_half)
                            if correlation_matrix.shape == (2, 2):
                                acoustic_echo = abs(correlation_matrix[0, 1])
                                if not np.isfinite(acoustic_echo):
                                    acoustic_echo = 0
                            else:
                                acoustic_echo = 0
                        else:
                            acoustic_echo = 0
                    else:
                        acoustic_echo = 0
                else:
                    acoustic_echo = 0
                
                acoustics_results[i] = [acoustic_power, acoustic_frequency, amplitude_modulation,
                                    phase_modulation, acoustic_echo]
                                    
            except Exception as e:
                # 計算失敗時は0で埋める（NaNよりも安全）
                acoustics_results[i] = [0.0, 0.0, 0.0, 0.0, 0.0]
                continue
        
        acoustics_names = ['acoustic_power', 'acoustic_frequency', 'amplitude_modulation',
                        'phase_modulation', 'acoustic_echo']
        
        for j, name in enumerate(acoustics_names):
            features[f'{name}_{window_size}'] = acoustics_results[:, j]
            
        return features

    # 言語学特徴量
    def calculate_linguistics_features(self, data: np.ndarray, window_size: int = 50) -> Dict[str, np.ndarray]:
        """言語学特徴量"""
        features = {}
        n = len(data)
        
        linguistics_results = np.zeros((n, 5))
        
        for i in range(window_size-1, n):
            window = data[i-window_size+1:i+1]
            
            # 価格を離散化（「単語」に相当）
            n_levels = max(3, min(10, int(len(window) / 5)))
            try:
                # Numbaでの離散化を使用
                digitized = self.robust_calculation_wrapper(self._compute_binned_data_numba, window, n_levels)
                if len(digitized) == 0:
                    continue
            except:
                continue
            
            # 1. 語彙の多様性（ユニークなパターンの多様性）
            unique_patterns = len(np.unique(digitized))
            vocabulary_diversity = unique_patterns / n_levels
            
            # 2. 文の構造（変化の滑らかさ）
            if len(window) >= 3:
                second_diff = np.diff(window, n=2)
                sentence_structure = 1 / (np.std(second_diff) + 1e-10)  # 滑らかさ
            else:
                sentence_structure = 0
            
            # 3. 言語的複雑性（大きな変化の頻度）
            returns = np.diff(window) if len(window) > 1 else np.array([0])
            large_changes = np.abs(returns) > np.std(returns) * 1.5
            linguistic_complexity = np.sum(large_changes) / len(returns) if len(returns) > 0 else 0
            
            # 4. 語順（価格順位の変化）
            if len(window) >= 4:
                # 各値の相対順位を計算
                ranks = []
                for j, val in enumerate(window):
                    rank = np.sum(window <= val) / len(window)
                    ranks.append(rank)
                
                # 順位の変動を測定
                rank_changes = np.diff(ranks) if len(ranks) > 1 else np.array([0])
                word_order = np.std(rank_changes)
            else:
                word_order = 0
            
            # 5. プロソディ（抑揚・リズム）
            returns = np.diff(window) if len(window) > 1 else np.array([0])
            prosody = np.std(np.abs(returns))  # 変動の抑揚
            
            linguistics_results[i] = [vocabulary_diversity, sentence_structure, linguistic_complexity,
                                    word_order, prosody]
        
        linguistics_names = ['vocabulary_diversity', 'sentence_structure', 'linguistic_complexity',
                            'word_order', 'prosody']
        
        for j, name in enumerate(linguistics_names):
            features[f'{name}_{window_size}'] = linguistics_results[:, j]
        
        return features

    # 美学特徴量
    def calculate_aesthetics_features(self, data: np.ndarray, window_size: int = 50) -> Dict[str, np.ndarray]:
        """美学特徴量（修正版）"""
        features = {}
        n = len(data)
        
        aesthetics_results = np.zeros((n, 5))
        golden_ratio = self.mathematical_constants['golden_ratio']
        
        for i in range(window_size-1, n):
            window = data[i-window_size+1:i+1]
            
            try:
                # 1. 黄金比への固着度
                if len(window) >= 2:
                    ranges = []
                    for j in range(1, len(window)):
                        if abs(window[j-1]) > 1e-10:  # ゼロ除算対策を強化
                            ratio = abs(window[j] / window[j-1])
                            # 異常に大きな比率を除外
                            if ratio < 1000:
                                ranges.append(ratio)
                    
                    if ranges:
                        # 各比率と黄金比の差
                        golden_deviations = [abs(ratio - golden_ratio) for ratio in ranges]
                        golden_ratio_adherence = 1 / (1 + np.mean(golden_deviations))
                    else:
                        golden_ratio_adherence = 0
                else:
                    golden_ratio_adherence = 0
                
                # 2. 対称性（前半と後半の反転相似性） - 配列長修正
                if len(window) >= 4:
                    half = len(window) // 2
                    first_half = window[:half]
                    # 修正: 後半部分を正確に取得し、長さを揃える
                    second_half = window[-half:]  # 末尾からhalf個取得
                    
                    # 後半を反転
                    second_half_reversed = second_half[::-1]
                    
                    # 両配列が同じ長さであることを確認
                    if (len(first_half) == len(second_half_reversed) and 
                        len(first_half) > 1 and
                        np.std(first_half) > 1e-10 and 
                        np.std(second_half_reversed) > 1e-10):
                        
                        correlation_matrix = np.corrcoef(first_half, second_half_reversed)
                        if correlation_matrix.shape == (2, 2):
                            symmetry_measure = abs(correlation_matrix[0, 1])
                            if not np.isfinite(symmetry_measure):
                                symmetry_measure = 0
                        else:
                            symmetry_measure = 0
                    else:
                        symmetry_measure = 0
                else:
                    symmetry_measure = 0
                
                # 3. 美的調和（滑らかさ・躍度の小ささ）
                if len(window) >= 4:
                    try:
                        # 3階微分（躍度）- より安全な計算
                        first_diff = np.diff(window)
                        if len(first_diff) >= 2:
                            second_diff = np.diff(first_diff)
                            if len(second_diff) >= 1:
                                third_diff = np.diff(second_diff)
                                if len(third_diff) > 0:
                                    jerk_std = np.std(third_diff)
                                    aesthetic_harmony = 1 / (1 + jerk_std)  # 躍度が小さいほど調和
                                else:
                                    aesthetic_harmony = 0.5  # デフォルト値
                            else:
                                aesthetic_harmony = 0.5
                        else:
                            aesthetic_harmony = 0.5
                    except:
                        aesthetic_harmony = 0.5
                else:
                    aesthetic_harmony = 0
                
                # 4. 比例美（上昇・下降期間の比率バランス）
                if len(window) > 1:
                    returns = np.diff(window)
                    up_periods = np.sum(returns > 1e-10)  # 小さな変動を除外
                    down_periods = np.sum(returns < -1e-10)
                    total_periods = len(returns)
                    
                    if total_periods > 0:
                        up_ratio = up_periods / total_periods
                        # バランスが取れているほど美しい（0.5に近いほど良い）
                        proportional_beauty = 1 - 2 * abs(up_ratio - 0.5)
                    else:
                        proportional_beauty = 0
                else:
                    proportional_beauty = 0
                
                # 5. 視覚的バランス（中央値に対する分布バランス）
                if len(window) > 2:
                    median_val = np.median(window)
                    # 中央値との比較にトレランスを追加
                    tolerance = np.std(window) * 0.01  # 標準偏差の1%
                    above_median = np.sum(window > median_val + tolerance)
                    below_median = np.sum(window < median_val - tolerance)
                    total_points = len(window)
                    
                    if total_points > 0:
                        above_ratio = above_median / total_points
                        visual_balance = 1 - 2 * abs(above_ratio - 0.5)
                    else:
                        visual_balance = 0
                else:
                    visual_balance = 0
                
                aesthetics_results[i] = [golden_ratio_adherence, symmetry_measure, aesthetic_harmony,
                                    proportional_beauty, visual_balance]
                                    
            except Exception as e:
                # 計算失敗時は中性値で初期化
                aesthetics_results[i] = [0.0, 0.0, 0.5, 0.0, 0.0]
                continue
        
        aesthetics_names = ['golden_ratio_adherence', 'symmetry_measure', 'aesthetic_harmony',
                        'proportional_beauty', 'visual_balance']
        
        for j, name in enumerate(aesthetics_names):
            features[f'{name}_{window_size}'] = aesthetics_results[:, j]
        
        return features

    # 音楽理論特徴量
    def calculate_music_theory_features(self, data: np.ndarray, window_size: int = 50) -> Dict[str, np.ndarray]:
        """音楽理論特徴量"""
        features = {}
        n = len(data)
        
        music_results = np.zeros((n, 6))
        
        for i in range(window_size-1, n):
            window = data[i-window_size+1:i+1]
            returns = np.diff(window) if len(window) > 1 else np.array([0])
            
            # 1. 調性（安定性と方向性の比率）
            return_mean = np.mean(returns)
            return_std = np.std(returns)
            tonality = return_std / (abs(return_mean) + 1e-10)
            
            # 2. リズムパターン（大きな変化の規則性）
            large_changes = np.abs(returns) > np.std(returns) * 1.5
            if np.sum(large_changes) > 1:
                # 大きな変化間の間隔
                change_indices = np.where(large_changes)[0]
                intervals = np.diff(change_indices)
                rhythm_pattern = 1 / (1 + np.std(intervals)) if len(intervals) > 1 else 0
            else:
                rhythm_pattern = 0
            
            # 3. 和声（FFT周波数比の協和音程性）
            if len(window) >= 8:
                fft_vals = fft(window)
                power_spectrum = np.abs(fft_vals[:len(fft_vals)//2])**2
                
                # 上位3つのピークを取得
                peak_indices = np.argsort(power_spectrum)[-3:]
                peak_freqs = peak_indices / len(window)
                
                # 周波数比を計算
                harmonies = []
                for j in range(len(peak_freqs)):
                    for k in range(j+1, len(peak_freqs)):
                        if peak_freqs[j] > 1e-10:
                            ratio = peak_freqs[k] / peak_freqs[j]
                            # 協和音程（2:1, 3:2, 4:3など）との近さ
                            harmonic_ratios = [2.0, 1.5, 1.33, 1.25, 1.2]
                            min_distance = min([abs(ratio - hr) for hr in harmonic_ratios])
                            harmonies.append(1 / (1 + min_distance))
                
                harmony = np.mean(harmonies) if harmonies else 0
            else:
                harmony = 0
            
            # 4. 旋律の輪郭（上昇傾向の割合）
            up_moves = np.sum(returns > 0)
            melody_contour = up_moves / len(returns) if len(returns) > 0 else 0.5
            
            # 5. 音楽的緊張（変動範囲の大きさ）
            musical_tension = np.max(window) - np.min(window) if len(window) > 0 else 0
            
            # 6. テンポ（変化の頻度）
            direction_changes = 0
            for j in range(1, len(returns)):
                if returns[j] * returns[j-1] < 0:  # 符号変化
                    direction_changes += 1
            tempo = direction_changes / len(returns) if len(returns) > 0 else 0
            
            music_results[i] = [tonality, rhythm_pattern, harmony, melody_contour, musical_tension, tempo]
        
        music_names = ['tonality', 'rhythm_pattern', 'harmony', 'melody_contour', 'musical_tension', 'tempo']
        
        for j, name in enumerate(music_names):
            features[f'{name}_{window_size}'] = music_results[:, j]
        
        return features

    # 天文学・宇宙論特徴量
    def calculate_astronomy_features(self, data: np.ndarray, window_size: int = 50) -> Dict[str, np.ndarray]:
        """天文学・宇宙論特徴量"""
        features = {}
        n = len(data)
        
        astronomy_results = np.zeros((n, 6))
        
        for i in range(window_size-1, n):
            window = data[i-window_size+1:i+1]
            
            # 1. 軌道力学（12期間ラグでの自己相関）
            lag = min(12, len(window) // 4)
            if len(window) > lag:
                x1 = window[:-lag]
                x2 = window[lag:]
                if np.std(x1) > 1e-10 and np.std(x2) > 1e-10:
                    orbital_mechanics = abs(np.corrcoef(x1, x2)[0, 1])
                    if not np.isfinite(orbital_mechanics):
                        orbital_mechanics = 0
                else:
                    orbital_mechanics = 0
            else:
                orbital_mechanics = 0
            
            # 2. 重力波（ボラティリティの周期変化）
            if len(window) >= 16:
                # ボラティリティの時系列
                volatilities = []
                sub_window_size = 4
                for j in range(sub_window_size, len(window)):
                    sub_window = window[j-sub_window_size:j]
                    volatilities.append(np.std(sub_window))
                
                if len(volatilities) >= 4:
                    # ボラティリティの周期性をFFTで検出
                    fft_vol = fft(volatilities)
                    power_spectrum = np.abs(fft_vol)**2
                    gravitational_wave = np.max(power_spectrum[1:len(power_spectrum)//2]) / np.sum(power_spectrum[1:])
                else:
                    gravitational_wave = 0
            else:
                gravitational_wave = 0
            
            # 3. 恒星脈動（主要周波数の相対強度）
            if len(window) >= 8:
                fft_vals = fft(window)
                power_spectrum = np.abs(fft_vals)**2
                max_power = np.max(power_spectrum[1:len(power_spectrum)//2])
                total_power = np.sum(power_spectrum[1:])
                stellar_pulsation = max_power / (total_power + 1e-10)
            else:
                stellar_pulsation = 0
            
            # 4. 宇宙膨張（総リターン）
            if len(window) > 1:
                cosmic_expansion = (window[-1] - window[0]) / window[0] if window[0] != 0 else 0
            else:
                cosmic_expansion = 0
            
            # 5. ダークエネルギー（加速度）
            if len(window) >= 3:
                velocity = np.diff(window)
                acceleration = np.diff(velocity)
                dark_energy = np.mean(acceleration) if len(acceleration) > 0 else 0
            else:
                dark_energy = 0
            
            # 6. ビッグバンエコー（最大ショックの時間荷重影響）
            if len(window) > 1:
                returns = np.diff(window)
                max_shock_idx = np.argmax(np.abs(returns))
                time_since_shock = len(returns) - max_shock_idx
                shock_magnitude = abs(returns[max_shock_idx])
                # 時間と共に減衰する影響
                big_bang_echo = shock_magnitude * np.exp(-time_since_shock / len(returns))
            else:
                big_bang_echo = 0
            
            astronomy_results[i] = [orbital_mechanics, gravitational_wave, stellar_pulsation,
                                cosmic_expansion, dark_energy, big_bang_echo]
        
        astronomy_names = ['orbital_mechanics', 'gravitational_wave', 'stellar_pulsation',
                        'cosmic_expansion', 'dark_energy', 'big_bang_echo']
        
        for j, name in enumerate(astronomy_names):
            features[f'{name}_{window_size}'] = astronomy_results[:, j]
        
        return features

    # 生体力学・パフォーマンス科学特徴量
    def calculate_biomechanics_features(self, data: np.ndarray, window_size: int = 50) -> Dict[str, np.ndarray]:
        """生体力学・パフォーマンス科学特徴量"""
        features = {}
        n = len(data)
        
        biomechanics_results = np.zeros((n, 7))
        
        for i in range(window_size-1, n):
            window = data[i-window_size+1:i+1]
            
            if len(window) < 2:
                continue

            # リターンを速度として扱う
            returns = np.diff(window)
            
            # 1. 運動エネルギー (Kinetic Energy) - 動きの激しさ
            kinetic_energy = 0.5 * np.sum(returns**2)
            
            # 2. 位置エネルギー (Potential Energy) - 価格水準からの乖離エネルギー
            potential_energy = np.sum((window - np.min(window))**2)
            
            # 3. 筋力 (Muscle Force) - 瞬発的な最大変動
            muscle_force = np.max(np.abs(returns)) if len(returns) > 0 else 0
            
            # 4. 関節可動域 (Joint Mobility) - 変動範囲の相対的な広さ
            price_range = np.max(window) - np.min(window)
            mean_price = np.mean(window)
            joint_mobility = price_range / mean_price if mean_price > 1e-10 else 0
            
            # 5. パフォーマンスの一貫性 (Performance Consistency) - 安定性
            std_returns = np.std(returns) if len(returns) > 0 else 0
            performance_consistency = 1 / (std_returns + 1e-10)
            
            # 6. 持久力 (Endurance) - 上昇トレンドの持続性
            endurance = np.sum(returns > 0) / len(returns) if len(returns) > 0 else 0.5
            
            # 7. 回復率 (Recovery Rate) - 最大ドローダウンからの回復
            cumulative_returns = np.cumsum(np.insert(returns, 0, 0))
            max_drawdown = np.max(np.maximum.accumulate(cumulative_returns) - cumulative_returns)
            total_return = cumulative_returns[-1]
            recovery_rate = total_return / (max_drawdown + 1e-10) if max_drawdown > 1e-10 else 1.0

            biomechanics_results[i] = [kinetic_energy, potential_energy, muscle_force,
                                       joint_mobility, performance_consistency, endurance, recovery_rate]
            
        biomechanics_names = ['kinetic_energy', 'potential_energy', 'muscle_force',
                              'joint_mobility', 'performance_consistency', 'endurance', 'recovery_rate']
        
        for j, name in enumerate(biomechanics_names):
            features[f'{name}_{window_size}'] = biomechanics_results[:, j]
            
        return features

    # =============================================================================
    # 統合メソッド: 全欠落特徴量の計算
    # =============================================================================

    def calculate_all_missing_features(self, high: np.ndarray, low: np.ndarray, close: np.ndarray, 
                                    volume: np.ndarray, open_prices: np.ndarray = None) -> Dict[str, np.ndarray]:
        """全ての欠落特徴量を統合計算"""
        logger.info("欠落特徴量の統合計算を開始")
        
        all_features = {}
        
        # Open価格がない場合はCloseで代用
        if open_prices is None:
            open_prices = np.roll(close, 1)
            open_prices[0] = close[0]
        
        try:
            # Block 1: ADX・基本オシレーター
            logger.info("ADX・基本オシレーター計算中...")
            all_features.update(self.calculate_adx_features(high, low, close))
            all_features.update(self.calculate_parabolic_sar_features(high, low, close))
            all_features.update(self.calculate_cci_features(high, low, close))
            all_features.update(self.calculate_williams_r_features(high, low, close))
            all_features.update(self.calculate_aroon_features(high, low))
            all_features.update(self.calculate_ultimate_oscillator_features(high, low, close))
            
            # Block 2: 出来高関連指標
            logger.info("出来高関連指標計算中...")
            all_features.update(self.calculate_vpt_features(close, volume))
            all_features.update(self.calculate_ad_line_features(high, low, close, volume))
            all_features.update(self.calculate_cmf_features(high, low, close, volume))
            all_features.update(self.calculate_chaikin_oscillator_features(high, low, close, volume))
            all_features.update(self.calculate_mfi_features(high, low, close, volume))
            all_features.update(self.calculate_vwap_features(high, low, close, volume))
            all_features.update(self.calculate_volume_oscillator_features(volume))
            all_features.update(self.calculate_ease_of_movement_features(high, low, volume))
            all_features.update(self.calculate_volume_roc_features(volume))
            
            # Block 3: トレンド分析・移動平均線
            logger.info("トレンド分析・移動平均線計算中...")
            all_features.update(self.calculate_wma_features(close))
            all_features.update(self.calculate_hma_features(close))
            all_features.update(self.calculate_tma_features(close))
            all_features.update(self.calculate_kama_features(close))
            all_features.update(self.calculate_zlema_features(close))
            all_features.update(self.calculate_dema_features(close))
            all_features.update(self.calculate_tema_features(close))
            all_features.update(self.calculate_ma_slope_features(close))
            all_features.update(self.calculate_ma_deviation_features(close))
            all_features.update(self.calculate_golden_death_cross_features(close))
            
            # Block 4: ボラティリティ・バンド指標
            logger.info("ボラティリティ・バンド指標計算中...")
            all_features.update(self.calculate_bb_squeeze_features(close, high, low))
            all_features.update(self.calculate_keltner_channel_features(high, low, close))
            all_features.update(self.calculate_donchian_channel_features(high, low, close))
            all_features.update(self.calculate_atr_bands_features(high, low, close))
            all_features.update(self.calculate_historical_volatility_features(close))
            all_features.update(self.calculate_volatility_ratio_features(close))
            all_features.update(self.calculate_chandelier_exit_features(high, low, close))
            all_features.update(self.calculate_volatility_breakout_features(high, low, close))
            
            # Block 5: サポート・レジスタンス・ローソク足
            logger.info("サポート・レジスタンス・ローソク足計算中...")
            all_features.update(self.calculate_pivot_points_features(high, low, close))
            all_features.update(self.calculate_price_channels_features(high, low, close))
            all_features.update(self.calculate_fibonacci_features(high, low, close))
            all_features.update(self.calculate_support_resistance_features(high, low, close))
            all_features.update(self.calculate_candlestick_patterns(open_prices, high, low, close))
            all_features.update(self.calculate_multi_candle_patterns(open_prices, high, low, close))
            all_features.update(self.calculate_candle_strength_features(open_prices, high, low, close))
            
            # Block 6: 数学・統計学応用特徴量
            logger.info("数学・統計学応用特徴量計算中...")
            all_features.update(self.calculate_distribution_fitting_features(close))
            all_features.update(self.calculate_order_statistics_features(close))
            all_features.update(self.calculate_normality_tests_features(close))
            all_features.update(self.calculate_autocorrelation_features(close))
            all_features.update(self.calculate_cross_correlation_features(close, volume))
            all_features.update(self.calculate_advanced_moments_features(close))
            all_features.update(self.calculate_hypothesis_test_features(close))
            
            # Block 7: 物理学・信号処理応用特徴量
            logger.info("物理学・信号処理応用特徴量計算中...")
            all_features.update(self.calculate_hilbert_transform_features(close))
            all_features.update(self.calculate_coherence_features(close, volume))
            all_features.update(self.calculate_zero_crossing_features(close))
            all_features.update(self.calculate_filter_features(close))
            all_features.update(self.calculate_energy_features(close))
            all_features.update(self.calculate_advanced_fft_features(close))
            all_features.update(self.calculate_cwt_features(close))
            all_features.update(self.calculate_spectral_analysis_features(close))
            all_features.update(self.calculate_signal_complexity_features(close))
            
            # Block 8: 情報理論特徴量
            logger.info("情報理論特徴量計算中...")
            all_features.update(self.calculate_advanced_entropy_features(close))
            all_features.update(self.calculate_sample_approximate_entropy(close))
            all_features.update(self.calculate_lempel_ziv_features(close))
            all_features.update(self.calculate_kolmogorov_complexity_features(close))
            all_features.update(self.calculate_mutual_information_features(close, volume))
            
            # Block 9: 学際的アナロジー特徴量
            logger.info("学際的アナロジー特徴量計算中...")
            all_features.update(self.calculate_game_theory_features(close))
            all_features.update(self.calculate_molecular_science_features(close))
            all_features.update(self.calculate_network_science_features(close))
            all_features.update(self.calculate_acoustics_features(close))
            all_features.update(self.calculate_linguistics_features(close))
            all_features.update(self.calculate_aesthetics_features(close))
            all_features.update(self.calculate_music_theory_features(close))
            all_features.update(self.calculate_astronomy_features(close))
            all_features.update(self.calculate_biomechanics_features(close))
            
            logger.info(f"欠落特徴量計算完了: {len(all_features)}個の特徴量を生成")
            
            return all_features
            
        except Exception as e:
            logger.error(f"欠落特徴量計算中にエラーが発生: {e}")
            import traceback
            traceback.print_exc()
            return all_features

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

    def save_features_polars(self, features_dict: Dict[str, np.ndarray], output_filename: str) -> Path:
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
                    # チャンクベース処理の実行
                    final_output_path = self._execute_chunk_based_processing(current_memmap, tf, test_mode)
                    
                    # 実行ごとのメタデータを準備し、リストに追加
                    calc_metadata = {
                        'timeframe': tf,
                        'test_mode': test_mode,
                        'calculation_summary': self.calculator.get_calculation_summary(),
                        'memory_summary': self.memory_manager.get_memory_summary(),
                        'data_shape': current_memmap.shape,
                        'total_chunks_processed': self.execution_stats['total_chunks_processed'],
                        'output_file': final_output_path
                    }
                    all_run_metadata.append(calc_metadata)
                    
                    logger.info(f"タイムフレーム {tf} 処理完了: {self.execution_stats['total_chunks_processed']}チャンク処理")
                    
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
            
            # 処理結果の集計
            processing_results = {
                'total_time_minutes': total_time / 60,
                'total_features_generated': self.execution_stats['total_features_generated'],
                'total_chunks_processed': self.execution_stats['total_chunks_processed'],
                'data_points_processed': sum(arr.shape[0] for arr in memmap_data.values()),
                'success_rate': (len(target_timeframes) - len(self.execution_stats['processing_errors'])) / len(target_timeframes),
                'processing_errors': self.execution_stats['processing_errors'],
                'target_timeframes': target_timeframes,
                'test_mode': test_mode,
                'chunk_size': self.chunk_size
            }
            
            # タイムフレーム別メタデータの統合
            processing_results['timeframe_details'] = all_run_metadata
            
            # 最終サマリーの保存と表示
            self.output_manager.save_final_summary_metadata(processing_results)
            self.output_manager.log_final_summary(processing_results)
            
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
                                      timeframe: str, test_mode: bool) -> str:
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
                actual_chunk_size = end_row - start_row
                
                logger.info(f"チャンク {chunk_idx+1}/{n_chunks} 処理中: 行 {start_row:,}-{end_row:,} ({actual_chunk_size:,}行)")
                
                try:
                    # ステップ1: チャンクデータをメモリに読み込み
                    chunk_data = memmap_data[start_row:end_row].copy()
                    logger.debug(f"チャンク {chunk_idx+1}: メモリ読み込み完了")
                    
                    # ステップ2: チャンクに対する特徴量計算
                    chunk_features = self._calculate_features_for_chunk(chunk_data, chunk_idx)
                    logger.debug(f"チャンク {chunk_idx+1}: 特徴量計算完了 ({len(chunk_features)}特徴量)")
                    
                    # ステップ3: 一時Parquetファイルとして保存
                    temp_file_path = self._save_chunk_to_temp_file(chunk_features, chunk_idx, timeframe, test_mode)
                    temp_files.append(temp_file_path)
                    logger.debug(f"チャンク {chunk_idx+1}: 一時ファイル保存完了")
                    
                    # ステップ4: メモリ解放
                    del chunk_data, chunk_features
                    self.memory_manager.force_garbage_collection()
                    
                    self.execution_stats['total_chunks_processed'] += 1
                    
                    # 進捗表示
                    progress = (chunk_idx + 1) / n_chunks * 100
                    logger.info(f"チャンク処理進捗: {progress:.1f}% ({chunk_idx+1}/{n_chunks})")
                    
                except Exception as e:
                    error_msg = f"チャンク {chunk_idx+1} 処理エラー: {e}"
                    logger.error(error_msg)
                    self.execution_stats['processing_errors'].append(error_msg)
                    continue
            
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
        単一チャンクに対する全特徴量計算
        """
        logger.debug(f"チャンク {chunk_idx+1} 特徴量計算開始: shape={chunk_data.shape}")
        
        try:
            # 価格データ抽出（OHLCV構造を想定）
            close_prices = chunk_data[:, 4]  # close列（インデックス4）
            high_prices = chunk_data[:, 2] if chunk_data.shape[1] > 2 else close_prices
            low_prices = chunk_data[:, 3] if chunk_data.shape[1] > 3 else close_prices
            volume_data = chunk_data[:, 5] if chunk_data.shape[1] > 5 else np.ones_like(close_prices)
            
            # 基本テクニカル指標
            basic_features = self._calculate_basic_technical_indicators(
                close_prices, high_prices, low_prices, volume_data
            )
            logger.debug(f"チャンク {chunk_idx+1}: 基本テクニカル指標 {len(basic_features)}特徴量")
            
            # 高度特徴量（MFDFA等）
            advanced_features = self.calculator.calculate_all_advanced_features(
                close_prices, high_prices, low_prices, volume_data
            )
            logger.debug(f"チャンク {chunk_idx+1}: 高度特徴量 {len(advanced_features)}特徴量")
            
            # 統合
            all_features = {}
            all_features.update(basic_features)
            all_features.update(advanced_features)
            
            # 品質監視（チャンク単位）
            quality_report = self.calculator.monitor_feature_quality_continuous(all_features)
            if quality_report['alerts']:
                self.execution_stats['quality_alerts'].extend(quality_report['alerts'])
                logger.debug(f"チャンク {chunk_idx+1}: 品質アラート {len(quality_report['alerts'])}件")
            
            self.execution_stats['total_features_generated'] += len(all_features)
            
            return all_features
            
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
            self.output_manager.save_features_polars(chunk_features, temp_filename)
            
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
    
    def _calculate_basic_technical_indicators(self, close: np.ndarray, high: np.ndarray, 
                                            low: np.ndarray, volume: np.ndarray) -> Dict[str, np.ndarray]:
        """
        基本テクニカル指標群を計算（Pandas依存を排除したNumba完全対応版）
        """
        features = {}
        n = len(close)
        
        try:
            # RSI
            for period in self.calculator.params['rsi_periods']:
                if n > period:
                    rsi = self.calculator._compute_rsi_numba(close, period)
                    features[f'rsi_{period}'] = rsi
                    features[f'rsi_{period}_normalized'] = (rsi - 50) / 50

            # MACD
            for fast, slow, signal in self.calculator.params['macd_settings']:
                if n > slow:
                    macd, sig, hist = self.calculator._compute_macd_numba(close, fast, slow, signal)
                    features[f'macd_{fast}_{slow}'] = macd
                    features[f'macd_signal_{fast}_{slow}'] = sig
                    features[f'macd_hist_{fast}_{slow}'] = hist

            # Bollinger Bands
            for period, std_mult in self.calculator.params['bollinger_settings']:
                if n > period:
                    bb_upper, bb_mid, bb_lower = self.calculator._compute_bollinger_bands_numba(close, period, std_mult)
                    features[f'bb_position_{period}'] = (close - bb_lower) / (bb_upper - bb_lower + 1e-8)
                    features[f'bb_width_{period}'] = (bb_upper - bb_lower) / (bb_mid + 1e-8)

            # ATR
            for period in self.calculator.params['atr_periods']:
                if n > period:
                    atr = self.calculator._compute_atr_numba(high, low, close, period)
                    features[f'atr_{period}'] = atr
                    features[f'atr_ratio_{period}'] = atr / (close + 1e-8)
                    features[f'atr_percent_{period}'] = (atr / (close + 1e-8)) * 100

            # Stochastic
            for k_period, d_period in self.calculator.params['stochastic_settings']:
                if n > k_period + d_period:
                    k, d = self.calculator._compute_stochastic_numba(high, low, close, k_period, d_period)
                    features[f'stoch_k_{k_period}_{d_period}'] = k
                    features[f'stoch_d_{k_period}_{d_period}'] = d
                    features[f'stoch_overbought_{k_period}'] = (k > 80).astype(float)
                    features[f'stoch_oversold_{k_period}'] = (k < 20).astype(float)

            # 移動平均群 (Numba化)
            for period in self.calculator.params['short_ma_periods'] + self.calculator.params['long_ma_periods']:
                if n > period:
                    sma = self._compute_rolling_mean_numba(close, period)
                    ema = self._compute_ema_numba(close, period)
                    features[f'sma_{period}'] = sma
                    features[f'price_above_sma_{period}'] = (close > sma).astype(float)
                    features[f'ema_{period}'] = ema
                    features[f'price_above_ema_{period}'] = (close > ema).astype(float)

            # 出来高指標
            if n > 1:
                obv = np.cumsum(np.where(np.diff(close, prepend=close[0]) > 0, volume, 
                                       np.where(np.diff(close, prepend=close[0]) < 0, -volume, 0)))
                features['obv'] = obv

            # 統計的特徴量 (Numba化)
            returns = np.diff(np.log(close + 1e-10), prepend=np.log(close[0] + 1e-10))
            for window in self.calculator.params['stat_windows']:
                if n > window:
                    mean, var, skew, kurt = self._compute_rolling_stats_numba(returns, window)
                    features[f'mean_{window}'] = mean
                    features[f'variance_{window}'] = var
                    features[f'skewness_{window}'] = skew
                    features[f'kurtosis_{window}'] = kurt
            
            logger.debug(f"基本テクニカル指標計算完了: {len(features)}特徴量")

        except Exception as e:
            logger.error(f"基本テクニカル指標計算エラー: {e}", exc_info=True)
        
        return features

    # ヘルパーメソッド（Numba最適化）
    @staticmethod
    @njit(cache=True)
    def _compute_rolling_mean_numba(data: np.ndarray, period: int) -> np.ndarray:
        n = len(data)
        output = np.full(n, np.nan)
        current_sum = 0.0
        for i in range(n):
            current_sum += data[i]
            if i >= period:
                current_sum -= data[i - period]
            if i >= period - 1:
                output[i] = current_sum / period
        return output

    @staticmethod
    @njit(cache=True)
    def _compute_ema_numba(data: np.ndarray, period: int) -> np.ndarray:
        n = len(data)
        output = np.full(n, np.nan)
        alpha = 2.0 / (period + 1.0)
        if n > 0:
            output[0] = data[0]
            for i in range(1, n):
                output[i] = alpha * data[i] + (1.0 - alpha) * output[i-1]
        return output

    @staticmethod
    @njit(cache=True)
    def _compute_rolling_stats_numba(data: np.ndarray, period: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        n = len(data)
        mean_out = np.full(n, np.nan)
        var_out = np.full(n, np.nan)
        skew_out = np.full(n, np.nan)
        kurt_out = np.full(n, np.nan)

        for i in range(period - 1, n):
            window = data[i - period + 1 : i + 1]
            mean = np.mean(window)
            std = np.std(window)
            mean_out[i] = mean
            var_out[i] = std**2
            if std > 1e-10:
                centered = window - mean
                skew_out[i] = np.mean((centered / std)**3)
                kurt_out[i] = np.mean((centered / std)**4) - 3.0
        return mean_out, var_out, skew_out, kurt_out

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
        """計算エンジン検証"""
        logger.info("計算エンジンテスト実行中...")
        
        try:
            # ダミーデータ生成
            np.random.seed(42)
            test_size = 1000 if test_mode else 100
            test_data = np.random.randn(test_size) * 0.01 + 100  # 価格データ風
            
            # Calculator初期化
            dummy_window_manager = WindowManager(window_size=50, overlap=0.3)
            dummy_memory_manager = MemoryManager()
            calculator = Calculator(dummy_window_manager, dummy_memory_manager)
            
            # 基本統計計算テスト
            statistical_features = calculator.calculate_statistical_moments(test_data, 20)
            robust_features = calculator.calculate_robust_statistics(test_data, 20)
            spectral_features = calculator.calculate_spectral_features(test_data, 32)
            
            # 高度特徴量テスト（データ長チェック付き）
            if len(test_data) >= 200:  # MFDFA用に十分な長さ
                mfdfa_features = calculator.calculate_mfdfa_features(test_data[:200])
                noise_features = calculator.calculate_microstructure_noise_features(test_data[:200])
            else:
                mfdfa_features = {}
                noise_features = {}
            
            # 計算結果検証
            all_features = {}
            all_features.update(statistical_features)
            all_features.update(robust_features)
            all_features.update(spectral_features)
            all_features.update(mfdfa_features)
            all_features.update(noise_features)
            
            # 品質スコア計算
            quality_scores = []
            nan_counts = []
            for name, values in all_features.items():
                if isinstance(values, np.ndarray) and len(values) > 0:
                    quality = calculator.calculate_quality_score(values)
                    quality_scores.append(quality)
                    nan_counts.append(np.isnan(values).sum())
            
            avg_quality = np.mean(quality_scores) if quality_scores else 0.0
            success_rate = len([q for q in quality_scores if q > 0.6]) / max(1, len(quality_scores))
            total_nan = sum(nan_counts)
            
            return {
                'status': 'success' if success_rate >= 0.6 and total_nan < len(test_data) * 0.1 else 'warning',
                'total_features': len(all_features),
                'average_quality': avg_quality,
                'success_rate': success_rate,
                'total_nan_values': total_nan,
                'calculation_summary': calculator.get_calculation_summary()
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
                output_path = output_manager.save_features_polars(dummy_features, test_filename)
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