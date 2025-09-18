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
# Calculator クラス - Polars第一主義・高速化リファクタリング版（ブロック1）
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
    Polars第一主義・高速化Calculator
    
    設計思想:
    1. 脱・行ごとループ: 各メソッドはチャンク全体を一度に処理
    2. Polars第一主義: 標準的な計算はPolars Expressionで実装
    3. Numbaの限定的投入: カスタムアルゴリズムに限定
    4. ベクトル化処理: 可能な限りNumPy配列演算で並列処理
    """
    
    def __init__(self, window_manager=None, memory_manager=None):
        self.window_manager = window_manager
        self.memory_manager = memory_manager
        
        # 計算パラメータ（Polars用に最適化）
        self.params = self._initialize_polars_params()
        
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
            'polars_calculations': 0,
            'numba_calculations': 0,
            'computation_times': []
        }
        
        logger.info("Calculator初期化完了 - Polars第一主義設計")

    def _initialize_polars_params(self) -> Dict[str, Any]:
        """Polars最適化されたパラメータセット"""
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
            
            # ボラティリティ・バンド用
            'volatility_bb_settings': [(20, 2.0), (50, 2.5)],
            'kc_periods': [20, 50],
            'dc_periods': [20, 50, 100],
            'atr_periods_vol': [14, 20],
            'hist_vol_periods': [20, 50, 100],
            
            # 移動平均・トレンド用
            'ma_deviation_periods': [20, 50, 100],
            'tma_periods': [14, 21, 50],
            'zlema_periods': [12, 21],
            'dema_periods': [12, 21],
            'tema_periods': [12, 21],
            
            # 出来高用
            'cmf_periods': [20, 50],
            'mfi_periods': [14, 21],
            'vol_roc_periods': [10, 20],
            
            # ADX・オシレーター用
            'adx_periods': [14, 21],
            'cci_periods': [14, 20],
            'williams_r_periods': [14, 21],
            'aroon_periods': [14, 25],
            
            # サポート・レジスタンス用
            'price_channel_periods': [20, 50, 100],
            
            # フィルタリング用
            'gaussian_sigmas': [1.0, 2.0, 3.0],
            'median_sizes': [3, 5, 7],
            'savgol_windows': [5, 11, 21],
            
            # 物理・信号処理用
            'fourier_windows': [32, 64, 128],
            'spectral_windows': [64, 128],
            'energy_windows': [20, 50],
            'cwt_windows': [100, 200],
            
            # 情報理論用
            'lz_windows': [50, 100],
            'kolmogorov_windows': [50, 100],
            'mutual_info_lags': [1, 2, 3, 5],
            'adv_entropy_windows': [50, 100]
        }

    # =========================================================================
    # Polars統合ユーティリティメソッド
    # =========================================================================
    
    def _ensure_polars_df(self, data: Union[np.ndarray, pl.DataFrame], 
                          column_name: str = "value") -> pl.DataFrame:
        """データをPolars DataFrameに変換"""
        if isinstance(data, pl.DataFrame):
            return data
        elif isinstance(data, np.ndarray):
            return pl.DataFrame({column_name: data})
        else:
            return pl.DataFrame({column_name: np.asarray(data)})
    
    def _polars_safe_calculation(self, func, *args, **kwargs) -> Union[pl.DataFrame, np.ndarray]:
        """Polars計算の安全なラッパー"""
        start_time = time.time()
        self.calculation_stats['total_calculations'] += 1
        
        try:
            result = func(*args, **kwargs)
            self.calculation_stats['successful_calculations'] += 1
            self.calculation_stats['polars_calculations'] += 1
            
            computation_time = time.time() - start_time
            self.calculation_stats['computation_times'].append(computation_time)
            
            return result
            
        except Exception as e:
            logger.debug(f"Polars計算エラー in {func.__name__}: {e}")
            # フォールバック値を返す
            if 'data' in kwargs and isinstance(kwargs['data'], (np.ndarray, pl.DataFrame)):
                data_len = len(kwargs['data'])
                return np.zeros(data_len)
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
    # Polarsベース基礎統計・テクニカル指標
    # =========================================================================
    
    def calculate_basic_technical_indicators_polars(self, 
                                                   high: np.ndarray, 
                                                   low: np.ndarray, 
                                                   close: np.ndarray, 
                                                   volume: np.ndarray) -> Dict[str, np.ndarray]:
        """Polarsベース基礎テクニカル指標計算"""
        features = {}
        
        # Polars DataFrameを構築
        df = pl.DataFrame({
            'high': high,
            'low': low, 
            'close': close,
            'volume': volume
        })
        
        try:
            # RSI（Polars実装）
            for period in self.params['rsi_periods']:
                rsi_result = self._polars_safe_calculation(
                    self._calculate_rsi_polars, df, period
                )
                if isinstance(rsi_result, np.ndarray):
                    features[f'rsi_{period}'] = rsi_result
                
            # MACD（Polars実装）
            for fast, slow, signal in self.params['macd_periods']:
                macd_result = self._polars_safe_calculation(
                    self._calculate_macd_polars, df, fast, slow, signal
                )
                if isinstance(macd_result, dict):
                    features.update(macd_result)
                
            # ボリンジャーバンド（Polars実装）
            for period in self.params['bb_periods']:
                for std_mult in self.params['bb_std_multipliers']:
                    bb_result = self._polars_safe_calculation(
                        self._calculate_bollinger_bands_polars, df, period, std_mult
                    )
                    if isinstance(bb_result, dict):
                        features.update(bb_result)
            
            # ATR（Polars実装）
            for period in self.params['atr_periods']:
                atr_result = self._polars_safe_calculation(
                    self._calculate_atr_polars, df, period
                )
                if isinstance(atr_result, np.ndarray):
                    features[f'atr_{period}'] = atr_result
            
            # Stochastic（Polars実装）
            for k_period, d_period in self.params['stoch_periods']:
                stoch_result = self._polars_safe_calculation(
                    self._calculate_stochastic_polars, df, k_period, d_period
                )
                if isinstance(stoch_result, dict):
                    features.update(stoch_result)
                    
        except Exception as e:
            logger.error(f"基礎テクニカル指標計算エラー: {e}")
        
        return features
    
    def _calculate_rsi_polars(self, df: pl.DataFrame, period: int) -> np.ndarray:
        """RSI計算（Polarsベース）"""
        result_df = df.with_columns([
            pl.col('close').diff().alias('price_change')
        ]).with_columns([
            pl.when(pl.col('price_change') > 0)
            .then(pl.col('price_change'))
            .otherwise(0.0)
            .alias('gains'),
            pl.when(pl.col('price_change') < 0)
            .then(-pl.col('price_change'))
            .otherwise(0.0)
            .alias('losses')
        ]).with_columns([
            pl.col('gains').ewm_mean(span=period, adjust=False).alias('avg_gain'),
            pl.col('losses').ewm_mean(span=period, adjust=False).alias('avg_loss')
        ]).with_columns([
            (100.0 - (100.0 / (1.0 + (pl.col('avg_gain') / (pl.col('avg_loss') + 1e-10)))))
            .alias('rsi')
        ])
        
        return result_df['rsi'].to_numpy()
    
    def _calculate_macd_polars(self, df: pl.DataFrame, fast: int, slow: int, signal: int) -> Dict[str, np.ndarray]:
        """MACD計算（Polarsベース）"""
        result_df = df.with_columns([
            pl.col('close').ewm_mean(span=fast, adjust=False).alias('ema_fast'),
            pl.col('close').ewm_mean(span=slow, adjust=False).alias('ema_slow')
        ]).with_columns([
            (pl.col('ema_fast') - pl.col('ema_slow')).alias('macd')
        ]).with_columns([
            pl.col('macd').ewm_mean(span=signal, adjust=False).alias('signal_line')
        ]).with_columns([
            (pl.col('macd') - pl.col('signal_line')).alias('histogram')
        ])
        
        return {
            f'macd_{fast}_{slow}_{signal}': result_df['macd'].to_numpy(),
            f'macd_signal_{fast}_{slow}_{signal}': result_df['signal_line'].to_numpy(),
            f'macd_histogram_{fast}_{slow}_{signal}': result_df['histogram'].to_numpy()
        }
    
    def _calculate_bollinger_bands_polars(self, df: pl.DataFrame, period: int, std_mult: float) -> Dict[str, np.ndarray]:
        """ボリンジャーバンド計算（Polarsベース）"""
        result_df = df.with_columns([
            pl.col('close').rolling_mean(window_size=period).alias('sma'),
            pl.col('close').rolling_std(window_size=period).alias('std')
        ]).with_columns([
            (pl.col('sma') + std_mult * pl.col('std')).alias('bb_upper'),
            pl.col('sma').alias('bb_middle'),
            (pl.col('sma') - std_mult * pl.col('std')).alias('bb_lower')
        ])
        
        return {
            f'bb_upper_{period}_{std_mult}': result_df['bb_upper'].to_numpy(),
            f'bb_middle_{period}_{std_mult}': result_df['bb_middle'].to_numpy(),
            f'bb_lower_{period}_{std_mult}': result_df['bb_lower'].to_numpy()
        }
    
    def _calculate_atr_polars(self, df: pl.DataFrame, period: int) -> np.ndarray:
        """ATR計算（Polarsベース）"""
        result_df = df.with_columns([
            pl.col('close').shift(1).alias('prev_close')
        ]).with_columns([
            pl.max_horizontal([
                pl.col('high') - pl.col('low'),
                (pl.col('high') - pl.col('prev_close')).abs(),
                (pl.col('low') - pl.col('prev_close')).abs()
            ]).alias('true_range')
        ]).with_columns([
            pl.col('true_range').ewm_mean(span=period, adjust=False).alias('atr')
        ])
        
        return result_df['atr'].to_numpy()
    
    def _calculate_stochastic_polars(self, df: pl.DataFrame, k_period: int, d_period: int) -> Dict[str, np.ndarray]:
        """Stochastic計算（Polarsベース）"""
        result_df = df.with_columns([
            pl.col('high').rolling_max(window_size=k_period).alias('highest_high'),
            pl.col('low').rolling_min(window_size=k_period).alias('lowest_low')
        ]).with_columns([
            (100.0 * (pl.col('close') - pl.col('lowest_low')) / 
             (pl.col('highest_high') - pl.col('lowest_low') + 1e-10)).alias('percent_k')
        ]).with_columns([
            pl.col('percent_k').rolling_mean(window_size=d_period).alias('percent_d')
        ])
        
        return {
            f'stoch_k_{k_period}_{d_period}': result_df['percent_k'].to_numpy(),
            f'stoch_d_{k_period}_{d_period}': result_df['percent_d'].to_numpy()
        }
    
    # =========================================================================
    # Tier S特徴量: MFDFA (Multi-Fractal Detrended Fluctuation Analysis)
    # =========================================================================
    
    def calculate_mfdfa_features_polars(self, data: np.ndarray) -> Dict[str, np.ndarray]:
        """MFDFA特徴量計算（NumPy主体+Polars補助版）"""
        features = {}
        
        # データをNumPy配列として直接処理
        if isinstance(data, np.ndarray):
            price_data = data.flatten() if data.ndim > 1 else data
        else:
            # Polarsから変換
            df = self._ensure_polars_df(data, 'price')
            price_data = df["price"].to_numpy()
        
        n = len(price_data)
        
        # MFDFA用パラメータ
        q_range = self.params['mfdfa_q_range']
        scales = self.params['mfdfa_scales']
        scales = scales[scales < n//4]
        
        window_size = 200
        if n < window_size:
            logger.warning(f"データ長{n}がMFDFA計算に不十分です")
            return features
        
        try:
            # NumPyで前処理（Polarsの問題を回避）
            price_mean = np.mean(price_data)
            price_centered = price_data - price_mean
            profile = np.cumsum(price_centered)
            
            # ローリングウィンドウ用のインデックス
            indices = np.arange(window_size-1, n)
            
            # MFDFA計算
            mfdfa_results = self._numba_safe_calculation(
                self._calculate_mfdfa_vectorized,
                profile,
                indices,
                window_size,
                q_range,
                scales
            )
            
            if mfdfa_results is not None and len(mfdfa_results.shape) == 2:
                result_df = pd.DataFrame(mfdfa_results)
                
                # 特徴量作成
                for j, q in enumerate(q_range):
                    feature_name = f'mfdfa_hurst_q{q}'
                    if j < result_df.shape[1]:
                        features[feature_name] = np.pad(
                            result_df.iloc[:, j].values, 
                            (window_size-1, 0), 
                            mode='constant', 
                            constant_values=np.nan
                        )
                
                # 統合特徴量
                if result_df.shape[1] >= len(q_range) + 3:
                    features['mfdfa_multifractal_width'] = np.pad(
                        result_df.iloc[:, -3].values, (window_size-1, 0), 
                        mode='constant', constant_values=np.nan
                    )
                    features['mfdfa_asymmetry'] = np.pad(
                        result_df.iloc[:, -2].values, (window_size-1, 0), 
                        mode='constant', constant_values=np.nan
                    )
                    features['mfdfa_complexity'] = np.pad(
                        result_df.iloc[:, -1].values, (window_size-1, 0), 
                        mode='constant', constant_values=np.nan
                    )
                    
        except Exception as e:
            logger.error(f"MFDFA計算エラー: {e}")
            # フォールバック値
            for q in q_range:
                features[f'mfdfa_hurst_q{q}'] = np.zeros(n)
            features['mfdfa_multifractal_width'] = np.zeros(n)  
            features['mfdfa_asymmetry'] = np.zeros(n)
            features['mfdfa_complexity'] = np.zeros(n)
        
        return features
    
    @staticmethod
    @njit(parallel=True, cache=True)
    def _calculate_mfdfa_vectorized(profile: np.ndarray, 
                                  indices: np.ndarray,
                                  window_size: int,
                                  q_range: np.ndarray, 
                                  scales: np.ndarray) -> np.ndarray:
        """
        MFDFA核心計算（完全ベクトル化・並列版）
        """
        n_windows = len(indices)
        n_q = len(q_range)
        results = np.zeros((n_windows, n_q + 3))
        
        # 並列計算
        for idx in prange(n_windows):
            i = indices[idx]
            window_profile = profile[i-window_size+1:i+1]
            
            if len(window_profile) < 20:
                continue
                
            hurst_values = np.zeros(n_q)
            
            # 各q値に対してスケーリング解析
            for qi in range(n_q):
                q = q_range[qi]
                fluctuations = np.zeros(len(scales))
                
                for si in range(len(scales)):
                    s = scales[si]
                    if s < 4 or s >= len(window_profile)//2:
                        continue
                    
                    # プロファイルを非重複セグメントに分割
                    Ns = len(window_profile) // s
                    F2_segments = np.zeros(Ns)
                    
                    for v in range(Ns):
                        start_idx = v * s
                        end_idx = (v + 1) * s
                        segment = window_profile[start_idx:end_idx]
                        
                        # 線形デトレンディング
                        x = np.arange(s, dtype=np.float64)
                        mean_x = np.mean(x)
                        mean_y = np.mean(segment)
                        
                        numerator = np.sum((x - mean_x) * (segment - mean_y))
                        denominator = np.sum((x - mean_x)**2)
                        
                        if denominator > 1e-10:
                            slope = numerator / denominator
                            intercept = mean_y - slope * mean_x
                            trend = slope * x + intercept
                            detrended = segment - trend
                        else:
                            detrended = segment - mean_y
                        
                        F2_segments[v] = np.mean(detrended**2)
                    
                    # q次揺動関数
                    if q == 0:
                        valid_segments = F2_segments[F2_segments > 1e-10]
                        if len(valid_segments) > 0:
                            fluctuations[si] = np.exp(0.5 * np.mean(np.log(valid_segments)))
                    elif q == 2:
                        fluctuations[si] = np.sqrt(np.mean(F2_segments))
                    else:
                        positive_segments = F2_segments[F2_segments > 1e-10]
                        if len(positive_segments) > 0:
                            Fq_segments = np.power(positive_segments, q/2.0)
                            fluctuations[si] = np.power(np.mean(Fq_segments), 1.0/q)
                
                # スケーリング関係 F_q(s) ~ s^h(q)
                valid_mask = fluctuations > 1e-10
                valid_count = np.sum(valid_mask)
                
                if valid_count >= 3:
                    log_s = np.log(scales[valid_mask].astype(np.float64))
                    log_Fq = np.log(fluctuations[valid_mask])
                    
                    mean_log_s = np.mean(log_s)
                    mean_log_Fq = np.mean(log_Fq)
                    
                    numerator = np.sum((log_s - mean_log_s) * (log_Fq - mean_log_Fq))
                    denominator = np.sum((log_s - mean_log_s)**2)
                    
                    if denominator > 1e-10:
                        hurst_values[qi] = numerator / denominator
                    else:
                        hurst_values[qi] = 0.5
                else:
                    hurst_values[qi] = 0.5
            
            # マルチフラクタルスペクトル解析
            multifractal_width = np.max(hurst_values) - np.min(hurst_values)
            h_mean = np.mean(hurst_values)
            asymmetry = (np.max(hurst_values) - h_mean) - (h_mean - np.min(hurst_values))
            complexity = np.std(hurst_values)
            
            # 結果格納
            results[idx, :n_q] = hurst_values
            results[idx, -3] = multifractal_width
            results[idx, -2] = asymmetry
            results[idx, -1] = complexity
        
        return results
    
    # =========================================================================
    # Tier S特徴量: Microstructure Noise Ratio
    # =========================================================================
    
    def calculate_microstructure_noise_features_polars(self, prices: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Microstructure Noise Ratio計算（Polars最適化版）
        真の価格変動 vs ノイズを分離する重要特徴量
        """
        features = {}
        n = len(prices)
        
        try:
            # Polarsでリターンを計算
            df_returns = pl.DataFrame({'price': prices}).with_columns([
                pl.col('price').diff().alias('returns')
            ])
            
            # Numba計算用にNumPy配列を準備 (diff()による先頭のnullを0で埋める)
            returns_data = df_returns['returns'].fill_null(0).to_numpy()

            # 複数ウィンドウサイズでMicrostructure Noise分析
            for window_size in self.params['microstructure_windows']:
                if n < window_size:
                    continue
                
                # Numbaで高速ノイズ比率計算（Polarsの複雑な処理は不要）
                noise_features = self._numba_safe_calculation(
                    self._calculate_microstructure_noise_vectorized,
                    returns_data,
                    window_size
                )
                
                if noise_features is not None and len(noise_features.shape) == 2:
                    # 結果をパディングして長さを合わせる
                    features[f'microstructure_noise_ratio_{window_size}'] = np.pad(noise_features[:, 0], (0, n - len(noise_features[:, 0])), 'constant')
                    features[f'signal_strength_{window_size}'] = np.pad(noise_features[:, 1], (0, n - len(noise_features[:, 1])), 'constant')
                    features[f'noise_persistence_{window_size}'] = np.pad(noise_features[:, 2], (0, n - len(noise_features[:, 2])), 'constant')
                else:
                    # フォールバック
                    features[f'microstructure_noise_ratio_{window_size}'] = np.zeros(n)
                    features[f'signal_strength_{window_size}'] = np.zeros(n)
                    features[f'noise_persistence_{window_size}'] = np.zeros(n)
                    
        except Exception as e:
            logger.error(f"Microstructure Noise計算エラー: {e}")
            # フォールバック値
            for window_size in self.params['microstructure_windows']:
                features[f'microstructure_noise_ratio_{window_size}'] = np.zeros(n)
                features[f'signal_strength_{window_size}'] = np.zeros(n)
                features[f'noise_persistence_{window_size}'] = np.zeros(n)
        
        return features
    
    @staticmethod
    @njit(parallel=True, cache=True)
    def _calculate_microstructure_noise_vectorized(returns: np.ndarray, 
                                                 window_size: int) -> np.ndarray:
        """
        Microstructure Noise核心計算（完全ベクトル化版）
        Roll (1984)の手法に基づくノイズ推定
        """
        n = len(returns)
        if n < window_size:
            return np.zeros((n, 3))
        
        results = np.zeros((n, 3))
        
        # 並列計算
        for i in prange(window_size-1, n):
            window_returns = returns[i-window_size+1:i+1]
            valid_returns = window_returns[np.isfinite(window_returns)]
            
            if len(valid_returns) < 10:
                continue
            
            # Roll (1984) ノイズ比率推定
            if len(valid_returns) > 1:
                returns_1 = valid_returns[1:]
                returns_2 = valid_returns[:-1]
                
                if len(returns_1) > 0 and len(returns_2) > 0:
                    covariance = np.mean((returns_1 - np.mean(returns_1)) * 
                                       (returns_2 - np.mean(returns_2)))
                else:
                    covariance = 0.0
            else:
                covariance = 0.0
            
            returns_variance = np.var(valid_returns)
            
            if returns_variance > 1e-10:
                noise_variance_estimate = -covariance if covariance < 0 else 0.0
                noise_ratio = min(0.5, max(0.0, noise_variance_estimate / returns_variance))
            else:
                noise_ratio = 0.0
            
            # シグナル強度（Hurst指数簡易推定）
            if len(valid_returns) >= 20:
                cumulative = np.cumsum(valid_returns - np.mean(valid_returns))
                R = np.max(cumulative) - np.min(cumulative)
                S = np.std(valid_returns)
                
                if S > 1e-10:
                    rs_ratio = R / S
                    hurst_estimate = np.log(rs_ratio) / np.log(len(valid_returns))
                    signal_strength = abs(hurst_estimate - 0.5) * 2
                else:
                    signal_strength = 0.0
            else:
                signal_strength = 0.0
            
            # ノイズ持続性（自己相関）
            if len(valid_returns) >= 10:
                returns_centered = valid_returns - np.mean(valid_returns)
                returns_var = np.sum(returns_centered**2)
                if returns_var > 1e-10:
                    autocorr = np.sum(returns_centered[1:] * returns_centered[:-1]) / returns_var
                    noise_persistence = abs(autocorr)
                else:
                    noise_persistence = 0.0
            else:
                noise_persistence = 0.0
            
            results[i, 0] = noise_ratio
            results[i, 1] = signal_strength
            results[i, 2] = noise_persistence
        
        return results
    
    # =========================================================================
    # Tier 1特徴量: 軽量版ショックモデル
    # =========================================================================
    
    def calculate_shock_model_features_polars(self, prices: np.ndarray) -> Dict[str, np.ndarray]:
        """
        軽量版ショックモデル特徴量（Polars最適化版）
        期待からの乖離とショック検出
        """
        features = {}
        
        # Polars DataFrameを構築
        df = self._ensure_polars_df(prices, 'price')
        n = len(df)
        
        try:
            # リターンと基本統計をPolarsで一括計算
            df_with_stats = df.with_columns([
                pl.col('price').diff().alias('returns')
            ])
            
            # 複数ウィンドウサイズでショックモデル分析
            for window_size in self.params['shock_model_windows']:
                if n < window_size:
                    continue
                
                # Polarsで期待リターンと標準偏差をローリング計算
                df_rolling = df_with_stats.with_columns([
                    pl.col('returns').rolling_mean(window_size=window_size).alias('expected_return'),
                    pl.col('returns').rolling_std(window_size=window_size).alias('return_std')
                ])
                
                # Numbaで高速ショック検出とモデル計算
                returns_data = df_with_stats['returns'].to_numpy()
                shock_features = self._numba_safe_calculation(
                    self._calculate_shock_model_vectorized,
                    returns_data,
                    window_size
                )
                
                if shock_features is not None and len(shock_features.shape) == 2:
                    features[f'shock_intensity_{window_size}'] = shock_features[:, 0]
                    features[f'shock_frequency_{window_size}'] = shock_features[:, 1]
                    features[f'recovery_speed_{window_size}'] = shock_features[:, 2]
                    features[f'expected_deviation_{window_size}'] = shock_features[:, 3]
                else:
                    # フォールバック
                    features[f'shock_intensity_{window_size}'] = np.zeros(n)
                    features[f'shock_frequency_{window_size}'] = np.zeros(n)
                    features[f'recovery_speed_{window_size}'] = np.zeros(n)
                    features[f'expected_deviation_{window_size}'] = np.zeros(n)
                    
        except Exception as e:
            logger.error(f"ショックモデル計算エラー: {e}")
            # フォールバック値
            for window_size in self.params['shock_model_windows']:
                features[f'shock_intensity_{window_size}'] = np.zeros(n)
                features[f'shock_frequency_{window_size}'] = np.zeros(n)
                features[f'recovery_speed_{window_size}'] = np.zeros(n)
                features[f'expected_deviation_{window_size}'] = np.zeros(n)
        
        return features
    
    @staticmethod
    @njit(parallel=True, cache=True)
    def _calculate_shock_model_vectorized(returns: np.ndarray, window_size: int) -> np.ndarray:
        """
        ショックモデル核心計算（完全ベクトル化版）
        """
        n = len(returns)
        if n < window_size:
            return np.zeros((n, 4))
        
        results = np.zeros((n, 4))
        
        # 並列計算
        for i in prange(window_size-1, n):
            window_returns = returns[i-window_size+1:i+1]
            valid_returns = window_returns[np.isfinite(window_returns)]
            
            if len(valid_returns) < 10:
                continue
            
            expected_return = np.mean(valid_returns)
            return_std = np.std(valid_returns)
            
            if return_std < 1e-10:
                continue
            
            # ショック検出（2σ閾値）
            shock_threshold = 2.0 * return_std
            shock_events = np.abs(valid_returns - expected_return) > shock_threshold
            
            # ショック強度
            if np.sum(shock_events) > 0:
                shock_returns = valid_returns[shock_events]
                shock_intensity = np.mean(np.abs(shock_returns - expected_return)) / return_std
            else:
                shock_intensity = 0.0
            
            # ショック頻度
            shock_frequency = np.sum(shock_events) / len(valid_returns)
            
            # 回復速度
            recovery_speeds = []
            for j in range(len(shock_events) - 1):
                if shock_events[j]:
                    current_deviation = valid_returns[j] - expected_return
                    next_deviation = valid_returns[j + 1] - expected_return
                    
                    if current_deviation != 0:
                        recovery = -next_deviation / current_deviation
                        recovery_speeds.append(recovery)
            
            recovery_speed = np.mean(recovery_speeds) if len(recovery_speeds) > 0 else 0.0
            
            # 期待からの乖離度（RMS）
            deviations = valid_returns - expected_return
            expected_deviation = np.sqrt(np.mean(deviations**2)) / (np.abs(expected_return) + 1e-8)
            
            results[i, 0] = shock_intensity
            results[i, 1] = shock_frequency
            results[i, 2] = recovery_speed
            results[i, 3] = expected_deviation
        
        return results
    
    # =========================================================================
    # Tier 1特徴量: Multi-Scale Volatility
    # =========================================================================
    
    def calculate_multiscale_volatility_polars(self, prices: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Multi-Scale Volatility計算（Polars最適化版）
        複数時間軸でのボラティリティ特性分析
        """
        features = {}
        
        # Polars DataFrameを構築
        df = self._ensure_polars_df(prices, 'price')
        n = len(df)
        
        try:
            # 対数リターンをPolarsで計算
            df_with_returns = df.with_columns([
                pl.col('price').log().diff().alias('log_returns')
            ])
            
            # 複数スケールでボラティリティ計算
            scales = self.params['multiscale_volatility_periods']
            
            for scale in scales:
                if scale >= n:
                    continue
                
                # Polarsでスケール別ボラティリティを効率計算
                scale_vol_result = self._polars_safe_calculation(
                    self._calculate_scale_volatility_polars,
                    df_with_returns,
                    scale
                )
                
                if isinstance(scale_vol_result, np.ndarray):
                    features[f'multiscale_volatility_{scale}'] = scale_vol_result
                else:
                    features[f'multiscale_volatility_{scale}'] = np.zeros(n)
            
            # クロススケール相関をPolarsで効率計算
            if len(scales) >= 2:
                cross_corr_features = self._polars_safe_calculation(
                    self._calculate_cross_scale_correlations_polars,
                    features,
                    scales,
                    n
                )
                
                if isinstance(cross_corr_features, dict):
                    features.update(cross_corr_features)
                    
        except Exception as e:
            logger.error(f"Multi-Scale Volatility計算エラー: {e}")
            # フォールバック値
            for scale in self.params['multiscale_volatility_periods']:
                features[f'multiscale_volatility_{scale}'] = np.zeros(n)
        
        return features
    
    def _calculate_scale_volatility_polars(self, df: pl.DataFrame, scale: int) -> np.ndarray:
        """スケール別ボラティリティ計算（Polarsベース）"""
        # Realized Volatility（年率化）
        result_df = df.with_columns([
            pl.col('log_returns').rolling_std(window_size=scale).alias('daily_vol')
        ]).with_columns([
            (pl.col('daily_vol') * np.sqrt(252 / scale)).alias('annualized_vol')
        ])
        
        return result_df['annualized_vol'].to_numpy()
    
    def _calculate_cross_scale_correlations_polars(self, 
                                                  volatility_features: Dict[str, np.ndarray],
                                                  scales: List[int],
                                                  n: int) -> Dict[str, np.ndarray]:
        """クロススケール相関計算（Polarsベース）"""
        cross_corr_features = {}
        
        # 有効なスケールのペアを抽出
        valid_scales = [s for s in scales if f'multiscale_volatility_{s}' in volatility_features]
        
        if len(valid_scales) < 2:
            return cross_corr_features
        
        # ボラティリティデータをPolars DataFrameに統合
        vol_data = {}
        for scale in valid_scales:
            vol_data[f'vol_{scale}'] = volatility_features[f'multiscale_volatility_{scale}']
        
        df_vol = pl.DataFrame(vol_data)
        
        # クロススケール相関を効率計算
        correlation_window = 50
        
        for i, s1 in enumerate(valid_scales[:-1]):
            for s2 in valid_scales[i+1:]:
                try:
                    # Polarsでローリング相関を計算
                    corr_result = df_vol.with_columns([
                        pl.corr(f'vol_{s1}', f'vol_{s2}', method='pearson')
                        .rolling_apply(function=lambda x: x, window_size=correlation_window)
                        .alias(f'cross_corr_{s1}_{s2}')
                    ])
                    
                    cross_corr_features[f'cross_scale_corr_{s1}_{s2}'] = (
                        corr_result[f'cross_corr_{s1}_{s2}'].to_numpy()
                    )
                    
                except Exception as e:
                    logger.debug(f"クロススケール相関計算エラー({s1}-{s2}): {e}")
                    cross_corr_features[f'cross_scale_corr_{s1}_{s2}'] = np.zeros(n)
        
        return cross_corr_features
    
    # =========================================================================
    # Tier 2特徴量: EMD/CEEMDAN (Empirical Mode Decomposition)
    # =========================================================================
    
    def calculate_emd_features_polars(self, data: np.ndarray) -> Dict[str, np.ndarray]:
        """
        EMD特徴量計算（Polars最適化・ウェーブレット代替版）
        """
        features = {}
        n = len(data)
        
        if not PYEMD_AVAILABLE:
            # ウェーブレット代替実装
            logger.info("PyEMD未利用のため、ウェーブレット代替実装を使用")
            return self._calculate_emd_substitute_polars(data)
        
        try:
            # Polars DataFrameを構築
            df = self._ensure_polars_df(data, 'signal')
            
            # EMD計算用のウィンドウサイズ
            window_sizes = self.params['emd_windows']
            max_imf = 8
            
            for window_size in window_sizes:
                if n < window_size:
                    continue
                
                # ローリングEMD計算（高速化）
                emd_features = self._numba_safe_calculation(
                    self._calculate_emd_rolling_vectorized,
                    data,
                    window_size,
                    max_imf
                )
                
                if emd_features is not None and len(emd_features.shape) == 3:
                    # 結果をPolarsで後処理
                    for imf_idx in range(max_imf):
                        for feature_idx, feature_name in enumerate(['energy', 'mean_freq', 'amplitude']):
                            feature_key = f'emd_imf_{imf_idx}_{feature_name}_{window_size}'
                            
                            if imf_idx < emd_features.shape[2] and feature_idx < emd_features.shape[1]:
                                padded_result = np.pad(
                                    emd_features[:, feature_idx, imf_idx],
                                    (window_size-1, 0),
                                    mode='constant',
                                    constant_values=0
                                )
                                features[feature_key] = padded_result
                            else:
                                features[feature_key] = np.zeros(n)
                else:
                    # フォールバック
                    for imf_idx in range(max_imf):
                        for feature_name in ['energy', 'mean_freq', 'amplitude']:
                            feature_key = f'emd_imf_{imf_idx}_{feature_name}_{window_size}'
                            features[feature_key] = np.zeros(n)
                            
        except Exception as e:
            logger.error(f"EMD計算エラー: {e}")
            # フォールバック値
            for window_size in self.params['emd_windows']:
                for imf_idx in range(8):
                    for feature_name in ['energy', 'mean_freq', 'amplitude']:
                        feature_key = f'emd_imf_{imf_idx}_{feature_name}_{window_size}'
                        features[feature_key] = np.zeros(n)
        
        return features
    
    @staticmethod
    @njit(parallel=True, cache=True)
    def _calculate_emd_rolling_vectorized(data: np.ndarray, 
                                        window_size: int,
                                        max_imf: int) -> np.ndarray:
        """
        EMD ローリング計算（ベクトル化・並列版）
        
        注意: 実際のEMDではなく簡易版IMF抽出
        """
        n = len(data)
        if n < window_size:
            return np.zeros((n-window_size+1, 3, max_imf))
        
        n_windows = n - window_size + 1
        results = np.zeros((n_windows, 3, max_imf))  # [windows, features, imfs]
        
        # 並列計算
        for idx in prange(n_windows):
            window = data[idx:idx+window_size]
            
            # 簡易IMF抽出（多重解像度解析風）
            current_signal = window.copy()
            
            for imf_idx in range(max_imf):
                if len(current_signal) < 8:
                    break
                
                # 簡易スプライン包絡線近似
                imf = np.zeros(len(current_signal))
                residual = current_signal.copy()
                
                # ピーク・バレー検出
                peaks = []
                valleys = []
                
                for i in range(1, len(current_signal)-1):
                    if (current_signal[i] > current_signal[i-1] and 
                        current_signal[i] > current_signal[i+1]):
                        peaks.append(i)
                    elif (current_signal[i] < current_signal[i-1] and 
                          current_signal[i] < current_signal[i+1]):
                        valleys.append(i)
                
                if len(peaks) > 2 and len(valleys) > 2:
                    # 上包絡線と下包絡線の平均を計算（線形補間）
                    upper_env = np.zeros(len(current_signal))
                    lower_env = np.zeros(len(current_signal))
                    
                    # 単純な線形補間
                    for i in range(len(current_signal)):
                        if len(peaks) >= 2:
                            # 最近接の2つのピークから補間
                            if i <= peaks[0]:
                                upper_env[i] = current_signal[peaks[0]]
                            elif i >= peaks[-1]:
                                upper_env[i] = current_signal[peaks[-1]]
                            else:
                                for p_idx in range(len(peaks)-1):
                                    if peaks[p_idx] <= i <= peaks[p_idx+1]:
                                        t = (i - peaks[p_idx]) / (peaks[p_idx+1] - peaks[p_idx])
                                        upper_env[i] = (current_signal[peaks[p_idx]] * (1-t) + 
                                                      current_signal[peaks[p_idx+1]] * t)
                                        break
                        
                        if len(valleys) >= 2:
                            # 最近接の2つのバレーから補間
                            if i <= valleys[0]:
                                lower_env[i] = current_signal[valleys[0]]
                            elif i >= valleys[-1]:
                                lower_env[i] = current_signal[valleys[-1]]
                            else:
                                for v_idx in range(len(valleys)-1):
                                    if valleys[v_idx] <= i <= valleys[v_idx+1]:
                                        t = (i - valleys[v_idx]) / (valleys[v_idx+1] - valleys[v_idx])
                                        lower_env[i] = (current_signal[valleys[v_idx]] * (1-t) + 
                                                      current_signal[valleys[v_idx+1]] * t)
                                        break
                    
                    mean_env = (upper_env + lower_env) / 2
                    imf = current_signal - mean_env
                else:
                    imf = current_signal
                    current_signal = np.zeros(len(current_signal))
                
                # IMF特徴量計算
                # エネルギー
                energy = np.sum(imf**2)
                
                # 平均周波数（ゼロ交差ベース）
                zero_crossings = 0
                for i in range(1, len(imf)):
                    if imf[i] * imf[i-1] < 0:
                        zero_crossings += 1
                mean_freq = zero_crossings / (2 * len(imf)) if len(imf) > 0 else 0
                
                # 振幅
                amplitude = np.std(imf)
                
                results[idx, 0, imf_idx] = energy
                results[idx, 1, imf_idx] = mean_freq
                results[idx, 2, imf_idx] = amplitude
                
                # 次のIMFのための残差更新
                current_signal = current_signal - imf
                
                # 収束判定（残差が十分小さい場合は終了）
                if np.std(current_signal) < 1e-6:
                    break
        
        return results
    
    def _calculate_emd_substitute_polars(self, data: np.ndarray) -> Dict[str, np.ndarray]:
        """
        EMDの代替実装（Polars+ウェーブレット）
        """
        features = {}
        n = len(data)
        
        try:
            # Polars DataFrameを構築
            df = self._ensure_polars_df(data, 'signal')
            
            # 複数ウェーブレットで多重解像度解析
            wavelets = ['db4', 'db8', 'haar']
            
            for wi, wavelet_name in enumerate(wavelets):
                try:
                    # ウェーブレット分解
                    coeffs = pywt.wavedec(data, wavelet_name, level=5)
                    
                    for i, coeff in enumerate(coeffs):
                        level_name = f'approx_{wi}' if i == 0 else f'detail_{wi}_{i}'
                        
                        # エネルギー
                        energy = np.sum(coeff**2)
                        features[f'emd_substitute_{level_name}_energy'] = np.full(n, energy)
                        
                        # 平均周波数推定
                        mean_freq = self._estimate_mean_frequency_safe(coeff)
                        features[f'emd_substitute_{level_name}_freq'] = np.full(n, mean_freq)
                        
                        # 振幅
                        amplitude = np.std(coeff)
                        features[f'emd_substitute_{level_name}_amplitude'] = np.full(n, amplitude)
                        
                except Exception as e:
                    logger.debug(f"ウェーブレット {wavelet_name} 処理エラー: {e}")
                    continue
                    
        except Exception as e:
            logger.error(f"EMD代替計算エラー: {e}")
        
        return features
    
    def _estimate_mean_frequency_safe(self, signal: np.ndarray) -> float:
        """信号の平均周波数推定（安全版）"""
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
                
        except Exception:
            return 0.0
    
    # =========================================================================
    # 統計的モーメント・ロバスト統計（Polars最適化）
    # =========================================================================
    
    def calculate_statistical_moments_polars(self, data: np.ndarray) -> Dict[str, np.ndarray]:
        """統計的モーメント計算（Polars最適化版）"""
        features = {}
        
        # Polars DataFrameを構築
        df = self._ensure_polars_df(data, 'value')
        n = len(df)
        
        try:
            # 複数ウィンドウサイズで統計的モーメント計算
            window_sizes = [20, 50, 100]
            
            for window_size in window_sizes:
                if n < window_size:
                    continue
                
                # Polarsで基本統計モーメントを効率計算
                moments_result = self._polars_safe_calculation(
                    self._calculate_moments_polars,
                    df,
                    window_size
                )
                
                if isinstance(moments_result, dict):
                    # ウィンドウサイズを特徴量名に追加
                    for key, value in moments_result.items():
                        features[f'{key}_{window_size}'] = value
                else:
                    # フォールバック
                    moment_names = ['mean', 'variance', 'skewness', 'kurtosis', 
                                  'moment_5', 'moment_6', 'moment_7', 'moment_8']
                    for name in moment_names:
                        features[f'statistical_{name}_{window_size}'] = np.zeros(n)
                        
        except Exception as e:
            logger.error(f"統計的モーメント計算エラー: {e}")
            # フォールバック値
            for window_size in [20, 50, 100]:
                moment_names = ['mean', 'variance', 'skewness', 'kurtosis', 
                              'moment_5', 'moment_6', 'moment_7', 'moment_8']
                for name in moment_names:
                    features[f'statistical_{name}_{window_size}'] = np.zeros(n)
        
        return features
    
    def _calculate_moments_polars(self, df: pl.DataFrame, window_size: int) -> Dict[str, np.ndarray]:
        """統計的モーメント計算（Polarsベース）"""
        
        # Polarsで基本統計量を計算
        result_df = df.with_columns([
            pl.col('value').rolling_mean(window_size=window_size).alias('rolling_mean'),
            pl.col('value').rolling_var(window_size=window_size).alias('rolling_variance'),
            pl.col('value').rolling_skew(window_size=window_size).alias('rolling_skewness'),
            pl.col('value').rolling_kurtosis(window_size=window_size).alias('rolling_kurtosis')
        ])
        
        # 高次モーメント（5-8次）をNumbaで効率計算
        values = df['value'].to_numpy()
        higher_moments = self._numba_safe_calculation(
            self._calculate_higher_moments_vectorized,
            values,
            window_size
        )
        
        results = {
            'statistical_mean': result_df['rolling_mean'].to_numpy(),
            'statistical_variance': result_df['rolling_variance'].to_numpy(),
            'statistical_skewness': result_df['rolling_skewness'].to_numpy(),
            'statistical_kurtosis': result_df['rolling_kurtosis'].to_numpy()
        }
        
        # 高次モーメントを追加
        if higher_moments is not None and len(higher_moments.shape) == 2:
            for i, moment_name in enumerate(['moment_5', 'moment_6', 'moment_7', 'moment_8']):
                results[f'statistical_{moment_name}'] = higher_moments[:, i]
        else:
            # フォールバック
            for moment_name in ['moment_5', 'moment_6', 'moment_7', 'moment_8']:
                results[f'statistical_{moment_name}'] = np.zeros(len(df))
        
        return results
    
    @staticmethod
    @njit(parallel=True, cache=True)
    def _calculate_higher_moments_vectorized(data: np.ndarray, window_size: int) -> np.ndarray:
        """高次モーメント計算（5-8次、ベクトル化・並列版）"""
        n = len(data)
        if n < window_size:
            return np.zeros((n, 4))
        
        results = np.zeros((n, 4))
        
        # 並列計算
        for i in prange(window_size-1, n):
            window = data[i-window_size+1:i+1]
            valid_window = window[np.isfinite(window)]
            
            if len(valid_window) < 10:
                continue
            
            mean_val = np.mean(valid_window)
            std_val = np.std(valid_window)
            
            if std_val > 1e-10:
                standardized = (valid_window - mean_val) / std_val
                
                # 5-8次モーメント
                results[i, 0] = np.mean(standardized**5)
                results[i, 1] = np.mean(standardized**6)
                results[i, 2] = np.mean(standardized**7)
                results[i, 3] = np.mean(standardized**8)
        
        return results
    
    # =========================================================================
    # ロバスト統計量（Polars最適化）
    # =========================================================================
    
    def calculate_robust_statistics_polars(self, data: np.ndarray) -> Dict[str, np.ndarray]:
        """ロバスト統計量計算（Polars最適化版）"""
        features = {}
        
        # Polars DataFrameを構築
        df = self._ensure_polars_df(data, 'value')
        n = len(df)
        
        try:
            # 複数ウィンドウサイズでロバスト統計
            window_sizes = [20, 50, 100]
            
            for window_size in window_sizes:
                if n < window_size:
                    continue
                
                # Polarsでメディアン等の基本ロバスト統計を計算
                robust_result = self._polars_safe_calculation(
                    self._calculate_robust_stats_polars,
                    df,
                    window_size
                )
                
                if isinstance(robust_result, dict):
                    # ウィンドウサイズを特徴量名に追加
                    for key, value in robust_result.items():
                        features[f'{key}_{window_size}'] = value
                else:
                    # フォールバック
                    robust_names = ['median', 'mad', 'iqr', 'trimmed_mean', 'winsorized_mean']
                    for name in robust_names:
                        features[f'robust_{name}_{window_size}'] = np.zeros(n)
                        
        except Exception as e:
            logger.error(f"ロバスト統計計算エラー: {e}")
            # フォールバック値
            for window_size in [20, 50, 100]:
                robust_names = ['median', 'mad', 'iqr', 'trimmed_mean', 'winsorized_mean']
                for name in robust_names:
                    features[f'robust_{name}_{window_size}'] = np.zeros(n)
        
        return features
    
    def _calculate_robust_stats_polars(self, df: pl.DataFrame, window_size: int) -> Dict[str, np.ndarray]:
        """ロバスト統計量計算（Polarsベース）"""
        
        # Polarsでメディアンと基本統計を計算
        result_df = df.with_columns([
            pl.col('value').rolling_median(window_size=window_size).alias('rolling_median'),
            pl.col('value').rolling_quantile(quantile=0.25, window_size=window_size).alias('q25'),
            pl.col('value').rolling_quantile(quantile=0.75, window_size=window_size).alias('q75')
        ]).with_columns([
            (pl.col('q75') - pl.col('q25')).alias('iqr')
        ])
        
        # MAD、Trimmed Mean、Winsorized Meanを高速計算
        values = df['value'].to_numpy()
        advanced_robust = self._numba_safe_calculation(
            self._calculate_advanced_robust_vectorized,
            values,
            window_size
        )
        
        results = {
            'robust_median': result_df['rolling_median'].to_numpy(),
            'robust_iqr': result_df['iqr'].to_numpy()
        }
        
        # 高度なロバスト統計を追加
        if advanced_robust is not None and len(advanced_robust.shape) == 2:
            results['robust_mad'] = advanced_robust[:, 0]
            results['robust_trimmed_mean'] = advanced_robust[:, 1]
            results['robust_winsorized_mean'] = advanced_robust[:, 2]
        else:
            # フォールバック
            results['robust_mad'] = np.zeros(len(df))
            results['robust_trimmed_mean'] = np.zeros(len(df))
            results['robust_winsorized_mean'] = np.zeros(len(df))
        
        return results
    
    @staticmethod
    @njit(parallel=True, cache=True)
    def _calculate_advanced_robust_vectorized(data: np.ndarray, window_size: int) -> np.ndarray:
        """高度ロバスト統計（MAD、Trimmed Mean、Winsorized Mean）"""
        n = len(data)
        if n < window_size:
            return np.zeros((n, 3))
        
        results = np.zeros((n, 3))
        
        # 並列計算
        for i in prange(window_size-1, n):
            window = data[i-window_size+1:i+1]
            valid_window = window[np.isfinite(window)]
            
            if len(valid_window) < 5:
                continue
            
            sorted_data = np.sort(valid_window)
            n_valid = len(sorted_data)
            
            # MAD (Median Absolute Deviation)
            median = np.median(sorted_data)
            mad = np.median(np.abs(sorted_data - median))
            
            # Trimmed Mean (10%トリミング)
            trim_count = max(1, int(n_valid * 0.1))
            if n_valid > 2 * trim_count:
                trimmed_data = sorted_data[trim_count:-trim_count]
                trimmed_mean = np.mean(trimmed_data)
            else:
                trimmed_mean = np.mean(sorted_data)
            
            # Winsorized Mean (5%ウィンソライゼーション)
            wins_count = max(1, int(n_valid * 0.05))
            winsorized_data = sorted_data.copy()
            if n_valid > 2 * wins_count:
                winsorized_data[:wins_count] = sorted_data[wins_count]
                winsorized_data[-wins_count:] = sorted_data[-wins_count-1]
            winsorized_mean = np.mean(winsorized_data)
            
            results[i, 0] = mad
            results[i, 1] = trimmed_mean
            results[i, 2] = winsorized_mean
        
        return results
    
    # =========================================================================
    # スペクトル特徴量（Polars+Numba最適化）
    # =========================================================================
    
    def calculate_spectral_features_polars(self, data: np.ndarray) -> Dict[str, np.ndarray]:
        """スペクトル特徴量計算（Polars最適化版）"""
        features = {}
        
        # Polars DataFrameを構築
        df = self._ensure_polars_df(data, 'signal')
        n = len(df)
        
        try:
            # 複数ウィンドウサイズでスペクトル解析
            window_sizes = [64, 128, 256]
            
            for window_size in window_sizes:
                if n < window_size:
                    continue
                
                # Numbaで高速スペクトル計算
                spectral_result = self._numba_safe_calculation(
                    self._calculate_spectral_vectorized,
                    data,
                    window_size
                )
                
                if spectral_result is not None and len(spectral_result.shape) == 2:
                    feature_names = ['spectral_centroid', 'spectral_bandwidth', 'spectral_rolloff', 
                                   'spectral_flux', 'spectral_flatness', 'spectral_entropy']
                    
                    for j, name in enumerate(feature_names):
                        if j < spectral_result.shape[1]:
                            padded_result = np.pad(
                                spectral_result[:, j],
                                (window_size-1, 0),
                                mode='constant',
                                constant_values=0
                            )
                            features[f'{name}_{window_size}'] = padded_result
                        else:
                            features[f'{name}_{window_size}'] = np.zeros(n)
                else:
                    # フォールバック
                    feature_names = ['spectral_centroid', 'spectral_bandwidth', 'spectral_rolloff', 
                                   'spectral_flux', 'spectral_flatness', 'spectral_entropy']
                    for name in feature_names:
                        features[f'{name}_{window_size}'] = np.zeros(n)
                        
        except Exception as e:
            logger.error(f"スペクトル特徴量計算エラー: {e}")
            # フォールバック値
            for window_size in [64, 128, 256]:
                feature_names = ['spectral_centroid', 'spectral_bandwidth', 'spectral_rolloff', 
                               'spectral_flux', 'spectral_flatness', 'spectral_entropy']
                for name in feature_names:
                    features[f'{name}_{window_size}'] = np.zeros(n)
        
        return features
    
    @staticmethod
    @njit(parallel=True, cache=True)
    def _calculate_spectral_vectorized(data: np.ndarray, window_size: int) -> np.ndarray:
        """スペクトル特徴量の核心計算（完全ベクトル化・並列版）"""
        n = len(data)
        if n < window_size:
            return np.zeros((n-window_size+1, 6))
        
        n_windows = n - window_size + 1
        results = np.zeros((n_windows, 6))
        
        # 並列計算
        for idx in prange(n_windows):
            window = data[idx:idx+window_size]
            
            if len(window) < 8:
                continue
            
            # FFT計算
            fft_data = np.fft.fft(window)
            power_spectrum = np.abs(fft_data[:window_size//2])**2
            freqs = np.arange(window_size//2) / window_size
            
            if np.sum(power_spectrum) < 1e-10:
                continue
            
            # 正規化
            power_spectrum_norm = power_spectrum / np.sum(power_spectrum)
            
            # 1. Spectral Centroid（重心周波数）
            spectral_centroid = np.sum(freqs * power_spectrum_norm)
            
            # 2. Spectral Bandwidth（帯域幅）
            spectral_bandwidth = np.sqrt(np.sum(((freqs - spectral_centroid)**2) * power_spectrum_norm))
            
            # 3. Spectral Rolloff（85%エネルギー点）
            cumsum = np.cumsum(power_spectrum_norm)
            rolloff_idx = np.where(cumsum >= 0.85)[0]
            spectral_rolloff = freqs[rolloff_idx[0]] if len(rolloff_idx) > 0 else 0.0
            
            # 4. Spectral Flux（前フレームとの差）
            spectral_flux = 0.0
            if idx > 0:
                # 前回の計算結果との比較（簡易版）
                spectral_flux = np.sum((power_spectrum_norm - results[idx-1, :len(power_spectrum_norm)])**2) if idx > 0 else 0.0
            
            # 5. Spectral Flatness（平坦度）
            spectral_flatness = 0.0
            if np.all(power_spectrum_norm > 1e-10):
                geo_mean = np.exp(np.mean(np.log(power_spectrum_norm + 1e-10)))
                arith_mean = np.mean(power_spectrum_norm)
                spectral_flatness = geo_mean / (arith_mean + 1e-10)
            
            # 6. Spectral Entropy（エントロピー）
            spectral_entropy = -np.sum(power_spectrum_norm * np.log2(power_spectrum_norm + 1e-10))
            
            results[idx, 0] = spectral_centroid
            results[idx, 1] = spectral_bandwidth
            results[idx, 2] = spectral_rolloff
            results[idx, 3] = spectral_flux
            results[idx, 4] = spectral_flatness
            results[idx, 5] = spectral_entropy
        
        return results
    
    # =========================================================================
    # ウェーブレット特徴量（Polars最適化）
    # =========================================================================
    
    def calculate_wavelet_features_polars(self, data: np.ndarray) -> Dict[str, np.ndarray]:
        """ウェーブレット特徴量計算（Polars最適化版）"""
        features = {}
        n = len(data)
        
        try:
            # ウェーブレット分解パラメータ
            wavelet_name = 'db4'
            level = 5
            window_sizes = self.params['wavelet_windows']
            
            for window_size in window_sizes:
                if n < window_size:
                    continue
                
                # ローリングウェーブレット分析
                wavelet_result = self._numba_safe_calculation(
                    self._calculate_wavelet_rolling_vectorized,
                    data,
                    window_size,
                    level
                )
                
                if wavelet_result is not None and len(wavelet_result.shape) == 3:
                    # 結果の展開
                    for j in range(level + 1):
                        level_name = 'approx' if j == 0 else f'detail_{j}'
                        
                        for k, feature_name in enumerate(['energy', 'entropy', 'mean', 'std']):
                            if k < wavelet_result.shape[2]:
                                padded_result = np.pad(
                                    wavelet_result[:, j, k],
                                    (window_size-1, 0),
                                    mode='constant',
                                    constant_values=0
                                )
                                features[f'wavelet_{level_name}_{feature_name}_{window_size}'] = padded_result
                            else:
                                features[f'wavelet_{level_name}_{feature_name}_{window_size}'] = np.zeros(n)
                else:
                    # フォールバック
                    for j in range(level + 1):
                        level_name = 'approx' if j == 0 else f'detail_{j}'
                        for feature_name in ['energy', 'entropy', 'mean', 'std']:
                            features[f'wavelet_{level_name}_{feature_name}_{window_size}'] = np.zeros(n)
                            
        except Exception as e:
            logger.error(f"ウェーブレット特徴量計算エラー: {e}")
            # フォールバック値
            for window_size in self.params['wavelet_windows']:
                for j in range(6):  # level + 1
                    level_name = 'approx' if j == 0 else f'detail_{j}'
                    for feature_name in ['energy', 'entropy', 'mean', 'std']:
                        features[f'wavelet_{level_name}_{feature_name}_{window_size}'] = np.zeros(n)
        
        return features
    
    @staticmethod
    @njit(parallel=True, cache=True)
    def _calculate_wavelet_rolling_vectorized(data: np.ndarray, 
                                            window_size: int,
                                            level: int) -> np.ndarray:
        """
        ローリングウェーブレット分析（ベクトル化・並列版）
        
        注意: 実際のウェーブレット変換ではなく、多重解像度解析の簡易版
        """
        n = len(data)
        if n < window_size:
            return np.zeros((n-window_size+1, level+1, 4))
        
        n_windows = n - window_size + 1
        results = np.zeros((n_windows, level+1, 4))  # [windows, levels, features]
        
        # 並列計算
        for idx in prange(n_windows):
            window = data[idx:idx+window_size]
            
            # 簡易多重解像度解析
            current_signal = window.copy()
            
            for lev in range(level + 1):
                if len(current_signal) < 4:
                    break
                
                # 簡易ローパスフィルタ（移動平均）
                if lev == 0:
                    # 近似係数（低周波成分）
                    approx = current_signal.copy()
                else:
                    # 詳細係数（高周波成分）
                    if len(current_signal) >= 2:
                        approx = np.zeros(len(current_signal)//2)
                        detail = np.zeros(len(current_signal)//2)
                        
                        for i in range(len(approx)):
                            if 2*i+1 < len(current_signal):
                                approx[i] = (current_signal[2*i] + current_signal[2*i+1]) / 2
                                detail[i] = (current_signal[2*i] - current_signal[2*i+1]) / 2
                        
                        if lev == 1:
                            coeff = detail
                        else:
                            coeff = approx
                            current_signal = approx
                    else:
                        coeff = current_signal
                
                if lev == 0:
                    coeff = approx
                
                # 特徴量計算
                if len(coeff) > 0:
                    # エネルギー
                    energy = np.sum(coeff**2)
                    
                    # エントロピー（簡易版）
                    if energy > 1e-10:
                        prob = coeff**2 / energy
                        prob = prob[prob > 1e-10]
                        entropy = -np.sum(prob * np.log2(prob + 1e-10)) if len(prob) > 0 else 0
                    else:
                        entropy = 0
                    
                    # 平均と標準偏差
                    mean_val = np.mean(coeff)
                    std_val = np.std(coeff)
                    
                    results[idx, lev, 0] = energy
                    results[idx, lev, 1] = entropy
                    results[idx, lev, 2] = mean_val
                    results[idx, lev, 3] = std_val
                
                # 次のレベルのための信号更新
                if lev < level and len(current_signal) >= 2:
                    # ダウンサンプリング
                    current_signal = current_signal[::2]
        
        return results
    
    # =========================================================================
    # カオス理論・フラクタル解析（Polars最適化）
    # =========================================================================
    
    def calculate_chaos_features_polars(self, data: np.ndarray) -> Dict[str, np.ndarray]:
        """カオス理論特徴量計算（Polars最適化版）"""
        features = {}
        
        # Polars DataFrameを構築
        df = self._ensure_polars_df(data, 'value')
        n = len(df)
        
        try:
            # 複数ウィンドウサイズでカオス解析
            window_sizes = [50, 100]
            
            for window_size in window_sizes:
                if n < window_size:
                    continue
                
                # Numbaで高速カオス特徴量計算
                chaos_result = self._numba_safe_calculation(
                    self._calculate_chaos_vectorized,
                    data,
                    window_size
                )
                
                if chaos_result is not None and len(chaos_result.shape) == 2:
                    feature_names = ['lyapunov_exponent', 'correlation_dimension', 'chaos_degree']
                    
                    for j, name in enumerate(feature_names):
                        if j < chaos_result.shape[1]:
                            padded_result = np.pad(
                                chaos_result[:, j],
                                (window_size-1, 0),
                                mode='constant',
                                constant_values=0
                            )
                            features[f'{name}_{window_size}'] = padded_result
                        else:
                            features[f'{name}_{window_size}'] = np.zeros(n)
                else:
                    # フォールバック
                    feature_names = ['lyapunov_exponent', 'correlation_dimension', 'chaos_degree']
                    for name in feature_names:
                        features[f'{name}_{window_size}'] = np.zeros(n)
                        
        except Exception as e:
            logger.error(f"カオス理論特徴量計算エラー: {e}")
            # フォールバック値
            for window_size in [50, 100]:
                feature_names = ['lyapunov_exponent', 'correlation_dimension', 'chaos_degree']
                for name in feature_names:
                    features[f'{name}_{window_size}'] = np.zeros(n)
        
        return features
    
    @staticmethod
    @njit(parallel=True, cache=True)
    def _calculate_chaos_vectorized(data: np.ndarray, window_size: int) -> np.ndarray:
        """カオス特徴量の核心計算（完全ベクトル化・並列版）"""
        n = len(data)
        if n < window_size:
            return np.zeros((n-window_size+1, 3))
        
        n_windows = n - window_size + 1
        results = np.zeros((n_windows, 3))
        
        # 並列計算
        for idx in prange(n_windows):
            window = data[idx:idx+window_size]
            
            if len(window) < 10:
                continue
            
            # 1. リアプノフ指数の近似計算
            returns = np.diff(window)
            if len(returns) < 5:
                continue
            
            total_log_div = 0.0
            count = 0
            
            for i in range(len(returns) - 5):
                for j in range(i + 1, min(i + 6, len(returns) - 5)):
                    initial_dist = abs(returns[i] - returns[j])
                    final_dist = abs(returns[i + 5] - returns[j + 5])
                    if initial_dist > 1e-8:
                        total_log_div += np.log(final_dist / initial_dist)
                        count += 1
            
            lyapunov_exponent = (total_log_div / count) / 5 if count > 0 else 0.0
            
            # 2. 相関次元の簡易計算
            embedding_dim = min(3, len(window) // 3)
            correlation_dimension = 0.0
            
            if embedding_dim >= 2:
                # 位相空間再構成
                n_points = len(window) - embedding_dim
                if n_points > 10:
                    # 距離計算（サンプリング）
                    distances = []
                    sample_size = min(100, n_points * (n_points - 1) // 2)
                    
                    for sample in range(sample_size):
                        i = sample % n_points
                        j = (sample + 1) % n_points
                        if i != j:
                            dist = 0.0
                            for k in range(embedding_dim):
                                dist += (window[i + k] - window[j + k])**2
                            distances.append(np.sqrt(dist))
                    
                    if len(distances) > 0:
                        distances_array = np.array(distances)
                        std_dist = np.std(distances_array)
                        if std_dist > 1e-10:
                            radius = std_dist * 0.1
                            correlation = np.sum(distances_array < radius) / len(distances_array)
                            if correlation > 1e-10:
                                correlation_dimension = np.log(correlation) / np.log(radius)
            
            # 3. カオス度（標準偏差 vs 平均の比）
            mean_abs = abs(np.mean(window))
            std_dev = np.std(window)
            chaos_degree = std_dev / (mean_abs + 1e-8) if mean_abs > 1e-8 else (1.0 if std_dev > 0 else 0.0)
            
            results[idx, 0] = lyapunov_exponent
            results[idx, 1] = correlation_dimension
            results[idx, 2] = chaos_degree
        
        return results
    
    # =========================================================================
    # ヒルベルト変換特徴量（Polars+Scipy最適化）
    # =========================================================================
    
    def calculate_hilbert_transform_features_polars(self, data: np.ndarray) -> Dict[str, np.ndarray]:
        """ヒルベルト変換特徴量計算（Polars最適化版）"""
        features = {}
        n = len(data)
        
        try:
            # ヒルベルト変換
            analytic_signal = hilbert(data)
            amplitude_envelope = np.abs(analytic_signal)
            instantaneous_phase = np.angle(analytic_signal)
            instantaneous_frequency = np.diff(np.unwrap(instantaneous_phase), prepend=0)
            
            # Polars DataFrameに統合
            df = pl.DataFrame({
                'amplitude': amplitude_envelope,
                'phase': instantaneous_phase,
                'frequency': instantaneous_frequency
            })
            
            # 基本ヒルベルト特徴量
            features['hilbert_amplitude'] = amplitude_envelope
            features['hilbert_phase'] = instantaneous_phase
            features['hilbert_frequency'] = instantaneous_frequency
            
            # 振幅と位相の統計（Polarsで効率計算）
            for window in self.params['hilbert_windows']:
                if n > window:
                    hilbert_stats = self._polars_safe_calculation(
                        self._calculate_hilbert_stats_polars,
                        df,
                        window
                    )
                    
                    if isinstance(hilbert_stats, dict):
                        for key, value in hilbert_stats.items():
                            features[f'{key}_{window}'] = value
                    else:
                        # フォールバック
                        stat_names = ['hilbert_amp_mean', 'hilbert_amp_std', 'hilbert_phase_var', 'hilbert_phase_stability']
                        for name in stat_names:
                            features[f'{name}_{window}'] = np.zeros(n)
                            
        except Exception as e:
            logger.error(f"ヒルベルト変換計算エラー: {e}")
            # フォールバック値
            features['hilbert_amplitude'] = np.zeros(n)
            features['hilbert_phase'] = np.zeros(n)
            features['hilbert_frequency'] = np.zeros(n)
            
            for window in self.params['hilbert_windows']:
                stat_names = ['hilbert_amp_mean', 'hilbert_amp_std', 'hilbert_phase_var', 'hilbert_phase_stability']
                for name in stat_names:
                    features[f'{name}_{window}'] = np.zeros(n)
        
        return features
    
    def _calculate_hilbert_stats_polars(self, df: pl.DataFrame, window: int) -> Dict[str, np.ndarray]:
        """ヒルベルト変換統計量（Polarsベース）"""
        
        # 振幅と位相の統計をPolarsで計算
        result_df = df.with_columns([
            pl.col('amplitude').rolling_mean(window_size=window).alias('amp_mean'),
            pl.col('amplitude').rolling_std(window_size=window).alias('amp_std'),
            pl.col('phase').rolling_var(window_size=window).alias('phase_var')
        ]).with_columns([
            pl.col('phase').diff().alias('phase_diff')
        ]).with_columns([
            pl.col('phase_diff').rolling_std(window_size=window).alias('phase_stability')
        ])
        
        return {
            'hilbert_amp_mean': result_df['amp_mean'].to_numpy(),
            'hilbert_amp_std': result_df['amp_std'].to_numpy(),
            'hilbert_phase_var': result_df['phase_var'].to_numpy(),
            'hilbert_phase_stability': result_df['phase_stability'].to_numpy()
        }
    
    # =========================================================================
    # ADX・基本オシレーター（Polars最適化）
    # =========================================================================
    
    def calculate_adx_features_polars(self, high: np.ndarray, low: np.ndarray, close: np.ndarray) -> Dict[str, np.ndarray]:
        """ADX関連特徴量の計算（Polars最適化版）"""
        features = {}
        
        # Polars DataFrameを構築
        df = pl.DataFrame({
            'high': high,
            'low': low,
            'close': close
        })
        
        try:
            for period in self.params['adx_periods']:
                # ADXをPolarsで効率計算
                adx_result = self._polars_safe_calculation(
                    self._calculate_adx_polars,
                    df,
                    period
                )
                
                if isinstance(adx_result, dict):
                    features.update(adx_result)
                else:
                    # フォールバック
                    n = len(df)
                    features[f'adx_{period}'] = np.zeros(n)
                    features[f'di_plus_{period}'] = np.zeros(n)
                    features[f'di_minus_{period}'] = np.zeros(n)
                    features[f'di_diff_{period}'] = np.zeros(n)
                    features[f'adx_strength_{period}'] = np.zeros(n)
                    features[f'trend_strength_{period}'] = np.zeros(n)
                    
        except Exception as e:
            logger.error(f"ADX計算エラー: {e}")
            n = len(df)
            for period in self.params['adx_periods']:
                features[f'adx_{period}'] = np.zeros(n)
                features[f'di_plus_{period}'] = np.zeros(n)
                features[f'di_minus_{period}'] = np.zeros(n)
                features[f'di_diff_{period}'] = np.zeros(n)
                features[f'adx_strength_{period}'] = np.zeros(n)
                features[f'trend_strength_{period}'] = np.zeros(n)
        
        return features
    
    def _calculate_adx_polars(self, df: pl.DataFrame, period: int) -> Dict[str, np.ndarray]:
        """ADX計算（Polarsベース）"""
        
        # True Rangeと方向性移動をPolarsで計算
        result_df = df.with_columns([
            pl.col('close').shift(1).alias('prev_close')
        ]).with_columns([
            pl.max_horizontal([
                pl.col('high') - pl.col('low'),
                (pl.col('high') - pl.col('prev_close')).abs(),
                (pl.col('low') - pl.col('prev_close')).abs()
            ]).alias('true_range'),
            (pl.col('high') - pl.col('high').shift(1)).alias('up_move'),
            (pl.col('low').shift(1) - pl.col('low')).alias('down_move')
        ]).with_columns([
            pl.when((pl.col('up_move') > pl.col('down_move')) & (pl.col('up_move') > 0))
            .then(pl.col('up_move'))
            .otherwise(0.0)
            .alias('dm_plus'),
            pl.when((pl.col('down_move') > pl.col('up_move')) & (pl.col('down_move') > 0))
            .then(pl.col('down_move'))
            .otherwise(0.0)
            .alias('dm_minus')
        ]).with_columns([
            # ATRとDMのスムージング
            pl.col('true_range').ewm_mean(span=period, adjust=False).alias('atr_smooth'),
            pl.col('dm_plus').ewm_mean(span=period, adjust=False).alias('dm_plus_smooth'),
            pl.col('dm_minus').ewm_mean(span=period, adjust=False).alias('dm_minus_smooth')
        ]).with_columns([
            # DI+とDI-の計算
            (100.0 * pl.col('dm_plus_smooth') / (pl.col('atr_smooth') + 1e-10)).alias('di_plus'),
            (100.0 * pl.col('dm_minus_smooth') / (pl.col('atr_smooth') + 1e-10)).alias('di_minus')
        ]).with_columns([
            # DXの計算
            (100.0 * (pl.col('di_plus') - pl.col('di_minus')).abs() / 
             (pl.col('di_plus') + pl.col('di_minus') + 1e-10)).alias('dx')
        ]).with_columns([
            # ADXの計算
            pl.col('dx').ewm_mean(span=period, adjust=False).alias('adx')
        ]).with_columns([
            # 追加特徴量
            (pl.col('di_plus') - pl.col('di_minus')).alias('di_diff'),
            (pl.col('adx') > 25).cast(pl.Float64).alias('adx_strength'),
            (pl.col('adx') / 100.0).alias('trend_strength')
        ])
        
        return {
            f'adx_{period}': result_df['adx'].to_numpy(),
            f'di_plus_{period}': result_df['di_plus'].to_numpy(),
            f'di_minus_{period}': result_df['di_minus'].to_numpy(),
            f'di_diff_{period}': result_df['di_diff'].to_numpy(),
            f'adx_strength_{period}': result_df['adx_strength'].to_numpy(),
            f'trend_strength_{period}': result_df['trend_strength'].to_numpy()
        }
    
    def calculate_parabolic_sar_features_polars(self, high: np.ndarray, low: np.ndarray, close: np.ndarray) -> Dict[str, np.ndarray]:
        """パラボリックSAR特徴量の計算（Polars最適化版）"""
        features = {}
        
        try:
            # NumbaでSAR計算（状態機械的性質のため）
            sar_result = self._numba_safe_calculation(
                self._calculate_parabolic_sar_vectorized,
                high, low, close
            )
            
            if sar_result is not None and len(sar_result.shape) == 2:
                sar_values = sar_result[:, 0]
                signal_values = sar_result[:, 1]
                
                # Polarsで追加特徴量を効率計算
                df = pl.DataFrame({
                    'close': close,
                    'sar': sar_values,
                    'signal': signal_values
                })
                
                sar_features = self._polars_safe_calculation(
                    self._calculate_sar_features_polars,
                    df
                )
                
                if isinstance(sar_features, dict):
                    features.update(sar_features)
                else:
                    # フォールバック
                    n = len(close)
                    features['parabolic_sar'] = np.zeros(n)
                    features['sar_signal'] = np.zeros(n)
                    features['sar_distance'] = np.zeros(n)
                    features['sar_above'] = np.zeros(n)
                    features['sar_trend_strength'] = np.zeros(n)
            else:
                # フォールバック
                n = len(close)
                features['parabolic_sar'] = np.zeros(n)
                features['sar_signal'] = np.zeros(n)
                features['sar_distance'] = np.zeros(n)
                features['sar_above'] = np.zeros(n)
                features['sar_trend_strength'] = np.zeros(n)
                
        except Exception as e:
            logger.error(f"Parabolic SAR計算エラー: {e}")
            n = len(close)
            features['parabolic_sar'] = np.zeros(n)
            features['sar_signal'] = np.zeros(n)
            features['sar_distance'] = np.zeros(n)
            features['sar_above'] = np.zeros(n)
            features['sar_trend_strength'] = np.zeros(n)
        
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
    
    def _calculate_sar_features_polars(self, df: pl.DataFrame) -> Dict[str, np.ndarray]:
        """SAR追加特徴量（Polarsベース）"""
        
        result_df = df.with_columns([
            ((pl.col('close') - pl.col('sar')) / (pl.col('close') + 1e-10)).alias('sar_distance'),
            (pl.col('close') > pl.col('sar')).cast(pl.Float64).alias('sar_above'),
            pl.col('signal').abs().alias('sar_trend_strength')
        ])
        
        return {
            'parabolic_sar': result_df['sar'].to_numpy(),
            'sar_signal': result_df['signal'].to_numpy(),
            'sar_distance': result_df['sar_distance'].to_numpy(),
            'sar_above': result_df['sar_above'].to_numpy(),
            'sar_trend_strength': result_df['sar_trend_strength'].to_numpy()
        }
    
    def calculate_cci_features_polars(self, high: np.ndarray, low: np.ndarray, close: np.ndarray) -> Dict[str, np.ndarray]:
        """CCI特徴量の計算（Polars最適化版）"""
        features = {}
        
        # Polars DataFrameを構築
        df = pl.DataFrame({
            'high': high,
            'low': low,
            'close': close
        })
        
        try:
            for period in self.params['cci_periods']:
                # CCIをPolarsで効率計算
                cci_result = self._polars_safe_calculation(
                    self._calculate_cci_polars,
                    df,
                    period
                )
                
                if isinstance(cci_result, dict):
                    features.update(cci_result)
                else:
                    # フォールバック
                    n = len(df)
                    features[f'cci_{period}'] = np.zeros(n)
                    features[f'cci_overbought_{period}'] = np.zeros(n)
                    features[f'cci_oversold_{period}'] = np.zeros(n)
                    features[f'cci_normalized_{period}'] = np.zeros(n)
                    
        except Exception as e:
            logger.error(f"CCI計算エラー: {e}")
            n = len(df)
            for period in self.params['cci_periods']:
                features[f'cci_{period}'] = np.zeros(n)
                features[f'cci_overbought_{period}'] = np.zeros(n)
                features[f'cci_oversold_{period}'] = np.zeros(n)
                features[f'cci_normalized_{period}'] = np.zeros(n)
        
        return features
    
    def _calculate_cci_polars(self, df: pl.DataFrame, period: int) -> Dict[str, np.ndarray]:
        """CCI計算（Polarsベース）"""
        
        # Typical PriceとCCIをPolarsで計算
        result_df = df.with_columns([
            ((pl.col('high') + pl.col('low') + pl.col('close')) / 3.0).alias('typical_price')
        ]).with_columns([
            pl.col('typical_price').rolling_mean(window_size=period).alias('sma_tp')
        ]).with_columns([
            (pl.col('typical_price') - pl.col('sma_tp')).abs().rolling_mean(window_size=period).alias('mad')
        ]).with_columns([
            ((pl.col('typical_price') - pl.col('sma_tp')) / (0.015 * pl.col('mad') + 1e-10)).alias('cci')
        ]).with_columns([
            (pl.col('cci') > 100).cast(pl.Float64).alias('cci_overbought'),
            (pl.col('cci') < -100).cast(pl.Float64).alias('cci_oversold'),
            (pl.col('cci') / 100.0).tanh().alias('cci_normalized')
        ])
        
        return {
            f'cci_{period}': result_df['cci'].to_numpy(),
            f'cci_overbought_{period}': result_df['cci_overbought'].to_numpy(),
            f'cci_oversold_{period}': result_df['cci_oversold'].to_numpy(),
            f'cci_normalized_{period}': result_df['cci_normalized'].to_numpy()
        }
    
    def calculate_williams_r_features_polars(self, high: np.ndarray, low: np.ndarray, close: np.ndarray) -> Dict[str, np.ndarray]:
        """ウィリアムズ%R特徴量の計算（Polars最適化版）"""
        features = {}
        
        # Polars DataFrameを構築
        df = pl.DataFrame({
            'high': high,
            'low': low,
            'close': close
        })
        
        try:
            for period in self.params['williams_r_periods']:
                # Williams %RをPolarsで効率計算
                williams_r_result = self._polars_safe_calculation(
                    self._calculate_williams_r_polars,
                    df,
                    period
                )
                
                if isinstance(williams_r_result, dict):
                    features.update(williams_r_result)
                else:
                    # フォールバック
                    n = len(df)
                    features[f'williams_r_{period}'] = np.zeros(n)
                    features[f'williams_r_overbought_{period}'] = np.zeros(n)
                    features[f'williams_r_oversold_{period}'] = np.zeros(n)
                    features[f'williams_r_normalized_{period}'] = np.zeros(n)
                    
        except Exception as e:
            logger.error(f"Williams %R計算エラー: {e}")
            n = len(df)
            for period in self.params['williams_r_periods']:
                features[f'williams_r_{period}'] = np.zeros(n)
                features[f'williams_r_overbought_{period}'] = np.zeros(n)
                features[f'williams_r_oversold_{period}'] = np.zeros(n)
                features[f'williams_r_normalized_{period}'] = np.zeros(n)
        
        return features
    
    def _calculate_williams_r_polars(self, df: pl.DataFrame, period: int) -> Dict[str, np.ndarray]:
        """Williams %R計算（Polarsベース）"""
        
        # Williams %RをPolarsで計算
        result_df = df.with_columns([
            pl.col('high').rolling_max(window_size=period).alias('highest_high'),
            pl.col('low').rolling_min(window_size=period).alias('lowest_low')
        ]).with_columns([
            (-100.0 * (pl.col('highest_high') - pl.col('close')) / 
             (pl.col('highest_high') - pl.col('lowest_low') + 1e-10)).alias('williams_r')
        ]).with_columns([
            (pl.col('williams_r') > -20).cast(pl.Float64).alias('williams_r_overbought'),
            (pl.col('williams_r') < -80).cast(pl.Float64).alias('williams_r_oversold'),
            ((pl.col('williams_r') + 50) / 50.0).alias('williams_r_normalized')
        ])
        
        return {
            f'williams_r_{period}': result_df['williams_r'].to_numpy(),
            f'williams_r_overbought_{period}': result_df['williams_r_overbought'].to_numpy(),
            f'williams_r_oversold_{period}': result_df['williams_r_oversold'].to_numpy(),
            f'williams_r_normalized_{period}': result_df['williams_r_normalized'].to_numpy()
        }
    
    def calculate_aroon_features_polars(self, high: np.ndarray, low: np.ndarray) -> Dict[str, np.ndarray]:
        """アルーン特徴量の計算（Polars最適化版）"""
        features = {}
        
        # Polars DataFrameを構築
        df = pl.DataFrame({
            'high': high,
            'low': low
        })
        
        try:
            for period in self.params['aroon_periods']:
                # AroonをPolarsで効率計算
                aroon_result = self._polars_safe_calculation(
                    self._calculate_aroon_polars,
                    df,
                    period
                )
                
                if isinstance(aroon_result, dict):
                    features.update(aroon_result)
                else:
                    # フォールバック
                    n = len(df)
                    features[f'aroon_up_{period}'] = np.zeros(n)
                    features[f'aroon_down_{period}'] = np.zeros(n)
                    features[f'aroon_oscillator_{period}'] = np.zeros(n)
                    features[f'aroon_trending_{period}'] = np.zeros(n)
                    
        except Exception as e:
            logger.error(f"Aroon計算エラー: {e}")
            n = len(df)
            for period in self.params['aroon_periods']:
                features[f'aroon_up_{period}'] = np.zeros(n)
                features[f'aroon_down_{period}'] = np.zeros(n)
                features[f'aroon_oscillator_{period}'] = np.zeros(n)
                features[f'aroon_trending_{period}'] = np.zeros(n)
        
        return features
    
    def _calculate_aroon_polars(self, df: pl.DataFrame, period: int) -> Dict[str, np.ndarray]:
        """Aroon計算（Polarsベース）"""
        
        # Aroon計算にはNumbaが効率的
        high_data = df['high'].to_numpy()
        low_data = df['low'].to_numpy()
        
        aroon_result = self._numba_safe_calculation(
            self._calculate_aroon_vectorized,
            high_data, low_data, period
        )
        
        if aroon_result is not None and len(aroon_result.shape) == 2:
            aroon_up = aroon_result[:, 0]
            aroon_down = aroon_result[:, 1]
            
            # Polarsで追加特徴量を計算
            result_df = pl.DataFrame({
                'aroon_up': aroon_up,
                'aroon_down': aroon_down
            }).with_columns([
                (pl.col('aroon_up') - pl.col('aroon_down')).alias('aroon_oscillator'),
                ((pl.col('aroon_up') - pl.col('aroon_down')).abs() > 50).cast(pl.Float64).alias('aroon_trending')
            ])
            
            return {
                f'aroon_up_{period}': result_df['aroon_up'].to_numpy(),
                f'aroon_down_{period}': result_df['aroon_down'].to_numpy(),
                f'aroon_oscillator_{period}': result_df['aroon_oscillator'].to_numpy(),
                f'aroon_trending_{period}': result_df['aroon_trending'].to_numpy()
            }
        else:
            # フォールバック
            n = len(df)
            return {
                f'aroon_up_{period}': np.zeros(n),
                f'aroon_down_{period}': np.zeros(n),
                f'aroon_oscillator_{period}': np.zeros(n),
                f'aroon_trending_{period}': np.zeros(n)
            }
    
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
    
    def calculate_ultimate_oscillator_features_polars(self, high: np.ndarray, low: np.ndarray, close: np.ndarray) -> Dict[str, np.ndarray]:
        """アルティメットオシレーター特徴量の計算（Polars最適化版）"""
        features = {}
        
        try:
            # Ultimate OscillatorはNumbaが効率的
            uo_result = self._numba_safe_calculation(
                self._calculate_ultimate_oscillator_vectorized,
                high, low, close
            )
            
            if uo_result is not None:
                # Polarsで追加特徴量を計算
                df = pl.DataFrame({
                    'ultimate_oscillator': uo_result
                }).with_columns([
                    (pl.col('ultimate_oscillator') > 70).cast(pl.Float64).alias('uo_overbought'),
                    (pl.col('ultimate_oscillator') < 30).cast(pl.Float64).alias('uo_oversold'),
                    ((pl.col('ultimate_oscillator') - 50) / 50.0).alias('uo_normalized')
                ])
                
                features['ultimate_oscillator'] = df['ultimate_oscillator'].to_numpy()
                features['uo_overbought'] = df['uo_overbought'].to_numpy()
                features['uo_oversold'] = df['uo_oversold'].to_numpy()
                features['uo_normalized'] = df['uo_normalized'].to_numpy()
            else:
                # フォールバック
                n = len(high)
                features['ultimate_oscillator'] = np.zeros(n)
                features['uo_overbought'] = np.zeros(n)
                features['uo_oversold'] = np.zeros(n)
                features['uo_normalized'] = np.zeros(n)
                
        except Exception as e:
            logger.error(f"Ultimate Oscillator計算エラー: {e}")
            n = len(high)
            features['ultimate_oscillator'] = np.zeros(n)
            features['uo_overbought'] = np.zeros(n)
            features['uo_oversold'] = np.zeros(n)
            features['uo_normalized'] = np.zeros(n)
        
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
    # 出来高関連指標（Polars最適化）
    # =========================================================================
    
    def calculate_volume_features_polars(self, high: np.ndarray, low: np.ndarray, 
                                       close: np.ndarray, volume: np.ndarray) -> Dict[str, np.ndarray]:
        """出来高関連特徴量の統合計算（Polars最適化版）"""
        features = {}
        
        # Polars DataFrameを構築
        df = pl.DataFrame({
            'high': high,
            'low': low,
            'close': close,
            'volume': volume
        })
        
        try:
            # VPT (Volume Price Trend)
            vpt_features = self._polars_safe_calculation(
                self._calculate_vpt_polars, df
            )
            if isinstance(vpt_features, dict):
                features.update(vpt_features)
            
            # A/D Line (Accumulation/Distribution)
            ad_features = self._polars_safe_calculation(
                self._calculate_ad_line_polars, df
            )
            if isinstance(ad_features, dict):
                features.update(ad_features)
            
            # CMF (Chaikin Money Flow)
            for period in self.params['cmf_periods']:
                cmf_features = self._polars_safe_calculation(
                    self._calculate_cmf_polars, df, period
                )
                if isinstance(cmf_features, dict):
                    features.update(cmf_features)
            
            # Chaikin Oscillator
            chaikin_features = self._polars_safe_calculation(
                self._calculate_chaikin_oscillator_polars, df
            )
            if isinstance(chaikin_features, dict):
                features.update(chaikin_features)
            
            # MFI (Money Flow Index)
            for period in self.params['mfi_periods']:
                mfi_features = self._polars_safe_calculation(
                    self._calculate_mfi_polars, df, period
                )
                if isinstance(mfi_features, dict):
                    features.update(mfi_features)
            
            # VWAP (Volume Weighted Average Price)
            for period in [20, 50, 100]:
                vwap_features = self._polars_safe_calculation(
                    self._calculate_vwap_polars, df, period
                )
                if isinstance(vwap_features, dict):
                    features.update(vwap_features)
            
            # Volume Oscillator
            volume_osc_features = self._polars_safe_calculation(
                self._calculate_volume_oscillator_polars, df
            )
            if isinstance(volume_osc_features, dict):
                features.update(volume_osc_features)
            
            # Ease of Movement
            eom_features = self._polars_safe_calculation(
                self._calculate_ease_of_movement_polars, df
            )
            if isinstance(eom_features, dict):
                features.update(eom_features)
            
            # Volume ROC
            vol_roc_features = self._polars_safe_calculation(
                self._calculate_volume_roc_polars, df
            )
            if isinstance(vol_roc_features, dict):
                features.update(vol_roc_features)
                
        except Exception as e:
            logger.error(f"出来高特徴量計算エラー: {e}")
        
        return features
    
    def _calculate_vpt_polars(self, df: pl.DataFrame) -> Dict[str, np.ndarray]:
        """VPT計算（Polarsベース）"""
        
        result_df = df.with_columns([
            pl.col('close').shift(1).alias('prev_close')
        ]).with_columns([
            ((pl.col('close') - pl.col('prev_close')) / (pl.col('prev_close') + 1e-10) * pl.col('volume'))
            .alias('vpt_change')
        ]).with_columns([
            pl.col('vpt_change').cumsum().alias('vpt')
        ]).with_columns([
            pl.col('vpt').diff().alias('vpt_signal')
        ])
        
        # VPTの移動平均
        for period in [10, 20, 50]:
            result_df = result_df.with_columns([
                pl.col('vpt').rolling_mean(window_size=period).alias(f'vpt_ma_{period}')
            ]).with_columns([
                (pl.col('vpt') > pl.col(f'vpt_ma_{period}')).cast(pl.Float64).alias(f'vpt_above_ma_{period}')
            ])
        
        features = {
            'vpt': result_df['vpt'].to_numpy(),
            'vpt_signal': result_df['vpt_signal'].to_numpy()
        }
        
        for period in [10, 20, 50]:
            features[f'vpt_ma_{period}'] = result_df[f'vpt_ma_{period}'].to_numpy()
            features[f'vpt_above_ma_{period}'] = result_df[f'vpt_above_ma_{period}'].to_numpy()
        
        return features
    
    def _calculate_ad_line_polars(self, df: pl.DataFrame) -> Dict[str, np.ndarray]:
        """A/D Line計算（Polarsベース）"""
        
        result_df = df.with_columns([
            # CLV (Close Location Value)
            (((pl.col('close') - pl.col('low')) - (pl.col('high') - pl.col('close'))) / 
             (pl.col('high') - pl.col('low') + 1e-10)).alias('clv')
        ]).with_columns([
            (pl.col('clv') * pl.col('volume')).alias('ad_change')
        ]).with_columns([
            pl.col('ad_change').cumsum().alias('ad_line')
        ]).with_columns([
            pl.col('ad_line').diff().alias('ad_line_momentum')
        ])
        
        # 価格とA/Dラインのダイバージェンス
        result_df = result_df.with_columns([
            pl.col('close').diff().alias('price_momentum')
        ]).with_columns([
            # 正規化してダイバージェンス計算
            (pl.col('price_momentum') / (pl.col('price_momentum').std() + 1e-10) - 
             pl.col('ad_line_momentum') / (pl.col('ad_line_momentum').std() + 1e-10))
            .alias('ad_price_divergence')
        ])
        
        return {
            'ad_line': result_df['ad_line'].to_numpy(),
            'ad_line_momentum': result_df['ad_line_momentum'].to_numpy(),
            'ad_price_divergence': result_df['ad_price_divergence'].to_numpy()
        }
    
    def _calculate_cmf_polars(self, df: pl.DataFrame, period: int) -> Dict[str, np.ndarray]:
        """CMF計算（Polarsベース）"""
        
        result_df = df.with_columns([
            # CLV計算
            (((pl.col('close') - pl.col('low')) - (pl.col('high') - pl.col('close'))) / 
             (pl.col('high') - pl.col('low') + 1e-10)).alias('clv')
        ]).with_columns([
            (pl.col('clv') * pl.col('volume')).alias('money_flow_volume')
        ]).with_columns([
            # CMF計算
            (pl.col('money_flow_volume').rolling_sum(window_size=period) / 
             (pl.col('volume').rolling_sum(window_size=period) + 1e-10)).alias('cmf')
        ]).with_columns([
            (pl.col('cmf') > 0).cast(pl.Float64).alias('cmf_positive'),
            (pl.col('cmf') > 0.2).cast(pl.Float64).alias('cmf_strong_positive'),
            (pl.col('cmf') < -0.2).cast(pl.Float64).alias('cmf_strong_negative')
        ])
        
        return {
            f'cmf_{period}': result_df['cmf'].to_numpy(),
            f'cmf_positive_{period}': result_df['cmf_positive'].to_numpy(),
            f'cmf_strong_positive_{period}': result_df['cmf_strong_positive'].to_numpy(),
            f'cmf_strong_negative_{period}': result_df['cmf_strong_negative'].to_numpy()
        }
    
    def _calculate_chaikin_oscillator_polars(self, df: pl.DataFrame) -> Dict[str, np.ndarray]:
        """チャイキンオシレーター計算（Polarsベース）"""
        
        result_df = df.with_columns([
            # CLV計算
            (((pl.col('close') - pl.col('low')) - (pl.col('high') - pl.col('close'))) / 
             (pl.col('high') - pl.col('low') + 1e-10)).alias('clv')
        ]).with_columns([
            (pl.col('clv') * pl.col('volume')).alias('ad_change')
        ]).with_columns([
            pl.col('ad_change').cumsum().alias('ad_line')
        ]).with_columns([
            # 3日と10日のEMA
            pl.col('ad_line').ewm_mean(span=3, adjust=False).alias('ema3'),
            pl.col('ad_line').ewm_mean(span=10, adjust=False).alias('ema10')
        ]).with_columns([
            (pl.col('ema3') - pl.col('ema10')).alias('chaikin_oscillator')
        ]).with_columns([
            (pl.col('chaikin_oscillator') > 0).cast(pl.Float64).alias('chaikin_positive'),
            pl.col('chaikin_oscillator').diff().alias('chaikin_momentum'),
            (pl.col('chaikin_oscillator') / (pl.col('chaikin_oscillator').std() + 1e-10)).tanh().alias('chaikin_normalized')
        ])
        
        return {
            'chaikin_oscillator': result_df['chaikin_oscillator'].to_numpy(),
            'chaikin_positive': result_df['chaikin_positive'].to_numpy(),
            'chaikin_momentum': result_df['chaikin_momentum'].to_numpy(),
            'chaikin_normalized': result_df['chaikin_normalized'].to_numpy()
        }
    
    def _calculate_mfi_polars(self, df: pl.DataFrame, period: int) -> Dict[str, np.ndarray]:
        """MFI計算（Polarsベース）"""
        
        result_df = df.with_columns([
            # Typical Price
            ((pl.col('high') + pl.col('low') + pl.col('close')) / 3.0).alias('typical_price')
        ]).with_columns([
            (pl.col('typical_price') * pl.col('volume')).alias('raw_money_flow'),
            pl.col('typical_price').shift(1).alias('prev_typical_price')
        ]).with_columns([
            pl.when(pl.col('typical_price') > pl.col('prev_typical_price'))
            .then(pl.col('raw_money_flow'))
            .otherwise(0.0)
            .alias('positive_money_flow'),
            pl.when(pl.col('typical_price') < pl.col('prev_typical_price'))
            .then(pl.col('raw_money_flow'))
            .otherwise(0.0)
            .alias('negative_money_flow')
        ]).with_columns([
            pl.col('positive_money_flow').rolling_sum(window_size=period).alias('positive_mf_sum'),
            pl.col('negative_money_flow').rolling_sum(window_size=period).alias('negative_mf_sum')
        ]).with_columns([
            (pl.col('positive_mf_sum') / (pl.col('negative_mf_sum') + 1e-10)).alias('money_ratio'),
        ]).with_columns([
            (100 - (100 / (1 + pl.col('money_ratio')))).alias('mfi')
        ]).with_columns([
            (pl.col('mfi') > 80).cast(pl.Float64).alias('mfi_overbought'),
            (pl.col('mfi') < 20).cast(pl.Float64).alias('mfi_oversold'),
            ((pl.col('mfi') - 50) / 50.0).alias('mfi_normalized')
        ])
        
        return {
            f'mfi_{period}': result_df['mfi'].to_numpy(),
            f'mfi_overbought_{period}': result_df['mfi_overbought'].to_numpy(),
            f'mfi_oversold_{period}': result_df['mfi_oversold'].to_numpy(),
            f'mfi_normalized_{period}': result_df['mfi_normalized'].to_numpy()
        }
    
    def _calculate_vwap_polars(self, df: pl.DataFrame, period: int) -> Dict[str, np.ndarray]:
        """VWAP計算（Polarsベース）"""
        
        result_df = df.with_columns([
            # Typical Price
            ((pl.col('high') + pl.col('low') + pl.col('close')) / 3.0).alias('typical_price')
        ]).with_columns([
            (pl.col('typical_price') * pl.col('volume')).alias('pv')
        ]).with_columns([
            # VWAP計算
            (pl.col('pv').rolling_sum(window_size=period) / 
             (pl.col('volume').rolling_sum(window_size=period) + 1e-10)).alias('vwap')
        ]).with_columns([
            (pl.col('close') > pl.col('vwap')).cast(pl.Float64).alias('price_above_vwap'),
            ((pl.col('close') - pl.col('vwap')) / (pl.col('close') + 1e-10)).alias('vwap_distance'),
            ((pl.col('close') - pl.col('vwap')).abs() / (pl.col('vwap') + 1e-10)).alias('vwap_deviation')
        ])
        
        return {
            f'vwap_{period}': result_df['vwap'].to_numpy(),
            f'price_above_vwap_{period}': result_df['price_above_vwap'].to_numpy(),
            f'vwap_distance_{period}': result_df['vwap_distance'].to_numpy(),
            f'vwap_deviation_{period}': result_df['vwap_deviation'].to_numpy()
        }
    
    def _calculate_volume_oscillator_polars(self, df: pl.DataFrame) -> Dict[str, np.ndarray]:
        """ボリュームオシレーター計算（Polarsベース）"""
        features = {}
        
        # 複数の期間設定
        period_pairs = [(5, 10), (10, 20), (14, 28)]
        
        for short, long in period_pairs:
            result_df = df.with_columns([
                pl.col('volume').rolling_mean(window_size=short).alias('vol_short_avg'),
                pl.col('volume').rolling_mean(window_size=long).alias('vol_long_avg')
            ]).with_columns([
                (100 * (pl.col('vol_short_avg') - pl.col('vol_long_avg')) / 
                 (pl.col('vol_long_avg') + 1e-10)).alias('volume_oscillator')
            ]).with_columns([
                (pl.col('volume_oscillator') > 0).cast(pl.Float64).alias('vo_positive'),
                pl.col('volume_oscillator').diff().alias('vo_momentum')
            ])
            
            features[f'volume_oscillator_{short}_{long}'] = result_df['volume_oscillator'].to_numpy()
            features[f'vo_positive_{short}_{long}'] = result_df['vo_positive'].to_numpy()
            features[f'vo_momentum_{short}_{long}'] = result_df['vo_momentum'].to_numpy()
        
        return features
    
    def _calculate_ease_of_movement_polars(self, df: pl.DataFrame) -> Dict[str, np.ndarray]:
        """Ease of Movement計算（Polarsベース）"""
        
        result_df = df.with_columns([
            # Distance Moved
            (((pl.col('high') + pl.col('low')) / 2.0) - 
             ((pl.col('high').shift(1) + pl.col('low').shift(1)) / 2.0)).alias('distance_moved'),
            # Box Height
            (pl.col('volume') / (pl.col('high') - pl.col('low') + 1e-10)).alias('box_height')
        ]).with_columns([
            (10000.0 * pl.col('distance_moved') / (pl.col('box_height') + 1e-10)).alias('eom')
        ]).with_columns([
            (pl.col('eom') > 0).cast(pl.Float64).alias('eom_positive'),
            pl.col('eom').diff().alias('eom_momentum')
        ])
        
        # EMVの移動平均
        for period in [14, 20]:
            result_df = result_df.with_columns([
                pl.col('eom').rolling_mean(window_size=period).alias(f'eom_ma_{period}')
            ]).with_columns([
                (pl.col('eom') > pl.col(f'eom_ma_{period}')).cast(pl.Float64).alias(f'eom_signal_{period}')
            ])
        
        features = {
            'ease_of_movement': result_df['eom'].to_numpy(),
            'eom_positive': result_df['eom_positive'].to_numpy(),
            'eom_momentum': result_df['eom_momentum'].to_numpy()
        }
        
        for period in [14, 20]:
            features[f'eom_ma_{period}'] = result_df[f'eom_ma_{period}'].to_numpy()
            features[f'eom_signal_{period}'] = result_df[f'eom_signal_{period}'].to_numpy()
        
        return features
    
    def _calculate_volume_roc_polars(self, df: pl.DataFrame) -> Dict[str, np.ndarray]:
        """出来高変化率計算（Polarsベース）"""
        features = {}
        
        for period in self.params['vol_roc_periods']:
            result_df = df.with_columns([
                pl.col('volume').shift(period).alias('volume_prev')
            ]).with_columns([
                (100 * (pl.col('volume') - pl.col('volume_prev')) / 
                 (pl.col('volume_prev') + 1e-10)).alias('volume_roc')
            ]).with_columns([
                (pl.col('volume_roc') > 0).cast(pl.Float64).alias('vol_roc_positive'),
                (pl.col('volume_roc').abs() > 50).cast(pl.Float64).alias('vol_roc_strong')
            ])
            
            features[f'volume_roc_{period}'] = result_df['volume_roc'].to_numpy()
            features[f'vol_roc_positive_{period}'] = result_df['vol_roc_positive'].to_numpy()
            features[f'vol_roc_strong_{period}'] = result_df['vol_roc_strong'].to_numpy()
        
        return features
    
    # =========================================================================
    # トレンド分析・移動平均線（Polars最適化）
    # =========================================================================
    
    def calculate_moving_averages_polars(self, close: np.ndarray) -> Dict[str, np.ndarray]:
        """移動平均線特徴量の統合計算（Polars最適化版）"""
        features = {}
        
        # Polars DataFrameを構築
        df = self._ensure_polars_df(close, 'close')
        
        try:
            # WMA (Weighted Moving Average)
            wma_features = self._polars_safe_calculation(
                self._calculate_wma_polars, df
            )
            if isinstance(wma_features, dict):
                features.update(wma_features)
            
            # HMA (Hull Moving Average) - Numbaが効率的
            hma_features = self._calculate_hma_features_numba(close)
            features.update(hma_features)
            
            # KAMA (Kaufman Adaptive Moving Average)
            kama_features = self._calculate_kama_features_numba(close)
            features.update(kama_features)
            
            # DEMA, TEMA
            dema_tema_features = self._polars_safe_calculation(
                self._calculate_dema_tema_polars, df
            )
            if isinstance(dema_tema_features, dict):
                features.update(dema_tema_features)
            
            # 移動平均線の傾きと乖離
            ma_analysis_features = self._polars_safe_calculation(
                self._calculate_ma_analysis_polars, df
            )
            if isinstance(ma_analysis_features, dict):
                features.update(ma_analysis_features)
            
            # ゴールデンクロス・デッドクロス
            cross_features = self._polars_safe_calculation(
                self._calculate_cross_signals_polars, df
            )
            if isinstance(cross_features, dict):
                features.update(cross_features)
                
        except Exception as e:
            logger.error(f"移動平均線特徴量計算エラー: {e}")
        
        return features
    
    def _calculate_wma_polars(self, df: pl.DataFrame) -> Dict[str, np.ndarray]:
        """WMA計算（Polarsベース）"""
        features = {}
        
        periods = [9, 21, 50]
        for period in periods:
            # WMAはNumbaが効率的
            close_data = df['close'].to_numpy()
            wma_result = self._numba_safe_calculation(
                self._calculate_wma_vectorized, close_data, period
            )
            
            if wma_result is not None:
                wma_df = pl.DataFrame({
                    'close': close_data,
                    'wma': wma_result
                }).with_columns([
                    (pl.col('close') > pl.col('wma')).cast(pl.Float64).alias('price_above_wma'),
                    pl.col('wma').diff().alias('wma_slope'),
                    ((pl.col('close') - pl.col('wma')) / (pl.col('close') + 1e-10)).alias('wma_distance')
                ])
                
                features[f'wma_{period}'] = wma_df['wma'].to_numpy()
                features[f'price_above_wma_{period}'] = wma_df['price_above_wma'].to_numpy()
                features[f'wma_slope_{period}'] = wma_df['wma_slope'].to_numpy()
                features[f'wma_distance_{period}'] = wma_df['wma_distance'].to_numpy()
            else:
                n = len(df)
                features[f'wma_{period}'] = np.zeros(n)
                features[f'price_above_wma_{period}'] = np.zeros(n)
                features[f'wma_slope_{period}'] = np.zeros(n)
                features[f'wma_distance_{period}'] = np.zeros(n)
        
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
                # Polarsで追加特徴量を計算
                df = pl.DataFrame({
                    'close': close,
                    'hma': hma_result
                }).with_columns([
                    (pl.col('close') > pl.col('hma')).cast(pl.Float64).alias('price_above_hma'),
                    pl.col('hma').diff().alias('hma_slope'),
                    pl.col('hma').diff().diff().alias('hma_momentum')
                ])
                
                features[f'hma_{period}'] = df['hma'].to_numpy()
                features[f'price_above_hma_{period}'] = df['price_above_hma'].to_numpy()
                features[f'hma_slope_{period}'] = df['hma_slope'].to_numpy()
                features[f'hma_momentum_{period}'] = df['hma_momentum'].to_numpy()
            else:
                n = len(close)
                features[f'hma_{period}'] = np.zeros(n)
                features[f'price_above_hma_{period}'] = np.zeros(n)
                features[f'hma_slope_{period}'] = np.zeros(n)
                features[f'hma_momentum_{period}'] = np.zeros(n)
        
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
                
                # Polarsで追加特徴量を計算
                df = pl.DataFrame({
                    'close': close,
                    'kama': kama_values,
                    'efficiency': efficiency_ratio
                }).with_columns([
                    (pl.col('close') > pl.col('kama')).cast(pl.Float64).alias('price_above_kama'),
                    pl.col('kama').diff().alias('kama_slope')
                ])
                
                features[f'kama_{period}'] = df['kama'].to_numpy()
                features[f'price_above_kama_{period}'] = df['price_above_kama'].to_numpy()
                features[f'kama_efficiency_{period}'] = df['efficiency'].to_numpy()
                features[f'kama_slope_{period}'] = df['kama_slope'].to_numpy()
            else:
                n = len(close)
                features[f'kama_{period}'] = np.zeros(n)
                features[f'price_above_kama_{period}'] = np.zeros(n)
                features[f'kama_efficiency_{period}'] = np.zeros(n)
                features[f'kama_slope_{period}'] = np.zeros(n)
        
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
    
    def _calculate_dema_tema_polars(self, df: pl.DataFrame) -> Dict[str, np.ndarray]:
        """DEMA・TEMA計算（Polarsベース）"""
        features = {}
        
        for period in self.params['dema_periods']:
            # DEMA計算
            result_df = df.with_columns([
                pl.col('close').ewm_mean(span=period, adjust=False).alias('ema1')
            ]).with_columns([
                pl.col('ema1').ewm_mean(span=period, adjust=False).alias('ema2')
            ]).with_columns([
                (2 * pl.col('ema1') - pl.col('ema2')).alias('dema')
            ]).with_columns([
                (pl.col('close') > pl.col('dema')).cast(pl.Float64).alias('price_above_dema'),
                pl.col('dema').diff().alias('dema_slope'),
                pl.col('dema').diff().diff().alias('dema_acceleration')
            ])
            
            features[f'dema_{period}'] = result_df['dema'].to_numpy()
            features[f'price_above_dema_{period}'] = result_df['price_above_dema'].to_numpy()
            features[f'dema_slope_{period}'] = result_df['dema_slope'].to_numpy()
            features[f'dema_acceleration_{period}'] = result_df['dema_acceleration'].to_numpy()
        
        for period in self.params['tema_periods']:
            # TEMA計算
            result_df = df.with_columns([
                pl.col('close').ewm_mean(span=period, adjust=False).alias('ema1')
            ]).with_columns([
                pl.col('ema1').ewm_mean(span=period, adjust=False).alias('ema2')
            ]).with_columns([
                pl.col('ema2').ewm_mean(span=period, adjust=False).alias('ema3')
            ]).with_columns([
                (3 * pl.col('ema1') - 3 * pl.col('ema2') + pl.col('ema3')).alias('tema')
            ]).with_columns([
                (pl.col('close') > pl.col('tema')).cast(pl.Float64).alias('price_above_tema'),
                pl.col('tema').diff().alias('tema_momentum'),
                pl.col('tema').diff().abs().alias('tema_trend_strength')
            ])
            
            features[f'tema_{period}'] = result_df['tema'].to_numpy()
            features[f'price_above_tema_{period}'] = result_df['price_above_tema'].to_numpy()
            features[f'tema_momentum_{period}'] = result_df['tema_momentum'].to_numpy()
            features[f'tema_trend_strength_{period}'] = result_df['tema_trend_strength'].to_numpy()
        
        return features
    
    def _calculate_ma_analysis_polars(self, df: pl.DataFrame) -> Dict[str, np.ndarray]:
        """移動平均線分析（Polarsベース）"""
        features = {}
        
        periods = [10, 20, 50, 200]
        
        # SMAとEMAを一括計算
        ma_df = df
        for period in periods:
            ma_df = ma_df.with_columns([
                pl.col('close').rolling_mean(window_size=period).alias(f'sma_{period}'),
                pl.col('close').ewm_mean(span=period, adjust=False).alias(f'ema_{period}')
            ])
        
        # 傾きと乖離率を計算
        for period in periods:
            ma_df = ma_df.with_columns([
                # 傾き
                pl.col(f'sma_{period}').diff().alias(f'sma_slope_{period}'),
                pl.col(f'ema_{period}').diff().alias(f'ema_slope_{period}'),
                # 傾きの強度
                pl.col(f'sma_{period}').diff().abs().alias(f'sma_slope_strength_{period}'),
                pl.col(f'ema_{period}').diff().abs().alias(f'ema_slope_strength_{period}'),
                # 上昇トレンド
                (pl.col(f'sma_{period}').diff() > 0).cast(pl.Float64).alias(f'sma_uptrend_{period}'),
                (pl.col(f'ema_{period}').diff() > 0).cast(pl.Float64).alias(f'ema_uptrend_{period}'),
                # 乖離率
                (100 * (pl.col('close') - pl.col(f'sma_{period}')) / (pl.col(f'sma_{period}') + 1e-10)).alias(f'sma_deviation_{period}'),
                (100 * (pl.col('close') - pl.col(f'ema_{period}')) / (pl.col(f'ema_{period}') + 1e-10)).alias(f'ema_deviation_{period}')
            ]).with_columns([
                # 絶対乖離率
                pl.col(f'sma_deviation_{period}').abs().alias(f'sma_abs_deviation_{period}'),
                pl.col(f'ema_deviation_{period}').abs().alias(f'ema_abs_deviation_{period}'),
                # 過大乖離
                (pl.col(f'sma_deviation_{period}').abs() > 5).cast(pl.Float64).alias(f'sma_excessive_deviation_{period}'),
                (pl.col(f'ema_deviation_{period}').abs() > 5).cast(pl.Float64).alias(f'ema_excessive_deviation_{period}')
            ])
        
        # 結果を辞書に変換
        for period in periods:
            feature_names = [
                f'sma_slope_{period}', f'ema_slope_{period}',
                f'sma_slope_strength_{period}', f'ema_slope_strength_{period}',
                f'sma_uptrend_{period}', f'ema_uptrend_{period}',
                f'sma_deviation_{period}', f'ema_deviation_{period}',
                f'sma_abs_deviation_{period}', f'ema_abs_deviation_{period}',
                f'sma_excessive_deviation_{period}', f'ema_excessive_deviation_{period}'
            ]
            
            for name in feature_names:
                features[name] = ma_df[name].to_numpy()
        
        return features
    
    def _calculate_cross_signals_polars(self, df: pl.DataFrame) -> Dict[str, np.ndarray]:
        """ゴールデンクロス・デッドクロス計算（Polarsベース）"""
        features = {}
        
        # 短期・長期MA組み合わせ
        cross_pairs = [(25, 75), (50, 200), (20, 60)]
        
        for short_period, long_period in cross_pairs:
            cross_df = df.with_columns([
                pl.col('close').rolling_mean(window_size=short_period).alias('short_ma'),
                pl.col('close').rolling_mean(window_size=long_period).alias('long_ma')
            ]).with_columns([
                # 現在のクロス状態
                (pl.col('short_ma') > pl.col('long_ma')).cast(pl.Float64).alias('golden_cross'),
                # MA間距離
                (100 * (pl.col('short_ma') - pl.col('long_ma')) / (pl.col('long_ma') + 1e-10)).alias('ma_distance'),
                # 収束度
                ((pl.col('short_ma') - pl.col('long_ma')).abs() / (pl.col('long_ma') + 1e-10) * 100).alias('ma_convergence')
            ]).with_columns([
                # クロス発生検出
                (pl.col('golden_cross').diff() == 1).cast(pl.Float64).alias('golden_cross_signal'),
                (pl.col('golden_cross').diff() == -1).cast(pl.Float64).alias('death_cross_signal')
            ])
            
            features[f'golden_cross_{short_period}_{long_period}'] = cross_df['golden_cross'].to_numpy()
            features[f'golden_cross_signal_{short_period}_{long_period}'] = cross_df['golden_cross_signal'].to_numpy()
            features[f'death_cross_signal_{short_period}_{long_period}'] = cross_df['death_cross_signal'].to_numpy()
            features[f'ma_distance_{short_period}_{long_period}'] = cross_df['ma_distance'].to_numpy()
            features[f'ma_convergence_{short_period}_{long_period}'] = cross_df['ma_convergence'].to_numpy()
        
        return features
    
    # =========================================================================
    # ボラティリティ・バンド指標（Polars最適化）
    # =========================================================================
    
    def calculate_volatility_bands_polars(self, high: np.ndarray, low: np.ndarray, close: np.ndarray) -> Dict[str, np.ndarray]:
        """ボラティリティ・バンド指標の統合計算（Polars最適化版）"""
        features = {}
        
        # Polars DataFrameを構築
        df = pl.DataFrame({
            'high': high,
            'low': low,
            'close': close
        })
        
        try:
            # ボリンジャーバンドスクイーズ
            bb_squeeze_features = self._polars_safe_calculation(
                self._calculate_bb_squeeze_polars, df
            )
            if isinstance(bb_squeeze_features, dict):
                features.update(bb_squeeze_features)
            
            # ケルトナーチャネル
            kc_features = self._polars_safe_calculation(
                self._calculate_keltner_channel_polars, df
            )
            if isinstance(kc_features, dict):
                features.update(kc_features)
            
            # ドンチャンチャネル
            dc_features = self._polars_safe_calculation(
                self._calculate_donchian_channel_polars, df
            )
            if isinstance(dc_features, dict):
                features.update(dc_features)
            
            # ATRバンド
            atr_bands_features = self._polars_safe_calculation(
                self._calculate_atr_bands_polars, df
            )
            if isinstance(atr_bands_features, dict):
                features.update(atr_bands_features)
            
            # ヒストリカルボラティリティ
            hist_vol_features = self._polars_safe_calculation(
                self._calculate_historical_volatility_polars, df
            )
            if isinstance(hist_vol_features, dict):
                features.update(hist_vol_features)
            
            # ボラティリティレシオ
            vol_ratio_features = self._polars_safe_calculation(
                self._calculate_volatility_ratio_polars, df
            )
            if isinstance(vol_ratio_features, dict):
                features.update(vol_ratio_features)
            
            # シャンデリアエグジット
            chandelier_features = self._calculate_chandelier_exit_features_numba(high, low, close)
            features.update(chandelier_features)
            
            # ボラティリティブレイクアウト
            vol_breakout_features = self._polars_safe_calculation(
                self._calculate_volatility_breakout_polars, df
            )
            if isinstance(vol_breakout_features, dict):
                features.update(vol_breakout_features)
                
        except Exception as e:
            logger.error(f"ボラティリティ・バンド指標計算エラー: {e}")
        
        return features
    
    def _calculate_bb_squeeze_polars(self, df: pl.DataFrame) -> Dict[str, np.ndarray]:
        """ボリンジャーバンドスクイーズ計算（Polarsベース）"""
        features = {}
        
        for period, std_mult in self.params['volatility_bb_settings']:
            # ボリンジャーバンドを計算
            bb_df = df.with_columns([
                pl.col('close').rolling_mean(window_size=period).alias('bb_mid'),
                pl.col('close').rolling_std(window_size=period).alias('bb_std')
            ]).with_columns([
                (pl.col('bb_mid') + std_mult * pl.col('bb_std')).alias('bb_upper'),
                (pl.col('bb_mid') - std_mult * pl.col('bb_std')).alias('bb_lower')
            ]).with_columns([
                # バンド幅
                ((pl.col('bb_upper') - pl.col('bb_lower')) / (pl.col('bb_mid') + 1e-10)).alias('bb_width')
            ]).with_columns([
                # バンド幅の移動平均
                pl.col('bb_width').rolling_mean(window_size=20).alias('bb_width_ma'),
                # バンド幅パーセンタイル
                pl.col('bb_width').rolling_quantile(quantile=0.8, window_size=100).alias('bb_width_80th')
            ]).with_columns([
                # スクイーズ検出
                (pl.col('bb_width') < pl.col('bb_width_ma')).cast(pl.Float64).alias('bb_squeeze'),
                # バンド幅変化率
                pl.col('bb_width').diff().alias('bb_width_change'),
                # エクスパンション検出
                (pl.col('bb_width').diff() > pl.col('bb_width').rolling_std(window_size=20)).cast(pl.Float64).alias('bb_expansion')
            ]).with_columns([
                # バンド幅パーセンタイル（ローリング）
                (pl.col('bb_width').rank(method='average') / pl.count()).alias('bb_width_percentile')
            ])
            
            features[f'bb_squeeze_{period}'] = bb_df['bb_squeeze'].to_numpy()
            features[f'bb_width_{period}'] = bb_df['bb_width'].to_numpy()
            features[f'bb_width_percentile_{period}'] = bb_df['bb_width_percentile'].to_numpy()
            features[f'bb_expansion_{period}'] = bb_df['bb_expansion'].to_numpy()
        
        return features
    
    def _calculate_keltner_channel_polars(self, df: pl.DataFrame) -> Dict[str, np.ndarray]:
        """ケルトナーチャネル計算（Polarsベース）"""
        features = {}
        
        for period in self.params['kc_periods']:
            # ATRを事前計算
            atr_df = df.with_columns([
                pl.col('close').shift(1).alias('prev_close')
            ]).with_columns([
                pl.max_horizontal([
                    pl.col('high') - pl.col('low'),
                    (pl.col('high') - pl.col('prev_close')).abs(),
                    (pl.col('low') - pl.col('prev_close')).abs()
                ]).alias('true_range')
            ]).with_columns([
                pl.col('true_range').ewm_mean(span=period, adjust=False).alias('atr')
            ])
            
            # ケルトナーチャネル
            kc_df = atr_df.with_columns([
                pl.col('close').ewm_mean(span=period, adjust=False).alias('kc_middle')
            ]).with_columns([
                (pl.col('kc_middle') + 2.0 * pl.col('atr')).alias('kc_upper'),
                (pl.col('kc_middle') - 2.0 * pl.col('atr')).alias('kc_lower')
            ]).with_columns([
                # ケルトナーチャネル内の価格位置
                ((pl.col('close') - pl.col('kc_lower')) / (pl.col('kc_upper') - pl.col('kc_lower') + 1e-8)).alias('kc_position'),
                # ケルトナーチャネル幅
                ((pl.col('kc_upper') - pl.col('kc_lower')) / (pl.col('kc_middle') + 1e-10)).alias('kc_width'),
                # ブレイクアウト検出
                (pl.col('close') > pl.col('kc_upper')).cast(pl.Float64).alias('kc_upper_break'),
                (pl.col('close') < pl.col('kc_lower')).cast(pl.Float64).alias('kc_lower_break'),
                # チャネル内
                ((pl.col('close') >= pl.col('kc_lower')) & (pl.col('close') <= pl.col('kc_upper'))).cast(pl.Float64).alias('kc_inside')
            ])
            
            features[f'kc_upper_{period}'] = kc_df['kc_upper'].to_numpy()
            features[f'kc_middle_{period}'] = kc_df['kc_middle'].to_numpy()
            features[f'kc_lower_{period}'] = kc_df['kc_lower'].to_numpy()
            features[f'kc_position_{period}'] = kc_df['kc_position'].to_numpy()
            features[f'kc_width_{period}'] = kc_df['kc_width'].to_numpy()
            features[f'kc_upper_break_{period}'] = kc_df['kc_upper_break'].to_numpy()
            features[f'kc_lower_break_{period}'] = kc_df['kc_lower_break'].to_numpy()
            features[f'kc_inside_{period}'] = kc_df['kc_inside'].to_numpy()
        
        return features
    
    def _calculate_donchian_channel_polars(self, df: pl.DataFrame) -> Dict[str, np.ndarray]:
        """ドンチャンチャネル計算（Polarsベース）"""
        features = {}
        
        for period in self.params['dc_periods']:
            dc_df = df.with_columns([
                pl.col('high').rolling_max(window_size=period).alias('dc_upper'),
                pl.col('low').rolling_min(window_size=period).alias('dc_lower')
            ]).with_columns([
                ((pl.col('dc_upper') + pl.col('dc_lower')) / 2.0).alias('dc_middle')
            ]).with_columns([
                # ドンチャンチャネル内の価格位置
                ((pl.col('close') - pl.col('dc_lower')) / (pl.col('dc_upper') - pl.col('dc_lower') + 1e-8)).alias('dc_position'),
                # ドンチャンチャネル幅
                ((pl.col('dc_upper') - pl.col('dc_lower')) / (pl.col('dc_middle') + 1e-10)).alias('dc_width'),
                # ブレイクアウト検出
                (pl.col('close') > pl.col('dc_upper')).cast(pl.Float64).alias('dc_upper_break'),
                (pl.col('close') < pl.col('dc_lower')).cast(pl.Float64).alias('dc_lower_break')
            ])
            
            # 新高値・新安値からの日数（Numbaが効率的）
            close_data = df['close'].to_numpy()
            dc_upper_data = dc_df['dc_upper'].to_numpy()
            dc_lower_data = dc_df['dc_lower'].to_numpy()
            
            days_since_high = self._numba_safe_calculation(
                self._calculate_days_since_extreme_vectorized, close_data, dc_upper_data, True
            )
            days_since_low = self._numba_safe_calculation(
                self._calculate_days_since_extreme_vectorized, close_data, dc_lower_data, False
            )
            
            features[f'dc_upper_{period}'] = dc_df['dc_upper'].to_numpy()
            features[f'dc_lower_{period}'] = dc_df['dc_lower'].to_numpy()
            features[f'dc_middle_{period}'] = dc_df['dc_middle'].to_numpy()
            features[f'dc_position_{period}'] = dc_df['dc_position'].to_numpy()
            features[f'dc_width_{period}'] = dc_df['dc_width'].to_numpy()
            features[f'dc_upper_break_{period}'] = dc_df['dc_upper_break'].to_numpy()
            features[f'dc_lower_break_{period}'] = dc_df['dc_lower_break'].to_numpy()
            
            if days_since_high is not None:
                features[f'dc_days_since_high_{period}'] = days_since_high
            else:
                features[f'dc_days_since_high_{period}'] = np.zeros(len(df))
            
            if days_since_low is not None:
                features[f'dc_days_since_low_{period}'] = days_since_low
            else:
                features[f'dc_days_since_low_{period}'] = np.zeros(len(df))
        
        return features
    
    @staticmethod
    @njit(cache=True)
    def _calculate_days_since_extreme_vectorized(close: np.ndarray, extreme_line: np.ndarray, is_high: bool) -> np.ndarray:
        """極値からの日数計算（Numba最適化版）"""
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
    
    def _calculate_atr_bands_polars(self, df: pl.DataFrame) -> Dict[str, np.ndarray]:
        """ATRバンド計算（Polarsベース）"""
        features = {}
        
        atr_multipliers = [1.0, 1.5, 2.0, 2.5]
        
        for period in self.params['atr_periods_vol']:
            # ATRを事前計算
            atr_df = df.with_columns([
                pl.col('close').shift(1).alias('prev_close')
            ]).with_columns([
                pl.max_horizontal([
                    pl.col('high') - pl.col('low'),
                    (pl.col('high') - pl.col('prev_close')).abs(),
                    (pl.col('low') - pl.col('prev_close')).abs()
                ]).alias('true_range')
            ]).with_columns([
                pl.col('true_range').rolling_mean(window_size=period).alias('atr')
            ])
            
            for mult in atr_multipliers:
                atr_bands_df = atr_df.with_columns([
                    (pl.col('close') + mult * pl.col('atr')).alias('atr_upper'),
                    (pl.col('close') - mult * pl.col('atr')).alias('atr_lower')
                ]).with_columns([
                    # ATRバンド内位置
                    ((pl.col('close') - pl.col('atr_lower')) / (pl.col('atr_upper') - pl.col('atr_lower') + 1e-8)).alias('atr_band_position'),
                    # ATRバンド幅
                    ((pl.col('atr_upper') - pl.col('atr_lower')) / (pl.col('close') + 1e-10)).alias('atr_band_width')
                ])
                
                features[f'atr_upper_{period}_{mult}'] = atr_bands_df['atr_upper'].to_numpy()
                features[f'atr_lower_{period}_{mult}'] = atr_bands_df['atr_lower'].to_numpy()
                features[f'atr_band_position_{period}_{mult}'] = atr_bands_df['atr_band_position'].to_numpy()
                features[f'atr_band_width_{period}_{mult}'] = atr_bands_df['atr_band_width'].to_numpy()
        
        return features
    
    def _calculate_historical_volatility_polars(self, df: pl.DataFrame) -> Dict[str, np.ndarray]:
        """ヒストリカルボラティリティ計算（Polarsベース）"""
        features = {}
        
        # 対数リターン計算
        vol_df = df.with_columns([
            pl.col('close').log().diff().alias('log_returns')
        ])
        
        for period in self.params['hist_vol_periods']:
            vol_df = vol_df.with_columns([
                # ローリング標準偏差
                pl.col('log_returns').rolling_std(window_size=period).alias(f'hist_vol_{period}'),
                # 年率化（252営業日ベース）
                (pl.col('log_returns').rolling_std(window_size=period) * np.sqrt(252)).alias(f'hist_vol_annualized_{period}')
            ]).with_columns([
                # ボラティリティの相対水準（パーセンタイル）
                pl.col(f'hist_vol_{period}').rolling_quantile(quantile=0.5, window_size=252).alias(f'vol_median_{period}'),
                pl.col(f'hist_vol_{period}').rolling_mean(window_size=50).alias(f'vol_ma_{period}')
            ]).with_columns([
                # ボラティリティレジーム
                (pl.col(f'hist_vol_{period}') > pl.col(f'vol_ma_{period}')).cast(pl.Float64).alias(f'vol_regime_{period}'),
                # ボラティリティパーセンタイル
                (pl.col(f'hist_vol_{period}').rank() / pl.count()).alias(f'vol_percentile_{period}')
            ])
            
            features[f'hist_vol_{period}'] = vol_df[f'hist_vol_{period}'].to_numpy()
            features[f'hist_vol_annualized_{period}'] = vol_df[f'hist_vol_annualized_{period}'].to_numpy()
            features[f'vol_percentile_{period}'] = vol_df[f'vol_percentile_{period}'].to_numpy()
            features[f'vol_regime_{period}'] = vol_df[f'vol_regime_{period}'].to_numpy()
        
        return features
    
    def _calculate_volatility_ratio_polars(self, df: pl.DataFrame) -> Dict[str, np.ndarray]:
        """ボラティリティレシオ計算（Polarsベース）"""
        features = {}
        
        # 対数リターン計算
        vol_df = df.with_columns([
            pl.col('close').log().diff().alias('log_returns')
        ])
        
        # 短期・長期ボラティリティ比較
        vol_pairs = [(5, 20), (10, 30), (20, 60)]
        
        for short, long in vol_pairs:
            vol_df = vol_df.with_columns([
                pl.col('log_returns').rolling_std(window_size=short).alias(f'short_vol_{short}'),
                pl.col('log_returns').rolling_std(window_size=long).alias(f'long_vol_{long}')
            ]).with_columns([
                (pl.col(f'short_vol_{short}') / (pl.col(f'long_vol_{long}') + 1e-10)).alias(f'volatility_ratio_{short}_{long}')
            ]).with_columns([
                (pl.col(f'volatility_ratio_{short}_{long}') > 1.2).cast(pl.Float64).alias(f'vol_ratio_high_{short}_{long}'),
                (pl.col(f'volatility_ratio_{short}_{long}') < 0.8).cast(pl.Float64).alias(f'vol_ratio_low_{short}_{long}'),
                # ボラティリティ変化率
                pl.col(f'volatility_ratio_{short}_{long}').diff().alias(f'vol_change_{short}_{long}')
            ])
            
            features[f'volatility_ratio_{short}_{long}'] = vol_df[f'volatility_ratio_{short}_{long}'].to_numpy()
            features[f'vol_ratio_high_{short}_{long}'] = vol_df[f'vol_ratio_high_{short}_{long}'].to_numpy()
            features[f'vol_ratio_low_{short}_{long}'] = vol_df[f'vol_ratio_low_{short}_{long}'].to_numpy()
            features[f'vol_change_{short}_{long}'] = vol_df[f'vol_change_{short}_{long}'].to_numpy()
        
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
                    
                    # Polarsで追加特徴量を計算
                    df = pl.DataFrame({
                        'close': close,
                        'chandelier_long': chandelier_long,
                        'chandelier_short': chandelier_short
                    }).with_columns([
                        # エグジットシグナル
                        (pl.col('close') < pl.col('chandelier_long')).cast(pl.Float64).alias('chandelier_long_exit'),
                        (pl.col('close') > pl.col('chandelier_short')).cast(pl.Float64).alias('chandelier_short_exit'),
                        # 価格との距離
                        ((pl.col('close') - pl.col('chandelier_long')) / (pl.col('close') + 1e-10)).alias('chandelier_long_distance'),
                        ((pl.col('chandelier_short') - pl.col('close')) / (pl.col('close') + 1e-10)).alias('chandelier_short_distance')
                    ])
                    
                    features[f'chandelier_long_{period}_{mult}'] = df['chandelier_long'].to_numpy()
                    features[f'chandelier_short_{period}_{mult}'] = df['chandelier_short'].to_numpy()
                    features[f'chandelier_long_exit_{period}_{mult}'] = df['chandelier_long_exit'].to_numpy()
                    features[f'chandelier_short_exit_{period}_{mult}'] = df['chandelier_short_exit'].to_numpy()
                    features[f'chandelier_long_distance_{period}_{mult}'] = df['chandelier_long_distance'].to_numpy()
                    features[f'chandelier_short_distance_{period}_{mult}'] = df['chandelier_short_distance'].to_numpy()
                else:
                    # フォールバック
                    n = len(close)
                    features[f'chandelier_long_{period}_{mult}'] = np.zeros(n)
                    features[f'chandelier_short_{period}_{mult}'] = np.zeros(n)
                    features[f'chandelier_long_exit_{period}_{mult}'] = np.zeros(n)
                    features[f'chandelier_short_exit_{period}_{mult}'] = np.zeros(n)
                    features[f'chandelier_long_distance_{period}_{mult}'] = np.zeros(n)
                    features[f'chandelier_short_distance_{period}_{mult}'] = np.zeros(n)
        
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
    
    def _calculate_volatility_breakout_polars(self, df: pl.DataFrame) -> Dict[str, np.ndarray]:
        """ボラティリティブレイクアウト計算（Polarsベース）"""
        features = {}
        
        # 前日比変動率
        breakout_df = df.with_columns([
            ((pl.col('high') - pl.col('low')) / (pl.col('close') + 1e-10)).alias('daily_range'),
            (pl.col('close').diff() / (pl.col('close') + 1e-10)).alias('price_change')
        ])
        
        # 複数期間でのボラティリティ分析
        periods = [5, 10, 20, 50]
        
        for period in periods:
            breakout_df = breakout_df.with_columns([
                # 平均日足レンジ
                pl.col('daily_range').rolling_mean(window_size=period).alias(f'avg_range_{period}'),
                # 価格変動の標準偏差
                pl.col('price_change').rolling_std(window_size=period).alias(f'price_change_std_{period}')
            ]).with_columns([
                # レンジ拡張
                (pl.col('daily_range') / (pl.col(f'avg_range_{period}') + 1e-10)).alias(f'range_expansion_{period}'),
                # 価格変動のZ-score
                (pl.col('price_change') / (pl.col(f'price_change_std_{period}') + 1e-10)).alias(f'price_change_z_{period}')
            ]).with_columns([
                # 高レンジ日
                (pl.col(f'range_expansion_{period}') > 1.5).cast(pl.Float64).alias(f'high_range_day_{period}'),
                # 低レンジ日
                (pl.col(f'range_expansion_{period}') < 0.5).cast(pl.Float64).alias(f'low_range_day_{period}'),
                # 価格ブレイクアウト
                (pl.col(f'price_change_z_{period}').abs() > 2.0).cast(pl.Float64).alias(f'price_breakout_{period}')
            ])
            
            features[f'range_expansion_{period}'] = breakout_df[f'range_expansion_{period}'].to_numpy()
            features[f'high_range_day_{period}'] = breakout_df[f'high_range_day_{period}'].to_numpy()
            features[f'low_range_day_{period}'] = breakout_df[f'low_range_day_{period}'].to_numpy()
            features[f'price_change_z_{period}'] = breakout_df[f'price_change_z_{period}'].to_numpy()
            features[f'price_breakout_{period}'] = breakout_df[f'price_breakout_{period}'].to_numpy()
        
        return features
    
    # =========================================================================
    # サポート・レジスタンス・ローソク足（Polars最適化）
    # =========================================================================
    
    def calculate_support_resistance_polars(self, high: np.ndarray, low: np.ndarray, 
                                          close: np.ndarray, open_prices: np.ndarray = None) -> Dict[str, np.ndarray]:
        """サポート・レジスタンス・ローソク足特徴量の統合計算（Polars最適化版）"""
        features = {}
        
        # Open価格がない場合はCloseで代用
        if open_prices is None:
            open_prices = np.roll(close, 1)
            open_prices[0] = close[0]
        
        # Polars DataFrameを構築
        df = pl.DataFrame({
            'high': high,
            'low': low,
            'close': close,
            'open': open_prices
        })
        
        try:
            # ピボットポイント
            pivot_features = self._polars_safe_calculation(
                self._calculate_pivot_points_polars, df
            )
            if isinstance(pivot_features, dict):
                features.update(pivot_features)
            
            # プライスチャネル
            price_channel_features = self._polars_safe_calculation(
                self._calculate_price_channels_polars, df
            )
            if isinstance(price_channel_features, dict):
                features.update(price_channel_features)
            
            # フィボナッチレベル
            fibonacci_features = self._calculate_fibonacci_features_numba(high, low, close)
            features.update(fibonacci_features)
            
            # サポート・レジスタンス検出
            sr_features = self._calculate_support_resistance_features_numba(high, low, close)
            features.update(sr_features)
            
            # ローソク足パターン
            candlestick_features = self._polars_safe_calculation(
                self._calculate_candlestick_patterns_polars, df
            )
            if isinstance(candlestick_features, dict):
                features.update(candlestick_features)
            
            # 複数ローソク足パターン
            multi_candle_features = self._calculate_multi_candle_patterns_numba(open_prices, high, low, close)
            features.update(multi_candle_features)
            
            # ローソク足強度
            candle_strength_features = self._polars_safe_calculation(
                self._calculate_candle_strength_polars, df
            )
            if isinstance(candle_strength_features, dict):
                features.update(candle_strength_features)
                
        except Exception as e:
            logger.error(f"サポート・レジスタンス・ローソク足特徴量計算エラー: {e}")
        
        return features
    
    def _calculate_pivot_points_polars(self, df: pl.DataFrame) -> Dict[str, np.ndarray]:
        """ピボットポイント計算（Polarsベース）"""
        
        pivot_df = df.with_columns([
            # 前日の高値、安値、終値
            pl.col('high').shift(1).alias('prev_high'),
            pl.col('low').shift(1).alias('prev_low'),
            pl.col('close').shift(1).alias('prev_close')
        ]).with_columns([
            # ピボットポイント
            ((pl.col('prev_high') + pl.col('prev_low') + pl.col('prev_close')) / 3.0).alias('pivot_point')
        ]).with_columns([
            # レジスタンス・サポート
            (2 * pl.col('pivot_point') - pl.col('prev_low')).alias('resistance1'),
            (2 * pl.col('pivot_point') - pl.col('prev_high')).alias('support1'),
            (pl.col('pivot_point') + (pl.col('prev_high') - pl.col('prev_low'))).alias('resistance2'),
            (pl.col('pivot_point') - (pl.col('prev_high') - pl.col('prev_low'))).alias('support2')
        ]).with_columns([
            # 現在価格との距離
            (100 * (pl.col('close') - pl.col('pivot_point')) / (pl.col('close') + 1e-10)).alias('distance_to_pivot'),
            (100 * (pl.col('resistance1') - pl.col('close')) / (pl.col('close') + 1e-10)).alias('distance_to_r1'),
            (100 * (pl.col('resistance2') - pl.col('close')) / (pl.col('close') + 1e-10)).alias('distance_to_r2'),
            (100 * (pl.col('close') - pl.col('support1')) / (pl.col('close') + 1e-10)).alias('distance_to_s1'),
            (100 * (pl.col('close') - pl.col('support2')) / (pl.col('close') + 1e-10)).alias('distance_to_s2'),
            # レベル突破検出
            (pl.col('close') > pl.col('pivot_point')).cast(pl.Float64).alias('above_pivot'),
            (pl.col('close') > pl.col('resistance1')).cast(pl.Float64).alias('above_r1'),
            (pl.col('close') > pl.col('resistance2')).cast(pl.Float64).alias('above_r2'),
            (pl.col('close') < pl.col('support1')).cast(pl.Float64).alias('below_s1'),
            (pl.col('close') < pl.col('support2')).cast(pl.Float64).alias('below_s2')
        ])
        
        return {
            'pivot_point': pivot_df['pivot_point'].to_numpy(),
            'resistance1': pivot_df['resistance1'].to_numpy(),
            'resistance2': pivot_df['resistance2'].to_numpy(),
            'support1': pivot_df['support1'].to_numpy(),
            'support2': pivot_df['support2'].to_numpy(),
            'distance_to_pivot': pivot_df['distance_to_pivot'].to_numpy(),
            'distance_to_r1': pivot_df['distance_to_r1'].to_numpy(),
            'distance_to_r2': pivot_df['distance_to_r2'].to_numpy(),
            'distance_to_s1': pivot_df['distance_to_s1'].to_numpy(),
            'distance_to_s2': pivot_df['distance_to_s2'].to_numpy(),
            'above_pivot': pivot_df['above_pivot'].to_numpy(),
            'above_r1': pivot_df['above_r1'].to_numpy(),
            'above_r2': pivot_df['above_r2'].to_numpy(),
            'below_s1': pivot_df['below_s1'].to_numpy(),
            'below_s2': pivot_df['below_s2'].to_numpy()
        }
    
    def _calculate_price_channels_polars(self, df: pl.DataFrame) -> Dict[str, np.ndarray]:
        """プライスチャネル計算（Polarsベース）"""
        features = {}
        
        for period in self.params['price_channel_periods']:
            channel_df = df.with_columns([
                pl.col('high').rolling_max(window_size=period).alias('channel_high'),
                pl.col('low').rolling_min(window_size=period).alias('channel_low')
            ]).with_columns([
                ((pl.col('channel_high') + pl.col('channel_low')) / 2.0).alias('channel_mid')
            ]).with_columns([
                # チャネル内の価格位置
                ((pl.col('close') - pl.col('channel_low')) / (pl.col('channel_high') - pl.col('channel_low') + 1e-8)).alias('channel_position'),
                # チャネル幅
                ((pl.col('channel_high') - pl.col('channel_low')) / (pl.col('channel_mid') + 1e-10)).alias('channel_width'),
                # ブレイクアウト検出
                (pl.col('close') > pl.col('channel_high')).cast(pl.Float64).alias('channel_breakout_up'),
                (pl.col('close') < pl.col('channel_low')).cast(pl.Float64).alias('channel_breakout_down')
            ])
            
            features[f'channel_high_{period}'] = channel_df['channel_high'].to_numpy()
            features[f'channel_low_{period}'] = channel_df['channel_low'].to_numpy()
            features[f'channel_mid_{period}'] = channel_df['channel_mid'].to_numpy()
            features[f'channel_position_{period}'] = channel_df['channel_position'].to_numpy()
            features[f'channel_width_{period}'] = channel_df['channel_width'].to_numpy()
            features[f'channel_breakout_up_{period}'] = channel_df['channel_breakout_up'].to_numpy()
            features[f'channel_breakout_down_{period}'] = channel_df['channel_breakout_down'].to_numpy()
        
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
                        features[f'fib_retracement_{level_str}_{period}'] = np.zeros(len(close))
                        features[f'fib_extension_{level_str}_{period}'] = np.zeros(len(close))
                        features[f'near_fib_retracement_{level_str}_{period}'] = np.zeros(len(close))
            else:
                # フォールバック
                for level in fib_levels:
                    level_str = str(level).replace('.', '')
                    features[f'fib_retracement_{level_str}_{period}'] = np.zeros(len(close))
                    features[f'fib_extension_{level_str}_{period}'] = np.zeros(len(close))
                    features[f'near_fib_retracement_{level_str}_{period}'] = np.zeros(len(close))
        
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
                features[f'nearest_resistance_{period}'] = np.zeros(n)
                features[f'nearest_support_{period}'] = np.zeros(n)
                features[f'resistance_distance_{period}'] = np.zeros(n)
                features[f'support_distance_{period}'] = np.zeros(n)
                features[f'resistance_strength_{period}'] = np.zeros(n)
                features[f'support_strength_{period}'] = np.zeros(n)
        
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
    
    def _calculate_candlestick_patterns_polars(self, df: pl.DataFrame) -> Dict[str, np.ndarray]:
        """ローソク足パターン計算（Polarsベース）"""
        
        # 基本的なローソク足要素
        candle_df = df.with_columns([
            (pl.col('close') - pl.col('open')).abs().alias('body_size'),
            (pl.col('high') - pl.max_horizontal([pl.col('open'), pl.col('close')])).alias('upper_shadow'),
            (pl.min_horizontal([pl.col('open'), pl.col('close')]) - pl.col('low')).alias('lower_shadow'),
            (pl.col('high') - pl.col('low')).alias('total_range')
        ]).with_columns([
            # 正規化（全体レンジに対する比率）
            (pl.col('body_size') / (pl.col('total_range') + 1e-8)).alias('body_ratio'),
            (pl.col('upper_shadow') / (pl.col('total_range') + 1e-8)).alias('upper_shadow_ratio'),
            (pl.col('lower_shadow') / (pl.col('total_range') + 1e-8)).alias('lower_shadow_ratio'),
            # ローソク足の方向
            (pl.col('close') > pl.col('open')).cast(pl.Float64).alias('is_bullish'),
            (pl.col('close') < pl.col('open')).cast(pl.Float64).alias('is_bearish')
        ]).with_columns([
            # 十字線（Doji）パターン
            (pl.col('body_ratio') < 0.1).cast(pl.Float64).alias('doji'),
            ((pl.col('body_ratio') < 0.1) & (pl.col('upper_shadow_ratio') > 0.3) & (pl.col('lower_shadow_ratio') > 0.3)).cast(pl.Float64).alias('long_legged_doji'),
            ((pl.col('body_ratio') < 0.1) & (pl.col('upper_shadow_ratio') < 0.1) & (pl.col('lower_shadow_ratio') > 0.3)).cast(pl.Float64).alias('dragonfly_doji'),
            ((pl.col('body_ratio') < 0.1) & (pl.col('upper_shadow_ratio') > 0.3) & (pl.col('lower_shadow_ratio') < 0.1)).cast(pl.Float64).alias('gravestone_doji')
        ]).with_columns([
            # ハンマー・ハンギングマン
            ((pl.col('lower_shadow_ratio') > 0.5) & (pl.col('upper_shadow_ratio') < 0.1) & (pl.col('body_ratio') < 0.3)).alias('hammer_condition'),
            # シューティングスター・インバーテッドハンマー
            ((pl.col('upper_shadow_ratio') > 0.5) & (pl.col('lower_shadow_ratio') < 0.1) & (pl.col('body_ratio') < 0.3)).alias('shooting_star_condition'),
            # マルボウズ（影のないローソク足）
            ((pl.col('upper_shadow_ratio') < 0.05) & (pl.col('lower_shadow_ratio') < 0.05) & (pl.col('body_ratio') > 0.9)).cast(pl.Float64).alias('marubozu'),
            # スピニングトップ
            ((pl.col('body_ratio') < 0.3) & (pl.col('upper_shadow_ratio') > 0.2) & (pl.col('lower_shadow_ratio') > 0.2)).cast(pl.Float64).alias('spinning_top')
        ]).with_columns([
            # ハンマー系パターンの分類
            (pl.col('hammer_condition') & pl.col('is_bullish')).cast(pl.Float64).alias('hammer'),
            (pl.col('hammer_condition') & pl.col('is_bearish')).cast(pl.Float64).alias('hanging_man'),
            (pl.col('shooting_star_condition') & pl.col('is_bearish')).cast(pl.Float64).alias('shooting_star'),
            (pl.col('shooting_star_condition') & pl.col('is_bullish')).cast(pl.Float64).alias('inverted_hammer'),
            # マルボウズの分類
            (pl.col('marubozu') & pl.col('is_bullish')).cast(pl.Float64).alias('white_marubozu'),
            (pl.col('marubozu') & pl.col('is_bearish')).cast(pl.Float64).alias('black_marubozu')
        ])
        
        pattern_names = [
            'body_size', 'upper_shadow', 'lower_shadow', 'body_ratio', 'upper_shadow_ratio', 'lower_shadow_ratio',
            'is_bullish', 'is_bearish', 'doji', 'long_legged_doji', 'dragonfly_doji', 'gravestone_doji',
            'hammer', 'hanging_man', 'shooting_star', 'inverted_hammer', 'marubozu', 'white_marubozu', 'black_marubozu', 'spinning_top'
        ]
        
        return {name: candle_df[name].to_numpy() for name in pattern_names}
    
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
                    features[name] = np.zeros(len(close))
        else:
            # フォールバック
            pattern_names = ['bullish_engulfing', 'bearish_engulfing', 'bullish_harami', 'bearish_harami', 
                           'gap_up', 'gap_down', 'has_gap']
            for name in pattern_names:
                features[name] = np.zeros(len(close))
        
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
    
    def _calculate_candle_strength_polars(self, df: pl.DataFrame) -> Dict[str, np.ndarray]:
        """ローソク足強度計算（Polarsベース）"""
        features = {}
        
        # 基本要素
        strength_df = df.with_columns([
            (pl.col('close') - pl.col('open')).abs().alias('body_size'),
            (pl.col('high') - pl.col('low')).alias('total_range')
        ])
        
        # 複数期間での相対的評価
        periods = [5, 10, 20]
        
        for period in periods:
            strength_df = strength_df.with_columns([
                # 実体サイズの相対的大きさ
                (pl.col('body_size').rank().over(pl.col('body_size').rolling_mean(window_size=period)) / period).alias(f'body_percentile_{period}'),
                (pl.col('total_range').rank().over(pl.col('total_range').rolling_mean(window_size=period)) / period).alias(f'range_percentile_{period}')
            ]).with_columns([
                # 強いローソク足の検出
                (pl.col(f'body_percentile_{period}') > 0.8).cast(pl.Float64).alias(f'strong_candle_{period}'),
                (pl.col(f'body_percentile_{period}') < 0.2).cast(pl.Float64).alias(f'weak_candle_{period}'),
                (pl.col(f'range_percentile_{period}') > 0.8).cast(pl.Float64).alias(f'high_volatility_candle_{period}')
            ])
            
            features[f'body_percentile_{period}'] = strength_df[f'body_percentile_{period}'].to_numpy()
            features[f'range_percentile_{period}'] = strength_df[f'range_percentile_{period}'].to_numpy()
            features[f'strong_candle_{period}'] = strength_df[f'strong_candle_{period}'].to_numpy()
            features[f'weak_candle_{period}'] = strength_df[f'weak_candle_{period}'].to_numpy()
            features[f'high_volatility_candle_{period}'] = strength_df[f'high_volatility_candle_{period}'].to_numpy()
        
        # ローソク足の連続性（Numbaが効率的）
        consecutive_result = self._numba_safe_calculation(
            self._calculate_consecutive_candles_vectorized, open_prices=df['open'].to_numpy(), close=df['close'].to_numpy()
        )
        
        if consecutive_result is not None and len(consecutive_result.shape) == 2:
            features['consecutive_bullish'] = consecutive_result[:, 0]
            features['consecutive_bearish'] = consecutive_result[:, 1]
            features['long_bullish_streak'] = consecutive_result[:, 2]
            features['long_bearish_streak'] = consecutive_result[:, 3]
        else:
            # フォールバック
            n = len(df)
            features['consecutive_bullish'] = np.zeros(n)
            features['consecutive_bearish'] = np.zeros(n)
            features['long_bullish_streak'] = np.zeros(n)
            features['long_bearish_streak'] = np.zeros(n)
        
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
    
    # =========================================================================
    # Block 9: 情報理論・学際的アナロジー特徴量（Polars+Numba最適化版）
    # =========================================================================
    
    def calculate_information_theory_features_polars(self, data: np.ndarray) -> Dict[str, np.ndarray]:
        """情報理論特徴量の統合計算（Polars最適化版）"""
        features = {}
        
        try:
            # 高度なエントロピー特徴量
            entropy_features = self.calculate_advanced_entropy_features_polars(data)
            features.update(entropy_features)
            
            # サンプル・近似エントロピー
            if ENTROPY_AVAILABLE:
                sample_entropy_features = self.calculate_sample_approximate_entropy_polars(data)
                features.update(sample_entropy_features)
            
            # Lempel-Ziv複雑性
            lz_features = self.calculate_lempel_ziv_features_polars(data)
            features.update(lz_features)
            
            # コルモゴロフ複雑性（近似）
            kolmogorov_features = self.calculate_kolmogorov_complexity_features_polars(data)
            features.update(kolmogorov_features)
            
            # 相互情報量（自己遅延・クロス）
            mi_features = self.calculate_mutual_information_features_polars(data)
            features.update(mi_features)
            
        except Exception as e:
            logger.error(f"情報理論特徴量計算エラー: {e}")
        
        return features
    
    def calculate_advanced_entropy_features_polars(self, data: np.ndarray) -> Dict[str, np.ndarray]:
        """高度なエントロピー特徴量（Polars+Numba版）"""
        features = {}
        
        for window_size in self.params['adv_entropy_windows']:
            if len(data) < window_size:
                continue
                
            # Numba最適化版エントロピー計算
            entropy_result = self._numba_safe_calculation(
                self._calculate_entropy_vectorized_optimized, data, window_size
            )
            
            if isinstance(entropy_result, np.ndarray) and entropy_result.shape[1] >= 6:
                entropy_names = ['shannon_entropy', 'renyi_entropy', 'tsallis_entropy', 
                               'conditional_entropy', 'effective_bins', 'max_probability']
                
                for j, name in enumerate(entropy_names):
                    features[f'{name}_{window_size}'] = entropy_result[:, j]
            else:
                # フォールバック値
                entropy_names = ['shannon_entropy', 'renyi_entropy', 'tsallis_entropy', 
                               'conditional_entropy', 'effective_bins', 'max_probability']
                for name in entropy_names:
                    features[f'{name}_{window_size}'] = np.zeros(len(data))
        
        return features
    
    @staticmethod
    @njit(parallel=True, cache=True)
    def _calculate_entropy_vectorized_optimized(data: np.ndarray, window_size: int) -> np.ndarray:
        """エントロピー計算（完全ベクトル化・並列最適化版）"""
        n = len(data)
        results = np.zeros((n, 6))
        
        if n < window_size:
            return results
        
        # 並列計算
        for i in prange(window_size-1, n):
            window = data[i-window_size+1:i+1]
            
            # 適応的ビン数（データの性質に応じて調整）
            unique_count = len(np.unique(window))
            n_bins = max(5, min(20, int(np.sqrt(unique_count))))
            
            # ヒストグラム計算（高精度版）
            min_val = np.min(window)
            max_val = np.max(window)
            
            if max_val - min_val < 1e-10:
                # すべての値が同じ場合
                results[i, 0] = 0.0  # Shannon entropy
                results[i, 1] = 0.0  # Renyi entropy
                results[i, 2] = 0.0  # Tsallis entropy
                results[i, 3] = 0.0  # Conditional entropy
                results[i, 4] = 1.0  # Effective bins
                results[i, 5] = 1.0  # Max probability
                continue
            
            bin_width = (max_val - min_val) / n_bins
            hist = np.zeros(n_bins)
            
            # ビン割り当て（精密版）
            for val in window:
                bin_idx = int((val - min_val) / bin_width)
                if bin_idx >= n_bins:
                    bin_idx = n_bins - 1
                if bin_idx < 0:
                    bin_idx = 0
                hist[bin_idx] += 1
            
            # 確率分布計算
            total_count = np.sum(hist)
            if total_count == 0:
                continue
                
            prob = hist / total_count
            prob_nonzero = prob[prob > 1e-10]
            
            if len(prob_nonzero) == 0:
                continue
            
            # 1. シャノンエントロピー
            shannon_entropy = -np.sum(prob_nonzero * np.log2(prob_nonzero))
            
            # 2. レニーエントロピー（q=2）
            sum_prob_squared = np.sum(prob_nonzero**2)
            renyi_entropy = -np.log2(sum_prob_squared) if sum_prob_squared > 0 else 0
            
            # 3. Tsallisエントロピー（q=2）
            tsallis_entropy = (1 - sum_prob_squared) / (2 - 1)
            
            # 4. 条件付きエントロピー（簡易版）
            conditional_entropy = 0.0
            if len(window) >= 4:
                # ペアワイズ遷移確率の計算
                transitions = {}
                for j in range(len(window) - 1):
                    current_bin = int((window[j] - min_val) / bin_width)
                    next_bin = int((window[j+1] - min_val) / bin_width)
                    
                    if current_bin >= n_bins:
                        current_bin = n_bins - 1
                    if next_bin >= n_bins:
                        next_bin = n_bins - 1
                    
                    key = current_bin * n_bins + next_bin
                    if key in transitions:
                        transitions[key] += 1
                    else:
                        transitions[key] = 1
                
                # 条件付き確率からエントロピー計算
                total_transitions = len(window) - 1
                if total_transitions > 0:
                    transition_entropy = 0.0
                    for count in transitions.values():
                        p = count / total_transitions
                        if p > 1e-10:
                            transition_entropy -= p * np.log2(p)
                    conditional_entropy = transition_entropy
            
            # 5. 有効ビン数
            effective_bins = len(prob_nonzero)
            
            # 6. 最大確率
            max_probability = np.max(prob_nonzero)
            
            results[i, 0] = shannon_entropy
            results[i, 1] = renyi_entropy
            results[i, 2] = tsallis_entropy
            results[i, 3] = conditional_entropy
            results[i, 4] = effective_bins
            results[i, 5] = max_probability
        
        return results
    
    def calculate_sample_approximate_entropy_polars(self, data: np.ndarray) -> Dict[str, np.ndarray]:
        """サンプルエントロピー・近似エントロピー特徴量（Polars最適化版）"""
        features = {}
        
        if not ENTROPY_AVAILABLE:
            logger.warning("entropyライブラリが利用できないため、サンプルエントロピー計算をスキップ")
            return features
        
        window_size = 50
        
        try:
            # Numba最適化版サンプルエントロピー計算
            entropy_result = self._numba_safe_calculation(
                self._calculate_sample_entropy_vectorized_optimized, data, window_size
            )
            
            if isinstance(entropy_result, np.ndarray) and entropy_result.shape[1] >= 3:
                features[f'sample_entropy_{window_size}'] = entropy_result[:, 0]
                features[f'approximate_entropy_{window_size}'] = entropy_result[:, 1]
                features[f'entropy_difference_{window_size}'] = entropy_result[:, 2]
            else:
                # フォールバック値
                features[f'sample_entropy_{window_size}'] = np.zeros(len(data))
                features[f'approximate_entropy_{window_size}'] = np.zeros(len(data))
                features[f'entropy_difference_{window_size}'] = np.zeros(len(data))
                
        except Exception as e:
            logger.warning(f"サンプル/近似エントロピー計算エラー: {e}")
            features[f'sample_entropy_{window_size}'] = np.zeros(len(data))
            features[f'approximate_entropy_{window_size}'] = np.zeros(len(data))
            features[f'entropy_difference_{window_size}'] = np.zeros(len(data))
        
        return features
    
    @staticmethod
    @njit(parallel=True, cache=True)
    def _calculate_sample_entropy_vectorized_optimized(data: np.ndarray, window_size: int) -> np.ndarray:
        """サンプルエントロピー・近似エントロピー計算（完全最適化版）"""
        n = len(data)
        results = np.zeros((n, 3))
        
        if n < window_size:
            return results
        
        # 並列計算
        for i in prange(window_size-1, n):
            window = data[i-window_size+1:i+1]
            
            if len(window) < 10:
                continue
            
            # パラメータ設定
            m = 2  # パターン長
            r = 0.2 * np.std(window)  # 許容誤差
            
            if r < 1e-10:
                continue
            
            # テンプレートマッチング（m長とm+1長）
            phi_m = 0
            phi_m_plus_1 = 0
            N = len(window)
            
            # m長パターンの計算
            for j in range(N - m):
                template_m = window[j:j+m]
                matches_m = 0
                matches_m_plus_1 = 0
                
                for k in range(N - m):
                    if j != k:
                        candidate_m = window[k:k+m]
                        
                        # m長パターンマッチング
                        match_m = True
                        for l in range(m):
                            if abs(template_m[l] - candidate_m[l]) > r:
                                match_m = False
                                break
                        
                        if match_m:
                            matches_m += 1
                            
                            # m+1長パターンマッチング
                            if j < N - m and k < N - m:
                                if abs(window[j + m] - window[k + m]) <= r:
                                    matches_m_plus_1 += 1
                
                phi_m += matches_m
                phi_m_plus_1 += matches_m_plus_1
            
            # エントロピー計算
            N_m = N - m
            if phi_m > 0 and phi_m_plus_1 > 0 and N_m > 0:
                phi_m_norm = phi_m / (N_m * N_m)
                phi_m_plus_1_norm = phi_m_plus_1 / (N_m * N_m)
                
                if phi_m_norm > 1e-10 and phi_m_plus_1_norm > 1e-10:
                    sample_entropy = -np.log(phi_m_plus_1_norm / phi_m_norm)
                    approximate_entropy = np.log(phi_m_norm) - np.log(phi_m_plus_1_norm)
                    entropy_difference = sample_entropy - approximate_entropy
                    
                    results[i, 0] = sample_entropy
                    results[i, 1] = approximate_entropy
                    results[i, 2] = entropy_difference
        
        return results
    
    def calculate_lempel_ziv_features_polars(self, data: np.ndarray) -> Dict[str, np.ndarray]:
        """Lempel-Ziv複雑性特徴量（Polars最適化版）"""
        features = {}
        
        for window_size in self.params['lz_windows']:
            if len(data) < window_size:
                continue
                
            # Numba最適化版LZ複雑性計算
            lz_result = self._numba_safe_calculation(
                self._calculate_lempel_ziv_vectorized_optimized, data, window_size
            )
            
            if isinstance(lz_result, np.ndarray):
                features[f'lempel_ziv_complexity_{window_size}'] = lz_result
                
                # 高複雑性検出（75パーセンタイルベース）
                lz_75th = self._numba_safe_calculation(
                    self._compute_rolling_percentile_numba, lz_result, window_size, 75
                )
                
                if isinstance(lz_75th, np.ndarray):
                    features[f'lz_high_complexity_{window_size}'] = (lz_result > lz_75th).astype(float)
                else:
                    features[f'lz_high_complexity_{window_size}'] = np.zeros(len(data))
            else:
                features[f'lempel_ziv_complexity_{window_size}'] = np.zeros(len(data))
                features[f'lz_high_complexity_{window_size}'] = np.zeros(len(data))
        
        return features
    
    @staticmethod
    @njit(parallel=True, cache=True)
    def _calculate_lempel_ziv_vectorized_optimized(data: np.ndarray, window_size: int) -> np.ndarray:
        """Lempel-Ziv複雑性計算（完全ベクトル化・並列最適化版）"""
        n = len(data)
        results = np.zeros(n)
        
        if n < window_size:
            return results
        
        # 並列計算
        for i in prange(window_size-1, n):
            window = data[i-window_size+1:i+1]
            
            # バイナリ化（中央値を閾値とする）
            threshold = np.median(window)
            binary_data = np.zeros(len(window), dtype=np.int32)
            for j in range(len(window)):
                binary_data[j] = 1 if window[j] > threshold else 0
            
            # Lempel-Ziv複雑性計算（改良版）
            complexity = 0
            pos = 0
            n_data = len(binary_data)
            
            while pos < n_data:
                max_match_len = 0
                
                # 最長一致検索
                for start in range(pos):
                    match_len = 0
                    
                    # パターンマッチング
                    while (pos + match_len < n_data and 
                           start + match_len < pos and 
                           binary_data[pos + match_len] == binary_data[start + match_len]):
                        match_len += 1
                    
                    max_match_len = max(max_match_len, match_len)
                
                # 新しいシンボルを追加
                pos += max(1, max_match_len)
                complexity += 1
            
            # 正規化（理論的最大値で割る）
            if n_data > 0:
                max_complexity = n_data / np.log2(n_data + 1)
                normalized_complexity = complexity / max_complexity
                results[i] = min(1.0, normalized_complexity)  # 上限を1に設定
        
        return results  
    
    def calculate_kolmogorov_complexity_features_polars(self, data: np.ndarray) -> Dict[str, np.ndarray]:
        """コルモゴロフ複雑性（近似）特徴量（Polars最適化版）"""
        features = {}
        
        for window_size in self.params['kolmogorov_windows']:
            if len(data) < window_size:
                continue
                
            # Numba最適化版コルモゴロフ複雑性計算
            kc_result = self._numba_safe_calculation(
                self._calculate_kolmogorov_complexity_vectorized_optimized, data, window_size
            )
            
            if isinstance(kc_result, np.ndarray):
                features[f'kolmogorov_complexity_{window_size}'] = kc_result
                
                # 高複雑性検出（80パーセンタイルベース）
                kc_80th = self._numba_safe_calculation(
                    self._compute_rolling_percentile_numba, kc_result, window_size, 80
                )
                
                if isinstance(kc_80th, np.ndarray):
                    features[f'kc_high_complexity_{window_size}'] = (kc_result > kc_80th).astype(float)
                else:
                    features[f'kc_high_complexity_{window_size}'] = np.zeros(len(data))
            else:
                features[f'kolmogorov_complexity_{window_size}'] = np.zeros(len(data))
                features[f'kc_high_complexity_{window_size}'] = np.zeros(len(data))
        
        return features
    
    @staticmethod
    @njit(parallel=True, cache=True)
    def _calculate_kolmogorov_complexity_vectorized_optimized(data: np.ndarray, window_size: int) -> np.ndarray:
        """コルモゴロフ複雑性（近似）計算（完全ベクトル化・並列最適化版）"""
        n = len(data)
        results = np.zeros(n)
        
        if n < window_size:
            return results
        
        # 並列計算
        for i in prange(window_size-1, n):
            window = data[i-window_size+1:i+1]
            
            # 複雑性の近似指標として複数の手法を組み合わせ
            
            # 1. 圧縮性（改良版ランレングス符号化）
            compressed_length = 1
            prev_val = window[0]
            std_threshold = np.std(window) * 0.1
            run_lengths = []
            current_run = 1
            
            for j in range(1, len(window)):
                if abs(window[j] - prev_val) <= std_threshold:
                    current_run += 1
                else:
                    run_lengths.append(current_run)
                    current_run = 1
                    compressed_length += 1
                prev_val = window[j]
            
            run_lengths.append(current_run)
            compression_ratio = compressed_length / len(window)
            
            # 2. パターンの反復性（改良版）
            pattern_complexity = 0.0
            max_pattern_length = min(10, len(window) // 4)
            
            if max_pattern_length >= 2:
                total_patterns = 0
                unique_patterns = 0
                
                # 複数のパターン長で解析
                for p_len in range(2, max_pattern_length + 1):
                    pattern_set = set()
                    
                    for j in range(len(window) - p_len + 1):
                        # パターンを数値化（ハッシュ値の改良版）
                        pattern_hash = 0
                        for k in range(p_len):
                            pattern_hash = pattern_hash * 31 + int(window[j + k] * 1000) % 10000
                        
                        pattern_set.add(pattern_hash % 100000)  # 衝突を減らすため大きな数で
                        total_patterns += 1
                    
                    unique_patterns += len(pattern_set)
                
                if total_patterns > 0:
                    pattern_complexity = unique_patterns / total_patterns
            
            # 3. エントロピーベースの複雑性（改良版）
            entropy_complexity = 0.0
            n_bins = max(5, int(np.sqrt(len(window))))
            min_val = np.min(window)
            max_val = np.max(window)
            
            if max_val - min_val > 1e-10:
                bin_width = (max_val - min_val) / n_bins
                hist = np.zeros(n_bins)
                
                # ヒストグラム作成
                for val in window:
                    bin_idx = int((val - min_val) / bin_width)
                    if bin_idx >= n_bins:
                        bin_idx = n_bins - 1
                    if bin_idx < 0:
                        bin_idx = 0
                    hist[bin_idx] += 1
                
                # エントロピー計算
                total_count = np.sum(hist)
                if total_count > 0:
                    for h in hist:
                        if h > 0:
                            p = h / total_count
                            entropy_complexity -= p * np.log2(p)
                    
                    # 正規化
                    max_entropy = np.log2(n_bins)
                    if max_entropy > 0:
                        entropy_complexity /= max_entropy
            
            # 4. 自己類似性の欠如（新規追加）
            self_similarity = 0.0
            if len(window) >= 8:
                half = len(window) // 2
                first_half = window[:half]
                second_half = window[half:half*2]
                
                # 相関係数の計算
                if len(first_half) == len(second_half) and len(first_half) > 1:
                    mean1 = np.mean(first_half)
                    mean2 = np.mean(second_half)
                    std1 = np.std(first_half)
                    std2 = np.std(second_half)
                    
                    if std1 > 1e-10 and std2 > 1e-10:
                        covariance = np.mean((first_half - mean1) * (second_half - mean2))
                        correlation = covariance / (std1 * std2)
                        self_similarity = 1.0 - abs(correlation)  # 相関が低いほど複雑
            
            # 5. 周波数複雑性（新規追加）
            frequency_complexity = 0.0
            if len(window) >= 8:
                # 簡易FFT風の周波数解析
                freq_energy = np.zeros(min(8, len(window) // 2))
                
                for freq_idx in range(len(freq_energy)):
                    freq = (freq_idx + 1) * 2 * np.pi / len(window)
                    real_part = 0.0
                    imag_part = 0.0
                    
                    for j in range(len(window)):
                        angle = freq * j
                        real_part += window[j] * np.cos(angle)
                        imag_part += window[j] * np.sin(angle)
                    
                    freq_energy[freq_idx] = real_part**2 + imag_part**2
                
                # 周波数エネルギーの分散（複雑性の指標）
                if np.sum(freq_energy) > 1e-10:
                    normalized_energy = freq_energy / np.sum(freq_energy)
                    # エントロピー計算
                    for energy in normalized_energy:
                        if energy > 1e-10:
                            frequency_complexity -= energy * np.log2(energy)
                    
                    # 正規化
                    max_freq_entropy = np.log2(len(freq_energy))
                    if max_freq_entropy > 0:
                        frequency_complexity /= max_freq_entropy
            
            # 複合複雑性スコア（重み付き平均）
            weights = np.array([0.2, 0.25, 0.25, 0.15, 0.15])  # 各成分の重み
            components = np.array([compression_ratio, pattern_complexity, entropy_complexity, 
                                 self_similarity, frequency_complexity])
            
            combined_complexity = np.sum(weights * components)
            results[i] = min(1.0, max(0.0, combined_complexity))  # 0-1に正規化
        
        return results
    
    def calculate_mutual_information_features_polars(self, data: np.ndarray, data2: np.ndarray = None) -> Dict[str, np.ndarray]:
        """相互情報量特徴量（Polars最適化版）"""
        features = {}
        
        # 自己遅延相互情報量
        for lag in self.params['mutual_info_lags']:
            if len(data) > lag + 50:  # 最小ウィンドウサイズを考慮
                mi_result = self._numba_safe_calculation(
                    self._calculate_mutual_information_vectorized_optimized, data, lag
                )
                
                if isinstance(mi_result, np.ndarray):
                    features[f'mutual_info_lag_{lag}'] = mi_result
                    
                    # 有意性検出（75パーセンタイルベース）
                    mi_75th = self._numba_safe_calculation(
                        self._compute_rolling_percentile_numba, mi_result, 50, 75
                    )
                    
                    if isinstance(mi_75th, np.ndarray):
                        features[f'mi_significant_lag_{lag}'] = (mi_result > mi_75th).astype(float)
                    else:
                        features[f'mi_significant_lag_{lag}'] = np.zeros(len(data))
                else:
                    features[f'mutual_info_lag_{lag}'] = np.zeros(len(data))
                    features[f'mi_significant_lag_{lag}'] = np.zeros(len(data))
        
        # クロス相互情報量（異なるデータ系列間）
        if data2 is not None and len(data2) == len(data):
            cross_mi_result = self._numba_safe_calculation(
                self._calculate_cross_mutual_information_vectorized, data, data2
            )
            
            if isinstance(cross_mi_result, np.ndarray):
                features['mutual_info_cross'] = cross_mi_result
                
                # クロス有意性検出
                cross_mi_75th = self._numba_safe_calculation(
                    self._compute_rolling_percentile_numba, cross_mi_result, 50, 75
                )
                
                if isinstance(cross_mi_75th, np.ndarray):
                    features['mi_cross_significant'] = (cross_mi_result > cross_mi_75th).astype(float)
                else:
                    features['mi_cross_significant'] = np.zeros(len(data))
            else:
                features['mutual_info_cross'] = np.zeros(len(data))
                features['mi_cross_significant'] = np.zeros(len(data))
        
        return features
    
    @staticmethod
    @njit(parallel=True, cache=True)
    def _calculate_mutual_information_vectorized_optimized(data: np.ndarray, lag: int) -> np.ndarray:
        """相互情報量計算（完全ベクトル化・並列最適化版）"""
        n = len(data)
        window_size = 100
        results = np.zeros(n)
        
        if n < window_size + lag:
            return results
        
        # 並列計算
        for i in prange(window_size-1, n-lag):
            x = data[i-window_size+1:i+1]
            y = data[i-window_size+1-lag:i+1-lag]
            
            if len(x) != len(y) or len(x) < 10:
                continue
            
            # 適応的ビン数
            unique_x = len(np.unique(x))
            unique_y = len(np.unique(y))
            n_bins = max(5, min(15, int(np.sqrt(min(unique_x, unique_y)))))
            
            # X軸のビン化（改良版）
            x_min, x_max = np.min(x), np.max(x)
            if x_max - x_min < 1e-10:
                continue
                
            x_bin_width = (x_max - x_min) / n_bins
            x_binned = np.zeros(len(x), dtype=np.int32)
            for j in range(len(x)):
                bin_idx = int((x[j] - x_min) / x_bin_width)
                if bin_idx >= n_bins:
                    bin_idx = n_bins - 1
                x_binned[j] = bin_idx
            
            # Y軸のビン化（改良版）
            y_min, y_max = np.min(y), np.max(y)
            if y_max - y_min < 1e-10:
                continue
                
            y_bin_width = (y_max - y_min) / n_bins
            y_binned = np.zeros(len(y), dtype=np.int32)
            for j in range(len(y)):
                bin_idx = int((y[j] - y_min) / y_bin_width)
                if bin_idx >= n_bins:
                    bin_idx = n_bins - 1
                y_binned[j] = bin_idx
            
            # ヒストグラム計算（高精度版）
            x_hist = np.zeros(n_bins)
            y_hist = np.zeros(n_bins)
            xy_hist = np.zeros((n_bins, n_bins))
            
            for j in range(len(x_binned)):
                x_bin = x_binned[j]
                y_bin = y_binned[j]
                x_hist[x_bin] += 1
                y_hist[y_bin] += 1
                xy_hist[x_bin, y_bin] += 1
            
            # 確率分布計算
            total_count = len(x)
            px = x_hist / total_count
            py = y_hist / total_count
            pxy = xy_hist / total_count
            
            # 相互情報量計算（数値安定化版）
            mi = 0.0
            for j in range(n_bins):
                for k in range(n_bins):
                    if pxy[j, k] > 1e-10 and px[j] > 1e-10 and py[k] > 1e-10:
                        mi_term = pxy[j, k] * np.log2(pxy[j, k] / (px[j] * py[k]))
                        if np.isfinite(mi_term):
                            mi += mi_term
            
            results[i] = max(0.0, mi)  # 負の値を0にクランプ
        
        # ラグ調整
        results_adjusted = np.zeros(n)
        if lag > 0:
            results_adjusted[lag:] = results[:-lag]
        else:
            results_adjusted = results
        
        return results_adjusted
    
    @staticmethod
    @njit(parallel=True, cache=True)
    def _calculate_cross_mutual_information_vectorized(data1: np.ndarray, data2: np.ndarray) -> np.ndarray:
        """クロス相互情報量計算（完全ベクトル化・並列版）"""
        n = min(len(data1), len(data2))
        window_size = 100
        results = np.zeros(n)
        
        if n < window_size:
            return results
        
        # 並列計算
        for i in prange(window_size-1, n):
            x = data1[i-window_size+1:i+1]
            y = data2[i-window_size+1:i+1]
            
            if len(x) != len(y) or len(x) < 10:
                continue
            
            # 適応的ビン数
            unique_x = len(np.unique(x))
            unique_y = len(np.unique(y))
            n_bins = max(5, min(15, int(np.sqrt(min(unique_x, unique_y)))))
            
            # X軸のビン化
            x_min, x_max = np.min(x), np.max(x)
            if x_max - x_min < 1e-10:
                continue
                
            x_bin_width = (x_max - x_min) / n_bins
            x_binned = np.zeros(len(x), dtype=np.int32)
            for j in range(len(x)):
                bin_idx = int((x[j] - x_min) / x_bin_width)
                if bin_idx >= n_bins:
                    bin_idx = n_bins - 1
                x_binned[j] = bin_idx
            
            # Y軸のビン化
            y_min, y_max = np.min(y), np.max(y)
            if y_max - y_min < 1e-10:
                continue
                
            y_bin_width = (y_max - y_min) / n_bins
            y_binned = np.zeros(len(y), dtype=np.int32)
            for j in range(len(y)):
                bin_idx = int((y[j] - y_min) / y_bin_width)
                if bin_idx >= n_bins:
                    bin_idx = n_bins - 1
                y_binned[j] = bin_idx
            
            # ヒストグラム計算
            x_hist = np.zeros(n_bins)
            y_hist = np.zeros(n_bins)
            xy_hist = np.zeros((n_bins, n_bins))
            
            for j in range(len(x_binned)):
                x_bin = x_binned[j]
                y_bin = y_binned[j]
                x_hist[x_bin] += 1
                y_hist[y_bin] += 1
                xy_hist[x_bin, y_bin] += 1
            
            # 確率分布計算
            total_count = len(x)
            px = x_hist / total_count
            py = y_hist / total_count
            pxy = xy_hist / total_count
            
            # 相互情報量計算
            mi = 0.0
            for j in range(n_bins):
                for k in range(n_bins):
                    if pxy[j, k] > 1e-10 and px[j] > 1e-10 and py[k] > 1e-10:
                        mi_term = pxy[j, k] * np.log2(pxy[j, k] / (px[j] * py[k]))
                        if np.isfinite(mi_term):
                            mi += mi_term
            
            results[i] = max(0.0, mi)
        
        return results
    
    # =========================================================================
    # 学際的アナロジー特徴量（統合・最適化版）
    # =========================================================================
    
    def calculate_interdisciplinary_features_polars(self, data: np.ndarray) -> Dict[str, np.ndarray]:
        """学際的アナロジー特徴量の統合計算（Polars+Numba最適化版）"""
        features = {}
        
        try:
            # ゲーム理論特徴量
            game_theory_features = self.calculate_game_theory_features_polars(data)
            features.update(game_theory_features)
            
            # 分子科学特徴量
            molecular_features = self.calculate_molecular_science_features_polars(data)
            features.update(molecular_features)
            
            # ネットワーク科学特徴量
            network_features = self.calculate_network_science_features_polars(data)
            features.update(network_features)
            
            # 音響学特徴量
            acoustics_features = self.calculate_acoustics_features_polars(data)
            features.update(acoustics_features)
            
            # 言語学特徴量
            linguistics_features = self.calculate_linguistics_features_polars(data)
            features.update(linguistics_features)
            
            # 美学特徴量
            aesthetics_features = self.calculate_aesthetics_features_polars(data)
            features.update(aesthetics_features)
            
            # 音楽理論特徴量
            music_features = self.calculate_music_theory_features_polars(data)
            features.update(music_features)
            
            # 天文学特徴量
            astronomy_features = self.calculate_astronomy_features_polars(data)
            features.update(astronomy_features)
            
            # 生体力学特徴量
            biomechanics_features = self.calculate_biomechanics_features_polars(data)
            features.update(biomechanics_features)
            
        except Exception as e:
            logger.error(f"学際的アナロジー特徴量計算エラー: {e}")
        
        return features
    
    def calculate_game_theory_features_polars(self, data: np.ndarray) -> Dict[str, np.ndarray]:
        """ゲーム理論特徴量（Polars最適化版）"""
        features = {}
        window_size = 50
        
        # Numba最適化版ゲーム理論計算
        game_result = self._numba_safe_calculation(
            self._calculate_game_theory_vectorized_optimized, data, window_size
        )
        
        if isinstance(game_result, np.ndarray) and game_result.shape[1] >= 6:
            game_names = ['nash_equilibrium', 'cooperation_index', 'strategy_diversity',
                         'zero_sum_indicator', 'prisoners_dilemma', 'minimax_strategy']
            
            for j, name in enumerate(game_names):
                features[f'{name}_{window_size}'] = game_result[:, j]
        else:
            # フォールバック値
            game_names = ['nash_equilibrium', 'cooperation_index', 'strategy_diversity',
                         'zero_sum_indicator', 'prisoners_dilemma', 'minimax_strategy']
            for name in game_names:
                features[f'{name}_{window_size}'] = np.zeros(len(data))
        
        return features
    
    @staticmethod
    @njit(parallel=True, cache=True)
    def _calculate_game_theory_vectorized_optimized(data: np.ndarray, window_size: int) -> np.ndarray:
        """ゲーム理論特徴量計算（完全ベクトル化・並列最適化版）"""
        n = len(data)
        results = np.zeros((n, 6))
        
        if n < window_size:
            return results
        
        # 並列計算
        for i in prange(window_size-1, n):
            window = data[i-window_size+1:i+1]
            returns = np.diff(window)
            
            if len(returns) < 2:
                continue
            
            # 1. ナッシュ均衡（安定性指標）- 改良版
            return_mean = np.mean(returns)
            return_std = np.std(returns)
            return_var = np.var(returns)
            
            # 安定性 vs 方向性のバランス
            if abs(return_mean) > 1e-10:
                nash_equilibrium = return_std / abs(return_mean)
            else:
                nash_equilibrium = return_std * 1000  # 方向性がない場合の高い不安定性
            
            # 正規化（0-1範囲）
            nash_equilibrium = min(1.0, nash_equilibrium / 10.0)
            
            # 2. 協力指数（連続リターンの相関）- 改良版
            cooperation_index = 0.0
            if len(returns) >= 3:
                returns_1 = returns[1:]
                returns_2 = returns[:-1]
                
                if len(returns_1) > 0 and len(returns_2) > 0:
                    mean_1 = np.mean(returns_1)
                    mean_2 = np.mean(returns_2)
                    std_1 = np.std(returns_1)
                    std_2 = np.std(returns_2)
                    
                    if std_1 > 1e-10 and std_2 > 1e-10:
                        covariance = np.mean((returns_1 - mean_1) * (returns_2 - mean_2))
                        cooperation_index = covariance / (std_1 * std_2)
                        
                        # -1から1の範囲を0から1に変換
                        cooperation_index = (cooperation_index + 1.0) / 2.0
            
            # 3. 戦略多様性（リターンの符号パターンの多様性）- 改良版
            return_signs = np.sign(returns)
            
            # 符号パターンの分析
            positive_count = np.sum(return_signs > 0)
            negative_count = np.sum(return_signs < 0)
            zero_count = np.sum(return_signs == 0)
            total_count = len(return_signs)
            
            if total_count > 0:
                positive_ratio = positive_count / total_count
                negative_ratio = negative_count / total_count
                zero_ratio = zero_count / total_count
                
                # シャノンエントロピーベースの多様性計算
                strategy_diversity = 0.0
                for ratio in [positive_ratio, negative_ratio, zero_ratio]:
                    if ratio > 1e-10:
                        strategy_diversity -= ratio * np.log2(ratio)
                
                # 最大エントロピーで正規化（3戦略の場合）
                max_entropy = np.log2(3)
                if max_entropy > 0:
                    strategy_diversity /= max_entropy
            else:
                strategy_diversity = 0.0
            
            # 4. ゼロサム指標（改良版）
            total_return = np.sum(returns)
            total_abs_return = np.sum(np.abs(returns))
            
            if total_abs_return > 1e-10:
                zero_sum_indicator = 1.0 - abs(total_return) / total_abs_return
            else:
                zero_sum_indicator = 1.0  # 変動がない場合は完全ゼロサム
            
            # 5. 囚人のジレンマ（裏切りvs協調の利得分析）- 改良版
            defection_payoff = 0.0
            cooperation_payoff = 0.0
            interaction_count = 0
            
            for j in range(len(returns) - 1):
                current_return = returns[j]
                next_return = returns[j+1]
                
                # 相互作用の分類
                if current_return * next_return < 0:  # 逆方向（裏切り）
                    defection_payoff += abs(next_return)
                elif abs(current_return) > 1e-10 and abs(next_return) > 1e-10:  # 同方向（協調）
                    cooperation_payoff += abs(next_return)
                
                interaction_count += 1
            
            total_payoff = defection_payoff + cooperation_payoff
            if total_payoff > 1e-10:
                prisoners_dilemma = defection_payoff / total_payoff
            else:
                prisoners_dilemma = 0.5  # 中性値
            
            # 6. ミニマックス戦略（損失最小化戦略の評価）- 改良版
            if len(returns) > 0:
                max_loss = np.min(returns)  # 最大損失
                max_gain = np.max(returns)  # 最大利得
                median_return = np.median(returns)
                
                # リスク調整後の評価
                total_range = max_gain - max_loss
                if total_range > 1e-10:
                    # 損失の相対的な大きさ
                    loss_magnitude = abs(max_loss) / total_range
                    # 中央値の位置（リスクバランス）
                    median_position = (median_return - max_loss) / total_range
                    
                    # ミニマックス戦略スコア（損失を重視）
                    minimax_strategy = loss_magnitude * (1.0 - median_position)
                else:
                    minimax_strategy = 0.0
            else:
                minimax_strategy = 0.0
            
            # 結果の格納
            results[i, 0] = nash_equilibrium
            results[i, 1] = cooperation_index
            results[i, 2] = strategy_diversity
            results[i, 3] = zero_sum_indicator
            results[i, 4] = prisoners_dilemma
            results[i, 5] = minimax_strategy
        
        return results
    
    def calculate_molecular_science_features_polars(self, data: np.ndarray) -> Dict[str, np.ndarray]:
        """分子科学特徴量（Polars最適化版）"""
        features = {}
        window_size = 50
        
        # Numba最適化版分子科学計算
        molecular_result = self._numba_safe_calculation(
            self._calculate_molecular_science_vectorized_optimized, data, window_size
        )
        
        if isinstance(molecular_result, np.ndarray) and molecular_result.shape[1] >= 6:
            molecular_names = ['molecular_vibration', 'bond_energy', 'electron_density',
                             'molecular_orbital', 'chemical_potential', 'intermolecular_force']
            
            for j, name in enumerate(molecular_names):
                features[f'{name}_{window_size}'] = molecular_result[:, j]
        else:
            # フォールバック値
            molecular_names = ['molecular_vibration', 'bond_energy', 'electron_density',
                             'molecular_orbital', 'chemical_potential', 'intermolecular_force']
            for name in molecular_names:
                features[f'{name}_{window_size}'] = np.zeros(len(data))
        
        return features
    
    @staticmethod
    @njit(parallel=True, cache=True)
    def _calculate_molecular_science_vectorized_optimized(data: np.ndarray, window_size: int) -> np.ndarray:
        """分子科学特徴量計算（完全ベクトル化・並列最適化版）"""
        n = len(data)
        results = np.zeros((n, 6))
        
        if n < window_size:
            return results
        
        # 並列計算
        for i in prange(window_size-1, n):
            window = data[i-window_size+1:i+1]
            
            # 1. 分子振動（高周波成分の強さ）- 改良版
            molecular_vibration = 0.0
            if len(window) >= 8:
                # 簡易FFT風の高周波解析
                high_freq_energy = 0.0
                total_energy = 0.0
                
                # 複数の高周波成分を解析
                for freq_multiplier in range(2, min(8, len(window) // 2)):
                    freq = freq_multiplier * 2 * np.pi / len(window)
                    real_part = 0.0
                    imag_part = 0.0
                    
                    for j in range(len(window)):
                        angle = freq * j
                        real_part += window[j] * np.cos(angle)
                        imag_part += window[j] * np.sin(angle)
                    
                    energy = real_part**2 + imag_part**2
                    if freq_multiplier >= len(window) // 4:  # 高周波成分
                        high_freq_energy += energy
                    total_energy += energy
                
                if total_energy > 1e-10:
                    molecular_vibration = high_freq_energy / total_energy
            
            # 2. 結合エネルギー（エントロピー様）- 改良版
            bond_energy = 0.0
            if len(window) > 3:
                # 適応的ビン数
                unique_vals = len(np.unique(window))
                n_bins = max(3, min(15, int(np.sqrt(unique_vals))))
                
                min_val = np.min(window)
                max_val = np.max(window)
                
                if max_val - min_val > 1e-10:
                    bin_width = (max_val - min_val) / n_bins
                    hist = np.zeros(n_bins)
                    
                    for val in window:
                        bin_idx = int((val - min_val) / bin_width)
                        if bin_idx >= n_bins:
                            bin_idx = n_bins - 1
                        hist[bin_idx] += 1
                    
                    total_count = np.sum(hist)
                    if total_count > 0:
                        # シャノンエントロピー計算
                        for h in hist:
                            if h > 0:
                                p = h / total_count
                                bond_energy -= p * np.log(p)
                        
                        # 正規化
                        max_entropy = np.log(n_bins)
                        if max_entropy > 0:
                            bond_energy /= max_entropy
            
            # 3. 電子密度（価格の二乗和）- 改良版
            electron_density = np.sum(window**2) / len(window)
            
            # 正規化（ウィンドウ内の標準偏差で）
            window_std = np.std(window)
            if window_std > 1e-10:
                electron_density = electron_density / (window_std**2)
            
            # 4. 分子軌道（時間加重エネルギー）- 改良版
            molecular_orbital = 0.0
            total_weight = 0.0
            
            for j in range(len(window)):
                # 時間重み（より最近の値により高い重み）
                time_weight = (j + 1) / len(window)
                energy_contribution = time_weight * window[j]**2
                molecular_orbital += energy_contribution
                total_weight += time_weight
            
            if total_weight > 1e-10:
                molecular_orbital /= total_weight
            
            # 5. 化学ポテンシャル（平均変化率）- 改良版
            chemical_potential = 0.0
            if len(window) > 1:
                differences = np.diff(window)
                
                if len(differences) > 0:
                    # 重み付き平均変化率（最近の変化により高い重み）
                    weighted_sum = 0.0
                    weight_sum = 0.0
                    
                    for j in range(len(differences)):
                        weight = (j + 1) / len(differences)
                        weighted_sum += weight * differences[j]
                        weight_sum += weight
                    
                    if weight_sum > 1e-10:
                        chemical_potential = weighted_sum / weight_sum
            
            # 6. 分子間力（低ボラティリティの度合い）- 改良版
            volatility = np.std(window)
            mean_val = np.mean(window)
            
            # 相対ボラティリティ
            if abs(mean_val) > 1e-10:
                relative_volatility = volatility / abs(mean_val)
            else:
                relative_volatility = volatility
            
            # 分子間力（ボラティリティが低いほど強い）
            intermolecular_force = 1.0 / (1.0 + relative_volatility)
            
            # 安定性ボーナス（変動の一貫性）
            if len(window) > 2:
                second_differences = np.abs(np.diff(window, n=2))
                stability_factor = 1.0 / (1.0 + np.mean(second_differences))
                intermolecular_force *= (1.0 + stability_factor)
            
            # 結果の格納
            results[i, 0] = molecular_vibration
            results[i, 1] = bond_energy
            results[i, 2] = electron_density
            results[i, 3] = molecular_orbital
            results[i, 4] = chemical_potential
            results[i, 5] = intermolecular_force
        
        return results
    
    def calculate_network_science_features_polars(self, data: np.ndarray) -> Dict[str, np.ndarray]:
        """ネットワーク科学特徴量（Polars最適化版）"""
        features = {}
        window_size = 50
        
        # Numba最適化版ネットワーク科学計算
        network_result = self._numba_safe_calculation(
            self._calculate_network_science_vectorized_optimized, data, window_size
        )
        
        if isinstance(network_result, np.ndarray) and network_result.shape[1] >= 5:
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
            
            # 1. ネットワーク密度（ユニークな価格状態の相対頻度）- 改良版
            # 価格レベルを離散化
            min_val = np.min(window)
            max_val = np.max(window)
            
            network_density = 0.0
            if max_val - min_val > 1e-10:
                # 適応的離散化レベル
                n_levels = max(5, min(20, int(np.sqrt(len(window)))))
                level_width = (max_val - min_val) / n_levels
                
                levels = np.zeros(len(window), dtype=np.int32)
                for j in range(len(window)):
                    level_idx = int((window[j] - min_val) / level_width)
                    if level_idx >= n_levels:
                        level_idx = n_levels - 1
                    levels[j] = level_idx
                
                # ユニークレベル数と接続性
                unique_levels = len(np.unique(levels))
                max_possible_connections = n_levels * (n_levels - 1) // 2
                
                # 実際の接続数（隣接レベル間の遷移）
                actual_connections = 0
                for j in range(len(levels) - 1):
                    if levels[j] != levels[j+1]:
                        actual_connections += 1
                
                if max_possible_connections > 0:
                    network_density = actual_connections / max_possible_connections
            
            # 2. 中心性（中央値と平均の関係）- 改良版
            median_val = np.median(window)
            mean_val = np.mean(window)
            std_val = np.std(window)
            
            centrality_measure = 0.0
            if std_val > 1e-10:
                # 分布の中心性を測定
                skewness_approx = (mean_val - median_val) / std_val
                # 対称性が高いほど中心性が高い
                centrality_measure = 1.0 / (1.0 + abs(skewness_approx))
            else:
                centrality_measure = 1.0  # 完全に均一な場合は最高の中心性
            
            # 3. クラスタリング係数（異なる時間軸の相関性）- 改良版
            clustering_coefficient = 0.0
            if len(window) >= 9:
                # 複数の時間軸でクラスター分析
                segment_size = len(window) // 3
                segments = []
                
                for seg_idx in range(3):
                    start_idx = seg_idx * segment_size
                    end_idx = start_idx + segment_size
                    if end_idx <= len(window):
                        segments.append(window[start_idx:end_idx])
                
                if len(segments) == 3:
                    # セグメント間の相関を計算
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
                                    # 最小長に合わせる
                                    min_len = min(len(seg1), len(seg2))
                                    seg1_trim = seg1[:min_len]
                                    seg2_trim = seg2[:min_len]
                                    
                                    covariance = np.mean((seg1_trim - mean1) * (seg2_trim - mean2))
                                    correlation = covariance / (std1 * std2)
                                    
                                    if np.isfinite(correlation):
                                        correlations.append(abs(correlation))
                    
                    if len(correlations) > 0:
                        clustering_coefficient = np.mean(correlations)
            
            # 4. 媒介中心性（中心期間の極値存在度）- 改良版
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
                    
                    # 中心領域の極値重要度
                    max_importance = 1.0 if abs(center_max - max_val) < 1e-10 else 0.0
                    min_importance = 1.0 if abs(center_min - min_val) < 1e-10 else 0.0
                    
                    # 中心領域の値の分散（情報伝達能力）
                    center_std = np.std(center_region)
                    window_std = np.std(window)
                    
                    if window_std > 1e-10:
                        variability_ratio = center_std / window_std
                    else:
                        variability_ratio = 0.0
                    
                    betweenness_centrality = (max_importance + min_importance) * 0.5 + variability_ratio * 0.5
            
            # 5. 固有ベクトル中心性（自己相関の累積）- 改良版
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
                                    # ラグが小さいほど重要度が高い
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
    
    def calculate_acoustics_features_polars(self, data: np.ndarray) -> Dict[str, np.ndarray]:
        """音響学特徴量（Polars最適化版）"""
        features = {}
        window_size = 64  # 音響学に適したウィンドウサイズ
        
        # Numba最適化版音響学計算
        acoustics_result = self._numba_safe_calculation(
            self._calculate_acoustics_vectorized_optimized, data, window_size
        )
        
        if isinstance(acoustics_result, np.ndarray) and acoustics_result.shape[1] >= 5:
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
                
                # 1. 音響パワー（リターンの二乗和）- 改良版
                acoustic_power = 0.0
                if len(returns) > 0:
                    # RMS（Root Mean Square）パワー
                    acoustic_power = np.sqrt(np.mean(returns**2))
                    
                    # 正規化（ウィンドウ標準偏差で）
                    window_std = np.std(window)
                    if window_std > 1e-10:
                        acoustic_power = acoustic_power / window_std
                
                # 2. 音響周波数（FFTによる主要周波数）- 改良版
                acoustic_frequency = 0.0
                if len(window) >= 8:
                    # 複数周波数成分の解析
                    max_energy = 0.0
                    dominant_freq_idx = 0
                    
                    # 基本周波数から高調波まで解析
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
                    
                    # 正規化された主要周波数
                    if len(window) > 2:
                        acoustic_frequency = dominant_freq_idx / (len(window) // 2)
                
                # 3. 振幅変調（ボラティリティの変動）- 改良版
                amplitude_modulation = 0.0
                if len(returns) >= 4:
                    # ローリングボラティリティの計算
                    rolling_vol = []
                    vol_window = min(4, len(returns) // 2)
                    
                    for j in range(vol_window - 1, len(returns)):
                        vol_segment = returns[j - vol_window + 1:j + 1]
                        rolling_vol.append(np.std(vol_segment))
                    
                    if len(rolling_vol) > 1:
                        # ボラティリティの変動率
                        vol_changes = np.diff(rolling_vol)
                        amplitude_modulation = np.std(vol_changes)
                        
                        # 正規化
                        mean_vol = np.mean(rolling_vol)
                        if mean_vol > 1e-10:
                            amplitude_modulation = amplitude_modulation / mean_vol
                
                # 4. 位相変調（FFT位相の安定性）- 改良版
                phase_modulation = 0.0
                if len(window) >= 8:
                    # 複数周波数の位相安定性を解析
                    phase_variations = []
                    
                    for freq_idx in range(1, min(len(window) // 4, 8)):
                        freq = freq_idx * 2 * np.pi / len(window)
                        phases = []
                        
                        # ウィンドウを小さなセグメントに分割して位相を追跡
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
                                # 位相差の分散
                                phase_diffs = []
                                for j in range(1, len(phases)):
                                    diff = phases[j] - phases[j-1]
                                    # 位相ラッピングを考慮
                                    while diff > np.pi:
                                        diff -= 2 * np.pi
                                    while diff < -np.pi:
                                        diff += 2 * np.pi
                                    phase_diffs.append(diff)
                                
                                if len(phase_diffs) > 0:
                                    phase_var = np.var(phase_diffs)
                                    phase_variations.append(phase_var)
                    
                    if len(phase_variations) > 0:
                        # 平均位相変動
                        avg_phase_var = np.mean(phase_variations)
                        # 位相変調度（変動が大きいほど高い）
                        phase_modulation = min(1.0, avg_phase_var / (np.pi**2))
                
                # 5. 音響エコー（前半と後半の相関）- 改良版
                acoustic_echo = 0.0
                if len(window) >= 8:
                    # ウィンドウを複数の部分に分割してエコー効果を検出
                    quarter = len(window) // 4
                    half = len(window) // 2
                    
                    if quarter > 0:
                        # 複数のエコー遅延を検証
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
                            
                            # 長さを揃える
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
    
    def calculate_linguistics_features_polars(self, data: np.ndarray) -> Dict[str, np.ndarray]:
        """言語学特徴量（Polars最適化版）"""
        features = {}
        window_size = 50
        
        # Numba最適化版言語学計算
        linguistics_result = self._numba_safe_calculation(
            self._calculate_linguistics_vectorized_optimized, data, window_size
        )
        
        if isinstance(linguistics_result, np.ndarray) and linguistics_result.shape[1] >= 5:
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
                # 価格が一定の場合は最小の多様性
                results[i, 0] = 0.0  # vocabulary_diversity
                results[i, 1] = 1.0  # sentence_structure (完全に平坦)
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
            
            # 1. 語彙の多様性（ユニークなパターンの多様性）- 改良版
            unique_words = len(np.unique(word_levels))
            vocabulary_diversity = unique_words / n_levels
            
            # 語彙の分布均一性も考慮
            if unique_words > 1:
                # 各語彙の出現頻度
                word_counts = np.zeros(n_levels)
                for level in word_levels:
                    word_counts[level] += 1
                
                # 非ゼロ要素のみで均一性を計算
                non_zero_counts = word_counts[word_counts > 0]
                if len(non_zero_counts) > 1:
                    # シャノンエントロピーによる均一性測定
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
            
            # 2. 文の構造（変化の滑らかさ）- 改良版
            sentence_structure = 0.0
            if len(window) >= 3:
                # 2次差分による滑らかさ測定
                first_diff = np.diff(window)
                second_diff = np.diff(first_diff)
                
                if len(second_diff) > 0:
                    # 急激な変化の少なさを測定
                    second_diff_std = np.std(second_diff)
                    first_diff_std = np.std(first_diff)
                    
                    if first_diff_std > 1e-10:
                        # 相対的な滑らかさ
                        smoothness = 1.0 / (1.0 + second_diff_std / first_diff_std)
                        sentence_structure = smoothness
                    else:
                        sentence_structure = 1.0  # 完全に平坦な場合
            
            # 3. 言語的複雑性（大きな変化とパターンの複雑さ）- 改良版
            linguistic_complexity = 0.0
            if len(window) > 1:
                returns = np.diff(window)
                
                # 標準偏差ベースの閾値
                threshold = np.std(returns) * 1.5
                large_changes = np.abs(returns) > threshold
                
                # 大きな変化の頻度
                change_frequency = np.sum(large_changes) / len(returns)
                
                # パターンの複雑さ（n-gramの多様性）
                pattern_complexity = 0.0
                max_pattern_len = min(5, len(word_levels) // 3)
                
                if max_pattern_len >= 2:
                    total_patterns = 0
                    unique_patterns = set()
                    
                    for pattern_len in range(2, max_pattern_len + 1):
                        for j in range(len(word_levels) - pattern_len + 1):
                            # パターンをハッシュ値として表現
                            pattern_hash = 0
                            for k in range(pattern_len):
                                pattern_hash = pattern_hash * n_levels + word_levels[j + k]
                            
                            unique_patterns.add(pattern_hash)
                            total_patterns += 1
                    
                    if total_patterns > 0:
                        pattern_complexity = len(unique_patterns) / total_patterns
                
                # 複雑性の統合
                linguistic_complexity = (change_frequency + pattern_complexity) / 2.0
            
            # 4. 語順（価格順位の変化パターン）- 改良版
            word_order = 0.0
            if len(window) >= 4:
                # 各価格の相対順位を計算
                ranks = np.zeros(len(window))
                for j in range(len(window)):
                    rank = 0
                    for k in range(len(window)):
                        if window[k] <= window[j]:
                            rank += 1
                    ranks[j] = rank / len(window)
                
                # 順位変化の分析
                rank_changes = np.abs(np.diff(ranks))
                
                if len(rank_changes) > 0:
                    # 順位変化の標準偏差（語順の変動性）
                    rank_volatility = np.std(rank_changes)
                    
                    # 順位の方向一貫性
                    rank_direction_changes = 0
                    for j in range(1, len(rank_changes)):
                        if rank_changes[j] * rank_changes[j-1] < 0:
                            rank_direction_changes += 1
                    
                    direction_consistency = 1.0 - (rank_direction_changes / max(1, len(rank_changes) - 1))
                    
                    # 語順スコア（変動性と一貫性の組み合わせ）
                    word_order = rank_volatility * direction_consistency
            
            # 5. プロソディ（抑揚・リズム）- 改良版
            prosody = 0.0
            if len(window) > 1:
                returns = np.diff(window)
                
                # 振幅の変動（抑揚）
                amplitude_variation = np.std(np.abs(returns))
                
                # リズムパターンの検出
                rhythm_score = 0.0
                if len(returns) >= 4:
                    # 周期性の検出（簡易自己相関）
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
                
                # 正規化（ウィンドウ標準偏差で）
                window_std = np.std(window)
                if window_std > 1e-10:
                    normalized_amplitude = amplitude_variation / window_std
                else:
                    normalized_amplitude = 0.0
                
                # プロソディスコア（抑揚とリズムの組み合わせ）
                prosody = (normalized_amplitude + rhythm_score) / 2.0
            
            # 結果の格納
            results[i, 0] = vocabulary_diversity
            results[i, 1] = sentence_structure
            results[i, 2] = linguistic_complexity
            results[i, 3] = word_order
            results[i, 4] = prosody
        
        return results
    
    def calculate_aesthetics_features_polars(self, data: np.ndarray) -> Dict[str, np.ndarray]:
        """美学特徴量（Polars最適化版）"""
        features = {}
        window_size = 50
        
        # Numba最適化版美学計算
        aesthetics_result = self._numba_safe_calculation(
            self._calculate_aesthetics_vectorized_optimized, data, window_size
        )
        
        if isinstance(aesthetics_result, np.ndarray) and aesthetics_result.shape[1] >= 5:
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
            
            # 1. 黄金比への固着度（改良版）
            golden_ratio_adherence = 0.0
            if len(window) >= 3:
                # 複数のスケールで黄金比を検証
                golden_deviations = []
                
                # 隣接比率
                for j in range(1, len(window)):
                    if abs(window[j-1]) > 1e-10:
                        ratio = abs(window[j] / window[j-1])
                        # 異常に大きな比率を除外
                        if 0.1 <= ratio <= 10.0:
                            deviation = abs(ratio - golden_ratio)
                            golden_deviations.append(deviation)
                
                # セグメント比率（長期vs短期）
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
            
            # 2. 対称性（前半と後半の反転相似性）- 改良版
            symmetry_measure = 0.0
            if len(window) >= 4:
                # 複数の対称性軸を検証
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
                
                # 3分割対称（黄金比分割）
                if len(window) >= 6:
                    golden_split = int(len(window) / golden_ratio)
                    if golden_split > 0 and golden_split < len(window) - 1:
                        left_segment = window[:golden_split]
                        right_segment = window[golden_split:]
                        right_segment_reversed = right_segment[::-1]
                        
                        # 長さを合わせる
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
            
            # 3. 美的調和（滑らかさ・躍度の小ささ）- 改良版
            aesthetic_harmony = 0.0
            if len(window) >= 4:
                try:
                    # 複数次数の微分による滑らかさ測定
                    first_diff = np.diff(window)
                    smoothness_scores = []
                    
                    if len(first_diff) >= 2:
                        second_diff = np.diff(first_diff)
                        
                        # 2次微分の滑らかさ
                        if len(second_diff) > 0:
                            second_diff_std = np.std(second_diff)
                            first_diff_std = np.std(first_diff)
                            
                            if first_diff_std > 1e-10:
                                smoothness_2nd = 1.0 / (1.0 + second_diff_std / first_diff_std)
                                smoothness_scores.append(smoothness_2nd)
                        
                        # 3次微分（躍度）の滑らかさ
                        if len(second_diff) >= 1:
                            third_diff = np.diff(second_diff)
                            
                            if len(third_diff) > 0:
                                third_diff_std = np.std(third_diff)
                                
                                if second_diff_std > 1e-10:
                                    smoothness_3rd = 1.0 / (1.0 + third_diff_std / second_diff_std)
                                    smoothness_scores.append(smoothness_3rd)
                    
                    # 曲率の一貫性
                    if len(first_diff) > 1:
                        curvature_changes = np.abs(np.diff(first_diff))
                        curvature_consistency = 1.0 / (1.0 + np.std(curvature_changes))
                        smoothness_scores.append(curvature_consistency)
                    
                    if len(smoothness_scores) > 0:
                        aesthetic_harmony = np.mean(smoothness_scores)
                
                except:
                    aesthetic_harmony = 0.5  # デフォルト値
            
            # 4. 比例美（上昇・下降期間の比率バランス）- 改良版
            proportional_beauty = 0.0
            if len(window) > 1:
                returns = np.diff(window)
                
                # 異なる閾値での上昇・下降分析
                thresholds = [0.0, np.std(returns) * 0.1, np.std(returns) * 0.5]
                balance_scores = []
                
                for threshold in thresholds:
                    up_periods = np.sum(returns > threshold)
                    down_periods = np.sum(returns < -threshold)
                    total_periods = up_periods + down_periods
                    
                    if total_periods > 0:
                        up_ratio = up_periods / total_periods
                        # 0.5に近いほど美しい（黄金比による調整）
                        optimal_ratio = 1.0 / golden_ratio  # ≈ 0.618
                        balance_score = 1.0 - abs(up_ratio - optimal_ratio)
                        balance_scores.append(max(0.0, balance_score))
                
                if len(balance_scores) > 0:
                    proportional_beauty = np.mean(balance_scores)
            
            # 5. 視覚的バランス（分布の中心性とバランス）- 改良版
            visual_balance = 0.0
            if len(window) > 2:
                # 複数の統計的中心の一致度
                mean_val = np.mean(window)
                median_val = np.median(window)
                
                # 四分位数によるバランス分析
                if len(window) >= 4:
                    sorted_window = np.sort(window)
                    q25_idx = len(sorted_window) // 4
                    q75_idx = 3 * len(sorted_window) // 4
                    
                    q25 = sorted_window[q25_idx]
                    q75 = sorted_window[q75_idx]
                    
                    # 中央値との距離
                    lower_dist = median_val - q25
                    upper_dist = q75 - median_val
                    
                    if lower_dist + upper_dist > 1e-10:
                        quartile_balance = 1.0 - abs(lower_dist - upper_dist) / (lower_dist + upper_dist)
                    else:
                        quartile_balance = 1.0
                    
                    # 平均と中央値の近さ
                    window_range = np.max(window) - np.min(window)
                    if window_range > 1e-10:
                        center_alignment = 1.0 - abs(mean_val - median_val) / window_range
                    else:
                        center_alignment = 1.0
                    
                    # 視覚的バランススコア
                    visual_balance = (quartile_balance + center_alignment) / 2.0
            
            # 結果の格納
            results[i, 0] = golden_ratio_adherence
            results[i, 1] = symmetry_measure
            results[i, 2] = aesthetic_harmony
            results[i, 3] = proportional_beauty
            results[i, 4] = visual_balance
        
        return results
    
    def calculate_music_theory_features_polars(self, data: np.ndarray) -> Dict[str, np.ndarray]:
        """音楽理論特徴量（Polars最適化版）"""
        features = {}
        window_size = 50
        
        # Numba最適化版音楽理論計算
        music_result = self._numba_safe_calculation(
            self._calculate_music_theory_vectorized_optimized, data, window_size
        )
        
        if isinstance(music_result, np.ndarray) and music_result.shape[1] >= 6:
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
            
            # 1. 調性（安定性と方向性の比率）- 改良版
            tonality = 0.0
            if len(returns) > 0:
                return_mean = np.mean(returns)
                return_std = np.std(returns)
                
                # メジャー/マイナーの傾向（統計的解釈）
                positive_count = np.sum(returns > 0)
                negative_count = np.sum(returns < 0)
                total_count = len(returns)
                
                if total_count > 0:
                    positive_ratio = positive_count / total_count
                    # メジャー調性（上昇傾向）vs マイナー調性（下降傾向）
                    major_tendency = positive_ratio
                    minor_tendency = 1.0 - positive_ratio
                    
                    # 調性の明確さ（傾向の強さ）
                    tonality_clarity = abs(major_tendency - minor_tendency)
                    
                    # 安定性（変動の小ささ）
                    if abs(return_mean) > 1e-10:
                        stability = 1.0 / (1.0 + return_std / abs(return_mean))
                    else:
                        stability = 1.0 / (1.0 + return_std)
                    
                    tonality = (tonality_clarity + stability) / 2.0
            
            # 2. リズムパターン（大きな変化の規則性）- 改良版
            rhythm_pattern = 0.0
            if len(returns) > 4:
                threshold = np.std(returns) * 1.5
                accents = np.abs(returns) > threshold
                
                if np.sum(accents) > 1:
                    # アクセントの間隔分析
                    accent_positions = []
                    for j in range(len(accents)):
                        if accents[j]:
                            accent_positions.append(j)
                    
                    if len(accent_positions) > 2:
                        intervals = np.diff(accent_positions)
                        
                        # 間隔の規則性（標準偏差の逆数）
                        if len(intervals) > 1:
                            interval_regularity = 1.0 / (1.0 + np.std(intervals))
                            
                            # 平均間隔の適切さ（音楽的に意味のある間隔）
                            mean_interval = np.mean(intervals)
                            # 2, 3, 4, 6, 8 拍子に対応
                            musical_intervals = np.array([2, 3, 4, 6, 8])
                            interval_musicality = 0.0
                            
                            for musical_int in musical_intervals:
                                closeness = 1.0 / (1.0 + abs(mean_interval - musical_int))
                                interval_musicality = max(interval_musicality, closeness)
                            
                            rhythm_pattern = (interval_regularity + interval_musicality) / 2.0
            
            # 3. 和声（FFT周波数比の協和音程性）- 改良版
            harmony = 0.0
            if len(window) >= 8:
                # 複数周波数成分の協和性分析
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
                    # エネルギーでソートして上位周波数を取得
                    sorted_indices = np.argsort(frequency_energies)[::-1]
                    top_freqs = sorted_indices[:3]
                    
                    # 協和音程の比率チェック
                    harmonic_ratios = []
                    for j in range(len(top_freqs)):
                        for k in range(j + 1, len(top_freqs)):
                            if top_freqs[j] > 0:
                                ratio = top_freqs[k] / top_freqs[j]
                                harmonic_ratios.append(ratio)
                    
                    # 協和音程（音楽理論ベース）
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
            
            # 4. 旋律の輪郭（上昇傾向の方向性）- 改良版
            melody_contour = 0.0
            if len(returns) > 0:
                # 基本的な上昇/下降比率
                up_moves = np.sum(returns > 0)
                down_moves = np.sum(returns < 0)
                total_moves = len(returns)
                
                if total_moves > 0:
                    up_ratio = up_moves / total_moves
                    
                    # 旋律的輪郭の複雑さ
                    direction_changes = 0
                    for j in range(1, len(returns)):
                        if returns[j] * returns[j-1] < 0:
                            direction_changes += 1
                    
                    contour_complexity = direction_changes / max(1, len(returns) - 1)
                    
                    # 音程の大きさの分析
                    large_leaps = np.sum(np.abs(returns) > np.std(returns) * 2)
                    leap_ratio = large_leaps / len(returns)
                    
                    # 旋律的滑らかさ（大きな跳躍の少なさ）
                    melodic_smoothness = 1.0 - leap_ratio
                    
                    melody_contour = (up_ratio + (1.0 - contour_complexity) + melodic_smoothness) / 3.0
            
            # 5. 音楽的緊張（変動範囲とダイナミクス）- 改良版
            musical_tension = 0.0
            if len(window) > 2:
                # ダイナミックレンジ
                dynamic_range = np.max(window) - np.min(window)
                mean_val = np.mean(window)
                
                if abs(mean_val) > 1e-10:
                    relative_range = dynamic_range / abs(mean_val)
                else:
                    relative_range = dynamic_range
                
                # 緊張の蓄積と解放の分析
                cumulative_tension = 0.0
                peak_releases = 0
                
                if len(returns) > 1:
                    cumulative_returns = np.cumsum(returns)
                    
                    for j in range(1, len(cumulative_returns)):
                        # 緊張の蓄積（同方向への継続）
                        if cumulative_returns[j] * cumulative_returns[j-1] > 0:
                            cumulative_tension += abs(returns[j])
                        else:
                            # 解放（方向転換）
                            if cumulative_tension > np.std(returns):
                                peak_releases += 1
                            cumulative_tension = 0
                    
                    # 緊張と解放のバランス
                    if len(returns) > 0:
                        release_frequency = peak_releases / len(returns)
                        tension_buildup = cumulative_tension / np.sum(np.abs(returns))
                        
                        musical_tension = (relative_range + tension_buildup + release_frequency) / 3.0
            
            # 6. テンポ（変化の頻度）- 改良版
            tempo = 0.0
            if len(returns) > 1:
                # 基本的な方向変化頻度
                direction_changes = 0
                for j in range(1, len(returns)):
                    if returns[j] * returns[j-1] < 0:
                        direction_changes += 1
                
                basic_tempo = direction_changes / len(returns)
                
                # 変化の強度を考慮したテンポ
                threshold = np.std(returns) * 0.5
                significant_changes = 0
                
                for j in range(len(returns)):
                    if abs(returns[j]) > threshold:
                        significant_changes += 1
                
                intensity_tempo = significant_changes / len(returns)
                
                # 周期性によるテンポ安定性
                tempo_stability = 0.0
                if len(returns) >= 4:
                    # 簡易自己相関による周期検出
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
    
    def calculate_astronomy_features_polars(self, data: np.ndarray) -> Dict[str, np.ndarray]:
        """天文学・宇宙論特徴量（Polars最適化版）"""
        features = {}
        window_size = 50
        
        # Numba最適化版天文学計算
        astronomy_result = self._numba_safe_calculation(
            self._calculate_astronomy_vectorized_optimized, data, window_size
        )
        
        if isinstance(astronomy_result, np.ndarray) and astronomy_result.shape[1] >= 6:
            astronomy_names = ['orbital_mechanics', 'gravitational_wave', 'stellar_pulsation',
                             'cosmic_expansion', 'dark_energy', 'big_bang_echo']
            
            for j, name in enumerate(astronomy_names):
                features[f'{name}_{window_size}'] = astronomy_result[:, j]
        else:
            # フォールバック値
            astronomy_names = ['orbital_mechanics', 'gravitational_wave', 'stellar_pulsation',
                             'cosmic_expansion', 'dark_energy', 'big_bang_echo']
            for name in astronomy_names:
                features[f'{name}_{window_size}'] = np.zeros(len(data))
        
        return features
    
    @staticmethod
    @njit(parallel=True, cache=True)
    def _calculate_astronomy_vectorized_optimized(data: np.ndarray, window_size: int) -> np.ndarray:
        """天文学・宇宙論特徴量計算（完全ベクトル化・並列最適化版）"""
        n = len(data)
        results = np.zeros((n, 6))
        
        if n < window_size:
            return results
        
        # 並列計算
        for i in prange(window_size-1, n):
            window = data[i-window_size+1:i+1]
            
            # 1. 軌道力学（周期的自己相関）- 改良版
            orbital_mechanics = 0.0
            orbital_periods = [12, 24, 36]  # 複数の軌道周期を検証
            
            for period in orbital_periods:
                if len(window) > period:
                    lag = min(period, len(window) // 4)
                    if lag > 0:
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
                                    orbital_mechanics = max(orbital_mechanics, abs(correlation))
            
            # ケプラーの法則風の楕円性分析
            if len(window) >= 8:
                # 速度変化の分析（ケプラーの第2法則）
                velocities = np.abs(np.diff(window))
                if len(velocities) > 1:
                    velocity_variation = np.std(velocities) / (np.mean(velocities) + 1e-10)
                    orbital_eccentricity = min(1.0, velocity_variation)
                    orbital_mechanics = (orbital_mechanics + orbital_eccentricity) / 2.0
            
            # 2. 重力波（ボラティリティの周期変化）- 改良版
            gravitational_wave = 0.0
            if len(window) >= 16:
                # ボラティリティの時系列構築
                vol_window_size = 4
                volatilities = []
                
                for j in range(vol_window_size - 1, len(window)):
                    vol_segment = window[j - vol_window_size + 1:j + 1]
                    volatilities.append(np.std(vol_segment))
                
                if len(volatilities) >= 8:
                    # ボラティリティの周期解析（重力波のchirp信号を模擬）
                    vol_array = np.array(volatilities)
                    
                    # 複数周波数での周期性検出
                    max_wave_energy = 0.0
                    for freq_idx in range(1, min(len(vol_array) // 2, 8)):
                        freq = freq_idx * 2 * np.pi / len(vol_array)
                        real_part = 0.0
                        imag_part = 0.0
                        
                        for k in range(len(vol_array)):
                            angle = freq * k
                            real_part += vol_array[k] * np.cos(angle)
                            imag_part += vol_array[k] * np.sin(angle)
                        
                        wave_energy = real_part**2 + imag_part**2
                        max_wave_energy = max(max_wave_energy, wave_energy)
                    
                    total_vol_energy = np.sum(vol_array**2)
                    if total_vol_energy > 1e-10:
                        gravitational_wave = max_wave_energy / total_vol_energy
            
            # 3. 恒星脈動（主要周波数の相対強度）- 改良版
            stellar_pulsation = 0.0
            if len(window) >= 8:
                # 複数タイプの脈動を検出
                pulsation_energies = []
                
                # 基本脈動から高次まで
                for freq_idx in range(1, min(len(window) // 2, 12)):
                    freq = freq_idx * 2 * np.pi / len(window)
                    real_part = 0.0
                    imag_part = 0.0
                    
                    for j in range(len(window)):
                        angle = freq * j
                        real_part += window[j] * np.cos(angle)
                        imag_part += window[j] * np.sin(angle)
                    
                    energy = real_part**2 + imag_part**2
                    pulsation_energies.append(energy)
                
                if len(pulsation_energies) > 0:
                    total_energy = np.sum(pulsation_energies)
                    if total_energy > 1e-10:
                        # 支配的脈動の強度
                        dominant_energy = np.max(pulsation_energies)
                        stellar_pulsation = dominant_energy / total_energy
                        
                        # 脈動の規則性（エネルギー分布の均一性）
                        energy_entropy = 0.0
                        for energy in pulsation_energies:
                            if energy > 1e-10:
                                p = energy / total_energy
                                energy_entropy -= p * np.log2(p)
                        
                        max_entropy = np.log2(len(pulsation_energies))
                        if max_entropy > 0:
                            regularity = energy_entropy / max_entropy
                            stellar_pulsation = (stellar_pulsation + regularity) / 2.0
            
            # 4. 宇宙膨張（総リターンと加速度分析）- 改良版
            cosmic_expansion = 0.0
            if len(window) > 1:
                total_return = window[-1] - window[0]
                initial_value = window[0]
                
                if abs(initial_value) > 1e-10:
                    expansion_rate = total_return / initial_value
                else:
                    expansion_rate = total_return
                
                # ハッブル定数風の距離-速度関係
                if len(window) >= 4:
                    # 時間区間での膨張率変化
                    quarter = len(window) // 4
                    expansion_phases = []
                    
                    for phase in range(4):
                        start_idx = phase * quarter
                        end_idx = min((phase + 1) * quarter, len(window))
                        if end_idx > start_idx + 1:
                            phase_data = window[start_idx:end_idx]
                            phase_expansion = (phase_data[-1] - phase_data[0]) / (phase_data[0] + 1e-10)
                            expansion_phases.append(phase_expansion)
                    
                    if len(expansion_phases) > 1:
                        # 膨張の加速（ダークエネルギー効果）
                        expansion_acceleration = np.diff(expansion_phases)
                        if len(expansion_acceleration) > 0:
                            avg_acceleration = np.mean(expansion_acceleration)
                            cosmic_expansion = (expansion_rate + avg_acceleration) / 2.0
                        else:
                            cosmic_expansion = expansion_rate
                else:
                    cosmic_expansion = expansion_rate
            
            # 5. ダークエネルギー（加速度の分析）- 改良版
            dark_energy = 0.0
            if len(window) >= 3:
                # 1次微分（速度）
                velocity = np.diff(window)
                
                if len(velocity) >= 2:
                    # 2次微分（加速度）
                    acceleration = np.diff(velocity)
                    
                    if len(acceleration) > 0:
                        # 平均加速度
                        mean_acceleration = np.mean(acceleration)
                        
                        # 加速度の一貫性
                        acceleration_consistency = 1.0 / (1.0 + np.std(acceleration))
                        
                        # ダークエネルギー風の反発力（正の加速度）
                        if mean_acceleration > 0:
                            dark_energy_strength = mean_acceleration / (np.std(window) + 1e-10)
                        else:
                            dark_energy_strength = 0.0
                        
                        dark_energy = (dark_energy_strength + acceleration_consistency) / 2.0
                        
                        # 3次微分（ジャーク）による更なる分析
                        if len(acceleration) >= 2:
                            jerk = np.diff(acceleration)
                            if len(jerk) > 0:
                                jerk_stability = 1.0 / (1.0 + np.std(jerk))
                                dark_energy = (dark_energy + jerk_stability) / 2.0
            
            # 6. ビッグバンエコー（最大ショックの時間荷重影響）- 改良版
            big_bang_echo = 0.0
            if len(window) > 1:
                returns = np.diff(window)
                
                if len(returns) > 0:
                    # 複数のショックイベントを検出
                    shock_threshold = np.std(returns) * 2.0
                    shock_events = []
                    
                    for j in range(len(returns)):
                        if abs(returns[j]) > shock_threshold:
                            shock_events.append({
                                'magnitude': abs(returns[j]),
                                'time_index': j,
                                'type': 'expansion' if returns[j] > 0 else 'contraction'
                            })
                    
                    if len(shock_events) > 0:
                        total_echo = 0.0
                        
                        for shock in shock_events:
                            # 時間経過による減衰（宇宙の冷却効果）
                            time_since_shock = len(returns) - shock['time_index']
                            decay_factor = np.exp(-time_since_shock / len(returns))
                            
                            # ショック強度の正規化
                            normalized_magnitude = shock['magnitude'] / (np.std(returns) + 1e-10)
                            
                            # エコー強度
                            echo_contribution = normalized_magnitude * decay_factor
                            total_echo += echo_contribution
                        
                        # 宇宙マイクロ波背景放射風の均一性
                        if len(shock_events) > 1:
                            shock_magnitudes = [s['magnitude'] for s in shock_events]
                            shock_uniformity = 1.0 / (1.0 + np.std(shock_magnitudes) / (np.mean(shock_magnitudes) + 1e-10))
                            big_bang_echo = (total_echo + shock_uniformity) / 2.0
                        else:
                            big_bang_echo = total_echo
            
            # 結果の格納
            results[i, 0] = orbital_mechanics
            results[i, 1] = gravitational_wave
            results[i, 2] = stellar_pulsation
            results[i, 3] = cosmic_expansion
            results[i, 4] = dark_energy
            results[i, 5] = big_bang_echo
        
        return results
    
    def calculate_biomechanics_features_polars(self, data: np.ndarray) -> Dict[str, np.ndarray]:
        """生体力学・パフォーマンス科学特徴量（Polars最適化版）"""
        features = {}
        window_size = 50
        
        # Numba最適化版生体力学計算
        biomechanics_result = self._numba_safe_calculation(
            self._calculate_biomechanics_vectorized_optimized, data, window_size
        )
        
        if isinstance(biomechanics_result, np.ndarray) and biomechanics_result.shape[1] >= 7:
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
            
            # リターンを速度として扱う
            returns = np.diff(window)
            
            # 1. 運動エネルギー（動きの激しさ）- 改良版
            kinetic_energy = 0.0
            if len(returns) > 0:
                # 基本運動エネルギー（1/2 * m * v^2）
                basic_ke = 0.5 * np.sum(returns**2)
                
                # 速度の分布分析
                velocity_variance = np.var(returns)
                velocity_skewness = 0.0
                
                if len(returns) >= 3 and velocity_variance > 1e-10:
                    # 歪度の計算
                    mean_velocity = np.mean(returns)
                    std_velocity = np.sqrt(velocity_variance)
                    
                    skew_sum = 0.0
                    for vel in returns:
                        normalized_vel = (vel - mean_velocity) / std_velocity
                        skew_sum += normalized_vel**3
                    
                    velocity_skewness = skew_sum / len(returns)
                
                # 運動の効率性（エネルギーの有効利用）
                total_displacement = abs(window[-1] - window[0])
                total_distance = np.sum(np.abs(returns))
                
                if total_distance > 1e-10:
                    movement_efficiency = total_displacement / total_distance
                else:
                    movement_efficiency = 0.0
                
                kinetic_energy = basic_ke * (1.0 + movement_efficiency)
            
            # 2. 位置エネルギー（価格水準からの乖離エネルギー）- 改良版
            potential_energy = 0.0
            if len(window) > 0:
                # 基準レベル（最低点）からの位置エネルギー
                min_level = np.min(window)
                height_energy = np.sum((window - min_level)**2)
                
                # 重力場の強度（価格の引力）
                center_of_mass = np.mean(window)
                gravitational_potential = 0.0
                
                for j in range(len(window)):
                    distance_from_center = abs(window[j] - center_of_mass)
                    # 重力ポテンシャル（距離の逆数）
                    if distance_from_center > 1e-10:
                        gravitational_potential += 1.0 / distance_from_center
                
                potential_energy = height_energy + gravitational_potential
            
            # 3. 筋力（瞬発的な最大変動）- 改良版
            muscle_force = 0.0
            if len(returns) > 0:
                # 最大瞬発力
                max_instantaneous_force = np.max(np.abs(returns))
                
                # 力の持続性分析
                force_threshold = np.std(returns) * 1.5
                sustained_force_periods = 0
                
                for j in range(len(returns)):
                    if abs(returns[j]) > force_threshold:
                        sustained_force_periods += 1
                
                force_endurance_ratio = sustained_force_periods / len(returns)
                
                # パワー（力 × 速度）
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
            
            # 4. 関節可動域（変動範囲の相対的な広さ）- 改良版
            joint_mobility = 0.0
            if len(window) > 0:
                # 基本可動域
                price_range = np.max(window) - np.min(window)
                mean_price = np.mean(window)
                
                if abs(mean_price) > 1e-10:
                    basic_mobility = price_range / abs(mean_price)
                else:
                    basic_mobility = price_range
                
                # 関節の柔軟性（滑らかな動き）
                if len(returns) >= 2:
                    acceleration = np.diff(returns)
                    acceleration_smoothness = 1.0 / (1.0 + np.std(acceleration))
                    
                    # 可動域の活用度
                    utilization_levels = np.linspace(np.min(window), np.max(window), 10)
                    utilization_count = 0
                    
                    for level in utilization_levels:
                        tolerance = price_range * 0.05  # 5%の許容範囲
                        if np.any(np.abs(window - level) <= tolerance):
                            utilization_count += 1
                    
                    range_utilization = utilization_count / len(utilization_levels)
                    
                    joint_mobility = (basic_mobility + acceleration_smoothness + range_utilization) / 3.0
                else:
                    joint_mobility = basic_mobility
            
            # 5. パフォーマンスの一貫性（安定性）- 改良版
            performance_consistency = 0.0
            if len(returns) > 0:
                # 基本的な一貫性（変動の小ささ）
                std_returns = np.std(returns)
                mean_abs_returns = np.mean(np.abs(returns))
                
                if mean_abs_returns > 1e-10:
                    coefficient_of_variation = std_returns / mean_abs_returns
                    basic_consistency = 1.0 / (1.0 + coefficient_of_variation)
                else:
                    basic_consistency = 1.0
                
                # 予測可能性（トレンドの一貫性）
                trend_consistency = 0.0
                if len(returns) >= 3:
                    # 移動平均からの乖離
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
            
            # 6. 持久力（上昇トレンドの持続性）- 改良版
            endurance = 0.0
            if len(returns) > 0:
                # 基本的な方向一貫性
                positive_moves = np.sum(returns > 0)
                total_moves = len(returns)
                basic_endurance = positive_moves / total_moves
                
                # 持続的なパフォーマンス分析
                streak_lengths = []
                current_streak = 0
                current_direction = 0
                
                for j in range(len(returns)):
                    if returns[j] > 0:  # 上昇
                        if current_direction >= 0:
                            current_streak += 1
                        else:
                            if current_streak > 0:
                                streak_lengths.append(current_streak)
                            current_streak = 1
                        current_direction = 1
                    elif returns[j] < 0:  # 下降
                        if current_direction <= 0:
                            current_streak += 1
                        else:
                            if current_streak > 0:
                                streak_lengths.append(current_streak)
                            current_streak = 1
                        current_direction = -1
                
                if current_streak > 0:
                    streak_lengths.append(current_streak)
                
                # 最長持続期間の分析
                if len(streak_lengths) > 0:
                    max_streak = np.max(streak_lengths)
                    avg_streak = np.mean(streak_lengths)
                    streak_endurance = max_streak / len(returns)
                    
                    endurance = (basic_endurance + streak_endurance) / 2.0
                else:
                    endurance = basic_endurance
            
            # 7. 回復率（最大ドローダウンからの回復）- 改良版
            recovery_rate = 0.0
            if len(returns) > 0:
                # 累積リターンの計算
                cumulative_returns = np.zeros(len(returns) + 1)
                for j in range(len(returns)):
                    cumulative_returns[j + 1] = cumulative_returns[j] + returns[j]
                
                # ドローダウンの計算
                running_max = np.zeros(len(cumulative_returns))
                drawdowns = np.zeros(len(cumulative_returns))
                
                running_max[0] = cumulative_returns[0]
                for j in range(1, len(cumulative_returns)):
                    running_max[j] = max(running_max[j-1], cumulative_returns[j])
                    drawdowns[j] = running_max[j] - cumulative_returns[j]
                
                max_drawdown = np.max(drawdowns)
                
                # 回復分析
                if max_drawdown > 1e-10:
                    final_return = cumulative_returns[-1]
                    
                    # 基本回復率
                    basic_recovery = final_return / max_drawdown
                    
                    # 回復速度の分析
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
                    recovery_rate = 1.0  # ドローダウンがない場合は完全回復
            
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
            # Block 1: 基礎テクニカル指標（Polarsベース）
            # =========================================================================
            logger.info("基礎テクニカル指標計算中...")
            basic_tech_start = time.time()
            
            basic_tech_features = self.calculate_basic_technical_indicators_polars(high, low, close, volume)
            all_features.update(basic_tech_features)
            
            logger.info(f"基礎テクニカル指標完了: {len(basic_tech_features)}個 ({time.time() - basic_tech_start:.2f}秒)")
            
            # =========================================================================
            # Block 2: Tier S特徴量（最優先・Numba最適化）
            # =========================================================================
            logger.info("Tier S特徴量計算中...")
            tier_s_start = time.time()
            
            # MFDFA
            mfdfa_features = self.calculate_mfdfa_features_polars(close)
            all_features.update(mfdfa_features)
            
            # Microstructure Noise Ratio
            noise_features = self.calculate_microstructure_noise_features_polars(close)
            all_features.update(noise_features)
            
            logger.info(f"Tier S特徴量完了: {len(mfdfa_features) + len(noise_features)}個 ({time.time() - tier_s_start:.2f}秒)")
            
            # =========================================================================
            # Block 3: Tier 1特徴量（重要・ハイブリッド最適化）
            # =========================================================================
            logger.info("Tier 1特徴量計算中...")
            tier_1_start = time.time()
            
            # ショックモデル
            shock_features = self.calculate_shock_model_features_polars(close)
            all_features.update(shock_features)
            
            # Multi-Scale Volatility
            multiscale_vol_features = self.calculate_multiscale_volatility_polars(close)
            all_features.update(multiscale_vol_features)
            
            logger.info(f"Tier 1特徴量完了: {len(shock_features) + len(multiscale_vol_features)}個 ({time.time() - tier_1_start:.2f}秒)")
            
            # =========================================================================
            # Block 4: Tier 2特徴量（標準・効率最適化）
            # =========================================================================
            logger.info("Tier 2特徴量計算中...")
            tier_2_start = time.time()
            
            # EMD/CEEMDAN
            emd_features = self.calculate_emd_features_polars(close)
            all_features.update(emd_features)
            
            # 統計的モーメント
            moments_features = self.calculate_statistical_moments_polars(close)
            all_features.update(moments_features)
            
            # ロバスト統計
            robust_features = self.calculate_robust_statistics_polars(close)
            all_features.update(robust_features)
            
            logger.info(f"Tier 2特徴量完了: {len(emd_features) + len(moments_features) + len(robust_features)}個 ({time.time() - tier_2_start:.2f}秒)")
            
            # =========================================================================
            # Block 5: スペクトル・信号処理特徴量
            # =========================================================================
            logger.info("スペクトル・信号処理特徴量計算中...")
            spectral_start = time.time()
            
            # スペクトル特徴量
            spectral_features = self.calculate_spectral_features_polars(close)
            all_features.update(spectral_features)
            
            # ウェーブレット特徴量
            wavelet_features = self.calculate_wavelet_features_polars(close)
            all_features.update(wavelet_features)
            
            # カオス理論特徴量
            chaos_features = self.calculate_chaos_features_polars(close)
            all_features.update(chaos_features)
            
            # ヒルベルト変換特徴量
            hilbert_features = self.calculate_hilbert_transform_features_polars(close)
            all_features.update(hilbert_features)
            
            logger.info(f"スペクトル・信号処理特徴量完了: {len(spectral_features) + len(wavelet_features) + len(chaos_features) + len(hilbert_features)}個 ({time.time() - spectral_start:.2f}秒)")
            
            # =========================================================================
            # Block 6: ADX・基本オシレーター
            # =========================================================================
            logger.info("ADX・基本オシレーター計算中...")
            osc_start = time.time()
            
            # ADX
            adx_features = self.calculate_adx_features_polars(high, low, close)
            all_features.update(adx_features)
            
            # Parabolic SAR
            sar_features = self.calculate_parabolic_sar_features_polars(high, low, close)
            all_features.update(sar_features)
            
            # CCI
            cci_features = self.calculate_cci_features_polars(high, low, close)
            all_features.update(cci_features)
            
            # Williams %R
            williams_features = self.calculate_williams_r_features_polars(high, low, close)
            all_features.update(williams_features)
            
            # Aroon
            aroon_features = self.calculate_aroon_features_polars(high, low)
            all_features.update(aroon_features)
            
            # Ultimate Oscillator
            uo_features = self.calculate_ultimate_oscillator_features_polars(high, low, close)
            all_features.update(uo_features)
            
            logger.info(f"ADX・基本オシレーター完了: {len(adx_features) + len(sar_features) + len(cci_features) + len(williams_features) + len(aroon_features) + len(uo_features)}個 ({time.time() - osc_start:.2f}秒)")
            
            # =========================================================================
            # Block 7: 出来高関連指標
            # =========================================================================
            logger.info("出来高関連指標計算中...")
            volume_start = time.time()
            
            volume_features = self.calculate_volume_features_polars(high, low, close, volume)
            all_features.update(volume_features)
            
            logger.info(f"出来高関連指標完了: {len(volume_features)}個 ({time.time() - volume_start:.2f}秒)")
            
            # =========================================================================
            # Block 8: 移動平均線・トレンド分析
            # =========================================================================
            logger.info("移動平均線・トレンド分析計算中...")
            ma_start = time.time()
            
            ma_features = self.calculate_moving_averages_polars(close)
            all_features.update(ma_features)
            
            logger.info(f"移動平均線・トレンド分析完了: {len(ma_features)}個 ({time.time() - ma_start:.2f}秒)")
            
            # =========================================================================
            # Block 9: ボラティリティ・バンド指標
            # =========================================================================
            logger.info("ボラティリティ・バンド指標計算中...")
            vol_start = time.time()
            
            volatility_features = self.calculate_volatility_bands_polars(high, low, close)
            all_features.update(volatility_features)
            
            logger.info(f"ボラティリティ・バンド指標完了: {len(volatility_features)}個 ({time.time() - vol_start:.2f}秒)")
            
            # =========================================================================
            # Block 10: サポート・レジスタンス・ローソク足
            # =========================================================================
            logger.info("サポート・レジスタンス・ローソク足計算中...")
            sr_start = time.time()
            
            sr_features = self.calculate_support_resistance_polars(high, low, close, open_prices)
            all_features.update(sr_features)
            
            logger.info(f"サポート・レジスタンス・ローソク足完了: {len(sr_features)}個 ({time.time() - sr_start:.2f}秒)")
            
            # =========================================================================
            # Block 11: 情報理論特徴量
            # =========================================================================
            logger.info("情報理論特徴量計算中...")
            info_start = time.time()
            
            info_features = self.calculate_information_theory_features_polars(close)
            all_features.update(info_features)
            
            logger.info(f"情報理論特徴量完了: {len(info_features)}個 ({time.time() - info_start:.2f}秒)")
            
            # =========================================================================
            # Block 12: 学際的アナロジー特徴量
            # =========================================================================
            logger.info("学際的アナロジー特徴量計算中...")
            interdisciplinary_start = time.time()
            
            interdisciplinary_features = self.calculate_interdisciplinary_features_polars(close)
            all_features.update(interdisciplinary_features)
            
            logger.info(f"学際的アナロジー特徴量完了: {len(interdisciplinary_features)}個 ({time.time() - interdisciplinary_start:.2f}秒)")
            
            # =========================================================================
            # 最終処理・品質保証
            # =========================================================================
            logger.info("品質保証・最終処理中...")
            qa_start = time.time()
            
            # 品質チェック・統計情報
            quality_report = self.generate_quality_report_polars(all_features)
            
            # 異常値処理（必要に応じて）
            all_features = self.apply_final_cleaning_polars(all_features)
            
            logger.info(f"品質保証完了 ({time.time() - qa_start:.2f}秒)")
            
        except Exception as e:
            logger.error(f"特徴量計算中にエラーが発生: {e}")
            import traceback
            traceback.print_exc()
        
        total_time = time.time() - start_time
        logger.info(f"全特徴量計算完了: {len(all_features)}個の特徴量を{total_time:.2f}秒で生成")
        logger.info(f"計算統計: Polars={self.calculation_stats['polars_calculations']}, Numba={self.calculation_stats['numba_calculations']}")
        
        return all_features
    
    # =========================================================================
    # 品質保証システム（Polars最適化版）
    # =========================================================================
    
    def generate_quality_report_polars(self, features: Dict[str, np.ndarray]) -> Dict[str, Any]:
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
    
    def apply_final_cleaning_polars(self, features: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """最終クリーニング処理（Polars最適化版）"""
        
        if not features:
            return features
        
        try:
            # Polars DataFrameに変換
            df = pl.DataFrame(features)
            
            # 一括クリーニング処理
            cleaned_df = df.with_columns([
                # NaN/Infを0で置換
                pl.all().fill_nan(0.0).fill_null(0.0)
            ]).with_columns([
                # 異常に大きな値をクランプ
                pl.all().clip(-1e10, 1e10)
            ])
            
            # 辞書に変換して返す
            cleaned_features = {}
            for col in cleaned_df.columns:
                cleaned_features[col] = cleaned_df[col].to_numpy()
            
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
        polars_ratio = stats['polars_calculations'] / total_calc
        numba_ratio = stats['numba_calculations'] / total_calc
        
        avg_time = np.mean(stats['computation_times']) if stats['computation_times'] else 0.0
        
        return {
            'total_calculations': total_calc,
            'success_rate': success_rate,
            'polars_calculation_ratio': polars_ratio,
            'numba_calculation_ratio': numba_ratio,
            'avg_computation_time_ms': avg_time * 1000,
            'total_computation_time_sec': sum(stats['computation_times']),
            'optimization_level': 'polars_first' if polars_ratio > 0.7 else 'hybrid',
            'performance_grade': (
                'S' if avg_time < 0.01 and success_rate > 0.95 and polars_ratio > 0.8 else
                'A' if avg_time < 0.05 and success_rate > 0.9 and polars_ratio > 0.6 else
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
            
            # 品質レポート生成
            quality_report = self.calculator.generate_quality_report_polars(all_features)
            if quality_report.get('warnings'):
                self.execution_stats['quality_alerts'].extend(quality_report['warnings'])
                logger.debug(f"チャンク {chunk_idx+1}: 品質アラート {len(quality_report['warnings'])}件")
            
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
            # ダミーデータ生成（MFDFA等の要件を考慮し、サイズを統一）
            np.random.seed(42)
            test_size = 250
            test_data = np.random.randn(test_size) * 0.01 + 100  # 価格データ風
            
            # Calculator初期化
            dummy_window_manager = WindowManager(window_size=50, overlap=0.3)
            dummy_memory_manager = MemoryManager()
            calculator = Calculator(dummy_window_manager, dummy_memory_manager)
            
            # 基本統計計算テスト
            statistical_features = calculator.calculate_statistical_moments_polars(test_data)
            robust_features = calculator.calculate_robust_statistics_polars(test_data)
            spectral_features = calculator.calculate_spectral_features_polars(test_data)
            
            # 高度特徴量テスト
            mfdfa_features = calculator.calculate_mfdfa_features_polars(test_data)
            noise_features = calculator.calculate_microstructure_noise_features_polars(test_data)
            
            # 計算結果検証
            all_features = {}
            all_features.update(statistical_features)
            all_features.update(robust_features)
            all_features.update(spectral_features)
            all_features.update(mfdfa_features)
            all_features.update(noise_features)
            
            # 品質レポート生成メソッドを呼び出すように変更
            quality_report = calculator.generate_quality_report_polars(all_features)
            
            # レポートから必要な値を取得
            avg_quality = quality_report.get('overall_quality_score', 0.0)
            total_nan = quality_report.get('data_quality', {}).get('total_nan_values', len(test_data))
            
            # 成功率の定義を更新 (高品質な特徴量の割合)
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