#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
革新的特徴量収集スクリプト実装 - Calculator中心再設計版
Project Forge 軍資金増大ミッション - 80%リソースCalculator集中アーキテクチャ
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
import json
import re

# Numbaデバッグ出力の無効化
import os
os.environ['NUMBA_DEBUG'] = '0'
os.environ['NUMBA_DEBUGINFO'] = '0'

import logging
logging.getLogger('numba').setLevel(logging.WARNING)

# 数値計算・データ処理
import numpy as np
import pandas as pd
import polars as pl
from numba import njit, jit, prange
import numba

# 科学計算ライブラリ
from scipy import stats
from scipy.stats import skew, kurtosis
from scipy.signal import welch, hilbert, savgol_filter
from scipy.ndimage import gaussian_filter, median_filter
from scipy.spatial.distance import pdist, squareform
from scipy.optimize import minimize_scalar
import pywt

# 機械学習
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

warnings.filterwarnings('ignore')

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
# グローバル定数
# =============================================================================

HARDWARE_SPEC = {
    'gpu_memory': '12GB',  # RTX 3060
    'cpu_cores': 6,        # i7-8700K
    'ram_limit': 64,       # 64GB RAM
}

DATA_CONFIG = {
    'base_path': Path('/workspaces/project_forge/data/1_XAUUSD_base_data/XAUUSD_tick_master_GOLDEN'),
    'output_path': Path('/workspaces/project_forge/data/2_feature_value/'),
    'timeframes': ['tick', 'M0.5', 'M1', 'M3', 'M5', 'M8', 'M15', 'M30', 'H1', 'H4', 'H6', 'H12', 'D1', 'W1', 'MN'],
}

# 数学的定数（Golden Ratio基準）
GOLDEN_RATIO = (1 + np.sqrt(5)) / 2
FIBONACCI_RATIOS = [0.236, 0.382, 0.618, 0.786, 1.618]

CALC_PARAMS = {
    # 基本期間（フィボナッチ数列ベース）
    'base_periods': [8, 13, 21, 34, 55, 89, 144],
    
    # ADX・オシレーター用
    'adx_periods': [13, 21, 34],
    'rsi_periods': [int(13 * GOLDEN_RATIO), 21, 34],
    'cci_periods': [13, int(21 * GOLDEN_RATIO)],
    'williams_periods': [13, 21],
    'aroon_periods': [13, 21],
    
    # 移動平均用（Golden Ratio調整）
    'ma_short_periods': [8, 13, 21],
    'ma_long_periods': [34, 55, 89],
    'ma_multipliers': [GOLDEN_RATIO, GOLDEN_RATIO**2],
    
    # ボラティリティ用
    'volatility_periods': [13, 21, 34, 55],
    'bb_multipliers': [1.618, 2.0, 2.618],  # Golden Ratioベース
    'atr_multipliers': [1.0, GOLDEN_RATIO, 2.0, GOLDEN_RATIO**2],
    
    # 出来高用
    'volume_periods': [13, 21, 34],
    'cmf_periods': [21, 34],
    'mfi_periods': [13, 21],
    
    # サポレジ・フィボナッチ用
    'pivot_periods': [13, 21, 34],
    'fibonacci_levels': FIBONACCI_RATIOS,
    'channel_periods': [21, 34, 55]
}

# =============================================================================
# システム検証・最適化関数
# =============================================================================

def validate_system_requirements() -> bool:
    """システム要件チェック（簡素化版）"""
    import psutil
    
    # メモリチェック（WSL2環境考慮）
    total_memory_gb = psutil.virtual_memory().total / (1024**3)
    min_memory_threshold = 16  # 32GBから16GBに調整
    
    if total_memory_gb < min_memory_threshold:
        logger.error(f"メモリ不足: {total_memory_gb:.1f}GB < {min_memory_threshold}GB 必要")
        return False
    elif total_memory_gb < 32:
        logger.warning(f"メモリ警告: {total_memory_gb:.1f}GB < 32GB 推奨（WSL2環境では正常）")
    
    # CPU チェック
    cpu_count = psutil.cpu_count()
    if cpu_count is None or cpu_count < 4:
        logger.error(f"CPU不足: {cpu_count}コア < 4コア推奨")
        return False
    
    # ディスク容量チェック
    disk_free_gb = psutil.disk_usage('/').free / (1024**3)
    if disk_free_gb < 10:
        logger.error(f"ディスク容量不足: {disk_free_gb:.1f}GB < 10GB推奨")
        return False
    
    logger.info("システム要件チェック: ✓ 全て合格")
    return True

def optimize_numpy_settings():
    """NumPy 最適化設定（i7-8700K 6コア最適化）"""
    # OpenBLAS/MKL スレッド数設定
    os.environ['OMP_NUM_THREADS'] = str(HARDWARE_SPEC['cpu_cores'])
    os.environ['MKL_NUM_THREADS'] = str(HARDWARE_SPEC['cpu_cores'])
    os.environ['OPENBLAS_NUM_THREADS'] = str(HARDWARE_SPEC['cpu_cores'])
    
    # Numba 設定
    os.environ['NUMBA_NUM_THREADS'] = str(HARDWARE_SPEC['cpu_cores'])
    os.environ['NUMBA_CACHE_DIR'] = '/tmp/numba_cache'
    
    logger.info(f"NumPy最適化設定完了: {HARDWARE_SPEC['cpu_cores']}スレッド")

def display_system_info():
    """システム情報表示（簡素化版）"""
    import psutil
    import platform
    
    # 基本情報取得
    cpu_count = psutil.cpu_count(logical=True)
    memory_gb = psutil.virtual_memory().total / (1024**3)
    disk_gb = psutil.disk_usage('/').total / (1024**3)
    
    logger.info(f"""
    ========== システム情報 ==========
    OS: {platform.system()} {platform.release()}
    CPU: {cpu_count}コア
    メモリ: {memory_gb:.1f}GB
    ディスク: {disk_gb:.1f}GB
    Python: {platform.python_version()}
    NumPy: {np.__version__}
    Polars: {pl.__version__}
    Numba: {numba.__version__}
    ================================
    """)

# =============================================================================
# DataProcessor（最低限実装・10%リソース）
# =============================================================================

class DataProcessor:
    """データ処理クラス - memmap統一使用"""
    
    def __init__(self, base_path: Path):
        self.base_path = Path(base_path)
        self.metadata_path = self.base_path / "_metadata"
        self.memmap_cache_dir = self.base_path.parent / "memmap_cache"
        self.memmap_cache_dir.mkdir(parents=True, exist_ok=True)
        
        self.partition_info = {}
        self.total_rows = 0
        self.schema_info = {}
        
        logger.info(f"DataProcessor初期化: {self.base_path}")

    def load_metadata(self) -> Dict[str, Any]:
        """メタデータファイルから構造情報を読み込み"""
        if not self.metadata_path.exists():
            logger.warning("メタデータファイルが見つかりません")
            return {'total_rows': 0, 'schema': {}}
        
        try:
            import pyarrow.parquet as pq
            metadata = pq.read_metadata(self.metadata_path)
            return {'total_rows': metadata.num_rows}
        except Exception as e:
            logger.error(f"メタデータ読み込みエラー: {e}")
            return {'total_rows': 0}
        
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
    
    
    def convert_to_memmap(self, timeframes: Optional[List[str]] = None) -> Dict[str, np.memmap]:
        """Parquetデータをmemmapに変換"""
        if timeframes is None:
            timeframes = ['tick']
        
        memmap_files = {}
        for tf in timeframes:
            memmap_path = self.memmap_cache_dir / f"{tf}_data.dat"
            
            if memmap_path.exists():
                try:
                    meta_path = memmap_path.parent / f"{memmap_path.stem}.meta"
                    if meta_path.exists():
                        meta_info = np.load(str(meta_path), allow_pickle=True).item()
                        memmap_files[tf] = np.memmap(str(memmap_path), dtype=meta_info['dtype'], 
                                                mode='r', shape=meta_info['shape'])
                        logger.info(f"既存memmap使用: {tf}")
                        continue
                except Exception as e:
                    logger.debug(f"既存memmap読み込み失敗: {e}")
                    
            # 新規作成
            logger.info(f"memmap再作成: {tf}")
            memmap_files[tf] = self._create_memmap_streaming(tf, memmap_path)
        
        return memmap_files
    
    def _create_memmap_streaming(self, timeframe: str, memmap_path: Path) -> np.memmap:
        """ストリーミング処理でmemmapファイル作成（プロンプト準拠版）"""
        try:
            # パーティション構造から正しいパスを取得
            if timeframe not in self.partition_info:
                self.scan_partition_structure()
            
            parquet_files = self.partition_info.get(timeframe, [])
            
            if not parquet_files:
                # Hiveパーティション以外の構造も試行
                parquet_files = list(self.base_path.glob(f"*{timeframe}*.parquet"))
            
            if not parquet_files:
                logger.warning(f"Parquetファイルが見つかりません: {timeframe}")
                # 最小限のテスト用memmap作成
                test_rows = 1000
                memmap_array = np.memmap(str(memmap_path), dtype=np.float64, mode='w+', 
                                    shape=(test_rows, 7))
                np.random.seed(42)
                close_prices = 2000 * np.cumprod(1 + np.random.normal(0, 0.001, test_rows))
                memmap_array[:, 0] = np.arange(test_rows)  # timestamp
                memmap_array[:, 1] = np.roll(close_prices, 1)  # open
                memmap_array[:, 2] = close_prices + np.random.uniform(0, 2, test_rows)  # high
                memmap_array[:, 3] = close_prices - np.random.uniform(0, 2, test_rows)  # low
                memmap_array[:, 4] = close_prices  # close
                memmap_array[:, 5] = np.random.exponential(1000, test_rows)  # volume
                memmap_array[:, 6] = np.random.randn(test_rows)  # additional
                memmap_array.flush()
                
                meta_info = {'shape': (test_rows, 7), 'dtype': str(np.float64)}
                np.save(str(memmap_path.with_suffix('.meta')), meta_info)
                return memmap_array
            
            # 実際のParquetファイルからストリーミング変換
            import pyarrow.parquet as pq
            
            # まず総行数を推定し、カラム数を確認
            total_rows = 0
            actual_columns = None
            for pf in parquet_files:
                parquet_file = pq.ParquetFile(str(pf))
                total_rows += parquet_file.metadata.num_rows
                if actual_columns is None:
                    # 最初のファイルからカラム数を確認
                    actual_columns = len(parquet_file.schema)
                    logger.debug(f"実際のカラム数: {actual_columns}")
            
            # memmap作成
            memmap_array = np.memmap(str(memmap_path), dtype=np.float64, mode='w+', 
                                shape=(total_rows, 7))
            
            # チャンク単位で変換（メモリ効率的）
            current_row = 0
            for parquet_file in parquet_files:
                try:
                    # Polarsで読み込み（シンプルな方法）
                    import polars as pl
                    df = pl.read_parquet(str(parquet_file))
                    
                    # DataFrameを数値配列に変換
                    # 各カラムを数値に変換
                    for col in df.columns:
                        if df[col].dtype == pl.Datetime:
                            # Timestampをエポック秒に変換
                            df = df.with_columns(pl.col(col).dt.epoch("s").alias(col))
                        elif not df[col].dtype.is_numeric():
                            # 非数値型は0で置換
                            df = df.with_columns(pl.lit(0).alias(col))
                    
                    batch_numpy = df.to_numpy().astype(np.float64)
                    
                    # カラム数調整（最初の7列のみ使用、または0でパディング）
                    if batch_numpy.shape[1] > 7:
                        batch_numpy = batch_numpy[:, :7]
                        if current_row == 0:  # 最初のバッチでのみログ
                            logger.debug(f"カラムトリミング: {actual_columns}列 → 7列")
                    elif batch_numpy.shape[1] < 7:
                        pad_cols = 7 - batch_numpy.shape[1]
                        batch_numpy = np.pad(batch_numpy, ((0, 0), (0, pad_cols)), 
                                        mode='constant', constant_values=0)
                        if current_row == 0:  # 最初のバッチでのみログ
                            logger.debug(f"カラムパディング: {batch_numpy.shape[1] - pad_cols}列 → 7列")
                    
                    # memmapに書き込み
                    batch_rows = min(batch_numpy.shape[0], total_rows - current_row)
                    if batch_rows > 0:
                        memmap_array[current_row:current_row + batch_rows] = batch_numpy[:batch_rows]
                        current_row += batch_rows
                        memmap_array.flush()
                    
                    # メモリ解放
                    del batch_numpy, df
                    
                    if current_row % 100000 == 0:
                        logger.debug(f"進捗: {current_row}/{total_rows}行処理済み")
                        
                except Exception as e:
                    logger.error(f"Parquetファイル処理エラー {parquet_file}: {e}")
                    continue
            
            # メタ情報保存
            actual_shape = (current_row, 7)
            meta_info = {'shape': actual_shape, 'dtype': str(np.float64)}
            np.save(str(memmap_path.with_suffix('.meta')), meta_info)
            
            # サイズ調整が必要な場合
            if current_row < total_rows:
                memmap_array = np.memmap(str(memmap_path), dtype=np.float64, 
                                    mode='r+', shape=actual_shape)
            
            logger.info(f"Parquet→memmap変換完了: {timeframe}, shape={actual_shape}")
            return memmap_array
            
        except Exception as e:
            logger.error(f"memmap作成エラー: {e}")
            raise

# =============================================================================
# WindowManager（薄い実装・5%リソース）
# =============================================================================

class WindowManager:
    """ウィンドウ管理クラス - 基本スライディング生成"""
    
    def __init__(self, window_size: int = 100, overlap: float = 0.5):
        self.window_size = window_size
        self.overlap = overlap
        self.step_size = int(window_size * (1 - overlap))
        logger.info(f"WindowManager初期化: window_size={window_size}, overlap={overlap}")
    
    def generate_window_indices(self, data_length: int) -> List[Tuple[int, int]]:
        """ウィンドウのインデックス範囲を生成"""
        indices = []
        for i in range(0, data_length - self.window_size + 1, self.step_size):
            start_idx = i
            end_idx = min(i + self.window_size, data_length)
            indices.append((start_idx, end_idx))
        return indices
    
    def create_sliding_windows(self, memmap_data: np.memmap) -> Iterator[Tuple[int, np.ndarray]]:
        """スライディングウィンドウを生成するジェネレータ"""
        for window_idx, (start_idx, end_idx) in enumerate(self.generate_window_indices(memmap_data.shape[0])):
            window_data = memmap_data[start_idx:end_idx]
            yield window_idx, window_data

# =============================================================================
# MemoryManager（監視のみ・2%リソース）
# =============================================================================

class MemoryManager:
    """メモリ管理クラス - 監視とGCのみ"""
    
    def __init__(self):
        self.ram_limit_gb = HARDWARE_SPEC['ram_limit']
        logger.info(f"MemoryManager初期化: RAM制限={self.ram_limit_gb}GB")
    
    def check_memory_status(self) -> Dict[str, Any]:
        """メモリ状態チェック"""
        import psutil
        memory = psutil.virtual_memory()
        current_gb = memory.used / (1024**3)
        usage_percent = (current_gb / self.ram_limit_gb) * 100
        
        if usage_percent > 80:
            logger.warning(f"メモリ使用量警告: {current_gb:.2f}GB ({usage_percent:.1f}%)")
        
        return {
            'current_gb': current_gb,
            'usage_percent': usage_percent,
            'status': 'warning' if usage_percent > 80 else 'normal'
        }
    
    def force_garbage_collection(self):
        """強制ガベージコレクション"""
        import gc
        gc.collect()

# =============================================================================
# OutputManager（機能的最小限・3%リソース）
# =============================================================================

class OutputManager:
    """出力管理クラス - 基本保存機能"""
    
    def __init__(self, output_path: Path = None):
        self.output_path = output_path or Path('./output')
        self.output_path.mkdir(parents=True, exist_ok=True)
        logger.info(f"OutputManager初期化: {self.output_path}")
    
    def save_features(self, features_dict: Dict[str, np.ndarray], filename: str) -> Path:
        """特徴量をParquetファイルとして保存"""
        try:
            if not filename.endswith('.parquet'):
                filename += '.parquet'
            output_path = self.output_path / filename
            
            # 特徴量値のクリーニング
            cleaned_features = {}
            for name, values in features_dict.items():
                cleaned_values = np.nan_to_num(values, nan=0.0, posinf=1e10, neginf=-1e10)
                cleaned_features[name] = cleaned_values
            
            # Polars DataFrameに変換して保存
            df = pl.DataFrame(cleaned_features)
            df.write_parquet(str(output_path), compression='snappy')
            
            logger.info(f"特徴量保存完了: {filename}")
            return output_path
            
        except Exception as e:
            logger.error(f"特徴量保存エラー: {e}")
            raise

# =============================================================================
# Calculator（80%リソース・2000-2500行の核心部分）
# =============================================================================
class Calculator:
    """
    特徴量計算エンジン - 4段階式・適応型数値安定性エンジン搭載
    80%リソース集中・Golden Ratio最適化・ロバスト統計対応
    """
    
    def __init__(self, window_manager=None, memory_manager=None):
        self.window_manager = window_manager
        self.memory_manager = memory_manager
        
        # 数学的定数・パラメータ
        self.golden_ratio = GOLDEN_RATIO
        self.fibonacci_ratios = FIBONACCI_RATIOS
        self.params = CALC_PARAMS
        
        # 4段階式数値安定性エンジン設定
        self.stability_config = {
            'quality_thresholds': {
                'level_1': 0.7,  # 軽量介入
                'level_2': 0.5,  # 中量介入  
                'level_3': 0.3,  # 重量介入
                'level_4': 0.0   # 最終手段
            },
            'eps_base': 1e-12,
            'condition_threshold': 1e12,
            'outlier_threshold': 5.0,
            'winsorize_limits': (0.01, 0.99),
            'mad_factor': 1.4826,  # 正規分布での標準偏差相当
            'nan_threshold': 0.3,
            'inf_threshold': 0.05,
            'diversity_min': 0.1,
            'range_violation_threshold': 3.0
        }
        
        # 計算統計
        self.stats = {
            'total_calculations': 0,
            'successful_calculations': 0,
            'stabilization_applications': {1: 0, 2: 0, 3: 0, 4: 0},
            'quality_scores': [],
            'computation_times': [],
            'fallback_count': 0
        }
        
        logger.info("Calculator初期化完了 - 4段階式・適応型数値安定性エンジン搭載")
    
    # =========================================================================
    # 4段階式・適応型数値安定性エンジン - 核心システム
    # =========================================================================
    
    def _calculate_quality_score(self, values: np.ndarray, feature_name: str = "") -> float:
        """
        数値配列の品質スコア算出 (0.0-1.0)
        評価基準: NaN率、Inf率、値の多様性、理論範囲からの逸脱度
        """
        if len(values) == 0:
            return 0.0
        
        try:
            # 基本統計
            total_count = len(values)
            finite_mask = np.isfinite(values)
            finite_count = np.sum(finite_mask)
            
            # 1. NaN率評価 (30%基準)
            nan_ratio = np.sum(np.isnan(values)) / total_count
            nan_score = max(0.0, 1.0 - (nan_ratio / self.stability_config['nan_threshold']))
            
            # 2. Inf率評価 (5%基準)
            inf_ratio = np.sum(np.isinf(values)) / total_count
            inf_score = max(0.0, 1.0 - (inf_ratio / self.stability_config['inf_threshold']))
            
            # 3. 値の多様性評価
            if finite_count > 1:
                finite_values = values[finite_mask]
                unique_ratio = len(np.unique(finite_values)) / finite_count
                diversity_score = min(1.0, unique_ratio / self.stability_config['diversity_min'])
            else:
                diversity_score = 0.0
            
            # 4. 理論的範囲からの逸脱度評価
            if finite_count > 0:
                finite_values = values[finite_mask]
                q1, q3 = np.percentile(finite_values, [25, 75])
                iqr = q3 - q1
                
                if iqr > self.stability_config['eps_base']:
                    lower_bound = q1 - self.stability_config['range_violation_threshold'] * iqr
                    upper_bound = q3 + self.stability_config['range_violation_threshold'] * iqr
                    outlier_ratio = np.sum((finite_values < lower_bound) | (finite_values > upper_bound)) / finite_count
                    range_score = max(0.0, 1.0 - outlier_ratio)
                else:
                    range_score = 0.5  # 全て同一値の場合
            else:
                range_score = 0.0
            
            # 総合品質スコア (重み付け平均)
            quality_score = (
                0.3 * nan_score +      # NaN率最重要
                0.2 * inf_score +      # Inf率重要
                0.2 * diversity_score + # 多様性重要
                0.3 * range_score      # 範囲妥当性最重要
            )
            
            # 統計記録
            self.stats['quality_scores'].append(quality_score)
            
            return np.clip(quality_score, 0.0, 1.0)
            
        except Exception as e:
            logger.debug(f"品質スコア計算エラー {feature_name}: {e}")
            return 0.0
    
    def _determine_intervention_level(self, quality_score: float) -> int:
        """品質スコアに基づく介入レベル決定"""
        config = self.stability_config['quality_thresholds']
        
        if quality_score > config['level_1']:
            return 1  # 軽量介入
        elif quality_score > config['level_2']:
            return 2  # 中量介入
        elif quality_score > config['level_3']:
            return 3  # 重量介入
        else:
            return 4  # 最終手段
    
    def _apply_stabilization(self, values: np.ndarray, level: int, feature_name: str) -> np.ndarray:
        """段階的介入ロジックの実行"""
        try:
            if level == 1:
                return self._level_1_stabilization(values)
            elif level == 2:
                return self._level_2_stabilization(values)
            elif level == 3:
                return self._level_3_stabilization(values)
            else:  # level == 4
                return self._level_4_stabilization(values, feature_name)
        except Exception as e:
            logger.error(f"安定化処理エラー Level {level} for {feature_name}: {e}")
            return self._level_4_stabilization(values, feature_name)
    
    def _level_1_stabilization(self, values: np.ndarray) -> np.ndarray:
        """第1段階: 軽量介入 - 基本的なNaN/Infの置換、極端な値のクリッピング"""
        result = values.copy()
        
        # NaN/Inf処理
        nan_mask = np.isnan(result)
        inf_mask = np.isinf(result)
        
        if np.any(nan_mask):
            result[nan_mask] = 0.0
        
        if np.any(inf_mask):
            finite_values = result[np.isfinite(result)]
            if len(finite_values) > 0:
                result[inf_mask & (result > 0)] = np.percentile(finite_values, 99)
                result[inf_mask & (result < 0)] = np.percentile(finite_values, 1)
            else:
                result[inf_mask] = 0.0
        
        # 極端値クリッピング
        result = np.clip(result, -1e10, 1e10)
        
        return result
    
    def _level_2_stabilization(self, values: np.ndarray) -> np.ndarray:
        """第2段階: 中量介入 - ウィンソライゼーション、スケール正則化"""
        result = self._level_1_stabilization(values)
        
        # ウィンソライゼーション
        finite_mask = np.isfinite(result)
        if np.sum(finite_mask) > 10:
            finite_values = result[finite_mask]
            lower_limit, upper_limit = self.stability_config['winsorize_limits']
            
            p_lower = np.percentile(finite_values, lower_limit * 100)
            p_upper = np.percentile(finite_values, upper_limit * 100)
            
            result = np.clip(result, p_lower, p_upper)
        
        # スケール正則化
        std_val = np.nanstd(result)
        if std_val > 100:  # 異常に大きなスケール
            result = result / (std_val / 10)
        
        return result
    
    def _level_3_stabilization(self, values: np.ndarray) -> np.ndarray:
        """第3段階: 重量介入 - ロバスト統計(中央値/MAD)完全切り替え"""
        result = self._level_1_stabilization(values)
        
        finite_mask = np.isfinite(result)
        if np.sum(finite_mask) > 5:
            finite_values = result[finite_mask]
            
            # ロバスト統計による中心化・正規化
            median_val = np.median(finite_values)
            mad = np.median(np.abs(finite_values - median_val))
            
            if mad > self.stability_config['eps_base']:
                # MADベース正規化
                normalized_mad = mad * self.stability_config['mad_factor']
                result = (result - median_val) / normalized_mad
                
                # ロバスト範囲制限 (±5σ相当)
                result = np.clip(result, -5.0, 5.0)
            else:
                # 全て同一値の場合
                result = np.where(finite_mask, 0.0, result)
        
        return result
    
    def _level_4_stabilization(self, values: np.ndarray, feature_name: str) -> np.ndarray:
        """第4段階: 最終手段 - 代替アルゴリズムフォールバック"""
        logger.warning(f"Feature '{feature_name}': 最終手段フォールバック実行")
        self.stats['fallback_count'] += 1
        
        # 単純移動平均による平滑化フォールバック
        n = len(values)
        if n < 5:
            return np.zeros(n)
        
        # 基本的な異常値除去
        finite_mask = np.isfinite(values)
        if np.sum(finite_mask) == 0:
            return np.zeros(n)
        
        # 中央値フィルタリング
        try:
            from scipy.ndimage import median_filter
            window_size = min(5, n // 4)
            if window_size >= 3:
                result = median_filter(values.astype(float), size=window_size, mode='constant')
            else:
                # 最も単純なフォールバック
                median_val = np.nanmedian(values)
                result = np.full(n, median_val if np.isfinite(median_val) else 0.0)
        except Exception:
            # 完全フォールバック: ゼロ埋め
            result = np.zeros(n)
        
        return result
    
    def _stabilized_calculation(self, func, *args, feature_name: str = "", **kwargs) -> np.ndarray:
        """
        安定化計算のメインラッパー
        計算→品質評価→介入レベル決定→安定化処理→ログ出力
        """
        start_time = time.time()
        self.stats['total_calculations'] += 1
        
        try:
            # 初期計算実行
            result = func(*args, **kwargs)
            
            if not isinstance(result, np.ndarray):
                result = np.asarray(result, dtype=np.float64)
            
            # 品質スコア評価
            quality_score = self._calculate_quality_score(result, feature_name)
            
            # 介入レベル決定
            intervention_level = self._determine_intervention_level(quality_score)
            
            # 安定化処理適用
            if intervention_level > 1 or quality_score < 0.9:
                result = self._apply_stabilization(result, intervention_level, feature_name)
                self.stats['stabilization_applications'][intervention_level] += 1
                
                # ログ出力
                level_names = {1: "軽量", 2: "中量", 3: "重量", 4: "最終手段"}
                logger.info(f"Feature '{feature_name}': Quality score {quality_score:.3f}. "
                          f"Applying Level {intervention_level} ({level_names[intervention_level]}) stabilization.")
            
            self.stats['successful_calculations'] += 1
            computation_time = time.time() - start_time
            self.stats['computation_times'].append(computation_time)
            
            return result
            
        except Exception as e:
            logger.error(f"安定化計算エラー '{feature_name}': {e}")
            # 緊急フォールバック
            if args and hasattr(args[0], '__len__'):
                fallback_result = np.zeros(len(args[0]))
                return self._apply_stabilization(fallback_result, 4, feature_name)
            return np.array([])
    
    # =========================================================================
    # ユーティリティメソッド群
    # =========================================================================
    
    def _ensure_numpy_array(self, data: Union[np.ndarray, Any]) -> np.ndarray:
        """データをNumPy配列に変換"""
        if isinstance(data, np.ndarray):
            return data.flatten() if data.ndim > 1 else data
        return np.asarray(data, dtype=np.float64)
    
    def _safe_divide(self, numerator: np.ndarray, denominator: np.ndarray) -> np.ndarray:
        """安全な除算（条件数チェック付き）"""
        denominator_safe = np.where(
            np.abs(denominator) < self.stability_config['eps_base'],
            self.stability_config['eps_base'],
            denominator
        )
        
        result = np.divide(numerator, denominator_safe)
        condition_number = np.abs(numerator) / (np.abs(denominator_safe) + self.stability_config['eps_base'])
        
        result = np.where(
            condition_number > self.stability_config['condition_threshold'],
            np.sign(result) * self.stability_config['condition_threshold'],
            result
        )
        
        return result
    
    @staticmethod
    @njit(cache=True)
    def _rolling_mean_numba(data: np.ndarray, period: int) -> np.ndarray:
        """高速ローリング平均（Numba最適化）"""
        n = len(data)
        result = np.full(n, np.nan)
        
        if period <= 0 or period > n:
            return result
        
        window_sum = np.sum(data[:period])
        result[period-1] = window_sum / period
        
        for i in range(period, n):
            window_sum = window_sum - data[i-period] + data[i]
            result[i] = window_sum / period
        
        return result
    
    @staticmethod
    @njit(cache=True)
    def _rolling_std_numba(data: np.ndarray, period: int) -> np.ndarray:
        """高速ローリング標準偏差（Numba最適化）"""
        n = len(data)
        result = np.full(n, np.nan)
        
        if period <= 1 or period > n:
            return result
        
        for i in range(period-1, n):
            window_data = data[i-period+1:i+1]
            mean_val = np.mean(window_data)
            var_val = np.mean((window_data - mean_val) ** 2)
            result[i] = np.sqrt(var_val)
        
        return result
    
    @staticmethod
    @njit(cache=True)
    def _ema_numba(data: np.ndarray, period: int) -> np.ndarray:
        """高速指数移動平均（Numba最適化）"""
        n = len(data)
        result = np.full(n, np.nan)
        
        if period <= 0 or period > n:
            return result
        
        alpha = 2.0 / (period + 1.0)
        result[0] = data[0]
        
        for i in range(1, n):
            result[i] = alpha * data[i] + (1 - alpha) * result[i-1]
        
        return result
    
    # =========================================================================
    # ADX・基本オシレーター群（独自数学実装）
    # =========================================================================
    
    def calculate_adx_features(self, high: np.ndarray, low: np.ndarray, close: np.ndarray) -> Dict[str, np.ndarray]:
        """ADX関連特徴量の計算（独自実装強化版）"""
        features = {}
        
        high = self._ensure_numpy_array(high)
        low = self._ensure_numpy_array(low)
        close = self._ensure_numpy_array(close)
        
        try:
            for period in self.params['adx_periods']:
                # 直接_calculate_enhanced_adxを呼び出し（_stabilized_calculationを通さない）
                adx_result = self._calculate_enhanced_adx(high, low, close, period)
                
                if isinstance(adx_result, dict):
                    # 各値に対して個別に安定化処理を適用
                    for key, value in adx_result.items():
                        features[key] = self._stabilized_calculation(
                            lambda x: x, value,
                            feature_name=key
                        )
                else:
                    # フォールバック
                    n = len(close)
                    features[f'e1_adx_{period}'] = np.full(n, np.nan)
                    features[f'e1_di_plus_{period}'] = np.full(n, np.nan)
                    features[f'e1_di_minus_{period}'] = np.full(n, np.nan)
                    
        except Exception as e:
            logger.error(f"ADX計算エラー: {e}")
            n = len(close)
            for period in self.params['adx_periods']:
                features[f'e1_adx_{period}'] = np.full(n, np.nan)
                features[f'e1_di_plus_{period}'] = np.full(n, np.nan)
                features[f'e1_di_minus_{period}'] = np.full(n, np.nan)
        
        return features
    
    def _calculate_enhanced_adx(self, high: np.ndarray, low: np.ndarray, 
                               close: np.ndarray, period: int) -> Dict[str, np.ndarray]:
        """強化版ADX計算（Golden Ratio調整・ロバスト統計）"""
        
        if len(close) < 2:
            n = len(close)
            return {
                f'e1_adx_{period}': np.full(n, np.nan),
                f'e1_di_plus_{period}': np.full(n, np.nan),
                f'e1_di_minus_{period}': np.full(n, np.nan)
            }
        
        # True Range計算（条件数チェック付き）
        prev_close = np.roll(close, 1)
        prev_close[0] = close[0]
        
        tr1 = high - low
        tr2 = np.abs(high - prev_close)
        tr3 = np.abs(low - prev_close)
        true_range = np.maximum(tr1, np.maximum(tr2, tr3))
        
        # 異常値処理
        true_range = np.clip(true_range, self.stability_config['eps_base'], 
                           np.percentile(true_range[np.isfinite(true_range)], 99.5))
        
        # 方向性移動計算
        up_move = high - np.roll(high, 1)
        down_move = np.roll(low, 1) - low
        up_move[0] = 0
        down_move[0] = 0
        
        # DM+とDM-（ロバスト化）
        dm_plus = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
        dm_minus = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
        
        # Golden Ratio調整スムージング
        golden_smoothing_factor = 2.0 / (period * self.golden_ratio + 1.0)
        
        atr_smooth = self._adaptive_ema(true_range, period, golden_smoothing_factor)
        dm_plus_smooth = self._adaptive_ema(dm_plus, period, golden_smoothing_factor)
        dm_minus_smooth = self._adaptive_ema(dm_minus, period, golden_smoothing_factor)
        
        # DI+とDI-の計算（条件数チェック付き）
        di_plus = 100.0 * self._safe_divide(dm_plus_smooth, atr_smooth)
        di_minus = 100.0 * self._safe_divide(dm_minus_smooth, atr_smooth)
        
        # DX計算（ロバスト統計適用）
        di_sum = di_plus + di_minus
        di_diff_abs = np.abs(di_plus - di_minus)
        dx = 100.0 * self._safe_divide(di_diff_abs, di_sum)
        
        # ADX計算（段階的安定化）
        adx = self._adaptive_ema(dx, period, golden_smoothing_factor)
        
        # 異常値クリッピング
        adx = np.clip(adx, 0, 100)
        di_plus = np.clip(di_plus, 0, 100)
        di_minus = np.clip(di_minus, 0, 100)
        
        return {
            f'e1_adx_{period}': adx,
            f'e1_di_plus_{period}': di_plus,
            f'e1_di_minus_{period}': di_minus
        }
    
    def _adaptive_ema(self, data: np.ndarray, period: int, base_alpha: float = None) -> np.ndarray:
        """適応的指数移動平均（独自実装）"""
        if base_alpha is None:
            base_alpha = 2.0 / (period + 1.0)
        
        # ボラティリティベース適応調整
        volatility = self._stabilized_calculation(
            self._rolling_std_numba, data, min(period, len(data)//4),
            feature_name="volatility_adaptive"
        )
        volatility_normalized = volatility / (np.nanmedian(volatility) + self.stability_config['eps_base'])
        
        # Golden Ratio調整係数
        adaptive_alpha = base_alpha * (1.0 + volatility_normalized / self.golden_ratio)
        adaptive_alpha = np.clip(adaptive_alpha, base_alpha * 0.5, base_alpha * 2.0)
        
        return self._stabilized_calculation(
            self._adaptive_ema_numba, data, adaptive_alpha,
            feature_name="adaptive_ema"
        )
    
    @staticmethod
    @njit(cache=True)
    def _adaptive_ema_numba(data: np.ndarray, alpha_array: np.ndarray) -> np.ndarray:
        """適応的EMA（Numba最適化）"""
        n = len(data)
        result = np.full(n, np.nan)
        
        if n == 0:
            return result
        
        result[0] = data[0]
        for i in range(1, n):
            alpha = alpha_array[i] if i < len(alpha_array) else alpha_array[-1]
            alpha = max(0.01, min(0.99, alpha))  # 安定性制約
            result[i] = alpha * data[i] + (1 - alpha) * result[i-1]
        
        return result
    
    def calculate_rsi_features(self, close: np.ndarray) -> Dict[str, np.ndarray]:
        """RSI特徴量（独自強化実装）"""
        features = {}
        close = self._ensure_numpy_array(close)
        
        try:
            for period in self.params['rsi_periods']:
                # 直接計算（_stabilized_calculationを通さない）
                rsi_result = self._calculate_enhanced_rsi(close, period)
                
                if isinstance(rsi_result, dict):
                    # 各値に対して個別に安定化処理
                    for key, value in rsi_result.items():
                        features[key] = self._stabilized_calculation(
                            lambda x: x, value,
                            feature_name=key
                        )
                else:
                    # フォールバック
                    n = len(close)
                    features[f'e1_rsi_{period}'] = np.full(n, np.nan)
                    
        except Exception as e:
            logger.error(f"RSI計算エラー: {e}")
            
        return features
    
    def _calculate_enhanced_rsi(self, close: np.ndarray, period: int) -> Dict[str, np.ndarray]:
        """強化版RSI計算"""
        delta = np.diff(close, prepend=close[0])
        
        gain = np.where(delta > 0, delta, 0)
        loss = np.where(delta < 0, -delta, 0)
        
        # Golden Ratioベース適応スムージング
        avg_gain = self._adaptive_ema(gain, period)
        avg_loss = self._adaptive_ema(loss, period)
        
        # RSI計算（ロバスト化）
        rs = self._safe_divide(avg_gain, avg_loss)
        rsi = 100 - (100 / (1 + rs))
        
        # 追加特徴量
        rsi_momentum = np.diff(rsi, prepend=rsi[0])
        rsi_divergence = self._calculate_price_rsi_divergence(close, rsi)
        
        return {
            f'e1_rsi_{period}': rsi,
            f'e1_rsi_momentum_{period}': rsi_momentum,
            f'e1_rsi_divergence_{period}': rsi_divergence
        }
    
    def _calculate_price_rsi_divergence(self, price: np.ndarray, rsi: np.ndarray) -> np.ndarray:
        """価格-RSIダイバージェンス検出（独自実装）"""
        # 短期トレンドの検出
        price_trend = self._stabilized_calculation(
            self._rolling_mean_numba, np.diff(price, prepend=price[0]), 10,
            feature_name="price_trend"
        )
        rsi_trend = self._stabilized_calculation(
            self._rolling_mean_numba, np.diff(rsi, prepend=rsi[0]), 10,
            feature_name="rsi_trend"
        )
        
        # 正規化ダイバージェンス計算
        price_trend_norm = price_trend / (np.nanstd(price_trend) + self.stability_config['eps_base'])
        rsi_trend_norm = rsi_trend / (np.nanstd(rsi_trend) + self.stability_config['eps_base'])
        
        divergence = price_trend_norm - rsi_trend_norm
        return divergence
    
    # =========================================================================
    # 移動平均線・トレンド分析群（並列化対応）
    # =========================================================================
    
    def calculate_moving_averages(self, close: np.ndarray) -> Dict[str, np.ndarray]:
        """移動平均線特徴量（並列化対応強化版）"""
        features = {}
        close = self._ensure_numpy_array(close)
        
        try:
            # 基本移動平均群
            ma_features = self._calculate_basic_moving_averages(close)
            features.update(ma_features)
            
            # 高度移動平均群（並列化可能）
            advanced_ma_features = self._calculate_advanced_moving_averages_parallel(close)
            features.update(advanced_ma_features)
            
            # トレンド分析特徴量
            trend_features = self._calculate_trend_analysis_features(close)
            features.update(trend_features)
            
        except Exception as e:
            logger.error(f"移動平均計算エラー: {e}")
            
        return features
    
    def _calculate_basic_moving_averages(self, close: np.ndarray) -> Dict[str, np.ndarray]:
        """基本移動平均計算"""
        features = {}
        
        for period in self.params['ma_short_periods'] + self.params['ma_long_periods']:
            # SMA
            sma = self._stabilized_calculation(
                self._rolling_mean_numba, close, period,
                feature_name=f"e1_sma_{period}"
            )
            features[f'e1_sma_{period}'] = sma
            
            # EMA
            ema = self._stabilized_calculation(
                self._ema_numba, close, period,
                feature_name=f"e1_ema_{period}"
            )
            features[f'e1_ema_{period}'] = ema
            
            # 価格との乖離率
            sma_deviation = 100 * self._safe_divide(close - sma, sma)
            ema_deviation = 100 * self._safe_divide(close - ema, ema)
            
            features[f'e1_sma_deviation_{period}'] = self._stabilized_calculation(
                lambda x: x, sma_deviation, feature_name=f"e1_sma_deviation_{period}"
            )
            features[f'e1_ema_deviation_{period}'] = self._stabilized_calculation(
                lambda x: x, ema_deviation, feature_name=f"e1_ema_deviation_{period}"
            )
        
        return features
    
    def _calculate_advanced_moving_averages_parallel(self, close: np.ndarray) -> Dict[str, np.ndarray]:
        """高度移動平均計算（並列化対応）"""
        features = {}
        
        # Golden Ratioオーバーラップ並列化の適用チェック
        if len(close) > 1000 and self._should_use_parallel_processing():
            features.update(self._calculate_wma_parallel(close))
            features.update(self._calculate_hma_parallel(close))
            features.update(self._calculate_kama_parallel(close))
        else:
            # 逐次処理
            features.update(self._calculate_wma_sequential(close))
            features.update(self._calculate_hma_sequential(close))
            features.update(self._calculate_kama_sequential(close))
        
        return features
    
    def _should_use_parallel_processing(self) -> bool:
        """並列処理使用判定"""
        import psutil
        cpu_count = psutil.cpu_count(logical=False) or 1
        available_memory_gb = psutil.virtual_memory().available / (1024**3)
        
        return cpu_count >= 4 and available_memory_gb >= 8
    
    def _calculate_wma_parallel(self, close: np.ndarray) -> Dict[str, np.ndarray]:
        """WMA並列計算（ゴールデンオーバーラップ適用）"""
        features = {}
        
        try:
            if len(close) < 500:  # 小データは逐次処理
                return self._calculate_wma_sequential(close)
            
            # ゴールデンオーバーラップパラメータ
            max_lookback = max(self.params['ma_long_periods'])
            overlap_size = int(max_lookback * self.golden_ratio)  # Golden Ratio倍オーバーラップ
            chunk_size = len(close) // HARDWARE_SPEC['cpu_cores']
            chunk_size = max(chunk_size, max_lookback * 4)  # 最小チャンクサイズ保証
            
            # 並列処理実行
            with ProcessPoolExecutor(max_workers=HARDWARE_SPEC['cpu_cores']) as executor:
                futures = []
                
                for period in self.params['ma_long_periods']:
                    future = executor.submit(
                        self._wma_chunk_processor,
                        close, period, chunk_size, overlap_size
                    )
                    futures.append((future, period))
                
                for future, period in futures:
                    try:
                        result = future.result(timeout=30)
                        features[f'e1_wma_{period}'] = self._stabilized_calculation(
                            lambda x: x, result, feature_name=f"e1_wma_{period}"
                        )
                        self.stats['successful_calculations'] += 1
                    except Exception as e:
                        logger.warning(f"WMA並列計算失敗 period={period}: {e}")
                        # フォールバック
                        features[f'e1_wma_{period}'] = self._calculate_wma_sequential_single(close, period)
                        
        except Exception as e:
            logger.error(f"WMA並列計算エラー: {e}")
            return self._calculate_wma_sequential(close)
        
        return features
    
    @staticmethod
    def _wma_chunk_processor(data: np.ndarray, period: int, chunk_size: int, overlap_size: int) -> np.ndarray:
        """WMAチャンク処理（静的メソッド・プロセス間共有用）"""
        n = len(data)
        result = np.full(n, np.nan)
        
        # チャンク分割
        chunks = []
        for i in range(0, n, chunk_size):
            start = max(0, i - overlap_size)
            end = min(n, i + chunk_size + overlap_size)
            chunks.append((start, end, i))
        
        # 各チャンクでWMA計算
        for start, end, original_start in chunks:
            chunk_data = data[start:end]
            chunk_wma = Calculator._wma_numba_static(chunk_data, period)
            
            # トリム処理（オーバーラップ部分削除）
            trim_start = overlap_size if original_start > 0 else 0
            trim_end = len(chunk_wma) - overlap_size if end < n else len(chunk_wma)
            
            if trim_start < trim_end:
                result_start = original_start
                result_end = min(n, original_start + (trim_end - trim_start))
                result[result_start:result_end] = chunk_wma[trim_start:trim_end]
        
        return result
    
    @staticmethod
    @njit(cache=True)
    def _wma_numba_static(data: np.ndarray, period: int) -> np.ndarray:
        """WMA計算（静的Numba版）"""
        n = len(data)
        result = np.full(n, np.nan)
        
        if period > n or period <= 0:
            return result
        
        weights = np.arange(1, period + 1)
        weight_sum = np.sum(weights)
        
        for i in range(period - 1, n):
            weighted_sum = 0.0
            for j in range(period):
                weighted_sum += data[i - period + 1 + j] * weights[j]
            result[i] = weighted_sum / weight_sum
        
        return result
    
    def _calculate_wma_sequential(self, close: np.ndarray) -> Dict[str, np.ndarray]:
        """WMA逐次計算"""
        features = {}
        for period in self.params['ma_long_periods']:
            features[f'e1_wma_{period}'] = self._calculate_wma_sequential_single(close, period)
        return features
    
    def _calculate_wma_sequential_single(self, close: np.ndarray, period: int) -> np.ndarray:
        """単一WMA計算"""
        return self._stabilized_calculation(
            self._wma_numba_static, close, period,
            feature_name=f"e1_wma_{period}"
        )
    
    def _calculate_hma_parallel(self, close: np.ndarray) -> Dict[str, np.ndarray]:
        """HMA並列計算（簡略化）"""
        # 実装簡素化のため逐次処理にフォールバック
        return self._calculate_hma_sequential(close)
    
    def _calculate_hma_sequential(self, close: np.ndarray) -> Dict[str, np.ndarray]:
        """HMA逐次計算"""
        features = {}
        for period in [21, 34, 55]:  # Golden Ratio調整期間
            hma = self._stabilized_calculation(
                self._hma_numba_static, close, period,
                feature_name=f"e1_hma_{period}"
            )
            features[f'e1_hma_{period}'] = hma
        return features
    
    @staticmethod
    @njit(cache=True)
    def _hma_numba_static(data: np.ndarray, period: int) -> np.ndarray:
        """HMA計算（Numba最適化）"""
        n = len(data)
        if n < period:
            return np.full(n, np.nan)
        
        half_period = period // 2
        sqrt_period = int(np.sqrt(period))
        
        # WMA(period/2)
        wma_half = np.full(n, np.nan)
        for i in range(half_period - 1, n):
            weighted_sum = 0.0
            weight_sum = 0.0
            for j in range(half_period):
                weight = half_period - j
                weighted_sum += data[i - j] * weight
                weight_sum += weight
            wma_half[i] = weighted_sum / weight_sum
        
        # WMA(period)
        wma_full = np.full(n, np.nan)
        for i in range(period - 1, n):
            weighted_sum = 0.0
            weight_sum = 0.0
            for j in range(period):
                weight = period - j
                weighted_sum += data[i - j] * weight
                weight_sum += weight
            wma_full[i] = weighted_sum / weight_sum
        
        # Raw HMA = 2*WMA(half) - WMA(full)
        raw_hma = 2 * wma_half - wma_full
        
        # WMA(sqrt(period)) of Raw HMA
        hma = np.full(n, np.nan)
        for i in range(period - 1 + sqrt_period - 1, n):
            if i - sqrt_period + 1 >= 0:
                weighted_sum = 0.0
                weight_sum = 0.0
                for j in range(sqrt_period):
                    if i - j >= 0 and not np.isnan(raw_hma[i - j]):
                        weight = sqrt_period - j
                        weighted_sum += raw_hma[i - j] * weight
                        weight_sum += weight
                if weight_sum > 0:
                    hma[i] = weighted_sum / weight_sum
        
        return hma
    
    def _calculate_kama_parallel(self, close: np.ndarray) -> Dict[str, np.ndarray]:
        """KAMA並列計算（簡略化）"""
        return self._calculate_kama_sequential(close)
    
    def _calculate_kama_sequential(self, close: np.ndarray) -> Dict[str, np.ndarray]:
        """KAMA逐次計算"""
        features = {}
        for period in [21, 34]:  # Golden Ratio調整
            kama_result = self._stabilized_calculation(
                self._kama_numba_static, close, period,
                feature_name=f"e1_kama_{period}"
            )
            features[f'e1_kama_{period}'] = kama_result
        return features
    
    @staticmethod
    @njit(cache=True)
    def _kama_numba_static(data: np.ndarray, period: int) -> np.ndarray:
        """KAMA計算（Numba最適化）"""
        n = len(data)
        result = np.full(n, np.nan)
        
        if n < period + 1:
            return result
        
        result[period] = np.mean(data[:period + 1])
        
        fast_alpha = 2.0 / 3.0
        slow_alpha = 2.0 / 31.0
        
        for i in range(period + 1, n):
            direction = abs(data[i] - data[i - period])
            
            volatility = 0.0
            for j in range(period):
                volatility += abs(data[i - j] - data[i - j - 1])
            
            if volatility > 1e-10:
                er = direction / volatility
            else:
                er = 1.0
            
            sc = er * (fast_alpha - slow_alpha) + slow_alpha
            alpha = sc * sc
            
            result[i] = result[i - 1] + alpha * (data[i] - result[i - 1])
        
        return result
    
    def _calculate_trend_analysis_features(self, close: np.ndarray) -> Dict[str, np.ndarray]:
        """トレンド分析特徴量"""
        features = {}
        
        # トレンド強度（Golden Ratio期間）
        for period in [int(21), int(34 * self.golden_ratio)]:
            trend_slope = self._calculate_trend_slope(close, period)
            trend_strength = np.abs(trend_slope)
            trend_consistency = self._calculate_trend_consistency(close, period)
            
            features[f'e1_trend_slope_{period}'] = self._stabilized_calculation(
                lambda x: x, trend_slope, feature_name=f"e1_trend_slope_{period}"
            )
            features[f'e1_trend_strength_{period}'] = self._stabilized_calculation(
                lambda x: x, trend_strength, feature_name=f"e1_trend_strength_{period}"
            )
            features[f'e1_trend_consistency_{period}'] = self._stabilized_calculation(
                lambda x: x, trend_consistency, feature_name=f"e1_trend_consistency_{period}"
            )
        
        return features
    
    def _calculate_trend_slope(self, close: np.ndarray, period: int) -> np.ndarray:
        """トレンド傾き計算"""
        n = len(close)
        slopes = np.full(n, np.nan)
        
        for i in range(period - 1, n):
            window_data = close[i - period + 1:i + 1]
            x = np.arange(len(window_data))
            
            # 最小二乗法による傾き計算
            mean_x = np.mean(x)
            mean_y = np.mean(window_data)
            
            numerator = np.sum((x - mean_x) * (window_data - mean_y))
            denominator = np.sum((x - mean_x) ** 2)
            
            if denominator > self.stability_config['eps_base']:
                slopes[i] = numerator / denominator
            else:
                slopes[i] = 0.0
        
        return slopes
    
    def _calculate_trend_consistency(self, close: np.ndarray, period: int) -> np.ndarray:
        """トレンド一貫性計算"""
        trend_slope = self._calculate_trend_slope(close, period)
        
        # 短期傾きの標準偏差（一貫性の逆指標）
        slope_std = self._stabilized_calculation(
            self._rolling_std_numba, trend_slope, period//2,
            feature_name="slope_std"
        )
        
        # 一貫性 = 1 / (1 + 正規化標準偏差)
        slope_std_norm = slope_std / (np.nanmedian(np.abs(slope_std)) + self.stability_config['eps_base'])
        consistency = 1.0 / (1.0 + slope_std_norm)
        
        return consistency

    # =========================================================================
    # ボラティリティ・バンド指標群（独自実装）
    # =========================================================================
    
    def calculate_volatility_features(self, high: np.ndarray, low: np.ndarray, close: np.ndarray) -> Dict[str, np.ndarray]:
        """ボラティリティ特徴量（Golden Ratio強化版）"""
        features = {}
        
        high = self._ensure_numpy_array(high)
        low = self._ensure_numpy_array(low)
        close = self._ensure_numpy_array(close)
        
        try:
            # ボリンジャーバンド（Golden Ratio調整）
            bb_features = self._calculate_enhanced_bollinger_bands(close)
            features.update(bb_features)
            
            # ATR関連（独自拡張）
            atr_features = self._calculate_enhanced_atr_features(high, low, close)
            features.update(atr_features)
            
            # ヒストリカルボラティリティ（ロバスト統計）
            hv_features = self._calculate_robust_historical_volatility(close)
            features.update(hv_features)
            
        except Exception as e:
            logger.error(f"ボラティリティ計算エラー: {e}")
            
        return features
    
    def _calculate_enhanced_bollinger_bands(self, close: np.ndarray) -> Dict[str, np.ndarray]:
        """強化版ボリンジャーバンド"""
        features = {}
        
        for period in self.params['volatility_periods']:
            # 基本BB
            bb_mid = self._stabilized_calculation(
                self._rolling_mean_numba, close, period,
                feature_name=f"bb_mid_{period}"
            )
            bb_std = self._stabilized_calculation(
                self._rolling_std_numba, close, period,
                feature_name=f"bb_std_{period}"
            )
            
            for multiplier in self.params['bb_multipliers']:
                bb_upper = bb_mid + multiplier * bb_std
                bb_lower = bb_mid - multiplier * bb_std
                
                # バンド内位置
                bb_position = self._safe_divide(close - bb_lower, bb_upper - bb_lower)
                
                # バンド幅（正規化）
                bb_width = self._safe_divide(bb_upper - bb_lower, bb_mid)
                
                # バンド幅パーセンタイル（動的基準）
                bb_width_pct = self._calculate_rolling_percentile(bb_width, period * 2)
                
                features[f'e1_bb_upper_{period}_{multiplier}'] = self._stabilized_calculation(
                    lambda x: x, bb_upper, feature_name=f"e1_bb_upper_{period}_{multiplier}"
                )
                features[f'e1_bb_lower_{period}_{multiplier}'] = self._stabilized_calculation(
                    lambda x: x, bb_lower, feature_name=f"e1_bb_lower_{period}_{multiplier}"
                )
                features[f'e1_bb_position_{period}_{multiplier}'] = self._stabilized_calculation(
                    lambda x: x, bb_position, feature_name=f"e1_bb_position_{period}_{multiplier}"
                )
                features[f'e1_bb_width_{period}_{multiplier}'] = self._stabilized_calculation(
                    lambda x: x, bb_width, feature_name=f"e1_bb_width_{period}_{multiplier}"
                )
                features[f'e1_bb_width_pct_{period}_{multiplier}'] = self._stabilized_calculation(
                    lambda x: x, bb_width_pct, feature_name=f"e1_bb_width_pct_{period}_{multiplier}"
                )
        
        return features
    
    def _calculate_rolling_percentile(self, data: np.ndarray, period: int, percentile: float = 0.5) -> np.ndarray:
        """ローリングパーセンタイル計算"""
        n = len(data)
        result = np.full(n, np.nan)
        
        for i in range(period - 1, n):
            window_data = data[i - period + 1:i + 1]
            valid_data = window_data[np.isfinite(window_data)]
            
            if len(valid_data) > 0:
                result[i] = np.percentile(valid_data, percentile * 100)
        
        return result
    
    def _calculate_enhanced_atr_features(self, high: np.ndarray, low: np.ndarray, close: np.ndarray) -> Dict[str, np.ndarray]:
        """強化版ATR特徴量"""
        features = {}
        
        # True Range計算
        prev_close = np.roll(close, 1)
        prev_close[0] = close[0]
        
        tr1 = high - low
        tr2 = np.abs(high - prev_close)
        tr3 = np.abs(low - prev_close)
        true_range = np.maximum(tr1, np.maximum(tr2, tr3))
        
        for period in self.params['volatility_periods']:
            # 基本ATR
            atr = self._stabilized_calculation(
                self._rolling_mean_numba, true_range, period,
                feature_name=f"e1_atr_{period}"
            )
            
            # ATR正規化（価格基準）
            atr_pct = 100 * self._safe_divide(atr, close)
            
            # ATRトレンド
            atr_trend = self._calculate_trend_slope(atr, period // 2)
            
            # ATRボラティリティ（ATRのボラティリティ）
            atr_volatility = self._stabilized_calculation(
                self._rolling_std_numba, atr, period,
                feature_name=f"atr_volatility_{period}"
            )
            
            features[f'e1_atr_{period}'] = atr
            features[f'e1_atr_pct_{period}'] = self._stabilized_calculation(
                lambda x: x, atr_pct, feature_name=f"e1_atr_pct_{period}"
            )
            features[f'e1_atr_trend_{period}'] = self._stabilized_calculation(
                lambda x: x, atr_trend, feature_name=f"e1_atr_trend_{period}"
            )
            features[f'e1_atr_volatility_{period}'] = atr_volatility
            
            # ATRベースバンド
            for multiplier in self.params['atr_multipliers']:
                atr_upper_band = close + multiplier * atr
                atr_lower_band = close - multiplier * atr
                
                features[f'e1_atr_upper_{period}_{multiplier}'] = self._stabilized_calculation(
                    lambda x: x, atr_upper_band, feature_name=f"e1_atr_upper_{period}_{multiplier}"
                )
                features[f'e1_atr_lower_{period}_{multiplier}'] = self._stabilized_calculation(
                    lambda x: x, atr_lower_band, feature_name=f"e1_atr_lower_{period}_{multiplier}"
                )
        
        return features
    
    def _calculate_robust_historical_volatility(self, close: np.ndarray) -> Dict[str, np.ndarray]:
        """ロバストヒストリカルボラティリティ"""
        features = {}
        
        # 対数リターン
        log_close = np.log(close + self.stability_config['eps_base'])  # ゼロ除算防止
        log_returns = np.diff(log_close, prepend=log_close[0])
        
        for period in self.params['volatility_periods']:
            # 標準的ヒストリカルボラティリティ
            hv_standard = self._stabilized_calculation(
                self._rolling_std_numba, log_returns, period,
                feature_name=f"e1_hv_standard_{period}"
            )
            
            # ロバストボラティリティ（MADベース）
            hv_robust = self._calculate_rolling_mad(log_returns, period)
            
            # 年率化（252営業日）
            hv_standard_annual = hv_standard * np.sqrt(252)
            hv_robust_annual = hv_robust * np.sqrt(252)
            
            # ボラティリティレジーム（相対水準）
            hv_regime = self._calculate_volatility_regime(hv_standard, period * 2)
            
            features[f'e1_hv_standard_{period}'] = hv_standard
            features[f'e1_hv_robust_{period}'] = self._stabilized_calculation(
                lambda x: x, hv_robust, feature_name=f"e1_hv_robust_{period}"
            )
            features[f'e1_hv_annual_{period}'] = self._stabilized_calculation(
                lambda x: x, hv_standard_annual, feature_name=f"e1_hv_annual_{period}"
            )
            features[f'e1_hv_robust_annual_{period}'] = self._stabilized_calculation(
                lambda x: x, hv_robust_annual, feature_name=f"e1_hv_robust_annual_{period}"
            )
            features[f'e1_hv_regime_{period}'] = self._stabilized_calculation(
                lambda x: x, hv_regime, feature_name=f"e1_hv_regime_{period}"
            )
        
        return features
    
    def _calculate_rolling_mad(self, data: np.ndarray, period: int) -> np.ndarray:
        """ローリングMAD（中央絶対偏差）"""
        n = len(data)
        result = np.full(n, np.nan)
        
        for i in range(period - 1, n):
            window_data = data[i - period + 1:i + 1]
            median_val = np.median(window_data)
            mad = np.median(np.abs(window_data - median_val))
            result[i] = mad * self.stability_config['mad_factor']  # 正規分布での標準偏差相当に調整
        
        return result
    
    def _calculate_volatility_regime(self, volatility: np.ndarray, lookback: int) -> np.ndarray:
        """ボラティリティレジーム判定"""
        n = len(volatility)
        regime = np.full(n, np.nan)
        
        for i in range(lookback - 1, n):
            window_vol = volatility[i - lookback + 1:i + 1]
            valid_vol = window_vol[np.isfinite(window_vol)]
            
            if len(valid_vol) > 0:
                current_vol = volatility[i]
                vol_percentile = np.sum(valid_vol <= current_vol) / len(valid_vol)
                
                # レジーム分類: Low(0-0.33), Medium(0.33-0.67), High(0.67-1.0)
                if vol_percentile < 0.33:
                    regime[i] = 0  # Low vol
                elif vol_percentile < 0.67:
                    regime[i] = 1  # Medium vol
                else:
                    regime[i] = 2  # High vol
        
        return regime
    
    # =========================================================================
    # 出来高関連指標群
    # =========================================================================
    
    def calculate_volume_features(self, high: np.ndarray, low: np.ndarray, 
                                close: np.ndarray, volume: np.ndarray) -> Dict[str, np.ndarray]:
        """出来高関連特徴量"""
        features = {}
        
        high = self._ensure_numpy_array(high)
        low = self._ensure_numpy_array(low)
        close = self._ensure_numpy_array(close)
        volume = self._ensure_numpy_array(volume)
        
        try:
            # CMF (Chaikin Money Flow)
            cmf_features = self._calculate_cmf_features(high, low, close, volume)
            features.update(cmf_features)
            
            # MFI (Money Flow Index)
            mfi_features = self._calculate_mfi_features(high, low, close, volume)
            features.update(mfi_features)
            
            # VWAP関連
            vwap_features = self._calculate_vwap_features(high, low, close, volume)
            features.update(vwap_features)
            
        except Exception as e:
            logger.error(f"出来高特徴量計算エラー: {e}")
            
        return features
    
    def _calculate_cmf_features(self, high: np.ndarray, low: np.ndarray, 
                              close: np.ndarray, volume: np.ndarray) -> Dict[str, np.ndarray]:
        """CMF特徴量計算"""
        features = {}
        
        # CLV (Close Location Value)
        clv = self._safe_divide((close - low) - (high - close), high - low)
        
        # Money Flow Volume
        money_flow_volume = clv * volume
        
        for period in self.params['cmf_periods']:
            mfv_sum = self._stabilized_calculation(
                self._rolling_sum_numba, money_flow_volume, period,
                feature_name=f"mfv_sum_{period}"
            )
            volume_sum = self._stabilized_calculation(
                self._rolling_sum_numba, volume, period,
                feature_name=f"volume_sum_{period}"
            )
            
            cmf = self._safe_divide(mfv_sum, volume_sum)
            
            features[f'e1_cmf_{period}'] = self._stabilized_calculation(
                lambda x: x, cmf, feature_name=f"e1_cmf_{period}"
            )
        
        return features
    
    @staticmethod
    @njit(cache=True)
    def _rolling_sum_numba(data: np.ndarray, period: int) -> np.ndarray:
        """高速ローリング合計"""
        n = len(data)
        result = np.full(n, np.nan)
        
        if period <= 0 or period > n:
            return result
        
        window_sum = np.sum(data[:period])
        result[period-1] = window_sum
        
        for i in range(period, n):
            window_sum = window_sum - data[i-period] + data[i]
            result[i] = window_sum
        
        return result
    
    def _calculate_mfi_features(self, high: np.ndarray, low: np.ndarray, 
                              close: np.ndarray, volume: np.ndarray) -> Dict[str, np.ndarray]:
        """MFI特徴量計算"""
        features = {}
        
        # Typical Price
        typical_price = (high + low + close) / 3.0
        
        # Raw Money Flow
        raw_money_flow = typical_price * volume
        
        # 前日Typical Price
        prev_typical_price = np.roll(typical_price, 1)
        prev_typical_price[0] = typical_price[0]
        
        # Positive/Negative Money Flow
        positive_mf = np.where(typical_price > prev_typical_price, raw_money_flow, 0.0)
        negative_mf = np.where(typical_price < prev_typical_price, raw_money_flow, 0.0)
        
        for period in self.params['mfi_periods']:
            pos_mf_sum = self._stabilized_calculation(
                self._rolling_sum_numba, positive_mf, period,
                feature_name=f"pos_mf_sum_{period}"
            )
            neg_mf_sum = self._stabilized_calculation(
                self._rolling_sum_numba, negative_mf, period,
                feature_name=f"neg_mf_sum_{period}"
            )
            
            money_ratio = self._safe_divide(pos_mf_sum, neg_mf_sum)
            mfi = 100 - (100 / (1 + money_ratio))
            
            features[f'e1_mfi_{period}'] = self._stabilized_calculation(
                lambda x: x, mfi, feature_name=f"e1_mfi_{period}"
            )
        
        return features
    
    def _calculate_vwap_features(self, high: np.ndarray, low: np.ndarray, 
                               close: np.ndarray, volume: np.ndarray) -> Dict[str, np.ndarray]:
        """VWAP特徴量計算"""
        features = {}
        
        # Typical Price
        typical_price = (high + low + close) / 3.0
        
        # Price * Volume
        pv = typical_price * volume
        
        for period in self.params['volume_periods']:
            pv_sum = self._stabilized_calculation(
                self._rolling_sum_numba, pv, period,
                feature_name=f"pv_sum_{period}"
            )
            volume_sum = self._stabilized_calculation(
                self._rolling_sum_numba, volume, period,
                feature_name=f"volume_sum_{period}"
            )
            
            vwap = self._safe_divide(pv_sum, volume_sum)
            
            # 価格とVWAPの乖離
            vwap_deviation = 100 * self._safe_divide(close - vwap, vwap)
            
            features[f'e1_vwap_{period}'] = self._stabilized_calculation(
                lambda x: x, vwap, feature_name=f"e1_vwap_{period}"
            )
            features[f'e1_vwap_deviation_{period}'] = self._stabilized_calculation(
                lambda x: x, vwap_deviation, feature_name=f"e1_vwap_deviation_{period}"
            )
        
        return features
    
    # =========================================================================
    # サポート・レジスタンス・ローソク足群
    # =========================================================================
    
    def calculate_support_resistance_features(self, high: np.ndarray, low: np.ndarray, 
                                            close: np.ndarray, open_prices: np.ndarray = None) -> Dict[str, np.ndarray]:
        """サポレジ・ローソク足特徴量"""
        features = {}
        
        high = self._ensure_numpy_array(high)
        low = self._ensure_numpy_array(low)
        close = self._ensure_numpy_array(close)
        
        if open_prices is None:
            open_prices = np.roll(close, 1)
            open_prices[0] = close[0]
        else:
            open_prices = self._ensure_numpy_array(open_prices)
        
        try:
            # ピボットポイント
            pivot_features = self._calculate_pivot_points(high, low, close)
            features.update(pivot_features)
            
            # フィボナッチレベル
            fibonacci_features = self._calculate_fibonacci_levels(high, low, close)
            features.update(fibonacci_features)
            
            # ローソク足パターン
            candlestick_features = self._calculate_candlestick_patterns(open_prices, high, low, close)
            features.update(candlestick_features)
            
        except Exception as e:
            logger.error(f"サポレジ・ローソク足計算エラー: {e}")
            
        return features
    
    def _calculate_pivot_points(self, high: np.ndarray, low: np.ndarray, close: np.ndarray) -> Dict[str, np.ndarray]:
        """ピボットポイント計算"""
        features = {}
        
        # 前日の高値・安値・終値
        prev_high = np.roll(high, 1)
        prev_low = np.roll(low, 1) 
        prev_close = np.roll(close, 1)
        
        prev_high[0] = high[0]
        prev_low[0] = low[0]
        prev_close[0] = close[0]
        
        # ピボットポイント
        pivot_point = (prev_high + prev_low + prev_close) / 3.0
        
        # サポート・レジスタンスレベル
        r1 = 2 * pivot_point - prev_low
        s1 = 2 * pivot_point - prev_high
        r2 = pivot_point + (prev_high - prev_low)
        s2 = pivot_point - (prev_high - prev_low)
        
        # 現在価格との距離（パーセント）
        pivot_distance = 100 * self._safe_divide(close - pivot_point, close)
        r1_distance = 100 * self._safe_divide(r1 - close, close)
        s1_distance = 100 * self._safe_divide(close - s1, close)
        
        features['e1_pivot_point'] = self._stabilized_calculation(
            lambda x: x, pivot_point, feature_name="e1_pivot_point"
        )
        features['e1_resistance1'] = self._stabilized_calculation(
            lambda x: x, r1, feature_name="e1_resistance1"
        )
        features['e1_support1'] = self._stabilized_calculation(
            lambda x: x, s1, feature_name="e1_support1"
        )
        features['e1_resistance2'] = self._stabilized_calculation(
            lambda x: x, r2, feature_name="e1_resistance2"
        )
        features['e1_support2'] = self._stabilized_calculation(
            lambda x: x, s2, feature_name="e1_support2"
        )
        features['e1_pivot_distance'] = self._stabilized_calculation(
            lambda x: x, pivot_distance, feature_name="e1_pivot_distance"
        )
        features['e1_r1_distance'] = self._stabilized_calculation(
            lambda x: x, r1_distance, feature_name="e1_r1_distance"
        )
        features['e1_s1_distance'] = self._stabilized_calculation(
            lambda x: x, s1_distance, feature_name="e1_s1_distance"
        )
        
        return features
    
    def _calculate_fibonacci_levels(self, high: np.ndarray, low: np.ndarray, close: np.ndarray) -> Dict[str, np.ndarray]:
        """フィボナッチレベル計算"""
        features = {}
        
        for period in self.params['channel_periods']:
            swing_high = self._stabilized_calculation(
                self._rolling_max_numba, high, period,
                feature_name=f"swing_high_{period}"
            )
            swing_low = self._stabilized_calculation(
                self._rolling_min_numba, low, period,
                feature_name=f"swing_low_{period}"
            )
            
            swing_range = swing_high - swing_low
            
            for ratio in self.params['fibonacci_levels']:
                # リトレースメントレベル
                fib_level = swing_low + ratio * swing_range
                
                # 現在価格との距離
                fib_distance = 100 * self._safe_divide(close - fib_level, close)
                
                # フィボナッチレベル近接判定
                fib_near = np.where(np.abs(fib_distance) < 1.0, 1.0, 0.0)  # 1%以内
                
                ratio_str = str(ratio).replace('.', '')
                features[f'e1_fib_level_{ratio_str}_{period}'] = self._stabilized_calculation(
                    lambda x: x, fib_level, feature_name=f"e1_fib_level_{ratio_str}_{period}"
                )
                features[f'e1_fib_distance_{ratio_str}_{period}'] = self._stabilized_calculation(
                    lambda x: x, fib_distance, feature_name=f"e1_fib_distance_{ratio_str}_{period}"
                )
                features[f'e1_fib_near_{ratio_str}_{period}'] = self._stabilized_calculation(
                    lambda x: x, fib_near, feature_name=f"e1_fib_near_{ratio_str}_{period}"
                )
        
        return features
    
    @staticmethod
    @njit(cache=True)
    def _rolling_max_numba(data: np.ndarray, period: int) -> np.ndarray:
        """高速ローリング最大値"""
        n = len(data)
        result = np.full(n, np.nan)
        
        for i in range(period-1, n):
            result[i] = np.max(data[i-period+1:i+1])
        
        return result
    
    @staticmethod
    @njit(cache=True)
    def _rolling_min_numba(data: np.ndarray, period: int) -> np.ndarray:
        """高速ローリング最小値"""
        n = len(data)
        result = np.full(n, np.nan)
        
        for i in range(period-1, n):
            result[i] = np.min(data[i-period+1:i+1])
        
        return result
    
    def _calculate_candlestick_patterns(self, open_prices: np.ndarray, high: np.ndarray, 
                                      low: np.ndarray, close: np.ndarray) -> Dict[str, np.ndarray]:
        """ローソク足パターン計算"""
        features = {}
        
        # 基本要素
        body_size = np.abs(close - open_prices)
        upper_shadow = high - np.maximum(open_prices, close)
        lower_shadow = np.minimum(open_prices, close) - low
        total_range = high - low
        
        # 正規化
        body_ratio = self._safe_divide(body_size, total_range)
        upper_shadow_ratio = self._safe_divide(upper_shadow, total_range)
        lower_shadow_ratio = self._safe_divide(lower_shadow, total_range)
        
        # 基本方向
        bullish = np.where(close > open_prices, 1.0, 0.0)
        bearish = np.where(close < open_prices, 1.0, 0.0)
        
        # パターン識別
        doji = np.where(body_ratio < 0.1, 1.0, 0.0)
        hammer = np.where((lower_shadow_ratio > 0.5) & (upper_shadow_ratio < 0.1) & 
                         (body_ratio < 0.3), 1.0, 0.0)
        shooting_star = np.where((upper_shadow_ratio > 0.5) & (lower_shadow_ratio < 0.1) & 
                                (body_ratio < 0.3), 1.0, 0.0)
        
        # 強いローソク足（実体が大きい）
        strong_candle = np.where(body_ratio > 0.7, 1.0, 0.0)
        
        features['e1_body_ratio'] = self._stabilized_calculation(
            lambda x: x, body_ratio, feature_name="e1_body_ratio"
        )
        features['e1_upper_shadow_ratio'] = self._stabilized_calculation(
            lambda x: x, upper_shadow_ratio, feature_name="e1_upper_shadow_ratio"
        )
        features['e1_lower_shadow_ratio'] = self._stabilized_calculation(
            lambda x: x, lower_shadow_ratio, feature_name="e1_lower_shadow_ratio"
        )
        features['e1_bullish'] = self._stabilized_calculation(
            lambda x: x, bullish, feature_name="e1_bullish"
        )
        features['e1_bearish'] = self._stabilized_calculation(
            lambda x: x, bearish, feature_name="e1_bearish"
        )
        features['e1_doji'] = self._stabilized_calculation(
            lambda x: x, doji, feature_name="e1_doji"
        )
        features['e1_hammer'] = self._stabilized_calculation(
            lambda x: x, hammer, feature_name="e1_hammer"
        )
        features['e1_shooting_star'] = self._stabilized_calculation(
            lambda x: x, shooting_star, feature_name="e1_shooting_star"
        )
        features['e1_strong_candle'] = self._stabilized_calculation(
            lambda x: x, strong_candle, feature_name="e1_strong_candle"
        )
        
        return features
    
    # =========================================================================
    # 統合特徴量計算メソッド（Polars最適化版）
    # =========================================================================
    
    def calculate_all_features_optimized(self, high: np.ndarray, low: np.ndarray, 
                                       close: np.ndarray, volume: np.ndarray,
                                       open_prices: np.ndarray = None) -> Dict[str, np.ndarray]:
        """
        全特徴量の統合計算（Calculator中心・80%リソース集中版）
        4段階式・適応型数値安定性エンジン完全適用
        """
        logger.info("Calculator統合計算開始 - 4段階式数値安定性エンジン適用")
        start_time = time.time()
        
        all_features = {}
        
        try:
            # 1. ADX・基本オシレーター（独自数学実装）
            logger.debug("ADX・オシレーター計算中...")
            adx_features = self.calculate_adx_features(high, low, close)
            all_features.update(adx_features)
            
            rsi_features = self.calculate_rsi_features(close)
            all_features.update(rsi_features)
            
            # 2. 移動平均線・トレンド分析（並列化対応）
            logger.debug("移動平均・トレンド分析計算中...")
            ma_features = self.calculate_moving_averages(close)
            all_features.update(ma_features)
            
            # 3. ボラティリティ・バンド指標（Golden Ratio強化）
            logger.debug("ボラティリティ特徴量計算中...")
            volatility_features = self.calculate_volatility_features(high, low, close)
            all_features.update(volatility_features)
            
            # 4. 出来高関連指標
            logger.debug("出来高特徴量計算中...")
            volume_features = self.calculate_volume_features(high, low, close, volume)
            all_features.update(volume_features)
            
            # 5. サポート・レジスタンス・ローソク足
            logger.debug("サポレジ・ローソク足計算中...")
            sr_features = self.calculate_support_resistance_features(high, low, close, open_prices)
            all_features.update(sr_features)
            
            total_time = time.time() - start_time
            logger.info(f"Calculator統合計算完了: {len(all_features)}特徴量を{total_time:.2f}秒で生成")
            
        except Exception as e:
            logger.error(f"統合計算エラー: {e}")
            import traceback
            traceback.print_exc()
        
        return all_features
    
    def generate_quality_report(self, features: Dict[str, np.ndarray]) -> Dict[str, Any]:
        """品質レポート生成（4段階エンジン統計含む）"""
        
        if not features:
            return {'status': 'no_features', 'total_features': 0}
        
        try:
            total_features = len(features)
            total_points = len(next(iter(features.values())))
            
            # 各特徴量の品質評価
            quality_scores = {}
            stabilization_summary = {1: 0, 2: 0, 3: 0, 4: 0}
            
            for name, values in features.items():
                quality_score = self._calculate_quality_score(values, name)
                quality_scores[name] = quality_score
                
                # 安定化レベルの集計
                if quality_score <= 0.3:
                    stabilization_summary[4] += 1
                elif quality_score <= 0.5:
                    stabilization_summary[3] += 1
                elif quality_score <= 0.7:
                    stabilization_summary[2] += 1
                else:
                    stabilization_summary[1] += 1
            
            # 統計
            avg_quality = np.mean(list(quality_scores.values()))
            high_quality_count = sum(1 for score in quality_scores.values() if score > 0.9)
            
            return {
                'status': 'completed',
                'total_features': total_features,
                'data_points': total_points,
                'average_quality_score': avg_quality,
                'high_quality_features': high_quality_count,
                'stabilization_summary': stabilization_summary,
                'stabilization_applications': self.stats['stabilization_applications'].copy(),
                'fallback_count': self.stats['fallback_count'],
                'calculation_stats': self.stats.copy()
            }
            
        except Exception as e:
            logger.error(f"品質レポート生成エラー: {e}")
            return {
                'status': 'error',
                'error_message': str(e),
                'total_features': len(features)
            }
    
    def get_calculation_summary(self) -> Dict[str, Any]:
        """計算統計サマリー（4段階エンジン統計含む）"""
        total_calc = max(1, self.stats['total_calculations'])
        
        return {
            'total_calculations': total_calc,
            'success_rate': self.stats['successful_calculations'] / total_calc,
            'avg_quality_score': np.mean(self.stats['quality_scores']) if self.stats['quality_scores'] else 0.0,
            'stabilization_level_1': self.stats['stabilization_applications'][1],
            'stabilization_level_2': self.stats['stabilization_applications'][2],
            'stabilization_level_3': self.stats['stabilization_applications'][3],
            'stabilization_level_4': self.stats['stabilization_applications'][4],
            'fallback_algorithm_uses': self.stats['fallback_count'],
            'avg_computation_time': np.mean(self.stats['computation_times']) if self.stats['computation_times'] else 0.0,
            'golden_ratio_optimizations': True,
            'robust_statistics_enabled': True,
            'adaptive_processing_enabled': True,
            'four_stage_stabilization_engine': True
        }    

# =============================================================================
# メインエンジンクラス（簡素化・Calculator中心）
# =============================================================================

class FeatureExtractionEngine:
    """
    特徴量抽出エンジン - Calculator中心アーキテクチャ
    80%リソースをCalculatorに集中投入
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        if config is None:
            config = DATA_CONFIG
        
        self.config = config
        
        # 各コンポーネント初期化（薄い実装）
        self.data_processor = DataProcessor(config['base_path'])
        self.window_manager = WindowManager(window_size=100, overlap=0.5)
        self.memory_manager = MemoryManager()
        self.output_manager = OutputManager(config['output_path'])
        
        # Calculator（80%リソース集中）
        self.calculator = Calculator(self.window_manager, self.memory_manager)
        
        logger.info("FeatureExtractionEngine初期化完了 - Calculator中心アーキテクチャ")
    
    def run_extraction(self, test_mode: bool = False, 
                      target_timeframes: List[str] = None) -> Dict[str, Any]:
        """
        特徴量抽出実行（Calculator中心処理）
        """
        logger.info("特徴量抽出開始 - Calculator中心処理モード")
        start_time = time.time()
        
        try:
            # データ準備（最小限）
            if target_timeframes is None:
                target_timeframes = ['tick']
            
            memmap_data = self.data_processor.convert_to_memmap(target_timeframes)
            
            results = {}
            for tf, memmap_array in memmap_data.items():
                logger.info(f"タイムフレーム {tf} 処理開始")
                
                # テストモードでは小規模データに制限
                if test_mode:
                    test_size = min(1000, memmap_array.shape[0])
                    current_data = memmap_array[:test_size]
                    logger.info(f"テストモード: {test_size}行に制限")
                else:
                    current_data = memmap_array
                
                # メモリチェック
                memory_status = self.memory_manager.check_memory_status()
                if memory_status['status'] == 'warning':
                    logger.warning("メモリ使用量が警告レベルに達しています")
                
                # Calculator中心処理（80%リソース集中）
                tf_features = self._process_timeframe_with_calculator(current_data, tf, test_mode)
                
                if tf_features:
                    # 出力処理（最小限）
                    output_filename = f"features_{tf}{'_test' if test_mode else ''}"
                    output_path = self.output_manager.save_features(tf_features, output_filename)
                    
                    results[tf] = {
                        'features_count': len(tf_features),
                        'output_path': str(output_path)
                    }
                    
                    logger.info(f"タイムフレーム {tf} 完了: {len(tf_features)}特徴量生成")
                
                # メモリ解放
                self.memory_manager.force_garbage_collection()
            
            total_time = time.time() - start_time
            
            # 最終結果
            final_results = {
                'total_time_minutes': total_time / 60,
                'processed_timeframes': list(results.keys()),
                'total_features_generated': sum(r['features_count'] for r in results.values()),
                'test_mode': test_mode,
                'calculator_stats': self.calculator.get_calculation_summary(),
                'timeframe_results': results
            }
            
            logger.info(f"特徴量抽出完了: {final_results['total_features_generated']}特徴量を{total_time:.2f}秒で生成")
            
            return final_results
            
        except Exception as e:
            logger.error(f"特徴量抽出エラー: {e}")
            import traceback
            traceback.print_exc()
            raise
    
    def _process_timeframe_with_calculator(self, memmap_data: np.memmap, 
                                         timeframe: str, test_mode: bool) -> Dict[str, np.ndarray]:
        """
        Calculator中心のタイムフレーム処理
        """
        try:
            # データ形状チェック
            if memmap_data.shape[1] < 7:
                logger.error(f"データ列数不足: {memmap_data.shape[1]} < 7")
                return {}
            
            # OHLCV抽出
            open_prices = memmap_data[:, 1]
            high_prices = memmap_data[:, 2] 
            low_prices = memmap_data[:, 3]
            close_prices = memmap_data[:, 4]
            volume_data = memmap_data[:, 5]
            
            # Calculator統合計算実行（80%リソース集中）
            all_features = self.calculator.calculate_all_features_optimized(
                high=high_prices,
                low=low_prices,
                close=close_prices,
                volume=volume_data,
                open_prices=open_prices
            )
            
            # 品質レポート生成
            quality_report = self.calculator.generate_quality_report(all_features)
            
            if quality_report.get('average_quality_score', 0) < 0.6:
                logger.warning(f"品質警告: 平均品質スコア {quality_report['average_quality_score']:.3f}")
            
            return all_features
            
        except Exception as e:
            logger.error(f"Calculator処理エラー: {e}")
            return {}

# =============================================================================
# システム検証・インタラクティブモード
# =============================================================================

def interactive_mode():
    """インタラクティブモード（完全版 - 出力パス選択機能付き）"""
    print("=" * 70)
    print("🚀 Project Forge 軍資金増大ミッション - 革新的特徴量収集 🚀")
    print("=" * 70)
    
    # システム情報表示
    display_system_info()
    
    # 1. 出力パス設定（プロンプト要件に従って追加）
    print("\n💾 出力ファイルパス設定:")
    default_output_path = "/workspaces/project_forge/data/2_feature_value/"
    print(f"デフォルト: {default_output_path}")
    
    while True:
        custom_path = input(f"カスタムパスを入力 (Enterでデフォルト使用): ").strip()
        
        if not custom_path:
            output_path = Path(default_output_path)
            print(f"✅ デフォルトパス使用: {output_path}")
            break
        else:
            output_path = Path(custom_path)
            try:
                output_path.mkdir(parents=True, exist_ok=True)
                if output_path.exists() and output_path.is_dir():
                    print(f"✅ カスタムパス使用: {output_path}")
                    break
                else:
                    print("❌ 無効なパスです。再入力してください。")
            except Exception as e:
                print(f"❌ パス作成エラー: {e}")
    
    # 2. 時間足選択
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
                target_timeframes = DATA_CONFIG['timeframes']
                break
            elif choice == '4':
                custom_input = input("タイムフレーム指定 (例: tick,M1,M5 または 1,2,3): ").strip()
                target_timeframes = parse_timeframe_selection(custom_input, DATA_CONFIG['timeframes'])
                if target_timeframes:
                    break
                else:
                    print("❌ 無効な選択です")
            else:
                print("❌ 1-4の範囲で選択してください")
        except (ValueError, KeyboardInterrupt):
            print("❌ 無効な入力です")
    
    # 3. テストモード選択
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
    
    # 4. 出力詳細度選択（プロンプト仕様に従って追加）
    print("\n📋 出力詳細度選択:")
    print("1. 基本（エラーと完了情報のみ）")
    print("2. 詳細（進捗状況、統計情報含む）")
    
    while True:
        try:
            detail_choice = int(input("選択してください (1-2): "))
            if detail_choice in [1, 2]:
                verbose_mode = detail_choice == 2
                break
            else:
                print("❌ 1-2の範囲で選択してください")
        except ValueError:
            print("❌ 数字を入力してください")
    
    # 5. ウィンドウサイズ選択（プロンプト仕様に従って追加）
    print("\n🪟 ウィンドウサイズ設定:")
    print("デフォルト推奨値: 100")
    
    while True:
        try:
            window_input = input("ウィンドウサイズを入力 (Enterでデフォルト100使用): ").strip()
            if not window_input:
                window_size = 100
                print(f"✅ デフォルトウィンドウサイズ使用: {window_size}")
                break
            else:
                window_size = int(window_input)
                if 10 <= window_size <= 10000:
                    print(f"✅ カスタムウィンドウサイズ使用: {window_size}")
                    break
                else:
                    print("❌ ウィンドウサイズは10-10000の範囲で入力してください")
        except ValueError:
            print("❌ 数値を入力してください")
    
    # 6. 確認画面
    print("\n" + "=" * 50)
    print("📋 実行設定確認")
    print("=" * 50)
    print(f"出力パス: {output_path}")
    print(f"対象タイムフレーム: {target_timeframes}")
    print(f"実行モード: {'テストモード（1000行制限）' if test_mode else '本番モード（全データ処理）'}")
    print(f"出力詳細度: {'詳細' if verbose_mode else '基本'}")
    print(f"ウィンドウサイズ: {window_size}")
    print("=" * 50)
    
    confirm = input("\n実行しますか？ (y/N): ").strip().lower()
    if confirm != 'y':
        print("❌ 実行をキャンセルしました")
        return None
    
    # 7. メインエンジン実行（カスタム設定を反映）
    try:
        # カスタム設定でコンフィグを更新
        custom_config = DATA_CONFIG.copy()
        custom_config['output_path'] = output_path
        
        engine = FeatureExtractionEngine(custom_config)
        
        # ウィンドウマネージャーもカスタム設定を反映
        engine.window_manager = WindowManager(window_size=window_size, overlap=0.5)
        
        print("\n🚀 革新的特徴量収集エンジン開始 - Project Forge軍資金増大ミッション 🚀")
        
        # 詳細モードの設定
        if verbose_mode:
            import logging
            logging.getLogger().setLevel(logging.DEBUG)
        
        # 正しいメソッド名を使用
        results = engine.run_extraction(
            test_mode=test_mode,
            target_timeframes=target_timeframes
        )
        
        print("\n🎉 Project Forge 軍資金増大ミッション完了！")
        print("Next: Project Chimera開発開始 🚀")
        
        # 結果詳細表示
        if verbose_mode and results:
            print(f"\n📊 処理結果詳細:")
            print(f"処理時間: {results.get('total_time_minutes', 0):.2f}分")
            print(f"生成特徴量数: {results.get('total_features_generated', 0)}")
            print(f"処理ファイルパス: {output_path}")
        
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
        # NumPy最適化設定
        os.environ['OMP_NUM_THREADS'] = str(HARDWARE_SPEC['cpu_cores'])
        os.environ['MKL_NUM_THREADS'] = str(HARDWARE_SPEC['cpu_cores']) 
        os.environ['NUMBA_NUM_THREADS'] = str(HARDWARE_SPEC['cpu_cores'])
        
        # インタラクティブモード実行
        results = interactive_mode()
        
        if results:
            logger.info("Calculator中心アーキテクチャによる処理成功")
        else:
            logger.warning("処理が完了しませんでした")
            
    except KeyboardInterrupt:
        logger.info("ユーザーによって中断されました")
    except Exception as e:
        logger.error(f"予期しないエラー: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

"""
Calculator中心再設計版完成 - 80%リソース集中アーキテクチャ

主要特徴:
- Calculator: 2000+行（全体の80%リソース）
- その他クラス: 各50-300行（合計20%リソース）
- Golden Ratio基準パラメータ最適化
- Numba JIT並列化（計算集約関数）
- ゴールデンオーバーラップ並列化（WMA等）
- 段階的安定化アプローチ
- ロバスト統計手法適用
- 独自数学実装による競争優位性確保

Next Phase: Project Chimera Development
"""