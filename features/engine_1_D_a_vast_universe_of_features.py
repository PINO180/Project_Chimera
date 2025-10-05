#!/usr/bin/env python3
"""
革新的な特徴量重厚集スクリプト - Engine 1D: Volume & Price Action Features (修正版)
【修正内容】垂直分割アーキテクチャをディスクベース物理分割に変更

Project Forge - 軍資金増大プロジェクト
最終目標: Project Chimera開発・完成のための資金調達

技術戦略: ジム・シモンズの思想的継承
- 経済学・ファンダメンタルズ・古典的テクニカル指標の完全排除
- 統計的に有意で非ランダムな微細パターン「マーケットの亡霊」の探索
- AIの頭脳による普遍的法則の読み解き

アーキテクチャ: 3クラス構成（最適化版）+ ディスクベース垂直分割
- DataEngine（30%）: Polars LazyFrame基盤
- CalculationEngine（60%）: 特徴量計算核心（物理的垂直分割実装）
- OutputEngine（10%）: ストリーミング出力
"""
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))

import blueprint as config
import os, sys, time, warnings, json, logging, math, tempfile, datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, Any
from dataclasses import dataclass, field

# 数値計算・データ処理
import numpy as np
import polars as pl
import numba as nb
from numba import guvectorize, float64, int64
from scipy import stats
from scipy.stats import jarque_bera, anderson, shapiro, trim_mean

# メモリ監視
import psutil

# 警告制御
warnings.filterwarnings('ignore', category=RuntimeWarning)
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=FutureWarning)

# Polars設定最適化の直後に追加  
pl.Config.set_streaming_chunk_size(100_000)
pl.Config.set_tbl_formatting("ASCII_MARKDOWN")
pl.Config.set_tbl_hide_dataframe_shape(True)
pl.enable_string_cache()

# ログ設定
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

def get_default_timeframes() -> List[str]:
    return ["tick", "M0.5", "M1", "M3", "M5", "M8", "M15", "M30",
            "H1", "H4", "H6", "H12", "D1", "W1", "MN"]

def get_default_window_sizes() -> Dict[str, List[int]]:
    return {
        "rsi": [14, 21, 30, 50],
        "atr": [13, 21, 34, 55],
        "adx": [13, 21, 34],
        "hma": [21, 34, 55],
        "kama": [21, 34],
        "general": [10, 20, 50, 100],
        "volume": [13, 21, 34],
        "volatility": [10, 20, 30, 50]
    }

@dataclass
class ProcessingConfig:
    """処理設定 - Project Forge統合版"""

    # データパス（Project Forge構造準拠） - config.pyから読み込む
    input_path: str = str(config.S1_BASE_MULTITIMEFRAME)
    partitioned_tick_path: str = str(config.S1_RAW_TICK_PARTITIONED)
    output_path: str = str(config.S2_FEATURES)

    # エンジン識別
    engine_id: str = "e1d"
    engine_name: str = "Engine_1D_VolumeActionFeatures"

    # 並列処理（戦略的並列処理スロットリング）
    max_threads: int = 4

    # メモリ制限（64GB RAM制約）
    memory_limit_gb: float = 55.0
    memory_warning_gb: float = 50.0

    timeframes: List[str] = field(default_factory=get_default_timeframes)
    window_sizes: Dict[str, List[int]] = field(default_factory=get_default_window_sizes)

    # 処理モード
    test_mode: bool = False
    test_rows: int = 10000

    # システムハイパーパラメータとしてW_maxを定義
    w_max: int = 200

    def validate(self) -> bool:
        """設定検証"""
        output_path_obj = Path(self.output_path)
        if not output_path_obj.exists():
            output_path_obj.mkdir(parents=True, exist_ok=True)
            logger.info(f"出力ディレクトリを作成: {output_path_obj}")
        return True

class MemoryMonitor:
    """メモリ使用量監視クラス - Project Forge準拠"""
    
    def __init__(self, limit_gb: float = 50.0, emergency_gb: float = 55.0):
        self.limit_gb = limit_gb
        self.emergency_gb = emergency_gb
        self.process = psutil.Process()
    
    def get_memory_usage_gb(self) -> float:
        """現在のメモリ使用量をGB単位で取得"""
        memory_info = self.process.memory_info()
        return memory_info.rss / (1024 ** 3)
    
    def check_memory_safety(self) -> Tuple[bool, str]:
        """メモリ安全性チェック - Project Forge基準"""
        current_gb = self.get_memory_usage_gb()
        
        if current_gb > self.emergency_gb:
            return False, f"緊急停止: メモリ使用量 {current_gb:.2f}GB > {self.emergency_gb}GB"
        elif current_gb > self.limit_gb:
            return True, f"警告: メモリ使用量 {current_gb:.2f}GB > {self.limit_gb}GB"
        else:
            return True, f"正常: メモリ使用量 {current_gb:.2f}GB"

class DataEngine:
    """
    データ基盤クラス（30%） - Project Forge統合版
    責務:
    - Parquetメタデータ事前検証
    - Polars scan_parquetによる遅延読み込み（LazyFrame生成）
    - 述語プッシュダウンによるフィルタリング最適化
    - timeframe別分割処理
    - メモリ使用量監視
    - エラー予防機能
    """
    
    def __init__(self, config: ProcessingConfig):
        self.config = config
        self.memory_monitor = MemoryMonitor(config.memory_limit_gb)
        self.metadata_cache: Dict[str, Any] = {}
    
    def validate_data_source(self) -> bool:
        """データソース検証 - XAU/USDデータ構造準拠"""
        input_path = Path(self.config.input_path)
        if not input_path.exists():
            logger.error(f"データソースが存在しません: {input_path}")
            return False
        
        # timeframeディレクトリの確認
        timeframe_dirs = [d for d in input_path.iterdir() if d.is_dir() and d.name.startswith("timeframe=")]
        
        if not timeframe_dirs:
            logger.error("timeframeディレクトリが見つかりません")
            return False
        
        logger.info(f"検出されたタイムフレーム: {len(timeframe_dirs)}個")
        return True
    
    def verify_parquet_metadata(self) -> Dict[str, Any]:
        """Parquetメタデータ検証 - Project Forge基準"""
        try:
            # globパターンでParquetファイルのみを指定
            parquet_pattern = f"{self.config.input_path}/**/*.parquet"
            
            # LazyFrameでメタデータ取得（実際の読み込みなし）
            lazy_frame = pl.scan_parquet(parquet_pattern)
            
            # スキーマ情報取得（警告回避）
            schema = lazy_frame.collect_schema()
            
            # 基本メタデータ収集
            metadata = {
                "schema": dict(schema),
                "columns": list(schema.keys()),
                "path_exists": Path(self.config.input_path).exists(),
                "estimated_memory_gb": 0.0
            }
            
            # 必須カラムチェック（XAU/USDデータ構造）
            required_columns = ["timestamp", "open", "high", "low", "close", "volume"]
            missing_columns = [col for col in required_columns if col not in metadata["columns"]]
            
            if missing_columns:
                raise ValueError(f"必須カラムが見つかりません: {missing_columns}")
            
            # Hiveパーティション構造の確認
            available_timeframes = self.config.timeframes
            logger.info("Hiveパーティション構造のためtimeframe確認をスキップ")
            
            metadata["available_timeframes"] = available_timeframes
            metadata["requested_timeframes"] = self.config.timeframes
            metadata["is_hive_partitioned"] = True
            
            self.metadata_cache = metadata
            logger.info(f"メタデータ検証完了: {len(metadata['columns'])}列, Hiveパーティション構造")
            
            return metadata
            
        except Exception as e:
            logger.error(f"メタデータ検証エラー: {e}")
            raise
    
    def create_lazy_frame(self, timeframe: str) -> pl.LazyFrame:
        """指定timeframeのLazyFrame生成 - Hiveパーティション対応"""
        try:
            is_safe, message = self.memory_monitor.check_memory_safety()
            if not is_safe:
                raise MemoryError(message)
            
            logger.info(f"LazyFrame生成開始: timeframe={timeframe}")
            
            # Hiveパーティション対応：特定timeframeディレクトリを直接指定
            timeframe_path = f"{self.config.input_path}/timeframe={timeframe}/*.parquet"
            
            # 指定timeframeのParquetファイルのみをスキャン
            lazy_frame = pl.scan_parquet(timeframe_path)
            
            # timeframe列を手動で追加（Hiveパーティション復元）
            lazy_frame = lazy_frame.with_columns([
                pl.lit(timeframe).alias("timeframe").cast(pl.Categorical)
            ])
            
            # スキーマを確認してから安全にキャスト処理を適用
            # まず小さなサンプルでスキーマを確認
            try:
                sample_schema = lazy_frame.limit(1).collect_schema()
                logger.info(f"検出されたスキーマ: {list(sample_schema.keys())}")
                
                # 基本データ型確認と最適化（必要な場合のみキャスト）
                cast_exprs = []
                
                # 各カラムが存在し、かつ適切な型でない場合のみキャスト
                if "timestamp" in sample_schema and sample_schema["timestamp"] != pl.Datetime("ns"):
                    cast_exprs.append(pl.col("timestamp").cast(pl.Datetime("ns")))
                if "open" in sample_schema and sample_schema["open"] != pl.Float64:
                    cast_exprs.append(pl.col("open").cast(pl.Float64))
                if "high" in sample_schema and sample_schema["high"] != pl.Float64:
                    cast_exprs.append(pl.col("high").cast(pl.Float64))
                if "low" in sample_schema and sample_schema["low"] != pl.Float64:
                    cast_exprs.append(pl.col("low").cast(pl.Float64))
                if "close" in sample_schema and sample_schema["close"] != pl.Float64:
                    cast_exprs.append(pl.col("close").cast(pl.Float64))
                if "volume" in sample_schema and sample_schema["volume"] != pl.Int64:
                    cast_exprs.append(pl.col("volume").cast(pl.Int64))
                
                # 必要なキャストのみ適用
                if cast_exprs:
                    logger.info(f"型変換を適用: {len(cast_exprs)}個のカラム")
                    lazy_frame = lazy_frame.with_columns(cast_exprs)
                else:
                    logger.info("型変換不要: 全カラムが適切な型です")
                
            except Exception as schema_error:
                logger.warning(f"スキーマ確認エラー、キャストをスキップ: {schema_error}")
                # キャストエラーの場合は処理を続行（データがすでに適切な型の可能性）
            
            # タイムスタンプソート
            lazy_frame = lazy_frame.sort("timestamp")
            
            logger.info(f"LazyFrame生成完了: timeframe={timeframe}")
            return lazy_frame
            
        except Exception as e:
            logger.error(f"LazyFrame生成エラー (timeframe={timeframe}): {e}")
            raise
    
    def get_data_summary(self, lazy_frame: pl.LazyFrame) -> Dict[str, Any]:
        """データサマリー情報取得（軽量）"""
        try:
            # 最小限のcollectでサマリー取得
            summary = lazy_frame.select([
                pl.len().alias("total_rows"),
                pl.col("timestamp").min().alias("start_time"),
                pl.col("timestamp").max().alias("end_time"),
                pl.col("close").mean().alias("avg_price"),
                pl.col("volume").sum().alias("total_volume")
            ]).collect()
            
            return {
                "total_rows": summary["total_rows"][0],
                "start_time": summary["start_time"][0],
                "end_time": summary["end_time"][0],
                "avg_price": summary["avg_price"][0],
                "total_volume": summary["total_volume"][0]
            }
        except Exception as e:
            logger.error(f"データサマリー取得エラー: {e}")
            return {"error": str(e)}

class QualityAssurance:
    """数値安定性保証システム - 2段階品質保証エンジン（Project Forge準拠）"""
    
    @staticmethod
    def calculate_quality_score(values: np.ndarray) -> float:
        """
        品質スコア算出：0.0（使用不可）〜 1.0（完璧）
        Project Forge厳密な統計的定義に基づく実装
        
        Args:
            values: 評価対象の数値配列
        Returns:
            float: 品質スコア
        """
        if len(values) == 0:
            return 0.0
        
        n_total = len(values)
        
        # 1. 有限値比率の厳密計算
        finite_count = 0
        nan_count = 0
        inf_count = 0
        
        for val in values:
            if np.isnan(val):
                nan_count += 1
            elif np.isinf(val):
                inf_count += 1
            else:
                finite_count += 1
        
        finite_ratio = finite_count / n_total
        
        if finite_count == 0:
            return 0.0
        
        # 有限値のみを抽出
        finite_values = np.zeros(finite_count)
        idx = 0
        for val in values:
            if np.isfinite(val):
                finite_values[idx] = val
                idx += 1
        
        # 2. 統計的多様性指標の厳密計算
        # シャノンエントロピーによる多様性評価
        unique_values, counts = np.unique(finite_values, return_counts=True)
        n_unique = len(unique_values)
        
        if n_unique == 1:
            diversity_score = 0.0  # 完全に均一
        else:
            # シャノンエントロピーの厳密計算
            shannon_entropy = 0.0
            for count in counts:
                p = count / finite_count
                if p > 0:
                    shannon_entropy -= p * np.log2(p)
            
            # 最大可能エントロピー（完全均等分布時）
            max_entropy = np.log2(min(n_unique, finite_count))
            if max_entropy > 0:
                diversity_score = shannon_entropy / max_entropy
            else:
                diversity_score = 0.0
        
        # 3. 数値的安定性指標の厳密計算
        if finite_count < 2:
            stability_score = 1.0
        else:
            # 不偏分散の厳密計算
            sum_values = 0.0
            for val in finite_values:
                sum_values += val
            mean_val = sum_values / finite_count
            
            sum_sq_deviations = 0.0
            for val in finite_values:
                deviation = val - mean_val
                sum_sq_deviations += deviation * deviation
            
            unbiased_variance = sum_sq_deviations / (finite_count - 1)
            std_dev = np.sqrt(unbiased_variance)
            
            # 変動係数による安定性評価
            if abs(mean_val) > 1e-15:
                cv = std_dev / abs(mean_val)
                stability_score = 1.0 / (1.0 + cv)
            else:
                stability_score = 1.0 / (1.0 + std_dev)
        
        # 4. 外れ値耐性指標の厳密計算
        if finite_count < 5:
            outlier_resistance = 1.0
        else:
            # Tukey's fences による外れ値検出
            sorted_values = np.sort(finite_values)
            
            # 四分位数の厳密計算（R-6 quantile method）
            def calculate_quantile_r6(sorted_arr, q):
                n = len(sorted_arr)
                h = n * q + 0.5
                h_floor = int(np.floor(h))
                h_ceil = int(np.ceil(h))
                if h_floor == h_ceil:
                    return sorted_arr[h_floor - 1]
                else:
                    gamma = h - h_floor
                    return sorted_arr[h_floor - 1] * (1 - gamma) + sorted_arr[h_ceil - 1] * gamma
            
            q1 = calculate_quantile_r6(sorted_values, 0.25)
            q3 = calculate_quantile_r6(sorted_values, 0.75)
            iqr = q3 - q1
            
            if iqr > 1e-15:
                # Tukey's fences
                lower_fence = q1 - 1.5 * iqr
                upper_fence = q3 + 1.5 * iqr
                
                outlier_count = 0
                for val in finite_values:
                    if val < lower_fence or val > upper_fence:
                        outlier_count += 1
                
                outlier_ratio = outlier_count / finite_count
                outlier_resistance = 1.0 - outlier_ratio
            else:
                outlier_resistance = 1.0
        
        # 総合品質スコアの厳密計算
        weights = {
            'finite_ratio': 0.4,  # 有限値率（最重要）
            'diversity': 0.25,  # 多様性
            'stability': 0.25,  # 安定性
            'outlier_resistance': 0.1  # 外れ値耐性
        }
        
        composite_score = (
            weights['finite_ratio'] * finite_ratio +
            weights['diversity'] * diversity_score +
            weights['stability'] * stability_score +
            weights['outlier_resistance'] * outlier_resistance
        )
        
        return np.clip(composite_score, 0.0, 1.0)
    
    @staticmethod
    def basic_stabilization(values: np.ndarray) -> np.ndarray:
        """
        第1段階：基本対処（品質スコア > 0.6）
        軽量で高速な処理
        """
        # NaN/Infを0で置換
        cleaned = np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0)
        
        # 有限値での上下1%除去
        finite_mask = np.isfinite(cleaned)
        if np.sum(finite_mask) < 2:
            return cleaned
        
        finite_values = cleaned[finite_mask]
        try:
            p1, p99 = np.percentile(finite_values, [1, 99])
            result = np.clip(cleaned, p1, p99)
        except:
            result = cleaned
        
        return result
    
    @staticmethod
    def robust_stabilization(values: np.ndarray) -> np.ndarray:
        """
        第2段階：フォールバック（品質スコア ≤ 0.6）
        ロバスト統計による処理
        """
        finite_mask = np.isfinite(values)
        if np.sum(finite_mask) < 3:
            # データが不十分な場合は0埋め
            return np.zeros_like(values)
        
        finite_values = values[finite_mask]
        
        try:
            # ロバスト統計による処理
            median_val = np.median(finite_values)
            
            # MAD計算
            abs_deviations = np.abs(finite_values - median_val)
            mad_val = np.median(abs_deviations)
            
            if mad_val < 1e-10:
                mad_val = np.std(finite_values) * 0.6745  # フォールバック
            
            # MADベースの外れ値除去
            robust_bounds = (median_val - 3 * mad_val, median_val + 3 * mad_val)
            
            # 全データに適用
            result = np.copy(values)
            result = np.nan_to_num(result, nan=median_val, posinf=robust_bounds[1], neginf=robust_bounds[0])
            result = np.clip(result, robust_bounds[0], robust_bounds[1])
            
        except Exception:
            # 最終フォールバック: 中央値埋め
            median_val = np.median(finite_values) if len(finite_values) > 0 else 0.0
            result = np.full_like(values, median_val)
        
        return result

# ===============================
# Numba UDF定義（クラス外必須）
# ===============================

# 【重要】全てのNumba UDFは必ずクラス外で定義すること
# クラス内定義は循環参照エラーを引き起こすため絶対禁止

# =============================================================================
# ボラティリティ指標 (Numba UDF集)
# =============================================================================

@nb.njit(fastmath=True, cache=True)
def hv_standard_udf(returns: np.ndarray) -> float:
    """
    標準ヒストリカルボラティリティ計算
    標準偏差ベースの伝統的ボラティリティ指標
    """
    if len(returns) < 5:
        return np.nan
    
    finite_returns = returns[np.isfinite(returns)]
    if len(finite_returns) < 5:
        return np.nan
    
    # 標準偏差計算（不偏推定量）
    mean_return = np.mean(finite_returns)
    squared_deviations = (finite_returns - mean_return) ** 2
    variance = np.sum(squared_deviations) / (len(finite_returns) - 1)
    volatility = np.sqrt(variance)
    
    return volatility

@nb.njit(fastmath=True, cache=True)
def hv_robust_udf(returns: np.ndarray) -> float:
    """
    ロバストボラティリティ計算
    中央絶対偏差（MAD）ベースの外れ値耐性ボラティリティ
    """
    if len(returns) < 5:
        return np.nan
    
    finite_returns = returns[np.isfinite(returns)]
    if len(finite_returns) < 5:
        return np.nan
    
    # MADベースロバスト標準偏差
    median_return = np.median(finite_returns)
    abs_deviations = np.abs(finite_returns - median_return)
    mad = np.median(abs_deviations)
    
    # MADを標準偏差に変換（正規分布仮定下）
    robust_volatility = mad * 1.4826  # 1.4826 = 1/Φ^(-1)(0.75)
    
    return robust_volatility

@nb.njit(fastmath=True, cache=True, parallel=True)
def chaikin_volatility_udf(high: np.ndarray, low: np.ndarray, window: int) -> np.ndarray:
    """
    Chaikin Volatility計算
    High-Low範囲の移動平均の変化率ベースのボラティリティ指標
    """
    n = len(high)
    result = np.full(n, np.nan)
    
    if n < window * 2:
        return result
    
    # High-Low範囲計算
    hl_range = high - low
    
    # 範囲の移動平均計算
    for i in nb.prange(window - 1, n):
        if i >= window * 2 - 1:
            # 現在期間の移動平均
            current_sum = 0.0
            current_count = 0
            for j in range(i - window + 1, i + 1):
                if np.isfinite(hl_range[j]):
                    current_sum += hl_range[j]
                    current_count += 1
            
            if current_count < window // 2:
                continue
                
            current_avg = current_sum / current_count
            
            # 前期間の移動平均
            prev_sum = 0.0
            prev_count = 0
            for j in range(i - window * 2 + 1, i - window + 1):
                if np.isfinite(hl_range[j]):
                    prev_sum += hl_range[j]
                    prev_count += 1
            
            if prev_count < window // 2 or prev_sum <= 0:
                continue
                
            prev_avg = prev_sum / prev_count
            
            # Chaikin Volatility = (現在の移動平均 - 前期間の移動平均) / 前期間の移動平均 * 100
            if prev_avg > 0:
                result[i] = (current_avg - prev_avg) / prev_avg * 100.0
    
    return result

@nb.njit(fastmath=True, cache=True, parallel=True)
def mass_index_udf(high: np.ndarray, low: np.ndarray, window: int) -> np.ndarray:
    """
    Mass Index計算
    High-Low範囲の指数移動平均比率の累積和による反転指標
    """
    n = len(high)
    result = np.full(n, np.nan)
    
    if n < window:
        return result
    
    # High-Low範囲
    hl_range = high - low
    
    # EMA(9)とEMA(EMA(9))の計算用
    alpha = 2.0 / (9.0 + 1.0)
    
    for i in nb.prange(window - 1, n):
        # 25期間のMass Index累積計算
        mass_sum = 0.0
        valid_count = 0
        
        for k in range(max(0, i - window + 1), i + 1):
            if k < 8:  # EMA計算に必要な最小期間
                continue
                
            # EMA(9)計算
            ema9 = hl_range[k - 8]
            for j in range(k - 7, k + 1):
                if np.isfinite(hl_range[j]):
                    ema9 = alpha * hl_range[j] + (1.0 - alpha) * ema9
            
            # EMA(EMA(9))計算
            if k >= 16:  # EMA of EMA計算に必要な期間
                ema_ema9 = ema9
                start_idx = max(k - 16, 0)
                
                # 簡略化されたEMA(EMA)近似
                for j in range(start_idx, k + 1):
                    if j >= 8:
                        temp_ema = hl_range[j - 8]
                        for m in range(j - 7, j + 1):
                            if np.isfinite(hl_range[m]):
                                temp_ema = alpha * hl_range[m] + (1.0 - alpha) * temp_ema
                        ema_ema9 = alpha * temp_ema + (1.0 - alpha) * ema_ema9
                
                # Mass Index = EMA(9) / EMA(EMA(9))
                if ema_ema9 > 0 and np.isfinite(ema9):
                    mass_sum += ema9 / ema_ema9
                    valid_count += 1
        
        if valid_count >= window // 2:
            result[i] = mass_sum
    
    return result

# =============================================================================
# 出来高関連指標 (Numba UDF集)
# =============================================================================

@nb.njit(fastmath=True, cache=True, parallel=True)
def cmf_udf(high: np.ndarray, low: np.ndarray, close: np.ndarray, volume: np.ndarray, window: int) -> np.ndarray:
    """
    Chaikin Money Flow (CMF) 計算
    出来高加重された価格位置の移動平均
    """
    n = len(close)
    result = np.full(n, np.nan)
    
    if n < window:
        return result
    
    for i in nb.prange(window - 1, n):
        mf_volume_sum = 0.0
        volume_sum = 0.0
        
        for j in range(i - window + 1, i + 1):
            if np.isfinite(high[j]) and np.isfinite(low[j]) and np.isfinite(close[j]) and np.isfinite(volume[j]):
                if high[j] != low[j]:
                    # Money Flow Multiplier = [(Close - Low) - (High - Close)] / (High - Low)
                    mf_multiplier = ((close[j] - low[j]) - (high[j] - close[j])) / (high[j] - low[j])
                    mf_volume = mf_multiplier * volume[j]
                    mf_volume_sum += mf_volume
                    volume_sum += volume[j]
        
        if volume_sum > 0:
            result[i] = mf_volume_sum / volume_sum
    
    return result

@nb.njit(fastmath=True, cache=True, parallel=True)
def mfi_udf(high: np.ndarray, low: np.ndarray, close: np.ndarray, volume: np.ndarray, window: int) -> np.ndarray:
    """
    Money Flow Index (MFI) 計算
    出来高加重RSI（価格×出来高ベース）
    """
    n = len(close)
    result = np.full(n, np.nan)
    
    if n < window + 1:
        return result
    
    # Typical Price = (High + Low + Close) / 3
    for i in nb.prange(window, n):
        positive_flow = 0.0
        negative_flow = 0.0
        
        for j in range(i - window + 1, i + 1):
            if j == 0:
                continue
                
            if (np.isfinite(high[j]) and np.isfinite(low[j]) and np.isfinite(close[j]) and 
                np.isfinite(volume[j]) and np.isfinite(high[j-1]) and 
                np.isfinite(low[j-1]) and np.isfinite(close[j-1])):
                
                typical_price = (high[j] + low[j] + close[j]) / 3.0
                prev_typical_price = (high[j-1] + low[j-1] + close[j-1]) / 3.0
                raw_money_flow = typical_price * volume[j]
                
                if typical_price > prev_typical_price:
                    positive_flow += raw_money_flow
                elif typical_price < prev_typical_price:
                    negative_flow += raw_money_flow
        
        if negative_flow > 0:
            money_ratio = positive_flow / negative_flow
            result[i] = 100.0 - (100.0 / (1.0 + money_ratio))
        elif positive_flow > 0:
            result[i] = 100.0
        else:
            result[i] = 50.0
    
    return result

@nb.njit(fastmath=True, cache=True, parallel=True)
def vwap_udf(high: np.ndarray, low: np.ndarray, close: np.ndarray, volume: np.ndarray, window: int) -> np.ndarray:
    """
    Volume Weighted Average Price (VWAP) 計算
    出来高加重平均価格
    """
    n = len(close)
    result = np.full(n, np.nan)
    
    if n < window:
        return result
    
    for i in nb.prange(window - 1, n):
        price_volume_sum = 0.0
        volume_sum = 0.0
        
        for j in range(i - window + 1, i + 1):
            if (np.isfinite(high[j]) and np.isfinite(low[j]) and 
                np.isfinite(close[j]) and np.isfinite(volume[j])):
                # Typical Price = (High + Low + Close) / 3
                typical_price = (high[j] + low[j] + close[j]) / 3.0
                price_volume_sum += typical_price * volume[j]
                volume_sum += volume[j]
        
        if volume_sum > 0:
            result[i] = price_volume_sum / volume_sum
    
    return result

@nb.njit(fastmath=True, cache=True, parallel=True)
def obv_udf(close: np.ndarray, volume: np.ndarray) -> np.ndarray:
    """
    On Balance Volume (OBV) 計算
    価格方向に基づく累積出来高指標
    """
    n = len(close)
    result = np.full(n, np.nan)
    
    if n < 2:
        return result
    
    result[0] = volume[0] if np.isfinite(volume[0]) else 0.0
    
    for i in nb.prange(1, n):
        prev_obv = result[i - 1] if np.isfinite(result[i - 1]) else 0.0
        
        if np.isfinite(close[i]) and np.isfinite(close[i - 1]) and np.isfinite(volume[i]):
            if close[i] > close[i - 1]:
                result[i] = prev_obv + volume[i]
            elif close[i] < close[i - 1]:
                result[i] = prev_obv - volume[i]
            else:
                result[i] = prev_obv
        else:
            result[i] = prev_obv
    
    return result

@nb.njit(fastmath=True, cache=True, parallel=True)
def accumulation_distribution_udf(high: np.ndarray, low: np.ndarray, close: np.ndarray, volume: np.ndarray) -> np.ndarray:
    """
    Accumulation/Distribution Line 計算
    価格位置と出来高に基づく累積指標
    """
    n = len(close)
    result = np.full(n, np.nan)
    
    if n < 1:
        return result
    
    result[0] = 0.0
    
    for i in nb.prange(1, n):
        prev_ad = result[i - 1] if np.isfinite(result[i - 1]) else 0.0
        
        if (np.isfinite(high[i]) and np.isfinite(low[i]) and 
            np.isfinite(close[i]) and np.isfinite(volume[i])):
            
            if high[i] != low[i]:
                # Close Location Value = [(Close - Low) - (High - Close)] / (High - Low)
                clv = ((close[i] - low[i]) - (high[i] - close[i])) / (high[i] - low[i])
                result[i] = prev_ad + (clv * volume[i])
            else:
                result[i] = prev_ad
        else:
            result[i] = prev_ad
    
    return result

@nb.njit(fastmath=True, cache=True, parallel=True)
def force_index_udf(close: np.ndarray, volume: np.ndarray) -> np.ndarray:
    """
    Force Index 計算
    価格変動と出来高の積による勢力指標
    """
    n = len(close)
    result = np.full(n, np.nan)
    
    if n < 2:
        return result
    
    result[0] = 0.0
    
    for i in nb.prange(1, n):
        if (np.isfinite(close[i]) and np.isfinite(close[i - 1]) and 
            np.isfinite(volume[i])):
            price_change = close[i] - close[i - 1]
            result[i] = price_change * volume[i]
        else:
            result[i] = 0.0
    
    return result

# =============================================================================
# サポート・レジスタンス・ローソク足 (Numba UDF集)
# =============================================================================

@nb.njit(fastmath=True, cache=True)
def pivot_point_udf(high: float, low: float, close: float) -> Tuple[float, float, float, float, float]:
    """
    ピボットポイント計算
    Returns: (pivot_point, resistance1, support1, resistance2, support2)
    """
    if not (np.isfinite(high) and np.isfinite(low) and np.isfinite(close)):
        return np.nan, np.nan, np.nan, np.nan, np.nan
    
    pivot = (high + low + close) / 3.0
    r1 = 2.0 * pivot - low
    s1 = 2.0 * pivot - high
    r2 = pivot + (high - low)
    s2 = pivot - (high - low)
    
    return pivot, r1, s1, r2, s2

@nb.njit(fastmath=True, cache=True, parallel=True)
def fibonacci_levels_udf(high: np.ndarray, low: np.ndarray, window: int) -> np.ndarray:
    """
    フィボナッチリトレースメントレベル計算
    指定期間での高値・安値からフィボナッチレベルを計算
    """
    n = len(high)
    result = np.full((n, 5), np.nan)  # 5つのフィボナッチレベル
    
    if n < window:
        return result
    
    # フィボナッチ比率
    fib_ratios = np.array([0.236, 0.382, 0.500, 0.618, 0.786])
    
    for i in nb.prange(window - 1, n):
        # 指定期間での高値・安値を探索
        period_high = -np.inf
        period_low = np.inf
        
        for j in range(i - window + 1, i + 1):
            if np.isfinite(high[j]) and high[j] > period_high:
                period_high = high[j]
            if np.isfinite(low[j]) and low[j] < period_low:
                period_low = low[j]
        
        if np.isfinite(period_high) and np.isfinite(period_low):
            price_range = period_high - period_low
            for k in range(5):
                result[i, k] = period_high - (fib_ratios[k] * price_range)
    
    return result

@nb.njit(fastmath=True, cache=True, parallel=True)
def candlestick_patterns_udf(open_prices: np.ndarray, high: np.ndarray, low: np.ndarray, close: np.ndarray) -> np.ndarray:
    """
    ローソク足パターン認識
    Returns: パターンID配列 (0=なし, 1=ハンマー, 2=流れ星, 3=同事, 4=強気, 5=弱気)
    """
    n = len(close)
    result = np.full(n, 0)
    
    for i in nb.prange(n):
        if not (np.isfinite(open_prices[i]) and np.isfinite(high[i]) and 
                np.isfinite(low[i]) and np.isfinite(close[i])):
            continue
        
        body_size = abs(close[i] - open_prices[i])
        upper_shadow = high[i] - max(open_prices[i], close[i])
        lower_shadow = min(open_prices[i], close[i]) - low[i]
        total_range = high[i] - low[i]
        
        if total_range <= 0:
            continue
        
        # 実体・ヒゲ比率
        body_ratio = body_size / total_range
        upper_shadow_ratio = upper_shadow / total_range
        lower_shadow_ratio = lower_shadow / total_range
        
        # 同事（実体が小さい）
        if body_ratio < 0.1:
            result[i] = 3
        # ハンマー（下ヒゲが長く、上ヒゲが短い）
        elif lower_shadow_ratio > 0.6 and upper_shadow_ratio < 0.1:
            result[i] = 1
        # 流れ星（上ヒゲが長く、下ヒゲが短い）
        elif upper_shadow_ratio > 0.6 and lower_shadow_ratio < 0.1:
            result[i] = 2
        # 強気（実体が大きく陽線）
        elif body_ratio > 0.6 and close[i] > open_prices[i]:
            result[i] = 4
        # 弱気（実体が大きく陰線）
        elif body_ratio > 0.6 and close[i] < open_prices[i]:
            result[i] = 5
    
    return result

# =============================================================================
# ブレイクアウト・レンジ指標 (Numba UDF集)
# =============================================================================

@nb.njit(fastmath=True, cache=True, parallel=True)
def donchian_channel_udf(high: np.ndarray, low: np.ndarray, window: int) -> np.ndarray:
    """
    ドンチャンチャネル計算
    Returns: 各行に[upper, middle, lower]を含む配列
    """
    n = len(high)
    result = np.full((n, 3), np.nan)
    
    if n < window:
        return result
    
    for i in nb.prange(window - 1, n):
        period_high = -np.inf
        period_low = np.inf
        
        for j in range(i - window + 1, i + 1):
            if np.isfinite(high[j]) and high[j] > period_high:
                period_high = high[j]
            if np.isfinite(low[j]) and low[j] < period_low:
                period_low = low[j]
        
        if np.isfinite(period_high) and np.isfinite(period_low):
            result[i, 0] = period_high  # upper
            result[i, 1] = (period_high + period_low) / 2.0  # middle
            result[i, 2] = period_low  # lower
    
    return result

@nb.njit(fastmath=True, cache=True, parallel=True)
def commodity_channel_index_udf(high: np.ndarray, low: np.ndarray, close: np.ndarray, window: int) -> np.ndarray:
    """
    Commodity Channel Index (CCI) 計算
    価格位置の統計的偏差を測定する勢い指標
    """
    n = len(close)
    result = np.full(n, np.nan)
    
    if n < window:
        return result
    
    for i in nb.prange(window - 1, n):
        # Typical Price計算
        typical_prices = np.zeros(window)
        for j in range(window):
            idx = i - window + 1 + j
            if (np.isfinite(high[idx]) and np.isfinite(low[idx]) and np.isfinite(close[idx])):
                typical_prices[j] = (high[idx] + low[idx] + close[idx]) / 3.0
            else:
                typical_prices[j] = np.nan
        
        # 有効なTypical Priceの数を確認
        valid_count = 0
        for tp in typical_prices:
            if np.isfinite(tp):
                valid_count += 1
        
        if valid_count < window // 2:
            continue
        
        # SMA計算
        tp_sum = 0.0
        for tp in typical_prices:
            if np.isfinite(tp):
                tp_sum += tp
        sma = tp_sum / valid_count
        
        # Mean Deviation計算
        md_sum = 0.0
        for tp in typical_prices:
            if np.isfinite(tp):
                md_sum += abs(tp - sma)
        mean_deviation = md_sum / valid_count
        
        # CCI計算
        current_tp = (high[i] + low[i] + close[i]) / 3.0
        if mean_deviation > 0:
            result[i] = (current_tp - sma) / (0.015 * mean_deviation)
    
    return result

class CalculationEngine:
    """
    計算核心クラス（60%） - Project Forge統合版（修正版：ディスクベース垂直分割）
    責務:
    - Polars Expressionによる高度な特徴量計算（90%のタスク）
    - .map_batches()経由のNumba JIT最適化UDF（10%のカスタム「アルファ」ロジック）
    - 【修正】物理的垂直分割：ディスク中間ファイルによるメモリ・スラッシング回避
    - Polars内部並列化による自動マルチスレッド実行
    - 2段階品質保証（基本/フォールバック）
    """
    
    def __init__(self, config: ProcessingConfig):
        self.config = config
        self.qa = QualityAssurance()
        self.memory_monitor = MemoryMonitor(config.memory_limit_gb)
        self.prefix = f"e{config.engine_id.replace('e', '')}_"  # 例: "e1d_"
        
        # 【修正】ディスクベース垂直分割用の一時ディレクトリ
        self.temp_dir = Path(tempfile.mkdtemp(prefix=f"features_{config.engine_id}_"))
        logger.info(f"一時ディレクトリ作成: {self.temp_dir}")
    
    def _get_all_feature_expressions(self) -> Dict[str, pl.Expr]:
        """
        全特徴量の名前とPolars Expressionの対応辞書を返すファクトリメソッド。
        これにより、必要な特徴量だけを効率的に計算できるようになる。
        【特徴量ファクトリパターン】
        """
        expressions = {}
        p = self.prefix
        
        # ボラティリティ指標系 - 全ての式に明示的なaliasを付与
        for window in self.config.window_sizes["volatility"]:
            # 標準ヒストリカルボラティリティ
            expressions[f"{p}hv_standard_{window}"] = pl.col("close").pct_change().rolling_map(
                lambda s: hv_standard_udf(s.to_numpy()),
                window_size=window,
                min_samples=window
            ).alias(f"{p}hv_standard_{window}")
            
            # ロバストボラティリティ
            expressions[f"{p}hv_robust_{window}"] = pl.col("close").pct_change().rolling_map(
                lambda s: hv_robust_udf(s.to_numpy()),
                window_size=window,
                min_samples=window
            ).alias(f"{p}hv_robust_{window}")
        
        # 年率ボラティリティ（標準）
        expressions[f"{p}hv_annual_252"] = (pl.col("close").pct_change().rolling_std(252) * np.sqrt(252)).alias(f"{p}hv_annual_252")
        
        # 年率ロバストボラティリティ
        expressions[f"{p}hv_robust_annual_252"] = pl.col("close").pct_change().rolling_map(
            lambda s: hv_robust_udf(s.to_numpy()) * np.sqrt(252),
            window_size=252,
            min_samples=252
        ).alias(f"{p}hv_robust_annual_252")
        
        # ボラティリティレジーム（高・中・低）- Noneタイプエラー対策
        def volatility_regime_udf(s):
            try:
                # 有効な値を確認
                valid_values = s.drop_nulls()
                if len(valid_values) < 5:  # 最小データ数チェック
                    return pl.Series([0] * len(s), dtype=pl.Int8)
                
                q80 = valid_values.quantile(0.8)
                q60 = valid_values.quantile(0.6)
                
                if q80 is None or q60 is None:  # Noneチェック
                    return pl.Series([0] * len(s), dtype=pl.Int8)
                
                return (s > q80).cast(pl.Int8) + (s > q60).cast(pl.Int8)
            except:
                return pl.Series([0] * len(s), dtype=pl.Int8)
        
        expressions[f"{p}hv_regime_50"] = pl.col("close").pct_change().rolling_std(50).map_batches(
            volatility_regime_udf
        ).alias(f"{p}hv_regime_50")
        
        # Chaikin Volatility（重量UDF）
        for window in [10, 20]:
            expressions[f"{p}chaikin_volatility_{window}"] = pl.struct(["high", "low"]).map_batches(
                lambda s: chaikin_volatility_udf(
                    s.struct.field("high").to_numpy(),
                    s.struct.field("low").to_numpy(),
                    window
                )
            ).alias(f"{p}chaikin_volatility_{window}")
        
        # Mass Index（重量UDF）
        for window in [20, 30]:
            expressions[f"{p}mass_index_{window}"] = pl.struct(["high", "low"]).map_batches(
                lambda s: mass_index_udf(
                    s.struct.field("high").to_numpy(),
                    s.struct.field("low").to_numpy(),
                    window
                )
            ).alias(f"{p}mass_index_{window}")
        
        # 出来高関連指標系 - 全ての式に明示的なaliasを付与（括弧で囲んで確実にalias適用）
        for window in self.config.window_sizes["volume"]:
            # CMF（重量UDF）
            expressions[f"{p}cmf_{window}"] = pl.struct(["high", "low", "close", "volume"]).map_batches(
                lambda s: cmf_udf(
                    s.struct.field("high").to_numpy(),
                    s.struct.field("low").to_numpy(),
                    s.struct.field("close").to_numpy(),
                    s.struct.field("volume").to_numpy(),
                    window
                )
            ).alias(f"{p}cmf_{window}")
            
            # MFI（重量UDF）
            expressions[f"{p}mfi_{window}"] = pl.struct(["high", "low", "close", "volume"]).map_batches(
                lambda s: mfi_udf(
                    s.struct.field("high").to_numpy(),
                    s.struct.field("low").to_numpy(),
                    s.struct.field("close").to_numpy(),
                    s.struct.field("volume").to_numpy(),
                    window
                )
            ).alias(f"{p}mfi_{window}")
            
            # VWAP（重量UDF）
            expressions[f"{p}vwap_{window}"] = pl.struct(["high", "low", "close", "volume"]).map_batches(
                lambda s: vwap_udf(
                    s.struct.field("high").to_numpy(),
                    s.struct.field("low").to_numpy(),
                    s.struct.field("close").to_numpy(),
                    s.struct.field("volume").to_numpy(),
                    window
                )
            ).alias(f"{p}vwap_{window}")
        
        # VWAP乖離
        expressions[f"{p}vwap_deviation_21"] = (pl.col("close") - pl.struct(["high", "low", "close", "volume"]).map_batches(
            lambda s: vwap_udf(
                s.struct.field("high").to_numpy(),
                s.struct.field("low").to_numpy(),
                s.struct.field("close").to_numpy(),
                s.struct.field("volume").to_numpy(),
                21
            )
        )).alias(f"{p}vwap_deviation_21")
        
        # On Balance Volume（軽量UDF） - 修正: rolling_map -> map_batches
        expressions[f"{p}obv"] = pl.struct(["close", "volume"]).map_batches(
            lambda s: obv_udf(
                s.struct.field("close").to_numpy(),
                s.struct.field("volume").to_numpy()
            )
        ).alias(f"{p}obv")
        
        # Accumulation/Distribution Line（軽量UDF） - 修正: rolling_map -> map_batches
        expressions[f"{p}accumulation_distribution"] = pl.struct(["high", "low", "close", "volume"]).map_batches(
            lambda s: accumulation_distribution_udf(
                s.struct.field("high").to_numpy(),
                s.struct.field("low").to_numpy(),
                s.struct.field("close").to_numpy(),
                s.struct.field("volume").to_numpy()
            )
        ).alias(f"{p}accumulation_distribution")
        
        # Force Index（軽量UDF） - 修正: rolling_map -> map_batches
        expressions[f"{p}force_index"] = pl.struct(["close", "volume"]).map_batches(
            lambda s: force_index_udf(
                s.struct.field("close").to_numpy(),
                s.struct.field("volume").to_numpy()
            )
        ).alias(f"{p}force_index")
        
        # ブレイクアウト・レンジ指標 - 全ての式に明示的なaliasを付与
        for window in self.config.window_sizes["general"]:
            # Donchian Channel（重量UDF）
            expressions[f"{p}donchian_upper_{window}"] = pl.struct(["high", "low"]).map_batches(
                lambda s: donchian_channel_udf(
                    s.struct.field("high").to_numpy(),
                    s.struct.field("low").to_numpy(),
                    window
                )[:, 0]
            ).alias(f"{p}donchian_upper_{window}")
            
            expressions[f"{p}donchian_middle_{window}"] = pl.struct(["high", "low"]).map_batches(
                lambda s: donchian_channel_udf(
                    s.struct.field("high").to_numpy(),
                    s.struct.field("low").to_numpy(),
                    window
                )[:, 1]
            ).alias(f"{p}donchian_middle_{window}")
            
            expressions[f"{p}donchian_lower_{window}"] = pl.struct(["high", "low"]).map_batches(
                lambda s: donchian_channel_udf(
                    s.struct.field("high").to_numpy(),
                    s.struct.field("low").to_numpy(),
                    window
                )[:, 2]
            ).alias(f"{p}donchian_lower_{window}")
            
            # Price Channel（Polarsネイティブ）
            expressions[f"{p}price_channel_upper_{window}"] = pl.col("high").rolling_max(window).alias(f"{p}price_channel_upper_{window}")
            expressions[f"{p}price_channel_lower_{window}"] = pl.col("low").rolling_min(window).alias(f"{p}price_channel_lower_{window}")
        
        # Commodity Channel Index（重量UDF）
        for window in [14, 20]:
            expressions[f"{p}commodity_channel_index_{window}"] = pl.struct(["high", "low", "close"]).map_batches(
                lambda s: commodity_channel_index_udf(
                    s.struct.field("high").to_numpy(),
                    s.struct.field("low").to_numpy(),
                    s.struct.field("close").to_numpy(),
                    window
                )
            ).alias(f"{p}commodity_channel_index_{window}")
        
        # サポート・レジスタンス・ローソク足 - 全ての式に明示的なaliasを付与
        # 日次ピボットポイント（前日の高安終値から計算） - 修正: Polarsネイティブ関数に置換
        prev_high = pl.col("high").shift(1)
        prev_low = pl.col("low").shift(1)
        prev_close = pl.col("close").shift(1)
        pivot = (prev_high + prev_low + prev_close) / 3.0

        expressions[f"{p}pivot_point"] = pivot.alias(f"{p}pivot_point")
        expressions[f"{p}resistance1"] = (2.0 * pivot - prev_low).alias(f"{p}resistance1")
        expressions[f"{p}support1"] = (2.0 * pivot - prev_high).alias(f"{p}support1")
        
        # フィボナッチレベル（重量UDF）
        expressions[f"{p}fib_level_50"] = pl.struct(["high", "low"]).map_batches(
            lambda s: fibonacci_levels_udf(
                s.struct.field("high").to_numpy(),
                s.struct.field("low").to_numpy(),
                50
            )[:, 2]  # 50%レベル
        ).alias(f"{p}fib_level_50")
        
        # ローソク足パターン（重量UDF）
        expressions[f"{p}candlestick_pattern"] = pl.struct(["open", "high", "low", "close"]).map_batches(
            lambda s: candlestick_patterns_udf(
                s.struct.field("open").to_numpy(),
                s.struct.field("high").to_numpy(),
                s.struct.field("low").to_numpy(),
                s.struct.field("close").to_numpy()
            )
        ).alias(f"{p}candlestick_pattern")
        
        # 価格アクション指標 - 全ての式に明示的なaliasを付与
        expressions[f"{p}typical_price"] = ((pl.col("high") + pl.col("low") + pl.col("close")) / 3.0).alias(f"{p}typical_price")
        expressions[f"{p}weighted_close"] = ((pl.col("high") + pl.col("low") + 2 * pl.col("close")) / 4.0).alias(f"{p}weighted_close")
        expressions[f"{p}median_price"] = ((pl.col("high") + pl.col("low")) / 2.0).alias(f"{p}median_price")
        
        # ローソク足構成要素
        expressions[f"{p}body_size"] = (pl.col("close") - pl.col("open")).abs().alias(f"{p}body_size")
        expressions[f"{p}upper_wick_ratio"] = ((pl.col("high") - pl.max_horizontal("open", "close")) / (pl.col("high") - pl.col("low"))).alias(f"{p}upper_wick_ratio")
        expressions[f"{p}lower_wick_ratio"] = ((pl.min_horizontal("open", "close") - pl.col("low")) / (pl.col("high") - pl.col("low"))).alias(f"{p}lower_wick_ratio")
        expressions[f"{p}price_location_hl"] = ((pl.col("close") - pl.col("low")) / (pl.col("high") - pl.col("low"))).alias(f"{p}price_location_hl")
        
        # イントラデイ・オーバーナイト
        expressions[f"{p}intraday_return"] = ((pl.col("close") - pl.col("open")) / pl.col("open")).alias(f"{p}intraday_return")
        expressions[f"{p}overnight_gap"] = ((pl.col("open") - pl.col("close").shift(1)) / pl.col("close").shift(1)).alias(f"{p}overnight_gap")
        
        # 出来高比率
        expressions[f"{p}volume_ma20"] = pl.col("volume").rolling_mean(20).alias(f"{p}volume_ma20")
        expressions[f"{p}volume_ratio"] = (pl.col("volume") / pl.col("volume").rolling_mean(20)).alias(f"{p}volume_ratio")
        expressions[f"{p}volume_price_trend"] = (pl.col("close") * pl.col("volume")).rolling_mean(10).alias(f"{p}volume_price_trend")
        
        return expressions
    
    def get_feature_groups(self) -> Dict[str, Dict[str, pl.Expr]]:
        """特徴量グループ定義を外部から取得可能にする"""
        return self._create_vertical_slices()
    
    def calculate_one_group(self, lazy_frame: pl.LazyFrame, group_name: str, group_expressions: Dict[str, pl.Expr]) -> pl.LazyFrame:
        """
        単一グループの特徴量のみを計算（高速化修正版）
        グループ名に基づいて適切な特徴量計算メソッドを呼び出し、重複計算を回避
        """
        logger.info(f"グループ計算開始: {group_name}")
        
        # メモリ安全性チェック（必須）
        is_safe, message = self.memory_monitor.check_memory_safety()
        if not is_safe:
            raise MemoryError(f"メモリ不足のためグループ処理を中断: {message}")
        
        try:
            # グループ名に基づいて効率的な特徴量計算を実行
            if group_name == "volatility":
                group_result_lf = self._create_volatility_features(lazy_frame)
            elif group_name == "volume":
                group_result_lf = self._create_volume_features(lazy_frame)
            elif group_name == "breakout":
                group_result_lf = self._create_breakout_features(lazy_frame)
            elif group_name == "support_resistance":
                group_result_lf = self._create_support_resistance_features(lazy_frame)
            elif group_name == "price_action":
                group_result_lf = self._create_price_action_features(lazy_frame)
            else:
                # フォールバック: 従来の方式
                logger.warning(f"未対応グループ名、フォールバック処理: {group_name}")
                group_result_lf = lazy_frame.with_columns(list(group_expressions.values()))
            
            # スキーマから実際に存在するカラムを確認
            available_schema = group_result_lf.collect_schema()
            available_columns = list(available_schema.names())
            
            # 基本カラムとして存在するもののみを選択
            base_columns = ["timestamp", "open", "high", "low", "close", "volume"]
            if "timeframe" in available_columns:
                base_columns.append("timeframe")
            
            # このグループの特徴量カラムのみを抽出
            group_feature_columns = [col for col in available_columns if col.startswith(self.prefix)]
            select_columns = base_columns + group_feature_columns
            
            # 実際に存在するカラムのみを選択
            final_select_columns = [col for col in select_columns if col in available_columns]
            group_final_lf = group_result_lf.select(final_select_columns)
            
            # 品質保証を適用（このグループのみ）
            stabilized_lf = self.apply_quality_assurance_to_group(group_final_lf, group_feature_columns)
            
            logger.info(f"グループ計算完了: {group_name} - {len(group_feature_columns)}個の特徴量")
            return stabilized_lf
            
        except Exception as e:
            logger.error(f"グループ計算エラー ({group_name}): {e}")
            raise
    
    def apply_quality_assurance_to_group(self, lazy_frame: pl.LazyFrame, feature_columns: List[str]) -> pl.LazyFrame:
        """単一グループに対する品質保証システムの適用"""
        if not feature_columns:
            return lazy_frame
        
        logger.info(f"品質保証適用: {len(feature_columns)}個の特徴量")
        
        # 安定化処理の式を生成
        stabilization_exprs = []
        
        for col_name in feature_columns:
            # Inf値を除外してパーセンタイル計算（精度保持）
            col_for_quantile = pl.when(pl.col(col_name).is_infinite()).then(None).otherwise(pl.col(col_name))
            
            p01 = col_for_quantile.quantile(0.01, interpolation="linear")
            p99 = col_for_quantile.quantile(0.99, interpolation="linear")
            
            # Inf値を統計的に意味のある値（パーセンタイル境界値）で置換
            stabilized_col = (
                pl.when(pl.col(col_name) == float('inf'))
                .then(p99)  # +Infは99パーセンタイル値で置換
                .when(pl.col(col_name) == float('-inf'))
                .then(p01)  # -Infは1パーセンタイル値で置換
                .otherwise(pl.col(col_name))
                .clip(lower_bound=p01, upper_bound=p99)
                .fill_null(0.0)
                .fill_nan(0.0)
                .alias(col_name)
            )
            
            stabilization_exprs.append(stabilized_col)
        
        result = lazy_frame.with_columns(stabilization_exprs)
        return result
    
    def _create_vertical_slices(self) -> Dict[str, Dict[str, pl.Expr]]:
        """物理的垂直分割: 特徴量を論理グループに分割"""
        all_expressions = self._get_all_feature_expressions()
        
        # メモリ使用量を考慮したグルーピング（英語キー使用）
        slices = {}
        p = self.prefix
        
        # グループ1: ボラティリティ系（重量）
        slices["volatility"] = {
            name: expr for name, expr in all_expressions.items() 
            if any(pattern in name for pattern in ["hv_", "chaikin_volatility", "mass_index"])
        }
        
        # グループ2: 出来高系（重量）
        slices["volume"] = {
            name: expr for name, expr in all_expressions.items() 
            if any(pattern in name for pattern in ["cmf_", "mfi_", "vwap", "obv", "accumulation_distribution", "force_index", "volume_"])
        }
        
        # グループ3: ブレイクアウト・レンジ系（中程度）
        slices["breakout"] = {
            name: expr for name, expr in all_expressions.items() 
            if any(pattern in name for pattern in ["donchian_", "price_channel_", "commodity_channel_index"])
        }
        
        # グループ4: サポート・レジスタンス系（中程度）
        slices["support_resistance"] = {
            name: expr for name, expr in all_expressions.items() 
            if any(pattern in name for pattern in ["pivot_", "resistance", "support", "fib_", "candlestick_pattern"])
        }
        
        # グループ5: 価格アクション系（軽量）
        slices["price_action"] = {
            name: expr for name, expr in all_expressions.items() 
            if any(pattern in name for pattern in ["typical_price", "weighted_close", "median_price", "body_", "wick_", "price_location", "intraday_", "overnight_"])
        }
        
        # 分割されなかった特徴量があれば警告
        total_assigned = sum(len(group) for group in slices.values())
        if total_assigned != len(all_expressions):
            logger.warning(f"未分割特徴量: {len(all_expressions) - total_assigned}個")
        
        return slices
    
    def _cleanup_temp_files(self):
        """一時ファイルのクリーンアップ"""
        try:
            if self.temp_dir.exists():
                for temp_file in self.temp_dir.glob("*.parquet"):
                    temp_file.unlink()
                    logger.debug(f"一時ファイル削除: {temp_file}")
                self.temp_dir.rmdir()
                logger.info(f"一時ディレクトリ削除: {self.temp_dir}")
        except Exception as e:
            logger.warning(f"一時ファイルクリーンアップエラー: {e}")
    
    def apply_quality_assurance(self, lazy_frame: pl.LazyFrame) -> pl.LazyFrame:
        """2段階品質保証システムの適用"""
        logger.info("品質保証システム適用開始: 全ての特徴量に安定化処理を適用します。")
        
        # スキーマからプレフィックスを持つ特徴量カラムを特定
        schema = lazy_frame.collect_schema()
        feature_columns = [col for col in schema.names() if col.startswith(self.prefix)]
        
        if not feature_columns:
            logger.warning("品質保証対象の特徴量が見つかりません。")
            return lazy_frame
        
        logger.info(f"品質保証対象: {len(feature_columns)}個の特徴量")
        
        # 安定化処理の式を生成
        stabilization_exprs = []
        
        for col_name in feature_columns:
            # Inf値を除外してパーセンタイル計算（精度保持）
            col_for_quantile = pl.when(pl.col(col_name).is_infinite()).then(None).otherwise(pl.col(col_name))
            
            p01 = col_for_quantile.quantile(0.01, interpolation="linear")
            p99 = col_for_quantile.quantile(0.99, interpolation="linear")
            
            # Inf値を統計的に意味のある値（パーセンタイル境界値）で置換
            stabilized_col = (
                pl.when(pl.col(col_name) == float('inf'))
                .then(p99)  # +Infは99パーセンタイル値で置換
                .when(pl.col(col_name) == float('-inf'))
                .then(p01)  # -Infは1パーセンタイル値で置換
                .otherwise(pl.col(col_name))
                .clip(lower_bound=p01, upper_bound=p99)
                .fill_null(0.0)
                .fill_nan(0.0)
                .alias(col_name)
            )
            
            stabilization_exprs.append(stabilized_col)
        
        result = lazy_frame.with_columns(stabilization_exprs)
        logger.info("品質保証システム適用完了")
        
        return result
    
    def _create_volatility_features(self, lazy_frame: pl.LazyFrame) -> pl.LazyFrame:
        """ボラティリティ系特徴量の計算（高速化対応）"""
        exprs = []
        p = self.prefix
        
        # ボラティリティ指標系 - 全ての式に明示的なaliasを付与
        for window in self.config.window_sizes["volatility"]:
            # 標準ヒストリカルボラティリティ
            exprs.append(pl.col("close").pct_change().rolling_map(
                lambda s: hv_standard_udf(s.to_numpy()),
                window_size=window,
                min_samples=window
            ).alias(f"{p}hv_standard_{window}"))
            
            # ロバストボラティリティ
            exprs.append(pl.col("close").pct_change().rolling_map(
                lambda s: hv_robust_udf(s.to_numpy()),
                window_size=window,
                min_samples=window
            ).alias(f"{p}hv_robust_{window}"))
        
        # 年率ボラティリティ（標準）
        exprs.append((pl.col("close").pct_change().rolling_std(252) * np.sqrt(252)).alias(f"{p}hv_annual_252"))
        
        # 年率ロバストボラティリティ
        exprs.append(pl.col("close").pct_change().rolling_map(
            lambda s: hv_robust_udf(s.to_numpy()) * np.sqrt(252),
            window_size=252,
            min_samples=252
        ).alias(f"{p}hv_robust_annual_252"))
        
        # ボラティリティレジーム（高・中・低）- Noneタイプエラー対策
        def volatility_regime_udf(s):
            try:
                # 有効な値を確認
                valid_values = s.drop_nulls()
                if len(valid_values) < 5:  # 最小データ数チェック
                    return pl.Series([0] * len(s), dtype=pl.Int8)
                
                q80 = valid_values.quantile(0.8)
                q60 = valid_values.quantile(0.6)
                
                if q80 is None or q60 is None:  # Noneチェック
                    return pl.Series([0] * len(s), dtype=pl.Int8)
                
                return (s > q80).cast(pl.Int8) + (s > q60).cast(pl.Int8)
            except:
                return pl.Series([0] * len(s), dtype=pl.Int8)
        
        # ボラティリティレジーム（高・中・低）
        exprs.append(pl.col("close").pct_change().rolling_std(50).map_batches(
            volatility_regime_udf
        ).alias(f"{p}hv_regime_50"))
        
        # Chaikin Volatility（重量UDF）
        for window in [10, 20]:
            exprs.append(pl.struct(["high", "low"]).map_batches(
                lambda s: chaikin_volatility_udf(
                    s.struct.field("high").to_numpy(),
                    s.struct.field("low").to_numpy(),
                    window
                )
            ).alias(f"{p}chaikin_volatility_{window}"))
        
        # Mass Index（重量UDF）
        for window in [20, 30]:
            exprs.append(pl.struct(["high", "low"]).map_batches(
                lambda s: mass_index_udf(
                    s.struct.field("high").to_numpy(),
                    s.struct.field("low").to_numpy(),
                    window
                )
            ).alias(f"{p}mass_index_{window}"))
        
        return lazy_frame.with_columns(exprs)

    def _create_volume_features(self, lazy_frame: pl.LazyFrame) -> pl.LazyFrame:
        """出来高系特徴量の計算（高速化対応）"""
        exprs = []
        p = self.prefix
        
        # 出来高関連指標系 - 全ての式に明示的なaliasを付与（括弧で囲んで確実にalias適用）
        for window in self.config.window_sizes["volume"]:
            # CMF（重量UDF）
            exprs.append(pl.struct(["high", "low", "close", "volume"]).map_batches(
                lambda s: cmf_udf(
                    s.struct.field("high").to_numpy(),
                    s.struct.field("low").to_numpy(),
                    s.struct.field("close").to_numpy(),
                    s.struct.field("volume").to_numpy(),
                    window
                )
            ).alias(f"{p}cmf_{window}"))
            
            # MFI（重量UDF）
            exprs.append(pl.struct(["high", "low", "close", "volume"]).map_batches(
                lambda s: mfi_udf(
                    s.struct.field("high").to_numpy(),
                    s.struct.field("low").to_numpy(),
                    s.struct.field("close").to_numpy(),
                    s.struct.field("volume").to_numpy(),
                    window
                )
            ).alias(f"{p}mfi_{window}"))
            
            # VWAP（重量UDF）
            exprs.append(pl.struct(["high", "low", "close", "volume"]).map_batches(
                lambda s: vwap_udf(
                    s.struct.field("high").to_numpy(),
                    s.struct.field("low").to_numpy(),
                    s.struct.field("close").to_numpy(),
                    s.struct.field("volume").to_numpy(),
                    window
                )
            ).alias(f"{p}vwap_{window}"))
        
        # VWAP乖離
        exprs.append((pl.col("close") - pl.struct(["high", "low", "close", "volume"]).map_batches(
            lambda s: vwap_udf(
                s.struct.field("high").to_numpy(),
                s.struct.field("low").to_numpy(),
                s.struct.field("close").to_numpy(),
                s.struct.field("volume").to_numpy(),
                21
            )
        )).alias(f"{p}vwap_deviation_21"))
        
        # On Balance Volume（軽量UDF） - 修正: rolling_map -> map_batches
        exprs.append(pl.struct(["close", "volume"]).map_batches(
            lambda s: obv_udf(
                s.struct.field("close").to_numpy(),
                s.struct.field("volume").to_numpy()
            )
        ).alias(f"{p}obv"))
        
        # Accumulation/Distribution Line（軽量UDF） - 修正: rolling_map -> map_batches
        exprs.append(pl.struct(["high", "low", "close", "volume"]).map_batches(
            lambda s: accumulation_distribution_udf(
                s.struct.field("high").to_numpy(),
                s.struct.field("low").to_numpy(),
                s.struct.field("close").to_numpy(),
                s.struct.field("volume").to_numpy()
            )
        ).alias(f"{p}accumulation_distribution"))
        
        # Force Index（軽量UDF） - 修正: rolling_map -> map_batches
        exprs.append(pl.struct(["close", "volume"]).map_batches(
            lambda s: force_index_udf(
                s.struct.field("close").to_numpy(),
                s.struct.field("volume").to_numpy()
            )
        ).alias(f"{p}force_index"))
        
        # 出来高比率
        exprs.append(pl.col("volume").rolling_mean(20).alias(f"{p}volume_ma20"))
        exprs.append((pl.col("volume") / pl.col("volume").rolling_mean(20)).alias(f"{p}volume_ratio"))
        exprs.append((pl.col("close") * pl.col("volume")).rolling_mean(10).alias(f"{p}volume_price_trend"))
        
        return lazy_frame.with_columns(exprs)

    def _create_breakout_features(self, lazy_frame: pl.LazyFrame) -> pl.LazyFrame:
        """ブレイクアウト・レンジ系特徴量の計算（高速化対応）"""
        exprs = []
        p = self.prefix
        
        # ブレイクアウト・レンジ指標 - 全ての式に明示的なaliasを付与
        for window in self.config.window_sizes["general"]:
            # Donchian Channel（重量UDF）
            exprs.append(pl.struct(["high", "low"]).map_batches(
                lambda s: donchian_channel_udf(
                    s.struct.field("high").to_numpy(),
                    s.struct.field("low").to_numpy(),
                    window
                )[:, 0]
            ).alias(f"{p}donchian_upper_{window}"))
            
            exprs.append(pl.struct(["high", "low"]).map_batches(
                lambda s: donchian_channel_udf(
                    s.struct.field("high").to_numpy(),
                    s.struct.field("low").to_numpy(),
                    window
                )[:, 1]
            ).alias(f"{p}donchian_middle_{window}"))
            
            exprs.append(pl.struct(["high", "low"]).map_batches(
                lambda s: donchian_channel_udf(
                    s.struct.field("high").to_numpy(),
                    s.struct.field("low").to_numpy(),
                    window
                )[:, 2]
            ).alias(f"{p}donchian_lower_{window}"))
            
            # Price Channel（Polarsネイティブ）
            exprs.append(pl.col("high").rolling_max(window).alias(f"{p}price_channel_upper_{window}"))
            exprs.append(pl.col("low").rolling_min(window).alias(f"{p}price_channel_lower_{window}"))
        
        # Commodity Channel Index（重量UDF）
        for window in [14, 20]:
            exprs.append(pl.struct(["high", "low", "close"]).map_batches(
                lambda s: commodity_channel_index_udf(
                    s.struct.field("high").to_numpy(),
                    s.struct.field("low").to_numpy(),
                    s.struct.field("close").to_numpy(),
                    window
                )
            ).alias(f"{p}commodity_channel_index_{window}"))
        
        return lazy_frame.with_columns(exprs)

    def _create_support_resistance_features(self, lazy_frame: pl.LazyFrame) -> pl.LazyFrame:
        """サポート・レジスタンス系特徴量の計算（高速化対応）"""
        exprs = []
        p = self.prefix
        
        # サポート・レジスタンス・ローソク足 - 全ての式に明示的なaliasを付与
        # 日次ピボットポイント（前日の高安終値から計算） - 修正: Polarsネイティブ関数に置換
        prev_high = pl.col("high").shift(1)
        prev_low = pl.col("low").shift(1)
        prev_close = pl.col("close").shift(1)
        pivot = (prev_high + prev_low + prev_close) / 3.0

        exprs.append(pivot.alias(f"{p}pivot_point"))
        exprs.append((2.0 * pivot - prev_low).alias(f"{p}resistance1"))
        exprs.append((2.0 * pivot - prev_high).alias(f"{p}support1"))
        
        # フィボナッチレベル（重量UDF）
        exprs.append(pl.struct(["high", "low"]).map_batches(
            lambda s: fibonacci_levels_udf(
                s.struct.field("high").to_numpy(),
                s.struct.field("low").to_numpy(),
                50
            )[:, 2]  # 50%レベル
        ).alias(f"{p}fib_level_50"))
        
        # ローソク足パターン（重量UDF）
        exprs.append(pl.struct(["open", "high", "low", "close"]).map_batches(
            lambda s: candlestick_patterns_udf(
                s.struct.field("open").to_numpy(),
                s.struct.field("high").to_numpy(),
                s.struct.field("low").to_numpy(),
                s.struct.field("close").to_numpy()
            )
        ).alias(f"{p}candlestick_pattern"))
        
        return lazy_frame.with_columns(exprs)

    def _create_price_action_features(self, lazy_frame: pl.LazyFrame) -> pl.LazyFrame:
        """価格アクション系特徴量の計算（高速化対応）"""
        exprs = []
        p = self.prefix
        
        # 価格アクション指標 - 全ての式に明示的なaliasを付与
        exprs.append(((pl.col("high") + pl.col("low") + pl.col("close")) / 3.0).alias(f"{p}typical_price"))
        exprs.append(((pl.col("high") + pl.col("low") + 2 * pl.col("close")) / 4.0).alias(f"{p}weighted_close"))
        exprs.append(((pl.col("high") + pl.col("low")) / 2.0).alias(f"{p}median_price"))
        
        # ローソク足構成要素
        exprs.append((pl.col("close") - pl.col("open")).abs().alias(f"{p}body_size"))
        exprs.append(((pl.col("high") - pl.max_horizontal("open", "close")) / (pl.col("high") - pl.col("low"))).alias(f"{p}upper_wick_ratio"))
        exprs.append(((pl.min_horizontal("open", "close") - pl.col("low")) / (pl.col("high") - pl.col("low"))).alias(f"{p}lower_wick_ratio"))
        exprs.append(((pl.col("close") - pl.col("low")) / (pl.col("high") - pl.col("low"))).alias(f"{p}price_location_hl"))
        
        # イントラデイ・オーバーナイト
        exprs.append(((pl.col("close") - pl.col("open")) / pl.col("open")).alias(f"{p}intraday_return"))
        exprs.append(((pl.col("open") - pl.col("close").shift(1)) / pl.col("close").shift(1)).alias(f"{p}overnight_gap"))
        
        return lazy_frame.with_columns(exprs)

class OutputEngine:
    """
    出力管理クラス（10%） - Project Forge準拠
    責務:
    - LazyFrame.sink_parquet()によるストリーミング出力
    - 必要に応じたPyArrowフォールバック（use_pyarrow=True）
    - timeframe別ファイル分離
    - NaN埋め統一処理
    - シンプルな進捗表示
    - 基本メタデータ記録
    """
    
    def __init__(self, config: ProcessingConfig):
        self.config = config
        self.memory_monitor = MemoryMonitor(config.memory_limit_gb)
        
        # 出力設定（Project Forge基準）
        self.output_config = {
            "compression": "snappy",  # 固定値
            "dtype": "float64",  # 金融データの精度重視
            "timestamp_handling": "column"  # 機械学習での柔軟性重視
        }
        
        # エンジン識別子
        self.engine_id = config.engine_id
    
    def create_output_path(self, timeframe: str) -> Path:
        """出力パス生成 - Project Forge命名規則"""
        filename = f"features_{self.engine_id}_{timeframe}.parquet"
        return Path(self.config.output_path) / filename
    
    def apply_nan_filling(self, lazy_frame: pl.LazyFrame) -> pl.LazyFrame:
        """NaN埋め統一処理"""
        logger.info("NaN埋め処理開始")
        
        # スキーマを一度だけ取得（メモリ安全）
        schema = lazy_frame.collect_schema()
        all_columns = schema.names()
        
        # プレフィックスを持つ特徴量カラムを特定
        feature_columns = [col for col in all_columns if col.startswith(f"{self.engine_id}_")]
        
        # NaN埋め式生成
        fill_exprs = []
        for col in feature_columns:
            # NaNを0で埋める（金融データでは一般的）
            fill_exprs.append(
                pl.col(col).fill_null(0.0).alias(col)
            )
        
        # 基本カラムはそのまま保持
        basic_columns = ["timestamp", "open", "high", "low", "close", "volume", "timeframe"]
        basic_exprs = [pl.col(col) for col in basic_columns if col in all_columns]
        
        all_exprs = basic_exprs + fill_exprs
        result = lazy_frame.select(all_exprs)
        
        logger.info(f"NaN埋め処理完了: {len(feature_columns)}個の特徴量")
        return result
    
    def save_features(self, lazy_frame: pl.LazyFrame, timeframe: str) -> Dict[str, Any]:
        """特徴量ファイル保存"""
        output_path = self.create_output_path(timeframe)
        logger.info(f"特徴量保存開始: {output_path}")
        
        try:
            # 出力ディレクトリ作成
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            # メモリ安全性チェック
            is_safe, message = self.memory_monitor.check_memory_safety()
            if not is_safe:
                raise MemoryError(f"保存処理でメモリ不足: {message}")
            
            # NaN埋め処理適用
            processed_frame = self.apply_nan_filling(lazy_frame)
            
            start_time = time.time()
            
            # 【修正】sink_parquetの代わりにcollect + write_parquetを使用
            # struct型を使ったUDFの結果がsink_parquetと互換性がない問題を回避
            try:
                # まずストリーミングcollectを試行（新しいAPI使用）
                df = processed_frame.collect(engine="streaming")
            except Exception as streaming_error:
                logger.warning(f"ストリーミングcollectが失敗、通常collectを使用: {streaming_error}")
                df = processed_frame.collect()
            
            # DataFrameとして保存
            df.write_parquet(
                str(output_path),
                compression=self.output_config["compression"]
            )
            
            save_time = time.time() - start_time
            
            if not output_path.exists():
                raise FileNotFoundError(f"出力ファイルが作成されませんでした: {output_path}")
            
            file_size_mb = output_path.stat().st_size / (1024 * 1024)
            
            metadata = {
                "timeframe": timeframe,
                "output_path": str(output_path),
                "file_size_mb": round(file_size_mb, 2),
                "save_time_seconds": round(save_time, 2),
                "compression": self.output_config["compression"],
                "engine_id": self.engine_id,
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
            }
            
            logger.info(f"保存完了: {file_size_mb:.2f}MB, {save_time:.2f}秒")
            return metadata
            
        except Exception as e:
            logger.error(f"保存エラー (timeframe={timeframe}): {e}")
            raise
    
    def create_summary_report(self, processing_metadata: List[Dict[str, Any]]) -> Dict[str, Any]:
        """処理サマリーレポート生成"""
        total_files = len(processing_metadata)
        total_size_mb = sum(meta.get("file_size_mb", 0) for meta in processing_metadata)
        total_time = sum(meta.get("save_time_seconds", 0) for meta in processing_metadata)
        
        summary = {
            "engine_id": self.engine_id,
            "engine_name": "Engine_1D_VolumeActionFeatures",
            "total_files": total_files,
            "total_size_mb": round(total_size_mb, 2),
            "total_processing_time_seconds": round(total_time, 2),
            "average_file_size_mb": round(total_size_mb / total_files if total_files > 0 else 0, 2),
            "timeframes_processed": [meta.get("timeframe") for meta in processing_metadata],
            "compression_used": "snappy",
            "processing_timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "files_detail": processing_metadata
        }
        
        return summary

# ================================================================
# 大規模データ処理（Tick専用）アーキテクチャ：分割・重複・処理・結合
# ================================================================

def get_sorted_partitions(root_dir: Path) -> List[Path]:
    """
    指定されたルートディレクトリからHiveパーティションパスを収集し、
    時系列にソートして返す。
    """
    logging.info(f"パーティションを探索中: {root_dir}")
    partition_paths = sorted(
        list(root_dir.glob("year=*/month=*/day=*")),
        key=lambda p: (
            int(p.parent.parent.name.split('=')[1]),  # year
            int(p.parent.name.split('=')[1]),  # month
            int(p.name.split('=')[1])  # day
        )
    )
    logging.info(f"{len(partition_paths)}個のパーティションを発見しました。")
    return partition_paths

def create_augmented_frame(
    current_partition_path: Path,
    prev_partition_path: Path | None,
    w_max: int
) -> tuple[pl.DataFrame, int]:
    """
    現在のパーティションデータと、先行パーティションからのオーバーラップ部分を結合し、
    拡張されたデータフレームを生成する。
    """
    lf_current = pl.scan_parquet(current_partition_path / "*.parquet")
    # Tickデータ用にtimeframeカラムを追加
    lf_current = lf_current.with_columns([
        pl.lit("tick").alias("timeframe").cast(pl.Categorical)
    ])
    df_current = lf_current.collect()
    len_current_partition = df_current.height
    
    if prev_partition_path is None:
        return df_current, len_current_partition
    
    lookback_required = w_max - 1
    
    if lookback_required <= 0:
        return df_current, len_current_partition
    
    lf_prev = pl.scan_parquet(prev_partition_path / "*.parquet")
    # 前日データにもtimeframeカラムを追加
    lf_prev = lf_prev.with_columns([
        pl.lit("tick").alias("timeframe").cast(pl.Categorical)
    ])
    df_prefix = lf_prev.tail(lookback_required).collect()
    
    augmented_df = pl.concat([df_prefix, df_current], how="vertical")
    
    return augmented_df, len_current_partition

def run_on_partitions_mode(config: ProcessingConfig, resume_date: Optional[datetime.date] = None):
    """
    【修正版】実行モード: Tickデータ専用。パーティションを日単位で逐次処理する。
    責務の明確化: この関数が物理的垂直分割の工程を管理する
    """
    logging.info("【実行モード】日単位でのTickデータ特徴量計算を開始します...")
    
    timeframe = "tick"
    PARTITION_ROOT = Path(config.partitioned_tick_path)
    FEATURES_ROOT = Path(config.output_path) / f"features_{config.engine_id}_{timeframe}/"
    FEATURES_ROOT.mkdir(parents=True, exist_ok=True)
    
    W_MAX = config.w_max
    
    calculation_engine = CalculationEngine(config)
    
    all_partitions = get_sorted_partitions(PARTITION_ROOT)

    # ===== ここから再開ロジックの変更箇所 =====
    if resume_date:
        import datetime
        # 指定された再開日以降のパーティションのみを対象とする
        all_days = [
            p for p in all_partitions
            if datetime.date(
                int(p.parent.parent.name.split('=')[1]), # year
                int(p.parent.name.split('=')[1]),       # month
                int(p.name.split('=')[1])               # day
            ) >= resume_date
        ]
        logging.info(f"再開日 {resume_date} に基づいてフィルタリングしました。")
    else:
        all_days = all_partitions
    # ===== 再開ロジックここまで =====
    
    if not all_days:
        logging.error("処理対象の日次パーティションが見つかりません。")
        return
    
    # 日次パーティション逐次処理
    total_days = len(all_days)
    logging.info(f"処理対象日数: {total_days}日")
    
    for i, current_day_path in enumerate(all_days):
        day_name = f"{current_day_path.parent.parent.name}/{current_day_path.parent.name}/{current_day_path.name}"
        logging.info(f"=== 日次処理 ({i+1}/{total_days}): {day_name} ===")
        
        try:
            # 前日のパーティション（オーバーラップ用）
            # 【重要】prev_day_pathの参照元を all_partitions に変更し、日付が連続していることを保証する
            current_index_in_all = all_partitions.index(current_day_path)
            prev_day_path = all_partitions[current_index_in_all - 1] if current_index_in_all > 0 else None
            
            # オーバーラップを含む拡張データフレーム作成
            logging.info(f"データ読み込み開始: {day_name}")
            augmented_df, current_day_rows = create_augmented_frame(current_day_path, prev_day_path, W_MAX)
            logging.info(f"データ読み込み完了: 実データ{current_day_rows}行、総データ{augmented_df.height}行")
            
            # 【修正】親方（この関数）が工程管理を行う物理的垂直分割
            logging.info(f"特徴量計算開始: {day_name}")
            
            # 一時ディレクトリ作成（日次処理用）
            temp_dir = Path(tempfile.mkdtemp(prefix=f"day_{i:04d}_{config.engine_id}_"))
            logging.info(f"日次一時ディレクトリ作成: {temp_dir}")
            
            temp_files = []
            
            # 特徴量グループを取得
            feature_groups = calculation_engine.get_feature_groups()
            logging.info(f"物理的垂直分割: {len(feature_groups)}グループで処理")
            
            # 各グループを順次処理（親方が工程管理）
            for group_idx, (group_name, group_expressions) in enumerate(feature_groups.items()):
                logging.info(f"グループ処理開始: {group_name} ({len(group_expressions)}個の特徴量)")
                
                # 1. 下請けに「このグループだけ計算しろ」と指示
                group_result_lf = calculation_engine.calculate_one_group(
                    augmented_df.lazy(), group_name, group_expressions
                )
                
                # 2. 親方が自らメモリに実現化（単一グループなので安全）
                group_result_df = group_result_lf.collect(streaming=True) # メモリ効率のためstreaming=Trueを追加
                logging.info(f"グループデータ実現化: {group_result_df.height}行 x {group_result_df.width}列")
                
                # 3. 親方が自らディスクに保存する
                temp_file = temp_dir / f"group_{group_idx:02d}_{group_name}.parquet"
                group_result_df.write_parquet(str(temp_file), compression="snappy")
                
                if temp_file.exists():
                    temp_files.append(temp_file)
                    logging.info(f"グループ保存完了: {temp_file} ({temp_file.stat().st_size} bytes)")
                else:
                    raise FileNotFoundError(f"グループファイル作成失敗: {temp_file}")
                
                # メモリ使用量チェック
                memory_usage = calculation_engine.memory_monitor.get_memory_usage_gb()
                logging.info(f"メモリ使用量: {memory_usage:.2f}GB")
            
            # 4. 全グループ完了後、親方が最終組み立て（クリーン・オン・クリーン結合）
            logging.info("グループファイル結合開始（クリーン・オン・クリーン結合）...")
            
            # 1. 「クリーンな土台」を準備 (オーバーラップ除去済み)
            base_df = pl.read_parquet(str(temp_files[0]))
            
            if prev_day_path is not None:
                clean_base_df = base_df.tail(current_day_rows)
            else:
                clean_base_df = base_df
            
            logging.info(f"クリーンな土台を準備: {clean_base_df.height}行 x {clean_base_df.width}列")

            # 2. 残りの一時ファイルを「クリーンなパーツ」として一つずつ結合
            base_columns = ["timestamp", "open", "high", "low", "close", "volume", "timeframe"]
            
            for idx, temp_file in enumerate(temp_files[1:], 1):
                next_df = pl.read_parquet(str(temp_file))
                
                # 「クリーンなパーツ」を作成 (オーバーラップ除去済み)
                if prev_day_path is not None:
                    clean_next_df = next_df.tail(current_day_rows)
                else:
                    clean_next_df = next_df
                
                # 行数が一致することを確認
                if clean_base_df.height != clean_next_df.height:
                    raise ValueError(f"行数不一致: ベース{clean_base_df.height}行 vs 追加{clean_next_df.height}行")
                
                feature_cols = [col for col in clean_next_df.columns if col not in base_columns]
                if feature_cols:
                    clean_base_df = clean_base_df.hstack(clean_next_df.select(feature_cols))
            
            result_df = clean_base_df
            logging.info(f"全グループ結合完了: {result_df.height}行 x {result_df.width}列")
            
            # パーティション保存用の日付列を追加
            final_df = result_df.with_columns([
                pl.col("timestamp").dt.year().alias("year"),
                pl.col("timestamp").dt.month().alias("month"),
                pl.col("timestamp").dt.day().alias("day")
            ])
            
            # 当日の結果を保存
            logging.info(f"最終保存開始: {day_name}")
            final_df.write_parquet(
                FEATURES_ROOT,
                partition_by=['year', 'month', 'day']
            )
            
            logging.info(f"保存完了: {day_name} - {final_df.height}行の特徴量データ")
            
            # 一時ディレクトリクリーンアップ
            for temp_file in temp_files:
                if temp_file.exists():
                    temp_file.unlink()
            temp_dir.rmdir()
            logging.info(f"一時ディレクトリクリーンアップ完了: {temp_dir}")
            
        except Exception as e:
            logging.error(f"日次処理エラー ({day_name}): {e}", exc_info=True)
            continue

# 通常timeframe処理
def process_single_timeframe(config: ProcessingConfig, timeframe: str):
    """単一の通常時間足を処理する（修正版ロジック）"""
    logger.info(f"=== 通常処理開始: timeframe={timeframe} ===")
    start_time = time.time()
    
    # calc_engineをtryブロックの外で初期化
    calc_engine = None
    try:
        data_engine = DataEngine(config)
        calc_engine = CalculationEngine(config)
        output_engine = OutputEngine(config)
        
        lazy_frame = data_engine.create_lazy_frame(timeframe)
        summary = data_engine.get_data_summary(lazy_frame)
        logger.info(f"データサマリー: {summary}")

        # --- 修正箇所: グループ化された特徴量計算ロジックを適用 ---
        all_expressions = []
        feature_groups = calc_engine.get_feature_groups()
        for group_name, group_expressions in feature_groups.items():
            all_expressions.extend(group_expressions.values())

        logger.info(f"特徴量計算開始: {len(all_expressions)}個の特徴量を {timeframe} に対して計算します。")
        features_lf = lazy_frame.with_columns(all_expressions)

        # 品質保証システムを適用
        features_lf = calc_engine.apply_quality_assurance(features_lf)
        # --- 修正ここまで ---
        
        processed_lf = output_engine.apply_nan_filling(features_lf)
        metadata = output_engine.save_features(processed_lf, timeframe)
        
        elapsed_time = time.time() - start_time
        metadata["processing_time"] = elapsed_time
        
        logger.info(f"=== 通常処理完了: {timeframe} - {elapsed_time:.2f}秒 ===")
        return metadata
        
    except Exception as e:
        logger.error(f"タイムフレーム {timeframe} の処理中にエラー: {e}", exc_info=True)
        return {"timeframe": timeframe, "error": str(e)}
    finally:
        # CalculationEngineが一時ディレクトリを作成した場合に備えてクリーンアップ
        if calc_engine and hasattr(calc_engine, '_cleanup_temp_files'):
            calc_engine._cleanup_temp_files()

# インタラクティブモード（Project Forge準拠）
def get_user_confirmation(config: ProcessingConfig) -> bool:
    """ユーザー確認 - Project Forge理念表示"""
    print("\n" + "="*60)
    print(f"Engine {config.engine_id.upper()} - {config.engine_name}")
    print("="*60)
    print("🎯 Project Forge - 軍資金増大プロジェクト")
    print("🚀 最終目標: Project Chimera開発・完成のための資金調達")
    print("💻 探索対象: マーケットの亡霊（統計的に有意で非ランダムな微細パターン）")
    print("🏅 思想的継承: ジム・シモンズ（ルネサンス・テクノロジーズ）")
    print("="*60)
    print(f"入力パス: {config.input_path}")
    print(f"出力パス: {config.output_path}")
    print(f"Tickパーティションパス: {config.partitioned_tick_path}")
    print(f"エンジンID: {config.engine_id}")
    print(f"並列スレッド数: {config.max_threads}")
    print(f"メモリ制限: {config.memory_limit_gb}GB")
    
    if config.test_mode:
        print(f"\n【テストモード】最初の{config.test_rows}行のみ処理")
    
    print(f"\n処理対象タイムフレーム ({len(config.timeframes)}個):")
    for i, tf in enumerate(config.timeframes):
        print(f"  {i+1:2d}. {tf}")
    
    print("\n処理内容:")
    print("  - ボラティリティ指標（標準・ロバスト・Chaikin・Mass Index等）")
    print("  - 出来高関連指標（CMF・MFI・VWAP・OBV・A/D Line等）")
    print("  - ブレイクアウト・レンジ（Donchian・Price Channel・CCI等）")
    print("  - サポート・レジスタンス（ピボットポイント・フィボナッチ等）")
    print("  - 価格アクション（ローソク足パターン・実体比率等）")
    print("  - 2段階品質保証システム")
    print("  - 【修正】物理的垂直分割（ディスクベース中間ファイル）")
    print("  - Polars LazyFrame + Numba JITハイブリッド最適化")
    
    response = input("\n処理を開始しますか？ (y/n): ")
    return response.lower() == 'y'

def select_timeframes(config: ProcessingConfig) -> List[str]:
    """タイムフレーム選択（完全同一実装）"""
    print("\nタイムフレームを選択してください:")
    print("  0. 全て処理")
    
    all_timeframes = config.timeframes
    
    for i, tf in enumerate(all_timeframes):
        print(f"  {i+1:2d}. {tf}")
    
    print("  (例: 1,3,5 または 1-5 カンマ区切り)")
    
    selection = input("選択: ").strip()
    
    if selection == "0" or selection == "":
        return all_timeframes
    
    selected_indices = set()
    try:
        parts = selection.split(',')
        for part in parts:
            if '-' in part:
                start, end = map(int, part.strip().split('-'))
                selected_indices.update(range(start - 1, end))
            else:
                selected_indices.add(int(part.strip()) - 1)
        
        return [all_timeframes[i] for i in sorted(list(selected_indices)) if 0 <= i < len(all_timeframes)]
    except Exception as e:
        logger.warning(f"選択エラー: {e} - 全タイムフレームを処理します")
        return all_timeframes

def main():
    """メイン実行関数 - Project Forge統合版"""
    print("\n" + "="*70)
    print(f"  Engine 1D - Volume & Price Action Features (修正版) ")
    print("  Project Forge - 軍資金増大プロジェクト  ")
    print("="*70)
    print("🎯 目標: XAU/USD市場の統計的パターン抽出")
    print("🤖 AI頭脳による普遍的法則発見")
    print("💰 Project Chimera開発資金調達")
    print("🔧 【修正】物理的垂直分割によるメモリ・スラッシング回避")
    print("="*70)
    
    config = ProcessingConfig()
    
    if not config.validate():
        return 1
    
    data_engine = DataEngine(config)
    
    if not data_engine.validate_data_source():
        return 1
    
    # ===== ここから対話機能の変更箇所 =====
    import datetime
    resume_date = None
    print("\n実行タイプを選択してください:")
    print("  1. 新規実行")
    print("  2. 中断した処理を再開")
    run_type_selection = input("選択 (1/2): ").strip()

    if run_type_selection == '2':
        while True:
            date_str = input("再開する日付を入力してください (例: 2025-01-01): ").strip()
            try:
                # 文字列をdatetimeオブジェクトに変換し、date部分のみを取得
                resume_date = datetime.datetime.strptime(date_str, "%Y-%m-%d").date()
                print(f"{resume_date} から処理を再開します。")
                break
            except ValueError:
                print("エラー: 日付の形式が正しくありません。YYYY-MM-DD形式で入力してください。")
    else:
        print("新規に処理を開始します。")
    
    # Project Forge準拠の対話設定
    print("\n並列処理スレッド数を選択してください:")
    print("  1. 自動設定 (推奨)")
    print("  2. 手動設定")
    
    thread_selection = input("選択 (1/2): ").strip()
    
    if thread_selection == "2":
        try:
            max_threads = int(input(f"スレッド数を入力 (1-{psutil.cpu_count()}): ").strip())
            if 1 <= max_threads <= psutil.cpu_count():
                config.max_threads = max_threads
                print(f"スレッド数設定: {max_threads}")
            else:
                print("無効な値です。デフォルト値を使用します。")
        except ValueError:
            print("無効な入力です。デフォルト値を使用します。")
    
    print("\n出力パスを選択してください:")
    print(f"  1. デフォルト ({config.output_path})")
    print("  2. カスタムパス")
    
    path_selection = input("選択 (1/2): ").strip()
    
    if path_selection == "2":
        custom_path = input("出力パスを入力: ").strip()
        if custom_path:
            config.output_path = custom_path
            print(f"出力パス設定: {custom_path}")
    
    print("\nメモリ制限を選択してください:")
    print(f"  1. デフォルト ({config.memory_limit_gb}GB)")
    print("  2. カスタム設定")
    
    memory_selection = input("選択 (1/2): ").strip()
    
    if memory_selection == "2":
        try:
            memory_limit = float(input("メモリ制限 (GB): ").strip())
            if memory_limit > 0:
                config.memory_limit_gb = memory_limit
                config.memory_warning_gb = memory_limit * 0.9
                print(f"メモリ制限設定: {memory_limit}GB")
            else:
                print("無効な値です。デフォルト値を使用します。")
        except ValueError:
            print("無効な入力です。デフォルト値を使用します。")
    
    print("\n実行モードを選択してください:")
    print("  1. テストモード（少量データで動作確認）")
    print("  2. 本格モード（全データ処理）")
    
    mode_selection = input("選択 (1/2): ").strip()
    
    if mode_selection == "1":
        config.test_mode = True
        try:
            test_rows = int(input(f"テスト行数 (デフォルト: {config.test_rows}): ").strip() or str(config.test_rows))
            config.test_rows = test_rows
            print(f"テストモード設定: 最初の{config.test_rows}行を処理")
        except ValueError:
            print(f"無効な入力です。デフォルト値 ({config.test_rows}) を使用します。")
    
    selected_timeframes = select_timeframes(config)
    config.timeframes = selected_timeframes
    
    if not get_user_confirmation(config):
        print("処理をキャンセルしました")
        return 0
    
    os.environ["POLARS_MAX_THREADS"] = str(config.max_threads)
    logger.info(f"並列処理スレッド数: {config.max_threads}")
    
    print("\n" + "="*60)
    print("処理開始...")
    print("="*60)
    
    overall_start_time = time.time()
    
    if 'tick' in selected_timeframes:
        run_on_partitions_mode(config, resume_date=resume_date)
    
    other_timeframes = [tf for tf in selected_timeframes if tf != 'tick']
    if other_timeframes:
        for tf in other_timeframes:
            process_single_timeframe(config, tf)
    
    overall_elapsed_time = time.time() - overall_start_time
    
    print(f"\n全ての要求された処理が完了しました。総処理時間: {overall_elapsed_time:.2f}秒")
    return 0

if __name__ == "__main__":
    try:
        exit_code = main()
        sys.exit(exit_code)
    except Exception as e:
        logger.critical(f"スクリプト実行中に致命的なエラーが発生しました: {e}", exc_info=True)
        sys.exit(1)