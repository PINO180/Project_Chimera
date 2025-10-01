#!/usr/bin/env python3
"""
学際的・実験的特徴量収集スクリプト - Engine 1F: Interdisciplinary Features (最適化版)
【最適化内容】rolling_map完全排除 → map_batchesベクトル化処理

Project Forge - 軍資金増大プロジェクト
最終目標: Project Chimera開発・完成のための資金調達

技術戦略: ジム・シモンズの思想的継承
- 経済学・ファンダメンタルズ・古典的テクニカル指標の完全排除
- 統計的に有意で非ランダムな微細パターン「マーケットの亡霊」の探索
- AIの頭脳による普遍的法則の読み解き

アーキテクチャ: 3クラス構成（最適化版）+ ディスクベース垂直分割
- DataEngine（30%）: Polars LazyFrame基盤
- CalculationEngine（60%）: 学際的特徴量計算核心（物理的垂直分割実装）
- OutputEngine（10%）: ストリーミング出力
"""

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
        "network": [20, 30, 50, 100],
        "linguistic": [15, 25, 40, 80],
        "aesthetic": [21, 34, 55, 89],
        "musical": [12, 24, 48, 96],
        "biomechanical": [10, 20, 40, 60],
        "general": [10, 20, 50, 100]
    }

@dataclass
class ProcessingConfig:
    """処理設定 - Project Forge統合版"""
    
    # データパス（Project Forge構造準拠）
    input_path: str = "/workspaces/project_forge/data/1_XAUUSD_base_data/XAUUSD_tick_master_GOLDEN"
    partitioned_tick_path: str = "/workspaces/project_forge/data/0_tick_partitioned/"
    output_path: str = "/workspaces/project_forge/data/2_feature_value"
    
    # エンジン識別
    engine_id: str = "e1f"  # 例: "e1a", "e1b", "e2a"
    engine_name: str = "Engine_1F_InterdisciplinaryFeatures"
    
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
    # この値は、全特徴量計算の最大ウィンドウサイズを反映しなければならない
    # 例：移動平均線でsma_200などを使う場合は200以上に設定
    w_max: int = 200
    
    def validate(self) -> bool:
        """設定検証"""
        if not Path(self.input_path).exists():
            logger.error(f"入力パスが存在しません: {self.input_path}")
            return False
        
        if not Path(self.output_path).exists():
            Path(self.output_path).mkdir(parents=True, exist_ok=True)
            logger.info(f"出力ディレクトリを作成: {self.output_path}")
        
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
    - timeframe別分割処理（述語による高速化）
    - メモリ使用量監視（50GB警告、55GB緊急停止）
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
# Numba UDF定義（クラス外必須）- 最適化版
# ===============================

# 【重要】全てのNumba UDFは必ずクラス外で定義すること
# クラス内定義は循環参照エラーを引き起こすため絶対禁止

# =============================================================================
# ネットワーク科学: 価格ネットワークの密度とクラスタリング (並列化版)
# =============================================================================

@nb.njit(fastmath=True, cache=True, parallel=True)
def rolling_network_density_udf(prices: np.ndarray, window_size: int) -> np.ndarray:
    """
    【並列化版】ネットワーク密度計算 - 価格変動の相互関係度
    """
    n = len(prices)
    results = np.full(n, np.nan)
    
    for i in nb.prange(window_size - 1, n):
        window_prices = prices[i - window_size + 1 : i + 1]
        
        if len(window_prices) < 10:
            continue
        
        finite_prices = window_prices[np.isfinite(window_prices)]
        if len(finite_prices) < 10:
            continue
        
        window_n = len(finite_prices)
        threshold = np.std(finite_prices) * 0.5
        
        edge_count = 0
        max_possible_edges = window_n * (window_n - 1) / 2
        
        for j in range(window_n - 1):
            for k in range(j + 1, window_n):
                price_diff = abs(finite_prices[j] - finite_prices[k])
                if price_diff <= threshold:
                    edge_count += 1
        
        if max_possible_edges > 0:
            density = edge_count / max_possible_edges
        else:
            density = 0.0
        
        results[i] = density
    
    return results

@nb.njit(fastmath=True, cache=True, parallel=True)
def rolling_network_clustering_udf(prices: np.ndarray, window_size: int) -> np.ndarray:
    """
    【並列化版】ネットワーククラスタリング係数 - 局所的密集度
    """
    n = len(prices)
    results = np.full(n, np.nan)
    
    for i in nb.prange(window_size - 1, n):
        window_prices = prices[i - window_size + 1 : i + 1]
        
        if len(window_prices) < 15:
            continue
        
        finite_prices = window_prices[np.isfinite(window_prices)]
        if len(finite_prices) < 15:
            continue
        
        window_n = len(finite_prices)
        threshold = np.std(finite_prices) * 0.5
        
        adjacency = np.zeros((window_n, window_n), dtype=nb.boolean)
        
        for j in range(window_n):
            for k in range(window_n):
                if j != k:
                    price_diff = abs(finite_prices[j] - finite_prices[k])
                    if price_diff <= threshold:
                        adjacency[j, k] = True
        
        total_clustering = 0.0
        valid_nodes = 0
        
        for j in range(window_n):
            neighbors = []
            for k in range(window_n):
                if adjacency[j, k]:
                    neighbors.append(k)
            
            k = len(neighbors)
            if k < 2:
                continue
            
            neighbor_connections = 0
            for idx1 in range(len(neighbors)):
                for idx2 in range(idx1 + 1, len(neighbors)):
                    n1, n2 = neighbors[idx1], neighbors[idx2]
                    if adjacency[n1, n2]:
                        neighbor_connections += 1
            
            max_connections = k * (k - 1) / 2
            if max_connections > 0:
                clustering_j = neighbor_connections / max_connections
                total_clustering += clustering_j
                valid_nodes += 1
        
        if valid_nodes > 0:
            clustering_coeff = total_clustering / valid_nodes
        else:
            clustering_coeff = 0.0
        
        results[i] = clustering_coeff
    
    return results

# =============================================================================
# 言語学: 語彙多様性・言語的複雑性・意味的流れ (並列化版)
# =============================================================================

@nb.njit(fastmath=True, cache=True, parallel=True)
def rolling_vocabulary_diversity_udf(prices: np.ndarray, window_size: int) -> np.ndarray:
    """
    【並列化版】語彙多様性指標 - 価格「語彙」の豊富さ
    """
    n = len(prices)
    results = np.full(n, np.nan)
    
    for i in nb.prange(window_size - 1, n):
        window_prices = prices[i - window_size + 1 : i + 1]
        
        if len(window_prices) < 10:
            continue
        
        finite_prices = window_prices[np.isfinite(window_prices)]
        if len(finite_prices) < 10:
            continue
        
        std_price = np.std(finite_prices)
        mean_price = np.mean(finite_prices)
        
        if std_price == 0:
            results[i] = 0.0
            continue
        
        n_bins = 10
        price_min = mean_price - 2 * std_price
        price_max = mean_price + 2 * std_price
        bin_width = (price_max - price_min) / n_bins
        
        used_vocabularies = set()
        total_tokens = 0
        
        for price in finite_prices:
            if bin_width > 0:
                bin_idx = int((price - price_min) / bin_width)
                bin_idx = max(0, min(bin_idx, n_bins - 1))
            else:
                bin_idx = 0
            
            used_vocabularies.add(bin_idx)
            total_tokens += 1
        
        if total_tokens > 0:
            vocabulary_diversity = len(used_vocabularies) / total_tokens
        else:
            vocabulary_diversity = 0.0
        
        results[i] = vocabulary_diversity
    
    return results

@nb.njit(fastmath=True, cache=True, parallel=True)
def rolling_linguistic_complexity_udf(prices: np.ndarray, window_size: int) -> np.ndarray:
    """
    【並列化版】言語的複雑性 - 統語構造の複雑さ
    """
    n = len(prices)
    results = np.full(n, np.nan)
    
    for i in nb.prange(window_size - 1, n):
        window_prices = prices[i - window_size + 1 : i + 1]
        
        if len(window_prices) < 20:
            continue
        
        finite_prices = window_prices[np.isfinite(window_prices)]
        if len(finite_prices) < 20:
            continue
        
        price_changes = np.diff(finite_prices)
        threshold = np.std(price_changes) * 0.1
        
        syntax_sequence = []
        for change in price_changes:
            if change > threshold:
                syntax_sequence.append(1)
            elif change < -threshold:
                syntax_sequence.append(-1)
            else:
                syntax_sequence.append(0)
        
        if len(syntax_sequence) < 3:
            continue
        
        bigrams = set()
        for j in range(len(syntax_sequence) - 1):
            bigram = (syntax_sequence[j], syntax_sequence[j + 1])
            bigrams.add(bigram)
        
        trigrams = set()
        for j in range(len(syntax_sequence) - 2):
            trigram = (syntax_sequence[j], syntax_sequence[j + 1], syntax_sequence[j + 2])
            trigrams.add(trigram)
        
        max_bigrams = min(9, len(syntax_sequence) - 1)
        max_trigrams = min(27, len(syntax_sequence) - 2)
        
        if max_bigrams > 0 and max_trigrams > 0:
            bigram_complexity = len(bigrams) / max_bigrams
            trigram_complexity = len(trigrams) / max_trigrams
            complexity = (bigram_complexity + trigram_complexity) / 2
        else:
            complexity = 0.0
        
        results[i] = complexity
    
    return results

@nb.njit(fastmath=True, cache=True, parallel=True)
def rolling_semantic_flow_udf(prices: np.ndarray, window_size: int) -> np.ndarray:
    """
    【並列化版】意味的流れ - 価格の「意味」の連続性
    """
    n = len(prices)
    results = np.full(n, np.nan)
    
    for i in nb.prange(window_size - 1, n):
        window_prices = prices[i - window_size + 1 : i + 1]
        
        if len(window_prices) < 15:
            continue
        
        finite_prices = window_prices[np.isfinite(window_prices)]
        if len(finite_prices) < 15:
            continue
        
        window_n = len(finite_prices)
        window_size_local = min(5, window_n // 3)
        
        semantic_vectors = []
        
        for j in range(window_size_local, window_n - window_size_local):
            neighborhood = finite_prices[j - window_size_local:j + window_size_local + 1]
            center_price = finite_prices[j]
            relative_positions = neighborhood - center_price
            
            if len(relative_positions) > 1:
                mean_rel = np.mean(relative_positions)
                std_rel = np.std(relative_positions)
                semantic_vector = np.array([mean_rel, std_rel])
                semantic_vectors.append(semantic_vector)
        
        if len(semantic_vectors) < 2:
            continue
        
        flow_continuity = 0.0
        valid_pairs = 0
        
        for j in range(len(semantic_vectors) - 1):
            vec1 = semantic_vectors[j]
            vec2 = semantic_vectors[j + 1]
            
            norm1 = np.sqrt(np.sum(vec1**2))
            norm2 = np.sqrt(np.sum(vec2**2))
            
            if norm1 > 1e-10 and norm2 > 1e-10:
                cosine_sim = np.sum(vec1 * vec2) / (norm1 * norm2)
                flow_continuity += cosine_sim
                valid_pairs += 1
        
        if valid_pairs > 0:
            semantic_flow = flow_continuity / valid_pairs
            semantic_flow = (semantic_flow + 1) / 2
        else:
            semantic_flow = 0.0
        
        results[i] = semantic_flow
    
    return results

# =============================================================================
# 美学: 黄金比・対称性・美的バランス (並列化版)
# =============================================================================

@nb.njit(fastmath=True, cache=True, parallel=True)
def rolling_golden_ratio_adherence_udf(prices: np.ndarray, window_size: int) -> np.ndarray:
    """
    【並列化版】黄金比固着度 - 価格変動の黄金比との適合性
    """
    n = len(prices)
    results = np.full(n, np.nan)
    golden_ratio = (1 + np.sqrt(5)) / 2
    
    for i in nb.prange(window_size - 1, n):
        window_prices = prices[i - window_size + 1 : i + 1]
        
        if len(window_prices) < 10:
            continue
        
        finite_prices = window_prices[np.isfinite(window_prices)]
        if len(finite_prices) < 10:
            continue
        
        local_window = min(8, len(finite_prices) // 2)
        adherence_scores = []
        
        for j in range(local_window, len(finite_prices) - local_window):
            local_subwindow = finite_prices[j-local_window:j+local_window+1]
            local_high = np.max(local_subwindow)
            local_low = np.min(local_subwindow)
            
            if local_low > 0:
                ratio = local_high / local_low
                deviation = abs(ratio - golden_ratio) / golden_ratio
                adherence = 1.0 / (1.0 + deviation)
                adherence_scores.append(adherence)
        
        if adherence_scores:
            results[i] = np.mean(np.array(adherence_scores))
    
    return results

@nb.njit(fastmath=True, cache=True, parallel=True)
def rolling_symmetry_measure_udf(prices: np.ndarray, window_size: int) -> np.ndarray:
    """
    【並列化版】対称性測定 - 価格パターンの鏡像対称性
    """
    n = len(prices)
    results = np.full(n, np.nan)
    
    for i in nb.prange(window_size - 1, n):
        window_prices = prices[i - window_size + 1 : i + 1]
        
        if len(window_prices) < 20:
            continue
        
        finite_prices = window_prices[np.isfinite(window_prices)]
        if len(finite_prices) < 20:
            continue
        
        window_n = len(finite_prices)
        center = window_n // 2
        
        left_half = finite_prices[:center]
        right_half = finite_prices[center+1:] if window_n % 2 == 1 else finite_prices[center:]
        
        right_half_reversed = right_half[::-1]
        
        min_len = min(len(left_half), len(right_half_reversed))
        if min_len < 5:
            continue
        
        left_normalized = left_half[-min_len:]
        right_normalized = right_half_reversed[:min_len]
        
        def normalize_series(series):
            mean_val = np.mean(series)
            std_val = np.std(series)
            if std_val > 1e-10:
                return (series - mean_val) / std_val
            else:
                return series - mean_val
        
        left_norm = normalize_series(left_normalized)
        right_norm = normalize_series(right_normalized)
        
        if len(left_norm) < 2:
            continue
        
        mean_left = np.mean(left_norm)
        mean_right = np.mean(right_norm)
        
        numerator = np.sum((left_norm - mean_left) * (right_norm - mean_right))
        denom_left = np.sum((left_norm - mean_left)**2)
        denom_right = np.sum((right_norm - mean_right)**2)
        
        if denom_left > 1e-10 and denom_right > 1e-10:
            correlation = numerator / np.sqrt(denom_left * denom_right)
            symmetry = (correlation + 1) / 2
        else:
            symmetry = 0.0
        
        results[i] = symmetry
    
    return results

@nb.njit(fastmath=True, cache=True, parallel=True)
def rolling_aesthetic_balance_udf(prices: np.ndarray, window_size: int) -> np.ndarray:
    """
    【並列化版】美的バランス - 価格変動の視覚的調和
    """
    n = len(prices)
    results = np.full(n, np.nan)
    
    for i in nb.prange(window_size - 1, n):
        window_prices = prices[i - window_size + 1 : i + 1]
        
        if len(window_prices) < 15:
            continue
        
        finite_prices = window_prices[np.isfinite(window_prices)]
        if len(finite_prices) < 15:
            continue
        
        gradients = np.diff(finite_prices)
        
        if len(gradients) < 5:
            continue
        
        abs_gradients = np.abs(gradients)
        mean_grad = np.mean(abs_gradients)
        std_grad = np.std(abs_gradients)
        
        if std_grad <= 1e-10:
            results[i] = 1.0
            continue
        
        gentle_threshold = mean_grad - 0.5 * std_grad
        moderate_threshold = mean_grad + 0.5 * std_grad
        intense_threshold = mean_grad + 1.5 * std_grad
        
        gentle_count = 0
        moderate_count = 0
        intense_count = 0
        
        for grad in abs_gradients:
            if grad <= gentle_threshold:
                gentle_count += 1
            elif grad <= moderate_threshold:
                moderate_count += 1
            elif grad <= intense_threshold:
                intense_count += 1
        
        total_counted = gentle_count + moderate_count + intense_count
        if total_counted == 0:
            results[i] = 0.0
            continue
        
        ideal_gentle = 0.6
        ideal_moderate = 0.3
        ideal_intense = 0.1
        
        actual_gentle = gentle_count / total_counted
        actual_moderate = moderate_count / total_counted
        actual_intense = intense_count / total_counted
        
        balance_deviation = (
            abs(actual_gentle - ideal_gentle) +
            abs(actual_moderate - ideal_moderate) +
            abs(actual_intense - ideal_intense)
        ) / 2
        
        aesthetic_balance = 1.0 - balance_deviation
        results[i] = max(0.0, aesthetic_balance)
    
    return results

# =============================================================================
# 音楽理論: 調性・リズムパターン・和声・音楽的緊張度 (並列化版)
# =============================================================================

@nb.njit(fastmath=True, cache=True, parallel=True)
def rolling_tonality_udf(prices: np.ndarray, window_size: int) -> np.ndarray:
    """
    【並列化版】調性 - 価格の「調」的特性
    """
    n = len(prices)
    results = np.full(n, np.nan)
    
    for i in nb.prange(window_size - 1, n):
        window_prices = prices[i - window_size + 1 : i + 1]
        
        if len(window_prices) < 12:
            continue
        
        finite_prices = window_prices[np.isfinite(window_prices)]
        if len(finite_prices) < 12:
            continue
        
        price_changes = np.diff(finite_prices)
        
        if len(price_changes) < 5:
            continue
        
        std_change = np.std(price_changes)
        if std_change <= 1e-10:
            results[i] = 0.5
            continue
        
        normalized_changes = price_changes / std_change
        
        scale_degrees = np.zeros(12)
        
        for change in normalized_changes:
            degree_idx = int((change + 3) / 6 * 11)
            degree_idx = max(0, min(degree_idx, 11))
            scale_degrees[degree_idx] += 1
        
        major_pattern = np.array([1, 0, 1, 0, 1, 1, 0, 1, 0, 1, 0, 1])
        minor_pattern = np.array([1, 0, 1, 1, 0, 1, 0, 1, 1, 0, 1, 0])
        
        if np.sum(scale_degrees) > 0:
            scale_distribution = scale_degrees / np.sum(scale_degrees)
            
            major_similarity = np.sum(scale_distribution * major_pattern)
            minor_similarity = np.sum(scale_distribution * minor_pattern)
            
            total_similarity = major_similarity + minor_similarity
            if total_similarity > 0:
                tonality_score = major_similarity / total_similarity
            else:
                tonality_score = 0.5
        else:
            tonality_score = 0.5
        
        results[i] = tonality_score
    
    return results

@nb.njit(fastmath=True, cache=True, parallel=True)
def rolling_rhythm_pattern_udf(prices: np.ndarray, window_size: int) -> np.ndarray:
    """
    【並列化版】リズムパターン - 価格変動のリズム的規則性
    """
    n = len(prices)
    results = np.full(n, np.nan)
    
    for i in nb.prange(window_size - 1, n):
        window_prices = prices[i - window_size + 1 : i + 1]
        
        if len(window_prices) < 20:
            continue
        
        finite_prices = window_prices[np.isfinite(window_prices)]
        if len(finite_prices) < 20:
            continue
        
        price_changes = np.diff(finite_prices)
        abs_changes = np.abs(price_changes)
        
        mean_change = np.mean(abs_changes)
        strong_beats = abs_changes > mean_change
        
        pattern_strengths = []
        
        for period in range(2, min(8, len(strong_beats) // 3)):
            pattern_score = 0.0
            pattern_count = 0
            
            for j in range(period, len(strong_beats)):
                if strong_beats[j] == strong_beats[j - period]:
                    pattern_score += 1.0
                pattern_count += 1
            
            if pattern_count > 0:
                pattern_strength = pattern_score / pattern_count
                pattern_strengths.append(pattern_strength)
        
        if pattern_strengths:
            rhythm_strength = np.max(np.array(pattern_strengths))
        else:
            rhythm_strength = 0.0
        
        results[i] = rhythm_strength
    
    return results

@nb.njit(fastmath=True, cache=True, parallel=True)
def rolling_harmony_udf(prices: np.ndarray, window_size: int) -> np.ndarray:
    """
    【並列化版】和声 - 複数価格レベルの協調性
    """
    n = len(prices)
    results = np.full(n, np.nan)
    
    for i in nb.prange(window_size - 1, n):
        window_prices = prices[i - window_size + 1 : i + 1]
        
        if len(window_prices) < 30:
            continue
        
        finite_prices = window_prices[np.isfinite(window_prices)]
        if len(finite_prices) < 30:
            continue
        
        window_n = len(finite_prices)
        
        short_window = max(3, window_n // 15)
        medium_window = max(5, window_n // 10)
        long_window = max(8, window_n // 6)
        
        if long_window >= window_n:
            continue
        
        short_ma = np.zeros(window_n - short_window + 1)
        medium_ma = np.zeros(window_n - medium_window + 1)
        long_ma = np.zeros(window_n - long_window + 1)
        
        for j in range(len(short_ma)):
            short_ma[j] = np.mean(finite_prices[j:j + short_window])
        
        for j in range(len(medium_ma)):
            medium_ma[j] = np.mean(finite_prices[j:j + medium_window])
        
        for j in range(len(long_ma)):
            long_ma[j] = np.mean(finite_prices[j:j + long_window])
        
        min_len = min(len(short_ma), len(medium_ma), len(long_ma))
        if min_len < 5:
            continue
        
        short_trend = np.diff(short_ma[-min_len:])
        medium_trend = np.diff(medium_ma[-min_len:])
        long_trend = np.diff(long_ma[-min_len:])
        
        harmony_scores = []
        
        for j in range(len(short_trend)):
            signs = [np.sign(short_trend[j]), np.sign(medium_trend[j]), np.sign(long_trend[j])]
            
            if len(set(signs)) == 1 and signs[0] != 0:
                harmony_scores.append(1.0)
            elif len(set(signs)) == 2:
                harmony_scores.append(0.5)
            else:
                harmony_scores.append(0.0)
        
        if harmony_scores:
            results[i] = np.mean(np.array(harmony_scores))
        else:
            results[i] = 0.0
    
    return results

@nb.njit(fastmath=True, cache=True, parallel=True)
def rolling_musical_tension_udf(prices: np.ndarray, window_size: int) -> np.ndarray:
    """
    【並列化版】音楽的緊張度 - 価格変動の「緊張と緩和」
    """
    n = len(prices)
    results = np.full(n, np.nan)
    
    for i in nb.prange(window_size - 1, n):
        window_prices = prices[i - window_size + 1 : i + 1]
        
        if len(window_prices) < 15:
            continue
        
        finite_prices = window_prices[np.isfinite(window_prices)]
        if len(finite_prices) < 15:
            continue
        
        price_changes = np.diff(finite_prices)
        
        if len(price_changes) < 5:
            continue
        
        local_window = min(5, len(price_changes) // 3)
        tension_scores = []
        
        for j in range(local_window, len(price_changes) - local_window):
            local_changes = price_changes[j - local_window:j + local_window + 1]
            
            signs = np.sign(local_changes)
            sign_changes = np.sum(np.diff(signs) != 0)
            direction_dissonance = sign_changes / len(signs) if len(signs) > 0 else 0
            
            volatility = np.std(local_changes)
            max_volatility = np.max(np.abs(local_changes))
            intensity_dissonance = max_volatility / (np.mean(np.abs(finite_prices)) + 1e-10)
            
            total_tension = (direction_dissonance + intensity_dissonance) / 2
            tension_scores.append(min(total_tension, 1.0))
        
        if tension_scores:
            results[i] = np.mean(np.array(tension_scores))
        else:
            results[i] = 0.0
    
    return results

# =============================================================================
# 生体力学: 運動エネルギー・筋力・生体力学効率・エネルギー消費量 (並列化版)
# =============================================================================

@nb.njit(fastmath=True, cache=True, parallel=True)
def rolling_kinetic_energy_udf(prices: np.ndarray, window_size: int) -> np.ndarray:
    """
    【並列化版】運動エネルギー - 価格「粒子」の運動エネルギー
    """
    n = len(prices)
    results = np.full(n, np.nan)
    
    for i in nb.prange(window_size - 1, n):
        window_prices = prices[i - window_size + 1 : i + 1]
        
        if len(window_prices) < 10:
            continue
        
        finite_prices = window_prices[np.isfinite(window_prices)]
        if len(finite_prices) < 10:
            continue
        
        velocities = np.diff(finite_prices)
        
        if len(velocities) < 2:
            continue
        
        masses = finite_prices[1:]
        
        kinetic_energies = 0.5 * masses * velocities**2
        
        if len(kinetic_energies) > 0:
            mean_kinetic_energy = np.mean(kinetic_energies)
            mean_price = np.mean(finite_prices)
            if mean_price > 1e-10:
                normalized_energy = mean_kinetic_energy / (mean_price**2)
            else:
                normalized_energy = mean_kinetic_energy
        else:
            normalized_energy = 0.0
        
        results[i] = normalized_energy
    
    return results

@nb.njit(fastmath=True, cache=True, parallel=True)
def rolling_muscle_force_udf(prices: np.ndarray, window_size: int) -> np.ndarray:
    """
    【並列化版】筋力 - 価格変動を起こす「筋力」
    """
    n = len(prices)
    results = np.full(n, np.nan)
    
    for i in nb.prange(window_size - 1, n):
        window_prices = prices[i - window_size + 1 : i + 1]
        
        if len(window_prices) < 15:
            continue
        
        finite_prices = window_prices[np.isfinite(window_prices)]
        if len(finite_prices) < 15:
            continue
        
        velocities = np.diff(finite_prices)
        
        if len(velocities) < 2:
            continue
        
        accelerations = np.diff(velocities)
        
        masses = finite_prices[2:]
        forces = masses * np.abs(accelerations)
        
        force_directions = np.sign(accelerations)
        sustained_forces = []
        
        current_direction = force_directions[0] if len(force_directions) > 0 else 0
        current_duration = 1
        current_force_sum = forces[0] if len(forces) > 0 else 0
        
        for j in range(1, len(force_directions)):
            if force_directions[j] == current_direction and current_direction != 0:
                current_duration += 1
                current_force_sum += forces[j]
            else:
                if current_duration > 1:
                    avg_sustained_force = current_force_sum / current_duration
                    sustained_forces.append(avg_sustained_force)
                
                current_direction = force_directions[j]
                current_duration = 1
                current_force_sum = forces[j]
        
        if current_duration > 1:
            avg_sustained_force = current_force_sum / current_duration
            sustained_forces.append(avg_sustained_force)
        
        instantaneous_force = np.mean(forces) if len(forces) > 0 else 0.0
        sustained_force = np.mean(np.array(sustained_forces)) if sustained_forces else 0.0
        
        muscle_force_score = 0.7 * instantaneous_force + 0.3 * sustained_force
        
        mean_price = np.mean(finite_prices)
        if mean_price > 1e-10:
            normalized_muscle_force = muscle_force_score / (mean_price**2)
        else:
            normalized_muscle_force = muscle_force_score
        
        results[i] = normalized_muscle_force
    
    return results

@nb.njit(fastmath=True, cache=True, parallel=True)
def rolling_biomechanical_efficiency_udf(prices: np.ndarray, window_size: int) -> np.ndarray:
    """
    【並列化版】生体力学効率 - エネルギー効率の良い価格変動
    """
    n = len(prices)
    results = np.full(n, np.nan)
    
    for i in nb.prange(window_size - 1, n):
        window_prices = prices[i - window_size + 1 : i + 1]
        
        if len(window_prices) < 20:
            continue
        
        finite_prices = window_prices[np.isfinite(window_prices)]
        if len(finite_prices) < 20:
            continue
        
        price_changes = np.diff(finite_prices)
        total_displacement = np.sum(np.abs(price_changes))
        
        velocities = price_changes
        accelerations = np.diff(velocities) if len(velocities) > 1 else np.array([0.0])
        
        kinetic_energy = np.sum(velocities**2)
        acceleration_energy = np.sum(accelerations**2) if len(accelerations) > 0 else 0
        total_energy = kinetic_energy + acceleration_energy
        
        if total_energy > 1e-10 and total_displacement > 1e-10:
            raw_efficiency = total_displacement / total_energy
            
            reference_efficiency = total_displacement / (np.sum(np.abs(velocities)) + 1e-10)
            normalized_efficiency = raw_efficiency / (reference_efficiency + 1e-10)
            efficiency = min(normalized_efficiency, 1.0)
        else:
            efficiency = 0.0
        
        results[i] = efficiency
    
    return results

@nb.njit(fastmath=True, cache=True, parallel=True)
def rolling_energy_expenditure_udf(prices: np.ndarray, window_size: int) -> np.ndarray:
    """
    【並列化版】エネルギー消費量 - 価格変動に要するエネルギー総量
    """
    n = len(prices)
    results = np.full(n, np.nan)
    
    for i in nb.prange(window_size - 1, n):
        window_prices = prices[i - window_size + 1 : i + 1]
        
        if len(window_prices) < 15:
            continue
        
        finite_prices = window_prices[np.isfinite(window_prices)]
        if len(finite_prices) < 15:
            continue
        
        baseline_energy = np.var(finite_prices)
        
        price_changes = np.diff(finite_prices)
        movement_energy = np.sum(price_changes**2)
        
        if len(price_changes) > 1:
            accelerations = np.diff(price_changes)
            acceleration_energy = np.sum(accelerations**2)
        else:
            acceleration_energy = 0.0
        
        if len(finite_prices) >= 3:
            x = np.arange(len(finite_prices), dtype=np.float64)
            n_points = len(x)
            
            sum_x = np.sum(x)
            sum_y = np.sum(finite_prices)
            sum_xy = np.sum(x * finite_prices)
            sum_x2 = np.sum(x * x)
            
            if n_points * sum_x2 - sum_x**2 != 0:
                slope = (n_points * sum_xy - sum_x * sum_y) / (n_points * sum_x2 - sum_x**2)
                intercept = (sum_y - slope * sum_x) / n_points
                
                linear_trend = intercept + slope * x
                
                nonlinearity_energy = np.sum((finite_prices - linear_trend)**2)
            else:
                nonlinearity_energy = 0
        else:
            nonlinearity_energy = 0
        
        total_energy = (
            baseline_energy +
            movement_energy +
            acceleration_energy +
            nonlinearity_energy
        )
        
        mean_price = np.mean(finite_prices)
        if mean_price > 1e-10:
            normalized_energy = total_energy / (mean_price**2)
        else:
            normalized_energy = total_energy
        
        results[i] = normalized_energy
    
    return results

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
        self.prefix = f"e{config.engine_id.replace('e', '')}_"  # 例: "e1f_"
        
        # 【修正】ディスクベース垂直分割用の一時ディレクトリ
        self.temp_dir = Path(tempfile.mkdtemp(prefix=f"features_{config.engine_id}_"))
        logger.info(f"一時ディレクトリ作成: {self.temp_dir}")
    
    def _get_all_feature_expressions(self) -> Dict[str, pl.Expr]:
        """
        全学際的・実験的特徴量の名前とPolars Expressionの対応辞書を返すファクトリメソッド。
        これにより、必要な特徴量だけを効率的に計算できるようになる。
        【特徴量ファクトリパターン】
        """
        expressions = {}
        p = self.prefix
        
        # ネットワーク科学系 - map_batchesを使用（仕様書準拠）【クロージャー問題修正版】
        for window in self.config.window_sizes["network"]:
            expressions[f"{p}network_density_{window}"] = pl.col("close").map_batches(
                lambda s, w=window: pl.Series(rolling_network_density_udf(s.to_numpy(), window_size=w))
            ).alias(f"{p}network_density_{window}")
            
            expressions[f"{p}network_clustering_{window}"] = pl.col("close").map_batches(
                lambda s, w=window: pl.Series(rolling_network_clustering_udf(s.to_numpy(), window_size=w))
            ).alias(f"{p}network_clustering_{window}")
        
        # 言語学系 - map_batchesを使用（仕様書準拠）【クロージャー問題修正版】
        for window in self.config.window_sizes["linguistic"]:
            expressions[f"{p}vocabulary_diversity_{window}"] = pl.col("close").map_batches(
                lambda s, w=window: pl.Series(rolling_vocabulary_diversity_udf(s.to_numpy(), window_size=w))
            ).alias(f"{p}vocabulary_diversity_{window}")
            
            expressions[f"{p}linguistic_complexity_{window}"] = pl.col("close").map_batches(
                lambda s, w=window: pl.Series(rolling_linguistic_complexity_udf(s.to_numpy(), window_size=w))
            ).alias(f"{p}linguistic_complexity_{window}")
            
            expressions[f"{p}semantic_flow_{window}"] = pl.col("close").map_batches(
                lambda s, w=window: pl.Series(rolling_semantic_flow_udf(s.to_numpy(), window_size=w))
            ).alias(f"{p}semantic_flow_{window}")
        
        # 美学系 - map_batchesを使用（仕様書準拠）【クロージャー問題修正版】
        for window in self.config.window_sizes["aesthetic"]:
            expressions[f"{p}golden_ratio_adherence_{window}"] = pl.col("close").map_batches(
                lambda s, w=window: pl.Series(rolling_golden_ratio_adherence_udf(s.to_numpy(), window_size=w))
            ).alias(f"{p}golden_ratio_adherence_{window}")
            
            expressions[f"{p}symmetry_measure_{window}"] = pl.col("close").map_batches(
                lambda s, w=window: pl.Series(rolling_symmetry_measure_udf(s.to_numpy(), window_size=w))
            ).alias(f"{p}symmetry_measure_{window}")
            
            expressions[f"{p}aesthetic_balance_{window}"] = pl.col("close").map_batches(
                lambda s, w=window: pl.Series(rolling_aesthetic_balance_udf(s.to_numpy(), window_size=w))
            ).alias(f"{p}aesthetic_balance_{window}")
        
        # 音楽理論系 - map_batchesを使用（仕様書準拠）【クロージャー問題修正版】
        for window in self.config.window_sizes["musical"]:
            expressions[f"{p}tonality_{window}"] = pl.col("close").map_batches(
                lambda s, w=window: pl.Series(rolling_tonality_udf(s.to_numpy(), window_size=w))
            ).alias(f"{p}tonality_{window}")
            
            expressions[f"{p}rhythm_pattern_{window}"] = pl.col("close").map_batches(
                lambda s, w=window: pl.Series(rolling_rhythm_pattern_udf(s.to_numpy(), window_size=w))
            ).alias(f"{p}rhythm_pattern_{window}")
            
            expressions[f"{p}harmony_{window}"] = pl.col("close").map_batches(
                lambda s, w=window: pl.Series(rolling_harmony_udf(s.to_numpy(), window_size=w))
            ).alias(f"{p}harmony_{window}")
            
            expressions[f"{p}musical_tension_{window}"] = pl.col("close").map_batches(
                lambda s, w=window: pl.Series(rolling_musical_tension_udf(s.to_numpy(), window_size=w))
            ).alias(f"{p}musical_tension_{window}")
        
        # 生体力学系 - map_batchesを使用（仕様書準拠）【クロージャー問題修正版】
        for window in self.config.window_sizes["biomechanical"]:
            expressions[f"{p}kinetic_energy_{window}"] = pl.col("close").map_batches(
                lambda s, w=window: pl.Series(rolling_kinetic_energy_udf(s.to_numpy(), window_size=w))
            ).alias(f"{p}kinetic_energy_{window}")
            
            expressions[f"{p}muscle_force_{window}"] = pl.col("close").map_batches(
                lambda s, w=window: pl.Series(rolling_muscle_force_udf(s.to_numpy(), window_size=w))
            ).alias(f"{p}muscle_force_{window}")
            
            expressions[f"{p}biomechanical_efficiency_{window}"] = pl.col("close").map_batches(
                lambda s, w=window: pl.Series(rolling_biomechanical_efficiency_udf(s.to_numpy(), window_size=w))
            ).alias(f"{p}biomechanical_efficiency_{window}")
            
            expressions[f"{p}energy_expenditure_{window}"] = pl.col("close").map_batches(
                lambda s, w=window: pl.Series(rolling_energy_expenditure_udf(s.to_numpy(), window_size=w))
            ).alias(f"{p}energy_expenditure_{window}")
        
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
            if group_name == "network_science":
                group_result_lf = self._create_network_science_features(lazy_frame)
            elif group_name == "linguistics":
                group_result_lf = self._create_linguistics_features(lazy_frame)
            elif group_name == "aesthetics":
                group_result_lf = self._create_aesthetics_features(lazy_frame)
            elif group_name == "music_theory":
                group_result_lf = self._create_music_theory_features(lazy_frame)
            elif group_name == "biomechanics":
                group_result_lf = self._create_biomechanics_features(lazy_frame)
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
            # 頑強なパーセンタイル計算のため、一旦Inf値をnullに変換
            col_expr = pl.col(col_name).map_batches(lambda s: s.replace([np.inf, -np.inf], None))
            
            p01 = col_expr.quantile(0.01, interpolation="linear")
            p99 = col_expr.quantile(0.99, interpolation="linear")
            
            stabilized_col = (
                pl.col(col_name)
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
        
        # グループ1: ネットワーク科学系（中程度）
        slices["network_science"] = {
            name: expr for name, expr in all_expressions.items() 
            if any(pattern in name for pattern in ["network_density", "network_clustering"])
        }
        
        # グループ2: 言語学系（重い）
        slices["linguistics"] = {
            name: expr for name, expr in all_expressions.items() 
            if any(pattern in name for pattern in ["vocabulary_diversity", "linguistic_complexity", "semantic_flow"])
        }
        
        # グループ3: 美学系（中程度）
        slices["aesthetics"] = {
            name: expr for name, expr in all_expressions.items() 
            if any(pattern in name for pattern in ["golden_ratio", "symmetry_measure", "aesthetic_balance"])
        }
        
        # グループ4: 音楽理論系（重い）
        slices["music_theory"] = {
            name: expr for name, expr in all_expressions.items() 
            if any(pattern in name for pattern in ["tonality", "rhythm_pattern", "harmony", "musical_tension"])
        }
        
        # グループ5: 生体力学系（重い）
        slices["biomechanics"] = {
            name: expr for name, expr in all_expressions.items() 
            if any(pattern in name for pattern in ["kinetic_energy", "muscle_force", "biomechanical_efficiency", "energy_expenditure"])
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
            # 頑強なパーセンタイル計算のため、一旦Inf値をnullに変換
            col_expr = pl.col(col_name).map_batches(lambda s: s.replace([np.inf, -np.inf], None))
            
            p01 = col_expr.quantile(0.01, interpolation="linear")
            p99 = col_expr.quantile(0.99, interpolation="linear")
            
            stabilized_col = (
                pl.col(col_name)
                .clip(lower_bound=p01, upper_bound=p99)
                .fill_null(0.0)
                .fill_nan(0.0)
                .alias(col_name)
            )
            
            stabilization_exprs.append(stabilized_col)
        
        result = lazy_frame.with_columns(stabilization_exprs)
        logger.info("品質保証システム適用完了")
        
        return result
    
    def _create_network_science_features(self, lazy_frame: pl.LazyFrame) -> pl.LazyFrame:
        """ネットワーク科学系特徴量の計算（高速化対応）【クロージャー問題修正版】"""
        exprs = []
        p = self.prefix
        
        for window in self.config.window_sizes["network"]:
            exprs.append(pl.col("close").map_batches(
                lambda s, w=window: pl.Series(rolling_network_density_udf(s.to_numpy(), window_size=w))
            ).alias(f"{p}network_density_{window}"))
            
            exprs.append(pl.col("close").map_batches(
                lambda s, w=window: pl.Series(rolling_network_clustering_udf(s.to_numpy(), window_size=w))
            ).alias(f"{p}network_clustering_{window}"))
        
        return lazy_frame.with_columns(exprs)

    def _create_linguistics_features(self, lazy_frame: pl.LazyFrame) -> pl.LazyFrame:
        """言語学系特徴量の計算（高速化対応）【クロージャー問題修正版】"""
        exprs = []
        p = self.prefix
        
        for window in self.config.window_sizes["linguistic"]:
            exprs.append(pl.col("close").map_batches(
                lambda s, w=window: pl.Series(rolling_vocabulary_diversity_udf(s.to_numpy(), window_size=w))
            ).alias(f"{p}vocabulary_diversity_{window}"))
            
            exprs.append(pl.col("close").map_batches(
                lambda s, w=window: pl.Series(rolling_linguistic_complexity_udf(s.to_numpy(), window_size=w))
            ).alias(f"{p}linguistic_complexity_{window}"))
            
            exprs.append(pl.col("close").map_batches(
                lambda s, w=window: pl.Series(rolling_semantic_flow_udf(s.to_numpy(), window_size=w))
            ).alias(f"{p}semantic_flow_{window}"))
        
        return lazy_frame.with_columns(exprs)

    def _create_aesthetics_features(self, lazy_frame: pl.LazyFrame) -> pl.LazyFrame:
        """美学系特徴量の計算（高速化対応）【クロージャー問題修正版】"""
        exprs = []
        p = self.prefix
        
        for window in self.config.window_sizes["aesthetic"]:
            exprs.append(pl.col("close").map_batches(
                lambda s, w=window: pl.Series(rolling_golden_ratio_adherence_udf(s.to_numpy(), window_size=w))
            ).alias(f"{p}golden_ratio_adherence_{window}"))
            
            exprs.append(pl.col("close").map_batches(
                lambda s, w=window: pl.Series(rolling_symmetry_measure_udf(s.to_numpy(), window_size=w))
            ).alias(f"{p}symmetry_measure_{window}"))
            
            exprs.append(pl.col("close").map_batches(
                lambda s, w=window: pl.Series(rolling_aesthetic_balance_udf(s.to_numpy(), window_size=w))
            ).alias(f"{p}aesthetic_balance_{window}"))
        
        return lazy_frame.with_columns(exprs)

    def _create_music_theory_features(self, lazy_frame: pl.LazyFrame) -> pl.LazyFrame:
        """音楽理論系特徴量の計算（高速化対応）【クロージャー問題修正版】"""
        exprs = []
        p = self.prefix
        
        for window in self.config.window_sizes["musical"]:
            exprs.append(pl.col("close").map_batches(
                lambda s, w=window: pl.Series(rolling_tonality_udf(s.to_numpy(), window_size=w))
            ).alias(f"{p}tonality_{window}"))
            
            exprs.append(pl.col("close").map_batches(
                lambda s, w=window: pl.Series(rolling_rhythm_pattern_udf(s.to_numpy(), window_size=w))
            ).alias(f"{p}rhythm_pattern_{window}"))
            
            exprs.append(pl.col("close").map_batches(
                lambda s, w=window: pl.Series(rolling_harmony_udf(s.to_numpy(), window_size=w))
            ).alias(f"{p}harmony_{window}"))
            
            exprs.append(pl.col("close").map_batches(
                lambda s, w=window: pl.Series(rolling_musical_tension_udf(s.to_numpy(), window_size=w))
            ).alias(f"{p}musical_tension_{window}"))
        
        return lazy_frame.with_columns(exprs)

    def _create_biomechanics_features(self, lazy_frame: pl.LazyFrame) -> pl.LazyFrame:
        """生体力学系特徴量の計算（高速化対応）【クロージャー問題修正版】"""
        exprs = []
        p = self.prefix
        
        for window in self.config.window_sizes["biomechanical"]:
            exprs.append(pl.col("close").map_batches(
                lambda s, w=window: pl.Series(rolling_kinetic_energy_udf(s.to_numpy(), window_size=w))
            ).alias(f"{p}kinetic_energy_{window}"))
            
            exprs.append(pl.col("close").map_batches(
                lambda s, w=window: pl.Series(rolling_muscle_force_udf(s.to_numpy(), window_size=w))
            ).alias(f"{p}muscle_force_{window}"))
            
            exprs.append(pl.col("close").map_batches(
                lambda s, w=window: pl.Series(rolling_biomechanical_efficiency_udf(s.to_numpy(), window_size=w))
            ).alias(f"{p}biomechanical_efficiency_{window}"))
            
            exprs.append(pl.col("close").map_batches(
                lambda s, w=window: pl.Series(rolling_energy_expenditure_udf(s.to_numpy(), window_size=w))
            ).alias(f"{p}energy_expenditure_{window}"))
        
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
            
            # 通常のストリーミング出力
            processed_frame.sink_parquet(
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
            "engine_name": "Engine_1F_InterdisciplinaryFeatures",
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
    print("  - 学際的・実験的特徴量（ネットワーク科学、言語学、美学、音楽理論、生体力学）")
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
    print(f"  Engine 1F - Interdisciplinary Features ")
    print("  Project Forge - 軍資金増大プロジェクト  ")
    print("="*70)
    print("🎯 目標: XAU/USD市場の学際的パターン抽出")
    print("🤖 AI頭脳による学際融合法則発見")
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

# ================================
# BLOCK 3 完了 - 全スクリプト完成
# ================================    