#!/usr/bin/env python3
"""
革新的な特徴量収集スクリプト - Engine 2: Rolling MFDFA Features
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
        "volatility": [10, 20, 30, 50],
        "mfdfa": [100, 200, 500, 1000]  # MFDFA専用ウィンドウサイズ
    }

@dataclass
class MFDFAConfig:
    """MFDFA計算設定"""
    scales: List[int] = field(default_factory=lambda: [10, 20, 30, 50, 100])
    q_values: List[float] = field(default_factory=lambda: [-5, -3, -2, -1, 0, 1, 2, 3, 5])
    poly_order: int = 3
    
    def __post_init__(self):
        # q_valuesをnumpy配列に変換
        self.q_values = np.array(self.q_values, dtype=np.float64)
        self.scales = np.array(self.scales, dtype=np.int32)

@dataclass
class ProcessingConfig:
    """処理設定 - Project Forge統合版"""
    
    # データパス（Project Forge構造準拠）
    input_path: str = "/workspaces/project_forge/data/1_XAUUSD_base_data/XAUUSD_tick_master_GOLDEN"
    partitioned_tick_path: str = "/workspaces/project_forge/data/0_tick_partitioned/"
    output_path: str = "/workspaces/project_forge/data/2_feature_value"
    
    # エンジン識別
    engine_id: str = "e2"  # 例: "e1a", "e1b", "e2a"
    engine_name: str = "Engine_2_RollingMFDFA"
    
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
    w_max: int = 1000  # MFDFA用に拡張
    
    # MFDFA設定
    mfdfa_config: MFDFAConfig = field(default_factory=MFDFAConfig)
    
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

# Numbaの計算コア関数（JIT最適化）
@nb.njit(nopython=True, fastmath=True, cache=True)
def _jit_remove_polynomial_trend(segment: np.ndarray, poly_order: int) -> np.ndarray:
    """
    Numba最適化されたポリノミアルトレンド除去
    科学的妥当性を保持したまま高速化
    """
    n = len(segment)
    if n <= poly_order:
        return np.zeros_like(segment)
    
    # 時間インデックス
    t = np.arange(n, dtype=np.float64)
    
    # 段階的降格による数値安定性確保
    for order in range(poly_order, 0, -1):
        try:
            # Vandermonde行列構築
            A = np.zeros((n, order + 1), dtype=np.float64)
            for i in range(n):
                for j in range(order + 1):
                    A[i, j] = t[i] ** j
            
            # 正規方程式による最小二乗法
            AtA = A.T @ A
            Atb = A.T @ segment
            
            # 条件数の簡易チェック（対角要素による）
            diag_min = np.min(np.diag(AtA))
            diag_max = np.max(np.diag(AtA))
            
            if diag_min > 0 and diag_max / diag_min < 1e12:
                # Cholesky分解による高速解法
                try:
                    L = np.linalg.cholesky(AtA)
                    y = np.linalg.solve(L, Atb)
                    coeffs = np.linalg.solve(L.T, y)
                    
                    # トレンド計算
                    trend = A @ coeffs
                    
                    # 結果の妥当性チェック
                    if np.all(np.isfinite(trend)):
                        return trend
                        
                except:
                    pass
            
            # LU分解による安定解法（フォールバック）
            try:
                coeffs = np.linalg.solve(AtA, Atb)
                trend = A @ coeffs
                
                if np.all(np.isfinite(trend)):
                    return trend
                    
            except:
                continue
        
        except:
            continue
    
    # 最終フォールバック: 線形回帰
    try:
        # 線形回帰の解析解
        t_mean = np.mean(t)
        s_mean = np.mean(segment)
        
        numerator = np.sum((t - t_mean) * (segment - s_mean))
        denominator = np.sum((t - t_mean) ** 2)
        
        if denominator > 1e-12:
            slope = numerator / denominator
            intercept = s_mean - slope * t_mean
            trend = slope * t + intercept
            return trend
        else:
            return np.full_like(segment, s_mean)
            
    except:
        # 絶対的フォールバック: 平均値
        return np.full_like(segment, np.mean(segment))

@nb.njit(nopython=True, fastmath=True, cache=True)
def _jit_compute_scale_fluctuations(profile: np.ndarray, scale: int, poly_order: int) -> np.ndarray:
    """
    Numba最適化されたスケール変動計算
    Forward/Backward両方向での統計的信頼性確保
    """
    n = len(profile)
    segments = n // scale
    
    if segments < 2:
        return np.array([np.nan])
    
    fluctuations = np.zeros(segments * 2, dtype=np.float64)  # Forward + Backward
    
    # Forward direction
    for i in range(segments):
        start = i * scale
        end = start + scale
        segment = profile[start:end]
        
        # ポリノミアルトレンド除去
        trend = _jit_remove_polynomial_trend(segment, poly_order)
        residuals = segment - trend
        
        # RMS fluctuation
        fluctuation = np.sqrt(np.mean(residuals**2))
        fluctuations[i] = fluctuation
    
    # Backward direction（統計的信頼性向上のため）
    for i in range(segments):
        start = n - (i + 1) * scale
        end = start + scale
        if start >= 0:
            segment = profile[start:end]
            
            # ポリノミアルトレンド除去
            trend = _jit_remove_polynomial_trend(segment, poly_order)
            residuals = segment - trend
            
            # RMS fluctuation
            fluctuation = np.sqrt(np.mean(residuals**2))
            fluctuations[segments + i] = fluctuation
    
    return fluctuations

@nb.njit(nopython=True, fastmath=True, cache=True)
def _jit_compute_q_fluctuation(fluctuations: np.ndarray, q: float) -> float:
    """
    Numba最適化されたq次揺らぎ計算
    数値安定性確保
    """
    # 有効な値のみフィルタリング
    valid_fluct = fluctuations[np.isfinite(fluctuations) & (fluctuations > 0)]
    
    if len(valid_fluct) < 2:
        return np.nan
    
    if abs(q) < 1e-10:  # q ≈ 0 の場合
        # 幾何平均（対数の算術平均）
        log_fluct = np.log(valid_fluct + 1e-15)  # ゼロ回避
        return np.exp(np.mean(log_fluct))
    else:
        # q次モーメント
        try:
            q_powers = np.power(valid_fluct, q)
            if np.all(np.isfinite(q_powers)):
                mean_q_power = np.mean(q_powers)
                if mean_q_power > 0:
                    return np.power(mean_q_power, 1.0 / q)
            return np.nan
        except:
            return np.nan

@nb.njit(nopython=True, fastmath=True, cache=True)
def _jit_compute_scaling_exponent(log_scales: np.ndarray, log_fluctuations: np.ndarray) -> float:
    """
    Numba最適化されたスケーリング指数計算
    ロバスト線形回帰
    """
    # 有効データのマスク
    valid_mask = np.isfinite(log_scales) & np.isfinite(log_fluctuations)
    
    if np.sum(valid_mask) < 3:
        return np.nan
    
    x = log_scales[valid_mask]
    y = log_fluctuations[valid_mask]
    n = len(x)
    
    # 線形回帰の解析解
    x_mean = np.mean(x)
    y_mean = np.mean(y)
    
    # 分子・分母計算
    numerator = np.sum((x - x_mean) * (y - y_mean))
    denominator = np.sum((x - x_mean) ** 2)
    
    if denominator > 1e-15:
        slope = numerator / denominator
        
        # 理論的妥当性チェック
        if 0.01 <= slope <= 2.5:  # Hurst指数の理論的範囲
            return slope
    
    return np.nan

@nb.njit(nopython=True, fastmath=True, cache=True)
def _jit_singularity_spectrum(q_values: np.ndarray, scaling_exponents: np.ndarray) -> tuple:
    """
    Numba最適化された特異スペクトラム計算
    Legendre変換による
    """
    valid_mask = np.isfinite(scaling_exponents)
    
    if np.sum(valid_mask) < 3:
        return (np.array([np.nan]), np.array([np.nan]), np.nan, np.nan, np.nan)
    
    valid_q = q_values[valid_mask]
    valid_tau = scaling_exponents[valid_mask]
    n_valid = len(valid_q)
    
    alpha = np.zeros(n_valid, dtype=np.float64)
    f_alpha = np.zeros(n_valid, dtype=np.float64)
    
    # Legendre変換によるsingularity spectrum計算
    for i in range(n_valid):
        if i == 0 and n_valid > 1:
            # 前進差分
            alpha[i] = (valid_tau[i+1] - valid_tau[i]) / (valid_q[i+1] - valid_q[i])
        elif i == n_valid - 1 and n_valid > 1:
            # 後退差分
            alpha[i] = (valid_tau[i] - valid_tau[i-1]) / (valid_q[i] - valid_q[i-1])
        elif n_valid > 2:
            # 中央差分
            alpha[i] = (valid_tau[i+1] - valid_tau[i-1]) / (valid_q[i+1] - valid_q[i-1])
        else:
            alpha[i] = valid_tau[i]
        
        # f(α) = qα - τ(q)
        f_alpha[i] = valid_q[i] * alpha[i] - valid_tau[i]
    
    # スペクトラム統計
    max_alpha = np.max(alpha) if len(alpha) > 0 else np.nan
    min_alpha = np.min(alpha) if len(alpha) > 0 else np.nan
    spectrum_width = max_alpha - min_alpha if np.isfinite(max_alpha) and np.isfinite(min_alpha) else np.nan
    
    return (alpha, f_alpha, max_alpha, min_alpha, spectrum_width)

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
        self.prefix = f"e{config.engine_id.replace('e', '')}_"  # 例: "e2_"
        
        # 【修正】ディスクベース垂直分割用の一時ディレクトリ
        self.temp_dir = Path(tempfile.mkdtemp(prefix=f"features_{config.engine_id}_"))
        logger.info(f"一時ディレクトリ作成: {self.temp_dir}")
    
    def _get_all_feature_expressions(self) -> Dict[str, pl.Expr]:
        """
        全特徴量の名前とPolars Expressionの対応辞書を返すファクトリメソッド。
        【効率化修正】MFDFA計算結果を共有して重複計算を排除
        """
        expressions = {}
        p = self.prefix
        mfdfa_config = self.config.mfdfa_config
        
        # 【効率化】各ウィンドウサイズごとに一回だけMFDFA計算を実行し、
        # 結果から全ての派生特徴量を抽出する
        for window in self.config.window_sizes["mfdfa"]:
            # 単一のMFDFA計算から全特徴量を抽出
            expressions.update(
                self._create_mfdfa_features_for_window(window)
            )
        
        return expressions
    
    def _create_mfdfa_features_for_window(self, window: int) -> Dict[str, pl.Expr]:
        """
        単一ウィンドウサイズに対する全MFDFA特徴量を効率的に生成
        【重要】一回のMFDFA計算から全派生特徴量を抽出
        【修正】特異スペクトラム特徴量を追加
        """
        features = {}
        p = self.prefix
        mfdfa_config = self.config.mfdfa_config
        
        # 【↓ここからが修正箇所↓】
        # UDFが返すNumPy配列を .tolist() でPythonリストに変換し、
        # pl.Series() でラップすることで、Polarsが期待するList型を保証する
        base_mfdfa_expr = pl.struct(["close"]).map_batches(
            lambda s, w=window: pl.Series(
                self._compute_all_mfdfa_features_udf(
                    s.struct.field("close").to_numpy(), w
                ).tolist()
            )
        ).alias(f"mfdfa_results_w{window}")
        # 【↑ここまでが修正箇所↑】
        
        # 基本MFDFA特徴量（各q値のHurst指数）
        n_q = len(mfdfa_config.q_values)
        for i, q in enumerate(mfdfa_config.q_values):
            features[f"{p}mfdfa_h_q{q:.1f}_w{window}"] = (
                base_mfdfa_expr.list.get(i).alias(f"{p}mfdfa_h_q{q:.1f}_w{window}")
            )
        
        # q=2のHurst指数（別名）
        q2_idx = np.where(np.abs(mfdfa_config.q_values - 2.0) < 0.1)[0]
        if len(q2_idx) > 0:
            features[f"{p}hurst_exponent_w{window}"] = (
                base_mfdfa_expr.list.get(q2_idx[0]).alias(f"{p}hurst_exponent_w{window}")
            )
        
        # 派生特徴量（計算済み結果から抽出）
        derived_feature_names = [
            "multifractal_width", "asymmetry_index", "correlation_strength",
            "persistence_measure", "anti_persistence_measure", "regime_indicator",
            "volatility_clustering", "long_memory_strength", "window_quality_score",
            "max_singularity", "min_singularity", "spectrum_width", "spectrum_peak"
        ]
        
        for i, name in enumerate(derived_feature_names):
            features[f"{p}{name}_w{window}"] = (
                base_mfdfa_expr.list.get(n_q + i).alias(f"{p}{name}_w{window}")
            )
        
        return features
    
    def _compute_all_mfdfa_features_udf(self, prices: np.ndarray, window_size: int) -> np.ndarray:
        """
        【効率化の核心】単一のMFDFA計算から全特徴量を一括抽出
        一回の計算で全q値のスケーリング指数 + 13個の派生特徴量を返す
        【修正】特異スペクトラム特徴量を追加
        """
        n = len(prices)
        mfdfa_config = self.config.mfdfa_config
        n_q = len(mfdfa_config.q_values)
        n_derived = 13  # 派生特徴量の数（特異スペクトラム4個を追加）
        
        # 結果配列: [q値別スケーリング指数(n_q個) + 派生特徴量(13個)]
        result = np.full((n, n_q + n_derived), np.nan, dtype=np.float64)
        
        if n < window_size:
            return result
        
        # 並列化されたローリング処理
        for i in range(window_size - 1, n):
            # ウィンドウデータ抽出
            window_data = prices[i - window_size + 1:i + 1]
            
            # 有効性チェック
            finite_mask = np.isfinite(window_data)
            if np.sum(finite_mask) < window_size // 2:
                continue
            
            # 【一回のMFDFA計算】全q値のスケーリング指数を計算
            scaling_exponents = self._single_window_mfdfa_array(window_data, mfdfa_config)
            
            # 基本特徴量（各q値のスケーリング指数）
            result[i, :n_q] = scaling_exponents
            
            # 【効率化】計算済み結果から派生特徴量を抽出
            if np.sum(np.isfinite(scaling_exponents)) > 0:
                # Multi-fractal幅
                valid_scaling = scaling_exponents[np.isfinite(scaling_exponents)]
                if len(valid_scaling) > 1:
                    result[i, n_q] = np.max(valid_scaling) - np.min(valid_scaling)
                
                # 非対称性指数
                q_pos = scaling_exponents[mfdfa_config.q_values > 0]
                q_neg = scaling_exponents[mfdfa_config.q_values < 0]
                q_pos_valid = q_pos[np.isfinite(q_pos)]
                q_neg_valid = q_neg[np.isfinite(q_neg)]
                if len(q_pos_valid) > 0 and len(q_neg_valid) > 0:
                    result[i, n_q + 1] = np.mean(q_pos_valid) - np.mean(q_neg_valid)
                
                # 相関強度 (q=2のHurst指数ベース)
                q2_idx = np.where(np.abs(mfdfa_config.q_values - 2.0) < 0.1)[0]
                if len(q2_idx) > 0:
                    h = scaling_exponents[q2_idx[0]]
                    if np.isfinite(h):
                        result[i, n_q + 2] = abs(h - 0.5)  # correlation_strength
                        result[i, n_q + 3] = max(0.0, h - 0.5)  # persistence_measure
                        result[i, n_q + 4] = max(0.0, 0.5 - h)  # anti_persistence_measure
                        result[i, n_q + 7] = abs(h - 0.5)  # long_memory_strength
                
                # 市場レジーム識別子
                if len(valid_scaling) > 1 and len(q2_idx) > 0:
                    mf_width = np.max(valid_scaling) - np.min(valid_scaling)
                    h = scaling_exponents[q2_idx[0]]
                    if np.isfinite(mf_width) and np.isfinite(h):
                        if mf_width > 0.2 and h > 0.6:
                            regime = 1.0  # Strong trending with multifractality
                        elif mf_width < 0.1 and abs(h - 0.5) < 0.1:
                            regime = 0.0  # Random walk-like
                        elif h < 0.4:
                            regime = -1.0  # Mean-reverting
                        else:
                            regime = 0.5  # Intermediate
                        result[i, n_q + 5] = regime
                
                # ボラティリティクラスタリング
                if len(valid_scaling) > 3:
                    result[i, n_q + 6] = np.std(valid_scaling)
                
                # 【追加】特異スペクトラム（Singularity Spectrum）特徴量の計算
                if len(valid_scaling) >= 3:
                    # Legendre変換による特異スペクトラム計算
                    alpha, f_alpha, max_alpha, min_alpha, spectrum_width = _jit_singularity_spectrum(
                        mfdfa_config.q_values, scaling_exponents
                    )
                    
                    # 特異スペクトラム特徴量
                    result[i, n_q + 9] = max_alpha      # max_singularity
                    result[i, n_q + 10] = min_alpha     # min_singularity  
                    result[i, n_q + 11] = spectrum_width # spectrum_width
                    
                    # スペクトラムピーク（最大f(α)に対応するα）
                    if len(f_alpha) > 0 and np.any(np.isfinite(f_alpha)):
                        valid_f_alpha = f_alpha[np.isfinite(f_alpha)]
                        valid_alpha = alpha[np.isfinite(f_alpha)]
                        if len(valid_f_alpha) > 0:
                            peak_idx = np.argmax(valid_f_alpha)
                            result[i, n_q + 12] = valid_alpha[peak_idx]  # spectrum_peak
                
                # ウィンドウ品質スコア
                valid_count = np.sum(np.isfinite(scaling_exponents))
                total_count = len(scaling_exponents)
                if valid_count >= total_count * 0.3:
                    valid_ratio = valid_count / total_count
                    # 簡略化された品質評価
                    validity_score = 1.0  # 詳細な評価は省略
                    result[i, n_q + 8] = min(1.0, valid_ratio * validity_score)
                else:
                    result[i, n_q + 8] = 0.0
        
        return result
    
    def _single_window_mfdfa_array(self, window_data: np.ndarray, mfdfa_config: MFDFAConfig) -> np.ndarray:
        """
        単一ウィンドウのMFDFA計算（配列版）
        【効率化】元の_single_window_mfdfa()と同じロジックだが、配列を直接返す
        """
        # 有効性チェック
        finite_mask = np.isfinite(window_data)
        if np.sum(finite_mask) < len(window_data) // 2:
            return np.full(len(mfdfa_config.q_values), np.nan)
        
        # 対数リターン計算
        valid_data = window_data[finite_mask]
        if len(valid_data) < 10:
            return np.full(len(mfdfa_config.q_values), np.nan)
            
        log_returns = np.diff(np.log(np.maximum(valid_data, 1e-10)))
        
        # 異常値除去（±5σ）
        mean_return = np.mean(log_returns)
        std_return = np.std(log_returns)
        if std_return > 1e-10:
            threshold = 5 * std_return
            outlier_mask = np.abs(log_returns - mean_return) > threshold
            log_returns[outlier_mask] = np.sign(log_returns[outlier_mask]) * threshold + mean_return
        
        # Profile計算（累積和）
        profile = np.cumsum(log_returns - np.mean(log_returns))
        
        # 各スケールでのFluctuation計算
        scaling_exponents = np.zeros(len(mfdfa_config.q_values), dtype=np.float64)
        
        for q_idx, q in enumerate(mfdfa_config.q_values):
            # スケールでの計算
            log_scales = []
            log_fluctuations = []
            
            for scale in mfdfa_config.scales:
                if scale >= len(profile):
                    continue
                
                # スケール変動計算
                scale_fluctuations = _jit_compute_scale_fluctuations(profile, scale, mfdfa_config.poly_order)
                q_fluctuation = _jit_compute_q_fluctuation(scale_fluctuations, q)
                
                if np.isfinite(q_fluctuation) and q_fluctuation > 0:
                    log_scales.append(np.log(scale))
                    log_fluctuations.append(np.log(q_fluctuation))
            
            # スケーリング指数計算
            if len(log_scales) >= 3:
                log_scales_arr = np.array(log_scales)
                log_fluct_arr = np.array(log_fluctuations)
                scaling_exp = _jit_compute_scaling_exponent(log_scales_arr, log_fluct_arr)
                scaling_exponents[q_idx] = scaling_exp
            else:
                scaling_exponents[q_idx] = np.nan
        
        return scaling_exponents
    
    def _compute_multifractal_width(self, prices: np.ndarray, window_size: int) -> np.ndarray:
        """Multi-fractal幅の計算"""
        n = len(prices)
        result = np.full(n, np.nan, dtype=np.float64)
        
        mfdfa_config = self.config.mfdfa_config
        
        for i in range(window_size - 1, n):
            # ウィンドウデータ抽出
            window_data = prices[i - window_size + 1:i + 1]
            
            # 有効性チェック
            finite_mask = np.isfinite(window_data)
            if np.sum(finite_mask) < window_size // 2:
                continue
            
            # MFDFA全q値計算
            scaling_exponents = self._single_window_mfdfa(window_data, mfdfa_config)
            
            # 有効な値での幅計算
            valid_scaling = scaling_exponents[np.isfinite(scaling_exponents)]
            if len(valid_scaling) > 1:
                result[i] = np.max(valid_scaling) - np.min(valid_scaling)
        
        return result
    
    def _compute_asymmetry_index(self, prices: np.ndarray, window_size: int) -> np.ndarray:
        """非対称性指数の計算"""
        n = len(prices)
        result = np.full(n, np.nan, dtype=np.float64)
        
        mfdfa_config = self.config.mfdfa_config
        
        for i in range(window_size - 1, n):
            window_data = prices[i - window_size + 1:i + 1]
            
            finite_mask = np.isfinite(window_data)
            if np.sum(finite_mask) < window_size // 2:
                continue
            
            scaling_exponents = self._single_window_mfdfa(window_data, mfdfa_config)
            
            # 正と負のq値での非対称性
            q_pos = scaling_exponents[mfdfa_config.q_values > 0]
            q_neg = scaling_exponents[mfdfa_config.q_values < 0]
            q_pos_valid = q_pos[np.isfinite(q_pos)]
            q_neg_valid = q_neg[np.isfinite(q_neg)]
            
            if len(q_pos_valid) > 0 and len(q_neg_valid) > 0:
                result[i] = np.mean(q_pos_valid) - np.mean(q_neg_valid)
        
        return result
    
    def _compute_correlation_strength(self, prices: np.ndarray, window_size: int) -> np.ndarray:
        """相関強度の計算"""
        n = len(prices)
        result = np.full(n, np.nan, dtype=np.float64)
        
        mfdfa_config = self.config.mfdfa_config
        
        # q=2に対応するインデックス
        q2_idx = np.where(np.abs(mfdfa_config.q_values - 2.0) < 0.1)[0]
        if len(q2_idx) == 0:
            return result
        
        for i in range(window_size - 1, n):
            window_data = prices[i - window_size + 1:i + 1]
            
            finite_mask = np.isfinite(window_data)
            if np.sum(finite_mask) < window_size // 2:
                continue
            
            scaling_exponents = self._single_window_mfdfa(window_data, mfdfa_config)
            
            h = scaling_exponents[q2_idx[0]]
            if np.isfinite(h):
                result[i] = abs(h - 0.5)
        
        return result
    
    def _compute_persistence_measure(self, prices: np.ndarray, window_size: int) -> np.ndarray:
        """持続性測度の計算"""
        n = len(prices)
        result = np.full(n, np.nan, dtype=np.float64)
        
        mfdfa_config = self.config.mfdfa_config
        q2_idx = np.where(np.abs(mfdfa_config.q_values - 2.0) < 0.1)[0]
        if len(q2_idx) == 0:
            return result
        
        for i in range(window_size - 1, n):
            window_data = prices[i - window_size + 1:i + 1]
            
            finite_mask = np.isfinite(window_data)
            if np.sum(finite_mask) < window_size // 2:
                continue
            
            scaling_exponents = self._single_window_mfdfa(window_data, mfdfa_config)
            
            h = scaling_exponents[q2_idx[0]]
            if np.isfinite(h):
                result[i] = max(0.0, h - 0.5)
        
        return result
    
    def _compute_anti_persistence_measure(self, prices: np.ndarray, window_size: int) -> np.ndarray:
        """反持続性測度の計算"""
        n = len(prices)
        result = np.full(n, np.nan, dtype=np.float64)
        
        mfdfa_config = self.config.mfdfa_config
        q2_idx = np.where(np.abs(mfdfa_config.q_values - 2.0) < 0.1)[0]
        if len(q2_idx) == 0:
            return result
        
        for i in range(window_size - 1, n):
            window_data = prices[i - window_size + 1:i + 1]
            
            finite_mask = np.isfinite(window_data)
            if np.sum(finite_mask) < window_size // 2:
                continue
            
            scaling_exponents = self._single_window_mfdfa(window_data, mfdfa_config)
            
            h = scaling_exponents[q2_idx[0]]
            if np.isfinite(h):
                result[i] = max(0.0, 0.5 - h)
        
        return result
    
    def _compute_regime_indicator(self, prices: np.ndarray, window_size: int) -> np.ndarray:
        """市場レジーム識別子の計算"""
        n = len(prices)
        result = np.full(n, np.nan, dtype=np.float64)
        
        mfdfa_config = self.config.mfdfa_config
        q2_idx = np.where(np.abs(mfdfa_config.q_values - 2.0) < 0.1)[0]
        if len(q2_idx) == 0:
            return result
        
        for i in range(window_size - 1, n):
            window_data = prices[i - window_size + 1:i + 1]
            
            finite_mask = np.isfinite(window_data)
            if np.sum(finite_mask) < window_size // 2:
                continue
            
            scaling_exponents = self._single_window_mfdfa(window_data, mfdfa_config)
            
            # Multi-fractal幅とHurst指数の組み合わせ
            valid_scaling = scaling_exponents[np.isfinite(scaling_exponents)]
            if len(valid_scaling) > 1:
                mf_width = np.max(valid_scaling) - np.min(valid_scaling)
                h = scaling_exponents[q2_idx[0]]
                
                if np.isfinite(mf_width) and np.isfinite(h):
                    if mf_width > 0.2 and h > 0.6:
                        regime = 1.0  # Strong trending with multifractality
                    elif mf_width < 0.1 and abs(h - 0.5) < 0.1:
                        regime = 0.0  # Random walk-like
                    elif h < 0.4:
                        regime = -1.0  # Mean-reverting
                    else:
                        regime = 0.5  # Intermediate
                    result[i] = regime
        
        return result
    
    def _compute_volatility_clustering(self, prices: np.ndarray, window_size: int) -> np.ndarray:
        """ボラティリティクラスタリング強度の計算"""
        n = len(prices)
        result = np.full(n, np.nan, dtype=np.float64)
        
        mfdfa_config = self.config.mfdfa_config
        
        for i in range(window_size - 1, n):
            window_data = prices[i - window_size + 1:i + 1]
            
            finite_mask = np.isfinite(window_data)
            if np.sum(finite_mask) < window_size // 2:
                continue
            
            scaling_exponents = self._single_window_mfdfa(window_data, mfdfa_config)
            
            # q値の範囲でのscaling exponentの変動
            valid_scaling = scaling_exponents[np.isfinite(scaling_exponents)]
            if len(valid_scaling) > 3:
                result[i] = np.std(valid_scaling)
        
        return result
    
    def _compute_long_memory_strength(self, prices: np.ndarray, window_size: int) -> np.ndarray:
        """長期記憶強度の計算"""
        n = len(prices)
        result = np.full(n, np.nan, dtype=np.float64)
        
        mfdfa_config = self.config.mfdfa_config
        q2_idx = np.where(np.abs(mfdfa_config.q_values - 2.0) < 0.1)[0]
        if len(q2_idx) == 0:
            return result
        
        for i in range(window_size - 1, n):
            window_data = prices[i - window_size + 1:i + 1]
            
            finite_mask = np.isfinite(window_data)
            if np.sum(finite_mask) < window_size // 2:
                continue
            
            scaling_exponents = self._single_window_mfdfa(window_data, mfdfa_config)
            
            h = scaling_exponents[q2_idx[0]]
            if np.isfinite(h):
                # 相関強度をベースとした長期記憶強度
                result[i] = abs(h - 0.5)
        
        return result
    
    def _compute_window_quality_score(self, prices: np.ndarray, window_size: int) -> np.ndarray:
        """ウィンドウ品質スコアの計算"""
        n = len(prices)
        result = np.full(n, np.nan, dtype=np.float64)
        
        mfdfa_config = self.config.mfdfa_config
        
        for i in range(window_size - 1, n):
            window_data = prices[i - window_size + 1:i + 1]
            
            finite_mask = np.isfinite(window_data)
            if np.sum(finite_mask) < window_size // 2:
                result[i] = 0.0
                continue
            
            scaling_exponents = self._single_window_mfdfa(window_data, mfdfa_config)
            
            # 品質評価
            valid_count = np.sum(np.isfinite(scaling_exponents))
            total_count = len(scaling_exponents)
            
            if valid_count < total_count * 0.3:  # 30%未満が有効
                result[i] = 0.0
                continue
            
            valid_ratio = valid_count / total_count
            
            # 理論的妥当性チェック
            valid_exp = scaling_exponents[np.isfinite(scaling_exponents)]
            if len(valid_exp) > 0:
                validity_count = 0
                for j, q in enumerate(mfdfa_config.q_values):
                    if j < len(scaling_exponents) and np.isfinite(scaling_exponents[j]):
                        exp_val = scaling_exponents[j]
                        if q > 0:
                            valid_range = (0.1 <= exp_val <= 1.5)
                        elif q < 0:
                            valid_range = (0.1 <= exp_val <= 2.5)
                        else:  # q=0
                            valid_range = (0.3 <= exp_val <= 1.2)
                        
                        if valid_range:
                            validity_count += 1
                
                validity_score = validity_count / len(mfdfa_config.q_values)
            else:
                validity_score = 0.0
            
            result[i] = min(1.0, valid_ratio * validity_score)
        
        return result
    
    def _single_window_mfdfa(self, window_data: np.ndarray, mfdfa_config: MFDFAConfig) -> np.ndarray:
        """単一ウィンドウのMFDFA計算"""
        # 有効性チェック
        finite_mask = np.isfinite(window_data)
        if np.sum(finite_mask) < len(window_data) // 2:
            return np.full(len(mfdfa_config.q_values), np.nan)
        
        # 対数リターン計算
        valid_data = window_data[finite_mask]
        if len(valid_data) < 10:
            return np.full(len(mfdfa_config.q_values), np.nan)
            
        log_returns = np.diff(np.log(np.maximum(valid_data, 1e-10)))
        
        # 異常値除去（±5σ）
        mean_return = np.mean(log_returns)
        std_return = np.std(log_returns)
        if std_return > 1e-10:
            threshold = 5 * std_return
            outlier_mask = np.abs(log_returns - mean_return) > threshold
            log_returns[outlier_mask] = np.sign(log_returns[outlier_mask]) * threshold + mean_return
        
        # Profile計算（累積和）
        profile = np.cumsum(log_returns - np.mean(log_returns))
        
        # 各スケールでのFluctuation計算
        scaling_exponents = np.zeros(len(mfdfa_config.q_values), dtype=np.float64)
        
        for q_idx, q in enumerate(mfdfa_config.q_values):
            # スケールでの計算
            log_scales = []
            log_fluctuations = []
            
            for scale in mfdfa_config.scales:
                if scale >= len(profile):
                    continue
                
                # スケール変動計算
                scale_fluctuations = _jit_compute_scale_fluctuations(profile, scale, mfdfa_config.poly_order)
                q_fluctuation = _jit_compute_q_fluctuation(scale_fluctuations, q)
                
                if np.isfinite(q_fluctuation) and q_fluctuation > 0:
                    log_scales.append(np.log(scale))
                    log_fluctuations.append(np.log(q_fluctuation))
            
            # スケーリング指数計算
            if len(log_scales) >= 3:
                log_scales_arr = np.array(log_scales)
                log_fluct_arr = np.array(log_fluctuations)
                scaling_exp = _jit_compute_scaling_exponent(log_scales_arr, log_fluct_arr)
                scaling_exponents[q_idx] = scaling_exp
            else:
                scaling_exponents[q_idx] = np.nan
        
        return scaling_exponents
    
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
    
    def calculate_one_group(self, lazy_frame: pl.LazyFrame, group_name: str, group_expressions: Dict[str, pl.Expr]) -> pl.LazyFrame:
        """単一グループの特徴量のみを計算"""
        logger.info(f"グループ計算開始: {group_name}")
        
        group_result_lf = lazy_frame.with_columns(list(group_expressions.values()))
        
        # このグループの計算で使用した一時的な配列カラムを特定して削除
        # group_nameは 'mfdfa_w100' のような形式
        window_size_str = group_name.split('w')[-1]
        temp_col_to_drop = f"mfdfa_results_w{window_size_str}"
        # 【↓ここを修正↓】
        group_result_lf = group_result_lf.drop(temp_col_to_drop, strict=False)
        # 【↑ここまで↑】
        
        available_schema = group_result_lf.collect_schema()
        available_columns = list(available_schema.names())
        
        base_columns = ["timestamp", "open", "high", "low", "close", "volume", "timeframe"]
        group_feature_columns = [col for col in available_columns if col.startswith(self.prefix)]
        select_columns = base_columns + group_feature_columns
        
        final_select_columns = [col for col in select_columns if col in available_columns]
        group_final_lf = group_result_lf.select(final_select_columns)
        
        stabilized_lf = self.apply_quality_assurance_to_group(group_final_lf, group_feature_columns)
        
        logger.info(f"グループ計算完了: {group_name} - {len(group_feature_columns)}個の特徴量")
        return stabilized_lf
    
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

        # --- 特徴量計算ロジック ---
        # 全特徴量を一度に取得して計算する
        all_expressions_dict = calc_engine._get_all_feature_expressions()
        all_expressions_list = list(all_expressions_dict.values())

        logger.info(f"特徴量計算開始: {len(all_expressions_list)}個の特徴量を {timeframe} に対して計算します。")
        
        # with_columnsで全特徴量を追加
        features_lf = lazy_frame.with_columns(all_expressions_list)
        
        # 計算に使用した一時的な配列カラムを削除する
        temp_cols_to_drop = [f"mfdfa_results_w{w}" for w in config.window_sizes["mfdfa"]]
        # 【↓ここを修正↓】
        features_lf = features_lf.drop(temp_cols_to_drop, strict=False)
        # 【↑ここまで↑】

        # 品質保証システムを適用
        schema = features_lf.collect_schema()
        feature_columns = [col for col in schema.names() if col.startswith(calc_engine.prefix)]
        features_lf = calc_engine.apply_quality_assurance_to_group(features_lf, feature_columns)
        
        # NaN埋めと保存はOutputEngineが担当
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
    print("  - ローリングMFDFA（多重フラクタル次元解析）の実行")
    print("    - q値ごとのHurst指数（市場の長期記憶・トレンド性）")
    print("    - マルチフラクタル性の幅（市場の複雑性）")
    print("    - 非対称性インデックス（価格変動の偏り）")
    print("  - 特異スペクトラム特徴量の計算")
    print("    - スペクトラムの幅・最大/最小値（フラクタル構造の豊かさ）")
    print("  - MFDFAからの派生指標")
    print("    - 市場レジーム（トレンド、ランダム、平均回帰）の推定")
    print("    - 持続性・反持続性の定量化")
    print("  - Project Forge共通アーキテクチャ")
    print("    - 品質保証システムによる数値安定化")
    print("    - Tickデータに対する物理的垂直分割")
    print("    - Polars LazyFrame + Numba JITによるハイブリッド最適化")
    
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