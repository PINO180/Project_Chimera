#!/usr/bin/env python3
"""
革新的特徴量収集スクリプト - Engine 2A: Complexity Theory Features (簡略版)
【実装内容】複雑性理論特徴量群(F16, F21のみ)

Project Forge - 軍資金増大プロジェクト
最終目標: Project Chimera開発・完成のための資金調達

技術戦略: ジム・シモンズの思想的継承
- 経済学・ファンダメンタルズ・古典的テクニカル指標の完全排除
- 統計的に有意で非ランダムな微細パターン「マーケットの亡霊」の探索
- AIの頭脳による普遍的法則の読み解き

アーキテクチャ: 3クラス構成(最適化版) + ディスクベース垂直分割
- DataEngine(30%): Polars LazyFrame基盤
- CalculationEngine(60%): 特徴量計算核心(物理的垂直分割実装)
- OutputEngine(10%): ストリーミング出力

【削除済み】
- F5: MFDFA（マルチフラクタル解析）
- F15: コルモゴロフ複雑性

【実装済み】
- F16: カオス理論指標（リアプノフ指数・相関次元）
- F21: Lempel-Ziv複雑性
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
        "chaos": [10000, 20000, 50000],  # カオス理論用ウィンドウ
        "lz_complexity": [500, 1000, 1500],  # Lempel-Ziv複雑性用ウィンドウ
        "general": [50, 100, 200, 500]
    }

@dataclass
class ProcessingConfig:
    """処理設定 - Project Forge統合版 (Engine 2A: Complexity Theory - Simplified)"""
    
    # データパス(Project Forge構造準拠)
    input_path: str = "/workspaces/project_forge/data/1_XAUUSD_base_data/XAUUSD_tick_master_GOLDEN"
    partitioned_tick_path: str = "/workspaces/project_forge/data/0_tick_partitioned/"
    output_path: str = "/workspaces/project_forge/data/2_feature_value"
    
    # エンジン識別
    engine_id: str = "e2b"  # Engine 2B: Complexity Theory
    engine_name: str = "Engine_2B_ComplexityTheory_Simplified"
    
    # 並列処理(戦略的並列処理スロットリング)
    max_threads: int = 4
    
    # メモリ制限(64GB RAM制約)
    memory_limit_gb: float = 55.0
    memory_warning_gb: float = 50.0
    
    timeframes: List[str] = field(default_factory=get_default_timeframes)
    window_sizes: Dict[str, List[int]] = field(default_factory=get_default_window_sizes)
    
    # 処理モード
    test_mode: bool = False
    test_rows: int = 10000
    
    # システムハイパーパラメータとしてW_maxを定義
    # この値は、全特徴量計算の最大ウィンドウサイズを反映しなければならない
    w_max: int = 500  # カオス理論とLZ複雑性の最大ウィンドウ
    
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

# ===============================
# Block 1 Complete: ヘッダーと基本設定
# ===============================

# ===============================
# Block 2 Start: カオス理論指標UDF定義（修正版）
# ===============================

# 【重要】全てのNumba UDFは必ずクラス外で定義すること
# クラス内定義は解決不能な循環参照エラーを引き起こすため絶対禁止

# =============================================================================
# カオス理論指標 (リアプノフ指数・相関次元) UDF群
# 【修正版】ローリング処理を内部化、map_batches対応、prange並列化
# =============================================================================

@nb.njit(fastmath=True, cache=True)
def estimate_time_delay(values: np.ndarray, max_delay: int = 50) -> int:
    """
    最適時間遅延の推定（自己相互情報量の最初の最小値）
    簡略版: 自己相関関数の最初のゼロ交差
    
    Args:
        values: 入力時系列
        max_delay: 最大遅延
    
    Returns:
        推定遅延τ
    """
    n = len(values)
    if n < max_delay * 2:
        return 1
    
    # 自己相関計算
    mean_val = np.mean(values)
    var_val = np.var(values)
    
    if var_val < 1e-10:
        return 1
    
    for tau in range(1, max_delay):
        if n - tau < 10:
            break
        
        # 相関計算
        correlation = 0.0
        count = 0
        for i in range(n - tau):
            correlation += (values[i] - mean_val) * (values[i + tau] - mean_val)
            count += 1
        
        correlation = correlation / (count * var_val) if count > 0 else 0.0
        
        # ゼロ交差検出
        if correlation < 0.1:  # 閾値
            return tau
    
    return 1

@nb.njit(fastmath=True, cache=True)
def estimate_embedding_dimension(values: np.ndarray, tau: int, max_dim: int = 10) -> int:
    """
    埋め込み次元の推定（False Nearest Neighbors法の簡略版）
    
    Args:
        values: 入力時系列
        tau: 時間遅延
        max_dim: 最大埋め込み次元
    
    Returns:
        推定埋め込み次元m
    """
    n = len(values)
    
    # 簡略版: 十分なデータ点を確保できる次元を選択
    for m in range(2, max_dim + 1):
        required_length = (m - 1) * tau + 1
        if required_length > n // 2:
            return max(2, m - 1)
    
    return 3  # デフォルト

@nb.njit(fastmath=True, cache=True)
def reconstruct_phase_space(values: np.ndarray, tau: int, m: int) -> np.ndarray:
    """
    位相空間の再構成
    
    Args:
        values: 入力時系列
        tau: 時間遅延
        m: 埋め込み次元
    
    Returns:
        位相空間座標 (n_points, m)
    """
    n = len(values)
    n_points = n - (m - 1) * tau
    
    if n_points <= 0:
        return np.zeros((0, m))
    
    phase_space = np.zeros((n_points, m))
    
    for i in range(n_points):
        for j in range(m):
            phase_space[i, j] = values[i + j * tau]
    
    return phase_space

@nb.njit(fastmath=True, cache=True)
def calculate_lyapunov_exponent(values: np.ndarray, tau: int, m: int) -> float:
    """
    最大リアプノフ指数の推定（Rosenstein法）
    
    Args:
        values: 入力時系列
        tau: 時間遅延
        m: 埋め込み次元
    
    Returns:
        最大リアプノフ指数λ
    """
    # 位相空間再構成
    phase_space = reconstruct_phase_space(values, tau, m)
    n_points = phase_space.shape[0]
    
    if n_points < 20:
        return np.nan
    
    # 最近傍点の探索と発散率の計算
    max_evolution = min(n_points // 2, 100)
    divergences = np.zeros(max_evolution)
    counts = np.zeros(max_evolution)
    
    for i in range(n_points - max_evolution):
        # 最近傍点を探索
        min_dist = np.inf
        nearest_idx = -1
        
        for j in range(n_points - max_evolution):
            if abs(i - j) < 5:  # 時間的に近いものは除外
                continue
            
            # ユークリッド距離
            dist = 0.0
            for k in range(m):
                diff = phase_space[i, k] - phase_space[j, k]
                dist += diff * diff
            dist = np.sqrt(dist)
            
            if dist < min_dist and dist > 1e-10:
                min_dist = dist
                nearest_idx = j
        
        if nearest_idx == -1:
            continue
        
        # 軌道の発散を追跡
        for t in range(max_evolution):
            if i + t >= n_points or nearest_idx + t >= n_points:
                break
            
            # 時刻tでの距離
            dist_t = 0.0
            for k in range(m):
                diff = phase_space[i + t, k] - phase_space[nearest_idx + t, k]
                dist_t += diff * diff
            dist_t = np.sqrt(dist_t)
            
            if dist_t > 1e-10:
                divergences[t] += np.log(dist_t)
                counts[t] += 1
    
    # 平均発散率からリアプノフ指数を推定
    if counts[0] > 0:
        for t in range(max_evolution):
            if counts[t] > 0:
                divergences[t] /= counts[t]
    
    # 線形回帰で傾きを推定
    valid_points = []
    for t in range(max_evolution):
        if counts[t] > 0 and np.isfinite(divergences[t]):
            valid_points.append((float(t), divergences[t]))
    
    if len(valid_points) < 10:
        return np.nan
    
    # 最小二乗法
    x_vals = np.array([p[0] for p in valid_points])
    y_vals = np.array([p[1] for p in valid_points])
    
    x_mean = np.mean(x_vals)
    y_mean = np.mean(y_vals)
    
    numerator = 0.0
    denominator = 0.0
    for i in range(len(x_vals)):
        x_diff = x_vals[i] - x_mean
        numerator += x_diff * (y_vals[i] - y_mean)
        denominator += x_diff * x_diff
    
    if abs(denominator) < 1e-10:
        return np.nan
    
    lyapunov = numerator / denominator
    return lyapunov

@nb.njit(fastmath=True, cache=True)
def calculate_correlation_dimension(values: np.ndarray, tau: int, m: int) -> float:
    """
    相関次元の推定（Grassberger-Procaccia法）
    
    Args:
        values: 入力時系列
        tau: 時間遅延
        m: 埋め込み次元
    
    Returns:
        相関次元D
    """
    # 位相空間再構成
    phase_space = reconstruct_phase_space(values, tau, m)
    n_points = phase_space.shape[0]
    
    if n_points < 50:
        return np.nan
    
    # 距離の範囲を決定
    all_distances = []
    sample_size = min(n_points, 500)  # サンプリングで高速化
    
    for i in range(0, sample_size, 5):
        for j in range(i + 1, sample_size, 5):
            dist = 0.0
            for k in range(m):
                diff = phase_space[i, k] - phase_space[j, k]
                dist += diff * diff
            dist = np.sqrt(dist)
            all_distances.append(dist)
    
    if len(all_distances) < 10:
        return np.nan
    
    all_distances_arr = np.array(all_distances)
    all_distances_arr.sort()
    
    # 半径rの範囲を設定
    r_min = np.percentile(all_distances_arr, 10)
    r_max = np.percentile(all_distances_arr, 50)
    
    if r_max <= r_min or r_min < 1e-10:
        return np.nan
    
    # 相関積分C(r)の計算
    n_radii = 10
    log_r_vals = []
    log_C_vals = []
    
    for i in range(n_radii):
        r = r_min * np.power(r_max / r_min, i / (n_radii - 1))
        
        # C(r) = P(distance < r)
        count = 0
        total = 0
        for i_pt in range(0, n_points, 5):
            for j_pt in range(i_pt + 1, n_points, 5):
                dist = 0.0
                for k in range(m):
                    diff = phase_space[i_pt, k] - phase_space[j_pt, k]
                    dist += diff * diff
                dist = np.sqrt(dist)
                
                if dist < r:
                    count += 1
                total += 1
        
        if total > 0:
            C_r = count / total
            if C_r > 1e-10:
                log_r_vals.append(np.log(r))
                log_C_vals.append(np.log(C_r))
    
    if len(log_r_vals) < 5:
        return np.nan
    
    # log(C(r)) vs log(r) の傾きから次元を推定
    x_arr = np.array(log_r_vals)
    y_arr = np.array(log_C_vals)
    
    x_mean = np.mean(x_arr)
    y_mean = np.mean(y_arr)
    
    numerator = 0.0
    denominator = 0.0
    for i in range(len(x_arr)):
        x_diff = x_arr[i] - x_mean
        numerator += x_diff * (y_arr[i] - y_mean)
        denominator += x_diff * x_diff
    
    if abs(denominator) < 1e-10:
        return np.nan
    
    correlation_dim = numerator / denominator
    return max(0.0, correlation_dim)

@nb.njit(fastmath=True, cache=True)
def chaos_indicators_single_window(prices: np.ndarray) -> np.ndarray:
    """
    単一ウィンドウのカオス理論指標計算（ヘルパー関数）
    
    Returns:
        [lyapunov_exponent, lyapunov_time, correlation_dim, embedding_dim]
    """
    result = np.full(4, np.nan)
    
    n = len(prices)
    if n < 100:
        return result
    
    # 1. 最適時間遅延の推定
    tau = estimate_time_delay(prices, max_delay=20)
    
    # 2. 埋め込み次元の推定
    m = estimate_embedding_dimension(prices, tau, max_dim=8)
    
    # 3. リアプノフ指数の計算
    lyapunov = calculate_lyapunov_exponent(prices, tau, m)
    
    # 4. リアプノフ時間の計算（予測可能期間）
    lyapunov_time = 1.0 / lyapunov if lyapunov > 1e-10 else np.nan
    
    # 5. 相関次元の計算
    corr_dim = calculate_correlation_dimension(prices, tau, m)
    
    result[0] = lyapunov
    result[1] = lyapunov_time
    result[2] = corr_dim
    result[3] = float(m)
    
    return result

@nb.njit(fastmath=True, cache=True, parallel=True)
def chaos_indicators_rolling_udf(prices: np.ndarray, window: int, component_idx: int) -> np.ndarray:
    """
    カオス理論指標ローリング計算（map_batches対応版）
    
    Args:
        prices: 価格時系列全体
        window: ウィンドウサイズ
        component_idx: 結果成分インデックス（0=lyapunov, 1=lyapunov_time, 2=corr_dim, 3=embedding_dim）
    
    Returns:
        結果配列（最初のwindow-1個はNaN）
    """
    n = len(prices)
    result = np.full(n, np.nan)
    
    # prange並列化：各ウィンドウ位置を並列処理
    for i in nb.prange(window, n + 1):
        window_data = prices[i - window:i]
        chaos_result = chaos_indicators_single_window(window_data)
        result[i - 1] = chaos_result[component_idx]
    
    return result

# ===============================
# Block 2 Complete: カオス理論指標UDF定義（修正版）
# ===============================

# ===============================
# Block 3 Start: Lempel-Ziv複雑性UDF定義（修正版）
# ===============================

# =============================================================================
# Lempel-Ziv複雑性 (F21) UDF群
# 【修正版】ローリング処理を内部化、map_batches対応、prange並列化
# =============================================================================

@nb.njit(fastmath=True, cache=True)
def lz76_complexity_detailed(sequence: np.ndarray) -> Tuple[float, int, float]:
    """
    Lempel-Ziv 76アルゴリズムによる詳細な複雑性計算
    
    Args:
        sequence: 整数符号化された時系列
    
    Returns:
        (正規化複雑性, 辞書サイズ, 複雑性成長率)
    """
    n = len(sequence)
    if n < 2:
        return 0.0, 0, 0.0
    
    # 辞書構築
    dictionary_size = 0
    i = 0
    growth_history = []
    
    while i < n:
        # 最長一致を探索
        max_match_length = 0
        
        for start in range(i):
            match_length = 0
            j = 0
            while (i + j < n and 
                   start + j < i and 
                   sequence[i + j] == sequence[start + j]):
                match_length += 1
                j += 1
            
            if match_length > max_match_length:
                max_match_length = match_length
        
        # 新パターン追加
        dictionary_size += 1
        
        # 成長率の記録（最初の数十個）
        if len(growth_history) < 50:
            growth_history.append(dictionary_size)
        
        # 進行
        if max_match_length == 0:
            i += 1
        else:
            i += max_match_length + 1
    
    # 正規化複雑性
    if n > 1:
        max_complexity = n / (np.log2(n) + 1e-10)
        normalized = dictionary_size / max_complexity if max_complexity > 0 else 0.0
    else:
        normalized = 0.0
    
    # 複雑性成長率（初期の傾き）
    growth_rate = 0.0
    if len(growth_history) >= 10:
        # 最初の10個の平均成長率
        for i in range(1, min(10, len(growth_history))):
            growth_rate += (growth_history[i] - growth_history[i-1])
        growth_rate /= min(9, len(growth_history) - 1)
    
    return min(normalized, 1.0), dictionary_size, growth_rate

@nb.njit(fastmath=True, cache=True)
def adaptive_binarization(values: np.ndarray, window: int = 50) -> np.ndarray:
    """
    適応的閾値によるバイナリ化
    ローリング中央値を使用
    
    Args:
        values: 入力時系列
        window: ローリングウィンドウサイズ
    
    Returns:
        バイナリ化系列
    """
    n = len(values)
    binary = np.zeros(n, dtype=np.int32)
    
    for i in range(n):
        # ローリング中央値計算
        start = max(0, i - window + 1)
        end = i + 1
        window_vals = values[start:end]
        
        if len(window_vals) > 0:
            median_val = np.median(window_vals)
            binary[i] = 1 if values[i] > median_val else 0
    
    return binary

@nb.njit(fastmath=True, cache=True)
def lempel_ziv_comprehensive_single_window(prices: np.ndarray) -> np.ndarray:
    """
    単一ウィンドウのLempel-Ziv複雑性包括計算（ヘルパー関数）
    
    Returns:
        [complexity, growth_rate, dictionary_saturation, pattern_innovation, complexity_volatility]
    """
    result = np.full(5, np.nan)
    
    n = len(prices)
    if n < 50:
        return result
    
    # 1. 対数リターン計算
    returns = np.zeros(n - 1)
    for i in range(n - 1):
        if prices[i] > 1e-10:
            returns[i] = np.log(prices[i + 1] / prices[i])
        else:
            returns[i] = 0.0
    
    # 2. 標準化
    returns_mean = np.mean(returns)
    returns_std = np.std(returns)
    
    if returns_std < 1e-10:
        return result
    
    standardized = np.zeros(len(returns))
    for i in range(len(returns)):
        standardized[i] = (returns[i] - returns_mean) / returns_std
    
    # 3. 適応的バイナリ化
    binary = adaptive_binarization(standardized, window=50)
    
    # 4. LZ複雑性計算
    complexity, dict_size, growth_rate = lz76_complexity_detailed(binary)
    
    # 5. 辞書飽和度（辞書サイズ vs 理論最大）
    theoretical_max = len(binary) / (np.log2(len(binary)) + 1e-10)
    saturation = dict_size / theoretical_max if theoretical_max > 0 else 0.0
    
    # 6. パターン革新率（ローリング複雑性の変化率）
    window_size = 100
    if len(binary) >= window_size * 2:
        # 前半と後半の複雑性比較
        mid_point = len(binary) // 2
        first_half = binary[:mid_point]
        second_half = binary[mid_point:]
        
        _, dict_first, _ = lz76_complexity_detailed(first_half)
        _, dict_second, _ = lz76_complexity_detailed(second_half)
        
        if dict_first > 0:
            innovation_rate = (dict_second - dict_first) / dict_first
        else:
            innovation_rate = 0.0
    else:
        innovation_rate = 0.0
    
    # 7. 複雑性ボラティリティ（ローリング複雑性の標準偏差）
    if len(binary) >= window_size * 3:
        n_windows = len(binary) // window_size
        window_complexities = np.zeros(n_windows)
        
        for i in range(n_windows):
            start = i * window_size
            end = start + window_size
            window_data = binary[start:end]
            comp, _, _ = lz76_complexity_detailed(window_data)
            window_complexities[i] = comp
        
        complexity_vol = np.std(window_complexities)
    else:
        complexity_vol = 0.0
    
    result[0] = complexity
    result[1] = growth_rate
    result[2] = saturation
    result[3] = innovation_rate
    result[4] = complexity_vol
    
    return result

@nb.njit(fastmath=True, cache=True, parallel=True)
def lempel_ziv_comprehensive_rolling_udf(prices: np.ndarray, window: int, component_idx: int) -> np.ndarray:
    """
    Lempel-Ziv複雑性包括ローリング計算（map_batches対応版）
    
    Args:
        prices: 価格時系列全体
        window: ウィンドウサイズ
        component_idx: 結果成分インデックス
    
    Returns:
        結果配列（最初のwindow-1個はNaN）
    """
    n = len(prices)
    result = np.full(n, np.nan)
    
    # prange並列化：各ウィンドウ位置を並列処理
    for i in nb.prange(window, n + 1):
        window_data = prices[i - window:i]
        lz_result = lempel_ziv_comprehensive_single_window(window_data)
        result[i - 1] = lz_result[component_idx]
    
    return result

# ===============================
# Block 3 Complete: Lempel-Ziv複雑性UDF定義（修正版）
# ===============================

# ===============================
# Block 4 Start: DataEngine/QualityAssurance クラス
# ===============================

class DataEngine:
    """
    データ基盤クラス(30%) - Project Forge統合版
    責務:
    - Parquetメタデータ事前検証
    - Polars scan_parquetによる遅延読み込み(LazyFrame生成)
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
            
            # LazyFrameでメタデータ取得(実際の読み込みなし)
            lazy_frame = pl.scan_parquet(parquet_pattern)
            
            # スキーマ情報取得(警告回避)
            schema = lazy_frame.collect_schema()
            
            # 基本メタデータ収集
            metadata = {
                "schema": dict(schema),
                "columns": list(schema.keys()),
                "path_exists": Path(self.config.input_path).exists(),
                "estimated_memory_gb": 0.0
            }
            
            # 必須カラムチェック(XAU/USDデータ構造)
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
            
            # Hiveパーティション対応:特定timeframeディレクトリを直接指定
            timeframe_path = f"{self.config.input_path}/timeframe={timeframe}/*.parquet"
            
            # 指定timeframeのParquetファイルのみをスキャン
            lazy_frame = pl.scan_parquet(timeframe_path)
            
            # timeframe列を手動で追加 (Hiveパーティション復元)
            lazy_frame = lazy_frame.with_columns([
                pl.lit(timeframe).alias("timeframe").cast(pl.Categorical)
            ])
            
            # スキーマを確認してから安全にキャスト処理を適用
            try:
                sample_schema = lazy_frame.limit(1).collect_schema()
                logger.info(f"検出されたスキーマ: {list(sample_schema.keys())}")
                
                # 基本データ型確認と最適化(必要な場合のみキャスト)
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
                # キャストエラーの場合は処理を続行(データがすでに適切な型の可能性)
            
            # タイムスタンプソート
            lazy_frame = lazy_frame.sort("timestamp")
            
            logger.info(f"LazyFrame生成完了: timeframe={timeframe}")
            return lazy_frame
            
        except Exception as e:
            logger.error(f"LazyFrame生成エラー (timeframe={timeframe}): {e}")
            raise
    
    def get_data_summary(self, lazy_frame: pl.LazyFrame) -> Dict[str, Any]:
        """データサマリー情報取得(軽量)"""
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
    """数値安定性保証システム - 2段階品質保証エンジン(Project Forge準拠)"""
    
    @staticmethod
    def calculate_quality_score(values: np.ndarray) -> float:
        """
        品質スコア算出:0.0(使用不可)〜 1.0(完璧)
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
        unique_values, counts = np.unique(finite_values, return_counts=True)
        n_unique = len(unique_values)
        
        if n_unique == 1:
            diversity_score = 0.0
        else:
            # シャノンエントロピーの厳密計算
            shannon_entropy = 0.0
            for count in counts:
                p = count / finite_count
                if p > 0:
                    shannon_entropy -= p * np.log2(p)
            
            max_entropy = np.log2(min(n_unique, finite_count))
            if max_entropy > 0:
                diversity_score = shannon_entropy / max_entropy
            else:
                diversity_score = 0.0
        
        # 3. 数値的安定性指標の厳密計算
        if finite_count < 2:
            stability_score = 1.0
        else:
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
            
            if abs(mean_val) > 1e-15:
                cv = std_dev / abs(mean_val)
                stability_score = 1.0 / (1.0 + cv)
            else:
                stability_score = 1.0 / (1.0 + std_dev)
        
        # 4. 外れ値耐性指標の厳密計算
        if finite_count < 5:
            outlier_resistance = 1.0
        else:
            sorted_values = np.sort(finite_values)
            
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
            'finite_ratio': 0.4,
            'diversity': 0.25,
            'stability': 0.25,
            'outlier_resistance': 0.1
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
        第1段階:基本対処(品質スコア > 0.6)
        軽量で高速な処理
        """
        cleaned = np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0)
        
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
        第2段階:フォールバック(品質スコア ≤ 0.6)
        ロバスト統計による処理
        """
        finite_mask = np.isfinite(values)
        if np.sum(finite_mask) < 3:
            return np.zeros_like(values)
        
        finite_values = values[finite_mask]
        
        try:
            median_val = np.median(finite_values)
            abs_deviations = np.abs(finite_values - median_val)
            mad_val = np.median(abs_deviations)
            
            if mad_val < 1e-10:
                mad_val = np.std(finite_values) * 0.6745
            
            robust_bounds = (median_val - 3 * mad_val, median_val + 3 * mad_val)
            
            result = np.copy(values)
            result = np.nan_to_num(result, nan=median_val, posinf=robust_bounds[1], neginf=robust_bounds[0])
            result = np.clip(result, robust_bounds[0], robust_bounds[1])
            
        except Exception:
            median_val = np.median(finite_values) if len(finite_values) > 0 else 0.0
            result = np.full_like(values, median_val)
        
        return result

# ===============================
# Block 4 Complete: DataEngine/QualityAssurance
# ===============================

# ===============================
# Block 5 Start: CalculationEngine クラス（簡略版）
# ===============================

class CalculationEngine:
    """
    計算核心クラス(60%) - Project Forge統合版(F16, F21のみ)
    【簡略版】カオス理論指標とLempel-Ziv複雑性のみ実装
    責務:
    - Polars Expressionによる高度な特徴量計算(90%のタスク)
    - .map_batches()経由のNumba JIT最適化UDF(10%のカスタム「アルファ」ロジック)
    - 【修正】物理的垂直分割:ディスク中間ファイルによるメモリ・スラッシング回避
    - Polars内部並列化による自動マルチスレッド実行
    - 2段階品質保証(基本/フォールバック)
    """
    
    def __init__(self, config: ProcessingConfig):
        self.config = config
        self.qa = QualityAssurance()
        self.memory_monitor = MemoryMonitor(config.memory_limit_gb)
        self.prefix = f"e{config.engine_id.replace('e', '')}_"
        
        # 【修正】ディスクベース垂直分割用の一時ディレクトリ
        self.temp_dir = Path(tempfile.mkdtemp(prefix=f"features_{config.engine_id}_"))
        logger.info(f"一時ディレクトリ作成: {self.temp_dir}")
    
    def _get_all_feature_expressions(self) -> Dict[str, pl.Expr]:
        """
        全特徴量の名前とPolars Expressionの対応辞書を返すファクトリメソッド。
        これにより、必要な特徴量だけを効率的に計算できるようになる。
        【特徴量ファクトリパターン】
        【重大修正】遅延束縛バグ修正:デフォルト引数による即時束縛
        【重大修正】map_batchesパターン:重量級UDF専用呼び出し
        """
        expressions = {}
        p = self.prefix
        
        # ====================================================================
        # F16: カオス理論指標
        # 【修正】rolling_map → map_batches、遅延束縛バグ修正
        # ====================================================================
        for window in self.config.window_sizes["chaos"]:
            # リアプノフ指数(重量UDF: map_batches使用)
            expressions[f"{p}lyapunov_exponent_{window}"] = pl.col("close").map_batches(
                lambda s, w=window: chaos_indicators_rolling_udf(s.to_numpy(), w, 0),
                return_dtype=pl.Float64
            ).alias(f"{p}lyapunov_exponent_{window}")
            
            expressions[f"{p}lyapunov_time_{window}"] = pl.col("close").map_batches(
                lambda s, w=window: chaos_indicators_rolling_udf(s.to_numpy(), w, 1),
                return_dtype=pl.Float64
            ).alias(f"{p}lyapunov_time_{window}")
            
            expressions[f"{p}correlation_dimension_{window}"] = pl.col("close").map_batches(
                lambda s, w=window: chaos_indicators_rolling_udf(s.to_numpy(), w, 2),
                return_dtype=pl.Float64
            ).alias(f"{p}correlation_dimension_{window}")
            
            expressions[f"{p}embedding_dimension_{window}"] = pl.col("close").map_batches(
                lambda s, w=window: chaos_indicators_rolling_udf(s.to_numpy(), w, 3),
                return_dtype=pl.Float64
            ).alias(f"{p}embedding_dimension_{window}")
        
        # ====================================================================
        # F21: Lempel-Ziv複雑性
        # 【修正】rolling_map → map_batches、遅延束縛バグ修正
        # ====================================================================
        for window in self.config.window_sizes["lz_complexity"]:
            # LZ複雑性包括計算(重量UDF: map_batches使用)
            expressions[f"{p}lz_complexity_{window}"] = pl.col("close").map_batches(
                lambda s, w=window: lempel_ziv_comprehensive_rolling_udf(s.to_numpy(), w, 0),
                return_dtype=pl.Float64
            ).alias(f"{p}lz_complexity_{window}")
            
            expressions[f"{p}lz_growth_rate_{window}"] = pl.col("close").map_batches(
                lambda s, w=window: lempel_ziv_comprehensive_rolling_udf(s.to_numpy(), w, 1),
                return_dtype=pl.Float64
            ).alias(f"{p}lz_growth_rate_{window}")
            
            expressions[f"{p}lz_dictionary_saturation_{window}"] = pl.col("close").map_batches(
                lambda s, w=window: lempel_ziv_comprehensive_rolling_udf(s.to_numpy(), w, 2),
                return_dtype=pl.Float64
            ).alias(f"{p}lz_dictionary_saturation_{window}")
            
            expressions[f"{p}lz_pattern_innovation_{window}"] = pl.col("close").map_batches(
                lambda s, w=window: lempel_ziv_comprehensive_rolling_udf(s.to_numpy(), w, 3),
                return_dtype=pl.Float64
            ).alias(f"{p}lz_pattern_innovation_{window}")
            
            expressions[f"{p}lz_complexity_volatility_{window}"] = pl.col("close").map_batches(
                lambda s, w=window: lempel_ziv_comprehensive_rolling_udf(s.to_numpy(), w, 4),
                return_dtype=pl.Float64
            ).alias(f"{p}lz_complexity_volatility_{window}")
        
        return expressions
    
    def get_feature_groups(self) -> Dict[str, Dict[str, pl.Expr]]:
        """特徴量グループ定義を外部から取得可能にする"""
        return self._create_vertical_slices()
    
    def calculate_one_group(self, lazy_frame: pl.LazyFrame, group_name: str, group_expressions: Dict[str, pl.Expr]) -> pl.LazyFrame:
        """
        単一グループの特徴量のみを計算(高速化修正版)
        グループ名に基づいて適切な特徴量計算メソッドを呼び出し、重複計算を回避
        """
        logger.info(f"グループ計算開始: {group_name}")
        
        # メモリ安全性チェック(必須)
        is_safe, message = self.memory_monitor.check_memory_safety()
        if not is_safe:
            raise MemoryError(f"メモリ不足のためグループ処理を中断: {message}")
        
        try:
            # グループ名に基づいて効率的な特徴量計算を実行
            if group_name == "chaos":
                group_result_lf = self._create_chaos_features(lazy_frame)
            elif group_name == "lempel_ziv":
                group_result_lf = self._create_lempel_ziv_features(lazy_frame)
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
            
            # 品質保証を適用(このグループのみ)
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
            # Inf値を除外してパーセンタイル計算(精度保持)
            col_for_quantile = pl.when(pl.col(col_name).is_infinite()).then(None).otherwise(pl.col(col_name))
            
            p01 = col_for_quantile.quantile(0.01, interpolation="linear")
            p99 = col_for_quantile.quantile(0.99, interpolation="linear")
            
            # Inf値を統計的に意味のある値(パーセンタイル境界値)で置換
            stabilized_col = (
                pl.when(pl.col(col_name) == float('inf'))
                .then(p99)
                .when(pl.col(col_name) == float('-inf'))
                .then(p01)
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
        
        # メモリ使用量を考慮したグルーピング(英語キー使用)
        slices = {}
        p = self.prefix
        
        # グループ1: カオス理論系(極重量)
        slices["chaos"] = {
            name: expr for name, expr in all_expressions.items() 
            if "lyapunov" in name or "correlation_dimension" in name or "embedding_dimension" in name
        }
        
        # グループ2: Lempel-Ziv複雑性系(重量)
        slices["lempel_ziv"] = {
            name: expr for name, expr in all_expressions.items() 
            if "lz_" in name
        }
        
        # 分割されなかった特徴量があれば警告
        total_assigned = sum(len(group) for group in slices.values())
        if total_assigned != len(all_expressions):
            logger.warning(f"未分割特徴量: {len(all_expressions) - total_assigned}個")
        
        return slices
    
    def _create_chaos_features(self, lazy_frame: pl.LazyFrame) -> pl.LazyFrame:
        """カオス理論系特徴量の計算(高速化対応)"""
        exprs = []
        p = self.prefix
        
        for window in self.config.window_sizes["chaos"]:
            exprs.append(pl.col("close").map_batches(
                lambda s, w=window: chaos_indicators_rolling_udf(s.to_numpy(), w, 0),
                return_dtype=pl.Float64
            ).alias(f"{p}lyapunov_exponent_{window}"))
            
            exprs.append(pl.col("close").map_batches(
                lambda s, w=window: chaos_indicators_rolling_udf(s.to_numpy(), w, 1),
                return_dtype=pl.Float64
            ).alias(f"{p}lyapunov_time_{window}"))
            
            exprs.append(pl.col("close").map_batches(
                lambda s, w=window: chaos_indicators_rolling_udf(s.to_numpy(), w, 2),
                return_dtype=pl.Float64
            ).alias(f"{p}correlation_dimension_{window}"))
            
            exprs.append(pl.col("close").map_batches(
                lambda s, w=window: chaos_indicators_rolling_udf(s.to_numpy(), w, 3),
                return_dtype=pl.Float64
            ).alias(f"{p}embedding_dimension_{window}"))
        
        return lazy_frame.with_columns(exprs)
    
    def _create_lempel_ziv_features(self, lazy_frame: pl.LazyFrame) -> pl.LazyFrame:
        """Lempel-Ziv複雑性系特徴量の計算(高速化対応)"""
        exprs = []
        p = self.prefix
        
        for window in self.config.window_sizes["lz_complexity"]:
            exprs.append(pl.col("close").map_batches(
                lambda s, w=window: lempel_ziv_comprehensive_rolling_udf(s.to_numpy(), w, 0),
                return_dtype=pl.Float64
            ).alias(f"{p}lz_complexity_{window}"))
            
            exprs.append(pl.col("close").map_batches(
                lambda s, w=window: lempel_ziv_comprehensive_rolling_udf(s.to_numpy(), w, 1),
                return_dtype=pl.Float64
            ).alias(f"{p}lz_growth_rate_{window}"))
            
            exprs.append(pl.col("close").map_batches(
                lambda s, w=window: lempel_ziv_comprehensive_rolling_udf(s.to_numpy(), w, 2),
                return_dtype=pl.Float64
            ).alias(f"{p}lz_dictionary_saturation_{window}"))
            
            exprs.append(pl.col("close").map_batches(
                lambda s, w=window: lempel_ziv_comprehensive_rolling_udf(s.to_numpy(), w, 3),
                return_dtype=pl.Float64
            ).alias(f"{p}lz_pattern_innovation_{window}"))
            
            exprs.append(pl.col("close").map_batches(
                lambda s, w=window: lempel_ziv_comprehensive_rolling_udf(s.to_numpy(), w, 4),
                return_dtype=pl.Float64
            ).alias(f"{p}lz_complexity_volatility_{window}"))
        
        return lazy_frame.with_columns(exprs)
    
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
            # Inf値を除外してパーセンタイル計算(精度保持)
            col_for_quantile = pl.when(pl.col(col_name).is_infinite()).then(None).otherwise(pl.col(col_name))
            
            p01 = col_for_quantile.quantile(0.01, interpolation="linear")
            p99 = col_for_quantile.quantile(0.99, interpolation="linear")
            
            # Inf値を統計的に意味のある値(パーセンタイル境界値)で置換
            stabilized_col = (
                pl.when(pl.col(col_name) == float('inf'))
                .then(p99)
                .when(pl.col(col_name) == float('-inf'))
                .then(p01)
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

# ===============================
# Block 5 Complete: CalculationEngine（簡略版）
# ===============================

# ===============================
# Block 6 Start: OutputEngine + 大規模データ処理アーキテクチャ
# ===============================

class OutputEngine:
    """
    出力管理クラス(10%) - Project Forge準拠  
    責務:
    - LazyFrame.sink_parquet()によるストリーミング出力
    - 必要に応じたPyArrowフォールバック(use_pyarrow=True)
    - timeframe別ファイル分離
    - NaN埋め統一処理
    - シンプルな進捗表示
    - 基本メタデータ記録
    """
    
    def __init__(self, config: ProcessingConfig):
        self.config = config
        self.memory_monitor = MemoryMonitor(config.memory_limit_gb)
        
        # 出力設定(Project Forge基準)
        self.output_config = {
            "compression": "snappy",
            "dtype": "float64",
            "timestamp_handling": "column"
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
        
        # スキーマを一度だけ取得(メモリ安全)
        schema = lazy_frame.collect_schema()
        all_columns = schema.names()
        
        # プレフィックスを持つ特徴量カラムを特定
        feature_columns = [col for col in all_columns if col.startswith(f"{self.engine_id}_")]
        
        # NaN埋め式生成
        fill_exprs = []
        for col in feature_columns:
            # NaNを0で埋める(金融データでは一般的)
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
            try:
                # まずストリーミングcollectを試行(新しいAPI使用)
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
            "engine_name": "Engine_2A_ComplexityTheory_Simplified",
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
# 大規模データ処理(Tick専用)アーキテクチャ:分割・重複・処理・結合
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
    FEATURES_ROOT = Path(config.output_path) / f"features_{config.engine_id}_{timeframe}"
    
    # 【安全なディレクトリ作成】
    # 出力ディレクトリの存在を確認し、なければ作成する。
    # 既存のファイルやディレクトリを削除する危険な処理は行わない。
    try:
        FEATURES_ROOT.mkdir(parents=True, exist_ok=True)
        logging.info(f"出力ディレクトリを確保しました: {FEATURES_ROOT}")
    except FileExistsError:
        logging.error(f"エラー: 出力パスがファイルとして存在しています。処理を中断します: {FEATURES_ROOT}")
        return # ここで処理を中断
    except Exception as e:
        logging.error(f"エラー: 出力ディレクトリの作成に失敗しました: {e}")
        return

    W_MAX = config.w_max
    
    calculation_engine = CalculationEngine(config)
    
    all_partitions = get_sorted_partitions(PARTITION_ROOT)

    if resume_date:
        import datetime
        all_days = [
            p for p in all_partitions
            if datetime.date(
                int(p.parent.parent.name.split('=')[1]),
                int(p.parent.name.split('=')[1]),
                int(p.name.split('=')[1])
            ) >= resume_date
        ]
        logging.info(f"再開日 {resume_date} に基づいてフィルタリングしました。")
    else:
        all_days = all_partitions
    
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
            # 前日のパーティション(オーバーラップ用)
            current_index_in_all = all_partitions.index(current_day_path)
            prev_day_path = all_partitions[current_index_in_all - 1] if current_index_in_all > 0 else None
            
            # オーバーラップを含む拡張データフレーム作成
            logging.info(f"データ読み込み開始: {day_name}")
            augmented_df, current_day_rows = create_augmented_frame(current_day_path, prev_day_path, W_MAX)
            logging.info(f"データ読み込み完了: 実データ{current_day_rows}行、総データ{augmented_df.height}行")
            
            logging.info(f"特徴量計算開始: {day_name}")
            
            # 一時ディレクトリ作成(日次処理用)
            temp_dir = Path(tempfile.mkdtemp(prefix=f"day_{i:04d}_{config.engine_id}_"))
            logging.info(f"日次一時ディレクトリ作成: {temp_dir}")
            
            temp_files = []
            
            # 特徴量グループを取得
            feature_groups = calculation_engine.get_feature_groups()
            logging.info(f"物理的垂直分割: {len(feature_groups)}グループで処理")
            
            # 各グループを順次処理(親方が工程管理)
            for group_idx, (group_name, group_expressions) in enumerate(feature_groups.items()):
                logging.info(f"グループ処理開始: {group_name} ({len(group_expressions)}個の特徴量)")
                
                # 1. 下請けに「このグループだけ計算しろ」と指示
                group_result_lf = calculation_engine.calculate_one_group(
                    augmented_df.lazy(), group_name, group_expressions
                )
                
                # 2. 親方が自らメモリに実現化(単一グループなので安全)
                group_result_df = group_result_lf.collect(streaming=True)
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
            
            # 4. 全グループ完了後、親方が最終組み立て(クリーン・オン・クリーン結合)
            logging.info("グループファイル結合開始(クリーン・オン・クリーン結合)...")
            
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

# ===============================
# Block 6 Complete: OutputEngine + 大規模データ処理アーキテクチャ
# ===============================

# ===============================
# Block 7 Start: メイン実行部（最終）
# ===============================

# 通常timeframe処理
def process_single_timeframe(config: ProcessingConfig, timeframe: str):
    """単一の通常時間足を処理する(修正版ロジック)"""
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

# インタラクティブモード(Project Forge準拠)
def get_user_confirmation(config: ProcessingConfig) -> bool:
    """ユーザー確認 - Project Forge理念表示"""
    print("\n" + "="*60)
    print(f"Engine {config.engine_id.upper()} - {config.engine_name}")
    print("="*60)
    print("🎯 Project Forge - 軍資金増大プロジェクト")
    print("🚀 最終目標: Project Chimera開発・完成のための資金調達")
    print("💻 探索対象: マーケットの亡霊(統計的に有意で非ランダムな微細パターン)")
    print("🏅 思想的継承: ジム・シモンズ(ルネサンス・テクノロジーズ)")
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
    
    print("\n処理内容【簡略版】:")
    print("  - F16: カオス理論指標 (リアプノフ指数・相関次元)")
    print("  - F21: Lempel-Ziv複雑性 (パターン新規性)")
    print("  - 2段階品質保証システム")
    print("  - 【修正】物理的垂直分割(ディスクベース中間ファイル)")
    print("  - 【修正】map_batches + prange並列化による重量級UDF最適化")
    print("  - 【修正】遅延束縛バグ完全修正")
    print("  - Polars LazyFrame + Numba JITハイブリッド最適化")
    
    response = input("\n処理を開始しますか? (y/n): ")
    return response.lower() == 'y'

def select_timeframes(config: ProcessingConfig) -> List[str]:
    """タイムフレーム選択(完全同一実装)"""
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
    """メイン実行関数 - Project Forge統合版（簡略版）"""
    print("\n" + "="*70)
    print(f"  Engine 2A - Complexity Theory Features (簡略版 - F16, F21のみ)")
    print("  Project Forge - 軍資金増大プロジェクト  ")
    print("="*70)
    print("🎯 目標: XAU/USD市場の統計的パターン抽出")
    print("🤖 AI頭脳による普遍的法則発見")
    print("💰 Project Chimera開発資金調達")
    print("🔧 【簡略版】F16(カオス理論) + F21(Lempel-Ziv複雑性)のみ")
    print("🔧 【修正】重量級UDF呼び出しパターンの設計規律遵守")
    print("🔧 【修正】遅延束縛バグの完全修正")
    print("🔧 【修正】map_batches + prange並列化の徹底")
    print("="*70)
    
    config = ProcessingConfig()
    
    if not config.validate():
        return 1
    
    data_engine = DataEngine(config)
    
    if not data_engine.validate_data_source():
        return 1
    
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
                resume_date = datetime.datetime.strptime(date_str, "%Y-%m-%d").date()
                print(f"{resume_date} から処理を再開します。")
                break
            except ValueError:
                print("エラー: 日付の形式が正しくありません。YYYY-MM-DD形式で入力してください。")
    else:
        print("新規に処理を開始します。")
    
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
    print("  1. テストモード(少量データで動作確認)")
    print("  2. 本格モード(全データ処理)")
    
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

# ===============================
# Block 7 Complete: メイン実行部
# ===============================
# 
# 全7ブロック完成: engine_2_complexity_theory_simplified.py
#
# 【簡略版完成】
# 削除された特徴量:
# - F5: MFDFA（マルチフラクタル解析）
# - F15: コルモゴロフ複雑性
#
# 実装完了した特徴量:
# - F16: カオス理論指標（リアプノフ指数・相関次元）
# - F21: Lempel-Ziv複雑性
#
# アーキテクチャ:
# - DataEngine: 参照スクリプト完全踏襲
# - CalculationEngine: F16とF21のみ専用実装（修正版）
# - OutputEngine: 参照スクリプト完全踏襲
# - 物理的垂直分割: ディスクベース中間ファイル方式
# - Numba UDF: 全て@nb.njit(parallel=True) + nb.prange並列化
#
# ===============================