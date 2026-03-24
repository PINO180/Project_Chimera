# ブロック1の開始
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Engine 1C (ATR-Only for Tick): テクニカル指標・トレンド分析特徴量エンジン
Technical Indicators and Trend Analysis Features Engine (ATR-Only for Tick)

対象: XAU/USD (Gold vs US Dollar)
入力: /workspace/data/XAUUSD/stratum_1_base/master_tick_exness_raw.parquet (単一ファイル)
出力: features_e1c_tick_atr_only_tick.parquet (単一ファイル)
目的: create_proxy_labels.py で使用する tickデータのATRを計算する。

ブロック1: インポート・設定・ヘルパー関数
"""

# ===== ブロック1開始 =====

# 標準ライブラリ
import os
import sys
import time
import warnings
import json
import logging
import math
import sys
from pathlib import Path
import pyarrow.parquet as pq

import tempfile  # 一時ディレクトリ作成のため
import shutil  # 一時ディレクトリ削除のため

import pyarrow as pa

# プロジェクトルートをsys.pathに追加
try:
    project_root = Path(__file__).resolve().parents[1]
    if str(project_root) not in sys.path:
        sys.path.append(str(project_root))
    import blueprint as config
except (ImportError, FileNotFoundError, NameError):
    # スクリプトが直接実行される場合や、blueprintが見つからない場合のフォールバック
    print("Warning: blueprint.py not found. Using fallback paths.")

    # フォールバック用の最小限のconfigモック
    class FallbackConfig:
        S1_BASE_MULTITIMEFRAME = Path("/workspace/data/XAUUSD/stratum_1_base")
        S1_RAW_TICK_PARTITIONED = Path(
            "/workspace/data/XAUUSD/stratum_1_base/master_tick_partitioned"
        )
        S2_FEATURES = Path("/workspace/data/XAUUSD/stratum_2_features")

    config = FallbackConfig()

from pathlib import Path

# ブロック1の修正後全文 (import typing 部分)
from typing import Dict, List, Optional, Tuple, Union, Any, Iterator  # Iterator を追加
from dataclasses import dataclass, field

# 外部ライブラリ（バージョン固定）
import numpy as np
import polars as pl
import numba as nb
from numba import guvectorize, float64, int64
from scipy import stats
from scipy.stats import jarque_bera, anderson, shapiro
import psutil

# ログ設定
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("engine_1c_atr_only.log", encoding="utf-8"),
    ],
)
logger = logging.getLogger(__name__)

# 警告フィルタ
warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=UserWarning, module="polars")


# ========================================
# 設定クラス
# ========================================
def get_default_timeframes() -> List[str]:
    """
    【変更点】
    このエンジンはtickデータのみを処理対象とする。
    """
    return ["tick"]


def get_default_window_sizes() -> Dict[str, List[int]]:
    """
    【構造維持】
    設定は残すが、ATR以外は使用されない。
    """
    return {
        "rsi": [14, 21, 30, 50],
        "atr": [13, 21, 34, 55],  # この設定のみが使用される
        "adx": [13, 21, 34],
        "hma": [21, 34, 55],
        "kama": [21, 34],
        "general": [10, 20, 50, 100],
    }


@dataclass
class ProcessingConfig:
    """処理設定"""

    # 【変更点】入力パスを単一のraw tickファイルに指定
    input_base_path: str = (
        "/workspace/data/XAUUSD/stratum_1_base/master_tick_exness_raw.parquet"
    )

    # 【構造維持】使用しないが、元のスクリプト構造維持のため残す
    partitioned_tick_path: str = str(config.S1_RAW_TICK_PARTITIONED)

    # 【変更点】出力先はS2のルート
    output_base_path: str = str(config.S2_FEATURES)

    # 【変更点】エンジンIDを固有のものに変更
    engine_id: str = "e1c_tick_atr_only"
    engine_name: str = "Engine_1C_ATR_Only_for_Tick"

    # 並列処理
    max_threads: int = 4

    # メモリ制限
    memory_limit_gb: float = 55.0
    memory_warning_gb: float = 50.0

    # 【変更点】デフォルトタイムフレームを "tick" のみに
    timeframes: List[str] = field(default_factory=get_default_timeframes)
    window_sizes: Dict[str, List[int]] = field(default_factory=get_default_window_sizes)

    # 処理モード
    test_mode: bool = False
    test_rows: int = 10000

    # システムハイパーパラメータとしてW_maxを定義 (このスクリプトでは使用しないが構造維持)
    w_max: int = 200

    def validate(self) -> bool:
        """設定検証"""
        output_path_obj = Path(self.output_base_path)
        if not output_path_obj.exists():
            output_path_obj.mkdir(parents=True, exist_ok=True)
            logger.info(f"出力ディレクトリを作成: {output_path_obj}")

        # 【変更点】入力がパーティションではなく単一ファイルであることを確認
        input_file_obj = Path(self.input_base_path)
        if not input_file_obj.exists():
            logger.error(f"入力ファイルが見つかりません: {self.input_base_path}")
            return False

        return True


# ========================================
# メモリ監視
# ========================================


class MemoryMonitor:
    """メモリ使用量監視 (構造維持のため変更なし)"""

    def __init__(self, limit_gb: float = 55.0, warning_gb: float = 50.0):
        self.limit_gb = limit_gb
        self.warning_gb = warning_gb
        self.process = psutil.Process()

    def get_memory_usage(self) -> Tuple[float, float]:
        memory_info = self.process.memory_info()
        used_gb = memory_info.rss / (1024**3)
        percent = psutil.virtual_memory().percent
        return used_gb, percent

    def check_memory_safety(self) -> Tuple[bool, str]:
        used_gb, percent = self.get_memory_usage()

        if used_gb > self.limit_gb:
            message = f"メモリ使用量が制限値を超過: {used_gb:.2f}GB / {self.limit_gb:.2f}GB ({percent:.1f}%)"
            logger.error(message)
            return False, message

        if used_gb > self.warning_gb:
            message = f"警告: メモリ使用量が高い: {used_gb:.2f}GB / {self.limit_gb:.2f}GB ({percent:.1f}%)"
            logger.warning(message)
            return True, message

        message = (
            f"メモリ使用量: {used_gb:.2f}GB / {self.limit_gb:.2f}GB ({percent:.1f}%)"
        )
        return True, message


# ========================================
# 品質保証システム
# ========================================


class QualityAssurance:
    """品質保証システム (構造維持のため変更なし)"""

    @staticmethod
    def calculate_quality_score(values: np.ndarray) -> float:
        nan_inf_ratio = (np.isnan(values).sum() + np.isinf(values).sum()) / len(values)
        unique_ratio = len(np.unique(values)) / len(values)
        unique_ratio_norm = min(unique_ratio, 0.1) / 0.1
        finite_values = values[np.isfinite(values)]
        if len(finite_values) == 0:
            range_norm = 0.0
        else:
            p1, p99 = np.percentile(finite_values, [1, 99])
            range_span = p99 - p1
            range_norm = 1.0 if range_span > 0 else 0.0
        score = (1 - nan_inf_ratio) * (0.5 * unique_ratio_norm + 0.5 * range_norm)
        return np.clip(score, 0.0, 1.0)

    @staticmethod
    def basic_stabilization(values: np.ndarray) -> np.ndarray:
        cleaned = np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0)
        finite_values = cleaned[np.isfinite(cleaned)]
        if len(finite_values) == 0:
            return np.zeros_like(values)
        p1, p99 = np.percentile(finite_values, [1, 99])
        result = np.clip(cleaned, p1, p99)
        return result

    @staticmethod
    def robust_stabilization(values: np.ndarray) -> np.ndarray:
        finite_values = values[np.isfinite(values)]
        if len(finite_values) == 0:
            return np.zeros_like(values)
        from scipy.stats import median_abs_deviation

        median_val = np.median(finite_values)
        mad_val = median_abs_deviation(finite_values)
        if mad_val == 0:
            mad_val = 1.0
        robust_bounds = (median_val - 3 * mad_val, median_val + 3 * mad_val)
        result = np.clip(values, *robust_bounds)
        result = np.nan_to_num(result, nan=median_val)
        return result


# ========================================
# Numba UDF関数（モジュールレベル）
# ========================================


# 【構造維持】RSI (使用されない)
@nb.guvectorize(
    ["void(float64[:], int64, float64[:])"], "(n),()->(n)", nopython=True, cache=True
)
def calculate_rsi_numba(prices, period, out):
    n = len(prices)
    for i in range(n):
        if i < period:
            out[i] = np.nan
        else:
            gains = 0.0
            losses = 0.0
            for j in range(i - period + 1, i + 1):
                diff = prices[j] - prices[j - 1]
                if diff > 0:
                    gains += diff
                else:
                    losses += abs(diff)
            if gains + losses == 0:
                out[i] = 50.0
            else:
                avg_gain = gains / period
                avg_loss = losses / period
                if avg_loss == 0:
                    out[i] = 100.0
                else:
                    rs = avg_gain / avg_loss
                    out[i] = 100.0 - (100.0 / (1.0 + rs))


# 修正後全文 (calculate_atr_numba)
@nb.guvectorize(
    ["void(float64[:], float64[:], float64[:], int64, float64[:])"],
    "(n),(n),(n),()->(n)",
    nopython=True,
    cache=True,
)
def calculate_atr_numba(ask, bid, mid_price, period, out):
    """
    ATR計算 (Numba最適化版) - Tickデータ用に ask/bid/mid_price を使用
    """
    n = len(ask)
    tr = np.zeros(n)
    for i in range(n):
        if i == 0:
            # 最初のデータポイントでは、ask-bid の差のみを使用
            tr[i] = ask[i] - bid[i]
            # 負の値にならないように
            if tr[i] < 0:
                tr[i] = 0.0
        else:
            # ask-bid
            h_l = ask[i] - bid[i]
            # ask - previous mid_price
            h_pc = abs(ask[i] - mid_price[i - 1])
            # bid - previous mid_price
            l_pc = abs(bid[i] - mid_price[i - 1])
            # True Range は3つの最大値
            tr[i] = max(h_l, h_pc, l_pc)
            # 負の値にならないように (ask < bid の場合など)
            if tr[i] < 0:
                tr[i] = 0.0

    # ATR計算 (Simple Moving Average of TR)
    for i in range(n):
        if i < period - 1:  # 最初の period-1 個はNaN
            out[i] = np.nan
        else:
            # Polarsのrolling_mean(period)と厳密に合わせるため、
            # 現在のインデックス i を含む過去 period 個の平均を取る
            sum_tr = 0.0
            start_idx = i - period + 1
            # 範囲チェックを追加 (start_idxが負にならないように)
            if start_idx < 0:
                start_idx = 0

            count = 0
            for j in range(start_idx, i + 1):
                # NaNが含まれていないかチェック (より安全に)
                if not np.isnan(tr[j]):
                    sum_tr += tr[j]
                    count += 1

            # period個のデータが揃っていない場合はNaNのままにするか、
            # または利用可能なデータ数で割るかを選択できる。
            # ここではPolarsの挙動に合わせ、period個未満ならNaNとする想定。
            # しかし、Numbaではrolling_meanのmin_periodsのような機能はないため、
            # ここでは単純にperiodで割る。NaNの扱いはPolars側で調整されることを期待。
            # 安全のため、countが0の場合は0を返す。
            if count == period:  # period 個のデータがある場合のみ計算
                out[i] = sum_tr / period
            elif count > 0:
                # 部分的なデータで計算する場合 (Polarsの挙動とは異なる可能性あり)
                # out[i] = sum_tr / count
                # Polarsのrolling().mean()は通常、ウィンドウが満たない場合はnullを返すため、NaNが適切
                out[i] = np.nan
            else:
                out[i] = np.nan  # 計算不能な場合


# 【構造維持】ADX (使用されない)
@nb.guvectorize(
    ["void(float64[:], float64[:], float64[:], int64, float64[:])"],
    "(n),(n),(n),()->(n)",
    nopython=True,
    cache=True,
)
def calculate_adx_numba(high, low, close, period, out):
    n = len(high)
    tr = np.zeros(n)
    dm_plus = np.zeros(n)
    dm_minus = np.zeros(n)
    for i in range(n):
        if i == 0:
            tr[i] = high[i] - low[i]
        else:
            h_l = high[i] - low[i]
            h_pc = abs(high[i] - close[i - 1])
            l_pc = abs(low[i] - close[i - 1])
            tr[i] = max(h_l, h_pc, l_pc)
            up_move = high[i] - high[i - 1]
            down_move = low[i - 1] - low[i]
            if up_move > down_move and up_move > 0:
                dm_plus[i] = up_move
            else:
                dm_plus[i] = 0.0
            if down_move > up_move and down_move > 0:
                dm_minus[i] = down_move
            else:
                dm_minus[i] = 0.0
    di_plus = np.zeros(n)
    di_minus = np.zeros(n)
    for i in range(n):
        if i < period:
            di_plus[i] = np.nan
            di_minus[i] = np.nan
            out[i] = np.nan
        else:
            atr_val = 0.0
            dm_plus_sum = 0.0
            dm_minus_sum = 0.0
            for j in range(i - period + 1, i + 1):
                atr_val += tr[j]
                dm_plus_sum += dm_plus[j]
                dm_minus_sum += dm_minus[j]
            atr_val = atr_val / period
            if atr_val > 0:
                di_plus[i] = (dm_plus_sum / period) / atr_val * 100
                di_minus[i] = (dm_minus_sum / period) / atr_val * 100
            else:
                di_plus[i] = 0.0
                di_minus[i] = 0.0
    for i in range(n):
        if i < period * 2:
            out[i] = np.nan
        else:
            dx_sum = 0.0
            for j in range(i - period + 1, i + 1):
                di_sum = di_plus[j] + di_minus[j]
                if di_sum > 0:
                    dx = abs(di_plus[j] - di_minus[j]) / di_sum * 100
                else:
                    dx = 0.0
                dx_sum += dx
            out[i] = dx_sum / period


# 【構造維持】DI+ (使用されない)
@nb.guvectorize(
    ["void(float64[:], float64[:], float64[:], int64, float64[:])"],
    "(n),(n),(n),()->(n)",
    nopython=True,
    cache=True,
)
def calculate_di_plus_numba(high, low, close, period, out):
    n = len(high)
    tr = np.zeros(n)
    dm_plus = np.zeros(n)
    for i in range(n):
        if i == 0:
            tr[i] = high[i] - low[i]
        else:
            h_l = high[i] - low[i]
            h_pc = abs(high[i] - close[i - 1])
            l_pc = abs(low[i] - close[i - 1])
            tr[i] = max(h_l, h_pc, l_pc)
            up_move = high[i] - high[i - 1]
            down_move = low[i - 1] - low[i]
            if up_move > down_move and up_move > 0:
                dm_plus[i] = up_move
            else:
                dm_plus[i] = 0.0
    for i in range(n):
        if i < period:
            out[i] = np.nan
        else:
            atr_val = 0.0
            dm_plus_sum = 0.0
            for j in range(i - period + 1, i + 1):
                atr_val += tr[j]
                dm_plus_sum += dm_plus[j]
            atr_val = atr_val / period
            if atr_val > 0:
                out[i] = (dm_plus_sum / period) / atr_val * 100
            else:
                out[i] = 0.0


# 【構造維持】DI- (使用されない)
@nb.guvectorize(
    ["void(float64[:], float64[:], float64[:], int64, float64[:])"],
    "(n),(n),(n),()->(n)",
    nopython=True,
    cache=True,
)
def calculate_di_minus_numba(high, low, close, period, out):
    n = len(high)
    tr = np.zeros(n)
    dm_minus = np.zeros(n)
    for i in range(n):
        if i == 0:
            tr[i] = high[i] - low[i]
        else:
            h_l = high[i] - low[i]
            h_pc = abs(high[i] - close[i - 1])
            l_pc = abs(low[i] - close[i - 1])
            tr[i] = max(h_l, h_pc, l_pc)
            up_move = high[i] - high[i - 1]
            down_move = low[i - 1] - low[i]
            if down_move > up_move and down_move > 0:
                dm_minus[i] = down_move
            else:
                dm_minus[i] = 0.0
    for i in range(n):
        if i < period:
            out[i] = np.nan
        else:
            atr_val = 0.0
            dm_minus_sum = 0.0
            for j in range(i - period + 1, i + 1):
                atr_val += tr[j]
                dm_minus_sum += dm_minus[j]
            atr_val = atr_val / period
            if atr_val > 0:
                out[i] = (dm_minus_sum / period) / atr_val * 100
            else:
                out[i] = 0.0


# ===== ブロック1完了 =====
# ブロック2の開始
# ===== ブロック2開始 =====

# ========================================
# Numba UDF関数（続き）
# (構造維持のため、呼び出されない関数も残す)
# ========================================


@nb.guvectorize(
    ["void(float64[:], int64, float64[:])"], "(n),()->(n)", nopython=True, cache=True
)
def calculate_hma_numba(prices, period, out):
    """Hull Moving Average計算 (Numba最適化版)"""
    n = len(prices)
    half_period = period // 2
    sqrt_period = int(np.sqrt(period))

    def wma(data, start, length):
        if start + length > len(data):
            return np.nan
        weight_sum = 0.0
        value_sum = 0.0
        for i in range(length):
            weight = length - i
            value_sum += data[start + i] * weight
            weight_sum += weight
        return value_sum / weight_sum if weight_sum > 0 else np.nan

    wma_half = np.zeros(n)
    wma_full = np.zeros(n)
    raw_hma = np.zeros(n)

    for i in range(n):
        if i >= half_period - 1:
            wma_half[i] = wma(prices, i - half_period + 1, half_period)
        else:
            wma_half[i] = np.nan
        if i >= period - 1:
            wma_full[i] = wma(prices, i - period + 1, period)
        else:
            wma_full[i] = np.nan
        if not np.isnan(wma_half[i]) and not np.isnan(wma_full[i]):
            raw_hma[i] = 2 * wma_half[i] - wma_full[i]
        else:
            raw_hma[i] = np.nan

    for i in range(n):
        if i >= sqrt_period - 1:
            out[i] = wma(raw_hma, i - sqrt_period + 1, sqrt_period)
        else:
            out[i] = np.nan


@nb.guvectorize(
    ["void(float64[:], int64, float64[:])"], "(n),()->(n)", nopython=True, cache=True
)
def calculate_kama_numba(prices, period, out):
    """Kaufman Adaptive Moving Average計算 (Numba最適化版)"""
    n = len(prices)
    fast_ema = 2
    slow_ema = 30
    for i in range(n):
        if i < period:
            out[i] = np.nan
        else:
            direction = abs(prices[i] - prices[i - period])
            volatility = 0.0
            for j in range(i - period + 1, i + 1):
                volatility += abs(prices[j] - prices[j - 1])
            if volatility == 0:
                er = 0.0
            else:
                er = direction / volatility
            fast_sc = 2.0 / (fast_ema + 1.0)
            slow_sc = 2.0 / (slow_ema + 1.0)
            sc = (er * (fast_sc - slow_sc) + slow_sc) ** 2
            if i == period:
                out[i] = prices[i]
            else:
                if not np.isnan(out[i - 1]):
                    out[i] = out[i - 1] + sc * (prices[i] - out[i - 1])
                else:
                    out[i] = prices[i]


@nb.guvectorize(
    ["void(float64[:], float64[:], float64[:], int64, int64, int64, float64[:])"],
    "(n),(n),(n),(),(),()->(n)",
    nopython=True,
    cache=True,
)
def calculate_stochastic_numba(high, low, close, k_period, d_period, slow_period, out):
    """Stochastic Oscillator計算 (Numba最適化版)"""
    n = len(high)
    k_values = np.zeros(n)
    for i in range(n):
        if i < k_period - 1:
            k_values[i] = np.nan
        else:
            highest = high[i]
            lowest = low[i]
            for j in range(i - k_period + 1, i):
                if high[j] > highest:
                    highest = high[j]
                if low[j] < lowest:
                    lowest = low[j]
            if highest - lowest > 0:
                k_values[i] = 100 * (close[i] - lowest) / (highest - lowest)
            else:
                k_values[i] = 50.0
    for i in range(n):
        if i < k_period + d_period - 2:
            out[i] = np.nan
        else:
            sum_k = 0.0
            count = 0
            for j in range(i - d_period + 1, i + 1):
                if not np.isnan(k_values[j]):
                    sum_k += k_values[j]
                    count += 1
            if count > 0:
                out[i] = sum_k / count
            else:
                out[i] = np.nan


@nb.guvectorize(
    ["void(float64[:], float64[:], float64[:], int64, float64[:])"],
    "(n),(n),(n),()->(n)",
    nopython=True,
    cache=True,
)
def calculate_williams_r_numba(high, low, close, period, out):
    """Williams %R計算 (Numba最適化版)"""
    n = len(high)
    for i in range(n):
        if i < period - 1:
            out[i] = np.nan
        else:
            highest = high[i]
            lowest = low[i]
            for j in range(i - period + 1, i):
                if high[j] > highest:
                    highest = high[j]
                if low[j] < lowest:
                    lowest = low[j]
            if highest - lowest > 0:
                out[i] = -100 * (highest - close[i]) / (highest - lowest)
            else:
                out[i] = -50.0


@nb.guvectorize(
    ["void(float64[:], int64, float64[:])"], "(n),()->(n)", nopython=True, cache=True
)
def calculate_trix_numba(prices, period, out):
    """TRIX指標計算 (Numba最適化版)"""
    n = len(prices)
    ema1 = np.zeros(n)
    ema2 = np.zeros(n)
    ema3 = np.zeros(n)
    alpha = 2.0 / (period + 1.0)
    for i in range(n):
        if i == 0:
            ema1[i] = prices[i]
        else:
            ema1[i] = alpha * prices[i] + (1 - alpha) * ema1[i - 1]
    for i in range(n):
        if i == 0:
            ema2[i] = ema1[i]
        else:
            ema2[i] = alpha * ema1[i] + (1 - alpha) * ema2[i - 1]
    for i in range(n):
        if i == 0:
            ema3[i] = ema2[i]
        else:
            ema3[i] = alpha * ema2[i] + (1 - alpha) * ema3[i - 1]
    for i in range(n):
        if i < period * 3:
            out[i] = np.nan
        elif ema3[i - 1] == 0:
            out[i] = 0.0
        else:
            out[i] = 10000 * (ema3[i] - ema3[i - 1]) / ema3[i - 1]


@nb.guvectorize(
    ["void(float64[:], float64[:], float64[:], float64[:], float64[:])"],
    "(n),(n),(n),(n)->(n)",
    nopython=True,
    cache=True,
)
def calculate_ultimate_oscillator_numba(high, low, close, volume, out):
    """Ultimate Oscillator計算 (Numba最適化版)"""
    n = len(high)
    periods = [7, 14, 28]
    weights = [4.0, 2.0, 1.0]
    bp = np.zeros(n)
    tr = np.zeros(n)
    for i in range(n):
        if i == 0:
            bp[i] = close[i] - low[i]
            tr[i] = high[i] - low[i]
        else:
            bp[i] = close[i] - min(low[i], close[i - 1])
            h_l = high[i] - low[i]
            h_pc = abs(high[i] - close[i - 1])
            l_pc = abs(low[i] - close[i - 1])
            tr[i] = max(h_l, h_pc, l_pc)
    for i in range(n):
        if i < max(periods):
            out[i] = np.nan
        else:
            weighted_sum = 0.0
            weight_total = sum(weights)
            for j, period in enumerate(periods):
                bp_sum = 0.0
                tr_sum = 0.0
                for k in range(i - period + 1, i + 1):
                    bp_sum += bp[k]
                    tr_sum += tr[k]
                if tr_sum > 0:
                    avg = bp_sum / tr_sum
                else:
                    avg = 0.0
                weighted_sum += avg * weights[j]
            out[i] = 100 * weighted_sum / weight_total


@nb.guvectorize(
    ["void(float64[:], int64, float64[:])"], "(n),()->(n)", nopython=True, cache=True
)
def calculate_aroon_up_numba(high, period, out):
    """Aroon Up計算 (Numba最適化版)"""
    n = len(high)
    for i in range(n):
        if i < period:
            out[i] = np.nan
        else:
            highest_idx = i
            highest_val = high[i]
            for j in range(i - period + 1, i):
                if high[j] > highest_val:
                    highest_val = high[j]
                    highest_idx = j
            periods_since = i - highest_idx
            out[i] = 100.0 * (period - periods_since) / period


@nb.guvectorize(
    ["void(float64[:], int64, float64[:])"], "(n),()->(n)", nopython=True, cache=True
)
def calculate_aroon_down_numba(low, period, out):
    """Aroon Down計算 (Numba最適化版)"""
    n = len(low)
    for i in range(n):
        if i < period:
            out[i] = np.nan
        else:
            lowest_idx = i
            lowest_val = low[i]
            for j in range(i - period + 1, i):
                if low[j] < lowest_val:
                    lowest_val = low[j]
                    lowest_idx = j
            periods_since = i - lowest_idx
            out[i] = 100.0 * (period - periods_since) / period


@nb.guvectorize(
    ["void(float64[:], int64, float64[:])"], "(n),()->(n)", nopython=True, cache=True
)
def calculate_tsi_numba(prices, period, out):
    """True Strength Index計算 (Numba最適化版)"""
    n = len(prices)
    long_period = period
    short_period = period // 2
    momentum = np.zeros(n)
    for i in range(1, n):
        momentum[i] = prices[i] - prices[i - 1]
    momentum[0] = 0.0
    alpha_long = 2.0 / (long_period + 1.0)
    alpha_short = 2.0 / (short_period + 1.0)
    ema1_mom = np.zeros(n)
    ema1_abs = np.zeros(n)
    for i in range(n):
        if i == 0:
            ema1_mom[i] = momentum[i]
            ema1_abs[i] = abs(momentum[i])
        else:
            ema1_mom[i] = alpha_long * momentum[i] + (1 - alpha_long) * ema1_mom[i - 1]
            ema1_abs[i] = (
                alpha_long * abs(momentum[i]) + (1 - alpha_long) * ema1_abs[i - 1]
            )
    ema2_mom = np.zeros(n)
    ema2_abs = np.zeros(n)
    for i in range(n):
        if i == 0:
            ema2_mom[i] = ema1_mom[i]
            ema2_abs[i] = ema1_abs[i]
        else:
            ema2_mom[i] = (
                alpha_short * ema1_mom[i] + (1 - alpha_short) * ema2_mom[i - 1]
            )
            ema2_abs[i] = (
                alpha_short * ema1_abs[i] + (1 - alpha_short) * ema2_abs[i - 1]
            )
    for i in range(n):
        if i < long_period + short_period:
            out[i] = np.nan
        elif ema2_abs[i] == 0:
            out[i] = 0.0
        else:
            out[i] = 100 * ema2_mom[i] / ema2_abs[i]


# ===== ブロック2完了 =====
# ブロック3の開始
# ===== ブロック3開始 =====

# ========================================
# DataEngine - データ基盤クラス
# ========================================


# 修正後全文 (DataEngine クラス)


class DataEngine:
    """データ基盤エンジン (単一ファイルTick入力 + チャンク処理対応)"""

    def __init__(self, config: ProcessingConfig):
        """初期化"""
        self.config = config
        self.base_path = Path(config.input_base_path)
        self.arrow_file: Optional[pq.ParquetFile] = None
        self.total_rows: Optional[int] = None
        logger.info(f"DataEngine初期化 (チャンクモード対応): {self.base_path}")

    def validate_data_source(self) -> bool:
        """データソース検証 (単一ファイル版)"""
        if not self.base_path.exists():
            logger.error(f"入力ファイルが存在しません: {self.base_path}")
            return False
        try:
            self.arrow_file = pq.ParquetFile(self.base_path)
            self.total_rows = self.arrow_file.metadata.num_rows
            logger.info(
                f"入力ファイルを検証 (pyarrow): {self.base_path}, 総行数: {self.total_rows}"
            )
            return True
        except Exception as e:
            logger.error(f"入力ファイルを開けません (pyarrow): {e}")
            return False

    def iter_chunk_dataframes(
        self, chunk_size: int, overlap_size: int
    ) -> Iterator[Tuple[pl.DataFrame, int, int]]:
        """
        Parquetファイルをチャンク単位で読み込み、オーバーラップを付けてDataFrameを生成するイテレータ。

        Args:
            chunk_size (int): 各チャンクの基本行数。
            overlap_size (int): 前のチャンクと重複させる行数。

        Yields:
            Tuple[pl.DataFrame, int, int]: オーバーラップを含むチャンクDataFrame,
                                            現在のチャンクの開始行インデックス (0始まり),
                                            現在のチャンクの終了行インデックス (含まず)
        """
        if self.arrow_file is None or self.total_rows is None:
            raise RuntimeError("Data source not validated or file not opened.")

        num_row_groups = self.arrow_file.num_row_groups
        current_pos = 0
        previous_chunk_overlap_data: Optional[pl.DataFrame] = None

        logger.info(
            f"チャンク読み込み開始: chunk_size={chunk_size}, overlap_size={overlap_size}"
        )

        # PyArrowの iter_batches を使う方法が効率的
        batch_iterator = self.arrow_file.iter_batches(batch_size=chunk_size)

        chunk_start_index = 0
        while True:
            try:
                # pyarrowのバッチを読み込む
                batch = next(batch_iterator)
                if batch is None or batch.num_rows == 0:
                    break  # データ終了

                # Polars DataFrameに変換
                current_chunk_df = pl.from_arrow(batch)
                actual_chunk_size = current_chunk_df.height
                chunk_end_index = chunk_start_index + actual_chunk_size

                # オーバーラップデータを結合
                if previous_chunk_overlap_data is not None:
                    # logger.debug(f"結合: 前のオーバーラップ {previous_chunk_overlap_data.height}行 + 現在チャンク {current_chunk_df.height}行")
                    chunk_with_overlap_df = pl.concat(
                        [previous_chunk_overlap_data, current_chunk_df], how="vertical"
                    )
                else:
                    chunk_with_overlap_df = current_chunk_df

                # 次のチャンクのためのオーバーラップデータを準備
                if overlap_size > 0 and chunk_end_index < self.total_rows:
                    # 現在のチャンク *本来のデータ* の末尾からオーバーラップ分を取得
                    # logger.debug(f"次のオーバーラップを準備: 現在チャンク末尾 {actual_chunk_size}行から {overlap_size}行を取得")
                    previous_chunk_overlap_data = current_chunk_df.tail(overlap_size)
                    # logger.debug(f"準備されたオーバーラップデータ: {previous_chunk_overlap_data.height}行")

                # timeframe列の手動復元 (元のデータに存在しない場合)
                if "timeframe" not in chunk_with_overlap_df.columns:
                    chunk_with_overlap_df = chunk_with_overlap_df.with_columns(
                        pl.lit("tick").alias("timeframe")
                    )

                logger.info(
                    f"チャンク生成: 行 {chunk_start_index} - {chunk_end_index - 1} (含オーバーラップ: {chunk_with_overlap_df.height}行)"
                )
                yield chunk_with_overlap_df, chunk_start_index, chunk_end_index

                chunk_start_index = chunk_end_index  # 次の開始位置を更新

            except StopIteration:
                logger.info("全チャンクの読み込み完了")
                break
            except Exception as e:
                logger.error(f"チャンク読み込み中にエラー発生: {e}", exc_info=True)
                raise

    def create_lazy_frame(self, timeframe: str) -> pl.LazyFrame:
        """【注意】このメソッドはチャンクモードでは直接使用しない"""
        logger.warning(
            "create_lazy_frameはチャンクモードでは直接使用されません。iter_chunk_dataframesを使用してください。"
        )
        # 互換性のために残すが、エラーを発生させるか空のLazyFrameを返す
        # return pl.LazyFrame() # 空を返す
        raise NotImplementedError("Use iter_chunk_dataframes in chunk processing mode.")

    # --- 以下のメソッドは変更なし ---
    def get_data_summary(self, lazy_frame: pl.LazyFrame) -> Dict[str, Any]:
        """データ概要取得 (変更なし)"""
        try:
            schema = lazy_frame.collect_schema()
            columns = schema.names()
            dtypes = {col: str(schema[col]) for col in columns}
            sample_df = lazy_frame.head(1000).collect()

            summary = {
                "columns": columns,
                "dtypes": dtypes,
                "sample_rows": len(sample_df),
                "memory_usage_mb": sample_df.estimated_size("mb"),
            }
            return summary
        except Exception as e:
            logger.error(f"データ概要取得エラー: {e}")
            return {}

    def estimate_memory_usage(self, timeframe: str) -> float:
        """メモリ使用量推定（GB）(単一ファイル版 - 変更なし)"""
        if timeframe != "tick":
            return 0.0
        if not self.base_path.exists():
            return 0.0

        total_size = self.base_path.stat().st_size
        estimated_memory_gb = (total_size * 3) / (1024**3)  # 概算
        return estimated_memory_gb


# ========================================
# OutputEngine - 出力管理クラス
# ========================================


class OutputEngine:
    """出力管理エンジン (単一ファイル出力のため変更なし)"""

    def __init__(self, config: ProcessingConfig):
        """初期化"""
        self.config = config
        self.output_path = Path(config.output_base_path)
        self.output_path.mkdir(parents=True, exist_ok=True)
        logger.info(f"OutputEngine初期化: {self.output_path}")

    def create_output_path(self, filename: str) -> Path:
        """出力パス生成"""
        return self.output_path / filename

    def save_features(self, lazy_frame: pl.LazyFrame, timeframe: str) -> Dict[str, Any]:
        """特徴量保存 (単一ファイル用)"""
        # 【変更点】ファイル名が engine_id と timeframe から自動生成される
        # 例: features_e1c_tick_atr_only_tick.parquet
        output_file = f"features_{self.config.engine_id}_{timeframe}.parquet"
        output_path = self.create_output_path(output_file)

        start_time = time.time()

        try:
            # ストリーミング出力
            lazy_frame.sink_parquet(str(output_path), compression="snappy")
            elapsed_time = time.time() - start_time

            if output_path.exists():
                file_size_mb = output_path.stat().st_size / (1024**2)
                metadata = {
                    "timeframe": timeframe,
                    "output_file": str(output_path),
                    "file_size_mb": file_size_mb,
                    "save_time_seconds": elapsed_time,
                    "compression": "snappy",
                    "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                }
                logger.info(
                    f"保存完了: {output_file} ({file_size_mb:.2f}MB, {elapsed_time:.2f}秒)"
                )
                return metadata
            else:
                raise IOError(f"ファイル保存に失敗: {output_path}")

        except Exception as e:
            logger.error(f"特徴量保存エラー: {e}")
            raise

    def apply_nan_filling(self, lazy_frame: pl.LazyFrame) -> pl.LazyFrame:
        """NaN埋め処理 (構造維持のため変更なし)"""
        try:
            schema = lazy_frame.collect_schema()
            numeric_columns = [
                col
                for col, dtype in schema.items()
                if dtype in [pl.Float32, pl.Float64, pl.Int32, pl.Int64]
                and col not in ["timestamp", "timeframe"]
            ]
            exprs = [pl.col(col).fill_null(0.0).alias(col) for col in numeric_columns]

            if exprs:
                result = lazy_frame.with_columns(exprs)
                logger.info(f"NaN埋め処理完了: {len(exprs)}列")
            else:
                result = lazy_frame
            return result
        except Exception as e:
            logger.error(f"NaN埋め処理エラー: {e}")
            return lazy_frame

    def merge_intermediate_files(
        self, file_paths: List[str], timeframe: str
    ) -> Dict[str, Any]:
        """【構造維持】中間ファイル統合 (このスクリプトでは使用されない)"""
        logger.warning(
            "merge_intermediate_filesはパーティションモード専用です。この実行では呼び出されません。"
        )
        return {}

    def save_processing_metadata(self, metadata: Dict[str, Any]):
        """処理メタデータ保存 (構造維持のため変更なし)"""
        metadata_file = (
            f"metadata_{self.config.engine_id}_{time.strftime('%Y%m%d_%H%M%S')}.json"
        )
        metadata_path = self.create_output_path(metadata_file)

        try:
            with open(metadata_path, "w", encoding="utf-8") as f:
                json.dump(metadata, f, indent=2, ensure_ascii=False)
            logger.info(f"メタデータ保存: {metadata_file}")
        except Exception as e:
            logger.error(f"メタデータ保存エラー: {e}")


# ===== ブロック3完了 =====
# ブロック4の開始
# ===== ブロック4開始 =====

# ========================================
# CalculationEngine - 計算核心クラス
# ========================================


class CalculationEngine:
    """テクニカル指標計算エンジン (ATRのみにロジックを限定)"""

    def __init__(self, config: ProcessingConfig):
        """初期化"""
        self.config = config
        self.qa = QualityAssurance()
        self.memory_monitor = MemoryMonitor(config.memory_limit_gb)
        # 【変更点】プレフィックスを engine_id に合わせる
        self.prefix = config.engine_id + "_"  # 例: "e1c_tick_atr_only_"

        logger.info(f"CalculationEngine初期化: {self.prefix}")

    def get_feature_groups(self) -> Dict[str, List[str]]:
        """【構造維持】特徴量グループ定義 (このスクリプトでは使用されない)"""
        return {
            "group_1a_rsi_basic": ["rsi_14", "rsi_21"],
            "group_4a_atr_basic": ["atr_13", "atr_21"],
            # ... (他のグループ定義は省略) ...
        }

    def calculate_all_features(self, lazy_frame: pl.LazyFrame) -> pl.LazyFrame:
        """
        【変更点】
        全特徴量計算（ATRのみに限定）
        """
        logger.info(f"=== {self.prefix} ATR限定計算開始 ===")

        # メモリ安全性チェック
        is_safe, message = self.memory_monitor.check_memory_safety()
        if not is_safe:
            raise MemoryError(f"メモリ不足のため処理を中断: {message}")
        logger.info(message)

        try:
            # 1. ATR関連特徴量のみを計算
            result = self.create_atr_features(lazy_frame)

            # 2. 品質保証システム適用
            result = self.apply_quality_assurance(result)

            logger.info(f"=== {self.prefix} ATR限定計算完了 ===")
            return result

        except Exception as e:
            logger.error(f"ATR特徴量計算エラー: {e}")
            raise

    def create_rsi_features(self, lazy_frame: pl.LazyFrame) -> pl.LazyFrame:
        """【構造維持】RSI関連特徴量群 (呼び出されない)"""
        logger.info("RSI特徴量計算開始 (スキップ)")
        return lazy_frame  # 何もせずそのまま返す

    def create_macd_features(self, lazy_frame: pl.LazyFrame) -> pl.LazyFrame:
        """【構造維持】MACD関連特徴量群 (呼び出されない)"""
        logger.info("MACD特徴量計算開始 (スキップ)")
        return lazy_frame

    def create_bollinger_features(self, lazy_frame: pl.LazyFrame) -> pl.LazyFrame:
        """【構造維持】ボリンジャーバンド特徴量群 (呼び出されない)"""
        logger.info("ボリンジャーバンド特徴量計算開始 (スキップ)")
        return lazy_frame

    # 修正後全文 (create_atr_features)
    def create_atr_features(self, lazy_frame: pl.LazyFrame) -> pl.LazyFrame:
        """【変更点】ATR関連特徴量群 (ask/bid/mid_price を使用)"""
        logger.info("ATR特徴量計算開始 (Tickデータモード)")

        exprs = []

        # ATR期間
        atr_periods = self.config.window_sizes["atr"]  # [13, 21, 34, 55]

        # 入力カラムの存在チェック
        required_cols = ["ask", "bid", "mid_price"]
        schema = lazy_frame.collect_schema()
        missing_cols = [col for col in required_cols if col not in schema.names()]
        if missing_cols:
            logger.error(
                f"ATR計算に必要なカラムが不足しています: {missing_cols}. 利用可能なカラム: {schema.names()}"
            )
            raise ValueError(
                f"Missing required columns for ATR calculation: {missing_cols}"
            )

        for period in atr_periods:
            # 基本ATR（Numba版 - ask/bid/mid_price を使用）
            atr = (
                pl.struct(["ask", "bid", "mid_price"])
                .map_batches(
                    lambda s: calculate_atr_numba(
                        s.struct.field("ask").to_numpy(),
                        s.struct.field("bid").to_numpy(),
                        s.struct.field(
                            "mid_price"
                        ).to_numpy(),  # close の代わりに mid_price
                        period,
                    ),
                    return_dtype=pl.Float64,
                )
                .alias(f"{self.prefix}atr_{period}")
            )
            exprs.append(atr)

            # ATRパーセンテージ（mid_priceに対する比率）
            atr_pct = (atr / pl.col("mid_price") * 100).alias(
                f"{self.prefix}atr_pct_{period}"
            )
            exprs.append(atr_pct)

            # ATRトレンド（ATRの変化率）
            atr_trend = atr.diff().alias(f"{self.prefix}atr_trend_{period}")
            exprs.append(atr_trend)

            # ATRボラティリティ（ATR自体の標準偏差）
            atr_volatility = atr.rolling_std(period).alias(
                f"{self.prefix}atr_volatility_{period}"
            )
            exprs.append(atr_volatility)

            # ATRベースのバンド (基準を mid_price に変更)
            atr_multipliers = [1.0, 1.5, 2.0, 2.5, 3.0]
            for mult in atr_multipliers:
                upper_band = (pl.col("mid_price") + atr * mult).alias(
                    f"{self.prefix}atr_upper_{period}_{mult}"
                )
                lower_band = (pl.col("mid_price") - atr * mult).alias(
                    f"{self.prefix}atr_lower_{period}_{mult}"
                )
                exprs.extend([upper_band, lower_band])

        result = lazy_frame.with_columns(exprs)
        logger.info(f"ATR特徴量計算完了: {len(exprs)}個")
        return result

    # ===== ブロック4完了 =====
    # ブロック5の開始
    # ===== ブロック5開始 =====

    def create_oscillator_features(self, lazy_frame: pl.LazyFrame) -> pl.LazyFrame:
        """【構造維持】オシレーター系特徴量群 (呼び出されない)"""
        logger.info("オシレーター特徴量計算開始 (スキップ)")
        return lazy_frame

    def create_momentum_features(self, lazy_frame: pl.LazyFrame) -> pl.LazyFrame:
        """【構造維持】モメンタム系特徴量群 (呼び出されない)"""
        logger.info("モメンタム特徴量計算開始 (スキップ)")
        return lazy_frame

    def create_advanced_features(self, lazy_frame: pl.LazyFrame) -> pl.LazyFrame:
        """【構造維持】高度な指標特徴量群 (呼び出されない)"""
        logger.info("高度な指標特徴量計算開始 (スキップ)")
        return lazy_frame

    def create_moving_average_features(self, lazy_frame: pl.LazyFrame) -> pl.LazyFrame:
        """【構造維持】移動平均線・トレンド分析特徴量群 (呼び出されない)"""
        logger.info("移動平均線特徴量計算開始 (スキップ)")
        return lazy_frame

    def create_basic_processing_features(
        self, lazy_frame: pl.LazyFrame
    ) -> pl.LazyFrame:
        """【構造維持】基本データ処理特徴量 (呼び出されない)"""
        logger.info("基本データ処理特徴量計算開始 (スキップ)")
        return lazy_frame

    def apply_quality_assurance(self, lazy_frame: pl.LazyFrame) -> pl.LazyFrame:
        """品質保証システム適用 (構造維持のため変更なし)"""
        logger.info("品質保証システム適用開始")

        try:
            schema = lazy_frame.collect_schema()

            # 【変更点】プレフィックスが動的になった
            feature_columns = [
                col
                for col in schema.names()
                if col.startswith(self.prefix)
                and schema[col] in [pl.Float32, pl.Float64]
            ]

            qa_exprs = []
            for col in feature_columns:
                qa_exprs.append(
                    pl.col(col)
                    .map_batches(
                        lambda s: self._apply_qa_to_series(s.to_numpy()),
                        return_dtype=pl.Float64,
                    )
                    .alias(col)
                )

            if qa_exprs:
                result = lazy_frame.with_columns(qa_exprs)
                logger.info(f"品質保証適用完了: {len(qa_exprs)}列")
            else:
                result = lazy_frame
                logger.info("品質保証対象列なし")

            return result

        except Exception as e:
            logger.error(f"品質保証エラー: {e}")
            return lazy_frame

    def _apply_qa_to_series(self, values: np.ndarray) -> np.ndarray:
        """配列への品質保証適用 (構造維持のため変更なし)"""
        score = self.qa.calculate_quality_score(values)

        if score > 0.6:
            return self.qa.basic_stabilization(values)
        else:
            return self.qa.robust_stabilization(values)

    def calculate_feature_group(
        self, lazy_frame: pl.LazyFrame, group_name: str, feature_list: List[str]
    ) -> pl.LazyFrame:
        """【構造維持】特定グループの特徴量計算 (呼び出されない)"""
        logger.warning(
            "calculate_feature_groupはパーティションモード専用です。この実行では呼び出されません。"
        )
        return lazy_frame


# ===== ブロック5完了 =====
# ブロック6の開始
# ===== ブロック6開始 =====

# ========================================
# メイン処理関数
# ========================================


def get_sorted_partitions(root_dir: Path) -> list[Path]:
    """【構造維持】パーティション処理のためのヘルパー関数 (呼び出されない)"""
    logging.info(f"パーティションを探索中: {root_dir} (この実行では使用されません)")
    return []


def create_augmented_frame(
    current_partition_path: Path, prev_partition_path: Path | None, w_max: int
) -> tuple[pl.DataFrame, int]:
    """【構造維持】拡張データフレーム生成 (呼び出されない)"""
    logging.warning(
        "create_augmented_frameはパーティションモード専用です。この実行では呼び出されません。"
    )
    return pl.DataFrame(), 0


def run_on_partitions_mode(config: ProcessingConfig):
    """【構造維持】実行モード: Tickデータ専用 (呼び出されない)"""
    logger.warning(
        "run_on_partitions_modeはパーティションモード専用です。この実行では呼び出されません。"
    )
    pass


def process_single_timeframe(config: ProcessingConfig, timeframe: str):
    """
    【変更点】
    単一タイムフレーム処理 (チャンク対応 + PyArrow ParquetWriter 追記版)
    """
    if timeframe != "tick":
        logger.error(
            f"このエンジンはtickデータ専用です。要求されたタイムフレーム '{timeframe}' は処理できません。"
        )
        return {
            "timeframe": timeframe,
            "error": "Invalid timeframe for chunk processing.",
        }

    logger.info(f"=== チャンク処理 (PyArrow Writer) 開始: timeframe={timeframe} ===")
    start_time = time.time()

    # --- チャンクパラメータ設定 ---
    CHUNK_SIZE = 5_000_000
    OVERLAP_SIZE = 100

    data_engine = DataEngine(config)
    if not data_engine.validate_data_source():
        return {"timeframe": timeframe, "error": "Input data validation failed."}
    total_rows = data_engine.total_rows

    calc_engine = CalculationEngine(config)
    output_engine = OutputEngine(config)

    # --- PyArrow Writer の準備 ---
    final_output_file = f"features_{config.engine_id}_{timeframe}.parquet"
    final_output_path = output_engine.create_output_path(final_output_file)
    writer: Optional[pq.ParquetWriter] = None
    arrow_schema: Optional[pa.Schema] = None
    # ---

    processed_rows = 0
    num_chunks = 0

    try:
        chunk_iterator = data_engine.iter_chunk_dataframes(CHUNK_SIZE, OVERLAP_SIZE)

        for chunk_df, chunk_start_index, chunk_end_index in chunk_iterator:
            num_chunks += 1
            logger.info(
                f"チャンク {chunk_start_index}-{chunk_end_index - 1} (チャンク {num_chunks}) を処理中..."
            )
            chunk_start_time = time.time()

            actual_chunk_size = chunk_end_index - chunk_start_index
            overlap_rows_in_this_chunk = chunk_df.height - actual_chunk_size

            # LazyFrameに変換して計算
            chunk_lf = chunk_df.lazy()
            features_lf = calc_engine.create_atr_features(chunk_lf)
            qa_lf = calc_engine.apply_quality_assurance(features_lf)
            processed_lf = output_engine.apply_nan_filling(qa_lf)
            processed_df = processed_lf.collect()

            # オーバーラップ部分を除去
            if chunk_start_index == 0:
                result_df_for_chunk = processed_df
            else:
                result_df_for_chunk = processed_df.slice(
                    overlap_rows_in_this_chunk, actual_chunk_size
                )

            # --- PyArrow Writer への書き込み ---
            if not result_df_for_chunk.is_empty():
                # Polars DataFrameをArrow Tableに変換
                arrow_table = result_df_for_chunk.to_arrow()

                if writer is None:
                    # 最初のチャンクでスキーマを取得し、Writerを開く
                    arrow_schema = arrow_table.schema
                    logger.info(
                        f"最初のチャンクからスキーマを取得し、ParquetWriter を開きます: {final_output_path}"
                    )
                    # 既存ファイルがあれば上書き (wはwrite mode)
                    writer = pq.ParquetWriter(
                        final_output_path, arrow_schema, compression="snappy"
                    )

                # スキーマの一貫性を確認 (念のため)
                if arrow_table.schema != arrow_schema:
                    logger.error(
                        "エラー: チャンク間でスキーマが異なります。処理を中断します。"
                    )
                    logger.error(f"期待されたスキーマ: {arrow_schema}")
                    logger.error(f"現在のスキーマ: {arrow_table.schema}")
                    raise ValueError("Schema mismatch between chunks")

                # Arrow TableをParquetファイルに書き込む
                writer.write_table(arrow_table)
                processed_rows += result_df_for_chunk.height
            # --- 書き込み完了 ---

            chunk_elapsed_time = time.time() - chunk_start_time
            logger.info(
                f"チャンク {chunk_start_index}-{chunk_end_index - 1} 処理完了 ({chunk_elapsed_time:.2f}秒)。 {result_df_for_chunk.height}行を追記。"
            )

            # メモリチェック
            is_safe, message = calc_engine.memory_monitor.check_memory_safety()
            if not is_safe:
                raise MemoryError(f"チャンク処理中にメモリ不足: {message}")
            logger.info(message)

        # --- 全チャンク処理完了後 ---
        logger.info(
            f"全 {num_chunks} 個のチャンク処理完了。合計 {processed_rows} 行処理。"
        )

        if writer:
            logger.info("ParquetWriter を閉じています...")
            writer.close()
            logger.info("ParquetWriter を閉じました。")
        else:
            logger.warning(
                "処理されたデータがないため、出力ファイルは作成されませんでした。"
            )
            return {
                "timeframe": timeframe,
                "warning": "No data processed, output file not created.",
            }

        elapsed_time = time.time() - start_time

        # 保存後の検証とメタデータ作成
        if final_output_path.exists():
            file_size_mb = final_output_path.stat().st_size / (1024**2)
            final_rows = processed_rows
            if total_rows is not None and final_rows != total_rows:
                logger.warning(
                    f"最終的な行数 ({final_rows}) が元のファイル行数 ({total_rows}) と一致しません。"
                )

            metadata = {
                "timeframe": timeframe,
                "output_file": str(final_output_path),
                "file_size_mb": file_size_mb,
                "total_processing_time_seconds": elapsed_time,
                "compression": "snappy",  # Writerで指定した圧縮形式
                "processed_rows": final_rows,
                "num_chunks": num_chunks,
                "chunk_size": CHUNK_SIZE,
                "overlap_size": OVERLAP_SIZE,
                "write_method": "pyarrow_parquetwriter_append",  # 書き込み方法を記録
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            }
            logger.info(
                f"=== チャンク処理 (PyArrow Writer) 完了: {timeframe} - {elapsed_time:.2f}秒 ==="
            )
            logger.info(
                f"最終ファイル: {final_output_path} ({file_size_mb:.2f}MB, {final_rows}行)"
            )
            return metadata
        else:
            # Writerが正常に閉じられれば通常ここには来ない
            raise IOError(f"最終ファイルの作成に失敗したようです: {final_output_path}")

    except Exception as e:
        logger.error(
            f"タイムフレーム {timeframe} のチャンク処理中にエラー: {e}", exc_info=True
        )
        # エラーが発生した場合でも、開いているWriterがあれば閉じる試み
        if writer:
            try:
                writer.close()
                logger.info("エラー発生後、ParquetWriterを閉じました。")
            except Exception as close_err:
                logger.error(
                    f"エラー発生後のParquetWriterクローズ中にさらにエラー: {close_err}"
                )
        return {"timeframe": timeframe, "error": str(e)}
    finally:
        # 一時ディレクトリはこの方法では不要なので削除
        pass


# --- ユーザーとの対話部分（元の関数を流用） ---
def get_user_confirmation(config: ProcessingConfig) -> bool:
    """ユーザー確認"""
    print("\n" + "=" * 60)
    print(f" {config.engine_name} ")
    print(" (Tickデータ用 ATR限定抽出バージョン) ")
    print("=" * 60)
    print(f"入力ファイル: {config.input_base_path}")
    print(f"出力ディレクトリ: {config.output_base_path}")
    print(f"エンジンID: {config.engine_id}")
    print(f"並列スレッド数: {config.max_threads}")
    print(f"メモリ制限: {config.memory_limit_gb}GB")

    if config.test_mode:
        print(f"\n【テストモード】 最初の{config.test_rows}行のみ処理")

    print(f"\n処理対象タイムフレーム: {config.timeframes}")

    print("\n処理内容:")
    print("  - ATR関連特徴量のみを計算")
    print("  - 単一のParquetファイルとして出力")

    response = input("\n処理を開始しますか？ (y/n): ")
    return response.lower() == "y"


def select_timeframes(config: ProcessingConfig) -> List[str]:
    """【変更点】タイムフレーム選択 (tickのみに限定)"""
    print("\nタイムフレーム選択:")
    print("  このエンジンは 'tick' データ専用です。")
    print("  1. tick (実行)")
    print("  (他の選択肢は無効です)")

    selection = input("選択 (1): ").strip()

    if selection == "1" or selection == "":
        return ["tick"]

    logger.warning("無効な選択です。'tick' を処理します。")
    return ["tick"]


def main():
    """設定を行い、処理モードを分岐させるメイン関数 (呼び出し部分は変更なし)"""
    print("\n" + "=" * 70)
    print(" Engine 1C - ATR Only for Tick (Chunk Processing Mode) ")
    print(" Tickデータ用 ATR限定抽出エンジン (チャンク処理版) ")
    print("=" * 70)

    config = ProcessingConfig()

    if not config.validate():  # ここで pyarrow ファイルオープンと行数取得も行われる
        return 1

    # --- 対話形式のセットアップ ---
    # (省略 - 元のコードと同じ)
    # ... スレッド数、パス、メモリ、テストモードの対話 ...
    print("\n並列処理スレッド数を選択してください:")
    print("  1. 自動設定 (推奨)")
    print("  2. 手動設定")
    thread_selection = input("選択 (1/2): ").strip()
    if thread_selection == "2":
        try:
            max_threads = int(
                input(f"スレッド数を入力 (1-{psutil.cpu_count()}): ").strip()
            )
            if 1 <= max_threads <= psutil.cpu_count():
                config.max_threads = max_threads
            else:
                print("無効な値です。デフォルト値を使用します。")
        except ValueError:
            print("無効な入力です。デフォルト値を使用します。")

    print("\n実行モードを選択してください:")
    print("  1. テストモード（少量データで動作確認）")
    print("  2. 本番モード（全データ処理）")
    mode_selection = input("選択 (1/2): ").strip()
    if mode_selection == "1":
        config.test_mode = True
        try:
            # テストモードは最初のチャンクのみ処理するように変更
            test_rows = int(
                input(
                    f"テスト行数 (最初のチャンクの最大行数, デフォルト: {config.test_rows}): "
                ).strip()
                or str(config.test_rows)
            )
            # CHUNK_SIZE より小さい値を設定可能にする
            # ただし、実際のチャンク処理では最初のCHUNK_SIZEまで読み込まれる可能性がある
            # ここではテストモードフラグを立てるだけに留める
            config.test_rows = test_rows  # この値は直接使われないが、フラグとして利用
            logger.info(f"テストモード設定: 最初のチャンクのみ処理します。")

        except ValueError:
            print(f"無効な入力です。デフォルト設定を使用します。")
            config.test_mode = False  # エラー時は本番モードにする

    # --- 対話形式のセットアップここまで ---

    selected_timeframes = ["tick"]
    config.timeframes = selected_timeframes
    logger.info(f"処理対象タイムフレームを '{config.timeframes}' に設定しました。")

    if not get_user_confirmation(config):
        print("処理をキャンセルしました")
        return 0

    os.environ["POLARS_MAX_THREADS"] = str(config.max_threads)
    logger.info(f"並列処理スレッド数: {config.max_threads}")

    print("\n" + "=" * 60)
    print("処理開始...")
    print("=" * 60)

    overall_start_time = time.time()

    processing_metadata = {}
    for tf in config.timeframes:
        # テストモードの場合、最初のチャンクのみ処理するように process_single_timeframe 内で制御
        # ここでは単純に呼び出す
        metadata = process_single_timeframe(config, tf)
        processing_metadata[tf] = metadata
        # テストモードの場合は最初のタイムフレーム処理後にループを抜ける
        if config.test_mode:
            logger.info("テストモードのため、最初のタイムフレーム処理後に終了します。")
            break

    # メタデータ保存
    output_engine = OutputEngine(config)
    output_engine.save_processing_metadata(processing_metadata)

    overall_elapsed_time = time.time() - overall_start_time
    print(
        f"\n全ての要求された処理が完了しました。総処理時間: {overall_elapsed_time:.2f}秒"
    )
    return 0


# --- スクリプト実行のエントリーポイント (変更なし) ---
if __name__ == "__main__":
    try:
        exit_code = main()
        sys.exit(exit_code)
    except Exception as e:
        logger.critical(
            f"スクリプト実行中に致命的なエラーが発生しました: {e}", exc_info=True
        )
        sys.exit(1)

# ===== ブロック6完了 =====
# ===== Engine 1C (ATR-Only for Tick, Chunk Processing) 実装完了 =====
