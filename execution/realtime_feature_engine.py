# /workspace/execution/realtime_feature_engine.py
import sys
from pathlib import Path
from collections import deque
import numpy as np
import numba as nb
from numba import njit, prange, float64, int64, boolean
import math
from typing import Dict, List, Optional, Tuple, Deque, Any
import logging
import scipy.stats as stats
import re
import polars as pl
from dataclasses import dataclass, field
from datetime import datetime
import pandas as pd  # リサンプリングで使用
import MetaTrader5 as mt5  # MT5時間足定数で使用

# --- プロジェクトのルートディレクトリをPythonの検索パスに追加 ---
project_root = Path(__file__).resolve().parents[1]
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

import blueprint as config

# ==================================================================
#
# 1. NUMBA UDF ライブラリ (Part 1: A-C)
# (全 engine_...py スクリプトから必要な Numba UDF を集約)
#
# ==================================================================

# ----------------------------------------
# Polarsネイティブ関数のNumba版
# ----------------------------------------


@njit(fastmath=True, cache=True)
def rolling_skew_numba(arr: np.ndarray) -> float:
    """ローリング歪度 (Numba JIT)"""
    n = len(arr)
    if n < 3:
        return np.nan
    mean_val = np.mean(arr)
    std_val = np.std(arr)
    if std_val < 1e-10:
        return 0.0
    m3 = np.mean((arr - mean_val) ** 3)
    skew = m3 / (std_val**3)
    return skew


@njit(fastmath=True, cache=True)
def rolling_kurtosis_numba(arr: np.ndarray) -> float:
    """ローリング尖度 (Numba JIT)"""
    n = len(arr)
    if n < 4:
        return np.nan
    mean_val = np.mean(arr)
    std_val = np.std(arr)
    if std_val < 1e-10:
        return 0.0
    m4 = np.mean((arr - mean_val) ** 4)
    kurtosis = m4 / (std_val**4) - 3.0
    return kurtosis


@njit(fastmath=True, cache=True)
def ewm_mean_numba(arr: np.ndarray, span: int) -> float:
    """指数移動平均 (Numba JIT) - 最新値のみ返す"""
    n = len(arr)
    if n == 0:
        return np.nan
    alpha = 2.0 / (span + 1.0)
    ema = arr[0]
    for i in range(1, n):
        ema = alpha * arr[i] + (1.0 - alpha) * ema
    return ema


# ----------------------------------------
# from engine_1_A_a_vast_universe_of_features.py
# ----------------------------------------


@njit(fastmath=True, cache=True)
def statistical_moment_numba(arr: np.ndarray, moment: int) -> float:
    """高次モーメント (Numba JIT)"""
    n = len(arr)
    if n < 2:
        return np.nan
    mean_val = np.mean(arr)
    std_val = np.std(arr)
    if std_val < 1e-10:
        return 0.0
    z = (arr - mean_val) / std_val
    return np.mean(z**moment)


@njit(fastmath=True, cache=True)
def mad_rolling_numba(arr: np.ndarray) -> float:
    """ローリングMAD (Numba JIT)"""
    n = len(arr)
    if n == 0:
        return np.nan
    median_val = np.median(arr)
    abs_deviations = np.abs(arr - median_val)
    return np.median(abs_deviations)


@njit(fastmath=True, cache=True)
def jarque_bera_statistic_numba(arr: np.ndarray) -> float:
    """ローリングJarque-Bera (Numba JIT)"""
    n = len(arr)
    if n < 20:
        return np.nan

    mean_val = np.mean(arr)
    std_val = np.std(arr)

    if std_val < 1e-10:
        return 0.0

    z = (arr - mean_val) / std_val
    skewness = np.mean(z**3)
    kurtosis = np.mean(z**4) - 3.0

    jb_stat = n * (skewness**2 / 6.0 + kurtosis**2 / 24.0)
    return jb_stat


@njit(fastmath=True, cache=True)
def anderson_darling_numba(arr: np.ndarray) -> float:
    """ローリングAnderson-Darling (Numba JIT)"""
    n = len(arr)
    if n < 10:
        return np.nan

    sorted_data = np.sort(arr)
    mean_val = np.mean(sorted_data)
    std_val = np.std(sorted_data)

    if std_val < 1e-10:
        return 0.0

    SQRT2 = 1.4142135623730951

    ad_sum = 0.0
    for j in range(n):
        # 標準正規分布のCDF (erf)
        z = (sorted_data[j] - mean_val) / std_val
        F_j = 0.5 * (1.0 + math.erf(z / SQRT2))

        z_nj = (sorted_data[n - 1 - j] - mean_val) / std_val
        F_nj = 0.5 * (1.0 + math.erf(z_nj / SQRT2))

        if F_j < 1e-15:
            F_j = 1e-15
        if F_nj < 1e-15:
            F_nj = 1e-15

        log_term = np.log(F_j) + np.log(1.0 - F_nj)
        ad_sum += (2.0 * j + 1.0) * log_term

    return -n - ad_sum / n


@njit(fastmath=True, cache=True)
def runs_test_numba(arr: np.ndarray) -> float:
    """ローリングRuns Test (Numba JIT)"""
    n = len(arr)
    if n < 10:
        return np.nan

    median_val = np.median(arr)
    binary_series = arr > median_val

    runs = 1
    for j in range(1, n):
        if binary_series[j] != binary_series[j - 1]:
            runs += 1

    n1 = np.sum(binary_series)
    n2 = n - n1

    if n1 == 0 or n2 == 0:
        return 0.0

    expected_runs = (2.0 * n1 * n2) / (n1 + n2) + 1.0
    var_runs = (2.0 * n1 * n2 * (2.0 * n1 * n2 - n1 - n2)) / (
        (n1 + n2) ** 2 * (n1 + n2 - 1.0)
    )

    if var_runs > 0:
        return (runs - expected_runs) / np.sqrt(var_runs)
    return 0.0


@njit(fastmath=True, cache=True)
def von_neumann_ratio_numba(arr: np.ndarray) -> float:
    """ローリングVon Neumann比 (Numba JIT)"""
    n = len(arr)
    if n < 3:
        return np.nan

    diff_sq_sum = np.sum(np.diff(arr) ** 2)
    mean_val = np.mean(arr)
    sum_sq_deviations = np.sum((arr - mean_val) ** 2)

    if sum_sq_deviations > 1e-15:
        vn_ratio = diff_sq_sum / sum_sq_deviations
        return min(max(vn_ratio, 0.0), 4.0)
    return 0.0


# ----------------------------------------
# from engine_1_B_a_vast_universe_of_features.py
# ----------------------------------------


@njit(fastmath=True, cache=True)
def adf_統計量_udf(prices: np.ndarray) -> float:
    """ADF検定統計量 (Numba JIT)"""
    if len(prices) < 10:
        return np.nan
    diff_prices = np.diff(prices)
    lagged_prices = prices[:-1]
    n = len(diff_prices)
    if n < 5:
        return np.nan
    X = np.column_stack((np.ones(n), lagged_prices))
    y = diff_prices
    try:
        XtX = X.T @ X
        XtX_inv = np.linalg.inv(XtX)
        XtY = X.T @ y
        beta = XtX_inv @ XtY
        residuals = y - X @ beta
        mse = np.sum(residuals**2) / (n - 2)
        se_beta = np.sqrt(mse * XtX_inv[1, 1])
        return beta[1] / se_beta if se_beta > 0 else np.nan
    except:
        return np.nan


@njit(fastmath=True, cache=True)
def kpss_統計量_udf(prices: np.ndarray) -> float:
    """KPSS検定統計量 (Numba JIT)"""
    if len(prices) < 10:
        return np.nan
    n = len(prices)
    t = np.arange(n, dtype=np.float64)
    try:
        sum_t = np.sum(t)
        sum_t2 = np.sum(t**2)
        sum_y = np.sum(prices)
        sum_ty = np.sum(t * prices)
        beta = (n * sum_ty - sum_t * sum_y) / (n * sum_t2 - sum_t**2)
        alpha = (sum_y - beta * sum_t) / n
        detrended = prices - (alpha + beta * t)
        cumsum = np.cumsum(detrended)
        sse = np.sum(detrended**2) / n
        return np.sum(cumsum**2) / (n**2 * sse) if sse > 0 else np.nan
    except:
        return np.nan


@njit(fastmath=True, cache=True)
def arima_残差分散_udf(prices: np.ndarray) -> float:
    """ARIMA(1,1,1)残差分散 (Numba JIT)"""
    if len(prices) < 15:
        return np.nan
    diff_prices = np.diff(prices)
    if len(diff_prices) < 10:
        return np.nan
    y = diff_prices[1:]
    x = diff_prices[:-1]
    n = len(y)
    if n < 5:
        return np.nan
    sum_x = np.sum(x)
    sum_y = np.sum(y)
    sum_xy = np.sum(x * y)
    sum_x2 = np.sum(x * x)
    if n * sum_x2 - sum_x**2 == 0:
        return np.nan
    phi = (n * sum_xy - sum_x * sum_y) / (n * sum_x2 - sum_x**2)
    intercept = (sum_y - phi * sum_x) / n
    residuals = y - (intercept + phi * x)
    return np.sum(residuals**2) / (n - 2)


@njit(fastmath=True, cache=True)
def holt_winters_レベル_udf(prices: np.ndarray) -> float:
    """Holt-Winters レベル (Numba JIT)"""
    if len(prices) < 10:
        return np.nan
    alpha = 0.3
    level = prices[0]
    for i in range(1, len(prices)):
        level = alpha * prices[i] + (1 - alpha) * level
    return level


@njit(fastmath=True, cache=True)
def holt_winters_トレンド_udf(prices: np.ndarray) -> float:
    """Holt-Winters トレンド (Numba JIT)"""
    if len(prices) < 10:
        return np.nan
    alpha = 0.3
    beta = 0.1
    level = prices[0]
    trend = prices[1] - prices[0] if len(prices) > 1 else 0.0
    for i in range(1, len(prices)):
        new_level = alpha * prices[i] + (1 - alpha) * (level + trend)
        new_trend = beta * (new_level - level) + (1 - beta) * trend
        level, trend = new_level, new_trend
    return trend


@njit(fastmath=True, cache=True)
def kalman_状態推定_udf(prices: np.ndarray) -> float:
    """Kalman状態推定 (Numba JIT)"""
    if len(prices) < 5:
        return np.nan
    x, P = prices[0], 1.0
    Q = np.var(np.diff(prices)) if len(prices) > 1 else 0.01
    R = np.var(prices) * 0.1
    if R == 0:
        R = 0.1
    for i in range(1, len(prices)):
        x_pred, P_pred = x, P + Q
        K = P_pred / (P_pred + R)
        x = x_pred + K * (prices[i] - x_pred)
        P = (1 - K) * P_pred
    return x


@njit(fastmath=True, cache=True)
def lowess_適合値_udf(prices: np.ndarray) -> float:
    """LOWESS適合値 (Numba JIT)"""
    if len(prices) < 10:
        return np.nan
    n = len(prices)
    h = max(3, int(0.3 * n))
    target_idx = n - 1
    distances = np.abs(np.arange(n) - target_idx)
    sorted_indices = np.argsort(distances)
    neighbor_indices = sorted_indices[:h]
    x_neighbors = neighbor_indices.astype(np.float64)
    y_neighbors = prices[neighbor_indices]
    max_dist = np.max(distances[neighbor_indices])
    if max_dist <= 0:
        return np.mean(y_neighbors)
    weights = np.zeros(h)
    for i in range(h):
        u = distances[neighbor_indices[i]] / max_dist
        weights[i] = (1 - u**3) ** 3 if u < 1.0 else 0.0
    sum_w = np.sum(weights)
    if sum_w <= 0:
        return np.mean(y_neighbors)
    x_mean = np.sum(weights * x_neighbors) / sum_w
    y_mean = np.sum(weights * y_neighbors) / sum_w
    numerator = np.sum(weights * (x_neighbors - x_mean) * (y_neighbors - y_mean))
    denominator = np.sum(weights * (x_neighbors - x_mean) ** 2)
    if denominator <= 0:
        return y_mean
    slope = numerator / denominator
    intercept = y_mean - slope * x_mean
    return intercept + slope * target_idx


@njit(fastmath=True, cache=True)
def theil_sen_傾き_udf(prices: np.ndarray) -> float:
    """Theil-Sen傾き (Numba JIT)"""
    if len(prices) < 10:
        return np.nan
    n = len(prices)
    slopes = []
    max_pairs = min(1000, (n * (n - 1)) // 2)
    step = max(1, ((n * (n - 1)) // 2) // max_pairs)
    count = 0
    for i in range(n - 1):
        for j in range(i + 1, n):
            if count % step == 0:
                slopes.append((prices[j] - prices[i]) / (j - i))
            count += 1
    return np.median(np.array(slopes)) if slopes else np.nan


@njit(fastmath=True, cache=True)
def t分布_自由度_udf(returns: np.ndarray) -> float:
    """t分布 自由度 (Numba JIT)"""
    if len(returns) < 10:
        return np.nan
    mean_ret = np.mean(returns)
    std_ret = np.std(returns)
    if std_ret <= 0:
        return np.nan
    standardized = (returns - mean_ret) / std_ret
    fourth_moment = np.mean(standardized**4)
    excess_kurtosis = fourth_moment - 3.0
    if excess_kurtosis > 0:
        dof = 4.0 * (3.0 + fourth_moment) / excess_kurtosis
        return max(2.1, min(dof, 100.0))
    return 100.0


@njit(fastmath=True, cache=True)
def t分布_尺度_udf(returns: np.ndarray) -> float:
    """t分布 尺度 (Numba JIT)"""
    if len(returns) < 5:
        return np.nan
    dof = t分布_自由度_udf(returns)
    if np.isnan(dof) or dof <= 2:
        return np.std(returns)
    sample_var = np.var(returns)
    scale_squared = sample_var * (dof - 2.0) / dof
    return np.sqrt(max(scale_squared, 1e-8))


# ----------------------------------------
# from engine_1_C_a_vast_universe_of_features.py
# ----------------------------------------


@njit(fastmath=True, cache=True)
def calculate_atr_numba(
    high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int
) -> np.ndarray:
    n = len(high)
    out = np.full(n, np.nan)
    tr = np.zeros(n)
    for i in range(n):
        if i == 0:
            tr[i] = high[i] - low[i]
        else:
            h_l = high[i] - low[i]
            h_pc = abs(high[i] - close[i - 1])
            l_pc = abs(low[i] - close[i - 1])
            tr[i] = max(h_l, h_pc, l_pc)
    if n <= period:
        return out
    out[period] = np.mean(tr[1 : period + 1])
    for i in range(period + 1, n):
        out[i] = (out[i - 1] * (period - 1) + tr[i]) / period
    return out


@njit(fastmath=True, cache=True)
def calculate_di_plus_numba(
    high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int
) -> np.ndarray:
    n = len(high)
    out = np.full(n, np.nan)
    tr = np.zeros(n)
    dm_plus = np.zeros(n)
    for i in range(1, n):
        h_l = high[i] - low[i]
        h_pc = abs(high[i] - close[i - 1])
        l_pc = abs(low[i] - close[i - 1])
        tr[i] = max(h_l, h_pc, l_pc)
        up_move = high[i] - high[i - 1]
        down_move = low[i - 1] - low[i]
        if up_move > down_move and up_move > 0:
            dm_plus[i] = up_move
    if n <= period:
        return out
    atr_val = np.mean(tr[1 : period + 1])
    di_plus_val = np.mean(dm_plus[1 : period + 1])
    di_plus_ema = np.zeros(n)
    atr_ema = np.zeros(n)
    di_plus_ema[period] = di_plus_val
    atr_ema[period] = atr_val
    for i in range(period + 1, n):
        di_plus_ema[i] = (di_plus_ema[i - 1] * (period - 1) + dm_plus[i]) / period
        atr_ema[i] = (atr_ema[i - 1] * (period - 1) + tr[i]) / period
    for i in range(period, n):
        out[i] = 100.0 * di_plus_ema[i] / atr_ema[i] if atr_ema[i] > 0 else 0.0
    return out


@njit(fastmath=True, cache=True)
def calculate_di_minus_numba(
    high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int
) -> np.ndarray:
    n = len(high)
    out = np.full(n, np.nan)
    tr = np.zeros(n)
    dm_minus = np.zeros(n)
    for i in range(1, n):
        h_l = high[i] - low[i]
        h_pc = abs(high[i] - close[i - 1])
        l_pc = abs(low[i] - close[i - 1])
        tr[i] = max(h_l, h_pc, l_pc)
        up_move = high[i] - high[i - 1]
        down_move = low[i - 1] - low[i]
        if down_move > up_move and down_move > 0:
            dm_minus[i] = down_move
    if n <= period:
        return out
    atr_val = np.mean(tr[1 : period + 1])
    di_minus_val = np.mean(dm_minus[1 : period + 1])
    di_minus_ema = np.zeros(n)
    atr_ema = np.zeros(n)
    di_minus_ema[period] = di_minus_val
    atr_ema[period] = atr_val
    for i in range(period + 1, n):
        di_minus_ema[i] = (di_minus_ema[i - 1] * (period - 1) + dm_minus[i]) / period
        atr_ema[i] = (atr_ema[i - 1] * (period - 1) + tr[i]) / period
    for i in range(period, n):
        out[i] = 100.0 * di_minus_ema[i] / atr_ema[i] if atr_ema[i] > 0 else 0.0
    return out


@njit(fastmath=True, cache=True)
def calculate_adx_numba(
    high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int
) -> np.ndarray:
    """ADX (Wilder's smoothing) (Numba JIT)"""
    n = len(high)
    out = np.full(n, np.nan)

    # 1. Get DI+ and DI-
    di_plus = calculate_di_plus_numba(high, low, close, period)
    di_minus = calculate_di_minus_numba(high, low, close, period)

    if n <= period * 2:  # Need period for DI AND period for ADX smoothing
        return out

    # 2. Calculate DX
    dx = np.zeros(n)
    for i in range(period, n):  # DI values start at index `period`
        di_sum = di_plus[i] + di_minus[i]
        if di_sum > 0:
            dx[i] = 100.0 * np.abs(di_plus[i] - di_minus[i]) / di_sum
        else:
            dx[i] = 0.0

    # 3. Calculate ADX (Wilder's smoothing of DX)
    # Initial SMA (DXの最初の`period`個の値で計算)
    out[period * 2 - 1] = np.mean(dx[period : period * 2])

    # Wilder's smoothing
    for i in range(period * 2, n):
        out[i] = (out[i - 1] * (period - 1) + dx[i]) / period

    return out


@njit(fastmath=True, cache=True)
def calculate_aroon_up_numba(high: np.ndarray, period: int) -> np.ndarray:
    n = len(high)
    out = np.full(n, np.nan)
    for i in range(period - 1, n):
        window = high[i - period + 1 : i + 1]
        highest_idx = np.argmax(window)
        periods_since = (period - 1) - highest_idx
        out[i] = 100.0 * (period - periods_since) / period
    return out


@njit(fastmath=True, cache=True)
def calculate_aroon_down_numba(low: np.ndarray, period: int) -> np.ndarray:
    n = len(low)
    out = np.full(n, np.nan)
    for i in range(period - 1, n):
        window = low[i - period + 1 : i + 1]
        lowest_idx = np.argmin(window)
        periods_since = (period - 1) - lowest_idx
        out[i] = 100.0 * (period - periods_since) / period
    return out


@njit(fastmath=True, cache=True)
def calculate_stochastic_numba(
    high: np.ndarray, low: np.ndarray, close: np.ndarray, k_period: int
) -> np.ndarray:
    # (元コードの L:593 では k_period, d_period, smooth_k があったが、
    #  L:1282 での使用法を見ると d_period, smooth_k は
    #  Python側 (polars) で .rolling_mean() しているため、
    #  Numba UDF は %K の計算のみでよい)
    n = len(high)
    k_values = np.full(n, np.nan)
    for i in range(k_period - 1, n):
        window_high = np.max(high[i - k_period + 1 : i + 1])
        window_low = np.min(low[i - k_period + 1 : i + 1])
        if window_high - window_low > 0:
            k_values[i] = 100.0 * (close[i] - window_low) / (window_high - window_low)
        else:
            k_values[i] = 50.0  # (レンジゼロの場合は中央値)
    return k_values


@njit(fastmath=True, cache=True)
def calculate_williams_r_numba(
    high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int
) -> np.ndarray:
    n = len(high)
    out = np.full(n, np.nan)
    for i in range(period - 1, n):
        window_high = np.max(high[i - period + 1 : i + 1])
        window_low = np.min(low[i - period + 1 : i + 1])
        if window_high - window_low > 0:
            out[i] = -100.0 * (window_high - close[i]) / (window_high - window_low)
        else:
            out[i] = -50.0
    return out


@njit(fastmath=True, cache=True)
def calculate_trix_numba(prices: np.ndarray, period: int) -> np.ndarray:
    n = len(prices)
    out = np.full(n, np.nan)
    if n < period * 3:
        return out
    alpha = 2.0 / (period + 1.0)
    ema1 = np.zeros(n)
    ema2 = np.zeros(n)
    ema3 = np.zeros(n)
    ema1[0] = prices[0]
    for i in range(1, n):
        ema1[i] = alpha * prices[i] + (1.0 - alpha) * ema1[i - 1]
    ema2[0] = ema1[0]
    for i in range(1, n):
        ema2[i] = alpha * ema1[i] + (1.0 - alpha) * ema2[i - 1]
    ema3[0] = ema2[0]
    for i in range(1, n):
        ema3[i] = alpha * ema2[i] + (1.0 - alpha) * ema3[i - 1]
    for i in range(1, n):
        if ema3[i - 1] != 0:
            out[i] = 10000.0 * (ema3[i] - ema3[i - 1]) / ema3[i - 1]
        else:
            out[i] = 0.0
    return out


@njit(fastmath=True, cache=True)
def calculate_ultimate_oscillator_numba(
    high: np.ndarray, low: np.ndarray, close: np.ndarray
) -> np.ndarray:
    n = len(high)
    out = np.full(n, np.nan)
    periods = [7, 14, 28]
    bp = np.zeros(n)
    tr = np.zeros(n)
    for i in range(1, n):
        bp[i] = close[i] - min(low[i], close[i - 1])
        tr[i] = max(
            high[i] - low[i], abs(high[i] - close[i - 1]), abs(low[i] - close[i - 1])
        )
    for i in range(max(periods) - 1, n):  # (修正: -1 が必要)
        avg_7 = np.sum(bp[i - 7 + 1 : i + 1]) / (np.sum(tr[i - 7 + 1 : i + 1]) + 1e-10)
        avg_14 = np.sum(bp[i - 14 + 1 : i + 1]) / (
            np.sum(tr[i - 14 + 1 : i + 1]) + 1e-10
        )
        avg_28 = np.sum(bp[i - 28 + 1 : i + 1]) / (
            np.sum(tr[i - 28 + 1 : i + 1]) + 1e-10
        )
        out[i] = 100.0 * (4.0 * avg_7 + 2.0 * avg_14 + 1.0 * avg_28) / (4.0 + 2.0 + 1.0)
    return out


@njit(fastmath=True, cache=True)
def calculate_tsi_numba(
    prices: np.ndarray, long_period: int, short_period: int
) -> np.ndarray:
    n = len(prices)
    out = np.full(n, np.nan)
    momentum = np.diff(prices)
    momentum = np.insert(momentum, 0, 0.0)
    alpha_long = 2.0 / (long_period + 1.0)
    alpha_short = 2.0 / (short_period + 1.0)
    ema1_mom = np.zeros(n)
    ema1_abs = np.zeros(n)
    ema2_mom = np.zeros(n)
    ema2_abs = np.zeros(n)
    ema1_mom[0] = momentum[0]
    ema1_abs[0] = abs(momentum[0])
    for i in range(1, n):
        ema1_mom[i] = alpha_long * momentum[i] + (1.0 - alpha_long) * ema1_mom[i - 1]
        ema1_abs[i] = (
            alpha_long * abs(momentum[i]) + (1.0 - alpha_long) * ema1_abs[i - 1]
        )
    ema2_mom[0] = ema1_mom[0]
    ema2_abs[0] = ema1_abs[0]
    for i in range(1, n):
        ema2_mom[i] = alpha_short * ema1_mom[i] + (1.0 - alpha_short) * ema2_mom[i - 1]
        ema2_abs[i] = alpha_short * ema1_abs[i] + (1.0 - alpha_short) * ema2_abs[i - 1]
    for i in range(long_period + short_period, n):
        out[i] = 100.0 * ema2_mom[i] / ema2_abs[i] if ema2_abs[i] != 0 else 0.0
    return out


@njit(fastmath=True, cache=True)
def calculate_hma_numba(prices: np.ndarray, period: int) -> np.ndarray:
    n = len(prices)
    out = np.full(n, np.nan)
    half_period = period // 2
    sqrt_period = int(np.sqrt(period))
    if n < period + sqrt_period:
        return out

    def wma_numba(data: np.ndarray) -> float:
        n_w = len(data)
        weight_sum = n_w * (n_w + 1) / 2.0
        value_sum = 0.0
        for i in range(n_w):
            value_sum += data[i] * (i + 1)
        return value_sum / weight_sum if weight_sum > 0 else np.nan

    wma_half = np.zeros(n)
    wma_full = np.zeros(n)
    raw_hma = np.zeros(n)
    for i in range(half_period - 1, n):
        wma_half[i] = wma_numba(prices[i - half_period + 1 : i + 1])
    for i in range(period - 1, n):
        wma_full[i] = wma_numba(prices[i - period + 1 : i + 1])
    for i in range(period - 1, n):
        raw_hma[i] = 2.0 * wma_half[i] - wma_full[i]
    for i in range(period - 1 + sqrt_period - 1, n):
        out[i] = wma_numba(raw_hma[i - sqrt_period + 1 : i + 1])
    return out


@nb.njit(fastmath=True, cache=True)
def fast_rolling_mean_numba(arr: np.ndarray, window: int) -> np.ndarray:
    """Numba最適化ローリング平均（カスタムウィンドウ用）"""
    n = len(arr)
    out = np.full(n, np.nan)
    if n < 1:
        return out

    for i in range(n):
        start = max(0, i - window + 1)
        window_data = arr[start : i + 1]

        window_sum = 0.0
        count = 0
        for j in range(len(window_data)):
            if not np.isnan(window_data[j]):
                window_sum += window_data[j]
                count += 1

        if count > 0:
            out[i] = window_sum / count
        else:
            out[i] = np.nan
    return out


@nb.njit(fastmath=True, cache=True)
def fast_rolling_std_numba(arr: np.ndarray, window: int) -> np.ndarray:
    """Numba最適化ローリング標準偏差"""
    n = len(arr)
    out = np.full(n, np.nan)
    if n < 1:
        return out

    for i in range(n):
        start = max(0, i - window + 1)
        window_data = arr[start : i + 1]

        # Numbaはnp.nanを無視するmean/stdをサポートしていないため手動実装
        finite_data = window_data[~np.isnan(window_data)]

        if len(finite_data) <= 1:
            out[i] = np.nan
            continue

        mean_val = np.mean(finite_data)

        var_sum = 0.0
        for val in finite_data:
            diff = val - mean_val
            var_sum += diff * diff

        variance = var_sum / (len(finite_data) - 1)  # 不偏標準偏差
        out[i] = np.sqrt(variance)
    return out


@nb.njit(fastmath=True, cache=True)
def basic_stabilization_numba(arr: np.ndarray) -> np.ndarray:
    n = len(arr)
    out = np.zeros(n)
    for i in range(n):
        if np.isnan(arr[i]) or np.isinf(arr[i]):
            out[i] = 0.0
        else:
            out[i] = arr[i]
    finite_count = 0
    for i in range(n):
        if np.isfinite(out[i]):
            finite_count += 1
    if finite_count > 10:
        min_val = np.nanmin(out)
        max_val = np.nanmax(out)
        range_val = max_val - min_val
        if range_val > 1e-10:
            clip_margin_low = np.percentile(out[np.isfinite(out)], 1)
            clip_margin_high = np.percentile(out[np.isfinite(out)], 99)
            for i in range(n):
                if np.isfinite(out[i]):
                    if out[i] < clip_margin_low:
                        out[i] = clip_margin_low
                    elif out[i] > clip_margin_high:
                        out[i] = clip_margin_high
    return out


@nb.njit(fastmath=True, cache=True)
def phillips_perron_統計量_udf(prices: np.ndarray) -> float:
    """
    フィリップス・ペロン検定統計量計算
    異分散性修正による単位根のノンパラメトリック検定
    """
    if len(prices) < 10:
        return np.nan

    finite_prices = prices[np.isfinite(prices)]
    if len(finite_prices) < 10:
        return np.nan

    # 1次差分とラグレベル計算
    diff_prices = np.diff(finite_prices)
    lagged_prices = finite_prices[:-1]

    if len(diff_prices) < 5:
        return np.nan

    n = len(diff_prices)

    # OLS回帰: Δy_t = α + βy_{t-1} + ε_t
    X = np.column_stack((np.ones(n), lagged_prices))
    y = diff_prices

    try:
        # OLS推定
        XtX_inv = np.linalg.inv(X.T @ X)
        beta = XtX_inv @ X.T @ y

        # 残差
        residuals = y - X @ beta

        # Newey-West修正による分散推定（簡略版）
        sigma2 = np.var(residuals)

        # PP検定統計量（簡略版）
        se_beta = np.sqrt(sigma2 * XtX_inv[1, 1])

        if se_beta > 0:
            pp_stat = beta[1] / se_beta
        else:
            pp_stat = np.nan

    except:
        pp_stat = np.nan

    return pp_stat


@nb.njit(fastmath=True, cache=True)
def wma_rolling_numba(arr: np.ndarray, window: int) -> np.ndarray:
    """Numba最適化ローリング加重移動平均"""
    n = len(arr)
    out = np.full(n, np.nan)
    if n < window or window <= 0:
        return out

    weight_sum = window * (window + 1) / 2.0
    if weight_sum <= 0:
        return out  # (window=0 の場合)

    for i in range(window - 1, n):
        window_data = arr[i - window + 1 : i + 1]
        value_sum = 0.0
        for j in range(window):
            value_sum += window_data[j] * (j + 1)

        out[i] = value_sum / weight_sum

    return out


@nb.njit(fastmath=True, cache=True)
def calculate_kama_numba(prices: np.ndarray, period: int) -> np.ndarray:
    """Kaufman Adaptive Moving Average計算 (Numba最適化版)"""
    n = len(prices)
    out = np.full(n, np.nan)
    fast_ema_period = 2
    slow_ema_period = 30

    if n <= period:
        return out

    fast_sc = 2.0 / (fast_ema_period + 1.0)
    slow_sc = 2.0 / (slow_ema_period + 1.0)

    out[period - 1] = np.mean(prices[:period])  # 初期値

    for i in range(period, n):
        # Efficiency Ratio (ER)
        direction = abs(prices[i] - prices[i - period])

        volatility = 0.0
        for j in range(i - period + 1, i + 1):
            volatility += abs(prices[j] - prices[j - 1])

        if volatility == 0:
            er = 0.0
        else:
            er = direction / volatility

        # Smoothing Constant (SC)
        sc = (er * (fast_sc - slow_sc) + slow_sc) ** 2

        # KAMA
        if np.isnan(out[i - 1]):
            out[i] = np.mean(prices[i - period + 1 : i + 1])  # フォールバック
        else:
            out[i] = out[i - 1] + sc * (prices[i] - out[i - 1])

    return out


@njit(fastmath=True, cache=True)
def calculate_rsi_numba(prices: np.ndarray, period: int) -> np.ndarray:
    """RSI (Wilder's smoothing) (Numba JIT)"""
    n = len(prices)
    out = np.full(n, np.nan)
    if n <= period:
        return out

    gains = np.zeros(n)
    losses = np.zeros(n)

    for i in range(1, n):
        diff = prices[i] - prices[i - 1]
        if diff > 0:
            gains[i] = diff
        else:
            losses[i] = abs(diff)

    # Initial SMA
    avg_gain = np.mean(gains[1 : period + 1])
    avg_loss = np.mean(losses[1 : period + 1])

    if avg_loss > 0:
        rs = avg_gain / avg_loss
        out[period] = 100.0 - (100.0 / (1.0 + rs))
    elif avg_gain > 0:
        out[period] = 100.0
    else:
        out[period] = 50.0

    # Wilder's smoothing
    for i in range(period + 1, n):
        avg_gain = (avg_gain * (period - 1) + gains[i]) / period
        avg_loss = (avg_loss * (period - 1) + losses[i]) / period

        if avg_loss > 0:
            rs = avg_gain / avg_loss
            out[i] = 100.0 - (100.0 / (1.0 + rs))
        elif avg_gain > 0:
            out[i] = 100.0
        else:
            out[i] = 50.0

    return out


# ==================================================================
#
# 1. NUMBA UDF ライブラリ (Part 2: D-F, 2A)
#
# ==================================================================

# ----------------------------------------
# from engine_1_D_a_vast_universe_of_features.py
# ----------------------------------------


@njit(fastmath=True, cache=True)
def hv_robust_udf(returns: np.ndarray) -> float:
    """ロバストボラティリティ (Numba JIT)"""
    if len(returns) < 5:
        return np.nan
    median_return = np.median(returns)
    mad = np.median(np.abs(returns - median_return))
    return mad * 1.4826


@njit(fastmath=True, cache=True)
def chaikin_volatility_udf(
    high: np.ndarray, low: np.ndarray, window: int
) -> np.ndarray:
    n = len(high)
    result = np.full(n, np.nan)
    if n < window * 2:
        return result
    hl_range = high - low
    ema = np.zeros(n)
    alpha = 2.0 / (window + 1.0)
    ema[0] = hl_range[0]
    for i in range(1, n):
        ema[i] = alpha * hl_range[i] + (1.0 - alpha) * ema[i - 1]
    for i in range(window * 2 - 1, n):
        prev_avg = ema[i - window]
        if prev_avg > 0:
            result[i] = (ema[i] - prev_avg) / prev_avg * 100.0
    return result


@njit(fastmath=True, cache=True)
def mass_index_udf(high: np.ndarray, low: np.ndarray, window: int) -> np.ndarray:
    n = len(high)
    result = np.full(n, np.nan)
    if n < window + 18:
        return result
    hl_range = high - low
    alpha = 2.0 / 10.0  # 9-period EMA
    ema9 = np.zeros(n)
    ema_ema9 = np.zeros(n)
    ratio = np.zeros(n)
    ema9[0] = hl_range[0]
    for i in range(1, n):
        ema9[i] = alpha * hl_range[i] + (1.0 - alpha) * ema9[i - 1]
    ema_ema9[0] = ema9[0]
    for i in range(1, n):
        ema_ema9[i] = alpha * ema9[i] + (1.0 - alpha) * ema_ema9[i - 1]
    for i in range(n):
        ratio[i] = ema9[i] / ema_ema9[i] if ema_ema9[i] > 0 else 1.0
    for i in range(window - 1, n):
        result[i] = np.sum(ratio[i - window + 1 : i + 1])
    return result


@njit(fastmath=True, cache=True)
def cmf_udf(
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    volume: np.ndarray,
    window: int,
) -> np.ndarray:
    n = len(close)
    result = np.full(n, np.nan)
    mf_volume = np.zeros(n)
    vol = np.zeros(n)
    for i in range(n):
        if high[i] != low[i]:
            mf_multiplier = ((close[i] - low[i]) - (high[i] - close[i])) / (
                high[i] - low[i]
            )
            mf_volume[i] = mf_multiplier * volume[i]
            vol[i] = volume[i]
    for i in range(window - 1, n):
        mf_volume_sum = np.sum(mf_volume[i - window + 1 : i + 1])
        volume_sum = np.sum(vol[i - window + 1 : i + 1])
        result[i] = mf_volume_sum / volume_sum if volume_sum > 0 else 0.0
    return result


@njit(fastmath=True, cache=True)
def mfi_udf(
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    volume: np.ndarray,
    window: int,
) -> np.ndarray:
    n = len(close)
    result = np.full(n, np.nan)
    if n < window + 1:
        return result
    typical_price = (high + low + close) / 3.0
    raw_money_flow = typical_price * volume
    positive_flow = np.zeros(n)
    negative_flow = np.zeros(n)
    for i in range(1, n):
        if typical_price[i] > typical_price[i - 1]:
            positive_flow[i] = raw_money_flow[i]
        elif typical_price[i] < typical_price[i - 1]:
            negative_flow[i] = raw_money_flow[i]
    for i in range(window, n):
        pos_sum = np.sum(positive_flow[i - window + 1 : i + 1])
        neg_sum = np.sum(negative_flow[i - window + 1 : i + 1])
        if neg_sum > 0:
            money_ratio = pos_sum / neg_sum
            result[i] = 100.0 - (100.0 / (1.0 + money_ratio))
        elif pos_sum > 0:
            result[i] = 100.0
        else:
            result[i] = 50.0
    return result


@njit(fastmath=True, cache=True)
def obv_udf(close: np.ndarray, volume: np.ndarray) -> np.ndarray:
    n = len(close)
    result = np.zeros(n)
    if n == 0:
        return result
    result[0] = volume[0]
    for i in range(1, n):
        if close[i] > close[i - 1]:
            result[i] = result[i - 1] + volume[i]
        elif close[i] < close[i - 1]:
            result[i] = result[i - 1] - volume[i]
        else:
            result[i] = result[i - 1]
    return result


@njit(fastmath=True, cache=True)
def accumulation_distribution_udf(
    high: np.ndarray, low: np.ndarray, close: np.ndarray, volume: np.ndarray
) -> np.ndarray:
    n = len(close)
    result = np.zeros(n)
    if n == 0:
        return result
    for i in range(1, n):
        if high[i] != low[i]:
            clv = ((close[i] - low[i]) - (high[i] - close[i])) / (high[i] - low[i])
            result[i] = result[i - 1] + (clv * volume[i])
        else:
            result[i] = result[i - 1]
    return result


@njit(fastmath=True, cache=True)
def force_index_udf(close: np.ndarray, volume: np.ndarray) -> np.ndarray:
    n = len(close)
    result = np.zeros(n)
    if n < 2:
        return result
    for i in range(1, n):
        price_change = close[i] - close[i - 1]
        result[i] = price_change * volume[i]
    return result


@njit(fastmath=True, cache=True)
def candlestick_patterns_udf(
    open_prices: np.ndarray, high: np.ndarray, low: np.ndarray, close: np.ndarray
) -> np.ndarray:
    n = len(close)
    result = np.full(n, 0)
    for i in range(n):
        body_size = abs(close[i] - open_prices[i])
        upper_shadow = high[i] - max(open_prices[i], close[i])
        lower_shadow = min(open_prices[i], close[i]) - low[i]
        total_range = high[i] - low[i]
        if total_range <= 0:
            continue
        body_ratio = body_size / total_range
        upper_shadow_ratio = upper_shadow / total_range
        lower_shadow_ratio = lower_shadow / total_range
        if body_ratio < 0.1:
            result[i] = 3
        elif lower_shadow_ratio > 0.6 and upper_shadow_ratio < 0.1:
            result[i] = 1
        elif upper_shadow_ratio > 0.6 and lower_shadow_ratio < 0.1:
            result[i] = 2
        elif body_ratio > 0.6 and close[i] > open_prices[i]:
            result[i] = 4
        elif body_ratio > 0.6 and close[i] < open_prices[i]:
            result[i] = 5
    return result


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
    return np.std(finite_returns)


# ----------------------------------------
# from engine_1_E_a_vast_universe_of_features.py
# ----------------------------------------
@njit(fastmath=True, cache=True)
def numba_fft(x: np.ndarray) -> np.ndarray:
    n = x.shape[0]
    if (n & (n - 1)) != 0 and n > 0:
        target_n = 1 << int(np.ceil(np.log2(n)))
        padded_x = np.zeros(target_n, dtype=np.complex128)
        padded_x[:n] = x
        x = padded_x
        n = target_n
    else:
        x = x.astype(np.complex128)
    if n <= 1:
        return x
    j = 0
    for i in range(1, n):
        bit = n >> 1
        while j & bit:
            j ^= bit
            bit >>= 1
        j ^= bit
        if i < j:
            x[i], x[j] = x[j], x[i]
    len_ = 2
    while len_ <= n:
        half_len = len_ >> 1
        w_step = np.exp(-2j * np.pi / len_)
        for i in range(0, n, len_):
            w = 1.0 + 0.0j
            for j in range(half_len):
                u = x[i + j]
                v = x[i + j + half_len] * w
                x[i + j] = u + v
                x[i + j + half_len] = u - v
                w *= w_step
        len_ <<= 1
    return x


@njit(fastmath=True, cache=True)
def spectral_centroid_udf(signal: np.ndarray) -> float:
    if len(signal) < 10:
        return np.nan
    fft_values = numba_fft(signal)
    magnitude_spectrum = np.abs(fft_values[: len(signal) // 2])
    if len(magnitude_spectrum) == 0:
        return np.nan
    freqs = np.linspace(0, 0.5, len(magnitude_spectrum))
    total_magnitude = np.sum(magnitude_spectrum)
    return (
        np.sum(freqs * magnitude_spectrum) / total_magnitude
        if total_magnitude > 0
        else 0.0
    )


@njit(fastmath=True, cache=True)
def spectral_bandwidth_udf(signal: np.ndarray) -> float:
    if len(signal) < 10:
        return np.nan
    fft_values = numba_fft(signal)
    magnitude_spectrum = np.abs(fft_values[: len(signal) // 2])
    if len(magnitude_spectrum) == 0:
        return np.nan
    freqs = np.linspace(0, 0.5, len(magnitude_spectrum))
    total_magnitude = np.sum(magnitude_spectrum)
    if total_magnitude > 0:
        centroid = np.sum(freqs * magnitude_spectrum) / total_magnitude
        return np.sqrt(
            np.sum(((freqs - centroid) ** 2) * magnitude_spectrum) / total_magnitude
        )
    return 0.0


@njit(fastmath=True, cache=True)
def spectral_rolloff_udf(signal: np.ndarray) -> float:
    if len(signal) < 10:
        return np.nan
    fft_values = numba_fft(signal)
    power_spectrum = np.abs(fft_values[: len(signal) // 2]) ** 2
    if len(power_spectrum) == 0:
        return np.nan
    total_power = np.sum(power_spectrum)
    if total_power > 0:
        cumulative_power = np.cumsum(power_spectrum)
        threshold = 0.85 * total_power
        rolloff_idx = np.where(cumulative_power >= threshold)[0]
        if len(rolloff_idx) > 0:
            return rolloff_idx[0] / (2.0 * len(power_spectrum))
    return 0.0


@njit(fastmath=True, cache=True)
def spectral_flatness_udf(signal: np.ndarray) -> float:
    if len(signal) < 10:
        return np.nan
    fft_values = numba_fft(signal)
    magnitude_spectrum = np.abs(fft_values[: len(signal) // 2]) + 1e-10
    if len(magnitude_spectrum) == 0:
        return np.nan
    geometric_mean = np.exp(np.mean(np.log(magnitude_spectrum)))
    arithmetic_mean = np.mean(magnitude_spectrum)
    return geometric_mean / arithmetic_mean if arithmetic_mean > 0 else 0.0


@njit(fastmath=True, cache=True)
def spectral_entropy_udf(signal: np.ndarray) -> float:
    if len(signal) < 10:
        return np.nan
    fft_values = numba_fft(signal)
    power_spectrum = np.abs(fft_values[: len(signal) // 2]) ** 2
    if len(power_spectrum) == 0:
        return np.nan
    total_power = np.sum(power_spectrum)
    if total_power > 0:
        probability = power_spectrum / total_power
        entropy = 0.0
        for p in probability:
            if p > 1e-10:
                entropy -= p * np.log2(p)
        return entropy
    return 0.0


@nb.njit(fastmath=True, cache=True)
def wavelet_energy_udf(signal: np.ndarray, window_size: int) -> np.ndarray:
    """
    ウェーブレットエネルギー計算（動的並列化制御版）
    各レベルでの近似エネルギー
    """
    n = len(signal)
    result = np.full(n, np.nan)
    levels = 4

    if n < window_size:
        return result

    num_iterations = n - (window_size - 1)

    loop_range = (
        nb.prange(window_size - 1, n)
        if num_iterations > 2000
        else range(window_size - 1, n)
    )

    for i in loop_range:
        window_data = signal[i - window_size + 1 : i + 1]
        finite_data = window_data[np.isfinite(window_data)]

        if len(finite_data) < window_size // 2:
            continue

        # 簡易ウェーブレット変換（移動平均とディファレンス）
        level_energy = 0.0
        current_signal = finite_data.copy()

        for level in range(min(levels, 4)):
            if len(current_signal) < 4:
                break

            # ローパスフィルタ（移動平均）
            filtered = np.zeros(len(current_signal) // 2)
            for j in range(len(filtered)):
                idx = j * 2
                if idx + 1 < len(current_signal):
                    filtered[j] = (current_signal[idx] + current_signal[idx + 1]) / 2.0

            # エネルギー計算
            level_energy += np.sum(filtered**2)
            current_signal = filtered

        result[i] = level_energy

    return result


@njit(fastmath=True, cache=True)
def wavelet_entropy_udf(signal: np.ndarray) -> float:
    if len(signal) > 0:
        squared_data = signal**2
        return -np.sum(squared_data * np.log2(np.abs(squared_data) + 1e-10))
    return np.nan


@njit(fastmath=True, cache=True)
def hilbert_amplitude_udf(signal: np.ndarray) -> float:
    if len(signal) < 10:
        return np.nan
    fft_signal = numba_fft(signal)
    n_samples = len(signal)
    hilbert_fft = fft_signal.copy()
    for j in range(1, n_samples // 2):
        hilbert_fft[j] *= -1j
        hilbert_fft[n_samples - j] *= 1j
    hilbert_signal = np.real(hilbert_fft)  # (簡易版 IFFT)
    amplitude_envelope = np.sqrt(signal**2 + hilbert_signal**2)
    return np.mean(amplitude_envelope)


@njit(fastmath=True, cache=True)
def hilbert_phase_var_udf(signal: np.ndarray) -> float:
    if len(signal) > 1:
        analytic_signal = signal + 1j * np.roll(signal, 1)
        return np.var(np.angle(analytic_signal))
    return np.nan


@njit(fastmath=True, cache=True)
def hilbert_phase_stability_udf(signal: np.ndarray) -> float:
    if len(signal) > 2:
        if np.std(signal) < 1e-10:
            return 1.0
        analytic_signal = signal + 1j * np.roll(signal, 1)
        phase_diff_std = np.std(np.diff(np.angle(analytic_signal)))
        return 1.0 / (1.0 + phase_diff_std + 1e-10)
    return np.nan


@njit(fastmath=True, cache=True)
def hilbert_freq_mean_udf(signal: np.ndarray) -> float:
    if len(signal) > 2:
        analytic_signal = signal + 1j * np.roll(signal, 1)
        instant_freq = np.abs(np.diff(np.angle(analytic_signal)))
        return np.mean(instant_freq)
    return np.nan


@njit(fastmath=True, cache=True)
def hilbert_freq_std_udf(signal: np.ndarray) -> float:
    if len(signal) > 2:
        analytic_signal = signal + 1j * np.roll(signal, 1)
        instant_freq = np.abs(np.diff(np.angle(analytic_signal)))
        return np.std(instant_freq)
    return np.nan


@njit(fastmath=True, cache=True)
def acoustic_power_udf(signal: np.ndarray) -> float:
    if len(signal) > 0:
        return np.sqrt(np.mean(signal**2))
    return np.nan


@njit(fastmath=True, cache=True)
def acoustic_frequency_udf(signal: np.ndarray) -> float:
    if len(signal) > 1:
        zero_crossings = 0
        for j in range(1, len(signal)):
            if signal[j - 1] * signal[j] < 0:
                zero_crossings += 1
        return zero_crossings / (2.0 * len(signal))
    return np.nan


# ----------------------------------------
# from engine_1_F_a_vast_universe_of_features.py
# ----------------------------------------


@njit(fastmath=True, cache=True)
def rolling_network_density_udf(prices: np.ndarray) -> float:
    if len(prices) < 10:
        return np.nan
    window_n = len(prices)
    threshold = np.std(prices) * 0.5
    edge_count = 0
    max_possible_edges = window_n * (window_n - 1) / 2.0
    for j in range(window_n - 1):
        for k in range(j + 1, window_n):
            if abs(prices[j] - prices[k]) <= threshold:
                edge_count += 1
    return edge_count / max_possible_edges if max_possible_edges > 0 else 0.0


@njit(fastmath=True, cache=True)
def rolling_network_clustering_udf(prices: np.ndarray) -> float:
    if len(prices) < 15:
        return np.nan
    window_n = len(prices)
    threshold = np.std(prices) * 0.5
    adjacency = np.zeros((window_n, window_n), dtype=boolean)
    for j in range(window_n):
        for k in range(window_n):
            if j != k and abs(prices[j] - prices[k]) <= threshold:
                adjacency[j, k] = True
    total_clustering = 0.0
    valid_nodes = 0
    for j in range(window_n):
        neighbors = []
        for k in range(window_n):
            if adjacency[j, k]:
                neighbors.append(k)
        k_j = len(neighbors)
        if k_j < 2:
            continue
        neighbor_connections = 0
        for idx1 in range(k_j):
            for idx2 in range(idx1 + 1, k_j):
                if adjacency[neighbors[idx1], neighbors[idx2]]:
                    neighbor_connections += 1
        total_clustering += neighbor_connections / (k_j * (k_j - 1) / 2.0)
        valid_nodes += 1
    return total_clustering / valid_nodes if valid_nodes > 0 else 0.0


@njit(fastmath=True, cache=True)
def rolling_vocabulary_diversity_udf(prices: np.ndarray) -> float:
    if len(prices) < 10:
        return np.nan
    std_price = np.std(prices)
    mean_price = np.mean(prices)
    if std_price == 0:
        return 0.0
    n_bins = 10
    price_min = mean_price - 2 * std_price
    bin_width = (mean_price + 2 * std_price - price_min) / n_bins
    if bin_width <= 0:
        return 0.0
    used_vocabularies = set()
    for price in prices:
        bin_idx = int((price - price_min) / bin_width)
        used_vocabularies.add(max(0, min(bin_idx, n_bins - 1)))
    return len(used_vocabularies) / len(prices)


@njit(fastmath=True, cache=True)
def rolling_linguistic_complexity_udf(prices: np.ndarray) -> float:
    if len(prices) < 20:
        return np.nan
    price_changes = np.diff(prices)
    threshold = np.std(price_changes) * 0.1
    syntax_sequence = np.zeros(len(price_changes), dtype=np.int8)
    for i, change in enumerate(price_changes):
        if change > threshold:
            syntax_sequence[i] = 1
        elif change < -threshold:
            syntax_sequence[i] = -1
    if len(syntax_sequence) < 3:
        return 0.0
    bigrams = set()
    for j in range(len(syntax_sequence) - 1):
        bigrams.add((syntax_sequence[j], syntax_sequence[j + 1]))
    trigrams = set()
    for j in range(len(syntax_sequence) - 2):
        trigrams.add(
            (syntax_sequence[j], syntax_sequence[j + 1], syntax_sequence[j + 2])
        )
    max_bigrams = min(9, len(syntax_sequence) - 1)
    max_trigrams = min(27, len(syntax_sequence) - 2)
    if max_bigrams > 0 and max_trigrams > 0:
        return (len(bigrams) / max_bigrams + len(trigrams) / max_trigrams) / 2.0
    return 0.0


@njit(fastmath=True, cache=True)
def rolling_semantic_flow_udf(prices: np.ndarray) -> float:
    if len(prices) < 15:
        return np.nan
    window_n = len(prices)
    window_size_local = min(5, window_n // 3)
    if window_size_local == 0:
        return np.nan
    semantic_vectors = []
    for j in range(window_size_local, window_n - window_size_local):
        neighborhood = prices[j - window_size_local : j + window_size_local + 1]
        relative_positions = neighborhood - prices[j]
        semantic_vectors.append(
            np.array([np.mean(relative_positions), np.std(relative_positions)])
        )
    if len(semantic_vectors) < 2:
        return np.nan
    flow_continuity = 0.0
    valid_pairs = 0
    for j in range(len(semantic_vectors) - 1):
        vec1, vec2 = semantic_vectors[j], semantic_vectors[j + 1]
        norm1 = np.sqrt(np.sum(vec1**2))
        norm2 = np.sqrt(np.sum(vec2**2))
        if norm1 > 1e-10 and norm2 > 1e-10:
            flow_continuity += np.sum(vec1 * vec2) / (norm1 * norm2)
            valid_pairs += 1
    return (flow_continuity / valid_pairs + 1.0) / 2.0 if valid_pairs > 0 else 0.0


@njit(fastmath=True, cache=True)
def rolling_golden_ratio_adherence_udf(prices: np.ndarray) -> float:
    if len(prices) < 10:
        return np.nan
    golden_ratio = (1.0 + np.sqrt(5.0)) / 2.0
    local_window = min(8, len(prices) // 2)
    if local_window == 0:
        return np.nan
    adherence_scores = []
    for j in range(local_window, len(prices) - local_window):
        local_subwindow = prices[j - local_window : j + local_window + 1]
        local_high = np.max(local_subwindow)
        local_low = np.min(local_subwindow)
        if local_low > 0:
            ratio = local_high / local_low
            adherence_scores.append(
                1.0 / (1.0 + abs(ratio - golden_ratio) / golden_ratio)
            )
    return np.mean(np.array(adherence_scores)) if adherence_scores else np.nan


@njit(fastmath=True, cache=True)
def rolling_symmetry_measure_udf(prices: np.ndarray) -> float:
    if len(prices) < 20:
        return np.nan
    window_n = len(prices)
    center = window_n // 2
    left_half = prices[:center]
    right_half_reversed = prices[center + 1 if window_n % 2 == 1 else center :][::-1]
    min_len = min(len(left_half), len(right_half_reversed))
    if min_len < 5:
        return np.nan
    left_norm = left_half[-min_len:]
    right_norm = right_half_reversed[:min_len]
    mean_left = np.mean(left_norm)
    std_left = np.std(left_norm)
    mean_right = np.mean(right_norm)
    std_right = np.std(right_norm)
    if std_left < 1e-10 or std_right < 1e-10:
        return 0.0
    left_norm = (left_norm - mean_left) / std_left
    right_norm = (right_norm - mean_right) / std_right
    correlation = np.corrcoef(left_norm, right_norm)[0, 1]
    return (correlation + 1.0) / 2.0


@njit(fastmath=True, cache=True)
def rolling_aesthetic_balance_udf(prices: np.ndarray) -> float:
    if len(prices) < 15:
        return np.nan
    gradients = np.diff(prices)
    if len(gradients) < 5:
        return np.nan
    abs_gradients = np.abs(gradients)
    mean_grad = np.mean(abs_gradients)
    std_grad = np.std(abs_gradients)
    if std_grad <= 1e-10:
        return 1.0
    gentle_threshold = mean_grad - 0.5 * std_grad
    moderate_threshold = mean_grad + 0.5 * std_grad
    intense_threshold = mean_grad + 1.5 * std_grad
    gentle_count, moderate_count, intense_count = 0, 0, 0
    for grad in abs_gradients:
        if grad <= gentle_threshold:
            gentle_count += 1
        elif grad <= moderate_threshold:
            moderate_count += 1
        elif grad <= intense_threshold:
            intense_count += 1
    total_counted = gentle_count + moderate_count + intense_count
    if total_counted == 0:
        return 0.0
    balance_deviation = (
        abs(gentle_count / total_counted - 0.6)
        + abs(moderate_count / total_counted - 0.3)
        + abs(intense_count / total_counted - 0.1)
    ) / 2.0
    return max(0.0, 1.0 - balance_deviation)


@njit(fastmath=True, cache=True)
def rolling_tonality_udf(prices: np.ndarray) -> float:
    if len(prices) < 12:
        return np.nan
    price_changes = np.diff(prices)
    if len(price_changes) < 5:
        return np.nan
    std_change = np.std(price_changes)
    if std_change <= 1e-10:
        return 0.5
    normalized_changes = price_changes / std_change
    scale_degrees = np.zeros(12)
    for change in normalized_changes:
        degree_idx = int((change + 3.0) / 6.0 * 11.0)
        scale_degrees[max(0, min(degree_idx, 11))] += 1
    major_pattern = np.array([1, 0, 1, 0, 1, 1, 0, 1, 0, 1, 0, 1])
    minor_pattern = np.array([1, 0, 1, 1, 0, 1, 0, 1, 1, 0, 1, 0])
    if np.sum(scale_degrees) > 0:
        scale_distribution = scale_degrees / np.sum(scale_degrees)
        major_similarity = np.sum(scale_distribution * major_pattern)
        minor_similarity = np.sum(scale_distribution * minor_pattern)
        total_similarity = major_similarity + minor_similarity
        return major_similarity / total_similarity if total_similarity > 0 else 0.5
    return 0.5


@njit(fastmath=True, cache=True)
def rolling_rhythm_pattern_udf(prices: np.ndarray) -> float:
    if len(prices) < 20:
        return np.nan
    abs_changes = np.abs(np.diff(prices))
    mean_change = np.mean(abs_changes)
    strong_beats = abs_changes > mean_change
    pattern_strengths = []
    for period in range(2, min(8, len(strong_beats) // 3)):
        pattern_score = 0.0
        for j in range(period, len(strong_beats)):
            if strong_beats[j] == strong_beats[j - period]:
                pattern_score += 1.0
        pattern_strengths.append(pattern_score / (len(strong_beats) - period))
    return np.max(np.array(pattern_strengths)) if pattern_strengths else 0.0


@njit(fastmath=True, cache=True)
def rolling_harmony_udf(prices: np.ndarray) -> float:
    if len(prices) < 30:
        return np.nan
    window_n = len(prices)
    short_window, medium_window, long_window = (
        max(3, window_n // 15),
        max(5, window_n // 10),
        max(8, window_n // 6),
    )
    if long_window >= window_n:
        return np.nan

    def moving_average(data, w):
        n_data = len(data)
        ma = np.zeros(n_data - w + 1)
        for i in range(len(ma)):
            ma[i] = np.mean(data[i : i + w])
        return ma

    short_ma = moving_average(prices, short_window)
    medium_ma = moving_average(prices, medium_window)
    long_ma = moving_average(prices, long_window)

    min_len = min(len(short_ma), len(medium_ma), len(long_ma))
    if min_len < 5:
        return np.nan

    short_trend = np.diff(short_ma[-min_len:])
    medium_trend = np.diff(medium_ma[-min_len:])
    long_trend = np.diff(long_ma[-min_len:])

    harmony_scores = 0.0
    for j in range(len(short_trend)):
        signs = np.sign(np.array([short_trend[j], medium_trend[j], long_trend[j]]))
        if abs(np.sum(signs)) == 3:
            harmony_scores += 1.0
        elif abs(np.sum(signs)) == 1:
            harmony_scores += 0.5

    return harmony_scores / len(short_trend) if len(short_trend) > 0 else 0.0


@njit(fastmath=True, cache=True)
def rolling_musical_tension_udf(prices: np.ndarray) -> float:
    if len(prices) < 15:
        return np.nan
    price_changes = np.diff(prices)
    if len(price_changes) < 5:
        return np.nan
    local_window = min(5, len(price_changes) // 3)
    if local_window == 0:
        return np.nan
    tension_scores = []
    for j in range(local_window, len(price_changes) - local_window):
        local_changes = price_changes[j - local_window : j + local_window + 1]
        signs = np.sign(local_changes)
        direction_dissonance = np.sum(np.diff(signs) != 0) / len(signs)
        intensity_dissonance = np.max(np.abs(local_changes)) / (
            np.mean(np.abs(prices)) + 1e-10
        )
        tension_scores.append(
            min((direction_dissonance + intensity_dissonance) / 2.0, 1.0)
        )
    return np.mean(np.array(tension_scores)) if tension_scores else np.nan


@njit(fastmath=True, cache=True)
def rolling_kinetic_energy_udf(prices: np.ndarray) -> float:
    if len(prices) < 10:
        return np.nan
    velocities = np.diff(prices)
    if len(velocities) < 2:
        return np.nan
    masses = prices[1:]
    kinetic_energies = 0.5 * masses * velocities**2
    mean_kinetic_energy = np.mean(kinetic_energies)
    mean_price = np.mean(prices)
    return (
        mean_kinetic_energy / (mean_price**2)
        if mean_price > 1e-10
        else mean_kinetic_energy
    )


@njit(fastmath=True, cache=True)
def rolling_muscle_force_udf(prices: np.ndarray) -> float:
    if len(prices) < 15:
        return np.nan
    velocities = np.diff(prices)
    if len(velocities) < 2:
        return np.nan
    accelerations = np.diff(velocities)
    if len(accelerations) == 0:
        return np.nan
    masses = prices[2:]
    forces = masses * np.abs(accelerations)
    instantaneous_force = np.mean(forces)
    return (
        instantaneous_force / (np.mean(prices) ** 2)
        if np.mean(prices) > 1e-10
        else instantaneous_force
    )


@njit(fastmath=True, cache=True)
def rolling_biomechanical_efficiency_udf(prices: np.ndarray) -> float:
    if len(prices) < 20:
        return np.nan
    price_changes = np.diff(prices)
    total_displacement = np.sum(np.abs(price_changes))
    velocities = price_changes
    accelerations = np.diff(velocities) if len(velocities) > 1 else np.array([0.0])
    total_energy = np.sum(velocities**2) + np.sum(accelerations**2)
    if total_energy > 1e-10:
        return total_displacement / total_energy
    return 0.0


@njit(fastmath=True, cache=True)
def rolling_energy_expenditure_udf(prices: np.ndarray) -> float:
    if len(prices) < 15:
        return np.nan
    baseline_energy = np.var(prices)
    price_changes = np.diff(prices)
    movement_energy = np.sum(price_changes**2)
    accelerations = (
        np.diff(price_changes) if len(price_changes) > 1 else np.array([0.0])
    )
    acceleration_energy = np.sum(accelerations**2)
    total_energy = baseline_energy + movement_energy + acceleration_energy
    mean_price = np.mean(prices)
    return total_energy / (mean_price**2) if mean_price > 1e-10 else total_energy


# ----------------------------------------
# from engine_2_A_complexity_theory_F05_F15.py
# ----------------------------------------


@njit(fastmath=True, cache=True)
def polynomial_fit_detrend(y: np.ndarray, degree: int = 1) -> np.ndarray:
    """
    多項式フィッティングとトレンド除去
    最小二乗法による多項式フィット
    """
    n = len(y)
    if n < degree + 1:
        return np.zeros(n)

    # Vandermonde行列構築
    x = np.arange(n, dtype=np.float64)

    # 正規方程式による多項式係数推定
    # 簡略化のため、1次(線形)トレンドのみサポート
    if degree == 1:
        # 線形回帰: y = a*x + b
        x_mean = np.mean(x)
        y_mean = np.mean(y)

        numerator = 0.0
        denominator = 0.0
        for i in range(n):
            x_diff = x[i] - x_mean
            numerator += x_diff * (y[i] - y_mean)
            denominator += x_diff * x_diff

        if abs(denominator) < 1e-10:
            return y - y_mean

        slope = numerator / denominator
        intercept = y_mean - slope * x_mean

        # トレンド除去
        detrended = np.empty(n)
        for i in range(n):
            trend = slope * x[i] + intercept
            detrended[i] = y[i] - trend

        return detrended

    return y - np.mean(y)


@njit(fastmath=True, cache=True)
def mfdfa_core_single_window(
    prices: np.ndarray, q_values: np.ndarray, scales: np.ndarray, poly_degree: int = 1
) -> np.ndarray:
    """
    単一ウィンドウのMFDFA計算(ヘルパー関数)
    """
    n = len(prices)
    n_q = len(q_values)
    n_scales = len(scales)
    result = np.full(3, np.nan)
    if n < 20:
        return result
    mean_price = np.mean(prices)
    profile = np.zeros(n)
    cumsum = 0.0
    for i in range(n):
        cumsum += prices[i] - mean_price
        profile[i] = cumsum
    F_q = np.zeros((n_q, n_scales))
    for s_idx in range(n_scales):
        scale = int(scales[s_idx])
        if scale < 4 or scale >= n // 4:
            continue
        n_segments = n // scale
        segment_variances = np.zeros(n_segments)
        for seg in range(n_segments):
            start = seg * scale
            end = start + scale
            segment = profile[start:end]
            detrended = polynomial_fit_detrend(segment, poly_degree)
            variance = np.mean(detrended**2)
            segment_variances[seg] = variance
        for q_idx in range(n_q):
            q = q_values[q_idx]
            valid_vars = segment_variances[segment_variances > 1e-10]
            if len(valid_vars) == 0:
                continue
            if abs(q) < 1e-10:
                F_q[q_idx, s_idx] = np.exp(0.5 * np.mean(np.log(valid_vars)))
            else:
                F_q[q_idx, s_idx] = np.power(
                    np.mean(np.power(valid_vars, q / 2.0)), 1.0 / q
                )
    h_q_values = np.zeros(n_q)
    for q_idx in range(n_q):
        valid_indices = np.where(F_q[q_idx, :] > 1e-10)[0]
        if len(valid_indices) < 2:
            continue
        log_scales = np.log(scales[valid_indices])
        log_F_vals = np.log(F_q[q_idx, valid_indices])
        if len(log_scales) >= 2:
            x_arr = log_scales
            y_arr = log_F_vals
            x_mean = np.mean(x_arr)
            y_mean = np.mean(y_arr)
            numerator = np.sum((x_arr - x_mean) * (y_arr - y_mean))
            denominator = np.sum((x_arr - x_mean) ** 2)
            if abs(denominator) > 1e-10:
                h_q_values[q_idx] = numerator / denominator
    valid_h = h_q_values[np.isfinite(h_q_values) & (h_q_values != 0)]
    if len(valid_h) >= 2:
        result[0] = np.mean(valid_h)
        result[1] = np.max(valid_h) - np.min(valid_h)
        result[2] = np.max(valid_h)
    return result


@njit(fastmath=True, cache=True)
def mfdfa_rolling_udf(prices: np.ndarray, component_idx: int) -> float:
    """MFDFA (Numba JIT) - 単一ウィンドウ、最新値のみ"""
    window = len(prices)
    if window < 20:
        return np.nan
    q_values = np.array([-5.0, -3.0, -1.0, 0.0, 1.0, 3.0, 5.0])
    # (スケールは `final_feature_set_v2.txt` に合わせて調整)
    scales_1000 = np.array([10.0, 20.0, 50.0, 100.0, 200.0])
    scales_2500 = np.array([25.0, 50.0, 100.0, 250.0, 500.0])
    scales_5000 = np.array([50.0, 100.0, 250.0, 500.0, 1000.0])

    scales_to_use = scales_1000
    if window >= 5000:
        scales_to_use = scales_5000
    elif window >= 2500:
        scales_to_use = scales_2500

    mfdfa_result = mfdfa_core_single_window(
        prices, q_values, scales_to_use, poly_degree=1
    )
    return mfdfa_result[component_idx]


@njit(fastmath=True, cache=True)
def lempel_ziv_complexity(sequence: np.ndarray) -> float:
    """Lempel-Ziv複雑性計算(LZ76)"""
    n = len(sequence)
    if n < 2:
        return 0.0

    complexity = 1
    i = 0
    while i < n:
        max_match_length = 0
        for start in range(i):
            match_length = 0
            j = 0
            while (
                i + j < n and start + j < i and sequence[i + j] == sequence[start + j]
            ):
                match_length += 1
                j += 1
            if match_length > max_match_length:
                max_match_length = match_length

        if max_match_length == 0:
            complexity += 1
            i += 1
        else:
            complexity += 1
            i += max_match_length

    if n > 1:
        max_complexity = n / (np.log2(n) + 1e-10)
        normalized_complexity = (
            complexity / max_complexity if max_complexity > 0 else 0.0
        )
        return min(normalized_complexity, 1.0)

    return 0.0


@njit(fastmath=True, cache=True)
def kolmogorov_complexity_rolling_udf(prices: np.ndarray, component_idx: int) -> float:
    """Kolmogorov複雑性 (Numba JIT) - 単一ウィンドウ、最新値のみ"""
    window = len(prices)
    if window < 10:
        return np.nan

    # 2. 対数リターン
    returns = np.zeros(window - 1)
    for i in range(window - 1):
        if prices[i] > 1e-10 and prices[i + 1] > 1e-10:
            returns[i] = np.log(prices[i + 1] / prices[i])

    # 3. 標準化
    returns_std = np.std(returns)
    if returns_std < 1e-10:
        return 0.0
    standardized = (returns - np.mean(returns)) / returns_std

    # 4. バイナリ化(中央値)
    median_val = np.median(standardized)
    encoded = np.zeros(len(standardized), dtype=np.int32)
    for i in range(len(standardized)):
        encoded[i] = 1 if standardized[i] > median_val else 0

    # 5. LZ複雑性
    complexity_val = lempel_ziv_complexity(encoded)

    if component_idx == 0:
        return complexity_val
    elif component_idx == 1:
        return 1.0 - complexity_val  # 圧縮率
    elif component_idx == 2:
        # パターン多様性 (Numbaは `set` をサポートしないため `np.unique`)
        unique_count = len(np.unique(encoded))
        return unique_count / len(encoded) if len(encoded) > 0 else 0.0

    return np.nan  # 不明なインデックス


@nb.njit(fastmath=True, cache=True)
def statistical_kurtosis_numba(arr: np.ndarray) -> float:
    """ローリング尖度 (Numba JIT)"""
    n = len(arr)
    if n < 4:  # 尖度は最低4サンプル必要
        return np.nan

    mean_val = np.mean(arr)
    std_val = np.std(arr)

    if std_val < 1e-10:
        return 0.0

    m4 = np.mean((arr - mean_val) ** 4)
    kurtosis = m4 / (std_val**4) - 3.0  # 過剰尖度
    return kurtosis


# ==================================================================
# 2. リアルタイム特徴量エンジン 本体 (B案対応 V6.0)
# ==================================================================


@dataclass
class Signal:
    """
    リアルタイムエンジンが main.py に返すシグナルオブジェクト
    """

    features: np.ndarray  # 純化済み特徴量ベクトル (1, 304)
    timestamp: datetime  # シグナル発生時刻 (バーのクローズ時刻)
    timeframe: str  # シグナル発生の時間足 (e.g., "M1", "M15")
    market_info: Dict[str, Any]  # リスクエンジンに渡す市場文脈 (V4 R4ルール)


class RealtimeFeatureEngine:
    """
    【B案戦略 V6.0】
    矛盾①(純化)と矛盾②(マルチバッファ)を解決したリアルタイムエンジン。

    - 矛盾②解決: 15時間足の独立したNumpyバッファを保持。
    - M1バーを受け取り、Pandas.resample() を使って他の全時間足(M3..MN)の
      バッファをリアルタイムで更新（リサンプリング）する。
    - 矛盾①解決: 特徴量計算時に `05_alpha_decay_analyzer.py` の純化
      ロジックを適用する。
    """

    # main.py と同じ定義
    ALL_TIMEFRAMES = {
        "M1": mt5.TIMEFRAME_M1,
        "M3": mt5.TIMEFRAME_M3,
        "M5": mt5.TIMEFRAME_M5,
        "M8": mt5.TIMEFRAME_M8,
        "M15": mt5.TIMEFRAME_M15,
        "M30": mt5.TIMEFRAME_M30,
        "H1": mt5.TIMEFRAME_H1,
        "H4": mt5.TIMEFRAME_H4,
        "H6": mt5.TIMEFRAME_H6,
        "H12": mt5.TIMEFRAME_H12,
        "D1": mt5.TIMEFRAME_D1,
        "W1": mt5.TIMEFRAME_W1,
        "MN": mt5.TIMEFRAME_MN,
        "tick": None,
        "M0.5": None,
    }

    # Pandas.resample() のためのルール定義
    TF_RESAMPLE_RULES = {
        "M3": "3T",
        "M5": "5T",
        "M8": "8T",
        "M15": "15T",
        "M30": "30T",
        "H1": "1H",
        "H4": "4H",
        "H6": "6H",
        "H12": "12H",
        "D1": "1D",
        "W1": "1W",
        "MN": "1MS",  # 月初 (Month Start)
    }

    # OHLCVカラム
    OHLCV_COLS = ["open", "high", "low", "close", "volume"]

    # デフォルトのルックバック (名簿にない時間足用)
    DEFAULT_LOOKBACK = 200

    # R4判定用
    ATR_REGIME_CUTOFF = 5.0
    ATR_CALC_PERIOD = 21  # (e1c_atr_21_...) [cite_start][cite: 351]

    def __init__(self, feature_list_path: str = str(config.S3_FEATURES_FOR_TRAINING)):
        self.logger = logging.getLogger("ProjectForge.FeatureEngine")

        # [cite_start]1. 特徴量名簿(final_feature_set_v2.txt [cite: 351])をロード
        try:
            with open(feature_list_path, "r") as f:
                self.feature_list: List[str] = [
                    line.strip() for line in f if line.strip()
                ]
            self.logger.info(
                f"特徴量名簿 ({len(self.feature_list)}個) をロードしました。"
            )
        except Exception as e:
            self.logger.critical(f"特徴量名簿 {feature_list_path} のロードに失敗: {e}")
            raise  # 起動時エラーとしてスロー

        # 2. 名簿から各時間足の最大ルックバック期間を特定 (矛盾②)
        self.lookbacks_by_tf = self._parse_feature_list_and_get_lookbacks(
            self.feature_list
        )

        # 3. 独立したデータバッファを初期化 (矛盾②)
        self.data_buffers: Dict[str, Dict[str, Deque[float]]] = {}
        self.is_buffer_filled: Dict[str, bool] = {}
        self.last_bar_timestamps: Dict[str, Optional[pd.Timestamp]] = {}

        for tf_name in self.ALL_TIMEFRAMES.keys():
            if self.ALL_TIMEFRAMES[tf_name] is None:
                continue  # tick, M0.5 はスキップ

            # この時間足で計算すべき特徴量があるか？
            if tf_name not in self.lookbacks_by_tf:
                self.logger.debug(
                    f"時間足 {tf_name} は特徴量名簿にないためスキップします。"
                )
                continue

            lookback = self.lookbacks_by_tf[tf_name]
            self.logger.info(f"  -> {tf_name} バッファ初期化 (Lookback: {lookback})")

            # 各OHLCVカラムのDequeを作成
            self.data_buffers[tf_name] = {
                col: deque(maxlen=lookback) for col in self.OHLCV_COLS
            }
            # 充填状態を初期化
            self.is_buffer_filled[tf_name] = False
            # 最終タイムスタンプ (リサンプリングの基準)
            self.last_bar_timestamps[tf_name] = None

        # 4. M1データを保持するDeque (リサンプリング元)
        # (pd.concatによるメモリコピー地獄を回避するため)
        # MFDFA(5000) * D1(1440) + マージン
        max_m1_bars_needed = max(self.lookbacks_by_tf.values()) * 1440 + 1000
        self.m1_dataframe: Deque[Dict[str, Any]] = deque(maxlen=max_m1_bars_needed)

        # --- [V6.5 修正: 問題3 ボトルネック解決] ---
        # 5. 純化(OLS)用 状態保持バッファ
        # 5a. 純化対象5特徴量の「履歴」を保持するDeque
        self.proxy_feature_buffers: Dict[str, Dict[str, Deque[float]]] = {}
        # 5b. OLSの逐次計算用パラメータ (Sum(x), Sum(y), Sum(xy), ...)
        self.ols_state: Dict[str, Dict[str, Dict[str, float]]] = {}

        PROXY_FEATURES = [
            "atr",
            "log_return",
            "price_momentum",
            "rolling_volatility",
            "volume_ratio",
        ]

        for tf_name in self.data_buffers.keys():
            lookback = self.lookbacks_by_tf[tf_name]
            # 5a. Dequeを初期化
            self.proxy_feature_buffers[tf_name] = {
                feat: deque(maxlen=lookback) for feat in PROXY_FEATURES
            }
            # 5b. OLS状態を初期化
            self.ols_state[tf_name] = {}
            for feat in PROXY_FEATURES:
                self.ols_state[tf_name][feat] = {
                    "sum_x": 0.0,
                    "sum_y": 0.0,
                    "sum_xy": 0.0,
                    "sum_x_sq": 0.0,
                    "sum_y_sq": 0.0,
                    "count": 0.0,
                }
        # --- [V6.5 修正 ここまで] ---

        self.logger.info(f"M1 Dequeバッファを初期化 (maxlen: {max_m1_bars_needed})")

    def _parse_feature_list_and_get_lookbacks(
        self, feature_list: List[str]
    ) -> Dict[str, int]:
        """
        [新規] (矛盾②解決)
        [cite_start]`final_feature_set_v2.txt` [cite: 351] を解析し、時間足ごとに
        必要な最大ルックバック期間（Numpyバッファの長さ）を決定する。
        """
        self.logger.info("特徴量名簿を解析し、時間足ごとの最大ルックバックを計算中...")

        # (e.g., "M1", "M3", ..., "D1", "MN", "tick")
        tf_pattern = re.compile(r"_(M[0-9\.]+|H[0-9]+|D[0-9]+|W[0-9]+|MN|tick)$")

        # (e.g., "_100_", "_50_")
        window_pattern = re.compile(r"_(\d+)_")

        lookbacks: Dict[str, int] = {}

        for feature_name in feature_list:
            # 1. 時間足の特定 (e.g., "D1")
            tf_match = tf_pattern.search(feature_name)
            if not tf_match:
                # self.logger.warning(f"特徴量名 {feature_name} から時間足を特定できません。")
                continue

            tf_name = tf_match.group(1)

            # 2. ウィンドウサイズの特定 (e.g., 50)
            # MFDFA/Kolmogorovは特別なウィンドウサイズを持つ
            if "e2a_mfdfa" in feature_name or "e2a_kolmogorov" in feature_name:
                if "5000" in feature_name:
                    window = 5000
                elif "2500" in feature_name:
                    window = 2500
                elif "1500" in feature_name:
                    window = 1500
                elif "1000" in feature_name:
                    window = 1000
                elif "500" in feature_name:
                    window = 500
                else:
                    window = self.DEFAULT_LOOKBACK
            else:
                # 通常の特徴量 (e.g., _50_)
                window_matches = window_pattern.findall(feature_name)
                if not window_matches:
                    window = 100  # デフォルト (e.g., adf_statistic)
                else:
                    window = max(int(w) for w in window_matches)

            # ADXなどは period*2 が必要
            if "adx" in feature_name:
                window *= 2
            # [cite_start]Mass Index は period + 18  [cite: 1-1310]
            if "mass_index" in feature_name:
                window += 18

            # 3. 最大ルックバックの更新
            current_max = lookbacks.get(tf_name, 0)
            lookbacks[tf_name] = max(current_max, window)

        # 4. R4判定用のATRルックバックを追加
        atr_lookback = self.ATR_CALC_PERIOD + 2  # (マージン)
        for tf_name in self.ALL_TIMEFRAMES.keys():
            if tf_name in lookbacks:  # 特徴量計算がある時間足のみ
                current_max = lookbacks.get(tf_name, 0)
                lookbacks[tf_name] = max(current_max, atr_lookback)

        # 5. 最終マージンを追加
        final_lookbacks = {}
        for tf_name, lookback in lookbacks.items():
            final_lookbacks[tf_name] = lookback + 5  # 予備バッファ
            self.logger.info(
                f"  -> {tf_name} 最大ルックバック: {final_lookbacks[tf_name]}"
            )

        return final_lookbacks

    def get_max_lookback_for_all_timeframes(self) -> Dict[str, int]:
        """[I/F] main.py が履歴データを取得するためにルックバック一覧を返す"""
        return self.lookbacks_by_tf

    def is_all_buffers_filled(self) -> bool:
        """[I/F] main.py が全バッファの充填完了を確認するために使用"""
        # [cite_start]lookbacks_by_tf (名簿[cite: 351]由来) に存在する時間足のみチェック
        for tf_name in self.lookbacks_by_tf.keys():
            if not self.is_buffer_filled.get(tf_name, False):
                self.logger.warning(f"バッファ {tf_name} はまだ充填されていません。")
                return False
        return True

    def _buffer_to_dataframe(self, tf_name: str) -> pd.DataFrame:
        """
        [V6.5 修正]
        指定された時間足のDequeバッファをPandas DataFrameに変換する。
        (ボトルネックだった純化用特徴量の履歴計算は削除)
        """
        df = pd.DataFrame(self.data_buffers[tf_name])

        # DequeはNumpy配列と違ってタイムスタンプを持っていないため、
        # 最後に記録したタイムスタンプから逆算してIndexを生成する
        last_ts = self.last_bar_timestamps[tf_name]
        if last_ts is None:
            raise ValueError(f"バッファ {tf_name} のタイムスタンプがありません。")

        freq_map = {
            "M1": "1T",
            "M3": "3T",
            "M5": "5T",
            "M8": "8T",
            "M15": "15T",
            "M30": "30T",
            "H1": "1H",
            "H4": "4H",
            "H6": "6H",
            "H12": "12H",
            "D1": "1D",
            "W1": "1W",
            "MN": "1MS",
        }
        freq = freq_map.get(tf_name, "1T")

        # タイムスタンプインデックスを生成
        timestamps = pd.date_range(
            end=last_ts, periods=len(self.data_buffers[tf_name]["close"]), freq=freq
        )
        df["timestamp"] = timestamps

        return df.set_index("timestamp")

    def _replace_buffer_from_dataframe(
        self,
        tf_name: str,
        df: pd.DataFrame,
        market_proxy_cache: Optional[pd.DataFrame] = None,
    ) -> None:
        """
        [V6.5 修正]
        DataFrameからバッファを充填する。
        - OHLCVバッファ (既存)
        - 純化用特徴量バッファ (新規)
        - OLS状態 (新規)
        """
        if tf_name not in self.data_buffers:
            self.logger.warning(f"_replace_buffer: {tf_name} は管理対象外です。")
            return

        buffer_len = self.lookbacks_by_tf[tf_name]
        df_slice = df.iloc[-buffer_len:]

        # 1. OHLCVバッファを充填
        for col in self.OHLCV_COLS:
            self.data_buffers[tf_name][col].clear()
            self.data_buffers[tf_name][col].extend(df_slice[col].values)
        self.last_bar_timestamps[tf_name] = df_slice.index[-1]
        if len(df_slice) >= buffer_len:
            self.is_buffer_filled[tf_name] = True

        self.logger.info(
            f"  -> {tf_name} OHLCVバッファを {len(df_slice)} 行で充填しました。"
        )

        # 2. [V6.5] 純化用バッファとOLS状態をバックフィル
        if market_proxy_cache is None or market_proxy_cache.empty:
            self.logger.warning(
                f"  -> {tf_name} OLSバックフィルスキップ (プロキシ未提供)"
            )
            return

        # 2a. 純化対象5特徴量の「全履歴」を計算
        # (この計算は起動時に1回だけ実行される)
        close_arr = df_slice["close"].to_numpy(dtype=np.float64)
        high_arr = df_slice["high"].to_numpy(dtype=np.float64)
        low_arr = df_slice["low"].to_numpy(dtype=np.float64)
        volume_arr = df_slice["volume"].to_numpy(dtype=np.float64)

        # 5特徴量のDFを作成
        proxy_feat_df = pd.DataFrame(index=df_slice.index)
        proxy_feat_df["atr"] = calculate_atr_numba(high_arr, low_arr, close_arr, 13)

        pct = np.full_like(close_arr, np.nan)
        if len(close_arr) >= 2:
            safe_denominator_pct = close_arr[:-1].copy()
            safe_denominator_pct[safe_denominator_pct == 0] = 1e-10
            pct_calc = np.diff(close_arr) / safe_denominator_pct
            pct[1:] = pct_calc
        close_pct = pct

        proxy_feat_df["log_return"] = np.concatenate(
            ([np.nan], np.log(close_arr[1:] / (close_arr[:-1] + 1e-10)))
        )
        proxy_feat_df["price_momentum"] = (
            df_slice["close"].diff(10).to_numpy()
        )  # 10期間
        proxy_feat_df["rolling_volatility"] = (
            pd.Series(close_pct).rolling(window=20).std().to_numpy()
        )
        proxy_feat_df["volume_ratio"] = volume_arr / (
            pd.Series(volume_arr).rolling(window=20).mean().to_numpy() + 1e-10
        )

        # 2b. 市場プロキシ (x) をAsof-Join
        aligned_df = proxy_feat_df.join(market_proxy_cache, how="left").ffill()
        aligned_df = aligned_df.fillna(0.0)  # NaNを0で初期化

        # 2c. バッファとOLS状態を充填
        for feat_name in self.proxy_feature_buffers[tf_name].keys():
            if feat_name == "market_proxy":
                continue  # 'atr' ループで処理

            y_history = aligned_df[feat_name].to_numpy()
            x_history = aligned_df["market_proxy"].to_numpy()

            # Deque充填
            self.proxy_feature_buffers[tf_name][feat_name].clear()
            self.proxy_feature_buffers[tf_name][feat_name].extend(y_history)
            if feat_name == "atr":
                self.proxy_feature_buffers[tf_name]["market_proxy"].clear()
                self.proxy_feature_buffers[tf_name]["market_proxy"].extend(x_history)

            # OLS状態を計算 (全履歴)
            state = self.ols_state[tf_name][feat_name]
            state["sum_x"] = np.sum(x_history)
            state["sum_y"] = np.sum(y_history)
            state["sum_xy"] = np.sum(x_history * y_history)
            state["sum_x_sq"] = np.sum(x_history**2)
            state["sum_y_sq"] = np.sum(y_history**2)
            state["count"] = float(len(x_history))

        self.logger.info(f"  -> {tf_name} 純化用バッファとOLS状態を充填しました。")

    def fill_all_buffers(
        self,
        history_data_map: Dict[str, pl.DataFrame],
        market_proxy_cache: pd.DataFrame,
    ) -> None:
        """
        [I/F] main.py から呼び出され、起動時に全バッファを履歴データで充填する
        """
        self.logger.info("全時間足の履歴データでNumpyバッファを一括充填中...")

        # M1データをPandas DFに変換して保存 (リサンプリング元)
        if "M1" not in history_data_map:
            raise ValueError("履歴データに M1 がありません。リサンプリングできません。")

        m1_history_pl = history_data_map["M1"]
        m1_history_pd = m1_history_pl.to_pandas()
        m1_history_pd["timestamp"] = pd.to_datetime(
            m1_history_pd["time"], unit="s", utc=True
        )
        m1_history_pd = m1_history_pd.set_index("timestamp").drop(columns="time")

        # M1バッファを充填 (Deque)
        self._replace_buffer_from_dataframe("M1", m1_history_pd)
        # M1のグローバルDequeを更新 (pd.concat回避)
        self.m1_dataframe.clear()
        # (m1_history_pd は timestamp が index になっている)
        m1_records = m1_history_pd.reset_index().to_dict("records")
        self.m1_dataframe.extend(m1_records)

        # M1以外の時間足のバッファを「リサンプリング」によって生成・充填
        self.logger.info("M1データから他の全時間足をリサンプリングして充填中...")
        for tf_name, rule in self.TF_RESAMPLE_RULES.items():
            if tf_name not in self.data_buffers:
                continue  # この時間足は不要

            try:
                # Pandasリサンプリング
                resampled_df = (
                    # --- [V6.4 修正] ---
                    # 誤: self.m1_dataframe (deque)
                    # 正: m1_history_pd (L1237で生成したPandas DF)
                    m1_history_pd.resample(rule)
                    # --- [修正ここまで] ---
                    .agg(
                        {
                            "open": "first",
                            "high": "max",
                            "low": "min",
                            "close": "last",
                            "volume": "sum",
                        }
                    )
                    .dropna()
                )  # 取引時間外のNaNを削除

                if resampled_df.empty:
                    self.logger.warning(f"{tf_name} のリサンプリング結果が空です。")
                    continue

                # バッファを置換 (V6.5: OLSバックフィルも実行)
                self._replace_buffer_from_dataframe(
                    tf_name, resampled_df, market_proxy_cache
                )

            except Exception as e:
                self.logger.error(f"{tf_name} のリサンプリング充填に失敗: {e}")

        self.logger.info("✓ 全バッファの初期充填が完了しました。")

    def _append_bar_to_buffer(
        self,
        tf_name: str,
        bar_df: pd.DataFrame,
        market_proxy_cache: pd.DataFrame,
    ) -> None:
        """
        [V6.5 修正]
        バッファに新しいバー (DataFrame形式) を追加し、
        純化(OLS)状態を逐次更新する。
        """
        if tf_name not in self.data_buffers:
            return  # 管理対象外

        try:
            bar_dict = bar_df.iloc[0].to_dict()
            bar_timestamp = bar_df.index[0]

            # 1. OHLCVバッファを更新
            for col in self.OHLCV_COLS:
                self.data_buffers[tf_name][col].append(bar_dict[col])
            self.last_bar_timestamps[tf_name] = bar_timestamp

            # 2. 純化用5特徴量の「最新値」を計算
            # (この時点では OHLCV Deque は更新済み)
            latest_proxy_features = self._calculate_proxy_features_incremental(
                tf_name, bar_df
            )

            # 3. OLS状態を逐次更新
            if latest_proxy_features:
                self._update_incremental_ols(
                    tf_name,
                    latest_proxy_features,
                    market_proxy_cache,
                    bar_timestamp,
                )

            # 4. 充填状態を更新
            if (
                not self.is_buffer_filled[tf_name]
                and len(self.data_buffers[tf_name]["close"])
                >= self.lookbacks_by_tf[tf_name]
            ):
                self.is_buffer_filled[tf_name] = True
                self.logger.info(f"✅ {tf_name} バッファが充填されました。")

        except KeyError as e:
            self.logger.error(f"バーデータ {tf_name} にキーがありません: {e}")
        except Exception as e:
            self.logger.error(f"バー {tf_name} の追加に失敗: {e}")

    def _resample_and_update_buffer(
        self, tf_name: str, rule: str, market_proxy_cache: pd.DataFrame
    ) -> List[pd.Timestamp]:
        """
        [Helper] (リサンプリングロジック)
        M1 DequeをDFに変換してリサンプリングし、新しいバーが生成されていたら
        対象のバッファに追加し、新バーのタイムスタンプを返す。

        [V6.2 修正] ...
        [V6.3 修正案]
        - (Issue 1) pd.DataFrame(list(self.m1_dataframe)) の
          致命的ボトルネックを修正。
          Dequeを逆から探索し、必要なデータ「だけ」を抽出してからDF化する。
        - (Issue 2) ...
        """
        try:
            # 2. 最後に処理したタイムスタンプを取得
            last_known_timestamp = self.last_bar_timestamps.get(tf_name)
            if last_known_timestamp is None:
                self.logger.warning(
                    f"{tf_name} の最終時刻が不明です。リサンプリングをスキップします。"
                )
                return []

            # --- [V6.3 ボトルネック修正] ---
            # 1. Dequeから必要なデータ「だけ」を抽出
            #    (720万行のDF化を回避)
            new_m1_bars_for_resampling = []
            for bar in reversed(self.m1_dataframe):
                bar_ts = bar["timestamp"]
                if bar_ts >= last_known_timestamp:
                    # 最後に知った時刻「以降」のバーをすべて集める
                    new_m1_bars_for_resampling.append(bar)
                else:
                    # 最後に知った時刻のバー自体もリサンプリングの「土台」として必要
                    new_m1_bars_for_resampling.append(bar)
                    break  # これ以上古いバーは不要

            # 順序を元に戻す (時系列順)
            new_m1_bars_for_resampling.reverse()

            if len(new_m1_bars_for_resampling) < 2:
                # (土台 + 新規バー) が最低2件ないとリサンプリングできない
                return []

            # 2. 「小さなリスト」からDFを生成
            new_m1_data = pd.DataFrame(new_m1_bars_for_resampling).set_index(
                "timestamp"
            )
            # --- [修正 ここまで] ---

            # --- [パフォーマンス(Issue 3) 修正済み] ---
            # 3. この「小さなDF」だけをリサンプリング
            # (V6.2の L1127-L1130 は、上記の処理に統合されたため不要)
            resampled_df = (
                new_m1_data.resample(rule)
                .agg(
                    {
                        "open": "first",
                        "high": "max",
                        "low": "min",
                        "close": "last",
                        "volume": "sum",
                    }
                )
                .dropna()
            )
            # --- [修正 ここまで] ---

            # --- [V6.2 Issue 2 修正] ---
            # (リサンプリングバグの修正)
            if len(resampled_df) < 2:
                # 少なくとも1つの確定バーと1つの形成中バーが必要
                return []

            # 4. 確定したバーのみを抽出 (形成中 = 最後の行 を除外)
            newly_closed_bars = resampled_df.iloc[:-1]

            # 5. 最後に処理した時刻よりも新しい「確定バー」のみを抽出
            new_bars = newly_closed_bars[newly_closed_bars.index > last_known_timestamp]
            # --- [修正 ここまで] ---

            if new_bars.empty:
                return []  # 新しいバーはまだ確定していない

            new_bar_timestamps = []

            # 6. 新しいバーをバッファに追加
            for timestamp, row in new_bars.iterrows():
                # (V6.5: _append_bar_to_buffer は DF を期待する)
                bar_df = pd.DataFrame(row).T
                bar_df.index = [timestamp]
                bar_df.index.name = "timestamp"

                self._append_bar_to_buffer(tf_name, bar_df, market_proxy_cache)
                new_bar_timestamps.append(timestamp)

            if new_bar_timestamps:
                self.logger.debug(
                    f"  -> {tf_name} バッファに {len(new_bars)} 件の確定バーを追加しました。"
                )
            return new_bar_timestamps

        except Exception as e:
            self.logger.error(
                f"{tf_name} のリサンプリング更新に失敗: {e}", exc_info=True
            )
            return []

    def process_new_m1_bar(
        self, m1_bar: Dict[str, Any], market_proxy_cache: pd.DataFrame
    ) -> List[Signal]:
        """
        [I/F] main.py から M1 バーを受け取り、全バッファを更新し、
        シグナルをチェックして返す。

        Args:
            m1_bar: MT5から取得した最新のM1バー (dict)
            market_proxy_cache: [矛盾①解決] main.pyがロードしたM5市場プロキシDF

        Returns:
            シグナルオブジェクトのリスト
        """
        signal_list: List[Signal] = []

        try:
            # 1. M1バッファに新しいバーを追加
            m1_timestamp = m1_bar["timestamp"]

            # Pandas Deque (リサンプリング元) に追加 (pd.concat回避)
            self.m1_dataframe.append(m1_bar)

            # Dequeバッファ (M1特徴量計算用) に追加
            # (V6.5: _append_bar_to_buffer は DF とプロキシを期待する)
            m1_bar_df = pd.DataFrame([m1_bar]).set_index("timestamp")
            self._append_bar_to_buffer("M1", m1_bar_df, market_proxy_cache)

            # 2. M1以外の全時間足バッファをリサンプリング更新
            #    (新しくバーが確定した時間足のリストを取得)
            newly_closed_timeframes: Dict[str, List[pd.Timestamp]] = {}
            for tf_name, rule in self.TF_RESAMPLE_RULES.items():
                if tf_name not in self.data_buffers:
                    continue

                # (V6.5) OLS逐次更新のためプロキシを渡す
                new_timestamps = self._resample_and_update_buffer(
                    tf_name, rule, market_proxy_cache
                )
                if new_timestamps:
                    newly_closed_timeframes[tf_name] = new_timestamps

            # M1もチェック対象に含める
            newly_closed_timeframes["M1"] = [m1_timestamp]

            # 3. 新しくバーが確定した各時間足についてシグナルをチェック
            for tf_name, timestamps in newly_closed_timeframes.items():
                for timestamp in timestamps:
                    # 3a. R4レジームフィルターをチェック
                    r4_check_result = self._check_for_signal(tf_name, timestamp)

                    if r4_check_result["is_r4"]:
                        # 3b. [矛盾①解決] R4の場合、純化済み特徴量ベクトルを計算
                        # (純化には M5 プロキシキャッシュが必要)
                        feature_vector = self.calculate_feature_vector(
                            tf_name, timestamp, market_proxy_cache
                        )

                        if feature_vector is not None:
                            # 3c. シグナルオブジェクトを作成
                            signal = Signal(
                                features=feature_vector,
                                timestamp=timestamp,
                                timeframe=tf_name,
                                # V4 R4ルールを market_info に詰める
                                market_info=r4_check_result["market_info"],
                            )
                            signal_list.append(signal)

            # 4. メモリ管理: (V6.2) M1 Deque は maxlen で自動トリムされるため不要

            return signal_list

        except Exception as e:
            self.logger.error(f"process_new_m1_bar でエラー: {e}", exc_info=True)
            return []

    def _check_for_signal(self, tf_name: str, timestamp: datetime) -> Dict[str, Any]:
        """
        [Helper] (V4戦略)
        指定された時間足のバッファがR4レジーム (atr_value >= 5.0) かを判定する。
        """
        if not self.is_buffer_filled[tf_name]:
            return {"is_r4": False, "reason": "buffer_not_filled"}

        try:
            # R4判定用に、最新のATR(21)を計算
            data = {
                "high": np.array(self.data_buffers[tf_name]["high"], dtype=np.float64),
                "low": np.array(self.data_buffers[tf_name]["low"], dtype=np.float64),
                "close": np.array(
                    self.data_buffers[tf_name]["close"], dtype=np.float64
                ),
            }

            # create_proxy_labels_polars_patch_regime.py (L:149) のロジック
            atr_21_arr = calculate_atr_numba(
                data["high"], data["low"], data["close"], self.ATR_CALC_PERIOD
            )
            atr_value = atr_21_arr[-1]

            if np.isnan(atr_value):
                return {"is_r4": False, "reason": "atr_is_nan"}

            # R4判定
            if atr_value >= self.ATR_REGIME_CUTOFF:
                # V4ルールブック (create_proxy_labels... L:50-55) の値をセット
                market_info = {
                    "atr_value": atr_value,
                    "current_price": data["close"][-1],
                    "pt_multiplier": 1.0,
                    "sl_multiplier": 5.0,
                    "payoff_ratio": 1.0 / 5.0,
                    "direction": 1,  # (V4はLongオンリー戦略)
                }

                self.logger.info(
                    f"  -> R4 Signal Check ({tf_name} @ {timestamp.strftime('%H:%M')}): "
                    f"PASSED (ATR: {atr_value:.2f} >= {self.ATR_REGIME_CUTOFF})"
                )
                return {"is_r4": True, "market_info": market_info}
            else:
                return {"is_r4": False, "reason": "not_r4_regime"}

        except Exception as e:
            self.logger.warning(f"_check_for_signal ({tf_name}) でエラー: {e}")
            return {"is_r4": False, "reason": "atr_calculation_error"}

    def _calculate_proxy_features_incremental(
        self, tf_name: str, ohlcv_df: pd.DataFrame
    ) -> Dict[str, float]:
        """
        [V6.5 新規]
        指定されたDF (通常は1行の最新バー) から、
        純化対象の5特徴量の「最新値」のみを計算する。
        """
        if ohlcv_df.empty:
            return {}

        # バッファ全体を取得 (計算用)
        ohlcv_buffer = self.data_buffers[tf_name]
        lookback = self.lookbacks_by_tf[tf_name]

        # 最新のOHLCV値 (Numpy配列として)
        close_arr = np.array(ohlcv_buffer["close"], dtype=np.float64)
        high_arr = np.array(ohlcv_buffer["high"], dtype=np.float64)
        low_arr = np.array(ohlcv_buffer["low"], dtype=np.float64)
        volume_arr = np.array(ohlcv_buffer["volume"], dtype=np.float64)

        if len(close_arr) < 2:
            return {}  # 履歴が足りない

        latest_features = {}

        # 1. atr (13)
        # (calculate_atr_numba は配列全体を必要とする)
        atr_13_arr = calculate_atr_numba(high_arr, low_arr, close_arr, 13)
        latest_features["atr"] = atr_13_arr[-1] if len(atr_13_arr) > 0 else np.nan

        # 2. log_return
        safe_close_prev = close_arr[-2]
        if safe_close_prev == 0:
            safe_close_prev = 1e-10
        latest_features["log_return"] = np.log(close_arr[-1] / safe_close_prev)

        # 3. price_momentum (10)
        if len(close_arr) > 10:
            latest_features["price_momentum"] = close_arr[-1] - close_arr[-11]
        else:
            latest_features["price_momentum"] = np.nan

        # 4. rolling_volatility (20)
        if len(close_arr) > 20:
            # pct_change を計算
            safe_denominator_pct = close_arr[-21:-1]  # 過去20個のリターン
            safe_denominator_pct[safe_denominator_pct == 0] = 1e-10
            pct_calc = np.diff(close_arr[-21:]) / safe_denominator_pct
            latest_features["rolling_volatility"] = np.std(pct_calc)
        else:
            latest_features["rolling_volatility"] = np.nan

        # 5. volume_ratio (20)
        if len(volume_arr) > 20:
            vol_mean_20 = np.mean(volume_arr[-20:])
            latest_features["volume_ratio"] = volume_arr[-1] / (vol_mean_20 + 1e-10)
        else:
            latest_features["volume_ratio"] = np.nan

        return latest_features

    def _update_incremental_ols(
        self,
        tf_name: str,
        latest_proxy_features: Dict[str, float],
        market_proxy_cache: pd.DataFrame,
        timestamp: datetime,
    ):
        """
        [V6.5 新規]
        純化対象の5特徴量について、OLS状態(sum_x, sum_y...)を
        逐次更新する。
        """
        try:
            # 1. 最新の市場プロキシ値 (x) を取得
            # (Pandas 2.0+ では get_indexer +iloc が高速)
            idx = market_proxy_cache.index.get_indexer([timestamp], method="ffill")[0]
            latest_x = market_proxy_cache.iloc[idx]["market_proxy"]
            if np.isnan(latest_x):
                latest_x = 0.0  # (ffillでもNaNの場合は0)

            for feat_name, latest_y in latest_proxy_features.items():
                if np.isnan(latest_y):
                    latest_y = 0.0  # (特徴量NaNは0)

                # 2. 対応するOLS状態を取得
                state = self.ols_state[tf_name][feat_name]
                buffer_len = self.lookbacks_by_tf[tf_name]

                # 3. (オプション) バッファが一杯なら古い値を減算 (Welford's method)
                if state["count"] >= buffer_len:
                    # 古い値を取得
                    old_x_deque = self.proxy_feature_buffers[tf_name]["market_proxy"]
                    old_y_deque = self.proxy_feature_buffers[tf_name][feat_name]
                    # (popleft() は Deque が空だとエラーになるためチェック)
                    if old_x_deque and old_y_deque:
                        old_x = old_x_deque[0]  # (popしない、maxlenで自動的に消える)
                        old_y = old_y_deque[0]
                        # 減算
                        state["sum_x"] -= old_x
                        state["sum_y"] -= old_y
                        state["sum_xy"] -= old_x * old_y
                        state["sum_x_sq"] -= old_x**2
                        state["sum_y_sq"] -= old_y**2
                        state["count"] -= 1.0

                # 4. 新しい値を加算
                state["sum_x"] += latest_x
                state["sum_y"] += latest_y
                state["sum_xy"] += latest_x * latest_y
                state["sum_x_sq"] += latest_x**2
                state["sum_y_sq"] += latest_y**2
                state["count"] += 1.0

                # 5. 最新の (x, y) もバッファに保存 (古い値を減算するため)
                self.proxy_feature_buffers[tf_name][feat_name].append(latest_y)
                if feat_name == "atr":  # (market_proxy は1回だけ保存)
                    self.proxy_feature_buffers[tf_name]["market_proxy"].append(latest_x)

        except Exception as e:
            self.logger.warning(
                f"[{tf_name}] 逐次OLSの更新に失敗 ({feat_name}): {e}", exc_info=False
            )

    def calculate_feature_vector(
        self, tf_name: str, timestamp: datetime, market_proxy_cache: pd.DataFrame
    ) -> Optional[np.ndarray]:
        """
        [修正] (矛盾①解決)
        指定された時間足の現在のバッファから304個の全特徴量を計算し、
        M5市場プロキシで「純化」する。

        Args:
            tf_name: 計算対象の時間足 (e.g., "M15")
            timestamp: シグナル発生時刻 (純化のアライメント用)
            market_proxy_cache: main.pyがロードしたM5市場プロキシDF

        Returns:
            純化済み特徴量ベクトル (Numpy配列)、またはエラーの場合は None
        """
        if not self.is_buffer_filled[tf_name]:
            self.logger.warning(f"特徴量計算スキップ ({tf_name}): バッファ未充填")
            return None

        try:
            # 1. バッファをNumpy配列に変換
            data = {
                col: np.array(self.data_buffers[tf_name][col], dtype=np.float64)
                for col in self.OHLCV_COLS
            }

            # 2. ベース特徴量の計算 (V5.1ロジック)
            # (e.g., {'e1c_atr_21': 1.85, 'e1a_statistical_kurtosis_50': 3.2, ...})
            base_features = self._calculate_base_features(data, tf_name)

            # 3. アルファの純化 (Neutralization) [矛盾①解決]
            #  (05_alpha_decay_analyzer.py のロジックを移植)
            neutralized_features = self._calculate_neutralized_features(
                base_features, tf_name, timestamp, market_proxy_cache
            )

            # 4. 名簿の順番通りにベクトルを構築
            feature_vector = []

            # サフィックス付きの名簿 (e.g., "e1a_statistical_kurtosis_50_neutralized_D1")
            for feature_name_in_list in self.feature_list:
                # 特徴量の時間足が、現在処理中の時間足(tf_name)と一致するか？
                if not feature_name_in_list.endswith(f"_{tf_name}"):
                    continue  # (e.g., D1 の特徴量は M15 ループでは無視)

                # サフィックスを除去 (e.g., "e1a_statistical_kurtosis_50")
                base_name = feature_name_in_list.split("_neutralized_")[0]

                if base_name in neutralized_features:
                    feature_vector.append(neutralized_features[base_name])
                else:
                    # この時間足で計算すべき特徴量だが、計算マップになかった
                    self.logger.warning(
                        f"特徴量 {base_name} (元: {feature_name_in_list}) が純化マップに見つかりません。0.0 を使用します。"
                    )
                    feature_vector.append(0.0)

            if not feature_vector:
                # この時間足(tf_name)で計算すべき特徴量は名簿になかった
                return None

            # 5. 最終的なベクトルをNaNチェック (0で埋める)
            final_vector = np.nan_to_num(
                np.array(feature_vector), nan=0.0, posinf=0.0, neginf=0.0
            )

            # (1, N) の形状にして返す (Nは
            # この時間足に属する特徴量の数)
            return np.array([final_vector])

        except Exception as e:
            self.logger.error(
                f"特徴量ベクトル計算中にエラー ({tf_name}): {e}", exc_info=True
            )
            return None

    def _calculate_neutralized_features(
        self,
        base_features_dict: Dict[str, float],
        tf_name: str,
        signal_timestamp: datetime,
        market_proxy_cache_df: pd.DataFrame,
    ) -> Dict[str, float]:
        """
        [V6.5 修正]
        逐次計算されたOLS状態 (sum_x, sum_y...) を使って、
        「瞬時」に残差 (純化済み特徴量) を計算する。
        """

        neutralized_features: Dict[str, float] = {}

        # (Numba互換のため定数定義)
        PROXY_FEATURES = [
            "atr",
            "log_return",
            "price_momentum",
            "rolling_volatility",
            "volume_ratio",
        ]

        try:
            # 1. 最新の市場プロキシ値 (x) を取得
            # (バッファから最新の値を取得)
            latest_x_deque = self.proxy_feature_buffers[tf_name]["market_proxy"]
            latest_x = latest_x_deque[-1] if latest_x_deque else 0.0

            # 2. ベース特徴量ごとに純化
            for base_name, latest_y in base_features_dict.items():
                # 3. 純化対象の5特徴量か？
                if base_name not in PROXY_FEATURES:
                    neutralized_features[base_name] = latest_y
                    continue

                # 4. OLS状態を取得
                state = self.ols_state[tf_name][base_name]
                n = state["count"]

                if n < 20:  # (十分なサンプルがない場合は純化しない)
                    neutralized_features[base_name] = latest_y
                    continue

                # 5. OLSパラメータ (Alpha, Beta) を計算
                # (Welford's online algorithm)
                mean_x = state["sum_x"] / n
                mean_y = state["sum_y"] / n

                # 共分散(xy) = E[XY] - E[X]E[Y]
                cov_xy = (state["sum_xy"] / n) - (mean_x * mean_y)
                # 分散(x) = E[X^2] - E[X]^2
                var_x = (state["sum_x_sq"] / n) - (mean_x**2)

                if var_x < 1e-10:
                    beta = 0.0
                else:
                    beta = cov_xy / var_x

                alpha = mean_y - beta * mean_x

                # 6. 最新値の残差を計算
                latest_y_safe = latest_y if np.isfinite(latest_y) else 0.0

                neutralized_value = latest_y_safe - (beta * latest_x + alpha)
                neutralized_features[base_name] = neutralized_value

            return neutralized_features

        except Exception as e:
            self.logger.error(f"アルファ純化 ({tf_name}) に失敗: {e}", exc_info=True)
            # 失敗した場合は、純化されていないベース特徴量をそのまま返す
            return base_features_dict

    def _calculate_base_features(
        self, data: Dict[str, np.ndarray], tf_name: str
    ) -> Dict[str, float]:
        """
        【V5.1ロジック完全移植版】
        Numpy配列を受け取り、`final_feature_set_v2.txt` [cite: 441] に基づいて
        300+個の「ベース」特徴量の最新値を計算する。

        NOTE:
        - `tf_name`引数はB案(V6.0)のI/F互換性のために存在しますが、
          この関数内の計算は(V5.1同様)渡されたNumpyバッファのみに依存します。
        - `_window(data_array, window_size)`: 配列の末尾から `window_size` 個の要素を取得
        - `_array(data_array)`: 配列全体をそのまま使用
        - `_last(result_array)`: UDFが返した配列の最新値（末尾）を取得
        - `_pct(data_array)`: Polarsの `pct_change()` と同様の挙動（先頭にNaNを追加）
        """

        features = {}

        # --- ヘルパー関数 ---
        def _window(arr: np.ndarray, window: int) -> np.ndarray:
            """配列の末尾から `window` 個の要素を取得"""
            if window <= 0:  # 0や負のウィンドウサイズはエラーを避ける
                return np.array([], dtype=arr.dtype)
            if window > len(arr):
                # self.logger.warning(f"Window {window} > Array {len(arr)}. Returning full array.")
                return arr
            return arr[-window:]

        def _array(arr: np.ndarray) -> np.ndarray:
            """配列全体をそのまま使用"""
            return arr

        def _last(arr: np.ndarray) -> float:
            """UDFが返した配列の最新値（末尾）を取得"""
            if len(arr) == 0:
                return np.nan
            # (Numba UDFがnp.nanを返す場合があるため、nanチェックはしない)
            return arr[-1]

        def _pct(arr: np.ndarray) -> np.ndarray:
            """Polarsのpct_change()のNumpy版 (先頭にNaN)"""
            if len(arr) < 2:
                return np.full_like(arr, np.nan)

            # ゼロ除算を回避
            arr_safe = arr[:-1].copy()
            arr_safe[arr_safe == 0] = 1e-10

            pct = np.diff(arr) / arr_safe
            return np.concatenate(([np.nan], pct))

        def _ema(arr: np.ndarray, span: int) -> np.ndarray:
            """EMAのNumpy実装"""
            alpha = 2.0 / (span + 1.0)
            ema = np.zeros_like(arr, dtype=np.float64)  # 型を明示
            if len(arr) == 0:
                return ema
            ema[0] = arr[0]
            for i in range(1, len(arr)):
                ema[i] = alpha * arr[i] + (1 - alpha) * ema[i - 1]
            return ema

        def _rolling_mean(arr: np.ndarray, window: int) -> np.ndarray:
            """ローリング平均 (Numpy実装)"""
            if len(arr) < window or window <= 0:
                return np.full(len(arr), np.nan)  # 元の配列と同じ長さで返す
            ret = np.cumsum(arr, dtype=float)
            ret[window:] = ret[window:] - ret[:-window]
            res = ret[window - 1 :] / window
            return np.concatenate((np.full(window - 1, np.nan), res))

        # --- 共通データの事前計算 ---
        # (バッファ全体で計算。NaNが含まれる可能性がある点に注意)
        close_pct = _pct(data["close"])

        # --- Engine 1A 特徴量 (engine_1_A_...py) ---
        features["e1a_anderson_darling_statistic_30"] = anderson_darling_numba(
            _window(data["close"], 30)
        )
        # [V6.5 修正] _array() を _window() に変更
        features["e1a_fast_basic_stabilization"] = _last(
            basic_stabilization_numba(_window(data["close"], 100))  # (安定化は100窓)
        )
        # [V6.5 修正] UDF呼び出しを高速なNumpy関数に変更
        features["e1a_fast_rolling_mean_10"] = np.mean(_window(data["close"], 10))
        features["e1a_fast_rolling_mean_50"] = np.mean(_window(data["close"], 50))
        features["e1a_fast_rolling_mean_5"] = np.mean(_window(data["close"], 5))

        features["e1a_fast_rolling_std_100"] = np.std(_window(data["close"], 100))
        features["e1a_fast_rolling_std_10"] = np.std(_window(data["close"], 10))
        features["e1a_fast_rolling_std_20"] = np.std(_window(data["close"], 20))
        features["e1a_fast_rolling_std_5"] = np.std(_window(data["close"], 5))

        features["e1a_fast_volume_mean_10"] = np.mean(_window(data["volume"], 10))
        features["e1a_fast_volume_mean_20"] = np.mean(_window(data["volume"], 20))
        features["e1a_fast_volume_mean_50"] = np.mean(_window(data["volume"], 50))
        features["e1a_fast_volume_mean_5"] = np.mean(_window(data["volume"], 5))
        features["e1a_jarque_bera_statistic_50"] = jarque_bera_statistic_numba(
            _window(data["close"], 50)
        )

        # (PercentileはNumpyネイティブ関数を使用)
        q75_10, q25_10 = np.percentile(_window(data["close"], 10), [75, 25])
        features["e1a_robust_iqr_10"] = q75_10 - q25_10
        q75_20, q25_20 = np.percentile(_window(data["close"], 20), [75, 25])
        features["e1a_robust_iqr_20"] = q75_20 - q25_20
        q75_50, q25_50 = np.percentile(_window(data["close"], 50), [75, 25])
        features["e1a_robust_iqr_50"] = q75_50 - q25_50

        features["e1a_robust_mad_20"] = mad_rolling_numba(_window(data["close"], 20))
        features["e1a_robust_median_50"] = np.median(_window(data["close"], 50))
        features["e1a_robust_q75_50"] = q75_50
        features["e1a_runs_test_statistic_30"] = runs_test_numba(
            _window(data["close"], 30)
        )

        mean_10 = np.mean(_window(data["close"], 10))
        mean_20 = np.mean(_window(data["close"], 20))
        mean_50 = np.mean(_window(data["close"], 50))
        std_10 = np.std(_window(data["close"], 10))
        std_20 = np.std(_window(data["close"], 20))
        std_50 = np.std(_window(data["close"], 50))

        features["e1a_statistical_cv_10"] = std_10 / (mean_10 + 1e-10)
        features["e1a_statistical_cv_20"] = std_20 / (mean_20 + 1e-10)
        features["e1a_statistical_cv_50"] = std_50 / (mean_50 + 1e-10)

        features["e1a_statistical_kurtosis_20"] = statistical_kurtosis_numba(
            _window(data["close"], 20)
        )
        features["e1a_statistical_kurtosis_50"] = statistical_kurtosis_numba(
            _window(data["close"], 50)
        )

        features["e1a_statistical_moment_5_20"] = statistical_moment_numba(
            _window(data["close"], 20), 5
        )
        features["e1a_statistical_moment_5_50"] = statistical_moment_numba(
            _window(data["close"], 50), 5
        )
        features["e1a_statistical_moment_6_20"] = statistical_moment_numba(
            _window(data["close"], 20), 6
        )
        features["e1a_statistical_moment_6_50"] = statistical_moment_numba(
            _window(data["close"], 50), 6
        )
        features["e1a_statistical_moment_7_20"] = statistical_moment_numba(
            _window(data["close"], 20), 7
        )
        features["e1a_statistical_moment_7_50"] = statistical_moment_numba(
            _window(data["close"], 50), 7
        )
        features["e1a_statistical_moment_8_50"] = statistical_moment_numba(
            _window(data["close"], 50), 8
        )
        # (Scipy.stats.skew はNumba JITできないため、Numba UDFを使用)
        features["e1a_statistical_skewness_20"] = rolling_skew_numba(
            _window(data["close"], 20)
        )
        features["e1a_statistical_skewness_50"] = rolling_skew_numba(
            _window(data["close"], 50)
        )

        features["e1a_statistical_variance_10"] = np.var(_window(data["close"], 10))
        features["e1a_von_neumann_ratio_30"] = von_neumann_ratio_numba(
            _window(data["close"], 30)
        )

        # --- Engine 1B 特徴量 (engine_1_B_...py) ---
        features["e1b_adf_statistic_100"] = adf_統計量_udf(_window(data["close"], 100))
        features["e1b_adf_statistic_50"] = adf_統計量_udf(_window(data["close"], 50))
        features["e1b_arima_residual_var_100"] = arima_残差分散_udf(
            _window(data["close"], 100)
        )
        features["e1b_arima_residual_var_50"] = arima_残差分散_udf(
            _window(data["close"], 50)
        )

        features["e1b_bollinger_lower_50"] = mean_50 - 2 * std_50
        features["e1b_bollinger_upper_50"] = mean_50 + 2 * std_50

        features["e1b_holt_level_100"] = holt_winters_レベル_udf(
            _window(data["close"], 100)
        )
        features["e1b_holt_level_50"] = holt_winters_レベル_udf(
            _window(data["close"], 50)
        )
        features["e1b_holt_trend_100"] = holt_winters_トレンド_udf(
            _window(data["close"], 100)
        )
        features["e1b_holt_trend_50"] = holt_winters_トレンド_udf(
            _window(data["close"], 50)
        )
        features["e1b_kalman_state_100"] = kalman_状態推定_udf(
            _window(data["close"], 100)
        )
        features["e1b_kpss_statistic_100"] = kpss_統計量_udf(
            _window(data["close"], 100)
        )
        features["e1b_kpss_statistic_50"] = kpss_統計量_udf(_window(data["close"], 50))
        features["e1b_lowess_fitted_100"] = lowess_適合値_udf(
            _window(data["close"], 100)
        )
        features["e1b_lowess_fitted_50"] = lowess_適合値_udf(_window(data["close"], 50))
        features["e1b_pp_statistic_100"] = phillips_perron_統計量_udf(
            _window(data["close"], 100)
        )

        features["e1b_price_change"] = _last(close_pct)
        features["e1b_price_range"] = data["high"][-1] - data["low"][-1]

        features["e1b_rolling_mean_100"] = np.mean(_window(data["close"], 100))
        features["e1b_rolling_median_100"] = np.median(_window(data["close"], 100))
        features["e1b_rolling_median_50"] = np.median(_window(data["close"], 50))

        features["e1b_t_dist_dof_50"] = t分布_自由度_udf(_window(close_pct, 50))
        features["e1b_t_dist_scale_50"] = t分布_尺度_udf(_window(close_pct, 50))

        features["e1b_theil_sen_slope_100"] = theil_sen_傾き_udf(
            _window(data["close"], 100)
        )
        features["e1b_theil_sen_slope_50"] = theil_sen_傾き_udf(
            _window(data["close"], 50)
        )

        features["e1b_volatility_20"] = np.std(_window(close_pct, 20))
        features["e1b_zscore_20"] = (data["close"][-1] - mean_20) / (std_20 + 1e-10)
        features["e1b_zscore_50"] = (data["close"][-1] - mean_50) / (std_50 + 1e-10)

        # --- Engine 1C 特徴量 (engine_1_C_...py) ---
        # (V5.1の `_calculate_base_features` L:1422-1457 のロジックを流用)
        atr_13_arr = calculate_atr_numba(
            _array(data["high"]), _array(data["low"]), _array(data["close"]), 13
        )
        atr_21_arr = calculate_atr_numba(
            _array(data["high"]), _array(data["low"]), _array(data["close"]), 21
        )
        atr_34_arr = calculate_atr_numba(
            _array(data["high"]), _array(data["low"]), _array(data["close"]), 34
        )
        atr_55_arr = calculate_atr_numba(
            _array(data["high"]), _array(data["low"]), _array(data["close"]), 55
        )
        atr_13, atr_21, atr_34, atr_55 = (
            _last(atr_13_arr),
            _last(atr_21_arr),
            _last(atr_34_arr),
            _last(atr_55_arr),
        )

        rsi_14_arr = calculate_rsi_numba(_array(data["close"]), 14)
        rsi_21_arr = calculate_rsi_numba(_array(data["close"]), 21)
        rsi_30_arr = calculate_rsi_numba(_array(data["close"]), 30)
        rsi_50_arr = calculate_rsi_numba(_array(data["close"]), 50)

        stoch_k_14_arr = calculate_stochastic_numba(
            _array(data["high"]), _array(data["low"]), _array(data["close"]), 14
        )
        stoch_k_21_arr = calculate_stochastic_numba(
            _array(data["high"]), _array(data["low"]), _array(data["close"]), 21
        )
        stoch_k_9_arr = calculate_stochastic_numba(
            _array(data["high"]), _array(data["low"]), _array(data["close"]), 9
        )

        di_plus_13_arr = calculate_di_plus_numba(
            _array(data["high"]), _array(data["low"]), _array(data["close"]), 13
        )
        di_minus_13_arr = calculate_di_minus_numba(
            _array(data["high"]), _array(data["low"]), _array(data["close"]), 13
        )
        di_plus_21_arr = calculate_di_plus_numba(
            _array(data["high"]), _array(data["low"]), _array(data["close"]), 21
        )
        di_minus_21_arr = calculate_di_minus_numba(
            _array(data["high"]), _array(data["low"]), _array(data["close"]), 21
        )
        di_plus_34_arr = calculate_di_plus_numba(
            _array(data["high"]), _array(data["low"]), _array(data["close"]), 34
        )
        di_minus_34_arr = calculate_di_minus_numba(
            _array(data["high"]), _array(data["low"]), _array(data["close"]), 34
        )

        aroon_up_14_arr = calculate_aroon_up_numba(_array(data["high"]), 14)
        aroon_down_14_arr = calculate_aroon_down_numba(_array(data["low"]), 14)

        williams_r_14_arr = calculate_williams_r_numba(
            _array(data["high"]), _array(data["low"]), _array(data["close"]), 14
        )
        williams_r_28_arr = calculate_williams_r_numba(
            _array(data["high"]), _array(data["low"]), _array(data["close"]), 28
        )
        williams_r_56_arr = calculate_williams_r_numba(
            _array(data["high"]), _array(data["low"]), _array(data["close"]), 56
        )

        trix_14_arr = calculate_trix_numba(_array(data["close"]), 14)
        trix_20_arr = calculate_trix_numba(_array(data["close"]), 20)
        trix_30_arr = calculate_trix_numba(_array(data["close"]), 30)

        tsi_13_arr = calculate_tsi_numba(_array(data["close"]), 25, 13)
        tsi_25_arr = calculate_tsi_numba(_array(data["close"]), 13, 25)

        ema_10_arr = _ema(data["close"], 10)
        ema_20_arr = _ema(data["close"], 20)
        ema_50_arr = _ema(data["close"], 50)
        ema_100_arr = _ema(data["close"], 100)
        ema_200_arr = _ema(data["close"], 200)

        # 1C 特徴量を辞書に追加
        features["e1c_adx_13"] = _last(
            calculate_adx_numba(
                _array(data["high"]), _array(data["low"]), _array(data["close"]), 13
            )
        )
        features["e1c_adx_21"] = _last(
            calculate_adx_numba(
                _array(data["high"]), _array(data["low"]), _array(data["close"]), 21
            )
        )
        features["e1c_aroon_down_14"] = _last(aroon_down_14_arr)
        features["e1c_aroon_up_14"] = _last(aroon_up_14_arr)
        features["e1c_aroon_oscillator_14"] = (
            features["e1c_aroon_up_14"] - features["e1c_aroon_down_14"]
        )
        features["e1c_atr_13"] = atr_13
        features["e1c_atr_lower_13_1.5"] = data["close"][-1] - (atr_13 * 1.5)
        features["e1c_atr_lower_13_2.0"] = data["close"][-1] - (atr_13 * 2.0)
        features["e1c_atr_lower_21_1.5"] = data["close"][-1] - (atr_21 * 1.5)
        features["e1c_atr_lower_21_2.0"] = data["close"][-1] - (atr_21 * 2.0)
        features["e1c_atr_lower_21_2.5"] = data["close"][-1] - (atr_21 * 2.5)
        features["e1c_atr_lower_34_1.5"] = data["close"][-1] - (atr_34 * 1.5)
        features["e1c_atr_lower_34_2.0"] = data["close"][-1] - (atr_34 * 2.0)
        features["e1c_atr_lower_34_2.5"] = data["close"][-1] - (atr_34 * 2.5)
        features["e1c_atr_lower_55_1.5"] = data["close"][-1] - (atr_55 * 1.5)
        features["e1c_atr_lower_55_2.0"] = data["close"][-1] - (atr_55 * 2.0)
        features["e1c_atr_lower_55_2.5"] = data["close"][-1] - (atr_55 * 2.5)
        features["e1c_atr_pct_13"] = (atr_13 / data["close"][-1]) * 100
        features["e1c_atr_pct_21"] = (atr_21 / data["close"][-1]) * 100
        features["e1c_atr_pct_34"] = (atr_34 / data["close"][-1]) * 100
        features["e1c_atr_pct_55"] = (atr_55 / data["close"][-1]) * 100
        features["e1c_atr_trend_13"] = atr_13_arr[-1] - atr_13_arr[-2]
        features["e1c_atr_trend_21"] = atr_21_arr[-1] - atr_21_arr[-2]
        features["e1c_atr_trend_34"] = atr_34_arr[-1] - atr_34_arr[-2]
        features["e1c_atr_trend_55"] = atr_55_arr[-1] - atr_55_arr[-2]
        features["e1c_atr_upper_13_1.5"] = data["close"][-1] + (atr_13 * 1.5)
        features["e1c_atr_upper_13_2.0"] = data["close"][-1] + (atr_13 * 2.0)
        features["e1c_atr_upper_13_2.5"] = data["close"][-1] + (atr_13 * 2.5)
        features["e1c_atr_upper_21_1.5"] = data["close"][-1] + (atr_21 * 1.5)
        features["e1c_atr_upper_21_2.0"] = data["close"][-1] + (atr_21 * 2.0)
        features["e1c_atr_upper_21_2.5"] = data["close"][-1] + (atr_21 * 2.5)
        features["e1c_atr_upper_34_1.5"] = data["close"][-1] + (atr_34 * 1.5)
        features["e1c_atr_upper_34_2.0"] = data["close"][-1] + (atr_34 * 2.0)
        features["e1c_atr_upper_34_2.5"] = data["close"][-1] + (atr_34 * 2.5)
        features["e1c_atr_upper_55_1.5"] = data["close"][-1] + (atr_55 * 1.5)
        features["e1c_atr_upper_55_2.0"] = data["close"][-1] + (atr_55 * 2.0)
        features["e1c_atr_upper_55_2.5"] = data["close"][-1] + (atr_55 * 2.5)
        features["e1c_atr_volatility_13"] = np.std(_window(atr_13_arr, 13))
        features["e1c_atr_volatility_21"] = np.std(_window(atr_21_arr, 21))
        features["e1c_atr_volatility_34"] = np.std(_window(atr_34_arr, 34))
        features["e1c_atr_volatility_55"] = np.std(_window(atr_55_arr, 55))

        # BB
        bb_mean_20, bb_std_20 = mean_20, std_20
        bb_mean_30, bb_std_30 = (
            np.mean(_window(data["close"], 30)),
            np.std(_window(data["close"], 30)),
        )
        bb_mean_50, bb_std_50 = mean_50, std_50

        for std_dev in [2.0, 2.5, 3.0]:
            for period, (mean, std) in [
                (20, (bb_mean_20, bb_std_20)),
                (30, (bb_mean_30, bb_std_30)),
                (50, (bb_mean_50, bb_std_50)),
            ]:
                if std < 1e-10:
                    continue  # ゼロ除算回避
                upper = mean + std_dev * std
                lower = mean - std_dev * std
                width = upper - lower
                features[f"e1c_bb_upper_{period}_{std_dev}"] = upper
                features[f"e1c_bb_lower_{period}_{std_dev}"] = lower
                features[f"e1c_bb_width_{period}_{std_dev}"] = width
                features[f"e1c_bb_percent_{period}_{std_dev}"] = (
                    data["close"][-1] - lower
                ) / (width + 1e-10)
                features[f"e1c_bb_width_pct_{period}_{std_dev}"] = (
                    width / (mean + 1e-10)
                ) * 100
                features[f"e1c_bb_position_{period}_{std_dev}"] = (
                    data["close"][-1] - mean
                ) / (std + 1e-10)

        # MACD (EMA実装)
        ema_12_arr = _ema(data["close"], 12)
        ema_26_arr = _ema(data["close"], 26)
        ema_5_arr = _ema(data["close"], 5)
        ema_35_arr = _ema(data["close"], 35)
        ema_19_arr = _ema(data["close"], 19)
        ema_39_arr = _ema(data["close"], 39)

        macd_12_26_arr = ema_12_arr - ema_26_arr
        macd_5_35_arr = ema_5_arr - ema_35_arr
        macd_19_39_arr = ema_19_arr - ema_39_arr

        signal_12_26_9_arr = _ema(macd_12_26_arr, 9)
        signal_5_35_5_arr = _ema(macd_5_35_arr, 5)
        signal_19_39_9_arr = _ema(macd_19_39_arr, 9)

        features["e1c_macd_12_26"] = _last(macd_12_26_arr)
        features["e1c_macd_19_39"] = _last(macd_19_39_arr)
        features["e1c_macd_5_35"] = _last(macd_5_35_arr)
        features["e1c_macd_signal_12_26_9"] = _last(signal_12_26_9_arr)
        features["e1c_macd_signal_19_39_9"] = _last(signal_19_39_9_arr)
        features["e1c_macd_signal_5_35_5"] = _last(signal_5_35_5_arr)
        features["e1c_macd_histogram_12_26_9"] = _last(macd_12_26_arr) - _last(
            signal_12_26_9_arr
        )
        features["e1c_macd_histogram_19_39_9"] = _last(macd_19_39_arr) - _last(
            signal_19_39_9_arr
        )
        features["e1c_macd_histogram_5_35_5"] = _last(macd_5_35_arr) - _last(
            signal_5_35_5_arr
        )

        # DPO (Detrended Price Oscillator)
        for period in [20, 30, 50]:
            lookback = period // 2 + 1
            if len(data["close"]) > period and len(data["close"]) > lookback:
                # (V5.1のロジックを忠実に再現)
                sma = np.mean(_window(data["close"], period))
                features[f"e1c_dpo_{period}"] = data["close"][-1 - lookback] - sma
            else:
                features[f"e1c_dpo_{period}"] = np.nan

        # EMA Deviations
        features["e1c_ema_deviation_10"] = (
            (data["close"][-1] - _last(ema_10_arr)) / (_last(ema_10_arr) + 1e-10) * 100
        )
        features["e1c_ema_deviation_20"] = (
            (data["close"][-1] - _last(ema_20_arr)) / (_last(ema_20_arr) + 1e-10) * 100
        )
        features["e1c_ema_deviation_50"] = (
            (data["close"][-1] - _last(ema_50_arr)) / (_last(ema_50_arr) + 1e-10) * 100
        )
        features["e1c_ema_deviation_100"] = (
            (data["close"][-1] - _last(ema_100_arr))
            / (_last(ema_100_arr) + 1e-10)
            * 100
        )
        features["e1c_ema_deviation_200"] = (
            (data["close"][-1] - _last(ema_200_arr))
            / (_last(ema_200_arr) + 1e-10)
            * 100
        )

        # HMA, KAMA
        features["e1c_hma_21"] = _last(calculate_hma_numba(_array(data["close"]), 21))
        features["e1c_hma_34"] = _last(calculate_hma_numba(_array(data["close"]), 34))
        features["e1c_hma_55"] = _last(calculate_hma_numba(_array(data["close"]), 55))
        features["e1c_kama_21"] = _last(calculate_kama_numba(_array(data["close"]), 21))
        features["e1c_kama_34"] = _last(calculate_kama_numba(_array(data["close"]), 34))

        # KST (Know Sure Thing)
        roc_10 = (data["close"][-1] - data["close"][-11]) / (data["close"][-11] + 1e-10)
        roc_15 = (data["close"][-1] - data["close"][-16]) / (data["close"][-16] + 1e-10)
        roc_20 = (data["close"][-1] - data["close"][-21]) / (data["close"][-21] + 1e-10)
        roc_30 = (data["close"][-1] - data["close"][-31]) / (data["close"][-31] + 1e-10)
        kst_val = (roc_10 * 1 + roc_15 * 2 + roc_20 * 3 + roc_30 * 4) / 10 * 100
        features["e1c_kst"] = kst_val
        # (KST SignalはKSTのSMA(9)のため、KSTの履歴が必要。リアルタイムでは複雑なため簡易実装)
        features["e1c_kst_signal"] = kst_val  # (簡易的に最新値)

        # Momentum, ROC
        for period in [10, 20, 30, 50]:
            if len(data["close"]) > period:
                features[f"e1c_momentum_{period}"] = (
                    data["close"][-1] - data["close"][-1 - period]
                )
                features[f"e1c_rate_of_change_{period}"] = (
                    (data["close"][-1] - data["close"][-1 - period])
                    / (data["close"][-1 - period] + 1e-10)
                    * 100
                )
            else:
                features[f"e1c_momentum_{period}"] = np.nan
                features[f"e1c_rate_of_change_{period}"] = np.nan

        # RVI (Relative Vigor Index)
        rvi_10_arr = _rolling_mean(data["close"] - data["open"], 10) / (
            _rolling_mean(data["high"] - data["low"], 10) + 1e-10
        )
        rvi_14_arr = _rolling_mean(data["close"] - data["open"], 14) / (
            _rolling_mean(data["high"] - data["low"], 14) + 1e-10
        )
        rvi_20_arr = _rolling_mean(data["close"] - data["open"], 20) / (
            _rolling_mean(data["high"] - data["low"], 20) + 1e-10
        )
        features["e1c_relative_vigor_index_10"] = _last(rvi_10_arr)
        features["e1c_relative_vigor_index_14"] = _last(rvi_14_arr)
        features["e1c_relative_vigor_index_20"] = _last(rvi_20_arr)
        features["e1c_rvi_signal_10"] = np.mean(_window(rvi_10_arr, 4))
        features["e1c_rvi_signal_14"] = np.mean(_window(rvi_14_arr, 4))
        features["e1c_rvi_signal_20"] = np.mean(_window(rvi_20_arr, 4))

        # RSI
        features["e1c_rsi_14"] = _last(rsi_14_arr)
        features["e1c_rsi_21"] = _last(rsi_21_arr)
        features["e1c_rsi_30"] = _last(rsi_30_arr)
        features["e1c_rsi_50"] = _last(rsi_50_arr)
        features["e1c_rsi_momentum_14"] = rsi_14_arr[-1] - rsi_14_arr[-2]
        features["e1c_rsi_momentum_21"] = rsi_21_arr[-1] - rsi_21_arr[-2]
        features["e1c_rsi_momentum_30"] = rsi_30_arr[-1] - rsi_30_arr[-2]
        features["e1c_rsi_momentum_50"] = rsi_50_arr[-1] - rsi_50_arr[-2]

        # Stochastic RSI
        rsi_14_window = _window(rsi_14_arr, 14)
        rsi_14_min = np.nanmin(rsi_14_window)
        rsi_14_max = np.nanmax(rsi_14_window)
        features["e1c_stochastic_rsi_14"] = (
            (rsi_14_arr[-1] - rsi_14_min) / (rsi_14_max - rsi_14_min + 1e-10) * 100
        )
        rsi_21_window = _window(rsi_21_arr, 21)
        rsi_21_min = np.nanmin(rsi_21_window)
        rsi_21_max = np.nanmax(rsi_21_window)
        features["e1c_stochastic_rsi_21"] = (
            (rsi_21_arr[-1] - rsi_21_min) / (rsi_21_max - rsi_21_min + 1e-10) * 100
        )

        # RSI Divergence
        features["e1c_rsi_divergence_14"] = (
            (data["close"][-1] - data["close"][-15]) / (data["close"][-15] + 1e-10)
        ) - ((rsi_14_arr[-1] - rsi_14_arr[-15]) / 50 - 1)
        features["e1c_rsi_divergence_21"] = (
            (data["close"][-1] - data["close"][-22]) / (data["close"][-22] + 1e-10)
        ) - ((rsi_21_arr[-1] - rsi_21_arr[-22]) / 50 - 1)

        # Schaff Trend Cycle
        stc_macd_12_26 = macd_12_26_arr
        stc_macd_23_50 = _ema(data["close"], 23) - _ema(data["close"], 50)

        stc_macd_12_26_window = _window(stc_macd_12_26, 9)
        stc_macd_12_26_min = np.nanmin(stc_macd_12_26_window)
        stc_macd_12_26_max = np.nanmax(stc_macd_12_26_window)
        stc_12_26_k = (
            (_last(stc_macd_12_26) - stc_macd_12_26_min)
            / (stc_macd_12_26_max - stc_macd_12_26_min + 1e-10)
            * 100
        )

        stc_macd_23_50_window = _window(stc_macd_23_50, 10)
        stc_macd_23_50_min = np.nanmin(stc_macd_23_50_window)
        stc_macd_23_50_max = np.nanmax(stc_macd_23_50_window)
        stc_23_50_k = (
            (_last(stc_macd_23_50) - stc_macd_23_50_min)
            / (stc_macd_23_50_max - stc_macd_23_50_min + 1e-10)
            * 100
        )
        features["e1c_schaff_trend_cycle_12_26_9"] = stc_12_26_k
        features["e1c_schaff_trend_cycle_23_50_10"] = stc_23_50_k

        # SMA
        for period in [10, 20, 50, 100, 200]:
            sma = np.mean(_window(data["close"], period))
            features[f"e1c_sma_{period}"] = sma
            features[f"e1c_sma_deviation_{period}"] = (
                (data["close"][-1] - sma) / (sma + 1e-10) * 100
            )

        # Stochastic
        features["e1c_stoch_k_14"] = _last(stoch_k_14_arr)
        features["e1c_stoch_k_21"] = _last(stoch_k_21_arr)
        features["e1c_stoch_k_9"] = _last(stoch_k_9_arr)
        features["e1c_stoch_d_14_3"] = np.mean(_window(stoch_k_14_arr, 3))
        features["e1c_stoch_d_21_5"] = np.mean(_window(stoch_k_21_arr, 5))
        features["e1c_stoch_d_9_3"] = np.mean(_window(stoch_k_9_arr, 3))

        stoch_d_14_3_arr = _rolling_mean(stoch_k_14_arr, 3)
        stoch_d_21_5_arr = _rolling_mean(stoch_k_21_arr, 5)
        stoch_d_9_3_arr = _rolling_mean(stoch_k_9_arr, 3)

        features["e1c_stoch_slow_d_14_3_3"] = np.mean(_window(stoch_d_14_3_arr, 3))
        features["e1c_stoch_slow_d_21_5_5"] = np.mean(_window(stoch_d_21_5_arr, 5))
        features["e1c_stoch_slow_d_9_3_3"] = np.mean(_window(stoch_d_9_3_arr, 3))

        # Trend
        for period in [20, 50, 100]:
            if len(data["close"]) > period:
                features[f"e1c_trend_slope_{period}"] = (
                    data["close"][-1] - data["close"][-1 - period]
                ) / period
                features[f"e1c_trend_strength_{period}"] = 1 / (
                    np.std(_window(data["close"], period)) + 1e-10
                )
                direction_changes = np.abs(np.diff(np.sign(np.diff(data["close"]))))
                features[f"e1c_trend_consistency_{period}"] = (
                    1 - np.mean(_window(direction_changes, period)) / 2
                )
            else:
                features[f"e1c_trend_slope_{period}"] = np.nan
                features[f"e1c_trend_strength_{period}"] = np.nan
                features[f"e1c_trend_consistency_{period}"] = np.nan

        # Trix, TSI, Williams %R
        features["e1c_trix_14"] = _last(trix_14_arr)
        features["e1c_trix_20"] = _last(trix_20_arr)
        features["e1c_trix_30"] = _last(trix_30_arr)
        features["e1c_tsi_13"] = _last(tsi_13_arr)
        features["e1c_tsi_25"] = _last(tsi_25_arr)
        features["e1c_williams_r_14"] = _last(williams_r_14_arr)
        features["e1c_williams_r_28"] = _last(williams_r_28_arr)
        features["e1c_williams_r_56"] = _last(williams_r_56_arr)

        # WMA (1.1で追加したNumba UDFを使用)
        wma_10_arr = wma_rolling_numba(_array(data["close"]), 10)
        wma_20_arr = wma_rolling_numba(_array(data["close"]), 20)
        wma_50_arr = wma_rolling_numba(_array(data["close"]), 50)
        wma_100_arr = wma_rolling_numba(_array(data["close"]), 100)
        wma_200_arr = wma_rolling_numba(_array(data["close"]), 200)
        features["e1c_wma_10"] = _last(wma_10_arr)
        features["e1c_wma_20"] = _last(wma_20_arr)
        features["e1c_wma_50"] = _last(wma_50_arr)
        features["e1c_wma_100"] = _last(wma_100_arr)
        features["e1c_wma_200"] = _last(wma_200_arr)

        # --- Engine 1D (V5.1 L:1609-1662) ---
        features["e1d_accumulation_distribution"] = _last(
            accumulation_distribution_udf(
                _array(data["high"]),
                _array(data["low"]),
                _array(data["close"]),
                _array(data["volume"]),
            )
        )
        features["e1d_body_size"] = abs(data["close"][-1] - data["open"][-1])
        features["e1d_candlestick_pattern"] = _last(
            candlestick_patterns_udf(
                _array(data["open"]),
                _array(data["high"]),
                _array(data["low"]),
                _array(data["close"]),
            )
        )
        features["e1d_chaikin_volatility_10"] = _last(
            chaikin_volatility_udf(_array(data["high"]), _array(data["low"]), 10)
        )
        features["e1d_cmf_13"] = _last(
            cmf_udf(
                _array(data["high"]),
                _array(data["low"]),
                _array(data["close"]),
                _array(data["volume"]),
                13,
            )
        )
        features["e1d_force_index"] = _last(
            force_index_udf(_array(data["close"]), _array(data["volume"]))
        )
        features["e1d_hv_robust_10"] = hv_robust_udf(_window(close_pct, 10))
        features["e1d_hv_robust_20"] = hv_robust_udf(_window(close_pct, 20))
        features["e1d_hv_robust_30"] = hv_robust_udf(_window(close_pct, 30))
        features["e1d_hv_robust_50"] = hv_robust_udf(_window(close_pct, 50))
        features["e1d_hv_robust_annual_252"] = hv_robust_udf(
            _window(close_pct, 252)
        ) * np.sqrt(252)
        features["e1d_hv_standard_10"] = hv_standard_udf(_window(close_pct, 10))
        features["e1d_hv_standard_30"] = hv_standard_udf(_window(close_pct, 30))
        features["e1d_hv_standard_50"] = hv_standard_udf(_window(close_pct, 50))
        features["e1d_intraday_return"] = (data["close"][-1] - data["open"][-1]) / (
            data["open"][-1] + 1e-10
        )
        features["e1d_lower_wick_ratio"] = (
            min(data["open"][-1], data["close"][-1]) - data["low"][-1]
        ) / (data["high"][-1] - data["low"][-1] + 1e-10)
        features["e1d_mass_index_20"] = _last(
            mass_index_udf(_array(data["high"]), _array(data["low"]), 20)
        )
        features["e1d_mfi_13"] = _last(
            mfi_udf(
                _array(data["high"]),
                _array(data["low"]),
                _array(data["close"]),
                _array(data["volume"]),
                13,
            )
        )
        features["e1d_obv"] = _last(
            obv_udf(_array(data["close"]), _array(data["volume"]))
        )
        features["e1d_overnight_gap"] = (
            (data["open"][-1] - data["close"][-2]) / (data["close"][-2] + 1e-10)
            if len(data["close"]) > 1
            else 0.0
        )
        features["e1d_price_channel_upper_100"] = np.max(_window(data["high"], 100))
        features["e1d_price_location_hl"] = (data["close"][-1] - data["low"][-1]) / (
            data["high"][-1] - data["low"][-1] + 1e-10
        )
        features["e1d_upper_wick_ratio"] = (
            data["high"][-1] - max(data["open"][-1], data["close"][-1])
        ) / (data["high"][-1] - data["low"][-1] + 1e-10)
        features["e1d_volume_price_trend"] = np.mean(
            _window(data["close"] * data["volume"], 10)
        )
        features["e1d_volume_ratio"] = data["volume"][-1] / (
            np.mean(_window(data["volume"], 20)) + 1e-10
        )
        features["e1d_hv_regime_50"] = (
            1.0 if hv_robust_udf(_window(close_pct, 50)) > 0.005 else 0.0
        )  # (V5.1 L:330-331 簡易ロジック)

        # --- Engine 1E (V5.1 L:1664-1715) ---
        features["e1e_acoustic_frequency_128"] = acoustic_frequency_udf(
            _window(close_pct, 128)
        )
        features["e1e_acoustic_frequency_256"] = acoustic_frequency_udf(
            _window(close_pct, 256)
        )
        features["e1e_acoustic_power_128"] = acoustic_power_udf(_window(close_pct, 128))
        features["e1e_acoustic_power_256"] = acoustic_power_udf(_window(close_pct, 256))
        features["e1e_acoustic_power_512"] = acoustic_power_udf(_window(close_pct, 512))
        features["e1e_hilbert_amp_cv_100"] = np.std(_window(data["close"], 100)) / (
            np.mean(_window(data["close"], 100)) + 1e-10
        )
        features["e1e_hilbert_amplitude_100"] = hilbert_amplitude_udf(
            _window(close_pct, 100)
        )
        features["e1e_hilbert_amplitude_200"] = hilbert_amplitude_udf(
            _window(close_pct, 200)
        )
        features["e1e_hilbert_amplitude_50"] = hilbert_amplitude_udf(
            _window(close_pct, 50)
        )
        features["e1e_hilbert_freq_energy_ratio_100"] = np.sum(
            _window(close_pct, 100) ** 2
        ) / (np.sum(_window(data["close"], 100) ** 2) + 1e-10)
        features["e1e_hilbert_freq_mean_100"] = hilbert_freq_mean_udf(
            _window(close_pct, 100)
        )
        features["e1e_hilbert_freq_std_100"] = hilbert_freq_std_udf(
            _window(close_pct, 100)
        )
        features["e1e_hilbert_phase_stability_50"] = hilbert_phase_stability_udf(
            _window(close_pct, 50)
        )
        features["e1e_hilbert_phase_var_50"] = hilbert_phase_var_udf(
            _window(close_pct, 50)
        )
        features["e1e_hilbert_amp_mean_100"] = np.mean(
            np.abs(_window(data["close"], 100))
        )
        features["e1e_hilbert_amp_std_100"] = np.std(
            np.abs(_window(data["close"], 100))
        )
        features["e1e_signal_crest_factor_50"] = np.max(
            np.abs(_window(data["close"], 50))
        ) / (np.sqrt(np.mean(_window(data["close"], 50) ** 2)) + 1e-10)
        features["e1e_signal_peak_to_peak_100"] = np.max(
            _window(data["close"], 100)
        ) - np.min(_window(data["close"], 100))
        features["e1e_signal_rms_50"] = np.sqrt(np.mean(_window(close_pct, 50) ** 2))
        features["e1e_spectral_bandwidth_128"] = spectral_bandwidth_udf(
            _window(close_pct, 128)
        )
        features["e1e_spectral_centroid_128"] = spectral_centroid_udf(
            _window(close_pct, 128)
        )
        features["e1e_spectral_energy_128"] = np.sum(_window(close_pct, 128) ** 2)
        features["e1e_spectral_energy_256"] = np.sum(_window(close_pct, 256) ** 2)
        features["e1e_spectral_energy_512"] = np.sum(_window(close_pct, 512) ** 2)
        features["e1e_spectral_energy_64"] = np.sum(_window(close_pct, 64) ** 2)
        features["e1e_spectral_entropy_64"] = spectral_entropy_udf(
            _window(close_pct, 64)
        )
        features["e1e_spectral_flatness_128"] = spectral_flatness_udf(
            _window(close_pct, 128)
        )
        features["e1e_spectral_rolloff_128"] = spectral_rolloff_udf(
            _window(close_pct, 128)
        )
        # [V6.5 修正] _array() を _window() に変更
        # (UDFが配列を返す前提のため、_last() は維持)
        features["e1e_wavelet_energy_128"] = _last(
            wavelet_energy_udf(_window(close_pct, 128), 128)
        )
        features["e1e_wavelet_energy_256"] = _last(
            wavelet_energy_udf(_window(close_pct, 256), 256)
        )
        features["e1e_wavelet_energy_32"] = _last(
            wavelet_energy_udf(_window(close_pct, 32), 32)
        )
        features["e1e_wavelet_energy_64"] = _last(
            wavelet_energy_udf(_window(close_pct, 64), 64)
        )
        features["e1e_wavelet_entropy_64"] = wavelet_entropy_udf(_window(close_pct, 64))
        features["e1e_wavelet_mean_128"] = np.mean(_window(close_pct, 128))
        features["e1e_wavelet_mean_256"] = np.mean(_window(close_pct, 256))
        features["e1e_wavelet_mean_32"] = np.mean(_window(close_pct, 32))
        features["e1e_wavelet_mean_64"] = np.mean(_window(close_pct, 64))
        features["e1e_wavelet_std_128"] = np.std(_window(close_pct, 128))
        features["e1e_wavelet_std_256"] = np.std(_window(close_pct, 256))
        features["e1e_wavelet_std_32"] = np.std(_window(close_pct, 32))
        features["e1e_wavelet_std_64"] = np.std(_window(close_pct, 64))

        # --- Engine 1F (V5.1 L:1717-1786) ---
        features["e1f_aesthetic_balance_21"] = rolling_aesthetic_balance_udf(
            _window(data["close"], 21)
        )
        features["e1f_aesthetic_balance_34"] = rolling_aesthetic_balance_udf(
            _window(data["close"], 34)
        )
        features["e1f_aesthetic_balance_55"] = rolling_aesthetic_balance_udf(
            _window(data["close"], 55)
        )
        features["e1f_aesthetic_balance_89"] = rolling_aesthetic_balance_udf(
            _window(data["close"], 89)
        )
        features["e1f_biomechanical_efficiency_20"] = (
            rolling_biomechanical_efficiency_udf(_window(data["close"], 20))
        )
        features["e1f_energy_expenditure_20"] = rolling_energy_expenditure_udf(
            _window(data["close"], 20)
        )
        features["e1f_energy_expenditure_40"] = rolling_energy_expenditure_udf(
            _window(data["close"], 40)
        )
        features["e1f_energy_expenditure_60"] = rolling_energy_expenditure_udf(
            _window(data["close"], 60)
        )
        features["e1f_golden_ratio_adherence_21"] = rolling_golden_ratio_adherence_udf(
            _window(data["close"], 21)
        )
        features["e1f_golden_ratio_adherence_34"] = rolling_golden_ratio_adherence_udf(
            _window(data["close"], 34)
        )
        features["e1f_golden_ratio_adherence_55"] = rolling_golden_ratio_adherence_udf(
            _window(data["close"], 55)
        )
        features["e1f_golden_ratio_adherence_89"] = rolling_golden_ratio_adherence_udf(
            _window(data["close"], 89)
        )
        features["e1f_harmony_48"] = rolling_harmony_udf(_window(data["close"], 48))
        features["e1f_harmony_96"] = rolling_harmony_udf(_window(data["close"], 96))
        features["e1f_kinetic_energy_10"] = rolling_kinetic_energy_udf(
            _window(data["close"], 10)
        )
        features["e1f_kinetic_energy_20"] = rolling_kinetic_energy_udf(
            _window(data["close"], 20)
        )
        features["e1f_kinetic_energy_40"] = rolling_kinetic_energy_udf(
            _window(data["close"], 40)
        )
        features["e1f_linguistic_complexity_15"] = rolling_linguistic_complexity_udf(
            _window(data["close"], 15)
        )
        features["e1f_linguistic_complexity_25"] = rolling_linguistic_complexity_udf(
            _window(data["close"], 25)
        )
        features["e1f_linguistic_complexity_40"] = rolling_linguistic_complexity_udf(
            _window(data["close"], 40)
        )
        features["e1f_linguistic_complexity_80"] = rolling_linguistic_complexity_udf(
            _window(data["close"], 80)
        )
        features["e1f_muscle_force_20"] = rolling_muscle_force_udf(
            _window(data["close"], 20)
        )
        features["e1f_musical_tension_24"] = rolling_musical_tension_udf(
            _window(data["close"], 24)
        )
        features["e1f_musical_tension_48"] = rolling_musical_tension_udf(
            _window(data["close"], 48)
        )
        features["e1f_musical_tension_96"] = rolling_musical_tension_udf(
            _window(data["close"], 96)
        )
        features["e1f_network_clustering_100"] = rolling_network_clustering_udf(
            _window(data["close"], 100)
        )
        features["e1f_network_clustering_20"] = rolling_network_clustering_udf(
            _window(data["close"], 20)
        )
        features["e1f_network_clustering_30"] = rolling_network_clustering_udf(
            _window(data["close"], 30)
        )
        features["e1f_network_clustering_50"] = rolling_network_clustering_udf(
            _window(data["close"], 50)
        )
        features["e1f_network_density_100"] = rolling_network_density_udf(
            _window(data["close"], 100)
        )
        features["e1f_network_density_20"] = rolling_network_density_udf(
            _window(data["close"], 20)
        )
        features["e1f_network_density_30"] = rolling_network_density_udf(
            _window(data["close"], 30)
        )
        features["e1f_network_density_50"] = rolling_network_density_udf(
            _window(data["close"], 50)
        )
        features["e1f_rhythm_pattern_24"] = rolling_rhythm_pattern_udf(
            _window(data["close"], 24)
        )
        features["e1f_rhythm_pattern_48"] = rolling_rhythm_pattern_udf(
            _window(data["close"], 48)
        )
        features["e1f_rhythm_pattern_96"] = rolling_rhythm_pattern_udf(
            _window(data["close"], 96)
        )
        features["e1f_semantic_flow_15"] = rolling_semantic_flow_udf(
            _window(data["close"], 15)
        )
        features["e1f_semantic_flow_25"] = rolling_semantic_flow_udf(
            _window(data["close"], 25)
        )
        features["e1f_semantic_flow_40"] = rolling_semantic_flow_udf(
            _window(data["close"], 40)
        )
        features["e1f_semantic_flow_80"] = rolling_semantic_flow_udf(
            _window(data["close"], 80)
        )
        features["e1f_symmetry_measure_21"] = rolling_symmetry_measure_udf(
            _window(data["close"], 21)
        )
        features["e1f_symmetry_measure_34"] = rolling_symmetry_measure_udf(
            _window(data["close"], 34)
        )
        features["e1f_symmetry_measure_55"] = rolling_symmetry_measure_udf(
            _window(data["close"], 55)
        )
        features["e1f_symmetry_measure_89"] = rolling_symmetry_measure_udf(
            _window(data["close"], 89)
        )
        features["e1f_tonality_12"] = rolling_tonality_udf(_window(data["close"], 12))
        features["e1f_tonality_24"] = rolling_tonality_udf(_window(data["close"], 24))
        features["e1f_tonality_48"] = rolling_tonality_udf(_window(data["close"], 48))
        features["e1f_tonality_96"] = rolling_tonality_udf(_window(data["close"], 96))
        features["e1f_vocabulary_diversity_15"] = rolling_vocabulary_diversity_udf(
            _window(data["close"], 15)
        )
        features["e1f_vocabulary_diversity_25"] = rolling_vocabulary_diversity_udf(
            _window(data["close"], 25)
        )
        features["e1f_vocabulary_diversity_40"] = rolling_vocabulary_diversity_udf(
            _window(data["close"], 40)
        )
        features["e1f_vocabulary_diversity_80"] = rolling_vocabulary_diversity_udf(
            _window(data["close"], 80)
        )

        # --- その他の特徴量 (V5.1 L:1807-1811) ---
        features["atr"] = atr_13  # (e1c_atr_13の計算結果を流用)
        features["log_return"] = np.log(
            (data["close"][-1] + 1e-10) / (data["close"][-2] + 1e-10)
        )
        features["price_momentum"] = (
            data["close"][-1] - data["close"][-11]
            if len(data["close"]) > 10
            else np.nan
        )
        features["rolling_volatility"] = np.std(_window(close_pct, 20))  # (20期間)
        features["volume_ratio"] = data["volume"][-1] / (
            np.mean(_window(data["volume"], 20)) + 1e-10
        )

        # -------------------------------------------------------------

        # (V5.1 L:1813-1821 のロジックは calculate_feature_vector に移動済み)

        return features
