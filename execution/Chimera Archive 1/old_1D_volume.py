# realtime_feature_engine_1D_volume.py

import numpy as np
import numba as nb
from numba import njit
from typing import Dict

"""
Project Cimera V5 - Feature Engine Module 1D
【Volume, Volatility & Price Action】

対象プレフィックス: e1d_
概要: 出来高、ボラティリティ、およびプライスアクション（ローソク足）に関する
純粋な数学的計算のみを提供する独立モジュール。
"""

# ==================================================================
# 共通ヘルパー関数 (Numba JIT)
# ==================================================================


@njit(fastmath=True, cache=True)
def pct_change_numba(arr: np.ndarray) -> np.ndarray:
    """
    配列の騰落率（pct_change）を計算するヘルパー関数。
    ボラティリティ系指標（HV等）の計算前処理として使用します。
    先頭要素は np.nan となります。
    """
    n = len(arr)
    pct = np.full(n, np.nan, dtype=np.float64)
    if n < 2:
        return pct

    for i in range(1, n):
        prev = arr[i - 1]
        if prev != 0.0:
            pct[i] = (arr[i] - prev) / prev
        else:
            pct[i] = 0.0

    return pct


@njit(fastmath=True, cache=True)
def get_window_numba(arr: np.ndarray, window: int) -> np.ndarray:
    """
    配列の末尾から指定されたウィンドウサイズの要素を取得するヘルパー関数。
    データ不足時の安全なスライス操作を保証します。
    """
    n = len(arr)
    if window <= 0:
        return np.empty(0, dtype=arr.dtype)
    if window >= n:
        return arr
    return arr[n - window :]


# ==================================================================
# 1D用 Numba UDF群（前半：Volume・Flow系指標）
# ==================================================================


@njit(fastmath=True, cache=True)
def accumulation_distribution_udf(
    high: np.ndarray, low: np.ndarray, close: np.ndarray, volume: np.ndarray
) -> np.ndarray:
    """
    Accumulation/Distribution Line (Numba JIT) - 配列返し
    対応: e1d_accumulation_distribution
    """
    n = len(close)
    ad = np.zeros(n, dtype=np.float64)

    if n > 0:
        if high[0] != low[0]:
            clv = ((close[0] - low[0]) - (high[0] - close[0])) / (high[0] - low[0])
            ad[0] = clv * volume[0]

    for i in range(1, n):
        prev = ad[i - 1]
        if high[i] != low[i]:
            clv = ((close[i] - low[i]) - (high[i] - close[i])) / (high[i] - low[i])
            flow = clv * volume[i]
            ad[i] = prev + flow
        else:
            ad[i] = prev

    return ad


@njit(fastmath=True, cache=True)
def chaikin_volatility_udf(
    high: np.ndarray, low: np.ndarray, window: int
) -> np.ndarray:
    """
    Chaikin Volatility (Numba JIT) - 配列返し
    対応: e1d_chaikin_volatility_10
    """
    n = len(high)
    result = np.full(n, np.nan, dtype=np.float64)

    if n < window * 2:
        return result

    hl_range = high - low
    sma = np.full(n, np.nan, dtype=np.float64)
    current_sum = 0.0

    for i in range(window):
        if np.isfinite(hl_range[i]):
            current_sum += hl_range[i]

    sma[window - 1] = current_sum / window

    for i in range(window, n):
        val_out = hl_range[i - window]
        val_in = hl_range[i]
        if np.isfinite(val_in) and np.isfinite(val_out):
            current_sum = current_sum - val_out + val_in
            sma[i] = current_sum / window
        else:
            sma[i] = np.nan

    for i in range(window * 2 - 1, n):
        curr_sma = sma[i]
        prev_sma = sma[i - window]

        if not np.isnan(curr_sma) and not np.isnan(prev_sma) and prev_sma > 0:
            result[i] = (curr_sma - prev_sma) / prev_sma * 100.0

    return result


@njit(fastmath=True, cache=True)
def cmf_udf(
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    volume: np.ndarray,
    window: int,
) -> np.ndarray:
    """
    Chaikin Money Flow (Numba JIT) - 配列返し
    対応: e1d_cmf_13
    """
    n = len(close)
    cmf = np.full(n, np.nan, dtype=np.float64)

    if n < window:
        return cmf

    mf_vol = np.zeros(n)

    for i in range(n):
        if high[i] != low[i]:
            mul = ((close[i] - low[i]) - (high[i] - close[i])) / (high[i] - low[i])
            mf_vol[i] = mul * volume[i]
        else:
            mf_vol[i] = 0.0

    sum_mf_vol = 0.0
    sum_vol = 0.0

    for i in range(window):
        sum_mf_vol += mf_vol[i]
        sum_vol += volume[i]

    if sum_vol > 0:
        cmf[window - 1] = sum_mf_vol / sum_vol

    for i in range(window, n):
        sum_mf_vol = sum_mf_vol - mf_vol[i - window] + mf_vol[i]
        sum_vol = sum_vol - volume[i - window] + volume[i]

        if sum_vol > 0:
            cmf[i] = sum_mf_vol / sum_vol
        else:
            cmf[i] = 0.0

    return cmf


@njit(fastmath=True, cache=True)
def force_index_udf(close: np.ndarray, volume: np.ndarray) -> np.ndarray:
    """
    Force Index (Numba JIT) - 配列返し
    対応: e1d_force_index
    """
    n = len(close)
    fi = np.zeros(n, dtype=np.float64)

    for i in range(1, n):
        fi[i] = (close[i] - close[i - 1]) * volume[i]

    return fi


@njit(fastmath=True, cache=True)
def mass_index_udf(high: np.ndarray, low: np.ndarray, window: int) -> np.ndarray:
    """
    Mass Index (Numba JIT) - 配列返し
    対応: e1d_mass_index_20
    """
    n = len(high)
    result = np.full(n, np.nan, dtype=np.float64)

    if n < window + 16:
        return result

    hl_range = high - low

    def calc_ema(arr, span):
        res = np.zeros(len(arr))
        alpha = 2.0 / (span + 1.0)
        curr = arr[0] if np.isfinite(arr[0]) else 0.0
        res[0] = curr
        for i in range(1, len(arr)):
            val = arr[i] if np.isfinite(arr[i]) else curr
            curr = alpha * val + (1.0 - alpha) * curr
            res[i] = curr
        return res

    ema9 = calc_ema(hl_range, 9)
    ema_ema9 = calc_ema(ema9, 9)

    ratio = np.zeros(n)
    for i in range(n):
        if ema_ema9[i] != 0:
            ratio[i] = ema9[i] / ema_ema9[i]

    sum_ratio = 0.0
    for i in range(window):
        sum_ratio += ratio[i]

    for i in range(window, n):
        sum_ratio = sum_ratio - ratio[i - window] + ratio[i]
        result[i] = sum_ratio

    return result


@njit(fastmath=True, cache=True)
def mfi_udf(
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    volume: np.ndarray,
    window: int,
) -> np.ndarray:
    """
    Money Flow Index (Numba JIT) - 配列返し
    対応: e1d_mfi_13
    """
    n = len(close)
    mfi = np.full(n, np.nan, dtype=np.float64)

    if n < window + 1:
        return mfi

    tp = (high + low + close) / 3.0
    rmf = tp * volume

    pos_flow = np.zeros(n)
    neg_flow = np.zeros(n)

    for i in range(1, n):
        if tp[i] > tp[i - 1]:
            pos_flow[i] = rmf[i]
        elif tp[i] < tp[i - 1]:
            neg_flow[i] = rmf[i]

    sum_pos = 0.0
    sum_neg = 0.0

    for i in range(1, window + 1):
        sum_pos += pos_flow[i]
        sum_neg += neg_flow[i]

    if sum_neg > 0:
        mr = sum_pos / sum_neg
        mfi[window] = 100.0 - (100.0 / (1.0 + mr))
    elif sum_pos > 0:
        mfi[window] = 100.0
    else:
        mfi[window] = 50.0

    for i in range(window + 1, n):
        sum_pos = sum_pos - pos_flow[i - window] + pos_flow[i]
        sum_neg = sum_neg - neg_flow[i - window] + neg_flow[i]

        if sum_neg > 0:
            mr = sum_pos / sum_neg
            mfi[i] = 100.0 - (100.0 / (1.0 + mr))
        elif sum_pos > 0:
            mfi[i] = 100.0
        else:
            mfi[i] = 50.0

    return mfi


@njit(fastmath=True, cache=True)
def obv_udf(close: np.ndarray, volume: np.ndarray) -> np.ndarray:
    """
    On Balance Volume (Numba JIT) - 配列返し
    対応: e1d_obv
    """
    n = len(close)
    obv = np.zeros(n, dtype=np.float64)

    if n > 0:
        obv[0] = volume[0]

    for i in range(1, n):
        prev = obv[i - 1]
        if close[i] > close[i - 1]:
            obv[i] = prev + volume[i]
        elif close[i] < close[i - 1]:
            obv[i] = prev - volume[i]
        else:
            obv[i] = prev

    return obv


# ==================================================================
# 1D用 Numba UDF群（後半：Volatility・Price Action系指標）
# ==================================================================


@njit(fastmath=True, cache=True)
def candlestick_patterns_udf(
    open_prices: np.ndarray, high: np.ndarray, low: np.ndarray, close: np.ndarray
) -> np.ndarray:
    """
    ローソク足パターン認識 (Numba JIT) - 配列返し
    対応: e1d_candlestick_pattern
    """
    n = len(close)
    patterns = np.zeros(n, dtype=np.float64)

    for i in range(n):
        o = open_prices[i]
        h = high[i]
        l = low[i]
        c = close[i]

        body_size = abs(c - o)
        upper_shadow = h - max(o, c)
        lower_shadow = min(o, c) - l
        total_range = h - l

        if total_range <= 0:
            patterns[i] = 0.0
            continue

        body_ratio = body_size / total_range
        upper_shadow_ratio = upper_shadow / total_range
        lower_shadow_ratio = lower_shadow / total_range

        # Doji
        if body_ratio < 0.1:
            patterns[i] = 3.0
        # Hammer
        elif lower_shadow_ratio > 0.6 and upper_shadow_ratio < 0.1:
            patterns[i] = 1.0
        # Shooting Star
        elif upper_shadow_ratio > 0.6 and lower_shadow_ratio < 0.1:
            patterns[i] = 2.0
        # Bullish
        elif body_ratio > 0.6 and c > o:
            patterns[i] = 4.0
        # Bearish
        elif body_ratio > 0.6 and c < o:
            patterns[i] = 5.0
        else:
            patterns[i] = 0.0

    return patterns


@njit(fastmath=True, cache=True)
def hv_robust_udf(returns: np.ndarray) -> float:
    """
    ロバストボラティリティ (Numba JIT) - 単一値
    対応: e1d_hv_robust_10, 20, 30, 50, annual_252, e1d_hv_regime_50
    """
    if len(returns) < 5:
        return np.nan

    finite_returns = returns[np.isfinite(returns)]
    if len(finite_returns) < 5:
        return np.nan

    median_return = np.median(finite_returns)
    abs_deviations = np.abs(finite_returns - median_return)
    mad = np.median(abs_deviations)

    return mad * 1.4826


@njit(fastmath=True, cache=True)
def hv_standard_udf(returns: np.ndarray) -> float:
    """
    標準ヒストリカルボラティリティ (Numba JIT) - 単一値
    対応: e1d_hv_standard_10, 30, 50
    """
    if len(returns) < 2:
        return np.nan

    finite_returns = returns[np.isfinite(returns)]
    if len(finite_returns) < 2:
        return np.nan

    return np.std(finite_returns)
