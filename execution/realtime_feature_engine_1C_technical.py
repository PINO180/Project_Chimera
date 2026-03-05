# realtime_feature_engine_1C_technical.py
# Project Cimera V5 - Feature Module 1C (Technical Indicators)

import numpy as np
import numba as nb
from numba import njit
import math
from typing import Dict

# ==================================================================
# 共通ヘルパー関数群 (Numpyベース)
# ==================================================================


def _window(arr: np.ndarray, window: int) -> np.ndarray:
    """配列の末尾から `window` 個の要素を取得"""
    if window <= 0:
        return np.array([], dtype=arr.dtype)
    if window > len(arr):
        return arr
    return arr[-window:]


def _array(arr: np.ndarray) -> np.ndarray:
    """配列全体をそのまま使用 (可読性向上のためのエイリアス)"""
    return arr


def _last(arr: np.ndarray) -> float:
    """配列の最新値（末尾）を取得。配列が空の場合はNaNを返す"""
    if len(arr) == 0:
        return np.nan
    return float(arr[-1])


def _ema(arr: np.ndarray, span: int) -> np.ndarray:
    """指数移動平均(EMA)のNumpy実装"""
    if len(arr) == 0:
        return np.zeros_like(arr, dtype=np.float64)

    alpha = 2.0 / (span + 1.0)
    ema = np.zeros_like(arr, dtype=np.float64)
    ema[0] = arr[0]

    for i in range(1, len(arr)):
        ema[i] = alpha * arr[i] + (1.0 - alpha) * ema[i - 1]
    return ema


def _rolling_mean(arr: np.ndarray, window: int) -> np.ndarray:
    """単純移動平均(SMA)の高速Numpy実装。先頭部分はNaNでパディング"""
    if len(arr) < window or window <= 0:
        return np.full(len(arr), np.nan)

    ret = np.cumsum(arr, dtype=float)
    ret[window:] = ret[window:] - ret[:-window]
    res = ret[window - 1 :] / window

    return np.concatenate((np.full(window - 1, np.nan), res))


def _pct_change_n_array(arr: np.ndarray, n: int) -> np.ndarray:
    """n期間の変化率(ROC)を配列全体に対して計算"""
    res = np.full_like(arr, np.nan)
    if len(arr) > n:
        safe_denom = arr[:-n].copy()
        safe_denom[safe_denom == 0] = 1e-10  # ゼロ除算防止
        res[n:] = (arr[n:] - safe_denom) / safe_denom
    return res


# ==================================================================
# 特徴量モジュールクラス (骨組み)
# ==================================================================


class FeatureModule1C:
    """
    Project Cimera V5 - 1Cテクニカル指標モジュール
    生き残りリストに指定された88個の特徴量のみを計算する純粋な数学的モジュール。
    """

    @staticmethod
    def calculate_features(data: Dict[str, np.ndarray]) -> Dict[str, float]:
        """
        メイン計算関数 (第4回で実装)
        """
        pass


# ==================================================================
# Numba JIT 関数群 その1 (ATR, RSI, ADX/DI, Aroon)
# ==================================================================


@njit(fastmath=True, cache=True)
def calculate_atr_numba(
    high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int
) -> np.ndarray:
    """ATR (Average True Range) 計算"""
    n = len(high)
    result = np.full(n, np.nan, dtype=np.float64)
    tr = np.zeros(n, dtype=np.float64)

    if n < period:
        return result

    # TR計算
    for i in range(n):
        if i == 0:
            tr[i] = high[i] - low[i]
        else:
            h_l = high[i] - low[i]
            h_pc = abs(high[i] - close[i - 1])
            l_pc = abs(low[i] - close[i - 1])
            tr[i] = max(h_l, max(h_pc, l_pc))

    # SMAベースのATR計算
    if period <= n:
        current_sum = 0.0
        for i in range(period):
            current_sum += tr[i]
        result[period - 1] = current_sum / period

        for i in range(period, n):
            current_sum = current_sum - tr[i - period] + tr[i]
            result[i] = current_sum / period

    return result


@njit(fastmath=True, cache=True)
def calculate_rsi_numba(prices: np.ndarray, period: int) -> np.ndarray:
    """RSI (Relative Strength Index) 計算"""
    n = len(prices)
    result = np.full(n, np.nan, dtype=np.float64)

    if n < period + 1:
        return result

    deltas = np.diff(prices)

    # 最初の平均ゲイン/ロス計算
    seed = deltas[:period]
    up = seed[seed >= 0].sum() / period
    down = -seed[seed < 0].sum() / period

    if down != 0:
        rs = up / down
        result[period] = 100.0 - 100.0 / (1.0 + rs)
    else:
        result[period] = 100.0

    # スライディングウィンドウ計算(SMAベース)
    for i in range(period + 1, n):
        start_idx = i - period
        window_deltas = deltas[start_idx:i]

        gains = 0.0
        losses = 0.0
        for d in window_deltas:
            if d > 0:
                gains += d
            else:
                losses += abs(d)

        avg_gain = gains / period
        avg_loss = losses / period

        if avg_loss == 0:
            result[i] = 100.0
        else:
            rs = avg_gain / avg_loss
            result[i] = 100.0 - (100.0 / (1.0 + rs))

    return result


@njit(fastmath=True, cache=True)
def calculate_di_plus_numba(
    high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int
) -> np.ndarray:
    """DI+ 計算"""
    n = len(high)
    di_plus_arr = np.full(n, np.nan, dtype=np.float64)

    if n < period:
        return di_plus_arr

    tr = np.zeros(n)
    dm_plus = np.zeros(n)

    for i in range(n):
        if i == 0:
            tr[i] = high[i] - low[i]
            dm_plus[i] = 0.0
        else:
            h_l = high[i] - low[i]
            h_pc = abs(high[i] - close[i - 1])
            l_pc = abs(low[i] - close[i - 1])
            tr[i] = max(h_l, max(h_pc, l_pc))

            up = high[i] - high[i - 1]
            down = low[i - 1] - low[i]
            if up > down and up > 0:
                dm_plus[i] = up
            else:
                dm_plus[i] = 0.0

    sum_tr = 0.0
    sum_dm_p = 0.0

    for i in range(period):
        sum_tr += tr[i]
        sum_dm_p += dm_plus[i]

    for i in range(period, n):
        sum_tr = sum_tr - tr[i - period] + tr[i]
        sum_dm_p = sum_dm_p - dm_plus[i - period] + dm_plus[i]

        if sum_tr > 0:
            di_plus_arr[i] = 100 * sum_dm_p / sum_tr
        else:
            di_plus_arr[i] = 0.0

    return di_plus_arr


@njit(fastmath=True, cache=True)
def calculate_di_minus_numba(
    high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int
) -> np.ndarray:
    """DI- 計算"""
    n = len(high)
    di_minus_arr = np.full(n, np.nan, dtype=np.float64)

    if n < period:
        return di_minus_arr

    tr = np.zeros(n)
    dm_minus = np.zeros(n)

    for i in range(n):
        if i == 0:
            tr[i] = high[i] - low[i]
            dm_minus[i] = 0.0
        else:
            h_l = high[i] - low[i]
            h_pc = abs(high[i] - close[i - 1])
            l_pc = abs(low[i] - close[i - 1])
            tr[i] = max(h_l, max(h_pc, l_pc))

            up = high[i] - high[i - 1]
            down = low[i - 1] - low[i]
            if down > up and down > 0:
                dm_minus[i] = down
            else:
                dm_minus[i] = 0.0

    sum_tr = 0.0
    sum_dm_m = 0.0

    for i in range(period):
        sum_tr += tr[i]
        sum_dm_m += dm_minus[i]

    for i in range(period, n):
        sum_tr = sum_tr - tr[i - period] + tr[i]
        sum_dm_m = sum_dm_m - dm_minus[i - period] + dm_minus[i]

        if sum_tr > 0:
            di_minus_arr[i] = 100 * sum_dm_m / sum_tr
        else:
            di_minus_arr[i] = 0.0

    return di_minus_arr


@njit(fastmath=True, cache=True)
def calculate_adx_numba(
    high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int
) -> np.ndarray:
    """ADX 計算"""
    n = len(high)
    adx = np.full(n, np.nan, dtype=np.float64)

    if n < period * 2:
        return adx

    di_plus = calculate_di_plus_numba(high, low, close, period)
    di_minus = calculate_di_minus_numba(high, low, close, period)

    dx = np.full(n, np.nan, dtype=np.float64)
    for i in range(period, n):
        sum_di = di_plus[i] + di_minus[i]
        if sum_di > 0:
            dx[i] = 100 * abs(di_plus[i] - di_minus[i]) / sum_di
        else:
            dx[i] = 0.0

    start_adx = period * 2 - 1
    if start_adx < n:
        sum_dx = 0.0
        for i in range(period, start_adx + 1):
            if not np.isnan(dx[i]):
                sum_dx += dx[i]

        if not np.isnan(dx[start_adx]):
            adx[start_adx] = sum_dx / period

        for i in range(start_adx + 1, n):
            if not np.isnan(dx[i]) and not np.isnan(dx[i - period]):
                sum_dx = sum_dx - dx[i - period] + dx[i]
                adx[i] = sum_dx / period

    return adx


@njit(fastmath=True, cache=True)
def calculate_aroon_up_numba(high: np.ndarray, period: int) -> np.ndarray:
    """Aroon Up 計算"""
    n = len(high)
    result = np.full(n, np.nan, dtype=np.float64)

    if n < period:
        return result

    for i in range(period, n):
        start_idx = i - period + 1
        highest_idx = -1
        highest_val = -1e30

        for j in range(start_idx, i + 1):
            if high[j] > highest_val:
                highest_val = high[j]
                highest_idx = j

        periods_since = i - highest_idx
        result[i] = 100.0 * (period - periods_since) / period

    return result


@njit(fastmath=True, cache=True)
def calculate_aroon_down_numba(low: np.ndarray, period: int) -> np.ndarray:
    """Aroon Down 計算"""
    n = len(low)
    result = np.full(n, np.nan, dtype=np.float64)

    if n < period:
        return result

    for i in range(period, n):
        start_idx = i - period + 1
        lowest_idx = -1
        lowest_val = 1e30

        for j in range(start_idx, i + 1):
            if low[j] < lowest_val:
                lowest_val = low[j]
                lowest_idx = j

        periods_since = i - lowest_idx
        result[i] = 100.0 * (period - periods_since) / period

    return result


# ==================================================================
# Numba JIT 関数群 その2 (オシレーター、移動平均派生、トレンド指標)
# ==================================================================


@njit(fastmath=True, cache=True)
def calculate_stochastic_numba(
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    k_period: int,
    d_period: int = 3,
    slow_period: int = 3,
) -> np.ndarray:
    """Stochastic Oscillator 計算 (%K, %D, Slow %D)"""
    n = len(close)
    stoch_k = np.full(n, np.nan, dtype=np.float64)
    stoch_d = np.full(n, np.nan, dtype=np.float64)
    stoch_slow_d = np.full(n, np.nan, dtype=np.float64)

    if n < k_period:
        return stoch_slow_d

    # %K 計算
    for i in range(k_period - 1, n):
        start_idx = i - k_period + 1
        window_high = -1e30
        window_low = 1e30

        for j in range(start_idx, i + 1):
            if high[j] > window_high:
                window_high = high[j]
            if low[j] < window_low:
                window_low = low[j]

        if window_high - window_low > 0:
            stoch_k[i] = 100.0 * (close[i] - window_low) / (window_high - window_low)
        else:
            stoch_k[i] = 50.0

    # %D 計算 (SMA of %K)
    for i in range(k_period + d_period - 2, n):
        sum_val = 0.0
        valid_count = 0
        for j in range(d_period):
            val = stoch_k[i - j]
            if not np.isnan(val):
                sum_val += val
                valid_count += 1
        if valid_count == d_period:
            stoch_d[i] = sum_val / d_period

    if slow_period == 1:
        return stoch_d

    # Slow %D 計算 (SMA of %D)
    for i in range(k_period + d_period + slow_period - 3, n):
        sum_val = 0.0
        valid_count = 0
        for j in range(slow_period):
            val = stoch_d[i - j]
            if not np.isnan(val):
                sum_val += val
                valid_count += 1
        if valid_count == slow_period:
            stoch_slow_d[i] = sum_val / slow_period

    # 呼び出し元で必要な配列を柔軟に取得するため、タプルではなく最も平滑化されたものを返す（元ロジック踏襲）
    # ※メイン側で %K や %D を使いたい場合は、k_periodのみ指定(slow_period=1)で呼び出すよう工夫します
    return stoch_slow_d


@njit(fastmath=True, cache=True)
def calculate_williams_r_numba(
    high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int
) -> np.ndarray:
    """Williams %R 計算"""
    n = len(close)
    result = np.full(n, np.nan, dtype=np.float64)

    if n < period:
        return result

    for i in range(period - 1, n):
        start_idx = i - period + 1
        highest = -1e30
        lowest = 1e30

        for j in range(start_idx, i + 1):
            if high[j] > highest:
                highest = high[j]
            if low[j] < lowest:
                lowest = low[j]

        if highest - lowest > 0:
            result[i] = -100.0 * (highest - close[i]) / (highest - lowest)
        else:
            result[i] = -50.0

    return result


@njit(fastmath=True, cache=True)
def calculate_trix_numba(prices: np.ndarray, period: int) -> np.ndarray:
    """TRIX 計算"""
    n = len(prices)
    trix = np.full(n, np.nan, dtype=np.float64)

    if n < period * 3:
        return trix

    alpha = 2.0 / (period + 1.0)
    ema1 = np.zeros(n)
    ema2 = np.zeros(n)
    ema3 = np.zeros(n)

    curr = prices[0]
    ema1[0] = curr
    for i in range(1, n):
        curr = alpha * prices[i] + (1.0 - alpha) * curr
        ema1[i] = curr

    curr = ema1[0]
    ema2[0] = curr
    for i in range(1, n):
        curr = alpha * ema1[i] + (1.0 - alpha) * curr
        ema2[i] = curr

    curr = ema2[0]
    ema3[0] = curr
    for i in range(1, n):
        curr = alpha * ema2[i] + (1.0 - alpha) * curr
        ema3[i] = curr

    for i in range(1, n):
        if i < period * 3:
            trix[i] = np.nan
            continue
        prev = ema3[i - 1]
        if prev != 0:
            trix[i] = 10000.0 * (ema3[i] - prev) / prev
        else:
            trix[i] = 0.0

    return trix


@njit(fastmath=True, cache=True)
def calculate_tsi_numba(
    prices: np.ndarray, long_period: int, short_period: int
) -> np.ndarray:
    """TSI (True Strength Index) 計算"""
    n = len(prices)
    tsi = np.full(n, np.nan, dtype=np.float64)

    if n < long_period + short_period:
        return tsi

    mom = np.zeros(n)
    abs_mom = np.zeros(n)
    for i in range(1, n):
        val = prices[i] - prices[i - 1]
        mom[i] = val
        abs_mom[i] = abs(val)

    def calc_ema_arr(arr, span):
        res = np.zeros(len(arr))
        alpha = 2.0 / (span + 1.0)
        curr = arr[0]
        res[0] = curr
        for i in range(1, len(arr)):
            curr = alpha * arr[i] + (1.0 - alpha) * curr
            res[i] = curr
        return res

    ema1_mom = calc_ema_arr(mom, long_period)
    ema1_abs = calc_ema_arr(abs_mom, long_period)

    ema2_mom = calc_ema_arr(ema1_mom, short_period)
    ema2_abs = calc_ema_arr(ema1_abs, short_period)

    for i in range(n):
        if ema2_abs[i] != 0:
            tsi[i] = 100.0 * ema2_mom[i] / ema2_abs[i]
        else:
            tsi[i] = 0.0

    return tsi


@njit(fastmath=True, cache=True)
def calculate_ultimate_oscillator_numba(
    high: np.ndarray, low: np.ndarray, close: np.ndarray, volume: np.ndarray
) -> np.ndarray:
    """Ultimate Oscillator 計算"""
    n = len(close)
    uo = np.full(n, np.nan, dtype=np.float64)

    p1, p2, p3 = 7, 14, 28
    w1, w2, w3 = 4.0, 2.0, 1.0

    if n < p3 + 1:
        return uo

    bp = np.zeros(n)
    tr = np.zeros(n)

    for i in range(1, n):
        curr_close = close[i]
        prev_close = close[i - 1]
        curr_low = low[i]
        curr_high = high[i]

        bp[i] = curr_close - min(curr_low, prev_close)
        tr[i] = max(
            curr_high - curr_low,
            max(abs(curr_high - prev_close), abs(curr_low - prev_close)),
        )

    for i in range(p3, n):
        bp_sum1, tr_sum1 = 0.0, 0.0
        bp_sum2, tr_sum2 = 0.0, 0.0
        bp_sum3, tr_sum3 = 0.0, 0.0

        for j in range(p1):
            bp_sum1 += bp[i - j]
            tr_sum1 += tr[i - j]
        for j in range(p2):
            bp_sum2 += bp[i - j]
            tr_sum2 += tr[i - j]
        for j in range(p3):
            bp_sum3 += bp[i - j]
            tr_sum3 += tr[i - j]

        avg1 = (bp_sum1 / tr_sum1) if tr_sum1 > 0 else 0.0
        avg2 = (bp_sum2 / tr_sum2) if tr_sum2 > 0 else 0.0
        avg3 = (bp_sum3 / tr_sum3) if tr_sum3 > 0 else 0.0

        uo[i] = 100.0 * (avg1 * w1 + avg2 * w2 + avg3 * w3) / (w1 + w2 + w3)

    return uo


@njit(fastmath=True, cache=True)
def calculate_hma_numba(prices: np.ndarray, period: int) -> np.ndarray:
    """Hull Moving Average (HMA) 計算"""
    n = len(prices)
    hma = np.full(n, np.nan, dtype=np.float64)
    half_period = int(period / 2)
    sqrt_period = int(math.sqrt(period))

    if n < period + sqrt_period:
        return hma

    def calc_wma_array(arr, w_period):
        res = np.full(len(arr), np.nan, dtype=np.float64)
        if len(arr) < w_period:
            return res
        denom = w_period * (w_period + 1) / 2.0
        for i in range(w_period - 1, len(arr)):
            numerator = 0.0
            for j in range(w_period):
                weight = j + 1
                idx = i - w_period + 1 + j
                numerator += arr[idx] * weight
            res[i] = numerator / denom
        return res

    wma_half = calc_wma_array(prices, half_period)
    wma_full = calc_wma_array(prices, period)

    raw_hma = np.full(n, np.nan, dtype=np.float64)
    for i in range(n):
        if not np.isnan(wma_half[i]) and not np.isnan(wma_full[i]):
            raw_hma[i] = 2 * wma_half[i] - wma_full[i]

    return calc_wma_array(raw_hma, sqrt_period)


@njit(fastmath=True, cache=True)
def wma_rolling_numba(arr: np.ndarray, window: int) -> float:
    """WMA (単一ポイント用) 計算"""
    n = len(arr)
    if n < window:
        return np.nan
    start_idx = n - window
    numerator = 0.0
    denominator = 0.0
    for i in range(window):
        val = arr[start_idx + i]
        if np.isnan(val):
            return np.nan
        weight = i + 1.0
        numerator += val * weight
        denominator += weight

    if denominator == 0:
        return np.nan
    return numerator / denominator


@njit(fastmath=True, cache=True)
def rolling_mean_numba(arr: np.ndarray, window: int) -> float:
    """SMA (単一ポイント用高速版)"""
    n = len(arr)
    if n < window:
        return np.nan
    s = 0.0
    for i in range(n - window, n):
        s += arr[i]
    return s / window


@njit(fastmath=True, cache=True)
def rolling_trend_consistency_numba(close: np.ndarray, period: int) -> float:
    """Trend Consistency 計算"""
    n = len(close)
    if n < period + 2:
        return np.nan

    changes_sum = 0.0
    count = 0

    for i in range(n - period, n):
        diff_curr = close[i] - close[i - 1]
        diff_prev = close[i - 1] - close[i - 2]
        sign_curr = np.sign(diff_curr)
        sign_prev = np.sign(diff_prev)
        change = abs(sign_curr - sign_prev)
        changes_sum += change
        count += 1

    if count > 0:
        mean_changes = changes_sum / count
        return 1.0 - mean_changes / 2.0
    else:
        return np.nan


# ==================================================================
# メイン計算モジュール本体
# ==================================================================


class FeatureModule1C:
    """
    Project Cimera V5 - 1Cテクニカル指標モジュール
    生き残りリストに指定された88個の特徴量のみを計算する純粋な数学的モジュール。
    """

    @staticmethod
    def calculate_features(data: Dict[str, np.ndarray]) -> Dict[str, float]:
        features = {}

        # 安全な参照用変数
        close_arr = data["close"]
        high_arr = data["high"]
        low_arr = data["low"]
        volume_arr = data["volume"]
        current_close = close_arr[-1] if len(close_arr) > 0 else np.nan

        # ---------------------------------------------------------
        # 1. ATR系 (16個)
        # ---------------------------------------------------------
        atr_13_arr = calculate_atr_numba(high_arr, low_arr, close_arr, 13)
        atr_21_arr = calculate_atr_numba(high_arr, low_arr, close_arr, 21)
        atr_34_arr = calculate_atr_numba(high_arr, low_arr, close_arr, 34)
        atr_55_arr = calculate_atr_numba(high_arr, low_arr, close_arr, 55)

        atr_13 = _last(atr_13_arr)

        features["e1c_atr_13"] = atr_13
        features["e1c_atr_lower_13_1.5"] = current_close - (atr_13 * 1.5)
        features["e1c_atr_lower_13_2.0"] = current_close - (atr_13 * 2.0)
        features["e1c_atr_pct_13"] = (
            (atr_13 / current_close) * 100 if current_close else np.nan
        )
        features["e1c_atr_trend_13"] = (
            atr_13_arr[-1] - atr_13_arr[-2] if len(atr_13_arr) >= 2 else np.nan
        )
        features["e1c_atr_upper_13_1.5"] = current_close + (atr_13 * 1.5)
        features["e1c_atr_upper_13_2.0"] = current_close + (atr_13 * 2.0)
        features["e1c_atr_upper_13_2.5"] = current_close + (atr_13 * 2.5)
        features["e1c_atr_volatility_13"] = np.std(_window(atr_13_arr, 13))
        features["e1c_atr_volatility_21"] = np.std(_window(atr_21_arr, 21))
        features["e1c_atr_volatility_34"] = np.std(_window(atr_34_arr, 34))
        features["e1c_atr_volatility_55"] = np.std(_window(atr_55_arr, 55))

        # ---------------------------------------------------------
        # 2. ADX, DI, Aroon系 (7個)
        # ---------------------------------------------------------
        features["e1c_adx_13"] = _last(
            calculate_adx_numba(high_arr, low_arr, close_arr, 13)
        )

        aroon_up_14 = _last(calculate_aroon_up_numba(high_arr, 14))
        aroon_down_14 = _last(calculate_aroon_down_numba(low_arr, 14))
        features["e1c_aroon_up_14"] = aroon_up_14
        features["e1c_aroon_down_14"] = aroon_down_14
        features["e1c_aroon_oscillator_14"] = aroon_up_14 - aroon_down_14

        features["e1c_di_minus_13"] = _last(
            calculate_di_minus_numba(high_arr, low_arr, close_arr, 13)
        )
        features["e1c_di_plus_13"] = _last(
            calculate_di_plus_numba(high_arr, low_arr, close_arr, 13)
        )
        features["e1c_di_plus_21"] = _last(
            calculate_di_plus_numba(high_arr, low_arr, close_arr, 21)
        )

        # ---------------------------------------------------------
        # 3. ボリンジャーバンド系 (14個)
        # ---------------------------------------------------------
        bb_mean_20, bb_std_20 = (
            np.mean(_window(close_arr, 20)),
            np.std(_window(close_arr, 20)),
        )
        bb_mean_30, bb_std_30 = (
            np.mean(_window(close_arr, 30)),
            np.std(_window(close_arr, 30)),
        )
        bb_mean_50, bb_std_50 = (
            np.mean(_window(close_arr, 50)),
            np.std(_window(close_arr, 50)),
        )

        # Period 20, 2.5
        features["e1c_bb_lower_20_2.5"] = bb_mean_20 - 2.5 * bb_std_20
        features["e1c_bb_upper_20_2.5"] = bb_mean_20 + 2.5 * bb_std_20
        features["e1c_bb_percent_20_2.5"] = (
            current_close - features["e1c_bb_lower_20_2.5"]
        ) / (
            (features["e1c_bb_upper_20_2.5"] - features["e1c_bb_lower_20_2.5"]) + 1e-10
        )

        # Period 30, 2.5
        features["e1c_bb_lower_30_2.5"] = bb_mean_30 - 2.5 * bb_std_30
        width_30_25 = 5.0 * bb_std_30
        features["e1c_bb_width_30_2.5"] = width_30_25
        features["e1c_bb_width_pct_30_2.5"] = (width_30_25 / (bb_mean_30 + 1e-10)) * 100
        features["e1c_bb_percent_30_2.5"] = (
            current_close - features["e1c_bb_lower_30_2.5"]
        ) / (width_30_25 + 1e-10)
        features["e1c_bb_position_30_2.5"] = (current_close - bb_mean_30) / (
            bb_std_30 + 1e-10
        )

        # Period 50, 2.0/2.5/3.0
        features["e1c_bb_percent_50_2"] = (
            current_close - (bb_mean_50 - 2.0 * bb_std_50)
        ) / (4.0 * bb_std_50 + 1e-10)

        features["e1c_bb_lower_50_2.5"] = bb_mean_50 - 2.5 * bb_std_50
        features["e1c_bb_upper_50_2.5"] = bb_mean_50 + 2.5 * bb_std_50
        features["e1c_bb_percent_50_2.5"] = (
            current_close - features["e1c_bb_lower_50_2.5"]
        ) / (5.0 * bb_std_50 + 1e-10)

        features["e1c_bb_lower_50_3"] = bb_mean_50 - 3.0 * bb_std_50
        features["e1c_bb_upper_50_3"] = bb_mean_50 + 3.0 * bb_std_50

        # ---------------------------------------------------------
        # 4. RSI系 (7個)
        # ---------------------------------------------------------
        rsi_14_arr = calculate_rsi_numba(close_arr, 14)
        rsi_21_arr = calculate_rsi_numba(close_arr, 21)

        features["e1c_rsi_14"] = _last(rsi_14_arr)
        features["e1c_rsi_momentum_14"] = (
            rsi_14_arr[-1] - rsi_14_arr[-2] if len(rsi_14_arr) >= 2 else np.nan
        )
        features["e1c_rsi_momentum_21"] = (
            rsi_21_arr[-1] - rsi_21_arr[-2] if len(rsi_21_arr) >= 2 else np.nan
        )

        rsi_14_win = _window(rsi_14_arr, 14)
        rsi_21_win = _window(rsi_21_arr, 21)

        rsi_14_min, rsi_14_max = np.nanmin(rsi_14_win), np.nanmax(rsi_14_win)
        rsi_21_min, rsi_21_max = np.nanmin(rsi_21_win), np.nanmax(rsi_21_win)

        features["e1c_stochastic_rsi_14"] = (
            (rsi_14_arr[-1] - rsi_14_min) / (rsi_14_max - rsi_14_min + 1e-10)
        ) * 100
        features["e1c_stochastic_rsi_21"] = (
            (rsi_21_arr[-1] - rsi_21_min) / (rsi_21_max - rsi_21_min + 1e-10)
        ) * 100

        features["e1c_rsi_divergence_14"] = (
            ((current_close - close_arr[-15]) / (close_arr[-15] + 1e-10))
            - ((rsi_14_arr[-1] - rsi_14_arr[-15]) / 50 - 1)
            if len(close_arr) >= 15
            else np.nan
        )
        features["e1c_rsi_divergence_21"] = (
            ((current_close - close_arr[-22]) / (close_arr[-22] + 1e-10))
            - ((rsi_21_arr[-1] - rsi_21_arr[-22]) / 50 - 1)
            if len(close_arr) >= 22
            else np.nan
        )

        # ---------------------------------------------------------
        # 5. MACD, STC, Coppock系 (8個)
        # ---------------------------------------------------------
        ema_12 = _ema(close_arr, 12)
        ema_26 = _ema(close_arr, 26)
        ema_19 = _ema(close_arr, 19)
        ema_39 = _ema(close_arr, 39)
        ema_5 = _ema(close_arr, 5)
        ema_35 = _ema(close_arr, 35)

        macd_12_26 = ema_12 - ema_26
        macd_19_39 = ema_19 - ema_39
        macd_5_35 = ema_5 - ema_35

        features["e1c_macd_12_26"] = _last(macd_12_26)
        features["e1c_macd_19_39"] = _last(macd_19_39)
        features["e1c_macd_histogram_12_26_9"] = _last(macd_12_26) - _last(
            _ema(macd_12_26, 9)
        )
        features["e1c_macd_histogram_19_39_9"] = _last(macd_19_39) - _last(
            _ema(macd_19_39, 9)
        )
        features["e1c_macd_histogram_5_35_5"] = _last(macd_5_35) - _last(
            _ema(macd_5_35, 5)
        )

        stc_12_26_win = _window(macd_12_26, 9)
        stc_12_26_min, stc_12_26_max = (
            np.nanmin(stc_12_26_win),
            np.nanmax(stc_12_26_win),
        )
        features["e1c_schaff_trend_cycle_12_26_9"] = (
            (_last(macd_12_26) - stc_12_26_min)
            / (stc_12_26_max - stc_12_26_min + 1e-10)
        ) * 100

        stc_23_50 = _ema(close_arr, 23) - _ema(close_arr, 50)
        stc_23_50_win = _window(stc_23_50, 10)
        stc_23_50_min, stc_23_50_max = (
            np.nanmin(stc_23_50_win),
            np.nanmax(stc_23_50_win),
        )
        features["e1c_schaff_trend_cycle_23_50_10"] = (
            (_last(stc_23_50) - stc_23_50_min) / (stc_23_50_max - stc_23_50_min + 1e-10)
        ) * 100

        if len(close_arr) >= 24:
            roc_11 = _pct_change_n_array(close_arr, 11) * 100
            roc_14 = _pct_change_n_array(close_arr, 14) * 100
            features["e1c_coppock_curve"] = np.mean(_window(roc_11 + roc_14, 10))
        else:
            features["e1c_coppock_curve"] = np.nan

        # ---------------------------------------------------------
        # 6. オシレーター、トレンド、モメンタム系 (36個)
        # ---------------------------------------------------------
        for p in [10, 20, 30, 50]:
            features[f"e1c_momentum_{p}"] = (
                current_close - close_arr[-1 - p] if len(close_arr) > p else np.nan
            )
        for p in [20, 50]:
            features[f"e1c_rate_of_change_{p}"] = (
                ((current_close - close_arr[-1 - p]) / (close_arr[-1 - p] + 1e-10))
                * 100
                if len(close_arr) > p
                else np.nan
            )

        # Stochastic
        stoch_k_14_arr = calculate_stochastic_numba(
            high_arr, low_arr, close_arr, 14, 1, 1
        )  # %K
        stoch_k_21_arr = calculate_stochastic_numba(
            high_arr, low_arr, close_arr, 21, 1, 1
        )  # %K

        features["e1c_stoch_k_14"] = _last(stoch_k_14_arr)
        features["e1c_stoch_d_14_3"] = np.mean(_window(stoch_k_14_arr, 3))
        features["e1c_stoch_d_21_5"] = np.mean(_window(stoch_k_21_arr, 5))

        stoch_d_21_5_arr = _rolling_mean(stoch_k_21_arr, 5)
        features["e1c_stoch_slow_d_21_5_5"] = np.mean(_window(stoch_d_21_5_arr, 5))

        # DPO: 元スクリプト準拠 close - SMA(period).shift(-lookback)
        # lookback = period // 2 + 1
        # リアルタイム: 現在のcloseから lookback バー前のSMAを引く
        # sma対象: close_arr[-(period + lookback) : -lookback] の末尾period個
        for p in [20, 30, 50]:
            _lookback = p // 2 + 1
            _required = p + _lookback
            if len(close_arr) >= _required:
                _sma_window = close_arr[-_required : len(close_arr) - _lookback]
                _sma_val = (
                    float(np.mean(_sma_window[-p:]))
                    if len(_sma_window) >= p
                    else np.nan
                )
                features[f"e1c_dpo_{p}"] = (
                    (current_close - _sma_val) if not np.isnan(_sma_val) else np.nan
                )
            else:
                features[f"e1c_dpo_{p}"] = np.nan

        # EMA Deviations
        for p in [10, 20, 50, 100, 200]:
            ema_val = _last(_ema(close_arr, p))
            features[f"e1c_ema_deviation_{p}"] = (
                (current_close - ema_val) / (ema_val + 1e-10)
            ) * 100

        # SMA Deviations & SMA
        sma_100 = rolling_mean_numba(_window(close_arr, 100), 100)
        features["e1c_sma_100"] = sma_100
        for p in [50, 100, 200]:
            sma_val = rolling_mean_numba(_window(close_arr, p), p)
            features[f"e1c_sma_deviation_{p}"] = (
                ((current_close - sma_val) / (sma_val + 1e-10)) * 100
                if not np.isnan(sma_val) and sma_val != 0
                else np.nan
            )

        # RVI
        for p in [10, 14, 20]:
            rvi_arr = _rolling_mean(close_arr - data["open"], p) / (
                _rolling_mean(high_arr - low_arr, p) + 1e-10
            )
            features[f"e1c_relative_vigor_index_{p}"] = _last(rvi_arr)
            features[f"e1c_rvi_signal_{p}"] = np.mean(_window(rvi_arr, 4))

        # KST
        if len(close_arr) >= 31:
            roc_10 = (current_close - close_arr[-11]) / (close_arr[-11] + 1e-10)
            roc_15 = (current_close - close_arr[-16]) / (close_arr[-16] + 1e-10)
            roc_20 = (current_close - close_arr[-21]) / (close_arr[-21] + 1e-10)
            roc_30 = (current_close - close_arr[-31]) / (close_arr[-31] + 1e-10)
            kst_val = (roc_10 * 1 + roc_15 * 2 + roc_20 * 3 + roc_30 * 4) / 10 * 100
            features["e1c_kst"] = kst_val
            # KST Signal = 9期間SMA of KST: 元スクリプト準拠 kst.rolling_mean(9)
            # リアルタイム実装: 末尾9バーのKST値の平均
            if len(close_arr) >= 39:  # 30(最大ROC期間) + 9(signal期間) - 1
                kst_arr = np.full(9, np.nan)
                for _k in range(9):
                    _offset = _k
                    _c = close_arr[-1 - _offset] if len(close_arr) > _offset else np.nan
                    if np.isnan(_c):
                        kst_arr[_k] = np.nan
                        continue
                    _roc10 = (
                        (_c - close_arr[-11 - _offset])
                        / (close_arr[-11 - _offset] + 1e-10)
                        if len(close_arr) > 10 + _offset
                        else np.nan
                    )
                    _roc15 = (
                        (_c - close_arr[-16 - _offset])
                        / (close_arr[-16 - _offset] + 1e-10)
                        if len(close_arr) > 15 + _offset
                        else np.nan
                    )
                    _roc20 = (
                        (_c - close_arr[-21 - _offset])
                        / (close_arr[-21 - _offset] + 1e-10)
                        if len(close_arr) > 20 + _offset
                        else np.nan
                    )
                    _roc30 = (
                        (_c - close_arr[-31 - _offset])
                        / (close_arr[-31 - _offset] + 1e-10)
                        if len(close_arr) > 30 + _offset
                        else np.nan
                    )
                    if any(np.isnan(v) for v in [_roc10, _roc15, _roc20, _roc30]):
                        kst_arr[_k] = np.nan
                    else:
                        kst_arr[_k] = (
                            (_roc10 * 1 + _roc15 * 2 + _roc20 * 3 + _roc30 * 4)
                            / 10
                            * 100
                        )
                valid = kst_arr[~np.isnan(kst_arr)]
                features["e1c_kst_signal"] = (
                    float(np.mean(valid)) if len(valid) > 0 else np.nan
                )
            else:
                features["e1c_kst_signal"] = np.nan
        else:
            features["e1c_kst"] = np.nan
            features["e1c_kst_signal"] = np.nan

        # Trend Strength & Consistency
        for p in [20, 50, 100]:
            if p != 100:  # e1c_trend_strength_100はリスト外
                features[f"e1c_trend_strength_{p}"] = 1.0 / (
                    np.std(_window(close_arr, p)) + 1e-10
                )
            features[f"e1c_trend_consistency_{p}"] = rolling_trend_consistency_numba(
                _window(close_arr, p + 10), p
            )

        # Others
        features["e1c_hma_21"] = _last(calculate_hma_numba(close_arr, 21))
        features["e1c_trix_14"] = _last(calculate_trix_numba(close_arr, 14))
        features["e1c_tsi_13"] = _last(calculate_tsi_numba(close_arr, 13, 6))
        features["e1c_ultimate_oscillator"] = _last(
            calculate_ultimate_oscillator_numba(
                high_arr, low_arr, close_arr, volume_arr
            )
        )
        features["e1c_williams_r_14"] = _last(
            calculate_williams_r_numba(high_arr, low_arr, close_arr, 14)
        )
        features["e1c_wma_200"] = rolling_mean_numba(_window(close_arr, 200), 200)

        return features
