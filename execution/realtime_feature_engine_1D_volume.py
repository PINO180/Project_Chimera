# realtime_feature_engine_1D_volume.py
# Project Cimera V5 - Feature Engine Module 1D【Volume, Volatility & Price Action】

import numpy as np
import numba as nb
from numba import njit
from typing import Dict


# ==================================================================
# 共通ヘルパー関数 (Numba JIT)
# ==================================================================


@njit(fastmath=True, cache=True)
def pct_change_numba(arr: np.ndarray) -> np.ndarray:

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


# ==================================================================
# 1D用 Numba UDF群（前半：Volume・Flow系指標）
# ==================================================================


@nb.njit(fastmath=True, cache=True, parallel=True)
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

    # Typical Price = (High + Low + Close) / 3
    for i in nb.prange(window, n):
        positive_flow = 0.0
        negative_flow = 0.0

        for j in range(i - window + 1, i + 1):
            if j == 0:
                continue

            if (
                np.isfinite(high[j])
                and np.isfinite(low[j])
                and np.isfinite(close[j])
                and np.isfinite(volume[j])
                and np.isfinite(high[j - 1])
                and np.isfinite(low[j - 1])
                and np.isfinite(close[j - 1])
            ):
                typical_price = (high[j] + low[j] + close[j]) / 3.0
                prev_typical_price = (high[j - 1] + low[j - 1] + close[j - 1]) / 3.0
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
def obv_udf(close: np.ndarray, volume: np.ndarray) -> np.ndarray:

    n = len(close)
    result = np.full(n, np.nan)

    if n < 2:
        return result

    result[0] = volume[0] if np.isfinite(volume[0]) else 0.0

    for i in nb.prange(1, n):
        prev_obv = result[i - 1] if np.isfinite(result[i - 1]) else 0.0

        if (
            np.isfinite(close[i])
            and np.isfinite(close[i - 1])
            and np.isfinite(volume[i])
        ):
            if close[i] > close[i - 1]:
                result[i] = prev_obv + volume[i]
            elif close[i] < close[i - 1]:
                result[i] = prev_obv - volume[i]
            else:
                result[i] = prev_obv
        else:
            result[i] = prev_obv

    return result


# ==================================================================
# 1D用 Numba UDF群（後半：Volatility・Price Action系指標）
# ==================================================================


@nb.njit(fastmath=True, cache=True, parallel=True)
def candlestick_patterns_udf(
    open_prices: np.ndarray, high: np.ndarray, low: np.ndarray, close: np.ndarray
) -> np.ndarray:

    n = len(close)
    result = np.full(n, 0)

    for i in nb.prange(n):
        if not (
            np.isfinite(open_prices[i])
            and np.isfinite(high[i])
            and np.isfinite(low[i])
            and np.isfinite(close[i])
        ):
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


@nb.njit(fastmath=True, cache=True)
def hv_robust_udf(returns: np.ndarray) -> float:

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


@nb.njit(fastmath=True, cache=True)
def hv_standard_udf(returns: np.ndarray) -> float:

    if len(returns) < 5:
        return np.nan

    finite_returns = returns[np.isfinite(returns)]
    if len(finite_returns) < 5:
        return np.nan

    # 標準偏差計算（不偏推定量: ddof=1）
    mean_return = np.mean(finite_returns)
    squared_deviations = (finite_returns - mean_return) ** 2
    variance = np.sum(squared_deviations) / (len(finite_returns) - 1)
    volatility = np.sqrt(variance)

    return volatility


# ==================================================================
# メイン計算モジュール
# ==================================================================


def _window(arr: np.ndarray, window: int) -> np.ndarray:
    """配列の末尾から `window` 個の要素を取得"""
    if window <= 0:
        return np.array([], dtype=arr.dtype)
    if window > len(arr):
        return arr
    return arr[-window:]


def _array(arr: np.ndarray) -> np.ndarray:

    return arr


def _last(arr: np.ndarray) -> float:

    if len(arr) == 0:
        return np.nan
    return float(arr[-1])


class FeatureModule1D:
    @staticmethod
    def calculate_features(data: Dict[str, np.ndarray]) -> Dict[str, float]:
        features = {}

        # 安全な参照用変数
        close_arr = data["close"]
        high_arr = data["high"]
        low_arr = data["low"]
        open_arr = data["open"]
        volume_arr = data["volume"]

        # pct_change (ボラティリティ系指標の前処理)
        close_pct = pct_change_numba(close_arr)

        # ---------------------------------------------------------
        # 1. Volume・Flow系指標
        # ---------------------------------------------------------
        features["e1d_mfi_13"] = _last(
            mfi_udf(
                _array(high_arr),
                _array(low_arr),
                _array(close_arr),
                _array(volume_arr),
                13,
            )
        )

        features["e1d_obv"] = _last(obv_udf(_array(close_arr), _array(volume_arr)))

        features["e1d_volume_price_trend"] = np.mean(
            _window(close_arr * volume_arr, 10)
        )

        # Polars準拠: ゼロ除算時は np.inf を返す
        vol_ma20 = np.mean(_window(volume_arr, 20))
        features["e1d_volume_ratio"] = (
            data["volume"][-1] / vol_ma20 if vol_ma20 != 0.0 else np.inf
        )

        # ---------------------------------------------------------
        # 2. Volatility系指標 (HV)
        # ---------------------------------------------------------
        features["e1d_hv_robust_20"] = hv_robust_udf(_window(close_pct, 20))
        features["e1d_hv_robust_30"] = hv_robust_udf(_window(close_pct, 30))
        features["e1d_hv_robust_50"] = hv_robust_udf(_window(close_pct, 50))

        features["e1d_hv_standard_10"] = hv_standard_udf(_window(close_pct, 10))
        features["e1d_hv_standard_30"] = hv_standard_udf(_window(close_pct, 30))
        features["e1d_hv_standard_50"] = hv_standard_udf(_window(close_pct, 50))

        # ---------------------------------------------------------
        # 3. Price Action系指標（ローソク足・価格位置）
        # ---------------------------------------------------------
        features["e1d_candlestick_pattern"] = _last(
            candlestick_patterns_udf(
                _array(open_arr),
                _array(high_arr),
                _array(low_arr),
                _array(close_arr),
            )
        )

        # Polars準拠: ゼロ除算時は np.inf を返す
        features["e1d_intraday_return"] = (
            (close_arr[-1] - open_arr[-1]) / open_arr[-1]
            if open_arr[-1] != 0.0
            else np.inf
        )

        hl_range = high_arr[-1] - low_arr[-1]

        features["e1d_lower_wick_ratio"] = (
            (min(open_arr[-1], close_arr[-1]) - low_arr[-1]) / hl_range
            if hl_range != 0.0
            else np.inf
        )

        features["e1d_overnight_gap"] = 0.0
        if len(close_arr) > 1:
            prev_close = close_arr[-2]
            features["e1d_overnight_gap"] = (
                (open_arr[-1] - prev_close) / prev_close
                if prev_close != 0.0
                else np.inf
            )

        features["e1d_price_location_hl"] = (
            (close_arr[-1] - low_arr[-1]) / hl_range if hl_range != 0.0 else np.inf
        )

        return features
