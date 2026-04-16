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
    """単純移動平均(SMA)の実装。先頭部分はNaNでパディング。
    NaN-aware: 窓内にNaNが1つでもあれば出力もNaN（Polars rolling_mean準拠）。
    先頭にNaNを含む配列（Numba出力など）でも正しく動作する。
    """
    if len(arr) < window or window <= 0:
        return np.full(len(arr), np.nan)

    out = np.full(len(arr), np.nan)
    for i in range(window - 1, len(arr)):
        w = arr[i - window + 1 : i + 1]
        if not np.any(np.isnan(w)):
            out[i] = np.mean(w)
    return out


def _pct_change_n_array(arr: np.ndarray, n: int) -> np.ndarray:
    """n期間の変化率(ROC)を配列全体に対して計算"""
    res = np.full_like(arr, np.nan)
    if len(arr) > n:
        denom = arr[:-n]
        # [QA-FIX] ゼロ除算時のinfを許容 (Polars完全準拠のため 1e-10 ガード撤廃)
        with np.errstate(divide="ignore", invalid="ignore"):
            res[n:] = (arr[n:] - denom) / denom
    return res


# ==================================================================
# Numba JIT 関数群 その1 (ATR, RSI, ADX/DI, Aroon)
# ==================================================================
# [FIX-01] 元スクリプトは @nb.guvectorize(..., nopython=True, cache=True)。
#          リアルタイム用に @njit へ変換するが、fastmath=True は
#          浮動小数点演算の厳密再現性を破壊するため、fastmath=False に統一する。
# ==================================================================


# [FIX-03] ATR: 元スクリプトと同様に各iで range(i-period+1, i+1) の全窓を再合算。
#          スライディングウィンドウ累積誤差を排除する。
@njit(fastmath=False, cache=True)
def calculate_atr_numba(
    high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int
) -> np.ndarray:
    """ATR (Average True Range) 計算
    [FIX-03] 元スクリプト準拠: 各インデックスで全窓のTRを再合算する方式
    """
    n = len(high)
    out = np.full(n, np.nan, dtype=np.float64)

    # TR計算 (元スクリプト準拠)
    tr = np.zeros(n, dtype=np.float64)
    for i in range(n):
        if i == 0:
            tr[i] = high[i] - low[i]
        else:
            h_l = high[i] - low[i]
            h_pc = abs(high[i] - close[i - 1])
            l_pc = abs(low[i] - close[i - 1])
            tr[i] = max(h_l, h_pc, l_pc)

    # ATR計算: 元スクリプトと同様に各iで全窓を再合算
    for i in range(n):
        if i < period:
            out[i] = np.nan
        else:
            sum_tr = 0.0
            for j in range(i - period + 1, i + 1):
                sum_tr += tr[j]
            out[i] = sum_tr / period

    return out


# [FIX-02] RSI: 元スクリプトと同様に各iで range(i-period+1, i+1) の
#          gains/lossesを全窓再計算するSMAベースのシンプルRSI。
#          seedベースのEMAスライディング方式は学習時と値が異なるため廃止。
@njit(fastmath=False, cache=True)
def calculate_rsi_numba(prices: np.ndarray, period: int) -> np.ndarray:
    """RSI (Relative Strength Index) 計算
    [FIX-02] 元スクリプト準拠: 各インデックスで全窓のgains/lossesを再計算するSMAベース
    """
    n = len(prices)
    out = np.full(n, np.nan, dtype=np.float64)

    for i in range(n):
        if i < period:
            out[i] = np.nan
        else:
            gains = 0.0
            losses = 0.0

            # 元スクリプト準拠: range(i - period + 1, i + 1) で差分計算
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

    return out


# [FIX-05] DI+: 有効開始を i < period でNaN、ループ範囲を元スクリプト準拠に修正
#          元スクリプト: for j in range(i-period+1, i+1) で全窓再合算
@njit(fastmath=False, cache=True)
def calculate_di_plus_numba(
    high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int
) -> np.ndarray:
    """DI+ 計算
    [FIX-05] 元スクリプト準拠: 有効開始 i < period でNaN、全窓再合算方式
    """
    n = len(high)
    out = np.full(n, np.nan, dtype=np.float64)

    # TR計算
    tr = np.zeros(n, dtype=np.float64)
    for i in range(n):
        if i == 0:
            tr[i] = high[i] - low[i]
        else:
            h_l = high[i] - low[i]
            h_pc = abs(high[i] - close[i - 1])
            l_pc = abs(low[i] - close[i - 1])
            tr[i] = max(h_l, h_pc, l_pc)

    # DM+ 計算
    dm_plus = np.zeros(n, dtype=np.float64)
    for i in range(1, n):
        up_move = high[i] - high[i - 1]
        down_move = low[i - 1] - low[i]
        if up_move > down_move and up_move > 0:
            dm_plus[i] = up_move
        else:
            dm_plus[i] = 0.0

    # DI+ 計算: 元スクリプト準拠の全窓再合算
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

    return out


# [FIX-05] Aroon Up: 有効開始を i < period - 1 でNaN、ループ範囲を元スクリプト準拠に修正
#          元スクリプト: highest_val = high[i] で初期化し range(i-period+1, i) を走査
@njit(fastmath=False, cache=True)
def calculate_aroon_up_numba(high: np.ndarray, period: int) -> np.ndarray:
    """Aroon Up 計算
    [FIX-05] 元スクリプト準拠: 有効開始 i < period-1 でNaN
             highest_val = high[i] で初期化, range(i-period+1, i) で走査
    """
    n = len(high)
    out = np.full(n, np.nan, dtype=np.float64)

    for i in range(n):
        if i < period - 1:  # 元スクリプト準拠: period - 1
            out[i] = np.nan
        else:
            highest_idx = i
            highest_val = high[i]  # 元スクリプト準拠: iを初期値とする

            # 元スクリプト準拠: range(i - period + 1, i) → iを含まない
            for j in range(i - period + 1, i):
                if high[j] > highest_val:
                    highest_val = high[j]
                    highest_idx = j

            periods_since = i - highest_idx
            out[i] = 100.0 * (period - periods_since) / period

    return out


# [FIX-05] Aroon Down: Aroon Upと同様の修正
@njit(fastmath=False, cache=True)
def calculate_aroon_down_numba(low: np.ndarray, period: int) -> np.ndarray:
    """Aroon Down 計算
    [FIX-05] 元スクリプト準拠: 有効開始 i < period-1 でNaN
             lowest_val = low[i] で初期化, range(i-period+1, i) で走査
    """
    n = len(low)
    out = np.full(n, np.nan, dtype=np.float64)

    for i in range(n):
        if i < period - 1:  # 元スクリプト準拠: period - 1
            out[i] = np.nan
        else:
            lowest_idx = i
            lowest_val = low[i]  # 元スクリプト準拠: iを初期値とする

            # 元スクリプト準拠: range(i - period + 1, i) → iを含まない
            for j in range(i - period + 1, i):
                if low[j] < lowest_val:
                    lowest_val = low[j]
                    lowest_idx = j

            periods_since = i - lowest_idx
            out[i] = 100.0 * (period - periods_since) / period

    return out


# ==================================================================
# Numba JIT 関数群 その2 (オシレーター、移動平均派生、トレンド指標)
# ==================================================================


@njit(fastmath=False, cache=True)
def calculate_stochastic_numba(
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    k_period: int,
    d_period: int = 3,
    slow_period: int = 3,
) -> np.ndarray:
    """Stochastic Oscillator 計算 (%K -> %D)
    元スクリプト準拠: for i in range(n) + if/else, k_values配列, count>0判定
    """
    n = len(high)
    out = np.full(n, np.nan, dtype=np.float64)

    if n < k_period:
        return out

    # %K計算 (元スクリプト準拠)
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

    # %D計算 (Simple MA of %K) (元スクリプト準拠: count>0判定)
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

    return out


@njit(fastmath=False, cache=True)
def calculate_williams_r_numba(
    high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int
) -> np.ndarray:
    """Williams %R 計算"""
    n = len(high)
    out = np.full(n, np.nan, dtype=np.float64)

    if n < period:
        return out

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

    return out


# [FIX-07] TRIX: 元スクリプト準拠の配列参照方式EMAに統一
@njit(fastmath=False, cache=True)
def calculate_trix_numba(prices: np.ndarray, period: int) -> np.ndarray:
    """TRIX 計算
    [FIX-07] 元スクリプト準拠: 配列参照方式のEMA計算
    """
    n = len(prices)
    out = np.full(n, np.nan, dtype=np.float64)

    # 元スクリプト準拠: ema配列を先に宣言し、その後alphaを定義する
    ema1 = np.zeros(n, dtype=np.float64)
    ema2 = np.zeros(n, dtype=np.float64)
    ema3 = np.zeros(n, dtype=np.float64)

    alpha = 2.0 / (period + 1.0)

    # 元スクリプト準拠: 配列参照によるEMA
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

    # TRIX計算 (元スクリプト準拠)
    for i in range(n):
        if i < period * 3:
            out[i] = np.nan
        elif ema3[i - 1] == 0:
            out[i] = 0.0
        else:
            out[i] = 10000 * (ema3[i] - ema3[i - 1]) / ema3[i - 1]

    return out


# [FIX-08] Ultimate Oscillator: i=0 の bp/tr 初期化を元スクリプト準拠で復元
@njit(fastmath=False, cache=True)
def calculate_ultimate_oscillator_numba(
    high: np.ndarray, low: np.ndarray, close: np.ndarray, volume: np.ndarray
) -> np.ndarray:
    """Ultimate Oscillator 計算
    [FIX-08] 元スクリプト準拠: i=0 の bp/tr 初期化を復元
    """
    n = len(high)
    out = np.full(n, np.nan, dtype=np.float64)

    periods = [7, 14, 28]
    weights = [4.0, 2.0, 1.0]

    # Buying Pressure and True Range計算 (元スクリプト準拠: i=0 を明示初期化)
    bp = np.zeros(n, dtype=np.float64)
    tr = np.zeros(n, dtype=np.float64)

    for i in range(n):
        if i == 0:
            bp[i] = close[i] - low[i]  # 元スクリプト準拠
            tr[i] = high[i] - low[i]  # 元スクリプト準拠
        else:
            bp[i] = close[i] - min(low[i], close[i - 1])
            h_l = high[i] - low[i]
            h_pc = abs(high[i] - close[i - 1])
            l_pc = abs(low[i] - close[i - 1])
            tr[i] = max(h_l, h_pc, l_pc)

    # Ultimate Oscillator計算 (元スクリプト準拠)
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

    return out


@njit(fastmath=False, cache=True)
def rolling_mean_numba(arr: np.ndarray, window: int) -> float:
    """SMA (単一ポイント用高速版)"""
    n = len(arr)
    if n < window:
        return np.nan
    s = 0.0
    for i in range(n - window, n):
        s += arr[i]
    return s / window


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
        # 1. ATR系 (厳選5個)
        # ---------------------------------------------------------
        atr_13_arr = calculate_atr_numba(high_arr, low_arr, close_arr, 13)
        atr_21_arr = calculate_atr_numba(high_arr, low_arr, close_arr, 21)
        atr_55_arr = calculate_atr_numba(high_arr, low_arr, close_arr, 55)

        atr_13 = _last(atr_13_arr)

        features["e1c_atr_pct_13"] = (
            (atr_13 / current_close) * 100 if current_close else np.nan
        )
        features["e1c_atr_trend_13"] = (
            atr_13_arr[-1] - atr_13_arr[-2] if len(atr_13_arr) >= 2 else np.nan
        )

        # ▼▼ 修正前: 配列長チェックがない
        # features["e1c_atr_volatility_13"] = float(
        #     np.std(_window(atr_13_arr, 13), ddof=1)
        # )
        # features["e1c_atr_volatility_21"] = float(
        #     np.std(_window(atr_21_arr, 21), ddof=1)
        # )
        # features["e1c_atr_volatility_55"] = float(
        #     np.std(_window(atr_55_arr, 55), ddof=1)
        # )

        # ▼▼ 修正後: 配列長が2以上の時のみ標準偏差(ddof=1)を計算
        w_atr_13 = _window(atr_13_arr, 13)
        features["e1c_atr_volatility_13"] = (
            float(np.std(w_atr_13, ddof=1)) if len(w_atr_13) >= 2 else np.nan
        )

        w_atr_21 = _window(atr_21_arr, 21)
        features["e1c_atr_volatility_21"] = (
            float(np.std(w_atr_21, ddof=1)) if len(w_atr_21) >= 2 else np.nan
        )

        w_atr_55 = _window(atr_55_arr, 55)
        features["e1c_atr_volatility_55"] = (
            float(np.std(w_atr_55, ddof=1)) if len(w_atr_55) >= 2 else np.nan
        )

        # ---------------------------------------------------------
        # 2. ADX, DI, Aroon系 (厳選3個)
        # ---------------------------------------------------------
        aroon_up_14 = _last(calculate_aroon_up_numba(high_arr, 14))
        aroon_down_14 = _last(calculate_aroon_down_numba(low_arr, 14))
        features["e1c_aroon_down_14"] = aroon_down_14
        features["e1c_aroon_oscillator_14"] = aroon_up_14 - aroon_down_14

        features["e1c_di_plus_21"] = _last(
            calculate_di_plus_numba(high_arr, low_arr, close_arr, 21)
        )

        # ---------------------------------------------------------
        # 3. ボリンジャーバンド系 (厳選3個)
        # ---------------------------------------------------------
        # ▼▼ 修正前: 配列長が0または1の時に落ちるリスク
        # bb_mean_30, bb_std_30 = (
        #     np.mean(_window(close_arr, 30)),
        #     np.std(_window(close_arr, 30), ddof=1),
        # )
        # bb_mean_50, bb_std_50 = (
        #     np.mean(_window(close_arr, 50)),
        #     np.std(_window(close_arr, 50), ddof=1),
        # )

        # ▼▼ 修正後: 平均は長1以上、標準偏差は長2以上の制約を明示
        w_bb_30 = _window(close_arr, 30)
        bb_mean_30 = float(np.mean(w_bb_30)) if len(w_bb_30) >= 1 else np.nan
        bb_std_30 = float(np.std(w_bb_30, ddof=1)) if len(w_bb_30) >= 2 else np.nan

        w_bb_50 = _window(close_arr, 50)
        bb_mean_50 = float(np.mean(w_bb_50)) if len(w_bb_50) >= 1 else np.nan
        bb_std_50 = float(np.std(w_bb_50, ddof=1)) if len(w_bb_50) >= 2 else np.nan

        # Period 30, 2.5
        bb_lower_30_25 = bb_mean_30 - 2.5 * bb_std_30
        width_30_25 = 5.0 * bb_std_30
        features["e1c_bb_width_pct_30_2.5"] = (
            (width_30_25 / bb_mean_30) * 100 if bb_mean_30 != 0.0 else np.inf
        )
        features["e1c_bb_percent_30_2.5"] = (current_close - bb_lower_30_25) / (
            width_30_25 + 1e-10
        )

        # Period 50, 2.5
        bb_lower_50_25 = bb_mean_50 - 2.5 * bb_std_50
        features["e1c_bb_percent_50_2.5"] = (current_close - bb_lower_50_25) / (
            5.0 * bb_std_50 + 1e-10
        )

        # ---------------------------------------------------------
        # 4. RSI系 (厳選2個)
        # ---------------------------------------------------------
        rsi_14_arr = calculate_rsi_numba(close_arr, 14)
        rsi_21_arr = calculate_rsi_numba(close_arr, 21)

        features["e1c_rsi_14"] = _last(rsi_14_arr)
        features["e1c_rsi_momentum_21"] = (
            rsi_21_arr[-1] - rsi_21_arr[-2] if len(rsi_21_arr) >= 2 else np.nan
        )

        # ---------------------------------------------------------
        # 5. MACD, STC, Coppock系 (厳選3個)
        # ---------------------------------------------------------
        ema_19 = _ema(close_arr, 19)
        ema_39 = _ema(close_arr, 39)
        ema_5 = _ema(close_arr, 5)
        ema_35 = _ema(close_arr, 35)

        macd_19_39 = ema_19 - ema_39
        macd_5_35 = ema_5 - ema_35

        features["e1c_macd_19_39"] = _last(macd_19_39)
        features["e1c_macd_histogram_19_39_9"] = _last(macd_19_39) - _last(
            _ema(macd_19_39, 9)
        )
        features["e1c_macd_histogram_5_35_5"] = _last(macd_5_35) - _last(
            _ema(macd_5_35, 5)
        )

        # ---------------------------------------------------------
        # 6. オシレーター、トレンド、モメンタム系 (厳選20個)
        # ---------------------------------------------------------
        features["e1c_momentum_10"] = (
            current_close - close_arr[-11] if len(close_arr) > 10 else np.nan
        )

        p_roc = 20
        if len(close_arr) > p_roc:
            denom = close_arr[-1 - p_roc]
            features[f"e1c_rate_of_change_{p_roc}"] = (
                ((current_close - denom) / denom) * 100 if denom != 0.0 else np.inf
            )
        else:
            features[f"e1c_rate_of_change_{p_roc}"] = np.nan

        # Stochastic
        stoch_k_14_arr = calculate_stochastic_numba(
            high_arr, low_arr, close_arr, 14, 3, 3
        )
        stoch_k_21_arr = calculate_stochastic_numba(
            high_arr, low_arr, close_arr, 21, 5, 5
        )

        features["e1c_stoch_k_14"] = _last(stoch_k_14_arr)
        stoch_d_21_5_arr = _rolling_mean(stoch_k_21_arr, 5)
        stoch_slow_d_21_5_5_arr = _rolling_mean(stoch_d_21_5_arr, 5)
        features["e1c_stoch_slow_d_21_5_5"] = _last(stoch_slow_d_21_5_5_arr)

        # EMA Deviations
        for p in [10, 20, 50, 200]:
            ema_val = _last(_ema(close_arr, p))
            features[f"e1c_ema_deviation_{p}"] = (
                ((current_close - ema_val) / ema_val) * 100
                if ema_val != 0.0
                else np.inf
            )

        # SMA Deviations
        sma_200 = rolling_mean_numba(_window(close_arr, 200), 200)
        if np.isnan(sma_200):
            features["e1c_sma_deviation_200"] = np.nan
        else:
            features["e1c_sma_deviation_200"] = (
                ((current_close - sma_200) / sma_200) * 100
                if sma_200 != 0.0
                else np.inf
            )

        # RVI
        for p in [10, 14, 20]:
            rvi_arr = _rolling_mean(close_arr - data["open"], p) / (
                _rolling_mean(high_arr - low_arr, p) + 1e-10
            )
            features[f"e1c_relative_vigor_index_{p}"] = _last(rvi_arr)
            if p in [10, 20]:
                # ▼▼ 修正前
                # features[f"e1c_rvi_signal_{p}"] = np.mean(_window(rvi_arr, 4))
                # ▼▼ 修正後 (Polarsのrolling_mean(4)の仕様に合わせ、長4未満はNaN)
                w_rvi = _window(rvi_arr, 4)
                features[f"e1c_rvi_signal_{p}"] = (
                    float(np.mean(w_rvi)) if len(w_rvi) >= 4 else np.nan
                )

        # KST
        if len(close_arr) >= 31:
            d10 = close_arr[-11]
            d15 = close_arr[-16]
            d20 = close_arr[-21]
            d30 = close_arr[-31]

            roc_10 = (current_close - d10) / d10 if d10 != 0.0 else np.inf
            roc_15 = (current_close - d15) / d15 if d15 != 0.0 else np.inf
            roc_20 = (current_close - d20) / d20 if d20 != 0.0 else np.inf
            roc_30 = (current_close - d30) / d30 if d30 != 0.0 else np.inf

            kst_val = (roc_10 * 1 + roc_15 * 2 + roc_20 * 3 + roc_30 * 4) / 10 * 100
            features["e1c_kst"] = kst_val
        else:
            features["e1c_kst"] = np.nan

        # Trend Strength
        for p in [20, 50]:
            # ▼▼ 修正前
            # features[f"e1c_trend_strength_{p}"] = 1.0 / (
            #     np.std(_window(close_arr, p), ddof=1) + 1e-10
            # )
            # ▼▼ 修正後
            w_trend = _window(close_arr, p)
            features[f"e1c_trend_strength_{p}"] = (
                1.0 / (np.std(w_trend, ddof=1) + 1e-10) if len(w_trend) >= 2 else np.nan
            )

        # Others
        features["e1c_trix_14"] = _last(calculate_trix_numba(close_arr, 14))
        features["e1c_ultimate_oscillator"] = _last(
            calculate_ultimate_oscillator_numba(
                high_arr, low_arr, close_arr, volume_arr
            )
        )
        features["e1c_williams_r_14"] = _last(
            calculate_williams_r_numba(high_arr, low_arr, close_arr, 14)
        )

        return features
