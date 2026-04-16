# realtime_feature_engine_1C_technical.py
# Project Cimera V5 - Feature Module 1C (Technical Indicators)

import sys
import math
from pathlib import Path
from typing import Dict

import numpy as np
import numba as nb
from numba import njit

sys.path.append(str(Path(__file__).resolve().parent.parent))  # /workspace をパスに追加
import blueprint as config
sys.path.append(str(config.CORE_DIR))
from core_indicators import (
    calculate_atr_wilder,
    calculate_rsi_wilder,
)


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


@njit(fastmath=False, cache=True)
def _ema(arr: np.ndarray, span: int) -> np.ndarray:
    """指数移動平均(EMA) — njit化によりPythonループを排除"""
    n = len(arr)
    out = np.zeros(n, dtype=np.float64)
    if n == 0:
        return out
    alpha = 2.0 / (span + 1.0)
    out[0] = arr[0]
    for i in range(1, n):
        out[i] = alpha * arr[i] + (1.0 - alpha) * out[i - 1]
    return out


@njit(fastmath=False, cache=True)
def _rolling_mean(arr: np.ndarray, window: int) -> np.ndarray:
    """単純移動平均(SMA) — njit化によりPythonループを排除。
    NaN-aware: 窓内にNaNが1つでもあれば出力もNaN（Polars rolling_mean準拠）。
    先頭にNaNを含む配列（Numba出力など）でも正しく動作する。
    """
    n = len(arr)
    out = np.full(n, np.nan)
    if window <= 0 or n < window:
        return out
    for i in range(window - 1, n):
        has_nan = False
        s = 0.0
        for j in range(i - window + 1, i + 1):
            if np.isnan(arr[j]):
                has_nan = True
                break
            s += arr[j]
        if not has_nan:
            out[i] = s / window
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
# Numba JIT 関数群 その1 (Aroon)
# ==================================================================
# calculate_atr_wilder  → core_indicators（Wilder平滑化ATR）に統一
# calculate_rsi_wilder  → core_indicators（Wilder平滑化RSI）に統一
# calculate_adx         → core_indicators（Wilder平滑化ADX）に統一
# ==================================================================

# calculate_adx (core_indicators) は ADX 配列のみを返す。
# DI+21 を同一の Wilder 平滑化で取得するため、calculate_adx 内部と
# 完全同一のアルゴリズムで DI+ の最終スカラー値を返す関数を定義する。
# ※ 将来 core_indicators に calculate_adx_full が追加された際はこの関数を削除すること。
@njit(fastmath=False, cache=True)
def _calculate_di_plus_scalar(
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    period: int,
) -> float:
    """
    DI+ の最終スカラー値を返す — calculate_adx (core_indicators) 内部と完全同一の
    Wilder 平滑化実装（初期値: 先頭 period 本の単純合計シード）。
    """
    n = len(high)
    if n <= period:
        return np.nan

    # TR, DM+ — calculate_adx 内部と同一
    tr = np.zeros(n, dtype=np.float64)
    dm_plus = np.zeros(n, dtype=np.float64)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        hl = high[i] - low[i]
        hc = abs(high[i] - close[i - 1])
        lc = abs(low[i] - close[i - 1])
        tr[i] = max(hl, hc, lc)
        up_move = high[i] - high[i - 1]
        down_move = low[i - 1] - low[i]
        dm_plus[i] = up_move if (up_move > down_move and up_move > 0.0) else 0.0

    # Wilder 平滑化（単純合計シード） — calculate_adx 内部と同一
    atr_w = 0.0
    dmp_w = 0.0
    for j in range(period):
        atr_w += tr[j]
        dmp_w += dm_plus[j]

    for i in range(period, n):
        atr_w = atr_w - atr_w / period + tr[i]
        dmp_w = dmp_w - dmp_w / period + dm_plus[i]

    if atr_w > 1e-10:
        return dmp_w / atr_w * 100.0
    return np.nan


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

        # 安全な参照用変数 — high/low/volume は get() でキーなし時も np.array([]) にフォールバック
        _empty = np.array([], dtype=np.float64)
        close_arr  = data.get("close",  _empty)
        high_arr   = data.get("high",   _empty)
        low_arr    = data.get("low",    _empty)
        volume_arr = data.get("volume", _empty)
        open_arr   = data.get("open",   _empty)
        current_close = close_arr[-1] if len(close_arr) > 0 else np.nan

        # ---------------------------------------------------------
        # 1. ATR系 (厳選5個)
        # ---------------------------------------------------------
        # calculate_atr_wilder: core_indicators の Wilder 版（SMA 方式から移行）
        atr_13_arr = calculate_atr_wilder(high_arr, low_arr, close_arr, 13)
        atr_21_arr = calculate_atr_wilder(high_arr, low_arr, close_arr, 21)
        atr_55_arr = calculate_atr_wilder(high_arr, low_arr, close_arr, 55)

        atr_13 = _last(atr_13_arr)

        # ② ATR が nan のケース: np.isfinite で両辺を確認してから除算
        features["e1c_atr_pct_13"] = (
            (atr_13 / current_close) * 100
            if (np.isfinite(atr_13) and np.isfinite(current_close) and current_close != 0.0)
            else np.nan
        )

        # atr_trend_13: 差分を ATR13 で割りスケール不変化（学習側と統一）
        # scale_by_atr: core_indicators の ATR割り統一関数
        if len(atr_13_arr) >= 2 and np.isfinite(atr_13) and atr_13 > 1e-10:
            features["e1c_atr_trend_13"] = (atr_13_arr[-1] - atr_13_arr[-2]) / atr_13
        else:
            features["e1c_atr_trend_13"] = np.nan

        # atr_volatility: std(ATR_window) を各 ATR で割りスケール不変化（学習側と統一）
        # 【設計上の制約】学習側との差異を記録する:
        #   学習側: rolling_std(atr_{period}_arr, window=period)[i] / atr_13_arr[i]
        #           → 分母は各バー時点の atr_13（時系列配列）
        #   リアルタイム側: std(_window(atr_{period}_arr, period)) / _last(atr_13_arr)
        #           → 分母は最新バーの atr_13 スカラー（時系列を保持できないため）
        #   リアルタイム推論では1本分のスカラーしか持てないため、この差異は構造的に許容する。
        #   ただし分子（std の対象配列）は学習側と合わせ atr_{period}_arr を使う。
        w_atr_13 = _window(atr_13_arr, 13)
        if len(w_atr_13) >= 2 and np.isfinite(atr_13) and atr_13 > 1e-10:
            features["e1c_atr_volatility_13"] = float(np.std(w_atr_13, ddof=1)) / atr_13
        else:
            features["e1c_atr_volatility_13"] = np.nan

        atr_21 = _last(atr_21_arr)
        w_atr_21 = _window(atr_21_arr, 21)  # 学習側と合わせ atr_21_arr の window を使う
        if len(w_atr_21) >= 2 and np.isfinite(atr_21) and atr_21 > 1e-10:
            features["e1c_atr_volatility_21"] = float(np.std(w_atr_21, ddof=1)) / atr_21
        else:
            features["e1c_atr_volatility_21"] = np.nan

        atr_55 = _last(atr_55_arr)
        w_atr_55 = _window(atr_55_arr, 55)  # 学習側と合わせ atr_55_arr の window を使う
        if len(w_atr_55) >= 2 and np.isfinite(atr_55) and atr_55 > 1e-10:
            features["e1c_atr_volatility_55"] = float(np.std(w_atr_55, ddof=1)) / atr_55
        else:
            features["e1c_atr_volatility_55"] = np.nan

        # ---------------------------------------------------------
        # 2. ADX, DI, Aroon系 (厳選3個)
        # ---------------------------------------------------------
        aroon_up_14 = _last(calculate_aroon_up_numba(high_arr, 14))
        aroon_down_14 = _last(calculate_aroon_down_numba(low_arr, 14))
        features["e1c_aroon_down_14"] = aroon_down_14
        features["e1c_aroon_oscillator_14"] = aroon_up_14 - aroon_down_14

        # calculate_adx: core_indicators の Wilder 版（SMA 方式 calculate_di_plus_numba から移行）
        # DI+21 は _calculate_di_plus_scalar で算出 — calculate_adx 内部と同一 Wilder 平滑化
        # （初期値: 先頭 period 本の単純合計シード → 以降 Wilder 漸化式）
        # ※ e1c_adx_21 は出力対象外のため adx 配列の計算は行わない
        features["e1c_di_plus_21"] = _calculate_di_plus_scalar(
            high_arr, low_arr, close_arr, 21
        )

        # ---------------------------------------------------------
        # 3. ボリンジャーバンド系 (厳選3個)
        # ---------------------------------------------------------
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
        # calculate_rsi_wilder: core_indicators の Wilder 版（SMA 方式から移行）
        rsi_14_arr = calculate_rsi_wilder(close_arr, 14)
        rsi_21_arr = calculate_rsi_wilder(close_arr, 21)

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

        # scale_by_atr: core_indicators の ATR割り統一関数（学習側 engine_1_C と統一）
        atr13_last = atr_13  # calculate_atr_wilder(period=13) の最終値
        features["e1c_macd_19_39"] = (
            _last(macd_19_39) / (atr13_last + 1e-10)
            if np.isfinite(atr13_last) else np.nan
        )
        macd_19_39_signal = _ema(macd_19_39, 9)
        features["e1c_macd_histogram_19_39_9"] = (
            (_last(macd_19_39) - _last(macd_19_39_signal)) / (atr13_last + 1e-10)
            if np.isfinite(atr13_last) else np.nan
        )
        macd_5_35_signal = _ema(macd_5_35, 5)
        features["e1c_macd_histogram_5_35_5"] = (
            (_last(macd_5_35) - _last(macd_5_35_signal)) / (atr13_last + 1e-10)
            if np.isfinite(atr13_last) else np.nan
        )

        # ---------------------------------------------------------
        # 6. オシレーター、トレンド、モメンタム系 (厳選20個)
        # ---------------------------------------------------------
        # momentum_10: scale_by_atr で ATR割り（学習側 engine_1_C と統一）
        features["e1c_momentum_10"] = (
            (current_close - close_arr[-11]) / (atr13_last + 1e-10)
            if (len(close_arr) > 10 and np.isfinite(atr13_last))
            else np.nan
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
            rvi_arr = _rolling_mean(close_arr - open_arr, p) / (
                _rolling_mean(high_arr - low_arr, p) + 1e-10
            )
            features[f"e1c_relative_vigor_index_{p}"] = _last(rvi_arr)
            if p in [10, 20]:
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

        # Trend Strength — normalized_std / ATR13 で無次元化（学習側 engine_1_C と統一）
        for p in [20, 50]:
            w_trend = _window(close_arr, p)
            if len(w_trend) >= 2 and np.isfinite(atr13_last) and atr13_last > 1e-10:
                normalized_std = np.std(w_trend, ddof=1) / atr13_last
                features[f"e1c_trend_strength_{p}"] = min(
                    1.0 / (normalized_std + 1e-10), 100.0
                )
            else:
                features[f"e1c_trend_strength_{p}"] = np.nan

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
