"""
stable_rolling.py — context-length 非依存の window 内 two-pass 実装。

学習側 (engine_1_A〜1_F) と本番側 (rfe_1A〜1_F) の両方から **同一の Numba JIT
関数** を呼ぶことで、Polars rolling_skew / rolling_var / rolling_mean の
内部 running 状態に起因する context 依存性を排除する。

設計方針:
  - 各 window で fresh に再計算 (running 更新を使わない)
  - 累積誤差ゼロ → context length 非依存 (= 学習 3.4M 行と本番 2,980 行で同じ値)
  - Numba JIT cache=True で初回コンパイル後は Polars rolling と同等の速度
  - 計算順序を fastmath=False で固定 (浮動小数演算順序の再現性)

提供関数:
  基本: stable_rolling_mean, stable_rolling_var, stable_rolling_std, stable_rolling_skew
  合成: stable_kurtosis_engine_formula, stable_moment_k_engine_formula
       (engine_1_A の独自 kurtosis / moment_k 式に対応する 1-shot 計算)
"""
from __future__ import annotations
import numpy as np
import numba


# ════════════════════════════════════════════════════════════════════
# 基本 stable rolling 関数
# ════════════════════════════════════════════════════════════════════

@numba.njit(cache=True, fastmath=False)
def stable_rolling_mean(arr: np.ndarray, window: int) -> np.ndarray:
    n = arr.shape[0]
    out = np.full(n, np.nan)
    if n < window or window < 1:
        return out
    for i in range(window - 1, n):
        s = 0.0
        for j in range(i - window + 1, i + 1):
            s += arr[j]
        out[i] = s / window
    return out


@numba.njit(cache=True, fastmath=False)
def stable_rolling_var(arr: np.ndarray, window: int, ddof: int = 1) -> np.ndarray:
    n = arr.shape[0]
    out = np.full(n, np.nan)
    if n < window or window < 1:
        return out
    denom = float(window - ddof)
    if denom <= 0.0:
        return out
    for i in range(window - 1, n):
        s = 0.0
        for j in range(i - window + 1, i + 1):
            s += arr[j]
        m = s / window
        v = 0.0
        for j in range(i - window + 1, i + 1):
            d = arr[j] - m
            v += d * d
        out[i] = v / denom
    return out


@numba.njit(cache=True, fastmath=False)
def stable_rolling_std(arr: np.ndarray, window: int, ddof: int = 1) -> np.ndarray:
    var = stable_rolling_var(arr, window, ddof)
    out = np.full(var.shape, np.nan)
    for i in range(var.shape[0]):
        if not np.isnan(var[i]) and var[i] >= 0.0:
            out[i] = np.sqrt(var[i])
    return out


@numba.njit(cache=True, fastmath=False)
def stable_rolling_skew(arr: np.ndarray, window: int) -> np.ndarray:
    """Polars rolling_skew(bias=True) 等価: m3 / m2^(3/2)"""
    n = arr.shape[0]
    out = np.full(n, np.nan)
    if n < window or window < 1:
        return out
    for i in range(window - 1, n):
        s = 0.0
        for j in range(i - window + 1, i + 1):
            s += arr[j]
        m = s / window
        m2 = 0.0
        m3 = 0.0
        for j in range(i - window + 1, i + 1):
            d = arr[j] - m
            d2 = d * d
            m2 += d2
            m3 += d2 * d
        m2 /= window
        m3 /= window
        if m2 <= 0.0:
            out[i] = 0.0
        else:
            out[i] = m3 / (m2 ** 1.5)
    return out


# ════════════════════════════════════════════════════════════════════
# engine_1_A 専用合成式 (kurtosis / moment_k)
# ════════════════════════════════════════════════════════════════════

@numba.njit(cache=True, fastmath=False)
def stable_kurtosis_engine_formula(close: np.ndarray, window: int) -> np.ndarray:
    """engine_1_A.statistical_kurtosis_{window}:
        (close - rolling_mean(W)).pow(4).rolling_mean(W) / (var_ddof0^2 + 1e-10) - 3
    where var_ddof0 = rolling_var(W, ddof=1) * (W-1)/W = sum((x-m)^2)/W
    """
    n = close.shape[0]
    out = np.full(n, np.nan)
    if n < 2 * window - 1 or window < 2:
        return out

    centered_pow4 = np.full(n, np.nan)
    for i in range(window - 1, n):
        s = 0.0
        for j in range(i - window + 1, i + 1):
            s += close[j]
        m = s / window
        d = close[i] - m
        d2 = d * d
        centered_pow4[i] = d2 * d2

    for i in range(2 * window - 2, n):
        m4_sum = 0.0
        for j in range(i - window + 1, i + 1):
            m4_sum += centered_pow4[j]
        m4 = m4_sum / window

        s = 0.0
        for j in range(i - window + 1, i + 1):
            s += close[j]
        m = s / window
        v = 0.0
        for j in range(i - window + 1, i + 1):
            d = close[j] - m
            v += d * d
        var_ddof0 = v / window
        std_ddof0_pow4 = var_ddof0 * var_ddof0
        out[i] = m4 / (std_ddof0_pow4 + 1e-10) - 3.0

    return out


@numba.njit(cache=True, fastmath=False)
def stable_moment_k_engine_formula(
    close: np.ndarray, window: int, moment: int
) -> np.ndarray:
    """engine_1_A.statistical_moment_{moment}_{window}:
        z[i] = (close[i] - rolling_mean(W)[i]) / sqrt(var_ddof0(W)[i] + 1e-10)
        moment_k[i] = rolling_mean(W) of z^moment
    """
    n = close.shape[0]
    out = np.full(n, np.nan)
    if n < 2 * window - 1 or window < 2:
        return out

    z = np.full(n, np.nan)
    for i in range(window - 1, n):
        s = 0.0
        for j in range(i - window + 1, i + 1):
            s += close[j]
        m = s / window
        v = 0.0
        for j in range(i - window + 1, i + 1):
            d = close[j] - m
            v += d * d
        var_ddof0 = v / window
        std_ddof0 = np.sqrt(var_ddof0 + 1e-10)
        z[i] = (close[i] - m) / std_ddof0

    for i in range(2 * window - 2, n):
        m_k_sum = 0.0
        for j in range(i - window + 1, i + 1):
            m_k_sum += z[j] ** moment
        out[i] = m_k_sum / window

    return out


# ════════════════════════════════════════════════════════════════════
# Deterministic EMA (Exponentially Weighted Mean)
# ════════════════════════════════════════════════════════════════════

@numba.njit(cache=True, fastmath=False)
def stable_ewm_mean(arr: np.ndarray, alpha: float) -> np.ndarray:
    """Polars ewm_mean(adjust=False, ignore_nulls=True) と bit-identical な
    deterministic EMA。

    α は呼出し側で span / half_life から計算:
      span notation:      α = 2.0 / (span + 1.0)
      half_life notation: α = 1.0 - exp(-ln(2.0) / half_life)

    recurrence (adjust=False, ignore_nulls=True):
      y[0] = x[0]
      y[i] = α * x[i] + (1 - α) * y[i-1]

    特性:
      - context 長非依存 (warmup 経過後は同じ結果 = bit-identical 保証)
      - Polars `ewm_mean(adjust=False)` の内部 SIMD/FMA 最適化に起因する
        context 長依存性を排除する (Phase E + EMA 拡張)
      - close 列のように NaN を含まない入力で正しく動作。NaN を含む場合は
        伝播するが、Polars `ignore_nulls=True` (default) の挙動とは異なる
        可能性があるので注意。
    """
    n = arr.shape[0]
    out = np.full(n, np.nan)
    if n == 0:
        return out
    out[0] = arr[0]
    one_minus_alpha = 1.0 - alpha
    for i in range(1, n):
        out[i] = alpha * arr[i] + one_minus_alpha * out[i - 1]
    return out
