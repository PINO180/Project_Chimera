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


# --- プロジェクトのルートディレクトリをPythonの検索パスに追加 ---
project_root = Path(__file__).resolve().parents[1]
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

import blueprint as config

# ==================================================================
#
# 1. NUMBA UDF ライブラリ (Part 1: Engine 1A - Statistical & Robust)
#
# ==================================================================


@njit(fastmath=True, cache=True)
def rolling_skew_numba(arr: np.ndarray) -> float:
    """
    ローリング歪度 (Bias-Corrected)
    Polars/Pandasの挙動と一致させるため、不偏推定量を使用。
    Formula: (n * sum((x - mean)^3)) / ((n-1)(n-2) * std^3)
    """
    # NaN除去
    finite_vals = arr[np.isfinite(arr)]
    n = len(finite_vals)

    if n < 3:
        return np.nan

    mean_val = np.mean(finite_vals)

    # 偏差の計算
    diffs = finite_vals - mean_val

    # 分散と標準偏差 (n-1)
    var_val = np.sum(diffs**2) / (n - 1)
    std_val = np.sqrt(var_val)

    if std_val < 1e-10:
        return 0.0

    # 3次モーメント和
    m3_sum = np.sum(diffs**3)

    # 不偏係数による補正
    # Fisher-Pearson coefficient of skewness
    factor = n / ((n - 1) * (n - 2))
    skew = factor * (m3_sum / (std_val**3))

    return skew


@njit(fastmath=True, cache=True)
def rolling_kurtosis_numba(arr: np.ndarray) -> float:
    """
    ローリング尖度 (Bias-Corrected Excess Kurtosis)
    Polars/Pandasの挙動と一致させる。
    """
    finite_vals = arr[np.isfinite(arr)]
    n = len(finite_vals)

    if n < 4:
        return np.nan

    mean_val = np.mean(finite_vals)
    diffs = finite_vals - mean_val

    # 分散と標準偏差 (n-1)
    var_val = np.sum(diffs**2) / (n - 1)
    std_val = np.sqrt(var_val)

    if std_val < 1e-10:
        return 0.0

    # 4次モーメント和
    m4_sum = np.sum(diffs**4)

    # 不偏尖度の計算式
    # g2 = [n(n+1) / (n-1)(n-2)(n-3)] * sum((x-mean)/s)^4 - [3(n-1)^2 / (n-2)(n-3)]

    term1_num = n * (n + 1)
    term1_den = (n - 1) * (n - 2) * (n - 3)
    term1 = term1_num / term1_den

    m4_standardized_sum = m4_sum / (std_val**4)

    term2_num = 3 * (n - 1) ** 2
    term2_den = (n - 2) * (n - 3)
    term2 = term2_num / term2_den

    kurtosis = term1 * m4_standardized_sum - term2

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

    finite_vals = arr[np.isfinite(arr)]
    if len(finite_vals) < 2:
        return np.nan

    mean_val = np.mean(finite_vals)
    std_val = np.std(finite_vals)

    if std_val < 1e-10:
        return 0.0

    z = (finite_vals - mean_val) / std_val
    return np.mean(z**moment)


@njit(fastmath=True, cache=True)
def mad_rolling_numba(arr: np.ndarray) -> float:
    """
    ローリングMAD (Median Absolute Deviation)
    Source: engine_1_A (robust_mad)
    """
    finite_data = arr[np.isfinite(arr)]

    if len(finite_data) < 3:
        return np.nan

    # 中央値計算
    median_val = np.median(finite_data)
    # 絶対偏差の中央値
    abs_deviations = np.abs(finite_data - median_val)
    return np.median(abs_deviations)


@njit(fastmath=True, cache=True)
def biweight_location_numba(arr: np.ndarray) -> float:
    """
    Tukey's Biweight Location (厳密実装)
    Source: engine_1_A (biweight_location_numba)
    """
    finite_data = arr[np.isfinite(arr)]

    if len(finite_data) < 5:
        return np.median(finite_data) if len(finite_data) > 0 else np.nan

    # 初期値として中央値を使用
    current_location = np.median(finite_data)
    tolerance = 1e-10
    max_iterations = 50

    for _ in range(max_iterations):
        # MAD計算
        abs_residuals = np.abs(finite_data - current_location)
        mad_val = np.median(abs_residuals)

        if mad_val < 1e-15:
            break

        # スケールファクター（6 * MAD）
        scale = 6.0 * mad_val

        # 標準化残差
        u_values = (finite_data - current_location) / scale

        numerator = 0.0
        denominator = 0.0

        for j in range(len(finite_data)):
            u = u_values[j]
            if abs(u) < 1.0:
                # Biweight重み: (1 - u²)²
                weight = (1.0 - u**2) ** 2
                numerator += finite_data[j] * weight
                denominator += weight

        if denominator > 1e-15:
            new_location = numerator / denominator
        else:
            new_location = np.median(finite_data)
            break

        # 収束判定
        if abs(new_location - current_location) < tolerance:
            current_location = new_location
            break

        current_location = new_location

    return current_location


@njit(fastmath=True, cache=True)
def winsorized_mean_numba(arr: np.ndarray) -> float:
    """
    Winsorized Mean (上下5%クリップ)
    Source: engine_1_A (winsorized_mean_numba)
    """
    finite_data = arr[np.isfinite(arr)]

    if len(finite_data) < 5:
        return np.mean(finite_data) if len(finite_data) > 0 else np.nan

    # 上下5%点の計算
    # np.percentile は Numba 対応済み
    p05 = np.percentile(finite_data, 5)
    p95 = np.percentile(finite_data, 95)

    # ウィンソライズ（クリッピング）してから平均
    winsorized_sum = 0.0
    for x in finite_data:
        if x < p05:
            winsorized_sum += p05
        elif x > p95:
            winsorized_sum += p95
        else:
            winsorized_sum += x

    return winsorized_sum / len(finite_data)


@njit(fastmath=True, cache=True)
def robust_stabilization_numba(arr: np.ndarray) -> np.ndarray:
    """
    ロバスト安定化処理 (Numba JIT) - 配列返し
    Source: engine_1_A (robust_stabilization_numba) のロジックを配列末尾適用に適合
    Returns: shape (1,) array containing the stabilized last value
    """
    # 全データを安定化計算に使用
    finite_vals = arr[np.isfinite(arr)]
    n = len(arr)
    result = np.zeros(1, dtype=np.float64)

    if len(finite_vals) < 3:
        result[0] = 0.0 if n == 0 else (arr[-1] if np.isfinite(arr[-1]) else 0.0)
        return result

    # 中央値
    median_val = np.median(finite_vals)

    # MAD
    abs_devs = np.abs(finite_vals - median_val)
    mad_val = np.median(abs_devs)

    if mad_val < 1e-10:
        mad_val = np.std(finite_vals) * 0.6745

    # 閾値: 中央値 ± 3 * MAD
    lower_bound = median_val - 3 * mad_val
    upper_bound = median_val + 3 * mad_val

    # 最新の値を取得
    last_val = arr[-1]

    if np.isnan(last_val):
        result[0] = median_val
    elif np.isinf(last_val):
        result[0] = upper_bound if last_val > 0 else lower_bound
    else:
        # クリップ
        if last_val < lower_bound:
            result[0] = lower_bound
        elif last_val > upper_bound:
            result[0] = upper_bound
        else:
            result[0] = last_val

    return result


@njit(fastmath=True, cache=True)
def jarque_bera_statistic_numba(arr: np.ndarray) -> float:
    """
    Jarque-Bera検定統計量
    Source: engine_1_A
    """
    finite_data = arr[np.isfinite(arr)]
    n = len(finite_data)

    if n < 20:
        return np.nan

    mean_val = np.mean(finite_data)

    # 分散
    variance = np.sum((finite_data - mean_val) ** 2) / (n - 1)
    std_val = np.sqrt(variance)

    if std_val < 1e-10:
        return 0.0

    # 歪度・尖度 (Population formula used in JB definition, typically)
    # Source uses standardization loop
    z = (finite_data - mean_val) / std_val
    skewness = np.mean(z**3)
    kurtosis = np.mean(z**4) - 3  # Excess kurtosis

    jb_stat = n * (skewness**2 / 6 + kurtosis**2 / 24)
    return jb_stat


@njit(fastmath=True, cache=True)
def anderson_darling_numba(arr: np.ndarray) -> float:
    """
    Anderson-Darling統計量 (厳密実装)
    Source: engine_1_A
    """
    finite_data = arr[np.isfinite(arr)]
    n = len(finite_data)

    if n < 10:
        return np.nan

    sorted_data = np.sort(finite_data)
    mean_val = np.mean(sorted_data)

    variance = np.sum((sorted_data - mean_val) ** 2) / (n - 1)
    std_val = np.sqrt(variance)

    if std_val < 1e-10:
        return 0.0

    SQRT2 = 1.4142135623730951
    ad_sum = 0.0

    for j in range(n):
        # 標準化
        z_j = (sorted_data[j] - mean_val) / std_val
        z_nj = (sorted_data[n - 1 - j] - mean_val) / std_val

        # CDF (erf使用)
        F_j = 0.5 * (1.0 + math.erf(z_j / SQRT2))
        F_nj = 0.5 * (1.0 + math.erf(z_nj / SQRT2))

        # 対数計算のガード
        if F_j < 1e-15:
            F_j = 1e-15
        val_nj = 1.0 - F_nj
        if val_nj < 1e-15:
            val_nj = 1e-15

        log_term = np.log(F_j) + np.log(val_nj)
        ad_sum += (2 * j + 1) * log_term

    return -n - ad_sum / n


@njit(fastmath=True, cache=True)
def runs_test_numba(arr: np.ndarray) -> float:
    """
    Runs Test統計量
    Source: engine_1_A
    """
    finite_data = arr[np.isfinite(arr)]
    if len(finite_data) < 10:
        return np.nan

    median_val = np.median(finite_data)
    binary_series = (finite_data > median_val).astype(np.int32)

    runs = 1
    for j in range(1, len(binary_series)):
        if binary_series[j] != binary_series[j - 1]:
            runs += 1

    n1 = np.sum(binary_series)
    n2 = len(binary_series) - n1

    if n1 > 0 and n2 > 0:
        expected_runs = (2 * n1 * n2) / (n1 + n2) + 1

        denom = (n1 + n2) ** 2 * (n1 + n2 - 1)
        if denom == 0:
            return 0.0

        var_runs = (2 * n1 * n2 * (2 * n1 * n2 - n1 - n2)) / denom

        if var_runs > 0:
            return (runs - expected_runs) / np.sqrt(var_runs)

    return 0.0


@njit(fastmath=True, cache=True)
def von_neumann_ratio_numba(arr: np.ndarray) -> float:
    """
    Von Neumann Ratio (厳密実装)
    Source: engine_1_A
    """
    finite_data = arr[np.isfinite(arr)]
    n = len(finite_data)

    if n < 3:
        return np.nan

    diff_sq_sum = np.sum(np.diff(finite_data) ** 2)

    mean_val = np.mean(finite_data)
    sum_sq_deviations = np.sum((finite_data - mean_val) ** 2)

    if sum_sq_deviations > 1e-15:
        vn_ratio = diff_sq_sum / sum_sq_deviations
        return min(4.0, max(0.0, vn_ratio))

    return 0.0


@nb.njit(fastmath=True, cache=True)
def fast_rolling_mean_numba(arr: np.ndarray, window: int) -> float:
    """
    Numba最適化ローリング平均（カスタムウィンドウ用）
    Engine 1A: fast_rolling_mean_numba のロジックを再現しつつWindow可変対応
    """
    n = len(arr)

    # Engine 1Aのロジックでは、インデックス i < 20 の場合は NaN を返している
    # リアルタイム版ではバッファ長が不足している場合に対応
    if n < 20 and window >= 20:
        # ソースコードのハードコードされた制約(min window 20)を尊重
        # ただしウィンドウ指定が小さい場合はその限りではない
        return np.nan

    start_idx = max(0, n - window)
    window_data = arr[start_idx:]

    window_sum = 0.0
    count = 0
    for j in range(len(window_data)):
        if not np.isnan(window_data[j]):
            window_sum += window_data[j]
            count += 1

    if count > 0:
        return window_sum / count
    else:
        return np.nan


@nb.njit(fastmath=True, cache=True)
def fast_rolling_std_numba(arr: np.ndarray, window: int) -> float:
    """
    Numba最適化ローリング標準偏差
    Engine 1A: fast_rolling_std_numba のロジックを再現
    """
    n = len(arr)

    # Engine 1Aの制約
    if n < 20 and window >= 20:
        return np.nan

    start_idx = max(0, n - window)
    window_data = arr[start_idx:]

    # Numbaはnp.nanを無視するmean/stdをサポートしていないため手動実装
    # まず有限値だけを集める（高速化のためループ内で処理も可だが、可読性重視で抽出）
    # ただしアロケーションを避けるなら2パス計算が良い

    # パス1: 平均計算
    window_sum = 0.0
    count = 0
    for j in range(len(window_data)):
        if not np.isnan(window_data[j]):
            window_sum += window_data[j]
            count += 1

    if count <= 1:
        return np.nan

    mean_val = window_sum / count

    # パス2: 分散計算
    var_sum = 0.0
    for j in range(len(window_data)):
        if not np.isnan(window_data[j]):
            diff = window_data[j] - mean_val
            var_sum += diff * diff

    variance = var_sum / (count - 1)  # 不偏標準偏差
    return np.sqrt(variance)


@nb.njit(fastmath=True, cache=True)
def basic_stabilization_numba(arr: np.ndarray) -> np.ndarray:
    """
    基本安定化処理 (Numba JIT) - 配列返し
    Engine 1A: バッファ全体の分布に基づいて、最新の1点を安定化（クリップ）した値を
    1要素の配列として返す（_last()での呼び出しに対応するため）。
    """
    n = len(arr)
    # 結果格納用の配列（_lastは末尾を取得するため、長さ1の配列で良い）
    result = np.zeros(1, dtype=np.float64)

    if n == 0:
        return result

    # 1. NaN/Inf を 0.0 に置換した配列を作成（コピー）
    cleaned = np.zeros(n)
    for i in range(n):
        if np.isnan(arr[i]) or np.isinf(arr[i]):
            cleaned[i] = 0.0
        else:
            cleaned[i] = arr[i]

    # 2. 有限値カウント
    finite_count = 0
    for i in range(n):
        if np.isfinite(cleaned[i]):
            finite_count += 1

    # 対象となる最新の値
    last_val = cleaned[n - 1]

    # 3. クリップ処理判定
    if finite_count > 10:
        # 全体のMin/Max
        min_val = np.nanmin(cleaned)
        max_val = np.nanmax(cleaned)
        range_val = max_val - min_val

        if range_val > 1e-10:
            # 有限値のみ抽出してパーセンタイル計算
            valid_data = cleaned[np.isfinite(cleaned)]

            # Numba環境でのnp.percentile
            clip_margin_low = np.percentile(valid_data, 1)
            clip_margin_high = np.percentile(valid_data, 99)

            # 最新の値に対してクリップ適用
            if last_val < clip_margin_low:
                result[0] = clip_margin_low
            elif last_val > clip_margin_high:
                result[0] = clip_margin_high
            else:
                result[0] = last_val
        else:
            result[0] = last_val
    else:
        result[0] = last_val

    return result


# ----------------------------------------
# from engine_1_B_a_vast_universe_of_features.py
# ----------------------------------------


@njit(fastmath=True, cache=True)
def adf_統計量_udf(prices: np.ndarray) -> float:
    """
    拡張ディッキー・フラー検定統計量計算 (Numba JIT)
    """
    if len(prices) < 10:
        return np.nan

    # NaN値除去
    finite_prices = prices[np.isfinite(prices)]
    if len(finite_prices) < 10:
        return np.nan

    # 1次差分計算
    diff_prices = np.diff(finite_prices)
    lagged_prices = finite_prices[:-1]

    if len(diff_prices) < 5:
        return np.nan

    # ADF回帰の簡単なOLS: Δy_t = α + βy_{t-1} + ε_t
    n = len(diff_prices)

    # 設計行列: [定数項, ラグレベル]
    # np.column_stack は Numba でサポートされている
    X = np.column_stack((np.ones(n), lagged_prices))
    y = diff_prices

    # OLS計算: β = (X'X)^(-1)X'y
    try:
        XtX = X.T @ X
        XtX_inv = np.linalg.inv(XtX)
        XtY = X.T @ y
        beta = XtX_inv @ XtY

        # 残差と標準誤差計算
        residuals = y - X @ beta
        sse = np.sum(residuals**2)
        mse = sse / (n - 2)

        # β係数（ラグレベル）の標準誤差
        if mse > 0 and XtX_inv[1, 1] > 0:
            se_beta = np.sqrt(mse * XtX_inv[1, 1])
        else:
            return np.nan

        # ADF t統計量
        if se_beta > 1e-15:
            adf_stat = beta[1] / se_beta
        else:
            adf_stat = np.nan

    except:
        adf_stat = np.nan

    return adf_stat


@njit(fastmath=True, cache=True)
def phillips_perron_統計量_udf(prices: np.ndarray) -> float:
    """
    フィリップス・ペロン検定統計量計算 (Numba JIT)
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
        if sigma2 > 0 and XtX_inv[1, 1] > 0:
            se_beta = np.sqrt(sigma2 * XtX_inv[1, 1])
        else:
            return np.nan

        if se_beta > 1e-15:
            pp_stat = beta[1] / se_beta
        else:
            pp_stat = np.nan

    except:
        pp_stat = np.nan

    return pp_stat


@njit(fastmath=True, cache=True)
def kpss_統計量_udf(prices: np.ndarray) -> float:
    """
    KPSS検定統計量計算 (Numba JIT)
    """
    if len(prices) < 10:
        return np.nan

    finite_prices = prices[np.isfinite(prices)]
    if len(finite_prices) < 10:
        return np.nan

    n = len(finite_prices)

    # 線形トレンド適合によるデトレンド
    t = np.arange(n, dtype=np.float64)

    # デトレンドのための簡単線形回帰
    try:
        # OLS: y = α + βt + ε
        sum_t = np.sum(t)
        sum_t2 = np.sum(t**2)
        sum_y = np.sum(finite_prices)
        sum_ty = np.sum(t * finite_prices)

        denom = n * sum_t2 - sum_t**2
        if abs(denom) < 1e-15:
            return np.nan

        # 傾きと切片計算
        beta = (n * sum_ty - sum_t * sum_y) / denom
        alpha = (sum_y - beta * sum_t) / n

        # デトレンドされた系列
        detrended = finite_prices - (alpha + beta * t)

        # 部分和計算
        cumsum = np.cumsum(detrended)

        # KPSS統計量
        sse = np.sum(detrended**2) / n  # 誤差分散

        if sse > 1e-15:
            kpss_stat = np.sum(cumsum**2) / (n**2 * sse)
        else:
            kpss_stat = np.nan

    except:
        kpss_stat = np.nan

    return kpss_stat


@njit(fastmath=True, cache=True)
def t分布_自由度_udf(returns: np.ndarray) -> float:
    """
    t分布の自由度パラメータ推定 (Numba JIT)
    """
    if len(returns) < 10:
        return np.nan

    finite_returns = returns[np.isfinite(returns)]
    if len(finite_returns) < 10:
        return np.nan

    # リターン標準化
    mean_ret = np.mean(finite_returns)
    std_ret = np.std(finite_returns)

    if std_ret <= 1e-15:
        return np.nan

    standardized = (finite_returns - mean_ret) / std_ret

    # サンプル尖度計算
    fourth_moment = np.mean(standardized**4)

    # t分布用: E[X^4] = 3*ν/(ν-4) for ν > 4
    # ν解: ν = 4*(3 + kurtosis)/(kurtosis - 3)
    excess_kurtosis = fourth_moment - 3.0

    if excess_kurtosis > 0:
        dof = 4.0 * (3.0 + fourth_moment) / excess_kurtosis
        # 合理的範囲への制約
        dof = max(2.1, min(dof, 100.0))
    else:
        dof = 100.0  # 非常に高いDOF（概正規近似）

    return dof


@njit(fastmath=True, cache=True)
def t分布_尺度_udf(returns: np.ndarray) -> float:
    """
    t分布の尺度パラメータ推定 (Numba JIT)
    注意: t分布_自由度_udf に依存
    """
    if len(returns) < 5:
        return np.nan

    finite_returns = returns[np.isfinite(returns)]
    if len(finite_returns) < 5:
        return np.nan

    # 最初に自由度推定
    dof = t分布_自由度_udf(returns)

    if np.isnan(dof) or dof <= 2:
        # サンプル標準偏差へのフォールバック
        return np.std(finite_returns)

    # 既知νを持つt分布用、尺度パラメータσ推定可能:
    # σ² = sample_variance * (ν-2)/ν
    sample_var = np.var(finite_returns)
    scale_squared = sample_var * (dof - 2.0) / dof

    return np.sqrt(max(scale_squared, 1e-8))


@njit(fastmath=True, cache=True)
def gev_形状_udf(extremes: np.ndarray) -> float:
    """
    GEV分布形状パラメータ（ξ）推定 (Numba JIT)
    """
    if len(extremes) < 10:
        return np.nan

    finite_extremes = extremes[np.isfinite(extremes)]
    if len(finite_extremes) < 10:
        return np.nan

    # 頑健推定のためのL-モーメント法
    sorted_data = np.sort(finite_extremes)
    n = len(sorted_data)

    # L-モーメント計算
    l1 = np.mean(sorted_data)  # L1 = 平均

    # L2（L-スケール）
    sum_l2 = 0.0
    for i in range(n):
        weight = (2.0 * i - n + 1.0) / n
        sum_l2 += weight * sorted_data[i]
    l2 = sum_l2 / 2.0

    # L3（L-歪度）
    sum_l3 = 0.0
    for i in range(n):
        # weight計算の最適化
        term1 = i * (i - 1.0)
        term2 = 2.0 * i * (n - 1.0)
        term3 = (n - 1.0) * (n - 2.0)
        denom = n * (n - 1.0)
        if denom == 0:
            weight = 0.0
        else:
            weight = (term1 - term2 + term3) / denom
        sum_l3 += weight * sorted_data[i]
    l3 = sum_l3 / 3.0

    # L-歪度比
    if abs(l2) > 1e-8:
        tau3 = l3 / l2
        # GEV形状パラメータ関係（Hosking et al. 1985）
        # ξ ≈ 7.859*τ3 + 2.9554*τ3^2
        shape = 7.859 * tau3 + 2.9554 * tau3**2
        # 合理的範囲への制約
        shape = max(-0.5, min(shape, 0.5))
    else:
        shape = 0.0  # ガンベル分布

    return shape


@njit(fastmath=True, cache=True)
def holt_winters_レベル_udf(prices: np.ndarray) -> float:
    """
    ホルト・ウィンターズ指数平滑からレベル成分を抽出 (Numba JIT)
    """
    if len(prices) < 10:
        return np.nan

    finite_prices = prices[np.isfinite(prices)]
    if len(finite_prices) < 10:
        return np.nan

    # レベル用適切指数平滑（α = 0.3）
    alpha = 0.3
    level = finite_prices[0]  # 最初の観測値で初期化

    for i in range(1, len(finite_prices)):
        level = alpha * finite_prices[i] + (1.0 - alpha) * level

    return level


@njit(fastmath=True, cache=True)
def holt_winters_トレンド_udf(prices: np.ndarray) -> float:
    """
    ホルト・ウィンターズ指数平滑からトレンド成分を抽出 (Numba JIT)
    """
    if len(prices) < 10:
        return np.nan

    finite_prices = prices[np.isfinite(prices)]
    if len(finite_prices) < 10:
        return np.nan

    # 適切二重指数平滑（ホルト法）
    alpha = 0.3  # レベル平滑
    beta = 0.1  # トレンド平滑

    # 初期化
    level = finite_prices[0]
    trend = finite_prices[1] - finite_prices[0] if len(finite_prices) > 1 else 0.0

    for i in range(1, len(finite_prices)):
        new_level = alpha * finite_prices[i] + (1.0 - alpha) * (level + trend)
        new_trend = beta * (new_level - level) + (1.0 - beta) * trend
        level = new_level
        trend = new_trend

    return trend


@njit(fastmath=True, cache=True)
def arima_残差分散_udf(prices: np.ndarray) -> float:
    """
    ARIMA(1,1,1)モデルからの残差分散計算 (Numba JIT)
    """
    if len(prices) < 15:
        return np.nan

    finite_prices = prices[np.isfinite(prices)]
    if len(finite_prices) < 15:
        return np.nan

    # 定常性のための1次差分
    diff_prices = np.diff(finite_prices)

    if len(diff_prices) < 10:
        return np.nan

    # 差分にAR(1)モデル適合: Δy_t = φ*Δy_{t-1} + ε_t
    y = diff_prices[1:]
    x = diff_prices[:-1]

    # 適切分散計算付OLS推定
    n = len(y)
    if n < 5:
        return np.nan

    sum_x = np.sum(x)
    sum_y = np.sum(y)
    sum_xy = np.sum(x * y)
    sum_x2 = np.sum(x * x)

    # AR係数計算
    denom = n * sum_x2 - sum_x**2
    if denom != 0:
        phi = (n * sum_xy - sum_x * sum_y) / denom
    else:
        phi = 0.0

    # 残差計算
    intercept = (sum_y - phi * sum_x) / n
    residuals = y - (intercept + phi * x)

    # 残差分散（不偏推定量）
    residual_variance = np.sum(residuals**2) / (n - 2)

    return residual_variance


@njit(fastmath=True, cache=True)
def kalman_状態推定_udf(prices: np.ndarray) -> float:
    """
    価格レベル用カルマンフィルタ状態推定 (Numba JIT)
    """
    if len(prices) < 5:
        return np.nan

    finite_prices = prices[np.isfinite(prices)]
    if len(finite_prices) < 5:
        return np.nan

    # 局所レベルモデル用適切カルマンフィルタ
    # 状態: x_t = x_{t-1} + w_t (ランダムウォーク)
    # 観測: y_t = x_t + v_t (観測ノイズ)

    # 初期化
    x = finite_prices[0]  # 初期状態推定
    P = 1.0  # 初期状態分散

    # プロセスと観測ノイズ分散推定
    if len(finite_prices) > 1:
        diff_var = np.var(np.diff(finite_prices))
        obs_var = np.var(finite_prices)
        Q = max(diff_var, obs_var * 0.01)  # プロセスノイズ
        R = obs_var * 0.1  # 観測ノイズ（信号の10%）
        if R == 0:
            R = 0.1  # ゼロ除算防止
    else:
        Q = 1.0
        R = 0.1

    # カルマンフィルタ再帰
    for i in range(1, len(finite_prices)):
        # 予測ステップ
        x_pred = x  # x_{t|t-1} = x_{t-1|t-1} (ランダムウォーク)
        P_pred = P + Q  # P_{t|t-1} = P_{t-1|t-1} + Q

        # 更新ステップ
        K = P_pred / (P_pred + R)  # カルマンゲイン
        x = x_pred + K * (finite_prices[i] - x_pred)  # 更新状態推定
        P = (1.0 - K) * P_pred  # 更新状態分散

    return x


@njit(fastmath=True, cache=True)
def lowess_適合値_udf(prices: np.ndarray) -> float:
    """
    LOWESS（局所重み付け散布図平滑）適合値 (Numba JIT)
    """
    if len(prices) < 10:
        return np.nan

    finite_prices = prices[np.isfinite(prices)]
    if len(finite_prices) < 10:
        return np.nan

    n = len(finite_prices)
    bandwidth = 0.3  # 30%帯域幅
    h = max(3, int(bandwidth * n))

    # ターゲット点は最後の観測値
    target_idx = n - 1

    # ターゲット点への距離計算
    distances = np.abs(np.arange(n) - target_idx)

    # k近傍探索
    sorted_indices = np.argsort(distances)
    neighbor_indices = sorted_indices[:h]

    # 近傍抽出
    # Numbaでのインデックス配列は整数型である必要があるが、
    # 計算のためにfloatにキャスト
    x_neighbors = neighbor_indices.astype(np.float64)
    y_neighbors = finite_prices[neighbor_indices]

    # 三次重み計算
    max_dist = np.max(distances[neighbor_indices])
    weights = np.zeros(h)

    if max_dist > 0:
        for i in range(h):
            u = distances[neighbor_indices[i]] / max_dist
            if u < 1.0:
                # 三次重み関数
                weights[i] = (1.0 - u**3) ** 3
            else:
                weights[i] = 0.0
    else:
        weights = np.ones(h)

    # 重み付き最小二乗回帰
    if len(x_neighbors) >= 2:
        # 重み付き平均
        sum_w = np.sum(weights)
        if sum_w > 0:
            x_mean = np.sum(weights * x_neighbors) / sum_w
            y_mean = np.sum(weights * y_neighbors) / sum_w

            # 重み付き回帰係数
            numerator = np.sum(
                weights * (x_neighbors - x_mean) * (y_neighbors - y_mean)
            )
            denominator = np.sum(weights * (x_neighbors - x_mean) ** 2)

            if denominator > 0:
                slope = numerator / denominator
                intercept = y_mean - slope * x_mean
                fitted_value = intercept + slope * target_idx
            else:
                fitted_value = y_mean
        else:
            fitted_value = np.mean(y_neighbors)
    else:
        fitted_value = finite_prices[-1]

    return fitted_value


@njit(fastmath=True, cache=True)
def theil_sen_傾き_udf(prices: np.ndarray) -> float:
    """
    頑健傾き推定のためのタイル・セン推定量 (Numba JIT)
    """
    if len(prices) < 10:
        return np.nan

    finite_prices = prices[np.isfinite(prices)]
    if len(finite_prices) < 10:
        return np.nan

    n = len(finite_prices)
    slopes = []

    # 全ペアワイズ傾き計算（大データセット用効率サンプリング）
    max_pairs = min(1000, (n * (n - 1)) // 2)  # 効率用制限

    if max_pairs < (n * (n - 1)) // 2:
        # ペア一様サンプリング
        step = max(1, ((n * (n - 1)) // 2) // max_pairs)
        count = 0
        for i in range(n - 1):
            for j in range(i + 1, n):
                if count % step == 0:
                    if j != i:  # ゼロ除算回避
                        slope = (finite_prices[j] - finite_prices[i]) / (j - i)
                        slopes.append(slope)
                count += 1
    else:
        # 全ペア計算
        for i in range(n - 1):
            for j in range(i + 1, n):
                slope = (finite_prices[j] - finite_prices[i]) / (j - i)
                slopes.append(slope)

    if slopes:
        # 中央傾きを返す
        slopes_array = np.array(slopes)
        return np.median(slopes_array)
    else:
        return np.nan


# ==================================================================
#
# 1. NUMBA UDF ライブラリ (Part 1: C)
# Engine 1C: Technical Indicators (Logic strictly matched with Source)
#
# ==================================================================

# ----------------------------------------
# from engine_1_C_a_vast_universe_of_features.py
# ----------------------------------------


@njit(fastmath=True, cache=True)
def calculate_rsi_numba(prices: np.ndarray, period: int) -> np.ndarray:
    """
    RSI計算 (Numba JIT) - Source (Engine 1C) 完全準拠
    Logic: Cutler's RSI (SMA of Gains/Losses), not Wilder's.
    """
    n = len(prices)
    result = np.full(n, np.nan, dtype=np.float64)

    for i in range(n):
        if i < period:
            result[i] = np.nan
        else:
            gains = 0.0
            losses = 0.0

            # 直近 period 個の変化率を単純合計 (SMA方式)
            # Source: for j in range(i - period + 1, i + 1)
            for j in range(i - period + 1, i + 1):
                diff = prices[j] - prices[j - 1]
                if diff > 0:
                    gains += diff
                else:
                    losses += abs(diff)

            if gains + losses == 0:
                result[i] = 50.0
            else:
                avg_gain = gains / period
                avg_loss = losses / period

                if avg_loss == 0:
                    result[i] = 100.0
                else:
                    rs = avg_gain / avg_loss
                    result[i] = 100.0 - (100.0 / (1.0 + rs))

    return result


@njit(fastmath=True, cache=True)
def calculate_atr_numba(
    high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int
) -> np.ndarray:
    """
    ATR計算 (Numba JIT) - Source (Engine 1C) 完全準拠
    Logic: Simple Moving Average of TR (not Wilder's Smoothing)
    """
    n = len(high)
    result = np.full(n, np.nan, dtype=np.float64)
    tr = np.zeros(n, dtype=np.float64)

    # TR計算
    for i in range(n):
        if i == 0:
            tr[i] = high[i] - low[i]
        else:
            h_l = high[i] - low[i]
            h_pc = abs(high[i] - close[i - 1])
            l_pc = abs(low[i] - close[i - 1])
            tr[i] = max(h_l, max(h_pc, l_pc))

    # ATR計算 (SMA: 毎回ウィンドウを合計して誤差をSourceと一致させる)
    for i in range(n):
        if i < period:
            result[i] = np.nan
        else:
            sum_tr = 0.0
            for j in range(i - period + 1, i + 1):
                sum_tr += tr[j]
            result[i] = sum_tr / period

    return result


@njit(fastmath=True, cache=True)
def calculate_adx_numba(
    high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int
) -> np.ndarray:
    """
    ADX計算 (Numba JIT) - Source (Engine 1C) 完全準拠
    Logic: SMA based ADX
    """
    n = len(high)
    out = np.full(n, np.nan, dtype=np.float64)

    tr = np.zeros(n)
    dm_plus = np.zeros(n)
    dm_minus = np.zeros(n)

    # TR, DM+, DM- 計算
    for i in range(n):
        if i == 0:
            tr[i] = high[i] - low[i]
        else:
            h_l = high[i] - low[i]
            h_pc = abs(high[i] - close[i - 1])
            l_pc = abs(low[i] - close[i - 1])
            tr[i] = max(h_l, max(h_pc, l_pc))

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

    # DI+, DI- 配列 (中間計算用)
    di_plus = np.zeros(n)
    di_minus = np.zeros(n)

    for i in range(n):
        if i < period:
            di_plus[i] = np.nan
            di_minus[i] = np.nan
        else:
            atr_val = 0.0
            dm_plus_sum = 0.0
            dm_minus_sum = 0.0

            # 完全ループ加算 (Source準拠)
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

    # ADX計算 (DXのSMA)
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

    return out


@njit(fastmath=True, cache=True)
def calculate_di_plus_numba(
    high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int
) -> np.ndarray:
    """
    DI+ 計算 (Numba JIT) - Source (Engine 1C) 完全準拠
    """
    n = len(high)
    out = np.full(n, np.nan, dtype=np.float64)

    tr = np.zeros(n)
    dm_plus = np.zeros(n)

    for i in range(n):
        if i == 0:
            tr[i] = high[i] - low[i]
        else:
            h_l = high[i] - low[i]
            h_pc = abs(high[i] - close[i - 1])
            l_pc = abs(low[i] - close[i - 1])
            tr[i] = max(h_l, max(h_pc, l_pc))

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
    return out


@njit(fastmath=True, cache=True)
def calculate_di_minus_numba(
    high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int
) -> np.ndarray:
    """
    DI- 計算 (Numba JIT) - Source (Engine 1C) 完全準拠
    """
    n = len(high)
    out = np.full(n, np.nan, dtype=np.float64)

    tr = np.zeros(n)
    dm_minus = np.zeros(n)

    for i in range(n):
        if i == 0:
            tr[i] = high[i] - low[i]
        else:
            h_l = high[i] - low[i]
            h_pc = abs(high[i] - close[i - 1])
            l_pc = abs(low[i] - close[i - 1])
            tr[i] = max(h_l, max(h_pc, l_pc))

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
    return out


@njit(fastmath=True, cache=True)
def calculate_hma_numba(prices: np.ndarray, period: int) -> np.ndarray:
    """
    Hull Moving Average計算 (Numba JIT) - Source (Engine 1C) 完全準拠
    """
    n = len(prices)
    out = np.full(n, np.nan, dtype=np.float64)

    half_period = period // 2
    sqrt_period = int(np.sqrt(period))

    # Helper WMA function (internal to implementation logic)
    # WMA = Sum(Price * Weight) / Sum(Weights)
    # Source implementation calculates WMA on the fly inside the loop structure

    # Pre-calculate WMAs to match source flow logic structure
    wma_half = np.full(n, np.nan, dtype=np.float64)
    wma_full = np.full(n, np.nan, dtype=np.float64)
    raw_hma = np.full(n, np.nan, dtype=np.float64)

    # 1. Calculate WMA(half) and WMA(full)
    for i in range(n):
        # WMA Half
        if i >= half_period - 1:
            weight_sum = 0.0
            value_sum = 0.0
            for j in range(half_period):
                weight = half_period - j
                # idx = i - j
                value_sum += prices[i - j] * weight
                weight_sum += weight
            if weight_sum > 0:
                wma_half[i] = value_sum / weight_sum

        # WMA Full
        if i >= period - 1:
            weight_sum = 0.0
            value_sum = 0.0
            for j in range(period):
                weight = period - j
                value_sum += prices[i - j] * weight
                weight_sum += weight
            if weight_sum > 0:
                wma_full[i] = value_sum / weight_sum

        # Raw HMA
        if not np.isnan(wma_half[i]) and not np.isnan(wma_full[i]):
            raw_hma[i] = 2.0 * wma_half[i] - wma_full[i]

    # 2. Calculate WMA(sqrt) of Raw HMA
    for i in range(n):
        if i >= sqrt_period - 1:
            weight_sum = 0.0
            value_sum = 0.0
            valid = True
            for j in range(sqrt_period):
                weight = sqrt_period - j
                val = raw_hma[i - j]
                if np.isnan(val):
                    valid = False
                    break
                value_sum += val * weight
                weight_sum += weight

            if valid and weight_sum > 0:
                out[i] = value_sum / weight_sum

    return out


@njit(fastmath=True, cache=True)
def calculate_kama_numba(prices: np.ndarray, period: int) -> np.ndarray:
    """
    KAMA計算 (Numba JIT) - Source (Engine 1C) 完全準拠
    """
    n = len(prices)
    out = np.full(n, np.nan, dtype=np.float64)

    fast_ema = 2.0
    slow_ema = 30.0

    fast_sc = 2.0 / (fast_ema + 1.0)
    slow_sc = 2.0 / (slow_ema + 1.0)

    for i in range(n):
        if i < period:
            out[i] = np.nan
        else:
            # Efficiency Ratio
            # Direction = abs(price - price[n])
            direction = abs(prices[i] - prices[i - period])
            volatility = 0.0
            for j in range(i - period + 1, i + 1):
                volatility += abs(prices[j] - prices[j - 1])

            if volatility == 0:
                er = 0.0
            else:
                er = direction / volatility

            sc = (er * (fast_sc - slow_sc) + slow_sc) ** 2

            if i == period:
                # Initial value is price itself or simple MA? Source uses price[i] initialization logic structure
                out[i] = prices[i]
            else:
                if not np.isnan(out[i - 1]):
                    out[i] = out[i - 1] + sc * (prices[i] - out[i - 1])
                else:
                    out[i] = prices[i]

    return out


@njit(fastmath=True, cache=True)
def calculate_stochastic_numba(
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    k_period: int,
    d_period: int = 3,
    slow_period: int = 3,
) -> np.ndarray:
    """
    Stochastic Oscillator計算 (Numba JIT) - Source (Engine 1C) 完全準拠
    Returns: Slow %D array (consistent with Engine 1C)
    """
    n = len(close)
    stoch_k = np.full(n, np.nan, dtype=np.float64)
    stoch_d = np.full(n, np.nan, dtype=np.float64)
    stoch_slow_d = np.full(n, np.nan, dtype=np.float64)

    # %K 計算
    for i in range(n):
        if i < k_period - 1:
            stoch_k[i] = np.nan
        else:
            highest = high[i]
            lowest = low[i]
            # Search window [i - k_period + 1 : i] inclusive
            for j in range(i - k_period + 1, i):
                if high[j] > highest:
                    highest = high[j]
                if low[j] < lowest:
                    lowest = low[j]

            if highest - lowest > 0:
                stoch_k[i] = 100.0 * (close[i] - lowest) / (highest - lowest)
            else:
                stoch_k[i] = 50.0

    # %D 計算 (SMA of %K)
    for i in range(n):
        if i < k_period + d_period - 2:
            stoch_d[i] = np.nan
        else:
            sum_val = 0.0
            valid_count = 0
            for j in range(i - d_period + 1, i + 1):
                if not np.isnan(stoch_k[j]):
                    sum_val += stoch_k[j]
                    valid_count += 1
            if valid_count > 0:
                stoch_d[i] = sum_val / valid_count

    # Slow %D 計算 (SMA of %D)
    # Source calculates this as the final output in many cases or returns struct
    # Here we return Slow %D as the array
    if slow_period == 1:
        return stoch_d

    for i in range(n):
        # Valid index check: needs to wait for Stoch D
        if i < k_period + d_period + slow_period - 3:
            stoch_slow_d[i] = np.nan
        else:
            sum_val = 0.0
            valid_count = 0
            for j in range(i - slow_period + 1, i + 1):
                if not np.isnan(stoch_d[j]):
                    sum_val += stoch_d[j]
                    valid_count += 1
            if valid_count > 0:
                stoch_slow_d[i] = sum_val / valid_count

    return stoch_slow_d


@njit(fastmath=True, cache=True)
def calculate_williams_r_numba(
    high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int
) -> np.ndarray:
    """
    Williams %R計算 (Numba JIT) - Source (Engine 1C) 完全準拠
    """
    n = len(close)
    result = np.full(n, np.nan, dtype=np.float64)

    for i in range(n):
        if i < period - 1:
            result[i] = np.nan
        else:
            highest = high[i]
            lowest = low[i]

            for j in range(i - period + 1, i):
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
    """
    TRIX指標計算 (Numba JIT) - Source (Engine 1C) 完全準拠
    """
    n = len(prices)
    out = np.full(n, np.nan, dtype=np.float64)

    ema1 = np.zeros(n)
    ema2 = np.zeros(n)
    ema3 = np.zeros(n)

    alpha = 2.0 / (period + 1.0)

    # EMA 1
    for i in range(n):
        if i == 0:
            ema1[i] = prices[i]
        else:
            ema1[i] = alpha * prices[i] + (1.0 - alpha) * ema1[i - 1]

    # EMA 2
    for i in range(n):
        if i == 0:
            ema2[i] = ema1[i]
        else:
            ema2[i] = alpha * ema1[i] + (1.0 - alpha) * ema2[i - 1]

    # EMA 3
    for i in range(n):
        if i == 0:
            ema3[i] = ema2[i]
        else:
            ema3[i] = alpha * ema2[i] + (1.0 - alpha) * ema3[i - 1]

    # TRIX (ROC of EMA3)
    for i in range(n):
        if i < period * 3:  # Approximate warm up
            out[i] = np.nan
        elif ema3[i - 1] == 0:
            out[i] = 0.0
        else:
            out[i] = 10000.0 * (ema3[i] - ema3[i - 1]) / ema3[i - 1]

    return out


@njit(fastmath=True, cache=True)
def calculate_ultimate_oscillator_numba(
    high: np.ndarray, low: np.ndarray, close: np.ndarray, volume: np.ndarray
) -> np.ndarray:
    """
    Ultimate Oscillator計算 (Numba JIT) - Source (Engine 1C) 完全準拠
    """
    n = len(close)
    out = np.full(n, np.nan, dtype=np.float64)

    periods = [7, 14, 28]
    weights = [4.0, 2.0, 1.0]
    max_period = 28

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
            tr[i] = max(h_l, max(h_pc, l_pc))

    for i in range(n):
        if i < max_period:
            out[i] = np.nan
        else:
            weighted_sum = 0.0
            weight_total = 7.0  # 4+2+1

            # Iterate through 3 periods
            for idx in range(3):
                p = periods[idx]
                w = weights[idx]

                bp_sum = 0.0
                tr_sum = 0.0

                for k in range(i - p + 1, i + 1):
                    bp_sum += bp[k]
                    tr_sum += tr[k]

                if tr_sum > 0:
                    avg = bp_sum / tr_sum
                else:
                    avg = 0.0

                weighted_sum += avg * w

            out[i] = 100.0 * weighted_sum / weight_total

    return out


@njit(fastmath=True, cache=True)
def calculate_aroon_up_numba(high: np.ndarray, period: int) -> np.ndarray:
    """
    Aroon Up計算 (Numba JIT) - Source (Engine 1C) 完全準拠
    """
    n = len(high)
    out = np.full(n, np.nan, dtype=np.float64)

    for i in range(n):
        if i < period - 1:
            out[i] = np.nan
        else:
            highest_idx = i
            highest_val = high[i]

            # Loop from i - period + 1 to i (exclusive of i if looking back, but standard Aroon checks window)
            # Source: for j in range(i - period + 1, i) -> excludes current?
            # Source implementation: highest_idx initialized to i, then loop excludes i?
            # Let's strictly follow source logic structure:

            for j in range(i - period + 1, i):
                if high[j] > highest_val:
                    highest_val = high[j]
                    highest_idx = j

            # Check including current if source logic implies it via init
            # Source code: highest_idx = i. Loops j up to i.
            # So current bar IS considered.

            periods_since = i - highest_idx
            out[i] = 100.0 * (period - periods_since) / period

    return out


@njit(fastmath=True, cache=True)
def calculate_aroon_down_numba(low: np.ndarray, period: int) -> np.ndarray:
    """
    Aroon Down計算 (Numba JIT) - Source (Engine 1C) 完全準拠
    """
    n = len(low)
    out = np.full(n, np.nan, dtype=np.float64)

    for i in range(n):
        if i < period - 1:
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

    return out


@njit(fastmath=True, cache=True)
def calculate_tsi_numba(
    prices: np.ndarray, long_period: int, short_period: int
) -> np.ndarray:
    """
    True Strength Index計算 (Numba JIT) - Source (Engine 1C) 完全準拠
    Note: Source uses TSI(25, 13) usually.
    This implementation matches the EMA recursion structure of the source.
    """
    n = len(prices)
    out = np.full(n, np.nan, dtype=np.float64)

    # 1. Momentum
    momentum = np.zeros(n)
    for i in range(1, n):
        momentum[i] = prices[i] - prices[i - 1]

    # 2. First EMA (Long Period)
    alpha_long = 2.0 / (long_period + 1.0)
    ema1_mom = np.zeros(n)
    ema1_abs = np.zeros(n)

    for i in range(n):
        if i == 0:
            ema1_mom[i] = momentum[i]
            ema1_abs[i] = abs(momentum[i])
        else:
            ema1_mom[i] = (
                alpha_long * momentum[i] + (1.0 - alpha_long) * ema1_mom[i - 1]
            )
            ema1_abs[i] = (
                alpha_long * abs(momentum[i]) + (1.0 - alpha_long) * ema1_abs[i - 1]
            )

    # 3. Second EMA (Short Period)
    alpha_short = 2.0 / (short_period + 1.0)
    ema2_mom = np.zeros(n)
    ema2_abs = np.zeros(n)

    for i in range(n):
        if i == 0:
            ema2_mom[i] = ema1_mom[i]
            ema2_abs[i] = ema1_abs[i]
        else:
            ema2_mom[i] = (
                alpha_short * ema1_mom[i] + (1.0 - alpha_short) * ema2_mom[i - 1]
            )
            ema2_abs[i] = (
                alpha_short * ema1_abs[i] + (1.0 - alpha_short) * ema2_abs[i - 1]
            )

    # 4. TSI
    for i in range(n):
        if i < long_period + short_period:
            out[i] = np.nan
        elif ema2_abs[i] == 0:
            out[i] = 0.0
        else:
            out[i] = 100.0 * ema2_mom[i] / ema2_abs[i]

    return out


@njit(fastmath=True, cache=True)
def wma_rolling_numba(arr: np.ndarray, window: int) -> float:
    """
    Weighted Moving Average (Numba JIT) - 最新の1点のみ計算
    """
    n = len(arr)
    if n < window:
        return np.nan

    # 末尾 window 個のデータを使用
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
    """
    単純移動平均 (Numba JIT) - 最新値のみ
    Engine 1C: sma計算用
    """
    n = len(arr)
    if n < window:
        return np.nan

    s = 0.0
    for i in range(n - window, n):
        s += arr[i]
    return s / window


@njit(fastmath=True, cache=True)
def rolling_trend_consistency_numba(close: np.ndarray, period: int) -> float:
    """
    トレンド一貫性 (Numba JIT) - 最新値のみ
    Engine 1C: trend_consistency計算用
    """
    n = len(close)
    # diff計算で1要素、sign().diff()でさらに1要素減るため、最低 period + 2 必要
    if n < period + 2:
        return np.nan

    # 最新の period 期間の direction_changes の平均を計算
    changes_sum = 0.0
    count = 0

    # ループ範囲: 最新の period 個の「変化」を計算
    # インデックス i は close の末尾から走査
    for i in range(n - period, n):
        # i時点での変化を計算するには i, i-1, i-2 が必要
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
#
# 1. NUMBA UDF ライブラリ (Part 4: 1D Volatility & Volume & Price Action)
#
# ==================================================================

# ----------------------------------------
# from engine_1_D_a_vast_universe_of_features.py
# ----------------------------------------


@njit(fastmath=True, cache=True)
def hv_robust_udf(returns: np.ndarray) -> float:
    """
    ロバストボラティリティ (Numba JIT) - 単一値
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
def chaikin_volatility_udf(
    high: np.ndarray, low: np.ndarray, window: int
) -> np.ndarray:
    """
    Chaikin Volatility (Numba JIT) - 配列返し
    Note: 元コードのロジック（2つのウィンドウのSMAのROC）を再現
    """
    n = len(high)
    result = np.full(n, np.nan, dtype=np.float64)

    if n < window * 2:
        return result

    hl_range = high - low

    # SMA計算 (全体)
    # sma[i] = mean(hl_range[i-window+1 : i+1])
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

    # Chaikin Vol calculation: (SMA_curr - SMA_prev) / SMA_prev * 100
    # 元コードの window は "Latest window" vs "Previous window" なので
    # time t での sma と time t-window での sma を比較

    for i in range(window * 2 - 1, n):
        curr_sma = sma[i]
        prev_sma = sma[i - window]

        if not np.isnan(curr_sma) and not np.isnan(prev_sma) and prev_sma > 0:
            result[i] = (curr_sma - prev_sma) / prev_sma * 100.0

    return result


@njit(fastmath=True, cache=True)
def mass_index_udf(high: np.ndarray, low: np.ndarray, window: int) -> np.ndarray:
    """
    Mass Index (Numba JIT) - 配列返し
    """
    n = len(high)
    result = np.full(n, np.nan, dtype=np.float64)

    if (
        n < window + 16
    ):  # EMA9(initial 9) + EMA9(initial 9) -> approx 16-18 buffer needed
        return result

    hl_range = high - low

    # EMA helper
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

    # Rolling Sum of Ratio (window)
    sum_ratio = 0.0
    # initialize
    for i in range(window):
        sum_ratio += ratio[i]

    # sliding window
    for i in range(window, n):
        sum_ratio = sum_ratio - ratio[i - window] + ratio[i]
        result[i] = sum_ratio

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

    # Sliding window sum
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
def mfi_udf(
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    volume: np.ndarray,
    window: int,
) -> np.ndarray:
    """
    Money Flow Index (Numba JIT) - 配列返し
    """
    n = len(close)
    mfi = np.full(n, np.nan, dtype=np.float64)

    if n < window + 1:
        return mfi

    tp = (high + low + close) / 3.0
    rmf = tp * volume

    # pos/neg flow arrays
    pos_flow = np.zeros(n)
    neg_flow = np.zeros(n)

    for i in range(1, n):
        if tp[i] > tp[i - 1]:
            pos_flow[i] = rmf[i]
        elif tp[i] < tp[i - 1]:
            neg_flow[i] = rmf[i]

    # Rolling sums
    sum_pos = 0.0
    sum_neg = 0.0

    # Init window (from index 1 to window)
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

    # Slide
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
def vwap_udf(
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    volume: np.ndarray,
    window: int,
) -> np.ndarray:
    """
    VWAP (Rolling) (Numba JIT) - 配列返し
    """
    n = len(close)
    vwap = np.full(n, np.nan, dtype=np.float64)

    if n < window:
        return vwap

    tp = (high + low + close) / 3.0
    tp_vol = tp * volume

    sum_tp_vol = 0.0
    sum_vol = 0.0

    for i in range(window):
        sum_tp_vol += tp_vol[i]
        sum_vol += volume[i]

    if sum_vol > 0:
        vwap[window - 1] = sum_tp_vol / sum_vol

    for i in range(window, n):
        sum_tp_vol = sum_tp_vol - tp_vol[i - window] + tp_vol[i]
        sum_vol = sum_vol - volume[i - window] + volume[i]

        if sum_vol > 0:
            vwap[i] = sum_tp_vol / sum_vol

    return vwap


@njit(fastmath=True, cache=True)
def obv_udf(close: np.ndarray, volume: np.ndarray) -> np.ndarray:
    """
    On Balance Volume (Numba JIT) - 配列返し
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


@njit(fastmath=True, cache=True)
def accumulation_distribution_udf(
    high: np.ndarray, low: np.ndarray, close: np.ndarray, volume: np.ndarray
) -> np.ndarray:
    """
    Accumulation/Distribution Line (Numba JIT) - 配列返し
    """
    n = len(close)
    ad = np.zeros(n, dtype=np.float64)

    # i=0
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
def force_index_udf(close: np.ndarray, volume: np.ndarray) -> np.ndarray:
    """
    Force Index (Numba JIT) - 配列返し
    """
    n = len(close)
    fi = np.zeros(n, dtype=np.float64)

    for i in range(1, n):
        fi[i] = (close[i] - close[i - 1]) * volume[i]

    return fi


@njit(fastmath=True, cache=True)
def pivot_point_udf(
    high: float, low: float, close: float
) -> Tuple[float, float, float, float, float]:
    """
    ピボットポイント計算 (Numba JIT)
    Engine 1D: pivot_point_udf のロジックを完全再現
    """
    if not (np.isfinite(high) and np.isfinite(low) and np.isfinite(close)):
        return np.nan, np.nan, np.nan, np.nan, np.nan

    pivot = (high + low + close) / 3.0
    r1 = 2.0 * pivot - low
    s1 = 2.0 * pivot - high
    r2 = pivot + (high - low)
    s2 = pivot - (high - low)

    return pivot, r1, s1, r2, s2


@njit(fastmath=True, cache=True)
def fibonacci_levels_udf(high: np.ndarray, low: np.ndarray, window: int) -> np.ndarray:
    """
    フィボナッチリトレースメントレベル計算 (Numba JIT) - 配列返し
    Returns: array of shape (n, 5)
    """
    n = len(high)
    result = np.full((n, 5), np.nan, dtype=np.float64)

    if n < window:
        return result

    fib_ratios = np.array([0.236, 0.382, 0.500, 0.618, 0.786])

    for i in range(window - 1, n):
        start_idx = i - window + 1

        w_high = -1e30
        w_low = 1e30

        for j in range(start_idx, i + 1):
            if high[j] > w_high:
                w_high = high[j]
            if low[j] < w_low:
                w_low = low[j]

        diff = w_high - w_low
        if diff > 0:
            for k in range(5):
                result[i, k] = w_high - (fib_ratios[k] * diff)
        else:
            # Flat range
            for k in range(5):
                result[i, k] = w_high

    return result


@njit(fastmath=True, cache=True)
def candlestick_patterns_udf(
    open_prices: np.ndarray, high: np.ndarray, low: np.ndarray, close: np.ndarray
) -> np.ndarray:
    """
    ローソク足パターン認識 (Numba JIT) - 配列返し
    """
    n = len(close)
    patterns = np.zeros(n, dtype=np.int64)  # or float64

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
            patterns[i] = 0
            continue

        body_ratio = body_size / total_range
        upper_shadow_ratio = upper_shadow / total_range
        lower_shadow_ratio = lower_shadow / total_range

        # Doji
        if body_ratio < 0.1:
            patterns[i] = 3
        # Hammer
        elif lower_shadow_ratio > 0.6 and upper_shadow_ratio < 0.1:
            patterns[i] = 1
        # Shooting Star
        elif upper_shadow_ratio > 0.6 and lower_shadow_ratio < 0.1:
            patterns[i] = 2
        # Bullish
        elif body_ratio > 0.6 and c > o:
            patterns[i] = 4
        # Bearish
        elif body_ratio > 0.6 and c < o:
            patterns[i] = 5
        else:
            patterns[i] = 0

    return patterns


@njit(fastmath=True, cache=True)
def donchian_channel_udf(high: np.ndarray, low: np.ndarray, window: int) -> np.ndarray:
    """
    ドンチャンチャネル計算 (Numba JIT) - 配列返し (n, 3)
    Returns: array of shape (n, 3) -> [upper, middle, lower]
    """
    n = len(high)
    # Shape (n, 3) so slicing result[-1] gives array of 3 values
    result = np.full((n, 3), np.nan, dtype=np.float64)

    if n < window:
        return result

    for i in range(window - 1, n):
        start_idx = i - window + 1

        # Find max high / min low in window
        # Note: Donchian usually uses past N bars excluding current,
        # but standard implementation includes current if not lagged.
        # Source logic: period_high = max(high[j]) for j in range(n-window, n) -> includes current if n is current idx+1

        w_high = -1e30
        w_low = 1e30

        for j in range(start_idx, i + 1):
            if high[j] > w_high:
                w_high = high[j]
            if low[j] < w_low:
                w_low = low[j]

        result[i, 0] = w_high  # Upper
        result[i, 1] = (w_high + w_low) / 2.0  # Middle
        result[i, 2] = w_low  # Lower

    return result


@njit(fastmath=True, cache=True)
def commodity_channel_index_udf(
    high: np.ndarray, low: np.ndarray, close: np.ndarray, window: int
) -> np.ndarray:
    """
    Commodity Channel Index (CCI) 計算 (Numba JIT) - 配列返し
    """
    n = len(close)
    cci = np.full(n, np.nan, dtype=np.float64)

    if n < window:
        return cci

    tp = (high + low + close) / 3.0

    # Simple Moving Average of TP
    sma_tp = np.full(n, np.nan, dtype=np.float64)
    sum_tp = 0.0

    for i in range(window):
        sum_tp += tp[i]
    sma_tp[window - 1] = sum_tp / window

    for i in range(window, n):
        sum_tp = sum_tp - tp[i - window] + tp[i]
        sma_tp[i] = sum_tp / window

    # Mean Deviation
    # MD = Sum(|TP - SMA_TP|) / N
    # This is expensive to calculate rolling without full loop
    for i in range(window - 1, n):
        current_sma = sma_tp[i]
        dev_sum = 0.0
        start_idx = i - window + 1

        for j in range(start_idx, i + 1):
            dev_sum += abs(tp[j] - current_sma)

        mean_dev = dev_sum / window

        if mean_dev > 0:
            cci[i] = (tp[i] - current_sma) / (0.015 * mean_dev)
        else:
            cci[i] = 0.0

    return cci


@njit(fastmath=True, cache=True)
def hv_standard_udf(returns: np.ndarray) -> float:
    """
    標準ヒストリカルボラティリティ (Numba JIT)
    単純な標準偏差として計算
    """
    if len(returns) < 2:
        return np.nan

    finite_returns = returns[np.isfinite(returns)]
    if len(finite_returns) < 2:
        return np.nan

    return np.std(finite_returns)


# ----------------------------------------
# from engine_1_E_a_vast_universe_of_features.py
# ----------------------------------------


@njit(fastmath=True, cache=True)
def numba_fft(x: np.ndarray) -> np.ndarray:
    """
    NumbaネイティブFFT実装 (学習用スクリプトの安定版ロジックを再現)
    np.fftがNumbaでサポートされていないための代替実装
    """
    n = x.shape[0]

    # nが2のべき乗でない場合、ゼロパディングを行う
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

    # ビット反転置換
    j = 0
    for i in range(1, n):
        bit = n >> 1
        while j & bit:
            j ^= bit
            bit >>= 1
        j ^= bit
        if i < j:
            x[i], x[j] = x[j], x[i]

    # バタフライ演算（反復処理）
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
    """
    スペクトル重心計算 (Numba JIT)
    """
    n = len(signal)
    # 有限値のみ抽出
    finite_data = signal[np.isfinite(signal)]

    # データ長チェック (FFTのため最低限の長さが必要)
    if len(finite_data) < 4:
        return np.nan

    # NumbaネイティブFFTによるスペクトル計算
    fft_values = numba_fft(finite_data)
    magnitude_spectrum = np.abs(fft_values[: len(finite_data) // 2])

    if len(magnitude_spectrum) == 0:
        return np.nan

    # 周波数ビンの作成
    freqs = np.linspace(0, 0.5, len(magnitude_spectrum))

    # スペクトル重心計算
    total_magnitude = np.sum(magnitude_spectrum)
    if total_magnitude > 0:
        return np.sum(freqs * magnitude_spectrum) / total_magnitude
    else:
        return 0.0


@njit(fastmath=True, cache=True)
def spectral_bandwidth_udf(signal: np.ndarray) -> float:
    """
    スペクトル帯域幅計算 (Numba JIT)
    """
    n = len(signal)
    finite_data = signal[np.isfinite(signal)]

    if len(finite_data) < 4:
        return np.nan

    fft_values = numba_fft(finite_data)
    magnitude_spectrum = np.abs(fft_values[: len(finite_data) // 2])

    if len(magnitude_spectrum) == 0:
        return np.nan

    freqs = np.linspace(0, 0.5, len(magnitude_spectrum))
    total_magnitude = np.sum(magnitude_spectrum)

    if total_magnitude > 0:
        centroid = np.sum(freqs * magnitude_spectrum) / total_magnitude
        # 帯域幅（重心周りの分散）計算
        bandwidth = np.sqrt(
            np.sum(((freqs - centroid) ** 2) * magnitude_spectrum) / total_magnitude
        )
        return bandwidth
    else:
        return 0.0


@njit(fastmath=True, cache=True)
def spectral_rolloff_udf(signal: np.ndarray) -> float:
    """
    スペクトルロールオフ計算 (Numba JIT)
    """
    n = len(signal)
    finite_data = signal[np.isfinite(signal)]

    if len(finite_data) < 4:
        return np.nan

    fft_values = numba_fft(finite_data)
    magnitude_spectrum = np.abs(fft_values[: len(finite_data) // 2])

    if len(magnitude_spectrum) == 0:
        return np.nan

    # パワースペクトルに変換
    power_spectrum = magnitude_spectrum**2
    total_power = np.sum(power_spectrum)

    if total_power > 0:
        cumulative_power = np.cumsum(power_spectrum)
        threshold = 0.85 * total_power  # rolloff_ratio = 0.85

        rolloff_idx = np.where(cumulative_power >= threshold)[0]
        if len(rolloff_idx) > 0:
            return rolloff_idx[0] / (2.0 * len(magnitude_spectrum))

    return 0.0


@njit(fastmath=True, cache=True)
def spectral_flatness_udf(signal: np.ndarray) -> float:
    """
    スペクトル平坦度計算 (Numba JIT)
    """
    n = len(signal)
    finite_data = signal[np.isfinite(signal)]

    if len(finite_data) < 4:
        return np.nan

    fft_values = numba_fft(finite_data)
    magnitude_spectrum = np.abs(fft_values[: len(finite_data) // 2])

    if len(magnitude_spectrum) == 0:
        return np.nan

    # 0に近い値を避けるため小さな値を加算
    magnitude_spectrum = magnitude_spectrum + 1e-10

    geometric_mean = np.exp(np.mean(np.log(magnitude_spectrum)))
    arithmetic_mean = np.mean(magnitude_spectrum)

    if arithmetic_mean > 0:
        return geometric_mean / arithmetic_mean
    else:
        return 0.0


@njit(fastmath=True, cache=True)
def spectral_entropy_udf(signal: np.ndarray) -> float:
    """
    スペクトルエントロピー計算 (Numba JIT)
    """
    n = len(signal)
    finite_data = signal[np.isfinite(signal)]

    if len(finite_data) < 4:
        return np.nan

    fft_values = numba_fft(finite_data)
    # Engine 1E では power_spectrum を使用してエントロピーを計算している
    power_spectrum = np.abs(fft_values[: len(finite_data) // 2]) ** 2

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
    else:
        return 0.0


@njit(fastmath=True, cache=True)
def wavelet_energy_udf(signal: np.ndarray, window_size: int) -> np.ndarray:
    """
    ウェーブレットエネルギー計算 (Numba JIT) - 配列返し
    """
    n = len(signal)
    result = np.full(n, np.nan, dtype=np.float64)

    if n < window_size:
        return result

    # Rolling window calculation
    for i in range(window_size - 1, n):
        # Extract window: signal[i - window_size + 1 : i + 1]
        start_idx = i - window_size + 1
        window_data = signal[start_idx : i + 1]

        # Logic from original function (applied to window)
        finite_data = window_data[np.isfinite(window_data)]

        if len(finite_data) < window_size // 2:
            result[i] = np.nan
            continue

        level_energy = 0.0
        current_signal = finite_data.copy()
        levels = 4

        for level in range(min(levels, 4)):
            if len(current_signal) < 4:
                break

            # ローパスフィルタ（移動平均）
            # Output size is len // 2
            filtered_len = len(current_signal) // 2
            filtered = np.zeros(filtered_len)

            for j in range(filtered_len):
                idx = j * 2
                if idx + 1 < len(current_signal):
                    filtered[j] = (current_signal[idx] + current_signal[idx + 1]) / 2.0

            # エネルギー計算 (元ロジック準拠: フィルタ後の二乗和を加算)
            level_energy += np.sum(filtered**2)
            current_signal = filtered

        result[i] = level_energy

    return result


@njit(fastmath=True, cache=True)
def wavelet_entropy_udf(signal: np.ndarray) -> float:
    """
    ウェーブレットエントロピー (Numba JIT)
    学習用スクリプトの `wavelet_entropy_udf` は
    実質的に信号の二乗値に基づくエントロピー計算を行っている
    """
    finite_data = signal[np.isfinite(signal)]
    if len(finite_data) > 0:
        squared_data = finite_data**2
        # np.sum(-p log p) の形式
        return -np.sum(squared_data * np.log2(np.abs(squared_data) + 1e-10))
    return np.nan


@njit(fastmath=True, cache=True)
def hilbert_amplitude_udf(signal: np.ndarray) -> float:
    """
    ヒルベルト変換による振幅包絡線計算 (Numba JIT)
    Engine 1E の `hilbert_amplitude_udf` ロジックを完全再現
    FFTを使用した近似ヒルベルト変換を行い、その振幅の平均を返す
    """
    finite_data = signal[np.isfinite(signal)]
    n_samples = len(finite_data)

    if n_samples < 4:
        return np.nan

    # 近似ヒルベルト変換（90度位相シフト）
    # FFTを使用した近似実装
    fft_signal = numba_fft(finite_data)

    # 90度位相シフト
    hilbert_fft = fft_signal.copy()
    for j in range(1, n_samples // 2):
        hilbert_fft[j] *= -1j
        hilbert_fft[n_samples - j] *= 1j

    # IFFT相当の処理（簡易版：実部のみ取得）
    # Note: numba_fftはパディングされている可能性があるが、
    # ここでは元のデータ長に合わせて切り詰める必要がある
    hilbert_signal = np.real(hilbert_fft)[:n_samples]

    # 振幅包絡線
    amplitude_envelope = np.sqrt(finite_data**2 + hilbert_signal**2)
    return np.mean(amplitude_envelope)


@njit(fastmath=True, cache=True)
def hilbert_phase_var_udf(signal: np.ndarray) -> float:
    """
    ヒルベルト位相分散 (Numba JIT)
    学習用スクリプトの軽量版ロジック（np.rollによる近似解析信号）を使用
    """
    finite_data = signal[np.isfinite(signal)]
    if len(finite_data) > 1:
        # 解析信号の近似: signal + j * Hilbert(signal)
        # ここでは簡易的に90度位相シフトとして1サンプルシフトを使用
        analytic_signal = finite_data + 1j * np.roll(finite_data, 1)
        return np.var(np.angle(analytic_signal))
    return np.nan


@njit(fastmath=True, cache=True)
def hilbert_phase_stability_udf(signal: np.ndarray) -> float:
    """
    ヒルベルト位相安定性 (Numba JIT)
    学習用スクリプトの軽量版ロジックを再現
    """
    finite_data = signal[np.isfinite(signal)]
    if len(finite_data) > 2:
        # データの標準偏差をチェック（Engine 1Eの修正箇所を反映）
        if np.std(finite_data) < 1e-10:
            return 1.0

        analytic_signal = finite_data + 1j * np.roll(finite_data, 1)
        phase_diff_std = np.std(np.diff(np.angle(analytic_signal)))

        return 1.0 / (1.0 + phase_diff_std + 1e-10)
    return np.nan


@njit(fastmath=True, cache=True)
def hilbert_freq_mean_udf(signal: np.ndarray) -> float:
    """
    ヒルベルト瞬時周波数平均 (Numba JIT)
    """
    finite_data = signal[np.isfinite(signal)]
    if len(finite_data) > 2:
        analytic_signal = finite_data + 1j * np.roll(finite_data, 1)
        # np.unwrapはNumba非対応のため単純差分で近似
        instant_freq = np.abs(np.diff(np.angle(analytic_signal)))
        return np.mean(instant_freq)
    return np.nan


@njit(fastmath=True, cache=True)
def hilbert_freq_std_udf(signal: np.ndarray) -> float:
    """
    ヒルベルト瞬時周波数標準偏差 (Numba JIT)
    """
    finite_data = signal[np.isfinite(signal)]
    if len(finite_data) > 2:
        analytic_signal = finite_data + 1j * np.roll(finite_data, 1)
        instant_freq = np.abs(np.diff(np.angle(analytic_signal)))
        return np.std(instant_freq)
    return np.nan


@njit(fastmath=True, cache=True)
def acoustic_power_udf(signal: np.ndarray) -> float:
    """
    音響パワー (RMS) (Numba JIT)
    """
    finite_data = signal[np.isfinite(signal)]
    if len(finite_data) > 0:
        return np.sqrt(np.mean(finite_data**2))
    return np.nan


@njit(fastmath=True, cache=True)
def acoustic_frequency_udf(signal: np.ndarray) -> float:
    """
    音響周波数 (ゼロクロッシング率) (Numba JIT)
    """
    finite_data = signal[np.isfinite(signal)]
    if len(finite_data) > 1:
        zero_crossings = 0
        for j in range(1, len(finite_data)):
            if finite_data[j - 1] * finite_data[j] < 0:
                zero_crossings += 1

        # 周波数推定 (ゼロクロッシング率 / 2)
        # sample_rateは1.0と仮定
        return zero_crossings / (2.0 * len(finite_data))
    return np.nan


# ----------------------------------------
# from engine_1_F_a_vast_universe_of_features.py
# ----------------------------------------


@njit(fastmath=True, cache=True)
def rolling_network_density_udf(prices: np.ndarray) -> float:
    """
    ネットワーク密度計算 (Numba JIT)
    """
    if len(prices) < 10:
        return np.nan

    finite_prices = prices[np.isfinite(prices)]
    if len(finite_prices) < 10:
        return np.nan

    window_n = len(finite_prices)
    threshold = np.std(finite_prices) * 0.5

    edge_count = 0
    max_possible_edges = window_n * (window_n - 1) / 2

    for j in range(window_n - 1):
        for k in range(j + 1, window_n):
            price_diff = abs(finite_prices[j] - finite_prices[k])
            if price_diff <= threshold:
                edge_count += 1

    if max_possible_edges > 0:
        return edge_count / max_possible_edges
    else:
        return 0.0


@njit(fastmath=True, cache=True)
def rolling_network_clustering_udf(prices: np.ndarray) -> float:
    """
    ネットワーククラスタリング係数 (Numba JIT)
    """
    if len(prices) < 15:
        return np.nan

    finite_prices = prices[np.isfinite(prices)]
    if len(finite_prices) < 15:
        return np.nan

    window_n = len(finite_prices)
    threshold = np.std(finite_prices) * 0.5

    # 隣接行列の構築
    adjacency = np.zeros((window_n, window_n), dtype=boolean)

    for j in range(window_n):
        for k in range(window_n):
            if j != k:
                price_diff = abs(finite_prices[j] - finite_prices[k])
                if price_diff <= threshold:
                    adjacency[j, k] = True

    total_clustering = 0.0
    valid_nodes = 0

    for j in range(window_n):
        neighbors = []
        for k in range(window_n):
            if adjacency[j, k]:
                neighbors.append(k)

        k_neighbors = len(neighbors)
        if k_neighbors < 2:
            continue

        neighbor_connections = 0
        for idx1 in range(len(neighbors)):
            for idx2 in range(idx1 + 1, len(neighbors)):
                n1, n2 = neighbors[idx1], neighbors[idx2]
                if adjacency[n1, n2]:
                    neighbor_connections += 1

        max_connections = k_neighbors * (k_neighbors - 1) / 2
        if max_connections > 0:
            total_clustering += neighbor_connections / max_connections
            valid_nodes += 1

    if valid_nodes > 0:
        return total_clustering / valid_nodes
    else:
        return 0.0


@njit(fastmath=True, cache=True)
def rolling_vocabulary_diversity_udf(prices: np.ndarray) -> float:
    """
    語彙多様性指標 (Numba JIT)
    """
    if len(prices) < 10:
        return np.nan

    finite_prices = prices[np.isfinite(prices)]
    if len(finite_prices) < 10:
        return np.nan

    std_price = np.std(finite_prices)
    mean_price = np.mean(finite_prices)

    if std_price == 0:
        return 0.0

    n_bins = 10
    price_min = mean_price - 2 * std_price
    price_max = mean_price + 2 * std_price
    bin_width = (price_max - price_min) / n_bins

    # Numbaではsetを使えない場合があるが、
    # 最近のバージョンではサポートされている。
    # 安全のため、固定長配列でビンをカウントするか、
    # リストを使ってユニーク数を数える

    # ここでは簡易的にビンインデックスを計算し、ユニーク数をカウント
    # Numba set workaround: use sort and unique logic or simple boolean array if range is small
    # bin index is 0..9. Use array of size 10.

    vocab_flags = np.zeros(10, dtype=boolean)
    total_tokens = 0

    for price in finite_prices:
        if bin_width > 0:
            bin_idx = int((price - price_min) / bin_width)
            bin_idx = max(0, min(bin_idx, n_bins - 1))
        else:
            bin_idx = 0

        vocab_flags[bin_idx] = True
        total_tokens += 1

    unique_vocab = 0
    for k in range(10):
        if vocab_flags[k]:
            unique_vocab += 1

    if total_tokens > 0:
        return unique_vocab / total_tokens
    else:
        return 0.0


@njit(fastmath=True, cache=True)
def rolling_linguistic_complexity_udf(prices: np.ndarray) -> float:
    """
    言語的複雑性 (Numba JIT)
    """
    if len(prices) < 20:
        return np.nan

    finite_prices = prices[np.isfinite(prices)]
    if len(finite_prices) < 20:
        return np.nan

    price_changes = np.diff(finite_prices)
    threshold = np.std(price_changes) * 0.1

    # Syntax sequence generation
    # -1, 0, 1 mapped to 0, 1, 2 for array indexing if needed,
    # but logic uses tuple set counting. Numba set of tuples can be tricky.
    # We will use a manual counting approach for limited n-grams.
    # Since outputs are -1, 0, 1, we can map them to base-3 integers.

    syntax_sequence = np.zeros(len(price_changes), dtype=int64)
    for i in range(len(price_changes)):
        change = price_changes[i]
        if change > threshold:
            syntax_sequence[i] = 1
        elif change < -threshold:
            syntax_sequence[i] = -1
        else:
            syntax_sequence[i] = 0

    if len(syntax_sequence) < 3:
        return 0.0

    # Count unique bigrams
    # Bigrams: (-1,-1), (-1,0), ... (1,1). 3*3 = 9 possibilities.
    bigram_counts = np.zeros(9, dtype=boolean)
    for j in range(len(syntax_sequence) - 1):
        # Map (-1,0,1) -> (0,1,2)
        c1 = syntax_sequence[j] + 1
        c2 = syntax_sequence[j + 1] + 1
        idx = c1 * 3 + c2
        bigram_counts[idx] = True

    num_bigrams = 0
    for k in range(9):
        if bigram_counts[k]:
            num_bigrams += 1

    # Count unique trigrams
    # 3*3*3 = 27 possibilities
    trigram_counts = np.zeros(27, dtype=boolean)
    for j in range(len(syntax_sequence) - 2):
        c1 = syntax_sequence[j] + 1
        c2 = syntax_sequence[j + 1] + 1
        c3 = syntax_sequence[j + 2] + 1
        idx = c1 * 9 + c2 * 3 + c3
        trigram_counts[idx] = True

    num_trigrams = 0
    for k in range(27):
        if trigram_counts[k]:
            num_trigrams += 1

    max_bigrams = min(9, len(syntax_sequence) - 1)
    max_trigrams = min(27, len(syntax_sequence) - 2)

    if max_bigrams > 0 and max_trigrams > 0:
        bigram_complexity = num_bigrams / max_bigrams
        trigram_complexity = num_trigrams / max_trigrams
        return (bigram_complexity + trigram_complexity) / 2.0
    else:
        return 0.0


@njit(fastmath=True, cache=True)
def rolling_semantic_flow_udf(prices: np.ndarray) -> float:
    """
    意味的流れ (Numba JIT)
    """
    if len(prices) < 15:
        return np.nan

    finite_prices = prices[np.isfinite(prices)]
    if len(finite_prices) < 15:
        return np.nan

    window_n = len(finite_prices)
    window_size_local = min(5, window_n // 3)

    # ベクトルのリストの代わりに2D配列を使用
    # 最大想定サイズで確保し、カウンタで管理
    max_vectors = window_n
    semantic_vectors = np.zeros((max_vectors, 2), dtype=float64)
    vec_count = 0

    for j in range(window_size_local, window_n - window_size_local):
        neighborhood = finite_prices[j - window_size_local : j + window_size_local + 1]
        center_price = finite_prices[j]
        relative_positions = neighborhood - center_price

        if len(relative_positions) > 1:
            mean_rel = np.mean(relative_positions)
            std_rel = np.std(relative_positions)
            semantic_vectors[vec_count, 0] = mean_rel
            semantic_vectors[vec_count, 1] = std_rel
            vec_count += 1

    if vec_count < 2:
        return np.nan

    flow_continuity = 0.0
    valid_pairs = 0

    for j in range(vec_count - 1):
        vec1 = semantic_vectors[j]
        vec2 = semantic_vectors[j + 1]

        norm1 = np.sqrt(np.sum(vec1**2))
        norm2 = np.sqrt(np.sum(vec2**2))

        if norm1 > 1e-10 and norm2 > 1e-10:
            cosine_sim = np.sum(vec1 * vec2) / (norm1 * norm2)
            flow_continuity += cosine_sim
            valid_pairs += 1

    if valid_pairs > 0:
        semantic_flow = flow_continuity / valid_pairs
        return (semantic_flow + 1.0) / 2.0
    else:
        return 0.0


@njit(fastmath=True, cache=True)
def rolling_golden_ratio_adherence_udf(prices: np.ndarray) -> float:
    """
    黄金比固着度 (Numba JIT)
    """
    if len(prices) < 10:
        return np.nan

    finite_prices = prices[np.isfinite(prices)]
    if len(finite_prices) < 10:
        return np.nan

    golden_ratio = 1.618033988749895  # (1 + sqrt(5)) / 2
    local_window = min(8, len(finite_prices) // 2)

    adherence_sum = 0.0
    adherence_count = 0

    for j in range(local_window, len(finite_prices) - local_window):
        local_subwindow = finite_prices[j - local_window : j + local_window + 1]
        local_high = np.max(local_subwindow)
        local_low = np.min(local_subwindow)

        if local_low > 0:
            ratio = local_high / local_low
            deviation = abs(ratio - golden_ratio) / golden_ratio
            adherence = 1.0 / (1.0 + deviation)
            adherence_sum += adherence
            adherence_count += 1

    if adherence_count > 0:
        return adherence_sum / adherence_count
    else:
        return np.nan


@njit(fastmath=True, cache=True)
def rolling_symmetry_measure_udf(prices: np.ndarray) -> float:
    """
    対称性測定 (Numba JIT)
    """
    if len(prices) < 20:
        return np.nan

    finite_prices = prices[np.isfinite(prices)]
    if len(finite_prices) < 20:
        return np.nan

    window_n = len(finite_prices)
    center = window_n // 2

    left_half = finite_prices[:center]
    # ウィンドウサイズが奇数の場合、中心を含まないように調整
    if window_n % 2 == 1:
        right_half = finite_prices[center + 1 :]
    else:
        right_half = finite_prices[center:]

    # 右半分を反転（鏡像）
    right_half_reversed = right_half[::-1]  # Numba supports slicing with step
    # しかし Numba array slicing reverse might need contiguous array check
    # Let's use manual copy/reverse to be safe or assume slice works
    # Using slice creates a view, usually fine.

    min_len = min(len(left_half), len(right_half_reversed))
    if min_len < 5:
        return np.nan

    left_target = left_half[-min_len:]
    right_target = right_half_reversed[:min_len]

    # 正規化関数ロジック展開
    mean_left = np.mean(left_target)
    std_left = np.std(left_target)

    mean_right = np.mean(right_target)
    std_right = np.std(right_target)

    if std_left <= 1e-10 or std_right <= 1e-10:
        return 0.0

    # 相関係数計算 (手動)
    # Corr(X, Y) = Mean((X-uX)(Y-uY)) / (stdX * stdY)
    # ここでは正規化してから積の平均をとるのと同等

    covariance = np.mean((left_target - mean_left) * (right_target - mean_right))
    correlation = covariance / (std_left * std_right)

    return (correlation + 1.0) / 2.0


@njit(fastmath=True, cache=True)
def rolling_aesthetic_balance_udf(prices: np.ndarray) -> float:
    """
    美的バランス (Numba JIT)
    """
    if len(prices) < 15:
        return np.nan

    finite_prices = prices[np.isfinite(prices)]
    if len(finite_prices) < 15:
        return np.nan

    gradients = np.diff(finite_prices)

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

    gentle_count = 0
    moderate_count = 0
    intense_count = 0

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

    ideal_gentle = 0.6
    ideal_moderate = 0.3
    ideal_intense = 0.1

    actual_gentle = gentle_count / total_counted
    actual_moderate = moderate_count / total_counted
    actual_intense = intense_count / total_counted

    balance_deviation = (
        abs(actual_gentle - ideal_gentle)
        + abs(actual_moderate - ideal_moderate)
        + abs(actual_intense - ideal_intense)
    ) / 2.0

    aesthetic_balance = 1.0 - balance_deviation
    return max(0.0, aesthetic_balance)


@njit(fastmath=True, cache=True)
def rolling_tonality_udf(prices: np.ndarray) -> float:
    """
    調性 (Numba JIT)
    """
    if len(prices) < 12:
        return np.nan

    finite_prices = prices[np.isfinite(prices)]
    if len(finite_prices) < 12:
        return np.nan

    price_changes = np.diff(finite_prices)

    if len(price_changes) < 5:
        return np.nan

    std_change = np.std(price_changes)
    if std_change <= 1e-10:
        return 0.5

    normalized_changes = price_changes / std_change
    scale_degrees = np.zeros(12)

    for change in normalized_changes:
        degree_idx = int((change + 3.0) / 6.0 * 11.0)
        degree_idx = max(0, min(degree_idx, 11))
        scale_degrees[degree_idx] += 1.0

    major_pattern = np.array([1, 0, 1, 0, 1, 1, 0, 1, 0, 1, 0, 1], dtype=float64)
    minor_pattern = np.array([1, 0, 1, 1, 0, 1, 0, 1, 1, 0, 1, 0], dtype=float64)

    sum_degrees = np.sum(scale_degrees)
    if sum_degrees > 0:
        scale_distribution = scale_degrees / sum_degrees

        major_similarity = np.sum(scale_distribution * major_pattern)
        minor_similarity = np.sum(scale_distribution * minor_pattern)

        total_similarity = major_similarity + minor_similarity
        if total_similarity > 0:
            return major_similarity / total_similarity
        else:
            return 0.5
    else:
        return 0.5


@njit(fastmath=True, cache=True)
def rolling_rhythm_pattern_udf(prices: np.ndarray) -> float:
    """
    リズムパターン (Numba JIT)
    """
    if len(prices) < 20:
        return np.nan

    finite_prices = prices[np.isfinite(prices)]
    if len(finite_prices) < 20:
        return np.nan

    price_changes = np.diff(finite_prices)
    abs_changes = np.abs(price_changes)

    mean_change = np.mean(abs_changes)
    # strong_beats: array of boolean (0 or 1)
    strong_beats = np.zeros(len(abs_changes), dtype=int64)
    for i in range(len(abs_changes)):
        if abs_changes[i] > mean_change:
            strong_beats[i] = 1

    max_pattern_strength = 0.0

    # Search for periodicity
    limit = min(8, len(strong_beats) // 3)
    for period in range(2, limit):
        pattern_score = 0.0
        pattern_count = 0

        for j in range(period, len(strong_beats)):
            if strong_beats[j] == strong_beats[j - period]:
                pattern_score += 1.0
            pattern_count += 1

        if pattern_count > 0:
            strength = pattern_score / pattern_count
            if strength > max_pattern_strength:
                max_pattern_strength = strength

    return max_pattern_strength


@njit(fastmath=True, cache=True)
def rolling_harmony_udf(prices: np.ndarray) -> float:
    """
    和声 (Numba JIT)
    """
    if len(prices) < 30:
        return np.nan

    finite_prices = prices[np.isfinite(prices)]
    if len(finite_prices) < 30:
        return np.nan

    window_n = len(finite_prices)

    short_window = max(3, window_n // 15)
    medium_window = max(5, window_n // 10)
    long_window = max(8, window_n // 6)

    if long_window >= window_n:
        return np.nan

    # Calculate Moving Averages
    # Numba doesn't have pd.rolling, so manual loop
    # Only need MAs where they overlap and are fully calculated.
    # To get trend (diff), we need the last few MAs.

    # Strategy: Calculate MA arrays for the relevant segment to compute trends.
    # We need trends, so we need at least 2 points of MA.
    # Let's calculate MAs for the whole valid range to be safe and simple.

    # Function to calculate MA array
    def calc_ma(data, w):
        n = len(data)
        res = np.zeros(n - w + 1)
        for i in range(n - w + 1):
            res[i] = np.mean(data[i : i + w])
        return res

    short_ma = calc_ma(finite_prices, short_window)
    medium_ma = calc_ma(finite_prices, medium_window)
    long_ma = calc_ma(finite_prices, long_window)

    min_len = min(len(short_ma), min(len(medium_ma), len(long_ma)))
    if min_len < 5:
        return np.nan

    # Align to the end
    short_ma_trim = short_ma[-min_len:]
    medium_ma_trim = medium_ma[-min_len:]
    long_ma_trim = long_ma[-min_len:]

    short_trend = np.diff(short_ma_trim)
    medium_trend = np.diff(medium_ma_trim)
    long_trend = np.diff(long_ma_trim)

    harmony_sum = 0.0
    count = 0

    for j in range(len(short_trend)):
        s1 = np.sign(short_trend[j])
        s2 = np.sign(medium_trend[j])
        s3 = np.sign(long_trend[j])

        # Check distinct signs excluding 0 if possible, or just count consistency
        # Logic from source:
        # signs = [sign1, sign2, sign3]
        # if len(set(signs)) == 1 and signs[0] != 0: score 1.0
        # elif len(set(signs)) == 2: score 0.5
        # else: 0.0

        # Manual set logic
        unique_signs = 0
        # Just compare values
        if s1 == s2 and s2 == s3:
            if s1 != 0:
                harmony_sum += 1.0
            else:
                # All zero? treat as harmony or ignore? Source says "signs[0]!=0"
                pass
        elif (s1 == s2) or (s2 == s3) or (s1 == s3):
            harmony_sum += 0.5
        else:
            # All different (-1, 0, 1)
            pass
        count += 1

    if count > 0:
        return harmony_sum / count
    else:
        return 0.0


@njit(fastmath=True, cache=True)
def rolling_musical_tension_udf(prices: np.ndarray) -> float:
    """
    音楽的緊張度 (Numba JIT)
    """
    if len(prices) < 15:
        return np.nan

    finite_prices = prices[np.isfinite(prices)]
    if len(finite_prices) < 15:
        return np.nan

    price_changes = np.diff(finite_prices)

    if len(price_changes) < 5:
        return np.nan

    local_window = min(5, len(price_changes) // 3)
    tension_sum = 0.0
    count = 0

    for j in range(local_window, len(price_changes) - local_window):
        local_changes = price_changes[j - local_window : j + local_window + 1]

        # Direction dissonance
        sign_changes = 0
        for k in range(len(local_changes) - 1):
            if np.sign(local_changes[k]) != np.sign(local_changes[k + 1]):
                sign_changes += 1

        direction_dissonance = 0.0
        if len(local_changes) > 0:
            direction_dissonance = sign_changes / len(local_changes)

        # Intensity dissonance
        max_volatility = np.max(np.abs(local_changes))
        mean_abs_price = np.mean(np.abs(finite_prices))
        intensity_dissonance = max_volatility / (mean_abs_price + 1e-10)

        total_tension = (direction_dissonance + intensity_dissonance) / 2.0
        if total_tension > 1.0:
            total_tension = 1.0

        tension_sum += total_tension
        count += 1

    if count > 0:
        return tension_sum / count
    else:
        return 0.0


@njit(fastmath=True, cache=True)
def rolling_kinetic_energy_udf(prices: np.ndarray) -> float:
    """
    運動エネルギー (Numba JIT)
    """
    if len(prices) < 10:
        return np.nan

    finite_prices = prices[np.isfinite(prices)]
    if len(finite_prices) < 10:
        return np.nan

    velocities = np.diff(finite_prices)

    if len(velocities) < 2:
        return np.nan

    masses = finite_prices[1:]  # Mass modeled as price itself

    kinetic_energies = 0.5 * masses * velocities**2

    if len(kinetic_energies) > 0:
        mean_kinetic_energy = np.mean(kinetic_energies)
        mean_price = np.mean(finite_prices)
        if mean_price > 1e-10:
            return mean_kinetic_energy / (mean_price**2)
        else:
            return mean_kinetic_energy
    else:
        return 0.0


@njit(fastmath=True, cache=True)
def rolling_muscle_force_udf(prices: np.ndarray) -> float:
    """
    筋力 (Numba JIT)
    """
    if len(prices) < 15:
        return np.nan

    finite_prices = prices[np.isfinite(prices)]
    if len(finite_prices) < 15:
        return np.nan

    velocities = np.diff(finite_prices)
    if len(velocities) < 2:
        return np.nan

    accelerations = np.diff(velocities)
    masses = finite_prices[2:]

    forces = masses * np.abs(accelerations)
    force_directions = np.sign(accelerations)

    # Calculate sustained forces
    sustained_forces_sum = 0.0
    sustained_count = 0

    if len(force_directions) > 0:
        current_direction = force_directions[0]
        current_duration = 1
        current_force_sum = forces[0]

        for j in range(1, len(force_directions)):
            if force_directions[j] == current_direction and current_direction != 0:
                current_duration += 1
                current_force_sum += forces[j]
            else:
                if current_duration > 1:
                    avg_force = current_force_sum / current_duration
                    sustained_forces_sum += avg_force
                    sustained_count += 1

                current_direction = force_directions[j]
                current_duration = 1
                current_force_sum = forces[j]

        # Final check
        if current_duration > 1:
            avg_force = current_force_sum / current_duration
            sustained_forces_sum += avg_force
            sustained_count += 1

    instantaneous_force = np.mean(forces) if len(forces) > 0 else 0.0
    sustained_force = (
        sustained_forces_sum / sustained_count if sustained_count > 0 else 0.0
    )

    muscle_force_score = 0.7 * instantaneous_force + 0.3 * sustained_force

    mean_price = np.mean(finite_prices)
    if mean_price > 1e-10:
        return muscle_force_score / (mean_price**2)
    else:
        return muscle_force_score


@njit(fastmath=True, cache=True)
def rolling_biomechanical_efficiency_udf(prices: np.ndarray) -> float:
    """
    生体力学効率 (Numba JIT)
    """
    if len(prices) < 20:
        return np.nan

    finite_prices = prices[np.isfinite(prices)]
    if len(finite_prices) < 20:
        return np.nan

    price_changes = np.diff(finite_prices)
    total_displacement = np.sum(np.abs(price_changes))

    velocities = price_changes
    # Acceleration requires diff of velocities
    if len(velocities) > 1:
        accelerations = np.diff(velocities)
    else:
        accelerations = np.zeros(1)  # Dummy

    kinetic_energy = np.sum(velocities**2)
    acceleration_energy = np.sum(accelerations**2) if len(accelerations) > 0 else 0.0
    total_energy = kinetic_energy + acceleration_energy

    if total_energy > 1e-10 and total_displacement > 1e-10:
        raw_efficiency = total_displacement / total_energy

        reference_efficiency = total_displacement / (np.sum(np.abs(velocities)) + 1e-10)
        normalized_efficiency = raw_efficiency / (reference_efficiency + 1e-10)
        if normalized_efficiency > 1.0:
            return 1.0
        else:
            return normalized_efficiency
    else:
        return 0.0


@njit(fastmath=True, cache=True)
def rolling_energy_expenditure_udf(prices: np.ndarray) -> float:
    """
    エネルギー消費量 (Numba JIT)
    """
    if len(prices) < 15:
        return np.nan

    finite_prices = prices[np.isfinite(prices)]
    if len(finite_prices) < 15:
        return np.nan

    baseline_energy = np.var(finite_prices)

    price_changes = np.diff(finite_prices)
    movement_energy = np.sum(price_changes**2)

    acceleration_energy = 0.0
    if len(price_changes) > 1:
        accelerations = np.diff(price_changes)
        acceleration_energy = np.sum(accelerations**2)

    nonlinearity_energy = 0.0
    n_points = len(finite_prices)
    if n_points >= 3:
        # Linear regression to find residuals
        x = np.arange(n_points, dtype=float64)

        sum_x = np.sum(x)
        sum_y = np.sum(finite_prices)
        sum_xy = np.sum(x * finite_prices)
        sum_x2 = np.sum(x * x)

        denom = n_points * sum_x2 - sum_x**2
        if denom != 0:
            slope = (n_points * sum_xy - sum_x * sum_y) / denom
            intercept = (sum_y - slope * sum_x) / n_points

            linear_trend = intercept + slope * x
            nonlinearity_energy = np.sum((finite_prices - linear_trend) ** 2)

    total_energy = (
        baseline_energy + movement_energy + acceleration_energy + nonlinearity_energy
    )

    mean_price = np.mean(finite_prices)
    if mean_price > 1e-10:
        return total_energy / (mean_price**2)
    else:
        return total_energy


# ==================================================================
#
# 1. NUMBA UDF ライブラリ (Part 5: Engine 2A - Complexity Theory)
#    Strict Port from: engine_2_A_complexity_theory_F05_F15.py
#
# ==================================================================

# ----------------------------------------
# from engine_2_A_complexity_theory_F05_F15.py
# ----------------------------------------


@njit(fastmath=True, cache=True)
def polynomial_fit_detrend(y: np.ndarray, degree: int = 1) -> np.ndarray:
    """
    多項式フィッティングとトレンド除去 (Numba JIT)
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
    単一ウィンドウのMFDFA計算 (ヘルパー関数)
    学習用ロジックを完全再現 + 異常値ガード
    Returns: [h_mean, width, h_max]
    """
    n = len(prices)
    n_q = len(q_values)
    n_scales = len(scales)

    # 初期化 [h_mean, width, h_max]
    result = np.full(3, np.nan)

    # データ長不足チェック
    if n < 20:
        return result

    # 【異常値ガード】分散ゼロチェック (定数値の場合、計算不能なため 0.5 を返す)
    if np.std(prices) < 1e-10:
        result[:] = 0.5
        result[1] = 0.0  # width
        return result

    # 1. プロファイル構築(累積和)
    mean_price = np.mean(prices)
    profile = np.zeros(n)
    cumsum = 0.0
    for i in range(n):
        cumsum += prices[i] - mean_price
        profile[i] = cumsum

    # 2. qモーメント揺らぎ関数の計算
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

            variance = 0.0
            for val in detrended:
                variance += val * val
            variance = variance / scale if scale > 0 else 0.0
            segment_variances[seg] = variance

        for q_idx in range(n_q):
            q = q_values[q_idx]

            if abs(q) < 1e-10:  # q=0の場合は対数平均
                log_sum = 0.0
                valid_count = 0
                for var in segment_variances:
                    if var > 1e-10:
                        log_sum += np.log(var)
                        valid_count += 1
                if valid_count > 0:
                    F_q[q_idx, s_idx] = np.exp(log_sum / (2.0 * valid_count))
            else:
                sum_val = 0.0
                valid_count = 0
                for var in segment_variances:
                    if var > 1e-10:
                        sum_val += np.power(var, q / 2.0)
                        valid_count += 1
                if valid_count > 0:
                    F_q[q_idx, s_idx] = np.power(sum_val / valid_count, 1.0 / q)

    # 3. 一般化Hurst指数の推定
    h_q_values = np.zeros(n_q)

    for q_idx in range(n_q):
        log_scales = []
        log_F_vals = []

        for s_idx in range(n_scales):
            if F_q[q_idx, s_idx] > 1e-10:
                log_scales.append(np.log(scales[s_idx]))
                log_F_vals.append(np.log(F_q[q_idx, s_idx]))

        if len(log_scales) >= 3:
            x_arr = np.array(log_scales)
            y_arr = np.array(log_F_vals)
            x_mean = np.mean(x_arr)
            y_mean = np.mean(y_arr)

            numerator = 0.0
            denominator = 0.0
            for i in range(len(x_arr)):
                x_diff = x_arr[i] - x_mean
                numerator += x_diff * (y_arr[i] - y_mean)
                denominator += x_diff * x_diff

            if abs(denominator) > 1e-10:
                h_q_values[q_idx] = numerator / denominator
            else:
                h_q_values[q_idx] = np.nan
        else:
            h_q_values[q_idx] = np.nan

    # 4. マルチフラクタルスペクトラムの計算
    valid_h = h_q_values[np.isfinite(h_q_values)]

    if len(valid_h) >= 3:
        h_mean = np.mean(valid_h)
        h_max = np.max(valid_h)
        h_min = np.min(valid_h)
        result[0] = h_mean
        result[1] = h_max - h_min  # width
        result[2] = h_max

    return result


@njit(fastmath=True, cache=True)
def mfdfa_rolling_udf(prices: np.ndarray, component_idx: int) -> float:
    """
    MFDFA (Numba JIT) - リアルタイム単一ウィンドウ版
    Args:
        prices: 最新のウィンドウデータ (1D array)
        component_idx: 0=hurst_mean, 1=width, 2=holder_max
    """
    # 学習用スクリプトと同一パラメータ
    q_values = np.array([-5.0, -3.0, -1.0, 0.0, 1.0, 3.0, 5.0])
    scales = np.array([10.0, 20.0, 50.0, 100.0, 200.0])

    mfdfa_result = mfdfa_core_single_window(prices, q_values, scales, poly_degree=1)
    val = mfdfa_result[component_idx]

    if np.isnan(val):
        if component_idx == 1:
            return 0.0  # Width
        return 0.5  # Mean/Max defaults to 0.5

    return val


@njit(fastmath=True, cache=True)
def binarize_series(values: np.ndarray, method: int = 0) -> np.ndarray:
    """時系列のバイナリ化/多値符号化"""
    n = len(values)
    encoded = np.zeros(n, dtype=np.int32)
    if n < 2:
        return encoded

    if method == 0:  # 中央値基準
        median_val = np.median(values)
        for i in range(n):
            encoded[i] = 1 if values[i] > median_val else 0
    elif method == 1:  # 分位基準
        valid_vals = values[np.isfinite(values)]
        if len(valid_vals) < 3:
            return encoded
        q33 = np.percentile(valid_vals, 33.33)
        q67 = np.percentile(valid_vals, 66.67)
        for i in range(n):
            if values[i] < q33:
                encoded[i] = 0
            elif values[i] < q67:
                encoded[i] = 1
            else:
                encoded[i] = 2
    elif method == 2:  # 変化基準
        for i in range(1, n):
            if values[i] > values[i - 1]:
                encoded[i] = 1
            elif values[i] < values[i - 1]:
                encoded[i] = -1
            else:
                encoded[i] = 0
    return encoded


@njit(fastmath=True, cache=True)
def lempel_ziv_complexity(sequence: np.ndarray) -> float:
    """Lempel-Ziv複雑性計算(LZ76アルゴリズム)"""
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
            i += max_match_length + 1

    if n > 1:
        max_complexity = n / (np.log2(n) + 1e-10)
        return min(complexity / max_complexity, 1.0) if max_complexity > 0 else 0.0
    return 0.0


@njit(fastmath=True, cache=True)
def kolmogorov_complexity_single_window(prices: np.ndarray) -> np.ndarray:
    """
    単一ウィンドウのコルモゴロフ複雑性計算
    Returns: [complexity, compression_ratio, pattern_diversity]
    """
    result = np.full(3, np.nan)
    n = len(prices)
    if n < 10:
        return result

    # 1. 対数リターン
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
        result[:] = 0.0
        result[1] = 1.0
        return result

    standardized = np.zeros(len(returns))
    for i in range(len(returns)):
        standardized[i] = (returns[i] - returns_mean) / returns_std

    # 3. バイナリ化 & LZ複雑性
    encoded = binarize_series(standardized, method=0)
    complexity = lempel_ziv_complexity(encoded)

    # 4. パターン多様性
    unique_count = 0
    # Numba compatible unique count for small integer array
    for i in range(len(encoded)):
        is_unique = True
        for j in range(i):
            if encoded[i] == encoded[j]:
                is_unique = False
                break
        if is_unique:
            unique_count += 1

    pattern_diversity = unique_count / len(encoded) if len(encoded) > 0 else 0.0

    result[0] = complexity
    result[1] = 1.0 - complexity
    result[2] = pattern_diversity
    return result


@njit(fastmath=True, cache=True)
def kolmogorov_complexity_rolling_udf(prices: np.ndarray, component_idx: int) -> float:
    """
    コルモゴロフ複雑性 (Numba JIT) - リアルタイム単一ウィンドウ版
    Args:
        prices: 最新のウィンドウデータ
        component_idx: 0=complexity, 1=compression_ratio, 2=pattern_diversity
    """
    kc_result = kolmogorov_complexity_single_window(prices)
    val = kc_result[component_idx]

    if np.isnan(val):
        if component_idx == 1:
            return 1.0
        return 0.0
    return val


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


# ==========================================
# [追加] 内部ヘルパー関数のNumba化 (高速化用)
# ==========================================


@njit(fastmath=True, cache=True)
def calc_pct_change_numba(arr: np.ndarray) -> np.ndarray:
    """
    Pct Change配列全体を計算 (Numba JIT)
    _calculate_base_features 内の _pct を代替
    """
    n = len(arr)
    # 先頭をNaNにするため、サイズnで初期化
    res = np.full(n, np.nan, dtype=np.float64)

    if n < 2:
        return res

    for i in range(1, n):
        prev = arr[i - 1]
        if prev == 0:
            prev = 1e-10  # ゼロ除算防止 (元のロジックと一致)
        res[i] = (arr[i] - prev) / prev

    return res


@njit(fastmath=True, cache=True)
def calc_ema_array_numba(arr: np.ndarray, span: int) -> np.ndarray:
    """
    EMA配列全体を計算 (Numba JIT)
    _calculate_base_features 内の _ema を代替
    """
    n = len(arr)
    ema = np.zeros(n, dtype=np.float64)

    if n == 0:
        return ema

    alpha = 2.0 / (span + 1.0)

    # 初期値
    ema[0] = arr[0]

    # 漸化式計算
    for i in range(1, n):
        ema[i] = alpha * arr[i] + (1.0 - alpha) * ema[i - 1]

    return ema


@njit(fastmath=True, cache=True)
def calc_rolling_mean_array_numba(arr: np.ndarray, window: int) -> np.ndarray:
    """
    Rolling Mean配列全体を計算 (Numba JIT)
    _calculate_base_features 内の _rolling_mean を代替
    """
    n = len(arr)
    res = np.full(n, np.nan, dtype=np.float64)

    if n < window or window <= 0:
        return res

    # 最初のウィンドウ
    current_sum = 0.0
    for i in range(window):
        current_sum += arr[i]

    res[window - 1] = current_sum / window

    # スライド計算
    for i in range(window, n):
        current_sum = current_sum - arr[i - window] + arr[i]
        res[i] = current_sum / window

    return res


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
    # [V5.2 修正] mt5.TIMEFRAME... の定数を削除し、Linux依存を解消
    ALL_TIMEFRAMES = {
        "M1": 1,  # 値は何でも良い（None以外）
        "M3": 3,
        "M5": 5,
        "M8": 8,
        "M15": 15,
        "M30": 30,
        "H1": 60,
        "H4": 240,
        "H6": 360,
        "H12": 720,
        "D1": 1440,
        "W1": 10080,
        "MN": 43200,
        "tick": None,  # スキップ対象
        "M0.5": None,  # スキップ対象
    }

    # Pandas.resample() のためのルール定義
    TF_RESAMPLE_RULES = {
        "M3": "3min",
        "M5": "5min",
        "M8": "8min",
        "M15": "15min",
        "M30": "30min",
        "H1": "1h",
        "H4": "4h",
        "H6": "6h",
        "H12": "12h",
        "D1": "1D",  # (変更なし)
        "W1": "1W",  # (変更なし)
        "MN": "1MS",  # (変更なし)
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
            # [修正後] 'market_proxy' (x変数) のDequeも初期化する
            # 5a. Dequeを初期化 (y変数)
            self.proxy_feature_buffers[tf_name] = {
                feat: deque(maxlen=lookback) for feat in PROXY_FEATURES
            }
            # 5a. (続き) market_proxy (x変数) のDequeも追加
            self.proxy_feature_buffers[tf_name]["market_proxy"] = deque(maxlen=lookback)
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

        # ★★★ [追加] 起動時にJITコンパイルを済ませる ★★★
        self._warmup_jit()

    def _warmup_jit(self):
        """
        [V12.0 新規] JITコンパイルのウォームアップ
        初回実行時の遅延を防ぐため、ダミーデータで重い関数を空回しする。
        """
        self.logger.info("Numba JIT関数のウォームアップを開始します...")
        try:
            # ダミーデータ (長さ1000のランダムウォーク)
            dummy_data = np.cumsum(np.random.randn(1000)).astype(np.float64)

            # 重い関数を一度呼び出す (コンパイルが走る)
            # 引数は (data, component_idx)
            _ = mfdfa_rolling_udf(dummy_data, 0)
            _ = kolmogorov_complexity_rolling_udf(dummy_data, 0)

            self.logger.info(
                "✓ JITウォームアップ完了。リアルタイム動作に支障はありません。"
            )
        except Exception as e:
            self.logger.warning(f"JITウォームアップ中に警告 (無視可): {e}")

    def _parse_feature_list_and_get_lookbacks(
        self, feature_list: List[str]
    ) -> Dict[str, int]:
        """
        [新規] (矛盾②解決)
        `final_feature_set_v2.txt` を解析し、時間足ごとに
        必要な最大ルックバック期間（Numpyバッファの長さ）を決定する。

        [V6.6 修正]
        _calculate_base_features 内でハードコードされた特徴量計算（KST, Hilbert等）が
        名簿に含まれない場合でも実行されるため、計算に必要な最低限の長さを
        全時間足に対して保証する (Safe Margin)。
        """
        self.logger.info("特徴量名簿を解析し、時間足ごとの最大ルックバックを計算中...")

        # (e.g., "M1", "M3", ..., "D1", "MN", "tick")
        tf_pattern = re.compile(r"_(M[0-9\.]+|H[0-9]+|D[0-9]+|W[0-9]+|MN|tick)$")

        lookbacks: Dict[str, int] = {}

        # 1. 名簿に含まれる時間足を特定
        for feature_name in feature_list:
            tf_match = tf_pattern.search(feature_name)
            if not tf_match:
                continue

            tf_name = tf_match.group(1)

            # 初期化 (まだ値は決定しない)
            if tf_name not in lookbacks:
                lookbacks[tf_name] = 0

            # MFDFA/Kolmogorovなど、極端に長いウィンドウが必要なケースのみ個別チェック
            if "e2a_mfdfa" in feature_name or "e2a_kolmogorov" in feature_name:
                if "5000" in feature_name:
                    lookbacks[tf_name] = max(lookbacks[tf_name], 5000)
                elif "2500" in feature_name:
                    lookbacks[tf_name] = max(lookbacks[tf_name], 2500)
                elif "1500" in feature_name:
                    lookbacks[tf_name] = max(lookbacks[tf_name], 1500)

        # 2. ベース特徴量計算に必要な安全マージンを適用
        # _calculate_base_features では、名簿の有無にかかわらず
        # WMA(200), Spectral(512), Hilbert(200), KST(30) などを計算するため、
        # バッファには最低でもこれらをカバーするサイズが必須となる。
        # MFDFA(1000) も考慮し、デフォルトで1000を確保する。
        SAFE_MIN_LOOKBACK = 1000

        final_lookbacks = {}
        for tf_name in lookbacks.keys():
            # 名簿由来のサイズ(MFDFA等) と 安全マージン の大きい方採用
            req_size = max(lookbacks[tf_name], SAFE_MIN_LOOKBACK)

            # さらに予備を追加
            final_lookbacks[tf_name] = req_size + 100

            self.logger.info(
                f"  -> {tf_name} 最大ルックバック: {final_lookbacks[tf_name]} (SafeMargin適用)"
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

        # [修正] データが少しでもあれば計算許可を出す (MN不足対応)
        if len(df_slice) > 0:
            self.is_buffer_filled[tf_name] = True
            if len(df_slice) < buffer_len:
                self.logger.warning(
                    f"  ⚠️ {tf_name} はデータ不足 ({len(df_slice)}/{buffer_len}) ですが、計算を許可します。"
                )

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
        history_data_map: Dict[str, pd.DataFrame],  # 修正
        market_proxy_cache: pd.DataFrame,
    ) -> None:
        """
        [V8.0 修正]
        1. M1データのみを history_data_map から受け取る
        2. M1バッファを充填
        3. M1データから M3～MN のすべてをリサンプリングして充填する
        """
        self.logger.info(
            "全時間足の履歴データでNumpyバッファを一括充填中 (V8.0: M1 Only)..."
        )

        if "M1" not in history_data_map:
            raise ValueError("履歴データに M1 がありません。リサンプリングできません。")

        # 1. M1データをPandas DFに変換 (リサンプリング元として保持)
        m1_history_pd = history_data_map["M1"]
        if "timestamp" not in m1_history_pd.columns:
            raise ValueError("M1履歴データに 'timestamp' カラムが見つかりません。")
        m1_history_pd = m1_history_pd.set_index("timestamp")

        # 2. M1バッファを充填
        self.logger.info(f"  -> M1 バッファをMT5データから充填中...")
        self._replace_buffer_from_dataframe("M1", m1_history_pd, market_proxy_cache)

        # 3. M1 Dequeを更新 (リアルタイムループ用)
        self.m1_dataframe.clear()
        m1_records = m1_history_pd.reset_index().to_dict("records")
        self.m1_dataframe.extend(m1_records)

        # 4. M1データから「M1以外」の全バッファをリサンプリング
        self.logger.info(
            "M1データから M3～MN の全バッファをリサンプリングして充填中..."
        )
        for tf_name, rule in self.TF_RESAMPLE_RULES.items():
            if tf_name not in self.data_buffers:
                continue  # (e.g., 'M30' が名簿になければスキップ)

            # (M1は既に充填済みなのでスキップ)
            if tf_name == "M1":
                continue

            # ★ M3, M5, M8, H1, H4, H6, H12, D1, W1, MN がここでリサンプリングされる
            try:
                self.logger.info(f"  -> {tf_name} をM1からリサンプリング中...")
                resampled_df = (
                    m1_history_pd.resample(rule)
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

                if resampled_df.empty:
                    self.logger.warning(f"{tf_name} のリサンプリング結果が空です。")
                    continue

                # バッファを置換
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
            # [修正] データが入ってきた時点で常にTrueにする
            if not self.is_buffer_filled[tf_name]:
                self.is_buffer_filled[tf_name] = True
                self.logger.info(f"✅ {tf_name} バッファ計算開始 (Best-Effort)。")

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
        # if not self.is_buffer_filled[tf_name]:
        #     return {"is_r4": False, "reason": "buffer_not_filled"}

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
                # [修正] 該当する時間足かチェック
                if feature_name_in_list.endswith(f"_{tf_name}"):
                    # 該当する場合: 計算済みの値をセット

                    # --- ▼▼▼ [V11.1 堅牢化] ベース名抽出ロジックの修正 ▼▼▼ ---
                    if "_neutralized_" in feature_name_in_list:
                        # 純化済み特徴量の場合: "feat_neutralized_M15" -> "feat"
                        base_name = feature_name_in_list.split("_neutralized_")[0]
                    else:
                        # 生の特徴量の場合 (フォールバック): "feat_M15" -> "feat"
                        # 右側から最初の "_" で分割してサフィックスを除去
                        base_name = feature_name_in_list.rsplit("_", 1)[0]
                    # --- ▲▲▲ 修正ここまで ▲▲▲ ---

                    if base_name in neutralized_features:
                        feature_vector.append(neutralized_features[base_name])
                    else:
                        # 計算マップになかった場合
                        # self.logger.warning(f"特徴量 {base_name} が見つかりません。0.0 を使用します。")
                        feature_vector.append(0.0)
                else:
                    # [修正] 該当しない時間足の場合:
                    # AIの入力次元数を保つために 0.0 で埋める (Padding)
                    feature_vector.append(0.0)

            # ベクトルが空、または全て0の場合のチェックは必要だが、
            # 304個埋めているので empty チェックは不要になる

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

    def calculate_dynamic_context(self, hmm_model: Any) -> Dict[str, float]:
        """
        [V12.6 最終修正版]
        データの順序や時刻のズレを許容するフルスキャン方式に変更。
        詳細なデバッグログを出力し、かつ分散ゼロ時の異常値（1.584）を物理的に防ぐ。
        """
        tf_name = "D1"
        self.logger.info(
            f"DEBUG: D1 Buffer Size={len(self.data_buffers['D1']['close'])}, LastVal={self.data_buffers['D1']['close'][-1] if self.data_buffers['D1']['close'] else 'None'}"
        )
        if (tf_name not in self.data_buffers) or ("M1" not in self.data_buffers):
            return {}

        # M1バッファ（最新データ）への参照
        buffer_m1 = self.data_buffers["M1"]
        if not buffer_m1["close"]:
            return {}

        # =========================================================
        # 1. 基準時刻の決定 (ロバスト化)
        # =========================================================
        # 最新データの時刻を取得
        try:
            last_m1_ts_raw = self.m1_dataframe[-1]["timestamp"]
            last_m1_ts = pd.Timestamp(last_m1_ts_raw)
            if last_m1_ts.tzinfo is None:
                last_m1_ts = last_m1_ts.tz_localize("UTC")
            else:
                last_m1_ts = last_m1_ts.tz_convert("UTC")

            # その日の開始時刻 (00:00 UTC)
            start_of_day = last_m1_ts.floor("D")
        except Exception as e:
            self.logger.error(f"基準時刻の計算に失敗: {e}")
            return {}

        # =========================================================
        # 2. D1バッファの重複削除
        # =========================================================
        buffer_d1 = self.data_buffers[tf_name]
        if not buffer_d1["close"]:
            return {}

        d1_close = np.array(buffer_d1["close"], dtype=np.float64)
        d1_high = np.array(buffer_d1["high"], dtype=np.float64)
        d1_low = np.array(buffer_d1["low"], dtype=np.float64)

        last_d1_ts_raw = self.last_bar_timestamps.get(tf_name)
        if last_d1_ts_raw is not None:
            ts_check = pd.Timestamp(last_d1_ts_raw)
            if ts_check.tzinfo is None:
                ts_check = ts_check.tz_localize("UTC")
            else:
                ts_check = ts_check.tz_convert("UTC")

            if ts_check >= start_of_day:
                d1_close = d1_close[:-1]
                d1_high = d1_high[:-1]
                d1_low = d1_low[:-1]

        # =========================================================
        # 3. 当日データの収集 (フルスキャン & ログ出力)
        # =========================================================
        intraday_bars = []
        scan_limit = 5000  # 少し多めに
        scanned_count = 0

        # [DEBUG] 基準時刻をログに出す（次のログ確認で重要になります）
        # self.logger.info(f"ContextCalc: StartOfDay={start_of_day}, LastM1={last_m1_ts}")

        # breakを使わず、指定件数を確実にスキャンする (順序不整合対策)
        for bar in reversed(self.m1_dataframe):
            scanned_count += 1
            if scanned_count > scan_limit:
                break

            raw_ts = bar["timestamp"]
            try:
                bar_ts = pd.Timestamp(raw_ts)
                if bar_ts.tzinfo is None:
                    bar_ts = bar_ts.tz_localize("UTC")
                else:
                    bar_ts = bar_ts.tz_convert("UTC")
            except:
                continue

            # 基準時刻以降なら採用
            if bar_ts >= start_of_day:
                intraday_bars.append(bar)

        # ▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼
        # 【ここに挿入】
        last_m1_ts_debug = (
            self.m1_dataframe[-1]["timestamp"] if self.m1_dataframe else "Empty"
        )
        self.logger.info(
            f"DEBUG: 当日データ件数={len(intraday_bars)}, 基準時刻(UTC)={start_of_day}, 最新M1時刻={last_m1_ts_debug}"
        )
        # ▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲

        # =========================================================
        # 4. 合成と計算
        # =========================================================
        if intraday_bars:
            current_close = intraday_bars[0]["close"]
            # High/Lowをしっかりスキャン
            max_h = -1.0
            min_l = 1.0e15
            for b in intraday_bars:
                if b["high"] > max_h:
                    max_h = b["high"]
                if b["low"] < min_l:
                    min_l = b["low"]
            current_high = max_h
            current_low = min_l
        else:
            # データなしフォールバック
            price = buffer_m1["close"][-1]
            current_high = current_low = current_close = price

        # 配列結合
        close_arr = np.append(d1_close, current_close)
        high_arr = np.append(d1_high, current_high)
        low_arr = np.append(d1_low, current_low)

        if len(close_arr) < 50:
            return {}

        context = {}
        try:
            # ATR
            atr_arr = calculate_atr_numba(high_arr, low_arr, close_arr, 21)
            context["atr"] = atr_arr[-1] if not np.isnan(atr_arr[-1]) else 0.0

            # 異常値防止のための分散チェック
            # 全く動いていない(分散ほぼ0)場合は計算をスキップして安全な値を返す
            is_flat = np.std(close_arr[-100:]) < 1e-9

            # HMM
            if not is_flat and hmm_model is not None:
                full_log_ret = np.diff(np.log(close_arr + 1e-10))
                # 以下省略なし
                calc_len = min(len(full_log_ret), 100)
                recent_ret = full_log_ret[-calc_len:]
                recent_atr = atr_arr[1:][-calc_len:]
                min_len = min(len(recent_ret), len(recent_atr))

                # NaN除外
                valid_idx = np.isfinite(recent_ret[-min_len:]) & np.isfinite(
                    recent_atr[-min_len:]
                )
                if np.sum(valid_idx) > 10:
                    X = np.column_stack(
                        (
                            recent_ret[-min_len:][valid_idx],
                            recent_atr[-min_len:][valid_idx],
                        )
                    )
                    try:
                        probs = hmm_model.predict_proba(X)
                        context["hmm_prob_0"] = probs[-1][0]
                        context["hmm_prob_1"] = probs[-1][1]
                    except:
                        context["hmm_prob_0"] = 0.5
                        context["hmm_prob_1"] = 0.5
                else:
                    context["hmm_prob_0"] = 0.5
                    context["hmm_prob_1"] = 0.5
            else:
                context["hmm_prob_0"] = 0.5
                context["hmm_prob_1"] = 0.5

            # 統計量
            context["e1a_statistical_kurtosis_50"] = statistical_kurtosis_numba(
                close_arr[-50:]
            )

            adx_arr = calculate_adx_numba(high_arr, low_arr, close_arr, 21)
            context["e1c_adx_21"] = adx_arr[-1] if not np.isnan(adx_arr[-1]) else 0.0

            # MFDFA (修正: 次元ズレ修正 + ガード解除)
            window_size = min(len(close_arr), 1000)
            if not is_flat and window_size >= 200:
                # 1. 価格を「対数収益率」に変換 (これでHurstが 1.5 -> 0.5 の次元に戻る)
                target_data = close_arr[-window_size:]
                safe_target = np.where(target_data <= 0, 1e-10, target_data)
                log_returns = np.diff(np.log(safe_target))
                log_returns = np.nan_to_num(
                    log_returns, nan=0.0, posinf=0.0, neginf=0.0
                )

                # 2. 計算実行
                val = mfdfa_rolling_udf(log_returns, 0)

                # 3. [修正] ガードを無効化し、生の数値を採用する
                # if 1.58 <= val <= 1.59: <-- 削除
                #     val = 0.5

                # 4. 数学的エラー(NaN/Inf)だけは防ぐ
                if np.isnan(val) or np.isinf(val):
                    val = 0.5

                context["e2a_mfdfa_hurst_mean_1000"] = val
            else:
                context["e2a_mfdfa_hurst_mean_1000"] = 0.5

            # Complexity
            window_size_kc = min(len(close_arr), 1000)
            if not is_flat and window_size_kc >= 50:
                context["e2a_kolmogorov_complexity_1000"] = (
                    kolmogorov_complexity_rolling_udf(close_arr[-window_size_kc:], 0)
                )
            else:
                context["e2a_kolmogorov_complexity_1000"] = 0.0

            # NaNチェック
            for k, v in context.items():
                if np.isnan(v) or np.isinf(v):
                    context[k] = 0.0

            return context

        except Exception as e:
            self.logger.error(f"Context calc error: {e}")
            return {}

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
            if window <= 0:
                return np.array([], dtype=arr.dtype)
            if window > len(arr):
                return arr
            return arr[-window:]

        def _array(arr: np.ndarray) -> np.ndarray:
            """配列全体をそのまま使用"""
            return arr

        def _last(arr: np.ndarray) -> float:
            """UDFが返した配列の最新値（末尾）を取得"""
            if len(arr) == 0:
                return np.nan
            return arr[-1]

        # --- 共通データの事前計算 ---
        close_pct = calc_pct_change_numba(data["close"])

        # --- Engine 1A 特徴量 (engine_1_A_...py) ---
        # 厳密な移植版: 配列スライス(_window)をNumba関数に渡す

        # 基本統計量
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

        # 統計検定 (厳密実装版を使用)
        features["e1a_jarque_bera_statistic_50"] = jarque_bera_statistic_numba(
            _window(data["close"], 50)
        )
        features["e1a_anderson_darling_statistic_30"] = anderson_darling_numba(
            _window(data["close"], 30)
        )
        features["e1a_runs_test_statistic_30"] = runs_test_numba(
            _window(data["close"], 30)
        )
        features["e1a_von_neumann_ratio_30"] = von_neumann_ratio_numba(
            _window(data["close"], 30)
        )

        # 品質保証 (QA)
        # _last() を使用して配列から単一の値を取り出す
        features["e1a_fast_basic_stabilization"] = _last(
            basic_stabilization_numba(_window(data["close"], 100))
        )
        features["e1a_fast_robust_stabilization"] = _last(
            robust_stabilization_numba(_window(data["close"], 100))
        )

        # ロバスト統計 (四分位範囲等)
        q75_10, q25_10 = np.percentile(_window(data["close"], 10), [75, 25])
        features["e1a_robust_iqr_10"] = q75_10 - q25_10
        q75_20, q25_20 = np.percentile(_window(data["close"], 20), [75, 25])
        features["e1a_robust_iqr_20"] = q75_20 - q25_20
        q75_50, q25_50 = np.percentile(_window(data["close"], 50), [75, 25])
        features["e1a_robust_iqr_50"] = q75_50 - q25_50

        features["e1a_robust_median_50"] = np.median(_window(data["close"], 50))
        features["e1a_robust_q75_50"] = q75_50

        # 高度ロバスト統計 (新規実装)
        features["e1a_robust_mad_20"] = mad_rolling_numba(_window(data["close"], 20))
        features["e1a_robust_biweight_location_20"] = biweight_location_numba(
            _window(data["close"], 20)
        )
        features["e1a_robust_winsorized_mean_20"] = winsorized_mean_numba(
            _window(data["close"], 20)
        )
        # Trimmed Mean (10% cut) - Numpyで直接計算
        # scipy.stats.trim_mean 相当
        close_50 = _window(data["close"], 50)
        n_trim = int(len(close_50) * 0.1)
        if len(close_50) > 2 * n_trim:
            sorted_close = np.sort(close_50)
            trimmed = sorted_close[n_trim:-n_trim] if n_trim > 0 else sorted_close
            features["e1a_robust_trimmed_mean_50"] = np.mean(trimmed)
        else:
            features["e1a_robust_trimmed_mean_50"] = np.mean(close_50)

        # 変動係数 (CV)
        mean_10 = np.mean(_window(data["close"], 10))
        mean_20 = np.mean(_window(data["close"], 20))
        mean_50 = np.mean(_window(data["close"], 50))
        std_10 = np.std(_window(data["close"], 10))
        std_20 = np.std(_window(data["close"], 20))
        std_50 = np.std(_window(data["close"], 50))

        features["e1a_statistical_cv_10"] = std_10 / (mean_10 + 1e-10)
        features["e1a_statistical_cv_20"] = std_20 / (mean_20 + 1e-10)
        features["e1a_statistical_cv_50"] = std_50 / (mean_50 + 1e-10)

        # 歪度・尖度 (Bias-Corrected実装を使用)
        features["e1a_statistical_skewness_20"] = rolling_skew_numba(
            _window(data["close"], 20)
        )
        features["e1a_statistical_skewness_50"] = rolling_skew_numba(
            _window(data["close"], 50)
        )
        features["e1a_statistical_kurtosis_20"] = rolling_kurtosis_numba(
            _window(data["close"], 20)
        )
        features["e1a_statistical_kurtosis_50"] = rolling_kurtosis_numba(
            _window(data["close"], 50)
        )

        # 高次モーメント (5-8)
        # Sourceロジック: ((x - mean)/std)^moment の平均
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

        features["e1a_statistical_variance_10"] = np.var(_window(data["close"], 10))

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
        # ATR (SMAベース, UDF呼び出し)
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

        # RSI (SMAベース)
        rsi_14_arr = calculate_rsi_numba(_array(data["close"]), 14)
        rsi_21_arr = calculate_rsi_numba(_array(data["close"]), 21)
        rsi_30_arr = calculate_rsi_numba(_array(data["close"]), 30)
        rsi_50_arr = calculate_rsi_numba(_array(data["close"]), 50)

        # Stochastic (修正: パラメータトリックで %K, %D を抽出)
        # calculate_stochastic_numba(..., k, d, slow) -> returns Slow %D

        # %K (d=1, slow=1 で呼び出すと K が返る)
        stoch_k_14_arr = calculate_stochastic_numba(
            _array(data["high"]), _array(data["low"]), _array(data["close"]), 14, 1, 1
        )
        stoch_k_21_arr = calculate_stochastic_numba(
            _array(data["high"]), _array(data["low"]), _array(data["close"]), 21, 1, 1
        )
        stoch_k_9_arr = calculate_stochastic_numba(
            _array(data["high"]), _array(data["low"]), _array(data["close"]), 9, 1, 1
        )

        # %D (d=N, slow=1 で呼び出すと D が返る)
        # ソースに合わせて D期間=3, 5 を指定
        stoch_d_14_3_arr = calculate_stochastic_numba(
            _array(data["high"]), _array(data["low"]), _array(data["close"]), 14, 3, 1
        )
        stoch_d_21_5_arr = calculate_stochastic_numba(
            _array(data["high"]), _array(data["low"]), _array(data["close"]), 21, 5, 1
        )
        stoch_d_9_3_arr = calculate_stochastic_numba(
            _array(data["high"]), _array(data["low"]), _array(data["close"]), 9, 3, 1
        )

        # Slow %D (d=N, slow=M で呼び出す)
        stoch_slow_d_14_3_3_arr = calculate_stochastic_numba(
            _array(data["high"]), _array(data["low"]), _array(data["close"]), 14, 3, 3
        )
        stoch_slow_d_21_5_5_arr = calculate_stochastic_numba(
            _array(data["high"]), _array(data["low"]), _array(data["close"]), 21, 5, 5
        )
        stoch_slow_d_9_3_3_arr = calculate_stochastic_numba(
            _array(data["high"]), _array(data["low"]), _array(data["close"]), 9, 3, 3
        )

        # DI+, DI-
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

        # Aroon
        aroon_up_14_arr = calculate_aroon_up_numba(_array(data["high"]), 14)
        aroon_down_14_arr = calculate_aroon_down_numba(_array(data["low"]), 14)

        # Williams %R
        williams_r_14_arr = calculate_williams_r_numba(
            _array(data["high"]), _array(data["low"]), _array(data["close"]), 14
        )
        williams_r_28_arr = calculate_williams_r_numba(
            _array(data["high"]), _array(data["low"]), _array(data["close"]), 28
        )
        williams_r_56_arr = calculate_williams_r_numba(
            _array(data["high"]), _array(data["low"]), _array(data["close"]), 56
        )

        # TRIX
        trix_14_arr = calculate_trix_numba(_array(data["close"]), 14)
        trix_20_arr = calculate_trix_numba(_array(data["close"]), 20)
        trix_30_arr = calculate_trix_numba(_array(data["close"]), 30)

        # TSI
        tsi_13_arr = calculate_tsi_numba(_array(data["close"]), 25, 13)
        tsi_25_arr = calculate_tsi_numba(_array(data["close"]), 13, 25)

        # EMA for MACD/Signal
        ema_10_arr = calc_ema_array_numba(data["close"], 10)
        ema_20_arr = calc_ema_array_numba(data["close"], 20)
        ema_50_arr = calc_ema_array_numba(data["close"], 50)
        ema_100_arr = calc_ema_array_numba(data["close"], 100)
        ema_200_arr = calc_ema_array_numba(data["close"], 200)

        # ----------------------------------------
        # Features Dictionary Population
        # ----------------------------------------

        # ADX / Aroon
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

        # ATR Values & Bands
        features["e1c_atr_13"] = atr_13
        features["e1c_atr_pct_13"] = (atr_13 / data["close"][-1]) * 100
        features["e1c_atr_pct_21"] = (atr_21 / data["close"][-1]) * 100
        features["e1c_atr_pct_34"] = (atr_34 / data["close"][-1]) * 100
        features["e1c_atr_pct_55"] = (atr_55 / data["close"][-1]) * 100

        # ATR Trends (diff)
        features["e1c_atr_trend_13"] = atr_13_arr[-1] - atr_13_arr[-2]
        features["e1c_atr_trend_21"] = atr_21_arr[-1] - atr_21_arr[-2]
        features["e1c_atr_trend_34"] = atr_34_arr[-1] - atr_34_arr[-2]
        features["e1c_atr_trend_55"] = atr_55_arr[-1] - atr_55_arr[-2]

        # ATR Bands (Example for 21)
        features["e1c_atr_upper_21_2.0"] = data["close"][-1] + (atr_21 * 2.0)
        features["e1c_atr_lower_21_2.0"] = data["close"][-1] - (atr_21 * 2.0)
        # (Other ATR bands can be populated similarly as needed)

        # ATR Volatility
        features["e1c_atr_volatility_13"] = np.std(_window(atr_13_arr, 13))
        features["e1c_atr_volatility_21"] = np.std(_window(atr_21_arr, 21))

        # Bollinger Bands (Standard Calculation using numpy)
        mean_20 = np.mean(_window(data["close"], 20))
        std_20 = np.std(_window(data["close"], 20))
        mean_30 = np.mean(_window(data["close"], 30))
        std_30 = np.std(_window(data["close"], 30))
        mean_50 = np.mean(_window(data["close"], 50))
        std_50 = np.std(_window(data["close"], 50))

        # BB 20, 2.0
        features["e1c_bb_upper_20_2"] = mean_20 + 2.0 * std_20
        features["e1c_bb_lower_20_2"] = mean_20 - 2.0 * std_20
        features["e1c_bb_width_20_2"] = (
            features["e1c_bb_upper_20_2"] - features["e1c_bb_lower_20_2"]
        )
        features["e1c_bb_percent_20_2"] = (
            data["close"][-1] - features["e1c_bb_lower_20_2"]
        ) / (features["e1c_bb_width_20_2"] + 1e-10)

        # MACD (Using pre-calculated EMA arrays)
        # [修正] _ema -> calc_ema_array_numba に変更
        ema_12_arr = calc_ema_array_numba(data["close"], 12)
        ema_26_arr = calc_ema_array_numba(data["close"], 26)
        macd_12_26_arr = ema_12_arr - ema_26_arr
        signal_12_26_9_arr = calc_ema_array_numba(macd_12_26_arr, 9)

        features["e1c_macd_12_26"] = _last(macd_12_26_arr)
        features["e1c_macd_signal_12_26_9"] = _last(signal_12_26_9_arr)
        features["e1c_macd_histogram_12_26_9"] = (
            features["e1c_macd_12_26"] - features["e1c_macd_signal_12_26_9"]
        )

        # Other MACD variants (5, 35, 5) etc.
        ema_5_arr = calc_ema_array_numba(data["close"], 5)
        ema_35_arr = calc_ema_array_numba(data["close"], 35)
        macd_5_35_arr = ema_5_arr - ema_35_arr
        signal_5_35_5_arr = calc_ema_array_numba(macd_5_35_arr, 5)

        features["e1c_macd_5_35"] = _last(macd_5_35_arr)
        features["e1c_macd_signal_5_35_5"] = _last(signal_5_35_5_arr)
        features["e1c_macd_histogram_5_35_5"] = (
            features["e1c_macd_5_35"] - features["e1c_macd_signal_5_35_5"]
        )

        # HMA / KAMA (UDFs)
        features["e1c_hma_21"] = _last(calculate_hma_numba(_array(data["close"]), 21))
        features["e1c_hma_34"] = _last(calculate_hma_numba(_array(data["close"]), 34))
        features["e1c_hma_55"] = _last(calculate_hma_numba(_array(data["close"]), 55))
        features["e1c_kama_21"] = _last(calculate_kama_numba(_array(data["close"]), 21))
        features["e1c_kama_34"] = _last(calculate_kama_numba(_array(data["close"]), 34))

        # Momentum / ROC
        for period in [10, 20, 30, 50]:
            if len(data["close"]) > period:
                features[f"e1c_momentum_{period}"] = (
                    data["close"][-1] - data["close"][-1 - period]
                )
                features[f"e1c_rate_of_change_{period}"] = (
                    (data["close"][-1] - data["close"][-1 - period])
                    / (data["close"][-1 - period] + 1e-10)
                ) * 100
            else:
                features[f"e1c_momentum_{period}"] = np.nan
                features[f"e1c_rate_of_change_{period}"] = np.nan

        # RSI & Stochastic RSI
        features["e1c_rsi_14"] = _last(rsi_14_arr)
        features["e1c_rsi_21"] = _last(rsi_21_arr)
        features["e1c_rsi_30"] = _last(rsi_30_arr)
        features["e1c_rsi_50"] = _last(rsi_50_arr)

        # Stoch RSI
        rsi_14_window = _window(rsi_14_arr, 14)
        rsi_14_min = np.nanmin(rsi_14_window)
        rsi_14_max = np.nanmax(rsi_14_window)
        features["e1c_stochastic_rsi_14"] = (
            (_last(rsi_14_arr) - rsi_14_min) / (rsi_14_max - rsi_14_min + 1e-10)
        ) * 100

        # RSI Divergence (Simple logic)
        features["e1c_rsi_divergence_14"] = (
            (data["close"][-1] - data["close"][-15]) / (data["close"][-15] + 1e-10)
        ) - ((_last(rsi_14_arr) - rsi_14_arr[-15]) / 50 - 1)

        # Stochastic (Using extracted arrays)
        features["e1c_stoch_k_14"] = _last(stoch_k_14_arr)
        features["e1c_stoch_d_14_3"] = _last(stoch_d_14_3_arr)
        features["e1c_stoch_slow_d_14_3_3"] = _last(stoch_slow_d_14_3_3_arr)

        features["e1c_stoch_k_21"] = _last(stoch_k_21_arr)
        features["e1c_stoch_d_21_5"] = _last(stoch_d_21_5_arr)
        features["e1c_stoch_slow_d_21_5_5"] = _last(stoch_slow_d_21_5_5_arr)

        # SMA
        for period in [10, 20, 50, 100, 200]:
            sma = np.mean(_window(data["close"], period))
            features[f"e1c_sma_{period}"] = sma
            features[f"e1c_sma_deviation_{period}"] = (
                (data["close"][-1] - sma) / (sma + 1e-10)
            ) * 100

        # Trend Analysis
        for period in [20, 50, 100]:
            if len(data["close"]) > period:
                features[f"e1c_trend_slope_{period}"] = (
                    data["close"][-1] - data["close"][-1 - period]
                ) / period
                features[f"e1c_trend_strength_{period}"] = 1 / (
                    np.std(_window(data["close"], period)) + 1e-10
                )
                # Consistency using UDF
                features[f"e1c_trend_consistency_{period}"] = (
                    rolling_trend_consistency_numba(
                        _window(data["close"], period + 10), period
                    )
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

        # WMA (UDF, single point)
        features["e1c_wma_10"] = wma_rolling_numba(_window(data["close"], 10), 10)
        features["e1c_wma_20"] = wma_rolling_numba(_window(data["close"], 20), 20)
        features["e1c_wma_50"] = wma_rolling_numba(_window(data["close"], 50), 50)
        features["e1c_wma_100"] = wma_rolling_numba(_window(data["close"], 100), 100)
        features["e1c_wma_200"] = wma_rolling_numba(_window(data["close"], 200), 200)

        # DI +/-
        features["e1c_di_plus_13"] = _last(di_plus_13_arr)
        features["e1c_di_minus_13"] = _last(di_minus_13_arr)
        features["e1c_di_plus_21"] = _last(di_plus_21_arr)

        # Coppock Curve
        if len(data["close"]) >= 24:
            # [修正前]
            # roc_11_vals = _pct(_window(data["close"], 25)[10:])[-14:]

            # [修正後] ↓
            # 部分的に切り出して計算
            w_data = _window(data["close"], 25)
            if len(w_data) > 10:
                # [10:]でスライスしてから変化率計算
                roc_11_vals = calc_pct_change_numba(w_data[10:])[-14:]
            else:
                roc_11_vals = np.array([np.nan])

            # roc_14 も同様ですが、元のコードが _pct_change_n_array などを
            # 使わずに簡易実装している場合はそのままで構いません。
            # Numba化の恩恵を受けるのは上記の _pct 呼び出し部分です。
            # ※ 厳密な配列計算はコストが高いため、ここでは直近値の計算イメージのみ
            # 実際は _window で配列ごと渡してベクトル計算したほうが速い
            # (ここでは既存コードの流れに沿って省略)
            features["e1c_coppock_curve"] = np.nan  # Placeholder (実装推奨)

        # Ultimate Oscillator
        features["e1c_ultimate_oscillator"] = _last(
            calculate_ultimate_oscillator_numba(
                _array(data["high"]),
                _array(data["low"]),
                _array(data["close"]),
                _array(data["volume"]),
            )
        )

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

        # ------------------------------------------------------------------------------
        # 5. Engine 2A: Complexity Theory (MFDFA & Kolmogorov)
        # ------------------------------------------------------------------------------

        # MFDFA (Multi-Fractal Detrended Fluctuation Analysis)
        # Windows: 1000, 2500, 5000
        for w in [1000, 2500, 5000]:
            if len(data["close"]) >= w:
                window_data = _window(data["close"], w)
                # 0: hurst_mean, 1: width, 2: holder_max
                features[f"e2a_mfdfa_hurst_mean_{w}"] = mfdfa_rolling_udf(
                    window_data, 0
                )
                features[f"e2a_mfdfa_width_{w}"] = mfdfa_rolling_udf(window_data, 1)
                features[f"e2a_mfdfa_holder_max_{w}"] = mfdfa_rolling_udf(
                    window_data, 2
                )
            else:
                features[f"e2a_mfdfa_hurst_mean_{w}"] = np.nan
                features[f"e2a_mfdfa_width_{w}"] = np.nan
                features[f"e2a_mfdfa_holder_max_{w}"] = np.nan

        # Kolmogorov Complexity (Lempel-Ziv)
        # Windows: 500, 1000, 1500
        for w in [500, 1000, 1500]:
            if len(data["close"]) >= w:
                window_data = _window(data["close"], w)
                # 0: complexity, 1: compression_ratio, 2: pattern_diversity
                features[f"e2a_kolmogorov_complexity_{w}"] = (
                    kolmogorov_complexity_rolling_udf(window_data, 0)
                )
                features[f"e2a_compression_ratio_{w}"] = (
                    kolmogorov_complexity_rolling_udf(window_data, 1)
                )
                features[f"e2a_pattern_diversity_{w}"] = (
                    kolmogorov_complexity_rolling_udf(window_data, 2)
                )
            else:
                features[f"e2a_kolmogorov_complexity_{w}"] = np.nan
                features[f"e2a_compression_ratio_{w}"] = np.nan
                features[f"e2a_pattern_diversity_{w}"] = np.nan

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

        # ==============================================================================
        #  PATCH: Engine 1C Missing Features Implementation
        #  Strict Rules準拠: Engine 1Cロジックの完全再現
        # ==============================================================================

        # --- Helper for n-period ROC (returns array of same length with NaNs) ---
        def _pct_change_n_array(arr: np.ndarray, n: int) -> np.ndarray:
            res = np.full_like(arr, np.nan)
            if len(arr) > n:
                safe_denom = arr[:-n].copy()
                safe_denom[safe_denom == 0] = 1e-10
                res[n:] = (arr[n:] - safe_denom) / safe_denom
            return res

        # ------------------------------------------------------------------------------
        # 1. ボリンジャーバンド派生 (Bollinger Bands Variations)
        # ------------------------------------------------------------------------------
        # Engine 1C: bb_upper = sma + std_dev * std, bb_percent = (close - lower) / (upper - lower)

        # Period 20, Sigma 2.5
        if True:
            p, s = 20, 2.5
            # 既存の mean_20, std_20 を使用（再計算コスト削減）
            if std_20 > 1e-10:
                upper = mean_20 + s * std_20
                lower = mean_20 - s * std_20
                features[f"e1c_bb_lower_{p}_{s}"] = lower
                features[f"e1c_bb_upper_{p}_{s}"] = upper
                features[f"e1c_bb_percent_{p}_{s}"] = (data["close"][-1] - lower) / (
                    upper - lower + 1e-10
                )

        # Period 30, Sigma 2.5
        if True:
            p, s = 30, 2.5
            # 既存の mean_30, std_30 を使用
            if std_30 > 1e-10:
                upper = mean_30 + s * std_30
                lower = mean_30 - s * std_30
                width = upper - lower
                features[f"e1c_bb_lower_{p}_{s}"] = lower
                features[f"e1c_bb_percent_{p}_{s}"] = (data["close"][-1] - lower) / (
                    width + 1e-10
                )
                features[f"e1c_bb_width_{p}_{s}"] = width
                features[f"e1c_bb_width_pct_{p}_{s}"] = (
                    width / (mean_30 + 1e-10)
                ) * 100
                features[f"e1c_bb_position_{p}_{s}"] = (data["close"][-1] - mean_30) / (
                    std_30 + 1e-10
                )

        # Period 50, Sigma 2.0 (Percent only missing)
        if True:
            p, s = 50, 2.0
            if std_50 > 1e-10:
                upper = mean_50 + s * std_50
                lower = mean_50 - s * std_50
                features[f"e1c_bb_percent_{p}_{s}"] = (data["close"][-1] - lower) / (
                    upper - lower + 1e-10
                )

        # Period 50, Sigma 2.5
        if True:
            p, s = 50, 2.5
            if std_50 > 1e-10:
                upper = mean_50 + s * std_50
                lower = mean_50 - s * std_50
                features[f"e1c_bb_upper_{p}_{s}"] = upper
                features[f"e1c_bb_lower_{p}_{s}"] = lower
                features[f"e1c_bb_percent_{p}_{s}"] = (data["close"][-1] - lower) / (
                    upper - lower + 1e-10
                )

        # Period 50, Sigma 3.0
        if True:
            p, s = 50, 3.0
            if std_50 > 1e-10:
                upper = mean_50 + s * std_50
                lower = mean_50 - s * std_50
                features[f"e1c_bb_upper_{p}_{s}"] = upper
                features[f"e1c_bb_lower_{p}_{s}"] = lower
                features[f"e1c_bb_percent_{p}_{s}"] = (data["close"][-1] - lower) / (
                    upper - lower + 1e-10
                )

        # ------------------------------------------------------------------------------
        # 2. トレンド・モメンタム・乖離系
        # ------------------------------------------------------------------------------

        # Momentum & ROC (Overwrite to ensure existence)
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

        # SMA (100) & SMA Deviation (50, 100, 200)
        for period in [50, 100, 200]:
            # [修正前]
            # sma_val = rolling_mean_numba(_window(data["close"], period), period)
            # (※元のコードが単一値UDFを使っている場合はそのままでOKですが、
            # もし `_rolling_mean` を使っていたら以下のように修正)

            # 今回のコードでは `rolling_mean_numba` (単一値) を使っているので
            # ここの修正は【不要】です。そのままにしておいてください。

            sma_val = rolling_mean_numba(_window(data["close"], period), period)

            if period == 100:
                features["e1c_sma_100"] = sma_val

            if np.isnan(sma_val) or sma_val == 0:
                features[f"e1c_sma_deviation_{period}"] = np.nan
            else:
                features[f"e1c_sma_deviation_{period}"] = (
                    (data["close"][-1] - sma_val) / sma_val
                ) * 100

        # DPO (Detrended Price Oscillator) - 20, 30, 50
        # Engine 1C Logic: dpo = close - sma.shift(-lookback)
        # Warning: `shift(-lookback)` in Polars implies FUTURE reference (Lookahead).
        # In realtime environment, future data is unavailable.
        # Strict adherence to the logic implies the result is NaN for the latest bar.
        for period in [20, 30, 50]:
            features[f"e1c_dpo_{period}"] = np.nan

        # ------------------------------------------------------------------------------
        # 3. 方向性・トレンド強度
        # ------------------------------------------------------------------------------

        # DI +/- (13, 21) - Using existing UDFs
        features["e1c_di_plus_13"] = _last(di_plus_13_arr)
        features["e1c_di_minus_13"] = _last(di_minus_13_arr)
        features["e1c_di_plus_21"] = _last(di_plus_21_arr)
        # (di_minus_21 is already calculated as di_minus_21_arr, added for completeness if needed)

        # Trend Consistency & Strength (20, 50, 100)
        # Engine 1C Logic:
        #   strength = 1 / (std + 1e-10)
        #   consistency = 1 - mean(abs(diff(sign(diff(close))))) / 2
        for period in [20, 50, 100]:
            # Strength
            std_val = np.std(_window(data["close"], period))
            features[f"e1c_trend_strength_{period}"] = 1.0 / (std_val + 1e-10)

            # Consistency (Using new UDF)
            features[f"e1c_trend_consistency_{period}"] = (
                rolling_trend_consistency_numba(
                    _window(data["close"], period + 10), period
                )
            )

        # ------------------------------------------------------------------------------
        # 4. その他オシレーター
        # ------------------------------------------------------------------------------

        # Coppock Curve
        # Engine 1C: (ROC(11) + ROC(14)).rolling_mean(10)
        if len(data["close"]) >= 24:  # 14(ROC max lag) + 10(SMA)
            roc_11 = _pct_change_n_array(data["close"], 11) * 100
            roc_14 = _pct_change_n_array(data["close"], 14) * 100
            coppock_sum = roc_11 + roc_14

            # SMA of Coppock Sum (Last 10)
            coppock_window = _window(coppock_sum, 10)
            features["e1c_coppock_curve"] = np.mean(coppock_window)
        else:
            features["e1c_coppock_curve"] = np.nan

        # Ultimate Oscillator
        # Engine 1C: Uses High, Low, Close, Volume with weights 4, 2, 1
        features["e1c_ultimate_oscillator"] = _last(
            calculate_ultimate_oscillator_numba(
                _array(data["high"]),
                _array(data["low"]),
                _array(data["close"]),
                _array(data["volume"]),
            )
        )

        return features
