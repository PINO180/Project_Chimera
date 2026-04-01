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

    # Numba互換のNaN除去
    finite_vals = arr[np.isfinite(arr)]
    if len(finite_vals) < 3:
        return np.nan

    mean_val = np.mean(finite_vals)
    std_val = np.std(finite_vals)

    if std_val < 1e-10:
        return 0.0

    m3 = np.mean((finite_vals - mean_val) ** 3)
    skew = m3 / (std_val**3)
    return skew


@njit(fastmath=True, cache=True)
def rolling_kurtosis_numba(arr: np.ndarray) -> float:
    """ローリング尖度 (Numba JIT)"""
    n = len(arr)
    if n < 4:
        return np.nan

    # Numba互換のNaN除去
    finite_vals = arr[np.isfinite(arr)]
    if len(finite_vals) < 4:
        return np.nan

    mean_val = np.mean(finite_vals)
    std_val = np.std(finite_vals)

    if std_val < 1e-10:
        return 0.0

    m4 = np.mean((finite_vals - mean_val) ** 4)
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
    ローリングMAD (Numba JIT)
    Engine 1A: mad_rolling_numba のロジックを完全再現
    Windowサイズはソースに合わせて固定(20)または配列長を使用
    """
    n = len(arr)
    # ソーススクリプトでは window=20 固定
    window = 20

    # バッファがウィンドウより大きい場合は末尾を使用、小さい場合は全体を使用
    start_idx = max(0, n - window)
    window_data = arr[start_idx:]

    # 有限値のみ抽出
    finite_data = window_data[~np.isnan(window_data)]

    if len(finite_data) < 3:
        return np.nan

    # 中央値計算
    median_val = np.median(finite_data)
    # 絶対偏差の中央値
    abs_deviations = np.abs(finite_data - median_val)
    return np.median(abs_deviations)


@njit(fastmath=True, cache=True)
def jarque_bera_statistic_numba(arr: np.ndarray) -> float:
    """
    ローリングJarque-Bera (Numba JIT)
    Engine 1A: jarque_bera_statistic_numba のロジックを完全再現
    """
    n = len(arr)
    # ソーススクリプトでは window=50 固定
    window = 50

    start_idx = max(0, n - window)
    window_data = arr[start_idx:]

    finite_data = window_data[~np.isnan(window_data)]

    if len(finite_data) < 20:
        return np.nan

    # 基本統計量
    mean_val = np.mean(finite_data)

    # 手動分散計算（Numba対応）
    variance = 0.0
    for val in finite_data:
        variance += (val - mean_val) ** 2
    variance = variance / (len(finite_data) - 1)
    std_val = np.sqrt(variance)

    if std_val < 1e-10:
        return 0.0

    # 標準化
    z_sum_3 = 0.0
    z_sum_4 = 0.0
    for val in finite_data:
        z = (val - mean_val) / std_val
        z_sum_3 += z**3
        z_sum_4 += z**4

    skewness = z_sum_3 / len(finite_data)
    kurtosis = z_sum_4 / len(finite_data) - 3

    # JB統計量
    jb_stat = len(finite_data) * (skewness**2 / 6.0 + kurtosis**2 / 24.0)
    return jb_stat


@njit(fastmath=True, cache=True)
def anderson_darling_numba(arr: np.ndarray) -> float:
    """
    ローリングAnderson-Darling (Numba JIT)
    Engine 1A: anderson_darling_numba のロジックを完全再現
    """
    n = len(arr)
    # ソーススクリプトでは window=30 固定
    window = 30

    start_idx = max(0, n - window)
    window_data = arr[start_idx:]

    finite_data = window_data[~np.isnan(window_data)]

    if len(finite_data) < 10:
        return np.nan

    # ソート
    sorted_data = np.sort(finite_data)
    n_data = len(sorted_data)

    # 手動で平均と標準偏差計算（Numba対応）
    mean_val = np.mean(sorted_data)

    # 手動分散計算
    variance = 0.0
    for val in sorted_data:
        variance += (val - mean_val) ** 2
    variance = variance / (n_data - 1)
    std_val = np.sqrt(variance)

    if std_val < 1e-10:
        return 0.0

    # 定数定義 (math.sqrt(2.0))
    SQRT2 = 1.4142135623730951

    # Anderson-Darling統計量の厳密計算
    ad_sum = 0.0
    for j in range(n_data):
        # 標準化
        z_j = (sorted_data[j] - mean_val) / std_val
        z_nj = (sorted_data[n_data - 1 - j] - mean_val) / std_val

        # 高速CDF (erf使用)
        F_j = 0.5 * (1.0 + math.erf(z_j / SQRT2))
        F_nj = 0.5 * (1.0 + math.erf(z_nj / SQRT2))

        # ゼロ除算・対数エラー回避
        if F_j < 1e-15:
            F_j = 1e-15
        if (1.0 - F_nj) < 1e-15:
            F_nj = 1.0 - 1e-15  # 1-F_nj should be guarded

        # ソースロジック: if F_j > 1e-15 and (1 - F_nj) > 1e-15:
        if F_j > 1e-15 and (1.0 - F_nj) > 1e-15:
            log_term = np.log(F_j) + np.log(1.0 - F_nj)
            ad_sum += (2 * j + 1) * log_term

    # Anderson-Darling統計量
    return -n_data - ad_sum / n_data


@njit(fastmath=True, cache=True)
def runs_test_numba(arr: np.ndarray) -> float:
    """
    ローリングRuns Test (Numba JIT)
    Engine 1A: runs_test_numba のロジックを完全再現
    """
    n = len(arr)
    # ソーススクリプトでは window=30 固定
    window = 30

    start_idx = max(0, n - window)
    window_data = arr[start_idx:]

    finite_data = window_data[~np.isnan(window_data)]

    if len(finite_data) < 10:
        return np.nan

    # 中央値を基準にバイナリ系列作成
    median_val = np.median(finite_data)
    # numbaでのbool array作成
    binary_series = (finite_data > median_val).astype(np.int32)

    # ランの数をカウント
    runs = 1
    for j in range(1, len(binary_series)):
        if binary_series[j] != binary_series[j - 1]:
            runs += 1

    # 期待ランの数と分散
    n1 = np.sum(binary_series)  # 1の個数
    n2 = len(binary_series) - n1  # 0の個数

    if n1 > 0 and n2 > 0:
        expected_runs = (2 * n1 * n2) / (n1 + n2) + 1
        denom = (n1 + n2) ** 2 * (n1 + n2 - 1)
        if denom == 0:
            return 0.0
        var_runs = (2 * n1 * n2 * (2 * n1 * n2 - n1 - n2)) / denom

        if var_runs > 0:
            # 標準化統計量
            return (runs - expected_runs) / np.sqrt(var_runs)
        else:
            return 0.0
    else:
        return 0.0


@njit(fastmath=True, cache=True)
def von_neumann_ratio_numba(arr: np.ndarray) -> float:
    """
    ローリングVon Neumann比 (Numba JIT)
    Engine 1A: von_neumann_ratio_numba のロジックを完全再現
    """
    n = len(arr)
    # ソーススクリプトでは window=30 固定
    window = 30

    start_idx = max(0, n - window)
    window_data = arr[start_idx:]

    finite_data = window_data[~np.isnan(window_data)]

    if len(finite_data) < 3:  # 最低3点必要
        return np.nan

    n_points = len(finite_data)

    # 1次差分の平方和（厳密計算）
    diff_sq_sum = 0.0
    for j in range(1, n_points):
        diff = finite_data[j] - finite_data[j - 1]
        diff_sq_sum += diff * diff

    # 平均値の厳密計算
    sum_values = 0.0
    for j in range(n_points):
        sum_values += finite_data[j]
    mean_val = sum_values / n_points

    # 不偏分散の厳密計算（n-1で除算）
    sum_sq_deviations = 0.0
    for j in range(n_points):
        deviation = finite_data[j] - mean_val
        sum_sq_deviations += deviation * deviation

    # Von Neumann比の厳密な定義
    if sum_sq_deviations > 1e-15:
        # 分子: 1次差分の平方和
        # 分母: 総平方和（不偏分散 × (n-1)）
        # Note: sum_sq_deviations は分散*(n-1) そのもの
        vn_ratio = diff_sq_sum / sum_sq_deviations

        # 理論的範囲チェック（0 ≤ VN比 ≤ 4）
        if vn_ratio < 0.0:
            return 0.0
        elif vn_ratio > 4.0:
            return 4.0
        else:
            return vn_ratio
    else:
        # 全て同じ値の場合、理論的にVN比は0
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


# ----------------------------------------
# from engine_1_C_a_vast_universe_of_features.py
# ----------------------------------------


@njit(fastmath=True, cache=True)
def calculate_rsi_numba(prices: np.ndarray, period: int) -> np.ndarray:
    """
    RSI計算 (Numba JIT) - 配列返し
    """
    n = len(prices)
    result = np.full(n, np.nan, dtype=np.float64)

    if n < period + 1:
        return result

    # 差分計算
    deltas = np.diff(prices)

    # 最初の平均ゲイン/ロスを計算 (SMA方式: Cutler's RSI)
    # または Wilder's Smoothing を使用する場合もあるが、
    # 元のコードが単純な区間合計(SMA的)なロジックだったため、それを維持しつつ配列化

    # ここでは一般的に安定している Wilder's Smoothing で実装し、配列全体を埋める
    # (元のロジックの「最新ウィンドウのみ」から「履歴全体」へ変更)

    seed = deltas[:period]
    up = seed[seed >= 0].sum() / period
    down = -seed[seed < 0].sum() / period

    if down != 0:
        rs = up / down
        result[period] = 100.0 - 100.0 / (1.0 + rs)
    else:
        result[period] = 100.0

    # その後の期間を計算
    for i in range(period + 1, n):
        delta = deltas[i - 1]  # diffは長さがn-1

        if delta > 0:
            up_val = delta
            down_val = 0.0
        else:
            up_val = 0.0
            down_val = -delta

        # Wilder's Smoothing
        # up = (up * (period - 1) + up_val) / period
        # down = (down * (period - 1) + down_val) / period

        # 元コードが SMA ベース (Cutler's) だったため、SMAロジックで再現
        # 直近 period 個の平均を使用

        # 直近 period 個の diff を取得
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
def calculate_atr_numba(
    high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int
) -> np.ndarray:
    """
    ATR計算 (Numba JIT) - 配列返し
    """
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

    # ATR計算 (SMAベース)
    # 最初の値を計算
    if period <= n:
        current_sum = 0.0
        for i in range(period):
            current_sum += tr[i]
        result[period - 1] = current_sum / period

        # 以降を計算
        for i in range(period, n):
            # SMA: window slide
            current_sum = current_sum - tr[i - period] + tr[i]
            result[i] = current_sum / period

    return result


@njit(fastmath=True, cache=True)
def calculate_adx_numba(
    high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int
) -> np.ndarray:
    """
    ADX計算 (Numba JIT) - 配列返し
    """
    n = len(high)
    adx = np.full(n, np.nan, dtype=np.float64)

    if n < period * 2:
        return adx

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

            up = high[i] - high[i - 1]
            down = low[i - 1] - low[i]

            if up > down and up > 0:
                dm_plus[i] = up
            else:
                dm_plus[i] = 0.0

            if down > up and down > 0:
                dm_minus[i] = down
            else:
                dm_minus[i] = 0.0

    # DX計算
    dx = np.full(n, np.nan, dtype=np.float64)

    # SMAロジックでDI+, DI-を計算し、そこからDXを算出
    # 効率化のため、スライディングウィンドウで和を更新

    sum_tr = 0.0
    sum_dm_p = 0.0
    sum_dm_m = 0.0

    # 初期ウィンドウ
    for i in range(period):
        sum_tr += tr[i]
        sum_dm_p += dm_plus[i]
        sum_dm_m += dm_minus[i]

    # period番目以降
    for i in range(period, n):
        # Update sums (SMA)
        sum_tr = sum_tr - tr[i - period] + tr[i]
        sum_dm_p = sum_dm_p - dm_plus[i - period] + dm_plus[i]
        sum_dm_m = sum_dm_m - dm_minus[i - period] + dm_minus[i]

        if sum_tr > 0:
            di_plus = 100 * (sum_dm_p / period) / (sum_tr / period)
            di_minus = 100 * (sum_dm_m / period) / (sum_tr / period)
        else:
            di_plus = 0.0
            di_minus = 0.0

        sum_di = di_plus + di_minus
        if sum_di > 0:
            dx[i] = 100 * abs(di_plus - di_minus) / sum_di
        else:
            dx[i] = 0.0

    # ADX = SMA(DX)
    # DXは index=period から値が入る。
    # ADX計算には period 個の DX が必要。つまり index = 2*period - 1 あたりから開始

    start_adx = period * 2 - 1
    if start_adx < n:
        sum_dx = 0.0
        # 初期ADXウィンドウ
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
def calculate_di_plus_numba(
    high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int
) -> np.ndarray:
    """
    DI+ 計算 (Numba JIT) - 配列返し
    """
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

    # 初期和
    for i in range(period):
        sum_tr += tr[i]
        sum_dm_p += dm_plus[i]

    # 計算
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
    """
    DI- 計算 (Numba JIT) - 配列返し
    """
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
def calculate_hma_numba(prices: np.ndarray, period: int) -> np.ndarray:
    """
    Hull Moving Average計算 (Numba JIT) - 配列返し
    """
    n = len(prices)
    hma = np.full(n, np.nan, dtype=np.float64)

    half_period = int(period / 2)
    sqrt_period = int(math.sqrt(period))

    if n < period + sqrt_period:
        return hma

    # 内部WMA関数 (インライン展開的処理)
    def calc_wma_array(arr, w_period):
        res = np.full(len(arr), np.nan, dtype=np.float64)
        if len(arr) < w_period:
            return res

        # 重みの合計
        denom = w_period * (w_period + 1) / 2.0

        # 重み付け合計をスライディングウィンドウで計算
        # WMA_t = (p_t*n + p_{t-1}*(n-1) + ... + p_{t-n+1}*1) / denom
        # 効率的な更新式は複雑なので、Numbaの速さを活かして都度計算する

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

    # 2 * WMA(n/2) - WMA(n)
    raw_hma = np.full(n, np.nan, dtype=np.float64)
    for i in range(n):
        if not np.isnan(wma_half[i]) and not np.isnan(wma_full[i]):
            raw_hma[i] = 2 * wma_half[i] - wma_full[i]

    # WMA(sqrt(n)) of raw_hma
    # raw_hmaは period-1 あたりから値が入る
    hma_temp = calc_wma_array(raw_hma, sqrt_period)

    return hma_temp


@njit(fastmath=True, cache=True)
def calculate_kama_numba(prices: np.ndarray, period: int) -> np.ndarray:
    """
    Kaufman Adaptive Moving Average計算 (Numba JIT) - 配列返し
    """
    n = len(prices)
    kama = np.full(n, np.nan, dtype=np.float64)

    if n < period:
        return kama

    fast_ema = 2.0
    slow_ema = 30.0

    fast_sc = 2.0 / (fast_ema + 1.0)
    slow_sc = 2.0 / (slow_ema + 1.0)

    # 初期化
    current_kama = prices[period - 1]
    kama[period - 1] = current_kama

    for i in range(period, n):
        # Change
        change = abs(prices[i] - prices[i - period])

        # Volatility (Sum of absolute differences)
        volatility = 0.0
        for j in range(period):
            volatility += abs(prices[i - j] - prices[i - j - 1])

        if volatility == 0:
            er = 0.0
        else:
            er = change / volatility

        sc = (er * (fast_sc - slow_sc) + slow_sc) ** 2

        current_kama = current_kama + sc * (prices[i] - current_kama)
        kama[i] = current_kama

    return kama


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
    Stochastic Oscillator計算 (Numba JIT) - 配列返し (Slow %D)
    呼び出し元が (h, l, c, k) だけ渡す場合にも対応できるようデフォルト引数設定
    """
    n = len(close)
    stoch_k = np.full(n, np.nan, dtype=np.float64)
    stoch_d = np.full(n, np.nan, dtype=np.float64)  # %KのSMA
    stoch_slow_d = np.full(n, np.nan, dtype=np.float64)  # %DのSMA

    if n < k_period:
        return stoch_slow_d

    # %K 計算
    for i in range(k_period - 1, n):
        # window: [i - k_period + 1 : i + 1]
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

    # Slow %D 計算 (SMA of %D)
    # UDF呼び出し側が返り値をどう使っているかによるが、通常ストキャスティクス関数は
    # %K, %D を返すべきだが、元のコードは Slow%D を返していた。
    # ここでは「計算された配列」として、最も平滑化されたもの（Slow %D）を返す。
    # 必要ならタプルで返す設計にすべきだが、互換性のため Slow%D 配列を返す。

    # もし slow_period が 1 なら %D と同じ
    if slow_period == 1:
        return stoch_d

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

    return stoch_slow_d


@njit(fastmath=True, cache=True)
def calculate_williams_r_numba(
    high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int
) -> np.ndarray:
    """
    Williams %R計算 (Numba JIT) - 配列返し
    """
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
    """
    TRIX指標計算 (Numba JIT) - 配列返し
    """
    n = len(prices)
    trix = np.full(n, np.nan, dtype=np.float64)

    if n < period * 3:
        return trix

    alpha = 2.0 / (period + 1.0)

    ema1 = np.zeros(n)
    ema2 = np.zeros(n)
    ema3 = np.zeros(n)

    # EMA1
    curr = prices[0]
    ema1[0] = curr
    for i in range(1, n):
        curr = alpha * prices[i] + (1.0 - alpha) * curr
        ema1[i] = curr

    # EMA2
    curr = ema1[0]
    ema2[0] = curr
    for i in range(1, n):
        curr = alpha * ema1[i] + (1.0 - alpha) * curr
        ema2[i] = curr

    # EMA3
    curr = ema2[0]
    ema3[0] = curr
    for i in range(1, n):
        curr = alpha * ema2[i] + (1.0 - alpha) * curr
        ema3[i] = curr

    # TRIX = 1-period ROC of EMA3
    for i in range(1, n):
        prev = ema3[i - 1]
        if prev != 0:
            trix[i] = 10000.0 * (ema3[i] - prev) / prev
        else:
            trix[i] = 0.0

    return trix


@njit(fastmath=True, cache=True)
def calculate_ultimate_oscillator_numba(
    high: np.ndarray, low: np.ndarray, close: np.ndarray, volume: np.ndarray
) -> np.ndarray:
    """
    Ultimate Oscillator計算 (Numba JIT) - 配列返し
    """
    n = len(close)
    uo = np.full(n, np.nan, dtype=np.float64)

    # Periods defined in standard
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

    # Sliding window sums
    # Need sum(bp, p) and sum(tr, p) at each point i

    for i in range(p3, n):
        bp_sum1 = 0.0
        tr_sum1 = 0.0
        bp_sum2 = 0.0
        tr_sum2 = 0.0
        bp_sum3 = 0.0
        tr_sum3 = 0.0

        # p1
        for j in range(p1):
            idx = i - j
            bp_sum1 += bp[idx]
            tr_sum1 += tr[idx]

        # p2
        for j in range(p2):
            idx = i - j
            bp_sum2 += bp[idx]
            tr_sum2 += tr[idx]

        # p3
        for j in range(p3):
            idx = i - j
            bp_sum3 += bp[idx]
            tr_sum3 += tr[idx]

        avg1 = (bp_sum1 / tr_sum1) if tr_sum1 > 0 else 0.0
        avg2 = (bp_sum2 / tr_sum2) if tr_sum2 > 0 else 0.0
        avg3 = (bp_sum3 / tr_sum3) if tr_sum3 > 0 else 0.0

        uo[i] = 100.0 * (avg1 * w1 + avg2 * w2 + avg3 * w3) / (w1 + w2 + w3)

    return uo


@njit(fastmath=True, cache=True)
def calculate_aroon_up_numba(high: np.ndarray, period: int) -> np.ndarray:
    """
    Aroon Up計算 (Numba JIT) - 配列返し
    """
    n = len(high)
    result = np.full(n, np.nan, dtype=np.float64)

    if n < period:
        return result

    for i in range(period, n):
        # range: [i - period + 1] to [i] inclusive
        start_idx = i - period + 1

        highest_idx = -1
        highest_val = -1e30

        for j in range(start_idx, i + 1):
            if high[j] > highest_val:
                highest_val = high[j]
                highest_idx = j

        # periods since highest
        periods_since = i - highest_idx
        result[i] = 100.0 * (period - periods_since) / period

    return result


@njit(fastmath=True, cache=True)
def calculate_aroon_down_numba(low: np.ndarray, period: int) -> np.ndarray:
    """
    Aroon Down計算 (Numba JIT) - 配列返し
    """
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


@njit(fastmath=True, cache=True)
def calculate_tsi_numba(
    prices: np.ndarray, long_period: int, short_period: int
) -> np.ndarray:
    """
    True Strength Index計算 (Numba JIT) - 配列返し
    Args: period引数は元のコードに合わせてシグネチャを変更
          (prices, period) ではなく (prices, long_period, short_period) が必要
          元の呼び出し側で引数を合わせるか、ここでラップする。
          依頼文のリストには `calculate_tsi_numba` とあり、
          呼び出し元コード `_calculate_base_features` では
          `calculate_tsi_numba(_array(data["close"]), 25, 13)` と呼んでいるため
          シグネチャを修正。
    """
    n = len(prices)
    tsi = np.full(n, np.nan, dtype=np.float64)

    if n < long_period + short_period:
        return tsi

    # Momentum
    mom = np.zeros(n)
    abs_mom = np.zeros(n)
    for i in range(1, n):
        val = prices[i] - prices[i - 1]
        mom[i] = val
        abs_mom[i] = abs(val)

    # EMA Function
    def calc_ema_arr(arr, span):
        res = np.zeros(len(arr))
        alpha = 2.0 / (span + 1.0)
        curr = arr[0]
        res[0] = curr
        for i in range(1, len(arr)):
            curr = alpha * arr[i] + (1.0 - alpha) * curr
            res[i] = curr
        return res

    # Double Smooth
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
    # direction_changes[k] = abs(sign(close[k] - close[k-1]) - sign(close[k-1] - close[k-2]))

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
    学習用スクリプトのロジックを完全再現
    """
    n = len(prices)
    n_q = len(q_values)
    n_scales = len(scales)

    # 初期化 [h_mean, width, h_max]
    result = np.full(3, np.nan)

    if n < 20:
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

        # セグメント数
        n_segments = n // scale

        # 各セグメントの揺らぎを計算
        segment_variances = np.zeros(n_segments)

        for seg in range(n_segments):
            start = seg * scale
            end = start + scale

            # セグメント抽出
            segment = profile[start:end]

            # トレンド除去
            detrended = polynomial_fit_detrend(segment, poly_degree)

            # 分散計算
            variance = 0.0
            for val in detrended:
                variance += val * val
            variance = variance / scale if scale > 0 else 0.0

            segment_variances[seg] = variance

        # qモーメント揺らぎ関数の計算
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
                # q≠0の場合
                sum_val = 0.0
                valid_count = 0
                for var in segment_variances:
                    if var > 1e-10:
                        sum_val += np.power(var, q / 2.0)
                        valid_count += 1

                if valid_count > 0:
                    F_q[q_idx, s_idx] = np.power(sum_val / valid_count, 1.0 / q)

    # 3. 一般化Hurst指数の推定(log(F_q) vs log(scale)の傾き)
    h_q_values = np.zeros(n_q)

    for q_idx in range(n_q):
        # 有効なスケールのみ使用
        log_scales = []
        log_F_vals = []

        for s_idx in range(n_scales):
            if F_q[q_idx, s_idx] > 1e-10:
                log_scales.append(np.log(scales[s_idx]))
                log_F_vals.append(np.log(F_q[q_idx, s_idx]))

        if len(log_scales) >= 3:
            # 線形回帰で傾きを推定
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

    # 4. マルチフラクタルスペクトラムの計算
    valid_h = h_q_values[np.isfinite(h_q_values)]

    if len(valid_h) >= 3:
        h_mean = np.mean(valid_h)
        h_max = np.max(valid_h)
        h_min = np.min(valid_h)
        multifractal_width = h_max - h_min

        result[0] = h_mean  # 平均Hurst指数
        result[1] = multifractal_width  # Δα
        result[2] = h_max  # α_max

    return result


@njit(fastmath=True, cache=True)
def mfdfa_rolling_udf(prices: np.ndarray, component_idx: int) -> float:
    """
    MFDFA (Numba JIT) - リアルタイム単一ウィンドウ版

    Args:
        prices: 最新のウィンドウデータ
        component_idx: 0=hurst_mean, 1=width, 2=holder_max
    """
    # 学習用スクリプトのデフォルトパラメータ定義
    q_values = np.array([-5.0, -3.0, -1.0, 0.0, 1.0, 3.0, 5.0])
    scales = np.array([10.0, 20.0, 50.0, 100.0, 200.0])

    # コア計算を実行
    mfdfa_result = mfdfa_core_single_window(prices, q_values, scales, poly_degree=1)

    return mfdfa_result[component_idx]


@njit(fastmath=True, cache=True)
def lempel_ziv_complexity(sequence: np.ndarray) -> float:
    """
    Lempel-Ziv複雑性計算(LZ76アルゴリズム)
    """
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
        normalized_complexity = (
            complexity / max_complexity if max_complexity > 0 else 0.0
        )
        return min(normalized_complexity, 1.0)

    return 0.0


@njit(fastmath=True, cache=True)
def kolmogorov_complexity_rolling_udf(prices: np.ndarray, component_idx: int) -> float:
    """
    コルモゴロフ複雑性 (Numba JIT) - リアルタイム単一ウィンドウ版

    Args:
        prices: 最新のウィンドウデータ
        component_idx: 0=complexity, 1=compression_ratio, 2=pattern_diversity
    """
    n = len(prices)
    if n < 10:
        return np.nan

    # 1. 対数リターン計算
    returns = np.zeros(n - 1)
    for i in range(n - 1):
        if prices[i] > 1e-10:
            returns[i] = np.log(prices[i + 1] / prices[i])

    # 2. 標準化
    returns_mean = np.mean(returns)
    returns_std = np.std(returns)

    if returns_std < 1e-10:
        # 標準偏差が0の場合、情報はゼロとみなす
        if component_idx == 0:
            return 0.0
        if component_idx == 1:
            return 1.0
        if component_idx == 2:
            return 0.0
        return np.nan

    standardized = np.zeros(len(returns))
    for i in range(len(returns)):
        standardized[i] = (returns[i] - returns_mean) / returns_std

    # 3. バイナリ化(中央値基準: Method 0)
    median_val = np.median(standardized)
    encoded = np.zeros(len(standardized), dtype=np.int32)
    for i in range(len(standardized)):
        encoded[i] = 1 if standardized[i] > median_val else 0

    # 4. LZ複雑性計算
    complexity = lempel_ziv_complexity(encoded)

    if component_idx == 0:
        return complexity

    elif component_idx == 1:
        # 5. 圧縮率推定
        return 1.0 - complexity

    elif component_idx == 2:
        # 6. パターン多様性(ユニーク値の割合)
        # Numbaでの簡易実装: 符号化された配列内のユニークな部分列のカウントはコストが高いので、
        # ここでは「エントロピー的」な多様性ではなく、学習用スクリプトにある
        # 「単純なバイナリの一致チェックによるユニークカウント」を再現する
        # (学習用スクリプトの kolmogorov_complexity_single_window ロジック)
        unique_count = 0
        for i in range(len(encoded)):
            is_unique = True
            for j in range(i):
                if (
                    encoded[i] == encoded[j]
                ):  # 注: 1bit符号化だとこれは0/1の頻度になるが、元ロジックを尊重
                    is_unique = False
                    break
            if is_unique:
                unique_count += 1

        return unique_count / len(encoded) if len(encoded) > 0 else 0.0

    return np.nan


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

        # WMA (修正: 単一ポイント計算に最適化)
        # _window() を使って必要な長さだけ切り出して渡す
        features["e1c_wma_10"] = wma_rolling_numba(_window(data["close"], 10), 10)
        features["e1c_wma_20"] = wma_rolling_numba(_window(data["close"], 20), 20)
        features["e1c_wma_50"] = wma_rolling_numba(_window(data["close"], 50), 50)
        features["e1c_wma_100"] = wma_rolling_numba(_window(data["close"], 100), 100)
        features["e1c_wma_200"] = wma_rolling_numba(_window(data["close"], 200), 200)

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
            # 既存の bb_mean_20, bb_std_20 を使用（再計算コスト削減）
            if bb_std_20 > 1e-10:
                upper = bb_mean_20 + s * bb_std_20
                lower = bb_mean_20 - s * bb_std_20
                features[f"e1c_bb_lower_{p}_{s}"] = lower
                features[f"e1c_bb_upper_{p}_{s}"] = upper
                features[f"e1c_bb_percent_{p}_{s}"] = (data["close"][-1] - lower) / (
                    upper - lower + 1e-10
                )

        # Period 30, Sigma 2.5
        if True:
            p, s = 30, 2.5
            # 既存の bb_mean_30, bb_std_30 を使用
            if bb_std_30 > 1e-10:
                upper = bb_mean_30 + s * bb_std_30
                lower = bb_mean_30 - s * bb_std_30
                width = upper - lower
                features[f"e1c_bb_lower_{p}_{s}"] = lower
                features[f"e1c_bb_percent_{p}_{s}"] = (data["close"][-1] - lower) / (
                    width + 1e-10
                )
                features[f"e1c_bb_width_{p}_{s}"] = width
                features[f"e1c_bb_width_pct_{p}_{s}"] = (
                    width / (bb_mean_30 + 1e-10)
                ) * 100
                features[f"e1c_bb_position_{p}_{s}"] = (
                    data["close"][-1] - bb_mean_30
                ) / (bb_std_30 + 1e-10)

        # Period 50, Sigma 2.0 (Percent only missing)
        if True:
            p, s = 50, 2.0
            if bb_std_50 > 1e-10:
                upper = bb_mean_50 + s * bb_std_50
                lower = bb_mean_50 - s * bb_std_50
                features[f"e1c_bb_percent_{p}_{s}"] = (data["close"][-1] - lower) / (
                    upper - lower + 1e-10
                )

        # Period 50, Sigma 2.5
        if True:
            p, s = 50, 2.5
            if bb_std_50 > 1e-10:
                upper = bb_mean_50 + s * bb_std_50
                lower = bb_mean_50 - s * bb_std_50
                features[f"e1c_bb_upper_{p}_{s}"] = upper
                features[f"e1c_bb_lower_{p}_{s}"] = lower
                features[f"e1c_bb_percent_{p}_{s}"] = (data["close"][-1] - lower) / (
                    upper - lower + 1e-10
                )

        # Period 50, Sigma 3.0
        if True:
            p, s = 50, 3.0
            if bb_std_50 > 1e-10:
                upper = bb_mean_50 + s * bb_std_50
                lower = bb_mean_50 - s * bb_std_50
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
            # Use newly added UDF or numpy
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
