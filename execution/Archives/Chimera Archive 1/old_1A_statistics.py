# realtime_feature_engine_1A_statistics.py
# Category 1A: 基本統計・分布検定系 (Basic Statistics & Distribution Tests)

import numpy as np
import numba as nb
from numba import njit
import math
from typing import Dict

# ==================================================================
# 1A. NUMBA UDF ライブラリ (Part 1: 基本統計・高速ローリング系)
# ==================================================================


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
def statistical_kurtosis_numba(arr: np.ndarray) -> float:
    """ローリング尖度 (Numba JIT)"""
    n = len(arr)
    if n < 4:  # 尖度は最低4サンプル必要
        return np.nan

    finite_vals = arr[np.isfinite(arr)]
    if len(finite_vals) < 4:
        return np.nan

    mean_val = np.mean(finite_vals)
    std_val = np.std(finite_vals)

    if std_val < 1e-10:
        return 0.0

    m4 = np.mean((finite_vals - mean_val) ** 4)
    kurtosis = m4 / (std_val**4) - 3.0  # 過剰尖度
    return kurtosis


@nb.njit(fastmath=True, cache=True)
def fast_rolling_mean_numba(arr: np.ndarray, window: int) -> float:
    """
    Numba最適化ローリング平均（カスタムウィンドウ用）
    """
    n = len(arr)
    if n < 20 and window >= 20:
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
    """
    n = len(arr)
    if n < 20 and window >= 20:
        return np.nan

    start_idx = max(0, n - window)
    window_data = arr[start_idx:]

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
    バッファ全体の分布に基づいて、最新の1点を安定化（クリップ）した値を
    1要素の配列として返す（_last()での呼び出しに対応するため）。
    """
    n = len(arr)
    result = np.zeros(1, dtype=np.float64)

    if n == 0:
        return result

    cleaned = np.zeros(n)
    for i in range(n):
        if np.isnan(arr[i]) or np.isinf(arr[i]):
            cleaned[i] = 0.0
        else:
            cleaned[i] = arr[i]

    finite_count = 0
    for i in range(n):
        if np.isfinite(cleaned[i]):
            finite_count += 1

    last_val = cleaned[n - 1]

    if finite_count > 10:
        min_val = np.nanmin(cleaned)
        max_val = np.nanmax(cleaned)
        range_val = max_val - min_val

        if range_val > 1e-10:
            valid_data = cleaned[np.isfinite(cleaned)]
            clip_margin_low = np.percentile(valid_data, 1)
            clip_margin_high = np.percentile(valid_data, 99)

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


# ==================================================================
# 1A. NUMBA UDF ライブラリ (Part 2: ロバスト統計・分布検定系)
# ==================================================================


@njit(fastmath=True, cache=True)
def mad_rolling_numba(arr: np.ndarray) -> float:
    """
    ローリングMAD (Numba JIT)
    """
    n = len(arr)
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
    """
    n = len(arr)
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
    """
    n = len(arr)
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
            F_nj = 1.0 - 1e-15

        if F_j > 1e-15 and (1.0 - F_nj) > 1e-15:
            log_term = np.log(F_j) + np.log(1.0 - F_nj)
            ad_sum += (2 * j + 1) * log_term

    # Anderson-Darling統計量
    return -n_data - ad_sum / n_data


@njit(fastmath=True, cache=True)
def runs_test_numba(arr: np.ndarray) -> float:
    """
    ローリングRuns Test (Numba JIT)
    """
    n = len(arr)
    window = 30

    start_idx = max(0, n - window)
    window_data = arr[start_idx:]

    finite_data = window_data[~np.isnan(window_data)]

    if len(finite_data) < 10:
        return np.nan

    # 中央値を基準にバイナリ系列作成
    median_val = np.median(finite_data)
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
    """
    n = len(arr)
    window = 30

    start_idx = max(0, n - window)
    window_data = arr[start_idx:]

    finite_data = window_data[~np.isnan(window_data)]

    if len(finite_data) < 3:
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
        vn_ratio = diff_sq_sum / sum_sq_deviations
        if vn_ratio < 0.0:
            return 0.0
        elif vn_ratio > 4.0:
            return 4.0
        else:
            return vn_ratio
    else:
        return 0.0


# ==================================================================
# 1A. NUMBA UDF ライブラリ (Part 3: 高次モーメント・その他の複雑な統計系)
# ==================================================================


@njit(fastmath=True, cache=True)
def statistical_moment_numba(arr: np.ndarray, moment: int) -> float:
    """
    高次モーメント (Numba JIT)
    """
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


# ==================================================================
# 1A. メイン計算関数 (Main Feature Calculation)
# ==================================================================


def calculate_1A_features(data: Dict[str, np.ndarray]) -> Dict[str, float]:
    """
    【カテゴリ1A: 基本統計・分布検定系】
    Numpyバッファを受け取り、V5の精鋭リストに残った37個のベース特徴量のみを計算して返す。
    ※ 戻り値のキーは、純化(neutralize)される前のベース名である必要があります。
    """
    features = {}

    # --- 内部ヘルパー関数 ---
    def _window(arr: np.ndarray, window: int) -> np.ndarray:
        """配列の末尾から `window` 個の要素を取得"""
        if window <= 0:
            return np.array([], dtype=arr.dtype)
        if window > len(arr):
            return arr
        return arr[-window:]

    def _last(arr: np.ndarray) -> float:
        """UDFが返した配列の最新値（末尾）を取得"""
        if len(arr) == 0:
            return np.nan
        return arr[-1]

    # バッファの取得
    close_data = data.get("close", np.array([], dtype=np.float64))
    volume_data = data.get("volume", np.array([], dtype=np.float64))

    # データ不足時のセーフガード
    if len(close_data) == 0:
        return features

    # ------------------------------------------------------------------
    # 1. Anderson-Darling Statistic
    features["e1a_anderson_darling_statistic_30"] = anderson_darling_numba(
        _window(close_data, 30)
    )

    # 2. Fast Basic Stabilization
    # _last を使用して配列からスカラー値を取り出す
    features["e1a_fast_basic_stabilization"] = _last(
        basic_stabilization_numba(_window(close_data, 100))
    )

    # 3-5. Fast Rolling Mean
    features["e1a_fast_rolling_mean_5"] = np.mean(_window(close_data, 5))
    features["e1a_fast_rolling_mean_10"] = np.mean(_window(close_data, 10))
    features["e1a_fast_rolling_mean_50"] = np.mean(_window(close_data, 50))

    # 6-9. Fast Rolling Std
    features["e1a_fast_rolling_std_5"] = np.std(_window(close_data, 5))
    features["e1a_fast_rolling_std_10"] = np.std(_window(close_data, 10))
    features["e1a_fast_rolling_std_20"] = np.std(_window(close_data, 20))
    features["e1a_fast_rolling_std_100"] = np.std(_window(close_data, 100))

    # 10-13. Fast Volume Mean
    features["e1a_fast_volume_mean_5"] = np.mean(_window(volume_data, 5))
    features["e1a_fast_volume_mean_10"] = np.mean(_window(volume_data, 10))
    features["e1a_fast_volume_mean_20"] = np.mean(_window(volume_data, 20))
    features["e1a_fast_volume_mean_50"] = np.mean(_window(volume_data, 50))

    # 14. Jarque-Bera Statistic
    features["e1a_jarque_bera_statistic_50"] = jarque_bera_statistic_numba(
        _window(close_data, 50)
    )

    # 15-17. Robust IQR & 20. Robust Q75
    w10 = _window(close_data, 10)
    if len(w10) > 0:
        q75_10, q25_10 = np.percentile(w10, [75, 25])
        features["e1a_robust_iqr_10"] = q75_10 - q25_10
    else:
        features["e1a_robust_iqr_10"] = np.nan

    w20 = _window(close_data, 20)
    if len(w20) > 0:
        q75_20, q25_20 = np.percentile(w20, [75, 25])
        features["e1a_robust_iqr_20"] = q75_20 - q25_20
    else:
        features["e1a_robust_iqr_20"] = np.nan

    w50 = _window(close_data, 50)
    if len(w50) > 0:
        q75_50, q25_50 = np.percentile(w50, [75, 25])
        features["e1a_robust_iqr_50"] = q75_50 - q25_50
        features["e1a_robust_q75_50"] = q75_50
    else:
        features["e1a_robust_iqr_50"] = np.nan
        features["e1a_robust_q75_50"] = np.nan

    # 18. Robust MAD
    features["e1a_robust_mad_20"] = mad_rolling_numba(_window(close_data, 20))

    # 19. Robust Median
    features["e1a_robust_median_50"] = np.median(_window(close_data, 50))

    # 21. Runs Test Statistic
    features["e1a_runs_test_statistic_30"] = runs_test_numba(_window(close_data, 30))

    # 22-24. Statistical CV (Coefficient of Variation)
    mean_10 = np.mean(_window(close_data, 10))
    std_10 = np.std(_window(close_data, 10))
    features["e1a_statistical_cv_10"] = std_10 / (mean_10 + 1e-10)

    mean_20 = np.mean(_window(close_data, 20))
    std_20 = np.std(_window(close_data, 20))
    features["e1a_statistical_cv_20"] = std_20 / (mean_20 + 1e-10)

    mean_50 = np.mean(_window(close_data, 50))
    std_50 = np.std(_window(close_data, 50))
    features["e1a_statistical_cv_50"] = std_50 / (mean_50 + 1e-10)

    # 25-26. Statistical Kurtosis
    features["e1a_statistical_kurtosis_20"] = statistical_kurtosis_numba(
        _window(close_data, 20)
    )
    features["e1a_statistical_kurtosis_50"] = statistical_kurtosis_numba(
        _window(close_data, 50)
    )

    # 27-33. Statistical Moments
    features["e1a_statistical_moment_5_20"] = statistical_moment_numba(
        _window(close_data, 20), 5
    )
    features["e1a_statistical_moment_5_50"] = statistical_moment_numba(
        _window(close_data, 50), 5
    )
    features["e1a_statistical_moment_6_20"] = statistical_moment_numba(
        _window(close_data, 20), 6
    )
    features["e1a_statistical_moment_6_50"] = statistical_moment_numba(
        _window(close_data, 50), 6
    )
    features["e1a_statistical_moment_7_20"] = statistical_moment_numba(
        _window(close_data, 20), 7
    )
    features["e1a_statistical_moment_7_50"] = statistical_moment_numba(
        _window(close_data, 50), 7
    )
    features["e1a_statistical_moment_8_50"] = statistical_moment_numba(
        _window(close_data, 50), 8
    )

    # 34-35. Statistical Skewness
    features["e1a_statistical_skewness_20"] = rolling_skew_numba(
        _window(close_data, 20)
    )
    features["e1a_statistical_skewness_50"] = rolling_skew_numba(
        _window(close_data, 50)
    )

    # 36. Statistical Variance
    features["e1a_statistical_variance_10"] = np.var(_window(close_data, 10))

    # 37. Von Neumann Ratio
    features["e1a_von_neumann_ratio_30"] = von_neumann_ratio_numba(
        _window(close_data, 30)
    )

    return features
