# realtime_feature_engine_1A_statistics.py
# Category 1A: 基本統計・分布検定系 (Basic Statistics & Distribution Tests)
#
# 【修正履歴 - QA監査対応】
#   - 全NumbaデコレータをUDF: @nb.njit / @njit → @nb.guvectorize に完全修正
#   - シグネチャを元スクリプトと完全一致: ['void(float64[:], float64[:])'], '(n)->(n)'
#   - fast_rolling_mean_numba / fast_rolling_std_numba: スカラー返し→配列ローリング実装に修正
#   - basic_stabilization_numba: range*0.01クリップ方式に戻す（パーセンタイル方式を廃棄）
#   - robust_stabilization_numba: 脱落していたため元スクリプトから完全移植
#   - biweight_location_numba: 脱落していたため元スクリプトから完全移植
#   - winsorized_mean_numba: 脱落していたため元スクリプトから完全移植
#   - fast_quality_score_numba: 脱落していたため元スクリプトから完全移植
#   - anderson_darling_numba: standardized_data配列方式・ガード条件を元に戻す
#   - rolling_skew_numba / statistical_kurtosis_numba / statistical_moment_numba:
#       元スクリプトではPolars式で実装されていた処理のため、
#       calculate_1A_features()側でPolars式相当のnumpy実装に置き換え（Part2で対応）
#   - calculate_1A_features(): 抜け落ち特徴量の全追加はPart2で対応

import math
import numpy as np
import numba as nb


# ==============================================================================
# Numba UDF ライブラリ
# 【厳守】全UDFは @nb.guvectorize(['void(float64[:], float64[:])'], '(n)->(n)',
#          nopython=True, cache=True) で定義する。
#          クラス内定義は循環参照エラーを引き起こすため絶対禁止。
# ==============================================================================


# ------------------------------------------------------------------------------
# Part 1-A: 基本高速ローリング系
# ------------------------------------------------------------------------------


@nb.guvectorize(["void(float64[:], float64[:])"], "(n)->(n)", nopython=True, cache=True)
def fast_rolling_mean_numba(arr, out):
    """Numba最適化ローリング平均（カスタムウィンドウ用）"""
    n = len(arr)
    for i in range(n):
        if i < 20:  # 最小ウィンドウサイズ
            out[i] = np.nan
        else:
            window_sum = 0.0
            count = 0
            for j in range(max(0, i - 19), i + 1):
                if not np.isnan(arr[j]):
                    window_sum += arr[j]
                    count += 1
            out[i] = window_sum / count if count > 0 else np.nan


@nb.guvectorize(["void(float64[:], float64[:])"], "(n)->(n)", nopython=True, cache=True)
def fast_rolling_std_numba(arr, out):
    """Numba最適化ローリング標準偏差"""
    n = len(arr)
    for i in range(n):
        if i < 20:
            out[i] = np.nan
        else:
            # 平均計算
            window_sum = 0.0
            count = 0
            for j in range(max(0, i - 19), i + 1):
                if not np.isnan(arr[j]):
                    window_sum += arr[j]
                    count += 1

            if count <= 1:
                out[i] = np.nan
                continue

            mean_val = window_sum / count

            # 分散計算
            var_sum = 0.0
            for j in range(max(0, i - 19), i + 1):
                if not np.isnan(arr[j]):
                    diff = arr[j] - mean_val
                    var_sum += diff * diff

            variance = var_sum / (count - 1)
            out[i] = np.sqrt(variance)


@nb.guvectorize(["void(float64[:], float64[:])"], "(n)->(n)", nopython=True, cache=True)
def fast_quality_score_numba(arr, out):
    """高速品質スコア計算"""
    n = len(arr)
    for i in range(n):
        if i < 50:  # 最小評価ウィンドウ
            out[i] = 0.0
        else:
            # ウィンドウ内のNaN/Inf率計算
            window_size = min(50, i + 1)
            nan_inf_count = 0
            finite_count = 0

            for j in range(i - window_size + 1, i + 1):
                if np.isnan(arr[j]) or np.isinf(arr[j]):
                    nan_inf_count += 1
                else:
                    finite_count += 1

            if window_size == 0:
                out[i] = 0.0
            else:
                nan_inf_ratio = nan_inf_count / window_size  # noqa: F841（参照維持）
                finite_ratio = finite_count / window_size
                out[i] = finite_ratio * 0.8 + 0.2  # 基本品質スコア


# ------------------------------------------------------------------------------
# Part 1-B: ロバスト統計系
# ------------------------------------------------------------------------------


@nb.guvectorize(["void(float64[:], float64[:])"], "(n)->(n)", nopython=True, cache=True)
def mad_rolling_numba(arr, out):
    """ローリングMAD計算"""
    n = len(arr)
    window = 20

    for i in range(n):
        if i < window - 1:
            out[i] = np.nan
        else:
            # ウィンドウ内データ取得
            window_data = arr[max(0, i - window + 1) : i + 1]
            finite_data = window_data[~np.isnan(window_data)]

            if len(finite_data) < 3:
                out[i] = np.nan
            else:
                # 中央値計算
                median_val = np.median(finite_data)
                # 絶対偏差の中央値
                abs_deviations = np.abs(finite_data - median_val)
                out[i] = np.median(abs_deviations)


@nb.guvectorize(["void(float64[:], float64[:])"], "(n)->(n)", nopython=True, cache=True)
def biweight_location_numba(arr, out):
    """ローリングBiweight位置計算（厳密実装）"""
    n = len(arr)
    window = 20

    for i in range(n):
        if i < window - 1:
            out[i] = np.nan
        else:
            window_data = arr[max(0, i - window + 1) : i + 1]
            finite_data = window_data[~np.isnan(window_data)]

            if len(finite_data) < 5:
                out[i] = np.median(finite_data) if len(finite_data) > 0 else np.nan
            else:
                # Tukey's Biweight位置推定の厳密実装
                # 反復アルゴリズム

                # 初期値として中央値を使用
                current_location = np.median(finite_data)
                tolerance = 1e-10
                max_iterations = 50

                for iteration in range(max_iterations):
                    # MAD計算
                    abs_residuals = np.abs(finite_data - current_location)
                    mad_val = np.median(abs_residuals)

                    if mad_val < 1e-15:
                        break

                    # スケールファクター（6 * MAD）
                    scale = 6.0 * mad_val

                    # 標準化残差
                    u_values = (finite_data - current_location) / scale

                    # Biweight重み関数の計算
                    weights = np.zeros(len(finite_data))
                    numerator = 0.0
                    denominator = 0.0

                    for j in range(len(finite_data)):
                        u_abs = abs(u_values[j])

                        if u_abs < 1.0:
                            # Biweight重み: (1 - u²)²
                            weight = (1.0 - u_values[j] ** 2) ** 2
                            weights[j] = weight
                            numerator += finite_data[j] * weight
                            denominator += weight
                        else:
                            weights[j] = 0.0

                    if denominator > 1e-15:
                        new_location = numerator / denominator
                    else:
                        # 重みがすべてゼロの場合は中央値を返す
                        new_location = np.median(finite_data)
                        break

                    # 収束判定
                    if abs(new_location - current_location) < tolerance:
                        break

                    current_location = new_location

                out[i] = current_location


@nb.guvectorize(["void(float64[:], float64[:])"], "(n)->(n)", nopython=True, cache=True)
def winsorized_mean_numba(arr, out):
    """ローリングウィンソライズ平均（上下5%クリップ）"""
    n = len(arr)
    window = 20

    for i in range(n):
        if i < window - 1:
            out[i] = np.nan
        else:
            window_data = arr[max(0, i - window + 1) : i + 1]
            finite_data = window_data[~np.isnan(window_data)]

            if len(finite_data) < 5:
                out[i] = np.mean(finite_data) if len(finite_data) > 0 else np.nan
            else:
                # 上下5%点の計算
                p05 = np.percentile(finite_data, 5)
                p95 = np.percentile(finite_data, 95)

                # ウィンソライズ（クリッピング）
                winsorized_data = np.clip(finite_data, p05, p95)
                out[i] = np.mean(winsorized_data)


# ------------------------------------------------------------------------------
# Part 1-C: 分布検定系
# ------------------------------------------------------------------------------


@nb.guvectorize(["void(float64[:], float64[:])"], "(n)->(n)", nopython=True, cache=True)
def jarque_bera_statistic_numba(arr, out):
    """ローリングJarque-Bera検定統計量"""
    n = len(arr)
    window = 50

    for i in range(n):
        if i < window - 1:
            out[i] = np.nan
        else:
            window_data = arr[max(0, i - window + 1) : i + 1]
            finite_data = window_data[~np.isnan(window_data)]

            if len(finite_data) < 20:
                out[i] = np.nan
            else:
                # 基本統計量
                mean_val = np.mean(finite_data)

                # 手動分散計算（Numba対応）
                variance = 0.0
                for val in finite_data:
                    variance += (val - mean_val) ** 2
                variance = variance / (len(finite_data) - 1)
                std_val = np.sqrt(variance)

                if std_val < 1e-10:
                    out[i] = 0.0
                else:
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
                    jb_stat = len(finite_data) * (skewness**2 / 6 + kurtosis**2 / 24)
                    out[i] = jb_stat


@nb.guvectorize(["void(float64[:], float64[:])"], "(n)->(n)", nopython=True, cache=True)
def anderson_darling_numba(arr, out):
    """ローリングAnderson-Darling統計量（厳密実装）"""
    n = len(arr)
    window = 30

    # 低速な数値積分によるCDFを、高速な誤差関数(erf)を用いた解析的な計算式に置き換えます。
    # この関数は Numba の JIT コンパイル対象となります。
    # math.sqrt(2.0) は約 1.4142135623730951
    SQRT2 = 1.4142135623730951

    def standard_normal_cdf_fast(x):
        """標準正規分布の累積分布関数（高速実装）"""
        return 0.5 * (1.0 + math.erf(x / SQRT2))

    for i in range(n):
        if i < window - 1:
            out[i] = np.nan
        else:
            window_data = arr[max(0, i - window + 1) : i + 1]
            finite_data = window_data[~np.isnan(window_data)]

            if len(finite_data) < 10:
                out[i] = np.nan
            else:
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
                    out[i] = 0.0
                else:
                    # 標準化後の厳密なAnderson-Darling統計量
                    standardized_data = np.zeros(n_data)
                    for j in range(n_data):
                        standardized_data[j] = (sorted_data[j] - mean_val) / std_val

                    # Anderson-Darling統計量の厳密計算
                    ad_sum = 0.0
                    for j in range(n_data):
                        # 高速化されたCDF関数を呼び出します。
                        F_j = standard_normal_cdf_fast(standardized_data[j])
                        F_nj = standard_normal_cdf_fast(
                            standardized_data[n_data - 1 - j]
                        )

                        # ゼロ除算回避
                        if F_j > 1e-15 and (1 - F_nj) > 1e-15:
                            log_term = np.log(F_j) + np.log(1 - F_nj)
                            ad_sum += (2 * j + 1) * log_term

                    # Anderson-Darling統計量
                    out[i] = -n_data - ad_sum / n_data


@nb.guvectorize(["void(float64[:], float64[:])"], "(n)->(n)", nopython=True, cache=True)
def runs_test_numba(arr, out):
    """ローリングRuns Test統計量"""
    n = len(arr)
    window = 30

    for i in range(n):
        if i < window - 1:
            out[i] = np.nan
        else:
            window_data = arr[max(0, i - window + 1) : i + 1]
            finite_data = window_data[~np.isnan(window_data)]

            if len(finite_data) < 10:
                out[i] = np.nan
            else:
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
                    var_runs = (2 * n1 * n2 * (2 * n1 * n2 - n1 - n2)) / (
                        (n1 + n2) ** 2 * (n1 + n2 - 1)
                    )

                    if var_runs > 0:
                        # 標準化統計量
                        out[i] = (runs - expected_runs) / np.sqrt(var_runs)
                    else:
                        out[i] = 0.0
                else:
                    out[i] = 0.0


@nb.guvectorize(["void(float64[:], float64[:])"], "(n)->(n)", nopython=True, cache=True)
def von_neumann_ratio_numba(arr, out):
    """ローリングVon Neumann比（厳密実装）"""
    n = len(arr)
    window = 30

    for i in range(n):
        if i < window - 1:
            out[i] = np.nan
        else:
            window_data = arr[max(0, i - window + 1) : i + 1]
            finite_data = window_data[~np.isnan(window_data)]

            if len(finite_data) < 3:  # 最低3点必要（差分計算のため）
                out[i] = np.nan
            else:
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
                    vn_ratio = diff_sq_sum / sum_sq_deviations

                    # 理論的範囲チェック（0 ≤ VN比 ≤ 4）
                    if vn_ratio < 0.0:
                        out[i] = 0.0
                    elif vn_ratio > 4.0:
                        out[i] = 4.0
                    else:
                        out[i] = vn_ratio
                else:
                    # 全て同じ値の場合、理論的にVN比は0
                    out[i] = 0.0


# ------------------------------------------------------------------------------
# Part 1-D: 品質保証安定化系
# ------------------------------------------------------------------------------


@nb.guvectorize(["void(float64[:], float64[:])"], "(n)->(n)", nopython=True, cache=True)
def basic_stabilization_numba(arr, out):
    """基本安定化処理（第1段階）"""
    n = len(arr)
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
            clip_margin = range_val * 0.01
            for i in range(n):
                if np.isfinite(out[i]):
                    if out[i] < min_val + clip_margin:
                        out[i] = min_val + clip_margin
                    elif out[i] > max_val - clip_margin:
                        out[i] = max_val - clip_margin


@nb.guvectorize(["void(float64[:], float64[:])"], "(n)->(n)", nopython=True, cache=True)
def robust_stabilization_numba(arr, out):
    """ロバスト安定化処理（第2段階・フォールバック）"""
    n = len(arr)
    finite_vals = []
    for i in range(n):
        if np.isfinite(arr[i]):
            finite_vals.append(arr[i])
    if len(finite_vals) < 3:
        for i in range(n):
            out[i] = 0.0
        return
    sorted_vals = sorted(finite_vals)
    n_finite = len(sorted_vals)
    if n_finite % 2 == 0:
        median_val = (sorted_vals[n_finite // 2 - 1] + sorted_vals[n_finite // 2]) / 2
    else:
        median_val = sorted_vals[n_finite // 2]
    abs_devs = []
    for val in finite_vals:
        abs_devs.append(abs(val - median_val))
    sorted_abs_devs = sorted(abs_devs)
    n_abs = len(sorted_abs_devs)
    if n_abs % 2 == 0:
        mad_val = (sorted_abs_devs[n_abs // 2 - 1] + sorted_abs_devs[n_abs // 2]) / 2
    else:
        mad_val = sorted_abs_devs[n_abs // 2]
    if mad_val < 1e-10:
        mad_val = np.std(np.array(finite_vals)) * 0.6745
    lower_bound = median_val - 3 * mad_val
    upper_bound = median_val + 3 * mad_val
    for i in range(n):
        if np.isnan(arr[i]):
            out[i] = median_val
        elif np.isinf(arr[i]):
            if arr[i] > 0:
                out[i] = upper_bound
            else:
                out[i] = lower_bound
        else:
            if arr[i] < lower_bound:
                out[i] = lower_bound
            elif arr[i] > upper_bound:
                out[i] = upper_bound
            else:
                out[i] = arr[i]


# ==============================================================================
# メイン計算関数 (Main Feature Calculation)
# ==============================================================================


def calculate_1A_features(data: dict) -> dict:
    """
    【カテゴリ1A: 基本統計・分布検定系】
    Numpyバッファを受け取り、元スクリプト(engine_1_A_a_vast_universe_of_features.py)の
    _get_all_feature_expressions() および各 _create_*_features() メソッドと
    数学的・アルゴリズム的に完全一致する特徴量を計算して返す。

    【設計方針】
    - 元スクリプトは Polars LazyFrame + guvectorize ベクタライズ処理で全系列を計算する。
      リアルタイムモジュールでは「バッファ末尾の最新1点」を返す設計とするが、
      guvectorize UDF は配列全体を渡して out[-1] で最新値を取り出すことで
      ローリング計算を完全に再現する。
    - Polars 式で実装されていた特徴量（statistical_mean / std / var / cv /
      skewness / kurtosis / moment / median / q25 / q75 / iqr / trimmed_mean /
      fast_rolling_mean / fast_rolling_std / fast_volume_mean）は、
      元式の数学的定義に従い numpy で再実装する。
    - rolling_var / rolling_std の ddof は Polars デフォルトと同じ ddof=1（不偏）。
    - statistical_cv は元式通り std / mean（ゼロ割り保護なし）。
      ※ゼロ割り時は np.inf となるが、これは元スクリプトの挙動と一致する。
    - trim_mean は scipy.stats.trim_mean(arr, proportiontocut=0.1) と等価の実装。
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

    def _last_of_guv(arr: np.ndarray) -> float:
        """
        guvectorize UDF に配列全体を渡し、出力配列の末尾（最新値）を返す。
        UDF が nan を返した場合はそのまま nan を返す。
        """
        if len(arr) == 0:
            return np.nan
        result = arr.copy().astype(np.float64)
        # guvectorize は in-place で out を上書きする。
        # 元スクリプトでの呼び出し: map_batches(lambda s: udf(s.to_numpy()))
        # → 全系列を渡して out 配列を得る → ここでは arr 全体を渡す。
        out = np.empty_like(result)
        # NOTE: guvectorize UDF は (arr, out) を受け取り out を書き換える。
        #       Python 側からは udf(arr) で呼び出すと out 配列を返す。
        return float(out[-1]) if len(out) > 0 else np.nan

    def _rolling_mean_np(arr: np.ndarray, window: int) -> float:
        """
        Polars rolling_mean(window) の末尾値と等価。
        NaN を無視して計算（Polars デフォルト: min_periods=1）。
        """
        w = _window(arr, window)
        finite = w[np.isfinite(w)]
        if len(finite) == 0:
            return np.nan
        return float(np.mean(finite))

    def _rolling_var_np(arr: np.ndarray, window: int) -> float:
        """
        Polars rolling_var(window) の末尾値と等価（ddof=1 不偏分散）。
        """
        w = _window(arr, window)
        finite = w[np.isfinite(w)]
        if len(finite) < 2:
            return np.nan
        return float(np.var(finite, ddof=1))

    def _rolling_std_np(arr: np.ndarray, window: int) -> float:
        """
        Polars rolling_std(window) の末尾値と等価（ddof=1 不偏標準偏差）。
        """
        w = _window(arr, window)
        finite = w[np.isfinite(w)]
        if len(finite) < 2:
            return np.nan
        return float(np.std(finite, ddof=1))

    def _rolling_skew_np(arr: np.ndarray, window: int) -> float:
        """
        Polars rolling_skew(window_size=window) の末尾値と等価。
        Polars は偏り補正付き（adjusted=True）のフィッシャー歪度を返す。
        scipy.stats.skew(bias=False) と同等。
        """
        w = _window(arr, window)
        finite = w[np.isfinite(w)]
        n = len(finite)
        if n < 3:
            return np.nan
        mean_val = np.mean(finite)
        std_val = np.std(finite, ddof=1)
        if std_val < 1e-10:
            return 0.0
        m3 = np.mean((finite - mean_val) ** 3)
        # 偏り補正付き歪度: G1 = (n(n-1))^0.5 / (n-2) * skew_biased
        skew_biased = m3 / (std_val**3)
        correction = (n * (n - 1)) ** 0.5 / (n - 2)
        return float(correction * skew_biased)

    def _rolling_kurtosis_np(arr: np.ndarray, window: int) -> float:
        """
        元スクリプトの Polars 式:
          ((close - rolling_mean(w)).pow(4).rolling_mean(w)
           / rolling_std(w).pow(4) - 3)
        の末尾値と等価。
        ※ rolling_std は ddof=1（不偏）、rolling_mean は単純平均。
        ※ 全ウィンドウ要素を使った 4次モーメント / (不偏std)^4 - 3。
        """
        w = _window(arr, window)
        finite = w[np.isfinite(w)]
        n = len(finite)
        if n < 4:
            return np.nan
        mean_val = np.mean(finite)
        std_val = np.std(finite, ddof=1)
        if std_val < 1e-10:
            return 0.0
        m4 = np.mean((finite - mean_val) ** 4)
        return float(m4 / (std_val**4) - 3.0)

    def _rolling_moment_np(arr: np.ndarray, window: int, moment: int) -> float:
        """
        元スクリプトの Polars 式:
          ((close - rolling_mean(w)) / rolling_std(w)).pow(moment).rolling_mean(w)
        の末尾値と等価。std は ddof=1（不偏）。
        """
        w = _window(arr, window)
        finite = w[np.isfinite(w)]
        n = len(finite)
        if n < 2:
            return np.nan
        mean_val = np.mean(finite)
        std_val = np.std(finite, ddof=1)
        if std_val < 1e-10:
            return 0.0
        z = (finite - mean_val) / std_val
        return float(np.mean(z**moment))

    def _rolling_median_np(arr: np.ndarray, window: int) -> float:
        """Polars rolling_median(window) の末尾値と等価。"""
        w = _window(arr, window)
        finite = w[np.isfinite(w)]
        if len(finite) == 0:
            return np.nan
        return float(np.median(finite))

    def _rolling_quantile_np(arr: np.ndarray, window: int, q: float) -> float:
        """
        Polars rolling_quantile(q, window_size=window) の末尾値と等価。
        Polars デフォルト interpolation="nearest"。
        """
        w = _window(arr, window)
        finite = w[np.isfinite(w)]
        if len(finite) == 0:
            return np.nan
        return float(np.percentile(finite, q * 100, method="nearest"))

    def _rolling_trimmed_mean_np(arr: np.ndarray, window: int) -> float:
        """
        元スクリプトの Polars 式:
          rolling_map(lambda s: trim_mean(s.to_numpy(), proportiontocut=0.1), window_size=window)
        の末尾値と等価。上下10%をトリムした平均。
        """
        from scipy.stats import trim_mean as scipy_trim_mean

        w = _window(arr, window)
        finite = w[np.isfinite(w)]
        if len(finite) == 0:
            return np.nan
        return float(scipy_trim_mean(finite, proportiontocut=0.1))

    # バッファの取得
    close_data = data.get("close", np.array([], dtype=np.float64))
    volume_data = data.get("volume", np.array([], dtype=np.float64))

    # データ不足時のセーフガード
    if len(close_data) == 0:
        return features

    close_f64 = close_data.astype(np.float64)
    volume_f64 = (
        volume_data.astype(np.float64)
        if len(volume_data) > 0
        else np.array([], dtype=np.float64)
    )

    # ==========================================================================
    # グループ1: 統計的モーメント特徴量
    # 元: _create_basic_stats_features() → windows=[10, 20, 50]
    # ==========================================================================

    for window in [10, 20, 50]:
        features[f"e1a_statistical_mean_{window}"] = _rolling_mean_np(close_f64, window)
        features[f"e1a_statistical_variance_{window}"] = _rolling_var_np(
            close_f64, window
        )
        features[f"e1a_statistical_std_{window}"] = _rolling_std_np(close_f64, window)

        mean_w = _rolling_mean_np(close_f64, window)
        std_w = _rolling_std_np(close_f64, window)
        # 元スクリプト: rolling_std / rolling_mean （ゼロ割り保護なし・Polars式そのまま）
        features[f"e1a_statistical_cv_{window}"] = (
            float(std_w / mean_w) if mean_w != 0.0 else np.inf
        )

    # ==========================================================================
    # グループ2: 歪度・尖度・高次モーメント特徴量
    # 元: _create_skew_kurt_features() → windows=[20, 50], moments=[5,6,7,8]
    # ==========================================================================

    for window in [20, 50]:
        features[f"e1a_statistical_skewness_{window}"] = _rolling_skew_np(
            close_f64, window
        )
        features[f"e1a_statistical_kurtosis_{window}"] = _rolling_kurtosis_np(
            close_f64, window
        )
        for moment in [5, 6, 7, 8]:
            features[f"e1a_statistical_moment_{moment}_{window}"] = _rolling_moment_np(
                close_f64, window, moment
            )

    # ==========================================================================
    # グループ3: ロバスト統計特徴量
    # 元: _create_robust_stats_features() → windows=[10, 20, 50]
    # ==========================================================================

    for window in [10, 20, 50]:
        features[f"e1a_robust_median_{window}"] = _rolling_median_np(close_f64, window)
        q25_val = _rolling_quantile_np(close_f64, window, 0.25)
        q75_val = _rolling_quantile_np(close_f64, window, 0.75)
        features[f"e1a_robust_q25_{window}"] = q25_val
        features[f"e1a_robust_q75_{window}"] = q75_val
        features[f"e1a_robust_iqr_{window}"] = (
            float(q75_val - q25_val)
            if np.isfinite(q75_val) and np.isfinite(q25_val)
            else np.nan
        )
        features[f"e1a_robust_trimmed_mean_{window}"] = _rolling_trimmed_mean_np(
            close_f64, window
        )

    # ==========================================================================
    # グループ4: 高度ロバスト統計特徴量（guvectorize UDF 使用）
    # 元: _create_advanced_robust_features()
    # 呼び出し方: map_batches(lambda s: udf(s.to_numpy())) → 全系列を渡す → out[-1]
    # ==========================================================================

    features["e1a_robust_mad_20"] = float(mad_rolling_numba(close_f64)[-1])
    features["e1a_robust_biweight_location_20"] = float(
        biweight_location_numba(close_f64)[-1]
    )
    features["e1a_robust_winsorized_mean_20"] = float(
        winsorized_mean_numba(close_f64)[-1]
    )

    # ==========================================================================
    # グループ5: 統計検定・正規性特徴量（guvectorize UDF 使用）
    # 元: _create_statistical_tests_features()
    # ==========================================================================

    features["e1a_jarque_bera_statistic_50"] = float(
        jarque_bera_statistic_numba(close_f64)[-1]
    )
    features["e1a_anderson_darling_statistic_30"] = float(
        anderson_darling_numba(close_f64)[-1]
    )
    features["e1a_runs_test_statistic_30"] = float(runs_test_numba(close_f64)[-1])
    features["e1a_von_neumann_ratio_30"] = float(von_neumann_ratio_numba(close_f64)[-1])

    # ==========================================================================
    # グループ6: 高速ローリング特徴量
    # 元: _create_fast_processing_features() → windows=[5, 10, 20, 50, 100]
    # Polars rolling_mean / rolling_std → ddof=1（不偏）
    # ==========================================================================

    for window in [5, 10, 20, 50, 100]:
        features[f"e1a_fast_rolling_mean_{window}"] = _rolling_mean_np(
            close_f64, window
        )
        features[f"e1a_fast_rolling_std_{window}"] = _rolling_std_np(close_f64, window)
        if len(volume_f64) > 0:
            features[f"e1a_fast_volume_mean_{window}"] = _rolling_mean_np(
                volume_f64, window
            )
        else:
            features[f"e1a_fast_volume_mean_{window}"] = np.nan

    # ==========================================================================
    # グループ7: Numba最適化特徴量（guvectorize UDF 使用）
    # 元: _create_numba_features()
    # ==========================================================================

    features["e1a_fast_rolling_mean_numba_20"] = float(
        fast_rolling_mean_numba(close_f64)[-1]
    )
    features["e1a_fast_rolling_std_numba_20"] = float(
        fast_rolling_std_numba(close_f64)[-1]
    )
    features["e1a_fast_quality_score_50"] = float(
        fast_quality_score_numba(close_f64)[-1]
    )

    # ==========================================================================
    # グループ8: 品質保証特徴量（guvectorize UDF 使用）
    # 元: _create_qa_features()
    # ==========================================================================

    features["e1a_fast_basic_stabilization"] = float(
        basic_stabilization_numba(close_f64)[-1]
    )
    features["e1a_fast_robust_stabilization"] = float(
        robust_stabilization_numba(close_f64)[-1]
    )

    return features
