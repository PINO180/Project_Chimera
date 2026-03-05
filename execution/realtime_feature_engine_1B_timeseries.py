"""
realtime_feature_engine_1B_timeseries.py  【QA修正版 v1.0】

Project Cimera V5 - 特徴量エンジンモジュール 【1B: 時系列・分布系】
生き残り特徴量（28個）の計算に特化した純粋な数学的計算モジュール。
不要な関数（デッドコード）は完全にパージ済。

【修正履歴】
  QA修正 v1.0 (2026-03-05):
    - [差異①] t分布_自由度_udf: std_ret ガード条件を元スクリプト準拠 (<= 0) に巻き戻し
    - [差異②] adf_統計量_udf: se_beta ガードを元スクリプト準拠 (se_beta > 0) に巻き戻し、
               mse/XtX_inv の事前チェック追加を削除
    - [差異③] phillips_perron_統計量_udf: se_beta ガードを元スクリプト準拠 (se_beta > 0) に巻き戻し
    - [差異④] kpss_統計量_udf: denom の事前チェックを削除、sse ガードを元スクリプト準拠 (> 0) に巻き戻し
    - [差異⑤] kalman_状態推定_udf: R==0 ガードを削除し元スクリプト準拠に巻き戻し
    - [差異⑥] holt_winters_レベル/トレンド_udf: (1.0 - alpha) 等の型明示を (1 - alpha) に巻き戻し
    - [差異⑧] zscore_20/50: np.std(ddof=0) → np.std(ddof=1) に修正し Polars rolling_std 準拠に統一
    - [差異⑨] bollinger_lower/upper_50: np.std(ddof=0) → np.std(ddof=1) に修正
    - [差異⑩] volatility_20: np.std(ddof=0) → np.std(ddof=1) に修正
"""

import numpy as np
import numba as nb
from numba import njit
from typing import Dict
import logging

# モジュール専用のロガーを設定
logger = logging.getLogger("ProjectForge.FeatureEngine.1B")

# ==================================================================
# 1. 基本統計・分布系 Numba JIT 関数群
# ==================================================================


@nb.njit(fastmath=True, cache=True)
def t分布_自由度_udf(returns: np.ndarray) -> float:
    """
    t分布の自由度パラメータ推定
    尖度ベース推定によるモーメント法使用
    """
    if len(returns) < 10:
        return np.nan

    finite_returns = returns[np.isfinite(returns)]
    if len(finite_returns) < 10:
        return np.nan

    # リターン標準化
    mean_ret = np.mean(finite_returns)
    std_ret = np.std(finite_returns)

    # ★修正①: 元スクリプト準拠 (<= 0) に巻き戻し
    if std_ret <= 0:
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


@nb.njit(fastmath=True, cache=True)
def t分布_尺度_udf(returns: np.ndarray) -> float:
    """
    t分布の尺度パラメータ推定
    尺度パラメータは分布のスプレッドを表す
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


# ==================================================================
# 2. 時系列・検定系 Numba JIT 関数群
# ==================================================================


@nb.njit(fastmath=True, cache=True)
def adf_統計量_udf(prices: np.ndarray) -> float:
    """
    拡張ディッキー・フラー検定統計量計算
    帰無仮説：系列が単位根を持つ（非定常）を検定
    低い値は単位根に対するより強い証拠を示す
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
        se_beta = np.sqrt(mse * XtX_inv[1, 1])

        # ★修正②: 元スクリプト準拠 (se_beta > 0) に巻き戻し
        if se_beta > 0:
            adf_stat = beta[1] / se_beta
        else:
            adf_stat = np.nan

    except:
        adf_stat = np.nan

    return adf_stat


@nb.njit(fastmath=True, cache=True)
def phillips_perron_統計量_udf(prices: np.ndarray) -> float:
    """
    フィリップス・ペロン検定統計量計算
    異分散性修正による単位根のノンパラメトリック検定
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
        se_beta = np.sqrt(sigma2 * XtX_inv[1, 1])

        # ★修正③: 元スクリプト準拠 (se_beta > 0) に巻き戻し
        if se_beta > 0:
            pp_stat = beta[1] / se_beta
        else:
            pp_stat = np.nan

    except:
        pp_stat = np.nan

    return pp_stat


@nb.njit(fastmath=True, cache=True)
def kpss_統計量_udf(prices: np.ndarray) -> float:
    """
    KPSS検定統計量計算
    帰無仮説：系列がトレンド周りで定常
    高い値は定常性に対するより強い証拠を示す
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

        # ★修正④: denom の事前チェックを削除し元スクリプト準拠に巻き戻し
        # 傾きと切片計算
        beta = (n * sum_ty - sum_t * sum_y) / (n * sum_t2 - sum_t**2)
        alpha = (sum_y - beta * sum_t) / n

        # デトレンドされた系列
        detrended = finite_prices - (alpha + beta * t)

        # 部分和計算
        cumsum = np.cumsum(detrended)

        # KPSS統計量
        sse = np.sum(detrended**2) / n  # 誤差分散

        # ★修正④続き: sse ガードを元スクリプト準拠 (> 0) に巻き戻し
        if sse > 0:
            kpss_stat = np.sum(cumsum**2) / (n**2 * sse)
        else:
            kpss_stat = np.nan

    except:
        kpss_stat = np.nan

    return kpss_stat


# ==================================================================
# 3. 高度な時系列モデリング Numba JIT 関数群
# ==================================================================


@nb.njit(fastmath=True, cache=True)
def holt_winters_レベル_udf(prices: np.ndarray) -> float:
    """
    ホルト・ウィンターズ指数平滑からレベル成分を抽出
    レベルは時系列の平滑化された局所平均を表す
    """
    if len(prices) < 10:
        return np.nan

    finite_prices = prices[np.isfinite(prices)]
    if len(finite_prices) < 10:
        return np.nan

    # レベル用適切指数平滑（α = 0.3）
    alpha = 0.3
    level = finite_prices[0]  # 最初の観測値で初期化

    # ★修正⑥: (1 - alpha) 表記を元スクリプト準拠に巻き戻し
    for i in range(1, len(finite_prices)):
        level = alpha * finite_prices[i] + (1 - alpha) * level

    return level


@nb.njit(fastmath=True, cache=True)
def holt_winters_トレンド_udf(prices: np.ndarray) -> float:
    """
    ホルト・ウィンターズ指数平滑からトレンド成分を抽出
    トレンドは時系列の平滑化された局所傾きを表す
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

    # ★修正⑥: (1 - alpha)/(1 - beta) 表記を元スクリプト準拠に巻き戻し
    for i in range(1, len(finite_prices)):
        new_level = alpha * finite_prices[i] + (1 - alpha) * (level + trend)
        new_trend = beta * (new_level - level) + (1 - beta) * trend
        level = new_level
        trend = new_trend

    return trend


@nb.njit(fastmath=True, cache=True)
def arima_残差分散_udf(prices: np.ndarray) -> float:
    """
    ARIMA(1,1,1)モデルからの残差分散計算
    差分付適切ARIMA残差計算
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
    if n * sum_x2 - sum_x**2 != 0:
        phi = (n * sum_xy - sum_x * sum_y) / (n * sum_x2 - sum_x**2)
    else:
        phi = 0.0

    # 残差計算
    intercept = (sum_y - phi * sum_x) / n
    residuals = y - (intercept + phi * x)

    # 残差分散（不偏推定量）
    residual_variance = np.sum(residuals**2) / (n - 2)

    return residual_variance


@nb.njit(fastmath=True, cache=True)
def kalman_状態推定_udf(prices: np.ndarray) -> float:
    """
    価格レベル用カルマンフィルタ状態推定
    ノイズ低減による基礎となる真の価格レベル推定
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
    # ★修正⑤: R==0 ガードを削除し元スクリプト準拠に巻き戻し
    if len(finite_prices) > 1:
        diff_var = np.var(np.diff(finite_prices))
        obs_var = np.var(finite_prices)
        Q = max(diff_var, obs_var * 0.01)  # プロセスノイズ
        R = obs_var * 0.1  # 観測ノイズ（信号の10%）
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
        P = (1 - K) * P_pred  # 更新状態分散

    return x


@nb.njit(fastmath=True, cache=True)
def lowess_適合値_udf(prices: np.ndarray) -> float:
    """
    LOWESS（局所重み付け散布図平滑）適合値
    トレンド推定のための局所回帰
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
    x_neighbors = neighbor_indices.astype(np.float64)
    y_neighbors = finite_prices[neighbor_indices]

    # 三次重み計算
    max_dist = np.max(distances[neighbor_indices])
    if max_dist > 0:
        weights = np.zeros(h)
        for i in range(h):
            u = distances[neighbor_indices[i]] / max_dist
            if u < 1.0:
                # 三次重み関数
                weights[i] = (1 - u**3) ** 3
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


@nb.njit(fastmath=True, cache=True)
def theil_sen_傾き_udf(prices: np.ndarray) -> float:
    """
    頑健傾き推定のためのタイル・セン推定量
    外れ値に耐性のある全ペアワイズ傾きの中央値
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
# 4. メイン特徴量計算関数 (1B: 時系列・分布系)
# ==================================================================


def calculate_1B_features(data: Dict[str, np.ndarray]) -> Dict[str, float]:
    """
    Project Cimera V5: 1Bモジュール (時系列・分布系) の特徴量計算関数
    生き残りリストに指定された28個の特徴量のみを計算し返却する。

    Args:
        data: "open", "high", "low", "close", "volume" の1次元Numpy配列を含む辞書
              (配列はすでに必要なルックバック分が確保されている前提)

    Returns:
        Dict[str, float]: ベース特徴量名と計算値のペア
    """
    features: Dict[str, float] = {}

    # --- ヘルパー関数 ---
    def _window(arr: np.ndarray, window: int) -> np.ndarray:
        """配列の末尾から指定した要素数を安全に取得"""
        if window <= 0:
            return np.array([], dtype=arr.dtype)
        if window > len(arr):
            return arr
        return arr[-window:]

    def _last(arr: np.ndarray) -> float:
        """配列の最新値（末尾）を取得"""
        if len(arr) == 0:
            return np.nan
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

    # --- 共通計算データの準備 ---
    close_arr = data["close"]
    high_arr = data["high"]
    low_arr = data["low"]

    if len(close_arr) == 0:
        logger.warning("1Bモジュール: 入力データ(close)が空です。")
        return features

    close_pct = _pct(close_arr)
    current_close = _last(close_arr)

    # 事前計算: 20期間、50期間の平均と標準偏差 (ボリンジャーバンド・Zスコア用)
    # ★修正⑧⑨⑩: Polars rolling_std に合わせ ddof=1 (不偏標準偏差) を使用
    window_20 = _window(close_arr, 20)
    window_50 = _window(close_arr, 50)

    mean_20 = np.mean(window_20) if len(window_20) > 0 else np.nan
    std_20 = (
        np.std(window_20, ddof=1) if len(window_20) > 1 else np.nan
    )  # ddof=1: Polars rolling_std 準拠
    mean_50 = np.mean(window_50) if len(window_50) > 0 else np.nan
    std_50 = (
        np.std(window_50, ddof=1) if len(window_50) > 1 else np.nan
    )  # ddof=1: Polars rolling_std 準拠

    # ==================================================================
    # 生き残りリスト(28個)の厳格なマッピング
    # ==================================================================

    # 1. ADF検定 (Augmented Dickey-Fuller)
    features["e1b_adf_statistic_100"] = adf_統計量_udf(_window(close_arr, 100))
    features["e1b_adf_statistic_50"] = adf_統計量_udf(_window(close_arr, 50))

    # 2. ARIMA残差分散
    features["e1b_arima_residual_var_100"] = arima_残差分散_udf(_window(close_arr, 100))
    features["e1b_arima_residual_var_50"] = arima_残差分散_udf(_window(close_arr, 50))

    # 3. ボリンジャーバンド (期間50、標準偏差2.0)
    # ★修正⑨: Polars (mean + 2*rolling_std[ddof=1]) 準拠
    features["e1b_bollinger_lower_50"] = mean_50 - 2.0 * std_50
    features["e1b_bollinger_upper_50"] = mean_50 + 2.0 * std_50

    # 4. Holt-Winters 平滑化
    features["e1b_holt_level_100"] = holt_winters_レベル_udf(_window(close_arr, 100))
    features["e1b_holt_level_50"] = holt_winters_レベル_udf(_window(close_arr, 50))
    features["e1b_holt_trend_100"] = holt_winters_トレンド_udf(_window(close_arr, 100))
    features["e1b_holt_trend_50"] = holt_winters_トレンド_udf(_window(close_arr, 50))

    # 5. カルマンフィルタ
    features["e1b_kalman_state_100"] = kalman_状態推定_udf(_window(close_arr, 100))

    # 6. KPSS検定
    features["e1b_kpss_statistic_100"] = kpss_統計量_udf(_window(close_arr, 100))
    features["e1b_kpss_statistic_50"] = kpss_統計量_udf(_window(close_arr, 50))

    # 7. LOWESS平滑化
    features["e1b_lowess_fitted_100"] = lowess_適合値_udf(_window(close_arr, 100))
    features["e1b_lowess_fitted_50"] = lowess_適合値_udf(_window(close_arr, 50))

    # 8. Phillips-Perron検定
    features["e1b_pp_statistic_100"] = phillips_perron_統計量_udf(
        _window(close_arr, 100)
    )

    # 9. 価格変動・レンジ
    features["e1b_price_change"] = _last(close_pct)
    if len(high_arr) > 0 and len(low_arr) > 0:
        features["e1b_price_range"] = _last(high_arr) - _last(low_arr)
    else:
        features["e1b_price_range"] = np.nan

    # 10. ローリング統計量 (平均、中央値)
    features["e1b_rolling_mean_100"] = np.mean(_window(close_arr, 100))
    features["e1b_rolling_median_100"] = np.median(_window(close_arr, 100))
    features["e1b_rolling_median_50"] = np.median(_window(close_arr, 50))

    # 11. t分布パラメータ推定
    features["e1b_t_dist_dof_50"] = t分布_自由度_udf(_window(close_pct, 50))
    features["e1b_t_dist_scale_50"] = t分布_尺度_udf(_window(close_pct, 50))

    # 12. Theil-Sen 推定量 (頑健な傾き)
    features["e1b_theil_sen_slope_100"] = theil_sen_傾き_udf(_window(close_arr, 100))
    features["e1b_theil_sen_slope_50"] = theil_sen_傾き_udf(_window(close_arr, 50))

    # 13. ボラティリティ
    # ★修正⑩: Polars rolling_std(ddof=1) 準拠
    features["e1b_volatility_20"] = np.std(_window(close_pct, 20), ddof=1)

    # 14. Zスコア
    # ★修正⑧: std_20/std_50 は ddof=1 で計算済み (Polars rolling_std 準拠)
    features["e1b_zscore_20"] = (current_close - mean_20) / (std_20 + 1e-10)
    features["e1b_zscore_50"] = (current_close - mean_50) / (std_50 + 1e-10)

    return features


# EOF
