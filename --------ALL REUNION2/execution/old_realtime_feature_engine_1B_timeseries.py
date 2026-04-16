"""
realtime_feature_engine_1B_timeseries.py
Project Cimera V5 - 特徴量エンジンモジュール 【1B: 時系列・分布系】

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

# ==================================================================
# 3. 高度な時系列モデリング Numba JIT 関数群
# ==================================================================


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


def _last(arr: np.ndarray) -> float:
    """配列の最新値（末尾）を取得。配列が空の場合はNaNを返す"""
    if len(arr) == 0:
        return np.nan
    return float(arr[-1])


# ▼▼ 修正前: Polars pct_change() 完全準拠版...
# ▼▼ 修正後: Rule 3に基づき分母に + 1e-10 を追加し、ゼロ除算を根本から防ぐ
def _pct_change(arr: np.ndarray) -> np.ndarray:
    """
    Polars pct_change() 準拠版 (先頭にNaN)。
    ゼロ除算を防止するため分母に 1e-10 を追加 (Rule 3準拠)。
    """
    if len(arr) < 2:
        return np.full_like(arr, np.nan)
    pct = np.diff(arr) / (arr[:-1] + 1e-10)
    return np.concatenate(([np.nan], pct))


class FeatureModule1B:
    @staticmethod
    def calculate_features(data: Dict[str, np.ndarray]) -> Dict[str, float]:
        features = {}

        # 安全な参照用変数
        close_arr = data["close"]
        high_arr = data["high"]
        low_arr = data["low"]

        if len(close_arr) == 0:
            return features

        close_pct = _pct_change(close_arr)

        # ---------------------------------------------------------
        # 1. 時系列モデル系 (Time Series Models)
        # ---------------------------------------------------------
        # ARIMA残差分散
        features["e1b_arima_residual_var_50"] = arima_残差分散_udf(
            _window(close_arr, 50)
        )
        features["e1b_arima_residual_var_100"] = arima_残差分散_udf(
            _window(close_arr, 100)
        )

        # Holt-Winters 平滑化 (トレンドのみ)
        features["e1b_holt_trend_100"] = holt_winters_トレンド_udf(
            _window(close_arr, 100)
        )

        # t分布パラメータ推定 (尺度のみ)
        features["e1b_t_dist_scale_50"] = t分布_尺度_udf(_window(close_pct, 50))

        # ---------------------------------------------------------
        # 2. 価格変動・レンジ (Price Change / Range)
        # ---------------------------------------------------------
        features["e1b_price_range"] = (
            float(_last(high_arr) - _last(low_arr))
            if len(high_arr) > 0 and len(low_arr) > 0
            else np.nan
        )

        # ---------------------------------------------------------
        # 3. ボラティリティ (Volatility)
        # ---------------------------------------------------------
        # ▼▼ 修正前: features["e1b_volatility_20"] = float(np.std(_window(close_pct, 20), ddof=1))
        # ▼▼ 修正後: Rule 5/3に基づき、有限値のフィルタリングと最低要素数(2本)のチェックを追加
        pct_20 = _window(close_pct, 20)
        valid_pct_20 = pct_20[np.isfinite(pct_20)]
        features["e1b_volatility_20"] = (
            float(np.std(valid_pct_20, ddof=1)) if len(valid_pct_20) >= 2 else np.nan
        )

        return features
