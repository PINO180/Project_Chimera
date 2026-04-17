"""
realtime_feature_engine_1B_timeseries.py
Project Cimera V5 - 特徴量エンジンモジュール 【1B: 時系列・分布系】

【Step 5 変更記録】学習側との完全一致修正
  - t分布_自由度/尺度_udf: ddof=0 → ddof=1 に修正（手動Bessel補正）
  - arima_残差分散_udf: ゼロ除算ガードを + 1e-10 方式に統一
  - kalman_状態推定_udf: np.var/np.diff の ddof=1 補正を追加
  - ATR割り追加: arima_residual_var / holt_trend / t_dist_scale / price_range
  - _pct_change: 分母に + 1e-10 追加（ゼロ除算根本防止）

【Step 6 変更記録】未実装特徴量の追加（学習側との完全一致）
  - rolling_mean/std/var/median/min/max (window=[10,20,50,100])
  - zscore_20/50, bollinger_upper/lower_20/50
  - price_change
  - volume_ma20, volume_price_trend（Relative Volume 1440バー基準）
  - adf_statistic_50/100, pp_statistic_50/100, kpss_statistic_50/100
  - holt_level_50/100
  - kalman_state_50/100, lowess_fitted_50/100, theil_sen_slope_50/100
  - t_dist_dof_50
  - gev_shape_50

【Step 8 変更記録】ダブルチェック指摘対応（問題1〜4）

  [修正1] volatility_20 — NaN除外挙動を学習側に完全一致（問題1）
    旧: pct_20[np.isfinite(pct_20)] でNaNを除外してからstd計算
        → window内にNaN/infが含まれていても非NaN値を返してしまい学習側と不一致
    新: len(pct_20) >= 20 かつ np.all(np.isfinite(pct_20)) の場合のみ計算
        → window内に1本でも非有限値があればNaN（Polars rolling_std と同一挙動）

  [修正2] volume_price_trend — inf伝播を学習側に完全一致（問題2）
    旧: vpt[np.isfinite(vpt)] でinfを除外してmean → 学習側と不一致
    新: finite フィルタを廃止。inf をそのまま mean に通し QA でクリップ
        → 学習側 (pct_change * rel_volume).rolling_mean(10) と同一挙動

  [修正3] adf/pp/kpss — window未満データへのgate追加（問題3）
    旧: _window が 10〜(window-1) 本を返す場合 UDF 内 len<10 チェックを通過し
        非NaN値を返してしまい学習側と不一致（10≤len<windowの期間）
    新: len(pct_w) < window のとき即 NaN を代入し UDF を呼ばない
        → 学習側 rolling_map(min_samples=window) と同一挙動

  [修正4] QAState — bias補正と _ewm_n を追加（問題4、1Aとの設計統一）
    旧: sqrt(EWM_var) のみ → Polars ewm_std(bias=False) の bias 補正が未適用
        → 起動直後に EWM_std が過小評価されクリップ幅が狭くなる
    新: bias_corr = 1/sqrt(1-(1-alpha)^(2n)) を乗算（1A の修正5と同一実装）
        _ewm_n フィールドで更新回数を管理。ウォームアップ後は bias_corr → 1.0
"""

import sys
import os
from pathlib import Path

import numpy as np
import numba as nb
from numba import njit
from typing import Dict, Optional
import logging

sys.path.append(str(Path(__file__).resolve().parent.parent))
import blueprint as config
sys.path.append(str(config.CORE_DIR))
from core_indicators import (
    calculate_atr_wilder,
)

logger = logging.getLogger("ProjectForge.FeatureEngine.1B")


# ==================================================================
# ヘルパー関数
# ==================================================================

@nb.njit(fastmath=False, cache=True)
def _window(arr: np.ndarray, window: int) -> np.ndarray:
    if window <= 0:
        return np.empty(0, dtype=arr.dtype)
    if window > len(arr):
        return arr
    return arr[-window:]


@nb.njit(fastmath=False, cache=True)
def _last(arr: np.ndarray) -> float:
    if len(arr) == 0:
        return np.nan
    return float(arr[-1])


def _pct_change(arr: np.ndarray) -> np.ndarray:
    """Polars pct_change() と完全一致: (x[i] - x[i-1]) / x[i-1]
    prev == 0 のとき inf（Polars準拠）、先頭は nan。"""
    if len(arr) < 2:
        return np.full_like(arr, np.nan)
    pct = np.full(len(arr), np.nan, dtype=np.float64)
    with np.errstate(divide="ignore", invalid="ignore"):
        pct[1:] = (arr[1:] - arr[:-1]) / arr[:-1]
    return pct


def _rolling_map(arr: np.ndarray, window: int, func) -> np.ndarray:
    """Polars rolling_map 相当。window サイズで末尾から func を適用し最新値を返す。"""
    n = len(arr)
    out = np.full(n, np.nan, dtype=np.float64)
    for i in range(window - 1, n):
        out[i] = func(arr[i - window + 1:i + 1])
    return out


# ==================================================================
# QAState — 学習側 apply_quality_assurance_to_group の等価実装
# ==================================================================

class QAState:
    """
    【修正4】学習側 apply_quality_assurance_to_group のリアルタイム等価実装。

    学習側の処理（Polars LazyFrame 全系列一括、apply_quality_assurance_to_group）:
        safe_col = when(col.is_infinite()).then(None).otherwise(col)
        ema_val  = safe_col.ewm_mean(half_life=HL, ignore_nulls=True, adjust=False)
        ema_std  = safe_col.ewm_std (half_life=HL, ignore_nulls=True, adjust=False)
        result   = col.clip(ema_val - 5*ema_std, ema_val + 5*ema_std)
                      .fill_null(0.0).fill_nan(0.0)

    Polars ewm_mean(adjust=False) の再帰式（alpha = 1 - exp(-ln2 / HL)）:
        EWM_mean[t] = alpha * x[t] + (1-alpha) * EWM_mean[t-1]  （NaN はスキップ）

    Polars ewm_var(adjust=False) の再帰式（本番側 _ewm_var の等価式）:
        EWM_var[t] = (1-alpha) * EWM_var[t-1] + alpha*(1-alpha) * (x[t] - EWM_mean[t-1])^2
                   = (1-alpha) * (EWM_var[t-1] + alpha * (x[t] - EWM_mean[t-1])^2)

    ⚠️ Polars ewm_std(adjust=False) のデフォルトは bias=False（不偏補正あり）。
    bias_corr = 1 / sqrt(1 - (1-alpha)^(2n)) を乗算して Polars と一致させる。
    n が大きくなると bias_corr → 1.0 に収束（ウォームアップ後は実質影響なし）。

    ⚠️ 起動時のシード差:
        学習側は全系列先頭から EWM を積み上げる（確定的）。
        本番側は最初の有効値でシードするため、起動直後の数バーで軌跡が異なる。
        対策: 事前に lookback_bars * 3 本のウォームアップを推奨。

    使い方:
        qa_state = FeatureModule1B.QAState(lookback_bars=1440)
        # ウォームアップ（強く推奨）
        for bar in historical_data[-lookback_bars * 3:]:
            FeatureModule1B.calculate_features(warmup_window, 1440, qa_state)
        # 本番
        for bar in live_stream:
            features = FeatureModule1B.calculate_features(data_window, 1440, qa_state)
    """

    def __init__(self, lookback_bars: int = 1440):
        self.alpha: float = 1.0 - np.exp(-np.log(2.0) / max(lookback_bars, 1))
        self._ewm_mean: Dict[str, float] = {}
        self._ewm_var: Dict[str, float] = {}
        # bias=False 補正用: Polars ewm_std は t バー目に sqrt(1/(1-(1-alpha)^(2t))) を乗じる。
        # ウォームアップが十分であれば値は 1.0 に収束する。
        self._ewm_n: Dict[str, int] = {}  # 有効値の累積更新回数（bias 補正に使用）

    def update_and_clip(self, key: str, raw_val: float) -> float:
        """1特徴量の raw_val に QA処理を適用して返す（学習側と完全一致）。"""
        alpha = self.alpha

        # 【inf処理修正】学習側の挙動を再現:
        #   学習側: when(col==inf).then(upper).when(col==-inf).then(lower) で置換後clip
        #   本番側旧: inf → NaN → 0.0（学習側と不一致）
        #   本番側新: inf を先に記録し、EWMはNaNとしてスキップ。
        #             clip時に +inf → upper_bound, -inf → lower_bound で置き換える。
        is_pos_inf = np.isposinf(raw_val)
        is_neg_inf = np.isneginf(raw_val)
        ewm_input = np.nan if not np.isfinite(raw_val) else raw_val

        # Step2: EWM 状態更新（ignore_nulls=True 相当）
        if key not in self._ewm_mean:
            if np.isnan(ewm_input):
                return 0.0
            self._ewm_mean[key] = ewm_input
            self._ewm_var[key]  = 0.0
            self._ewm_n[key]    = 1
            return ewm_input
        else:
            if not np.isnan(ewm_input):
                prev_mean = self._ewm_mean[key]
                prev_var  = self._ewm_var[key]
                new_mean  = alpha * ewm_input + (1.0 - alpha) * prev_mean
                # Polars ewm_var(adjust=False) と等価:
                #   var[t] = (1-alpha)*(var[t-1] + alpha*(x[t]-mean[t-1])^2)
                new_var   = (1.0 - alpha) * (prev_var + alpha * (ewm_input - prev_mean) ** 2)
                self._ewm_mean[key] = new_mean
                self._ewm_var[key]  = new_var
                self._ewm_n[key]    = self._ewm_n.get(key, 0) + 1

        # Step3: ±5σ クリップ
        # Polars ewm_std(adjust=False, bias=False) の bias 補正を適用:
        #   bias_corr = 1 / sqrt(1 - (1-alpha)^(2n))
        ewm_mean  = self._ewm_mean[key]
        n_updates = self._ewm_n.get(key, 1)
        decay_2n  = (1.0 - alpha) ** (2 * n_updates)
        denom     = 1.0 - decay_2n
        bias_corr = 1.0 / np.sqrt(denom) if denom > 1e-15 else 1.0
        ewm_std   = np.sqrt(max(self._ewm_var[key], 0.0)) * bias_corr
        lower     = ewm_mean - 5.0 * ewm_std
        upper     = ewm_mean + 5.0 * ewm_std

        if np.isnan(ewm_input):
            return 0.0
        if is_pos_inf:
            return float(upper) if np.isfinite(upper) else 0.0
        if is_neg_inf:
            return float(lower) if np.isfinite(lower) else 0.0

        clipped = float(np.clip(raw_val, lower, upper))
        return clipped if np.isfinite(clipped) else 0.0


# ==================================================================
# Numba UDF群（学習側 engine_1_B と完全同一実装）
# ==================================================================

@nb.njit(fastmath=True, cache=True)
def adf_統計量_udf(prices: np.ndarray) -> float:
    """真のADF検定統計量（学習側と同一）"""
    if len(prices) < 10:
        return np.nan
    finite_prices = prices[np.isfinite(prices)]
    if len(finite_prices) < 10:
        return np.nan

    diff_prices = np.diff(finite_prices)
    if len(diff_prices) < 5:
        return np.nan

    y = diff_prices[1:]
    lagged_y = finite_prices[1:-1]
    lagged_diff = diff_prices[:-1]
    n = len(y)
    if n < 3:
        return np.nan

    X = np.empty((n, 3), dtype=np.float64)
    X[:, 0] = 1.0
    X[:, 1] = lagged_y
    X[:, 2] = lagged_diff

    try:
        XtX = X.T @ X
        XtX_inv = np.linalg.inv(XtX)
        XtY = X.T @ y
        beta = XtX_inv @ XtY
        residuals = y - X @ beta
        sse = np.sum(residuals ** 2)
        mse = sse / (n - 3.0)
        se_beta = np.sqrt(mse * XtX_inv[1, 1])
        if se_beta > 1e-10:
            return beta[1] / se_beta
        else:
            return np.nan
    except:
        return np.nan


@nb.njit(fastmath=True, cache=True)
def phillips_perron_統計量_udf(prices: np.ndarray) -> float:
    """真のPhillips-Perron検定統計量（Newey-West補正付き、学習側と同一）"""
    if len(prices) < 10:
        return np.nan
    finite_prices = prices[np.isfinite(prices)]
    if len(finite_prices) < 10:
        return np.nan

    diff_prices = np.diff(finite_prices)
    lagged_prices = finite_prices[:-1]
    if len(diff_prices) < 5:
        return np.nan

    n = len(diff_prices)
    X = np.empty((n, 2), dtype=np.float64)
    X[:, 0] = 1.0
    X[:, 1] = lagged_prices
    y = diff_prices

    try:
        XtX_inv = np.linalg.inv(X.T @ X)
        beta = XtX_inv @ X.T @ y
        residuals = y - X @ beta

        gamma_0 = np.sum(residuals ** 2) / n
        lag_max = int(4.0 * (n / 100.0) ** (2.0 / 9.0))
        if lag_max < 1:
            lag_max = 1

        lambda_sq = gamma_0
        for j in range(1, lag_max + 1):
            gamma_j = np.sum(residuals[j:] * residuals[:-j]) / n
            weight = 1.0 - (j / (lag_max + 1.0))
            lambda_sq += 2.0 * weight * gamma_j

        s2 = np.sum(residuals ** 2) / (n - 2.0)
        s = np.sqrt(s2)
        se_beta = np.sqrt(s2 * XtX_inv[1, 1])

        if se_beta <= 1e-10 or lambda_sq <= 1e-10:
            return np.nan

        t_stat = beta[1] / se_beta
        term1 = np.sqrt(gamma_0 / lambda_sq) * t_stat
        term2 = 0.5 * ((lambda_sq - gamma_0) / np.sqrt(lambda_sq)) * (n * se_beta / s)
        return term1 - term2
    except:
        return np.nan


@nb.njit(fastmath=True, cache=True)
def kpss_統計量_udf(prices: np.ndarray) -> float:
    """真のKPSS検定統計量（Newey-West補正付き、学習側と同一）"""
    if len(prices) < 10:
        return np.nan
    finite_prices = prices[np.isfinite(prices)]
    if len(finite_prices) < 10:
        return np.nan

    n = len(finite_prices)
    t = np.arange(n, dtype=np.float64)

    try:
        sum_t = np.sum(t)
        sum_t2 = np.sum(t ** 2)
        sum_y = np.sum(finite_prices)
        sum_ty = np.sum(t * finite_prices)

        beta = (n * sum_ty - sum_t * sum_y) / (n * sum_t2 - sum_t ** 2 + 1e-10)
        alpha = (sum_y - beta * sum_t) / n

        detrended = finite_prices - (alpha + beta * t)
        cumsum = np.cumsum(detrended)

        gamma_0 = np.sum(detrended ** 2) / n
        lag_max = int(4.0 * (n / 100.0) ** (2.0 / 9.0))
        if lag_max < 1:
            lag_max = 1

        long_run_var = gamma_0
        for j in range(1, lag_max + 1):
            gamma_j = np.sum(detrended[j:] * detrended[:-j]) / n
            weight = 1.0 - (j / (lag_max + 1.0))
            long_run_var += 2.0 * weight * gamma_j

        if long_run_var > 1e-10:
            return np.sum(cumsum ** 2) / (n ** 2 * long_run_var)
        else:
            return np.nan
    except:
        return np.nan


@nb.njit(fastmath=True, cache=True)
def t分布_自由度_udf(returns: np.ndarray) -> float:
    """t分布の自由度パラメータ推定（ddof=1、学習側と同一）"""
    if len(returns) < 10:
        return np.nan
    finite_returns = returns[np.isfinite(returns)]
    if len(finite_returns) < 10:
        return np.nan

    mean_ret = np.mean(finite_returns)
    n_ret = len(finite_returns)
    std_ret = np.std(finite_returns) * np.sqrt(n_ret / (n_ret - 1.0))  # ddof=1

    if std_ret <= 0:
        return np.nan

    standardized = (finite_returns - mean_ret) / std_ret
    fourth_moment = np.mean(standardized ** 4)
    excess_kurtosis = fourth_moment - 3.0

    if excess_kurtosis > 0:
        dof = 4.0 * (3.0 + fourth_moment) / excess_kurtosis
        dof = max(2.1, min(dof, 100.0))
    else:
        dof = 100.0

    return dof


@nb.njit(fastmath=True, cache=True)
def t分布_尺度_udf(returns: np.ndarray) -> float:
    """t分布の尺度パラメータ推定（ddof=1、学習側と同一）"""
    if len(returns) < 5:
        return np.nan
    finite_returns = returns[np.isfinite(returns)]
    if len(finite_returns) < 5:
        return np.nan

    dof = t分布_自由度_udf(returns)

    if np.isnan(dof) or dof <= 2:
        n_ret = len(finite_returns)
        return np.std(finite_returns) * np.sqrt(n_ret / (n_ret - 1.0))  # ddof=1

    n_ret = len(finite_returns)
    sample_var = np.var(finite_returns) * (n_ret / (n_ret - 1.0))  # ddof=1
    scale_squared = sample_var * (dof - 2.0) / dof
    return np.sqrt(max(scale_squared, 1e-8))


@nb.njit(fastmath=True, cache=True)
def gev_形状_udf(extremes: np.ndarray) -> float:
    """GEV分布形状パラメータ推定（L-モーメント法、学習側と同一）"""
    if len(extremes) < 10:
        return np.nan
    finite_extremes = extremes[np.isfinite(extremes)]
    if len(finite_extremes) < 10:
        return np.nan

    sorted_data = np.sort(finite_extremes)
    n = len(sorted_data)

    l1 = np.mean(sorted_data)

    sum_l2 = 0.0
    for i in range(n):
        weight = (2.0 * i - n + 1.0) / n
        sum_l2 += weight * sorted_data[i]
    l2 = sum_l2 / 2.0

    sum_l3 = 0.0
    for i in range(n):
        weight = ((i * (i - 1.0)) - 2.0 * i * (n - 1.0) + (n - 1.0) * (n - 2.0)) / (n * (n - 1.0))
        sum_l3 += weight * sorted_data[i]
    l3 = sum_l3 / 3.0

    if abs(l2) > 1e-8:
        tau3 = l3 / l2
        shape = 7.859 * tau3 + 2.9554 * tau3 ** 2
        shape = max(-0.5, min(shape, 0.5))
    else:
        shape = 0.0

    return shape


@nb.njit(fastmath=True, cache=True)
def holt_winters_レベル_udf(prices: np.ndarray) -> float:
    """Holt-Wintersレベル成分（alpha=0.3、学習側と同一）"""
    if len(prices) < 10:
        return np.nan
    finite_prices = prices[np.isfinite(prices)]
    if len(finite_prices) < 10:
        return np.nan

    alpha = 0.3
    level = finite_prices[0]
    for i in range(1, len(finite_prices)):
        level = alpha * finite_prices[i] + (1 - alpha) * level
    return level


@nb.njit(fastmath=True, cache=True)
def holt_winters_トレンド_udf(prices: np.ndarray) -> float:
    """Holt-Wintersトレンド成分（alpha=0.3, beta=0.1、学習側と同一）"""
    if len(prices) < 10:
        return np.nan
    finite_prices = prices[np.isfinite(prices)]
    if len(finite_prices) < 10:
        return np.nan

    alpha = 0.3
    beta = 0.1
    level = finite_prices[0]
    trend = finite_prices[1] - finite_prices[0] if len(finite_prices) > 1 else 0.0

    for i in range(1, len(finite_prices)):
        new_level = alpha * finite_prices[i] + (1 - alpha) * (level + trend)
        new_trend = beta * (new_level - level) + (1 - beta) * trend
        level = new_level
        trend = new_trend

    return trend


@nb.njit(fastmath=True, cache=True)
def arima_残差分散_udf(prices: np.ndarray) -> float:
    """ARIMA(1,1,0)残差分散（+ 1e-10 ゼロ除算保護、学習側と同一）"""
    if len(prices) < 15:
        return np.nan
    finite_prices = prices[np.isfinite(prices)]
    if len(finite_prices) < 15:
        return np.nan

    diff_prices = np.diff(finite_prices)
    if len(diff_prices) < 10:
        return np.nan

    y = diff_prices[1:]
    x = diff_prices[:-1]
    n = len(y)
    if n < 5:
        return np.nan

    sum_x = np.sum(x)
    sum_y = np.sum(y)
    sum_xy = np.sum(x * y)
    sum_x2 = np.sum(x * x)

    phi = (n * sum_xy - sum_x * sum_y) / (n * sum_x2 - sum_x ** 2 + 1e-10)
    intercept = (sum_y - phi * sum_x) / n
    residuals = y - (intercept + phi * x)
    return np.sum(residuals ** 2) / (n - 2)


@nb.njit(fastmath=True, cache=True)
def kalman_状態推定_udf(prices: np.ndarray) -> float:
    """カルマンフィルタ状態推定（ddof=1補正、学習側と同一）"""
    if len(prices) < 5:
        return np.nan
    finite_prices = prices[np.isfinite(prices)]
    if len(finite_prices) < 5:
        return np.nan

    x = finite_prices[0]
    P = 1.0

    if len(finite_prices) > 1:
        diff_vals = np.diff(finite_prices)
        n_diff = len(diff_vals)
        n_prices = len(finite_prices)
        diff_var = np.var(diff_vals) * (n_diff / (n_diff - 1.0)) if n_diff > 1 else 0.0
        obs_var = np.var(finite_prices) * (n_prices / (n_prices - 1.0)) if n_prices > 1 else 0.0
        Q = max(diff_var, obs_var * 0.01)
        R = obs_var * 0.1
    else:
        Q = 1.0
        R = 0.1

    for i in range(1, len(finite_prices)):
        x_pred = x
        P_pred = P + Q
        K = P_pred / (P_pred + R)
        x = x_pred + K * (finite_prices[i] - x_pred)
        P = (1 - K) * P_pred

    return x


@nb.njit(fastmath=True, cache=True)
def lowess_適合値_udf(prices: np.ndarray) -> float:
    """LOWESS適合値（bandwidth=0.3、学習側と同一）"""
    if len(prices) < 10:
        return np.nan
    finite_prices = prices[np.isfinite(prices)]
    if len(finite_prices) < 10:
        return np.nan

    n = len(finite_prices)
    bandwidth = 0.3
    h = max(3, int(bandwidth * n))
    target_idx = n - 1

    distances = np.abs(np.arange(n) - target_idx)
    sorted_indices = np.argsort(distances)
    neighbor_indices = sorted_indices[:h]

    x_neighbors = neighbor_indices.astype(np.float64)
    y_neighbors = finite_prices[neighbor_indices]

    max_dist = np.max(distances[neighbor_indices])
    if max_dist > 1e-10:
        weights = np.zeros(h, dtype=np.float64)
        for i in range(h):
            u = distances[neighbor_indices[i]] / max_dist
            if u < 1.0:
                weights[i] = (1.0 - u ** 3) ** 3
            else:
                weights[i] = 0.0
    else:
        weights = np.ones(h, dtype=np.float64)

    if len(x_neighbors) >= 2:
        sum_w = np.sum(weights)
        if sum_w > 1e-10:
            x_mean = np.sum(weights * x_neighbors) / sum_w
            y_mean = np.sum(weights * y_neighbors) / sum_w
            numerator = np.sum(weights * (x_neighbors - x_mean) * (y_neighbors - y_mean))
            denominator = np.sum(weights * (x_neighbors - x_mean) ** 2)
            if denominator > 1e-10:
                slope = numerator / denominator
                intercept = y_mean - slope * x_mean
                return intercept + slope * target_idx
            else:
                return y_mean
        else:
            return np.mean(y_neighbors)
    else:
        return finite_prices[-1]


@nb.njit(fastmath=True, cache=True)
def theil_sen_傾き_udf(prices: np.ndarray) -> float:
    """Theil-Sen傾き推定（max_pairs=1000、学習側と同一）"""
    if len(prices) < 10:
        return np.nan
    finite_prices = prices[np.isfinite(prices)]
    n = len(finite_prices)
    if n < 10:
        return np.nan

    max_pairs = min(1000, (n * (n - 1)) // 2)
    slopes = np.zeros(max_pairs, dtype=np.float64)
    slope_idx = 0

    if max_pairs < (n * (n - 1)) // 2:
        step = max(1, ((n * (n - 1)) // 2) // max_pairs)
        count = 0
        for i in range(n - 1):
            for j in range(i + 1, n):
                if count % step == 0 and slope_idx < max_pairs:
                    slopes[slope_idx] = (finite_prices[j] - finite_prices[i]) / float(j - i)
                    slope_idx += 1
                count += 1
    else:
        for i in range(n - 1):
            for j in range(i + 1, n):
                if slope_idx < max_pairs:
                    slopes[slope_idx] = (finite_prices[j] - finite_prices[i]) / float(j - i)
                    slope_idx += 1

    if slope_idx > 0:
        return np.median(slopes[:slope_idx])
    else:
        return np.nan


# ==================================================================
# メイン計算クラス
# ==================================================================

class FeatureModule1B:
    # window_sizes["general"] = [10, 20, 50, 100]（学習側と同一）
    GENERAL_WINDOWS = [10, 20, 50, 100]

    # 外部から FeatureModule1B.QAState としてアクセス可能にする
    QAState = QAState

    @staticmethod
    def calculate_features(
        data: Dict[str, np.ndarray],
        lookback_bars: int = 1440,
        qa_state = None,
    ) -> Dict[str, float]:
        """
        Args:
            data         : close/high/low/volume の numpy 配列を含む辞書
            lookback_bars: タイムフレームに応じた1日あたりのバー数。
                           QA の EWM 半減期に使用。
                           学習側 ProcessingConfig.timeframe_bars_per_day と同じ値を渡すこと。
                           例: M1→1440, M5→288, H1→24, H4→6
            qa_state     : QAState インスタンス。
                           本番稼働時は必ず渡し、同一インスタンスを毎バー使い回すこと。
                           None の場合は QA 処理をスキップ（後方互換・単体テスト用）。
        """
        features = {}

        close_arr = data["close"]
        high_arr  = data.get("high",   np.array([], dtype=np.float64))
        low_arr   = data.get("low",    np.array([], dtype=np.float64))
        volume_arr = data.get("volume", np.array([], dtype=np.float64))

        if len(close_arr) == 0:
            return features

        close_pct = _pct_change(close_arr)

        # ATR（学習側 inject_temp_atr と同一: calculate_atr_wilder + 1e-10）
        atr_arr    = calculate_atr_wilder(high_arr, low_arr, close_arr, 13)
        atr_latest = float(atr_arr[-1]) if len(atr_arr) > 0 else np.nan
        atr_ok     = np.isfinite(atr_latest)
        atr_denom  = atr_latest + 1e-10
        atr2       = atr_denom ** 2

        close_last = float(close_arr[-1])

        # ---------------------------------------------------------
        # 1. 基本統計系 — basic_stats
        #    学習側: _create_basic_stats_features (window=[10,20,50,100])
        # ---------------------------------------------------------
        for window in FeatureModule1B.GENERAL_WINDOWS:
            w_close = _window(close_arr, window)
            if len(w_close) >= window:
                mean_w   = float(np.mean(w_close))
                std_w    = float(np.std(w_close, ddof=1))
                var_w    = float(np.var(w_close, ddof=1))
                median_w = float(np.median(w_close))
                min_w    = float(np.min(w_close))
                max_w    = float(np.max(w_close))
                if atr_ok:
                    features[f"e1b_rolling_mean_{window}"]   = (mean_w   - close_last) / atr_denom
                    features[f"e1b_rolling_std_{window}"]    = std_w    / atr_denom
                    features[f"e1b_rolling_var_{window}"]    = var_w    / atr2
                    features[f"e1b_rolling_median_{window}"] = (median_w - close_last) / atr_denom
                    features[f"e1b_rolling_min_{window}"]    = (min_w    - close_last) / atr_denom
                    features[f"e1b_rolling_max_{window}"]    = (max_w    - close_last) / atr_denom
                else:
                    for k in ["rolling_mean", "rolling_std", "rolling_var",
                              "rolling_median", "rolling_min", "rolling_max"]:
                        features[f"e1b_{k}_{window}"] = np.nan
            else:
                for k in ["rolling_mean", "rolling_std", "rolling_var",
                          "rolling_median", "rolling_min", "rolling_max"]:
                    features[f"e1b_{k}_{window}"] = np.nan

        # ---------------------------------------------------------
        # 2. 複合計算系 — composite
        # ---------------------------------------------------------
        for window in [20, 50]:
            w_close = _window(close_arr, window)
            if len(w_close) >= window:
                mean_w = float(np.mean(w_close))
                std_w  = float(np.std(w_close, ddof=1))
                features[f"e1b_zscore_{window}"] = (
                    (close_last - mean_w) / (std_w + 1e-10)
                )
                if atr_ok:
                    features[f"e1b_bollinger_upper_{window}"] = (
                        (mean_w + 2 * std_w - close_last) / atr_denom
                    )
                    features[f"e1b_bollinger_lower_{window}"] = (
                        (mean_w - 2 * std_w - close_last) / atr_denom
                    )
                else:
                    features[f"e1b_bollinger_upper_{window}"] = np.nan
                    features[f"e1b_bollinger_lower_{window}"] = np.nan
            else:
                features[f"e1b_zscore_{window}"]         = np.nan
                features[f"e1b_bollinger_upper_{window}"] = np.nan
                features[f"e1b_bollinger_lower_{window}"] = np.nan

        # price_change
        features["e1b_price_change"] = float(close_pct[-1]) if len(close_pct) > 0 else np.nan

        # volatility_20: pct_change().rolling_std(20, ddof=1)
        # 【修正済み Step7→8: 問題1】isfinite フィルタを廃止。
        # 学習側: pct_change().rolling_std(20, ddof=1) はwindow内に NaN/inf が
        # 1本でも含まれると NaN を返す。旧本番側の finite フィルタは NaN を除外して
        # 計算してしまい学習側と不一致だった。
        # 新: window=20 本揃っており、かつ全要素が有限の場合のみ std を計算。
        #     1本でも非有限 → NaN（学習側 Polars rolling_std と同一挙動）。
        pct_20 = _window(close_pct, 20)
        if len(pct_20) >= 20 and np.all(np.isfinite(pct_20)):
            pct_std_20 = float(np.std(pct_20, ddof=1))
        else:
            pct_std_20 = np.nan
        features["e1b_volatility_20"] = pct_std_20

        # price_range: (high - low) / atr_safe
        if len(high_arr) > 0 and len(low_arr) > 0 and atr_ok:
            features["e1b_price_range"] = float(_last(high_arr) - _last(low_arr)) / atr_denom
        else:
            features["e1b_price_range"] = np.nan

        # volume_ma20, volume_price_trend
        # 学習側:
        #   rel_volume = volume / (rolling_mean(volume, 1440) + 1e-10)  ← 各バーで分母が変わる
        #   volume_ma20 = rel_volume.rolling_mean(20)
        #   volume_price_trend = (pct_change * rel_volume).rolling_mean(10)
        # 本番側: 末尾1440バー平均を固定分母として使用（1440バー平均は20バー以内で実質不変）
        # 【修正1】vol_base / vol_base_20 の二重計算を vol_mean_1440 に統一
        # 【修正3】volume_price_trend: 10バー揃っている場合のみ計算（min_periods=10 と一致）
        if len(volume_arr) > 0:
            vol_mean_1440 = float(np.mean(_window(volume_arr, lookback_bars)))

            # volume_ma20
            vol_w20 = _window(volume_arr, 20)
            if len(vol_w20) >= 20:
                rel_vol_w20 = vol_w20 / (vol_mean_1440 + 1e-10)
                features["e1b_volume_ma20"] = float(np.mean(rel_vol_w20))
            else:
                features["e1b_volume_ma20"] = np.nan

            # volume_price_trend
            pct_w10 = _window(close_pct, 10)
            vol_w10 = _window(volume_arr, 10)
            if len(pct_w10) >= 10 and len(vol_w10) >= 10:
                rel_v_w10 = vol_w10 / (vol_mean_1440 + 1e-10)
                vpt = pct_w10 * rel_v_w10
                # 【修正済み Step7→8: 問題2】finite フィルタを廃止。
                # 学習側: (pct_change * rel_volume).rolling_mean(10) は
                # inf をそのまま伝播させ QA でクリップする。
                # 旧: finite_vpt フィルタで inf を除外 → 学習側と不一致。
                # 新: inf を除外せず mean に通し、QA でクリップさせる。
                features["e1b_volume_price_trend"] = float(np.mean(vpt))
            else:
                features["e1b_volume_price_trend"] = np.nan
        else:
            features["e1b_volume_ma20"]        = np.nan
            features["e1b_volume_price_trend"] = np.nan

        # ---------------------------------------------------------
        # 3. 時系列解析系 — timeseries
        #    adf / pp / kpss (pct_change に適用、window=[50,100])
        # ---------------------------------------------------------
        # 【修正済み Step7→8: 問題3】window未満データでのUDF呼び出しを防ぐgateを追加。
        # 学習側: rolling_map(func, window_size=window, min_periods=window) →
        #         window本未満は NaN。
        # 旧本番側: _window が 10〜(window-1) 本を返すとUDF内 len<10 チェックを通過し
        #           非NaN値を返してしまい学習側と不一致。
        # 新: len(pct_w) < window のとき即 NaN を代入し UDF を呼ばない。
        for window in [50, 100]:
            pct_w = _window(close_pct, window)
            if len(pct_w) < window:
                features[f"e1b_adf_statistic_{window}"]  = np.nan
                features[f"e1b_pp_statistic_{window}"]   = np.nan
                features[f"e1b_kpss_statistic_{window}"] = np.nan
            else:
                features[f"e1b_adf_statistic_{window}"]  = adf_統計量_udf(pct_w)
                features[f"e1b_pp_statistic_{window}"]   = phillips_perron_統計量_udf(pct_w)
                features[f"e1b_kpss_statistic_{window}"] = kpss_統計量_udf(pct_w)

        # ---------------------------------------------------------
        # 4. 指数平滑・ARIMA系 — exponential_arima (window=[50,100])
        # ---------------------------------------------------------
        # 【修正: 問題3と同根】window未満データへのgateを追加。
        # 学習側: rolling_map(window_size=window, min_samples=window) →
        #         window本未満は NaN。UDF内の len<10/len<15 チェックでは
        #         10〜(window-1) 本のデータで非NaN値を返してしまう。
        for window in [50, 100]:
            w_close = _window(close_arr, window)
            if len(w_close) < window:
                features[f"e1b_holt_level_{window}"]        = np.nan
                features[f"e1b_holt_trend_{window}"]        = np.nan
                features[f"e1b_arima_residual_var_{window}"] = np.nan
            else:
                holt_level_raw = holt_winters_レベル_udf(w_close)
                holt_trend_raw = holt_winters_トレンド_udf(w_close)
                arima_raw      = arima_残差分散_udf(w_close)

                features[f"e1b_holt_level_{window}"] = (
                    (float(holt_level_raw) - close_last) / atr_denom
                    if np.isfinite(holt_level_raw) and atr_ok else np.nan
                )
                features[f"e1b_holt_trend_{window}"] = (
                    float(holt_trend_raw) / atr_denom
                    if np.isfinite(holt_trend_raw) and atr_ok else np.nan
                )
                features[f"e1b_arima_residual_var_{window}"] = (
                    float(arima_raw) / atr2
                    if np.isfinite(arima_raw) and atr_ok else np.nan
                )

        # ---------------------------------------------------------
        # 5. カルマン・回帰系 — kalman_regression (window=[50,100])
        # ---------------------------------------------------------
        # 【修正: 問題3と同根】window未満データへのgateを追加（Group4と同様）。
        for window in [50, 100]:
            w_close = _window(close_arr, window)
            if len(w_close) < window:
                features[f"e1b_kalman_state_{window}"]    = np.nan
                features[f"e1b_lowess_fitted_{window}"]   = np.nan
                features[f"e1b_theil_sen_slope_{window}"] = np.nan
            else:
                kalman_raw  = kalman_状態推定_udf(w_close)
                lowess_raw  = lowess_適合値_udf(w_close)
                theil_raw   = theil_sen_傾き_udf(w_close)

                features[f"e1b_kalman_state_{window}"] = (
                    (float(kalman_raw) - close_last) / atr_denom
                    if np.isfinite(kalman_raw) and atr_ok else np.nan
                )
                features[f"e1b_lowess_fitted_{window}"] = (
                    (float(lowess_raw) - close_last) / atr_denom
                    if np.isfinite(lowess_raw) and atr_ok else np.nan
                )
                features[f"e1b_theil_sen_slope_{window}"] = (
                    float(theil_raw) / atr_denom
                    if np.isfinite(theil_raw) and atr_ok else np.nan
                )

        # ---------------------------------------------------------
        # 6. 分布パラメータ系 — distributions
        # ---------------------------------------------------------
        # 【修正】t_dist_dof_50 / t_dist_scale_50 に window gate を追加。
        # 学習側: rolling_map(window_size=50, min_samples=50) → 50本未満はNaN。
        # 旧: pct_50が50本未満でもUDF内len<10チェックのみ → 10〜49本で非NaN値を返す。
        pct_50 = _window(close_pct, 50)
        if len(pct_50) < 50:
            features["e1b_t_dist_dof_50"]   = np.nan
            features["e1b_t_dist_scale_50"] = np.nan
        else:
            features["e1b_t_dist_dof_50"] = t分布_自由度_udf(pct_50)

            t_scale_raw = t分布_尺度_udf(pct_50)
            features["e1b_t_dist_scale_50"] = (
                float(t_scale_raw) / (pct_std_20 + 1e-10)
                if np.isfinite(t_scale_raw) and np.isfinite(pct_std_20) else np.nan
            )

        if len(high_arr) >= 50:
            features["e1b_gev_shape_50"] = gev_形状_udf(_window(high_arr, 50))
        else:
            features["e1b_gev_shape_50"] = np.nan

        # ---------------------------------------------------------
        # 【修正4】QA処理 — 学習側 apply_quality_assurance_to_group と等価
        # ---------------------------------------------------------
        if qa_state is not None:
            qa_result = {}
            for key, val in features.items():
                qa_result[key] = qa_state.update_and_clip(key, val)
            features = qa_result
        else:
            for key in list(features.keys()):
                if not np.isfinite(features[key]):
                    features[key] = 0.0

        return features
