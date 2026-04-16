"""
realtime_feature_engine_1B_timeseries.py
Project Cimera V5 - 特徴量エンジンモジュール 【1B: 時系列・分布系】

"""

import sys
import os
from pathlib import Path

import numpy as np
import numba as nb
from numba import njit
from typing import Dict
import logging

# blueprint より先にパスを追加しないと ModuleNotFoundError が発生する
sys.path.append(str(Path(__file__).resolve().parent.parent))  # /workspace をパスに追加
import blueprint as config
sys.path.append(str(config.CORE_DIR))
from core_indicators import (
    calculate_atr_wilder,   # Wilder平滑化ATR（学習側と完全一致）
    # scale_by_atr は配列→配列のAPIのためスカラー用途には不使用
    # stddev_unbiased はローリング配列版のためスカラー用途には不使用
)

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
    # [REFACTORED: Step 5] Bessel補正 (ddof=1相当) を追加して engine_1_B 学習側と一致させる
    # 旧実装: std_ret = np.std(finite_returns)  ← ddof=0（バイアスあり）
    # 新実装: 手動ベッセル補正 = np.std() * sqrt(n/(n-1))
    n_ret = len(finite_returns)
    std_ret = np.std(finite_returns) * np.sqrt(n_ret / (n_ret - 1.0))

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
        # [REFACTORED: Step 5] Bessel補正 (ddof=1相当) を追加して engine_1_B 学習側と一致させる
        # 旧実装: return np.std(finite_returns)  ← ddof=0（バイアスあり）
        n_ret = len(finite_returns)
        return np.std(finite_returns) * np.sqrt(n_ret / (n_ret - 1.0))

    # 既知νを持つt分布用、尺度パラメータσ推定可能:
    # σ² = sample_variance * (ν-2)/ν
    # [REFACTORED: Step 5] Bessel補正 (ddof=1相当) を追加して engine_1_B 学習側と一致させる
    # 旧実装: sample_var = np.var(finite_returns)  ← ddof=0（バイアスあり）
    n_ret = len(finite_returns)
    sample_var = np.var(finite_returns) * (n_ret / (n_ret - 1.0))
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
    # [REFACTORED: Step 5] ゼロ除算ガードを engine_1_B 学習側の + 1e-10 方式に統一
    # 旧実装: if n * sum_x2 - sum_x**2 != 0: ... else: phi = 0.0  ← 分岐方式
    # 新実装: + 1e-10 による保護（engine_1_B 学習側と完全一致）
    phi = (n * sum_xy - sum_x * sum_y) / (n * sum_x2 - sum_x**2 + 1e-10)

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
        # ① high / low の KeyError ガード（realtime側はキーが欠落する可能性がある）
        high_arr = data.get("high", np.array([], dtype=np.float64))
        low_arr  = data.get("low",  np.array([], dtype=np.float64))

        if len(close_arr) == 0:
            return features

        close_pct = _pct_change(close_arr)

        # [REFACTORED: Step 5] ATR を core_indicators.calculate_atr_wilder で計算
        # 学習側 (engine_1_B) の inject_temp_atr と完全に同一のロジック。
        # + 1e-10 は scale_by_atr 内部で処理されるため、ここでは生ATRを保持する。
        atr_arr = calculate_atr_wilder(high_arr, low_arr, close_arr, 13)
        atr_latest = float(atr_arr[-1]) if len(atr_arr) > 0 else np.nan

        # [確認③対処] atr_latest が nan の場合、ATR依存の全特徴量を nan にするため
        # atr2 も事前に有限チェックを行う。データが十分にあれば実用上 nan にはならない。
        atr_denom = atr_latest + 1e-10          # スカラー ATR割り分母（線形）
        atr2      = atr_denom ** 2              # スカラー ATR割り分母（二乗）
        atr_ok    = np.isfinite(atr_latest)     # ATRが有効かどうかのフラグ

        # ---------------------------------------------------------
        # 1. 時系列モデル系 (Time Series Models)
        # ---------------------------------------------------------
        # ARIMA残差分散: 学習側では atr_safe**2 で割っている → リアルタイムも同様にATR²割り
        # [REFACTORED: Step 5] ATR割り追加（欠落修正）
        arima_raw_50 = arima_残差分散_udf(_window(close_arr, 50))
        arima_raw_100 = arima_残差分散_udf(_window(close_arr, 100))
        features["e1b_arima_residual_var_50"] = (
            float(arima_raw_50) / atr2
            if (np.isfinite(arima_raw_50) and atr_ok)
            else np.nan
        )
        features["e1b_arima_residual_var_100"] = (
            float(arima_raw_100) / atr2
            if (np.isfinite(arima_raw_100) and atr_ok)
            else np.nan
        )

        # Holt-Winters 平滑化 (トレンド): 学習側では atr_safe で割っている
        # [REFACTORED: Step 5] ATR割り追加（欠落修正）
        holt_raw_100 = holt_winters_トレンド_udf(_window(close_arr, 100))
        features["e1b_holt_trend_100"] = (
            float(holt_raw_100) / atr_denom
            if (np.isfinite(holt_raw_100) and atr_ok)
            else np.nan
        )

        # t分布パラメータ推定 (尺度): 学習側では pct_change rolling_std(20) + 1e-10 で割っている
        # [REFACTORED: Step 5] ATR割り追加（欠落修正）
        # 学習側: t_dist_scale_50 / (pct_change.rolling_std(20, ddof=1) + 1e-10)
        pct_20 = _window(close_pct, 20)
        valid_pct_20 = pct_20[np.isfinite(pct_20)]
        pct_std_20 = (
            float(np.std(valid_pct_20, ddof=1)) if len(valid_pct_20) >= 2 else np.nan
        )
        t_scale_raw = t分布_尺度_udf(_window(close_pct, 50))
        features["e1b_t_dist_scale_50"] = (
            float(t_scale_raw) / (pct_std_20 + 1e-10)
            if np.isfinite(t_scale_raw) and np.isfinite(pct_std_20)
            else np.nan
        )

        # ---------------------------------------------------------
        # 2. 価格変動・レンジ (Price Change / Range)
        # ---------------------------------------------------------
        # price_range: 学習側では atr_safe で割っている
        # [REFACTORED: Step 5] ATR割り追加（欠落修正）
        if len(high_arr) > 0 and len(low_arr) > 0 and atr_ok:
            raw_range = float(_last(high_arr) - _last(low_arr))
            features["e1b_price_range"] = raw_range / atr_denom
        else:
            features["e1b_price_range"] = np.nan

        # ---------------------------------------------------------
        # 3. ボラティリティ (Volatility)
        # ---------------------------------------------------------
        # e1b_volatility_20: 学習側は pct_change().rolling_std(20, ddof=1)
        # リアルタイム側も ddof=1 の不偏標準偏差として一致させる
        # ※ これはpct_changeベース（ATR割り不要）— 学習側実装と同じ
        features["e1b_volatility_20"] = (
            float(np.std(valid_pct_20, ddof=1)) if len(valid_pct_20) >= 2 else np.nan
        )

        return features
