# realtime_feature_engine_1D_volume.py
# Project Cimera V5 - Feature Engine Module 1D【Volume, Volatility & Price Action】
#
# [Step 9] リファクタリング方針:
#   1. core_indicators から calculate_atr_wilder / scale_by_atr /
#      calculate_sample_weight をインポートし、学習側 engine_1_D と統一
#   2. ATR割り（/ (atr + 1e-10) 直書き）を scale_by_atr に統一
#   3. e1d_sample_weight を calculate_sample_weight で追加
#   4. 学習側に存在するが realtime 側に欠落していた特徴量を全て追加:
#        chaikin_volatility_{10,20}, mass_index_{20,30},
#        cmf_{13,21,34}, vwap_dist_{13,21,34},
#        obv_rel, accumulation_distribution_rel, force_index_norm,
#        volume_ma20_rel, volume_price_trend_norm,
#        donchian_*_dist_{10,20,50,100}, price_channel_*_dist_{10,20,50,100},
#        commodity_channel_index_{14,20},
#        pivot_dist, resistance1_dist, support1_dist, fib_level_50_dist,
#        typical_price_dist, weighted_close_dist, median_price_dist,
#        body_size_atr, upper_wick_ratio,
#        hv_annual_252, hv_robust_annual_252, hv_regime_50,
#        hv_standard_{10,20,50} / hv_robust_{10,20,50} (窓サイズを学習側に統一)
#   5. 旧 e1d_obv / e1d_volume_price_trend / e1d_mfi_13 を
#      学習側対応名（_rel / _norm / window別）に修正
#   6. parallel=True を njit から除去（realtime は単スレッド想定）

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

import blueprint as config

# --- [Step 9] core_indicators: Single Source of Truth ---
sys.path.append(str(config.CORE_DIR))
from core_indicators import (
    calculate_atr_wilder,      # Wilder平滑化ATR（学習側と完全統一）
    scale_by_atr,              # ゼロ除算保護付きATR割り
    calculate_sample_weight,   # Zスコアサンプルウェイト
)
# --------------------------------------------------------

import numpy as np
import numba as nb
from numba import njit
from typing import Dict, Optional


# ==================================================================
# 共通ヘルパー関数 (Numba JIT)
# ==================================================================


@njit(fastmath=True, cache=True)
def pct_change_numba(arr: np.ndarray) -> np.ndarray:
    """Polars pct_change() と完全一致: (x[i] - x[i-1]) / x[i-1]
    prev == 0 のとき inf（Polars準拠）、先頭は nan。
    fastmath=True は engine_1_D 学習側の全 UDF と統一。"""
    n = len(arr)
    pct = np.full(n, np.nan, dtype=np.float64)
    if n < 2:
        return pct
    for i in range(1, n):
        prev = arr[i - 1]
        if prev != 0.0:
            pct[i] = (arr[i] - prev) / prev
        else:
            pct[i] = np.inf
    return pct


# ==================================================================
# 1D用 Numba UDF群（Volume・Flow系）
# ==================================================================


@nb.njit(fastmath=True, cache=True)
def cmf_udf(
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    volume: np.ndarray,
    window: int,
) -> np.ndarray:
    """Chaikin Money Flow"""
    n = len(close)
    result = np.full(n, np.nan, dtype=np.float64)
    if n < window:
        return result

    for i in range(window - 1, n):
        mf_vol_sum = 0.0
        vol_sum = 0.0
        for j in range(i - window + 1, i + 1):
            if (
                np.isfinite(high[j])
                and np.isfinite(low[j])
                and np.isfinite(close[j])
                and np.isfinite(volume[j])
            ):
                # 学習側と一致: hl==0 のときはスキップ（vol_sum に加算しない）
                if high[j] != low[j]:
                    hl_range = high[j] - low[j]
                    clv = ((close[j] - low[j]) - (high[j] - close[j])) / (hl_range + 1e-10)
                    mf_vol_sum += clv * volume[j]
                    vol_sum += volume[j]
        if vol_sum > 0:
            result[i] = mf_vol_sum / vol_sum

    return result


@nb.njit(fastmath=True, cache=True)
def mfi_udf(
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    volume: np.ndarray,
    window: int,
) -> np.ndarray:
    """Money Flow Index（parallel=True を除去: realtime は単スレッド想定）"""
    n = len(close)
    result = np.full(n, np.nan, dtype=np.float64)
    if n < window + 1:
        return result

    for i in range(window, n):
        positive_flow = 0.0
        negative_flow = 0.0
        for j in range(i - window + 1, i + 1):
            if j == 0:
                continue
            if (
                np.isfinite(high[j])
                and np.isfinite(low[j])
                and np.isfinite(close[j])
                and np.isfinite(volume[j])
                and np.isfinite(high[j - 1])
                and np.isfinite(low[j - 1])
                and np.isfinite(close[j - 1])
            ):
                typical_price = (high[j] + low[j] + close[j]) / 3.0
                prev_typical_price = (high[j - 1] + low[j - 1] + close[j - 1]) / 3.0
                raw_money_flow = typical_price * volume[j]
                if typical_price > prev_typical_price:
                    positive_flow += raw_money_flow
                elif typical_price < prev_typical_price:
                    negative_flow += raw_money_flow

        if negative_flow > 0:
            money_ratio = positive_flow / negative_flow
            result[i] = 100.0 - (100.0 / (1.0 + money_ratio))
        elif positive_flow > 0:
            result[i] = 100.0
        else:
            result[i] = 50.0

    return result


@nb.njit(fastmath=True, cache=True)
def vwap_udf(
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    volume: np.ndarray,
    window: int,
) -> np.ndarray:
    """Volume Weighted Average Price"""
    n = len(close)
    result = np.full(n, np.nan, dtype=np.float64)
    if n < window:
        return result

    for i in range(window - 1, n):
        pv_sum = 0.0
        vol_sum = 0.0
        for j in range(i - window + 1, i + 1):
            if (
                np.isfinite(high[j])
                and np.isfinite(low[j])
                and np.isfinite(close[j])
                and np.isfinite(volume[j])
            ):
                typical_price = (high[j] + low[j] + close[j]) / 3.0
                pv_sum += typical_price * volume[j]
                vol_sum += volume[j]
        if vol_sum > 0:
            result[i] = pv_sum / vol_sum

    return result


@nb.njit(fastmath=True, cache=True)
def obv_udf(close: np.ndarray, volume: np.ndarray) -> np.ndarray:
    """On Balance Volume（parallel=True を除去）"""
    n = len(close)
    result = np.full(n, np.nan, dtype=np.float64)
    if n < 2:
        return result

    result[0] = volume[0] if np.isfinite(volume[0]) else 0.0
    for i in range(1, n):
        prev_obv = result[i - 1] if np.isfinite(result[i - 1]) else 0.0
        if (
            np.isfinite(close[i])
            and np.isfinite(close[i - 1])
            and np.isfinite(volume[i])
        ):
            if close[i] > close[i - 1]:
                result[i] = prev_obv + volume[i]
            elif close[i] < close[i - 1]:
                result[i] = prev_obv - volume[i]
            else:
                result[i] = prev_obv
        else:
            result[i] = prev_obv

    return result


@nb.njit(fastmath=True, cache=True)
def accumulation_distribution_udf(
    high: np.ndarray, low: np.ndarray, close: np.ndarray, volume: np.ndarray
) -> np.ndarray:
    """Accumulation/Distribution Line"""
    n = len(close)
    result = np.full(n, np.nan, dtype=np.float64)
    if n < 1:
        return result

    result[0] = 0.0
    for i in range(1, n):
        prev_ad = result[i - 1] if np.isfinite(result[i - 1]) else 0.0
        if (
            np.isfinite(high[i])
            and np.isfinite(low[i])
            and np.isfinite(close[i])
            and np.isfinite(volume[i])
        ):
            # 学習側と一致: hl==0 のときは clv 計算をスキップし prev_ad を継続
            if high[i] != low[i]:
                hl_range = high[i] - low[i]
                clv = ((close[i] - low[i]) - (high[i] - close[i])) / (hl_range + 1e-10)
                result[i] = prev_ad + (clv * volume[i])
            else:
                result[i] = prev_ad
        else:
            result[i] = prev_ad

    return result


# ==================================================================
# 1D用 Numba UDF群（Volatility系）
# ==================================================================


@nb.njit(fastmath=True, cache=True)
def chaikin_volatility_udf(
    high: np.ndarray, low: np.ndarray, window: int
) -> np.ndarray:
    """Chaikin Volatility (真のEMAベース: engine_1_D と同一実装)"""
    n = len(high)
    result = np.full(n, np.nan, dtype=np.float64)
    if n < window * 2:
        return result

    hl_range = high - low
    ema = np.full(n, np.nan, dtype=np.float64)

    valid_start = 0
    while valid_start < n and not np.isfinite(hl_range[valid_start]):
        valid_start += 1
    if valid_start + window > n:
        return result

    sma_init = 0.0
    for i in range(valid_start, valid_start + window):
        sma_init += hl_range[i]
    ema[valid_start + window - 1] = sma_init / window

    alpha = 2.0 / (window + 1.0)
    for i in range(valid_start + window, n):
        ema[i] = alpha * hl_range[i] + (1.0 - alpha) * ema[i - 1]

    for i in range(valid_start + window * 2 - 1, n):
        prev_ema = ema[i - window]
        if np.isfinite(ema[i]) and np.isfinite(prev_ema) and prev_ema > 0:
            result[i] = (ema[i] - prev_ema) / prev_ema * 100.0

    return result


@nb.njit(fastmath=True, cache=True)
def mass_index_udf(high: np.ndarray, low: np.ndarray, window: int) -> np.ndarray:
    """Mass Index (真の連続EMA(9) / EMA(EMA(9)) 累積和: engine_1_D と同一実装)"""
    n = len(high)
    result = np.full(n, np.nan, dtype=np.float64)
    if n < 9 + 9 + window:
        return result

    hl_range = high - low
    ema9 = np.full(n, np.nan, dtype=np.float64)
    ema_ema9 = np.full(n, np.nan, dtype=np.float64)
    alpha = 2.0 / (9.0 + 1.0)

    valid_start = 0
    while valid_start < n and not np.isfinite(hl_range[valid_start]):
        valid_start += 1
    if valid_start + 9 > n:
        return result

    sma_init = 0.0
    for i in range(valid_start, valid_start + 9):
        sma_init += hl_range[i]
    ema9[valid_start + 8] = sma_init / 9.0
    for i in range(valid_start + 9, n):
        ema9[i] = alpha * hl_range[i] + (1.0 - alpha) * ema9[i - 1]

    if valid_start + 17 > n:
        return result

    sma_ema_init = 0.0
    for i in range(valid_start + 8, valid_start + 17):
        sma_ema_init += ema9[i]
    ema_ema9[valid_start + 16] = sma_ema_init / 9.0
    for i in range(valid_start + 17, n):
        ema_ema9[i] = alpha * ema9[i] + (1.0 - alpha) * ema_ema9[i - 1]

    for i in range(valid_start + 16 + window - 1, n):
        mi_sum = 0.0
        is_valid = True
        for j in range(i - window + 1, i + 1):
            if not np.isfinite(ema9[j]) or not np.isfinite(ema_ema9[j]) or ema_ema9[j] <= 0:
                is_valid = False
                break
            mi_sum += ema9[j] / ema_ema9[j]
        if is_valid:
            result[i] = mi_sum

    return result


@nb.njit(fastmath=True, cache=True)
def hv_robust_udf(returns: np.ndarray) -> float:
    """ロバストボラティリティ (MADベース, ddof=1相当)"""
    if len(returns) < 5:
        return np.nan
    finite_returns = returns[np.isfinite(returns)]
    if len(finite_returns) < 5:
        return np.nan
    median_return = np.median(finite_returns)
    abs_deviations = np.abs(finite_returns - median_return)
    mad = np.median(abs_deviations)
    return mad * 1.4826


@nb.njit(fastmath=True, cache=True)
def hv_standard_udf(returns: np.ndarray) -> float:
    """標準ヒストリカルボラティリティ (不偏推定量: ddof=1)"""
    if len(returns) < 5:
        return np.nan
    finite_returns = returns[np.isfinite(returns)]
    if len(finite_returns) < 5:
        return np.nan
    mean_return = np.mean(finite_returns)
    squared_deviations = (finite_returns - mean_return) ** 2
    variance = np.sum(squared_deviations) / (len(finite_returns) - 1)
    return np.sqrt(variance)


# ==================================================================
# 1D用 Numba UDF群（Breakout・Support/Resistance・Price Action系）
# ==================================================================


@nb.njit(fastmath=True, cache=True)
def commodity_channel_index_udf(
    high: np.ndarray, low: np.ndarray, close: np.ndarray, window: int
) -> np.ndarray:
    """Commodity Channel Index (CCI)"""
    n = len(close)
    result = np.full(n, np.nan, dtype=np.float64)
    if n < window:
        return result

    typical_prices = np.zeros(window, dtype=np.float64)

    for i in range(window - 1, n):
        typical_prices.fill(0.0)
        valid_count = 0
        for j in range(window):
            idx = i - window + 1 + j
            if (
                np.isfinite(high[idx])
                and np.isfinite(low[idx])
                and np.isfinite(close[idx])
            ):
                typical_prices[j] = (high[idx] + low[idx] + close[idx]) / 3.0
                valid_count += 1
            else:
                typical_prices[j] = np.nan

        if valid_count < window // 2:
            continue

        tp_sum = 0.0
        for k in range(window):
            if np.isfinite(typical_prices[k]):
                tp_sum += typical_prices[k]
        sma = tp_sum / valid_count

        md_sum = 0.0
        for k in range(window):
            if np.isfinite(typical_prices[k]):
                md_sum += abs(typical_prices[k] - sma)
        mean_deviation = md_sum / valid_count

        current_tp = (high[i] + low[i] + close[i]) / 3.0
        if mean_deviation > 0:
            result[i] = (current_tp - sma) / (0.015 * mean_deviation)

    return result


@nb.njit(fastmath=True, cache=True)
def fibonacci_levels_udf(
    high: np.ndarray, low: np.ndarray, window: int
) -> np.ndarray:
    """フィボナッチリトレースメントレベル (5レベル)"""
    n = len(high)
    result = np.full((n, 5), np.nan, dtype=np.float64)
    if n < window:
        return result

    fib_ratios = np.array([0.236, 0.382, 0.500, 0.618, 0.786])

    for i in range(window - 1, n):
        period_high = -np.inf
        period_low = np.inf
        for j in range(i - window + 1, i + 1):
            if np.isfinite(high[j]) and high[j] > period_high:
                period_high = high[j]
            if np.isfinite(low[j]) and low[j] < period_low:
                period_low = low[j]
        if np.isfinite(period_high) and np.isfinite(period_low):
            price_range = period_high - period_low
            for k in range(5):
                result[i, k] = period_high - (fib_ratios[k] * price_range)

    return result


@nb.njit(fastmath=True, cache=True)
def candlestick_patterns_udf(
    open_prices: np.ndarray,
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
) -> np.ndarray:
    """ローソク足パターン認識（parallel=True 除去・dtype=float64 統一）"""
    n = len(close)
    result = np.full(n, 0.0, dtype=np.float64)

    for i in range(n):
        if not (
            np.isfinite(open_prices[i])
            and np.isfinite(high[i])
            and np.isfinite(low[i])
            and np.isfinite(close[i])
        ):
            continue

        body_size = abs(close[i] - open_prices[i])
        upper_shadow = high[i] - max(open_prices[i], close[i])
        lower_shadow = min(open_prices[i], close[i]) - low[i]
        total_range = high[i] - low[i]

        if total_range <= 0:
            continue

        body_ratio = body_size / total_range
        upper_shadow_ratio = upper_shadow / total_range
        lower_shadow_ratio = lower_shadow / total_range

        if body_ratio < 0.1:
            result[i] = 3.0
        elif lower_shadow_ratio > 0.6 and upper_shadow_ratio < 0.1:
            result[i] = 1.0
        elif upper_shadow_ratio > 0.6 and lower_shadow_ratio < 0.1:
            result[i] = 2.0
        elif body_ratio > 0.6 and close[i] > open_prices[i]:
            result[i] = 4.0
        elif body_ratio > 0.6 and close[i] < open_prices[i]:
            result[i] = 5.0

    return result


# ==================================================================
# QAState — 学習側 apply_quality_assurance_to_group の等価実装
# ==================================================================


class QAState:
    """
    学習側 apply_quality_assurance_to_group のリアルタイム等価実装。

    学習側の処理（Polars LazyFrame 全系列一括）:
        col_expr = when(col.is_infinite()).then(None).otherwise(col)
        ewm_mean = col_expr.ewm_mean(half_life=HL, adjust=False, ignore_nulls=True)
        ewm_std  = col_expr.ewm_std (half_life=HL, adjust=False, ignore_nulls=True)
        result   = when(col==inf).then(upper).when(col==-inf).then(lower)
                   .otherwise(col).clip(lower, upper).fill_null(0.0).fill_nan(0.0)

    alpha = 1 - exp(-ln2 / half_life)
    EWM_mean[t] = alpha * x[t] + (1-alpha) * EWM_mean[t-1]  (NaN/inf はスキップ)
    EWM_var[t]  = (1-alpha) * (EWM_var[t-1] + alpha * (x[t] - EWM_mean[t-1])^2)

    ⚠️ Polars ewm_std(adjust=False) のデフォルトは bias=False（不偏補正あり）。
    bias_corr = 1 / sqrt(1 - (1-alpha)^(2n)) を乗算して Polars と一致させる。
    n が大きくなると bias_corr → 1.0 に収束（ウォームアップ後は実質影響なし）。

    使い方:
        qa_state = FeatureModule1D.QAState(lookback_bars=1440)
        for bar in live_stream:
            features = FeatureModule1D.calculate_features(data_window, 1440, qa_state)
    """

    def __init__(self, lookback_bars: int = 1440):
        self.alpha: float = 1.0 - np.exp(-np.log(2.0) / max(lookback_bars, 1))
        self._ewm_mean: Dict[str, float] = {}
        self._ewm_var: Dict[str, float] = {}
        # bias=False 補正用: Polars ewm_std は t バー目に sqrt(1/(1-(1-alpha)^(2t))) を乗じる。
        self._ewm_n: Dict[str, int] = {}  # 有効値の累積更新回数（bias 補正に使用）

    def update_and_clip(self, key: str, raw_val: float) -> float:
        """1特徴量の raw_val に QA処理を適用して返す（学習側と完全一致）。"""
        alpha = self.alpha

        # 【inf処理修正】学習側の挙動を再現:
        #   学習側: when(col==inf).then(upper).when(col==-inf).then(lower).otherwise(col).clip()
        #   本番側旧: inf → NaN → 0.0（学習側と不一致）
        #   本番側新: inf を先に記録し、clip時にupper/lower_boundで置き換える。
        is_pos_inf = np.isposinf(raw_val)
        is_neg_inf = np.isneginf(raw_val)
        ewm_input = np.nan if not np.isfinite(raw_val) else raw_val

        # EWM 状態更新（ignore_nulls=True 相当）
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
                new_var   = (1.0 - alpha) * (prev_var + alpha * (ewm_input - prev_mean) ** 2)
                self._ewm_mean[key] = new_mean
                self._ewm_var[key]  = new_var
                self._ewm_n[key]    = self._ewm_n.get(key, 0) + 1

        # ±5σ クリップ
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
# ヘルパー関数
# ==================================================================


@nb.njit(fastmath=False, cache=True)
def _window(arr: np.ndarray, window: int) -> np.ndarray:
    """配列の末尾から `window` 個の要素を取得"""
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


# ==================================================================
# メイン計算モジュール
# ==================================================================


class FeatureModule1D:

    # 外部から FeatureModule1D.QAState としてアクセス可能にする
    QAState = QAState

    @staticmethod
    def calculate_features(
        data: Dict[str, np.ndarray],
        lookback_bars: int = 1440,
        qa_state: Optional[QAState] = None,
    ) -> Dict[str, float]:
        features: Dict[str, float] = {}

        # ① KeyError ガード: 必須キーが存在しない場合は空dictを返す
        for _key in ("close", "high", "low", "open", "volume"):
            if _key not in data:
                return features

        close_arr = data["close"]
        high_arr  = data["high"]
        low_arr   = data["low"]
        open_arr  = data["open"]
        volume_arr = data["volume"]

        if len(close_arr) == 0:
            return features

        close_pct = pct_change_numba(close_arr)

        # ---------------------------------------------------------
        # [Step 9] ATR13 (Wilder統一) — 全ATR割りの共通分母
        # ---------------------------------------------------------
        atr13 = calculate_atr_wilder(high_arr, low_arr, close_arr, 13)
        # [修正3] フォールバックは np.nan で統一（他エンジンと同様）
        # 1e-10 を使うと ATR が無効な場合に以降のATR割りが巨大値になるため不適切
        atr13_last = float(atr13[-1]) if np.isfinite(atr13[-1]) else np.nan
        # ② ATR nan ガード: atr13_last が有効かどうかをフラグで管理
        # 無効な場合は ATR割り特徴量を np.nan で返す（NaN 伝播で統一）
        _atr_valid = np.isfinite(atr13_last)

        # ---------------------------------------------------------
        # [Step 9] サンプルウェイト
        # ---------------------------------------------------------
        features["e1d_sample_weight"] = float(
            calculate_sample_weight(high_arr, low_arr, close_arr)[-1]
        )

        # ---------------------------------------------------------
        # 1. Volatility系指標
        # ---------------------------------------------------------

        # HV standard / robust: window=[10, 20, 30, 50]（学習側 volatility=[10,20,30,50] と統一）
        # 【修正: 問題3】window未満データへのgateを追加。
        # 学習側: rolling_map(..., window_size=w, min_samples=w) → w本未満はNaN。
        # 旧: _window が w未満を返してもUDF内len<5チェックのみで非NaN値を返してしまう。
        for w in [10, 20, 30, 50]:
            pct_w = _window(close_pct, w)
            if len(pct_w) < w:
                features[f"e1d_hv_standard_{w}"] = np.nan
                features[f"e1d_hv_robust_{w}"]   = np.nan
            else:
                features[f"e1d_hv_standard_{w}"] = hv_standard_udf(pct_w)
                features[f"e1d_hv_robust_{w}"]   = hv_robust_udf(pct_w)

        # 年率ボラティリティ (252本基準)
        # 【修正: 問題4】252本未満のgateを追加（rolling_std(252) min_periods=252 と一致）。
        pct_252 = _window(close_pct, 252)
        if len(pct_252) < 252:
            features["e1d_hv_annual_252"]        = np.nan
            features["e1d_hv_robust_annual_252"] = np.nan
        else:
            features["e1d_hv_annual_252"] = (
                hv_standard_udf(pct_252) * np.sqrt(252)
            )
            features["e1d_hv_robust_annual_252"] = (
                hv_robust_udf(pct_252) * np.sqrt(252)
            )

        # ボラティリティレジーム: 直近HV50 vs 過去1440本の各時点HV50の分位数
        # 学習側: Polars rolling_quantile(0.8/0.6, window=1440) on rolling_std(50)
        # realtime側: 過去(1440+50)本の close_pct から各時点の HV50 を計算し
        #             そのうち直近1440本の 80/60 パーセンタイルと比較する
        cur_hv50 = hv_standard_udf(_window(close_pct, 50))
        n_needed = 1440 + 50
        if len(close_pct) >= n_needed and np.isfinite(cur_hv50):
            # 過去 (1440+50) 本の close_pct を取得
            hist_pct = close_pct[-n_needed:]
            # 各時点の HV50 をローリング計算（window=50 が確保できる点のみ）
            hv50_hist = np.full(n_needed, np.nan, dtype=np.float64)
            for _i in range(50 - 1, n_needed):
                hv50_hist[_i] = hv_standard_udf(hist_pct[_i - 49 : _i + 1])
            # 有効な直近1440本の HV50 から分位数を計算
            hv50_window = hv50_hist[-1440:]
            hv50_finite = hv50_window[np.isfinite(hv50_window)]
            if len(hv50_finite) >= 10:
                # Polars rolling_quantile のデフォルトは interpolation="nearest"
                q80 = float(np.percentile(hv50_finite, 80, method="nearest"))
                q60 = float(np.percentile(hv50_finite, 60, method="nearest"))
                features["e1d_hv_regime_50"] = float(
                    int(cur_hv50 > q80) + int(cur_hv50 > q60)
                )
            else:
                features["e1d_hv_regime_50"] = 0.0
        else:
            features["e1d_hv_regime_50"] = 0.0

        # Chaikin Volatility: window=[10, 20]
        for w in [10, 20]:
            features[f"e1d_chaikin_volatility_{w}"] = _last(
                chaikin_volatility_udf(high_arr, low_arr, w)
            )

        # Mass Index: window=[20, 30]
        for w in [20, 30]:
            features[f"e1d_mass_index_{w}"] = _last(
                mass_index_udf(high_arr, low_arr, w)
            )

        # ---------------------------------------------------------
        # 2. Volume・Flow系指標
        # ---------------------------------------------------------

        # CMF: window=[13, 21, 34]
        for w in [13, 21, 34]:
            features[f"e1d_cmf_{w}"] = _last(
                cmf_udf(high_arr, low_arr, close_arr, volume_arr, w)
            )

        # MFI: window=[13, 21, 34]（旧 e1d_mfi_13 → window別に拡張）
        for w in [13, 21, 34]:
            features[f"e1d_mfi_{w}"] = _last(
                mfi_udf(high_arr, low_arr, close_arr, volume_arr, w)
            )

        # VWAP距離 (ATR割り): window=[13, 21, 34]
        # [Step 9] 旧版は VWAP 絶対値のみ → scale_by_atr で相対値化
        for w in [13, 21, 34]:
            vwap_arr = vwap_udf(high_arr, low_arr, close_arr, volume_arr, w)
            dist_arr = close_arr - vwap_arr
            features[f"e1d_vwap_dist_{w}"] = float(scale_by_atr(dist_arr, atr13)[-1])

        # 基準出来高 (1日 ≒ 1440本 MA)
        # [修正4] 定義では素のMAのみ保持し、使用箇所で + 1e-10 を明示する
        vol_ma1440 = float(np.mean(_window(volume_arr, lookback_bars)))

        # OBV relative: diff / vol_ma1440
        # [Step 9] 旧 e1d_obv（累積値そのまま）→ 学習側 e1d_obv_rel に統一
        obv_arr = obv_udf(close_arr, volume_arr)
        obv_diff = np.diff(obv_arr, prepend=np.nan)
        features["e1d_obv_rel"] = float(obv_diff[-1] / (vol_ma1440 + 1e-10))

        # A/D Line relative: diff / vol_ma1440
        ad_arr = accumulation_distribution_udf(high_arr, low_arr, close_arr, volume_arr)
        ad_diff = np.diff(ad_arr, prepend=np.nan)
        features["e1d_accumulation_distribution_rel"] = float(
            ad_diff[-1] / (vol_ma1440 + 1e-10)
        )

        # Force Index normalized: price_change * volume / (atr * vol_ma1440)
        # 【修正: 問題2】分母を atr13_last * vol_ma1440 + 1e-10 に統一。
        # 学習側: atr_13_internal_expr * vol_ma1440 + 1e-10（積の後にepsilon）。
        # 旧本番側: atr13_last * (vol_ma1440 + 1e-10) + 1e-10（vol側に先にepsilon）→ 数値差。
        if _atr_valid and len(close_arr) >= 2:
            price_change = close_arr[-1] - close_arr[-2]
            force_raw = price_change * float(volume_arr[-1])
            features["e1d_force_index_norm"] = float(
                force_raw / (atr13_last * vol_ma1440 + 1e-10)
            )
        else:
            features["e1d_force_index_norm"] = np.nan

        # Volume MA20 relative: ma20 / vol_ma1440
        # [修正4] vol_ma20 の定義では + 1e-10 を持たせず、
        # 使用箇所ごとに明示的にゼロ除算保護を適用する（vol_ma1440 と対称）
        vol_ma20 = float(np.mean(_window(volume_arr, 20)))
        features["e1d_volume_ma20_rel"] = float(vol_ma20 / (vol_ma1440 + 1e-10))

        # Volume ratio: vol[-1] / ma20
        features["e1d_volume_ratio"] = float(float(volume_arr[-1]) / (vol_ma20 + 1e-10))

        # Volume Price Trend normalized: mean(close_pct * volume, 10) / vol_ma1440
        # [Step 9] 旧 e1d_volume_price_trend（絶対値） → 学習側 e1d_volume_price_trend_norm に統一
        # 【修正】np.nanmean → np.mean に変更。
        # 学習側: (pct_change * vol).rolling_mean(10) / vol_ma1440 → NaN/inf はそのまま伝播してQAでclip。
        # 旧本番側: np.nanmean → NaN/infを除外してしまい学習側と不一致。
        vpt_window = _window(close_pct * volume_arr, 10)
        features["e1d_volume_price_trend_norm"] = float(
            np.mean(vpt_window) / (vol_ma1440 + 1e-10)
        )

        # ---------------------------------------------------------
        # 3. Breakout・レンジ系指標 (ATR割り): window=[10, 20, 50, 100]
        # ---------------------------------------------------------
        for w in [10, 20, 50, 100]:
            if len(high_arr) >= w:
                don_upper  = float(np.max(_window(high_arr, w)))
                don_lower  = float(np.min(_window(low_arr, w)))
                don_middle = (don_upper + don_lower) / 2.0
            else:
                don_upper = don_lower = don_middle = np.nan

            # ② ATR nan ガード: atr13_last と各値の両方が有効な場合のみ ATR 割りを実行
            if _atr_valid and np.isfinite(don_upper):
                features[f"e1d_donchian_upper_dist_{w}"]  = float((don_upper  - close_arr[-1]) / (atr13_last + 1e-10))
            else:
                features[f"e1d_donchian_upper_dist_{w}"]  = np.nan

            if _atr_valid and np.isfinite(don_middle):
                features[f"e1d_donchian_middle_dist_{w}"] = float((don_middle - close_arr[-1]) / (atr13_last + 1e-10))
            else:
                features[f"e1d_donchian_middle_dist_{w}"] = np.nan

            if _atr_valid and np.isfinite(don_lower):
                features[f"e1d_donchian_lower_dist_{w}"]  = float((close_arr[-1] - don_lower)  / (atr13_last + 1e-10))
            else:
                features[f"e1d_donchian_lower_dist_{w}"]  = np.nan

            # [Step 9 修正2] price_channel は学習側でも donchian と完全同値
            # (p_upper = high.rolling_max(w), p_lower = low.rolling_min(w))
            # 冗長特徴量のため削除。学習側でも同じ値が格納されている。
            features[f"e1d_price_channel_upper_dist_{w}"] = features[f"e1d_donchian_upper_dist_{w}"]
            features[f"e1d_price_channel_lower_dist_{w}"] = features[f"e1d_donchian_lower_dist_{w}"]

        # CCI: window=[14, 20]（ATR割り不要のため _atr_valid ガード不要）
        for w in [14, 20]:
            features[f"e1d_commodity_channel_index_{w}"] = _last(
                commodity_channel_index_udf(high_arr, low_arr, close_arr, w)
            )

        # ---------------------------------------------------------
        # 4. Support・Resistance系指標 (ATR割り)
        # ② ATR nan ガード: atr13_last が無効な場合は全て nan を設定してスキップ
        # ---------------------------------------------------------
        if not _atr_valid:
            for k in ("e1d_pivot_dist", "e1d_resistance1_dist", "e1d_support1_dist",
                      "e1d_fib_level_50_dist"):
                features[k] = np.nan
        else:
            # ローリングピボット（直近20期間の高安 + 1本前のclose）
            if len(high_arr) >= 21 and len(low_arr) >= 21:
                prev_high_20 = float(np.max(high_arr[-21:-1]))
                prev_low_20 = float(np.min(low_arr[-21:-1]))
                prev_close_1 = float(close_arr[-2])
                pivot = (prev_high_20 + prev_low_20 + prev_close_1) / 3.0
                r1 = 2.0 * pivot - prev_low_20
                s1 = 2.0 * pivot - prev_high_20
                features["e1d_pivot_dist"] = float(
                    (close_arr[-1] - pivot) / (atr13_last + 1e-10)
                )
                features["e1d_resistance1_dist"] = float(
                    (r1 - close_arr[-1]) / (atr13_last + 1e-10)
                )
                features["e1d_support1_dist"] = float(
                    (close_arr[-1] - s1) / (atr13_last + 1e-10)
                )
            else:
                features["e1d_pivot_dist"] = np.nan
                features["e1d_resistance1_dist"] = np.nan
                features["e1d_support1_dist"] = np.nan

            # フィボナッチ 50%レベル距離
            # fib_arr は shape=(n,5)。[:, 2] で 50%レベルの1D列を取り出してから _last を適用
            # → _last(1D配列)[-1] で最終行の50%レベルを取得（動作確認済み）
            fib_arr = fibonacci_levels_udf(high_arr, low_arr, 50)
            fib50 = _last(fib_arr[:, 2])
            features["e1d_fib_level_50_dist"] = (
                float((close_arr[-1] - fib50) / (atr13_last + 1e-10))
                if np.isfinite(fib50)
                else np.nan
            )

        # ローソク足パターン
        features["e1d_candlestick_pattern"] = _last(
            candlestick_patterns_udf(open_arr, high_arr, low_arr, close_arr)
        )

        # ---------------------------------------------------------
        # 5. Price Action系指標 (ATR割り)
        # ② ATR nan ガード: body_size_atr / *_dist 系のみ ATR に依存
        #    hl_range 比率系（wick / price_location）は ATR 不要のため常時計算
        # ---------------------------------------------------------
        typical_p = (high_arr[-1] + low_arr[-1] + close_arr[-1]) / 3.0
        weighted_c = (high_arr[-1] + low_arr[-1] + 2.0 * close_arr[-1]) / 4.0
        median_p = (high_arr[-1] + low_arr[-1]) / 2.0

        if _atr_valid:
            features["e1d_typical_price_dist"] = float(
                (typical_p - close_arr[-1]) / (atr13_last + 1e-10)
            )
            features["e1d_weighted_close_dist"] = float(
                (weighted_c - close_arr[-1]) / (atr13_last + 1e-10)
            )
            features["e1d_median_price_dist"] = float(
                (median_p - close_arr[-1]) / (atr13_last + 1e-10)
            )
            body_size = abs(close_arr[-1] - open_arr[-1])
            features["e1d_body_size_atr"] = float(body_size / (atr13_last + 1e-10))
        else:
            features["e1d_typical_price_dist"] = np.nan
            features["e1d_weighted_close_dist"] = np.nan
            features["e1d_median_price_dist"] = np.nan
            features["e1d_body_size_atr"] = np.nan

        # HL比率系は ATR 不依存のため常時計算
        hl_range = high_arr[-1] - low_arr[-1] + 1e-10
        features["e1d_upper_wick_ratio"] = float(
            (high_arr[-1] - max(open_arr[-1], close_arr[-1])) / hl_range
        )
        features["e1d_lower_wick_ratio"] = float(
            (min(open_arr[-1], close_arr[-1]) - low_arr[-1]) / hl_range
        )
        features["e1d_price_location_hl"] = float(
            (close_arr[-1] - low_arr[-1]) / hl_range
        )

        # イントラデイ・オーバーナイト（ATR不依存）
        features["e1d_intraday_return"] = float(
            (close_arr[-1] - open_arr[-1]) / (open_arr[-1] + 1e-10)
        )

        if len(close_arr) > 1:
            prev_close = close_arr[-2]
            features["e1d_overnight_gap"] = float(
                (open_arr[-1] - prev_close) / (prev_close + 1e-10)
            )
        else:
            features["e1d_overnight_gap"] = 0.0

        # ----------------------------------------------------------
        # QA処理 — 学習側 apply_quality_assurance_to_group と等価
        #   学習側: inf→null → EWM(half_life=lookback_bars)±5σクリップ → fill_null/nan(0.0)
        #   qa_state=None の場合: inf/NaN → 0.0 のみ（後方互換）
        # ----------------------------------------------------------
        if qa_state is not None:
            qa_result: Dict[str, float] = {}
            for key, val in features.items():
                qa_result[key] = qa_state.update_and_clip(key, val)
            features = qa_result
        else:
            for key in list(features.keys()):
                if not np.isfinite(features[key]):
                    features[key] = 0.0

        return features
