# realtime_feature_engine_1D_volume.py

import numpy as np
import numba as nb
from numba import njit
from typing import Dict

"""
Project Cimera V5 - Feature Engine Module 1D
【Volume, Volatility & Price Action】

対象プレフィックス: e1d_
概要: 出来高、ボラティリティ、およびプライスアクション（ローソク足）に関する
純粋な数学的計算のみを提供する独立モジュール。

【修正履歴】
v1.0 → v1.1 (QA監査による修正):
  - hv_standard_udf: np.std()(標本標準偏差)を不偏推定量(ddof=1相当)の手動計算に修正
  - hv_standard_udf: 最小サンプル閾値を 2 → 5 に修正（元スクリプト準拠）
  - chaikin_volatility_udf: @nb.njit に parallel=True を追加、アルゴリズムをnb.prange方式に差し替え
  - mass_index_udf: parallel=True を追加、アルゴリズムをnb.prange+二重ループ方式に差し替え
  - cmf_udf: parallel=True を追加、isfinite()ガードを追加、nb.prange方式に差し替え
  - mfi_udf: parallel=True を追加、isfinite()ガードを追加、nb.prange方式に差し替え
  - obv_udf: parallel=True を追加、isfinite()ガードを追加、result初期化をnp.full(n,np.nan)に修正
  - accumulation_distribution_udf: parallel=True を追加、isfinite()ガードを追加、index0をresult[0]=0.0固定に修正
  - force_index_udf: parallel=True を追加、isfinite()ガードを追加、result初期化をnp.full(n,np.nan)に修正
  - candlestick_patterns_udf: parallel=True を追加、isfinite()ガードを追加
"""

# ==================================================================
# 共通ヘルパー関数 (Numba JIT)
# ==================================================================


@njit(fastmath=True, cache=True)
def pct_change_numba(arr: np.ndarray) -> np.ndarray:
    """
    配列の騰落率（pct_change）を計算するヘルパー関数。
    ボラティリティ系指標（HV等）の計算前処理として使用します。
    先頭要素は np.nan となります。
    """
    n = len(arr)
    pct = np.full(n, np.nan, dtype=np.float64)
    if n < 2:
        return pct

    for i in range(1, n):
        prev = arr[i - 1]
        if prev != 0.0:
            pct[i] = (arr[i] - prev) / prev
        else:
            pct[i] = 0.0

    return pct


@njit(fastmath=True, cache=True)
def get_window_numba(arr: np.ndarray, window: int) -> np.ndarray:
    """
    配列の末尾から指定されたウィンドウサイズの要素を取得するヘルパー関数。
    データ不足時の安全なスライス操作を保証します。
    """
    n = len(arr)
    if window <= 0:
        return np.empty(0, dtype=arr.dtype)
    if window >= n:
        return arr
    return arr[n - window :]


# ==================================================================
# 1D用 Numba UDF群（前半：Volume・Flow系指標）
# ==================================================================


@nb.njit(fastmath=True, cache=True, parallel=True)
def accumulation_distribution_udf(
    high: np.ndarray, low: np.ndarray, close: np.ndarray, volume: np.ndarray
) -> np.ndarray:
    """
    Accumulation/Distribution Line (Numba JIT) - 配列返し
    対応: e1d_accumulation_distribution

    【修正箇所】
    - @nb.njit に parallel=True を追加（元スクリプト準拠）
    - result 初期化を np.full(n, np.nan) に変更
    - result[0] = 0.0 固定（元スクリプト準拠）。旧版は index0 でも clv*volume 計算をしていたが誤り
    - 全インデックスに np.isfinite() ガードを追加（元スクリプト準拠）
    - NaN入力時は prev_ad を維持する安全処理を追加
    """
    n = len(close)
    result = np.full(n, np.nan)

    if n < 1:
        return result

    result[0] = 0.0

    for i in nb.prange(1, n):
        prev_ad = result[i - 1] if np.isfinite(result[i - 1]) else 0.0

        if (
            np.isfinite(high[i])
            and np.isfinite(low[i])
            and np.isfinite(close[i])
            and np.isfinite(volume[i])
        ):
            if high[i] != low[i]:
                # Close Location Value = [(Close - Low) - (High - Close)] / (High - Low)
                clv = ((close[i] - low[i]) - (high[i] - close[i])) / (high[i] - low[i])
                result[i] = prev_ad + (clv * volume[i])
            else:
                result[i] = prev_ad
        else:
            result[i] = prev_ad

    return result


@nb.njit(fastmath=True, cache=True, parallel=True)
def chaikin_volatility_udf(
    high: np.ndarray, low: np.ndarray, window: int
) -> np.ndarray:
    """
    Chaikin Volatility (Numba JIT) - 配列返し
    対応: e1d_chaikin_volatility_10

    【修正箇所】
    - @nb.njit に parallel=True を追加（元スクリプト準拠）
    - アルゴリズムを nb.prange による各インデックス独立計算方式に差し替え（元スクリプト準拠）
      旧版はインクリメンタル sma 更新方式だったが、元スクリプトは各 i ごとに
      現在ウィンドウと前期間ウィンドウを独立集計する実装
    - prev_avg > 0 のガード条件を元スクリプト準拠（prev_sum <= 0 で continue）に統一
    """
    n = len(high)
    result = np.full(n, np.nan)

    if n < window * 2:
        return result

    # High-Low範囲計算
    hl_range = high - low

    # 範囲の移動平均計算
    for i in nb.prange(window - 1, n):
        if i >= window * 2 - 1:
            # 現在期間の移動平均
            current_sum = 0.0
            current_count = 0
            for j in range(i - window + 1, i + 1):
                if np.isfinite(hl_range[j]):
                    current_sum += hl_range[j]
                    current_count += 1

            if current_count < window // 2:
                continue

            current_avg = current_sum / current_count

            # 前期間の移動平均
            prev_sum = 0.0
            prev_count = 0
            for j in range(i - window * 2 + 1, i - window + 1):
                if np.isfinite(hl_range[j]):
                    prev_sum += hl_range[j]
                    prev_count += 1

            if prev_count < window // 2 or prev_sum <= 0:
                continue

            prev_avg = prev_sum / prev_count

            # Chaikin Volatility = (現在の移動平均 - 前期間の移動平均) / 前期間の移動平均 * 100
            if prev_avg > 0:
                result[i] = (current_avg - prev_avg) / prev_avg * 100.0

    return result


@nb.njit(fastmath=True, cache=True, parallel=True)
def cmf_udf(
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    volume: np.ndarray,
    window: int,
) -> np.ndarray:
    """
    Chaikin Money Flow (Numba JIT) - 配列返し
    対応: e1d_cmf_13

    【修正箇所】
    - @nb.njit に parallel=True を追加（元スクリプト準拠）
    - アルゴリズムを nb.prange による各インデックス独立計算方式に差し替え（元スクリプト準拠）
      旧版は累積加算インクリメンタル更新方式だったが不一致
    - 全インデックスに np.isfinite() ガードを追加（元スクリプト準拠）
    """
    n = len(close)
    result = np.full(n, np.nan)

    if n < window:
        return result

    for i in nb.prange(window - 1, n):
        mf_volume_sum = 0.0
        volume_sum = 0.0

        for j in range(i - window + 1, i + 1):
            if (
                np.isfinite(high[j])
                and np.isfinite(low[j])
                and np.isfinite(close[j])
                and np.isfinite(volume[j])
            ):
                if high[j] != low[j]:
                    # Money Flow Multiplier = [(Close - Low) - (High - Close)] / (High - Low)
                    mf_multiplier = ((close[j] - low[j]) - (high[j] - close[j])) / (
                        high[j] - low[j]
                    )
                    mf_volume = mf_multiplier * volume[j]
                    mf_volume_sum += mf_volume
                    volume_sum += volume[j]

        if volume_sum > 0:
            result[i] = mf_volume_sum / volume_sum

    return result


@nb.njit(fastmath=True, cache=True, parallel=True)
def force_index_udf(close: np.ndarray, volume: np.ndarray) -> np.ndarray:
    """
    Force Index (Numba JIT) - 配列返し
    対応: e1d_force_index

    【修正箇所】
    - @nb.njit に parallel=True を追加（元スクリプト準拠）
    - result 初期化を np.full(n, np.nan) + result[0]=0.0 に修正（元スクリプト準拠）
      旧版は np.zeros(n) で全ゼロ初期化だったが不一致
    - np.isfinite() ガードを追加（元スクリプト準拠）
    - NaN入力時は result[i] = 0.0 の安全処理を追加（元スクリプト準拠）
    """
    n = len(close)
    result = np.full(n, np.nan)

    if n < 2:
        return result

    result[0] = 0.0

    for i in nb.prange(1, n):
        if (
            np.isfinite(close[i])
            and np.isfinite(close[i - 1])
            and np.isfinite(volume[i])
        ):
            price_change = close[i] - close[i - 1]
            result[i] = price_change * volume[i]
        else:
            result[i] = 0.0

    return result


@nb.njit(fastmath=True, cache=True, parallel=True)
def mass_index_udf(high: np.ndarray, low: np.ndarray, window: int) -> np.ndarray:
    """
    Mass Index (Numba JIT) - 配列返し
    対応: e1d_mass_index_20

    【修正箇所】
    - @nb.njit に parallel=True を追加（元スクリプト準拠）
    - アルゴリズムを nb.prange + 二重ループによる各インデックス独立計算に差し替え（元スクリプト準拠）
      旧版は calc_ema ネスト関数で全配列事前計算する方式だったが別アルゴリズム
    - ガード条件を `n < window` に修正（元スクリプト準拠）
      旧版は `n < window + 16` だった
    - EMA の再計算ロジックを元スクリプトの二重ループ構造に完全準拠
    """
    n = len(high)
    result = np.full(n, np.nan)

    if n < window:
        return result

    # High-Low範囲
    hl_range = high - low

    # EMA(9)とEMA(EMA(9))の計算用
    alpha = 2.0 / (9.0 + 1.0)

    for i in nb.prange(window - 1, n):
        # 25期間のMass Index累積計算
        mass_sum = 0.0
        valid_count = 0

        for k in range(max(0, i - window + 1), i + 1):
            if k < 8:  # EMA計算に必要な最小期間
                continue

            # EMA(9)計算
            ema9 = hl_range[k - 8]
            for j in range(k - 7, k + 1):
                if np.isfinite(hl_range[j]):
                    ema9 = alpha * hl_range[j] + (1.0 - alpha) * ema9

            # EMA(EMA(9))計算
            if k >= 16:  # EMA of EMA計算に必要な期間
                ema_ema9 = ema9
                start_idx = max(k - 16, 0)

                # 簡略化されたEMA(EMA)近似
                for j in range(start_idx, k + 1):
                    if j >= 8:
                        temp_ema = hl_range[j - 8]
                        for m in range(j - 7, j + 1):
                            if np.isfinite(hl_range[m]):
                                temp_ema = (
                                    alpha * hl_range[m] + (1.0 - alpha) * temp_ema
                                )
                        ema_ema9 = alpha * temp_ema + (1.0 - alpha) * ema_ema9

                # Mass Index = EMA(9) / EMA(EMA(9))
                if ema_ema9 > 0 and np.isfinite(ema9):
                    mass_sum += ema9 / ema_ema9
                    valid_count += 1

        if valid_count >= window // 2:
            result[i] = mass_sum

    return result


@nb.njit(fastmath=True, cache=True, parallel=True)
def mfi_udf(
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    volume: np.ndarray,
    window: int,
) -> np.ndarray:
    """
    Money Flow Index (Numba JIT) - 配列返し
    対応: e1d_mfi_13

    【修正箇所】
    - @nb.njit に parallel=True を追加（元スクリプト準拠）
    - アルゴリズムを nb.prange による各インデックス独立計算方式に差し替え（元スクリプト準拠）
      旧版は pos_flow/neg_flow 配列を全走査後に滑動加算する方式で不一致
    - np.isfinite() ガードを追加（元スクリプト準拠）
    - 前の足の typical_price も isfinite チェック対象に追加（元スクリプト準拠）
    """
    n = len(close)
    result = np.full(n, np.nan)

    if n < window + 1:
        return result

    # Typical Price = (High + Low + Close) / 3
    for i in nb.prange(window, n):
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


@nb.njit(fastmath=True, cache=True, parallel=True)
def obv_udf(close: np.ndarray, volume: np.ndarray) -> np.ndarray:
    """
    On Balance Volume (Numba JIT) - 配列返し
    対応: e1d_obv

    【修正箇所】
    - @nb.njit に parallel=True を追加（元スクリプト準拠）
    - result 初期化を np.full(n, np.nan) に修正（元スクリプト準拠）
      旧版は np.zeros(n) で全ゼロ初期化だったが不一致
    - result[0] に isfinite ガードを追加: `volume[0] if np.isfinite(volume[0]) else 0.0`
    - NaN 入力時は result[i] = prev_obv を維持する安全処理を追加（元スクリプト準拠）
    """
    n = len(close)
    result = np.full(n, np.nan)

    if n < 2:
        return result

    result[0] = volume[0] if np.isfinite(volume[0]) else 0.0

    for i in nb.prange(1, n):
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


# ==================================================================
# 1D用 Numba UDF群（後半：Volatility・Price Action系指標）
# ==================================================================


@nb.njit(fastmath=True, cache=True, parallel=True)
def candlestick_patterns_udf(
    open_prices: np.ndarray, high: np.ndarray, low: np.ndarray, close: np.ndarray
) -> np.ndarray:
    """
    ローソク足パターン認識 (Numba JIT) - 配列返し
    対応: e1d_candlestick_pattern
    Returns: パターンID配列 (0=なし, 1=ハンマー, 2=流れ星, 3=同事, 4=強気, 5=弱気)

    【修正箇所】
    - @nb.njit に parallel=True を追加（元スクリプト準拠）
    - result 初期化を np.full(n, 0) に変更（元スクリプト準拠）
      旧版は np.zeros(n, dtype=np.float64) だったが型が不一致
    - np.isfinite() ガードを追加（元スクリプト準拠）
    """
    n = len(close)
    result = np.full(n, 0)

    for i in nb.prange(n):
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

        # 実体・ヒゲ比率
        body_ratio = body_size / total_range
        upper_shadow_ratio = upper_shadow / total_range
        lower_shadow_ratio = lower_shadow / total_range

        # 同事（実体が小さい）
        if body_ratio < 0.1:
            result[i] = 3
        # ハンマー（下ヒゲが長く、上ヒゲが短い）
        elif lower_shadow_ratio > 0.6 and upper_shadow_ratio < 0.1:
            result[i] = 1
        # 流れ星（上ヒゲが長く、下ヒゲが短い）
        elif upper_shadow_ratio > 0.6 and lower_shadow_ratio < 0.1:
            result[i] = 2
        # 強気（実体が大きく陽線）
        elif body_ratio > 0.6 and close[i] > open_prices[i]:
            result[i] = 4
        # 弱気（実体が大きく陰線）
        elif body_ratio > 0.6 and close[i] < open_prices[i]:
            result[i] = 5

    return result


@nb.njit(fastmath=True, cache=True)
def hv_robust_udf(returns: np.ndarray) -> float:
    """
    ロバストボラティリティ (Numba JIT) - 単一値
    対応: e1d_hv_robust_10, 20, 30, 50, annual_252, e1d_hv_regime_50

    【差異なし】元スクリプトと完全一致を確認済み。
    """
    if len(returns) < 5:
        return np.nan

    finite_returns = returns[np.isfinite(returns)]
    if len(finite_returns) < 5:
        return np.nan

    # MADベースロバスト標準偏差
    median_return = np.median(finite_returns)
    abs_deviations = np.abs(finite_returns - median_return)
    mad = np.median(abs_deviations)

    # MADを標準偏差に変換（正規分布仮定下）
    robust_volatility = mad * 1.4826  # 1.4826 = 1/Φ^(-1)(0.75)

    return robust_volatility


@nb.njit(fastmath=True, cache=True)
def hv_standard_udf(returns: np.ndarray) -> float:
    """
    標準ヒストリカルボラティリティ (Numba JIT) - 単一値
    対応: e1d_hv_standard_10, 30, 50

    【修正箇所】
    1. 最小サンプル閾値を 2 → 5 に修正（元スクリプト準拠）
       旧版: `if len(returns) < 2` / `if len(finite_returns) < 2`
    2. 分散計算を np.std()(標本標準偏差, ddof=0) から
       不偏推定量の手動計算 (ddof=1相当) に修正（元スクリプト準拠）
       旧版: return np.std(finite_returns)
       正版: mean → squared_deviations → sum / (n-1) → sqrt
    """
    if len(returns) < 5:
        return np.nan

    finite_returns = returns[np.isfinite(returns)]
    if len(finite_returns) < 5:
        return np.nan

    # 標準偏差計算（不偏推定量: ddof=1）
    mean_return = np.mean(finite_returns)
    squared_deviations = (finite_returns - mean_return) ** 2
    variance = np.sum(squared_deviations) / (len(finite_returns) - 1)
    volatility = np.sqrt(variance)

    return volatility
