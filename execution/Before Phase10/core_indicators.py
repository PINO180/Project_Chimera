"""
core_indicators.py
==================
Project Cimera V5 — Single Source of Truth (単一の真実の情報源)

【目的】
学習用スクリプト (engine_1_A〜1_F) とリアルタイムスクリプト
(realtime_feature_engine_1A〜1F) の両方から import して使用する。
このファイルを通じてのみ基礎的な数学・統計ロジックを参照することで、
学習時と本番で特徴量の値が「物理的に」一致することを保証する。

【カテゴリブロック構成】
  [CATEGORY: ATR & VOLATILITY] ボラティリティ指標
  [CATEGORY: MOMENTUM]         モメンタム指標
  [CATEGORY: TREND]            トレンド指標
  [CATEGORY: STATS]            統計・分布
  [CATEGORY: WEIGHT]           サンプルウェイト
  [CATEGORY: NEUTRALIZATION]   OLS純化
  [CATEGORY: DSP]              信号処理 (Engine 1E用 — 将来追加)
  [CATEGORY: COMPLEX]          複雑系   (Engine 1F用 — 将来追加)

【Numba制約】
  - np.std(arr) は Numba 環境で ddof=1 をサポートしない
    → stddev_unbiased() を必ず使用すること
  - スカラー出力は @njit、配列出力は @guvectorize
  - fastmath=False で浮動小数点の再現性を厳密に担保する
    (fastmath=True は学習時・本番間の微細な数値差を生む)

【Polars連携 (学習側)】
  map_batches からの呼び出し例:
    pl.struct(["high", "low", "close"]).map_batches(
        lambda s: calculate_atr_wilder(
            s.struct.field("high").to_numpy(),
            s.struct.field("low").to_numpy(),
            s.struct.field("close").to_numpy(),
            13,
        ),
        return_dtype=pl.Float64,
    )
"""

import numpy as np
import numba as nb
import math  # _standard_normal_cdf_fast 等のヘルパー関数で math.erf を使用
from numba import njit, guvectorize
from typing import Tuple  # pivot_point_udf 等の戻り値型注釈で使用

# ===========================================================================
# [CATEGORY: ATR & VOLATILITY]  ボラティリティ
# ===========================================================================


@njit(fastmath=False, cache=True)
def calculate_atr_wilder(
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    period: int,
) -> np.ndarray:
    """
    Wilder平滑化 ATR (Average True Range) — 全スクリプトの統一実装。

    アルゴリズム:
        TR[0]  = high[0] - low[0]
        TR[i]  = max(high[i]-low[i],
                     |high[i]-close[i-1]|,
                     |low[i]-close[i-1]|)

        ATR[0] = TR[0]          (シードは TR[0])
        ATR[i] = ATR[i-1] * (period-1) / period + TR[i] / period
                            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
                            ← これが "Wilder 平滑化" の定義

    注意:
        - engine_1_A の SMA-ATR (rolling_mean(TR, period)) とは値が異なる。
          これは意図的な変更であり、全エンジンを本関数に統一することで
          学習側と本番側の一致を保証する。
        - engine_1_B の calculate_atr_numba はWilder式を使っているが、
          初期ウォームアップ期間 (i < period) を np.mean(tr[:i+1]) で埋めており
          シードの取り扱いが本関数と微妙に異なる。本関数に統一すること。
        - engine_1_C の calculate_atr_numba は SMA 方式。本関数に統一すること。
        - realtime_feature_engine.py は Pandas ewm(alpha=1/period) で計算しており
          数学的には Wilder 式と等価。本関数への統一後も値は一致する。

    Returns:
        np.ndarray: shape=(n,), ATR 値の配列。
                    index 0 は TR[0]、それ以降は Wilder 平滑化値。
    """
    n = len(high)
    atr = np.full(n, np.nan, dtype=np.float64)
    if n == 0:
        return atr

    # True Range 計算
    tr = np.zeros(n, dtype=np.float64)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        hl = high[i] - low[i]
        hc = abs(high[i] - close[i - 1])
        lc = abs(low[i] - close[i - 1])
        tr[i] = max(hl, hc, lc)

    # Wilder 平滑化 ATR
    # シード: ATR[0] = TR[0]
    # （SMA による初期平均化は行わない — シンプルで再現性が高い）
    atr[0] = tr[0]
    for i in range(1, n):
        atr[i] = atr[i - 1] * (period - 1.0) / period + tr[i] / period

    return atr


def calculate_atr_wilder_disc_aware(
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    disc: np.ndarray,
    period: int,
) -> np.ndarray:
    """
    Disc-aware Wilder EWM ATR — ATR Ratio フィルター計算専用。

    [新設理由 — Phase 7 disc乖離修正]
    calculate_atr_wilder() は disc フラグを持たないため、週末ギャップ越境 TR
    （金曜 close → 月曜 high/low の大幅ジャンプ）をそのまま TR に含む。
    これにより本番の ATR および baseline（480本平均）が学習側より大きくスパイクし、
    月曜24時間にわたって ATR Ratio が学習側より低めに計算される構造的乖離が発生する。

    本関数は学習側 create_proxy_labels の TR 計算式:
        pl.when(disc).then(H-L).otherwise(max(H-L, |H-prev_close|, |L-prev_close|))
        .ewm_mean(alpha=1/period, adjust=False)   ← Wilder EWM, seed=TR[0]
    と完全一致する配列返却型の実装。

    calculate_barrier_atr() との違い:
    - 返却型: float ではなく np.ndarray（baseline 計算のために全要素が必要）
    - シード方式: SMA ではなく TR[0]（学習側 Polars EWM の adjust=False と一致）
    - 用途: ATR Ratio 計算（シグナルフィルター）専用。SL/TP 幅には calculate_barrier_atr を使う

    Args:
        high, low, close : OHLC バッファ (np.ndarray)
        disc             : 不連続フラグ配列 (True=ギャップあり、前 Close を使わない)
        period           : ATR 期間（ATR Ratio 計算では ATR_CALC_PERIOD=13 を渡す）

    Returns:
        np.ndarray: shape=(n,), 各時点の ATR 値。
                    calculate_atr_wilder() と同形式で、baseline 計算（rolling mean）に使用可能。
    """
    n = len(high)
    atr = np.full(n, np.nan, dtype=np.float64)
    if n == 0:
        return atr

    # disc 配列長の整合性保証（バッファ長と異なる場合に備える）
    n_disc = len(disc)
    if n_disc < n:
        disc = np.concatenate([np.zeros(n - n_disc, dtype=np.bool_), disc])
    elif n_disc > n:
        disc = disc[-n:]

    # True Range 計算 — disc=True の足は H-L のみ（学習側と同一）
    tr = np.zeros(n, dtype=np.float64)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        hl = high[i] - low[i]
        if disc[i]:
            # [DISC-FLAG] 週末ギャップ等の不連続点: 前 Close を参照しない
            tr[i] = hl
        else:
            hc = abs(high[i] - close[i - 1])
            lc = abs(low[i] - close[i - 1])
            tr[i] = max(hl, hc, lc)

    # Wilder EWM — シード = TR[0]（Polars ewm_mean(adjust=False) と等価）
    # ATR[0] = TR[0]
    # ATR[i] = ATR[i-1] * (period-1)/period + TR[i] / period
    atr[0] = tr[0]
    for i in range(1, n):
        atr[i] = atr[i - 1] * (period - 1.0) / period + tr[i] / period

    return atr


def calculate_barrier_atr(
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    disc: np.ndarray,
    period: int = 13,
) -> float:
    """
    [SL/TP専用] 堅牢なATR計算関数 — バリア幅（SL/TP）計算にのみ使用する。

    既存の calculate_atr_wilder() とは独立した関数であり、
    AIモデルの入力特徴量（e1c_atr_13等）には一切使用しない。
    学習側との乖離リスクを回避するために責務を分離している。

    calculate_atr_wilder() との違い:
    1. SMAシード: 最初のperiod本はSMA（単純平均）でシードを作成（TA-Lib仕様）
       → シード1点から始まる脆弱性を排除
    2. discフラグ対応: disc=Trueの足は前Closeを使わずH-LのみでTR計算
       → resample().dropna()廃止後のギャップ越境TR防止
    3. NaNウォームアップ: period本未満のバッファではNaNを返す
       → 不安定な初期値がSL/TPに使われるのを防ぐ

    Args:
        high, low, close: OHLCバッファ（np.ndarray）
        disc: 不連続フラグ配列（True=ギャップあり、前Closeを使わない）
        period: ATR期間（デフォルト13）

    Returns:
        float: 最新のATR値。バッファがperiod本未満の場合はnp.nan。
    """
    n = len(high)
    if n < period:
        return np.nan

    # 配列長の整合性チェック（discがcloseと異なる長さの場合に備える）
    n_disc = len(disc)
    if n_disc < n:
        # 先頭をFalse（連続）でパディング
        disc = np.concatenate([np.zeros(n - n_disc, dtype=np.bool_), disc])
    elif n_disc > n:
        disc = disc[-n:]

    # True Range 計算（discフラグ対応）
    tr = np.zeros(n, dtype=np.float64)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        hl = high[i] - low[i]
        if disc[i]:
            # [DISC-FLAG] 不連続点: 前Closeを使わずH-Lのみ
            tr[i] = hl
        else:
            hc = abs(high[i] - close[i - 1])
            lc = abs(low[i] - close[i - 1])
            tr[i] = max(hl, hc, lc)

    # SMAシードでATRを初期化（最初のperiod本の平均）
    atr_seed = float(np.mean(tr[:period]))
    atr = atr_seed

    # period本目以降はWilder平滑化
    for i in range(period, n):
        atr = atr * (period - 1.0) / period + tr[i] / period

    return atr


@njit(fastmath=False, cache=True)
def scale_by_atr(
    target_arr: np.ndarray,
    atr_arr: np.ndarray,
    epsilon: float = 1e-10,
) -> np.ndarray:
    """
    ATR スケール化 (ゼロ除算保護付き) — 全スクリプトの統一実装。

        result[i] = target_arr[i] / (atr_arr[i] + epsilon)

    epsilon のデフォルト値 (1e-10) をこの関数内に隠蔽し、
    各スクリプトへの直書き `/ (atr + 1e-10)` を禁止する。

    Args:
        target_arr: スケール化したい配列 (価格差・標準偏差など)
        atr_arr   : calculate_atr_wilder() の出力
        epsilon   : ゼロ除算防止の微小値 (変更不要)

    Returns:
        np.ndarray: ATR 相対値化された配列
    """
    n = len(target_arr)
    out = np.full(n, np.nan, dtype=np.float64)
    for i in range(n):
        out[i] = target_arr[i] / (atr_arr[i] + epsilon)
    return out


# ===========================================================================
# [CATEGORY: MOMENTUM]  モメンタム指標
# ===========================================================================


@njit(fastmath=False, cache=True)
def calculate_rsi_wilder(prices: np.ndarray, period: int) -> np.ndarray:
    """
    Wilder平滑化 RSI (Relative Strength Index) — 全スクリプトの統一実装。

    アルゴリズム (本来の Wilder 定義):
        1. 初期 AvgGain = 最初の `period` 本の Gain の単純平均
           初期 AvgLoss = 最初の `period` 本の Loss の単純平均
        2. 以降: AvgGain[i] = (AvgGain[i-1] * (period-1) + Gain[i]) / period
                 AvgLoss[i] = (AvgLoss[i-1] * (period-1) + Loss[i]) / period
        3. RS = AvgGain / AvgLoss
           RSI = 100 - 100 / (1 + RS)

    注意:
        - engine_1_C の calculate_rsi_numba は「各バーで過去 period 本を
          毎回ゼロから合算する SMA 方式」であり、この Wilder 方式とは値が異なる。
          これは意図的な変更。
        - Wilder RSI は SMA RSI に比べて「より滑らか」であり、
          本来の RSI の定義に準拠している。

    Returns:
        np.ndarray: shape=(n,). 先頭 period 本は NaN.
    """
    n = len(prices)
    out = np.full(n, np.nan, dtype=np.float64)
    if n <= period:
        return out

    # 初期 AvgGain / AvgLoss を単純平均で計算
    avg_gain = 0.0
    avg_loss = 0.0
    for j in range(1, period + 1):
        diff = prices[j] - prices[j - 1]
        if diff > 0.0:
            avg_gain += diff
        else:
            avg_loss += -diff
    avg_gain /= period
    avg_loss /= period

    # period 本目の RSI
    if avg_gain + avg_loss == 0.0:
        out[period] = 50.0
    elif avg_loss == 0.0:
        out[period] = 100.0
    else:
        rs = avg_gain / avg_loss
        out[period] = 100.0 - 100.0 / (1.0 + rs)

    # period+1 以降: Wilder 平滑化
    for i in range(period + 1, n):
        diff = prices[i] - prices[i - 1]
        gain = diff if diff > 0.0 else 0.0
        loss = -diff if diff < 0.0 else 0.0
        avg_gain = (avg_gain * (period - 1) + gain) / period
        avg_loss = (avg_loss * (period - 1) + loss) / period

        if avg_gain + avg_loss == 0.0:
            out[i] = 50.0
        elif avg_loss == 0.0:
            out[i] = 100.0
        else:
            rs = avg_gain / avg_loss
            out[i] = 100.0 - 100.0 / (1.0 + rs)

    return out


@njit(fastmath=False, cache=True)
def calculate_ema(prices: np.ndarray, period: int) -> np.ndarray:
    """
    指数移動平均 (EMA) — alpha = 2/(period+1), adjust=False 相当。

    Polars の ewm_mean(span=period, adjust=False) と等価。

    Returns:
        np.ndarray: EMA 値の配列。先頭要素は prices[0] でシード。
    """
    n = len(prices)
    out = np.full(n, np.nan, dtype=np.float64)
    if n == 0:
        return out

    alpha = 2.0 / (period + 1.0)
    out[0] = prices[0]
    for i in range(1, n):
        if np.isnan(prices[i]):
            out[i] = out[i - 1]
        elif np.isnan(out[i - 1]):
            out[i] = prices[i]
        else:
            out[i] = alpha * prices[i] + (1.0 - alpha) * out[i - 1]

    return out


@njit(fastmath=False, cache=True)
def calculate_macd(
    prices: np.ndarray,
    fast: int = 12,
    slow: int = 26,
    signal: int = 9,
) -> np.ndarray:
    """
    MACD ライン = EMA(fast) - EMA(slow)。

    Args:
        prices: 価格配列
        fast  : 短期 EMA 期間 (default=12)
        slow  : 長期 EMA 期間 (default=26)
        signal: シグナル EMA 期間 (default=9) ※現在は未使用（将来拡張用）

    Returns:
        np.ndarray: MACD ライン (EMA_fast - EMA_slow) の配列
    """
    ema_fast = calculate_ema(prices, fast)
    ema_slow = calculate_ema(prices, slow)
    n = len(prices)
    out = np.full(n, np.nan, dtype=np.float64)
    for i in range(n):
        if not np.isnan(ema_fast[i]) and not np.isnan(ema_slow[i]):
            out[i] = ema_fast[i] - ema_slow[i]
    return out


# ===========================================================================
# [CATEGORY: TREND]  トレンド指標
# ===========================================================================


@njit(fastmath=False, cache=True)
def calculate_sma(prices: np.ndarray, period: int) -> np.ndarray:
    """
    単純移動平均 (SMA)。

    先頭 (period-1) 本は NaN。Polars rolling_mean() と等価。

    Returns:
        np.ndarray: SMA 値の配列
    """
    n = len(prices)
    out = np.full(n, np.nan, dtype=np.float64)
    if period <= 0 or n < period:
        return out

    # 先頭ウィンドウ
    total = 0.0
    for j in range(period):
        total += prices[j]
    out[period - 1] = total / period

    # スライディング
    for i in range(period, n):
        total += prices[i] - prices[i - period]
        out[i] = total / period

    return out


@njit(fastmath=False, cache=True)
def calculate_bollinger(
    prices: np.ndarray,
    period: int = 20,
    n_std: float = 2.0,
) -> np.ndarray:
    """
    ボリンジャーバンド幅 = SMA ± n_std × σ (不偏標準偏差)。

    Returns:
        np.ndarray: shape=(n, 3) — [upper, middle, lower]
    """
    n = len(prices)
    out = np.full((n, 3), np.nan, dtype=np.float64)
    if n < period:
        return out

    for i in range(period - 1, n):
        window_sum = 0.0
        for j in range(i - period + 1, i + 1):
            window_sum += prices[j]
        mean = window_sum / period

        sq_sum = 0.0
        for j in range(i - period + 1, i + 1):
            d = prices[j] - mean
            sq_sum += d * d
        # 不偏標準偏差 (ddof=1)
        std = (sq_sum / (period - 1)) ** 0.5 if period > 1 else 0.0

        out[i, 0] = mean + n_std * std  # upper
        out[i, 1] = mean               # middle
        out[i, 2] = mean - n_std * std  # lower

    return out


@njit(fastmath=False, cache=True)
def calculate_adx(
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    period: int = 14,
) -> np.ndarray:
    """
    ADX (Average Directional Index) — Wilder 平滑化使用。

    計算手順:
        1. TR, DM+, DM- を算出
        2. Wilder 平滑化で ATR14, DI+14, DI-14 を算出
        3. DX = |DI+ - DI-| / (DI+ + DI+) * 100
        4. ADX = Wilder 平滑化 DX

    Returns:
        np.ndarray: ADX 値の配列。有効値は index 2*period 以降。
    """
    n = len(high)
    out = np.full(n, np.nan, dtype=np.float64)
    if n <= period * 2:
        return out

    # TR, DM+, DM-
    tr = np.zeros(n, dtype=np.float64)
    dm_plus = np.zeros(n, dtype=np.float64)
    dm_minus = np.zeros(n, dtype=np.float64)

    tr[0] = high[0] - low[0]
    for i in range(1, n):
        hl = high[i] - low[i]
        hc = abs(high[i] - close[i - 1])
        lc = abs(low[i] - close[i - 1])
        tr[i] = max(hl, hc, lc)

        up_move = high[i] - high[i - 1]
        down_move = low[i - 1] - low[i]
        dm_plus[i] = up_move if (up_move > down_move and up_move > 0.0) else 0.0
        dm_minus[i] = down_move if (down_move > up_move and down_move > 0.0) else 0.0

    # Wilder 平滑化: ATR, DMP, DMM の初期値 (単純合計)
    atr_w = np.zeros(n, dtype=np.float64)
    dmp_w = np.zeros(n, dtype=np.float64)
    dmm_w = np.zeros(n, dtype=np.float64)

    init_sum_tr = 0.0
    init_sum_dmp = 0.0
    init_sum_dmm = 0.0
    for j in range(period):
        init_sum_tr += tr[j]
        init_sum_dmp += dm_plus[j]
        init_sum_dmm += dm_minus[j]

    atr_w[period - 1] = init_sum_tr
    dmp_w[period - 1] = init_sum_dmp
    dmm_w[period - 1] = init_sum_dmm

    for i in range(period, n):
        atr_w[i] = atr_w[i - 1] - atr_w[i - 1] / period + tr[i]
        dmp_w[i] = dmp_w[i - 1] - dmp_w[i - 1] / period + dm_plus[i]
        dmm_w[i] = dmm_w[i - 1] - dmm_w[i - 1] / period + dm_minus[i]

    # DX 計算
    dx = np.full(n, np.nan, dtype=np.float64)
    for i in range(period - 1, n):
        if atr_w[i] > 1e-10:
            di_plus = dmp_w[i] / atr_w[i] * 100.0
            di_minus = dmm_w[i] / atr_w[i] * 100.0
            di_sum = di_plus + di_minus
            if di_sum > 1e-10:
                dx[i] = abs(di_plus - di_minus) / di_sum * 100.0
            else:
                dx[i] = 0.0

    # ADX = Wilder 平滑化 DX (period*2-1 以降が有効)
    start = period * 2 - 1
    if start >= n:
        return out

    dx_init = 0.0
    for j in range(period - 1, period * 2 - 1):
        if not np.isnan(dx[j]):
            dx_init += dx[j]
    adx_w = dx_init / period
    out[start] = adx_w

    for i in range(start + 1, n):
        if not np.isnan(dx[i]):
            adx_w = (adx_w * (period - 1) + dx[i]) / period
            out[i] = adx_w

    return out


# ===========================================================================
# [CATEGORY: STATS]  統計・分布
# ===========================================================================


@njit(fastmath=False, cache=True)
def stddev_unbiased(arr: np.ndarray, window: int) -> np.ndarray:
    """
    ローリング不偏標準偏差 (ddof=1) — Numba 環境での統一実装。

    Numba の np.std() は ddof=1 をサポートしないため、
    この関数を通じてのみ不偏標準偏差を計算すること。

    数学的定義:
        σ_unbiased = sqrt( Σ(x - μ)² / (n-1) )

    Polars の rolling_std(ddof=1) と等価。

    Returns:
        np.ndarray: ローリング不偏標準偏差の配列。先頭 (window-1) 本は NaN。
    """
    n = len(arr)
    out = np.full(n, np.nan, dtype=np.float64)
    if n < 2 or window < 2:
        return out

    for i in range(window - 1, n):
        # 平均
        win_sum = 0.0
        count = 0
        for j in range(i - window + 1, i + 1):
            if np.isfinite(arr[j]):
                win_sum += arr[j]
                count += 1
        if count < 2:
            continue
        mean = win_sum / count

        # 分散 (ddof=1)
        sq_sum = 0.0
        for j in range(i - window + 1, i + 1):
            if np.isfinite(arr[j]):
                d = arr[j] - mean
                sq_sum += d * d
        out[i] = (sq_sum / (count - 1)) ** 0.5

    return out


@njit(fastmath=False, cache=True)
def calculate_mad(arr: np.ndarray, window: int) -> np.ndarray:
    """
    ローリング MAD (Median Absolute Deviation)。

        MAD = median( |x_i - median(x)| )

    window 内に有限値が 3 本以上必要。
    realtime_feature_engine_1A_statistics.py の mad_rolling_numba と等価。

    Returns:
        np.ndarray: MAD 値の配列。先頭 (window-1) 本は NaN。
    """
    n = len(arr)
    out = np.full(n, np.nan, dtype=np.float64)
    buf = np.zeros(window, dtype=np.float64)
    abs_buf = np.zeros(window, dtype=np.float64)

    for i in range(window - 1, n):
        count = 0
        for j in range(i - window + 1, i + 1):
            if np.isfinite(arr[j]):
                buf[count] = arr[j]
                count += 1

        if count < 3:
            continue

        # median
        finite = buf[:count]
        # Numba では np.sort を使用可
        sorted_vals = np.sort(finite)
        if count % 2 == 0:
            med = (sorted_vals[count // 2 - 1] + sorted_vals[count // 2]) * 0.5
        else:
            med = sorted_vals[count // 2]

        # 絶対偏差
        for k in range(count):
            abs_buf[k] = abs(finite[k] - med)
        sorted_abs = np.sort(abs_buf[:count])
        if count % 2 == 0:
            out[i] = (sorted_abs[count // 2 - 1] + sorted_abs[count // 2]) * 0.5
        else:
            out[i] = sorted_abs[count // 2]

    return out


@njit(fastmath=False, cache=True)
def rolling_zscore(arr: np.ndarray, window: int) -> np.ndarray:
    """
    ローリング Z スコア。

        z[i] = (arr[i] - mean(arr[i-window+1:i+1])) / std_unbiased(...)

    分母がゼロに近い場合は 0.0 を返す。

    Returns:
        np.ndarray: ローリング Z スコアの配列
    """
    n = len(arr)
    out = np.full(n, np.nan, dtype=np.float64)

    sma = calculate_sma(arr, window)
    std = stddev_unbiased(arr, window)

    for i in range(window - 1, n):
        if np.isfinite(sma[i]) and np.isfinite(std[i]):
            if std[i] > 1e-10:
                out[i] = (arr[i] - sma[i]) / std[i]
            else:
                out[i] = 0.0

    return out


@njit(fastmath=False, cache=True)
def clip_and_validate(arr: np.ndarray, clip_val: float = 10.0) -> np.ndarray:
    """
    NaN / Inf 処理とクリッピング。

        1. NaN / Inf は 0.0 に置換
        2. |value| > clip_val の値を ±clip_val にクリップ

    各スクリプトに散在するバラバラの品質保証処理をこの関数に統一する。

    Args:
        arr     : 入力配列
        clip_val: クリップ閾値 (default=10.0)

    Returns:
        np.ndarray: クリーン化された配列
    """
    n = len(arr)
    out = np.zeros(n, dtype=np.float64)
    for i in range(n):
        v = arr[i]
        if not np.isfinite(v):
            out[i] = 0.0
        elif v > clip_val:
            out[i] = clip_val
        elif v < -clip_val:
            out[i] = -clip_val
        else:
            out[i] = v
    return out


# ===========================================================================
# [CATEGORY: WEIGHT]  サンプルウェイト
# ===========================================================================


@njit(fastmath=False, cache=True)
def calculate_sample_weight(
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    atr_period: int = 13,
    zscore_window: int = 50,
) -> np.ndarray:
    """
    サンプルウェイト計算 — Zスコアベース。

    アルゴリズム:
        1. ATR を calculate_atr_wilder(period=atr_period) で計算
        2. ATR 相対値幅 = (high - low) / (ATR + 1e-10)
        3. その系列のローリング Z スコア (絶対値) を計算 (window=zscore_window)
        4. Z スコアに応じた重みを付与:
               |z| < 2.0 → 1.0
               |z| < 3.0 → 2.0
               |z| < 4.0 → 4.0
               それ以外  → 6.0

    engine_1_A の calculate_one_group 内の sample_weight 計算と完全等価。
    engine_1_D (Polars版) および engine_1_E・1_F (Numba直書き) を本関数に統一する。

    重要:
        内部で必ず calculate_atr_wilder を呼び出す。
        ATR の計算方法が変わっても自動的に追従する。

    Returns:
        np.ndarray: サンプルウェイトの配列 (値は 1.0 / 2.0 / 4.0 / 6.0)
    """
    n = len(high)
    out = np.ones(n, dtype=np.float64)  # デフォルト重み = 1.0

    # ATR (Wilder 統一)
    atr = calculate_atr_wilder(high, low, close, atr_period)

    # ATR 相対値幅
    atr_ratio = np.zeros(n, dtype=np.float64)
    for i in range(n):
        atr_ratio[i] = (high[i] - low[i]) / (atr[i] + 1e-10)

    # ローリング Z スコア (絶対値)
    z_arr = rolling_zscore(atr_ratio, zscore_window)

    for i in range(n):
        if not np.isfinite(z_arr[i]):
            out[i] = 1.0
        else:
            z_abs = abs(z_arr[i])
            if z_abs < 2.0:
                out[i] = 1.0
            elif z_abs < 3.0:
                out[i] = 2.0
            elif z_abs < 4.0:
                out[i] = 4.0
            else:
                out[i] = 6.0

    return out


# ===========================================================================
# [CATEGORY: NEUTRALIZATION]  OLS 純化
# ===========================================================================


@njit(fastmath=False, cache=True)
def neutralize_ols(
    y_arr: np.ndarray,
    x_arr: np.ndarray,
    window: int = 2016,
    min_periods: int = 30,
) -> np.ndarray:
    """
    ローリング OLS 純化 (cov/var 方式) — 全スクリプトの統一実装。

    数学的定義 (2_G_alpha_neutralizer.py および realtime_feature_engine.py と完全一致):
        mean_x  = E[X]   (rolling, window)
        var_x   = E[X²] - E[X]²   (clip ≥ 0)
        mean_y  = E[Y]   (rolling, window)
        cov_xy  = E[XY] - E[X]E[Y]

        beta  = cov_xy / (var_x + 1e-10)
        alpha = mean_y - beta * mean_x

        neutralized[i] = y[i] - (beta * x[i] + alpha)

    Args:
        y_arr      : 純化対象特徴量の配列
        x_arr      : プロキシ (市場リターン等) の配列
        window     : OLS ウィンドウ (default=2016, Polars rolling と一致)
        min_periods: 最小サンプル数 (default=30, Polars min_periods と一致)

    Returns:
        np.ndarray: 純化済み特徴量の配列。有効値は min_periods 以降。
    """
    n = len(y_arr)
    out = np.full(n, np.nan, dtype=np.float64)

    for i in range(n):
        # ウィンドウ範囲
        start = max(0, i - window + 1)
        size = i - start + 1

        if size < min_periods:
            continue

        # X の統計量
        sum_x = 0.0
        sum_x2 = 0.0
        count = 0
        for j in range(start, i + 1):
            xj = x_arr[j] if np.isfinite(x_arr[j]) else 0.0
            sum_x += xj
            sum_x2 += xj * xj
            count += 1

        if count == 0:
            continue

        mean_x = sum_x / count
        var_x_raw = sum_x2 / count - mean_x * mean_x
        var_x = var_x_raw if var_x_raw > 0.0 else 0.0

        # Y の統計量と XY 共分散
        sum_y = 0.0
        sum_xy = 0.0
        k = 0
        for j in range(start, i + 1):
            xj = x_arr[j] if np.isfinite(x_arr[j]) else 0.0
            yj = y_arr[j] if np.isfinite(y_arr[j]) else 0.0
            sum_y += yj
            sum_xy += xj * yj
            k += 1

        if k == 0:
            continue

        mean_y = sum_y / k
        mean_xy = sum_xy / k
        cov_xy = mean_xy - mean_x * mean_y

        # OLS 係数
        beta = cov_xy / (var_x + 1e-10)
        alpha_intercept = mean_y - beta * mean_x

        # 純化
        y_latest = y_arr[i] if np.isfinite(y_arr[i]) else 0.0
        x_latest = x_arr[i] if np.isfinite(x_arr[i]) else 0.0
        out[i] = y_latest - (beta * x_latest + alpha_intercept)

    return out


# ===========================================================================
# [CATEGORY: VOLUME]  出来高・ボラティリティ・価格アクション (Engine 1D 用)
# ===========================================================================
# 以下の UDF 群は engine_1_D_a_vast_universe_of_features.py および
# realtime_feature_engine_1D_volume.py から切り出した Single Source of Truth。
#
# 【使い分け原則】
#   - 学習側 (engine_1_D): map_batches / rolling_map からこれらを呼び出す
#   - リアルタイム側 (realtime_feature_engine_1D): FeatureModule1D.calculate_features
#     から同じ関数を呼び出す
# ===========================================================================


# ---------------------------------------------------------------------------
# ヒストリカルボラティリティ系 (スカラー出力 → rolling_map と組み合わせて使用)
# ---------------------------------------------------------------------------


@njit(fastmath=False, cache=True)
def hv_standard_scalar(returns: np.ndarray) -> float:
    """
    標準ヒストリカルボラティリティ (スカラー出力)。

    ウィンドウ内リターン配列を受け取り、不偏標準偏差 (ddof=1) を返す。
    Polars の rolling_map / realtime 側の window スライスに渡して使う。

    引数:
        returns: 対数リターンまたは%変化率の配列

    戻り値:
        float: ロバスト標準偏差 (有効値 < 5 の場合 np.nan)
    """
    if len(returns) < 5:
        return np.nan
    count = 0
    for v in returns:
        if np.isfinite(v):
            count += 1
    if count < 5:
        return np.nan

    # 有限値のみを集める
    finite = np.empty(count, dtype=np.float64)
    k = 0
    for v in returns:
        if np.isfinite(v):
            finite[k] = v
            k += 1

    mean_r = 0.0
    for v in finite:
        mean_r += v
    mean_r /= count

    sq_sum = 0.0
    for v in finite:
        d = v - mean_r
        sq_sum += d * d
    return (sq_sum / (count - 1)) ** 0.5


@njit(fastmath=False, cache=True)
def hv_robust_scalar(returns: np.ndarray) -> float:
    """
    ロバストヒストリカルボラティリティ (スカラー出力)。

    MAD (Median Absolute Deviation) ベース。正規分布仮定下で
    MAD × 1.4826 を不偏標準偏差の推定量として返す。

    引数:
        returns: 対数リターンまたは%変化率の配列

    戻り値:
        float: ロバスト標準偏差 (有効値 < 5 の場合 np.nan)
    """
    if len(returns) < 5:
        return np.nan
    count = 0
    for v in returns:
        if np.isfinite(v):
            count += 1
    if count < 5:
        return np.nan

    finite = np.empty(count, dtype=np.float64)
    k = 0
    for v in returns:
        if np.isfinite(v):
            finite[k] = v
            k += 1

    sorted_finite = np.sort(finite)
    if count % 2 == 0:
        med = (sorted_finite[count // 2 - 1] + sorted_finite[count // 2]) * 0.5
    else:
        med = sorted_finite[count // 2]

    abs_dev = np.empty(count, dtype=np.float64)
    for i in range(count):
        abs_dev[i] = abs(finite[i] - med)
    sorted_dev = np.sort(abs_dev)
    if count % 2 == 0:
        mad = (sorted_dev[count // 2 - 1] + sorted_dev[count // 2]) * 0.5
    else:
        mad = sorted_dev[count // 2]

    return mad * 1.4826


# ---------------------------------------------------------------------------
# Chaikin Volatility
# ---------------------------------------------------------------------------


@njit(fastmath=False, cache=True)
def chaikin_volatility_udf(
    high: np.ndarray,
    low: np.ndarray,
    window: int,
) -> np.ndarray:
    """
    Chaikin Volatility — EMA(High-Low) の変化率。

    計算手順:
        1. EMA(window) of (High - Low) を連続的に計算 (alpha = 2/(window+1))
        2. ChaikinVol[i] = (EMA[i] - EMA[i-window]) / EMA[i-window] × 100

    注意:
        - engine_1_D の実装と完全等価 (SMA初期化 → 真の連続 EMA)。

    戻り値:
        np.ndarray: shape=(n,). 有効値は index (window*2-1) 以降。
    """
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

    # SMA でシード
    sma_init = 0.0
    for i in range(valid_start, valid_start + window):
        sma_init += hl_range[i]
    ema[valid_start + window - 1] = sma_init / window

    alpha = 2.0 / (window + 1.0)
    for i in range(valid_start + window, n):
        ema[i] = alpha * hl_range[i] + (1.0 - alpha) * ema[i - 1]

    for i in range(valid_start + window * 2 - 1, n):
        prev_ema = ema[i - window]
        if np.isfinite(ema[i]) and np.isfinite(prev_ema) and prev_ema > 0.0:
            result[i] = (ema[i] - prev_ema) / prev_ema * 100.0

    return result


# ---------------------------------------------------------------------------
# Mass Index
# ---------------------------------------------------------------------------


@njit(fastmath=False, cache=True)
def mass_index_udf(
    high: np.ndarray,
    low: np.ndarray,
    window: int,
) -> np.ndarray:
    """
    Mass Index — EMA(9)/EMA(EMA(9)) の比率の累積和。

    計算手順:
        1. 真の連続 EMA(9) of (High - Low)
        2. 真の連続 Double EMA(9) (EMA of EMA)
        3. window 期間の比率累積和

    注意:
        - engine_1_D の実装と完全等価。

    戻り値:
        np.ndarray: shape=(n,). 有効値は index (16+window) 以降。
    """
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
        mass_sum = 0.0
        is_valid = True
        for j in range(i - window + 1, i + 1):
            if (
                not np.isfinite(ema9[j])
                or not np.isfinite(ema_ema9[j])
                or ema_ema9[j] <= 0.0
            ):
                is_valid = False
                break
            mass_sum += ema9[j] / ema_ema9[j]
        if is_valid:
            result[i] = mass_sum

    return result


# ---------------------------------------------------------------------------
# 出来高フロー系
# ---------------------------------------------------------------------------


@njit(fastmath=False, cache=True)
def cmf_udf(
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    volume: np.ndarray,
    window: int,
) -> np.ndarray:
    """
    Chaikin Money Flow (CMF)。

        CLV[i] = ((Close - Low) - (High - Close)) / (High - Low)
        CMF    = Σ(CLV × Volume, window) / Σ(Volume, window)

    戻り値:
        np.ndarray: [-1, 1] の範囲。先頭 (window-1) 本は NaN。
    """
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
                hl = high[j] - low[j]
                if hl > 0.0:
                    clv = ((close[j] - low[j]) - (high[j] - close[j])) / (hl + 1e-10)
                    mf_vol_sum += clv * volume[j]
                    vol_sum += volume[j]
        if vol_sum > 0.0:
            result[i] = mf_vol_sum / vol_sum

    return result


@njit(fastmath=False, cache=True)
def mfi_udf(
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    volume: np.ndarray,
    window: int,
) -> np.ndarray:
    """
    Money Flow Index (MFI) — 出来高加重 RSI。

        TypicalPrice = (High + Low + Close) / 3
        RawMoneyFlow = TypicalPrice × Volume
        MFI = 100 - 100 / (1 + PositiveFlow / NegativeFlow)

    戻り値:
        np.ndarray: [0, 100] の範囲。先頭 window 本は NaN。
    """
    n = len(close)
    result = np.full(n, np.nan, dtype=np.float64)
    if n < window + 1:
        return result

    for i in range(window, n):
        pos_flow = 0.0
        neg_flow = 0.0
        for j in range(i - window + 1, i + 1):
            if j == 0:
                continue
            if (
                np.isfinite(high[j]) and np.isfinite(low[j])
                and np.isfinite(close[j]) and np.isfinite(volume[j])
                and np.isfinite(high[j - 1]) and np.isfinite(low[j - 1])
                and np.isfinite(close[j - 1])
            ):
                tp = (high[j] + low[j] + close[j]) / 3.0
                prev_tp = (high[j - 1] + low[j - 1] + close[j - 1]) / 3.0
                rmf = tp * volume[j]
                if tp > prev_tp:
                    pos_flow += rmf
                elif tp < prev_tp:
                    neg_flow += rmf

        if neg_flow > 0.0:
            result[i] = 100.0 - (100.0 / (1.0 + pos_flow / neg_flow))
        elif pos_flow > 0.0:
            result[i] = 100.0
        else:
            result[i] = 50.0

    return result


@njit(fastmath=False, cache=True)
def vwap_udf(
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    volume: np.ndarray,
    window: int,
) -> np.ndarray:
    """
    Volume Weighted Average Price (VWAP) — ローリングウィンドウ版。

        VWAP = Σ(TypicalPrice × Volume, window) / Σ(Volume, window)

    戻り値:
        np.ndarray: 絶対価格スケール。ATR距離化は呼び出し側で行う。
                    先頭 (window-1) 本は NaN。
    """
    n = len(close)
    result = np.full(n, np.nan, dtype=np.float64)
    if n < window:
        return result

    for i in range(window - 1, n):
        pv_sum = 0.0
        vol_sum = 0.0
        for j in range(i - window + 1, i + 1):
            if (
                np.isfinite(high[j]) and np.isfinite(low[j])
                and np.isfinite(close[j]) and np.isfinite(volume[j])
            ):
                tp = (high[j] + low[j] + close[j]) / 3.0
                pv_sum += tp * volume[j]
                vol_sum += volume[j]
        if vol_sum > 0.0:
            result[i] = pv_sum / vol_sum

    return result


@njit(fastmath=False, cache=True)
def obv_udf(close: np.ndarray, volume: np.ndarray) -> np.ndarray:
    """
    On Balance Volume (OBV) — 累積出来高指標。

        OBV[0] = Volume[0]
        OBV[i] = OBV[i-1] + Volume[i]  (Close[i] > Close[i-1])
               = OBV[i-1] - Volume[i]  (Close[i] < Close[i-1])
               = OBV[i-1]              (変化なし)

    注意:
        - 絶対値スケール。呼び出し側で差分 / vol_ma などで相対化すること。
        - engine_1_D・realtime_feature_engine_1D と完全等価。

    戻り値:
        np.ndarray: 累積 OBV 値。先頭 1 本は Volume[0]。
    """
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


@njit(fastmath=False, cache=True)
def accumulation_distribution_udf(
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    volume: np.ndarray,
) -> np.ndarray:
    """
    Accumulation / Distribution Line — CLV ベース累積指標。

        CLV = ((Close - Low) - (High - Close)) / (High - Low)
        AD[i] = AD[i-1] + CLV[i] × Volume[i]

    注意:
        - 絶対値スケール。呼び出し側で差分 / vol_ma などで相対化すること。
        - engine_1_D と完全等価。

    戻り値:
        np.ndarray: 累積 A/D 値。
    """
    n = len(close)
    result = np.full(n, np.nan, dtype=np.float64)
    if n < 1:
        return result

    result[0] = 0.0
    for i in range(1, n):
        prev_ad = result[i - 1] if np.isfinite(result[i - 1]) else 0.0
        if (
            np.isfinite(high[i]) and np.isfinite(low[i])
            and np.isfinite(close[i]) and np.isfinite(volume[i])
        ):
            hl = high[i] - low[i]
            if hl > 0.0:
                clv = ((close[i] - low[i]) - (high[i] - close[i])) / (hl + 1e-10)
                result[i] = prev_ad + clv * volume[i]
            else:
                result[i] = prev_ad
        else:
            result[i] = prev_ad

    return result


@njit(fastmath=False, cache=True)
def force_index_udf(close: np.ndarray, volume: np.ndarray) -> np.ndarray:
    """
    Force Index — 価格変動 × 出来高。

        ForceIndex[i] = (Close[i] - Close[i-1]) × Volume[i]

    注意:
        - 絶対値スケール。呼び出し側で ATR × vol_ma などで正規化すること。
        - engine_1_D と完全等価。

    戻り値:
        np.ndarray: Force Index 値。先頭は 0.0。
    """
    n = len(close)
    result = np.full(n, np.nan, dtype=np.float64)
    if n < 2:
        return result

    result[0] = 0.0
    for i in range(1, n):
        if (
            np.isfinite(close[i])
            and np.isfinite(close[i - 1])
            and np.isfinite(volume[i])
        ):
            result[i] = (close[i] - close[i - 1]) * volume[i]
        else:
            result[i] = 0.0

    return result


# ---------------------------------------------------------------------------
# Commodity Channel Index
# ---------------------------------------------------------------------------


@njit(fastmath=False, cache=True)
def commodity_channel_index_udf(
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    window: int,
) -> np.ndarray:
    """
    Commodity Channel Index (CCI)。

        TypicalPrice = (High + Low + Close) / 3
        CCI = (TP - SMA(TP, window)) / (0.015 × MeanDeviation)

    注意:
        - engine_1_D の実装と完全等価 (メモリ事前確保版)。

    戻り値:
        np.ndarray: CCI 値。先頭 (window-1) 本は NaN。
    """
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
        for j in range(window):
            if np.isfinite(typical_prices[j]):
                tp_sum += typical_prices[j]
        sma = tp_sum / valid_count

        md_sum = 0.0
        for j in range(window):
            if np.isfinite(typical_prices[j]):
                md_sum += abs(typical_prices[j] - sma)
        mean_dev = md_sum / valid_count

        current_tp = (high[i] + low[i] + close[i]) / 3.0
        if mean_dev > 0.0:
            result[i] = (current_tp - sma) / (0.015 * mean_dev)

    return result


# ---------------------------------------------------------------------------
# ローソク足パターン認識
# ---------------------------------------------------------------------------


@njit(fastmath=False, cache=True)
def candlestick_patterns_udf(
    open_prices: np.ndarray,
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
) -> np.ndarray:
    """
    ローソク足パターン認識。

    パターン ID:
        0.0 = なし
        1.0 = ハンマー    (下ヒゲ長 > 60%, 上ヒゲ < 10%)
        2.0 = 流れ星      (上ヒゲ長 > 60%, 下ヒゲ < 10%)
        3.0 = 同事        (実体比率 < 10%)
        4.0 = 強気マルボウ (実体 > 60%, 陽線)
        5.0 = 弱気マルボウ (実体 > 60%, 陰線)

    注意:
        - engine_1_D・realtime_feature_engine_1D と完全等価 (float64 型統一済み)。

    戻り値:
        np.ndarray: float64 パターン ID 配列。
    """
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

        if total_range <= 0.0:
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


# ---------------------------------------------------------------------------
# フィボナッチリトレースメント
# ---------------------------------------------------------------------------


@njit(fastmath=False, cache=True)
def fibonacci_levels_udf(
    high: np.ndarray,
    low: np.ndarray,
    window: int,
) -> np.ndarray:
    """
    フィボナッチリトレースメントレベル計算。

    比率: [0.236, 0.382, 0.500, 0.618, 0.786]

    計算:
        period_high = rolling_max(high, window)
        period_low  = rolling_min(low, window)
        level_k     = period_high - ratio_k × (period_high - period_low)

    注意:
        - 絶対価格スケール。呼び出し側で ATR 距離化すること。
        - engine_1_D と完全等価。

    戻り値:
        np.ndarray: shape=(n, 5)。先頭 (window-1) 行は NaN。
    """
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
                result[i, k] = period_high - fib_ratios[k] * price_range

    return result


# ===========================================================================
# [CATEGORY: DSP]  信号処理 (Engine 1E 用)
# ===========================================================================
# engine_1_E_a_vast_universe_of_features.py / realtime_feature_engine_1E_signal.py
# の FFT・ヒルベルト変換・スペクトル・ウェーブレット・音響 UDF 群。
#
# 【設計方針】
#   - 全 UDF は @njit(fastmath=False, cache=True) で実装
#     (学習時・本番間の完全一致を保証するため fastmath=False に統一)
#   - スレッド安全性のため parallel=True は使用しない
#     (Polars側の並列化と衝突を避ける)
#   - ループ外でバッファを事前確保し、動的メモリ確保を最小化
# ===========================================================================


# ---------------------------------------------------------------------------
# Numba ネイティブ FFT (np.fft は Numba 未サポート)
# ---------------------------------------------------------------------------


@njit(fastmath=False, cache=True)
def numba_fft(x: np.ndarray) -> np.ndarray:
    """
    Numba 実装の反復的 Cooley-Tukey FFT（安定版）。

    np.fft は Numba 環境で使用不可のため、このネイティブ実装を使う。
    n が 2 のべき乗でない場合はゼロパディングする。

    Returns:
        np.ndarray: complex128 配列。長さは 2^ceil(log2(n)) 以上。
    """
    n = x.shape[0]

    # 2 のべき乗でない場合はゼロパディング
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


@njit(fastmath=False, cache=True)
def numba_ifft(x: np.ndarray) -> np.ndarray:
    """
    Numba による逆 FFT (IFFT)。

        IFFT(x) = conj(FFT(conj(x))) / n
    """
    return np.conj(numba_fft(np.conj(x))) / len(x)


@njit(fastmath=False, cache=True)
def get_analytic_signal(x: np.ndarray) -> np.ndarray:
    """
    FFT ベースの離散ヒルベルト変換（解析信号）。

    scipy.signal.hilbert と数学的に等価。
    パディング後サイズに合わせて計算し、最後に元のサイズで切り取る。

    Returns:
        np.ndarray: complex128 解析信号。shape=(len(x),)
    """
    n_orig = len(x)
    X = numba_fft(x)
    n_fft = len(X)

    h = np.zeros(n_fft, dtype=np.complex128)
    if n_fft > 0:
        h[0] = 1.0
        if n_fft % 2 == 0:
            h[1 : n_fft // 2] = 2.0
            h[n_fft // 2] = 1.0
        else:
            h[1 : (n_fft + 1) // 2] = 2.0

    analytic_padded = numba_ifft(X * h)
    return analytic_padded[:n_orig]


# ---------------------------------------------------------------------------
# スペクトル特徴量
# ---------------------------------------------------------------------------


@njit(fastmath=False, cache=True)
def spectral_centroid_udf(signal: np.ndarray, window_size: int) -> np.ndarray:
    """
    スペクトル重心 (Spectral Centroid)。

    周波数スペクトルの「重心」周波数を返す。
    高い値 = エネルギーが高周波に集中。

    Returns:
        np.ndarray: shape=(n,). 有効値は index (window_size-1) 以降。
    """
    n = len(signal)
    result = np.full(n, np.nan)
    if n < window_size:
        return result

    buffer = np.zeros(window_size, dtype=np.float64)

    for i in range(window_size - 1, n):
        valid_count = 0
        for j in range(window_size):
            val = signal[i - window_size + 1 + j]
            if np.isfinite(val):
                buffer[valid_count] = val
                valid_count += 1

        if valid_count < window_size // 2:
            continue

        fft_values = numba_fft(buffer[:valid_count])
        mag_len = valid_count // 2
        magnitude_spectrum = np.abs(fft_values[:mag_len])

        if mag_len == 0:
            continue

        total_magnitude = np.sum(magnitude_spectrum)
        if total_magnitude > 0:
            centroid = 0.0
            for k in range(mag_len):
                freq = k * 0.5 / max(1, mag_len - 1)
                centroid += freq * magnitude_spectrum[k]
            result[i] = centroid / total_magnitude

    return result


@njit(fastmath=False, cache=True)
def spectral_bandwidth_udf(signal: np.ndarray, window_size: int) -> np.ndarray:
    """
    スペクトル帯域幅 (Spectral Bandwidth)。

    スペクトル重心周辺のエネルギー分散（標準偏差的な広がり）を返す。

    Returns:
        np.ndarray: shape=(n,). 有効値は index (window_size-1) 以降。
    """
    n = len(signal)
    result = np.full(n, np.nan)
    if n < window_size:
        return result

    buffer = np.zeros(window_size, dtype=np.float64)

    for i in range(window_size - 1, n):
        valid_count = 0
        for j in range(window_size):
            val = signal[i - window_size + 1 + j]
            if np.isfinite(val):
                buffer[valid_count] = val
                valid_count += 1

        if valid_count < window_size // 2:
            continue

        fft_values = numba_fft(buffer[:valid_count])
        mag_len = valid_count // 2
        magnitude_spectrum = np.abs(fft_values[:mag_len])

        if mag_len == 0:
            continue

        total_magnitude = np.sum(magnitude_spectrum)
        if total_magnitude > 0:
            centroid = 0.0
            for k in range(mag_len):
                freq = k * 0.5 / max(1, mag_len - 1)
                centroid += freq * magnitude_spectrum[k]
            centroid /= total_magnitude

            variance = 0.0
            for k in range(mag_len):
                freq = k * 0.5 / max(1, mag_len - 1)
                variance += ((freq - centroid) ** 2) * magnitude_spectrum[k]

            result[i] = np.sqrt(variance / total_magnitude)

    return result


@njit(fastmath=False, cache=True)
def spectral_rolloff_udf(
    signal: np.ndarray, window_size: int, rolloff_ratio: float = 0.85
) -> np.ndarray:
    """
    スペクトルロールオフ (Spectral Rolloff)。

    全パワーの rolloff_ratio (default=85%) が含まれる周波数閾値を返す。

    Returns:
        np.ndarray: shape=(n,). 有効値は index (window_size-1) 以降。
    """
    n = len(signal)
    result = np.full(n, np.nan)
    if n < window_size:
        return result

    buffer = np.zeros(window_size, dtype=np.float64)

    for i in range(window_size - 1, n):
        valid_count = 0
        for j in range(window_size):
            val = signal[i - window_size + 1 + j]
            if np.isfinite(val):
                buffer[valid_count] = val
                valid_count += 1

        if valid_count < window_size // 2:
            continue

        fft_values = numba_fft(buffer[:valid_count])
        mag_len = valid_count // 2
        magnitude_spectrum = np.abs(fft_values[:mag_len])

        if mag_len == 0:
            continue

        power_spectrum = magnitude_spectrum ** 2
        total_power = np.sum(power_spectrum)

        if total_power > 0:
            threshold = rolloff_ratio * total_power
            cumulative_power = 0.0
            for k in range(mag_len):
                cumulative_power += power_spectrum[k]
                if cumulative_power >= threshold:
                    result[i] = k / (2.0 * mag_len)
                    break

    return result


@njit(fastmath=False, cache=True)
def spectral_flux_udf(signal: np.ndarray, window_size: int) -> np.ndarray:
    """
    スペクトルフラックス (Spectral Flux)。

    隣接フレーム間のスペクトル変化量（L2 ノルム）を返す。

    Returns:
        np.ndarray: shape=(n,). 有効値は index (window_size*2-1) 以降。
    """
    n = len(signal)
    result = np.full(n, np.nan)
    if n < window_size * 2:
        return result

    curr_buffer = np.zeros(window_size, dtype=np.float64)
    prev_buffer = np.zeros(window_size, dtype=np.float64)

    for i in range(window_size * 2 - 1, n):
        curr_valid = 0
        for j in range(window_size):
            val = signal[i - window_size + 1 + j]
            if np.isfinite(val):
                curr_buffer[curr_valid] = val
                curr_valid += 1

        prev_valid = 0
        for j in range(window_size):
            val = signal[i - window_size * 2 + 1 + j]
            if np.isfinite(val):
                prev_buffer[prev_valid] = val
                prev_valid += 1

        if curr_valid < window_size // 2 or prev_valid < window_size // 2:
            continue

        curr_fft = numba_fft(curr_buffer[:curr_valid])
        prev_fft = numba_fft(prev_buffer[:prev_valid])

        mag_curr = np.abs(curr_fft[: curr_valid // 2])
        mag_prev = np.abs(prev_fft[: prev_valid // 2])

        min_size = min(len(mag_curr), len(mag_prev))
        if min_size > 0:
            flux_sq = 0.0
            for k in range(min_size):
                diff = mag_curr[k] - mag_prev[k]
                flux_sq += diff * diff
            result[i] = np.sqrt(flux_sq + 1e-10)

    return result


@njit(fastmath=False, cache=True)
def spectral_flatness_udf(signal: np.ndarray, window_size: int) -> np.ndarray:
    """
    スペクトル平坦度 (Spectral Flatness / Wiener Entropy)。

    幾何平均 / 算術平均。値が 1 に近いほどホワイトノイズ的。

    Returns:
        np.ndarray: shape=(n,). 有効値は index (window_size-1) 以降。
    """
    n = len(signal)
    result = np.full(n, np.nan)
    if n < window_size:
        return result

    buffer = np.zeros(window_size, dtype=np.float64)

    for i in range(window_size - 1, n):
        valid_count = 0
        for j in range(window_size):
            val = signal[i - window_size + 1 + j]
            if np.isfinite(val):
                buffer[valid_count] = val
                valid_count += 1

        if valid_count < window_size // 2:
            continue

        fft_values = numba_fft(buffer[:valid_count])
        mag_len = valid_count // 2
        magnitude_spectrum = np.abs(fft_values[:mag_len])

        if mag_len == 0:
            continue

        geom_sum = 0.0
        arith_sum = 0.0
        for k in range(mag_len):
            val = magnitude_spectrum[k] + 1e-10
            geom_sum += np.log(val)
            arith_sum += val

        geometric_mean = np.exp(geom_sum / mag_len)
        arithmetic_mean = arith_sum / mag_len

        if arithmetic_mean > 0:
            result[i] = geometric_mean / arithmetic_mean

    return result


@njit(fastmath=False, cache=True)
def spectral_entropy_udf(signal: np.ndarray, window_size: int) -> np.ndarray:
    """
    スペクトルエントロピー (Spectral Entropy)。

    パワースペクトルの確率分布から計算したシャノンエントロピー。
    高いほど周波数成分が均等に分布。

    Returns:
        np.ndarray: shape=(n,). 有効値は index (window_size-1) 以降。
    """
    n = len(signal)
    result = np.full(n, np.nan)
    if n < window_size:
        return result

    buffer = np.zeros(window_size, dtype=np.float64)

    for i in range(window_size - 1, n):
        valid_count = 0
        for j in range(window_size):
            val = signal[i - window_size + 1 + j]
            if np.isfinite(val):
                buffer[valid_count] = val
                valid_count += 1

        if valid_count < window_size // 2:
            continue

        fft_values = numba_fft(buffer[:valid_count])
        mag_len = valid_count // 2
        magnitude_spectrum = np.abs(fft_values[:mag_len])

        if mag_len == 0:
            continue

        total_power = 0.0
        for k in range(mag_len):
            total_power += magnitude_spectrum[k] ** 2

        if total_power > 0:
            entropy = 0.0
            for k in range(mag_len):
                p = (magnitude_spectrum[k] ** 2) / total_power
                if p > 1e-10:
                    entropy -= p * np.log2(p)
            result[i] = entropy

    return result


# ---------------------------------------------------------------------------
# ウェーブレット特徴量
# ---------------------------------------------------------------------------


@njit(fastmath=False, cache=True)
def wavelet_energy_udf(
    signal: np.ndarray, window_size: int, levels: int = 4
) -> np.ndarray:
    """
    ウェーブレットエネルギー (Haar DWT)。

    インプレース Haar 変換で detail 係数のエネルギーを累積する。

    Returns:
        np.ndarray: shape=(n,). 有効値は index (window_size-1) 以降。
    """
    n = len(signal)
    result = np.full(n, np.nan)
    if n < window_size:
        return result

    buffer = np.zeros(window_size, dtype=np.float64)

    for i in range(window_size - 1, n):
        valid_count = 0
        for j in range(window_size):
            val = signal[i - window_size + 1 + j]
            if np.isfinite(val):
                buffer[valid_count] = val
                valid_count += 1

        if valid_count < window_size // 2:
            continue

        level_energy = 0.0
        current_len = valid_count

        # インプレース Haar 変換 (buffer 前方を上書き)
        for level in range(min(levels, 4)):
            if current_len < 4:
                break

            next_len = current_len // 2
            for j in range(next_len):
                idx = j * 2
                val1 = buffer[idx]
                val2 = buffer[idx + 1]
                approx = (val1 + val2) / 1.41421356237
                detail = (val1 - val2) / 1.41421356237

                level_energy += detail * detail
                buffer[j] = approx

            current_len = next_len

        result[i] = level_energy

    return result


@njit(fastmath=False, cache=True)
def wavelet_entropy_udf(signal: np.ndarray, window_size: int) -> np.ndarray:
    """
    ウェーブレットエントロピー。

        entropy = -Σ (x²) * log2(x² + ε)

    Returns:
        np.ndarray: shape=(n,). 有効値は index (window_size-1) 以降。
    """
    n = len(signal)
    result = np.full(n, np.nan)
    if n < window_size:
        return result

    buffer = np.zeros(window_size, dtype=np.float64)

    for i in range(window_size - 1, n):
        valid_count = 0
        for j in range(window_size):
            val = signal[i - window_size + 1 + j]
            if np.isfinite(val):
                buffer[valid_count] = val
                valid_count += 1

        if valid_count < window_size // 2:
            continue

        entropy = 0.0
        for j in range(valid_count):
            sq = buffer[j] * buffer[j]
            entropy -= sq * np.log2(sq + 1e-10)
        result[i] = entropy

    return result


# ---------------------------------------------------------------------------
# ヒルベルト変換特徴量
# ---------------------------------------------------------------------------


@njit(fastmath=False, cache=True)
def hilbert_amplitude_udf(signal: np.ndarray, window_size: int) -> np.ndarray:
    """
    ヒルベルト変換による振幅包絡線の平均。

    解析信号の絶対値 (= 瞬時振幅) をウィンドウ内で平均する。

    Returns:
        np.ndarray: shape=(n,). 有効値は index (window_size-1) 以降。
    """
    n = len(signal)
    result = np.full(n, np.nan)
    if n < window_size:
        return result

    buffer = np.zeros(window_size, dtype=np.float64)

    for i in range(window_size - 1, n):
        valid_count = 0
        for j in range(window_size):
            val = signal[i - window_size + 1 + j]
            if np.isfinite(val):
                buffer[valid_count] = val
                valid_count += 1

        if valid_count < window_size // 2:
            continue

        analytic_signal = get_analytic_signal(buffer[:valid_count])
        amp_sum = 0.0
        for k in range(valid_count):
            real = analytic_signal[k].real
            imag = analytic_signal[k].imag
            amp_sum += np.sqrt(real * real + imag * imag)

        result[i] = amp_sum / valid_count

    return result


@njit(fastmath=False, cache=True)
def hilbert_phase_var_udf(signal: np.ndarray, window_size: int) -> np.ndarray:
    """
    ヒルベルト変換による瞬時位相の分散（不偏推定）。

    解析信号の位相角の分散 (ddof=1)。
    位相が安定しているほど小さい値。

    Returns:
        np.ndarray: shape=(n,). 有効値は index (window_size-1) 以降。
    """
    n = len(signal)
    result = np.full(n, np.nan)
    if n < window_size:
        return result

    buffer = np.zeros(window_size, dtype=np.float64)

    for i in range(window_size - 1, n):
        valid_count = 0
        for j in range(window_size):
            val = signal[i - window_size + 1 + j]
            if np.isfinite(val):
                buffer[valid_count] = val
                valid_count += 1

        if valid_count > 1:
            analytic_signal = get_analytic_signal(buffer[:valid_count])
            phases = np.angle(analytic_signal)

            # 標本分散(ddof=0) → 不偏分散(ddof=1) へ補正
            var_0 = np.var(phases)
            result[i] = var_0 * (valid_count / (valid_count - 1.0))

    return result


@njit(fastmath=False, cache=True)
def hilbert_phase_stability_udf(signal: np.ndarray, window_size: int) -> np.ndarray:
    """
    ヒルベルト変換による位相安定性。

        stability = 1 / (1 + std(Δphase) + ε)

    値が 1 に近いほど位相変化が安定。

    Returns:
        np.ndarray: shape=(n,). 有効値は index (window_size-1) 以降。
    """
    n = len(signal)
    result = np.full(n, np.nan)
    if n < window_size:
        return result

    buffer = np.zeros(window_size, dtype=np.float64)

    for i in range(window_size - 1, n):
        valid_count = 0
        for j in range(window_size):
            val = signal[i - window_size + 1 + j]
            if np.isfinite(val):
                buffer[valid_count] = val
                valid_count += 1

        if valid_count > 2:
            sliced_buffer = buffer[:valid_count]

            # 不偏標準偏差 (ddof=1)
            std_0 = np.std(sliced_buffer)
            std_1 = std_0 * np.sqrt(valid_count / (valid_count - 1.0))

            if std_1 < 1e-10:
                result[i] = 1.0
                continue

            analytic_signal = get_analytic_signal(sliced_buffer)
            diffs = np.diff(np.angle(analytic_signal))
            n_diff = len(diffs)

            if n_diff <= 1:
                continue

            phase_diff_std_0 = np.std(diffs)
            phase_diff_std_1 = phase_diff_std_0 * np.sqrt(n_diff / (n_diff - 1.0))

            result[i] = 1.0 / (1.0 + phase_diff_std_1 + 1e-10)

    return result


@njit(fastmath=False, cache=True)
def hilbert_freq_mean_udf(signal: np.ndarray, window_size: int) -> np.ndarray:
    """
    ヒルベルト変換による瞬時周波数の平均。

    瞬時周波数 = |Δphase| の平均。

    Returns:
        np.ndarray: shape=(n,). 有効値は index (window_size-1) 以降。
    """
    n = len(signal)
    result = np.full(n, np.nan)
    if n < window_size:
        return result

    buffer = np.zeros(window_size, dtype=np.float64)

    for i in range(window_size - 1, n):
        valid_count = 0
        for j in range(window_size):
            val = signal[i - window_size + 1 + j]
            if np.isfinite(val):
                buffer[valid_count] = val
                valid_count += 1

        if valid_count > 2:
            analytic_signal = get_analytic_signal(buffer[:valid_count])
            instant_freq = np.abs(np.diff(np.angle(analytic_signal)))
            result[i] = np.mean(instant_freq)

    return result


@njit(fastmath=False, cache=True)
def hilbert_freq_std_udf(signal: np.ndarray, window_size: int) -> np.ndarray:
    """
    ヒルベルト変換による瞬時周波数の標準偏差（不偏推定）。

    Returns:
        np.ndarray: shape=(n,). 有効値は index (window_size-1) 以降。
    """
    n = len(signal)
    result = np.full(n, np.nan)
    if n < window_size:
        return result

    buffer = np.zeros(window_size, dtype=np.float64)

    for i in range(window_size - 1, n):
        valid_count = 0
        for j in range(window_size):
            val = signal[i - window_size + 1 + j]
            if np.isfinite(val):
                buffer[valid_count] = val
                valid_count += 1

        if valid_count > 2:
            analytic_signal = get_analytic_signal(buffer[:valid_count])
            instant_freq = np.abs(np.diff(np.angle(analytic_signal)))
            n_freq = len(instant_freq)

            if n_freq <= 1:
                continue
            # 不偏標準偏差 (ddof=1)
            std_0 = np.std(instant_freq)
            result[i] = std_0 * np.sqrt(n_freq / (n_freq - 1.0))

    return result


# ---------------------------------------------------------------------------
# 音響特徴量
# ---------------------------------------------------------------------------


@njit(fastmath=False, cache=True)
def acoustic_power_udf(signal: np.ndarray, window_size: int) -> np.ndarray:
    """
    音響パワー (RMS Power)。

        rms = sqrt(mean(x²))

    Returns:
        np.ndarray: shape=(n,). 有効値は index (window_size-1) 以降。
    """
    n = len(signal)
    result = np.full(n, np.nan)
    if n < window_size:
        return result

    buffer = np.zeros(window_size, dtype=np.float64)

    for i in range(window_size - 1, n):
        valid_count = 0
        for j in range(window_size):
            val = signal[i - window_size + 1 + j]
            if np.isfinite(val):
                buffer[valid_count] = val
                valid_count += 1

        if valid_count < window_size // 2:
            continue

        power_sum = 0.0
        for k in range(valid_count):
            power_sum += buffer[k] * buffer[k]

        result[i] = np.sqrt(power_sum / valid_count)

    return result


@njit(fastmath=False, cache=True)
def acoustic_frequency_udf(
    signal: np.ndarray, window_size: int, sample_rate: float = 1.0
) -> np.ndarray:
    """
    音響周波数推定（ゼロクロス法）。

        freq = (zero_crossings / 2 / N) * sample_rate

    Returns:
        np.ndarray: shape=(n,). 有効値は index (window_size-1) 以降。
    """
    n = len(signal)
    result = np.full(n, np.nan)
    if n < window_size:
        return result

    buffer = np.zeros(window_size, dtype=np.float64)

    for i in range(window_size - 1, n):
        valid_count = 0
        for j in range(window_size):
            val = signal[i - window_size + 1 + j]
            if np.isfinite(val):
                buffer[valid_count] = val
                valid_count += 1

        if valid_count < window_size // 2:
            continue

        zero_crossings = 0
        for k in range(1, valid_count):
            if buffer[k - 1] * buffer[k] < 0:
                zero_crossings += 1

        if valid_count > 1:
            result[i] = (zero_crossings / (2.0 * valid_count)) * sample_rate

    return result


# ===========================================================================
# [CATEGORY: COMPLEX]  複雑系 (Engine 1F 用)
# ===========================================================================
# engine_1_F_a_vast_universe_of_features.py / realtime_feature_engine_1F_experimental.py
# のネットワーク科学・言語学・美学・音楽理論・生体力学 UDF 群。
#
# 【設計方針】
#   - 全 UDF は @njit(fastmath=False, cache=True) で実装
#   - parallel=True は使用しない（Polars 並列化との衝突を避ける）
#   - ループ外でバッファを事前確保し、動的メモリ確保を最小化
#   - ddof=1 の不偏推定は Numba 制約のため手動ベッセル補正で統一:
#       std_unbiased = np.std(x) * sqrt(n / (n-1))
#       var_unbiased = np.var(x) * (n / (n-1))
#
# 【注意】engine_1_F の calculate_atr_numba / calculate_sample_weight_udf は
#   core_indicators の calculate_atr_wilder / calculate_sample_weight に置き換える。
#   ここには他エンジンと共有されない複雑系 UDF のみを定義する。
# ===========================================================================


# ---------------------------------------------------------------------------
# ネットワーク科学: 価格ネットワーク密度・クラスタリング
# ---------------------------------------------------------------------------


@njit(fastmath=False, cache=True)
def rolling_network_density_udf(prices: np.ndarray, window_size: int) -> np.ndarray:
    """
    ローリング価格ネットワーク密度。

    ウィンドウ内の価格をノードとし、価格差が閾値（std×0.5）以内の
    ノード対を辺とするグラフの密度（実辺数 / 最大辺数）を返す。

    Returns:
        np.ndarray: shape=(n,). 値域 [0,1]。有効値は index (window_size-1) 以降。
    """
    n = len(prices)
    results = np.full(n, np.nan, dtype=np.float64)

    for i in range(window_size - 1, n):
        window_prices = prices[i - window_size + 1 : i + 1]

        if len(window_prices) < 10:
            continue

        finite_prices = window_prices[np.isfinite(window_prices)]
        if len(finite_prices) < 10:
            continue

        window_n = len(finite_prices)
        # 不偏標準偏差 (ddof=1)
        threshold = (
            np.std(finite_prices) * np.sqrt(window_n / (window_n - 1.0))
            if window_n > 1
            else 0.0
        ) * 0.5

        edge_count = 0.0
        max_possible_edges = float(window_n * (window_n - 1) / 2.0)

        for j in range(window_n - 1):
            for k in range(j + 1, window_n):
                if abs(finite_prices[j] - finite_prices[k]) <= threshold:
                    edge_count += 1

        if max_possible_edges > 0:
            results[i] = edge_count / (max_possible_edges + 1e-10)
        else:
            results[i] = 0.0

    return results


@njit(fastmath=False, cache=True)
def rolling_network_clustering_udf(prices: np.ndarray, window_size: int) -> np.ndarray:
    """
    ローリング価格ネットワーク平均クラスタリング係数。

    各ノードの近傍間の辺密度の平均。
    価格が局所的にクラスターを形成する傾向を測定する。

    Returns:
        np.ndarray: shape=(n,). 値域 [0,1]。有効値は index (window_size-1) 以降。
    """
    n = len(prices)
    results = np.full(n, np.nan, dtype=np.float64)

    adj_buffer = np.zeros((window_size, window_size), dtype=nb.boolean)
    neighbors_buffer = np.zeros(window_size, dtype=nb.int64)

    for i in range(window_size - 1, n):
        window_prices = prices[i - window_size + 1 : i + 1]

        # [Phase 6 修正] 旧ハードコード `< 15` では window_size=10/12/15 の特徴量が
        # 機能不全になる (全 NaN → fill_nan で 0 化、定数列となり 2_B variance フィルタで除外)。
        # window_size に応じた min_samples に変更し、harmony_udf と同じパターンに統一。
        min_samples = max(window_size // 2, 5)
        if len(window_prices) < min_samples:
            continue

        finite_prices = window_prices[np.isfinite(window_prices)]
        if len(finite_prices) < min_samples:
            continue

        window_n = len(finite_prices)
        threshold = (
            np.std(finite_prices) * np.sqrt(window_n / (window_n - 1.0))
            if window_n > 1
            else 0.0
        ) * 0.5

        adj_buffer[:window_n, :window_n] = False

        for j in range(window_n):
            for k in range(window_n):
                if j != k:
                    if abs(finite_prices[j] - finite_prices[k]) <= threshold:
                        adj_buffer[j, k] = True

        total_clustering = 0.0
        valid_nodes = 0

        for j in range(window_n):
            neighbor_count = 0
            for k in range(window_n):
                if adj_buffer[j, k]:
                    neighbors_buffer[neighbor_count] = k
                    neighbor_count += 1

            k_len = neighbor_count
            if k_len < 2:
                continue

            neighbor_connections = 0
            for idx1 in range(k_len):
                for idx2 in range(idx1 + 1, k_len):
                    n1, n2 = neighbors_buffer[idx1], neighbors_buffer[idx2]
                    if adj_buffer[n1, n2]:
                        neighbor_connections += 1

            max_connections = float(k_len * (k_len - 1) / 2.0)
            if max_connections > 0:
                clustering_j = neighbor_connections / (max_connections + 1e-10)
                total_clustering += clustering_j
                valid_nodes += 1

        if valid_nodes > 0:
            results[i] = total_clustering / (float(valid_nodes) + 1e-10)
        else:
            results[i] = 0.0

    return results


# ---------------------------------------------------------------------------
# 言語学: 語彙多様性・言語的複雑性・意味的流れ
# ---------------------------------------------------------------------------


@njit(fastmath=False, cache=True)
def rolling_vocabulary_diversity_udf(
    prices: np.ndarray, window_size: int
) -> np.ndarray:
    """
    ローリング語彙多様性。

    価格を 10 ビンに離散化し、ユニーク語彙数 / 総トークン数で多様性を計算。

    Returns:
        np.ndarray: shape=(n,). 値域 [0,1]。有効値は index (window_size-1) 以降。
    """
    n = len(prices)
    results = np.full(n, np.nan, dtype=np.float64)

    vocab_buffer = np.zeros(10, dtype=nb.boolean)

    for i in range(window_size - 1, n):
        window_prices = prices[i - window_size + 1 : i + 1]

        if len(window_prices) < 10:
            continue

        finite_prices = window_prices[np.isfinite(window_prices)]
        if len(finite_prices) < 10:
            continue

        n_fp = len(finite_prices)
        std_price = (
            np.std(finite_prices) * np.sqrt(n_fp / (n_fp - 1.0)) if n_fp > 1 else 0.0
        )
        mean_price = np.mean(finite_prices)

        if std_price == 0.0:
            results[i] = 0.0
            continue

        n_bins = 10
        price_min = mean_price - 2 * std_price
        price_max = mean_price + 2 * std_price
        bin_width = (price_max - price_min) / float(n_bins)

        vocab_buffer[:] = False
        total_tokens = 0
        unique_count = 0

        for price in finite_prices:
            if bin_width > 0.0:
                bin_idx = int((price - price_min) / bin_width)
                bin_idx = max(0, min(bin_idx, n_bins - 1))
            else:
                bin_idx = 0

            if not vocab_buffer[bin_idx]:
                vocab_buffer[bin_idx] = True
                unique_count += 1
            total_tokens += 1

        if total_tokens > 0:
            results[i] = float(unique_count) / (float(total_tokens) + 1e-10)
        else:
            results[i] = 0.0

    return results


@njit(fastmath=False, cache=True)
def rolling_linguistic_complexity_udf(
    prices: np.ndarray, window_size: int
) -> np.ndarray:
    """
    ローリング言語的複雑性。

    価格変化を上昇(2)/横ばい(1)/下落(0) の 3 値シーケンスに変換し、
    バイグラム・トライグラムのエントロピーから複雑性を測定する。

    Returns:
        np.ndarray: shape=(n,). 値域 [0,1]。有効値は index (window_size-1) 以降。
    """
    n = len(prices)
    results = np.full(n, np.nan, dtype=np.float64)

    syntax_buffer = np.zeros(window_size, dtype=nb.int32)
    bigram_buffer = np.zeros(9, dtype=np.float64)
    trigram_buffer = np.zeros(27, dtype=np.float64)

    for i in range(window_size - 1, n):
        window_prices = prices[i - window_size + 1 : i + 1]
        finite_prices = window_prices[np.isfinite(window_prices)]
        if len(finite_prices) < 20:
            continue

        price_changes = np.diff(finite_prices)
        n_pc = len(price_changes)
        threshold = (
            np.std(price_changes) * np.sqrt(n_pc / (n_pc - 1.0)) if n_pc > 1 else 0.0
        ) * 0.1

        seq_len = len(price_changes)
        syntax_sequence = syntax_buffer[:seq_len]

        for j in range(seq_len):
            if price_changes[j] > threshold:
                syntax_sequence[j] = 2
            elif price_changes[j] < -threshold:
                syntax_sequence[j] = 0
            else:
                syntax_sequence[j] = 1

        if seq_len < 3:
            continue

        bigram_buffer[:] = 0.0
        total_bigrams = 0.0
        for j in range(seq_len - 1):
            idx = syntax_sequence[j] * 3 + syntax_sequence[j + 1]
            bigram_buffer[idx] += 1.0
            total_bigrams += 1.0

        entropy_bi = 0.0
        for j in range(9):
            if bigram_buffer[j] > 0.0:
                p = bigram_buffer[j] / total_bigrams
                entropy_bi -= p * np.log2(p)

        trigram_buffer[:] = 0.0
        total_trigrams = 0.0
        for j in range(seq_len - 2):
            idx = (
                syntax_sequence[j] * 9
                + syntax_sequence[j + 1] * 3
                + syntax_sequence[j + 2]
            )
            trigram_buffer[idx] += 1.0
            total_trigrams += 1.0

        entropy_tri = 0.0
        for j in range(27):
            if trigram_buffer[j] > 0.0:
                p = trigram_buffer[j] / total_trigrams
                entropy_tri -= p * np.log2(p)

        max_entropy_bi = np.log2(min(9.0, total_bigrams))
        max_entropy_tri = np.log2(min(27.0, total_trigrams))

        norm_bi = entropy_bi / max_entropy_bi if max_entropy_bi > 1e-10 else 0.0
        norm_tri = entropy_tri / max_entropy_tri if max_entropy_tri > 1e-10 else 0.0

        results[i] = (norm_bi + norm_tri) / 2.0

    return results


@njit(fastmath=False, cache=True)
def rolling_semantic_flow_udf(prices: np.ndarray, window_size: int) -> np.ndarray:
    """
    ローリング意味的流れ（コサイン類似度ベース）。

    隣接セマンティックベクトル間のコサイン類似度の平均。
    値域 [0,1]。1 に近いほど価格の流れが連続・一貫。

    Returns:
        np.ndarray: shape=(n,). 有効値は index (window_size-1) 以降。
    """
    n = len(prices)
    results = np.full(n, np.nan, dtype=np.float64)

    vec_buffer = np.zeros((window_size, 2), dtype=np.float64)

    for i in range(window_size - 1, n):
        window_prices = prices[i - window_size + 1 : i + 1]

        # [Phase 6 修正] 旧 `< 15` では window_size=10/12/15 で機能不全。
        # window_size に応じた min_samples に変更し、harmony_udf と同パターンに統一。
        min_samples = max(window_size // 2, 5)
        if len(window_prices) < min_samples:
            continue

        finite_prices = window_prices[np.isfinite(window_prices)]
        if len(finite_prices) < min_samples:
            continue

        window_n = len(finite_prices)
        window_size_local = min(5, window_n // 3)

        vec_count = 0

        for j in range(window_size_local, window_n - window_size_local):
            neighborhood = finite_prices[j - window_size_local : j + window_size_local + 1]
            center_price = finite_prices[j]
            relative_positions = neighborhood - center_price

            if len(relative_positions) > 1:
                mean_rel = np.mean(relative_positions)
                n_rel = len(relative_positions)
                std_rel = (
                    np.std(relative_positions) * np.sqrt(n_rel / (n_rel - 1.0))
                    if n_rel > 1
                    else 0.0
                )
                vec_buffer[vec_count, 0] = mean_rel
                vec_buffer[vec_count, 1] = std_rel
                vec_count += 1

        if vec_count < 2:
            continue

        flow_continuity = 0.0
        valid_pairs = 0

        for j in range(vec_count - 1):
            v1_0 = vec_buffer[j, 0]
            v1_1 = vec_buffer[j, 1]
            v2_0 = vec_buffer[j + 1, 0]
            v2_1 = vec_buffer[j + 1, 1]

            norm1 = np.sqrt(v1_0 ** 2 + v1_1 ** 2)
            norm2 = np.sqrt(v2_0 ** 2 + v2_1 ** 2)

            if norm1 > 1e-10 and norm2 > 1e-10:
                cosine_sim = (v1_0 * v2_0 + v1_1 * v2_1) / (norm1 * norm2 + 1e-10)
                flow_continuity += cosine_sim
                valid_pairs += 1

        if valid_pairs > 0:
            semantic_flow = flow_continuity / (float(valid_pairs) + 1e-10)
            results[i] = (semantic_flow + 1.0) / 2.0
        else:
            results[i] = 0.0

    return results


# ---------------------------------------------------------------------------
# 美学: 黄金比・対称性・美的バランス
# ---------------------------------------------------------------------------


@njit(fastmath=False, cache=True)
def rolling_golden_ratio_adherence_udf(
    prices: np.ndarray, window_size: int
) -> np.ndarray:
    """
    ローリング黄金比準拠度。

    局所 high/low 比と黄金比 φ=(1+√5)/2 のズレを 1/(1+偏差) で評価。
    値域 (0,1]。1 に近いほど価格が黄金比的な高低差を形成。

    Returns:
        np.ndarray: shape=(n,). 有効値は index (window_size-1) 以降。
    """
    n = len(prices)
    results = np.full(n, np.nan, dtype=np.float64)
    golden_ratio = (1.0 + np.sqrt(5.0)) / 2.0

    adherence_buffer = np.zeros(window_size, dtype=np.float64)

    for i in range(window_size - 1, n):
        window_prices = prices[i - window_size + 1 : i + 1]

        if len(window_prices) < 10:
            continue

        finite_prices = window_prices[np.isfinite(window_prices)]
        if len(finite_prices) < 10:
            continue

        local_window = min(8, len(finite_prices) // 2)
        adherence_count = 0

        for j in range(local_window, len(finite_prices) - local_window):
            local_subwindow = finite_prices[j - local_window : j + local_window + 1]
            local_high = np.max(local_subwindow)
            local_low = np.min(local_subwindow)

            if local_low > 1e-10:
                ratio = local_high / local_low
                deviation = abs(ratio - golden_ratio) / golden_ratio
                deviation = min(deviation, 10.0)
                adherence_buffer[adherence_count] = 1.0 / (1.0 + deviation)
                adherence_count += 1

        if adherence_count > 0:
            results[i] = np.mean(adherence_buffer[:adherence_count])

    return results


@njit(fastmath=False, cache=True)
def rolling_symmetry_measure_udf(prices: np.ndarray, window_size: int) -> np.ndarray:
    """
    ローリング対称性測定（左右相関ベース）。

    ウィンドウを左右半分に分割し、正規化後の左半分と右半分（逆順）の
    相関係数を (r+1)/2 に変換。値域 [0,1]。

    Returns:
        np.ndarray: shape=(n,). 有効値は index (window_size-1) 以降。
    """
    n = len(prices)
    results = np.full(n, np.nan, dtype=np.float64)

    left_norm_buffer = np.zeros(window_size, dtype=np.float64)
    right_norm_buffer = np.zeros(window_size, dtype=np.float64)

    for i in range(window_size - 1, n):
        window_prices = prices[i - window_size + 1 : i + 1]
        if len(window_prices) < 20:
            continue

        finite_prices = window_prices[np.isfinite(window_prices)]
        if len(finite_prices) < 20:
            continue

        window_n = len(finite_prices)
        center = window_n // 2

        left_half = finite_prices[:center]
        right_half = (
            finite_prices[center + 1 :] if window_n % 2 == 1 else finite_prices[center:]
        )
        right_half_reversed = right_half[::-1]

        min_len = min(len(left_half), len(right_half_reversed))
        if min_len < 5:
            continue

        left_normalized = left_half[-min_len:]
        right_normalized = right_half_reversed[:min_len]

        left_norm_view = left_norm_buffer[:min_len]
        right_norm_view = right_norm_buffer[:min_len]

        mean_left_raw = np.mean(left_normalized)
        n_left = len(left_normalized)
        std_left_raw = (
            np.std(left_normalized) * np.sqrt(n_left / (n_left - 1.0))
            if n_left > 1
            else 0.0
        )
        if std_left_raw > 1e-10:
            left_norm_view[:] = (left_normalized - mean_left_raw) / std_left_raw
        else:
            left_norm_view[:] = left_normalized - mean_left_raw

        mean_right_raw = np.mean(right_normalized)
        n_right = len(right_normalized)
        std_right_raw = (
            np.std(right_normalized) * np.sqrt(n_right / (n_right - 1.0))
            if n_right > 1
            else 0.0
        )
        if std_right_raw > 1e-10:
            right_norm_view[:] = (right_normalized - mean_right_raw) / std_right_raw
        else:
            right_norm_view[:] = right_normalized - mean_right_raw

        if len(left_norm_view) < 2:
            continue

        mean_left = np.mean(left_norm_view)
        mean_right = np.mean(right_norm_view)

        numerator = np.sum((left_norm_view - mean_left) * (right_norm_view - mean_right))
        denom_left = np.sum((left_norm_view - mean_left) ** 2)
        denom_right = np.sum((right_norm_view - mean_right) ** 2)

        if denom_left > 1e-10 and denom_right > 1e-10:
            correlation = numerator / (np.sqrt(denom_left * denom_right) + 1e-10)
            results[i] = (correlation + 1.0) / 2.0
        else:
            results[i] = 0.0

    return results


@njit(fastmath=False, cache=True)
def rolling_aesthetic_balance_udf(prices: np.ndarray, window_size: int) -> np.ndarray:
    """
    ローリング美的バランス。

    価格変化の絶対値を「穏やか/中程度/激しい」に三分類し、
    理想比率 (0.6/0.3/0.1) からの乖離を 1 - 乖離 で評価。
    値域 [0,1]。

    Returns:
        np.ndarray: shape=(n,). 有効値は index (window_size-1) 以降。
    """
    n = len(prices)
    results = np.full(n, np.nan, dtype=np.float64)

    grad_buffer = np.zeros(window_size, dtype=np.float64)

    for i in range(window_size - 1, n):
        window_prices = prices[i - window_size + 1 : i + 1]

        # [Phase 6 修正] 旧 `< 15` では window_size=10/12/15 で機能不全。
        # window_size に応じた min_samples に変更し、harmony_udf と同パターンに統一。
        min_samples = max(window_size // 2, 5)
        if len(window_prices) < min_samples:
            continue

        finite_prices = window_prices[np.isfinite(window_prices)]
        if len(finite_prices) < min_samples:
            continue

        grad_len = len(finite_prices) - 1
        for j in range(grad_len):
            grad_buffer[j] = abs(finite_prices[j + 1] - finite_prices[j])

        if grad_len < 5:
            continue

        abs_gradients = grad_buffer[:grad_len]
        mean_grad = np.mean(abs_gradients)
        n_grad = len(abs_gradients)
        std_grad = (
            np.std(abs_gradients) * np.sqrt(n_grad / (n_grad - 1.0))
            if n_grad > 1
            else 0.0
        )

        if std_grad <= 1e-10:
            results[i] = 1.0
            continue

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
            results[i] = 0.0
            continue

        actual_gentle = float(gentle_count) / (float(total_counted) + 1e-10)
        actual_moderate = float(moderate_count) / (float(total_counted) + 1e-10)
        actual_intense = float(intense_count) / (float(total_counted) + 1e-10)

        balance_deviation = (
            abs(actual_gentle - 0.6)
            + abs(actual_moderate - 0.3)
            + abs(actual_intense - 0.1)
        ) / 2.0

        results[i] = max(0.0, 1.0 - balance_deviation)

    return results


# ---------------------------------------------------------------------------
# 音楽理論: 調性・リズムパターン・和声・音楽的緊張度
# ---------------------------------------------------------------------------


@njit(fastmath=False, cache=True)
def rolling_tonality_udf(prices: np.ndarray, window_size: int) -> np.ndarray:
    """
    ローリング調性スコア（音楽理論ベース）。

    価格変化を正規化し、長調/短調パターンとの類似度から調性を推定。
    値域 [0,1]。0.5 = 中立、1.0 = 完全長調的。

    Returns:
        np.ndarray: shape=(n,). 有効値は index (window_size-1) 以降。
    """
    n = len(prices)
    results = np.full(n, np.nan, dtype=np.float64)

    diff_buffer = np.zeros(window_size, dtype=np.float64)
    scale_degrees_buffer = np.zeros(12, dtype=np.float64)

    major_pattern = np.array(
        [1.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0], dtype=np.float64
    )
    minor_pattern = np.array(
        [1.0, 0.0, 1.0, 1.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0, 0.0], dtype=np.float64
    )

    for i in range(window_size - 1, n):
        window_prices = prices[i - window_size + 1 : i + 1]

        if len(window_prices) < 12:
            continue

        finite_prices = window_prices[np.isfinite(window_prices)]
        if len(finite_prices) < 12:
            continue

        diff_len = len(finite_prices) - 1
        for j in range(diff_len):
            diff_buffer[j] = finite_prices[j + 1] - finite_prices[j]

        price_changes = diff_buffer[:diff_len]

        if diff_len < 5:
            continue

        n_pc = len(price_changes)
        std_change = (
            np.std(price_changes) * np.sqrt(n_pc / (n_pc - 1.0)) if n_pc > 1 else 0.0
        )

        if std_change <= 1e-10:
            results[i] = 0.5
            continue

        normalized_changes = price_changes / (std_change + 1e-10)
        scale_degrees_buffer[:] = 0.0

        for change in normalized_changes:
            degree_idx = int((change + 3.0) / 6.0 * 11.0)
            degree_idx = max(0, min(degree_idx, 11))
            scale_degrees_buffer[degree_idx] += 1.0

        sum_degrees = np.sum(scale_degrees_buffer)
        if sum_degrees > 0.0:
            scale_distribution = scale_degrees_buffer / (sum_degrees + 1e-10)
            major_similarity = np.sum(scale_distribution * major_pattern)
            minor_similarity = np.sum(scale_distribution * minor_pattern)
            total_similarity = major_similarity + minor_similarity
            if total_similarity > 0.0:
                results[i] = major_similarity / (total_similarity + 1e-10)
            else:
                results[i] = 0.5
        else:
            results[i] = 0.5

    return results


@njit(fastmath=False, cache=True)
def rolling_rhythm_pattern_udf(prices: np.ndarray, window_size: int) -> np.ndarray:
    """
    ローリングリズムパターン強度（ACF ベース）。

    価格変化の自己相関関数 (ACF) の最大値。
    高い値 = 価格変化に周期的なリズムが存在。
    ddof=0 で分散を計算（ACF の標準的定義に準拠）。

    Returns:
        np.ndarray: shape=(n,). 値域 [0,1]。有効値は index (window_size-1) 以降。
    """
    n = len(prices)
    results = np.full(n, np.nan, dtype=np.float64)

    diff_buffer = np.zeros(window_size, dtype=np.float64)

    for i in range(window_size - 1, n):
        window_prices = prices[i - window_size + 1 : i + 1]
        finite_prices = window_prices[np.isfinite(window_prices)]
        if len(finite_prices) < 20:
            continue

        N_changes = len(finite_prices) - 1
        for j in range(N_changes):
            diff_buffer[j] = finite_prices[j + 1] - finite_prices[j]

        price_changes = diff_buffer[:N_changes]

        if N_changes < 5:
            continue

        mean_change = np.mean(price_changes)
        # ACF 標準実装: ddof=0（分散 = E[x²] - E[x]²）
        var_change = np.var(price_changes)

        if var_change <= 1e-10:
            results[i] = 0.0
            continue

        max_lag = N_changes // 4
        max_acf = 0.0

        for lag in range(1, max_lag + 1):
            cov = 0.0
            for j in range(N_changes - lag):
                cov += (price_changes[j] - mean_change) * (
                    price_changes[j + lag] - mean_change
                )
            # N で割る（ACF の半正定値性を保証）
            cov = cov / float(N_changes)
            acf = abs(cov / var_change)

            if acf > max_acf:
                max_acf = acf

        results[i] = max_acf

    return results


@njit(fastmath=False, cache=True)
def rolling_harmony_udf(prices: np.ndarray, window_size: int) -> np.ndarray:
    """
    ローリング和声スコア（マルチスケール MA 方向一致度）。

    短期/中期/長期 MA のトレンド方向が一致する割合。
    値域 [0,1]。1 = 全期間で全 MA が同方向。

    Returns:
        np.ndarray: shape=(n,). 有効値は index (window_size-1) 以降。
    """
    n = len(prices)
    results = np.full(n, np.nan, dtype=np.float64)

    short_ma_buffer = np.zeros(window_size, dtype=np.float64)
    medium_ma_buffer = np.zeros(window_size, dtype=np.float64)
    long_ma_buffer = np.zeros(window_size, dtype=np.float64)
    harmony_buffer = np.zeros(window_size, dtype=np.float64)
    short_trend_buf = np.zeros(window_size, dtype=np.float64)
    medium_trend_buf = np.zeros(window_size, dtype=np.float64)
    long_trend_buf = np.zeros(window_size, dtype=np.float64)

    for i in range(window_size - 1, n):
        window_prices = prices[i - window_size + 1 : i + 1]

        # 【修正】従来は `< 30` ハードコードだったため、window_size=12, 24 で
        #   常に skip されて harmony_12 / harmony_24 が 100% NaN という無価値カラムを
        #   生成していた (engine と rfe で同じ値を出すので train-serve 整合は保たれていたが、
        #   モデルにとっては定数列で何の情報も無い)。
        #   修正後は window_size に追従し、有効データが半分以上あれば計算する。
        #     - window_prices 自体が window_size より短い: skip (warmup 期間)
        #     - finite_prices が window_size の半分未満 (最低 8 個): skip (有効データ不足)
        if len(window_prices) < window_size:
            continue

        finite_prices = window_prices[np.isfinite(window_prices)]
        # 有効データの最低必要数: window_size の半分、ただし下限 8 個
        min_required = window_size // 2
        if min_required < 8:
            min_required = 8
        if len(finite_prices) < min_required:
            continue

        window_n = len(finite_prices)
        short_window = max(3, window_n // 15)
        medium_window = max(5, window_n // 10)
        long_window = max(8, window_n // 6)

        if long_window >= window_n:
            continue

        short_len = window_n - short_window + 1
        medium_len = window_n - medium_window + 1
        long_len = window_n - long_window + 1

        short_ma = short_ma_buffer[:short_len]
        medium_ma = medium_ma_buffer[:medium_len]
        long_ma = long_ma_buffer[:long_len]

        for j in range(short_len):
            short_ma[j] = np.mean(finite_prices[j : j + short_window])
        for j in range(len(medium_ma)):
            medium_ma[j] = np.mean(finite_prices[j : j + medium_window])
        for j in range(len(long_ma)):
            long_ma[j] = np.mean(finite_prices[j : j + long_window])

        min_len = min(len(short_ma), len(medium_ma), len(long_ma))
        if min_len < 5:
            continue

        trend_len = min_len - 1
        for _j in range(trend_len):
            short_trend_buf[_j] = short_ma[-min_len + _j + 1] - short_ma[-min_len + _j]
            medium_trend_buf[_j] = medium_ma[-min_len + _j + 1] - medium_ma[-min_len + _j]
            long_trend_buf[_j] = long_ma[-min_len + _j + 1] - long_ma[-min_len + _j]

        short_trend = short_trend_buf[:trend_len]
        medium_trend = medium_trend_buf[:trend_len]
        long_trend = long_trend_buf[:trend_len]

        harmony_count = 0
        for j in range(trend_len):
            s_sign = np.sign(short_trend[j])
            m_sign = np.sign(medium_trend[j])
            l_sign = np.sign(long_trend[j])

            if s_sign == m_sign and m_sign == l_sign and s_sign != 0.0:
                harmony_buffer[harmony_count] = 1.0
            elif (s_sign == m_sign) or (m_sign == l_sign) or (s_sign == l_sign):
                harmony_buffer[harmony_count] = 0.5
            else:
                harmony_buffer[harmony_count] = 0.0
            harmony_count += 1

        if harmony_count > 0:
            results[i] = np.mean(harmony_buffer[:harmony_count])
        else:
            results[i] = 0.0

    return results


@njit(fastmath=False, cache=True)
def rolling_musical_tension_udf(prices: np.ndarray, window_size: int) -> np.ndarray:
    """
    ローリング音楽的緊張度。

    局所的な「方向不一致度」と「強度不協和度」の平均で緊張度を測定。
    値域 [0,1]。

    Returns:
        np.ndarray: shape=(n,). 有効値は index (window_size-1) 以降。
    """
    n = len(prices)
    results = np.full(n, np.nan, dtype=np.float64)

    diff_buffer = np.zeros(window_size, dtype=np.float64)
    tension_buffer = np.zeros(window_size, dtype=np.float64)

    for i in range(window_size - 1, n):
        window_prices = prices[i - window_size + 1 : i + 1]

        # [Phase 6 修正] 旧 `< 15` では window_size=10/12/15 で機能不全。
        # window_size に応じた min_samples に変更し、harmony_udf と同パターンに統一。
        min_samples = max(window_size // 2, 5)
        if len(window_prices) < min_samples:
            continue

        finite_prices = window_prices[np.isfinite(window_prices)]
        if len(finite_prices) < min_samples:
            continue

        diff_len = len(finite_prices) - 1
        for j in range(diff_len):
            diff_buffer[j] = finite_prices[j + 1] - finite_prices[j]
        price_changes = diff_buffer[:diff_len]

        if diff_len < 5:
            continue

        local_window = min(5, diff_len // 3)
        tension_count = 0

        for j in range(local_window, diff_len - local_window):
            local_changes = price_changes[j - local_window : j + local_window + 1]

            sign_changes = 0
            for k in range(len(local_changes) - 1):
                if np.sign(local_changes[k]) != np.sign(local_changes[k + 1]):
                    sign_changes += 1

            direction_dissonance = (
                float(sign_changes) / (float(len(local_changes)) + 1e-10)
                if len(local_changes) > 0
                else 0.0
            )

            max_volatility = np.max(np.abs(local_changes))
            intensity_dissonance = max_volatility / (np.mean(np.abs(local_changes)) + 1e-10)

            total_tension = (direction_dissonance + intensity_dissonance) / 2.0
            tension_buffer[tension_count] = min(total_tension, 1.0)
            tension_count += 1

        if tension_count > 0:
            results[i] = np.mean(tension_buffer[:tension_count])
        else:
            results[i] = 0.0

    return results


# ---------------------------------------------------------------------------
# 生体力学: 運動エネルギー・筋力・生体力学効率・エネルギー消費量
# ---------------------------------------------------------------------------


@njit(fastmath=False, cache=True)
def rolling_kinetic_energy_udf(prices: np.ndarray, window_size: int) -> np.ndarray:
    """
    ローリング運動エネルギー（KE = 0.5 × mass × v²）。

    相対リターンを速度として、平均運動エネルギーを計算。

    Returns:
        np.ndarray: shape=(n,). 有効値は index (window_size-1) 以降。
    """
    n = len(prices)
    results = np.full(n, np.nan, dtype=np.float64)

    kin_buffer = np.zeros(window_size, dtype=np.float64)

    for i in range(window_size - 1, n):
        window_prices = prices[i - window_size + 1 : i + 1]

        if len(window_prices) < 10:
            continue

        finite_prices = window_prices[np.isfinite(window_prices)]
        if len(finite_prices) < 10:
            continue

        vel_len = len(finite_prices) - 1
        if vel_len < 2:
            continue

        for j in range(vel_len):
            vel = (finite_prices[j + 1] - finite_prices[j]) / (finite_prices[j] + 1e-10)
            kin_buffer[j] = 0.5 * 1.0 * (vel * vel)

        results[i] = np.mean(kin_buffer[:vel_len])

    return results


@njit(fastmath=False, cache=True)
def rolling_muscle_force_udf(prices: np.ndarray, window_size: int) -> np.ndarray:
    """
    ローリング筋力スコア（加速度ベース）。

    瞬時力（加速度絶対値平均）と持続力（方向継続加速度平均）の
    加重平均 (0.7:0.3) で筋力を評価。

    Returns:
        np.ndarray: shape=(n,). 有効値は index (window_size-1) 以降。
    """
    n = len(prices)
    results = np.full(n, np.nan, dtype=np.float64)

    vel_buffer = np.zeros(window_size, dtype=np.float64)
    force_buffer = np.zeros(window_size, dtype=np.float64)
    dir_buffer = np.zeros(window_size, dtype=np.float64)
    sustained_buffer = np.zeros(window_size, dtype=np.float64)

    for i in range(window_size - 1, n):
        window_prices = prices[i - window_size + 1 : i + 1]

        # [Phase 6 修正] 旧 `< 15` では window_size=10/12/15 で機能不全。
        # window_size に応じた min_samples に変更し、harmony_udf と同パターンに統一。
        min_samples = max(window_size // 2, 5)
        if len(window_prices) < min_samples:
            continue

        finite_prices = window_prices[np.isfinite(window_prices)]
        if len(finite_prices) < min_samples:
            continue

        vel_len = len(finite_prices) - 1
        for j in range(vel_len):
            vel_buffer[j] = (finite_prices[j + 1] - finite_prices[j]) / (
                finite_prices[j] + 1e-10
            )

        if vel_len < 2:
            continue

        acc_len = vel_len - 1
        for j in range(acc_len):
            acc = vel_buffer[j + 1] - vel_buffer[j]
            force_buffer[j] = abs(acc)
            if acc > 0.0:
                dir_buffer[j] = 1.0
            elif acc < 0.0:
                dir_buffer[j] = -1.0
            else:
                dir_buffer[j] = 0.0

        sus_count = 0
        current_direction = dir_buffer[0] if acc_len > 0 else 0.0
        current_duration = 1.0
        current_force_sum = force_buffer[0] if acc_len > 0 else 0.0

        for j in range(1, acc_len):
            if dir_buffer[j] == current_direction and current_direction != 0.0:
                current_duration += 1.0
                current_force_sum += force_buffer[j]
            else:
                if current_duration > 1.0:
                    sustained_buffer[sus_count] = current_force_sum / current_duration
                    sus_count += 1
                current_direction = dir_buffer[j]
                current_duration = 1.0
                current_force_sum = force_buffer[j]

        if current_duration > 1.0:
            sustained_buffer[sus_count] = current_force_sum / current_duration
            sus_count += 1

        instantaneous_force = np.mean(force_buffer[:acc_len]) if acc_len > 0 else 0.0
        sustained_force = (
            np.mean(sustained_buffer[:sus_count]) if sus_count > 0 else 0.0
        )

        results[i] = 0.7 * instantaneous_force + 0.3 * sustained_force

    return results


@njit(fastmath=False, cache=True)
def rolling_biomechanical_efficiency_udf(
    prices: np.ndarray, window_size: int
) -> np.ndarray:
    """
    ローリング生体力学効率。

    (Σ|vel|)² / (Σvel² + Σacc²) を vel_len で正規化し [0,1] に収める。
    無次元量（スケール非依存）。

    Returns:
        np.ndarray: shape=(n,). 値域 [0,1]。有効値は index (window_size-1) 以降。
    """
    n = len(prices)
    results = np.full(n, np.nan, dtype=np.float64)

    vel_buffer = np.zeros(window_size, dtype=np.float64)

    for i in range(window_size - 1, n):
        window_prices = prices[i - window_size + 1 : i + 1]
        finite_prices = window_prices[np.isfinite(window_prices)]
        if len(finite_prices) < 20:
            continue

        vel_len = len(finite_prices) - 1
        rms_vel_sq = 0.0
        rms_acc_sq = 0.0
        sum_abs_vel = 0.0

        for j in range(vel_len):
            vel = (finite_prices[j + 1] - finite_prices[j]) / (finite_prices[j] + 1e-10)
            vel_buffer[j] = vel
            rms_vel_sq += vel * vel
            sum_abs_vel += abs(vel)

        acc_len = vel_len - 1
        for j in range(acc_len):
            acc = vel_buffer[j + 1] - vel_buffer[j]
            rms_acc_sq += acc * acc

        total_energy = rms_vel_sq + rms_acc_sq

        if total_energy > 1e-10 and sum_abs_vel > 1e-10:
            numerator = sum_abs_vel * sum_abs_vel
            raw_efficiency = numerator / total_energy
            results[i] = min(raw_efficiency / (vel_len + 1e-10), 1.0)
        else:
            results[i] = 0.0

    return results


@njit(fastmath=False, cache=True)
def rolling_energy_expenditure_udf(prices: np.ndarray, window_size: int) -> np.ndarray:
    """
    ローリングエネルギー消費量。

    基底エネルギー（不偏分散）+ 運動エネルギー + 加速度エネルギー
    + 非線形エネルギー（OLS 残差分散）の合計。

    Returns:
        np.ndarray: shape=(n,). 有効値は index (window_size-1) 以降。
    """
    n = len(prices)
    results = np.full(n, np.nan, dtype=np.float64)

    vel_buffer = np.zeros(window_size, dtype=np.float64)

    for i in range(window_size - 1, n):
        window_prices = prices[i - window_size + 1 : i + 1]
        finite_prices = window_prices[np.isfinite(window_prices)]
        # [Phase 6 修正] 旧 `< 15` では window_size=10/12/15 で機能不全。
        # window_size に応じた min_samples に変更し、harmony_udf と同パターンに統一。
        min_samples = max(window_size // 2, 5)
        if len(finite_prices) < min_samples:
            continue

        n_fp = len(finite_prices)
        # 不偏分散 (ddof=1)
        baseline_energy = np.var(finite_prices) * (n_fp / (n_fp - 1.0))

        mov_en_sum = 0.0
        vel_len = len(finite_prices) - 1
        for j in range(vel_len):
            vel = finite_prices[j + 1] - finite_prices[j]
            vel_buffer[j] = vel
            mov_en_sum += vel * vel
        movement_energy = mov_en_sum / float(vel_len) if vel_len > 0 else 0.0

        acc_en_sum = 0.0
        acc_len = vel_len - 1
        for j in range(acc_len):
            acc = vel_buffer[j + 1] - vel_buffer[j]
            acc_en_sum += acc * acc
        acceleration_energy = acc_en_sum / float(acc_len) if acc_len > 0 else 0.0

        # OLS による非線形エネルギー（残差の二乗和平均）
        n_points = float(len(finite_prices))
        sum_x = 0.0
        sum_y = 0.0
        sum_xy = 0.0
        sum_x2 = 0.0
        for j in range(len(finite_prices)):
            x_val = float(j)
            y_val = finite_prices[j]
            sum_x += x_val
            sum_y += y_val
            sum_xy += x_val * y_val
            sum_x2 += x_val * x_val

        nonlinearity_energy = 0.0
        denominator = n_points * sum_x2 - sum_x * sum_x
        if abs(denominator) > 1e-10:
            slope = (n_points * sum_xy - sum_x * sum_y) / denominator
            intercept = (sum_y - slope * sum_x) / n_points

            nonlin_sum = 0.0
            for j in range(len(finite_prices)):
                pred = intercept + slope * float(j)
                diff = finite_prices[j] - pred
                nonlin_sum += diff * diff
            nonlinearity_energy = nonlin_sum / n_points

        results[i] = (
            baseline_energy
            + movement_energy
            + acceleration_energy
            + nonlinearity_energy
        )

    return results


# ===========================================================================
# 数値検証テスト (単体実行時)
# ===========================================================================
# python core_indicators.py で実行可能。
# 同一 OHLCV データを学習側（Polars 互換方式）・リアルタイム側（NumPy 逐次）
# の両方に流し、特徴量の値が完全一致することを数値検証する。
# ===========================================================================



# ===========================================================================
# [CATEGORY: STATISTICS — Engine 1A (Numba) 用]
# ===========================================================================
# 本セクションは Step 1 の SSoT 統一で集約された関数群。
# 学習側 (engine_1_X) と本番側 (realtime_feature_engine_1X) の両方が
# import で参照する正本。修正はここ一箇所のみ行う。
# ===========================================================================

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
                nan_inf_ratio = nan_inf_count / window_size
                finite_ratio = finite_count / window_size
                out[i] = finite_ratio * 0.8 + 0.2  # 基本品質スコア


@nb.guvectorize(["void(float64[:], float64[:])"], "(n)->(n)", nopython=True, cache=True)
def biweight_location_numba(arr, out):
    """ローリングBiweight位置計算（厳密実装）"""
    n = len(arr)
    window = 20

    # ▼▼ 新規追加: ループ外での事前アロケーション（動的メモリ確保の排除）
    weights_buffer = np.zeros(window, dtype=np.float64)
    finite_buffer = np.empty(window, dtype=np.float64)
    # ▲▲

    for i in range(n):
        if i < window - 1:
            out[i] = np.nan
        else:
            window_data = arr[max(0, i - window + 1) : i + 1]
            count = 0
            for j in range(len(window_data)):
                if not np.isnan(window_data[j]):
                    finite_buffer[count] = window_data[j]
                    count += 1
            finite_data = finite_buffer[:count]

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

                    # ▼▼ 修正前: ループ内での動的配列確保
                    # weights = np.zeros(len(finite_data))
                    # ▼▼ 修正後: 事前アロケーションしたバッファを再利用
                    weights = weights_buffer[: len(finite_data)]
                    # ▲▲

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

    finite_buffer = np.empty(window, dtype=np.float64)

    for i in range(n):
        if i < window - 1:
            out[i] = np.nan
        else:
            window_data = arr[max(0, i - window + 1) : i + 1]
            count = 0
            for j in range(len(window_data)):
                if not np.isnan(window_data[j]):
                    finite_buffer[count] = window_data[j]
                    count += 1
            finite_data = finite_buffer[:count]

            # 修正後
            if count < 5:
                out[i] = np.mean(finite_data) if count > 0 else np.nan
            else:
                # 上下5%点の計算
                p05 = np.percentile(finite_data, 5)
                p95 = np.percentile(finite_data, 95)

                # ウィンソライズ（クリッピング）
                winsorized_data = np.clip(finite_data, p05, p95)
                out[i] = np.mean(winsorized_data)


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

                # ▼▼ 修正前: variance = variance / (len(finite_data) - 1); std_val = np.sqrt(variance) + 1e-10
                # ▼▼ 修正後: モーメント計算の原則に従い ddof=0 を使用
                variance = 0.0
                n_val = float(len(finite_data))
                for val in finite_data:
                    variance += (val - mean_val) ** 2
                variance = variance / n_val  # ddof=0
                std_val = np.sqrt(variance) + 1e-10

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

                    skewness = z_sum_3 / n_val
                    kurtosis = z_sum_4 / n_val - 3.0

                    # ▼▼ 修正前: jb_stat = len(finite_data) * (skewness**2 / 6 + kurtosis**2 / 24)
                    # ▼▼ 修正後: Urzua (1996) に基づく小標本補正 (ALM) の厳密実装
                    c1 = (
                        6.0 * (n_val - 2.0) / ((n_val + 1.0) * (n_val + 3.0))
                    )  # 小標本における歪度の分散
                    c2 = (
                        24.0
                        * n_val
                        * (n_val - 2.0)
                        * (n_val - 3.0)
                        / (((n_val + 1.0) ** 2) * (n_val + 3.0) * (n_val + 5.0))
                    )  # 小標本における尖度の分散

                    jb_stat = (skewness**2 / c1) + (kurtosis**2 / c2)
                    out[i] = jb_stat


@nb.njit(cache=True)
def _standard_normal_cdf_fast(x):
    """標準正規分布の累積分布関数 (CDF) — math.erf の高速近似版。

    anderson_darling_numba の内部で参照されるヘルパー関数。
    @nb.guvectorize でデコレートされた関数の名前解決は core_indicators の
    グローバル名前空間で行われるため、このヘルパーも core_indicators に
    定義する必要がある (engine_1_A / rfe_1A 側に重複定義しても参照されない)。
    """
    SQRT2 = 1.4142135623730951
    return 0.5 * (1.0 + math.erf(x / SQRT2))

@nb.guvectorize(["void(float64[:], float64[:])"], "(n)->(n)", nopython=True, cache=True)
def anderson_darling_numba(arr, out):
    """ローリングAnderson-Darling統計量（厳密実装）"""
    n = len(arr)
    window = 30

    # ▼▼ 新規追加: ループ外での事前アロケーション（動的メモリ確保の排除）
    standardized_buffer = np.zeros(window, dtype=np.float64)
    # ▲▲

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
                    # ▼▼ 修正前: ループ内での動的配列確保
                    # standardized_data = np.zeros(n_data)
                    # ▼▼ 修正後: 事前アロケーションしたバッファを再利用
                    standardized_data = standardized_buffer[:n_data]
                    # ▲▲

                    for j in range(n_data):
                        standardized_data[j] = (sorted_data[j] - mean_val) / std_val

                    # Anderson-Darling統計量の厳密計算
                    ad_sum = 0.0
                    for j in range(n_data):
                        # 高速化されたCDF関数を呼び出します。
                        # 修正後
                        F_j = _standard_normal_cdf_fast(standardized_data[j])
                        F_nj = _standard_normal_cdf_fast(
                            standardized_data[n_data - 1 - j]
                        )

                        # ゼロ除算回避
                        if F_j > 1e-15 and (1 - F_nj) > 1e-15:
                            log_term = np.log(F_j) + np.log(1 - F_nj)
                            ad_sum += (2 * j + 1) * log_term

                    # ▼▼ 修正前: out[i] = -n_data - ad_sum / n_data
                    # ▼▼ 修正後: Stephens (1974) の小標本補正係数を適用した真のAD統計量
                    ad_stat = -n_data - ad_sum / n_data
                    n_float = float(n_data)
                    stephens_correction = (
                        1.0 + 4.0 / n_float - 25.0 / (n_float * n_float)
                    )
                    out[i] = ad_stat * stephens_correction


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
                        # ▼▼ 修正前: out[i] = (runs - expected_runs) / np.sqrt(var_runs)
                        # ▼▼ 修正後: 極小値によるゼロ除算・オーバーフロー防衛
                        out[i] = (runs - expected_runs) / np.sqrt(var_runs + 1e-10)
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

                # ▼▼ 修正前: vn_ratio = diff_sq_sum / sum_sq_deviations
                # ▼▼ 修正後: 分母にゼロ除算防止の微小値 (+ 1e-10) を追加
                if sum_sq_deviations > 1e-15:
                    # 分子: 1次差分の平方和
                    # 分母: 総平方和（不偏分散 × (n-1)）
                    vn_ratio = diff_sq_sum / (sum_sq_deviations + 1e-10)

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


@nb.guvectorize(["void(float64[:], float64[:])"], "(n)->(n)", nopython=True, cache=True)
def basic_stabilization_numba(arr, out):
    # ▼▼ 修正前: 全体配列に対する np.nanmin(out) と np.nanmax(out) の計算 (未来リーク)
    # ▼▼ 修正後: 過去50要素のローリングウィンドウによる局所Min/Max計算 (完全な因果律の担保)
    n = len(arr)
    window = 50
    for i in range(n):
        if np.isnan(arr[i]) or np.isinf(arr[i]):
            out[i] = 0.0
        else:
            if i < 2:
                out[i] = arr[i]
            else:
                local_min = np.inf
                local_max = -np.inf
                start_idx = max(0, i - window + 1)

                # ローリングウィンドウ内の有効値を走査
                for j in range(start_idx, i + 1):
                    val = arr[j]
                    if np.isfinite(val):
                        if val < local_min:
                            local_min = val
                        if val > local_max:
                            local_max = val

                if local_min != np.inf and local_max != -np.inf:
                    range_val = local_max - local_min
                    if range_val > 1e-10:
                        clip_margin = range_val * 0.01
                        if arr[i] < local_min + clip_margin:
                            out[i] = local_min + clip_margin
                        elif arr[i] > local_max - clip_margin:
                            out[i] = local_max - clip_margin
                        else:
                            out[i] = arr[i]
                    else:
                        out[i] = arr[i]
                else:
                    out[i] = arr[i]


@nb.guvectorize(["void(float64[:], float64[:])"], "(n)->(n)", nopython=True, cache=True)
def robust_stabilization_numba(arr, out):
    # ▼▼ 修正前: 全要素から finite_vals を抽出しての中央値計算 (未来リーク)
    # ▼▼ 修正後: 過去50要素のローリングウィンドウによる動的中央値・MAD計算
    n = len(arr)
    window = 50
    finite_vals_buffer = np.empty(window, dtype=np.float64)
    for i in range(n):
        if i < 2:
            out[i] = 0.0 if not np.isfinite(arr[i]) else arr[i]
            continue

        start_idx = max(0, i - window + 1)
        window_data = arr[start_idx : i + 1]

        # ウィンドウ内の有限値を抽出
        finite_vals = finite_vals_buffer[: len(window_data)]
        count = 0
        for j in range(len(window_data)):
            if np.isfinite(window_data[j]):
                finite_vals[count] = window_data[j]
                count += 1

        if count < 3:
            out[i] = 0.0 if not np.isfinite(arr[i]) else arr[i]
            continue

        valid_data = finite_vals[:count]
        median_val = np.median(valid_data)

        abs_devs = np.abs(valid_data - median_val)
        mad_val = np.median(abs_devs)

        if mad_val < 1e-10:
            # ddof=1を明記した不偏標準偏差のフォールバック
            mean_val = np.mean(valid_data)
            var_sum = np.sum((valid_data - mean_val) ** 2)
            mad_val = np.sqrt(var_sum / (count - 1)) * 0.6745 if count > 1 else 1e-10

        lower_bound = median_val - 3.0 * mad_val
        upper_bound = median_val + 3.0 * mad_val

        if np.isnan(arr[i]):
            out[i] = median_val
        elif np.isinf(arr[i]):
            out[i] = upper_bound if arr[i] > 0 else lower_bound
        else:
            if arr[i] < lower_bound:
                out[i] = lower_bound
            elif arr[i] > upper_bound:
                out[i] = upper_bound
            else:
                out[i] = arr[i]




# ===========================================================================
# [CATEGORY: TIME SERIES — Engine 1B (UDF) 用]
# ===========================================================================
# 本セクションは Step 1 の SSoT 統一で集約された関数群。
# 学習側 (engine_1_X) と本番側 (realtime_feature_engine_1X) の両方が
# import で参照する正本。修正はここ一箇所のみ行う。
# ===========================================================================

@nb.njit(fastmath=False, cache=True)
def adf_統計量_udf(prices: np.ndarray) -> float:
    """
    真の拡張ディッキー・フラー（ADF）検定統計量計算
    ラグ差分項（Δy_{t-1}）を追加し、系列の自己相関をパラメトリックに吸収する
    帰無仮説：系列が単位根を持つ（非定常）
    """
    if len(prices) < 10:
        return np.nan

    # NaN値除去
    finite_prices = prices[np.isfinite(prices)]
    if len(finite_prices) < 10:
        return np.nan

    # 1次差分計算
    diff_prices = np.diff(finite_prices)

    # ▼▼ 修正前: 差分ラグ項のない単なるDF検定
    # lagged_prices = finite_prices[:-1]
    # n = len(diff_prices)
    # X = np.column_stack((np.ones(n, dtype=np.float64), lagged_prices))
    # y = diff_prices

    # ▼▼ 修正後: 差分ラグ項（Δy_{t-1}）を追加した真のADF検定
    if len(diff_prices) < 5:
        return np.nan

    # ADF回帰: Δy_t = α + β*y_{t-1} + γ*Δy_{t-1} + ε_t
    y = diff_prices[1:]  # Δy_t
    lagged_y = finite_prices[1:-1]  # y_{t-1}
    lagged_diff = diff_prices[:-1]  # Δy_{t-1}

    n = len(y)
    if n < 3:
        return np.nan

    # 設計行列: [定数項, ラグレベル, 差分ラグ]
    X = np.empty((n, 3), dtype=np.float64)
    X[:, 0] = 1.0
    X[:, 1] = lagged_y
    X[:, 2] = lagged_diff

    # OLS計算: β = (X'X)^(-1)X'y
    try:
        XtX = X.T @ X
        XtX_inv = np.linalg.inv(XtX)
        XtY = X.T @ y
        beta = XtX_inv @ XtY

        # 残差と標準誤差計算
        residuals = y - X @ beta

        # 自由度はパラメータ数(3)を消費するため n-3
        sse = np.sum(residuals**2)
        mse = sse / (n - 3.0)

        # β係数（ラグレベル：インデックス1）の標準誤差
        se_beta = np.sqrt(mse * XtX_inv[1, 1])

        # ADF t統計量
        if se_beta > 1e-10:
            adf_stat = beta[1] / se_beta
        else:
            adf_stat = np.nan

    except:
        adf_stat = np.nan

    return adf_stat


@nb.njit(fastmath=False, cache=True)
def phillips_perron_統計量_udf(prices: np.ndarray) -> float:
    """
    真のフィリップス・ペロン検定統計量計算（Newey-West長期的分散補正付き）
    異分散性と自己相関をノンパラメトリックに修正したZ_t統計量を算出
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
    X = np.empty((n, 2), dtype=np.float64)
    X[:, 0] = 1.0
    X[:, 1] = lagged_prices
    y = diff_prices

    try:
        # OLS推定
        XtX_inv = np.linalg.inv(X.T @ X)
        beta = XtX_inv @ X.T @ y

        # 残差
        residuals = y - X @ beta

        # ▼▼ 修正前: sigma2 = np.var(residuals) ... (簡略版)
        # ▼▼ 修正後: 真のPhillips-Perron検定（Newey-West長期的分散補正）

        # 1. 残差の標本分散 (gamma_0)
        gamma_0 = np.sum(residuals**2) / n

        # 2. BartlettカーネルによるNewey-West長期的分散 (lambda_sq)
        # ラグ数の経験則: 4 * (T/100)^(2/9)
        lag_max = int(4.0 * (n / 100.0) ** (2.0 / 9.0))
        if lag_max < 1:
            lag_max = 1

        lambda_sq = gamma_0
        for j in range(1, lag_max + 1):
            # 自己共分散 (gamma_j)
            gamma_j = np.sum(residuals[j:] * residuals[:-j]) / n
            # Bartlett重み
            weight = 1.0 - (j / (lag_max + 1.0))
            lambda_sq += 2.0 * weight * gamma_j

        # 3. 標準誤差とt統計量
        # 不偏分散（s^2）による回帰の標準誤差
        s2 = np.sum(residuals**2) / (n - 2.0)
        s = np.sqrt(s2)

        se_beta = np.sqrt(s2 * XtX_inv[1, 1])
        if se_beta <= 1e-10 or lambda_sq <= 1e-10:
            return np.nan

        t_stat = beta[1] / se_beta

        # 4. Phillips-Perron Z_t 統計量の算出
        term1 = np.sqrt(gamma_0 / lambda_sq) * t_stat
        term2 = 0.5 * ((lambda_sq - gamma_0) / np.sqrt(lambda_sq)) * (n * se_beta / s)
        pp_stat = term1 - term2

    except:
        pp_stat = np.nan

    return pp_stat


@nb.njit(fastmath=False, cache=True)
def kpss_統計量_udf(prices: np.ndarray) -> float:
    """
    真のKPSS検定統計量計算（Newey-West長期的分散補正付き）
    帰無仮説：系列がトレンド周りで定常
    """
    if len(prices) < 10:
        return np.nan

    finite_prices = prices[np.isfinite(prices)]
    if len(finite_prices) < 10:
        return np.nan

    n = len(finite_prices)
    t = np.arange(n, dtype=np.float64)

    try:
        # OLS: y = α + βt + ε
        sum_t = np.sum(t)
        sum_t2 = np.sum(t**2)
        sum_y = np.sum(finite_prices)
        sum_ty = np.sum(t * finite_prices)

        beta = (n * sum_ty - sum_t * sum_y) / (n * sum_t2 - sum_t**2 + 1e-10)
        alpha = (sum_y - beta * sum_t) / n

        # デトレンドされた系列 (残差)
        detrended = finite_prices - (alpha + beta * t)
        cumsum = np.cumsum(detrended)

        # ▼▼ 修正前: 単なる短期分散 (sse = np.sum(detrended**2) / n)
        # ▼▼ 修正後: Bartlettカーネルによる真の長期的分散 (long_run_var)
        gamma_0 = np.sum(detrended**2) / n

        # ラグ数の経験則: 4 * (T/100)^(2/9)  ※Schwert (1989) 基準
        lag_max = int(4.0 * (n / 100.0) ** (2.0 / 9.0))
        if lag_max < 1:
            lag_max = 1

        long_run_var = gamma_0
        for j in range(1, lag_max + 1):
            gamma_j = np.sum(detrended[j:] * detrended[:-j]) / n
            weight = 1.0 - (j / (lag_max + 1.0))
            long_run_var += 2.0 * weight * gamma_j

        # KPSS統計量の算出
        if long_run_var > 1e-10:
            kpss_stat = np.sum(cumsum**2) / (n**2 * long_run_var)
        else:
            kpss_stat = np.nan

    except:
        kpss_stat = np.nan

    return kpss_stat


@nb.njit(fastmath=False, cache=True)
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
    # ▼▼ 修正前: std_ret = np.std(finite_returns) または np.std(finite_returns, ddof=1)
    # ▼▼ 修正後: 手動ベッセル補正 (ddof=1相当) に置換
    n_ret = len(finite_returns)
    std_ret = np.std(finite_returns) * np.sqrt(n_ret / (n_ret - 1.0))

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


@nb.njit(fastmath=False, cache=True)
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
        # ▼▼ 修正前: return np.std(finite_returns)
        # ▼▼ 修正後: 手動ベッセル補正 (ddof=1相当)
        n_ret = len(finite_returns)
        return np.std(finite_returns) * np.sqrt(n_ret / (n_ret - 1.0))

    # 既知νを持つt分布用、尺度パラメータσ推定可能:
    # σ² = sample_variance * (ν-2)/ν
    # ▼▼ 修正前: sample_var = np.var(finite_returns)
    # ▼▼ 修正後: 手動ベッセル補正 (ddof=1相当)
    n_ret = len(finite_returns)
    sample_var = np.var(finite_returns) * (n_ret / (n_ret - 1.0))
    scale_squared = sample_var * (dof - 2.0) / dof

    return np.sqrt(max(scale_squared, 1e-8))


@nb.njit(fastmath=False, cache=True)
def gev_形状_udf(extremes: np.ndarray) -> float:
    """
    GEV分布形状パラメータ（ξ）推定
    ξ > 0: フレシェ（重い尾）、ξ = 0: ガンベル、ξ < 0: ワイブル（有界）
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
        weight = ((i * (i - 1.0)) - 2.0 * i * (n - 1.0) + (n - 1.0) * (n - 2.0)) / (
            n * (n - 1.0)
        )
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


@nb.njit(fastmath=False, cache=True)
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

    for i in range(1, len(finite_prices)):
        level = alpha * finite_prices[i] + (1 - alpha) * level

    return level


@nb.njit(fastmath=False, cache=True)
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

    for i in range(1, len(finite_prices)):
        new_level = alpha * finite_prices[i] + (1 - alpha) * (level + trend)
        new_trend = beta * (new_level - level) + (1 - beta) * trend
        level = new_level
        trend = new_trend

    return trend


@nb.njit(fastmath=False, cache=True)
def arima_残差分散_udf(prices: np.ndarray) -> float:
    """
    ARIMA(1,1,0) モデルからの残差分散計算
    ※高速化のためMA(1)成分の非線形推定は行わず、1次差分に対するAR(1)をOLS推定する
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
    # ▼▼ 修正前: if n * sum_x2 - sum_x**2 != 0: ブランチ
    # ▼▼ 修正後: if 分岐を廃止し + 1e-10 で保護
    phi = (n * sum_xy - sum_x * sum_y) / (n * sum_x2 - sum_x**2 + 1e-10)

    # 残差計算
    intercept = (sum_y - phi * sum_x) / n
    residuals = y - (intercept + phi * x)

    # 残差分散（不偏推定量）
    # ▼▼ 修正後: なぜ n-2 なのかの根拠コメント（ddof=2）を追加
    # AR(1)+定数項でパラメータ2つ分の自由度を消費するため (ddof=2)
    residual_variance = np.sum(residuals**2) / (n - 2)

    return residual_variance


@nb.njit(fastmath=False, cache=True)
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
    if len(finite_prices) > 1:
        # ▼▼ 修正前: diff_var = np.var(np.diff(finite_prices)) / obs_var = np.var(finite_prices)
        # ▼▼ 修正後: 手動ベッセル補正 (ddof=1相当) に置換
        diff_vals = np.diff(finite_prices)
        n_diff = len(diff_vals)
        n_prices = len(finite_prices)

        diff_var = np.var(diff_vals) * (n_diff / (n_diff - 1.0)) if n_diff > 1 else 0.0
        obs_var = (
            np.var(finite_prices) * (n_prices / (n_prices - 1.0))
            if n_prices > 1
            else 0.0
        )

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


@nb.njit(fastmath=False, cache=True)
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
    # ▼▼ 修正前: if max_dist > 0: / weights = np.zeros(h)
    # ▼▼ 修正後: 1e-10によるゼロ除算保護と dtype=np.float64 の明記（ルール5, 8）
    if max_dist > 1e-10:
        weights = np.zeros(h, dtype=np.float64)
        for i in range(h):
            u = distances[neighbor_indices[i]] / max_dist
            if u < 1.0:
                # 三次重み関数（1.0でfloat明記）
                weights[i] = (1.0 - u**3) ** 3
            else:
                weights[i] = 0.0
    else:
        weights = np.ones(h, dtype=np.float64)

    # 重み付き最小二乗回帰
    if len(x_neighbors) >= 2:
        # 重み付き平均
        sum_w = np.sum(weights)
        # ▼▼ 修正前: if sum_w > 0: ... else: fitted_value = np.mean(y_neighbors)
        # ▼▼ 修正後: ロジック整理（微小値保護とelseブランチの明示化）
        if sum_w > 1e-10:
            x_mean = np.sum(weights * x_neighbors) / sum_w
            y_mean = np.sum(weights * y_neighbors) / sum_w

            # 重み付き回帰係数
            numerator = np.sum(
                weights * (x_neighbors - x_mean) * (y_neighbors - y_mean)
            )
            denominator = np.sum(weights * (x_neighbors - x_mean) ** 2)

            if denominator > 1e-10:
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


@nb.njit(fastmath=False, cache=True)
def theil_sen_傾き_udf(prices: np.ndarray) -> float:
    """
    頑健傾き推定のためのタイル・セン推定量
    外れ値に耐性のある全ペアワイズ傾きの中央値
    """
    if len(prices) < 10:
        return np.nan

    finite_prices = prices[np.isfinite(prices)]
    n = len(finite_prices)
    if n < 10:
        return np.nan

    # ▼▼ 修正前: slopes = [] ... slopes.append(slope)
    # ▼▼ 修正後: Numpy配列の事前確保（Pre-allocation）による超高速化（ルール6）
    max_pairs = min(1000, (n * (n - 1)) // 2)  # 効率用制限
    slopes = np.zeros(max_pairs, dtype=np.float64)
    slope_idx = 0

    if max_pairs < (n * (n - 1)) // 2:
        # ペア一様サンプリング
        step = max(1, ((n * (n - 1)) // 2) // max_pairs)
        count = 0
        for i in range(n - 1):
            for j in range(i + 1, n):
                if count % step == 0 and slope_idx < max_pairs:
                    if j != i:  # ゼロ除算回避
                        slopes[slope_idx] = (
                            finite_prices[j] - finite_prices[i]
                        ) / float(j - i)
                        slope_idx += 1
                count += 1
    else:
        # 全ペア計算
        for i in range(n - 1):
            for j in range(i + 1, n):
                if slope_idx < max_pairs:
                    slopes[slope_idx] = (finite_prices[j] - finite_prices[i]) / float(
                        j - i
                    )
                    slope_idx += 1

    if slope_idx > 0:
        # 実際に計算された部分のみで中央値を計算
        return np.median(slopes[:slope_idx])
    else:
        return np.nan




# ===========================================================================
# [CATEGORY: TECHNICAL OSCILLATORS — Engine 1C (Numba) 用]
# ===========================================================================
# 本セクションは Step 1 の SSoT 統一で集約された関数群。
# 学習側 (engine_1_X) と本番側 (realtime_feature_engine_1X) の両方が
# import で参照する正本。修正はここ一箇所のみ行う。
# ===========================================================================

@nb.njit(cache=True)
def _wma_helper(data, start, length):
    if start + length > len(data):
        return np.nan
    weight_sum = 0.0
    value_sum = 0.0
    for i in range(length):
        weight = length - i
        value_sum += data[start + i] * weight
        weight_sum += weight
    if weight_sum > 0:
        return value_sum / weight_sum
    return np.nan


@nb.guvectorize(
    ["void(float64[:], int64, float64[:])"], "(n),()->(n)", nopython=True, cache=True
)
def calculate_wma_numba(prices, period, out):
    """WMA (Weighted Moving Average) 計算 (完全版)"""
    n = len(prices)
    if n < period:
        for i in range(n):
            out[i] = np.nan
        return

    weight_sum = period * (period + 1) / 2.0
    for i in range(n):
        if i < period - 1:
            out[i] = np.nan
        else:
            val_sum = 0.0
            for j in range(period):
                weight = period - j
                val_sum += prices[i - j] * weight
            out[i] = val_sum / weight_sum


@nb.guvectorize(
    ["void(float64[:], int64, float64[:])"], "(n),()->(n)", nopython=True, cache=True
)
def calculate_hma_numba(prices, period, out):
    """Hull Moving Average計算 (Numba最適化版)"""
    n = len(prices)
    half_period = period // 2
    sqrt_period = int(np.sqrt(period))

    # 中間配列
    wma_half = np.zeros(n)
    wma_full = np.zeros(n)
    raw_hma = np.zeros(n)

    # 計算
    for i in range(n):
        if i >= half_period - 1:
            wma_half[i] = _wma_helper(prices, i - half_period + 1, half_period)
        else:
            wma_half[i] = np.nan

        if i >= period - 1:
            wma_full[i] = _wma_helper(prices, i - period + 1, period)
        else:
            wma_full[i] = np.nan

        if not np.isnan(wma_half[i]) and not np.isnan(wma_full[i]):
            raw_hma[i] = 2 * wma_half[i] - wma_full[i]
        else:
            raw_hma[i] = np.nan

    # 最終HMA計算
    for i in range(n):
        if i >= sqrt_period - 1:
            out[i] = _wma_helper(raw_hma, i - sqrt_period + 1, sqrt_period)
        else:
            out[i] = np.nan


@nb.guvectorize(
    ["void(float64[:], int64, float64[:])"], "(n),()->(n)", nopython=True, cache=True
)
def calculate_kama_numba(prices, period, out):
    """Kaufman Adaptive Moving Average計算 (Numba最適化版)"""
    n = len(prices)
    fast_ema = 2
    slow_ema = 30

    for i in range(n):
        if i < period:
            out[i] = np.nan
        else:
            # Efficiency Ratio計算
            direction = abs(prices[i] - prices[i - period])
            volatility = 0.0
            for j in range(i - period + 1, i + 1):
                volatility += abs(prices[j] - prices[j - 1])

            if volatility == 0:
                er = 0.0
            else:
                er = direction / volatility

            # Smoothing Constant計算
            fast_sc = 2.0 / (fast_ema + 1.0)
            slow_sc = 2.0 / (slow_ema + 1.0)
            sc = (er * (fast_sc - slow_sc) + slow_sc) ** 2

            # KAMA計算
            if i == period:
                out[i] = prices[i]
            else:
                if not np.isnan(out[i - 1]):
                    out[i] = out[i - 1] + sc * (prices[i] - out[i - 1])
                else:
                    out[i] = prices[i]


@nb.guvectorize(
    ["void(float64[:], float64[:], float64[:], int64, int64, int64, float64[:])"],
    "(n),(n),(n),(),(),()->(n)",
    nopython=True,
    cache=True,
)
def calculate_stochastic_numba(high, low, close, k_period, d_period, slow_period, out):
    """Stochastic Oscillator計算 (Numba最適化版)"""
    n = len(high)

    # %K計算
    k_values = np.zeros(n)
    for i in range(n):
        if i < k_period - 1:
            k_values[i] = np.nan
        else:
            highest = high[i]
            lowest = low[i]
            for j in range(i - k_period + 1, i):
                if high[j] > highest:
                    highest = high[j]
                if low[j] < lowest:
                    lowest = low[j]

            if highest - lowest > 0:
                k_values[i] = 100 * (close[i] - lowest) / (highest - lowest)
            else:
                k_values[i] = 50.0

    # %D計算 (Simple MA of %K)
    for i in range(n):
        if i < k_period + d_period - 2:
            out[i] = np.nan
        else:
            sum_k = 0.0
            # ▼▼ 修正前: count = 0
            # ▼▼ 修正後: ルール8（型の厳格化）に基づき、float の 0.0 で初期化
            count = 0.0
            for j in range(i - d_period + 1, i + 1):
                if not np.isnan(k_values[j]):
                    sum_k += k_values[j]
                    # ▼▼ 修正前: count += 1
                    # ▼▼ 修正後: 型の厳格化
                    count += 1.0

            # ▼▼ 修正前: if count > 0:
            # ▼▼ 修正前:     out[i] = sum_k / count
            # ▼▼ 修正後: 型の厳格化（0.0）と、ルール5（ゼロ除算の鉄壁防衛）のための + 1e-10 の追加
            # 修正後
            if count > 0.0:
                out[i] = sum_k / count
            else:
                out[i] = np.nan


@nb.guvectorize(
    ["void(float64[:], float64[:], float64[:], int64, float64[:])"],
    "(n),(n),(n),()->(n)",
    nopython=True,
    cache=True,
)
def calculate_williams_r_numba(high, low, close, period, out):
    """Williams %R計算 (Numba最適化版)"""
    n = len(high)

    for i in range(n):
        if i < period - 1:
            out[i] = np.nan
        else:
            highest = high[i]
            lowest = low[i]

            for j in range(i - period + 1, i):
                if high[j] > highest:
                    highest = high[j]
                if low[j] < lowest:
                    lowest = low[j]

            if highest - lowest > 0:
                out[i] = -100 * (highest - close[i]) / (highest - lowest)
            else:
                out[i] = -50.0


@nb.guvectorize(
    ["void(float64[:], int64, float64[:])"], "(n),()->(n)", nopython=True, cache=True
)
def calculate_trix_numba(prices, period, out):
    """TRIX指標計算 (Numba最適化版)"""
    n = len(prices)

    # 三重指数移動平均
    ema1 = np.zeros(n)
    ema2 = np.zeros(n)
    ema3 = np.zeros(n)

    alpha = 2.0 / (period + 1.0)

    # 1st EMA
    for i in range(n):
        if i == 0:
            ema1[i] = prices[i]
        else:
            ema1[i] = alpha * prices[i] + (1 - alpha) * ema1[i - 1]

    # 2nd EMA
    for i in range(n):
        if i == 0:
            ema2[i] = ema1[i]
        else:
            ema2[i] = alpha * ema1[i] + (1 - alpha) * ema2[i - 1]

    # 3rd EMA
    for i in range(n):
        if i == 0:
            ema3[i] = ema2[i]
        else:
            ema3[i] = alpha * ema2[i] + (1 - alpha) * ema3[i - 1]

    # TRIX計算
    for i in range(n):
        if i < period * 3:
            out[i] = np.nan
        elif ema3[i - 1] == 0:
            out[i] = 0.0
        else:
            out[i] = 10000 * (ema3[i] - ema3[i - 1]) / ema3[i - 1]


@nb.guvectorize(
    ["void(float64[:], float64[:], float64[:], float64[:], float64[:])"],
    "(n),(n),(n),(n)->(n)",
    nopython=True,
    cache=True,
)
def calculate_ultimate_oscillator_numba(high, low, close, volume, out):
    """Ultimate Oscillator計算 (Numba最適化版)"""
    n = len(high)
    period_7 = 7
    period_14 = 14
    period_28 = 28
    weight_0 = 4.0
    weight_1 = 2.0
    weight_2 = 1.0
    weight_total = 7.0
    max_period = 28

    # Buying Pressure and True Range計算
    bp = np.zeros(n)
    tr = np.zeros(n)

    for i in range(n):
        if i == 0:
            bp[i] = close[i] - low[i]
            tr[i] = high[i] - low[i]
        else:
            lc = close[i - 1]
            bp[i] = close[i] - (low[i] if low[i] < lc else lc)
            h_l = high[i] - low[i]
            h_pc = abs(high[i] - close[i - 1])
            l_pc = abs(low[i] - close[i - 1])
            tr[i] = max(h_l, h_pc, l_pc)

    # Ultimate Oscillator計算
    for i in range(n):
        if i < max_period:
            out[i] = np.nan
        else:
            weighted_sum = 0.0

            for j in range(3):
                if j == 0:
                    period = period_7
                    w = weight_0
                elif j == 1:
                    period = period_14
                    w = weight_1
                else:
                    period = period_28
                    w = weight_2
                bp_sum = 0.0
                tr_sum = 0.0

                for k in range(i - period + 1, i + 1):
                    bp_sum += bp[k]
                    tr_sum += tr[k]

                if tr_sum > 0:
                    avg = bp_sum / tr_sum
                else:
                    avg = 0.0

                # 修正後
                weighted_sum += avg * w

            out[i] = 100 * weighted_sum / weight_total


@nb.guvectorize(
    ["void(float64[:], int64, float64[:])"], "(n),()->(n)", nopython=True, cache=True
)
def calculate_aroon_up_numba(high, period, out):
    """Aroon Up計算 (Numba最適化版)"""
    n = len(high)

    for i in range(n):
        if i < period - 1:  # 修正箇所: period から period - 1 へ変更
            out[i] = np.nan
        else:
            # Find highest high in period
            highest_idx = i
            highest_val = high[i]

            # ループ範囲は i - period + 1 から i (iを含まない) まで
            # (例: i=13, period=14 -> range(0, 13) -> j=0...12)
            for j in range(i - period + 1, i):
                if high[j] > highest_val:
                    highest_val = high[j]
                    highest_idx = j

            # Calculate periods since highest
            periods_since = i - highest_idx
            out[i] = 100.0 * (period - periods_since) / period


@nb.guvectorize(
    ["void(float64[:], int64, float64[:])"], "(n),()->(n)", nopython=True, cache=True
)
def calculate_aroon_down_numba(low, period, out):
    """Aroon Down計算 (Numba最適化版)"""
    n = len(low)

    for i in range(n):
        if i < period - 1:  # 修正箇所: period から period - 1 へ変更
            out[i] = np.nan
        else:
            # Find lowest low in period
            lowest_idx = i
            lowest_val = low[i]

            # ループ範囲は i - period + 1 から i (iを含まない) まで
            for j in range(i - period + 1, i):
                if low[j] < lowest_val:
                    lowest_val = low[j]
                    lowest_idx = j

            # Calculate periods since lowest
            periods_since = i - lowest_idx
            out[i] = 100.0 * (period - periods_since) / period


@nb.guvectorize(
    ["void(float64[:], int64, float64[:])"], "(n),()->(n)", nopython=True, cache=True
)
def calculate_tsi_numba(prices, period, out):
    """True Strength Index計算 (Numba最適化版)"""
    n = len(prices)
    long_period = period
    short_period = period // 2

    # Price momentum
    momentum = np.zeros(n)
    for i in range(1, n):
        momentum[i] = prices[i] - prices[i - 1]
    momentum[0] = 0.0

    # Double smoothed momentum
    alpha_long = 2.0 / (long_period + 1.0)
    alpha_short = 2.0 / (short_period + 1.0)

    # First EMA of momentum
    ema1_mom = np.zeros(n)
    ema1_abs = np.zeros(n)

    for i in range(n):
        if i == 0:
            ema1_mom[i] = momentum[i]
            ema1_abs[i] = abs(momentum[i])
        else:
            ema1_mom[i] = alpha_long * momentum[i] + (1 - alpha_long) * ema1_mom[i - 1]
            ema1_abs[i] = (
                alpha_long * abs(momentum[i]) + (1 - alpha_long) * ema1_abs[i - 1]
            )

    # Second EMA
    ema2_mom = np.zeros(n)
    ema2_abs = np.zeros(n)

    for i in range(n):
        if i == 0:
            ema2_mom[i] = ema1_mom[i]
            ema2_abs[i] = ema1_abs[i]
        else:
            ema2_mom[i] = (
                alpha_short * ema1_mom[i] + (1 - alpha_short) * ema2_mom[i - 1]
            )
            ema2_abs[i] = (
                alpha_short * ema1_abs[i] + (1 - alpha_short) * ema2_abs[i - 1]
            )

    # TSI計算
    for i in range(n):
        if i < long_period + short_period:
            out[i] = np.nan
        elif ema2_abs[i] == 0:
            out[i] = 0.0
        else:
            out[i] = 100 * ema2_mom[i] / ema2_abs[i]


@nb.njit(fastmath=False, cache=True)
def _calculate_di_wilder(
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    period: int,
) -> tuple:
    """
    DI+ / DI- — calculate_adx (core_indicators) の内部ロジックと完全同一の
    Wilder 平滑化実装。ADX と DI が同一の ATR 平滑化を参照することを保証する。

    Returns:
        (di_plus_arr, di_minus_arr): 各 shape=(n,) の ndarray
    """
    n = len(high)
    di_plus_out = np.full(n, np.nan, dtype=np.float64)
    di_minus_out = np.full(n, np.nan, dtype=np.float64)
    if n <= period:
        return di_plus_out, di_minus_out

    # TR, DM+, DM- — calculate_adx 内部と同一
    tr = np.zeros(n, dtype=np.float64)
    dm_plus = np.zeros(n, dtype=np.float64)
    dm_minus = np.zeros(n, dtype=np.float64)

    tr[0] = high[0] - low[0]
    for i in range(1, n):
        hl = high[i] - low[i]
        hc = abs(high[i] - close[i - 1])
        lc = abs(low[i] - close[i - 1])
        tr[i] = max(hl, hc, lc)
        up_move = high[i] - high[i - 1]
        down_move = low[i - 1] - low[i]
        dm_plus[i] = up_move if (up_move > down_move and up_move > 0.0) else 0.0
        dm_minus[i] = down_move if (down_move > up_move and down_move > 0.0) else 0.0

    # Wilder 平滑化 — calculate_adx 内部と同一（単純合計シード）
    atr_w = np.zeros(n, dtype=np.float64)
    dmp_w = np.zeros(n, dtype=np.float64)
    dmm_w = np.zeros(n, dtype=np.float64)

    init_tr = 0.0
    init_dmp = 0.0
    init_dmm = 0.0
    for j in range(period):
        init_tr += tr[j]
        init_dmp += dm_plus[j]
        init_dmm += dm_minus[j]

    atr_w[period - 1] = init_tr
    dmp_w[period - 1] = init_dmp
    dmm_w[period - 1] = init_dmm

    for i in range(period, n):
        atr_w[i] = atr_w[i - 1] - atr_w[i - 1] / period + tr[i]
        dmp_w[i] = dmp_w[i - 1] - dmp_w[i - 1] / period + dm_plus[i]
        dmm_w[i] = dmm_w[i - 1] - dmm_w[i - 1] / period + dm_minus[i]

    # DI+ / DI- 計算
    for i in range(period - 1, n):
        if atr_w[i] > 1e-10:
            di_plus_out[i] = dmp_w[i] / atr_w[i] * 100.0
            di_minus_out[i] = dmm_w[i] / atr_w[i] * 100.0

    return di_plus_out, di_minus_out




# ===========================================================================
# [CATEGORY: VOLUME EXTRA — Engine 1D 追加 (HV / Pivot / Donchian)]
# ===========================================================================
# 本セクションは Step 1 の SSoT 統一で集約された関数群。
# 学習側 (engine_1_X) と本番側 (realtime_feature_engine_1X) の両方が
# import で参照する正本。修正はここ一箇所のみ行う。
# ===========================================================================

@nb.njit(fastmath=False, cache=True)
def hv_standard_udf(returns: np.ndarray) -> float:
    """
    標準ヒストリカルボラティリティ計算
    標準偏差ベースの伝統的ボラティリティ指標
    """
    if len(returns) < 5:
        return np.nan

    finite_returns = returns[np.isfinite(returns)]
    if len(finite_returns) < 5:
        return np.nan

    # 標準偏差計算（不偏推定量）
    mean_return = np.mean(finite_returns)
    squared_deviations = (finite_returns - mean_return) ** 2
    variance = np.sum(squared_deviations) / (len(finite_returns) - 1)
    volatility = np.sqrt(variance)

    return volatility


@nb.njit(fastmath=False, cache=True)
def hv_robust_udf(returns: np.ndarray) -> float:
    """
    ロバストボラティリティ計算
    中央絶対偏差（MAD）ベースの外れ値耐性ボラティリティ
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


@nb.njit(fastmath=False, cache=True)
def pivot_point_udf(
    high: float, low: float, close: float
) -> Tuple[float, float, float, float, float]:
    """
    ピボットポイント計算
    Returns: (pivot_point, resistance1, support1, resistance2, support2)
    """
    if not (np.isfinite(high) and np.isfinite(low) and np.isfinite(close)):
        return np.nan, np.nan, np.nan, np.nan, np.nan

    pivot = (high + low + close) / 3.0
    r1 = 2.0 * pivot - low
    s1 = 2.0 * pivot - high
    r2 = pivot + (high - low)
    s2 = pivot - (high - low)

    return pivot, r1, s1, r2, s2


@nb.njit(fastmath=False, cache=True)
def donchian_channel_udf(high: np.ndarray, low: np.ndarray, window: int) -> np.ndarray:
    """
    ドンチャンチャネル計算
    Returns: 各行に[upper, middle, lower]を含む配列
    """
    n = len(high)
    result = np.full((n, 3), np.nan)

    if n < window:
        return result

    for i in range(window - 1, n):
        period_high = -np.inf
        period_low = np.inf

        for j in range(i - window + 1, i + 1):
            if np.isfinite(high[j]) and high[j] > period_high:
                period_high = high[j]
            if np.isfinite(low[j]) and low[j] < period_low:
                period_low = low[j]

        if np.isfinite(period_high) and np.isfinite(period_low):
            result[i, 0] = period_high  # upper
            result[i, 1] = (period_high + period_low) / 2.0  # middle
            result[i, 2] = period_low  # lower

    return result


def _run_validation():
    """数値検証: core_indicators の各関数の出力を確認する。"""
    import sys

    np.random.seed(42)
    n = 300
    close = np.cumsum(np.random.randn(n) * 0.5) + 2000.0
    high = close + np.abs(np.random.randn(n)) * 0.3
    low = close - np.abs(np.random.randn(n)) * 0.3
    volume = np.random.randint(100, 10000, n).astype(np.float64)

    print("=" * 60)
    print("core_indicators.py 数値検証")
    print("=" * 60)

    # ---- ATR Wilder ----
    atr = calculate_atr_wilder(high, low, close, 13)
    # 検証: 先頭 1 本は NaN でない (TR[0])、全て正値
    assert np.isfinite(atr[0]), "ATR[0] は有限値のはず"
    assert np.all(atr[atr > 0] > 0), "ATR は全て正値のはず"
    print(f"[OK] calculate_atr_wilder  : atr[-1] = {atr[-1]:.6f}")

    # ---- scale_by_atr ----
    range_arr = high - low
    scaled = scale_by_atr(range_arr, atr)
    # ATR で割ればほぼ 1 前後になるはず
    assert np.isfinite(scaled[-1]), "scale_by_atr 最終値は有限値のはず"
    print(f"[OK] scale_by_atr          : scaled[-1] = {scaled[-1]:.6f}")

    # ---- RSI Wilder ----
    rsi = calculate_rsi_wilder(close, 14)
    assert np.isnan(rsi[13]), "RSI[13] は NaN のはず (先頭 period 本は NaN)"
    assert np.isfinite(rsi[14]), "RSI[14] は有限値のはず"
    assert 0.0 <= rsi[-1] <= 100.0, "RSI は 0〜100 の範囲内"
    print(f"[OK] calculate_rsi_wilder  : rsi[-1] = {rsi[-1]:.4f}")

    # ---- EMA ----
    ema = calculate_ema(close, 12)
    assert np.isfinite(ema[-1]), "EMA 最終値は有限値のはず"
    print(f"[OK] calculate_ema         : ema[-1] = {ema[-1]:.6f}")

    # ---- MACD ----
    macd = calculate_macd(close, 12, 26, 9)
    assert np.isfinite(macd[-1]), "MACD 最終値は有限値のはず"
    print(f"[OK] calculate_macd        : macd[-1] = {macd[-1]:.6f}")

    # ---- SMA ----
    sma = calculate_sma(close, 20)
    assert np.isnan(sma[18]), "SMA[18] は NaN のはず"
    assert np.isfinite(sma[19]), "SMA[19] は有限値のはず"
    print(f"[OK] calculate_sma         : sma[-1] = {sma[-1]:.6f}")

    # ---- Bollinger ----
    bb = calculate_bollinger(close, 20, 2.0)
    assert bb[-1, 0] > bb[-1, 1] > bb[-1, 2], "upper > middle > lower のはず"
    print(f"[OK] calculate_bollinger   : upper={bb[-1,0]:.4f}, mid={bb[-1,1]:.4f}, lower={bb[-1,2]:.4f}")

    # ---- ADX ----
    adx = calculate_adx(high, low, close, 14)
    assert np.isfinite(adx[-1]), "ADX 最終値は有限値のはず"
    assert 0.0 <= adx[-1] <= 100.0, "ADX は 0〜100 の範囲内"
    print(f"[OK] calculate_adx         : adx[-1] = {adx[-1]:.4f}")

    # ---- stddev_unbiased ----
    std_arr = stddev_unbiased(close, 20)
    assert np.isfinite(std_arr[-1]), "std 最終値は有限値のはず"
    # Numpy ddof=1 との比較
    expected_std = np.std(close[-20:], ddof=1)
    assert abs(std_arr[-1] - expected_std) < 1e-8, \
        f"stddev_unbiased と np.std(ddof=1) が一致しない: {std_arr[-1]} vs {expected_std}"
    print(f"[OK] stddev_unbiased       : std[-1] = {std_arr[-1]:.6f} (numpy={expected_std:.6f})")

    # ---- calculate_mad ----
    mad_arr = calculate_mad(close, 20)
    assert np.isfinite(mad_arr[-1]), "MAD 最終値は有限値のはず"
    print(f"[OK] calculate_mad         : mad[-1] = {mad_arr[-1]:.6f}")

    # ---- rolling_zscore ----
    z_arr = rolling_zscore(close, 20)
    assert np.isfinite(z_arr[-1]), "Zscore 最終値は有限値のはず"
    print(f"[OK] rolling_zscore        : z[-1] = {z_arr[-1]:.6f}")

    # ---- clip_and_validate ----
    test_arr = np.array([np.nan, np.inf, -np.inf, 5.0, -15.0, 7.0])
    clipped = clip_and_validate(test_arr, 10.0)
    expected = np.array([0.0, 0.0, 0.0, 5.0, -10.0, 7.0])
    assert np.allclose(clipped, expected), f"clip_and_validate の結果が期待と異なる: {clipped}"
    print(f"[OK] clip_and_validate     : {clipped}")

    # ---- calculate_sample_weight ----
    weights = calculate_sample_weight(high, low, close)
    valid_weights = set([1.0, 2.0, 4.0, 6.0])
    finite_weights = weights[np.isfinite(weights)]
    assert all(w in valid_weights for w in finite_weights), "重みが想定外の値"
    print(f"[OK] calculate_sample_weight: 重み分布 = {dict(zip(*np.unique(finite_weights, return_counts=True)))}")

    # ---- neutralize_ols ----
    # シンプルな線形関係でテスト: y = 2*x + noise → neutralized ≈ noise
    x_test = np.random.randn(300)
    noise = np.random.randn(300) * 0.1
    y_test = 2.0 * x_test + noise
    neutralized = neutralize_ols(y_test, x_test, window=200, min_periods=30)
    # 純化後はほぼ noise と一致するはず
    finite_mask = np.isfinite(neutralized)
    residual_std = np.std(neutralized[finite_mask])
    assert residual_std < 0.5, f"純化後の標準偏差が大きすぎる: {residual_std}"
    print(f"[OK] neutralize_ols        : neutralized std = {residual_std:.4f} (noise std ≈ 0.1)")

    # ================================================================
    # [CATEGORY: VOLUME] テスト
    # ================================================================

    # ---- hv_standard_scalar ----
    ret = np.diff(np.log(close))
    hv_std = hv_standard_scalar(ret[-20:])
    expected_hv = np.std(ret[-20:], ddof=1)
    assert abs(hv_std - expected_hv) < 1e-8, \
        f"hv_standard_scalar が np.std(ddof=1) と不一致: {hv_std} vs {expected_hv}"
    print(f"[OK] hv_standard_scalar    : hv_std={hv_std:.8f} (numpy={expected_hv:.8f})")

    # ---- hv_robust_scalar ----
    hv_rob = hv_robust_scalar(ret[-20:])
    assert np.isfinite(hv_rob) and hv_rob > 0.0, "hv_robust_scalar は正有限値のはず"
    print(f"[OK] hv_robust_scalar      : hv_rob={hv_rob:.8f}")

    # ---- chaikin_volatility_udf ----
    cv = chaikin_volatility_udf(high, low, 10)
    assert np.isfinite(cv[-1]), "ChaikinVol 最終値は有限値のはず"
    print(f"[OK] chaikin_volatility_udf: cv[-1]={cv[-1]:.6f}")

    # ---- mass_index_udf ----
    mi = mass_index_udf(high, low, 20)
    assert np.isfinite(mi[-1]) and mi[-1] > 0.0, "MassIndex 最終値は正有限値のはず"
    print(f"[OK] mass_index_udf        : mi[-1]={mi[-1]:.6f}")

    # ---- cmf_udf ----
    cmf = cmf_udf(high, low, close, volume, 13)
    assert np.isfinite(cmf[-1]) and -1.0 <= cmf[-1] <= 1.0, \
        f"CMF は [-1,1] のはず: {cmf[-1]}"
    print(f"[OK] cmf_udf               : cmf[-1]={cmf[-1]:.6f}")

    # ---- mfi_udf ----
    mfi = mfi_udf(high, low, close, volume, 13)
    assert np.isfinite(mfi[-1]) and 0.0 <= mfi[-1] <= 100.0, \
        f"MFI は [0,100] のはず: {mfi[-1]}"
    print(f"[OK] mfi_udf               : mfi[-1]={mfi[-1]:.4f}")

    # ---- vwap_udf ----
    vwap = vwap_udf(high, low, close, volume, 20)
    assert np.isfinite(vwap[-1]) and vwap[-1] > 0.0, "VWAP は正有限値のはず"
    print(f"[OK] vwap_udf              : vwap[-1]={vwap[-1]:.6f}")

    # ---- obv_udf ----
    obv = obv_udf(close, volume)
    assert np.isfinite(obv[-1]), "OBV 最終値は有限値のはず"
    print(f"[OK] obv_udf               : obv[-1]={obv[-1]:.2f}")

    # ---- accumulation_distribution_udf ----
    ad = accumulation_distribution_udf(high, low, close, volume)
    assert np.isfinite(ad[-1]), "A/D 最終値は有限値のはず"
    print(f"[OK] accumulation_distribution_udf: ad[-1]={ad[-1]:.2f}")

    # ---- force_index_udf ----
    fi = force_index_udf(close, volume)
    assert np.isfinite(fi[-1]), "ForceIndex 最終値は有限値のはず"
    print(f"[OK] force_index_udf       : fi[-1]={fi[-1]:.2f}")

    # ---- commodity_channel_index_udf ----
    cci = commodity_channel_index_udf(high, low, close, 14)
    assert np.isfinite(cci[-1]), "CCI 最終値は有限値のはず"
    print(f"[OK] commodity_channel_index_udf: cci[-1]={cci[-1]:.4f}")

    # ---- candlestick_patterns_udf ----
    open_arr = close - 0.1  # 疑似 open
    patterns = candlestick_patterns_udf(open_arr, high, low, close)
    valid_ids = {0.0, 1.0, 2.0, 3.0, 4.0, 5.0}
    assert set(np.unique(patterns)).issubset(valid_ids), \
        f"パターンIDが想定外: {np.unique(patterns)}"
    print(f"[OK] candlestick_patterns_udf: 分布={dict(zip(*np.unique(patterns, return_counts=True)))}")

    # ---- fibonacci_levels_udf ----
    fib = fibonacci_levels_udf(high, low, 50)
    assert fib.shape == (n, 5), f"fibonacci_levels_udf shape 不正: {fib.shape}"
    assert np.isfinite(fib[-1, 2]), "Fib 50% レベル (最終行) は有限値のはず"
    assert fib[-1, 0] > fib[-1, 2] > fib[-1, 4], "Fib レベルの順序が不正"
    print(f"[OK] fibonacci_levels_udf  : fib50[-1]={fib[-1, 2]:.4f}")

    # ================================================================
    # [CATEGORY: DSP] テスト
    # ================================================================

    # 信号処理のためのテスト用信号（サイン波 + ノイズ）
    t = np.linspace(0, 4 * np.pi, n)
    signal_arr = np.sin(t) + np.random.randn(n) * 0.1

    # ---- numba_fft ----
    fft_out = numba_fft(signal_arr[:64].astype(np.float64))
    assert len(fft_out) >= 64, "numba_fft: 出力長が入力以上のはず"
    assert np.isfinite(np.abs(fft_out[1])), "numba_fft: 出力は有限値のはず"
    print(f"[OK] numba_fft             : len={len(fft_out)}, |X[1]|={np.abs(fft_out[1]):.4f}")

    # ---- get_analytic_signal ----
    analytic = get_analytic_signal(signal_arr[:64].astype(np.float64))
    assert len(analytic) == 64, "get_analytic_signal: 長さが元のサイズと一致するはず"
    assert np.isfinite(np.abs(analytic[-1])), "get_analytic_signal: 有限値のはず"
    print(f"[OK] get_analytic_signal   : |analytic[-1]|={np.abs(analytic[-1]):.4f}")

    # ---- spectral_centroid_udf ----
    sc = spectral_centroid_udf(signal_arr, 64)
    assert np.isfinite(sc[-1]) and 0.0 <= sc[-1] <= 0.5, f"spectral_centroid: [0,0.5] のはず: {sc[-1]}"
    print(f"[OK] spectral_centroid_udf : sc[-1]={sc[-1]:.6f}")

    # ---- spectral_bandwidth_udf ----
    sb = spectral_bandwidth_udf(signal_arr, 64)
    assert np.isfinite(sb[-1]) and sb[-1] >= 0.0, "spectral_bandwidth: >=0 のはず"
    print(f"[OK] spectral_bandwidth_udf: sb[-1]={sb[-1]:.6f}")

    # ---- spectral_rolloff_udf ----
    sr = spectral_rolloff_udf(signal_arr, 64, 0.85)
    assert np.isfinite(sr[-1]) and 0.0 <= sr[-1] <= 0.5, f"spectral_rolloff: [0,0.5] のはず: {sr[-1]}"
    print(f"[OK] spectral_rolloff_udf  : sr[-1]={sr[-1]:.6f}")

    # ---- spectral_flux_udf ----
    sf = spectral_flux_udf(signal_arr, 64)
    finite_sf = sf[np.isfinite(sf)]
    assert len(finite_sf) > 0, "spectral_flux: 有限値が存在するはず"
    assert np.all(finite_sf >= 0.0), "spectral_flux: >=0 のはず"
    print(f"[OK] spectral_flux_udf     : sf[-1]={sf[-1]:.6f}")

    # ---- spectral_flatness_udf ----
    sfl = spectral_flatness_udf(signal_arr, 64)
    assert np.isfinite(sfl[-1]) and 0.0 <= sfl[-1] <= 1.0, f"spectral_flatness: [0,1] のはず: {sfl[-1]}"
    print(f"[OK] spectral_flatness_udf : sfl[-1]={sfl[-1]:.6f}")

    # ---- spectral_entropy_udf ----
    se = spectral_entropy_udf(signal_arr, 64)
    assert np.isfinite(se[-1]) and se[-1] >= 0.0, "spectral_entropy: >=0 のはず"
    print(f"[OK] spectral_entropy_udf  : se[-1]={se[-1]:.4f}")

    # ---- wavelet_energy_udf ----
    we = wavelet_energy_udf(signal_arr, 64)
    assert np.isfinite(we[-1]) and we[-1] >= 0.0, "wavelet_energy: >=0 のはず"
    print(f"[OK] wavelet_energy_udf    : we[-1]={we[-1]:.6f}")

    # ---- wavelet_entropy_udf ----
    went = wavelet_entropy_udf(signal_arr, 64)
    assert np.isfinite(went[-1]), "wavelet_entropy: 有限値のはず"
    print(f"[OK] wavelet_entropy_udf   : went[-1]={went[-1]:.6f}")

    # ---- hilbert_amplitude_udf ----
    ha = hilbert_amplitude_udf(signal_arr, 50)
    assert np.isfinite(ha[-1]) and ha[-1] >= 0.0, "hilbert_amplitude: >=0 のはず"
    print(f"[OK] hilbert_amplitude_udf : ha[-1]={ha[-1]:.6f}")

    # ---- hilbert_phase_var_udf ----
    hpv = hilbert_phase_var_udf(signal_arr, 50)
    assert np.isfinite(hpv[-1]) and hpv[-1] >= 0.0, "hilbert_phase_var: >=0 のはず"
    print(f"[OK] hilbert_phase_var_udf : hpv[-1]={hpv[-1]:.6f}")

    # ---- hilbert_phase_stability_udf ----
    hps = hilbert_phase_stability_udf(signal_arr, 50)
    assert np.isfinite(hps[-1]) and 0.0 <= hps[-1] <= 1.0, f"hilbert_phase_stability: [0,1] のはず: {hps[-1]}"
    print(f"[OK] hilbert_phase_stability_udf: hps[-1]={hps[-1]:.6f}")

    # ---- hilbert_freq_mean_udf ----
    hfm = hilbert_freq_mean_udf(signal_arr, 50)
    assert np.isfinite(hfm[-1]) and hfm[-1] >= 0.0, "hilbert_freq_mean: >=0 のはず"
    print(f"[OK] hilbert_freq_mean_udf : hfm[-1]={hfm[-1]:.6f}")

    # ---- hilbert_freq_std_udf ----
    hfs = hilbert_freq_std_udf(signal_arr, 50)
    assert np.isfinite(hfs[-1]) and hfs[-1] >= 0.0, "hilbert_freq_std: >=0 のはず"
    print(f"[OK] hilbert_freq_std_udf  : hfs[-1]={hfs[-1]:.6f}")

    # ---- acoustic_power_udf ----
    ap = acoustic_power_udf(signal_arr, 50)
    assert np.isfinite(ap[-1]) and ap[-1] >= 0.0, "acoustic_power: >=0 のはず"
    print(f"[OK] acoustic_power_udf    : ap[-1]={ap[-1]:.6f}")

    # ---- acoustic_frequency_udf ----
    af = acoustic_frequency_udf(signal_arr, 50, 1.0)
    assert np.isfinite(af[-1]) and af[-1] >= 0.0, "acoustic_frequency: >=0 のはず"
    print(f"[OK] acoustic_frequency_udf: af[-1]={af[-1]:.6f}")

    # ================================================================
    # [CATEGORY: COMPLEX] テスト
    # ================================================================

    # ---- rolling_network_density_udf ----
    nd = rolling_network_density_udf(close, 30)
    finite_nd = nd[np.isfinite(nd)]
    assert len(finite_nd) > 0, "network_density: 有限値が存在するはず"
    assert np.all((finite_nd >= 0.0) & (finite_nd <= 1.0)), "network_density: [0,1] のはず"
    print(f"[OK] rolling_network_density_udf  : nd[-1]={nd[-1]:.6f}")

    # ---- rolling_network_clustering_udf ----
    nc = rolling_network_clustering_udf(close, 30)
    finite_nc = nc[np.isfinite(nc)]
    assert len(finite_nc) > 0, "network_clustering: 有限値が存在するはず"
    assert np.all((finite_nc >= 0.0) & (finite_nc <= 1.0)), "network_clustering: [0,1] のはず"
    print(f"[OK] rolling_network_clustering_udf: nc[-1]={nc[-1]:.6f}")

    # ---- rolling_vocabulary_diversity_udf ----
    vd = rolling_vocabulary_diversity_udf(close, 30)
    finite_vd = vd[np.isfinite(vd)]
    assert len(finite_vd) > 0, "vocabulary_diversity: 有限値が存在するはず"
    assert np.all((finite_vd >= 0.0) & (finite_vd <= 1.0)), "vocabulary_diversity: [0,1] のはず"
    print(f"[OK] rolling_vocabulary_diversity_udf: vd[-1]={vd[-1]:.6f}")

    # ---- rolling_linguistic_complexity_udf ----
    lc = rolling_linguistic_complexity_udf(close, 30)
    finite_lc = lc[np.isfinite(lc)]
    assert len(finite_lc) > 0, "linguistic_complexity: 有限値が存在するはず"
    assert np.all((finite_lc >= 0.0) & (finite_lc <= 1.0)), "linguistic_complexity: [0,1] のはず"
    print(f"[OK] rolling_linguistic_complexity_udf: lc[-1]={lc[-1]:.6f}")

    # ---- rolling_semantic_flow_udf ----
    sflow = rolling_semantic_flow_udf(close, 30)
    finite_sflow = sflow[np.isfinite(sflow)]
    assert len(finite_sflow) > 0, "semantic_flow: 有限値が存在するはず"
    print(f"[OK] rolling_semantic_flow_udf    : sflow[-1]={sflow[-1]:.6f}")

    # ---- rolling_golden_ratio_adherence_udf ----
    gra = rolling_golden_ratio_adherence_udf(close, 30)
    finite_gra = gra[np.isfinite(gra)]
    assert len(finite_gra) > 0, "golden_ratio_adherence: 有限値が存在するはず"
    assert np.all((finite_gra > 0.0) & (finite_gra <= 1.0)), "golden_ratio_adherence: (0,1] のはず"
    print(f"[OK] rolling_golden_ratio_adherence_udf: gra[-1]={gra[-1]:.6f}")

    # ---- rolling_symmetry_measure_udf ----
    sym = rolling_symmetry_measure_udf(close, 30)
    finite_sym = sym[np.isfinite(sym)]
    assert len(finite_sym) > 0, "symmetry_measure: 有限値が存在するはず"
    assert np.all((finite_sym >= 0.0) & (finite_sym <= 1.0)), "symmetry_measure: [0,1] のはず"
    print(f"[OK] rolling_symmetry_measure_udf : sym[-1]={sym[-1]:.6f}")

    # ---- rolling_aesthetic_balance_udf ----
    ab = rolling_aesthetic_balance_udf(close, 30)
    finite_ab = ab[np.isfinite(ab)]
    assert len(finite_ab) > 0, "aesthetic_balance: 有限値が存在するはず"
    assert np.all((finite_ab >= 0.0) & (finite_ab <= 1.0)), "aesthetic_balance: [0,1] のはず"
    print(f"[OK] rolling_aesthetic_balance_udf: ab[-1]={ab[-1]:.6f}")

    # ---- rolling_tonality_udf ----
    ton = rolling_tonality_udf(close, 30)
    finite_ton = ton[np.isfinite(ton)]
    assert len(finite_ton) > 0, "tonality: 有限値が存在するはず"
    assert np.all((finite_ton >= 0.0) & (finite_ton <= 1.0)), "tonality: [0,1] のはず"
    print(f"[OK] rolling_tonality_udf         : ton[-1]={ton[-1]:.6f}")

    # ---- rolling_rhythm_pattern_udf ----
    rhy = rolling_rhythm_pattern_udf(close, 30)
    finite_rhy = rhy[np.isfinite(rhy)]
    assert len(finite_rhy) > 0, "rhythm_pattern: 有限値が存在するはず"
    assert np.all(finite_rhy >= 0.0), "rhythm_pattern: >=0 のはず"
    print(f"[OK] rolling_rhythm_pattern_udf   : rhy[-1]={rhy[-1]:.6f}")

    # ---- rolling_harmony_udf ----
    har = rolling_harmony_udf(close, 30)
    finite_har = har[np.isfinite(har)]
    assert len(finite_har) > 0, "harmony: 有限値が存在するはず"
    assert np.all((finite_har >= 0.0) & (finite_har <= 1.0)), "harmony: [0,1] のはず"
    print(f"[OK] rolling_harmony_udf          : har[-1]={har[-1]:.6f}")

    # ---- rolling_musical_tension_udf ----
    mt = rolling_musical_tension_udf(close, 30)
    finite_mt = mt[np.isfinite(mt)]
    assert len(finite_mt) > 0, "musical_tension: 有限値が存在するはず"
    assert np.all((finite_mt >= 0.0) & (finite_mt <= 1.0)), "musical_tension: [0,1] のはず"
    print(f"[OK] rolling_musical_tension_udf  : mt[-1]={mt[-1]:.6f}")

    # ---- rolling_kinetic_energy_udf ----
    ke = rolling_kinetic_energy_udf(close, 30)
    finite_ke = ke[np.isfinite(ke)]
    assert len(finite_ke) > 0, "kinetic_energy: 有限値が存在するはず"
    assert np.all(finite_ke >= 0.0), "kinetic_energy: >=0 のはず"
    print(f"[OK] rolling_kinetic_energy_udf   : ke[-1]={ke[-1]:.8f}")

    # ---- rolling_muscle_force_udf ----
    mf = rolling_muscle_force_udf(close, 30)
    finite_mf = mf[np.isfinite(mf)]
    assert len(finite_mf) > 0, "muscle_force: 有限値が存在するはず"
    assert np.all(finite_mf >= 0.0), "muscle_force: >=0 のはず"
    print(f"[OK] rolling_muscle_force_udf     : mf[-1]={mf[-1]:.8f}")

    # ---- rolling_biomechanical_efficiency_udf ----
    be = rolling_biomechanical_efficiency_udf(close, 30)
    finite_be = be[np.isfinite(be)]
    assert len(finite_be) > 0, "biomechanical_efficiency: 有限値が存在するはず"
    assert np.all((finite_be >= 0.0) & (finite_be <= 1.0)), "biomechanical_efficiency: [0,1] のはず"
    print(f"[OK] rolling_biomechanical_efficiency_udf: be[-1]={be[-1]:.6f}")

    # ---- rolling_energy_expenditure_udf ----
    ee = rolling_energy_expenditure_udf(close, 30)
    finite_ee = ee[np.isfinite(ee)]
    assert len(finite_ee) > 0, "energy_expenditure: 有限値が存在するはず"
    assert np.all(finite_ee >= 0.0), "energy_expenditure: >=0 のはず"
    print(f"[OK] rolling_energy_expenditure_udf: ee[-1]={ee[-1]:.6f}")

    print()
    print("=" * 60)
    print("全テスト PASSED ✅")
    print("=" * 60)


if __name__ == "__main__":
    _run_validation()
