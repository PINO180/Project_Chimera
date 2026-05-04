# realtime_feature_engine_1C_technical.py
# Project Cimera V5 - Feature Module 1C (Technical Indicators)
#
# 【Step 改訂】学習側との完全一致修正
#   - RSI系全バリアント追加 (rsi_14/21/30/50, momentum, stochastic_rsi, divergence)
#   - MACD系全バリアント追加 (12_26, 5_35, 19_39 + signal/histogram)
#   - BB系全バリアント追加 (period=[20,30,50] × std=[2,2.5,3])
#   - ATR系全バリアント追加 (atr_13/21/34/55, pct, trend, volatility)
#   - Oscillator系全追加 (stoch全, ADX全, DI+/-, Aroon全, Williams %R全)
#   - Momentum系全追加 (dpo, trix, uo, tsi, roc, momentum全, STC, coppock, kst_signal, price_oscillator)
#   - MA系全追加 (sma/deviation, ema/deviation, wma, hma, kama, trend_slope, trend_strength, trend_consistency)

import sys
import math
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import numba as nb
from numba import njit

sys.path.append(str(Path(__file__).resolve().parent.parent))
import blueprint as config
sys.path.append(str(config.CORE_DIR))
from core_indicators import (
    calculate_atr_wilder,
    calculate_rsi_wilder,
    calculate_adx,
    # [SSoT 統一] Engine 1C の Numba 関数 (案3: 案-3完全SSoT)
    # core_indicators の guvectorize 版を import → out 引数省略で配列を返す形式で利用可
    # (旧: 本ファイル内で _arr サフィックス付き等で重複定義 = SSoT 違反)
    calculate_wma_numba,
    calculate_hma_numba,
    calculate_kama_numba,
    calculate_stochastic_numba,
    calculate_williams_r_numba,
    calculate_trix_numba,
    calculate_ultimate_oscillator_numba,
    calculate_aroon_up_numba,
    calculate_aroon_down_numba,
    calculate_tsi_numba,
    _calculate_di_wilder,
)


# ==================================================================
# 共通ヘルパー関数群
# ==================================================================

@njit(fastmath=False, cache=True)
def _window(arr: np.ndarray, window: int) -> np.ndarray:
    if window <= 0:
        return np.empty(0, dtype=arr.dtype)
    if window > len(arr):
        return arr
    return arr[-window:]

@njit(fastmath=False, cache=True)
def _last(arr: np.ndarray) -> float:
    if len(arr) == 0:
        return np.nan
    return float(arr[-1])

@njit(fastmath=False, cache=True)
def _ema(arr: np.ndarray, span: int) -> np.ndarray:
    n = len(arr)
    out = np.zeros(n, dtype=np.float64)
    if n == 0:
        return out
    alpha = 2.0 / (span + 1.0)
    out[0] = arr[0]
    for i in range(1, n):
        out[i] = alpha * arr[i] + (1.0 - alpha) * out[i - 1]
    return out

@njit(fastmath=False, cache=True)
def _rolling_mean(arr: np.ndarray, window: int) -> np.ndarray:
    """NaN-aware SMA（Polars rolling_mean 準拠）"""
    n = len(arr)
    out = np.full(n, np.nan)
    if window <= 0 or n < window:
        return out
    for i in range(window - 1, n):
        has_nan = False
        s = 0.0
        for j in range(i - window + 1, i + 1):
            if np.isnan(arr[j]):
                has_nan = True
                break
            s += arr[j]
        if not has_nan:
            out[i] = s / window
    return out

@njit(fastmath=False, cache=True)
def rolling_mean_numba(arr: np.ndarray, window: int) -> float:
    n = len(arr)
    if n < window:
        return np.nan
    s = 0.0
    for i in range(n - window, n):
        s += arr[i]
    return s / window


# ==================================================================
# QAState — 学習側 apply_quality_assurance の等価実装
# ==================================================================

class QAState:
    """
    【修正4】学習側 apply_quality_assurance のリアルタイム等価実装。

    学習側の処理（Polars LazyFrame 全系列一括）:
        ewm_mean = col.ewm_mean(half_life=HL, adjust=False, ignore_nulls=True)
        ewm_std  = col.ewm_std (half_life=HL, adjust=False, ignore_nulls=True)
        result   = col.fill_nan(0.0).fill_null(0.0).clip(ewm_mean - 5*ewm_std, ewm_mean + 5*ewm_std)

    Polars ewm_mean(adjust=False) の再帰式:
        alpha = 1 - exp(-ln2 / half_life)
        EWM_mean[t] = alpha * x[t] + (1-alpha) * EWM_mean[t-1]  (NaN はスキップ)
    Polars ewm_std(adjust=False) の再帰式:
        EWM_var[t] = (1-alpha) * (EWM_var[t-1] + alpha * (x[t] - EWM_mean[t-1])^2)

    使い方:
        qa_state = FeatureModule1C.QAState(lookback_bars=1440)
        for bar in live_stream:
            features = FeatureModule1C.calculate_features(data_window, 1440, qa_state)
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
        #   学習側: EWMは元col（inf含む）で計算（ignore_nulls=TrueでinfはNaN扱いスキップ）
        #           fill_nan(0.0).fill_null(0.0).clip() → infはclipでupper/lowerにclip
        #   本番側旧: inf → 0.0 変換後 EWMに投入（学習側と不一致）
        #   本番側新: inf を先に記録し、EWMはNaNとしてスキップ。
        #             clip時に +inf → upper_bound, -inf → lower_bound で置き換える。
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

        # ±5σ クリップ（bias補正付き）
        # Polars ewm_std(adjust=False, bias=False) と完全一致させる:
        #   m = n - 1, sum_w2 = α²(1-r2^m)/(1-r2) + r2^m, bias_factor_var = 1/(1-sum_w2)
        ewm_mean  = self._ewm_mean[key]
        n_updates = self._ewm_n.get(key, 1)
        if n_updates <= 1:
            ewm_std = 0.0
        else:
            r2 = (1.0 - alpha) ** 2
            m  = n_updates - 1
            if r2 < 1.0 - 1e-15:
                sum_w2 = alpha * alpha * (1.0 - r2 ** m) / (1.0 - r2) + r2 ** m
            else:
                sum_w2 = 1.0
            if sum_w2 < 1.0 - 1e-15:
                bias_factor_var = 1.0 / (1.0 - sum_w2)
                ewm_std = np.sqrt(max(self._ewm_var[key] * bias_factor_var, 0.0))
            else:
                ewm_std = 0.0
        lower     = ewm_mean - 5.0 * ewm_std
        upper     = ewm_mean + 5.0 * ewm_std

        # 【修正済み】Option B: チェック順序入れ替え
        if is_pos_inf:
            return float(upper) if np.isfinite(upper) else 0.0
        if is_neg_inf:
            return float(lower) if np.isfinite(lower) else 0.0
        if np.isnan(raw_val):
            return 0.0

        clipped  = float(np.clip(raw_val, lower, upper))
        return clipped if np.isfinite(clipped) else 0.0


# ==================================================================
# Numba UDF群（学習側と完全同一）
# ==================================================================

# ==================================================================
# メイン計算クラス
# ==================================================================

class FeatureModule1C:

    # 外部から FeatureModule1C.QAState としてアクセス可能にする
    QAState = QAState

    @staticmethod
    def calculate_features(
        data: Dict[str, np.ndarray],
        lookback_bars: int = 1440,
        qa_state: Optional[QAState] = None,
    ) -> Dict[str, float]:
        features = {}

        _empty = np.array([], dtype=np.float64)
        close_arr  = data.get("close",  _empty)
        high_arr   = data.get("high",   _empty)
        low_arr    = data.get("low",    _empty)
        volume_arr = data.get("volume", _empty)
        open_arr   = data.get("open",   _empty)

        if len(close_arr) == 0:
            return features

        current_close = float(close_arr[-1])

        # ATR（全特徴量共有）
        atr_13_arr = calculate_atr_wilder(high_arr, low_arr, close_arr, 13)
        atr_21_arr = calculate_atr_wilder(high_arr, low_arr, close_arr, 21)
        atr_34_arr = calculate_atr_wilder(high_arr, low_arr, close_arr, 34)
        atr_55_arr = calculate_atr_wilder(high_arr, low_arr, close_arr, 55)

        atr_13 = _last(atr_13_arr)
        atr_ok = np.isfinite(atr_13) and atr_13 > 1e-10
        atr_denom = atr_13 + 1e-10  # スカラー ATR割り分母

        # ---------------------------------------------------------
        # 1. ATR系
        # ---------------------------------------------------------
        # [TRAIN-SERVE-FIX] atr_pct のゼロ保護を学習側と完全一致させる:
        #   学習側: atr_raw / (close + 1e-10) * 100  ← + 1e-10 ゼロ保護
        #   旧本番: if current_close != 0.0 else np.nan  ← 条件分岐式の保護
        #   新本番: atr_raw / (close + 1e-10) * 100  ← 学習側と完全同一
        # XAU/USDのcloseは事実上ゼロにならないが、思想として完全に揃える。
        # 以下、最初のatr_pct_13/21の二重計算ブロックは削除し、後段ループで一括処理する。
        # atr_13/21/34/55 = atr_period / atr_13（比率）
        # atr_pct_period = atr_period / (close + 1e-10) * 100  （学習側完全一致）
        for period, arr in [(13, atr_13_arr), (21, atr_21_arr), (34, atr_34_arr), (55, atr_55_arr)]:
            atr_val = _last(arr)
            features[f"e1c_atr_{period}"] = (atr_val / atr_denom) if (np.isfinite(atr_val) and atr_ok) else np.nan
            # atr_pct: 学習側 atr_raw / (close + 1e-10) * 100 と完全一致
            features[f"e1c_atr_pct_{period}"] = (
                (atr_val / (current_close + 1e-10)) * 100
                if (np.isfinite(atr_val) and np.isfinite(current_close)) else np.nan
            )
            # atr_trend: diff / atr_13
            if len(arr) >= 2 and np.isfinite(atr_val) and atr_ok:
                features[f"e1c_atr_trend_{period}"] = (atr_val - arr[-2]) / atr_denom
            else:
                features[f"e1c_atr_trend_{period}"] = np.nan
            # atr_volatility: rolling_std(atr_period, window=period) / atr_13
            # 学習側: rolling_std(atr_raw, period)[i] / atr_13_arr[i] と完全一致
            w_atr = _window(arr, period)
            if len(w_atr) >= period and atr_ok:
                features[f"e1c_atr_volatility_{period}"] = float(np.std(w_atr, ddof=1)) / atr_denom
            else:
                features[f"e1c_atr_volatility_{period}"] = np.nan

        # ---------------------------------------------------------
        # 2. RSI系
        # ---------------------------------------------------------
        for period in [14, 21, 30, 50]:
            rsi_arr = calculate_rsi_wilder(close_arr, period)
            rsi_last = _last(rsi_arr)
            features[f"e1c_rsi_{period}"] = rsi_last
            features[f"e1c_rsi_momentum_{period}"] = (
                float(rsi_arr[-1] - rsi_arr[-2]) if len(rsi_arr) >= 2 else np.nan
            )

        for period in [14, 21]:
            rsi_arr = calculate_rsi_wilder(close_arr, period)
            rsi_min = _last(_rolling_mean(np.array([
                float(np.min(_window(rsi_arr, period))) if len(_window(rsi_arr, period)) > 0 else np.nan
            ]), 1))
            w_rsi = _window(rsi_arr, period)
            if len(w_rsi) >= period:
                rsi_min_w = float(np.min(w_rsi))
                rsi_max_w = float(np.max(w_rsi))
                rsi_last = _last(rsi_arr)
                features[f"e1c_stochastic_rsi_{period}"] = (
                    (rsi_last - rsi_min_w) / (rsi_max_w - rsi_min_w + 1e-10) * 100
                )
            else:
                features[f"e1c_stochastic_rsi_{period}"] = np.nan

            # rsi_divergence: price_change - rsi_change
            if len(close_arr) > period and len(rsi_arr) > period:
                price_prev = close_arr[-1 - period]
                price_change = (current_close - price_prev) / price_prev if price_prev != 0.0 else np.nan
                rsi_change = (rsi_arr[-1] - rsi_arr[-1 - period]) / 50 - 1
                features[f"e1c_rsi_divergence_{period}"] = (
                    price_change - rsi_change if np.isfinite(price_change) else np.nan
                )
            else:
                features[f"e1c_rsi_divergence_{period}"] = np.nan

        # ---------------------------------------------------------
        # 3. MACD系
        # ---------------------------------------------------------
        for fast, slow, signal in [(12, 26, 9), (5, 35, 5), (19, 39, 9)]:
            ema_f = _ema(close_arr, fast)
            ema_s = _ema(close_arr, slow)
            macd_arr = ema_f - ema_s
            sig_arr = _ema(macd_arr, signal)
            hist_arr = macd_arr - sig_arr
            features[f"e1c_macd_{fast}_{slow}"] = _last(macd_arr) / atr_denom if atr_ok else np.nan
            features[f"e1c_macd_signal_{fast}_{slow}_{signal}"] = _last(sig_arr) / atr_denom if atr_ok else np.nan
            features[f"e1c_macd_histogram_{fast}_{slow}_{signal}"] = _last(hist_arr) / atr_denom if atr_ok else np.nan

        # ---------------------------------------------------------
        # 4. ボリンジャーバンド系
        # ---------------------------------------------------------
        for period in [20, 30, 50]:
            w = _window(close_arr, period)
            if len(w) >= period:
                sma = float(np.mean(w))
                std = float(np.std(w, ddof=1))
                for num_std in [2, 2.5, 3]:
                    upper_raw = sma + num_std * std
                    lower_raw = sma - num_std * std
                    width_raw = upper_raw - lower_raw
                    features[f"e1c_bb_upper_{period}_{num_std}"] = (upper_raw - current_close) / atr_denom if atr_ok else np.nan
                    features[f"e1c_bb_lower_{period}_{num_std}"] = (current_close - lower_raw) / atr_denom if atr_ok else np.nan
                    features[f"e1c_bb_percent_{period}_{num_std}"] = (current_close - lower_raw) / (width_raw + 1e-10)
                    features[f"e1c_bb_width_{period}_{num_std}"]   = width_raw / atr_denom if atr_ok else np.nan
                    features[f"e1c_bb_width_pct_{period}_{num_std}"] = (width_raw / (sma + 1e-10)) * 100
                    features[f"e1c_bb_position_{period}_{num_std}"] = (current_close - sma) / (std + 1e-10)
            else:
                for num_std in [2, 2.5, 3]:
                    for k in ["bb_upper", "bb_lower", "bb_percent", "bb_width", "bb_width_pct", "bb_position"]:
                        features[f"e1c_{k}_{period}_{num_std}"] = np.nan

        # ---------------------------------------------------------
        # 5. ADX, DI, Aroon系
        # ---------------------------------------------------------
        for period in [13, 21, 34]:
            adx_arr = calculate_adx(high_arr, low_arr, close_arr, period)
            features[f"e1c_adx_{period}"]      = _last(adx_arr)
            _di_p_arr, _di_m_arr = _calculate_di_wilder(high_arr, low_arr, close_arr, period)

            features[f"e1c_di_plus_{period}"]  = float(_di_p_arr[-1]) if np.isfinite(_di_p_arr[-1]) else np.nan

            features[f"e1c_di_minus_{period}"] = float(_di_m_arr[-1]) if np.isfinite(_di_m_arr[-1]) else np.nan

        for period in [14, 25, 50]:
            aroon_up   = _last(calculate_aroon_up_numba(high_arr, period))
            aroon_down = _last(calculate_aroon_down_numba(low_arr, period))
            features[f"e1c_aroon_up_{period}"]          = aroon_up
            features[f"e1c_aroon_down_{period}"]         = aroon_down
            features[f"e1c_aroon_oscillator_{period}"]   = aroon_up - aroon_down

        # =====================================================================
        # 【修正済み】Williams %R の period=56 固定実装は今回の再学習で解消。
        # 学習側 engine_1_C の late binding バグ修正と同期して、本番側も
        # period=14, 28, 56 を個別に計算する正しい挙動に戻す。
        # =====================================================================
        for period in [14, 28, 56]:
            features[f"e1c_williams_r_{period}"] = _last(
                calculate_williams_r_numba(high_arr, low_arr, close_arr, period)
            )

        # Stochastic (14,3,3), (21,5,5), (9,3,3)
        for k_period, d_period, slow_period in [(14, 3, 3), (21, 5, 5), (9, 3, 3)]:
            stoch_k_arr = calculate_stochastic_numba(high_arr, low_arr, close_arr, k_period, d_period, slow_period)
            features[f"e1c_stoch_k_{k_period}"]  = _last(stoch_k_arr)
            stoch_d_arr = _rolling_mean(stoch_k_arr, d_period)
            features[f"e1c_stoch_d_{k_period}_{d_period}"] = _last(stoch_d_arr)
            slow_d_arr  = _rolling_mean(stoch_d_arr, slow_period)
            features[f"e1c_stoch_slow_d_{k_period}_{d_period}_{slow_period}"] = _last(slow_d_arr)

        # ---------------------------------------------------------
        # 6. モメンタム系
        # ---------------------------------------------------------
        # DPO
        for period in [20, 30, 50]:
            w = _window(close_arr, period)
            sma = float(np.mean(w)) if len(w) >= period else np.nan
            dpo_raw = current_close - sma if np.isfinite(sma) else np.nan
            features[f"e1c_dpo_{period}"] = dpo_raw / atr_denom if (np.isfinite(dpo_raw) and atr_ok) else np.nan

        # TRIX
        for period in [14, 20, 30]:
            features[f"e1c_trix_{period}"] = _last(calculate_trix_numba(close_arr, period))

        # Ultimate Oscillator
        features["e1c_ultimate_oscillator"] = _last(calculate_ultimate_oscillator_numba(high_arr, low_arr, close_arr, volume_arr))

        # TSI
        for period in [25, 13]:
            features[f"e1c_tsi_{period}"] = _last(calculate_tsi_numba(close_arr, period))

        # Rate of Change
        for period in [10, 20, 30, 50]:
            if len(close_arr) > period:
                denom = close_arr[-1 - period]
                features[f"e1c_rate_of_change_{period}"] = (
                    ((current_close - denom) / denom) * 100 if denom != 0.0 else np.inf
                )
            else:
                features[f"e1c_rate_of_change_{period}"] = np.nan

        # Momentum
        for period in [10, 20, 30, 50]:
            if len(close_arr) > period and atr_ok:
                features[f"e1c_momentum_{period}"] = (current_close - close_arr[-1 - period]) / atr_denom
            else:
                features[f"e1c_momentum_{period}"] = np.nan

        # KST
        if len(close_arr) >= 31:
            d = {10: close_arr[-11], 15: close_arr[-16], 20: close_arr[-21], 30: close_arr[-31]}
            roc_kst = {p: ((current_close - d[p]) / d[p] if d[p] != 0.0 else np.inf) for p in d}
            kst_val = (roc_kst[10] * 1 + roc_kst[15] * 2 + roc_kst[20] * 3 + roc_kst[30] * 4) / 10 * 100
            features["e1c_kst"] = kst_val
            # kst_signal: rolling_mean(kst_arr, 9)
            # 2116本の配列から過去9バー分のKST値を計算して平均を取る
            kst_arr = np.full(len(close_arr), np.nan)
            # 学習側に合わせてROCは小数のまま（* 100 なし）→ kst = / 10 * 100 で1回だけ変換
            for idx in range(30, len(close_arr)):
                c = close_arr[idx]
                d10 = close_arr[idx - 10]; d15 = close_arr[idx - 15]
                d20 = close_arr[idx - 20]; d30 = close_arr[idx - 30]
                r10 = (c - d10) / d10 if d10 != 0.0 else np.nan
                r15 = (c - d15) / d15 if d15 != 0.0 else np.nan
                r20 = (c - d20) / d20 if d20 != 0.0 else np.nan
                r30 = (c - d30) / d30 if d30 != 0.0 else np.nan
                if np.isfinite(r10) and np.isfinite(r15) and np.isfinite(r20) and np.isfinite(r30):
                    kst_arr[idx] = (r10 * 1 + r15 * 2 + r20 * 3 + r30 * 4) / 10 * 100
            # kst_signal: rolling_mean(9)準拠 — 末尾9バーが揃っている場合のみ計算
            w_kst = _window(kst_arr, 9)
            w_kst_finite = w_kst[np.isfinite(w_kst)]
            features["e1c_kst_signal"] = float(np.mean(w_kst_finite)) if len(w_kst_finite) == 9 else np.nan
        else:
            features["e1c_kst"]        = np.nan
            features["e1c_kst_signal"] = np.nan

        # Trend Strength
        # 【修正】条件を >= 2 → >= period に変更。
        # 学習側: rolling_std(period) → period本未満はNaN。
        for period in [20, 50, 100]:
            w = _window(close_arr, period)
            if len(w) >= period and atr_ok:
                normalized_std = float(np.std(w, ddof=1)) / atr_denom
                features[f"e1c_trend_strength_{period}"] = min(1.0 / (normalized_std + 1e-10), 100.0)
            else:
                features[f"e1c_trend_strength_{period}"] = np.nan

        # Trend Consistency: 方向変化の頻度
        # [TRAIN-SERVE-FIX] 学習側 Polars rolling_mean(period) はデフォルト
        # min_samples=period のため、period 本未満では NaN を返す。
        # close.diff().sign().diff().abs() は先頭2バーが NaN になるため、
        # 有効な direction_changes が period 本揃うには close が period + 2 本必要。
        #
        # 旧: if len(w) >= 3:  → period=20 でも 3本あれば計算してしまい学習側と不一致
        # 新: if len(w) >= period + 2:  → period 本の direction_changes が揃ってから計算
        for period in [20, 50, 100]:
            w = _window(close_arr, period + 2)
            if len(w) >= period + 2:
                diff1 = np.diff(w)
                sign1 = np.sign(diff1)
                direction_changes = np.abs(np.diff(sign1))
                features[f"e1c_trend_consistency_{period}"] = 1 - float(np.mean(direction_changes)) / 2
            else:
                features[f"e1c_trend_consistency_{period}"] = np.nan

        # Coppock Curve: (ROC(11) + ROC(14)).rolling_mean(10)
        # 学習側: (roc_11 + roc_14).rolling_mean(10)
        if len(close_arr) >= 25:
            coppock_arr = np.full(len(close_arr), np.nan)
            for idx in range(14, len(close_arr)):
                c = close_arr[idx]
                p11 = close_arr[idx - 11]; p14 = close_arr[idx - 14]
                r11 = (c - p11) / p11 * 100 if p11 != 0.0 else np.nan
                r14 = (c - p14) / p14 * 100 if p14 != 0.0 else np.nan
                if np.isfinite(r11) and np.isfinite(r14):
                    coppock_arr[idx] = r11 + r14
            # coppock_curve: rolling_mean(10)準拠 — 末尾10バーが揃っている場合のみ計算
            w_copp = _window(coppock_arr, 10)
            w_copp_finite = w_copp[np.isfinite(w_copp)]
            features["e1c_coppock_curve"] = float(np.mean(w_copp_finite)) if len(w_copp_finite) == 10 else np.nan
        else:
            features["e1c_coppock_curve"] = np.nan

        # Schaff Trend Cycle
        # 【修正1】学習側と完全一致:
        #   - EMA に half_life=period を使用（学習側: ewm_mean(half_life=fast_period, adjust=False)）
        #     alpha = 1 - exp(-ln2/half_life)  ← span=period とは alpha 定義が異なる
        #   - 2nd Stochastic + Final Smooth を省略せず完全実装
        #   - smooth_period=3 は span=3 の EWM（学習側と同一）
        for fast_period, slow_period_stc, cycle_period in [(23, 50, 10), (12, 26, 9)]:
            n = len(close_arr)
            # half_life EMA（学習側 ewm_mean(half_life=X, adjust=False) と等価）
            alpha_f = 1.0 - np.exp(-np.log(2.0) / fast_period)
            alpha_s = 1.0 - np.exp(-np.log(2.0) / slow_period_stc)
            alpha3  = 2.0 / (3 + 1.0)  # span=3 の EWM（smooth_period=3）

            fast_ma = np.zeros(n, dtype=np.float64)
            slow_ma = np.zeros(n, dtype=np.float64)
            fast_ma[0] = close_arr[0]
            slow_ma[0] = close_arr[0]
            for i in range(1, n):
                fast_ma[i] = alpha_f * close_arr[i] + (1.0 - alpha_f) * fast_ma[i - 1]
                slow_ma[i] = alpha_s * close_arr[i] + (1.0 - alpha_s) * slow_ma[i - 1]
            macd_stc = fast_ma - slow_ma

            # 1st Stochastic (%K of MACD) using rolling_min/max over cycle_period
            stoch_macd = np.full(n, np.nan, dtype=np.float64)
            for i in range(cycle_period - 1, n):
                w = macd_stc[i - cycle_period + 1:i + 1]
                mn, mx = np.min(w), np.max(w)
                stoch_macd[i] = (macd_stc[i] - mn) / (mx - mn + 1e-10) * 100

            # Smooth 1st Stochastic with EMA span=3
            sm1 = np.full(n, np.nan, dtype=np.float64)
            started = False
            for i in range(n):
                if np.isnan(stoch_macd[i]):
                    continue
                if not started:
                    sm1[i] = stoch_macd[i]
                    started = True
                else:
                    prev = sm1[i - 1] if not np.isnan(sm1[i - 1]) else stoch_macd[i]
                    sm1[i] = alpha3 * stoch_macd[i] + (1.0 - alpha3) * prev

            # 2nd Stochastic (%K of Smoothed)
            # 【修正】NaN除外(w2f)を廃止。
            # 学習側: Polars rolling_min/max はNaNを伝播させる。
            # 旧: w2f = w2[~np.isnan(w2)] でNaNを除外 → 学習側と不一致。
            # 新: w2内に非有限値が1本でもあればskip（NaN伝播と同一挙動）。
            stoch2 = np.full(n, np.nan, dtype=np.float64)
            for i in range(cycle_period - 1, n):
                w2 = sm1[i - cycle_period + 1:i + 1]
                if not np.all(np.isfinite(w2)):
                    continue
                mn2, mx2 = np.min(w2), np.max(w2)
                if np.isnan(sm1[i]):
                    continue
                stoch2[i] = (sm1[i] - mn2) / (mx2 - mn2 + 1e-10) * 100

            # Final Smooth with EMA span=3
            stc_arr = np.full(n, np.nan, dtype=np.float64)
            started2 = False
            for i in range(n):
                if np.isnan(stoch2[i]):
                    continue
                if not started2:
                    stc_arr[i] = stoch2[i]
                    started2 = True
                else:
                    prev2 = stc_arr[i - 1] if not np.isnan(stc_arr[i - 1]) else stoch2[i]
                    stc_arr[i] = alpha3 * stoch2[i] + (1.0 - alpha3) * prev2

            features[f"e1c_schaff_trend_cycle_{fast_period}_{slow_period_stc}_{cycle_period}"] = float(stc_arr[-1]) if np.isfinite(stc_arr[-1]) else np.nan

        # Price Oscillator
        for fast, slow in [(12, 26), (5, 35), (10, 20)]:
            ema_f = _ema(close_arr, fast)
            ema_s = _ema(close_arr, slow)
            ema_s_last = _last(ema_s)
            po = (_last(ema_f) - ema_s_last) / ema_s_last * 100 if ema_s_last != 0.0 else np.nan
            features[f"e1c_price_oscillator_{fast}_{slow}"] = po

        # RVI
        for period in [10, 14, 20]:
            if len(close_arr) >= period and len(open_arr) >= period and len(high_arr) >= period and len(low_arr) >= period:
                rvi_arr = _rolling_mean(close_arr - open_arr, period) / (_rolling_mean(high_arr - low_arr, period) + 1e-10)
                features[f"e1c_relative_vigor_index_{period}"] = _last(rvi_arr)
                # 学習側: rvi_signal は period=[10,14,20] 全てに対して生成
                w_rvi = _window(rvi_arr, 4)
                features[f"e1c_rvi_signal_{period}"] = float(np.mean(w_rvi)) if len(w_rvi) >= 4 else np.nan
            else:
                features[f"e1c_relative_vigor_index_{period}"] = np.nan
                features[f"e1c_rvi_signal_{period}"] = np.nan

        # ---------------------------------------------------------
        # 7. 移動平均系
        # ---------------------------------------------------------
        for period in [10, 20, 50, 100, 200]:
            w = _window(close_arr, period)
            # SMA
            sma_val = float(np.mean(w)) if len(w) >= period else np.nan
            features[f"e1c_sma_{period}"] = (sma_val - current_close) / atr_denom if (np.isfinite(sma_val) and atr_ok) else np.nan
            features[f"e1c_sma_deviation_{period}"] = (current_close - sma_val) / (sma_val + 1e-10) * 100 if np.isfinite(sma_val) else np.nan
            # EMA
            ema_val = _last(_ema(close_arr, period))
            features[f"e1c_ema_{period}"] = (ema_val - current_close) / atr_denom if (np.isfinite(ema_val) and atr_ok) else np.nan
            features[f"e1c_ema_deviation_{period}"] = (current_close - ema_val) / (ema_val + 1e-10) * 100 if np.isfinite(ema_val) else np.nan
            # WMA
            wma_val = _last(calculate_wma_numba(close_arr, period))
            features[f"e1c_wma_{period}"] = (wma_val - current_close) / atr_denom if (np.isfinite(wma_val) and atr_ok) else np.nan

        # HMA
        for period in [21, 34, 55]:
            hma_val = _last(calculate_hma_numba(close_arr, period))
            features[f"e1c_hma_{period}"] = (hma_val - current_close) / atr_denom if (np.isfinite(hma_val) and atr_ok) else np.nan

        # KAMA
        for period in [21, 34]:
            kama_val = _last(calculate_kama_numba(close_arr, period))
            features[f"e1c_kama_{period}"] = (kama_val - current_close) / atr_denom if (np.isfinite(kama_val) and atr_ok) else np.nan

        # Trend Slope: OLS slope = 6*(WMA - SMA)/(period-1) / atr13
        for period in [20, 50, 100]:
            w = _window(close_arr, period)
            if len(w) >= period and atr_ok:
                sma_w = float(np.mean(w))
                wma_w = _last(calculate_wma_numba(w, period))
                if np.isfinite(wma_w):
                    slope = 6.0 * (wma_w - sma_w) / (period - 1.0)
                    features[f"e1c_trend_slope_{period}"] = slope / atr_denom
                else:
                    features[f"e1c_trend_slope_{period}"] = np.nan
            else:
                features[f"e1c_trend_slope_{period}"] = np.nan

        # ----------------------------------------------------------
        # 【修正4】QA処理 — 学習側 apply_quality_assurance と等価
        #   学習側: fill_nan(0.0).fill_null(0.0).clip(ewm±5σ)
        #   本番側: QAState が EWM を跨バー保持し逐次更新・クリップ
        #   qa_state=None の場合: fill_nan/fill_null(0.0) のみ適用（後方互換）
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
