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
    # [SSoT 統一] Engine 1D の UDF を core_indicators から import (案3 完全SSoT)
    # 学習側 engine_1_D / 本番側 rfe_1D ともシグネチャ完全一致のため単純置換可能。
    # 旧: 本ファイル内で重複定義 → 学習側と二重実装 = SSoT 違反
    cmf_udf,
    mfi_udf,
    vwap_udf,
    obv_udf,
    accumulation_distribution_udf,
    chaikin_volatility_udf,
    mass_index_udf,
    hv_robust_udf,
    hv_standard_udf,
    commodity_channel_index_udf,
    fibonacci_levels_udf,
    candlestick_patterns_udf,
    # 残り 4 関数 (engine 1D 追加分: 学習側のみ計算するもの)
    # force_index_udf, donchian_channel_udf, pivot_point_udf は本番側で未使用のためimport不要
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
    prev == 0 のとき: x[i] > 0 → +inf, x[i] < 0 → -inf, x[i] == 0 → NaN
    （Polars/numpy 準拠: 符号を保持し、0/0 のみ NaN）
    先頭は nan。fastmath=True は engine_1_D 学習側の全 UDF と統一。
    [TRAIN-SERVE-FIX] 旧版は prev==0 のとき常に np.inf を返していたが、
    arr[i] が負の場合に学習側 Polars では -inf になるため、符号判定を追加。"""
    n = len(arr)
    pct = np.full(n, np.nan, dtype=np.float64)
    if n < 2:
        return pct
    for i in range(1, n):
        prev = arr[i - 1]
        if prev != 0.0:
            pct[i] = (arr[i] - prev) / prev
        else:
            # prev == 0: 学習側 Polars / numpy と同じく、x[i] の符号で inf を返す
            cur = arr[i]
            if cur > 0.0:
                pct[i] = np.inf
            elif cur < 0.0:
                pct[i] = -np.inf
            else:
                pct[i] = np.nan  # 0 / 0
    return pct


# ==================================================================
# 1D用 Numba UDF群（Volume・Flow系）
# ==================================================================


# ==================================================================
# 1D用 Numba UDF群（Volatility系）
# ==================================================================


# ==================================================================
# 1D用 Numba UDF群（Breakout・Support/Resistance・Price Action系）
# ==================================================================


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
        # [TRAIN-SERVE-FIX] 学習側 Polars rolling_std(252, ddof=1) と完全一致させる:
        #   学習側: pct_change().rolling_std(252, ddof=1) * sqrt(252)
        #     → Polars rolling_std はウィンドウ内に NaN が1本でも含まれると NaN を返す
        #   旧本番側: hv_standard_udf(pct_252) * sqrt(252)
        #     → UDF は finite フィルタで NaN を除外して計算 → 学習側と挙動が異なる
        #   新本番側: ウィンドウ内全要素が有限値の場合のみ ddof=1 で std を計算
        # 注: hv_robust_annual_252 は学習側でも UDF (hv_robust_udf * sqrt(252)) を使用しているため
        #     本番側も hv_robust_udf を維持する（NaN 除外挙動は学習側と同一）。
        pct_252 = _window(close_pct, 252)
        if len(pct_252) < 252:
            features["e1d_hv_annual_252"]        = np.nan
            features["e1d_hv_robust_annual_252"] = np.nan
        else:
            # hv_annual_252: 学習側 rolling_std(252, ddof=1) と完全一致
            # ウィンドウ内に1本でも NaN/inf があれば NaN（Polars 挙動）
            if np.all(np.isfinite(pct_252)):
                features["e1d_hv_annual_252"] = float(np.std(pct_252, ddof=1)) * np.sqrt(252)
            else:
                features["e1d_hv_annual_252"] = np.nan
            # hv_robust_annual_252: 学習側 UDF と一致（finite 除外挙動）
            features["e1d_hv_robust_annual_252"] = (
                hv_robust_udf(pct_252) * np.sqrt(252)
            )

        # ボラティリティレジーム: 直近HV50 vs 過去1440本の各時点HV50の分位数
        # 学習側: Polars rolling_quantile(0.8/0.6, window=1440) on rolling_std(50, ddof=1)
        #   hv_50[t] = std(pct_change[t-49:t+1], ddof=1)  ← Polars rolling_std はウィンドウ内NaNあれば NaN
        #   q80_roll[t] = quantile(hv_50[t-1439:t+1], 0.8)  ← Polars rolling_quantile は NaN を除外
        #   結果 = (hv_50[t] > q80_roll[t]) + (hv_50[t] > q60_roll[t])
        #
        # [TRAIN-SERVE-FIX] 旧本番側は cur_hv50 / hv50_hist 計算に hv_standard_udf を使っていたが、
        # hv_standard_udf は finite フィルタで NaN を除外して計算する一方、
        # 学習側 Polars rolling_std はウィンドウ内に NaN が1本でもあれば NaN を返すため
        # 挙動が異なる。本番側を rolling_std(50, ddof=1) と完全一致させる。
        # （rolling_quantile は両者とも NaN を除外する点で挙動一致）

        def _rolling_std_50_at(pct_arr: np.ndarray) -> float:
            """学習側 Polars rolling_std(50, ddof=1) と等価。
            ウィンドウ内に NaN/inf が1本でもあれば NaN を返す。"""
            if len(pct_arr) < 50:
                return np.nan
            if not np.all(np.isfinite(pct_arr)):
                return np.nan
            return float(np.std(pct_arr, ddof=1))

        cur_hv50 = _rolling_std_50_at(_window(close_pct, 50))
        n_needed = 1440 + 50
        if len(close_pct) >= n_needed and np.isfinite(cur_hv50):
            # 過去 (1440+50) 本の close_pct を取得
            hist_pct = close_pct[-n_needed:]
            # 各時点の HV50 をローリング計算（学習側 rolling_std(50, ddof=1) と完全一致）
            hv50_hist = np.full(n_needed, np.nan, dtype=np.float64)
            for _i in range(50 - 1, n_needed):
                hv50_hist[_i] = _rolling_std_50_at(hist_pct[_i - 49 : _i + 1])
            # 有効な直近1440本の HV50 から分位数を計算
            # Polars rolling_quantile は NaN を除外する挙動のため、本番側も同じく除外
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

        # Force Index normalized: price_change * volume / (atr * (vol_ma1440 + 1e-10) + 1e-10)
        # [TRAIN-SERVE-FIX] 学習側 Polars 式と完全一致させる:
        #   学習側: vol_ma1440 = pl.col("volume").rolling_mean(1440) + 1e-10  ← 既に +1e-10
        #           force_raw / (atr_13_internal_expr * vol_ma1440 + 1e-10)
        #         = force_raw / (atr * (mean(vol_1440) + 1e-10) + 1e-10)
        #         = force_raw / (atr * mean(vol_1440) + atr * 1e-10 + 1e-10)
        #   旧本番側: vol_ma1440 = np.mean(volume[-1440:])  ← +1e-10 なし
        #             force_raw / (atr13_last * vol_ma1440 + 1e-10)
        #           = force_raw / (atr * mean(vol_1440) + 1e-10)  ← atr*1e-10 が抜けている
        #   新本番側: 学習側と同じく vol_ma1440 に + 1e-10 を加えて計算
        if _atr_valid and len(close_arr) >= 2:
            price_change = close_arr[-1] - close_arr[-2]
            force_raw = price_change * float(volume_arr[-1])
            features["e1d_force_index_norm"] = float(
                force_raw / (atr13_last * (vol_ma1440 + 1e-10) + 1e-10)
            )
        else:
            features["e1d_force_index_norm"] = np.nan

        # Volume MA20 relative: ma20 / vol_ma1440
        # [修正4] vol_ma20 の定義では + 1e-10 を持たせず、
        # 使用箇所ごとに明示的にゼロ除算保護を適用する（vol_ma1440 と対称）
        vol_ma20 = float(np.mean(_window(volume_arr, 20)))
        features["e1d_volume_ma20_rel"] = float(vol_ma20 / (vol_ma1440 + 1e-10))

        # Volume ratio: vol[-1] / ma20
        # [TRAIN-SERVE-FIX] 学習側 Polars 式は volume / rolling_mean(volume, 20) で
        # ゼロ保護なし（vol_ma20=0 のとき inf を伝播させて QA でクリップ）。
        # 旧本番側: volume[-1] / (vol_ma20 + 1e-10)  ← +1e-10 で学習側より値が微小に小さくなる
        # 新本番側: 学習側と同じくゼロ保護なし。vol_ma20=0 の場合のみ +inf を返す。
        if vol_ma20 != 0.0:
            features["e1d_volume_ratio"] = float(float(volume_arr[-1]) / vol_ma20)
        else:
            # vol_ma20=0 → 学習側 Polars は inf を返す（数値の符号を保持）
            v_last = float(volume_arr[-1])
            if v_last > 0.0:
                features["e1d_volume_ratio"] = np.inf
            elif v_last < 0.0:
                features["e1d_volume_ratio"] = -np.inf
            else:
                features["e1d_volume_ratio"] = np.nan

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
        #   e1d_sample_weight は QA 対象外（学習側と同一設計）
        #     【Phase 5 修正 (#36)】学習側 engine_1_D は e1d_sample_weight を base_columns に含めて
        #     QA 対象外としているが、本番側 rfe_1D で QA 適用していたため train-serve skew が発生していた。
        #     rfe_1E と同型のパターンで除外する。
        #   qa_state=None の場合: inf/NaN → 0.0 のみ（後方互換）
        # ----------------------------------------------------------
        if qa_state is not None:
            qa_result: Dict[str, float] = {}
            for key, val in features.items():
                if key == "e1d_sample_weight":
                    qa_result[key] = val  # sample_weight は QA 対象外
                else:
                    qa_result[key] = qa_state.update_and_clip(key, val)
            features = qa_result
        else:
            for key in list(features.keys()):
                if key != "e1d_sample_weight" and not np.isfinite(features[key]):
                    features[key] = 0.0

        return features
