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
    # [SSoT 統一] Engine 1B の UDF を core_indicators から import
    # (旧: 本ファイル内で定義 → 学習側 engine_1_B と二重定義 = SSoT 違反)
    adf_統計量_udf,
    phillips_perron_統計量_udf,
    kpss_統計量_udf,
    t分布_自由度_udf,
    t分布_尺度_udf,
    gev_形状_udf,
    holt_winters_レベル_udf,
    holt_winters_トレンド_udf,
    arima_残差分散_udf,
    kalman_状態推定_udf,
    lowess_適合値_udf,
    theil_sen_傾き_udf,
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
        #
        # 【修正前 (誤式)】
        #   bias_corr = 1 / sqrt(1 - (1-alpha)^(2n))
        #
        # 【修正後 (Polars 互換式 — 1e-15 精度で完全一致)】
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

        # 【修正済み】Option B: チェック順序入れ替え (is_pos_inf → is_neg_inf → np.isnan(raw_val))
        # 旧バグ: np.isnan(ewm_input) が先に発火し is_pos_inf に到達不能 → +inf 入力で常に 0.0
        if is_pos_inf:
            return float(upper) if np.isfinite(upper) else 0.0
        if is_neg_inf:
            return float(lower) if np.isfinite(lower) else 0.0
        if np.isnan(raw_val):
            return 0.0

        clipped = float(np.clip(raw_val, lower, upper))
        return clipped if np.isfinite(clipped) else 0.0


# ==================================================================
# Numba UDF群（学習側 engine_1_B と完全同一実装）
# ==================================================================

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
        #   rel_volume[i] = volume[i] / (rolling_mean_1440[i] + 1e-10)  ← 各バーで分母再計算
        #     ※ Polars rolling_mean のデフォルト min_samples = window_size のため、
        #        i < lookback_bars - 1 のバーでは rolling_mean_1440[i] = NaN
        #        → rel_volume[i] = volume[i] / NaN = NaN
        #   volume_ma20[t] = rel_volume.rolling_mean(20)[t]
        #     = (1/20) * Σ_{i=t-19}^{t} rel_volume[i]
        #     ※ ウィンドウ内に NaN が1つでもあれば結果は NaN
        #   volume_price_trend[t] = (pct_change * rel_volume).rolling_mean(10)[t]
        #     = (1/10) * Σ_{i=t-9}^{t} (pct_change[i] * rel_volume[i])
        # [TRAIN-SERVE-FIX] 旧本番側は「最終バー時点の rolling_mean_1440 を分母として
        # 固定使用」していたため、学習側と微小乖離（1〜3%）が発生していた。
        # 新本番側では各バーごとに rel_volume を計算し、学習側と完全一致させる。
        if len(volume_arr) > 0:
            n_vol = len(volume_arr)

            # 各バーごとの rel_volume を計算（学習側と完全同等）
            # rel_volume[i] = volume[i] / (rolling_mean_1440[i] + 1e-10)
            # i < lookback_bars - 1 のバーは rolling_mean_1440[i] = NaN → rel_volume[i] = NaN
            #
            # volume_ma20 計算には直近20バー、volume_price_trend には直近10バーの rel_volume が必要。
            # 効率化のため、必要な末尾20バーのみ rel_volume を計算する。
            # 各バー i について: mean(volume[max(0, i-1439):i+1])
            # ただし学習側 Polars rolling_mean(1440) は i < 1439 で NaN を返すため、
            # i < lookback_bars - 1 のバーは NaN を割り当てる。

            def _rolling_mean_1440_at(i: int) -> float:
                """学習側 Polars rolling_mean(lookback_bars) と同等。
                i < lookback_bars - 1 のバーは NaN を返す。"""
                if i < lookback_bars - 1:
                    return np.nan
                return float(np.mean(volume_arr[i - lookback_bars + 1: i + 1]))

            # 直近20バー分の rel_volume を計算
            # rel_vol_recent[k] (k=0..N-1) は配列末尾から (N-1-k) バー前に対応
            # つまり rel_vol_recent[-1] が現バー、rel_vol_recent[-2] が1バー前... の順
            n_compute = min(20, n_vol)
            rel_vol_recent = np.full(n_compute, np.nan, dtype=np.float64)
            for k in range(n_compute):
                # i は配列末尾から (n_compute - 1 - k) バー前
                i = n_vol - 1 - (n_compute - 1 - k)
                rolling_mean_at_i = _rolling_mean_1440_at(i)
                if np.isnan(rolling_mean_at_i):
                    rel_vol_recent[k] = np.nan
                else:
                    rel_vol_recent[k] = volume_arr[i] / (rolling_mean_at_i + 1e-10)

            # volume_ma20: 直近20バーが揃っており、かつ全バーで rel_volume が有効であれば計算
            # （Polars rolling_mean(20) は min_samples=20 デフォルトで、ウィンドウ内に NaN が
            #  1つでも含まれると NaN を返す）
            if n_vol >= 20 and not np.any(np.isnan(rel_vol_recent)):
                features["e1b_volume_ma20"] = float(np.mean(rel_vol_recent))
            else:
                features["e1b_volume_ma20"] = np.nan

            # volume_price_trend: 直近10バーの (pct_change * rel_volume) の平均
            # rel_vol_recent の末尾10要素が直近10バーに対応（n_compute >= 10 のとき）
            if n_vol >= 10 and len(close_pct) >= 10 and n_compute >= 10:
                pct_w10 = close_pct[-10:]
                rel_v_w10 = rel_vol_recent[-10:]
                if not np.any(np.isnan(rel_v_w10)) and not np.any(np.isnan(pct_w10)):
                    # 学習側: (pct_change * rel_volume).rolling_mean(10) は
                    # inf をそのまま伝播させ QA でクリップする。
                    # finite フィルタは適用せず、infも mean に通して QA に任せる。
                    vpt = pct_w10 * rel_v_w10
                    features["e1b_volume_price_trend"] = float(np.mean(vpt))
                else:
                    features["e1b_volume_price_trend"] = np.nan
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
