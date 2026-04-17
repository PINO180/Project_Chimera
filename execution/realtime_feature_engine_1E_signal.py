# realtime_feature_engine_1E_signal.py
# Project Cimera V5: Feature Engine Module 1E (Signal / Spectral / Wavelet / Hilbert)
#
# [Step 11 リファクタリング]
# 全 DSP UDF を core_indicators.py (Single Source of Truth) から import することで
# engine_1_E (学習側) との完全な数値一致を保証する。
#
# 主な変更点:
#   1. numba_fft / spectral_centroid_udf / spectral_flatness_udf /
#      wavelet_entropy_udf / hilbert_phase_var_udf /
#      hilbert_phase_stability_udf / acoustic_power_udf の
#      ローカル定義を廃止 → core_indicators から import
#   2. hilbert_phase_var_udf / hilbert_phase_stability_udf の
#      np.roll() 近似 → get_analytic_signal() (FFTベース厳密実装) に統一
#   3. fastmath=True (旧) → fastmath=False (core_indicators 統一値) に変更
#   4. signal_peak_to_peak_100 への ATR 割り追加
#      (calculate_atr_wilder + scale_by_atr 経由)
#   5. wavelet_std_* の np.std(ddof=1) 直書き → stddev_unbiased に統一
#   6. e1e_sample_weight を calculate_sample_weight で計算・返却

import sys
from pathlib import Path

# -----------------------------------------------------------------------
# パス解決: blueprint → core_indicators の順で解決する
# ① まず親ディレクトリ (/workspace) を sys.path に追加して blueprint を解決
# ② 次に blueprint.CORE_DIR (/workspace/core) を追加して core_indicators を解決
# ③ blueprint が見つからない場合は相対パスの fallback を使用
# -----------------------------------------------------------------------
_parent_dir = str(Path(__file__).resolve().parents[1])  # /workspace
if _parent_dir not in sys.path:
    sys.path.append(_parent_dir)

try:
    import blueprint as config
    _core_dir = str(config.CORE_DIR)          # /workspace/core
    if _core_dir not in sys.path:
        sys.path.append(_core_dir)
except ModuleNotFoundError:
    # blueprint が見つからない場合 (テスト環境など) は相対 fallback
    _fallback_core = str(Path(__file__).resolve().parent / "core")
    if _fallback_core not in sys.path:
        sys.path.append(_fallback_core)

from core_indicators import (
    # [ATR & VOLATILITY]
    calculate_atr_wilder,
    scale_by_atr,
    # [STATS]
    stddev_unbiased,
    # [WEIGHT]
    calculate_sample_weight,
    # [DSP] — スペクトル系
    spectral_centroid_udf,
    spectral_bandwidth_udf,
    spectral_rolloff_udf,
    spectral_flux_udf,
    spectral_flatness_udf,
    spectral_entropy_udf,
    # [DSP] — ウェーブレット系
    wavelet_energy_udf,
    wavelet_entropy_udf,
    # [DSP] — ヒルベルト系
    hilbert_amplitude_udf,
    hilbert_phase_var_udf,
    hilbert_phase_stability_udf,
    hilbert_freq_mean_udf,
    hilbert_freq_std_udf,
    # [DSP] — 音響系
    acoustic_power_udf,
    acoustic_frequency_udf,
)

import numpy as np
from numba import njit
from typing import Dict, Any, Optional

# ==================================================================
# [Step 11] ローカル UDF 定義廃止ノート
# ==================================================================
# 以下の関数はすべて core_indicators.[CATEGORY: DSP] で定義されており、
# 上記 import 文で取り込まれています。このファイルへの重複定義は廃止。
#
#   numba_fft              → core_indicators.numba_fft
#   spectral_centroid_udf  → core_indicators.spectral_centroid_udf
#   spectral_flatness_udf  → core_indicators.spectral_flatness_udf
#   wavelet_entropy_udf    → core_indicators.wavelet_entropy_udf
#   hilbert_phase_var_udf  → core_indicators.hilbert_phase_var_udf
#   hilbert_phase_stability_udf → core_indicators.hilbert_phase_stability_udf
#   acoustic_power_udf     → core_indicators.acoustic_power_udf
#
# アルゴリズム変更（乖離解消）:
#   hilbert_phase_var_udf / hilbert_phase_stability_udf:
#     旧: np.roll(window_data, 1) による近似（粗く、scipy.signal.hilbert と異なる）
#     新: get_analytic_signal() (FFTベース厳密実装) — engine_1_E 学習側と完全一致
#
#   wavelet_entropy_udf:
#     旧: parallel=True (スレッド衝突リスクあり)
#     新: parallel=False (core_indicators 統一値)
#
#   fastmath: True → False (浮動小数点の再現性を優先)
# ==================================================================


# ==================================================================
# QAState — 学習側 apply_quality_assurance_to_group の等価実装
# ==================================================================


class QAState:
    """
    学習側 apply_quality_assurance_to_group のリアルタイム等価実装。

    学習側の処理（Polars LazyFrame 全系列一括）:
        safe_col = when(col.is_infinite()).then(None).otherwise(col)
        ewm_mean = safe_col.ewm_mean(half_life=HL, adjust=False, ignore_nulls=True)
        ewm_std  = safe_col.ewm_std (half_life=HL, adjust=False, ignore_nulls=True)
        result   = when(col==inf).then(p99).when(col==-inf).then(p01)
                   .otherwise(col).clip(p01, p99).fill_null(0.0).fill_nan(0.0)

    alpha = 1 - exp(-ln2 / half_life)
    EWM_mean[t] = alpha * x[t] + (1-alpha) * EWM_mean[t-1]  (NaN/inf はスキップ)
    EWM_var[t]  = (1-alpha) * (EWM_var[t-1] + alpha * (x[t] - EWM_mean[t-1])^2)

    使い方:
        qa_state = FeatureModule1E.QAState(lookback_bars=1440)
        for bar in live_stream:
            features = FeatureModule1E.calculate_features(data_window, 1440, qa_state)
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
        p01       = ewm_mean - 5.0 * ewm_std
        p99       = ewm_mean + 5.0 * ewm_std

        if np.isnan(ewm_input):
            return 0.0
        if is_pos_inf:
            return float(p99) if np.isfinite(p99) else 0.0
        if is_neg_inf:
            return float(p01) if np.isfinite(p01) else 0.0

        clipped = float(np.clip(raw_val, p01, p99))
        return clipped if np.isfinite(clipped) else 0.0


# ==================================================================
# ユーティリティ関数（このファイル固有・軽量処理用）
# ==================================================================


@njit(fastmath=False, cache=True)
def _window(arr: np.ndarray, window: int) -> np.ndarray:
    """直近 window 本のスライスを返す。"""
    if window <= 0:
        return np.empty(0, dtype=arr.dtype)
    if window > len(arr):
        return arr
    return arr[-window:]


@njit(fastmath=False, cache=True)
def _last(arr: np.ndarray) -> float:
    """配列の最終要素をスカラーで返す。空配列は nan。"""
    if len(arr) == 0:
        return np.nan
    return float(arr[-1])


def _pct_change(arr: np.ndarray) -> np.ndarray:
    """
    価格変化率（pct_change）を計算する。先頭要素は nan。

    Polars pct_change() と完全一致:
      (arr[i] - arr[i-1]) / arr[i-1]
      prev == 0 の場合は inf（Polars準拠）。
    """
    n = len(arr)
    pct = np.full(n, np.nan, dtype=np.float64)
    if n < 2:
        return pct
    with np.errstate(divide="ignore", invalid="ignore"):
        pct[1:] = (arr[1:] - arr[:-1]) / arr[:-1]
    return pct


# ==================================================================
# メイン計算モジュール
# ==================================================================


class FeatureModule1E:

    # 外部から FeatureModule1E.QAState としてアクセス可能にする
    QAState = QAState

    @staticmethod
    def calculate_features(
        data: Dict[str, np.ndarray],
        lookback_bars: int = 1440,
        qa_state: Optional[QAState] = None,
    ) -> Dict[str, float]:
        """
        リアルタイム特徴量計算。学習側 engine_1_E と完全一致。

        Args:
            data        : {"close": np.ndarray, "high": np.ndarray, "low": np.ndarray}
            lookback_bars: タイムフレームに応じた1日あたりのバー数（QA EWM半減期）
            qa_state    : QAState インスタンス。本番稼働時は必ず渡し、毎バー使い回すこと。
                          None の場合は QA 処理をスキップ（後方互換）。

        Returns:
            特徴量名 → スカラー値 の辞書。e1e_sample_weight も含む。
        """
        features: Dict[str, float] = {}

        close_arr = data["close"]
        high_arr  = data.get("high", close_arr)
        low_arr   = data.get("low",  close_arr)

        if len(close_arr) == 0:
            return features

        close_pct = _pct_change(close_arr)

        # ---------------------------------------------------------
        # 0. サンプルウェイト
        # ---------------------------------------------------------
        sw_arr = calculate_sample_weight(
            high_arr.astype(np.float64),
            low_arr.astype(np.float64),
            close_arr.astype(np.float64),
        )
        features["e1e_sample_weight"] = float(sw_arr[-1]) if len(sw_arr) > 0 else 1.0

        # ---------------------------------------------------------
        # 1. スペクトル系 (window=[64,128,256,512])
        #    学習側: _create_spectral_features
        # ---------------------------------------------------------
        for window in [64, 128, 256, 512]:
            features[f"e1e_spectral_centroid_{window}"] = _last(
                spectral_centroid_udf(_window(close_pct, window), window)
            )
            features[f"e1e_spectral_bandwidth_{window}"] = _last(
                spectral_bandwidth_udf(_window(close_pct, window), window)
            )
            features[f"e1e_spectral_rolloff_{window}"] = _last(
                spectral_rolloff_udf(_window(close_pct, window), window)
            )
            features[f"e1e_spectral_flux_{window}"] = _last(
                spectral_flux_udf(_window(close_pct, window), window)
            )
            features[f"e1e_spectral_flatness_{window}"] = _last(
                spectral_flatness_udf(_window(close_pct, window), window)
            )
            features[f"e1e_spectral_entropy_{window}"] = _last(
                spectral_entropy_udf(_window(close_pct, window), window)
            )
            # spectral_energy: sum(pct_change^2) over window（Polarsネイティブと等価）
            w_e = _window(close_pct, window)
            features[f"e1e_spectral_energy_{window}"] = (
                float(np.sum(w_e ** 2)) if len(w_e) >= window else np.nan
            )

        # spectral_peak_freq_128: rolling_max(128) / (rolling_std(128, ddof=1) + 1e-10)
        w_pf = _window(close_pct, 128)
        if len(w_pf) >= 128:
            features["e1e_spectral_peak_freq_128"] = (
                float(np.max(w_pf)) / (float(np.std(w_pf, ddof=1)) + 1e-10)
            )
        else:
            features["e1e_spectral_peak_freq_128"] = np.nan

        # ---------------------------------------------------------
        # 2. ウェーブレット系 (window=[32,64,128,256])
        #    学習側: _create_wavelet_features
        # ---------------------------------------------------------
        for window in [32, 64, 128, 256]:
            features[f"e1e_wavelet_energy_{window}"] = _last(
                wavelet_energy_udf(_window(close_pct, window), window)
            )
            w_wv = _window(close_pct, window)
            features[f"e1e_wavelet_mean_{window}"] = (
                float(np.mean(w_wv)) if len(w_wv) >= window else np.nan
            )
            # wavelet_std: rolling_std(ddof=1) 学習側と一致
            # 【修正】stddev_unbiased → np.std(ddof=1) に変更。
            # stddev_unbiased はNaN要素を除外して計算するため、
            # len(close_arr)==window のとき close_pct[0](NaN)が混入し
            # 学習側 Polars rolling_std（NaN1本でもNaN返却）と不一致。
            # np.std はNaN伝播するため学習側と完全一致。
            if len(w_wv) >= window:
                std_val = float(np.std(w_wv, ddof=1))
                features[f"e1e_wavelet_std_{window}"] = (
                    float(std_val) if np.isfinite(std_val) else np.nan
                )
            else:
                features[f"e1e_wavelet_std_{window}"] = np.nan

        features["e1e_wavelet_entropy_64"] = _last(
            wavelet_entropy_udf(_window(close_pct, 64), 64)
        )

        # ---------------------------------------------------------
        # 3. ヒルベルト系 (window=[50,100,200])
        #    学習側: _create_hilbert_features
        # ---------------------------------------------------------
        for window in [50, 100, 200]:
            features[f"e1e_hilbert_amplitude_{window}"] = _last(
                hilbert_amplitude_udf(_window(close_pct, window), window)
            )

        # hilbert_amp_mean_100 / std_100 / cv_100: |pct_change|のrolling統計
        w_amp = _window(np.abs(close_pct), 100)
        if len(w_amp) >= 100:
            amp_mean = float(np.mean(w_amp))
            features["e1e_hilbert_amp_mean_100"] = amp_mean
            # 【修正】stddev_unbiased → np.std(ddof=1) に変更（NaN伝播を学習側と一致させる）
            amp_std = float(np.std(w_amp, ddof=1))
            amp_std = float(amp_std) if np.isfinite(amp_std) else np.nan
            features["e1e_hilbert_amp_std_100"] = amp_std
            features["e1e_hilbert_amp_cv_100"] = (
                amp_std / (amp_mean + 1e-10) if np.isfinite(amp_mean) else np.nan
            )
        else:
            amp_mean = np.nan
            features["e1e_hilbert_amp_mean_100"] = np.nan
            features["e1e_hilbert_amp_std_100"]  = np.nan
            features["e1e_hilbert_amp_cv_100"]   = np.nan

        features["e1e_hilbert_phase_stability_50"] = _last(
            hilbert_phase_stability_udf(_window(close_pct, 50), 50)
        )
        features["e1e_hilbert_phase_var_50"] = _last(
            hilbert_phase_var_udf(_window(close_pct, 50), 50)
        )

        features["e1e_hilbert_freq_mean_100"] = _last(
            hilbert_freq_mean_udf(_window(close_pct, 100), 100)
        )
        features["e1e_hilbert_freq_std_100"] = _last(
            hilbert_freq_std_udf(_window(close_pct, 100), 100)
        )

        # hilbert_freq_energy_ratio_100:
        # sum(pct_change^2, 100) / ((atr_13/close)^2 * 100 + 1e-10)
        w_fe = _window(close_pct, 100)
        pct_energy = float(np.sum(w_fe ** 2)) if len(w_fe) >= 100 else np.nan
        atr13_full = calculate_atr_wilder(
            high_arr.astype(np.float64),
            low_arr.astype(np.float64),
            close_arr.astype(np.float64),
            13,
        )
        atr13_last = float(atr13_full[-1]) if len(atr13_full) > 0 else np.nan
        close_last = float(close_arr[-1])
        atr13_pct  = atr13_last / (close_last + 1e-10)
        features["e1e_hilbert_freq_energy_ratio_100"] = (
            pct_energy / (atr13_pct ** 2 * 100 + 1e-10)
            if (np.isfinite(pct_energy) and np.isfinite(atr13_pct))
            else np.nan
        )

        # ---------------------------------------------------------
        # 4. 音響系 (window=[128,256,512])
        #    学習側: _create_acoustic_features
        # ---------------------------------------------------------
        for window in [128, 256, 512]:
            features[f"e1e_acoustic_power_{window}"] = _last(
                acoustic_power_udf(_window(close_pct, window), window)
            )
            features[f"e1e_acoustic_frequency_{window}"] = _last(
                acoustic_frequency_udf(_window(close_pct, window), window)
            )

        # ---------------------------------------------------------
        # 5. 信号統計系 (Signal Stats)
        #    学習側: _create_signal_stats_features
        # ---------------------------------------------------------
        # signal_rms_50: sqrt(rolling_mean(pct_change^2, 50))
        w_rms_50 = _window(close_pct, 50)
        features["e1e_signal_rms_50"] = (
            float(np.sqrt(np.mean(w_rms_50 ** 2))) if len(w_rms_50) >= 50 else np.nan
        )

        # signal_peak_to_peak_100: (max - min) / (atr_100 + 1e-10)
        w_sig_100 = _window(close_arr, 100)
        w_hi_100  = _window(high_arr,  100)
        w_lo_100  = _window(low_arr,   100)
        if len(w_sig_100) >= 100:
            atr_100 = calculate_atr_wilder(
                w_hi_100.astype(np.float64),
                w_lo_100.astype(np.float64),
                w_sig_100.astype(np.float64),
                100,
            )
            atr_last_val = atr_100[-1] if len(atr_100) > 0 else np.nan
            if np.isfinite(atr_last_val):
                raw_range = np.array([np.max(w_sig_100) - np.min(w_sig_100)], dtype=np.float64)
                atr_last  = np.array([atr_last_val], dtype=np.float64)
                features["e1e_signal_peak_to_peak_100"] = float(scale_by_atr(raw_range, atr_last)[0])
            else:
                features["e1e_signal_peak_to_peak_100"] = np.nan
        else:
            features["e1e_signal_peak_to_peak_100"] = np.nan

        # signal_crest_factor_50:
        # 学習側: rolling_max(|pct_change|, 50) / (sqrt(rolling_mean(pct_change^2, 50)) + 1e-10)
        # 注意: 学習側は pct_change().rolling_max(50).abs() = abs(rolling_max)
        #       = ウィンドウ内最大値の絶対値（max(abs) とは異なる）
        w_cf = _window(close_pct, 50)
        if len(w_cf) >= 50:
            rms_cf = float(np.sqrt(np.mean(w_cf ** 2)))
            # 学習側と一致: abs(rolling_max) = abs(max of signed values)
            features["e1e_signal_crest_factor_50"] = (
                abs(float(np.max(w_cf))) / (rms_cf + 1e-10)
            )
        else:
            features["e1e_signal_crest_factor_50"] = np.nan

        # ----------------------------------------------------------
        # QA処理 — 学習側 apply_quality_assurance_to_group と等価
        #   学習側: inf→null → EWM(half_life=lookback_bars)±5σクリップ → fill_null/nan(0.0)
        #   e1e_sample_weight は QA 対象外（学習側と同一設計）
        #   qa_state=None の場合: inf/NaN → 0.0 のみ（後方互換）
        # ----------------------------------------------------------
        if qa_state is not None:
            qa_result: Dict[str, float] = {}
            for key, val in features.items():
                if key == "e1e_sample_weight":
                    qa_result[key] = val  # sample_weight は QA 対象外
                else:
                    qa_result[key] = qa_state.update_and_clip(key, val)
            features = qa_result
        else:
            for key in list(features.keys()):
                if key != "e1e_sample_weight" and not np.isfinite(features[key]):
                    features[key] = 0.0

        return features
