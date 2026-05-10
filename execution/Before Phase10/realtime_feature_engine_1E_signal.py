# realtime_feature_engine_1E_signal.py
# Category 1E: 信号処理系 (Spectral / Wavelet / Hilbert / Acoustic / Signal Stats)
#
# ==================================================================
# 【Phase 9b 改修】司令塔統合 .select() 対応 (FFI overhead 削減)
# ==================================================================
#
# 目的: Phase 9 (Step B) で達成した Polars 直呼び + DSP UDF 直接呼びの
#       2 層構造を保ったまま、6 モジュールの Polars 式を司令塔で 1 回の
#       .select() に統合できるよう構造を分解する。
#
# 【Phase 9b の改修】
#   追加: `_build_polars_pieces(data, lookback_bars) -> (columns, exprs, layer2)`
#     - columns: close + __temp_atr_13 + __temp_atr_100 (raw ATR、+1e-10 は割り算時)
#     - exprs:   Polars 式リスト (spectral_energy/peak_freq, wavelet_mean/std,
#                hilbert_amp_*, hilbert_freq_energy_ratio, signal_rms/peak_to_peak/crest_factor)
#     - layer2:  DSP Numba UDF 直接呼び結果 (spectral_centroid/bandwidth/rolloff/
#                flux/flatness/entropy, wavelet_energy/entropy, hilbert_amplitude/phase/freq,
#                acoustic_power/frequency) + e1e_sample_weight
#   変更: `calculate_features` は `_build_polars_pieces` を呼んで単独計算する
#         薄いラッパーへ。後方互換完全維持。
#
# 【1E の特徴】
#   1E は最初から Layer 1 (Polars rolling 統計) と Layer 2 (DSP UDF 直接呼び)
#   が明示的に分離されており、Phase 9b への分解がもっとも素直なモジュール。
#
#   DSP UDF は O(window²) の FFT 計算で重く、最後の window 本のみ渡せば最終バー
#   の値が決まるため、numpy 直接呼びが最適 (学習側 map_batches と最終バーで同値)。
#
# 【ATR の扱い】
#   学習側: pl.struct(...).map_batches(calculate_atr_wilder(..., 13/100))
#           → 割り算時に + 1e-10 を加える
#   本番側: numpy で事前計算して __temp_atr_{13,100} 列に raw 値を入れる
#           → 割り算時に Polars 式で `(pl.col("__temp_atr_13") + 1e-10)` を使う
#   結果: 学習側と完全同値の計算経路
#
# 【SSoT 階層】(Phase 9 から不変)
#   Layer 1 (rolling 統計): Polars Rust エンジン
#   Layer 2 (Numba UDF):    core_indicators (SSoT)
#
# 【保持される過去の修正】
#   ・QAState (apply_quality_assurance_to_group の等価実装、bias=False 補正)
#   ・hilbert_phase_*_udf は core_indicators の FFT-Hilbert 厳密実装を使用
#   ・e1e_sample_weight は学習側 base_columns 扱いで QA 対象外
# ==================================================================

import sys
from pathlib import Path

# -----------------------------------------------------------------------
# パス解決: blueprint → core_indicators の順で解決する
# -----------------------------------------------------------------------
_parent_dir = str(Path(__file__).resolve().parents[1])
if _parent_dir not in sys.path:
    sys.path.append(_parent_dir)

try:
    import blueprint as config
    _core_dir = str(config.CORE_DIR)
    if _core_dir not in sys.path:
        sys.path.append(_core_dir)
except ModuleNotFoundError:
    _fallback_core = str(Path(__file__).resolve().parent / "core")
    if _fallback_core not in sys.path:
        sys.path.append(_fallback_core)

from core_indicators import (
    # [ATR & VOLATILITY]
    calculate_atr_wilder,
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
import polars as pl
import numba as nb
from typing import Dict, Optional, Tuple, List


# ==================================================================
# ヘルパー関数
# ==================================================================

@nb.njit(fastmath=False, cache=True)
def _pct_change(arr: np.ndarray) -> np.ndarray:
    """Polars pct_change() と完全一致: (x[i] - x[i-1]) / x[i-1]
    prev == 0 のとき: x[i] > 0 → +inf, x[i] < 0 → -inf, x[i] == 0 → NaN
    先頭は nan。
    """
    n = len(arr)
    out = np.full(n, np.nan, dtype=np.float64)
    if n < 2:
        return out
    for i in range(1, n):
        prev = arr[i - 1]
        if prev != 0.0:
            out[i] = (arr[i] - prev) / prev
        else:
            cur = arr[i]
            if cur > 0.0:
                out[i] = np.inf
            elif cur < 0.0:
                out[i] = -np.inf
            else:
                out[i] = np.nan
    return out


# ==================================================================
# QAState — 学習側 apply_quality_assurance_to_group の等価実装
# (1A〜1D と完全に同一の実装。Phase 9b では変更なし。)
# ==================================================================

class QAState:
    """学習側 apply_quality_assurance_to_group のリアルタイム等価実装。
    詳細は realtime_feature_engine_1A_statistics.py の QAState を参照。
    """

    def __init__(self, lookback_bars: int = 1440):
        self.alpha: float = 1.0 - np.exp(-np.log(2.0) / max(lookback_bars, 1))
        self._ewm_mean: Dict[str, float] = {}
        self._ewm_var: Dict[str, float] = {}
        self._ewm_n: Dict[str, int] = {}

    def update_and_clip(self, key: str, raw_val: float) -> float:
        alpha = self.alpha

        is_pos_inf = np.isposinf(raw_val)
        is_neg_inf = np.isneginf(raw_val)
        ewm_input = np.nan if not np.isfinite(raw_val) else raw_val

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
                new_mean = alpha * ewm_input + (1.0 - alpha) * prev_mean
                new_var  = (1.0 - alpha) * (prev_var + alpha * (ewm_input - prev_mean) ** 2)
                self._ewm_mean[key] = new_mean
                self._ewm_var[key]  = new_var
                self._ewm_n[key]    = self._ewm_n.get(key, 0) + 1

        ewm_mean = self._ewm_mean[key]
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
        p01 = ewm_mean - 5.0 * ewm_std
        p99 = ewm_mean + 5.0 * ewm_std

        if is_pos_inf:
            return float(p99) if np.isfinite(p99) else 0.0
        if is_neg_inf:
            return float(p01) if np.isfinite(p01) else 0.0
        if np.isnan(raw_val):
            return 0.0

        clipped = float(np.clip(raw_val, p01, p99))
        return clipped if np.isfinite(clipped) else 0.0


# ==================================================================
# メイン計算クラス
# ==================================================================

class FeatureModule1E:

    QAState = QAState

    @staticmethod
    def _build_polars_pieces(
        data: Dict[str, np.ndarray],
        lookback_bars: int = 1440,
    ) -> Tuple[Dict[str, np.ndarray], List[pl.Expr], Dict[str, float]]:
        """
        統合 .select() 用の 3 要素を返す。

        Returns:
            columns: Dict[str, np.ndarray]
                共通列 (close) + 1E 固有の __temp_atr_13 / __temp_atr_100 (raw)。
            exprs: List[pl.Expr]
                Polars rolling 統計式リスト (alias は e1e_* の最終特徴量名)。
                spectral_energy/peak_freq, wavelet_mean/std, hilbert_amp_*,
                hilbert_freq_energy_ratio, signal_rms/peak_to_peak/crest_factor。
            layer2: Dict[str, float]
                DSP UDF 直接呼び結果 (close_pct[-window:] に対する最終バー値)
                + e1e_sample_weight (QA対象外)。
        """
        close_arr = data["close"].astype(np.float64)
        if len(close_arr) == 0:
            return {}, [], {}

        high_arr  = (
            data["high"].astype(np.float64) if "high" in data and len(data["high"]) > 0
            else close_arr
        )
        low_arr   = (
            data["low"].astype(np.float64) if "low" in data and len(data["low"]) > 0
            else close_arr
        )

        # ---------------------------------------------------------
        # ATR 系列の事前計算 (学習側 atr_13_expr_hilbert / atr_100_expr と完全一致)
        # 学習側は割り算時に + 1e-10 を加えるため、ここでは raw ATR を保持する。
        # ---------------------------------------------------------
        atr13_arr  = calculate_atr_wilder(high_arr, low_arr, close_arr, 13)
        atr100_arr = calculate_atr_wilder(high_arr, low_arr, close_arr, 100)

        # close_pct を numpy で 1 度だけ計算 (学習側 Polars pct_change と semantics 一致)
        close_pct = _pct_change(close_arr)
        n = len(close_pct)

        # ===== columns =====
        columns: Dict[str, np.ndarray] = {
            "close":         close_arr,
            "__temp_atr_13":  atr13_arr,
            "__temp_atr_100": atr100_arr,
        }

        # ===== exprs (Layer 1: Polars rolling 統計) =====
        # 学習側 engine_1_E のうち rolling 統計に該当する式を集約。
        exprs: List[pl.Expr] = []

        # ----- Spectral group (Polars 部分) -----
        # spectral_energy: (pct_change ** 2).rolling_sum(window)
        # 参照: engine_1_E L1166-1171
        for window in [64, 128, 256, 512]:
            exprs.append(
                (pl.col("close").pct_change() ** 2)
                .rolling_sum(window)
                .alias(f"e1e_spectral_energy_{window}")
            )

        # spectral_peak_freq_128: rolling_max / (rolling_std + 1e-10)
        # 参照: engine_1_E L1175-1180
        exprs.append(
            (
                pl.col("close").pct_change().rolling_max(128)
                / (pl.col("close").pct_change().rolling_std(128, ddof=1) + 1e-10)
            ).alias("e1e_spectral_peak_freq_128")
        )

        # ----- Wavelet group (Polars 部分) -----
        # wavelet_mean / wavelet_std (Polars-native rolling stats)
        # 参照: engine_1_E L1202-1215
        for window in [32, 64, 128, 256]:
            exprs.append(
                pl.col("close").pct_change().rolling_mean(window)
                .alias(f"e1e_wavelet_mean_{window}")
            )
            exprs.append(
                pl.col("close").pct_change().rolling_std(window, ddof=1)
                .alias(f"e1e_wavelet_std_{window}")
            )

        # ----- Hilbert group (Polars 部分) -----
        # hilbert_amp_mean_100 / std_100 / cv_100 (Polars-native rolling stats on |pct_change|)
        # 参照: engine_1_E L1252-1273
        exprs.append(
            pl.col("close").pct_change().abs().rolling_mean(100)
            .alias("e1e_hilbert_amp_mean_100")
        )
        exprs.append(
            pl.col("close").pct_change().abs().rolling_std(100, ddof=1)
            .alias("e1e_hilbert_amp_std_100")
        )
        exprs.append(
            (
                pl.col("close").pct_change().abs().rolling_std(100, ddof=1)
                / (pl.col("close").pct_change().abs().rolling_mean(100) + 1e-10)
            ).alias("e1e_hilbert_amp_cv_100")
        )

        # hilbert_freq_energy_ratio_100:
        #   学習側: (close.pct_change()^2).rolling_sum(100) / ((atr_13/close)^2 * 100 + 1e-10)
        # 参照: engine_1_E L1335-1342
        atr_13_pct_expr = pl.col("__temp_atr_13") / (pl.col("close") + 1e-10)
        exprs.append(
            (
                (pl.col("close").pct_change() ** 2).rolling_sum(100)
                / (atr_13_pct_expr.pow(2) * 100 + 1e-10)
            ).alias("e1e_hilbert_freq_energy_ratio_100")
        )

        # ----- Signal Stats group (Polars 部分) -----
        # signal_rms_50: sqrt(rolling_mean(pct_change^2, 50))
        # 参照: engine_1_E L1397-1402
        exprs.append(
            (pl.col("close").pct_change() ** 2)
            .rolling_mean(50)
            .sqrt()
            .alias("e1e_signal_rms_50")
        )

        # signal_peak_to_peak_100: (close.rolling_max(100) - close.rolling_min(100)) / (atr_100 + 1e-10)
        # 参照: engine_1_E L1404-1410
        exprs.append(
            (
                (pl.col("close").rolling_max(100) - pl.col("close").rolling_min(100))
                / (pl.col("__temp_atr_100") + 1e-10)
            ).alias("e1e_signal_peak_to_peak_100")
        )

        # signal_crest_factor_50:
        #   学習側: pct_change.rolling_max(50).abs() / ((pct_change^2).rolling_mean(50).sqrt() + 1e-10)
        # 参照: engine_1_E L1412-1418
        exprs.append(
            (
                pl.col("close").pct_change().rolling_max(50).abs()
                / ((pl.col("close").pct_change() ** 2).rolling_mean(50).sqrt() + 1e-10)
            ).alias("e1e_signal_crest_factor_50")
        )

        # ===== layer2 (Layer 2: DSP UDF 直接呼び + sample_weight) =====
        # 各 UDF は rolling 計算であり、最終バー (index = window-1 in slice) の値は
        # 直近 window 本のみで決まる。学習側は全系列に対して UDF を呼び、その最終
        # 要素を採用するが、本番側は最終 window 本のみ渡しても同一値。
        layer2: Dict[str, float] = {}

        # ----- Spectral UDFs (window=[64,128,256,512]) -----
        # 参照: engine_1_E._create_spectral_features L1098-1163
        for window in [64, 128, 256, 512]:
            if n >= window:
                w_arr = close_pct[-window:]
                layer2[f"e1e_spectral_centroid_{window}"]  = float(spectral_centroid_udf(w_arr, window)[-1])
                layer2[f"e1e_spectral_bandwidth_{window}"] = float(spectral_bandwidth_udf(w_arr, window)[-1])
                layer2[f"e1e_spectral_rolloff_{window}"]   = float(spectral_rolloff_udf(w_arr, window)[-1])
                layer2[f"e1e_spectral_flatness_{window}"]  = float(spectral_flatness_udf(w_arr, window)[-1])
                layer2[f"e1e_spectral_entropy_{window}"]   = float(spectral_entropy_udf(w_arr, window)[-1])
            else:
                layer2[f"e1e_spectral_centroid_{window}"]  = np.nan
                layer2[f"e1e_spectral_bandwidth_{window}"] = np.nan
                layer2[f"e1e_spectral_rolloff_{window}"]   = np.nan
                layer2[f"e1e_spectral_flatness_{window}"]  = np.nan
                layer2[f"e1e_spectral_entropy_{window}"]   = np.nan

            # spectral_flux は隣接 2 フレーム必要 (window*2 本)
            if n >= window * 2:
                w_arr2 = close_pct[-(window * 2):]
                layer2[f"e1e_spectral_flux_{window}"] = float(spectral_flux_udf(w_arr2, window)[-1])
            else:
                layer2[f"e1e_spectral_flux_{window}"] = np.nan

        # ----- Wavelet UDFs -----
        # 参照: engine_1_E._create_wavelet_features L1190-1227
        for window in [32, 64, 128, 256]:
            if n >= window:
                layer2[f"e1e_wavelet_energy_{window}"] = float(
                    wavelet_energy_udf(close_pct[-window:], window)[-1]
                )
            else:
                layer2[f"e1e_wavelet_energy_{window}"] = np.nan

        # wavelet_entropy_64
        if n >= 64:
            layer2["e1e_wavelet_entropy_64"] = float(
                wavelet_entropy_udf(close_pct[-64:], 64)[-1]
            )
        else:
            layer2["e1e_wavelet_entropy_64"] = np.nan

        # ----- Hilbert UDFs -----
        # 参照: engine_1_E._create_hilbert_features L1237-1296
        for window in [50, 100, 200]:
            if n >= window:
                layer2[f"e1e_hilbert_amplitude_{window}"] = float(
                    hilbert_amplitude_udf(close_pct[-window:], window)[-1]
                )
            else:
                layer2[f"e1e_hilbert_amplitude_{window}"] = np.nan

        # phase_var_50, phase_stability_50
        if n >= 50:
            layer2["e1e_hilbert_phase_var_50"]       = float(hilbert_phase_var_udf(close_pct[-50:], 50)[-1])
            layer2["e1e_hilbert_phase_stability_50"] = float(hilbert_phase_stability_udf(close_pct[-50:], 50)[-1])
        else:
            layer2["e1e_hilbert_phase_var_50"]       = np.nan
            layer2["e1e_hilbert_phase_stability_50"] = np.nan

        # freq_mean_100, freq_std_100
        if n >= 100:
            layer2["e1e_hilbert_freq_mean_100"] = float(hilbert_freq_mean_udf(close_pct[-100:], 100)[-1])
            layer2["e1e_hilbert_freq_std_100"]  = float(hilbert_freq_std_udf(close_pct[-100:], 100)[-1])
        else:
            layer2["e1e_hilbert_freq_mean_100"] = np.nan
            layer2["e1e_hilbert_freq_std_100"]  = np.nan

        # ----- Acoustic UDFs (window=[128,256,512]) -----
        # 参照: engine_1_E._create_acoustic_features L1352-1373
        for window in [128, 256, 512]:
            if n >= window:
                w_arr = close_pct[-window:]
                layer2[f"e1e_acoustic_power_{window}"]     = float(acoustic_power_udf(w_arr, window)[-1])
                layer2[f"e1e_acoustic_frequency_{window}"] = float(acoustic_frequency_udf(w_arr, window)[-1])
            else:
                layer2[f"e1e_acoustic_power_{window}"]     = np.nan
                layer2[f"e1e_acoustic_frequency_{window}"] = np.nan

        # ----- サンプルウェイト (学習側 base_columns 扱いと一致、QA対象外) -----
        # 参照: engine_1_E L1733-1742
        sample_weight_arr = calculate_sample_weight(high_arr, low_arr, close_arr)
        layer2["e1e_sample_weight"] = (
            float(sample_weight_arr[-1]) if len(sample_weight_arr) > 0 else 1.0
        )

        return columns, exprs, layer2

    @staticmethod
    def calculate_features(
        data: Dict[str, np.ndarray],
        lookback_bars: int = 1440,
        qa_state: Optional[QAState] = None,
    ) -> Dict[str, float]:
        """
        【Phase 9b 改修版】単独計算用ラッパー。
        司令塔は _build_polars_pieces を直接呼んで全モジュール統合 .select() を
        行うが、本メソッドは後方互換のためモジュール単独で動作する形を維持する。
        """
        columns, exprs, layer2 = FeatureModule1E._build_polars_pieces(data, lookback_bars)
        if not columns:
            return {}

        df = pl.DataFrame(columns)
        result_df = df.lazy().select(exprs).tail(1).collect()
        polars_result = result_df.to_dicts()[0]

        features: Dict[str, float] = {}
        for k, v in polars_result.items():
            features[k] = float(v) if v is not None else np.nan
        features.update(layer2)

        # QA 処理
        # e1e_sample_weight は学習側 base_columns 扱いで QA 対象外
        if qa_state is not None:
            qa_result: Dict[str, float] = {}
            for key, val in features.items():
                if key == "e1e_sample_weight":
                    qa_result[key] = val
                else:
                    qa_result[key] = qa_state.update_and_clip(key, val)
            features = qa_result
        else:
            for key in list(features.keys()):
                if key != "e1e_sample_weight" and not np.isfinite(features[key]):
                    features[key] = 0.0

        return features
