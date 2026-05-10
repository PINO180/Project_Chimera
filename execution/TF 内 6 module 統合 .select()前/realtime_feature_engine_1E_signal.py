# realtime_feature_engine_1E_signal.py
# Category 1E: 信号処理系 (Spectral / Wavelet / Hilbert / Acoustic / Signal Stats)
#
# ==================================================================
# 【Step B 改修】numpy ベース → Polars 直呼び (ローリング統計部分のみ)
# ==================================================================
#
# 目的: 学習側 engine_1_E_a_vast_universe_of_features.py と
#       「同じ Polars Rust エンジン」で計算することで、ビット完全一致を達成する。
#
# 【設計判断】
#   1E のボトルネックは、(a) Polars でなく numpy で行っていたローリング統計
#   (rolling_mean / rolling_std / rolling_max / rolling_min / rolling_sum) と、
#   (b) atr_wilder の二度引き等。
#   一方で DSP UDF (spectral_*, hilbert_*, wavelet_*, acoustic_*) は
#   それ自体が O(window²) の FFT 計算で重く、最後の window 本のみ
#   渡せば最終バーの値が決まるため、numpy 直接呼びが最適。
#
#   学習側 engine_1_E は `pl.col("close").pct_change().map_batches(udf, ...)`
#   で全系列を UDF に流すが、これは「全バーの値が必要だから」であって
#   本番側は最終バーのみ計算する。UDF の数式は両者で完全同一なので、
#   `udf(close_pct[-window:])[-1] == udf(close_pct)[-1]` (rolling 性質より)。
#
#   結果:
#     - DSP UDF 群: numpy で close_pct[-window:] を渡して直接呼び出し (旧版と同様)
#     - ローリング統計: Polars 単一 .select() に集約 (新規追加)
#     - ATR13 / ATR100: numpy で 1 度だけ計算 (旧版は同じ ATR を 2-3 度計算)
#
# 【SSoT 階層】
#   Layer 1 (Polars rolling 統計):  Polars Rust エンジン (両側共通)
#   Layer 2 (DSP/Numba UDF):       core_indicators から import (Phase 5 確立済)
#
# 【保持される過去の修正】
#   ・QAState (apply_quality_assurance_to_group の等価実装)
#   ・ewm_std bias=False 補正の Polars 互換式
#   ・hilbert_phase_*_udf は core_indicators の FFT-Hilbert 厳密実装を使用
#   ・e1e_sample_weight は QA 対象外 (学習側 base_columns 扱いと一致)
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
from typing import Dict, Optional


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
# ==================================================================

class QAState:
    """
    学習側 apply_quality_assurance_to_group のリアルタイム等価実装。

    学習側の処理 (Polars LazyFrame 全系列一括):
        safe_col = when(col.is_infinite()).then(None).otherwise(col).fill_nan(None)
        ewm_mean = safe_col.ewm_mean(half_life=HL, adjust=False, ignore_nulls=True)
                                 .forward_fill()
        ewm_std  = safe_col.ewm_std (half_life=HL, adjust=False, ignore_nulls=True)
                                 .forward_fill()
        result   = when(col==inf).then(p99).when(col==-inf).then(p01)
                   .otherwise(col).clip(p01, p99).fill_null(0.0).fill_nan(0.0)

    Polars ewm_mean(adjust=False) の再帰式 (alpha = 1 - exp(-ln2 / HL)):
        EWM_mean[t] = alpha * x[t] + (1-alpha) * EWM_mean[t-1]

    Polars ewm_var(adjust=False) の再帰式:
        EWM_var[t] = (1-alpha) * (EWM_var[t-1] + alpha * (x[t] - EWM_mean[t-1])^2)
        EWM_std[t] = sqrt(EWM_var[t]) * bias_correction

    bias_correction (Polars ewm_std(adjust=False, bias=False) と完全一致):
        adjust=False の重み: w_k = alpha*(1-alpha)^k (k=0..n-2) と
                            w_{n-1} = (1-alpha)^(n-1) (最古項を正規化保持)
        sum_w  = 1, sum_w2 = 重みの2乗和
        bias_factor_var = 1 / (1 - sum_w2)
          r2     = (1 - alpha)^2
          m      = n - 1
          sum_w2 = alpha^2 * (1 - r2^m) / (1 - r2) + r2^m   (m >= 1)
          ewm_std = sqrt(ewm_var * bias_factor_var)
    """

    def __init__(self, lookback_bars: int = 1440):
        self.alpha: float = 1.0 - np.exp(-np.log(2.0) / max(lookback_bars, 1))
        self._ewm_mean: Dict[str, float] = {}
        self._ewm_var: Dict[str, float] = {}
        self._ewm_n: Dict[str, int] = {}

    def update_and_clip(self, key: str, raw_val: float) -> float:
        """1特徴量の raw_val に対して QA処理を適用し、処理済みスカラーを返す。"""
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
    def calculate_features(
        data: Dict[str, np.ndarray],
        lookback_bars: int = 1440,
        qa_state: Optional[QAState] = None,
    ) -> Dict[str, float]:
        """
        【Step B改修版】Polars 直呼び + 軽量 DSP UDF 直接呼び出し。

        アーキテクチャ:
            Layer 1 (Polars 単一 .select(), rolling 統計のみ):
                spectral_energy / spectral_peak_freq_128 / wavelet_mean,std /
                hilbert_amp_mean,std,cv / hilbert_freq_energy_ratio /
                signal_rms / signal_peak_to_peak / signal_crest_factor
                → Polars Rust エンジンで一括計算 (学習側と同一の rolling 式)。
            Layer 2 (Numba DSP UDF, 直接呼び出し):
                spectral_centroid/bandwidth/rolloff/flux/flatness/entropy +
                wavelet_energy,entropy + hilbert_amplitude/phase/freq +
                acoustic_power/frequency
                → close_pct[-window:] を渡して呼び出し、最終バーの値を取得。
                  学習側は同一 UDF を全系列に適用するが、UDF の rolling 性質より
                  最終バーは window 本のみで決まるため数値完全一致。
            Layer 3 (sample_weight, QA対象外):
                calculate_sample_weight(high, low, close)[-1]

        Args:
            data         : close/high/low の numpy 配列を含む辞書 (high/low 任意)
            lookback_bars: QA の EWM 半減期に使用 (1日あたりのバー数)
            qa_state     : QAState インスタンス。本番稼働時は必ず渡し、同一
                           インスタンスを毎バー使い回すこと。
        """
        features: Dict[str, float] = {}

        close_arr = data["close"].astype(np.float64)
        high_arr  = (
            data["high"].astype(np.float64) if "high" in data and len(data["high"]) > 0
            else close_arr
        )
        low_arr   = (
            data["low"].astype(np.float64) if "low" in data and len(data["low"]) > 0
            else close_arr
        )

        if len(close_arr) == 0:
            return features

        # ---------------------------------------------------------
        # ATR 系列の事前計算 (学習側 atr_13_expr_hilbert / atr_100_expr と完全一致)
        #
        # 学習側 (engine_1_E._create_*_features 内):
        #   atr_13_expr_hilbert = pl.struct([...]).map_batches(calculate_atr_wilder(...,13))
        #   atr_100_expr        = pl.struct([...]).map_batches(calculate_atr_wilder(...,100))
        # 本番側: numpy で 1 度だけ計算して __temp_atr_{13,100} 列として渡す。
        # 学習側は割り算時に + 1e-10 を加えるため、ここでは raw ATR を保持する。
        # ---------------------------------------------------------
        atr13_arr  = calculate_atr_wilder(high_arr, low_arr, close_arr, 13)
        atr100_arr = calculate_atr_wilder(high_arr, low_arr, close_arr, 100)

        # close_pct を numpy で 1 度だけ計算 (学習側 Polars pct_change と semantics 一致)
        close_pct = _pct_change(close_arr)

        # ---------------------------------------------------------
        # Layer 1: Polars DataFrame 構築と式リスト
        # 学習側 engine_1_E のうち rolling 統計に該当する式を集約。
        # ---------------------------------------------------------
        df = pl.DataFrame({
            "close":         close_arr,
            "__temp_atr_13":  atr13_arr,
            "__temp_atr_100": atr100_arr,
        })

        exprs = []

        # ===== Spectral group (Polars 部分) =====
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

        # ===== Wavelet group (Polars 部分) =====
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

        # ===== Hilbert group (Polars 部分) =====
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

        # ===== Signal Stats group (Polars 部分) =====
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

        # =====================================================================
        # Polars 単一 .select() で全 rolling 式を一括計算 (FFI overhead 1回のみ)
        # =====================================================================
        result_df = df.lazy().select(exprs).tail(1).collect()
        polars_result = result_df.to_dicts()[0]

        for k, v in polars_result.items():
            features[k] = float(v) if v is not None else np.nan

        # =====================================================================
        # Layer 2: DSP Numba UDF 群 (close_pct[-window:] に直接呼び出し)
        #
        # 各 UDF は rolling 計算であり、最終バー (index = window-1 in slice) の
        # 値は直近 window 本のみで決まる。学習側は全系列に対して UDF を呼び、
        # その最終要素を採用するが、本番側は最終 window 本のみ渡しても同一値。
        # =====================================================================
        n = len(close_pct)

        # ----- Spectral UDFs (window=[64,128,256,512]) -----
        # 参照: engine_1_E._create_spectral_features L1098-1163
        for window in [64, 128, 256, 512]:
            if n >= window:
                w_arr = close_pct[-window:]
                features[f"e1e_spectral_centroid_{window}"]  = float(spectral_centroid_udf(w_arr, window)[-1])
                features[f"e1e_spectral_bandwidth_{window}"] = float(spectral_bandwidth_udf(w_arr, window)[-1])
                features[f"e1e_spectral_rolloff_{window}"]   = float(spectral_rolloff_udf(w_arr, window)[-1])
                features[f"e1e_spectral_flatness_{window}"]  = float(spectral_flatness_udf(w_arr, window)[-1])
                features[f"e1e_spectral_entropy_{window}"]   = float(spectral_entropy_udf(w_arr, window)[-1])
            else:
                features[f"e1e_spectral_centroid_{window}"]  = np.nan
                features[f"e1e_spectral_bandwidth_{window}"] = np.nan
                features[f"e1e_spectral_rolloff_{window}"]   = np.nan
                features[f"e1e_spectral_flatness_{window}"]  = np.nan
                features[f"e1e_spectral_entropy_{window}"]   = np.nan

            # spectral_flux は隣接 2 フレーム必要 (window*2 本)
            if n >= window * 2:
                w_arr2 = close_pct[-(window * 2):]
                features[f"e1e_spectral_flux_{window}"] = float(spectral_flux_udf(w_arr2, window)[-1])
            else:
                features[f"e1e_spectral_flux_{window}"] = np.nan

        # ----- Wavelet UDFs -----
        # 参照: engine_1_E._create_wavelet_features L1190-1227
        for window in [32, 64, 128, 256]:
            if n >= window:
                features[f"e1e_wavelet_energy_{window}"] = float(
                    wavelet_energy_udf(close_pct[-window:], window)[-1]
                )
            else:
                features[f"e1e_wavelet_energy_{window}"] = np.nan

        # wavelet_entropy_64
        if n >= 64:
            features["e1e_wavelet_entropy_64"] = float(
                wavelet_entropy_udf(close_pct[-64:], 64)[-1]
            )
        else:
            features["e1e_wavelet_entropy_64"] = np.nan

        # ----- Hilbert UDFs -----
        # 参照: engine_1_E._create_hilbert_features L1237-1296
        for window in [50, 100, 200]:
            if n >= window:
                features[f"e1e_hilbert_amplitude_{window}"] = float(
                    hilbert_amplitude_udf(close_pct[-window:], window)[-1]
                )
            else:
                features[f"e1e_hilbert_amplitude_{window}"] = np.nan

        # phase_var_50, phase_stability_50
        if n >= 50:
            features["e1e_hilbert_phase_var_50"]       = float(hilbert_phase_var_udf(close_pct[-50:], 50)[-1])
            features["e1e_hilbert_phase_stability_50"] = float(hilbert_phase_stability_udf(close_pct[-50:], 50)[-1])
        else:
            features["e1e_hilbert_phase_var_50"]       = np.nan
            features["e1e_hilbert_phase_stability_50"] = np.nan

        # freq_mean_100, freq_std_100
        if n >= 100:
            features["e1e_hilbert_freq_mean_100"] = float(hilbert_freq_mean_udf(close_pct[-100:], 100)[-1])
            features["e1e_hilbert_freq_std_100"]  = float(hilbert_freq_std_udf(close_pct[-100:], 100)[-1])
        else:
            features["e1e_hilbert_freq_mean_100"] = np.nan
            features["e1e_hilbert_freq_std_100"]  = np.nan

        # ----- Acoustic UDFs (window=[128,256,512]) -----
        # 参照: engine_1_E._create_acoustic_features L1352-1373
        for window in [128, 256, 512]:
            if n >= window:
                w_arr = close_pct[-window:]
                features[f"e1e_acoustic_power_{window}"]     = float(acoustic_power_udf(w_arr, window)[-1])
                features[f"e1e_acoustic_frequency_{window}"] = float(acoustic_frequency_udf(w_arr, window)[-1])
            else:
                features[f"e1e_acoustic_power_{window}"]     = np.nan
                features[f"e1e_acoustic_frequency_{window}"] = np.nan

        # =====================================================================
        # Layer 3: サンプルウェイト (学習側 base_columns 扱いと一致、QA対象外)
        # 参照: engine_1_E L1733-1742
        # =====================================================================
        sample_weight_arr = calculate_sample_weight(high_arr, low_arr, close_arr)
        features["e1e_sample_weight"] = (
            float(sample_weight_arr[-1]) if len(sample_weight_arr) > 0 else 1.0
        )

        # =====================================================================
        # QA 処理 (学習側 apply_quality_assurance_to_group と等価)
        # e1e_sample_weight は学習側 base_columns 扱いで QA 対象外。
        # qa_state=None の場合は inf/NaN → 0.0 のフォールバックのみ。
        # =====================================================================
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
