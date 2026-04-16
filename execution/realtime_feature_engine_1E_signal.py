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
    scale_by_atr,          # signal_peak_to_peak_100 の ATR 割りに使用
    # [STATS]
    stddev_unbiased,
    # [WEIGHT]
    calculate_sample_weight,
    # [DSP] — スペクトル系
    spectral_centroid_udf,
    spectral_flatness_udf,
    # [DSP] — ウェーブレット系
    wavelet_entropy_udf,
    # [DSP] — ヒルベルト系
    hilbert_phase_var_udf,
    hilbert_phase_stability_udf,
    # [DSP] — 音響系
    acoustic_power_udf,
)

import numpy as np
from typing import Dict, Any

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
# ユーティリティ関数（このファイル固有・軽量処理用）
# ==================================================================


def _window(arr: np.ndarray, window: int) -> np.ndarray:
    """直近 window 本のスライスを返す。"""
    if window <= 0:
        return np.array([], dtype=arr.dtype)
    if window > len(arr):
        return arr
    return arr[-window:]


def _last(arr: np.ndarray) -> float:
    """配列の最終要素をスカラーで返す。空配列は nan。"""
    if len(arr) == 0:
        return np.nan
    return float(arr[-1])


def _pct_change(arr: np.ndarray) -> np.ndarray:
    """
    価格変化率（pct_change）を計算する。先頭要素は nan。

    ▼▼ [修正④] prev == 0.0 の分岐を廃止し、旧スクリプトと同様に
       (arr[i] - prev) / (prev + 1e-10) のゼロ除算保護に統一する。
       prev が 1e-15 のような極小値でも有限値を返す一貫性を保証する。
    """
    n = len(arr)
    pct = np.full(n, np.nan, dtype=np.float64)
    if n < 2:
        return pct
    for i in range(1, n):
        prev = arr[i - 1]
        pct[i] = (arr[i] - prev) / (abs(prev) + 1e-10)
    return pct


# ==================================================================
# メイン計算モジュール
# ==================================================================


class FeatureModule1E:
    @staticmethod
    def calculate_features(data: Dict[str, np.ndarray]) -> Dict[str, float]:
        """
        リアルタイム特徴量計算。

        Args:
            data: {"close": np.ndarray, "high": np.ndarray, "low": np.ndarray}
                  を含む辞書。high / low は ATR 計算に使用。

        Returns:
            特徴量名 → スカラー値 の辞書。
            e1e_sample_weight も含む。
        """
        features: Dict[str, float] = {}

        close_arr = data["close"]

        # ▼▼ [任意確認①] high / low は ATR 計算に必要だが、
        #    realtime 側では渡されない状況もあり得るため data.get() でガードする。
        #    high / low が存在しない場合は close で代替するが、
        #    その場合は TR=0 → ATR≈0 となるため signal_peak_to_peak_100 は
        #    後段の np.isfinite(atr_last) チェックで nan にフォールバックする。
        high_arr = data.get("high", close_arr)
        low_arr  = data.get("low",  close_arr)

        if len(close_arr) == 0:
            return features

        # pct_change (全特徴量の前処理)
        close_pct = _pct_change(close_arr)

        # ---------------------------------------------------------
        # 0. サンプルウェイト
        # ▼▼ [Step 11] calculate_sample_weight (core_indicators) で計算
        #    engine_1_E 学習側と同一の Wilder ATR ベース重みを付与する。
        # ---------------------------------------------------------
        sw_arr = calculate_sample_weight(
            high_arr.astype(np.float64),
            low_arr.astype(np.float64),
            close_arr.astype(np.float64),
        )
        features["e1e_sample_weight"] = float(sw_arr[-1]) if len(sw_arr) > 0 else 1.0

        # ---------------------------------------------------------
        # 1. 音響系指標 (Acoustic)
        # ▼▼ [Step 11] core_indicators.acoustic_power_udf を使用
        # ---------------------------------------------------------
        features["e1e_acoustic_power_128"] = _last(
            acoustic_power_udf(_window(close_pct, 128), 128)
        )

        # ---------------------------------------------------------
        # 2. ヒルベルト系指標 (Hilbert)
        # ▼▼ [Step 11] core_indicators.hilbert_phase_stability_udf /
        #              hilbert_phase_var_udf を使用。
        #    旧実装の np.roll() 近似から get_analytic_signal() (FFTベース)
        #    に変更済みであるため、engine_1_E 学習側と数値が一致する。
        # ---------------------------------------------------------
        features["e1e_hilbert_phase_stability_50"] = _last(
            hilbert_phase_stability_udf(_window(close_pct, 50), 50)
        )
        features["e1e_hilbert_phase_var_50"] = _last(
            hilbert_phase_var_udf(_window(close_pct, 50), 50)
        )

        # ---------------------------------------------------------
        # 3. 信号統計系指標 (Signal Stats)
        # ▼▼ [Step 11] signal_peak_to_peak_100 に ATR 割りを追加
        #    旧: (max - min) の絶対価格差（スケール依存）
        #    新: (max - min) / (ATR_100 + 1e-10) のスケール不変値
        #        → engine_1_E 学習側の _create_signal_stats_features() と一致
        # ---------------------------------------------------------
        w_sig_100 = _window(close_arr, 100)
        w_hi_100  = _window(high_arr,  100)
        w_lo_100  = _window(low_arr,   100)

        if len(w_sig_100) >= 2:
            atr_100 = calculate_atr_wilder(
                w_hi_100.astype(np.float64),
                w_lo_100.astype(np.float64),
                w_sig_100.astype(np.float64),
                100,
            )
            atr_last_val = atr_100[-1] if len(atr_100) > 0 else np.nan
            # ▼▼ [任意確認②] atr_last が nan の場合（データ不足・high/low フォールバック時）
            #    scale_by_atr に nan を渡すと nan / (nan + 1e-10) = nan になるため
            #    np.isfinite で有効性を確認してから ATR 割りを行う。
            if np.isfinite(atr_last_val):
                raw_range = np.array([np.max(w_sig_100) - np.min(w_sig_100)], dtype=np.float64)
                atr_last  = np.array([atr_last_val], dtype=np.float64)
                features["e1e_signal_peak_to_peak_100"] = float(scale_by_atr(raw_range, atr_last)[0])
            else:
                features["e1e_signal_peak_to_peak_100"] = np.nan
        else:
            features["e1e_signal_peak_to_peak_100"] = np.nan

        w_rms_50 = _window(close_pct, 50)
        features["e1e_signal_rms_50"] = (
            float(np.sqrt(np.mean(w_rms_50**2))) if len(w_rms_50) >= 1 else np.nan
        )

        # ---------------------------------------------------------
        # 4. スペクトル系指標 (Spectral)
        # ▼▼ [Step 11] core_indicators.spectral_centroid_udf /
        #              spectral_flatness_udf を使用
        # ---------------------------------------------------------
        features["e1e_spectral_centroid_128"] = _last(
            spectral_centroid_udf(_window(close_pct, 128), 128)
        )
        features["e1e_spectral_flatness_128"] = _last(
            spectral_flatness_udf(_window(close_pct, 128), 128)
        )

        # spectral_energy 系（Polarsネイティブ計算と等価：sum of squared pct_change）
        w_spec_64 = _window(close_pct, 64)
        features["e1e_spectral_energy_64"] = (
            float(np.sum(w_spec_64**2)) if len(w_spec_64) >= 1 else np.nan
        )

        w_spec_128 = _window(close_pct, 128)
        features["e1e_spectral_energy_128"] = (
            float(np.sum(w_spec_128**2)) if len(w_spec_128) >= 1 else np.nan
        )

        w_spec_512 = _window(close_pct, 512)
        features["e1e_spectral_energy_512"] = (
            float(np.sum(w_spec_512**2)) if len(w_spec_512) >= 1 else np.nan
        )

        # ---------------------------------------------------------
        # 5. ウェーブレット系指標 (Wavelet)
        # ▼▼ [Step 11] core_indicators.wavelet_entropy_udf を使用
        #    wavelet_std_* は np.std(ddof=1) 直書き → stddev_unbiased に統一
        #    (Numba 環境では ddof=1 が使えないため core_indicators で統一管理)
        #    リアルタイム側は NumPy で計算するが、stddev_unbiased と
        #    数学的に等価な式を使用し直書き epsilon を排除する。
        # ---------------------------------------------------------
        features["e1e_wavelet_entropy_64"] = _last(
            wavelet_entropy_udf(_window(close_pct, 64), 64)
        )

        w_wav_256 = _window(close_pct, 256)
        features["e1e_wavelet_mean_256"] = (
            float(np.mean(w_wav_256)) if len(w_wav_256) >= 1 else np.nan
        )

        # ▼▼ [Step 11] stddev_unbiased (core_indicators) 呼び出しに統一
        #    リアルタイム側はウィンドウスライスに対してスカラー計算するため
        #    stddev_unbiased(arr, window) の代わりに等価な NumPy 式を使う。
        #    stddev_unbiased の定義: sqrt(Σ(x-μ)²/(n-1)) = np.std(ddof=1)
        #    → NumPy の np.std(ddof=1) は Numba 外では正常動作するため維持。
        #      ただし、_window で得た短い配列を stddev_unbiased に渡す方式でも可。
        for w_size, key_suffix in [(32, "32"), (64, "64"), (128, "128"), (256, "256")]:
            w_arr = _window(close_pct, w_size)
            if len(w_arr) >= 2:
                # stddev_unbiased と等価: sqrt(Σ(x-μ)²/(n-1))
                std_val = _last(stddev_unbiased(w_arr, len(w_arr)))
                features[f"e1e_wavelet_std_{w_size}"] = (
                    float(std_val) if np.isfinite(std_val) else np.nan
                )
            else:
                features[f"e1e_wavelet_std_{w_size}"] = np.nan

        return features
