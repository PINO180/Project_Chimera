# realtime_feature_engine_1E_signal.py
# Project Cimera V5: Feature Engine Module 1E (Signal / Spectral / Wavelet / Hilbert)

import numpy as np
import numba as nb
from numba import njit
from typing import Dict, Any
import polars as pl

# ==================================================================
# 1. 基盤関数 (Base Functions)
# ==================================================================


@nb.njit(fastmath=True, cache=True)
def numba_fft(x: np.ndarray) -> np.ndarray:

    n = x.shape[0]

    # nが2のべき乗でない場合、ゼロパディングを行う
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


# ==================================================================
# 2. スペクトル系 JIT関数群
# ==================================================================


@nb.njit(fastmath=True, cache=True, parallel=True)
def spectral_centroid_udf(signal: np.ndarray, window_size: int) -> np.ndarray:

    n = len(signal)
    result = np.full(n, np.nan)

    if n < window_size:
        return result

    num_iterations = n - (window_size - 1)

    # 仕事量が多い場合のみ、prangeで並列化する
    if num_iterations > 2000:
        for i in nb.prange(window_size - 1, n):
            window_data = signal[i - window_size + 1 : i + 1]
            finite_data = window_data[np.isfinite(window_data)]

            if len(finite_data) < window_size // 2:
                continue

            # NumbaネイティブFFTによるスペクトル計算
            fft_values = numba_fft(finite_data)
            magnitude_spectrum = np.abs(fft_values[: len(finite_data) // 2])

            if len(magnitude_spectrum) == 0:
                continue

            # 周波数ビンの作成
            freqs = np.linspace(0, 0.5, len(magnitude_spectrum))

            # スペクトル重心計算
            total_magnitude = np.sum(magnitude_spectrum)
            if total_magnitude > 0:
                centroid = np.sum(freqs * magnitude_spectrum) / total_magnitude
                result[i] = centroid
    else:
        # 仕事量が少ない場合は、オーバーヘッドのない通常のforループで処理
        for i in range(window_size - 1, n):
            window_data = signal[i - window_size + 1 : i + 1]
            finite_data = window_data[np.isfinite(window_data)]

            if len(finite_data) < window_size // 2:
                continue

            # NumbaネイティブFFTによるスペクトル計算
            fft_values = numba_fft(finite_data)
            magnitude_spectrum = np.abs(fft_values[: len(finite_data) // 2])

            if len(magnitude_spectrum) == 0:
                continue

            # 周波数ビンの作成
            freqs = np.linspace(0, 0.5, len(magnitude_spectrum))

            # スペクトル重心計算
            total_magnitude = np.sum(magnitude_spectrum)
            if total_magnitude > 0:
                centroid = np.sum(freqs * magnitude_spectrum) / total_magnitude
                result[i] = centroid

    return result


@nb.njit(fastmath=True, cache=True, parallel=True)
def spectral_flatness_udf(signal: np.ndarray, window_size: int) -> np.ndarray:

    n = len(signal)
    result = np.full(n, np.nan)

    if n < window_size:
        return result

    num_iterations = n - (window_size - 1)

    # 仕事量が多い場合のみ、prangeで並列化する
    if num_iterations > 2000:
        for i in nb.prange(window_size - 1, n):
            window_data = signal[i - window_size + 1 : i + 1]
            finite_data = window_data[np.isfinite(window_data)]

            if len(finite_data) < window_size // 2:
                continue

            # NumbaネイティブFFTによるスペクトル計算
            fft_values = numba_fft(finite_data)
            magnitude_spectrum = np.abs(fft_values[: len(finite_data) // 2])

            if len(magnitude_spectrum) == 0:
                continue

            # 0に近い値を避けるため小さな値を加算
            magnitude_spectrum = magnitude_spectrum + 1e-10

            # 幾何平均と算術平均
            geometric_mean = np.exp(np.mean(np.log(magnitude_spectrum)))
            arithmetic_mean = np.mean(magnitude_spectrum)

            if arithmetic_mean > 0:
                flatness = geometric_mean / arithmetic_mean
                result[i] = flatness
    else:
        # 仕事量が少ない場合は、オーバーヘッドのない通常のforループで処理
        for i in range(window_size - 1, n):
            window_data = signal[i - window_size + 1 : i + 1]
            finite_data = window_data[np.isfinite(window_data)]

            if len(finite_data) < window_size // 2:
                continue

            # NumbaネイティブFFTによるスペクトル計算
            fft_values = numba_fft(finite_data)
            magnitude_spectrum = np.abs(fft_values[: len(finite_data) // 2])

            if len(magnitude_spectrum) == 0:
                continue

            # 0に近い値を避けるため小さな値を加算
            magnitude_spectrum = magnitude_spectrum + 1e-10

            # 幾何平均と算術平均
            geometric_mean = np.exp(np.mean(np.log(magnitude_spectrum)))
            arithmetic_mean = np.mean(magnitude_spectrum)

            if arithmetic_mean > 0:
                flatness = geometric_mean / arithmetic_mean
                result[i] = flatness

    return result


# ==================================================================
# 3. ウェーブレット系 JIT関数群
# ==================================================================


@nb.njit(fastmath=True, cache=True, parallel=True)
def wavelet_entropy_udf(signal: np.ndarray, window_size: int) -> np.ndarray:

    n = len(signal)
    result = np.full(n, np.nan)
    num_iterations = n - (window_size - 1)

    if num_iterations > 2000:
        for i in nb.prange(window_size - 1, n):
            window_data = signal[i - window_size + 1 : i + 1]
            if len(window_data) > 0:
                squared_data = window_data**2
                result[i] = -np.sum(
                    squared_data * np.log2(np.abs(squared_data) + 1e-10)
                )
    else:
        for i in range(window_size - 1, n):
            window_data = signal[i - window_size + 1 : i + 1]
            if len(window_data) > 0:
                squared_data = window_data**2
                result[i] = -np.sum(
                    squared_data * np.log2(np.abs(squared_data) + 1e-10)
                )

    return result


# ==================================================================
# 4. ヒルベルト変換系 JIT関数群
# ==================================================================


@nb.njit(fastmath=True, cache=True, parallel=True)
def hilbert_phase_var_udf(signal: np.ndarray, window_size: int) -> np.ndarray:

    n = len(signal)
    result = np.full(n, np.nan)
    num_iterations = n - (window_size - 1)

    loop_range = (
        nb.prange(window_size - 1, n)
        if num_iterations > 2000
        else range(window_size - 1, n)
    )
    for i in loop_range:
        window_data = signal[i - window_size + 1 : i + 1]
        if len(window_data) > 1:
            # 位相計算の近似
            analytic_signal = window_data + 1j * np.roll(window_data, 1)
            result[i] = np.var(np.angle(analytic_signal))

    return result


@nb.njit(fastmath=True, cache=True, parallel=True)
def hilbert_phase_stability_udf(signal: np.ndarray, window_size: int) -> np.ndarray:

    n = len(signal)
    result = np.full(n, np.nan)
    num_iterations = n - (window_size - 1)

    loop_range = (
        nb.prange(window_size - 1, n)
        if num_iterations > 2000
        else range(window_size - 1, n)
    )
    for i in loop_range:
        window_data = signal[i - window_size + 1 : i + 1]
        if len(window_data) > 2:
            # データの標準偏差をチェック
            if np.std(window_data) < 1e-10:
                # 全て同じ値の場合、位相は完全に安定していると見なし、最大値1.0を返す
                result[i] = 1.0
                continue

            analytic_signal = window_data + 1j * np.roll(window_data, 1)
            phase_diff_std = np.std(np.diff(np.angle(analytic_signal)))

            result[i] = 1.0 / (1.0 + phase_diff_std + 1e-10)

    return result


# ==================================================================
# 5. 音響系 JIT関数群
# ==================================================================


@nb.njit(fastmath=True, cache=True, parallel=True)
def acoustic_power_udf(signal: np.ndarray, window_size: int) -> np.ndarray:

    n = len(signal)
    result = np.full(n, np.nan)

    if n < window_size:
        return result

    num_iterations = n - (window_size - 1)

    # 仕事量が多い場合のみ、prangeで並列化する
    if num_iterations > 2000:
        for i in nb.prange(window_size - 1, n):
            window_data = signal[i - window_size + 1 : i + 1]
            finite_data = window_data[np.isfinite(window_data)]

            if len(finite_data) < window_size // 2:
                continue

            # RMSパワー計算
            rms_power = np.sqrt(np.mean(finite_data**2))
            result[i] = rms_power
    else:
        # 仕事量が少ない場合は、オーバーヘッドのない通常のforループで処理
        for i in range(window_size - 1, n):
            window_data = signal[i - window_size + 1 : i + 1]
            finite_data = window_data[np.isfinite(window_data)]

            if len(finite_data) < window_size // 2:
                continue

            # RMSパワー計算
            rms_power = np.sqrt(np.mean(finite_data**2))
            result[i] = rms_power

    return result


# ==================================================================
# メイン計算モジュール
# ==================================================================


def _window(arr: np.ndarray, window: int) -> np.ndarray:

    if window <= 0:
        return np.array([], dtype=arr.dtype)
    if window > len(arr):
        return arr
    return arr[-window:]


def _last(arr: np.ndarray) -> float:

    if len(arr) == 0:
        return np.nan
    return float(arr[-1])


def _pct_change(arr: np.ndarray) -> np.ndarray:

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


class FeatureModule1E:
    @staticmethod
    def calculate_features(data: Dict[str, np.ndarray]) -> Dict[str, float]:
        features = {}

        # 安全な参照用変数
        close_arr = data["close"]

        if len(close_arr) == 0:
            return features

        # pct_change (全特徴量の前処理)
        close_pct = _pct_change(close_arr)

        # ---------------------------------------------------------
        # 1. 音響系指標 (Acoustic)
        # ---------------------------------------------------------
        features["e1e_acoustic_power_128"] = _last(
            acoustic_power_udf(_window(close_pct, 128), 128)
        )

        # ---------------------------------------------------------
        # 2. ヒルベルト系指標 (Hilbert)
        # ---------------------------------------------------------
        features["e1e_hilbert_phase_stability_50"] = _last(
            hilbert_phase_stability_udf(_window(close_pct, 50), 50)
        )
        features["e1e_hilbert_phase_var_50"] = _last(
            hilbert_phase_var_udf(_window(close_pct, 50), 50)
        )

        # ---------------------------------------------------------
        # 3. 信号統計系指標 (Signal Stats)
        # ---------------------------------------------------------
        # ▼▼ 修正前: 配列長チェックがないためエラーや警告のリスク
        # features["e1e_signal_peak_to_peak_100"] = np.max(
        #     _window(close_arr, 100)
        # ) - np.min(_window(close_arr, 100))
        #
        # features["e1e_signal_rms_50"] = np.sqrt(np.mean(_window(close_pct, 50) ** 2))

        # ▼▼ 修正後: 最低要素数のチェックを追加
        w_sig_100 = _window(close_arr, 100)
        features["e1e_signal_peak_to_peak_100"] = (
            float(np.max(w_sig_100) - np.min(w_sig_100))
            if len(w_sig_100) >= 1
            else np.nan
        )

        w_rms_50 = _window(close_pct, 50)
        features["e1e_signal_rms_50"] = (
            float(np.sqrt(np.mean(w_rms_50**2))) if len(w_rms_50) >= 1 else np.nan
        )

        # ---------------------------------------------------------
        # 4. スペクトル系指標 (Spectral)
        # ---------------------------------------------------------
        features["e1e_spectral_centroid_128"] = _last(
            spectral_centroid_udf(_window(close_pct, 128), 128)
        )
        features["e1e_spectral_flatness_128"] = _last(
            spectral_flatness_udf(_window(close_pct, 128), 128)
        )
        # ▼▼ 修正前: 配列長チェックがない
        # features["e1e_spectral_energy_64"] = np.sum(_window(close_pct, 64) ** 2)
        # features["e1e_spectral_energy_128"] = np.sum(_window(close_pct, 128) ** 2)
        # features["e1e_spectral_energy_512"] = np.sum(_window(close_pct, 512) ** 2)

        # ▼▼ 修正後: 配列長チェックを追加
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
        # ---------------------------------------------------------
        features["e1e_wavelet_entropy_64"] = _last(
            wavelet_entropy_udf(_window(close_pct, 64), 64)
        )
        # ▼▼ 修正前: 配列長チェックがない
        # features["e1e_wavelet_mean_256"] = np.mean(_window(close_pct, 256))
        #
        # # [QA確認済] Polars準拠の不偏標準偏差 (ddof=1)
        # features["e1e_wavelet_std_32"] = np.std(_window(close_pct, 32), ddof=1)
        # features["e1e_wavelet_std_64"] = np.std(_window(close_pct, 64), ddof=1)
        # features["e1e_wavelet_std_128"] = np.std(_window(close_pct, 128), ddof=1)
        # features["e1e_wavelet_std_256"] = np.std(_window(close_pct, 256), ddof=1)

        # ▼▼ 修正後: 平均は長1以上、標準偏差は長2以上の制約を追加
        w_wav_256 = _window(close_pct, 256)
        features["e1e_wavelet_mean_256"] = (
            float(np.mean(w_wav_256)) if len(w_wav_256) >= 1 else np.nan
        )

        w_wav_32 = _window(close_pct, 32)
        features["e1e_wavelet_std_32"] = (
            float(np.std(w_wav_32, ddof=1)) if len(w_wav_32) >= 2 else np.nan
        )

        w_wav_64 = _window(close_pct, 64)
        features["e1e_wavelet_std_64"] = (
            float(np.std(w_wav_64, ddof=1)) if len(w_wav_64) >= 2 else np.nan
        )

        w_wav_128 = _window(close_pct, 128)
        features["e1e_wavelet_std_128"] = (
            float(np.std(w_wav_128, ddof=1)) if len(w_wav_128) >= 2 else np.nan
        )

        features["e1e_wavelet_std_256"] = (
            float(np.std(w_wav_256, ddof=1)) if len(w_wav_256) >= 2 else np.nan
        )

        return features
