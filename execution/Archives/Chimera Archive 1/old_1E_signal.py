# realtime_feature_engine_1E_signal.py
# Project Cimera V5: Feature Engine Module 1E (Signal / Spectral / Wavelet / Hilbert)

import numpy as np
import numba as nb
from numba import njit
from typing import Dict

# ==================================================================
# 1. 基盤関数 (Base Functions)
# ==================================================================


@njit(fastmath=True, cache=True)
def numba_fft(x: np.ndarray) -> np.ndarray:
    """
    NumbaネイティブFFT実装 [cite: 1]
    np.fftがNumbaでサポートされていないための代替実装 [cite: 1]
    """
    n = x.shape[0]

    # nが2のべき乗でない場合、ゼロパディングを行う [cite: 1]
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

    # ビット反転置換 [cite: 1]
    j = 0
    for i in range(1, n):
        bit = n >> 1
        while j & bit:
            j ^= bit
            bit >>= 1
        j ^= bit
        if i < j:
            x[i], x[j] = x[j], x[i]

    # バタフライ演算（反復処理） [cite: 1]
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
# 2. スペクトル系・音響系 JIT関数群
# ==================================================================


@njit(fastmath=True, cache=True)
def spectral_centroid_udf(signal: np.ndarray) -> float:
    """
    スペクトル重心計算 (Numba JIT)
    """
    n = len(signal)
    # 有限値のみ抽出
    finite_data = signal[np.isfinite(signal)]

    # データ長チェック (FFTのため最低限の長さが必要)
    if len(finite_data) < 4:
        return np.nan

    # NumbaネイティブFFTによるスペクトル計算
    fft_values = numba_fft(finite_data)
    magnitude_spectrum = np.abs(fft_values[: len(finite_data) // 2])

    if len(magnitude_spectrum) == 0:
        return np.nan

    # 周波数ビンの作成
    freqs = np.linspace(0, 0.5, len(magnitude_spectrum))

    # スペクトル重心計算
    total_magnitude = np.sum(magnitude_spectrum)
    if total_magnitude > 0:
        return np.sum(freqs * magnitude_spectrum) / total_magnitude
    else:
        return 0.0


@njit(fastmath=True, cache=True)
def spectral_bandwidth_udf(signal: np.ndarray) -> float:
    """
    スペクトル帯域幅計算 (Numba JIT)
    """
    n = len(signal)
    finite_data = signal[np.isfinite(signal)]

    if len(finite_data) < 4:
        return np.nan

    fft_values = numba_fft(finite_data)
    magnitude_spectrum = np.abs(fft_values[: len(finite_data) // 2])

    if len(magnitude_spectrum) == 0:
        return np.nan

    freqs = np.linspace(0, 0.5, len(magnitude_spectrum))
    total_magnitude = np.sum(magnitude_spectrum)

    if total_magnitude > 0:
        centroid = np.sum(freqs * magnitude_spectrum) / total_magnitude
        # 帯域幅（重心周りの分散）計算
        bandwidth = np.sqrt(
            np.sum(((freqs - centroid) ** 2) * magnitude_spectrum) / total_magnitude
        )
        return bandwidth
    else:
        return 0.0


@njit(fastmath=True, cache=True)
def spectral_rolloff_udf(signal: np.ndarray) -> float:
    """
    スペクトルロールオフ計算 (Numba JIT)
    """
    n = len(signal)
    finite_data = signal[np.isfinite(signal)]

    if len(finite_data) < 4:
        return np.nan

    fft_values = numba_fft(finite_data)
    magnitude_spectrum = np.abs(fft_values[: len(finite_data) // 2])

    if len(magnitude_spectrum) == 0:
        return np.nan

    # パワースペクトルに変換
    power_spectrum = magnitude_spectrum**2
    total_power = np.sum(power_spectrum)

    if total_power > 0:
        cumulative_power = np.cumsum(power_spectrum)
        threshold = 0.85 * total_power  # rolloff_ratio = 0.85

        rolloff_idx = np.where(cumulative_power >= threshold)[0]
        if len(rolloff_idx) > 0:
            return rolloff_idx[0] / (2.0 * len(magnitude_spectrum))

    return 0.0


@njit(fastmath=True, cache=True)
def spectral_flatness_udf(signal: np.ndarray) -> float:
    """
    スペクトル平坦度計算 (Numba JIT)
    """
    n = len(signal)
    finite_data = signal[np.isfinite(signal)]

    if len(finite_data) < 4:
        return np.nan

    fft_values = numba_fft(finite_data)
    magnitude_spectrum = np.abs(fft_values[: len(finite_data) // 2])

    if len(magnitude_spectrum) == 0:
        return np.nan

    # 0に近い値を避けるため小さな値を加算
    magnitude_spectrum = magnitude_spectrum + 1e-10

    geometric_mean = np.exp(np.mean(np.log(magnitude_spectrum)))
    arithmetic_mean = np.mean(magnitude_spectrum)

    if arithmetic_mean > 0:
        return geometric_mean / arithmetic_mean
    else:
        return 0.0


@njit(fastmath=True, cache=True)
def spectral_entropy_udf(signal: np.ndarray) -> float:
    """
    スペクトルエントロピー計算 (Numba JIT)
    """
    n = len(signal)
    finite_data = signal[np.isfinite(signal)]

    if len(finite_data) < 4:
        return np.nan

    fft_values = numba_fft(finite_data)
    power_spectrum = np.abs(fft_values[: len(finite_data) // 2]) ** 2

    if len(power_spectrum) == 0:
        return np.nan

    total_power = np.sum(power_spectrum)
    if total_power > 0:
        probability = power_spectrum / total_power

        entropy = 0.0
        for p in probability:
            if p > 1e-10:
                entropy -= p * np.log2(p)
        return entropy
    else:
        return 0.0


@njit(fastmath=True, cache=True)
def acoustic_power_udf(signal: np.ndarray) -> float:
    """
    音響パワー (RMS) (Numba JIT)
    """
    finite_data = signal[np.isfinite(signal)]
    if len(finite_data) > 0:
        return np.sqrt(np.mean(finite_data**2))
    return np.nan


@njit(fastmath=True, cache=True)
def acoustic_frequency_udf(signal: np.ndarray) -> float:
    """
    音響周波数 (ゼロクロッシング率) (Numba JIT)
    """
    finite_data = signal[np.isfinite(signal)]
    if len(finite_data) > 1:
        zero_crossings = 0
        for j in range(1, len(finite_data)):
            if finite_data[j - 1] * finite_data[j] < 0:
                zero_crossings += 1

        # 周波数推定 (ゼロクロッシング率 / 2)
        # sample_rateは1.0と仮定
        return zero_crossings / (2.0 * len(finite_data))
    return np.nan


# ==================================================================
# 3. ヒルベルト変換系・ウェーブレット系 JIT関数群
# ==================================================================


@njit(fastmath=True, cache=True)
def hilbert_amplitude_udf(signal: np.ndarray) -> float:
    """
    ヒルベルト変換による振幅包絡線計算 (Numba JIT)
    FFTを使用した近似ヒルベルト変換を行い、その振幅の平均を返す
    """
    finite_data = signal[np.isfinite(signal)]
    n_samples = len(finite_data)

    if n_samples < 4:
        return np.nan

    # 近似ヒルベルト変換（90度位相シフト）
    # FFTを使用した近似実装
    fft_signal = numba_fft(finite_data)

    # 90度位相シフト
    hilbert_fft = fft_signal.copy()
    for j in range(1, n_samples // 2):
        hilbert_fft[j] *= -1j
        hilbert_fft[n_samples - j] *= 1j

    # IFFT相当の処理（簡易版：実部のみ取得）
    # Note: numba_fftはパディングされている可能性があるが、
    # ここでは元のデータ長に合わせて切り詰める必要がある
    hilbert_signal = np.real(hilbert_fft)[:n_samples]

    # 振幅包絡線
    amplitude_envelope = np.sqrt(finite_data**2 + hilbert_signal**2)
    return np.mean(amplitude_envelope)


@njit(fastmath=True, cache=True)
def hilbert_phase_var_udf(signal: np.ndarray) -> float:
    """
    ヒルベルト位相分散 (Numba JIT)
    """
    finite_data = signal[np.isfinite(signal)]
    if len(finite_data) > 1:
        # 解析信号の近似: signal + j * Hilbert(signal)
        # ここでは簡易的に90度位相シフトとして1サンプルシフトを使用
        analytic_signal = finite_data + 1j * np.roll(finite_data, 1)
        return np.var(np.angle(analytic_signal))
    return np.nan


@njit(fastmath=True, cache=True)
def hilbert_phase_stability_udf(signal: np.ndarray) -> float:
    """
    ヒルベルト位相安定性 (Numba JIT)
    """
    finite_data = signal[np.isfinite(signal)]
    if len(finite_data) > 2:
        # データの標準偏差をチェック
        if np.std(finite_data) < 1e-10:
            return 1.0

        analytic_signal = finite_data + 1j * np.roll(finite_data, 1)
        phase_diff_std = np.std(np.diff(np.angle(analytic_signal)))

        return 1.0 / (1.0 + phase_diff_std + 1e-10)
    return np.nan


@njit(fastmath=True, cache=True)
def hilbert_freq_mean_udf(signal: np.ndarray) -> float:
    """
    ヒルベルト瞬時周波数平均 (Numba JIT)
    """
    finite_data = signal[np.isfinite(signal)]
    if len(finite_data) > 2:
        analytic_signal = finite_data + 1j * np.roll(finite_data, 1)
        # np.unwrapはNumba非対応のため単純差分で近似
        instant_freq = np.abs(np.diff(np.angle(analytic_signal)))
        return np.mean(instant_freq)
    return np.nan


@njit(fastmath=True, cache=True)
def hilbert_freq_std_udf(signal: np.ndarray) -> float:
    """
    ヒルベルト瞬時周波数標準偏差 (Numba JIT)
    """
    finite_data = signal[np.isfinite(signal)]
    if len(finite_data) > 2:
        analytic_signal = finite_data + 1j * np.roll(finite_data, 1)
        instant_freq = np.abs(np.diff(np.angle(analytic_signal)))
        return np.std(instant_freq)
    return np.nan


@njit(fastmath=True, cache=True)
def wavelet_entropy_udf(signal: np.ndarray) -> float:
    """
    ウェーブレットエントロピー (Numba JIT)
    信号の二乗値に基づくエントロピー計算
    """
    finite_data = signal[np.isfinite(signal)]
    if len(finite_data) > 0:
        squared_data = finite_data**2
        # np.sum(-p log p) の形式
        return -np.sum(squared_data * np.log2(np.abs(squared_data) + 1e-10))
    return np.nan
