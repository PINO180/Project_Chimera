# realtime_feature_engine_1E_signal.py
# Project Cimera V5: Feature Engine Module 1E (Signal / Spectral / Wavelet / Hilbert)
# 【修正済み】engine_1_E_a_vast_universe_of_features.py との完全一致照合・修正版
#
# 修正内容サマリー:
#   1. 全UDF: @nb.njit(fastmath=True, cache=True, parallel=True) に統一（parallel=True 追加）
#   2. 全UDF: シグネチャを (signal, window_size) に変更 → ローリングウィンドウ計算（配列→配列）
#   3. 全UDF: 動的並列化制御ロジック (num_iterations > 2000 ? prange : range) を復元
#   4. spectral_flux_udf: 欠落していたため追加
#   5. wavelet_energy_udf: 欠落していたため追加
#   6. spectral_rolloff_udf: rolloff_ratio パラメータを復元
#   7. acoustic_frequency_udf: sample_rate パラメータを復元
#   8. wavelet_entropy_udf: window_data をそのまま使用（NaN含む元ロジックに一致）
#   9. hilbert_phase_stability_udf: window_data に対してstdチェック（元ロジックに一致）

import numpy as np
import numba as nb
from numba import njit
from typing import Dict

# ==================================================================
# 1. 基盤関数 (Base Functions)
# ==================================================================


@nb.njit(fastmath=True, cache=True)
def numba_fft(x: np.ndarray) -> np.ndarray:
    """
    Numbaで実装された反復的Cooley-Tukey FFTアルゴリズム（安定版）。
    再帰を排除し、メモリ効率と並列処理への耐性を向上。
    """
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
    """
    スペクトル重心計算（動的並列化制御版）
    周波数スペクトルの重心（平均周波数）を計算
    """
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
def spectral_bandwidth_udf(signal: np.ndarray, window_size: int) -> np.ndarray:
    """
    スペクトル帯域幅計算（動的並列化制御版）
    スペクトル重心周りの分散を計算
    """
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

                # 帯域幅（重心周りの分散）計算
                bandwidth = np.sqrt(
                    np.sum(((freqs - centroid) ** 2) * magnitude_spectrum)
                    / total_magnitude
                )
                result[i] = bandwidth
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

                # 帯域幅（重心周りの分散）計算
                bandwidth = np.sqrt(
                    np.sum(((freqs - centroid) ** 2) * magnitude_spectrum)
                    / total_magnitude
                )
                result[i] = bandwidth

    return result


@nb.njit(fastmath=True, cache=True, parallel=True)
def spectral_rolloff_udf(
    signal: np.ndarray, window_size: int, rolloff_ratio: float = 0.85
) -> np.ndarray:
    """
    スペクトルロールオフ計算（動的並列化制御版）
    累積エネルギーが指定割合に達する周波数
    """
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

            # パワースペクトルに変換
            power_spectrum = magnitude_spectrum**2
            total_power = np.sum(power_spectrum)

            if total_power > 0:
                # 累積パワースペクトル
                cumulative_power = np.cumsum(power_spectrum)
                threshold = rolloff_ratio * total_power

                # ロールオフ周波数の特定
                rolloff_idx = np.where(cumulative_power >= threshold)[0]
                if len(rolloff_idx) > 0:
                    rolloff_freq = rolloff_idx[0] / (2.0 * len(magnitude_spectrum))
                    result[i] = rolloff_freq
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

            # パワースペクトルに変換
            power_spectrum = magnitude_spectrum**2
            total_power = np.sum(power_spectrum)

            if total_power > 0:
                # 累積パワースペクトル
                cumulative_power = np.cumsum(power_spectrum)
                threshold = rolloff_ratio * total_power

                # ロールオフ周波数の特定
                rolloff_idx = np.where(cumulative_power >= threshold)[0]
                if len(rolloff_idx) > 0:
                    rolloff_freq = rolloff_idx[0] / (2.0 * len(magnitude_spectrum))
                    result[i] = rolloff_freq

    return result


@nb.njit(fastmath=True, cache=True, parallel=True)
def spectral_flux_udf(signal: np.ndarray, window_size: int) -> np.ndarray:
    """
    スペクトルフラックス計算（動的並列化制御版）
    連続するフレーム間のスペクトル変化量
    【修正】元スクリプトに存在したが新モジュールに欠落していたため追加
    """
    n = len(signal)
    result = np.full(n, np.nan)

    if n < window_size * 2:
        return result

    num_iterations = n - (window_size * 2 - 1)

    # 仕事量が多い場合のみ、prangeで並列化する
    if num_iterations > 2000:
        for i in nb.prange(window_size * 2 - 1, n):
            # 現在フレーム
            current_data = signal[i - window_size + 1 : i + 1]
            current_finite = current_data[np.isfinite(current_data)]

            # 前フレーム
            prev_data = signal[i - window_size * 2 + 1 : i - window_size + 1]
            prev_finite = prev_data[np.isfinite(prev_data)]

            if (
                len(current_finite) < window_size // 2
                or len(prev_finite) < window_size // 2
            ):
                continue

            # 両フレームのスペクトル計算
            current_fft = numba_fft(current_finite)
            current_spectrum = np.abs(current_fft[: len(current_finite) // 2])

            prev_fft = numba_fft(prev_finite)
            prev_spectrum = np.abs(prev_fft[: len(prev_finite) // 2])

            # サイズを揃える
            min_size = min(len(current_spectrum), len(prev_spectrum))
            if min_size > 0:
                current_spectrum = current_spectrum[:min_size]
                prev_spectrum = prev_spectrum[:min_size]

                # スペクトルフラックス計算（L2ノルム）
                flux = np.sqrt(np.sum((current_spectrum - prev_spectrum) ** 2))
                result[i] = flux
    else:
        # 仕事量が少ない場合は、オーバーヘッドのない通常のforループで処理
        for i in range(window_size * 2 - 1, n):
            # 現在フレーム
            current_data = signal[i - window_size + 1 : i + 1]
            current_finite = current_data[np.isfinite(current_data)]

            # 前フレーム
            prev_data = signal[i - window_size * 2 + 1 : i - window_size + 1]
            prev_finite = prev_data[np.isfinite(prev_data)]

            if (
                len(current_finite) < window_size // 2
                or len(prev_finite) < window_size // 2
            ):
                continue

            # 両フレームのスペクトル計算
            current_fft = numba_fft(current_finite)
            current_spectrum = np.abs(current_fft[: len(current_finite) // 2])

            prev_fft = numba_fft(prev_finite)
            prev_spectrum = np.abs(prev_fft[: len(prev_finite) // 2])

            # サイズを揃える
            min_size = min(len(current_spectrum), len(prev_spectrum))
            if min_size > 0:
                current_spectrum = current_spectrum[:min_size]
                prev_spectrum = prev_spectrum[:min_size]

                # スペクトルフラックス計算（L2ノルム）
                flux = np.sqrt(np.sum((current_spectrum - prev_spectrum) ** 2))
                result[i] = flux

    return result


@nb.njit(fastmath=True, cache=True, parallel=True)
def spectral_flatness_udf(signal: np.ndarray, window_size: int) -> np.ndarray:
    """
    スペクトル平坦度計算（動的並列化制御版）
    幾何平均と算術平均の比（Tonality係数）
    """
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


@nb.njit(fastmath=True, cache=True, parallel=True)
def spectral_entropy_udf(signal: np.ndarray, window_size: int) -> np.ndarray:
    """
    スペクトルエントロピー計算（動的並列化制御版）
    周波数分布の不確実性を測定
    """
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
            power_spectrum = np.abs(fft_values[: len(finite_data) // 2]) ** 2

            if len(power_spectrum) == 0:
                continue

            # 正規化して確率分布に変換
            total_power = np.sum(power_spectrum)
            if total_power > 0:
                probability = power_spectrum / total_power

                # エントロピー計算
                entropy = 0.0
                for p in probability:
                    if p > 1e-10:
                        entropy -= p * np.log2(p)

                result[i] = entropy
    else:
        # 仕事量が少ない場合は、オーバーヘッドのない通常のforループで処理
        for i in range(window_size - 1, n):
            window_data = signal[i - window_size + 1 : i + 1]
            finite_data = window_data[np.isfinite(window_data)]

            if len(finite_data) < window_size // 2:
                continue

            # NumbaネイティブFFTによるスペクトル計算
            fft_values = numba_fft(finite_data)
            power_spectrum = np.abs(fft_values[: len(finite_data) // 2]) ** 2

            if len(power_spectrum) == 0:
                continue

            # 正規化して確率分布に変換
            total_power = np.sum(power_spectrum)
            if total_power > 0:
                probability = power_spectrum / total_power

                # エントロピー計算
                entropy = 0.0
                for p in probability:
                    if p > 1e-10:
                        entropy -= p * np.log2(p)

                result[i] = entropy

    return result


# ==================================================================
# 3. ウェーブレット系 JIT関数群
# ==================================================================


@nb.njit(fastmath=True, cache=True, parallel=True)
def wavelet_energy_udf(
    signal: np.ndarray, window_size: int, levels: int = 4
) -> np.ndarray:
    """
    ウェーブレットエネルギー計算（動的並列化制御版）
    各レベルでの近似エネルギー
    【修正】元スクリプトに存在したが新モジュールに欠落していたため追加
    """
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

            # 簡易ウェーブレット変換（移動平均とディファレンス）
            level_energy = 0.0
            current_signal = finite_data.copy()

            for level in range(min(levels, 4)):
                if len(current_signal) < 4:
                    break

                # ローパスフィルタ（移動平均）
                filtered = np.zeros(len(current_signal) // 2)
                for j in range(len(filtered)):
                    idx = j * 2
                    if idx + 1 < len(current_signal):
                        filtered[j] = (
                            current_signal[idx] + current_signal[idx + 1]
                        ) / 2.0

                # エネルギー計算
                level_energy += np.sum(filtered**2)
                current_signal = filtered

            result[i] = level_energy
    else:
        # 仕事量が少ない場合は、オーバーヘッドのない通常のforループで処理
        for i in range(window_size - 1, n):
            window_data = signal[i - window_size + 1 : i + 1]
            finite_data = window_data[np.isfinite(window_data)]

            if len(finite_data) < window_size // 2:
                continue

            # 簡易ウェーブレット変換（移動平均とディファレンス）
            level_energy = 0.0
            current_signal = finite_data.copy()

            for level in range(min(levels, 4)):
                if len(current_signal) < 4:
                    break

                # ローパスフィルタ（移動平均）
                filtered = np.zeros(len(current_signal) // 2)
                for j in range(len(filtered)):
                    idx = j * 2
                    if idx + 1 < len(current_signal):
                        filtered[j] = (
                            current_signal[idx] + current_signal[idx + 1]
                        ) / 2.0

                # エネルギー計算
                level_energy += np.sum(filtered**2)
                current_signal = filtered

            result[i] = level_energy

    return result


@nb.njit(fastmath=True, cache=True, parallel=True)
def wavelet_entropy_udf(signal: np.ndarray, window_size: int) -> np.ndarray:
    """
    ウェーブレットエントロピー計算（動的並列化制御版）
    信号の二乗値に基づくエントロピー計算
    【修正】window_data をそのまま使用（元スクリプトに一致）
    """
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
def hilbert_amplitude_udf(signal: np.ndarray, window_size: int) -> np.ndarray:
    """
    ヒルベルト変換による振幅包絡線計算（動的並列化制御版）
    FFTを使用した近似ヒルベルト変換を行い、その振幅の平均を返す
    """
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

            # 近似ヒルベルト変換（90度位相シフト）
            # FFTを使用した近似実装
            fft_signal = numba_fft(finite_data)
            n_samples = len(finite_data)

            # 90度位相シフト
            hilbert_fft = fft_signal.copy()
            for j in range(1, n_samples // 2):
                hilbert_fft[j] *= -1j
                hilbert_fft[n_samples - j] *= 1j

            # IFFT相当の処理（簡易版）
            hilbert_signal = np.real(hilbert_fft)[: len(finite_data)]

            # 振幅包絡線
            amplitude_envelope = np.sqrt(finite_data**2 + hilbert_signal**2)
            result[i] = np.mean(amplitude_envelope)
    else:
        # 仕事量が少ない場合は、オーバーヘッドのない通常のforループで処理
        for i in range(window_size - 1, n):
            window_data = signal[i - window_size + 1 : i + 1]
            finite_data = window_data[np.isfinite(window_data)]

            if len(finite_data) < window_size // 2:
                continue

            # 近似ヒルベルト変換（90度位相シフト）
            # FFTを使用した近似実装
            fft_signal = numba_fft(finite_data)
            n_samples = len(finite_data)

            # 90度位相シフト
            hilbert_fft = fft_signal.copy()
            for j in range(1, n_samples // 2):
                hilbert_fft[j] *= -1j
                hilbert_fft[n_samples - j] *= 1j

            # IFFT相当の処理（簡易版）
            hilbert_signal = np.real(hilbert_fft)[: len(finite_data)]

            # 振幅包絡線
            amplitude_envelope = np.sqrt(finite_data**2 + hilbert_signal**2)
            result[i] = np.mean(amplitude_envelope)

    return result


@nb.njit(fastmath=True, cache=True, parallel=True)
def hilbert_phase_var_udf(signal: np.ndarray, window_size: int) -> np.ndarray:
    """
    ヒルベルト位相分散（動的並列化制御版）
    """
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
    """
    ヒルベルト位相安定性（動的並列化制御版）
    【修正】window_data に対して std チェック（元スクリプトに一致）
    """
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


@nb.njit(fastmath=True, cache=True, parallel=True)
def hilbert_freq_mean_udf(signal: np.ndarray, window_size: int) -> np.ndarray:
    """
    ヒルベルト瞬時周波数平均（動的並列化制御版）
    np.unwrapはNumbaでサポートされていないため、単純な差分で近似
    """
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
            analytic_signal = window_data + 1j * np.roll(window_data, 1)
            # np.unwrapはNumbaでサポートされていないため、単純な差分で近似
            instant_freq = np.abs(np.diff(np.angle(analytic_signal)))
            result[i] = np.mean(instant_freq)

    return result


@nb.njit(fastmath=True, cache=True, parallel=True)
def hilbert_freq_std_udf(signal: np.ndarray, window_size: int) -> np.ndarray:
    """
    ヒルベルト瞬時周波数標準偏差（動的並列化制御版）
    np.unwrapはNumbaでサポートされていないため、単純な差分で近似
    """
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
            analytic_signal = window_data + 1j * np.roll(window_data, 1)
            # np.unwrapはNumbaでサポートされていないため、単純な差分で近似
            instant_freq = np.abs(np.diff(np.angle(analytic_signal)))
            result[i] = np.std(instant_freq)

    return result


# ==================================================================
# 5. 音響系 JIT関数群
# ==================================================================


@nb.njit(fastmath=True, cache=True, parallel=True)
def acoustic_power_udf(signal: np.ndarray, window_size: int) -> np.ndarray:
    """
    音響パワー計算（動的並列化制御版）
    RMS（Root Mean Square）パワーの計算
    """
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


@nb.njit(fastmath=True, cache=True, parallel=True)
def acoustic_frequency_udf(
    signal: np.ndarray, window_size: int, sample_rate: float = 1.0
) -> np.ndarray:
    """
    音響周波数計算（動的並列化制御版）
    ゼロクロッシング率に基づく周波数推定
    """
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

            # ゼロクロッシング率計算
            zero_crossings = 0
            for j in range(1, len(finite_data)):
                if finite_data[j - 1] * finite_data[j] < 0:
                    zero_crossings += 1

            # 周波数推定（ゼロクロッシング率 / 2）
            if len(finite_data) > 1:
                frequency = (zero_crossings / (2.0 * len(finite_data))) * sample_rate
                result[i] = frequency
    else:
        # 仕事量が少ない場合は、オーバーヘッドのない通常のforループで処理
        for i in range(window_size - 1, n):
            window_data = signal[i - window_size + 1 : i + 1]
            finite_data = window_data[np.isfinite(window_data)]

            if len(finite_data) < window_size // 2:
                continue

            # ゼロクロッシング率計算
            zero_crossings = 0
            for j in range(1, len(finite_data)):
                if finite_data[j - 1] * finite_data[j] < 0:
                    zero_crossings += 1

            # 周波数推定（ゼロクロッシング率 / 2）
            if len(finite_data) > 1:
                frequency = (zero_crossings / (2.0 * len(finite_data))) * sample_rate
                result[i] = frequency

    return result
