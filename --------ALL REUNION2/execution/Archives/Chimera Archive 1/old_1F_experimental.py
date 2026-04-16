# ==============================================================================
# Project Cimera V5 - Feature Engine Module: 1F Experimental
# File: realtime_feature_engine_1F_experimental.py
# Description: 複雑系・音楽・運動学特徴量 (Network, Linguistics, Physics, Harmony)
# Note: 最終選考を通過した304特徴量に必要なUDFのみを厳選して収録 (God Object解体版)
# ==============================================================================

import numpy as np
from numba import njit, float64, int64, boolean
from typing import Dict

# ==================================================================
# 1. ネットワーク・言語系 UDF群 (Network & Linguistics)
# ==================================================================


@njit(fastmath=True, cache=True)
def rolling_network_density_udf(prices: np.ndarray) -> float:
    """
    ネットワーク密度計算 (Numba JIT)
    """
    if len(prices) < 10:
        return np.nan

    finite_prices = prices[np.isfinite(prices)]
    if len(finite_prices) < 10:
        return np.nan

    window_n = len(finite_prices)
    threshold = np.std(finite_prices) * 0.5

    edge_count = 0
    max_possible_edges = window_n * (window_n - 1) / 2

    for j in range(window_n - 1):
        for k in range(j + 1, window_n):
            price_diff = abs(finite_prices[j] - finite_prices[k])
            if price_diff <= threshold:
                edge_count += 1

    if max_possible_edges > 0:
        return edge_count / max_possible_edges
    else:
        return 0.0


@njit(fastmath=True, cache=True)
def rolling_network_clustering_udf(prices: np.ndarray) -> float:
    """
    ネットワーククラスタリング係数 (Numba JIT)
    """
    if len(prices) < 15:
        return np.nan

    finite_prices = prices[np.isfinite(prices)]
    if len(finite_prices) < 15:
        return np.nan

    window_n = len(finite_prices)
    threshold = np.std(finite_prices) * 0.5

    # 隣接行列の構築
    adjacency = np.zeros((window_n, window_n), dtype=boolean)

    for j in range(window_n):
        for k in range(window_n):
            if j != k:
                price_diff = abs(finite_prices[j] - finite_prices[k])
                if price_diff <= threshold:
                    adjacency[j, k] = True

    total_clustering = 0.0
    valid_nodes = 0

    for j in range(window_n):
        neighbors = []
        for k in range(window_n):
            if adjacency[j, k]:
                neighbors.append(k)

        k_neighbors = len(neighbors)
        if k_neighbors < 2:
            continue

        neighbor_connections = 0
        for idx1 in range(len(neighbors)):
            for idx2 in range(idx1 + 1, len(neighbors)):
                n1, n2 = neighbors[idx1], neighbors[idx2]
                if adjacency[n1, n2]:
                    neighbor_connections += 1

        max_connections = k_neighbors * (k_neighbors - 1) / 2
        if max_connections > 0:
            total_clustering += neighbor_connections / max_connections
            valid_nodes += 1

    if valid_nodes > 0:
        return total_clustering / valid_nodes
    else:
        return 0.0


@njit(fastmath=True, cache=True)
def rolling_vocabulary_diversity_udf(prices: np.ndarray) -> float:
    """
    語彙多様性指標 (Numba JIT)
    """
    if len(prices) < 10:
        return np.nan

    finite_prices = prices[np.isfinite(prices)]
    if len(finite_prices) < 10:
        return np.nan

    std_price = np.std(finite_prices)
    mean_price = np.mean(finite_prices)

    if std_price == 0:
        return 0.0

    n_bins = 10
    price_min = mean_price - 2 * std_price
    price_max = mean_price + 2 * std_price
    bin_width = (price_max - price_min) / n_bins

    vocab_flags = np.zeros(10, dtype=boolean)
    total_tokens = 0

    for price in finite_prices:
        if bin_width > 0:
            bin_idx = int((price - price_min) / bin_width)
            bin_idx = max(0, min(bin_idx, n_bins - 1))
        else:
            bin_idx = 0

        vocab_flags[bin_idx] = True
        total_tokens += 1

    unique_vocab = 0
    for k in range(10):
        if vocab_flags[k]:
            unique_vocab += 1

    if total_tokens > 0:
        return unique_vocab / total_tokens
    else:
        return 0.0


@njit(fastmath=True, cache=True)
def rolling_linguistic_complexity_udf(prices: np.ndarray) -> float:
    """
    言語的複雑性 (Numba JIT)
    """
    if len(prices) < 20:
        return np.nan

    finite_prices = prices[np.isfinite(prices)]
    if len(finite_prices) < 20:
        return np.nan

    price_changes = np.diff(finite_prices)
    threshold = np.std(price_changes) * 0.1

    syntax_sequence = np.zeros(len(price_changes), dtype=int64)
    for i in range(len(price_changes)):
        change = price_changes[i]
        if change > threshold:
            syntax_sequence[i] = 1
        elif change < -threshold:
            syntax_sequence[i] = -1
        else:
            syntax_sequence[i] = 0

    if len(syntax_sequence) < 3:
        return 0.0

    # Count unique bigrams
    bigram_counts = np.zeros(9, dtype=boolean)
    for j in range(len(syntax_sequence) - 1):
        c1 = syntax_sequence[j] + 1
        c2 = syntax_sequence[j + 1] + 1
        idx = c1 * 3 + c2
        bigram_counts[idx] = True

    num_bigrams = 0
    for k in range(9):
        if bigram_counts[k]:
            num_bigrams += 1

    # Count unique trigrams
    trigram_counts = np.zeros(27, dtype=boolean)
    for j in range(len(syntax_sequence) - 2):
        c1 = syntax_sequence[j] + 1
        c2 = syntax_sequence[j + 1] + 1
        c3 = syntax_sequence[j + 2] + 1
        idx = c1 * 9 + c2 * 3 + c3
        trigram_counts[idx] = True

    num_trigrams = 0
    for k in range(27):
        if trigram_counts[k]:
            num_trigrams += 1

    max_bigrams = min(9, len(syntax_sequence) - 1)
    max_trigrams = min(27, len(syntax_sequence) - 2)

    if max_bigrams > 0 and max_trigrams > 0:
        bigram_complexity = num_bigrams / max_bigrams
        trigram_complexity = num_trigrams / max_trigrams
        return (bigram_complexity + trigram_complexity) / 2.0
    else:
        return 0.0


@njit(fastmath=True, cache=True)
def rolling_semantic_flow_udf(prices: np.ndarray) -> float:
    """
    意味的流れ (Numba JIT)
    """
    if len(prices) < 15:
        return np.nan

    finite_prices = prices[np.isfinite(prices)]
    if len(finite_prices) < 15:
        return np.nan

    window_n = len(finite_prices)
    window_size_local = min(5, window_n // 3)

    max_vectors = window_n
    semantic_vectors = np.zeros((max_vectors, 2), dtype=float64)
    vec_count = 0

    for j in range(window_size_local, window_n - window_size_local):
        neighborhood = finite_prices[j - window_size_local : j + window_size_local + 1]
        center_price = finite_prices[j]
        relative_positions = neighborhood - center_price

        if len(relative_positions) > 1:
            mean_rel = np.mean(relative_positions)
            std_rel = np.std(relative_positions)
            semantic_vectors[vec_count, 0] = mean_rel
            semantic_vectors[vec_count, 1] = std_rel
            vec_count += 1

    if vec_count < 2:
        return np.nan

    flow_continuity = 0.0
    valid_pairs = 0

    for j in range(vec_count - 1):
        vec1 = semantic_vectors[j]
        vec2 = semantic_vectors[j + 1]

        norm1 = np.sqrt(np.sum(vec1**2))
        norm2 = np.sqrt(np.sum(vec2**2))

        if norm1 > 1e-10 and norm2 > 1e-10:
            cosine_sim = np.sum(vec1 * vec2) / (norm1 * norm2)
            flow_continuity += cosine_sim
            valid_pairs += 1

    if valid_pairs > 0:
        semantic_flow = flow_continuity / valid_pairs
        return (semantic_flow + 1.0) / 2.0
    else:
        return 0.0


# ==================================================================
# 2. 幾何学・バランス系 UDF群 (Geometry & Balance)
# ==================================================================


@njit(fastmath=True, cache=True)
def rolling_golden_ratio_adherence_udf(prices: np.ndarray) -> float:
    """
    黄金比固着度 (Numba JIT)
    """
    if len(prices) < 10:
        return np.nan

    finite_prices = prices[np.isfinite(prices)]
    if len(finite_prices) < 10:
        return np.nan

    golden_ratio = 1.618033988749895  # (1 + sqrt(5)) / 2
    local_window = min(8, len(finite_prices) // 2)

    adherence_sum = 0.0
    adherence_count = 0

    for j in range(local_window, len(finite_prices) - local_window):
        local_subwindow = finite_prices[j - local_window : j + local_window + 1]
        local_high = np.max(local_subwindow)
        local_low = np.min(local_subwindow)

        if local_low > 0:
            ratio = local_high / local_low
            deviation = abs(ratio - golden_ratio) / golden_ratio
            adherence = 1.0 / (1.0 + deviation)
            adherence_sum += adherence
            adherence_count += 1

    if adherence_count > 0:
        return adherence_sum / adherence_count
    else:
        return np.nan


@njit(fastmath=True, cache=True)
def rolling_symmetry_measure_udf(prices: np.ndarray) -> float:
    """
    対称性測定 (Numba JIT)
    """
    if len(prices) < 20:
        return np.nan

    finite_prices = prices[np.isfinite(prices)]
    if len(finite_prices) < 20:
        return np.nan

    window_n = len(finite_prices)
    center = window_n // 2

    left_half = finite_prices[:center]
    # ウィンドウサイズが奇数の場合、中心を含まないように調整
    if window_n % 2 == 1:
        right_half = finite_prices[center + 1 :]
    else:
        right_half = finite_prices[center:]

    # 右半分を反転（鏡像）
    right_half_reversed = right_half[::-1]

    min_len = min(len(left_half), len(right_half_reversed))
    if min_len < 5:
        return np.nan

    left_target = left_half[-min_len:]
    right_target = right_half_reversed[:min_len]

    # 正規化関数ロジック展開
    mean_left = np.mean(left_target)
    std_left = np.std(left_target)

    mean_right = np.mean(right_target)
    std_right = np.std(right_target)

    if std_left <= 1e-10 or std_right <= 1e-10:
        return 0.0

    # 相関係数計算 (手動)
    # Corr(X, Y) = Mean((X-uX)(Y-uY)) / (stdX * stdY)
    covariance = np.mean((left_target - mean_left) * (right_target - mean_right))
    correlation = covariance / (std_left * std_right)

    return (correlation + 1.0) / 2.0


@njit(fastmath=True, cache=True)
def rolling_aesthetic_balance_udf(prices: np.ndarray) -> float:
    """
    美的バランス (Numba JIT)
    """
    if len(prices) < 15:
        return np.nan

    finite_prices = prices[np.isfinite(prices)]
    if len(finite_prices) < 15:
        return np.nan

    gradients = np.diff(finite_prices)

    if len(gradients) < 5:
        return np.nan

    abs_gradients = np.abs(gradients)
    mean_grad = np.mean(abs_gradients)
    std_grad = np.std(abs_gradients)

    if std_grad <= 1e-10:
        return 1.0

    gentle_threshold = mean_grad - 0.5 * std_grad
    moderate_threshold = mean_grad + 0.5 * std_grad
    intense_threshold = mean_grad + 1.5 * std_grad

    gentle_count = 0
    moderate_count = 0
    intense_count = 0

    for grad in abs_gradients:
        if grad <= gentle_threshold:
            gentle_count += 1
        elif grad <= moderate_threshold:
            moderate_count += 1
        elif grad <= intense_threshold:
            intense_count += 1

    total_counted = gentle_count + moderate_count + intense_count
    if total_counted == 0:
        return 0.0

    ideal_gentle = 0.6
    ideal_moderate = 0.3
    ideal_intense = 0.1

    actual_gentle = gentle_count / total_counted
    actual_moderate = moderate_count / total_counted
    actual_intense = intense_count / total_counted

    balance_deviation = (
        abs(actual_gentle - ideal_gentle)
        + abs(actual_moderate - ideal_moderate)
        + abs(actual_intense - ideal_intense)
    ) / 2.0

    aesthetic_balance = 1.0 - balance_deviation
    return max(0.0, aesthetic_balance)


# ==================================================================
# 3. 音楽・運動学系 UDF群 (Music & Kinetics)
# ==================================================================


@njit(fastmath=True, cache=True)
def rolling_tonality_udf(prices: np.ndarray) -> float:
    """
    調性 (Numba JIT)
    """
    if len(prices) < 12:
        return np.nan

    finite_prices = prices[np.isfinite(prices)]
    if len(finite_prices) < 12:
        return np.nan

    price_changes = np.diff(finite_prices)

    if len(price_changes) < 5:
        return np.nan

    std_change = np.std(price_changes)
    if std_change <= 1e-10:
        return 0.5

    normalized_changes = price_changes / std_change
    scale_degrees = np.zeros(12)

    for change in normalized_changes:
        degree_idx = int((change + 3.0) / 6.0 * 11.0)
        degree_idx = max(0, min(degree_idx, 11))
        scale_degrees[degree_idx] += 1.0

    major_pattern = np.array([1, 0, 1, 0, 1, 1, 0, 1, 0, 1, 0, 1], dtype=float64)
    minor_pattern = np.array([1, 0, 1, 1, 0, 1, 0, 1, 1, 0, 1, 0], dtype=float64)

    sum_degrees = np.sum(scale_degrees)
    if sum_degrees > 0:
        scale_distribution = scale_degrees / sum_degrees

        major_similarity = np.sum(scale_distribution * major_pattern)
        minor_similarity = np.sum(scale_distribution * minor_pattern)

        total_similarity = major_similarity + minor_similarity
        if total_similarity > 0:
            return major_similarity / total_similarity
        else:
            return 0.5
    else:
        return 0.5


@njit(fastmath=True, cache=True)
def rolling_rhythm_pattern_udf(prices: np.ndarray) -> float:
    """
    リズムパターン (Numba JIT)
    """
    if len(prices) < 20:
        return np.nan

    finite_prices = prices[np.isfinite(prices)]
    if len(finite_prices) < 20:
        return np.nan

    price_changes = np.diff(finite_prices)
    abs_changes = np.abs(price_changes)

    mean_change = np.mean(abs_changes)
    strong_beats = np.zeros(len(abs_changes), dtype=int64)
    for i in range(len(abs_changes)):
        if abs_changes[i] > mean_change:
            strong_beats[i] = 1

    max_pattern_strength = 0.0

    limit = min(8, len(strong_beats) // 3)
    for period in range(2, limit):
        pattern_score = 0.0
        pattern_count = 0

        for j in range(period, len(strong_beats)):
            if strong_beats[j] == strong_beats[j - period]:
                pattern_score += 1.0
            pattern_count += 1

        if pattern_count > 0:
            strength = pattern_score / pattern_count
            if strength > max_pattern_strength:
                max_pattern_strength = strength

    return max_pattern_strength


@njit(fastmath=True, cache=True)
def rolling_harmony_udf(prices: np.ndarray) -> float:
    """
    和声 (Numba JIT)
    """
    if len(prices) < 30:
        return np.nan

    finite_prices = prices[np.isfinite(prices)]
    if len(finite_prices) < 30:
        return np.nan

    window_n = len(finite_prices)

    short_window = max(3, window_n // 15)
    medium_window = max(5, window_n // 10)
    long_window = max(8, window_n // 6)

    if long_window >= window_n:
        return np.nan

    def calc_ma(data, w):
        n = len(data)
        res = np.zeros(n - w + 1)
        for i in range(n - w + 1):
            res[i] = np.mean(data[i : i + w])
        return res

    short_ma = calc_ma(finite_prices, short_window)
    medium_ma = calc_ma(finite_prices, medium_window)
    long_ma = calc_ma(finite_prices, long_window)

    min_len = min(len(short_ma), min(len(medium_ma), len(long_ma)))
    if min_len < 5:
        return np.nan

    short_ma_trim = short_ma[-min_len:]
    medium_ma_trim = medium_ma[-min_len:]
    long_ma_trim = long_ma[-min_len:]

    short_trend = np.diff(short_ma_trim)
    medium_trend = np.diff(medium_ma_trim)
    long_trend = np.diff(long_ma_trim)

    harmony_sum = 0.0
    count = 0

    for j in range(len(short_trend)):
        s1 = np.sign(short_trend[j])
        s2 = np.sign(medium_trend[j])
        s3 = np.sign(long_trend[j])

        if s1 == s2 and s2 == s3:
            if s1 != 0:
                harmony_sum += 1.0
        elif (s1 == s2) or (s2 == s3) or (s1 == s3):
            harmony_sum += 0.5

        count += 1

    if count > 0:
        return harmony_sum / count
    else:
        return 0.0


@njit(fastmath=True, cache=True)
def rolling_musical_tension_udf(prices: np.ndarray) -> float:
    """
    音楽的緊張度 (Numba JIT)
    """
    if len(prices) < 15:
        return np.nan

    finite_prices = prices[np.isfinite(prices)]
    if len(finite_prices) < 15:
        return np.nan

    price_changes = np.diff(finite_prices)

    if len(price_changes) < 5:
        return np.nan

    local_window = min(5, len(price_changes) // 3)
    tension_sum = 0.0
    count = 0

    for j in range(local_window, len(price_changes) - local_window):
        local_changes = price_changes[j - local_window : j + local_window + 1]

        sign_changes = 0
        for k in range(len(local_changes) - 1):
            if np.sign(local_changes[k]) != np.sign(local_changes[k + 1]):
                sign_changes += 1

        direction_dissonance = 0.0
        if len(local_changes) > 0:
            direction_dissonance = sign_changes / len(local_changes)

        max_volatility = np.max(np.abs(local_changes))
        mean_abs_price = np.mean(np.abs(finite_prices))
        intensity_dissonance = max_volatility / (mean_abs_price + 1e-10)

        total_tension = (direction_dissonance + intensity_dissonance) / 2.0
        if total_tension > 1.0:
            total_tension = 1.0

        tension_sum += total_tension
        count += 1

    if count > 0:
        return tension_sum / count
    else:
        return 0.0


@njit(fastmath=True, cache=True)
def rolling_kinetic_energy_udf(prices: np.ndarray) -> float:
    """
    運動エネルギー (Numba JIT)
    """
    if len(prices) < 10:
        return np.nan

    finite_prices = prices[np.isfinite(prices)]
    if len(finite_prices) < 10:
        return np.nan

    velocities = np.diff(finite_prices)

    if len(velocities) < 2:
        return np.nan

    masses = finite_prices[1:]

    kinetic_energies = 0.5 * masses * velocities**2

    if len(kinetic_energies) > 0:
        mean_kinetic_energy = np.mean(kinetic_energies)
        mean_price = np.mean(finite_prices)
        if mean_price > 1e-10:
            return mean_kinetic_energy / (mean_price**2)
        else:
            return mean_kinetic_energy
    else:
        return 0.0


@njit(fastmath=True, cache=True)
def rolling_muscle_force_udf(prices: np.ndarray) -> float:
    """
    筋力 (Numba JIT)
    """
    if len(prices) < 15:
        return np.nan

    finite_prices = prices[np.isfinite(prices)]
    if len(finite_prices) < 15:
        return np.nan

    velocities = np.diff(finite_prices)
    if len(velocities) < 2:
        return np.nan

    accelerations = np.diff(velocities)
    masses = finite_prices[2:]

    forces = masses * np.abs(accelerations)
    force_directions = np.sign(accelerations)

    sustained_forces_sum = 0.0
    sustained_count = 0

    if len(force_directions) > 0:
        current_direction = force_directions[0]
        current_duration = 1
        current_force_sum = forces[0]

        for j in range(1, len(force_directions)):
            if force_directions[j] == current_direction and current_direction != 0:
                current_duration += 1
                current_force_sum += forces[j]
            else:
                if current_duration > 1:
                    avg_force = current_force_sum / current_duration
                    sustained_forces_sum += avg_force
                    sustained_count += 1

                current_direction = force_directions[j]
                current_duration = 1
                current_force_sum = forces[j]

        if current_duration > 1:
            avg_force = current_force_sum / current_duration
            sustained_forces_sum += avg_force
            sustained_count += 1

    instantaneous_force = np.mean(forces) if len(forces) > 0 else 0.0
    sustained_force = (
        sustained_forces_sum / sustained_count if sustained_count > 0 else 0.0
    )

    muscle_force_score = 0.7 * instantaneous_force + 0.3 * sustained_force

    mean_price = np.mean(finite_prices)
    if mean_price > 1e-10:
        return muscle_force_score / (mean_price**2)
    else:
        return muscle_force_score


@njit(fastmath=True, cache=True)
def rolling_biomechanical_efficiency_udf(prices: np.ndarray) -> float:
    """
    生体力学効率 (Numba JIT)
    """
    if len(prices) < 20:
        return np.nan

    finite_prices = prices[np.isfinite(prices)]
    if len(finite_prices) < 20:
        return np.nan

    price_changes = np.diff(finite_prices)
    total_displacement = np.sum(np.abs(price_changes))

    velocities = price_changes
    if len(velocities) > 1:
        accelerations = np.diff(velocities)
    else:
        accelerations = np.zeros(1)

    kinetic_energy = np.sum(velocities**2)
    acceleration_energy = np.sum(accelerations**2) if len(accelerations) > 0 else 0.0
    total_energy = kinetic_energy + acceleration_energy

    if total_energy > 1e-10 and total_displacement > 1e-10:
        raw_efficiency = total_displacement / total_energy

        reference_efficiency = total_displacement / (np.sum(np.abs(velocities)) + 1e-10)
        normalized_efficiency = raw_efficiency / (reference_efficiency + 1e-10)
        if normalized_efficiency > 1.0:
            return 1.0
        else:
            return normalized_efficiency
    else:
        return 0.0


@njit(fastmath=True, cache=True)
def rolling_energy_expenditure_udf(prices: np.ndarray) -> float:
    """
    エネルギー消費量 (Numba JIT)
    """
    if len(prices) < 15:
        return np.nan

    finite_prices = prices[np.isfinite(prices)]
    if len(finite_prices) < 15:
        return np.nan

    baseline_energy = np.var(finite_prices)

    price_changes = np.diff(finite_prices)
    movement_energy = np.sum(price_changes**2)

    acceleration_energy = 0.0
    if len(price_changes) > 1:
        accelerations = np.diff(price_changes)
        acceleration_energy = np.sum(accelerations**2)

    nonlinearity_energy = 0.0
    n_points = len(finite_prices)
    if n_points >= 3:
        x = np.arange(n_points, dtype=float64)

        sum_x = np.sum(x)
        sum_y = np.sum(finite_prices)
        sum_xy = np.sum(x * finite_prices)
        sum_x2 = np.sum(x * x)

        denom = n_points * sum_x2 - sum_x**2
        if denom != 0:
            slope = (n_points * sum_xy - sum_x * sum_y) / denom
            intercept = (sum_y - slope * sum_x) / n_points

            linear_trend = intercept + slope * x
            nonlinearity_energy = np.sum((finite_prices - linear_trend) ** 2)

    total_energy = (
        baseline_energy + movement_energy + acceleration_energy + nonlinearity_energy
    )

    mean_price = np.mean(finite_prices)
    if mean_price > 1e-10:
        return total_energy / (mean_price**2)
    else:
        return total_energy
