# Project Cimera V5 - Feature Engine Module: 1F Experimental
# File: realtime_feature_engine_1F_experimental.py
# Description: 複雑系・音楽・運動学特徴量 (Network, Linguistics, Physics, Harmony)

import numpy as np
import numba as nb
from numba import float64, int64, boolean
from typing import Dict

# ==================================================================
# 1. ネットワーク・言語系 UDF群 (Network & Linguistics)
# ==================================================================


@nb.njit(fastmath=True, cache=True, parallel=True)
def rolling_network_density_udf(prices: np.ndarray, window_size: int) -> np.ndarray:

    n = len(prices)
    results = np.full(n, np.nan)

    for i in nb.prange(window_size - 1, n):
        window_prices = prices[i - window_size + 1 : i + 1]

        if len(window_prices) < 10:
            continue

        finite_prices = window_prices[np.isfinite(window_prices)]
        if len(finite_prices) < 10:
            continue

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
            density = edge_count / max_possible_edges
        else:
            density = 0.0

        results[i] = density

    return results


@nb.njit(fastmath=True, cache=True, parallel=True)
def rolling_network_clustering_udf(prices: np.ndarray, window_size: int) -> np.ndarray:

    n = len(prices)
    results = np.full(n, np.nan)

    for i in nb.prange(window_size - 1, n):
        window_prices = prices[i - window_size + 1 : i + 1]

        if len(window_prices) < 15:
            continue

        finite_prices = window_prices[np.isfinite(window_prices)]
        if len(finite_prices) < 15:
            continue

        window_n = len(finite_prices)
        threshold = np.std(finite_prices) * 0.5

        adjacency = np.zeros((window_n, window_n), dtype=nb.boolean)

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

            k = len(neighbors)
            if k < 2:
                continue

            neighbor_connections = 0
            for idx1 in range(len(neighbors)):
                for idx2 in range(idx1 + 1, len(neighbors)):
                    n1, n2 = neighbors[idx1], neighbors[idx2]
                    if adjacency[n1, n2]:
                        neighbor_connections += 1

            max_connections = k * (k - 1) / 2
            if max_connections > 0:
                clustering_j = neighbor_connections / max_connections
                total_clustering += clustering_j
                valid_nodes += 1

        if valid_nodes > 0:
            clustering_coeff = total_clustering / valid_nodes
        else:
            clustering_coeff = 0.0

        results[i] = clustering_coeff

    return results


# =============================================================================
# 言語学: 語彙多様性・言語的複雑性・意味的流れ (並列化版)
# =============================================================================


@nb.njit(fastmath=True, cache=True, parallel=True)
def rolling_vocabulary_diversity_udf(
    prices: np.ndarray, window_size: int
) -> np.ndarray:

    n = len(prices)
    results = np.full(n, np.nan)

    for i in nb.prange(window_size - 1, n):
        window_prices = prices[i - window_size + 1 : i + 1]

        if len(window_prices) < 10:
            continue

        finite_prices = window_prices[np.isfinite(window_prices)]
        if len(finite_prices) < 10:
            continue

        std_price = np.std(finite_prices)
        mean_price = np.mean(finite_prices)

        if std_price == 0:
            results[i] = 0.0
            continue

        n_bins = 10
        price_min = mean_price - 2 * std_price
        price_max = mean_price + 2 * std_price
        bin_width = (price_max - price_min) / n_bins

        used_vocabularies = set()
        total_tokens = 0

        for price in finite_prices:
            if bin_width > 0:
                bin_idx = int((price - price_min) / bin_width)
                bin_idx = max(0, min(bin_idx, n_bins - 1))
            else:
                bin_idx = 0

            used_vocabularies.add(bin_idx)
            total_tokens += 1

        if total_tokens > 0:
            vocabulary_diversity = len(used_vocabularies) / total_tokens
        else:
            vocabulary_diversity = 0.0

        results[i] = vocabulary_diversity

    return results


@nb.njit(fastmath=True, cache=True, parallel=True)
def rolling_semantic_flow_udf(prices: np.ndarray, window_size: int) -> np.ndarray:

    n = len(prices)
    results = np.full(n, np.nan)

    for i in nb.prange(window_size - 1, n):
        window_prices = prices[i - window_size + 1 : i + 1]

        if len(window_prices) < 15:
            continue

        finite_prices = window_prices[np.isfinite(window_prices)]
        if len(finite_prices) < 15:
            continue

        window_n = len(finite_prices)
        window_size_local = min(5, window_n // 3)

        semantic_vectors = []

        for j in range(window_size_local, window_n - window_size_local):
            neighborhood = finite_prices[
                j - window_size_local : j + window_size_local + 1
            ]
            center_price = finite_prices[j]
            relative_positions = neighborhood - center_price

            if len(relative_positions) > 1:
                mean_rel = np.mean(relative_positions)
                std_rel = np.std(relative_positions)
                semantic_vector = np.array([mean_rel, std_rel])
                semantic_vectors.append(semantic_vector)

        if len(semantic_vectors) < 2:
            continue

        flow_continuity = 0.0
        valid_pairs = 0

        for j in range(len(semantic_vectors) - 1):
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
            semantic_flow = (semantic_flow + 1) / 2
        else:
            semantic_flow = 0.0

        results[i] = semantic_flow

    return results


# =============================================================================
# 美学: 黄金比・対称性・美的バランス (並列化版)
# =============================================================================


@nb.njit(fastmath=True, cache=True, parallel=True)
def rolling_golden_ratio_adherence_udf(
    prices: np.ndarray, window_size: int
) -> np.ndarray:

    n = len(prices)
    results = np.full(n, np.nan)
    golden_ratio = (1 + np.sqrt(5)) / 2

    for i in nb.prange(window_size - 1, n):
        window_prices = prices[i - window_size + 1 : i + 1]

        if len(window_prices) < 10:
            continue

        finite_prices = window_prices[np.isfinite(window_prices)]
        if len(finite_prices) < 10:
            continue

        local_window = min(8, len(finite_prices) // 2)
        adherence_scores = []

        for j in range(local_window, len(finite_prices) - local_window):
            local_subwindow = finite_prices[j - local_window : j + local_window + 1]
            local_high = np.max(local_subwindow)
            local_low = np.min(local_subwindow)

            if local_low > 0:
                ratio = local_high / local_low
                deviation = abs(ratio - golden_ratio) / golden_ratio
                adherence = 1.0 / (1.0 + deviation)
                adherence_scores.append(adherence)

        if adherence_scores:
            results[i] = np.mean(np.array(adherence_scores))

    return results


@nb.njit(fastmath=True, cache=True, parallel=True)
def rolling_symmetry_measure_udf(prices: np.ndarray, window_size: int) -> np.ndarray:

    n = len(prices)
    results = np.full(n, np.nan)

    for i in nb.prange(window_size - 1, n):
        window_prices = prices[i - window_size + 1 : i + 1]

        if len(window_prices) < 20:
            continue

        finite_prices = window_prices[np.isfinite(window_prices)]
        if len(finite_prices) < 20:
            continue

        window_n = len(finite_prices)
        center = window_n // 2

        left_half = finite_prices[:center]
        right_half = (
            finite_prices[center + 1 :] if window_n % 2 == 1 else finite_prices[center:]
        )

        right_half_reversed = right_half[::-1]

        min_len = min(len(left_half), len(right_half_reversed))
        if min_len < 5:
            continue

        left_normalized = left_half[-min_len:]
        right_normalized = right_half_reversed[:min_len]

        def normalize_series(series):
            mean_val = np.mean(series)
            std_val = np.std(series)
            if std_val > 1e-10:
                return (series - mean_val) / std_val
            else:
                return series - mean_val

        left_norm = normalize_series(left_normalized)
        right_norm = normalize_series(right_normalized)

        if len(left_norm) < 2:
            continue

        mean_left = np.mean(left_norm)
        mean_right = np.mean(right_norm)

        numerator = np.sum((left_norm - mean_left) * (right_norm - mean_right))
        denom_left = np.sum((left_norm - mean_left) ** 2)
        denom_right = np.sum((right_norm - mean_right) ** 2)

        if denom_left > 1e-10 and denom_right > 1e-10:
            correlation = numerator / np.sqrt(denom_left * denom_right)
            symmetry = (correlation + 1) / 2
        else:
            symmetry = 0.0

        results[i] = symmetry

    return results


@nb.njit(fastmath=True, cache=True, parallel=True)
def rolling_aesthetic_balance_udf(prices: np.ndarray, window_size: int) -> np.ndarray:

    n = len(prices)
    results = np.full(n, np.nan)

    for i in nb.prange(window_size - 1, n):
        window_prices = prices[i - window_size + 1 : i + 1]

        if len(window_prices) < 15:
            continue

        finite_prices = window_prices[np.isfinite(window_prices)]
        if len(finite_prices) < 15:
            continue

        gradients = np.diff(finite_prices)

        if len(gradients) < 5:
            continue

        abs_gradients = np.abs(gradients)
        mean_grad = np.mean(abs_gradients)
        std_grad = np.std(abs_gradients)

        if std_grad <= 1e-10:
            results[i] = 1.0
            continue

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
            results[i] = 0.0
            continue

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
        ) / 2

        aesthetic_balance = 1.0 - balance_deviation
        results[i] = max(0.0, aesthetic_balance)

    return results


# =============================================================================
# 音楽理論: 調性・リズムパターン・和声・音楽的緊張度 (並列化版)
# =============================================================================


# =============================================================================
# 生体力学: 運動エネルギー・筋力・生体力学効率・エネルギー消費量 (並列化版)
# =============================================================================


@nb.njit(fastmath=True, cache=True, parallel=True)
def rolling_kinetic_energy_udf(prices: np.ndarray, window_size: int) -> np.ndarray:

    n = len(prices)
    results = np.full(n, np.nan)

    for i in nb.prange(window_size - 1, n):
        window_prices = prices[i - window_size + 1 : i + 1]

        if len(window_prices) < 10:
            continue

        finite_prices = window_prices[np.isfinite(window_prices)]
        if len(finite_prices) < 10:
            continue

        velocities = np.diff(finite_prices)

        if len(velocities) < 2:
            continue

        masses = finite_prices[1:]

        kinetic_energies = 0.5 * masses * velocities**2

        if len(kinetic_energies) > 0:
            mean_kinetic_energy = np.mean(kinetic_energies)
            mean_price = np.mean(finite_prices)
            if mean_price > 1e-10:
                normalized_energy = mean_kinetic_energy / (mean_price**2)
            else:
                normalized_energy = mean_kinetic_energy
        else:
            normalized_energy = 0.0

        results[i] = normalized_energy

    return results


@nb.njit(fastmath=True, cache=True, parallel=True)
def rolling_muscle_force_udf(prices: np.ndarray, window_size: int) -> np.ndarray:

    n = len(prices)
    results = np.full(n, np.nan)

    for i in nb.prange(window_size - 1, n):
        window_prices = prices[i - window_size + 1 : i + 1]

        if len(window_prices) < 15:
            continue

        finite_prices = window_prices[np.isfinite(window_prices)]
        if len(finite_prices) < 15:
            continue

        velocities = np.diff(finite_prices)

        if len(velocities) < 2:
            continue

        accelerations = np.diff(velocities)

        masses = finite_prices[2:]
        forces = masses * np.abs(accelerations)

        force_directions = np.sign(accelerations)
        sustained_forces = []

        current_direction = force_directions[0] if len(force_directions) > 0 else 0
        current_duration = 1
        current_force_sum = forces[0] if len(forces) > 0 else 0

        for j in range(1, len(force_directions)):
            if force_directions[j] == current_direction and current_direction != 0:
                current_duration += 1
                current_force_sum += forces[j]
            else:
                if current_duration > 1:
                    avg_sustained_force = current_force_sum / current_duration
                    sustained_forces.append(avg_sustained_force)

                current_direction = force_directions[j]
                current_duration = 1
                current_force_sum = forces[j]

        if current_duration > 1:
            avg_sustained_force = current_force_sum / current_duration
            sustained_forces.append(avg_sustained_force)

        instantaneous_force = np.mean(forces) if len(forces) > 0 else 0.0
        sustained_force = (
            np.mean(np.array(sustained_forces)) if sustained_forces else 0.0
        )

        muscle_force_score = 0.7 * instantaneous_force + 0.3 * sustained_force

        mean_price = np.mean(finite_prices)
        if mean_price > 1e-10:
            normalized_muscle_force = muscle_force_score / (mean_price**2)
        else:
            normalized_muscle_force = muscle_force_score

        results[i] = normalized_muscle_force

    return results


@nb.njit(fastmath=True, cache=True, parallel=True)
def rolling_biomechanical_efficiency_udf(
    prices: np.ndarray, window_size: int
) -> np.ndarray:

    n = len(prices)
    results = np.full(n, np.nan)

    for i in nb.prange(window_size - 1, n):
        window_prices = prices[i - window_size + 1 : i + 1]

        if len(window_prices) < 20:
            continue

        finite_prices = window_prices[np.isfinite(window_prices)]
        if len(finite_prices) < 20:
            continue

        price_changes = np.diff(finite_prices)
        total_displacement = np.sum(np.abs(price_changes))

        velocities = price_changes
        accelerations = np.diff(velocities) if len(velocities) > 1 else np.array([0.0])

        kinetic_energy = np.sum(velocities**2)
        acceleration_energy = np.sum(accelerations**2) if len(accelerations) > 0 else 0
        total_energy = kinetic_energy + acceleration_energy

        if total_energy > 1e-10 and total_displacement > 1e-10:
            raw_efficiency = total_displacement / total_energy

            reference_efficiency = total_displacement / (
                np.sum(np.abs(velocities)) + 1e-10
            )
            normalized_efficiency = raw_efficiency / (reference_efficiency + 1e-10)
            efficiency = min(normalized_efficiency, 1.0)
        else:
            efficiency = 0.0

        results[i] = efficiency

    return results


@nb.njit(fastmath=True, cache=True, parallel=True)
def rolling_energy_expenditure_udf(prices: np.ndarray, window_size: int) -> np.ndarray:

    n = len(prices)
    results = np.full(n, np.nan)

    for i in nb.prange(window_size - 1, n):
        window_prices = prices[i - window_size + 1 : i + 1]

        if len(window_prices) < 15:
            continue

        finite_prices = window_prices[np.isfinite(window_prices)]
        if len(finite_prices) < 15:
            continue

        baseline_energy = np.var(finite_prices)

        price_changes = np.diff(finite_prices)
        movement_energy = np.sum(price_changes**2)

        if len(price_changes) > 1:
            accelerations = np.diff(price_changes)
            acceleration_energy = np.sum(accelerations**2)
        else:
            acceleration_energy = 0.0

        if len(finite_prices) >= 3:
            x = np.arange(len(finite_prices), dtype=np.float64)
            n_points = len(x)

            sum_x = np.sum(x)
            sum_y = np.sum(finite_prices)
            sum_xy = np.sum(x * finite_prices)
            sum_x2 = np.sum(x * x)

            if n_points * sum_x2 - sum_x**2 != 0:
                slope = (n_points * sum_xy - sum_x * sum_y) / (
                    n_points * sum_x2 - sum_x**2
                )
                intercept = (sum_y - slope * sum_x) / n_points

                linear_trend = intercept + slope * x

                nonlinearity_energy = np.sum((finite_prices - linear_trend) ** 2)
            else:
                nonlinearity_energy = 0
        else:
            nonlinearity_energy = 0

        total_energy = (
            baseline_energy
            + movement_energy
            + acceleration_energy
            + nonlinearity_energy
        )

        mean_price = np.mean(finite_prices)
        if mean_price > 1e-10:
            normalized_energy = total_energy / (mean_price**2)
        else:
            normalized_energy = total_energy

        results[i] = normalized_energy

    return results


# ==================================================================
# メイン計算モジュール
# ==================================================================


def _window(arr: np.ndarray, window: int) -> np.ndarray:

    if window <= 0:
        return np.array([], dtype=arr.dtype)
    if window > len(arr):
        return arr
    return arr[-window:]


# [QA修正] 配列から最新1Tick（スカラー値）を抽出する関数を追加
def _last(arr: np.ndarray) -> float:

    if len(arr) == 0:
        return np.nan
    return float(arr[-1])


class FeatureModule1F:
    @staticmethod
    def calculate_features(data: Dict[str, np.ndarray]) -> Dict[str, float]:
        features = {}

        # 安全な参照用変数
        close_arr = data["close"]

        # ▼▼ 追加: Rule 5に基づき、空配列時の無駄な計算と潜在的なエラーを入り口で遮断
        if len(close_arr) == 0:
            return features

        # ---------------------------------------------------------
        # 1. 美的バランス系指標 (Aesthetic Balance)
        # ---------------------------------------------------------
        features["e1f_aesthetic_balance_21"] = _last(
            rolling_aesthetic_balance_udf(_window(close_arr, 21), 21)
        )
        # 34, 55, 89 はリスト外のため削除

        # ---------------------------------------------------------
        # 2. バイオメカニクス系指標 (Biomechanics)
        # ---------------------------------------------------------
        features["e1f_biomechanical_efficiency_20"] = _last(
            rolling_biomechanical_efficiency_udf(_window(close_arr, 20), 20)
        )

        features["e1f_energy_expenditure_20"] = _last(
            rolling_energy_expenditure_udf(_window(close_arr, 20), 20)
        )
        features["e1f_energy_expenditure_60"] = _last(
            rolling_energy_expenditure_udf(_window(close_arr, 60), 60)
        )
        # 40はリスト外のため削除

        features["e1f_kinetic_energy_10"] = _last(
            rolling_kinetic_energy_udf(_window(close_arr, 10), 10)
        )
        features["e1f_kinetic_energy_20"] = _last(
            rolling_kinetic_energy_udf(_window(close_arr, 20), 20)
        )
        features["e1f_kinetic_energy_40"] = _last(
            rolling_kinetic_energy_udf(_window(close_arr, 40), 40)
        )

        features["e1f_muscle_force_20"] = _last(
            rolling_muscle_force_udf(_window(close_arr, 20), 20)
        )

        # ---------------------------------------------------------
        # 3. 黄金比系指標 (Golden Ratio)
        # ---------------------------------------------------------
        features["e1f_golden_ratio_adherence_21"] = _last(
            rolling_golden_ratio_adherence_udf(_window(close_arr, 21), 21)
        )
        features["e1f_golden_ratio_adherence_34"] = _last(
            rolling_golden_ratio_adherence_udf(_window(close_arr, 34), 34)
        )
        features["e1f_golden_ratio_adherence_55"] = _last(
            rolling_golden_ratio_adherence_udf(_window(close_arr, 55), 55)
        )

        # ---------------------------------------------------------
        # 4. 対称性・調和系指標 (Symmetry / Harmony)
        # ---------------------------------------------------------
        features["e1f_symmetry_measure_21"] = _last(
            rolling_symmetry_measure_udf(_window(close_arr, 21), 21)
        )
        features["e1f_symmetry_measure_34"] = _last(
            rolling_symmetry_measure_udf(_window(close_arr, 34), 34)
        )
        features["e1f_symmetry_measure_55"] = _last(
            rolling_symmetry_measure_udf(_window(close_arr, 55), 55)
        )
        features["e1f_symmetry_measure_89"] = _last(
            rolling_symmetry_measure_udf(_window(close_arr, 89), 89)
        )
        # Harmony系はリスト外のため削除

        # ---------------------------------------------------------
        # 5. 音楽・リズム系指標 (Musical / Rhythm)
        # ---------------------------------------------------------
        # 音楽・リズム系は全てリスト外のため完全削除

        # ---------------------------------------------------------
        # 6. ネットワーク系指標 (Network)
        # ---------------------------------------------------------
        features["e1f_network_clustering_50"] = _last(
            rolling_network_clustering_udf(_window(close_arr, 50), 50)
        )
        # 20, 30, 100 はリスト外のため削除

        features["e1f_network_density_20"] = _last(
            rolling_network_density_udf(_window(close_arr, 20), 20)
        )
        features["e1f_network_density_50"] = _last(
            rolling_network_density_udf(_window(close_arr, 50), 50)
        )
        # 30, 100 はリスト外のため削除

        # ---------------------------------------------------------
        # 7. 言語・意味系指標 (Linguistic / Semantic)
        # ---------------------------------------------------------
        # Linguistic Complexity はリスト外のため削除

        features["e1f_semantic_flow_25"] = _last(
            rolling_semantic_flow_udf(_window(close_arr, 25), 25)
        )
        # 15, 40 はリスト外のため削除

        features["e1f_vocabulary_diversity_15"] = _last(
            rolling_vocabulary_diversity_udf(_window(close_arr, 15), 15)
        )
        # 25, 40, 80 はリスト外のため削除

        return features
