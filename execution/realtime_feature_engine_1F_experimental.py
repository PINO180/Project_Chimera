# Project Cimera V5 - Feature Engine Module: 1F Experimental
# File: realtime_feature_engine_1F_experimental.py
# Description: 複雑系・音楽・運動学特徴量 (Network, Linguistics, Physics, Harmony)
#
# ▼▼ Step 13: core_indicators.py [CATEGORY: COMPLEX] へ移管済み。
#    全 UDF は core_indicators から import して使用する。
#    Single Source of Truth: このファイルに UDF 定義を持たない。
#
# 【アルゴリズム変更の注記】（意図的な変更）
#    core_indicators に取り込まれた実装は engine_1_F 学習側（Step 12）版であり、
#    旧リアルタイム版（このファイル）とは以下が異なる:
#      - parallel=True / nb.prange → 除去済み（Polars並列化との衝突回避）
#      - np.std(ddof=0) → Bessel補正 np.std(x)*sqrt(n/(n-1)) に統一
#      - ゼロ除算保護なし → + 1e-10 保護を追加
#      - list()/set() 動的確保 → バッファ配列（adj_buffer等）に置き換え
#    これらは学習側・リアルタイム側の特徴量値を物理的に一致させるための
#    意図的な変更である。旧リアルタイム版との数値差は許容済み。

import sys
from pathlib import Path

# ▼▼ /workspace をパスに追加してから blueprint をインポートする必要がある
sys.path.append(str(Path(__file__).resolve().parent.parent))  # /workspace をパスに追加
import blueprint as config
sys.path.append(str(config.CORE_DIR))

from core_indicators import (
    rolling_network_density_udf,
    rolling_network_clustering_udf,
    rolling_vocabulary_diversity_udf,
    rolling_semantic_flow_udf,
    rolling_golden_ratio_adherence_udf,
    rolling_symmetry_measure_udf,
    rolling_aesthetic_balance_udf,
    rolling_kinetic_energy_udf,
    rolling_muscle_force_udf,
    rolling_biomechanical_efficiency_udf,
    rolling_energy_expenditure_udf,
)

import numpy as np
from typing import Dict

# ==================================================================
# Numba UDF 定義
# ▼▼ Step 13 完了: 全 COMPLEX UDF は core_indicators.py に移管済み。
#    冒頭の from core_indicators import ... でインポートしているため、
#    ここでの再定義は不要。Single Source of Truth を維持する。
# ==================================================================
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
        # 音楽・リズム系は全てリスト外のため完全削除。
        # core_indicators には rolling_tonality_udf / rolling_rhythm_pattern_udf /
        # rolling_harmony_udf / rolling_musical_tension_udf が実装済みだが、
        # 旧リアルタイム版でも未実装・未呼び出しであり、かつ特徴量選択後の
        # 生存リスト外のため、意図的にインポート・呼び出しを省略している。
        # 復活させる場合は冒頭の from core_indicators import ... に追加すること。

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
