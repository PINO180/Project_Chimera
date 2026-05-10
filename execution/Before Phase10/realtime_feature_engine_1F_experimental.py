# realtime_feature_engine_1F_experimental.py
# Category 1F: 複雑系・言語学・美学・音楽理論・生体力学
# (Network / Linguistics / Aesthetics / Music Theory / Biomechanics)
#
# ==================================================================
# 【Phase 9b 改修】司令塔統合 .select() 対応 (FFI overhead 削減)
# ==================================================================
#
# 目的: Phase 9 (Step B) で達成した「全特徴量 Numba UDF 直接呼び」構造を
#       保ったまま、6 モジュールの式を司令塔で 1 回の .select() に統合できる
#       よう構造を分解する。
#
# 【Phase 9b の改修】
#   追加: `_build_polars_pieces(data, lookback_bars) -> (columns, exprs, layer2)`
#     - columns: {} (空) — 1F は Polars に渡す列を持たない
#     - exprs:   [] (空) — 1F は Polars 式を持たない
#     - layer2:  全 64 特徴量を格納 (rolling UDF 直接呼び結果)
#   変更: `calculate_features` は `_build_polars_pieces` を呼んで単独計算する
#         薄いラッパーへ。columns が空でも layer2 から特徴量を取り出すロジック。
#
# 【1F の特殊性】
#   1F はローリング統計 (rolling_mean / rolling_std 等) を一切持たず、
#   全特徴量が core_indicators の rolling UDF (各 UDF 自身が rolling 計算
#   を内包) の出力である。そのため:
#
#     - 1A/1D/1E のような Polars `.select()` への集約対象が存在しない
#     - 各 UDF は最終位置の値が「直近 window 本」だけで決まる rolling 性質
#     - 学習側は `pl.col("close").map_batches(udf, return_dtype=pl.Float64)`
#       で全系列を UDF に流すが、本番側は `udf(close_arr[-w:], w)[-1]` で
#       同一値を取得できる (rolling UDF の数式定義より自明)
#
#   司令塔 _calculate_base_features の統合 .select() においても、1F の
#   columns は他モジュール (1A〜1E) の columns に何も寄与せず、exprs も
#   寄与しない。layer2 のみ統合 layer2 にマージされ、Polars 経由なしで
#   全 64 特徴量がそのまま結果に含まれる (FFI overhead ゼロ)。
#
# 【SSoT 階層】
#   Layer 1: なし (1F に該当する rolling 統計式はない)
#   Layer 2 (Numba UDF): core_indicators から import (Phase 5 確立済)
#
# 【保持される過去の修正】
#   ・QAState (apply_quality_assurance_to_group の等価実装、bias=False 補正)
#   ・energy_expenditure の入力は close.pct_change() を渡す (学習側準拠)
#   ・1F は sample_weight を持たない (学習側 base_columns 扱い、append_sample_weight
#     で別途付与される共有列のため、本ファイルでは計算しない)
# ==================================================================

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))
import blueprint as config
sys.path.append(str(config.CORE_DIR))

from core_indicators import (
    # [COMPLEX] — 全 UDF (Phase 5 で集約済み、SSoT)
    rolling_network_density_udf,
    rolling_network_clustering_udf,
    rolling_vocabulary_diversity_udf,
    rolling_linguistic_complexity_udf,
    rolling_semantic_flow_udf,
    rolling_golden_ratio_adherence_udf,
    rolling_symmetry_measure_udf,
    rolling_aesthetic_balance_udf,
    rolling_tonality_udf,
    rolling_rhythm_pattern_udf,
    rolling_harmony_udf,
    rolling_musical_tension_udf,
    rolling_kinetic_energy_udf,
    rolling_muscle_force_udf,
    rolling_biomechanical_efficiency_udf,
    rolling_energy_expenditure_udf,
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

    rolling_energy_expenditure_udf の入力に使用 (学習側 Polars
    pct_change と semantics 一致)。
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
# (1A〜1E と完全に同一の実装。Phase 9b では変更なし。)
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
        lower = ewm_mean - 5.0 * ewm_std
        upper = ewm_mean + 5.0 * ewm_std

        if is_pos_inf:
            return float(upper) if np.isfinite(upper) else 0.0
        if is_neg_inf:
            return float(lower) if np.isfinite(lower) else 0.0
        if np.isnan(raw_val):
            return 0.0

        clipped = float(np.clip(raw_val, lower, upper))
        return clipped if np.isfinite(clipped) else 0.0


# ==================================================================
# メイン計算クラス
# ==================================================================

class FeatureModule1F:

    QAState = QAState

    @staticmethod
    def _build_polars_pieces(
        data: Dict[str, np.ndarray],
        lookback_bars: int = 1440,
    ) -> Tuple[Dict[str, np.ndarray], List[pl.Expr], Dict[str, float]]:
        """
        統合 .select() 用の 3 要素を返す。

        【1F は特殊ケース】
            columns: {} (空) — Polars に渡す列を持たない
            exprs:   [] (空) — Polars 式を持たない
            layer2:  全 64 特徴量を格納 (rolling UDF 直接呼び結果)

        司令塔の統合 .select() においては、1F の columns/exprs は他モジュール
        (1A〜1E) に何も寄与せず、layer2 のみ統合結果に含まれる。Polars 経由
        なしで全 64 特徴量がそのまま結果に含まれる (FFI overhead ゼロ)。
        """
        if "close" not in data:
            return {}, [], {}

        close_arr = np.asarray(data["close"], dtype=np.float64)
        if len(close_arr) == 0:
            return {}, [], {}

        # rolling_energy_expenditure_udf の入力に使用
        # 学習側: pl.col("close").pct_change().map_batches(rolling_energy_expenditure_udf, ...)
        close_pct = _pct_change(close_arr)
        n = len(close_arr)

        # ===== columns / exprs =====
        # 1F は Polars を経由しないため空
        columns: Dict[str, np.ndarray] = {}
        exprs: List[pl.Expr] = []

        # ===== layer2 (全 64 特徴量を直接呼び) =====
        layer2: Dict[str, float] = {}

        # ---------------------------------------------------------
        # 1. ネットワーク科学系 (window=[20,30,50,100])
        # 参照: engine_1_F._create_network_science_features
        # ---------------------------------------------------------
        for w in [20, 30, 50, 100]:
            if n >= w:
                slc = close_arr[-w:]
                layer2[f"e1f_network_density_{w}"]    = float(rolling_network_density_udf(slc, w)[-1])
                layer2[f"e1f_network_clustering_{w}"] = float(rolling_network_clustering_udf(slc, w)[-1])
            else:
                layer2[f"e1f_network_density_{w}"]    = np.nan
                layer2[f"e1f_network_clustering_{w}"] = np.nan

        # ---------------------------------------------------------
        # 2. 言語学系 (window=[15,25,40,80])
        # 参照: engine_1_F._create_linguistics_features
        # ---------------------------------------------------------
        for w in [15, 25, 40, 80]:
            if n >= w:
                slc = close_arr[-w:]
                layer2[f"e1f_vocabulary_diversity_{w}"]    = float(rolling_vocabulary_diversity_udf(slc, w)[-1])
                layer2[f"e1f_linguistic_complexity_{w}"]   = float(rolling_linguistic_complexity_udf(slc, w)[-1])
                layer2[f"e1f_semantic_flow_{w}"]           = float(rolling_semantic_flow_udf(slc, w)[-1])
            else:
                layer2[f"e1f_vocabulary_diversity_{w}"]    = np.nan
                layer2[f"e1f_linguistic_complexity_{w}"]   = np.nan
                layer2[f"e1f_semantic_flow_{w}"]           = np.nan

        # ---------------------------------------------------------
        # 3. 美学系 (window=[21,34,55,89])
        # 参照: engine_1_F._create_aesthetics_features
        # ---------------------------------------------------------
        for w in [21, 34, 55, 89]:
            if n >= w:
                slc = close_arr[-w:]
                layer2[f"e1f_golden_ratio_adherence_{w}"] = float(rolling_golden_ratio_adherence_udf(slc, w)[-1])
                layer2[f"e1f_symmetry_measure_{w}"]       = float(rolling_symmetry_measure_udf(slc, w)[-1])
                layer2[f"e1f_aesthetic_balance_{w}"]      = float(rolling_aesthetic_balance_udf(slc, w)[-1])
            else:
                layer2[f"e1f_golden_ratio_adherence_{w}"] = np.nan
                layer2[f"e1f_symmetry_measure_{w}"]       = np.nan
                layer2[f"e1f_aesthetic_balance_{w}"]      = np.nan

        # ---------------------------------------------------------
        # 4. 音楽理論系 (window=[12,24,48,96])
        # 参照: engine_1_F._create_music_theory_features
        # ---------------------------------------------------------
        for w in [12, 24, 48, 96]:
            if n >= w:
                slc = close_arr[-w:]
                layer2[f"e1f_tonality_{w}"]        = float(rolling_tonality_udf(slc, w)[-1])
                layer2[f"e1f_rhythm_pattern_{w}"]  = float(rolling_rhythm_pattern_udf(slc, w)[-1])
                layer2[f"e1f_harmony_{w}"]         = float(rolling_harmony_udf(slc, w)[-1])
                layer2[f"e1f_musical_tension_{w}"] = float(rolling_musical_tension_udf(slc, w)[-1])
            else:
                layer2[f"e1f_tonality_{w}"]        = np.nan
                layer2[f"e1f_rhythm_pattern_{w}"]  = np.nan
                layer2[f"e1f_harmony_{w}"]         = np.nan
                layer2[f"e1f_musical_tension_{w}"] = np.nan

        # ---------------------------------------------------------
        # 5. 生体力学系 (window=[10,20,40,60])
        # 参照: engine_1_F._create_biomechanics_features
        # 注: energy_expenditure のみ pct_change を入力 (学習側準拠)
        # ---------------------------------------------------------
        for w in [10, 20, 40, 60]:
            if n >= w:
                slc      = close_arr[-w:]
                slc_pct  = close_pct[-w:]
                layer2[f"e1f_kinetic_energy_{w}"]            = float(rolling_kinetic_energy_udf(slc, w)[-1])
                layer2[f"e1f_muscle_force_{w}"]              = float(rolling_muscle_force_udf(slc, w)[-1])
                layer2[f"e1f_biomechanical_efficiency_{w}"]  = float(rolling_biomechanical_efficiency_udf(slc, w)[-1])
                layer2[f"e1f_energy_expenditure_{w}"]        = float(rolling_energy_expenditure_udf(slc_pct, w)[-1])
            else:
                layer2[f"e1f_kinetic_energy_{w}"]            = np.nan
                layer2[f"e1f_muscle_force_{w}"]              = np.nan
                layer2[f"e1f_biomechanical_efficiency_{w}"]  = np.nan
                layer2[f"e1f_energy_expenditure_{w}"]        = np.nan

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

        1F は columns/exprs が空のため Polars `.select()` をスキップし、
        layer2 のみで features を構築する。
        """
        columns, exprs, layer2 = FeatureModule1F._build_polars_pieces(data, lookback_bars)
        # 入力データ空 → 空 dict (1F では columns 空でも layer2 が非空なら通常ケース)
        if not columns and not layer2:
            return {}

        features: Dict[str, float] = {}
        # 1F では columns/exprs 空のためこのブロックはスキップされる
        if columns and exprs:
            df = pl.DataFrame(columns)
            polars_result = df.lazy().select(exprs).tail(1).collect().to_dicts()[0]
            for k, v in polars_result.items():
                features[k] = float(v) if v is not None else np.nan
        features.update(layer2)

        # QA 処理 (1F は sample_weight を持たないため全特徴量が QA 対象)
        if qa_state is not None:
            qa_result: Dict[str, float] = {}
            for key, val in features.items():
                qa_result[key] = qa_state.update_and_clip(key, val)
            features = qa_result
        else:
            for key in list(features.keys()):
                if not np.isfinite(features[key]):
                    features[key] = 0.0

        return features
