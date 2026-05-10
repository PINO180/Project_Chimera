# realtime_feature_engine_1F_experimental.py
# Category 1F: 複雑系・言語学・美学・音楽理論・生体力学
# (Network / Linguistics / Aesthetics / Music Theory / Biomechanics)
#
# ==================================================================
# 【Step B 改修ノート】
# ==================================================================
#
# 目的: 学習側 engine_1_F_a_vast_universe_of_features.py と
#       「同じ Numba UDF (core_indicators)」で計算することで、
#       ビット完全一致を達成する。
#
# 【1F の特性】
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
#   結果として 1F の本番側は:
#     - core_indicators UDF を直接呼び出し (numpy スライス)
#     - Polars は経由しない (= FFI overhead ゼロ)
#     - 検証済み: engine_1_F.CalculationEngine と全 64 特徴量で完全一致
#
# 【SSoT 階層】
#   Layer 1: なし (1F に該当する rolling 統計式はない)
#   Layer 2 (Numba UDF): core_indicators から import (Phase 5 確立済)
#
# 【保持される過去の修正】
#   ・QAState (apply_quality_assurance_to_group の等価実装、bias=False 補正、Option B)
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
# (Step B では変更なし - Layer 2 のロジックとは独立)
# ==================================================================

class QAState:
    """
    学習側 apply_quality_assurance_to_group のリアルタイム等価実装。

    学習側の処理 (Polars LazyFrame 全系列一括):
        col_expr = pl.col(name).map_batches(
            lambda s: s.replace([inf,-inf], None).fill_nan(None), ...)
        ewm_mean = col_expr.ewm_mean(half_life=HL, adjust=False, ignore_nulls=True)
                                .forward_fill()
        ewm_std  = col_expr.ewm_std (half_life=HL, adjust=False, ignore_nulls=True)
                                .forward_fill()
        result   = col.clip(lower=ewm_mean - 5*ewm_std, upper=ewm_mean + 5*ewm_std)
                      .fill_null(0.0).fill_nan(0.0)

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
        lower = ewm_mean - 5.0 * ewm_std
        upper = ewm_mean + 5.0 * ewm_std

        # +inf → upper, -inf → lower, NaN → 0.0, 有限値 → clip
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
    def calculate_features(
        data: Dict[str, np.ndarray],
        lookback_bars: int = 1440,
        qa_state: Optional[QAState] = None,
    ) -> Dict[str, float]:
        """
        【Step B改修版】Numba UDF 直接呼び出しで全特徴量を計算する。

        アーキテクチャ:
            Layer 2 のみ (1F は Layer 1 を持たない):
                各 UDF を close_arr[-w:] (または close_pct[-w:]) に直接適用し、
                最終バーの値を取得する。学習側の
                  pl.col("close").map_batches(udf, return_dtype=pl.Float64)
                は同一 UDF を全系列に適用するが、UDF の rolling 性質により
                本番側の最終バー結果と完全一致する。

        Args:
            data         : close の numpy 配列を含む辞書 (1F は他列を使用しない)
            lookback_bars: QA の EWM 半減期に使用 (1日あたりのバー数)
            qa_state     : QAState インスタンス。本番稼働時は必ず渡し、
                           同一インスタンスを毎バー使い回すこと。
        """
        features: Dict[str, float] = {}

        close_arr = data["close"].astype(np.float64)
        if len(close_arr) == 0:
            return features

        # rolling_energy_expenditure_udf の入力に使用
        # 学習側: pl.col("close").pct_change().map_batches(rolling_energy_expenditure_udf, ...)
        close_pct = _pct_change(close_arr)
        n = len(close_arr)

        # =====================================================================
        # 1. ネットワーク科学系 (window=[20,30,50,100])
        # 参照: engine_1_F._create_network_science_features
        # =====================================================================
        for w in [20, 30, 50, 100]:
            if n >= w:
                slc = close_arr[-w:]
                features[f"e1f_network_density_{w}"]    = float(rolling_network_density_udf(slc, w)[-1])
                features[f"e1f_network_clustering_{w}"] = float(rolling_network_clustering_udf(slc, w)[-1])
            else:
                features[f"e1f_network_density_{w}"]    = np.nan
                features[f"e1f_network_clustering_{w}"] = np.nan

        # =====================================================================
        # 2. 言語学系 (window=[15,25,40,80])
        # 参照: engine_1_F._create_linguistics_features
        # =====================================================================
        for w in [15, 25, 40, 80]:
            if n >= w:
                slc = close_arr[-w:]
                features[f"e1f_vocabulary_diversity_{w}"]    = float(rolling_vocabulary_diversity_udf(slc, w)[-1])
                features[f"e1f_linguistic_complexity_{w}"]   = float(rolling_linguistic_complexity_udf(slc, w)[-1])
                features[f"e1f_semantic_flow_{w}"]           = float(rolling_semantic_flow_udf(slc, w)[-1])
            else:
                features[f"e1f_vocabulary_diversity_{w}"]    = np.nan
                features[f"e1f_linguistic_complexity_{w}"]   = np.nan
                features[f"e1f_semantic_flow_{w}"]           = np.nan

        # =====================================================================
        # 3. 美学系 (window=[21,34,55,89])
        # 参照: engine_1_F._create_aesthetics_features
        # =====================================================================
        for w in [21, 34, 55, 89]:
            if n >= w:
                slc = close_arr[-w:]
                features[f"e1f_golden_ratio_adherence_{w}"] = float(rolling_golden_ratio_adherence_udf(slc, w)[-1])
                features[f"e1f_symmetry_measure_{w}"]       = float(rolling_symmetry_measure_udf(slc, w)[-1])
                features[f"e1f_aesthetic_balance_{w}"]      = float(rolling_aesthetic_balance_udf(slc, w)[-1])
            else:
                features[f"e1f_golden_ratio_adherence_{w}"] = np.nan
                features[f"e1f_symmetry_measure_{w}"]       = np.nan
                features[f"e1f_aesthetic_balance_{w}"]      = np.nan

        # =====================================================================
        # 4. 音楽理論系 (window=[12,24,48,96])
        # 参照: engine_1_F._create_music_theory_features
        # =====================================================================
        for w in [12, 24, 48, 96]:
            if n >= w:
                slc = close_arr[-w:]
                features[f"e1f_tonality_{w}"]        = float(rolling_tonality_udf(slc, w)[-1])
                features[f"e1f_rhythm_pattern_{w}"]  = float(rolling_rhythm_pattern_udf(slc, w)[-1])
                features[f"e1f_harmony_{w}"]         = float(rolling_harmony_udf(slc, w)[-1])
                features[f"e1f_musical_tension_{w}"] = float(rolling_musical_tension_udf(slc, w)[-1])
            else:
                features[f"e1f_tonality_{w}"]        = np.nan
                features[f"e1f_rhythm_pattern_{w}"]  = np.nan
                features[f"e1f_harmony_{w}"]         = np.nan
                features[f"e1f_musical_tension_{w}"] = np.nan

        # =====================================================================
        # 5. 生体力学系 (window=[10,20,40,60])
        # 参照: engine_1_F._create_biomechanics_features
        # 注: energy_expenditure のみ pct_change を入力 (学習側準拠)
        # =====================================================================
        for w in [10, 20, 40, 60]:
            if n >= w:
                slc      = close_arr[-w:]
                slc_pct  = close_pct[-w:]
                features[f"e1f_kinetic_energy_{w}"]            = float(rolling_kinetic_energy_udf(slc, w)[-1])
                features[f"e1f_muscle_force_{w}"]              = float(rolling_muscle_force_udf(slc, w)[-1])
                features[f"e1f_biomechanical_efficiency_{w}"]  = float(rolling_biomechanical_efficiency_udf(slc, w)[-1])
                features[f"e1f_energy_expenditure_{w}"]        = float(rolling_energy_expenditure_udf(slc_pct, w)[-1])
            else:
                features[f"e1f_kinetic_energy_{w}"]            = np.nan
                features[f"e1f_muscle_force_{w}"]              = np.nan
                features[f"e1f_biomechanical_efficiency_{w}"]  = np.nan
                features[f"e1f_energy_expenditure_{w}"]        = np.nan

        # =====================================================================
        # QA 処理 (学習側 apply_quality_assurance_to_group と等価)
        # 1F は sample_weight を持たないため、全特徴量が QA 対象。
        # qa_state=None の場合は inf/NaN → 0.0 のフォールバックのみ。
        # =====================================================================
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
