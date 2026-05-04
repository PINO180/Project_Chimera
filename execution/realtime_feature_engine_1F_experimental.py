# Project Cimera V5 - Feature Engine Module: 1F Experimental
# File: realtime_feature_engine_1F_experimental.py
# Description: 複雑系・音楽・運動学特徴量 (Network, Linguistics, Physics, Harmony)
#
# ▼▼ Step 13: core_indicators.py [CATEGORY: COMPLEX] へ移管済み。
#    全 UDF は core_indicators から import して使用する。
#    Single Source of Truth: このファイルに UDF 定義を持たない。
#
# 【完全一致修正】学習側 engine_1_F との全特徴量一致
#   - 欠落していた全 window の特徴量を追加
#   - linguistic_complexity / tonality / rhythm_pattern / harmony / musical_tension を追加
#   - energy_expenditure の入力を close_arr → pct_change(close_arr) に修正（学習側準拠）
#   - 対応する import を追加

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))
import blueprint as config
sys.path.append(str(config.CORE_DIR))

from core_indicators import (
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
from numba import njit
from typing import Dict, Optional


# ==================================================================
# QAState — 学習側 apply_quality_assurance_to_group の等価実装
# ==================================================================


class QAState:
    """
    学習側 apply_quality_assurance_to_group のリアルタイム等価実装。

    学習側の処理（Polars LazyFrame 全系列一括）:
        col_expr = col.replace([inf, -inf], None)  # inf → null
        ewm_mean = col_expr.ewm_mean(half_life=HL, adjust=False, ignore_nulls=True)
        ewm_std  = col_expr.ewm_std (half_life=HL, adjust=False, ignore_nulls=True)
        result   = col.clip(ewm_mean - 5*ewm_std, ewm_mean + 5*ewm_std)
                      .fill_null(0.0).fill_nan(0.0)

    alpha = 1 - exp(-ln2 / half_life)
    EWM_mean[t] = alpha * x[t] + (1-alpha) * EWM_mean[t-1]  (NaN/inf はスキップ)
    EWM_var[t]  = (1-alpha) * (EWM_var[t-1] + alpha * (x[t] - EWM_mean[t-1])^2)

    使い方:
        qa_state = FeatureModule1F.QAState(lookback_bars=1440)
        for bar in live_stream:
            features = FeatureModule1F.calculate_features(data_window, 1440, qa_state)
    """

    def __init__(self, lookback_bars: int = 1440):
        self.alpha: float = 1.0 - np.exp(-np.log(2.0) / max(lookback_bars, 1))
        self._ewm_mean: Dict[str, float] = {}
        self._ewm_var: Dict[str, float] = {}
        # bias=False 補正用: Polars ewm_std(adjust=False, bias=False) は t バー目に
        # 1 / sqrt(1 - sum_w2_t) を分散に掛ける。ここで sum_w2_t は重みの2乗和:
        #   sum_w2_t = alpha^2 * (1 - r2^t) / (1 - r2) + r2^t  (r2 = (1-alpha)^2)
        # 詳細は update_and_clip 内のコメント参照。
        self._ewm_n: Dict[str, int] = {}  # 有効値の累積更新回数（bias 補正に使用）

    def update_and_clip(self, key: str, raw_val: float) -> float:
        """1特徴量の raw_val に QA処理を適用して返す（学習側と完全一致）。"""
        alpha = self.alpha

        # 【inf 処理】学習側 Polars (修正後の engine_1_F) と一致 (Option B):
        #   学習側 engine_1_F.apply_quality_assurance_to_group の挙動:
        #     1. col_expr = pl.col(name).map_batches(
        #            lambda s: s.replace([inf,-inf], None).fill_nan(None), ...)
        #        で inf も NaN も null 化
        #     2. ewm_mean / ewm_std を col_expr から計算 → 有効値だけで EWM 進行
        #        さらに forward_fill() で inf/NaN 位置の null bounds を直前の値で埋める
        #     3. clip 適用時に inf 位置でも有効な bounds が使えるため、
        #        col.clip(lower=lower, upper=upper) で +inf → upper、-inf → lower に clip
        #     4. fill_nan(0.0) で NaN を 0.0 に置換
        #   本番側もこれに合わせる:
        #     - +inf, -inf は upper/lower bound で clip
        #     - NaN は 0.0
        #     - EWM 状態更新時は inf/NaN を除外 (= ignore_nulls=True 相当)
        is_pos_inf = np.isposinf(raw_val)
        is_neg_inf = np.isneginf(raw_val)
        # EWM 更新用: inf → NaN として扱う (replace([inf,-inf], None) 相当)
        ewm_input = np.nan if not np.isfinite(raw_val) else raw_val

        # EWM 状態更新（ignore_nulls=True 相当）
        if key not in self._ewm_mean:
            if np.isnan(ewm_input):
                # inf の場合は EWM 未初期化なので 0.0 を返す
                return 0.0
            self._ewm_mean[key] = ewm_input
            self._ewm_var[key]  = 0.0
            self._ewm_n[key]    = 1
            return ewm_input
        else:
            if not np.isnan(ewm_input):
                prev_mean = self._ewm_mean[key]
                prev_var  = self._ewm_var[key]
                new_mean  = alpha * ewm_input + (1.0 - alpha) * prev_mean
                new_var   = (1.0 - alpha) * (prev_var + alpha * (ewm_input - prev_mean) ** 2)
                self._ewm_mean[key] = new_mean
                self._ewm_var[key]  = new_var
                self._ewm_n[key]    = self._ewm_n.get(key, 0) + 1

        # ±5σ クリップ
        # Polars ewm_std(adjust=False, bias=False) の bias 補正を適用:
        #
        # 【修正前 (誤式)】
        #   bias_corr = 1 / sqrt(1 - (1-alpha)^(2n))
        #   この式は Polars の真の bias correction とズレており、
        #   ウォームアップ初期 (n < ~100バー) で σ が過小評価される。
        #
        # 【修正後 (Polars 互換式 — 実証検証で 1e-15 精度で完全一致)】
        #   adjust=False の重み: w_k = alpha*(1-alpha)^k (k=0..n-2) と
        #                       w_{n-1} = (1-alpha)^(n-1) (最古項を正規化保持)
        #   sum_w = 1 (常に)、sum_w2 = 重みの2乗和
        #   bias_factor_var = 1 / (1 - sum_w2)
        #     r2     = (1 - alpha)^2
        #     m      = n - 1                          # 漸化式は1段先送りで n-1 が正解
        #     sum_w2 = alpha^2 * (1 - r2^m) / (1 - r2) + r2^m   (m >= 1)
        #     sum_w2 = 1                               (n == 1 退化ケース)
        #     ewm_std = sqrt(ewm_var * bias_factor_var)
        #
        #   実証検証 (HL=1440, M1スキャ用デフォルト, 5000サンプル):
        #     n=2   : 真値 0.4490 / 修正後 0.4490 (差 ~1e-15)
        #     n=10  : 真値 0.5131 / 修正後 0.5131 (差 ~1e-15)
        #     n>=50 : 真値と完全一致
        ewm_mean  = self._ewm_mean[key]
        n_updates = self._ewm_n.get(key, 1)
        if n_updates <= 1:
            # n=1 は分散自体が 0 なので bias 補正不要
            ewm_std = 0.0
        else:
            r2 = (1.0 - alpha) ** 2
            m  = n_updates - 1
            if r2 < 1.0 - 1e-15:
                sum_w2 = alpha * alpha * (1.0 - r2 ** m) / (1.0 - r2) + r2 ** m
            else:
                # alpha が極端に小さい (HL → ∞) 退化ケース
                sum_w2 = 1.0
            if sum_w2 < 1.0 - 1e-15:
                bias_factor_var = 1.0 / (1.0 - sum_w2)
                ewm_std = np.sqrt(max(self._ewm_var[key] * bias_factor_var, 0.0))
            else:
                ewm_std = 0.0
        lower     = ewm_mean - 5.0 * ewm_std
        upper     = ewm_mean + 5.0 * ewm_std

        # =====================================================================
        # 【修正済み】学習側 Polars との完全一致 (Option B)
        #
        # 経緯:
        #   検証側の指摘で 2 つのバグが発覚:
        #     バグ1: チェック順序ミスで inf 入力時に upper/lower bound 分岐に到達不能
        #     バグ2: 旧 docstring の挙動が当時の学習側挙動と不整合
        #
        #   一旦 Option A (inf をそのまま通過) に修正したが、その後
        #   engine_1_F.apply_quality_assurance_to_group を:
        #     - col_expr に .fill_nan(None) を追加 (NaN も null 化、状態汚染防止)
        #     - ewm_mean / ewm_std に .forward_fill() を追加 (inf 位置でも有効 bounds)
        #   と修正したため、学習側でも inf が upper/lower で正しく clip されるようになった。
        #   それに合わせ本番側も Option B (チェック順序入れ替え + bound 置換) に変更:
        #
        #   - +inf 入力 → upper bound を返す (engine_1_F と同じ)
        #   - -inf 入力 → lower bound を返す
        #   - NaN 入力  → 0.0 を返す (fill_nan(0.0) 等価)
        #   - 有限値    → clip(lower, upper) で ±5σ クリップ
        # =====================================================================
        if is_pos_inf:
            return float(upper) if np.isfinite(upper) else 0.0
        if is_neg_inf:
            return float(lower) if np.isfinite(lower) else 0.0
        if np.isnan(raw_val):
            return 0.0

        clipped = float(np.clip(raw_val, lower, upper))
        return clipped if np.isfinite(clipped) else 0.0


@njit(fastmath=False, cache=True)
def _window(arr: np.ndarray, window: int) -> np.ndarray:
    if window <= 0:
        return np.empty(0, dtype=arr.dtype)
    if window > len(arr):
        return arr
    return arr[-window:]


@njit(fastmath=False, cache=True)
def _last(arr: np.ndarray) -> float:
    if len(arr) == 0:
        return np.nan
    return float(arr[-1])


def _pct_change(arr: np.ndarray) -> np.ndarray:
    """Polars pct_change() と完全一致: (x[i] - x[i-1]) / x[i-1]
    prev == 0 のとき inf（Polars準拠）、先頭は nan。"""
    n = len(arr)
    out = np.full(n, np.nan, dtype=np.float64)
    if n < 2:
        return out
    with np.errstate(divide="ignore", invalid="ignore"):
        out[1:] = (arr[1:] - arr[:-1]) / arr[:-1]
    return out


class FeatureModule1F:

    # 外部から FeatureModule1F.QAState としてアクセス可能にする
    QAState = QAState

    @staticmethod
    def calculate_features(
        data: Dict[str, np.ndarray],
        lookback_bars: int = 1440,
        qa_state: Optional[QAState] = None,
    ) -> Dict[str, float]:
        features = {}

        close_arr = data["close"]

        if len(close_arr) == 0:
            return features

        # pct_change（energy_expenditure の入力として使用）
        # 学習側: pl.col("close").pct_change() — Polars準拠
        close_pct = _pct_change(close_arr)

        # ---------------------------------------------------------
        # 1. ネットワーク系 (network=[20,30,50,100])
        # ---------------------------------------------------------
        for w in [20, 30, 50, 100]:
            features[f"e1f_network_density_{w}"] = _last(
                rolling_network_density_udf(_window(close_arr, w), w)
            )
            features[f"e1f_network_clustering_{w}"] = _last(
                rolling_network_clustering_udf(_window(close_arr, w), w)
            )

        # ---------------------------------------------------------
        # 2. 言語・意味系 (linguistic=[15,25,40,80])
        # ---------------------------------------------------------
        for w in [15, 25, 40, 80]:
            features[f"e1f_vocabulary_diversity_{w}"] = _last(
                rolling_vocabulary_diversity_udf(_window(close_arr, w), w)
            )
            features[f"e1f_linguistic_complexity_{w}"] = _last(
                rolling_linguistic_complexity_udf(_window(close_arr, w), w)
            )
            features[f"e1f_semantic_flow_{w}"] = _last(
                rolling_semantic_flow_udf(_window(close_arr, w), w)
            )

        # ---------------------------------------------------------
        # 3. 美学系 (aesthetic=[21,34,55,89])
        # ---------------------------------------------------------
        for w in [21, 34, 55, 89]:
            features[f"e1f_golden_ratio_adherence_{w}"] = _last(
                rolling_golden_ratio_adherence_udf(_window(close_arr, w), w)
            )
            features[f"e1f_symmetry_measure_{w}"] = _last(
                rolling_symmetry_measure_udf(_window(close_arr, w), w)
            )
            features[f"e1f_aesthetic_balance_{w}"] = _last(
                rolling_aesthetic_balance_udf(_window(close_arr, w), w)
            )

        # ---------------------------------------------------------
        # 4. 音楽理論系 (musical=[12,24,48,96])
        # ---------------------------------------------------------
        for w in [12, 24, 48, 96]:
            features[f"e1f_tonality_{w}"] = _last(
                rolling_tonality_udf(_window(close_arr, w), w)
            )
            features[f"e1f_rhythm_pattern_{w}"] = _last(
                rolling_rhythm_pattern_udf(_window(close_arr, w), w)
            )
            features[f"e1f_harmony_{w}"] = _last(
                rolling_harmony_udf(_window(close_arr, w), w)
            )
            features[f"e1f_musical_tension_{w}"] = _last(
                rolling_musical_tension_udf(_window(close_arr, w), w)
            )

        # ---------------------------------------------------------
        # 5. 生体力学系 (biomechanical=[10,20,40,60])
        #    energy_expenditure のみ pct_change を入力（学習側準拠）
        # ---------------------------------------------------------
        for w in [10, 20, 40, 60]:
            features[f"e1f_kinetic_energy_{w}"] = _last(
                rolling_kinetic_energy_udf(_window(close_arr, w), w)
            )
            features[f"e1f_muscle_force_{w}"] = _last(
                rolling_muscle_force_udf(_window(close_arr, w), w)
            )
            features[f"e1f_biomechanical_efficiency_{w}"] = _last(
                rolling_biomechanical_efficiency_udf(_window(close_arr, w), w)
            )
            # 学習側: pl.col("close").pct_change() を UDF に渡す
            features[f"e1f_energy_expenditure_{w}"] = _last(
                rolling_energy_expenditure_udf(_window(close_pct, w), w)
            )

        # ----------------------------------------------------------
        # QA処理 — 学習側 apply_quality_assurance_to_group と等価
        #   学習側: inf/-inf→null → EWM(half_life=lookback_bars)±5σクリップ → fill_null/nan(0.0)
        #   qa_state=None の場合: inf/NaN → 0.0 のみ（後方互換）
        # ----------------------------------------------------------
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
