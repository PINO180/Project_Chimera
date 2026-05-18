# realtime_feature_engine_1A_statistics.py
# Category 1A: 基本統計・分布検定系 (Basic Statistics & Distribution Tests)
#
# ==================================================================
# 【Step B 改修】numpy ベース → Polars 直呼び に全面移行
# ==================================================================
#
# 目的: 学習側 engine_1_A_a_vast_universe_of_features.py と「同じ Polars
#       Rust エンジン」で計算することで、ビット完全一致を達成する。
#
# 【背景】
#   旧実装(numpy):
#     - O(window²) の二重ループで kurtosis/moment 計算 (Group 2) → 重い
#     - rolling_mean/std/var で機械イプシロン差 (naive 実装、Welford ではない)
#     - 学習側 Polars と数値が微小に乖離する可能性
#   新実装(Polars 直呼び):
#     - 全 rolling 計算を 1 つの .select() に集約 → SIMD/Rayon 最適化
#     - Polars Rust エンジン直呼び → 学習側とビット完全一致 (Welford 系)
#     - 期待速度向上: 1029ms → 数十ms (40〜60倍)
#
# 【ATR (__temp_atr_13) のゼロ保護に関する修正】
#   旧本番側: atr_last_safe = atr_last_raw  (ゼロ保護なし)
#   学習側:    __temp_atr_13 = calculate_atr_wilder(...) + 1e-10  (ゼロ保護あり)
#   → 旧本番のコメントは誤りで、実際には学習側と 1e-11 オーダーで乖離していた。
#     新実装では学習側と同じ + 1e-10 を加えて完全一致させる。
#
# 【SSoT 階層】
#   Layer 1 (rolling 統計): 学習・本番ともに Polars Rust エンジン → ビット一致
#   Layer 2 (Numba UDF):    学習・本番ともに core_indicators から import → ビット一致
#                            (Phase 5 で確立済み、変更なし)
#
# 【保持される過去の修正】
#   ・Step 5 修正3: QAState (apply_quality_assurance_to_group の等価実装)
#   ・Step 6 修正5: ewm_std bias=False 補正の Polars 互換式
#   ・Step 6 修正6: window 条件 >= window (Polars rolling は window 本未満で NaN)
# ==================================================================


import sys
import math
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))
import numpy as np
import polars as pl
from typing import Dict, Optional

import blueprint as config
sys.path.append(str(config.CORE_DIR))
from core_indicators import (
    calculate_atr_wilder,
    calculate_mad,
    # [SSoT 統一] Engine 1A の Numba 関数を core_indicators から import
    fast_quality_score_numba,
    biweight_location_numba,
    winsorized_mean_numba,
    jarque_bera_statistic_numba,
    anderson_darling_numba,
    runs_test_numba,
    von_neumann_ratio_numba,
    basic_stabilization_numba,
    robust_stabilization_numba,
)
# ─────────────────────────────────────────────────────────────
# [Phase E (stable_rolling SSoT)] Polars rolling running 実装の
# context 長依存性を排除するため、両側で共通の Numba 関数を import。
# /workspace/core/stable_rolling.py に SSoT 配置。
# ─────────────────────────────────────────────────────────────
from stable_rolling import (
    stable_rolling_mean,
    stable_rolling_var,
    stable_rolling_std,
    stable_rolling_skew,
    stable_kurtosis_engine_formula,
    stable_moment_k_engine_formula,
)

# ─────────────────────────────────────────────────────────────
# [Plan §B.12.14.10] _pct_change SSoT 集約 — local 重複定義を撤廃。
# canonical 実装は core/numpy_helpers.py の pct_change_polars_compat。
# rfe_1A/1B/1D/1E/1F の 5 file で共通 import に統一する。
# ─────────────────────────────────────────────────────────────
from numpy_helpers import pct_change_polars_compat as _pct_change


import numba as nb


# ==================================================================
# ヘルパー関数 (Step B 後の最小セット)
#
# 【削除済み (Step B)】
#   _window           : Polars が窓スライスを内部処理するため不要
#   _last             : arr[-1] で十分のため不要
#   _skewness_bias_true: Polars rolling_skew(bias=True) を直接呼ぶため不要
# ==================================================================

@nb.njit(fastmath=False, cache=True)
def _trim_mean(arr: np.ndarray, proportiontocut: float) -> float:
    """scipy.stats.trim_mean と完全等価。最終バーのみ計算する用途で使用。

    学習側は rolling_map(scipy.stats.trim_mean, window_size=window) で全バー計算
    しているが、本番側では最終バーのスカラー値のみ必要なため Numba で高速化する
    (結果は数値的に同一)。
    """
    n = len(arr)
    k = int(np.floor(proportiontocut * n))
    sorted_arr = np.sort(arr)
    trimmed = sorted_arr[k: n - k]  # scipy と同様: n-k <= k なら空配列 → mean = nan
    if len(trimmed) == 0:
        return np.nan
    return float(np.mean(trimmed))


# ==================================================================
# QAState — 学習側 apply_quality_assurance_to_group の等価実装
# (Step B では変更なし - Layer 1/2 のロジックとは独立)
# ==================================================================

class QAState:
    """
    学習側 apply_quality_assurance_to_group のリアルタイム等価実装。

    学習側の処理（Polars LazyFrame 全系列一括）:
        col_expr = col.replace([inf, -inf], None)   # inf → null
        ema_val  = col_expr.ewm_mean(half_life=HL, ignore_nulls=True, adjust=False)
        ema_std  = col_expr.ewm_std (half_life=HL, ignore_nulls=True, adjust=False)
        result   = col.clip(ema_val - 5*ema_std, ema_val + 5*ema_std)
                      .fill_null(0.0).fill_nan(0.0)

    Polars ewm_mean(adjust=False) の再帰式（alpha = 1 - exp(-ln2 / HL)）:
        EWM_mean[t] = alpha * x[t] + (1-alpha) * EWM_mean[t-1]  （ignore_nulls: NaNはスキップ）

    Polars ewm_var(adjust=False) の再帰式（本番側 _ewm_var の等価式）:
        EWM_var[t] = (1-alpha) * EWM_var[t-1] + alpha*(1-alpha) * (x[t] - EWM_mean[t-1])^2
                   = (1-alpha) * (EWM_var[t-1] + alpha * (x[t] - EWM_mean[t-1])^2)
        EWM_std[t] = sqrt(EWM_var[t]) * bias_correction

    bias_correction (Polars ewm_std(adjust=False, bias=False) と完全一致):
        adjust=False の重み: w_k = alpha*(1-alpha)^k (k=0..n-2) と
                            w_{n-1} = (1-alpha)^(n-1) (最古項を正規化保持)
        sum_w = 1 (常に)、sum_w2 = 重みの2乗和
        bias_factor_var = 1 / (1 - sum_w2)
          r2     = (1 - alpha)^2
          m      = n - 1
          sum_w2 = alpha^2 * (1 - r2^m) / (1 - r2) + r2^m   (m >= 1)
          ewm_std = sqrt(ewm_var * bias_factor_var)

    ⚠️ 起動時のシード差:
        学習側は全系列先頭から EWM を積み上げる。
        本番側は「稼働開始時点の最初の有効値」でシードする。
        対策: lookback_bars * 3 本のウォームアップを推奨。

    使い方:
        qa_state = FeatureModule1A.QAState(lookback_bars=1440)
        # ウォームアップ
        for bar in historical_data[-lookback_bars * 3:]:
            FeatureModule1A.calculate_features(warmup_window, 1440, qa_state)
        # 本番
        for bar in live_stream:
            features = FeatureModule1A.calculate_features(data_window, 1440, qa_state)
    """

    def __init__(
        self,
        lookback_bars: int = 1440,
        artifact: Optional[Dict[str, Dict[str, float]]] = None,
    ):
        self.alpha: float = 1.0 - np.exp(-np.log(2.0) / max(lookback_bars, 1))
        self._ewm_mean: Dict[str, float] = {}
        self._ewm_var: Dict[str, float] = {}
        self._ewm_n: Dict[str, int] = {}

        # ─────────────────────────────────────────────────────────────
        # [Phase 9d 発見 #66 Phase D-3] 学習側 QAState seed artifact の load
        # ─────────────────────────────────────────────────────────────
        # artifact が渡された場合、各 feature の EWM 状態を学習側 5 年分の
        # 成熟状態で初期化する。これにより本番側 QAState の seed 不足が解消
        # され、Train-Serve Skew が根治される。
        #
        # artifact の形式:
        #   {feature_name: {"ewm_mean": float, "ewm_var": float, "ewm_n": int}}
        # キーは production の update_and_clip(key=...) と完全一致する feature
        # 名 (例: "e1a_statistical_kurtosis_20")。
        #
        # _artifact_loaded フラグ:
        #   warmup loop で update_and_clip(skip_update=True) を指示するため、
        #   司令塔 (process_new_m05_bar / _replace_buffer_from_dataframe) が
        #   このフラグを参照する。warmup 期間中の追加 update を抑止すれば、
        #   artifact の状態がそのまま維持され、学習側との数値完全一致が保証
        #   される (1e-15 精度)。
        # ─────────────────────────────────────────────────────────────
        self._artifact_loaded: bool = False
        if artifact is not None:
            for feat_name, state in artifact.items():
                # 防御的型変換 (pickle 経由で float64 / int64 のはずだが念のため)
                self._ewm_mean[feat_name] = float(state["ewm_mean"])
                self._ewm_var[feat_name] = float(state["ewm_var"])
                self._ewm_n[feat_name] = int(state["ewm_n"])
            self._artifact_loaded = True

    def update_and_clip(
        self, key: str, raw_val: float, skip_update: bool = False
    ) -> float:
        """1特徴量の raw_val に対して QA処理を適用し、処理済みスカラーを返す。

        Args:
            key: feature 名 (例: "e1a_statistical_kurtosis_20")
            raw_val: pre-clip の生値
            skip_update: True の場合、EWM 状態を更新せずに clip のみ適用。
                Phase D-3 で artifact 由来の状態を warmup loop で破壊しないため
                に司令塔から渡される。デフォルト False (旧挙動互換)。
        """
        alpha = self.alpha

        # 学習側 col.replace([inf,-inf], None) 相当
        is_pos_inf = np.isposinf(raw_val)
        is_neg_inf = np.isneginf(raw_val)
        ewm_input = np.nan if not np.isfinite(raw_val) else raw_val

        # EWM 状態更新 (skip_update=True なら state を変更しない)
        if key not in self._ewm_mean:
            # key 未初期化: artifact が無いケース or artifact に存在しない feature
            if np.isnan(ewm_input):
                return 0.0
            if not skip_update:
                self._ewm_mean[key] = ewm_input
                self._ewm_var[key] = 0.0
                self._ewm_n[key] = 1
            # 初回 update は clip 範囲未確立 → そのまま返す (旧挙動)
            return ewm_input
        else:
            if not np.isnan(ewm_input) and not skip_update:
                prev_mean = self._ewm_mean[key]
                prev_var = self._ewm_var[key]
                new_mean = alpha * ewm_input + (1.0 - alpha) * prev_mean
                new_var = (1.0 - alpha) * (
                    prev_var + alpha * (ewm_input - prev_mean) ** 2
                )
                self._ewm_mean[key] = new_mean
                self._ewm_var[key] = new_var
                self._ewm_n[key] = self._ewm_n.get(key, 0) + 1

        # ±5σ クリップ (bias=False 補正適用)
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


# ════════════════════════════════════════════════════════════════
# [Phase E (stable_rolling SSoT)] Polars expression helpers
# ────────────────────────────────────────────────────────────────
# ════════════════════════════════════════════════════════════════
# [§B.12.13 Option C2] helper 関数 (_srm_expr 等) 削除済み
#
# 旧実装では Polars map_batches 経由で Numba 関数を呼んでいたが、
# 6 engine 統合 select で Polars Rust の最適化 plan が
# non-deterministic になる原因が判明 (§B.12.13)。
#
# 修正方針 (Option C2):
#   - Numba 関数を numpy で直呼びして DataFrame 列として注入
#   - Polars expression は単純な列演算 (pl.col のみ) に変更
#   - これで Polars CSE / Slice Pushdown / Rayon 等の
#     最適化 plan の非決定性に依存しない構造になる
#
# 数値結果は map_batches 経由と bit-identical (§B.12.12.7 実機実証済)。
# SSoT は Layer 2 (stable_rolling.py / core_indicators.py の Numba 関数) で維持。
# 学習側 engine_1_A は Polars map_batches 経由のまま (性能維持、3.4M bars 一括処理)。
# Layer 1 で経路非対称となるが、Polars version up が本番に直撃しない防御力強化。
# ════════════════════════════════════════════════════════════════


class FeatureModule1A:

    # 外部から FeatureModule1A.QAState としてアクセス可能にする
    QAState = QAState

    @staticmethod
    def _build_polars_pieces(
        data: Dict[str, np.ndarray],
        lookback_bars: int = 1440,
    ) -> tuple:
        """
        統合 .select() 用の 3 要素を返す。
        司令塔は本メソッドを直接呼び、全 6 モジュールから収集した columns/exprs を
        統合 DataFrame で 1 度の .select() で計算する。

        Returns:
            columns: Dict[str, np.ndarray] — DataFrame に追加する列 (close/__temp_atr_13/volume)
            exprs:   List[pl.Expr] — Polars 式リスト (alias は最終特徴量名)
            layer2:  Dict[str, float] — Polars 経由しないスカラー特徴量
                     (Numba UDF 直接呼び結果 + 後処理が必要なもの)
        """
        close_arr = data["close"].astype(np.float64)
        if len(close_arr) == 0:
            return {}, [], {}

        high_arr = (
            data["high"].astype(np.float64)
            if len(data.get("high", [])) > 0
            else np.array([], dtype=np.float64)
        )
        low_arr = (
            data["low"].astype(np.float64)
            if len(data.get("low", [])) > 0
            else np.array([], dtype=np.float64)
        )
        volume_arr = (
            data["volume"].astype(np.float64)
            if len(data.get("volume", [])) > 0
            else np.array([], dtype=np.float64)
        )

        # ATR13 (学習側 __temp_atr_13 と完全一致 = + 1e-10)
        if len(high_arr) > 0 and len(low_arr) > 0:
            atr_arr_raw = calculate_atr_wilder(high_arr, low_arr, close_arr, 13)
            atr_arr = atr_arr_raw + 1e-10
        else:
            atr_arr = np.full(len(close_arr), np.nan, dtype=np.float64)

        atr_last = float(atr_arr[-1]) if np.isfinite(atr_arr[-1]) else np.nan
        close_last = float(close_arr[-1])

        # ===== columns =====
        columns = {
            "close": close_arr,
            "__temp_atr_13": atr_arr,
        }
        if len(volume_arr) > 0:
            columns["volume"] = volume_arr

        # ▼▼ [§B.12.13 Option C2] numpy 事前計算 + 列注入 ▼▼
        # Polars map_batches 経由の Numba 直呼びを排除し、numpy で先に
        # 計算してから DataFrame 列として注入する。
        # 命名規約: __num_<func>_<col>_<window>  (例: __num_srm_close_10)
        #            __num_smom<moment>_<col>_<window>
        # 6 engine 統合 select 内では pl.col() 参照のみとなり、
        # Polars Rust の最適化 plan の非決定性に依存しない。

        # Group 1 + Group 6 共通: stable_rolling_{mean, std} の全 window
        for _w in [5, 10, 20, 50, 100]:
            columns[f"__num_srm_close_{_w}"] = stable_rolling_mean(close_arr, _w)
            columns[f"__num_srs_close_{_w}"] = stable_rolling_std(close_arr, _w, 1)

        # Group 1: stable_rolling_var (window 10, 20, 50)
        for _w in [10, 20, 50]:
            columns[f"__num_srv_close_{_w}"] = stable_rolling_var(close_arr, _w, 1)

        # Group 2: skewness, kurtosis, moments (window 20, 50)
        for _w in [20, 50]:
            columns[f"__num_sskew_close_{_w}"] = stable_rolling_skew(close_arr, _w)
            columns[f"__num_skurt_close_{_w}"] = stable_kurtosis_engine_formula(close_arr, _w)
            for _m in [5, 6, 7, 8]:
                columns[f"__num_smom{_m}_close_{_w}"] = stable_moment_k_engine_formula(close_arr, _w, _m)

        # Group 6: volume 系 (volume があるときのみ)
        if len(volume_arr) > 0:
            for _w in [5, 10, 20, 50, 100]:
                columns[f"__num_srm_volume_{_w}"] = stable_rolling_mean(volume_arr, _w)
            columns["__num_srm_volume_lookback"] = stable_rolling_mean(volume_arr, lookback_bars)
        # ▲▲ [§B.12.13 Option C2] ▲▲

        # ===== exprs (Group 1, 2, 3 (median/q25/q75/iqr), 6) =====
        exprs = []

        # ▼▼ [Phase E (stable_rolling SSoT)] Polars rolling_* の context 長依存を排除 ▼▼
        # Group 1: 統計的モーメント (stable Numba 実装 — Option C2 で numpy 事前計算)
        for window in [10, 20, 50]:
            exprs.append(
                ((pl.col("close") - pl.col(f"__num_srm_close_{window}"))
                 / pl.col("__temp_atr_13"))
                .alias(f"e1a_statistical_mean_{window}")
            )
            exprs.append(
                (pl.col(f"__num_srv_close_{window}")
                 / pl.col("__temp_atr_13").pow(2))
                .alias(f"e1a_statistical_variance_{window}")
            )
            exprs.append(
                (pl.col(f"__num_srs_close_{window}")
                 / pl.col("__temp_atr_13"))
                .alias(f"e1a_statistical_std_{window}")
            )
            exprs.append(
                (pl.col(f"__num_srs_close_{window}")
                 / (pl.col(f"__num_srm_close_{window}") + 1e-10))
                .alias(f"e1a_statistical_cv_{window}")
            )

        # Group 2: 歪度・尖度・高次モーメント (stable Numba 実装 — Option C2 で numpy 事前計算)
        for window in [20, 50]:
            exprs.append(
                pl.col(f"__num_sskew_close_{window}")
                .alias(f"e1a_statistical_skewness_{window}")
            )
            exprs.append(
                pl.col(f"__num_skurt_close_{window}")
                .alias(f"e1a_statistical_kurtosis_{window}")
            )
            for moment in [5, 6, 7, 8]:
                exprs.append(
                    pl.col(f"__num_smom{moment}_close_{window}")
                    .alias(f"e1a_statistical_moment_{moment}_{window}")
                )
        # ▲▲ [Phase E (stable_rolling SSoT)] ▲▲

        # Group 3 (Polars部分): median/q25/q75/iqr
        for window in [10, 20, 50]:
            exprs.append(
                ((pl.col("close") - pl.col("close").rolling_median(window))
                 / pl.col("__temp_atr_13"))
                .alias(f"e1a_robust_median_{window}")
            )
            q25_expr = pl.col("close").rolling_quantile(0.25, window_size=window)
            q75_expr = pl.col("close").rolling_quantile(0.75, window_size=window)
            exprs.append(
                ((pl.col("close") - q25_expr) / pl.col("__temp_atr_13"))
                .alias(f"e1a_robust_q25_{window}")
            )
            exprs.append(
                ((pl.col("close") - q75_expr) / pl.col("__temp_atr_13"))
                .alias(f"e1a_robust_q75_{window}")
            )
            exprs.append(
                ((q75_expr - q25_expr) / pl.col("__temp_atr_13"))
                .alias(f"e1a_robust_iqr_{window}")
            )

        # ▼▼ [Phase E (stable_rolling SSoT)] Group 6 fast_rolling active path ▼▼
        # (Option C2: numpy 事前計算 + pl.col 参照)
        for window in [5, 10, 20, 50, 100]:
            exprs.append(
                ((pl.col("close") - pl.col(f"__num_srm_close_{window}"))
                 / pl.col("__temp_atr_13"))
                .alias(f"e1a_fast_rolling_mean_{window}")
            )
            exprs.append(
                (pl.col(f"__num_srs_close_{window}")
                 / pl.col("__temp_atr_13"))
                .alias(f"e1a_fast_rolling_std_{window}")
            )
            if len(volume_arr) > 0:
                exprs.append(
                    (pl.col(f"__num_srm_volume_{window}")
                     / (pl.col("__num_srm_volume_lookback") + 1e-10))
                    .alias(f"e1a_fast_volume_mean_{window}")
                )
        # ▲▲ [Phase E (stable_rolling SSoT)] ▲▲

        # ===== layer2 (Numba UDF 直接呼び + 後処理) =====
        layer2: Dict[str, float] = {}

        # Group 3 trimmed_mean
        for window in [10, 20, 50]:
            if len(close_arr) >= window:
                trim_w = _trim_mean(close_arr[-window:], 0.1)
                layer2[f"e1a_robust_trimmed_mean_{window}"] = (
                    (close_last - trim_w) / atr_last if np.isfinite(trim_w) else np.nan
                )
            else:
                layer2[f"e1a_robust_trimmed_mean_{window}"] = np.nan

        # Volume なしの場合の fast_volume_mean フォールバック
        if len(volume_arr) == 0:
            for window in [5, 10, 20, 50, 100]:
                layer2[f"e1a_fast_volume_mean_{window}"] = np.nan

        # Group 4
        mad_arr = calculate_mad(close_arr, 20)
        layer2["e1a_robust_mad_20"] = (
            float(mad_arr[-1]) / atr_last
            if len(mad_arr) > 0 and np.isfinite(mad_arr[-1])
            else np.nan
        )
        biweight_arr = biweight_location_numba(close_arr)
        layer2["e1a_robust_biweight_location_20"] = (
            (close_last - float(biweight_arr[-1])) / atr_last
            if np.isfinite(biweight_arr[-1]) else np.nan
        )
        winsorized_arr = winsorized_mean_numba(close_arr)
        layer2["e1a_robust_winsorized_mean_20"] = (
            (close_last - float(winsorized_arr[-1])) / atr_last
            if np.isfinite(winsorized_arr[-1]) else np.nan
        )

        # Group 5
        pct_arr = _pct_change(close_arr)
        jb_arr = jarque_bera_statistic_numba(pct_arr)
        layer2["e1a_jarque_bera_statistic_50"] = float(jb_arr[-1])
        ad_arr = anderson_darling_numba(pct_arr)
        layer2["e1a_anderson_darling_statistic_30"] = float(ad_arr[-1])
        runs_arr = runs_test_numba(pct_arr)
        layer2["e1a_runs_test_statistic_30"] = float(runs_arr[-1])
        vn_arr = von_neumann_ratio_numba(pct_arr)
        layer2["e1a_von_neumann_ratio_30"] = float(vn_arr[-1])

        # Group 7
        qs_arr = fast_quality_score_numba(close_arr)
        layer2["e1a_fast_quality_score_50"] = float(qs_arr[-1])

        # Group 8
        bs_arr = basic_stabilization_numba(close_arr)
        layer2["e1a_fast_basic_stabilization"] = (
            (close_last - float(bs_arr[-1])) / atr_last
            if np.isfinite(bs_arr[-1]) else np.nan
        )
        rs_arr = robust_stabilization_numba(close_arr)
        layer2["e1a_fast_robust_stabilization"] = (
            (close_last - float(rs_arr[-1])) / atr_last
            if np.isfinite(rs_arr[-1]) else np.nan
        )

        return columns, exprs, layer2

    @staticmethod
    def calculate_features(
        data: Dict[str, np.ndarray],
        lookback_bars: int = 1440,
        qa_state: Optional[QAState] = None,
    ) -> Dict[str, float]:
        """
        【Phase 9b 改修版】単独計算用ラッパー。
        司令塔は _build_polars_pieces を直接呼んで全モジュール統合 .select() を行うが、
        本メソッドは後方互換のためモジュール単独で動作する形を維持する。
        """
        columns, exprs, layer2 = FeatureModule1A._build_polars_pieces(data, lookback_bars)
        if not columns:
            return {}

        df = pl.DataFrame(columns)
        result_df = df.lazy().select(exprs).tail(1).collect()
        polars_result = result_df.to_dicts()[0]

        features: Dict[str, float] = {}
        for k, v in polars_result.items():
            features[k] = float(v) if v is not None else np.nan
        features.update(layer2)

        # QA 処理
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

