"""
realtime_feature_engine_1B_timeseries.py
Project Cimera V5 - 特徴量エンジンモジュール 【1B: 時系列・分布系】

==================================================================
【Phase 9b 改修】司令塔統合 .select() 対応 (FFI overhead 削減)
==================================================================

目的: Phase 9 (Step B) で達成した Polars 直呼びによる学習側との
      ビット完全一致を保ったまま、6 モジュールの Polars 式を司令塔
      で 1 回の .select() に統合できるよう構造を分解する。

【Phase 9b の改修】
  追加: `_build_polars_pieces(data, lookback_bars) -> (columns, exprs, layer2)`
    - columns: DataFrame に追加する列辞書 (close/__temp_atr_safe/high/low/volume)
    - exprs:   Polars 式リスト (各 alias は最終特徴量名 e1b_*)
    - layer2:  Polars 経由しないスカラー特徴量 (Numba UDF 直接呼び結果)
  変更: `calculate_features` は `_build_polars_pieces` を呼んで単独計算する
        薄いラッパーへ。後方互換は保つ (戻り値は従来と完全同一)。

【特殊事項】
  e1b_t_dist_scale_50 は学習側で:
    pct_change.rolling_map(t分布_尺度_udf, ..., 50)
      / (stable_rolling_std(pct_change, 20) + 1e-10)
  と Polars 内で完結している (Phase E 適用後)。本番側でも分子は Numba UDF を
  numpy で precompute してスカラー化 (rolling_map の Python ループを回避する
  ため) し、pl.lit で Polars 式に注入。分母は Phase E (Option C2) で numpy
  事前計算した `__num_srs_pct_close_20` 列を参照 (e1b_volatility_20 と同じ列
  を共有 — CSE 不要で明示的共有)、割り算は Polars 内で実行する (学習側と数値
  完全一致)。

【Phase E (stable_rolling SSoT) 適用】
  Polars 組込 rolling_{mean,var,std} は内部 running 累積実装で context 長
  依存性がある (学習側 3.4M bars と本番側 ~2980 bars deque で結果が乖離)。
  本ファイルでは全 rolling_{mean,var,std} 呼出しを Option C2 で置換:
    - numpy で stable_rolling_{mean,var,std} を事前計算 → `__num_*` 列に注入
    - expression 内では `pl.col("__num_...")` で参照のみ
    - map_batches は完全排除 (CSE non-determinism 回避)
  rolling_{median,min,max,quantile} は context 長非依存のため変更なし。

【SSoT 階層】(§B.12.13.9 で再定義)
  Layer 1 (経路): 学習側 = Polars map_batches、本番側 = numpy 直呼び + 列注入
  Layer 2 (真の SSoT): stable_rolling.py (Numba 関数) + core_indicators.py
==================================================================
"""

import sys
from pathlib import Path

import numpy as np
import polars as pl
from typing import Dict, Optional, Tuple, List
import logging

sys.path.append(str(Path(__file__).resolve().parent.parent))
import blueprint as config
sys.path.append(str(config.CORE_DIR))
from core_indicators import (
    calculate_atr_wilder,
    # [SSoT 統一] Engine 1B の UDF を core_indicators から import
    adf_統計量_udf,
    phillips_perron_統計量_udf,
    kpss_統計量_udf,
    t分布_自由度_udf,
    t分布_尺度_udf,
    gev_形状_udf,
    holt_winters_レベル_udf,
    holt_winters_トレンド_udf,
    arima_残差分散_udf,
    kalman_状態推定_udf,
    lowess_適合値_udf,
    theil_sen_傾き_udf,
)
# ─────────────────────────────────────────────────────────────
# [Phase E (stable_rolling SSoT)] Polars rolling running 実装の
# context 長依存性を排除するため、両側で共通の Numba 関数を import。
# 本番側は Option C2: numpy 事前計算 + 列注入で使用 (map_batches 完全排除)。
# ─────────────────────────────────────────────────────────────
from stable_rolling import (
    stable_rolling_mean,
    stable_rolling_var,
    stable_rolling_std,
)

# ─────────────────────────────────────────────────────────────
# [Plan §B.12.14.10] _pct_change SSoT 集約 — local 重複定義を撤廃。
# canonical 実装は core/numpy_helpers.py の pct_change_polars_compat。
# rfe_1A/1B/1D/1E/1F の 5 file で共通 import に統一する。
# ─────────────────────────────────────────────────────────────
from numpy_helpers import pct_change_polars_compat as _pct_change

logger = logging.getLogger("ProjectForge.FeatureEngine.1B")


# ==================================================================
# ヘルパー関数
# ==================================================================


# ==================================================================
# QAState — 学習側 apply_quality_assurance_to_group の等価実装
# (1A と同一実装。Phase 9b では変更なし。)
# ==================================================================

class QAState:
    """
    学習側 apply_quality_assurance_to_group のリアルタイム等価実装。
    詳細は realtime_feature_engine_1A_statistics.py の QAState を参照。
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
        # 1A と完全に同じ仕組み (詳細は realtime_feature_engine_1A_statistics.py
        # の QAState を参照)。 artifact が渡された場合、各 feature の EWM 状態を
        # 学習側 5 年分の成熟状態で初期化する。これにより本番側 QAState の
        # seed 不足が解消され、Train-Serve Skew が根治される。
        # ─────────────────────────────────────────────────────────────
        self._artifact_loaded: bool = False
        if artifact is not None:
            for feat_name, state in artifact.items():
                self._ewm_mean[feat_name] = float(state["ewm_mean"])
                self._ewm_var[feat_name] = float(state["ewm_var"])
                self._ewm_n[feat_name] = int(state["ewm_n"])
            self._artifact_loaded = True

    def update_and_clip(
        self, key: str, raw_val: float, skip_update: bool = False
    ) -> float:
        alpha = self.alpha

        is_pos_inf = np.isposinf(raw_val)
        is_neg_inf = np.isneginf(raw_val)
        ewm_input = np.nan if not np.isfinite(raw_val) else raw_val

        if key not in self._ewm_mean:
            # key 未初期化: artifact 不在 or artifact に存在しない feature
            if np.isnan(ewm_input):
                return 0.0
            if not skip_update:
                self._ewm_mean[key] = ewm_input
                self._ewm_var[key]  = 0.0
                self._ewm_n[key]    = 1
            # 初回 update は clip 範囲未確立 → そのまま返す (旧挙動)
            return ewm_input
        else:
            if not np.isnan(ewm_input) and not skip_update:
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

class FeatureModule1B:
    # window_sizes["general"] = [10, 20, 50, 100]（学習側と同一）
    GENERAL_WINDOWS = [10, 20, 50, 100]

    # 外部から FeatureModule1B.QAState としてアクセス可能にする
    QAState = QAState

    @staticmethod
    def _build_polars_pieces(
        data: Dict[str, np.ndarray],
        lookback_bars: int = 1440,
    ) -> Tuple[Dict[str, np.ndarray], List[pl.Expr], Dict[str, float]]:
        """
        統合 .select() 用の 3 要素を返す。
        司令塔は本メソッドを直接呼び、全 6 モジュールから収集した
        columns/exprs を統合 DataFrame で 1 度の .select() で計算する。

        Returns:
            columns: Dict[str, np.ndarray]
                DataFrame に追加する列辞書。共通列 (close/high/low/volume) と
                1B 固有の `__temp_atr_safe` (= ATR13 + 1e-10) を含む。
            exprs: List[pl.Expr]
                Polars 式リスト。各 alias は最終特徴量名 (e1b_*)。
            layer2: Dict[str, float]
                Polars 経由しないスカラー特徴量 (Numba UDF + 後処理)。
        """
        # ---------------------------------------------------------
        # 入力配列の準備 (contiguous float64 化)
        # ---------------------------------------------------------
        close_arr = np.asarray(data["close"], dtype=np.float64)
        if len(close_arr) == 0:
            return {}, [], {}

        high_arr = (
            np.asarray(data["high"], dtype=np.float64)
            if len(data.get("high", [])) > 0
            else np.array([], dtype=np.float64)
        )
        low_arr = (
            np.asarray(data["low"], dtype=np.float64)
            if len(data.get("low", [])) > 0
            else np.array([], dtype=np.float64)
        )
        volume_arr = (
            np.asarray(data["volume"], dtype=np.float64)
            if len(data.get("volume", [])) > 0
            else np.array([], dtype=np.float64)
        )

        n = len(close_arr)
        close_last = float(close_arr[-1])

        # pct_change を一度だけ計算 (Layer 2 と pct_std_20 で再利用)
        close_pct = _pct_change(close_arr)

        # ---------------------------------------------------------
        # ATR13 計算 (学習側 inject_temp_atr と完全一致)
        #   __temp_atr_safe = calculate_atr_wilder(high, low, close, 13) + 1e-10
        # ---------------------------------------------------------
        if len(high_arr) > 0 and len(low_arr) > 0:
            atr_arr_raw = calculate_atr_wilder(high_arr, low_arr, close_arr, 13)
            atr_safe_arr = atr_arr_raw + 1e-10
        else:
            atr_safe_arr = np.full(n, np.nan, dtype=np.float64)

        atr_last = float(atr_safe_arr[-1]) if np.isfinite(atr_safe_arr[-1]) else np.nan
        atr_ok = np.isfinite(atr_last)
        atr2_last = atr_last ** 2 if atr_ok else np.nan

        # ===== columns =====
        columns: Dict[str, np.ndarray] = {
            "close": close_arr,
            "__temp_atr_safe": atr_safe_arr,
        }
        if len(high_arr) > 0:
            columns["high"] = high_arr
        if len(low_arr) > 0:
            columns["low"] = low_arr
        if len(volume_arr) > 0:
            columns["volume"] = volume_arr

        # ▼▼ [§B.12.13.7 Option C2 / Phase E] numpy 事前計算 + 列注入 ▼▼
        # Polars map_batches を経由しない (CSE non-determinism 回避)。
        # 命名規約:
        #   単純: __num_<func>_<col>_<window>  (例: __num_srm_close_10)
        #   複合: __num_<func>_<expr_id>_<window>  (例: __num_srs_pct_close_20)

        # --- 単純パターン: close に対する rolling_{mean, std, var} ---
        # window 値リスト:
        #   - basic_stats: GENERAL_WINDOWS = [10, 20, 50, 100] (mean/std/var)
        #   - composite (zscore/bollinger): [20, 50] (mean/std) — 上記に含まれる
        for _w in FeatureModule1B.GENERAL_WINDOWS:
            columns[f"__num_srm_close_{_w}"] = stable_rolling_mean(close_arr, _w)
            columns[f"__num_srs_close_{_w}"] = stable_rolling_std(close_arr, _w, 1)
            # rolling_var(W, ddof=1) は学習側と同じく stable_rolling_var を直接呼ぶ。
            # std**2 は IEEE 754 で sqrt(x)*sqrt(x) != x のため 1 ULP ズレうるので使わない
            # (学習側 _srv_expr → stable_rolling_var 直呼び、本番側も bit-identical を保証)。
            columns[f"__num_srv_close_{_w}"] = stable_rolling_var(close_arr, _w, 1)

        # --- 複合式パターン ---
        # ▼ pct_close: pct_change(close) を numpy で先に計算 ▼
        # _pct_change は既に L237 で close_pct として計算済みなので再利用。
        # ▼ pct_close.rolling_std(20, ddof=1) → __num_srs_pct_close_20 ▼
        # 用途:
        #   - e1b_volatility_20 (L334 の active expr)
        #   - e1b_t_dist_scale_50 の分母 (L474 の active expr)
        # CSE で同じ列を共有するので 2 箇所で再利用可能。
        columns["__num_srs_pct_close_20"] = stable_rolling_std(
            close_pct.astype(np.float64), 20, 1
        )

        # ▼ volume が存在する場合の volume 系複合式 ▼
        if len(volume_arr) > 0:
            # rel_volume = volume / (rolling_mean(volume, lookback_bars) + 1e-10)
            srm_volume_lookback = stable_rolling_mean(volume_arr, lookback_bars)
            columns["__num_srm_volume_lookback"] = srm_volume_lookback
            rel_volume_arr = volume_arr / (srm_volume_lookback + 1e-10)
            # rel_volume.rolling_mean(20) → __num_srm_rel_volume_20
            columns["__num_srm_rel_volume_20"] = stable_rolling_mean(
                rel_volume_arr.astype(np.float64), 20
            )
            # (pct_close * rel_volume).rolling_mean(10) → __num_srm_vpt_10
            vpt_input = (close_pct * rel_volume_arr).astype(np.float64)
            columns["__num_srm_vpt_10"] = stable_rolling_mean(vpt_input, 10)
        # ▲▲ [§B.12.13.7 Option C2 / Phase E] ▲▲

        # ===== exprs =====
        exprs: List[pl.Expr] = []

        # ---------------------------------------------------------
        # basic_stats: 6 stats × 4 windows = 24 features
        # 参照: engine_1_B._create_basic_stats_features (L1006-1048)
        # ▼▼ [Phase E] rolling_{mean,var,std} → __num_* 列参照 ▼▼
        # ---------------------------------------------------------
        for window in FeatureModule1B.GENERAL_WINDOWS:
            exprs.append(
                ((pl.col(f"__num_srm_close_{window}") - pl.col("close"))
                 / pl.col("__temp_atr_safe"))
                .alias(f"e1b_rolling_mean_{window}")
            )
            exprs.append(
                (pl.col(f"__num_srs_close_{window}")
                 / pl.col("__temp_atr_safe"))
                .alias(f"e1b_rolling_std_{window}")
            )
            exprs.append(
                (pl.col(f"__num_srv_close_{window}")
                 / pl.col("__temp_atr_safe").pow(2))
                .alias(f"e1b_rolling_var_{window}")
            )
            exprs.append(
                ((pl.col("close").rolling_median(window) - pl.col("close"))
                 / pl.col("__temp_atr_safe"))
                .alias(f"e1b_rolling_median_{window}")
            )
            exprs.append(
                ((pl.col("close").rolling_min(window) - pl.col("close"))
                 / pl.col("__temp_atr_safe"))
                .alias(f"e1b_rolling_min_{window}")
            )
            exprs.append(
                ((pl.col("close").rolling_max(window) - pl.col("close"))
                 / pl.col("__temp_atr_safe"))
                .alias(f"e1b_rolling_max_{window}")
            )
        # ▲▲ [Phase E] ▲▲

        # ---------------------------------------------------------
        # composite: zscore + bollinger × 2 windows = 6 features
        # 参照: engine_1_B._create_composite_features (L1050-1110)
        # ▼▼ [Phase E] rolling_{mean,std} → __num_* 列参照 ▼▼
        # ---------------------------------------------------------
        for window in [20, 50]:
            mean_col = pl.col(f"__num_srm_close_{window}")
            std_col  = pl.col(f"__num_srs_close_{window}")
            exprs.append(
                ((pl.col("close") - mean_col) / (std_col + 1e-10))
                .alias(f"e1b_zscore_{window}")
            )
            exprs.append(
                (((mean_col + 2 * std_col) - pl.col("close"))
                 / pl.col("__temp_atr_safe"))
                .alias(f"e1b_bollinger_upper_{window}")
            )
            exprs.append(
                (((mean_col - 2 * std_col) - pl.col("close"))
                 / pl.col("__temp_atr_safe"))
                .alias(f"e1b_bollinger_lower_{window}")
            )
        # ▲▲ [Phase E] ▲▲

        # ---------------------------------------------------------
        # composite: 単純な追加特徴量
        # ---------------------------------------------------------
        # price_change: pct_change の最終値
        exprs.append(pl.col("close").pct_change().alias("e1b_price_change"))

        # ▼▼ [Phase E] volatility_20: pct_change.rolling_std → __num_srs_pct_close_20 列参照
        # 参照: engine_1_B._create_composite_features の volatility_20 (L1149)
        exprs.append(
            pl.col("__num_srs_pct_close_20").alias("e1b_volatility_20")
        )
        # ▲▲ [Phase E] ▲▲

        # price_range: (high - low) / atr_safe (high/low が存在する場合のみ)
        if len(high_arr) > 0 and len(low_arr) > 0:
            exprs.append(
                ((pl.col("high") - pl.col("low")) / pl.col("__temp_atr_safe"))
                .alias("e1b_price_range")
            )

        # ▼▼ [Phase E] volume_ma20 / volume_price_trend: __num_* 列参照 ▼▼
        # 参照: engine_1_B._create_composite_features (L1157-1163)
        if len(volume_arr) > 0:
            exprs.append(
                pl.col("__num_srm_rel_volume_20").alias("e1b_volume_ma20")
            )
            exprs.append(
                pl.col("__num_srm_vpt_10").alias("e1b_volume_price_trend")
            )
        # ▲▲ [Phase E] ▲▲

        # ===== layer2 =====
        layer2: Dict[str, float] = {}

        # ---------------------------------------------------------
        # high/low/volume が無い場合のフォールバック (Step B と同一)
        # ---------------------------------------------------------
        if len(high_arr) == 0 or len(low_arr) == 0:
            layer2["e1b_price_range"] = np.nan
        if len(volume_arr) == 0:
            layer2["e1b_volume_ma20"] = np.nan
            layer2["e1b_volume_price_trend"] = np.nan

        # ---------------------------------------------------------
        # timeseries: adf / pp / kpss × [50, 100] (pct_change に適用)
        # 参照: engine_1_B._create_timeseries_features (L1112-1151)
        # ---------------------------------------------------------
        for window in [50, 100]:
            if len(close_pct) >= window:
                pct_w = close_pct[-window:]
                layer2[f"e1b_adf_statistic_{window}"]  = float(adf_統計量_udf(pct_w))
                layer2[f"e1b_pp_statistic_{window}"]   = float(phillips_perron_統計量_udf(pct_w))
                layer2[f"e1b_kpss_statistic_{window}"] = float(kpss_統計量_udf(pct_w))
            else:
                layer2[f"e1b_adf_statistic_{window}"]  = np.nan
                layer2[f"e1b_pp_statistic_{window}"]   = np.nan
                layer2[f"e1b_kpss_statistic_{window}"] = np.nan

        # ---------------------------------------------------------
        # exp_arima: holt_level / holt_trend / arima_residual_var × [50, 100]
        # 参照: engine_1_B._create_exponential_arima_features (L1153-1199)
        # ---------------------------------------------------------
        for window in [50, 100]:
            if len(close_arr) < window:
                layer2[f"e1b_holt_level_{window}"]         = np.nan
                layer2[f"e1b_holt_trend_{window}"]         = np.nan
                layer2[f"e1b_arima_residual_var_{window}"] = np.nan
                continue

            w_close = close_arr[-window:]
            holt_level_raw = holt_winters_レベル_udf(w_close)
            holt_trend_raw = holt_winters_トレンド_udf(w_close)
            arima_raw      = arima_残差分散_udf(w_close)

            layer2[f"e1b_holt_level_{window}"] = (
                (float(holt_level_raw) - close_last) / atr_last
                if np.isfinite(holt_level_raw) and atr_ok else np.nan
            )
            layer2[f"e1b_holt_trend_{window}"] = (
                float(holt_trend_raw) / atr_last
                if np.isfinite(holt_trend_raw) and atr_ok else np.nan
            )
            layer2[f"e1b_arima_residual_var_{window}"] = (
                float(arima_raw) / atr2_last
                if np.isfinite(arima_raw) and atr_ok else np.nan
            )

        # ---------------------------------------------------------
        # kalman_regression: kalman_state / lowess_fitted / theil_sen_slope × [50, 100]
        # 参照: engine_1_B._create_kalman_regression_features (L1201-1249)
        # ---------------------------------------------------------
        for window in [50, 100]:
            if len(close_arr) < window:
                layer2[f"e1b_kalman_state_{window}"]    = np.nan
                layer2[f"e1b_lowess_fitted_{window}"]   = np.nan
                layer2[f"e1b_theil_sen_slope_{window}"] = np.nan
                continue

            w_close = close_arr[-window:]
            kalman_raw = kalman_状態推定_udf(w_close)
            lowess_raw = lowess_適合値_udf(w_close)
            theil_raw  = theil_sen_傾き_udf(w_close)

            layer2[f"e1b_kalman_state_{window}"] = (
                (float(kalman_raw) - close_last) / atr_last
                if np.isfinite(kalman_raw) and atr_ok else np.nan
            )
            layer2[f"e1b_lowess_fitted_{window}"] = (
                (float(lowess_raw) - close_last) / atr_last
                if np.isfinite(lowess_raw) and atr_ok else np.nan
            )
            layer2[f"e1b_theil_sen_slope_{window}"] = (
                float(theil_raw) / atr_last
                if np.isfinite(theil_raw) and atr_ok else np.nan
            )

        # ---------------------------------------------------------
        # distributions: t_dist_dof_50 / t_dist_scale_50 / gev_shape_50
        # 参照: engine_1_B._create_distributions_features (L1251-1289)
        #
        # 【Phase 9b 設計】
        #   学習側 t_dist_scale_50 は Polars 内で完結している:
        #     pct_change.rolling_map(t分布_尺度_udf, window_size=50, min_samples=50)
        #       / (pct_change.rolling_std(20, ddof=1) + 1e-10)
        #
        #   本番側でも Polars 統一を貫徹する:
        #     - 分子: Numba UDF を numpy で precompute (rolling_map 回避のため)
        #             → スカラー値を pl.lit で Polars に注入
        #     - 分母: Polars rolling_std(20, ddof=1) のまま (e1b_volatility_20 と
        #             同一サブグラフ → Polars クエリープランナーが CSE する)
        #     - 割り算: Polars 内で実行 (学習側と完全同一の計算経路)
        # ---------------------------------------------------------
        if len(close_pct) >= 50:
            pct_50 = close_pct[-50:]
            layer2["e1b_t_dist_dof_50"] = float(t分布_自由度_udf(pct_50))

            t_scale_raw = float(t分布_尺度_udf(pct_50))
        else:
            layer2["e1b_t_dist_dof_50"] = np.nan
            t_scale_raw = float("nan")

        # t_dist_scale_50 は Polars 式として exprs に追加 (学習側と完全同一の経路)
        # pl.lit(NaN) / 何か = NaN なので不足データの場合も自然に NaN 伝播
        # ▼▼ [Phase E] 分母 pct_change.rolling_std(20) → __num_srs_pct_close_20 列参照
        # (e1b_volatility_20 と同じ列を共有 — CSE 不要で明示的共有)
        exprs.append(
            (
                pl.lit(t_scale_raw, dtype=pl.Float64)
                / (pl.col("__num_srs_pct_close_20") + 1e-10)
            ).alias("e1b_t_dist_scale_50")
        )
        # ▲▲ [Phase E] ▲▲

        if len(high_arr) >= 50:
            layer2["e1b_gev_shape_50"] = float(gev_形状_udf(high_arr[-50:]))
        else:
            layer2["e1b_gev_shape_50"] = np.nan

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
        columns, exprs, layer2 = FeatureModule1B._build_polars_pieces(data, lookback_bars)
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
