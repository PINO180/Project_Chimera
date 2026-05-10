"""
realtime_feature_engine_1B_timeseries.py
Project Cimera V5 - 特徴量エンジンモジュール 【1B: 時系列・分布系】

==================================================================
【Step B 改修】numpy ベース → Polars 直呼び に全面移行
==================================================================

目的: 学習側 engine_1_B_a_vast_universe_of_features.py と「同じ Polars
      Rust エンジン」で計算することで、ビット完全一致を達成する。

【背景】
  旧実装(numpy):
    - 各 rolling 統計を毎バー O(N) で再計算 (rolling_mean/std/var/median/min/max)
    - volume_ma20 / volume_price_trend で per-bar rel_volume を Python ループで生成
    - 学習側 Polars と数値が機械イプシロン差で乖離する可能性
  新実装(Polars 直呼び):
    - basic_stats / composite 系を全て Polars 式リストに集約 (35 特徴量)
    - 単一 .lazy().select(exprs).tail(1).collect() で一括計算
    - Numba UDF (adf/pp/kpss/holt/arima/kalman/lowess/theil/t_dist/gev) は
      core_indicators の SSoT を直接呼び、最終バーの値のみ計算 (21 特徴量)
    - 学習側 rolling_map と数値完全一致 (同じ UDF を同じ入力で呼ぶため)

【ATR (__temp_atr_safe) の扱い】
  学習側 (engine_1_B.inject_temp_atr L453-468):
    __temp_atr_safe = calculate_atr_wilder(...) + 1e-10
  本番側 (旧実装でも + 1e-10 が正しく適用されていた):
    atr_denom = atr_latest + 1e-10
  → 本番側 1B は 1A と異なり ATR 修正不要 (元から学習側と一致)。

【SSoT 階層】
  Layer 1 (rolling 統計): 学習・本番ともに Polars Rust エンジン → ビット一致
  Layer 2 (Numba UDF):    学習・本番ともに core_indicators から import → ビット一致
                           (Phase 5 で確立済み、変更なし)

【保持される過去の修正】
  ・Step 5: t分布 ddof 修正、ATR割り追加、_pct_change ゼロ除算保護
  ・Step 6: 全特徴量実装 (rolling_*, zscore, bollinger, etc.)
  ・Step 8 修正1: volatility_20 で window 内に NaN/inf あれば NaN
  ・Step 8 修正2: volume_price_trend は inf を mean 通過 (QA でクリップ)
  ・Step 8 修正3: window 未満データへの gate (rolling_map 互換)
  ・Step 8 修正4: QAState bias 補正 (1A と同一実装)
==================================================================
"""

import sys
import os
from pathlib import Path

import numpy as np
import polars as pl
from typing import Dict, Optional
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

logger = logging.getLogger("ProjectForge.FeatureEngine.1B")


# ==================================================================
# ヘルパー関数 (Step B 後の最小セット)
#
# 【削除済み (Step B)】
#   _window           : Polars が窓スライスを内部処理、Numba UDF には arr[-window:] で十分
#   _last             : arr[-1] で十分
#   _rolling_map      : Polars rolling_map を使わなくなったため不要
# ==================================================================

def _pct_change(arr: np.ndarray) -> np.ndarray:
    """Polars pct_change() と完全一致: (x[i] - x[i-1]) / x[i-1]
    prev == 0 のとき inf (Polars 準拠)、先頭は nan。
    Group 3/6 の Numba UDF 入力として使用。
    """
    if len(arr) < 2:
        return np.full_like(arr, np.nan, dtype=np.float64)
    pct = np.full(len(arr), np.nan, dtype=np.float64)
    with np.errstate(divide="ignore", invalid="ignore"):
        pct[1:] = (arr[1:] - arr[:-1]) / arr[:-1]
    return pct


# ==================================================================
# QAState — 学習側 apply_quality_assurance_to_group の等価実装
# (1A と同一実装。Step B では変更なし。)
# ==================================================================

class QAState:
    """
    学習側 apply_quality_assurance_to_group のリアルタイム等価実装。

    詳細は realtime_feature_engine_1A_statistics.py の QAState ドキュメントを参照。
    本クラスは 1A と完全に同一の実装。
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

class FeatureModule1B:
    # window_sizes["general"] = [10, 20, 50, 100]（学習側と同一）
    GENERAL_WINDOWS = [10, 20, 50, 100]

    # 外部から FeatureModule1B.QAState としてアクセス可能にする
    QAState = QAState

    @staticmethod
    def calculate_features(
        data: Dict[str, np.ndarray],
        lookback_bars: int = 1440,
        qa_state: Optional[QAState] = None,
    ) -> Dict[str, float]:
        """
        【Step B改修版】Polars 直呼びで全特徴量を計算する。

        アーキテクチャ:
            Layer 1 (rolling 統計, basic_stats / composite):
                Polars 式リストを構築 → 単一 .lazy().select(exprs).tail(1).collect()
                で 35 特徴量を一括計算。学習側 engine_1_B の Polars 式と完全一致。
            Layer 2 (Numba UDF, timeseries / exp_arima / kalman / distributions):
                core_indicators の SSoT 関数を直接呼び、最終バーの値を取得。
                21 特徴量。学習側も同じ UDF を同じ入力で呼ぶため数値完全一致。

        Args:
            data         : close/high/low/volume の numpy 配列を含む辞書
            lookback_bars: タイムフレームに応じた1日あたりのバー数
            qa_state     : QAState インスタンス
        """
        features: Dict[str, float] = {}

        # ---------------------------------------------------------
        # 入力配列の準備 (contiguous float64 化)
        # ---------------------------------------------------------
        close_arr = np.asarray(data["close"], dtype=np.float64)
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

        if len(close_arr) == 0:
            return features

        n = len(close_arr)
        close_last = float(close_arr[-1])

        # pct_change を一度だけ計算 (Group 3, 6 で再利用)
        close_pct = _pct_change(close_arr)

        # ---------------------------------------------------------
        # ATR13 計算 (学習側 inject_temp_atr と完全一致)
        #   __temp_atr_safe = calculate_atr_wilder(high, low, close, 13) + 1e-10
        # ---------------------------------------------------------
        if len(high_arr) > 0 and len(low_arr) > 0:
            atr_arr_raw = calculate_atr_wilder(high_arr, low_arr, close_arr, 13)
            atr_safe_arr = atr_arr_raw + 1e-10  # 学習側と完全一致
        else:
            atr_safe_arr = np.full(n, np.nan, dtype=np.float64)

        atr_last = float(atr_safe_arr[-1]) if np.isfinite(atr_safe_arr[-1]) else np.nan

        # ---------------------------------------------------------
        # Polars DataFrame 構築 (Zero-copy)
        # ---------------------------------------------------------
        df_dict = {
            "close": close_arr,
            "__temp_atr_safe": atr_safe_arr,
        }
        if len(high_arr) > 0:
            df_dict["high"] = high_arr
        if len(low_arr) > 0:
            df_dict["low"] = low_arr
        if len(volume_arr) > 0:
            df_dict["volume"] = volume_arr
        df = pl.DataFrame(df_dict)

        # =====================================================================
        # Polars 式リスト構築 (basic_stats + composite)
        # =====================================================================
        exprs = []

        # ---------------------------------------------------------
        # basic_stats: 6 stats × 4 windows = 24 features
        # 参照: engine_1_B._create_basic_stats_features (L1006-1048)
        #
        # 注意: 学習側は (rolling_X - close) / atr_safe の方向 (1A と逆)
        # ---------------------------------------------------------
        for window in FeatureModule1B.GENERAL_WINDOWS:
            exprs.append(
                ((pl.col("close").rolling_mean(window) - pl.col("close"))
                 / pl.col("__temp_atr_safe"))
                .alias(f"e1b_rolling_mean_{window}")
            )
            exprs.append(
                (pl.col("close").rolling_std(window, ddof=1)
                 / pl.col("__temp_atr_safe"))
                .alias(f"e1b_rolling_std_{window}")
            )
            exprs.append(
                (pl.col("close").rolling_var(window, ddof=1)
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

        # ---------------------------------------------------------
        # composite: zscore + bollinger × 2 windows = 6 features
        # 参照: engine_1_B._create_composite_features (L1050-1110)
        # ---------------------------------------------------------
        for window in [20, 50]:
            mean_col = pl.col("close").rolling_mean(window)
            std_col  = pl.col("close").rolling_std(window, ddof=1)
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

        # ---------------------------------------------------------
        # composite: 単純な追加特徴量 = 5 features
        # ---------------------------------------------------------
        # price_change: pct_change の最終値
        exprs.append(pl.col("close").pct_change().alias("e1b_price_change"))

        # volatility_20: pct_change.rolling_std(20, ddof=1)
        exprs.append(
            pl.col("close").pct_change().rolling_std(20, ddof=1)
            .alias("e1b_volatility_20")
        )

        # price_range: (high - low) / atr_safe (high/low が存在する場合のみ)
        if len(high_arr) > 0 and len(low_arr) > 0:
            exprs.append(
                ((pl.col("high") - pl.col("low")) / pl.col("__temp_atr_safe"))
                .alias("e1b_price_range")
            )

        # volume_ma20 / volume_price_trend: rel_volume ベース (volume が存在する場合のみ)
        # 参照: engine_1_B._create_composite_features (L1101-1108)
        #   rel_volume = volume / (volume.rolling_mean(lookback_bars) + 1e-10)
        #   volume_ma20 = rel_volume.rolling_mean(20)
        #   volume_price_trend = (close.pct_change() * rel_volume).rolling_mean(10)
        if len(volume_arr) > 0:
            rel_volume = pl.col("volume") / (
                pl.col("volume").rolling_mean(lookback_bars) + 1e-10
            )
            exprs.append(
                rel_volume.rolling_mean(20).alias("e1b_volume_ma20")
            )
            exprs.append(
                (pl.col("close").pct_change() * rel_volume).rolling_mean(10)
                .alias("e1b_volume_price_trend")
            )

        # =====================================================================
        # Polars 単一 .select() で全式を一括計算
        # =====================================================================
        result_df = df.lazy().select(exprs).tail(1).collect()
        polars_result = result_df.to_dicts()[0]

        # Polars null → np.nan に変換
        for k, v in polars_result.items():
            features[k] = float(v) if v is not None else np.nan

        # ---------------------------------------------------------
        # high/low/volume が無い場合の特徴量フォールバック
        # ---------------------------------------------------------
        if len(high_arr) == 0 or len(low_arr) == 0:
            features["e1b_price_range"] = np.nan
        if len(volume_arr) == 0:
            features["e1b_volume_ma20"] = np.nan
            features["e1b_volume_price_trend"] = np.nan

        # =====================================================================
        # Layer 2: Numba UDF 直呼び (最終バーのみ)
        # 学習側は rolling_map で全バー計算するが、本番側は最終バーのみ必要なため
        # 直接 UDF を呼ぶ。同じ入力 (arr[-window:]) で同じ UDF を呼ぶため数値同一。
        # =====================================================================

        # ---------------------------------------------------------
        # timeseries: adf / pp / kpss × [50, 100] (pct_change に適用)
        # 参照: engine_1_B._create_timeseries_features (L1112-1151)
        # ---------------------------------------------------------
        for window in [50, 100]:
            if len(close_pct) >= window:
                pct_w = close_pct[-window:]
                features[f"e1b_adf_statistic_{window}"]  = float(adf_統計量_udf(pct_w))
                features[f"e1b_pp_statistic_{window}"]   = float(phillips_perron_統計量_udf(pct_w))
                features[f"e1b_kpss_statistic_{window}"] = float(kpss_統計量_udf(pct_w))
            else:
                features[f"e1b_adf_statistic_{window}"]  = np.nan
                features[f"e1b_pp_statistic_{window}"]   = np.nan
                features[f"e1b_kpss_statistic_{window}"] = np.nan

        # ---------------------------------------------------------
        # exp_arima: holt_level / holt_trend / arima_residual_var × [50, 100]
        # 参照: engine_1_B._create_exponential_arima_features (L1153-1199)
        #   holt_level: (rolling_map(holt_level_udf) - close) / atr_safe
        #   holt_trend: rolling_map(holt_trend_udf) / atr_safe
        #   arima_residual_var: rolling_map(arima_udf) / atr_safe^2
        # ---------------------------------------------------------
        atr_ok = np.isfinite(atr_last)
        atr2_last = atr_last ** 2 if atr_ok else np.nan

        for window in [50, 100]:
            if len(close_arr) < window:
                features[f"e1b_holt_level_{window}"]         = np.nan
                features[f"e1b_holt_trend_{window}"]         = np.nan
                features[f"e1b_arima_residual_var_{window}"] = np.nan
                continue

            w_close = close_arr[-window:]
            holt_level_raw = holt_winters_レベル_udf(w_close)
            holt_trend_raw = holt_winters_トレンド_udf(w_close)
            arima_raw      = arima_残差分散_udf(w_close)

            features[f"e1b_holt_level_{window}"] = (
                (float(holt_level_raw) - close_last) / atr_last
                if np.isfinite(holt_level_raw) and atr_ok else np.nan
            )
            features[f"e1b_holt_trend_{window}"] = (
                float(holt_trend_raw) / atr_last
                if np.isfinite(holt_trend_raw) and atr_ok else np.nan
            )
            features[f"e1b_arima_residual_var_{window}"] = (
                float(arima_raw) / atr2_last
                if np.isfinite(arima_raw) and atr_ok else np.nan
            )

        # ---------------------------------------------------------
        # kalman_regression: kalman_state / lowess_fitted / theil_sen_slope × [50, 100]
        # 参照: engine_1_B._create_kalman_regression_features (L1201-1249)
        # ---------------------------------------------------------
        for window in [50, 100]:
            if len(close_arr) < window:
                features[f"e1b_kalman_state_{window}"]    = np.nan
                features[f"e1b_lowess_fitted_{window}"]   = np.nan
                features[f"e1b_theil_sen_slope_{window}"] = np.nan
                continue

            w_close = close_arr[-window:]
            kalman_raw = kalman_状態推定_udf(w_close)
            lowess_raw = lowess_適合値_udf(w_close)
            theil_raw  = theil_sen_傾き_udf(w_close)

            features[f"e1b_kalman_state_{window}"] = (
                (float(kalman_raw) - close_last) / atr_last
                if np.isfinite(kalman_raw) and atr_ok else np.nan
            )
            features[f"e1b_lowess_fitted_{window}"] = (
                (float(lowess_raw) - close_last) / atr_last
                if np.isfinite(lowess_raw) and atr_ok else np.nan
            )
            features[f"e1b_theil_sen_slope_{window}"] = (
                float(theil_raw) / atr_last
                if np.isfinite(theil_raw) and atr_ok else np.nan
            )

        # ---------------------------------------------------------
        # distributions: t_dist_dof_50 / t_dist_scale_50 / gev_shape_50
        # 参照: engine_1_B._create_distributions_features (L1251-1289)
        #
        #   t_dist_dof_50: rolling_map(t分布_自由度_udf, pct_change, window=50)
        #   t_dist_scale_50: rolling_map(t分布_尺度_udf, pct_change, window=50)
        #                    / (pct_change.rolling_std(20, ddof=1) + 1e-10)
        #   gev_shape_50: rolling_map(gev_形状_udf, high, window=50)
        #
        # 注意: t_dist_scale_50 の分母は volatility_20 と同じ式。
        #       Polars で先に計算済みのため features["e1b_volatility_20"] を再利用。
        # ---------------------------------------------------------
        if len(close_pct) >= 50:
            pct_50 = close_pct[-50:]
            features["e1b_t_dist_dof_50"] = float(t分布_自由度_udf(pct_50))

            t_scale_raw = t分布_尺度_udf(pct_50)
            pct_std_20 = features.get("e1b_volatility_20", np.nan)
            features["e1b_t_dist_scale_50"] = (
                float(t_scale_raw) / (pct_std_20 + 1e-10)
                if np.isfinite(t_scale_raw) and np.isfinite(pct_std_20) else np.nan
            )
        else:
            features["e1b_t_dist_dof_50"]   = np.nan
            features["e1b_t_dist_scale_50"] = np.nan

        if len(high_arr) >= 50:
            features["e1b_gev_shape_50"] = float(gev_形状_udf(high_arr[-50:]))
        else:
            features["e1b_gev_shape_50"] = np.nan

        # =====================================================================
        # QA 処理 (学習側 apply_quality_assurance_to_group と等価)
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
