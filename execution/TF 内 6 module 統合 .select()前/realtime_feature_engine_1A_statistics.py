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


def _pct_change(arr: np.ndarray) -> np.ndarray:
    """Polars .pct_change() と完全一致: (x[i] - x[i-1]) / x[i-1]
    prev == 0 のとき inf (Polars 準拠)、先頭は nan。
    Group 5 (統計検定) の Numba UDF への入力として使用。
    """
    n = len(arr)
    out = np.full(n, np.nan, dtype=np.float64)
    if n < 2:
        return out
    with np.errstate(divide="ignore", invalid="ignore"):
        out[1:] = (arr[1:] - arr[:-1]) / arr[:-1]
    return out


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

    def __init__(self, lookback_bars: int = 1440):
        self.alpha: float = 1.0 - np.exp(-np.log(2.0) / max(lookback_bars, 1))
        self._ewm_mean: Dict[str, float] = {}
        self._ewm_var: Dict[str, float] = {}
        self._ewm_n: Dict[str, int] = {}

    def update_and_clip(self, key: str, raw_val: float) -> float:
        """1特徴量の raw_val に対して QA処理を適用し、処理済みスカラーを返す。"""
        alpha = self.alpha

        # 学習側 col.replace([inf,-inf], None) 相当
        is_pos_inf = np.isposinf(raw_val)
        is_neg_inf = np.isneginf(raw_val)
        ewm_input = np.nan if not np.isfinite(raw_val) else raw_val

        # EWM 状態更新
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

class FeatureModule1A:

    # 外部から FeatureModule1A.QAState としてアクセス可能にする
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
            Layer 1 (rolling 統計, Group 1/2/3/6):
                Polars 式リストを構築 → 単一 .lazy().select(exprs).tail(1).collect()
                で一括計算。学習側 engine_1_A の Polars 式と完全一致。
            Layer 2 (Numba UDF, Group 4/5/7/8):
                core_indicators の SSoT 関数を直接呼び、最終バーの値を取得。
                学習側も同じ関数を呼ぶため数値完全一致。

        Args:
            data         : close/high/low/volume の numpy 配列を含む辞書
            lookback_bars: タイムフレームに応じた1日あたりのバー数。
                           fast_volume_mean_* の分母および QA の EWM 半減期に使用。
            qa_state     : QAState インスタンス。本番稼働時は必ず渡し、同一
                           インスタンスを毎バー使い回すこと。
        """
        features: Dict[str, float] = {}

        # ---------------------------------------------------------
        # 入力配列の準備
        # ---------------------------------------------------------
        close_arr = data["close"].astype(np.float64)
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

        if len(close_arr) == 0:
            return features

        # ---------------------------------------------------------
        # ATR13 計算 (学習側 __temp_atr_13 と完全一致)
        #
        # 学習側 (engine_1_A.calculate_one_group L660-675):
        #   __temp_atr_13 = calculate_atr_wilder(high, low, close, 13) + 1e-10
        #
        # 旧本番側のコメントは "学習側と完全同一" と記載されていたが誤りで、
        # 実際には + 1e-10 が抜けて 1e-11 オーダーで学習側と乖離していた。
        # Step B で完全一致するよう修正。
        # ---------------------------------------------------------
        if len(high_arr) > 0 and len(low_arr) > 0:
            atr_arr_raw = calculate_atr_wilder(high_arr, low_arr, close_arr, 13)
            atr_arr = atr_arr_raw + 1e-10  # 学習側 __temp_atr_13 と完全一致
        else:
            atr_arr = np.full(len(close_arr), np.nan, dtype=np.float64)

        atr_last = float(atr_arr[-1]) if np.isfinite(atr_arr[-1]) else np.nan
        close_last = float(close_arr[-1])

        # ---------------------------------------------------------
        # Polars DataFrame 構築 (Zero-copy)
        # ---------------------------------------------------------
        df_dict = {
            "close": close_arr,
            "__temp_atr_13": atr_arr,
        }
        if len(volume_arr) > 0:
            df_dict["volume"] = volume_arr
        df = pl.DataFrame(df_dict)

        # =====================================================================
        # Polars 式リスト構築 (Group 1, 2, 3, 6)
        #
        # 学習側 engine_1_A の以下の関数群と Polars 式が完全一致:
        #   _create_basic_stats_features    (Group 1)
        #   _create_skew_kurt_features      (Group 2)
        #   _create_robust_stats_features   (Group 3 - trimmed_mean を除く)
        #   _create_fast_processing_features (Group 6)
        # =====================================================================
        exprs = []

        # ---------------------------------------------------------
        # Group 1: 統計的モーメント (mean/var/std/cv) × window=[10,20,50]
        # 参照: engine_1_A._create_basic_stats_features (L1037-1071)
        # ---------------------------------------------------------
        for window in [10, 20, 50]:
            exprs.append(
                ((pl.col("close") - pl.col("close").rolling_mean(window))
                 / pl.col("__temp_atr_13"))
                .alias(f"e1a_statistical_mean_{window}")
            )
            exprs.append(
                (pl.col("close").rolling_var(window, ddof=1)
                 / pl.col("__temp_atr_13").pow(2))
                .alias(f"e1a_statistical_variance_{window}")
            )
            exprs.append(
                (pl.col("close").rolling_std(window, ddof=1)
                 / pl.col("__temp_atr_13"))
                .alias(f"e1a_statistical_std_{window}")
            )
            exprs.append(
                (pl.col("close").rolling_std(window, ddof=1)
                 / (pl.col("close").rolling_mean(window) + 1e-10))
                .alias(f"e1a_statistical_cv_{window}")
            )

        # ---------------------------------------------------------
        # Group 2: 歪度・尖度・高次モーメント × window=[20,50]
        # 参照: engine_1_A._create_skew_kurt_features (L1073-1113)
        #
        # 旧実装は O(window²) の二重ループで近似していたが、Polars の
        # rolling_skew / rolling_var / rolling_mean を直接呼ぶことで
        # 学習側と数式・実装ともに完全一致する。
        # ---------------------------------------------------------
        for window in [20, 50]:
            # skewness: Polars rolling_skew (デフォルト bias=True、Fisher 補正なし)
            exprs.append(
                pl.col("close").rolling_skew(window_size=window)
                .alias(f"e1a_statistical_skewness_{window}")
            )

            # kurtosis: 4次中心モーメント / std_ddof0^4 - 3
            #   var_ddof0 = rolling_var(window, ddof=1) * (window-1)/window
            #   分子      = ((close - rolling_mean)^4).rolling_mean(window)
            #   分母      = (var_ddof0)^2 + 1e-10
            var_ddof0 = pl.col("close").rolling_var(window, ddof=1) * ((window - 1.0) / window)
            std_ddof0_pow4 = var_ddof0.pow(2)
            exprs.append(
                ((pl.col("close") - pl.col("close").rolling_mean(window))
                 .pow(4)
                 .rolling_mean(window)
                 / (std_ddof0_pow4 + 1e-10)
                 - 3)
                .alias(f"e1a_statistical_kurtosis_{window}")
            )

            # 高次モーメント (5,6,7,8): ((close - mean) / std_ddof0)^moment の rolling_mean
            mean_col = pl.col("close").rolling_mean(window)
            std_ddof0 = (var_ddof0 + 1e-10).sqrt()
            for moment in [5, 6, 7, 8]:
                exprs.append(
                    (((pl.col("close") - mean_col) / std_ddof0)
                     .pow(moment)
                     .rolling_mean(window))
                    .alias(f"e1a_statistical_moment_{moment}_{window}")
                )

        # ---------------------------------------------------------
        # Group 3: ロバスト統計 (median/q25/q75/iqr) × window=[10,20,50]
        # 参照: engine_1_A._create_robust_stats_features (L1115-1157)
        #
        # 注意: trimmed_mean は学習側で rolling_map(scipy.trim_mean) を使うため
        #       超低速。本番側では最終バーのみ Numba _trim_mean で計算する
        #       (結果は scipy と数値的に同一)。後段の処理で features 辞書に追加。
        # ---------------------------------------------------------
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

        # ---------------------------------------------------------
        # Group 6: 高速ローリング統計 × window=[5,10,20,50,100]
        # 参照: engine_1_A._create_fast_processing_features (L1248-1278)
        # ---------------------------------------------------------
        for window in [5, 10, 20, 50, 100]:
            exprs.append(
                ((pl.col("close") - pl.col("close").rolling_mean(window))
                 / pl.col("__temp_atr_13"))
                .alias(f"e1a_fast_rolling_mean_{window}")
            )
            exprs.append(
                (pl.col("close").rolling_std(window, ddof=1)
                 / pl.col("__temp_atr_13"))
                .alias(f"e1a_fast_rolling_std_{window}")
            )
            if len(volume_arr) > 0:
                exprs.append(
                    (pl.col("volume").rolling_mean(window)
                     / (pl.col("volume").rolling_mean(lookback_bars) + 1e-10))
                    .alias(f"e1a_fast_volume_mean_{window}")
                )

        # =====================================================================
        # Polars 単一 .select() で全式を一括計算 (FFI overhead 1回のみ)
        #
        # df.lazy() : LazyFrame 化してクエリオプティマイザを起動
        # .select() : 全 expression を一括で受け取り、共通部分式を最適化
        # .tail(1)  : 最終 1 行のみ計算 (最終バーの特徴量のみ必要)
        # .collect(): Eager 実行 (Rust マルチスレッド/SIMD)
        # =====================================================================
        result_df = df.lazy().select(exprs).tail(1).collect()
        polars_result = result_df.to_dicts()[0]

        # Polars null → np.nan に変換 (to_dicts は null を Python None として返す)
        for k, v in polars_result.items():
            features[k] = float(v) if v is not None else np.nan

        # ---------------------------------------------------------
        # Group 3: trimmed_mean (最終バーのみ Numba で計算)
        # 学習側 rolling_map(scipy.trim_mean) と数値的に同一。
        # ---------------------------------------------------------
        for window in [10, 20, 50]:
            if len(close_arr) >= window:
                trim_w = _trim_mean(close_arr[-window:], 0.1)
                features[f"e1a_robust_trimmed_mean_{window}"] = (
                    (close_last - trim_w) / atr_last if np.isfinite(trim_w) else np.nan
                )
            else:
                features[f"e1a_robust_trimmed_mean_{window}"] = np.nan

        # ---------------------------------------------------------
        # Volume が存在しない場合の fast_volume_mean フォールバック
        # ---------------------------------------------------------
        if len(volume_arr) == 0:
            for window in [5, 10, 20, 50, 100]:
                features[f"e1a_fast_volume_mean_{window}"] = np.nan

        # =====================================================================
        # Group 4: 高度ロバスト統計 (Numba UDF / SSoT)
        # 学習側と本番側で完全同一の core_indicators 関数を呼ぶ。
        # 参照: engine_1_A._create_advanced_robust_features (L1159-1203)
        # =====================================================================
        # MAD: 配列を返すので最終値を取得
        mad_arr = calculate_mad(close_arr, 20)
        features["e1a_robust_mad_20"] = (
            float(mad_arr[-1]) / atr_last
            if len(mad_arr) > 0 and np.isfinite(mad_arr[-1])
            else np.nan
        )

        biweight_arr = biweight_location_numba(close_arr)
        features["e1a_robust_biweight_location_20"] = (
            (close_last - float(biweight_arr[-1])) / atr_last
            if np.isfinite(biweight_arr[-1]) else np.nan
        )

        winsorized_arr = winsorized_mean_numba(close_arr)
        features["e1a_robust_winsorized_mean_20"] = (
            (close_last - float(winsorized_arr[-1])) / atr_last
            if np.isfinite(winsorized_arr[-1]) else np.nan
        )

        # =====================================================================
        # Group 5: 統計検定 (Numba UDF / SSoT)
        # pct_change を計算してから検定関数を呼ぶ。
        # 参照: engine_1_A._create_statistical_tests_features (L1205-1245)
        # =====================================================================
        pct_arr = _pct_change(close_arr)

        jb_arr = jarque_bera_statistic_numba(pct_arr)
        features["e1a_jarque_bera_statistic_50"] = float(jb_arr[-1])

        ad_arr = anderson_darling_numba(pct_arr)
        features["e1a_anderson_darling_statistic_30"] = float(ad_arr[-1])

        runs_arr = runs_test_numba(pct_arr)
        features["e1a_runs_test_statistic_30"] = float(runs_arr[-1])

        vn_arr = von_neumann_ratio_numba(pct_arr)
        features["e1a_von_neumann_ratio_30"] = float(vn_arr[-1])

        # =====================================================================
        # Group 7: Numba 最適化 (SSoT)
        # 参照: engine_1_A._create_numba_features (L1280-1298)
        # =====================================================================
        qs_arr = fast_quality_score_numba(close_arr)
        features["e1a_fast_quality_score_50"] = float(qs_arr[-1])

        # =====================================================================
        # Group 8: 品質保証 (Numba UDF / SSoT)
        # 参照: engine_1_A._create_qa_features (L1300-1331)
        # =====================================================================
        bs_arr = basic_stabilization_numba(close_arr)
        features["e1a_fast_basic_stabilization"] = (
            (close_last - float(bs_arr[-1])) / atr_last
            if np.isfinite(bs_arr[-1]) else np.nan
        )

        rs_arr = robust_stabilization_numba(close_arr)
        features["e1a_fast_robust_stabilization"] = (
            (close_last - float(rs_arr[-1])) / atr_last
            if np.isfinite(rs_arr[-1]) else np.nan
        )

        # =====================================================================
        # QA 処理 (学習側 apply_quality_assurance_to_group と等価)
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
