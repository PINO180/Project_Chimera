# realtime_feature_engine_1A_statistics.py
# Category 1A: 基本統計・分布検定系 (Basic Statistics & Distribution Tests)

import math
import numpy as np
import numba as nb
from typing import Dict

# ------------------------------------------------------------------------------
# Part 1: ロバスト統計系 UDF (残存必須分)
# ------------------------------------------------------------------------------


@nb.guvectorize(["void(float64[:], float64[:])"], "(n)->(n)", nopython=True, cache=True)
def mad_rolling_numba(arr, out):
    """ローリングMAD計算 (e1a_robust_mad_20 用)"""
    n = len(arr)
    window = 20

    for i in range(n):
        if i < window - 1:
            out[i] = np.nan
        else:
            # ウィンドウ内データ取得
            window_data = arr[max(0, i - window + 1) : i + 1]
            finite_data = window_data[~np.isnan(window_data)]

            if len(finite_data) < 3:
                out[i] = np.nan
            else:
                # 中央値計算
                median_val = np.median(finite_data)
                # 絶対偏差の中央値
                abs_deviations = np.abs(finite_data - median_val)
                out[i] = np.median(abs_deviations)


# ==================================================================
# メイン計算モジュール
# ==================================================================


def _window(arr: np.ndarray, window: int) -> np.ndarray:
    """配列の末尾から `window` 個の要素を取得"""
    if window <= 0:
        return np.array([], dtype=arr.dtype)
    if window > len(arr):
        return arr
    return arr[-window:]


def _last(arr: np.ndarray) -> float:
    """配列の最新値（末尾）を取得。配列が空の場合はNaNを返す"""
    if len(arr) == 0:
        return np.nan
    return float(arr[-1])


class FeatureModule1A:
    # ------------------------------------------------------------------
    # 内部ヘルパーメソッド
    # ------------------------------------------------------------------

    @staticmethod
    def _rolling_moment_np(arr: np.ndarray, window: int, moment: int) -> float:
        """
        oldスクリプトの statistical_moment_numba と等価。
        各時点の Zスコア^moment の rolling_mean。
        """
        n_needed = 2 * window - 1
        work_arr = _window(arr, n_needed)
        n_work = len(work_arr)
        if n_work < window:
            return np.nan
        z_moment_sum = 0.0
        count = 0
        for i in range(n_work - window, n_work):
            sub_w = work_arr[i - window + 1 : i + 1]
            sub_finite = sub_w[np.isfinite(sub_w)]
            if len(sub_finite) >= 2:
                sub_mean = np.mean(sub_finite)
                sub_std = np.std(sub_finite, ddof=1)
                val = work_arr[i]
                if np.isfinite(val) and sub_std > 1e-10:
                    z = (val - sub_mean) / sub_std
                    z_moment_sum += z**moment
                    count += 1
        if count == 0:
            return np.nan
        return float(z_moment_sum / count)

    # ------------------------------------------------------------------
    # メイン計算
    # ------------------------------------------------------------------

    @staticmethod
    def calculate_features(data: Dict[str, np.ndarray]) -> Dict[str, float]:
        features = {}

        # 安全な参照用変数
        close_arr = data["close"].astype(np.float64)
        volume_arr = (
            data["volume"].astype(np.float64)
            if len(data.get("volume", [])) > 0
            else np.array([], dtype=np.float64)
        )

        if len(close_arr) == 0:
            return features

        # ---------------------------------------------------------
        # 1. 統計的モーメント系 (Moments)
        # Polarsの rolling_var / rolling_std に完全準拠するため ddof=1 を明記
        # ---------------------------------------------------------
        # 分散
        features["e1a_statistical_variance_10"] = float(
            np.var(_window(close_arr, 10), ddof=1)
        )

        # 変動係数 (CV)
        mean_w10 = float(np.mean(_window(close_arr, 10)))
        std_w10 = float(np.std(_window(close_arr, 10), ddof=1))
        features["e1a_statistical_cv_10"] = std_w10 / (mean_w10 + 1e-10)

        # 高次モーメント
        _mom = FeatureModule1A._rolling_moment_np
        features["e1a_statistical_moment_5_20"] = _mom(close_arr, 20, 5)
        features["e1a_statistical_moment_6_20"] = _mom(close_arr, 20, 6)
        features["e1a_statistical_moment_7_20"] = _mom(close_arr, 20, 7)
        features["e1a_statistical_moment_7_50"] = _mom(close_arr, 50, 7)

        # ---------------------------------------------------------
        # 2. 高速ローリング統計 (Fast Rolling Stats)
        # ---------------------------------------------------------
        # ローリング標準偏差
        features["e1a_fast_rolling_std_10"] = float(
            np.std(_window(close_arr, 10), ddof=1)
        )
        features["e1a_fast_rolling_std_20"] = float(
            np.std(_window(close_arr, 20), ddof=1)
        )
        features["e1a_fast_rolling_std_100"] = float(
            np.std(_window(close_arr, 100), ddof=1)
        )

        # 出来高ローリング平均
        features["e1a_fast_volume_mean_50"] = (
            float(np.mean(_window(volume_arr, 50))) if len(volume_arr) > 0 else np.nan
        )

        # ---------------------------------------------------------
        # 3. ロバスト統計 (Robust Statistics)
        # ---------------------------------------------------------
        # MAD: guvectorize UDF (配列全体渡し → _last())
        features["e1a_robust_mad_20"] = _last(mad_rolling_numba(close_arr))

        return features
