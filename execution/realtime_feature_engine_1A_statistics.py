# realtime_feature_engine_1A_statistics.py
# Category 1A: 基本統計・分布検定系 (Basic Statistics & Distribution Tests)
#
# 【Step 3 変更記録】core_indicators.py への移行 (Single Source of Truth)
#   - mad_rolling_numba UDF定義を削除 → calculate_mad に置き換え
#   - ATR割り欠落を修正: statistical_variance / fast_rolling_std / robust_mad
#     に calculate_atr_wilder による ATR スケール化を追加
#   - fast_volume_mean_50 を Relative Volume 化（学習側と一致）
#   - 不要になった import numba / import math を削除

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))  # /workspace をパスに追加
import numpy as np
from typing import Dict

import blueprint as config
sys.path.append(str(config.CORE_DIR))
from core_indicators import (
    calculate_atr_wilder,  # Wilder平滑化ATR
    calculate_mad,         # ローリングMAD（mad_rolling_numbaを置き換え）
)


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


import numba as nb


@nb.njit(fastmath=False, cache=True)
def _rolling_moment(arr: np.ndarray, window: int, moment: int) -> float:
    """
    statistical_moment_numba と等価。各時点のZスコア^moment の rolling_mean。
    [PERF] njit化によりPythonループを排除。
    NaN-aware: スカラーループでNaNをスキップ（ブールインデックス不使用）。
    数値結果は旧実装(_rolling_moment_np)と同一。
    """
    n_needed = 2 * window - 1
    # _window相当: 末尾n_needed要素を取得
    start = len(arr) - n_needed if len(arr) >= n_needed else 0
    n_work = len(arr) - start
    if n_work < window:
        return np.nan

    z_moment_sum = 0.0
    count = 0
    for i in range(n_work - window, n_work):
        # サブウィンドウのNaN除外平均・標準偏差をスカラーループで計算
        sub_start = i - window + 1
        sub_n = 0
        sub_sum = 0.0
        for j in range(sub_start, i + 1):
            v = arr[start + j]
            if not np.isnan(v):
                sub_sum += v
                sub_n += 1
        if sub_n < 2:
            continue
        sub_mean = sub_sum / sub_n

        sub_sq = 0.0
        for j in range(sub_start, i + 1):
            v = arr[start + j]
            if not np.isnan(v):
                sub_sq += (v - sub_mean) ** 2
        sub_std = np.sqrt(sub_sq / (sub_n - 1))  # ddof=1

        val = arr[start + i]
        if not np.isnan(val) and sub_std > 1e-10:
            z = (val - sub_mean) / sub_std
            z_moment_sum += z ** moment
            count += 1

    if count == 0:
        return np.nan
    return z_moment_sum / count


class FeatureModule1A:
    # ------------------------------------------------------------------
    # 内部ヘルパーメソッド
    # ------------------------------------------------------------------

    pass  # _rolling_moment_np は下記モジュールレベル関数に移管

    # ------------------------------------------------------------------
    # メイン計算
    # ------------------------------------------------------------------

    @staticmethod
    def calculate_features(data: Dict[str, np.ndarray]) -> Dict[str, float]:
        features = {}

        # 安全な参照用変数
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

        # ATR13を一度だけ計算（全特徴量で共有）
        # high/low が揃っていてかつ atr_last が有限値の場合のみATR割りを適用。
        # calculate_atr_wilder はシード TR[0] から始まるため nan は返さないが、
        # high/low が欠落している場合や close が極端に短い場合を np.isfinite でガード。
        if len(high_arr) > 0 and len(low_arr) > 0:
            atr_arr = calculate_atr_wilder(high_arr, low_arr, close_arr, 13)
            atr_last_raw = float(atr_arr[-1]) if len(atr_arr) > 0 else np.nan
        else:
            atr_last_raw = np.nan

        atr_valid = np.isfinite(atr_last_raw) and atr_last_raw > 0.0
        atr_last_safe = atr_last_raw + 1e-10 if atr_valid else np.nan

        # ---------------------------------------------------------
        # 1. 統計的モーメント系 (Moments)
        # Polarsの rolling_var / rolling_std に完全準拠するため ddof=1 を明記
        # ---------------------------------------------------------
        w10_arr = _window(close_arr, 10)
        if len(w10_arr) >= 2:
            # ▼▼ Step 3修正: statistical_variance は ATR² で割る（学習側と一致）
            features["e1a_statistical_variance_10"] = float(np.var(w10_arr, ddof=1)) / (atr_last_safe ** 2)
            mean_w10 = float(np.mean(w10_arr))
            std_w10 = float(np.std(w10_arr, ddof=1))
            # statistical_cv は rolling_mean で割る（ATR割りなし・学習側と一致）
            features["e1a_statistical_cv_10"] = std_w10 / (mean_w10 + 1e-10)
        else:
            features["e1a_statistical_variance_10"] = np.nan
            features["e1a_statistical_cv_10"] = np.nan

        # 高次モーメント（Zスコアベースで無次元 → ATR割り不要）
        features["e1a_statistical_moment_5_20"] = _rolling_moment(close_arr, 20, 5)
        features["e1a_statistical_moment_6_20"] = _rolling_moment(close_arr, 20, 6)
        features["e1a_statistical_moment_7_20"] = _rolling_moment(close_arr, 20, 7)
        features["e1a_statistical_moment_7_50"] = _rolling_moment(close_arr, 50, 7)

        # ---------------------------------------------------------
        # 2. 高速ローリング統計 (Fast Rolling Stats)
        # ---------------------------------------------------------
        # ▼▼ Step 3修正: fast_rolling_std は ATR で割る（学習側と一致）
        w10_fast = _window(close_arr, 10)
        features["e1a_fast_rolling_std_10"] = (
            float(np.std(w10_fast, ddof=1)) / atr_last_safe if len(w10_fast) >= 2 else np.nan
        )

        w20_fast = _window(close_arr, 20)
        features["e1a_fast_rolling_std_20"] = (
            float(np.std(w20_fast, ddof=1)) / atr_last_safe if len(w20_fast) >= 2 else np.nan
        )

        w100_fast = _window(close_arr, 100)
        features["e1a_fast_rolling_std_100"] = (
            float(np.std(w100_fast, ddof=1)) / atr_last_safe if len(w100_fast) >= 2 else np.nan
        )

        # ▼▼ Step 3修正: Relative Volume化（学習側は rolling_mean(50) / rolling_mean(lookback) の比率）
        # リアルタイム側では lookback_bars が timeframe 依存のため volume_mean_200 で近似
        if len(volume_arr) > 0:
            vol_mean_50 = float(np.mean(_window(volume_arr, 50)))
            vol_mean_200 = float(np.mean(_window(volume_arr, 200)))
            features["e1a_fast_volume_mean_50"] = vol_mean_50 / (vol_mean_200 + 1e-10)
        else:
            features["e1a_fast_volume_mean_50"] = np.nan

        # ---------------------------------------------------------
        # 3. ロバスト統計 (Robust Statistics)
        # ---------------------------------------------------------
        # ▼▼ Step 3修正: calculate_mad に置き換え + ATR割りを追加（学習側と一致）
        features["e1a_robust_mad_20"] = _last(calculate_mad(close_arr, 20)) / atr_last_safe

        return features
