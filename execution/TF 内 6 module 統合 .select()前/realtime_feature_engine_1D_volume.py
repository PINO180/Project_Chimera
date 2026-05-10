# realtime_feature_engine_1D_volume.py
# Category 1D: 出来高・ボラティリティ・プライスアクション系
# (Volume, Volatility & Price Action)
#
# ==================================================================
# 【Step B 改修】numpy ベース → Polars 直呼び に全面移行
# ==================================================================
#
# 目的: 学習側 engine_1_D_a_vast_universe_of_features.py と
#       「同じ Polars Rust エンジン」で計算することで、ビット完全一致を達成する。
#
# 【背景】
#   旧実装 (numpy):
#     - 全ローリング統計を numpy で毎回再計算 (O(window) × 多数)
#     - donchian / price_channel / pivot 等を手動ループ
#     - 学習側 Polars と数値が微小に乖離する可能性
#   新実装 (Polars 直呼び):
#     - 全 rolling 計算を 1 つの .select() に集約 → SIMD/Rayon 最適化
#     - Polars Rust エンジン直呼び → 学習側と同一エンジンでビット完全一致
#     - core_indicators の Numba UDF は SSoT として直接呼び出し
#     - 期待速度向上: 1029ms → 数十ms (40〜60倍)
#
# 【SSoT 階層】
#   Layer 1 (rolling 統計): 学習・本番ともに Polars Rust エンジン → ビット一致
#   Layer 2 (Numba UDF):    学習・本番ともに core_indicators から import → ビット一致
#                            (Phase 5 で確立済み、変更なし)
#
# 【保持される過去の修正】
#   ・QAState (apply_quality_assurance_to_group の等価実装)
#   ・ewm_std bias=False 補正の Polars 互換式
#   ・e1d_sample_weight は QA 対象外 (学習側 base_columns 扱いと一致)
# ==================================================================

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

import blueprint as config

# --- core_indicators: Single Source of Truth ---
sys.path.append(str(config.CORE_DIR))
from core_indicators import (
    calculate_atr_wilder,        # Wilder平滑化ATR (学習側と完全統一)
    calculate_sample_weight,     # Zスコアサンプルウェイト
    # [SSoT 統一] Engine 1D の UDF を core_indicators から import
    cmf_udf,
    mfi_udf,
    vwap_udf,
    obv_udf,
    accumulation_distribution_udf,
    chaikin_volatility_udf,
    mass_index_udf,
    hv_robust_udf,
    hv_standard_udf,
    commodity_channel_index_udf,
    fibonacci_levels_udf,
    candlestick_patterns_udf,
    force_index_udf,
)
# --------------------------------------------------------

import numpy as np
import polars as pl
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

    Layer 2 (Numba UDF) のうち hv_standard_udf / hv_robust_udf 等の
    入力に使用する。Polars rolling_map 経由で同じ pct_change を渡す
    学習側と数値完全一致。
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
                out[i] = np.nan  # 0 / 0
    return out


# ==================================================================
# QAState — 学習側 apply_quality_assurance_to_group の等価実装
# (Step B では変更なし - Layer 1/2 のロジックとは独立)
# ==================================================================

class QAState:
    """
    学習側 apply_quality_assurance_to_group のリアルタイム等価実装。

    学習側の処理 (Polars LazyFrame 全系列一括):
        col_expr = when(col.is_infinite()).then(None).otherwise(col)
        ewm_mean = col_expr.ewm_mean(half_life=HL, adjust=False, ignore_nulls=True)
        ewm_std  = col_expr.ewm_std (half_life=HL, adjust=False, ignore_nulls=True)
        result   = when(col==inf).then(upper).when(col==-inf).then(lower)
                   .otherwise(col).clip(lower, upper).fill_null(0.0).fill_nan(0.0)

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

class FeatureModule1D:

    # 外部から FeatureModule1D.QAState としてアクセス可能にする
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
            Layer 1 (rolling 統計, 各 group の Polars 式群):
                Polars 式リストを構築 → 単一 .lazy().select(exprs).tail(1).collect()
                で一括計算。学習側 engine_1_D の Polars 式と逐語一致。
            Layer 2 (Numba UDF, hv_standard / hv_robust / sample_weight):
                core_indicators の SSoT 関数を直接呼び、最終バーの値を取得。
                学習側も同じ関数を呼ぶため数値完全一致。

        Args:
            data         : close/high/low/open/volume の numpy 配列を含む辞書
            lookback_bars: タイムフレームに応じた1日あたりのバー数。
                           Volume の relative volume 分母および QA の EWM 半減期に使用。
            qa_state     : QAState インスタンス。本番稼働時は必ず渡し、同一
                           インスタンスを毎バー使い回すこと。
        """
        features: Dict[str, float] = {}

        # ① 必須キーガード: 必須キーが存在しない場合は空dictを返す
        for _key in ("close", "high", "low", "open", "volume"):
            if _key not in data:
                return features

        close_arr  = data["close"].astype(np.float64)
        high_arr   = data["high"].astype(np.float64)
        low_arr    = data["low"].astype(np.float64)
        open_arr   = data["open"].astype(np.float64)
        volume_arr = data["volume"].astype(np.float64)

        if len(close_arr) == 0:
            return features

        # ---------------------------------------------------------
        # ATR13 計算 (学習側 atr_13_internal_expr と完全一致)
        #
        # 学習側 (engine_1_D._create_*_features 各所):
        #   atr_13_internal_expr = pl.struct([...]).map_batches(calculate_atr_wilder(...))
        # 本番側: numpy で事前計算して __temp_atr_13 列として渡す。
        # 学習側は割り算時に + 1e-10 を加えるため、ここでは raw ATR を保持する。
        # ---------------------------------------------------------
        atr13_arr = calculate_atr_wilder(high_arr, low_arr, close_arr, 13)

        # ---------------------------------------------------------
        # Polars DataFrame 構築 (Zero-copy)
        # ---------------------------------------------------------
        df = pl.DataFrame({
            "close":  close_arr,
            "high":   high_arr,
            "low":    low_arr,
            "open":   open_arr,
            "volume": volume_arr,
            "__temp_atr_13": atr13_arr,  # 学習側 atr_13_internal_expr と数値同一
        })

        # =====================================================================
        # Polars 式リスト構築
        #
        # 学習側 engine_1_D の以下の関数群と Polars 式が完全一致:
        #   _create_volatility_features        (volatility group)
        #   _create_volume_features            (volume group)
        #   _create_breakout_features          (breakout group)
        #   _create_support_resistance_features (support_resistance group)
        #   _create_price_action_features      (price_action group)
        # =====================================================================
        exprs = []

        # =====================================================================
        # Volatility Group (Polars-only 部分)
        # 参照: engine_1_D._create_volatility_features (L1183-1309)
        #
        # 注: hv_standard / hv_robust / hv_robust_annual_252 は scalar UDF を
        #     rolling_map で呼ぶ形式のため Layer 2 で直接 UDF 呼び出し
        #     (本番側は最終バーのみ計算する)。
        # =====================================================================

        # hv_annual_252: 学習側 rolling_std(252, ddof=1) * sqrt(252) と完全一致
        exprs.append(
            (pl.col("close").pct_change().rolling_std(252, ddof=1) * np.sqrt(252))
            .alias("e1d_hv_annual_252")
        )

        # hv_regime_50: 学習側 Polars ネイティブのローリング分位数判定と完全一致
        # 参照: engine_1_D L1265-1273
        hv_50 = pl.col("close").pct_change().rolling_std(50, ddof=1)
        q80_roll = hv_50.rolling_quantile(0.8, window_size=1440)
        q60_roll = hv_50.rolling_quantile(0.6, window_size=1440)
        exprs.append(
            ((hv_50 > q80_roll).cast(pl.Int8) + (hv_50 > q60_roll).cast(pl.Int8))
            .fill_null(0)
            .alias("e1d_hv_regime_50")
        )

        # Chaikin Volatility (重量UDF, map_batches): window=[10, 20]
        # 参照: engine_1_D L1278-1291
        for window in [10, 20]:
            exprs.append(
                pl.struct(["high", "low"]).map_batches(
                    lambda s, w=window: chaikin_volatility_udf(
                        s.struct.field("high").to_numpy(),
                        s.struct.field("low").to_numpy(),
                        w,
                    ),
                    return_dtype=pl.Float64,
                ).alias(f"e1d_chaikin_volatility_{window}")
            )

        # Mass Index (重量UDF, map_batches): window=[20, 30]
        # 参照: engine_1_D L1294-1307
        for window in [20, 30]:
            exprs.append(
                pl.struct(["high", "low"]).map_batches(
                    lambda s, w=window: mass_index_udf(
                        s.struct.field("high").to_numpy(),
                        s.struct.field("low").to_numpy(),
                        w,
                    ),
                    return_dtype=pl.Float64,
                ).alias(f"e1d_mass_index_{window}")
            )

        # =====================================================================
        # Volume Group
        # 参照: engine_1_D._create_volume_features (L1311-1437)
        # =====================================================================

        # vol_ma1440 (with +1e-10 baked in - 学習側と完全一致):
        # 学習側: vol_ma1440 = pl.col("volume").rolling_mean(lookback_bars) + 1e-10
        vol_ma1440 = pl.col("volume").rolling_mean(lookback_bars) + 1e-10

        # CMF / MFI / VWAP距離: window=[13, 21, 34]
        # 参照: engine_1_D L1328-1378
        for window in [13, 21, 34]:
            # CMF (重量UDF)
            exprs.append(
                pl.struct(["high", "low", "close", "volume"]).map_batches(
                    lambda s, w=window: cmf_udf(
                        s.struct.field("high").to_numpy(),
                        s.struct.field("low").to_numpy(),
                        s.struct.field("close").to_numpy(),
                        s.struct.field("volume").to_numpy(),
                        w,
                    ),
                    return_dtype=pl.Float64,
                ).alias(f"e1d_cmf_{window}")
            )

            # MFI (重量UDF)
            exprs.append(
                pl.struct(["high", "low", "close", "volume"]).map_batches(
                    lambda s, w=window: mfi_udf(
                        s.struct.field("high").to_numpy(),
                        s.struct.field("low").to_numpy(),
                        s.struct.field("close").to_numpy(),
                        s.struct.field("volume").to_numpy(),
                        w,
                    ),
                    return_dtype=pl.Float64,
                ).alias(f"e1d_mfi_{window}")
            )

            # VWAP距離 (ATR割り)
            vwap_expr = pl.struct(["high", "low", "close", "volume"]).map_batches(
                lambda s, w=window: vwap_udf(
                    s.struct.field("high").to_numpy(),
                    s.struct.field("low").to_numpy(),
                    s.struct.field("close").to_numpy(),
                    s.struct.field("volume").to_numpy(),
                    w,
                ),
                return_dtype=pl.Float64,
            )
            exprs.append(
                ((pl.col("close") - vwap_expr) / (pl.col("__temp_atr_13") + 1e-10))
                .alias(f"e1d_vwap_dist_{window}")
            )

        # OBV relative: 学習側 obv_raw.diff() / vol_ma1440
        # 参照: engine_1_D L1384-1390
        obv_raw = pl.struct(["close", "volume"]).map_batches(
            lambda s: obv_udf(
                s.struct.field("close").to_numpy(),
                s.struct.field("volume").to_numpy(),
            ),
            return_dtype=pl.Float64,
        )
        exprs.append((obv_raw.diff() / vol_ma1440).alias("e1d_obv_rel"))

        # A/D Line relative: 学習側 ad_raw.diff() / vol_ma1440
        # 参照: engine_1_D L1393-1404
        ad_raw = pl.struct(["high", "low", "close", "volume"]).map_batches(
            lambda s: accumulation_distribution_udf(
                s.struct.field("high").to_numpy(),
                s.struct.field("low").to_numpy(),
                s.struct.field("close").to_numpy(),
                s.struct.field("volume").to_numpy(),
            ),
            return_dtype=pl.Float64,
        )
        exprs.append(
            (ad_raw.diff() / vol_ma1440).alias("e1d_accumulation_distribution_rel")
        )

        # Force Index normalized:
        #   学習側: force_raw / (atr_13_internal_expr * vol_ma1440 + 1e-10)
        #   ここで vol_ma1440 は既に +1e-10 込みのため、結果として
        #   force_raw / (atr * (rolling_mean(volume,1440) + 1e-10) + 1e-10) になる。
        # 参照: engine_1_D L1407-1417
        force_raw = pl.struct(["close", "volume"]).map_batches(
            lambda s: force_index_udf(
                s.struct.field("close").to_numpy(),
                s.struct.field("volume").to_numpy(),
            ),
            return_dtype=pl.Float64,
        )
        exprs.append(
            (force_raw / (pl.col("__temp_atr_13") * vol_ma1440 + 1e-10))
            .alias("e1d_force_index_norm")
        )

        # Volume MA20 relative: 学習側 rolling_mean(20) / vol_ma1440
        # 参照: engine_1_D L1420-1424
        exprs.append(
            (pl.col("volume").rolling_mean(20) / vol_ma1440)
            .alias("e1d_volume_ma20_rel")
        )

        # Volume ratio: 学習側 volume / rolling_mean(20)  ← +1e-10 なし (inf 伝播)
        # 参照: engine_1_D L1425-1428
        exprs.append(
            (pl.col("volume") / pl.col("volume").rolling_mean(20))
            .alias("e1d_volume_ratio")
        )

        # Volume Price Trend normalized:
        #   学習側: (pct_change * volume).rolling_mean(10) / vol_ma1440
        # 参照: engine_1_D L1430-1435
        exprs.append(
            (
                (pl.col("close").pct_change() * pl.col("volume")).rolling_mean(10)
                / vol_ma1440
            ).alias("e1d_volume_price_trend_norm")
        )

        # =====================================================================
        # Breakout / Range Group
        # 参照: engine_1_D._create_breakout_features (L1439-1514)
        # =====================================================================
        for window in [10, 20, 50, 100]:
            donchian_upper  = pl.col("high").rolling_max(window)
            donchian_lower  = pl.col("low").rolling_min(window)
            donchian_middle = (donchian_upper + donchian_lower) / 2.0

            # Donchian distances (ATR割り)
            exprs.append(
                ((donchian_upper - pl.col("close")) / (pl.col("__temp_atr_13") + 1e-10))
                .alias(f"e1d_donchian_upper_dist_{window}")
            )
            exprs.append(
                ((donchian_middle - pl.col("close")) / (pl.col("__temp_atr_13") + 1e-10))
                .alias(f"e1d_donchian_middle_dist_{window}")
            )
            exprs.append(
                ((pl.col("close") - donchian_lower) / (pl.col("__temp_atr_13") + 1e-10))
                .alias(f"e1d_donchian_lower_dist_{window}")
            )

            # Price Channel: 学習側でも donchian と完全同値 (high.rolling_max / low.rolling_min)
            # 参照: engine_1_D L1483-1494
            p_upper = pl.col("high").rolling_max(window)
            p_lower = pl.col("low").rolling_min(window)
            exprs.append(
                ((p_upper - pl.col("close")) / (pl.col("__temp_atr_13") + 1e-10))
                .alias(f"e1d_price_channel_upper_dist_{window}")
            )
            exprs.append(
                ((pl.col("close") - p_lower) / (pl.col("__temp_atr_13") + 1e-10))
                .alias(f"e1d_price_channel_lower_dist_{window}")
            )

        # Commodity Channel Index (重量UDF): window=[14, 20]
        # 参照: engine_1_D L1499-1513
        for window in [14, 20]:
            exprs.append(
                pl.struct(["high", "low", "close"]).map_batches(
                    lambda s, w=window: commodity_channel_index_udf(
                        s.struct.field("high").to_numpy(),
                        s.struct.field("low").to_numpy(),
                        s.struct.field("close").to_numpy(),
                        w,
                    ),
                    return_dtype=pl.Float64,
                ).alias(f"e1d_commodity_channel_index_{window}")
            )

        # =====================================================================
        # Support / Resistance Group
        # 参照: engine_1_D._create_support_resistance_features (L1516-1599)
        # =====================================================================
        # 真のローリングピボット (直近20期間の波から算出)
        # 参照: engine_1_D L1542-1546
        prev_high  = pl.col("high").rolling_max(20).shift(1)
        prev_low   = pl.col("low").rolling_min(20).shift(1)
        prev_close = pl.col("close").shift(1)
        pivot = (prev_high + prev_low + prev_close) / 3.0
        r1 = 2.0 * pivot - prev_low
        s1 = 2.0 * pivot - prev_high

        exprs.append(
            ((pl.col("close") - pivot) / (pl.col("__temp_atr_13") + 1e-10))
            .alias("e1d_pivot_dist")
        )
        exprs.append(
            ((r1 - pl.col("close")) / (pl.col("__temp_atr_13") + 1e-10))
            .alias("e1d_resistance1_dist")
        )
        exprs.append(
            ((pl.col("close") - s1) / (pl.col("__temp_atr_13") + 1e-10))
            .alias("e1d_support1_dist")
        )

        # フィボナッチレベル (重量UDF, 50% レベルのみ取得)
        # 参照: engine_1_D L1570-1582
        fib_50_raw = pl.struct(["high", "low"]).map_batches(
            lambda s: fibonacci_levels_udf(
                s.struct.field("high").to_numpy(),
                s.struct.field("low").to_numpy(),
                50,
            )[:, 2],
            return_dtype=pl.Float64,
        )
        exprs.append(
            ((pl.col("close") - fib_50_raw) / (pl.col("__temp_atr_13") + 1e-10))
            .alias("e1d_fib_level_50_dist")
        )

        # ローソク足パターン (重量UDF)
        # 参照: engine_1_D L1583-1597
        exprs.append(
            pl.struct(["open", "high", "low", "close"]).map_batches(
                lambda s: candlestick_patterns_udf(
                    s.struct.field("open").to_numpy(),
                    s.struct.field("high").to_numpy(),
                    s.struct.field("low").to_numpy(),
                    s.struct.field("close").to_numpy(),
                ),
                return_dtype=pl.Float64,
            ).alias("e1d_candlestick_pattern")
        )

        # =====================================================================
        # Price Action Group
        # 参照: engine_1_D._create_price_action_features (L1601-1674)
        # =====================================================================
        typical_p  = (pl.col("high") + pl.col("low") + pl.col("close")) / 3.0
        weighted_c = (pl.col("high") + pl.col("low") + 2 * pl.col("close")) / 4.0
        median_p   = (pl.col("high") + pl.col("low")) / 2.0

        exprs.append(
            ((typical_p - pl.col("close")) / (pl.col("__temp_atr_13") + 1e-10))
            .alias("e1d_typical_price_dist")
        )
        exprs.append(
            ((weighted_c - pl.col("close")) / (pl.col("__temp_atr_13") + 1e-10))
            .alias("e1d_weighted_close_dist")
        )
        exprs.append(
            ((median_p - pl.col("close")) / (pl.col("__temp_atr_13") + 1e-10))
            .alias("e1d_median_price_dist")
        )

        # ローソク足構成要素
        exprs.append(
            (
                (pl.col("close") - pl.col("open")).abs()
                / (pl.col("__temp_atr_13") + 1e-10)
            ).alias("e1d_body_size_atr")
        )

        # HL比率系 (ATR 不依存)
        hl_range_safe = pl.col("high") - pl.col("low") + 1e-10
        exprs.append(
            ((pl.col("high") - pl.max_horizontal("open", "close")) / hl_range_safe)
            .alias("e1d_upper_wick_ratio")
        )
        exprs.append(
            ((pl.min_horizontal("open", "close") - pl.col("low")) / hl_range_safe)
            .alias("e1d_lower_wick_ratio")
        )
        exprs.append(
            ((pl.col("close") - pl.col("low")) / hl_range_safe)
            .alias("e1d_price_location_hl")
        )

        # イントラデイ・オーバーナイト (ATR 不依存)
        exprs.append(
            ((pl.col("close") - pl.col("open")) / (pl.col("open") + 1e-10))
            .alias("e1d_intraday_return")
        )
        exprs.append(
            (
                (pl.col("open") - pl.col("close").shift(1))
                / (pl.col("close").shift(1) + 1e-10)
            ).alias("e1d_overnight_gap")
        )

        # =====================================================================
        # Polars 単一 .select() で全式を一括計算 (FFI overhead 1回のみ)
        # =====================================================================
        result_df = df.lazy().select(exprs).tail(1).collect()
        polars_result = result_df.to_dicts()[0]

        # Polars null → np.nan に変換
        for k, v in polars_result.items():
            features[k] = float(v) if v is not None else np.nan

        # =====================================================================
        # Layer 2: Numba UDF / SSoT (scalar UDFs, 直接呼び出し)
        # 学習側も同一 UDF を rolling_map で呼ぶため最終バーは数値完全一致。
        # =====================================================================

        # pct_change を numpy で計算 (Polars pct_change と semantics 一致)
        pct_arr = _pct_change(close_arr)

        # hv_standard / hv_robust per window
        # 学習側: rolling_map(lambda s: hv_*_udf(s.to_numpy()), window_size=w, min_samples=w)
        # 本番側: 最終バーのみ hv_*_udf(pct_arr[-w:]) で計算
        # min_samples=w → w 本未満は NaN を返す挙動も学習側と一致。
        for w in [10, 20, 30, 50]:
            if len(pct_arr) < w:
                features[f"e1d_hv_standard_{w}"] = np.nan
                features[f"e1d_hv_robust_{w}"]   = np.nan
            else:
                features[f"e1d_hv_standard_{w}"] = float(hv_standard_udf(pct_arr[-w:]))
                features[f"e1d_hv_robust_{w}"]   = float(hv_robust_udf(pct_arr[-w:]))

        # hv_robust_annual_252:
        # 学習側: rolling_map(lambda s: hv_robust_udf(s.to_numpy()) * sqrt(252),
        #                      window_size=252, min_samples=252)
        if len(pct_arr) < 252:
            features["e1d_hv_robust_annual_252"] = np.nan
        else:
            features["e1d_hv_robust_annual_252"] = float(
                hv_robust_udf(pct_arr[-252:]) * np.sqrt(252)
            )

        # サンプルウェイト (学習側 base_columns 扱いと一致、QA対象外)
        # 参照: engine_1_D L1199-1211
        sample_weight_arr = calculate_sample_weight(high_arr, low_arr, close_arr)
        features["e1d_sample_weight"] = float(sample_weight_arr[-1])

        # =====================================================================
        # QA 処理 (学習側 apply_quality_assurance_to_group と等価)
        # e1d_sample_weight は学習側 base_columns 扱いで QA 対象外 (Phase 5 #36 修正)。
        # qa_state=None の場合は inf/NaN → 0.0 のフォールバックのみ。
        # =====================================================================
        if qa_state is not None:
            qa_result: Dict[str, float] = {}
            for key, val in features.items():
                if key == "e1d_sample_weight":
                    qa_result[key] = val  # sample_weight は QA 対象外
                else:
                    qa_result[key] = qa_state.update_and_clip(key, val)
            features = qa_result
        else:
            for key in list(features.keys()):
                if key != "e1d_sample_weight" and not np.isfinite(features[key]):
                    features[key] = 0.0

        return features
