# realtime_feature_engine_1D_volume.py
# Category 1D: 出来高・ボラティリティ・プライスアクション系
# (Volume, Volatility & Price Action)
#
# ==================================================================
# 【Phase 9b 改修】司令塔統合 .select() 対応 (FFI overhead 削減)
# ==================================================================
#
# 目的: Phase 9 (Step B) で達成した Polars 直呼びによる学習側との
#       ビット完全一致を保ったまま、6 モジュールの Polars 式を司令塔
#       で 1 回の .select() に統合できるよう構造を分解する。
#
# 【Phase 9b の改修】
#   追加: `_build_polars_pieces(data, lookback_bars) -> (columns, exprs, layer2)`
#     - columns: DataFrame に追加する列辞書 (close/high/low/open/volume +
#                __temp_atr_13 (raw ATR13))
#     - exprs:   Polars 式リスト (各 alias は最終特徴量名 e1d_*)
#     - layer2:  Polars 経由しないスカラー特徴量
#                (hv_standard/hv_robust × 4 windows + hv_robust_annual_252 +
#                 e1d_sample_weight)
#   変更: `calculate_features` は `_build_polars_pieces` を呼んで単独計算する
#         薄いラッパーへ。後方互換完全維持。
#
# 【1D の特徴】
#   多数の重量 UDF (chaikin_volatility/mass_index/cmf/mfi/vwap/obv/ad/
#   force_index/cci/fibonacci/candlestick) を Polars `map_batches` で
#   呼び出す構造になっている。これらは全て exprs に集約され、司令塔の
#   統合 .select() でも同じ map_batches 経由で呼ばれる (Polars 統一を維持)。
#
#   Layer 2 (scalar UDF) は学習側 rolling_map 形式の hv_standard/hv_robust
#   と calculate_sample_weight のみ。これらは元から numpy 配列を直接渡す
#   形式で、Polars には乗っていない (学習側もスカラー UDF を rolling_map で
#   呼んでいるため、本番側は最終バーのみ計算でビット一致)。
#
# 【ATR13 の扱い】
#   学習側: atr_13_internal_expr = pl.struct(...).map_batches(calculate_atr_wilder)
#           → 割り算時に + 1e-10 を加える (`expr + 1e-10`)
#   本番側: numpy で事前計算して __temp_atr_13 列として渡す (raw、+1e-10 なし)
#           → 割り算時に Polars 式で `(pl.col("__temp_atr_13") + 1e-10)` を使う
#   結果: 学習側と完全同値の計算式
#
# 【SSoT 階層】(Phase 9 から不変)
#   Layer 1 (rolling 統計): Polars Rust エンジン
#   Layer 2 (Numba UDF):    core_indicators (SSoT)
#
# 【保持される過去の修正】
#   ・QAState (apply_quality_assurance_to_group の等価実装、bias=False 補正)
#   ・e1d_sample_weight は学習側 base_columns 扱いで QA 対象外 (Phase 5 #36)
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
from typing import Dict, Optional, Tuple, List


# ==================================================================
# ヘルパー関数
# ==================================================================

@nb.njit(fastmath=False, cache=True)
def _pct_change(arr: np.ndarray) -> np.ndarray:
    """Polars pct_change() と完全一致: (x[i] - x[i-1]) / x[i-1]
    prev == 0 のとき: x[i] > 0 → +inf, x[i] < 0 → -inf, x[i] == 0 → NaN
    先頭は nan。

    Layer 2 (Numba UDF) のうち hv_standard_udf / hv_robust_udf 等の入力に使用。
    Polars rolling_map 経由で同じ pct_change を渡す学習側と数値完全一致。
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
# (Phase 9b では変更なし - Layer 1/2 のロジックとは独立)
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

class FeatureModule1D:

    QAState = QAState

    @staticmethod
    def _build_polars_pieces(
        data: Dict[str, np.ndarray],
        lookback_bars: int = 1440,
    ) -> Tuple[Dict[str, np.ndarray], List[pl.Expr], Dict[str, float]]:
        """
        統合 .select() 用の 3 要素を返す。

        Returns:
            columns: Dict[str, np.ndarray]
                共通列 (close/high/low/open/volume) + 1D 固有の __temp_atr_13 (raw)。
            exprs: List[pl.Expr]
                Polars 式リスト (alias は e1d_* の最終特徴量名)。
                重量 UDF (chaikin_volatility/mass_index/cmf/mfi/vwap/obv/ad/
                force_index/cci/fibonacci/candlestick) は map_batches 経由で
                Polars 式に組み込まれている (学習側と同一の Polars 統一)。
            layer2: Dict[str, float]
                Polars 経由しないスカラー特徴量。
                - hv_standard/hv_robust × [10, 20, 30, 50]
                - hv_robust_annual_252
                - e1d_sample_weight (QA対象外)
        """
        # 必須キーガード
        for _key in ("close", "high", "low", "open", "volume"):
            if _key not in data:
                return {}, [], {}

        close_arr  = data["close"].astype(np.float64)
        high_arr   = data["high"].astype(np.float64)
        low_arr    = data["low"].astype(np.float64)
        open_arr   = data["open"].astype(np.float64)
        volume_arr = data["volume"].astype(np.float64)

        if len(close_arr) == 0:
            return {}, [], {}

        # ---------------------------------------------------------
        # ATR13 計算 (学習側 atr_13_internal_expr と完全一致)
        #   学習側: pl.struct([...]).map_batches(calculate_atr_wilder)
        #   本番側: numpy で事前計算して __temp_atr_13 列に
        # 学習側は割り算時に + 1e-10 を加えるため、ここでは raw ATR を保持する。
        # ---------------------------------------------------------
        atr13_arr = calculate_atr_wilder(high_arr, low_arr, close_arr, 13)

        # ===== columns =====
        columns: Dict[str, np.ndarray] = {
            "close":  close_arr,
            "high":   high_arr,
            "low":    low_arr,
            "open":   open_arr,
            "volume": volume_arr,
            "__temp_atr_13": atr13_arr,
        }

        # ===== exprs =====
        exprs: List[pl.Expr] = []

        # =====================================================================
        # Volatility Group (Polars 部分)
        # 参照: engine_1_D._create_volatility_features (L1183-1309)
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

        # ===== layer2 =====
        # Numba scalar UDF 直接呼び (学習側 rolling_map と最終バー値が同一)
        layer2: Dict[str, float] = {}

        # pct_change を numpy で計算 (Polars pct_change と semantics 一致)
        pct_arr = _pct_change(close_arr)

        # hv_standard / hv_robust per window
        # 学習側: rolling_map(lambda s: hv_*_udf(s.to_numpy()), window_size=w, min_samples=w)
        # 本番側: 最終バーのみ hv_*_udf(pct_arr[-w:]) で計算
        # min_samples=w → w 本未満は NaN を返す挙動も学習側と一致。
        for w in [10, 20, 30, 50]:
            if len(pct_arr) < w:
                layer2[f"e1d_hv_standard_{w}"] = np.nan
                layer2[f"e1d_hv_robust_{w}"]   = np.nan
            else:
                layer2[f"e1d_hv_standard_{w}"] = float(hv_standard_udf(pct_arr[-w:]))
                layer2[f"e1d_hv_robust_{w}"]   = float(hv_robust_udf(pct_arr[-w:]))

        # hv_robust_annual_252:
        # 学習側: rolling_map(lambda s: hv_robust_udf(s.to_numpy()) * sqrt(252),
        #                      window_size=252, min_samples=252)
        if len(pct_arr) < 252:
            layer2["e1d_hv_robust_annual_252"] = np.nan
        else:
            layer2["e1d_hv_robust_annual_252"] = float(
                hv_robust_udf(pct_arr[-252:]) * np.sqrt(252)
            )

        # サンプルウェイト (学習側 base_columns 扱いと一致、QA対象外)
        # 参照: engine_1_D L1199-1211
        sample_weight_arr = calculate_sample_weight(high_arr, low_arr, close_arr)
        layer2["e1d_sample_weight"] = float(sample_weight_arr[-1])

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
        """
        columns, exprs, layer2 = FeatureModule1D._build_polars_pieces(data, lookback_bars)
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
        # e1d_sample_weight は学習側 base_columns 扱いで QA 対象外 (Phase 5 #36)
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
