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
# 【Phase E (stable_rolling SSoT) 適用】
#   Polars 組込 rolling_{mean,std} は内部 running 累積実装で context 長依存性
#   がある (学習側 3.4M bars と本番側 ~2980 bars deque で結果が乖離)。本ファイル
#   では全 rolling_{mean,std} 呼出しを Option C2 で置換:
#     - numpy で stable_rolling_{mean,std} を事前計算 → `__num_*` 列に注入
#     - expression 内では `pl.col("__num_...")` で参照のみ
#     - map_batches は stable_rolling 経由のものを完全排除 (CSE non-determinism 回避)
#   rolling_{max,min,quantile,map} は context 長非依存のため変更なし。
#
#   ⚠ verify 時のリスク注記:
#     D-pair 適用で rfe_1D の expression 構造が変化 (rolling 6 個が
#     `pl.col(__num_*)` 参照に置換、列 6 個が新規追加)。Cluster A の機序
#     (= expression 集合の構造変化が Polars plan / CSE 経路を切替) の
#     観点では、他 engine (e1a/e1b/e1c/e1e/e1f) の結果が変動する可能性は
#     ゼロではない。shadow_mode 検証時に e1d 以外の cells も diff チェック
#     することを推奨。
#
# 【SSoT 階層】(§B.12.13.9 で再定義)
#   Layer 1 (経路): 学習側 = Polars map_batches、本番側 = numpy 直呼び + 列注入
#   Layer 2 (真の SSoT): stable_rolling.py (Numba 関数) + core_indicators.py
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
# ─────────────────────────────────────────────────────────────
# [Phase E (stable_rolling SSoT)] Polars rolling running 実装の
# context 長依存性を排除するため、両側で共通の Numba 関数を import。
# 本番側は Option C2: numpy 事前計算 + 列注入で使用 (map_batches 完全排除)。
# ─────────────────────────────────────────────────────────────
from stable_rolling import (
    stable_rolling_mean,
    stable_rolling_std,
)

# ─────────────────────────────────────────────────────────────
# [Plan §B.12.14.10] _pct_change SSoT 集約 — local 重複定義を撤廃。
# canonical 実装は core/numpy_helpers.py の pct_change_polars_compat。
# rfe_1A/1B/1D/1E/1F の 5 file で共通 import に統一する。
# 旧実装は @nb.njit JIT 版だったが、純 numpy vectorized でも本番運用上
# 十分高速 (3500 行で ~9.5μs/call、1 時間あたり 0.0007 秒の差で実害ゼロ)、
# JIT cache 不要で cache_key 設計が単純化する。
# ─────────────────────────────────────────────────────────────
from numpy_helpers import pct_change_polars_compat as _pct_change
# --------------------------------------------------------

import numpy as np
import polars as pl
import numba as nb
from typing import Dict, Optional, Tuple, List


# ==================================================================
# ヘルパー関数
# ==================================================================


# ==================================================================
# QAState — 学習側 apply_quality_assurance_to_group の等価実装
# (Phase 9b では変更なし - Layer 1/2 のロジックとは独立)
# ==================================================================

class QAState:
    """学習側 apply_quality_assurance_to_group のリアルタイム等価実装。
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

        # ▼▼ [§B.12.13.7 Option C2 / Phase E] numpy 事前計算 + 列注入 ▼▼
        # Polars map_batches を経由しない (CSE non-determinism 回避)。
        # 命名規約:
        #   単純: __num_<func>_<col>_<window>  (例: __num_srm_volume_20)
        #   複合: __num_<func>_<expr_id>_<window>  (例: __num_srs_pct_close_252)

        # --- 単純パターン: volume.rolling_mean ---
        # 用途:
        #   - volume.rolling_mean(lookback_bars) → vol_ma1440 (Relative Volume base)
        #   - volume.rolling_mean(20) → volume_ma20_rel と volume_ratio で共用
        columns["__num_srm_volume_lookback"] = stable_rolling_mean(volume_arr, lookback_bars)
        columns["__num_srm_volume_20"] = stable_rolling_mean(volume_arr, 20)

        # --- 複合パターン: pct_change(close) を numpy で先に計算 ---
        # pct_change: 先頭 1 NaN、stable_rolling_std は window 内 NaN で出力 NaN
        # → Polars `pct_change().rolling_std(W, ddof=1)` と NaN 位置・値が完全一致
        # (実機で B 側にて検証済)
        # 注意: 計算式は `(c[i] - c[i-1]) / c[i-1]` 形式を使う。
        #   `c[i] / c[i-1] - 1.0` は数学的等価だが IEEE 754 で 1 ULP 差が出るため不可。
        #   Polars `pct_change()` の内部実装に合わせる。
        pct_close = np.empty(len(close_arr), dtype=np.float64)
        pct_close[0] = np.nan
        if len(close_arr) > 1:
            with np.errstate(divide="ignore", invalid="ignore"):
                pct_close[1:] = (close_arr[1:] - close_arr[:-1]) / close_arr[:-1]

        # hv_annual_252: pct_change.rolling_std(252, ddof=1)
        # → __num_srs_pct_close_252
        columns["__num_srs_pct_close_252"] = stable_rolling_std(pct_close, 252, 1)

        # hv_regime_50 内の hv_50: pct_change.rolling_std(50, ddof=1)
        # → __num_srs_pct_close_50
        columns["__num_srs_pct_close_50"] = stable_rolling_std(pct_close, 50, 1)

        # --- 複合パターン: (pct_change(close) * volume).rolling_mean(10) ---
        # 用途: volume_price_trend_norm の分子
        # pv = pct_close * volume_arr (先頭 1 NaN を継承)
        # stable_rolling_mean(pv, 10) → 先頭 10 NaN (window 内 NaN 含むため)
        # ※ 命名: __num_srm_pv_10 (pv = pct_close × volume)
        #   注意: B 側の __num_srm_vpt_10 は別物 (B 側は pct_close × rel_volume)。
        #   D 側では「pct_change * volume」を rolling した結果に vol_ma1440 で割るため、
        #   分子と分母を分離している。
        _pv = (pct_close * volume_arr).astype(np.float64)
        columns["__num_srm_pv_10"] = stable_rolling_mean(_pv, 10)
        # ▲▲ [§B.12.13.7 Option C2 / Phase E] ▲▲

        # ===== exprs =====
        exprs: List[pl.Expr] = []

        # =====================================================================
        # Volatility Group (Polars 部分)
        # 参照: engine_1_D._create_volatility_features (L1183-1309)
        # =====================================================================

        # hv_annual_252: 学習側 rolling_std(252, ddof=1) * sqrt(252) と完全一致
        # ▼▼ [Phase E] pct_change.rolling_std(252) → __num_srs_pct_close_252 列参照
        exprs.append(
            (pl.col("__num_srs_pct_close_252") * np.sqrt(252))
            .alias("e1d_hv_annual_252")
        )
        # ▲▲ [Phase E] ▲▲

        # hv_regime_50: 学習側 Polars ネイティブのローリング分位数判定と完全一致
        # 参照: engine_1_D L1265-1273
        # ▼▼ [Phase E] pct_change.rolling_std(50) → __num_srs_pct_close_50 列参照
        hv_50 = pl.col("__num_srs_pct_close_50")
        # ▲▲ [Phase E] ▲▲
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
        # ▼▼ [Phase E] volume.rolling_mean(lookback_bars) → __num_srm_volume_lookback 列参照
        vol_ma1440 = pl.col("__num_srm_volume_lookback") + 1e-10
        # ▲▲ [Phase E] ▲▲

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
        # ▼▼ [Phase E] volume.rolling_mean(20) → __num_srm_volume_20 列参照
        exprs.append(
            (pl.col("__num_srm_volume_20") / vol_ma1440)
            .alias("e1d_volume_ma20_rel")
        )

        # Volume ratio: 学習側 volume / rolling_mean(20)  ← +1e-10 なし (inf 伝播)
        # 参照: engine_1_D L1425-1428
        exprs.append(
            (pl.col("volume") / pl.col("__num_srm_volume_20"))
            .alias("e1d_volume_ratio")
        )

        # Volume Price Trend normalized:
        #   学習側: (pct_change * volume).rolling_mean(10) / vol_ma1440
        # 参照: engine_1_D L1430-1435
        # ▼▼ [Phase E] (pct_change*volume).rolling_mean(10) → __num_srm_pv_10 列参照
        exprs.append(
            (pl.col("__num_srm_pv_10") / vol_ma1440)
            .alias("e1d_volume_price_trend_norm")
        )
        # ▲▲ [Phase E] ▲▲

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
