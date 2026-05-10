# realtime_feature_engine_1C_technical.py
# Project Cimera V5 - Feature Module 1C (Technical Indicators)
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
#                Numba UDF 事前計算結果列 __rsi_14, __atr_13_safe 等)
#     - exprs:   Polars 式リスト (各 alias は最終特徴量名 e1c_*)
#     - layer2:  Polars 経由しないスカラー特徴量 (1C では RVI フォールバックのみ)
#   変更: `calculate_features` は `_build_polars_pieces` を呼んで単独計算する
#         薄いラッパーへ。後方互換完全維持。
#
# 【1C の特徴】
#   1C は全 191 特徴量がきれいに 2 層に分かれている:
#     - Numba UDF の出力を事前計算して列化 (__rsi_*, __atr_*_raw, __adx_*,
#       __di_*, __aroon_*, __williams_*, __stoch_*, __trix_*, __tsi_*, __uo,
#       __wma_*, __hma_*, __kama_*)
#     - 全特徴量を Polars 式で表現 (BB/MACD/STC/KST/Coppock/SMA/EMA/RVI 等)
#   この構造は Phase 9b の `_build_polars_pieces` に素直にマッピングできる。
#   1B の t_dist_scale_50 のような Layer 1/Layer 2 の cross-dependency は無い。
#
# 【SSoT 階層】(Phase 9 から不変)
#   Layer 1 (rolling 統計): Polars Rust エンジン
#   Layer 2 (Numba UDF):    core_indicators (SSoT、列化して Polars に注入)
#
# 【保持される過去の修正】
#   ・atr_ok ガード廃止 → __atr_13_safe = atr_13_raw + 1e-10 統一 (Phase 9 #52)
#   ・Williams %R: late binding バグ解消後の period=14/28/56 個別計算
#   ・atr_pct のゼロ保護: close + 1e-10 で学習側と完全一致
#   ・STC の half_life EMA + span=3 EMA + 2nd Stochastic + Final Smooth 完全実装
# ==================================================================

import sys
from pathlib import Path
from typing import Dict, Optional, Tuple, List

import numpy as np
import polars as pl

sys.path.append(str(Path(__file__).resolve().parent.parent))
import blueprint as config
sys.path.append(str(config.CORE_DIR))
from core_indicators import (
    calculate_atr_wilder,
    calculate_rsi_wilder,
    calculate_adx,
    # [SSoT 統一] Engine 1C の Numba 関数 (案-3完全SSoT)
    calculate_wma_numba,
    calculate_hma_numba,
    calculate_kama_numba,
    calculate_stochastic_numba,
    calculate_williams_r_numba,
    calculate_trix_numba,
    calculate_ultimate_oscillator_numba,
    calculate_aroon_up_numba,
    calculate_aroon_down_numba,
    calculate_tsi_numba,
    _calculate_di_wilder,
)


# ==================================================================
# QAState — 学習側 apply_quality_assurance の等価実装
# (1A/1B と完全に同一の実装。Phase 9b では変更なし。)
# ==================================================================

class QAState:
    """学習側 apply_quality_assurance のリアルタイム等価実装。
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

class FeatureModule1C:

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
                DataFrame に追加する列辞書。共通列 (close/high/low/open/volume) と
                1C 固有の Numba UDF 事前計算結果列 (__rsi_*, __atr_*_safe 等)。
            exprs: List[pl.Expr]
                Polars 式リスト。各 alias は最終特徴量名 (e1c_*)。
            layer2: Dict[str, float]
                Polars 経由しないスカラー特徴量 (1C では RVI フォールバックのみ)。
        """
        # ---------------------------------------------------------
        # 入力配列の準備
        # ---------------------------------------------------------
        _empty = np.array([], dtype=np.float64)
        close_arr  = np.asarray(data.get("close",  _empty), dtype=np.float64)
        if len(close_arr) == 0:
            return {}, [], {}

        high_arr   = np.asarray(data.get("high",   _empty), dtype=np.float64)
        low_arr    = np.asarray(data.get("low",    _empty), dtype=np.float64)
        volume_arr = np.asarray(data.get("volume", _empty), dtype=np.float64)
        open_arr   = np.asarray(data.get("open",   _empty), dtype=np.float64)

        n = len(close_arr)
        nan_arr = np.full(n, np.nan, dtype=np.float64)

        # ---------------------------------------------------------
        # ATR 配列を全期間分事前計算 (学習側 map_batches と等価)
        # 学習側は scale_by_atr(target, atr_raw) を使うため + 1e-10 はゼロ保護用。
        # 本番側では __atr_13_safe = atr_13_raw + 1e-10 を列化して
        # `target / __atr_13_safe` で割れば scale_by_atr と完全一致 (Phase 9 #52)。
        # ---------------------------------------------------------
        if len(high_arr) > 0 and len(low_arr) > 0:
            atr_13_raw = calculate_atr_wilder(high_arr, low_arr, close_arr, 13)
            atr_21_raw = calculate_atr_wilder(high_arr, low_arr, close_arr, 21)
            atr_34_raw = calculate_atr_wilder(high_arr, low_arr, close_arr, 34)
            atr_55_raw = calculate_atr_wilder(high_arr, low_arr, close_arr, 55)
        else:
            atr_13_raw = nan_arr.copy()
            atr_21_raw = nan_arr.copy()
            atr_34_raw = nan_arr.copy()
            atr_55_raw = nan_arr.copy()

        atr_13_safe = atr_13_raw + 1e-10  # scale_by_atr 等価

        # ---------------------------------------------------------
        # 全 Numba UDF を事前計算 (学習側 map_batches と等価、配列を返す)
        # ---------------------------------------------------------
        # RSI
        rsi_14_arr = calculate_rsi_wilder(close_arr, 14)
        rsi_21_arr = calculate_rsi_wilder(close_arr, 21)
        rsi_30_arr = calculate_rsi_wilder(close_arr, 30)
        rsi_50_arr = calculate_rsi_wilder(close_arr, 50)

        # ADX / DI / Aroon / Williams / Stochastic (high/low 必要)
        if len(high_arr) > 0 and len(low_arr) > 0:
            adx_13_arr = calculate_adx(high_arr, low_arr, close_arr, 13)
            adx_21_arr = calculate_adx(high_arr, low_arr, close_arr, 21)
            adx_34_arr = calculate_adx(high_arr, low_arr, close_arr, 34)
            di_p_13, di_m_13 = _calculate_di_wilder(high_arr, low_arr, close_arr, 13)
            di_p_21, di_m_21 = _calculate_di_wilder(high_arr, low_arr, close_arr, 21)
            di_p_34, di_m_34 = _calculate_di_wilder(high_arr, low_arr, close_arr, 34)

            aroon_u_14 = calculate_aroon_up_numba(high_arr, 14)
            aroon_u_25 = calculate_aroon_up_numba(high_arr, 25)
            aroon_u_50 = calculate_aroon_up_numba(high_arr, 50)
            aroon_d_14 = calculate_aroon_down_numba(low_arr, 14)
            aroon_d_25 = calculate_aroon_down_numba(low_arr, 25)
            aroon_d_50 = calculate_aroon_down_numba(low_arr, 50)

            williams_14 = calculate_williams_r_numba(high_arr, low_arr, close_arr, 14)
            williams_28 = calculate_williams_r_numba(high_arr, low_arr, close_arr, 28)
            williams_56 = calculate_williams_r_numba(high_arr, low_arr, close_arr, 56)

            stoch_k_14_3_3 = calculate_stochastic_numba(high_arr, low_arr, close_arr, 14, 3, 3)
            stoch_k_21_5_5 = calculate_stochastic_numba(high_arr, low_arr, close_arr, 21, 5, 5)
            stoch_k_9_3_3  = calculate_stochastic_numba(high_arr, low_arr, close_arr, 9,  3, 3)
        else:
            adx_13_arr = adx_21_arr = adx_34_arr = nan_arr.copy()
            di_p_13 = di_m_13 = di_p_21 = di_m_21 = di_p_34 = di_m_34 = nan_arr.copy()
            aroon_u_14 = aroon_u_25 = aroon_u_50 = nan_arr.copy()
            aroon_d_14 = aroon_d_25 = aroon_d_50 = nan_arr.copy()
            williams_14 = williams_28 = williams_56 = nan_arr.copy()
            stoch_k_14_3_3 = stoch_k_21_5_5 = stoch_k_9_3_3 = nan_arr.copy()

        # TRIX / TSI / UO (UO は volume 必要)
        trix_14 = calculate_trix_numba(close_arr, 14)
        trix_20 = calculate_trix_numba(close_arr, 20)
        trix_30 = calculate_trix_numba(close_arr, 30)
        tsi_25  = calculate_tsi_numba(close_arr, 25)
        tsi_13  = calculate_tsi_numba(close_arr, 13)

        if len(high_arr) > 0 and len(low_arr) > 0 and len(volume_arr) > 0:
            uo_arr = calculate_ultimate_oscillator_numba(high_arr, low_arr, close_arr, volume_arr)
        else:
            uo_arr = nan_arr.copy()

        # WMA / HMA / KAMA
        wma_10  = calculate_wma_numba(close_arr, 10)
        wma_20  = calculate_wma_numba(close_arr, 20)
        wma_50  = calculate_wma_numba(close_arr, 50)
        wma_100 = calculate_wma_numba(close_arr, 100)
        wma_200 = calculate_wma_numba(close_arr, 200)
        hma_21  = calculate_hma_numba(close_arr, 21)
        hma_34  = calculate_hma_numba(close_arr, 34)
        hma_55  = calculate_hma_numba(close_arr, 55)
        kama_21 = calculate_kama_numba(close_arr, 21)
        kama_34 = calculate_kama_numba(close_arr, 34)

        # ===== columns =====
        columns: Dict[str, np.ndarray] = {
            "close": close_arr,
            "__atr_13_raw": atr_13_raw,
            "__atr_21_raw": atr_21_raw,
            "__atr_34_raw": atr_34_raw,
            "__atr_55_raw": atr_55_raw,
            "__atr_13_safe": atr_13_safe,
            "__rsi_14": rsi_14_arr,
            "__rsi_21": rsi_21_arr,
            "__rsi_30": rsi_30_arr,
            "__rsi_50": rsi_50_arr,
            "__adx_13": adx_13_arr,
            "__adx_21": adx_21_arr,
            "__adx_34": adx_34_arr,
            "__di_p_13": di_p_13, "__di_m_13": di_m_13,
            "__di_p_21": di_p_21, "__di_m_21": di_m_21,
            "__di_p_34": di_p_34, "__di_m_34": di_m_34,
            "__aroon_u_14": aroon_u_14, "__aroon_d_14": aroon_d_14,
            "__aroon_u_25": aroon_u_25, "__aroon_d_25": aroon_d_25,
            "__aroon_u_50": aroon_u_50, "__aroon_d_50": aroon_d_50,
            "__williams_14": williams_14,
            "__williams_28": williams_28,
            "__williams_56": williams_56,
            "__stoch_k_14_3_3": stoch_k_14_3_3,
            "__stoch_k_21_5_5": stoch_k_21_5_5,
            "__stoch_k_9_3_3":  stoch_k_9_3_3,
            "__trix_14": trix_14, "__trix_20": trix_20, "__trix_30": trix_30,
            "__tsi_25": tsi_25,   "__tsi_13": tsi_13,
            "__uo": uo_arr,
            "__wma_10": wma_10, "__wma_20": wma_20, "__wma_50": wma_50,
            "__wma_100": wma_100, "__wma_200": wma_200,
            "__hma_21": hma_21, "__hma_34": hma_34, "__hma_55": hma_55,
            "__kama_21": kama_21, "__kama_34": kama_34,
        }
        if len(high_arr) > 0:
            columns["high"] = high_arr
        if len(low_arr) > 0:
            columns["low"] = low_arr
        if len(open_arr) > 0:
            columns["open"] = open_arr
        if len(volume_arr) > 0:
            columns["volume"] = volume_arr

        # ===== exprs =====
        # 学習側 engine_1_C の各 create_*_features と完全一致する式
        exprs: List[pl.Expr] = []
        atr_safe = pl.col("__atr_13_safe")  # ショートカット

        # ---------------------------------------------------------
        # ATR系 (4 stats × 4 periods = 16)
        # 参照: engine_1_C.create_atr_features (L1188-1281)
        # ---------------------------------------------------------
        for period, raw_col_name in [(13, "__atr_13_raw"), (21, "__atr_21_raw"),
                                      (34, "__atr_34_raw"), (55, "__atr_55_raw")]:
            atr_raw = pl.col(raw_col_name)
            exprs.append((atr_raw / atr_safe).alias(f"e1c_atr_{period}"))
            exprs.append((atr_raw / (pl.col("close") + 1e-10) * 100)
                         .alias(f"e1c_atr_pct_{period}"))
            exprs.append((atr_raw.diff() / atr_safe).alias(f"e1c_atr_trend_{period}"))
            exprs.append((atr_raw.rolling_std(period, ddof=1) / atr_safe)
                         .alias(f"e1c_atr_volatility_{period}"))

        # ---------------------------------------------------------
        # RSI系 (rsi/rsi_momentum × 4 + stochastic_rsi/rsi_divergence × 2 = 12)
        # 参照: engine_1_C.create_rsi_features (L934-998)
        # ---------------------------------------------------------
        for period in [14, 21, 30, 50]:
            exprs.append(pl.col(f"__rsi_{period}").alias(f"e1c_rsi_{period}"))
            exprs.append(pl.col(f"__rsi_{period}").diff().alias(f"e1c_rsi_momentum_{period}"))

        for period in [14, 21]:
            rsi_col = pl.col(f"__rsi_{period}")
            exprs.append(
                ((rsi_col - rsi_col.rolling_min(period))
                 / (rsi_col.rolling_max(period) - rsi_col.rolling_min(period) + 1e-10)
                 * 100)
                .alias(f"e1c_stochastic_rsi_{period}")
            )
            price_change = (
                (pl.col("close") - pl.col("close").shift(period))
                / pl.col("close").shift(period)
            )
            rsi_change = (rsi_col - rsi_col.shift(period)) / 50 - 1
            exprs.append((price_change - rsi_change).alias(f"e1c_rsi_divergence_{period}"))

        # ---------------------------------------------------------
        # MACD系 (3 stats × 3 configs = 9)
        # 参照: engine_1_C.create_macd_features (L1000-1080)
        # ---------------------------------------------------------
        for fast, slow, signal in [(12, 26, 9), (5, 35, 5), (19, 39, 9)]:
            ema_fast = pl.col("close").ewm_mean(span=fast, adjust=False)
            ema_slow = pl.col("close").ewm_mean(span=slow, adjust=False)
            macd_raw = ema_fast - ema_slow
            signal_raw = macd_raw.ewm_mean(span=signal, adjust=False)
            hist_raw = macd_raw - signal_raw
            exprs.append((macd_raw / atr_safe).alias(f"e1c_macd_{fast}_{slow}"))
            exprs.append((signal_raw / atr_safe).alias(f"e1c_macd_signal_{fast}_{slow}_{signal}"))
            exprs.append((hist_raw / atr_safe).alias(f"e1c_macd_histogram_{fast}_{slow}_{signal}"))

        # ---------------------------------------------------------
        # Bollinger系 (6 stats × 3 periods × 3 stdevs = 54)
        # 参照: engine_1_C.create_bollinger_features (L1082-1186)
        # ---------------------------------------------------------
        for period in [20, 30, 50]:
            sma = pl.col("close").rolling_mean(period)
            std = pl.col("close").rolling_std(period, ddof=1)
            for num_std in [2, 2.5, 3]:
                upper_raw = sma + num_std * std
                lower_raw = sma - num_std * std
                exprs.append(((upper_raw - pl.col("close")) / atr_safe)
                             .alias(f"e1c_bb_upper_{period}_{num_std}"))
                exprs.append(((pl.col("close") - lower_raw) / atr_safe)
                             .alias(f"e1c_bb_lower_{period}_{num_std}"))
                exprs.append(((pl.col("close") - lower_raw) / (upper_raw - lower_raw + 1e-10))
                             .alias(f"e1c_bb_percent_{period}_{num_std}"))
                exprs.append(((upper_raw - lower_raw) / atr_safe)
                             .alias(f"e1c_bb_width_{period}_{num_std}"))
                exprs.append(((upper_raw - lower_raw) / (sma + 1e-10) * 100)
                             .alias(f"e1c_bb_width_pct_{period}_{num_std}"))
                exprs.append(((pl.col("close") - sma) / (std + 1e-10))
                             .alias(f"e1c_bb_position_{period}_{num_std}"))

        # ---------------------------------------------------------
        # ADX/DI系 (3 stats × 3 periods = 9)
        # ---------------------------------------------------------
        for p in [13, 21, 34]:
            exprs.append(pl.col(f"__adx_{p}").alias(f"e1c_adx_{p}"))
            exprs.append(pl.col(f"__di_p_{p}").alias(f"e1c_di_plus_{p}"))
            exprs.append(pl.col(f"__di_m_{p}").alias(f"e1c_di_minus_{p}"))

        # ---------------------------------------------------------
        # Aroon系 (3 stats × 3 periods = 9)
        # ---------------------------------------------------------
        for p in [14, 25, 50]:
            exprs.append(pl.col(f"__aroon_u_{p}").alias(f"e1c_aroon_up_{p}"))
            exprs.append(pl.col(f"__aroon_d_{p}").alias(f"e1c_aroon_down_{p}"))
            exprs.append((pl.col(f"__aroon_u_{p}") - pl.col(f"__aroon_d_{p}"))
                         .alias(f"e1c_aroon_oscillator_{p}"))

        # ---------------------------------------------------------
        # Williams %R (3)
        # ---------------------------------------------------------
        for p in [14, 28, 56]:
            exprs.append(pl.col(f"__williams_{p}").alias(f"e1c_williams_r_{p}"))

        # ---------------------------------------------------------
        # Stochastic (3 stats × 3 configs = 9)
        # 参照: engine_1_C.create_oscillator_features (L1287-1326)
        # ---------------------------------------------------------
        for kp, dp, sp in [(14, 3, 3), (21, 5, 5), (9, 3, 3)]:
            stoch_k = pl.col(f"__stoch_k_{kp}_{dp}_{sp}")
            exprs.append(stoch_k.alias(f"e1c_stoch_k_{kp}"))
            stoch_d = stoch_k.rolling_mean(dp)
            exprs.append(stoch_d.alias(f"e1c_stoch_d_{kp}_{dp}"))
            slow_d = stoch_d.rolling_mean(sp)
            exprs.append(slow_d.alias(f"e1c_stoch_slow_d_{kp}_{dp}_{sp}"))

        # ---------------------------------------------------------
        # DPO (3)
        # 参照: engine_1_C.create_momentum_features (L1473-1488)
        # ---------------------------------------------------------
        for p in [20, 30, 50]:
            sma_dpo = pl.col("close").rolling_mean(p)
            dpo_raw = pl.col("close") - sma_dpo
            exprs.append((dpo_raw / atr_safe).alias(f"e1c_dpo_{p}"))

        # ---------------------------------------------------------
        # TRIX (3) / UO (1) / TSI (2)
        # ---------------------------------------------------------
        for p in [14, 20, 30]:
            exprs.append(pl.col(f"__trix_{p}").alias(f"e1c_trix_{p}"))
        exprs.append(pl.col("__uo").alias("e1c_ultimate_oscillator"))
        for p in [25, 13]:
            exprs.append(pl.col(f"__tsi_{p}").alias(f"e1c_tsi_{p}"))

        # ---------------------------------------------------------
        # Rate of Change (4) / Momentum (4)
        # 参照: engine_1_C.create_momentum_features (L1535-1559)
        # ---------------------------------------------------------
        for p in [10, 20, 30, 50]:
            roc = ((pl.col("close") - pl.col("close").shift(p))
                   / pl.col("close").shift(p) * 100)
            exprs.append(roc.alias(f"e1c_rate_of_change_{p}"))

        for p in [10, 20, 30, 50]:
            mom_raw = pl.col("close") - pl.col("close").shift(p)
            exprs.append((mom_raw / atr_safe).alias(f"e1c_momentum_{p}"))

        # ---------------------------------------------------------
        # KST (2: kst, kst_signal)
        # 参照: engine_1_C.create_advanced_features (L1573-1593)
        # ---------------------------------------------------------
        roc_10 = (pl.col("close") - pl.col("close").shift(10)) / pl.col("close").shift(10)
        roc_15 = (pl.col("close") - pl.col("close").shift(15)) / pl.col("close").shift(15)
        roc_20 = (pl.col("close") - pl.col("close").shift(20)) / pl.col("close").shift(20)
        roc_30 = (pl.col("close") - pl.col("close").shift(30)) / pl.col("close").shift(30)
        kst = (roc_10 * 1 + roc_15 * 2 + roc_20 * 3 + roc_30 * 4) / 10 * 100
        exprs.append(kst.alias("e1c_kst"))
        exprs.append(kst.rolling_mean(9).alias("e1c_kst_signal"))

        # ---------------------------------------------------------
        # Coppock Curve (1)
        # 参照: engine_1_C.create_advanced_features (L1644-1657)
        # ---------------------------------------------------------
        roc_11 = (pl.col("close") - pl.col("close").shift(11)) / pl.col("close").shift(11) * 100
        roc_14 = (pl.col("close") - pl.col("close").shift(14)) / pl.col("close").shift(14) * 100
        exprs.append((roc_11 + roc_14).rolling_mean(10).alias("e1c_coppock_curve"))

        # ---------------------------------------------------------
        # Schaff Trend Cycle (2)
        # 参照: engine_1_C.create_advanced_features (L1611-1640)
        #
        # 重要: 学習側は ewm_mean(half_life=fast_period, adjust=False) と
        # ewm_mean(span=3, adjust=False) を使用。Polars 直呼びで完全一致。
        # ---------------------------------------------------------
        for fast_period, slow_period_stc, cycle_period in [(23, 50, 10), (12, 26, 9)]:
            fast_ma = pl.col("close").ewm_mean(half_life=fast_period, adjust=False)
            slow_ma = pl.col("close").ewm_mean(half_life=slow_period_stc, adjust=False)
            macd_stc = fast_ma - slow_ma
            macd_min1 = macd_stc.rolling_min(cycle_period)
            macd_max1 = macd_stc.rolling_max(cycle_period)
            stoch_macd = ((macd_stc - macd_min1) / (macd_max1 - macd_min1 + 1e-10)) * 100
            stoch_macd_smoothed = stoch_macd.ewm_mean(span=3, adjust=False)
            stoch_min2 = stoch_macd_smoothed.rolling_min(cycle_period)
            stoch_max2 = stoch_macd_smoothed.rolling_max(cycle_period)
            stoch_stoch = ((stoch_macd_smoothed - stoch_min2) / (stoch_max2 - stoch_min2 + 1e-10)) * 100
            stc = stoch_stoch.ewm_mean(span=3, adjust=False)
            exprs.append(stc.alias(f"e1c_schaff_trend_cycle_{fast_period}_{slow_period_stc}_{cycle_period}"))

        # ---------------------------------------------------------
        # Price Oscillator (3)
        # 参照: engine_1_C.create_advanced_features (L1660-1667)
        # ---------------------------------------------------------
        for fast, slow in [(12, 26), (5, 35), (10, 20)]:
            ema_f = pl.col("close").ewm_mean(span=fast, adjust=False)
            ema_s = pl.col("close").ewm_mean(span=slow, adjust=False)
            exprs.append(((ema_f - ema_s) / ema_s * 100)
                         .alias(f"e1c_price_oscillator_{fast}_{slow}"))

        # ---------------------------------------------------------
        # RVI (relative_vigor_index/rvi_signal × 3 = 6)
        # 参照: engine_1_C.create_advanced_features (L1597-1608)
        # 注意: open/high/low が存在しないと計算不能 → layer2 でフォールバック
        # ---------------------------------------------------------
        rvi_available = (len(open_arr) > 0 and len(high_arr) > 0 and len(low_arr) > 0)
        if rvi_available:
            for p in [10, 14, 20]:
                num = (pl.col("close") - pl.col("open")).rolling_mean(p)
                den = (pl.col("high") - pl.col("low")).rolling_mean(p)
                rvi = num / (den + 1e-10)
                exprs.append(rvi.alias(f"e1c_relative_vigor_index_{p}"))
                exprs.append(rvi.rolling_mean(4).alias(f"e1c_rvi_signal_{p}"))

        # ---------------------------------------------------------
        # 移動平均: SMA + SMA_dev + EMA + EMA_dev + WMA × 5 periods = 25
        # 参照: engine_1_C.create_moving_average_features (L1693-1752)
        # ---------------------------------------------------------
        for p in [10, 20, 50, 100, 200]:
            sma_raw = pl.col("close").rolling_mean(p)
            exprs.append(((sma_raw - pl.col("close")) / atr_safe).alias(f"e1c_sma_{p}"))
            exprs.append(((pl.col("close") - sma_raw) / (sma_raw + 1e-10) * 100)
                         .alias(f"e1c_sma_deviation_{p}"))

            ema_raw = pl.col("close").ewm_mean(span=p, adjust=False)
            exprs.append(((ema_raw - pl.col("close")) / atr_safe).alias(f"e1c_ema_{p}"))
            exprs.append(((pl.col("close") - ema_raw) / (ema_raw + 1e-10) * 100)
                         .alias(f"e1c_ema_deviation_{p}"))

            exprs.append(((pl.col(f"__wma_{p}") - pl.col("close")) / atr_safe)
                         .alias(f"e1c_wma_{p}"))

        # ---------------------------------------------------------
        # HMA (3)
        # 参照: engine_1_C.create_moving_average_features (L1755-1771)
        # ---------------------------------------------------------
        for p in [21, 34, 55]:
            exprs.append(((pl.col(f"__hma_{p}") - pl.col("close")) / atr_safe)
                         .alias(f"e1c_hma_{p}"))

        # ---------------------------------------------------------
        # KAMA (2)
        # 参照: engine_1_C.create_moving_average_features (L1774-1793)
        # ---------------------------------------------------------
        for p in [21, 34]:
            exprs.append(((pl.col(f"__kama_{p}") - pl.col("close")) / atr_safe)
                         .alias(f"e1c_kama_{p}"))

        # ---------------------------------------------------------
        # Trend Slope/Strength/Consistency (3 stats × 3 periods = 9)
        # 参照: engine_1_C.create_moving_average_features (L1795-1844)
        # ---------------------------------------------------------
        for p in [20, 50, 100]:
            sma_for_slope = pl.col("close").rolling_mean(p)
            wma_for_slope = pl.col(f"__wma_{p}")  # 既に計算済み
            true_ols_slope = 6.0 * (wma_for_slope - sma_for_slope) / (p - 1.0)
            exprs.append((true_ols_slope / atr_safe).alias(f"e1c_trend_slope_{p}"))

            normalized_std = pl.col("close").rolling_std(p, ddof=1) / atr_safe
            exprs.append(
                (1.0 / (normalized_std + 1e-10)).clip(upper_bound=100.0)
                .alias(f"e1c_trend_strength_{p}")
            )

            direction_changes = pl.col("close").diff().sign().diff().abs()
            exprs.append(
                (1 - direction_changes.rolling_mean(p) / 2)
                .alias(f"e1c_trend_consistency_{p}")
            )

        # ===== layer2 =====
        # 1C は基本的に Polars で完結。RVI のみ open/high/low 不在時に
        # NaN フォールバックが必要。
        layer2: Dict[str, float] = {}
        if not rvi_available:
            for p in [10, 14, 20]:
                layer2[f"e1c_relative_vigor_index_{p}"] = np.nan
                layer2[f"e1c_rvi_signal_{p}"]          = np.nan

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
        columns, exprs, layer2 = FeatureModule1C._build_polars_pieces(data, lookback_bars)
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
