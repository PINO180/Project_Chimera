# realtime_feature_engine_1E_signal.py
# Category 1E: 信号処理系 (Spectral / Wavelet / Hilbert / Acoustic / Signal Stats)
#
# ==================================================================
# 【Phase 9b 改修】司令塔統合 .select() 対応 (FFI overhead 削減)
# ==================================================================
#
# 目的: Phase 9 (Step B) で達成した Polars 直呼び + DSP UDF 直接呼びの
#       2 層構造を保ったまま、6 モジュールの Polars 式を司令塔で 1 回の
#       .select() に統合できるよう構造を分解する。
#
# 【Phase 9b の改修】
#   追加: `_build_polars_pieces(data, lookback_bars) -> (columns, exprs, layer2)`
#     - columns: close + __temp_atr_13 + __temp_atr_100 (raw ATR、+1e-10 は割り算時)
#     - exprs:   Polars 式リスト (spectral_energy/peak_freq, wavelet_mean/std,
#                hilbert_amp_*, hilbert_freq_energy_ratio, signal_rms/peak_to_peak/crest_factor)
#     - layer2:  DSP Numba UDF 直接呼び結果 (spectral_centroid/bandwidth/rolloff/
#                flux/flatness/entropy, wavelet_energy/entropy, hilbert_amplitude/phase/freq,
#                acoustic_power/frequency) + e1e_sample_weight
#   変更: `calculate_features` は `_build_polars_pieces` を呼んで単独計算する
#         薄いラッパーへ。後方互換完全維持。
#
# 【1E の特徴】
#   1E は最初から Layer 1 (Polars rolling 統計) と Layer 2 (DSP UDF 直接呼び)
#   が明示的に分離されており、Phase 9b への分解がもっとも素直なモジュール。
#
#   DSP UDF は O(window²) の FFT 計算で重く、最後の window 本のみ渡せば最終バー
#   の値が決まるため、numpy 直接呼びが最適 (学習側 map_batches と最終バーで同値)。
#
# 【ATR の扱い】
#   学習側: pl.struct(...).map_batches(calculate_atr_wilder(..., 13/100))
#           → 割り算時に + 1e-10 を加える
#   本番側: numpy で事前計算して __temp_atr_{13,100} 列に raw 値を入れる
#           → 割り算時に Polars 式で `(pl.col("__temp_atr_13") + 1e-10)` を使う
#   結果: 学習側と完全同値の計算経路
#
# 【Phase E (stable_rolling SSoT) 適用】
#   Polars 組込 rolling_{mean,std} は内部 running 累積実装で context 長依存性
#   がある (学習側 3.4M bars と本番側 ~2980 bars deque で結果が乖離)。本ファイル
#   では全 rolling_{mean,std} 呼出しを Option C2 で置換:
#     - close_pct / abs_pct_close / pct_close_sq を numpy で先に計算
#     - 各 window の stable_rolling_{mean,std} を事前計算 → `__num_*` 列に注入
#     - expression 内では `pl.col("__num_...")` で参照のみ
#     - map_batches は stable_rolling 経由のものを完全排除 (CSE non-determinism 回避)
#   rolling_{max,min,sum,quantile} は context 長非依存のため変更なし
#   (spectral_energy の rolling_sum, peak_to_peak の rolling_max/min は維持)。
#
#   ⚠ verify 時のリスク注記:
#     E-pair 適用で rfe_1E の expression 構造が変化 (rolling_{mean,std} の
#     pct_change/abs/^2 ベース 14 個が `pl.col(__num_*)` 参照に置換、列 13 個
#     が新規追加)。Cluster A の機序 (= expression 集合の構造変化が Polars plan
#     / CSE 経路を切替) の観点では、他 engine (e1a/e1b/e1c/e1d/e1f) の結果が
#     変動する可能性はゼロではない。shadow_mode 検証時に e1e 以外の cells も
#     diff チェックすることを推奨。
#
# 【SSoT 階層】(§B.12.13.9 で再定義)
#   Layer 1 (経路): 学習側 = Polars map_batches、本番側 = numpy 直呼び + 列注入
#   Layer 2 (真の SSoT): stable_rolling.py (Numba 関数) + core_indicators.py
#
# 【保持される過去の修正】
#   ・QAState (apply_quality_assurance_to_group の等価実装、bias=False 補正)
#   ・hilbert_phase_*_udf は core_indicators の FFT-Hilbert 厳密実装を使用
#   ・e1e_sample_weight は学習側 base_columns 扱いで QA 対象外
# ==================================================================

import sys
from pathlib import Path

# -----------------------------------------------------------------------
# パス解決: blueprint → core_indicators の順で解決する
# -----------------------------------------------------------------------
_parent_dir = str(Path(__file__).resolve().parents[1])
if _parent_dir not in sys.path:
    sys.path.append(_parent_dir)

try:
    import blueprint as config
    _core_dir = str(config.CORE_DIR)
    if _core_dir not in sys.path:
        sys.path.append(_core_dir)
except ModuleNotFoundError:
    _fallback_core = str(Path(__file__).resolve().parent / "core")
    if _fallback_core not in sys.path:
        sys.path.append(_fallback_core)

from core_indicators import (
    # [ATR & VOLATILITY]
    calculate_atr_wilder,
    # [WEIGHT]
    calculate_sample_weight,
    # [DSP] — スペクトル系
    spectral_centroid_udf,
    spectral_bandwidth_udf,
    spectral_rolloff_udf,
    spectral_flux_udf,
    spectral_flatness_udf,
    spectral_entropy_udf,
    # [DSP] — ウェーブレット系
    wavelet_energy_udf,
    wavelet_entropy_udf,
    # [DSP] — ヒルベルト系
    hilbert_amplitude_udf,
    hilbert_phase_var_udf,
    hilbert_phase_stability_udf,
    hilbert_freq_mean_udf,
    hilbert_freq_std_udf,
    # [DSP] — 音響系
    acoustic_power_udf,
    acoustic_frequency_udf,
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
# 注: Plan §B.12.14.7 警告 6 で指摘された「rfe_1E _pct_change の else 句
# 欠落 bug」は Plan 起草時点の状態であり、現コードでは既に解消済 (Numba
# JIT 化と同時に else 句が追加された)。本 refactor は bug 修正ではなく
# canonical 1 本化が目的。
# ─────────────────────────────────────────────────────────────
from numpy_helpers import pct_change_polars_compat as _pct_change

import numpy as np
import polars as pl
import numba as nb
from typing import Dict, Optional, Tuple, List


# ==================================================================
# ヘルパー関数
# ==================================================================


# ==================================================================
# QAState — 学習側 apply_quality_assurance_to_group の等価実装
# (1A〜1D と完全に同一の実装。Phase 9b では変更なし。)
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
        p01 = ewm_mean - 5.0 * ewm_std
        p99 = ewm_mean + 5.0 * ewm_std

        if is_pos_inf:
            return float(p99) if np.isfinite(p99) else 0.0
        if is_neg_inf:
            return float(p01) if np.isfinite(p01) else 0.0
        if np.isnan(raw_val):
            return 0.0

        clipped = float(np.clip(raw_val, p01, p99))
        return clipped if np.isfinite(clipped) else 0.0


# ==================================================================
# メイン計算クラス
# ==================================================================

class FeatureModule1E:

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
                共通列 (close) + 1E 固有の __temp_atr_13 / __temp_atr_100 (raw)。
            exprs: List[pl.Expr]
                Polars rolling 統計式リスト (alias は e1e_* の最終特徴量名)。
                spectral_energy/peak_freq, wavelet_mean/std, hilbert_amp_*,
                hilbert_freq_energy_ratio, signal_rms/peak_to_peak/crest_factor。
            layer2: Dict[str, float]
                DSP UDF 直接呼び結果 (close_pct[-window:] に対する最終バー値)
                + e1e_sample_weight (QA対象外)。
        """
        close_arr = data["close"].astype(np.float64)
        if len(close_arr) == 0:
            return {}, [], {}

        high_arr  = (
            data["high"].astype(np.float64) if "high" in data and len(data["high"]) > 0
            else close_arr
        )
        low_arr   = (
            data["low"].astype(np.float64) if "low" in data and len(data["low"]) > 0
            else close_arr
        )

        # ---------------------------------------------------------
        # ATR 系列の事前計算 (学習側 atr_13_expr_hilbert / atr_100_expr と完全一致)
        # 学習側は割り算時に + 1e-10 を加えるため、ここでは raw ATR を保持する。
        # ---------------------------------------------------------
        atr13_arr  = calculate_atr_wilder(high_arr, low_arr, close_arr, 13)
        atr100_arr = calculate_atr_wilder(high_arr, low_arr, close_arr, 100)

        # close_pct を numpy で 1 度だけ計算 (学習側 Polars pct_change と semantics 一致)
        close_pct = _pct_change(close_arr)
        n = len(close_pct)

        # ===== columns =====
        columns: Dict[str, np.ndarray] = {
            "close":         close_arr,
            "__temp_atr_13":  atr13_arr,
            "__temp_atr_100": atr100_arr,
        }

        # ▼▼ [§B.12.13.7 Option C2 / Phase E] numpy 事前計算 + 列注入 ▼▼
        # Polars map_batches を経由しない (CSE non-determinism 回避)。
        # 命名規約:
        #   複合: __num_<func>_<expr_id>_<window>
        # 注: E では rolling_{mean,std} は全て close.pct_change() ベースで使用される
        #     ため、すべて複合パターン。close_pct (既に L280 で計算済) を中間配列として
        #     再利用する。
        #
        # NaN handling 注意: _pct_change は先頭 1 NaN を返す。stable_rolling_X は
        # window 内 NaN で出力 NaN。これは Polars `pct_change().rolling_X(W)` と
        # 同じ NaN 位置を生む (B 側で検証済)。

        # --- 中間配列: |pct_close|, pct_close^2 ---
        abs_pct_close = np.abs(close_pct).astype(np.float64)
        pct_close_sq = (close_pct ** 2).astype(np.float64)

        # --- pct_close 列注入 (Polars `close.pct_change()` の代替) ---
        # 用途: spectral_energy / spectral_peak_freq_128 分子 / signal_crest_factor_50
        #       分子 / hilbert_freq_energy_ratio_100 のように、rolling_{max,sum} の
        #       入力として Polars 経路で pct_change が必要な箇所。
        # close_pct は既に numpy で計算済 (_pct_change(close_arr))。これを列注入する
        # ことで、expression 内では `pl.col("close").pct_change()` の代わりに
        # `pl.col("__num_pct_close")` を使える。Polars の pct_change 計算が plan
        # から消えて、CSE 経路の不確実性をさらに減らせる。
        columns["__num_pct_close"] = close_pct.astype(np.float64)

        # --- Wavelet group: wavelet_mean_{W} / wavelet_std_{W} ---
        # 学習側: pct_change.rolling_mean(W) / pct_change.rolling_std(W, ddof=1)
        # window: [32, 64, 128, 256]
        # 注: spectral_peak_freq_128 分母も同じ `__num_srs_pct_close_128` を参照する
        #     (window=128, ddof=1 で完全一致するため列を共有 — CSE 不要で明示的共有)。
        for _w in [32, 64, 128, 256]:
            columns[f"__num_srm_pct_close_{_w}"] = stable_rolling_mean(close_pct.astype(np.float64), _w)
            columns[f"__num_srs_pct_close_{_w}"] = stable_rolling_std(close_pct.astype(np.float64), _w, 1)

        # --- Hilbert group: hilbert_amp_mean_100 / std_100 / cv_100 ---
        # 学習側: pct_change.abs().rolling_mean(100), pct_change.abs().rolling_std(100, ddof=1)
        columns["__num_srm_abs_pct_close_100"] = stable_rolling_mean(abs_pct_close, 100)
        columns["__num_srs_abs_pct_close_100"] = stable_rolling_std(abs_pct_close, 100, 1)

        # --- Signal Stats group: signal_rms_50, signal_crest_factor_50 分母 ---
        # 学習側: (pct_change ** 2).rolling_mean(50)
        # 用途: signal_rms_50 = sqrt(__num_srm_pct_close_sq_50)
        #       signal_crest_factor_50 分母 = sqrt(__num_srm_pct_close_sq_50) + 1e-10
        columns["__num_srm_pct_close_sq_50"] = stable_rolling_mean(pct_close_sq, 50)
        # ▲▲ [§B.12.13.7 Option C2 / Phase E] ▲▲

        # ===== exprs (Layer 1: Polars rolling 統計) =====
        # 学習側 engine_1_E のうち rolling 統計に該当する式を集約。
        exprs: List[pl.Expr] = []

        # ----- Spectral group (Polars 部分) -----
        # spectral_energy: (pct_change ** 2).rolling_sum(window)
        # 参照: engine_1_E L1166-1171
        # ▼▼ [Phase E #3 統一] pl.col("close").pct_change() → __num_pct_close 参照
        for window in [64, 128, 256, 512]:
            exprs.append(
                (pl.col("__num_pct_close") ** 2)
                .rolling_sum(window)
                .alias(f"e1e_spectral_energy_{window}")
            )

        # spectral_peak_freq_128: rolling_max / (rolling_std + 1e-10)
        # 参照: engine_1_E L1175-1180
        # ▼▼ [Phase E] pct_change.rolling_std(128) → __num_srs_pct_close_128 列参照
        exprs.append(
            (
                pl.col("__num_pct_close").rolling_max(128)
                / (pl.col("__num_srs_pct_close_128") + 1e-10)
            ).alias("e1e_spectral_peak_freq_128")
        )
        # ▲▲ [Phase E] ▲▲

        # ----- Wavelet group (Polars 部分) -----
        # wavelet_mean / wavelet_std (Polars-native rolling stats)
        # 参照: engine_1_E L1202-1215
        # ▼▼ [Phase E] pct_change.rolling_{mean,std}(W) → __num_srm/srs_pct_close_{W} 列参照
        for window in [32, 64, 128, 256]:
            exprs.append(
                pl.col(f"__num_srm_pct_close_{window}")
                .alias(f"e1e_wavelet_mean_{window}")
            )
            exprs.append(
                pl.col(f"__num_srs_pct_close_{window}")
                .alias(f"e1e_wavelet_std_{window}")
            )
        # ▲▲ [Phase E] ▲▲

        # ----- Hilbert group (Polars 部分) -----
        # hilbert_amp_mean_100 / std_100 / cv_100 (Polars-native rolling stats on |pct_change|)
        # 参照: engine_1_E L1252-1273
        # ▼▼ [Phase E] pct_change.abs().rolling_{mean,std}(100) → __num_srm/srs_abs_pct_close_100
        exprs.append(
            pl.col("__num_srm_abs_pct_close_100")
            .alias("e1e_hilbert_amp_mean_100")
        )
        exprs.append(
            pl.col("__num_srs_abs_pct_close_100")
            .alias("e1e_hilbert_amp_std_100")
        )
        exprs.append(
            (
                pl.col("__num_srs_abs_pct_close_100")
                / (pl.col("__num_srm_abs_pct_close_100") + 1e-10)
            ).alias("e1e_hilbert_amp_cv_100")
        )
        # ▲▲ [Phase E] ▲▲

        # hilbert_freq_energy_ratio_100:
        #   学習側: (close.pct_change()^2).rolling_sum(100) / ((atr_13/close)^2 * 100 + 1e-10)
        # 参照: engine_1_E L1335-1342
        # ▼▼ [Phase E #3 統一] pl.col("close").pct_change() → __num_pct_close 参照
        atr_13_pct_expr = pl.col("__temp_atr_13") / (pl.col("close") + 1e-10)
        exprs.append(
            (
                (pl.col("__num_pct_close") ** 2).rolling_sum(100)
                / (atr_13_pct_expr.pow(2) * 100 + 1e-10)
            ).alias("e1e_hilbert_freq_energy_ratio_100")
        )
        # ▲▲ [Phase E #3 統一] ▲▲

        # ----- Signal Stats group (Polars 部分) -----
        # signal_rms_50: sqrt(rolling_mean(pct_change^2, 50))
        # 参照: engine_1_E L1397-1402
        # ▼▼ [Phase E] (pct_change**2).rolling_mean(50) → __num_srm_pct_close_sq_50
        exprs.append(
            pl.col("__num_srm_pct_close_sq_50")
            .sqrt()
            .alias("e1e_signal_rms_50")
        )
        # ▲▲ [Phase E] ▲▲

        # signal_peak_to_peak_100: (close.rolling_max(100) - close.rolling_min(100)) / (atr_100 + 1e-10)
        # 参照: engine_1_E L1404-1410
        exprs.append(
            (
                (pl.col("close").rolling_max(100) - pl.col("close").rolling_min(100))
                / (pl.col("__temp_atr_100") + 1e-10)
            ).alias("e1e_signal_peak_to_peak_100")
        )

        # signal_crest_factor_50:
        #   学習側: pct_change.rolling_max(50).abs() / ((pct_change^2).rolling_mean(50).sqrt() + 1e-10)
        # 参照: engine_1_E L1412-1418
        # ▼▼ [Phase E] 分母 (pct_change**2).rolling_mean(50) → __num_srm_pct_close_sq_50 列参照
        # (signal_rms_50 と同じ列を共有 — CSE 不要で明示的共有)
        # ▼▼ [Phase E #3 統一] 分子 pl.col("close").pct_change() → __num_pct_close 参照
        exprs.append(
            (
                pl.col("__num_pct_close").rolling_max(50).abs()
                / (pl.col("__num_srm_pct_close_sq_50").sqrt() + 1e-10)
            ).alias("e1e_signal_crest_factor_50")
        )
        # ▲▲ [Phase E] ▲▲

        # ===== layer2 (Layer 2: DSP UDF 直接呼び + sample_weight) =====
        # 各 UDF は rolling 計算であり、最終バー (index = window-1 in slice) の値は
        # 直近 window 本のみで決まる。学習側は全系列に対して UDF を呼び、その最終
        # 要素を採用するが、本番側は最終 window 本のみ渡しても同一値。
        layer2: Dict[str, float] = {}

        # ----- Spectral UDFs (window=[64,128,256,512]) -----
        # 参照: engine_1_E._create_spectral_features L1098-1163
        for window in [64, 128, 256, 512]:
            if n >= window:
                w_arr = close_pct[-window:]
                layer2[f"e1e_spectral_centroid_{window}"]  = float(spectral_centroid_udf(w_arr, window)[-1])
                layer2[f"e1e_spectral_bandwidth_{window}"] = float(spectral_bandwidth_udf(w_arr, window)[-1])
                layer2[f"e1e_spectral_rolloff_{window}"]   = float(spectral_rolloff_udf(w_arr, window)[-1])
                layer2[f"e1e_spectral_flatness_{window}"]  = float(spectral_flatness_udf(w_arr, window)[-1])
                layer2[f"e1e_spectral_entropy_{window}"]   = float(spectral_entropy_udf(w_arr, window)[-1])
            else:
                layer2[f"e1e_spectral_centroid_{window}"]  = np.nan
                layer2[f"e1e_spectral_bandwidth_{window}"] = np.nan
                layer2[f"e1e_spectral_rolloff_{window}"]   = np.nan
                layer2[f"e1e_spectral_flatness_{window}"]  = np.nan
                layer2[f"e1e_spectral_entropy_{window}"]   = np.nan

            # spectral_flux は隣接 2 フレーム必要 (window*2 本)
            if n >= window * 2:
                w_arr2 = close_pct[-(window * 2):]
                layer2[f"e1e_spectral_flux_{window}"] = float(spectral_flux_udf(w_arr2, window)[-1])
            else:
                layer2[f"e1e_spectral_flux_{window}"] = np.nan

        # ----- Wavelet UDFs -----
        # 参照: engine_1_E._create_wavelet_features L1190-1227
        for window in [32, 64, 128, 256]:
            if n >= window:
                layer2[f"e1e_wavelet_energy_{window}"] = float(
                    wavelet_energy_udf(close_pct[-window:], window)[-1]
                )
            else:
                layer2[f"e1e_wavelet_energy_{window}"] = np.nan

        # wavelet_entropy_64
        if n >= 64:
            layer2["e1e_wavelet_entropy_64"] = float(
                wavelet_entropy_udf(close_pct[-64:], 64)[-1]
            )
        else:
            layer2["e1e_wavelet_entropy_64"] = np.nan

        # ----- Hilbert UDFs -----
        # 参照: engine_1_E._create_hilbert_features L1237-1296
        for window in [50, 100, 200]:
            if n >= window:
                layer2[f"e1e_hilbert_amplitude_{window}"] = float(
                    hilbert_amplitude_udf(close_pct[-window:], window)[-1]
                )
            else:
                layer2[f"e1e_hilbert_amplitude_{window}"] = np.nan

        # phase_var_50, phase_stability_50
        if n >= 50:
            layer2["e1e_hilbert_phase_var_50"]       = float(hilbert_phase_var_udf(close_pct[-50:], 50)[-1])
            layer2["e1e_hilbert_phase_stability_50"] = float(hilbert_phase_stability_udf(close_pct[-50:], 50)[-1])
        else:
            layer2["e1e_hilbert_phase_var_50"]       = np.nan
            layer2["e1e_hilbert_phase_stability_50"] = np.nan

        # freq_mean_100, freq_std_100
        if n >= 100:
            layer2["e1e_hilbert_freq_mean_100"] = float(hilbert_freq_mean_udf(close_pct[-100:], 100)[-1])
            layer2["e1e_hilbert_freq_std_100"]  = float(hilbert_freq_std_udf(close_pct[-100:], 100)[-1])
        else:
            layer2["e1e_hilbert_freq_mean_100"] = np.nan
            layer2["e1e_hilbert_freq_std_100"]  = np.nan

        # ----- Acoustic UDFs (window=[128,256,512]) -----
        # 参照: engine_1_E._create_acoustic_features L1352-1373
        for window in [128, 256, 512]:
            if n >= window:
                w_arr = close_pct[-window:]
                layer2[f"e1e_acoustic_power_{window}"]     = float(acoustic_power_udf(w_arr, window)[-1])
                layer2[f"e1e_acoustic_frequency_{window}"] = float(acoustic_frequency_udf(w_arr, window)[-1])
            else:
                layer2[f"e1e_acoustic_power_{window}"]     = np.nan
                layer2[f"e1e_acoustic_frequency_{window}"] = np.nan

        # ----- サンプルウェイト (学習側 base_columns 扱いと一致、QA対象外) -----
        # 参照: engine_1_E L1733-1742
        sample_weight_arr = calculate_sample_weight(high_arr, low_arr, close_arr)
        layer2["e1e_sample_weight"] = (
            float(sample_weight_arr[-1]) if len(sample_weight_arr) > 0 else 1.0
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
        司令塔は _build_polars_pieces を直接呼んで全モジュール統合 .select() を
        行うが、本メソッドは後方互換のためモジュール単独で動作する形を維持する。
        """
        columns, exprs, layer2 = FeatureModule1E._build_polars_pieces(data, lookback_bars)
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
        # e1e_sample_weight は学習側 base_columns 扱いで QA 対象外
        if qa_state is not None:
            qa_result: Dict[str, float] = {}
            for key, val in features.items():
                if key == "e1e_sample_weight":
                    qa_result[key] = val
                else:
                    qa_result[key] = qa_state.update_and_clip(key, val)
            features = qa_result
        else:
            for key in list(features.keys()):
                if key != "e1e_sample_weight" and not np.isfinite(features[key]):
                    features[key] = 0.0

        return features
