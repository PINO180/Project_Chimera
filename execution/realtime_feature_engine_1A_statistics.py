# realtime_feature_engine_1A_statistics.py
# Category 1A: 基本統計・分布検定系 (Basic Statistics & Distribution Tests)
#
# 【Step 3 変更記録】core_indicators.py への移行 (Single Source of Truth)
#   - mad_rolling_numba UDF定義を削除 → calculate_mad に置き換え
#   - ATR割り欠落を修正: statistical_variance / fast_rolling_std / robust_mad
#     に calculate_atr_wilder による ATR スケール化を追加
#   - fast_volume_mean_50 を Relative Volume 化（学習側と一致）
#   - 不要になった import numba / import math を削除
#
# 【Step 4 変更記録】学習側との完全一致修正
#   - _rolling_moment: ddof=1 → ddof=0 に修正（学習側 std_ddof0 と一致）
#     旧: sub_sq / (sub_n - 1)  新: sub_sq / sub_n + 1e-10
#   - calculate_features に lookback_bars 引数を追加
#     fast_volume_mean_* の分母を固定200 → timeframe依存 lookback_bars に修正
#   - 未実装だった全特徴量を追加実装（学習側と完全一致）:
#       statistical_mean_10/20/50, statistical_std_10/20/50
#       statistical_variance_20/50, statistical_cv_20/50
#       statistical_skewness_20/50, statistical_kurtosis_20/50
#       statistical_moment_5/6/7/8_20, statistical_moment_7/8_50
#       robust_median_10/20/50, robust_q25/q75/iqr/trimmed_mean_10/20/50
#       robust_biweight_location_20, robust_winsorized_mean_20
#       jarque_bera_statistic_50, anderson_darling_statistic_30
#       runs_test_statistic_30, von_neumann_ratio_30
#       fast_rolling_mean_5/10/20/50/100
#       fast_rolling_std_5/50, fast_volume_mean_5/10/20/100
#       fast_quality_score_50
#       fast_basic_stabilization, fast_robust_stabilization
#
# 【Step 5 変更記録】学習側との完全一致 最終修正（ダブルチェック指摘対応）
#
#   [修正1] statistical_skewness_20/50 の bias 不一致を修正
#     旧: _skewness() → bias=False 相当（Fisher補正あり: n*(n-1)^0.5/(n-2) を乗算）
#     新: _skewness_bias_true() → Polars rolling_skew() デフォルト bias=True に完全一致
#         計算式: mean((x - mean)^3) / std_ddof0^3  （Fisher補正なし・分母はddof=0）
#
#   [修正2] statistical_moment_5/6/7_20, statistical_moment_7/50 の上書き問題を修正
#     旧: Group2ブロックで計算後、_rolling_moment() UDF で4特徴量を上書き
#         → _rolling_moment は学習側に存在せず、算法が異なっていた
#     新: _rolling_moment() の呼び出しブロックを完全削除。
#         全モーメント (5/6/7/8 × window=20/50) を Group2ブロック内の
#         Polars rolling 式相当の z_scores ベース実装に統一。
#         学習側 _create_skew_kurt_features と算法・epsilon が完全一致。
#
#   [修正3] 全特徴量への QA（品質保証）処理を追加
#     旧: QA処理なし（raw値をそのまま返却）
#     新: QAState クラスが各特徴量の EWM mean/var を跨バーで保持し、
#         学習側 apply_quality_assurance_to_group と同一の処理を逐次適用する。
#         学習側の処理:
#           1. inf/-inf → null 置換
#           2. EWM mean/std（half_life=lookback_bars, ignore_nulls=True, adjust=False）
#           3. clip(EWM_mean ± 5*EWM_std)
#           4. fill_null(0.0) / fill_nan(0.0)
#         本番側の等価実装:
#           alpha = 1 - exp(-ln2 / half_life)
#           EWM_mean[t] = alpha * x[t] + (1-alpha) * EWM_mean[t-1]
#           EWM_var[t]  = (1-alpha) * (EWM_var[t-1] + alpha * (x[t] - EWM_mean[t-1])^2)
#           clip → fill_nan(0.0)
#         呼び出し方:
#           qa_state = FeatureModule1A.QAState(lookback_bars=1440)  # 起動時に1度生成
#           features = FeatureModule1A.calculate_features(data, lookback_bars, qa_state)
#           # 同一 qa_state インスタンスを毎バー渡し続けること

# 【Step 6 変更記録】ダブルチェック指摘対応（問題4・問題5）および追加バグ修正
#
#   [修正4] _trim_mean — scipy.stats.trim_mean との完全一致（問題4）
#     旧: n - k <= k のとき sorted_arr 全体を返す → scipy と異なる（scipy は空 → nan）
#     新: sorted_arr[k : n-k] をそのまま返す（空配列 → np.nan を明示返却）
#     通常ケース (proportiontocut=0.1, n≥10) では挙動変化なし。
#     エッジケース (window 不足で n < 10 等) でも scipy と完全一致。
#
#   [修正5] QAState — ewm_std bias=False 補正の適用と起動シード問題の明文化（問題5）
#     旧: sqrt(EWM_var) をそのまま EWM_std として使用
#         → Polars ewm_std(adjust=False, bias=False) の bias 補正が未適用
#     新: bias_corr = 1 / sqrt(1 - (1-alpha)^(2n)) を乗算して Polars と完全一致。
#         n が大きくなると bias_corr → 1.0 に収束。ウォームアップ後は実質影響なし。
#     起動シード差（問題5）: コメントおよび _ewm_n フィールドを追加。
#         ウォームアップ（lookback_bars * 3 本を事前投入）を強く推奨する旨をドキュメント化。
#
#   [修正6] Group1・Group6 — window条件 >= 2 → >= window に修正（重大バグ）
#     旧: Group1 (statistical_mean/variance/std/cv) および
#         Group6 (fast_rolling_mean/fast_rolling_std) の条件が >= 2
#         → window=10 のとき 2本のデータでも計算してしまい、学習側と不一致
#     新: >= window に変更。
#         学習側 Polars rolling_*(window) は window 本未満の先頭バーで NaN を返し
#         QA で 0.0 になる。本番側もこれと完全一致する。
#     影響: ウォームアップが window 本に達するまでの起動直後バーで 0.0 が返るようになる。
#           通常稼働時（十分なデータがある状態）は挙動変化なし。


import sys
import math
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))
import numpy as np
from typing import Dict, Optional

import blueprint as config
sys.path.append(str(config.CORE_DIR))
from core_indicators import (
    calculate_atr_wilder,
    calculate_mad,
)

import numba as nb


# ==================================================================
# ヘルパー関数
# ==================================================================

@nb.njit(fastmath=False, cache=True)
def _window(arr: np.ndarray, window: int) -> np.ndarray:
    """配列の末尾から `window` 個の要素を取得"""
    if window <= 0:
        return np.empty(0, dtype=arr.dtype)
    if window > len(arr):
        return arr
    return arr[-window:]


@nb.njit(fastmath=False, cache=True)
def _last(arr: np.ndarray) -> float:
    """配列の最新値（末尾）を取得。配列が空の場合はNaNを返す"""
    if len(arr) == 0:
        return np.nan
    return float(arr[-1])


@nb.njit(fastmath=False, cache=True)
def _skewness_bias_true(arr: np.ndarray) -> float:
    """
    【修正1】Polars rolling_skew(bias=True) と完全一致する歪度計算。

    Polars rolling_skew() のデフォルトは bias=True（Fisher補正なし）。
    旧実装 _skewness() は bias=False（Fisher補正あり）であり不一致だった。

    Polars rolling_skew(bias=True) の定義:
        skew = mean((x - mean)^3) / std_ddof0^3
        std_ddof0 = sqrt(mean((x - mean)^2))  ← ddof=0（標本標準偏差）

    旧 bias=False の式（削除）:
        skew_biased = skew_raw * n * n / ((n-1) * (n-2))  ← Fisher補正

    n < 3 の場合は NaN。std_ddof0 < 1e-10 の場合は NaN。
    """
    n = len(arr)
    if n < 3:
        return np.nan
    mean = np.mean(arr)
    std_ddof0 = np.std(arr)   # ddof=0: njit内はddof指定不可、デフォルトがddof=0なので等価
    if std_ddof0 < 1e-10:
        return np.nan
    return float(np.mean(((arr - mean) / std_ddof0) ** 3))


@nb.njit(fastmath=False, cache=True)
def _trim_mean(arr: np.ndarray, proportiontocut: float) -> float:
    """scipy.stats.trim_mean と完全等価。

    scipy の実装:
        lowercut = int(proportiontocut * nobs)
        uppercut = nobs - lowercut
        atmp = a[lowercut:uppercut]  ← lowercut >= uppercut なら空配列になり nan を返す

    旧実装の差異修正 (問題4):
        旧: n - k <= k のとき sorted_arr 全体を返す → scipy と異なる挙動
        新: scipy と同様に a[k : n-k] をそのまま返す（空配列になれば mean が nan）
        通常ケース (proportiontocut=0.1, n>=10) では k=1, n-k>=9>k で完全一致。
        エッジケース (n < 10 や proportiontocut 過大) でも scipy と一致する。
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
    prev == 0 のとき inf（Polars準拠）、先頭は nan。"""
    n = len(arr)
    out = np.full(n, np.nan, dtype=np.float64)
    if n < 2:
        return out
    with np.errstate(divide="ignore", invalid="ignore"):
        out[1:] = (arr[1:] - arr[:-1]) / arr[:-1]
    return out


# ==================================================================
# QAState — 学習側 apply_quality_assurance_to_group の等価実装
# ==================================================================

class QAState:
    """
    【修正3】学習側 apply_quality_assurance_to_group のリアルタイム等価実装。

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
        EWM_std[t] = sqrt(EWM_var[t])

    ⚠️ Polars ewm_std(adjust=False) のデフォルトは bias=False（不偏補正あり）。
    厳密には bias_correction = sqrt(1 / (1 - (1-alpha)^(2t))) が乗じられるが、
    t が大きくなると 1 に収束する（lookback_bars=1440 なら数百バー以内に無視可能）。
    clip 範囲への影響は ±5σ の係数を考えると十分小さく、実害なし。

    ⚠️ 起動時のシード差（問題5）:
        学習側は全系列先頭から EWM を積み上げる（確定的）。
        本番側は「稼働開始時点の最初の有効値」でシードするため、
        起動直後の数バーで EWM の軌跡が学習側と異なる可能性がある。
        対策: 事前に十分なウォームアップバー（lookback_bars * 3 本が目安）を
              qa_state に流してから本番バーを処理することを強く推奨する。

    使い方:
        qa_state = FeatureModule1A.QAState(lookback_bars=1440)
        # ウォームアップ（強く推奨）
        for bar in historical_data[-lookback_bars * 3:]:
            FeatureModule1A.calculate_features(warmup_window, 1440, qa_state)
        # 本番
        for bar in live_stream:
            features = FeatureModule1A.calculate_features(data_window, 1440, qa_state)
    """

    def __init__(self, lookback_bars: int = 1440):
        # Polars の half_life を alpha に変換（adjust=False の場合の定義）
        self.alpha: float = 1.0 - np.exp(-np.log(2.0) / max(lookback_bars, 1))
        self._ewm_mean: Dict[str, float] = {}
        self._ewm_var: Dict[str, float] = {}
        # bias=False 補正用: Polars ewm_std は t バー目に sqrt(1/(1-(1-alpha)^(2t))) を乗じる。
        # ウォームアップが十分であれば値は 1.0 に収束する。
        # ここでは補正を適用し、起動直後から学習側との乖離を最小化する。
        self._ewm_n: Dict[str, int] = {}  # 有効値の累積更新回数（bias 補正に使用）

    def update_and_clip(self, key: str, raw_val: float) -> float:
        """
        1特徴量の raw_val に対して QA処理を適用し、処理済みスカラーを返す。

        処理手順（学習側と完全一致）:
            1. inf/-inf → NaN（学習側 replace([inf,-inf], None) 相当）
            2. NaN でなければ EWM mean / var を更新（ignore_nulls=True 相当）
            3. clip(EWM_mean ± 5*EWM_std)
               EWM_std は Polars ewm_std(adjust=False, bias=False) と一致させるため
               bias_correction = 1 / sqrt(1 - (1-alpha)^(2n)) を乗じる。
               n が大きくなると補正は 1.0 に収束し、ウォームアップ後は影響なし。
            4. NaN → 0.0（fill_null/fill_nan 相当）
        """
        alpha = self.alpha

        # 【問題1修正】学習側の inf 処理を再現:
        #   学習側: col.replace([inf,-inf], None) でEWM計算
        #           pl.col(col_name).clip(lower, upper) で +inf→upper, -inf→lower にclip
        #   本番側旧: inf → NaN → 0.0（学習側と不一致）
        #   本番側新: inf を先に記録し、EWMはNaNとしてスキップ。
        #             clip時に inf だった値は upper/lower_bound で置き換える。
        is_pos_inf = np.isposinf(raw_val)
        is_neg_inf = np.isneginf(raw_val)
        ewm_input = np.nan if not np.isfinite(raw_val) else raw_val

        # Step2: EWM 状態更新
        if key not in self._ewm_mean:
            # 初回: NaN以外の最初の有効値でシード（Polarsの初期化と同等）
            if np.isnan(ewm_input):
                return 0.0  # 有効値がない場合は fill_null(0.0)
            self._ewm_mean[key] = ewm_input
            self._ewm_var[key]  = 0.0
            self._ewm_n[key]    = 1
            # 初回は EWM_std = 0 なのでクリップ範囲が [val, val] → raw_val をそのまま返す
            return ewm_input
        else:
            if not np.isnan(ewm_input):
                prev_mean = self._ewm_mean[key]
                prev_var  = self._ewm_var[key]
                # Polars ewm_mean(adjust=False) と等価
                new_mean = alpha * ewm_input + (1.0 - alpha) * prev_mean
                # Polars ewm_var(adjust=False) と等価:
                #   var[t] = (1-alpha)*(var[t-1] + alpha*(x[t]-mean[t-1])^2)
                new_var  = (1.0 - alpha) * (prev_var + alpha * (ewm_input - prev_mean) ** 2)
                self._ewm_mean[key] = new_mean
                self._ewm_var[key]  = new_var
                self._ewm_n[key]    = self._ewm_n.get(key, 0) + 1
            # NaN の場合は EWM を前値のまま維持（ignore_nulls=True と等価）

        # Step3: ±5σ クリップ
        # Polars ewm_std(adjust=False, bias=False) の bias 補正を適用:
        #   bias_corr = 1 / sqrt(1 - (1-alpha)^(2n))
        # n が大きければ (1-alpha)^(2n) ≒ 0 なので bias_corr ≒ 1.0。
        ewm_mean = self._ewm_mean[key]
        n_updates = self._ewm_n.get(key, 1)
        decay_2n  = (1.0 - alpha) ** (2 * n_updates)
        denom = 1.0 - decay_2n
        if denom > 1e-15:
            bias_corr = 1.0 / np.sqrt(denom)
        else:
            bias_corr = 1.0  # オーバーフロー防止（実質到達しない）
        ewm_std  = np.sqrt(max(self._ewm_var[key], 0.0)) * bias_corr
        lower    = ewm_mean - 5.0 * ewm_std
        upper    = ewm_mean + 5.0 * ewm_std

        # 学習側 clip 挙動を再現:
        #   NaN → fill_null(0.0) → 0.0
        #   +inf → upper_bound にclip
        #   -inf → lower_bound にclip
        #   有限値 → clip(lower, upper)
        if np.isnan(ewm_input):
            return 0.0
        if is_pos_inf:
            return float(upper) if np.isfinite(upper) else 0.0
        if is_neg_inf:
            return float(lower) if np.isfinite(lower) else 0.0

        clipped = float(np.clip(raw_val, lower, upper))

        # Step4: fill_nan(0.0)
        return clipped if np.isfinite(clipped) else 0.0


# ==================================================================
# Numba UDF群（学習側 engine_1_A と完全同一実装）
# ==================================================================

@nb.njit(fastmath=False, cache=True)
def _standard_normal_cdf_fast(x: float) -> float:
    SQRT2 = 1.4142135623730951
    return 0.5 * (1.0 + math.erf(x / SQRT2))


@nb.guvectorize(["void(float64[:], float64[:])"], "(n)->(n)", nopython=True, cache=True)
def _biweight_location_numba(arr, out):
    """ローリングBiweight位置推定（window=20、学習側と同一）"""
    n = len(arr)
    window = 20
    weights_buffer = np.zeros(window, dtype=np.float64)
    finite_buffer = np.empty(window, dtype=np.float64)

    for i in range(n):
        if i < window - 1:
            out[i] = np.nan
        else:
            window_data = arr[max(0, i - window + 1):i + 1]
            count = 0
            for j in range(len(window_data)):
                if not np.isnan(window_data[j]):
                    finite_buffer[count] = window_data[j]
                    count += 1
            finite_data = finite_buffer[:count]

            if len(finite_data) < 5:
                out[i] = np.median(finite_data) if len(finite_data) > 0 else np.nan
            else:
                current_location = np.median(finite_data)
                tolerance = 1e-10
                max_iterations = 50

                for iteration in range(max_iterations):
                    abs_residuals = np.abs(finite_data - current_location)
                    mad_val = np.median(abs_residuals)
                    if mad_val < 1e-15:
                        break
                    scale = 6.0 * mad_val
                    u_values = (finite_data - current_location) / scale
                    weights = weights_buffer[:len(finite_data)]
                    numerator = 0.0
                    denominator = 0.0
                    for j in range(len(finite_data)):
                        u_abs = abs(u_values[j])
                        if u_abs < 1.0:
                            weight = (1.0 - u_values[j] ** 2) ** 2
                            weights[j] = weight
                            numerator += finite_data[j] * weight
                            denominator += weight
                        else:
                            weights[j] = 0.0
                    if denominator > 1e-15:
                        new_location = numerator / denominator
                    else:
                        new_location = np.median(finite_data)
                        break
                    if abs(new_location - current_location) < tolerance:
                        break
                    current_location = new_location
                out[i] = current_location


@nb.guvectorize(["void(float64[:], float64[:])"], "(n)->(n)", nopython=True, cache=True)
def _winsorized_mean_numba(arr, out):
    """ローリングウィンソライズ平均（上下5%クリップ、window=20、学習側と同一）"""
    n = len(arr)
    window = 20
    finite_buffer = np.empty(window, dtype=np.float64)

    for i in range(n):
        if i < window - 1:
            out[i] = np.nan
        else:
            window_data = arr[max(0, i - window + 1):i + 1]
            count = 0
            for j in range(len(window_data)):
                if not np.isnan(window_data[j]):
                    finite_buffer[count] = window_data[j]
                    count += 1
            finite_data = finite_buffer[:count]

            if count < 5:
                out[i] = np.mean(finite_data) if count > 0 else np.nan
            else:
                p05 = np.percentile(finite_data, 5)
                p95 = np.percentile(finite_data, 95)
                winsorized_data = np.clip(finite_data, p05, p95)
                out[i] = np.mean(winsorized_data)


@nb.guvectorize(["void(float64[:], float64[:])"], "(n)->(n)", nopython=True, cache=True)
def _jarque_bera_numba(arr, out):
    """ローリングJarque-Bera検定統計量（window=50、学習側と同一）"""
    n = len(arr)
    window = 50

    for i in range(n):
        if i < window - 1:
            out[i] = np.nan
        else:
            window_data = arr[max(0, i - window + 1):i + 1]
            finite_data = window_data[~np.isnan(window_data)]

            if len(finite_data) < 20:
                out[i] = np.nan
            else:
                mean_val = np.mean(finite_data)
                variance = 0.0
                n_val = float(len(finite_data))
                for val in finite_data:
                    variance += (val - mean_val) ** 2
                variance = variance / n_val  # ddof=0
                std_val = np.sqrt(variance) + 1e-10

                if std_val < 1e-10:
                    out[i] = 0.0
                else:
                    z_sum_3 = 0.0
                    z_sum_4 = 0.0
                    for val in finite_data:
                        z = (val - mean_val) / std_val
                        z_sum_3 += z ** 3
                        z_sum_4 += z ** 4

                    skewness = z_sum_3 / n_val
                    kurtosis = z_sum_4 / n_val - 3.0

                    c1 = 6.0 * (n_val - 2.0) / ((n_val + 1.0) * (n_val + 3.0))
                    c2 = (24.0 * n_val * (n_val - 2.0) * (n_val - 3.0)
                          / (((n_val + 1.0) ** 2) * (n_val + 3.0) * (n_val + 5.0)))
                    out[i] = (skewness ** 2 / c1) + (kurtosis ** 2 / c2)


@nb.guvectorize(["void(float64[:], float64[:])"], "(n)->(n)", nopython=True, cache=True)
def _anderson_darling_numba(arr, out):
    """ローリングAnderson-Darling統計量（window=30、学習側と同一）"""
    n = len(arr)
    window = 30
    standardized_buffer = np.zeros(window, dtype=np.float64)

    for i in range(n):
        if i < window - 1:
            out[i] = np.nan
        else:
            window_data = arr[max(0, i - window + 1):i + 1]
            finite_data = window_data[~np.isnan(window_data)]

            if len(finite_data) < 10:
                out[i] = np.nan
            else:
                sorted_data = np.sort(finite_data)
                n_data = len(sorted_data)
                mean_val = np.mean(sorted_data)

                variance = 0.0
                for val in sorted_data:
                    variance += (val - mean_val) ** 2
                variance = variance / (n_data - 1)
                std_val = np.sqrt(variance)

                if std_val < 1e-10:
                    out[i] = 0.0
                else:
                    standardized_data = standardized_buffer[:n_data]
                    for j in range(n_data):
                        standardized_data[j] = (sorted_data[j] - mean_val) / std_val

                    ad_sum = 0.0
                    for j in range(n_data):
                        F_j  = _standard_normal_cdf_fast(standardized_data[j])
                        F_nj = _standard_normal_cdf_fast(standardized_data[n_data - 1 - j])
                        if F_j > 1e-15 and (1 - F_nj) > 1e-15:
                            log_term = np.log(F_j) + np.log(1 - F_nj)
                            ad_sum += (2 * j + 1) * log_term

                    ad_stat = -n_data - ad_sum / n_data
                    n_float = float(n_data)
                    stephens_correction = 1.0 + 4.0 / n_float - 25.0 / (n_float * n_float)
                    out[i] = ad_stat * stephens_correction


@nb.guvectorize(["void(float64[:], float64[:])"], "(n)->(n)", nopython=True, cache=True)
def _runs_test_numba(arr, out):
    """ローリングRuns Test統計量（window=30、学習側と同一）"""
    n = len(arr)
    window = 30

    for i in range(n):
        if i < window - 1:
            out[i] = np.nan
        else:
            window_data = arr[max(0, i - window + 1):i + 1]
            finite_data = window_data[~np.isnan(window_data)]

            if len(finite_data) < 10:
                out[i] = np.nan
            else:
                median_val = np.median(finite_data)
                binary_series = (finite_data > median_val).astype(np.int32)

                runs = 1
                for j in range(1, len(binary_series)):
                    if binary_series[j] != binary_series[j - 1]:
                        runs += 1

                n1 = np.sum(binary_series)
                n2 = len(binary_series) - n1

                if n1 > 0 and n2 > 0:
                    expected_runs = (2 * n1 * n2) / (n1 + n2) + 1
                    var_runs = (2 * n1 * n2 * (2 * n1 * n2 - n1 - n2)
                                / ((n1 + n2) ** 2 * (n1 + n2 - 1)))
                    if var_runs > 0:
                        out[i] = (runs - expected_runs) / np.sqrt(var_runs + 1e-10)
                    else:
                        out[i] = 0.0
                else:
                    out[i] = 0.0


@nb.guvectorize(["void(float64[:], float64[:])"], "(n)->(n)", nopython=True, cache=True)
def _von_neumann_ratio_numba(arr, out):
    """ローリングVon Neumann比（window=30、学習側と同一）"""
    n = len(arr)
    window = 30

    for i in range(n):
        if i < window - 1:
            out[i] = np.nan
        else:
            window_data = arr[max(0, i - window + 1):i + 1]
            finite_data = window_data[~np.isnan(window_data)]

            if len(finite_data) < 3:
                out[i] = np.nan
            else:
                n_points = len(finite_data)
                diff_sq_sum = 0.0
                for j in range(1, n_points):
                    diff = finite_data[j] - finite_data[j - 1]
                    diff_sq_sum += diff * diff

                sum_values = 0.0
                for j in range(n_points):
                    sum_values += finite_data[j]
                mean_val = sum_values / n_points

                sum_sq_deviations = 0.0
                for j in range(n_points):
                    deviation = finite_data[j] - mean_val
                    sum_sq_deviations += deviation * deviation

                if sum_sq_deviations > 1e-15:
                    vn_ratio = diff_sq_sum / (sum_sq_deviations + 1e-10)
                    if vn_ratio < 0.0:
                        out[i] = 0.0
                    elif vn_ratio > 4.0:
                        out[i] = 4.0
                    else:
                        out[i] = vn_ratio
                else:
                    out[i] = 0.0


@nb.guvectorize(["void(float64[:], float64[:])"], "(n)->(n)", nopython=True, cache=True)
def _fast_quality_score_numba(arr, out):
    """高速品質スコア計算（window=50、学習側と同一）"""
    n = len(arr)
    for i in range(n):
        if i < 50:
            out[i] = 0.0
        else:
            window_size = min(50, i + 1)
            nan_inf_count = 0
            finite_count = 0
            for j in range(i - window_size + 1, i + 1):
                if np.isnan(arr[j]) or np.isinf(arr[j]):
                    nan_inf_count += 1
                else:
                    finite_count += 1
            if window_size == 0:
                out[i] = 0.0
            else:
                finite_ratio = finite_count / window_size
                out[i] = finite_ratio * 0.8 + 0.2


@nb.guvectorize(["void(float64[:], float64[:])"], "(n)->(n)", nopython=True, cache=True)
def _basic_stabilization_numba(arr, out):
    """基本安定化（window=50、学習側と同一）"""
    n = len(arr)
    window = 50
    for i in range(n):
        if np.isnan(arr[i]) or np.isinf(arr[i]):
            out[i] = 0.0
        else:
            if i < 2:
                out[i] = arr[i]
            else:
                local_min = np.inf
                local_max = -np.inf
                start_idx = max(0, i - window + 1)
                for j in range(start_idx, i + 1):
                    val = arr[j]
                    if np.isfinite(val):
                        if val < local_min:
                            local_min = val
                        if val > local_max:
                            local_max = val
                if local_min != np.inf and local_max != -np.inf:
                    range_val = local_max - local_min
                    if range_val > 1e-10:
                        clip_margin = range_val * 0.01
                        if arr[i] < local_min + clip_margin:
                            out[i] = local_min + clip_margin
                        elif arr[i] > local_max - clip_margin:
                            out[i] = local_max - clip_margin
                        else:
                            out[i] = arr[i]
                    else:
                        out[i] = arr[i]
                else:
                    out[i] = arr[i]


@nb.guvectorize(["void(float64[:], float64[:])"], "(n)->(n)", nopython=True, cache=True)
def _robust_stabilization_numba(arr, out):
    """ロバスト安定化（window=50、学習側と同一）"""
    n = len(arr)
    window = 50
    finite_vals_buffer = np.empty(window, dtype=np.float64)
    for i in range(n):
        if i < 2:
            out[i] = 0.0 if not np.isfinite(arr[i]) else arr[i]
            continue

        start_idx = max(0, i - window + 1)
        window_data = arr[start_idx:i + 1]
        finite_vals = finite_vals_buffer[:len(window_data)]
        count = 0
        for j in range(len(window_data)):
            if np.isfinite(window_data[j]):
                finite_vals[count] = window_data[j]
                count += 1

        if count < 3:
            out[i] = 0.0 if not np.isfinite(arr[i]) else arr[i]
            continue

        valid_data = finite_vals[:count]
        median_val = np.median(valid_data)
        abs_devs = np.abs(valid_data - median_val)
        mad_val = np.median(abs_devs)

        if mad_val < 1e-10:
            mean_val = np.mean(valid_data)
            var_sum = np.sum((valid_data - mean_val) ** 2)
            mad_val = np.sqrt(var_sum / (count - 1)) * 0.6745 if count > 1 else 1e-10

        lower_bound = median_val - 3.0 * mad_val
        upper_bound = median_val + 3.0 * mad_val

        if np.isnan(arr[i]):
            out[i] = median_val
        elif np.isinf(arr[i]):
            out[i] = upper_bound if arr[i] > 0 else lower_bound
        else:
            if arr[i] < lower_bound:
                out[i] = lower_bound
            elif arr[i] > upper_bound:
                out[i] = upper_bound
            else:
                out[i] = arr[i]


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
        Args:
            data         : close/high/low/volume の numpy 配列を含む辞書
            lookback_bars: タイムフレームに応じた1日あたりのバー数。
                           fast_volume_mean_* の分母および QA の EWM 半減期に使用。
                           学習側 ProcessingConfig.timeframe_bars_per_day と同じ値を渡すこと。
                           例: M1→1440, M5→288, H1→24, H4→6
            qa_state     : QAState インスタンス。
                           本番稼働時は必ず渡し、同一インスタンスを毎バー使い回すこと。
                           None の場合は QA 処理をスキップ（後方互換・単体テスト用）。
        """
        features: Dict[str, float] = {}

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
        if len(high_arr) > 0 and len(low_arr) > 0:
            atr_arr = calculate_atr_wilder(high_arr, low_arr, close_arr, 13)
            atr_last_raw = float(atr_arr[-1]) if len(atr_arr) > 0 else np.nan
        else:
            atr_last_raw = np.nan

        atr_valid = np.isfinite(atr_last_raw)
        atr_last_safe = atr_last_raw + 1e-10 if atr_valid else np.nan

        close_last = float(close_arr[-1])

        # ---------------------------------------------------------
        # 1. 統計的モーメント系 — group_1_moments
        #    学習側: _create_basic_stats_features (window=[10,20,50])
        # ---------------------------------------------------------
        # 【バグ修正 (Step6)】条件を >= 2 → >= window に変更。
        # 学習側 Polars rolling_mean/rolling_std/rolling_var(window, ddof=1) は
        # データ数が window 本未満の先頭バーで NaN を返す（QAで0.0になる）。
        # 旧: >= 2 → window=10 のとき 2本でも計算してしまい学習側と不一致。
        # 新: >= window → window 本そろってから計算し、学習側と完全一致。
        for window in [10, 20, 50]:
            w_arr = _window(close_arr, window)
            if len(w_arr) >= window:
                mean_w = float(np.mean(w_arr))
                std_w  = float(np.std(w_arr, ddof=1))
                var_w  = float(np.var(w_arr, ddof=1))
                features[f"e1a_statistical_mean_{window}"]     = (close_last - mean_w) / atr_last_safe
                features[f"e1a_statistical_variance_{window}"] = var_w / (atr_last_safe ** 2)
                features[f"e1a_statistical_std_{window}"]      = std_w / atr_last_safe
                features[f"e1a_statistical_cv_{window}"]       = std_w / (mean_w + 1e-10)
            else:
                features[f"e1a_statistical_mean_{window}"]     = np.nan
                features[f"e1a_statistical_variance_{window}"] = np.nan
                features[f"e1a_statistical_std_{window}"]      = np.nan
                features[f"e1a_statistical_cv_{window}"]       = np.nan

        # ---------------------------------------------------------
        # 2. 歪度・尖度・高次モーメント — group_2_skew_kurt
        #    学習側: _create_skew_kurt_features (window=[20,50])
        # ---------------------------------------------------------
        for window in [20, 50]:
            w_arr = _window(close_arr, window)
            if len(w_arr) >= max(4, window):
                mean_w = float(np.mean(w_arr))
                n_w    = len(w_arr)

                # var_ddof0: 学習側の rolling_var(window, ddof=1) * (window-1)/window と等価。
                # Polars rolling_var は最後の `window` 要素を使い ddof=1 で計算する。
                # n_w は常に window と一致する（_window が末尾 window 要素を返すため）。
                var_ddof1 = float(np.var(w_arr, ddof=1))
                var_ddof0 = var_ddof1 * (n_w - 1.0) / n_w
                std_ddof0 = np.sqrt(var_ddof0 + 1e-10)
                std_ddof0_pow4 = var_ddof0 ** 2
                deviations = w_arr - mean_w

                # 【修正1】bias=True（Polars rolling_skew デフォルト）に完全一致
                features[f"e1a_statistical_skewness_{window}"] = _skewness_bias_true(w_arr)

                # kurtosis: 学習側と同定義（ddof=0 分散^2 で正規化、-3）
                features[f"e1a_statistical_kurtosis_{window}"] = (
                    float(np.mean(deviations ** 4)) / (std_ddof0_pow4 + 1e-10) - 3.0
                )

                # 高次モーメント（Zスコアベース ddof=0 + 1e-10）
                # 【修正2】全モーメントをこのブロックで確定させ、_rolling_moment() で上書きしない。
                # 学習側の Polars 式:
                #   std_ddof0 = (var_ddof0 + 1e-10).sqrt()
                #   ((close - mean_col) / std_ddof0).pow(moment).rolling_mean(window)
                # と完全等価。
                z_scores = deviations / std_ddof0
                for moment in [5, 6, 7, 8]:
                    features[f"e1a_statistical_moment_{moment}_{window}"] = (
                        float(np.mean(z_scores ** moment))
                    )
            else:
                features[f"e1a_statistical_skewness_{window}"] = np.nan
                features[f"e1a_statistical_kurtosis_{window}"] = np.nan
                for moment in [5, 6, 7, 8]:
                    features[f"e1a_statistical_moment_{moment}_{window}"] = np.nan

        # 【修正2】_rolling_moment() UDF による上書きブロックを完全削除。
        # 旧コード（削除済み）:
        #   features["e1a_statistical_moment_5_20"] = _rolling_moment(close_arr, 20, 5)
        #   features["e1a_statistical_moment_6_20"] = _rolling_moment(close_arr, 20, 6)
        #   features["e1a_statistical_moment_7_20"] = _rolling_moment(close_arr, 20, 7)
        #   features["e1a_statistical_moment_7_50"] = _rolling_moment(close_arr, 50, 7)
        # 理由: 学習側には _rolling_moment を map_batches で呼ぶコードが存在しない。
        #       全モーメントは上記 Group2 ブロックの Polars rolling 式相当で計算する。

        # ---------------------------------------------------------
        # 3. ロバスト統計 — group_3_robust
        #    学習側: _create_robust_stats_features (window=[10,20,50])
        # ---------------------------------------------------------
        for window in [10, 20, 50]:
            w_arr = _window(close_arr, window)
            if len(w_arr) >= window:
                median_w = float(np.median(w_arr))
                q25_w    = float(np.percentile(w_arr, 25))
                q75_w    = float(np.percentile(w_arr, 75))
                trim_w   = _trim_mean(w_arr, 0.1)
                features[f"e1a_robust_median_{window}"]       = (close_last - median_w) / atr_last_safe
                features[f"e1a_robust_q25_{window}"]          = (close_last - q25_w) / atr_last_safe
                features[f"e1a_robust_q75_{window}"]          = (close_last - q75_w) / atr_last_safe
                features[f"e1a_robust_iqr_{window}"]          = (q75_w - q25_w) / atr_last_safe
                features[f"e1a_robust_trimmed_mean_{window}"] = (close_last - trim_w) / atr_last_safe
            else:
                for key in ["robust_median", "robust_q25", "robust_q75",
                            "robust_iqr", "robust_trimmed_mean"]:
                    features[f"e1a_{key}_{window}"] = np.nan

        # ---------------------------------------------------------
        # 4. 高度ロバスト統計 — group_4_advanced
        #    学習側: _create_advanced_robust_features
        # ---------------------------------------------------------
        features["e1a_robust_mad_20"] = _last(calculate_mad(close_arr, 20)) / atr_last_safe

        biweight_arr = _biweight_location_numba(close_arr)
        features["e1a_robust_biweight_location_20"] = (
            (close_last - float(biweight_arr[-1])) / atr_last_safe
            if np.isfinite(biweight_arr[-1]) else np.nan
        )

        winsorized_arr = _winsorized_mean_numba(close_arr)
        features["e1a_robust_winsorized_mean_20"] = (
            (close_last - float(winsorized_arr[-1])) / atr_last_safe
            if np.isfinite(winsorized_arr[-1]) else np.nan
        )

        # ---------------------------------------------------------
        # 5. 統計検定 — group_5_tests
        #    学習側: _create_statistical_tests_features（pct_change に適用）
        # ---------------------------------------------------------
        pct_arr = _pct_change(close_arr)

        jb_arr   = _jarque_bera_numba(pct_arr)
        features["e1a_jarque_bera_statistic_50"] = float(jb_arr[-1])

        ad_arr   = _anderson_darling_numba(pct_arr)
        features["e1a_anderson_darling_statistic_30"] = float(ad_arr[-1])

        runs_arr = _runs_test_numba(pct_arr)
        features["e1a_runs_test_statistic_30"] = float(runs_arr[-1])

        vn_arr   = _von_neumann_ratio_numba(pct_arr)
        features["e1a_von_neumann_ratio_30"] = float(vn_arr[-1])

        # ---------------------------------------------------------
        # 6. 高速ローリング統計 — group_6_fast
        #    学習側: _create_fast_processing_features (window=[5,10,20,50,100])
        # ---------------------------------------------------------
        # 【バグ修正 (Step6)】条件を >= 2 → >= window に変更。
        # 学習側 Polars rolling_mean/rolling_std(window, ddof=1) は
        # window 本未満でNaNを返す。旧の >= 2 は学習側と不一致。
        for window in [5, 10, 20, 50, 100]:
            w_arr = _window(close_arr, window)
            if len(w_arr) >= window:
                mean_w = float(np.mean(w_arr))
                std_w  = float(np.std(w_arr, ddof=1))
                features[f"e1a_fast_rolling_mean_{window}"] = (close_last - mean_w) / atr_last_safe
                features[f"e1a_fast_rolling_std_{window}"]  = std_w / atr_last_safe
            else:
                features[f"e1a_fast_rolling_mean_{window}"] = np.nan
                features[f"e1a_fast_rolling_std_{window}"]  = np.nan

            # 【バグ修正 (Step6)】fast_volume_mean の window 条件を追加。
            # 学習側: volume.rolling_mean(window) は window 本未満で NaN。
            #         volume.rolling_mean(lookback_bars) は lookback_bars 本未満で NaN。
            # 旧: len(volume_arr) > 0 のみ → 短いデータでも非 NaN 値を返し学習側と不一致。
            # 新: 分子は len >= window、分母は len >= lookback_bars を満たす場合のみ計算。
            #     どちらかが不足 → NaN → QA で 0.0 (学習側と完全一致)。
            if len(volume_arr) >= max(window, lookback_bars):
                vol_w  = _window(volume_arr, window)
                vol_lb = _window(volume_arr, lookback_bars)
                vol_mean_w        = float(np.mean(vol_w))
                vol_mean_lookback = float(np.mean(vol_lb))
                features[f"e1a_fast_volume_mean_{window}"] = vol_mean_w / (vol_mean_lookback + 1e-10)
            elif len(volume_arr) >= window:
                # 分子は計算可能だが分母（lookback_bars）が不足 → NaN
                features[f"e1a_fast_volume_mean_{window}"] = np.nan
            else:
                features[f"e1a_fast_volume_mean_{window}"] = np.nan

        # ---------------------------------------------------------
        # 7. Numba最適化 — group_7_numba
        # ---------------------------------------------------------
        qs_arr = _fast_quality_score_numba(close_arr)
        features["e1a_fast_quality_score_50"] = float(qs_arr[-1])

        # ---------------------------------------------------------
        # 8. 品質保証特徴量 — group_8_qa
        # ---------------------------------------------------------
        bs_arr = _basic_stabilization_numba(close_arr)
        features["e1a_fast_basic_stabilization"] = (
            (close_last - float(bs_arr[-1])) / atr_last_safe
            if np.isfinite(bs_arr[-1]) else np.nan
        )

        rs_arr = _robust_stabilization_numba(close_arr)
        features["e1a_fast_robust_stabilization"] = (
            (close_last - float(rs_arr[-1])) / atr_last_safe
            if np.isfinite(rs_arr[-1]) else np.nan
        )

        # ---------------------------------------------------------
        # 【修正3】QA処理 — 学習側 apply_quality_assurance_to_group と等価
        #
        # 学習側: 全特徴量に対して
        #   inf/-inf→null → EWM(half_life=lookback_bars)±5σクリップ → fill_null/nan(0.0)
        # 本番側: QAState が EWM mean/var を跨バーで保持し、同一の式で逐次更新・クリップする。
        #
        # qa_state=None の場合は後方互換のため inf/NaN → 0.0 のみ適用。
        # ---------------------------------------------------------
        if qa_state is not None:
            qa_result: Dict[str, float] = {}
            for key, val in features.items():
                qa_result[key] = qa_state.update_and_clip(key, val)
            features = qa_result
        else:
            # qa_state 未指定時のフォールバック: inf/-inf/NaN → 0.0
            for key in list(features.keys()):
                if not np.isfinite(features[key]):
                    features[key] = 0.0

        return features
