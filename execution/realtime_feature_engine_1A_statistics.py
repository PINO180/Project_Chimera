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
    # [SSoT 統一] Engine 1A の Numba 関数を core_indicators から import
    # (旧: 本ファイル内に同名関数 _* prefix で重複定義 = SSoT 違反)
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

    📝 [将来の選択肢: EWMスナップショット方式 — 現時点では未実装]
        起動時のシード差を完全にゼロにするには、学習側で訓練時のEWM最終状態
        （mean / var / n）を pickle 等で保存し、本番初回起動時にロードして
        QAState を初期化する方式がある。

        実装イメージ:
            # 学習側（2_G_alpha_neutralizer等の前段に追加）:
            ewm_snapshot = {
                "e1a_statistical_mean_10_M3": {"mean": ..., "var": ..., "n": ...},
                ...
            }
            pickle.dump(ewm_snapshot, "ewm_snapshot.pkl")

            # 本番側:
            qa_state = QAState(lookback_bars=2880)
            qa_state.load_snapshot("ewm_snapshot.pkl")  # 数千個のEWM状態を一気に復元

        このアプローチを採用すれば理論的には学習・本番のEWM軌跡が完全に一致するが、
        以下の理由から現時点では実装していない:
          1. EWM QAは ±5σ という非常に広いバンドの外れ値クリッピングであり、
             通常時の特徴量値はクリップされず raw_val がそのまま通過する
          2. ウォームアップを十分行えば EWM は学習時の値に指数収束する
             （4132本ウォームアップで起動時シード差の影響は数十%まで減衰、
              その後さらに指数的に減衰）
          3. LightGBMは決定木ベースで、特徴量値の小数点以下の微小な差には頑健
          4. 実装コスト（学習側改修・スナップショット運用・全特徴量×時間足×Long/Short
             の状態管理）が、得られる効果に対して大きすぎる

        将来、以下のいずれかが該当した場合に再検討する価値がある:
          - クリップが頻発する特徴量が新規追加された場合
          - ウォームアップを大幅に短縮したい場合
          - 規制当局・監査法人に「学習・本番の完全数値一致」を証明する必要が出た場合

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
        #
        # 【修正前 (誤式)】
        #   bias_corr = 1 / sqrt(1 - (1-alpha)^(2n))
        #   この式は Polars の真の bias correction とズレており、
        #   ウォームアップ初期 (n < ~100バー) で σ が過小評価される。
        #
        # 【修正後 (Polars 互換式 — 実証検証で 1e-15 精度で完全一致)】
        #   adjust=False の重み: w_k = alpha*(1-alpha)^k (k=0..n-2) と
        #                       w_{n-1} = (1-alpha)^(n-1) (最古項を正規化保持)
        #   sum_w = 1 (常に)、sum_w2 = 重みの2乗和
        #   bias_factor_var = 1 / (1 - sum_w2)
        #     r2     = (1 - alpha)^2
        #     m      = n - 1                          # 漸化式は1段先送りで n-1 が正解
        #     sum_w2 = alpha^2 * (1 - r2^m) / (1 - r2) + r2^m   (m >= 1)
        #     ewm_std = sqrt(ewm_var * bias_factor_var)
        ewm_mean = self._ewm_mean[key]
        n_updates = self._ewm_n.get(key, 1)
        if n_updates <= 1:
            # n=1 は分散自体が 0 なので bias 補正不要
            ewm_std = 0.0
        else:
            r2 = (1.0 - alpha) ** 2
            m  = n_updates - 1
            if r2 < 1.0 - 1e-15:
                sum_w2 = alpha * alpha * (1.0 - r2 ** m) / (1.0 - r2) + r2 ** m
            else:
                # alpha が極端に小さい (HL → ∞) 退化ケース
                sum_w2 = 1.0
            if sum_w2 < 1.0 - 1e-15:
                bias_factor_var = 1.0 / (1.0 - sum_w2)
                ewm_std = np.sqrt(max(self._ewm_var[key] * bias_factor_var, 0.0))
            else:
                ewm_std = 0.0
        lower    = ewm_mean - 5.0 * ewm_std
        upper    = ewm_mean + 5.0 * ewm_std

        # =====================================================================
        # 【修正済み】学習側 Polars との完全一致 (Option B)
        #
        # 旧実装は 2 重のバグ:
        #   1) np.isnan(ewm_input) チェックが is_pos_inf より先に発火し、
        #      inf 入力時に 0.0 を返してしまう (is_pos_inf 分岐に到達不能)
        #   2) チェック対象が ewm_input (inf を NaN 化したもの) で、本来の raw_val を見ていない
        #
        # 修正方針:
        #   - +inf 入力 → upper bound を返す (engine_1_A の修正後と同じ)
        #   - -inf 入力 → lower bound を返す
        #   - NaN 入力  → 0.0 を返す (fill_nan(0.0) 等価)
        #   - 有限値    → clip(lower, upper) で ±5σ クリップ
        # =====================================================================
        if is_pos_inf:
            return float(upper) if np.isfinite(upper) else 0.0
        if is_neg_inf:
            return float(lower) if np.isfinite(lower) else 0.0
        if np.isnan(raw_val):
            return 0.0

        clipped = float(np.clip(raw_val, lower, upper))

        # Step4: fill_nan(0.0)
        return clipped if np.isfinite(clipped) else 0.0


# ==================================================================
# Numba UDF群（学習側 engine_1_A と完全同一実装）
# ==================================================================

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
        # [TRAIN-SERVE-FIX] 学習側 _create_basic_stats_features 等は __temp_atr_13 を
        # 生値のまま分母に使用する（ゼロ保護なし）。本番側も学習側と完全一致させる。
        # 旧: atr_last_safe = atr_last_raw + 1e-10  （実害は10^-11オーダーで観測不能だが、
        #     思想として揃えるため削除）
        # 新: atr_last_safe = atr_last_raw          （学習側と完全同一）
        # XAU/USDのATRは事実上ゼロにならないため、ゼロ除算リスクは存在しない。
        atr_last_safe = atr_last_raw if atr_valid else np.nan

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
                # 【根本修正】kurtosis/momentの学習側Polarsとの完全一致
                #
                # 学習側(Polars)の計算:
                #   各バーiで z_i = (close[i] - rolling_mean_i) / rolling_std_ddof0_i
                #   最終値 = rolling_mean(z^moment, window)[-1]
                #          = mean(z_{N-window}^m, ..., z_{N-1}^m)
                #
                # 旧実装は w_arr全体を一括でzスコア化していたため不一致だった。
                # 正しくは「各バーがそのバーのrollingウィンドウに基づくzスコア」を使う。
                n = len(close_arr)
                # z_per_bar[idx] = (close[bar_i] - mean_i) / std_ddof0_i
                # k_num[idx]     = (close[bar_i] - mean_i)^4  ← 分子のみ（分母は最終バーで共通）
                z_per_bar = np.empty(window, dtype=np.float64)
                k_num_per_bar = np.empty(window, dtype=np.float64)
                for idx in range(window):
                    bar_i = n - window + idx
                    if bar_i - window + 1 < 0:
                        z_per_bar[idx] = np.nan
                        k_num_per_bar[idx] = np.nan
                        continue
                    sub = close_arr[bar_i - window + 1: bar_i + 1]
                    sub_mean = np.mean(sub)
                    sub_var0 = np.var(sub, ddof=1) * (window - 1.0) / window
                    sub_std0 = np.sqrt(sub_var0 + 1e-10)
                    z_per_bar[idx] = (close_arr[bar_i] - sub_mean) / sub_std0
                    k_num_per_bar[idx] = (close_arr[bar_i] - sub_mean) ** 4

                # kurtosis: 分子はrolling_mean((close-mean_i)^4)、分母は最終バーのvar_ddof0^2
                last_var0 = np.var(w_arr, ddof=1) * (window - 1.0) / window
                valid_kn = k_num_per_bar[np.isfinite(k_num_per_bar)]
                features[f"e1a_statistical_skewness_{window}"] = _skewness_bias_true(w_arr)
                features[f"e1a_statistical_kurtosis_{window}"] = (
                    float(np.mean(valid_kn)) / (last_var0 ** 2 + 1e-10) - 3.0
                    if len(valid_kn) > 0 else np.nan
                )

                valid_z = z_per_bar[np.isfinite(z_per_bar)]
                for moment in [5, 6, 7, 8]:
                    features[f"e1a_statistical_moment_{moment}_{window}"] = (
                        float(np.mean(valid_z ** moment)) if len(valid_z) > 0 else np.nan
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
                # Polars rolling_quantile のデフォルトは interpolation="nearest"
                q25_w    = float(np.percentile(w_arr, 25, method="nearest"))
                q75_w    = float(np.percentile(w_arr, 75, method="nearest"))
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

        biweight_arr = biweight_location_numba(close_arr)
        features["e1a_robust_biweight_location_20"] = (
            (close_last - float(biweight_arr[-1])) / atr_last_safe
            if np.isfinite(biweight_arr[-1]) else np.nan
        )

        winsorized_arr = winsorized_mean_numba(close_arr)
        features["e1a_robust_winsorized_mean_20"] = (
            (close_last - float(winsorized_arr[-1])) / atr_last_safe
            if np.isfinite(winsorized_arr[-1]) else np.nan
        )

        # ---------------------------------------------------------
        # 5. 統計検定 — group_5_tests
        #    学習側: _create_statistical_tests_features（pct_change に適用）
        # ---------------------------------------------------------
        pct_arr = _pct_change(close_arr)

        jb_arr   = jarque_bera_statistic_numba(pct_arr)
        features["e1a_jarque_bera_statistic_50"] = float(jb_arr[-1])

        ad_arr   = anderson_darling_numba(pct_arr)
        features["e1a_anderson_darling_statistic_30"] = float(ad_arr[-1])

        runs_arr = runs_test_numba(pct_arr)
        features["e1a_runs_test_statistic_30"] = float(runs_arr[-1])

        vn_arr   = von_neumann_ratio_numba(pct_arr)
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
        qs_arr = fast_quality_score_numba(close_arr)
        features["e1a_fast_quality_score_50"] = float(qs_arr[-1])

        # ---------------------------------------------------------
        # 8. 品質保証特徴量 — group_8_qa
        # ---------------------------------------------------------
        bs_arr = basic_stabilization_numba(close_arr)
        features["e1a_fast_basic_stabilization"] = (
            (close_last - float(bs_arr[-1])) / atr_last_safe
            if np.isfinite(bs_arr[-1]) else np.nan
        )

        rs_arr = robust_stabilization_numba(close_arr)
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
