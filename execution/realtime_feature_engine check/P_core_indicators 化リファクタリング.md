承知しました。リファクタリング用の引き継ぎプロンプトを作成します。

core_indicators 化リファクタリング 引き継ぎプロンプト
# Project Forge / Cimera V5 — core_indicators 化リファクタリング

## 目的
特徴量計算エンジン1A〜1Dの全UDFを core_indicators.py に移動し、
学習側 (engine_1_X) と本番側 (realtime_feature_engine_1X) の両方が
同じ関数オブジェクトを import する SSOT (Single Source of Truth) パターンに
統一する。これにより Train-Serve Skew リスクを構造的に排除する。

エンジン1E/1Fは既に SSOT 化済みのため対象外。

## 絶対遵守ルール（最重要）

### ルール1: byte-perfect 移動のみ
既存実装を1文字も変えずに core_indicators に移動する。
以下の変更は一切しない：
- デコレータの引数（fastmath=True/False、cache、parallel）
- 関数引数の名前・型・順序
- 内部のロジック・ループ順序
- 変数名・コメント
- ddof、epsilon、min_periods 等のパラメータ

「ついでに改善」の誘惑に絶対に屈しない。改善は別タスク。

### ルール2: 数値再現性を絶対に保証する
リファクタリング前後で全特徴量の値が完全一致することを確認するまで
コミットしない。微差すら許容しない。

### ルール3: 1エンジンずつ独立にロールバック可能にする
全エンジン同時にいじらない。エンジン1Aが完了→検証→コミットしてから
1Bに進む。問題発生時に直前のエンジンに戻せる状態を維持する。

### ルール4: 再学習を発生させない
モデルは学習時の特徴量値で訓練済み。本作業は計算ロジックを変更しない
ため、モデルの再学習は不要であり、また発生させてはならない。
もし数値乖離が発見された場合、リファクタリング側にバグがあるとみなし
修正する。学習データを再生成しない。

## 対象ファイルと現状

### 対象UDF（学習側・本番側で個別ローカル定義されている）

エンジン1A (statistics)：
- statistical_mean_udf, statistical_var_udf, statistical_std_udf
- statistical_cv_udf, statistical_skewness_udf, statistical_kurtosis_udf
- statistical_moment_udf, robust_mean_udf, robust_var_udf, robust_std_udf
- jarque_bera_udf, anderson_darling_udf, runs_test_udf, von_neumann_udf
- fast_rolling_mean_udf, fast_rolling_std_udf, fast_quality_score_udf
- basic_stabilization_udf, robust_stabilization_udf
- 等、約25個

エンジン1B (timeseries)：
- adf_統計量_udf, phillips_perron_統計量_udf, kpss_統計量_udf
- t分布_自由度_udf, t分布_尺度_udf, gev_形状_udf
- holt_winters_レベル_udf, holt_winters_トレンド_udf
- arima_残差分散_udf, kalman_状態推定_udf
- lowess_適合値_udf, theil_sen_傾き_udf
- 計12個

エンジン1C (technical)：
- _calculate_di_plus_scalar, _calculate_di_minus_scalar
- calculate_aroon_up_numba, calculate_aroon_down_numba
- calculate_stochastic_numba, calculate_williams_r_numba
- calculate_trix_numba, calculate_ultimate_oscillator_numba
- calculate_tsi_numba, _wma_helper
- calculate_wma_numba_arr (本番側名), calculate_hma_numba_arr (本番側名)
- calculate_kama_numba_arr (本番側名)
- 計約10個
- 注: calculate_atr_wilder, calculate_rsi_wilder, calculate_macd,
  calculate_sma, calculate_bollinger, calculate_adx は既に core_indicators
  にあるので対象外

エンジン1D (volume)：
- chaikin_volatility_udf, mass_index_udf
- cmf_udf, mfi_udf, vwap_udf
- obv_udf, accumulation_distribution_udf, force_index_udf
- commodity_channel_index_udf, candlestick_patterns_udf
- fibonacci_levels_udf, hv_standard_udf, hv_robust_udf
- 計約13個
- 注: これらの多くは既に core_indicators にも別バージョンが存在する
  「孤児関数」になっている（誰も import していない）。
  まず学習側と本番側のローカル定義が同一か確認し、
  core_indicators 既存版と差分があれば学習側を正とする。

### 対象ファイル

学習側（修正対象）：
- /workspace/path/to/engine_1_A_a_vast_universe_of_features.py
- /workspace/path/to/engine_1_B_a_vast_universe_of_features.py
- /workspace/path/to/engine_1_C_a_vast_universe_of_features.py
- /workspace/path/to/engine_1_D_a_vast_universe_of_features.py

本番側（修正対象）：
- /mnt/user-data/outputs/realtime_feature_engine_1A_statistics.py
- /mnt/user-data/outputs/realtime_feature_engine_1B_timeseries.py
- /mnt/user-data/outputs/realtime_feature_engine_1C_technical.py
- /mnt/user-data/outputs/realtime_feature_engine_1D_volume.py

共通モジュール（追記対象）：
- /workspace/core/core_indicators.py

参考（既に SSOT 化済み、対象外）：
- engine_1_E_a_vast_universe_of_features.py
- engine_1_F_a_vast_universe_of_features.py
- realtime_feature_engine_1E_signal.py
- realtime_feature_engine_1F_experimental.py

## 実施手順（1エンジンずつ）

### Phase 0: ベースライン特徴量データの保存（最初に1回だけ実施）

リファクタリング着手前に、現状のコードで全特徴量を生成して Parquet で保存する。
これがリファクタリング後の数値完全一致を検証する「正解値」となる。

```python
# 例: 学習データの末尾10000本に対して全特徴量を計算
import polars as pl
from engine_1_A_a_vast_universe_of_features import CalculationEngine as Engine1A
from engine_1_B_a_vast_universe_of_features import CalculationEngine as Engine1B
# ... 1C, 1D も同様

baseline_lf = pl.scan_parquet("...input...")
result_1A = Engine1A(...).calculate_all_features(baseline_lf)
result_1A.collect().write_parquet("/tmp/baseline_1A.parquet")
# 1B, 1C, 1D も同様
```

また、本番側の各バーで生成される特徴量値も保存する：
```python
# realtime_feature_engine_1X.calculate_features() を1000本ぐらい走らせて、
# 各特徴量の値を pickle で保存
import pickle
results = {}
for bar_idx in range(len(test_data)):
    window = get_window(test_data, bar_idx)
    feats = FeatureModule1A.calculate_features(window, 1440, qa_state=None)
    results[bar_idx] = feats
with open("/tmp/baseline_realtime_1A.pkl", "wb") as f:
    pickle.dump(results, f)
```

### Phase 1: エンジン1A をリファクタリング

#### Step 1: 学習側UDFの場所を特定
```bash
grep -nE "^@nb\.|^@njit|^def [a-z_]+_udf|^def [a-z_]+_numba" \
  engine_1_A_a_vast_universe_of_features.py
```

#### Step 2: 本番側UDFの場所を特定
```bash
grep -nE "^@nb\.|^@njit|^def [a-z_]+_udf|^def [a-z_]+_numba" \
  realtime_feature_engine_1A_statistics.py
```

#### Step 3: 各UDFについて、学習側と本番側の実装が一致することを確認
```bash
# 例: statistical_mean_udf
diff <(sed -n '<学習側の行範囲>p' engine_1_A_*.py) \
     <(sed -n '<本番側の行範囲>p' realtime_feature_engine_1A_*.py)
```

差分があれば学習側を正とする（モデルは学習側で訓練されているため）。

#### Step 4: 学習側のUDFをそのまま core_indicators.py に追加
```python
# core_indicators.py の末尾近くに、新カテゴリブロックとして追加：
# ===========================================================================
# [CATEGORY: STATISTICS]  統計系UDF (engine_1_A 由来、Step 14 で SSOT 化)
# ===========================================================================

@nb.njit(fastmath=True, cache=True)  # ← 学習側と完全同一のデコレータ
def statistical_mean_udf(arr: np.ndarray) -> float:
    # 学習側からそのままコピー、1文字も変更しない
    ...
```

注意：
- デコレータの引数（fastmath, cache, parallel）を一切変更しない
- 関数名・引数名・実装すべて学習側と完全一致
- core_indicators の既存スタイルに「合わせよう」としない

#### Step 5: 学習側 engine_1_A_*.py を更新
```python
# 既存の "from core_indicators import (...)" に追加：
from core_indicators import (
    calculate_atr_wilder,
    scale_by_atr,
    calculate_sample_weight,
    # 以下を追加
    statistical_mean_udf,
    statistical_var_udf,
    # ... 25個分
)
```

そしてファイル内のローカル定義を削除する：
```python
# 削除対象（学習側ファイル内）：
@nb.njit(fastmath=True, cache=True)
def statistical_mean_udf(arr):
    ...
# ↑ これらをすべて削除（import で取得するため）
```

#### Step 6: 本番側 realtime_feature_engine_1A_*.py を更新
学習側と同じ要領で：
- core_indicators からの import に切り替え
- ローカル定義を削除

#### Step 7: 数値再現性の検証

リファクタリング後の学習側で再度特徴量を生成：
```python
result_1A_after = Engine1A(...).calculate_all_features(baseline_lf)
df_after = result_1A_after.collect()
df_before = pl.read_parquet("/tmp/baseline_1A.parquet")

# 全特徴量×全行で完全一致を確認
for col in df_after.columns:
    if col in df_before.columns:
        diff = (df_after[col] - df_before[col]).abs().max()
        assert diff == 0.0 or (np.isnan(diff)), f"差異発見: {col}, max diff = {diff}"
print("学習側 全特徴量完全一致 ✅")
```

本番側も同様：
```python
results_after = {}
for bar_idx in range(len(test_data)):
    window = get_window(test_data, bar_idx)
    feats = FeatureModule1A.calculate_features(window, 1440, qa_state=None)
    results_after[bar_idx] = feats

import pickle
with open("/tmp/baseline_realtime_1A.pkl", "rb") as f:
    results_before = pickle.load(f)

for bar_idx, feats in results_after.items():
    for k, v in feats.items():
        before_v = results_before[bar_idx][k]
        if np.isnan(v) and np.isnan(before_v):
            continue
        assert v == before_v, f"差異発見 bar={bar_idx} key={k}: {before_v} vs {v}"
print("本番側 全特徴量完全一致 ✅")
```

#### Step 8: コミット
ここで初めて git commit。エンジン1Aのリファクタリング完了。

### Phase 2: エンジン1B をリファクタリング
Phase 1 と同じ要領で実施。

### Phase 3: エンジン1C をリファクタリング
Phase 1 と同じ要領。注意点：calculate_atr_wilder などは既に core_indicators
にあるので対象外。新規追加するのは Williams%R late binding バグ込みの
オシレーター系UDFなど。

### Phase 4: エンジン1D をリファクタリング
注意：core_indicators に既に同名の chaikin_volatility_udf 等が存在する
（孤児関数）。学習側のローカル定義と core_indicators の既存版を diff し、
学習側が正であることを確認する。差分があれば core_indicators 側を学習側で
上書き。本番側 import 先を core_indicators の更新版に向ける。

### Phase 5: 全エンジン通し検証
全エンジン1A〜1Dのリファクタリング完了後、エンドツーエンドで本番システムを
1000バー程度走らせ、ベースラインと数値完全一致することを確認する。

## 数値乖離が発生した場合の対処

### Case A: 1ULP（10^-15程度）の差
浮動小数点演算順序の影響。numba の JIT コンパイル時の最適化が変わった
可能性。LightGBM は決定木ベースで微差に頑健なため実用上問題ないが、
本リファクタリングでは「完全一致」を目標とするため、原因を特定して修正する。

考えられる原因：
- numba のキャッシュが古い → /tmp/numba_cache/ を削除して再実行
- デコレータの引数が微妙に違う → 学習側と完全一致を再確認
- 関数のグローバル変数依存があった → 移動先でも同じグローバル変数が
  見えるか確認

### Case B: 数%レベルの差
リファクタリング中にバグを混入した可能性が高い。即座にエンジン全体を
ロールバックし、原因を特定して再実施する。

### Case C: 完全一致したが本番モデルの予測値が変わった
理論的にはありえない。もし発生したら：
- QAState の EWM の状態が変わった可能性 → ウォームアップ手順の確認
- リファクタリング前後で QAState の更新順序が変わった可能性 → 確認

それでも解決しない場合のみ、再学習を検討する。

## 完了の定義

- 全UDFが core_indicators に移動済み
- engine_1_A〜D と realtime_feature_engine_1A〜D が同じ関数オブジェクトを
  import している（grep で local def が無いことを確認）
- 全特徴量がリファクタリング前のベースラインと完全一致
- 本番システムで実走テスト 1000本 → エラーなく完走
- このレポートに作業内容と検証結果を追記

## 開始時のステップ

1. このプロンプトを読んだら、まずユーザーに「Phase 0: ベースライン保存
   から始めます」と確認を取る
2. ベースライン保存スクリプトを作成・実行
3. ユーザー承認後、Phase 1（エンジン1A）に着手
4. Phase 1 完了時、ユーザーに数値検証結果を報告し承認を得てから Phase 2 に進む

## 必要なアップロードファイル（次チャットでこれを使う場合）

最低限：
- /home/claude/engine_1_A_a_vast_universe_of_features.py
- /home/claude/engine_1_B_a_vast_universe_of_features.py
- /home/claude/engine_1_C_a_vast_universe_of_features.py
- /home/claude/engine_1_D_a_vast_universe_of_features.py
- /home/claude/realtime_feature_engine_1A_statistics.py
- /home/claude/realtime_feature_engine_1B_timeseries.py
- /home/claude/realtime_feature_engine_1C_technical.py
- /home/claude/realtime_feature_engine_1D_volume.py
- /home/claude/core_indicators.py
- /home/claude/Train_Serve_Skew_Audit_Report.md（参考）

参考用：
- /home/claude/engine_1_E_a_vast_universe_of_features.py（SSOT 完成形の見本）
- /home/claude/realtime_feature_engine_1E_signal.py（SSOT 完成形の見本）
- /home/claude/realtime_feature_engine_1F_experimental.py（SSOT 完成形の見本）

## 期待される最終状態

リファクタリング完了後、以下が成立する：

```bash
# 1. 全 _udf / _numba 関数が core_indicators に集約されている
grep -c "^def .*_udf\|^def .*_numba" core_indicators.py
# → 約60個

# 2. engine_1_A〜D と realtime_feature_engine_1A〜D には UDF 定義が存在しない
grep -c "^def .*_udf\|^def .*_numba" engine_1_A_*.py
# → 0個（または極めて少数）
grep -c "^def .*_udf\|^def .*_numba" realtime_feature_engine_1A_*.py
# → 0個

# 3. 各エンジンが同じ関数オブジェクトを import している
python3 -c "
from engine_1_D_a_vast_universe_of_features import chaikin_volatility_udf as a
from realtime_feature_engine_1D_volume import chaikin_volatility_udf as b
assert a is b, 'Different objects!'
print('Same object ✅')
"
```

これで Train-Serve Skew は構造的に発生不可能になる。