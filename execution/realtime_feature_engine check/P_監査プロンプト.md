新チャット引き継ぎ用プロンプト
# Project Forge / Cimera V5 — Train-Serve Skew Audit 引き継ぎ

## プロジェクト概要
MQL5(EA) + Python アルゴ取引システム「Project Forge / Cimera V5」のXAU/USD M3 双方向取引で、
シミュレータでは利益が出るがライブではTO（タイムアウト）損失になる問題を調査中。
原因は学習側と本番側の特徴量計算の Train-Serve Skew と推定。

特徴量エンジンA〜Fの6個について、学習側 (engine_1_X_a_vast_universe_of_features.py) と
本番側 (realtime_feature_engine_1X_*.py) を1特徴量ずつ厳密比較し、
すべての乖離を修正することが目的。

## 進捗状況
エンジンA, B, C, D は完了済み（修正済みファイルが /mnt/user-data/outputs/ にある）。
**次はエンジンE → F の比較・修正**。

## アップロードするファイル
1. /mnt/user-data/outputs/Train_Serve_Skew_Audit_Report.md  ← 監査レポート
2. /mnt/user-data/outputs/realtime_feature_engine_1A_statistics.py  ← 修正済みA
3. /mnt/user-data/outputs/realtime_feature_engine_1B_timeseries.py  ← 修正済みB
4. /mnt/user-data/outputs/realtime_feature_engine_1C_technical.py   ← 修正済みC
5. /mnt/user-data/outputs/realtime_feature_engine_1D_volume.py      ← 修正済みD
6. /home/claude/realtime_feature_engine_1E_*.py（485行・未修正）   ← 次のターゲット
7. /home/claude/realtime_feature_engine_1F_*.py（282行・未修正）   ← Fも残っている
8. /home/claude/engine_1_E_a_vast_universe_of_features.py          ← 学習側E
9. /home/claude/engine_1_F_a_vast_universe_of_features.py          ← 学習側F
10. /home/claude/core_indicators.py                                 ← 共通指標ライブラリ

## 監査の手順（重要）
**ユーザーの絶対方針：「実用上問題なく差は観測不能」は許されない。
「そういうところから綻びが生まれる」ため、すべての乖離を厳密に修正すること。**

各エンジンについて以下の手順を必ず実施：

### Step 1: 特徴量リスト整理
学習側 _get_all_feature_expressions() の `expressions[f"{p}xxx"] = ...` を全列挙して、
特徴量数と各 window 値（period, num_std 等）を python で展開してカウント。

### Step 2: 本番側の出力一覧確認
本番側 calculate_features() の `features[f"e1X_xxx"] = ...` を全列挙し、
特徴量数が学習側と一致するか確認。

### Step 3: 各特徴量の数式比較
学習側の Polars 式（rolling_mean, rolling_std, ewm_mean, scale_by_atr, +1e-10 等）と
本番側の numpy/numba 実装を1つずつ比較。以下の点を必ずチェック：

- **ゼロ保護の epsilon 配置**：`(a + 1e-10) + 1e-10` vs `a + 1e-10` のような微差
- **ATR の扱い**：学習側 scale_by_atr(target, atr) は内部で `target / (atr + 1e-10)` 
- **rolling_mean / rolling_std のmin_samples**：Polars デフォルトは `min_samples=window_size`
  → ウィンドウ未満では NaN を返す。本番側でも同じ条件分岐が必要。
- **NaN 伝搬挙動**：Polars rolling_std はウィンドウ内NaN→NaN、
  Polars rolling_mean もウィンドウ内NaN→NaN、
  Polars rolling_quantile はNaNをスキップ（除外して計算）
- **vol_ma1440 系**：学習側は各バーで rolling_mean(1440) 再計算、
  本番側は最終バーで固定。最終バーの値だけならどちらも一致する場合があるが、
  例えば「rel_volume[i] = volume[i] / rolling_mean_1440[i] を全バーで計算してから rolling_mean(20)」
  のような式では各バーごとに rel_volume を再計算する必要がある（エンジンBの修正例）
- **late binding バグ**：学習側にバグがあって本番側が意図的にバグに合わせている場合がある
  （Williams%R の period=56 固定など）。コメントを必ず確認。
- **pct_change のゼロ除算**：prev==0 のとき、cur>0 → +inf、cur<0 → -inf、cur==0 → NaN
- **rolling_quantile のデフォルト**：Polars は interpolation='nearest'、
  numpy は np.percentile(..., method="nearest") で一致

### Step 4: UDF 実装の line-by-line 比較
学習側 @nb.guvectorize と本番側 @njit の UDF を sed/grep で並べて diff。
変数名・パラメータ・条件分岐がすべて同一であることを確認。

### Step 5: 乖離発見時の対応
ユーザーに乖離を報告し、修正方針を提示。**「実用上問題なく差は観測不能」を理由にしない**。
ユーザーの承認後、本番側を学習側に合わせて修正。コメントには：
- `[TRAIN-SERVE-FIX]` マーカー
- 旧/新の式を明示
- 学習側の式を引用
- 修正の理由

### Step 6: 構文チェック → 出力
- `python3 -c "compile(open('...').read(), '...', 'exec')"` で構文確認
- /mnt/user-data/outputs/ にコピー
- present_files で提示

## エンジンA-Dの実施済み修正（要約）

### エンジンA（statistics）
- ATR ゼロ保護を学習側と一致：`atr_last_safe = atr_last_raw + 1e-10` → `atr_last_raw`
  （学習側 __temp_atr_13 は生値）
- QAState の選択肢3（EWMスナップショット方式）コメントを追加（実装はせず）

### エンジンB（timeseries）
- volume_ma20, volume_price_trend を「各バーごとに rel_volume を計算」する実装に変更
  （旧版は最終バーの rolling_mean_1440 を分母固定 → 1〜3% の乖離発生）

### エンジンC（technical）
- trend_consistency の window 条件：`len(w) >= 3` → `len(w) >= period + 2`
  （学習側 Polars rolling_mean(period) min_samples=period と一致）
- atr_pct のゼロ保護：`if current_close != 0.0` 条件分岐 → `(close + 1e-10)` 加算保護
  （学習側 `atr_raw / (close + 1e-10) * 100` と完全一致）
- 冗長だった atr_pct_13/21 の二重計算ブロックを削除

### エンジンD（volume）
- hv_annual_252：`hv_standard_udf` → `np.std(pct_252, ddof=1)`（ウィンドウ全体finite時のみ）
  （学習側 rolling_std(252, ddof=1) と完全一致、UDFはfinite除外で挙動差あり）
- force_index_norm 分母：`atr*vol_ma1440 + 1e-10` → `atr*(vol_ma1440 + 1e-10) + 1e-10`
  （学習側 vol_ma1440 自体が `rolling_mean(1440) + 1e-10`）
- volume_ratio：`vol_ma20 + 1e-10` → ゼロ保護削除（学習側はゼロ保護なし）
- pct_change_numba：prev==0 で常に np.inf → cur の符号で +inf/-inf/NaN を返す
- hv_regime_50：`hv_standard_udf` → `_rolling_std_50_at`（ウィンドウ内NaN伝搬）

## 学習側の頻出パターン（チートシート）

```python
# 学習側 Polars 式の主要パターン
atr_13_internal_expr = ...calculate_atr_wilder(..., 13)...  # 生値（+1e-10なし）
atr_safe = pl.col("__temp_atr_safe")  # = atr + 1e-10（エンジンB/D）

# scale_by_atr の中身（core_indicators）
def scale_by_atr(target, atr, epsilon=1e-10):
    return target / (atr + epsilon)

# vol_ma1440（エンジンD）
vol_ma1440 = pl.col("volume").rolling_mean(lookback_bars) + 1e-10  # +1e-10 込み
```

## 開始指示
このプロンプトを読んだら、まず以下を実施：

1. /mnt/user-data/outputs/Train_Serve_Skew_Audit_Report.md を view で読む
2. /home/claude/engine_1_E_a_vast_universe_of_features.py の構造を grep で把握
3. /home/claude/realtime_feature_engine_1E_*.py の構造を grep で把握
4. エンジンEの特徴量を python で全列挙
5. ユーザーに「エンジンE開始します。特徴量X個確認しました」と報告
6. その後 step-by-step で比較・修正を実施

エンジンE完了後はエンジンFも同様に実施し、最後に Train_Serve_Skew_Audit_Report.md を更新する。

これをコピーして新チャットの最初に貼り付けてください。10ファイルもアップロードしてもらえれば、続きから同じやり方で進められます。
念のため、現時点でのアップロード必要ファイル一覧を整理しておきます：
新チャットでアップロードするファイル（10個）：
修正済み（完了分）：

Train_Serve_Skew_Audit_Report.md
realtime_feature_engine_1A_statistics.py
realtime_feature_engine_1B_timeseries.py
realtime_feature_engine_1C_technical.py
realtime_feature_engine_1D_volume.py

未修正（次にやる分）：
6. realtime_feature_engine_1E_*.py
7. realtime_feature_engine_1F_*.py
学習側（参照用）：
8. engine_1_E_a_vast_universe_of_features.py
9. engine_1_F_a_vast_universe_of_features.py
共通：
10. core_indicators.py