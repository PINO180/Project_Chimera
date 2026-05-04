# 診断スクリプト 一覧・使い方ガイド

Project Cimera V5 の診断に使用する4つのスクリプトの説明です。
配置場所: `/workspace/diagnostics/`

---

## スクリプトの位置づけ

```
【開発・修正時】                    【本番稼働時】
module_unit_test.py                feature_snapshot_tool.py
  ↓                                  ↓
計算ロジックの一致確認              シグナル発生時の特徴量を記録
（エンジン改修後に実行）                  ↓
                                  snapshot_comparator.py
                                    ↓
                                  本番値がS6分布と一致しているか確認

【モデル再学習後】
reproducibility_verifier.py
  ↓
S6特徴量→モデル→予測値がOOFと一致するか確認
```

---

## 1. `module_unit_test.py`

### 目的
学習側エンジン（Polars）と本番側エンジン（Numpy）に**同じS1データ**を食わせて、全特徴量の計算値をQAなし・純化なしのraw値で1対1比較する。エンジン改修後に「計算ロジックが一致しているか」を確認するためのツール。

### 入力
- `S1_MULTITIMEFRAME` のparquet（指定タイムスタンプ直前の2116本）
- 学習側: `/workspace/features/engine_1_X_*.py`
- 本番側: `/workspace/execution/realtime_feature_engine_1X_*.py`

### 出力
```
/workspace/data/diagnostics/unit_test/
  detail_YYYYMMDD_HHMMSS.csv    # 全特徴量のlearn値・rt値・乖離率
  summary_YYYYMMDD_HHMMSS.txt   # サマリーレポート
```

### 使い方
```bash
# 全モジュール・全時間足（デフォルト）
python module_unit_test.py

# 特定モジュールのみ
python module_unit_test.py --module 1C

# 特定時間足のみ
python module_unit_test.py --timeframe M1 M3

# タイムスタンプ・lookbackを指定
python module_unit_test.py \
  --timestamps "2023-03-15 09:00:00" "2023-07-20 14:30:00" \
  --lookback 2116

# 上位N件表示
python module_unit_test.py --top_n 100
```

### 判定基準
| 記号 | 乖離率 | 意味 |
|---|---|---|
| ✅ 一致 | < 0.01% | 問題なし |
| 🟡 近似 | < 1% | ほぼ問題なし |
| 🟠 要注意 | < 5% | 要確認 |
| 🔴 乖離 | 5%以上 | バグの疑い |

### 注意事項
- `--lookback 2116` が本番バッファサイズ（`SAFE_MIN_LOOKBACK=2016+100`）と一致するためデフォルト値
- lookbackが短いと計算に必要なウォームアップ不足で偽の乖離が出る
- `e1e_sample_weight` は学習側に存在しないため比較対象外
- `e1c_williams_r_14/28/56` は学習側のlate bindingバグによりperiod=56固定（意図的・詳細は両エンジンのコメント参照）

---

## 2. `feature_snapshot_tool.py`

### 目的
本番稼働中にシグナルが発生した瞬間の全特徴量値をCSVに保存する。後から`snapshot_comparator.py`で分析するためのデータ収集ツール。

### 入力
- `main.py` のシグナル処理部分から呼び出す

### 出力
```
/workspace/data/diagnostics/feature_snapshots/
  snapshot_YYYYMMDD_HHMMSS_L{m2l:.3f}_S{m2s:.3f}.csv
```

### 使い方
`main.py` のシグナル処理部分に以下を追記:
```python
from feature_snapshot_tool import save_feature_snapshot

save_feature_snapshot(
    signal.feature_dict,
    p_long_m1_raw,
    p_short_m1_raw,
    p_long_m2_raw,
    p_short_m2_raw,
    atr_ratio,
)
```

### 注意事項
- 最大500件保持（古いものから自動削除）
- `max_snapshots` 引数で変更可能
- 市場が動いていないと収集できない（土日は不可）

---

## 3. `snapshot_comparator.py`

### 目的
`feature_snapshot_tool.py` が記録したスナップショットの特徴量値を、S6データのランダムサンプルの統計（平均・標準偏差）と比較する。**本番が生成している特徴量値がS6の分布から外れていないか**をzスコアで確認する。

### 入力
- スナップショットCSV（`/workspace/data/diagnostics/feature_snapshots/snapshot_*.csv`）
- S6データ（`S6_WEIGHTED_DATASET`）からランダム200件

### 出力
```
/workspace/data/diagnostics/comparisons/
  comparison_YYYYMMDD_HHMMSS.csv    # 全特徴量のzスコア・統計値一覧
  summary_YYYYMMDD_HHMMSS.txt       # サマリーレポート
```

### 使い方
```bash
# 最新スナップショットを使用（デフォルト）
python snapshot_comparator.py

# スナップショットファイルを指定
python snapshot_comparator.py --snapshot_file snapshot_20240513_011800_L0.750_S0.120.csv

# 上位N件表示
python snapshot_comparator.py --top_n 100

# S6サンプル数を増やす（統計精度向上）
python snapshot_comparator.py --n_s6_samples 1000

# 純化前raw特徴量も含めて比較（問題切り分け用）
python snapshot_comparator.py --include_raw
```

### 判定基準
| 記号 | zスコア | 意味 |
|---|---|---|
| ✅ 正常 | ≤ 1σ | 分布内 |
| 🟡 1〜2σ | 1〜2σ | やや外れ |
| 🟠 2〜3σ | 2〜3σ | 要確認 |
| 🔴 3σ超 | 3σ超 | 異常の疑い |

### 注意事項
- S6とのタイムスタンプ1対1比較ではなく**分布比較**のため精度は統計的
- デフォルトは `_neutralized` 特徴量のみ比較。`--include_raw` でraw特徴量も追加
- raw特徴量はS6に存在しないカラムがある場合は自動スキップ
- 修正前のエンジンで生成したスナップショットを使っても意味がない（要注意）

---

## 4. `reproducibility_verifier.py`

### 目的
S6データの特徴量をM1/M2モデルに直接入力して予測値を再現し、OOFファイルの予測値と比較する。**モデルの推論パイプライン（入力順序・欠損値処理・logit変換）が正しいか**を確認する。リアルタイムエンジンは使わない。

### 入力
- S6データ（`S6_WEIGHTED_DATASET`）
- OOFファイル（`S7_M2_OOF_PREDICTIONS_LONG/SHORT`）
- モデルpkl（`S7_M1/M2_MODEL_LONG/SHORT_PKL`）
- 特徴量リスト（`m1/m2_long/short_features.txt`）

### 出力
```
/workspace/data/diagnostics/reproducibility/
  comparison_{direction}.csv        # サンプルごとのOOF予測値vs再現予測値
  summary_report_{direction}.txt    # サマリーレポート
```

### 使い方
```bash
# long・short両方（デフォルト）
python reproducibility_verifier.py

# 特定方向のみ
python reproducibility_verifier.py --direction long
python reproducibility_verifier.py --direction short

# サンプル数を増やす
python reproducibility_verifier.py --n_samples 100

# 閾値を下げる（サンプルが少ない場合）
python reproducibility_verifier.py --min_proba 0.60
```

### 見方
| 指標 | 理想値 | 意味 |
|---|---|---|
| MAE | < 0.01 | 平均絶対誤差 |
| 相関係数 | > 0.99 | OOFとの相関 |
| 差分 < 0.01 | > 90% | ほぼ完全一致の割合 |

### 注意事項
- リアルタイムエンジンの修正とは独立（S6→モデルのパイプライン検証）
- モデル再学習後・特徴量リスト変更後に実行する
- 乖離が大きいタイムスタンプはS6にデータが存在しないケースも多い


## 5. `analyze_fold_divergence.py`

### 目的
OOFファイルのタイムスタンプからfoldインデックスを逆算し、**foldの時系列位置と乖離量の相関**を定量的に検証する。M3メタ補正モデルの設計根拠を実証するための分析ツール。

### 入力
- OOFファイル（M1またはM2、Long/Short）
- S6データ（特徴量取得用）
- モデルpkl（M1/M2）

### 出力
/workspace/data/diagnostics/fold_divergence/
fold_divergence_{direction}.csv    # サンプルごとのfold位置・乖離量
summary_{direction}.txt            # サマリーレポート

### 使い方
```bash
# M2・long方向・1000件サンプル（デフォルト）
python analyze_fold_divergence.py

# 両方向・全件
python analyze_fold_divergence.py --direction both --n_samples 0

# short方向・1000件
python analyze_fold_divergence.py --direction short --n_samples 1000

# M1レベルで確認
python analyze_fold_divergence.py --oof_level m1
```

### 見方
`fold_position vs |logit_diff|` の相関係数が鍵。負なら古いfoldほど乖離大（Gemini仮説支持）、正なら新しいfoldほど乖離大（逆仮説）、0付近ならfold位置は無関係。

### 注意事項
- foldインデックスはタイムスタンプから**逆算**（Ax2のfold分割を再現）。purge/embargo境界は近似。
- 次回Ax2再実行時にfold_indexをOOFに追記すれば完全一致になる
- M2は常にM1より乖離が大きい（カスケードによる増幅）
- `--n_samples 0`で全件処理（M2全件は約7分）
---

## 実行フロー（用途別）

### エンジン改修後の検証
```
1. module_unit_test.py          ← 計算ロジックの一致確認
2. 本番デプロイ
3. feature_snapshot_tool.py     ← 月曜市場開始後にスナップショット収集
4. snapshot_comparator.py       ← 本番値がS6分布と一致しているか確認
```

### モデル再学習後の検証
```
1. reproducibility_verifier.py  ← モデルの推論パイプライン確認
2. 本番デプロイ
3. feature_snapshot_tool.py → snapshot_comparator.py
```
