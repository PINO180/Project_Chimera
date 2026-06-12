# Production × Training 乖離検証ツールセット

「BTより本番のシグナルが多い」「方向が逆」「明らかに要らんところでシグナルが出る」 — これら3症状の根本原因を、 全特徴量の系統的乖離分析で特定するためのツール群です。

## ファイル一覧

| ファイル | 役割 |
|---|---|
| `snapshot_training_inference.py` | 学習側パイプライン経由 (S6 + 学習済モデル) で signal snapshot を生成 |
| `compare_snapshots.py` | Production と Training の snapshot を突合し、 乖離レポート生成 |
| `main_py_dry_run_patch.md` | main.py に `--dry-run` を追加するパッチ案 (リスクゼロでライブデータ収集) |

---

## 検証フロー (5/25 検証)

### Step 1 — 学習側 snapshot を生成

```bash
cd /workspace/models/
python snapshot_training_inference.py \
    --start 2026-05-25 \
    --end   2026-05-25 \
    --start-time 21:00:00 \
    --end-time   22:30:00 \
    --out /workspace/data/diagnostics/training_snapshot_20260525.parquet \
    --m2-proba 0.70 \
    --m2-delta 0.30 \
    --min-atr  0.80
```

**出力**:
- `training_snapshot_20260525.parquet`
- カラム: `timestamp`, `action`, `p_m1_long_raw`, `p_m1_short_raw`, `p_m2_long_raw`, `p_m2_short_raw`, `delta`, `passes_*`, + **全 S6 特徴量**

### Step 2 — Production × Training を突合

```bash
python compare_snapshots.py \
    --production /workspace/logs/triggered_features_log.csv \
    --training   /workspace/data/diagnostics/training_snapshot_20260525.parquet \
    --start 2026-05-25 \
    --end   2026-05-25 \
    --start-time 21:00:00 \
    --end-time   22:30:00 \
    --out-dir /workspace/data/diagnostics/compare_20260525 \
    --top-n 30
```

**出力**:
- `compare_20260525/report.md` — 人間可読サマリー (これを開発者が見る)
- `compare_20260525/feature_diff_summary.parquet` — 全特徴量の乖離テーブル
- `compare_20260525/signal_set_details.parquet` — prod_only / train_only / both の timestamp 詳細

---

## report.md の読み方

### ① シグナル発火集合
発火一致率を見る。 5/25 21:09-22:30 だと、 production が約 24 件、 training は約 11 件のはず。
両方発火した timestamp が「比較対象」 となる。

### ② 方向一致率
両方発火の中で、 BUY-BUY / SELL-SELL は一致、 BUY-SELL / SELL-BUY は反転。
21:15, 21:21, 21:24, 22:09 が反転していることが ここで確定する。

### ③ 予測値の系統的乖離
M1_long / M1_short / M2_long / M2_short の `mean_diff`, `correlation` を見る。
- correlation が高い (≈1.0) → 同じ値を出している (= feature が同じ)
- correlation が低い (≈0 or 負) → systematic に違う値を出している
- mean_diff の符号 → どっち向きにバイアスがあるか

### ④ 特徴量別の系統的乖離 TOP-30 (rel_diff 降順)
**ここが真犯人候補リスト**。

- `rel_diff = mean(|prod - train|) / mean(|train|)` 
- 1.0 以上 → ほぼ完全に違う値
- 0.1〜1.0 → 大きく乖離
- 0.001〜0.1 → 中程度
- ≤ 1e-7 → bit-identical (健全)

### ⑤ |mean_diff| 降順 = 符号バイアス
mean_diff が systematic に + か − に振れている特徴量。 これは「production だけ常にこの方向にズレる」 = **計算経路の根本差** を示唆する。

### ⑥ correlation 昇順
prod と train が完全に逆相関 (corr ≈ -1) の特徴量があれば、 それは **符号が逆転している特徴量**。 これも決定的な手がかり。

### ⑦ 全体サマリー
比較対象特徴量のうち、 bit-identical, 軽微, 中程度, 重度 の割合を集計。

---

## 期待される結果のパターン

### パターン A: 真犯人が数個に集中
TOP-N の rel_diff が 0.1 以上で、 残り (~900+) が ≤ 1e-7 になる。
→ 特定の特徴量計算 (e.g., 特定の rolling, OLS 純化) にバグがある。 計算経路をピンポイントで掘れる。

### パターン B: 多数の特徴量が中程度の乖離
TOP-N の rel_diff が 1e-3 〜 0.1。 数十〜百個。
→ 共通の上流計算 (M3 OHLC, ATR, disc など) でズレ → 後段に伝播。 OHLC レベルの不一致を疑う。

### パターン C: 全特徴量が均等に小さくズレる
rel_diff のヒストグラムが 1e-7〜1e-5 に集中。
→ dtype / precision / 浮動小数点累積誤差。 真の方向反転原因は別。

---

## triggered_features_log.csv のフォーマット (前提)

main.py L1377-1402 より:
```
Header: Timestamp, Action, Price, P_Long_M2, P_Short_M2, <feature_1>, <feature_2>, ...
Row:    "2026-05-25 21:15:01", "SELL", 4556.351, 0.0, 0.9996, ...
```

**注意**: `feature_keys` の順序は `feature_dict.keys()` 由来 (= Phase B 並列処理の到着順)。 比較スクリプトは **列名ベース join** なので順序不問。 ヘッダーが揃っていれば良い。

---

## トラブルシューティング

### 「比較対象特徴量が 0 列」 と出る
両 snapshot で feature の列名が **完全一致** していない。 production CSV のヘッダーと training parquet のスキーマを並べて確認:

```bash
# production
head -1 /workspace/logs/triggered_features_log.csv | tr ',' '\n' | sort > /tmp/prod_cols.txt

# training (Python で)
python -c "import polars as pl; df = pl.read_parquet('/workspace/data/diagnostics/training_snapshot_20260525.parquet'); print('\n'.join(sorted(df.columns)))" > /tmp/train_cols.txt

# 差分
diff /tmp/prod_cols.txt /tmp/train_cols.txt
```

### 「production のみに存在 / training のみに存在」 の警告が大量
→ 両者で feature の **集合自体** がズレている。 これも実は真因の手がかり (片方しか計算していない特徴量がある = バグ)。

### 期間内に「両方発火」 が 1 件もない
→ ① 段階で training のシグナル数が 0 になっていないか確認。 `--m2-proba` / `--m2-delta` / `--min-atr` が production と同じ値か確認。

---

## Step 3 — Dry-Run モードでデータ収集 (任意・推奨)

`main_py_dry_run_patch.md` の手順で main.py を改修すると、 リアル口座 feed を流したまま発注なしで動かせる。 数日〜数週間動かしてサンプルを蓄積し、 同じ手順で snapshot 比較すれば、 統計的に robust な検証ができる。

```bash
python main.py --dry-run
```

---

## 想定される検証期間 (拡張時)

- 5/25 のみ (本番初投入 52 分): 17 trades サンプル → 統計的に弱い
- Dry-Run で 1 週間: ~3,000 M3 boundary, ~500-1,000 trade候補 → robust
- Dry-Run で 1 ヶ月: 統計的に決定的な検証可能
