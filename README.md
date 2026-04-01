# 🔥 Project Forge V1.0

> **XAU/USD 完全自動化クオンツ取引システム**
> 統計的優位性に基づく金融時系列解析・Two-Brain メタラベリングアーキテクチャ

---

## 🎯 概要

Project Forge は XAU/USD（金/米ドル）を対象とした完全自動化の定量的取引システムです。
人間の認知バイアスを排除し、データと統計的優位性のみに基づいてエントリー・エグジットを判断します。

- **対象市場**：XAU/USD（金/米ドル）
- **戦略**：M3 スキャルピング（TD=30分・PT=ATR×1.0・SL=ATR×5.0）
- **モデル**：Two-Brain（M1 方向予測 + M2 メタラベリング）× Long/Short 独立 = 4モデル
- **データ**：2021〜2026年の Tick データ（2.1億行超）

---

## 🔥 哲学

金融市場への扉が広く開かれた現代、多くの人々がその舞台に足を踏み入れる。
そこで待ち受けるのは、しばしば偶然の成功という名の甘い罠だ。

一度その蜜を味わった者は自らの才能を過信し、聖杯を探し求めるかのように
無数のテクニカル指標を組み合わせ、「必勝法」を見つけようと躍起になる。
しかしそれは、市場という巨大な生態系を前にして、
たった一種のプランクトンの動きを追いかけているに等しい。

確かにこの世界には例外が存在する。
深い経験と天賦の才を持つ「裁量の天才」と呼ばれる人々だ。
彼らは需給のうねりや投資家心理の微細な変化を瞬時に見抜く。
ありふれたテクニカル指標でさえ、市場参加者の欲望と恐怖を映し出す鏡と化す。
それはもはや第六感と呼ぶべきもので、再現性もなければ他者に教え授けることもできない、一種の芸術だ。

しかし、我々凡人は彼らにはなれない。
人間の脳は、無数の変数が複雑に絡み合う市場の混沌を冷徹かつ高速に処理するようには設計されていない。
恐怖は視野を狭め、欲望は判断を鈍らせる。
過去の成功体験と手痛い失敗の記憶が、目の前の客観的な事実を歪め続ける。

であるならば、我々が取るべき道は一つしかない。
自らの認知能力の限界を謙虚に認め、客観的かつ体系的なアプローチに活路を見出すことだ。
それは、嵐の夜に自らの感覚を疑い、計器飛行にすべてを委ねるパイロットの決断に似ている。

Project Forge はその答えだ。

2億1千万行のティックデータから 1722 個の特徴量を生成し、
KS検定・相関フィルター・OLS純化という三重の防衛網でアルファを蒸留する。
トリプルバリアラベリングは OHLCデータでは解決できない「ヒゲの中の真実」を
マイクロ秒単位のティックで裁定し、純粋な教師データを作り上げる。
Two-Brain アーキテクチャは Long と Short を独立した知性として扱い、
Purged K-Fold でデータリークを排除して鍛えられたモデルが推論する。

このシステムが問うているのは「相場は予測できるか」ではない。
「統計的に優位なゲームを繰り返し、期待値の積分として利益を得られるか」だ。

個別のトレードに執着しない。勝率より期待値を信じる。短期の損失に揺るがない。
これが個人投資家としての、唯一の正解だと確信している。

---

## 📐 設計思想

### スケール不変特徴量

全ての特徴量を ATR13 割り・パーセンテージ・比率で正規化し、価格水準・ボラティリティレジームに依存しない表現を採用する。

### ATR Ratio フィルター

エントリー判定に ATR 絶対値ではなく ATR Ratio（現在 ATR / 過去 1 日平均 ATR）を使用する。相場のボラティリティが過去平均に対して十分な水準にある場合のみエントリーする（閾値 0.8）。

### Two-Brain アーキテクチャ

Long と Short を完全に独立した 2 つのモデルとして扱う。M1 が方向を判定し、M2 がそのシグナルの信頼性を判定する 2 段構成。

### 未来情報リーク排除

トリプルバリアラベリングは Tick データの時刻情報を使用し、バーの OHLC では判定できない「ヒゲの中のどちらが先か」問題を完全に解決する。

---

## 🗂️ データ構造（7 層ストラタム）

```
data/XAUUSD/
│
├── stratum_0_raw/                    生 Tick データ（MT5 ダウンロード元）
│
├── stratum_1_base/                   クリーニング・整形済みデータ
│   ├── master_tick_raw.parquet       単一 Parquet（全期間 Tick）
│   ├── master_tick_partitioned/      Hive パーティション（year/month/day）
│   ├── master_multitimeframe/        15 時間足 OHLCV
│   └── master_processed/            特徴量付加・型統一済み（timeframe= で分割）
│
├── stratum_2_features/               Chapter1 生成の全特徴量
│   ├── feature_value_a_vast_universeA/  基礎統計・ロバスト統計
│   ├── feature_value_a_vast_universeB/  時系列・分布・回帰
│   ├── feature_value_a_vast_universeC/  テクニカル指標（ATR Ratio 化済み）
│   ├── feature_value_a_vast_universeD/  出来高・価格アクション
│   ├── feature_value_a_vast_universeE/  信号処理・周波数解析
│   └── feature_value_a_vast_universeF/  学際的・実験的特徴量
│   └── stratum_2_features_validated/    Chapter2 フィルター通過後の有効特徴量
│
├── stratum_3_artifacts/              Chapter2〜3 の中間成果物・特徴量リスト
│   ├── final_feature_set.txt         Chapter2 完了時点の全特徴量リスト（1722件）
│   ├── final_feature_set_v5.txt      Chapter3 学習用最終特徴量リスト
│   ├── selected_features_v5/         1周目学習後 Gain>0 特徴量リスト
│   ├── selected_features_purified_v5/ 2周目学習後 Gain>0 特徴量リスト（本番用）
│   ├── concurrency_results.parquet_v2 並行数計算結果
│   └── optuna_results/               Optuna 最適化 CSV（スプレッド別）
│
├── stratum_5_alpha/
│   └── neutralized_alpha_set_partitioned/  OLS 純化済み特徴量（Chapter2 出力）
│
├── stratum_6_training/               Chapter3 ラベリング・重み付け済みデータ
│   ├── labeled_dataset_partitioned_v2/    トリプルバリアラベル付きデータ
│   ├── labeled_dataset_monthly_v2/        月次集約版（DuckDB 処理用）
│   └── weighted_dataset_partitioned_v2/   サンプル一意性重み付け済み（学習用）
│
└── stratum_7_models/                 学習済みモデル・OOF 予測
    ├── m1_model_v2_long.pkl          M1 Long モデル（本番用・2周目）
    ├── m1_model_v2_short.pkl         M1 Short モデル（本番用・2周目）
    ├── m2_model_v2_long.pkl          M2 Long モデル（本番用・2周目）
    ├── m2_model_v2_short.pkl         M2 Short モデル（本番用・2周目）
    ├── m1_oof_predictions_long.parquet
    ├── m1_oof_predictions_short.parquet
    └── model_performance_report_long/short.json
```

---

## 🔄 処理フロー（Chapter1〜4）

### Chapter 1：データ取り込み・特徴量生成

```
s1_0_X_filter.py          → Tick フィルタリング・クリーニング
s1_1_A_ingest.py          → Tick → Parquet 変換・Hive パーティション化
s1_1_B_build_ohlcv.py     → 15 時間足 OHLCV 生成
s1_1_C_enrich.py          → S1_PROCESSED 生成（型統一・ローリング計算準備）
engine_1_A〜1_F.py        → 6 カテゴリ・1722 特徴量をタイムフレーム別に生成
                              A: 基礎統計    B: 時系列      C: テクニカル
                              D: 出来高      E: 信号処理    F: 学際的
```

**設計原則**：
- 固定箱（時間足単位の一括更新）を廃止。M1 バーが更新されるたびに全特徴量をローリング再計算。
- D1・W1 特徴量も M1 粒度で存在。全時間足のサンプル数は 1:1。
- 全特徴量をスケール不変（ATR13 割り・%・比率）に変換。価格絶対値は出力しない。

---

### Chapter 2：特徴量フィルタリング・アルファ純化

```
2_A KS 安定性フィルター      → 分布変化が大きい特徴量を除去
2_B 相関フィルター           → 冗長な特徴量を除去（HF 優先）
2_C LF シグナルスコア生成    → 低周波特徴量の環境スコア生成
2_E HF メタモデル学習        → HF 特徴量の重要度スコア算出
2_F 特徴量集約               → LF + HF を統合
2_G OLS アルファ純化         → 市場プロキシ（M5 リターン）との OLS 回帰で
                               ベータ除去・純粋アルファを S5 に出力
```

**出力**：S5_NEUTRALIZED_ALPHA_SET（1722 特徴量・M1 粒度・全期間）

---

### Chapter 3：ラベリング・モデル学習

```
optuna_cv_pure_atr.py               → バリアパラメータ最適化（Long）
optuna_cv_short_pure_atr.py         → バリアパラメータ最適化（Short）
create_proxy_labels_Universal_Brain.py → トリプルバリアラベル生成 → S6_LABELED
aggregate_daily_to_monthly.py        → 日次 → 月次集約
sample_uniqueness_weighting_calculate.py → 並行数計算
sample_uniqueness_weighting_join.py  → 一意性重み付け → S6_WEIGHTED
update_feature_list_v5.py           → S3_FEATURES_FOR_TRAINING_V5 生成

【1周目】
model_training_metalabeling_A.py    → M1 OOF 予測生成
model_training_metalabeling_B.py    → M2 学習データ生成
model_training_metalabeling_C.py    → M1/M2 モデル学習・保存
analyze_importance_v5.py            → Gain>0 特徴量リスト生成（selected_features_v5）

【2周目・本番モデル】
model_training_metalabeling_Ax_purified.py → M1 OOF 予測再生成
model_training_metalabeling_Bx_purified.py → M2 学習データ再生成
model_training_metalabeling_Cx_purified.py → M1/M2 本番モデル学習・上書き保存
analyze_importance_purified.py             → 本番用特徴量リスト生成
                                             → selected_features_purified_v5/
```

**V1.0 確定パラメータ（Optuna 実証）**：
- 時間足：M3 単体
- ATR Ratio 閾値：0.8
- PT 倍率：1.0 / SL 倍率：5.0 / TD：30 分
- スプレッドコスト（ラベリング基準）：0.50 ドル

---

### Chapter 4：リアルタイム本番稼働

```
main.py                          → 統合制御・取引ループ
mql5_bridge_publisher.py         → ZMQ 通信（MT5 ↔ Python）
realtime_feature_engine.py       → 特徴量計算オーケストレーター
realtime_feature_engine_1A〜1F.py → リアルタイム特徴量計算（バッチ版と完全一致）
extreme_risk_engine.py           → ロット計算・SL/TP 計算・ATR Ratio フィルター
state_manager.py                 → 状態管理・チェックポイント・イベントソーシング
ProjectForgeReceiver.mq5         → MT5 EA（ZMQ ブリッジ・発注実行）
risk_config.json                 → バリア設定・リスクパラメータ（ホットリロード対応）
```

**取引ループ（M1 バー確定ごと）**：

```
① スナップショット保存（15 分間隔）
② risk_config.json ホットリロード検知
③ TO タイムアウト監視・強制決済
④ サイレントクローズ捕捉
⑤ ブローカー状態同期
--- M1 バー確定 ---
⑥ 市場プロキシ更新
⑦ RealtimeFeatureEngine → シグナルリスト取得
⑧ Two-Brain 推論（Long M1→M2 / Short M1→M2）
⑨ Delta フィルター・閾値フィルター
⑩ 4 段防衛線チェック
⑪ リスクエンジン → 発注コマンド生成
⑫ ZMQ 発注 → ACK 確認 → 状態同期
```

---

## ⚙️ risk_config.json

```json
{
  "base_capital": 1000.0,
  "lot_per_base": 0.1,
  "max_lot_absolute": 200.0,
  "contract_size": 100.0,
  "min_lot_size": 0.01,
  "base_leverage": 2000.0,
  "sl_multiplier_long": 5.0,
  "pt_multiplier_long": 1.0,
  "sl_multiplier_short": 5.0,
  "pt_multiplier_short": 1.0,
  "td_minutes_long": 30.0,
  "td_minutes_short": 30.0,
  "m2_proba_threshold": 0.30,
  "m2_delta_threshold": 0.30,
  "max_consecutive_sl": 2,
  "cooldown_minutes_after_sl": 30,
  "spread_pips": 36.0,
  "value_per_pip": 1.0,
  "prevent_simultaneous_orders": true,
  "max_drawdown": 10.0,
  "max_positions": 100,
  "min_atr_threshold": 0.8,
  "use_fixed_risk": true,
  "fixed_risk_percent": 0.05,
  "max_allowed_spread": 50.0,
  "margin_call_percent": 100.0,
  "stop_out_percent": 20.0
}
```

`min_atr_threshold` は ATR 絶対値ではなく ATR Ratio の閾値（現在 ATR / 過去 1 日平均 ATR）。
`spread_pips` はロット計算のスプレッドコスト見積もり用。実際のスプレッドに合わせて調整する（通常 28〜36）。

---

## 🛠️ 開発環境・システム要件

### ホストマシン（PinoPino）

| 項目 | 仕様 |
|---|---|
| OS | Windows 11 Pro (64-bit) |
| CPU | Intel Core i7-8700K @ 3.70GHz |
| RAM | 64 GB |
| GPU | NVIDIA GeForce RTX 3060 |
| マザーボード | ASUSTeK TUF Z390M-PRO GAMING |
| ストレージ | BIWIN NV7400 4TB NVMe SSD（X ドライブ・本システム格納）<br>CT1000P310SSD8 1.0TB NVMe SSD<br>WDC WDS250G2B0A 250GB SSD<br>Hitachi HDT721010SLA360 1.0TB HDD |
| ネットワーク | Realtek 8812BU Wireless LAN 802.11ac |

### 実行環境

VS Code + Docker コンテナで開発・実行する。

| コンテナ名 | 用途 |
|---|---|
| hephaestus-zero | メイン開発・学習・バックテスト（常時起動） |
| hermes_talaria | 予備・実験用 |

- イメージ：`vsc-project_forge-82f43...`
- Chapter1〜3 の全処理はコンテナ内で実行
- Chapter4 本番稼働もコンテナ内から ZMQ で MT5（Windows 側）と通信

### 主要ライブラリ（CPU 環境）

```
# データ処理
polars, pyarrow, duckdb, dask[distributed], numpy<2.0

# 機械学習
lightgbm, scikit-learn, xgboost, optuna, shap

# 統計・時系列
statsmodels, arch, hmmlearn, pmdarima, tslearn

# 特殊解析
MFDFA, nolds, PyCausality, PyWavelets, dcor, EMD-signal, entropy

# インフラ
pyzmq, tqdm, joblib, psutil, pathos

# 開発
jupyterlab>=4.0
```

GPU 環境では上記に加えて `torch`, `torchvision`, `tensorflow` を使用。

---

## 📁 スクリプト配置

```
/workspace/
├── blueprint.py              パス・定数の一元管理
├── risk_config.json          リスクパラメータ（ホットリロード対応）
│
├── scripts/                  Chapter1・2 前処理スクリプト
├── features/                 特徴量生成エンジン（1A〜1F）
├── models/                   Chapter3 学習スクリプト
├── execution/                Chapter4 本番稼働スクリプト
│
├── data/XAUUSD/              7 層ストラタム（上記参照）
├── logs/                     システムログ
└── state/                    チェックポイント・イベントログ
```

---

## 🚨 重要な注意事項

### e1c_atr_13 の使用禁止

`e1c_atr_13` は `ATR / ATR_13 ≈ 1.0` の相対値であり ATR 絶対値ではない。
バリア計算・リスクエンジンへの入力には必ず S1_PROCESSED の OHLCV から Wilder 平滑化で自前計算した ATR 絶対値を使用すること。
`e1c_atr_13` をリスクエンジンに渡すと SL 幅が数ドルになり即座に損切りされ続ける。

### 学習時とリアルタイムの完全一致原則

Chapter1 バッチ版（engine_1_A〜1_F）とリアルタイム版（realtime_feature_engine_1A〜1F）は特徴量名・計算式・スケール不変化処理・ddof が完全に一致していなければならない。
どちらか一方を改訂した場合は必ずもう一方も同期すること。

### blueprint.py

全スクリプト共通のパス定数・設定定数を一元管理。
スクリプトは全て `from blueprint import ...` でパスを取得する。
パスのハードコード（`/workspace` 直書き等）は禁止。

```python
# 主要定数
SYMBOL = "XAUUSD"
BARRIER_ATR_PERIOD = 13
ATR_BASELINE_DAYS = 1  # ATR Ratio のベースライン期間（日）
```

---

## 📊 V1.0 実績（Optuna 最適化結果）

| 指標 | Long | Short |
|---|---|---|
| 対象時間足 | M3 | M3 |
| ATR Ratio 閾値 | 0.8 | 0.8 |
| TD | 30 分 | 30 分 |
| Bets 数（全期間） | 344,767 | 344,767 |
| WIN 率（ラベリング） | 46.2% | 45.8% |
| scale_pos_weight | 1.17 | 1.19 |
| Adjusted PF（spread=0.50） | 1.37 | 1.74 |

WIN 率が 50% を下回るのはクールダウンなしで全シグナルを評価しているためであり正常。
SPW は LightGBM で容易に調整可能な範囲。

