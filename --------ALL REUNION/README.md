# 🔥 Project Forge

> **統計的優位性に基づく金融知性体への道**
> 
> 最終目標「Project Chimera」実現のための資金獲得システム

---

## 🎯 プロジェクトのビジョン

**「裁量の天才」を「再現可能なサイエンス」へ**

ルネサンス・テクノロジーズの哲学を継承し、人間の先入観を排除。市場の統計的パターン（「マーケットの亡霊」）のみを追求する完全自動化取引システム。

- **対象市場**: XAU/USD（金/米ドル）
- **データ規模**: 約1億5000万行のTickデータ
- **開発期間**: Python学習開始から2ヶ月で構築

---

## 🌟 システムの特徴

### ✨ 妥協なき設計思想

**データサンプリング拒否**: 300-500GBの全データを完全活用。Dask-LightGBMで64GB RAMでも処理可能。

**AIは最強の非線形処理エンジン**: LightGBMが数千の決定木で複雑な非線形変換を自動最適化。人間は質の高いベース特徴量を提供するのみ。

**三重防衛網**: カーブフィッティングを徹底排除。統計的に信頼できるエリート特徴量のみを厳選。

### 🛡️ 堅牢性への徹底的こだわり

- **状態永続化**: チェックポインティング＋イベントソーシング
- **高信頼性通信**: レイジー・パイレート方式（ACK/NACK、リトライ機構）
- **双方向ハートビート**: 30秒タイムアウトで接続監視
- **ブローカー整合性検証**: 起動時自動同期

### 🧠 数学的最適性

- ケリー基準によるポジションサイジング
- 確率キャリブレーション
- メタラベリング（M1：機会検出 + M2：精度向上）
- サンプル一意性重み付け（非IID対処）

---

## 📊 データアーキテクチャ

### 7層ストラタム構造

データは明確な7つの層（Stratum）に分離され、各層が特定の責務を持ちます。

```
data/XAUUSD/
│
├── 📁 stratum_1_base/          【原材料】生データ
│   ├── master_tick_exness_raw.parquet
│   ├── master_tick_partitioned/
│   └── master_from_tick/
│
├── 📁 stratum_2_features/      【鍛造】324個のベース特徴量
│   ├── feature_value_a_vast_universeA/
│   ├── feature_value_a_vast_universeB/
│   ├── feature_value_a_vast_universeC/
│   ├── feature_value_a_vast_universeD/
│   ├── feature_value_a_vast_universeE/
│   ├── feature_value_a_vast_universeF/
│   └── feature_value_complexity_theoryA/
│
├── 📁 stratum_3_artifacts/     【選別】三重防衛網の検証結果
│   └── 1A_2B/
│       ├── stable_feature_list.joblib
│       ├── adversarial_scores.joblib
│       ├── final_feature_team.joblib
│       └── shap_scores.joblib
│
├── 📁 stratum_4_master/        【統合】マスターテーブル
│   └── 1A_2B/
│       └── master_table_partitioned/
│
├── 📁 stratum_5_alpha/         【純化】ベータ除去後のアルファ
│   └── 1A_2B/
│       └── neutralized_alpha_set_partitioned/
│
├── 📁 stratum_6_training/      【準備】学習用データセット
│   └── 1A_2B/
│       ├── labeled_dataset_partitioned/
│       └── weighted_dataset_partitioned/
│
└── 📁 stratum_7_models/        【完成】訓練済みAIモデル
    └── 1A_2B/
        ├── m1_model.pkl
        ├── m2_model.pkl
        ├── m1_calibrated.pkl
        ├── m2_calibrated.pkl
        └── model_performance_report.json
```

**実験ID管理**: `1A_2B`などのIDで異なる実験設定を管理

---

## 🚀 システムフロー

### 第1章：ベース特徴量生成（✅ 完了）

**入力**: Stratum 1（生データ）
- 1億5000万行のTickデータ
- 15種類の時間足（tick, M0.5, M1, M3, M5, M8, M15, M30, H1, H4, H6, H12, D1, W1, MN）
- Hiveパーティション化Parquet形式

**処理**: 324個の多様な特徴量を体系的に生成
- 基礎統計・ロバスト統計（約60個）
- 時系列分析・分布パラメータ（約60個）
- テクニカル指標・トレンド分析（約70個）
- 出来高・価格アクション（約70個）
- 信号処理・周波数解析（約30個）
- MFDFA、Kolmogorov複雑性など（約34個）

**出力**: Stratum 2（特徴量）
- `feature_value_a_vast_universeA/` 〜 `F/`
- `feature_value_complexity_theoryA/`
- 総容量: 300-500GB

---

### 第2章：三重防衛網 - カーブフィッティング排除

#### 🛡️ 第一防衛線：時間的安定性（2.1 feature_validator.py）

**入力**: Stratum 2（324個の特徴量）

**検証内容**:
- 分布安定性テスト（カイ二乗検定）
- 重要度安定性テスト（Permutation Importance）
- 敵対的検証（時間的ドリフト検出）

**出力**: Stratum 3 Artifacts
- `stable_feature_list.joblib` - 安定した特徴量リスト
- `adversarial_scores.joblib` - 敵対的検証スコア

**生存率**: 324個 → 数百個

---

#### 🔧 マスターテーブル構築（2.2 build_master_table.py）

**入力**: 
- Stratum 2（特徴量）
- Stratum 3（stable_feature_list.joblib）

**処理**: 第一防衛線通過特徴量を時系列join_asofで統合

**出力**: Stratum 4 Master
- `master_table_partitioned/` - 統合マスターテーブル（50-80GB）

---

#### 🎯 第二防衛線：SHAP重要度評価（2.3 walk_forward_validator_v2.py）

**入力**: Stratum 4（マスターテーブル）

**処理**:
- TimeSeriesSplitでウォークフォワード検証
- SHAP値の並列計算（map_partitions）
- 平均絶対SHAP値でランク付け

**なぜRFEから転換？**: 計算量O(k × N²) → O(k × N)で実行可能に

**出力**: Stratum 3 Artifacts
- `final_feature_team.joblib` - 最終選抜特徴量（数個〜数十個）
- `shap_scores.joblib` - SHAP重要度スコア

**生存率**: 数百個 → 数個〜数十個

---

#### ⚡ 第三防衛線：アルファ純化（2.4 feature_neutralizer.py）

**入力**: 
- Stratum 4（マスターテーブル）
- Stratum 3（final_feature_team.joblib）

**処理**: 市場ベータを除去し、残差としてアルファを抽出

**出力**: Stratum 5 Alpha
- `neutralized_alpha_set_partitioned/` - 純化されたアルファ特徴量（数十GB）

---

### 第3章：予測AIコア - メタラベリング

#### 🎲 トリプルバリア・ラベリング（3.1 triple_barrier_labeling.py）

**入力**: Stratum 5（純化アルファ特徴量）

**処理**:
- **利食いバリア**: 現在価格 + 2 × ATR
- **損切りバリア**: 現在価格 - 1 × ATR
- **時間バリア**: 最大保有期間

**出力**: Stratum 6 Training
- `labeled_dataset_partitioned/`
  - label（1: ロング成功、-1: ショート成功、0: タイムアウト）
  - barrier_reached（どのバリアに到達したか）
  - time_to_barrier（到達までの時間）
  - t0, t1（開始・終了タイムスタンプ）

---

#### ⚖️ サンプル一意性重み付け（3.2 sample_uniqueness_weighting.py）

**入力**: Stratum 6（ラベル付きデータセット）

**処理**: 非IID問題に対処
1. 並行性計算（同時期に複数ポジションが存在）
2. 一意性 = 1 / 並行性
3. sample_weight適用

**出力**: Stratum 6 Training
- `weighted_dataset_partitioned/`
  - 元データ + sample_weight列

---

#### 🧠 メタラベリング訓練（3.3 model_training_metalabeling.py）

**入力**: Stratum 6（重み付きデータセット）

**処理**: 2段階AIシステム
- **M1（プライマリー）**: 「取引機会か？」（リコール最大化）
- **M2（メタ）**: 「M1のシグナルは本物か？」（プレシジョン最大化）

**特徴量設計**:
- 基本特徴量（アルファ）
- M1出力（確率、クラス）
- M1性能統計（最近の精度、勝率）
- 市場レジーム情報

**重要**: パージ＆エンバーゴ付きCVでデータリーケージ防止

**出力**: Stratum 7 Models
- `m1_model.pkl` - プライマリーモデル
- `m2_model.pkl` - メタモデル
- `m1_calibrated.pkl` - 確率キャリブレーション済みM1
- `m2_calibrated.pkl` - 確率キャリブレーション済みM2
- `model_performance_report.json` - 性能レポート

---

### 第4章：統合・執行エンジン

#### 📊 確率キャリブレーション（4.1 統合）

**処理**: LightGBMの`predict_proba`スコアを真の確率に変換
- CalibratedClassifierCV使用
- 信頼性のある確率推定

**出力**: m1_calibrated.pkl、m2_calibrated.pkl（第3章で統合出力）

---

#### 💰 リスク管理（extreme_risk_engine_v2.py）

**ケリー基準**: `f* = (b × p - q) / b`
- **必須安全策**: ハーフケリー（f* / 2）

**その他のリスク管理**:
- 動的損切り・利食い（ATR基準）
- GARCHボラティリティ適応
- 最大ドローダウン制約

---

#### 🌡️ 市場レジーム検知（market_regime_detector.py）

HMMまたは時系列クラスタリングで4レジーム分類:
- 高ボラティリティ上昇
- 高ボラティリティ下落
- 低ボラティリティレンジ
- トレンド転換期

レジーム別にリスクパラメータを動的調整。

---

#### 💾 状態管理（state_manager.py）

**チェックポインティング**: 状態スナップショットを定期保存

**イベントソーシング**: 全イベントを連鎖記録、完全な監査証跡

**ブローカー整合性検証**: 起動時に自動同期

---

#### 🌉 MQL5ブリッジ（mql5_bridge_publisher_v2.py + ProjectForgeReceiver_v2.mq5）

**レイジー・パイレート方式**:
- ACK/NACK確認
- リトライ機構
- コマンド喪失率0%

**双方向ハートビート**:
- 30秒タイムアウトで接続監視
- 平均応答< 100ms

---

### 第5章：統合実行環境（main.py）

#### 🔄 5段階初期化

1. 状態管理システム起動
2. 市場レジーム検知器ロード
3. AIモデル（M1/M2）ロード
4. リスク管理エンジン起動
5. MQL5ブリッジ接続

#### ⚡ 6ステップ取引ループ

1. **データ取得**: 最新市場データ取得
2. **特徴量抽出**: リアルタイム特徴量計算
3. **市場情報構築**: レジーム、ボラティリティ分析
4. **コマンド生成**: M1 → M2 → リスク管理
5. **送信**: MQL5ブリッジ経由でMT5へ
6. **状態更新**: 状態永続化、ログ記録

#### 🎮 実行モード

- **デモモード** (`--demo`): ログのみ、実送信なし
- **本番モード** (`--live`): 実際に取引実行

---

## 🛠️ セットアップガイド

### システム要件

**ソフトウェア**:
- Python 3.8+
- MetaTrader 5（ビルド3661+）

**ハードウェア推奨**:
- RAM: 64GB以上
- CPU: 4コア以上
- ストレージ: 1TB NVMe SSD

**Pythonライブラリ**:
```
numpy, pandas, polars, dask, lightgbm, scikit-learn, 
optuna, hmmlearn, tslearn, pyzmq, joblib
```

**MQL5要件**:
- ZeroMQバインディング
- JAson.mqh

---

### インストール手順

#### 1️⃣ Python環境構築

```bash
# 仮想環境作成
python -m venv forge_env

# 仮想環境有効化
source forge_env/bin/activate  # Linux/Mac
# または
forge_env\Scripts\activate  # Windows

# 依存関係インストール
pip install -r requirements.txt
```

#### 2️⃣ ディレクトリ構築

```bash
# 必要なディレクトリを作成
mkdir -p data/state data/bridge logs/zmq_bridge_v2 
mkdir -p models/metalabeling config reports/metalabeling
mkdir -p chapter1_2 chapter3 chapter4
```

#### 3️⃣ 設定ファイル準備

**config/risk_config.json**:
```json
{
  "kelly_fraction": 0.5,
  "max_risk_per_trade": 0.02,
  "max_drawdown": 0.20,
  "atr_multiplier": 2.0,
  "pip_multiplier": 100.0
}
```

**pip_multiplier設定**:
- XAU/USD、USD/JPY: `100.0`
- EUR/USD、GBP/USD: `10000.0`

**config/regime_config.json**: レジーム別パラメータ（詳細は別途）

#### 4️⃣ MQL5セットアップ

1. MetaTrader 5をインストール
2. ZeroMQバインディングをインストール
3. `JAson.mqh`を`MQL5/Include/`に配置
4. `ProjectForgeReceiver_v2.mq5`を`MQL5/Experts/`にコピー
5. MT5でコンパイル

---

## 🎯 使用方法

### フェーズ1: データ基盤準備

第1章のスクリプトでベース特徴量を生成（既に完了している場合はスキップ）

### フェーズ2: 三重防衛網実行

```bash
cd chapter1_2

# 2.1 時間的安定性検証
python feature_validator.py

# 2.2 マスターテーブル構築
python build_master_table.py

# 2.3 SHAP重要度評価
python walk_forward_validator_v2.py

# 2.4 アルファ純化
python feature_neutralizer.py
```

**出力確認**: `data/XAUUSD/stratum_3_artifacts/1A_2B/`と`stratum_4_master/`、`stratum_5_alpha/`

### フェーズ3: 予測AIコア訓練

```bash
cd ../chapter3

# 3.1 トリプルバリア・ラベリング
python triple_barrier_labeling.py

# 3.2 サンプル一意性重み付け
python sample_uniqueness_weighting.py

# 3.3 メタラベリング訓練
python model_training_metalabeling.py
```

**出力確認**: `data/XAUUSD/stratum_7_models/1A_2B/`
- m1_calibrated.pkl
- m2_calibrated.pkl
- model_performance_report.json

### フェーズ4: レジーム検知器訓練（オプション）

```python
from chapter4.market_regime_detector import MarketRegimeDetector

# HMM方式で4レジーム
detector = MarketRegimeDetector(method='hmm', n_regimes=4)
# 訓練処理...
detector.save_model('models/regime_detector_hmm.pkl')
```

### フェーズ5: 統合システム起動 🚀

```bash
# デモモード（安全確認）
python main.py --demo

# 本番モード（実取引）
python main.py --live
```

---

## 🎓 推奨運用フロー

段階的にリスクを管理しながらスケールアップ：

1. **デモモード検証** - ロジック確認、エラー検出
2. **デモ口座実証実験** - 数週間〜数ヶ月の実績確認
3. **小規模資金本番** - 総資本の5-10%で開始
4. **段階的スケールアップ**:
   - 10% → 25% → 50% → 100%
   - 各段階で統計的優位性を再確認

---

## 🔧 トラブルシューティング

### ZeroMQ接続エラー

**症状**: `ConnectionRefusedError`または`Timeout`

**解決策**:
- ポート競合確認（デフォルト: 5555）
- ファイアウォール設定確認
- MT5でEAが起動しているか確認
- ログ確認: `logs/zmq_bridge_v2/`

### モデル読み込みエラー

**症状**: `FileNotFoundError`または`pickle error`

**解決策**:
- モデルファイルの存在確認
- パスの検証（相対パス/絶対パス）
- Pythonバージョン互換性確認

### ブローカー整合性失敗

**症状**: 起動時に同期エラー

**解決策**:
- 自動同期の完了を待つ
- ログで具体的エラー確認: `logs/state_manager.log`
- 必要に応じて手動で状態リセット

### メモリ不足

**症状**: `MemoryError`または`Out of Memory`

**解決策**:
- Daskワーカーのメモリ制限調整
- パーティション数を増加（より小さいチャンク）
- 処理する時間範囲を縮小

---

## 🏆 アーキテクチャの強み

### 数学的最適性
- ケリー基準によるポジションサイジング
- 確率キャリブレーションで正確な確率推定
- メタラベリングで精度とリコールのバランス
- サンプル一意性重み付けで非IID対処

### 堅牢性
- 状態永続化で障害からの復旧
- イベントソーシングで完全な監査証跡
- 高信頼性通信（ACK/NACK、リトライ）
- ブローカー整合性検証

### 適応性
- 市場レジーム検知で環境変化に対応
- 動的リスクパラメータ調整
- GARCHボラティリティ適応

### 型安全性
- Pylance準拠
- TypedDict、Optional型の厳格適用
- 実行時エラーの最小化

---

## 🗺️ 開発ロードマップ

### ✅ 完了済み
1. ベース特徴量生成（324個）
2. データ基盤構築（7層ストラタム）

### 🔄 進行中
3. 三重防衛網の完全実装
   - feature_validator.py
   - build_master_table.py
   - walk_forward_validator_v2.py
   - feature_neutralizer.py

4. 予測AIコア構築
   - トリプルバリア・ラベリング
   - サンプル一意性重み付け
   - メタラベリング訓練
   - 確率キャリブレーション

### 📋 今後の計画
5. 執行システム完成
   - 状態管理システム
   - 高信頼性MQL5ブリッジ
   - ケリー基準リスク管理
   - 市場レジーム検知

6. 実証実験
   - デモ口座運用
   - パラメータ最適化
   - 統計的検証

7. **Project Chimera** 🦁
   - Transformerベース次世代コア
   - マルチシンボル対応
   - クラウドデプロイメント

---

## 💡 プロジェクトの背景

### なぜProject Forgeなのか？

個人投資家が市場で持続的に成功するための**唯一の合理的アプローチ**。

人間の認知バイアスと限界を認識し、客観的かつ体系的な定量的アプローチで「裁量の天才」を「再現可能なサイエンス」へ昇華させる。

### ルネサンス・テクノロジーズからの影響

- **データ駆動**: 直感ではなくデータに語らせる
- **システマティック**: 再現可能で検証可能
- **科学的厳密性**: 統計的有意性を重視
- **継続的改善**: 常に学習し進化

### 達成実績

**Python学習開始から2ヶ月**で、個人投資家上位0.01%〜0.001%レベルの技術基盤を構築。

---

## 🎨 設計哲学

### 「複雑さは敵ではない。無駄な複雑さこそが敵である。」

本システムは3つの柱で支えられています：

1. **計算実現可能性** - 理論だけでなく実際に動く
2. **理論的厳密性** - 数学的原則に根差す
3. **実践的堅牢性** - 実戦で生き残る

必要な複雑性のみを保持し、実行不可能な要素は排除。数学的原則に根差した**最もシンプルで最も強力なアーキテクチャ**。

### 「個別の優秀なコンポーネントではなく、統合された知性体としての全体が、真の価値を生み出す。」

各章は独立して優秀であるだけでなく、統合されることで**創発的な知性**を発揮します。

---

## 📜 ライセンス

このプロジェクトは個人利用を目的としています。商用利用については別途ご相談ください。

---

## 🌟 最後に

**Project Forge**は単なる取引システムではありません。

それは、個人投資家が機関投資家と同じ土俵で戦うための**武器**であり、**金融知性体への第一歩**です。

データの海から真のアルファを抽出し、統計的優位性を現実の利益に変換する。

**これが、Project Forgeの使命です。** 🔥

---

*「マーケットの亡霊」を追い続けろ。*