# Project Forge

## プロジェクト概要

統計的優位性に基づく金融取引システム。最終目標「Project Chimera（金融知性体）」実現のための資金獲得を目的とする。

思想的支柱: ルネサンス・テクノロジーズの哲学 - 人間の先入観を排除し、市場の統計的パターン（「マーケットの亡霊」）のみを追求。

**対象市場**: XAU/USD（金/米ドル）主戦場

---

## 基本哲学

### AIは最強の非線形処理エンジン

LightGBMは内部で数千の決定木を構築し、複雑な非線形変換を自動最適化する。手動での非線形変換（交互作用項、多項式）は不要。

**我々の役割**:
1. 多様で質の高いベース特徴量を生成（324個）
2. 統計的に信頼できる特徴量のみを厳選（三重防衛網）
3. LightGBMに最適な戦術を学習させる
4. AIの判断を堅牢なリスク管理で実行

### データサンプリング拒否

300-500GBのデータを10-20%にサンプリングする妥協は哲学と矛盾。Dask-LightGBMでヒストグラム統合により全データの文脈を完全保持（実効メモリ20-30GB）。

---

## 入力データ仕様

- **総行数**: 約1億5000万行のTickデータ
- **時間軸**: 15種類の時間足（tick, M0.5, M1, M3, M5, M8, M15, M30, H1, H4, H6, H12, D1, W1, MN）
- **保存形式**: Hiveパーティション化されたParquet形式

---

## システムアーキテクチャ

### 第1章：ベース特徴量の体系的生成（完了済み）

**324個の特徴量**（300-500GB）:
- 基礎統計・ロバスト統計（約60個）
- 時系列分析・分布パラメータ（約60個）
- テクニカル指標・トレンド分析（約70個）
- 出来高・価格アクション（約70個）
- 信号処理・周波数解析（約30個）
- 学際的・実験的特徴量（約16個）
- MFDFA（9個）、Kolmogorov複雑性（9個）

**データ**: 1.5億行Tickデータ、15種類の時間足、Hiveパーティション化Parquet形式

### 第2章：カーブフィッティング三重防衛網 2.0

統計的に信頼できるエリート特徴量のみを選抜。

**Out-of-Core処理**: Dask-LightGBMで300-500GBを64GB RAMで処理可能。

#### 第一防衛線：時間的安定性（feature_validator.py）

- 分布安定性テスト（カイ二乗検定）
- 重要度安定性テスト（Permutation Importance）
- 敵対的検証（時間的ドリフト検出）

生存率: 全特徴量 → 数百個

#### マスターテーブル構築（build_master_table.py）

第一防衛線通過特徴量をjoin_asofで統合（50-80GB）。

#### 第二防衛線：SHAP重要度評価（walk_forward_validator_v2.py）

**RFEからの転換**: 計算量O(k × N²) → O(k × N)、実行不可能 → 実行可能。

- TimeSeriesSplitでウォークフォワード検証
- SHAP値の並列計算（map_partitions）
- 平均絶対SHAP値でランク付け

生存率: 数百個 → 数個〜数十個

#### 第三防衛線：アルファ純化（feature_neutralizer.py）

市場ベータ除去、残差としてアルファ抽出（数十GB）。

### 第3章：予測AIコア 2.0 - メタラベリング

#### トリプルバリア・ラベリング（triple_barrier_labeling.py）

- 利食いバリア: 現在価格 + 2 × ATR
- 損切りバリア: 現在価格 - 1 × ATR
- 時間バリア: 最大保有期間

出力: label, barrier_reached, time_to_barrier, t0, t1

#### サンプル一意性重み付け（sample_uniqueness_weighting.py）

非IID対処。並行性計算 → 一意性 = 1 / 並行性 → sample_weight適用。

#### メタラベリング訓練（model_training_metalabeling.py）

**M1（プライマリー）**: 「取引機会か？」（リコール最大化）  
**M2（メタ）**: 「M1シグナルは本物か？」（プレシジョン最大化）

特徴量設計: 基本特徴量 + M1出力 + M1性能統計 + 市場レジーム

**パージ＆エンバーゴ付きCV**: データリーケージ防止。

出力: M1.pkl, M2.pkl, 性能レポート.json

### 第4章：統合・執行エンジン 2.0

#### 確率キャリブレーション

LightGBMのpredict_probaスコアを真の確率に変換。CalibratedClassifierCV使用。

出力: M1_calibrated.pkl, M2_calibrated.pkl

#### リスク管理（extreme_risk_engine_v2.py）

**ケリー基準**: f* = (b × p - q) / b  
必須安全策: ハーフケリー（f* / 2）

その他: 動的損切り・利食い、GARCHボラティリティ適応、最大ドローダウン制約。

#### 市場レジーム検知（market_regime_detector.py）

HMMまたは時系列クラスタリングで4レジーム分類。レジーム別リスクパラメータ動的調整。

#### 状態管理（state_manager.py）

- チェックポインティング: 状態スナップショット定期保存
- イベントソーシング: イベント連鎖記録、完全監査証跡
- ブローカー整合性検証: 起動時自動同期

#### MQL5ブリッジ（mql5_bridge_publisher_v2.py, ProjectForgeReceiver_v2.mq5）

- レイジー・パイレート: ACK/NACK確認、リトライ機構
- 双方向ハートビート: 30秒タイムアウトで接続監視

信頼性: コマンド喪失率0%、平均応答< 100ms

### 第5章：統合実行環境

#### 統合実行（main.py）

**5段階初期化**: 状態管理 → レジーム検知 → AIモデル → リスク管理 → MQL5ブリッジ

**6ステップ取引ループ**: データ取得 → 特徴量抽出 → 市場情報構築 → コマンド生成 → 送信 → 状態更新

**実行モード**: デモ（ログのみ）/ 本番（実送信）

---

## システム要件

**Python**: 3.8+、numpy, pandas, polars, dask, lightgbm, scikit-learn, optuna, hmmlearn, tslearn, pyzmq, joblib

**MQL5**: ZeroMQバインディング、JAson.mqh

**ハードウェア**: 64GB RAM推奨、4コア以上CPU、1TB NVMe SSD

**その他**: MetaTrader 5（ビルド3661+）

---

## インストール

### 1. Python環境

```bash
python -m venv forge_env
source forge_env/bin/activate
pip install -r requirements.txt
```

### 2. ディレクトリ作成

```bash
mkdir -p data/state data/bridge logs/zmq_bridge_v2 models/metalabeling config reports/metalabeling chapter1_2 chapter3 chapter4
```

### 3. 設定ファイル

`config/risk_config.json`:
```json
{
  "kelly_fraction": 0.5,
  "max_risk_per_trade": 0.02,
  "max_drawdown": 0.20,
  "atr_multiplier": 2.0,
  "pip_multiplier": 100.0
}
```

**pip_multiplier**: XAU/USD・USD/JPY=100.0、EUR/USD・GBP/USD=10000.0

`config/regime_config.json`: レジーム別パラメータ設定

### 4. MQL5セットアップ

1. MT5インストール
2. ZeroMQバインディングインストール
3. JAson.mqhを`MQL5/Include`に配置
4. `ProjectForgeReceiver_v2.mq5`を`Experts`にコピー、コンパイル

---

## 使用方法

### フェーズ1: データ基盤準備

ベース特徴量生成（第1章スクリプト実行）

### フェーズ2: 三重防衛網

```bash
cd chapter1_2
python feature_validator.py
python build_master_table.py
python walk_forward_validator_v2.py
python feature_neutralizer.py
```

### フェーズ3: 予測AIコア訓練

```bash
cd ../chapter3
python triple_barrier_labeling.py
python sample_uniqueness_weighting.py
python model_training_metalabeling.py
```

出力: `models/metalabeling/M1_calibrated.pkl`, `M2_calibrated.pkl`

### フェーズ4: レジーム検知器訓練（オプション）

```python
from chapter4.market_regime_detector import MarketRegimeDetector
detector = MarketRegimeDetector(method='hmm', n_regimes=4)
# 訓練・保存
detector.save_model('models/regime_detector_hmm.pkl')
```

### フェーズ5: 統合システム起動

```bash
# デモモード
python main.py --demo

# 本番モード
python main.py --live
```

**推奨運用フロー**:
1. デモモード検証
2. デモ口座実証実験
3. 小規模資金本番（総資本5-10%）
4. フルスケール（段階的増加: 10% → 25% → 50% → 100%）

---

## トラブルシューティング

**ZeroMQ接続エラー**: ポート競合確認、ファイアウォール設定、MT5 EA起動確認

**モデル読み込みエラー**: モデルファイル存在確認、パス検証

**ブローカー整合性失敗**: 自動同期実行、ログ確認

**メモリ不足**: Daskワーカーメモリ制限調整、パーティション数増加

---

## アーキテクチャの特徴

**数学的最適性**: ケリー基準、確率キャリブレーション、メタラベリング、サンプル一意性重み付け

**堅牢性**: 状態永続化、イベントソーシング、高信頼性通信、ブローカー整合性検証

**適応性**: 市場レジーム検知、動的リスクパラメータ、GARCHボラティリティ適応

**型安全性**: Pylance準拠、TypedDict、Optional型厳格適用

---

## 開発ロードマップ

1. **三重防衛網完全実装**: feature_validator.py → build_master_table.py → walk_forward_validator_v2.py → feature_neutralizer.py
2. **予測AIコア構築**: トリプルバリア、サンプル一意性、メタラベリング、確率キャリブレーション
3. **執行システム完成**: 状態管理、高信頼性通信、ケリー基準、レジーム検知
4. **実証実験**: デモ口座運用、パラメータ最適化、統計的検証
5. **Project Chimera**: Transformerベース次世代コア、マルチシンボル、クラウドデプロイ

---

## プロジェクト背景

個人投資家が市場で持続的に成功するための唯一の合理的アプローチ。人間の認知バイアスと限界を認識し、客観的かつ体系的な定量的アプローチで「裁量の天才」を「再現可能なサイエンス」へ昇華。

ルネサンス・テクノロジーズの影響: データ駆動、システマティック、科学的厳密性、継続的改善。

Python学習開始から2ヶ月で個人投資家上位0.01%〜0.001%の技術基盤構築達成。

---

## 設計哲学

**「複雑さは敵ではない。無駄な複雑さこそが敵である。」**

計算実現可能性、理論的厳密性、実践的堅牢性の三位一体。必要な複雑性のみ保持、実行不可能要素排除、数学的原則に根差した最もシンプルで最も強力なアーキテクチャ。

**「個別の優秀なコンポーネントではなく、統合された知性体としての全体が、真の価値を生み出す。」**

---

**Project Forge - 金融知性体への道**