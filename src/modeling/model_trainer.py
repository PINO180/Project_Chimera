import pandas as pd
import numpy as np
import joblib
import lightgbm as lgb
from sklearn.metrics import accuracy_score, f1_score, mean_squared_error
from common.logger_setup import logger
import os

# --- 設定 ---
# core/prediction_core.pyが生成したファイルのパス
TRAIN_SET_PATH = 'data/train_set.joblib'
VALIDATION_SET_PATH = 'data/validation_set.joblib'

# 完成したAIモデルの保存先
MODELS_DIR = 'models'
OUTCOME_MODEL_PATH = os.path.join(MODELS_DIR, 'outcome_model.joblib')
DURATION_MODEL_PATH = os.path.join(MODELS_DIR, 'duration_model.joblib')


class ModelTrainer:
    def __init__(self):
        logger.info("--- 🧠 フェーズ3.2: AIモデル訓練パイプラインの構築を開始 ---")
        self.train_set = None
        self.validation_set = None
        self.features = None

    def load_data(self):
        """先行実装で作成した訓練・検証データセットを読み込む"""
        logger.info(f"訓練データセット '{TRAIN_SET_PATH}' を読み込み中...")
        self.train_set = joblib.load(TRAIN_SET_PATH)
        
        logger.info(f"検証データセット '{VALIDATION_SET_PATH}' を読み込み中...")
        self.validation_set = joblib.load(VALIDATION_SET_PATH)
        
        # 特徴量カラムを特定 (outcomeとdurationは除く)
        self.features = [col for col in self.train_set.columns if col not in ['outcome', 'duration']]
        logger.info(f"データ読み込み完了。{len(self.features)}個の特徴量を検出。")

    def train_outcome_model(self):
        """取引結果（勝ち/負け）を予測する分類モデルを訓練する"""
        logger.info("--- 1. 結果予測モデル（分類）の訓練を開始 ---")
        
        X_train = self.train_set[self.features]
        y_train = self.train_set['outcome']
        
        X_val = self.validation_set[self.features]
        y_val = self.validation_set['outcome']

        # LightGBM 分類器モデル
        # クラスの不均衡を考慮する is_unbalance=True を設定
        model = lgb.LGBMClassifier(objective='multiclass', num_class=3, is_unbalance=True, random_state=42)
        
        logger.info(f"訓練データ {len(X_train)}行でモデルを訓練中...")
        model.fit(X_train, y_train,
                  eval_set=[(X_val, y_val)],
                  eval_metric='multi_logloss',
                  callbacks=[lgb.early_stopping(10, verbose=False)])

        logger.info("訓練完了。検証データで性能を評価...")
        y_pred = model.predict(X_val)
        
        accuracy = accuracy_score(y_val, y_pred)
        f1 = f1_score(y_val, y_pred, average='weighted')
        
        logger.info(f"  -> 検証性能: Accuracy = {accuracy:.4f}, Weighted F1-Score = {f1:.4f}")
        
        # モデルを保存
        os.makedirs(MODELS_DIR, exist_ok=True)
        joblib.dump(model, OUTCOME_MODEL_PATH)
        logger.info(f"訓練済みモデルを '{OUTCOME_MODEL_PATH}' に保存しました。")

    def train_duration_model(self):
        """決着までの時間を予測する回帰モデルを訓練する"""
        logger.info("--- 2. 時間予測モデル（回帰）の訓練を開始 ---")
        
        # 時間予測は「勝ち」または「負け」の場合のみ意味を持つため、データをフィルタリング
        train_filtered = self.train_set[self.train_set['outcome'] != 0]
        val_filtered = self.validation_set[self.validation_set['outcome'] != 0]

        X_train = train_filtered[self.features]
        y_train = train_filtered['duration']
        
        X_val = val_filtered[self.features]
        y_val = val_filtered['duration']

        # LightGBM 回帰モデル
        model = lgb.LGBMRegressor(objective='regression_l1', random_state=42)
        
        logger.info(f"訓練データ {len(X_train)}行でモデルを訓練中...")
        model.fit(X_train, y_train,
                  eval_set=[(X_val, y_val)],
                  eval_metric='mae',
                  callbacks=[lgb.early_stopping(10, verbose=False)])

        logger.info("訓練完了。検証データで性能を評価...")
        y_pred = model.predict(X_val)
        
        mae = mean_squared_error(y_val, y_pred, squared=False) # RMSE
        
        logger.info(f"  -> 検証性能: 平均絶対誤差 (MAE) = {mae:.4f} 分")
        
        # モデルを保存
        joblib.dump(model, DURATION_MODEL_PATH)
        logger.info(f"訓練済みモデルを '{DURATION_MODEL_PATH}' に保存しました。")

if __name__ == '__main__':
    trainer = ModelTrainer()
    trainer.load_data()
    trainer.train_outcome_model()
    trainer.train_duration_model()
    logger.info("--- ✅ 全てのAIモデルの訓練が完了しました ---")