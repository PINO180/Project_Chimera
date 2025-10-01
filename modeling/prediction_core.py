import pandas as pd
import numpy as np
import joblib
from numba import njit
from common.logger_setup import logger

# --- 設定 ---
NEUTRALIZED_FEATURES_PATH = 'data/neutralized_feature_set.joblib'
# ATRを読み込むために、純化前の一時的な特徴量ユニバースも参照
TEMP_FEATURE_UNIVERSE_PATH = 'data/xauusd_feature_universe.joblib' 

# 動的トリプルバリア方式のパラメータ
PT_ATR_MULTIPLE = 2.0  # 利益確定ライン = ATRの2.0倍
SL_ATR_MULTIPLE = 1.5  # 損切りライン = ATRの1.5倍
HOLDING_PERIOD = 60      # 最大保有期間 (60分) - まずは固定

# データ分割比率
TRAIN_PCT = 0.70
VALIDATION_PCT = 0.15
# 残りがテストデータ (0.15)

@njit
def _calculate_labels_nb(prices, pt_levels, sl_levels, holding_periods):
    """Numbaで高速化されたトリプルバリア方式のコア計算ロジック"""
    n = len(prices)
    outcomes = np.zeros(n, dtype=np.int8)  # -1: SL, 0: Timeout, 1: PT
    durations = np.full(n, holding_periods, dtype=np.int32)

    for i in range(n - 1):
        pt = pt_levels[i]
        sl = sl_levels[i]
        
        for j in range(1, holding_periods + 1):
            if i + j >= n:
                break
            
            future_price = prices[i + j]
            
            if future_price >= pt:
                outcomes[i] = 1
                durations[i] = j
                break
            elif future_price <= sl:
                outcomes[i] = -1
                durations[i] = j
                break
                
    return outcomes, durations

class PredictionCore:
    def __init__(self):
        logger.info("--- 🧠 フェーズ3.1: 予測AIコア（動的水平バリア版）の構築を開始 ---")
        self.features = None
        self.prices = None
        self.atr = None

    def load_data(self):
        """フェーズ1と2で生成したデータを読み込む"""
        logger.info(f"純化された特徴量セット '{NEUTRALIZED_FEATURES_PATH}' を読み込み中...")
        self.features = joblib.load(NEUTRALIZED_FEATURES_PATH)
        
        logger.info(f"ボラティリティ指標（ATR）を含む特徴量ユニバース '{TEMP_FEATURE_UNIVERSE_PATH}' を読み込み中...")
        full_features = joblib.load(TEMP_FEATURE_UNIVERSE_PATH)
        
        # 必要なデータを抽出し、インデックスを揃える
        self.prices = full_features['close'].loc[self.features.index]
        # 1時間足のATRをボラティリティ指標として使用
        self.atr = full_features['ATR_14_1H'].loc[self.features.index] 
        
        logger.info(f"データ読み込み完了。{len(self.features)}行 x {len(self.features.columns)}特徴量。")

    def create_labels(self):
        """動的トリプルバリア方式で目的変数を生成する"""
        logger.info(f"動的トリプルバリア方式でラベルを生成中 (PT={PT_ATR_MULTIPLE}*ATR, SL={SL_ATR_MULTIPLE}*ATR, Hold={HOLDING_PERIOD}min)...")
        
        close_prices = self.prices.values
        
        # --- ▼▼▼ ここが動的バリアの核心 ▼▼▼ ---
        # 各時点のATRに基づいて、利益確定と損切りラインを動的に計算
        pt_levels = close_prices + (self.atr.values * PT_ATR_MULTIPLE)
        sl_levels = close_prices - (self.atr.values * SL_ATR_MULTIPLE)
        # --- ▲▲▲ ここまで ▲▲▲ ---

        outcomes, durations = _calculate_labels_nb(close_prices, pt_levels, sl_levels, HOLDING_PERIOD)

        self.features['outcome'] = outcomes
        self.features['duration'] = durations
        
        self.features = self.features[self.features['outcome'] != 0] # タイムアウトした取引は学習から除外
        
        logger.info(f"ラベル生成完了。{len(self.features)}行がラベリングされました。")
        # outcomeが1（勝ち）と-1（負け）の比率が極端でないか確認
        if len(self.features) > 0:
            win_loss_ratio = np.sum(self.features['outcome'] == 1) / np.sum(self.features['outcome'] == -1) if np.sum(self.features['outcome'] == -1) > 0 else np.inf
            logger.info(f"勝ち(PT): {np.sum(self.features['outcome'] == 1)}件, 負け(SL): {np.sum(self.features['outcome'] == -1)}件 (Win/Loss Ratio: {win_loss_ratio:.2f})")

    def split_data(self):
        """データを訓練・検証・テスト用に時系列で分割する"""
        logger.info("データを訓練(70%)・検証(15%)・テスト(15%)に分割...")
        n = len(self.features)
        train_end = int(n * TRAIN_PCT)
        val_end = int(n * (TRAIN_PCT + VALIDATION_PCT))
        
        train_set = self.features.iloc[:train_end]
        validation_set = self.features.iloc[train_end:val_end]
        test_set = self.features.iloc[val_end:]
        
        logger.info(f"分割完了: Train={len(train_set)}, Validation={len(validation_set)}, Test={len(test_set)}")
        
        logger.info("分割済みデータセットを'data/'フォルダに保存中...")
        joblib.dump(train_set, 'data/train_set.joblib')
        joblib.dump(validation_set, 'data/validation_set.joblib')
        joblib.dump(test_set, 'data/test_set.joblib')
        logger.info("保存完了。")

if __name__ == '__main__':
    core = PredictionCore()
    core.load_data()
    core.create_labels()
    core.split_data()
    logger.info("--- ✅ データ準備と動的ラベリングの全工程が完了しました ---")