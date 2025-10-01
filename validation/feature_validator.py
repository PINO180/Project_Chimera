import sys
import os
# プロジェクトルートをパスに追加
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from common.logger_setup import logger

import pandas as pd
import numpy as np
import joblib
import lightgbm as lgb
from sklearn.inspection import permutation_importance
from scipy.stats import ks_2samp
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from common.logger_setup import logger
import argparse
import math

class FeatureValidator:
    """
    【v1.6 - バッチ内モデル訓練】
    LightGBMエラーを解決する論理修正版。
    """
    def __init__(self,
                 feature_universe_path: str,
                 n_jobs: int = 1,
                 ks_p_value_threshold: float = 0.05,
                 importance_change_threshold: float = 0.5):
        self.feature_universe_path = feature_universe_path
        self.n_jobs = n_jobs
        self.ks_p_value_threshold = ks_p_value_threshold
        self.importance_change_threshold = importance_change_threshold
        self.df_features = None
        self.feature_columns = None

    def _load_data(self):
        logger.info(f"第一防衛線: 特徴量ユニバース '{self.feature_universe_path}' を読み込んでいます...")
        
        # パスに応じて適切な読み込み方法を選択
        if self.feature_universe_path.endswith('.parquet'):
            self.df_features = pd.read_parquet(self.feature_universe_path)
        else:
            self.df_features = joblib.load(self.feature_universe_path)
            
        logger.info("メモリ効率化のため、データ型を変換しています...")
        for col in tqdm(self.df_features.columns, desc="Data Type Conversion"):
            if self.df_features[col].dtype == 'float64': 
                self.df_features[col] = self.df_features[col].astype('float32')
            if self.df_features[col].dtype == 'int64': 
                self.df_features[col] = self.df_features[col].astype('int32')
                
        self.feature_columns = [col for col in self.df_features.columns 
                               if col not in ['open', 'high', 'low', 'close', 'volume']]
        logger.info(f"読み込み完了。{len(self.df_features)}行 x {len(self.feature_columns)}特徴量を検出。")

    def _test_distribution_stability(self, p1_features: pd.DataFrame, p2_features: pd.DataFrame) -> set:
        logger.info("--- 分布安定性テスト（KS検定）を開始 ---")
        unstable_features = set()
        for feature in tqdm(self.feature_columns, desc="KS Test"):
            p1_series = p1_features[feature].dropna().loc[np.isfinite(p1_features[feature])]
            p2_series = p2_features[feature].dropna().loc[np.isfinite(p2_features[feature])]
            if len(p1_series) < 20 or len(p2_series) < 20: 
                continue
            stat, p_value = ks_2samp(p1_series.values, p2_series.values)
            if p_value < self.ks_p_value_threshold: 
                unstable_features.add(feature)
        logger.info(f"分布安定性テスト完了。{len(unstable_features)}個の不安定な特徴量を検出。")
        return unstable_features

    def _test_importance_stability(self, p1_features: pd.DataFrame, p2_features: pd.DataFrame, target: pd.Series) -> set:
        logger.info("--- 重要度安定性テスト（Permutation Importance / バッチ処理モード）を開始 ---")
        p1_target, p2_target = target.loc[p1_features.index], target.loc[p2_features.index]
        
        batch_size = 100
        n_batches = math.ceil(len(self.feature_columns) / batch_size)
        
        all_p1_importances = pd.Series(dtype='float32')
        all_p2_importances = pd.Series(dtype='float32')

        # --- 前半期間の処理 ---
        logger.info(f"前半期間のモデルで特徴量重要度を計算中... (並列処理: {self.n_jobs}スレッド)")
        X1_train_full, X1_val_full, y1_train, y1_val = train_test_split(p1_features, p1_target, test_size=0.3, shuffle=False)
        
        for i in tqdm(range(n_batches), desc="Permutation Importance (前半)"):
            start, end = i * batch_size, (i + 1) * batch_size
            batch_features = self.feature_columns[start:end]
            
            X1_train_batch, X1_val_batch = X1_train_full[batch_features], X1_val_full[batch_features]
            
            model_batch = lgb.LGBMRegressor(n_estimators=50, n_jobs=self.n_jobs, random_state=42, verbosity=-1)
            model_batch.fit(X1_train_batch, y1_train)
            
            imp_result = permutation_importance(model_batch, X1_val_batch, y1_val, n_repeats=5, random_state=42, n_jobs=self.n_jobs)
            batch_importances = pd.Series(imp_result.importances_mean, index=batch_features)
            all_p1_importances = pd.concat([all_p1_importances, batch_importances])

        # --- 後半期間の処理 ---
        logger.info(f"後半期間のモデルで特徴量重要度を計算中... (並列処理: {self.n_jobs}スレッド)")
        X2_train_full, X2_val_full, y2_train, y2_val = train_test_split(p2_features, p2_target, test_size=0.3, shuffle=False)
        
        for i in tqdm(range(n_batches), desc="Permutation Importance (後半)"):
            start, end = i * batch_size, (i + 1) * batch_size
            batch_features = self.feature_columns[start:end]

            X2_train_batch, X2_val_batch = X2_train_full[batch_features], X2_val_full[batch_features]
            
            model_batch = lgb.LGBMRegressor(n_estimators=50, n_jobs=self.n_jobs, random_state=42, verbosity=-1)
            model_batch.fit(X2_train_batch, y2_train)
            
            imp_result = permutation_importance(model_batch, X2_val_batch, y2_val, n_repeats=5, random_state=42, n_jobs=self.n_jobs)
            batch_importances = pd.Series(imp_result.importances_mean, index=batch_features)
            all_p2_importances = pd.concat([all_p2_importances, batch_importances])

        importance_change = (all_p2_importances - all_p1_importances) / (all_p1_importances.abs() + 1e-9)
        unstable_features = set(importance_change[importance_change.abs() > self.importance_change_threshold].index)
        logger.info(f"重要度安定性テスト完了。{len(unstable_features)}個の不安定な特徴量を検出。")
        return unstable_features

    def run_validation(self, target_horizon: int = 60) -> list:
        self._load_data()
        target = self.df_features['close'].pct_change(periods=target_horizon).shift(-target_horizon)
        valid_indices = target.dropna().index
        features, target = self.df_features.loc[valid_indices], target.loc[valid_indices]
        split_point = len(features) // 2
        p1_features, p2_features = features.iloc[:split_point], features.iloc[split_point:]
        logger.info(f"データを前半 {len(p1_features)}行 と 後半 {len(p2_features)}行 に分割しました。")
        unstable_by_dist = self._test_distribution_stability(p1_features, p2_features)
        unstable_by_importance = self._test_importance_stability(p1_features, p2_features, target)
        total_unstable_features = unstable_by_dist.union(unstable_by_importance)
        stable_features = [f for f in self.feature_columns if f not in total_unstable_features]
        logger.info("-" * 50)
        logger.info("🎉 第一防衛線: 特徴量選抜完了 🎉")
        logger.info(f"初期候補: {len(self.feature_columns)}個")
        logger.info(f"不安定（分布）: {len(unstable_by_dist)}個")
        logger.info(f"不安定（重要度）: {len(unstable_by_importance)}個")
        logger.info(f"合計除外数: {len(total_unstable_features)}個")
        logger.info(f"一次選抜通過（安定）: {len(stable_features)}個")
        logger.info("-" * 50)
        return stable_features

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="特徴量安定性検証スクリプト")
    parser.add_argument('--jobs', type=int, default=10, help='Permutation Importanceで使用する並列処理の数（スレッド数）')
    args = parser.parse_args()
    logger.info(f"実行パラメータ: 使用スレッド数 = {args.jobs}")
    validator = FeatureValidator(
        feature_universe_path='data/temp_chunks/feature_chunk_0.parquet',
        n_jobs=args.jobs
    )
    stable_feature_list = validator.run_validation()
    
    # 複数形式で保存
    joblib.dump(stable_feature_list, 'data/temp_chunks/defense_results/joblib/stable_feature_list.joblib')
    
    # CSV形式で保存（人間確認用）
    stable_df = pd.DataFrame({'feature_name': stable_feature_list})
    stable_df.to_csv('data/temp_chunks/defense_results/csv/stable_feature_list.csv', index=False)
    
    # JSON形式で保存（メタデータ含む）
    import json
    result_info = {
        'total_features': len(stable_feature_list),
        'features': stable_feature_list,
        'validation_type': 'first_defense_line',
        'timestamp': pd.Timestamp.now().isoformat()
    }
    with open('data/temp_chunks/defense_results/json/stable_feature_list.json', 'w') as f:
        json.dump(result_info, f, indent=2)
    
    logger.info("一次選抜を通過した特徴量リストを複数形式で保存しました。")
    logger.info("- JOBLIB: data/temp_chunks/defense_results/joblib/stable_feature_list.joblib")
    logger.info("- CSV: data/temp_chunks/defense_results/csv/stable_feature_list.csv")
    logger.info("- JSON: data/temp_chunks/defense_results/json/stable_feature_list.json")