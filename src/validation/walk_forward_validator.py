import pandas as pd
import numpy as np
import joblib
import lightgbm as lgb
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import roc_auc_score
from collections import Counter
from tqdm import tqdm
from common.logger_setup import logger

class WalkForwardValidator:
    """
    【最終確定版 v1.7 - メモリ対策クレンジング】
    幽霊列の削除ロジックを、メモリに優しい反復チェック方式に変更。
    """
    def __init__(self,
                 feature_universe_path: str,
                 stable_features_path: str,
                 n_splits: int = 5,
                 final_feature_count: int = 15):
        self.feature_universe_path = feature_universe_path
        self.stable_features_path = stable_features_path
        self.n_splits = n_splits
        self.final_feature_count = final_feature_count
        self.df_universe = None
        self.stable_features = None

    def _load_data(self):
        """データと一次選抜通過リストをロードし、クレンジングを行う"""
        logger.info(f"第二防衛線: 特徴量ユニバース '{self.feature_universe_path}' を読み込み中...")
        
        # パスに応じて適切な読み込み方法を選択
        if self.feature_universe_path.endswith('.parquet'):
            self.df_universe = pd.read_parquet(self.feature_universe_path)
        else:
            self.df_universe = joblib.load(self.feature_universe_path)
        
        # クレンジング処理：メモリに優しい方法で幽霊列を特定
        logger.info("データクレンジング：全要素がNaNの列（幽霊列）をスキャンしています...")
        ghost_cols = []
        for col in tqdm(self.df_universe.columns, desc="Scanning for ghost columns"):
            if self.df_universe[col].isna().all():
                ghost_cols.append(col)
        
        # 特定した幽霊列をまとめて削除
        if ghost_cols:
            logger.warning(f"{len(ghost_cols)}個の幽霊列を削除します: {ghost_cols}")
            self.df_universe.drop(columns=ghost_cols, inplace=True)
        else:
            logger.info("幽霊列は検出されませんでした。")

        logger.info("メモリ効率化のため、データ型を変換しています...")
        for col in tqdm(self.df_universe.columns, desc="Data Type Conversion"):
            if self.df_universe[col].dtype == 'float64': 
                self.df_universe[col] = self.df_universe[col].astype('float32')
            if self.df_universe[col].dtype == 'int64': 
                self.df_universe[col] = self.df_universe[col].astype('int32')
            
        logger.info(f"一次選抜通過リスト '{self.stable_features_path}' を読み込み中...")
        self.stable_features = joblib.load(self.stable_features_path)
        
        original_stable_count = len(self.stable_features)
        self.stable_features = [f for f in self.stable_features if f not in ghost_cols]
        new_stable_count = len(self.stable_features)
        if original_stable_count != new_stable_count:
            logger.warning(f"{original_stable_count - new_stable_count}個の幽霊列が安定特徴量リストから除外されました。")
            
        logger.info(f"読み込み完了。{new_stable_count}個のクレンジング済み安定特徴量を対象とします。")

    def _define_target(self, target_horizon: int = 60, profit_thresh: float = 0.0005):
        logger.info("予測目標（ターゲット）を計算中...")
        forward_rolling_max = self.df_universe['close'].iloc[::-1].rolling(window=target_horizon, min_periods=1).max().iloc[::-1]
        future_returns = (forward_rolling_max / self.df_universe['close']) - 1
        target = (future_returns > profit_thresh)
        data = self.df_universe[self.stable_features].copy()
        data['target'] = target
        data.dropna(subset=['target'], inplace=True)
        data['target'] = data['target'].astype(int)
        data.dropna(inplace=True)
        self.X = data.drop('target', axis=1)
        self.y = data['target']
        if self.X.empty:
            raise RuntimeError("致命的エラー: データ準備後に分析対象データが0行になりました。")
        logger.info(f"ターゲット計算完了。分析対象データは {len(self.X)} 行です。")

    def _find_best_features_for_split(self, X_train: pd.DataFrame, y_train: pd.Series) -> list:
        features = list(X_train.columns)
        num_to_eliminate = len(features) - self.final_feature_count
        if num_to_eliminate <= 0: 
            return features
        pbar = tqdm(range(num_to_eliminate), desc="  -> RFE-like Selection", leave=False)
        while len(features) > self.final_feature_count:
            model = lgb.LGBMClassifier(random_state=42, verbosity=-1, n_jobs=-1)
            model.fit(X_train[features], y_train)
            importances = pd.Series(model.feature_importances_, index=features)
            worst_feature = importances.idxmin()
            features.remove(worst_feature)
            pbar.update(1)
        pbar.close()
        return features

    def run_validation(self):
        self._load_data()
        self._define_target()
        tscv = TimeSeriesSplit(n_splits=self.n_splits)
        selected_features_across_splits = []
        logger.info(f"--- 第二防衛線: 全{self.n_splits}分割のウォークフォワード検証を開始 ---")
        for i, (train_index, test_index) in enumerate(tqdm(tscv.split(self.X), total=self.n_splits, desc="Walk-Forward Validation")):
            logger.info(f"--- [分割 {i+1}/{self.n_splits}] ---")
            X_train, X_test = self.X.iloc[train_index], self.X.iloc[test_index]
            y_train, y_test = self.y.iloc[train_index], self.y.iloc[test_index]
            logger.info(f"  -> 訓練データ（{len(X_train)}行）で最強チームを探索中...")
            best_features_for_this_split = self._find_best_features_for_split(X_train, y_train)
            selected_features_across_splits.extend(best_features_for_this_split)
            logger.info(f"  -> 分割 {i+1} の最強チーム（{len(best_features_for_this_split)}名）を選出。")
            model = lgb.LGBMClassifier(random_state=42, verbosity=-1, n_jobs=-1)
            model.fit(X_train[best_features_for_this_split], y_train)
            y_pred_oos = model.predict_proba(X_test[best_features_for_this_split])[:, 1]
            oos_score = roc_auc_score(y_test, y_pred_oos)
            logger.info(f"  -> テストデータ（{len(X_test)}行）での性能（AUC）: {oos_score:.4f}")
        feature_counts = Counter(selected_features_across_splits)
        final_team = [feature for feature, count in feature_counts.most_common(self.final_feature_count)]
        logger.info("-" * 50)
        logger.info("🎉 第二防衛線: 最強チームの選抜完了 🎉")
        logger.info(f"最終選抜メンバー（{len(final_team)}名）:")
        for f in final_team:
            logger.info(f"  - {f} (選出回数: {feature_counts[f]}回)")
        logger.info("-" * 50)
        return final_team

if __name__ == '__main__':
    validator = WalkForwardValidator(
        feature_universe_path='data/temp_chunks/feature_chunk_0.parquet',
        stable_features_path='data/stable_feature_list.joblib'
    )
    final_feature_team = validator.run_validation()
    
    # 複数形式で保存
    joblib.dump(final_feature_team, 'data/temp_chunks/defense_results/joblib/final_feature_team.joblib')
    
    # CSV形式で保存（人間確認用）
    import pandas as pd
    final_df = pd.DataFrame({'feature_name': final_feature_team})
    final_df.to_csv('data/temp_chunks/defense_results/csv/final_feature_team.csv', index=False)
    
    # JSON形式で保存（メタデータ含む）
    import json
    result_info = {
        'total_features': len(final_feature_team),
        'features': final_feature_team,
        'validation_type': 'second_defense_line',
        'timestamp': pd.Timestamp.now().isoformat()
    }
    with open('data/temp_chunks/defense_results/json/final_feature_team.json', 'w') as f:
        json.dump(result_info, f, indent=2)
    
    logger.info("最終選抜メンバーリストを複数形式で保存しました。")
    logger.info("- JOBLIB: data/temp_chunks/defense_results/joblib/final_feature_team.joblib")
    logger.info("- CSV: data/temp_chunks/defense_results/csv/final_feature_team.csv")
    logger.info("- JSON: data/temp_chunks/defense_results/json/final_feature_team.json")