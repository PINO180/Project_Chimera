"""
feature_validator.py - v3.0
第一防衛線：時間的安定性とレジーム適応性の確保

統合設計図V準拠：
- サンプリング廃止（全データ使用）
- 敵対的検証（Adversarial Validation）追加
- Dask-LightGBM完全対応
- Pylance厳格型定義準拠
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import logging
from typing import List, Set, Dict, Tuple
from pathlib import Path
import json

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

import dask
import dask.dataframe as dd
from dask.dataframe.core import DataFrame as DaskDataFrame
import dask_lightgbm.core as dlgb
from scipy.stats import chi2_contingency
import numpy as np
import joblib
from tqdm import tqdm
import pandas as pd


class FeatureValidator:
    """
    【Dask版 v3.0 - 統合設計図V準拠】
    数百GB～TB級の特徴量ユニバースを全データで処理。
    敵対的検証による市場レジーム適応性評価を追加。
    """
    
    def __init__(
        self,
        feature_universe_path: str,
        n_jobs: int = 1,
        ks_p_value_threshold: float = 0.05,
        importance_change_threshold: float = 0.5,
        adversarial_auc_threshold: float = 0.7
    ):
        self.feature_universe_path = feature_universe_path
        self.n_jobs = n_jobs
        self.ks_p_value_threshold = ks_p_value_threshold
        self.importance_change_threshold = importance_change_threshold
        self.adversarial_auc_threshold = adversarial_auc_threshold
        self.ddf: DaskDataFrame | None = None
        self.feature_columns: List[str] = []
        
    def _load_data(self) -> None:
        """Dask DataFrameとしてパーティション化されたParquetを読み込む"""
        logger.info(f"第一防衛線: 特徴量ユニバース '{self.feature_universe_path}' をDask DataFrameとして読み込んでいます...")
        
        self.ddf = dd.read_parquet(
            self.feature_universe_path,
            engine='pyarrow'
        )
        
        logger.info("スキーマ情報を取得中...")
        all_columns = self.ddf.columns.tolist()
        self.feature_columns = [
            col for col in all_columns 
            if col not in ['open', 'high', 'low', 'close', 'volume', 'timestamp']
        ]
        
        n_partitions = self.ddf.npartitions
        logger.info(f"読み込み完了。{n_partitions}パーティション、{len(self.feature_columns)}特徴量を検出。")
        
    def _test_distribution_stability(self) -> Set[str]:
        """
        【v3.0】分布安定性テスト
        全データのヒストグラムを並列計算し、カイ二乗検定で比較
        """
        logger.info("--- 分布安定性テスト（カイ二乗検定）を開始 ---")
        
        if self.ddf is None:
            raise ValueError("Dask DataFrame not loaded.")
        
        split_point = self.ddf.npartitions // 2
        p1_ddf = self.ddf.partitions[:split_point]
        p2_ddf = self.ddf.partitions[split_point:]
        
        unstable_features: Set[str] = set()
        feature_batches = [
            self.feature_columns[i:i + 20] 
            for i in range(0, len(self.feature_columns), 20)
        ]
        
        for batch in tqdm(feature_batches, desc="Chi-Squared Test"):
            tasks = []
            for feature in batch:
                tasks.append(p1_ddf[feature].histogram(bins=50))
                tasks.append(p2_ddf[feature].histogram(bins=50))
            
            histograms = dask.compute(*tasks)
            
            for i, feature in enumerate(batch):
                hist1_counts, _ = histograms[i*2]
                hist2_counts, _ = histograms[i*2 + 1]
                
                observed = np.array([hist1_counts, hist2_counts])
                observed = observed[:, observed.sum(axis=0) > 0]
                
                if observed.shape[1] < 2:
                    continue
                
                try:
                    _, p_value, _, _ = chi2_contingency(observed)
                    if p_value < self.ks_p_value_threshold:
                        unstable_features.add(feature)
                except ValueError:
                    continue
        
        logger.info(f"分布安定性テスト完了。{len(unstable_features)}個の不安定な特徴量を検出。")
        return unstable_features
    
    def _test_importance_stability(self) -> Set[str]:
        """
        【v3.0】重要度安定性テスト
        全データでPermutation Importanceを計算
        """
        logger.info("--- 重要度安定性テスト（Permutation Importance）を開始 ---")
        
        if self.ddf is None:
            raise ValueError("Dask DataFrame not loaded.")
        
        logger.info("ターゲット変数を計算中...")
        target_horizon = 60
        target = self.ddf['close'].pct_change(periods=target_horizon).shift(-target_horizon)
        ddf_with_target = self.ddf.assign(target=target).dropna()
        
        split_point = ddf_with_target.npartitions // 2
        p1_ddf = ddf_with_target.partitions[:split_point]
        p2_ddf = ddf_with_target.partitions[split_point:]
        
        all_p1_importances: Dict[str, float] = {}
        all_p2_importances: Dict[str, float] = {}
        
        # 前半期間
        logger.info("前半期間のモデルで特徴量重要度を計算中...")
        p1_train_ddf, p1_val_ddf = p1_ddf.random_split([0.7, 0.3], random_state=42)
        
        X1_train = p1_train_ddf[self.feature_columns]
        y1_train = p1_train_ddf['target']
        X1_val = p1_val_ddf[self.feature_columns]
        y1_val = p1_val_ddf['target']
        
        model1 = dlgb.LGBMRegressor(n_estimators=50, random_state=42, verbosity=-1, n_jobs=self.n_jobs)
        model1.fit(X1_train, y1_train)
        
        logger.info("前半期間: ベースラインスコアを計算...")
        baseline_score1 = model1.score(X1_val, y1_val)
        
        for feature in tqdm(self.feature_columns, desc="Permutation Importance (前半)"):
            shuffled_col = X1_val[feature].map_partitions(
                lambda s: s.sample(frac=1, random_state=42)
            )
            X1_val_permuted = X1_val.assign(**{feature: shuffled_col})
            permuted_score = model1.score(X1_val_permuted, y1_val)
            all_p1_importances[feature] = baseline_score1 - permuted_score
        
        # 後半期間
        logger.info("後半期間のモデルで特徴量重要度を計算中...")
        p2_train_ddf, p2_val_ddf = p2_ddf.random_split([0.7, 0.3], random_state=42)
        
        X2_train = p2_train_ddf[self.feature_columns]
        y2_train = p2_train_ddf['target']
        X2_val = p2_val_ddf[self.feature_columns]
        y2_val = p2_val_ddf['target']
        
        model2 = dlgb.LGBMRegressor(n_estimators=50, random_state=42, verbosity=-1, n_jobs=self.n_jobs)
        model2.fit(X2_train, y2_train)
        
        logger.info("後半期間: ベースラインスコアを計算...")
        baseline_score2 = model2.score(X2_val, y2_val)
        
        for feature in tqdm(self.feature_columns, desc="Permutation Importance (後半)"):
            shuffled_col = X2_val[feature].map_partitions(
                lambda s: s.sample(frac=1, random_state=42)
            )
            X2_val_permuted = X2_val.assign(**{feature: shuffled_col})
            permuted_score = model2.score(X2_val_permuted, y2_val)
            all_p2_importances[feature] = baseline_score2 - permuted_score
        
        unstable_features: Set[str] = set()
        for feature in self.feature_columns:
            p1_imp = all_p1_importances.get(feature, 0.0)
            p2_imp = all_p2_importances.get(feature, 0.0)
            
            importance_change = abs(p2_imp - p1_imp) / (abs(p1_imp) + 1e-9)
            if importance_change > self.importance_change_threshold:
                unstable_features.add(feature)
        
        logger.info(f"重要度安定性テスト完了。{len(unstable_features)}個の不安定な特徴量を検出。")
        return unstable_features
    
    def _test_adversarial_validation(self) -> Tuple[Set[str], Dict[str, float]]:
        """
        【v3.0新規】敵対的検証
        訓練データと検証データを識別するモデルを構築し、
        時間的ドリフトの兆候を示す特徴量を検出
        """
        logger.info("--- 敵対的検証（Adversarial Validation）を開始 ---")
        
        if self.ddf is None:
            raise ValueError("Dask DataFrame not loaded.")
        
        split_point = self.ddf.npartitions // 2
        train_ddf = self.ddf.partitions[:split_point]
        val_ddf = self.ddf.partitions[split_point:]
        
        # ラベル付与
        train_ddf = train_ddf.assign(adversarial_label=0)
        val_ddf = val_ddf.assign(adversarial_label=1)
        
        # 結合
        combined_ddf = dd.concat([train_ddf, val_ddf], axis=0)
        combined_ddf = combined_ddf.dropna()
        
        logger.info("敵対的検証用モデルを訓練中...")
        X = combined_ddf[self.feature_columns]
        y = combined_ddf['adversarial_label']
        
        model = dlgb.LGBMClassifier(
            n_estimators=50,
            random_state=42,
            verbosity=-1,
            n_jobs=self.n_jobs
        )
        model.fit(X, y)
        
        # AUC計算（メモリ効率のため予測とターゲットを同時にcompute）
        logger.info("敵対的検証の性能を評価中...")
        y_pred_proba = model.predict_proba(X)
        y_computed, y_pred_computed = dask.compute(y, y_pred_proba)
        
        from sklearn.metrics import roc_auc_score
        auc = roc_auc_score(y_computed, y_pred_computed[:, 1])
        logger.info(f"敵対的検証AUC: {auc:.4f}")
        
        # 特徴量重要度取得
        importances = model.feature_importances_
        adversarial_scores: Dict[str, float] = dict(zip(self.feature_columns, importances))
        
        # 不安定フラグ付与
        unstable_features: Set[str] = set()
        if auc > self.adversarial_auc_threshold:
            logger.warning(f"時間的ドリフト検出（AUC={auc:.4f} > {self.adversarial_auc_threshold}）")
            # 上位20%の特徴量を不安定とマーク
            threshold_importance = np.percentile(list(adversarial_scores.values()), 80)
            unstable_features = {
                feat for feat, score in adversarial_scores.items()
                if score > threshold_importance
            }
            logger.info(f"{len(unstable_features)}個の特徴量に不安定フラグ付与。")
        else:
            logger.info("時間的ドリフトは検出されませんでした。")
        
        return unstable_features, adversarial_scores
    
    def run_validation(self) -> Tuple[List[str], Dict[str, float]]:
        """
        【v3.0】検証パイプライン全体を実行
        """
        self._load_data()
        
        unstable_by_dist = self._test_distribution_stability()
        unstable_by_importance = self._test_importance_stability()
        unstable_by_adversarial, adversarial_scores = self._test_adversarial_validation()
        
        total_unstable_features = unstable_by_dist.union(
            unstable_by_importance
        ).union(unstable_by_adversarial)
        
        stable_features = [
            f for f in self.feature_columns 
            if f not in total_unstable_features
        ]
        
        logger.info("-" * 50)
        logger.info("🎉 第一防衛線: 特徴量選抜完了 (全データ使用 + 敵対的検証) 🎉")
        logger.info(f"初期候補: {len(self.feature_columns)}個")
        logger.info(f"不安定（分布）: {len(unstable_by_dist)}個")
        logger.info(f"不安定（重要度）: {len(unstable_by_importance)}個")
        logger.info(f"不安定（敵対的）: {len(unstable_by_adversarial)}個")
        logger.info(f"合計除外数: {len(total_unstable_features)}個")
        logger.info(f"一次選抜通過（安定）: {len(stable_features)}個")
        logger.info("-" * 50)
        
        return stable_features, adversarial_scores


if __name__ == '__main__':
    dask.config.set({'dataframe.query-planning': True})
    
    validator = FeatureValidator(
        feature_universe_path='data/master_table_partitioned',
        n_jobs=4
    )
    
    stable_features, adversarial_scores = validator.run_validation()
    
    output_dir = Path('data/temp_chunks/defense_results')
    (output_dir / 'joblib').mkdir(parents=True, exist_ok=True)
    (output_dir / 'csv').mkdir(parents=True, exist_ok=True)
    (output_dir / 'json').mkdir(parents=True, exist_ok=True)
    
    joblib.dump(stable_features, output_dir / 'joblib' / 'stable_feature_list.joblib')
    joblib.dump(adversarial_scores, output_dir / 'joblib' / 'adversarial_scores.joblib')
    
    stable_df = pd.DataFrame({'feature_name': stable_features})
    stable_df.to_csv(output_dir / 'csv' / 'stable_feature_list.csv', index=False)
    
    result_info = {
        'total_features': len(stable_features),
        'features': stable_features,
        'validation_type': 'first_defense_line_v3',
        'adversarial_auc_threshold': validator.adversarial_auc_threshold,
        'timestamp': pd.Timestamp.now().isoformat()
    }
    
    with open(output_dir / 'json' / 'stable_feature_list.json', 'w') as f:
        json.dump(result_info, f, indent=2)
    
    logger.info("一次選抜通過特徴量リストを複数形式で保存しました。")
    logger.info(f"- JOBLIB: {output_dir / 'joblib' / 'stable_feature_list.joblib'}")
    logger.info(f"- CSV: {output_dir / 'csv' / 'stable_feature_list.csv'}")
    logger.info(f"- JSON: {output_dir / 'json' / 'stable_feature_list.json'}")