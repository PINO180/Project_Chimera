import sys
import os
# プロジェクトルートをパスに追加
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import logging

# 標準のロギング設定
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

import dask
import dask.dataframe as dd
import dask_lightgbm.core as dlgb
from scipy.stats import chi2_contingency
import numpy as np
import joblib
import lightgbm as lgb
from sklearn.inspection import permutation_importance
from scipy.stats import ks_2samp
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import argparse
import math
import json
from pathlib import Path

class FeatureValidator:
    """
    【Dask版 v2.0 - Out-of-Core処理対応】
    数百GB～TB級の特徴量ユニバースを64GBのRAMで処理可能。
    大規模サンプリングによる分布・重要度安定性テストを実行。
    """
    def __init__(self,
                 feature_universe_path: str,
                 n_jobs: int = 1,
                 ks_p_value_threshold: float = 0.05,
                 importance_change_threshold: float = 0.5,
                 sample_frac: float = 0.1):
        self.feature_universe_path = feature_universe_path
        self.n_jobs = n_jobs
        self.ks_p_value_threshold = ks_p_value_threshold
        self.importance_change_threshold = importance_change_threshold
        self.sample_frac = sample_frac  # サンプリング比率（デフォルト10%）
        self.ddf = None
        self.feature_columns: list[str] = []

    def _load_data(self):
        """Dask DataFrameとしてパーティション化されたParquetを読み込む"""
        logger.info(f"第一防衛線: 特徴量ユニバース '{self.feature_universe_path}' をDask DataFrameとして読み込んでいます...")
        
        # Daskで遅延読み込み
        self.ddf = dd.read_parquet(  # type: ignore
            self.feature_universe_path,
            engine='pyarrow'
        )
        
        logger.info("スキーマ情報を取得中...")
        # 特徴量カラムの特定（OHLCVを除外）
        all_columns = self.ddf.columns.tolist()
        self.feature_columns = [col for col in all_columns 
                               if col not in ['open', 'high', 'low', 'close', 'volume']]
        
        # データの基本情報を取得（遅延実行なので高速）
        n_partitions = self.ddf.npartitions
        logger.info(f"読み込み完了。{n_partitions}パーティション、{len(self.feature_columns)}特徴量を検出。")

    def _extract_large_sample(self):
        """
        代表的な大規模サンプルを抽出
        このサンプルを分布・重要度テスト両方で使用することでメモリ効率を最大化
        """
        logger.info(f"--- 大規模サンプル抽出（{self.sample_frac*100:.1f}%）を開始 ---")
        
        # Daskのサンプリング機能を使用（deterministic）
        sampled_ddf = self.ddf.sample(frac=self.sample_frac, random_state=42)
        
        # サンプルをメモリに収集
        logger.info("サンプルデータをメモリに収集中...")
        df_sample = sampled_ddf.compute()
        
        logger.info("メモリ効率化のため、データ型を変換しています...")
        for col in tqdm(df_sample.columns, desc="Data Type Conversion"):
            if df_sample[col].dtype == 'float64': 
                df_sample[col] = df_sample[col].astype('float32')
            if df_sample[col].dtype == 'int64': 
                df_sample[col] = df_sample[col].astype('int32')
        
        logger.info(f"サンプル抽出完了。{len(df_sample)}行のデータを取得。")
        return df_sample

    def _test_distribution_stability(self) -> set:
        """
        【高度化版】
        Daskの分散コンピューティングを使い、全データのヒストグラムを計算して比較する。
        KS検定の代わりにカイ二乗検定を使用し、スケーラビリティを確保。
        """
        logger.info("--- 分布安定性テスト（分散ヒストグラム比較）を開始 ---")
        
        if self.ddf is None:
            raise ValueError("Dask DataFrame not loaded.")

        # 全データを前半・後半のDask DataFrameに分割（遅延実行）
        split_point = self.ddf.npartitions // 2
        p1_ddf = self.ddf.partitions[:split_point]
        p2_ddf = self.ddf.partitions[split_point:]
        
        unstable_features = set()
        
        # 特徴量をバッチ処理してdask.computeの呼び出し回数を削減
        feature_batches = [self.feature_columns[i:i + 20] for i in range(0, len(self.feature_columns), 20)]

        for batch in tqdm(feature_batches, desc="Chi-Squared Test"):
            tasks = []
            for feature in batch:
                # 各特徴量に対して、前半と後半のヒストグラム計算をタスクとして登録
                # bins=50 は分布の形状を捉えるのに十分な解像度
                tasks.append(p1_ddf[feature].histogram(bins=50))
                tasks.append(p2_ddf[feature].histogram(bins=50))

            # バッチ内の全タスクを一度に並列実行
            histograms = dask.compute(*tasks)  # type: ignore
            
            for i, feature in enumerate(batch):
                # 実行結果からヒストグラムを取得
                hist1_counts, _ = histograms[i*2]
                hist2_counts, _ = histograms[i*2 + 1]
                
                # 観測度数表を作成
                observed = np.array([hist1_counts, hist2_counts])
                
                # カイ二乗検定を実行
                # 合計が0の列（データが存在しないビン）を除外
                observed = observed[:, observed.sum(axis=0) > 0]
                if observed.shape[1] < 2:
                    continue

                try:
                    chi2, p_value, _, _ = chi2_contingency(observed)
                    if p_value < self.ks_p_value_threshold:
                        unstable_features.add(feature)
                except ValueError:
                    # 期待度数が低すぎる場合のエラーを無視
                    continue
        
        logger.info(f"分布安定性テスト完了。{len(unstable_features)}個の不安定な特徴量を検出。")
        return unstable_features

    def _test_importance_stability(self) -> set:
        """
        【高度化版】
        dask-lightgbmで全データを使用してモデルを訓練。
        Permutation Importanceでは、map_partitionsによるブロック単位パーミュテーションを実行。
        """
        logger.info("--- 重要度安定性テスト（全データ / ブロック単位パーミュテーション）を開始 ---")
        
        if self.ddf is None:
            raise ValueError("Dask DataFrame not loaded.")

        # ターゲット変数の計算
        logger.info("ターゲット変数を計算中...")
        target_horizon = 60
        target = self.ddf['close'].pct_change(periods=target_horizon).shift(-target_horizon)
        ddf_with_target = self.ddf.assign(target=target).dropna()

        # 全データを前半・後半に分割
        split_point = ddf_with_target.npartitions // 2
        p1_ddf = ddf_with_target.partitions[:split_point]
        p2_ddf = ddf_with_target.partitions[split_point:]

        all_p1_importances = {}
        all_p2_importances = {}

        # --- 前半期間の処理 ---
        logger.info(f"前半期間のモデルで特徴量重要度を計算中...")
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
            # ブロック単位（パーティション単位）でシャッフル
            shuffled_col = X1_val[feature].map_partitions(lambda s: s.sample(frac=1, random_state=42))
            X1_val_permuted = X1_val.assign(**{feature: shuffled_col})
            
            permuted_score = model1.score(X1_val_permuted, y1_val)
            all_p1_importances[feature] = baseline_score1 - permuted_score

        # --- 後半期間の処理（前半と同様） ---
        logger.info(f"後半期間のモデルで特徴量重要度を計算中...")
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
            shuffled_col = X2_val[feature].map_partitions(lambda s: s.sample(frac=1, random_state=42))
            X2_val_permuted = X2_val.assign(**{feature: shuffled_col})
            permuted_score = model2.score(X2_val_permuted, y2_val)
            all_p2_importances[feature] = baseline_score2 - permuted_score

        # 重要度の変化を計算
        unstable_features = set()
        for feature in self.feature_columns:
            p1_imp = all_p1_importances.get(feature, 0.0)
            p2_imp = all_p2_importances.get(feature, 0.0)
            
            importance_change = abs(p2_imp - p1_imp) / (abs(p1_imp) + 1e-9)
            if importance_change > self.importance_change_threshold:
                unstable_features.add(feature)
        
        logger.info(f"重要度安定性テスト完了。{len(unstable_features)}個の不安定な特徴量を検出。")
        return unstable_features

    def run_validation(self) -> list:
        """
        【高度化版】検証パイプライン全体を実行
        サンプリングを廃止し、全てのテストを全データに対して実行
        """
        # データ読み込み
        self._load_data()
        
        # 分布安定性テスト
        unstable_by_dist = self._test_distribution_stability()
        
        # 重要度安定性テスト
        # このメソッド内でターゲット計算とデータ分割が行われる
        unstable_by_importance = self._test_importance_stability()
        
        # 不安定な特徴量を統合
        total_unstable_features = unstable_by_dist.union(unstable_by_importance)
        stable_features = [f for f in self.feature_columns if f not in total_unstable_features]
        
        logger.info("-" * 50)
        logger.info("🎉 第一防衛線: 特徴量選抜完了 (全データ使用) 🎉")
        logger.info(f"初期候補: {len(self.feature_columns)}個")
        logger.info(f"不安定（分布）: {len(unstable_by_dist)}個")
        logger.info(f"不安定（重要度）: {len(unstable_by_importance)}個")
        logger.info(f"合計除外数: {len(total_unstable_features)}個")
        logger.info(f"一次選抜通過（安定）: {len(stable_features)}個")
        logger.info("-" * 50)
        
        return stable_features