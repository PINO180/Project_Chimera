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
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))

import blueprint as config
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
from dask.dataframe import DataFrame as DaskDataFrame # .coreを削除
from dask import config as dask_config # dask専用の別名を付ける
# lightgbm.daskからDask用のクラスをインポート
from lightgbm.dask import DaskLGBMRegressor, DaskLGBMClassifier
from dask.base import compute # dask -> dask.base に変更
from scipy.stats import chi2_contingency
import numpy as np
import joblib
from tqdm import tqdm
import pandas as pd
from scipy.stats import ks_2samp  # 追加

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
        """
        Dask DataFrameとして特徴量ユニバースを読み込む (v4.1 - タイムフレームサフィックス版)
        """
        feature_universe_path = Path(self.feature_universe_path)
        logger.info(f"第一防衛線: 特徴量ユニバース '{feature_universe_path}' をDask DataFrameとして読み込み中...")

        if not feature_universe_path.exists() or not feature_universe_path.is_dir():
            raise FileNotFoundError(f"特徴量ユニバースの親ディレクトリが見つかりません: {feature_universe_path}")

        # 全ての.parquetファイルとパーティション化ディレクトリを再帰的に検出
        all_parquet_paths = []
        
        for engine_dir in feature_universe_path.iterdir():
            if not engine_dir.is_dir():
                continue
            
            for item in engine_dir.iterdir():
                if item.is_file() and item.suffix == '.parquet':
                    all_parquet_paths.append(item)
                elif item.is_dir():
                    all_parquet_paths.append(item)
        
        if not all_parquet_paths:
            raise FileNotFoundError("Parquetファイルまたはパーティションが見つかりません")

        logger.info(f"{len(all_parquet_paths)}個のParquetデータソースを検出。")

        non_feature_columns = ['timestamp', 'open', 'high', 'low', 'close', 'volume', 'timeframe', 'year', 'month', 'day']
        
        base_ddf = None
        remaining_ddfs = []
        
        for path in tqdm(all_parquet_paths, desc="データ読み込み中"):
            try:
                temp_ddf = dd.read_parquet(
                    str(path),
                    engine='pyarrow',
                    dtype_backend='numpy_nullable'
                )
                
                # 型を統一
                for col in ['year', 'month', 'day']:
                    if col in temp_ddf.columns:
                        temp_ddf[col] = temp_ddf[col].astype('int32')
                
                # タイムフレーム情報を抽出（ファイル名から）
                # 例: features_e1a_D1.parquet → _D1
                #     features_e1a_M0.5.parquet → _M0.5
                #     features_e1a_tick → _tick
                if path.is_dir():
                    # ディレクトリの場合（tickデータ）
                    timeframe_suffix = '_tick'
                else:
                    # ファイルの場合
                    filename = path.name  # "features_e1a_M0.5.parquet"
                    # .parquetを除去
                    base_name = filename.replace('.parquet', '')  # "features_e1a_M0.5"
                    # 最後のアンダースコア以降を取得
                    parts = base_name.split('_')
                    if len(parts) > 1:
                        timeframe_suffix = f"_{parts[-1]}"  # "_M0.5"
                    else:
                        timeframe_suffix = ""
                
                # 特徴量カラムにタイムフレームサフィックスを追加
                rename_dict = {}
                for col in temp_ddf.columns:
                    if col not in non_feature_columns and col not in ['year', 'month', 'day']:
                        rename_dict[col] = f"{col}{timeframe_suffix}"
                
                if rename_dict:
                    temp_ddf = temp_ddf.rename(columns=rename_dict)
                
                if base_ddf is None:
                    base_ddf = temp_ddf
                    logger.info(f"ベースデータセット: {path.name} ({len(temp_ddf.columns)}カラム)")
                else:
                    remaining_ddfs.append((path.name, temp_ddf))
                    
            except Exception as e:
                logger.warning(f"{path.name} の読み込みに失敗: {e}")
                continue
        
        if base_ddf is None:
            raise ValueError("有効なデータソースが1つも読み込めませんでした")
        
        logger.info("特徴量カラムを結合中...")
        
        for name, ddf in tqdm(remaining_ddfs, desc="結合中"):
            if 'timestamp' not in ddf.columns:
                logger.warning(f"{name}: timestampカラムがありません。スキップします。")
                continue
            
            feature_cols = [col for col in ddf.columns if col not in non_feature_columns]
            
            if not feature_cols:
                logger.warning(f"{name}: 特徴量カラムが見つかりません。スキップします。")
                continue
            
            ddf_features = ddf[['timestamp'] + feature_cols]
            base_ddf = base_ddf.merge(ddf_features, on='timestamp', how='outer')
            logger.info(f"{name}: {len(feature_cols)}個の特徴量を追加")
        
        self.ddf = base_ddf
        
        logger.info("スキーマ情報を取得中...")
        all_columns = self.ddf.columns.tolist()
        self.feature_columns = [
            col for col in all_columns
            if col not in non_feature_columns
        ]

        n_partitions = self.ddf.npartitions
        logger.info(f"読み込み完了。{n_partitions}パーティション、{len(self.feature_columns)}特徴量を検出。")
        
    def _test_distribution_stability(self) -> Set[str]:
        """
        【v3.1 - KS検定版】分布安定性テスト
        Kolmogorov-Smirnov検定で時間的分布変化を検出
        """
        logger.info("--- 分布安定性テスト（Kolmogorov-Smirnov検定）を開始 ---")
        
        if self.ddf is None:
            raise ValueError("Dask DataFrame not loaded.")
        
        # データを前半・後半に分割
        split_point = self.ddf.npartitions // 2
        p1_ddf = self.ddf.partitions[:split_point]
        p2_ddf = self.ddf.partitions[split_point:]
        
        unstable_features: Set[str] = set()
        
        # バッチ処理（メモリ効率化）
        feature_batches = [
            self.feature_columns[i:i + 20] 
            for i in range(0, len(self.feature_columns), 20)
        ]
        
        for batch in tqdm(feature_batches, desc="KS Test"):
            # バッチごとにデータをcompute
            p1_batch = p1_ddf[batch].compute()
            p2_batch = p2_ddf[batch].compute()
            
            for feature in batch:
                # NaN/Inf除外
                data1 = p1_batch[feature].dropna()
                data2 = p2_batch[feature].dropna()
                
                # データ量チェック
                if len(data1) < 10 or len(data2) < 10:
                    logger.warning(f"特徴量 {feature} はデータ不足のためスキップ")
                    continue
                
                # scipy KS検定実行
                statistic, p_value = ks_2samp(data1, data2)
                
                if p_value < self.ks_p_value_threshold:
                    unstable_features.add(feature)
        
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
        
        model1 = DaskLGBMRegressor(n_estimators=50, random_state=42, verbosity=-1, n_jobs=self.n_jobs)
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
        
        model2 = DaskLGBMRegressor(n_estimators=50, random_state=42, verbosity=-1, n_jobs=self.n_jobs)
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
        
        model = DaskLGBMClassifier(
            n_estimators=50,
            random_state=42,
            verbosity=-1,
            n_jobs=self.n_jobs
        )
        model.fit(X, y)
        
        # AUC計算（メモリ効率のため予測とターゲットを同時にcompute）
        logger.info("敵対的検証の性能を評価中...")
        y_pred_proba = model.predict_proba(X)
        y_computed, y_pred_computed = compute(y, y_pred_proba)
        
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
    # Daskの最新エンジンを明示的に有効化
    dask_config.set({'dataframe.query-planning': True})

    validator = FeatureValidator(
        feature_universe_path=str(config.S2_FEATURES2),
        n_jobs=4
    )

    stable_features, adversarial_scores = validator.run_validation()

    output_dir = config.S3_ARTIFACTS
    output_dir.mkdir(parents=True, exist_ok=True)

    stable_list_path = config.S3_STABLE_FEATURE_LIST
    adversarial_scores_path = config.S3_ADVERSARIAL_SCORES

    joblib.dump(stable_features, stable_list_path)
    joblib.dump(adversarial_scores, adversarial_scores_path)

    stable_df = pd.DataFrame({'feature_name': stable_features})
    stable_df.to_csv(output_dir / 'stable_feature_list.csv', index=False)

    result_info = {
        'total_features': len(stable_features),
        'features': stable_features,
        'validation_type': 'first_defense_line_v3',
        'adversarial_auc_threshold': validator.adversarial_auc_threshold,
        'timestamp': pd.Timestamp.now().isoformat()
    }

    with open(output_dir / 'stable_feature_list.json', 'w') as f:
        json.dump(result_info, f, indent=2)

    logger.info("一次選抜通過特徴量リストを複数形式で保存しました。")
    logger.info(f"- JOBLIB (Stable List): {stable_list_path}")
    logger.info(f"- JOBLIB (Adversarial Scores): {adversarial_scores_path}")
    logger.info(f"- CSV: {output_dir / 'stable_feature_list.csv'}")
    logger.info(f"- JSON: {output_dir / 'stable_feature_list.json'}")