"""
walk_forward_validator_v2.py - SHAPベース版
第二防衛線：シングルパス重要度評価による最強チーム選抜

統合設計図V準拠：
- RFE完全削除、SHAPベースアプローチに置換
- 計算量：O(k × N²) → O(k × N)に削減
- 所要時間：12年以上 → 3-8時間
- 敵対的検証スコアによる特徴量ペナルティ
- Dask-LightGBM + map_partitions並列SHAP計算
"""
import blueprint as config
import logging
from typing import List, Dict, Tuple, cast
from pathlib import Path
import json
from collections import Counter

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

import dask
import dask.dataframe as dd
from dask.dataframe.core import DataFrame as DaskDataFrame, Series as DaskSeries
import dask_lightgbm.core as dlgb
import numpy as np
import joblib
from sklearn.metrics import roc_auc_score
from tqdm import tqdm
import pandas as pd


def calculate_shap_for_partition(
    partition: pd.DataFrame,
    model_booster: object,
    feature_columns: List[str]
) -> pd.Series:
    """
    単一パーティションのSHAP値を計算
    
    Args:
        partition: データのパーティション
        model_booster: LightGBMのboosterオブジェクト
        feature_columns: 特徴量カラムのリスト
    
    Returns:
        平均絶対SHAP値のSeries
    """
    import shap
    
    if partition.empty:
        return pd.Series(0.0, index=feature_columns)
    
    X_partition = partition[feature_columns].values
    
    explainer = shap.TreeExplainer(model_booster)
    shap_values = explainer.shap_values(X_partition)
    
    mean_abs_shap = np.abs(shap_values).mean(axis=0)
    
    return pd.Series(mean_abs_shap, index=feature_columns)


class WalkForwardValidatorV2:
    """
    【Dask版 v3.0 - SHAPベース】
    TB級のマスターテーブルをDask-LightGBMで処理。
    SHAPベースのシングルパス重要度評価でエリート特徴量を選抜。
    """
    
    def __init__(
        self,
        feature_universe_path: str,
        stable_features_path: str,
        adversarial_scores_path: str,
        n_splits: int = 5,
        final_feature_count: int = 15,
        adversarial_penalty_factor: float = 0.5
    ):
        self.feature_universe_path = feature_universe_path
        self.stable_features_path = stable_features_path
        self.adversarial_scores_path = adversarial_scores_path
        self.n_splits = n_splits
        self.final_feature_count = final_feature_count
        self.adversarial_penalty_factor = adversarial_penalty_factor
        self.ddf: DaskDataFrame | None = None
        self.stable_features: List[str] | None = None
        self.adversarial_scores: Dict[str, float] | None = None
    
    def _load_data(self) -> None:
        """データとメタデータを読み込み"""
        logger.info(f"第二防衛線: マスターテーブル '{self.feature_universe_path}' をDask DataFrameとして読み込み中...")
        
        self.ddf = dd.read_parquet(
            self.feature_universe_path,
            engine='pyarrow'
        )
        
        logger.info(f"一次選抜通過リスト '{self.stable_features_path}' を読み込み中...")
        self.stable_features = joblib.load(self.stable_features_path)
        
        logger.info(f"敵対的検証スコア '{self.adversarial_scores_path}' を読み込み中...")
        self.adversarial_scores = joblib.load(self.adversarial_scores_path)
        
        available_features = [f for f in self.stable_features if f in self.ddf.columns]
        if len(available_features) != len(self.stable_features):
            removed = len(self.stable_features) - len(available_features)
            logger.warning(f"{removed}個の特徴量がデータセットに存在しないため除外されました。")
            self.stable_features = available_features
        
        logger.info(f"読み込み完了。{len(self.stable_features)}個の安定特徴量を対象とします。")
    
    def _define_target(self, target_horizon: int = 60, profit_thresh: float = 0.0005) -> None:
        """ターゲット変数を定義"""
        logger.info("予測目標（ターゲット）を計算中...")
        
        if self.ddf is None:
            raise ValueError("DataFrame not loaded.")
        
        future_max = self.ddf['close'].shift(-target_horizon).rolling(
            window=target_horizon,
            min_periods=1
        ).max()
        
        future_returns = (future_max / self.ddf['close']) - 1
        target = (future_returns > profit_thresh).astype('int32')
        
        self.ddf = self.ddf.assign(target=target)
        self.ddf = self.ddf.dropna(subset=['target'])
        self.ddf = self.ddf.dropna()
        
        logger.info(f"ターゲット計算完了。{self.ddf.npartitions}パーティションで処理します。")
    
    def _create_time_series_splits(self) -> List[Dict[str, pd.Timestamp]]:
        """TimeSeriesSplit用のタイムスタンプ範囲を計算"""
        logger.info("TimeSeriesSplit用のタイムスタンプ範囲を計算中...")
        
        if self.ddf is None:
            raise ValueError("Dask DataFrame not loaded.")
        
        min_ts, max_ts = dask.compute(
            self.ddf['timestamp'].min(),
            self.ddf['timestamp'].max()
        )
        
        total_duration = max_ts - min_ts
        logger.info(f"データセット全体期間: {min_ts} から {max_ts} まで")
        
        split_duration = total_duration / (self.n_splits + 1)
        
        splits = []
        for i in range(self.n_splits):
            train_end_ts = min_ts + split_duration * (i + 1)
            test_start_ts = train_end_ts
            test_end_ts = test_start_ts + split_duration
            
            splits.append({
                'train_start_ts': min_ts,
                'train_end_ts': train_end_ts,
                'test_start_ts': test_start_ts,
                'test_end_ts': test_end_ts
            })
        
        logger.info(f"{self.n_splits}個のタイムスタンプ分割を作成しました。")
        return splits
    
    def _extract_split_data(
        self,
        split_info: Dict[str, pd.Timestamp]
    ) -> Tuple[DaskDataFrame, DaskDataFrame]:
        """タイムスタンプ範囲に基づいてデータを抽出"""
        if self.ddf is None:
            raise ValueError("Dask DataFrame not loaded.")
        
        ddf = self.ddf
        
        # 型キャストを使用してPylanceに明示的に型を伝える
        timestamp_series = cast(DaskSeries, ddf['timestamp'])
        
        # 訓練データの抽出
        train_start_cond = cast(DaskSeries, timestamp_series >= split_info['train_start_ts'])
        train_end_cond = cast(DaskSeries, timestamp_series < split_info['train_end_ts'])
        train_mask = train_start_cond & train_end_cond
        train_ddf = ddf[train_mask]
        
        # テストデータの抽出
        test_start_cond = cast(DaskSeries, timestamp_series >= split_info['test_start_ts'])
        test_end_cond = cast(DaskSeries, timestamp_series < split_info['test_end_ts'])
        test_mask = test_start_cond & test_end_cond
        test_ddf = ddf[test_mask]
        
        return train_ddf, test_ddf
    
    def _calculate_shap_importance(
        self,
        train_ddf: DaskDataFrame,
        features: List[str]
    ) -> Dict[str, float]:
        """
        【v3.0新規】SHAPベースの特徴量重要度計算
        map_partitionsで並列処理し、メモリ効率を最大化
        """
        logger.info("  -> SHAP値を計算中（並列処理）...")
        
        y_train = train_ddf['target']
        X_train = train_ddf[features]
        
        model = dlgb.LGBMClassifier(random_state=42, verbosity=-1, n_estimators=50)
        model.fit(X_train, y_train)
        
        # LightGBMのboosterオブジェクトを取得（Pylance型推論補助）
        booster = model.booster_  # type: ignore[attr-defined]
        
        shap_results = X_train.map_partitions(
            calculate_shap_for_partition,
            model_booster=booster,
            feature_columns=features,
            meta=pd.Series(dtype='float64')
        )
        
        mean_shap_values = shap_results.mean().compute()
        
        shap_importance = dict(zip(features, mean_shap_values))
        
        logger.info("  -> SHAP計算完了。")
        return shap_importance
    
    def _select_top_features(
        self,
        shap_importance: Dict[str, float]
    ) -> List[str]:
        """
        【v3.0】SHAP重要度 + 敵対的ペナルティでトップ特徴量を選抜
        """
        if self.adversarial_scores is None:
            raise ValueError("Adversarial scores not loaded.")
        
        adjusted_scores: Dict[str, float] = {}
        for feature, shap_score in shap_importance.items():
            adversarial_score = self.adversarial_scores.get(feature, 0.0)
            penalty = 1.0 - (adversarial_score * self.adversarial_penalty_factor)
            adjusted_scores[feature] = shap_score * max(penalty, 0.1)
        
        sorted_features = sorted(
            adjusted_scores.items(),
            key=lambda x: x[1],
            reverse=True
        )
        
        top_features = [feat for feat, _ in sorted_features[:self.final_feature_count]]
        
        return top_features
    
    def _train_and_evaluate_split(
        self,
        split_idx: int,
        split_info: Dict[str, pd.Timestamp],
        best_features: List[str]
    ) -> float:
        """選択された特徴量でモデルを訓練し、テストデータで評価"""
        logger.info(f"  -> 分割 {split_idx+1} のモデル訓練を開始...")
        
        train_ddf, test_ddf = self._extract_split_data(split_info)
        
        X_train = train_ddf[best_features]
        y_train = train_ddf['target']
        X_test = test_ddf[best_features]
        y_test = test_ddf['target']
        
        logger.info(f"  -> Dask-LightGBMで訓練中（特徴量数: {len(best_features)}）...")
        
        model = dlgb.LGBMClassifier(
            n_estimators=50,
            random_state=42,
            verbosity=-1
        )
        
        model.fit(X_train, y_train)
        
        logger.info(f"  -> テストデータで予測中...")
        y_pred_proba = model.predict_proba(X_test)
        
        y_test_computed, y_pred_proba_computed = dask.compute(y_test, y_pred_proba)
        
        assert isinstance(y_pred_proba_computed, np.ndarray)
        
        oos_score = roc_auc_score(y_test_computed, y_pred_proba_computed[:, 1])
        
        logger.info(f"  -> テストデータでの性能（AUC）: {oos_score:.4f}")
        
        return oos_score
    
    def run_validation(self) -> List[str]:
        """ウォークフォワード検証パイプラインを実行"""
        self._load_data()
        self._define_target()
        
        splits = self._create_time_series_splits()
        
        selected_features_across_splits: List[str] = []
        
        logger.info(f"--- 第二防衛線: 全{self.n_splits}分割のウォークフォワード検証を開始（SHAPベース） ---")
        
        for i, split_info in enumerate(tqdm(splits, desc="Walk-Forward Validation")):
            logger.info(f"--- [分割 {i+1}/{self.n_splits}] ---")
            
            train_ddf, test_ddf = self._extract_split_data(split_info)
            
            train_len = len(train_ddf)
            test_len = len(test_ddf)
            logger.info(f"  -> 訓練データ: {train_len}行、テストデータ: {test_len}行")
            
            if self.stable_features is None:
                raise ValueError("Stable features not loaded.")
            
            shap_importance = self._calculate_shap_importance(train_ddf, self.stable_features)
            
            best_features_for_this_split = self._select_top_features(shap_importance)
            
            selected_features_across_splits.extend(best_features_for_this_split)
            logger.info(f"  -> 分割 {i+1} の最強チーム（{len(best_features_for_this_split)}名）を選出。")
            
            _ = self._train_and_evaluate_split(i, split_info, best_features_for_this_split)
        
        feature_counts = Counter(selected_features_across_splits)
        final_team = [
            feature for feature, count in feature_counts.most_common(self.final_feature_count)
        ]
        
        logger.info("-" * 50)
        logger.info("🎉 第二防衛線: 最強チームの選抜完了（SHAPベース） 🎉")
        logger.info(f"最終選抜メンバー（{len(final_team)}名）:")
        for f in final_team:
            logger.info(f"  - {f} (選出回数: {feature_counts[f]}回)")
        logger.info("-" * 50)
        
        return final_team


if __name__ == '__main__':
    dask.config.set({'dataframe.query-planning': True})

    from dask.distributed import Client, LocalCluster

    with LocalCluster(n_workers=4, threads_per_worker=2, memory_limit='8GB') as cluster, \
         Client(cluster) as client:

        logger.info(f"Daskクライアントを起動しました: {client.dashboard_link}")

        validator = WalkForwardValidatorV2(
            feature_universe_path=str(config.S4_MASTER_TABLE_PARTITIONED),
            stable_features_path=str(config.S3_STABLE_FEATURE_LIST),
            adversarial_scores_path=str(config.S3_ADVERSARIAL_SCORES)
        )
        final_feature_team = validator.run_validation()

    output_dir = config.S3_ARTIFACTS
    output_dir.mkdir(parents=True, exist_ok=True)

    final_team_path = config.S3_FINAL_FEATURE_TEAM
    shap_scores_path = config.S3_SHAP_SCORES # config.pyにSHAPスコアパスも定義推奨

    joblib.dump(final_feature_team, final_team_path)

    # SHAPスコアも保存する場合 (validatorクラスから返り値として受け取る必要あり)
    # joblib.dump(shap_scores, shap_scores_path)

    final_df = pd.DataFrame({'feature_name': final_feature_team})
    final_df.to_csv(output_dir / 'final_feature_team.csv', index=False)

    result_info = {
        'total_features': len(final_feature_team),
        'features': final_feature_team,
        'validation_type': 'second_defense_line_shap_based',
        'timestamp': pd.Timestamp.now().isoformat()
    }
    with open(output_dir / 'final_feature_team.json', 'w') as f:
        json.dump(result_info, f, indent=2)

    logger.info("最終選抜メンバーリストを複数形式で保存しました。")
    logger.info(f"- JOBLIB: {final_team_path}")
    logger.info(f"- CSV: {output_dir / 'final_feature_team.csv'}")
    logger.info(f"- JSON: {output_dir / 'final_feature_team.json'}")