import dask
import dask.dataframe as dd
from dask.dataframe.core import DataFrame
import dask_lightgbm.core as dlgb
import numpy as np
import joblib
from sklearn.metrics import roc_auc_score
from collections import Counter
from tqdm import tqdm

import logging

# 標準のロギング設定
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

from pathlib import Path
import json
import pandas as pd

class WalkForwardValidator:
    """
    【Dask版 v2.0 - Out-of-Core処理対応】
    TB級のマスターテーブルをDask-LightGBMで処理。
    TimeSeriesSplitをDask DataFrameで再現し、ウォークフォワード検証を実行。
    """
    def __init__(self,
                 feature_universe_path: str,
                 stable_features_path: str,
                 n_splits: int = 5,
                 final_feature_count: int = 15,
                 rfe_step: int = 1):  # デフォルトを最も厳密な「1」に設定
        self.feature_universe_path = feature_universe_path
        self.stable_features_path = stable_features_path
        self.n_splits = n_splits
        self.final_feature_count = final_feature_count
        self.ddf: DataFrame | None = None
        self.stable_features: list[str] | None = None
        self.rfe_step = rfe_step # パラメータを保存

    def _load_data(self):
        """Dask DataFrameとしてマスターテーブルをロード"""
        logger.info(f"第二防衛線: マスターテーブル '{self.feature_universe_path}' をDask DataFrameとして読み込み中...")
        
        # Daskで遅延読み込み
        self.ddf = dd.read_parquet(  # type: ignore
            self.feature_universe_path,
            engine='pyarrow'
        )
        
        logger.info(f"一次選抜通過リスト '{self.stable_features_path}' を読み込み中...")
        self.stable_features = joblib.load(self.stable_features_path)
        
        # 幽霊列のチェック（Daskでは列の存在確認のみ）
        available_features = [f for f in self.stable_features if f in self.ddf.columns]
        if len(available_features) != len(self.stable_features):
            removed = len(self.stable_features) - len(available_features)
            logger.warning(f"{removed}個の特徴量がデータセットに存在しないため除外されました。")
            self.stable_features = available_features
        
        logger.info(f"読み込み完了。{len(self.stable_features)}個の安定特徴量を対象とします。")

    def _define_target(self, target_horizon: int = 60, profit_thresh: float = 0.0005):
        """
        ターゲット変数を定義（Dask APIを使用）
        """
        logger.info("予測目標（ターゲット）を計算中...")
        
        # self.ddfがNoneでないことを保証する（Pylanceとコードの安全性の両方の為）
        if self.ddf is None:
            logger.error("Dask DataFrameがロードされていません。_load_data()を先に呼び出す必要があります。")
            raise ValueError("DataFrame not loaded.")
        
        # Dask DataFrameで計算（遅延実行）
        # 将来のリターンを計算
        future_max = self.ddf['close'].shift(-target_horizon).rolling(
            window=target_horizon, 
            min_periods=1
        ).max()
        
        future_returns = (future_max / self.ddf['close']) - 1
        target = (future_returns > profit_thresh).astype('int32')
        
        # ターゲット列を追加
        self.ddf = self.ddf.assign(target=target)
        
        # NaNを除去（遅延実行のまま）
        self.ddf = self.ddf.dropna(subset=['target'])
        self.ddf = self.ddf.dropna()
        
        # パーティション数を確認
        n_partitions = self.ddf.npartitions
        logger.info(f"ターゲット計算完了。{n_partitions}パーティションで処理します。")

    def _create_time_series_splits(self):
        """
        【改善版】TimeSeriesSplitをタイムスタンプベースで再現
        Daskで高負荷なlen()やiloc[]の使用を回避する
        """
        logger.info("TimeSeriesSplit用のタイムスタンプ範囲を計算中...")
        
        if self.ddf is None:
            raise ValueError("Dask DataFrame not loaded.")
        
        # 始点と終点のタイムスタンプを取得（高速な処理）
        min_ts, max_ts = dask.compute(self.ddf['timestamp'].min(), self.ddf['timestamp'].max())  # type: ignore
        total_duration = max_ts - min_ts
        logger.info(f"データセット全体期間: {min_ts} から {max_ts} まで")
        
        # 各分割の期間を計算
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

    def _extract_split_data(self, split_info):
        """
        【改善版】タイムスタンプ範囲に基づいてデータを抽出（Daskネイティブな高速処理）
        """
        if self.ddf is None:
            raise ValueError("Dask DataFrame not loaded.")

        # Daskが得意なブール値フィルタリングを使用
        train_ddf = self.ddf[
            (self.ddf['timestamp'] >= split_info['train_start_ts']) &
            (self.ddf['timestamp'] < split_info['train_end_ts'])
        ]
        test_ddf = self.ddf[
            (self.ddf['timestamp'] >= split_info['test_start_ts']) &
            (self.ddf['timestamp'] < split_info['test_end_ts'])
        ]
        
        return train_ddf, test_ddf

    def _find_best_features_for_split(self, train_ddf, features_to_consider):
        """
        【改善版 v2.1】RFEをサンプリングせず、全データで実行
        ループ内でDataFrameを再生成せず、特徴量リストのみを更新することで効率化
        """
        logger.info("  -> 訓練データの「全データ」を使い、最強チームを選抜中...")
        
        # ターゲットを定義
        y_train = train_ddf['target']
        
        # 削除する特徴量のリストをコピーして使用
        features = list(features_to_consider)
        num_to_eliminate = len(features) - self.final_feature_count
        
        if num_to_eliminate <= 0:
            return features
        
        # RFEループ
        pbar = tqdm(range(num_to_eliminate), desc="  -> RFE Selection (Full Data)", leave=False)
        while len(features) > self.final_feature_count:
            # Dask-LightGBMをループ内で使用
            # 毎回、元のtrain_ddfを更新されたfeaturesリストでスライスする
            model = dlgb.LGBMClassifier(random_state=42, verbosity=-1, n_estimators=50)
            model.fit(train_ddf[features], y_train)
            
            # 重要度を取得
            importances = model.feature_importances_
            
            # 最も重要度が低い特徴量を特定
            worst_feature_idx = np.argmin(importances)
            
            # 特徴量リストから最も重要度の低い特徴量を削除
            # Dask DataFrameオブジェクトは変更しない
            features.pop(worst_feature_idx)
            
            pbar.update(1)
        
        pbar.close()
        return features

    def _train_and_evaluate_split(self, split_idx, split_info, best_features):
        """
        選択された特徴量でモデルを訓練し、テストデータで評価
        """
        logger.info(f"  -> 分割 {split_idx+1} のモデル訓練を開始...")
        
        # データ抽出
        train_ddf, test_ddf = self._extract_split_data(split_info)
        
        # 特徴量とターゲットを準備（type: ignoreでPylanceのエラーを抑制）
        X_train = train_ddf[best_features]  # type: ignore[index]
        y_train = train_ddf['target']  # type: ignore[index]
        X_test = test_ddf[best_features]  # type: ignore[index]
        y_test = test_ddf['target']  # type: ignore[index]
        
        # Dask-LightGBMでモデルを訓練
        logger.info(f"  -> Dask-LightGBMで訓練中（特徴量数: {len(best_features)}）...")
        
        # モデル訓練
        # 外部で起動されたDaskクライアントを自動的に使用します
        model = dlgb.LGBMClassifier(
            n_estimators=50,
            random_state=42,
            verbosity=-1
        )
        
        model.fit(X_train, y_train)
        
        # 予測（テストデータ）
        logger.info(f"  -> テストデータで予測中...")
        y_pred_proba = model.predict_proba(X_test)
        
        # compute()で結果を取得
        y_test_computed, y_pred_proba_computed = dask.compute(y_test, y_pred_proba)  # type: ignore
        
        # Pylanceに型を明確に伝えるためのassert文
        assert isinstance(y_pred_proba_computed, np.ndarray)
        
        # AUCスコアを計算
        oos_score = roc_auc_score(y_test_computed, y_pred_proba_computed[:, 1])
        
        logger.info(f"  -> テストデータでの性能（AUC）: {oos_score:.4f}")
        
        return oos_score

    def run_validation(self):
        """
        ウォークフォワード検証パイプラインを実行
        """
        self._load_data()
        self._define_target()
        
        # TimeSeriesSplitの作成
        splits = self._create_time_series_splits()
        
        selected_features_across_splits = []
        
        logger.info(f"--- 第二防衛線: 全{self.n_splits}分割のウォークフォワード検証を開始 ---")
        
        for i, split_info in enumerate(tqdm(splits, desc="Walk-Forward Validation")):
            logger.info(f"--- [分割 {i+1}/{self.n_splits}] ---")
            
            # データ抽出
            train_ddf, test_ddf = self._extract_split_data(split_info)
            
            # 訓練データのサイズを表示
            train_len = len(train_ddf)
            test_len = len(test_ddf)
            logger.info(f"  -> 訓練データ: {train_len}行、テストデータ: {test_len}行")
            
            # 最適な特徴量セットを選択
            best_features_for_this_split = self._find_best_features_for_split(
                train_ddf, 
                self.stable_features
            )
            
            selected_features_across_splits.extend(best_features_for_this_split)
            logger.info(f"  -> 分割 {i+1} の最強チーム（{len(best_features_for_this_split)}名）を選出。")
            
            # モデルを訓練・評価
            _ = self._train_and_evaluate_split(i, split_info, best_features_for_this_split)
        
        # 最も頻繁に選ばれた特徴量を最終チームとする
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
    # Daskの設定
    dask.config.set({'dataframe.query-planning': True})
    
    # Daskクライアントをここで一度だけ起動
    from dask.distributed import Client, LocalCluster  # type: ignore
    
    # withブロックを使うことで、処理終了後にクライアントが自動的にクリーンアップされる
    with LocalCluster(n_workers=4, threads_per_worker=2, memory_limit='8GB') as cluster, \
         Client(cluster) as client:
        
        logger.info(f"Daskクライアントを起動しました: {client.dashboard_link}")
        
        validator = WalkForwardValidator(
            feature_universe_path='data/master_table_partitioned',
            stable_features_path='data/temp_chunks/defense_results/joblib/stable_feature_list.joblib'
        )
        final_feature_team = validator.run_validation()
    
    # 出力ディレクトリの作成
    output_dir = Path('data/temp_chunks/defense_results')
    (output_dir / 'joblib').mkdir(parents=True, exist_ok=True)
    (output_dir / 'csv').mkdir(parents=True, exist_ok=True)
    (output_dir / 'json').mkdir(parents=True, exist_ok=True)
    
    # 複数形式で保存
    joblib.dump(final_feature_team, output_dir / 'joblib' / 'final_feature_team.joblib')
    
    # CSV形式で保存（人間確認用）
    final_df = pd.DataFrame({'feature_name': final_feature_team})
    final_df.to_csv(output_dir / 'csv' / 'final_feature_team.csv', index=False)
    
    # JSON形式で保存（メタデータ含む）
    result_info = {
        'total_features': len(final_feature_team),
        'features': final_feature_team,
        'validation_type': 'second_defense_line',
        'timestamp': pd.Timestamp.now().isoformat()
    }
    with open(output_dir / 'json' / 'final_feature_team.json', 'w') as f:
        json.dump(result_info, f, indent=2)
    
    logger.info("最終選抜メンバーリストを複数形式で保存しました。")
    logger.info(f"- JOBLIB: {output_dir / 'joblib' / 'final_feature_team.joblib'}")
    logger.info(f"- CSV: {output_dir / 'csv' / 'final_feature_team.csv'}")
    logger.info(f"- JSON: {output_dir / 'json' / 'final_feature_team.json'}")