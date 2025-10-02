import dask
import dask.dataframe as dd
from dask.dataframe.core import DataFrame
import dask_lightgbm.core as dlgb
import numpy as np
import pandas as pd
import joblib
import optuna  # type: ignore
from optuna.trial import Trial  # type: ignore
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, mean_squared_error, mean_absolute_error
from pathlib import Path
import json
import logging
from typing import Dict, Any, List, Tuple, Optional

# ロギング設定
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class DualModelTrainer:
    """
    【Dask-LightGBM版】デュアルモデル・アーキテクチャ
    分類モデル（バリア到達結果）と回帰モデル（到達時間）を同時に訓練
    """
    
    def __init__(self,
                 input_path: str,
                 output_dir: str,
                 n_splits: int = 5,
                 n_trials: int = 100):
        """
        Args:
            input_path: ラベル付きデータセットのパス（Parquet）
            output_dir: モデル・レポート保存ディレクトリ
            n_splits: TimeSeriesSplitの分割数
            n_trials: Optunaの試行回数
        """
        self.input_path = input_path
        self.output_dir = Path(output_dir)
        self.n_splits = n_splits
        self.n_trials = n_trials
        self.ddf: Optional[DataFrame] = None
        self.feature_cols: List[str] = []
        
    def _load_data(self) -> None:
        """ラベル付きデータセットを読み込む"""
        logger.info(f"ラベル付きデータセット '{self.input_path}' を読み込み中...")
        
        self.ddf = dd.read_parquet(  # type: ignore
            self.input_path,
            engine='pyarrow'
        )
        
        # 必須カラムの確認
        required_cols = ['label', 'time_to_barrier', 'close', 'timestamp']
        missing_cols = [col for col in required_cols if col not in self.ddf.columns]
        
        if missing_cols:
            raise ValueError(f"必須カラムが見つかりません: {missing_cols}")
        
        # 特徴量カラムを特定（OHLCV、ラベル、メタデータを除外）
        exclude_cols = ['open', 'high', 'low', 'close', 'volume', 
                       'label', 'barrier_reached', 'time_to_barrier', 'timestamp']
        self.feature_cols = [col for col in self.ddf.columns if col not in exclude_cols]
        
        logger.info(f"データ読み込み完了。特徴量数: {len(self.feature_cols)}, パーティション数: {self.ddf.npartitions}")
    
    def _create_time_series_splits(self) -> List[Dict[str, Any]]:
        """TimeSeriesSplitをタイムスタンプベースで作成"""
        if self.ddf is None:
            raise ValueError("データが読み込まれていません。")
        
        logger.info(f"TimeSeriesSplit（{self.n_splits}分割）の範囲を計算中...")
        
        min_ts, max_ts = dask.compute(self.ddf['timestamp'].min(), self.ddf['timestamp'].max())  # type: ignore
        total_duration = max_ts - min_ts
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
        
        logger.info(f"{self.n_splits}個のタイムスタンプ分割を作成完了。")
        return splits
    
    def _extract_split_data(self, split_info: Dict[str, Any]) -> Tuple[DataFrame, DataFrame]:
        """タイムスタンプ範囲に基づいてデータを抽出"""
        if self.ddf is None:
            raise ValueError("データが読み込まれていません。")
        
        train_ddf = self.ddf[
            (self.ddf['timestamp'] >= split_info['train_start_ts']) &
            (self.ddf['timestamp'] < split_info['train_end_ts'])
        ]
        test_ddf = self.ddf[
            (self.ddf['timestamp'] >= split_info['test_start_ts']) &
            (self.ddf['timestamp'] < split_info['test_end_ts'])
        ]
        
        return train_ddf, test_ddf
    
    def _objective_classifier(self, trial: Trial, splits: List[Dict[str, Any]]) -> float:
        """分類モデルの目的関数（Optuna用）"""
        # ハイパーパラメータの提案
        params = {
            'n_estimators': trial.suggest_int('clf_n_estimators', 100, 2000),
            'max_depth': trial.suggest_int('clf_max_depth', 3, 15),
            'learning_rate': trial.suggest_float('clf_learning_rate', 0.01, 0.3),
            'num_leaves': trial.suggest_int('clf_num_leaves', 31, 255),
            'min_child_samples': trial.suggest_int('clf_min_child_samples', 20, 500),
            'subsample': trial.suggest_float('clf_subsample', 0.5, 1.0),
            'colsample_bytree': trial.suggest_float('clf_colsample_bytree', 0.5, 1.0),
            'random_state': 42,
            'verbosity': -1
        }
        
        auc_scores = []
        
        for i, split_info in enumerate(splits):
            train_ddf, test_ddf = self._extract_split_data(split_info)
            
            X_train = train_ddf[self.feature_cols]
            y_train = train_ddf['label']
            X_test = test_ddf[self.feature_cols]
            y_test = test_ddf['label']
            
            # モデル訓練
            model = dlgb.LGBMClassifier(**params)
            model.fit(X_train, y_train)
            
            # 予測
            y_pred_proba = model.predict_proba(X_test)
            
            # 計算実行
            y_test_computed, y_pred_proba_computed = dask.compute(y_test, y_pred_proba)  # type: ignore
            
            # AUCスコア計算（3クラスなのでOvR方式）
            try:
                auc = roc_auc_score(y_test_computed, y_pred_proba_computed, multi_class='ovr')
                auc_scores.append(auc)
            except Exception as e:
                logger.warning(f"分割{i+1}でAUC計算エラー: {e}")
                auc_scores.append(0.0)
        
        mean_auc = np.mean(auc_scores)
        return mean_auc
    
    def _objective_regressor(self, trial: Trial, splits: List[Dict[str, Any]]) -> float:
        """回帰モデルの目的関数（Optuna用）"""
        # ハイパーパラメータの提案
        params = {
            'n_estimators': trial.suggest_int('reg_n_estimators', 100, 2000),
            'max_depth': trial.suggest_int('reg_max_depth', 3, 15),
            'learning_rate': trial.suggest_float('reg_learning_rate', 0.01, 0.3),
            'num_leaves': trial.suggest_int('reg_num_leaves', 31, 255),
            'min_child_samples': trial.suggest_int('reg_min_child_samples', 20, 500),
            'subsample': trial.suggest_float('reg_subsample', 0.5, 1.0),
            'colsample_bytree': trial.suggest_float('reg_colsample_bytree', 0.5, 1.0),
            'random_state': 42,
            'verbosity': -1
        }
        
        rmse_scores = []
        
        for i, split_info in enumerate(splits):
            train_ddf, test_ddf = self._extract_split_data(split_info)
            
            X_train = train_ddf[self.feature_cols]
            y_train = train_ddf['time_to_barrier']
            X_test = test_ddf[self.feature_cols]
            y_test = test_ddf['time_to_barrier']
            
            # モデル訓練
            model = dlgb.LGBMRegressor(**params)
            model.fit(X_train, y_train)
            
            # 予測
            y_pred = model.predict(X_test)
            
            # 計算実行
            y_test_computed, y_pred_computed = dask.compute(y_test, y_pred)  # type: ignore
            
            # RMSEスコア計算
            rmse = np.sqrt(mean_squared_error(y_test_computed, y_pred_computed))
            rmse_scores.append(rmse)
        
        mean_rmse = np.mean(rmse_scores)
        return mean_rmse
    
    def optimize_and_train(self) -> Tuple[Any, Any, Dict[str, Any], Dict[str, Any]]:
        """Optunaでハイパーパラメータ最適化し、最終モデルを訓練"""
        self._load_data()
        splits = self._create_time_series_splits()
        
        # 出力ディレクトリの作成
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # === 分類モデルの最適化 ===
        logger.info("=" * 50)
        logger.info("分類モデルのハイパーパラメータ最適化を開始...")
        logger.info("=" * 50)
        
        study_clf = optuna.create_study(direction='maximize', study_name='classifier_optimization')
        study_clf.optimize(lambda trial: self._objective_classifier(trial, splits), n_trials=self.n_trials)
        
        best_params_clf = study_clf.best_params
        logger.info(f"分類モデルの最適パラメータ: {best_params_clf}")
        logger.info(f"最高AUC: {study_clf.best_value:.4f}")
        
        # === 回帰モデルの最適化 ===
        logger.info("=" * 50)
        logger.info("回帰モデルのハイパーパラメータ最適化を開始...")
        logger.info("=" * 50)
        
        study_reg = optuna.create_study(direction='minimize', study_name='regressor_optimization')
        study_reg.optimize(lambda trial: self._objective_regressor(trial, splits), n_trials=self.n_trials)
        
        best_params_reg = study_reg.best_params
        logger.info(f"回帰モデルの最適パラメータ: {best_params_reg}")
        logger.info(f"最小RMSE: {study_reg.best_value:.4f}")
        
        # === 最終モデルの訓練（全データ使用） ===
        logger.info("=" * 50)
        logger.info("最適パラメータで最終モデルを訓練中...")
        logger.info("=" * 50)
        
        if self.ddf is None:
            raise ValueError("データが読み込まれていません。")
        
        X_full = self.ddf[self.feature_cols]
        y_clf_full = self.ddf['label']
        y_reg_full = self.ddf['time_to_barrier']
        
        # 分類モデル
        final_clf = dlgb.LGBMClassifier(**best_params_clf)
        final_clf.fit(X_full, y_clf_full)
        
        # 回帰モデル
        final_reg = dlgb.LGBMRegressor(**best_params_reg)
        final_reg.fit(X_full, y_reg_full)
        
        logger.info("最終モデルの訓練完了。")
        
        # === モデルの保存 ===
        logger.info("モデルを保存中...")
        
        # テキスト形式で保存
        final_clf.booster_.save_model(str(self.output_dir / 'classifier_model.txt'))
        final_reg.booster_.save_model(str(self.output_dir / 'regressor_model.txt'))
        
        # Pickle形式で保存
        joblib.dump(final_clf, self.output_dir / 'classifier_model.pkl')
        joblib.dump(final_reg, self.output_dir / 'regressor_model.pkl')
        
        # Optunaスタディの保存
        joblib.dump(study_clf, self.output_dir / 'optuna_study_classifier.pkl')
        joblib.dump(study_reg, self.output_dir / 'optuna_study_regressor.pkl')
        
        logger.info(f"モデル保存完了: {self.output_dir}")
        
        return final_clf, final_reg, best_params_clf, best_params_reg
    
    def evaluate_and_save_report(self, clf_model: Any, reg_model: Any, 
                                best_params_clf: Dict[str, Any], 
                                best_params_reg: Dict[str, Any]) -> None:
        """モデルを評価し、性能レポートを保存"""
        logger.info("=" * 50)
        logger.info("最終モデルの性能評価を実行中...")
        logger.info("=" * 50)
        
        splits = self._create_time_series_splits()
        
        clf_metrics: List[Dict[str, float]] = []
        reg_metrics: List[Dict[str, float]] = []
        
        for i, split_info in enumerate(splits):
            logger.info(f"分割 {i+1}/{self.n_splits} を評価中...")
            
            train_ddf, test_ddf = self._extract_split_data(split_info)
            
            X_test = test_ddf[self.feature_cols]
            y_clf_test = test_ddf['label']
            y_reg_test = test_ddf['time_to_barrier']
            
            # 分類モデルの評価
            y_clf_pred_proba = clf_model.predict_proba(X_test)
            y_clf_pred = clf_model.predict(X_test)
            
            y_clf_test_computed, y_clf_pred_proba_computed, y_clf_pred_computed = dask.compute(  # type: ignore
                y_clf_test, y_clf_pred_proba, y_clf_pred
            )
            
            auc = roc_auc_score(y_clf_test_computed, y_clf_pred_proba_computed, multi_class='ovr')
            accuracy = accuracy_score(y_clf_test_computed, y_clf_pred_computed)
            precision = precision_score(y_clf_test_computed, y_clf_pred_computed, average='macro')
            recall = recall_score(y_clf_test_computed, y_clf_pred_computed, average='macro')
            
            clf_metrics.append({
                'split': i + 1,
                'auc': float(auc),
                'accuracy': float(accuracy),
                'precision': float(precision),
                'recall': float(recall)
            })
            
            # 回帰モデルの評価
            y_reg_pred = reg_model.predict(X_test)
            y_reg_test_computed, y_reg_pred_computed = dask.compute(y_reg_test, y_reg_pred)  # type: ignore
            
            rmse = np.sqrt(mean_squared_error(y_reg_test_computed, y_reg_pred_computed))
            mae = mean_absolute_error(y_reg_test_computed, y_reg_pred_computed)
            
            reg_metrics.append({
                'split': i + 1,
                'rmse': float(rmse),
                'mae': float(mae)
            })
        
        # 平均メトリクスの計算
        avg_clf_metrics = {
            'auc': np.mean([m['auc'] for m in clf_metrics]),
            'accuracy': np.mean([m['accuracy'] for m in clf_metrics]),
            'precision': np.mean([m['precision'] for m in clf_metrics]),
            'recall': np.mean([m['recall'] for m in clf_metrics])
        }
        
        avg_reg_metrics = {
            'rmse': np.mean([m['rmse'] for m in reg_metrics]),
            'mae': np.mean([m['mae'] for m in reg_metrics])
        }
        
        # 特徴量重要度
        feature_importance_clf = dict(zip(self.feature_cols, clf_model.feature_importances_))
        feature_importance_reg = dict(zip(self.feature_cols, reg_model.feature_importances_))
        
        # ソートして上位20個を取得
        top_features_clf = dict(sorted(feature_importance_clf.items(), key=lambda x: x[1], reverse=True)[:20])
        top_features_reg = dict(sorted(feature_importance_reg.items(), key=lambda x: x[1], reverse=True)[:20])
        
        # レポート作成
        report = {
            'classifier': {
                'best_params': best_params_clf,
                'cv_metrics': clf_metrics,
                'avg_metrics': avg_clf_metrics,
                'top_features': {k: float(v) for k, v in top_features_clf.items()}
            },
            'regressor': {
                'best_params': best_params_reg,
                'cv_metrics': reg_metrics,
                'avg_metrics': avg_reg_metrics,
                'top_features': {k: float(v) for k, v in top_features_reg.items()}
            },
            'training_info': {
                'n_features': len(self.feature_cols),
                'n_splits': self.n_splits,
                'n_trials': self.n_trials,
                'timestamp': pd.Timestamp.now().isoformat()
            }
        }
        
        # JSONで保存
        report_path = self.output_dir / 'model_performance_report.json'
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        logger.info("=" * 50)
        logger.info("🎉 モデル訓練・評価完了 🎉")
        logger.info("=" * 50)
        logger.info(f"分類モデル - 平均AUC: {avg_clf_metrics['auc']:.4f}, 精度: {avg_clf_metrics['accuracy']:.4f}")
        logger.info(f"回帰モデル - 平均RMSE: {avg_reg_metrics['rmse']:.4f}, MAE: {avg_reg_metrics['mae']:.4f}")
        logger.info(f"性能レポート保存: {report_path}")
        logger.info("=" * 50)
    
    def run(self) -> None:
        """パイプライン全体を実行"""
        clf_model, reg_model, best_params_clf, best_params_reg = self.optimize_and_train()
        self.evaluate_and_save_report(clf_model, reg_model, best_params_clf, best_params_reg)


if __name__ == '__main__':
    # Daskの設定
    dask.config.set({'dataframe.query-planning': True})
    
    # Daskクライアントの起動
    from dask.distributed import Client, LocalCluster  # type: ignore
    
    with LocalCluster(n_workers=4, threads_per_worker=2, memory_limit='8GB') as cluster, \
         Client(cluster) as client:
        
        logger.info(f"Daskクライアントを起動: {client.dashboard_link}")
        
        trainer = DualModelTrainer(
            input_path='data/temp_chunks/training_data/labeled_dataset_partitioned',
            output_dir='data/models',
            n_splits=5,
            n_trials=100
        )
        
        trainer.run()