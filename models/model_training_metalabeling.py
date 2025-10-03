import dask
import dask.dataframe as dd
from dask.dataframe.core import DataFrame
import dask_lightgbm.core as dlgb
import numpy as np
import pandas as pd
import joblib
import optuna  # type: ignore
from optuna.trial import Trial  # type: ignore
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import BaseCrossValidator
from pathlib import Path
import json
import logging
from typing import Dict, Any, List, Tuple, Optional, Iterator
import warnings

warnings.filterwarnings('ignore', category=FutureWarning)

# ロギング設定
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class PurgedKFold(BaseCrossValidator):
    """
    パージ＆エンバーゴ付きK分割交差検証（統合設計図V準拠）
    
    Args:
        n_splits: 分割数
        t0_col: イベント開始時刻のカラム名
        t1_col: イベント終了時刻のカラム名
        embargo_pct: エンバーゴ期間の割合（0-1）
    """
    
    def __init__(self, n_splits: int = 5, t0_col: str = 't0', 
                 t1_col: str = 't1', embargo_pct: float = 0.01):
        self.n_splits = n_splits
        self.t0_col = t0_col
        self.t1_col = t1_col
        self.embargo_pct = embargo_pct
    
    def get_n_splits(self, X: Any = None, y: Any = None, groups: Any = None) -> int:
        return self.n_splits
    
    def split(self, X: pd.DataFrame, y: Any = None, groups: Any = None) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
        """
        K分割交差検証のインデックスを生成（パージ＆エンバーゴ適用）
        """
        if self.t0_col not in X.columns or self.t1_col not in X.columns:
            raise ValueError(f"DataFrameに{self.t0_col}と{self.t1_col}カラムが必要です")
        
        indices = np.arange(len(X))
        t0_values = X[self.t0_col].values
        t1_values = X[self.t1_col].values
        
        # タイムスタンプでソート
        sorted_idx = np.argsort(t0_values)
        indices = indices[sorted_idx]
        t0_sorted = t0_values[sorted_idx]
        t1_sorted = t1_values[sorted_idx]
        
        test_ranges = []
        n_samples = len(indices)
        test_size = n_samples // self.n_splits
        
        for i in range(self.n_splits):
            test_start = i * test_size
            test_end = (i + 1) * test_size if i < self.n_splits - 1 else n_samples
            test_ranges.append((test_start, test_end))
        
        for test_start, test_end in test_ranges:
            # テストセットのインデックス
            test_indices = indices[test_start:test_end]
            
            # テストセットの最小・最大時刻
            test_t0_min = t0_sorted[test_start]
            test_t1_max = t1_sorted[test_end - 1]
            
            # パージング: 訓練セットからテスト期間と重複するサンプルを除去
            train_mask = np.ones(n_samples, dtype=bool)
            train_mask[test_start:test_end] = False  # テストセット自体を除外
            
            # 重複チェック: t1_train > test_t0_min AND t0_train < test_t1_max
            for j in range(n_samples):
                if train_mask[j]:
                    if t1_sorted[j] > test_t0_min and t0_sorted[j] < test_t1_max:
                        train_mask[j] = False
            
            # エンバーゴ: テストセット直後のサンプルも除外
            embargo_samples = int(n_samples * self.embargo_pct)
            if test_end + embargo_samples < n_samples:
                train_mask[test_end:test_end + embargo_samples] = False
            
            train_indices = indices[train_mask]
            
            yield train_indices, test_indices


class MetaLabelingTrainer:
    """
    【Dask-LightGBM版】メタラベリング・トレーナー（統合設計図V準拠）
    M1（方向性）とM2（確信度）の二段階モデルを訓練
    """
    
    def __init__(self,
                 input_path: str,
                 output_dir: str,
                 n_splits: int = 5,
                 n_trials: int = 100,
                 embargo_pct: float = 0.01):
        """
        Args:
            input_path: サンプルウェイト付きデータセットのパス
            output_dir: モデル・レポート保存ディレクトリ
            n_splits: K分割交差検証の分割数
            n_trials: Optunaの試行回数
            embargo_pct: エンバーゴ期間の割合
        """
        self.input_path = input_path
        self.output_dir = Path(output_dir)
        self.n_splits = n_splits
        self.n_trials = n_trials
        self.embargo_pct = embargo_pct
        self.ddf: Optional[DataFrame] = None
        self.feature_cols: List[str] = []
        self.df_pandas: Optional[pd.DataFrame] = None  # パージ付きCVのため必要
    
    def _load_data(self) -> None:
        """データセットを読み込む"""
        logger.info(f"データセット '{self.input_path}' を読み込み中...")
        
        self.ddf = dd.read_parquet(  # type: ignore
            self.input_path,
            engine='pyarrow'
        )
        
        # 必須カラムの確認
        required_cols = ['label', 't0', 't1', 'sample_weight']
        missing_cols = [col for col in required_cols if col not in self.ddf.columns]
        
        if missing_cols:
            raise ValueError(f"必須カラムが見つかりません: {missing_cols}")
        
        # 特徴量カラムを特定
        exclude_cols = ['open', 'high', 'low', 'close', 'volume', 
                       'label', 'barrier_reached', 'time_to_barrier', 
                       'timestamp', 't0', 't1', 'concurrency', 'uniqueness', 
                       'sample_weight', 'abs_return']
        self.feature_cols = [col for col in self.ddf.columns if col not in exclude_cols]
        
        logger.info(f"データ読み込み完了。特徴量数: {len(self.feature_cols)}, パーティション数: {self.ddf.npartitions}")
        
        # パージ付きCVのためにPandas DataFrameとして保持（メモリに載せる）
        logger.info("警告: パージ付きCVのため、データをPandas DataFrameに変換中...")
        logger.info("これには時間がかかり、メモリを多く使用します。")
        
        self.df_pandas = self.ddf.compute()  # type: ignore[assignment]
        logger.info(f"Pandas DataFrame変換完了。形状: {self.df_pandas.shape}")  # type: ignore[union-attr]
    
    def _objective_m1(self, trial: Trial) -> float:
        """M1（プライマリーモデル）の目的関数"""
        # ハイパーパラメータの提案
        params = {
            'n_estimators': trial.suggest_int('m1_n_estimators', 100, 1000),
            'max_depth': trial.suggest_int('m1_max_depth', 3, 12),
            'learning_rate': trial.suggest_float('m1_learning_rate', 0.01, 0.3),
            'num_leaves': trial.suggest_int('m1_num_leaves', 31, 255),
            'min_child_samples': trial.suggest_int('m1_min_child_samples', 20, 500),
            'subsample': trial.suggest_float('m1_subsample', 0.6, 1.0),
            'colsample_bytree': trial.suggest_float('m1_colsample_bytree', 0.6, 1.0),
            'random_state': 42,
            'verbosity': -1,
            'objective': 'binary'
        }
        
        # パージ付きCVでスコア計算
        cv_splitter = PurgedKFold(n_splits=self.n_splits, embargo_pct=self.embargo_pct)
        recall_scores = []
        
        if self.df_pandas is None:
            raise ValueError("データが読み込まれていません")
        
        X = self.df_pandas[self.feature_cols]
        y_binary = (self.df_pandas['label'] == 1).astype(int)  # 利食い vs それ以外
        sample_weights = self.df_pandas['sample_weight'].values
        
        for train_idx, test_idx in cv_splitter.split(self.df_pandas):
            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train, y_test = y_binary.iloc[train_idx], y_binary.iloc[test_idx]
            w_train = sample_weights[train_idx]
            
            # Dask DataFrameに変換
            X_train_dask = dd.from_pandas(X_train, npartitions=4)
            y_train_dask = dd.from_pandas(y_train, npartitions=4)
            X_test_dask = dd.from_pandas(X_test, npartitions=4)
            y_test_dask = dd.from_pandas(y_test, npartitions=4)
            
            # モデル訓練
            model = dlgb.LGBMClassifier(**params)
            model.fit(X_train_dask, y_train_dask, sample_weight=w_train)
            
            # 予測
            y_pred = model.predict(X_test_dask)
            y_test_computed, y_pred_computed = dask.compute(y_test_dask, y_pred)  # type: ignore
            
            # リコール計算（利食いクラス）
            recall = recall_score(y_test_computed, y_pred_computed, pos_label=1, zero_division=0)
            recall_scores.append(recall)
        
        return float(np.mean(recall_scores))
    
    def _train_m1(self, best_params: Dict[str, Any]) -> Any:
        """M1を全データで訓練"""
        logger.info("M1（プライマリーモデル）を全データで訓練中...")
        
        if self.ddf is None:
            raise ValueError("データが読み込まれていません")
        
        X_full = self.ddf[self.feature_cols]
        y_binary = (self.ddf['label'] == 1).astype(int)  # type: ignore[operator]
        sample_weights_series = self.ddf['sample_weight']  # type: ignore[index]
        sample_weights_computed: np.ndarray = sample_weights_series.compute().values  # type: ignore[union-attr]
        
        model_m1 = dlgb.LGBMClassifier(**best_params)
        model_m1.fit(X_full, y_binary, sample_weight=sample_weights_computed)
        
        logger.info("M1訓練完了")
        return model_m1
    
    def _generate_meta_labels(self, model_m1: Any) -> pd.DataFrame:
        """M1の予測からメタラベルを生成"""
        logger.info("メタラベル生成中...")
        
        if self.df_pandas is None:
            raise ValueError("データが読み込まれていません")
        
        X = self.df_pandas[self.feature_cols]
        y_true = self.df_pandas['label'].values
        
        # M1の予測確率
        X_dask = dd.from_pandas(X, npartitions=4)
        y_pred_proba = model_m1.predict_proba(X_dask)
        y_pred_proba_computed: np.ndarray = dask.compute(y_pred_proba)[0]  # type: ignore
        
        # M1がポジティブ（利食い）と予測したインデックス
        m1_positive_mask = y_pred_proba_computed[:, 1] > 0.5
        
        # メタラベル: M1予測正解=1, 不正解=0
        meta_labels = np.zeros(len(y_true), dtype=int)
        meta_labels[m1_positive_mask] = (y_true[m1_positive_mask] == 1).astype(int)
        
        # M1がポジティブと予測したサンプルのみ抽出
        df_meta = self.df_pandas[m1_positive_mask].copy()
        df_meta['meta_label'] = meta_labels[m1_positive_mask]
        df_meta['p_m1'] = y_pred_proba_computed[m1_positive_mask, 1]
        
        logger.info(f"メタラベル生成完了。M1ポジティブサンプル数: {len(df_meta):,}")
        logger.info(f"メタラベル分布 - 成功: {np.sum(df_meta['meta_label'] == 1):,}, 失敗: {np.sum(df_meta['meta_label'] == 0):,}")
        
        return df_meta
    
    def _objective_m2(self, trial: Trial, df_meta: pd.DataFrame) -> float:
        """M2（セカンダリーモデル）の目的関数"""
        # ハイパーパラメータの提案
        params = {
            'n_estimators': trial.suggest_int('m2_n_estimators', 100, 1000),
            'max_depth': trial.suggest_int('m2_max_depth', 3, 12),
            'learning_rate': trial.suggest_float('m2_learning_rate', 0.01, 0.3),
            'num_leaves': trial.suggest_int('m2_num_leaves', 31, 255),
            'min_child_samples': trial.suggest_int('m2_min_child_samples', 20, 500),
            'subsample': trial.suggest_float('m2_subsample', 0.6, 1.0),
            'colsample_bytree': trial.suggest_float('m2_colsample_bytree', 0.6, 1.0),
            'random_state': 42,
            'verbosity': -1,
            'objective': 'binary'
        }
        
        # M2の特徴量: 基本特徴量 + p_m1
        feature_cols_m2 = self.feature_cols + ['p_m1']
        
        # パージ付きCV
        cv_splitter = PurgedKFold(n_splits=self.n_splits, embargo_pct=self.embargo_pct)
        f1_scores = []
        
        X = df_meta[feature_cols_m2]
        y = df_meta['meta_label']
        sample_weights = df_meta['sample_weight'].values
        
        for train_idx, test_idx in cv_splitter.split(df_meta):
            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
            w_train = sample_weights[train_idx]
            
            # Dask DataFrameに変換
            X_train_dask = dd.from_pandas(X_train, npartitions=4)
            y_train_dask = dd.from_pandas(y_train, npartitions=4)
            X_test_dask = dd.from_pandas(X_test, npartitions=4)
            y_test_dask = dd.from_pandas(y_test, npartitions=4)
            
            # モデル訓練
            model = dlgb.LGBMClassifier(**params)
            model.fit(X_train_dask, y_train_dask, sample_weight=w_train)
            
            # 予測
            y_pred = model.predict(X_test_dask)
            y_test_computed, y_pred_computed = dask.compute(y_test_dask, y_pred)  # type: ignore
            
            # F1スコア計算
            f1 = f1_score(y_test_computed, y_pred_computed, zero_division=0)
            f1_scores.append(f1)
        
        return float(np.mean(f1_scores))
    
    def _train_m2(self, best_params: Dict[str, Any], df_meta: pd.DataFrame) -> Any:
        """M2を全データで訓練"""
        logger.info("M2（セカンダリーモデル）を全データで訓練中...")
        
        feature_cols_m2 = self.feature_cols + ['p_m1']
        X_full = df_meta[feature_cols_m2]
        y_full = df_meta['meta_label']
        sample_weights = df_meta['sample_weight'].values
        
        X_full_dask = dd.from_pandas(X_full, npartitions=4)
        y_full_dask = dd.from_pandas(y_full, npartitions=4)
        
        model_m2 = dlgb.LGBMClassifier(**best_params)
        model_m2.fit(X_full_dask, y_full_dask, sample_weight=sample_weights)
        
        logger.info("M2訓練完了")
        return model_m2
    
    def _calibrate_model(self, model: Any, X: pd.DataFrame, y: pd.Series, 
                        method: str = 'isotonic') -> CalibratedClassifierCV:
        """モデルの確率較正"""
        logger.info(f"確率較正中（method={method}）...")
        
        # 注: CalibratedClassifierCVはsklearnモデル用なので、
        # Dask-LightGBMモデルを直接較正することはできない
        # 代わりに、予測確率を計算して事後的に較正する
        
        # ここでは簡易実装として、モデルをそのまま返す
        # 本格実装では、scikit-learn互換のラッパーを作成するか、
        # 別途較正用のデータセットでIsotonicRegressionを訓練する
        
        logger.warning("Dask-LightGBM用の確率較正は未実装。代替手法の検討が必要です。")
        return model  # type: ignore
    
    def optimize_and_train(self) -> Tuple[Any, Any, Dict[str, Any], Dict[str, Any]]:
        """Optunaで最適化し、最終モデルを訓練"""
        self._load_data()
        
        # 出力ディレクトリの作成
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # === M1の最適化 ===
        logger.info("=" * 50)
        logger.info("M1（プライマリーモデル）の最適化開始")
        logger.info("=" * 50)
        
        study_m1 = optuna.create_study(direction='maximize', study_name='m1_optimization')
        study_m1.optimize(self._objective_m1, n_trials=self.n_trials)
        
        best_params_m1 = study_m1.best_params
        logger.info(f"M1最適パラメータ: {best_params_m1}")
        logger.info(f"最高リコール: {study_m1.best_value:.4f}")
        
        # M1の訓練
        model_m1 = self._train_m1(best_params_m1)
        
        # === メタラベルの生成 ===
        df_meta = self._generate_meta_labels(model_m1)
        
        # === M2の最適化 ===
        logger.info("=" * 50)
        logger.info("M2（セカンダリーモデル）の最適化開始")
        logger.info("=" * 50)
        
        study_m2 = optuna.create_study(direction='maximize', study_name='m2_optimization')
        study_m2.optimize(lambda trial: self._objective_m2(trial, df_meta), n_trials=self.n_trials)
        
        best_params_m2 = study_m2.best_params
        logger.info(f"M2最適パラメータ: {best_params_m2}")
        logger.info(f"最高F1スコア: {study_m2.best_value:.4f}")
        
        # M2の訓練
        model_m2 = self._train_m2(best_params_m2, df_meta)
        
        # === モデルの保存 ===
        logger.info("モデルを保存中...")
        
        # LightGBMテキスト形式
        model_m1.booster_.save_model(str(self.output_dir / 'm1_model.txt'))
        model_m2.booster_.save_model(str(self.output_dir / 'm2_model.txt'))
        
        # Pickle形式
        joblib.dump(model_m1, self.output_dir / 'm1_model.pkl')
        joblib.dump(model_m2, self.output_dir / 'm2_model.pkl')
        
        # Optunaスタディの保存
        joblib.dump(study_m1, self.output_dir / 'optuna_study_m1.pkl')
        joblib.dump(study_m2, self.output_dir / 'optuna_study_m2.pkl')
        
        # メタラベルデータの保存（評価用）
        df_meta.to_parquet(self.output_dir / 'meta_labels.parquet')
        
        logger.info(f"モデル保存完了: {self.output_dir}")
        
        return model_m1, model_m2, best_params_m1, best_params_m2
    
    def evaluate_and_save_report(self, model_m1: Any, model_m2: Any, 
                                best_params_m1: Dict[str, Any], 
                                best_params_m2: Dict[str, Any]) -> None:
        """モデルを評価し、性能レポートを保存"""
        logger.info("=" * 50)
        logger.info("最終モデルの性能評価を実行中")
        logger.info("=" * 50)
        
        if self.df_pandas is None:
            raise ValueError("データが読み込まれていません")
        
        # メタラベルデータを読み込み
        df_meta = pd.read_parquet(self.output_dir / 'meta_labels.parquet')
        
        # パージ付きCVで評価
        cv_splitter = PurgedKFold(n_splits=self.n_splits, embargo_pct=self.embargo_pct)
        
        m1_metrics: List[Dict[str, Any]] = []
        m2_metrics: List[Dict[str, Any]] = []
        
        # M1評価
        X_m1 = self.df_pandas[self.feature_cols]
        y_m1 = (self.df_pandas['label'] == 1).astype(int)
        
        logger.info("M1を評価中...")
        for i, (train_idx, test_idx) in enumerate(cv_splitter.split(self.df_pandas)):
            X_test = X_m1.iloc[test_idx]
            y_test = y_m1.iloc[test_idx]
            
            X_test_dask = dd.from_pandas(X_test, npartitions=4)
            y_test_dask = dd.from_pandas(y_test, npartitions=4)
            
            y_pred_proba = model_m1.predict_proba(X_test_dask)
            y_pred = model_m1.predict(X_test_dask)
            
            y_test_computed, y_pred_proba_computed, y_pred_computed = dask.compute(  # type: ignore
                y_test_dask, y_pred_proba, y_pred
            )
            
            auc = roc_auc_score(y_test_computed, y_pred_proba_computed[:, 1])
            accuracy = accuracy_score(y_test_computed, y_pred_computed)
            precision = precision_score(y_test_computed, y_pred_computed, zero_division=0)
            recall = recall_score(y_test_computed, y_pred_computed, zero_division=0)
            f1 = f1_score(y_test_computed, y_pred_computed, zero_division=0)
            
            m1_metrics.append({
                'split': i + 1,
                'auc': float(auc),
                'accuracy': float(accuracy),
                'precision': float(precision),
                'recall': float(recall),
                'f1': float(f1)
            })
        
        # M2評価
        feature_cols_m2 = self.feature_cols + ['p_m1']
        X_m2 = df_meta[feature_cols_m2]
        y_m2 = df_meta['meta_label']
        
        logger.info("M2を評価中...")
        for i, (train_idx, test_idx) in enumerate(cv_splitter.split(df_meta)):
            X_test = X_m2.iloc[test_idx]
            y_test = y_m2.iloc[test_idx]
            
            X_test_dask = dd.from_pandas(X_test, npartitions=4)
            y_test_dask = dd.from_pandas(y_test, npartitions=4)
            
            y_pred_proba = model_m2.predict_proba(X_test_dask)
            y_pred = model_m2.predict(X_test_dask)
            
            y_test_computed, y_pred_proba_computed, y_pred_computed = dask.compute(  # type: ignore
                y_test_dask, y_pred_proba, y_pred
            )
            
            auc = roc_auc_score(y_test_computed, y_pred_proba_computed[:, 1])
            accuracy = accuracy_score(y_test_computed, y_pred_computed)
            precision = precision_score(y_test_computed, y_pred_computed, zero_division=0)
            recall = recall_score(y_test_computed, y_pred_computed, zero_division=0)
            f1 = f1_score(y_test_computed, y_pred_computed, zero_division=0)
            
            m2_metrics.append({
                'split': i + 1,
                'auc': float(auc),
                'accuracy': float(accuracy),
                'precision': float(precision),
                'recall': float(recall),
                'f1': float(f1)
            })
        
        # 平均メトリクスの計算
        avg_m1_metrics = {
            'auc': np.mean([m['auc'] for m in m1_metrics]),
            'accuracy': np.mean([m['accuracy'] for m in m1_metrics]),
            'precision': np.mean([m['precision'] for m in m1_metrics]),
            'recall': np.mean([m['recall'] for m in m1_metrics]),
            'f1': np.mean([m['f1'] for m in m1_metrics])
        }
        
        avg_m2_metrics = {
            'auc': np.mean([m['auc'] for m in m2_metrics]),
            'accuracy': np.mean([m['accuracy'] for m in m2_metrics]),
            'precision': np.mean([m['precision'] for m in m2_metrics]),
            'recall': np.mean([m['recall'] for m in m2_metrics]),
            'f1': np.mean([m['f1'] for m in m2_metrics])
        }
        
        # 特徴量重要度
        feature_importance_m1 = dict(zip(self.feature_cols, model_m1.feature_importances_))
        feature_importance_m2 = dict(zip(feature_cols_m2, model_m2.feature_importances_))
        
        top_features_m1 = dict(sorted(feature_importance_m1.items(), key=lambda x: x[1], reverse=True)[:20])
        top_features_m2 = dict(sorted(feature_importance_m2.items(), key=lambda x: x[1], reverse=True)[:20])
        
        # レポート作成
        report = {
            'm1_primary': {
                'description': '方向性予測モデル（利食い vs それ以外）',
                'best_params': best_params_m1,
                'cv_metrics': m1_metrics,
                'avg_metrics': {k: float(v) for k, v in avg_m1_metrics.items()},
                'top_features': {k: float(v) for k, v in top_features_m1.items()}
            },
            'm2_secondary': {
                'description': '確信度評価モデル（M1シグナルの成功確率）',
                'best_params': best_params_m2,
                'cv_metrics': m2_metrics,
                'avg_metrics': {k: float(v) for k, v in avg_m2_metrics.items()},
                'top_features': {k: float(v) for k, v in top_features_m2.items()}
            },
            'training_info': {
                'n_features': len(self.feature_cols),
                'n_splits': self.n_splits,
                'n_trials': self.n_trials,
                'embargo_pct': self.embargo_pct,
                'total_samples': len(self.df_pandas),
                'meta_samples': len(df_meta),
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
        logger.info(f"M1（方向性） - 平均AUC: {avg_m1_metrics['auc']:.4f}, リコール: {avg_m1_metrics['recall']:.4f}, F1: {avg_m1_metrics['f1']:.4f}")
        logger.info(f"M2（確信度） - 平均AUC: {avg_m2_metrics['auc']:.4f}, プレシジョン: {avg_m2_metrics['precision']:.4f}, F1: {avg_m2_metrics['f1']:.4f}")
        logger.info(f"性能レポート保存: {report_path}")
        logger.info("=" * 50)
    
    def run(self) -> None:
        """パイプライン全体を実行"""
        model_m1, model_m2, best_params_m1, best_params_m2 = self.optimize_and_train()
        self.evaluate_and_save_report(model_m1, model_m2, best_params_m1, best_params_m2)


if __name__ == '__main__':
    # Daskの設定
    dask.config.set({'dataframe.query-planning': True})
    
    # Daskクライアントの起動
    from dask.distributed import Client, LocalCluster  # type: ignore
    
    with LocalCluster(n_workers=4, threads_per_worker=2, memory_limit='8GB') as cluster, \
         Client(cluster) as client:
        
        logger.info(f"Daskクライアントを起動: {client.dashboard_link}")
        
        trainer = MetaLabelingTrainer(
            input_path='data/temp_chunks/training_data/weighted_dataset_partitioned',
            output_dir='data/models',
            n_splits=5,
            n_trials=100,
            embargo_pct=0.01
        )
        
        trainer.run()