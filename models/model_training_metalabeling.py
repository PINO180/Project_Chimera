import config
import dask
import dask.dataframe as dd
from dask.dataframe.core import DataFrame, Series
import dask_lightgbm.core as dlgb
import numpy as np
import pandas as pd
import joblib
import optuna  # type: ignore
from optuna.trial import Trial  # type: ignore
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, f1_score
from sklearn.isotonic import IsotonicRegression
from pathlib import Path
import json
import logging
from typing import Dict, Any, List, Tuple, Optional
import warnings

warnings.filterwarnings('ignore', category=FutureWarning)

# ロギング設定
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class TimeSeriesSplitInfo:
    """タイムスタンプベースの分割情報"""
    
    @staticmethod
    def create_splits(ddf: DataFrame, n_splits: int, 
                     embargo_pct: float) -> List[Dict[str, Any]]:
        """
        タイムスタンプ範囲ベースの分割情報を作成
        
        Args:
            ddf: Dask DataFrame
            n_splits: 分割数
            embargo_pct: エンバーゴ期間の割合
        
        Returns:
            分割情報のリスト
        """
        logger.info(f"TimeSeriesSplit情報を作成中（{n_splits}分割）...")
        
        # 最小・最大タイムスタンプを取得
        min_ts_delayed = ddf['t0'].min()
        max_ts_delayed = ddf['t0'].max()
        min_ts, max_ts = dask.compute(min_ts_delayed, max_ts_delayed)  # type: ignore
        
        total_duration = max_ts - min_ts
        test_duration = total_duration / n_splits
        embargo_duration = total_duration * embargo_pct
        
        splits = []
        
        for i in range(n_splits):
            # テスト期間
            test_start = min_ts + test_duration * i
            test_end = min_ts + test_duration * (i + 1)
            
            # 訓練期間（テスト開始前まで、ただしパージとエンバーゴを考慮）
            train_start = min_ts
            train_end = test_start
            
            # エンバーゴ：テスト終了直後も除外
            embargo_end = test_end + embargo_duration
            
            splits.append({
                'split_id': i + 1,
                'train_start': train_start,
                'train_end': train_end,
                'test_start': test_start,
                'test_end': test_end,
                'purge_start': train_end,  # パージ開始（訓練終了時点）
                'purge_end': test_end,      # パージ終了（テスト終了時点）
                'embargo_end': embargo_end
            })
        
        logger.info(f"{n_splits}個の分割情報を作成完了")
        return splits
    
    @staticmethod
    def filter_by_split(ddf: DataFrame, split_info: Dict[str, Any], 
                       mode: str) -> DataFrame:
        """
        分割情報に基づいてデータをフィルタリング
        
        Args:
            ddf: Dask DataFrame
            split_info: 分割情報
            mode: 'train' または 'test'
        
        Returns:
            フィルタリングされたDataFrame
        """
        if mode == 'train':
            # 訓練データ：train_start <= t0 < train_end
            # かつ、テスト期間と重複しない（パージ）
            filtered = ddf[
                (ddf['t0'] >= split_info['train_start']) &
                (ddf['t0'] < split_info['train_end']) &
                (ddf['t1'] <= split_info['test_start'])  # パージ: t1がテスト開始前
            ]
        elif mode == 'test':
            # テストデータ：test_start <= t0 < test_end
            filtered = ddf[
                (ddf['t0'] >= split_info['test_start']) &
                (ddf['t0'] < split_info['test_end'])
            ]
        else:
            raise ValueError(f"Invalid mode: {mode}")
        
        return filtered  # type: ignore[return-value]


def generate_m1_performance_features_partition(partition: pd.DataFrame, 
                                               window: int = 50) -> pd.DataFrame:
    """
    M1性能特徴量を生成（パーティション処理）
    
    Args:
        partition: パーティションデータ（p_m1, label含む）
        window: ローリングウィンドウサイズ
    
    Returns:
        M1性能特徴量を追加したDataFrame
    """
    if partition.empty or len(partition) < window:
        result = partition.copy()
        result['m1_rolling_precision'] = 0.0
        result['m1_rolling_recall'] = 0.0
        result['m1_rolling_f1'] = 0.0
        return result
    
    result = partition.copy()
    
    # M1の予測（p_m1 > 0.5 で利食い予測）
    m1_pred = (result['p_m1'] > 0.5).astype(int)
    m1_true = (result['label'] == 1).astype(int)
    
    # ローリング統計
    m1_correct = (m1_pred == m1_true).astype(float)
    m1_positive = m1_pred.astype(float)
    m1_true_positive = (m1_pred & m1_true).astype(float)
    
    # Precision: TP / (TP + FP)
    rolling_tp = m1_true_positive.rolling(window, min_periods=1).sum()
    rolling_positive = m1_positive.rolling(window, min_periods=1).sum()
    result['m1_rolling_precision'] = np.where(
        rolling_positive > 0,
        rolling_tp / rolling_positive,
        0.0
    )
    
    # Recall: TP / (TP + FN)
    rolling_actual_positive = m1_true.astype(float).rolling(window, min_periods=1).sum()
    result['m1_rolling_recall'] = np.where(
        rolling_actual_positive > 0,
        rolling_tp / rolling_actual_positive,
        0.0
    )
    
    # F1スコア: 2 * (Precision * Recall) / (Precision + Recall)
    precision = result['m1_rolling_precision']
    recall = result['m1_rolling_recall']
    result['m1_rolling_f1'] = np.where(
        (precision + recall) > 0,
        2 * (precision * recall) / (precision + recall),
        0.0
    )
    
    return result


class MetaLabelingTrainer:
    """
    【完全Out-of-Core版】メタラベリング・トレーナー（統合設計図V完全準拠）
    M1（方向性）とM2（確信度）の二段階モデルを訓練
    """
    
    def __init__(self,
                 input_path: str,
                 output_dir: str,
                 n_splits: int = 5,
                 n_trials: int = 50,
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
        self.splits: List[Dict[str, Any]] = []
    
    def _load_data(self) -> None:
        """データセットを読み込む（メモリに載せない）"""
        logger.info(f"データセット '{self.input_path}' を読み込み中...")
        
        self.ddf = dd.read_parquet(  # type: ignore
            self.input_path,
            engine='pyarrow'
        )
        
        # 必須カラムの確認
        required_cols = ['label', 't0', 't1', 'sample_weight', 'close', 'timestamp']
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
        logger.info("✅ Out-of-Core処理：データはメモリに載せていません")
        
        # 分割情報を作成
        self.splits = TimeSeriesSplitInfo.create_splits(
            self.ddf, self.n_splits, self.embargo_pct
        )
    
    def _objective_m1(self, trial: Trial) -> float:
        """M1（プライマリーモデル）の目的関数"""
        params = {
            'n_estimators': trial.suggest_int('m1_n_estimators', 100, 1000),
            'max_depth': trial.suggest_int('m1_max_depth', 3, 12),
            'learning_rate': trial.suggest_float('m1_learning_rate', 0.01, 0.3, log=True),
            'num_leaves': trial.suggest_int('m1_num_leaves', 31, 255),
            'min_child_samples': trial.suggest_int('m1_min_child_samples', 20, 500),
            'subsample': trial.suggest_float('m1_subsample', 0.6, 1.0),
            'colsample_bytree': trial.suggest_float('m1_colsample_bytree', 0.6, 1.0),
            'random_state': 42,
            'verbosity': -1,
            'objective': 'binary'
        }
        
        recall_scores = []
        
        for split_info in self.splits:
            train_ddf = TimeSeriesSplitInfo.filter_by_split(self.ddf, split_info, 'train')  # type: ignore
            test_ddf = TimeSeriesSplitInfo.filter_by_split(self.ddf, split_info, 'test')  # type: ignore
            
            X_train = train_ddf[self.feature_cols]
            y_train = (train_ddf['label'] == 1).astype(int)  # type: ignore[operator]
            X_test = test_ddf[self.feature_cols]
            y_test = (test_ddf['label'] == 1).astype(int)  # type: ignore[operator]
            
            # サンプルウェイト取得
            w_train_series: Series = train_ddf['sample_weight']  # type: ignore[assignment]
            w_train: np.ndarray = w_train_series.compute().values  # type: ignore[union-attr]
            
            # モデル訓練
            model = dlgb.LGBMClassifier(**params)
            model.fit(X_train, y_train, sample_weight=w_train)
            
            # 予測
            y_pred = model.predict(X_test)
            y_test_computed, y_pred_computed = dask.compute(y_test, y_pred)  # type: ignore
            
            # リコール計算
            recall = recall_score(y_test_computed, y_pred_computed, pos_label=1, zero_division=0)
            recall_scores.append(recall)
        
        return float(np.mean(recall_scores))
    
    def _train_m1_final(self, best_params: Dict[str, Any]) -> Any:
        """M1を全データで訓練"""
        logger.info("M1（プライマリーモデル）を全データで訓練中...")
        
        if self.ddf is None:
            raise ValueError("データが読み込まれていません")
        
        X_full = self.ddf[self.feature_cols]
        y_full = (self.ddf['label'] == 1).astype(int)  # type: ignore[operator]
        
        # サンプルウェイト取得
        w_full_series: Series = self.ddf['sample_weight']  # type: ignore[assignment]
        w_full: np.ndarray = w_full_series.compute().values  # type: ignore[union-attr]
        
        model_m1 = dlgb.LGBMClassifier(**best_params)
        model_m1.fit(X_full, y_full, sample_weight=w_full)
        
        logger.info("M1訓練完了")
        return model_m1
    
    def _generate_meta_labels_dask(self, model_m1: Any) -> DataFrame:
        """M1の予測からメタラベルを生成（Dask版・型安全）"""
        logger.info("メタラベル生成中（Out-of-Core）...")
        
        if self.ddf is None:
            raise ValueError("データが読み込まれていません")
        
        X = self.ddf[self.feature_cols]
        
        # M1の予測確率を計算
        y_pred_proba_dask = model_m1.predict_proba(X)
        
        # Dask Arrayから正の確率のみを抽出してSeriesに変換
        p_m1_positive: Series = y_pred_proba_dask[:, 1]  # type: ignore[assignment]
        
        # 新しいカラムとして追加（型安全）
        ddf_with_proba = self.ddf.assign(p_m1=p_m1_positive)  # type: ignore[attr-defined]
        ddf_with_proba = ddf_with_proba.assign(  # type: ignore[attr-defined]
            m1_positive=ddf_with_proba['p_m1'] > 0.5  # type: ignore[index]
        )
        ddf_with_proba = ddf_with_proba.assign(  # type: ignore[attr-defined]
            meta_label=(
                ddf_with_proba['m1_positive'] & (ddf_with_proba['label'] == 1)  # type: ignore[operator]
            ).astype(int)
        )
        
        # M1がポジティブと予測したサンプルのみフィルタリング
        ddf_meta_filtered: DataFrame = ddf_with_proba[ddf_with_proba['m1_positive']]  # type: ignore[assignment, index]
        
        # サンプル数を計算
        n_meta = len(ddf_meta_filtered)  # type: ignore[arg-type]
        n_meta_computed: int = int(dask.compute(n_meta)[0])  # type: ignore
        
        # メタラベル分布を計算
        meta_label_sum = ddf_meta_filtered['meta_label'].sum()  # type: ignore[union-attr]
        n_success: int = int(dask.compute(meta_label_sum)[0])  # type: ignore
        n_failure = n_meta_computed - n_success
        
        logger.info(f"メタラベル生成完了。M1ポジティブサンプル数: {n_meta_computed:,}")
        logger.info(f"メタラベル分布 - 成功: {n_success:,}, 失敗: {n_failure:,}")
        
        return ddf_meta_filtered
    
    def _add_m2_features(self, ddf_meta: DataFrame) -> DataFrame:
        """
        M2用の追加特徴量を生成（設計図V 3.3節準拠）
        
        追加特徴量:
        - M1性能特徴量（ローリングプレシジョン、リコール、F1）
        - 市場レジーム特徴量（ATR、ボラティリティ、トレンド強度）
        
        注意: ローリング統計はmap_overlapで境界問題を解決
        """
        logger.info("M2用追加特徴量を生成中...")
        
        # メタデータ定義
        meta_with_m2_features = ddf_meta._meta.copy()  # type: ignore[union-attr]
        meta_with_m2_features['m1_rolling_precision'] = 0.0
        meta_with_m2_features['m1_rolling_recall'] = 0.0
        meta_with_m2_features['m1_rolling_f1'] = 0.0
        
        # M1性能特徴量を追加（map_overlapでパーティション境界問題を解決）
        window = 50
        ddf_with_m1_perf: DataFrame = ddf_meta.map_overlap(  # type: ignore[assignment]
            generate_m1_performance_features_partition,
            before=window,  # 前のパーティションからwindow分のデータを取得
            after=0,
            window=window,
            meta=meta_with_m2_features
        )
        
        # 市場レジーム特徴量の確認（型安全な方法）
        available_cols = list(ddf_with_m1_perf.columns)  # type: ignore[attr-defined]
        if 'ATR' not in available_cols:
            logger.warning("ATRカラムが見つかりません。市場レジーム特徴量は追加されません。")
        
        logger.info("M2用追加特徴量生成完了")
        return ddf_with_m1_perf
    
    def _objective_m2(self, trial: Trial, ddf_meta: DataFrame, 
                     feature_cols_m2: List[str]) -> float:
        """M2（セカンダリーモデル）の目的関数"""
        params = {
            'n_estimators': trial.suggest_int('m2_n_estimators', 100, 1000),
            'max_depth': trial.suggest_int('m2_max_depth', 3, 12),
            'learning_rate': trial.suggest_float('m2_learning_rate', 0.01, 0.3, log=True),
            'num_leaves': trial.suggest_int('m2_num_leaves', 31, 255),
            'min_child_samples': trial.suggest_int('m2_min_child_samples', 20, 500),
            'subsample': trial.suggest_float('m2_subsample', 0.6, 1.0),
            'colsample_bytree': trial.suggest_float('m2_colsample_bytree', 0.6, 1.0),
            'random_state': 42,
            'verbosity': -1,
            'objective': 'binary'
        }
        
        f1_scores = []
        
        for split_info in self.splits:
            train_ddf = TimeSeriesSplitInfo.filter_by_split(ddf_meta, split_info, 'train')
            test_ddf = TimeSeriesSplitInfo.filter_by_split(ddf_meta, split_info, 'test')
            
            X_train = train_ddf[feature_cols_m2]
            y_train = train_ddf['meta_label']  # type: ignore[index]
            X_test = test_ddf[feature_cols_m2]
            y_test = test_ddf['meta_label']  # type: ignore[index]
            
            # サンプルウェイト取得
            w_train_series: Series = train_ddf['sample_weight']  # type: ignore[assignment]
            w_train: np.ndarray = w_train_series.compute().values  # type: ignore[union-attr]
            
            # モデル訓練
            model = dlgb.LGBMClassifier(**params)
            model.fit(X_train, y_train, sample_weight=w_train)
            
            # 予測
            y_pred = model.predict(X_test)
            y_test_computed, y_pred_computed = dask.compute(y_test, y_pred)  # type: ignore
            
            # F1スコア計算
            f1 = f1_score(y_test_computed, y_pred_computed, zero_division=0)
            f1_scores.append(f1)
        
        return float(np.mean(f1_scores))
    
    def _train_m2_final(self, best_params: Dict[str, Any], 
                       ddf_meta: DataFrame, feature_cols_m2: List[str]) -> Any:
        """M2を全データで訓練"""
        logger.info("M2（セカンダリーモデル）を全データで訓練中...")
        
        X_full = ddf_meta[feature_cols_m2]
        y_full = ddf_meta['meta_label']  # type: ignore[index]
        
        # サンプルウェイト取得
        w_full_series: Series = ddf_meta['sample_weight']  # type: ignore[assignment]
        w_full: np.ndarray = w_full_series.compute().values  # type: ignore[union-attr]
        
        model_m2 = dlgb.LGBMClassifier(**best_params)
        model_m2.fit(X_full, y_full, sample_weight=w_full)
        
        logger.info("M2訓練完了")
        return model_m2
    
    def _calibrate_probabilities(self, model: Any, ddf_val: DataFrame, 
                                 feature_cols: List[str], 
                                 is_m1: bool = True) -> IsotonicRegression:
        """
        確率キャリブレーション（設計図V 4.1節準拠）
        
        Args:
            model: 訓練済みモデル
            ddf_val: 検証データ（Dask DataFrame）
            feature_cols: 特徴量カラム
            is_m1: M1かM2か（ターゲット変数の計算に使用）
        
        Returns:
            較正済みIsotonicRegressionモデル
        """
        logger.info("確率キャリブレーション実行中（Isotonic Regression）...")
        
        X_val = ddf_val[feature_cols]
        
        # ターゲット変数を計算
        if is_m1:
            # M1: label == 1 で利食い
            y_val = (ddf_val['label'] == 1).astype(int)  # type: ignore[operator]
        else:
            # M2: meta_label
            y_val = ddf_val['meta_label']  # type: ignore[index]
        
        # 予測確率を取得
        y_pred_proba = model.predict_proba(X_val)
        
        # compute（較正には全データ必要）
        y_val_computed, y_pred_proba_computed = dask.compute(y_val, y_pred_proba)  # type: ignore
        
        # IsotonicRegressionで較正
        calibrator = IsotonicRegression(out_of_bounds='clip')
        calibrator.fit(y_pred_proba_computed[:, 1], y_val_computed)
        
        logger.info("確率キャリブレーション完了")
        return calibrator
    
    def optimize_and_train(self) -> Tuple[Any, Any, Any, Any, Dict[str, Any], Dict[str, Any]]:
        """Optunaで最適化し、最終モデルを訓練"""
        self._load_data()
        
        # 出力ディレクトリの作成
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # === M1の最適化 ===
        logger.info("=" * 50)
        logger.info("M1（プライマリーモデル）の最適化開始")
        logger.info("=" * 50)
        
        study_m1 = optuna.create_study(direction='maximize', study_name='m1_optimization')
        study_m1.optimize(self._objective_m1, n_trials=self.n_trials, show_progress_bar=True)
        
        best_params_m1 = study_m1.best_params
        logger.info(f"M1最適パラメータ: {best_params_m1}")
        logger.info(f"最高リコール: {study_m1.best_value:.4f}")
        
        # M1の訓練
        model_m1 = self._train_m1_final(best_params_m1)
        
        # M1の確率キャリブレーション（最後の分割を検証データとして使用）
        last_split = self.splits[-1]
        val_ddf = TimeSeriesSplitInfo.filter_by_split(self.ddf, last_split, 'test')  # type: ignore
        calibrator_m1 = self._calibrate_probabilities(
            model_m1, val_ddf, self.feature_cols, is_m1=True
        )
        
        # === メタラベルの生成 ===
        ddf_meta = self._generate_meta_labels_dask(model_m1)
        
        # === M2用特徴量の追加 ===
        ddf_meta = self._add_m2_features(ddf_meta)
        
        # M2の特徴量リスト（動的に生成）
        feature_cols_m2 = (self.feature_cols + 
                          ['p_m1', 'm1_rolling_precision', 
                           'm1_rolling_recall', 'm1_rolling_f1'])
        
        # ATRなどの市場レジーム特徴量があれば追加（型安全な確認）
        available_cols = list(ddf_meta.columns)  # type: ignore[attr-defined]
        if 'ATR' in available_cols:
            feature_cols_m2.append('ATR')
            logger.info("ATRを市場レジーム特徴量として追加")
        
        logger.info(f"M2特徴量数: {len(feature_cols_m2)}（基本 + M1出力 + M1性能 + 市場レジーム）")
        
        # === M2の最適化 ===
        logger.info("=" * 50)
        logger.info("M2（セカンダリーモデル）の最適化開始")
        logger.info("=" * 50)
        
        study_m2 = optuna.create_study(direction='maximize', study_name='m2_optimization')
        study_m2.optimize(
            lambda trial: self._objective_m2(trial, ddf_meta, feature_cols_m2),
            n_trials=self.n_trials,
            show_progress_bar=True
        )
        
        best_params_m2 = study_m2.best_params
        logger.info(f"M2最適パラメータ: {best_params_m2}")
        logger.info(f"最高F1スコア: {study_m2.best_value:.4f}")
        
        # M2の訓練
        model_m2 = self._train_m2_final(best_params_m2, ddf_meta, feature_cols_m2)
        
        # M2の確率キャリブレーション
        val_meta_ddf = TimeSeriesSplitInfo.filter_by_split(ddf_meta, last_split, 'test')
        calibrator_m2 = self._calibrate_probabilities(
            model_m2, val_meta_ddf, feature_cols_m2, is_m1=False
        )
        
        # === モデルの保存 ===
        logger.info("モデルを保存中...")
        
        # LightGBMテキスト形式
        model_m1.booster_.save_model(str(self.output_dir / 'm1_model.txt'))
        model_m2.booster_.save_model(str(self.output_dir / 'm2_model.txt'))
        
        # Pickle形式
        joblib.dump(model_m1, self.output_dir / 'm1_model.pkl')
        joblib.dump(model_m2, self.output_dir / 'm2_model.pkl')
        joblib.dump(calibrator_m1, self.output_dir / 'm1_calibrator.pkl')
        joblib.dump(calibrator_m2, self.output_dir / 'm2_calibrator.pkl')
        
        # Optunaスタディの保存
        joblib.dump(study_m1, self.output_dir / 'optuna_study_m1.pkl')
        joblib.dump(study_m2, self.output_dir / 'optuna_study_m2.pkl')
        
        # M2特徴量リストの保存
        with open(self.output_dir / 'm2_feature_cols.json', 'w') as f:
            json.dump(feature_cols_m2, f, indent=2)
        
        logger.info(f"モデル保存完了: {self.output_dir}")
        
        return model_m1, model_m2, calibrator_m1, calibrator_m2, best_params_m1, best_params_m2
    
    def evaluate_and_save_report(self, model_m1: Any, model_m2: Any,
                                calibrator_m1: IsotonicRegression,
                                calibrator_m2: IsotonicRegression,
                                best_params_m1: Dict[str, Any], 
                                best_params_m2: Dict[str, Any]) -> None:
        """モデルを評価し、性能レポートを保存"""
        logger.info("=" * 50)
        logger.info("最終モデルの性能評価を実行中")
        logger.info("=" * 50)
        
        # M2特徴量リストを読み込み
        with open(self.output_dir / 'm2_feature_cols.json', 'r') as f:
            feature_cols_m2 = json.load(f)
        
        # メタラベルデータを再生成
        ddf_meta = self._generate_meta_labels_dask(model_m1)
        ddf_meta = self._add_m2_features(ddf_meta)
        
        m1_metrics: List[Dict[str, Any]] = []
        m2_metrics: List[Dict[str, Any]] = []
        
        # 各分割で評価
        for split_info in self.splits:
            split_id = split_info['split_id']
            logger.info(f"分割 {split_id}/{self.n_splits} を評価中...")
            
            # M1評価
            test_ddf = TimeSeriesSplitInfo.filter_by_split(self.ddf, split_info, 'test')  # type: ignore
            X_test_m1 = test_ddf[self.feature_cols]
            y_test_m1 = (test_ddf['label'] == 1).astype(int)  # type: ignore[operator]
            
            y_pred_proba_m1 = model_m1.predict_proba(X_test_m1)
            y_pred_m1 = model_m1.predict(X_test_m1)
            
            y_test_m1_computed, y_pred_proba_m1_computed, y_pred_m1_computed = dask.compute(  # type: ignore
                y_test_m1, y_pred_proba_m1, y_pred_m1
            )
            
            # 較正後の確率
            proba_calibrated_m1 = calibrator_m1.predict(y_pred_proba_m1_computed[:, 1])
            
            auc_m1 = roc_auc_score(y_test_m1_computed, proba_calibrated_m1)
            accuracy_m1 = accuracy_score(y_test_m1_computed, y_pred_m1_computed)
            precision_m1 = precision_score(y_test_m1_computed, y_pred_m1_computed, zero_division=0)
            recall_m1 = recall_score(y_test_m1_computed, y_pred_m1_computed, zero_division=0)
            f1_m1 = f1_score(y_test_m1_computed, y_pred_m1_computed, zero_division=0)
            
            m1_metrics.append({
                'split': split_id,
                'auc': float(auc_m1),
                'auc_calibrated': float(auc_m1),  # 較正後
                'accuracy': float(accuracy_m1),
                'precision': float(precision_m1),
                'recall': float(recall_m1),
                'f1': float(f1_m1)
            })
            
            # M2評価
            test_meta_ddf = TimeSeriesSplitInfo.filter_by_split(ddf_meta, split_info, 'test')
            X_test_m2 = test_meta_ddf[feature_cols_m2]
            y_test_m2 = test_meta_ddf['meta_label']  # type: ignore[index]
            
            y_pred_proba_m2 = model_m2.predict_proba(X_test_m2)
            y_pred_m2 = model_m2.predict(X_test_m2)
            
            y_test_m2_computed, y_pred_proba_m2_computed, y_pred_m2_computed = dask.compute(  # type: ignore
                y_test_m2, y_pred_proba_m2, y_pred_m2
            )
            
            # 較正後の確率
            proba_calibrated_m2 = calibrator_m2.predict(y_pred_proba_m2_computed[:, 1])
            
            auc_m2 = roc_auc_score(y_test_m2_computed, proba_calibrated_m2)
            accuracy_m2 = accuracy_score(y_test_m2_computed, y_pred_m2_computed)
            precision_m2 = precision_score(y_test_m2_computed, y_pred_m2_computed, zero_division=0)
            recall_m2 = recall_score(y_test_m2_computed, y_pred_m2_computed, zero_division=0)
            f1_m2 = f1_score(y_test_m2_computed, y_pred_m2_computed, zero_division=0)
            
            m2_metrics.append({
                'split': split_id,
                'auc': float(auc_m2),
                'auc_calibrated': float(auc_m2),
                'accuracy': float(accuracy_m2),
                'precision': float(precision_m2),
                'recall': float(recall_m2),
                'f1': float(f1_m2)
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
                'optimization_goal': 'リコール最大化',
                'best_params': best_params_m1,
                'cv_metrics': m1_metrics,
                'avg_metrics': {k: float(v) for k, v in avg_m1_metrics.items()},
                'top_features': {k: float(v) for k, v in top_features_m1.items()},
                'calibration': 'Isotonic Regression'
            },
            'm2_secondary': {
                'description': '確信度評価モデル（M1シグナルの成功確率）',
                'optimization_goal': 'F1スコア最大化',
                'best_params': best_params_m2,
                'cv_metrics': m2_metrics,
                'avg_metrics': {k: float(v) for k, v in avg_m2_metrics.items()},
                'top_features': {k: float(v) for k, v in top_features_m2.items()},
                'calibration': 'Isotonic Regression',
                'additional_features': ['p_m1', 'm1_rolling_precision', 
                                      'm1_rolling_recall', 'm1_rolling_f1']
            },
            'training_info': {
                'n_features_m1': len(self.feature_cols),
                'n_features_m2': len(feature_cols_m2),
                'n_splits': self.n_splits,
                'n_trials': self.n_trials,
                'embargo_pct': self.embargo_pct,
                'out_of_core': True,
                'timestamp': pd.Timestamp.now().isoformat()
            }
        }
        
        # JSONで保存
        report_path = self.output_dir / 'model_performance_report.json'
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        logger.info("=" * 50)
        logger.info("モデル訓練・評価完了")
        logger.info("=" * 50)
        logger.info(f"M1（方向性） - 平均AUC: {avg_m1_metrics['auc']:.4f}, リコール: {avg_m1_metrics['recall']:.4f}, F1: {avg_m1_metrics['f1']:.4f}")
        logger.info(f"M2（確信度） - 平均AUC: {avg_m2_metrics['auc']:.4f}, プレシジョン: {avg_m2_metrics['precision']:.4f}, F1: {avg_m2_metrics['f1']:.4f}")
        logger.info(f"✅ 確率キャリブレーション適用済み（Isotonic Regression）")
        logger.info(f"✅ M2追加特徴量: M1性能指標 + 市場レジーム")
        logger.info(f"性能レポート保存: {report_path}")
        logger.info("=" * 50)
    
    def run(self) -> None:
        """パイプライン全体を実行"""
        results = self.optimize_and_train()
        model_m1, model_m2, calibrator_m1, calibrator_m2, best_params_m1, best_params_m2 = results
        self.evaluate_and_save_report(
            model_m1, model_m2, calibrator_m1, calibrator_m2,
            best_params_m1, best_params_m2
        )


if __name__ == '__main__':
    # Daskの設定
    dask.config.set({'dataframe.query-planning': True})

    # Daskクライアントの起動
    from dask.distributed import Client, LocalCluster  # type: ignore

    with LocalCluster(n_workers=4, threads_per_worker=2, memory_limit='8GB') as cluster, \
         Client(cluster) as client:

        logger.info(f"Daskクライアントを起動: {client.dashboard_link}")

        trainer = MetaLabelingTrainer(
            input_path=str(config.S6_WEIGHTED_DATASET),
            output_dir=str(config.S7_MODELS),
            n_splits=5,
            n_trials=50,  # 本番は100推奨
            embargo_pct=0.01
        )

        trainer.run()