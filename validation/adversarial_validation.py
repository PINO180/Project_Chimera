"""
adversarial_validation.py
第一防衛線 - 敵対的検証（Adversarial Validationのみ）

統合設計図V準拠：
- 全データ使用
- Dask-LightGBM完全対応
- Pylance厳格型定義準拠
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))

import blueprint as config
import logging
from typing import List, Set, Dict
from pathlib import Path
import json
import time
from datetime import datetime

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

import dask
import dask.dataframe as dd
from dask.dataframe import DataFrame as DaskDataFrame
from dask import config as dask_config
from dask.distributed import Client, LocalCluster
from lightgbm.dask import DaskLGBMClassifier
from dask.base import compute
import numpy as np
import joblib
import psutil
from tqdm import tqdm


class ProcessingConfig:
    """設定管理クラス"""
    def __init__(self):
        self.feature_universe_path = str(config.S2_FEATURES2)
        self.output_dir = config.S3_ARTIFACTS
        self.n_jobs = 4
        self.adversarial_auc_threshold = 0.7
        self.memory_warning_gb = 50
        self.memory_critical_gb = 55


class AdversarialValidator:
    """敵対的検証による時間的ドリフト検出"""
    
    def __init__(self, cfg: ProcessingConfig):
        self.cfg = cfg
        self.ddf: DaskDataFrame | None = None
        self.feature_columns: List[str] = []
        
    def _check_memory(self) -> None:
        """メモリ使用量監視"""
        mem = psutil.virtual_memory()
        used_gb = mem.used / (1024**3)
        
        if used_gb > self.cfg.memory_critical_gb:
            logger.error(f"メモリ使用量が危険レベル: {used_gb:.2f}GB")
            raise MemoryError("メモリ不足により処理を中断します")
        elif used_gb > self.cfg.memory_warning_gb:
            logger.warning(f"メモリ使用量が高レベル: {used_gb:.2f}GB")
    
    def _load_data(self) -> None:
        """
        Dask DataFrameとして特徴量ユニバースを読み込む (v4.1 - タイムフレームサフィックス版)
        """
        feature_universe_path = Path(self.cfg.feature_universe_path)
        logger.info(f"敵対的検証: 特徴量ユニバース '{feature_universe_path}' をDask DataFrameとして読み込み中...")

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
                    if col not in non_feature_columns:
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
    
    def run_adversarial_validation(self) -> tuple[Set[str], Dict[str, float], float, float]:
        """
        敵対的検証を実行
        """
        logger.info("=" * 60)
        logger.info("敵対的検証（Adversarial Validation）を開始")
        logger.info("=" * 60)
        
        start_time = time.time()
        
        if self.ddf is None:
            raise ValueError("Dask DataFrame not loaded.")
        
        self._check_memory()
        
        split_point = self.ddf.npartitions // 2
        train_ddf = self.ddf.partitions[:split_point]
        val_ddf = self.ddf.partitions[split_point:]
        
        # ラベル付与
        logger.info("データにラベルを付与中...")
        train_ddf = train_ddf.assign(adversarial_label=0)
        val_ddf = val_ddf.assign(adversarial_label=1)
        
        # 結合
        logger.info("訓練データと検証データを結合中...")
        combined_ddf = dd.concat([train_ddf, val_ddf], axis=0)
        combined_ddf = combined_ddf.dropna()
        
        self._check_memory()
        
        logger.info("敵対的検証用モデルを訓練中...")
        X = combined_ddf[self.feature_columns]
        y = combined_ddf['adversarial_label']
        
        model = DaskLGBMClassifier(
            n_estimators=50,
            random_state=42,
            verbosity=-1,
            n_jobs=self.cfg.n_jobs
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
        logger.info("特徴量重要度を取得中...")
        importances = model.feature_importances_
        adversarial_scores: Dict[str, float] = dict(zip(self.feature_columns, importances))
        
        # 不安定フラグ付与
        unstable_features: Set[str] = set()
        if auc > self.cfg.adversarial_auc_threshold:
            logger.warning(f"時間的ドリフト検出（AUC={auc:.4f} > {self.cfg.adversarial_auc_threshold}）")
            # 上位20%の特徴量を不安定とマーク
            threshold_importance = np.percentile(list(adversarial_scores.values()), 80)
            unstable_features = {
                feat for feat, score in adversarial_scores.items()
                if score > threshold_importance
            }
            logger.info(f"{len(unstable_features)}個の特徴量に不安定フラグ付与。")
        else:
            logger.info("時間的ドリフトは検出されませんでした。")
        
        execution_time = time.time() - start_time
        
        logger.info("=" * 60)
        logger.info("敵対的検証完了")
        logger.info(f"検証特徴量数: {len(self.feature_columns)}")
        logger.info(f"不安定特徴量数: {len(unstable_features)}")
        logger.info(f"AUCスコア: {auc:.4f}")
        logger.info(f"実行時間: {execution_time:.2f}秒")
        logger.info("=" * 60)
        
        return unstable_features, adversarial_scores, auc, execution_time
    
    def save_results(self, unstable_features: Set[str], adversarial_scores: Dict[str, float], 
                     auc: float, execution_time: float) -> None:
        """結果をJSON形式とjoblib形式で保存"""
        self.cfg.output_dir.mkdir(parents=True, exist_ok=True)
        
        # JSON形式で結果を保存
        output_path = self.cfg.output_dir / "adversarial_test_results.json"
        
        results = {
            "unstable_features": sorted(list(unstable_features)),
            "test_name": "敵対的検証",
            "total_features_tested": len(self.feature_columns),
            "unstable_count": len(unstable_features),
            "execution_time_seconds": int(execution_time),
            "auc_score": float(auc),
            "timestamp": datetime.now().isoformat()
        }
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        logger.info(f"結果を保存: {output_path}")
        
        # adversarial scoresをjoblib形式で保存
        scores_path = self.cfg.output_dir / "adversarial_scores.joblib"
        joblib.dump(adversarial_scores, scores_path)
        logger.info(f"敵対的重要度スコアを保存: {scores_path}")


def main():
    """メイン実行"""
    # Daskの最新エンジンを明示的に有効化
    dask_config.set({'dataframe.query-planning': True})
    
    # Dask分散クライアントを作成（LightGBM用）
    cluster = LocalCluster(
        n_workers=4,
        threads_per_worker=1,
        memory_limit='8GB',
        silence_logs=logging.ERROR
    )
    
    with Client(cluster) as client:
        logger.info(f"Daskクライアント起動: {client.dashboard_link}")
        
        cfg = ProcessingConfig()
        validator = AdversarialValidator(cfg)
        
        try:
            validator._load_data()
            unstable_features, adversarial_scores, auc, execution_time = validator.run_adversarial_validation()
            validator.save_results(unstable_features, adversarial_scores, auc, execution_time)
            
            logger.info("✅ 敵対的検証が正常に完了しました")
            
        except Exception as e:
            logger.error(f"❌ エラーが発生しました: {e}")
            raise


if __name__ == '__main__':
    main()