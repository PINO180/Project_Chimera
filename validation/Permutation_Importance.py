"""
Permutation_Importance.py
第一防衛線 - 重要度安定性テスト（Permutation Importanceのみ）

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

# 不要な警告を抑制
import warnings
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=FutureWarning)
logging.getLogger('distributed').setLevel(logging.ERROR)
logging.getLogger('distributed.shuffle').setLevel(logging.ERROR)

import dask
import dask.dataframe as dd
from dask.dataframe import DataFrame as DaskDataFrame
from dask import config as dask_config
from dask.distributed import Client, LocalCluster
from lightgbm.dask import DaskLGBMRegressor
import psutil
from tqdm import tqdm
import joblib


class ProcessingConfig:
    """設定管理クラス"""
    def __init__(self):
        self.feature_universe_path = str(config.S2_FEATURES2)
        self.output_dir = config.S3_ARTIFACTS
        self.n_jobs = 4
        self.importance_change_threshold = 0.5
        self.memory_warning_gb = 50
        self.memory_critical_gb = 55


class PermutationImportanceValidator:
    """Permutation Importanceによる重要度安定性テスト"""
    
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
        logger.info(f"PI検定: 特徴量ユニバース '{feature_universe_path}' をDask DataFrameとして読み込み中...")

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
    
    def run_pi_test(self) -> tuple[Set[str], float]:
        """
        Permutation Importanceによる重要度安定性テスト
        """
        logger.info("=" * 60)
        logger.info("Permutation Importanceテストを開始")
        logger.info("=" * 60)
        
        start_time = time.time()
        
        if self.ddf is None:
            raise ValueError("Dask DataFrame not loaded.")
        
        self._check_memory()
        
        logger.info("ターゲット変数を計算中...")
        target_horizon = 60
        # pct_change()の手動実装: (future_close - current_close) / current_close
        close_col = self.ddf['close']
        future_close = close_col.shift(-target_horizon)
        target = (future_close - close_col) / close_col
        ddf_with_target = self.ddf.assign(target=target).dropna()
        
        split_point = ddf_with_target.npartitions // 2
        p1_ddf = ddf_with_target.partitions[:split_point]
        p2_ddf = ddf_with_target.partitions[split_point:]
        
        all_p1_importances: Dict[str, float] = {}
        all_p2_importances: Dict[str, float] = {}
        
        # 前半期間
        logger.info("前半期間のモデルで特徴量重要度を計算中...")
        
        # モデルチェックポイント
        model1_file = self.cfg.output_dir / "pi_model1.joblib"
        baseline1_file = self.cfg.output_dir / "pi_baseline1.joblib"
        
        if model1_file.exists() and baseline1_file.exists():
            logger.info("保存済みモデルを読み込み（訓練スキップ）")
            model1 = joblib.load(model1_file)
            baseline_score1 = joblib.load(baseline1_file)
        else:
            # データ分割
            n_p1 = p1_ddf.npartitions
            train_end = int(n_p1 * 0.7)
            p1_train_ddf = p1_ddf.partitions[:train_end]
            p1_val_ddf = p1_ddf.partitions[train_end:]
            
            X1_train = p1_train_ddf[self.feature_columns]
            y1_train = p1_train_ddf['target']
            
            logger.info("前半期間モデルを訓練中...")
            model1 = DaskLGBMRegressor(n_estimators=50, random_state=42, verbosity=-1, n_jobs=self.cfg.n_jobs)
            model1.fit(X1_train, y1_train)
            
            # モデル保存
            joblib.dump(model1, model1_file)
            logger.info(f"モデル保存: {model1_file}")
        
        # 検証データ（毎回必要）
        n_p1 = p1_ddf.npartitions
        train_end = int(n_p1 * 0.7)
        p1_val_ddf = p1_ddf.partitions[train_end:]
        X1_val = p1_val_ddf[self.feature_columns]
        y1_val = p1_val_ddf['target']
        
        if not baseline1_file.exists():
            logger.info("前半期間: ベースラインスコアを計算...")
            baseline_score1 = model1.score(X1_val, y1_val)
            joblib.dump(baseline_score1, baseline1_file)
            logger.info(f"ベースラインスコア: {baseline_score1:.6f}")
        
        self._check_memory()
        
        for feature in tqdm(self.feature_columns, desc="PI (前半期間)"):
            shuffled_col = X1_val[feature].map_partitions(
                lambda s: s.sample(frac=1, random_state=42)
            )
            X1_val_permuted = X1_val.assign(**{feature: shuffled_col})
            permuted_score = model1.score(X1_val_permuted, y1_val)
            all_p1_importances[feature] = baseline_score1 - permuted_score
        
        # 後半期間
        logger.info("後半期間のモデルで特徴量重要度を計算中...")
        
        # モデルチェックポイント
        model2_file = self.cfg.output_dir / "pi_model2.joblib"
        baseline2_file = self.cfg.output_dir / "pi_baseline2.joblib"
        
        if model2_file.exists() and baseline2_file.exists():
            logger.info("保存済みモデルを読み込み（訓練スキップ）")
            model2 = joblib.load(model2_file)
            baseline_score2 = joblib.load(baseline2_file)
        else:
            # データ分割
            n_p2 = p2_ddf.npartitions
            train_end = int(n_p2 * 0.7)
            p2_train_ddf = p2_ddf.partitions[:train_end]
            p2_val_ddf = p2_ddf.partitions[train_end:]
            
            X2_train = p2_train_ddf[self.feature_columns]
            y2_train = p2_train_ddf['target']
            
            logger.info("後半期間モデルを訓練中...")
            model2 = DaskLGBMRegressor(n_estimators=50, random_state=42, verbosity=-1, n_jobs=self.cfg.n_jobs)
            model2.fit(X2_train, y2_train)
            
            # モデル保存
            joblib.dump(model2, model2_file)
            logger.info(f"モデル保存: {model2_file}")
        
        # 検証データ（毎回必要）
        n_p2 = p2_ddf.npartitions
        train_end = int(n_p2 * 0.7)
        p2_val_ddf = p2_ddf.partitions[train_end:]
        X2_val = p2_val_ddf[self.feature_columns]
        y2_val = p2_val_ddf['target']
        
        if not baseline2_file.exists():
            logger.info("後半期間: ベースラインスコアを計算...")
            baseline_score2 = model2.score(X2_val, y2_val)
            joblib.dump(baseline_score2, baseline2_file)
            logger.info(f"ベースラインスコア: {baseline_score2:.6f}")
        
        self._check_memory()
        
        for feature in tqdm(self.feature_columns, desc="PI (後半期間)"):
            shuffled_col = X2_val[feature].map_partitions(
                lambda s: s.sample(frac=1, random_state=42)
            )
            X2_val_permuted = X2_val.assign(**{feature: shuffled_col})
            permuted_score = model2.score(X2_val_permuted, y2_val)
            all_p2_importances[feature] = baseline_score2 - permuted_score
        
        # 不安定特徴量の検出
        unstable_features: Set[str] = set()
        for feature in self.feature_columns:
            p1_imp = all_p1_importances.get(feature, 0.0)
            p2_imp = all_p2_importances.get(feature, 0.0)
            
            importance_change = abs(p2_imp - p1_imp) / (abs(p1_imp) + 1e-9)
            if importance_change > self.cfg.importance_change_threshold:
                unstable_features.add(feature)
        
        execution_time = time.time() - start_time
        
        logger.info("=" * 60)
        logger.info("Permutation Importanceテスト完了")
        logger.info(f"検証特徴量数: {len(self.feature_columns)}")
        logger.info(f"不安定特徴量数: {len(unstable_features)}")
        logger.info(f"実行時間: {execution_time:.2f}秒")
        logger.info("=" * 60)
        
        return unstable_features, execution_time
    
    def save_results(self, unstable_features: Set[str], execution_time: float) -> None:
        """結果をJSON形式で保存"""
        self.cfg.output_dir.mkdir(parents=True, exist_ok=True)
        
        output_path = self.cfg.output_dir / "pi_test_results.json"
        
        results = {
            "unstable_features": sorted(list(unstable_features)),
            "test_name": "Permutation Importance",
            "total_features_tested": len(self.feature_columns),
            "unstable_count": len(unstable_features),
            "execution_time_seconds": int(execution_time),
            "timestamp": datetime.now().isoformat()
        }
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        logger.info(f"結果を保存: {output_path}")


def main():
    """メイン実行"""
    # Daskの最新エンジンを明示的に有効化
    dask_config.set({'dataframe.query-planning': True})
    
    # Dask分散クライアントを作成（LightGBM用）
    cluster = LocalCluster(
        n_workers=2,  # Worker数削減でメモリ余裕を確保
        threads_per_worker=2,
        memory_limit='16GB',  # メモリ上限増加
        silence_logs=logging.ERROR
    )
    
    with Client(cluster) as client:
        logger.info(f"Daskクライアント起動: {client.dashboard_link}")
        
        cfg = ProcessingConfig()
        validator = PermutationImportanceValidator(cfg)
        
        try:
            validator._load_data()
            unstable_features, execution_time = validator.run_pi_test()
            validator.save_results(unstable_features, execution_time)
            
            logger.info("✅ Permutation Importanceテストが正常に完了しました")
            
        except Exception as e:
            logger.error(f"❌ エラーが発生しました: {e}")
            raise


if __name__ == '__main__':
    main()