"""
adversarial_validation_futures.py - 最終版（完全ディスク経由処理）
第一防衛線 - 敵対的検証

アーキテクチャ：
- 非tick・tick両方をディスク経由2段階処理
- メモリ不足を完全回避
- 全データで正確なAUC計算（精度維持）
"""
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))

import blueprint as config
import logging
from dask.distributed import Client, LocalCluster
from typing import List, Set, Dict, Tuple
import json
import time
from datetime import datetime
import shutil

import pyarrow as pa
import pyarrow.parquet as pq

import warnings
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=FutureWarning)

import pandas as pd
import dask.dataframe as dd
from lightgbm import LGBMClassifier
from lightgbm.dask import DaskLGBMClassifier
from sklearn.metrics import roc_auc_score
import numpy as np
import scipy.sparse as sp
import joblib
import gc

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ProcessingConfig:
    def __init__(self):
        self.feature_universe_path = str(config.S2_FEATURES_AFTER_KS)
        self.output_dir = config.S3_ARTIFACTS
        self.adversarial_auc_threshold = 0.7

def process_single_file_to_disk(file_path: str, output_dir: Path, file_index: int) -> Dict:
    """
    単一ファイルを処理し、予測結果をディスク保存
    """
    path_obj = Path(file_path)
    path_name = path_obj.name
    
    try:
        df = pd.read_parquet(file_path, engine='pyarrow')
        
        non_feature_columns = ['timestamp', 'open', 'high', 'low', 'close', 'volume', 'timeframe']
        feature_columns = [col for col in df.columns if col not in non_feature_columns]

        if not feature_columns or 'timestamp' not in df.columns or len(df) < 100:
            return {"path": path_name, "status": "skipped", "features": None}

        split_point = len(df) // 2
        df['adversarial_label'] = 0
        df.loc[df.index[split_point:], 'adversarial_label'] = 1

        if df['adversarial_label'].nunique() < 2:
            return {"path": path_name, "status": "skipped", "features": None}

        X = df[feature_columns]
        y = df['adversarial_label']
        
        model = LGBMClassifier(
            n_estimators=100,
            max_depth=5,
            learning_rate=0.1,
            random_state=42,
            verbosity=-1,
            n_jobs=-1
        )
        model.fit(X, y)

        predictions = model.predict_proba(X)
        is_sparse = sp.issparse(predictions)
        if is_sparse:
            predictions_array = predictions.toarray()
        else:
            predictions_array = predictions
        y_pred_proba = predictions_array[:, 1]  # type: ignore
        
        # 予測結果をディスク保存
        pred_df = pd.DataFrame({
            'y_true': y.values,
            'y_pred': y_pred_proba
        })
        pred_df.to_parquet(output_dir / f"pred_{file_index:04d}.parquet")
        
        # 特徴量重要度とタイムフレーム
        importances = model.feature_importances_
        timeframe_suffix = f"_{path_obj.stem.split('_')[-1]}"
        
        feature_importance_dict = dict(zip(feature_columns, importances))
        
        del df, X, y, model, predictions, pred_df
        if is_sparse:
            del predictions_array
        gc.collect()
        
        logger.info(f"✓ {path_name} 予測保存完了")
        
        return {
            "path": path_name,
            "status": "success",
            "features": feature_columns,
            "importances": feature_importance_dict,
            "timeframe_suffix": timeframe_suffix
        }
        
    except Exception as e:
        logger.error(f"✗ {path_name} 処理失敗: {str(e)}")
        return {"path": path_name, "status": "failed", "error": str(e), "features": None}


class AdversarialValidator:
    def __init__(self, cfg: ProcessingConfig):
        self.cfg = cfg

    def _get_data_sources(self) -> Tuple[List[Path], List[Path]]:
        """データソースを検出し、tickと非tickを分離"""
        feature_universe_path = Path(self.cfg.feature_universe_path)
        if not feature_universe_path.exists():
            raise FileNotFoundError(f"ディレクトリが見つかりません: {feature_universe_path}")

        all_parquet_files = []
        tick_directories = []
        
        # 全てのエンジンディレクトリをスキャンする元のロジックに戻す
        for engine_dir in feature_universe_path.iterdir():
            if not engine_dir.is_dir():
                continue
            for item in engine_dir.iterdir():
                if item.is_dir():
                    tick_directories.append(item)
                    logger.info(f"tickデータを検出: {item.name}")
                elif item.is_file() and item.suffix == '.parquet':
                    all_parquet_files.append(item)
        
        logger.info(f"非tickデータ: {len(all_parquet_files)}個")
        logger.info(f"tickデータ: {len(tick_directories)}個")
        
        return all_parquet_files, tick_directories

    def _process_non_tick_files(self, parquet_files: List[Path]) -> Tuple[float, Dict, Set]:
        """非tickファイルをディスク経由で処理"""
        temp_dir = self.cfg.output_dir / "temp_non_tick"
        temp_dir.mkdir(parents=True, exist_ok=True)
        
        try:
            logger.info(f"フェーズ1: {len(parquet_files)}個の非tickファイルを処理中...")
            
            all_importances = []
            
            for i, file_path in enumerate(parquet_files):
                result = process_single_file_to_disk(str(file_path), temp_dir, i)
                
                if result['status'] == 'success':
                    all_importances.append({
                        'importances': result['importances'],
                        'suffix': result['timeframe_suffix'],
                        'features': result['features']
                    })
            
            logger.info("フェーズ2: 全予測を集約してAUC計算中...")
            
            # 全予測ファイルを読み込んでAUC計算
            all_y_true = []
            all_y_pred = []
            
            pred_files = sorted(temp_dir.glob("pred_*.parquet"))
            for pred_file in pred_files:
                pred_df = pd.read_parquet(pred_file)
                all_y_true.extend(pred_df['y_true'].tolist())
                all_y_pred.extend(pred_df['y_pred'].tolist())
            
            auc = roc_auc_score(all_y_true, all_y_pred) if all_y_true else 0.5
            
            # 特徴量重要度集約（サフィックス付き）
            all_scores = {}
            unstable_features = set()
            
            for imp_data in all_importances:
                suffix = imp_data['suffix']
                for feat, score in imp_data['importances'].items():
                    feat_with_suffix = f"{feat}{suffix}"
                    all_scores[feat_with_suffix] = float(score)
            
            # 不安定特徴量判定
            if auc > self.cfg.adversarial_auc_threshold:
                threshold_importance = np.percentile(list(all_scores.values()), 80)
                unstable_features = {
                    feat for feat, score in all_scores.items()
                    if score > threshold_importance
                }
            
            return auc, all_scores, unstable_features
            
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)

    def _process_tick_data(self, tick_path: Path) -> Tuple[float, Dict, Set]:
        """tickデータを時系列分割で処理（真の脱結合・完全ディスク経由方式）"""
        path_name = tick_path.name
        temp_dir = self.cfg.output_dir / f"temp_{path_name}"
        
        if temp_dir.exists():
            shutil.rmtree(temp_dir)
        temp_dir.mkdir(parents=True, exist_ok=True)
        
        try:
            logger.info(f"tickデータ処理開始: {path_name} (真の脱結合モード)")
            
            partition_files = []
            for pf in tick_path.rglob("*.parquet"):
                parts = pf.parts
                year = month = day = 0
                for part in parts:
                    if part.startswith('year='): year = int(part.split('=')[1])
                    elif part.startswith('month='): month = int(part.split('=')[1])
                    elif part.startswith('day='): day = int(part.split('=')[1])
                partition_files.append((year, month, day, pf))
            
            partition_files.sort(key=lambda x: (x[0], x[1], x[2]))
            partition_paths = [pf[3] for pf in partition_files]
            
            logger.info(f"{path_name}: {len(partition_paths)}個のパーティションを検出")

            logger.info(f"フェーズ1: {len(partition_paths)}個のパーティションで「ミニAV」を逐次実行中...")
            all_feature_importances = []
            feature_columns = None
            
            for i, part_path in enumerate(partition_paths):
                try:
                    df = pd.read_parquet(part_path, dtype_backend='numpy_nullable')
                    
                    if feature_columns is None:
                        non_feature_columns = ['timestamp', 'open', 'high', 'low', 'close', 'volume', 'timeframe', 'year', 'month', 'day']
                        feature_columns = [col for col in df.columns if col not in non_feature_columns]

                    if len(df) < 100: continue

                    split_point = len(df) // 2
                    df['adversarial_label'] = 0
                    df.iloc[split_point:, df.columns.get_loc('adversarial_label')] = 1
                    
                    if df['adversarial_label'].nunique() < 2: continue
                    
                    X = df[feature_columns]
                    y = df['adversarial_label']
                    
                    model = LGBMClassifier(
                        n_estimators=100, max_depth=5, learning_rate=0.1,
                        random_state=42, verbosity=-1, n_jobs=-1
                    )
                    model.fit(X, y)
                    
                    predictions = model.predict_proba(X)
                    is_sparse = sp.issparse(predictions)
                    y_pred_proba = predictions.toarray()[:, 1] if is_sparse else predictions[:, 1] #type: ignore
                    
                    pred_df = pd.DataFrame({'y_true': y.values, 'y_pred': y_pred_proba})
                    pred_df.to_parquet(temp_dir / f"pred_{i:04d}.parquet")
                    
                    all_feature_importances.append(dict(zip(feature_columns, model.feature_importances_)))
                    
                    del df, X, y, model, pred_df
                    gc.collect()

                    # --- ▼▼▼ ここに進捗表示を追加 ▼▼▼ ---
                    if (i + 1) % 50 == 0:
                        logger.info(f"  ... {i + 1}/{len(partition_paths)} パーティションの処理完了")
                    # --- ▲▲▲ 修正ここまで ▲▲▲ ---

                except Exception as e:
                    logger.warning(f"パーティション {part_path.name} の処理でエラー: {e}")
                    continue

            logger.info(f"{path_name}: フェーズ1完了。全予測を集約してAUC計算中...")

            all_y_true, all_y_pred = [], []
            pred_files = sorted(temp_dir.glob("pred_*.parquet"))
            for pred_file in pred_files:
                pred_df = pd.read_parquet(pred_file)
                all_y_true.extend(pred_df['y_true'].tolist())
                all_y_pred.extend(pred_df['y_pred'].tolist())

            if not all_y_true or len(np.unique(all_y_true)) < 2:
                auc = 0.5
                logger.warning(f"{path_name}: AUC計算不可（ラベルが1種類のみ）。AUC=0.5とします。")
            else:
                auc = roc_auc_score(all_y_true, all_y_pred)
            
            all_features = set()
            for imp_dict in all_feature_importances:
                all_features.update(imp_dict.keys())
            
            adversarial_scores = {
                f"{feat}_tick": float(np.mean([imp.get(feat, 0) for imp in all_feature_importances]))
                for feat in all_features
            }
            
            unstable_features = set()
            if auc > self.cfg.adversarial_auc_threshold:
                threshold_importance = np.percentile(list(adversarial_scores.values()), 80)
                unstable_features = {feat for feat, score in adversarial_scores.items() if score > threshold_importance}
            
            logger.info(f"✓ {path_name} 処理成功 (AUC: {auc:.4f})")
            
            return auc, adversarial_scores, unstable_features
            
        except Exception as e:
            logger.error(f"✗ {path_name} 処理失敗: {str(e)}")
            return 0.5, {}, set()
        finally:
            if temp_dir.exists():
                shutil.rmtree(temp_dir)
                logger.info(f"一時ディレクトリを削除: {temp_dir}")

    def run_validation(self) -> None:
        """完全ディスク経由検証パイプライン"""
        start_time = time.time()
        logger.info("=" * 60)
        logger.info("敵対的検証（完全ディスク経由処理）を開始")
        logger.info("=" * 60)
        
        parquet_files, tick_directories = self._get_data_sources()
        
        all_unstable_features: Set[str] = set()
        all_adversarial_scores: Dict[str, float] = {}
        
        # 非tick処理
        if parquet_files:
            non_tick_auc, non_tick_scores, non_tick_unstable = self._process_non_tick_files(parquet_files)
            all_adversarial_scores.update(non_tick_scores)
            all_unstable_features.update(non_tick_unstable)
            logger.info(f"非tick処理完了 (AUC: {non_tick_auc:.4f})")
        
        # tick処理
        for tick_dir in tick_directories:
            tick_auc, tick_scores, tick_unstable = self._process_tick_data(tick_dir)
            all_adversarial_scores.update(tick_scores)
            all_unstable_features.update(tick_unstable)
        
        execution_time = time.time() - start_time
        
        logger.info("=" * 60)
        logger.info("敵対的検証完了")
        logger.info(f"不安定特徴量総数: {len(all_unstable_features)}")
        logger.info(f"実行時間: {execution_time:.2f}秒")
        logger.info("=" * 60)

        self.save_results(all_unstable_features, all_adversarial_scores, execution_time)

    def save_results(self, unstable_features: Set[str], adversarial_scores: Dict[str, float], 
                     execution_time: float) -> None:
        """結果を保存"""
        self.cfg.output_dir.mkdir(parents=True, exist_ok=True)
        
        output_path_unstable = self.cfg.output_dir / "av_unstable_features.json"
        results_unstable = {
            "unstable_features": sorted(list(unstable_features)),
            "test_name": "敵対的検証（完全ディスク経由 v1.0）",
            "unstable_count": len(unstable_features),
            "execution_time_seconds": round(execution_time, 2),
            "timestamp": datetime.now().isoformat()
        }
        with open(output_path_unstable, 'w', encoding='utf-8') as f:
            json.dump(results_unstable, f, indent=2, ensure_ascii=False)
        logger.info(f"✓ 不安定特徴量リスト保存: {output_path_unstable}")
        
        scores_path = self.cfg.output_dir / "adversarial_scores.joblib"
        joblib.dump(adversarial_scores, scores_path)
        logger.info(f"✓ 敵対的重要度スコア保存: {scores_path}")


def main():
    """メイン実行（Dask完全排除・シングルプロセスモード）"""
    try:
        logger.info("✓ シングルプロセスモード（Dask不使用）で開始します")
        
        cfg = ProcessingConfig()
        validator = AdversarialValidator(cfg)
    
        validator.run_validation()
        logger.info("✅ 敵対的検証パイプラインが正常に完了しました")
        
    except Exception as e:
        logger.critical(f"❌ パイプライン実行中に致命的なエラー: {e}", exc_info=True)
        raise
    
    finally:
        logger.info("処理が終了しました。")
        time.sleep(2)


if __name__ == '__main__':
    main()