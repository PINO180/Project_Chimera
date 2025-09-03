"""
統合特徴量パイプライン - Project Forge
Phase 1（独立特徴量）+ Phase 2（非線形組み合わせ）の統合制御
メモリ効率・エラーハンドリング・ログ管理を統合
"""

import pandas as pd
import numpy as np
import os
import sys
import time
import logging
from typing import Dict, List, Optional, Tuple
import gc
import psutil
from pathlib import Path

# プロジェクトパス追加
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root / "features"))

try:
    from independent_features import AdvancedIndependentFeatures, process_chunk_advanced_features
    from combination_features import NonlinearCombinationEngine, process_combination_features
except ImportError as e:
    print(f"モジュールインポートエラー: {e}")
    print("features/ディレクトリにindependent_features.pyとcombination_features.pyが必要です")
    sys.exit(1)

# ロガー設定
logger = logging.getLogger(__name__)

class FeaturePipeline:
    """
    統合特徴量パイプライン
    Phase 1→Phase 2を効率的に実行し、数万の特徴量を生成
    """
    
    def __init__(self, 
                 output_dir: str = "C:/project_forge/output",
                 chunk_size: int = 2000000,
                 memory_limit_gb: float = 8.0,
                 n_processes: Optional[int] = None):
        """
        初期化
        
        Args:
            output_dir: 出力ディレクトリ
            chunk_size: チャンク分割サイズ
            memory_limit_gb: メモリ使用上限（GB）
            n_processes: 並列プロセス数
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.chunk_size = chunk_size
        self.memory_limit_bytes = memory_limit_gb * 1024**3
        self.n_processes = n_processes
        
        # ステータス管理
        self.phase1_completed = False
        self.phase2_completed = False
        self.total_features_generated = 0
        
        # 結果格納
        self.phase1_results = {}
        self.phase2_results = {}
        
        logger.info(f"統合パイプライン初期化: chunk_size={chunk_size}, memory_limit={memory_limit_gb}GB")
    
    def run_complete_pipeline(self, 
                             df: pd.DataFrame,
                             run_phase1: bool = True,
                             run_phase2: bool = True,
                             phase1_params: Dict = None,
                             phase2_params: Dict = None) -> pd.DataFrame:
        """
        完全パイプライン実行
        
        Args:
            df: 入力データ（OHLCV必須）
            run_phase1: Phase 1実行フラグ
            run_phase2: Phase 2実行フラグ
            phase1_params: Phase 1パラメータ
            phase2_params: Phase 2パラメータ
            
        Returns:
            全特徴量が統合されたDataFrame
        """
        logger.info("=== 統合特徴量パイプライン開始 ===")
        logger.info(f"入力データ shape: {df.shape}")
        
        start_time = time.time()
        result_df = df.copy()
        
        try:
            # Phase 1: 独立特徴量生成
            if run_phase1:
                logger.info("Phase 1: 独立特徴量生成開始")
                result_df = self._run_phase1(result_df, phase1_params or {})
                self.phase1_completed = True
                self._save_intermediate_results(result_df, "phase1")
            
            # メモリチェック
            self._check_memory_usage("Phase 1完了後")
            
            # Phase 2: 非線形組み合わせ
            if run_phase2 and self.phase1_completed:
                logger.info("Phase 2: 非線形組み合わせ開始")
                result_df = self._run_phase2(result_df, phase2_params or {})
                self.phase2_completed = True
                self._save_intermediate_results(result_df, "phase2")
            
            # 最終結果保存
            self._save_final_results(result_df)
            
            end_time = time.time()
            total_time = end_time - start_time
            
            logger.info("=== パイプライン完了 ===")
            logger.info(f"総処理時間: {total_time:.2f}秒")
            logger.info(f"最終特徴量数: {result_df.shape[1]}")
            logger.info(f"生成特徴量数: {result_df.shape[1] - df.shape[1]}")
            
            return result_df
            
        except Exception as e:
            logger.error(f"パイプライン実行エラー: {e}")
            self._save_error_state(result_df, e)
            raise
    
    def _run_phase1(self, df: pd.DataFrame, params: Dict) -> pd.DataFrame:
        """Phase 1: 独立特徴量生成"""
        try:
            # パラメータ設定
            window_size = params.get('window_size', 100)
            n_processes = params.get('n_processes', self.n_processes)
            
            # 大きなデータセットの場合はチャンク分割
            if len(df) > self.chunk_size:
                logger.info(f"大容量データ検出。チャンク分割実行: {len(df)} → {self.chunk_size}")
                return self._process_chunks_phase1(df, params)
            
            # 通常処理
            processor = AdvancedIndependentFeatures(
                n_processes=n_processes,
                window_size=window_size
            )
            
            result = processor.calculate_all_features(df, chunk_id="FULL")
            
            phase1_features = result.shape[1] - df.shape[1]
            logger.info(f"Phase 1完了: {phase1_features}個の独立特徴量生成")
            
            return result
            
        except Exception as e:
            logger.error(f"Phase 1エラー: {e}")
            raise
    
    def _run_phase2(self, df: pd.DataFrame, params: Dict) -> pd.DataFrame:
        """Phase 2: 非線形組み合わせ"""
        try:
            # パラメータ設定
            max_combinations = params.get('max_combinations', 20000)
            feature_types = params.get('feature_types', None)
            correlation_threshold = params.get('correlation_threshold', 0.95)
            
            # メモリ使用量に基づく動的調整
            available_memory = self._get_available_memory()
            if available_memory < self.memory_limit_bytes * 0.3:  # 30%未満の場合
                max_combinations = min(max_combinations, 5000)
                logger.warning(f"メモリ不足のため組み合わせ数を制限: {max_combinations}")
            
            # 大きなデータセットの場合はチャンク分割
            if len(df) > self.chunk_size // 2:  # Phase 2はメモリ消費が大きいため半分
                logger.info(f"Phase 2: チャンク分割実行")
                return self._process_chunks_phase2(df, params)
            
            # 通常処理
            engine = NonlinearCombinationEngine(
                n_processes=self.n_processes,
                max_combinations=max_combinations,
                correlation_threshold=correlation_threshold
            )
            
            result = engine.generate_all_combinations(
                df, 
                feature_types=feature_types,
                chunk_id="FULL"
            )
            
            phase2_features = result.shape[1] - df.shape[1]
            logger.info(f"Phase 2完了: {phase2_features}個の組み合わせ特徴量生成")
            
            return result
            
        except Exception as e:
            logger.error(f"Phase 2エラー: {e}")
            raise
    
    def _process_chunks_phase1(self, df: pd.DataFrame, params: Dict) -> pd.DataFrame:
        """Phase 1チャンク分割処理"""
        chunks = []
        
        for i in range(0, len(df), self.chunk_size):
            chunk = df.iloc[i:i+self.chunk_size].copy()
            chunk_id = f"CHUNK_{i//self.chunk_size + 1}"
            
            logger.info(f"Phase 1 {chunk_id} 処理中: {len(chunk)}行")
            
            # チャンク処理
            processed_chunk = process_chunk_advanced_features(
                chunk, chunk_id, 
                n_processes=params.get('n_processes', self.n_processes),
                window_size=params.get('window_size', 100)
            )
            
            chunks.append(processed_chunk)
            
            # メモリクリーンアップ
            del chunk, processed_chunk
            gc.collect()
        
        # チャンク結合
        logger.info("Phase 1チャンク結合中...")
        result = pd.concat(chunks, ignore_index=True)
        
        return result
    
    def _process_chunks_phase2(self, df: pd.DataFrame, params: Dict) -> pd.DataFrame:
        """Phase 2チャンク分割処理"""
        chunks = []
        chunk_size = self.chunk_size // 2  # Phase 2はメモリ消費大
        
        for i in range(0, len(df), chunk_size):
            chunk = df.iloc[i:i+chunk_size].copy()
            chunk_id = f"COMBO_CHUNK_{i//chunk_size + 1}"
            
            logger.info(f"Phase 2 {chunk_id} 処理中: {len(chunk)}行")
            
            # チャンク処理
            processed_chunk = process_combination_features(
                chunk, chunk_id,
                feature_types=params.get('feature_types', None),
                max_combinations=params.get('max_combinations', 10000)
            )
            
            chunks.append(processed_chunk)
            
            # メモリクリーンアップ
            del chunk, processed_chunk
            gc.collect()
        
        # チャンク結合
        logger.info("Phase 2チャンク結合中...")
        result = pd.concat(chunks, ignore_index=True)
        
        return result
    
    def _check_memory_usage(self, stage: str):
        """メモリ使用量チェック"""
        memory_info = psutil.virtual_memory()
        used_gb = (memory_info.total - memory_info.available) / 1024**3
        
        logger.info(f"{stage} - メモリ使用量: {used_gb:.2f}GB / {memory_info.total/1024**3:.2f}GB")
        
        if used_gb > self.memory_limit_bytes / 1024**3:
            logger.warning(f"メモリ使用量が上限({self.memory_limit_bytes/1024**3:.1f}GB)を超過")
            gc.collect()  # ガベージコレクション実行
    
    def _get_available_memory(self) -> int:
        """利用可能メモリ量取得"""
        memory_info = psutil.virtual_memory()
        return memory_info.available
    
    def _save_intermediate_results(self, df: pd.DataFrame, phase: str):
        """中間結果保存"""
        try:
            filename = self.output_dir / f"{phase}_results.parquet"
            df.to_parquet(filename, compression='snappy')
            logger.info(f"{phase} 中間結果保存: {filename}")
        except Exception as e:
            logger.warning(f"中間結果保存エラー: {e}")
    
    def _save_final_results(self, df: pd.DataFrame):
        """最終結果保存"""
        try:
            # Parquet形式（推奨）
            parquet_file = self.output_dir / "final_features.parquet"
            df.to_parquet(parquet_file, compression='snappy')
            
            # CSV形式（バックアップ）
            csv_file = self.output_dir / "final_features.csv"
            df.to_csv(csv_file, index=False)
            
            # 統計情報保存
            stats_file = self.output_dir / "feature_statistics.txt"
            with open(stats_file, 'w', encoding='utf-8') as f:
                f.write(f"=== Project Forge 特徴量統計 ===\n")
                f.write(f"データ行数: {len(df):,}\n")
                f.write(f"総特徴量数: {df.shape[1]:,}\n")
                f.write(f"メモリ使用量: {df.memory_usage(deep=True).sum() / 1024**2:.1f} MB\n")
                f.write(f"Phase 1完了: {self.phase1_completed}\n")
                f.write(f"Phase 2完了: {self.phase2_completed}\n")
            
            logger.info(f"最終結果保存完了: {parquet_file}")
            
        except Exception as e:
            logger.error(f"最終結果保存エラー: {e}")
    
    def _save_error_state(self, df: pd.DataFrame, error: Exception):
        """エラー時の状態保存"""
        try:
            error_file = self.output_dir / "error_state.parquet"
            df.to_parquet(error_file)
            
            error_log = self.output_dir / "error_log.txt"
            with open(error_log, 'w', encoding='utf-8') as f:
                f.write(f"エラー発生時刻: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"エラー内容: {str(error)}\n")
                f.write(f"Phase 1完了: {self.phase1_completed}\n")
                f.write(f"Phase 2完了: {self.phase2_completed}\n")
                f.write(f"データ形状: {df.shape}\n")
            
            logger.info(f"エラー状態保存: {error_file}")
            
        except Exception as save_error:
            logger.error(f"エラー状態保存失敗: {save_error}")


def run_feature_pipeline(input_file: str,
                        output_dir: str = "C:/project_forge/output",
                        run_phase1: bool = True,
                        run_phase2: bool = True,
                        **kwargs) -> str:
    """
    特徴量パイプライン実行（外部呼び出し用）
    
    Args:
        input_file: 入力データファイル（CSV/Parquet）
        output_dir: 出力ディレクトリ
        run_phase1: Phase 1実行フラグ
        run_phase2: Phase 2実行フラグ
        **kwargs: 追加パラメータ
        
    Returns:
        出力ファイルパス
    """
    # ログ設定
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(f"{output_dir}/pipeline.log"),
            logging.StreamHandler()
        ]
    )
    
    try:
        # データ読み込み
        logger.info(f"データ読み込み: {input_file}")
        if input_file.endswith('.parquet'):
            df = pd.read_parquet(input_file)
        else:
            df = pd.read_csv(input_file)
        
        # パイプライン実行
        pipeline = FeaturePipeline(output_dir=output_dir, **kwargs)
        result_df = pipeline.run_complete_pipeline(
            df, 
            run_phase1=run_phase1,
            run_phase2=run_phase2
        )
        
        output_file = f"{output_dir}/final_features.parquet"
        logger.info(f"パイプライン完了: {output_file}")
        
        return output_file
        
    except Exception as e:
        logger.error(f"パイプライン実行エラー: {e}")
        raise


if __name__ == "__main__":
    # テスト実行
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    
    # サンプルデータ生成
    np.random.seed(42)
    n_samples = 5000
    
    test_data = pd.DataFrame({
        'open': np.random.uniform(95, 105, n_samples),
        'high': np.random.uniform(100, 110, n_samples),
        'low': np.random.uniform(90, 100, n_samples),
        'close': np.random.uniform(95, 105, n_samples),
        'volume': np.random.lognormal(8, 1, n_samples).astype(int),
        'timestamp': pd.date_range('2020-01-01', periods=n_samples, freq='H')
    })
    
    print("=== 統合特徴量パイプラインテスト開始 ===")
    print(f"テストデータ shape: {test_data.shape}")
    
    # パイプライン実行
    pipeline = FeaturePipeline(
        output_dir="./test_output",
        chunk_size=2000,  # テスト用に小さく
        memory_limit_gb=4.0
    )
    
    start_time = time.time()
    
    result = pipeline.run_complete_pipeline(
        test_data,
        run_phase1=True,
        run_phase2=True,
        phase1_params={'window_size': 50},
        phase2_params={'max_combinations': 1000}
    )
    
    end_time = time.time()
    
    print(f"\n=== テスト完了 ===")
    print(f"処理時間: {end_time - start_time:.2f}秒")
    print(f"入力特徴量数: {test_data.shape[1]}")
    print(f"最終特徴量数: {result.shape[1]}")
    print(f"生成特徴量数: {result.shape[1] - test_data.shape[1]}")
    print("🎉 統合パイプラインが正常に動作しています！")