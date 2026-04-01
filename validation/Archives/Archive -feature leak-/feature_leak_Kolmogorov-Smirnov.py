"""
Kolmogorov-Smirnov.py v5.1 - メモリ安全版
第一防衛線 - 分布安定性テスト（KS検定のみ）

アーキテクチャ：完全な「脱・結合」
- v5.0のメモリリーク問題を解決
- スキーマ読み込みをpyarrow.parquet.read_schemaに変更し、巨大データのメモリ展開を完全に回避
"""
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))

import blueprint as config
import logging
import time
import json
from datetime import datetime
from typing import List, Set, Tuple

import dask
import dask.dataframe as dd
from dask.delayed import delayed  # dask.delayed から delayed をインポート
from dask.base import compute    # dask.base から compute をインポート
from scipy.stats import ks_2samp
from tqdm import tqdm
import pyarrow.parquet as pq # ★★★ pyarrow.parquetを直接インポート

# --- ロギング設定 ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- 設定クラス ---
class ProcessingConfig:
    def __init__(self):
        self.feature_universe_path = str(config.S2_FEATURES_FIXED)
        self.output_dir = config.S3_ARTIFACTS
        self.ks_p_value_threshold = 0.05
        self.sample_size = 1_000_000

# --- 本体クラス ---
class KSTestValidator:
    def __init__(self, cfg: ProcessingConfig):
        self.cfg = cfg

    def _get_data_sources(self) -> List[Path]:
        """データソース（単一Parquetファイル or パーティションディレクトリ）を検出する"""
        feature_universe_path = Path(self.cfg.feature_universe_path)
        if not feature_universe_path.exists() or not feature_universe_path.is_dir():
            raise FileNotFoundError(f"ディレクトリが見つかりません: {feature_universe_path}")

        all_data_sources = []
        for engine_dir in feature_universe_path.iterdir():
            if not engine_dir.is_dir():
                continue
            for item in engine_dir.iterdir():
                if item.is_dir() or (item.is_file() and item.suffix == '.parquet'):
                    all_data_sources.append(item)
        
        if not all_data_sources:
            raise FileNotFoundError("データソースが見つかりません")
        
        logger.info(f"{len(all_data_sources)}個のデータソースを検出")
        return all_data_sources

    @staticmethod
    @delayed
    def _run_single_ks_test(
        path: str, 
        column_name: str, 
        p_value_threshold: float,
        sample_size: int
    ) -> Tuple[str, bool, str]:
        """単一の特徴量カラムに対してKS検定を実行する遅延タスク"""
        try:
            # 必要な列だけを読み込むように最適化
            ddf = dd.read_parquet(path, columns=['timestamp', column_name], engine='pyarrow')
            ddf['timestamp'] = ddf['timestamp'].astype('datetime64[ns]')

            split_point = ddf.npartitions // 2
            if split_point == 0:
                return (column_name, True, "データ不足")

            ddf_first_half = ddf.partitions[:split_point]
            ddf_second_half = ddf.partitions[split_point:]

            count1, count2 = compute(ddf_first_half[column_name].count(), ddf_second_half[column_name].count())

            if count1 < 100 or count2 < 100:
                 return (column_name, True, "サンプル数不足")

            frac1 = min(sample_size / count1, 1.0)
            frac2 = min(sample_size / count2, 1.0)
            
            sample1_future = ddf_first_half[column_name].dropna().sample(frac=frac1, random_state=42)
            sample2_future = ddf_second_half[column_name].dropna().sample(frac=frac2, random_state=42)
            sample1, sample2 = compute(sample1_future, sample2_future)

            if len(sample1) < 10 or len(sample2) < 10:
                return (column_name, True, "サンプル数不足")

            _, p_value = ks_2samp(sample1, sample2)
            
            is_stable = p_value >= p_value_threshold
            return (column_name, is_stable, f"p_value={p_value:.4f}")
        
        except Exception as e:
            return (column_name, False, f"エラー: {str(e)[:50]}")


    def run_validation(self) -> None:
        """検証パイプライン全体を実行"""
        start_time = time.time()
        logger.info("=" * 60)
        logger.info("KS検定（最終アーキテクチャ・メモリ安全版）を開始")
        logger.info("=" * 60)
        
        data_sources = self._get_data_sources()
        all_tasks = []
        total_feature_count = 0

        logger.info("全データソースをスキャンし、検定タスクを生成中...")
        for path in tqdm(data_sources, desc="データソーススキャン中", ncols=100):
            try:
                # --- ★★★ ここが最重要修正点 ★★★ ---
                # pandasで全読み込みするのではなく、pyarrowでスキーマだけを高速に読み込む
                if path.is_dir():
                    # ディレクトリの場合は、中の最初のParquetファイルを探してスキーマを読む
                    first_file = next(path.glob("**/*.parquet"), None)
                    if first_file is None:
                        continue
                    schema = pq.read_schema(first_file)
                else:
                    # ファイルの場合は、そのファイルのスキーマを読む
                    schema = pq.read_schema(path)
                
                columns = schema.names
                # --- 修正ここまで ---
                
                non_feature_columns = ['timestamp', 'open', 'high', 'low', 'close', 'volume', 'timeframe']
                
                timeframe_suffix = ""
                if path.is_dir() or '_tick' in path.name:
                    timeframe_suffix = '_tick'
                else:
                    parts = path.stem.split('_')
                    if len(parts) > 1:
                        timeframe_suffix = f"_{parts[-1]}"

                for col in columns:
                    if col not in non_feature_columns:
                        final_feature_name = f"{col}{timeframe_suffix}"
                        task = self._run_single_ks_test(
                            str(path), col, self.cfg.ks_p_value_threshold, self.cfg.sample_size
                        )
                        all_tasks.append((final_feature_name, task))
                        total_feature_count += 1
            except Exception as e:
                logger.warning(f"'{path.name}' のスキャンに失敗: {e}")
        
        logger.info(f"{total_feature_count}個の検定タスク生成完了。並列実行を開始します...")

        feature_names = [t[0] for t in all_tasks]
        delayed_objects = [t[1] for t in all_tasks]
        
        results = []
        BATCH_SIZE = 500 # 一度に実行するタスク数（PCのコア数に応じて調整）
        for i in tqdm(range(0, len(delayed_objects), BATCH_SIZE), desc="検定タスク並列実行中", ncols=100):
            batch = delayed_objects[i:i+BATCH_SIZE]
            results.extend(compute(*batch, scheduler='processes'))
        
        logger.info("全検定タスクの並列実行完了。")

        # 結果を集計
        unstable_features: Set[str] = set()
        for i, feature_name in enumerate(feature_names):
            original_col_name, is_stable, reason = results[i]
            if not is_stable:
                unstable_features.add(feature_name)

        execution_time = time.time() - start_time
        
        logger.info("=" * 60)
        logger.info("KS検定完了")
        logger.info(f"検証特徴量総数: {total_feature_count}")
        logger.info(f"不安定特徴量数: {len(unstable_features)}")
        logger.info(f"安定特徴量数: {total_feature_count - len(unstable_features)}")
        logger.info(f"実行時間: {execution_time:.2f}秒")
        logger.info("=" * 60)

        self.save_results(unstable_features, total_feature_count, execution_time)

    def save_results(self, unstable_features: Set[str], total_count: int, execution_time: float) -> None:
        """結果をJSON形式で保存"""
        self.cfg.output_dir.mkdir(parents=True, exist_ok=True)
        output_path = self.cfg.output_dir / "ks_unstable_features.json"
        
        results = {
            "unstable_features": sorted(list(unstable_features)),
            "test_name": "KS検定（v5.1 - メモリ安全版）",
            "total_features_tested": total_count,
            "unstable_count": len(unstable_features),
            "execution_time_seconds": round(execution_time, 2),
            "timestamp": datetime.now().isoformat()
        }
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        logger.info(f"不安定特徴量のリストを保存しました: {output_path}")

def main():
    """メイン実行"""
    try:
        cfg = ProcessingConfig()
        validator = KSTestValidator(cfg)
        validator.run_validation()
        logger.info("✅ KS検定パイプラインが正常に完了しました。")
        
    except Exception as e:
        logger.critical(f"❌ パイプライン実行中に致命的なエラーが発生しました: {e}", exc_info=True)
        raise

if __name__ == '__main__':
    main()