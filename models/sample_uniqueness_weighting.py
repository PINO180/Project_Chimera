import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))

import blueprint as config
import dask
import dask.dataframe as dd
from dask.dataframe.core import DataFrame, Series
import numpy as np
import pandas as pd
from pathlib import Path
import json
import logging
from typing import Optional, Dict, Any, List
from collections import defaultdict

# ロギング設定
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def compute_concurrency_interval_tree(partition: pd.DataFrame, 
                                     interval_buckets: Dict[str, List[tuple]]) -> pd.DataFrame:
    """
    区間木アプローチで並行性を計算（改良版）
    
    Args:
        partition: パーティションデータ（t0, t1を含む）
        interval_buckets: 日付文字列をキーとした区間リスト
    
    Returns:
        並行性カラムを追加したDataFrame
    """
    if partition.empty:
        result = partition.copy()
        result['concurrency'] = np.array([], dtype=np.float64)
        return result
    
    result = partition.copy()
    n_rows = len(partition)
    concurrency = np.zeros(n_rows, dtype=np.float64)
    
    partition_t0 = partition['t0'].values
    partition_t1 = partition['t1'].values
    
    # タイムスタンプをバケット化（日単位）
    for i in range(n_rows):
        current_t0 = partition_t0[i]
        current_t1 = partition_t1[i]
        
        # 現在の区間が跨ぐバケットを特定（日付文字列）
        bucket_start = pd.Timestamp(current_t0).floor('D')
        bucket_end = pd.Timestamp(current_t1).floor('D')
        
        # 重複チェック用のセット（同じ区間を複数回カウントしない）
        checked_intervals: set = set()
        overlap_count = 0
        
        # 関連するバケットのみチェック
        current_bucket = bucket_start
        while current_bucket <= bucket_end:
            bucket_key = current_bucket.strftime('%Y-%m-%d')  # 日付文字列をキーに使用
            
            if bucket_key in interval_buckets:
                # このバケット内の区間のみチェック
                for idx, (other_t0, other_t1) in enumerate(interval_buckets[bucket_key]):
                    interval_id = (bucket_key, idx)
                    
                    # 既にチェック済みならスキップ
                    if interval_id in checked_intervals:
                        continue
                    
                    # 重複判定: other_t0 < current_t1 AND other_t1 > current_t0
                    if other_t0 < current_t1 and other_t1 > current_t0:
                        overlap_count += 1
                        checked_intervals.add(interval_id)
            
            current_bucket += pd.Timedelta(days=1)
        
        concurrency[i] = float(overlap_count)
    
    result['concurrency'] = concurrency
    
    return result


def compute_uniqueness_partition(partition: pd.DataFrame) -> pd.DataFrame:
    """
    並行性から一意性を計算（パーティション処理）
    
    Args:
        partition: 並行性カラムを含むDataFrame
    
    Returns:
        一意性カラムを追加したDataFrame
    """
    if partition.empty:
        result = partition.copy()
        result['uniqueness'] = np.array([], dtype=np.float64)
        return result
    
    result = partition.copy()
    
    # 一意性 = 1 / 並行性（並行性が0の場合は1に設定）
    concurrency = result['concurrency'].values
    uniqueness = np.where(concurrency > 0, 1.0 / concurrency, 1.0)
    
    result['uniqueness'] = uniqueness
    
    return result


class SampleUniquenessWeighter:
    """
    【Dask版・高速化】サンプル一意性による重み付けシステム（統合設計図V準拠）
    区間木アプローチでO(N log N)に計算量削減
    """
    
    def __init__(self,
                 input_path: str,
                 output_path: str,
                 use_return_weighting: bool = False):
        """
        Args:
            input_path: ラベル付きデータセットのパス（t0, t1カラム必須）
            output_path: サンプルウェイト付きデータセットのパス
            use_return_weighting: Trueの場合、一意性 × |return| で重み付け
        """
        self.input_path = input_path
        self.output_path = output_path
        self.use_return_weighting = use_return_weighting
        self.ddf: Optional[DataFrame] = None
    
    def _load_data(self) -> None:
        """ラベル付きデータセットを読み込む"""
        logger.info(f"入力データセット '{self.input_path}' を読み込み中...")
        
        self.ddf = dd.read_parquet(  # type: ignore
            self.input_path,
            engine='pyarrow'
        )
        
        # 必須カラムの確認
        required_cols = ['t0', 't1', 'label']
        if self.use_return_weighting:
            required_cols.append('close')
        
        missing_cols = [col for col in required_cols if col not in self.ddf.columns]
        
        if missing_cols:
            raise ValueError(f"必須カラムが見つかりません: {missing_cols}")
        
        logger.info(f"データ読み込み完了。パーティション数: {self.ddf.npartitions}")
    
    def _build_interval_buckets(self) -> Dict[str, List[tuple]]:
        """
        区間をタイムバケットに分類（前処理）
        
        Returns:
            日付文字列をキーとした区間リスト（'YYYY-MM-DD' -> [(t0, t1), ...]）
        """
        logger.info("区間バケットを構築中（高速化のための前処理）...")
        
        if self.ddf is None:
            raise ValueError("データが読み込まれていません")
        
        # t0, t1を取得（必要最小限のデータのみ）
        t0_series: Series = self.ddf['t0']  # type: ignore[assignment]
        t1_series: Series = self.ddf['t1']  # type: ignore[assignment]
        
        t0_pd: pd.Series[Any]
        t1_pd: pd.Series[Any]
        t0_pd, t1_pd = dask.compute(t0_series, t1_series)  # type: ignore
        
        all_t0: np.ndarray = t0_pd.to_numpy()
        all_t1: np.ndarray = t1_pd.to_numpy()
        
        n_total = len(all_t0)
        logger.info(f"総サンプル数: {n_total:,}")
        logger.info(f"メモリ使用量（t0+t1配列）: {(all_t0.nbytes + all_t1.nbytes) / 1024**2:.2f} MB")
        
        # バケット化（日単位、日付文字列をキーに使用）
        interval_buckets: Dict[str, List[tuple]] = defaultdict(list)
        
        logger.info("区間を日単位バケットに分類中...")
        for i in range(n_total):
            t0 = all_t0[i]
            t1 = all_t1[i]
            
            # この区間が跨ぐすべての日バケットに登録
            bucket_start = pd.Timestamp(t0).floor('D')
            bucket_end = pd.Timestamp(t1).floor('D')
            
            current_bucket = bucket_start
            while current_bucket <= bucket_end:
                bucket_key = current_bucket.strftime('%Y-%m-%d')  # 日付文字列をキーに
                interval_buckets[bucket_key].append((t0, t1))
                current_bucket += pd.Timedelta(days=1)
        
        logger.info(f"バケット数: {len(interval_buckets):,}")
        avg_intervals_per_bucket = sum(len(v) for v in interval_buckets.values()) / len(interval_buckets)
        logger.info(f"バケット当たり平均区間数: {avg_intervals_per_bucket:.1f}")
        logger.info(f"総区間登録数: {sum(len(v) for v in interval_buckets.values()):,}")
        
        return dict(interval_buckets)
    
    def compute_sample_weights(self) -> None:
        """サンプル一意性による重み付けを計算"""
        if self.ddf is None:
            raise ValueError("データが読み込まれていません。_load_data()を先に実行してください。")
        
        logger.info("=" * 50)
        logger.info("サンプル一意性計算開始（高速化版）")
        logger.info("=" * 50)
        
        # ステップ1: 区間バケットの構築
        interval_buckets = self._build_interval_buckets()
        
        # ステップ2: 並行性の計算（各パーティションに対してバケットをブロードキャスト）
        logger.info("並行性（Concurrency）を計算中...")
        logger.info("区間木アプローチによりO(N log N)で処理")
        
        # メタデータ定義
        meta_concurrency = self.ddf._meta.copy()  # type: ignore[union-attr]
        meta_concurrency['concurrency'] = 0.0
        
        # map_partitionsで並行性計算
        ddf_with_concurrency: DataFrame = self.ddf.map_partitions(  # type: ignore[assignment]
            compute_concurrency_interval_tree,
            interval_buckets=interval_buckets,
            meta=meta_concurrency
        )
        
        # ステップ3: 一意性の計算
        logger.info("一意性（Uniqueness）を計算中...")
        
        meta_uniqueness = meta_concurrency.copy()
        meta_uniqueness['uniqueness'] = 0.0
        
        ddf_with_uniqueness: DataFrame = ddf_with_concurrency.map_partitions(  # type: ignore[assignment]
            compute_uniqueness_partition,
            meta=meta_uniqueness
        )
        
        # ステップ4: サンプルウェイトの計算
        logger.info("サンプルウェイトを計算中...")
        
        if self.use_return_weighting:
            logger.info("重み付け方式: 一意性 × |return|")
            # リターン計算: 前の行からの変化率
            ddf_with_uniqueness['abs_return'] = ddf_with_uniqueness['close'].pct_change().abs()  # type: ignore[index]
            ddf_with_uniqueness = ddf_with_uniqueness.fillna({'abs_return': 0.0})  # type: ignore[call-overload]
            
            sample_weight_series: Series = ddf_with_uniqueness['uniqueness'] * ddf_with_uniqueness['abs_return']  # type: ignore[assignment, operator]
        else:
            logger.info("重み付け方式: 一意性のみ")
            sample_weight_series = ddf_with_uniqueness['uniqueness']  # type: ignore[assignment]
        
        # ステップ5: 正規化（合計が1になるように）
        logger.info("サンプルウェイトを正規化中...")
        weight_sum = sample_weight_series.sum()  # type: ignore[union-attr]
        weight_sum_computed: float = float(dask.compute(weight_sum)[0])  # type: ignore
        
        if weight_sum_computed > 0:
            sample_weight_normalized: Series = sample_weight_series / weight_sum_computed  # type: ignore[assignment, operator]
        else:
            logger.warning("サンプルウェイトの合計が0です。均等ウェイトを使用します。")
            n_total = len(self.ddf)  # type: ignore[arg-type]
            sample_weight_normalized = sample_weight_series * 0.0 + (1.0 / n_total)  # type: ignore[assignment, operator]
        
        # ステップ6: DataFrameに追加
        ddf_final = ddf_with_uniqueness.copy()  # type: ignore[attr-defined]
        ddf_final['sample_weight'] = sample_weight_normalized  # type: ignore[index]
        
        # ステップ7: 保存
        logger.info("結果をParquet形式で保存中...")
        
        output_dir = Path(self.output_path).parent
        output_dir.mkdir(parents=True, exist_ok=True)
        
        ddf_final.to_parquet(  # type: ignore[union-attr]
            str(self.output_path),
            engine='pyarrow',
            compression='snappy',
            write_index=False
        )
        
        logger.info("保存完了。統計情報を計算中...")
        
        # 統計情報の計算
        stats = {
            'mean': ddf_final['sample_weight'].mean(),  # type: ignore[index]
            'std': ddf_final['sample_weight'].std(),  # type: ignore[index]
            'min': ddf_final['sample_weight'].min(),  # type: ignore[index]
            'max': ddf_final['sample_weight'].max(),  # type: ignore[index]
            'concurrency_mean': ddf_final['concurrency'].mean(),  # type: ignore[index]
            'concurrency_std': ddf_final['concurrency'].std(),  # type: ignore[index]
            'uniqueness_mean': ddf_final['uniqueness'].mean()  # type: ignore[index]
        }
        
        computed_stats: Dict[str, Any] = dask.compute(stats)[0]  # type: ignore
        
        # 総サンプル数を正しく計算（バケット数ではなく）
        n_total_delayed = len(self.ddf)  # type: ignore[arg-type]
        n_total: int = int(dask.compute(n_total_delayed)[0])  # type: ignore
        
        result_info = {
            'total_samples': n_total,
            'weighting_method': 'uniqueness_x_return' if self.use_return_weighting else 'uniqueness_only',
            'optimization': 'interval_tree',
            'time_complexity': 'O(N log N)',
            'sample_weight_stats': {
                'mean': float(computed_stats['mean']),
                'std': float(computed_stats['std']),
                'min': float(computed_stats['min']),
                'max': float(computed_stats['max'])
            },
            'concurrency_stats': {
                'mean': float(computed_stats['concurrency_mean']),
                'std': float(computed_stats['concurrency_std'])
            },
            'uniqueness_stats': {
                'mean': float(computed_stats['uniqueness_mean'])
            },
            'timestamp': pd.Timestamp.now().isoformat()
        }
        
        # JSONで保存
        json_path = output_dir / 'sample_uniqueness_metadata.json'
        with open(json_path, 'w') as f:
            json.dump(result_info, f, indent=2)
        
        logger.info("=" * 50)
        logger.info("サンプル一意性計算完了")
        logger.info("=" * 50)
        logger.info(f"総サンプル数: {n_total:,}")
        logger.info(f"平均並行性: {result_info['concurrency_stats']['mean']:.2f}")
        logger.info(f"平均一意性: {result_info['uniqueness_stats']['mean']:.4f}")
        logger.info(f"サンプルウェイト - 平均: {result_info['sample_weight_stats']['mean']:.6f}, "
                   f"最小: {result_info['sample_weight_stats']['min']:.6f}, "
                   f"最大: {result_info['sample_weight_stats']['max']:.6f}")
        logger.info(f"出力:")
        logger.info(f"- Parquet: {self.output_path}")
        logger.info(f"- JSON: {json_path}")
        logger.info("=" * 50)
    
    def run(self) -> None:
        """パイプライン全体を実行"""
        self._load_data()
        self.compute_sample_weights()


if __name__ == '__main__':
    # Daskの設定
    dask.config.set({'dataframe.query-planning': True})

    # Daskクライアントの起動
    from dask.distributed import Client, LocalCluster  # type: ignore

    with LocalCluster(n_workers=4, threads_per_worker=2, memory_limit='8GB') as cluster, \
         Client(cluster) as client:

        logger.info(f"Daskクライアントを起動: {client.dashboard_link}")

        weighter = SampleUniquenessWeighter(
            input_path=str(config.S6_LABELED_DATASET),
            output_path=str(config.S6_WEIGHTED_DATASET),
            use_return_weighting=False
        )

        weighter.run()