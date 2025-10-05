import config
import dask
import dask.dataframe as dd
from dask.dataframe.core import DataFrame
import numpy as np
import pandas as pd
from pathlib import Path
import json
import logging
from typing import Optional, Dict, Any

# ロギング設定
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def label_partition(partition: pd.DataFrame, profit_multiplier: float, 
                    loss_multiplier: float, time_barrier: int) -> pd.DataFrame:
    """
    単一パーティション内でトリプルバリアラベリングを実行する関数
    
    Args:
        partition: パーティションデータ（Pandas DataFrame）
        profit_multiplier: 利食いバリアの倍率（ATRに対する）
        loss_multiplier: 損切りバリアの倍率（ATRに対する）
        time_barrier: 時間バリア（バー数）
    
    Returns:
        ラベル付きDataFrame (t0, t1カラムを含む)
    """
    if partition.empty:
        # 空のパーティションの場合、スキーマを維持
        result_cols = list(partition.columns) + ['label', 'barrier_reached', 'time_to_barrier', 't0', 't1']
        return pd.DataFrame(columns=result_cols)
    
    # 必要なカラムの存在確認
    required_cols = ['close', 'ATR', 'timestamp']
    for col in required_cols:
        if col not in partition.columns:
            logger.error(f"必須カラム '{col}' がパーティションに存在しません。")
            return pd.DataFrame()
    
    result_df = partition.copy()
    n_rows = len(partition)
    
    # 結果カラムを初期化
    labels = np.zeros(n_rows, dtype=np.int32)
    barriers = np.empty(n_rows, dtype=object)
    times = np.zeros(n_rows, dtype=np.int32)
    t0_array = partition['timestamp'].values.copy()
    t1_array = np.empty(n_rows, dtype='datetime64[ns]')
    
    close_values = partition['close'].values
    atr_values = partition['ATR'].values
    timestamp_values = partition['timestamp'].values
    
    for i in range(n_rows):
        current_price = close_values[i]
        current_atr = atr_values[i]
        current_timestamp = timestamp_values[i]
        
        # ATRが無効な場合はスキップ
        if pd.isna(current_atr) or current_atr <= 0:
            labels[i] = 0
            barriers[i] = 'none'
            times[i] = 0
            t1_array[i] = current_timestamp
            continue
        
        # バリアを計算
        profit_barrier = current_price + profit_multiplier * current_atr
        loss_barrier = current_price - loss_multiplier * current_atr
        
        # 将来の価格を確認
        max_look_ahead = min(time_barrier, n_rows - i - 1)
        
        if max_look_ahead <= 0:
            # 将来のデータがない場合
            labels[i] = 0
            barriers[i] = 'time'
            times[i] = 0
            t1_array[i] = current_timestamp
            continue
        
        future_prices = close_values[i+1:i+1+max_look_ahead]
        
        # 各バリアに到達するかチェック
        profit_hit = future_prices >= profit_barrier
        loss_hit = future_prices <= loss_barrier
        
        profit_hit_idx = np.where(profit_hit)[0]
        loss_hit_idx = np.where(loss_hit)[0]
        
        # 最初に到達したバリアを判定
        first_profit = profit_hit_idx[0] + 1 if len(profit_hit_idx) > 0 else float('inf')
        first_loss = loss_hit_idx[0] + 1 if len(loss_hit_idx) > 0 else float('inf')
        
        if first_profit < first_loss:
            # 利食いバリアに先に到達
            labels[i] = 1
            barriers[i] = 'profit'
            times[i] = int(first_profit)
            hit_idx = i + int(first_profit)
            t1_array[i] = timestamp_values[hit_idx] if hit_idx < n_rows else timestamp_values[n_rows - 1]
        elif first_loss < first_profit:
            # 損切りバリアに先に到達
            labels[i] = -1
            barriers[i] = 'loss'
            times[i] = int(first_loss)
            hit_idx = i + int(first_loss)
            t1_array[i] = timestamp_values[hit_idx] if hit_idx < n_rows else timestamp_values[n_rows - 1]
        else:
            # どちらにも到達せず（時間切れ）
            labels[i] = 0
            barriers[i] = 'time'
            times[i] = max_look_ahead
            end_idx = i + max_look_ahead
            t1_array[i] = timestamp_values[end_idx] if end_idx < n_rows else timestamp_values[n_rows - 1]
    
    # 結果をDataFrameに追加
    result_df['label'] = labels
    result_df['barrier_reached'] = barriers
    result_df['time_to_barrier'] = times
    result_df['t0'] = t0_array  # イベント開始時刻
    result_df['t1'] = t1_array  # イベント終了時刻
    
    return result_df


class TripleBarrierLabeler:
    """
    【Dask版】トリプルバリアラベリングシステム（統合設計図V準拠）
    純化された特徴量データセットに対して現実的なターゲット変数を生成
    """
    
    def __init__(self,
                 input_path: str,
                 output_path: str,
                 profit_multiplier: float = 2.0,
                 loss_multiplier: float = 1.0,
                 time_barrier: int = 60):
        """
        Args:
            input_path: 入力データセットのパス（Parquet）
            output_path: 出力データセットのパス
            profit_multiplier: 利食いバリアの倍率（ATRに対する）
            loss_multiplier: 損切りバリアの倍率（ATRに対する）
            time_barrier: 時間バリア（バー数）
        """
        self.input_path = input_path
        self.output_path = output_path
        self.profit_multiplier = profit_multiplier
        self.loss_multiplier = loss_multiplier
        self.time_barrier = time_barrier
        self.ddf: Optional[DataFrame] = None
    
    def _load_data(self) -> None:
        """純化された特徴量データセットをDask DataFrameとして読み込む"""
        logger.info(f"入力データセット '{self.input_path}' を読み込み中...")
        
        self.ddf = dd.read_parquet(  # type: ignore
            self.input_path,
            engine='pyarrow'
        )
        
        # 必須カラムの確認
        required_cols = ['close', 'ATR', 'timestamp']
        missing_cols = [col for col in required_cols if col not in self.ddf.columns]
        
        if missing_cols:
            raise ValueError(f"必須カラムが見つかりません: {missing_cols}")
        
        logger.info(f"データ読み込み完了。パーティション数: {self.ddf.npartitions}")
    
    def apply_triple_barrier_labeling(self) -> None:
        """トリプルバリアラベリングを実行"""
        if self.ddf is None:
            raise ValueError("データが読み込まれていません。_load_data()を先に実行してください。")
        
        logger.info("=" * 50)
        logger.info("トリプルバリアラベリング開始")
        logger.info("=" * 50)
        logger.info(f"設定: 利食い={self.profit_multiplier}xATR, 損切り={self.loss_multiplier}xATR, 時間={self.time_barrier}バー")
        
        # 出力スキーマを定義
        meta_dict = {col: self.ddf[col].dtype for col in self.ddf.columns}
        meta_dict['label'] = 'int32'
        meta_dict['barrier_reached'] = 'object'
        meta_dict['time_to_barrier'] = 'int32'
        meta_dict['t0'] = 'datetime64[ns]'
        meta_dict['t1'] = 'datetime64[ns]'
        
        # 【重要】map_overlapを使用してパーティション境界問題を解決
        logger.info(f"map_overlapを使用（オーバーラップ: {self.time_barrier}バー先）")
        labeled_ddf = self.ddf.map_overlap(  # type: ignore[assignment]
            label_partition,
            before=0,
            after=self.time_barrier,
            profit_multiplier=self.profit_multiplier,
            loss_multiplier=self.loss_multiplier,
            time_barrier=self.time_barrier,
            meta=meta_dict
        )
        
        logger.info("ラベリング完了。結果をParquet形式で保存中...")
        
        # 出力ディレクトリの作成
        output_dir = Path(self.output_path).parent
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Parquet形式で保存
        labeled_ddf.to_parquet(
            str(self.output_path),
            engine='pyarrow',
            compression='snappy',
            write_index=False
        )
        
        logger.info("保存完了。メタデータを計算中...")
        
        # メタデータの計算
        label_counts_series = labeled_ddf['label'].value_counts()  # type: ignore[index]
        n_rows_delayed: Any = len(labeled_ddf)  # type: ignore[arg-type]
        columns_list = list(labeled_ddf.columns)  # type: ignore[attr-defined]
        n_cols: int = len(columns_list)
        
        # 一度に計算を実行
        computed_results = dask.compute(label_counts_series, n_rows_delayed)  # type: ignore
        label_counts_series_computed: pd.Series = computed_results[0]  # type: ignore
        n_rows: int = int(computed_results[1])
        
        # 辞書に変換
        label_counts: Dict[int, int] = label_counts_series_computed.to_dict()  # type: ignore
        
        # 統計情報
        result_info = {
            'total_rows': n_rows,
            'total_columns': n_cols,
            'label_distribution': {
                'profit': int(label_counts.get(1, 0)),
                'loss': int(label_counts.get(-1, 0)),
                'time': int(label_counts.get(0, 0))
            },
            'label_distribution_percent': {
                'profit': float(label_counts.get(1, 0) / n_rows * 100) if n_rows > 0 else 0.0,
                'loss': float(label_counts.get(-1, 0) / n_rows * 100) if n_rows > 0 else 0.0,
                'time': float(label_counts.get(0, 0) / n_rows * 100) if n_rows > 0 else 0.0
            },
            'settings': {
                'profit_multiplier': self.profit_multiplier,
                'loss_multiplier': self.loss_multiplier,
                'time_barrier': self.time_barrier
            },
            'timestamp': pd.Timestamp.now().isoformat()
        }
        
        # JSONで保存
        json_path = output_dir / 'triple_barrier_metadata.json'
        with open(json_path, 'w') as f:
            json.dump(result_info, f, indent=2)
        
        logger.info("=" * 50)
        logger.info("🎉 トリプルバリアラベリング完了 🎉")
        logger.info("=" * 50)
        logger.info(f"総行数: {n_rows:,}")
        logger.info(f"ラベル分布: 利食い={result_info['label_distribution']['profit']:,} ({result_info['label_distribution_percent']['profit']:.2f}%), "
                   f"損切り={result_info['label_distribution']['loss']:,} ({result_info['label_distribution_percent']['loss']:.2f}%), "
                   f"時間切れ={result_info['label_distribution']['time']:,} ({result_info['label_distribution_percent']['time']:.2f}%)")
        logger.info(f"出力:")
        logger.info(f"- Parquet: {self.output_path}")
        logger.info(f"- JSON: {json_path}")
        logger.info("=" * 50)
    
    def run(self) -> None:
        """パイプライン全体を実行"""
        self._load_data()
        self.apply_triple_barrier_labeling()


if __name__ == '__main__':
    # Daskの設定
    dask.config.set({'dataframe.query-planning': True})

    # Daskクライアントの起動
    from dask.distributed import Client, LocalCluster  # type: ignore

    with LocalCluster(n_workers=4, threads_per_worker=2, memory_limit='8GB') as cluster, \
         Client(cluster) as client:

        logger.info(f"Daskクライアントを起動: {client.dashboard_link}")

        labeler = TripleBarrierLabeler(
            input_path=str(config.S5_NEUTRALIZED_ALPHA_SET),
            output_path=str(config.S6_LABELED_DATASET),
            profit_multiplier=2.0,
            loss_multiplier=1.0,
            time_barrier=60
        )

        labeler.run()