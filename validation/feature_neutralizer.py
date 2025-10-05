"""
feature_neutralizer.py - v2.1
第三防衛線：アルファの純化

統合設計図V準拠：
- map_partitionsによる並列処理
- 市場ベータの影響除去
- 純粋なアルファ成分の抽出
- Pylance厳格型定義準拠
"""
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))

import blueprint as config
import logging
from typing import List, Dict, Any
from pathlib import Path
import json

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

import dask
import dask.dataframe as dd
from dask.dataframe.core import DataFrame as DaskDataFrame
import joblib
import numpy as np
import pandas as pd


def neutralize_partition(partition: pd.DataFrame, final_team: List[str]) -> pd.DataFrame:
    """
    単一のデータパーティション（Pandas DataFrame）を中立化する関数
    ddf.map_partitionsによって各ワーカで並列実行される
    
    Args:
        partition: データのパーティション
        final_team: 中立化対象の特徴量リスト
    
    Returns:
        中立化された特徴量を含むDataFrame
    """
    from sklearn.linear_model import LinearRegression
    
    if partition.empty:
        out_cols = ['open', 'high', 'low', 'close', 'volume'] + [
            f"{f}_neutralized" for f in final_team
        ]
        return pd.DataFrame(columns=out_cols)
    
    X_market = partition[['market_proxy']].values
    
    result_df = partition[['open', 'high', 'low', 'close', 'volume']].copy()
    
    for feature in final_team:
        if feature not in partition.columns:
            continue
        
        y_feature = partition[feature].values.reshape(-1, 1)
        
        model = LinearRegression(fit_intercept=True)
        model.fit(X_market, y_feature)
        
        predicted_by_market = model.predict(X_market)
        
        result_df[f"{feature}_neutralized"] = (y_feature - predicted_by_market).ravel()
    
    return result_df


class FeatureNeutralizer:
    """
    【Dask版 v2.1 - Out-of-Core処理対応】
    TB級のマスターテーブルに対して、市場ベータの影響を除去し、
    純粋なアルファ成分を抽出する。全ての処理をDaskの遅延実行で行う。
    """
    
    def __init__(
        self,
        feature_universe_path: str,
        final_team_path: str,
        output_path: str
    ):
        self.feature_universe_path = feature_universe_path
        self.final_team_path = final_team_path
        self.output_path = output_path
        self.ddf: DaskDataFrame | None = None
        self.final_team: List[str] | None = None
    
    def _load_data(self) -> None:
        """必要なデータをDask DataFrameとして読み込む"""
        logger.info(f"マスターテーブル '{self.feature_universe_path}' をDask DataFrameとして読み込み中...")
        
        self.ddf = dd.read_parquet(
            self.feature_universe_path,
            engine='pyarrow'
        )
        
        logger.info(f"最終選抜チーム '{self.final_team_path}' を読み込み中...")
        self.final_team = joblib.load(self.final_team_path)
        
        if self.ddf is None or self.final_team is None:
            raise ValueError("Data loading failed.")
        
        available_features = [f for f in self.final_team if f in self.ddf.columns]
        if len(available_features) != len(self.final_team):
            removed = len(self.final_team) - len(available_features)
            logger.warning(f"{removed}個の特徴量がデータセットに存在しないため除外されました。")
            self.final_team = available_features
        
        logger.info(f"読み込み完了。{len(self.final_team)}個の特徴量を処理します。")
    
    def _define_market_proxy(self, window: int = 240) -> None:
        """
        市場全体の動きを代表する指標（プロキシ）を定義する
        Dask APIを使用して遅延実行のまま計算
        
        Args:
            window: 移動平均のウィンドウサイズ（分）
        """
        logger.info(f"市場トレンドの代理変数（{window}分移動平均線）を計算中...")
        
        if self.ddf is None:
            raise ValueError("DataFrame not loaded. Run _load_data() first.")
        
        self.ddf = self.ddf.assign(
            market_proxy=self.ddf['close'].rolling(window=window).mean()
        )
        
        logger.info("市場プロキシの計算完了（遅延実行として登録）。")
    
    def neutralize_features(self) -> None:
        """
        【v2.1】特徴量の純化
        ddf.map_partitionsを使い、各パーティションを並列処理することで、
        計算グラフの肥大化を防ぎ、メモリ効率と実行速度を向上させる。
        """
        if self.ddf is None or self.final_team is None:
            raise ValueError("Data not loaded. Run _load_data() first.")
        
        ddf = self.ddf
        final_team = self.final_team
        
        logger.info(f"--- 第三防衛線: 全{len(final_team)}個の最終特徴量の純化を開始 (map_partitionsモード) ---")
        
        cols_to_keep = ['open', 'high', 'low', 'close', 'volume', 'market_proxy'] + final_team
        ddf = ddf[cols_to_keep].dropna()
        
        meta_dict: Dict[str, str] = {
            'open': 'f8',
            'high': 'f8',
            'low': 'f8',
            'close': 'f8',
            'volume': 'i8'
        }
        for feature in final_team:
            meta_dict[f"{feature}_neutralized"] = 'f8'
        
        neutralized_ddf = ddf.map_partitions(
            neutralize_partition,
            final_team=final_team,
            meta=meta_dict
        )
        
        logger.info("特徴量の純化完了。最終ファイルをパーティション化されたParquet形式で保存しています...")
        
        output_dir = Path(self.output_path).parent
        output_dir.mkdir(parents=True, exist_ok=True)
        
        output_parquet = Path(self.output_path)
        logger.info(f"パーティション化されたParquetとして保存中: {output_parquet}")
        
        neutralized_ddf.to_parquet(
            str(output_parquet),
            engine='pyarrow',
            compression='snappy',
            write_index=False
        )
        
        logger.info("メタデータを計算中...")
        n_rows, n_cols = dask.compute(len(neutralized_ddf), len(neutralized_ddf.columns))
        
        result_info: Dict[str, Any] = {
            'total_rows': n_rows,
            'total_columns': n_cols,
            'feature_columns': [
                col for col in neutralized_ddf.columns
                if col not in ['open', 'high', 'low', 'close', 'volume']
            ],
            'validation_type': 'third_defense_line',
            'timestamp': pd.Timestamp.now().isoformat()
        }
        
        json_path = output_dir / 'neutralized_feature_set.json'
        with open(json_path, 'w') as f:
            json.dump(result_info, f, indent=2)
        
        logger.info("-" * 50)
        logger.info(f"🎉 第三防衛線 完了: 純化された特徴量をパーティション化Parquet形式で保存しました。🎉")
        logger.info(f"データセット列数: {n_cols}")
        logger.info(f"- Parquet: {output_parquet}")
        logger.info(f"- JSON: {json_path}")
        logger.info("-" * 50)
    
    def run(self) -> None:
        """パイプライン全体を実行する"""
        self._load_data()
        self._define_market_proxy()
        self.neutralize_features()


if __name__ == '__main__':
    dask.config.set({'dataframe.query-planning': True})

    neutralizer = FeatureNeutralizer(
        feature_universe_path=str(config.S4_MASTER_TABLE_PARTITIONED),
        final_team_path=str(config.S3_FINAL_FEATURE_TEAM),
        output_path=str(config.S5_NEUTRALIZED_ALPHA_SET)
    )
    neutralizer.run()