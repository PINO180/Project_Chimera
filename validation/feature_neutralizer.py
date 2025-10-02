import dask
import dask.dataframe as dd
import dask.array as da
import joblib
from dask_ml.linear_model import LinearRegression  # type: ignore
from tqdm import tqdm
import logging
from dask.dataframe.core import DataFrame
import pandas as pd

# 標準のロギング設定
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

import numpy as np
from pathlib import Path
import json

def neutralize_partition(partition, final_team):
    """
    単一のデータパーティション（Pandas DataFrame）を中立化する関数
    ddf.map_partitionsによって各ワーカで並列実行される
    """
    # map_partitions内で必要なライブラリをインポートするのが安全
    from sklearn.linear_model import LinearRegression
    import pandas as pd
    
    # パーティションが空の場合はそのまま返す
    if partition.empty:
        # スキーマを維持するため、空のDataFrameを生成
        out_cols = ['open', 'high', 'low', 'close', 'volume'] + [f"{f}_neutralized" for f in final_team]
        return pd.DataFrame(columns=out_cols)

    # 説明変数（市場プロキシ）
    X_market = partition[['market_proxy']].values
    
    # 結果を格納するDataFrameを準備
    result_df = partition[['open', 'high', 'low', 'close', 'volume']].copy()
    
    # 各特徴量をループ処理
    for feature in final_team:
        if feature not in partition.columns:
            continue
            
        y_feature = partition[feature].values.reshape(-1, 1)
        
        # 通常の（高速な）scikit-learnのLinearRegressionを使用
        model = LinearRegression(fit_intercept=True)
        model.fit(X_market, y_feature)
        
        predicted_by_market = model.predict(X_market)
        
        # 残差を計算し、結果のDataFrameに追加
        result_df[f"{feature}_neutralized"] = (y_feature - predicted_by_market).ravel()
        
    return result_df

class FeatureNeutralizer:
    """
    【Dask版 v2.0 - Out-of-Core処理対応】
    TB級のマスターテーブルに対して、市場ベータの影響を除去し、
    純粋なアルファ成分を抽出する。全ての処理をDaskの遅延実行で行う。
    """
    def __init__(self,
                 feature_universe_path: str,
                 final_team_path: str,
                 output_path: str):
        self.feature_universe_path = feature_universe_path
        self.final_team_path = final_team_path
        self.output_path = output_path
        self.ddf: DataFrame | None = None
        self.final_team: list[str] | None = None

    def _load_data(self):
        """必要なデータをDask DataFrameとして読み込む"""
        logger.info(f"マスターテーブル '{self.feature_universe_path}' をDask DataFrameとして読み込み中...")
        
        # Daskで遅延読み込み
        self.ddf = dd.read_parquet(  # type: ignore
            self.feature_universe_path,
            engine='pyarrow'
        )
        
        logger.info(f"最終選抜チーム '{self.final_team_path}' を読み込み中...")
        self.final_team = joblib.load(self.final_team_path)
        
        # 最終チームの特徴量がデータセットに存在するか確認
        available_features = [f for f in self.final_team if f in self.ddf.columns]
        if len(available_features) != len(self.final_team):
            removed = len(self.final_team) - len(available_features)
            logger.warning(f"{removed}個の特徴量がデータセットに存在しないため除外されました。")
            self.final_team = available_features
        
        logger.info(f"読み込み完了。{len(self.final_team)}個の特徴量を処理します。")

    def _define_market_proxy(self, window: int = 240):
        """
        市場全体の動きを代表する指標（プロキシ）を定義する
        Dask APIを使用して遅延実行のまま計算
        """
        logger.info(f"市場トレンドの代理変数（{window}分移動平均線）を計算中...")
        
        if self.ddf is None:
            raise ValueError("DataFrame not loaded. Run _load_data() first.")

        # Daskのrolling関数を使用（遅延実行）
        self.ddf = self.ddf.assign(
            market_proxy=self.ddf['close'].rolling(window=window).mean()
        )
        
        logger.info("市場プロキシの計算完了（遅延実行として登録）。")

    def neutralize_features(self):
        """
        【改善版 v2.1】
        ddf.map_partitionsを使い、各パーティションを並列処理することで、
        計算グラフの肥大化を防ぎ、メモリ効率と実行速度を向上させる。
        """
        if self.ddf is None or self.final_team is None:
            raise ValueError("Data not loaded. Run _load_data() first.")

        # Pylanceの型推論を助けるため、ローカル変数に割り当てる
        ddf = self.ddf
        final_team = self.final_team

        logger.info(f"--- 第三防衛線: 全{len(final_team)}個の最終特徴量の純化を開始 (map_partitionsモード) ---")
        
        # 処理に必要な列のみに絞り、NaNを除去
        cols_to_keep = ['open', 'high', 'low', 'close', 'volume', 'market_proxy'] + final_team
        ddf = ddf[cols_to_keep].dropna()
        
        # Daskに出力後のDataFrameの構造（スキーマ）を教えるためのmetaを定義
        meta_dict = {
            'open': 'f8', 'high': 'f8', 'low': 'f8', 'close': 'f8', 'volume': 'i8'
        }
        for feature in final_team:
            meta_dict[f"{feature}_neutralized"] = 'f8'
        
        # map_partitionsを呼び出し、各パーティションでneutralize_partition関数を並列実行
        neutralized_ddf = ddf.map_partitions(
            neutralize_partition,
            final_team=final_team,
            meta=meta_dict
        )
        
        logger.info("特徴量の純化完了。最終ファイルをパーティション化されたParquet形式で保存しています...")
        
        # 出力ディレクトリの作成
        output_dir = Path(self.output_path).parent
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Parquet形式で保存（この瞬間に全ての遅延計算が実行される）
        output_parquet = Path(self.output_path)
        logger.info(f"パーティション化されたParquetとして保存中: {output_parquet}")
        
        neutralized_ddf.to_parquet(
            str(output_parquet),
            engine='pyarrow',
            compression='snappy',
            write_index=False
        )
        
        # メタデータの保存（JSON形式）
        logger.info("メタデータを計算中...")
        n_rows, n_cols = dask.compute(len(neutralized_ddf), len(neutralized_ddf.columns))  # type: ignore
        
        result_info = {
            'total_rows': n_rows,
            'total_columns': n_cols,
            'feature_columns': [col for col in neutralized_ddf.columns 
                               if col not in ['open', 'high', 'low', 'close', 'volume']],
            'validation_type': 'third_defense_line',
            'timestamp': pd.Timestamp.now().isoformat() # 【バグ修正】DaskではなくPandasで現在時刻を取得
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

    def run(self):
        """パイプライン全体を実行する"""
        self._load_data()
        self._define_market_proxy()
        self.neutralize_features()


if __name__ == '__main__':
    # Daskの設定
    dask.config.set({'dataframe.query-planning': True})
    
    # 出力ディレクトリの準備
    output_base = Path('data/temp_chunks/defense_results')
    output_base.mkdir(parents=True, exist_ok=True)
    
    neutralizer = FeatureNeutralizer(
        feature_universe_path='data/master_table_partitioned',
        final_team_path='data/temp_chunks/defense_results/joblib/final_feature_team.joblib',
        output_path='data/temp_chunks/defense_results/neutralized_feature_set_partitioned'
    )
    neutralizer.run()