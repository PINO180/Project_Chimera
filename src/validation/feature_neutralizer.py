import pandas as pd
import joblib
from sklearn.linear_model import LinearRegression
from tqdm import tqdm
from common.logger_setup import logger
import numpy as np

class FeatureNeutralizer:
    """
    【第三防衛線 v1.0】
    選抜された特徴量チームから、市場全体の動き（マーケットβ）の影響を除去し、
    純粋なアルファ成分を抽出する。
    """
    def __init__(self,
                 feature_universe_path: str,
                 final_team_path: str,
                 output_path: str):
        self.feature_universe_path = feature_universe_path
        self.final_team_path = final_team_path
        self.output_path = output_path
        self.df_universe = None
        self.final_team = None

    def _load_data(self):
        """必要なデータをメモリに読み込む"""
        logger.info(f"特徴量ユニバース '{self.feature_universe_path}' を読み込み中...")
        
        # パスに応じて適切な読み込み方法を選択
        if self.feature_universe_path.endswith('.parquet'):
            self.df_universe = pd.read_parquet(self.feature_universe_path)
        else:
            self.df_universe = joblib.load(self.feature_universe_path)
        
        logger.info(f"最終選抜チーム '{self.final_team_path}' を読み込み中...")
        self.final_team = joblib.load(self.final_team_path)
        
        logger.info("読み込み完了。")

    def _define_market_proxy(self, window: int = 240):
        """市場全体の動きを代表する指標（プロキシ）を定義する"""
        logger.info(f"市場トレンドの代理変数（{window}分移動平均線）を計算中...")
        # 1分足データなので、4時間(240分)の移動平均を市場トレンドとする
        self.df_universe['market_proxy'] = self.df_universe['close'].rolling(window=window).mean()

    def neutralize_features(self):
        """
        各特徴量を市場プロキシに対して回帰させ、その残差を中立化された特徴量とする。
        """
        logger.info(f"--- 第三防衛線: 全{len(self.final_team)}個の最終特徴量の純化を開始 ---")
        
        # 中立化された特徴量を格納する新しいDataFrameを準備
        neutralized_features_df = pd.DataFrame(index=self.df_universe.index)
        
        # 処理対象のデータ（市場プロキシと最終チームの特徴量）を準備
        # これによりNaNを含む期間が適切に処理される
        cols_to_process = self.final_team + ['market_proxy']
        processing_df = self.df_universe[cols_to_process].dropna()

        X_market = processing_df[['market_proxy']] # 回帰分析のため2D配列にする

        for feature in tqdm(self.final_team, desc="Neutralizing Features"):
            y_feature = processing_df[feature]
            
            # 線形回帰モデルを作成し、学習
            model = LinearRegression()
            model.fit(X_market, y_feature)
            
            # 市場トレンドから予測される値を計算
            predicted_by_market = model.predict(X_market)
            
            # 元の特徴量から市場トレンドの影響を差し引く（＝残差）
            residuals = y_feature - predicted_by_market
            
            # 結果を格納
            neutralized_features_df[f"{feature}_neutralized"] = residuals
        
        # 元のOHLCVデータと結合して最終的なデータセットを作成
        final_df = pd.concat([self.df_universe[['open', 'high', 'low', 'close', 'volume']], neutralized_features_df], axis=1)
        
        logger.info("特徴量の純化完了。最終ファイルを複数形式で保存しています...")
        
        # JOBLIB形式で保存
        joblib.dump(final_df, 'data/temp_chunks/defense_results/joblib/neutralized_feature_set.joblib', compress=('gzip', 3))
        
        # CSV形式で保存（人間確認用）
        final_df.to_csv('data/temp_chunks/defense_results/csv/neutralized_feature_set.csv', index=False)
        
        # JSON形式で保存（メタデータのみ）
        import json
        result_info = {
            'total_rows': len(final_df),
            'total_columns': len(final_df.columns),
            'feature_columns': [col for col in final_df.columns if col not in ['open', 'high', 'low', 'close', 'volume']],
            'validation_type': 'third_defense_line',
            'timestamp': pd.Timestamp.now().isoformat()
        }
        with open('data/temp_chunks/defense_results/json/neutralized_feature_set.json', 'w') as f:
            json.dump(result_info, f, indent=2)
        
        logger.info("-" * 50)
        logger.info(f"🎉 第三防衛線 完了: 純化された特徴量を複数形式で保存しました。🎉")
        logger.info(f"最終データセットの形状: {final_df.shape}")
        logger.info("- JOBLIB: data/temp_chunks/defense_results/joblib/neutralized_feature_set.joblib")
        logger.info("- CSV: data/temp_chunks/defense_results/csv/neutralized_feature_set.csv")
        logger.info("- JSON: data/temp_chunks/defense_results/json/neutralized_feature_set.json")
        logger.info(f"最終データセットの形状: {final_df.shape}")
        logger.info("-" * 50)

    def run(self):
        """パイプライン全体を実行する"""
        self._load_data()
        self._define_market_proxy()
        self.neutralize_features()

if __name__ == '__main__':
    neutralizer = FeatureNeutralizer(
        feature_universe_path='data/temp_chunks/feature_chunk_0.parquet',
        final_team_path='data/final_feature_team.joblib',
        output_path='data/neutralized_feature_set.joblib'
    )
    neutralizer.run()