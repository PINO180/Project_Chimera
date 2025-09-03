# tool_1_recover_feature.py

"""
エラー復旧ツール ステップ1：欠落特徴量の再生産

このスクリプトは、'independent_features.py'の実行中にエラーで失敗した
特定の関数のみを再実行し、欠落した特徴量（パッチ）を生成します。

#################################################################
# 使い方 (How to Use)
#################################################################
#
# 1. 下記の「--- 設定 ---」セクションを編集します。
#    - input_path: 計算の元となるベースファイル（通常は '01_mfdfa_...'）を指定。
#    - output_path: 生成されるパッチファイルの名前を、復旧する内容が分かるように指定。
#
# 2. 「失敗した関数だけを直接呼び出す」セクションを編集します。
#    - calculator._calculate_... の部分を、実際に失敗した関数の名前に変更します。
#
# 3. PowerShellで実行します。
#    python tool_1_recover_feature.py
#
#################################################################
"""

import pandas as pd
import logging
from pathlib import Path
from features.independent_features import AdvancedIndependentFeatures 

# ロガー設定
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def recover_missing_features():
    """
    失敗したタスクのみを再実行して、欠落した特徴量（パッチ）を生成する
    """
    try:
        # --- 設定 ---
        # 計算の元となる、ベースの特徴量ファイル
        input_path = Path(r"C:\project_forge\data\temp_chunks\parquet\01_mfdfa_45features.parquet")
        # 生成した欠落特徴量（パッチ）の出力先
        output_path = Path(r"C:\project_forge\data\temp_chunks\parquet\recovery_moving_averages_advanced.parquet")
        
        logging.info("=== 欠落特徴量の個別収集開始 ===")
        
        # 1. ベースとなる入力データを読み込み
        logging.info(f"入力ファイルを読み込み中: {input_path}")
        df = pd.read_parquet(input_path)
        
        # 2. 特徴量計算クラスをインスタンス化
        calculator = AdvancedIndependentFeatures()
        
        # 3. 失敗した関数だけを直接呼び出す
        logging.info("'moving_averages_advanced' の計算を開始...")
        failed_features_dict = calculator._calculate_moving_averages_advanced(df)
        
        # 4. 結果をDataFrameに変換
        failed_features_df = pd.DataFrame(failed_features_dict, index=df.index)
        logging.info(f"計算完了。{failed_features_df.shape[1]}個の特徴量を生成しました。")
        
        # 5. パッチファイルをParquet形式で保存
        logging.info(f"パッチファイルを保存中: {output_path}")
        failed_features_df.to_parquet(output_path, compression='snappy')
        
        logging.info(f"=== 個別収集完了！パッチファイルが作成されました: {output_path} ===")
        
    except Exception as e:
        logging.error(f"処理中にエラーが発生しました: {e}")
        raise

if __name__ == "__main__":
    recover_missing_features()