# tool_3_combine_generic.py

"""
汎用Parquetファイル結合ツール (複数ファイル対応版)

このスクリプトは、コマンドラインから指定された複数のParquetファイルを
読み込み、横方向（axis=1）に結合し、新しいParquetファイルとして保存します。
結合時に重複した列名があった場合は、最初の列を残して自動的に除去します。

#################################################################
# 使い方 (How to Use)
#################################################################
#
# PowerShellやコマンドプロンプトから以下のように実行します。
#
# ■ 基本的な構文:
# python tool_3_combine_generic.py [入力ファイル1] [入力ファイル2] ... [出力ファイル]
#
# ■ 具体的な使用例（3つのファイルを結合する場合）:
# python tool_3_combine_generic.py C:\data\file_A.parquet C:\data\file_B.parquet C:\data\file_C.parquet C:\data\combined_ABC.parquet
#
#################################################################
"""

import pandas as pd
import logging
from pathlib import Path
import sys

# ロガー設定
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def combine_multiple_files(input_paths: list, output_path: str):
    """
    複数のParquetファイルを読み込み、横方向に結合して保存する汎用関数
    """
    try:
        logging.info("=== 汎用データセット結合（複数ファイル対応版）開始 ===")
        
        list_of_dfs = []
        for path in input_paths:
            logging.info(f"ファイルを読み込み中: {path}")
            df = pd.read_parquet(path)
            list_of_dfs.append(df)
        
        logging.info(f"{len(list_of_dfs)}個のファイルを結合中...")
        combined_df = pd.concat(list_of_dfs, axis=1)
        
        is_duplicate = combined_df.columns.duplicated(keep=False)
        if is_duplicate.any():
            duplicate_cols = combined_df.columns[is_duplicate].tolist()
            logging.warning(f"警告: 重複列が検出されました -> {duplicate_cols}")
            combined_df = combined_df.loc[:, ~combined_df.columns.duplicated(keep='first')]
            logging.info("重複列を除去しました。")
        else:
            logging.info("重複列はありませんでした。")

        logging.info(f"結合後のデータ Shape: {combined_df.shape}")
        
        logging.info(f"結合結果を保存中: {output_path}")
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        combined_df.to_parquet(output_path, compression='snappy')
        
        logging.info(f"=== 結合完了！ファイルが作成されました: {output_path} ===")

    except FileNotFoundError:
        logging.error("エラー: 指定されたファイルが見つかりません。パスを確認してください。")
    except Exception as e:
        logging.error(f"処理中にエラーが発生しました: {e}")
        raise

if __name__ == "__main__":
    if len(sys.argv) < 4:
        print("\n使い方: python tool_3_combine_generic.py [入力ファイル1] [入力ファイル2] ... [出力ファイル]")
        print(r"例: python tool_3_combine_generic.py file_A.parquet file_B.parquet file_C.parquet combined_ABC.parquet")
    else:
        input_files = sys.argv[1:-1]
        output_file = sys.argv[-1]
        combine_multiple_files(input_files, output_file)