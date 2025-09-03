import pandas as pd
import os
from common.logger_setup import logger

# --- 設定 ---
SOURCE_FILE_NAME = 'export.csv' 
OUTPUT_FILE_PATH = 'data/XAUUSD_1m.csv'
# ----------------

def format_mt5_data():
    """
    MT5からエクスポートされた巨大なCSVファイルを読み込み、
    プロジェクトで使用可能な形式に整形して保存する。
    """
    logger.info("--- MT5データ整形処理を開始します (修正版 v2) ---")

    if not os.path.exists(SOURCE_FILE_NAME):
        logger.error(f"エラー: プロジェクトルートに '{SOURCE_FILE_NAME}' が見つかりません。")
        logger.error("MT5からエクスポートしたCSVファイルをプロジェクトフォルダ直下に配置してください。")
        return

    try:
        logger.info(f"'{SOURCE_FILE_NAME}' を読み込んでいます...")
        
        # 修正点: header=0 を指定し、ファイルの1行目をヘッダーとして正しく読み込む
        df = pd.read_csv(
            SOURCE_FILE_NAME,
            sep='\t',  # MT5のエクスポートはタブ区切り
            header=0   # 1行目をヘッダーとして使用
        )
        logger.info(f"ファイルの読み込み完了。合計 {len(df)} 行のデータを検出。")
        
        # 期待される列名が存在するかチェック
        expected_cols = ['<DATE>', '<TIME>', '<OPEN>', '<HIGH>', '<LOW>', '<CLOSE>', '<TICKVOL>']
        if not all(col in df.columns for col in expected_cols):
            logger.error("エラー: ファイルの列名が期待される形式と異なります。")
            logger.error(f"期待される列名: {expected_cols}")
            logger.error(f"実際の列名: {df.columns.tolist()}")
            return

        logger.info("列の整形を開始...")
        
        # 1. datetime列を作成
        #    ここでエラーが発生していたが、ヘッダーが正しく読み込まれたため解決される
        df['datetime'] = pd.to_datetime(df['<DATE>'] + ' ' + df['<TIME>'])
        
        # 2. 最終的なDataFrameを作成し、列名を小文字に統一
        df_final = pd.DataFrame({
            'datetime': df['datetime'],
            'open': df['<OPEN>'],
            'high': df['<HIGH>'],
            'low': df['<LOW>'],
            'close': df['<CLOSE>'],
            'volume': df['<TICKVOL>']
        })

        logger.info("列の整形完了。")

        # 3. 整形後のファイルを保存
        os.makedirs(os.path.dirname(OUTPUT_FILE_PATH), exist_ok=True)
        logger.info(f"整形後のデータを '{OUTPUT_FILE_PATH}' に保存しています...")
        df_final.to_csv(OUTPUT_FILE_PATH, index=False)

        logger.info("--- ✅ データ整形処理が正常に完了しました ---")

    except Exception as e:
        logger.error(f"処理中にエラーが発生しました: {e}", exc_info=True)
        logger.error("ファイルの形式がMT5エクスポートの標準と異なる可能性があります。")

if __name__ == '__main__':
    format_mt5_data()