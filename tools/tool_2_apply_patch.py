# tool_2_apply_patch.py

"""
エラー復旧ツール ステップ2：パッチの適用と結合

このスクリプトは、ステップ1で作成したパッチファイル（欠落特徴量）を、
不完全なメインデータセットに結合し、完全なデータセットを完成させます。

#################################################################
# 使い方 (How to Use)
#################################################################
#
# 1. 下記の「--- 設定 ---」セクションを編集します。
#    - main_path: 不完全なメインファイル（'02_combined_...'など）を指定。
#    - patch_path: ステップ1で作成したパッチファイル（'recovery_...'）を指定。
#    - final_output_path: 完成したデータセットの最終的な出力パスを指定。
#
# 2. PowerShellで実行します。
#    python tool_2_apply_patch.py
#
#################################################################
"""

import pandas as pd
import logging
from pathlib import Path

# ロガー設定
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def apply_recovery_patch():
    """
    不完全なデータセットと、復旧した特徴量（パッチ）を結合する
    """
    try:
        # --- 設定 ---
        # 不完全なメインファイル（エラー時に生成されたもの）
        main_path = Path(r"C:\project_forge\data\temp_chunks\parquet\02_combined_563features.parquet")
        # 復旧した特徴量ファイル（パッチ）
        patch_path = Path(r"C:\project_forge\data\temp_chunks\parquet\recovery_moving_averages_advanced.parquet")
        # 最終的な完全版データセットの出力先
        final_output_path = Path(r"C:\project_forge\data\temp_chunks\parquet\02_combined_final_complete.parquet")

        logging.info("=== データセット結合開始 ===")
        
        # 1. 両方のファイルを読み込み
        logging.info(f"不完全なメインファイルを読み込み中: {main_path}")
        main_df = pd.read_parquet(main_path)
        
        logging.info(f"パッチファイルを読み込み中: {patch_path}")
        patch_df = pd.read_parquet(patch_path)
        
        # 2. 横方向に結合
        logging.info(f"データを結合中... (メイン: {main_df.shape}, パッチ: {patch_df.shape})")
        combined_df = pd.concat([main_df, patch_df], axis=1)
        
        # 3. 重複列のチェックと修正
        is_duplicate = combined_df.columns.duplicated(keep=False)
        if is_duplicate.any():
            duplicate_cols = combined_df.columns[is_duplicate].tolist()
            logging.warning(f"警告: 重複列が検出されました -> {duplicate_cols}")
            combined_df = combined_df.loc[:, ~combined_df.columns.duplicated(keep='first')]
            logging.info("重複列を除去しました。")
        else:
            logging.info("重複列はありませんでした。")

        logging.info(f"結合後のデータ Shape: {combined_df.shape}")
        
        # 4. 最終的な完全版ファイルを保存
        logging.info(f"完全版データセットを保存中: {final_output_path}")
        combined_df.to_parquet(final_output_path, compression='snappy')
        
        logging.info(f"=== 結合完了！完全なデータセットが作成されました: {final_output_path} ===")

    except Exception as e:
        logging.error(f"処理中にエラーが発生しました: {e}")
        raise

if __name__ == "__main__":
    apply_recovery_patch()