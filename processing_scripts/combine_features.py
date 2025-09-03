import pandas as pd
import os
import joblib
from common.logger_setup import logger
from tqdm import tqdm
import gc

def combine_features_final():
    """
    修正版v2.2 - 診断機能付きの安全な特徴量結合
    """
    source_dir = 'data/temp_chunks'
    output_file = 'data/xauusd_feature_universe.joblib'

    chunk_files = sorted([f for f in os.listdir(source_dir) if f.endswith('.parquet')])
    if not chunk_files:
        logger.error(f"チャンクファイルが見つかりません: {source_dir}")
        return

    logger.info(f"--- 🚀 ステージ1: {len(chunk_files)}個のチャンクの診断と結合 ---")
    
    # ステップ1: 各チャンクの診断
    all_chunks_list = []
    chunk_diagnostics = []
    
    for f in tqdm(chunk_files, desc="Loading and diagnosing chunks"):
        chunk_path = os.path.join(source_dir, f)
        chunk = pd.read_parquet(chunk_path)
        
        # チャンクの基本情報
        diagnostic = {
            'file': f,
            'shape': chunk.shape,
            'date_range': (chunk.index.min(), chunk.index.max()) if len(chunk) > 0 else None
        }
        
        # MFDFA列の診断
        mfdfa_cols = [col for col in chunk.columns if 'mf_spectrum_width' in col]
        diagnostic['mfdfa_columns'] = len(mfdfa_cols)
        
        for col in mfdfa_cols[:2]:  # 最初の2列のみ詳細チェック
            valid_count = chunk[col].count()
            total_count = len(chunk)
            valid_pct = valid_count / total_count if total_count > 0 else 0
            
            diagnostic[f'{col}_valid_pct'] = valid_pct
            if valid_count > 0:
                diagnostic[f'{col}_range'] = (chunk[col].min(), chunk[col].max())
            
        chunk_diagnostics.append(diagnostic)
        
        logger.info(f"チャンク {f}: 形状{chunk.shape}, "
                   f"MFDFA列数: {len(mfdfa_cols)}")
        
        if mfdfa_cols:
            for col in mfdfa_cols[:1]:  # 主要列のみ表示
                valid_count = chunk[col].count()
                total_count = len(chunk)
                if valid_count > 0:
                    logger.info(f"  {col}: {valid_count}/{total_count} valid "
                               f"({valid_count/total_count:.2%}), "
                               f"範囲: {chunk[col].min():.4f} - {chunk[col].max():.4f}")
                else:
                    logger.warning(f"  {col}: すべてNaN!")
        
        all_chunks_list.append(chunk)
    
    # ステップ2: チャンク結合
    logger.info("--- 🚀 ステージ2: チャンクを結合します ---")
    combined_df = pd.concat(all_chunks_list, ignore_index=False)
    del all_chunks_list
    gc.collect()

    logger.info(f"結合後の形状: {combined_df.shape}")
    
    # ステップ3: 結合後の全体診断
    logger.info("--- 🚀 ステージ3: 結合後の診断 ---")
    
    # 全MFDFA特徴量の状態確認
    mfdfa_cols = [col for col in combined_df.columns if 'mf_spectrum_width' in col]
    logger.info(f"検出されたMFDFA特徴量: {len(mfdfa_cols)}列")
    
    valid_mfdfa_cols = []
    for col in mfdfa_cols:
        valid_count = combined_df[col].count()
        total_count = len(combined_df)
        valid_pct = valid_count / total_count
        
        if valid_count > 0:
            logger.info(f"  ✓ {col}: {valid_count}/{total_count} valid ({valid_pct:.2%})")
            logger.info(f"    範囲: {combined_df[col].min():.4f} - {combined_df[col].max():.4f}")
            valid_mfdfa_cols.append(col)
        else:
            logger.warning(f"  ❌ {col}: すべてNaN!")
    
    # ステップ4: 欠損値補完
    logger.info("--- 🚀 ステージ4: 欠損値補完 ---")
    logger.info("ffill/bfillで欠損値を補完中...")
    combined_df.ffill(inplace=True)
    combined_df.bfill(inplace=True)

    # ステップ5: クレンジング戦略の決定
    logger.info("--- 🚀 ステージ5: クレンジング ---")
    
    target_col = 'mf_spectrum_width_1000_1T'
    
    if target_col in combined_df.columns and target_col in valid_mfdfa_cols:
        # 有効な基準列がある場合
        logger.info(f"基準列 '{target_col}' でクレンジングを実行...")
        original_rows = len(combined_df)
        combined_df.dropna(subset=[target_col], inplace=True)
        removed_rows = original_rows - len(combined_df)
        logger.info(f"クレンジング完了: {removed_rows}行除去, {len(combined_df)}行残存")
        
    elif len(valid_mfdfa_cols) > 0:
        # 他の有効な列がある場合
        alternative_col = valid_mfdfa_cols[0]
        logger.info(f"基準列が無効のため、代替列 '{alternative_col}' を使用...")
        original_rows = len(combined_df)
        combined_df.dropna(subset=[alternative_col], inplace=True)
        removed_rows = original_rows - len(combined_df)
        logger.info(f"代替クレンジング完了: {removed_rows}行除去, {len(combined_df)}行残存")
        
    else:
        # 有効なMFDFA列がない場合
        logger.error("❌ 有効なMFDFA特徴量が見つかりません!")
        logger.info("基本的なOHLCV列の完全性チェックのみ実行...")
        original_rows = len(combined_df)
        combined_df.dropna(subset=['close'], inplace=True)
        removed_rows = original_rows - len(combined_df)
        logger.info(f"最小クレンジング完了: {removed_rows}行除去, {len(combined_df)}行残存")
        
        # 診断情報の保存
        _save_diagnostic_info(chunk_diagnostics, combined_df, mfdfa_cols)

    # ステップ6: 最終確認と保存
    logger.info(f"最終的な特徴量ユニバースの形状: {combined_df.shape}")
    
    if len(combined_df) > 0:
        # 最終的な特徴量統計
        logger.info("=== 最終統計情報 ===")
        logger.info(f"期間: {combined_df.index.min()} ～ {combined_df.index.max()}")
        logger.info(f"総特徴量数: {combined_df.shape[1]}列")
        
        # 有効なMFDFA特徴量の再確認
        final_valid_mfdfa = []
        for col in mfdfa_cols:
            if combined_df[col].count() > len(combined_df) * 0.5:  # 50%以上有効
                final_valid_mfdfa.append(col)
        
        logger.info(f"有効なMFDFA特徴量: {len(final_valid_mfdfa)}列")
        
        logger.info(f"最終ファイルを '{output_file}' に保存中...")
        joblib.dump(combined_df, output_file, compress=('gzip', 3))
        logger.info("--- ✅ 全ての処理が完了しました！ ---")
        
    else:
        logger.error("--- ❌ 処理後にデータが残っていません ---")
        logger.info("問題解決のための推奨事項:")
        logger.info("1. MFDFAのwindowサイズを小さくする (例: 500)")
        logger.info("2. チャンクサイズを大きくして計算精度を向上させる")
        logger.info("3. 元データの品質を確認する")


def _save_diagnostic_info(chunk_diagnostics, combined_df, mfdfa_cols):
    """診断情報をファイルに保存"""
    diagnostic_file = 'data/mfdfa_diagnostic.txt'
    
    with open(diagnostic_file, 'w', encoding='utf-8') as f:
        f.write("=== MFDFA特徴量生成診断レポート ===\n\n")
        
        f.write("チャンク別診断:\n")
        for diag in chunk_diagnostics:
            f.write(f"- {diag['file']}: {diag['shape']}, MFDFA列数: {diag['mfdfa_columns']}\n")
            for key, value in diag.items():
                if 'valid_pct' in key:
                    f.write(f"  {key}: {value:.2%}\n")
        
        f.write(f"\n結合後統計:\n")
        f.write(f"- 最終形状: {combined_df.shape}\n")
        f.write(f"- MFDFA列数: {len(mfdfa_cols)}\n")
        
        for col in mfdfa_cols:
            valid_pct = combined_df[col].count() / len(combined_df)
            f.write(f"- {col}: {valid_pct:.2%} valid\n")
    
    logger.info(f"診断情報を {diagnostic_file} に保存しました")


if __name__ == '__main__':
    combine_features_final()