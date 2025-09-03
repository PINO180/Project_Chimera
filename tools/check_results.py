# check_results.py - 各段階の結果を検証するスクリプト

import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt

def check_mfdfa_results(file_path):
    """MFDFA結果の検証"""
    print("=" * 50)
    print("MFDFA結果検証")
    print("=" * 50)
    
    try:
        df = pd.read_parquet(file_path)
        print(f"ファイル読み込み成功: {file_path}")
        print(f"データ形状: {df.shape}")
        
        # MFDFA特徴量の列
        mfdfa_columns = ['mfdfa_h2', 'mfdfa_h_minus_2', 'mfdfa_spectrum_width', 'mfdfa_h_diff']
        
        print(f"\n--- MFDFA特徴量のNaN率 ---")
        for col in mfdfa_columns:
            if col in df.columns:
                total = len(df)
                nan_count = df[col].isna().sum()
                nan_rate = nan_count / total * 100
                print(f"  {col:25s}: {nan_count:,} / {total:,} ({nan_rate:.1f}% がNaN)")
                
                # 有効値の統計
                valid_data = df[col].dropna()
                if len(valid_data) > 0:
                    print(f"    有効値統計: 平均={valid_data.mean():.4f}, 標準偏差={valid_data.std():.4f}")
                    print(f"    範囲: {valid_data.min():.4f} ~ {valid_data.max():.4f}")
            else:
                print(f"  {col}: 列が見つかりません")
        
        # 成功判定
        valid_columns = [col for col in mfdfa_columns if col in df.columns]
        if valid_columns:
            overall_success_rate = (1 - df[valid_columns].isna().all(axis=1).mean()) * 100
            print(f"\n📊 総合成功率: {overall_success_rate:.1f}%")
            
            if overall_success_rate > 80:
                print("✅ MFDFA計算は成功しています")
                return True
            elif overall_success_rate > 10:
                print("⚠️  MFDFA計算は部分的に成功していますが、改善の余地があります")
                return True
            else:
                print("❌ MFDFA計算はほぼ失敗しています")
                return False
        else:
            print("❌ MFDFA列が見つかりません")
            return False
            
    except Exception as e:
        print(f"❌ エラー: {e}")
        return False

def check_garch_results(file_path):
    """GARCH結果の検証"""
    print("=" * 50)
    print("GARCH結果検証")
    print("=" * 50)
    
    try:
        df = pd.read_parquet(file_path)
        print(f"ファイル読み込み成功: {file_path}")
        print(f"データ形状: {df.shape}")
        
        # GARCH特徴量の列
        garch_columns = ['garch_vol_forecast', 'standardized_residual']
        
        print(f"\n--- GARCH特徴量のNaN率 ---")
        for col in garch_columns:
            if col in df.columns:
                total = len(df)
                nan_count = df[col].isna().sum()
                nan_rate = nan_count / total * 100
                print(f"  {col:25s}: {nan_count:,} / {total:,} ({nan_rate:.1f}% がNaN)")
                
                # 有効値の統計
                valid_data = df[col].dropna()
                if len(valid_data) > 0:
                    print(f"    有効値統計: 平均={valid_data.mean():.4f}, 標準偏差={valid_data.std():.4f}")
                    print(f"    範囲: {valid_data.min():.4f} ~ {valid_data.max():.4f}")
                    
                    # 異常値チェック
                    if col == 'garch_vol_forecast':
                        negative_count = (valid_data < 0).sum()
                        extreme_count = (valid_data > 1).sum()  # 100%を超えるボラティリティ
                        print(f"    異常値: 負の値={negative_count}, 極端に大きい値={extreme_count}")
                    elif col == 'standardized_residual':
                        extreme_residual = (np.abs(valid_data) > 10).sum()
                        print(f"    異常値: |標準化残差| > 10 の数={extreme_residual}")
            else:
                print(f"  {col}: 列が見つかりません")
        
        # 成功判定
        valid_columns = [col for col in garch_columns if col in df.columns]
        if valid_columns:
            overall_success_rate = (1 - df[valid_columns].isna().all(axis=1).mean()) * 100
            print(f"\n📊 総合成功率: {overall_success_rate:.1f}%")
            
            if overall_success_rate > 80:
                print("✅ GARCH計算は成功しています")
                return True
            elif overall_success_rate > 10:
                print("⚠️  GARCH計算は部分的に成功していますが、改善の余地があります")
                return True
            else:
                print("❌ GARCH計算はほぼ失敗しています")
                return False
        else:
            print("❌ GARCH列が見つかりません")
            return False
            
    except Exception as e:
        print(f"❌ エラー: {e}")
        return False

def visualize_features(file_path, feature_type="both"):
    """特徴量の可視化"""
    print("=" * 50)
    print("特徴量可視化")
    print("=" * 50)
    
    try:
        df = pd.read_parquet(file_path)
        
        # 最新の1000点のみプロット（高速化）
        df_plot = df.tail(1000).copy()
        
        if feature_type in ["mfdfa", "both"]:
            mfdfa_columns = ['mfdfa_h2', 'mfdfa_h_minus_2', 'mfdfa_spectrum_width', 'mfdfa_h_diff']
            valid_mfdfa = [col for col in mfdfa_columns if col in df_plot.columns and df_plot[col].notna().sum() > 10]
            
            if valid_mfdfa:
                fig, axes = plt.subplots(2, 2, figsize=(15, 10))
                fig.suptitle('MFDFA特徴量（最新1000点）', fontsize=16)
                
                for i, col in enumerate(valid_mfdfa[:4]):
                    row, col_idx = i // 2, i % 2
                    data = df_plot[col].dropna()
                    if len(data) > 0:
                        axes[row, col_idx].plot(data.index, data.values, linewidth=0.8)
                        axes[row, col_idx].set_title(col)
                        axes[row, col_idx].grid(True, alpha=0.3)
                
                plt.tight_layout()
                plt.savefig('mfdfa_features.png', dpi=150, bbox_inches='tight')
                plt.show()
                print("MFDFA特徴量のプロットを保存しました: mfdfa_features.png")
        
        if feature_type in ["garch", "both"]:
            garch_columns = ['garch_vol_forecast', 'standardized_residual']
            valid_garch = [col for col in garch_columns if col in df_plot.columns and df_plot[col].notna().sum() > 10]
            
            if valid_garch:
                fig, axes = plt.subplots(len(valid_garch), 1, figsize=(15, 6*len(valid_garch)))
                if len(valid_garch) == 1:
                    axes = [axes]
                fig.suptitle('GARCH特徴量（最新1000点）', fontsize=16)
                
                for i, col in enumerate(valid_garch):
                    data = df_plot[col].dropna()
                    if len(data) > 0:
                        axes[i].plot(data.index, data.values, linewidth=0.8)
                        axes[i].set_title(col)
                        axes[i].grid(True, alpha=0.3)
                        
                        # ボラティリティの場合は正の値のみ表示
                        if col == 'garch_vol_forecast':
                            axes[i].set_ylim(bottom=0)
                
                plt.tight_layout()
                plt.savefig('garch_features.png', dpi=150, bbox_inches='tight')
                plt.show()
                print("GARCH特徴量のプロットを保存しました: garch_features.png")
                
    except Exception as e:
        print(f"可視化エラー: {e}")

def main():
    """メイン実行関数"""
    base_dir = Path(r"C:\project_forge\data\temp_chunks\parquet")
    
    print("🔍 Tier1特徴量結果検証ツール")
    print("=" * 50)
    
    # 1. MFDFAファイルの確認
    mfdfa_files = list(base_dir.glob("*mfdfa*.parquet"))
    if mfdfa_files:
        print("📁 見つかったMFDFAファイル:")
        for i, file in enumerate(mfdfa_files):
            print(f"  {i+1}. {file.name}")
        
        # 最新のMFDFAファイルを自動選択
        latest_mfdfa = max(mfdfa_files, key=lambda x: x.stat().st_mtime)
        print(f"\n最新のMFDFAファイルを検証: {latest_mfdfa.name}")
        
        mfdfa_success = check_mfdfa_results(latest_mfdfa)
        
        # 可視化
        if mfdfa_success:
            visualize_features(latest_mfdfa, "mfdfa")
    else:
        print("📁 MFDFAファイルが見つかりません")
        mfdfa_success = False
    
    print("\n" + "="*50)
    
    # 2. GARCHファイルの確認
    garch_files = list(base_dir.glob("*garch*.parquet")) + list(base_dir.glob("*tier1_complete*.parquet"))
    if garch_files:
        print("📁 見つかったGARCHファイル:")
        for i, file in enumerate(garch_files):
            print(f"  {i+1}. {file.name}")
        
        # 最新のGARCHファイルを自動選択
        latest_garch = max(garch_files, key=lambda x: x.stat().st_mtime)
        print(f"\n最新のGARCHファイルを検証: {latest_garch.name}")
        
        garch_success = check_garch_results(latest_garch)
        
        # 可視化
        if garch_success:
            visualize_features(latest_garch, "garch")
    else:
        print("📁 GARCHファイルが見つかりません")
    
    # 総合判定
    print("\n" + "="*60)
    print("📋 総合結果")
    print("="*60)
    
    if mfdfa_success:
        print("✅ MFDFA特徴量: 正常")
    else:
        print("❌ MFDFA特徴量: 問題あり")
    
    if 'garch_success' in locals() and garch_success:
        print("✅ GARCH特徴量: 正常")
    elif 'garch_success' in locals():
        print("❌ GARCH特徴量: 問題あり")
    else:
        print("⏸️  GARCH特徴量: 未実行")

if __name__ == "__main__":
    main()