# === プロジェクトルートに保存: diagnosis.py ===
import pandas as pd
import numpy as np
import os
import sys

# プロジェクトのパスを追加
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from fathon import MFDFA
    from common.logger_setup import logger
except ImportError as e:
    print(f"インポートエラー: {e}")
    print("必要なライブラリをインストールしてください")
    sys.exit(1)

def quick_diagnosis():
    """クイック診断"""
    print("=== MFDFA問題クイック診断 ===")
    
    # 1. 入力ファイルの確認
    input_file = 'data/XAUUSD_1m.csv'
    if not os.path.exists(input_file):
        print(f"❌ 入力ファイルが見つかりません: {input_file}")
        return False
    
    print(f"✅ 入力ファイル確認: {input_file}")
    
    # 2. チャンクディレクトリの確認
    temp_dir = 'data/temp_chunks'
    chunk_files = []
    if os.path.exists(temp_dir):
        chunk_files = [f for f in os.listdir(temp_dir) if f.endswith('.parquet')]
        print(f"✅ 既存チャンク数: {len(chunk_files)}")
    else:
        print(f"⚠️ チャンクディレクトリなし: {temp_dir}")
    
    # 3. サンプルデータでMFDFAテスト
    print("\n=== サンプルデータでMFDFAテスト ===")
    try:
        # シンプルなテストデータ生成
        np.random.seed(42)
        test_prices = np.cumsum(np.random.randn(2000)) + 100
        
        # MFDFA計算テスト
        window = 500
        series = test_prices[-window:]
        
        # 対数リターンに変換
        returns = np.diff(np.log(series))
        
        # ウィンドウサイズ設定
        max_win = len(returns) // 8
        min_win = 10
        winSizes = np.logspace(np.log10(min_win), np.log10(max_win), 15).astype(int)
        winSizes = np.unique(winSizes)
        
        print(f"テスト系列長: {len(returns)}")
        print(f"ウィンドウサイズ: {len(winSizes)}個 ({winSizes[0]} - {winSizes[-1]})")
        
        # MFDFA実行
        q_vals = np.array([-2, -0.5, 0.5, 2])
        mfd_d_fa = MFDFA(returns, lag=winSizes, q=q_vals, order=2)
        
        alpha_spec = mfd_d_fa.alpha()
        h_spec = mfd_d_fa.Hq()
        
        # 特徴量計算
        mf_width = alpha_spec.max() - alpha_spec.min()
        h_q2 = h_spec[3]
        h_q_neg2 = h_spec[0]
        h_diff = h_q2 - h_q_neg2
        
        print(f"✅ MFDFA計算成功!")
        print(f"   MF width: {mf_width:.4f}")
        print(f"   H_q2: {h_q2:.4f}")
        print(f"   H_q-2: {h_q_neg2:.4f}")
        print(f"   H_diff: {h_diff:.4f}")
        
        return True
        
    except Exception as e:
        print(f"❌ MFDFA計算失敗: {e}")
        import traceback
        traceback.print_exc()
        return False

def check_existing_chunks():
    """既存チャンクの簡易チェック"""
    temp_dir = 'data/temp_chunks'
    if not os.path.exists(temp_dir):
        print("チャンクディレクトリなし")
        return
    
    chunk_files = [f for f in os.listdir(temp_dir) if f.endswith('.parquet')]
    print(f"\n=== 既存チャンク確認 ({len(chunk_files)}個) ===")
    
    for i, chunk_file in enumerate(chunk_files[:2]):  # 最初の2つのみ
        try:
            chunk_path = os.path.join(temp_dir, chunk_file)
            chunk_df = pd.read_parquet(chunk_path)
            
            print(f"\nチャンク {i+1}: {chunk_file}")
            print(f"  形状: {chunk_df.shape}")
            
            # MFDFA列の確認
            mfdfa_cols = [col for col in chunk_df.columns if 'mf_spectrum_width' in col]
            print(f"  MFDFA列数: {len(mfdfa_cols)}")
            
            for col in mfdfa_cols[:1]:  # 1列のみチェック
                valid_count = chunk_df[col].count()
                total_count = len(chunk_df)
                
                if valid_count > 0:
                    print(f"  ✅ {col}: {valid_count}/{total_count} valid ({valid_count/total_count:.1%})")
                else:
                    print(f"  ❌ {col}: すべてNaN")
                    
        except Exception as e:
            print(f"  ❌ チャンク読み込みエラー: {e}")

def main():
    print("🔍 MFDFA問題の診断を開始します...\n")
    
    # 基本診断
    mfdfa_works = quick_diagnosis()
    
    # 既存チャンク確認
    check_existing_chunks()
    
    # 推奨事項
    print("\n=== 推奨対策 ===")
    if mfdfa_works:
        print("✅ MFDFA自体は動作します")
        print("📋 次の手順を試してください:")
        print("1. 既存チャンクを削除: rm -rf data/temp_chunks/*")
        print("2. より大きなチャンクサイズで再実行:")
        print("   python -m features.run_feature_engineering --workers 4 --chunk_size 500000")
        print("3. 修正版のfeature_logic.pyを使用")
    else:
        print("❌ MFDFA計算に問題があります")
        print("📋 以下を確認してください:")
        print("1. fathonライブラリが正しくインストールされているか")
        print("2. 依存関係の問題がないか")
        print("3. Pythonのバージョンが対応しているか")

if __name__ == '__main__':
    main()