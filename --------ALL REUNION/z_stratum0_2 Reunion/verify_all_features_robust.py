import dask
import dask.dataframe as dd
import numpy as np
import warnings
import gc
import time

warnings.filterwarnings("ignore")

# --- 設定 ---
# 元データのパス
SOURCE_PATH = "/workspaces/project_forge/data/1_XAUUSD_base_data/XAUUSD_tick_master_FINAL_PATCHED"

# 検証対象の特徴量カラム
FEATURES_TO_CHECK = [
    'log_return', 'rolling_volatility', 'rolling_avg_volume', 
    'atr', 'price_direction', 'price_momentum', 'volume_ratio'
]

print(f"--- 【超堅牢版】全特徴量・データ品質検証を開始します ---")
print("--> 速度よりも完走を最優先します。処理には時間がかかる場合があります。")
print(f"対象ディレクトリ: {SOURCE_PATH}")

def main():
    start_time = time.time()
    
    # メモリ効率を最優先するため、小さなブロックサイズで読み込む
    ddf = dd.read_parquet(SOURCE_PATH, blocksize='128MB')
    all_timeframes = sorted(ddf['timeframe'].unique().compute().tolist())
    
    print("\n--- 各タイムフレームの検証結果 ---")
    
    # 各時間足をループして検証
    for tf in all_timeframes:
        print(f"\n▼▼▼ タイムフレーム '{tf}' の検証中... ▼▼▼")
        # Daskの計算グラフを小さく保つため、ループごとにDataFrameを再定義
        tf_ddf = dd.read_parquet(SOURCE_PATH, filters=[('timeframe', '==', tf)], blocksize='128MB')
        
        issues_found_in_tf = False
        
        for col in FEATURES_TO_CHECK:
            if col not in tf_ddf.columns:
                continue

            # 各チェック項目を一つずつ、個別に計算・実行する
            # これにより、Daskのメモリ負荷を最小限に抑える
            
            # 1. ゼロ値のチェック
            zeros_count = (tf_ddf[col] == 0).sum().compute()
            if zeros_count > 0:
                print(f"  ❌ カラム '{col}': ゼロ値が {zeros_count:,} 個あります。")
                issues_found_in_tf = True

            # 2. 欠損値(NA)のチェック
            nas_count = tf_ddf[col].isna().sum().compute()
            if nas_count > 0:
                print(f"  ❌ カラム '{col}': 欠損値(NA)が {nas_count:,} 個あります。")
                issues_found_in_tf = True
                
            # 3. 無限大(inf)のチェック
            pos_inf_count = (tf_ddf[col] == np.inf).sum().compute()
            if pos_inf_count > 0:
                print(f"  ❌ カラム '{col}': 無限大(inf)が {pos_inf_count:,} 個あります。")
                issues_found_in_tf = True
                
            # 4. 無限大(-inf)のチェック
            neg_inf_count = (tf_ddf[col] == -np.inf).sum().compute()
            if neg_inf_count > 0:
                print(f"  ❌ カラム '{col}': 無限大(-inf)が {neg_inf_count:,} 個あります。")
                issues_found_in_tf = True

        if not issues_found_in_tf:
            print(f"  ✅ このタイムフレームでは問題は見つかりませんでした。")
            
        # 次のタイムフレームに移る前に、明示的にメモリを解放
        del tf_ddf
        gc.collect()
            
    total_duration = time.time() - start_time
    print("\n--- 検証完了 ---")
    print(f"総処理時間: {total_duration / 60:.1f} 分")
    print("データセットの品質評価が完了しました。")

if __name__ == "__main__":
    # 安定性を最優先するシングルスレッドスケジューラを設定
    dask.config.set({
        'scheduler': 'single-threaded'
    })
    
    main()