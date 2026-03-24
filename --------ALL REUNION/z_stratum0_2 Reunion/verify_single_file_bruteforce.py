# verify_single_file_bruteforce.py
import cudf
from pathlib import Path
import warnings

# --- 基本設定 ---
warnings.filterwarnings("ignore")

# --- 設定 ---
# ★★★ 必ず、最新のGOLDEN_v2ディレクトリを指定してください ★★★
BASE_DATA_PATH = Path("/workspaces/project_forge/data/1_XAUUSD_base_data/XAUUSD_tick_master_GOLDEN_MANUAL")

# ★★★ 検証したいタイムフレームを指定してください ★★★
TIMEFRAME_TO_INSPECT = 'M1' 

# --- 実行 ---
if __name__ == "__main__":
    print(f"--- 【キャッシュ無視・単一ファイル強制検証】 ---")
    
    # タイムフレームのディレクトリパスを構築
    partition_path = BASE_DATA_PATH / f"timeframe={TIMEFRAME_TO_INSPECT}"
    print(f"調査対象ディレクトリ: {partition_path}")

    if not partition_path.is_dir():
        print(f"❌ エラー: ディレクトリが見つかりません。")
    else:
        # ディレクトリ内の最初のParquetファイルを取得
        try:
            first_file = next(partition_path.glob("*.parquet"))
            print(f"読み込み対象ファイル: {first_file}")
            
            # cudfを使って、Daskを介さずにファイルを直接読み込む
            df = cudf.read_parquet(first_file)
            
            print("\n--- 読み込み成功 ---")
            print("【データ型情報】")
            df.info() # .info()は直接出力するのでprintは不要
            
            # timeframeカラムのデータ型をピンポイントで確認
            timeframe_dtype = df['timeframe'].dtype
            print(f"\ntimeframeカラムの実際のデータ型: {timeframe_dtype}")
            
            if str(timeframe_dtype) == 'object' or str(timeframe_dtype) == 'string':
                print("\n✅ 成功！ timeframeカラムは正しく'string'型として保存されています。")
            else:
                print(f"\n❌ 失敗... timeframeカラムはまだ'{timeframe_dtype}'型のままです。")

        except StopIteration:
            print(f"❌ エラー: '{partition_path}' 内にParquetファイルが見つかりません。")
        except Exception as e:
            print(f"\n❌ データ読み込み中にエラーが発生しました: {e}")