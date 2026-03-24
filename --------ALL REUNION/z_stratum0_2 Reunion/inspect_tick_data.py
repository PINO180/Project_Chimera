import cudf
import warnings

warnings.filterwarnings("ignore", category=UserWarning)

# --- 設定 ---
BASE_DATA_PATH = "/workspaces/project_forge/data/1_XAUUSD_base_data/XAUUSD_tick_master_FINAL_PATCHED"
TIMEFRAME_TO_INSPECT = 'tick'

print(f"--- '{TIMEFRAME_TO_INSPECT}' データの構造を調査します ---")

try:
    # cudfを使い、指定したパーティションのデータのみを読み込む
    df = cudf.read_parquet(
        BASE_DATA_PATH,
        filters=[('timeframe', '==', TIMEFRAME_TO_INSPECT)]
    )

    if df.empty:
        print("データは空です。")
    else:
        print("\n【カラム一覧】")
        print(list(df.columns))

        print("\n【データ型情報】")
        df.info() # .info()は直接出力するのでprintは不要

        print("\n【先頭10行のデータサンプル】")
        print(df.head(10)) # head()の引数に10を指定して先頭10行を表示

except Exception as e:
    print(f"\n❌ データ読み込み中にエラーが発生しました: {e}")