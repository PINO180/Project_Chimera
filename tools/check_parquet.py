import pandas as pd
from pathlib import Path

# 調査対象のファイルパスを正確に指定
file_path = Path(r"C:\project_forge\data\temp_chunks\parquet\01_mfdfa_45features.parquet")

print(f"🔍 ファイルを調査中: {file_path}")

if not file_path.exists():
    print("❌ ファイルが見つかりません。パスを確認してください。")
else:
    try:
        print("\n--- 方法1: メタデータのみ読み込み (pandas/pyarrow) ---")
        
        # メインスクリプトでエラーが発生しているのと同じ方法
        df_meta = pd.read_parquet(file_path, columns=[])
        
        print(f"✅ メタデータ読み込み成功")
        
        columns_list = df_meta.columns.tolist()
        print(f"  -> カラム数: {len(columns_list)}")
        
        if columns_list:
            print(f"  -> カラム名 (最初の5つ): {columns_list[:5]}")
            print("\n💡 結論: ファイルのメタデータは正常に読み込めています。")
        else:
            print("\n🚨 結論: カラムが見つかりませんでした。メタデータが破損している可能性が非常に高いです。")

    except Exception as e:
        print(f"\n🚨 エラーが発生しました: {e}")
        print("💡 結論: ファイルが破損しているか、ライブラリとの互換性がないため読み込めません。")