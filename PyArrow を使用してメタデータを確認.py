import pyarrow.parquet as pq

# 確認したいファイルのパス
file_path = "/workspace/data/XAUUSD/stratum_2_features_fixed/feature_value_complexity_theory_MFDFA&kolmogorov/features_e2a_M5.parquet"

try:
    # Parquetファイルのスキーマ情報のみを読み込む
    schema = pq.read_schema(file_path)
    
    print("✅ ファイルスキーマの確認に成功しました。")
    print("="*50)
    print(schema)
    print("="*50)

except Exception as e:
    print(f"❌ ファイルの確認中にエラーが発生しました: {e}")