# python /workspace/models/inspect_one.py として保存して実行
import polars as pl

# 検査対象のファイルを1つ指定（パスは実際の環境に合わせてください）
target_file = "/workspace/data/XAUUSD/stratum_6_training/1A_2B/labeled_dataset_partitioned_v2/year=2021/month=8/day=9/data.parquet"

df = pl.read_parquet(target_file)

print(f"=== ファイル: {target_file} ===")
print(f"総行数: {df.height}")
print("\n=== 重複チェック (Top 5) ===")
# 同じ時刻・時間足でカウント
counts = (
    df.group_by(["timestamp", "timeframe"]).len().sort("len", descending=True).head(5)
)
print(counts)

if counts["len"][0] == 1:
    print("\n✅ 最大重複数は「1」です。つまり重複は存在しません。")
else:
    print(f"\n❌ 異常あり！重複数が {counts['len'][0]} です。")
