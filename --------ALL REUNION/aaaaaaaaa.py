import polars as pl
from pathlib import Path

p = Path(
    "/workspace/data/XAUUSD/stratum_5_alpha/1A_2B/neutralized_alpha_set_partitioned/feature_value_a_vast_universeA/features_e1a_H1_neutralized.parquet"
)
df = pl.read_parquet(p)

# 特徴量カラムのみで統計確認
feat_cols = [c for c in df.columns if c != "timestamp"]

# 0.0以外の値が存在するか確認
non_zero = df.select(
    [pl.col(c).filter(pl.col(c) != 0.0).count().alias(c) for c in feat_cols[:3]]
)
print("0.0以外の行数（先頭3カラム）:")
print(non_zero)

# 中盤のデータを確認
print("\n中盤100行付近のデータ:")
print(df.slice(100, 3).select(["timestamp"] + feat_cols[:3]))
