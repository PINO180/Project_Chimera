# check_duplicates_source.py
import polars as pl
from pathlib import Path

# プロジェクトパス (環境に合わせてください)
BASE_DIR = Path("/workspace/data/XAUUSD")

# 1. 入力元: S5 (Neutralized Alpha) のチェック
s5_path = BASE_DIR / "stratum_5_alpha/1A_2B/neutralized_alpha_set_partitioned"

# 2. 結合対象: S2 (Feature Fixed) または S1 (Base) のチェック
# ※ create_proxy... が何を結合しているかによりますが、主要な入力を見ます


def quick_check(path, name):
    print(f"--- Checking {name} ---")
    if not path.exists():
        print(f"❌ Path not found: {path}")
        return

    # 最新の1日分だけ読む（これで十分）
    files = sorted(list(path.glob("**/*.parquet")))
    if not files:
        print("❌ No files found.")
        return

    # 最後の1ファイルをサンプルとして読む
    sample_file = files[-1]
    print(f"📖 Reading sample: {sample_file}")

    df = pl.read_parquet(sample_file)

    # 重複チェック
    # timestamp と timeframe (あれば) でグループ化してカウント
    subset = ["timestamp"]
    if "timeframe" in df.columns:
        subset.append("timeframe")

    dup_check = df.group_by(subset).len().filter(pl.col("len") > 1)

    if dup_check.height > 0:
        print(f"😱 DUPLICATES FOUND in {name}!")
        print(f"   Max duplication count: {dup_check['len'].max()}")
        print(dup_check.sort("len", descending=True).head(3))
    else:
        print(f"✅ Clean (No duplicates in {name})")


quick_check(s5_path, "S5 (Alpha Input)")
# 必要であれば他の入力ソースも同様にチェック
