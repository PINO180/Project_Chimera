# /workspace/models/check_mtf_bug.py
import sys
from pathlib import Path
import polars as pl

# プロジェクトのパスを追加
PROJECT_ROOT = Path("/workspace")
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from blueprint import S6_LABELED_DATASET


def main():
    print(f"🔍 Checking S6 Dataset for MTF (Multi-Timeframe) Integrity...")

    # S6のParquetファイルを1つ取得
    files = list(S6_LABELED_DATASET.rglob("*.parquet"))
    if not files:
        print("❌ S6 Parquet files not found.")
        return

    target_file = files[0]
    print(f"📂 Loading sample file: {target_file.name}\n")

    df = pl.read_parquet(target_file)

    # 特徴量カラム（e1c_ や e1f_ で始まるもの）を抽出
    feature_cols = [c for c in df.columns if c.startswith("e1")]

    m1_cols = [c for c in feature_cols if c.endswith("_M1")]
    other_cols = [c for c in feature_cols if not c.endswith("_M1")]

    print(f"📊 Found {len(m1_cols)} M1 features.")
    print(f"📊 Found {len(other_cols)} Upper timeframe (H1, D1, etc.) features.")

    if not other_cols:
        print(
            "\n⚠️ そもそも上位足の特徴量がデータセットに存在しません！（抽出リストから漏れています）"
        )
        return

    print("\n" + "=" * 60)
    print(" 🕵️‍♂️ Upper Timeframe Features Status (Sample of 5)")
    print("=" * 60)

    total_rows = df.height

    # 上位足のカラムから適当に5つピックアップして状態を確認
    for col in other_cols[:5]:
        null_count = df[col].null_count()
        # 0.0 で埋められている行数をカウント
        zero_count = df.filter(pl.col(col) == 0.0).height

        print(f"\n🔹 Feature: {col}")
        print(f"   - Total Rows : {total_rows}")
        print(
            f"   - Null Count : {null_count} ({(null_count / total_rows) * 100:.2f}%)"
        )
        print(
            f"   - Zero Count : {zero_count} ({(zero_count / total_rows) * 100:.2f}%)"
        )

        if (null_count + zero_count) == total_rows:
            print(
                "   👉 【異常検知】このカラムはすべてNullか0で埋まっています！AIが無視する原因です。"
            )
        else:
            print(
                "   ✅ このカラムには正常な値が入っています。純粋にAIが不要と判断したようです。"
            )

    print("\n" + "=" * 60)
    print(" 💡 比較用: M1 Features Status (Sample of 2)")
    print("=" * 60)
    for col in m1_cols[:2]:
        null_count = df[col].null_count()
        zero_count = df.filter(pl.col(col) == 0.0).height
        print(f"\n🔸 Feature: {col}")
        print(
            f"   - Null Count : {null_count} ({(null_count / total_rows) * 100:.2f}%)"
        )
        print(
            f"   - Zero Count : {zero_count} ({(zero_count / total_rows) * 100:.2f}%)"
        )


if __name__ == "__main__":
    main()
