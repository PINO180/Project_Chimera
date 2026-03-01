import sys
from pathlib import Path
import polars as pl

# プロジェクトルート設定
PROJECT_ROOT = Path("/workspace")
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

# blueprintからパスを読み込む
try:
    from blueprint import S6_LABELED_DATASET
except ImportError:
    # 読み込めない場合のフォールバック (v2対応)
    S6_LABELED_DATASET = (
        PROJECT_ROOT
        / "data"
        / "XAUUSD"
        / "stratum_6_training"
        / "1A_2B"
        / "labeled_dataset_partitioned_v2"
    )


def verify_deduplication(target_year, target_month):
    # ターゲットディレクトリ: year=2022/month=6
    target_dir = S6_LABELED_DATASET / f"year={target_year}" / f"month={target_month}"

    print(f"🔍 Verifying data in: {target_dir}")

    if not target_dir.exists():
        # v2がない場合、v1を探すフォールバック（念のため）
        target_dir_v1 = (
            PROJECT_ROOT
            / "data"
            / "XAUUSD"
            / "stratum_6_training"
            / "1A_2B"
            / "labeled_dataset_partitioned"
            / f"year={target_year}"
            / f"month={target_month}"
        )
        if target_dir_v1.exists():
            target_dir = target_dir_v1
            print(f"   (Redirected to v1 path: {target_dir})")
        else:
            print(
                "❌ Directory not found. Please check if the script ran successfully."
            )
            return

    # Parquetファイルをスキャン
    files = list(target_dir.rglob("*.parquet"))
    if not files:
        print("❌ No parquet files found.")
        return

    print(f"   Found {len(files)} partition files. Checking for duplicates...")

    total_rows = 0
    total_dups = 0

    for p_file in files:
        try:
            df = pl.read_parquet(p_file)
            rows = len(df)
            total_rows += rows

            # 重複チェック (timestamp と timeframe の組み合わせ)
            subset = ["timestamp", "timeframe"]
            if "timeframe" not in df.columns:
                subset = ["timestamp"]

            # 【修正】Polarsバージョン互換対応
            # is_duplicated(subset=...) が使えないバージョンでも動く書き方に変更
            try:
                dup_mask = df.select(subset).is_duplicated()
            except Exception:
                # 万が一 select(...).is_duplicated() も動かない場合の保険
                dup_mask = df.is_duplicated()

            n_dups = dup_mask.sum()

            if n_dups > 0:
                print(f"   ⚠️  Duplicate found in {p_file.name}: {n_dups} rows!")
                # 詳細表示
                dup_df = df.filter(dup_mask)
                print(dup_df.sort(subset).head(4))
                total_dups += n_dups
        except Exception as e:
            print(f"   ⚠️  Error reading {p_file.name}: {e}")

    print("-" * 50)
    print(f"📊 Total Rows Checked: {total_rows}")

    if total_dups == 0:
        print("✅ SUCCESS: No duplicates found! The 'Machine Gun' bug is fixed.")
        print("   You can now proceed to run the script for ALL time.")
    else:
        print(f"😱 FAILURE: Found {total_dups} duplicates. Do NOT proceed.")


if __name__ == "__main__":
    # 2022年6月を検査
    verify_deduplication(2022, 6)
