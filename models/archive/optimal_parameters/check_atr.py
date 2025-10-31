import polars as pl
from pathlib import Path

# ログに出力されたパスを指定
output_dir = Path(
    "/workspace/data/XAUUSD/stratum_6_training/1A_2B/weighted_dataset_partitioned"
)

# ★★★ 修正箇所 ★★★
# globパターンを "month=09" から "month=9" に修正
glob_pattern = "year=2025/month=9/**/*.parquet"
# ★★★ 修正完了 ★★★

try:
    file_to_check = next(output_dir.glob(glob_pattern))
    print(f"--- チェック対象ファイル ---")
    print(f"{file_to_check.relative_to(output_dir.parent)}\n")

    # ファイルを読み込む
    df = pl.read_parquet(file_to_check)

    # atr_value カラムの統計情報を表示
    print("--- 'atr_value' カラムの統計情報 ---")
    print(df.select("atr_value").describe())

except StopIteration:
    print(
        f"エラー: {output_dir} に 2025/09 のデータが見つかりません。(パターン: {glob_pattern})"
    )
except pl.exceptions.ColumnNotFoundError:
    print(f"エラー: {file_to_check} に 'atr_value' カラムが見つかりません。")
