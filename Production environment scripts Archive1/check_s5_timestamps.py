# /workspace/scripts/check_s5_timestamps_v2.py
import sys
from pathlib import Path
import polars as pl

# プロジェクトルートを追加
sys.path.append(str(Path(__file__).resolve().parents[1]))
from blueprint import S5_NEUTRALIZED_ALPHA_SET


def main():
    print("=" * 60)
    print("### S5 Timestamp Integrity Check V2 (Expanded) ###")
    print(f"Target: {S5_NEUTRALIZED_ALPHA_SET}")
    print("=" * 60)

    # 全ファイルを検索
    files = list(S5_NEUTRALIZED_ALPHA_SET.rglob("*.parquet"))
    if not files:
        print("No S5 files found.")
        return

    # チェック対象の時間足（ご要望の H1, D1 を追加）
    target_tfs = ["M5", "M8", "M15", "H1", "D1"]

    for target_tf in target_tfs:
        print(f"\n>>> Checking Timeframe: {target_tf} <<<")

        found_file = False

        # ファイル名に "_M15_" のようなパターンが含まれているか探す
        # (例: features_e1c_M15_neutralized.parquet)
        target_pattern = f"_{target_tf}_"

        for p_file in files:
            if target_pattern in p_file.name:
                try:
                    # データを少しだけ読む
                    lf = pl.scan_parquet(p_file)
                    # timestamp列だけ取得 (カラム名は問わない)
                    df = lf.select(["timestamp"]).head(20).collect()

                    timestamps = df["timestamp"].to_list()
                    if not timestamps:
                        continue

                    print(f"File: {p_file.name}")
                    print("Sample Timestamps (UTC):")
                    for ts in timestamps[:5]:
                        print(f"  - {ts}")

                    # 分(minute)を抽出してチェック
                    minutes = (
                        df.select(pl.col("timestamp").dt.minute()).to_series().to_list()
                    )
                    non_zero_minutes = [m for m in minutes if m != 0]

                    if non_zero_minutes:
                        print(
                            f"✅ OK: Minutes contain non-zero values: {non_zero_minutes[:10]}..."
                        )
                        print("   -> This file seems correct (Not rounded to H1).")
                    else:
                        print(
                            f"⚠️  WARNING: All timestamps align to XX:00! (Minutes: {minutes})"
                        )
                        if target_tf in ["M5", "M8", "M15"]:
                            print(
                                "   -> 🚨 CONFIRMED: This is the cause of 'Machine Gun Entry'."
                            )

                    found_file = True
                    # 1つの時間足につき1ファイル見れば十分なのでbreak
                    break

                except Exception as e:
                    print(f"Error reading {p_file.name}: {e}")
                    continue

        if not found_file:
            print(
                f"No file found for timeframe {target_tf} (Filename pattern mismatch?)"
            )


if __name__ == "__main__":
    main()
