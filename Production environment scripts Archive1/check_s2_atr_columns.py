# /workspace/scripts/check_s2_atr_columns.py
import sys
from pathlib import Path
import polars as pl
import re

# プロジェクトルートを追加してblueprintを読み込めるようにする
sys.path.append(str(Path(__file__).resolve().parents[1]))

from blueprint import S2_FEATURES_FIXED


def main():
    print("=" * 60)
    print("### S2 Data Column Inspection: ATR Existence Check ###")
    print(f"Target Directory: {S2_FEATURES_FIXED}")
    print("=" * 60)

    # Cat C (Universe C) がATRを含んでいるはずの場所
    # フォルダ名やファイル構成は `05_alpha_decay_analyzer.py` の情報を元に推定
    target_dir = S2_FEATURES_FIXED / "feature_value_a_vast_universeC"

    if not target_dir.exists():
        print(f"Error: Directory not found: {target_dir}")
        return

    # チェック対象のParquetファイルを検索
    parquet_files = list(target_dir.rglob("*.parquet"))
    print(f"Found {len(parquet_files)} parquet files in Universe C.")

    # ATRカラム（e1c_atr_21_XXX）を探すための正規表現
    atr_pattern = re.compile(r"e1c_atr_21_([a-zA-Z0-9]+)")

    found_timeframes = set()
    missing_timeframes_files = []

    print("\n--- Scanning Columns ---")
    for p_file in parquet_files:
        try:
            # スキーマ（列名）のみを高速に読み込む
            schema = pl.read_parquet_schema(p_file)
            cols = schema.keys()

            # このファイルに含まれる ATR 列を探す
            atr_cols = [c for c in cols if "e1c_atr_21" in c]

            if atr_cols:
                print(f"\n[File] {p_file.name}")
                for c in atr_cols:
                    match = atr_pattern.search(c)
                    if match:
                        tf = match.group(1)
                        print(f"  - Found: {c} (Timeframe: {tf})")
                        found_timeframes.add(tf)
                    else:
                        # e1c_atr_21 そのもの（恐らくBaseのM1かTick）
                        print(f"  - Found: {c} (Base/Unknown)")
            else:
                # ATRが含まれていないファイル
                # print(f".", end="", flush=True)
                pass

        except Exception as e:
            print(f"Error reading {p_file.name}: {e}")

    print("\n\n" + "=" * 60)
    print("### SUMMARY ###")
    print(f"Timeframes with ATR columns found in S2: {sorted(list(found_timeframes))}")

    expected_timeframes = [
        "M1",
        "M3",
        "M5",
        "M8",
        "M15",
        "M30",
        "H1",
        "H4",
        "H6",
        "H12",
        "D1",
    ]
    missing = [tf for tf in expected_timeframes if tf not in found_timeframes]

    if missing:
        print(f"⚠️  MISSING Timeframes (Likely causing S6 drop): {missing}")
    else:
        print("✅ All expected timeframes have ATR columns.")
    print("=" * 60)


if __name__ == "__main__":
    main()
