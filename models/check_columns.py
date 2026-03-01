import polars as pl
from pathlib import Path
import sys

# 調査対象のディレクトリリスト
TARGET_DIRS = [
    "/workspace/data/XAUUSD/stratum_6_training/1A_2B/labeled_dataset_partitioned_v2",
    "/workspace/data/XAUUSD/stratum_6_training/1A_2B/labeled_dataset_monthly_v2",
    "/workspace/data/XAUUSD/stratum_6_training/1A_2B/weighted_dataset_partitioned_v2",
    "/workspace/data/XAUUSD/stratum_6_training/1A_2B/20260211ATR廃止前/weighted_dataset_partitioned_v2",
]

# 探したい重要カラム（存在確認の対象）
KEY_COLUMNS = [
    "atr_value",
    "e1c_atr_21",
    "open",
    "high",
    "low",
    "close",
    "timestamp",
    "label",
]


def check_directory(dir_path_str):
    path = Path(dir_path_str)
    print(f"\n{'=' * 60}")
    print(f"📂 調査ディレクトリ: {path}")

    if not path.exists():
        print("❌ [ERROR] ディレクトリが存在しません。")
        return

    # 再帰的にParquetファイルを探す
    files = list(path.glob("**/*.parquet"))
    if not files:
        print("⚠️ [WARNING] Parquetファイルが見つかりません。")
        return

    print(f"   -> 発見ファイル数: {len(files)} 件")

    # 最初のファイルをサンプルとして読み込む
    sample_file = files[0]
    print(f"📄 サンプル読込: {sample_file.relative_to(path)}")

    try:
        # スキーマ（カラム一覧）だけを取得（高速）
        schema = pl.scan_parquet(str(sample_file)).collect_schema()
        all_cols = schema.names()

        # 重要カラムのチェック
        print(f"\n🔍 【重要カラムの生存確認】")
        found_any_atr = False
        for key in KEY_COLUMNS:
            if key in all_cols:
                print(f"   ✅ {key:<15}: 存在する")
                if "atr" in key:
                    found_any_atr = True
            else:
                print(f"   ❌ {key:<15}: 無い")

        # ATRっぽい名前のカラムが他にあるか検索
        atr_candidates = [
            c for c in all_cols if "atr" in c.lower() and c not in KEY_COLUMNS
        ]
        if atr_candidates:
            print(f"\n💡 その他 'atr' を含むカラム: {atr_candidates}")

        # 結論
        if found_any_atr:
            print("\n👉 結論: ATRデータは生きています。")
        else:
            print("\n👉 結論: ATRデータは見当たりません。")

    except Exception as e:
        print(f"❌ [ERROR] ファイル読み込みエラー: {e}")


if __name__ == "__main__":
    print("🚀 カラム生存確認を開始します...")
    for d in TARGET_DIRS:
        check_directory(d)
    print(f"\n{'=' * 60}")
    print("🏁 確認終了")
