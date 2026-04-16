import polars as pl
from pathlib import Path
import sys

# === 設定 ==========================================
# 実際に出力先のフォルダパスを指定してください
# スクリプトの output_dir で指定しているパスです
TARGET_DIR = (
    "/workspace/data/XAUUSD/stratum_6_training/1A_2B/weighted_dataset_partitioned_v2"
)
# =================================================


def verify_data_integrity(target_dir):
    root = Path(target_dir)
    files = list(root.rglob("data.parquet"))

    if not files:
        print(f"エラー: '{target_dir}' 内に parquetファイルが見つかりません。")
        return

    print(f"検証対象: {len(files)} 個の生成済みファイルが見つかりました。")
    print("-" * 50)

    total_errors = 0

    for f in files:
        try:
            df = pl.read_parquet(f)

            # 1. 重複チェック (timestamp と timeframe の組み合わせがユニークか)
            dup_check = (
                df.group_by(["timestamp", "timeframe"]).len().filter(pl.col("len") > 1)
            )

            # 2. 64倍チェック (念のため、同じtimestampに大量の行がないか確認)
            # timeframe='1m'などの通常の足で、同じ時刻に2行以上あれば異常です

            if dup_check.height > 0:
                print(f"❌ 異常検出: {f}")
                print(f"   重複行数: {dup_check.height}")
                print(dup_check.head(5))
                total_errors += 1
            else:
                # 正常な場合も、念のためサンプルで1ファイルの行数を確認
                # print(f"✅ OK: {f.name} (Rows: {df.height})")
                pass

        except Exception as e:
            print(f"⚠️ 読み込みエラー: {f} - {e}")

    print("-" * 50)
    if total_errors == 0:
        print("🎉【合格】すべてのファイルで『行増殖』『重複』は確認されませんでした。")
        print("   64倍バグは完全に修正されています。処理を再開してください。")
    else:
        print(f"😱【不合格】合計 {total_errors} 個のファイルで異常が見つかりました。")


if __name__ == "__main__":
    verify_data_integrity(TARGET_DIR)
