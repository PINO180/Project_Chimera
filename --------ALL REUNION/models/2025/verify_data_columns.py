import polars as pl
from pathlib import Path
import sys

# --- 設定: パスはご環境に合わせて調整してください ---
BASE_DIR = Path("/workspace/data/XAUUSD/stratum_7_models/1A_2B")

# 検証対象のファイル/ディレクトリ
TARGETS = [
    {
        "name": "Script B Output (M2学習データ)",
        "path": BASE_DIR / "meta_labeled_oof_partitioned_v2",
        "is_dir": True,  # ディレクトリ（パーティション）か？
    },
    {
        "name": "Script C Output (M2予測結果)",
        "path": BASE_DIR / "m2_oof_predictions_v2.parquet",
        "is_dir": False,
    },
]

# 確認したいカラム
CHECK_COLS = ["trend_bias_25", "atr_ratio"]


def inspect_parquet(file_path, description):
    print(f"\n{'=' * 80}")
    print(f"🔍 検査対象: {description}")
    print(f"📂 パス: {file_path}")

    if not file_path.exists():
        print(f"❌ ファイル/ディレクトリが見つかりません。")
        return

    try:
        # データ読み込み
        if file_path.is_dir():
            # ディレクトリの場合は中にある最初のparquetファイルを探して読む
            files = list(file_path.glob("**/*.parquet"))
            if not files:
                print("❌ ディレクトリ内にparquetファイルが見つかりません。")
                return
            target_file = files[0]  # サンプルとして最初の1つを確認
            print(f"📄 サンプル読み込み: {target_file.name}")
            df = pl.read_parquet(target_file)
        else:
            df = pl.read_parquet(file_path)

        # カラム確認
        existing_cols = df.columns
        print(f"📊 総カラム数: {len(existing_cols)}")

        missing = []
        found = []

        for col in CHECK_COLS:
            if col in existing_cols:
                found.append(col)
            else:
                missing.append(col)

        # 結果表示
        if found:
            print(f"✅ 発見したカラム: {found}")
            print("\n--- 📝 データサンプル (先頭5行) ---")
            print(df.select(found).head())

            print("\n--- 📉 欠損値(Null)の確認 ---")
            # null_count() を表示
            print(
                df.select([pl.col(c).null_count().alias(f"{c}_nulls") for c in found])
            )

            print("\n--- 📈 統計情報 ---")
            print(df.select(found).describe())

        if missing:
            print(f"\n⚠️ 見つからなかったカラム: {missing}")
            if "Script C" in description:
                print(
                    "   (※ Script Cの出力は予測値のみ保存する仕様の場合、ここに含まれないのは正常です)"
                )
            else:
                print(
                    "   (※ Script Bの出力に含まれていない場合、データ結合設定を見直す必要があります)"
                )

    except Exception as e:
        print(f"❌ エラーが発生しました: {e}")


def main():
    print("🚀 データ検証を開始します...")
    for target in TARGETS:
        inspect_parquet(target["path"], target["name"])
    print(f"\n{'=' * 80}")
    print("検証終了")


if __name__ == "__main__":
    main()
