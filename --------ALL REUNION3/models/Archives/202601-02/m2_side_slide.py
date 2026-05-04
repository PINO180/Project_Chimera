import polars as pl
from pathlib import Path
import sys

# --- プロジェクトのルートパス設定 ---
PROJECT_ROOT = Path("/workspace")
SYMBOL = "XAUUSD"
FEATURE_SET_ID = "1A_2B"

# --- パス定義 (blueprint.py 準拠) ---
# Stratum 6: 訓練データ (M1の入力)
S6_TRAINING_DIR = PROJECT_ROOT / "data" / SYMBOL / "stratum_6_training" / FEATURE_SET_ID
S6_DATASET_PATH = S6_TRAINING_DIR / "weighted_dataset_partitioned_v2"

# Stratum 7: AIモデル成果物 (M1出力 & M2入力)
S7_MODELS_DIR = PROJECT_ROOT / "data" / SYMBOL / "stratum_7_models" / FEATURE_SET_ID
S7_M1_OOF_PATH = S7_MODELS_DIR / "m1_oof_predictions_v2.parquet"
S7_M2_INPUT_PATH = S7_MODELS_DIR / "meta_labeled_oof_partitioned_v2"


def inspect_parquet_structure(path: Path, name: str):
    print(f"\n{'=' * 20} {name} {'=' * 20}")

    if not path.exists():
        print(f"❌ Path not found: {path}")
        return None

    try:
        # ディレクトリ(パーティション)か単一ファイルかで読み込みを変える
        if path.is_dir():
            # 再帰的に最初の.parquetファイルを探す
            first_file = next(path.rglob("*.parquet"), None)
            if not first_file:
                print("❌ No parquet files found in directory.")
                return None
            # パーティションデータとしてスキャン
            df = pl.scan_parquet(path / "**/*.parquet")
            print(f"📂 Type: Partitioned Directory (Found: {first_file.name})")
        else:
            df = pl.scan_parquet(path)
            print(f"📄 Type: Single File")

        # スキーマ取得
        schema = df.collect_schema()
        columns = schema.names()

        print(f"📊 Column Count: {len(columns)}")
        print(f"📋 Columns (First 10): {columns[:10]} ...")

        # 特徴的なカラムの有無をチェック
        if "prediction" in columns or "m1_proba" in columns:
            print("ℹ️  Contains Prediction column.")

        return set(columns)

    except Exception as e:
        print(f"⚠️ Error reading file: {e}")
        return None


def main():
    print("Checking Data Combination Logic (Horizontal vs Vertical)...")
    print(f"Target Feature Set: {FEATURE_SET_ID}")

    # 1. M1の入力データ (S6)
    cols_s6 = inspect_parquet_structure(S6_DATASET_PATH, "Step 1: M1 Input (S6)")

    # 2. M1の出力データ (OOF)
    cols_m1_oof = inspect_parquet_structure(S7_M1_OOF_PATH, "Step 2: M1 Output (OOF)")

    # 3. M2の入力データ (S7 Meta Labeled)
    cols_m2_input = inspect_parquet_structure(
        S7_M2_INPUT_PATH, "Step 3: M2 Input (Meta Labeled)"
    )

    # 比較判定
    if cols_s6 and cols_m2_input:
        print(f"\n{'=' * 20} DIAGNOSIS {'=' * 20}")

        len_s6 = len(cols_s6)
        len_m2 = len(cols_m2_input)

        print(f"M1 Input Columns: {len_s6}")
        print(f"M2 Input Columns: {len_m2}")

        # M2入力のカラム数がS6より多いか確認
        if len_m2 > len_s6:
            diff = len_m2 - len_s6
            print(f"✅ [JUDGMENT: M2 uses HORIZONTAL JOIN]")
            print(f"   M2 Input has {diff} more columns than M1 Input.")

            # 差分カラムの特定
            new_cols = cols_m2_input - cols_s6
            print(f"   New Columns (Example): {list(new_cols)[:5]} ...")

            if "prediction" in new_cols or "m1_proba" in new_cols:
                print("   -> Confirmed: M1 predictions are added as new features.")
        else:
            print(
                f"⚠️ Column counts are similar. M2 might be using only M1 output or raw features without join."
            )


if __name__ == "__main__":
    main()
