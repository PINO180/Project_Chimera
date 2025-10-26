import polars as pl
from pathlib import Path

# --- 設定 ---
# 調査対象のParquetファイル
file_to_check = Path(
    "/workspace/data/XAUUSD/stratum_6_training/1A_2B/weighted_dataset_partitioned/year=2021/month=8/day=1/data.parquet"
)

# 比較元の特徴量リストファイル
feature_list_file = Path(
    "/workspace/data/XAUUSD/stratum_3_artifacts/1A_2B/train_12m_val_6m/final_feature_set.txt"
)


def analyze_parquet_features(parquet_path: Path, expected_features_path: Path):
    """
    Parquetファイル内の特徴量カラムを読み込み、期待される特徴量リストと比較・分析する
    """
    print(f"--- Analyzing Feature Set in: {parquet_path.name} ---")

    if not parquet_path.exists():
        print(f"❌ ERROR: File not found at {parquet_path}")
        return

    if not expected_features_path.exists():
        print(f"❌ ERROR: Feature list not found at {expected_features_path}")
        return

    try:
        # 1. 期待される特徴量リストを読み込み、セットに変換
        with open(expected_features_path, "r") as f:
            # .txtファイルから読み込む際、末尾の改行などを除去
            expected_features = {line.strip() for line in f if line.strip()}

        # 2. Parquetファイルを読み込み、そのカラムをセットに変換
        df = pl.read_parquet(parquet_path)
        actual_features = set(df.columns)

        # 3. 比較分析
        # 基本カラムやラベルなど、特徴量リストに含まれないものを除外
        base_and_label_cols = {"timestamp", "t1", "label", "year", "month", "day"}
        actual_feature_only_set = actual_features - base_and_label_cols

        # 集合演算で差分を計算
        missing_features = expected_features - actual_feature_only_set
        extra_features = actual_feature_only_set - expected_features

        # 4. 結果の表示
        print(f"✅ File loaded successfully.")
        print(f"   - Total columns in Parquet: {len(df.columns)}")
        print(f"   - Expected features in list: {len(expected_features)}")
        print(f"   - Actual features in Parquet: {len(actual_feature_only_set)}")

        print("\n--- Comparison Results ---")
        if not missing_features and not extra_features:
            print("✅ PERFECT MATCH! The feature sets are identical.")
        else:
            if missing_features:
                print(f"\n⚠️ Missing Features ({len(missing_features)}):")
                for i, feat in enumerate(sorted(list(missing_features))):
                    print(f"   {i + 1:03d}: {feat}")

            if extra_features:
                print(f"\n⚠️ Extra Features ({len(extra_features)}):")
                for i, feat in enumerate(sorted(list(extra_features))):
                    print(f"   {i + 1:03d}: {feat}")

    except Exception as e:
        print(f"❌ An error occurred during analysis: {e}")


# スクリプトの実行
if __name__ == "__main__":
    analyze_parquet_features(file_to_check, feature_list_file)
