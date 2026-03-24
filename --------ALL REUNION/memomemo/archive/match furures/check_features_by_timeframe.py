import polars as pl
import pandas as pd
from pathlib import Path
from collections import Counter

# --- 設定 ---
# 調査対象のParquetファイル
file_to_check = Path(
    "/workspace/data/XAUUSD/stratum_6_training/1A_2B/labeled_dataset_partitioned/year=2022/month=5/day=1/data.parquet"
)

# 比較元の特徴量リストファイル (正しいパスに修正済み)
feature_list_file = Path(
    "/workspace/data/XAUUSD/stratum_3_artifacts/1A_2B/train_12m_val_6m/final_feature_set.txt"
)

# プロジェクトで定義されているタイムフレームのリスト
VALID_TIMEFRAMES = [
    "tick",
    "M0.5",
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
    "W1",
    "MN",
]


def extract_timeframe(feature_name: str) -> str:
    """特徴量名からタイムフレームのサフィックスを抽出する"""
    parts = feature_name.split("_")
    # 末尾の要素がタイムフレームリストにあればそれを返す
    if parts[-1] in VALID_TIMEFRAMES:
        return parts[-1]
    return "N/A"  # タイムフレームが見つからない場合


def analyze_timeframe_counts(parquet_path: Path, expected_features_path: Path):
    """
    Parquetファイルと特徴量リストを読み込み、タイムフレームごとの特徴量数を比較する
    """
    print(f"--- Analyzing Feature Counts by Timeframe ---")

    if not parquet_path.exists():
        print(f"❌ ERROR: File not found at {parquet_path}")
        return
    if not expected_features_path.exists():
        print(f"❌ ERROR: Feature list not found at {expected_features_path}")
        return

    try:
        # 1. 期待される特徴量リストを読み込み、タイムフレーム別に集計
        with open(expected_features_path, "r") as f:
            expected_features = [line.strip() for line in f if line.strip()]
        expected_timeframes = [extract_timeframe(name) for name in expected_features]
        expected_counts = Counter(expected_timeframes)

        # 2. Parquetファイルを読み込み、実際のカラムからタイムフレーム別に集計
        actual_columns = pl.read_parquet_schema(parquet_path).keys()
        non_feature_cols = {"timestamp", "t1", "label", "year", "month", "day"}
        actual_features = [col for col in actual_columns if col not in non_feature_cols]
        actual_timeframes = [extract_timeframe(name) for name in actual_features]
        actual_counts = Counter(actual_timeframes)

        # 3. Pandas DataFrameで比較表を作成
        df_expected = pd.DataFrame(
            expected_counts.items(), columns=["Timeframe", "Expected_Count"]
        ).set_index("Timeframe")
        df_actual = pd.DataFrame(
            actual_counts.items(), columns=["Timeframe", "Actual_Count"]
        ).set_index("Timeframe")

        # 両方のデータを結合
        comparison_df = (
            pd.concat([df_expected, df_actual], axis=1).fillna(0).astype(int)
        )
        comparison_df = (
            comparison_df.reindex(VALID_TIMEFRAMES).fillna(0).astype(int)
        )  # プロジェクトの順序にソート

        # 差分を計算
        comparison_df["Difference"] = (
            comparison_df["Actual_Count"] - comparison_df["Expected_Count"]
        )

        # 4. 結果表示
        print("\n✅ Analysis complete. Here is the breakdown:\n")
        print(comparison_df.to_string())

        total_diff = comparison_df["Difference"].abs().sum()
        if total_diff == 0:
            print("\n✅ PERFECT MATCH! All timeframe counts are identical.")
        else:
            print(
                f"\n⚠️ MISMATCH FOUND! The counts for one or more timeframes do not match."
            )
            print("Please review the 'Difference' column in the table above.")

    except Exception as e:
        print(f"❌ An error occurred during analysis: {e}")


# スクリプトの実行
if __name__ == "__main__":
    analyze_timeframe_counts(file_to_check, feature_list_file)
