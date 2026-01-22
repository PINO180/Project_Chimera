# /workspace/update_feature_list.py
import polars as pl
from pathlib import Path

# --- 設定 ---
# 代表的なParquetファイル（どの日のものでもOK）
# 実行中のプロセスが完了した後で、確実に存在するものを使用
source_parquet_file = Path(
    "/workspace/data/XAUUSD/stratum_6_training/1A_2B/labeled_dataset_partitioned_v2/year=2022/month=5/day=1/data.parquet"
)

# 出力する新しい特徴量リストのパス
new_feature_list_path = Path(
    "/workspace/data/XAUUSD/stratum_3_artifacts/1A_2B/train_12m_val_6m/final_feature_set_v3.txt"
)


def create_updated_feature_list(source_file: Path, output_file: Path):
    """
    ラベル付きデータセットから特徴量カラムを抽出し、新しいリストファイルを作成する
    """
    print("--- Creating Updated Feature List ---")

    if not source_file.exists():
        print(f"❌ ERROR: Source file not found: {source_file}")
        return

    try:
        # 1. データからカラム名を取得
        df_columns = pl.read_parquet_schema(source_file).keys()

        # 2. 特徴量ではない基本カラムやラベルを除外
        non_feature_cols = {
            "timestamp",
            "t1",
            "label",
            "year",
            "month",
            "day",
            "close",
            "timeframe",  # 以前手動で削除していたもの
            "payoff_ratio",  # ケリー計算/PnL計算用
            "atr_value",  # ラベル付け時の生ATR (特徴量としては冗長/リークの可能性)
            "sl_multiplier",  # ラベル付け設定パラメータ
            "pt_multiplier",  # ラベル付け設定パラメータ
            "direction",  # ラベル付け時の方向性 (現在は固定)
        }
        feature_cols = [col for col in df_columns if col not in non_feature_cols]

        # 3. 辞書順でソートして一貫性を保つ
        feature_cols.sort()

        # 4. 新しいファイルに書き出す
        output_file.parent.mkdir(parents=True, exist_ok=True)
        with open(output_file, "w") as f:
            for feature in feature_cols:
                f.write(f"{feature}\n")

        print(
            f"✅ Successfully created new feature list with {len(feature_cols)} features."
        )
        print(f"   -> Saved to: {output_file}")

    except Exception as e:
        print(f"❌ An error occurred: {e}")


if __name__ == "__main__":
    create_updated_feature_list(source_parquet_file, new_feature_list_path)
