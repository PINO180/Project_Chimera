# /workspace/update_feature_list_v5.py
import sys
import polars as pl
from pathlib import Path

# プロジェクトのルートディレクトリをPythonの検索パスに追加
sys.path.append(str(Path(__file__).resolve().parents[1]))

from blueprint import S6_WEIGHTED_DATASET, S3_FEATURES_FOR_TRAINING_V5


def find_latest_partition(base_dir: Path) -> Path:
    """最新の日次パーティションファイルを動的に検索する"""
    parquet_files = sorted(base_dir.rglob("*.parquet"), reverse=True)
    if not parquet_files:
        raise FileNotFoundError(
            f"S6_WEIGHTED_DATASETにparquetファイルが見つかりません: {base_dir}"
        )
    return parquet_files[0]


# --- 設定 ---
# S6_WEIGHTED_DATASETから最新パーティションを動的に検索する
source_parquet_file = find_latest_partition(S6_WEIGHTED_DATASET)

# 出力する新しい特徴量リストのパス
new_feature_list_path = S3_FEATURES_FOR_TRAINING_V5


def create_updated_feature_list(source_file: Path, output_file: Path):
    """
    ラベル付きデータセットから特徴量カラムを抽出し、新しいリストファイルを作成する
    """
    print("--- Creating Updated Feature List (V5 Project Cimera) ---")

    if not source_file.exists():
        print(f"❌ ERROR: Source file not found: {source_file}")
        return

    try:
        # 1. データからカラム名を取得
        df_columns = pl.read_parquet_schema(source_file).keys()

        # 2. 特徴量ではない基本カラムやメタデータ（V5対応）を定義
        non_feature_cols = {
            # --- 共通基本カラム ---
            "timestamp",
            "year",
            "month",
            "day",
            "timeframe",
            # --- V5 必須メタデータ ---
            "is_trigger",  # エントリータイミングのフラグ
            "close",  # エントリー価格 (シミュレーター用)
            "atr_value",  # 動的SL幅計算基準 (シミュレーター用)
            "atr_ratio",  # ATR Ratio（シミュレーター用・再計算不要）
            # --- V5 双方向ラベリング (Long用) ---
            "label_long",
            "duration_long",
            "uniqueness_long",
            "concurrency_long",  # <--- 追加！(未来情報のカンニング防止)
            "sl_multiplier_long",  # (将来的な拡張用)
            "pt_multiplier_long",  # (将来的な拡張用)
            "payoff_ratio_long",  # (将来的な拡張用)
            # --- V5 双方向ラベリング (Short用) ---
            "label_short",
            "duration_short",
            "uniqueness_short",
            "concurrency_short",  # <--- 追加！(未来情報のカンニング防止)
            "sl_multiplier_short",  # (将来的な拡張用)
            "pt_multiplier_short",  # (将来的な拡張用)
            "payoff_ratio_short",  # (将来的な拡張用)
            # --- 旧仕様の残骸 (フェイルセーフ用) ---
            "t1",
            "label",
            "uniqueness",
            "payoff_ratio",
            "sl_multiplier",
            "pt_multiplier",
            "direction",
            # --- 不要な残留価格データと生ATR ---
            "open",
            "high",
            "low",
            "e1c_atr_13_M1",
            # --- Chapter2 LFスコア（特徴量ではなくスコア系・除外必須）---
            "lf_short_score",
            "lf_mid_score",
            "lf_long_score",
        }

        # 動的除外ルールの適用 (完全一致だけでなく、前方一致等も弾く)
        feature_cols = []
        for col in df_columns:
            if col in non_feature_cols:
                continue
            # 動的に生成されるトリガーフラグ系なども特徴量から除外
            if col.startswith("is_trigger_on"):
                continue
            feature_cols.append(col)

        # =====================================================================
        # 【緊急防壁】未来情報（データリーク）を含む汚染特徴量の強制除外
        # =====================================================================
        # startswithで一括判定するため、必ずタプル () で定義してください
        LEAKED_PREFIXES = ("e1a_fast_basic_stabilization", "e1c_dpo", "e1d_hv_regime")

        original_count = len(feature_cols)

        # 指定したプレフィックスから始まるカラムをリストから一掃する
        feature_cols = [
            col for col in feature_cols if not col.startswith(LEAKED_PREFIXES)
        ]

        removed_count = original_count - len(feature_cols)
        if removed_count > 0:
            print(
                f"🚨 SECURITY WARNING: Removed {removed_count} leaked features from the training list!"
            )
        # =====================================================================

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
