import polars as pl
from pathlib import Path
import sys
import warnings

# Polarsの警告を抑制
warnings.filterwarnings("ignore", category=DeprecationWarning)

# パス定義
BASE_DIR = Path("/workspace/data/XAUUSD/stratum_7_models/1A_2B")
M1_OOF_PATH = BASE_DIR / "m1_oof_predictions_v2.parquet"
CONTEXT_PATH = BASE_DIR / "context_features_v2.parquet"
META_LABELED_PATH = BASE_DIR / "meta_labeled_oof_partitioned_v2"


def check_path_robust(path, name):
    print(f"\n🔍 検査: {name}")
    if not path.exists():
        print(f"   ⚠️ パスが見つかりません: {path}")
        return

    try:
        if path.is_dir():
            # ディレクトリの場合
            # missing_columns="insert": 足りない列にはnullを入れる（旧 allow_missing_columns=True）
            # extra_columns="ignore": 余計な列があっても無視する（今回のエラー対策）
            df = (
                pl.scan_parquet(
                    str(path / "**/*.parquet"),
                    missing_columns="insert",
                    extra_columns="ignore",
                )
                .select("timestamp")
                .collect()
            )
        else:
            # 単一ファイルの場合
            df = pl.read_parquet(path, columns=["timestamp"])

        rows = df.height
        print(f"   総行数: {rows}")

        # 重複チェック
        dup_check = df.group_by("timestamp").len().filter(pl.col("len") > 1)

        if dup_check.height > 0:
            max_dup = dup_check["len"].max()
            print(f"   ❌ 【汚染あり】 最大重複数: {max_dup} 行")
            if max_dup >= 5:
                print(f"      👉 警告: {max_dup}倍に増殖しています。")
        else:
            print(f"   ✅ 正常 (重複なし)")

    except Exception as e:
        print(f"   エラー: {e}")


if __name__ == "__main__":
    # Script Aの出力
    check_path_robust(M1_OOF_PATH, "Script A 出力 (M1 OOF)")

    # 文脈データ
    check_path_robust(CONTEXT_PATH, "文脈データ (Context Features)")

    # Script Bの出力（エラーが出ていた箇所）
    check_path_robust(META_LABELED_PATH, "Script B 出力 (Meta Labeled)")
