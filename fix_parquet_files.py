import polars as pl
from pathlib import Path
from tqdm import tqdm
import shutil

SOURCE_PARENT_DIR = Path("/workspace/data/XAUUSD/stratum_2_features/feature_value_a_vast_universeA")
DEST_PARENT_DIR = Path("/workspace/data/XAUUSD/stratum_2_features_fixed/feature_value_a_vast_universeA")

def convert_parquet_file(source_file: Path, dest_file: Path):
    """
    Polarsで読み込み、pyarrow互換・高性能形式で書き出す
    
    重要: 辞書エンコーディングを完全に無効化し、
    int64ベースの形式で保存する
    """
    try:
        # Polarsで読み込み（uint32インデックスも問題なく読める）
        df = pl.read_parquet(source_file)
        
        # 全ての文字列・カテゴリカル列を通常の文字列型に変換
        # これにより辞書エンコーディング自体を回避
        for col in df.columns:
            if df[col].dtype == pl.Categorical:
                df = df.with_columns(pl.col(col).cast(pl.Utf8))
        
        # pyarrow互換形式で書き出し
        df.write_parquet(
            dest_file,
            compression='snappy',  # pyarrowと最も互換性が高い
            use_pyarrow=True,
            pyarrow_options={
                "coerce_timestamps": "us",  # タイムスタンプをマイクロ秒に統一
                "data_page_size": 1024*1024,  # 1MB（読み込み最適化）
            }
        )
        return True
    except Exception as e:
        print(f"\nエラー [{source_file.name}]: {e}")
        return False

def main():
    print("=" * 60)
    print("Project Forge データ修復（pyarrow互換版）")
    print("=" * 60)
    print(f"入力: {SOURCE_PARENT_DIR}")
    print(f"出力: {DEST_PARENT_DIR}\n")

    if not SOURCE_PARENT_DIR.exists():
        print(f"エラー: 入力ディレクトリが存在しません")
        return

    if DEST_PARENT_DIR.exists():
        print("既存の出力ディレクトリを削除中...")
        shutil.rmtree(DEST_PARENT_DIR)
    DEST_PARENT_DIR.mkdir(parents=True)

    all_parquet_files = list(SOURCE_PARENT_DIR.glob("**/*.parquet"))
    
    if not all_parquet_files:
        print(f"エラー: Parquetファイルが見つかりません")
        return

    print(f"{len(all_parquet_files)}個のファイルを修復します\n")

    success_count = 0
    fail_count = 0
    
    for source_file in tqdm(all_parquet_files, desc="修復中"):
        relative_path = source_file.relative_to(SOURCE_PARENT_DIR)
        dest_file = DEST_PARENT_DIR / relative_path
        dest_file.parent.mkdir(parents=True, exist_ok=True)
        
        if convert_parquet_file(source_file, dest_file):
            success_count += 1
        else:
            fail_count += 1

    print("\n" + "=" * 60)
    print(f"修復完了: 成功 {success_count}件, 失敗 {fail_count}件")
    print(f"出力先: {DEST_PARENT_DIR}")
    print("=" * 60)

if __name__ == "__main__":
    main()