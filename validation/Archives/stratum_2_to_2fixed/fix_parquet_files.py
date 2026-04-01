# 修正後全文 (fix_parquet_files.py - 単一ファイル処理・チャンク・ハードコード版)
import polars as pl
from pathlib import Path
import time
import shutil
import pyarrow as pa
import pyarrow.parquet as pq
import logging  # ログ出力を追加
from typing import Optional
import gc

# --- ログ設定 ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(
            "fix_parquet_files_chunked.log", encoding="utf-8"
        ),  # ログファイル名変更
    ],
)
logger = logging.getLogger(__name__)

# --- 設定 (ファイルパスを直接指定) ---
SOURCE_FILE = Path(
    "/workspace/data/XAUUSD/stratum_2_features/features_e1c_tick_atr_only_tick.parquet"
)
DEST_FILE = Path(
    "/workspace/data/XAUUSD/stratum_2_features_fixed/features_e1c_tick_atr_only_tick_fixed.parquet"
)

# --- チャンクパラメータ ---
CHUNK_SIZE = 1_000_000  # 100万行ごと (メモリに応じて調整)


def convert_parquet_file_chunked(source_file: Path, dest_file: Path, chunk_size: int):
    """
    巨大Parquetファイルをチャンクで読み込み、型変換・タイムスタンプ調整を行い、
    PyArrow ParquetWriterで追記していく。
    """
    writer: Optional[pq.ParquetWriter] = None
    arrow_schema_written: Optional[pa.Schema] = None
    total_rows_processed = 0

    try:
        logger.info(f"入力ファイルを開いています (pyarrow): {source_file}")
        parquet_file = pq.ParquetFile(source_file)
        num_row_groups = parquet_file.num_row_groups  # 参考情報
        total_rows = parquet_file.metadata.num_rows
        logger.info(
            f"ファイル情報: 総行数={total_rows:,}, 行グループ数={num_row_groups}"
        )
        logger.info(f"チャンクサイズ: {chunk_size:,} 行")

        # iter_batches でチャンクごとに読み込み
        batch_iterator = parquet_file.iter_batches(batch_size=chunk_size)
        num_chunks = (total_rows + chunk_size - 1) // chunk_size  # チャンク数の概算

        for i, batch in enumerate(batch_iterator):
            chunk_start_time = time.time()
            logger.info(
                f"チャンク {i + 1}/{num_chunks} を処理中 (行数: {batch.num_rows:,})..."
            )

            if batch is None or batch.num_rows == 0:
                logger.info("空のバッチをスキップしました。")
                continue

            # ArrowバッチをPolars DataFrameに変換
            df_chunk = pl.from_arrow(batch)

            # --- 型変換 ---
            converted_cols = 0
            schema = df_chunk.schema
            update_expressions = []
            for col_name, col_dtype in schema.items():
                if col_dtype == pl.Categorical:
                    update_expressions.append(pl.col(col_name).cast(pl.Utf8))
                    converted_cols += 1

            if update_expressions:
                df_chunk = df_chunk.with_columns(update_expressions)
                # logger.debug(f"チャンク {i+1}: {converted_cols} 個のカテゴリカル列を Utf8 に変換。") # DEBUGレベルに変更
            # --- 型変換完了 ---

            # --- PyArrow Writer への書き込み ---
            # Polars DataFrameをArrow Tableに変換
            arrow_table = df_chunk.to_arrow()

            if writer is None:
                # 最初のチャンクでスキーマを取得し、Writerを開く
                # ★重要★: PyArrow書き込み時にタイムスタンプを 'us' に変換するスキーマを準備
                original_schema = arrow_table.schema
                fields_coerced = []
                for field in original_schema:
                    if pa.types.is_timestamp(field.type):
                        # 既存のタイムゾーン情報を保持しつつ単位を 'us' に変更
                        tz = field.type.tz
                        fields_coerced.append(
                            pa.field(field.name, pa.timestamp("us", tz=tz))
                        )
                        logger.info(
                            f"タイムスタンプ列 '{field.name}' を 'us' 単位に設定しました。"
                        )
                    else:
                        fields_coerced.append(field)
                arrow_schema_written = pa.schema(fields_coerced)

                logger.info(
                    f"最初のチャンクからスキーマを取得し、ParquetWriter を開きます: {dest_file}"
                )
                dest_file.parent.mkdir(
                    parents=True, exist_ok=True
                )  # 親ディレクトリ作成
                writer = pq.ParquetWriter(
                    dest_file,
                    arrow_schema_written,
                    compression="snappy",
                    # PyArrow Writer のオプション (必要に応じて追加)
                    # data_page_size=1024*1024,
                )

            # スキーマの一貫性を確認 (変換後のスキーマと比較)
            # Polars -> Arrow 変換後のスキーマと、Writerを開いたスキーマを比較
            # coerce_timestampsがあるので完全一致はしない。カラム名と基本的な型で比較
            if set(arrow_table.schema.names) != set(arrow_schema_written.names):
                logger.error(
                    "エラー: チャンク間でカラム名が異なります。処理を中断します。"
                )
                logger.error(f"期待されたカラム: {set(arrow_schema_written.names)}")
                logger.error(f"現在のカラム: {set(arrow_table.schema.names)}")
                raise ValueError("Schema name mismatch between chunks")
            # より詳細な型チェックが必要な場合はここに追加

            # Arrow TableをParquetファイルに書き込む (書き込み時にタイムスタンプ変換が適用される)
            # writer.write_table() に渡すテーブルは *変換前* のスキーマを持つもので良い
            # PyArrow Writer が coerce_timestamps の設定に従い処理する (はずだが、明示的にキャストする方が安全かも)

            # --- 安全策: Polars側で明示的にキャスト ---
            cast_expressions = []
            for field in arrow_schema_written:
                if pa.types.is_timestamp(field.type) and field.type.unit == "us":
                    # Polarsで datetime('us') にキャスト
                    # タイムゾーンはPolarsが自動で扱うか、明示的に指定する必要がある
                    # ここでは一旦タイムゾーン指定なしで試す
                    cast_expressions.append(pl.col(field.name).cast(pl.Datetime("us")))

            if cast_expressions:
                df_chunk_casted = df_chunk.with_columns(cast_expressions)
                arrow_table_casted = df_chunk_casted.to_arrow()
                writer.write_table(arrow_table_casted)
                del df_chunk_casted, arrow_table_casted  # メモリ解放
            else:
                # タイムスタンプ列がないか、既に 'us' だった場合
                writer.write_table(arrow_table)

            total_rows_processed += arrow_table.num_rows
            # --- 書き込み完了 ---

            del df_chunk, arrow_table  # 明示的にメモリ解放
            import gc  # ★★★ 関数内で gc をインポート ★★★

            gc.collect()  # ガベージコレクションを強制

            chunk_elapsed_time = time.time() - chunk_start_time
            logger.info(
                f"チャンク {i + 1}/{num_chunks} 処理完了 ({chunk_elapsed_time:.2f}秒)。 {batch.num_rows:,}行を追記。"
            )

        # --- 全チャンク処理完了後 ---
        logger.info(
            f"全 {num_chunks} 個のチャンク処理完了。合計 {total_rows_processed:,} 行処理。"
        )

        if writer:
            logger.info("ParquetWriter を閉じています...")
            writer.close()
            logger.info("ParquetWriter を閉じました。")
            return True
        else:
            logger.warning(
                "処理されたデータがないため、出力ファイルは作成されませんでした。"
            )
            return False  # 失敗として返す

    except Exception as e:
        logger.error(f"チャンク処理中にエラー: {e}", exc_info=True)
        # エラーが発生した場合でも、開いているWriterがあれば閉じる試み
        if writer:
            try:
                writer.close()
                logger.info("エラー発生後、ParquetWriterを閉じました。")
            except Exception as close_err:
                logger.error(
                    f"エラー発生後のParquetWriterクローズ中にさらにエラー: {close_err}"
                )
        return False  # 失敗として返す
    finally:
        # ガベージコレクション
        import gc  # ★★★ ここにも import gc があることを確認 ★★★

        gc.collect()


def main():
    print("=" * 60)
    print("Project Forge データ修復（pyarrow互換版 - チャンク処理・単一ファイル用）")
    print("=" * 60)
    print(f"入力ファイル: {SOURCE_FILE}")
    print(f"出力ファイル: {DEST_FILE}\n")

    if not SOURCE_FILE.exists():
        logger.error(f"エラー: 入力ファイルが存在しません: {SOURCE_FILE}")
        return

    if DEST_FILE.exists():
        overwrite = (
            input(
                f"警告: 出力ファイル {DEST_FILE} は既に存在します。上書きしますか？ (y/n): "
            )
            .strip()
            .lower()
        )
        if overwrite != "y":
            logger.info("処理を中止しました。")
            return
        else:
            logger.info("既存の出力ファイルを上書きします...")
            try:
                DEST_FILE.unlink()
            except OSError as e:
                logger.error(f"エラー: 既存ファイルの削除に失敗しました - {e}")
                return

    logger.info(
        f"ファイルを修復します: {SOURCE_FILE.name} (チャンクサイズ: {CHUNK_SIZE:,} 行)"
    )

    start_time = time.time()

    success = convert_parquet_file_chunked(SOURCE_FILE, DEST_FILE, CHUNK_SIZE)

    end_time = time.time()
    elapsed_time = end_time - start_time

    print("\n" + "=" * 60)
    if success:
        logger.info(f"修復完了: 成功 1件")
        logger.info(f"出力先: {DEST_FILE}")
        # ファイルサイズの確認
        try:
            output_size_mb = DEST_FILE.stat().st_size / (
                1024**2
            )  # ★★★ .size を .st_size に修正 ★★★
            logger.info(f"出力ファイルサイズ: {output_size_mb:.2f} MB")
        except FileNotFoundError:
            logger.error("エラー：出力ファイルが見つかりません。")
        logger.info(f"処理時間: {elapsed_time:.2f} 秒")
    else:
        logger.error(f"修復完了: 失敗 1件")
    print("=" * 60)


if __name__ == "__main__":
    # time と gc をインポート
    import time
    import gc

    main()
