# /workspace/scripts/00_aggregate_daily_to_monthly.py

import sys
import duckdb
import logging
from pathlib import Path

# プロジェクトのルートディレクトリをPythonの検索パスに追加
sys.path.append(str(Path(__file__).resolve().parents[1]))

# blueprintから一元管理された設定を読み込む
from blueprint import S6_LABELED_DATASET, S6_LABELED_DATASET_MONTHLY

# --- ロギング設定 ---
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


def main():
    """
    日次でパーティション化されたParquetデータを、月次パーティションに集約するスクリプト。
    DuckDBのOut-of-Core能力を最大限に活用し、メモリ安全性を保証する。
    """
    logging.info("### Stage 0: Intermediate Aggregation Script ###")
    logging.info("Aggregating daily partitioned data into monthly partitions...")

    # blueprintから入出力パスを取得
    input_path_glob = str(S6_LABELED_DATASET / "*/*/*/*.parquet")
    output_path = str(S6_LABELED_DATASET_MONTHLY)

    logging.info(f"Input data glob: {input_path_glob}")
    logging.info(f"Output directory: {output_path}")

    # この単一のSQLコマンドが、全ての魔法を実行する
    # DuckDBが裏側でファイルの読み込み、データの再編成、書き出し、メモリ管理を全て行う
    query = f"""
    -- COPYコマンドは、SELECTクエリの結果を別の場所に書き出すための強力な命令
    COPY (
        -- read_parquet関数で、Hive形式で分割された全ファイルを一つの論理テーブルとして読み込む
        SELECT *
        FROM read_parquet(
            '{input_path_glob}',
            hive_partitioning=true
        )
    ) TO '{output_path}' (
        -- 出力形式はParquet
        FORMAT PARQUET,
        -- yearとmonth列の値に基づいて、新しいパーティション・ディレクトリを作成
        PARTITION_BY (year, month),
        -- もし出力先に同名のパーティションが存在する場合、上書きする
        OVERWRITE_OR_IGNORE true
    );
    """

    try:
        # DuckDBに接続（ファイルパスを指定しない場合、インメモリデータベースとして動作）
        con = duckdb.connect()

        logging.info("Executing aggregation query with DuckDB...")
        # クエリを実行。DuckDBが数分かけて、数百GBのデータを安全に処理します。
        con.execute(query)

        logging.info("Aggregation query executed successfully.")

    except Exception as e:
        logging.error(f"An error occurred during DuckDB execution: {e}", exc_info=True)
        raise
    finally:
        # 接続を閉じる
        con.close()
        logging.info("DuckDB connection closed.")

    logging.info("\n" + "=" * 60)
    logging.info("### Intermediate Aggregation COMPLETED! ###")
    logging.info(f"Monthly aggregated data is now ready in: {output_path}")
    logging.info("=" * 60)


if __name__ == "__main__":
    main()
