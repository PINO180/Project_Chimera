# /workspace/scripts/sample_uniqueness_weighting_calculate.py

import sys
import duckdb
import logging
from pathlib import Path
import os

# プロジェクトのルートディレクトリをPythonの検索パスに追加
sys.path.append(str(Path(__file__).resolve().parents[1]))

from blueprint import S6_LABELED_DATASET_MONTHLY, BASE_DIR, S3_CONCURRENCY_RESULTS

# --- 定数 ---
# Dockerでマウントしたコンテナ内のパス
DUCKDB_TEMP_DIR_CONTAINER = "/duckdb_temp"
# 最終的な出力ファイル
OUTPUT_PATH = S3_CONCURRENCY_RESULTS

# --- ロギング設定 ---
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


def main():
    """
    第一段階：ウィンドウ関数を駆使した高効率アルゴリズムで「並行数(concurrency)」の計算のみに特化し、
    結果を小さな中間ファイルとして出力する。
    """
    logging.info(
        "### Final Battle Stage 1: Calculating Concurrency with Window Functions ###"
    )

    input_path = str(S6_LABELED_DATASET_MONTHLY)

    # 出力先ディレクトリを準備
    os.makedirs(OUTPUT_PATH.parent, exist_ok=True)

    logging.info(f"Input data path: {input_path}")
    logging.info(f"Temporary directory: {DUCKDB_TEMP_DIR_CONTAINER}")
    logging.info(f"Output path for results: {OUTPUT_PATH}")

    # GeminiDeepResearchの最終報告書で推奨された、実運用レベルの最終SQLクエリ
    query = f"""
    -- 設定：これらの設定は安定性のために【極めて重要】
    SET temp_directory = '{DUCKDB_TEMP_DIR_CONTAINER}';
    SET memory_limit = '20GB'; 
    SET preserve_insertion_order = false; -- 最重要チューニング項目
    SET enable_progress_bar = true; -- 進捗表示を有効化！
    SET threads = 12; -- CPUコアをフル活用

    -- 最終結果を直接ファイルに書き出す
    COPY (
        -- WITH句で処理を段階的に定義
        WITH events_with_id AS (
            -- ステップ0: event_idを付与
            SELECT
                timestamp,
                t1,
                row_number() OVER (ORDER BY timestamp, t1) AS event_id
            FROM read_parquet('{input_path}', hive_partitioning=true)
        ),
        event_stream AS (
            -- ステップ1: イベント期間を「+1(開始)」と「-1(終了)」のポイントイベントに分解
            SELECT event_id, timestamp AS event_time, 1 AS type FROM events_with_id
            UNION ALL
            SELECT event_id, t1 AS event_time, -1 AS type FROM events_with_id
        ),
        concurrency_levels AS (
            -- ステップ2: イベントを時間順に並べ、累積和で各時点の同時イベント数を計算
            SELECT
                event_time,
                -- このウィンドウ関数がアルゴリズムの中核
                SUM(type) OVER (ORDER BY event_time ASC, type DESC) AS active_intervals
            FROM event_stream
        ),
        final_concurrency AS (
            -- ステップ3: 元のイベント開始時刻に、計算した同時イベント数を結合
            SELECT
                e.event_id,
                MAX(c.active_intervals) AS concurrency
            FROM events_with_id AS e
            JOIN concurrency_levels AS c ON e.timestamp = c.event_time
            GROUP BY e.event_id
        )
        -- 最終出力
        SELECT * FROM final_concurrency
        
    ) TO '{OUTPUT_PATH}' (FORMAT PARQUET);
    """

    try:
        con = duckdb.connect()
        logging.info("Executing the final, optimized query... The end is near.")
        con.execute(query)
        logging.info("The final query executed successfully.")
    except Exception as e:
        logging.error(f"An error occurred during DuckDB execution: {e}", exc_info=True)
        raise
    finally:
        if "con" in locals():
            con.close()
            logging.info("DuckDB connection closed.")

    logging.info("\n" + "=" * 60)
    logging.info("### VICTORY! Stage 1 COMPLETED! ###")
    logging.info(f"The 'engine block' is perfectly forged at: {OUTPUT_PATH}")
    logging.info("=" * 60)


if __name__ == "__main__":
    main()
