# /workspace/scripts/sample_uniqueness_weighting_calculate_v5.py

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
    V5仕様（Project Cimera）対応版：
    ウィンドウ関数を用いた高効率アルゴリズムで、Long/Short双方の
    「並行数(concurrency)」を計算し、中間ファイルとして出力する。
    """
    logging.info(
        "### Final Battle Stage 1 (V5): Calculating Dual Concurrency with Window Functions ###"
    )

    input_path = str(S6_LABELED_DATASET_MONTHLY)

    # 出力先ディレクトリを準備
    os.makedirs(OUTPUT_PATH.parent, exist_ok=True)

    logging.info(f"Input data path: {input_path}")
    logging.info(f"Temporary directory: {DUCKDB_TEMP_DIR_CONTAINER}")
    logging.info(f"Output path for results: {OUTPUT_PATH}")

    # V5仕様に基づく双方向の並行数計算SQLクエリ
    query = f"""
    -- 設定：安定性のためのチューニング
    SET temp_directory = '{DUCKDB_TEMP_DIR_CONTAINER}';
    SET memory_limit = '20GB'; 
    SET preserve_insertion_order = false; 
    SET enable_progress_bar = true; 
    SET threads = 12; 

    COPY (
        WITH base_events AS (
            -- ステップ0: トリガー発生時点のみを抽出し、Long/Shortそれぞれの終了時刻を計算
            SELECT
                timestamp,
                timeframe, -- ★追加
                timestamp + interval '1 minute' * duration_long AS t1_long,
                timestamp + interval '1 minute' * duration_short AS t1_short
            FROM read_parquet('{input_path}', hive_partitioning=true)
            WHERE is_trigger = 1
        ),
        
        -- ==========================================
        -- Longサイドの並行数計算
        -- ==========================================
        event_stream_long AS (
            SELECT timestamp AS event_time, 1 AS type FROM base_events
            UNION ALL
            SELECT t1_long AS event_time, -1 AS type FROM base_events
        ),
        concurrency_levels_long AS (
            SELECT
                event_time,
                SUM(type) OVER (ORDER BY event_time ASC, type DESC) AS active_intervals
            FROM event_stream_long
        ),
        final_concurrency_long AS (
            SELECT
                e.timestamp,
                e.timeframe, -- ★追加
                MAX(c.active_intervals) AS concurrency_long
            FROM base_events AS e
            JOIN concurrency_levels_long AS c ON e.timestamp = c.event_time
            GROUP BY e.timestamp, e.timeframe -- ★追加
        ),
        
        -- ==========================================
        -- Shortサイドの並行数計算
        -- ==========================================
        event_stream_short AS (
            SELECT timestamp AS event_time, 1 AS type FROM base_events
            UNION ALL
            SELECT t1_short AS event_time, -1 AS type FROM base_events
        ),
        concurrency_levels_short AS (
            SELECT
                event_time,
                SUM(type) OVER (ORDER BY event_time ASC, type DESC) AS active_intervals
            FROM event_stream_short
        ),
        final_concurrency_short AS (
            SELECT
                e.timestamp,
                e.timeframe, -- ★追加
                MAX(c.active_intervals) AS concurrency_short
            FROM base_events AS e
            JOIN concurrency_levels_short AS c ON e.timestamp = c.event_time
            GROUP BY e.timestamp, e.timeframe -- ★追加
        )
        
        -- ==========================================
        -- 最終結果の結合
        -- ==========================================
        SELECT 
            l.timestamp,
            l.timeframe, -- ★追加
            l.concurrency_long,
            s.concurrency_short
        FROM final_concurrency_long l
        JOIN final_concurrency_short s ON l.timestamp = s.timestamp AND l.timeframe = s.timeframe -- ★追加
        ORDER BY l.timestamp, l.timeframe -- ★追加
        
    ) TO '{OUTPUT_PATH}' (FORMAT PARQUET);
    """

    try:
        con = duckdb.connect()
        logging.info("Executing the V5 optimized query for dual concurrency...")
        con.execute(query)
        logging.info("The V5 query executed successfully.")
    except Exception as e:
        logging.error(f"An error occurred during DuckDB execution: {e}", exc_info=True)
        raise
    finally:
        if "con" in locals():
            con.close()
            logging.info("DuckDB connection closed.")

    logging.info("\n" + "=" * 60)
    logging.info("### VICTORY! V5 Stage 1 COMPLETED! ###")
    logging.info(f"The dual concurrency results are safely written to: {OUTPUT_PATH}")
    logging.info("=" * 60)


if __name__ == "__main__":
    main()
