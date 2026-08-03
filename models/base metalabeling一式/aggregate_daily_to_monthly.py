# /workspace/scripts/00_aggregate_daily_to_monthly.py

import sys
import duckdb
import logging
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from blueprint import S6_LABELED_DATASET, S6_LABELED_DATASET_MONTHLY

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


def main():
    logging.info("### Stage 0: Intermediate Aggregation Script ###")
    logging.info("Aggregating daily partitioned data into monthly partitions...")

    input_base  = S6_LABELED_DATASET
    output_path = Path(str(S6_LABELED_DATASET_MONTHLY))

    logging.info(f"Input base: {input_base}")
    logging.info(f"Output directory: {output_path}")

    # 存在するyear/monthの組み合わせを列挙
    year_dirs = sorted([p for p in input_base.iterdir()
                        if p.is_dir() and p.name.startswith("year=")])
    ym_list = []
    for year_dir in year_dirs:
        year = int(year_dir.name.split("=")[1])
        month_dirs = sorted([p for p in year_dir.iterdir()
                             if p.is_dir() and p.name.startswith("month=")])
        for month_dir in month_dirs:
            month = int(month_dir.name.split("=")[1])
            ym_list.append((year, month))

    logging.info(f"Processing {len(ym_list)} months one by one to avoid OOM...")

    try:
        con = duckdb.connect()

        for year, month in ym_list:
            month_glob = str(input_base / f"year={year}/month={month}/day=*/*.parquet")
            out_dir  = output_path / f"year={year}" / f"month={month}"
            out_dir.mkdir(parents=True, exist_ok=True)
            out_file = str(out_dir / "data.parquet")

            query = f"""
            COPY (
                SELECT *
                FROM read_parquet('{month_glob}', hive_partitioning=false)
            ) TO '{out_file}' (FORMAT PARQUET);
            """
            try:
                con.execute(query)
                logging.info(f"  -> Done: {year}-{month:02d}")
            except Exception as e:
                logging.warning(f"  -> Skipped {year}-{month:02d}: {e}")

        logging.info("All months processed successfully.")

    except Exception as e:
        logging.error(f"An error occurred: {e}", exc_info=True)
        raise
    finally:
        con.close()
        logging.info("DuckDB connection closed.")

    logging.info("\n" + "=" * 60)
    logging.info("### Intermediate Aggregation COMPLETED! ###")
    logging.info(f"Monthly aggregated data is now ready in: {output_path}")
    logging.info("=" * 60)


if __name__ == "__main__":
    main()
