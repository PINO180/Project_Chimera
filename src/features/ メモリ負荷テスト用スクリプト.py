import polars as pl
import pandas as pd
from pathlib import Path
import logging
import psutil
import gc
import time

# --- このスクリプトをエンジンと同じディレクトリに置いてください ---
from engine_1_A_a_vast_universe_of_features import (
    ProcessingConfig, 
    CalculationEngine, 
    get_sorted_partitions, 
    create_augmented_frame
)

# --- 基本設定 ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
PROCESS = psutil.Process()

def get_memory_usage_gb() -> float:
    """現在のプロセスのメモリ使用量をGB単位で取得"""
    return PROCESS.memory_info().rss / (1024 ** 3)

def find_heaviest_year(all_days: list[Path]) -> tuple[str | None, list[str] | None]:
    """最も行数が多い年を特定する"""
    logging.info("最も処理が重い年を特定しています...")
    yearly_groups = {}
    for day_path in all_days:
        year_key = day_path.parent.parent.name
        if year_key not in yearly_groups:
            yearly_groups[year_key] = []
        yearly_groups[year_key].append(str(day_path / "*.parquet"))

    max_rows = 0
    heaviest_year = None
    heaviest_year_files = None

    for year, files in yearly_groups.items():
        # collect()せずにLazyFrameのまま行数を取得
        row_count = pl.scan_parquet(files).select(pl.len()).collect().item()
        logging.info(f"  - {year}: {row_count:,} 行")
        if row_count > max_rows:
            max_rows = row_count
            heaviest_year = year
            heaviest_year_files = files
            
    if heaviest_year:
        logging.info(f"最も重い年を特定しました: {heaviest_year} ({max_rows:,} 行)")
    return heaviest_year, heaviest_year_files

def final_performance_and_memory_test():
    """
    最も重い1年分だけを対象に、計算時間とピークメモリを測定する。
    """
    logging.info("最終パフォーマンス＆メモリ負荷テストを開始します...")
    
    config = ProcessingConfig()
    W_MAX = config.w_max
    calculation_engine = CalculationEngine(config)
    
    all_days = get_sorted_partitions(Path(config.partitioned_tick_path))
    if not all_days:
        logging.error("日次パーティションが見つかりません。")
        return

    # 最も重い年を特定
    heaviest_year_key, heaviest_year_files = find_heaviest_year(all_days)
    if not heaviest_year_key:
        logging.error("処理対象の年が見つかりませんでした。")
        return
        
    logging.info(f"--- チャンク処理開始: {heaviest_year_key} ---")
    try:
        # チャンクの直前のパーティションパスを取得 (オーバーラップ処理のため)
        first_day_of_chunk = Path(heaviest_year_files[0]).parent
        first_day_index = all_days.index(first_day_of_chunk)
        prev_path = all_days[first_day_index - 1] if first_day_index > 0 else None

        # データロード
        augmented_df, len_current = create_augmented_frame(first_day_of_chunk, prev_path, W_MAX)
        if len(heaviest_year_files) > 1:
            remaining_df = pl.scan_parquet(heaviest_year_files[1:]).collect()
            augmented_df = pl.concat([augmented_df, remaining_df], how="vertical")
        
        rows_in_chunk = augmented_df.height
        data_size_gb = augmented_df.estimated_size('gb')
        mem_after_load = get_memory_usage_gb()
        logging.info(f"データロード完了。行数: {rows_in_chunk:,}行, データサイズ: {data_size_gb:.2f}GB, メモリ: {mem_after_load:.2f}GB")

        # 計算時間と計算中メモリの測定
        start_time = time.time()
        logging.info("全特徴量の計算を開始します...")
        
        result_df = calculation_engine.calculate_all_features(augmented_df.lazy()).collect()
        
        end_time = time.time()
        
        processing_time = end_time - start_time
        peak_mem_during_calc = get_memory_usage_gb()
        
        logging.info(f"計算完了。")

        # --- 最終サマリー ---
        logging.info("="*60)
        logging.info(f"最終テスト完了サマリー ({heaviest_year_key})")
        logging.info("="*60)
        logging.info(f"処理時間: {processing_time:.2f}秒 ({processing_time/60:.2f}分)")
        logging.info(f"データロード後のメモリ: {mem_after_load:.2f}GB")
        logging.info(f"計算中の最大ピークメモリ: {peak_mem_during_calc:.2f}GB")
        
        if peak_mem_during_calc > 55.0:
            logging.error("結論: 年単位処理は危険です。最悪の年でOOMエラーのリスクが極めて高いです。")
        elif peak_mem_during_calc > 50.0:
            logging.warning("結論: 年単位処理はリスクが高いです。最悪の年でメモリ警告閾値を超えました。")
        else:
            logging.info("結論: 年単位処理は安全かつ実行可能であると最終判断します。")

    except Exception as e:
        logging.error(f"チャンク {heaviest_year_key} の処理中にエラーが発生しました: {e}", exc_info=True)

if __name__ == "__main__":
    final_performance_and_memory_test()