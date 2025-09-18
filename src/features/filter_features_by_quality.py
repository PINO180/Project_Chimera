# filter_features_by_quality.py

import sys
import logging
from pathlib import Path
import polars as pl
import numpy as np

# 親ディレクトリをパスに追加して、engine_1のクラスをインポート
# このスクリプトをengine_1.pyと同じディレクトリか、src/
# ディレクトリに置くことを想定しています。
# 環境に合わせて調整してください。
try:
    # 仮に engine_1 が src/features/ にあると仮定
    sys.path.append(str(Path(__file__).parent.parent))
    from features.engine_1_a_vast_universe_of_features import Calculator, WindowManager, MemoryManager
except ImportError:
    print("engine_1_a_vast_universe_of_features.pyが見つかりません。")
    print("このスクリプトを適切な場所に配置するか、sys.pathを調整してください。")
    sys.exit(1)

# --- 設定項目 ---
INPUT_DIR = Path("/workspaces/project_forge/data/2_feature_value/feature_value_a_vast_universe")
OUTPUT_DIR = INPUT_DIR.parent / "3_high_quality_features"
QUALITY_THRESHOLD = 0.6  # 品質スコアの足切り閾値 (0.0 ~ 1.0)
# --- 設定ここまで ---

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def run_quality_filtering():
    """
    生成された特徴量ファイルを品質スコアでフィルタリングする
    """
    logging.info(f"🚀 品質フィルタリング処理開始 - 閾値: {QUALITY_THRESHOLD}")
    OUTPUT_DIR.mkdir(exist_ok=True)
    
    # engine_1のCalculatorを品質評価のためにインスタンス化
    # WindowManagerとMemoryManagerはダミーでOK
    calculator = Calculator(WindowManager(), MemoryManager())

    # 入力ディレクトリ内のParquetファイルを検索
    # feature_value_..._test.parquet のようなテストファイルのみを対象とする
    feature_files = list(INPUT_DIR.glob("feature_value_*_test.parquet"))
    
    if not feature_files:
        logging.warning(f"処理対象のParquetファイルが見つかりません: {INPUT_DIR}")
        return

    logging.info(f"{len(feature_files)}個のタイムフレームファイルを処理します。")
    
    total_before = 0
    total_after = 0

    for file_path in sorted(feature_files):
        try:
            timeframe_name = file_path.stem.split('_')[-2]
            logging.info(f"--- 処理中: {timeframe_name} ({file_path.name}) ---")
            
            df = pl.read_parquet(file_path)
            
            high_quality_columns = []
            
            for column in df.columns:
                feature_values = df[column].to_numpy()
                
                # 品質スコアを計算
                score = calculator.calculate_quality_score(feature_values)
                
                if score >= QUALITY_THRESHOLD:
                    high_quality_columns.append(column)
                else:
                    logging.debug(f"  [除外] {column} (スコア: {score:.3f})")

            initial_count = len(df.columns)
            final_count = len(high_quality_columns)
            survival_rate = final_count / initial_count if initial_count > 0 else 0
            
            total_before += initial_count
            total_after += final_count
            
            logging.info(f"  結果: {initial_count}個中 {final_count}個の特徴量を維持 (生存率: {survival_rate:.1%})")

            if final_count > 0:
                # 高品質な特徴量のみを選択して新しいDataFrameを作成
                filtered_df = df.select(high_quality_columns)
                
                # 新しいディレクトリに保存
                output_path = OUTPUT_DIR / file_path.name
                filtered_df.write_parquet(output_path, compression='snappy')
                logging.info(f"  高品質な特徴量を保存しました: {output_path}")

        except Exception as e:
            logging.error(f"{file_path.name} の処理中にエラーが発生: {e}")
            continue
            
    logging.info("="*50)
    final_survival_rate = total_after / total_before if total_before > 0 else 0
    logging.info(f"🎉 全ての処理が完了しました。")
    logging.info(f"  総特徴量: {total_before}個 → {total_after}個 (全体の生存率: {final_survival_rate:.1%})")
    logging.info(f"  高品質な特徴量セットは {OUTPUT_DIR} に保存されました。")
    logging.info("="*50)


if __name__ == "__main__":
    run_quality_filtering()