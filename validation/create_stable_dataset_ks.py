import sys
from pathlib import Path
import json
import shutil
from tqdm import tqdm
import logging

# プロジェクトのルートパスを追加
sys.path.append(str(Path(__file__).resolve().parents[1]))

try:
    import blueprint
except ImportError:
    print("エラー: blueprint.pyが見つかりません。パスを確認してください。")
    sys.exit(1)

import polars as pl

# --- ログ設定 ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- 設定 ---
INPUT_DIR = Path(blueprint.S2_FEATURES_FIXED)
OUTPUT_DIR = Path("/workspace/data/XAUUSD/stratum_2_features_after_ks")
UNSTABLE_LIST_PATH = Path(blueprint.S3_ARTIFACTS) / "ks_unstable_features.json"

def load_unstable_features(path: Path) -> set:
    """不安定特徴量のリストをJSONファイルから読み込み、セットとして返す。"""
    if not path.exists():
        raise FileNotFoundError(f"不安定特徴量リストが見つかりません: {path}")
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    unstable_set = set(data.get("unstable_features", []))
    logger.info(f"{len(unstable_set)}個の不安定特徴量を読み込みました。")
    return unstable_set

def process_file(input_path: Path, output_path: Path, unstable_set: set):
    """単一の非tick Parquetファイルを処理する。"""
    try:
        df = pl.read_parquet(input_path)
        
        # ファイル名からタイムフレームサフィックスを抽出 (例: _M1, _H4)
        timeframe_suffix = f"_{input_path.stem.split('_')[-1]}"
        
        # このファイルに存在する列で、不安定リストに含まれるものを特定
        original_cols = set(df.columns)
        cols_to_drop = {
            col for col in original_cols 
            if f"{col}{timeframe_suffix}" in unstable_set
        }
        
        if cols_to_drop:
            df_cleaned = df.drop(list(cols_to_drop))
            df_cleaned.write_parquet(output_path, compression='snappy', use_pyarrow=True)
            logger.debug(f"  - {input_path.name}: {len(cols_to_drop)}列を削除")
        else:
            # 削除する列がない場合は、ファイルをそのままコピー
            shutil.copy2(input_path, output_path)
            logger.debug(f"  - {input_path.name}: 削除対象なし、ファイルをコピー")
            
    except Exception as e:
        logger.error(f"ファイル処理エラー {input_path.name}: {e}")

def process_tick_directory(input_path: Path, output_path: Path, unstable_set: set):
    """パーティション化されたtickディレクトリを処理する（パーティション単位のストリーミング処理による最終確定版）"""
    try:
        logger.info(f"  - {input_path.name}: パーティション単位でのストリーミング処理を開始")
        
        # 1. 入力元の全パーティションパスを取得
        all_input_partitions = sorted(list(input_path.rglob("*.parquet")))
        
        total_cols_dropped = 0
        
        # 2. 各パーティションを一つずつループ処理
        for i, partition_file_path in enumerate(all_input_partitions):
            
            # 2a. 単一のパーティション（数十MB）のみをメモリに読み込む
            df_partition = pl.read_parquet(partition_file_path)
            
            # 2b. 不安定な列を特定して削除
            timeframe_suffix = "_tick"
            original_cols = set(df_partition.columns)
            cols_to_drop = {
                col for col in original_cols 
                if f"{col}{timeframe_suffix}" in unstable_set
            }
            
            if cols_to_drop:
                df_cleaned_partition = df_partition.drop(list(cols_to_drop))
                total_cols_dropped += len(cols_to_drop)
            else:
                df_cleaned_partition = df_partition

            # 2c. 出力先のパスを構築（元のパーティション構造を維持）
            relative_path = partition_file_path.relative_to(input_path)
            output_file_path = output_path / relative_path
            
            # 出力先ディレクトリを作成
            output_file_path.parent.mkdir(parents=True, exist_ok=True)
            
            # 2d. 処理済みの小さなパーティションをディスクに書き出す
            df_cleaned_partition.write_parquet(
                output_file_path,
                compression='snappy',
                use_pyarrow=True
            )

        avg_cols_dropped = total_cols_dropped / len(all_input_partitions) if all_input_partitions else 0
        logger.info(f"  - {input_path.name}: {len(all_input_partitions)}個のパーティションを処理完了。平均{avg_cols_dropped:.1f}列を削除。")

    except Exception as e:
        logger.error(f"Tickディレクトリ処理エラー {input_path.name}: {e}", exc_info=True)

def main():
    """メイン実行関数"""
    logger.info("--- KS検定後の安定特徴量データセット作成開始 ---")
    
    # 1. 不安定特徴量リストをロード
    unstable_set = load_unstable_features(UNSTABLE_LIST_PATH)
    
    # 2. 出力ディレクトリを準備（既存の場合はクリーンアップ）
    if OUTPUT_DIR.exists():
        logger.warning(f"既存の出力ディレクトリを削除します: {OUTPUT_DIR}")
        shutil.rmtree(OUTPUT_DIR)
    OUTPUT_DIR.mkdir(parents=True)
    logger.info(f"出力ディレクトリを作成しました: {OUTPUT_DIR}")

    # 3. 入力ディレクトリを走査して処理
    engine_dirs = [d for d in INPUT_DIR.iterdir() if d.is_dir()]
    
    for engine_dir in tqdm(engine_dirs, desc="Engineディレクトリ処理中"):
        output_engine_dir = OUTPUT_DIR / engine_dir.name
        output_engine_dir.mkdir()
        
        items_to_process = list(engine_dir.iterdir())
        for item_path in tqdm(items_to_process, desc=f"  {engine_dir.name}", leave=False):
            
            output_item_path = output_engine_dir / item_path.name

            if item_path.is_file() and item_path.suffix == '.parquet':
                # 非tickファイルの処理
                process_file(item_path, output_item_path, unstable_set)
            
            elif item_path.is_dir() and 'tick' in item_path.name:
                # tickディレクトリの処理
                process_tick_directory(item_path, output_item_path, unstable_set)

    logger.info("--- 全ての処理が完了しました ---")

if __name__ == "__main__":
    main()