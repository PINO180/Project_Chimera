"""
tickデータのパーティションカラム型不一致を修復するスクリプト
year, month, day を全てint32に統一
"""
import polars as pl
from pathlib import Path
from tqdm import tqdm
import logging
import shutil

# ログ設定
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def fix_tick_directory(tick_dir: Path, output_suffix: str = "_fixed") -> None:
    """
    tickディレクトリのパーティションカラム（year, month, day）をint32に統一して再保存
    
    Args:
        tick_dir: 修復対象のtickディレクトリ
        output_suffix: 出力ディレクトリ名のサフィックス
    """
    try:
        logger.info(f"修復開始: {tick_dir.name}")
        
        # 出力先ディレクトリ
        output_dir = tick_dir.parent / f"{tick_dir.name}{output_suffix}"
        if output_dir.exists():
            logger.info(f"既存の出力ディレクトリを削除: {output_dir.name}")
            shutil.rmtree(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # 全parquetファイルを検出
        all_parquet_files = list(tick_dir.glob("**/*.parquet"))
        logger.info(f"{len(all_parquet_files)}個のファイルを処理")
        
        success_count = 0
        fail_count = 0
        
        for source_file in tqdm(all_parquet_files, desc=f"  {tick_dir.name}", leave=False):
            try:
                # 各ファイルを個別に処理
                df = pl.read_parquet(source_file)
                
                # パーティションカラムをint32に統一
                partition_cols = ['year', 'month', 'day']
                cols_to_cast = [col for col in partition_cols if col in df.columns]
                
                if cols_to_cast:
                    for col in cols_to_cast:
                        df = df.with_columns(pl.col(col).cast(pl.Int32))
                
                # 出力先パスを維持（パーティション構造を保持）
                relative_path = source_file.relative_to(tick_dir)
                dest_file = output_dir / relative_path
                dest_file.parent.mkdir(parents=True, exist_ok=True)
                
                # pyarrow互換形式で保存
                df.write_parquet(
                    dest_file,
                    compression='snappy',
                    use_pyarrow=True
                )
                
                success_count += 1
                
            except Exception as e:
                logger.error(f"ファイル処理エラー {source_file.name}: {e}")
                fail_count += 1
                continue
        
        logger.info(f"✓ 完了: {tick_dir.name} → {output_dir.name} (成功: {success_count}, 失敗: {fail_count})")
        
    except Exception as e:
        logger.error(f"✗ エラー: {tick_dir.name} - {e}")
        raise


def main():
    """メイン処理"""
    base_path = Path('/workspace/data/XAUUSD/stratum_2_features_fixed')
    
    # 全てのtickディレクトリを検出
    tick_dirs = []
    for engine_dir in base_path.iterdir():
        if not engine_dir.is_dir():
            continue
        for item in engine_dir.iterdir():
            if item.is_dir() and 'tick' in item.name and '_fixed' not in item.name:
                tick_dirs.append(item)
    
    if not tick_dirs:
        logger.warning("tickディレクトリが見つかりませんでした")
        return
    
    logger.info(f"{len(tick_dirs)}個のtickディレクトリを検出")
    
    # 各ディレクトリを修復
    for tick_dir in tqdm(tick_dirs, desc="tickデータ修復中"):
        fix_tick_directory(tick_dir)
    
    logger.info("全てのtickデータ修復が完了しました")
    
    # 修復後の処理指示
    print("\n" + "="*70)
    print("修復完了！次のステップ：")
    print("1. 元のtickディレクトリをバックアップ（オプション）")
    print("2. 元のtickディレクトリを削除")
    print("3. _fixedサフィックスのディレクトリ名から_fixedを削除")
    print("\n例:")
    print("  mv features_e1a_tick features_e1a_tick_backup")
    print("  mv features_e1a_tick_fixed features_e1a_tick")
    print("="*70)


if __name__ == "__main__":
    main()