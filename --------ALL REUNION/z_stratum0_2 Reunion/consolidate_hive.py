#!/usr/bin/env python3
"""
Hiveパーティション統合スクリプト

「遅いは速い」戦略に基づいて、GPU処理を避け、確実性を重視したアプローチで
XAUUSDデータのHiveパーティションを単一のParquetファイルに統合します。

主な特徴:
- メモリ効率重視（一度に大量のデータを読み込まない）
- 順序保証（timeframe順に処理）
- エラー処理（問題のあるパーティションをスキップ）
- プログレス表示
- スキーマ一貫性の保証
"""

import os
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from pathlib import Path
import logging
from typing import List, Optional, Tuple
import re
from datetime import datetime
import glob

# ログ設定
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('hive_consolidation.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class HivePartitionConsolidator:
    """Hiveパーティションを安全に統合するクラス"""
    
    def __init__(self, input_dir: str, output_file: str):
        self.input_dir = Path(input_dir)
        self.output_file = Path(output_file)
        self.processed_count = 0
        self.error_count = 0
        self.schema = None
        
    def extract_timeframe_order(self, timeframe: str) -> Tuple[int, str]:
        """
        timeframeから並び順を決定する関数
        数値部分とアルファベット部分を分離して適切にソート
        
        例: M1 -> (1, 'M'), H4 -> (4, 'H'), tick -> (0, 'tick')
        """
        if timeframe == 'tick':
            return (0, 'tick')
        
        # 数値とアルファベットを分離
        match = re.match(r'([A-Za-z]+)(\d+)', timeframe)
        if match:
            unit = match.group(1)
            number = int(match.group(2))
            
            # 時間単位の優先順位を定義
            unit_priority = {
                'M': 1,     # 分
                'H': 2,     # 時間
                'D': 3,     # 日
                'W': 4,     # 週
                'MN': 5     # 月
            }
            
            return (unit_priority.get(unit, 999), number)
        
        # 不明な形式の場合は最後に配置
        return (999, timeframe)
    
    def discover_partitions(self) -> List[Tuple[str, Path]]:
        """
        パーティションを発見し、適切な順序でソートする
        
        Returns:
            List[Tuple[str, Path]]: (timeframe, partition_path) のリスト
        """
        partitions = []
        
        # timeframe=* パターンのディレクトリを検索
        for partition_dir in self.input_dir.glob('timeframe=*'):
            if partition_dir.is_dir():
                timeframe = partition_dir.name.replace('timeframe=', '')
                partitions.append((timeframe, partition_dir))
        
        if not partitions:
            raise ValueError(f"パーティションが見つかりません: {self.input_dir}")
        
        # カスタムソート関数を使用して並び替え
        partitions.sort(key=lambda x: self.extract_timeframe_order(x[0]))
        
        logger.info(f"発見されたパーティション ({len(partitions)}個):")
        for timeframe, path in partitions:
            logger.info(f"  - {timeframe}")
        
        return partitions
    
    def get_parquet_files(self, partition_dir: Path) -> List[Path]:
        """
        パーティションディレクトリ内のparquetファイルを取得
        ファイル名でソートして順序を保証
        """
        parquet_files = list(partition_dir.glob('*.parquet'))
        parquet_files.sort()  # ファイル名でソート（part.0.parquet, part.1.parquet, ...）
        return parquet_files
    
    def read_partition_safely(self, partition_dir: Path, timeframe: str) -> Optional[pd.DataFrame]:
        """
        パーティションを安全に読み込む
        
        エラーハンドリングを含み、失敗した場合はNoneを返す
        """
        try:
            parquet_files = self.get_parquet_files(partition_dir)
            
            if not parquet_files:
                logger.warning(f"Parquetファイルが見つかりません: {partition_dir}")
                return None
            
            logger.info(f"読み込み中: {timeframe} ({len(parquet_files)}ファイル)")
            
            # 各parquetファイルを順番に読み込んで結合
            dfs = []
            for pfile in parquet_files:
                try:
                    df_part = pd.read_parquet(pfile)
                    if not df_part.empty:
                        dfs.append(df_part)
                except Exception as e:
                    logger.error(f"ファイル読み込みエラー {pfile}: {e}")
                    self.error_count += 1
                    continue
            
            if not dfs:
                logger.warning(f"有効なデータが見つかりません: {timeframe}")
                return None
            
            # データフレームを結合
            df = pd.concat(dfs, ignore_index=True)
            
            # timeframe列を追加（元のHive構造を保持）
            df['timeframe'] = timeframe
            
            logger.info(f"読み込み完了: {timeframe} - {len(df):,}行")
            return df
            
        except Exception as e:
            logger.error(f"パーティション読み込みエラー {timeframe}: {e}")
            self.error_count += 1
            return None
    
    def validate_and_harmonize_schema(self, df: pd.DataFrame, timeframe: str) -> pd.DataFrame:
        """
        スキーマの検証と統一化
        
        資料の原則3「契約としてのスキーマ」に従い、データ型の一貫性を保証
        """
        if self.schema is None:
            # 最初のデータフレームからスキーマを定義
            self.schema = df.dtypes.to_dict()
            logger.info(f"基準スキーマを設定: {timeframe}")
            for col, dtype in self.schema.items():
                logger.info(f"  {col}: {dtype}")
        else:
            # スキーマの整合性をチェックし、必要に応じて型変換
            for col in df.columns:
                if col in self.schema:
                    expected_dtype = self.schema[col]
                    current_dtype = df[col].dtype
                    
                    if current_dtype != expected_dtype:
                        logger.warning(f"型不一致検出 {timeframe}.{col}: {current_dtype} -> {expected_dtype}")
                        try:
                            df[col] = df[col].astype(expected_dtype)
                        except Exception as e:
                            logger.error(f"型変換失敗 {timeframe}.{col}: {e}")
                            # 安全な型（文字列）に変換
                            df[col] = df[col].astype('string')
                            logger.info(f"文字列型にフォールバック: {col}")
        
        return df
    
    def consolidate(self, chunk_size: int = 50000) -> bool:
        """
        パーティションを統合してParquetファイルを作成
        
        Args:
            chunk_size: メモリ効率のためのチャンクサイズ
            
        Returns:
            bool: 成功したかどうか
        """
        try:
            logger.info("=== Hiveパーティション統合を開始 ===")
            logger.info(f"入力ディレクトリ: {self.input_dir}")
            logger.info(f"出力ファイル: {self.output_file}")
            
            # パーティションを発見
            partitions = self.discover_partitions()
            
            # 出力ディレクトリを作成
            self.output_file.parent.mkdir(parents=True, exist_ok=True)
            
            # Parquetライターを初期化（後で使用）
            writer = None
            total_rows = 0
            
            try:
                for i, (timeframe, partition_dir) in enumerate(partitions):
                    logger.info(f"[{i+1}/{len(partitions)}] 処理中: {timeframe}")
                    
                    # このパーティション内の全parquetファイルを取得
                    parquet_files = self.get_parquet_files(partition_dir)
                    
                    if not parquet_files:
                        logger.warning(f"スキップ: {timeframe} (ファイルなし)")
                        continue
                    
                    # 1ファイルずつ処理（メモリ効率最大化）
                    for j, pfile in enumerate(parquet_files):
                        try:
                            logger.info(f"  ファイル [{j+1}/{len(parquet_files)}]: {pfile.name}")
                            
                            # 単一ファイルを読み込み
                            df = pd.read_parquet(pfile)
                            
                            if df.empty:
                                logger.warning(f"  空ファイルをスキップ: {pfile.name}")
                                continue
                            
                            # timeframe列を追加
                            df['timeframe'] = timeframe
                            
                            # スキーマの検証と統一化
                            df = self.validate_and_harmonize_schema(df, timeframe)
                            
                            # Parquetライターの初期化（最初のデータから）
                            if writer is None:
                                table = pa.Table.from_pandas(df)
                                writer = pq.ParquetWriter(self.output_file, table.schema)
                                logger.info("Parquetライターを初期化")
                            
                            # チャンクに分けて書き込み
                            for start_idx in range(0, len(df), chunk_size):
                                end_idx = min(start_idx + chunk_size, len(df))
                                chunk = df.iloc[start_idx:end_idx]
                                
                                table_chunk = pa.Table.from_pandas(chunk)
                                writer.write_table(table_chunk)
                                
                                total_rows += len(chunk)
                            
                            # メモリを即座に解放
                            del df
                            
                        except Exception as e:
                            logger.error(f"  ファイル処理エラー {pfile}: {e}")
                            self.error_count += 1
                            continue
                    
                    self.processed_count += 1
                    logger.info(f"完了: {timeframe} (累計: {total_rows:,}行)")
                
            finally:
                # ライターを閉じる
                if writer is not None:
                    writer.close()
            
            # 統計情報を表示
            logger.info("=== 統合完了 ===")
            logger.info(f"処理済みパーティション: {self.processed_count}")
            logger.info(f"エラー数: {self.error_count}")
            logger.info(f"総行数: {total_rows:,}")
            
            if self.output_file.exists():
                file_size = self.output_file.stat().st_size / (1024 * 1024)  # MB
                logger.info(f"出力ファイルサイズ: {file_size:.2f} MB")
                logger.info(f"出力ファイル: {self.output_file}")
                return True
            else:
                logger.error("出力ファイルが作成されませんでした")
                return False
                
        except Exception as e:
            logger.error(f"統合処理でエラーが発生: {e}")
            return False

def main():
    """メイン実行関数"""
    
    # 設定（必要に応じて変更してください）
    INPUT_DIR = "/workspaces/project_forge/data/1_XAUUSD_base_data/XAUUSD_tick_master_FINAL_PATCHED"  # あなたのHiveパーティションディレクトリ
    OUTPUT_FILE = "consolidated_XAUUSD_data.parquet"  # 出力ファイル名
    
    # 入力パスの存在確認
    if not Path(INPUT_DIR).exists():
        logger.error(f"入力ディレクトリが見つかりません: {INPUT_DIR}")
        logger.error("スクリプトと同じディレクトリにXAUUSD_tick_masterフォルダを配置してください")
        return False
    
    # 統合処理を実行
    consolidator = HivePartitionConsolidator(INPUT_DIR, OUTPUT_FILE)
    success = consolidator.consolidate()
    
    if success:
        logger.info("✅ 統合が正常に完了しました")
        
        # 検証のため、出力ファイルを簡単にチェック
        try:
            logger.info("=== 出力ファイルの検証 ===")
            df_result = pd.read_parquet(OUTPUT_FILE)
            logger.info(f"最終行数: {len(df_result):,}")
            logger.info(f"列数: {len(df_result.columns)}")
            logger.info(f"timeframe別行数:")
            if 'timeframe' in df_result.columns:
                timeframe_counts = df_result['timeframe'].value_counts().sort_index()
                for tf, count in timeframe_counts.items():
                    logger.info(f"  {tf}: {count:,} 行")
            logger.info("✅ 検証完了")
        except Exception as e:
            logger.warning(f"検証でエラー: {e}")
        
        return True
    else:
        logger.error("❌ 統合に失敗しました")
        return False

if __name__ == "__main__":
    main()