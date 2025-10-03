"""
build_master_table.py - 安定版マスターテーブル構築スクリプト

Project Forge - 第一防衛線通過後の特徴量統合システム

統合設計図V準拠：
- Polars高速データ処理
- join_asof結合による時間軸統合
- 日次パーティション化保存
- Pylance厳格型定義準拠
"""

import sys
from pathlib import Path
from dataclasses import dataclass
from typing import List, Optional
import logging
from datetime import datetime
import psutil

import polars as pl
import joblib

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('build_master_table.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)


@dataclass
class MasterTableConfig:
    """
    マスターテーブル構築の設定を一元管理するdataclass
    
    Attributes:
        feature_dir: 個別特徴量ファイルが格納されているディレクトリ
        stable_list_path: 安定特徴量リストのパス（.joblib）
        output_dir: 出力先ディレクトリ
        output_name: 出力ファイル名（パーティションディレクトリ名）
        timeframes: 処理対象の時間足リスト（tickを除く）
        memory_warning_gb: メモリ警告閾値（GB）
        memory_critical_gb: メモリ緊急停止閾値（GB）
    """
    feature_dir: Path
    stable_list_path: Path
    output_dir: Path
    output_name: str = "stable_master_table"
    timeframes: Optional[List[str]] = None
    memory_warning_gb: float = 50.0
    memory_critical_gb: float = 55.0
    
    def __post_init__(self) -> None:
        """デフォルト値の設定とパス検証"""
        if self.timeframes is None:
            self.timeframes = [
                "M0.5", "M1", "M3", "M5", "M8", "M15", "M30",
                "H1", "H4", "H6", "H12", "D1", "W1", "MN"
            ]
        
        self.feature_dir = Path(self.feature_dir)
        self.stable_list_path = Path(self.stable_list_path)
        self.output_dir = Path(self.output_dir)
        
        if not self.feature_dir.exists():
            raise FileNotFoundError(f"特徴量ディレクトリが見つかりません: {self.feature_dir}")
        if not self.stable_list_path.exists():
            raise FileNotFoundError(f"安定特徴量リストが見つかりません: {self.stable_list_path}")
        
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info("設定初期化完了:")
        logger.info(f"  特徴量ディレクトリ: {self.feature_dir}")
        logger.info(f"  安定リスト: {self.stable_list_path}")
        logger.info(f"  出力先: {self.output_dir / self.output_name}")


class MemoryMonitor:
    """メモリ使用量を監視し、閾値超過時に警告・停止を実行するクラス"""
    
    def __init__(self, warning_gb: float, critical_gb: float):
        """
        Args:
            warning_gb: 警告閾値（GB）
            critical_gb: 緊急停止閾値（GB）
        """
        self.warning_gb = warning_gb
        self.critical_gb = critical_gb
        self.warning_issued = False
    
    def check(self, operation: str = "操作") -> None:
        """
        現在のメモリ使用量をチェックし、必要に応じて警告・停止を実行
        
        Args:
            operation: 実行中の操作名（ログ出力用）
        
        Raises:
            MemoryError: メモリ使用量が緊急停止閾値を超えた場合
        """
        memory_info = psutil.virtual_memory()
        used_gb = memory_info.used / (1024 ** 3)
        
        if used_gb > self.critical_gb:
            error_msg = (
                f"メモリ使用量が緊急停止閾値を超えました: {used_gb:.2f}GB > {self.critical_gb}GB "
                f"({operation}中)"
            )
            logger.error(error_msg)
            raise MemoryError(error_msg)
        
        if used_gb > self.warning_gb and not self.warning_issued:
            logger.warning(
                f"メモリ使用量が警告閾値を超えました: {used_gb:.2f}GB > {self.warning_gb}GB "
                f"({operation}中)"
            )
            self.warning_issued = True
        
        logger.debug(f"メモリ使用量: {used_gb:.2f}GB / {memory_info.total / (1024 ** 3):.2f}GB")


def load_stable_feature_list(path: Path) -> List[str]:
    """
    安定特徴量リストを読み込む
    
    Args:
        path: stable_feature_list.joblibのパス
    
    Returns:
        安定特徴量の名前リスト
    
    Raises:
        FileNotFoundError: ファイルが存在しない場合
        ValueError: ファイルの形式が不正な場合
    """
    logger.info(f"安定特徴量リストを読み込み中: {path}")
    
    try:
        stable_features: List[str] = joblib.load(path)
    except Exception as e:
        logger.error(f"安定特徴量リストの読み込みに失敗: {e}")
        raise
    
    if not isinstance(stable_features, list):
        raise ValueError(
            f"安定特徴量リストの形式が不正です。リストが期待されていますが、"
            f"{type(stable_features)}が見つかりました"
        )
    
    if not all(isinstance(f, str) for f in stable_features):
        raise ValueError("安定特徴量リストに文字列以外の要素が含まれています")
    
    logger.info(f"安定特徴量リスト読み込み完了: {len(stable_features)}個の特徴量")
    
    return stable_features


def identify_tick_features(stable_features: List[str]) -> List[str]:
    """
    安定特徴量リストからtick専用特徴量（_tickで終わる）を特定
    
    Args:
        stable_features: 安定特徴量の名前リスト
    
    Returns:
        tick専用特徴量の名前リスト
    """
    tick_features = [f for f in stable_features if f.endswith("_tick")]
    logger.info(f"tick専用特徴量を特定: {len(tick_features)}個")
    
    return tick_features


def find_feature_file(
    feature_dir: Path,
    timeframe: str,
    engine_prefixes: Optional[List[str]] = None
) -> Optional[Path]:
    """
    指定された時間足の特徴量ファイルを検索
    
    Args:
        feature_dir: 特徴量ファイルが格納されているディレクトリ
        timeframe: 時間足（例: "tick", "M1", "H1"）
        engine_prefixes: 検索対象のエンジンプレフィックスリスト
    
    Returns:
        見つかったファイルのパス、または見つからない場合はNone
    """
    if engine_prefixes is None:
        engine_prefixes = [f"e{i}{chr(ord('a')+j)}" for i in range(1, 10) for j in range(6)]
    
    for prefix in engine_prefixes:
        file_path = feature_dir / f"features_{prefix}_{timeframe}.parquet"
        if file_path.exists():
            return file_path
        
        dir_path = feature_dir / f"features_{prefix}_{timeframe}"
        if dir_path.exists() and dir_path.is_dir():
            return dir_path
    
    return None


def load_master_timeaxis(
    feature_dir: Path,
    tick_features: List[str],
    memory_monitor: MemoryMonitor
) -> pl.DataFrame:
    """
    tickデータからマスター時間軸を作成
    
    Args:
        feature_dir: 特徴量ファイルが格納されているディレクトリ
        tick_features: tick専用特徴量の名前リスト
        memory_monitor: メモリ監視オブジェクト
    
    Returns:
        マスター時間軸DataFrame（timestampでソート済み）
    
    Raises:
        FileNotFoundError: tickデータファイルが見つからない場合
        MemoryError: メモリ不足の場合
    """
    logger.info("=" * 80)
    logger.info("マスター時間軸（tickデータ）の作成を開始")
    logger.info("=" * 80)
    
    memory_monitor.check("マスター時間軸作成開始前")
    
    tick_file = find_feature_file(feature_dir, "tick")
    if tick_file is None:
        raise FileNotFoundError(
            f"tickデータファイルが見つかりません: {feature_dir}/features_*_tick.parquet"
        )
    
    logger.info(f"tickデータファイル: {tick_file}")
    
    base_columns = ["timestamp", "open", "high", "low", "close", "volume"]
    select_columns = base_columns + tick_features if tick_features else None
    
    try:
        if tick_file.is_dir():
            logger.info("パーティション化されたtickデータを読み込み中...")
            master_df = pl.read_parquet(tick_file, columns=select_columns)
        else:
            logger.info("単一tickファイルを読み込み中...")
            master_df = pl.read_parquet(tick_file, columns=select_columns)
        
        memory_monitor.check("tickデータ読み込み後")
        
        logger.info("timestampでソート中...")
        master_df = master_df.sort("timestamp")
        
        memory_monitor.check("ソート後")
        
        logger.info(f"マスター時間軸作成完了: {len(master_df):,}行 × {len(master_df.columns)}列")
        logger.info(f"列: {master_df.columns}")
        
        return master_df
    
    except Exception as e:
        logger.error(f"マスター時間軸の作成中にエラーが発生: {e}")
        raise


def join_timeframe_features(
    master_df: pl.DataFrame,
    feature_dir: Path,
    timeframe: str,
    stable_features: List[str],
    memory_monitor: MemoryMonitor
) -> pl.DataFrame:
    """
    指定された時間足の特徴量をマスターテーブルに結合
    
    Args:
        master_df: マスター時間軸DataFrame
        feature_dir: 特徴量ファイルが格納されているディレクトリ
        timeframe: 時間足（例: "M1", "H1"）
        stable_features: 安定特徴量の名前リスト
        memory_monitor: メモリ監視オブジェクト
    
    Returns:
        特徴量が結合されたDataFrame
    
    Raises:
        FileNotFoundError: 特徴量ファイルが見つからない場合
        MemoryError: メモリ不足の場合
    """
    logger.info("-" * 80)
    logger.info(f"時間足 {timeframe} の特徴量を結合中...")
    logger.info("-" * 80)
    
    memory_monitor.check(f"{timeframe}結合開始前")
    
    feature_file = find_feature_file(feature_dir, timeframe)
    if feature_file is None:
        logger.warning(f"時間足 {timeframe} の特徴量ファイルが見つかりません。スキップします。")
        return master_df
    
    logger.info(f"特徴量ファイル: {feature_file}")
    
    timeframe_suffix = f"_{timeframe.replace('.', '')}"
    timeframe_features = [
        f for f in stable_features
        if f.endswith(timeframe_suffix)
    ]
    
    if not timeframe_features:
        logger.info(f"時間足 {timeframe} に安定特徴量が存在しません。スキップします。")
        return master_df
    
    logger.info(f"結合対象の安定特徴量: {len(timeframe_features)}個")
    
    select_columns = ["timestamp"] + timeframe_features
    
    try:
        if feature_file.is_dir():
            logger.info("パーティション化データを読み込み中...")
            feature_df = pl.read_parquet(feature_file, columns=select_columns)
        else:
            logger.info("単一ファイルを読み込み中...")
            feature_df = pl.read_parquet(feature_file, columns=select_columns)
        
        memory_monitor.check(f"{timeframe}データ読み込み後")
        
        logger.info("timestampでソート中...")
        feature_df = feature_df.sort("timestamp")
        
        memory_monitor.check(f"{timeframe}ソート後")
        
        logger.info(f"特徴量データ: {len(feature_df):,}行 × {len(feature_df.columns)}列")
        
        logger.info("join_asofで結合中...")
        master_df = master_df.join_asof(
            feature_df,
            on="timestamp",
            strategy="backward"
        )
        
        memory_monitor.check(f"{timeframe}結合後")
        
        logger.info(f"結合完了: 現在の列数 = {len(master_df.columns)}")
        
        return master_df
    
    except Exception as e:
        logger.error(f"時間足 {timeframe} の結合中にエラーが発生: {e}")
        raise


def save_master_table(
    master_df: pl.DataFrame,
    output_dir: Path,
    output_name: str,
    memory_monitor: MemoryMonitor
) -> None:
    """
    マスターテーブルを日次パーティション化して保存
    
    Args:
        master_df: 保存するマスターテーブル
        output_dir: 出力ディレクトリ
        output_name: 出力名（パーティションディレクトリ名）
        memory_monitor: メモリ監視オブジェクト
    
    Raises:
        MemoryError: メモリ不足の場合
    """
    logger.info("=" * 80)
    logger.info("マスターテーブルの保存を開始")
    logger.info("=" * 80)
    
    memory_monitor.check("保存開始前")
    
    logger.info("年月日カラムを追加中...")
    master_df = master_df.with_columns([
        pl.col("timestamp").dt.year().alias("year"),
        pl.col("timestamp").dt.month().alias("month"),
        pl.col("timestamp").dt.day().alias("day")
    ])
    
    output_path = output_dir / output_name
    
    logger.info(f"出力先: {output_path}")
    logger.info(f"データサイズ: {len(master_df):,}行 × {len(master_df.columns)}列")
    logger.info("パーティション化して保存中...")
    
    try:
        master_df.write_parquet(
            output_path,
            compression="snappy",
            partition_by=["year", "month", "day"]
        )
        
        memory_monitor.check("保存完了後")
        
        logger.info("=" * 80)
        logger.info("マスターテーブルの保存が完了しました")
        logger.info("=" * 80)
        
    except Exception as e:
        logger.error(f"マスターテーブルの保存中にエラーが発生: {e}")
        raise


def build_master_table(config: MasterTableConfig) -> None:
    """
    安定版マスターテーブルを構築するメイン処理
    
    Args:
        config: マスターテーブル構築の設定
    
    Raises:
        FileNotFoundError: 入力ファイルが見つからない場合
        MemoryError: メモリ不足の場合
    """
    start_time = datetime.now()
    logger.info("=" * 80)
    logger.info("安定版マスターテーブル構築を開始")
    logger.info("=" * 80)
    logger.info(f"開始時刻: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    
    memory_monitor = MemoryMonitor(
        warning_gb=config.memory_warning_gb,
        critical_gb=config.memory_critical_gb
    )
    
    try:
        logger.info("\n【ステップ1】安定特徴量リストの読み込み")
        stable_features = load_stable_feature_list(config.stable_list_path)
        
        logger.info("\n【ステップ2】tick専用特徴量の特定")
        tick_features = identify_tick_features(stable_features)
        
        logger.info("\n【ステップ3】マスター時間軸の作成")
        master_df = load_master_timeaxis(
            config.feature_dir,
            tick_features,
            memory_monitor
        )
        
        logger.info("\n【ステップ4】各時間足の特徴量を結合")
        if config.timeframes is None:
            raise ValueError("Timeframes not configured.")
        
        logger.info(f"処理対象時間足: {', '.join(config.timeframes)}")
        
        for i, timeframe in enumerate(config.timeframes, 1):
            logger.info(f"\n[{i}/{len(config.timeframes)}] {timeframe} を処理中...")
            
            master_df = join_timeframe_features(
                master_df,
                config.feature_dir,
                timeframe,
                stable_features,
                memory_monitor
            )
        
        logger.info("\n【ステップ5】マスターテーブルの保存")
        save_master_table(
            master_df,
            config.output_dir,
            config.output_name,
            memory_monitor
        )
        
        end_time = datetime.now()
        elapsed_time = end_time - start_time
        
        logger.info("=" * 80)
        logger.info("全処理が正常に完了しました")
        logger.info("=" * 80)
        logger.info(f"開始時刻: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
        logger.info(f"終了時刻: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
        logger.info(f"処理時間: {elapsed_time}")
        logger.info(f"最終データサイズ: {len(master_df):,}行 × {len(master_df.columns)}列")
        
    except Exception as e:
        logger.error("=" * 80)
        logger.error("処理中に致命的なエラーが発生しました")
        logger.error("=" * 80)
        logger.error(f"エラー内容: {e}", exc_info=True)
        raise


def main() -> None:
    """メインエントリーポイント"""
    print("=" * 80)
    print("安定版マスターテーブル構築スクリプト")
    print("Project Forge - 第一防衛線通過特徴量の統合")
    print("=" * 80)
    print()
    
    default_feature_dir = Path("/workspaces/project_forge/data/2_feature_value")
    default_stable_list = Path("/workspaces/project_forge/data/3_validation_results/stable_feature_list.joblib")
    default_output_dir = Path("/workspaces/project_forge/data/4_master_table")
    
    print("【設定】")
    feature_dir_input = input(f"特徴量ディレクトリのパス [{default_feature_dir}]: ").strip()
    feature_dir = Path(feature_dir_input) if feature_dir_input else default_feature_dir
    
    stable_list_input = input(f"安定特徴量リストのパス [{default_stable_list}]: ").strip()
    stable_list_path = Path(stable_list_input) if stable_list_input else default_stable_list
    
    output_dir_input = input(f"出力ディレクトリのパス [{default_output_dir}]: ").strip()
    output_dir = Path(output_dir_input) if output_dir_input else default_output_dir
    
    output_name = input("出力名（パーティションディレクトリ名） [stable_master_table]: ").strip() or "stable_master_table"
    
    memory_warning = input("メモリ警告閾値（GB） [50.0]: ").strip()
    memory_warning_gb = float(memory_warning) if memory_warning else 50.0
    
    memory_critical = input("メモリ緊急停止閾値（GB） [55.0]: ").strip()
    memory_critical_gb = float(memory_critical) if memory_critical else 55.0
    
    print()
    print("=" * 80)
    print("設定確認")
    print("=" * 80)
    print(f"特徴量ディレクトリ: {feature_dir}")
    print(f"安定特徴量リスト: {stable_list_path}")
    print(f"出力ディレクトリ: {output_dir}")
    print(f"出力名: {output_name}")
    print(f"メモリ警告閾値: {memory_warning_gb}GB")
    print(f"メモリ緊急停止閾値: {memory_critical_gb}GB")
    print("=" * 80)
    print()
    
    confirm = input("この設定で実行しますか？ (yes/no): ").strip().lower()
    if confirm not in ["yes", "y"]:
        print("処理を中止しました。")
        return
    
    print()
    
    try:
        config = MasterTableConfig(
            feature_dir=feature_dir,
            stable_list_path=stable_list_path,
            output_dir=output_dir,
            output_name=output_name,
            memory_warning_gb=memory_warning_gb,
            memory_critical_gb=memory_critical_gb
        )
    except Exception as e:
        logger.error(f"設定の初期化に失敗: {e}")
        return
    
    try:
        build_master_table(config)
    except Exception as e:
        logger.error(f"マスターテーブルの構築に失敗: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()