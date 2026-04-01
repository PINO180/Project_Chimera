"""
Kolmogorov-Smirnov.py v5.3 - 極限メモリ最適化＆物理フィルタリング版
第一防衛線 - 分布安定性テスト（KS検定のみ）

アーキテクチャ：
- Daskを完全排除し、ProcessPoolExecutorによる厳格なメモリ上限管理へ移行
- Polarsの遅延評価とFloat32キャストによるメモリ消費の75%削減
- 【NEW】KS検定合格後のデータを S2_FEATURES_AFTER_KS へ物理保存する機能を追加
"""

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

import blueprint as config
import logging
import time
import json
import gc
import multiprocessing
import shutil
from datetime import datetime
from typing import List, Set, Tuple
from concurrent.futures import ProcessPoolExecutor, as_completed

from tqdm import tqdm
import pyarrow.parquet as pq

# --- ロギング設定 ---
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


# --- 設定クラス ---
class ProcessingConfig:
    def __init__(self):
        self.feature_universe_path = str(config.S2_FEATURES_FIXED)
        self.output_dir = config.S3_ARTIFACTS
        self.after_ks_dir = config.S2_FEATURES_AFTER_KS  # ★追加：物理保存先
        self.ks_p_value_threshold = 0.05
        self.sample_size = 30_000


# --- 本体クラス ---
class KSTestValidator:
    def __init__(self, cfg: ProcessingConfig):
        self.cfg = cfg

    def _get_data_sources(self) -> List[Path]:
        feature_universe_path = Path(self.cfg.feature_universe_path)
        if not feature_universe_path.exists() or not feature_universe_path.is_dir():
            raise FileNotFoundError(
                f"ディレクトリが見つかりません: {feature_universe_path}"
            )

        all_data_sources = []
        for engine_dir in feature_universe_path.iterdir():
            if not engine_dir.is_dir():
                continue
            for item in engine_dir.iterdir():
                if item.is_dir() or (item.is_file() and item.suffix == ".parquet"):
                    all_data_sources.append(item)

        if not all_data_sources:
            raise FileNotFoundError("データソースが見つかりません")

        logger.info(f"{len(all_data_sources)}個のデータソースを検出")
        return all_data_sources

    @staticmethod
    def _run_single_ks_test(
        path_str: str,
        column_name: str,
        p_value_threshold: float,
        sample_size: int,
        feature_name: str,
    ) -> Tuple[str, bool, str]:
        """単一の特徴量カラムに対してKS検定を実行する（極限省メモリ＆Hiveパーテーション対応）"""
        try:
            import polars as pl
            import numpy as np
            from scipy.stats import ks_2samp
            import os
            import re
            from pathlib import Path
            import gc

            # Polarsの内部マルチスレッド暴走をワーカー単位で強制封印
            os.environ["POLARS_MAX_THREADS"] = "1"

            path_obj = Path(path_str)

            if path_obj.is_dir():
                # 【Hiveパーテーション対応：季節性同期カレンダー比較】
                # 過去ブロック: 2021/8 〜 2022/7
                # 現在ブロック: 2024/8 〜 2025/7
                past_files = []
                present_files = []

                for p in path_obj.rglob("*.parquet"):
                    match = re.search(r"year=(\d+)[/\\]month=(\d+)", str(p))
                    if match:
                        y, m = int(match.group(1)), int(match.group(2))
                        # 過去ブロック: 2021/8 〜 2022/7
                        if (y == 2021 and m >= 8) or (y == 2022 and m <= 7):
                            past_files.append(str(p))
                        # 現在ブロック: 2024/8 〜 2025/7
                        elif (y == 2024 and m >= 8) or (y == 2025 and m <= 7):
                            present_files.append(str(p))

                if not past_files or not present_files:
                    return (feature_name, True, "指定された比較期間のデータが不足")

                # 指定期間のファイルをPolarsで一括読み込み＆Numpy配列化
                first_half = (
                    pl.scan_parquet(past_files)
                    .select([pl.col(column_name).cast(pl.Float32)])
                    .drop_nulls()
                    .collect()
                    .to_series()
                    .to_numpy()
                )
                second_half = (
                    pl.scan_parquet(present_files)
                    .select([pl.col(column_name).cast(pl.Float32)])
                    .drop_nulls()
                    .collect()
                    .to_series()
                    .to_numpy()
                )

            else:
                # 非Tick（M1, D1など）は単一ファイルで小さいので一括読み込み
                df = (
                    pl.scan_parquet(path_str)
                    .select([pl.col(column_name).cast(pl.Float32)])
                    .drop_nulls()
                    .collect()
                )
                arr = df.to_series().to_numpy()
                del df  # 即座にPolarsのメモリ解放

                n = len(arr)
                if n < 200:
                    return (feature_name, True, "サンプル数不足")

                split_point = n // 2
                first_half = arr[:split_point]
                second_half = arr[split_point:]

            # --- サンプリングとKS検定 ---
            if len(first_half) < 10 or len(second_half) < 10:
                return (feature_name, True, "サンプル数不足")

            if len(first_half) > sample_size:
                np.random.seed(42)
                first_half = np.random.choice(first_half, sample_size, replace=False)
            if len(second_half) > sample_size:
                np.random.seed(42)
                second_half = np.random.choice(second_half, sample_size, replace=False)

            _, p_value = ks_2samp(first_half, second_half)
            is_stable = p_value >= p_value_threshold

            # 念押しのガベージコレクション
            del first_half, second_half
            if not path_obj.is_dir():
                del arr
            gc.collect()

            return (feature_name, is_stable, f"p_value={p_value:.4f}")

        except Exception as e:
            return (feature_name, False, f"エラー: {str(e)[:50]}")

    def run_validation(self) -> None:
        start_time = time.time()
        logger.info("=" * 60)
        logger.info("KS検定（V5.3 - 極限メモリ最適化＆物理フィルタリング版）を開始")
        logger.info("=" * 60)

        data_sources = self._get_data_sources()
        tasks = []
        total_feature_count = 0

        logger.info("全データソースをスキャンし、検定タスクを生成中...")
        for path in tqdm(data_sources, desc="データソーススキャン中", ncols=100):
            try:
                if path.is_dir():
                    first_file = next(path.glob("**/*.parquet"), None)
                    if first_file is None:
                        continue
                    schema = pq.read_schema(first_file)
                else:
                    schema = pq.read_schema(path)

                columns = schema.names
                non_feature_columns = [
                    "timestamp",
                    "open",
                    "high",
                    "low",
                    "close",
                    "volume",
                    "timeframe",
                ]
                timeframe_suffix = (
                    "_tick"
                    if path.is_dir() or "_tick" in path.name
                    else f"_{path.stem.split('_')[-1]}"
                    if len(path.stem.split("_")) > 1
                    else ""
                )

                for col in columns:
                    if col not in non_feature_columns:
                        final_feature_name = f"{col}{timeframe_suffix}"
                        tasks.append(
                            (
                                str(path),
                                col,
                                self.cfg.ks_p_value_threshold,
                                self.cfg.sample_size,
                                final_feature_name,
                            )
                        )
                        total_feature_count += 1
            except Exception as e:
                logger.warning(f"'{path.name}' のスキャンに失敗: {e}")

        logger.info(f"{total_feature_count}個の検定タスク生成完了。")

        # --- 極限メモリ管理のための並列制御 ---
        MAX_WORKERS = max(1, min(multiprocessing.cpu_count() // 2, 6))
        logger.info(f"並列実行を開始します (安全ワーカー数制限: {MAX_WORKERS})")

        unstable_features: Set[str] = set()

        with ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
            future_to_task = {
                executor.submit(
                    self._run_single_ks_test, t[0], t[1], t[2], t[3], t[4]
                ): t[4]
                for t in tasks
            }

            for future in tqdm(
                as_completed(future_to_task),
                total=len(tasks),
                desc="検定実行中",
                ncols=100,
            ):
                feature_name, is_stable, reason = future.result()
                if not is_stable:
                    unstable_features.add(feature_name)

        # ==========================================
        # ★追加部分：物理フィルタリングと保存処理
        # ==========================================
        logger.info(f"検定完了。合格データを {self.cfg.after_ks_dir} へ出力します...")
        import polars as pl

        if self.cfg.after_ks_dir.exists():
            shutil.rmtree(self.cfg.after_ks_dir)
        self.cfg.after_ks_dir.mkdir(parents=True, exist_ok=True)

        for path in tqdm(data_sources, desc="合格データ物理保存中", ncols=100):
            timeframe_suffix = (
                "_tick"
                if path.is_dir() or "_tick" in path.name
                else f"_{path.stem.split('_')[-1]}"
                if len(path.stem.split("_")) > 1
                else ""
            )

            # スキーマを読み込む
            if path.is_dir():
                first_file = next(path.glob("**/*.parquet"), None)
                if first_file is None:
                    continue
                schema = pq.read_schema(first_file)
            else:
                schema = pq.read_schema(path)

            # 合格したカラム + 必須カラムだけを抽出リストに
            keep_cols = [
                c
                for c in schema.names
                if f"{c}{timeframe_suffix}" not in unstable_features
                or c
                in [
                    "timestamp",
                    "open",
                    "high",
                    "low",
                    "close",
                    "volume",
                    "timeframe",
                    "year",
                    "month",
                    "day",
                ]
            ]

            target_base_path = self.cfg.after_ks_dir / path.parent.name / path.name

            if path.is_dir():
                # Hiveパーティション構造を維持するため、内部のファイルを個別に処理
                for file_path in path.rglob("*.parquet"):
                    rel_path = file_path.relative_to(path)
                    target_file_path = target_base_path / rel_path
                    target_file_path.parent.mkdir(parents=True, exist_ok=True)

                    pl.read_parquet(file_path, columns=keep_cols).write_parquet(
                        target_file_path
                    )
            else:
                target_base_path.parent.mkdir(parents=True, exist_ok=True)
                pl.read_parquet(path, columns=keep_cols).write_parquet(target_base_path)

        # ==========================================

        execution_time = time.time() - start_time

        logger.info("=" * 60)
        logger.info("KS検定＆物理保存完了")
        logger.info(f"検証特徴量総数: {total_feature_count}")
        logger.info(f"不安定特徴量数: {len(unstable_features)}")
        logger.info(f"安定特徴量数: {total_feature_count - len(unstable_features)}")
        logger.info(f"実行時間: {execution_time:.2f}秒")
        logger.info("=" * 60)

        self.save_results(unstable_features, total_feature_count, execution_time)

    def save_results(
        self, unstable_features: Set[str], total_count: int, execution_time: float
    ) -> None:
        self.cfg.output_dir.mkdir(parents=True, exist_ok=True)
        output_path = self.cfg.output_dir / "ks_unstable_features.json"

        results = {
            "unstable_features": sorted(list(unstable_features)),
            "test_name": "KS検定（v5.3 - 極限メモリ最適化＆物理フィルタリング版）",
            "total_features_tested": total_count,
            "unstable_count": len(unstable_features),
            "execution_time_seconds": round(execution_time, 2),
            "timestamp": datetime.now().isoformat(),
        }

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)

        logger.info(f"不安定特徴量のリストを保存しました: {output_path}")


def main():
    try:
        cfg = ProcessingConfig()
        validator = KSTestValidator(cfg)
        validator.run_validation()
        logger.info("✅ KS検定パイプラインが正常に完了しました。")
    except Exception as e:
        logger.critical(
            f"❌ パイプライン実行中に致命的なエラーが発生しました: {e}", exc_info=True
        )
        raise


if __name__ == "__main__":
    main()
