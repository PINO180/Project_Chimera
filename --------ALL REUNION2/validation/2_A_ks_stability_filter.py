"""
2_A_ks_stability_filter.py
第一防衛線 - 分布安定性フィルター（KS検定のみ・AV廃止版）

フロー：
    S2_FEATURES（全エンジン出力）を走査
        ↓
    時間足を判定
      → LF_ALL_TIMEFRAMES（H4/H6/H12/D1/W1/MN）: KS検定を実施
      → HF_TIMEFRAMES（M0.5〜H1）           : KS検定スキップ・そのままコピー
        ↓
    KS合格特徴量 + HF全特徴量 → S2_FEATURES_VALIDATED に物理保存
        ↓
    S3_ARTIFACTS / "ks_unstable_features.json" に除外特徴量リストを保存
"""

import sys
from pathlib import Path

# ルール7：ハードコードパス禁止・動的解決
sys.path.append(str(Path(__file__).resolve().parents[1]))

import blueprint as config
import glob
import json
import logging
import multiprocessing
import shutil
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
from typing import Dict, List, Set, Tuple

import re

import polars as pl
import pyarrow.parquet as pq
from tqdm import tqdm

# --- ロギング設定 ---
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# --- blueprintからパス・定数をインポート（ルール7・ルール9） ---
S2_FEATURES          = config.S2_FEATURES
S2_FEATURES_VALIDATED = config.S2_FEATURES_VALIDATED
S3_ARTIFACTS         = config.S3_ARTIFACTS
LF_ALL_TIMEFRAMES    = config.LF_ALL_TIMEFRAMES   # ["H4","H6","H12","D1","W1","MN"]
HF_TIMEFRAMES        = config.HF_TIMEFRAMES        # ["M0.5","M1",..."H1"]
KS_SAMPLE_SIZE       = config.KS_SAMPLE_SIZE       # 100_000
MAX_WORKERS_DIVISOR  = config.MAX_WORKERS_DIVISOR  # 1

# KS検定p値閾値（プロンプト仕様）
KS_P_VALUE_THRESHOLD = 0.05

# 必ず保持するメタカラム（プロンプト仕様）
META_COLS = {"timestamp", "timeframe", "year", "month", "day"}

# 旧スクリプトで除外していたOHLCVカラム群
OHLCV_COLS = {"open", "high", "low", "close", "volume"}


# =============================================================================
# Hiveパーティション読み込みユーティリティ（プロンプト仕様の関数をそのまま実装）
# =============================================================================

def load_hive_partition_for_period(
    base_dir: Path,
    year_start: int,
    month_start: int,
    year_end: int,
    month_end: int,
) -> pl.DataFrame:
    """Hiveパーティション構造から指定期間のファイルを収集してDataFrameを返す。"""
    files: List[str] = []
    for y in range(year_start, year_end + 1):
        for m in range(1, 13):
            if y == year_start and m < month_start:
                continue
            if y == year_end and m > month_end:
                continue
            pattern = base_dir / f"year={y}" / f"month={m}" / "*.parquet"
            files.extend(glob.glob(str(pattern)))
    if not files:
        return pl.DataFrame()
    return pl.read_parquet(files)


# =============================================================================
# データソーススキャン
# =============================================================================

def _get_data_sources() -> List[Path]:
    """S2_FEATURES 配下のサブフォルダ・ファイルを列挙する。"""
    if not S2_FEATURES.exists() or not S2_FEATURES.is_dir():
        raise FileNotFoundError(f"S2_FEATURES ディレクトリが見つかりません: {S2_FEATURES}")

    sources: List[Path] = []
    for engine_dir in S2_FEATURES.iterdir():
        if not engine_dir.is_dir():
            continue
        for item in engine_dir.iterdir():
            if item.is_dir() or (item.is_file() and item.suffix == ".parquet"):
                sources.append(item)

    if not sources:
        raise FileNotFoundError("S2_FEATURES 配下にデータソースが見つかりません")

    logger.info(f"{len(sources)} 個のデータソースを検出しました")
    return sources


def _detect_timeframe(path: Path) -> str:
    """パス名から時間足文字列を推定する。

    部分一致（tf.lower() in name.lower()）ではなく単語境界マッチを使用する。
    例: "M1" が "features_M15.parquet" にヒットしないよう (?<![A-Za-z0-9]) / (?![A-Za-z0-9]) で境界を判定。
    """
    # ファイル名（ファイルの場合）または直上ディレクトリ名（Hiveの場合）を優先チェック
    name = path.name if path.is_file() else path.parent.name
    for tf in LF_ALL_TIMEFRAMES + HF_TIMEFRAMES:
        pattern = r"(?<![A-Za-z0-9])" + re.escape(tf) + r"(?![A-Za-z0-9])"
        if re.search(pattern, name, re.IGNORECASE):
            return tf
    # 上記でヒットしなければパス全体の各パーツを走査（Hiveパーティション対応）
    for part in path.parts:
        for tf in LF_ALL_TIMEFRAMES + HF_TIMEFRAMES:
            pattern = r"(?<![A-Za-z0-9])" + re.escape(tf) + r"(?![A-Za-z0-9])"
            if re.search(pattern, part, re.IGNORECASE):
                return tf
    return "unknown"


# =============================================================================
# KS検定タスク（ProcessPoolExecutor用・静的メソッド）
# =============================================================================

def _run_single_ks_test(
    path_str: str,
    column_name: str,
    p_value_threshold: float,
    sample_size: int,
    feature_name: str,
) -> Tuple[str, bool, float, str]:
    """
    単一の特徴量カラムに対してKS検定を実行する。

    Returns
    -------
    (feature_name, is_stable, p_value, reason)
    """
    import gc
    import os

    import numpy as np
    import polars as pl
    from scipy.stats import ks_2samp

    # Polarsの内部マルチスレッドをワーカー単位で封印（メモリ暴走防止）
    os.environ["POLARS_MAX_THREADS"] = "1"

    try:
        path_obj = Path(path_str)

        if path_obj.is_dir():
            # --- Hiveパーティション構造：季節同期比較 ---
            # 過去ブロック: 2021-08 〜 2022-07
            # 現在ブロック: 2024-08 〜 2025-07
            past_files: List[str] = []
            present_files: List[str] = []

            for p in path_obj.rglob("*.parquet"):
                parts_str = str(p)
                # year=N/month=M パターンを検索
                import re
                match = re.search(r"year=(\d+)[/\\]month=(\d+)", parts_str)
                if match:
                    y, m = int(match.group(1)), int(match.group(2))
                    if (y == 2021 and m >= 8) or (y == 2022 and m <= 7):
                        past_files.append(parts_str)
                    elif (y == 2024 and m >= 8) or (y == 2025 and m <= 7):
                        present_files.append(parts_str)

            if not past_files or not present_files:
                return (feature_name, True, float("nan"), "指定期間のデータが不足")

            # ルール8: Float32で読み込み（KS検定は分布形のみ比較）
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
            # --- 非Hive（単一ファイル）：前半・後半に二分割して比較 ---
            # プロンプト仕様：「全行を読み込み、timestampでソート後に前半・後半に二分割」
            df = (
                pl.scan_parquet(path_str)
                .select(["timestamp", pl.col(column_name).cast(pl.Float32)])
                .drop_nulls()
                .sort("timestamp")
                .collect()
            )
            arr = df.get_column(column_name).to_numpy()
            del df
            gc.collect()

            n = len(arr)
            if n < 200:
                return (feature_name, True, float("nan"), "サンプル数不足")

            split_point = n // 2
            first_half = arr[:split_point]
            second_half = arr[split_point:]

        # --- サンプリング（ルール4: シードを固定） ---
        if len(first_half) < 10 or len(second_half) < 10:
            return (feature_name, True, float("nan"), "サンプル数不足")

        rng = np.random.default_rng(seed=42)
        if len(first_half) > sample_size:
            first_half = rng.choice(first_half, sample_size, replace=False)
        if len(second_half) > sample_size:
            second_half = rng.choice(second_half, sample_size, replace=False)

        # --- KS検定 ---
        _, p_value = ks_2samp(first_half, second_half)
        is_stable = bool(p_value >= p_value_threshold)

        del first_half, second_half
        gc.collect()

        return (feature_name, is_stable, float(p_value), f"p_value={p_value:.6f}")

    except Exception as e:
        return (feature_name, False, float("nan"), f"エラー: {str(e)[:80]}")


# =============================================================================
# メイン処理
# =============================================================================

def run_ks_filter() -> None:
    start_time = time.time()
    logger.info("=" * 70)
    logger.info("2_A_ks_stability_filter.py を開始します")
    logger.info(f"  入力 : {S2_FEATURES}")
    logger.info(f"  出力 : {S2_FEATURES_VALIDATED}")
    logger.info(f"  成果物: {S3_ARTIFACTS}")
    logger.info("=" * 70)

    data_sources = _get_data_sources()

    # =========================================================================
    # Step 1: タスク生成・時間足判定
    # =========================================================================
    logger.info("Step 1: データソースをスキャンし、検定タスクを生成中...")

    # LFタスク: KS検定が必要
    lf_tasks: List[Tuple] = []
    # HFソース: KS検定スキップ（そのままコピー）
    hf_sources: List[Path] = []
    # すべての特徴量カラム名を追跡（保存時のフィルタ用）
    lf_feature_names_per_source: Dict[str, List[str]] = {}  # path_str -> [feature_name, ...]
    hf_feature_names_per_source: Dict[str, List[str]] = {}

    for path in tqdm(data_sources, desc="スキャン中", ncols=100):
        try:
            if path.is_dir():
                first_file = next(path.glob("**/*.parquet"), None)
                if first_file is None:
                    continue
                schema = pq.read_schema(first_file)
            else:
                schema = pq.read_schema(path)
        except Exception as e:
            logger.warning(f"スキーマ読み込み失敗: {path.name} - {e}")
            continue

        tf = _detect_timeframe(path)
        is_lf = tf in LF_ALL_TIMEFRAMES
        is_hf = tf in HF_TIMEFRAMES

        # 特徴量カラム = 全カラム - meta/OHLCV
        skip_cols = META_COLS | OHLCV_COLS
        feature_cols = [c for c in schema.names if c not in skip_cols]

        path_str = str(path)

        if is_lf:
            lf_feature_names_per_source[path_str] = []
            for col in feature_cols:
                # 特徴量識別子: "{col}__{tf}" 形式でタスクを識別する。
                # 前提: エンジンの特徴量カラム名には "__" が含まれない（e1a_statistical_mean_20_M1 形式）。
                # "__" を含むカラム名が存在する場合は rsplit("__", 1) による復元が誤動作する可能性がある。
                feature_name = f"{col}__{tf}"
                lf_tasks.append(
                    (path_str, col, KS_P_VALUE_THRESHOLD, KS_SAMPLE_SIZE, feature_name)
                )
                lf_feature_names_per_source[path_str].append(feature_name)
        elif is_hf:
            hf_sources.append(path)
            hf_feature_names_per_source[path_str] = [
                f"{col}__{tf}" for col in feature_cols
            ]
        else:
            # 時間足不明はLFとして扱い検定対象にする（安全側）
            logger.warning(f"時間足を特定できませんでした（LFとして検定対象にします）: {path.name}")
            lf_feature_names_per_source[path_str] = []
            for col in feature_cols:
                feature_name = f"{col}__unknown"
                lf_tasks.append(
                    (path_str, col, KS_P_VALUE_THRESHOLD, KS_SAMPLE_SIZE, feature_name)
                )
                lf_feature_names_per_source[path_str].append(feature_name)

    logger.info(f"  LFタスク数 : {len(lf_tasks)}")
    logger.info(f"  HFソース数 : {len(hf_sources)}（KS検定スキップ）")

    # =========================================================================
    # Step 2: LF特徴量に対してKS検定（並列実行）
    # =========================================================================
    logger.info("Step 2: LF特徴量のKS検定を開始します...")

    # ルール: cpu_count() // MAX_WORKERS_DIVISOR
    max_workers = max(1, multiprocessing.cpu_count() // MAX_WORKERS_DIVISOR)
    logger.info(f"  並列ワーカー数: {max_workers}（MAX_WORKERS_DIVISOR={MAX_WORKERS_DIVISOR}）")

    unstable_features: Set[str] = set()
    ks_results: List[Dict] = []  # JSON保存用の詳細結果

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        future_to_name = {
            executor.submit(_run_single_ks_test, *task): task[4]
            for task in lf_tasks
        }
        for future in tqdm(
            as_completed(future_to_name),
            total=len(lf_tasks),
            desc="KS検定実行中",
            ncols=100,
        ):
            feature_name, is_stable, p_value, reason = future.result()
            ks_results.append(
                {
                    "feature": feature_name,
                    "is_stable": is_stable,
                    "p_value": p_value if not (p_value != p_value) else None,  # NaN→None
                    "reason": reason,
                }
            )
            if not is_stable:
                unstable_features.add(feature_name)

    stable_count = len(lf_tasks) - len(unstable_features)
    logger.info(f"  検定完了: 合格={stable_count} / 不合格={len(unstable_features)} / 計={len(lf_tasks)}")

    # =========================================================================
    # Step 3: S2_FEATURES_VALIDATED への物理保存
    # =========================================================================
    logger.info("Step 3: S2_FEATURES_VALIDATED へ物理保存を開始します...")

    # ディレクトリを初期化（プロンプト仕様通り）
    if S2_FEATURES_VALIDATED.exists():
        shutil.rmtree(S2_FEATURES_VALIDATED)
    S2_FEATURES_VALIDATED.mkdir(parents=True, exist_ok=True)

    def _save_filtered(src_path: Path, survived_cols: List[str]) -> None:
        """合格カラム + メタカラムを保持してS2_FEATURES_VALIDATEDへ書き出す。"""
        # S2_FEATURES からの相対パスを維持してコピー先を決定
        rel = src_path.relative_to(S2_FEATURES)
        dest = S2_FEATURES_VALIDATED / rel

        if src_path.is_dir():
            # Hiveパーティション：内部ファイルを個別に処理してディレクトリ構造を維持
            for file_path in src_path.rglob("*.parquet"):
                rel_inner = file_path.relative_to(src_path)
                dest_file = dest / rel_inner
                dest_file.parent.mkdir(parents=True, exist_ok=True)
                df = pl.read_parquet(file_path)
                keep = [c for c in df.columns if c in survived_cols]
                # ルール8: Float32で書き出し
                df.select(keep).with_columns(
                    [pl.col(c).cast(pl.Float32) for c in keep if c not in META_COLS]
                ).write_parquet(dest_file, compression="zstd")
        else:
            dest.parent.mkdir(parents=True, exist_ok=True)
            df = pl.read_parquet(src_path)
            keep = [c for c in df.columns if c in survived_cols]
            # ルール8: Float32で書き出し
            df.select(keep).with_columns(
                [pl.col(c).cast(pl.Float32) for c in keep if c not in META_COLS]
            ).write_parquet(dest, compression="zstd")

    # --- LFソース: 合格特徴量 + メタカラムのみ保存 ---
    lf_source_paths = list(lf_feature_names_per_source.keys())
    for path_str in tqdm(lf_source_paths, desc="LF合格データ保存中", ncols=100):
        path = Path(path_str)
        feature_names_for_path = lf_feature_names_per_source[path_str]

        # 合格した特徴量の元カラム名を復元（"col__tf" → "col"）
        survived_original_cols = set()
        for feat_name in feature_names_for_path:
            if feat_name not in unstable_features:
                # サフィックスを除去してオリジナルカラム名に戻す
                col = feat_name.rsplit("__", 1)[0]
                survived_original_cols.add(col)

        # メタカラムを必ず追加
        survived_cols = list(survived_original_cols | META_COLS)

        if not survived_original_cols:
            logger.warning(f"全特徴量が除外されました（メタカラムのみ保存）: {path.name}")

        try:
            _save_filtered(path, survived_cols)
        except Exception as e:
            logger.error(f"保存失敗: {path.name} - {e}")

    # --- HFソース: KS検定なし・全特徴量をそのままコピー ---
    for path in tqdm(hf_sources, desc="HF全特徴量コピー中", ncols=100):
        try:
            if path.is_dir():
                first_file = next(path.glob("**/*.parquet"), None)
                if first_file is None:
                    continue
                schema = pq.read_schema(first_file)
            else:
                schema = pq.read_schema(path)
            all_cols = schema.names  # 全カラムを保持
            _save_filtered(path, all_cols)
        except Exception as e:
            logger.error(f"HFコピー失敗: {path.name} - {e}")

    # =========================================================================
    # Step 4: 成果物の保存（ks_unstable_features.json）
    # =========================================================================
    logger.info("Step 4: S3_ARTIFACTS へ結果を保存します...")

    S3_ARTIFACTS.mkdir(parents=True, exist_ok=True)
    output_path = S3_ARTIFACTS / "ks_unstable_features.json"

    execution_time = time.time() - start_time

    results_json = {
        "script": "2_A_ks_stability_filter.py",
        "timestamp": datetime.now().isoformat(),
        "execution_time_seconds": round(execution_time, 2),
        "parameters": {
            "ks_p_value_threshold": KS_P_VALUE_THRESHOLD,
            "ks_sample_size": KS_SAMPLE_SIZE,
            "past_block": "2021-08 to 2022-07",
            "present_block": "2024-08 to 2025-07",
            "lf_timeframes": LF_ALL_TIMEFRAMES,
            "hf_timeframes": HF_TIMEFRAMES,
        },
        "summary": {
            "lf_features_tested": len(lf_tasks),
            "hf_features_skipped": sum(
                len(v) for v in hf_feature_names_per_source.values()
            ),
            "unstable_count": len(unstable_features),
            "stable_count": stable_count,
        },
        "unstable_features": sorted(unstable_features),
        "detail": sorted(ks_results, key=lambda x: x["feature"]),
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results_json, f, indent=2, ensure_ascii=False)

    logger.info(f"  結果を保存しました: {output_path}")

    # =========================================================================
    # サマリーログ
    # =========================================================================
    logger.info("=" * 70)
    logger.info("2_A_ks_stability_filter.py 完了")
    logger.info(f"  検定LF特徴量数    : {len(lf_tasks)}")
    logger.info(f"  不安定LF特徴量数  : {len(unstable_features)}")
    logger.info(f"  安定LF特徴量数    : {stable_count}")
    logger.info(f"  HF特徴量数（スキップ）: {results_json['summary']['hf_features_skipped']}")
    logger.info(f"  実行時間          : {execution_time:.2f} 秒")
    logger.info(f"  出力先            : {S2_FEATURES_VALIDATED}")
    logger.info(f"  成果物            : {output_path}")
    logger.info("=" * 70)


# =============================================================================
# エントリーポイント
# =============================================================================

def main() -> None:
    try:
        run_ks_filter()
        logger.info("✅ 2_A_ks_stability_filter.py が正常に完了しました。")
    except Exception as e:
        logger.critical(
            f"❌ パイプライン実行中に致命的なエラーが発生しました: {e}", exc_info=True
        )
        raise


if __name__ == "__main__":
    main()
