#!/usr/bin/env python3
"""
学際的・実験的特徴量収集スクリプト - Engine 1F: Interdisciplinary Features (最適化版)
【最適化内容】rolling_map完全排除 → map_batchesベクトル化処理

Project Forge - 軍資金増大プロジェクト
最終目標: Project Chimera開発・完成のための資金調達

技術戦略: ジム・シモンズの思想的継承
- 経済学・ファンダメンタルズ・古典的テクニカル指標の完全排除
- 統計的に有意で非ランダムな微細パターン「マーケットの亡霊」の探索
- AIの頭脳による普遍的法則の読み解き

アーキテクチャ: 3クラス構成（最適化版）+ ディスクベース垂直分割
- DataEngine（30%）: Polars LazyFrame基盤
- CalculationEngine（60%）: 学際的特徴量計算核心（物理的垂直分割実装）
- OutputEngine（10%）: ストリーミング出力
"""

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

import blueprint as config
import os, sys, time, warnings, json, logging, math, tempfile, datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, Any
from dataclasses import dataclass, field

# 数値計算・データ処理
import numpy as np
import polars as pl
import numba as nb
from numba import guvectorize, float64, int64
from scipy import stats
from scipy.stats import jarque_bera, anderson, shapiro, trim_mean

# メモリ監視
import psutil

# 警告制御
warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# Polars設定最適化の直後に追加
pl.Config.set_streaming_chunk_size(100_000)
pl.Config.set_tbl_formatting("ASCII_MARKDOWN")
pl.Config.set_tbl_hide_dataframe_shape(True)
pl.enable_string_cache()

# ログ設定
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger(__name__)


def get_default_timeframes() -> List[str]:
    return [
        "tick",
        "M0.5",
        "M1",
        "M3",
        "M5",
        "M8",
        "M15",
        "M30",
        "H1",
        "H4",
        "H6",
        "H12",
        "D1",
        "W1",
        "MN",
    ]


def get_default_window_sizes() -> Dict[str, List[int]]:
    return {
        "network": [20, 30, 50, 100],
        "linguistic": [15, 25, 40, 80],
        "aesthetic": [21, 34, 55, 89],
        "musical": [12, 24, 48, 96],
        "biomechanical": [10, 20, 40, 60],
        "general": [10, 20, 50, 100],
    }


@dataclass
class ProcessingConfig:
    """処理設定 - Project Forge統合版"""

    # ▼▼ 修正前: input_path: str = str(config.S1_BASE_MULTITIMEFRAME)
    # ▼▼ 修正後: Blueprint v4 に準拠し S1_PROCESSED を入力元に変更
    input_path: str = str(config.S1_PROCESSED)
    partitioned_tick_path: str = str(config.S1_RAW_TICK_PARTITIONED)
    output_path: str = str(config.S2_FEATURES)

    # エンジン識別
    engine_id: str = "e1f"
    engine_name: str = "Engine_1F_InterdisciplinaryFeatures"

    # 並列処理（戦略的並列処理スロットリング）
    max_threads: int = 4

    # メモリ制限（64GB RAM制約）
    memory_limit_gb: float = 55.0
    memory_warning_gb: float = 50.0

    timeframes: List[str] = field(default_factory=get_default_timeframes)
    window_sizes: Dict[str, List[int]] = field(default_factory=get_default_window_sizes)

    # ▼▼ 追加: タイムフレームごとの1日あたりのバー数（half_life動的計算用）
    timeframe_bars_per_day: Dict[str, int] = field(
        default_factory=lambda: {
            "tick": 1440,
            "M0.5": 2880,
            "M1": 1440,
            "M3": 480,
            "M5": 288,
            "M8": 180,
            "M15": 96,
            "M30": 48,
            "H1": 24,
            "H4": 6,
            "H6": 4,
            "H12": 2,
            "D1": 1,
            "W1": 1,
            "MN": 1,
        }
    )

    # 処理モード
    test_mode: bool = False
    test_rows: int = 10000

    # システムハイパーパラメータとしてW_maxを定義
    w_max: int = 200

    def validate(self) -> bool:
        """設定検証"""
        output_path_obj = Path(self.output_path)
        if not output_path_obj.exists():
            output_path_obj.mkdir(parents=True, exist_ok=True)
            logger.info(f"出力ディレクトリを作成: {output_path_obj}")
        return True


class MemoryMonitor:
    """メモリ使用量監視クラス - Project Forge準拠"""

    def __init__(self, limit_gb: float = 50.0, emergency_gb: float = 55.0):
        self.limit_gb = limit_gb
        self.emergency_gb = emergency_gb
        self.process = psutil.Process()

    def get_memory_usage_gb(self) -> float:
        """現在のメモリ使用量をGB単位で取得"""
        memory_info = self.process.memory_info()
        return memory_info.rss / (1024**3)

    def check_memory_safety(self) -> Tuple[bool, str]:
        """メモリ安全性チェック - Project Forge基準"""
        current_gb = self.get_memory_usage_gb()

        if current_gb > self.emergency_gb:
            return (
                False,
                f"緊急停止: メモリ使用量 {current_gb:.2f}GB > {self.emergency_gb}GB",
            )
        elif current_gb > self.limit_gb:
            return True, f"警告: メモリ使用量 {current_gb:.2f}GB > {self.limit_gb}GB"
        else:
            return True, f"正常: メモリ使用量 {current_gb:.2f}GB"


class DataEngine:
    """
    データ基盤クラス（30%） - Project Forge統合版
    責務:
    - Parquetメタデータ事前検証
    - Polars scan_parquetによる遅延読み込み（LazyFrame生成）
    - 述語プッシュダウンによるフィルタリング最適化
    - timeframe別分割処理（述語による高速化）
    - メモリ使用量監視（50GB警告、55GB緊急停止）
    - エラー予防機能
    """

    def __init__(self, config: ProcessingConfig):
        self.config = config
        self.memory_monitor = MemoryMonitor(config.memory_limit_gb)
        self.metadata_cache: Dict[str, Any] = {}

    def validate_data_source(self) -> bool:
        """データソース検証 - XAU/USDデータ構造準拠"""
        input_path = Path(self.config.input_path)
        if not input_path.exists():
            logger.error(f"データソースが存在しません: {input_path}")
            return False

        # timeframeディレクトリの確認
        timeframe_dirs = [
            d
            for d in input_path.iterdir()
            if d.is_dir() and d.name.startswith("timeframe=")
        ]

        if not timeframe_dirs:
            logger.error("timeframeディレクトリが見つかりません")
            return False

        logger.info(f"検出されたタイムフレーム: {len(timeframe_dirs)}個")
        return True

    def verify_parquet_metadata(self) -> Dict[str, Any]:
        """Parquetメタデータ検証 - Project Forge基準"""
        try:
            # globパターンでParquetファイルのみを指定
            parquet_pattern = f"{self.config.input_path}/**/*.parquet"

            # LazyFrameでメタデータ取得（実際の読み込みなし）
            lazy_frame = pl.scan_parquet(parquet_pattern)

            # スキーマ情報取得（警告回避）
            schema = lazy_frame.collect_schema()

            # 基本メタデータ収集
            metadata = {
                "schema": dict(schema),
                "columns": list(schema.keys()),
                "path_exists": Path(self.config.input_path).exists(),
                "estimated_memory_gb": 0.0,
            }

            # 必須カラムチェック（XAU/USDデータ構造）
            required_columns = ["timestamp", "open", "high", "low", "close", "volume"]
            missing_columns = [
                col for col in required_columns if col not in metadata["columns"]
            ]

            if missing_columns:
                raise ValueError(f"必須カラムが見つかりません: {missing_columns}")

            # Hiveパーティション構造の確認
            available_timeframes = self.config.timeframes
            logger.info("Hiveパーティション構造のためtimeframe確認をスキップ")

            metadata["available_timeframes"] = available_timeframes
            metadata["requested_timeframes"] = self.config.timeframes
            metadata["is_hive_partitioned"] = True

            self.metadata_cache = metadata
            logger.info(
                f"メタデータ検証完了: {len(metadata['columns'])}列, Hiveパーティション構造"
            )

            return metadata

        except Exception as e:
            logger.error(f"メタデータ検証エラー: {e}")
            raise

    def create_lazy_frame(self, timeframe: str) -> pl.LazyFrame:
        """指定timeframeのLazyFrame生成 - Hiveパーティション対応"""
        try:
            is_safe, message = self.memory_monitor.check_memory_safety()
            if not is_safe:
                raise MemoryError(message)

            logger.info(f"LazyFrame生成開始: timeframe={timeframe}")

            # Hiveパーティション対応：特定timeframeディレクトリを直接指定
            timeframe_path = f"{self.config.input_path}/timeframe={timeframe}/*.parquet"

            # 指定timeframeのParquetファイルのみをスキャン
            lazy_frame = pl.scan_parquet(timeframe_path)

            # timeframe列を手動で追加（Hiveパーティション復元）
            lazy_frame = lazy_frame.with_columns(
                [pl.lit(timeframe).alias("timeframe").cast(pl.Categorical)]
            )

            # スキーマを確認してから安全にキャスト処理を適用
            # まず小さなサンプルでスキーマを確認
            try:
                sample_schema = lazy_frame.limit(1).collect_schema()
                logger.info(f"検出されたスキーマ: {list(sample_schema.keys())}")

                # 基本データ型確認と最適化（必要な場合のみキャスト）
                cast_exprs = []

                # 各カラムが存在し、かつ適切な型でない場合のみキャスト
                if "timestamp" in sample_schema and sample_schema[
                    "timestamp"
                ] != pl.Datetime("ns"):
                    # 修正後
                    cast_exprs.append(pl.col("timestamp").cast(pl.Datetime("us")))
                if "open" in sample_schema and sample_schema["open"] != pl.Float64:
                    cast_exprs.append(pl.col("open").cast(pl.Float64))
                if "high" in sample_schema and sample_schema["high"] != pl.Float64:
                    cast_exprs.append(pl.col("high").cast(pl.Float64))
                if "low" in sample_schema and sample_schema["low"] != pl.Float64:
                    cast_exprs.append(pl.col("low").cast(pl.Float64))
                if "close" in sample_schema and sample_schema["close"] != pl.Float64:
                    cast_exprs.append(pl.col("close").cast(pl.Float64))
                if "volume" in sample_schema and sample_schema["volume"] != pl.Int64:
                    cast_exprs.append(pl.col("volume").cast(pl.Int64))

                # 必要なキャストのみ適用
                if cast_exprs:
                    logger.info(f"型変換を適用: {len(cast_exprs)}個のカラム")
                    lazy_frame = lazy_frame.with_columns(cast_exprs)
                else:
                    logger.info("型変換不要: 全カラムが適切な型です")

            except Exception as schema_error:
                logger.warning(
                    f"スキーマ確認エラー、キャストをスキップ: {schema_error}"
                )
                # キャストエラーの場合は処理を続行（データがすでに適切な型の可能性）

            # タイムスタンプソート
            lazy_frame = lazy_frame.sort("timestamp")

            logger.info(f"LazyFrame生成完了: timeframe={timeframe}")
            return lazy_frame

        except Exception as e:
            logger.error(f"LazyFrame生成エラー (timeframe={timeframe}): {e}")
            raise

    def get_data_summary(self, lazy_frame: pl.LazyFrame) -> Dict[str, Any]:
        """データサマリー情報取得（軽量）"""
        try:
            # 最小限のcollectでサマリー取得
            summary = lazy_frame.select(
                [
                    pl.len().alias("total_rows"),
                    pl.col("timestamp").min().alias("start_time"),
                    pl.col("timestamp").max().alias("end_time"),
                    pl.col("close").mean().alias("avg_price"),
                    pl.col("volume").sum().alias("total_volume"),
                ]
            ).collect()

            return {
                "total_rows": summary["total_rows"][0],
                "start_time": summary["start_time"][0],
                "end_time": summary["end_time"][0],
                "avg_price": summary["avg_price"][0],
                "total_volume": summary["total_volume"][0],
            }
        except Exception as e:
            logger.error(f"データサマリー取得エラー: {e}")
            return {"error": str(e)}


# ===============================
# Numba UDF定義（クラス外必須）- 最適化版
# ===============================


# ▼▼ 修正後: sample_weight計算用のNumba UDFを追加
@nb.njit(fastmath=True, cache=True)
def calculate_atr_numba(
    high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int = 13
) -> np.ndarray:
    n = len(close)
    atr = np.full(n, np.nan, dtype=np.float64)
    tr = np.zeros(n, dtype=np.float64)
    if n == 0:
        return atr

    tr[0] = high[0] - low[0]
    for i in range(1, n):
        hl = high[i] - low[i]
        hc = abs(high[i] - close[i - 1])
        lc = abs(low[i] - close[i - 1])
        tr[i] = max(hl, hc, lc)

    if n > period:
        atr[period - 1] = np.mean(tr[:period])
        for i in range(period, n):
            atr[i] = (atr[i - 1] * (period - 1) + tr[i]) / period
    return atr


@nb.njit(fastmath=True, cache=True)
def calculate_sample_weight_udf(
    high: np.ndarray, low: np.ndarray, close: np.ndarray
) -> np.ndarray:
    n = len(close)
    weights = np.ones(n, dtype=np.float64)
    atr_13 = calculate_atr_numba(high, low, close, 13)

    # NaNを避けるための安全なATR
    safe_atr = np.where(np.isnan(atr_13) | (atr_13 == 0), 1e-10, atr_13)
    normalized_range = (high - low) / safe_atr

    # ローリングZスコア計算（ルックバック期間=1440）
    window = 1440
    for i in range(window, n):
        window_slice = normalized_range[i - window : i]
        mean_val = np.mean(window_slice)

        # ▼▼ 修正前: std_val = np.std(window_slice, ddof=1)
        # ▼▼ 修正後: ベッセル補正による不偏標準偏差の手動計算
        n_slice = len(window_slice)
        std_val = (
            np.std(window_slice) * np.sqrt(n_slice / (n_slice - 1.0))
            if n_slice > 1
            else 0.0
        )

        if std_val > 1e-10:
            z_score = abs(normalized_range[i] - mean_val) / std_val
            if z_score >= 4.0:
                weights[i] = 6.0
            elif z_score >= 3.0:
                weights[i] = 4.0
            elif z_score >= 2.0:
                weights[i] = 2.0
    return weights


# =============================================================================
# ネットワーク科学: 価格ネットワークの密度とクラスタリング (並列化版)
# =============================================================================


# ▼▼ 修正前: @nb.njit(fastmath=True, cache=True, parallel=True)
# ▼▼ 修正後: parallel=True を削除
@nb.njit(fastmath=True, cache=True)
def rolling_network_density_udf(prices: np.ndarray, window_size: int) -> np.ndarray:
    n = len(prices)
    results = np.full(n, np.nan, dtype=np.float64)

    # ▼▼ 修正前: for i in nb.prange(window_size - 1, n):
    # ▼▼ 修正後: nb.prange を range に変更
    for i in range(window_size - 1, n):
        window_prices = prices[i - window_size + 1 : i + 1]

        if len(window_prices) < 10:
            continue

        finite_prices = window_prices[np.isfinite(window_prices)]
        if len(finite_prices) < 10:
            continue

        window_n = len(finite_prices)
        # ▼▼ 修正前: threshold = np.std(finite_prices, ddof=1) * 0.5
        # ▼▼ 修正後: Numbaエラー回避のため、手動のベッセル補正に置換
        threshold = (
            np.std(finite_prices) * np.sqrt(window_n / (window_n - 1.0))
            if window_n > 1
            else 0.0
        ) * 0.5

        edge_count = 0.0  # float化
        max_possible_edges = float(window_n * (window_n - 1) / 2.0)

        for j in range(window_n - 1):
            for k in range(j + 1, window_n):
                price_diff = abs(finite_prices[j] - finite_prices[k])
                if price_diff <= threshold:
                    edge_count += 1

        if max_possible_edges > 0:
            # ▼▼ 修正前: density = edge_count / max_possible_edges
            # ▼▼ 修正後: ゼロ除算防止の徹底
            density = edge_count / (max_possible_edges + 1e-10)
        else:
            density = 0.0

        results[i] = density

    return results


# ▼▼ 修正前: @nb.njit(fastmath=True, cache=True, parallel=True)
# ▼▼ 修正後: parallel=True を削除し、バッファを事前確保
@nb.njit(fastmath=True, cache=True)
def rolling_network_clustering_udf(prices: np.ndarray, window_size: int) -> np.ndarray:
    n = len(prices)
    results = np.full(n, np.nan, dtype=np.float64)

    # 事前確保バッファ
    adj_buffer = np.zeros((window_size, window_size), dtype=nb.boolean)
    neighbors_buffer = np.zeros(window_size, dtype=nb.int64)

    # ▼▼ 修正前: for i in nb.prange(window_size - 1, n):
    # ▼▼ 修正後: nb.prange を range に変更
    for i in range(window_size - 1, n):
        window_prices = prices[i - window_size + 1 : i + 1]

        if len(window_prices) < 15:
            continue

        finite_prices = window_prices[np.isfinite(window_prices)]
        if len(finite_prices) < 15:
            continue

        window_n = len(finite_prices)
        # ▼▼ 修正前: threshold = np.std(finite_prices, ddof=1) * 0.5
        # ▼▼ 修正後: 手動のベッセル補正に置換
        threshold = (
            np.std(finite_prices) * np.sqrt(window_n / (window_n - 1.0))
            if window_n > 1
            else 0.0
        ) * 0.5

        # ループ外で確保したバッファを初期化して再利用
        adj_buffer[:window_n, :window_n] = False

        for j in range(window_n):
            for k in range(window_n):
                if j != k:
                    price_diff = abs(finite_prices[j] - finite_prices[k])
                    if price_diff <= threshold:
                        adj_buffer[j, k] = True

        total_clustering = 0.0
        valid_nodes = 0

        for j in range(window_n):
            # ▼▼ 修正前: neighbors = []
            # ▼▼ 修正後: 事前確保バッファとインデックスカウンタを使用
            neighbor_count = 0
            for k in range(window_n):
                if adj_buffer[j, k]:
                    neighbors_buffer[neighbor_count] = k
                    neighbor_count += 1

            k_len = neighbor_count
            if k_len < 2:
                continue

            neighbor_connections = 0
            for idx1 in range(k_len):
                for idx2 in range(idx1 + 1, k_len):
                    n1, n2 = neighbors_buffer[idx1], neighbors_buffer[idx2]
                    if adj_buffer[n1, n2]:
                        neighbor_connections += 1

            # 修正後
            max_connections = float(k_len * (k_len - 1) / 2.0)
            if max_connections > 0:
                # ▼▼ 修正前: clustering_j = neighbor_connections / max_connections
                # ▼▼ 修正後: ゼロ除算防止の徹底
                clustering_j = neighbor_connections / (max_connections + 1e-10)
                total_clustering += clustering_j
                valid_nodes += 1

        if valid_nodes > 0:
            # ▼▼ 修正前: clustering_coeff = total_clustering / valid_nodes
            # ▼▼ 修正後: ゼロ除算防止の徹底
            clustering_coeff = total_clustering / (float(valid_nodes) + 1e-10)
        else:
            clustering_coeff = 0.0

        results[i] = clustering_coeff

    return results


# =============================================================================
# 言語学: 語彙多様性・言語的複雑性・意味的流れ (並列化版)
# =============================================================================


# ▼▼ 修正前: @nb.njit(fastmath=True, cache=True, parallel=True)
# ▼▼ 修正後: parallel=True を削除し、バッファを事前確保
@nb.njit(fastmath=True, cache=True)
def rolling_vocabulary_diversity_udf(
    prices: np.ndarray, window_size: int
) -> np.ndarray:
    n = len(prices)
    results = np.full(n, np.nan, dtype=np.float64)

    # 事前確保バッファ (n_bins=10)
    vocab_buffer = np.zeros(10, dtype=nb.boolean)

    # ▼▼ 修正前: for i in nb.prange(window_size - 1, n):
    # ▼▼ 修正後: nb.prange を range に変更
    for i in range(window_size - 1, n):
        window_prices = prices[i - window_size + 1 : i + 1]

        if len(window_prices) < 10:
            continue

        finite_prices = window_prices[np.isfinite(window_prices)]
        if len(finite_prices) < 10:
            continue

        # ▼▼ 修正前: std_price = np.std(finite_prices, ddof=1)
        # ▼▼ 修正後: 手動のベッセル補正に置換
        n_fp = len(finite_prices)
        std_price = (
            np.std(finite_prices) * np.sqrt(n_fp / (n_fp - 1.0)) if n_fp > 1 else 0.0
        )
        mean_price = np.mean(finite_prices)

        if std_price == 0.0:
            results[i] = 0.0
            continue

        n_bins = 10
        price_min = mean_price - 2 * std_price
        price_max = mean_price + 2 * std_price
        bin_width = (price_max - price_min) / float(n_bins)

        # ▼▼ 修正前: used_vocabularies = set()
        # ▼▼ 修正後: ループ外で確保したバッファを初期化して再利用
        vocab_buffer[:] = False
        total_tokens = 0
        unique_count = 0

        for price in finite_prices:
            if bin_width > 0.0:
                bin_idx = int((price - price_min) / bin_width)
                bin_idx = max(0, min(bin_idx, n_bins - 1))
            else:
                bin_idx = 0

            if not vocab_buffer[bin_idx]:
                vocab_buffer[bin_idx] = True
                unique_count += 1
            total_tokens += 1

        if total_tokens > 0:
            vocabulary_diversity = float(unique_count) / (float(total_tokens) + 1e-10)
        else:
            vocabulary_diversity = 0.0

        results[i] = vocabulary_diversity

    return results


# ▼▼ 修正前: @nb.njit(fastmath=True, cache=True, parallel=True)
# ▼▼ 修正後: parallel=True を削除し、バッファを事前確保
@nb.njit(fastmath=True, cache=True)
def rolling_linguistic_complexity_udf(
    prices: np.ndarray, window_size: int
) -> np.ndarray:
    n = len(prices)
    results = np.full(n, np.nan, dtype=np.float64)

    # 事前確保バッファ
    syntax_buffer = np.zeros(window_size, dtype=np.int32)
    bigram_buffer = np.zeros(9, dtype=np.float64)
    trigram_buffer = np.zeros(27, dtype=np.float64)

    # ▼▼ 修正前: for i in nb.prange(window_size - 1, n):
    # ▼▼ 修正後: nb.prange を range に変更
    for i in range(window_size - 1, n):
        window_prices = prices[i - window_size + 1 : i + 1]
        finite_prices = window_prices[np.isfinite(window_prices)]
        if len(finite_prices) < 20:
            continue

        price_changes = np.diff(finite_prices)
        # ▼▼ 修正前: threshold = np.std(price_changes, ddof=1) * 0.1
        # ▼▼ 修正後: 手動のベッセル補正に置換
        n_pc = len(price_changes)
        threshold = (
            np.std(price_changes) * np.sqrt(n_pc / (n_pc - 1.0)) if n_pc > 1 else 0.0
        ) * 0.1

        # 事前確保バッファを使用しスライスで長さを管理
        seq_len = len(price_changes)
        syntax_sequence = syntax_buffer[:seq_len]

        for j in range(seq_len):
            if price_changes[j] > threshold:
                syntax_sequence[j] = 2
            elif price_changes[j] < -threshold:
                syntax_sequence[j] = 0
            else:
                syntax_sequence[j] = 1

        if seq_len < 3:
            continue

        # ▼▼ 修正前: bigram_counts = np.zeros(9, dtype=np.float64)
        # ▼▼ 修正後: バッファを明示的に初期化
        bigram_buffer[:] = 0.0
        total_bigrams = 0.0
        for j in range(seq_len - 1):
            idx = syntax_sequence[j] * 3 + syntax_sequence[j + 1]
            bigram_buffer[idx] += 1.0
            total_bigrams += 1.0

        entropy_bi = 0.0
        for j in range(9):
            if bigram_buffer[j] > 0.0:
                p = bigram_buffer[j] / total_bigrams
                entropy_bi -= p * np.log2(p)

        # ▼▼ 修正前: trigram_counts = np.zeros(27, dtype=np.float64)
        # ▼▼ 修正後: バッファを明示的に初期化
        trigram_buffer[:] = 0.0
        total_trigrams = 0.0
        for j in range(seq_len - 2):
            idx = (
                syntax_sequence[j] * 9
                + syntax_sequence[j + 1] * 3
                + syntax_sequence[j + 2]
            )
            trigram_buffer[idx] += 1.0
            total_trigrams += 1.0

        entropy_tri = 0.0
        for j in range(27):
            if trigram_buffer[j] > 0.0:
                p = trigram_buffer[j] / total_trigrams
                entropy_tri -= p * np.log2(p)

        # 理論上の最大エントロピーで正規化 (0.0 〜 1.0)
        max_entropy_bi = np.log2(min(9.0, total_bigrams))
        max_entropy_tri = np.log2(min(27.0, total_trigrams))

        norm_bi = entropy_bi / max_entropy_bi if max_entropy_bi > 1e-10 else 0.0
        norm_tri = entropy_tri / max_entropy_tri if max_entropy_tri > 1e-10 else 0.0

        results[i] = (norm_bi + norm_tri) / 2.0

    return results


# ▼▼ 修正前: @nb.njit(fastmath=True, cache=True, parallel=True)
# ▼▼ 修正後: parallel=True を削除し、バッファを事前確保
@nb.njit(fastmath=True, cache=True)
def rolling_semantic_flow_udf(prices: np.ndarray, window_size: int) -> np.ndarray:
    n = len(prices)
    results = np.full(n, np.nan, dtype=np.float64)

    # 事前確保バッファ (最大でwindow_size個のベクトル)
    vec_buffer = np.zeros((window_size, 2), dtype=np.float64)

    # ▼▼ 修正前: for i in nb.prange(window_size - 1, n):
    # ▼▼ 修正後: nb.prange を range に変更
    for i in range(window_size - 1, n):
        window_prices = prices[i - window_size + 1 : i + 1]

        if len(window_prices) < 15:
            continue

        finite_prices = window_prices[np.isfinite(window_prices)]
        if len(finite_prices) < 15:
            continue

        window_n = len(finite_prices)
        window_size_local = min(5, window_n // 3)

        # ▼▼ 修正前: semantic_vectors = []
        # ▼▼ 修正後: インデックスカウンタと事前バッファで管理
        vec_count = 0

        for j in range(window_size_local, window_n - window_size_local):
            neighborhood = finite_prices[
                j - window_size_local : j + window_size_local + 1
            ]
            center_price = finite_prices[j]
            relative_positions = neighborhood - center_price

            if len(relative_positions) > 1:
                mean_rel = np.mean(relative_positions)
                # ▼▼ 修正前: std_rel = np.std(relative_positions, ddof=1)
                # ▼▼ 修正後: 手動のベッセル補正に置換
                n_rel = len(relative_positions)
                std_rel = (
                    np.std(relative_positions) * np.sqrt(n_rel / (n_rel - 1.0))
                    if n_rel > 1
                    else 0.0
                )
                vec_buffer[vec_count, 0] = mean_rel
                vec_buffer[vec_count, 1] = std_rel
                vec_count += 1
        if vec_count < 2:
            continue

        flow_continuity = 0.0
        valid_pairs = 0

        # ▼▼ 修正前: for j in range(len(semantic_vectors) - 1):
        # ▼▼ 修正後: 有効なインデックス長(vec_count)を使用しバッファにアクセス
        for j in range(vec_count - 1):
            v1_0 = vec_buffer[j, 0]
            v1_1 = vec_buffer[j, 1]
            v2_0 = vec_buffer[j + 1, 0]
            v2_1 = vec_buffer[j + 1, 1]

            norm1 = np.sqrt(v1_0**2 + v1_1**2)
            norm2 = np.sqrt(v2_0**2 + v2_1**2)

            if norm1 > 1e-10 and norm2 > 1e-10:
                cosine_sim = (v1_0 * v2_0 + v1_1 * v2_1) / (norm1 * norm2 + 1e-10)
                flow_continuity += cosine_sim
                valid_pairs += 1
        if valid_pairs > 0:
            # ▼▼ 修正前: semantic_flow = flow_continuity / valid_pairs
            # ▼▼ 修正後: ゼロ除算防止の徹底
            semantic_flow = flow_continuity / (float(valid_pairs) + 1e-10)
            semantic_flow = (semantic_flow + 1.0) / 2.0
        else:
            semantic_flow = 0.0

        results[i] = semantic_flow

    return results


# =============================================================================
# 美学: 黄金比・対称性・美的バランス (並列化版)
# =============================================================================


# ▼▼ 修正前: @nb.njit(fastmath=True, cache=True, parallel=True)
# ▼▼ 修正後: parallel=True を削除し、バッファを事前確保
@nb.njit(fastmath=True, cache=True)
def rolling_golden_ratio_adherence_udf(
    prices: np.ndarray, window_size: int
) -> np.ndarray:
    n = len(prices)
    results = np.full(n, np.nan, dtype=np.float64)
    golden_ratio = (1.0 + np.sqrt(5.0)) / 2.0

    # 事前確保バッファ
    adherence_buffer = np.zeros(window_size, dtype=np.float64)

    # ▼▼ 修正前: for i in nb.prange(window_size - 1, n):
    # ▼▼ 修正後: nb.prange を range に変更
    for i in range(window_size - 1, n):
        window_prices = prices[i - window_size + 1 : i + 1]

        if len(window_prices) < 10:
            continue

        finite_prices = window_prices[np.isfinite(window_prices)]
        if len(finite_prices) < 10:
            continue

        local_window = min(8, len(finite_prices) // 2)

        # ▼▼ 修正前: adherence_scores = []
        # ▼▼ 修正後: 事前確保バッファとインデックスカウンタを使用
        adherence_count = 0

        for j in range(local_window, len(finite_prices) - local_window):
            local_subwindow = finite_prices[j - local_window : j + local_window + 1]
            local_high = np.max(local_subwindow)
            local_low = np.min(local_subwindow)

            if local_low > 1e-10:
                ratio = local_high / local_low
                deviation = abs(ratio - golden_ratio) / golden_ratio
                deviation = min(deviation, 10.0)
                adherence = 1.0 / (1.0 + deviation)

                adherence_buffer[adherence_count] = adherence
                adherence_count += 1

        if adherence_count > 0:
            results[i] = np.mean(adherence_buffer[:adherence_count])

    return results


# ▼▼ 修正前: @nb.njit(fastmath=True, cache=True, parallel=True)
# ▼▼ 修正後: parallel=True を削除し、バッファを事前確保
@nb.njit(fastmath=True, cache=True)
def rolling_symmetry_measure_udf(prices: np.ndarray, window_size: int) -> np.ndarray:
    n = len(prices)
    results = np.full(n, np.nan, dtype=np.float64)

    # 事前確保バッファ
    left_norm_buffer = np.zeros(window_size, dtype=np.float64)
    right_norm_buffer = np.zeros(window_size, dtype=np.float64)

    # ▼▼ 修正前: for i in nb.prange(window_size - 1, n):
    # ▼▼ 修正後: nb.prange を range に変更
    for i in range(window_size - 1, n):
        window_prices = prices[i - window_size + 1 : i + 1]
        if len(window_prices) < 20:
            continue

        finite_prices = window_prices[np.isfinite(window_prices)]
        if len(finite_prices) < 20:
            continue

        window_n = len(finite_prices)
        center = window_n // 2

        left_half = finite_prices[:center]
        right_half = (
            finite_prices[center + 1 :] if window_n % 2 == 1 else finite_prices[center:]
        )

        right_half_reversed = right_half[::-1]

        min_len = min(len(left_half), len(right_half_reversed))
        if min_len < 5:
            continue

        left_normalized = left_half[-min_len:]
        right_normalized = right_half_reversed[:min_len]

        # ▼▼ 修正前: 内部関数 def normalize_series(series): ... (遅延・再確保の元)
        # ▼▼ 修正後: 内部関数を廃止し、事前バッファのスライスに直接書き込み
        left_norm_view = left_norm_buffer[:min_len]
        right_norm_view = right_norm_buffer[:min_len]

        mean_left_raw = np.mean(left_normalized)
        # ▼▼ 修正前: std_left_raw = np.std(left_normalized, ddof=1)
        # ▼▼ 修正後: 手動のベッセル補正に置換
        n_left = len(left_normalized)
        std_left_raw = (
            np.std(left_normalized) * np.sqrt(n_left / (n_left - 1.0))
            if n_left > 1
            else 0.0
        )
        if std_left_raw > 1e-10:
            left_norm_view[:] = (left_normalized - mean_left_raw) / std_left_raw
        else:
            left_norm_view[:] = left_normalized - mean_left_raw

        mean_right_raw = np.mean(right_normalized)
        # ▼▼ 修正前: std_right_raw = np.std(right_normalized, ddof=1)
        # ▼▼ 修正後: 手動のベッセル補正に置換
        n_right = len(right_normalized)
        std_right_raw = (
            np.std(right_normalized) * np.sqrt(n_right / (n_right - 1.0))
            if n_right > 1
            else 0.0
        )
        if std_right_raw > 1e-10:
            right_norm_view[:] = (right_normalized - mean_right_raw) / std_right_raw
        else:
            right_norm_view[:] = right_normalized - mean_right_raw

        if len(left_norm_view) < 2:
            continue

        mean_left = np.mean(left_norm_view)
        mean_right = np.mean(right_norm_view)

        numerator = np.sum(
            (left_norm_view - mean_left) * (right_norm_view - mean_right)
        )
        denom_left = np.sum((left_norm_view - mean_left) ** 2)
        denom_right = np.sum((right_norm_view - mean_right) ** 2)

        if denom_left > 1e-10 and denom_right > 1e-10:
            # ▼▼ 修正: ゼロ除算防止の徹底
            correlation = numerator / (np.sqrt(denom_left * denom_right) + 1e-10)
            symmetry = (correlation + 1.0) / 2.0
        else:
            symmetry = 0.0

        results[i] = symmetry

    return results


# ▼▼ 修正前: @nb.njit(fastmath=True, cache=True, parallel=True)
# ▼▼ 修正後: parallel=True を削除し、バッファを事前確保
@nb.njit(fastmath=True, cache=True)
def rolling_aesthetic_balance_udf(prices: np.ndarray, window_size: int) -> np.ndarray:
    n = len(prices)
    results = np.full(n, np.nan, dtype=np.float64)

    # 事前確保バッファ
    grad_buffer = np.zeros(window_size, dtype=np.float64)

    # ▼▼ 修正前: for i in nb.prange(window_size - 1, n):
    # ▼▼ 修正後: nb.prange を range に変更
    for i in range(window_size - 1, n):
        window_prices = prices[i - window_size + 1 : i + 1]

        if len(window_prices) < 15:
            continue

        finite_prices = window_prices[np.isfinite(window_prices)]
        if len(finite_prices) < 15:
            continue

        # ▼▼ 修正前: gradients = np.diff(finite_prices)
        # ▼▼ 修正後: np.diff の動的確保を避け、手動ループでバッファに直接書き込み
        grad_len = len(finite_prices) - 1
        for j in range(grad_len):
            grad_buffer[j] = abs(finite_prices[j + 1] - finite_prices[j])

        if grad_len < 5:
            continue

        abs_gradients = grad_buffer[:grad_len]
        mean_grad = np.mean(abs_gradients)
        # ▼▼ 修正前: std_grad = np.std(abs_gradients, ddof=1)
        # ▼▼ 修正後: 手動のベッセル補正に置換
        n_grad = len(abs_gradients)
        std_grad = (
            np.std(abs_gradients) * np.sqrt(n_grad / (n_grad - 1.0))
            if n_grad > 1
            else 0.0
        )

        if std_grad <= 1e-10:
            results[i] = 1.0
            continue

        gentle_threshold = mean_grad - 0.5 * std_grad
        moderate_threshold = mean_grad + 0.5 * std_grad
        intense_threshold = mean_grad + 1.5 * std_grad

        gentle_count = 0
        moderate_count = 0
        intense_count = 0

        for grad in abs_gradients:
            if grad <= gentle_threshold:
                gentle_count += 1
            elif grad <= moderate_threshold:
                moderate_count += 1
            elif grad <= intense_threshold:
                intense_count += 1

        total_counted = gentle_count + moderate_count + intense_count
        if total_counted == 0:
            results[i] = 0.0
            continue

        ideal_gentle = 0.6
        ideal_moderate = 0.3
        ideal_intense = 0.1

        # ▼▼ 修正: float化とゼロ除算防止の徹底
        actual_gentle = float(gentle_count) / (float(total_counted) + 1e-10)
        actual_moderate = float(moderate_count) / (float(total_counted) + 1e-10)
        actual_intense = float(intense_count) / (float(total_counted) + 1e-10)

        balance_deviation = (
            abs(actual_gentle - ideal_gentle)
            + abs(actual_moderate - ideal_moderate)
            + abs(actual_intense - ideal_intense)
        ) / 2.0

        aesthetic_balance = 1.0 - balance_deviation
        results[i] = max(0.0, aesthetic_balance)

    return results


# =============================================================================
# 音楽理論: 調性・リズムパターン・和声・音楽的緊張度 (並列化版)
# =============================================================================


# ▼▼ 修正前: @nb.njit(fastmath=True, cache=True, parallel=True)
# ▼▼ 修正後: parallel=True を削除し、バッファを事前確保
@nb.njit(fastmath=True, cache=True)
def rolling_tonality_udf(prices: np.ndarray, window_size: int) -> np.ndarray:
    n = len(prices)
    results = np.full(n, np.nan, dtype=np.float64)

    # 事前確保バッファ
    diff_buffer = np.zeros(window_size, dtype=np.float64)
    scale_degrees_buffer = np.zeros(12, dtype=np.float64)

    major_pattern = np.array(
        [1.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0], dtype=np.float64
    )
    minor_pattern = np.array(
        [1.0, 0.0, 1.0, 1.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0, 0.0], dtype=np.float64
    )

    # ▼▼ 修正前: for i in nb.prange(window_size - 1, n):
    # ▼▼ 修正後: nb.prange を range に変更
    for i in range(window_size - 1, n):
        window_prices = prices[i - window_size + 1 : i + 1]

        if len(window_prices) < 12:
            continue

        finite_prices = window_prices[np.isfinite(window_prices)]
        if len(finite_prices) < 12:
            continue

        # ▼▼ 修正前: price_changes = np.diff(finite_prices)
        # ▼▼ 修正後: バッファを使用して動的メモリ確保を排除
        diff_len = len(finite_prices) - 1
        for j in range(diff_len):
            diff_buffer[j] = finite_prices[j + 1] - finite_prices[j]

        price_changes = diff_buffer[:diff_len]

        if diff_len < 5:
            continue

        # ▼▼ 修正前: std_change = np.std(price_changes, ddof=1)
        # ▼▼ 修正後: 手動のベッセル補正に置換
        n_pc = len(price_changes)
        std_change = (
            np.std(price_changes) * np.sqrt(n_pc / (n_pc - 1.0)) if n_pc > 1 else 0.0
        )

        if std_change <= 1e-10:
            results[i] = 0.5
            continue

        normalized_changes = price_changes / (std_change + 1e-10)

        # ▼▼ 修正前: scale_degrees = np.zeros(12, dtype=np.float64)
        # ▼▼ 修正後: 事前確保したバッファを明示的にゼロリセット
        scale_degrees_buffer[:] = 0.0

        for change in normalized_changes:
            degree_idx = int((change + 3.0) / 6.0 * 11.0)
            degree_idx = max(0, min(degree_idx, 11))
            scale_degrees_buffer[degree_idx] += 1.0

        sum_degrees = np.sum(scale_degrees_buffer)
        if sum_degrees > 0.0:
            scale_distribution = scale_degrees_buffer / (sum_degrees + 1e-10)

            major_similarity = np.sum(scale_distribution * major_pattern)
            minor_similarity = np.sum(scale_distribution * minor_pattern)

            total_similarity = major_similarity + minor_similarity
            if total_similarity > 0.0:
                # ▼▼ 修正: ゼロ除算防止の徹底
                tonality_score = major_similarity / (total_similarity + 1e-10)
            else:
                tonality_score = 0.5
        else:
            tonality_score = 0.5

        results[i] = tonality_score

    return results


# ▼▼ 修正前: @nb.njit(fastmath=True, cache=True, parallel=True)
# ▼▼ 修正後: parallel=True を削除し、バッファを事前確保
@nb.njit(fastmath=True, cache=True)
def rolling_rhythm_pattern_udf(prices: np.ndarray, window_size: int) -> np.ndarray:
    n = len(prices)
    results = np.full(n, np.nan, dtype=np.float64)

    # 事前確保バッファ
    diff_buffer = np.zeros(window_size, dtype=np.float64)

    # ▼▼ 修正前: for i in nb.prange(window_size - 1, n):
    # ▼▼ 修正後: nb.prange を range に変更
    for i in range(window_size - 1, n):
        window_prices = prices[i - window_size + 1 : i + 1]
        finite_prices = window_prices[np.isfinite(window_prices)]
        if len(finite_prices) < 20:
            continue

        # ▼▼ 修正前: price_changes = np.diff(finite_prices)
        # ▼▼ 修正後: バッファを使用して動的メモリ確保を排除
        N_changes = len(finite_prices) - 1
        for j in range(N_changes):
            diff_buffer[j] = finite_prices[j + 1] - finite_prices[j]

        price_changes = diff_buffer[:N_changes]

        if N_changes < 5:
            continue

        mean_change = np.mean(price_changes)
        # ▼▼ 修正前: var_change = np.var(price_changes, ddof=1)
        # ▼▼ 修正後: 時系列解析の標準ACF推定量に合わせ、標本分散(ddof=0)に統一
        var_change = np.var(price_changes)

        if var_change <= 1e-10:
            results[i] = 0.0
            continue

        max_lag = N_changes // 4
        max_acf = 0.0

        for lag in range(1, max_lag + 1):
            cov = 0.0
            for j in range(N_changes - lag):
                cov += (price_changes[j] - mean_change) * (
                    price_changes[j + lag] - mean_change
                )

            # ▼▼ 修正前: cov = cov / float(N_changes - lag)
            # ▼▼ 修正後: 分散と同じく N_changes で割り、自己相関行列の半正定値性を保証
            cov = cov / float(N_changes)
            acf = abs(cov / var_change)

            if acf > max_acf:
                max_acf = acf

        results[i] = max_acf

    return results


# ▼▼ 修正前: @nb.njit(fastmath=True, cache=True, parallel=True)
# ▼▼ 修正後: parallel=True を削除し、バッファを事前確保
@nb.njit(fastmath=True, cache=True)
def rolling_harmony_udf(prices: np.ndarray, window_size: int) -> np.ndarray:
    n = len(prices)
    results = np.full(n, np.nan, dtype=np.float64)

    # 修正後
    # 事前確保バッファ群
    short_ma_buffer = np.zeros(window_size, dtype=np.float64)
    medium_ma_buffer = np.zeros(window_size, dtype=np.float64)
    long_ma_buffer = np.zeros(window_size, dtype=np.float64)
    harmony_buffer = np.zeros(window_size, dtype=np.float64)
    short_trend_buf = np.zeros(window_size, dtype=np.float64)
    medium_trend_buf = np.zeros(window_size, dtype=np.float64)
    long_trend_buf = np.zeros(window_size, dtype=np.float64)

    # ▼▼ 修正前: for i in nb.prange(window_size - 1, n):
    # ▼▼ 修正後: nb.prange を range に変更
    for i in range(window_size - 1, n):
        window_prices = prices[i - window_size + 1 : i + 1]

        if len(window_prices) < 30:
            continue

        finite_prices = window_prices[np.isfinite(window_prices)]
        if len(finite_prices) < 30:
            continue

        window_n = len(finite_prices)

        short_window = max(3, window_n // 15)
        medium_window = max(5, window_n // 10)
        long_window = max(8, window_n // 6)

        if long_window >= window_n:
            continue

        # ▼▼ 修正前: short_ma = np.zeros(window_n - short_window + 1, dtype=np.float64) (等)
        # ▼▼ 修正後: バッファのスライス再利用と明示的ゼロリセット
        short_len = window_n - short_window + 1
        medium_len = window_n - medium_window + 1
        long_len = window_n - long_window + 1

        short_ma = short_ma_buffer[:short_len]
        medium_ma = medium_ma_buffer[:medium_len]
        long_ma = long_ma_buffer[:long_len]

        for j in range(short_len):
            short_ma[j] = np.mean(finite_prices[j : j + short_window])

        for j in range(len(medium_ma)):
            medium_ma[j] = np.mean(finite_prices[j : j + medium_window])

        for j in range(len(long_ma)):
            long_ma[j] = np.mean(finite_prices[j : j + long_window])

        min_len = min(len(short_ma), len(medium_ma), len(long_ma))
        if min_len < 5:
            continue

        # 修正後
        trend_len = min_len - 1
        for _j in range(trend_len):
            short_trend_buf[_j] = short_ma[-min_len + _j + 1] - short_ma[-min_len + _j]
            medium_trend_buf[_j] = (
                medium_ma[-min_len + _j + 1] - medium_ma[-min_len + _j]
            )
            long_trend_buf[_j] = long_ma[-min_len + _j + 1] - long_ma[-min_len + _j]
        short_trend = short_trend_buf[:trend_len]
        medium_trend = medium_trend_buf[:trend_len]
        long_trend = long_trend_buf[:trend_len]

        # ▼▼ 修正前: harmony_scores = []
        # ▼▼ 修正後: 事前バッファとインデックスカウンタで管理
        harmony_count = 0

        # 修正後
        for j in range(trend_len):
            s_sign = np.sign(short_trend[j])
            m_sign = np.sign(medium_trend[j])
            l_sign = np.sign(long_trend[j])

            if s_sign == m_sign and m_sign == l_sign and s_sign != 0.0:
                harmony_buffer[harmony_count] = 1.0
            elif (s_sign == m_sign) or (m_sign == l_sign) or (s_sign == l_sign):
                harmony_buffer[harmony_count] = 0.5
            else:
                harmony_buffer[harmony_count] = 0.0
            harmony_count += 1

        if harmony_count > 0:
            results[i] = np.mean(harmony_buffer[:harmony_count])
        else:
            results[i] = 0.0

    return results


# ▼▼ 修正前: @nb.njit(fastmath=True, cache=True, parallel=True)
# ▼▼ 修正後: parallel=True を削除し、バッファを事前確保
@nb.njit(fastmath=True, cache=True)
def rolling_musical_tension_udf(prices: np.ndarray, window_size: int) -> np.ndarray:
    n = len(prices)
    results = np.full(n, np.nan, dtype=np.float64)

    # 事前確保バッファ
    diff_buffer = np.zeros(window_size, dtype=np.float64)
    tension_buffer = np.zeros(window_size, dtype=np.float64)

    # ▼▼ 修正前: for i in nb.prange(window_size - 1, n):
    # ▼▼ 修正後: nb.prange を range に変更
    for i in range(window_size - 1, n):
        window_prices = prices[i - window_size + 1 : i + 1]

        if len(window_prices) < 15:
            continue

        finite_prices = window_prices[np.isfinite(window_prices)]
        if len(finite_prices) < 15:
            continue

        # ▼▼ 修正前: price_changes = np.diff(finite_prices)
        # ▼▼ 修正後: バッファを使用して動的メモリ確保を排除
        diff_len = len(finite_prices) - 1
        for j in range(diff_len):
            diff_buffer[j] = finite_prices[j + 1] - finite_prices[j]
        price_changes = diff_buffer[:diff_len]

        if diff_len < 5:
            continue

        local_window = min(5, diff_len // 3)
        # ▼▼ 修正前: tension_scores = []
        # ▼▼ 修正後: 事前確保バッファとインデックスカウンタを使用
        tension_count = 0

        for j in range(local_window, diff_len - local_window):
            local_changes = price_changes[j - local_window : j + local_window + 1]

            sign_changes = 0
            for k in range(len(local_changes) - 1):
                if np.sign(local_changes[k]) != np.sign(local_changes[k + 1]):
                    sign_changes += 1

            direction_dissonance = (
                float(sign_changes) / (float(len(local_changes)) + 1e-10)
                if len(local_changes) > 0
                else 0.0
            )

            # 修正後
            max_volatility = np.max(np.abs(local_changes))
            intensity_dissonance = max_volatility / (
                np.mean(np.abs(local_changes)) + 1e-10
            )

            total_tension = (direction_dissonance + intensity_dissonance) / 2.0
            tension_buffer[tension_count] = min(total_tension, 1.0)
            tension_count += 1

        if tension_count > 0:
            results[i] = np.mean(tension_buffer[:tension_count])
        else:
            results[i] = 0.0

    return results


# =============================================================================
# 生体力学: 運動エネルギー・筋力・生体力学効率・エネルギー消費量 (並列化版)
# =============================================================================


# ▼▼ 修正前: @nb.njit(fastmath=True, cache=True, parallel=True)
# ▼▼ 修正後: parallel=True を削除し、バッファを事前確保
@nb.njit(fastmath=True, cache=True)
def rolling_kinetic_energy_udf(prices: np.ndarray, window_size: int) -> np.ndarray:
    n = len(prices)
    results = np.full(n, np.nan, dtype=np.float64)

    # 事前確保バッファ
    kin_buffer = np.zeros(window_size, dtype=np.float64)

    # ▼▼ 修正前: for i in nb.prange(window_size - 1, n):
    # ▼▼ 修正後: nb.prange を range に変更
    for i in range(window_size - 1, n):
        window_prices = prices[i - window_size + 1 : i + 1]

        if len(window_prices) < 10:
            continue

        finite_prices = window_prices[np.isfinite(window_prices)]
        if len(finite_prices) < 10:
            continue

        # ▼▼ 修正前: velocities = np.diff(finite_prices) / (finite_prices[:-1] + 1e-10)
        # ▼▼ 修正前: kinetic_energies = 0.5 * masses * velocities**2
        # ▼▼ 修正後: 動的確保を排除し、バッファに直接運動エネルギーを書き込む
        vel_len = len(finite_prices) - 1
        if vel_len < 2:
            continue

        for j in range(vel_len):
            vel = (finite_prices[j + 1] - finite_prices[j]) / (finite_prices[j] + 1e-10)
            kin_buffer[j] = 0.5 * 1.0 * (vel * vel)  # masses = 1.0

        results[i] = np.mean(kin_buffer[:vel_len])

    return results


# ▼▼ 修正前: @nb.njit(fastmath=True, cache=True, parallel=True)
# ▼▼ 修正後: parallel=True を削除し、バッファを事前確保
@nb.njit(fastmath=True, cache=True)
def rolling_muscle_force_udf(prices: np.ndarray, window_size: int) -> np.ndarray:
    n = len(prices)
    results = np.full(n, np.nan, dtype=np.float64)

    # 事前確保バッファ
    vel_buffer = np.zeros(window_size, dtype=np.float64)
    force_buffer = np.zeros(window_size, dtype=np.float64)
    dir_buffer = np.zeros(window_size, dtype=np.float64)
    sustained_buffer = np.zeros(window_size, dtype=np.float64)

    # ▼▼ 修正前: for i in nb.prange(window_size - 1, n):
    # ▼▼ 修正後: nb.prange を range に変更
    for i in range(window_size - 1, n):
        window_prices = prices[i - window_size + 1 : i + 1]

        if len(window_prices) < 15:
            continue

        finite_prices = window_prices[np.isfinite(window_prices)]
        if len(finite_prices) < 15:
            continue

        # ▼▼ 修正前: velocities = np.diff(finite_prices) / (finite_prices[:-1] + 1e-10)
        # ▼▼ 修正前: accelerations = np.diff(velocities)
        # ▼▼ 修正後: バッファを活用し、配列作成と一時オブジェクトを排除
        vel_len = len(finite_prices) - 1
        for j in range(vel_len):
            vel_buffer[j] = (finite_prices[j + 1] - finite_prices[j]) / (
                finite_prices[j] + 1e-10
            )

        if vel_len < 2:
            continue

        acc_len = vel_len - 1
        for j in range(acc_len):
            acc = vel_buffer[j + 1] - vel_buffer[j]
            force_buffer[j] = abs(acc)
            if acc > 0.0:
                dir_buffer[j] = 1.0
            elif acc < 0.0:
                dir_buffer[j] = -1.0
            else:
                dir_buffer[j] = 0.0

        # ▼▼ 修正前: force_directions = np.sign(accelerations) (等)
        # ▼▼ 修正後: 事前確保したバッファとインデックスカウンタで管理
        sus_count = 0
        current_direction = dir_buffer[0] if acc_len > 0 else 0.0
        current_duration = 1.0
        current_force_sum = force_buffer[0] if acc_len > 0 else 0.0

        for j in range(1, acc_len):
            if dir_buffer[j] == current_direction and current_direction != 0.0:
                current_duration += 1.0
                current_force_sum += force_buffer[j]
            else:
                if current_duration > 1.0:
                    sustained_buffer[sus_count] = current_force_sum / current_duration
                    sus_count += 1

                current_direction = dir_buffer[j]
                current_duration = 1.0
                current_force_sum = force_buffer[j]

        if current_duration > 1.0:
            sustained_buffer[sus_count] = current_force_sum / current_duration
            sus_count += 1

        instantaneous_force = np.mean(force_buffer[:acc_len]) if acc_len > 0 else 0.0
        sustained_force = (
            np.mean(sustained_buffer[:sus_count]) if sus_count > 0 else 0.0
        )

        results[i] = 0.7 * instantaneous_force + 0.3 * sustained_force

    return results


# ▼▼ 修正前: @nb.njit(fastmath=True, cache=True, parallel=True)
# ▼▼ 修正後: parallel=True を削除し、バッファを事前確保
@nb.njit(fastmath=True, cache=True)
def rolling_biomechanical_efficiency_udf(
    prices: np.ndarray, window_size: int
) -> np.ndarray:
    n = len(prices)
    results = np.full(n, np.nan, dtype=np.float64)

    # 事前確保バッファ
    vel_buffer = np.zeros(window_size, dtype=np.float64)

    for i in range(window_size - 1, n):
        window_prices = prices[i - window_size + 1 : i + 1]
        finite_prices = window_prices[np.isfinite(window_prices)]
        if len(finite_prices) < 20:
            continue

        # ▼▼ 修正後: 速度を絶対差分(Δp)[P]から相対リターン(Δp/p)[r]に変換
        # vel, acc が [r] 次元に統一され、次元が (Σ|vel|)²/Σ(vel²+acc²) = [r]²/[r]² = 無次元になる
        vel_len = len(finite_prices) - 1
        rms_vel_sq = 0.0  # Σ(vel²) → [r]²
        rms_acc_sq = 0.0  # Σ(acc²) → [r]²
        sum_abs_vel = 0.0  # Σ|vel|  → [r]

        for j in range(vel_len):
            # 相対リターン [r]
            vel = (finite_prices[j + 1] - finite_prices[j]) / (finite_prices[j] + 1e-10)
            vel_buffer[j] = vel
            rms_vel_sq += vel * vel
            sum_abs_vel += abs(vel)

        acc_len = vel_len - 1
        for j in range(acc_len):
            acc = vel_buffer[j + 1] - vel_buffer[j]  # [r]
            rms_acc_sq += acc * acc

        total_energy = rms_vel_sq + rms_acc_sq  # [r]²

        if total_energy > 1e-10 and sum_abs_vel > 1e-10:
            # ▼▼ 修正前: total_displacement([P]) / total_energy([P]²) → [P]⁻¹ スケール依存
            # ▼▼ 修正後: (Σ|vel|)²([r]²) / total_energy([r]²) → 無次元 ✅
            numerator = sum_abs_vel * sum_abs_vel  # [r]²
            raw_efficiency = numerator / total_energy  # [r]²/[r]² = 無次元
            # 理論的上限はvel_len（全エネルギーが一方向変位に使われた場合）なのでvel_lenで正規化
            results[i] = min(raw_efficiency / (vel_len + 1e-10), 1.0)
        else:
            results[i] = 0.0

    return results


# ▼▼ 修正前: @nb.njit(fastmath=True, cache=True, parallel=True)
# ▼▼ 修正後: parallel=True を削除し、バッファを事前確保
@nb.njit(fastmath=True, cache=True)
def rolling_energy_expenditure_udf(prices: np.ndarray, window_size: int) -> np.ndarray:
    n = len(prices)
    results = np.full(n, np.nan, dtype=np.float64)

    # 事前確保バッファ
    vel_buffer = np.zeros(window_size, dtype=np.float64)

    # ▼▼ 修正前: for i in nb.prange(window_size - 1, n):
    # ▼▼ 修正後: nb.prange を range に変更
    for i in range(window_size - 1, n):
        window_prices = prices[i - window_size + 1 : i + 1]
        finite_prices = window_prices[np.isfinite(window_prices)]
        if len(finite_prices) < 15:
            continue

        # ▼▼ 修正前: baseline_energy = np.var(finite_prices, ddof=1)
        # ▼▼ 修正後: ベッセル補正による不偏分散の手動計算
        n_fp = len(finite_prices)
        baseline_energy = np.var(finite_prices) * (n_fp / (n_fp - 1.0))

        mov_en_sum = 0.0
        vel_len = len(finite_prices) - 1
        for j in range(vel_len):
            vel = finite_prices[j + 1] - finite_prices[j]
            vel_buffer[j] = vel
            mov_en_sum += vel * vel
        movement_energy = mov_en_sum / float(vel_len) if vel_len > 0 else 0.0

        acc_en_sum = 0.0
        acc_len = vel_len - 1
        for j in range(acc_len):
            acc = vel_buffer[j + 1] - vel_buffer[j]
            acc_en_sum += acc * acc
        acceleration_energy = acc_en_sum / float(acc_len) if acc_len > 0 else 0.0

        # ▼▼ 修正前: x = np.arange(len(finite_prices), dtype=np.float64)
        # ▼▼ 修正後: 配列 x と linear_trend の動的生成を排除し、スカラー計算に置換
        n_points = float(len(finite_prices))
        sum_x = 0.0
        sum_y = 0.0
        sum_xy = 0.0
        sum_x2 = 0.0
        for j in range(len(finite_prices)):
            x_val = float(j)
            y_val = finite_prices[j]
            sum_x += x_val
            sum_y += y_val
            sum_xy += x_val * y_val
            sum_x2 += x_val * x_val

        nonlinearity_energy = 0.0
        denominator = n_points * sum_x2 - sum_x * sum_x
        if abs(denominator) > 1e-10:
            slope = (n_points * sum_xy - sum_x * sum_y) / denominator
            intercept = (sum_y - slope * sum_x) / n_points

            nonlin_sum = 0.0
            for j in range(len(finite_prices)):
                pred = intercept + slope * float(j)
                diff = finite_prices[j] - pred
                nonlin_sum += diff * diff
            nonlinearity_energy = nonlin_sum / n_points

        results[i] = (
            baseline_energy
            + movement_energy
            + acceleration_energy
            + nonlinearity_energy
        )
    return results


class CalculationEngine:
    """
    計算核心クラス（60%） - Project Forge統合版（修正版：ディスクベース垂直分割）
    責務:
    - Polars Expressionによる高度な特徴量計算（90%のタスク）
    - .map_batches()経由のNumba JIT最適化UDF（10%のカスタム「アルファ」ロジック）
    - 【修正】物理的垂直分割：ディスク中間ファイルによるメモリ・スラッシング回避
    - Polars内部並列化による自動マルチスレッド実行
    - 2段階品質保証（基本/フォールバック）
    """

    def __init__(self, config: ProcessingConfig):
        self.config = config
        # ▼▼ 削除: QualityAssuranceクラスは削除されたため、この行も削除
        # self.qa = QualityAssurance()  <-- この行を完全に削除してください
        self.memory_monitor = MemoryMonitor(config.memory_limit_gb)
        self.prefix = f"e{config.engine_id.replace('e', '')}_"  # 例: "e1f_"
        # 【修正】ディスクベース垂直分割用の一時ディレクトリ
        self.temp_dir = Path(tempfile.mkdtemp(prefix=f"features_{config.engine_id}_"))
        logger.info(f"一時ディレクトリ作成: {self.temp_dir}")

    # ▼▼ 修正: sample_weightを計算して付与するメソッドを追加
    def append_sample_weight(self, lazy_frame: pl.LazyFrame) -> pl.LazyFrame:
        """システム統一要件: ATR13に基づくZスコアウェイトの計算と付与"""

        def _compute_weight(s: pl.Series) -> pl.Series:
            df = s.struct.unnest()
            h = df["high"].to_numpy()
            l = df["low"].to_numpy()
            c = df["close"].to_numpy()
            w = calculate_sample_weight_udf(h, l, c)
            return pl.Series(w)

        return lazy_frame.with_columns(
            [
                # 修正後
                pl.struct(["high", "low", "close"])
                .map_batches(_compute_weight, return_dtype=pl.Float64)
                .alias("sample_weight")
            ]
        )

    # ▲▲ 追加ここまで ▲▲

    def _get_all_feature_expressions(self) -> Dict[str, pl.Expr]:
        """
        全学際的・実験的特徴量の名前とPolars Expressionの対応辞書を返すファクトリメソッド。
        これにより、必要な特徴量だけを効率的に計算できるようになる。
        【特徴量ファクトリパターン】
        """
        expressions = {}
        p = self.prefix

        # ネットワーク科学系 - map_batchesを使用（仕様書準拠）【クロージャー問題修正版】
        for window in self.config.window_sizes["network"]:
            expressions[f"{p}network_density_{window}"] = (
                pl.col("close")
                .map_batches(
                    lambda s, w=window: pl.Series(
                        rolling_network_density_udf(s.to_numpy(), window_size=w)
                    ),
                    # ▼▼ 追加: 戻り値の型を明記し、型推論エラーによるクラッシュを防ぐ
                    return_dtype=pl.Float64,
                )
                .alias(f"{p}network_density_{window}")
            )

            expressions[f"{p}network_clustering_{window}"] = (
                pl.col("close")
                .map_batches(
                    lambda s, w=window: pl.Series(
                        rolling_network_clustering_udf(s.to_numpy(), window_size=w)
                    ),
                    # ▼▼ 追加: 戻り値の型を明記し、型推論エラーによるクラッシュを防ぐ
                    return_dtype=pl.Float64,
                )
                .alias(f"{p}network_clustering_{window}")
            )

        # 言語学系 - map_batchesを使用（仕様書準拠）【クロージャー問題修正版】
        for window in self.config.window_sizes["linguistic"]:
            expressions[f"{p}vocabulary_diversity_{window}"] = (
                pl.col("close")
                .map_batches(
                    lambda s, w=window: pl.Series(
                        rolling_vocabulary_diversity_udf(s.to_numpy(), window_size=w)
                    ),
                    # ▼▼ 追加: 戻り値の型を明記し、型推論エラーによるクラッシュを防ぐ
                    return_dtype=pl.Float64,
                )
                .alias(f"{p}vocabulary_diversity_{window}")
            )

            expressions[f"{p}linguistic_complexity_{window}"] = (
                pl.col("close")
                .map_batches(
                    lambda s, w=window: pl.Series(
                        rolling_linguistic_complexity_udf(s.to_numpy(), window_size=w)
                    ),
                    # ▼▼ 追加: 戻り値の型を明記し、型推論エラーによるクラッシュを防ぐ
                    return_dtype=pl.Float64,
                )
                .alias(f"{p}linguistic_complexity_{window}")
            )

            expressions[f"{p}semantic_flow_{window}"] = (
                pl.col("close")
                .map_batches(
                    lambda s, w=window: pl.Series(
                        rolling_semantic_flow_udf(s.to_numpy(), window_size=w)
                    ),
                    # ▼▼ 追加: 戻り値の型を明記し、型推論エラーによるクラッシュを防ぐ
                    return_dtype=pl.Float64,
                )
                .alias(f"{p}semantic_flow_{window}")
            )

        # 美学系 - map_batchesを使用（仕様書準拠）【クロージャー問題修正版】
        for window in self.config.window_sizes["aesthetic"]:
            expressions[f"{p}golden_ratio_adherence_{window}"] = (
                pl.col("close")
                .map_batches(
                    lambda s, w=window: pl.Series(
                        rolling_golden_ratio_adherence_udf(s.to_numpy(), window_size=w)
                    ),
                    # ▼▼ 追加: 戻り値の型を明記し、型推論エラーによるクラッシュを防ぐ
                    return_dtype=pl.Float64,
                )
                .alias(f"{p}golden_ratio_adherence_{window}")
            )

            expressions[f"{p}symmetry_measure_{window}"] = (
                pl.col("close")
                .map_batches(
                    lambda s, w=window: pl.Series(
                        rolling_symmetry_measure_udf(s.to_numpy(), window_size=w)
                    ),
                    # ▼▼ 追加: 戻り値の型を明記し、型推論エラーによるクラッシュを防ぐ
                    return_dtype=pl.Float64,
                )
                .alias(f"{p}symmetry_measure_{window}")
            )

            expressions[f"{p}aesthetic_balance_{window}"] = (
                pl.col("close")
                .map_batches(
                    lambda s, w=window: pl.Series(
                        rolling_aesthetic_balance_udf(s.to_numpy(), window_size=w)
                    ),
                    # ▼▼ 追加: 戻り値の型を明記し、型推論エラーによるクラッシュを防ぐ
                    return_dtype=pl.Float64,
                )
                .alias(f"{p}aesthetic_balance_{window}")
            )

        # 音楽理論系 - map_batchesを使用（仕様書準拠）【クロージャー問題修正版】
        for window in self.config.window_sizes["musical"]:
            expressions[f"{p}tonality_{window}"] = (
                pl.col("close")
                .map_batches(
                    lambda s, w=window: pl.Series(
                        rolling_tonality_udf(s.to_numpy(), window_size=w)
                    ),
                    # ▼▼ 追加: 戻り値の型を明記し、型推論エラーによるクラッシュを防ぐ
                    return_dtype=pl.Float64,
                )
                .alias(f"{p}tonality_{window}")
            )

            expressions[f"{p}rhythm_pattern_{window}"] = (
                pl.col("close")
                .map_batches(
                    lambda s, w=window: pl.Series(
                        rolling_rhythm_pattern_udf(s.to_numpy(), window_size=w)
                    ),
                    # ▼▼ 追加: 戻り値の型を明記し、型推論エラーによるクラッシュを防ぐ
                    return_dtype=pl.Float64,
                )
                .alias(f"{p}rhythm_pattern_{window}")
            )

            expressions[f"{p}harmony_{window}"] = (
                pl.col("close")
                .map_batches(
                    lambda s, w=window: pl.Series(
                        rolling_harmony_udf(s.to_numpy(), window_size=w)
                    ),
                    # ▼▼ 追加: 戻り値の型を明記し、型推論エラーによるクラッシュを防ぐ
                    return_dtype=pl.Float64,
                )
                .alias(f"{p}harmony_{window}")
            )

            expressions[f"{p}musical_tension_{window}"] = (
                pl.col("close")
                .map_batches(
                    lambda s, w=window: pl.Series(
                        rolling_musical_tension_udf(s.to_numpy(), window_size=w)
                    ),
                    # ▼▼ 追加: 戻り値の型を明記し、型推論エラーによるクラッシュを防ぐ
                    return_dtype=pl.Float64,
                )
                .alias(f"{p}musical_tension_{window}")
            )

        # 生体力学系 - map_batchesを使用（仕様書準拠）【クロージャー問題修正版】
        for window in self.config.window_sizes["biomechanical"]:
            expressions[f"{p}kinetic_energy_{window}"] = (
                pl.col("close")
                .map_batches(
                    lambda s, w=window: pl.Series(
                        rolling_kinetic_energy_udf(s.to_numpy(), window_size=w)
                    ),
                    # ▼▼ 追加: 戻り値の型を明記し、型推論エラーによるクラッシュを防ぐ
                    return_dtype=pl.Float64,
                )
                .alias(f"{p}kinetic_energy_{window}")
            )

            expressions[f"{p}muscle_force_{window}"] = (
                pl.col("close")
                .map_batches(
                    lambda s, w=window: pl.Series(
                        rolling_muscle_force_udf(s.to_numpy(), window_size=w)
                    ),
                    # ▼▼ 追加: 戻り値の型を明記し、型推論エラーによるクラッシュを防ぐ
                    return_dtype=pl.Float64,
                )
                .alias(f"{p}muscle_force_{window}")
            )

            expressions[f"{p}biomechanical_efficiency_{window}"] = (
                pl.col("close")
                .map_batches(
                    lambda s, w=window: pl.Series(
                        rolling_biomechanical_efficiency_udf(
                            s.to_numpy(), window_size=w
                        )
                    ),
                    # ▼▼ 追加: 戻り値の型を明記し、型推論エラーによるクラッシュを防ぐ
                    return_dtype=pl.Float64,
                )
                .alias(f"{p}biomechanical_efficiency_{window}")
            )

            expressions[f"{p}energy_expenditure_{window}"] = (
                pl.col("close")
                .pct_change()
                .map_batches(
                    lambda s, w=window: pl.Series(
                        rolling_energy_expenditure_udf(s.to_numpy(), window_size=w)
                    ),
                    return_dtype=pl.Float64,
                )
                .alias(f"{p}energy_expenditure_{window}")
            )
        return expressions

    def get_feature_groups(self) -> Dict[str, Dict[str, pl.Expr]]:
        """特徴量グループ定義を外部から取得可能にする"""
        return self._create_vertical_slices()

    # ▼▼ 修正: 引数に timeframe: str = "M1" を追加
    def calculate_one_group(
        self,
        lazy_frame: pl.LazyFrame,
        group_name: str,
        group_expressions: Dict[str, pl.Expr],
        timeframe: str = "M1",
    ) -> pl.LazyFrame:
        """
        単一グループの特徴量のみを計算（高速化修正版）
        グループ名に基づいて適切な特徴量計算メソッドを呼び出し、重複計算を回避
        """
        logger.info(f"グループ計算開始: {group_name}")

        # メモリ安全性チェック（必須）
        is_safe, message = self.memory_monitor.check_memory_safety()
        if not is_safe:
            raise MemoryError(f"メモリ不足のためグループ処理を中断: {message}")

        try:
            # グループ名に基づいて効率的な特徴量計算を実行
            if group_name == "network_science":
                group_result_lf = self._create_network_science_features(lazy_frame)
            elif group_name == "linguistics":
                group_result_lf = self._create_linguistics_features(lazy_frame)
            elif group_name == "aesthetics":
                group_result_lf = self._create_aesthetics_features(lazy_frame)
            elif group_name == "music_theory":
                group_result_lf = self._create_music_theory_features(lazy_frame)
            elif group_name == "biomechanics":
                group_result_lf = self._create_biomechanics_features(lazy_frame)
            else:
                # フォールバック: 従来の方式
                logger.warning(f"未対応グループ名、フォールバック処理: {group_name}")
                group_result_lf = lazy_frame.with_columns(
                    list(group_expressions.values())
                )

            # スキーマから実際に存在するカラムを確認
            available_schema = group_result_lf.collect_schema()
            available_columns = list(available_schema.names())

            # 基本カラムとして存在するもののみを選択
            base_columns = ["timestamp", "open", "high", "low", "close", "volume"]
            if "timeframe" in available_columns:
                base_columns.append("timeframe")
            if "sample_weight" in available_columns:
                base_columns.append("sample_weight")

            # このグループの特徴量カラムのみを抽出
            group_feature_columns = [
                col for col in available_columns if col.startswith(self.prefix)
            ]
            select_columns = base_columns + group_feature_columns

            # 実際に存在するカラムのみを選択
            final_select_columns = [
                col for col in select_columns if col in available_columns
            ]
            group_final_lf = group_result_lf.select(final_select_columns)

            # ▼▼ 修正: timeframe を引数として渡す
            stabilized_lf = self.apply_quality_assurance_to_group(
                group_final_lf, group_feature_columns, timeframe=timeframe
            )

            logger.info(
                f"グループ計算完了: {group_name} - {len(group_feature_columns)}個の特徴量"
            )
            return stabilized_lf

        except Exception as e:
            logger.error(f"グループ計算エラー ({group_name}): {e}")
            raise

    # ▼▼ 修正: 引数に timeframe: str = "M1" を追加
    def apply_quality_assurance_to_group(
        self,
        lazy_frame: pl.LazyFrame,
        feature_columns: List[str],
        timeframe: str = "M1",
    ) -> pl.LazyFrame:
        """単一グループに対する品質保証システムの適用"""
        if not feature_columns:
            return lazy_frame

        # ▼▼ 追加: timeframeに基づく動的half_life計算
        half_life = self.config.timeframe_bars_per_day.get(timeframe, 1440)
        logger.info(
            f"品質保証適用: {len(feature_columns)}個の特徴量 (EWM 5σクリッピング, half_life={half_life})"
        )

        stabilization_exprs = []

        for col_name in feature_columns:
            col_expr = pl.col(col_name).map_batches(
                lambda s: s.replace([np.inf, -np.inf], None),
                return_dtype=pl.Float64,
            )

            # ▼▼ 修正: half_life=half_life に置き換え
            # 修正後
            ewm_mean = col_expr.ewm_mean(
                half_life=half_life, ignore_nulls=True, adjust=False
            )
            ewm_std = col_expr.ewm_std(
                half_life=half_life, ignore_nulls=True, adjust=False
            )

            lower_bound = ewm_mean - 5.0 * ewm_std
            upper_bound = ewm_mean + 5.0 * ewm_std

            stabilized_col = (
                pl.col(col_name)
                .clip(lower_bound=lower_bound, upper_bound=upper_bound)
                .fill_null(0.0)
                .fill_nan(0.0)
                .alias(col_name)
            )

            stabilization_exprs.append(stabilized_col)

        result = lazy_frame.with_columns(stabilization_exprs)
        return result

    def _create_vertical_slices(self) -> Dict[str, Dict[str, pl.Expr]]:
        """物理的垂直分割: 特徴量を論理グループに分割"""
        all_expressions = self._get_all_feature_expressions()

        # メモリ使用量を考慮したグルーピング（英語キー使用）
        slices = {}
        p = self.prefix

        # グループ1: ネットワーク科学系（中程度）
        slices["network_science"] = {
            name: expr
            for name, expr in all_expressions.items()
            if any(
                pattern in name for pattern in ["network_density", "network_clustering"]
            )
        }

        # グループ2: 言語学系（重い）
        slices["linguistics"] = {
            name: expr
            for name, expr in all_expressions.items()
            if any(
                pattern in name
                for pattern in [
                    "vocabulary_diversity",
                    "linguistic_complexity",
                    "semantic_flow",
                ]
            )
        }

        # グループ3: 美学系（中程度）
        slices["aesthetics"] = {
            name: expr
            for name, expr in all_expressions.items()
            if any(
                pattern in name
                for pattern in ["golden_ratio", "symmetry_measure", "aesthetic_balance"]
            )
        }

        # グループ4: 音楽理論系（重い）
        slices["music_theory"] = {
            name: expr
            for name, expr in all_expressions.items()
            if any(
                pattern in name
                for pattern in [
                    "tonality",
                    "rhythm_pattern",
                    "harmony",
                    "musical_tension",
                ]
            )
        }

        # グループ5: 生体力学系（重い）
        slices["biomechanics"] = {
            name: expr
            for name, expr in all_expressions.items()
            if any(
                pattern in name
                for pattern in [
                    "kinetic_energy",
                    "muscle_force",
                    "biomechanical_efficiency",
                    "energy_expenditure",
                ]
            )
        }

        # 分割されなかった特徴量があれば警告
        total_assigned = sum(len(group) for group in slices.values())
        if total_assigned != len(all_expressions):
            logger.warning(f"未分割特徴量: {len(all_expressions) - total_assigned}個")

        return slices

    def _cleanup_temp_files(self):
        """一時ファイルのクリーンアップ"""
        # 修正後
        try:
            if self.temp_dir.exists():
                import shutil

                shutil.rmtree(self.temp_dir)
                logger.info(f"一時ディレクトリ削除: {self.temp_dir}")
        except Exception as e:
            logger.warning(f"一時ファイルクリーンアップエラー: {e}")

    # ▼▼ 修正: 引数に timeframe: str = "M1" を追加（※Engine 1A-1E用、または1Fでの全体適用用）
    def apply_quality_assurance(
        self, lazy_frame: pl.LazyFrame, timeframe: str = "M1"
    ) -> pl.LazyFrame:
        """2段階品質保証システムの適用"""
        # ▼▼ 追加: timeframeに基づく動的half_life計算
        half_life = self.config.timeframe_bars_per_day.get(timeframe, 1440)
        logger.info(
            f"品質保証システム適用開始: {half_life}バー基準のEWMクリッピングを適用します。"
        )

        schema = lazy_frame.collect_schema()
        feature_columns = [col for col in schema.names() if col.startswith(self.prefix)]

        if not feature_columns:
            return lazy_frame

        stabilization_exprs = []
        for col_name in feature_columns:
            col_expr = pl.col(col_name).map_batches(
                lambda s: s.replace([np.inf, -np.inf], None),
                return_dtype=pl.Float64,
            )

            # ▼▼ 修正: half_life=half_life に置き換え
            # 修正後
            ewm_mean = col_expr.ewm_mean(
                half_life=half_life, ignore_nulls=True, adjust=False
            )
            ewm_std = col_expr.ewm_std(
                half_life=half_life, ignore_nulls=True, adjust=False
            )
            lower_bound = ewm_mean - 5.0 * ewm_std
            upper_bound = ewm_mean + 5.0 * ewm_std

            stabilized_col = (
                pl.col(col_name)
                .clip(lower_bound=lower_bound, upper_bound=upper_bound)
                .fill_null(0.0)
                .fill_nan(0.0)
                .alias(col_name)
            )
            stabilization_exprs.append(stabilized_col)

        result = lazy_frame.with_columns(stabilization_exprs)
        return result

    def _create_network_science_features(
        self, lazy_frame: pl.LazyFrame
    ) -> pl.LazyFrame:
        """ネットワーク科学系特徴量の計算（高速化対応）【クロージャー問題修正版】"""
        exprs = []
        p = self.prefix

        for window in self.config.window_sizes["network"]:
            # ▼▼ 修正前: return_dtypeなし
            # ▼▼ 修正後: return_dtype=pl.Float64 を明記
            exprs.append(
                pl.col("close")
                .map_batches(
                    lambda s, w=window: pl.Series(
                        rolling_network_density_udf(s.to_numpy(), window_size=w)
                    ),
                    return_dtype=pl.Float64,
                )
                .alias(f"{p}network_density_{window}")
            )

            exprs.append(
                pl.col("close")
                .map_batches(
                    lambda s, w=window: pl.Series(
                        rolling_network_clustering_udf(s.to_numpy(), window_size=w)
                    ),
                    return_dtype=pl.Float64,
                )
                .alias(f"{p}network_clustering_{window}")
            )

        return lazy_frame.with_columns(exprs)

    def _create_linguistics_features(self, lazy_frame: pl.LazyFrame) -> pl.LazyFrame:
        """言語学系特徴量の計算（高速化対応）【クロージャー問題修正版】"""
        exprs = []
        p = self.prefix

        for window in self.config.window_sizes["linguistic"]:
            exprs.append(
                pl.col("close")
                .map_batches(
                    lambda s, w=window: pl.Series(
                        rolling_vocabulary_diversity_udf(s.to_numpy(), window_size=w)
                    ),
                    # ▼▼ 追加: 戻り値の型を明記し、型推論エラーによるクラッシュを防ぐ
                    return_dtype=pl.Float64,
                )
                .alias(f"{p}vocabulary_diversity_{window}")
            )

            exprs.append(
                pl.col("close")
                .map_batches(
                    lambda s, w=window: pl.Series(
                        rolling_linguistic_complexity_udf(s.to_numpy(), window_size=w)
                    ),
                    # ▼▼ 追加: 戻り値の型を明記し、型推論エラーによるクラッシュを防ぐ
                    return_dtype=pl.Float64,
                )
                .alias(f"{p}linguistic_complexity_{window}")
            )

            exprs.append(
                pl.col("close")
                .map_batches(
                    lambda s, w=window: pl.Series(
                        rolling_semantic_flow_udf(s.to_numpy(), window_size=w)
                    ),
                    # ▼▼ 追加: 戻り値の型を明記し、型推論エラーによるクラッシュを防ぐ
                    return_dtype=pl.Float64,
                )
                .alias(f"{p}semantic_flow_{window}")
            )

        return lazy_frame.with_columns(exprs)

    def _create_aesthetics_features(self, lazy_frame: pl.LazyFrame) -> pl.LazyFrame:
        """美学系特徴量の計算（高速化対応）【クロージャー問題修正版】"""
        exprs = []
        p = self.prefix

        for window in self.config.window_sizes["aesthetic"]:
            exprs.append(
                pl.col("close")
                .map_batches(
                    lambda s, w=window: pl.Series(
                        rolling_golden_ratio_adherence_udf(s.to_numpy(), window_size=w)
                    ),
                    # ▼▼ 追加: 戻り値の型を明記し、型推論エラーによるクラッシュを防ぐ
                    return_dtype=pl.Float64,
                )
                .alias(f"{p}golden_ratio_adherence_{window}")
            )

            exprs.append(
                pl.col("close")
                .map_batches(
                    lambda s, w=window: pl.Series(
                        rolling_symmetry_measure_udf(s.to_numpy(), window_size=w)
                    ),
                    # ▼▼ 追加: 戻り値の型を明記し、型推論エラーによるクラッシュを防ぐ
                    return_dtype=pl.Float64,
                )
                .alias(f"{p}symmetry_measure_{window}")
            )

            exprs.append(
                pl.col("close")
                .map_batches(
                    lambda s, w=window: pl.Series(
                        rolling_aesthetic_balance_udf(s.to_numpy(), window_size=w)
                    ),
                    # ▼▼ 追加: 戻り値の型を明記し、型推論エラーによるクラッシュを防ぐ
                    return_dtype=pl.Float64,
                )
                .alias(f"{p}aesthetic_balance_{window}")
            )

        return lazy_frame.with_columns(exprs)

    def _create_music_theory_features(self, lazy_frame: pl.LazyFrame) -> pl.LazyFrame:
        """音楽理論系特徴量の計算（高速化対応）【クロージャー問題修正版】"""
        exprs = []
        p = self.prefix

        for window in self.config.window_sizes["musical"]:
            exprs.append(
                pl.col("close")
                .map_batches(
                    lambda s, w=window: pl.Series(
                        rolling_tonality_udf(s.to_numpy(), window_size=w)
                    ),
                    # ▼▼ 追加: 戻り値の型を明記し、型推論エラーによるクラッシュを防ぐ
                    return_dtype=pl.Float64,
                )
                .alias(f"{p}tonality_{window}")
            )

            exprs.append(
                pl.col("close")
                .map_batches(
                    lambda s, w=window: pl.Series(
                        rolling_rhythm_pattern_udf(s.to_numpy(), window_size=w)
                    ),
                    # ▼▼ 追加: 戻り値の型を明記し、型推論エラーによるクラッシュを防ぐ
                    return_dtype=pl.Float64,
                )
                .alias(f"{p}rhythm_pattern_{window}")
            )

            exprs.append(
                pl.col("close")
                .map_batches(
                    lambda s, w=window: pl.Series(
                        rolling_harmony_udf(s.to_numpy(), window_size=w)
                    ),
                    # ▼▼ 追加: 戻り値の型を明記し、型推論エラーによるクラッシュを防ぐ
                    return_dtype=pl.Float64,
                )
                .alias(f"{p}harmony_{window}")
            )

            exprs.append(
                pl.col("close")
                .map_batches(
                    lambda s, w=window: pl.Series(
                        rolling_musical_tension_udf(s.to_numpy(), window_size=w)
                    ),
                    # ▼▼ 追加: 戻り値の型を明記し、型推論エラーによるクラッシュを防ぐ
                    return_dtype=pl.Float64,
                )
                .alias(f"{p}musical_tension_{window}")
            )

        return lazy_frame.with_columns(exprs)

    def _create_biomechanics_features(self, lazy_frame: pl.LazyFrame) -> pl.LazyFrame:
        """生体力学系特徴量の計算（高速化対応）【クロージャー問題修正版】"""
        exprs = []
        p = self.prefix

        for window in self.config.window_sizes["biomechanical"]:
            exprs.append(
                pl.col("close")
                .map_batches(
                    lambda s, w=window: pl.Series(
                        rolling_kinetic_energy_udf(s.to_numpy(), window_size=w)
                    ),
                    # ▼▼ 追加: 戻り値の型を明記し、型推論エラーによるクラッシュを防ぐ
                    return_dtype=pl.Float64,
                )
                .alias(f"{p}kinetic_energy_{window}")
            )

            exprs.append(
                pl.col("close")
                .map_batches(
                    lambda s, w=window: pl.Series(
                        rolling_muscle_force_udf(s.to_numpy(), window_size=w)
                    ),
                    # ▼▼ 追加: 戻り値の型を明記し、型推論エラーによるクラッシュを防ぐ
                    return_dtype=pl.Float64,
                )
                .alias(f"{p}muscle_force_{window}")
            )

            exprs.append(
                pl.col("close")
                .map_batches(
                    lambda s, w=window: pl.Series(
                        rolling_biomechanical_efficiency_udf(
                            s.to_numpy(), window_size=w
                        )
                    ),
                    # ▼▼ 追加: 戻り値の型を明記し、型推論エラーによるクラッシュを防ぐ
                    return_dtype=pl.Float64,
                )
                .alias(f"{p}biomechanical_efficiency_{window}")
            )

            # 修正後
            exprs.append(
                pl.col("close")
                .pct_change()
                .map_batches(
                    lambda s, w=window: pl.Series(
                        rolling_energy_expenditure_udf(s.to_numpy(), window_size=w)
                    ),
                    return_dtype=pl.Float64,
                )
                .alias(f"{p}energy_expenditure_{window}")
            )

        return lazy_frame.with_columns(exprs)


class OutputEngine:
    """
    出力管理クラス（10%） - Project Forge準拠
    責務:
    - LazyFrame.sink_parquet()によるストリーミング出力
    - 必要に応じたPyArrowフォールバック（use_pyarrow=True）
    - timeframe別ファイル分離
    - NaN埋め統一処理
    - シンプルな進捗表示
    - 基本メタデータ記録
    """

    def __init__(self, config: ProcessingConfig):
        self.config = config
        self.memory_monitor = MemoryMonitor(config.memory_limit_gb)

        # 出力設定（Project Forge基準）
        self.output_config = {
            "compression": "snappy",  # 固定値
            "dtype": "float64",  # 金融データの精度重視
            "timestamp_handling": "column",  # 機械学習での柔軟性重視
        }

        # エンジン識別子
        self.engine_id = config.engine_id

    def create_output_path(self, timeframe: str) -> Path:
        """出力パス生成 - Project Forge命名規則"""
        filename = f"features_{self.engine_id}_{timeframe}.parquet"
        # ▼▼ 修正前: return Path(self.config.output_path) / filename
        # ▼▼ 修正後: サブフォルダ階層を挟む仕様に変更
        subfolder = "feature_value_a_vast_universeF"
        return Path(self.config.output_path) / subfolder / filename

    def apply_nan_filling(self, lazy_frame: pl.LazyFrame) -> pl.LazyFrame:
        """NaN埋め統一処理"""
        logger.info("NaN埋め処理開始")

        # スキーマを一度だけ取得（メモリ安全）
        schema = lazy_frame.collect_schema()
        all_columns = schema.names()

        # プレフィックスを持つ特徴量カラムを特定
        feature_columns = [
            col for col in all_columns if col.startswith(f"{self.engine_id}_")
        ]

        # NaN埋め式生成
        fill_exprs = []
        for col in feature_columns:
            # NaNを0で埋める（金融データでは一般的）
            fill_exprs.append(pl.col(col).fill_null(0.0).alias(col))

        # 基本カラムはそのまま保持
        basic_columns = [
            "timestamp",
            "open",
            "high",
            "low",
            "close",
            "volume",
            "timeframe",
            "sample_weight",
        ]
        basic_exprs = [pl.col(col) for col in basic_columns if col in all_columns]

        all_exprs = basic_exprs + fill_exprs
        result = lazy_frame.select(all_exprs)

        logger.info(f"NaN埋め処理完了: {len(feature_columns)}個の特徴量")
        return result

    def save_features(self, lazy_frame: pl.LazyFrame, timeframe: str) -> Dict[str, Any]:
        """特徴量ファイル保存"""
        output_path = self.create_output_path(timeframe)
        logger.info(f"特徴量保存開始: {output_path}")

        try:
            # 出力ディレクトリ作成
            output_path.parent.mkdir(parents=True, exist_ok=True)

            # メモリ安全性チェック
            is_safe, message = self.memory_monitor.check_memory_safety()
            if not is_safe:
                raise MemoryError(f"保存処理でメモリ不足: {message}")

            # NaN埋め処理適用
            processed_frame = self.apply_nan_filling(lazy_frame)

            # ▼▼ 修正後: Blueprint v4準拠の型変換（Categorical排除、datetime[us]、int32化）を追加
            schema_names = processed_frame.collect_schema().names()
            cast_exprs = []
            if "timestamp" in schema_names:
                cast_exprs.append(pl.col("timestamp").cast(pl.Datetime("us")))
            if "timeframe" in schema_names:
                cast_exprs.append(pl.col("timeframe").cast(pl.Utf8))
            for col in ["year", "month", "day"]:
                if col in schema_names:
                    cast_exprs.append(pl.col(col).cast(pl.Int32))

            if cast_exprs:
                processed_frame = processed_frame.with_columns(cast_exprs)
            # ▲▲ 追加ここまで ▲▲

            start_time = time.time()

            # 通常のストリーミング出力
            processed_frame.sink_parquet(
                str(output_path), compression=self.output_config["compression"]
            )

            save_time = time.time() - start_time

            if not output_path.exists():
                raise FileNotFoundError(
                    f"出力ファイルが作成されませんでした: {output_path}"
                )

            file_size_mb = output_path.stat().st_size / (1024 * 1024)

            metadata = {
                "timeframe": timeframe,
                "output_path": str(output_path),
                "file_size_mb": round(file_size_mb, 2),
                "save_time_seconds": round(save_time, 2),
                "compression": self.output_config["compression"],
                "engine_id": self.engine_id,
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            }

            logger.info(f"保存完了: {file_size_mb:.2f}MB, {save_time:.2f}秒")
            return metadata

        except Exception as e:
            logger.error(f"保存エラー (timeframe={timeframe}): {e}")
            raise

    def create_summary_report(
        self, processing_metadata: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """処理サマリーレポート生成"""
        total_files = len(processing_metadata)
        total_size_mb = sum(meta.get("file_size_mb", 0) for meta in processing_metadata)
        total_time = sum(
            meta.get("save_time_seconds", 0) for meta in processing_metadata
        )

        summary = {
            "engine_id": self.engine_id,
            "engine_name": "Engine_1F_InterdisciplinaryFeatures",
            "total_files": total_files,
            "total_size_mb": round(total_size_mb, 2),
            "total_processing_time_seconds": round(total_time, 2),
            "average_file_size_mb": round(
                total_size_mb / total_files if total_files > 0 else 0, 2
            ),
            "timeframes_processed": [
                meta.get("timeframe") for meta in processing_metadata
            ],
            "compression_used": "snappy",
            "processing_timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "files_detail": processing_metadata,
        }

        return summary


# ================================================================
# 大規模データ処理（Tick専用）アーキテクチャ：分割・重複・処理・結合
# ================================================================


def get_sorted_partitions(root_dir: Path) -> List[Path]:
    """
    指定されたルートディレクトリからHiveパーティションパスを収集し、
    時系列にソートして返す。
    """
    logging.info(f"パーティションを探索中: {root_dir}")
    partition_paths = sorted(
        list(root_dir.glob("year=*/month=*/day=*")),
        key=lambda p: (
            int(p.parent.parent.name.split("=")[1]),  # year
            int(p.parent.name.split("=")[1]),  # month
            int(p.name.split("=")[1]),  # day
        ),
    )
    logging.info(f"{len(partition_paths)}個のパーティションを発見しました。")
    return partition_paths


def create_augmented_frame(
    current_partition_path: Path, prev_partition_path: Path | None, w_max: int
) -> tuple[pl.DataFrame, int]:
    """
    現在のパーティションデータと、先行パーティションからのオーバーラップ部分を結合し、
    拡張されたデータフレームを生成する。
    """
    lf_current = pl.scan_parquet(current_partition_path / "*.parquet")
    # Tickデータ用にtimeframeカラムを追加
    lf_current = lf_current.with_columns(
        [pl.lit("tick").alias("timeframe").cast(pl.Categorical)]
    )
    df_current = lf_current.collect()
    len_current_partition = df_current.height

    if prev_partition_path is None:
        return df_current, len_current_partition

    lookback_required = w_max - 1

    if lookback_required <= 0:
        return df_current, len_current_partition

    lf_prev = pl.scan_parquet(prev_partition_path / "*.parquet")
    # 前日データにもtimeframeカラムを追加
    lf_prev = lf_prev.with_columns(
        [pl.lit("tick").alias("timeframe").cast(pl.Categorical)]
    )
    df_prefix = lf_prev.tail(lookback_required).collect()

    augmented_df = pl.concat([df_prefix, df_current], how="vertical")

    return augmented_df, len_current_partition


def run_on_partitions_mode(
    config: ProcessingConfig, resume_date: Optional[datetime.date] = None
):
    """
    【修正版】実行モード: Tickデータ専用。パーティションを日単位で逐次処理する。
    責務の明確化: この関数が物理的垂直分割の工程を管理する
    """
    logging.info("【実行モード】日単位でのTickデータ特徴量計算を開始します...")

    timeframe = "tick"
    PARTITION_ROOT = Path(config.partitioned_tick_path)
    FEATURES_ROOT = (
        Path(config.output_path)
        / "feature_value_a_vast_universeF"
        / f"features_{config.engine_id}_{timeframe}/"
    )
    FEATURES_ROOT.mkdir(parents=True, exist_ok=True)

    W_MAX = config.w_max

    calculation_engine = CalculationEngine(config)

    all_partitions = get_sorted_partitions(PARTITION_ROOT)

    # ===== ここから再開ロジックの変更箇所 =====
    if resume_date:
        import datetime

        # 指定された再開日以降のパーティションのみを対象とする
        all_days = [
            p
            for p in all_partitions
            if datetime.date(
                int(p.parent.parent.name.split("=")[1]),  # year
                int(p.parent.name.split("=")[1]),  # month
                int(p.name.split("=")[1]),  # day
            )
            >= resume_date
        ]
        logging.info(f"再開日 {resume_date} に基づいてフィルタリングしました。")
    else:
        all_days = all_partitions
    # ===== 再開ロジックここまで =====

    if not all_days:
        logging.error("処理対象の日次パーティションが見つかりません。")
        return

    # 日次パーティション逐次処理
    total_days = len(all_days)
    logging.info(f"処理対象日数: {total_days}日")

    for i, current_day_path in enumerate(all_days):
        day_name = f"{current_day_path.parent.parent.name}/{current_day_path.parent.name}/{current_day_path.name}"
        logging.info(f"=== 日次処理 ({i + 1}/{total_days}): {day_name} ===")

        try:
            # 前日のパーティション（オーバーラップ用）
            # 【重要】prev_day_pathの参照元を all_partitions に変更し、日付が連続していることを保証する
            current_index_in_all = all_partitions.index(current_day_path)
            prev_day_path = (
                all_partitions[current_index_in_all - 1]
                if current_index_in_all > 0
                else None
            )

            # オーバーラップを含む拡張データフレーム作成
            logging.info(f"データ読み込み開始: {day_name}")
            augmented_df, current_day_rows = create_augmented_frame(
                current_day_path, prev_day_path, W_MAX
            )
            logging.info(
                f"データ読み込み完了: 実データ{current_day_rows}行、総データ{augmented_df.height}行"
            )

            # ▼▼ 修正: 特徴量計算の直前に sample_weight を付与
            augmented_lf = calculation_engine.append_sample_weight(augmented_df.lazy())
            # ▲▲ 追加ここまで ▲▲

            # 【修正】親方（この関数）が工程管理を行う物理的垂直分割
            logging.info(f"特徴量計算開始: {day_name}")

            # 一時ディレクトリ作成（日次処理用）
            temp_dir = Path(tempfile.mkdtemp(prefix=f"day_{i:04d}_{config.engine_id}_"))
            logging.info(f"日次一時ディレクトリ作成: {temp_dir}")

            temp_files = []

            # 特徴量グループを取得
            feature_groups = calculation_engine.get_feature_groups()
            logging.info(f"物理的垂直分割: {len(feature_groups)}グループで処理")

            # 各グループを順次処理（親方が工程管理）
            for group_idx, (group_name, group_expressions) in enumerate(
                feature_groups.items()
            ):
                logging.info(
                    f"グループ処理開始: {group_name} ({len(group_expressions)}個の特徴量)"
                )

                # 1. 下請けに「このグループだけ計算しろ」と指示
                group_result_lf = calculation_engine.calculate_one_group(
                    augmented_lf, group_name, group_expressions, timeframe=timeframe
                )

                # 2. 親方が自らメモリに実現化（単一グループなので安全）
                # ▼▼ 修正前: group_result_df = group_result_lf.collect(streaming=True) # メモリ効率のためstreaming=Trueを追加
                # ▼▼ 修正後: 非推奨警告回避のため engine="streaming" に変更
                group_result_df = group_result_lf.collect(engine="streaming")
                logging.info(
                    f"グループデータ実現化: {group_result_df.height}行 x {group_result_df.width}列"
                )

                # 3. 親方が自らディスクに保存する
                temp_file = temp_dir / f"group_{group_idx:02d}_{group_name}.parquet"
                group_result_df.write_parquet(str(temp_file), compression="snappy")

                if temp_file.exists():
                    temp_files.append(temp_file)
                    logging.info(
                        f"グループ保存完了: {temp_file} ({temp_file.stat().st_size} bytes)"
                    )
                else:
                    raise FileNotFoundError(f"グループファイル作成失敗: {temp_file}")

                # メモリ使用量チェック
                memory_usage = calculation_engine.memory_monitor.get_memory_usage_gb()
                logging.info(f"メモリ使用量: {memory_usage:.2f}GB")

            # 4. 全グループ完了後、親方が最終組み立て（クリーン・オン・クリーン結合）
            logging.info("グループファイル結合開始（クリーン・オン・クリーン結合）...")

            # 1. 「クリーンな土台」を準備 (オーバーラップ除去済み)
            base_df = pl.read_parquet(str(temp_files[0]))

            if prev_day_path is not None:
                clean_base_df = base_df.tail(current_day_rows)
            else:
                clean_base_df = base_df

            logging.info(
                f"クリーンな土台を準備: {clean_base_df.height}行 x {clean_base_df.width}列"
            )

            # 2. 残りの一時ファイルを「クリーンなパーツ」として一つずつ結合
            base_columns = [
                "timestamp",
                "open",
                "high",
                "low",
                "close",
                "volume",
                "timeframe",
            ]
            # ▼▼ 修正: sample_weight もマージ時の重複スキップ対象に追加
            base_columns.append("sample_weight")
            # ▲▲ 追加ここまで ▲▲

            for idx, temp_file in enumerate(temp_files[1:], 1):
                next_df = pl.read_parquet(str(temp_file))

                # 「クリーンなパーツ」を作成 (オーバーラップ除去済み)
                if prev_day_path is not None:
                    clean_next_df = next_df.tail(current_day_rows)
                else:
                    clean_next_df = next_df

                # 行数が一致することを確認
                if clean_base_df.height != clean_next_df.height:
                    raise ValueError(
                        f"行数不一致: ベース{clean_base_df.height}行 vs 追加{clean_next_df.height}行"
                    )

                feature_cols = [
                    col for col in clean_next_df.columns if col not in base_columns
                ]
                if feature_cols:
                    clean_base_df = clean_base_df.hstack(
                        clean_next_df.select(feature_cols)
                    )

            result_df = clean_base_df
            logging.info(
                f"全グループ結合完了: {result_df.height}行 x {result_df.width}列"
            )

            # パーティション保存用の日付列を追加
            final_df = result_df.with_columns(
                [
                    pl.col("timestamp").cast(pl.Datetime("us")),
                    pl.col("timeframe").cast(pl.Utf8),
                    pl.col("timestamp").dt.year().cast(pl.Int32).alias("year"),
                    pl.col("timestamp").dt.month().cast(pl.Int32).alias("month"),
                    pl.col("timestamp").dt.day().cast(pl.Int32).alias("day"),
                ]
            )

            # 当日の結果を保存
            logging.info(f"最終保存開始: {day_name}")
            final_df.write_parquet(FEATURES_ROOT, partition_by=["year", "month", "day"])

            logging.info(f"保存完了: {day_name} - {final_df.height}行の特徴量データ")

            # 一時ディレクトリクリーンアップ
            for temp_file in temp_files:
                if temp_file.exists():
                    temp_file.unlink()
            temp_dir.rmdir()
            logging.info(f"一時ディレクトリクリーンアップ完了: {temp_dir}")

        except Exception as e:
            logging.error(f"日次処理エラー ({day_name}): {e}", exc_info=True)
            continue


# 通常timeframe処理
def process_single_timeframe(config: ProcessingConfig, timeframe: str):
    """単一の通常時間足を処理する（修正版ロジック）"""
    logger.info(f"=== 通常処理開始: timeframe={timeframe} ===")
    start_time = time.time()

    calc_engine = None
    try:
        data_engine = DataEngine(config)
        calc_engine = CalculationEngine(config)
        output_engine = OutputEngine(config)

        lazy_frame = data_engine.create_lazy_frame(timeframe)
        summary = data_engine.get_data_summary(lazy_frame)
        logger.info(f"データサマリー: {summary}")

        lazy_frame = calc_engine.append_sample_weight(lazy_frame)

        # ▼▼ 修正前: features_lf = lazy_frame.with_columns(all_expressions) （等の一括計算）
        # ▼▼ 修正後: ディスクベースの物理的垂直分割とgc.collect()を用いた完全置換
        import gc

        feature_groups = calc_engine.get_feature_groups()
        temp_dir = calc_engine.temp_dir / f"tf_{timeframe}"
        temp_dir.mkdir(parents=True, exist_ok=True)
        temp_files = []

        base_df = lazy_frame.collect()

        for group_idx, (group_name, group_expressions) in enumerate(
            feature_groups.items()
        ):
            logger.info(
                f"グループ処理開始: {group_name} ({len(group_expressions)}個の特徴量)"
            )

            group_result_lf = calc_engine.calculate_one_group(
                base_df.lazy(), group_name, group_expressions, timeframe=timeframe
            )

            group_result_df = group_result_lf.collect(engine="streaming")

            temp_file = temp_dir / f"group_{group_idx:02d}_{group_name}.parquet"
            group_result_df.write_parquet(str(temp_file), compression="snappy")
            temp_files.append(temp_file)

            del group_result_df
            gc.collect()

        logger.info("全グループ計算完了。ファイルの結合を開始します...")
        result_df = pl.read_parquet(str(temp_files[0]))
        # ※Engine 1F固有: sample_weightの重複結合を防ぐため base_columns に追加
        base_columns = [
            "timestamp",
            "open",
            "high",
            "low",
            "close",
            "volume",
            "timeframe",
            "sample_weight",
        ]

        for temp_file in temp_files[1:]:
            next_df = pl.read_parquet(str(temp_file))
            feature_cols = [col for col in next_df.columns if col not in base_columns]
            if feature_cols:
                result_df = result_df.hstack(next_df.select(feature_cols))
            del next_df
            gc.collect()

        features_lf = result_df.lazy()
        # ▲▲ 修正ここまで ▲▲

        processed_lf = output_engine.apply_nan_filling(features_lf)
        metadata = output_engine.save_features(processed_lf, timeframe)

        elapsed_time = time.time() - start_time
        metadata["processing_time"] = elapsed_time

        logger.info(f"=== 通常処理完了: {timeframe} - {elapsed_time:.2f}秒 ===")
        return metadata

    except Exception as e:
        logger.error(f"タイムフレーム {timeframe} の処理中にエラー: {e}", exc_info=True)
        return {"timeframe": timeframe, "error": str(e)}
    finally:
        # CalculationEngineが一時ディレクトリを作成した場合に備えてクリーンアップ
        if calc_engine and hasattr(calc_engine, "_cleanup_temp_files"):
            calc_engine._cleanup_temp_files()


# インタラクティブモード（Project Forge準拠）
def get_user_confirmation(config: ProcessingConfig) -> bool:
    """ユーザー確認 - Project Forge理念表示"""
    print("\n" + "=" * 60)
    print(f"Engine {config.engine_id.upper()} - {config.engine_name}")
    print("=" * 60)
    print("🎯 Project Forge - 軍資金増大プロジェクト")
    print("🚀 最終目標: Project Chimera開発・完成のための資金調達")
    print("💻 探索対象: マーケットの亡霊（統計的に有意で非ランダムな微細パターン）")
    print("🏅 思想的継承: ジム・シモンズ（ルネサンス・テクノロジーズ）")
    print("=" * 60)
    print(f"入力パス: {config.input_path}")
    print(f"出力パス: {config.output_path}")
    print(f"Tickパーティションパス: {config.partitioned_tick_path}")
    print(f"エンジンID: {config.engine_id}")
    print(f"並列スレッド数: {config.max_threads}")
    print(f"メモリ制限: {config.memory_limit_gb}GB")

    if config.test_mode:
        print(f"\n【テストモード】最初の{config.test_rows}行のみ処理")

    print(f"\n処理対象タイムフレーム ({len(config.timeframes)}個):")
    for i, tf in enumerate(config.timeframes):
        print(f"  {i + 1:2d}. {tf}")

    print("\n処理内容:")
    print(
        "  - 学際的・実験的特徴量（ネットワーク科学、言語学、美学、音楽理論、生体力学）"
    )
    print("  - 2段階品質保証システム")
    print("  - 【修正】物理的垂直分割（ディスクベース中間ファイル）")
    print("  - Polars LazyFrame + Numba JITハイブリッド最適化")

    response = input("\n処理を開始しますか？ (y/n): ")
    return response.lower() == "y"


def select_timeframes(config: ProcessingConfig) -> List[str]:
    """タイムフレーム選択（完全同一実装）"""
    print("\nタイムフレームを選択してください:")
    print("  0. 全て処理")

    all_timeframes = config.timeframes

    for i, tf in enumerate(all_timeframes):
        print(f"  {i + 1:2d}. {tf}")

    print("  (例: 1,3,5 または 1-5 カンマ区切り)")

    selection = input("選択: ").strip()

    if selection == "0" or selection == "":
        return all_timeframes

    selected_indices = set()
    try:
        parts = selection.split(",")
        for part in parts:
            if "-" in part:
                start, end = map(int, part.strip().split("-"))
                selected_indices.update(range(start - 1, end))
            else:
                selected_indices.add(int(part.strip()) - 1)

        return [
            all_timeframes[i]
            for i in sorted(list(selected_indices))
            if 0 <= i < len(all_timeframes)
        ]
    except Exception as e:
        logger.warning(f"選択エラー: {e} - 全タイムフレームを処理します")
        return all_timeframes


def main():
    """メイン実行関数 - Project Forge統合版"""
    print("\n" + "=" * 70)
    print(f"  Engine 1F - Interdisciplinary Features ")
    print("  Project Forge - 軍資金増大プロジェクト  ")
    print("=" * 70)
    print("🎯 目標: XAU/USD市場の学際的パターン抽出")
    print("🤖 AI頭脳による学際融合法則発見")
    print("💰 Project Chimera開発資金調達")
    print("🔧 【修正】物理的垂直分割によるメモリ・スラッシング回避")
    print("=" * 70)

    config = ProcessingConfig()

    if not config.validate():
        return 1

    data_engine = DataEngine(config)

    if not data_engine.validate_data_source():
        return 1

    # ===== ここから対話機能の変更箇所 =====
    import datetime

    resume_date = None
    print("\n実行タイプを選択してください:")
    print("  1. 新規実行")
    print("  2. 中断した処理を再開")
    run_type_selection = input("選択 (1/2): ").strip()

    if run_type_selection == "2":
        while True:
            date_str = input(
                "再開する日付を入力してください (例: 2025-01-01): "
            ).strip()
            try:
                # 文字列をdatetimeオブジェクトに変換し、date部分のみを取得
                resume_date = datetime.datetime.strptime(date_str, "%Y-%m-%d").date()
                print(f"{resume_date} から処理を再開します。")
                break
            except ValueError:
                print(
                    "エラー: 日付の形式が正しくありません。YYYY-MM-DD形式で入力してください。"
                )
    else:
        print("新規に処理を開始します。")

    # Project Forge準拠の対話設定
    print("\n並列処理スレッド数を選択してください:")
    print("  1. 自動設定 (推奨)")
    print("  2. 手動設定")

    thread_selection = input("選択 (1/2): ").strip()

    if thread_selection == "2":
        try:
            max_threads = int(
                input(f"スレッド数を入力 (1-{psutil.cpu_count()}): ").strip()
            )
            if 1 <= max_threads <= psutil.cpu_count():
                config.max_threads = max_threads
                print(f"スレッド数設定: {max_threads}")
            else:
                print("無効な値です。デフォルト値を使用します。")
        except ValueError:
            print("無効な入力です。デフォルト値を使用します。")

    print("\n出力パスを選択してください:")
    print(f"  1. デフォルト ({config.output_path})")
    print("  2. カスタムパス")

    path_selection = input("選択 (1/2): ").strip()

    if path_selection == "2":
        custom_path = input("出力パスを入力: ").strip()
        if custom_path:
            config.output_path = custom_path
            print(f"出力パス設定: {custom_path}")

    print("\nメモリ制限を選択してください:")
    print(f"  1. デフォルト ({config.memory_limit_gb}GB)")
    print("  2. カスタム設定")

    memory_selection = input("選択 (1/2): ").strip()

    if memory_selection == "2":
        try:
            memory_limit = float(input("メモリ制限 (GB): ").strip())
            if memory_limit > 0:
                config.memory_limit_gb = memory_limit
                config.memory_warning_gb = memory_limit * 0.9
                print(f"メモリ制限設定: {memory_limit}GB")
            else:
                print("無効な値です。デフォルト値を使用します。")
        except ValueError:
            print("無効な入力です。デフォルト値を使用します。")

    print("\n実行モードを選択してください:")
    print("  1. テストモード（少量データで動作確認）")
    print("  2. 本格モード（全データ処理）")

    mode_selection = input("選択 (1/2): ").strip()

    if mode_selection == "1":
        config.test_mode = True
        try:
            test_rows = int(
                input(f"テスト行数 (デフォルト: {config.test_rows}): ").strip()
                or str(config.test_rows)
            )
            config.test_rows = test_rows
            print(f"テストモード設定: 最初の{config.test_rows}行を処理")
        except ValueError:
            print(f"無効な入力です。デフォルト値 ({config.test_rows}) を使用します。")

    selected_timeframes = select_timeframes(config)
    config.timeframes = selected_timeframes

    if not get_user_confirmation(config):
        print("処理をキャンセルしました")
        return 0

    os.environ["POLARS_MAX_THREADS"] = str(config.max_threads)
    logger.info(f"並列処理スレッド数: {config.max_threads}")

    print("\n" + "=" * 60)
    print("処理開始...")
    print("=" * 60)

    overall_start_time = time.time()

    if "tick" in selected_timeframes:
        run_on_partitions_mode(config, resume_date=resume_date)

    other_timeframes = [tf for tf in selected_timeframes if tf != "tick"]
    if other_timeframes:
        for tf in other_timeframes:
            process_single_timeframe(config, tf)

    overall_elapsed_time = time.time() - overall_start_time

    print(
        f"\n全ての要求された処理が完了しました。総処理時間: {overall_elapsed_time:.2f}秒"
    )
    return 0


if __name__ == "__main__":
    try:
        exit_code = main()
        sys.exit(exit_code)
    except Exception as e:
        logger.critical(
            f"スクリプト実行中に致命的なエラーが発生しました: {e}", exc_info=True
        )
        sys.exit(1)

# ================================
# BLOCK 3 完了 - 全スクリプト完成
# ================================
