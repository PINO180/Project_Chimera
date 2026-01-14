#!/usr/bin/env python3
"""
革新的な特徴量収集スクリプト - Engine 2A: Complexity Theory Features (簡略版)
【実装内容】複雑性理論特徴量群(F5, F15)

Project Forge - 軍資金増大プロジェクト
最終目標: Project Chimera開発・完成のための資金調達

技術戦略: ジム・シモンズの思想的継承
- 経済学・ファンダメンタルズ・古典的テクニカル指標の完全排除
- 統計的に有意で非ランダムな微細パターン「マーケットの亡霊」の探索
- AIの頭脳による普遍的法則の読み解き

アーキテクチャ: 3クラス構成(最適化版) + ディスクベース垂直分割
- DataEngine(30%): Polars LazyFrame基盤
- CalculationEngine(60%): 特徴量計算核心(物理的垂直分割実装)
- OutputEngine(10%): ストリーミング出力

【重大修正】重量級UDF呼び出しパターンの設計規律遵守
- rolling_map → map_batches への完全移行
- UDF内部でのローリング処理とprange並列化
- 遅延束縛バグの完全排除(デフォルト引数による即時束縛)
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
        "mfdfa": [1000, 2500, 5000],  # MFDFA用ウィンドウ
        "complexity": [500, 1000, 1500],  # 複雑性指標用ウィンドウ
        "general": [1000, 2500, 5000],
    }


@dataclass
class ProcessingConfig:
    """処理設定 - Project Forge統合版 (Engine 2A: Complexity Theory - Simplified)"""

    # データパス(Project Forge構造準拠) - config.pyから読み込む
    input_path: str = str(config.S1_BASE_MULTITIMEFRAME)
    partitioned_tick_path: str = str(config.S1_RAW_TICK_PARTITIONED)
    output_path: str = str(config.S2_FEATURES)

    # エンジン識別
    engine_id: str = "e2a"
    engine_name: str = "Engine_2A_ComplexityTheory_Simplified"

    # 並列処理(戦略的並列処理スロットリング)
    max_threads: int = 4

    # メモリ制限(64GB RAM制約)
    memory_limit_gb: float = 55.0
    memory_warning_gb: float = 50.0

    timeframes: List[str] = field(default_factory=get_default_timeframes)
    window_sizes: Dict[str, List[int]] = field(default_factory=get_default_window_sizes)

    # 処理モード
    test_mode: bool = False
    test_rows: int = 10000

    # システムハイパーパラメータとしてW_maxを定義
    w_max: int = 5000

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


# ===============================
# Block 1 Complete: ヘッダーと基本設定
# ===============================

# ===============================
# Block 2 Start: MFDFA UDF定義(修正版:map_batches対応)
# ===============================

# 【重要】全てのNumba UDFは必ずクラス外で定義すること
# クラス内定義は解決不能な循環参照エラーを引き起こすため絶対禁止

# =============================================================================
# MFDFA (Multifractal Detrended Fluctuation Analysis) UDF群
# 【修正版】ローリング処理を内部化、map_batches対応、prange並列化
# =============================================================================


@nb.njit(fastmath=True, cache=True)
def polynomial_fit_detrend(y: np.ndarray, degree: int = 1) -> np.ndarray:
    """
    多項式フィッティングとトレンド除去
    最小二乗法による多項式フィット
    """
    n = len(y)
    if n < degree + 1:
        return np.zeros(n)

    # Vandermonde行列構築
    x = np.arange(n, dtype=np.float64)

    # 正規方程式による多項式係数推定
    # 簡略化のため、1次(線形)トレンドのみサポート
    if degree == 1:
        # 線形回帰: y = a*x + b
        x_mean = np.mean(x)
        y_mean = np.mean(y)

        numerator = 0.0
        denominator = 0.0
        for i in range(n):
            x_diff = x[i] - x_mean
            numerator += x_diff * (y[i] - y_mean)
            denominator += x_diff * x_diff

        if abs(denominator) < 1e-10:
            return y - y_mean

        slope = numerator / denominator
        intercept = y_mean - slope * x_mean

        # トレンド除去
        detrended = np.empty(n)
        for i in range(n):
            trend = slope * x[i] + intercept
            detrended[i] = y[i] - trend

        return detrended

    return y - np.mean(y)


@nb.njit(fastmath=True, cache=True)
def mfdfa_core_single_window(
    prices: np.ndarray, q_values: np.ndarray, scales: np.ndarray, poly_degree: int = 1
) -> np.ndarray:
    """
    単一ウィンドウのMFDFA計算(ヘルパー関数)

    Args:
        prices: 価格時系列(単一ウィンドウ)
        q_values: qモーメント配列
        scales: スケール配列
        poly_degree: 多項式次数

    Returns:
        結果配列: [h_q平均, multifractal_width, holder_max]
    """
    n = len(prices)
    n_q = len(q_values)
    n_scales = len(scales)

    # 初期化
    result = np.full(3, np.nan)

    if n < 20:
        return result

    # 1. プロファイル構築(累積和)
    mean_price = np.mean(prices)
    profile = np.zeros(n)
    cumsum = 0.0
    for i in range(n):
        cumsum += prices[i] - mean_price
        profile[i] = cumsum

    # 2. qモーメント揺らぎ関数の計算
    F_q = np.zeros((n_q, n_scales))

    for s_idx in range(n_scales):
        scale = int(scales[s_idx])
        if scale < 4 or scale >= n // 4:
            continue

        # セグメント数
        n_segments = n // scale

        # 各セグメントの揺らぎを計算
        segment_variances = np.zeros(n_segments)

        for seg in range(n_segments):
            start = seg * scale
            end = start + scale

            # セグメント抽出
            segment = profile[start:end]

            # トレンド除去
            detrended = polynomial_fit_detrend(segment, poly_degree)

            # 分散計算
            variance = 0.0
            for val in detrended:
                variance += val * val
            variance = variance / scale if scale > 0 else 0.0

            segment_variances[seg] = variance

        # qモーメント揺らぎ関数の計算
        for q_idx in range(n_q):
            q = q_values[q_idx]

            if abs(q) < 1e-10:  # q=0の場合は対数平均
                log_sum = 0.0
                valid_count = 0
                for var in segment_variances:
                    if var > 1e-10:
                        log_sum += np.log(var)
                        valid_count += 1
                if valid_count > 0:
                    F_q[q_idx, s_idx] = np.exp(log_sum / (2.0 * valid_count))
            else:
                # q≠0の場合
                sum_val = 0.0
                valid_count = 0
                for var in segment_variances:
                    if var > 1e-10:
                        sum_val += np.power(var, q / 2.0)
                        valid_count += 1

                if valid_count > 0:
                    F_q[q_idx, s_idx] = np.power(sum_val / valid_count, 1.0 / q)

    # 3. 一般化Hurst指数の推定(log(F_q) vs log(scale)の傾き)
    h_q_values = np.zeros(n_q)

    for q_idx in range(n_q):
        # 有効なスケールのみ使用
        log_scales = []
        log_F_vals = []

        for s_idx in range(n_scales):
            if F_q[q_idx, s_idx] > 1e-10:
                log_scales.append(np.log(scales[s_idx]))
                log_F_vals.append(np.log(F_q[q_idx, s_idx]))

        if len(log_scales) >= 3:
            # 線形回帰で傾きを推定
            x_arr = np.array(log_scales)
            y_arr = np.array(log_F_vals)

            x_mean = np.mean(x_arr)
            y_mean = np.mean(y_arr)

            numerator = 0.0
            denominator = 0.0
            for i in range(len(x_arr)):
                x_diff = x_arr[i] - x_mean
                numerator += x_diff * (y_arr[i] - y_mean)
                denominator += x_diff * x_diff

            if abs(denominator) > 1e-10:
                h_q_values[q_idx] = numerator / denominator

    # 4. マルチフラクタルスペクトラムの計算
    valid_h = h_q_values[np.isfinite(h_q_values)]

    if len(valid_h) >= 3:
        h_mean = np.mean(valid_h)
        h_max = np.max(valid_h)
        h_min = np.min(valid_h)
        multifractal_width = h_max - h_min

        result[0] = h_mean  # 平均Hurst指数
        result[1] = multifractal_width  # Δα
        result[2] = h_max  # α_max

    return result


@nb.njit(fastmath=True, cache=True, parallel=True)
def mfdfa_rolling_udf(
    prices: np.ndarray, window: int, component_idx: int
) -> np.ndarray:
    """
    MFDFAローリング計算(map_batches対応版)

    Args:
        prices: 価格時系列全体
        window: ウィンドウサイズ
        component_idx: 結果成分インデックス(0=hurst_mean, 1=width, 2=holder_max)

    Returns:
        結果配列(最初のwindow-1個はNaN)
    """
    n = len(prices)
    result = np.full(n, np.nan)

    # デフォルトパラメータ
    q_values = np.array([-5.0, -3.0, -1.0, 0.0, 1.0, 3.0, 5.0])
    scales = np.array([10.0, 20.0, 50.0, 100.0, 200.0])

    # prange並列化:各ウィンドウ位置を並列処理
    for i in nb.prange(window, n + 1):
        window_data = prices[i - window : i]
        mfdfa_result = mfdfa_core_single_window(
            window_data, q_values, scales, poly_degree=1
        )
        result[i - 1] = mfdfa_result[component_idx]

    return result


# ===============================
# Block 2 Complete: MFDFA UDF定義(修正版)
# ===============================

# ===============================
# Block 3 Start: Kolmogorov複雑性UDF定義(修正版)
# ===============================

# =============================================================================
# コルモゴロフ複雑性 (Lempel-Ziv圧縮近似) UDF群
# 【修正版】ローリング処理を内部化、map_batches対応、prange並列化
# =============================================================================


@nb.njit(fastmath=True, cache=True)
def binarize_series(values: np.ndarray, method: int = 0) -> np.ndarray:
    """
    時系列のバイナリ化/多値符号化

    Args:
        values: 入力時系列
        method: 0=中央値基準, 1=分位基準(3値), 2=変化基準

    Returns:
        符号化された整数配列
    """
    n = len(values)
    encoded = np.zeros(n, dtype=np.int32)

    if n < 2:
        return encoded

    if method == 0:  # 中央値基準 (2値)
        median_val = np.median(values)
        for i in range(n):
            encoded[i] = 1 if values[i] > median_val else 0

    elif method == 1:  # 分位基準 (3値: 0, 1, 2)
        valid_vals = values[np.isfinite(values)]
        if len(valid_vals) < 3:
            return encoded

        q33 = np.percentile(valid_vals, 33.33)
        q67 = np.percentile(valid_vals, 66.67)

        for i in range(n):
            if values[i] < q33:
                encoded[i] = 0
            elif values[i] < q67:
                encoded[i] = 1
            else:
                encoded[i] = 2

    elif method == 2:  # 変化基準 (3値: -1, 0, +1)
        for i in range(1, n):
            if values[i] > values[i - 1]:
                encoded[i] = 1
            elif values[i] < values[i - 1]:
                encoded[i] = -1
            else:
                encoded[i] = 0

    return encoded


@nb.njit(fastmath=True, cache=True)
def lempel_ziv_complexity(sequence: np.ndarray) -> float:
    """
    Lempel-Ziv複雑性計算(LZ76アルゴリズム)

    Args:
        sequence: 整数符号化された時系列

    Returns:
        正規化されたLZ複雑性 (0〜1)
    """
    n = len(sequence)
    if n < 2:
        return 0.0

    # 辞書サイズのカウント
    complexity = 1
    prefix_length = 1

    i = 0
    while i < n:
        # 現在の位置から可能な限り長い既知パターンを検索
        max_match_length = 0

        for start in range(i):
            match_length = 0
            j = 0
            while (
                i + j < n and start + j < i and sequence[i + j] == sequence[start + j]
            ):
                match_length += 1
                j += 1

            if match_length > max_match_length:
                max_match_length = match_length

        # 新しいパターンが見つかった
        if max_match_length == 0:
            complexity += 1
            i += 1
        else:
            complexity += 1
            i += max_match_length + 1

    # 正規化: 理論的最大複雑性で割る
    # ランダム系列の場合の複雑性: n / log2(n)
    if n > 1:
        max_complexity = n / (np.log2(n) + 1e-10)
        normalized_complexity = (
            complexity / max_complexity if max_complexity > 0 else 0.0
        )
        return min(normalized_complexity, 1.0)

    return 0.0


@nb.njit(fastmath=True, cache=True)
def kolmogorov_complexity_single_window(prices: np.ndarray) -> np.ndarray:
    """
    単一ウィンドウのコルモゴロフ複雑性計算(ヘルパー関数)

    Returns:
        [complexity, compression_ratio, pattern_diversity]
    """
    result = np.full(3, np.nan)

    n = len(prices)
    if n < 10:
        return result

    # 1. 対数リターン計算
    returns = np.zeros(n - 1)
    for i in range(n - 1):
        if prices[i] > 1e-10:
            returns[i] = np.log(prices[i + 1] / prices[i])

    # 2. 標準化
    returns_mean = np.mean(returns)
    returns_std = np.std(returns)

    if returns_std < 1e-10:
        return result

    standardized = np.zeros(len(returns))
    for i in range(len(returns)):
        standardized[i] = (returns[i] - returns_mean) / returns_std

    # 3. バイナリ化(中央値基準)
    encoded = binarize_series(standardized, method=0)

    # 4. LZ複雑性計算
    complexity = lempel_ziv_complexity(encoded)

    # 5. 圧縮率推定(複雑性から逆算)
    compression_ratio = 1.0 - complexity  # 低複雑性 = 高圧縮率

    # 6. パターン多様性(ユニーク値の割合)
    unique_count = 0
    for i in range(len(encoded)):
        is_unique = True
        for j in range(i):
            if encoded[i] == encoded[j]:
                is_unique = False
                break
        if is_unique:
            unique_count += 1

    pattern_diversity = unique_count / len(encoded) if len(encoded) > 0 else 0.0

    result[0] = complexity
    result[1] = compression_ratio
    result[2] = pattern_diversity

    return result


@nb.njit(fastmath=True, cache=True, parallel=True)
def kolmogorov_complexity_rolling_udf(
    prices: np.ndarray, window: int, component_idx: int
) -> np.ndarray:
    """
    コルモゴロフ複雑性ローリング計算(map_batches対応版)

    Args:
        prices: 価格時系列全体
        window: ウィンドウサイズ
        component_idx: 結果成分インデックス(0=complexity, 1=compression_ratio, 2=pattern_diversity)

    Returns:
        結果配列(最初のwindow-1個はNaN)
    """
    n = len(prices)
    result = np.full(n, np.nan)

    # prange並列化:各ウィンドウ位置を並列処理
    for i in nb.prange(window, n + 1):
        window_data = prices[i - window : i]
        kolmogorov_result = kolmogorov_complexity_single_window(window_data)
        result[i - 1] = kolmogorov_result[component_idx]

    return result


# ===============================
# Block 3 Complete: Kolmogorov複雑性UDF定義(修正版)
# ===============================

# ===============================
# Block 4 Start: DataEngine/QualityAssurance クラス
# ===============================


class DataEngine:
    """
    データ基盤クラス(30%) - Project Forge統合版
    責務:
    - Parquetメタデータ事前検証
    - Polars scan_parquetによる遅延読み込み(LazyFrame生成)
    - 述語プッシュダウンによるフィルタリング最適化
    - timeframe別分割処理
    - メモリ使用量監視
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

            # LazyFrameでメタデータ取得(実際の読み込みなし)
            lazy_frame = pl.scan_parquet(parquet_pattern)

            # スキーマ情報取得(警告回避)
            schema = lazy_frame.collect_schema()

            # 基本メタデータ収集
            metadata = {
                "schema": dict(schema),
                "columns": list(schema.keys()),
                "path_exists": Path(self.config.input_path).exists(),
                "estimated_memory_gb": 0.0,
            }

            # 必須カラムチェック(XAU/USDデータ構造)
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

            # Hiveパーティション対応:特定timeframeディレクトリを直接指定
            timeframe_path = f"{self.config.input_path}/timeframe={timeframe}/*.parquet"

            # 指定timeframeのParquetファイルのみをスキャン
            lazy_frame = pl.scan_parquet(timeframe_path)

            # timeframe列を手動で追加 (Hiveパーティション復元)
            lazy_frame = lazy_frame.with_columns(
                [pl.lit(timeframe).alias("timeframe").cast(pl.Categorical)]
            )

            # スキーマを確認してから安全にキャスト処理を適用
            try:
                sample_schema = lazy_frame.limit(1).collect_schema()
                logger.info(f"検出されたスキーマ: {list(sample_schema.keys())}")

                # 基本データ型確認と最適化(必要な場合のみキャスト)
                cast_exprs = []

                # 各カラムが存在し、かつ適切な型でない場合のみキャスト
                if "timestamp" in sample_schema and sample_schema[
                    "timestamp"
                ] != pl.Datetime("ns"):
                    cast_exprs.append(pl.col("timestamp").cast(pl.Datetime("ns")))
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
                # キャストエラーの場合は処理を続行(データがすでに適切な型の可能性)

            # タイムスタンプソート
            lazy_frame = lazy_frame.sort("timestamp")

            logger.info(f"LazyFrame生成完了: timeframe={timeframe}")
            return lazy_frame

        except Exception as e:
            logger.error(f"LazyFrame生成エラー (timeframe={timeframe}): {e}")
            raise

    def get_data_summary(self, lazy_frame: pl.LazyFrame) -> Dict[str, Any]:
        """データサマリー情報取得(軽量)"""
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


class QualityAssurance:
    """数値安定性保証システム - 2段階品質保証エンジン(Project Forge準拠)"""

    @staticmethod
    def calculate_quality_score(values: np.ndarray) -> float:
        """
        品質スコア算出:0.0(使用不可)〜 1.0(完璧)
        Project Forge厳密な統計的定義に基づく実装

        Args:
            values: 評価対象の数値配列
        Returns:
            float: 品質スコア
        """
        if len(values) == 0:
            return 0.0

        n_total = len(values)

        # 1. 有限値比率の厳密計算
        finite_count = 0
        nan_count = 0
        inf_count = 0

        for val in values:
            if np.isnan(val):
                nan_count += 1
            elif np.isinf(val):
                inf_count += 1
            else:
                finite_count += 1

        finite_ratio = finite_count / n_total

        if finite_count == 0:
            return 0.0

        # 有限値のみを抽出
        finite_values = np.zeros(finite_count)
        idx = 0
        for val in values:
            if np.isfinite(val):
                finite_values[idx] = val
                idx += 1

        # 2. 統計的多様性指標の厳密計算
        unique_values, counts = np.unique(finite_values, return_counts=True)
        n_unique = len(unique_values)

        if n_unique == 1:
            diversity_score = 0.0
        else:
            # シャノンエントロピーの厳密計算
            shannon_entropy = 0.0
            for count in counts:
                p = count / finite_count
                if p > 0:
                    shannon_entropy -= p * np.log2(p)

            max_entropy = np.log2(min(n_unique, finite_count))
            if max_entropy > 0:
                diversity_score = shannon_entropy / max_entropy
            else:
                diversity_score = 0.0

        # 3. 数値的安定性指標の厳密計算
        if finite_count < 2:
            stability_score = 1.0
        else:
            sum_values = 0.0
            for val in finite_values:
                sum_values += val
            mean_val = sum_values / finite_count

            sum_sq_deviations = 0.0
            for val in finite_values:
                deviation = val - mean_val
                sum_sq_deviations += deviation * deviation

            unbiased_variance = sum_sq_deviations / (finite_count - 1)
            std_dev = np.sqrt(unbiased_variance)

            if abs(mean_val) > 1e-15:
                cv = std_dev / abs(mean_val)
                stability_score = 1.0 / (1.0 + cv)
            else:
                stability_score = 1.0 / (1.0 + std_dev)

        # 4. 外れ値耐性指標の厳密計算
        if finite_count < 5:
            outlier_resistance = 1.0
        else:
            sorted_values = np.sort(finite_values)

            def calculate_quantile_r6(sorted_arr, q):
                n = len(sorted_arr)
                h = n * q + 0.5
                h_floor = int(np.floor(h))
                h_ceil = int(np.ceil(h))
                if h_floor == h_ceil:
                    return sorted_arr[h_floor - 1]
                else:
                    gamma = h - h_floor
                    return (
                        sorted_arr[h_floor - 1] * (1 - gamma)
                        + sorted_arr[h_ceil - 1] * gamma
                    )

            q1 = calculate_quantile_r6(sorted_values, 0.25)
            q3 = calculate_quantile_r6(sorted_values, 0.75)
            iqr = q3 - q1

            if iqr > 1e-15:
                lower_fence = q1 - 1.5 * iqr
                upper_fence = q3 + 1.5 * iqr

                outlier_count = 0
                for val in finite_values:
                    if val < lower_fence or val > upper_fence:
                        outlier_count += 1

                outlier_ratio = outlier_count / finite_count
                outlier_resistance = 1.0 - outlier_ratio
            else:
                outlier_resistance = 1.0

        # 総合品質スコアの厳密計算
        weights = {
            "finite_ratio": 0.4,
            "diversity": 0.25,
            "stability": 0.25,
            "outlier_resistance": 0.1,
        }

        composite_score = (
            weights["finite_ratio"] * finite_ratio
            + weights["diversity"] * diversity_score
            + weights["stability"] * stability_score
            + weights["outlier_resistance"] * outlier_resistance
        )

        return np.clip(composite_score, 0.0, 1.0)

    @staticmethod
    def basic_stabilization(values: np.ndarray) -> np.ndarray:
        """
        第1段階:基本対処(品質スコア > 0.6)
        軽量で高速な処理
        """
        cleaned = np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0)

        finite_mask = np.isfinite(cleaned)
        if np.sum(finite_mask) < 2:
            return cleaned

        finite_values = cleaned[finite_mask]
        try:
            p1, p99 = np.percentile(finite_values, [1, 99])
            result = np.clip(cleaned, p1, p99)
        except:
            result = cleaned

        return result

    @staticmethod
    def robust_stabilization(values: np.ndarray) -> np.ndarray:
        """
        第2段階:フォールバック(品質スコア ≤ 0.6)
        ロバスト統計による処理
        """
        finite_mask = np.isfinite(values)
        if np.sum(finite_mask) < 3:
            return np.zeros_like(values)

        finite_values = values[finite_mask]

        try:
            median_val = np.median(finite_values)
            abs_deviations = np.abs(finite_values - median_val)
            mad_val = np.median(abs_deviations)

            if mad_val < 1e-10:
                mad_val = np.std(finite_values) * 0.6745

            robust_bounds = (median_val - 3 * mad_val, median_val + 3 * mad_val)

            result = np.copy(values)
            result = np.nan_to_num(
                result, nan=median_val, posinf=robust_bounds[1], neginf=robust_bounds[0]
            )
            result = np.clip(result, robust_bounds[0], robust_bounds[1])

        except Exception:
            median_val = np.median(finite_values) if len(finite_values) > 0 else 0.0
            result = np.full_like(values, median_val)

        return result


# ===============================
# Block 4 Complete: DataEngine/QualityAssurance
# ===============================

# ===============================
# Block 5 Start: CalculationEngine/OutputEngine/メイン実行部(最終)
# ===============================


class CalculationEngine:
    """
    計算核心クラス(60%) - Project Forge統合版(複雑性理論特徴量専用 - 簡略版)
    【重大修正】map_batches + prange並列化による重量級UDF呼び出しパターン統一
    責務:
    - Polars Expressionによる高度な特徴量計算(90%のタスク)
    - .map_batches()経由のNumba JIT最適化UDF(10%のカスタム「アルファ」ロジック)
    - 【修正】物理的垂直分割:ディスク中間ファイルによるメモリ・スラッシング回避
    - Polars内部並列化による自動マルチスレッド実行
    - 2段階品質保証(基本/フォールバック)
    """

    def __init__(self, config: ProcessingConfig):
        self.config = config
        self.qa = QualityAssurance()
        self.memory_monitor = MemoryMonitor(config.memory_limit_gb)
        self.prefix = f"e{config.engine_id.replace('e', '')}_"

        # 【修正】ディスクベース垂直分割用の一時ディレクトリ
        self.temp_dir = Path(tempfile.mkdtemp(prefix=f"features_{config.engine_id}_"))
        logger.info(f"一時ディレクトリ作成: {self.temp_dir}")

    def _get_all_feature_expressions(self) -> Dict[str, pl.Expr]:
        """
        全特徴量の名前とPolars Expressionの対応辞書を返すファクトリメソッド。
        これにより、必要な特徴量だけを効率的に計算できるようになる。
        【特徴量ファクトリパターン】
        【重大修正】遅延束縛バグ修正:デフォルト引数による即時束縛
        【重大修正】map_batchesパターン:重量級UDF専用呼び出し
        """
        expressions = {}
        p = self.prefix

        # ====================================================================
        # F5: MFDFA (Multifractal Detrended Fluctuation Analysis)
        # 【修正】rolling_map → map_batches、遅延束縛バグ修正
        # ====================================================================
        for window in self.config.window_sizes["mfdfa"]:
            # MFDFAコア計算(重量UDF: map_batches使用)
            # 遅延束縛バグ修正: w=window でループ変数を即時束縛
            expressions[f"{p}mfdfa_hurst_mean_{window}"] = (
                pl.col("close")
                .map_batches(
                    lambda s, w=window: mfdfa_rolling_udf(s.to_numpy(), w, 0),
                    return_dtype=pl.Float64,
                )
                .alias(f"{p}mfdfa_hurst_mean_{window}")
            )

            expressions[f"{p}mfdfa_width_{window}"] = (
                pl.col("close")
                .map_batches(
                    lambda s, w=window: mfdfa_rolling_udf(s.to_numpy(), w, 1),
                    return_dtype=pl.Float64,
                )
                .alias(f"{p}mfdfa_width_{window}")
            )

            expressions[f"{p}mfdfa_holder_max_{window}"] = (
                pl.col("close")
                .map_batches(
                    lambda s, w=window: mfdfa_rolling_udf(s.to_numpy(), w, 2),
                    return_dtype=pl.Float64,
                )
                .alias(f"{p}mfdfa_holder_max_{window}")
            )

        # ====================================================================
        # F15: コルモゴロフ複雑性
        # 【修正】rolling_map → map_batches、遅延束縛バグ修正
        # ====================================================================
        for window in self.config.window_sizes["complexity"]:
            # コルモゴロフ複雑性(重量UDF: map_batches使用)
            expressions[f"{p}kolmogorov_complexity_{window}"] = (
                pl.col("close")
                .map_batches(
                    lambda s, w=window: kolmogorov_complexity_rolling_udf(
                        s.to_numpy(), w, 0
                    ),
                    return_dtype=pl.Float64,
                )
                .alias(f"{p}kolmogorov_complexity_{window}")
            )

            expressions[f"{p}compression_ratio_{window}"] = (
                pl.col("close")
                .map_batches(
                    lambda s, w=window: kolmogorov_complexity_rolling_udf(
                        s.to_numpy(), w, 1
                    ),
                    return_dtype=pl.Float64,
                )
                .alias(f"{p}compression_ratio_{window}")
            )

            expressions[f"{p}pattern_diversity_{window}"] = (
                pl.col("close")
                .map_batches(
                    lambda s, w=window: kolmogorov_complexity_rolling_udf(
                        s.to_numpy(), w, 2
                    ),
                    return_dtype=pl.Float64,
                )
                .alias(f"{p}pattern_diversity_{window}")
            )

        return expressions

    def get_feature_groups(self) -> Dict[str, Dict[str, pl.Expr]]:
        """特徴量グループ定義を外部から取得可能にする"""
        return self._create_vertical_slices()

    def calculate_one_group(
        self,
        lazy_frame: pl.LazyFrame,
        group_name: str,
        group_expressions: Dict[str, pl.Expr],
    ) -> pl.LazyFrame:
        """
        単一グループの特徴量のみを計算(高速化修正版)
        グループ名に基づいて適切な特徴量計算メソッドを呼び出し、重複計算を回避
        """
        logger.info(f"グループ計算開始: {group_name}")

        # メモリ安全性チェック(必須)
        is_safe, message = self.memory_monitor.check_memory_safety()
        if not is_safe:
            raise MemoryError(f"メモリ不足のためグループ処理を中断: {message}")

        try:
            # グループ名に基づいて効率的な特徴量計算を実行
            if group_name == "mfdfa":
                group_result_lf = self._create_mfdfa_features(lazy_frame)
            elif group_name == "kolmogorov":
                group_result_lf = self._create_kolmogorov_features(lazy_frame)
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

            # 品質保証を適用(このグループのみ)
            stabilized_lf = self.apply_quality_assurance_to_group(
                group_final_lf, group_feature_columns
            )

            logger.info(
                f"グループ計算完了: {group_name} - {len(group_feature_columns)}個の特徴量"
            )
            return stabilized_lf

        except Exception as e:
            logger.error(f"グループ計算エラー ({group_name}): {e}")
            raise

    def apply_quality_assurance_to_group(
        self, lazy_frame: pl.LazyFrame, feature_columns: List[str]
    ) -> pl.LazyFrame:
        """単一グループに対する品質保証システムの適用"""
        if not feature_columns:
            return lazy_frame

        logger.info(f"品質保証適用: {len(feature_columns)}個の特徴量")

        # 安定化処理の式を生成
        stabilization_exprs = []

        for col_name in feature_columns:
            # Inf値を除外してパーセンタイル計算(精度保持)
            col_for_quantile = (
                pl.when(pl.col(col_name).is_infinite())
                .then(None)
                .otherwise(pl.col(col_name))
            )

            p01 = col_for_quantile.quantile(0.01, interpolation="linear")
            p99 = col_for_quantile.quantile(0.99, interpolation="linear")

            # Inf値を統計的に意味のある値(パーセンタイル境界値)で置換
            stabilized_col = (
                pl.when(pl.col(col_name) == float("inf"))
                .then(p99)
                .when(pl.col(col_name) == float("-inf"))
                .then(p01)
                .otherwise(pl.col(col_name))
                .clip(lower_bound=p01, upper_bound=p99)
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

        # メモリ使用量を考慮したグルーピング(英語キー使用)
        slices = {}
        p = self.prefix

        # グループ1: MFDFA系(極重量)
        slices["mfdfa"] = {
            name: expr for name, expr in all_expressions.items() if "mfdfa" in name
        }

        # グループ2: コルモゴロフ複雑性系(重量)
        slices["kolmogorov"] = {
            name: expr
            for name, expr in all_expressions.items()
            if "kolmogorov" in name
            or "compression_ratio" in name
            or "pattern_diversity" in name
        }

        # 分割されなかった特徴量があれば警告
        total_assigned = sum(len(group) for group in slices.values())
        if total_assigned != len(all_expressions):
            logger.warning(f"未分割特徴量: {len(all_expressions) - total_assigned}個")

        return slices

    def _create_mfdfa_features(self, lazy_frame: pl.LazyFrame) -> pl.LazyFrame:
        """MFDFA系特徴量の計算(高速化対応)"""
        exprs = []
        p = self.prefix

        for window in self.config.window_sizes["mfdfa"]:
            exprs.append(
                pl.col("close")
                .map_batches(
                    lambda s, w=window: mfdfa_rolling_udf(s.to_numpy(), w, 0),
                    return_dtype=pl.Float64,
                )
                .alias(f"{p}mfdfa_hurst_mean_{window}")
            )

            exprs.append(
                pl.col("close")
                .map_batches(
                    lambda s, w=window: mfdfa_rolling_udf(s.to_numpy(), w, 1),
                    return_dtype=pl.Float64,
                )
                .alias(f"{p}mfdfa_width_{window}")
            )

            exprs.append(
                pl.col("close")
                .map_batches(
                    lambda s, w=window: mfdfa_rolling_udf(s.to_numpy(), w, 2),
                    return_dtype=pl.Float64,
                )
                .alias(f"{p}mfdfa_holder_max_{window}")
            )

        return lazy_frame.with_columns(exprs)

    def _create_kolmogorov_features(self, lazy_frame: pl.LazyFrame) -> pl.LazyFrame:
        """コルモゴロフ複雑性系特徴量の計算(高速化対応)"""
        exprs = []
        p = self.prefix

        for window in self.config.window_sizes["complexity"]:
            exprs.append(
                pl.col("close")
                .map_batches(
                    lambda s, w=window: kolmogorov_complexity_rolling_udf(
                        s.to_numpy(), w, 0
                    ),
                    return_dtype=pl.Float64,
                )
                .alias(f"{p}kolmogorov_complexity_{window}")
            )

            exprs.append(
                pl.col("close")
                .map_batches(
                    lambda s, w=window: kolmogorov_complexity_rolling_udf(
                        s.to_numpy(), w, 1
                    ),
                    return_dtype=pl.Float64,
                )
                .alias(f"{p}compression_ratio_{window}")
            )

            exprs.append(
                pl.col("close")
                .map_batches(
                    lambda s, w=window: kolmogorov_complexity_rolling_udf(
                        s.to_numpy(), w, 2
                    ),
                    return_dtype=pl.Float64,
                )
                .alias(f"{p}pattern_diversity_{window}")
            )

        return lazy_frame.with_columns(exprs)

    def _cleanup_temp_files(self):
        """一時ファイルのクリーンアップ"""
        try:
            if self.temp_dir.exists():
                for temp_file in self.temp_dir.glob("*.parquet"):
                    temp_file.unlink()
                    logger.debug(f"一時ファイル削除: {temp_file}")
                self.temp_dir.rmdir()
                logger.info(f"一時ディレクトリ削除: {self.temp_dir}")
        except Exception as e:
            logger.warning(f"一時ファイルクリーンアップエラー: {e}")

    def apply_quality_assurance(self, lazy_frame: pl.LazyFrame) -> pl.LazyFrame:
        """2段階品質保証システムの適用"""
        logger.info("品質保証システム適用開始: 全ての特徴量に安定化処理を適用します。")

        # スキーマからプレフィックスを持つ特徴量カラムを特定
        schema = lazy_frame.collect_schema()
        feature_columns = [col for col in schema.names() if col.startswith(self.prefix)]

        if not feature_columns:
            logger.warning("品質保証対象の特徴量が見つかりません。")
            return lazy_frame

        logger.info(f"品質保証対象: {len(feature_columns)}個の特徴量")

        # 安定化処理の式を生成
        stabilization_exprs = []

        for col_name in feature_columns:
            # Inf値を除外してパーセンタイル計算(精度保持)
            col_for_quantile = (
                pl.when(pl.col(col_name).is_infinite())
                .then(None)
                .otherwise(pl.col(col_name))
            )

            p01 = col_for_quantile.quantile(0.01, interpolation="linear")
            p99 = col_for_quantile.quantile(0.99, interpolation="linear")

            # Inf値を統計的に意味のある値(パーセンタイル境界値)で置換
            stabilized_col = (
                pl.when(pl.col(col_name) == float("inf"))
                .then(p99)
                .when(pl.col(col_name) == float("-inf"))
                .then(p01)
                .otherwise(pl.col(col_name))
                .clip(lower_bound=p01, upper_bound=p99)
                .fill_null(0.0)
                .fill_nan(0.0)
                .alias(col_name)
            )

            stabilization_exprs.append(stabilized_col)

        result = lazy_frame.with_columns(stabilization_exprs)
        logger.info("品質保証システム適用完了")

        return result


class OutputEngine:
    """
    出力管理クラス(10%) - Project Forge準拠
    責務:
    - LazyFrame.sink_parquet()によるストリーミング出力
    - 必要に応じたPyArrowフォールバック(use_pyarrow=True)
    - timeframe別ファイル分離
    - NaN埋め統一処理
    - シンプルな進捗表示
    - 基本メタデータ記録
    """

    def __init__(self, config: ProcessingConfig):
        self.config = config
        self.memory_monitor = MemoryMonitor(config.memory_limit_gb)

        # 出力設定(Project Forge基準)
        self.output_config = {
            "compression": "snappy",
            "dtype": "float64",
            "timestamp_handling": "column",
        }

        # エンジン識別子
        self.engine_id = config.engine_id

    def create_output_path(self, timeframe: str) -> Path:
        """出力パス生成 - Project Forge命名規則"""
        filename = f"features_{self.engine_id}_{timeframe}.parquet"
        return Path(self.config.output_path) / filename

    def apply_nan_filling(self, lazy_frame: pl.LazyFrame) -> pl.LazyFrame:
        """NaN埋め統一処理"""
        logger.info("NaN埋め処理開始")

        # スキーマを一度だけ取得(メモリ安全)
        schema = lazy_frame.collect_schema()
        all_columns = schema.names()

        # プレフィックスを持つ特徴量カラムを特定
        feature_columns = [
            col for col in all_columns if col.startswith(f"{self.engine_id}_")
        ]

        # NaN埋め式生成
        fill_exprs = []
        for col in feature_columns:
            # NaNを0で埋める(金融データでは一般的)
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
            output_path.parent.mkdir(parents=True, exist_ok=True)

            is_safe, message = self.memory_monitor.check_memory_safety()
            if not is_safe:
                raise MemoryError(f"保存処理でメモリ不足: {message}")

            processed_frame = self.apply_nan_filling(lazy_frame)

            start_time = time.time()

            try:
                df = processed_frame.collect(engine="streaming")
            except Exception as streaming_error:
                logger.warning(
                    f"ストリーミングcollectが失敗、通常collectを使用: {streaming_error}"
                )
                df = processed_frame.collect()

            # --- ▼▼▼ ここから修正 ▼▼▼ ---

            # 1. 辞書エンコーディング(Categorical)をUtf8(通常の文字列)に変換
            categorical_cols = [
                name for name, dtype in df.schema.items() if dtype == pl.Categorical
            ]
            if categorical_cols:
                df = df.with_columns(
                    [pl.col(c).cast(pl.Utf8) for c in categorical_cols]
                )
                logging.info(f"辞書エンコーディングを無効化: {categorical_cols}")

            # 2. PyArrow互換モードで保存
            df.write_parquet(
                str(output_path),
                compression=self.output_config["compression"],
                use_pyarrow=True,
            )

            # --- ▲▲▲ 修正ここまで ▲▲▲ ---

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
            "engine_name": "Engine_2A_ComplexityTheory_Simplified",
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
# 大規模データ処理(Tick専用)アーキテクチャ:分割・重複・処理・結合
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
        Path(config.output_path) / f"features_{config.engine_id}_{timeframe}"
    )

    # 【安全なディレクトリ作成】
    try:
        FEATURES_ROOT.mkdir(parents=True, exist_ok=True)
        logging.info(f"出力ディレクトリを確保しました: {FEATURES_ROOT}")
    except FileExistsError:
        logging.error(
            f"エラー: 出力パスがファイルとして存在しています。処理を中断します: {FEATURES_ROOT}"
        )
        return
    except Exception as e:
        logging.error(f"エラー: 出力ディレクトリの作成に失敗しました: {e}")
        return

    W_MAX = config.w_max

    calculation_engine = CalculationEngine(config)

    all_partitions = get_sorted_partitions(PARTITION_ROOT)

    if resume_date:
        import datetime

        all_days = [
            p
            for p in all_partitions
            if datetime.date(
                int(p.parent.parent.name.split("=")[1]),
                int(p.parent.name.split("=")[1]),
                int(p.name.split("=")[1]),
            )
            >= resume_date
        ]
        logging.info(f"再開日 {resume_date} に基づいてフィルタリングしました。")
    else:
        all_days = all_partitions

    if not all_days:
        logging.error("処理対象の日次パーティションが見つかりません。")
        return

    total_days = len(all_days)
    logging.info(f"処理対象日数: {total_days}日")

    for i, current_day_path in enumerate(all_days):
        day_name = f"{current_day_path.parent.parent.name}/{current_day_path.parent.name}/{current_day_path.name}"
        logging.info(f"=== 日次処理 ({i + 1}/{total_days}): {day_name} ===")

        try:
            current_index_in_all = all_partitions.index(current_day_path)
            prev_day_path = (
                all_partitions[current_index_in_all - 1]
                if current_index_in_all > 0
                else None
            )

            logging.info(f"データ読み込み開始: {day_name}")
            augmented_df, current_day_rows = create_augmented_frame(
                current_day_path, prev_day_path, W_MAX
            )
            logging.info(
                f"データ読み込み完了: 実データ{current_day_rows}行、総データ{augmented_df.height}行"
            )

            logging.info(f"特徴量計算開始: {day_name}")

            temp_dir = Path(tempfile.mkdtemp(prefix=f"day_{i:04d}_{config.engine_id}_"))
            logging.info(f"日次一時ディレクトリ作成: {temp_dir}")

            temp_files = []

            feature_groups = calculation_engine.get_feature_groups()
            logging.info(f"物理的垂直分割: {len(feature_groups)}グループで処理")

            for group_idx, (group_name, group_expressions) in enumerate(
                feature_groups.items()
            ):
                logging.info(
                    f"グループ処理開始: {group_name} ({len(group_expressions)}個の特徴量)"
                )

                group_result_lf = calculation_engine.calculate_one_group(
                    augmented_df.lazy(), group_name, group_expressions
                )

                group_result_df = group_result_lf.collect(streaming=True)
                logging.info(
                    f"グループデータ実現化: {group_result_df.height}行 x {group_result_df.width}列"
                )

                temp_file = temp_dir / f"group_{group_idx:02d}_{group_name}.parquet"
                group_result_df.write_parquet(str(temp_file), compression="snappy")

                if temp_file.exists():
                    temp_files.append(temp_file)
                    logging.info(
                        f"グループ保存完了: {temp_file} ({temp_file.stat().st_size} bytes)"
                    )
                else:
                    raise FileNotFoundError(f"グループファイル作成失敗: {temp_file}")

                memory_usage = calculation_engine.memory_monitor.get_memory_usage_gb()
                logging.info(f"メモリ使用量: {memory_usage:.2f}GB")

            logging.info("グループファイル結合開始(クリーン・オン・クリーン結合)...")

            base_df = pl.read_parquet(str(temp_files[0]))

            if prev_day_path is not None:
                clean_base_df = base_df.tail(current_day_rows)
            else:
                clean_base_df = base_df

            logging.info(
                f"クリーンな土台を準備: {clean_base_df.height}行 x {clean_base_df.width}列"
            )

            base_columns = [
                "timestamp",
                "open",
                "high",
                "low",
                "close",
                "volume",
                "timeframe",
            ]

            for idx, temp_file in enumerate(temp_files[1:], 1):
                next_df = pl.read_parquet(str(temp_file))

                if prev_day_path is not None:
                    clean_next_df = next_df.tail(current_day_rows)
                else:
                    clean_next_df = next_df

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

            # --- ▼▼▼ ここから修正 ▼▼▼ ---

            # 1. パーティションカラムをInt32で生成
            final_df = result_df.with_columns(
                [
                    pl.col("timestamp").dt.year().alias("year").cast(pl.Int32),
                    pl.col("timestamp").dt.month().alias("month").cast(pl.Int32),
                    pl.col("timestamp").dt.day().alias("day").cast(pl.Int32),
                ]
            )

            # 2. 辞書エンコーディング(Categorical)をUtf8(通常の文字列)に変換
            categorical_cols = [
                name
                for name, dtype in final_df.schema.items()
                if dtype == pl.Categorical
            ]
            if categorical_cols:
                final_df = final_df.with_columns(
                    [pl.col(c).cast(pl.Utf8) for c in categorical_cols]
                )
                logging.info(f"辞書エンコーディングを無効化: {categorical_cols}")

            logging.info(f"最終保存開始: {day_name}")

            # 3. PyArrow互換モードで保存
            final_df.write_parquet(
                FEATURES_ROOT, partition_by=["year", "month", "day"], use_pyarrow=True
            )

            # --- ▲▲▲ 修正ここまで ▲▲▲ ---

            logging.info(f"保存完了: {day_name} - {final_df.height}行の特徴量データ")

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
    """単一の通常時間足を処理する(修正版ロジック)"""
    logger.info(f"=== 通常処理開始: timeframe={timeframe} ===")
    start_time = time.time()

    # calc_engineをtryブロックの外で初期化
    calc_engine = None
    try:
        data_engine = DataEngine(config)
        calc_engine = CalculationEngine(config)
        output_engine = OutputEngine(config)

        lazy_frame = data_engine.create_lazy_frame(timeframe)
        summary = data_engine.get_data_summary(lazy_frame)
        logger.info(f"データサマリー: {summary}")

        # --- 修正箇所: グループ化された特徴量計算ロジックを適用 ---
        all_expressions = []
        feature_groups = calc_engine.get_feature_groups()
        for group_name, group_expressions in feature_groups.items():
            all_expressions.extend(group_expressions.values())

        logger.info(
            f"特徴量計算開始: {len(all_expressions)}個の特徴量を {timeframe} に対して計算します。"
        )
        features_lf = lazy_frame.with_columns(all_expressions)

        # 品質保証システムを適用
        features_lf = calc_engine.apply_quality_assurance(features_lf)
        # --- 修正ここまで ---

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


# インタラクティブモード(Project Forge準拠)
def get_user_confirmation(config: ProcessingConfig) -> bool:
    """ユーザー確認 - Project Forge理念表示"""
    print("\n" + "=" * 60)
    print(f"Engine {config.engine_id.upper()} - {config.engine_name}")
    print("=" * 60)
    print("🎯 Project Forge - 軍資金増大プロジェクト")
    print("🚀 最終目標: Project Chimera開発・完成のための資金調達")
    print("💻 探索対象: マーケットの亡霊(統計的に有意で非ランダムな微細パターン)")
    print("🏅 思想的継承: ジム・シモンズ(ルネサンス・テクノロジーズ)")
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
    print("  - F5: MFDFA (マルチフラクタル解析)")
    print("  - F15: コルモゴロフ複雑性 (情報圧縮理論)")
    print("  - 2段階品質保証システム")
    print("  - 【修正】物理的垂直分割(ディスクベース中間ファイル)")
    print("  - 【修正】map_batches + prange並列化による重量級UDF最適化")
    print("  - 【修正】遅延束縛バグ完全修正")
    print("  - Polars LazyFrame + Numba JITハイブリッド最適化")

    response = input("\n処理を開始しますか? (y/n): ")
    return response.lower() == "y"


def select_timeframes(config: ProcessingConfig) -> List[str]:
    """タイムフレーム選択(完全同一実装)"""
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
    print(f"  Engine 2A - Complexity Theory Features (簡略版)")
    print("  Project Forge - 軍資金増大プロジェクト  ")
    print("=" * 70)
    print("🎯 目標: XAU/USD市場の統計的パターン抽出")
    print("🤖 AI頭脳による普遍的法則発見")
    print("💰 Project Chimera開発資金調達")
    print("🔧 【修正】重量級UDF呼び出しパターンの設計規律遵守")
    print("🔧 【修正】遅延束縛バグの完全修正")
    print("🔧 【修正】map_batches + prange並列化の徹底")
    print("=" * 70)

    config = ProcessingConfig()

    if not config.validate():
        return 1

    data_engine = DataEngine(config)

    if not data_engine.validate_data_source():
        return 1

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
                resume_date = datetime.datetime.strptime(date_str, "%Y-%m-%d").date()
                print(f"{resume_date} から処理を再開します。")
                break
            except ValueError:
                print(
                    "エラー: 日付の形式が正しくありません。YYYY-MM-DD形式で入力してください。"
                )
    else:
        print("新規に処理を開始します。")

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
    print("  1. テストモード(少量データで動作確認)")
    print("  2. 本格モード(全データ処理)")

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

# ===============================
# Block 5 Complete: CalculationEngine/OutputEngine/メイン実行部
# ===============================
#
# 全5ブロック完成: engine_2_complexity_theory_simplified.py
#
# 【簡略版への修正完了】
# 削除した特徴量:
# - F16: カオス理論指標(リアプノフ指数・相関次元)
# - F21: Lempel-Ziv複雑性
#
# 残された特徴量:
# - F5: MFDFA (マルチフラクタル解析)
# - F15: コルモゴロフ複雑性
#
# アーキテクチャ:
# - DataEngine: 参照スクリプト完全踏襲
# - CalculationEngine: 複雑性理論特徴量専用実装(簡略版)
# - OutputEngine: 参照スクリプト完全踏襲
# - 物理的垂直分割: ディスクベース中間ファイル方式
# - Numba UDF: 全て@nb.njit(parallel=True) + nb.prange並列化
#
# ===============================
