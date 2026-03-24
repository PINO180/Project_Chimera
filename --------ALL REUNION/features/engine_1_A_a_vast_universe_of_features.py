#!/usr/bin/env python3
"""
革新的特徴量収集スクリプト - Engine 1A: 基礎統計・ロバスト統計特徴量 (アーキテクチャ刷新版)
【修正内容】参照スクリプトのアーキテクチャに完全準拠、特徴量計算ロジックは従来版を保持

Project Forge - 軍資金増大プロジェクト
最終目標: Project Chimera開発・完成のための資金調達

技術戦略: ジム・シモンズの思想的継承
- 経済学・ファンダメンタルズ・古典的テクニカル指標の完全排除
- 統計的に有意で非ランダムな微細パターン「マーケットの亡霊」の探索
- AIの頭脳による普遍的法則の読み解き

アーキテクチャ: 3クラス構成（最適化版）+ ディスクベース垂直分割
- DataEngine（30%）: Polars LazyFrame基盤
- CalculationEngine（60%）: 特徴量計算核心（物理的垂直分割実装）
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
        "rsi": [14, 21, 30, 50],
        "atr": [13, 21, 34, 55],
        "adx": [13, 21, 34],
        "hma": [21, 34, 55],
        "kama": [21, 34],
        "general": [10, 20, 50, 100],
    }


@dataclass
class ProcessingConfig:
    """処理設定 - Project Forge統合版"""

    # データパス（Project Forge構造準拠） - config.pyから読み込む
    input_path: str = str(config.S1_BASE_MULTITIMEFRAME)
    partitioned_tick_path: str = str(config.S1_RAW_TICK_PARTITIONED)
    output_path: str = str(config.S2_FEATURES)

    # エンジン識別
    engine_id: str = "e1a"
    engine_name: str = "Engine_1A_Basic_Statistical_Refactored"

    # 並列処理（戦略的並列処理スロットリング）
    max_threads: int = 4

    # メモリ制限（64GB RAM制約）
    memory_limit_gb: float = 55.0
    memory_warning_gb: float = 50.0

    timeframes: List[str] = field(default_factory=get_default_timeframes)
    window_sizes: Dict[str, List[int]] = field(default_factory=get_default_window_sizes)

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
    データ基盤クラス（30%）- Project Forge統合版
    責務:
    - Parquetメタデータ事前検証
    - Polars scan_parquetによる遅延読み込み（LazyFrame生成）
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


class QualityAssurance:
    """数値安定性保証システム - 2段階品質保証エンジン（Project Forge準拠）"""

    @staticmethod
    def calculate_quality_score(values: np.ndarray) -> float:
        """
        品質スコア算出：0.0（使用不可）〜 1.0（完璧）
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
        # シャノンエントロピーによる多様性評価
        unique_values, counts = np.unique(finite_values, return_counts=True)
        n_unique = len(unique_values)

        if n_unique == 1:
            diversity_score = 0.0  # 完全に均一
        else:
            # シャノンエントロピーの厳密計算
            shannon_entropy = 0.0
            for count in counts:
                p = count / finite_count
                if p > 0:
                    shannon_entropy -= p * np.log2(p)

            # 最大可能エントロピー（完全均等分布時）
            max_entropy = np.log2(min(n_unique, finite_count))
            if max_entropy > 0:
                diversity_score = shannon_entropy / max_entropy
            else:
                diversity_score = 0.0

        # 3. 数値的安定性指標の厳密計算
        if finite_count < 2:
            stability_score = 1.0
        else:
            # 不偏分散の厳密計算
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

            # 変動係数による安定性評価
            if abs(mean_val) > 1e-15:
                cv = std_dev / abs(mean_val)
                stability_score = 1.0 / (1.0 + cv)
            else:
                stability_score = 1.0 / (1.0 + std_dev)

        # 4. 外れ値耐性指標の厳密計算
        if finite_count < 5:
            outlier_resistance = 1.0
        else:
            # Tukey's fences による外れ値検出
            sorted_values = np.sort(finite_values)

            # 四分位数の厳密計算（R-6 quantile method）
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
                # Tukey's fences
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
            "finite_ratio": 0.4,  # 有限値率（最重要）
            "diversity": 0.25,  # 多様性
            "stability": 0.25,  # 安定性
            "outlier_resistance": 0.1,  # 外れ値耐性
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
        第1段階：基本対処（品質スコア > 0.6）
        軽量で高速な処理
        """
        # NaN/Infを0で置換
        cleaned = np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0)

        # 有限値での上下1%除去
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
        第2段階：フォールバック（品質スコア ≤ 0.6）
        ロバスト統計による処理
        """
        finite_mask = np.isfinite(values)
        if np.sum(finite_mask) < 3:
            # データが不十分な場合は0埋め
            return np.zeros_like(values)

        finite_values = values[finite_mask]

        try:
            # ロバスト統計による処理
            median_val = np.median(finite_values)

            # MAD計算
            abs_deviations = np.abs(finite_values - median_val)
            mad_val = np.median(abs_deviations)

            if mad_val < 1e-10:
                mad_val = np.std(finite_values) * 0.6745  # フォールバック

            # MADベースの外れ値除去
            robust_bounds = (median_val - 3 * mad_val, median_val + 3 * mad_val)

            # 全データに適用
            result = np.copy(values)
            result = np.nan_to_num(
                result, nan=median_val, posinf=robust_bounds[1], neginf=robust_bounds[0]
            )
            result = np.clip(result, robust_bounds[0], robust_bounds[1])

        except Exception:
            # 最終フォールバック: 中央値埋め
            median_val = np.median(finite_values) if len(finite_values) > 0 else 0.0
            result = np.full_like(values, median_val)

        return result


# ===============================
# Numba UDF定義（クラス外必須）
# ===============================

# 【重要】全てのNumba UDFは必ずクラス外で定義すること
# クラス内定義は循環参照エラーを引き起こすため絶対禁止


# 改修対象スクリプトからNumba UDFを移植
@nb.guvectorize(["void(float64[:], float64[:])"], "(n)->(n)", nopython=True, cache=True)
def fast_rolling_mean_numba(arr, out):
    """Numba最適化ローリング平均（カスタムウィンドウ用）"""
    n = len(arr)
    for i in range(n):
        if i < 20:  # 最小ウィンドウサイズ
            out[i] = np.nan
        else:
            window_sum = 0.0
            count = 0
            for j in range(max(0, i - 19), i + 1):
                if not np.isnan(arr[j]):
                    window_sum += arr[j]
                    count += 1
            out[i] = window_sum / count if count > 0 else np.nan


@nb.guvectorize(["void(float64[:], float64[:])"], "(n)->(n)", nopython=True, cache=True)
def fast_rolling_std_numba(arr, out):
    """Numba最適化ローリング標準偏差"""
    n = len(arr)
    for i in range(n):
        if i < 20:
            out[i] = np.nan
        else:
            # 平均計算
            window_sum = 0.0
            count = 0
            for j in range(max(0, i - 19), i + 1):
                if not np.isnan(arr[j]):
                    window_sum += arr[j]
                    count += 1

            if count <= 1:
                out[i] = np.nan
                continue

            mean_val = window_sum / count

            # 分散計算
            var_sum = 0.0
            for j in range(max(0, i - 19), i + 1):
                if not np.isnan(arr[j]):
                    diff = arr[j] - mean_val
                    var_sum += diff * diff

            variance = var_sum / (count - 1)
            out[i] = np.sqrt(variance)


@nb.guvectorize(["void(float64[:], float64[:])"], "(n)->(n)", nopython=True, cache=True)
def fast_quality_score_numba(arr, out):
    """高速品質スコア計算"""
    n = len(arr)
    for i in range(n):
        if i < 50:  # 最小評価ウィンドウ
            out[i] = 0.0
        else:
            # ウィンドウ内のNaN/Inf率計算
            window_size = min(50, i + 1)
            nan_inf_count = 0
            finite_count = 0

            for j in range(i - window_size + 1, i + 1):
                if np.isnan(arr[j]) or np.isinf(arr[j]):
                    nan_inf_count += 1
                else:
                    finite_count += 1

            if window_size == 0:
                out[i] = 0.0
            else:
                nan_inf_ratio = nan_inf_count / window_size
                finite_ratio = finite_count / window_size
                out[i] = finite_ratio * 0.8 + 0.2  # 基本品質スコア


# 改修対象スクリプトから高度なロバスト統計のNumba UDFを移植
@nb.guvectorize(["void(float64[:], float64[:])"], "(n)->(n)", nopython=True, cache=True)
def mad_rolling_numba(arr, out):
    """ローリングMAD計算"""
    n = len(arr)
    window = 20

    for i in range(n):
        if i < window - 1:
            out[i] = np.nan
        else:
            # ウィンドウ内データ取得
            window_data = arr[max(0, i - window + 1) : i + 1]
            finite_data = window_data[~np.isnan(window_data)]

            if len(finite_data) < 3:
                out[i] = np.nan
            else:
                # 中央値計算
                median_val = np.median(finite_data)
                # 絶対偏差の中央値
                abs_deviations = np.abs(finite_data - median_val)
                out[i] = np.median(abs_deviations)


@nb.guvectorize(["void(float64[:], float64[:])"], "(n)->(n)", nopython=True, cache=True)
def biweight_location_numba(arr, out):
    """ローリングBiweight位置計算（厳密実装）"""
    n = len(arr)
    window = 20

    for i in range(n):
        if i < window - 1:
            out[i] = np.nan
        else:
            window_data = arr[max(0, i - window + 1) : i + 1]
            finite_data = window_data[~np.isnan(window_data)]

            if len(finite_data) < 5:
                out[i] = np.median(finite_data) if len(finite_data) > 0 else np.nan
            else:
                # Tukey's Biweight位置推定の厳密実装
                # 反復アルゴリズム

                # 初期値として中央値を使用
                current_location = np.median(finite_data)
                tolerance = 1e-10
                max_iterations = 50

                for iteration in range(max_iterations):
                    # MAD計算
                    abs_residuals = np.abs(finite_data - current_location)
                    mad_val = np.median(abs_residuals)

                    if mad_val < 1e-15:
                        break

                    # スケールファクター（6 * MAD）
                    scale = 6.0 * mad_val

                    # 標準化残差
                    u_values = (finite_data - current_location) / scale

                    # Biweight重み関数の計算
                    weights = np.zeros(len(finite_data))
                    numerator = 0.0
                    denominator = 0.0

                    for j in range(len(finite_data)):
                        u_abs = abs(u_values[j])

                        if u_abs < 1.0:
                            # Biweight重み: (1 - u²)²
                            weight = (1.0 - u_values[j] ** 2) ** 2
                            weights[j] = weight
                            numerator += finite_data[j] * weight
                            denominator += weight
                        else:
                            weights[j] = 0.0

                    if denominator > 1e-15:
                        new_location = numerator / denominator
                    else:
                        # 重みがすべてゼロの場合は中央値を返す
                        new_location = np.median(finite_data)
                        break

                    # 収束判定
                    if abs(new_location - current_location) < tolerance:
                        break

                    current_location = new_location

                out[i] = current_location


@nb.guvectorize(["void(float64[:], float64[:])"], "(n)->(n)", nopython=True, cache=True)
def winsorized_mean_numba(arr, out):
    """ローリングウィンソライズ平均（上下5%クリップ）"""
    n = len(arr)
    window = 20

    for i in range(n):
        if i < window - 1:
            out[i] = np.nan
        else:
            window_data = arr[max(0, i - window + 1) : i + 1]
            finite_data = window_data[~np.isnan(window_data)]

            if len(finite_data) < 5:
                out[i] = np.mean(finite_data) if len(finite_data) > 0 else np.nan
            else:
                # 上下5%点の計算
                p05 = np.percentile(finite_data, 5)
                p95 = np.percentile(finite_data, 95)

                # ウィンソライズ（クリッピング）
                winsorized_data = np.clip(finite_data, p05, p95)
                out[i] = np.mean(winsorized_data)


@nb.guvectorize(["void(float64[:], float64[:])"], "(n)->(n)", nopython=True, cache=True)
def jarque_bera_statistic_numba(arr, out):
    """ローリングJarque-Bera検定統計量"""
    n = len(arr)
    window = 50

    for i in range(n):
        if i < window - 1:
            out[i] = np.nan
        else:
            window_data = arr[max(0, i - window + 1) : i + 1]
            finite_data = window_data[~np.isnan(window_data)]

            if len(finite_data) < 20:
                out[i] = np.nan
            else:
                # 基本統計量
                mean_val = np.mean(finite_data)

                # 手動分散計算（Numba対応）
                variance = 0.0
                for val in finite_data:
                    variance += (val - mean_val) ** 2
                variance = variance / (len(finite_data) - 1)
                std_val = np.sqrt(variance)

                if std_val < 1e-10:
                    out[i] = 0.0
                else:
                    # 標準化
                    z_sum_3 = 0.0
                    z_sum_4 = 0.0
                    for val in finite_data:
                        z = (val - mean_val) / std_val
                        z_sum_3 += z**3
                        z_sum_4 += z**4

                    skewness = z_sum_3 / len(finite_data)
                    kurtosis = z_sum_4 / len(finite_data) - 3

                    # JB統計量
                    jb_stat = len(finite_data) * (skewness**2 / 6 + kurtosis**2 / 24)
                    out[i] = jb_stat


@nb.guvectorize(["void(float64[:], float64[:])"], "(n)->(n)", nopython=True, cache=True)
def anderson_darling_numba(arr, out):
    """ローリングAnderson-Darling統計量（厳密実装）"""
    n = len(arr)
    window = 30

    # 低速な数値積分によるCDFを、高速な誤差関数(erf)を用いた解析的な計算式に置き換えます。
    # この関数は Numba の JIT コンパイル対象となります。
    # math.sqrt(2.0) は約 1.4142135623730951
    SQRT2 = 1.4142135623730951

    def standard_normal_cdf_fast(x):
        """標準正規分布の累積分布関数（高速実装）"""
        return 0.5 * (1.0 + math.erf(x / SQRT2))

    for i in range(n):
        if i < window - 1:
            out[i] = np.nan
        else:
            window_data = arr[max(0, i - window + 1) : i + 1]
            finite_data = window_data[~np.isnan(window_data)]

            if len(finite_data) < 10:
                out[i] = np.nan
            else:
                # ソート
                sorted_data = np.sort(finite_data)
                n_data = len(sorted_data)

                # 手動で平均と標準偏差計算（Numba対応）
                mean_val = np.mean(sorted_data)

                # 手動分散計算
                variance = 0.0
                for val in sorted_data:
                    variance += (val - mean_val) ** 2
                variance = variance / (n_data - 1)
                std_val = np.sqrt(variance)

                if std_val < 1e-10:
                    out[i] = 0.0
                else:
                    # 標準化後の厳密なAnderson-Darling統計量
                    standardized_data = np.zeros(n_data)
                    for j in range(n_data):
                        standardized_data[j] = (sorted_data[j] - mean_val) / std_val

                    # Anderson-Darling統計量の厳密計算
                    ad_sum = 0.0
                    for j in range(n_data):
                        # 高速化されたCDF関数を呼び出します。
                        F_j = standard_normal_cdf_fast(standardized_data[j])
                        F_nj = standard_normal_cdf_fast(
                            standardized_data[n_data - 1 - j]
                        )

                        # ゼロ除算回避
                        if F_j > 1e-15 and (1 - F_nj) > 1e-15:
                            log_term = np.log(F_j) + np.log(1 - F_nj)
                            ad_sum += (2 * j + 1) * log_term

                    # Anderson-Darling統計量
                    out[i] = -n_data - ad_sum / n_data


@nb.guvectorize(["void(float64[:], float64[:])"], "(n)->(n)", nopython=True, cache=True)
def runs_test_numba(arr, out):
    """ローリングRuns Test統計量"""
    n = len(arr)
    window = 30

    for i in range(n):
        if i < window - 1:
            out[i] = np.nan
        else:
            window_data = arr[max(0, i - window + 1) : i + 1]
            finite_data = window_data[~np.isnan(window_data)]

            if len(finite_data) < 10:
                out[i] = np.nan
            else:
                # 中央値を基準にバイナリ系列作成
                median_val = np.median(finite_data)
                binary_series = (finite_data > median_val).astype(np.int32)

                # ランの数をカウント
                runs = 1
                for j in range(1, len(binary_series)):
                    if binary_series[j] != binary_series[j - 1]:
                        runs += 1

                # 期待ランの数と分散
                n1 = np.sum(binary_series)  # 1の個数
                n2 = len(binary_series) - n1  # 0の個数

                if n1 > 0 and n2 > 0:
                    expected_runs = (2 * n1 * n2) / (n1 + n2) + 1
                    var_runs = (2 * n1 * n2 * (2 * n1 * n2 - n1 - n2)) / (
                        (n1 + n2) ** 2 * (n1 + n2 - 1)
                    )

                    if var_runs > 0:
                        # 標準化統計量
                        out[i] = (runs - expected_runs) / np.sqrt(var_runs)
                    else:
                        out[i] = 0.0
                else:
                    out[i] = 0.0


@nb.guvectorize(["void(float64[:], float64[:])"], "(n)->(n)", nopython=True, cache=True)
def von_neumann_ratio_numba(arr, out):
    """ローリングVon Neumann比（厳密実装）"""
    n = len(arr)
    window = 30

    for i in range(n):
        if i < window - 1:
            out[i] = np.nan
        else:
            window_data = arr[max(0, i - window + 1) : i + 1]
            finite_data = window_data[~np.isnan(window_data)]

            if len(finite_data) < 3:  # 最低3点必要（差分計算のため）
                out[i] = np.nan
            else:
                n_points = len(finite_data)

                # 1次差分の平方和（厳密計算）
                diff_sq_sum = 0.0
                for j in range(1, n_points):
                    diff = finite_data[j] - finite_data[j - 1]
                    diff_sq_sum += diff * diff

                # 平均値の厳密計算
                sum_values = 0.0
                for j in range(n_points):
                    sum_values += finite_data[j]
                mean_val = sum_values / n_points

                # 不偏分散の厳密計算（n-1で除算）
                sum_sq_deviations = 0.0
                for j in range(n_points):
                    deviation = finite_data[j] - mean_val
                    sum_sq_deviations += deviation * deviation

                # Von Neumann比の厳密な定義
                if sum_sq_deviations > 1e-15:
                    # 分子: 1次差分の平方和
                    # 分母: 総平方和（不偏分散 × (n-1)）
                    vn_ratio = diff_sq_sum / sum_sq_deviations

                    # 理論的範囲チェック（0 ≤ VN比 ≤ 4）
                    if vn_ratio < 0.0:
                        out[i] = 0.0
                    elif vn_ratio > 4.0:
                        out[i] = 4.0
                    else:
                        out[i] = vn_ratio
                else:
                    # 全て同じ値の場合、理論的にVN比は0
                    out[i] = 0.0


@nb.guvectorize(["void(float64[:], float64[:])"], "(n)->(n)", nopython=True, cache=True)
def basic_stabilization_numba(arr, out):
    n = len(arr)
    for i in range(n):
        if np.isnan(arr[i]) or np.isinf(arr[i]):
            out[i] = 0.0
        else:
            out[i] = arr[i]
    finite_count = 0
    for i in range(n):
        if np.isfinite(out[i]):
            finite_count += 1
    if finite_count > 10:
        min_val = np.nanmin(out)
        max_val = np.nanmax(out)
        range_val = max_val - min_val
        if range_val > 1e-10:
            clip_margin = range_val * 0.01
            for i in range(n):
                if np.isfinite(out[i]):
                    if out[i] < min_val + clip_margin:
                        out[i] = min_val + clip_margin
                    elif out[i] > max_val - clip_margin:
                        out[i] = max_val - clip_margin


@nb.guvectorize(["void(float64[:], float64[:])"], "(n)->(n)", nopython=True, cache=True)
def robust_stabilization_numba(arr, out):
    n = len(arr)
    finite_vals = []
    for i in range(n):
        if np.isfinite(arr[i]):
            finite_vals.append(arr[i])
    if len(finite_vals) < 3:
        for i in range(n):
            out[i] = 0.0
        return
    sorted_vals = sorted(finite_vals)
    n_finite = len(sorted_vals)
    if n_finite % 2 == 0:
        median_val = (sorted_vals[n_finite // 2 - 1] + sorted_vals[n_finite // 2]) / 2
    else:
        median_val = sorted_vals[n_finite // 2]
    abs_devs = []
    for val in finite_vals:
        abs_devs.append(abs(val - median_val))
    sorted_abs_devs = sorted(abs_devs)
    n_abs = len(sorted_abs_devs)
    if n_abs % 2 == 0:
        mad_val = (sorted_abs_devs[n_abs // 2 - 1] + sorted_abs_devs[n_abs // 2]) / 2
    else:
        mad_val = sorted_abs_devs[n_abs // 2]
    if mad_val < 1e-10:
        mad_val = np.std(np.array(finite_vals)) * 0.6745
    lower_bound = median_val - 3 * mad_val
    upper_bound = median_val + 3 * mad_val
    for i in range(n):
        if np.isnan(arr[i]):
            out[i] = median_val
        elif np.isinf(arr[i]):
            if arr[i] > 0:
                out[i] = upper_bound
            else:
                out[i] = lower_bound
        else:
            if arr[i] < lower_bound:
                out[i] = lower_bound
            elif arr[i] > upper_bound:
                out[i] = upper_bound
            else:
                out[i] = arr[i]


class CalculationEngine:
    """
    計算核心クラス（60%）- Project Forge統合版（修正版：ディスクベース垂直分割）
    責務:
    - Polars Expressionによる高度な特徴量計算（90%のタスク）
    - .map_batches()経由のNumba JIT最適化UDF（10%のカスタム「アルファ」ロジック）
    - 【修正】物理的垂直分割：ディスク中間ファイルによるメモリ・スラッシング回避
    - Polars内部並列化による自動マルチスレッド実行
    - 2段階品質保証（基本/フォールバック）
    """

    def __init__(self, config: ProcessingConfig):
        self.config = config
        self.qa = QualityAssurance()
        self.memory_monitor = MemoryMonitor(config.memory_limit_gb)
        self.prefix = f"e{config.engine_id.replace('e', '')}_"  # 例: "e1a_"

        # 【修正】ディスクベース垂直分割用の一時ディレクトリ
        self.temp_dir = Path(tempfile.mkdtemp(prefix=f"features_{config.engine_id}_"))
        logger.info(f"一時ディレクトリ作成: {self.temp_dir}")

    def _get_all_feature_expressions(self) -> Dict[str, pl.Expr]:
        """
        全特徴量の名前とPolars Expressionの対応辞書を返すファクトリメソッド。
        これにより、必要な特徴量だけを効率的に計算できるようになる。
        【特徴量ファクトリパターン】

        改修対象スクリプト（Engine 1A）の特徴量計算ロジックを移植
        """
        expressions = {}
        p = self.prefix

        # === 改修対象スクリプトから特徴量定義を移植 ===

        # --- Moments, Skew, Kurtosis Features ---
        for window in [10, 20, 50]:
            expressions[f"{p}statistical_mean_{window}"] = (
                pl.col("close")
                .rolling_mean(window)
                .alias(f"{p}statistical_mean_{window}")
            )
            expressions[f"{p}statistical_variance_{window}"] = (
                pl.col("close")
                .rolling_var(window)
                .alias(f"{p}statistical_variance_{window}")
            )
            expressions[f"{p}statistical_std_{window}"] = (
                pl.col("close")
                .rolling_std(window)
                .alias(f"{p}statistical_std_{window}")
            )
            expressions[f"{p}statistical_cv_{window}"] = (
                pl.col("close").rolling_std(window)
                / pl.col("close").rolling_mean(window)
            ).alias(f"{p}statistical_cv_{window}")

        for window in [20, 50]:
            expressions[f"{p}statistical_skewness_{window}"] = (
                pl.col("close")
                .rolling_skew(window_size=window)
                .alias(f"{p}statistical_skewness_{window}")
            )
            expressions[f"{p}statistical_kurtosis_{window}"] = (
                (pl.col("close") - pl.col("close").rolling_mean(window))
                .pow(4)
                .rolling_mean(window)
                / pl.col("close").rolling_std(window).pow(4)
                - 3
            ).alias(f"{p}statistical_kurtosis_{window}")
            for moment in [5, 6, 7, 8]:
                mean_col = pl.col("close").rolling_mean(window)
                std_col = pl.col("close").rolling_std(window)
                expr = (
                    ((pl.col("close") - mean_col) / std_col)
                    .pow(moment)
                    .rolling_mean(window)
                )
                expressions[f"{p}statistical_moment_{moment}_{window}"] = expr.alias(
                    f"{p}statistical_moment_{moment}_{window}"
                )

        # --- Robust Statistics Features ---
        for window in [10, 20, 50]:
            expressions[f"{p}robust_median_{window}"] = (
                pl.col("close")
                .rolling_median(window)
                .alias(f"{p}robust_median_{window}")
            )
            q25 = pl.col("close").rolling_quantile(0.25, window_size=window)
            q75 = pl.col("close").rolling_quantile(0.75, window_size=window)
            expressions[f"{p}robust_q25_{window}"] = q25.alias(
                f"{p}robust_q25_{window}"
            )
            expressions[f"{p}robust_q75_{window}"] = q75.alias(
                f"{p}robust_q75_{window}"
            )
            expressions[f"{p}robust_iqr_{window}"] = (q75 - q25).alias(
                f"{p}robust_iqr_{window}"
            )
            expressions[f"{p}robust_trimmed_mean_{window}"] = (
                pl.col("close")
                .rolling_map(
                    lambda s: trim_mean(s.to_numpy(), proportiontocut=0.1),
                    window_size=window,
                )
                .alias(f"{p}robust_trimmed_mean_{window}")
            )

        # --- Advanced Robust Features ---
        expressions[f"{p}robust_mad_20"] = (
            pl.col("close")
            .map_batches(
                lambda s: mad_rolling_numba(s.to_numpy()), return_dtype=pl.Float64
            )
            .alias(f"{p}robust_mad_20")
        )
        expressions[f"{p}robust_biweight_location_20"] = (
            pl.col("close")
            .map_batches(
                lambda s: biweight_location_numba(s.to_numpy()), return_dtype=pl.Float64
            )
            .alias(f"{p}robust_biweight_location_20")
        )
        expressions[f"{p}robust_winsorized_mean_20"] = (
            pl.col("close")
            .map_batches(
                lambda s: winsorized_mean_numba(s.to_numpy()), return_dtype=pl.Float64
            )
            .alias(f"{p}robust_winsorized_mean_20")
        )

        # --- Statistical Tests Features ---
        expressions[f"{p}jarque_bera_statistic_50"] = (
            pl.col("close")
            .pct_change()
            .map_batches(
                lambda s: jarque_bera_statistic_numba(s.to_numpy()),
                return_dtype=pl.Float64,
            )
            .alias(f"{p}jarque_bera_statistic_50")
        )
        expressions[f"{p}anderson_darling_statistic_30"] = (
            pl.col("close")
            .pct_change()
            .map_batches(
                lambda s: anderson_darling_numba(s.to_numpy()), return_dtype=pl.Float64
            )
            .alias(f"{p}anderson_darling_statistic_30")
        )
        expressions[f"{p}runs_test_statistic_30"] = (
            pl.col("close")
            .pct_change()
            .map_batches(
                lambda s: runs_test_numba(s.to_numpy()), return_dtype=pl.Float64
            )
            .alias(f"{p}runs_test_statistic_30")
        )
        expressions[f"{p}von_neumann_ratio_30"] = (
            pl.col("close")
            .pct_change()
            .map_batches(
                lambda s: von_neumann_ratio_numba(s.to_numpy()),
                return_dtype=pl.Float64,
            )
            .alias(f"{p}von_neumann_ratio_30")
        )

        # --- Basic Processing / Fast Features ---
        for window in [5, 10, 20, 50, 100]:
            expressions[f"{p}fast_rolling_mean_{window}"] = (
                pl.col("close")
                .rolling_mean(window)
                .alias(f"{p}fast_rolling_mean_{window}")
            )
            expressions[f"{p}fast_rolling_std_{window}"] = (
                pl.col("close")
                .rolling_std(window)
                .alias(f"{p}fast_rolling_std_{window}")
            )
            expressions[f"{p}fast_volume_mean_{window}"] = (
                pl.col("volume")
                .rolling_mean(window)
                .alias(f"{p}fast_volume_mean_{window}")
            )

        # --- Numba Optimized Basic Features ---
        expressions[f"{p}fast_rolling_mean_numba_20"] = (
            pl.col("close")
            .map_batches(
                lambda s: fast_rolling_mean_numba(s.to_numpy()), return_dtype=pl.Float64
            )
            .alias(f"{p}fast_rolling_mean_numba_20")
        )
        expressions[f"{p}fast_rolling_std_numba_20"] = (
            pl.col("close")
            .map_batches(
                lambda s: fast_rolling_std_numba(s.to_numpy()), return_dtype=pl.Float64
            )
            .alias(f"{p}fast_rolling_std_numba_20")
        )
        expressions[f"{p}fast_quality_score_50"] = (
            pl.col("close")
            .map_batches(
                lambda s: fast_quality_score_numba(s.to_numpy()),
                return_dtype=pl.Float64,
            )
            .alias(f"{p}fast_quality_score_50")
        )

        # --- Quality Assurance Features ---
        expressions[f"{p}fast_basic_stabilization"] = (
            pl.col("close")
            .map_batches(
                lambda s: basic_stabilization_numba(s.to_numpy()),
                return_dtype=pl.Float64,
            )
            .alias(f"{p}fast_basic_stabilization")
        )
        expressions[f"{p}fast_robust_stabilization"] = (
            pl.col("close")
            .map_batches(
                lambda s: robust_stabilization_numba(s.to_numpy()),
                return_dtype=pl.Float64,
            )
            .alias(f"{p}fast_robust_stabilization")
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
            if group_name == "group_1_moments":
                group_result_lf = self._create_basic_stats_features(lazy_frame)
            elif group_name == "group_2_skew_kurt":
                group_result_lf = self._create_skew_kurt_features(lazy_frame)
            elif group_name == "group_3_robust":
                group_result_lf = self._create_robust_stats_features(lazy_frame)
            elif group_name == "group_4_advanced":
                group_result_lf = self._create_advanced_robust_features(lazy_frame)
            elif group_name == "group_5_tests":
                group_result_lf = self._create_statistical_tests_features(lazy_frame)
            elif group_name == "group_6_fast":
                group_result_lf = self._create_fast_processing_features(lazy_frame)
            elif group_name == "group_7_numba":
                group_result_lf = self._create_numba_features(lazy_frame)
            elif group_name == "group_8_qa":
                group_result_lf = self._create_qa_features(lazy_frame)
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

            # 品質保証を適用（このグループのみ）
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
            # 頑健なパーセンタイル計算のため、一旦Inf値をnullに変換
            col_expr = pl.col(col_name).map_batches(
                lambda s: s.replace([np.inf, -np.inf], None)
            )

            p01 = col_expr.quantile(0.01, interpolation="linear")
            p99 = col_expr.quantile(0.99, interpolation="linear")

            stabilized_col = (
                pl.col(col_name)
                .clip(lower_bound=p01, upper_bound=p99)
                .fill_null(0.0)
                .fill_nan(0.0)
                .alias(col_name)
            )

            stabilization_exprs.append(stabilized_col)

        result = lazy_frame.with_columns(stabilization_exprs)
        return result

    def _create_vertical_slices(self) -> Dict[str, Dict[str, pl.Expr]]:
        """物理的垂直分割: 特徴量を論理グループに分割（Engine 1Aの特徴量に合わせて再構成）"""
        all_expressions = self._get_all_feature_expressions()

        # メモリ使用量を考慮したグルーピング（改修対象スクリプトに基づく）
        slices = {}
        p = self.prefix

        # グループ1: 統計的モーメント特徴量（軽量）
        slices["group_1_moments"] = {
            name: expr
            for name, expr in all_expressions.items()
            if any(
                pattern in name
                for pattern in [
                    "statistical_mean",
                    "statistical_variance",
                    "statistical_std",
                    "statistical_cv",
                ]
            )
        }

        # グループ2: 歪度・尖度（中程度）
        slices["group_2_skew_kurt"] = {
            name: expr
            for name, expr in all_expressions.items()
            if any(
                pattern in name
                for pattern in [
                    "statistical_skewness",
                    "statistical_kurtosis",
                    "statistical_moment",
                ]
            )
        }

        # グループ3: ロバスト統計（中程度）
        slices["group_3_robust"] = {
            name: expr
            for name, expr in all_expressions.items()
            if any(
                pattern in name
                for pattern in [
                    "robust_median",
                    "robust_q25",
                    "robust_q75",
                    "robust_iqr",
                    "robust_trimmed_mean",
                ]
            )
        }

        # グループ4: 高度ロバスト統計（重い）
        slices["group_4_advanced"] = {
            name: expr
            for name, expr in all_expressions.items()
            if any(
                pattern in name
                for pattern in [
                    "robust_mad",
                    "robust_biweight_location",
                    "robust_winsorized_mean",
                ]
            )
        }

        # グループ5: 統計検定（重い）
        slices["group_5_tests"] = {
            name: expr
            for name, expr in all_expressions.items()
            if any(
                pattern in name
                for pattern in [
                    "jarque_bera",
                    "anderson_darling",
                    "runs_test",
                    "von_neumann_ratio",
                ]
            )
        }

        # グループ6: 高速処理特徴量（軽量）
        slices["group_6_fast"] = {
            name: expr
            for name, expr in all_expressions.items()
            if any(
                pattern in name
                for pattern in [
                    "fast_rolling_mean",
                    "fast_rolling_std",
                    "fast_volume_mean",
                ]
            )
            and "numba" not in name
        }

        # グループ7: Numba最適化（中程度）
        slices["group_7_numba"] = {
            name: expr
            for name, expr in all_expressions.items()
            if any(
                pattern in name
                for pattern in [
                    "fast_rolling_mean_numba",
                    "fast_rolling_std_numba",
                    "fast_quality_score",
                ]
            )
        }

        # グループ8: 品質保証（軽量）
        slices["group_8_qa"] = {
            name: expr
            for name, expr in all_expressions.items()
            if any(
                pattern in name
                for pattern in ["fast_basic_stabilization", "fast_robust_stabilization"]
            )
        }

        # 分割されなかった特徴量があれば警告
        total_assigned = sum(len(group) for group in slices.values())
        if total_assigned != len(all_expressions):
            logger.warning(f"未分割特徴量: {len(all_expressions) - total_assigned}個")

        return slices

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
            # 頑健なパーセンタイル計算のため、一旦Inf値をnullに変換
            col_expr = pl.col(col_name).map_batches(
                lambda s: s.replace([np.inf, -np.inf], None)
            )

            p01 = col_expr.quantile(0.01, interpolation="linear")
            p99 = col_expr.quantile(0.99, interpolation="linear")

            stabilized_col = (
                pl.col(col_name)
                .clip(lower_bound=p01, upper_bound=p99)
                .fill_null(0.0)
                .fill_nan(0.0)
                .alias(col_name)
            )

            stabilization_exprs.append(stabilized_col)

        result = lazy_frame.with_columns(stabilization_exprs)
        logger.info("品質保証システム適用完了")

        return result

    # === 改修対象スクリプトの特徴量計算メソッドを移植 ===

    def _create_basic_stats_features(self, lazy_frame: pl.LazyFrame) -> pl.LazyFrame:
        """統計的モーメント特徴量の計算（高速化対応）"""
        exprs = []
        p = self.prefix

        # 基本統計系 - 全ての式に明示的なaliasを付与
        for window in [10, 20, 50]:
            exprs.append(
                pl.col("close")
                .rolling_mean(window)
                .alias(f"{p}statistical_mean_{window}")
            )
            exprs.append(
                pl.col("close")
                .rolling_var(window)
                .alias(f"{p}statistical_variance_{window}")
            )
            exprs.append(
                pl.col("close")
                .rolling_std(window)
                .alias(f"{p}statistical_std_{window}")
            )
            exprs.append(
                (
                    pl.col("close").rolling_std(window)
                    / pl.col("close").rolling_mean(window)
                ).alias(f"{p}statistical_cv_{window}")
            )

        return lazy_frame.with_columns(exprs)

    def _create_skew_kurt_features(self, lazy_frame: pl.LazyFrame) -> pl.LazyFrame:
        """歪度・尖度特徴量の計算"""
        exprs = []
        p = self.prefix

        for window in [20, 50]:
            exprs.append(
                pl.col("close")
                .rolling_skew(window_size=window)
                .alias(f"{p}statistical_skewness_{window}")
            )
            exprs.append(
                (
                    (pl.col("close") - pl.col("close").rolling_mean(window))
                    .pow(4)
                    .rolling_mean(window)
                    / pl.col("close").rolling_std(window).pow(4)
                    - 3
                ).alias(f"{p}statistical_kurtosis_{window}")
            )

            # 高次モーメント
            mean_col = pl.col("close").rolling_mean(window)
            std_col = pl.col("close").rolling_std(window)
            for moment in [5, 6, 7, 8]:
                expr = (
                    ((pl.col("close") - mean_col) / std_col)
                    .pow(moment)
                    .rolling_mean(window)
                )
                exprs.append(expr.alias(f"{p}statistical_moment_{moment}_{window}"))

        return lazy_frame.with_columns(exprs)

    def _create_robust_stats_features(self, lazy_frame: pl.LazyFrame) -> pl.LazyFrame:
        """ロバスト統計特徴量の計算"""
        exprs = []
        p = self.prefix

        for window in [10, 20, 50]:
            exprs.append(
                pl.col("close")
                .rolling_median(window)
                .alias(f"{p}robust_median_{window}")
            )

            q25_expr = pl.col("close").rolling_quantile(0.25, window_size=window)
            q75_expr = pl.col("close").rolling_quantile(0.75, window_size=window)

            exprs.extend(
                [
                    q25_expr.alias(f"{p}robust_q25_{window}"),
                    q75_expr.alias(f"{p}robust_q75_{window}"),
                    (q75_expr - q25_expr).alias(f"{p}robust_iqr_{window}"),
                    pl.col("close")
                    .rolling_map(
                        lambda s: trim_mean(s.to_numpy(), proportiontocut=0.1),
                        window_size=window,
                    )
                    .alias(f"{p}robust_trimmed_mean_{window}"),
                ]
            )

        return lazy_frame.with_columns(exprs)

    def _create_advanced_robust_features(
        self, lazy_frame: pl.LazyFrame
    ) -> pl.LazyFrame:
        """高度ロバスト統計特徴量の計算（Numba UDF使用）"""
        exprs = []
        p = self.prefix

        # MAD（中央絶対偏差）のNumba実装
        exprs.append(
            pl.col("close")
            .map_batches(
                lambda s: mad_rolling_numba(s.to_numpy()), return_dtype=pl.Float64
            )
            .alias(f"{p}robust_mad_20")
        )

        # Biweight位置のNumba実装
        exprs.append(
            pl.col("close")
            .map_batches(
                lambda s: biweight_location_numba(s.to_numpy()), return_dtype=pl.Float64
            )
            .alias(f"{p}robust_biweight_location_20")
        )

        # ウィンソライズ平均のNumba実装
        exprs.append(
            pl.col("close")
            .map_batches(
                lambda s: winsorized_mean_numba(s.to_numpy()), return_dtype=pl.Float64
            )
            .alias(f"{p}robust_winsorized_mean_20")
        )

        return lazy_frame.with_columns(exprs)

    def _create_statistical_tests_features(
        self, lazy_frame: pl.LazyFrame
    ) -> pl.LazyFrame:
        """統計検定・正規性特徴量の計算（Numba UDF使用）"""
        exprs = []
        p = self.prefix

        # 全統計検定特徴量をLazyFrameに追加
        # pct_change()でリターン系列に変換してから入力
        exprs.extend(
            [
                pl.col("close")
                .pct_change()
                .map_batches(
                    lambda s: jarque_bera_statistic_numba(s.to_numpy()),
                    return_dtype=pl.Float64,
                )
                .alias(f"{p}jarque_bera_statistic_50"),
                pl.col("close")
                .pct_change()
                .map_batches(
                    lambda s: anderson_darling_numba(s.to_numpy()),
                    return_dtype=pl.Float64,
                )
                .alias(f"{p}anderson_darling_statistic_30"),
                pl.col("close")
                .pct_change()
                .map_batches(
                    lambda s: runs_test_numba(s.to_numpy()), return_dtype=pl.Float64
                )
                .alias(f"{p}runs_test_statistic_30"),
                pl.col("close")
                .pct_change()
                .map_batches(
                    lambda s: von_neumann_ratio_numba(s.to_numpy()),
                    return_dtype=pl.Float64,
                )
                .alias(f"{p}von_neumann_ratio_30"),
            ]
        )

        return lazy_frame.with_columns(exprs)

    def _create_fast_processing_features(
        self, lazy_frame: pl.LazyFrame
    ) -> pl.LazyFrame:
        """基本データ処理特徴量（高速実装）"""
        exprs = []
        p = self.prefix

        # 高速ローリング平均（複数ウィンドウ）
        for window in [5, 10, 20, 50, 100]:
            # 高速ローリング平均
            exprs.append(
                pl.col("close")
                .rolling_mean(window)
                .alias(f"{p}fast_rolling_mean_{window}")
            )

            # 高速ローリング標準偏差
            exprs.append(
                pl.col("close")
                .rolling_std(window)
                .alias(f"{p}fast_rolling_std_{window}")
            )

            # 出来高の高速ローリング平均
            exprs.append(
                pl.col("volume")
                .rolling_mean(window)
                .alias(f"{p}fast_volume_mean_{window}")
            )

        return lazy_frame.with_columns(exprs)

    def _create_numba_features(self, lazy_frame: pl.LazyFrame) -> pl.LazyFrame:
        """Numba最適化特徴量の計算"""
        exprs = []
        p = self.prefix

        # Numba最適化版の特徴量
        exprs.extend(
            [
                # カスタム高速ローリング平均（Numba実装）
                pl.col("close")
                .map_batches(
                    lambda s: fast_rolling_mean_numba(s.to_numpy()),
                    return_dtype=pl.Float64,
                )
                .alias(f"{p}fast_rolling_mean_numba_20"),
                # カスタム高速ローリング標準偏差（Numba実装）
                pl.col("close")
                .map_batches(
                    lambda s: fast_rolling_std_numba(s.to_numpy()),
                    return_dtype=pl.Float64,
                )
                .alias(f"{p}fast_rolling_std_numba_20"),
                # 高速品質スコア
                pl.col("close")
                .map_batches(
                    lambda s: fast_quality_score_numba(s.to_numpy()),
                    return_dtype=pl.Float64,
                )
                .alias(f"{p}fast_quality_score_50"),
            ]
        )

        return lazy_frame.with_columns(exprs)

    def _create_qa_features(self, lazy_frame: pl.LazyFrame) -> pl.LazyFrame:
        """品質保証特徴量の計算"""
        exprs = []
        p = self.prefix

        # 品質保証特徴量
        exprs.extend(
            [
                pl.col("close")
                .map_batches(
                    lambda s: basic_stabilization_numba(s.to_numpy()),
                    return_dtype=pl.Float64,
                )
                .alias(f"{p}fast_basic_stabilization"),
                pl.col("close")
                .map_batches(
                    lambda s: robust_stabilization_numba(s.to_numpy()),
                    return_dtype=pl.Float64,
                )
                .alias(f"{p}fast_robust_stabilization"),
            ]
        )

        return lazy_frame.with_columns(exprs)


class OutputEngine:
    """
    出力管理クラス（10%）- Project Forge準拠
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
        return Path(self.config.output_path) / filename

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
            "engine_name": "Engine_1A_Basic_Statistical_Refactored",
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
        Path(config.output_path) / f"features_{config.engine_id}_{timeframe}/"
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
                    augmented_df.lazy(), group_name, group_expressions
                )

                # 2. 親方が自らメモリに実現化（単一グループなので安全）
                group_result_df = group_result_lf.collect(
                    streaming=True
                )  # メモリ効率のためstreaming=Trueを追加
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
                    pl.col("timestamp").dt.year().alias("year"),
                    pl.col("timestamp").dt.month().alias("month"),
                    pl.col("timestamp").dt.day().alias("day"),
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
    """単一の通常時間足を処理する（従来のロジック）"""
    logger.info(f"=== 通常処理開始: timeframe={timeframe} ===")
    start_time = time.time()

    try:
        data_engine = DataEngine(config)
        calc_engine = CalculationEngine(config)
        output_engine = OutputEngine(config)

        lazy_frame = data_engine.create_lazy_frame(timeframe)
        summary = data_engine.get_data_summary(lazy_frame)
        logger.info(f"データサマリー: {summary}")

        # 全特徴量を一括計算（通常時間足では垂直分割不要）
        features_lf = _calculate_all_features_unified(calc_engine, lazy_frame)
        processed_lf = output_engine.apply_nan_filling(features_lf)
        metadata = output_engine.save_features(processed_lf, timeframe)

        elapsed_time = time.time() - start_time
        metadata["processing_time"] = elapsed_time

        logger.info(f"=== 通常処理完了: {timeframe} - {elapsed_time:.2f}秒 ===")
        return metadata

    except Exception as e:
        logger.error(f"タイムフレーム {timeframe} の処理中にエラー: {e}", exc_info=True)
        return {"timeframe": timeframe, "error": str(e)}


def _calculate_all_features_unified(calc_engine, lazy_frame):
    """通常時間足用の統合特徴量計算"""
    # 各グループの特徴量を順次計算
    result_lf = calc_engine._create_basic_stats_features(lazy_frame)
    result_lf = calc_engine._create_skew_kurt_features(result_lf)
    result_lf = calc_engine._create_robust_stats_features(result_lf)
    result_lf = calc_engine._create_advanced_robust_features(result_lf)
    result_lf = calc_engine._create_statistical_tests_features(result_lf)
    result_lf = calc_engine._create_fast_processing_features(result_lf)
    result_lf = calc_engine._create_numba_features(result_lf)
    result_lf = calc_engine._create_qa_features(result_lf)

    # 品質保証適用
    result_lf = calc_engine.apply_quality_assurance(result_lf)

    return result_lf


# インタラクティブモード（Project Forge準拠）
def get_user_confirmation(config: ProcessingConfig) -> bool:
    """ユーザー確認 - Project Forge理念表示"""
    print("\n" + "=" * 60)
    print(f"Engine {config.engine_id.upper()} - {config.engine_name}")
    print("=" * 60)
    print("🎯 Project Forge - 軍資金増大プロジェクト")
    print("🚀 最終目標: Project Chimera開発・完成のための資金調達")
    print("💻 探索対象: マーケットの亡霊（統計的に有意で非ランダムな微細パターン）")
    print("🏆 思想的継承: ジム・シモンズ（ルネサンス・テクノロジーズ）")
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
    print("  - 統計的モーメント特徴量")
    print("  - ロバスト統計特徴量")
    print("  - 高度ロバスト統計特徴量")
    print("  - 統計検定・正規性特徴量")
    print("  - 基本データ処理特徴量")
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
    print(f"  Engine 1A - Basic Statistical and Robust Statistical (刷新版) ")
    print("  Project Forge - 軍資金増大プロジェクト  ")
    print("=" * 70)
    print("🎯 目標: XAU/USD市場の統計的パターン抽出")
    print("🤖 AIの頭脳による普遍的法則発見")
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
