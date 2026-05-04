#!/usr/bin/env python3
"""
革新的な特徴量重厚集積スクリプト - Engine 1E: Signal Processing & Frequency Analysis Features
【新規特徴量】標準信号処理・周波数解析特徴量を実装

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

# -----------------------------------------------------------------------
# パス解決: blueprint → core_indicators の順で解決する
# ① まず親ディレクトリ (/workspace) を sys.path に追加して blueprint を解決
# ② 次に blueprint.CORE_DIR (/workspace/core) を追加して core_indicators を解決
# ③ blueprint が見つからない場合は相対パスの fallback を使用
# -----------------------------------------------------------------------
_parent_dir = str(Path(__file__).resolve().parents[1])  # /workspace
if _parent_dir not in sys.path:
    sys.path.append(_parent_dir)

import blueprint as config

# =============================================================================
# [Step 10] core_indicators — Single Source of Truth からのインポート
# =============================================================================
try:
    _core_dir = str(config.CORE_DIR)          # /workspace/core
    if _core_dir not in sys.path:
        sys.path.append(_core_dir)
except (AttributeError, ModuleNotFoundError):
    # blueprint が見つからない場合 (テスト環境など) は相対 fallback
    _fallback_core = str(Path(__file__).resolve().parent / "core")
    if _fallback_core not in sys.path:
        sys.path.append(_fallback_core)

from core_indicators import (
    # [ATR & VOLATILITY]
    calculate_atr_wilder,
    scale_by_atr,
    # [WEIGHT]
    calculate_sample_weight,
    # [DSP] — FFT・解析信号・スペクトル系
    numba_fft,
    numba_ifft,
    get_analytic_signal,
    spectral_centroid_udf,
    spectral_bandwidth_udf,
    spectral_rolloff_udf,
    spectral_flux_udf,
    spectral_flatness_udf,
    spectral_entropy_udf,
    # [DSP] — ウェーブレット系
    wavelet_energy_udf,
    wavelet_entropy_udf,
    # [DSP] — ヒルベルト変換系
    hilbert_amplitude_udf,
    hilbert_phase_var_udf,
    hilbert_phase_stability_udf,
    hilbert_freq_mean_udf,
    hilbert_freq_std_udf,
    # [DSP] — 音響系
    acoustic_power_udf,
    acoustic_frequency_udf,
)

import os, sys, time, warnings, json, logging, math, tempfile, datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, Any
from dataclasses import dataclass, field

# 数値計算・データ処理
import numpy as np
import polars as pl
import numba as nb
# ▼▼ [Step 10] guvectorize / scipy / pywt はこのファイルでは直接使用しない。
#    DSP UDF はすべて core_indicators 経由で提供される。
#    将来このファイル独自のUDFが必要になった場合のみ再インポートすること。
# from numba import guvectorize, float64, int64  # 廃止
# from scipy import stats                          # 廃止
# from scipy.stats import jarque_bera, anderson, shapiro, trim_mean  # 廃止
# from scipy.fft import fft, fftfreq              # 廃止
# from scipy.signal import hilbert                # 廃止
# import pywt                                     # 廃止

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
        "spectral": [64, 128, 256, 512],  # スペクトル解析用
        "wavelet": [32, 64, 128, 256],  # ウェーブレット解析用
        "hilbert": [50, 100, 200],  # ヒルベルト変換用
        "general": [10, 20, 50, 100],  # 一般用途
        "acoustic": [128, 256, 512],  # 音響解析用
    }


@dataclass
class ProcessingConfig:
    """処理設定 - Project Forge統合版"""

    # ▼▼ 修正前: input_path: str = str(config.S1_BASE_MULTITIMEFRAME)
    # ▼▼ 修正後: Blueprint v4 に合わせた正規のパスに変更
    input_path: str = str(config.S1_PROCESSED)
    partitioned_tick_path: str = str(config.S1_RAW_TICK_PARTITIONED)
    output_path: str = str(config.S2_FEATURES)

    # エンジン識別
    engine_id: str = "e1e"
    engine_name: str = "Engine_1E_SignalProcessingFeatures"

    # 並列処理（戦略的並列処理スロットリング）
    max_threads: int = 4

    # メモリ制限（64GB RAM制約）
    memory_limit_gb: float = 55.0
    memory_warning_gb: float = 50.0

    timeframes: List[str] = field(default_factory=get_default_timeframes)
    window_sizes: Dict[str, List[int]] = field(default_factory=get_default_window_sizes)

    # ▼▼ 新規追加: 各時間足における1日(24時間)あたりのバー本数マップ
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
    w_max: int = 512

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
                ] != pl.Datetime("us"):
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



# =============================================================================
# [Step 10 リファクタリング] DSP UDF 群は core_indicators.py から一括 import 済み
# =============================================================================
# 以下の関数はすべて core_indicators.[CATEGORY: DSP] セクションで定義されており、
# ファイル冒頭の import 文で取り込まれています。このファイルへの重複定義は廃止。
#
#   numba_fft, numba_ifft, get_analytic_signal
#   spectral_centroid_udf, spectral_bandwidth_udf, spectral_rolloff_udf
#   spectral_flux_udf, spectral_flatness_udf, spectral_entropy_udf
#   wavelet_energy_udf, wavelet_entropy_udf
#   hilbert_amplitude_udf, hilbert_phase_var_udf, hilbert_phase_stability_udf
#   hilbert_freq_mean_udf, hilbert_freq_std_udf
#   acoustic_power_udf, acoustic_frequency_udf
#
# 変更の恩恵:
#   - engine_1_E (学習側) と realtime_feature_engine_1E (本番側) が
#     同一の Numba 関数オブジェクトを参照するため、物理的に値の乖離が発生しない。
#   - fastmath=True (旧) → fastmath=False (core_indicators 統一値) に変更され、
#     浮動小数点の再現性が向上する。
# =============================================================================

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
        # ▼▼ 修正前: self.qa = QualityAssurance()
        # ▼▼ 修正後: クラス削除に伴いインスタンス生成を削除
        self.memory_monitor = MemoryMonitor(config.memory_limit_gb)
        self.prefix = (
            f"{config.engine_id}_"  # 修正箇所: シンプルで確実なプレフィックス生成に変更
        )

        # 【修正】ディスクベース垂直分割用の一時ディレクトリ
        self.temp_dir = Path(tempfile.mkdtemp(prefix=f"features_{config.engine_id}_"))
        logger.info(f"一時ディレクトリ作成: {self.temp_dir}")

    def _get_all_feature_expressions(self) -> Dict[str, pl.Expr]:
        """
        全特徴量の名前とPolars Expressionの対応辞書を返すファクトリメソッド。
        これにより、必要な特徴量だけを効率的に計算できるようになる。
        【特徴量ファクトリパターン】
        """
        expressions = {}
        p = self.prefix

        # スペクトル特徴量系 - 重量UDF（map_batches + prange）
        for window in self.config.window_sizes["spectral"]:
            # スペクトル重心（重量UDF）
            expressions[f"{p}spectral_centroid_{window}"] = (
                pl.col("close")
                .pct_change()
                .map_batches(
                    lambda s, w=window: spectral_centroid_udf(s.to_numpy(), w),
                    return_dtype=pl.Float64,
                )
                .alias(f"{p}spectral_centroid_{window}")
            )

            # スペクトル帯域幅（重量UDF）
            expressions[f"{p}spectral_bandwidth_{window}"] = (
                pl.col("close")
                .pct_change()
                .map_batches(
                    lambda s, w=window: spectral_bandwidth_udf(s.to_numpy(), w),
                    return_dtype=pl.Float64,
                )
                .alias(f"{p}spectral_bandwidth_{window}")
            )

            # スペクトルロールオフ（重量UDF）
            expressions[f"{p}spectral_rolloff_{window}"] = (
                pl.col("close")
                .pct_change()
                .map_batches(
                    lambda s, w=window: spectral_rolloff_udf(s.to_numpy(), w),
                    return_dtype=pl.Float64,
                )
                .alias(f"{p}spectral_rolloff_{window}")
            )

            # スペクトルフラックス（重量UDF）
            expressions[f"{p}spectral_flux_{window}"] = (
                pl.col("close")
                .pct_change()
                .map_batches(
                    lambda s, w=window: spectral_flux_udf(s.to_numpy(), w),
                    return_dtype=pl.Float64,
                )
                .alias(f"{p}spectral_flux_{window}")
            )

            # スペクトル平坦度（重量UDF）
            expressions[f"{p}spectral_flatness_{window}"] = (
                pl.col("close")
                .pct_change()
                .map_batches(
                    lambda s, w=window: spectral_flatness_udf(s.to_numpy(), w),
                    return_dtype=pl.Float64,
                )
                .alias(f"{p}spectral_flatness_{window}")
            )

            # スペクトルエントロピー（重量UDF）
            expressions[f"{p}spectral_entropy_{window}"] = (
                pl.col("close")
                .pct_change()
                .map_batches(
                    lambda s, w=window: spectral_entropy_udf(s.to_numpy(), w),
                    return_dtype=pl.Float64,
                )
                .alias(f"{p}spectral_entropy_{window}")
            )
        # スペクトルエネルギー（Polarsネイティブ）
        for window in self.config.window_sizes["spectral"]:
            expressions[f"{p}spectral_energy_{window}"] = (
                (pl.col("close").pct_change() ** 2)
                .rolling_sum(window)
                .alias(f"{p}spectral_energy_{window}")
            )

        # スペクトルピーク周波数（Polarsネイティブ - 近似）
        # ▼▼ 修正前: pl.col("close").pct_change().rolling_max(128)
        # ▼▼ 修正後: 期間ボラティリティで割ってスケール不変（Zスコア化）
        expressions[f"{p}spectral_peak_freq_128"] = (
            pl.col("close").pct_change().rolling_max(128)
            / (pl.col("close").pct_change().rolling_std(128, ddof=1) + 1e-10)
        ).alias(f"{p}spectral_peak_freq_128")

        # ウェーブレット特徴量系
        for window in self.config.window_sizes["wavelet"]:
            expressions[f"{p}wavelet_energy_{window}"] = (
                pl.col("close")
                .pct_change()
                .map_batches(
                    lambda s, w=window: wavelet_energy_udf(s.to_numpy(), w),
                    return_dtype=pl.Float64,
                )
                .alias(f"{p}wavelet_energy_{window}")
            )

            # ウェーブレット統計量（Polarsネイティブ）
            expressions[f"{p}wavelet_mean_{window}"] = (
                pl.col("close")
                .pct_change()
                .rolling_mean(window)
                .alias(f"{p}wavelet_mean_{window}")
            )
            # ▼▼ 修正後: ddof=1 の明示
            expressions[f"{p}wavelet_std_{window}"] = (
                pl.col("close")
                .pct_change()
                .rolling_std(window, ddof=1)
                .alias(f"{p}wavelet_std_{window}")
            )

        # ウェーブレットエントロピー（軽量UDF - rolling_map）
        expressions[f"{p}wavelet_entropy_64"] = (
            pl.col("close")
            .pct_change()
            .map_batches(
                lambda s: wavelet_entropy_udf(s.to_numpy(), 64),
                return_dtype=pl.Float64,
            )
            .alias(f"{p}wavelet_entropy_64")
        )

        # ヒルベルト変換特徴量系
        for window in self.config.window_sizes["hilbert"]:
            expressions[f"{p}hilbert_amplitude_{window}"] = (
                pl.col("close")
                .pct_change()
                .map_batches(
                    lambda s, w=window: hilbert_amplitude_udf(s.to_numpy(), w),
                    return_dtype=pl.Float64,
                )
                .alias(f"{p}hilbert_amplitude_{window}")
            )

        # ヒルベルト統計的特徴量（軽量UDF - rolling_map）
        # ▼▼ 修正前: pl.col("close").abs().rolling_mean(100)
        # ▼▼ 修正後: スケール依存を排除するため、価格変動率(pct_change)の絶対値に変更
        expressions[f"{p}hilbert_amp_mean_100"] = (
            pl.col("close")
            .pct_change()
            .abs()
            .rolling_mean(100)
            .alias(f"{p}hilbert_amp_mean_100")
        )
        expressions[f"{p}hilbert_amp_std_100"] = (
            pl.col("close")
            .pct_change()
            .abs()
            .rolling_std(100, ddof=1)
            .alias(f"{p}hilbert_amp_std_100")
        )

        # ヒルベルト変動係数（Polarsネイティブ）
        expressions[f"{p}hilbert_amp_cv_100"] = (
            pl.col("close").pct_change().abs().rolling_std(100, ddof=1)
            / (pl.col("close").pct_change().abs().rolling_mean(100) + 1e-10)
        ).alias(f"{p}hilbert_amp_cv_100")
        # 位相関連特徴量（UDF）
        expressions[f"{p}hilbert_phase_var_50"] = (
            pl.col("close")
            .pct_change()
            .map_batches(
                lambda s: hilbert_phase_var_udf(s.to_numpy(), 50),
                return_dtype=pl.Float64,
            )
            .alias(f"{p}hilbert_phase_var_50")
        )

        expressions[f"{p}hilbert_phase_stability_50"] = (
            pl.col("close")
            .pct_change()
            .map_batches(
                lambda s: hilbert_phase_stability_udf(s.to_numpy(), 50),
                return_dtype=pl.Float64,
            )
            .alias(f"{p}hilbert_phase_stability_50")
        )

        # 瞬時周波数特徴量（UDF）
        expressions[f"{p}hilbert_freq_mean_100"] = (
            pl.col("close")
            .pct_change()
            .map_batches(
                lambda s: hilbert_freq_mean_udf(s.to_numpy(), 100),
                return_dtype=pl.Float64,
            )
            .alias(f"{p}hilbert_freq_mean_100")
        )

        expressions[f"{p}hilbert_freq_std_100"] = (
            pl.col("close")
            .pct_change()
            .map_batches(
                lambda s: hilbert_freq_std_udf(s.to_numpy(), 100),
                return_dtype=pl.Float64,
            )
            .alias(f"{p}hilbert_freq_std_100")
        )

        # 周波数エネルギー比（Polarsネイティブ）
        # ▼▼ [Step 10] SMA rolling_mean(13) → calculate_atr_wilder(period=13) に統一
        # ▼▼ 修正前: atr_13_expr_hilbert = pl.max_horizontal(...).rolling_mean(13)
        # ▼▼ 修正後: core_indicators.calculate_atr_wilder を map_batches 経由で呼び出す
        atr_13_expr_hilbert = (
            pl.struct(["high", "low", "close"])
            .map_batches(
                lambda s: calculate_atr_wilder(
                    s.struct.field("high").to_numpy(),
                    s.struct.field("low").to_numpy(),
                    s.struct.field("close").to_numpy(),
                    13,
                ),
                return_dtype=pl.Float64,
            )
        )
        atr_13_pct_expr = atr_13_expr_hilbert / (pl.col("close") + 1e-10)

        expressions[f"{p}hilbert_freq_energy_ratio_100"] = (
            (pl.col("close").pct_change() ** 2).rolling_sum(100)
            / ((atr_13_pct_expr**2) * 100 + 1e-10)
        ).alias(f"{p}hilbert_freq_energy_ratio_100")

        # 音響特徴量系
        for window in self.config.window_sizes["acoustic"]:
            expressions[f"{p}acoustic_power_{window}"] = (
                pl.col("close")
                .pct_change()
                .map_batches(
                    lambda s, w=window: acoustic_power_udf(s.to_numpy(), w),
                    return_dtype=pl.Float64,
                )
                .alias(f"{p}acoustic_power_{window}")
            )

            # 音響周波数（重量UDF）
            expressions[f"{p}acoustic_frequency_{window}"] = (
                pl.col("close")
                .pct_change()
                .map_batches(
                    lambda s, w=window: acoustic_frequency_udf(s.to_numpy(), w),
                    return_dtype=pl.Float64,
                )
                .alias(f"{p}acoustic_frequency_{window}")
            )

        # ▼▼ [Step 10] SMA rolling_mean(100) → calculate_atr_wilder(period=100) に統一
        # ▼▼ インラインコメント: Polars CSEの代わりに map_batches 経由で Wilder ATR を呼び出す
        atr_100_expr = (
            pl.struct(["high", "low", "close"])
            .map_batches(
                lambda s: calculate_atr_wilder(
                    s.struct.field("high").to_numpy(),
                    s.struct.field("low").to_numpy(),
                    s.struct.field("close").to_numpy(),
                    100,
                ),
                return_dtype=pl.Float64,
            )
        )

        expressions[f"{p}signal_rms_50"] = (
            (pl.col("close").pct_change() ** 2)
            .rolling_mean(50)
            .sqrt()
            .alias(f"{p}signal_rms_50")
        )

        # ▼▼ 修正後: ATR100で除算し、期間を100本レンジと整合
        expressions[f"{p}signal_peak_to_peak_100"] = (
            (pl.col("close").rolling_max(100) - pl.col("close").rolling_min(100))
            / (atr_100_expr + 1e-10)
        ).alias(f"{p}signal_peak_to_peak_100")

        # ▼▼ 修正前: (pl.col("close").rolling_max(50).abs() / ...)
        # ▼▼ 修正後: close絶対値ではなく、pct_changeの絶対最大値ベースに変更
        expressions[f"{p}signal_crest_factor_50"] = (
            pl.col("close").pct_change().rolling_max(50).abs()
            / ((pl.col("close").pct_change() ** 2).rolling_mean(50).sqrt() + 1e-10)
        ).alias(f"{p}signal_crest_factor_50")

        return expressions

    def get_feature_groups(self) -> Dict[str, Dict[str, pl.Expr]]:
        """特徴量グループ定義を外部から取得可能にする"""
        return self._create_vertical_slices()

    # ▼▼ 修正前: def calculate_one_group(self, lazy_frame: pl.LazyFrame, group_name: str, group_expressions: Dict[str, pl.Expr]) -> pl.LazyFrame:
    # ▼▼ 修正後: timeframe 引数を追加
    def calculate_one_group(
        self,
        lazy_frame: pl.LazyFrame,
        group_name: str,
        group_expressions: Dict[str, pl.Expr],
        timeframe: str = "M1",
    ) -> pl.LazyFrame:
        """単一グループの特徴量のみを計算（高速化修正版）"""
        logger.info(f"グループ計算開始: {group_name}")

        # メモリ安全性チェック（必須）
        is_safe, message = self.memory_monitor.check_memory_safety()
        if not is_safe:
            raise MemoryError(f"メモリ不足のためグループ処理を中断: {message}")

        try:
            # グループ名に基づいて効率的な特徴量計算を実行
            if group_name == "spectral":
                group_result_lf = self._create_spectral_features(lazy_frame)
            elif group_name == "wavelet":
                group_result_lf = self._create_wavelet_features(lazy_frame)
            elif group_name == "hilbert":
                group_result_lf = self._create_hilbert_features(lazy_frame)
            elif group_name == "acoustic":
                group_result_lf = self._create_acoustic_features(lazy_frame)
            elif group_name == "signal_stats":
                group_result_lf = self._create_signal_stats_features(lazy_frame)
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
            base_columns = [
                "timestamp",
                "open",
                "high",
                "low",
                "close",
                "volume",
                "e1e_sample_weight",
            ]
            if "timeframe" in available_columns:
                base_columns.append("timeframe")

            # このグループの特徴量カラムのみを抽出
            # ▼▼ e1e_sample_weight は base_columns で管理するため除外する
            #    prefix ("e1e_") でフィルタすると e1e_sample_weight も拾われて重複するため
            group_feature_columns = [
                col for col in available_columns
                if col.startswith(self.prefix) and col != "e1e_sample_weight"
            ]
            select_columns = base_columns + group_feature_columns

            # 実際に存在するカラムのみを選択
            final_select_columns = [
                col for col in select_columns if col in available_columns
            ]
            group_final_lf = group_result_lf.select(final_select_columns)

            # ▼▼ 修正前: stabilized_lf = self.apply_quality_assurance_to_group(group_final_lf, group_feature_columns)
            # ▼▼ 修正後: timeframe 引数を渡す
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

    # ▼▼ 修正前: def apply_quality_assurance_to_group(self, lazy_frame: pl.LazyFrame, feature_columns: List[str]) -> pl.LazyFrame:
    # ▼▼ 修正後: timeframe 引数を追加
    def apply_quality_assurance_to_group(
        self,
        lazy_frame: pl.LazyFrame,
        feature_columns: List[str],
        timeframe: str = "M1",
    ) -> pl.LazyFrame:
        """単一グループに対する品質保証システムの適用"""
        if not feature_columns:
            return lazy_frame

        logger.info(f"品質保証適用: {len(feature_columns)}個の特徴量")

        # ▼▼ 新規追加: 時間足に応じた1日あたりのバー本数（EWMの半減期）を取得
        half_life = self.config.timeframe_bars_per_day.get(timeframe, 1440)

        stabilization_exprs = []
        for col_name in feature_columns:
            col_expr = pl.col(col_name)
            safe_col = pl.when(col_expr.is_infinite()).then(None).otherwise(col_expr)

            # ▼▼ 修正後: ハードコード(1440)を half_life に置換
            ewm_mean = safe_col.ewm_mean(
                half_life=half_life, adjust=False, ignore_nulls=True
            )
            ewm_std = safe_col.ewm_std(
                half_life=half_life, adjust=False, ignore_nulls=True
            )

            p01 = ewm_mean - 5.0 * ewm_std
            p99 = ewm_mean + 5.0 * ewm_std

            # Inf値を統計的に意味のある値（動的境界値）で置換
            stabilized_col = (
                pl.when(col_expr == float("inf"))
                .then(p99)  # +Infは動的上限値で置換
                .when(col_expr == float("-inf"))
                .then(p01)  # -Infは動的下限値で置換
                .otherwise(col_expr)
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

        # メモリ使用量を考慮したグルーピング（英語キー使用）
        slices = {}
        p = self.prefix

        # グループ1: スペクトル系（重量）
        slices["spectral"] = {
            name: expr
            for name, expr in all_expressions.items()
            if any(
                pattern in name
                for pattern in [
                    "spectral_centroid",
                    "spectral_bandwidth",
                    "spectral_rolloff",
                    "spectral_flux",
                    "spectral_flatness",
                    "spectral_entropy",
                    "spectral_energy",
                    "spectral_peak_freq",
                ]
            )
        }

        # グループ2: ウェーブレット系（重量）
        slices["wavelet"] = {
            name: expr
            for name, expr in all_expressions.items()
            if any(
                pattern in name
                for pattern in [
                    "wavelet_energy",
                    "wavelet_entropy",
                    "wavelet_mean",
                    "wavelet_std",
                ]
            )
        }

        # グループ3: ヒルベルト変換系（中程度）
        slices["hilbert"] = {
            name: expr
            for name, expr in all_expressions.items()
            if any(
                pattern in name
                for pattern in [
                    "hilbert_amplitude",
                    "hilbert_amp_",
                    "hilbert_phase",
                    "hilbert_freq",
                ]
            )
        }

        # グループ4: 音響系（中程度）
        slices["acoustic"] = {
            name: expr
            for name, expr in all_expressions.items()
            if any(
                pattern in name for pattern in ["acoustic_power", "acoustic_frequency"]
            )
        }

        # グループ5: 信号統計系（軽量）
        slices["signal_stats"] = {
            name: expr
            for name, expr in all_expressions.items()
            if any(
                pattern in name
                for pattern in [
                    "signal_rms",
                    "signal_peak_to_peak",
                    "signal_crest_factor",
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
        try:
            if self.temp_dir.exists():
                for temp_file in self.temp_dir.glob("*.parquet"):
                    temp_file.unlink()
                    logger.debug(f"一時ファイル削除: {temp_file}")
                self.temp_dir.rmdir()
                logger.info(f"一時ディレクトリ削除: {self.temp_dir}")
        except Exception as e:
            logger.warning(f"一時ファイルクリーンアップエラー: {e}")

    # ▼▼ 修正前: def apply_quality_assurance(self, lazy_frame: pl.LazyFrame) -> pl.LazyFrame:
    # ▼▼ 修正後: timeframe 引数を追加
    def apply_quality_assurance(
        self, lazy_frame: pl.LazyFrame, timeframe: str = "M1"
    ) -> pl.LazyFrame:
        """2段階品質保証システムの適用"""
        logger.info("品質保証システム適用開始: 全ての特徴量に安定化処理を適用します。")

        # スキーマからプレフィックスを持つ特徴量カラムを特定
        # ▼▼ e1e_sample_weight は QA（EWMクリッピング）対象外のため除外する
        schema = lazy_frame.collect_schema()
        feature_columns = [
            col for col in schema.names()
            if col.startswith(self.prefix) and col != "e1e_sample_weight"
        ]

        if not feature_columns:
            logger.warning("品質保証対象の特徴量が見つかりません。")
            return lazy_frame

        logger.info(f"品質保証対象: {len(feature_columns)}個の特徴量")

        # ▼▼ 新規追加: 時間足に応じた1日あたりのバー本数（EWMの半減期）を取得
        half_life = self.config.timeframe_bars_per_day.get(timeframe, 1440)

        stabilization_exprs = []
        for col_name in feature_columns:
            col_expr = pl.col(col_name)
            safe_col = pl.when(col_expr.is_infinite()).then(None).otherwise(col_expr)

            # ▼▼ 修正後: ハードコード(1440)を half_life に置換
            ewm_mean = safe_col.ewm_mean(
                half_life=half_life, adjust=False, ignore_nulls=True
            )
            ewm_std = safe_col.ewm_std(
                half_life=half_life, adjust=False, ignore_nulls=True
            )

            p01 = ewm_mean - 5.0 * ewm_std
            p99 = ewm_mean + 5.0 * ewm_std

            # Inf値を統計的に意味のある値（動的境界値）で置換
            stabilized_col = (
                pl.when(col_expr == float("inf"))
                .then(p99)  # +Infは動的上限値で置換
                .when(col_expr == float("-inf"))
                .then(p01)  # -Infは動的下限値で置換
                .otherwise(col_expr)
                .clip(lower_bound=p01, upper_bound=p99)
                .fill_null(0.0)
                .fill_nan(0.0)
                .alias(col_name)
            )

            stabilization_exprs.append(stabilized_col)

        result = lazy_frame.with_columns(stabilization_exprs)
        logger.info("品質保証システム適用完了")

        return result

    def _create_spectral_features(self, lazy_frame: pl.LazyFrame) -> pl.LazyFrame:
        """スペクトル系特徴量の計算（高速化対応）"""
        exprs = []
        p = self.prefix

        # スペクトル特徴量系 - 重量UDF（map_batches + prange）
        for window in self.config.window_sizes["spectral"]:
            # スペクトル重心（重量UDF）
            exprs.append(
                pl.col("close")
                .pct_change()
                .map_batches(
                    lambda s, w=window: spectral_centroid_udf(s.to_numpy(), w),
                    return_dtype=pl.Float64,
                )
                .alias(f"{p}spectral_centroid_{window}")
            )

            # スペクトル帯域幅（重量UDF）
            exprs.append(
                pl.col("close")
                .pct_change()
                .map_batches(
                    lambda s, w=window: spectral_bandwidth_udf(s.to_numpy(), w),
                    return_dtype=pl.Float64,
                )
                .alias(f"{p}spectral_bandwidth_{window}")
            )

            # スペクトルロールオフ（重量UDF）
            exprs.append(
                pl.col("close")
                .pct_change()
                .map_batches(
                    lambda s, w=window: spectral_rolloff_udf(s.to_numpy(), w),
                    return_dtype=pl.Float64,
                )
                .alias(f"{p}spectral_rolloff_{window}")
            )

            # スペクトルフラックス（重量UDF）
            exprs.append(
                pl.col("close")
                .pct_change()
                .map_batches(
                    lambda s, w=window: spectral_flux_udf(s.to_numpy(), w),
                    return_dtype=pl.Float64,
                )
                .alias(f"{p}spectral_flux_{window}")
            )

            # スペクトル平坦度（重量UDF）
            exprs.append(
                pl.col("close")
                .pct_change()
                .map_batches(
                    lambda s, w=window: spectral_flatness_udf(s.to_numpy(), w),
                    return_dtype=pl.Float64,
                )
                .alias(f"{p}spectral_flatness_{window}")
            )

            # スペクトルエントロピー（重量UDF）
            exprs.append(
                pl.col("close")
                .pct_change()
                .map_batches(
                    lambda s, w=window: spectral_entropy_udf(s.to_numpy(), w),
                    return_dtype=pl.Float64,
                )
                .alias(f"{p}spectral_entropy_{window}")
            )

        # スペクトルエネルギー（Polarsネイティブ）
        for window in self.config.window_sizes["spectral"]:
            exprs.append(
                (pl.col("close").pct_change() ** 2)
                .rolling_sum(window)
                .alias(f"{p}spectral_energy_{window}")
            )

        # スペクトルピーク周波数（Polarsネイティブ - 近似）
        # ▼▼ 修正後: 期間ボラティリティで割ってスケール不変（Zスコア化）
        exprs.append(
            (
                pl.col("close").pct_change().rolling_max(128)
                / (pl.col("close").pct_change().rolling_std(128, ddof=1) + 1e-10)
            ).alias(f"{p}spectral_peak_freq_128")
        )

        return lazy_frame.with_columns(exprs)

    def _create_wavelet_features(self, lazy_frame: pl.LazyFrame) -> pl.LazyFrame:
        """ウェーブレット系特徴量の計算（高速化対応）"""
        exprs = []
        p = self.prefix

        # ウェーブレット特徴量系
        for window in self.config.window_sizes["wavelet"]:
            # ウェーブレットエネルギー（重量UDF）
            exprs.append(
                pl.col("close")
                .pct_change()
                .map_batches(
                    lambda s, w=window: wavelet_energy_udf(s.to_numpy(), w),
                    return_dtype=pl.Float64,
                )
                .alias(f"{p}wavelet_energy_{window}")
            )

            # ウェーブレット統計量（Polarsネイティブ）
            exprs.append(
                pl.col("close")
                .pct_change()
                .rolling_mean(window)
                .alias(f"{p}wavelet_mean_{window}")
            )
            # ▼▼ 修正後: ddof=1 の明示
            exprs.append(
                pl.col("close")
                .pct_change()
                .rolling_std(window, ddof=1)
                .alias(f"{p}wavelet_std_{window}")
            )

        # ===== ここからが修正箇所 =====
        # ウェーブレットエントロピー（軽量UDF -> 重量UDFパターンに昇格）
        exprs.append(
            pl.col("close")
            .pct_change()
            .map_batches(
                lambda s: wavelet_entropy_udf(s.to_numpy(), 64), return_dtype=pl.Float64
            )
            .alias(f"{p}wavelet_entropy_64")
        )
        # ===== ここまでが修正箇所 =====

        return lazy_frame.with_columns(exprs)

    def _create_hilbert_features(self, lazy_frame: pl.LazyFrame) -> pl.LazyFrame:
        """ヒルベルト変換系特徴量の計算（高速化対応）"""
        exprs = []
        p = self.prefix

        # ヒルベルト変換特徴量系
        for window in self.config.window_sizes["hilbert"]:
            # ヒルベルト振幅包絡線（重量UDF）
            exprs.append(
                pl.col("close")
                .pct_change()
                # ▼▼ 修正箇所: return_dtype=pl.Float64 を追加
                .map_batches(
                    lambda s, w=window: hilbert_amplitude_udf(s.to_numpy(), w),
                    return_dtype=pl.Float64,
                )
                .alias(f"{p}hilbert_amplitude_{window}")
            )

        # ▼▼ 修正前: pl.col("close").abs().rolling_mean(100)
        # ▼▼ 修正後: pct_change() の絶対値に変更
        exprs.append(
            pl.col("close")
            .pct_change()
            .abs()
            .rolling_mean(100)
            .alias(f"{p}hilbert_amp_mean_100")
        )
        exprs.append(
            pl.col("close")
            .pct_change()
            .abs()
            .rolling_std(100, ddof=1)
            .alias(f"{p}hilbert_amp_std_100")
        )

        # ↓ 修正箇所1：Polarsネイティブ計算の分母に微小値を追加 + pct_change適用
        exprs.append(
            (
                pl.col("close").pct_change().abs().rolling_std(100, ddof=1)
                / (pl.col("close").pct_change().abs().rolling_mean(100) + 1e-10)
            ).alias(f"{p}hilbert_amp_cv_100")
        )

        # 位相関連特徴量（UDF）
        exprs.append(
            pl.col("close")
            .pct_change()
            # ▼▼ 修正箇所: return_dtype=pl.Float64 を追加
            .map_batches(
                lambda s: hilbert_phase_var_udf(s.to_numpy(), 50),
                return_dtype=pl.Float64,
            )
            .alias(f"{p}hilbert_phase_var_50")
        )

        exprs.append(
            pl.col("close")
            .pct_change()
            # ▼▼ 修正箇所: return_dtype=pl.Float64 を追加
            .map_batches(
                lambda s: hilbert_phase_stability_udf(s.to_numpy(), 50),
                return_dtype=pl.Float64,
            )
            .alias(f"{p}hilbert_phase_stability_50")
        )

        # 瞬時周波数特徴量（UDF）
        exprs.append(
            pl.col("close")
            .pct_change()
            # ▼▼ 修正箇所: return_dtype=pl.Float64 を追加
            .map_batches(
                lambda s: hilbert_freq_mean_udf(s.to_numpy(), 100),
                return_dtype=pl.Float64,
            )
            .alias(f"{p}hilbert_freq_mean_100")
        )

        exprs.append(
            pl.col("close")
            .pct_change()
            # ▼▼ 修正箇所: return_dtype=pl.Float64 を追加
            .map_batches(
                lambda s: hilbert_freq_std_udf(s.to_numpy(), 100),
                return_dtype=pl.Float64,
            )
            .alias(f"{p}hilbert_freq_std_100")
        )

        # 周波数エネルギー比（Polarsネイティブ）
        # ▼▼ [Step 10] SMA rolling_mean(13) → calculate_atr_wilder(period=13) に統一
        atr_13_expr_hilbert = (
            pl.struct(["high", "low", "close"])
            .map_batches(
                lambda s: calculate_atr_wilder(
                    s.struct.field("high").to_numpy(),
                    s.struct.field("low").to_numpy(),
                    s.struct.field("close").to_numpy(),
                    13,
                ),
                return_dtype=pl.Float64,
            )
        )
        atr_13_pct_expr = atr_13_expr_hilbert / (pl.col("close") + 1e-10)

        exprs.append(
            (
                (pl.col("close").pct_change() ** 2).rolling_sum(100)
                / ((atr_13_pct_expr**2) * 100 + 1e-10)
            ).alias(f"{p}hilbert_freq_energy_ratio_100")
        )

        return lazy_frame.with_columns(exprs)

    def _create_acoustic_features(self, lazy_frame: pl.LazyFrame) -> pl.LazyFrame:
        """音響系特徴量の計算（高速化対応）"""
        exprs = []
        p = self.prefix

        # 音響特徴量系
        for window in self.config.window_sizes["acoustic"]:
            # 音響パワー（重量UDF）
            exprs.append(
                pl.col("close")
                .pct_change()
                .map_batches(
                    lambda s, w=window: acoustic_power_udf(s.to_numpy(), w),
                    return_dtype=pl.Float64,
                )
                .alias(f"{p}acoustic_power_{window}")
            )

            # 音響周波数（重量UDF）
            exprs.append(
                pl.col("close")
                .pct_change()
                .map_batches(
                    lambda s, w=window: acoustic_frequency_udf(s.to_numpy(), w),
                    return_dtype=pl.Float64,
                )
                .alias(f"{p}acoustic_frequency_{window}")
            )

        return lazy_frame.with_columns(exprs)

    def _create_signal_stats_features(self, lazy_frame: pl.LazyFrame) -> pl.LazyFrame:
        """信号統計系特徴量の計算（高速化対応）"""
        exprs = []
        p = self.prefix

        # ▼▼ [Step 10] SMA rolling_mean(100) → calculate_atr_wilder(period=100) に統一
        atr_100_expr = (
            pl.struct(["high", "low", "close"])
            .map_batches(
                lambda s: calculate_atr_wilder(
                    s.struct.field("high").to_numpy(),
                    s.struct.field("low").to_numpy(),
                    s.struct.field("close").to_numpy(),
                    100,
                ),
                return_dtype=pl.Float64,
            )
        )

        # 追加の信号処理特徴量（Polarsネイティブ）
        exprs.append(
            (pl.col("close").pct_change() ** 2)
            .rolling_mean(50)
            .sqrt()
            .alias(f"{p}signal_rms_50")
        )

        # ▼▼ 修正後: ATR100によるスケール不変対応（100本レンジと期間整合）
        exprs.append(
            (
                (pl.col("close").rolling_max(100) - pl.col("close").rolling_min(100))
                / (atr_100_expr + 1e-10)
            ).alias(f"{p}signal_peak_to_peak_100")
        )

        # ▼▼ 修正後: pct_change() ベースによるスケール依存排除
        exprs.append(
            (
                pl.col("close").pct_change().rolling_max(50).abs()
                / ((pl.col("close").pct_change() ** 2).rolling_mean(50).sqrt() + 1e-10)
            ).alias(f"{p}signal_crest_factor_50")
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
        # ▼▼ 修正箇所: 正しいフォルダ名に変更
        subfolder_name = "feature_value_a_vast_universeE"
        return Path(self.config.output_path) / subfolder_name / filename

    def apply_nan_filling(self, lazy_frame: pl.LazyFrame) -> pl.LazyFrame:
        """NaN埋め統一処理"""
        logger.info("NaN埋め処理開始")

        # スキーマを一度だけ取得（メモリ安全）
        schema = lazy_frame.collect_schema()
        all_columns = schema.names()

        # プレフィックスを持つ特徴量カラムを特定
        # ▼▼ e1e_sample_weight は basic_columns で管理するため除外する
        feature_columns = [
            col for col in all_columns
            if col.startswith(f"{self.engine_id}_") and col != "e1e_sample_weight"
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
            "e1e_sample_weight",
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

            schema = processed_frame.collect_schema()
            cast_exprs = [
                pl.col("timestamp").cast(pl.Datetime("us")),
                pl.col("timeframe").cast(pl.Utf8),
            ]
            for col_name, dtype in schema.items():
                if dtype == pl.Categorical and col_name != "timeframe":
                    cast_exprs.append(pl.col(col_name).cast(pl.Utf8))
            if cast_exprs:
                processed_frame = processed_frame.with_columns(cast_exprs)

            start_time = time.time()

            # 【修正】sink_parquetの代わりにcollect + write_parquetを使用
            # struct型を使ったUDFの結果がsink_parquetと互換性がない問題を回避
            try:
                # まずストリーミングcollectを試行（新しいAPI使用）
                df = processed_frame.collect(engine="streaming")
            except Exception as streaming_error:
                logger.warning(
                    f"ストリーミングcollectが失敗、通常collectを使用: {streaming_error}"
                )
                df = processed_frame.collect()

            # DataFrameとして保存
            df.write_parquet(
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
            # ▼▼ 修正箇所: 1Dの名残を1Eに修正
            "engine_name": "Engine_1E_SignalProcessingFeatures",
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
    # ▼▼ 修正箇所: 正しいフォルダ名に変更
    subfolder_name = "feature_value_a_vast_universeE"
    FEATURES_ROOT = (
        Path(config.output_path)
        / subfolder_name
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

            # ▼▼ [Step 10] e1e_sample_weight を calculate_sample_weight (Wilder ATR統一版) で計算・付加
            # 意図: engine_1_D が先行して付加した列（SMA-ATRベース）であっても
            #       Wilder ATR統一後の値で必ず上書きする。
            sw = calculate_sample_weight(
                augmented_df["high"].to_numpy(),
                augmented_df["low"].to_numpy(),
                augmented_df["close"].to_numpy(),
            )
            augmented_df = augmented_df.with_columns(
                pl.Series("e1e_sample_weight", sw)
            )
            logging.info("[Step 10] e1e_sample_weight を calculate_sample_weight (Wilder ATR) で計算・上書きしました。")

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
                    augmented_df.lazy(),
                    group_name,
                    group_expressions,
                    timeframe=timeframe,
                )

                # 2. 親方が自らメモリに実現化（単一グループなので安全）
                # ▼▼ 修正箇所: streaming=True を engine="streaming" に変更
                group_result_df = group_result_lf.collect(
                    engine="streaming"
                )  # メモリ効率のためストリーミングAPIを使用
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
                "e1e_sample_weight",
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

            # NaN埋め処理を適用
            output_engine = OutputEngine(config)
            result_lf = output_engine.apply_nan_filling(result_df.lazy())
            result_df = result_lf.collect()

            # パーティション保存用の日付列を追加
            # ▼▼ 修正前: ただの dt.year() 等
            # ▼▼ 修正後: Int32への厳密なキャストを追加 (AthenaやSparkでの読み込みエラー防止)
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

    # calc_engineをtryブロックの外で初期化
    calc_engine = None
    try:
        data_engine = DataEngine(config)
        calc_engine = CalculationEngine(config)
        output_engine = OutputEngine(config)

        lazy_frame = data_engine.create_lazy_frame(timeframe)
        summary = data_engine.get_data_summary(lazy_frame)
        logger.info(f"データサマリー: {summary}")

        # ▼▼ 修正前: 全特徴量をメモリ上で一度に計算しようとしてOOM（Killed）を引き起こしていた
        # all_expressions = []
        # feature_groups = calc_engine.get_feature_groups()
        # for group_name, group_expressions in feature_groups.items():
        #     all_expressions.extend(group_expressions.values())
        #
        # logger.info(f"特徴量計算開始: {len(all_expressions)}個の特徴量を {timeframe} に対して計算します。")
        # features_lf = lazy_frame.with_columns(all_expressions)
        #
        # # 品質保証システムを適用
        # features_lf = calc_engine.apply_quality_assurance(features_lf, timeframe=timeframe)

        # ▼▼ 修正後: 通常時間足にも物理的垂直分割（グループ単位で計算・ディスク退避・結合）を適用
        import gc

        feature_groups = calc_engine.get_feature_groups()
        temp_dir = calc_engine.temp_dir / f"tf_{timeframe}"
        temp_dir.mkdir(parents=True, exist_ok=True)
        temp_files = []

        # 各グループ計算の土台とするため、ベースのOHLCVを一度実現化（300万行でもOHLCVのみなら数百MBで超軽量）
        base_df = lazy_frame.collect()

        # ▼▼ [Step 10] e1e_sample_weight を calculate_sample_weight (Wilder ATR統一版) で計算・付加
        # 意図: engine_1_D が先行して付加した列（SMA-ATRベース）であっても
        #       Wilder ATR統一後の値で必ず上書きする。
        #       if ガードは設けず、常に上書きすることで Single Source of Truth を保証する。
        sw = calculate_sample_weight(
            base_df["high"].to_numpy(),
            base_df["low"].to_numpy(),
            base_df["close"].to_numpy(),
        )
        base_df = base_df.with_columns(
            pl.Series("e1e_sample_weight", sw)
        )
        logger.info("[Step 10] e1e_sample_weight を calculate_sample_weight (Wilder ATR) で計算・上書きしました。")

        for group_idx, (group_name, group_expressions) in enumerate(
            feature_groups.items()
        ):
            logger.info(
                f"グループ処理開始: {group_name} ({len(group_expressions)}個の特徴量)"
            )

            # 1グループ（約10個の特徴量）だけを安全に計算し、内部で品質保証(QA)も適用
            group_result_lf = calc_engine.calculate_one_group(
                base_df.lazy(), group_name, group_expressions, timeframe=timeframe
            )

            group_result_df = group_result_lf.collect(engine="streaming")
            logger.info(f"グループデータ実現化完了: {group_result_df.height}行")

            # ディスクに退避
            temp_file = temp_dir / f"group_{group_idx:02d}_{group_name}.parquet"
            group_result_df.write_parquet(str(temp_file), compression="snappy")
            temp_files.append(temp_file)

            # 確実なメモリ解放
            del group_result_df
            gc.collect()

        # 全グループをディスクから読み出して水平結合
        logger.info("全グループ計算完了。ファイルの結合を開始します...")
        result_df = pl.read_parquet(str(temp_files[0]))
        base_columns = [
            "timestamp",
            "open",
            "high",
            "low",
            "close",
            "volume",
            "timeframe",
            "e1e_sample_weight",
        ]

        for temp_file in temp_files[1:]:
            next_df = pl.read_parquet(str(temp_file))
            feature_cols = [col for col in next_df.columns if col not in base_columns]
            if feature_cols:
                result_df = result_df.hstack(next_df.select(feature_cols))
            del next_df
            gc.collect()

        features_lf = result_df.lazy()

        # ▼▼ 次回以降のための追加箇所: 結合完了後に明示的にファイルを削除する ▼▼
        for temp_file in temp_files:
            if temp_file.exists():
                temp_file.unlink()
        if temp_dir.exists():
            temp_dir.rmdir()
        # ▲▲ 追加箇所ここまで ▲▲

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
    # ▼▼ 修正箇所: Engine 1E の正しい内容に変更
    print("  - スペクトル解析（重心・帯域幅・ロールオフ・フラックス・エントロピー）")
    print("  - ウェーブレット解析（Haar DWTエネルギー・エントロピー）")
    print("  - ヒルベルト変換（振幅包絡線・瞬時位相・瞬時周波数）")
    print("  - 音響特徴量（RMSパワー・ゼロクロッシング周波数）")
    print("  - 信号統計（Crest Factor・Peak-to-Peak・相対エネルギー比）")
    print("  - 2段階品質保証システム（1日周期 EWM動的5σクリッピング）")
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
    # ▼▼ 修正箇所: 1Eの正しいタイトルに変更
    print(f"  Engine 1E - Signal Processing & Frequency Analysis Features  ")
    print("  Project Forge - 軍資金増大プロジェクト  ")
    print("=" * 70)
    print("🎯 目標: XAU/USD市場の統計的パターン抽出")
    print("🤖 AI頭脳による普遍的法則発見")
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
