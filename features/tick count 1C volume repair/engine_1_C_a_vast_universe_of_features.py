#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Engine 1C: テクニカル指標・トレンド分析特徴量エンジン
Technical Indicators and Trend Analysis Features Engine

対象: XAU/USD (Gold vs US Dollar)

特徴量カテゴリ:
- 基本テクニカル指標 (RSI, MACD, BB, ATR, Stochastic)
- ADX・方向性指標
- トレンド・モメンタム指標
- 移動平均線・トレンド分析

ブロック1: インポート・設定・ヘルパー関数
"""

# ===== ブロック1開始 =====

# 標準ライブラリ
import os
import sys
import time
import warnings
import json
import logging
import math
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, Any
from dataclasses import dataclass, field

sys.path.append(str(Path(__file__).resolve().parent.parent))  # /workspace をパスに追加
import blueprint as config

sys.path.append(str(config.CORE_DIR))
from core_indicators import (
    calculate_atr_wilder,
    scale_by_atr,
    calculate_rsi_wilder,
    calculate_adx,
    calculate_sample_weight,
    # [SSoT 統一] Engine 1C の Numba 関数を core_indicators から import
    # (旧: 本ファイル内で定義 → 本番側 rfe_1C と二重定義 = SSoT 違反)
    calculate_wma_numba,
    calculate_hma_numba,
    calculate_kama_numba,
    calculate_stochastic_numba,
    calculate_williams_r_numba,
    calculate_trix_numba,
    calculate_ultimate_oscillator_numba,
    calculate_aroon_up_numba,
    calculate_aroon_down_numba,
    calculate_tsi_numba,
    _calculate_di_wilder,
)

# 外部ライブラリ（バージョン固定）
import numpy as np
import polars as pl
import numba as nb
from numba import guvectorize, float64, int64
from scipy import stats
from scipy.stats import jarque_bera, anderson, shapiro
import psutil

# ログ設定
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("engine_1b_technical.log", encoding="utf-8"),
    ],
)
logger = logging.getLogger(__name__)

# 警告フィルタ
warnings.filterwarnings("ignore", category=RuntimeWarning)


# ========================================
# 設定クラス
# ========================================
# ================================================================
# ===== ここから新規挿入ブロック =====
# ================================================================
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


# ================================================================
# ===== 新規挿入ブロックここまで =====
# ================================================================


@dataclass
class ProcessingConfig:
    """処理設定"""

    # データパス - config.pyから読み込む
    input_base_path: str = str(config.S1_PROCESSED)
    partitioned_tick_path: str = str(config.S1_RAW_TICK_PARTITIONED)
    # ▼▼ 修正前: output_base_path: str = str(config.S2_FEATURES / "feature_value_c_vast_universeC")
    # ▼▼ 修正後: 正しいディレクトリ名に修正
    output_base_path: str = str(config.S2_FEATURES / "feature_value_a_vast_universeC")

    # 各時間足の1日あたりのバー数定義
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
    # ▲▲ 追加ここまで

    # エンジン識別
    engine_id: str = "e1c"
    engine_name: str = "Engine_1C_Technical_Indicators"

    # 並列処理
    max_threads: int = 4

    # メモリ制限
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
        output_path_obj = Path(self.output_base_path)
        if not output_path_obj.exists():
            output_path_obj.mkdir(parents=True, exist_ok=True)
            logger.info(f"出力ディレクトリを作成: {output_path_obj}")
        return True


# ========================================
# メモリ監視
# ========================================


class MemoryMonitor:
    """メモリ使用量監視"""

    def __init__(self, limit_gb: float = 55.0, warning_gb: float = 50.0):
        self.limit_gb = limit_gb
        self.warning_gb = warning_gb
        self.process = psutil.Process()

    def get_memory_usage(self) -> Tuple[float, float]:
        """現在のメモリ使用量を取得"""
        memory_info = self.process.memory_info()
        used_gb = memory_info.rss / (1024**3)
        percent = psutil.virtual_memory().percent
        return used_gb, percent

    def check_memory_safety(self) -> Tuple[bool, str]:
        """メモリ安全性チェック"""
        used_gb, percent = self.get_memory_usage()

        if used_gb > self.limit_gb:
            message = f"メモリ使用量が制限値を超過: {used_gb:.2f}GB / {self.limit_gb:.2f}GB ({percent:.1f}%)"
            logger.error(message)
            return False, message

        if used_gb > self.warning_gb:
            message = f"警告: メモリ使用量が高い: {used_gb:.2f}GB / {self.limit_gb:.2f}GB ({percent:.1f}%)"
            logger.warning(message)
            return True, message

        message = (
            f"メモリ使用量: {used_gb:.2f}GB / {self.limit_gb:.2f}GB ({percent:.1f}%)"
        )
        return True, message


# ========================================
# Numba UDF関数（モジュールレベル）
# ========================================


# calculate_adx (core_indicators) は ADX のみを返す。
# DI+/DI- を同一の Wilder 平滑化アルゴリズムで計算するため、
# ここに calculate_adx 内部と完全一致したローカルヘルパーを定義する。
# ※ 将来 core_indicators に calculate_adx_full(returns ADX+DI+DI-) が追加された際は
#    このローカル関数を削除して移行すること。
# ========================================
# Numba UDF関数（続き）
# ========================================


# ========================================
# 【Phase 5 修正 (#37)】SSoT 統一作業の残骸を削除
# B 群の SSoT 統一作業 (engine_1_C → core_indicators への関数移植) で、
# 関数本体は core_indicators に移したが、@nb.guvectorize デコレータだけが
# 残ってしまっていた。これらが直後の class DataEngine に適用されてしまい、
# `TypeError: The decorated object is not a function (got type <class 'type'>)`
# でクラッシュしていた。該当の浮いたデコレータ 11 個をすべて削除する。
#
# 対応する関数 (すでに core_indicators にて定義済み):
#   - calculate_wma_numba, calculate_hma_numba, calculate_kama_numba,
#     calculate_stochastic_numba, calculate_williams_r_numba, calculate_trix_numba,
#     calculate_ultimate_oscillator_numba, calculate_aroon_up_numba,
#     calculate_aroon_down_numba, calculate_tsi_numba, _calculate_di_wilder
# ========================================


# ========================================
# DataEngine - データ基盤クラス
# ========================================


class DataEngine:
    """データ基盤エンジン"""

    def __init__(self, config: ProcessingConfig):
        """初期化"""
        self.config = config
        self.base_path = Path(config.input_base_path)

        logger.info(f"DataEngine初期化: {self.base_path}")

    def validate_data_source(self) -> bool:
        """データソース検証"""
        if not self.base_path.exists():
            logger.error(f"データソースが存在しません: {self.base_path}")
            return False

        # timeframeディレクトリの確認
        timeframe_dirs = [
            d
            for d in self.base_path.iterdir()
            if d.is_dir() and d.name.startswith("timeframe=")
        ]

        if not timeframe_dirs:
            logger.error("timeframeディレクトリが見つかりません")
            return False

        logger.info(f"検出されたタイムフレーム: {len(timeframe_dirs)}個")
        return True

    def get_parquet_paths(self, timeframe: str) -> List[Path]:
        """指定タイムフレームのParquetファイルパス取得"""
        timeframe_path = self.base_path / f"timeframe={timeframe}"

        if not timeframe_path.exists():
            logger.warning(
                f"タイムフレームディレクトリが存在しません: {timeframe_path}"
            )
            return []

        parquet_files = list(timeframe_path.glob("*.parquet"))

        if not parquet_files:
            logger.warning(f"Parquetファイルが見つかりません: {timeframe_path}")
            return []

        return sorted(parquet_files)

    def create_lazy_frame(self, timeframe: str) -> pl.LazyFrame:
        """LazyFrame生成"""
        parquet_paths = self.get_parquet_paths(timeframe)

        if not parquet_paths:
            raise ValueError(f"データファイルが見つかりません: timeframe={timeframe}")

        # scan_parquetでLazyFrame生成
        lazy_frame = pl.scan_parquet(parquet_paths)

        # timeframe列の手動復元
        lazy_frame = lazy_frame.with_columns([pl.lit(timeframe).alias("timeframe")])

        # テストモード対応
        if self.config.test_mode:
            lazy_frame = lazy_frame.head(self.config.test_rows)
            logger.info(f"テストモード: {self.config.test_rows}行に制限")

        return lazy_frame

    def get_data_summary(self, lazy_frame: pl.LazyFrame) -> Dict[str, Any]:
        """データ概要取得"""
        try:
            # スキーマ取得
            schema = lazy_frame.collect_schema()
            columns = schema.names()
            dtypes = {col: str(schema[col]) for col in columns}

            # 基本統計（小サンプル）
            sample_df = lazy_frame.head(1000).collect()

            summary = {
                "columns": columns,
                "dtypes": dtypes,
                "sample_rows": len(sample_df),
                "memory_usage_mb": sample_df.estimated_size("mb"),
            }

            return summary

        except Exception as e:
            logger.error(f"データ概要取得エラー: {e}")
            return {}

    def estimate_memory_usage(self, timeframe: str) -> float:
        """メモリ使用量推定（GB）"""
        parquet_paths = self.get_parquet_paths(timeframe)

        if not parquet_paths:
            return 0.0

        total_size = sum(p.stat().st_size for p in parquet_paths)

        # Parquet圧縮率を考慮した推定（約3倍）
        estimated_memory_gb = (total_size * 3) / (1024**3)

        return estimated_memory_gb


# ========================================
# OutputEngine - 出力管理クラス
# ========================================


class OutputEngine:
    """出力管理エンジン"""

    def __init__(self, config: ProcessingConfig):
        """初期化"""
        self.config = config
        self.output_path = Path(config.output_base_path)
        self.output_path.mkdir(parents=True, exist_ok=True)

        logger.info(f"OutputEngine初期化: {self.output_path}")

    def create_output_path(self, filename: str) -> Path:
        """出力パス生成"""
        return self.output_path / filename

    def save_features(self, lazy_frame: pl.LazyFrame, timeframe: str) -> Dict[str, Any]:
        """特徴量保存"""
        output_file = f"features_{self.config.engine_id}_{timeframe}.parquet"
        output_path = self.create_output_path(output_file)

        start_time = time.time()

        try:
            # blueprint.py v4 に準拠した型キャスト処理を内包（後処理スクリプトへの依存排除）
            cast_exprs = [
                pl.col("timestamp").cast(pl.Datetime("us")),
                pl.col("timeframe").cast(pl.Utf8),
            ]

            # tick時間足などでyear, month, dayが存在する場合のみint32にキャスト
            schema_names = lazy_frame.collect_schema().names()
            for c in ["year", "month", "day"]:
                if c in schema_names:
                    cast_exprs.append(pl.col(c).cast(pl.Int32))

            lazy_frame = lazy_frame.with_columns(cast_exprs)

            # ▼▼ 修正前: Categorical型をすべてUtf8へ一括キャスト（重複エラーの危険あり）
            # ▼▼ 修正前: lazy_frame = lazy_frame.with_columns(pl.col(pl.Categorical).cast(pl.Utf8))
            # ▼▼ 修正後: timeframe等の重複を避けるため、スキーマから明示的に除外してキャスト
            categorical_cols = [
                col
                for col, dtype in lazy_frame.collect_schema().items()
                if dtype == pl.Categorical and col != "timeframe"
            ]
            if categorical_cols:
                lazy_frame = lazy_frame.with_columns(
                    [pl.col(c).cast(pl.Utf8) for c in categorical_cols]
                )
            # ▲▲ 修正ここまで

            # ストリーミング出力
            lazy_frame.sink_parquet(str(output_path), compression="snappy")

            elapsed_time = time.time() - start_time

            # 保存後の検証
            if output_path.exists():
                file_size_mb = output_path.stat().st_size / (1024**2)

                # メタデータ
                metadata = {
                    "timeframe": timeframe,
                    "output_file": str(output_path),
                    "file_size_mb": file_size_mb,
                    "save_time_seconds": elapsed_time,
                    "compression": "snappy",
                    "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                }

                logger.info(
                    f"保存完了: {output_file} ({file_size_mb:.2f}MB, {elapsed_time:.2f}秒)"
                )

                return metadata
            else:
                raise IOError(f"ファイル保存に失敗: {output_path}")

        except Exception as e:
            logger.error(f"特徴量保存エラー: {e}")
            raise

    def apply_nan_filling(self, lazy_frame: pl.LazyFrame) -> pl.LazyFrame:
        """NaN埋め処理 (ホワイトリスト方式)

        【Phase 5 修正 (#35)】
        旧実装は `with_columns` 方式で、入力 LazyFrame の不要なカラム (disc 等) を
        そのまま保持していた。S1_PROCESSED に追加された disc 列が下流の 2_A KS
        フィルタや 2_E HF メタモデル等に「特徴量」として誤認される問題があったため、
        他 engine (1_A/B/D/E/F) と同じ select 方式に統一する。

        ホワイトリスト方式:
          - KEEP_BASIC で明示した基本カラムのみ保持
          - 数値型の特徴量カラム (e1c_*) のみ保持
          - それ以外 (disc 等) は select で自動除外

        将来 S1_PROCESSED に新たなメタデータ列が追加されても、自動で除外される。
        """
        try:
            schema = lazy_frame.collect_schema()

            # 明示ホワイトリスト (他 engine と整合)
            KEEP_BASIC = [
                "timestamp",
                "open",
                "high",
                "low",
                "close",
                "volume",
                "timeframe",
                "sample_weight",
            ]

            # 数値型の特徴量カラム (基本カラムを除く)
            numeric_feature_columns = [
                col
                for col, dtype in schema.items()
                if dtype in [pl.Float32, pl.Float64, pl.Int32, pl.Int64]
                and col not in KEEP_BASIC
            ]

            # 基本カラムは存在するもののみ保持 (entity の有無に依存)
            basic_exprs = [pl.col(c) for c in KEEP_BASIC if c in schema.names()]
            # 特徴量は NaN を 0.0 で埋める
            feature_exprs = [
                pl.col(c).fill_null(0.0).alias(c) for c in numeric_feature_columns
            ]

            result = lazy_frame.select(basic_exprs + feature_exprs)
            logger.info(
                f"NaN埋め処理完了: {len(numeric_feature_columns)}個の特徴量 "
                f"(basic={len(basic_exprs)}, ホワイトリスト方式)"
            )
            return result

        except Exception as e:
            logger.error(f"NaN埋め処理エラー: {e}")
            return lazy_frame

    def merge_intermediate_files(
        self, file_paths: List[str], timeframe: str
    ) -> Dict[str, Any]:
        """中間ファイル統合（tickデータ用）"""
        start_time = time.time()

        try:
            # 全ファイルをLazyFrameとして読み込み
            lazy_frames = []
            for path in file_paths:
                lf = pl.scan_parquet(path)
                lazy_frames.append(lf)

            # 水平結合（行の並びが同じため）
            if len(lazy_frames) == 1:
                combined_lf = lazy_frames[0]
            else:
                # timestamp列を基準に結合
                combined_lf = lazy_frames[0]
                for lf in lazy_frames[1:]:
                    # timestamp列を除外して結合
                    schema = lf.collect_schema()
                    non_timestamp_cols = [
                        col for col in schema.names() if col != "timestamp"
                    ]
                    combined_lf = combined_lf.join(
                        lf.select(["timestamp"] + non_timestamp_cols),
                        on="timestamp",
                        how="left",
                    )

            # 最終出力
            output_file = f"features_{self.config.engine_id}_{timeframe}.parquet"
            output_path = self.create_output_path(output_file)

            combined_lf.sink_parquet(str(output_path), compression="snappy")

            # 中間ファイル削除
            for path in file_paths:
                try:
                    Path(path).unlink()
                    logger.info(f"中間ファイル削除: {Path(path).name}")
                except Exception as e:
                    logger.warning(f"中間ファイル削除失敗: {e}")

            elapsed_time = time.time() - start_time
            file_size_mb = output_path.stat().st_size / (1024**2)

            metadata = {
                "timeframe": timeframe,
                "output_file": str(output_path),
                "file_size_mb": file_size_mb,
                "merge_time_seconds": elapsed_time,
                "intermediate_files_count": len(file_paths),
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            }

            logger.info(
                f"統合完了: {output_file} ({file_size_mb:.2f}MB, {elapsed_time:.2f}秒)"
            )

            return metadata

        except Exception as e:
            logger.error(f"ファイル統合エラー: {e}")
            raise

    def save_processing_metadata(self, metadata: Dict[str, Any]):
        """処理メタデータ保存"""
        metadata_file = (
            f"metadata_{self.config.engine_id}_{time.strftime('%Y%m%d_%H%M%S')}.json"
        )
        metadata_path = self.create_output_path(metadata_file)

        try:
            with open(metadata_path, "w", encoding="utf-8") as f:
                json.dump(metadata, f, indent=2, ensure_ascii=False)

            logger.info(f"メタデータ保存: {metadata_file}")

        except Exception as e:
            logger.error(f"メタデータ保存エラー: {e}")


# ===== ブロック3完了 =====


# ===== ブロック4開始 =====

# ========================================
# CalculationEngine - 計算核心クラス
# ========================================


class CalculationEngine:
    """テクニカル指標計算エンジン"""

    def __init__(self, config: ProcessingConfig):
        """初期化"""
        self.config = config
        self.memory_monitor = MemoryMonitor(config.memory_limit_gb)
        self.prefix = "e1c_"  # エンジン識別子

        # ▼▼ 追加: 一時ディレクトリの作成
        import tempfile
        from pathlib import Path

        self.temp_dir = Path(tempfile.mkdtemp(prefix=f"features_{config.engine_id}_"))
        logger.info(f"一時ディレクトリ作成: {self.temp_dir}")
        # ▲▲ 追加ここまで

        logger.info(f"CalculationEngine初期化: {self.prefix}")

    def get_feature_groups(self) -> Dict[str, List[str]]:
        """特徴量グループ定義（tickデータ対応：16グループに細分化）"""
        return {
            # RSI系を4分割
            "group_1a_rsi_basic": [
                "rsi_14",
                "rsi_21",
                "rsi_momentum_14",
                "rsi_momentum_21",
            ],
            "group_1b_rsi_extended": [
                "rsi_30",
                "rsi_50",
                "rsi_momentum_30",
                "rsi_momentum_50",
            ],
            "group_1c_rsi_stochastic": ["stochastic_rsi_14", "stochastic_rsi_21"],
            "group_1d_rsi_divergence": ["rsi_divergence_14", "rsi_divergence_21"],
            # MACD系を2分割
            "group_2a_macd_standard": [
                "macd_12_26",
                "macd_signal_12_26_9",
                "macd_histogram_12_26_9",
            ],
            "group_2b_macd_variants": [
                "macd_5_35",
                "macd_signal_5_35_5",
                "macd_histogram_5_35_5",
                "macd_19_39",
                "macd_signal_19_39_9",
                "macd_histogram_19_39_9",
            ],
            # ボリンジャーバンド系を3分割
            "group_3a_bb_20": [
                "bb_upper_20_2",
                "bb_lower_20_2",
                "bb_percent_20_2",
                "bb_width_20_2",
                "bb_width_pct_20_2",
                "bb_position_20_2",
            ],
            "group_3b_bb_30_50": [
                "bb_upper_30_2",
                "bb_lower_30_2",
                "bb_percent_30_2",
                "bb_width_30_2",
                "bb_upper_50_2",
                "bb_lower_50_2",
                "bb_percent_50_2",
                "bb_width_50_2",
            ],
            "group_3c_bb_variants": [
                "bb_upper_20_2.5",
                "bb_lower_20_2.5",
                "bb_percent_20_2.5",
                "bb_upper_20_3",
                "bb_lower_20_3",
                "bb_percent_20_3",
            ],
            # ATR系を2分割
            "group_4a_atr_basic": ["atr_13", "atr_21", "atr_pct_13", "atr_pct_21"],
            "group_4b_atr_extended": [
                "atr_34",
                "atr_55",
                "atr_trend_13",
                "atr_trend_21",
                "atr_volatility_13",
                "atr_volatility_21",
            ],
            # オシレーター系を2分割
            "group_5a_oscillators_basic": [
                "adx_13",
                "adx_21",
                "di_plus_13",
                "di_plus_21",
                "di_minus_13",
                "di_minus_21",
            ],
            "group_5b_oscillators_extended": [
                "stoch_k_14",
                "stoch_d_14_3",
                "aroon_up_14",
                "aroon_down_14",
                "williams_r_14",
            ],
            # モメンタム系を2分割
            "group_6a_momentum_basic": [
                "dpo_20",
                "trix_14",
                "momentum_10",
                "momentum_20",
            ],
            "group_6b_momentum_extended": [
                "ultimate_oscillator",
                "tsi_25",
                "rate_of_change_10",
                "rate_of_change_20",
            ],
            # 移動平均線系を2分割に統合
            "group_7_moving_averages": [
                "sma_10",
                "sma_20",
                "sma_50",
                "ema_10",
                "ema_20",
                "ema_50",
                "hma_21",
                "kama_21",
            ],
        }

    # ▼ 引数に timeframe を追加
    def _get_all_feature_expressions(
        self, lazy_frame: pl.LazyFrame, timeframe: str = "M1"
    ) -> pl.LazyFrame:
        """QAなし・raw値の全特徴量を計算して返す（unit test用）。
        
        calculate_all_features と同じ計算ステップを踏むが、
        最後の apply_quality_assurance を適用しない。
        学習パイプライン・本番パイプラインでは使用しないこと。
        """
        result = self.create_rsi_features(lazy_frame)
        result = self.create_macd_features(result)
        result = self.create_bollinger_features(result)
        result = self.create_atr_features(result)
        result = self.create_basic_processing_features(result)
        return result

    def calculate_all_features(
        self, lazy_frame: pl.LazyFrame, timeframe: str = "M1"
    ) -> pl.LazyFrame:
        """全特徴量計算（6段階処理）"""
        logger.info(f"=== カテゴリ1B: 全特徴量計算開始 ===")

        # メモリ安全性チェック
        is_safe, message = self.memory_monitor.check_memory_safety()
        if not is_safe:
            raise MemoryError(f"メモリ不足のため処理を中断: {message}")
        logger.info(message)

        try:
            # 1. RSI関連特徴量
            result = self.create_rsi_features(lazy_frame)

            # 2. MACD関連特徴量
            result = self.create_macd_features(result)

            # 3. ボリンジャーバンド特徴量（Numba UDF必須）
            result = self.create_bollinger_features(result)

            # 4. ATR関連特徴量（Numba UDF必須）
            result = self.create_atr_features(result)

            # 5. 基本データ処理特徴量
            result = self.create_basic_processing_features(result)

            # 6. 品質保証システム適用
            # ▼ timeframe を渡す
            result = self.apply_quality_assurance(result, timeframe=timeframe)

            logger.info(f"=== カテゴリ1B: 全特徴量計算完了 ===")
            return result

        except Exception as e:
            logger.error(f"特徴量計算エラー: {e}")
            raise

    def create_rsi_features(self, lazy_frame: pl.LazyFrame) -> pl.LazyFrame:
        """RSI関連特徴量群"""
        logger.info("RSI特徴量計算開始")

        exprs = []

        # 標準RSI（Polars Expression）
        # calculate_rsi_wilder: core_indicators の Wilder 平滑化 RSI (SMA 方式から移行)
        for period in self.config.window_sizes["rsi"]:
            exprs.append(
                pl.col("close")
                .map_batches(
                    lambda s, p=period: calculate_rsi_wilder(s.to_numpy(), p),
                    return_dtype=pl.Float64,
                )
                .alias(f"{self.prefix}rsi_{period}")
            )

            exprs.append(
                pl.col("close")
                .map_batches(
                    lambda s, p=period: calculate_rsi_wilder(s.to_numpy(), p),
                    return_dtype=pl.Float64,
                )
                .diff()
                .alias(f"{self.prefix}rsi_momentum_{period}")
            )

        for period in [14, 21]:
            rsi_col = pl.col("close").map_batches(
                lambda s, p=period: calculate_rsi_wilder(s.to_numpy(), p),
                return_dtype=pl.Float64,
            )

            exprs.append(
                (
                    (rsi_col - rsi_col.rolling_min(period))
                    / (
                        rsi_col.rolling_max(period)
                        - rsi_col.rolling_min(period)
                        + 1e-10
                    )
                    * 100
                ).alias(f"{self.prefix}stochastic_rsi_{period}")
            )

        for period in [14, 21]:
            price_change = (pl.col("close") - pl.col("close").shift(period)) / pl.col(
                "close"
            ).shift(period)
            rsi_col = pl.col("close").map_batches(
                lambda s, p=period: calculate_rsi_wilder(s.to_numpy(), p),
                return_dtype=pl.Float64,
            )
            rsi_change = (rsi_col - rsi_col.shift(period)) / 50 - 1

            exprs.append(
                (price_change - rsi_change).alias(
                    f"{self.prefix}rsi_divergence_{period}"
                )
            )

        result = lazy_frame.with_columns(exprs)
        logger.info(f"RSI特徴量計算完了: {len(exprs)}個")
        return result

    def create_macd_features(self, lazy_frame: pl.LazyFrame) -> pl.LazyFrame:
        """MACD関連特徴量群"""
        logger.info("MACD特徴量計算開始")

        exprs = []

        # MACD標準設定
        macd_configs = [
            (12, 26, 9),  # 標準
            (5, 35, 5),  # 短期
            (19, 39, 9),  # 長期
        ]

        # 内部正規化用のATR13ベース (calculate_atr_wilder: core_indicators の Wilder 版)
        atr_13_base = pl.struct(["high", "low", "close"]).map_batches(
            lambda s: calculate_atr_wilder(
                s.struct.field("high").to_numpy(),
                s.struct.field("low").to_numpy(),
                s.struct.field("close").to_numpy(),
                13,
            ),
            return_dtype=pl.Float64,
        )

        for fast, slow, signal in macd_configs:
            # EMA計算（Polars内蔵）
            ema_fast = pl.col("close").ewm_mean(span=fast, adjust=False)
            ema_slow = pl.col("close").ewm_mean(span=slow, adjust=False)

            # 生MACDライン（エイリアスなし・内部計算用）
            macd_raw = ema_fast - ema_slow

            # MACDライン（scale_by_atr: core_indicators の ATR割り統一関数）
            macd_line = (
                pl.struct(
                    [macd_raw.alias("_macd"), atr_13_base.alias("_atr")]
                ).map_batches(
                    lambda s: scale_by_atr(
                        s.struct.field("_macd").to_numpy(),
                        s.struct.field("_atr").to_numpy(),
                    ),
                    return_dtype=pl.Float64,
                )
            ).alias(f"{self.prefix}macd_{fast}_{slow}")
            exprs.append(macd_line)

            # 生シグナルライン（エイリアスなし・内部計算用）
            signal_raw = macd_raw.ewm_mean(span=signal, adjust=False)

            # シグナルライン（scale_by_atr: core_indicators の ATR割り統一関数）
            signal_line = (
                pl.struct(
                    [signal_raw.alias("_sig"), atr_13_base.alias("_atr")]
                ).map_batches(
                    lambda s: scale_by_atr(
                        s.struct.field("_sig").to_numpy(),
                        s.struct.field("_atr").to_numpy(),
                    ),
                    return_dtype=pl.Float64,
                )
            ).alias(f"{self.prefix}macd_signal_{fast}_{slow}_{signal}")
            exprs.append(signal_line)

            # ヒストグラム（scale_by_atr: core_indicators の ATR割り統一関数）
            hist_raw = macd_raw - signal_raw
            histogram = (
                pl.struct(
                    [hist_raw.alias("_hist"), atr_13_base.alias("_atr")]
                ).map_batches(
                    lambda s: scale_by_atr(
                        s.struct.field("_hist").to_numpy(),
                        s.struct.field("_atr").to_numpy(),
                    ),
                    return_dtype=pl.Float64,
                )
            ).alias(f"{self.prefix}macd_histogram_{fast}_{slow}_{signal}")
            exprs.append(histogram)

        result = lazy_frame.with_columns(exprs)
        logger.info(f"MACD特徴量計算完了: {len(exprs)}個")
        return result

    def create_bollinger_features(self, lazy_frame: pl.LazyFrame) -> pl.LazyFrame:
        """ボリンジャーバンド特徴量群（Numba UDF使用）"""
        logger.info("ボリンジャーバンド特徴量計算開始")

        exprs = []

        # ▼▼ 追加: 内部計算用のATR13（スケール不変性用）
        # calculate_atr_wilder: core_indicators の Wilder 版（SMA 方式から移行）
        atr_13_expr = pl.struct(["high", "low", "close"]).map_batches(
            lambda s: calculate_atr_wilder(
                s.struct.field("high").to_numpy(),
                s.struct.field("low").to_numpy(),
                s.struct.field("close").to_numpy(),
                13,
            ),
            return_dtype=pl.Float64,
        )

        # ボリンジャーバンド設定
        bb_periods = [20, 30, 50]
        bb_stdevs = [2, 2.5, 3]

        for period in bb_periods:
            for num_std in bb_stdevs:
                # 移動平均と標準偏差
                # ▼▼ 修正前: std = pl.col("close").rolling_std(period)
                # ▼▼ 修正後: 分散の罠を回避（ddof=1を明記）
                sma = pl.col("close").rolling_mean(period)
                std = pl.col("close").rolling_std(period, ddof=1)

                # 上限・下限バンド
                # ▼▼ 修正前: 絶対値でのバンド算出
                # ▼▼ 修正後: 生価格で計算後、ATR割りによるスケール不変表現へ変換
                upper_raw = sma + num_std * std
                lower_raw = sma - num_std * std

                # scale_by_atr: core_indicators の ATR割り統一関数
                upper = (
                    pl.struct(
                        [
                            (upper_raw - pl.col("close")).alias("_u"),
                            atr_13_expr.alias("_atr"),
                        ]
                    ).map_batches(
                        lambda s: scale_by_atr(
                            s.struct.field("_u").to_numpy(),
                            s.struct.field("_atr").to_numpy(),
                        ),
                        return_dtype=pl.Float64,
                    )
                ).alias(f"{self.prefix}bb_upper_{period}_{num_std}")
                lower = (
                    pl.struct(
                        [
                            (pl.col("close") - lower_raw).alias("_l"),
                            atr_13_expr.alias("_atr"),
                        ]
                    ).map_batches(
                        lambda s: scale_by_atr(
                            s.struct.field("_l").to_numpy(),
                            s.struct.field("_atr").to_numpy(),
                        ),
                        return_dtype=pl.Float64,
                    )
                ).alias(f"{self.prefix}bb_lower_{period}_{num_std}")

                exprs.extend([upper, lower])

                # BB内の位置（%B）※これは既に比率なのでraw値を使用
                percent_b = (
                    (pl.col("close") - lower_raw) / (upper_raw - lower_raw + 1e-10)
                ).alias(f"{self.prefix}bb_percent_{period}_{num_std}")
                exprs.append(percent_b)

                # BB幅 (scale_by_atr: core_indicators の ATR割り統一関数)
                width = (
                    pl.struct(
                        [(upper_raw - lower_raw).alias("_w"), atr_13_expr.alias("_atr")]
                    ).map_batches(
                        lambda s: scale_by_atr(
                            s.struct.field("_w").to_numpy(),
                            s.struct.field("_atr").to_numpy(),
                        ),
                        return_dtype=pl.Float64,
                    )
                ).alias(f"{self.prefix}bb_width_{period}_{num_std}")
                exprs.append(width)

                # BB幅のパーセンタイル
                # ▼▼ 修正前: width_pct = (width / sma * 100).alias(...)
                # ▼▼ 修正後: 生価格同士で計算し、純粋なパーセント（無次元）へ正常化
                width_pct = ((upper_raw - lower_raw) / (sma + 1e-10) * 100).alias(
                    f"{self.prefix}bb_width_pct_{period}_{num_std}"
                )
                exprs.append(width_pct)

                # BB内での相対位置
                position = ((pl.col("close") - sma) / (std + 1e-10)).alias(
                    f"{self.prefix}bb_position_{period}_{num_std}"
                )
                exprs.append(position)

        result = lazy_frame.with_columns(exprs)
        logger.info(f"ボリンジャーバンド特徴量計算完了: {len(exprs)}個")
        return result

    def create_atr_features(self, lazy_frame: pl.LazyFrame) -> pl.LazyFrame:
        """ATR関連特徴量群（Numba UDF使用）"""
        logger.info("ATR特徴量計算開始")

        exprs = []

        # ATR期間
        atr_periods = self.config.window_sizes["atr"]  # [13, 21, 34, 55]

        # 内部正規化用のATR13ベース（スケール不変化の基準）
        # calculate_atr_wilder: core_indicators の Wilder 版（SMA 方式から移行）
        atr_13_base = pl.struct(["high", "low", "close"]).map_batches(
            lambda s: calculate_atr_wilder(
                s.struct.field("high").to_numpy(),
                s.struct.field("low").to_numpy(),
                s.struct.field("close").to_numpy(),
                13,
            ),
            return_dtype=pl.Float64,
        )

        for period in atr_periods:
            # 生ATR計算（エイリアスなし・内部計算用）
            # calculate_atr_wilder: core_indicators の Wilder 版（SMA 方式から移行）
            atr_raw = pl.struct(["high", "low", "close"]).map_batches(
                lambda s, p=period: calculate_atr_wilder(
                    s.struct.field("high").to_numpy(),
                    s.struct.field("low").to_numpy(),
                    s.struct.field("close").to_numpy(),
                    p,
                ),
                return_dtype=pl.Float64,
            )

            # ATR比率（ATR13で割ったスケール不変値）- scale_by_atr: core_indicators の ATR割り統一関数
            exprs.append(
                pl.struct([atr_raw.alias("_atr_raw"), atr_13_base.alias("_atr13")])
                .map_batches(
                    lambda s: scale_by_atr(
                        s.struct.field("_atr_raw").to_numpy(),
                        s.struct.field("_atr13").to_numpy(),
                    ),
                    return_dtype=pl.Float64,
                )
                .alias(f"{self.prefix}atr_{period}")
            )

            # ATRパーセンテージ（closeに対する比率）はすでにスケール不変
            exprs.append(
                (atr_raw / (pl.col("close") + 1e-10) * 100).alias(
                    f"{self.prefix}atr_pct_{period}"
                )
            )

            # ATRトレンド（ATRの変化分をATR13で割りスケール不変化）- scale_by_atr: core_indicators
            exprs.append(
                pl.struct([atr_raw.diff().alias("_diff"), atr_13_base.alias("_atr13")])
                .map_batches(
                    lambda s: scale_by_atr(
                        s.struct.field("_diff").to_numpy(),
                        s.struct.field("_atr13").to_numpy(),
                    ),
                    return_dtype=pl.Float64,
                )
                .alias(f"{self.prefix}atr_trend_{period}")
            )

            # ATRボラティリティ（ATR自体の標準偏差をATR13で割りスケール不変化）- scale_by_atr: core_indicators
            # 本番側も同様に atr_13 で除算（realtime_feature_engine_1C と完全一致）
            exprs.append(
                pl.struct(
                    [
                        atr_raw.rolling_std(period, ddof=1).alias("_std"),
                        atr_13_base.alias("_atr13"),
                    ]
                )
                .map_batches(
                    lambda s: scale_by_atr(
                        s.struct.field("_std").to_numpy(),
                        s.struct.field("_atr13").to_numpy(),
                    ),
                    return_dtype=pl.Float64,
                )
                .alias(f"{self.prefix}atr_volatility_{period}")
            )

            # ▼▼ 削除: ATRベースの絶対値バンド（atr_upper / atr_lower）のループ処理を丸ごと削除
            #        atr_multipliers = [1.5, 2.0, 2.5]
            #        for mult in atr_multipliers: ...
            # ▲▲ 削除ここまで

        result = lazy_frame.with_columns(exprs)
        logger.info(f"ATR特徴量計算完了: {len(exprs)}個")
        return result

    # ===== ブロック4完了 =====

    # ===== ブロック5開始 =====

    def create_oscillator_features(self, lazy_frame: pl.LazyFrame) -> pl.LazyFrame:
        """オシレーター系特徴量群"""
        logger.info("オシレーター特徴量計算開始")

        exprs = []

        # Stochastic Oscillator
        stoch_periods = [(14, 3, 3), (21, 5, 5), (9, 3, 3)]
        for k_period, d_period, slow_period in stoch_periods:
            stoch_k = (
                pl.struct(["high", "low", "close"])
                # 修正後
                .map_batches(
                    lambda s, kp=k_period, dp=d_period, sp=slow_period: (
                        calculate_stochastic_numba(
                            s.struct.field("high").to_numpy(),
                            s.struct.field("low").to_numpy(),
                            s.struct.field("close").to_numpy(),
                            kp,
                            dp,
                            sp,
                        )
                    ),
                    return_dtype=pl.Float64,
                )
                .alias(f"{self.prefix}stoch_k_{k_period}")
            )
            exprs.append(stoch_k)

            # %D (signal line)
            stoch_d = stoch_k.rolling_mean(d_period).alias(
                f"{self.prefix}stoch_d_{k_period}_{d_period}"
            )
            exprs.append(stoch_d)

            # Slow %D
            slow_d = stoch_d.rolling_mean(slow_period).alias(
                f"{self.prefix}stoch_slow_d_{k_period}_{d_period}_{slow_period}"
            )
            exprs.append(slow_d)

        # ADX, DI+, DI- — calculate_adx (core_indicators) と _calculate_di_wilder を使用
        # _calculate_di_wilder は calculate_adx 内部と完全同一の Wilder 平滑化を実装しており、
        # ADX と DI が同一の ATR 平滑化を参照することを保証する。
        for period in self.config.window_sizes["adx"]:  # [13, 21, 34]
            # ADX (calculate_adx: core_indicators の Wilder 版)
            adx = (
                pl.struct(["high", "low", "close"])
                .map_batches(
                    lambda s, p=period: calculate_adx(
                        s.struct.field("high").to_numpy(),
                        s.struct.field("low").to_numpy(),
                        s.struct.field("close").to_numpy(),
                        p,
                    ),
                    return_dtype=pl.Float64,
                )
                .alias(f"{self.prefix}adx_{period}")
            )
            exprs.append(adx)

            # DI+ (_calculate_di_wilder: calculate_adx 内部と同一 Wilder 平滑化)
            di_plus = (
                pl.struct(["high", "low", "close"])
                .map_batches(
                    lambda s, p=period: _calculate_di_wilder(
                        s.struct.field("high").to_numpy(),
                        s.struct.field("low").to_numpy(),
                        s.struct.field("close").to_numpy(),
                        p,
                    )[0],
                    return_dtype=pl.Float64,
                )
                .alias(f"{self.prefix}di_plus_{period}")
            )
            exprs.append(di_plus)

            # DI- (_calculate_di_wilder: calculate_adx 内部と同一 Wilder 平滑化)
            di_minus = (
                pl.struct(["high", "low", "close"])
                .map_batches(
                    lambda s, p=period: _calculate_di_wilder(
                        s.struct.field("high").to_numpy(),
                        s.struct.field("low").to_numpy(),
                        s.struct.field("close").to_numpy(),
                        p,
                    )[1],
                    return_dtype=pl.Float64,
                )
                .alias(f"{self.prefix}di_minus_{period}")
            )
            exprs.append(di_minus)

        # Aroon Indicator
        aroon_periods = [14, 25, 50]
        for period in aroon_periods:
            # Aroon Up
            aroon_up = (
                pl.col("high")
                # 修正後
                .map_batches(
                    lambda s, p=period: calculate_aroon_up_numba(s.to_numpy(), p),
                    return_dtype=pl.Float64,
                )
                .alias(f"{self.prefix}aroon_up_{period}")
            )
            exprs.append(aroon_up)

            # Aroon Down
            aroon_down = (
                pl.col("low")
                # 修正後
                .map_batches(
                    lambda s, p=period: calculate_aroon_down_numba(s.to_numpy(), p),
                    return_dtype=pl.Float64,
                )
                .alias(f"{self.prefix}aroon_down_{period}")
            )
            exprs.append(aroon_down)

            # Aroon Oscillator
            aroon_osc = (aroon_up - aroon_down).alias(
                f"{self.prefix}aroon_oscillator_{period}"
            )
            exprs.append(aroon_osc)

        # =====================================================================
        # 【修正済み】williams_r の late binding バグ (今回の再学習機会で解消)
        #
        # 【旧バグ】
        #   lambda s: calculate_williams_r_numba(..., period, ...)
        #   ↑ Python の late binding により全 period が常に 56 で計算されていた。
        #   結果として e1c_williams_r_14, e1c_williams_r_28, e1c_williams_r_56 の
        #   3カラムすべてが period=56 の同一値で保存・学習されていた。
        #
        # 【今回の修正】
        #   lambda s, p=period: calculate_williams_r_numba(..., p, ...)
        #   ↑ デフォルト引数 p=period でキャプチャすることで各 period が独立して評価される。
        #   再学習機会の今回、本来の period=14/28/56 の独立した値が初めてモデルに反映される。
        #
        # 【本番側 (rfe_1C) 対応】
        #   rfe_1C 側の period=56 固定実装も同時に解除し、3 period 個別計算に戻す。
        # =====================================================================
        # Williams %R
        williams_periods = [14, 28, 56]
        for period in williams_periods:
            williams_r = (
                pl.struct(["high", "low", "close"])
                .map_batches(
                    lambda s, p=period: calculate_williams_r_numba(  # ★ 修正: p=period でキャプチャ
                        s.struct.field("high").to_numpy(),
                        s.struct.field("low").to_numpy(),
                        s.struct.field("close").to_numpy(),
                        p,
                    ),
                    return_dtype=pl.Float64,
                )
                .alias(f"{self.prefix}williams_r_{period}")
            )
            exprs.append(williams_r)

        result = lazy_frame.with_columns(exprs)
        logger.info(f"オシレーター特徴量計算完了: {len(exprs)}個")
        return result

    def create_momentum_features(self, lazy_frame: pl.LazyFrame) -> pl.LazyFrame:
        """モメンタム系特徴量群"""
        logger.info("モメンタム特徴量計算開始")

        exprs = []

        # ▼▼ 追加: 内部計算用のATR13
        # calculate_atr_wilder: core_indicators の Wilder 版（SMA 方式から移行）
        atr_13_expr = pl.struct(["high", "low", "close"]).map_batches(
            lambda s: calculate_atr_wilder(
                s.struct.field("high").to_numpy(),
                s.struct.field("low").to_numpy(),
                s.struct.field("close").to_numpy(),
                13,
            ),
            return_dtype=pl.Float64,
        )

        # Detrended Price Oscillator (DPO) — scale_by_atr: core_indicators の ATR割り統一関数
        dpo_periods = [20, 30, 50]
        for period in dpo_periods:
            sma = pl.col("close").rolling_mean(period)
            dpo_raw = pl.col("close") - sma
            dpo = (
                pl.struct(
                    [dpo_raw.alias("_dpo"), atr_13_expr.alias("_atr")]
                ).map_batches(
                    lambda s: scale_by_atr(
                        s.struct.field("_dpo").to_numpy(),
                        s.struct.field("_atr").to_numpy(),
                    ),
                    return_dtype=pl.Float64,
                )
            ).alias(f"{self.prefix}dpo_{period}")
            exprs.append(dpo)

        # TRIX (Numba版)
        trix_periods = [14, 20, 30]
        for period in trix_periods:
            trix = (
                pl.col("close")
                # 修正後
                .map_batches(
                    lambda s, p=period: calculate_trix_numba(s.to_numpy(), p),
                    return_dtype=pl.Float64,
                )
                .alias(f"{self.prefix}trix_{period}")
            )
            exprs.append(trix)

        # Ultimate Oscillator (Numba版)
        uo = (
            pl.struct(["high", "low", "close", "volume"])
            .map_batches(
                lambda s: calculate_ultimate_oscillator_numba(
                    s.struct.field("high").to_numpy(),
                    s.struct.field("low").to_numpy(),
                    s.struct.field("close").to_numpy(),
                    s.struct.field("volume").to_numpy(),
                ),
                return_dtype=pl.Float64,
            )
            .alias(f"{self.prefix}ultimate_oscillator")
        )
        exprs.append(uo)

        # True Strength Index (TSI) (Numba版)
        tsi_periods = [25, 13]
        for period in tsi_periods:
            tsi = (
                pl.col("close")
                # 修正後
                .map_batches(
                    lambda s, p=period: calculate_tsi_numba(s.to_numpy(), p),
                    return_dtype=pl.Float64,
                )
                .alias(f"{self.prefix}tsi_{period}")
            )
            exprs.append(tsi)

        # Rate of Change (ROC)
        roc_periods = [10, 20, 30, 50]
        for period in roc_periods:
            roc = (
                (pl.col("close") - pl.col("close").shift(period))
                / pl.col("close").shift(period)
                * 100
            ).alias(f"{self.prefix}rate_of_change_{period}")
            exprs.append(roc)

        # Momentum — scale_by_atr: core_indicators の ATR割り統一関数
        momentum_periods = [10, 20, 30, 50]
        for period in momentum_periods:
            mom_raw = pl.col("close") - pl.col("close").shift(period)
            momentum = (
                pl.struct(
                    [mom_raw.alias("_mom"), atr_13_expr.alias("_atr")]
                ).map_batches(
                    lambda s: scale_by_atr(
                        s.struct.field("_mom").to_numpy(),
                        s.struct.field("_atr").to_numpy(),
                    ),
                    return_dtype=pl.Float64,
                )
            ).alias(f"{self.prefix}momentum_{period}")
            exprs.append(momentum)

        result = lazy_frame.with_columns(exprs)
        logger.info(f"モメンタム特徴量計算完了: {len(exprs)}個")
        return result

    def create_advanced_features(self, lazy_frame: pl.LazyFrame) -> pl.LazyFrame:
        """高度な指標特徴量群"""
        logger.info("高度な指標特徴量計算開始")

        exprs = []

        # Know Sure Thing (KST)
        # ROC(10)*1 + ROC(15)*2 + ROC(20)*3 + ROC(30)*4の加重平均
        roc_10 = (pl.col("close") - pl.col("close").shift(10)) / pl.col("close").shift(
            10
        )
        roc_15 = (pl.col("close") - pl.col("close").shift(15)) / pl.col("close").shift(
            15
        )
        roc_20 = (pl.col("close") - pl.col("close").shift(20)) / pl.col("close").shift(
            20
        )
        roc_30 = (pl.col("close") - pl.col("close").shift(30)) / pl.col("close").shift(
            30
        )

        kst = ((roc_10 * 1 + roc_15 * 2 + roc_20 * 3 + roc_30 * 4) / 10 * 100).alias(
            f"{self.prefix}kst"
        )
        exprs.append(kst)

        # KST Signal Line
        kst_signal = kst.rolling_mean(9).alias(f"{self.prefix}kst_signal")
        exprs.append(kst_signal)

        # Relative Vigor Index (RVI)
        rvi_periods = [10, 14, 20]
        for period in rvi_periods:
            # (Close - Open) / (High - Low)の移動平均
            numerator = (pl.col("close") - pl.col("open")).rolling_mean(period)
            denominator = (pl.col("high") - pl.col("low")).rolling_mean(period)
            rvi = (numerator / (denominator + 1e-10)).alias(
                f"{self.prefix}relative_vigor_index_{period}"
            )
            exprs.append(rvi)

            # RVI Signal
            rvi_signal = rvi.rolling_mean(4).alias(f"{self.prefix}rvi_signal_{period}")
            exprs.append(rvi_signal)

        # Schaff Trend Cycle (STC) - 簡略版からの完全純化
        stc_periods = [(23, 50, 10), (12, 26, 9)]
        for fast_period, slow_period, cycle_period in stc_periods:
            # MACDベース
            fast_ma = pl.col("close").ewm_mean(half_life=fast_period, adjust=False)
            slow_ma = pl.col("close").ewm_mean(half_life=slow_period, adjust=False)
            macd = fast_ma - slow_ma

            # 1st Stochastic (%K of MACD)
            macd_min1 = macd.rolling_min(cycle_period)
            macd_max1 = macd.rolling_max(cycle_period)
            stoch_macd = ((macd - macd_min1) / (macd_max1 - macd_min1 + 1e-10)) * 100

            # ▼▼ 修正前: smooth_period = max(2, cycle_period // 2)
            # ▼▼ 修正前: stoch_macd_smoothed = stoch_macd.ewm_mean(half_life=smooth_period, adjust=False)
            # ▼▼ 修正後: 原論文（Doug Schaff）およびStochasticの伝統的コンセンサスに忠実な固定値(span=3)を採用
            smooth_period = 3
            stoch_macd_smoothed = stoch_macd.ewm_mean(span=smooth_period, adjust=False)

            # 2nd Stochastic (%K of Smoothed %D)
            stoch_min2 = stoch_macd_smoothed.rolling_min(cycle_period)
            stoch_max2 = stoch_macd_smoothed.rolling_max(cycle_period)
            stoch_stoch = (
                (stoch_macd_smoothed - stoch_min2) / (stoch_max2 - stoch_min2 + 1e-10)
            ) * 100

            # Final Smoothing (True STC)
            stc = stoch_stoch.ewm_mean(span=smooth_period, adjust=False).alias(
                f"{self.prefix}schaff_trend_cycle_{fast_period}_{slow_period}_{cycle_period}"
            )
            exprs.append(stc)

        # Coppock Curve
        # ROC(14) + ROC(11)の10期間WMA
        roc_11 = (
            (pl.col("close") - pl.col("close").shift(11))
            / pl.col("close").shift(11)
            * 100
        )
        roc_14 = (
            (pl.col("close") - pl.col("close").shift(14))
            / pl.col("close").shift(14)
            * 100
        )
        coppock = (
            (roc_11 + roc_14).rolling_mean(10).alias(f"{self.prefix}coppock_curve")
        )
        exprs.append(coppock)

        # Price Oscillator
        po_configs = [(12, 26), (5, 35), (10, 20)]
        for fast, slow in po_configs:
            fast_ma = pl.col("close").ewm_mean(span=fast, adjust=False)
            slow_ma = pl.col("close").ewm_mean(span=slow, adjust=False)
            po = ((fast_ma - slow_ma) / slow_ma * 100).alias(
                f"{self.prefix}price_oscillator_{fast}_{slow}"
            )
            exprs.append(po)

        result = lazy_frame.with_columns(exprs)
        logger.info(f"高度な指標特徴量計算完了: {len(exprs)}個")
        return result

    def create_moving_average_features(self, lazy_frame: pl.LazyFrame) -> pl.LazyFrame:
        """移動平均線・トレンド分析特徴量群"""
        logger.info("移動平均線特徴量計算開始")

        exprs = []
        # 内部計算用のATR13
        # calculate_atr_wilder: core_indicators の Wilder 版（SMA 方式から移行）
        atr_13_expr = pl.struct(["high", "low", "close"]).map_batches(
            lambda s: calculate_atr_wilder(
                s.struct.field("high").to_numpy(),
                s.struct.field("low").to_numpy(),
                s.struct.field("close").to_numpy(),
                13,
            ),
            return_dtype=pl.Float64,
        )

        # 基本移動平均
        ma_periods = [10, 20, 50, 100, 200]

        for period in ma_periods:
            # SMA — scale_by_atr: core_indicators の ATR割り統一関数
            sma_raw = pl.col("close").rolling_mean(period)
            sma = (
                pl.struct(
                    [(sma_raw - pl.col("close")).alias("_d"), atr_13_expr.alias("_atr")]
                ).map_batches(
                    lambda s: scale_by_atr(
                        s.struct.field("_d").to_numpy(),
                        s.struct.field("_atr").to_numpy(),
                    ),
                    return_dtype=pl.Float64,
                )
            ).alias(f"{self.prefix}sma_{period}")
            exprs.append(sma)

            # SMA乖離率 (生価格ベースでの乖離比率を維持)
            sma_deviation = (
                (pl.col("close") - sma_raw) / (sma_raw + 1e-10) * 100
            ).alias(f"{self.prefix}sma_deviation_{period}")
            exprs.append(sma_deviation)

            # EMA — scale_by_atr: core_indicators の ATR割り統一関数
            ema_raw = pl.col("close").ewm_mean(span=period, adjust=False)
            ema = (
                pl.struct(
                    [(ema_raw - pl.col("close")).alias("_d"), atr_13_expr.alias("_atr")]
                ).map_batches(
                    lambda s: scale_by_atr(
                        s.struct.field("_d").to_numpy(),
                        s.struct.field("_atr").to_numpy(),
                    ),
                    return_dtype=pl.Float64,
                )
            ).alias(f"{self.prefix}ema_{period}")
            exprs.append(ema)

            # EMA乖離率
            ema_deviation = (
                (pl.col("close") - ema_raw) / (ema_raw + 1e-10) * 100
            ).alias(f"{self.prefix}ema_deviation_{period}")
            exprs.append(ema_deviation)

            # WMA (Numba版) — scale_by_atr: core_indicators の ATR割り統一関数
            wma_raw = pl.col("close").map_batches(
                lambda s, p=period: calculate_wma_numba(s.to_numpy(), p),
                return_dtype=pl.Float64,
            )
            wma = (
                pl.struct(
                    [(wma_raw - pl.col("close")).alias("_d"), atr_13_expr.alias("_atr")]
                ).map_batches(
                    lambda s: scale_by_atr(
                        s.struct.field("_d").to_numpy(),
                        s.struct.field("_atr").to_numpy(),
                    ),
                    return_dtype=pl.Float64,
                )
            ).alias(f"{self.prefix}wma_{period}")
            exprs.append(wma)

        # HMA (Hull Moving Average) - Numba版 — scale_by_atr: core_indicators の ATR割り統一関数
        for period in self.config.window_sizes["hma"]:  # [21, 34, 55]
            hma_raw = pl.col("close").map_batches(
                lambda s, p=period: calculate_hma_numba(s.to_numpy(), p),
                return_dtype=pl.Float64,
            )
            hma = (
                pl.struct(
                    [(hma_raw - pl.col("close")).alias("_d"), atr_13_expr.alias("_atr")]
                ).map_batches(
                    lambda s: scale_by_atr(
                        s.struct.field("_d").to_numpy(),
                        s.struct.field("_atr").to_numpy(),
                    ),
                    return_dtype=pl.Float64,
                )
            ).alias(f"{self.prefix}hma_{period}")
            exprs.append(hma)

        # KAMA (Kaufman Adaptive Moving Average) - Numba版 — scale_by_atr: core_indicators の ATR割り統一関数
        for period in self.config.window_sizes["kama"]:  # [21, 34]
            kama_raw = pl.col("close").map_batches(
                lambda s, p=period: calculate_kama_numba(s.to_numpy(), p),
                return_dtype=pl.Float64,
            )
            kama = (
                pl.struct(
                    [
                        (kama_raw - pl.col("close")).alias("_d"),
                        atr_13_expr.alias("_atr"),
                    ]
                ).map_batches(
                    lambda s: scale_by_atr(
                        s.struct.field("_d").to_numpy(),
                        s.struct.field("_atr").to_numpy(),
                    ),
                    return_dtype=pl.Float64,
                )
            ).alias(f"{self.prefix}kama_{period}")
            exprs.append(kama)

        # トレンド分析
        trend_periods = [20, 50, 100]
        for period in trend_periods:
            # 真の線形回帰(OLS)の傾きを WMA と SMA の恒等式から計算し、ATR割り
            # scale_by_atr: core_indicators の ATR割り統一関数
            sma_for_slope = pl.col("close").rolling_mean(period)
            wma_for_slope = pl.col("close").map_batches(
                lambda s, p=period: calculate_wma_numba(s.to_numpy(), p),
                return_dtype=pl.Float64,
            )
            true_ols_slope = 6.0 * (wma_for_slope - sma_for_slope) / (period - 1.0)

            trend_slope = (
                pl.struct(
                    [true_ols_slope.alias("_slope"), atr_13_expr.alias("_atr")]
                ).map_batches(
                    lambda s: scale_by_atr(
                        s.struct.field("_slope").to_numpy(),
                        s.struct.field("_atr").to_numpy(),
                    ),
                    return_dtype=pl.Float64,
                )
            ).alias(f"{self.prefix}trend_slope_{period}")
            exprs.append(trend_slope)

            # トレンド強度 — scale_by_atr で無次元化した normalized_std を利用
            normalized_std = pl.struct(
                [
                    pl.col("close").rolling_std(period, ddof=1).alias("_std"),
                    atr_13_expr.alias("_atr"),
                ]
            ).map_batches(
                lambda s: scale_by_atr(
                    s.struct.field("_std").to_numpy(), s.struct.field("_atr").to_numpy()
                ),
                return_dtype=pl.Float64,
            )
            trend_strength = (
                (1.0 / (normalized_std + 1e-10))
                .clip(upper_bound=100.0)
                .alias(f"{self.prefix}trend_strength_{period}")
            )
            exprs.append(trend_strength)

            # トレンド一貫性（方向変化の頻度）
            direction_changes = pl.col("close").diff().sign().diff().abs()
            trend_consistency = (1 - direction_changes.rolling_mean(period) / 2).alias(
                f"{self.prefix}trend_consistency_{period}"
            )
            exprs.append(trend_consistency)

        result = lazy_frame.with_columns(exprs)
        logger.info(f"移動平均線特徴量計算完了: {len(exprs)}個")
        return result

    def create_basic_processing_features(
        self, lazy_frame: pl.LazyFrame
    ) -> pl.LazyFrame:
        """基本データ処理特徴量"""
        logger.info("基本データ処理特徴量計算開始")

        # 追加のオシレーター、モメンタム、高度な指標、移動平均を計算
        result = self.create_oscillator_features(lazy_frame)
        result = self.create_momentum_features(result)
        result = self.create_advanced_features(result)
        result = self.create_moving_average_features(result)

        # ▼▼ sample_weight: calculate_sample_weight (core_indicators) に統一
        # Polars 直書き式（atr_numba + z_score 計算）を廃止し、
        # core_indicators の calculate_sample_weight（Wilder ATR・|z| < 2.0 基準）に統一する。
        result = result.with_columns(
            pl.struct(["high", "low", "close"])
            .map_batches(
                lambda s: calculate_sample_weight(
                    s.struct.field("high").to_numpy(),
                    s.struct.field("low").to_numpy(),
                    s.struct.field("close").to_numpy(),
                ),
                return_dtype=pl.Float64,
            )
            .alias("sample_weight")
        )

        logger.info("基本データ処理特徴量計算完了")
        return result

    # ▼▼ 修正前: def apply_quality_assurance(self, lazy_frame: pl.LazyFrame) -> pl.LazyFrame: (旧方式)
    # ▼▼ 修正前: def _apply_qa_to_series(self, values: np.ndarray) -> np.ndarray: (削除)
    # ▼▼ 修正後: EWMベースの動的±5σクリッピングへ完全一本化（未来情報の混入を防止）
    # ▼ 引数に timeframe を追加
    # ▼▼ Phase 5 修正: inf null化 + NaN null化 + forward_fill による状態汚染防止
    # ▼▼ Phase 5 F群 (#34) 修正: API 均質化 — canonical な実装を _to_group に移し、
    #    apply_quality_assurance は auto-discovery 後に _to_group へ委譲する後方互換ラッパーへ
    def apply_quality_assurance_to_group(
        self,
        lazy_frame: pl.LazyFrame,
        feature_columns: List[str],
        timeframe: str = "M1",
    ) -> pl.LazyFrame:
        """単一グループに対する品質保証システムの適用 (canonical 実装)。

        engine_1_A/B/D/E/F の apply_quality_assurance_to_group と同じ意味を持つ
        faithful 実装。feature_columns で指定された列のみに QA を適用する。

        【Phase 5 F群 (#34)】API 均質化のために導入されたメソッド。
        engine_1_C は歴史的経緯で apply_quality_assurance (auto-discovery 方式) を
        持っていたが、本メソッドの追加により他 engine と統一されたインターフェースを
        提供する。

        【Phase 5 D群】1A-1B-1D-1E-1F と同じ正しい QA パターン:
          - inf を null 化する `pl.when(is_infinite).then(None)` を適用
          - .fill_nan(None) で NaN も null 化 (Polars EWM 状態汚染防止)
          - .forward_fill() を ewm_mean/std に適用 (inf/NaN 位置でも有効 bounds 維持)
          - clip 適用は pl.col(col) (元の値) に対して行い、inf 位置は
            pl.when(col==inf).then(upper).when(col==-inf).then(lower) で置換
        """
        if not feature_columns:
            return lazy_frame

        half_life = self.config.timeframe_bars_per_day.get(timeframe, 1440)
        logger.info(
            f"品質保証適用: {len(feature_columns)}個の特徴量 (timeframe={timeframe})"
        )

        qa_exprs = []
        for col in feature_columns:
            safe_col = (
                pl.when(pl.col(col).is_infinite())
                .then(None)
                .otherwise(pl.col(col))
                .fill_nan(None)  # ★ NaN も null 化 (Polars EWM 仕様への対応)
            )

            # ★ + forward_fill() で inf/NaN 位置でも有効な bounds を提供
            ewm_mean = safe_col.ewm_mean(
                half_life=half_life, adjust=False, ignore_nulls=True
            ).forward_fill()  # ★ inf/NaN 位置でも直前の有効 mean を維持
            ewm_std = safe_col.ewm_std(
                half_life=half_life, adjust=False, ignore_nulls=True
            ).forward_fill()  # ★ inf/NaN 位置でも直前の有効 std を維持

            upper_bound = ewm_mean + 5.0 * ewm_std
            lower_bound = ewm_mean - 5.0 * ewm_std

            # Inf値を動的境界値で置換し、全体をクリッピング (engine_1_B/D と同型)
            clipped_expr = (
                pl.when(pl.col(col) == float("inf"))
                .then(upper_bound)
                .when(pl.col(col) == float("-inf"))
                .then(lower_bound)
                .otherwise(pl.col(col))
                .clip(lower_bound, upper_bound)
                .fill_null(0.0)
                .fill_nan(0.0)
                .alias(col)
            )
            qa_exprs.append(clipped_expr)

        return lazy_frame.with_columns(qa_exprs)

    def apply_quality_assurance(
        self, lazy_frame: pl.LazyFrame, timeframe: str = "M1"
    ) -> pl.LazyFrame:
        """品質保証システム適用 (旧 API)。

        【Phase 5 F群 (#34) リファクタ】
        本メソッドは Polars LazyFrame の schema から prefix マッチする特徴量カラムを
        auto-discovery で取得し、apply_quality_assurance_to_group に委譲する
        後方互換ラッパーへリファクタされた。

        既存の呼び出し元 (engine_1_C 内 L789, L1893) のシグネチャを保つため、
        内部実装変更による数値結果への影響は無し (差 0)。

        新規コードは可能な限り apply_quality_assurance_to_group を直接使うべき。
        """
        logger.info(f"品質保証システム適用開始 (EWMベース, timeframe={timeframe})")

        try:
            schema = lazy_frame.collect_schema()
            feature_columns = [
                col
                for col in schema.names()
                if col.startswith(self.prefix)
                and schema[col] in [pl.Float32, pl.Float64]
            ]

            result = self.apply_quality_assurance_to_group(
                lazy_frame, feature_columns, timeframe=timeframe
            )

            if feature_columns:
                logger.info(f"品質保証適用完了: {len(feature_columns)}列")

            return result

        except Exception as e:
            logger.error(f"品質保証エラー: {e}")
            return lazy_frame

    # ▼ 引数に timeframe を追加 (Engine 1C における calculate_one_group 相当)
    def calculate_feature_group(
        self,
        lazy_frame: pl.LazyFrame,
        group_name: str,
        feature_list: List[str],
        timeframe: str = "M1",
    ) -> pl.LazyFrame:
        """特定グループの特徴量計算（細分化対応）"""
        logger.info(f"グループ計算開始: {group_name}")

        # グループに応じた計算（細分化対応）
        if "rsi" in group_name:
            result = self.create_rsi_features(lazy_frame)
        elif "macd" in group_name:
            result = self.create_macd_features(lazy_frame)
        elif "bb" in group_name:
            result = self.create_bollinger_features(lazy_frame)
        elif "atr" in group_name:
            result = self.create_atr_features(lazy_frame)
        elif "oscillator" in group_name:
            result = self.create_oscillator_features(lazy_frame)
        elif "momentum" in group_name:
            result = self.create_momentum_features(lazy_frame)
        elif "advanced" in group_name:
            result = self.create_advanced_features(lazy_frame)
        elif "moving" in group_name:
            result = self.create_moving_average_features(lazy_frame)
        else:
            result = lazy_frame

        # 該当グループの特徴量のみを抽出
        # 修正後
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
        schema = result.collect_schema()
        available_features = [
            col for col in schema.names() if col.startswith(self.prefix)
        ]

        # feature_listとの照合は緩い条件で（部分一致）
        selected_features = []
        for feature in feature_list:
            matching_cols = [col for col in available_features if feature in col]
            selected_features.extend(matching_cols)

        # 基本列 + 選択された特徴量のみ選択
        final_columns = base_columns + list(set(selected_features))
        result = result.select(
            [pl.col(col) for col in final_columns if col in schema.names()]
        )

        # 品質保証適用
        # ▼ timeframe を渡す
        result = self.apply_quality_assurance(result, timeframe=timeframe)

        logger.info(
            f"グループ計算完了: {group_name} - {len(selected_features)}個の特徴量"
        )
        return result


# ===== ブロック5完了 =====

# ================================================================
# ===== ここから新規挿入ブロック =====
# ================================================================


# --- パーティション処理のためのヘルパー関数 ---
def get_sorted_partitions(root_dir: Path) -> list[Path]:
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
    df_current = lf_current.collect()

    # ▼▼ 修正: 早期リターンの前に、現在のDFへ確実に timeframe を追加
    df_current = df_current.with_columns(
        pl.lit("tick").alias("timeframe").cast(pl.Utf8)
    )

    len_current_partition = df_current.height

    if prev_partition_path is None:
        return df_current, len_current_partition

    lookback_required = w_max - 1
    if lookback_required <= 0:
        return df_current, len_current_partition

    lf_prev = pl.scan_parquet(prev_partition_path / "*.parquet")
    df_prefix = lf_prev.tail(lookback_required).collect()

    # ▼▼ 修正: 過去のDF（prefix）にも結合前に timeframe を追加（スキーマ不一致エラー防止）
    df_prefix = df_prefix.with_columns(pl.lit("tick").alias("timeframe").cast(pl.Utf8))

    augmented_df = pl.concat([df_prefix, df_current], how="vertical")

    return augmented_df, len_current_partition


# --- Tickデータ専用の処理モード ---
def run_on_partitions_mode(config: ProcessingConfig):
    """実行モード: Tickデータ専用。パーティションを一つずつ処理する。"""
    logging.info(
        "【実行モード】パーティション化されたTickデータの特徴量計算を開始します..."
    )

    timeframe = "tick"  # timeframeを明示的に定義

    # ▼▼ ここから下の定義が欠落していました
    PARTITION_ROOT = Path(config.partitioned_tick_path)
    FEATURES_ROOT = (
        Path(config.output_base_path) / f"features_{config.engine_id}_{timeframe}/"
    )
    FEATURES_ROOT.mkdir(parents=True, exist_ok=True)

    W_MAX = config.w_max
    calculation_engine = CalculationEngine(config)

    sorted_partitions = get_sorted_partitions(PARTITION_ROOT)

    # 運用上の改善（再開ロジック）
    resume_date = getattr(config, "resume_date", None)
    if resume_date:
        logging.info(f"再開モード: {resume_date} 以降のパーティションを処理します。")

    for i, current_path in enumerate(sorted_partitions):
        if resume_date:
            date_str = f"{current_path.parent.parent.name.split('=')[1]}-{current_path.parent.name.split('=')[1].zfill(2)}-{current_path.name.split('=')[1].zfill(2)}"
            if date_str < resume_date:
                continue

        logging.info(
            f"=== 水平処理 ({i + 1}/{len(sorted_partitions)}): {current_path.relative_to(PARTITION_ROOT)} ==="
        )

        prev_path = sorted_partitions[i - 1] if i > 0 else None

        try:
            augmented_df, len_current = create_augmented_frame(
                current_path, prev_path, W_MAX
            )

            # --- ここから垂直分割（特徴量グループごと）のループ ---
            intermediate_group_files = []
            feature_groups = calculation_engine.get_feature_groups()

            for group_name, feature_list in feature_groups.items():
                logging.info(f"--- 垂直グループ処理: {group_name} ---")

                # ▼ timeframe を引数として明示的に渡す
                group_lf_result = calculation_engine.calculate_feature_group(
                    augmented_df.lazy(), group_name, feature_list, timeframe=timeframe
                )

                # 一時ファイルとして保存
                temp_group_path = (
                    FEATURES_ROOT / f"temp_{current_path.name}_{group_name}.parquet"
                )
                group_lf_result.collect().write_parquet(temp_group_path)
                intermediate_group_files.append(str(temp_group_path))

            # --- 垂直分割ループ完了 ---

            logging.info(f"{current_path.name} の全グループを結合中...")

            if not intermediate_group_files:
                logging.warning("結合する中間ファイルがありません。スキップします。")
                continue

            # 1. 最初のファイルを「土台」として読み込む
            base_df = pl.read_parquet(intermediate_group_files[0])

            # 2. 2つ目以降のファイルをループで処理
            for f in intermediate_group_files[1:]:
                # 特徴量ファイルを追加で読み込む
                feature_df = pl.read_parquet(f)

                # 土台にすでに存在する列（timestamp, open, high, low, close, volumeなど）を除外
                new_feature_columns = [
                    col for col in feature_df.columns if col not in base_df.columns
                ]

                # 新しい特徴量列のみを水平に結合
                base_df = base_df.with_columns(feature_df.select(new_feature_columns))

            # 最終的な結合結果
            day_result_df = base_df

            # 全ての中間ファイルを削除
            for f in intermediate_group_files:
                Path(f).unlink()

            lookback_required = W_MAX - 1
            if prev_path is not None and lookback_required > 0:
                final_output_df = day_result_df.tail(len_current)
            else:
                final_output_df = day_result_df

            assert final_output_df.height == len_current, (
                f"出力行数が一致しません: expected={len_current}, got={final_output_df.height}"
            )

            output_path = FEATURES_ROOT / current_path.relative_to(PARTITION_ROOT)
            output_path.mkdir(parents=True, exist_ok=True)
            final_output_df.write_parquet(output_path / "features.parquet")

            logging.info(f"保存完了: {output_path}")

        except Exception as e:
            logging.error(
                f"パーティション {current_path} の処理中にエラー: {e}", exc_info=True
            )
            continue


# ================================================================
# ===== 新規挿入ブロックここまで =====
# ================================================================

# ===== ブロック6開始 =====

# ========================================
# メイン処理関数
# ========================================


# --- 通常時間足（M1, H1など）の処理関数 ---
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

        # ▼▼ 修正前: features_lf = calc_engine.calculate_all_features(lazy_frame, timeframe=timeframe)
        # ▼▼ 修正後: 物理的垂直分割（ディスク退避・結合ロジック）に完全置換
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

            group_result_lf = calc_engine.calculate_feature_group(
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

        # 修正後
        for temp_file in temp_files[1:]:
            next_df = pl.read_parquet(str(temp_file))
            feature_cols = [
                col
                for col in next_df.columns
                if col not in base_columns and col not in result_df.columns
            ]
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
        # ▲▲ 修正ここまで

        processed_lf = output_engine.apply_nan_filling(features_lf)
        metadata = output_engine.save_features(processed_lf, timeframe)

        elapsed_time = time.time() - start_time
        metadata["processing_time"] = elapsed_time
        logger.info(f"=== 通常処理完了: {timeframe} - {elapsed_time:.2f}秒 ===")
        return metadata
    except Exception as e:
        logger.error(f"タイムフレーム {timeframe} の処理中にエラー: {e}", exc_info=True)
        return {"timeframe": timeframe, "error": str(e)}


# --- ユーザーとの対話部分（元の関数をそのまま利用） ---
def get_user_confirmation(config: ProcessingConfig) -> bool:
    """ユーザー確認"""
    print("\n" + "=" * 60)
    print("Engine 1B - テクニカル指標・トレンド分析エンジン")
    print("=" * 60)
    print(f"入力パス: {config.input_base_path}")
    print(f"出力パス: {config.output_base_path}")
    print(f"Tickパーティションパス: {config.partitioned_tick_path}")
    print(f"エンジンID: {config.engine_id}")
    print(f"並列スレッド数: {config.max_threads}")
    print(f"メモリ制限: {config.memory_limit_gb}GB")

    if config.test_mode:
        print(f"\n【テストモード】 最初の{config.test_rows}行のみ処理")

    print(f"\n処理対象タイムフレーム ({len(config.timeframes)}個):")
    for i, tf in enumerate(config.timeframes):
        print(f"  {i + 1:2d}. {tf}")

    print("\n処理内容:")
    print("  - RSI関連指標")
    print("  - MACD")
    print("  - ボリンジャーバンド")
    print("  - ATR")
    print("  - ADX・方向性指標")
    print("  - オシレーター系")
    print("  - モメンタム指標")
    print("  - 移動平均線・トレンド分析")

    response = input("\n処理を開始しますか？ (y/n): ")
    return response.lower() == "y"


def select_timeframes(config: ProcessingConfig) -> List[str]:
    """タイムフレーム選択"""
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


# --- メインの実行関数（司令塔） ---
def main():
    """設定を行い、処理モードを分岐させるメイン関数"""
    print("\n" + "=" * 70)
    print(" Engine 1B - Technical Indicators and Trend Analysis Engine ")
    print(" テクニカル指標・トレンド分析特徴量生成エンジン ")
    print("=" * 70)

    config = ProcessingConfig()

    if not config.validate():
        return 1

    data_engine = DataEngine(config)
    if not data_engine.validate_data_source():
        return 1

    # --- ここから対話形式のセットアップ（省略なし） ---
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
    print(f"  1. デフォルト ({config.output_base_path})")
    print("  2. カスタムパス")

    path_selection = input("選択 (1/2): ").strip()
    if path_selection == "2":
        custom_path = input("出力パスを入力: ").strip()
        if custom_path:
            config.output_base_path = custom_path
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
    print("  2. 本番モード（全データ処理）")

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
    # --- 対話形式のセットアップここまで ---

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

    # 選択された時間足に 'tick' が含まれているかで処理を分岐
    if "tick" in selected_timeframes:
        run_on_partitions_mode(config)

    # Tick以外の時間足を処理
    other_timeframes = [tf for tf in selected_timeframes if tf != "tick"]
    if other_timeframes:
        for tf in other_timeframes:
            process_single_timeframe(config, tf)

    overall_elapsed_time = time.time() - overall_start_time
    print(
        f"\n全ての要求された処理が完了しました。総処理時間: {overall_elapsed_time:.2f}秒"
    )
    return 0


# --- スクリプト実行のエントリーポイント ---
if __name__ == "__main__":
    try:
        exit_code = main()
        sys.exit(exit_code)
    except Exception as e:
        logger.critical(
            f"スクリプト実行中に致命的なエラーが発生しました: {e}", exc_info=True
        )
        sys.exit(1)

# ===== ブロック6完了 =====
# ===== Engine 1C 実装完了 =====
