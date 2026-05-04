#!/usr/bin/env python3
"""
革新的特徴量収集スクリプト - Engine 1B: A Vast Universe of Features (修正版)
【修正内容】垂直分割アーキテクチャをディスクベース物理分割に変更

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
import os
from pathlib import Path

# blueprint より先にパスを追加しないと ModuleNotFoundError が発生する
sys.path.append(str(Path(__file__).resolve().parent.parent))  # /workspace をパスに追加
import blueprint as config
sys.path.append(str(config.CORE_DIR))
from core_indicators import (
    calculate_atr_wilder,
    # [SSoT 統一] Engine 1B の UDF を core_indicators から import
    # (旧: 本ファイル内で定義 → 本番側 rfe_1B と二重定義 = SSoT 違反)
    adf_統計量_udf,
    phillips_perron_統計量_udf,
    kpss_統計量_udf,
    t分布_自由度_udf,
    t分布_尺度_udf,
    gev_形状_udf,
    holt_winters_レベル_udf,
    holt_winters_トレンド_udf,
    arima_残差分散_udf,
    kalman_状態推定_udf,
    lowess_適合値_udf,
    theil_sen_傾き_udf,
)

import time, warnings, json, logging, math, tempfile, datetime
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

    # ▼▼ 修正前: 古いS1パス
    # ▼▼ 修正後: blueprint v4 準拠のパス定義に変更
    input_path: str = str(config.S1_PROCESSED)
    partitioned_tick_path: str = str(config.S1_RAW_TICK_PARTITIONED)
    output_path: str = str(config.S2_FEATURES)

    # エンジン識別
    engine_id: str = "e1b"
    engine_name: str = "Engine_1B_VastUniverse"

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

    # ▼▼ 修正後: タイムフレームごとの1日あたりのバー数（動的half_life計算用）を追加
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


# ===============================
# Numba UDF定義（クラス外必須）
# ===============================

# 【重要】全てのNumba UDFは必ずクラス外で定義すること
# クラス内定義は循環参照エラーを引き起こすため絶対禁止


# [REFACTORED: Step 4]
# calculate_atr_numba (旧: ウォームアップ期間に np.mean(tr[:i+1]) を使う独自実装) は削除。
# core_indicators.calculate_atr_wilder (Wilder平滑化・シード=TR[0]) を使用すること。
# → from core_indicators import calculate_atr_wilder（ファイル冒頭でインポート済み）


# =============================================================================
# 時系列解析: 単位根検定 (Numba UDF集)
# =============================================================================


# =============================================================================
# 分布パラメータ: t分布 (Numba UDF集)
# =============================================================================


# =============================================================================
# 分布パラメータ: 一般化極値（GEV）分布
# =============================================================================


# =============================================================================
# 指数平滑: ホルト・ウィンターズ成分
# =============================================================================


# =============================================================================
# ARIMAモデル成分
# =============================================================================


# =============================================================================
# カルマンフィルタ成分
# =============================================================================


# =============================================================================
# 局所回帰（LOWESS）と頑健回帰
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
        # ▼▼ 修正後: デッドコードに伴い削除
        self.memory_monitor = MemoryMonitor(config.memory_limit_gb)
        self.prefix = f"e{config.engine_id.replace('e', '')}_"  # 例: "e1b_"

        # 【修正】ディスクベース垂直分割用の一時ディレクトリ
        self.temp_dir = Path(tempfile.mkdtemp(prefix=f"features_{config.engine_id}_"))
        logger.info(f"一時ディレクトリ作成: {self.temp_dir}")

    # ▼▼ 修正後: 事前計算用メソッドの新規追加
    def inject_temp_atr(self, lazy_frame: pl.LazyFrame) -> pl.LazyFrame:
        """グループ分割前にATR13を一度だけ計算して注入する（パフォーマンス最適化）"""
        logger.info("内部計算用ATR13 (__temp_atr_safe) を事前注入します...")
        # [REFACTORED: Step 4] calculate_atr_numba → calculate_atr_wilder (core_indicators)
        # 旧実装はウォームアップ期間 (i < period) に np.mean(tr[:i+1]) を使っていたが、
        # calculate_atr_wilder はシード=TR[0] の純粋なWilder平滑化で統一する。
        atr_expr = pl.struct(["high", "low", "close"]).map_batches(
            lambda s: calculate_atr_wilder(
                s.struct.field("high").to_numpy(),
                s.struct.field("low").to_numpy(),
                s.struct.field("close").to_numpy(),
                13,
            ),
            return_dtype=pl.Float64,
        )
        return lazy_frame.with_columns((atr_expr + 1e-10).alias("__temp_atr_safe"))

    # ▼▼ 修正前: 毎回 map_batches の Expression を生成
    # ▼▼ 修正後: 事前注入されたカラムを参照するだけの軽量Expressionに変更
    def _get_atr_safe_expr(self) -> pl.Expr:
        """事前計算されたATR13（ゼロ除算防止付き）のExpressionを取得"""
        return pl.col("__temp_atr_safe")

    def _get_all_feature_expressions(self, timeframe: str = "M1") -> Dict[str, pl.Expr]:
        """
        全特徴量の名前とPolars Expressionの対応辞書を返すファクトリメソッド。
        これにより、必要な特徴量だけを効率的に計算できるようになる。
        【特徴量ファクトリパターン】
        """
        expressions = {}
        p = self.prefix
        lookback_bars = self.config.timeframe_bars_per_day.get(timeframe, 1440)

        # ▼▼ 修正後: atr_safeの取得（ゼロ除算保護付き）
        atr_safe = self._get_atr_safe_expr()

        # 基本統計系 - 全ての式に明示的なaliasを付与 (絶対値からATR相対値への変換)
        for window in self.config.window_sizes["general"]:
            expressions[f"{p}rolling_mean_{window}"] = (
                (pl.col("close").rolling_mean(window) - pl.col("close")) / atr_safe
            ).alias(f"{p}rolling_mean_{window}")
            expressions[f"{p}rolling_std_{window}"] = (
                pl.col("close").rolling_std(window, ddof=1) / atr_safe
            ).alias(f"{p}rolling_std_{window}")
            expressions[f"{p}rolling_var_{window}"] = (
                pl.col("close").rolling_var(window, ddof=1) / (atr_safe**2)
            ).alias(f"{p}rolling_var_{window}")
            expressions[f"{p}rolling_median_{window}"] = (
                (pl.col("close").rolling_median(window) - pl.col("close")) / atr_safe
            ).alias(f"{p}rolling_median_{window}")
            expressions[f"{p}rolling_min_{window}"] = (
                (pl.col("close").rolling_min(window) - pl.col("close")) / atr_safe
            ).alias(f"{p}rolling_min_{window}")
            expressions[f"{p}rolling_max_{window}"] = (
                (pl.col("close").rolling_max(window) - pl.col("close")) / atr_safe
            ).alias(f"{p}rolling_max_{window}")

        # 複合計算系 - ゼロ除算防止(1e-10)とATRスケール不変性の適用
        for window in [20, 50]:
            mean_col = pl.col("close").rolling_mean(window)
            std_col = pl.col("close").rolling_std(window, ddof=1)
            # zscoreはすでにスケール不変。分母のゼロ除算のみ保護
            expressions[f"{p}zscore_{window}"] = (
                (pl.col("close") - mean_col) / (std_col + 1e-10)
            ).alias(f"{p}zscore_{window}")
            # ボリンジャーバンドはCloseからの乖離をATR割り
            expressions[f"{p}bollinger_upper_{window}"] = (
                ((mean_col + 2 * std_col) - pl.col("close")) / atr_safe
            ).alias(f"{p}bollinger_upper_{window}")
            expressions[f"{p}bollinger_lower_{window}"] = (
                ((mean_col - 2 * std_col) - pl.col("close")) / atr_safe
            ).alias(f"{p}bollinger_lower_{window}")

        # Numba UDF統合系 - rolling_mapを使用（仕様書準拠）- min_periods非推奨警告対応
        for window in [50, 100]:
            expressions[f"{p}adf_statistic_{window}"] = (
                pl.col("close")
                .pct_change()
                .rolling_map(
                    lambda s: adf_統計量_udf(s.to_numpy()),
                    window_size=window,
                    min_samples=window,
                )
                .alias(f"{p}adf_statistic_{window}")
            )

            expressions[f"{p}pp_statistic_{window}"] = (
                pl.col("close")
                .pct_change()
                .rolling_map(
                    lambda s: phillips_perron_統計量_udf(s.to_numpy()),
                    window_size=window,
                    min_samples=window,
                )
                .alias(f"{p}pp_statistic_{window}")
            )

            expressions[f"{p}kpss_statistic_{window}"] = (
                pl.col("close")
                .pct_change()
                .rolling_map(
                    lambda s: kpss_統計量_udf(s.to_numpy()),
                    window_size=window,
                    min_samples=window,
                )
                .alias(f"{p}kpss_statistic_{window}")
            )

            # ▼▼ 修正後: 指数平滑・回帰系の絶対値表現をATR相対値に変換
            expressions[f"{p}holt_level_{window}"] = (
                (
                    pl.col("close").rolling_map(
                        lambda s: holt_winters_レベル_udf(s.to_numpy()),
                        window_size=window,
                        min_samples=window,
                    )
                    - pl.col("close")
                )
                / atr_safe
            ).alias(f"{p}holt_level_{window}")

            expressions[f"{p}holt_trend_{window}"] = (
                pl.col("close").rolling_map(
                    lambda s: holt_winters_トレンド_udf(s.to_numpy()),
                    window_size=window,
                    min_samples=window,
                )
                / atr_safe
            ).alias(f"{p}holt_trend_{window}")

            expressions[f"{p}arima_residual_var_{window}"] = (
                pl.col("close").rolling_map(
                    lambda s: arima_残差分散_udf(s.to_numpy()),
                    window_size=window,
                    min_samples=window,
                )
                / (atr_safe**2)
            ).alias(f"{p}arima_residual_var_{window}")

            expressions[f"{p}kalman_state_{window}"] = (
                (
                    pl.col("close").rolling_map(
                        lambda s: kalman_状態推定_udf(s.to_numpy()),
                        window_size=window,
                        min_samples=window,
                    )
                    - pl.col("close")
                )
                / atr_safe
            ).alias(f"{p}kalman_state_{window}")

            expressions[f"{p}lowess_fitted_{window}"] = (
                (
                    pl.col("close").rolling_map(
                        lambda s: lowess_適合値_udf(s.to_numpy()),
                        window_size=window,
                        min_samples=window,
                    )
                    - pl.col("close")
                )
                / atr_safe
            ).alias(f"{p}lowess_fitted_{window}")

            expressions[f"{p}theil_sen_slope_{window}"] = (
                pl.col("close").rolling_map(
                    lambda s: theil_sen_傾き_udf(s.to_numpy()),
                    window_size=window,
                    min_samples=window,
                )
                / atr_safe
            ).alias(f"{p}theil_sen_slope_{window}")

        expressions[f"{p}t_dist_dof_50"] = (
            pl.col("close")
            .pct_change()
            .rolling_map(
                lambda s: t分布_自由度_udf(s.to_numpy()), window_size=50, min_samples=50
            )
            .alias(f"{p}t_dist_dof_50")
        )

        expressions[f"{p}t_dist_scale_50"] = (
            (
                pl.col("close")
                .pct_change()
                .rolling_map(
                    lambda s: t分布_尺度_udf(s.to_numpy()),
                    window_size=50,
                    min_samples=50,
                )
            )
            / (pl.col("close").pct_change().rolling_std(20, ddof=1) + 1e-10)
        ).alias(f"{p}t_dist_scale_50")

        expressions[f"{p}gev_shape_50"] = (
            pl.col("high")
            .rolling_map(
                lambda s: gev_形状_udf(s.to_numpy()), window_size=50, min_samples=50
            )
            .alias(f"{p}gev_shape_50")
        )

        # 基本データ処理特徴量 - 全ての式に明示的なaliasを付与
        expressions[f"{p}price_change"] = (
            pl.col("close").pct_change().alias(f"{p}price_change")
        )
        expressions[f"{p}volatility_20"] = (
            pl.col("close")
            .pct_change()
            .rolling_std(20, ddof=1)
            .alias(f"{p}volatility_20")
        )
        expressions[f"{p}price_range"] = (
            (pl.col("high") - pl.col("low")) / atr_safe
        ).alias(f"{p}price_range")

        # ▼▼ 修正前: volumeの絶対値出力
        # ▼▼ 修正後: Relative Volume (1440=1日想定) 化によるスケール不変性の確保
        rel_volume = pl.col("volume") / (pl.col("volume").rolling_mean(lookback_bars) + 1e-10)

        expressions[f"{p}volume_ma20"] = rel_volume.rolling_mean(20).alias(
            f"{p}volume_ma20"
        )
        expressions[f"{p}volume_price_trend"] = (
            (pl.col("close").pct_change() * rel_volume)
            .rolling_mean(10)
            .alias(f"{p}volume_price_trend")
        )

        return expressions

    def get_feature_groups(self) -> Dict[str, Dict[str, pl.Expr]]:
        """特徴量グループ定義を外部から取得可能にする"""
        return self._create_vertical_slices()

    # ▼▼ 修正前: def calculate_one_group(self, lazy_frame: pl.LazyFrame, group_name: str, group_expressions: Dict[str, pl.Expr]) -> pl.LazyFrame:
    # ▼▼ 修正後: timeframe引数を追加し、QAメソッドへ伝播させる
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
            if group_name == "basic_stats":
                group_result_lf = self._create_basic_stats_features(lazy_frame, timeframe)
            elif group_name == "composite":
                group_result_lf = self._create_composite_features(lazy_frame, timeframe)
            elif group_name == "timeseries":
                group_result_lf = self._create_timeseries_features(lazy_frame)
            elif group_name == "exponential_arima":
                group_result_lf = self._create_exponential_arima_features(lazy_frame)
            elif group_name == "kalman_regression":
                group_result_lf = self._create_kalman_regression_features(lazy_frame)
            elif group_name == "distributions":
                group_result_lf = self._create_distributions_features(lazy_frame)
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
                "sample_weight",
            ]
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

            # ▼▼ 修正前: stabilized_lf = self.apply_quality_assurance_to_group(group_final_lf, group_feature_columns)
            # ▼▼ 修正後: timeframeを渡す
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

    # ▼▼ 修正後: timeframe引数の追加とhalf_lifeの動的計算
    def apply_quality_assurance_to_group(
        self,
        lazy_frame: pl.LazyFrame,
        feature_columns: List[str],
        timeframe: str = "M1",
    ) -> pl.LazyFrame:
        """単一グループに対する品質保証システムの適用"""
        if not feature_columns:
            return lazy_frame

        half_life = self.config.timeframe_bars_per_day.get(timeframe, 1440)
        logger.info(
            f"品質保証適用: {len(feature_columns)}個の特徴量 (EWM half_life={half_life} + 5σクリッピング)"
        )

        stabilization_exprs = []

        for col_name in feature_columns:
            # ─────────────────────────────────────────────────────────────
            # 【修正 (Phase 5)】EWM 状態汚染防止と inf 入力時の bounds 維持
            #
            # 旧実装の問題:
            #   1) safe_col = pl.when(is_infinite).then(None).otherwise(col)
            #      で inf は null 化していたが、NaN はそのまま残していた。
            #      Polars の ewm_mean(ignore_nulls=True) は null だけ無視するが
            #      NaN は値として扱い、後続を全部 NaN 化する (状態汚染)。
            #   2) inf 位置でも ewm_mean / ewm_std の出力が null になるため、
            #      その時点での bounds が null となり、pl.when(col==inf).then(upper_bound)
            #      が機能せず inf がそのまま流れていた。
            #
            # 修正:
            #   - safe_col に .fill_nan(None) を追加 → NaN も null 化 (状態汚染防止)
            #   - ewm_mean / ewm_std に .forward_fill() を追加 → inf 位置でも有効 bounds
            # ─────────────────────────────────────────────────────────────
            safe_col = (
                pl.when(pl.col(col_name).is_infinite())
                .then(None)
                .otherwise(pl.col(col_name))
                .fill_nan(None)  # ★ NaN も null 化 (Polars EWM 仕様への対応)
            )

            # ▼▼ 修正後: ハードコードを排除し、half_life変数を使用
            # ★ + forward_fill() で inf/NaN 位置でも有効な bounds を提供
            ewm_mean = safe_col.ewm_mean(
                half_life=half_life, ignore_nulls=True, adjust=False
            ).forward_fill()  # ★ inf/NaN 位置でも直前の有効 mean を維持
            ewm_std = safe_col.ewm_std(
                half_life=half_life, ignore_nulls=True, adjust=False
            ).forward_fill()  # ★ inf/NaN 位置でも直前の有効 std を維持

            upper_bound = ewm_mean + 5 * ewm_std
            lower_bound = ewm_mean - 5 * ewm_std

            # Inf値を動的境界値で置換し、全体をクリッピング
            stabilized_col = (
                pl.when(pl.col(col_name) == float("inf"))
                .then(upper_bound)
                .when(pl.col(col_name) == float("-inf"))
                .then(lower_bound)
                .otherwise(pl.col(col_name))
                .clip(lower_bound=lower_bound, upper_bound=upper_bound)
                .fill_null(0.0)
                .fill_nan(0.0)
                .alias(col_name)
            )

            stabilization_exprs.append(stabilized_col)

        result = lazy_frame.with_columns(stabilization_exprs)
        return result

    def _create_vertical_slices(self, timeframe: str = "M1") -> Dict[str, Dict[str, pl.Expr]]:
        """物理的垂直分割: 特徴量を論理グループに分割"""
        all_expressions = self._get_all_feature_expressions(timeframe)

        # メモリ使用量を考慮したグルーピング（英語キー使用）
        slices = {}
        p = self.prefix

        # グループ1: 基本統計系（軽量）
        slices["basic_stats"] = {
            name: expr
            for name, expr in all_expressions.items()
            if any(
                pattern in name
                for pattern in [
                    "rolling_mean",
                    "rolling_std",
                    "rolling_var",
                    "rolling_median",
                    "rolling_min",
                    "rolling_max",
                ]
            )
        }

        # グループ2: 複合計算系（中程度）
        slices["composite"] = {
            name: expr
            for name, expr in all_expressions.items()
            if any(
                pattern in name
                for pattern in [
                    "zscore",
                    "bollinger",
                    "price_change",
                    "volatility",
                    "price_range",
                    "volume",
                ]
            )
        }

        # グループ3: 時系列解析系（重い）
        slices["timeseries"] = {
            name: expr
            for name, expr in all_expressions.items()
            if any(
                pattern in name
                for pattern in ["adf_statistic", "pp_statistic", "kpss_statistic"]
            )
        }

        # グループ4: 指数平滑・ARIMA系（重い）
        slices["exponential_arima"] = {
            name: expr
            for name, expr in all_expressions.items()
            if any(
                pattern in name
                for pattern in ["holt_level", "holt_trend", "arima_residual"]
            )
        }

        # グループ5: カルマン・回帰系（重い）
        slices["kalman_regression"] = {
            name: expr
            for name, expr in all_expressions.items()
            if any(
                pattern in name
                for pattern in ["kalman_state", "lowess_fitted", "theil_sen"]
            )
        }

        # グループ6: 分布パラメータ系（重い）
        slices["distributions"] = {
            name: expr
            for name, expr in all_expressions.items()
            if any(pattern in name for pattern in ["t_dist", "gev_shape"])
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

    # ▼▼ 修正後: timeframe引数の追加とhalf_lifeの動的計算
    def apply_quality_assurance(
        self, lazy_frame: pl.LazyFrame, timeframe: str = "M1"
    ) -> pl.LazyFrame:
        """2段階品質保証システムの適用"""
        logger.info("品質保証システム適用開始: 全ての特徴量に安定化処理を適用します。")

        # スキーマからプレフィックスを持つ特徴量カラムを特定
        schema = lazy_frame.collect_schema()
        feature_columns = [col for col in schema.names() if col.startswith(self.prefix)]

        if not feature_columns:
            logger.warning("品質保証対象の特徴量が見つかりません。")
            return lazy_frame

        logger.info(f"品質保証対象: {len(feature_columns)}個の特徴量")

        half_life = self.config.timeframe_bars_per_day.get(timeframe, 1440)
        stabilization_exprs = []

        for col_name in feature_columns:
            # ─────────────────────────────────────────────────────────────
            # 【修正 (Phase 5)】EWM 状態汚染防止と inf 入力時の bounds 維持
            # 詳細は apply_quality_assurance_to_group のコメント参照。
            # ─────────────────────────────────────────────────────────────
            safe_col = (
                pl.when(pl.col(col_name).is_infinite())
                .then(None)
                .otherwise(pl.col(col_name))
                .fill_nan(None)  # ★ NaN も null 化 (Polars EWM 仕様への対応)
            )

            # ▼▼ 修正後: ハードコードを排除し、half_life変数を使用
            # ★ + forward_fill() で inf/NaN 位置でも有効な bounds を提供
            ewm_mean = safe_col.ewm_mean(
                half_life=half_life, ignore_nulls=True, adjust=False
            ).forward_fill()  # ★ inf/NaN 位置でも直前の有効 mean を維持
            ewm_std = safe_col.ewm_std(
                half_life=half_life, ignore_nulls=True, adjust=False
            ).forward_fill()  # ★ inf/NaN 位置でも直前の有効 std を維持

            upper_bound = ewm_mean + 5 * ewm_std
            lower_bound = ewm_mean - 5 * ewm_std

            # Inf値を動的境界値で置換し、全体をクリッピング
            stabilized_col = (
                pl.when(pl.col(col_name) == float("inf"))
                .then(upper_bound)
                .when(pl.col(col_name) == float("-inf"))
                .then(lower_bound)
                .otherwise(pl.col(col_name))
                .clip(lower_bound=lower_bound, upper_bound=upper_bound)
                .fill_null(0.0)
                .fill_nan(0.0)
                .alias(col_name)
            )

            stabilization_exprs.append(stabilized_col)

        result = lazy_frame.with_columns(stabilization_exprs)
        logger.info("品質保証システム適用完了")

        return result

    def _create_basic_stats_features(self, lazy_frame: pl.LazyFrame, timeframe: str = "M1") -> pl.LazyFrame:
        """基本統計系特徴量の計算（高速化対応）"""
        exprs = []
        p = self.prefix
        lookback_bars = self.config.timeframe_bars_per_day.get(timeframe, 1440)
        # ▼▼ 修正後: atr_safeの取得
        atr_safe = self._get_atr_safe_expr()

        # 基本統計系 - 全ての式に明示的なaliasを付与 (絶対値からATR相対値への変換)
        for window in self.config.window_sizes["general"]:
            exprs.append(
                (
                    (pl.col("close").rolling_mean(window) - pl.col("close")) / atr_safe
                ).alias(f"{p}rolling_mean_{window}")
            )
            exprs.append(
                (pl.col("close").rolling_std(window, ddof=1) / atr_safe).alias(
                    f"{p}rolling_std_{window}"
                )
            )
            exprs.append(
                (pl.col("close").rolling_var(window, ddof=1) / (atr_safe**2)).alias(
                    f"{p}rolling_var_{window}"
                )
            )
            exprs.append(
                (
                    (pl.col("close").rolling_median(window) - pl.col("close"))
                    / atr_safe
                ).alias(f"{p}rolling_median_{window}")
            )
            exprs.append(
                (
                    (pl.col("close").rolling_min(window) - pl.col("close")) / atr_safe
                ).alias(f"{p}rolling_min_{window}")
            )
            exprs.append(
                (
                    (pl.col("close").rolling_max(window) - pl.col("close")) / atr_safe
                ).alias(f"{p}rolling_max_{window}")
            )

        return lazy_frame.with_columns(exprs)

    def _create_composite_features(
        self, lazy_frame: pl.LazyFrame, timeframe: str = "M1"
    ) -> pl.LazyFrame:
        """複合計算系特徴量の計算（高速化対応）

        【Phase 5 修正 (#36)】既存バグ修正:
            旧実装は `lookback_bars` 変数を未定義のまま L1089 で参照しており、
            composite グループの計算で NameError を発生させていた。
            timeframe 引数を追加し、メソッド内で lookback_bars を計算するよう修正。
            (他 engine の同型メソッドと同じパターン)
        """
        exprs = []
        p = self.prefix
        atr_safe = self._get_atr_safe_expr()

        # 【Phase 5 修正】timeframe に応じた lookback_bars (1日分のバー数) を計算
        # M0.5: 2880, M1: 1440, M3: 480, ..., D1: 1 等
        lookback_bars = self.config.timeframe_bars_per_day.get(timeframe, 1440)

        # 複合計算系 - ゼロ除算防止(1e-10)とATRスケール不変性の適用
        for window in [20, 50]:
            mean_col = pl.col("close").rolling_mean(window)
            std_col = pl.col("close").rolling_std(window, ddof=1)
            exprs.append(
                ((pl.col("close") - mean_col) / (std_col + 1e-10)).alias(
                    f"{p}zscore_{window}"
                )
            )
            exprs.append(
                (((mean_col + 2 * std_col) - pl.col("close")) / atr_safe).alias(
                    f"{p}bollinger_upper_{window}"
                )
            )
            exprs.append(
                (((mean_col - 2 * std_col) - pl.col("close")) / atr_safe).alias(
                    f"{p}bollinger_lower_{window}"
                )
            )

        # 基本データ処理特徴量
        exprs.append(pl.col("close").pct_change().alias(f"{p}price_change"))
        exprs.append(
            pl.col("close")
            .pct_change()
            .rolling_std(20, ddof=1)
            .alias(f"{p}volatility_20")
        )
        exprs.append(
            ((pl.col("high") - pl.col("low")) / atr_safe).alias(f"{p}price_range")
        )

        # ▼▼ 修正後: Relative Volume の適用
        rel_volume = pl.col("volume") / (pl.col("volume").rolling_mean(lookback_bars) + 1e-10)
        exprs.append(rel_volume.rolling_mean(20).alias(f"{p}volume_ma20"))
        exprs.append(
            (pl.col("close").pct_change() * rel_volume)
            .rolling_mean(10)
            .alias(f"{p}volume_price_trend")
        )

        return lazy_frame.with_columns(exprs)

    def _create_timeseries_features(self, lazy_frame: pl.LazyFrame) -> pl.LazyFrame:
        """時系列解析系特徴量の計算（高速化対応）"""
        exprs = []
        p = self.prefix

        # Numba UDF統合系 - rolling_mapを使用（仕様書準拠）- min_periods非推奨警告対応
        for window in [50, 100]:
            exprs.append(
                pl.col("close")
                .pct_change()
                .rolling_map(
                    lambda s: adf_統計量_udf(s.to_numpy()),
                    window_size=window,
                    min_samples=window,
                )
                .alias(f"{p}adf_statistic_{window}")
            )

            exprs.append(
                pl.col("close")
                .pct_change()
                .rolling_map(
                    lambda s: phillips_perron_統計量_udf(s.to_numpy()),
                    window_size=window,
                    min_samples=window,
                )
                .alias(f"{p}pp_statistic_{window}")
            )

            exprs.append(
                pl.col("close")
                .pct_change()
                .rolling_map(
                    lambda s: kpss_統計量_udf(s.to_numpy()),
                    window_size=window,
                    min_samples=window,
                )
                .alias(f"{p}kpss_statistic_{window}")
            )
        return lazy_frame.with_columns(exprs)

    def _create_exponential_arima_features(
        self, lazy_frame: pl.LazyFrame
    ) -> pl.LazyFrame:
        """指数平滑・ARIMA系特徴量の計算（高速化対応）"""
        exprs = []
        p = self.prefix
        # ▼▼ 修正後: atr_safeの取得
        atr_safe = self._get_atr_safe_expr()

        for window in [50, 100]:
            exprs.append(
                (
                    (
                        pl.col("close").rolling_map(
                            lambda s: holt_winters_レベル_udf(s.to_numpy()),
                            window_size=window,
                            min_samples=window,
                        )
                        - pl.col("close")
                    )
                    / atr_safe
                ).alias(f"{p}holt_level_{window}")
            )

            exprs.append(
                (
                    pl.col("close").rolling_map(
                        lambda s: holt_winters_トレンド_udf(s.to_numpy()),
                        window_size=window,
                        min_samples=window,
                    )
                    / atr_safe
                ).alias(f"{p}holt_trend_{window}")
            )

            exprs.append(
                (
                    pl.col("close").rolling_map(
                        lambda s: arima_残差分散_udf(s.to_numpy()),
                        window_size=window,
                        min_samples=window,
                    )
                    / (atr_safe**2)
                ).alias(f"{p}arima_residual_var_{window}")
            )

        return lazy_frame.with_columns(exprs)

    def _create_kalman_regression_features(
        self, lazy_frame: pl.LazyFrame
    ) -> pl.LazyFrame:
        """カルマン・回帰系特徴量の計算（高速化対応）"""
        exprs = []
        p = self.prefix
        atr_safe = self._get_atr_safe_expr()

        for window in [50, 100]:
            exprs.append(
                (
                    (
                        pl.col("close").rolling_map(
                            lambda s: kalman_状態推定_udf(s.to_numpy()),
                            window_size=window,
                            min_samples=window,
                        )
                        - pl.col("close")
                    )
                    / atr_safe
                ).alias(f"{p}kalman_state_{window}")
            )

            exprs.append(
                (
                    (
                        pl.col("close").rolling_map(
                            lambda s: lowess_適合値_udf(s.to_numpy()),
                            window_size=window,
                            min_samples=window,
                        )
                        - pl.col("close")
                    )
                    / atr_safe
                ).alias(f"{p}lowess_fitted_{window}")
            )

            exprs.append(
                (
                    pl.col("close").rolling_map(
                        lambda s: theil_sen_傾き_udf(s.to_numpy()),
                        window_size=window,
                        min_samples=window,
                    )
                    / atr_safe
                ).alias(f"{p}theil_sen_slope_{window}")
            )

        return lazy_frame.with_columns(exprs)

    def _create_distributions_features(self, lazy_frame: pl.LazyFrame) -> pl.LazyFrame:
        """分布パラメータ系特徴量の計算（高速化対応）"""
        exprs = []
        p = self.prefix

        # 分布パラメータ用（リターン系列ベース）- pct_change()を直接使用 + alias追加
        exprs.append(
            pl.col("close")
            .pct_change()
            .rolling_map(
                lambda s: t分布_自由度_udf(s.to_numpy()), window_size=50, min_samples=50
            )
            .alias(f"{p}t_dist_dof_50")
        )

        exprs.append(
            (
                (
                    pl.col("close")
                    .pct_change()
                    .rolling_map(
                        lambda s: t分布_尺度_udf(s.to_numpy()),
                        window_size=50,
                        min_samples=50,
                    )
                )
                / (pl.col("close").pct_change().rolling_std(20, ddof=1) + 1e-10)
            ).alias(f"{p}t_dist_scale_50")
        )

        exprs.append(
            pl.col("high")
            .rolling_map(
                lambda s: gev_形状_udf(s.to_numpy()), window_size=50, min_samples=50
            )
            .alias(f"{p}gev_shape_50")
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

        # ▼▼ 修正後: Engine 1B 専用のサブフォルダ名を定義 (共有事項2対応)
        self.subfolder_name = "feature_value_a_vast_universeB"

        # 出力設定（Project Forge基準）
        self.output_config = {
            "compression": "snappy",  # 固定値
            "dtype": "float64",  # 金融データの精度重視
            "timestamp_handling": "column",  # 機械学習での柔軟性重視
        }

        # エンジン識別子
        self.engine_id = config.engine_id

    def create_output_path(self, timeframe: str) -> Path:
        """出力パス生成 - Project Forge命名規則 (サブフォルダ階層維持版)"""
        filename = f"features_{self.engine_id}_{timeframe}.parquet"
        # ▼▼ 修正後: config.S2_FEATURES とファイル名の間に専用サブフォルダを挿入
        return Path(self.config.output_path) / self.subfolder_name / filename

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

            # ▼▼ 修正後: 重複エラーを回避しつつデータ型を厳格化 (共有事項1対応)
            schema = processed_frame.collect_schema()
            cast_exprs = []

            # 1. timeframeは最優先でUtf8に固定（重複登録を防ぐため、以降の判定から除外）
            if "timeframe" in schema.names():
                cast_exprs.append(pl.col("timeframe").cast(pl.Utf8))

            # 2. その他のパーティション用カラムなどを個別に指定してキャスト
            if "timestamp" in schema.names():
                cast_exprs.append(pl.col("timestamp").cast(pl.Datetime("us")))
            if "year" in schema.names():
                cast_exprs.append(pl.col("year").cast(pl.Int32))
            if "month" in schema.names():
                cast_exprs.append(pl.col("month").cast(pl.Int32))
            if "day" in schema.names():
                cast_exprs.append(pl.col("day").cast(pl.Int32))

            if cast_exprs:
                processed_frame = processed_frame.with_columns(cast_exprs)

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
            "engine_name": "Engine_1B_VastUniverse",
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
    """
    logging.info("【実行モード】日単位でのTickデータ特徴量計算を開始します...")

    timeframe = "tick"
    # ▼▼ 修正後: 保存先を Engine 1B 専用サブフォルダ配下に設定 (共有事項2対応)
    subfolder_name = "feature_value_a_vast_universeB"
    PARTITION_ROOT = Path(config.partitioned_tick_path)
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

            # 【修正】親方（この関数）が工程管理を行う物理的垂直分割
            logging.info(f"特徴量計算開始: {day_name}")

            # 一時ディレクトリ作成（日次処理用）
            temp_dir = Path(tempfile.mkdtemp(prefix=f"day_{i:04d}_{config.engine_id}_"))
            logging.info(f"日次一時ディレクトリ作成: {temp_dir}")

            temp_files = []

            # 特徴量グループを取得
            feature_groups = calculation_engine.get_feature_groups()
            logging.info(f"物理的垂直分割: {len(feature_groups)}グループで処理")

            # ▼▼ 修正後: 全グループの処理前に ATR を一括で事前計算・注入
            base_lf = calculation_engine.inject_temp_atr(augmented_df.lazy())

            # 各グループを順次処理（親方が工程管理）
            for group_idx, (group_name, group_expressions) in enumerate(
                feature_groups.items()
            ):
                logging.info(
                    f"グループ処理開始: {group_name} ({len(group_expressions)}個の特徴量)"
                )

                # 1. 下請けに「このグループだけ計算しろ」と指示
                group_result_lf = calculation_engine.calculate_one_group(
                    base_lf,
                    group_name,
                    group_expressions,
                    timeframe=timeframe,
                )

                # ▼▼ 修正前: group_result_df = group_result_lf.collect(streaming=True)
                # ▼▼ 修正後: Polarsの最新仕様 (engine="streaming") に準拠
                # 2. 親方が自らメモリに実現化（単一グループなので安全）
                group_result_df = group_result_lf.collect(
                    engine="streaming"
                )  # メモリ効率のためストリーミング実行
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
                "sample_weight",
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
            # ▼▼ 修正前: 型指定なしでyear/month/dayを追加
            # ▼▼ 修正後: Int32・Datetime[us]・Utf8への厳密な型キャストを追加
            final_df = result_df.with_columns(
                [
                    pl.col("timestamp").dt.year().cast(pl.Int32).alias("year"),
                    pl.col("timestamp").dt.month().cast(pl.Int32).alias("month"),
                    pl.col("timestamp").dt.day().cast(pl.Int32).alias("day"),
                    pl.col("timestamp").cast(pl.Datetime("us")),
                    pl.col("timeframe").cast(pl.Utf8),
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

        # ▼▼ 修正前: with_columnsによる一括計算
        # all_expressions = []
        # feature_groups = calc_engine.get_feature_groups()
        # for group_name, group_expressions in feature_groups.items():
        #     all_expressions.extend(group_expressions.values())
        #
        # logger.info(f"特徴量計算開始: {len(all_expressions)}個の特徴量を {timeframe} に対して計算します。")
        # features_lf = lazy_frame.with_columns(all_expressions)
        #
        # # 品質保証システムを適用
        # features_lf = calc_engine.apply_quality_assurance(features_lf)

        # ▼▼ 修正後: 物理的垂直分割・完全ディスクベース結合ロジックへの置換
        import gc

        feature_groups = calc_engine.get_feature_groups()
        temp_dir = calc_engine.temp_dir / f"tf_{timeframe}"
        temp_dir.mkdir(parents=True, exist_ok=True)
        temp_files = []

        base_df = lazy_frame.collect()
        # ▼▼ 修正: ATR13を事前注入してからグループ計算に渡す
        base_lf_with_atr = calc_engine.inject_temp_atr(base_df.lazy())

        for group_idx, (group_name, group_expressions) in enumerate(
            feature_groups.items()
        ):
            logger.info(
                f"グループ処理開始: {group_name} ({len(group_expressions)}個の特徴量)"
            )

            group_result_lf = calc_engine.calculate_one_group(
                base_lf_with_atr, group_name, group_expressions, timeframe=timeframe
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
    print("  - 時系列解析特徴量（ADF、PP、KPSS検定等）")
    print("  - 分布パラメータ特徴量（t分布、GEV分布）")
    print("  - 指数平滑特徴量（ホルト・ウィンターズ）")
    print("  - 回帰予測特徴量（ARIMA、カルマンフィルタ、頑健回帰）")
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
    print(f"  Engine 1B - A Vast Universe of Features (修正版) ")
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
