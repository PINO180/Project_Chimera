# pipelines/streamlined_intelligent_timeseries_synthesis.py

import dask_cudf
import cupy as cp
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Union, Tuple, Any
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
import logging
import gc
import time
from tqdm import tqdm
import psutil
from sklearn.linear_model import LinearRegression
from scipy import stats
import warnings
import json
from datetime import datetime

# ==============================================================================
# --- 事前データ品質分析 (PRE-ANALYSIS DATA QUALITY) ---
# ==============================================================================

@dataclass
class QualityIssue:
    """品質問題の記録"""
    severity: str  # "LOW", "MEDIUM", "HIGH", "CRITICAL"
    category: str  # "missing", "outlier", "variance", "cardinality", "schema"
    description: str
    recommendation: str
    affected_features: List[str] = field(default_factory=list)

@dataclass
class PreAnalysisReport:
    """事前分析レポート"""
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    total_rows: int = 0
    total_columns: int = 0
    sample_size: int = 0
    
    issues: List[QualityIssue] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)
    
    overall_risk: str = "UNKNOWN"
    proceed_recommendation: bool = True
    estimated_processing_time: float = 0.0

class SimpleDataQualityAnalyzer:
    """軽量な事前データ品質分析"""
    
    def __init__(self, sample_size: int = 100_000):
        self.sample_size = sample_size
        self.logger = logging.getLogger(self.__class__.__name__)
    
    def analyze_sample(self, df: pd.DataFrame) -> PreAnalysisReport:
        """データサンプルの簡潔な品質分析"""
        self.logger.info(f"事前品質分析開始 (サンプルサイズ: {len(df):,})")
        
        report = PreAnalysisReport(
            total_rows=len(df),
            total_columns=len(df.columns),
            sample_size=min(len(df), self.sample_size)
        )
        
        # サンプリング
        if len(df) > self.sample_size:
            df_sample = df.sample(n=self.sample_size, random_state=42)
        else:
            df_sample = df
        
        # 簡潔なチェック実行
        self._check_critical_issues(df_sample, report)
        self._check_feature_quality(df_sample, report)
        self._assess_overall_risk(report)
        
        return report
    
    def _check_critical_issues(self, df: pd.DataFrame, report: PreAnalysisReport):
        """重大な問題のみチェック"""
        
        # 1. 全欠損列
        completely_missing = df.isnull().all()
        if completely_missing.any():
            missing_cols = completely_missing[completely_missing].index.tolist()
            report.issues.append(QualityIssue(
                severity="CRITICAL",
                category="missing",
                description=f"{len(missing_cols)}列が完全に欠損",
                recommendation="これらの列は削除を検討",
                affected_features=missing_cols
            ))
        
        # 2. 定数列
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        constant_cols = []
        for col in numeric_cols:
            if df[col].nunique() <= 1:
                constant_cols.append(col)
        
        if constant_cols:
            report.issues.append(QualityIssue(
                severity="HIGH",
                category="variance",
                description=f"{len(constant_cols)}列が定数値",
                recommendation="定数列は自動削除されます",
                affected_features=constant_cols
            ))
        
        # 3. 極端な欠損率
        high_missing_cols = []
        for col in df.columns:
            missing_rate = df[col].isnull().mean()
            if missing_rate > 0.8:  # 80%以上欠損
                high_missing_cols.append(col)
        
        if high_missing_cols:
            report.issues.append(QualityIssue(
                severity="MEDIUM",
                category="missing",
                description=f"{len(high_missing_cols)}列で80%以上欠損",
                recommendation="補完戦略または除外を検討",
                affected_features=high_missing_cols
            ))
    
    def _check_feature_quality(self, df: pd.DataFrame, report: PreAnalysisReport):
        """基本的な特徴量品質チェック"""
        
        # カーディナリティチェック
        high_cardinality_cols = []
        for col in df.columns:
            if col in ['timestamp', 'datetime']:
                continue
            cardinality_ratio = df[col].nunique() / len(df)
            if cardinality_ratio > 0.95:  # 95%以上がユニーク
                high_cardinality_cols.append(col)
        
        if high_cardinality_cols:
            report.issues.append(QualityIssue(
                severity="MEDIUM", 
                category="cardinality",
                description=f"{len(high_cardinality_cols)}列で高カーディナリティ",
                recommendation="エンコーディング戦略を考慮",
                affected_features=high_cardinality_cols
            ))
        
        # 基本統計チェック
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        outlier_cols = []
        for col in numeric_cols:
            if df[col].isnull().all():
                continue
            try:
                z_scores = np.abs(stats.zscore(df[col].dropna()))
                outlier_rate = (z_scores > 4).mean()  # 4σ以上
                if outlier_rate > 0.05:  # 5%以上が外れ値
                    outlier_cols.append(col)
            except:
                continue
        
        if outlier_cols:
            report.issues.append(QualityIssue(
                severity="LOW",
                category="outlier", 
                description=f"{len(outlier_cols)}列で多数の外れ値",
                recommendation="ウィンザライゼーションまたは対数変換を検討",
                affected_features=outlier_cols
            ))
    
    def _assess_overall_risk(self, report: PreAnalysisReport):
        """総合リスク評価"""
        critical_count = sum(1 for issue in report.issues if issue.severity == "CRITICAL")
        high_count = sum(1 for issue in report.issues if issue.severity == "HIGH")
        medium_count = sum(1 for issue in report.issues if issue.severity == "MEDIUM")
        
        if critical_count > 0:
            report.overall_risk = "CRITICAL"
            report.proceed_recommendation = False
            report.recommendations.append("重大な品質問題があります。データクリーニングが必要です。")
        elif high_count > 2:
            report.overall_risk = "HIGH" 
            report.proceed_recommendation = True
            report.recommendations.append("品質問題がありますが、処理継続可能です。")
        elif medium_count > 3:
            report.overall_risk = "MEDIUM"
            report.proceed_recommendation = True
            report.recommendations.append("軽微な品質問題があります。監視推奨。")
        else:
            report.overall_risk = "LOW"
            report.proceed_recommendation = True
            report.recommendations.append("データ品質は良好です。")

# ==============================================================================
# --- 軽量化された設定 (STREAMLINED CONFIGURATION) ---
# ==============================================================================

@dataclass
class StreamlinedConfig:
    """シンプルな統合設定"""
    
    # 基本処理設定
    chunk_size_rows: int = 500_000
    max_memory_gb: float = 8.0
    output_precision: str = "float32"
    compression: str = "snappy"
    
    # 品質チェック設定
    enable_pre_analysis: bool = True
    pre_analysis_sample_size: int = 100_000
    
    # 時間足別窓サイズ (簡略化)
    timeframe_windows: Dict[str, List[int]] = field(default_factory=lambda: {
        "tick": [10, 50, 100],
        "M0.5": [4, 12, 24], 
        "M1": [5, 15, 60],
        "M3": [5, 20, 80],
        "M5": [3, 12, 48],
        "M8": [4, 15, 45],
        "M15": [4, 16, 32],
        "M30": [2, 8, 24],
        "H1": [4, 12, 24],
        "H4": [3, 6, 18],
        "H6": [2, 8, 20], 
        "H12": [2, 7, 14],
        "D1": [5, 20, 50],
        "W1": [4, 13, 26],
        "MN": [3, 6, 12]
    })
    
    # 演算子配分 (簡略化)
    operator_distribution: Dict[str, List[str]] = field(default_factory=lambda: {
        "HIGH_FREQ": ["rolling_mean", "rolling_std", "ewm", "zscore"],
        "MID_FREQ": ["zscore", "slope", "diff", "pct_change", "rolling_mean"], 
        "LOW_FREQ": ["rolling_skew", "rolling_kurt", "decay", "slope"]
    })
    
    # 時間足カテゴリ
    timeframe_categories: Dict[str, str] = field(default_factory=lambda: {
        "tick": "HIGH_FREQ", "M0.5": "HIGH_FREQ", "M1": "HIGH_FREQ", "M3": "HIGH_FREQ",
        "M5": "MID_FREQ", "M8": "MID_FREQ", "M15": "MID_FREQ", "M30": "MID_FREQ", "H1": "MID_FREQ",
        "H4": "LOW_FREQ", "H6": "LOW_FREQ", "H12": "LOW_FREQ", "D1": "LOW_FREQ", "W1": "LOW_FREQ", "MN": "LOW_FREQ"
    })

# ==============================================================================
# --- 時系列演算子 (TIMESERIES OPERATORS) ---
# ==============================================================================

class TimeSeriesOperator(ABC):
    @property
    @abstractmethod
    def name(self) -> str:
        pass
    
    @abstractmethod
    def apply(self, series: pd.Series, window: int = None) -> pd.Series:
        pass

# 演算子実装 (軽量化)
class RollingMeanOperator(TimeSeriesOperator):
    @property
    def name(self) -> str:
        return "rolling_mean"
    
    def apply(self, series: pd.Series, window: int) -> pd.Series:
        return series.rolling(window, min_periods=1).mean()

class RollingStdOperator(TimeSeriesOperator):
    @property
    def name(self) -> str:
        return "rolling_std"
    
    def apply(self, series: pd.Series, window: int) -> pd.Series:
        return series.rolling(window, min_periods=1).std()

class ZScoreOperator(TimeSeriesOperator):
    @property
    def name(self) -> str:
        return "zscore"
    
    def apply(self, series: pd.Series, window: int) -> pd.Series:
        rolling_mean = series.rolling(window, min_periods=1).mean()
        rolling_std = series.rolling(window, min_periods=1).std()
        return (series - rolling_mean) / (rolling_std + 1e-9)

class SlopeOperator(TimeSeriesOperator):
    @property
    def name(self) -> str:
        return "slope"
    
    def apply(self, series: pd.Series, window: int) -> pd.Series:
        def calc_slope(x):
            if len(x) < 2:
                return 0.0
            try:
                y = x.values.reshape(-1, 1)
                X = np.arange(len(x)).reshape(-1, 1)
                model = LinearRegression().fit(X, y)
                return model.coef_[0][0]
            except:
                return 0.0
        return series.rolling(window, min_periods=2).apply(calc_slope, raw=False)

class DiffOperator(TimeSeriesOperator):
    @property
    def name(self) -> str:
        return "diff"
    
    def apply(self, series: pd.Series, window: int) -> pd.Series:
        return series.diff(window)

class PctChangeOperator(TimeSeriesOperator):
    @property
    def name(self) -> str:
        return "pct_change"
    
    def apply(self, series: pd.Series, window: int) -> pd.Series:
        return series.pct_change(window)

class EWMOperator(TimeSeriesOperator):
    @property
    def name(self) -> str:
        return "ewm"
    
    def apply(self, series: pd.Series, window: int) -> pd.Series:
        return series.ewm(span=window, min_periods=1).mean()

class DecayOperator(TimeSeriesOperator):
    @property
    def name(self) -> str:
        return "decay"
    
    def apply(self, series: pd.Series, window: int) -> pd.Series:
        return series.ewm(halflife=window, min_periods=1).mean()

class RollingSkewOperator(TimeSeriesOperator):
    @property
    def name(self) -> str:
        return "rolling_skew"
    
    def apply(self, series: pd.Series, window: int) -> pd.Series:
        return series.rolling(window, min_periods=3).skew()

class RollingKurtOperator(TimeSeriesOperator):
    @property
    def name(self) -> str:
        return "rolling_kurt"
    
    def apply(self, series: pd.Series, window: int) -> pd.Series:
        return series.rolling(window, min_periods=4).kurt()

# ==============================================================================
# --- 軽量化された処理エンジン (STREAMLINED PROCESSING ENGINE) ---
# ==============================================================================

class StreamlinedProcessor:
    """高速化された特徴量生成処理"""
    
    def __init__(self, config: StreamlinedConfig):
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # 演算子マップ
        self.operators = {
            "rolling_mean": RollingMeanOperator(),
            "rolling_std": RollingStdOperator(),
            "zscore": ZScoreOperator(),
            "slope": SlopeOperator(),
            "diff": DiffOperator(),
            "pct_change": PctChangeOperator(),
            "ewm": EWMOperator(),
            "decay": DecayOperator(),
            "rolling_skew": RollingSkewOperator(),
            "rolling_kurt": RollingKurtOperator(),
        }
    
    def get_config_for_timeframe(self, timeframe: str) -> Tuple[List[str], List[int]]:
        """時間足の設定を取得"""
        category = self.config.timeframe_categories.get(timeframe, "MID_FREQ")
        operators = self.config.operator_distribution[category]
        windows = self.config.timeframe_windows.get(timeframe, [5, 20, 50])
        
        return operators, windows
    
    def process_chunk(self, df_chunk: pd.DataFrame, features: List[str], timeframe: str) -> pd.DataFrame:
        """チャンク処理 (品質チェックなし、高速化)"""
        
        operators, windows = self.get_config_for_timeframe(timeframe)
        new_features = {}
        
        for feature in features:
            if feature not in df_chunk.columns:
                continue
            
            series = df_chunk[feature]
            
            for op_name in operators:
                if op_name not in self.operators:
                    continue
                
                operator = self.operators[op_name]
                
                # 窓ベース演算子
                if op_name in ["rolling_mean", "rolling_std", "zscore", "slope", "ewm", "decay", "rolling_skew", "rolling_kurt"]:
                    for window in windows:
                        feature_name = f"TS_{op_name}_{window}_OF_{feature}"
                        try:
                            result = operator.apply(series, window)
                            new_features[feature_name] = result.astype(self.config.output_precision)
                        except Exception as e:
                            self.logger.warning(f"演算エラー {feature_name}: {e}")
                
                # 期間ベース演算子
                elif op_name in ["diff", "pct_change"]:
                    periods = [w for w in windows if w <= 20]  # 適度な期間のみ
                    for period in periods:
                        feature_name = f"TS_{op_name}_{period}_OF_{feature}"
                        try:
                            result = operator.apply(series, period)
                            new_features[feature_name] = result.astype(self.config.output_precision)
                        except Exception as e:
                            self.logger.warning(f"演算エラー {feature_name}: {e}")
        
        # 新特徴量追加
        for name, values in new_features.items():
            df_chunk[name] = values
        
        return df_chunk

# ==============================================================================
# --- メインパイプライン (MAIN PIPELINE) ---
# ==============================================================================

class StreamlinedTimeSeriesPipeline:
    """軽量化された時系列特徴量生成パイプライン"""
    
    def __init__(self, input_path: str, output_dir: str, config: StreamlinedConfig):
        self.input_path = Path(input_path)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)
        self.processor = StreamlinedProcessor(config)
        
        # 基本特徴量
        self.base_features = [
            "open", "high", "low", "close", "volume",
            "tick_count", "bid", "ask", "spread", "volatility",
            "avg_volume", "price_change_pct", "range", "body_size"
        ]
        
        self.quality_analyzer = SimpleDataQualityAnalyzer(config.pre_analysis_sample_size)
    
    def run_pre_analysis(self) -> PreAnalysisReport:
        """事前分析実行"""
        if not self.config.enable_pre_analysis:
            return PreAnalysisReport(proceed_recommendation=True)
        
        self.logger.info("事前データ品質分析開始...")
        
        # サンプルデータ読み込み
        ddf = dask_cudf.read_parquet(self.input_path)
        sample_size = min(self.config.pre_analysis_sample_size, len(ddf))
        
        sample_df = ddf.sample(n=sample_size, random_state=42).compute()
        
        # 品質分析実行
        report = self.quality_analyzer.analyze_sample(sample_df)
        
        # レポート保存
        self._save_pre_analysis_report(report)
        
        # 結果表示
        self._display_pre_analysis_results(report)
        
        return report
    
    def _save_pre_analysis_report(self, report: PreAnalysisReport):
        """事前分析レポート保存"""
        report_path = self.output_dir / "pre_analysis_report.json"
        
        report_dict = {
            'timestamp': report.timestamp,
            'data_overview': {
                'total_rows': report.total_rows,
                'total_columns': report.total_columns,
                'sample_size': report.sample_size
            },
            'quality_issues': [
                {
                    'severity': issue.severity,
                    'category': issue.category,
                    'description': issue.description,
                    'recommendation': issue.recommendation,
                    'affected_features': issue.affected_features
                } for issue in report.issues
            ],
            'overall_assessment': {
                'risk_level': report.overall_risk,
                'proceed_recommendation': report.proceed_recommendation,
                'recommendations': report.recommendations
            }
        }
        
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(report_dict, f, indent=2, ensure_ascii=False)
        
        self.logger.info(f"事前分析レポート保存: {report_path}")
    
    def _display_pre_analysis_results(self, report: PreAnalysisReport):
        """事前分析結果表示"""
        self.logger.info("=" * 50)
        self.logger.info("事前データ品質分析結果")
        self.logger.info("=" * 50)
        self.logger.info(f"データ概要: {report.total_rows:,}行 x {report.total_columns}列")
        self.logger.info(f"サンプルサイズ: {report.sample_size:,}行")
        self.logger.info(f"総合リスク: {report.overall_risk}")
        
        if report.issues:
            self.logger.info(f"\n検出された問題 ({len(report.issues)}件):")
            for issue in report.issues:
                self.logger.info(f"  {issue.severity}: {issue.description}")
        
        if report.recommendations:
            self.logger.info(f"\n推奨事項:")
            for rec in report.recommendations:
                self.logger.info(f"  - {rec}")
        
        self.logger.info("=" * 50)
    
    def run_main_processing(self) -> None:
        """メイン処理実行"""
        self.logger.info("メイン特徴量生成処理開始")
        
        start_time = time.time()
        
        # データ読み込み
        ddf = dask_cudf.read_parquet(self.input_path)
        unique_timeframes = ddf['timeframe'].unique().compute().tolist()
        
        # 時間足別処理
        all_chunks = []
        
        for timeframe in tqdm(unique_timeframes, desc="時間足処理"):
            self.logger.info(f"時間足 '{timeframe}' 処理中...")
            
            tf_ddf = ddf[ddf['timeframe'] == timeframe]
            total_rows = len(tf_ddf)
            
            # チャンク分割
            chunk_size = min(self.config.chunk_size_rows, total_rows)
            n_chunks = (total_rows + chunk_size - 1) // chunk_size
            
            for chunk_idx in tqdm(range(n_chunks), desc=f"Chunks ({timeframe})", leave=False):
                start_idx = chunk_idx * chunk_size
                end_idx = min(start_idx + chunk_size, total_rows)
                
                chunk_df = tf_ddf.iloc[start_idx:end_idx].compute()
                
                # メモリチェック
                memory = psutil.virtual_memory()
                if memory.percent > 85:
                    gc.collect()
                
                # 特徴量生成
                processed_chunk = self.processor.process_chunk(
                    chunk_df, self.base_features, timeframe
                )
                
                all_chunks.append(processed_chunk)
                
                # 定期的な中間保存
                if len(all_chunks) >= 50:  # 50チャンクごと
                    self._save_intermediate_chunks(all_chunks, timeframe, chunk_idx)
                    all_chunks = []
                    gc.collect()
        
        # 残りのチャンク保存
        if all_chunks:
            self._save_intermediate_chunks(all_chunks, "final", 999999)
        
        # 最終結合
        self._combine_final_results()
        
        total_time = time.time() - start_time
        self.logger.info(f"メイン処理完了: {total_time/60:.1f}分")
    
    def _save_intermediate_chunks(self, chunks: List[pd.DataFrame], timeframe: str, chunk_idx: int):
        """中間チャンク保存"""
        if not chunks:
            return
        
        combined = pd.concat(chunks, ignore_index=True)
        filename = f"intermediate_{timeframe}_{chunk_idx:06d}.parquet"
        filepath = self.output_dir / filename
        
        combined.to_parquet(filepath, compression=self.config.compression)
        self.logger.debug(f"中間保存: {len(combined):,}行")
    
    def _combine_final_results(self):
        """最終結果結合"""
        self.logger.info("最終結果結合中...")
        
        intermediate_files = list(self.output_dir.glob("intermediate_*.parquet"))
        if not intermediate_files:
            raise FileNotFoundError("中間ファイルが見つかりません")
        
        final_ddf = dask_cudf.read_parquet(intermediate_files)
        output_path = self.output_dir / "final_timeseries_features.parquet"
        
        final_ddf.to_parquet(output_path, compression=self.config.compression)
        
        # 中間ファイル削除
        for file in intermediate_files:
            file.unlink()
        
        # 統計出力
        final_shape = (len(final_ddf), len(final_ddf.columns))
        file_size_gb = output_path.stat().st_size / (1024**3)
        
        self.logger.info(f"最終ファイル: {output_path}")
        self.logger.info(f"データ形状: {final_shape[0]:,} x {final_shape[1]:,}")
        self.logger.info(f"ファイルサイズ: {file_size_gb:.2f} GB")
    
    def run(self) -> bool:
        """完全パイプライン実行"""
        try:
            # 事前分析
            if self.config.enable_pre_analysis:
                report = self.run_pre_analysis()
                
                if not report.proceed_recommendation:
                    self.logger.error("事前分析により処理継続不可と判定されました")
                    return False
                
                if report.overall_risk in ["HIGH", "CRITICAL"]:
                    response = input(f"リスクレベル: {report.overall_risk}. 続行しますか? (y/N): ")
                    if response.lower() != 'y':
                        self.logger.info("ユーザーにより処理キャンセル")
                        return False
            
            # メイン処理
            self.run_main_processing()
            
            self.logger.info("パイプライン正常完了")
            return True
            
        except Exception as e:
            self.logger.error(f"パイプライン実行エラー: {e}", exc_info=True)
            return False

# ==============================================================================
# --- 設定バリエーション (CONFIGURATION VARIANTS) ---
# ==============================================================================

def create_conservative_config() -> StreamlinedConfig:
    """保守的設定 (ストレージ節約)"""
    config = StreamlinedConfig()
    
    # 窓サイズ削減
    for timeframe in config.timeframe_windows:
        config.timeframe_windows[timeframe] = config.timeframe_windows[timeframe][:2]
    
    # 演算子削減
    config.operator_distribution = {
        "HIGH_FREQ": ["rolling_mean", "rolling_std"],
        "MID_FREQ": ["zscore", "slope"],
        "LOW_FREQ": ["rolling_skew", "decay"]
    }
    
    return config

def create_performance_config() -> StreamlinedConfig:
    """高性能設定 (特徴量最大化)"""
    config = StreamlinedConfig()
    
    # より大きなチャンクサイズ
    config.chunk_size_rows = 1_000_000
    
    # より多くの演算子
    config.operator_distribution = {
        "HIGH_FREQ": ["rolling_mean", "rolling_std", "ewm", "zscore", "diff"],
        "MID_FREQ": ["zscore", "slope", "diff", "pct_change", "rolling_mean", "rolling_std"], 
        "LOW_FREQ": ["rolling_skew", "rolling_kurt", "decay", "slope", "ewm"]
    }
    
    return config

def estimate_processing_resources(config: StreamlinedConfig, 
                                 n_rows: int = 150_000_000,
                                 n_timeframes: int = 14) -> Dict[str, float]:
    """処理リソース推定"""
    
    # 特徴量数推定
    avg_operators = sum(len(ops) for ops in config.operator_distribution.values()) / 3
    avg_windows = sum(len(windows) for windows in config.timeframe_windows.values()) / len(config.timeframe_windows)
    
    features_per_timeframe = 14 * avg_operators * avg_windows * 0.8  # 80%が窓ベース
    total_features = features_per_timeframe * n_timeframes
    
    # ストレージ推定
    bytes_per_value = 4 if config.output_precision == "float32" else 8
    storage_gb = (n_rows * total_features * bytes_per_value) / (1024**3)
    
    # 処理時間推定
    chunk_count = n_rows / config.chunk_size_rows
    processing_time_minutes = chunk_count * 0.1  # チャンクあたり0.1分と仮定
    
    return {
        "estimated_features": int(total_features),
        "estimated_storage_gb": storage_gb,
        "estimated_processing_minutes": processing_time_minutes,
        "memory_usage_gb": config.chunk_size_rows * 50 / (1024**3)  # チャンクメモリ使用量
    }

# ==============================================================================
# --- ユーティリティ関数 (UTILITY FUNCTIONS) ---
# ==============================================================================

def setup_logging(level: int = logging.INFO) -> None:
    """ログ設定"""
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('streamlined_timeseries_synthesis.log')
        ]
    )

def display_config_comparison():
    """設定比較表示"""
    configs = {
        "保守的": create_conservative_config(),
        "標準": StreamlinedConfig(),
        "高性能": create_performance_config()
    }
    
    print("\n設定比較:")
    print("=" * 60)
    print(f"{'設定':<8} {'推定特徴量数':<12} {'推定サイズ(GB)':<15} {'推定時間(分)':<12}")
    print("-" * 60)
    
    for name, config in configs.items():
        resources = estimate_processing_resources(config)
        print(f"{name:<8} {resources['estimated_features']:<12,} "
              f"{resources['estimated_storage_gb']:<15.1f} "
              f"{resources['estimated_processing_minutes']:<12.1f}")
    
    print("=" * 60)

# ==============================================================================
# --- メイン実行関数 (MAIN EXECUTION FUNCTIONS) ---
# ==============================================================================

def main():
    """メイン実行関数"""
    setup_logging()
    logger = logging.getLogger(__name__)
    
    logger.info("Streamlined Intelligent TimeSeries Pipeline")
    logger.info("=" * 60)
    
    # 設定比較表示
    display_config_comparison()
    
    # 設定選択
    print("\n設定選択:")
    print("1. 標準設定 (推奨)")
    print("2. 保守的設定 (ストレージ節約)")  
    print("3. 高性能設定 (特徴量最大化)")
    
    choice = input("選択してください (1-3): ").strip()
    
    if choice == "2":
        config = create_conservative_config()
        logger.info("保守的設定で実行")
    elif choice == "3":
        config = create_performance_config()
        logger.info("高性能設定で実行")
    else:
        config = StreamlinedConfig()
        logger.info("標準設定で実行")
    
    # リソース推定
    resources = estimate_processing_resources(config)
    
    logger.info("=" * 60)
    logger.info("リソース推定結果:")
    logger.info(f"  推定特徴量数: {resources['estimated_features']:,}")
    logger.info(f"  推定ストレージ: {resources['estimated_storage_gb']:.1f} GB")
    logger.info(f"  推定処理時間: {resources['estimated_processing_minutes']:.1f} 分")
    logger.info(f"  メモリ使用量: {resources['memory_usage_gb']:.1f} GB")
    logger.info("=" * 60)
    
    # 300GB制限チェック
    if resources['estimated_storage_gb'] > 300:
        logger.warning(f"推定ストレージが300GBを超えています: {resources['estimated_storage_gb']:.1f} GB")
        response = input("続行しますか? (y/N): ")
        if response.lower() != 'y':
            logger.info("処理をキャンセルしました")
            return
    
    # パイプライン実行
    pipeline = StreamlinedTimeSeriesPipeline(
        input_path="/workspaces/project_forge/data/1_XAUUSD_base_data/XAUUSDmulti_15timeframe_bars_exness.parquet",
        output_dir="/workspaces/project_forge/data/4_streamlined_timeseries_features/",
        config=config
    )
    
    success = pipeline.run()
    
    if success:
        logger.info("パイプライン正常終了")
    else:
        logger.error("パイプライン実行失敗")

def run_quick_test():
    """クイックテスト実行"""
    setup_logging()
    logger = logging.getLogger(__name__)
    
    # テスト用小設定
    test_config = StreamlinedConfig()
    test_config.chunk_size_rows = 10_000
    test_config.pre_analysis_sample_size = 1_000
    test_config.timeframe_windows = {tf: windows[:2] for tf, windows in test_config.timeframe_windows.items()}
    
    logger.info("クイックテスト開始")
    
    # 小さなテストデータでパイプライン実行
    pipeline = StreamlinedTimeSeriesPipeline(
        input_path="/workspaces/project_forge/data/1_XAUUSD_base_data/XAUUSDmulti_15timeframe_bars_exness.parquet",
        output_dir="/workspaces/project_forge/data/test_output/",
        config=test_config
    )
    
    # 事前分析のみ実行
    report = pipeline.run_pre_analysis()
    logger.info(f"テスト完了 - リスクレベル: {report.overall_risk}")

def analyze_data_only(input_path: str, output_dir: str):
    """データ分析のみ実行"""
    setup_logging()
    
    config = StreamlinedConfig()
    config.enable_pre_analysis = True
    
    pipeline = StreamlinedTimeSeriesPipeline(input_path, output_dir, config)
    report = pipeline.run_pre_analysis()
    
    print(f"分析完了 - 総合リスク: {report.overall_risk}")
    return report

def batch_processing_mode(input_dir: str, output_base_dir: str):
    """バッチ処理モード（複数ファイル処理）"""
    setup_logging()
    logger = logging.getLogger(__name__)
    
    input_path = Path(input_dir)
    parquet_files = list(input_path.glob("*.parquet"))
    
    if not parquet_files:
        logger.error(f"Parquetファイルが見つかりません: {input_dir}")
        return
    
    logger.info(f"{len(parquet_files)}個のファイルをバッチ処理開始")
    
    config = StreamlinedConfig()
    success_count = 0
    
    for file_path in parquet_files:
        try:
            logger.info(f"処理中: {file_path.name}")
            
            output_dir = Path(output_base_dir) / file_path.stem
            
            pipeline = StreamlinedTimeSeriesPipeline(
                str(file_path), str(output_dir), config
            )
            
            if pipeline.run():
                success_count += 1
                logger.info(f"成功: {file_path.name}")
            else:
                logger.error(f"失敗: {file_path.name}")
                
        except Exception as e:
            logger.error(f"エラー処理 {file_path.name}: {e}")
    
    logger.info(f"バッチ処理完了: {success_count}/{len(parquet_files)} 成功")

# ==============================================================================
# --- コマンドライン実行 (COMMAND LINE EXECUTION) ---
# ==============================================================================

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        mode = sys.argv[1].lower()
        
        if mode == "test":
            run_quick_test()
        elif mode == "analyze":
            if len(sys.argv) >= 4:
                analyze_data_only(sys.argv[2], sys.argv[3])
            else:
                analyze_data_only(
                    "/workspaces/project_forge/data/1_XAUUSD_base_data/XAUUSDmulti_15timeframe_bars_exness.parquet",
                    "/workspaces/project_forge/data/analysis_only/"
                )
        elif mode == "batch":
            if len(sys.argv) >= 4:
                batch_processing_mode(sys.argv[2], sys.argv[3])
            else:
                print("使用法: python script.py batch <input_dir> <output_base_dir>")
        elif mode == "compare":
            display_config_comparison()
        else:
            print("使用法:")
            print("  python script.py              # 通常実行")
            print("  python script.py test         # クイックテスト")
            print("  python script.py analyze      # データ分析のみ")
            print("  python script.py batch <in> <out>  # バッチ処理")
            print("  python script.py compare      # 設定比較")
    else:
        main()