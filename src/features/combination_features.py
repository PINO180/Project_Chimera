"""
非線形組み合わせ特徴量エンジン - Project Forge
任意の特徴量DataFrameから非線形組み合わせを大量生成
独立特徴量、MFDFA、Tier1-3すべてに対応可能な汎用エンジン
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union, Set
import logging
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp
from itertools import combinations, permutations
import warnings
warnings.filterwarnings('ignore')
from pathlib import Path

# 数学関数
from scipy.stats import spearmanr, pearsonr
from scipy.special import expit  # sigmoid
from sklearn.preprocessing import StandardScaler, MinMaxScaler

# ロガー設定
logger = logging.getLogger(__name__)

class NonlinearCombinationEngine:
    """
    非線形組み合わせ特徴量生成エンジン
    任意の特徴量DataFrameから数万～数十万の組み合わせ特徴量を生成
    """
    
    def __init__(self, 
                 n_processes: Optional[int] = None,
                 max_combinations: int = 50000,
                 exclude_patterns: List[str] = None,
                 correlation_threshold: float = 0.95):
        """
        初期化
        
        Args:
            n_processes: 並列プロセス数
            max_combinations: 最大組み合わせ数（メモリ制限）
            exclude_patterns: 除外する特徴量パターン
            correlation_threshold: 高相関特徴量除外閾値
        """
        self.n_processes = n_processes or max(1, mp.cpu_count() - 1)
        self.max_combinations = max_combinations
        self.exclude_patterns = exclude_patterns or ['timestamp', 'date', 'id']
        self.correlation_threshold = correlation_threshold
        
        logger.info(f"非線形組み合わせエンジン初期化: {self.n_processes}プロセス, 最大{max_combinations}組み合わせ")
    
    def generate_all_combinations(self, df: pd.DataFrame, 
                                 feature_types: List[str] = None,
                                 chunk_id: str = "") -> pd.DataFrame:
        """
        全非線形組み合わせ特徴量生成
        
        Args:
            df: 入力DataFrame（任意の特徴量）
            feature_types: 生成する組み合わせ種類
            chunk_id: チャンクID（ログ用）
            
        Returns:
            組み合わせ特徴量が追加されたDataFrame
        """
        logger.info(f"非線形組み合わせ開始 - Chunk {chunk_id}, 入力Shape: {df.shape}")
        
        # 特徴量列抽出（数値列のみ、除外パターン適用）
        feature_columns = self._extract_feature_columns(df)
        logger.info(f"有効特徴量列数: {len(feature_columns)}")
        
        # 組み合わせ種類設定
        if feature_types is None:
            feature_types = ['ratio', 'product', 'difference', 'power', 'trigonometric', 
                           'logarithmic', 'exponential', 'statistical']
        
        # 結果用DataFrame
        result_df = df.copy()
        
        # 組み合わせ生成タスク
        combination_tasks = []
        for feature_type in feature_types:
            combination_tasks.append((feature_type, feature_columns))
        
        # 並列処理実行
        with ProcessPoolExecutor(max_workers=self.n_processes) as executor:
            future_to_type = {
                executor.submit(self._generate_combination_type, df[feature_columns], 
                               feature_type, feature_columns): feature_type
                for feature_type, feature_columns in combination_tasks
            }
            
            # 結果回収
            for future in as_completed(future_to_type):
                combination_type = future_to_type[future]
                try:
                    combinations_dict = future.result()
                    
                    # 特徴量マージ（メモリ効率考慮）
                    valid_combinations = 0
                    for col_name, values in combinations_dict.items():
                        if len(result_df) + valid_combinations < self.max_combinations:
                            result_df[col_name] = values
                            valid_combinations += 1
                        else:
                            logger.warning(f"最大組み合わせ数{self.max_combinations}に到達")
                            break
                    
                    logger.info(f"{combination_type} 完了: {len(combinations_dict)} 特徴量")
                    
                except Exception as exc:
                    logger.error(f"{combination_type} でエラー: {exc}")
                    continue
        
        # 高相関特徴量除去
        result_df = self._remove_highly_correlated_features(result_df, feature_columns)
        
        logger.info(f"非線形組み合わせ完了 - Chunk {chunk_id}, 最終Shape: {result_df.shape}")
        return result_df
    
    def _extract_feature_columns(self, df: pd.DataFrame) -> List[str]:
        """有効な特徴量列を抽出"""
        feature_columns = []
        
        for col in df.columns:
            # 数値列チェック
            if not pd.api.types.is_numeric_dtype(df[col]):
                continue
            
            # 除外パターンチェック
            if any(pattern in col.lower() for pattern in self.exclude_patterns):
                continue
            
            # NaN率チェック（50%以上NaNは除外）
            if df[col].isnull().sum() / len(df) > 0.5:
                continue
            
            # 分散チェック（分散0は除外）
            if df[col].var() == 0:
                continue
            
            feature_columns.append(col)
        
        return feature_columns
    
    def _generate_combination_type(self, df: pd.DataFrame, 
                                  combination_type: str, 
                                  feature_columns: List[str]) -> Dict[str, np.ndarray]:
        """特定タイプの組み合わせ特徴量生成"""
        combinations_dict = {}
        
        if combination_type == 'ratio':
            combinations_dict.update(self._generate_ratio_features(df, feature_columns))
        elif combination_type == 'product':
            combinations_dict.update(self._generate_product_features(df, feature_columns))
        elif combination_type == 'difference':
            combinations_dict.update(self._generate_difference_features(df, feature_columns))
        elif combination_type == 'power':
            combinations_dict.update(self._generate_power_features(df, feature_columns))
        elif combination_type == 'trigonometric':
            combinations_dict.update(self._generate_trigonometric_features(df, feature_columns))
        elif combination_type == 'logarithmic':
            combinations_dict.update(self._generate_logarithmic_features(df, feature_columns))
        elif combination_type == 'exponential':
            combinations_dict.update(self._generate_exponential_features(df, feature_columns))
        elif combination_type == 'statistical':
            combinations_dict.update(self._generate_statistical_features(df, feature_columns))
        
        return combinations_dict
    
    def _generate_ratio_features(self, df: pd.DataFrame, 
                                feature_columns: List[str]) -> Dict[str, np.ndarray]:
        """比率特徴量生成"""
        ratio_features = {}
        
        # 2項組み合わせ
        for i, col1 in enumerate(feature_columns):
            for j, col2 in enumerate(feature_columns[i+1:], i+1):
                try:
                    # A/B
                    ratio_name = f"ratio_{col1}_div_{col2}"
                    ratio_values = df[col1] / (df[col2] + 1e-8)
                    if self._is_valid_feature(ratio_values):
                        ratio_features[ratio_name] = ratio_values.values
                    
                    # B/A
                    ratio_name = f"ratio_{col2}_div_{col1}"
                    ratio_values = df[col2] / (df[col1] + 1e-8)
                    if self._is_valid_feature(ratio_values):
                        ratio_features[ratio_name] = ratio_values.values
                    
                    # (A+B)/(A-B)
                    ratio_name = f"ratio_{col1}_plus_{col2}_div_diff"
                    ratio_values = (df[col1] + df[col2]) / (df[col1] - df[col2] + 1e-8)
                    if self._is_valid_feature(ratio_values):
                        ratio_features[ratio_name] = ratio_values.values
                        
                except Exception:
                    continue
        
        return ratio_features
    
    def _generate_product_features(self, df: pd.DataFrame, 
                                  feature_columns: List[str]) -> Dict[str, np.ndarray]:
        """積特徴量生成"""
        product_features = {}
        
        for i, col1 in enumerate(feature_columns):
            for j, col2 in enumerate(feature_columns[i+1:], i+1):
                try:
                    # A×B
                    product_name = f"product_{col1}_mult_{col2}"
                    product_values = df[col1] * df[col2]
                    if self._is_valid_feature(product_values):
                        product_features[product_name] = product_values.values
                    
                    # A×B×sign(A-B)
                    product_name = f"product_{col1}_mult_{col2}_sign"
                    product_values = df[col1] * df[col2] * np.sign(df[col1] - df[col2])
                    if self._is_valid_feature(product_values):
                        product_features[product_name] = product_values.values
                        
                except Exception:
                    continue
        
        return product_features
    
    def _generate_difference_features(self, df: pd.DataFrame, 
                                     feature_columns: List[str]) -> Dict[str, np.ndarray]:
        """差分特徴量生成"""
        difference_features = {}
        
        for i, col1 in enumerate(feature_columns):
            for j, col2 in enumerate(feature_columns[i+1:], i+1):
                try:
                    # A-B
                    diff_name = f"diff_{col1}_minus_{col2}"
                    diff_values = df[col1] - df[col2]
                    if self._is_valid_feature(diff_values):
                        difference_features[diff_name] = diff_values.values
                    
                    # |A-B|
                    diff_name = f"abs_diff_{col1}_{col2}"
                    diff_values = np.abs(df[col1] - df[col2])
                    if self._is_valid_feature(diff_values):
                        difference_features[diff_name] = diff_values.values
                    
                    # (A-B)²
                    diff_name = f"squared_diff_{col1}_{col2}"
                    diff_values = (df[col1] - df[col2])**2
                    if self._is_valid_feature(diff_values):
                        difference_features[diff_name] = diff_values.values
                        
                except Exception:
                    continue
        
        return difference_features
    
    def _generate_power_features(self, df: pd.DataFrame, 
                                feature_columns: List[str]) -> Dict[str, np.ndarray]:
        """べき乗特徴量生成"""
        power_features = {}
        
        for i, col1 in enumerate(feature_columns):
            for j, col2 in enumerate(feature_columns[i+1:], i+1):
                try:
                    # A^(B/10) - スケール調整
                    power_name = f"power_{col1}_pow_{col2}"
                    normalized_exp = df[col2] / (10 * np.std(df[col2]) + 1e-8)
                    normalized_exp = np.clip(normalized_exp, -5, 5)  # 安全範囲
                    power_values = np.abs(df[col1] + 1e-8) ** normalized_exp
                    if self._is_valid_feature(power_values):
                        power_features[power_name] = power_values.values
                    
                    # sqrt(|A×B|)
                    power_name = f"sqrt_product_{col1}_{col2}"
                    power_values = np.sqrt(np.abs(df[col1] * df[col2]) + 1e-8)
                    if self._is_valid_feature(power_values):
                        power_features[power_name] = power_values.values
                        
                except Exception:
                    continue
        
        return power_features
    
    def _generate_trigonometric_features(self, df: pd.DataFrame, 
                                        feature_columns: List[str]) -> Dict[str, np.ndarray]:
        """三角関数特徴量生成"""
        trig_features = {}
        
        for i, col1 in enumerate(feature_columns):
            for j, col2 in enumerate(feature_columns[i+1:], i+1):
                try:
                    # sin(A×B)
                    trig_name = f"sin_product_{col1}_{col2}"
                    normalized_input = (df[col1] * df[col2]) / (np.std(df[col1] * df[col2]) + 1e-8)
                    trig_values = np.sin(normalized_input)
                    if self._is_valid_feature(trig_values):
                        trig_features[trig_name] = trig_values.values
                    
                    # cos(A+B)
                    trig_name = f"cos_sum_{col1}_{col2}"
                    normalized_input = (df[col1] + df[col2]) / (np.std(df[col1] + df[col2]) + 1e-8)
                    trig_values = np.cos(normalized_input)
                    if self._is_valid_feature(trig_values):
                        trig_features[trig_name] = trig_values.values
                    
                    # tanh(A-B)
                    trig_name = f"tanh_diff_{col1}_{col2}"
                    normalized_input = (df[col1] - df[col2]) / (np.std(df[col1] - df[col2]) + 1e-8)
                    trig_values = np.tanh(normalized_input)
                    if self._is_valid_feature(trig_values):
                        trig_features[trig_name] = trig_values.values
                        
                except Exception:
                    continue
        
        return trig_features
    
    def _generate_logarithmic_features(self, df: pd.DataFrame, 
                                      feature_columns: List[str]) -> Dict[str, np.ndarray]:
        """対数特徴量生成"""
        log_features = {}
        
        for i, col1 in enumerate(feature_columns):
            for j, col2 in enumerate(feature_columns[i+1:], i+1):
                try:
                    # log(|A|+|B|+1)
                    log_name = f"log_sum_abs_{col1}_{col2}"
                    log_values = np.log(np.abs(df[col1]) + np.abs(df[col2]) + 1)
                    if self._is_valid_feature(log_values):
                        log_features[log_name] = log_values.values
                    
                    # log(A²+B²+1)
                    log_name = f"log_euclidean_{col1}_{col2}"
                    log_values = np.log(df[col1]**2 + df[col2]**2 + 1)
                    if self._is_valid_feature(log_values):
                        log_features[log_name] = log_values.values
                    
                    # log(|A/B|+1)
                    log_name = f"log_ratio_{col1}_{col2}"
                    log_values = np.log(np.abs(df[col1] / (df[col2] + 1e-8)) + 1)
                    if self._is_valid_feature(log_values):
                        log_features[log_name] = log_values.values
                        
                except Exception:
                    continue
        
        return log_features
    
    def _generate_exponential_features(self, df: pd.DataFrame, 
                                      feature_columns: List[str]) -> Dict[str, np.ndarray]:
        """指数特徴量生成"""
        exp_features = {}
        
        for i, col1 in enumerate(feature_columns):
            for j, col2 in enumerate(feature_columns[i+1:], i+1):
                try:
                    # sigmoid(A×B)
                    exp_name = f"sigmoid_product_{col1}_{col2}"
                    normalized_input = (df[col1] * df[col2]) / (np.std(df[col1] * df[col2]) + 1e-8)
                    normalized_input = np.clip(normalized_input, -10, 10)  # 安全範囲
                    exp_values = expit(normalized_input)  # sigmoid
                    if self._is_valid_feature(exp_values):
                        exp_features[exp_name] = exp_values.values
                    
                    # exp(-|A-B|/σ)
                    exp_name = f"gaussian_diff_{col1}_{col2}"
                    diff = np.abs(df[col1] - df[col2])
                    sigma = np.std(diff) + 1e-8
                    exp_values = np.exp(-diff / sigma)
                    if self._is_valid_feature(exp_values):
                        exp_features[exp_name] = exp_values.values
                        
                except Exception:
                    continue
        
        return exp_features
    
    def _generate_statistical_features(self, df: pd.DataFrame, 
                                      feature_columns: List[str]) -> Dict[str, np.ndarray]:
        """統計的組み合わせ特徴量生成"""
        stat_features = {}
        
        for i, col1 in enumerate(feature_columns):
            for j, col2 in enumerate(feature_columns[i+1:], i+1):
                try:
                    # 相関係数 (ローリング)
                    stat_name = f"rolling_corr_{col1}_{col2}"
                    rolling_corr = df[[col1, col2]].rolling(window=20, min_periods=10).corr().iloc[0::2, -1].values
                    if self._is_valid_feature(pd.Series(rolling_corr)):
                        stat_features[stat_name] = rolling_corr
                    
                    # 順位相関
                    stat_name = f"rank_corr_{col1}_{col2}"
                    rank1 = df[col1].rank()
                    rank2 = df[col2].rank()
                    rank_corr_values = rank1 * rank2
                    if self._is_valid_feature(rank_corr_values):
                        stat_features[stat_name] = rank_corr_values.values
                    
                    # 最大・最小
                    stat_name = f"max_{col1}_{col2}"
                    max_values = np.maximum(df[col1], df[col2])
                    if self._is_valid_feature(max_values):
                        stat_features[stat_name] = max_values.values
                    
                    stat_name = f"min_{col1}_{col2}"
                    min_values = np.minimum(df[col1], df[col2])
                    if self._is_valid_feature(min_values):
                        stat_features[stat_name] = min_values.values
                        
                except Exception:
                    continue
        
        return stat_features
    
    def _is_valid_feature(self, values: pd.Series) -> bool:
        """特徴量の有効性チェック"""
        try:
            # NaN/inf チェック
            if pd.isna(values).all() or np.isinf(values).any():
                return False
            
            # 分散チェック
            if values.var() == 0:
                return False
            
            # 異常値チェック（99.9%ile以内）
            if np.abs(values).quantile(0.999) > 1e10:
                return False
            
            return True
        except:
            return False
    
    def _remove_highly_correlated_features(self, df: pd.DataFrame, 
                                          original_columns: List[str]) -> pd.DataFrame:
        """高相関特徴量除去"""
        try:
            # 新規特徴量のみ対象
            new_columns = [col for col in df.columns if col not in original_columns]
            
            if len(new_columns) < 2:
                return df
            
            # 相関行列計算
            correlation_matrix = df[new_columns].corr().abs()
            
            # 高相関ペア特定
            upper_triangle = correlation_matrix.where(
                np.triu(np.ones(correlation_matrix.shape), k=1).astype(bool)
            )
            
            # 除去対象列
            to_drop = [column for column in upper_triangle.columns 
                      if any(upper_triangle[column] > self.correlation_threshold)]
            
            logger.info(f"高相関特徴量除去: {len(to_drop)}個")
            
            return df.drop(columns=to_drop)
            
        except Exception as e:
            logger.warning(f"相関除去でエラー: {e}")
            return df


def process_combination_features(df: pd.DataFrame, 
                                chunk_id: str = "",
                                feature_types: List[str] = None,
                                max_combinations: int = 50000) -> pd.DataFrame:
    """
    非線形組み合わせ特徴量処理（外部呼び出し用）
    
    Args:
        df: 入力DataFrame（任意の特徴量）
        chunk_id: チャンクID
        feature_types: 生成する組み合わせ種類
        max_combinations: 最大組み合わせ数
        
    Returns:
        組み合わせ特徴量が追加されたDataFrame
    """
    try:
        engine = NonlinearCombinationEngine(max_combinations=max_combinations)
        result = engine.generate_all_combinations(df, feature_types, chunk_id)
        return result
    except Exception as e:
        logger.error(f"組み合わせ特徴量処理エラー - Chunk {chunk_id}: {e}")
        return df

def main_with_file_separation():
    """
    分離方式での組み合わせ特徴量生成 (新命名規則対応)
    """
    # パス設定 (新命名規則適用)
    mfdfa_path = r"C:\project_forge\data\temp_chunks\parquet\01_mfdfa_45features.parquet"
    independent_path = r"C:\project_forge\data\temp_chunks\parquet\02_independent_296features.parquet"
    output_path = r"C:\project_forge\data\temp_chunks\parquet\03_combination_xxxfeatures.parquet"
    
    logger.info("=== 分離方式組み合わせ特徴量生成開始 ===")
    
    try:
        # 既存特徴量読み込み
        logger.info("既存特徴量読み込み...")
        mfdfa_df = pd.read_parquet(mfdfa_path)
        independent_df = pd.read_parquet(independent_path)
        
        # データ統合 (組み合わせ生成用)
        combined_df = pd.concat([mfdfa_df, independent_df], axis=1)
        logger.info(f"統合データ: {combined_df.shape[0]:,} 行 × {combined_df.shape[1]} 特徴量")
        
        # 組み合わせ特徴量生成
        logger.info("組み合わせ特徴量生成開始...")
        result_df = process_combination_features(
            df=combined_df,
            chunk_id="MAIN",
            max_combinations=100000  # 大きめに設定
        )
        
        # 元の特徴量を除去して組み合わせ特徴量のみ抽出
        original_columns = list(mfdfa_df.columns) + list(independent_df.columns)
        combination_only = result_df.drop(columns=original_columns)
        
        logger.info(f"生成された組み合わせ特徴量: {combination_only.shape[1]:,} 個")
        
        # ★ 変更点: Parquet形式のみで保存 ★
        logger.info(f"PARQUET形式で保存開始: {output_path}")
        combination_only.to_parquet(output_path, compression='snappy')
        logger.info(f"PARQUET形式で保存完了")
        
        # 実際の特徴量数でファイル名を更新
        actual_count = combination_only.shape[1]
        new_output_path = output_path.replace('xxxfeatures', f'{actual_count}features')
        if output_path != new_output_path:
            Path(output_path).rename(new_output_path)
            logger.info(f"ファイル名更新: {Path(new_output_path).name}")
        
        # 統計情報
        file_size = Path(new_output_path).stat().st_size / 1024 / 1024
        logger.info(f"保存完了: {combination_only.shape[0]:,} 行 × {actual_count:,} 特徴量")
        logger.info(f"ファイルサイズ: {file_size:.2f} MB")
        
        logger.info("🎉 Parquet形式での保存が正常に完了しました！ 🎉")
        
    except Exception as e:
        logger.error(f"エラー発生: {e}")
        raise

if __name__ == "__main__":
    # 分離方式での実行
    main_with_file_separation()
    
    # 以下はテスト用（必要に応じてコメントアウト）
    # import time
    
    # ログ設定
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    
    # サンプルデータ生成（independent_features.py出力をシミュレート）
    np.random.seed(42)
    n_samples = 1000
    
    # 基本特徴量（independent_features.py風）
    test_data = pd.DataFrame({
        'rsi_14': np.random.uniform(20, 80, n_samples),
        'macd_12_26': np.random.normal(0, 1, n_samples),
        'bb_position_20': np.random.uniform(0, 1, n_samples),
        'atr_14': np.random.uniform(0.5, 2.0, n_samples),
        'volume_ma_10': np.random.lognormal(8, 1, n_samples),
        'sma_20': np.random.uniform(95, 105, n_samples),
        'ema_50': np.random.uniform(95, 105, n_samples),
        'volatility_20': np.random.uniform(0.1, 0.5, n_samples),
        'momentum_10': np.random.normal(0, 2, n_samples),
        'trend_strength': np.random.uniform(-1, 1, n_samples),
        # 除外対象（テスト用）
        'timestamp': pd.date_range('2020-01-01', periods=n_samples, freq='H'),
        'constant_feature': np.ones(n_samples)  # 分散0
    })
    
    print("=== 非線形組み合わせ特徴量エンジンテスト開始 ===")
    print(f"入力データ shape: {test_data.shape}")
    
    # テスト実行
    engine = NonlinearCombinationEngine(n_processes=4, max_combinations=1000)
    
    start_time = time.time()
    result = engine.generate_all_combinations(test_data, chunk_id="TEST")
    end_time = time.time()
    
    processing_time = end_time - start_time
    
    print(f"\n=== 処理結果 ===")
    print(f"処理時間: {processing_time:.2f}秒")
    print(f"入力特徴量数: {test_data.shape[1]}")
    print(f"出力特徴量数: {result.shape[1]}")
    print(f"新規生成特徴量数: {result.shape[1] - test_data.shape[1]}")
    print(f"データサイズ: {result.shape}")
    
    # 生成された特徴量のサンプル表示
    new_columns = [col for col in result.columns if col not in test_data.columns]
    print(f"\n=== 生成特徴量サンプル（最初の10個） ===")
    for i, col in enumerate(new_columns[:10]):
        values = result[col].dropna()
        if len(values) > 0:
            print(f"{col:40s}: mean={values.mean():.4f}, std={values.std():.4f}")
    
    # カテゴリ別統計
    categories = {
        'ratio': [col for col in new_columns if 'ratio_' in col],
        'product': [col for col in new_columns if 'product_' in col],
        'difference': [col for col in new_columns if 'diff_' in col],
        'power': [col for col in new_columns if 'power_' in col or 'sqrt_' in col],
        'trigonometric': [col for col in new_columns if any(trig in col for trig in ['sin_', 'cos_', 'tanh_'])],
        'logarithmic': [col for col in new_columns if 'log_' in col],
        'exponential': [col for col in new_columns if 'sigmoid_' in col or 'gaussian_' in col],
        'statistical': [col for col in new_columns if any(stat in col for stat in ['corr_', 'rank_', 'max_', 'min_'])]
    }
    
    print(f"\n=== カテゴリ別特徴量統計 ===")
    total_generated = 0
    for category, features in categories.items():
        count = len(features)
        if count > 0:
            print(f"{category:15s}: {count:4d} 特徴量")
            total_generated += count
    
    print(f"{'='*40}")
    print(f"{'総生成特徴量数':15s}: {total_generated:4d} 特徴量")
    
    # メモリ使用量
    memory_usage = result.memory_usage(deep=True).sum() / 1024**2
    print(f"\nメモリ使用量: {memory_usage:.1f} MB")
    
    print(f"\n=== テスト完了 ===")
    print("🎉 非線形組み合わせエンジンが正常に動作しています！")
    print(f"⚡ {result.shape[1] - test_data.shape[1]} 個の組み合わせ特徴量を {processing_time:.2f}秒で生成")
    print("🔧 任意の特徴量DataFrameに対応可能")