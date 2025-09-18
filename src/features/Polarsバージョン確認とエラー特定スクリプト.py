#!/usr/bin/env python3
"""
Polarsバージョン確認とエラー特定スクリプト
"""

import polars as pl
import numpy as np
import pandas as pd
from typing import Dict, Any

def debug_polars_environment():
    """現在のPolars環境をデバッグ"""
    print("=" * 50)
    print("Polars環境デバッグ")
    print("=" * 50)
    
    # バージョン情報
    print(f"Polars version: {pl.__version__}")
    print(f"Available methods on pl.col():")
    
    # テストデータ作成
    test_data = np.random.randn(100)
    df = pl.DataFrame({"price": test_data})
    
    print("\n1. 基本的なPolars操作テスト:")
    try:
        result = df.select(pl.col("price").mean())
        print(f"   ✅ mean(): {result.item()}")
    except Exception as e:
        print(f"   ❌ mean() エラー: {e}")
    
    print("\n2. cumsum()テスト:")
    try:
        result = df.with_columns(pl.col("price").cumsum().alias("cumsum"))
        print(f"   ✅ cumsum(): 正常")
    except Exception as e:
        print(f"   ❌ cumsum() エラー: {e}")
    
    print("\n3. is_finite()テスト:")
    try:
        result = df.select(pl.col("price").is_finite().sum())
        print(f"   ✅ is_finite(): {result.item()}")
    except Exception as e:
        print(f"   ❌ is_finite() エラー: {e}")
    
    print("\n4. is_infinite()テスト:")
    try:
        result = df.select(pl.col("price").is_infinite().sum())
        print(f"   ✅ is_infinite(): {result.item()}")
    except Exception as e:
        print(f"   ❌ is_infinite() エラー: {e}")
    
    print("\n5. Expr vs Series の確認:")
    expr = pl.col("price").mean()
    print(f"   pl.col('price').mean() の型: {type(expr)}")
    
    series_result = df.select(pl.col("price").mean())
    print(f"   df.select(pl.col('price').mean()) の型: {type(series_result)}")
    
    return df

def test_mfdfa_fix_v1(df):
    """MFDFA修正版1: 完全にNumPyベース"""
    print("\n" + "=" * 50)
    print("MFDFA修正版1テスト (NumPyベース)")
    print("=" * 50)
    
    try:
        # Polarsから直接NumPy配列を取得
        price_data = df["price"].to_numpy()
        
        # NumPyで全て処理
        price_mean = np.mean(price_data)
        price_centered = price_data - price_mean
        profile = np.cumsum(price_centered)
        
        print("✅ MFDFA修正版1: 成功")
        return profile
    except Exception as e:
        print(f"❌ MFDFA修正版1 エラー: {e}")
        return None

def test_mfdfa_fix_v2(df):
    """MFDFA修正版2: Polars lazy evaluation"""
    print("\n" + "=" * 50)
    print("MFDFA修正版2テスト (Polars lazy)")
    print("=" * 50)
    
    try:
        # Lazy evaluation を使用
        result = (
            df.lazy()
            .with_columns([
                (pl.col("price") - pl.col("price").mean()).alias("price_centered")
            ])
            .with_columns([
                pl.col("price_centered").cumsum().alias("profile")
            ])
            .collect()
        )
        
        profile = result["profile"].to_numpy()
        print("✅ MFDFA修正版2: 成功")
        return profile
    except Exception as e:
        print(f"❌ MFDFA修正版2 エラー: {e}")
        return None

def test_quality_report_fix(df):
    """品質レポート修正版テスト"""
    print("\n" + "=" * 50)
    print("品質レポート修正版テスト")
    print("=" * 50)
    
    # テスト用特徴量データ
    features = {
        "feature1": np.random.randn(100),
        "feature2": np.array([np.nan] * 10 + list(np.random.randn(90))),
        "feature3": np.array([np.inf, -np.inf] + list(np.random.randn(98)))
    }
    
    try:
        df_features = pl.DataFrame(features)
        
        # 各メトリクスを個別に計算
        results = {}
        
        for col in df_features.columns:
            # null count
            null_count = df_features.select(pl.col(col).null_count()).item()
            
            # finite count  
            finite_count = df_features.select(pl.col(col).is_finite().sum()).item()
            
            # infinite count
            inf_count = df_features.select(pl.col(col).is_infinite().sum()).item()
            
            results[col] = {
                'null_count': null_count,
                'finite_count': finite_count,
                'inf_count': inf_count
            }
        
        print("✅ 品質レポート修正版: 成功")
        print(f"   結果: {results}")
        return results
        
    except Exception as e:
        print(f"❌ 品質レポート修正版 エラー: {e}")
        return None

def generate_final_fixes():
    """最終修正版のコード生成"""
    print("\n" + "=" * 80)
    print("最終修正版コード")
    print("=" * 80)
    
    mfdfa_code = '''
def calculate_mfdfa_features_polars(self, data: np.ndarray) -> Dict[str, np.ndarray]:
    """MFDFA特徴量計算（NumPy主体+Polars補助版）"""
    features = {}
    
    # データをNumPy配列として直接処理
    if isinstance(data, np.ndarray):
        price_data = data.flatten() if data.ndim > 1 else data
    else:
        # Polarsから変換
        df = self._ensure_polars_df(data, 'price')
        price_data = df["price"].to_numpy()
    
    n = len(price_data)
    
    # MFDFA用パラメータ
    q_range = self.params['mfdfa_q_range']
    scales = self.params['mfdfa_scales']
    scales = scales[scales < n//4]
    
    window_size = 200
    if n < window_size:
        logger.warning(f"データ長{n}がMFDFA計算に不十分です")
        return features
    
    try:
        # NumPyで前処理（Polarsの問題を回避）
        price_mean = np.mean(price_data)
        price_centered = price_data - price_mean
        profile = np.cumsum(price_centered)
        
        # ローリングウィンドウ用のインデックス
        indices = np.arange(window_size-1, n)
        
        # MFDFA計算
        mfdfa_results = self._numba_safe_calculation(
            self._calculate_mfdfa_vectorized,
            profile,
            indices,
            window_size,
            q_range,
            scales
        )
        
        if mfdfa_results is not None and len(mfdfa_results.shape) == 2:
            result_df = pd.DataFrame(mfdfa_results)
            
            # 特徴量作成
            for j, q in enumerate(q_range):
                feature_name = f'mfdfa_hurst_q{q}'
                if j < result_df.shape[1]:
                     features[feature_name] = np.pad(
                        result_df.iloc[:, j].values, 
                        (window_size-1, 0), 
                        mode='constant', 
                        constant_values=np.nan
                    )
            
            # 統合特徴量
            if result_df.shape[1] >= len(q_range) + 3:
                features['mfdfa_multifractal_width'] = np.pad(
                    result_df.iloc[:, -3].values, (window_size-1, 0), 
                    mode='constant', constant_values=np.nan
                )
                features['mfdfa_asymmetry'] = np.pad(
                    result_df.iloc[:, -2].values, (window_size-1, 0), 
                    mode='constant', constant_values=np.nan
                )
                features['mfdfa_complexity'] = np.pad(
                    result_df.iloc[:, -1].values, (window_size-1, 0), 
                    mode='constant', constant_values=np.nan
                )
                
    except Exception as e:
        logger.error(f"MFDFA計算エラー: {e}")
        # フォールバック値
        for q in q_range:
            features[f'mfdfa_hurst_q{q}'] = np.zeros(n)
        features['mfdfa_multifractal_width'] = np.zeros(n)  
        features['mfdfa_asymmetry'] = np.zeros(n)
        features['mfdfa_complexity'] = np.zeros(n)
    
    return features
'''
    
    quality_code = '''
def generate_quality_report_polars(self, features: Dict[str, np.ndarray]) -> Dict[str, Any]:
    """品質レポート生成（NumPy主体+Polars補助版）"""
    
    if not features:
        return {'status': 'no_features', 'total_features': 0}
    
    try:
        # NumPyで直接計算（Polarsの複雑な操作を回避）
        total_features = len(features)
        total_points = len(next(iter(features.values())))
        
        # 各メトリクスをNumPyで計算
        null_counts = {}
        finite_counts = {}
        inf_counts = {}
        
        for name, values in features.items():
            null_counts[name] = np.sum(np.isnan(values))
            finite_counts[name] = np.sum(np.isfinite(values))
            inf_counts[name] = np.sum(np.isinf(values))
        
        # 比率計算
        null_ratios = np.array(list(null_counts.values())) / total_points
        finite_ratios = np.array(list(finite_counts.values())) / total_points
        inf_ratios = np.array(list(inf_counts.values())) / total_points
        
        # 統計値
        null_ratio_avg = float(np.mean(null_ratios))
        finite_ratio_avg = float(np.mean(finite_ratios))
        inf_ratio_avg = float(np.mean(inf_ratios))
        overall_quality = finite_ratio_avg
        
        # カテゴリ別カウント
        high_quality_count = int(np.sum(finite_ratios > 0.95))
        medium_quality_count = int(np.sum((finite_ratios >= 0.8) & (finite_ratios <= 0.95)))
        low_quality_count = int(np.sum(finite_ratios < 0.8))
        
        quality_report = {
            'status': 'completed',
            'total_features': total_features,
            'data_points': total_points,
            'overall_quality_score': overall_quality,
            'null_ratio_avg': null_ratio_avg,
            'inf_ratio_avg': inf_ratio_avg,
            'finite_ratio_avg': finite_ratio_avg,
            'high_quality_features': high_quality_count,
            'medium_quality_features': medium_quality_count,
            'low_quality_features': low_quality_count,
            'calculation_stats': getattr(self, 'calculation_stats', {}),
            'polars_optimization_ratio': (
                getattr(self, 'calculation_stats', {}).get('polars_calculations', 0) / 
                max(1, getattr(self, 'calculation_stats', {}).get('total_calculations', 1))
            )
        }
        
        # 警告・推奨事項
        warnings = []
        recommendations = []
        
        if overall_quality < 0.9:
            warnings.append(f"全体品質スコアが低下: {overall_quality:.3f}")
            recommendations.append("数値安定化処理の強化を検討")
        
        if inf_ratio_avg > 0.01:
            warnings.append(f"無限値の比率が高い: {inf_ratio_avg:.3f}")
            recommendations.append("数値計算の安定性を改善")
        
        if null_ratio_avg > 0.05:
            warnings.append(f"欠損値の比率が高い: {null_ratio_avg:.3f}")
            recommendations.append("欠損値補完アルゴリズムの改善")
        
        quality_report['warnings'] = warnings
        quality_report['recommendations'] = recommendations
        
        return quality_report
        
    except Exception as e:
        logger.error(f"品質レポート生成エラー: {e}")
        return {
            'status': 'error',
            'error_message': str(e),
            'total_features': len(features)
        }
'''
    
    print("MFDFA関数:")
    print(mfdfa_code)
    print("\n品質レポート関数:")
    print(quality_code)

if __name__ == "__main__":
    # 環境デバッグ実行
    df = debug_polars_environment()
    test_mfdfa_fix_v1(df)
    test_mfdfa_fix_v2(df)
    test_quality_report_fix(df)
    generate_final_fixes()