# tier1_garch_optimized.py - ARIMA-GARCH特徴量生成（最適化版）

import pandas as pd
import numpy as np
import logging
from pathlib import Path
import warnings
import time
import multiprocessing as mp
from functools import partial
import joblib
from concurrent.futures import ProcessPoolExecutor, as_completed
import gc

# GARCH用ライブラリ
import pmdarima as pm
from arch import arch_model

warnings.filterwarnings('ignore')

# ロガー設定
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class OptimizedGARCHFeatureEngine:
    """
    最適化版ARIMA-GARCH特徴量生成エンジン
    MFDFAの並列化成功を受けた部分並列化実装
    """
    
    def __init__(self, n_jobs=8):
        """
        Parameters:
        -----------
        n_jobs : int
            並列処理数（i7-8700Kでは8が推奨）
        """
        self.n_jobs = min(n_jobs, mp.cpu_count() - 2)  # システム安定性確保
        logger.info(f"並列処理数設定: {self.n_jobs}")
    
    def add_arima_garch_residuals(self, df: pd.DataFrame, 
                                 estimation_window: int = 1000, 
                                 reestimation_frequency: int = 50,
                                 checkpoint_frequency: int = 50000) -> pd.DataFrame:
        """
        最適化版ARIMA-GARCH特徴量生成
        
        Parameters:
        -----------
        estimation_window : int
            モデル推定に使用するデータ数
        reestimation_frequency : int  
            モデル再推定の間隔（実用性重視：50）
        checkpoint_frequency : int
            途中保存の間隔（5万ステップごと）
        """
        logger.info(f"=== 最適化版ARIMA-GARCH特徴量生成開始 ===")
        logger.info(f"推定ウィンドウ: {estimation_window}")
        logger.info(f"再推定間隔: {reestimation_frequency}")
        logger.info(f"途中保存間隔: {checkpoint_frequency}")
        logger.info(f"並列処理数: {self.n_jobs}")
        
        returns = df['close'].pct_change().dropna()
        logger.info(f"リターンデータ数: {len(returns):,}")
        
        # データ品質チェック
        self._validate_data_quality(returns)
        
        # 結果格納用の列を初期化
        df['garch_vol_forecast'] = np.nan
        df['standardized_residual'] = np.nan
        
        # 処理統計
        total_predictions = len(returns) - estimation_window
        total_reestimations = total_predictions // reestimation_frequency + 1
        
        logger.info(f"処理統計:")
        logger.info(f"  総予測ポイント数: {total_predictions:,}")
        logger.info(f"  予想再推定回数: {total_reestimations:,}")
        
        if total_predictions <= 0:
            logger.error("推定ウィンドウがデータサイズより大きいため処理不可能")
            return df
        
        # 途中保存用ディレクトリ
        checkpoint_dir = Path("./temp_garch_checkpoints")
        checkpoint_dir.mkdir(exist_ok=True)
        
        # メイン処理実行
        df_result = self._execute_garch_pipeline(
            df, returns, estimation_window, reestimation_frequency, 
            checkpoint_frequency, checkpoint_dir
        )
        
        # クリーンアップ
        self._cleanup_checkpoints(checkpoint_dir)
        
        return df_result
    
    def _validate_data_quality(self, returns):
        """データ品質チェック"""
        nan_count = returns.isna().sum()
        inf_count = np.isinf(returns).sum()
        zero_var = returns.var() == 0
        
        logger.info(f"データ品質チェック:")
        logger.info(f"  NaN: {nan_count}, 無限値: {inf_count}, 分散ゼロ: {zero_var}")
        
        if nan_count > 0 or inf_count > 0 or zero_var:
            logger.warning("データ品質に問題があります。結果の信頼性が低下する可能性があります。")
    
    def _execute_garch_pipeline(self, df, returns, estimation_window, 
                               reestimation_frequency, checkpoint_frequency, checkpoint_dir):
        """メインのGARCH処理パイプライン"""
        
        # 再推定ポイントを事前計算
        reestimation_points = list(range(
            estimation_window, 
            len(returns), 
            reestimation_frequency
        ))
        
        logger.info(f"再推定ポイント数: {len(reestimation_points)}")
        
        # 並列再推定実行
        models_cache = self._parallel_model_estimation(
            returns, reestimation_points, estimation_window
        )
        
        # 逐次予測実行（時系列依存性のため）
        df_result = self._sequential_prediction(
            df, returns, estimation_window, reestimation_frequency,
            models_cache, checkpoint_frequency, checkpoint_dir
        )
        
        return df_result
    
    def _parallel_model_estimation(self, returns, reestimation_points, estimation_window):
        """並列化された模型推定"""
        logger.info("=== 並列模型推定開始 ===")
        
        # 推定タスクを準備
        estimation_tasks = []
        for point in reestimation_points:
            estimation_data = returns.iloc[point - estimation_window : point]
            estimation_tasks.append((point, estimation_data))
        
        # 並列実行
        models_cache = {}
        
        with ProcessPoolExecutor(max_workers=self.n_jobs) as executor:
            # タスクを並列実行
            future_to_point = {
                executor.submit(_estimate_arima_garch_model, task[1]): task[0] 
                for task in estimation_tasks
            }
            
            completed = 0
            for future in as_completed(future_to_point):
                point = future_to_point[future]
                try:
                    model_result = future.result()
                    models_cache[point] = model_result
                    completed += 1
                    
                    if completed % 10 == 0:
                        logger.info(f"  模型推定進捗: {completed}/{len(estimation_tasks)}")
                        
                except Exception as e:
                    logger.warning(f"  ポイント {point} で推定エラー: {e}")
                    models_cache[point] = None
        
        logger.info(f"=== 並列模型推定完了: {len(models_cache)}/{len(reestimation_points)} 成功 ===")
        return models_cache
    
    def _sequential_prediction(self, df, returns, estimation_window, reestimation_frequency,
                              models_cache, checkpoint_frequency, checkpoint_dir):
        """逐次予測実行"""
        logger.info("=== 逐次予測開始 ===")
        
        current_model = None
        garch_params = {'omega': 0.0, 'alpha[1]': 0.0, 'beta[1]': 0.0}
        last_variance = returns.var()
        
        successful_predictions = 0
        failed_predictions = 0
        start_time = time.time()
        
        for i in range(estimation_window, len(returns)):
            # モデル更新チェック
            if (i - estimation_window) % reestimation_frequency == 0:
                if i in models_cache and models_cache[i] is not None:
                    current_model, garch_params, last_variance = models_cache[i]
                    logger.info(f"  ポイント {i}: モデル更新完了")
            
            # 予測実行
            if current_model is not None:
                try:
                    vol_forecast, std_resid, new_variance = self._predict_single_step(
                        current_model, garch_params, returns, i, last_variance
                    )
                    
                    # 結果保存
                    df.loc[returns.index[i], 'garch_vol_forecast'] = vol_forecast
                    df.loc[returns.index[i], 'standardized_residual'] = std_resid
                    
                    last_variance = new_variance
                    successful_predictions += 1
                    
                except Exception as e:
                    failed_predictions += 1
                    continue
            
            # 途中保存
            if (i - estimation_window) % checkpoint_frequency == 0:
                self._save_checkpoint(df, checkpoint_dir, i)
            
            # 進捗報告
            if (i - estimation_window + 1) % 5000 == 0:
                self._report_progress(i, estimation_window, len(returns), 
                                    successful_predictions, failed_predictions, start_time)
        
        # 最終統計
        self._report_final_statistics(successful_predictions, failed_predictions, 
                                    time.time() - start_time, df)
        
        return df
    
    def _predict_single_step(self, arima_model, garch_params, returns, i, last_variance):
        """単一ステップ予測"""
        # ARIMAによるリターン予測
        recent_data = returns.iloc[max(0, i-min(100, 500)):i]
        if len(recent_data) >= 10:
            try:
                arima_forecast = arima_model.predict(n_periods=1)[0]
            except:
                arima_forecast = recent_data.tail(10).mean()
        else:
            arima_forecast = 0
        
        # 実際のリターンと残差
        actual_return = returns.iloc[i]
        residual = actual_return - arima_forecast
        
        # GARCH(1,1)の数学的予測式: σ²(t+1) = ω + α*ε²(t) + β*σ²(t)
        variance_forecast = (
            garch_params['omega'] + 
            garch_params['alpha[1]'] * (residual ** 2) + 
            garch_params['beta[1]'] * last_variance
        )
        
        vol_forecast = np.sqrt(max(variance_forecast, 1e-8))
        std_resid = residual / vol_forecast if vol_forecast > 1e-8 else np.nan
        
        return vol_forecast, std_resid, variance_forecast
    
    def _save_checkpoint(self, df, checkpoint_dir, i):
        """途中保存"""
        checkpoint_path = checkpoint_dir / f"garch_checkpoint_{i}.parquet"
        try:
            df.to_parquet(checkpoint_path, compression='snappy')
            logger.info(f"  途中保存実行: ステップ {i}")
        except Exception as e:
            logger.warning(f"途中保存エラー: {e}")
    
    def _report_progress(self, i, estimation_window, total_length, 
                        successful_predictions, failed_predictions, start_time):
        """進捗報告"""
        current_step = i - estimation_window + 1
        total_steps = total_length - estimation_window
        progress = current_step / total_steps * 100
        
        elapsed_time = time.time() - start_time
        if current_step > 0:
            estimated_total_time = elapsed_time / current_step * total_steps
            remaining_time = estimated_total_time - elapsed_time
            success_rate = successful_predictions / current_step * 100
            
            logger.info(f"  予測進捗: {progress:.1f}% ({current_step:,} / {total_steps:,})")
            logger.info(f"  成功率: {success_rate:.1f}%, 残り時間: {remaining_time/3600:.1f}時間")
    
    def _report_final_statistics(self, successful_predictions, failed_predictions, 
                               total_time, df):
        """最終統計報告"""
        logger.info(f"=== 最終処理統計 ===")
        logger.info(f"総処理時間: {total_time/3600:.2f}時間")
        logger.info(f"成功した予測: {successful_predictions} / {successful_predictions + failed_predictions} ({successful_predictions/(successful_predictions + failed_predictions)*100:.1f}%)")
        
        valid_vol = df['garch_vol_forecast'].notna().sum()
        valid_resid = df['standardized_residual'].notna().sum()
        
        logger.info(f"有効なボラティリティ予測: {valid_vol:,}")
        logger.info(f"有効な標準化残差: {valid_resid:,}")
    
    def _cleanup_checkpoints(self, checkpoint_dir):
        """途中保存ファイルのクリーンアップ"""
        try:
            for checkpoint_file in checkpoint_dir.glob("*.parquet"):
                checkpoint_file.unlink()
            checkpoint_dir.rmdir()
            logger.info("途中保存ファイルをクリーンアップしました")
        except:
            logger.warning("途中保存ファイルのクリーンアップに失敗")

# 並列処理用の独立関数
def _estimate_arima_garch_model(estimation_data):
    """
    単一のARIMA-GARCHモデル推定（並列処理用）
    """
    try:
        # ARIMA推定
        arima_model = pm.auto_arima(
            estimation_data, 
            max_p=3, max_q=3, max_d=2, 
            seasonal=False,
            suppress_warnings=True, 
            error_action='ignore', 
            stepwise=True, 
            n_jobs=1,  # 個別プロセス内では並列化しない
            information_criterion='aic',
            maxiter=50
        )
        
        # GARCH推定
        arima_residuals = arima_model.resid()
        
        # 残差品質チェック
        if len(arima_residuals) < 20 or arima_residuals.std() < 1e-8:
            raise ValueError("ARIMAの残差が不適切")
        
        garch_fit = arch_model(
            arima_residuals, 
            vol='Garch', 
            p=1, q=1,
            rescale=False
        ).fit(disp='off', show_warning=False, options={'maxiter': 100})
        
        # パラメータ抽出
        garch_params = dict(garch_fit.params)
        last_variance = garch_fit.conditional_volatility.iloc[-1]**2
        
        # メモリクリーンアップ
        del garch_fit
        gc.collect()
        
        return arima_model, garch_params, last_variance
        
    except Exception as e:
        return None

def main():
    """メイン実行関数"""
    
    # === パス設定 ===
    base_dir = Path(r"C:\project_forge\data\temp_chunks\parquet")
    
    # MFDFAが完了したファイルを自動検出
    mfdfa_files = list(base_dir.glob("*mfdfa*.parquet"))
    
    if not mfdfa_files:
        logger.error("MFDFAファイルが見つかりません")
        logger.error("tier1_mfdfa.pyの実行完了を待ってください")
        return
    
    # 最新のMFDFAファイルを使用
    input_path = max(mfdfa_files, key=lambda x: x.stat().st_mtime)
    output_path = base_dir / "04_with_tier1_complete_garch.parquet"

    logger.info(f"入力ファイル: {input_path}")
    logger.info(f"出力ファイル: {output_path}")

    # === データ読み込み ===
    try:
        df = pd.read_parquet(input_path)
        logger.info(f"データ読み込み完了: {df.shape}")
    except Exception as e:
        logger.error(f"データ読み込みエラー: {e}")
        return

    # === 処理実行 ===
    # パラメータ設定
    print("\n=== 最適化版GARCH処理パラメータ ===")
    print("推奨設定:")
    print("  高精度（8時間）: reestimation_frequency=50, n_jobs=8")
    print("  バランス（6時間）: reestimation_frequency=75, n_jobs=8") 
    print("  高速（4時間）: reestimation_frequency=100, n_jobs=8")
    
    user_freq = input("\n再推定間隔を入力してください（デフォルト: 50）: ")
    try:
        reestimation_freq = int(user_freq) if user_freq.strip() else 50
    except:
        reestimation_freq = 50
    
    user_jobs = input(f"並列処理数を入力してください（デフォルト: 8）: ")
    try:
        n_jobs = int(user_jobs) if user_jobs.strip() else 8
    except:
        n_jobs = 8
    
    # エンジン初期化と実行
    engine = OptimizedGARCHFeatureEngine(n_jobs=n_jobs)
    
    df_with_garch = engine.add_arima_garch_residuals(
        df, 
        estimation_window=1000,
        reestimation_frequency=reestimation_freq,
        checkpoint_frequency=50000  # 5万ステップごとに途中保存
    )

    # === 保存 ===
    final_feature_count = df_with_garch.shape[1]
    final_output_path = output_path.parent / f"04_with_tier1_complete_{final_feature_count}features.parquet"
    
    logger.info(f"最終結果を保存中: {final_output_path}")
    df_with_garch.to_parquet(final_output_path, compression='snappy')
    
    logger.info("🎉 最適化版GARCH特徴量の追加が完了しました！ 🎉")
    logger.info(f"出力ファイル: {final_output_path}")
    logger.info(f"最終特徴量数: {final_feature_count}")

if __name__ == "__main__":
    main()