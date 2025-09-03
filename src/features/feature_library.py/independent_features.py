import numba
from numba import njit
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union
import logging
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp
from functools import partial
import warnings
import time
warnings.filterwarnings('ignore')
from pathlib import Path

# 数学・統計
from scipy import stats
from scipy.stats import skew, kurtosis, entropy as scipy_entropy
from scipy.stats import normaltest, shapiro, anderson
from scipy.special import gamma, beta, digamma, polygamma
from scipy.integrate import quad

# 信号処理・物理
from scipy.signal import welch, periodogram, find_peaks, savgol_filter
from scipy.signal import hilbert, correlate, coherence, spectrogram
from scipy.fft import fft, fftfreq, rfft, rfftfreq, fftshift
from scipy.ndimage import gaussian_filter, uniform_filter, median_filter

# 空間・幾何
from scipy.spatial.distance import pdist, squareform
from scipy.spatial import ConvexHull, Voronoi
from scipy.optimize import curve_fit, minimize_scalar

# ウェーブレット
import pywt

# エントロピー
try:
    import entropy as ent
    ENTROPY_AVAILABLE = True
except ImportError:
    ENTROPY_AVAILABLE = False

# 機械学習
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA, FastICA
from sklearn.cluster import KMeans
from sklearn.metrics import mutual_info_score
from sklearn.feature_selection import mutual_info_regression

# ログ設定
logger = logging.getLogger(__name__)

from functools import wraps

# ファイルの最初の方にあるこの関数を置き換えてください
def handle_zero_std(func):
    """
    標準偏差がゼロに近い、またはNaNの場合に計算をスキップするデコレータ。
    rolling().apply()で使うヘルパー関数を安全にする。
    🔥 複数列データ（DataFrame）が渡された場合にも対応するロバスト版 🔥
    """
    @wraps(func)
    def wrapper(self, x, *args, **kwargs):
        try:
            # 安全装置：入力データ(x)の標準偏差をチェック
            if hasattr(x, 'std'):
                std_dev = x.std()
            else:
                std_dev = np.std(x)

            # --- ここからが修正箇所 ---
            is_scalar = np.isscalar(std_dev)
            
            # std_devが配列やSeriesの場合のチェック
            if not is_scalar and isinstance(std_dev, (pd.Series, np.ndarray)):
                # 全ての標準偏差がゼロに近いか、または全てNaNかチェック
                if (std_dev < 1e-9).all() or pd.isna(std_dev).all():
                    return 0.0
            
            # std_devがスカラーの場合のチェック
            elif is_scalar:
                if std_dev < 1e-9 or np.isnan(std_dev):
                    return 0.0
            # --- ここまでが修正箇所 ---

            # チェックをパスしたら、元の関数を実行
            return func(self, x, *args, **kwargs)

        except Exception as e:
            # 予期せぬエラーが発生した場合でも、計算を止めずに0.0を返すフォールバック
            # logger.warning(f"Inside handle_zero_std for {func.__name__}: an error occurred: {e}")
            return 0.0
    return wrapper


# ========== 🔥 重大改善: 並列処理用グローバル変数（巨大データコピー完全排除）🔥 ==========
GLOBAL_DF = None
GLOBAL_AVAILABLE_TIMEFRAMES = None
GLOBAL_MULTI_TIMEFRAME_MODE = False

def worker_initializer(df_data, available_timeframes, multi_timeframe_mode):
    """
    🚀 並列処理ワーカープロセス初期化関数
    
    各ワーカープロセスが起動時に1回だけ実行され、
    巨大なDataFrameをプロセス内のグローバル変数として共有する。
    これにより、プロセス間での巨大データコピーを完全に排除し、
    パフォーマンスを劇的に向上させる。
    """
    global GLOBAL_DF, GLOBAL_AVAILABLE_TIMEFRAMES, GLOBAL_MULTI_TIMEFRAME_MODE
    GLOBAL_DF = df_data
    GLOBAL_AVAILABLE_TIMEFRAMES = available_timeframes
    GLOBAL_MULTI_TIMEFRAME_MODE = multi_timeframe_mode

class AdvancedIndependentFeatures:
    """
    全分野網羅型独立特徴量クラス - マルチタイムフレーム完全対応版（並列処理最適化済み）
    数学、物理、生物学、経済学、情報理論等から1200+特徴量を実装
    
    🔥【重大改善点】🔥
    1. 並列処理時のインデックス不整合を完全解決
    2. 巨大データの無駄なコピーを完全排除（initializer pattern導入）
    3. 既存コードは一切破壊されない安全設計
    4. テストモード・進捗表示・エラー耐性の大幅向上
    5. 高速Pandasベクトル化処理
    6. broadcast エラー完全解決
    7. 細分化並列処理
    """
    
    def __init__(self, n_processes: Optional[int] = None, window_size: int = 100,
                 multi_timeframe: bool = False, test_mode: bool = False,
                 params: Optional[Dict[str, List]] = None):
        self.n_processes = n_processes or max(1, mp.cpu_count() - 1)
        self.window_size = window_size
        self.multi_timeframe = multi_timeframe
        self.test_mode = test_mode

        # デフォルトのパラメータを設定
        default_params = {
            # 1. 基本テクニカル指標
            "rsi_periods": [7, 14, 21, 30, 50],
            "macd_settings": [(8, 21, 5), (12, 26, 9), (19, 39, 9)],
            "bollinger_settings": [(10, 1.5), (20, 2), (30, 2.5)],
            "atr_periods": [14, 21, 30],
            "adx_periods": [14, 21],
            "cci_periods": [14, 20, 30],
            "williams_r_periods": [14, 21],
            "aroon_periods": [14, 25],
            "stochastic_settings": [(14, 3), (21, 5), (5, 3)],
            # 2. 出来高系指標
            "cmf_periods": [20, 21],
            "mfi_periods": [14, 21],
            "vol_roc_periods": [12, 25],
            # 3. トレンド系指標
            "short_ma_periods": [5, 8, 10, 13, 20, 21],
            "long_ma_periods": [34, 50, 89, 144, 200],
            "ma_deviation_periods": [20, 50, 200],
            "tma_periods": [20, 50],
            "zlema_periods": [20, 50],
            "dema_periods": [20, 50],
            "tema_periods": [20, 50],
            # 4. ボラティリティ系指標
            "volatility_bb_settings": [(10, 1.5), (20, 2), (20, 2.5), (50, 2)],
            "kc_periods": [20, 50],
            "dc_periods": [20, 55],
            "atr_periods_vol": [14, 21],
            "hist_vol_periods": [20, 30, 60],
            # 5. サポート・レジスタンス系指標
            "price_channel_periods": [20, 50],
            # 7. 数学・統計学
            "stat_windows": [10, 20, 50],
            "dist_windows": [20, 50],
            "robust_stat_windows": [15, 30],
            "order_stat_windows": [10, 25, 50],
            # 8. 物理学・工学
            "hilbert_windows": [32, 64],
            "autocorr_lags": [5, 10, 20],
            "spectral_windows": [64, 128],
            "fourier_windows": [32, 64, 128],
            "wavelets": ['db4', 'haar', 'coif2', 'bior2.2'],
            "cwt_windows": [32, 64],
            "gaussian_sigmas": [1, 2, 3],
            "median_sizes": [3, 5, 7],
            "savgol_windows": [11, 21],
            "energy_windows": [10, 20, 50],
            # 9. 情報理論
            "entropy_windows": [20, 50],
            "adv_entropy_windows": [30, 60],
            "lz_windows": [50, 100],
            "kolmogorov_windows": [30, 60],
            "mutual_info_lags": [1, 5, 10],
            # 12. 地球科学・物理
            "self_similarity_scales": [5, 10, 20],
            # 14. 経済学・金融
            "var_confidence_levels": [0.95, 0.99]
        }
        
        # ユーザー指定のパラメータでデフォルトを上書き
        if params:
            default_params.update(params)

        # パラメータをインスタンス変数として設定
        for key, value in default_params.items():
            setattr(self, key, value)
        
        # マルチタイムフレーム用の新機能
        self.available_timeframes = []
        self.failed_timeframes = []
        self.failed_features = []
        self.total_features_generated = 0
        
        logger.info(f"""
        🚀 高度特徴量処理エンジン初期化
        ├── プロセス数: {self.n_processes}
        ├── 窓サイズ: {window_size}
        ├── マルチタイムフレーム: {'有効' if multi_timeframe else '無効（従来モード）'}
        └── テストモード: {'有効（1000行制限）' if test_mode else '無効（全データ処理）'}
        """)
    
    def _detect_available_timeframes(self, df: pd.DataFrame) -> List[str]:
        """
        🔍 データから利用可能な時間軸を自動検出
        
        Args:
            df: OHLCVデータ
            
        Returns:
            利用可能な時間軸のリスト（例: ['1T', '5T', '15T', '1H']）
        """
        detected_timeframes = ['1T']  # 1分足は基本として常に含む
        
        base_cols = ['open', 'high', 'low', 'close', 'volume']
        for col in df.columns:
            for base in base_cols:
                if col.startswith(f'{base}_') and col != base:
                    tf = col.split('_')[1]  # close_5T → '5T'
                    if tf not in detected_timeframes:
                        detected_timeframes.append(tf)
        
        # 時間軸を論理的順序でソート
        timeframe_order = {'1T': 1, '5T': 5, '15T': 15, '30T': 30, '1H': 60, '4H': 240, '1D': 1440}
        detected_timeframes.sort(key=lambda x: timeframe_order.get(x, 999))
        
        self.available_timeframes = detected_timeframes
        logger.info(f"🔍 検出された時間軸: {detected_timeframes}")
        return detected_timeframes
    
    def _extract_ohlcv_for_timeframe(self, df: pd.DataFrame, timeframe: str) -> Dict[str, pd.Series]:
        """
        📊 指定時間軸のOHLCVデータを取得
        
        Args:
            df: 全データ
            timeframe: 時間軸（例: '5T', '1H'）
            
        Returns:
            指定時間軸のOHLCVデータ辞書
        """
        if timeframe == '1T':
            # 1分足（基本）
            return {
                'open': pd.Series(df['open'].values),
                'high': pd.Series(df['high'].values),
                'low': pd.Series(df['low'].values),
                'close': pd.Series(df['close'].values),
                'volume': pd.Series(df['volume'].values)
            }
        else:
            # マルチタイムフレーム
            suffix = f'_{timeframe}'
            return {
                'open': pd.Series(df[f'open{suffix}'].values) if f'open{suffix}' in df.columns else pd.Series(df['open'].values),
                'high': pd.Series(df[f'high{suffix}'].values) if f'high{suffix}' in df.columns else pd.Series(df['high'].values),
                'low': pd.Series(df[f'low{suffix}'].values) if f'low{suffix}' in df.columns else pd.Series(df['low'].values),
                'close': pd.Series(df[f'close{suffix}'].values) if f'close{suffix}' in df.columns else pd.Series(df['close'].values),
                'volume': pd.Series(df[f'volume{suffix}'].values) if f'volume{suffix}' in df.columns else pd.Series(df['volume'].values)
            }
    
    def _estimate_remaining_time(self, current_step: int, total_steps: int, start_time: float) -> float:
        """⏰ 残り時間を動的に予測"""
        elapsed = time.time() - start_time
        if current_step == 0:
            return 0.0
        
        time_per_step = elapsed / current_step
        remaining_steps = total_steps - current_step
        return (remaining_steps * time_per_step) / 60  # 分単位
    
    def _count_expected_features(self) -> int:
        """📊 期待される特徴量数を概算"""
        if not self.multi_timeframe:
            return 574  # 従来モード
        
        base_features_per_timeframe = 120  # 概算
        return len(self.available_timeframes) * base_features_per_timeframe
    
    def calculate_all_features(self, df: pd.DataFrame, chunk_id: str = "") -> pd.DataFrame:
        """
        全ての独立特徴量を計算（マルチタイムフレーム完全対応版）
        
        Args:
            df: 価格データ (必須列: open, high, low, close, volume)
            chunk_id: チャンクID（ログ用）
            
        Returns:
            特徴量が追加されたDataFrame
        """
        if self.test_mode and len(df) > 1000:
            df = df.head(1000)
            logger.info("🧪 テストモード: 1000行で動作確認")
        
        logger.info(f"🚀 高度特徴量計算開始 - Chunk {chunk_id}, Shape: {df.shape}")
        logger.info(f"📊 マルチタイムフレームモード: {'有効' if self.multi_timeframe else '無効'}")
        
        # 基本チェック
        required_cols = ['open', 'high', 'low', 'close', 'volume']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            raise ValueError(f"必須列が不足: {missing_cols}")
        
        # マルチタイムフレーム用の追加処理
        if self.multi_timeframe:
            self._detect_available_timeframes(df)
            expected_features = self._count_expected_features()
            logger.info(f"""
            🎯 マルチタイムフレーム特徴量計算開始
            ├── データサイズ: {df.shape}
            ├── 利用可能時間軸: {self.available_timeframes}
            ├── 予想特徴量数: {expected_features}個
            └── 従来比拡張: {expected_features/574:.1f}倍
            """)
        
        result_df = df.copy()
        start_time = time.time()
        
        # ★ 🔥 重大改善: 並列処理の巨大データコピー問題を完全解決 ★
        # initializer パターンを使用してプロセス間での巨大データコピーを排除
        
        # ★ 細分化並列処理タスクリスト（マルチタイムフレーム対応）★
        feature_tasks = [
            # 基本テクニカル（細分化）
            'rsi_group', 'macd_group', 'bollinger_group', 'atr_group', 'adx_group',
            'oscillator_basic', 'oscillator_advanced',
            
            # 出来高系（細分化）
            'volume_basic', 'volume_advanced', 'volume_price_trend',
            
            # トレンド系（細分化）
            'moving_averages_short', 'moving_averages_long', 'moving_averages_advanced', 'trend_crosses',
            
            # ボラティリティ系（細分化）
            'volatility_bands', 'volatility_channels', 'volatility_measures',
            
            # サポート・レジスタンス系
            'support_resistance',
            
            # ローソク足系
            'candlestick_patterns',
            
            # 数学・統計学（細分化）
            'statistical_moments_basic', 'statistical_moments_advanced',
            'probability_distributions', 'robust_statistics', 'order_statistics',
            
            # 物理学・工学（細分化）
            'signal_processing_basic', 'signal_processing_advanced',
            'fourier_analysis', 'wavelet_analysis', 'filtering_features', 'energy_features',
            
            # 情報理論（細分化）
            'entropy_features_basic', 'entropy_features_advanced',
            'complexity_features', 'information_measures',
            
            # 生物学・医学
            'biometric_features', 'physiological_features', 'circadian_features',
            
            # 心理学・認知科学
            'psychological_features', 'behavioral_features',
            
            # 地球科学・物理
            'fractal_features',
            'chaos_features',
            'turbulence_features',
            
            # 化学・材料科学
            'molecular_features', 'crystallographic_features',
            
            # 経済学・金融
            'econometric_features', 
            'risk_features', 'game_theory_features',
            
            # その他分野
            'network_features', 'social_physics_features', 'acoustic_features',
            'linguistic_features', 'aesthetic_features', 'musical_features',
            'astronomical_features', 
            'cosmological_features',
            'biomechanical_features', 'performance_features'
        ]
        
        # 🔥 並列処理実行（initializer pattern使用）🔥
        with ProcessPoolExecutor(
            max_workers=self.n_processes,
            initializer=worker_initializer,
            initargs=(df, self.available_timeframes, self.multi_timeframe)
        ) as executor:
            
            # タスク投入（データ渡しなし - グローバル変数から参照）
            future_to_name = {
                executor.submit(self._execute_feature_calculation, task_name): task_name
                for task_name in feature_tasks
            }
            
            # 結果回収（進捗表示付き）
            total_tasks = len(feature_tasks)
            completed = 0

            for future in as_completed(future_to_name):
                feature_name = future_to_name[future]
                completed += 1
                progress = (completed / total_tasks) * 100
                remaining_time = self._estimate_remaining_time(completed, total_tasks, start_time)

                try:
                    features = future.result()
                    # 特徴量をマージ
                    for col_name, values in features.items():
                        result_df[col_name] = values
                    
                    new_features_count = len(features)
                    self.total_features_generated += new_features_count
                    
                    logger.info(f"⚡ 進捗: {progress:.1f}% ({completed}/{total_tasks}) - {feature_name} 完了: {new_features_count}特徴量 (残り約{remaining_time:.1f}分)")
                    
                except Exception as exc:
                    logger.error(f"❌ 進捗: {progress:.1f}% ({completed}/{total_tasks}) - {feature_name} でエラー: {exc}")
                    self.failed_features.append(feature_name)
                    continue
        
        # 最終サマリー
        total_time = time.time() - start_time
        success_rate = (total_tasks - len(self.failed_features)) / total_tasks * 100
        final_features = len(result_df.columns)
        
        logger.info(f"""
        🎉 高度特徴量計算完了！ - Chunk {chunk_id}
        ├── 最終shape: {result_df.shape}
        ├── 生成特徴量: {final_features}個
        ├── 成功率: {success_rate:.1f}%
        ├── 処理時間: {total_time/60:.1f}分
        └── 失敗タスク: {self.failed_features}
        """)
        
        return result_df
    
    def _execute_feature_calculation(self, task_name: str) -> Dict[str, np.ndarray]:
        """
        🔥 並列処理用特徴量計算実行関数（グローバル変数参照版）
        
        Args:
            task_name: 計算する特徴量グループ名
            
        Returns:
            計算された特徴量の辞書
        """
        # --- ここからが新しいコード ---
        import io
        import contextlib
        from traceback import format_exc

        global GLOBAL_DF, GLOBAL_AVAILABLE_TIMEFRAMES, GLOBAL_MULTI_TIMEFRAME_MODE
        
        df = GLOBAL_DF
        self.available_timeframes = GLOBAL_AVAILABLE_TIMEFRAMES
        self.multi_timeframe = GLOBAL_MULTI_TIMEFRAME_MODE
        
        method_name = f'_calculate_{task_name}'
        if hasattr(self, method_name):
            method = getattr(self, method_name)
            
            # 低レベルの標準エラー出力をキャプチャするための準備
            stderr_capture = io.StringIO()
            
            try:
                # withブロックの間だけ、標準エラー出力がstderr_captureにリダイレクトされる
                with contextlib.redirect_stderr(stderr_capture):
                    result = method(df)
                
                # キャプチャしたエラー出力を確認
                captured_output = stderr_capture.getvalue()
                if 'DLASCLS' in captured_output:
                    # DLASCLSメッセージが検出された場合、タスク名と共に出力
                    logger.error(f"""
                    🚨 LOW-LEVEL WARNING DETECTED in task: '{task_name}' 🚨
                    ├── The underlying library (LAPACK) printed the following message:
                    └───> {captured_output.strip()}
                    """)
                
                return result

            except Exception as e:
                # Pythonレベルで発生したその他のエラーもログに出力
                logger.error(f"An unexpected Python error occurred in task '{task_name}': {e}\n{format_exc()}")
                return {} # エラーが発生した場合は空の辞書を返す
            # --- ここまでが新しいコード ---
        else:
            logger.warning(f"未定義の特徴量計算タスク: {task_name}")
            return {}
        
    def _process_feature_group(self, df: pd.DataFrame, calculation_logic: callable, group_name: str) -> Dict[str, np.ndarray]:
        """
        特徴量グループ計算の共通処理。
        時間軸ごとのループとエラーハンドリングを共通化する。
        """
        features = {}
        timeframes_to_process = self.available_timeframes if self.multi_timeframe else ['1T']

        for tf in timeframes_to_process:
            try:
                # 各時間軸のOHLCVデータを取得
                ohlcv = self._extract_ohlcv_for_timeframe(df, tf)
                # 各特徴量グループの計算ロジックを実行
                new_features = calculation_logic(ohlcv, tf)
                features.update(new_features)
            except Exception as e:
                logger.warning(f"{group_name}計算エラー - 時間軸 {tf}: {e}")
                continue
        return features
        
# ========== 基本テクニカル指標（マルチタイムフレーム完全対応版） ==========
    
    def _rsi_calculation_logic(self, ohlcv: Dict[str, pd.Series], timeframe: str) -> Dict[str, np.ndarray]:
        """RSIグループの具体的な計算ロジック"""
        features = {}
        close_series = ohlcv['close']
        suffix = f'_{timeframe}' if self.multi_timeframe else ''

        for period in self.rsi_periods:
            rsi = self._compute_rsi_vectorized(close_series, period)
            features[f'rsi_{period}{suffix}'] = rsi.values
            features[f'rsi_{period}_normalized{suffix}'] = ((rsi - 50) / 50).values
        return features

    def _calculate_rsi_group(self, df: pd.DataFrame) -> Dict[str, np.ndarray]:
        """RSI系指標グループ（共通処理ヘルパー使用）"""
        logic = partial(self._rsi_calculation_logic)
        return self._process_feature_group(df, logic, "RSI")
        
    def _macd_calculation_logic(self, ohlcv: Dict[str, pd.Series], timeframe: str) -> Dict[str, np.ndarray]:
        """MACDグループの具体的な計算ロジック"""
        features = {}
        close_series = ohlcv['close']
        suffix = f'_{timeframe}' if self.multi_timeframe else ''

        for fast, slow, signal in self.macd_settings:
            macd, sig, hist = self._compute_macd_vectorized(close_series, fast, slow, signal)
            features[f'macd_{fast}_{slow}{suffix}'] = macd.values
            features[f'macd_signal_{fast}_{slow}{suffix}'] = sig.values
            features[f'macd_hist_{fast}_{slow}{suffix}'] = hist.values
        return features

    def _calculate_macd_group(self, df: pd.DataFrame) -> Dict[str, np.ndarray]:
        """MACD系指標グループ（共通処理ヘルパー使用）"""
        logic = partial(self._macd_calculation_logic)
        return self._process_feature_group(df, logic, "MACD")
        
    def _bollinger_calculation_logic(self, ohlcv: Dict[str, pd.Series], timeframe: str) -> Dict[str, np.ndarray]:
        """ボリンジャーバンドグループの具体的な計算ロジック"""
        features = {}
        close_series = ohlcv['close']
        suffix = f'_{timeframe}' if self.multi_timeframe else ''
        
        for period, std_mult in self.bollinger_settings:
            bb_upper, bb_mid, bb_lower = self._compute_bollinger_bands_vectorized(close_series, period, std_mult)
            features[f'bb_position_{period}{suffix}'] = ((close_series - bb_lower) / (bb_upper - bb_lower + 1e-8)).values
            features[f'bb_width_{period}{suffix}'] = ((bb_upper - bb_lower) / bb_mid).values
        return features

    def _calculate_bollinger_group(self, df: pd.DataFrame) -> Dict[str, np.ndarray]:
        """ボリンジャーバンド系指標グループ（共通処理ヘルパー使用）"""
        logic = partial(self._bollinger_calculation_logic)
        return self._process_feature_group(df, logic, "Bollinger")

    def _atr_calculation_logic(self, ohlcv: Dict[str, pd.Series], timeframe: str) -> Dict[str, np.ndarray]:
        """ATRグループの具体的な計算ロジック"""
        features = {}
        high_series = ohlcv['high']
        low_series = ohlcv['low']
        close_series = ohlcv['close']
        suffix = f'_{timeframe}' if self.multi_timeframe else ''

        for period in self.atr_periods:
            atr = self._compute_atr_vectorized(high_series, low_series, close_series, period)
            features[f'atr_{period}{suffix}'] = atr.values
            features[f'atr_ratio_{period}{suffix}'] = (atr / close_series).values
            features[f'atr_percent_{period}{suffix}'] = ((atr / close_series) * 100).values
        return features

    def _calculate_atr_group(self, df: pd.DataFrame) -> Dict[str, np.ndarray]:
        """ATR系指標グループ（共通処理ヘルパー使用）"""
        logic = partial(self._atr_calculation_logic)
        return self._process_feature_group(df, logic, "ATR")
    
    def _adx_calculation_logic(self, ohlcv: Dict[str, pd.Series], timeframe: str) -> Dict[str, np.ndarray]:
        """ADXグループの具体的な計算ロジック"""
        features = {}
        high_series = ohlcv['high']
        low_series = ohlcv['low']
        close_series = ohlcv['close']
        suffix = f'_{timeframe}' if self.multi_timeframe else ''

        for period in self.adx_periods:
            adx, di_plus, di_minus = self._compute_adx_vectorized(high_series, low_series, close_series, period)
            features[f'adx_{period}{suffix}'] = adx.values
            features[f'di_plus_{period}{suffix}'] = di_plus.values
            features[f'di_minus_{period}{suffix}'] = di_minus.values
            features[f'di_diff_{period}{suffix}'] = (di_plus - di_minus).values
            features[f'dx_{period}{suffix}'] = (np.abs(di_plus - di_minus) / (di_plus + di_minus + 1e-8) * 100).values
        return features

    def _calculate_adx_group(self, df: pd.DataFrame) -> Dict[str, np.ndarray]:
        """ADX系指標グループ（共通処理ヘルパー使用）"""
        logic = partial(self._adx_calculation_logic)
        return self._process_feature_group(df, logic, "ADX")

    
    def _oscillator_basic_logic(self, ohlcv: Dict[str, pd.Series], timeframe: str) -> Dict[str, np.ndarray]:
        """基本オシレーターの具体的な計算ロジック"""
        features = {}
        close_series = ohlcv['close']
        high_series = ohlcv['high']
        low_series = ohlcv['low']
        suffix = f'_{timeframe}' if self.multi_timeframe else ''

        # Parabolic SAR
        sar = self._compute_parabolic_sar_iterative(high_series, low_series, close_series)
        features[f'parabolic_sar{suffix}'] = sar.values
        features[f'sar_signal{suffix}'] = (close_series > sar).astype(int).values
        
        # Commodity Channel Index (CCI)
        for period in self.cci_periods:
            cci = self._compute_cci_vectorized(high_series, low_series, close_series, period)
            features[f'cci_{period}{suffix}'] = cci.values
            features[f'cci_overbought_{period}{suffix}'] = (cci > 100).astype(int).values
            features[f'cci_oversold_{period}{suffix}'] = (cci < -100).astype(int).values
        return features

    def _calculate_oscillator_basic(self, df: pd.DataFrame) -> Dict[str, np.ndarray]:
        """基本オシレーター系指標（共通処理ヘルパー使用）"""
        logic = partial(self._oscillator_basic_logic)
        return self._process_feature_group(df, logic, "Basic Oscillators")
        
    def _oscillator_advanced_logic(self, ohlcv: Dict[str, pd.Series], timeframe: str) -> Dict[str, np.ndarray]:
        """高度オシレーターの具体的な計算ロジック"""
        features = {}
        close_series = ohlcv['close']
        high_series = ohlcv['high']
        low_series = ohlcv['low']
        suffix = f'_{timeframe}' if self.multi_timeframe else ''

        # Williams %R
        for period in self.williams_r_periods:
            williams_r = self._compute_williams_r_vectorized(high_series, low_series, close_series, period)
            features[f'williams_r_{period}{suffix}'] = williams_r.values
            features[f'williams_overbought_{period}{suffix}'] = (williams_r > -20).astype(int).values
            features[f'williams_oversold_{period}{suffix}'] = (williams_r < -80).astype(int).values
        
        # Ultimate Oscillator
        ult_osc = self._compute_ultimate_oscillator_vectorized(high_series, low_series, close_series)
        features[f'ultimate_oscillator{suffix}'] = ult_osc.values
        features[f'ult_osc_overbought{suffix}'] = (ult_osc > 70).astype(int).values
        features[f'ult_osc_oversold{suffix}'] = (ult_osc < 30).astype(int).values
        
        # Aroon Indicator
        for period in self.aroon_periods:
            aroon_up, aroon_down = self._compute_aroon_vectorized(high_series, low_series, period)
            features[f'aroon_up_{period}{suffix}'] = aroon_up.values
            features[f'aroon_down_{period}{suffix}'] = aroon_down.values
            features[f'aroon_oscillator_{period}{suffix}'] = (aroon_up - aroon_down).values
            if period == 14:
                features[f'AROONU_14{suffix}'] = aroon_up.values
                if timeframe == '1T' or not self.multi_timeframe:
                    features['AROONU_14_1T'] = aroon_up.values
        
        # Stochastic Oscillator
        for k_period, d_period in self.stochastic_settings:
            k_percent, d_percent = self._compute_stochastic_vectorized(high_series, low_series, close_series, k_period, d_period)
            features[f'stoch_k_{k_period}_{d_period}{suffix}'] = k_percent.values
            features[f'stoch_d_{k_period}_{d_period}{suffix}'] = d_percent.values
            features[f'stoch_overbought_{k_period}{suffix}'] = (k_percent > 80).astype(int).values
            features[f'stoch_oversold_{k_period}{suffix}'] = (k_percent < 20).astype(int).values
        
        # Slow Stochastic
        slow_k, slow_d = self._compute_slow_stochastic_vectorized(high_series, low_series, close_series)
        features[f'slow_stoch_k{suffix}'] = slow_k.values
        features[f'slow_stoch_d{suffix}'] = slow_d.values
        return features

    def _calculate_oscillator_advanced(self, df: pd.DataFrame) -> Dict[str, np.ndarray]:
        """高度オシレーター系指標（共通処理ヘルパー使用）"""
        logic = partial(self._oscillator_advanced_logic)
        return self._process_feature_group(df, logic, "Advanced Oscillators")
    
# ========== ベクトル化計算メソッド（Pandas最適化済み） ==========

    @staticmethod
    @njit(cache=True)
    def _numba_mean_abs_deviation(x: np.ndarray) -> float:
        """Numba JIT化: 平均絶対偏差 (CCI用)"""
        if x.std() < 1e-9:
            return 1.0 # ゼロ除算を避けるため1を返す
        return np.mean(np.abs(x - np.mean(x)))

    @staticmethod
    @njit(cache=True)
    def _numba_aroon_up(x: np.ndarray, period: int) -> float:
        """Numba JIT化: Aroon-Up"""
        if x.size == 0:
            return 0.0
        # np.argmaxはNumbaでサポートされている
        return (period - (x.size - 1 - np.argmax(x))) / period * 100

    @staticmethod
    @njit(cache=True)
    def _numba_aroon_down(x: np.ndarray, period: int) -> float:
        """Numba JIT化: Aroon-Down"""
        if x.size == 0:
            return 0.0
        # np.argminはNumbaでサポートされている
        return (period - (x.size - 1 - np.argmin(x))) / period * 100


    def _compute_parabolic_sar_iterative(self, high_series: pd.Series, low_series: pd.Series, close_series: pd.Series) -> pd.Series:
        """
        Parabolic SARの逐次計算実装。
        ※計算の特性上、過去の値に依存するためループ処理を使用しています。
        """
        try:
            length = len(high_series)
            sar = np.zeros(length)
            ep = np.zeros(length)
            af = np.zeros(length)
            trend = np.zeros(length)
            
            if length < 2:
                return pd.Series(close_series.values)
            
            sar[0] = low_series.iloc[0]
            ep[0] = high_series.iloc[0]
            af[0] = 0.02
            trend[0] = 1
            
            for i in range(1, length):
                sar[i] = sar[i-1] + af[i-1] * (ep[i-1] - sar[i-1])
                
                if trend[i-1] == 1:
                    if low_series.iloc[i] <= sar[i]:
                        trend[i] = -1
                        sar[i] = ep[i-1]
                        ep[i] = low_series.iloc[i]
                        af[i] = 0.02
                    else:
                        trend[i] = 1
                        if high_series.iloc[i] > ep[i-1]:
                            ep[i] = high_series.iloc[i]
                            af[i] = min(0.2, af[i-1] + 0.02)
                        else:
                            ep[i] = ep[i-1]
                            af[i] = af[i-1]
                else:
                    if high_series.iloc[i] >= sar[i]:
                        trend[i] = 1
                        sar[i] = ep[i-1]
                        ep[i] = high_series.iloc[i]
                        af[i] = 0.02
                    else:
                        trend[i] = -1
                        if low_series.iloc[i] < ep[i-1]:
                            ep[i] = low_series.iloc[i]
                            af[i] = min(0.2, af[i-1] + 0.02)
                        else:
                            ep[i] = ep[i-1]
                            af[i] = af[i-1]
            
            return pd.Series(sar, index=close_series.index)
            
        except Exception as e:
            logger.warning(f"Parabolic SAR計算エラー: {e}")
            return pd.Series(close_series.values)

    def _compute_cci_vectorized(self, high_series: pd.Series, low_series: pd.Series, close_series: pd.Series, period: int) -> pd.Series:
        """CCIベクトル化計算 (Numba高速化適用)"""
        typical_price = (high_series + low_series + close_series) / 3
        sma_tp = typical_price.rolling(window=period).mean()
        # Numbaヘルパーを呼び出すように変更
        mad = typical_price.rolling(window=period).apply(self._numba_mean_abs_deviation, raw=True)
        cci = (typical_price - sma_tp) / (0.015 * mad)
        return cci.fillna(0)

    def _compute_williams_r_vectorized(self, high_series: pd.Series, low_series: pd.Series, close_series: pd.Series, period: int) -> pd.Series:
        """Williams %Rベクトル化計算"""
        highest_high = high_series.rolling(window=period).max()
        lowest_low = low_series.rolling(window=period).min()
        williams_r = -100 * (highest_high - close_series) / (highest_high - lowest_low + 1e-8)
        return williams_r.fillna(0)

    def _compute_ultimate_oscillator_vectorized(self, high_series: pd.Series, low_series: pd.Series, close_series: pd.Series) -> pd.Series:
        """Ultimate Oscillatorベクトル化計算"""
        bp = close_series - pd.concat([low_series, close_series.shift(1)], axis=1).min(axis=1)
        tr = pd.concat([high_series - low_series, 
                       np.abs(high_series - close_series.shift(1)), 
                       np.abs(low_series - close_series.shift(1))], axis=1).max(axis=1)
        
        avg7 = bp.rolling(window=7).sum() / (tr.rolling(window=7).sum() + 1e-8)
        avg14 = bp.rolling(window=14).sum() / (tr.rolling(window=14).sum() + 1e-8)
        avg28 = bp.rolling(window=28).sum() / (tr.rolling(window=28).sum() + 1e-8)
        
        ult_osc = 100 * (4 * avg7 + 2 * avg14 + avg28) / 7
        return ult_osc.fillna(0)

    def _compute_aroon_vectorized(self, high_series: pd.Series, low_series: pd.Series, period: int) -> Tuple[pd.Series, pd.Series]:
        """Aroon Indicatorベクトル化計算 (Numba高速化適用)"""
        # Numbaヘルパーを呼び出すように変更
        aroon_up_func = partial(self._numba_aroon_up, period=period)
        aroon_down_func = partial(self._numba_aroon_down, period=period)
        
        aroon_up = high_series.rolling(window=period).apply(aroon_up_func, raw=True)
        aroon_down = low_series.rolling(window=period).apply(aroon_down_func, raw=True)
        return aroon_up.fillna(0), aroon_down.fillna(0)

    def _compute_stochastic_vectorized(self, high_series: pd.Series, low_series: pd.Series, close_series: pd.Series, k_period: int, d_period: int) -> Tuple[pd.Series, pd.Series]:
        """Stochastic Oscillatorベクトル化計算"""
        lowest_low = low_series.rolling(window=k_period).min()
        highest_high = high_series.rolling(window=k_period).max()
        
        k_percent = 100 * (close_series - lowest_low) / (highest_high - lowest_low + 1e-8)
        d_percent = k_percent.rolling(window=d_period).mean()
        
        return k_percent.fillna(0), d_percent.fillna(0)

    def _compute_slow_stochastic_vectorized(self, high_series: pd.Series, low_series: pd.Series, close_series: pd.Series) -> Tuple[pd.Series, pd.Series]:
        """Slow Stochasticベクトル化計算"""
        k_fast, _ = self._compute_stochastic_vectorized(high_series, low_series, close_series, 14, 3)
        slow_k = k_fast.rolling(window=3).mean()
        slow_d = slow_k.rolling(window=3).mean()
        return slow_k.fillna(0), slow_d.fillna(0)
    
    def _compute_rsi_vectorized(self, close_series: pd.Series, period: int) -> pd.Series:
        """RSIベクトル化計算（Pandas最適化版）"""
        delta = close_series.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / (loss + 1e-8)
        rsi = 100 - (100 / (1 + rs))
        return rsi.fillna(50)
    
    def _compute_macd_vectorized(self, close_series: pd.Series, fast: int, slow: int, signal: int) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """MACDベクトル化計算（Pandas最適化版）"""
        ema_fast = close_series.ewm(span=fast).mean()
        ema_slow = close_series.ewm(span=slow).mean()
        macd = ema_fast - ema_slow
        signal_line = macd.ewm(span=signal).mean()
        histogram = macd - signal_line
        return macd.fillna(0), signal_line.fillna(0), histogram.fillna(0)
    
    def _compute_bollinger_bands_vectorized(self, close_series: pd.Series, period: int, std_mult: float) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """ボリンジャーバンドベクトル化計算（Pandas最適化版）"""
        sma = close_series.rolling(window=period).mean()
        std = close_series.rolling(window=period).std()
        upper = sma + (std * std_mult)
        lower = sma - (std * std_mult)
        return upper.fillna(close_series), sma.fillna(close_series), lower.fillna(close_series)
    
    def _compute_atr_vectorized(self, high_series: pd.Series, low_series: pd.Series, close_series: pd.Series, period: int) -> pd.Series:
        """ATRベクトル化計算（Pandas最適化版）"""
        high_low = high_series - low_series
        high_close = np.abs(high_series - close_series.shift(1))
        low_close = np.abs(low_series - close_series.shift(1))
        true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        atr = true_range.rolling(window=period).mean()
        return atr.fillna(0)        

    def _compute_adx_vectorized(self, high_series: pd.Series, low_series: pd.Series, close_series: pd.Series, period: int) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """ADXベクトル化計算（Pandas最適化版）"""
        # True Range
        high_low = high_series - low_series
        high_close = np.abs(high_series - close_series.shift(1))
        low_close = np.abs(low_series - close_series.shift(1))
        true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        
        # Directional Movement
        dm_plus = np.where((high_series.diff() > low_series.diff().abs()) & (high_series.diff() > 0), high_series.diff(), 0)
        dm_minus = np.where((low_series.diff().abs() > high_series.diff()) & (low_series.diff() < 0), low_series.diff().abs(), 0)
        
        # Smoothed values
        tr_smooth = true_range.rolling(window=period).mean()
        dm_plus_smooth = pd.Series(dm_plus).rolling(window=period).mean()
        dm_minus_smooth = pd.Series(dm_minus).rolling(window=period).mean()
        
        # Directional Indicators
        di_plus = (dm_plus_smooth / tr_smooth) * 100
        di_minus = (dm_minus_smooth / tr_smooth) * 100
        
        # ADX
        dx = (np.abs(di_plus - di_minus) / (di_plus + di_minus + 1e-8)) * 100
        adx = dx.rolling(window=period).mean()
        
        return adx.fillna(0), di_plus.fillna(0), di_minus.fillna(0)

 # ========== 出来高系テクニカル指標（マルチタイムフレーム完全対応版） ==========
    
    def _volume_basic_logic(self, ohlcv: Dict[str, pd.Series], timeframe: str) -> Dict[str, np.ndarray]:
        """基本出来高指標の具体的な計算ロジック"""
        features = {}
        close_series = ohlcv['close']
        volume_series = ohlcv['volume']
        suffix = f'_{timeframe}' if self.multi_timeframe else ''

        # On Balance Volume (OBV)
        obv = self._compute_obv_vectorized(close_series, volume_series)
        features[f'obv{suffix}'] = obv.values
        obv_ma_10 = obv.rolling(window=10).mean()
        features[f'obv_ma_10{suffix}'] = obv_ma_10.values
        features[f'obv_signal{suffix}'] = (obv > obv_ma_10).astype(int).values
        
        # Volume Price Trend (VPT)
        vpt = self._compute_vpt_vectorized(close_series, volume_series)
        features[f'vpt{suffix}'] = vpt.values
        return features

    def _calculate_volume_basic(self, df: pd.DataFrame) -> Dict[str, np.ndarray]:
        """基本出来高系指標（共通処理ヘルパー使用）"""
        logic = partial(self._volume_basic_logic)
        return self._process_feature_group(df, logic, "Basic Volume")

    # --- 9/10: Advanced Volume ---
    def _volume_advanced_logic(self, ohlcv: Dict[str, pd.Series], timeframe: str) -> Dict[str, np.ndarray]:
        """高度出来高指標の具体的な計算ロジック"""
        features = {}
        close_series = ohlcv['close']
        high_series = ohlcv['high']
        low_series = ohlcv['low']
        volume_series = ohlcv['volume']
        suffix = f'_{timeframe}' if self.multi_timeframe else ''

        # Accumulation/Distribution Line
        ad_line = self._compute_ad_line_vectorized(high_series, low_series, close_series, volume_series)
        features[f'ad_line{suffix}'] = ad_line.values
        ad_line_ma_10 = ad_line.rolling(window=10).mean()
        features[f'ad_line_ma_10{suffix}'] = ad_line_ma_10.values
        
        # Chaikin Money Flow (CMF)
        for period in self.cmf_periods:
            cmf = self._compute_chaikin_money_flow_vectorized(high_series, low_series, close_series, volume_series, period)
            features[f'cmf_{period}{suffix}'] = cmf.values
            features[f'cmf_bullish_{period}{suffix}'] = (cmf > 0).astype(int).values
        
        # Chaikin A/D Oscillator
        chaikin_osc = self._compute_chaikin_oscillator_vectorized(high_series, low_series, close_series, volume_series)
        features[f'chaikin_oscillator{suffix}'] = chaikin_osc.values
        
        # Money Flow Index (MFI)
        for period in self.mfi_periods:
            mfi = self._compute_money_flow_index_vectorized(high_series, low_series, close_series, volume_series, period)
            features[f'mfi_{period}{suffix}'] = mfi.values
            features[f'mfi_overbought_{period}{suffix}'] = (mfi > 80).astype(int).values
            features[f'mfi_oversold_{period}{suffix}'] = (mfi < 20).astype(int).values
        return features
        
    def _calculate_volume_advanced(self, df: pd.DataFrame) -> Dict[str, np.ndarray]:
        """高度出来高系指標（共通処理ヘルパー使用）"""
        logic = partial(self._volume_advanced_logic)
        return self._process_feature_group(df, logic, "Advanced Volume")

    # --- 10/10: Volume Price Trend ---
    def _volume_price_trend_logic(self, ohlcv: Dict[str, pd.Series], timeframe: str) -> Dict[str, np.ndarray]:
        """出来高価格トレンド指標の具体的な計算ロジック"""
        features = {}
        close_series = ohlcv['close']
        high_series = ohlcv['high']
        low_series = ohlcv['low']
        volume_series = ohlcv['volume']
        suffix = f'_{timeframe}' if self.multi_timeframe else ''

        # Volume Weighted Average Price (VWAP)
        vwap = self._compute_vwap_vectorized(high_series, low_series, close_series, volume_series)
        features[f'vwap{suffix}'] = vwap.values
        features[f'vwap_ratio{suffix}'] = (close_series / (vwap + 1e-8)).values
        
        # Time Weighted Average Price (TWAP)
        twap = close_series.rolling(window=20).mean()
        features[f'twap{suffix}'] = twap.values
        features[f'twap_ratio{suffix}'] = (close_series / (twap + 1e-8)).values
        
        # Volume Rate of Change
        for period in self.vol_roc_periods:
            vol_roc = ((volume_series / volume_series.shift(period)) - 1) * 100
            features[f'volume_roc_{period}{suffix}'] = vol_roc.values
        
        # Ease of Movement (EOM)
        eom = self._compute_ease_of_movement_vectorized(high_series, low_series, volume_series)
        features[f'ease_of_movement{suffix}'] = eom.values
        eom_ma_14 = eom.rolling(window=14).mean()
        features[f'eom_ma_14{suffix}'] = eom_ma_14.values
        
        # Volume Oscillator
        volume_osc = self._compute_volume_oscillator_vectorized(volume_series)
        features[f'volume_oscillator{suffix}'] = volume_osc.values
        
        # Price Volume Trend Oscillator
        pvt_osc = self._compute_pvt_oscillator_vectorized(close_series, volume_series)
        features[f'pvt_oscillator{suffix}'] = pvt_osc.values
        return features

    def _calculate_volume_price_trend(self, df: pd.DataFrame) -> Dict[str, np.ndarray]:
        """出来高価格トレンド系指標（共通処理ヘルパー使用）"""
        logic = partial(self._volume_price_trend_logic)
        return self._process_feature_group(df, logic, "Volume Price Trend")
        
    # ========== トレンド系指標（マルチタイムフレーム完全対応版） ==========
    
    # ========== 3. トレンド系指標 ==========

    # --- 11/57: Short-Term Moving Averages ---
    def _moving_averages_short_logic(self, ohlcv: Dict[str, pd.Series], timeframe: str) -> Dict[str, np.ndarray]:
        """短期移動平均の具体的な計算ロジック"""
        features = {}
        close_series = ohlcv['close']
        suffix = f'_{timeframe}' if self.multi_timeframe else ''
        
        for period in self.short_ma_periods:
            # Simple Moving Average
            sma = close_series.rolling(window=period).mean()
            features[f'sma_{period}{suffix}'] = sma.values
            features[f'price_above_sma_{period}{suffix}'] = (close_series > sma).astype(int).values
            sma_slope = sma.diff()
            features[f'sma_slope_{period}{suffix}'] = sma_slope.values
            
            # Exponential Moving Average
            ema = close_series.ewm(span=period).mean()
            features[f'ema_{period}{suffix}'] = ema.values
            features[f'price_above_ema_{period}{suffix}'] = (close_series > ema).astype(int).values
            ema_slope = ema.diff()
            features[f'ema_slope_{period}{suffix}'] = ema_slope.values
            
            # Weighted Moving Average
            wma = self._weighted_ma_vectorized(close_series, period)
            features[f'wma_{period}{suffix}'] = wma.values
        return features

    def _calculate_moving_averages_short(self, df: pd.DataFrame) -> Dict[str, np.ndarray]:
        """短期移動平均系指標（共通処理ヘルパー使用）"""
        logic = partial(self._moving_averages_short_logic)
        return self._process_feature_group(df, logic, "Short-Term Moving Averages")

    # --- 12/57: Long-Term Moving Averages ---
    def _moving_averages_long_logic(self, ohlcv: Dict[str, pd.Series], timeframe: str) -> Dict[str, np.ndarray]:
        """長期移動平均の具体的な計算ロジック"""
        features = {}
        close_series = ohlcv['close']
        suffix = f'_{timeframe}' if self.multi_timeframe else ''

        for period in self.long_ma_periods:
            # Simple Moving Average
            sma = close_series.rolling(window=period).mean()
            features[f'sma_{period}{suffix}'] = sma.values
            features[f'price_above_sma_{period}{suffix}'] = (close_series > sma).astype(int).values
            sma_slope = sma.diff()
            features[f'sma_slope_{period}{suffix}'] = sma_slope.values
            
            # Exponential Moving Average
            ema = close_series.ewm(span=period).mean()
            features[f'ema_{period}{suffix}'] = ema.values
            features[f'price_above_ema_{period}{suffix}'] = (close_series > ema).astype(int).values
            ema_slope = ema.diff()
            features[f'ema_slope_{period}{suffix}'] = ema_slope.values
            
            # Hull Moving Average
            hma = self._hull_ma_vectorized(close_series, period)
            features[f'hma_{period}{suffix}'] = hma.values
        
        # 移動平均の乖離率
        for period in self.ma_deviation_periods:
            sma = close_series.rolling(window=period).mean()
            features[f'deviation_from_sma_{period}{suffix}'] = (((close_series - sma) / sma) * 100).values
        return features

    def _calculate_moving_averages_long(self, df: pd.DataFrame) -> Dict[str, np.ndarray]:
        """長期移動平均系指標（共通処理ヘルパー使用）"""
        logic = partial(self._moving_averages_long_logic)
        return self._process_feature_group(df, logic, "Long-Term Moving Averages")

    # --- 13/57: Advanced Moving Averages ---
    def _moving_averages_advanced_logic(self, ohlcv: Dict[str, pd.Series], timeframe: str) -> Dict[str, np.ndarray]:
        """高度移動平均の具体的な計算ロジック"""
        features = {}
        close_series = ohlcv['close']
        suffix = f'_{timeframe}' if self.multi_timeframe else ''

        # Triangular Moving Average
        for period in self.tma_periods:
            tma = self._triangular_ma_vectorized(close_series, period)
            features[f'tma_{period}{suffix}'] = tma.values
        
        # Kaufman's Adaptive Moving Average (KAMA)
        kama = self._compute_kama_vectorized(close_series)
        features[f'kama{suffix}'] = kama.values
        kama_slope = kama.diff()
        features[f'kama_slope{suffix}'] = kama_slope.values
        
        # Zero Lag EMA
        for period in self.zlema_periods:
            zlema = self._compute_zlema_vectorized(close_series, period)
            features[f'zlema_{period}{suffix}'] = zlema.values
        
        # Double Exponential Moving Average (DEMA)
        for period in self.dema_periods:
            dema = self._compute_dema_vectorized(close_series, period)
            features[f'dema_{period}{suffix}'] = dema.values
        
        # Triple Exponential Moving Average (TEMA)
        for period in self.tema_periods:
            tema = self._compute_tema_vectorized(close_series, period)
            features[f'tema_{period}{suffix}'] = tema.values
        
        # Variable Moving Average (VMA)
        vma = self._compute_vma_vectorized(close_series)
        features[f'vma{suffix}'] = vma.values
        return features

    def _calculate_moving_averages_advanced(self, df: pd.DataFrame) -> Dict[str, np.ndarray]:
        """高度移動平均系指標（共通処理ヘルパー使用）"""
        logic = partial(self._moving_averages_advanced_logic)
        return self._process_feature_group(df, logic, "Advanced Moving Averages")

    # --- 14/57: Trend Crosses ---
    def _trend_crosses_logic(self, ohlcv: Dict[str, pd.Series], timeframe: str) -> Dict[str, np.ndarray]:
        """トレンドクロスの具体的な計算ロジック"""
        features = {}
        close_series = ohlcv['close']
        suffix = f'_{timeframe}' if self.multi_timeframe else ''

        # 必要な移動平均を計算
        sma_20 = close_series.rolling(window=20).mean()
        sma_50 = close_series.rolling(window=50).mean()
        sma_200 = close_series.rolling(window=200).mean()
        
        # ゴールデンクロス・デッドクロス
        features[f'golden_cross_50_200{suffix}'] = self._detect_ma_cross_vectorized(sma_50, sma_200).values
        features[f'death_cross_50_200{suffix}'] = self._detect_ma_cross_vectorized(sma_200, sma_50).values
        features[f'golden_cross_20_50{suffix}'] = self._detect_ma_cross_vectorized(sma_20, sma_50).values
        return features

    def _calculate_trend_crosses(self, df: pd.DataFrame) -> Dict[str, np.ndarray]:
        """トレンドクロス系指標（共通処理ヘルパー使用）"""
        logic = partial(self._trend_crosses_logic)
        return self._process_feature_group(df, logic, "Trend Crosses")

    # ========== 4. ボラティリティ系指標 ==========

    # --- 15/57: Volatility Bands ---
    def _volatility_bands_logic(self, ohlcv: Dict[str, pd.Series], timeframe: str) -> Dict[str, np.ndarray]:
        """ボラティリティバンドの具体的な計算ロジック"""
        features = {}
        close_series = ohlcv['close']
        suffix = f'_{timeframe}' if self.multi_timeframe else ''

        for period, std_mult in self.volatility_bb_settings:
            bb_upper, bb_middle, bb_lower = self._compute_bollinger_bands_vectorized(close_series, period, std_mult)
            
            features[f'bb_upper_{period}_{int(std_mult*10)}{suffix}'] = bb_upper.values
            features[f'bb_lower_{period}_{int(std_mult*10)}{suffix}'] = bb_lower.values
            features[f'bb_width_{period}_{int(std_mult*10)}{suffix}'] = (bb_upper - bb_lower).values
            features[f'bb_percent_{period}_{int(std_mult*10)}{suffix}'] = ((close_series - bb_lower) / (bb_upper - bb_lower + 1e-8)).values
            
            bb_width = bb_upper - bb_lower
            bb_width_shifted = bb_width.shift(20)
            features[f'bb_squeeze_{period}_{int(std_mult*10)}{suffix}'] = (bb_width < bb_width_shifted).astype(int).values
        return features

    def _calculate_volatility_bands(self, df: pd.DataFrame) -> Dict[str, np.ndarray]:
        """ボラティリティバンド系指標（共通処理ヘルパー使用）"""
        logic = partial(self._volatility_bands_logic)
        return self._process_feature_group(df, logic, "Volatility Bands")

    # --- 16/57: Volatility Channels ---
    def _volatility_channels_logic(self, ohlcv: Dict[str, pd.Series], timeframe: str) -> Dict[str, np.ndarray]:
        """ボラティリティチャンネルの具体的な計算ロジック"""
        features = {}
        close_series = ohlcv['close']
        high_series = ohlcv['high']
        low_series = ohlcv['low']
        suffix = f'_{timeframe}' if self.multi_timeframe else ''

        # Keltner Channels
        for period in self.kc_periods:
            kc_upper, kc_middle, kc_lower = self._compute_keltner_channels_vectorized(high_series, low_series, close_series, period)
            features[f'kc_upper_{period}{suffix}'] = kc_upper.values
            features[f'kc_lower_{period}{suffix}'] = kc_lower.values
            features[f'kc_percent_{period}{suffix}'] = ((close_series - kc_lower) / (kc_upper - kc_lower + 1e-8)).values
        
        # Donchian Channels
        for period in self.dc_periods:
            dc_upper = high_series.rolling(window=period).max()
            dc_lower = low_series.rolling(window=period).min()
            features[f'dc_upper_{period}{suffix}'] = dc_upper.values
            features[f'dc_lower_{period}{suffix}'] = dc_lower.values
            features[f'dc_middle_{period}{suffix}'] = ((dc_upper + dc_lower) / 2).values
            features[f'dc_percent_{period}{suffix}'] = ((close_series - dc_lower) / (dc_upper - dc_lower + 1e-8)).values
        return features

    def _calculate_volatility_channels(self, df: pd.DataFrame) -> Dict[str, np.ndarray]:
        """ボラティリティチャンネル系指標（共通処理ヘルパー使用）"""
        logic = partial(self._volatility_channels_logic)
        return self._process_feature_group(df, logic, "Volatility Channels")

    # --- 17/57: Volatility Measures ---
    def _volatility_measures_logic(self, ohlcv: Dict[str, pd.Series], timeframe: str) -> Dict[str, np.ndarray]:
        """ボラティリティ測定の具体的な計算ロジック"""
        features = {}
        close_series = ohlcv['close']
        high_series = ohlcv['high']
        low_series = ohlcv['low']
        suffix = f'_{timeframe}' if self.multi_timeframe else ''

        # Average True Range variants
        for period in self.atr_periods_vol: # atr_periodsと重複しないように名前を変更
            atr = self._compute_atr_vectorized(high_series, low_series, close_series, period)
            features[f'atr_bands_upper_{period}{suffix}'] = (close_series + (2 * atr)).values
            features[f'atr_bands_lower_{period}{suffix}'] = (close_series - (2 * atr)).values
        
        # Historical Volatility
        for period in self.hist_vol_periods:
            log_returns = np.log(close_series / close_series.shift(1))
            hist_vol = log_returns.rolling(window=period).std() * np.sqrt(252)
            features[f'historical_vol_{period}{suffix}'] = hist_vol.values
        
        # Volatility Ratio
        # NOTE: この特徴量は同じ関数内で計算された他の特徴量に依存しているため、キーをサフィックス付きで指定する必要があります
        hist_vol_20_key = f'historical_vol_20{suffix}'
        hist_vol_60_key = f'historical_vol_60{suffix}'
        if hist_vol_20_key in features and hist_vol_60_key in features:
            hist_vol_20 = features[hist_vol_20_key]
            hist_vol_60 = features[hist_vol_60_key]
            features[f'volatility_ratio{suffix}'] = hist_vol_20 / (hist_vol_60 + 1e-8)
        
        # Chandelier Exit
        chandelier_long, chandelier_short = self._compute_chandelier_exit_vectorized(high_series, low_series, close_series)
        features[f'chandelier_long{suffix}'] = chandelier_long.values
        features[f'chandelier_short{suffix}'] = chandelier_short.values
        return features

    def _calculate_volatility_measures(self, df: pd.DataFrame) -> Dict[str, np.ndarray]:
        """ボラティリティ測定系指標（共通処理ヘルパー使用）"""
        logic = partial(self._volatility_measures_logic)
        return self._process_feature_group(df, logic, "Volatility Measures")

    # ========== 5. サポート・レジスタンス系指標 ==========

    # --- 18/57: Support / Resistance ---
    def _support_resistance_logic(self, ohlcv: Dict[str, pd.Series], timeframe: str) -> Dict[str, np.ndarray]:
        """サポート・レジスタンスの具体的な計算ロジック"""
        features = {}
        close_series = ohlcv['close']
        high_series = ohlcv['high']
        low_series = ohlcv['low']
        suffix = f'_{timeframe}' if self.multi_timeframe else ''

        # Pivot Points
        pivot_classic = self._compute_pivot_points_classic_vectorized(high_series, low_series, close_series)
        for key, value in pivot_classic.items():
            features[f'{key}{suffix}'] = value
        
        pivot_fibonacci = self._compute_pivot_points_fibonacci_vectorized(high_series, low_series, close_series)
        for key, value in pivot_fibonacci.items():
            features[f'{key}{suffix}'] = value
        
        # Price Channels
        for period in self.price_channel_periods:
            price_channel_high = high_series.rolling(window=period).max()
            price_channel_low = low_series.rolling(window=period).min()
            features[f'price_channel_high_{period}{suffix}'] = price_channel_high.values
            features[f'price_channel_low_{period}{suffix}'] = price_channel_low.values
            features[f'price_channel_middle_{period}{suffix}'] = ((price_channel_high + price_channel_low) / 2).values
            features[f'price_channel_position_{period}{suffix}'] = ((close_series - price_channel_low) / (price_channel_high - price_channel_low + 1e-8)).values
        
        # Fibonacci Retracements
        fib_levels = self._compute_fibonacci_retracements_vectorized(high_series, low_series, close_series)
        for key, value in fib_levels.items():
            features[f'{key}{suffix}'] = value
        
        # Support/Resistance Levels
        support_levels, resistance_levels = self._identify_support_resistance_vectorized(high_series, low_series, close_series)
        features[f'nearest_support{suffix}'] = support_levels.values
        features[f'nearest_resistance{suffix}'] = resistance_levels.values
        features[f'support_distance{suffix}'] = ((close_series - support_levels) / close_series).values
        features[f'resistance_distance{suffix}'] = ((resistance_levels - close_series) / close_series).values
        return features

    def _calculate_support_resistance(self, df: pd.DataFrame) -> Dict[str, np.ndarray]:
        """サポート・レジスタンス系指標（共通処理ヘルパー使用）"""
        logic = partial(self._support_resistance_logic)
        return self._process_feature_group(df, logic, "Support/Resistance")

    # ========== 6. ローソク足パターン系指標 ==========

    # --- 19/57: Candlestick Patterns ---
    def _candlestick_patterns_logic(self, ohlcv: Dict[str, pd.Series], timeframe: str) -> Dict[str, np.ndarray]:
        """ローソク足パターンの具体的な計算ロジック"""
        features = {}
        open_series = ohlcv['open']
        high_series = ohlcv['high']
        low_series = ohlcv['low']
        close_series = ohlcv['close']
        suffix = f'_{timeframe}' if self.multi_timeframe else ''

        features[f'body_size{suffix}'] = np.abs(close_series - open_series).values
        features[f'upper_shadow{suffix}'] = (high_series - pd.concat([open_series, close_series], axis=1).max(axis=1)).values
        features[f'lower_shadow{suffix}'] = (pd.concat([open_series, close_series], axis=1).min(axis=1) - low_series).values
        features[f'total_range{suffix}'] = (high_series - low_series).values
        
        total_range_safe = features[f'total_range{suffix}'] + 1e-8
        features[f'body_ratio{suffix}'] = features[f'body_size{suffix}'] / total_range_safe
        features[f'upper_shadow_ratio{suffix}'] = features[f'upper_shadow{suffix}'] / total_range_safe
        features[f'lower_shadow_ratio{suffix}'] = features[f'lower_shadow{suffix}'] / total_range_safe
        
        features[f'bullish_candle{suffix}'] = (close_series > open_series).astype(int).values
        features[f'bearish_candle{suffix}'] = (close_series < open_series).astype(int).values
        features[f'doji{suffix}'] = (np.abs(close_series - open_series) < 0.001 * close_series).astype(int).values
        
        features[f'hammer{suffix}'] = self._detect_hammer_vectorized(open_series, high_series, low_series, close_series).values
        features[f'shooting_star{suffix}'] = self._detect_shooting_star_vectorized(open_series, high_series, low_series, close_series).values
        features[f'spinning_top{suffix}'] = self._detect_spinning_top_vectorized(open_series, high_series, low_series, close_series).values
        features[f'marubozu{suffix}'] = self._detect_marubozu_vectorized(open_series, high_series, low_series, close_series).values
        
        features[f'engulfing_bullish{suffix}'] = self._detect_engulfing_bullish_vectorized(open_series, high_series, low_series, close_series).values
        features[f'engulfing_bearish{suffix}'] = self._detect_engulfing_bearish_vectorized(open_series, high_series, low_series, close_series).values
        features[f'harami{suffix}'] = self._detect_harami_vectorized(open_series, high_series, low_series, close_series).values
        
        features[f'gap_up{suffix}'] = self._detect_gap_up_vectorized(high_series, low_series).values
        features[f'gap_down{suffix}'] = self._detect_gap_down_vectorized(high_series, low_series).values
        return features

    def _calculate_candlestick_patterns(self, df: pd.DataFrame) -> Dict[str, np.ndarray]:
        """日本のローソク足パターン（共通処理ヘルパー使用）"""
        logic = partial(self._candlestick_patterns_logic)
        return self._process_feature_group(df, logic, "Candlestick Patterns")

    # ========== 7. 数学・統計学 ==========

    # --- 20/57: Basic Statistical Moments ---
    def _statistical_moments_basic_logic(self, ohlcv: Dict[str, pd.Series], timeframe: str) -> Dict[str, np.ndarray]:
        """基本統計モーメントの具体的な計算ロジック"""
        features = {}
        close_series = ohlcv['close']
        returns = np.log(close_series / close_series.shift(1)).fillna(0)
        suffix = f'_{timeframe}' if self.multi_timeframe else ''

        for window in self.stat_windows:
            features[f'mean_{window}{suffix}'] = returns.rolling(window=window).mean().values
            features[f'variance_{window}{suffix}'] = returns.rolling(window=window).var().values
            features[f'skewness_{window}{suffix}'] = returns.rolling(window=window).skew().values
            features[f'kurtosis_{window}{suffix}'] = returns.rolling(window=window).kurt().values
        return features

    def _calculate_statistical_moments_basic(self, df: pd.DataFrame) -> Dict[str, np.ndarray]:
        """基本統計的モーメント（共通処理ヘルパー使用）"""
        logic = partial(self._statistical_moments_basic_logic)
        return self._process_feature_group(df, logic, "Basic Statistical Moments")

    # --- 21/57: Advanced Statistical Moments ---
    def _statistical_moments_advanced_logic(self, ohlcv: Dict[str, pd.Series], timeframe: str) -> Dict[str, np.ndarray]:
        """高次統計モーメントの具体的な計算ロジック"""
        features = {}
        close_series = ohlcv['close']
        returns = np.log(close_series / close_series.shift(1)).fillna(0)
        suffix = f'_{timeframe}' if self.multi_timeframe else ''

        for window in self.stat_windows:
            for moment in range(5, 9):
                func = partial(self._numba_raw_moment, moment=moment) # Numba化された関数を使用
                features[f'moment_{moment}_{window}{suffix}'] = returns.rolling(window=window).apply(func, raw=True).values
            
            features[f'central_moment_3_{window}{suffix}'] = returns.rolling(window=window).apply(lambda x: np.mean((x - np.mean(x))**3), raw=True).values
            features[f'central_moment_4_{window}{suffix}'] = returns.rolling(window=window).apply(lambda x: np.mean((x - np.mean(x))**4), raw=True).values
        return features

    def _calculate_statistical_moments_advanced(self, df: pd.DataFrame) -> Dict[str, np.ndarray]:
        """高次統計的モーメント（共通処理ヘルパー使用）"""
        logic = partial(self._statistical_moments_advanced_logic)
        return self._process_feature_group(df, logic, "Advanced Statistical Moments")

    # --- 22/57: Probability Distributions ---
    def _probability_distributions_logic(self, ohlcv: Dict[str, pd.Series], timeframe: str) -> Dict[str, np.ndarray]:
        """確率分布フィッティングの具体的な計算ロジック"""
        features = {}
        close_series = ohlcv['close']
        returns = np.log(close_series / close_series.shift(1)).fillna(0)
        suffix = f'_{timeframe}' if self.multi_timeframe else ''

        for window in self.dist_windows:
            features[f'normal_mu_{window}{suffix}'] = returns.rolling(window=window).mean().values
            features[f'normal_sigma_{window}{suffix}'] = returns.rolling(window=window).std().values
            
            features[f't_dist_df_{window}{suffix}'] = returns.rolling(window=window).apply(self._estimate_t_df_vectorized, raw=True).values
            
            abs_returns = np.abs(returns)
            features[f'gamma_shape_{window}{suffix}'] = abs_returns.rolling(window=window).apply(self._estimate_gamma_shape_vectorized, raw=True).values
            
            norm_returns = (returns - returns.min()) / (returns.max() - returns.min() + 1e-8)
            features[f'beta_alpha_{window}{suffix}'] = norm_returns.rolling(window=window).apply(self._estimate_beta_alpha_vectorized, raw=True).values
        return features

    def _calculate_probability_distributions(self, df: pd.DataFrame) -> Dict[str, np.ndarray]:
        """確率分布フィッティング（共通処理ヘルパー使用）"""
        logic = partial(self._probability_distributions_logic)
        return self._process_feature_group(df, logic, "Probability Distributions")

    # --- 23/57: Robust Statistics ---
    def _robust_statistics_logic(self, ohlcv: Dict[str, pd.Series], timeframe: str) -> Dict[str, np.ndarray]:
        """ロバスト統計量の具体的な計算ロジック"""
        features = {}
        close_series = ohlcv['close']
        returns = np.log(close_series / close_series.shift(1)).fillna(0)
        suffix = f'_{timeframe}' if self.multi_timeframe else ''

        for window in self.robust_stat_windows:
            features[f'median_{window}{suffix}'] = returns.rolling(window=window).median().values
            features[f'mad_{window}{suffix}'] = returns.rolling(window=window).apply(self._median_abs_deviation_vectorized, raw=True).values
            
            q25 = returns.rolling(window=window).quantile(0.25)
            q75 = returns.rolling(window=window).quantile(0.75)
            features[f'q25_{window}{suffix}'] = q25.values
            features[f'q75_{window}{suffix}'] = q75.values
            features[f'iqr_{window}{suffix}'] = (q75 - q25).values
            
            features[f'trim_mean_10_{window}{suffix}'] = returns.rolling(window=window).apply(lambda x: stats.trim_mean(x, 0.1), raw=True).values
            features[f'trim_mean_20_{window}{suffix}'] = returns.rolling(window=window).apply(lambda x: stats.trim_mean(x, 0.2), raw=True).values
            
            features[f'winsor_mean_{window}{suffix}'] = returns.rolling(window=window).apply(self._winsorized_mean_vectorized, raw=True).values
        return features

    def _calculate_robust_statistics(self, df: pd.DataFrame) -> Dict[str, np.ndarray]:
        """ロバスト統計量（共通処理ヘルパー使用）"""
        logic = partial(self._robust_statistics_logic)
        return self._process_feature_group(df, logic, "Robust Statistics")

    # --- 24/57: Order Statistics ---
    def _order_statistics_logic(self, ohlcv: Dict[str, pd.Series], timeframe: str) -> Dict[str, np.ndarray]:
        """順序統計量の具体的な計算ロジック"""
        features = {}
        close_series = ohlcv['close']
        suffix = f'_{timeframe}' if self.multi_timeframe else ''

        for window in self.order_stat_windows:
            features[f'price_rank_{window}{suffix}'] = close_series.rolling(window=window).apply(lambda x: (x.iloc[-1] - x.min()) / (x.max() - x.min() + 1e-8), raw=False).values
            features[f'quantile_pos_{window}{suffix}'] = close_series.rolling(window=window).apply(lambda x: stats.percentileofscore(x, x.iloc[-1]) / 100, raw=False).values
            features[f'extreme_ratio_{window}{suffix}'] = close_series.rolling(window=window).apply(self._extreme_value_ratio_vectorized, raw=False).values
        return features

    def _calculate_order_statistics(self, df: pd.DataFrame) -> Dict[str, np.ndarray]:
        """順序統計量（共通処理ヘルパー使用）"""
        logic = partial(self._order_statistics_logic)
        return self._process_feature_group(df, logic, "Order Statistics")

    # ========== 8. 信号処理系 ==========

    # --- 25/57: Basic Signal Processing ---
    def _signal_processing_basic_logic(self, ohlcv: Dict[str, pd.Series], timeframe: str) -> Dict[str, np.ndarray]:
        """基本信号処理の具体的な計算ロジック"""
        features = {}
        close_series = ohlcv['close']
        volume_series = ohlcv['volume']
        suffix = f'_{timeframe}' if self.multi_timeframe else ''

        for window in self.hilbert_windows:
            features[f'hilbert_amplitude_{window}{suffix}'] = close_series.rolling(window=window).apply(self._hilbert_amplitude_vectorized, raw=True).values
            features[f'hilbert_phase_{window}{suffix}'] = close_series.rolling(window=window).apply(self._hilbert_phase_vectorized, raw=True).values
            features[f'instantaneous_freq_{window}{suffix}'] = close_series.rolling(window=window).apply(self._instantaneous_frequency_vectorized, raw=True).values
        
        for lag in self.autocorr_lags:
            features[f'autocorr_lag_{lag}{suffix}'] = close_series.rolling(window=50).apply(lambda x: self._autocorrelation_vectorized(x, lag), raw=False).values
        
        features[f'cross_correlation{suffix}'] = close_series.rolling(window=50).apply(lambda x: self._cross_correlation_vectorized(x, volume_series.loc[x.index]), raw=False).values
        return features

    def _calculate_signal_processing_basic(self, df: pd.DataFrame) -> Dict[str, np.ndarray]:
        """基本信号処理特徴量（共通処理ヘルパー使用）"""
        logic = partial(self._signal_processing_basic_logic)
        return self._process_feature_group(df, logic, "Basic Signal Processing")

    # --- 26/57: Advanced Signal Processing ---
    def _signal_processing_advanced_logic(self, ohlcv: Dict[str, pd.Series], timeframe: str) -> Dict[str, np.ndarray]:
        """高度信号処理の具体的な計算ロジック"""
        features = {}
        close_series = ohlcv['close']
        volume_series = ohlcv['volume']
        suffix = f'_{timeframe}' if self.multi_timeframe else ''

        for window in self.spectral_windows:
            features[f'spectral_centroid_{window}{suffix}'] = close_series.rolling(window=window).apply(self._spectral_centroid_vectorized, raw=True).values
            features[f'spectral_bandwidth_{window}{suffix}'] = close_series.rolling(window=window).apply(self._spectral_bandwidth_vectorized, raw=True).values
            features[f'spectral_rolloff_{window}{suffix}'] = close_series.rolling(window=window).apply(self._spectral_rolloff_vectorized, raw=True).values
            features[f'spectral_flux_{window}{suffix}'] = close_series.rolling(window=window).apply(self._spectral_flux_vectorized, raw=True).values
        
        features[f'zero_crossing_rate{suffix}'] = close_series.rolling(window=50).apply(self._zero_crossing_rate_vectorized, raw=True).values
        features[f'coherence_measure{suffix}'] = close_series.rolling(window=100).apply(lambda x: self._coherence_measure_vectorized(x, volume_series.loc[x.index]), raw=False).values
        return features

    def _calculate_signal_processing_advanced(self, df: pd.DataFrame) -> Dict[str, np.ndarray]:
        """高度信号処理特徴量（共通処理ヘルパー使用）"""
        logic = partial(self._signal_processing_advanced_logic)
        return self._process_feature_group(df, logic, "Advanced Signal Processing")
    
   # ========== 出来高系・トレンド系・ボラティリティ系ヘルパー関数 ==========

    @staticmethod
    @njit(cache=True)
    def _numba_weighted_ma(x: np.ndarray, weights: np.ndarray) -> float:
        """Numba JIT化: 加重移動平均"""
        return np.dot(x, weights) / weights.sum()

    @staticmethod
    @njit(cache=True)
    def _numba_kama(series_values: np.ndarray) -> np.ndarray:
        """Numba JIT化: Kaufman's Adaptive Moving Average"""
        period = 10
        fast_sc = 2 / (2 + 1)
        slow_sc = 2 / (30 + 1)
        
        # change
        change = np.abs(series_values[period:] - series_values[:-period])

        # volatility
        volatility = np.zeros_like(change)
        diffs = np.abs(np.diff(series_values))
        for i in range(len(volatility)):
            volatility[i] = np.sum(diffs[i:i+period])

        # efficiency_ratio
        efficiency_ratio = change / (volatility + 1e-8)
        
        # sc
        sc = (efficiency_ratio * (fast_sc - slow_sc) + slow_sc) ** 2
        
        kama = np.zeros_like(series_values)
        kama[:period] = series_values[:period] # 初期値
        
        for i in range(period, len(series_values)):
            # scのインデックスを調整
            current_sc = sc[i-period]
            kama[i] = kama[i-1] + current_sc * (series_values[i] - kama[i-1])
            
        return kama

    def _compute_obv_vectorized(self, close_series: pd.Series, volume_series: pd.Series) -> pd.Series:
        """On Balance Volumeベクトル化計算"""
        price_change = close_series.diff()
        volume_direction = np.where(price_change > 0, volume_series, 
                                  np.where(price_change < 0, -volume_series, 0))
        obv = pd.Series(volume_direction).cumsum()
        return obv.fillna(0)
    
    def _compute_vpt_vectorized(self, close_series: pd.Series, volume_series: pd.Series) -> pd.Series:
        """Volume Price Trendベクトル化計算"""
        price_change_pct = close_series.pct_change()
        vpt = (price_change_pct * volume_series).cumsum()
        return vpt.fillna(0)
    
    def _compute_ad_line_vectorized(self, high_series: pd.Series, low_series: pd.Series, close_series: pd.Series, volume_series: pd.Series) -> pd.Series:
        """Accumulation/Distribution Lineベクトル化計算"""
        clv = ((close_series - low_series) - (high_series - close_series)) / (high_series - low_series + 1e-8)
        ad_line = (clv * volume_series).cumsum()
        return ad_line.fillna(0)
    
    def _compute_chaikin_money_flow_vectorized(self, high_series: pd.Series, low_series: pd.Series, close_series: pd.Series, volume_series: pd.Series, period: int) -> pd.Series:
        """Chaikin Money Flowベクトル化計算"""
        clv = ((close_series - low_series) - (high_series - close_series)) / (high_series - low_series + 1e-8)
        money_flow_volume = clv * volume_series
        cmf = money_flow_volume.rolling(window=period).sum() / volume_series.rolling(window=period).sum()
        return cmf.fillna(0)
    
    def _compute_chaikin_oscillator_vectorized(self, high_series: pd.Series, low_series: pd.Series, close_series: pd.Series, volume_series: pd.Series) -> pd.Series:
        """Chaikin A/D Oscillatorベクトル化計算"""
        ad_line = self._compute_ad_line_vectorized(high_series, low_series, close_series, volume_series)
        ema_3 = ad_line.ewm(span=3).mean()
        ema_10 = ad_line.ewm(span=10).mean()
        return ema_3 - ema_10
    
    def _compute_money_flow_index_vectorized(self, high_series: pd.Series, low_series: pd.Series, close_series: pd.Series, volume_series: pd.Series, period: int) -> pd.Series:
        """Money Flow Indexベクトル化計算"""
        typical_price = (high_series + low_series + close_series) / 3
        raw_money_flow = typical_price * volume_series
        
        positive_flow = raw_money_flow.where(typical_price > typical_price.shift(1), 0)
        negative_flow = raw_money_flow.where(typical_price < typical_price.shift(1), 0)
        
        positive_mf = positive_flow.rolling(window=period).sum()
        negative_mf = negative_flow.rolling(window=period).sum()
        
        mfi = 100 - (100 / (1 + positive_mf / (negative_mf + 1e-8)))
        return mfi.fillna(0)
    
    def _compute_vwap_vectorized(self, high_series: pd.Series, low_series: pd.Series, close_series: pd.Series, volume_series: pd.Series) -> pd.Series:
        """VWAPベクトル化計算"""
        typical_price = (high_series + low_series + close_series) / 3
        vwap = (typical_price * volume_series).cumsum() / volume_series.cumsum()
        return vwap.fillna(close_series)
    
    def _compute_ease_of_movement_vectorized(self, high_series: pd.Series, low_series: pd.Series, volume_series: pd.Series) -> pd.Series:
        """Ease of Movementベクトル化計算"""
        distance_moved = ((high_series + low_series) / 2) - ((high_series.shift(1) + low_series.shift(1)) / 2)
        box_height = (volume_series / 1000000) / (high_series - low_series + 1e-8)
        eom = distance_moved / box_height
        return eom.fillna(0)
    
    def _compute_volume_oscillator_vectorized(self, volume_series: pd.Series) -> pd.Series:
        """Volume Oscillatorベクトル化計算"""
        short_vol_ma = volume_series.rolling(window=14).mean()
        long_vol_ma = volume_series.rolling(window=28).mean()
        volume_osc = ((short_vol_ma - long_vol_ma) / long_vol_ma) * 100
        return volume_osc.fillna(0)
    
    def _compute_pvt_oscillator_vectorized(self, close_series: pd.Series, volume_series: pd.Series) -> pd.Series:
        """Price Volume Trend Oscillatorベクトル化計算"""
        pvt = self._compute_vpt_vectorized(close_series, volume_series)
        pvt_ma = pvt.rolling(window=20).mean()
        pvt_osc = pvt - pvt_ma
        return pvt_osc.fillna(0)
    
    def _detect_ma_cross_vectorized(self, fast_ma: pd.Series, slow_ma: pd.Series) -> pd.Series:
        """移動平均クロス検出"""
        cross_up = ((fast_ma > slow_ma) & (fast_ma.shift(1) <= slow_ma.shift(1))).astype(int)
        return cross_up
    
    def _weighted_ma_vectorized(self, series: pd.Series, period: int) -> pd.Series:
        """加重移動平均ベクトル化計算 (Numba高速化適用)"""
        weights = np.arange(1, period + 1)
        # Numbaヘルパーを呼び出すように変更
        func = partial(self._numba_weighted_ma, weights=weights)
        wma = series.rolling(window=period).apply(func, raw=True)
        return wma.fillna(series)
    
    def _hull_ma_vectorized(self, series: pd.Series, period: int) -> pd.Series:
        """Hull Moving Averageベクトル化計算"""
        try:
            half_period = int(period / 2)
            sqrt_period = int(np.sqrt(period))
            
            wma_half = self._weighted_ma_vectorized(series, half_period)
            wma_full = self._weighted_ma_vectorized(series, period)
            
            raw_hma = 2 * wma_half - wma_full
            hma = self._weighted_ma_vectorized(raw_hma, sqrt_period)
            
            return hma.fillna(series)
        except:
            return series
    
    def _triangular_ma_vectorized(self, series: pd.Series, period: int) -> pd.Series:
        """Triangular Moving Averageベクトル化計算"""
        try:
            first_sma = series.rolling(window=period).mean()
            tma = first_sma.rolling(window=period).mean()
            return tma.fillna(series)
        except:
            return series
    
    def _compute_kama_vectorized(self, series: pd.Series) -> pd.Series:
        """Kaufman's Adaptive Moving Averageベクトル化計算 (Numba高速化適用)"""
        try:
            # Numbaヘルパーを呼び出すように変更
            kama_values = self._numba_kama(series.values)
            return pd.Series(kama_values, index=series.index)
        except:
            return series
    
    def _compute_zlema_vectorized(self, series: pd.Series, period: int) -> pd.Series:
        """Zero Lag EMAベクトル化計算"""
        try:
            lag = int((period - 1) / 2)
            ema_data = 2 * series - series.shift(lag)
            zlema = ema_data.ewm(span=period).mean()
            return zlema.fillna(series)
        except:
            return series
    
    def _compute_dema_vectorized(self, series: pd.Series, period: int) -> pd.Series:
        """Double Exponential Moving Averageベクトル化計算"""
        try:
            ema1 = series.ewm(span=period).mean()
            ema2 = ema1.ewm(span=period).mean()
            dema = 2 * ema1 - ema2
            return dema.fillna(series)
        except:
            return series
    
    def _compute_tema_vectorized(self, series: pd.Series, period: int) -> pd.Series:
        """Triple Exponential Moving Averageベクトル化計算"""
        try:
            ema1 = series.ewm(span=period).mean()
            ema2 = ema1.ewm(span=period).mean()
            ema3 = ema2.ewm(span=period).mean()
            tema = 3 * ema1 - 3 * ema2 + ema3
            return tema.fillna(series)
        except:
            return series
    
    def _compute_vma_vectorized(self, series: pd.Series) -> pd.Series:
        """Variable Moving Averageベクトル化計算（簡易版）"""
        try:
            volatility = series.rolling(window=9).std()
            normalized_vol = (volatility - volatility.rolling(window=30).min()) / (volatility.rolling(window=30).max() - volatility.rolling(window=30).min() + 1e-8)
            period = 2 + 28 * normalized_vol
            
            vma = series.ewm(span=period.fillna(15)).mean()
            return vma.fillna(series)
        except:
            return series
    
    def _compute_keltner_channels_vectorized(self, high_series: pd.Series, low_series: pd.Series, close_series: pd.Series, period: int) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """Keltner Channelsベクトル化計算"""
        try:
            middle = close_series.ewm(span=period).mean()
            atr = self._compute_atr_vectorized(high_series, low_series, close_series, period)
            multiplier = 2
            
            upper = middle + (multiplier * atr)
            lower = middle - (multiplier * atr)
            
            return upper.fillna(close_series), middle.fillna(close_series), lower.fillna(close_series)
        except:
            return close_series, close_series, close_series
    
    def _compute_chandelier_exit_vectorized(self, high_series: pd.Series, low_series: pd.Series, close_series: pd.Series) -> Tuple[pd.Series, pd.Series]:
        """Chandelier Exitベクトル化計算"""
        try:
            period = 22
            multiplier = 3
            
            atr = self._compute_atr_vectorized(high_series, low_series, close_series, period)
            highest_high = high_series.rolling(window=period).max()
            lowest_low = low_series.rolling(window=period).min()
            
            chandelier_long = highest_high - (multiplier * atr)
            chandelier_short = lowest_low + (multiplier * atr)
            
            return chandelier_long.fillna(close_series), chandelier_short.fillna(close_series)
        except:
            return close_series, close_series
        
# ========== サポート・レジスタンス系指標（マルチタイムフレーム完全対応版） ==========
    
    def _calculate_support_resistance(self, df: pd.DataFrame) -> Dict[str, np.ndarray]:
        """サポート・レジスタンス系指標（マルチタイムフレーム対応版）"""
        features = {}
        
        for tf in self.available_timeframes if self.multi_timeframe else ['1T']:
            try:
                ohlcv = self._extract_ohlcv_for_timeframe(df, tf)
                close_series = ohlcv['close']
                high_series = ohlcv['high']
                low_series = ohlcv['low']
                suffix = f'_{tf}' if self.multi_timeframe else ''
                
                # Pivot Points (複数種類)
                pivot_classic = self._compute_pivot_points_classic_vectorized(high_series, low_series, close_series)
                for key, value in pivot_classic.items():
                    features[f'{key}{suffix}'] = value
                
                pivot_fibonacci = self._compute_pivot_points_fibonacci_vectorized(high_series, low_series, close_series)
                for key, value in pivot_fibonacci.items():
                    features[f'{key}{suffix}'] = value
                
                # Price Channels
                for period in [20, 50]:
                    price_channel_high = high_series.rolling(window=period).max()
                    price_channel_low = low_series.rolling(window=period).min()
                    features[f'price_channel_high_{period}{suffix}'] = price_channel_high.values
                    features[f'price_channel_low_{period}{suffix}'] = price_channel_low.values
                    features[f'price_channel_middle_{period}{suffix}'] = ((price_channel_high + price_channel_low) / 2).values
                    features[f'price_channel_position_{period}{suffix}'] = ((close_series - price_channel_low) / (price_channel_high - price_channel_low + 1e-8)).values
                
                # Fibonacci Retracements
                fib_levels = self._compute_fibonacci_retracements_vectorized(high_series, low_series, close_series)
                for key, value in fib_levels.items():
                    features[f'{key}{suffix}'] = value
                
                # Support/Resistance Levels
                support_levels, resistance_levels = self._identify_support_resistance_vectorized(high_series, low_series, close_series)
                features[f'nearest_support{suffix}'] = support_levels.values
                features[f'nearest_resistance{suffix}'] = resistance_levels.values
                features[f'support_distance{suffix}'] = ((close_series - support_levels) / close_series).values
                features[f'resistance_distance{suffix}'] = ((resistance_levels - close_series) / close_series).values
                
            except Exception as e:
                logger.warning(f"サポートレジスタンス計算エラー - 時間軸 {tf}: {e}")
                continue
        
        return features

    # ========== ローソク足パターン系指標（マルチタイムフレーム完全対応版） ==========
    
    def _calculate_candlestick_patterns(self, df: pd.DataFrame) -> Dict[str, np.ndarray]:
        """日本のローソク足パターン（マルチタイムフレーム対応版）"""
        features = {}
        
        for tf in self.available_timeframes if self.multi_timeframe else ['1T']:
            try:
                ohlcv = self._extract_ohlcv_for_timeframe(df, tf)
                open_series = ohlcv['open']
                high_series = ohlcv['high']
                low_series = ohlcv['low']
                close_series = ohlcv['close']
                suffix = f'_{tf}' if self.multi_timeframe else ''
                
                # 基本的なローソク足要素
                features[f'body_size{suffix}'] = np.abs(close_series - open_series).values
                features[f'upper_shadow{suffix}'] = (high_series - pd.concat([open_series, close_series], axis=1).max(axis=1)).values
                features[f'lower_shadow{suffix}'] = (pd.concat([open_series, close_series], axis=1).min(axis=1) - low_series).values
                features[f'total_range{suffix}'] = (high_series - low_series).values
                
                # 正規化要素
                total_range_safe = features[f'total_range{suffix}'] + 1e-8
                features[f'body_ratio{suffix}'] = features[f'body_size{suffix}'] / total_range_safe
                features[f'upper_shadow_ratio{suffix}'] = features[f'upper_shadow{suffix}'] / total_range_safe
                features[f'lower_shadow_ratio{suffix}'] = features[f'lower_shadow{suffix}'] / total_range_safe
                
                # ローソク足の種類
                features[f'bullish_candle{suffix}'] = (close_series > open_series).astype(int).values
                features[f'bearish_candle{suffix}'] = (close_series < open_series).astype(int).values
                features[f'doji{suffix}'] = (np.abs(close_series - open_series) < 0.001 * close_series).astype(int).values
                
                # 特殊なローソク足パターン
                features[f'hammer{suffix}'] = self._detect_hammer_vectorized(open_series, high_series, low_series, close_series).values
                features[f'shooting_star{suffix}'] = self._detect_shooting_star_vectorized(open_series, high_series, low_series, close_series).values
                features[f'spinning_top{suffix}'] = self._detect_spinning_top_vectorized(open_series, high_series, low_series, close_series).values
                features[f'marubozu{suffix}'] = self._detect_marubozu_vectorized(open_series, high_series, low_series, close_series).values
                
                # 複数足パターン
                features[f'engulfing_bullish{suffix}'] = self._detect_engulfing_bullish_vectorized(open_series, high_series, low_series, close_series).values
                features[f'engulfing_bearish{suffix}'] = self._detect_engulfing_bearish_vectorized(open_series, high_series, low_series, close_series).values
                features[f'harami{suffix}'] = self._detect_harami_vectorized(open_series, high_series, low_series, close_series).values
                
                # ギャップ
                features[f'gap_up{suffix}'] = self._detect_gap_up_vectorized(high_series, low_series).values
                features[f'gap_down{suffix}'] = self._detect_gap_down_vectorized(high_series, low_series).values
                
            except Exception as e:
                logger.warning(f"ローソク足パターン計算エラー - 時間軸 {tf}: {e}")
                continue
        
        return features
    
    # ========== 3. サポート・レジスタンス系ヘルパー関数 ==========
    
    def _compute_pivot_points_classic_vectorized(self, high_series: pd.Series, low_series: pd.Series, close_series: pd.Series) -> Dict[str, np.ndarray]:
        """Classic Pivot Pointsベクトル化計算"""
        try:
            prev_high = high_series.shift(1)
            prev_low = low_series.shift(1)
            prev_close = close_series.shift(1)
            
            pivot_point = (prev_high + prev_low + prev_close) / 3
            
            resistance_1 = 2 * pivot_point - prev_low
            support_1 = 2 * pivot_point - prev_high
            resistance_2 = pivot_point + (prev_high - prev_low)
            support_2 = pivot_point - (prev_high - prev_low)
            resistance_3 = prev_high + 2 * (pivot_point - prev_low)
            support_3 = prev_low - 2 * (prev_high - pivot_point)
            
            return {
                'pivot_point': pivot_point.values,
                'resistance_1': resistance_1.values,
                'support_1': support_1.values,
                'resistance_2': resistance_2.values,
                'support_2': support_2.values,
                'resistance_3': resistance_3.values,
                'support_3': support_3.values
            }
        except Exception as e:
            logger.warning(f"Classic Pivot Points計算エラー: {e}")
            return {}
    
    def _compute_pivot_points_fibonacci_vectorized(self, high_series: pd.Series, low_series: pd.Series, close_series: pd.Series) -> Dict[str, np.ndarray]:
        """Fibonacci Pivot Pointsベクトル化計算"""
        try:
            prev_high = high_series.shift(1)
            prev_low = low_series.shift(1)
            prev_close = close_series.shift(1)
            
            pivot_point = (prev_high + prev_low + prev_close) / 3
            
            range_val = prev_high - prev_low
            
            fib_resistance_1 = pivot_point + 0.382 * range_val
            fib_support_1 = pivot_point - 0.382 * range_val
            fib_resistance_2 = pivot_point + 0.618 * range_val
            fib_support_2 = pivot_point - 0.618 * range_val
            fib_resistance_3 = pivot_point + range_val
            fib_support_3 = pivot_point - range_val
            
            return {
                'fib_pivot_point': pivot_point.values,
                'fib_resistance_1': fib_resistance_1.values,
                'fib_support_1': fib_support_1.values,
                'fib_resistance_2': fib_resistance_2.values,
                'fib_support_2': fib_support_2.values,
                'fib_resistance_3': fib_resistance_3.values,
                'fib_support_3': fib_support_3.values
            }
        except Exception as e:
            logger.warning(f"Fibonacci Pivot Points計算エラー: {e}")
            return {}
    
    def _compute_fibonacci_retracements_vectorized(self, high_series: pd.Series, low_series: pd.Series, close_series: pd.Series) -> Dict[str, np.ndarray]:
        """Fibonacci Retracementsベクトル化計算"""
        try:
            window = 50
            swing_high = high_series.rolling(window=window).max()
            swing_low = low_series.rolling(window=window).min()
            
            range_val = swing_high - swing_low
            
            fib_23_6 = swing_high - 0.236 * range_val
            fib_38_2 = swing_high - 0.382 * range_val
            fib_50_0 = swing_high - 0.500 * range_val
            fib_61_8 = swing_high - 0.618 * range_val
            fib_78_6 = swing_high - 0.786 * range_val
            
            fib_distance_23_6 = np.abs(close_series - fib_23_6) / close_series
            fib_distance_38_2 = np.abs(close_series - fib_38_2) / close_series
            fib_distance_50_0 = np.abs(close_series - fib_50_0) / close_series
            fib_distance_61_8 = np.abs(close_series - fib_61_8) / close_series
            fib_distance_78_6 = np.abs(close_series - fib_78_6) / close_series
            
            return {
                'fib_retracement_23_6': fib_23_6.values,
                'fib_retracement_38_2': fib_38_2.values,
                'fib_retracement_50_0': fib_50_0.values,
                'fib_retracement_61_8': fib_61_8.values,
                'fib_retracement_78_6': fib_78_6.values,
                'fib_distance_23_6': fib_distance_23_6.values,
                'fib_distance_38_2': fib_distance_38_2.values,
                'fib_distance_50_0': fib_distance_50_0.values,
                'fib_distance_61_8': fib_distance_61_8.values,
                'fib_distance_78_6': fib_distance_78_6.values
            }
        except Exception as e:
            logger.warning(f"Fibonacci Retracements計算エラー: {e}")
            return {}
    
    def _identify_support_resistance_vectorized(self, high_series: pd.Series, low_series: pd.Series, close_series: pd.Series) -> Tuple[pd.Series, pd.Series]:
        """Support/Resistance Levelsベクトル化計算（簡易版）"""
        try:
            window = 20
            
            local_highs = high_series.rolling(window=window, center=True).max()
            local_lows = low_series.rolling(window=window, center=True).min()
            
            support_levels = local_lows.fillna(method='ffill')
            resistance_levels = local_highs.fillna(method='ffill')
            
            return support_levels, resistance_levels
        except Exception as e:
            logger.warning(f"Support/Resistance識別エラー: {e}")
            return close_series, close_series
    
# ========== 4. ローソク足パターン検出ヘルパー関数 ==========
    
    def _detect_hammer_vectorized(self, open_series: pd.Series, high_series: pd.Series, low_series: pd.Series, close_series: pd.Series) -> pd.Series:
        """ハンマー検出ベクトル化"""
        try:
            body_size = np.abs(close_series - open_series)
            total_range = high_series - low_series
            upper_shadow = high_series - pd.concat([open_series, close_series], axis=1).max(axis=1)
            lower_shadow = pd.concat([open_series, close_series], axis=1).min(axis=1) - low_series
            
            hammer_conditions = (
                (lower_shadow > 2 * body_size) &
                (upper_shadow < body_size * 0.5) &
                (body_size > 0) &
                (total_range > 0)
            )
            
            return hammer_conditions.astype(int)
        except:
            return pd.Series(0, index=close_series.index)
    
    def _detect_shooting_star_vectorized(self, open_series: pd.Series, high_series: pd.Series, low_series: pd.Series, close_series: pd.Series) -> pd.Series:
        """流れ星検出ベクトル化"""
        try:
            body_size = np.abs(close_series - open_series)
            total_range = high_series - low_series
            upper_shadow = high_series - pd.concat([open_series, close_series], axis=1).max(axis=1)
            lower_shadow = pd.concat([open_series, close_series], axis=1).min(axis=1) - low_series
            
            shooting_star_conditions = (
                (upper_shadow > 2 * body_size) &
                (lower_shadow < body_size * 0.5) &
                (body_size > 0) &
                (total_range > 0)
            )
            
            return shooting_star_conditions.astype(int)
        except:
            return pd.Series(0, index=close_series.index)
    
    def _detect_spinning_top_vectorized(self, open_series: pd.Series, high_series: pd.Series, low_series: pd.Series, close_series: pd.Series) -> pd.Series:
        """コマ検出ベクトル化"""
        try:
            body_size = np.abs(close_series - open_series)
            total_range = high_series - low_series
            upper_shadow = high_series - pd.concat([open_series, close_series], axis=1).max(axis=1)
            lower_shadow = pd.concat([open_series, close_series], axis=1).min(axis=1) - low_series
            
            spinning_top_conditions = (
                (body_size < total_range * 0.3) &
                (upper_shadow > body_size) &
                (lower_shadow > body_size) &
                (total_range > 0)
            )
            
            return spinning_top_conditions.astype(int)
        except:
            return pd.Series(0, index=close_series.index)
    
    def _detect_marubozu_vectorized(self, open_series: pd.Series, high_series: pd.Series, low_series: pd.Series, close_series: pd.Series) -> pd.Series:
        """丸坊主検出ベクトル化"""
        try:
            body_size = np.abs(close_series - open_series)
            total_range = high_series - low_series
            
            marubozu_conditions = (
                (body_size > total_range * 0.95) &
                (total_range > 0)
            )
            
            return marubozu_conditions.astype(int)
        except:
            return pd.Series(0, index=close_series.index)
    
    def _detect_engulfing_bullish_vectorized(self, open_series: pd.Series, high_series: pd.Series, low_series: pd.Series, close_series: pd.Series) -> pd.Series:
        """強気包み線検出ベクトル化"""
        try:
            prev_open = open_series.shift(1)
            prev_close = close_series.shift(1)
            
            engulfing_bullish_conditions = (
                (prev_close < prev_open) &
                (close_series > open_series) &
                (open_series < prev_close) &
                (close_series > prev_open)
            )
            
            return engulfing_bullish_conditions.astype(int)
        except:
            return pd.Series(0, index=close_series.index)
    
    def _detect_engulfing_bearish_vectorized(self, open_series: pd.Series, high_series: pd.Series, low_series: pd.Series, close_series: pd.Series) -> pd.Series:
        """弱気包み線検出ベクトル化"""
        try:
            prev_open = open_series.shift(1)
            prev_close = close_series.shift(1)
            
            engulfing_bearish_conditions = (
                (prev_close > prev_open) &
                (close_series < open_series) &
                (open_series > prev_close) &
                (close_series < prev_open)
            )
            
            return engulfing_bearish_conditions.astype(int)
        except:
            return pd.Series(0, index=close_series.index)
    
    def _detect_harami_vectorized(self, open_series: pd.Series, high_series: pd.Series, low_series: pd.Series, close_series: pd.Series) -> pd.Series:
        """はらみ線検出ベクトル化"""
        try:
            prev_open = open_series.shift(1)
            prev_close = close_series.shift(1)
            prev_body_max = pd.concat([prev_open, prev_close], axis=1).max(axis=1)
            prev_body_min = pd.concat([prev_open, prev_close], axis=1).min(axis=1)
            
            current_body_max = pd.concat([open_series, close_series], axis=1).max(axis=1)
            current_body_min = pd.concat([open_series, close_series], axis=1).min(axis=1)
            
            harami_conditions = (
                (current_body_max < prev_body_max) &
                (current_body_min > prev_body_min)
            )
            
            return harami_conditions.astype(int)
        except:
            return pd.Series(0, index=close_series.index)
    
    def _detect_gap_up_vectorized(self, high_series: pd.Series, low_series: pd.Series) -> pd.Series:
        """上方ギャップ検出ベクトル化"""
        try:
            prev_high = high_series.shift(1)
            gap_up_conditions = low_series > prev_high
            return gap_up_conditions.astype(int)
        except:
            return pd.Series(0, index=high_series.index)
    
    def _detect_gap_down_vectorized(self, high_series: pd.Series, low_series: pd.Series) -> pd.Series:
        """下方ギャップ検出ベクトル化"""
        try:
            prev_low = low_series.shift(1)
            gap_down_conditions = high_series < prev_low
            return gap_down_conditions.astype(int)
        except:
            return pd.Series(0, index=high_series.index)

# ========== 数学・統計学（マルチタイムフレーム完全対応版） ==========
    
    def _calculate_statistical_moments_basic(self, df: pd.DataFrame) -> Dict[str, np.ndarray]:
        """基本統計的モーメント（マルチタイムフレーム対応版）"""
        features = {}
        
        for tf in self.available_timeframes if self.multi_timeframe else ['1T']:
            try:
                ohlcv = self._extract_ohlcv_for_timeframe(df, tf)
                close_series = ohlcv['close']
                returns = np.log(close_series / close_series.shift(1)).fillna(0)
                suffix = f'_{tf}' if self.multi_timeframe else ''
                
                # 窓ごとの基本モーメント
                for window in [10, 20, 50]:
                    # 基本モーメント
                    features[f'mean_{window}{suffix}'] = returns.rolling(window=window).mean().values
                    features[f'variance_{window}{suffix}'] = returns.rolling(window=window).var().values
                    features[f'skewness_{window}{suffix}'] = returns.rolling(window=window).skew().values
                    features[f'kurtosis_{window}{suffix}'] = returns.rolling(window=window).kurt().values
                
            except Exception as e:
                logger.warning(f"基本統計的モーメント計算エラー - 時間軸 {tf}: {e}")
                continue
        
        return features
    
    def _calculate_statistical_moments_advanced(self, df: pd.DataFrame) -> Dict[str, np.ndarray]:
        """高次統計的モーメント（マルチタイムフレーム対応版）"""
        features = {}
        
        for tf in self.available_timeframes if self.multi_timeframe else ['1T']:
            try:
                ohlcv = self._extract_ohlcv_for_timeframe(df, tf)
                close_series = ohlcv['close']
                returns = np.log(close_series / close_series.shift(1)).fillna(0)
                suffix = f'_{tf}' if self.multi_timeframe else ''
                
                 # 窓ごとの高次モーメント
                for window in [10, 20, 50]:
                    # 高次モーメント（5次～8次）
                    for moment in range(5, 9):
                        # Numba化された静的メソッドを呼び出す
                        # partialを使って第二引数momentを固定
                        func = partial(self._numba_raw_moment, moment=moment)
                        features[f'moment_{moment}_{window}{suffix}'] = returns.rolling(window=window).apply(
                            func, raw=True
                        ).values
                    
                    # 中心モーメント
                    features[f'central_moment_3_{window}{suffix}'] = returns.rolling(window=window).apply(
                        lambda x: np.mean((x - np.mean(x))**3), raw=True
                    ).values
                    features[f'central_moment_4_{window}{suffix}'] = returns.rolling(window=window).apply(
                        lambda x: np.mean((x - np.mean(x))**4), raw=True
                    ).values
                
            except Exception as e:
                logger.warning(f"高次統計的モーメント計算エラー - 時間軸 {tf}: {e}")
                continue
        
        return features
    
    def _calculate_probability_distributions(self, df: pd.DataFrame) -> Dict[str, np.ndarray]:
        """確率分布フィッティング（マルチタイムフレーム対応版）"""
        features = {}
        
        for tf in self.available_timeframes if self.multi_timeframe else ['1T']:
            try:
                ohlcv = self._extract_ohlcv_for_timeframe(df, tf)
                close_series = ohlcv['close']
                returns = np.log(close_series / close_series.shift(1)).fillna(0)
                suffix = f'_{tf}' if self.multi_timeframe else ''
                
                # 窓ごとの分布パラメータ
                for window in [20, 50]:
                    # 正規分布パラメータ
                    features[f'normal_mu_{window}{suffix}'] = returns.rolling(window=window).mean().values
                    features[f'normal_sigma_{window}{suffix}'] = returns.rolling(window=window).std().values
                    
                    # t分布自由度推定
                    features[f't_dist_df_{window}{suffix}'] = returns.rolling(window=window).apply(
                        self._estimate_t_df_vectorized, raw=True
                    ).values
                    
                    # ガンマ分布パラメータ（正の値のみ）
                    abs_returns = np.abs(returns)
                    features[f'gamma_shape_{window}{suffix}'] = abs_returns.rolling(window=window).apply(
                        self._estimate_gamma_shape_vectorized, raw=True
                    ).values
                    
                    # ベータ分布パラメータ（0-1正規化後）
                    norm_returns = (returns - returns.min()) / (returns.max() - returns.min() + 1e-8)
                    features[f'beta_alpha_{window}{suffix}'] = norm_returns.rolling(window=window).apply(
                        self._estimate_beta_alpha_vectorized, raw=True
                    ).values
                
            except Exception as e:
                logger.warning(f"確率分布フィッティング計算エラー - 時間軸 {tf}: {e}")
                continue
        
        return features
    
    def _calculate_robust_statistics(self, df: pd.DataFrame) -> Dict[str, np.ndarray]:
        """ロバスト統計量（マルチタイムフレーム対応版）"""
        features = {}
        
        for tf in self.available_timeframes if self.multi_timeframe else ['1T']:
            try:
                ohlcv = self._extract_ohlcv_for_timeframe(df, tf)
                close_series = ohlcv['close']
                returns = np.log(close_series / close_series.shift(1)).fillna(0)
                suffix = f'_{tf}' if self.multi_timeframe else ''
                
                for window in [15, 30]:
                    # 中央値系
                    features[f'median_{window}{suffix}'] = returns.rolling(window=window).median().values
                    features[f'mad_{window}{suffix}'] = returns.rolling(window=window).apply(
                        self._median_abs_deviation_vectorized, raw=True
                    ).values
                    
                    # 四分位数
                    features[f'q25_{window}{suffix}'] = returns.rolling(window=window).quantile(0.25).values
                    features[f'q75_{window}{suffix}'] = returns.rolling(window=window).quantile(0.75).values
                    features[f'iqr_{window}{suffix}'] = (features[f'q75_{window}{suffix}'] - features[f'q25_{window}{suffix}'])
                    
                    # トリム平均
                    features[f'trim_mean_10_{window}{suffix}'] = returns.rolling(window=window).apply(
                        lambda x: stats.trim_mean(x, 0.1), raw=True
                    ).values
                    features[f'trim_mean_20_{window}{suffix}'] = returns.rolling(window=window).apply(
                        lambda x: stats.trim_mean(x, 0.2), raw=True
                    ).values
                    
                    # ウィンザー化統計
                    features[f'winsor_mean_{window}{suffix}'] = returns.rolling(window=window).apply(
                        self._winsorized_mean_vectorized, raw=True
                    ).values
                
            except Exception as e:
                logger.warning(f"ロバスト統計量計算エラー - 時間軸 {tf}: {e}")
                continue
        
        return features
    
    def _calculate_order_statistics(self, df: pd.DataFrame) -> Dict[str, np.ndarray]:
        """順序統計量（マルチタイムフレーム対応版）"""
        features = {}
        
        for tf in self.available_timeframes if self.multi_timeframe else ['1T']:
            try:
                ohlcv = self._extract_ohlcv_for_timeframe(df, tf)
                close_series = ohlcv['close']
                suffix = f'_{tf}' if self.multi_timeframe else ''
                
                # 価格順位
                for window in [10, 25, 50]:
                    features[f'price_rank_{window}{suffix}'] = close_series.rolling(window=window).apply(
                        lambda x: (x.iloc[-1] - x.min()) / (x.max() - x.min() + 1e-8), raw=False
                    ).values
                    
                    # 分位数位置
                    features[f'quantile_pos_{window}{suffix}'] = close_series.rolling(window=window).apply(
                        lambda x: stats.percentileofscore(x, x.iloc[-1]) / 100, raw=False
                    ).values
                    
                    # 極値比率
                    features[f'extreme_ratio_{window}{suffix}'] = close_series.rolling(window=window).apply(
                        self._extreme_value_ratio_vectorized, raw=False
                    ).values
                
            except Exception as e:
                logger.warning(f"順序統計量計算エラー - 時間軸 {tf}: {e}")
                continue
        
        return features
    
    # ========== 信号処理系（マルチタイムフレーム完全対応版） ==========
    
    def _calculate_signal_processing_basic(self, df: pd.DataFrame) -> Dict[str, np.ndarray]:
        """基本信号処理特徴量（マルチタイムフレーム対応版）"""
        features = {}
        
        for tf in self.available_timeframes if self.multi_timeframe else ['1T']:
            try:
                ohlcv = self._extract_ohlcv_for_timeframe(df, tf)
                close_series = ohlcv['close']
                suffix = f'_{tf}' if self.multi_timeframe else ''
                
                # ヒルベルト変換による位相・振幅
                for window in [32, 64]:
                    features[f'hilbert_amplitude_{window}{suffix}'] = close_series.rolling(window=window).apply(
                        self._hilbert_amplitude_vectorized, raw=True
                    ).values
                    features[f'hilbert_phase_{window}{suffix}'] = close_series.rolling(window=window).apply(
                        self._hilbert_phase_vectorized, raw=True
                    ).values
                    features[f'instantaneous_freq_{window}{suffix}'] = close_series.rolling(window=window).apply(
                        self._instantaneous_frequency_vectorized, raw=True
                    ).values
                
                # 自己相関
                for lag in [5, 10, 20]:
                    features[f'autocorr_lag_{lag}{suffix}'] = close_series.rolling(window=50).apply(
                        lambda x: self._autocorrelation_vectorized(x, lag), raw=False
                    ).values
                
                # 相互相関（価格と出来高）
                volume_series = ohlcv['volume']
                features[f'cross_correlation{suffix}'] = close_series.rolling(window=50).apply(
                    lambda x: self._cross_correlation_vectorized(x, volume_series.loc[x.index]), raw=False
                ).values
                
            except Exception as e:
                logger.warning(f"基本信号処理計算エラー - 時間軸 {tf}: {e}")
                continue
        
        return features
    
    def _calculate_signal_processing_advanced(self, df: pd.DataFrame) -> Dict[str, np.ndarray]:
        """高度信号処理特徴量（マルチタイムフレーム対応版）"""
        features = {}
        
        for tf in self.available_timeframes if self.multi_timeframe else ['1T']:
            try:
                ohlcv = self._extract_ohlcv_for_timeframe(df, tf)
                close_series = ohlcv['close']
                suffix = f'_{tf}' if self.multi_timeframe else ''
                
                # スペクトログラム特徴
                for window in [64, 128]:
                    features[f'spectral_centroid_{window}{suffix}'] = close_series.rolling(window=window).apply(
                        self._spectral_centroid_vectorized, raw=True
                    ).values
                    features[f'spectral_bandwidth_{window}{suffix}'] = close_series.rolling(window=window).apply(
                        self._spectral_bandwidth_vectorized, raw=True
                    ).values
                    features[f'spectral_rolloff_{window}{suffix}'] = close_series.rolling(window=window).apply(
                        self._spectral_rolloff_vectorized, raw=True
                    ).values
                    features[f'spectral_flux_{window}{suffix}'] = close_series.rolling(window=window).apply(
                        self._spectral_flux_vectorized, raw=True
                    ).values
                
                # ゼロクロッシング率
                features[f'zero_crossing_rate{suffix}'] = close_series.rolling(window=50).apply(
                    self._zero_crossing_rate_vectorized, raw=True
                ).values
                
                # コヒーレンス（価格と出来高）
                volume_series = ohlcv['volume']
                features[f'coherence_measure{suffix}'] = close_series.rolling(window=100).apply(
                    lambda x: self._coherence_measure_vectorized(x, volume_series.loc[x.index]), raw=False
                ).values
                
            except Exception as e:
                logger.warning(f"高度信号処理計算エラー - 時間軸 {tf}: {e}")
                continue
        
        return features
    
   # ========== 統計・信号処理ヘルパー関数 ==========

    @staticmethod
    @njit(cache=True)
    def _numba_raw_moment(x: np.ndarray, moment: int) -> float:
        """Numba JIT化された生モーメント計算"""
        # Numba内で安全に標準偏差をチェック (handle_zero_stdの代替)
        if x.std() < 1e-9:
            return 0.0
        # Numbaはtry-exceptを限定的にしかサポートしないため、条件分岐で対応
        return np.mean(x**moment)

    # 他の多くのヘルパー関数も同様に@staticmethodと@njitで高速化できます

    @handle_zero_std
    def _median_abs_deviation_vectorized(self, x: np.ndarray) -> float:
        """中央絶対偏差計算"""
        try:
            return np.median(np.abs(x - np.median(x)))
        except:
            return 0.0
    
    @handle_zero_std
    def _winsorized_mean_vectorized(self, x: np.ndarray) -> float:
        """ウィンザー化平均計算"""
        try:
            q25, q75 = np.percentile(x, [25, 75])
            x_winsorized = np.clip(x, q25, q75)
            return np.mean(x_winsorized)
        except:
            return 0.0
    
    @handle_zero_std
    def _extreme_value_ratio_vectorized(self, series: pd.Series) -> float:
        """極値比率計算"""
        try:
            values = series.values
            q95 = np.percentile(values, 95)
            q05 = np.percentile(values, 5)
            extreme_count = np.sum((values > q95) | (values < q05))
            return extreme_count / len(values)
        except:
            return 0.0
    
    @handle_zero_std
    def _estimate_t_df_vectorized(self, x: np.ndarray) -> float:
        """t分布自由度推定"""
        try:
            if len(x) < 5:
                return 5.0
            
            # 簡易推定：サンプル分散とt分布の理論分散から逆算
            sample_var = np.var(x)
            sample_kurt = stats.kurtosis(x)
            
            # t分布の尖度から自由度を推定
            if sample_kurt > 0:
                df_estimate = 6 / sample_kurt + 4
                return max(2.0, min(30.0, df_estimate))
            else:
                return 10.0
        except:
            return 5.0
    
    @handle_zero_std
    def _estimate_gamma_shape_vectorized(self, x: np.ndarray) -> float:
        """ガンマ分布形状パラメータ推定"""
        try:
            if len(x) < 3 or np.any(x <= 0):
                return 1.0
            
            mean_x = np.mean(x)
            var_x = np.var(x)
            
            if var_x <= 0:
                return 1.0
            
            # モーメント法による推定
            shape = mean_x**2 / var_x
            return max(0.1, min(10.0, shape))
        except:
            return 1.0
    
    @handle_zero_std
    def _estimate_beta_alpha_vectorized(self, x: np.ndarray) -> float:
        """ベータ分布αパラメータ推定"""
        try:
            if len(x) < 3:
                return 1.0
            
            # データを[0,1]区間に正規化済みと仮定
            x_clipped = np.clip(x, 1e-6, 1-1e-6)
            
            mean_x = np.mean(x_clipped)
            var_x = np.var(x_clipped)
            
            if var_x <= 0 or mean_x <= 0 or mean_x >= 1:
                return 1.0
            
            # モーメント法による推定
            alpha = mean_x * (mean_x * (1 - mean_x) / var_x - 1)
            return max(0.1, min(10.0, alpha))
        except:
            return 1.0
    
    @handle_zero_std
    def _hilbert_amplitude_vectorized(self, x: np.ndarray) -> float:
        """ヒルベルト変換による瞬時振幅"""
        try:
            if len(x) < 4:
                return 0.0
            analytic_signal = hilbert(x)
            amplitude = np.abs(analytic_signal)
            return np.mean(amplitude)
        except:
            return 0.0
    
    @handle_zero_std
    def _hilbert_phase_vectorized(self, x: np.ndarray) -> float:
        """ヒルベルト変換による瞬時位相"""
        try:
            if len(x) < 4:
                return 0.0
            analytic_signal = hilbert(x)
            phase = np.angle(analytic_signal)
            return np.std(np.unwrap(phase))
        except:
            return 0.0
    
    @handle_zero_std
    def _instantaneous_frequency_vectorized(self, x: np.ndarray) -> float:
        """瞬時周波数計算"""
        try:
            if len(x) < 4:
                return 0.0
            analytic_signal = hilbert(x)
            phase = np.unwrap(np.angle(analytic_signal))
            inst_freq = np.diff(phase) / (2 * np.pi)
            return np.mean(np.abs(inst_freq))
        except:
            return 0.0
    
    @handle_zero_std
    def _autocorrelation_vectorized(self, series: pd.Series, lag: int) -> float:
        """自己相関計算"""
        try:
            if len(series) <= lag:
                return 0.0
            return series.autocorr(lag=lag)
        except:
            return 0.0
    
    @handle_zero_std
    def _cross_correlation_vectorized(self, x: pd.Series, y: pd.Series) -> float:
        """相互相関計算"""
        try:
            if len(x) != len(y) or len(x) < 5:
                return 0.0
            correlation = np.corrcoef(x.values, y.values)[0, 1]
            return correlation if not np.isnan(correlation) else 0.0
        except:
            return 0.0
    
    @handle_zero_std
    def _spectral_centroid_vectorized(self, x: np.ndarray) -> float:
        """スペクトル重心計算"""
        try:
            if len(x) < 8:
                return 0.0
            
            # FFTスペクトル
            spectrum = np.abs(fft(x))
            freqs = fftfreq(len(x))
            
            # 正の周波数のみ
            positive_freqs = freqs[:len(freqs)//2]
            positive_spectrum = spectrum[:len(spectrum)//2]
            
            if np.sum(positive_spectrum) == 0:
                return 0.0
            
            # スペクトル重心
            centroid = np.sum(positive_freqs * positive_spectrum) / np.sum(positive_spectrum)
            return abs(centroid)
        except:
            return 0.0
    
    @handle_zero_std
    def _spectral_bandwidth_vectorized(self, x: np.ndarray) -> float:
        """スペクトル帯域幅計算"""
        try:
            if len(x) < 8:
                return 0.0
            
            spectrum = np.abs(fft(x))
            freqs = fftfreq(len(x))
            
            positive_freqs = freqs[:len(freqs)//2]
            positive_spectrum = spectrum[:len(spectrum)//2]
            
            if np.sum(positive_spectrum) == 0:
                return 0.0
            
            # スペクトル重心
            centroid = np.sum(positive_freqs * positive_spectrum) / np.sum(positive_spectrum)
            
            # 帯域幅（重心からの偏差の重み付き標準偏差）
            bandwidth = np.sqrt(
                np.sum(((positive_freqs - centroid)**2) * positive_spectrum) / 
                np.sum(positive_spectrum)
            )
            return bandwidth
        except:
            return 0.0
    
    @handle_zero_std
    def _spectral_rolloff_vectorized(self, x: np.ndarray) -> float:
        """スペクトルロールオフ計算"""
        try:
            if len(x) < 8:
                return 0.0
            
            spectrum = np.abs(fft(x))
            freqs = fftfreq(len(x))
            
            positive_freqs = freqs[:len(freqs)//2]
            positive_spectrum = spectrum[:len(spectrum)//2]
            
            if np.sum(positive_spectrum) == 0:
                return 0.0
            
            # 85%エネルギー点を見つける
            total_energy = np.sum(positive_spectrum**2)
            cumulative_energy = np.cumsum(positive_spectrum**2)
            
            rolloff_index = np.where(cumulative_energy >= 0.85 * total_energy)[0]
            if len(rolloff_index) > 0:
                return positive_freqs[rolloff_index[0]]
            else:
                return positive_freqs[-1]
        except:
            return 0.0
    
    @handle_zero_std
    def _spectral_flux_vectorized(self, x: np.ndarray) -> float:
        """スペクトルフラックス計算"""
        try:
            if len(x) < 16:
                return 0.0
            
            # 前半と後半に分けて比較
            mid = len(x) // 2
            spectrum1 = np.abs(fft(x[:mid]))
            spectrum2 = np.abs(fft(x[mid:]))
            
            # 長さを合わせる
            min_len = min(len(spectrum1), len(spectrum2))
            spectrum1 = spectrum1[:min_len]
            spectrum2 = spectrum2[:min_len]
            
            # スペクトル差の2乗和
            flux = np.sum((spectrum2 - spectrum1)**2)
            return flux
        except:
            return 0.0
    
    @handle_zero_std
    def _zero_crossing_rate_vectorized(self, x: np.ndarray) -> float:
        """ゼロクロッシング率計算"""
        try:
            if len(x) < 3:
                return 0.0
            
            # 平均を引いてゼロ中心にする
            x_centered = x - np.mean(x)
            
            # 符号の変化をカウント
            sign_changes = np.sum(np.abs(np.diff(np.sign(x_centered))))
            zero_crossing_rate = sign_changes / (2 * len(x))
            
            return zero_crossing_rate
        except:
            return 0.0
    
    @handle_zero_std
    def _coherence_measure_vectorized(self, x: pd.Series, y: pd.Series) -> float:
        """コヒーレンス測定"""
        try:
            if len(x) != len(y) or len(x) < 10:
                return 0.0
            
            # パワースペクトル密度とクロススペクトル密度を計算
            f, Pxx = welch(x.values, nperseg=min(len(x)//4, 256))
            f, Pyy = welch(y.values, nperseg=min(len(y)//4, 256))
            f, Pxy = welch(x.values, y.values, nperseg=min(len(x)//4, 256))
            
            # コヒーレンス
            coherence = np.abs(Pxy)**2 / (Pxx * Pyy + 1e-8)
            
            # 平均コヒーレンス
            return np.mean(coherence)
        except:
            # フォールバック：単純な相関
            try:
                correlation = np.corrcoef(x.values, y.values)[0, 1]
                return abs(correlation) if not np.isnan(correlation) else 0.0
            except:
                return 0.0

# ========== 9. フーリエ・ウェーブレット・フィルタリング ==========

    # --- 27/57: Fourier Analysis ---
    def _fourier_analysis_logic(self, ohlcv: Dict[str, pd.Series], timeframe: str) -> Dict[str, np.ndarray]:
        """フーリエ解析の具体的な計算ロジック"""
        features = {}
        close_series = ohlcv['close']
        suffix = f'_{timeframe}' if self.multi_timeframe else ''

        for window in self.fourier_windows:
            features[f'fft_power_mean_{window}{suffix}'] = close_series.rolling(window=window).apply(
                self._fft_power_mean_vectorized, raw=True
            ).values
            features[f'fft_power_std_{window}{suffix}'] = close_series.rolling(window=window).apply(
                self._fft_power_std_vectorized, raw=True
            ).values
            features[f'fft_peak_freq_{window}{suffix}'] = close_series.rolling(window=window).apply(
                self._fft_dominant_frequency_vectorized, raw=True
            ).values
            features[f'fft_bandwidth_{window}{suffix}'] = close_series.rolling(window=window).apply(
                self._fft_bandwidth_vectorized, raw=True
            ).values
            features[f'fft_phase_coherence_{window}{suffix}'] = close_series.rolling(window=window).apply(
                self._fft_phase_coherence_vectorized, raw=True
            ).values
        return features

    def _calculate_fourier_analysis(self, df: pd.DataFrame) -> Dict[str, np.ndarray]:
        """フーリエ解析特徴量（共通処理ヘルパー使用）"""
        logic = partial(self._fourier_analysis_logic)
        return self._process_feature_group(df, logic, "Fourier Analysis")

    # --- 28/57: Wavelet Analysis ---
    def _wavelet_analysis_logic(self, ohlcv: Dict[str, pd.Series], timeframe: str) -> Dict[str, np.ndarray]:
        """ウェーブレット解析の具体的な計算ロジック"""
        features = {}
        close_series = ohlcv['close']
        suffix = f'_{timeframe}' if self.multi_timeframe else ''

        for wavelet in self.wavelets:
            try:
                for window in self.cwt_windows:
                    features[f'cwt_energy_{wavelet}_{window}{suffix}'] = close_series.rolling(window=window).apply(
                        lambda x: self._cwt_energy_vectorized(x, wavelet), raw=True
                    ).values
                    features[f'cwt_entropy_{wavelet}_{window}{suffix}'] = close_series.rolling(window=window).apply(
                        lambda x: self._cwt_entropy_vectorized(x, wavelet), raw=True
                    ).values
            except Exception as e:
                logger.warning(f"Wavelet calculation failed for {wavelet} on timeframe {timeframe}: {e}")
                continue
        return features

    def _calculate_wavelet_analysis(self, df: pd.DataFrame) -> Dict[str, np.ndarray]:
        """ウェーブレット解析（共通処理ヘルパー使用）"""
        logic = partial(self._wavelet_analysis_logic)
        return self._process_feature_group(df, logic, "Wavelet Analysis")

    # --- 29/57: Filtering Features ---
    def _filtering_features_logic(self, ohlcv: Dict[str, pd.Series], timeframe: str) -> Dict[str, np.ndarray]:
        """フィルタリングの具体的な計算ロジック"""
        features = {}
        close_series = ohlcv['close']
        suffix = f'_{timeframe}' if self.multi_timeframe else ''

        for sigma in self.gaussian_sigmas:
            filtered = gaussian_filter(close_series.values, sigma=sigma)
            features[f'gaussian_filtered_{sigma}{suffix}'] = filtered
            features[f'gaussian_residual_{sigma}{suffix}'] = close_series.values - filtered
        
        for size in self.median_sizes:
            filtered = median_filter(close_series.values, size=size)
            features[f'median_filtered_{size}{suffix}'] = filtered
            features[f'median_residual_{size}{suffix}'] = close_series.values - filtered
        
        if len(close_series) > 11:
            for window in self.savgol_windows:
                if len(close_series) >= window:
                    filtered = savgol_filter(close_series.values, window, 3)
                    features[f'savgol_filtered_{window}{suffix}'] = filtered
                    features[f'savgol_residual_{window}{suffix}'] = close_series.values - filtered
        return features

    def _calculate_filtering_features(self, df: pd.DataFrame) -> Dict[str, np.ndarray]:
        """フィルタリング特徴量（共通処理ヘルパー使用）"""
        logic = partial(self._filtering_features_logic)
        return self._process_feature_group(df, logic, "Filtering Features")

    # --- 30/57: Energy Features ---
    def _energy_features_logic(self, ohlcv: Dict[str, pd.Series], timeframe: str) -> Dict[str, np.ndarray]:
        """エネルギー特徴量の具体的な計算ロジック"""
        features = {}
        close_series = ohlcv['close']
        volume_series = ohlcv['volume']
        suffix = f'_{timeframe}' if self.multi_timeframe else ''

        for window in self.energy_windows:
            features[f'rms_energy_{window}{suffix}'] = close_series.rolling(window=window).apply(
                lambda x: np.sqrt(np.mean(x**2)), raw=True
            ).values
            features[f'power_{window}{suffix}'] = close_series.rolling(window=window).apply(
                lambda x: np.mean(x**2), raw=True
            ).values
            features[f'crest_factor_{window}{suffix}'] = close_series.rolling(window=window).apply(
                lambda x: np.max(np.abs(x)) / (np.sqrt(np.mean(x**2)) + 1e-8), raw=True
            ).values
        
        features[f'volume_energy{suffix}'] = np.sqrt(np.cumsum(volume_series.values**2))
        return features

    def _calculate_energy_features(self, df: pd.DataFrame) -> Dict[str, np.ndarray]:
        """エネルギー特徴量（共通処理ヘルパー使用）"""
        logic = partial(self._energy_features_logic)
        return self._process_feature_group(df, logic, "Energy Features")
        
# ========== フーリエ・ウェーブレット・エネルギー解析ヘルパー関数 ==========
    
    @handle_zero_std
    def _fft_power_mean_vectorized(self, x: np.ndarray) -> float:
        """FFTパワー平均計算（ベクトル化版）"""
        try:
            X = np.abs(fft(x))**2
            return np.mean(X)
        except:
            return 0.0
    
    @handle_zero_std
    def _fft_power_std_vectorized(self, x: np.ndarray) -> float:
        """FFTパワー標準偏差計算（ベクトル化版）"""
        try:
            X = np.abs(fft(x))**2
            return np.std(X)
        except:
            return 0.0
    
    @handle_zero_std
    def _fft_dominant_frequency_vectorized(self, x: np.ndarray) -> float:
        """FFT主要周波数計算（ベクトル化版）"""
        try:
            X = np.abs(fft(x))
            freqs = fftfreq(len(x))
            return freqs[np.argmax(X)]
        except:
            return 0.0
    
    @handle_zero_std
    def _fft_bandwidth_vectorized(self, x: np.ndarray) -> float:
        """FFT帯域幅計算（ベクトル化版）"""
        try:
            X = np.abs(fft(x))**2
            freqs = fftfreq(len(x))
            total_power = np.sum(X)
            if total_power == 0:
                return 0.0
            
            # 重心周波数
            centroid = np.sum(freqs * X) / total_power
            
            # 帯域幅
            bandwidth = np.sqrt(np.sum(((freqs - centroid)**2) * X) / total_power)
            return bandwidth
        except:
            return 0.0
    
    @handle_zero_std
    def _fft_phase_coherence_vectorized(self, x: np.ndarray) -> float:
        """FFT位相コヒーレンス計算（ベクトル化版）"""
        try:
            X = fft(x)
            phases = np.angle(X)
            # 位相の一貫性を測定
            phase_coherence = np.abs(np.mean(np.exp(1j * phases)))
            return phase_coherence
        except:
            return 0.0
    
    @handle_zero_std
    def _cwt_energy_vectorized(self, x: np.ndarray, wavelet: str) -> float:
        """CWTエネルギー計算（ベクトル化版）"""
        try:
            scales = np.arange(1, min(32, len(x)//4))
            if len(scales) == 0:
                return 0.0
            coeffs, _ = pywt.cwt(x, scales, wavelet)
            return np.sum(np.abs(coeffs)**2)
        except:
            return 0.0
    
    @handle_zero_std
    def _cwt_entropy_vectorized(self, x: np.ndarray, wavelet: str) -> float:
        """CWTエントロピー計算（ベクトル化版）"""
        try:
            scales = np.arange(1, min(32, len(x)//4))
            if len(scales) == 0:
                return 0.0
            coeffs, _ = pywt.cwt(x, scales, wavelet)
            energy = np.abs(coeffs)**2
            total_energy = np.sum(energy)
            if total_energy == 0:
                return 0.0
            
            prob = energy / total_energy
            prob = prob[prob > 0]  # ゼロを除去
            return -np.sum(prob * np.log2(prob))
        except:
            return 0.0
        
    # ========== 10. 情報理論系特徴量 ==========

    # --- 31/57: Basic Entropy Features ---
    def _entropy_features_basic_logic(self, ohlcv: Dict[str, pd.Series], timeframe: str) -> Dict[str, np.ndarray]:
        """基本エントロピーの具体的な計算ロジック"""
        features = {}
        close_series = ohlcv['close']
        returns = np.log(close_series / close_series.shift(1)).fillna(0)
        suffix = f'_{timeframe}' if self.multi_timeframe else ''

        for window in self.entropy_windows:
            features[f'shannon_entropy_{window}{suffix}'] = returns.rolling(window=window).apply(
                self._shannon_entropy_vectorized, raw=True
            ).values
            features[f'conditional_entropy_{window}{suffix}'] = returns.rolling(window=window).apply(
                self._conditional_entropy_vectorized, raw=True
            ).values
        return features

    def _calculate_entropy_features_basic(self, df: pd.DataFrame) -> Dict[str, np.ndarray]:
        """基本エントロピー系特徴量（共通処理ヘルパー使用）"""
        logic = partial(self._entropy_features_basic_logic)
        return self._process_feature_group(df, logic, "Basic Entropy")

    # --- 32/57: Advanced Entropy Features ---
    def _entropy_features_advanced_logic(self, ohlcv: Dict[str, pd.Series], timeframe: str) -> Dict[str, np.ndarray]:
        """高度エントロピーの具体的な計算ロジック"""
        features = {}
        if not ENTROPY_AVAILABLE:
            return features
            
        close_series = ohlcv['close']
        returns = np.log(close_series / close_series.shift(1)).fillna(0)
        suffix = f'_{timeframe}' if self.multi_timeframe else ''

        for window in self.adv_entropy_windows:
            features[f'sample_entropy_{window}{suffix}'] = returns.rolling(window=window).apply(
                self._safe_sample_entropy, raw=True
            ).values
            features[f'approx_entropy_{window}{suffix}'] = returns.rolling(window=window).apply(
                self._safe_approx_entropy, raw=True
            ).values
        return features

    def _calculate_entropy_features_advanced(self, df: pd.DataFrame) -> Dict[str, np.ndarray]:
        """高度エントロピー系特徴量（共通処理ヘルパー使用）"""
        logic = partial(self._entropy_features_advanced_logic)
        return self._process_feature_group(df, logic, "Advanced Entropy")

    # --- 33/57: Complexity Features ---
    def _complexity_features_logic(self, ohlcv: Dict[str, pd.Series], timeframe: str) -> Dict[str, np.ndarray]:
        """複雑性特徴量の具体的な計算ロジック"""
        features = {}
        close_series = ohlcv['close']
        suffix = f'_{timeframe}' if self.multi_timeframe else ''

        for window in self.lz_windows:
            features[f'lz_complexity_{window}{suffix}'] = close_series.rolling(window=window).apply(
                self._lempel_ziv_complexity_vectorized, raw=True
            ).values
        
        for window in self.kolmogorov_windows:
            features[f'kolmogorov_approx_{window}{suffix}'] = close_series.rolling(window=window).apply(
                self._kolmogorov_complexity_approx_vectorized, raw=True
            ).values
        
        features[f'encoding_length{suffix}'] = close_series.rolling(window=40).apply(
            self._encoding_length_vectorized, raw=True
        ).values
        return features

    def _calculate_complexity_features(self, df: pd.DataFrame) -> Dict[str, np.ndarray]:
        """複雑性特徴量（共通処理ヘルパー使用）"""
        logic = partial(self._complexity_features_logic)
        return self._process_feature_group(df, logic, "Complexity Features")

    # --- 34/57: Information Measures ---
    def _information_measures_logic(self, ohlcv: Dict[str, pd.Series], timeframe: str) -> Dict[str, np.ndarray]:
        """情報量特徴量の具体的な計算ロジック"""
        features = {}
        close_series = ohlcv['close']
        volume_series = ohlcv['volume']
        suffix = f'_{timeframe}' if self.multi_timeframe else ''

        close_values = pd.Series(close_series.values, index=range(len(close_series)))
        volume_values = pd.Series(volume_series.values, index=range(len(volume_series)))
        
        for lag in self.mutual_info_lags:
            features[f'mutual_info_lag_{lag}{suffix}'] = close_values.rolling(window=50).apply(
                lambda x: self._mutual_information_safe_vectorized(x, volume_values, lag), raw=False
            ).values
        
        features[f'transfer_entropy{suffix}'] = close_values.rolling(window=50).apply(
            self._transfer_entropy_vectorized, raw=True
        ).values
        return features

    def _calculate_information_measures(self, df: pd.DataFrame) -> Dict[str, np.ndarray]:
        """情報量特徴量（共通処理ヘルパー使用）"""
        logic = partial(self._information_measures_logic)
        return self._process_feature_group(df, logic, "Information Measures")
        
# ========== エントロピー・情報理論ヘルパー関数 ==========

    # --- Numba 高速化対象の関数 ---
    @staticmethod
    @njit(cache=True)
    def _numba_shannon_entropy(x: np.ndarray) -> float:
        """Numba JIT化: Shannonエントロピー計算"""
        if x.std() < 1e-9:
            return 0.0

        bins = min(10, len(x) // 3)
        if bins < 2:
            return 0.0
        
        hist, _ = np.histogram(x, bins=bins)
        
        # ゼロを除去し、確率を計算
        total_sum = 0
        for count in hist:
            total_sum += count
        
        if total_sum == 0:
            return 0.0

        entropy = 0.0
        for count in hist:
            if count > 0:
                prob = count / total_sum
                entropy -= prob * np.log2(prob)
        return entropy

    @staticmethod
    @njit(cache=True)
    def _numba_conditional_entropy(x: np.ndarray) -> float:
        """Numba JIT化: 条件付きエントロピー計算"""
        if len(x) < 2 or x.std() < 1e-9:
            return 0.0
        
        bins = min(5, len(x) // 5)
        if bins < 2:
            return 0.0
            
        # Numbaはnp.histogramの戻り値の型推論に厳密なため、明示的に型を指定
        bin_edges = np.histogram(x, bins=bins)[1]
        x_binned = np.digitize(x[:-1], bin_edges)
        y_binned = np.digitize(x[1:], bin_edges)
        
        joint_counts = np.zeros((bins + 2, bins + 2), dtype=np.float64)
        for i in range(len(x_binned)):
            joint_counts[x_binned[i], y_binned[i]] += 1
        
        total = np.sum(joint_counts)
        if total == 0:
            return 0.0
            
        entropy = 0.0
        for i in range(joint_counts.shape[0]):
            px = np.sum(joint_counts[i, :])
            if px > 0:
                for j in range(joint_counts.shape[1]):
                    pxy = joint_counts[i, j]
                    if pxy > 0:
                        entropy -= (pxy / total) * np.log2((pxy / total) / (px / total))
        return entropy

    @staticmethod
    @njit(cache=True)
    def _numba_encoding_length(x: np.ndarray) -> float:
        """Numba JIT化: 符号化長計算"""
        if x.std() < 1e-9 or len(x) == 0:
            return 0.0
        
        # Numbaでnp.uniqueを再現
        unique_vals = np.unique(x)
        if len(unique_vals) <= 1:
            return 0.0

        counts = np.zeros(len(unique_vals), dtype=np.int64)
        for i in range(len(unique_vals)):
            counts[i] = np.sum(x == unique_vals[i])

        total = len(x)
        prob = counts / total
        
        entropy = 0.0
        for p in prob:
            if p > 0:
                entropy -= p * np.log2(p)
                
        return entropy # 符号化長/Total

    # --- Numba 高速化対象外の関数 ---
    @handle_zero_std
    def _lempel_ziv_complexity_vectorized(self, x: np.ndarray) -> float:
        """Lempel-Ziv複雑度計算（ベクトル化版）"""
        try:
            x_binary = (x > np.median(x)).astype(int)
            
            sub_strings = set()
            n = len(x_binary)
            i = 0
            complexity = 0
            while i < n:
                j = i
                while j < n:
                    sub = tuple(x_binary[i:j+1])
                    if sub in sub_strings:
                        j += 1
                    else:
                        sub_strings.add(sub)
                        complexity += 1
                        break
                i = j + 1
            
            return complexity / n if n > 0 else 0.0
        except:
            return 0.0
    
    @handle_zero_std
    def _kolmogorov_complexity_approx_vectorized(self, x: np.ndarray) -> float:
        """Kolmogorov複雑度近似計算（ベクトル化版）"""
        try:
            import zlib
            x_str = ''.join(map(str, (x > np.median(x)).astype(int)))
            compressed_size = len(zlib.compress(x_str.encode()))
            return compressed_size / len(x_str) if len(x_str) > 0 else 0.0
        except:
            return 0.0
    
    @handle_zero_std
    def _mutual_information_safe_vectorized(self, x: pd.Series, volume_full: pd.Series, lag: int) -> float:
        """インデックス安全な相互情報量計算"""
        try:
            if len(x) <= lag: return 0.0
            valid_indices = x.index
            if lag > 0:
                if len(valid_indices) <= lag: return 0.0
                x_lagged = x.iloc[:-lag].values
                if len(volume_full) > max(valid_indices) + lag:
                    y_lagged = volume_full.iloc[valid_indices[lag:]].values
                else:
                    available_end = min(len(volume_full), len(x))
                    if available_end <= lag: return 0.0
                    x_lagged = x.iloc[:available_end-lag].values
                    y_lagged = volume_full.iloc[lag:available_end].values
            else:
                x_lagged = x.values
                y_lagged = volume_full.iloc[:len(x)].values if len(volume_full) >= len(x) else volume_full.values

            min_len = min(len(x_lagged), len(y_lagged))
            if min_len < 5: return 0.0
            
            x_lagged, y_lagged = x_lagged[:min_len], y_lagged[:min_len]
            bins = min(5, min_len//10)
            if bins < 2: return 0.0
            
            x_binned = np.digitize(x_lagged, np.histogram(x_lagged, bins=bins)[1])
            y_binned = np.digitize(y_lagged, np.histogram(y_lagged, bins=bins)[1])
            
            from sklearn.metrics import mutual_info_score
            return mutual_info_score(x_binned, y_binned)
        except:
            try:
                correlation = np.corrcoef(x.values, volume_full.iloc[:len(x)].values)[0, 1]
                return abs(correlation) if not np.isnan(correlation) else 0.0
            except:
                return 0.0
    
    @handle_zero_std
    def _transfer_entropy_vectorized(self, x: np.ndarray) -> float:
        """転送エントロピー計算（ベクトル化版）"""
        try:
            if len(x) < 3:
                return 0.0
            
            x1, x2, x3 = x[:-2], x[1:-1], x[2:]
            
            correlation = np.corrcoef([x1, x2, x3])
            te_approx = -0.5 * np.log(1 - correlation[0, 2]**2 + 1e-8)
            
            return max(0.0, te_approx)
        except:
            return 0.0
        
# ========== 11. 生物学・医学系特徴量 ==========

    # --- 35/57: Biometric Features ---
    def _biometric_features_logic(self, ohlcv: Dict[str, pd.Series], timeframe: str) -> Dict[str, np.ndarray]:
        """生体測定特徴量の具体的な計算ロジック"""
        features = {}
        close_series = ohlcv['close']
        high_series = ohlcv['high']
        low_series = ohlcv['low']
        volume_series = ohlcv['volume']
        suffix = f'_{timeframe}' if self.multi_timeframe else ''
        returns = np.log(close_series / close_series.shift(1)).fillna(0)
        
        features[f'hrv_rmssd{suffix}'] = returns.rolling(window=50).apply(lambda x: np.sqrt(np.mean(np.diff(x)**2)), raw=True).values
        features[f'hrv_pnn50{suffix}'] = returns.rolling(window=50).apply(lambda x: np.sum(np.abs(np.diff(x)) > 0.02) / (len(x) - 1) if len(x) > 1 else 0, raw=True).values
        
        pressure_proxy = (high_series + low_series) / 2
        features[f'systolic_trend{suffix}'] = pressure_proxy.rolling(window=20).apply(self._safe_polyfit_slope, raw=False).values
        features[f'diastolic_variability{suffix}'] = pressure_proxy.rolling(window=20).std().values
        
        vol_pattern = returns.rolling(window=20).std()
        features[f'respiratory_rate{suffix}'] = vol_pattern.rolling(window=100).apply(self._estimate_dominant_period, raw=False).values
        
        features[f'price_volume_bmi{suffix}'] = (close_series / (volume_series.rolling(window=20).mean() + 1e-8)).values
        
        ma_20 = close_series.rolling(window=20).mean()
        features[f'temperature_deviation{suffix}'] = ((close_series - ma_20) / ma_20 * 100).values
        
        features[f'glucose_volatility{suffix}'] = returns.rolling(window=10).std().values
        return features

    def _calculate_biometric_features(self, df: pd.DataFrame) -> Dict[str, np.ndarray]:
        """生体測定特徴量（共通処理ヘルパー使用）"""
        logic = partial(self._biometric_features_logic)
        return self._process_feature_group(df, logic, "Biometric Features")

    # --- 36/57: Physiological Features ---
    def _physiological_features_logic(self, ohlcv: Dict[str, pd.Series], timeframe: str) -> Dict[str, np.ndarray]:
        """生理学的特徴量の具体的な計算ロジック"""
        features = {}
        close_series = ohlcv['close']
        volume_series = ohlcv['volume']
        suffix = f'_{timeframe}' if self.multi_timeframe else ''
        returns = np.log(close_series / close_series.shift(1)).fillna(0)
        
        features[f'metabolic_rate{suffix}'] = (returns.abs() * volume_series).rolling(window=20).mean().values
        features[f'muscle_activity{suffix}'] = returns.rolling(window=10).apply(lambda x: np.sum(np.abs(x) > np.std(x)), raw=True).values
        features[f'neural_activity{suffix}'] = returns.rolling(window=30).apply(lambda x: len(np.where(np.diff(np.sign(x)))[0]), raw=True).values
        features[f'endocrine_activity{suffix}'] = returns.rolling(window=60).apply(lambda x: np.sum(np.abs(x) > 2*np.std(x)), raw=True).values
        features[f'immune_response{suffix}'] = volume_series.rolling(window=30).apply(lambda x: np.sum(x > np.percentile(x, 90)) / len(x), raw=True).values
        return features

    def _calculate_physiological_features(self, df: pd.DataFrame) -> Dict[str, np.ndarray]:
        """生理学的特徴量（共通処理ヘルパー使用）"""
        logic = partial(self._physiological_features_logic)
        return self._process_feature_group(df, logic, "Physiological Features")

    # --- 37/57: Circadian Features ---
    def _circadian_features_logic(self, ohlcv: Dict[str, pd.Series], timeframe: str) -> Dict[str, np.ndarray]:
        """概日リズム特徴量の具体的な計算ロジック"""
        features = {}
        close_series = ohlcv['close']
        suffix = f'_{timeframe}' if self.multi_timeframe else ''
        returns = np.log(close_series / close_series.shift(1)).fillna(0)
        
        features[f'circadian_rhythm{suffix}'] = returns.rolling(window=1440).apply(lambda x: np.abs(np.fft.fft(x)[1440//24]) if len(x) >= 1440//24 else 0, raw=True).values
        features[f'diurnal_pattern{suffix}'] = close_series.rolling(window=480).std().values
        features[f'weekly_rhythm{suffix}'] = returns.rolling(window=10080).apply(lambda x: np.abs(np.fft.fft(x)[10080//1440]) if len(x) >= 10080//1440 else 0, raw=True).values
        features[f'monthly_rhythm{suffix}'] = returns.rolling(window=43200).apply(lambda x: np.abs(np.fft.fft(x)[43200//1440]) if len(x) >= 43200//1440 else 0, raw=True).values
        return features

    def _calculate_circadian_features(self, df: pd.DataFrame) -> Dict[str, np.ndarray]:
        """概日リズム特徴量（共通処理ヘルパー使用）"""
        logic = partial(self._circadian_features_logic)
        return self._process_feature_group(df, logic, "Circadian Features")

    # ========== 12. 心理学・認知科学系特徴量 ==========

    # --- 38/57: Psychological Features ---
    def _psychological_features_logic(self, ohlcv: Dict[str, pd.Series], timeframe: str) -> Dict[str, np.ndarray]:
        """心理学特徴量の具体的な計算ロジック"""
        features = {}
        close_series = ohlcv['close']
        volume_series = ohlcv['volume']
        suffix = f'_{timeframe}' if self.multi_timeframe else ''
        returns = np.log(close_series / close_series.shift(1)).fillna(0)

        features[f'fear_greed_index{suffix}'] = returns.rolling(window=30).apply(lambda x: (np.sum(x > 0) - np.sum(x < 0)) / len(x), raw=True).values
        features[f'confirmation_bias{suffix}'] = returns.rolling(window=50).apply(self._measure_trend_persistence, raw=True).values
        features[f'anchoring_effect{suffix}'] = close_series.rolling(window=30).apply(self._calculate_anchoring_effect, raw=False).values
        features[f'loss_aversion{suffix}'] = returns.rolling(window=30).apply(lambda x: np.abs(np.mean(x[x < 0])) / (np.mean(x[x > 0]) + 1e-8) if np.any(x > 0) and np.any(x < 0) else 1, raw=True).values
        
        volume_ma = volume_series.rolling(window=20).mean()
        features[f'herd_behavior{suffix}'] = (volume_series / (volume_ma + 1e-8)).rolling(window=10).apply(lambda x: np.sum(x > 1.5) / len(x), raw=True).values
        
        features[f'attention_span{suffix}'] = returns.rolling(window=40).apply(self._calculate_attention_metric, raw=True).values
        features[f'stress_level{suffix}'] = returns.rolling(window=20).apply(lambda x: np.sum(np.abs(x) > 2 * np.std(x)) / len(x), raw=True).values
        features[f'decision_pattern{suffix}'] = returns.rolling(window=25).apply(lambda x: np.sum(np.abs(np.diff(np.sign(x)))) / len(x), raw=True).values
        features[f'learning_effect{suffix}'] = returns.rolling(window=100).apply(self._calculate_learning_effect, raw=True).values
        return features

    def _calculate_psychological_features(self, df: pd.DataFrame) -> Dict[str, np.ndarray]:
        """心理学特徴量（共通処理ヘルパー使用）"""
        logic = partial(self._psychological_features_logic)
        return self._process_feature_group(df, logic, "Psychological Features")

    # --- 39/57: Behavioral Features ---
    def _behavioral_features_logic(self, ohlcv: Dict[str, pd.Series], timeframe: str) -> Dict[str, np.ndarray]:
        """行動経済学特徴量の具体的な計算ロジック"""
        features = {}
        close_series = ohlcv['close']
        volume_series = ohlcv['volume']
        suffix = f'_{timeframe}' if self.multi_timeframe else ''
        returns = np.log(close_series / close_series.shift(1)).fillna(0)

        features[f'prospect_theory{suffix}'] = returns.rolling(window=50).apply(lambda x: np.mean(np.where(x >= 0, x, 2.25 * x)), raw=True).values
        features[f'mental_accounting{suffix}'] = returns.rolling(window=30).apply(lambda x: np.std(x[x > 0]) / (np.std(x[x < 0]) + 1e-8) if np.any(x > 0) and np.any(x < 0) else 1, raw=True).values
        features[f'herding_behavior{suffix}'] = volume_series.rolling(window=20).apply(lambda x: np.sum(x > np.percentile(x, 75)) / len(x), raw=True).values
        features[f'overconfidence{suffix}'] = returns.rolling(window=40).apply(lambda x: np.mean(np.abs(x)) / (np.std(x) + 1e-8), raw=True).values
        features[f'representativeness{suffix}'] = returns.rolling(window=30).apply(lambda x: len(np.where(np.diff(np.sign(x)))[0]) / len(x), raw=True).values
        features[f'availability_heuristic{suffix}'] = returns.rolling(window=50).apply(lambda x: np.sum(np.abs(x[-10:]) > np.std(x)) / 10, raw=True).values
        return features

    def _calculate_behavioral_features(self, df: pd.DataFrame) -> Dict[str, np.ndarray]:
        """行動経済学特徴量（共通処理ヘルパー使用）"""
        logic = partial(self._behavioral_features_logic)
        return self._process_feature_group(df, logic, "Behavioral Features")

 # ========== 13. フラクタル・カオス・乱流系特徴量 ==========

    # --- 40/57: Fractal Features ---
    def _fractal_features_logic(self, ohlcv: Dict[str, pd.Series], timeframe: str) -> Dict[str, np.ndarray]:
        """フラクタル特徴量の具体的な計算ロジック"""
        features = {}
        close_series = ohlcv['close']
        suffix = f'_{timeframe}' if self.multi_timeframe else ''

        features[f'hurst_exponent{suffix}'] = close_series.rolling(window=100).apply(self._calculate_hurst_exponent, raw=False).values
        features[f'fractal_dimension{suffix}'] = close_series.rolling(window=100).apply(self._calculate_fractal_dimension, raw=False).values
        
        for scale in self.self_similarity_scales:
            features[f'self_similarity_{scale}{suffix}'] = close_series.rolling(window=100).apply(lambda x: self._measure_self_similarity(x, scale), raw=False).values
        
        features[f'dfa_alpha{suffix}'] = close_series.rolling(window=200).apply(self._calculate_dfa_alpha, raw=False).values
        features[f'multifractal_width{suffix}'] = close_series.rolling(window=150).apply(self._calculate_multifractal_width, raw=False).values
        features[f'fractal_efficiency{suffix}'] = close_series.rolling(window=80).apply(self._calculate_fractal_efficiency, raw=False).values
        return features

    def _calculate_fractal_features(self, df: pd.DataFrame) -> Dict[str, np.ndarray]:
        """フラクタル特徴量（共通処理ヘルパー使用）"""
        logic = partial(self._fractal_features_logic)
        return self._process_feature_group(df, logic, "Fractal Features")

    # --- 41/57: Chaos Features ---
    def _chaos_features_logic(self, ohlcv: Dict[str, pd.Series], timeframe: str) -> Dict[str, np.ndarray]:
        """カオス理論特徴量の具体的な計算ロジック"""
        features = {}
        close_series = ohlcv['close']
        suffix = f'_{timeframe}' if self.multi_timeframe else ''
        returns = np.log(close_series / close_series.shift(1)).fillna(0)

        features[f'lyapunov_exponent{suffix}'] = returns.rolling(window=100).apply(self._estimate_lyapunov_exponent, raw=True).values
        features[f'correlation_dimension{suffix}'] = close_series.rolling(window=100).apply(self._calculate_correlation_dimension, raw=False).values
        features[f'chaos_degree{suffix}'] = returns.rolling(window=50).apply(self._measure_chaos_degree, raw=True).values
        features[f'strange_attractor{suffix}'] = close_series.rolling(window=100).apply(self._detect_strange_attractor, raw=False).values
        features[f'poincare_section{suffix}'] = returns.rolling(window=80).apply(self._analyze_poincare_section, raw=True).values
        features[f'nonlinear_dynamics{suffix}'] = returns.rolling(window=60).apply(self._measure_nonlinear_dynamics, raw=True).values
        return features

    def _calculate_chaos_features(self, df: pd.DataFrame) -> Dict[str, np.ndarray]:
        """カオス理論特徴量（共通処理ヘルパー使用）"""
        logic = partial(self._chaos_features_logic)
        return self._process_feature_group(df, logic, "Chaos Features")

    # --- 42/57: Turbulence Features ---
    def _turbulence_features_logic(self, ohlcv: Dict[str, pd.Series], timeframe: str) -> Dict[str, np.ndarray]:
        """乱流特徴量の具体的な計算ロジック"""
        features = {}
        close_series = ohlcv['close']
        volume_series = ohlcv['volume']
        suffix = f'_{timeframe}' if self.multi_timeframe else ''
        returns = np.log(close_series / close_series.shift(1)).fillna(0)

        features[f'reynolds_number{suffix}'] = (volume_series * np.abs(returns)).rolling(window=20).apply(
            lambda x: np.mean(x) / (np.std(x) + 1e-8), raw=True
        ).values
        features[f'vorticity{suffix}'] = returns.rolling(window=20).apply(self._calculate_vorticity, raw=True).values
        features[f'turbulence_intensity{suffix}'] = returns.rolling(window=30).apply(
            lambda x: np.std(x) / (np.abs(np.mean(x)) + 1e-8), raw=True
        ).values
        features[f'energy_cascade{suffix}'] = returns.rolling(window=50).apply(self._measure_energy_cascade, raw=True).values
        features[f'kolmogorov_scale{suffix}'] = returns.rolling(window=40).apply(self._estimate_kolmogorov_scale, raw=True).values
        features[f'taylor_scale{suffix}'] = returns.rolling(window=60).apply(self._estimate_taylor_scale, raw=True).values
        return features

    def _calculate_turbulence_features(self, df: pd.DataFrame) -> Dict[str, np.ndarray]:
        """乱流特徴量（共通処理ヘルパー使用）"""
        logic = partial(self._turbulence_features_logic)
        return self._process_feature_group(df, logic, "Turbulence Features")
    
    
# ========== 14. 経済学・金融系特徴量 ==========

    # --- 43/57: Econometric Features ---
    def _econometric_features_logic(self, ohlcv: Dict[str, pd.Series], timeframe: str) -> Dict[str, np.ndarray]:
        """計量経済学特徴量の具体的な計算ロジック"""
        features = {}
        close_series = ohlcv['close']
        volume_series = ohlcv['volume']
        suffix = f'_{timeframe}' if self.multi_timeframe else ''

        close_values = pd.Series(close_series.values, index=range(len(close_series)))
        volume_values = pd.Series(volume_series.values, index=range(len(volume_series)))
        returns = np.log(close_values / close_values.shift(1)).fillna(0)
        
        features[f'unit_root_stat{suffix}'] = returns.rolling(window=100).apply(self._adf_statistic, raw=True).values
        features[f'cointegration_stat{suffix}'] = close_values.rolling(window=100).apply(self._cointegration_test, raw=False).values
        features[f'arch_effect{suffix}'] = returns.rolling(window=50).apply(self._test_arch_effect, raw=True).values
        features[f'granger_causality{suffix}'] = close_values.rolling(window=100).apply(
            lambda x: self._granger_causality_safe_test(x, volume_values), raw=False
        ).values
        features[f'market_efficiency{suffix}'] = returns.rolling(window=50).apply(self._test_market_efficiency, raw=True).values
        features[f'mean_reversion_speed{suffix}'] = close_values.rolling(window=50).apply(self._estimate_mean_reversion_speed, raw=False).values
        features[f'volatility_clustering{suffix}'] = returns.rolling(window=40).apply(self._volatility_clustering, raw=True).values
        features[f'long_memory{suffix}'] = returns.rolling(window=100).apply(self._test_long_memory, raw=True).values
        return features

    def _calculate_econometric_features(self, df: pd.DataFrame) -> Dict[str, np.ndarray]:
        """計量経済学特徴量（共通処理ヘルパー使用）"""
        logic = partial(self._econometric_features_logic)
        return self._process_feature_group(df, logic, "Econometric Features")

    # --- 44/57: Risk Features ---
    def _risk_features_logic(self, ohlcv: Dict[str, pd.Series], timeframe: str) -> Dict[str, np.ndarray]:
        """リスク管理特徴量の具体的な計算ロジック"""
        features = {}
        close_series = ohlcv['close']
        suffix = f'_{timeframe}' if self.multi_timeframe else ''
        returns = np.log(close_series / close_series.shift(1)).fillna(0)

        for confidence in self.var_confidence_levels:
            features[f'var_{int(confidence*100)}{suffix}'] = returns.rolling(window=50).apply(
                lambda x: np.percentile(x, (1-confidence)*100), raw=True
            ).values
            features[f'cvar_{int(confidence*100)}{suffix}'] = returns.rolling(window=50).apply(
                lambda x: np.mean(x[x <= np.percentile(x, (1-confidence)*100)]), raw=True
            ).values
            
        features[f'max_drawdown{suffix}'] = close_series.rolling(window=100).apply(
            lambda x: (x.min() - x.max()) / x.max(), raw=False
        ).values
        features[f'sharpe_ratio{suffix}'] = returns.rolling(window=50).apply(
            lambda x: np.mean(x) / (np.std(x) + 1e-8), raw=True
        ).values
        features[f'sortino_ratio{suffix}'] = returns.rolling(window=50).apply(
            lambda x: np.mean(x) / (np.std(x[x < 0]) + 1e-8), raw=True
        ).values
        features[f'calmar_ratio{suffix}'] = returns.rolling(window=100).apply(
            lambda x: np.mean(x) * 252 / (abs(np.min(np.cumsum(x))) + 1e-8), raw=True
        ).values
        features[f'information_ratio{suffix}'] = returns.rolling(window=60).apply(
            lambda x: np.mean(x) / (np.std(x - np.mean(x)) + 1e-8), raw=True
        ).values
        features[f'tail_ratio{suffix}'] = returns.rolling(window=50).apply(
            lambda x: np.percentile(x, 95) / abs(np.percentile(x, 5)), raw=True
        ).values
        return features

    def _calculate_risk_features(self, df: pd.DataFrame) -> Dict[str, np.ndarray]:
        """リスク管理特徴量（共通処理ヘルパー使用）"""
        logic = partial(self._risk_features_logic)
        return self._process_feature_group(df, logic, "Risk Features")

    # --- 45/57: Game Theory Features ---
    def _game_theory_features_logic(self, ohlcv: Dict[str, pd.Series], timeframe: str) -> Dict[str, np.ndarray]:
        """ゲーム理論特徴量の具体的な計算ロジック"""
        features = {}
        close_series = ohlcv['close']
        suffix = f'_{timeframe}' if self.multi_timeframe else ''
        returns = np.log(close_series / close_series.shift(1)).fillna(0)

        features[f'nash_equilibrium{suffix}'] = returns.rolling(window=50).apply(
            lambda x: 1 - np.std(x) / (np.abs(np.mean(x)) + 1e-8), raw=True
        ).values
        features[f'cooperation_index{suffix}'] = returns.rolling(window=30).apply(
            self._cooperation_index, raw=True
        ).values
        features[f'strategy_diversity{suffix}'] = returns.rolling(window=50).apply(
            lambda x: len(np.unique(np.sign(x))) / 3, raw=True
        ).values
        features[f'zero_sum_indicator{suffix}'] = returns.rolling(window=40).apply(
            lambda x: abs(np.sum(x)) / (np.sum(np.abs(x)) + 1e-8), raw=True
        ).values
        features[f'prisoners_dilemma{suffix}'] = returns.rolling(window=30).apply(
            lambda x: self._calculate_prisoners_dilemma_index(x), raw=True
        ).values
        features[f'minimax_strategy{suffix}'] = returns.rolling(window=25).apply(
            lambda x: np.min(x) / (np.max(x) + 1e-8), raw=True
        ).values
        return features

    def _calculate_game_theory_features(self, df: pd.DataFrame) -> Dict[str, np.ndarray]:
        """ゲーム理論特徴量（共通処理ヘルパー使用）"""
        logic = partial(self._game_theory_features_logic)
        return self._process_feature_group(df, logic, "Game Theory Features")

# ========== 15. 分子・結晶学系特徴量 ==========

    # --- 46/57: Molecular Features ---
    def _molecular_features_logic(self, ohlcv: Dict[str, pd.Series], timeframe: str) -> Dict[str, np.ndarray]:
        """分子科学特徴量の具体的な計算ロジック"""
        features = {}
        close_series = ohlcv['close']
        suffix = f'_{timeframe}' if self.multi_timeframe else ''
        returns = np.log(close_series / close_series.shift(1)).fillna(0)

        features[f'molecular_vibration{suffix}'] = returns.rolling(window=30).apply(lambda x: np.sum(np.abs(np.fft.fft(x)[:len(x)//2])), raw=True).values
        features[f'bond_energy{suffix}'] = returns.rolling(window=20).apply(lambda x: -np.sum(x * np.log(np.abs(x) + 1e-8)), raw=True).values
        features[f'electron_density{suffix}'] = close_series.rolling(window=50).apply(lambda x: np.trapz(x**2) / len(x), raw=False).values
        features[f'molecular_orbital{suffix}'] = returns.rolling(window=40).apply(lambda x: np.sum(x**2 * np.arange(1, len(x)+1)), raw=True).values
        features[f'chemical_potential{suffix}'] = close_series.rolling(window=30).apply(lambda x: (x.iloc[-1] - x.iloc[0]) / len(x), raw=False).values
        features[f'intermolecular_force{suffix}'] = close_series.rolling(window=25).apply(lambda x: np.sum(1 / (np.diff(x)**2 + 1e-8)), raw=False).values
        return features

    def _calculate_molecular_features(self, df: pd.DataFrame) -> Dict[str, np.ndarray]:
        """分子科学特徴量（共通処理ヘルパー使用）"""
        logic = partial(self._molecular_features_logic)
        return self._process_feature_group(df, logic, "Molecular Features")

    # --- 47/57: Crystallographic Features ---
    def _crystallographic_features_logic(self, ohlcv: Dict[str, pd.Series], timeframe: str) -> Dict[str, np.ndarray]:
        """結晶学特徴量の具体的な計算ロジック"""
        features = {}
        close_series = ohlcv['close']
        suffix = f'_{timeframe}' if self.multi_timeframe else ''

        features[f'lattice_constant{suffix}'] = close_series.rolling(window=100).apply(lambda x: np.mean(np.abs(np.diff(x))), raw=False).values
        features[f'crystal_structure{suffix}'] = close_series.rolling(window=50).apply(lambda x: np.std(x) / np.mean(x) if np.mean(x) != 0 else 0, raw=False).values
        features[f'xray_diffraction{suffix}'] = close_series.rolling(window=64).apply(lambda x: np.max(np.abs(np.fft.fft(x))), raw=False).values
        features[f'defect_density{suffix}'] = close_series.rolling(window=30).apply(lambda x: np.sum(np.abs(x - x.mean()) > 2*x.std()) / len(x), raw=False).values
        features[f'lattice_vibration{suffix}'] = close_series.rolling(window=40).apply(lambda x: np.var(np.diff(x)), raw=False).values
        features[f'crystal_orientation{suffix}'] = close_series.rolling(window=60).apply(lambda x: np.max(np.correlate(x[:len(x)//2], x[len(x)//2:], mode='valid')), raw=False).values
        return features

    def _calculate_crystallographic_features(self, df: pd.DataFrame) -> Dict[str, np.ndarray]:
        """結晶学特徴量（共通処理ヘルパー使用）"""
        logic = partial(self._crystallographic_features_logic)
        return self._process_feature_group(df, logic, "Crystallographic Features")

# ========== 16. その他学際的特徴量 ==========

    # --- 48/57: Network Features ---
    def _network_features_logic(self, ohlcv: Dict[str, pd.Series], timeframe: str) -> Dict[str, np.ndarray]:
        """ネットワーク科学特徴量の具体的な計算ロジック"""
        features = {}
        close_series = ohlcv['close']
        suffix = f'_{timeframe}' if self.multi_timeframe else ''

        features[f'network_density{suffix}'] = close_series.rolling(window=50).apply(self._calculate_network_density, raw=False).values
        features[f'centrality_measure{suffix}'] = close_series.rolling(window=30).apply(self._calculate_centrality, raw=False).values
        features[f'clustering_coefficient{suffix}'] = close_series.rolling(window=40).apply(self._calculate_clustering_coefficient, raw=False).values
        features[f'betweenness_centrality{suffix}'] = close_series.rolling(window=35).apply(self._calculate_betweenness_centrality, raw=False).values
        features[f'eigenvector_centrality{suffix}'] = close_series.rolling(window=45).apply(self._calculate_eigenvector_centrality, raw=False).values
        return features

    def _calculate_network_features(self, df: pd.DataFrame) -> Dict[str, np.ndarray]:
        """ネットワーク科学特徴量（共通処理ヘルパー使用）"""
        logic = partial(self._network_features_logic)
        return self._process_feature_group(df, logic, "Network Features")

    # --- 49/57: Social Physics Features ---
    def _social_physics_features_logic(self, ohlcv: Dict[str, pd.Series], timeframe: str) -> Dict[str, np.ndarray]:
        """社会物理学特徴量の具体的な計算ロジック"""
        features = {}
        close_series = ohlcv['close']
        volume_series = ohlcv['volume']
        suffix = f'_{timeframe}' if self.multi_timeframe else ''

        features[f'social_influence{suffix}'] = volume_series.rolling(window=20).apply(lambda x: np.sum(x > np.mean(x)) / len(x), raw=True).values
        features[f'crowd_density{suffix}'] = volume_series.rolling(window=10).apply(lambda x: np.std(x) / np.mean(x) if np.mean(x) > 0 else 0, raw=True).values
        features[f'social_inertia{suffix}'] = close_series.rolling(window=30).apply(self._social_inertia, raw=False).values
        features[f'collective_behavior{suffix}'] = volume_series.rolling(window=25).apply(lambda x: np.sum(x > np.percentile(x, 80)) / len(x), raw=True).values
        features[f'social_contagion{suffix}'] = close_series.rolling(window=40).apply(lambda x: len(np.where(np.diff(np.sign(np.diff(x))))[0]) / len(x), raw=False).values
        return features

    def _calculate_social_physics_features(self, df: pd.DataFrame) -> Dict[str, np.ndarray]:
        """社会物理学特徴量（共通処理ヘルパー使用）"""
        logic = partial(self._social_physics_features_logic)
        return self._process_feature_group(df, logic, "Social Physics Features")

    # --- 50/57: Acoustic Features ---
    def _acoustic_features_logic(self, ohlcv: Dict[str, pd.Series], timeframe: str) -> Dict[str, np.ndarray]:
        """音響学特徴量の具体的な計算ロジック"""
        features = {}
        close_series = ohlcv['close']
        suffix = f'_{timeframe}' if self.multi_timeframe else ''
        returns = np.log(close_series / close_series.shift(1)).fillna(0)

        features[f'acoustic_power{suffix}'] = returns.rolling(window=50).apply(lambda x: np.sum(x**2), raw=True).values
        features[f'acoustic_frequency{suffix}'] = returns.rolling(window=64).apply(lambda x: np.argmax(np.abs(np.fft.fft(x)[:len(x)//2])), raw=True).values
        features[f'amplitude_modulation{suffix}'] = returns.rolling(window=32).apply(lambda x: np.std(np.abs(x)), raw=True).values
        features[f'phase_modulation{suffix}'] = returns.rolling(window=32).apply(lambda x: np.std(np.unwrap(np.angle(np.fft.fft(x)))), raw=True).values
        features[f'acoustic_echo{suffix}'] = returns.rolling(window=60).apply(lambda x: np.max(np.correlate(x[:len(x)//2], x[len(x)//2:], mode='valid')), raw=True).values
        return features

    def _calculate_acoustic_features(self, df: pd.DataFrame) -> Dict[str, np.ndarray]:
        """音響学特徴量（共通処理ヘルパー使用）"""
        logic = partial(self._acoustic_features_logic)
        return self._process_feature_group(df, logic, "Acoustic Features")

    # --- 51/57: Linguistic Features ---
    def _linguistic_features_logic(self, ohlcv: Dict[str, pd.Series], timeframe: str) -> Dict[str, np.ndarray]:
        """言語学特徴量の具体的な計算ロジック"""
        features = {}
        close_series = ohlcv['close']
        suffix = f'_{timeframe}' if self.multi_timeframe else ''
        returns = np.log(close_series / close_series.shift(1)).fillna(0)

        features[f'vocabulary_diversity{suffix}'] = returns.rolling(window=100).apply(lambda x: len(np.unique(np.digitize(x, np.histogram(x, bins=10)[1]))) / 10, raw=True).values
        features[f'sentence_structure{suffix}'] = returns.rolling(window=50).apply(lambda x: np.mean(np.abs(np.diff(x))), raw=True).values
        features[f'linguistic_complexity{suffix}'] = returns.rolling(window=80).apply(lambda x: len(np.where(np.abs(x) > np.std(x))[0]) / len(x), raw=True).values
        features[f'word_order{suffix}'] = returns.rolling(window=40).apply(lambda x: np.sum(np.diff(np.argsort(x)) != 0) / len(x), raw=True).values
        features[f'prosody{suffix}'] = returns.rolling(window=30).apply(lambda x: np.std(np.abs(x)), raw=True).values
        return features

    def _calculate_linguistic_features(self, df: pd.DataFrame) -> Dict[str, np.ndarray]:
        """言語学特徴量（共通処理ヘルパー使用）"""
        logic = partial(self._linguistic_features_logic)
        return self._process_feature_group(df, logic, "Linguistic Features")

    # --- 52/57: Aesthetic Features ---
    def _aesthetic_features_logic(self, ohlcv: Dict[str, pd.Series], timeframe: str) -> Dict[str, np.ndarray]:
        """美学特徴量の具体的な計算ロジック"""
        features = {}
        close_series = ohlcv['close']
        suffix = f'_{timeframe}' if self.multi_timeframe else ''

        features[f'golden_ratio_adherence{suffix}'] = close_series.rolling(window=50).apply(self._measure_golden_ratio_adherence, raw=False).values
        features[f'symmetry_measure{suffix}'] = close_series.rolling(window=40).apply(self._measure_symmetry, raw=False).values
        features[f'aesthetic_harmony{suffix}'] = close_series.rolling(window=60).apply(self._measure_aesthetic_harmony, raw=False).values
        features[f'proportional_beauty{suffix}'] = close_series.rolling(window=35).apply(self._measure_proportional_beauty, raw=False).values
        features[f'visual_balance{suffix}'] = close_series.rolling(window=45).apply(self._measure_visual_balance, raw=False).values
        return features

    def _calculate_aesthetic_features(self, df: pd.DataFrame) -> Dict[str, np.ndarray]:
        """美学特徴量（共通処理ヘルパー使用）"""
        logic = partial(self._aesthetic_features_logic)
        return self._process_feature_group(df, logic, "Aesthetic Features")

    # --- 53/57: Musical Features ---
    def _musical_features_logic(self, ohlcv: Dict[str, pd.Series], timeframe: str) -> Dict[str, np.ndarray]:
        """音楽理論特徴量の具体的な計算ロジック"""
        features = {}
        close_series = ohlcv['close']
        suffix = f'_{timeframe}' if self.multi_timeframe else ''
        returns = np.log(close_series / close_series.shift(1)).fillna(0)

        features[f'tonality{suffix}'] = returns.rolling(window=100).apply(lambda x: np.std(x) / (np.abs(np.mean(x)) + 1e-8), raw=True).values
        features[f'rhythm_pattern{suffix}'] = returns.rolling(window=50).apply(lambda x: len(np.where(np.abs(x) > np.std(x))[0]) / len(x), raw=True).values
        features[f'harmony{suffix}'] = returns.rolling(window=60).apply(self._analyze_harmony, raw=True).values
        features[f'melody_contour{suffix}'] = close_series.rolling(window=40).apply(lambda x: np.sum(np.diff(x) > 0) / len(x), raw=False).values
        features[f'musical_tension{suffix}'] = returns.rolling(window=30).apply(lambda x: np.max(x) - np.min(x), raw=True).values
        features[f'tempo{suffix}'] = returns.rolling(window=25).apply(lambda x: len(np.where(np.diff(np.sign(x)))[0]), raw=True).values
        return features

    def _calculate_musical_features(self, df: pd.DataFrame) -> Dict[str, np.ndarray]:
        """音楽理論特徴量（共通処理ヘルパー使用）"""
        logic = partial(self._musical_features_logic)
        return self._process_feature_group(df, logic, "Musical Features")

    # --- 54/57: Astronomical Features ---
    def _astronomical_features_logic(self, ohlcv: Dict[str, pd.Series], timeframe: str) -> Dict[str, np.ndarray]:
        """天文学特徴量の具体的な計算ロジック"""
        features = {}
        close_series = ohlcv['close']
        suffix = f'_{timeframe}' if self.multi_timeframe else ''

        features[f'orbital_mechanics{suffix}'] = close_series.rolling(window=100).apply(self._calculate_orbital_indicator, raw=False).values
        features[f'gravitational_wave{suffix}'] = close_series.rolling(window=80).apply(self._detect_gravitational_wave_pattern, raw=False).values
        features[f'stellar_pulsation{suffix}'] = close_series.rolling(window=60).apply(self._analyze_stellar_pulsation, raw=False).values
        features[f'planetary_motion{suffix}'] = close_series.rolling(window=120).apply(self._analyze_planetary_motion, raw=False).values
        features[f'galactic_rotation{suffix}'] = close_series.rolling(window=200).apply(self._analyze_galactic_rotation, raw=False).values
        return features

    def _calculate_astronomical_features(self, df: pd.DataFrame) -> Dict[str, np.ndarray]:
        """天文学特徴量（共通処理ヘルパー使用）"""
        logic = partial(self._astronomical_features_logic)
        return self._process_feature_group(df, logic, "Astronomical Features")

    # --- 55/57: Cosmological Features ---
    def _cosmological_features_logic(self, ohlcv: Dict[str, pd.Series], timeframe: str) -> Dict[str, np.ndarray]:
        """宇宙論特徴量の具体的な計算ロジック"""
        features = {}
        close_series = ohlcv['close']
        suffix = f'_{timeframe}' if self.multi_timeframe else ''

        features[f'cosmic_expansion{suffix}'] = close_series.rolling(window=200).apply(lambda x: (x.iloc[-1] - x.iloc[0]) / x.iloc[0] if x.iloc[0] != 0 else 0, raw=False).values
        features[f'dark_energy{suffix}'] = close_series.rolling(window=150).apply(self._estimate_dark_energy, raw=False).values
        features[f'big_bang_echo{suffix}'] = close_series.rolling(window=100).apply(self._detect_big_bang_echo, raw=False).values
        features[f'cmb_pattern{suffix}'] = close_series.rolling(window=300).apply(self._analyze_cmb_pattern, raw=False).values
        features[f'cosmic_inflation{suffix}'] = close_series.rolling(window=80).apply(self._measure_cosmic_inflation, raw=False).values
        return features

    def _calculate_cosmological_features(self, df: pd.DataFrame) -> Dict[str, np.ndarray]:
        """宇宙論特徴量（共通処理ヘルパー使用）"""
        logic = partial(self._cosmological_features_logic)
        return self._process_feature_group(df, logic, "Cosmological Features")

    # --- 56/57: Biomechanical Features ---
    def _biomechanical_features_logic(self, ohlcv: Dict[str, pd.Series], timeframe: str) -> Dict[str, np.ndarray]:
        """生体力学特徴量の具体的な計算ロジック"""
        features = {}
        close_series = ohlcv['close']
        suffix = f'_{timeframe}' if self.multi_timeframe else ''
        returns = np.log(close_series / close_series.shift(1)).fillna(0)

        features[f'kinetic_energy{suffix}'] = returns.rolling(window=20).apply(lambda x: 0.5 * np.sum(x**2), raw=True).values
        features[f'potential_energy{suffix}'] = close_series.rolling(window=20).apply(lambda x: np.sum((x - x.min())**2), raw=False).values
        features[f'muscle_force{suffix}'] = returns.rolling(window=15).apply(lambda x: np.max(np.abs(x)), raw=True).values
        features[f'joint_mobility{suffix}'] = close_series.rolling(window=30).apply(lambda x: (x.max() - x.min()) / x.mean(), raw=False).values
        features[f'biomechanical_balance{suffix}'] = returns.rolling(window=25).apply(lambda x: 1 / (np.std(x) + 1e-8), raw=True).values
        features[f'gait_pattern{suffix}'] = returns.rolling(window=40).apply(self._analyze_gait_pattern, raw=True).values
        return features

    def _calculate_biomechanical_features(self, df: pd.DataFrame) -> Dict[str, np.ndarray]:
        """生体力学特徴量（共通処理ヘルパー使用）"""
        logic = partial(self._biomechanical_features_logic)
        return self._process_feature_group(df, logic, "Biomechanical Features")

    # --- 57/57: Performance Features ---
    def _performance_features_logic(self, ohlcv: Dict[str, pd.Series], timeframe: str) -> Dict[str, np.ndarray]:
        """パフォーマンス科学特徴量の具体的な計算ロジック"""
        features = {}
        close_series = ohlcv['close']
        suffix = f'_{timeframe}' if self.multi_timeframe else ''
        returns = np.log(close_series / close_series.shift(1)).fillna(0)

        features[f'performance_consistency{suffix}'] = returns.rolling(window=30).apply(lambda x: 1 / (np.std(x) + 1e-8), raw=True).values
        features[f'peak_performance{suffix}'] = returns.rolling(window=50).apply(lambda x: np.max(x), raw=True).values
        features[f'endurance{suffix}'] = returns.rolling(window=100).apply(lambda x: np.sum(x > 0) / len(x), raw=True).values
        features[f'recovery_rate{suffix}'] = returns.rolling(window=40).apply(self._measure_recovery_rate, raw=True).values
        features[f'adaptability{suffix}'] = returns.rolling(window=60).apply(lambda x: 1 - np.corrcoef(range(len(x)), np.abs(x))[0,1] if len(x) > 1 else 0.5, raw=True).values
        features[f'explosive_power{suffix}'] = returns.rolling(window=20).apply(lambda x: np.max(np.abs(np.diff(x))), raw=True).values
        return features

    def _calculate_performance_features(self, df: pd.DataFrame) -> Dict[str, np.ndarray]:
        """パフォーマンス科学特徴量（共通処理ヘルパー使用）"""
        logic = partial(self._performance_features_logic)
        return self._process_feature_group(df, logic, "Performance Features")

# ========== 全ヘルパー関数の完全実装 ==========

    # --- Numba 高速化対象の関数 ---

    @staticmethod
    @njit(cache=True)
    def _numba_measure_trend_persistence(x: np.ndarray) -> float:
        """Numba JIT化: トレンド持続性測定"""
        if len(x) < 3 or x.std() < 1e-9:
            return 0.0
        
        signs = np.sign(x)
        if len(signs) <= 1:
            return 0.0

        # Numbaはリストのappendが遅い場合があるため、固定長の配列で処理
        runs = np.zeros(len(signs), dtype=np.int64)
        run_idx = 0
        current_run = 1
        
        for i in range(1, len(signs)):
            if signs[i] == signs[i-1]:
                current_run += 1
            else:
                runs[run_idx] = current_run
                run_idx += 1
                current_run = 1
        runs[run_idx] = current_run
        run_idx += 1
        
        return np.mean(runs[:run_idx]) / len(x)

    @staticmethod
    @njit(cache=True)
    def _numba_calculate_attention_metric(x: np.ndarray) -> float:
        """Numba JIT化: 注意力メトリック計算"""
        if len(x) < 5 or x.std() < 1e-9:
            return 0.0
        
        changes = np.abs(np.diff(x))
        if len(changes) == 0:
            return 0.0
        threshold = np.mean(changes) + np.std(changes)
        attention_events = np.sum(changes > threshold)
        
        return attention_events / len(changes)

    @staticmethod
    @njit(cache=True)
    def _numba_calculate_hurst_exponent(x_values: np.ndarray) -> float:
        """Numba JIT化: Hurst指数計算"""
        n = len(x_values)
        if n < 20:
            return 0.5
        
        max_lag = min(20, n // 4)
        lags = np.arange(2, max_lag)
        
        rs_values = np.zeros(len(lags), dtype=np.float64)
        
        for i, lag in enumerate(lags):
            y = x_values[:(n // lag) * lag].reshape(-1, lag)
            z = np.cumsum(y - np.mean(y), axis=1)
            r = np.max(z, axis=1) - np.min(z, axis=1)
            s = np.std(y, axis=1)
            
            non_zero_s = s > 1e-9
            if np.sum(non_zero_s) > 0:
                rs_values[i] = np.mean(r[non_zero_s] / s[non_zero_s])

        valid_rs = rs_values > 1e-9
        if np.sum(valid_rs) < 2:
            return 0.5
            
        log_lags = np.log(lags[valid_rs])
        log_rs = np.log(rs_values[valid_rs])
        
        p = np.polyfit(log_lags, log_rs, 1)
        hurst = p[0]
        
        return min(1.0, max(0.0, hurst))

    @staticmethod
    @njit(cache=True)
    def _numba_calculate_fractal_dimension(x_values: np.ndarray) -> float:
        """Numba JIT化: フラクタル次元計算（Higuchi法）"""
        n = len(x_values)
        if n < 10:
            return 1.5

        k_max = int(n / 4)
        if k_max < 2:
             return 1.5
        
        l_k = np.zeros(k_max -1)
        k_values = np.arange(1, k_max)

        for k in k_values:
            l_m_k = 0.0
            for m in range(k):
                n_max = (n - (m + 1)) // k
                norm_factor = (n - 1) / (n_max * k)
                l_m_k += np.sum(np.abs(np.diff(x_values[m::k]))) * norm_factor
            l_k[k-1] = l_m_k / k
        
        valid_l_k = l_k > 1e-9
        if np.sum(valid_l_k) < 2:
            return 1.5

        log_l = np.log(l_k[valid_l_k])
        log_k = np.log(1.0 / k_values[valid_l_k])

        p = np.polyfit(log_k, log_l, 1)
        dimension = p[0]
        return min(3.0, max(1.0, dimension))
    
    @staticmethod
    @njit(cache=True)
    def _numba_calculate_dfa_alpha(x_values: np.ndarray) -> float:
        """Numba JIT化: DFAアルファ計算"""
        n = len(x_values)
        if n < 50:
            return 0.5
        
        y = np.cumsum(x_values - np.mean(x_values))
        scales = np.unique(np.round(np.logspace(1, np.log10(n // 4), 10)).astype(np.int64))
        
        fluctuations = np.zeros(len(scales))
        
        for i, scale in enumerate(scales):
            if scale < 4: continue
            segments = n // scale
            if segments == 0: continue
            
            rms = np.zeros(segments)
            t = np.arange(scale)
            for j in range(segments):
                segment = y[j*scale:(j+1)*scale]
                p = np.polyfit(t, segment, 1)
                detrended = segment - (p[0] * t + p[1])
                rms[j] = np.sqrt(np.mean(detrended**2))
            
            fluctuations[i] = np.mean(rms)

        valid_fluc = fluctuations > 1e-9
        if np.sum(valid_fluc) < 2:
            return 0.5
            
        log_scales = np.log(scales[valid_fluc])
        log_fluctuations = np.log(fluctuations[valid_fluc])
        
        p = np.polyfit(log_scales, log_fluctuations, 1)
        alpha = p[0]
        return min(2.0, max(0.0, alpha))

    @staticmethod
    @njit(cache=True)
    def _numba_estimate_lyapunov_exponent(x: np.ndarray) -> float:
        """Numba JIT化: リアプノフ指数推定"""
        n = len(x)
        if n < 10:
            return 0.0

        total_log_div = 0.0
        count = 0
        
        for i in range(n - 5):
            for j in range(i + 1, min(i + 6, n - 5)):
                initial_dist = np.abs(x[i] - x[j])
                final_dist = np.abs(x[i + 5] - x[j + 5])
                if initial_dist > 1e-8:
                    total_log_div += np.log(final_dist / initial_dist)
                    count += 1
        
        return (total_log_div / count) / 5 if count > 0 else 0.0

    @staticmethod
    @njit(cache=True)
    def _numba_measure_chaos_degree(x: np.ndarray) -> float:
        """Numba JIT化: カオス度測定"""
        if len(x) == 0: return 0.0
        mean_abs = np.abs(np.mean(x))
        std_dev = np.std(x)
        if mean_abs < 1e-8:
            return 1.0 # Or some other representation of high chaos for zero mean
        return min(1.0, std_dev / mean_abs)

    @staticmethod
    @njit(cache=True)
    def _numba_detect_strange_attractor(x: np.ndarray) -> float:
        """Numba JIT化: ストレンジアトラクター指標"""
        if len(x) == 0:
            return 0.0
        return len(np.unique(x)) / len(x)
        
    @staticmethod
    @njit(cache=True)
    def _numba_calculate_vorticity(x: np.ndarray) -> float:
        """Numba JIT化: 渦度計算"""
        if len(x) < 3:
            return 0.0
        return np.mean(np.abs(np.diff(np.diff(x))))

    @staticmethod
    @njit(cache=True)
    def _numba_measure_energy_cascade(x: np.ndarray) -> float:
        """Numba JIT化: エネルギーカスケード測定"""
        n = len(x)
        if n < 8:
            return 0.0
        scales = np.array([2, 4, 8])
        energies = np.zeros(len(scales))
        
        for i, scale in enumerate(scales):
            if n >= scale:
                num_segments = n // scale
                reshaped = x[:num_segments * scale].reshape(num_segments, scale)
                energies[i] = np.var(np.mean(reshaped, axis=1))

        if len(energies) < 2:
            return 0.0
        
        return -np.mean(np.diff(np.log(energies + 1e-8)))

    @staticmethod
    @njit(cache=True)
    def _numba_calculate_network_density(x: np.ndarray) -> float:
        """Numba JIT化: ネットワーク密度"""
        if len(x) == 0:
            return 0.0
        return len(np.unique(x)) / len(x)

    @staticmethod
    @njit(cache=True)
    def _numba_calculate_centrality(x: np.ndarray) -> float:
        """Numba JIT化: 中心性"""
        if len(x) == 0:
            return 0.0
        mean_val = np.mean(x)
        if np.abs(mean_val) < 1e-8:
            return 0.0 # Or handle as per definition
        return np.median(x) / mean_val

    # --- Numba 高速化対象外の関数 (Scipy/Pandas等に依存) ---
    @handle_zero_std
    def _estimate_dominant_period(self, series: pd.Series) -> float:
        """主要周期推定"""
        try:
            if len(series) < 4:
                return 1.0
            
            fft_vals = np.abs(fft(series.values))
            freqs = fftfreq(len(series))
            fft_vals[0] = 0
            
            if np.max(fft_vals) < 1e-9:
                return 1.0
            
            dominant_freq = freqs[np.argmax(fft_vals)]
            period = 1 / (abs(dominant_freq) + 1e-8)
            
            return min(len(series), max(1.0, period))
        except:
            return 1.0

    @handle_zero_std
    def _measure_self_similarity(self, series: pd.Series, scale: int) -> float:
        """自己相似性測定（簡易版）"""
        try:
            if len(series) < scale * 2:
                return 0.0
            
            downsampled = series.iloc[::scale]
            if len(downsampled) < 2:
                return 0.0
            
            original_norm = (series - series.mean()) / (series.std() + 1e-8)
            downsampled_norm = (downsampled - downsampled.mean()) / (downsampled.std() + 1e-8)
            
            min_len = min(len(original_norm), len(downsampled_norm))
            correlation = np.corrcoef(
                original_norm.iloc[:min_len], 
                downsampled_norm.iloc[:min_len]
            )[0, 1]
            
            return abs(correlation) if not np.isnan(correlation) else 0.0
        except:
            return 0.0

    @handle_zero_std
    def _calculate_correlation_dimension(self, series: pd.Series) -> float:
        """相関次元計算"""
        if len(series) < 20:
            return 1.0
        embedding_dim = 3
        points = []
        series_vals = series.values
        for i in range(len(series_vals) - embedding_dim):
            point = series_vals[i:i+embedding_dim]
            points.append(point)
        
        if len(points) < 10:
            return 1.0
        
        distances = pdist(np.array(points))
        dist_std = np.std(distances)
        if dist_std < 1e-9:
            return 1.0
        
        radius_range = np.logspace(-2, 0, 10) * dist_std
        correlations = []
        for radius in radius_range:
            correlation = np.sum(distances < radius) / len(distances)
            correlations.append(correlation + 1e-10)
        
        log_radius = np.log(radius_range)
        log_correlations = np.log(correlations)

        if not np.all(np.isfinite(log_radius)) or not np.all(np.isfinite(log_correlations)):
            return 1.0
        if np.std(log_radius) < 1e-9 or np.std(log_correlations) < 1e-9:
            return 1.0

        dimension = np.polyfit(log_radius, log_correlations, 1)[0]
        return min(5.0, max(0.5, dimension))

    @handle_zero_std
    def _measure_golden_ratio_adherence(self, series: pd.Series) -> float:
        """黄金比固着度測定"""
        try:
            ratios = series.rolling(window=2).apply(lambda x: x.iloc[1] / (x.iloc[0] + 1e-8), raw=False)
            golden_ratio = 1.618
            adherence = 1 - np.mean(np.abs(ratios - golden_ratio)) / golden_ratio
            return float(max(0, adherence))
        except:
            return 0.0
    
    @handle_zero_std
    def _calculate_orbital_indicator(self, series: pd.Series) -> float:
        """軌道指標"""
        return float(series.autocorr(lag=12))

    # --- Numba 高速化対象の関数 ---
    @staticmethod
    @njit(cache=True)
    def _numba_adf_statistic(x: np.ndarray) -> float:
        """Numba JIT化: ADF統計量の簡易版"""
        if len(x) < 10:
            return 0.0
        y = np.diff(x)
        x_lag = x[:-1]
        
        # Numbaで相関係数を計算
        mean_y = np.mean(y)
        mean_x_lag = np.mean(x_lag)
        cov = np.sum((y - mean_y) * (x_lag - mean_x_lag))
        var_y = np.sum((y - mean_y)**2)
        var_x_lag = np.sum((x_lag - mean_x_lag)**2)

        if var_y <= 1e-9 or var_x_lag <= 1e-9:
            return 0.0
        
        corr = cov / np.sqrt(var_y * var_x_lag)
        return corr

    @staticmethod
    @njit(cache=True)
    def _numba_test_arch_effect(x: np.ndarray) -> float:
        """Numba JIT化: ARCH効果の検定"""
        if len(x) < 2: return 0.0
        res_sq = x**2
        res_sq_1 = res_sq[:-1]
        res_sq_2 = res_sq[1:]
        
        mean_1 = np.mean(res_sq_1)
        mean_2 = np.mean(res_sq_2)
        cov = np.sum((res_sq_1 - mean_1) * (res_sq_2 - mean_2))
        var_1 = np.sum((res_sq_1 - mean_1)**2)
        var_2 = np.sum((res_sq_2 - mean_2)**2)

        if var_1 <= 1e-9 or var_2 <= 1e-9:
            return 0.0
            
        corr = cov / np.sqrt(var_1 * var_2)
        return corr
        
    @staticmethod
    @njit(cache=True)
    def _numba_test_market_efficiency(x: np.ndarray) -> float:
        """Numba JIT化: 市場効率性の検定"""
        if len(x) < 2: return 0.5
        x1 = x[:-1]
        x2 = x[1:]

        mean_1 = np.mean(x1)
        mean_2 = np.mean(x2)
        cov = np.sum((x1 - mean_1) * (x2 - mean_2))
        var_1 = np.sum((x1 - mean_1)**2)
        var_2 = np.sum((x2 - mean_2)**2)

        if var_1 <= 1e-9 or var_2 <= 1e-9:
            return 0.0

        corr = cov / np.sqrt(var_1 * var_2)
        return 1.0 - np.abs(corr)

    @staticmethod
    @njit(cache=True)
    def _numba_analyze_poincare_section(x: np.ndarray) -> float:
        """Numba JIT化: ポアンカレ断面解析"""
        if len(x) < 2:
            return 0.0
        distances = (x[:-1] - x[1:]) / np.sqrt(2.0)
        return np.std(distances)

    @staticmethod
    @njit(cache=True)
    def _numba_estimate_kolmogorov_scale(x: np.ndarray) -> float:
        """Numba JIT化: コルモゴロフスケール推定"""
        if len(x) < 4:
            return 0.0
        epsilon_proxy = np.mean(np.abs(x**3))
        nu_proxy = np.var(x)
        if epsilon_proxy < 1e-9 or nu_proxy < 1e-9:
            return 0.0
        scale = (nu_proxy**3 / epsilon_proxy)**0.25
        return scale

    @staticmethod
    @njit(cache=True)
    def _numba_calculate_prisoners_dilemma_index(x: np.ndarray) -> float:
        """Numba JIT化: 囚人のジレンマ指数"""
        if len(x) < 2:
            return 0.0
        signs = np.sign(x)
        actions = signs[:-1] * signs[1:]
        return np.sum(actions > 0) / len(actions)

    @staticmethod
    @njit(cache=True)
    def _numba_measure_recovery_rate(x: np.ndarray) -> float:
        """Numba JIT化: 回復率"""
        if len(x) < 2:
            return 0.0
        
        prices = np.exp(np.cumsum(x))
        
        max_price_so_far = np.maximum.accumulate(prices)
        drawdowns = max_price_so_far - prices
        peak_index = np.argmax(drawdowns)
        
        trough_index = 0
        if peak_index > 0:
            trough_index = np.argmax(prices[:peak_index+1])

        if peak_index == trough_index:
            return 1.0
        
        drawdown_val = prices[trough_index] - prices[peak_index]
        if abs(drawdown_val) < 1e-9:
            return 1.0

        final_recovery = (prices[-1] - prices[peak_index]) / abs(drawdown_val)
        return max(0.0, final_recovery)

    @staticmethod
    @njit(cache=True)
    def _numba_analyze_gait_pattern(x: np.ndarray) -> float:
        """Numba JIT化: 歩行パターン分析"""
        if len(x) < 5:
            return 0.0
        
        signs = np.sign(x)
        runs = np.zeros(len(signs))
        run_idx = 0
        current_run = 1
        
        for i in range(1, len(signs)):
            if signs[i] == signs[i-1] or signs[i] == 0:
                current_run += 1
            else:
                if signs[i-1] != 0:
                    runs[run_idx] = current_run
                    run_idx += 1
                current_run = 1
        if signs[-1] != 0:
            runs[run_idx] = current_run
            run_idx += 1

        if run_idx == 0:
            return 1.0
        
        valid_runs = runs[:run_idx]
        return 1.0 / (1.0 + np.std(valid_runs) / np.mean(valid_runs))

    @staticmethod
    @njit(cache=True)
    def _numba_estimate_taylor_scale(x: np.ndarray) -> float:
        """Numba JIT化: テイラースケール推定"""
        if len(x) < 2:
            return 0.0
        variance_x = np.var(x)
        variance_dx = np.var(np.diff(x))
        if variance_dx < 1e-9:
            return 0.0
        return np.sqrt(variance_x / variance_dx)

    @staticmethod
    @njit(cache=True)
    def _numba_measure_nonlinear_dynamics(x: np.ndarray) -> float:
        """Numba JIT化: 非線形動力学測定"""
        if len(x) < 5 or np.std(x) < 1e-9:
            return 0.0
        
        t = np.arange(len(x))
        
        p_linear = np.polyfit(t, x, 1)
        error_linear = np.sum((x - (p_linear[0] * t + p_linear[1]))**2)
        
        if error_linear < 1e-9:
            return 0.0
        
        p_nonlinear = np.polyfit(t, x, 2)
        error_nonlinear = np.sum((x - (p_nonlinear[0] * t**2 + p_nonlinear[1] * t + p_nonlinear[2]))**2)
        
        improvement_ratio = 1.0 - (error_nonlinear / error_linear)
        return max(0.0, improvement_ratio)

    @staticmethod
    @njit(cache=True)
    def _numba_measure_aesthetic_harmony(x_values: np.ndarray) -> float:
        """Numba JIT化: 美的調和測定"""
        if len(x_values) < 4:
            return 0.0
        total_range = np.max(x_values) - np.min(x_values)
        if total_range < 1e-9:
            return 1.0
        
        jerk = np.diff(x_values, 3)
        mean_abs_jerk = np.mean(np.abs(jerk))
        return 1.0 / (1.0 + mean_abs_jerk / total_range)

    @staticmethod
    @njit(cache=True)
    def _numba_calculate_fractal_efficiency(x_values: np.ndarray) -> float:
        """Numba JIT化: フラクタル効率性"""
        if len(x_values) < 2:
            return 0.5
        net_change = np.abs(x_values[-1] - x_values[0])
        total_path_length = np.sum(np.abs(np.diff(x_values)))
        if total_path_length < 1e-9:
            return 1.0
        return net_change / total_path_length

    @staticmethod
    @njit(cache=True)
    def _numba_analyze_galactic_rotation(x_values: np.ndarray) -> float:
        """Numba JIT化: 銀河系回転分析"""
        if len(x_values) < 5:
            return 0.0
        
        mean_val = np.mean(x_values)
        if np.abs(mean_val) < 1e-9:
            return 0.0
        
        t = np.arange(len(x_values))
        p = np.polyfit(t, x_values, 1)
        trend_line = p[0] * t + p[1]
        
        rms_deviation = np.sqrt(np.mean((x_values - trend_line)**2))
        return rms_deviation / mean_val

    @staticmethod
    @njit(cache=True)
    def _numba_measure_proportional_beauty(x_values: np.ndarray) -> float:
        """Numba JIT化: 比例美測定"""
        if len(x_values) < 10:
            return 0.0
        
        signs = np.sign(np.diff(x_values))
        runs = np.zeros(len(signs))
        run_idx = 0
        current_run = 1
        for i in range(1, len(signs)):
            if signs[i] == signs[i-1]:
                current_run += 1
            else:
                runs[run_idx] = current_run
                run_idx += 1
                current_run = 1
        runs[run_idx] = current_run
        run_idx += 1

        if run_idx < 2:
            return 1.0
            
        valid_runs = runs[:run_idx]
        ratios = valid_runs[:-1] / (valid_runs[1:] + 1e-9)
        return 1.0 / (1.0 + np.std(ratios))

    @staticmethod
    @njit(cache=True)
    def _numba_measure_visual_balance(x_values: np.ndarray) -> float:
        """Numba JIT化: 視覚的バランス測定"""
        if len(x_values) < 2:
            return 0.0
        
        median_price = np.median(x_values)
        time_above = np.sum(x_values > median_price)
        time_below = np.sum(x_values < median_price)
        total_time = time_above + time_below
        if total_time == 0:
            return 1.0
        
        balance_ratio = time_above / total_time
        return 1.0 - np.abs(balance_ratio - 0.5) * 2

    @staticmethod
    @njit(cache=True)
    def _numba_safe_polyfit_slope(x: np.ndarray) -> float:
        """Numba JIT化: 安全なpolyfit傾き計算"""
        if len(x) < 2 or np.std(x) < 1e-9:
            return 0.0
        return np.polyfit(np.arange(len(x)), x, 1)[0]
    
    @staticmethod
    @njit(cache=True)
    def _numba_calculate_anchoring_effect(x_values: np.ndarray) -> float:
        """Numba JIT化: アンカリング効果"""
        if len(x_values) <= 1:
            return 0.0
        
        distances = np.abs(x_values - x_values[0])
        if np.std(distances) < 1e-9:
            return 0.0
            
        time_index = np.arange(len(x_values))
        
        mean_t = np.mean(time_index)
        mean_d = np.mean(distances)
        cov = np.sum((time_index - mean_t) * (distances - mean_d))
        var_t = np.sum((time_index - mean_t)**2)
        var_d = np.sum((distances - mean_d)**2)

        if var_t <= 1e-9 or var_d <= 1e-9:
            return 0.0
        
        corr = cov / np.sqrt(var_t * var_d)
        return -corr

    @staticmethod
    @njit(cache=True)
    def _numba_calculate_learning_effect(x: np.ndarray) -> float:
        """Numba JIT化: 学習効果"""
        if len(x) <= 1:
            return 0.0
        abs_x = np.abs(x)
        if np.std(abs_x) < 1e-9:
            return 0.0
        
        time_index = np.arange(len(x))
        mean_t = np.mean(time_index)
        mean_abs_x = np.mean(abs_x)
        cov = np.sum((time_index - mean_t) * (abs_x - mean_abs_x))
        var_t = np.sum((time_index - mean_t)**2)
        var_abs_x = np.sum((abs_x - mean_abs_x)**2)

        if var_t <= 1e-9 or var_abs_x <= 1e-9:
            return 0.0
            
        return cov / np.sqrt(var_t * var_abs_x)

    @staticmethod
    @njit(cache=True)
    def _numba_volatility_clustering(x: np.ndarray) -> float:
        """Numba JIT化: ボラティリティクラスタリング"""
        if len(x) <= 2:
            return 0.0
        abs_x1 = np.abs(x[:-1])
        abs_x2 = np.abs(x[1:])
        
        if np.std(abs_x1) < 1e-9 or np.std(abs_x2) < 1e-9:
            return 0.0
        
        mean_1 = np.mean(abs_x1)
        mean_2 = np.mean(abs_x2)
        cov = np.sum((abs_x1 - mean_1) * (abs_x2 - mean_2))
        var_1 = np.sum((abs_x1 - mean_1)**2)
        var_2 = np.sum((abs_x2 - mean_2)**2)

        if var_1 <= 1e-9 or var_2 <= 1e-9:
            return 0.0
        
        return cov / np.sqrt(var_1 * var_2)

    @staticmethod
    @njit(cache=True)
    def _numba_cooperation_index(x: np.ndarray) -> float:
        """Numba JIT化: 協力度指標"""
        if len(x) <= 2 or np.std(x) < 1e-9:
            return 0.0
        
        x1 = x[:-1]
        x2 = x[1:]
        
        mean_1 = np.mean(x1)
        mean_2 = np.mean(x2)
        cov = np.sum((x1 - mean_1) * (x2 - mean_2))
        var_1 = np.sum((x1 - mean_1)**2)
        var_2 = np.sum((x2 - mean_2)**2)

        if var_1 <= 1e-9 or var_2 <= 1e-9:
            return 0.0
        
        return cov / np.sqrt(var_1 * var_2)


    # --- Numba 高速化対象外の関数 (Scipy/Pandas等に依存) ---
    @handle_zero_std
    def _cointegration_test(self, series: pd.Series) -> float:
        """共和分検定の簡易版（安全な実装）"""
        try:
            if len(series) <= 10 or series.std() < 1e-9:
                return 0.0
            autocorr = series.autocorr(lag=10)
            return float(autocorr) if not np.isnan(autocorr) else 0.0
        except Exception:
            return 0.0

    @handle_zero_std
    def _granger_causality_safe_test(self, x: pd.Series, y_full: pd.Series) -> float:
        """インデックス安全なGranger因果性テスト"""
        try:
            min_len = min(len(x), len(y_full))
            if min_len < 10: return 0.0
            x_safe, y_safe = x.iloc[:min_len], y_full.iloc[:min_len]
            correlation = np.corrcoef(x_safe.diff().dropna(), y_safe.shift(1).dropna())[0, 1]
            return float(correlation) if not np.isnan(correlation) else 0.0
        except:
            return 0.0

    @handle_zero_std
    def _estimate_mean_reversion_speed(self, series: pd.Series) -> float:
        """平均回帰速度の推定"""
        try:
            if len(series) < 2 or series.std() < 1e-9:
                return 0.5
            speed = abs(series.autocorr(lag=1))
            return float(1 - speed) if not np.isnan(speed) else 0.5
        except:
            return 0.5
    
    @handle_zero_std
    def _calculate_multifractal_width(self, series: pd.Series) -> float:
        """マルチフラクタル幅の代替指標"""
        try:
            if len(series) < 50: return 0.0
            returns = series.pct_change().dropna()
            if len(returns) < 20: return 0.0
            
            volatility = returns.rolling(window=10).std()
            median_vol = volatility.median()
            
            high_vol_returns = returns[volatility > median_vol]
            low_vol_returns = returns[volatility <= median_vol]

            if len(high_vol_returns) < 20 or len(low_vol_returns) < 20: return 0.0

            # Numba化されたHurst指数計算を呼び出す
            hurst_high = self._numba_calculate_hurst_exponent(high_vol_returns.values)
            hurst_low = self._numba_calculate_hurst_exponent(low_vol_returns.values)
            
            width = abs(hurst_high - hurst_low)
            return width if not np.isnan(width) else 0.0
        except Exception:
            return 0.0

    @handle_zero_std
    def _calculate_clustering_coefficient(self, series: pd.Series) -> float:
        """クラスタリング係数の代替指標"""
        try:
            if len(series) < 30: return 0.0
            
            ret_short = series.pct_change(periods=5).dropna()
            ret_mid = series.pct_change(periods=10).dropna()
            ret_long = series.pct_change(periods=20).dropna()
            
            df_rets = pd.concat([ret_short, ret_mid, ret_long], axis=1).dropna()
            if len(df_rets) < 2: return 0.0
            
            corr_matrix = df_rets.corr().values
            n = len(corr_matrix)
            avg_corr = (np.sum(np.abs(corr_matrix)) - np.trace(np.abs(corr_matrix))) / (n**2 - n)
            return avg_corr if not np.isnan(avg_corr) else 0.0
        except Exception:
            return 0.0

    @handle_zero_std
    def _test_long_memory(self, x: np.ndarray) -> float:
        """長期記憶性の検定"""
        try:
            # Numba化されたHurst指数計算を呼び出す
            return self._numba_calculate_hurst_exponent(x)
        except Exception:
            return 0.5
            
    @handle_zero_std
    def _analyze_harmony(self, x: np.ndarray) -> float:
        """和声分析の簡易版"""
        try:
            if len(x) < 16: return 0.0
            
            power_spectrum = np.abs(fft(x)[1:len(x)//2])**2
            freqs = fftfreq(len(x), 1)[1:len(x)//2]
            
            if len(power_spectrum) < 3: return 0.0
            
            top_3_indices = np.argsort(power_spectrum)[-3:]
            top_3_freqs = freqs[top_3_indices]
            
            if np.min(top_3_freqs) < 1e-9: return 0.0
                
            r1 = top_3_freqs[1] / top_3_freqs[0]
            r2 = top_3_freqs[2] / top_3_freqs[0]

            harmony_error = min(abs(r1 - round(r1)), abs(r1 - round(r1*2)/2)) + \
                            min(abs(r2 - round(r2)), abs(r2 - round(r2*2)/2))
            
            return 1 / (1 + harmony_error)
        except Exception:
            return 0.0
    
    @handle_zero_std
    def _estimate_dark_energy(self, series: pd.Series) -> float:
        """ダークエネルギーの代替指標"""
        try:
            if len(series) < 3: return 0.0
            velocity = series.diff().dropna()
            if len(velocity) < 2: return 0.0
            acceleration = velocity.diff().dropna()
            
            return acceleration.mean() / (series.std() + 1e-9)
        except Exception:
            return 0.0
    
    @handle_zero_std
    def _detect_gravitational_wave_pattern(self, series: pd.Series) -> float:
        """重力波パターンの代替指標"""
        try:
            if len(series) < 40: return 0.0
            
            mid = len(series) // 2
            vol_first = series.iloc[:mid].pct_change().dropna().rolling(window=5).std()
            vol_second = series.iloc[mid:].pct_change().dropna().rolling(window=5).std()
            
            if len(vol_first) < 10 or len(vol_second) < 10: return 0.0

            period_first = self._estimate_dominant_period(vol_first)
            period_second = self._estimate_dominant_period(vol_second)
            
            if period_first < 1e-9: return 0.0

            return max(-1, min(1, (period_first - period_second) / period_first))
        except Exception:
            return 0.0
    
    @handle_zero_std
    def _measure_symmetry(self, series: pd.Series) -> float:
        """時間的対称性の測定"""
        try:
            if len(series) < 10: return 0.0
            
            mid = len(series) // 2
            first_half = series.iloc[:mid].values
            second_half = series.iloc[-mid:].values
            
            second_half_reversed = second_half[::-1]
            
            correlation = np.corrcoef(stats.zscore(first_half), stats.zscore(second_half_reversed))[0, 1]
            
            return abs(correlation) if not np.isnan(correlation) else 0.0
        except Exception:
            return 0.0
            
    @handle_zero_std
    def _detect_big_bang_echo(self, series: pd.Series) -> float:
        """ビッグバン・エコーの代替指標"""
        try:
            if len(series) < 10: return 0.0
            
            returns = series.pct_change().abs().dropna()
            if returns.empty: return 0.0
            
            max_shock_index = returns.argmax()
            max_shock_value = returns.iloc[max_shock_index]
            
            time_since_shock = (len(series) - max_shock_index) / len(series)
            
            return max_shock_value * time_since_shock
        except Exception:
            return 0.0

    @handle_zero_std
    def _analyze_stellar_pulsation(self, series: pd.Series) -> float:
        """恒星脈動の代替指標"""
        try:
            if len(series) < 16: return 0.0
            
            power_spectrum = np.abs(fft(series.values)[1:len(series)//2])**2
            if len(power_spectrum) == 0: return 0.0
            total_power = np.sum(power_spectrum)
            if total_power < 1e-9: return 0.0
            
            return np.max(power_spectrum) / total_power
        except Exception:
            return 0.0

    @handle_zero_std
    def _calculate_betweenness_centrality(self, series: pd.Series) -> float:
        """媒介中心性の代替指標"""
        try:
            if len(series) < 5: return 0.0
            
            peaks, _ = find_peaks(series.values)
            troughs, _ = find_peaks(-series.values)
            
            center_index = len(series) // 2
            
            all_extrema = np.concatenate([peaks, troughs])
            if len(all_extrema) == 0: return 0.0
                
            min_dist_from_center = np.min(np.abs(all_extrema - center_index))
            
            return 1.0 - (min_dist_from_center / center_index)
        except Exception:
            return 0.0

    @handle_zero_std
    def _analyze_planetary_motion(self, series: pd.Series) -> float:
        """惑星運動の代替指標"""
        try:
            if len(series) < 10: return 0.0
            
            mean_price = series.mean()
            distance_from_mean = np.abs(series - mean_price)
            returns = series.pct_change().abs().fillna(0)
            
            correlation = np.corrcoef(distance_from_mean.values[1:], returns.values[1:])[0, 1]
            return correlation if not np.isnan(correlation) else 0.0
        except Exception:
            return 0.0

    @handle_zero_std
    def _analyze_cmb_pattern(self, series: pd.Series) -> float:
        """CMBパターンの代替指標"""
        try:
            if len(series) < 16: return 0.0
            
            power_spectrum = np.abs(fft(series.values)[1:len(series)//2])**2
            if np.sum(power_spectrum) < 1e-9: return 0.0
            
            prob_spectrum = power_spectrum / np.sum(power_spectrum)
            return -np.sum(prob_spectrum * np.log2(prob_spectrum + 1e-9))
        except Exception:
            return 0.0

    @handle_zero_std
    def _measure_cosmic_inflation(self, series: pd.Series) -> float:
        """宇宙インフレーションの代替指標"""
        try:
            if len(series) < 5: return 0.0
            returns = series.pct_change().dropna()
            if len(returns) < 4: return 0.0
            return stats.kurtosis(returns.values)
        except Exception:
            return 0.0

    @handle_zero_std
    def _calculate_eigenvector_centrality(self, series: pd.Series) -> float:
        """固有ベクトル中心性の代替指標"""
        try:
            if len(series) < 10: return 0.0
            returns = series.pct_change().dropna()
            if len(returns) < 5: return 0.0
            return sum(np.abs(returns.autocorr(lag=i)) for i in range(1, 5))
        except Exception:
            return 0.0

    @handle_zero_std
    def _safe_sample_entropy(self, x: np.ndarray) -> float:
        """安全なSample Entropy計算"""
        try:
            std_dev = np.std(x)
            if std_dev < 1e-9: return 0.0
            return ent.sample_entropy(x, 2, 0.2 * std_dev)
        except Exception:
            return 0.0

    @handle_zero_std
    def _safe_approx_entropy(self, x: np.ndarray) -> float:
        """安全なApproximate Entropy計算"""
        try:
            std_dev = np.std(x)
            if std_dev < 1e-9: return 0.0
            return ent.app_entropy(x, 2, 0.2 * std_dev)
        except Exception:
            return 0.0
    
    @handle_zero_std
    def _social_inertia(self, x: pd.Series) -> float:
        """社会的慣性 (ラグ1自己相関)"""
        try:
            if len(x) <= 2: return 0.0
            autocorr = x.autocorr(lag=1)
            return autocorr if np.isfinite(autocorr) else 0.0
        except Exception:
            return 0.0
        
# ========== メイン実行関数（マルチタイムフレーム完全対応版） ==========

def main_with_file_separation():
    """
    🔥 最適化済み独立特徴量計算のメイン実行関数（マルチタイムフレーム完全対応版）
    
    【重大改善点】
    1. 並列処理時のインデックス不整合を完全解決
    2. 巨大データの無駄なコピーを完全排除
    3. Parabolic SARの正しい実装
    4. 全特徴量のマルチタイムフレーム完全対応
    5. エラー耐性とパフォーマンスの大幅向上
    """
    import time
    from pathlib import Path
    import multiprocessing as mp

    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    # ベースとなるディレクトリパスを定義
    base_data_dir = Path(r"C:\project_forge\data\temp_chunks")
    input_path = base_data_dir / "parquet" / "01_mfdfa_45features.parquet"
    
    # 🔥 新機能：実行時設定選択 🔥
    print("DEBUG: 設定選択部分に到達しました")
    print("=" * 70)
    print("🚀 マルチタイムフレーム独立特徴量計算エンジン 🚀")
    print("=" * 70)
    
    # データ分割設定
    print("\n📊 データ分割設定:")
    print("1. 分割なし（全データ一括処理）")
    print("2. 2分割")
    print("3. 4分割")
    print("4. 8分割（推奨：大容量データ）")
    
    while True:
        try:
            chunk_choice = int(input("選択してください (1-4): "))
            if chunk_choice in [1, 2, 3, 4]:
                chunk_map = {1: 1, 2: 2, 3: 4, 4: 8}
                n_chunks = chunk_map[chunk_choice]
                break
            else:
                print("❌ 1-4の範囲で選択してください")
        except ValueError:
            print("❌ 数字を入力してください")
    
    # ワーカー数設定
    max_workers = mp.cpu_count()
    print(f"\n⚡ 並列処理設定（最大CPU数: {max_workers}）:")
    print(f"1. 自動設定（CPU数-2 = {max(1, max_workers-2)}）")
    print(f"2. 最大パフォーマンス（CPU数 = {max_workers}）")
    print("3. 手動設定")
    
    while True:
        try:
            worker_choice = int(input("選択してください (1-3): "))
            if worker_choice == 1:
                num_processes = max(1, max_workers - 2)
                break
            elif worker_choice == 2:
                num_processes = max_workers
                break
            elif worker_choice == 3:
                while True:
                    try:
                        num_processes = int(input(f"ワーカー数を入力 (1-{max_workers}): "))
                        if 1 <= num_processes <= max_workers:
                            break
                        else:
                            print(f"❌ 1-{max_workers}の範囲で入力してください")
                    except ValueError:
                        print("❌ 数字を入力してください")
                break
            else:
                print("❌ 1-3の範囲で選択してください")
        except ValueError:
            print("❌ 数字を入力してください")
    
    # テストモード設定
    print("\n🧪 実行モード選択:")
    print("1. 本番モード（全データ処理）")
    print("2. テストモード（1000行制限）")
    
    while True:
        try:
            test_choice = int(input("選択してください (1-2): "))
            if test_choice in [1, 2]:
                test_mode = test_choice == 2
                break
            else:
                print("❌ 1-2の範囲で選択してください")
        except ValueError:
            print("❌ 数字を入力してください")
    
    # マルチタイムフレーム設定
    print("\n🌐 タイムフレーム設定:")
    print("1. シングルタイムフレーム（1T のみ）")
    print("2. マルチタイムフレーム（1T/5T/15T/1H/4H 全対応）")
    
    while True:
        try:
            timeframe_choice = int(input("選択してください (1-2): "))
            if timeframe_choice in [1, 2]:
                multi_timeframe = timeframe_choice == 2
                break
            else:
                print("❌ 1-2の範囲で選択してください")
        except ValueError:
            print("❌ 数字を入力してください")
    
    # 設定確認
    print("\n" + "=" * 50)
    print("📋 実行設定確認")
    print("=" * 50)
    print(f"📊 データ分割数: {n_chunks}")
    print(f"⚡ ワーカー数: {num_processes}")
    print(f"🧪 テストモード: {'ON（1000行制限）' if test_mode else 'OFF（全データ処理）'}")
    print(f"🌐 タイムフレーム: {'マルチ（1T/5T/15T/1H/4H）' if multi_timeframe else 'シングル（1T のみ）'}")
    print("=" * 50)
    
    confirm = input("\n実行しますか？ (y/N): ").strip().lower()
    if confirm != 'y':
        print("❌ 実行をキャンセルしました")
        return
    
    try:
        if not input_path.exists():
            logger.error(f"入力ファイルが見つかりません: {input_path}")
            return
        
        logger.info(f"MFDFAファイル読み込み開始: {input_path}")
        df = pd.read_parquet(input_path)
        logger.info(f"データ読み込み完了: {df.shape}")
        
        required_cols = ['open', 'high', 'low', 'close', 'volume']
        if not all(col in df.columns for col in required_cols):
            logger.error(f"必須列が不足しています: {required_cols}")
            return
        
        start_time = time.time()
        logger.info("🚀 マルチタイムフレーム独立特徴量計算開始 🚀")
        
        # 🔥 設定に基づく計算機初期化 🔥
        calculator = AdvancedIndependentFeatures(
            n_processes=num_processes,
            multi_timeframe=multi_timeframe,
            test_mode=test_mode
        )
        
        # データ分割処理
        if n_chunks == 1:
            # 分割なし
            result_df = calculator.calculate_all_features(df)
        else:
            # データ分割処理
            logger.info(f"📊 データを{n_chunks}個のチャンクに分割して処理開始")
            chunk_size = len(df) // n_chunks
            results = []
            
            for i in range(n_chunks):
                start_idx = i * chunk_size
                end_idx = start_idx + chunk_size if i < n_chunks - 1 else len(df)
                
                chunk_df = df.iloc[start_idx:end_idx].copy()
                logger.info(f"🔄 チャンク {i+1}/{n_chunks} 処理中 (行数: {len(chunk_df)})")
                
                chunk_result = calculator.calculate_all_features(chunk_df, chunk_id=f"{i+1}")
                results.append(chunk_result)
            
            # チャンク結合
            logger.info("🔗 チャンク結合開始")
            result_df = pd.concat(results, ignore_index=True)
            logger.info(f"🎯 チャンク結合完了: {result_df.shape}")
        
        calculation_time = time.time() - start_time
        logger.info(f"🎉 特徴量計算完了！時間: {calculation_time:.2f}秒 🎉")
        
        total_features = len(result_df.columns)
        
        # Parquet形式のみで保存
        output_dir = base_data_dir / "parquet"
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # ファイル名に設定情報を含める
        mode_suffix = "test" if test_mode else "prod"
        tf_suffix = "multi" if multi_timeframe else "single"
        output_path = output_dir / f"02_{tf_suffix}timeframe_{total_features}features_{mode_suffix}.parquet"
        
        logger.info(f"PARQUET形式で保存開始: {output_path}")
        result_df.to_parquet(output_path, compression='snappy')
        logger.info(f"PARQUET形式で保存完了")

        logger.info(f"""
🎊 マルチタイムフレーム特徴量生成完了！🎊
┌───────────────────────────────────────────┐
│ 🔥 BREAKTHROUGH ACHIEVEMENT 🔥              │
├───────────────────────────────────────────┤
│ 🚀 拡張特徴量: {total_features:,}個                    │
│ ⚡ 処理時間: {calculation_time:.1f}秒                     │
│ 📊 分割数: {n_chunks}チャンク                     │
│ 👥 ワーカー数: {num_processes}プロセス                  │
│ 🧪 テストモード: {'ON' if test_mode else 'OFF'}                   │
│ 🌐 時間軸: {'マルチ対応' if multi_timeframe else 'シングル'}               │
│ 💾 出力ファイル: {output_path.name}         │
└───────────────────────────────────────────┘
        """)
        
    except Exception as e:
        logger.error(f"エラーが発生しました: {e}")
        raise

if __name__ == "__main__":
    main_with_file_separation()                                                         