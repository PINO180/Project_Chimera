"""
市場レジーム検知エンジン
HMMと時系列クラスタリングによる市場環境の分類と動的リスクパラメータ調整
"""
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))

import blueprint as config
import numpy as np
import pandas as pd
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List, Optional, Tuple
import logging
from hmmlearn import hmm
from tslearn.clustering import TimeSeriesKMeans
from tslearn.preprocessing import TimeSeriesScalerMeanVariance
from sklearn.preprocessing import StandardScaler
import joblib

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class MarketRegimeDetector:
    """
    市場レジーム検知エンジン
    HMMまたは時系列クラスタリングで市場環境を分類
    """
    
    def __init__(self,
                method: str = 'hmm',
                n_regimes: int = 4,
                model_path: Optional[str] = None,
                config_path: str = str(config.CONFIG_REGIME)):
        """
        Args:
            method: 検知手法 ('hmm' or 'clustering')
            n_regimes: レジーム数（状態数またはクラスタ数）
            model_path: 学習済みモデルのパス
            config_path: レジーム別リスクパラメータ設定ファイル
        """
        self.method = method
        self.n_regimes = n_regimes
        self.model_path = model_path
        self.config_path = Path(config_path)

        # モデル
        self.model: Optional[Any] = None
        self.scaler: Optional[StandardScaler] = None

        # レジーム別リスクパラメータ
        self.regime_params = self._load_regime_config()

        # モデル読み込み
        if model_path and Path(model_path).exists():
            self.load_model(model_path)

        logger.info(f"MarketRegimeDetector初期化: method={method}, n_regimes={n_regimes}")
    
    def _load_regime_config(self) -> Dict[int, Dict[str, Any]]:
        """レジーム別リスクパラメータ設定を読み込む"""
        if not self.config_path.exists():
            logger.warning(f"設定ファイル '{self.config_path}' が見つかりません。デフォルト設定を使用します。")
            default_config = {
                0: {  # 低ボラティリティ
                    'name': 'Low Volatility',
                    'kelly_fraction': 0.5,
                    'atr_multiplier_sl': 1.5,
                    'atr_multiplier_tp': 2.0,
                    'confidence_threshold': 0.65
                },
                1: {  # 高ボラティリティ
                    'name': 'High Volatility',
                    'kelly_fraction': 0.25,
                    'atr_multiplier_sl': 3.0,
                    'atr_multiplier_tp': 4.0,
                    'confidence_threshold': 0.70
                },
                2: {  # トレンド
                    'name': 'Trending',
                    'kelly_fraction': 0.5,
                    'atr_multiplier_sl': 1.5,
                    'atr_multiplier_tp': 3.0,
                    'confidence_threshold': 0.60
                },
                3: {  # レンジ
                    'name': 'Range-bound',
                    'kelly_fraction': 0.25,
                    'atr_multiplier_sl': 1.5,
                    'atr_multiplier_tp': 2.0,
                    'confidence_threshold': 0.75
                }
            }
            
            # デフォルト設定を保存
            self.config_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.config_path, 'w') as f:
                json.dump(default_config, f, indent=2)
            
            return default_config
        
        with open(self.config_path, 'r') as f:
            config = json.load(f)
            # キーを整数に変換
            return {int(k): v for k, v in config.items()}
    
    # ========== 特徴量抽出 ==========
    
    def extract_features(self, price_data: pd.DataFrame) -> np.ndarray:
        """
        市場データから特徴量を抽出
        
        Args:
            price_data: OHLCVデータ（必須カラム: open, high, low, close, volume）
        
        Returns:
            特徴量行列 (n_samples, n_features)
        """
        df = price_data.copy()
        
        # リターン
        df['returns'] = df['close'].pct_change()
        
        # 実現ボラティリティ（20日移動窓）
        df['realized_vol'] = df['returns'].rolling(window=20).std()
        
        # ATR（14日）
        df['high_low'] = df['high'] - df['low']
        df['high_close'] = abs(df['high'] - df['close'].shift(1))
        df['low_close'] = abs(df['low'] - df['close'].shift(1))
        df['true_range'] = df[['high_low', 'high_close', 'low_close']].max(axis=1)
        df['atr'] = df['true_range'].rolling(window=14).mean()
        
        # トレンド強度（ADX風の計算）
        df['price_change'] = df['close'].diff()
        df['trend_strength'] = abs(df['price_change'].rolling(window=14).mean()) / df['atr']
        
        # 出来高変化率
        df['volume_change'] = df['volume'].pct_change()
        
        # 価格変化の歪度（非対称性）
        df['returns_skew'] = df['returns'].rolling(window=20).skew()
        
        # 価格変化の尖度（極端な動きの頻度）
        df['returns_kurt'] = df['returns'].rolling(window=20).kurt()
        
        # 特徴量を選択
        features = df[[
            'returns',
            'realized_vol',
            'atr',
            'trend_strength',
            'volume_change',
            'returns_skew',
            'returns_kurt'
        ]].dropna()
        
        return features.values
    
    # ========== HMMによるレジーム検知 ==========
    
    def train_hmm(self, features: np.ndarray) -> None:
        """
        HMMモデルを訓練
        
        Args:
            features: 特徴量行列 (n_samples, n_features)
        """
        logger.info(f"HMMモデル訓練開始: n_regimes={self.n_regimes}")
        
        # データの正規化
        self.scaler = StandardScaler()
        features_scaled = self.scaler.fit_transform(features)
        
        # ガウシアンHMMの作成
        self.model = hmm.GaussianHMM(
            n_components=self.n_regimes,
            covariance_type="full",
            n_iter=100,
            random_state=42
        )
        
        # 訓練
        self.model.fit(features_scaled)
        
        logger.info("✓ HMMモデル訓練完了")
        logger.info(f"  収束スコア: {self.model.score(features_scaled):.4f}")
    
    def predict_regime_hmm(self, features: np.ndarray) -> Tuple[int, np.ndarray]:
        """
        HMMで現在のレジームを予測
        
        Args:
            features: 特徴量行列（直近の観測データ）
        
        Returns:
            (予測レジーム, 各レジームの事後確率)
        """
        if self.model is None or self.scaler is None:
            raise ValueError("モデルが訓練されていません。train_hmm()を先に実行してください。")
        
        # 正規化
        features_scaled = self.scaler.transform(features)
        
        # 予測（最尤レジーム）
        regime = self.model.predict(features_scaled)[-1]
        
        # 事後確率
        posterior = self.model.predict_proba(features_scaled)[-1]
        
        return int(regime), posterior
    
    # ========== 時系列クラスタリングによるレジーム検知 ==========
    
    def train_clustering(self, 
                        price_data: pd.DataFrame,
                        window_size: int = 60) -> None:
        """
        時系列クラスタリングモデルを訓練
        
        Args:
            price_data: 価格データ
            window_size: セグメントのウィンドウサイズ
        """
        logger.info(f"時系列クラスタリング訓練開始: n_regimes={self.n_regimes}, "
                   f"window_size={window_size}")
        
        # リターンを計算
        returns = price_data['close'].pct_change().dropna().values
        
        # 固定長ウィンドウに分割
        segments = []
        for i in range(len(returns) - window_size + 1):
            segment = returns[i:i + window_size]
            segments.append(segment)
        
        segments = np.array(segments)
        
        # 正規化
        scaler = TimeSeriesScalerMeanVariance()
        segments_scaled = scaler.fit_transform(segments)
        
        # TimeSeriesKMeans（DTW距離）
        self.model = TimeSeriesKMeans(
            n_clusters=self.n_regimes,
            metric="dtw",
            max_iter=10,
            random_state=42,
            n_jobs=-1
        )
        
        self.model.fit(segments_scaled)
        self.scaler = scaler
        
        logger.info("✓ 時系列クラスタリング訓練完了")
        logger.info(f"  イナーシャ: {self.model.inertia_:.4f}")
    
    def predict_regime_clustering(self, 
                                 recent_returns: np.ndarray) -> Tuple[int, np.ndarray]:
        """
        時系列クラスタリングで現在のレジームを予測
        
        Args:
            recent_returns: 直近のリターン系列
        
        Returns:
            (予測レジーム, 各クラスタへの距離)
        """
        if self.model is None or self.scaler is None:
            raise ValueError("モデルが訓練されていません。train_clustering()を先に実行してください。")
        
        # 正規化
        segment = recent_returns.reshape(1, -1, 1)
        segment_scaled = self.scaler.transform(segment)
        
        # 予測
        regime = self.model.predict(segment_scaled)[0]
        
        # 各クラスタ中心への距離
        distances = self.model.transform(segment_scaled)[0]
        
        return int(regime), distances
    
    # ========== 統合インターフェース ==========
    
    def detect_current_regime(self, market_data: pd.DataFrame) -> Dict[str, Any]:
        """
        現在の市場レジームを検知（統合インターフェース）
        
        Args:
            market_data: 市場データ（直近のOHLCV）
        
        Returns:
            レジーム情報辞書
                {
                    'regime': int,
                    'regime_name': str,
                    'confidence': float or np.ndarray,
                    'risk_params': Dict[str, Any]
                }
        """
        if self.method == 'hmm':
            # 特徴量抽出
            features = self.extract_features(market_data)
            
            # HMMで予測
            regime, posterior = self.predict_regime_hmm(features[-20:])  # 直近20日
            confidence = posterior[regime]
        
        elif self.method == 'clustering':
            # リターン系列を取得
            returns = market_data['close'].pct_change().dropna().values
            
            # クラスタリングで予測
            regime, distances = self.predict_regime_clustering(returns[-60:])  # 直近60日
            confidence = 1.0 / (1.0 + distances[regime])  # 距離を信頼度に変換
        
        else:
            raise ValueError(f"未対応の手法: {self.method}")
        
        # レジーム名とパラメータを取得
        regime_info = self.regime_params.get(regime, self.regime_params[0])
        
        result = {
            'regime': regime,
            'regime_name': regime_info['name'],
            'confidence': float(confidence),
            'risk_params': {
                'kelly_fraction': regime_info['kelly_fraction'],
                'atr_multiplier_sl': regime_info['atr_multiplier_sl'],
                'atr_multiplier_tp': regime_info['atr_multiplier_tp'],
                'confidence_threshold': regime_info['confidence_threshold']
            }
        }
        
        logger.info(f"検知されたレジーム: {regime_info['name']} (信頼度: {confidence:.2%})")
        
        return result
    
    # ========== モデルの保存・読み込み ==========
    
    def save_model(self, path: str) -> None:
        """モデルを保存"""
        model_data = {
            'method': self.method,
            'n_regimes': self.n_regimes,
            'model': self.model,
            'scaler': self.scaler
        }
        
        joblib.dump(model_data, path)
        logger.info(f"✓ モデル保存完了: {path}")
    
    def load_model(self, path: str) -> None:
        """モデルを読み込み"""
        model_data = joblib.load(path)
        
        self.method = model_data['method']
        self.n_regimes = model_data['n_regimes']
        self.model = model_data['model']
        self.scaler = model_data['scaler']
        
        logger.info(f"✓ モデル読み込み完了: {path}")


# ========== 補助関数 ==========

def analyze_regimes(detector: MarketRegimeDetector,
                   price_data: pd.DataFrame) -> pd.DataFrame:
    """
    全期間のレジームを分析
    
    Args:
        detector: 訓練済みのMarketRegimeDetector
        price_data: 価格データ
    
    Returns:
        レジーム付き価格データ
    """
    if detector.method == 'hmm':
        features = detector.extract_features(price_data)
        features_scaled = detector.scaler.transform(features)
        regimes = detector.model.predict(features_scaled)
        
        # 元のデータフレームにレジームを追加
        result = price_data.iloc[len(price_data) - len(regimes):].copy()
        result['regime'] = regimes
    
    elif detector.method == 'clustering':
        # 実装省略（各ウィンドウごとに予測が必要）
        result = price_data.copy()
        result['regime'] = -1
    
    return result


# 使用例
if __name__ == '__main__':
    # サンプルデータの生成
    np.random.seed(42)
    n_samples = 1000

    dates = pd.date_range(start='2020-01-01', periods=n_samples, freq='D')

    # シンプルな価格シミュレーション
    returns = np.random.randn(n_samples) * 0.01
    prices = 100 * np.exp(np.cumsum(returns))

    sample_data = pd.DataFrame({
        'date': dates,
        'open': prices * (1 + np.random.randn(n_samples) * 0.001),
        'high': prices * (1 + abs(np.random.randn(n_samples)) * 0.005),
        'low': prices * (1 - abs(np.random.randn(n_samples)) * 0.005),
        'close': prices,
        'volume': np.random.randint(1000000, 5000000, n_samples)
    })

    # HMMによるレジーム検知
    print("=" * 60)
    print("HMMによるレジーム検知")
    print("=" * 60)

    detector_hmm = MarketRegimeDetector(method='hmm', n_regimes=4)

    # 訓練
    features = detector_hmm.extract_features(sample_data)
    detector_hmm.train_hmm(features)

    # 現在のレジーム検知
    current_regime = detector_hmm.detect_current_regime(sample_data.tail(100))
    print(f"\n現在のレジーム:")
    print(f"  名前: {current_regime['regime_name']}")
    print(f"  信頼度: {current_regime['confidence']:.2%}")
    print(f"  リスクパラメータ:")
    for key, value in current_regime['risk_params'].items():
        print(f"    {key}: {value}")

    # モデル保存
    model_path = config.S7_REGIME_MODEL
    model_path.parent.mkdir(parents=True, exist_ok=True)
    detector_hmm.save_model(str(model_path))

    print(f"\n✓ すべてのテスト完了")