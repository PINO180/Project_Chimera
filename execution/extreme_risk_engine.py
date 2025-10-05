"""
極限リスク管理エンジン 2.0
ケリー基準、確率キャリブレーション、状態管理、市場レジーム適応を統合
"""
import config
import numpy as np
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional, List, Tuple, TypedDict
import logging
import joblib
from sklearn.calibration import CalibratedClassifierCV
from collections import deque

# 独自モジュール
from state_manager import StateManager, SystemState, Position, EventType
from market_regime_detector import MarketRegimeDetector

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


# 型定義
class MarketInfo(TypedDict):
    """市場情報の型定義"""
    current_price: float
    atr: float
    predicted_time: Optional[float]
    volatility_ratio: Optional[float]  # GARCH予測 / 平均ATR


class ExtremeRiskEngineV2:
    """
    極限リスク管理エンジン 2.0
    数学的原則に基づいた最適資本配分と動的市場適応
    """
    
    def __init__(self,
                config_path: str = str(config.CONFIG_RISK),
                state_manager: Optional[StateManager] = None,
                regime_detector: Optional[MarketRegimeDetector] = None,
                m1_model_path: Optional[str] = None,
                m2_model_path: Optional[str] = None):
        """
        Args:
            config_path: リスク管理設定ファイル
            state_manager: 状態管理マネージャー
            regime_detector: 市場レジーム検知器
            m1_model_path: M1較正済みモデルのパス
            m2_model_path: M2較正済みモデルのパス
        """
        self.config = self._load_config(config_path)

        # 状態管理
        self.state_manager = state_manager or StateManager()

        # 市場レジーム検知
        self.regime_detector = regime_detector

        # 較正済みモデル
        self.m1_calibrated: Optional[CalibratedClassifierCV] = None
        self.m2_calibrated: Optional[CalibratedClassifierCV] = None

        # M1性能履歴（ローリング統計用）
        self.m1_precision_history: deque = deque(maxlen=20)
        self.m1_f1_history: deque = deque(maxlen=20)

        if m1_model_path:
            self.load_calibrated_model(m1_model_path, model_type='M1')
        if m2_model_path:
            self.load_calibrated_model(m2_model_path, model_type='M2')

        logger.info("ExtremeRiskEngineV2を初期化しました。")
    
    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """リスク管理設定を読み込む"""
        config_file = Path(config_path)
        
        if not config_file.exists():
            logger.warning(f"設定ファイル '{config_path}' が見つかりません。デフォルト設定を使用します。")
            default_config = {
                'base_risk_percent': 0.02,
                'max_drawdown': 0.20,
                'drawdown_reduction_threshold': 0.15,
                'base_confidence_threshold': 0.60,
                'min_risk_reward_ratio': 2.0,
                'base_atr_sl_multiplier': 1.0,
                'base_atr_tp_multiplier': 2.0,
                'kelly_fraction': 0.5,  # ハーフケリー
                'max_risk_per_trade': 0.05,  # 1取引あたり最大5%
                'max_positions': 3,
                'contract_size': 100000,
                'pip_value_per_lot': 10,
                'pip_multiplier': 100.0,  # 価格差→pips変換係数（USD/JPY: 100, XAU/USD: 100, EUR/USD: 10000）
                'time_filters': {
                    'blocked_hours_before_news': 0.5,
                    'blocked_hours_after_news': 0.5,
                    'blocked_hours_weekend_close': 4,
                    'blocked_hours_weekend_open': 2
                }
            }
            
            config_file.parent.mkdir(parents=True, exist_ok=True)
            with open(config_file, 'w') as f:
                json.dump(default_config, f, indent=2)
            
            return default_config
        
        with open(config_file, 'r') as f:
            return json.load(f)
    
    # ========== モデル管理 ==========
    
    def load_calibrated_model(self, model_path: str, model_type: str) -> None:
        """
        較正済みモデルを読み込む
        
        Args:
            model_path: モデルファイルパス
            model_type: 'M1' or 'M2'
        """
        try:
            model = joblib.load(model_path)
            
            if model_type == 'M1':
                self.m1_calibrated = model
                logger.info(f"✓ M1較正済みモデル読み込み: {model_path}")
            elif model_type == 'M2':
                self.m2_calibrated = model
                logger.info(f"✓ M2較正済みモデル読み込み: {model_path}")
            else:
                raise ValueError(f"無効なモデルタイプ: {model_type}")
        
        except Exception as e:
            logger.error(f"✗ モデル読み込み失敗 ({model_type}): {e}")
    
    def predict_with_confidence(self, 
                               features: np.ndarray,
                               market_info: MarketInfo) -> Tuple[float, float]:
        """
        較正済みモデルで予測と確信度を取得
        
        Args:
            features: 基本特徴量ベクトル
            market_info: 市場情報（ATR、ボラティリティ等）
        
        Returns:
            (M1予測確率, M2成功確率)
        """
        if self.m1_calibrated is None or self.m2_calibrated is None:
            raise ValueError("較正済みモデルが読み込まれていません。")
        
        # M1予測（利食い確率）
        p_m1 = self.m1_calibrated.predict_proba(features)[0, 1]
        
        # M2用拡張特徴量の構築
        # 1. 基本特徴量
        # 2. M1出力
        # 3. M1性能特徴量（ローリング統計）
        # 4. 市場レジーム特徴量
        
        extended_features_list = [features[0]]  # 基本特徴量
        
        # M1出力
        extended_features_list.append(p_m1)
        
        # M1性能特徴量（ウォークフォワード、リーケージ防止）
        if len(self.m1_precision_history) > 0:
            m1_rolling_precision = float(np.mean(self.m1_precision_history))
            m1_rolling_f1 = float(np.mean(self.m1_f1_history))
        else:
            m1_rolling_precision = 0.5  # デフォルト
            m1_rolling_f1 = 0.5
        
        extended_features_list.extend([m1_rolling_precision, m1_rolling_f1])
        
        # 市場レジーム特徴量
        extended_features_list.extend([
            market_info['atr'],
            market_info.get('volatility_ratio', 1.0),
        ])
        
        # 配列に変換
        extended_features = np.array([extended_features_list])
        
        # M2予測（M1シグナルの成功確率）
        p_m2 = self.m2_calibrated.predict_proba(extended_features)[0, 1]
        
        return float(p_m1), float(p_m2)
    
    def update_m1_performance(self, precision: float, f1_score: float) -> None:
        """
        M1モデルの性能統計を更新
        
        Args:
            precision: 最新のプレシジョン
            f1_score: 最新のF1スコア
        """
        self.m1_precision_history.append(precision)
        self.m1_f1_history.append(f1_score)
        logger.debug(f"M1性能更新: Precision={precision:.4f}, F1={f1_score:.4f}")
    
    def calculate_volatility_adjustment(self,
                                       current_volatility: float,
                                       historical_avg: float) -> float:
        """
        ボラティリティ比率による調整係数を計算（GARCH適応）
        
        Args:
            current_volatility: 現在のボラティリティ（GARCH予測値）
            historical_avg: 過去平均ATR
        
        Returns:
            調整係数（0.7〜1.2）
        """
        if historical_avg <= 0:
            logger.warning("過去平均ATRが無効です。調整係数1.0を返します。")
            return 1.0
        
        volatility_ratio = current_volatility / historical_avg
        
        # 設定から閾値を取得（既存実装との互換性）
        vol_config = {
            'high_threshold': 1.5,
            'high_multiplier': 0.7,
            'low_threshold': 0.5,
            'low_multiplier': 1.2
        }
        
        if volatility_ratio > vol_config['high_threshold']:
            adjustment = vol_config['high_multiplier']
            logger.info(f"高ボラティリティ検出（比率: {volatility_ratio:.2f}）。"
                       f"ロット調整: {adjustment}")
        elif volatility_ratio < vol_config['low_threshold']:
            adjustment = vol_config['low_multiplier']
            logger.info(f"低ボラティリティ検出（比率: {volatility_ratio:.2f}）。"
                       f"ロット調整: {adjustment}")
        else:
            adjustment = 1.0
        
        return adjustment
    
    def calculate_kelly_fraction(self,
                                 p_win: float,
                                 win_loss_ratio: float,
                                 kelly_fraction: float = 0.5) -> float:
        """
        ケリー基準で最適な資本配分比率を計算
        
        Args:
            p_win: 勝利確率（較正済み）
            win_loss_ratio: 勝敗比率（利食い幅 / 損切り幅）
            kelly_fraction: ケリー分数（0.5でハーフケリー、0.25でクォーターケリー）
        
        Returns:
            最適資本配分比率（0-1）
        """
        # ケリー基準の公式: f* = (b*p - q) / b
        # b = win_loss_ratio, p = p_win, q = 1 - p_win
        
        if win_loss_ratio <= 0 or p_win <= 0 or p_win >= 1:
            logger.warning("ケリー基準の計算に無効なパラメータ")
            return 0.0
        
        q_lose = 1.0 - p_win
        
        # ケリー基準
        kelly_f = (win_loss_ratio * p_win - q_lose) / win_loss_ratio
        
        # 負の値（期待値マイナス）の場合は0
        if kelly_f <= 0:
            logger.info(f"期待値マイナスのシグナル（Kelly={kelly_f:.4f}）。エントリー不可。")
            return 0.0
        
        # 分数ケリー（リスク軽減）
        fractional_kelly = kelly_f * kelly_fraction
        
        logger.debug(f"Kelly計算: p_win={p_win:.4f}, b={win_loss_ratio:.2f}, "
                    f"f*={kelly_f:.4f}, fractional={fractional_kelly:.4f}")
        
        return fractional_kelly
    
    def calculate_position_size_kelly(self,
                                     account_balance: float,
                                     p_m2: float,
                                     entry_price: float,
                                     stop_loss_price: float,
                                     take_profit_price: float,
                                     current_drawdown: float = 0.0,
                                     volatility_adjustment: float = 1.0,
                                     regime_params: Optional[Dict[str, Any]] = None) -> Tuple[float, Dict[str, Any]]:
        """
        ケリー基準でポジションサイズを計算
        
        Args:
            account_balance: 口座残高
            p_m2: M2の成功確率（較正済み）
            entry_price: エントリー価格
            stop_loss_price: 損切り価格
            take_profit_price: 利食い価格
            current_drawdown: 現在のドローダウン（0-1）
            volatility_adjustment: ボラティリティ調整係数
            regime_params: 市場レジーム別パラメータ
        
        Returns:
            (ロット数, 計算詳細)
        """
        # レジームパラメータの取得
        if regime_params:
            kelly_fraction = regime_params['kelly_fraction']
        else:
            kelly_fraction = self.config['kelly_fraction']
        
        # ドローダウンチェック
        if current_drawdown >= self.config['max_drawdown']:
            logger.warning(f"最大ドローダウン超過（{current_drawdown:.2%}）。エントリー不可。")
            return 0.0, {'reason': 'max_drawdown_exceeded'}
        
        if current_drawdown >= self.config['drawdown_reduction_threshold']:
            kelly_fraction *= 0.5
            logger.info(f"ドローダウン警戒レベル（{current_drawdown:.2%}）。Kelly分数を50%削減。")
        
        # 勝敗比率の計算
        sl_distance = abs(entry_price - stop_loss_price)
        tp_distance = abs(entry_price - take_profit_price)
        
        if sl_distance <= 0:
            logger.error("損切り幅が無効です。")
            return 0.0, {'reason': 'invalid_sl_distance'}
        
        win_loss_ratio = tp_distance / sl_distance
        
        # ケリー基準で最適比率を計算
        kelly_f = self.calculate_kelly_fraction(p_m2, win_loss_ratio, kelly_fraction)
        
        if kelly_f <= 0:
            return 0.0, {'reason': 'negative_expected_value'}
        
        # 最大リスク制約
        risk_fraction = min(kelly_f, self.config['max_risk_per_trade'])
        
        # リスク額
        risk_amount = account_balance * risk_fraction
        
        # ロット計算（pips単位）
        sl_distance_pips = sl_distance * 100  # USD/JPY想定
        lots = risk_amount / (sl_distance_pips * self.config['pip_value_per_lot'])
        
        # ボラティリティ調整を適用
        lots *= volatility_adjustment
        
        # 0.01単位に丸める
        lots = round(lots, 2)
        
        # 最小ロット確認
        if lots < 0.01:
            logger.warning("計算されたロットが最小値未満です。")
            return 0.0, {'reason': 'lots_below_minimum'}
        
        details = {
            'kelly_f_raw': kelly_f / kelly_fraction,  # 生ケリー
            'kelly_f_fractional': kelly_f,
            'risk_fraction': risk_fraction,
            'risk_amount': risk_amount,
            'win_loss_ratio': win_loss_ratio,
            'sl_distance_pips': sl_distance_pips,
            'volatility_adjustment': volatility_adjustment,
            'reason': 'success'
        }
        
        logger.info(f"Kelly最適ロット: {lots:.2f} (リスク: {risk_fraction:.2%}, "
                   f"ボラ調整: {volatility_adjustment:.2f})")
        
        return lots, details
    
    # ========== 損切り・利食いの計算 ==========
    
    def calculate_sl_tp(self,
                       entry_price: float,
                       atr: float,
                       direction: str,
                       predicted_time: Optional[float] = None,
                       regime_params: Optional[Dict[str, Any]] = None) -> Dict[str, float]:
        """
        損切り・利食いラインを計算
        
        Args:
            entry_price: エントリー価格
            atr: 現在のATR値
            direction: 取引方向（'BUY' or 'SELL'）
            predicted_time: AI予測の到達時間（バー数）
            regime_params: 市場レジーム別パラメータ
        
        Returns:
            {'stop_loss': float, 'take_profit': float}
        """
        # レジームパラメータの取得
        if regime_params:
            sl_multiplier = regime_params['atr_multiplier_sl']
            tp_multiplier = regime_params['atr_multiplier_tp']
        else:
            sl_multiplier = self.config['base_atr_sl_multiplier']
            tp_multiplier = self.config['base_atr_tp_multiplier']
        
        # 予測時間に基づく動的調整
        if predicted_time is not None:
            if predicted_time < 30:
                tp_multiplier *= 1.25
            elif predicted_time > 90:
                tp_multiplier *= 0.75
        
        # 計算
        if direction == 'BUY':
            stop_loss = entry_price - sl_multiplier * atr
            take_profit = entry_price + tp_multiplier * atr
        elif direction == 'SELL':
            stop_loss = entry_price + sl_multiplier * atr
            take_profit = entry_price - tp_multiplier * atr
        else:
            raise ValueError(f"無効な取引方向: {direction}")
        
        return {
            'stop_loss': round(stop_loss, 2),
            'take_profit': round(take_profit, 2)
        }
    
    # ========== エントリー条件チェック ==========
    
    def check_entry_conditions(self,
                              p_m2: float,
                              current_positions: int,
                              current_time: Optional[datetime] = None,
                              news_times: Optional[List[datetime]] = None,
                              regime_params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        エントリー条件をチェック
        
        Args:
            p_m2: M2成功確率
            current_positions: 現在の保有ポジション数
            current_time: 現在時刻
            news_times: 経済指標発表時刻リスト
            regime_params: 市場レジーム別パラメータ
        
        Returns:
            {'allowed': bool, 'reason': str}
        """
        # 確信度チェック（レジーム適応）
        if regime_params:
            threshold = regime_params['confidence_threshold']
        else:
            threshold = self.config['base_confidence_threshold']
        
        if p_m2 < threshold:
            return {
                'allowed': False,
                'reason': f"確信度不足（{p_m2:.2%} < {threshold:.2%}）"
            }
        
        # ポジション数チェック
        if current_positions >= self.config['max_positions']:
            return {
                'allowed': False,
                'reason': f"最大ポジション数到達（{current_positions}/{self.config['max_positions']}）"
            }
        
        # 時間帯フィルター
        if current_time is not None:
            # ニュース前後
            if news_times is not None:
                for news_time in news_times:
                    time_diff = abs((current_time - news_time).total_seconds() / 3600)
                    blocked_hours = self.config['time_filters']['blocked_hours_before_news']
                    if time_diff < blocked_hours:
                        return {
                            'allowed': False,
                            'reason': f"経済指標発表前後の取引禁止時間帯（{time_diff:.1f}時間前）"
                        }
            
            # 週末前後
            if current_time.weekday() == 4 and current_time.hour >= 21:
                return {
                    'allowed': False,
                    'reason': "週末クローズ前の取引禁止時間帯"
                }
            
            if current_time.weekday() == 0 and current_time.hour < 9:
                return {
                    'allowed': False,
                    'reason': "週明けオープン後の取引禁止時間帯"
                }
        
        return {'allowed': True, 'reason': 'すべての条件をクリア'}
    
    # ========== 統合取引コマンド生成 ==========
    
    def generate_trade_command(self,
                              features: np.ndarray,
                              market_info: MarketInfo,
                              market_data_for_regime: Optional[Any] = None,
                              current_time: Optional[datetime] = None,
                              news_times: Optional[List[datetime]] = None) -> Dict[str, Any]:
        """
        取引コマンドを生成（統合メイン関数）
        
        Args:
            features: AI予測用の基本特徴量
            market_info: 市場情報
            market_data_for_regime: 市場レジーム検知用のデータ（pd.DataFrame）
            current_time: 現在時刻
            news_times: 経済指標発表時刻リスト
        
        Returns:
            取引コマンド辞書
        """
        # 状態の取得
        if self.state_manager.current_state is None:
            logger.error("システム状態が初期化されていません。")
            return self._generate_hold_command("state_not_initialized")
        
        state = self.state_manager.current_state
        
        # 市場レジームの検知
        regime_params = None
        if self.regime_detector and market_data_for_regime is not None:
            try:
                regime_info = self.regime_detector.detect_current_regime(market_data_for_regime)
                regime_params = regime_info['risk_params']
                logger.info(f"現在の市場レジーム: {regime_info['regime_name']}")
            except Exception as e:
                logger.warning(f"レジーム検知失敗: {e}。デフォルトパラメータを使用。")
        
        # AI予測（較正済み確率）
        try:
            p_m1, p_m2 = self.predict_with_confidence(features, market_info)
            logger.info(f"AI予測: P(M1)={p_m1:.4f}, P(M2)={p_m2:.4f}")
        except Exception as e:
            logger.error(f"AI予測失敗: {e}")
            return self._generate_hold_command("prediction_failed")
        
        # エントリー条件チェック
        entry_check = self.check_entry_conditions(
            p_m2=p_m2,
            current_positions=len(state.open_positions),
            current_time=current_time,
            news_times=news_times,
            regime_params=regime_params
        )
        
        if not entry_check['allowed']:
            logger.info(f"エントリー不可: {entry_check['reason']}")
            return self._generate_hold_command(entry_check['reason'])
        
        # 取引方向の決定
        direction = 'BUY' if p_m1 > 0.5 else 'SELL'
        
        # SL/TPの計算
        sl_tp = self.calculate_sl_tp(
            entry_price=market_info['current_price'],
            atr=market_info['atr'],
            direction=direction,
            predicted_time=market_info.get('predicted_time'),
            regime_params=regime_params
        )
        
        # ボラティリティ調整係数の計算
        volatility_adjustment = 1.0
        if market_info.get('volatility_ratio') is not None:
            # GARCHベースのボラティリティ適応
            volatility_adjustment = self.calculate_volatility_adjustment(
                current_volatility=market_info['atr'],
                historical_avg=market_info['atr'] / market_info['volatility_ratio']
            )
        
        # ポジションサイズの計算（ケリー基準）
        lots, calc_details = self.calculate_position_size_kelly(
            account_balance=state.current_balance,
            p_m2=p_m2,
            entry_price=market_info['current_price'],
            stop_loss_price=sl_tp['stop_loss'],
            take_profit_price=sl_tp['take_profit'],
            current_drawdown=state.current_drawdown,
            volatility_adjustment=volatility_adjustment,
            regime_params=regime_params
        )
        
        if lots <= 0:
            logger.info(f"ポジションサイズ計算でエントリー不可: {calc_details.get('reason')}")
            return self._generate_hold_command(calc_details.get('reason', 'zero_position_size'))
        
        # 取引コマンドの生成
        trade_command = {
            'action': direction,
            'lots': lots,
            'entry_price': market_info['current_price'],
            'stop_loss': sl_tp['stop_loss'],
            'take_profit': sl_tp['take_profit'],
            'confidence_m2': p_m2,
            'confidence_m1': p_m1,
            'predicted_time': market_info.get('predicted_time', 0),
            'reason': (f"M2確信度{p_m2:.2%}, Kelly={calc_details['kelly_f_fractional']:.2%}, "
                      f"リスク{calc_details['risk_fraction']:.2%}, {direction}シグナル"),
            'risk_amount': calc_details['risk_amount'],
            'win_loss_ratio': calc_details['win_loss_ratio'],
            'kelly_fraction': calc_details['kelly_f_fractional'],
            'volatility_adjustment': calc_details['volatility_adjustment'],
            'timestamp': (current_time.isoformat() if current_time else datetime.now().isoformat())
        }
        
        logger.info(f"✓ 取引コマンド生成: {direction} {lots}ロット @ {market_info['current_price']:.2f}")
        logger.info(f"  SL: {sl_tp['stop_loss']:.2f}, TP: {sl_tp['take_profit']:.2f}")
        logger.info(f"  M2確信度: {p_m2:.2%}, Kelly: {calc_details['kelly_f_fractional']:.2%}")
        
        # イベント記録
        if self.state_manager.use_event_sourcing:
            self.state_manager.append_event(
                EventType.TRADE_SIGNAL_SENT,
                {'command': trade_command}
            )
        
        return trade_command
    
    def _generate_hold_command(self, reason: str) -> Dict[str, Any]:
        """HOLDコマンドを生成"""
        return {
            'action': 'HOLD',
            'lots': 0.0,
            'entry_price': 0.0,
            'stop_loss': 0.0,
            'take_profit': 0.0,
            'confidence_m2': 0.0,
            'confidence_m1': 0.0,
            'predicted_time': 0,
            'reason': reason,
            'risk_amount': 0.0,
            'timestamp': datetime.now().isoformat()
        }


# 使用例
if __name__ == '__main__':
    import pandas as pd
    
    # 状態管理の初期化
    state_manager = StateManager(use_event_sourcing=True)
    
    # 初期状態の設定
    initial_state = SystemState(
        timestamp=datetime.now().isoformat(),
        current_equity=1000000.0,
        current_balance=1000000.0,
        current_drawdown=0.0,
        open_positions={},
        m1_rolling_precision=[],
        m2_rolling_auc=[],
        recent_trades_count=0
    )
    state_manager.current_state = initial_state
    state_manager.save_checkpoint(initial_state)
    
    # リスクエンジンの初期化
    engine = ExtremeRiskEngineV2(
        state_manager=state_manager
    )
    
    # サンプルデータ
    sample_features = np.random.randn(1, 50)  # ダミー特徴量
    
    market_info_typed: MarketInfo = {
        'current_price': 150.25,
        'atr': 0.85,
        'predicted_time': 45.0,
        'volatility_ratio': 1.2
    }
    
    # 注: 実際の使用では較正済みモデルとレジーム検知器が必要
    print("=" * 60)
    print("取引コマンド生成テスト")
    print("=" * 60)
    print("\n注: このテストは較正済みモデルなしで実行されます。")
    print("実運用では M1/M2 較正済みモデルが必須です。\n")
    
    # モデルなしではHOLDになる
    command = engine.generate_trade_command(
        features=sample_features,
        market_info=market_info_typed,
        current_time=datetime.now()
    )
    
    print(f"\n生成されたコマンド:")
    print(json.dumps(command, indent=2, ensure_ascii=False))
    
    print("\n✓ テスト完了")