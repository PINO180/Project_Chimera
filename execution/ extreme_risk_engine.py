import numpy as np
import json
from pathlib import Path
from datetime import datetime, time
from typing import Dict, Any, Optional, List
import logging

# ロギング設定
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class ExtremeRiskEngine:
    """
    極限リスク管理エンジン
    AIの予測結果から最適なポジションサイズとリスクパラメータを計算し、取引コマンドを生成
    """
    
    def __init__(self, config_path: str = 'config/risk_config.json'):
        """
        Args:
            config_path: リスク管理設定ファイルのパス
        """
        self.config = self._load_config(config_path)
        logger.info(f"ExtremeRiskEngineを初期化しました。設定: {config_path}")
    
    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """リスク管理設定をJSONファイルから読み込む"""
        config_file = Path(config_path)
        
        if not config_file.exists():
            # デフォルト設定を作成
            logger.warning(f"設定ファイル '{config_path}' が見つかりません。デフォルト設定を使用します。")
            default_config = {
                'risk_percent': 0.02,  # 口座残高の2%をリスクに晒す
                'max_drawdown': 0.20,  # 最大ドローダウン20%
                'drawdown_reduction_threshold': 0.15,  # 15%でロット縮小
                'confidence_threshold': 0.60,  # エントリー最低確信度
                'min_risk_reward_ratio': 2.0,  # 最小リスクリワード比
                'atr_sl_multiplier': 1.0,  # 損切りATR倍率
                'atr_tp_multiplier': 2.0,  # 利食いATR倍率
                'volatility_adjustment': {
                    'high_threshold': 1.5,  # 高ボラティリティ閾値
                    'high_multiplier': 0.7,  # 高ボラ時のロット倍率
                    'low_threshold': 0.5,   # 低ボラティリティ閾値
                    'low_multiplier': 1.2    # 低ボラ時のロット倍率
                },
                'time_filters': {
                    'blocked_hours_before_news': 0.5,  # ニュース前30分
                    'blocked_hours_after_news': 0.5,   # ニュース後30分
                    'blocked_hours_weekend_close': 4,   # 週末クローズ前4時間
                    'blocked_hours_weekend_open': 2     # 週明けオープン後2時間
                },
                'max_positions': 3,  # 同時保有ポジション数
                'contract_size': 100000,  # 1ロットの契約サイズ（標準）
                'pip_value_per_lot': 10  # 1pipあたりの価値（USD/JPY想定）
            }
            
            # デフォルト設定を保存
            config_file.parent.mkdir(parents=True, exist_ok=True)
            with open(config_file, 'w') as f:
                json.dump(default_config, f, indent=2)
            
            return default_config
        
        with open(config_file, 'r') as f:
            return json.load(f)
    
    def calculate_position_size(self,
                               account_balance: float,
                               entry_price: float,
                               stop_loss_price: float,
                               confidence: float,
                               current_drawdown: float = 0.0,
                               volatility_ratio: float = 1.0) -> float:
        """
        ポジションサイズを計算
        
        Args:
            account_balance: 口座残高
            entry_price: エントリー価格
            stop_loss_price: 損切り価格
            confidence: AIの確信度（0-1）
            current_drawdown: 現在のドローダウン（0-1）
            volatility_ratio: ボラティリティ比率（GARCH予測値 / 平均ATR）
        
        Returns:
            計算されたロット数
        """
        # 基本リスク額の計算
        base_risk_amount = account_balance * self.config['risk_percent']
        
        # ドローダウンによる調整
        if current_drawdown >= self.config['max_drawdown']:
            logger.warning(f"最大ドローダウン超過（{current_drawdown:.2%}）。エントリー不可。")
            return 0.0
        
        if current_drawdown >= self.config['drawdown_reduction_threshold']:
            base_risk_amount *= 0.5
            logger.info(f"ドローダウン警戒レベル（{current_drawdown:.2%}）。リスクを50%削減。")
        
        # 損切り幅の計算（pips）
        sl_distance_pips = abs(entry_price - stop_loss_price) * 100  # USD/JPY想定
        
        if sl_distance_pips <= 0:
            logger.error("損切り幅が無効です。")
            return 0.0
        
        # 基本ロットサイズの計算
        base_lots = base_risk_amount / (sl_distance_pips * self.config['pip_value_per_lot'])
        
        # 確信度による調整
        confidence_adjustment = (confidence - self.config['confidence_threshold']) / \
                               (1.0 - self.config['confidence_threshold'])
        adjusted_lots = base_lots * (1.0 + confidence_adjustment * 0.5)  # 最大1.5倍
        
        # ボラティリティによる調整
        vol_config = self.config['volatility_adjustment']
        if volatility_ratio > vol_config['high_threshold']:
            # 高ボラティリティ環境
            adjusted_lots *= vol_config['high_multiplier']
            logger.info(f"高ボラティリティ検出（比率: {volatility_ratio:.2f}）。ロットを{vol_config['high_multiplier']}倍に調整。")
        elif volatility_ratio < vol_config['low_threshold']:
            # 低ボラティリティ環境
            adjusted_lots *= vol_config['low_multiplier']
            logger.info(f"低ボラティリティ検出（比率: {volatility_ratio:.2f}）。ロットを{vol_config['low_multiplier']}倍に調整。")
        
        # ロット数を0.01単位に丸める
        final_lots = round(adjusted_lots, 2)
        
        # 最小ロット確認
        if final_lots < 0.01:
            logger.warning("計算されたロットが最小値未満です。")
            return 0.0
        
        return final_lots
    
    def calculate_sl_tp(self,
                       entry_price: float,
                       atr: float,
                       direction: str,
                       predicted_time: Optional[float] = None) -> Dict[str, float]:
        """
        損切り・利食いラインを計算
        
        Args:
            entry_price: エントリー価格
            atr: 現在のATR値
            direction: 取引方向（'BUY' or 'SELL'）
            predicted_time: AI予測の到達時間（バー数）
        
        Returns:
            {'stop_loss': float, 'take_profit': float}
        """
        # 基本的なSL/TP倍率
        sl_multiplier = self.config['atr_sl_multiplier']
        tp_multiplier = self.config['atr_tp_multiplier']
        
        # 予測時間に基づく動的調整
        if predicted_time is not None:
            if predicted_time < 30:
                # 短時間予測: 利食いを拡大
                tp_multiplier *= 1.25
            elif predicted_time > 90:
                # 長時間予測: 利食いを縮小
                tp_multiplier *= 0.75
        
        if direction == 'BUY':
            stop_loss = entry_price - sl_multiplier * atr
            take_profit = entry_price + tp_multiplier * atr
        elif direction == 'SELL':
            stop_loss = entry_price + sl_multiplier * atr
            take_profit = entry_price - tp_multiplier * atr
        else:
            raise ValueError(f"無効な取引方向: {direction}")
        
        # 価格を2桁に丸める（USD/JPY想定）
        return {
            'stop_loss': round(stop_loss, 2),
            'take_profit': round(take_profit, 2)
        }
    
    def check_entry_conditions(self,
                              confidence: float,
                              prob_profit: float,
                              prob_loss: float,
                              current_positions: int,
                              current_time: Optional[datetime] = None,
                              news_times: Optional[List[datetime]] = None) -> Dict[str, Any]:
        """
        エントリー条件をチェック
        
        Args:
            confidence: 確信度
            prob_profit: 利食い確率
            prob_loss: 損切り確率
            current_positions: 現在の保有ポジション数
            current_time: 現在時刻
            news_times: 経済指標発表時刻リスト
        
        Returns:
            {'allowed': bool, 'reason': str}
        """
        # 確信度チェック
        if confidence < self.config['confidence_threshold']:
            return {
                'allowed': False,
                'reason': f"確信度不足（{confidence:.2%} < {self.config['confidence_threshold']:.2%}）"
            }
        
        # リスクリワード比チェック
        if prob_loss > 0:
            risk_reward = prob_profit / prob_loss
            if risk_reward < self.config['min_risk_reward_ratio']:
                return {
                    'allowed': False,
                    'reason': f"リスクリワード比不足（{risk_reward:.2f} < {self.config['min_risk_reward_ratio']:.2f}）"
                }
        
        # ポジション数チェック
        if current_positions >= self.config['max_positions']:
            return {
                'allowed': False,
                'reason': f"最大ポジション数到達（{current_positions}/{self.config['max_positions']}）"
            }
        
        # 時間帯フィルター
        if current_time is not None:
            # ニュース前後のチェック
            if news_times is not None:
                for news_time in news_times:
                    time_diff = abs((current_time - news_time).total_seconds() / 3600)
                    blocked_hours = self.config['time_filters']['blocked_hours_before_news']
                    if time_diff < blocked_hours:
                        return {
                            'allowed': False,
                            'reason': f"経済指標発表前後の取引禁止時間帯（{time_diff:.1f}時間前）"
                        }
            
            # 週末クローズ前後のチェック（金曜日21:00以降、月曜日9:00まで想定）
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
    
    def generate_trade_command(self,
                              ai_prediction: Dict[str, Any],
                              account_info: Dict[str, float],
                              market_info: Dict[str, float],
                              current_time: Optional[datetime] = None,
                              news_times: Optional[List[datetime]] = None) -> Dict[str, Any]:
        """
        取引コマンドを生成（メイン関数）
        
        Args:
            ai_prediction: AIモデルの予測結果
                {
                    'prob_profit': float,  # 利食い確率
                    'prob_loss': float,    # 損切り確率
                    'prob_time': float,    # 時間切れ確率
                    'predicted_time': float  # 予測到達時間
                }
            account_info: 口座情報
                {
                    'balance': float,           # 口座残高
                    'drawdown': float,          # 現在のドローダウン（0-1）
                    'current_positions': int    # 現在の保有ポジション数
                }
            market_info: 市場情報
                {
                    'current_price': float,   # 現在価格
                    'atr': float,             # ATR値
                    'volatility_ratio': float # ボラティリティ比率（GARCH/平均ATR）
                }
            current_time: 現在時刻
            news_times: 経済指標発表時刻リスト
        
        Returns:
            取引コマンド辞書
        """
        # 確信度の計算（利食い確率をそのまま使用）
        confidence = ai_prediction['prob_profit']
        
        # エントリー条件チェック
        entry_check = self.check_entry_conditions(
            confidence=confidence,
            prob_profit=ai_prediction['prob_profit'],
            prob_loss=ai_prediction['prob_loss'],
            current_positions=account_info['current_positions'],
            current_time=current_time,
            news_times=news_times
        )
        
        if not entry_check['allowed']:
            logger.info(f"エントリー不可: {entry_check['reason']}")
            return {
                'action': 'HOLD',
                'lots': 0.0,
                'stop_loss': 0.0,
                'take_profit': 0.0,
                'confidence': confidence,
                'reason': entry_check['reason'],
                'timestamp': current_time.isoformat() if current_time else None
            }
        
        # 取引方向の決定（利食い確率が高ければBUY）
        direction = 'BUY' if ai_prediction['prob_profit'] > 0.5 else 'SELL'
        
        # SL/TPの計算
        sl_tp = self.calculate_sl_tp(
            entry_price=market_info['current_price'],
            atr=market_info['atr'],
            direction=direction,
            predicted_time=ai_prediction.get('predicted_time')
        )
        
        # ポジションサイズの計算
        lots = self.calculate_position_size(
            account_balance=account_info['balance'],
            entry_price=market_info['current_price'],
            stop_loss_price=sl_tp['stop_loss'],
            confidence=confidence,
            current_drawdown=account_info['drawdown'],
            volatility_ratio=market_info.get('volatility_ratio', 1.0)
        )
        
        if lots <= 0:
            return {
                'action': 'HOLD',
                'lots': 0.0,
                'stop_loss': 0.0,
                'take_profit': 0.0,
                'confidence': confidence,
                'reason': 'ポジションサイズ計算でエントリー不可',
                'timestamp': current_time.isoformat() if current_time else None
            }
        
        # 最終的な取引コマンド
        trade_command = {
            'action': direction,
            'lots': lots,
            'entry_price': market_info['current_price'],
            'stop_loss': sl_tp['stop_loss'],
            'take_profit': sl_tp['take_profit'],
            'confidence': confidence,
            'predicted_time': ai_prediction.get('predicted_time', 0),
            'reason': f"確信度{confidence:.2%}、リスク{self.config['risk_percent']:.1%}、{direction}シグナル",
            'risk_amount': account_info['balance'] * self.config['risk_percent'],
            'timestamp': current_time.isoformat() if current_time else None
        }
        
        logger.info(f"取引コマンド生成: {direction} {lots}ロット @ {market_info['current_price']:.2f}")
        logger.info(f"  SL: {sl_tp['stop_loss']:.2f}, TP: {sl_tp['take_profit']:.2f}")
        logger.info(f"  確信度: {confidence:.2%}")
        
        return trade_command


# 使用例
if __name__ == '__main__':
    # リスクエンジンの初期化
    engine = ExtremeRiskEngine(config_path='config/risk_config.json')
    
    # サンプルデータ
    ai_prediction = {
        'prob_profit': 0.72,    # 72%の確率で利食い
        'prob_loss': 0.18,      # 18%の確率で損切り
        'prob_time': 0.10,      # 10%の確率で時間切れ
        'predicted_time': 45.0  # 45バーで決着予測
    }
    
    account_info = {
        'balance': 1000000.0,      # 100万円
        'drawdown': 0.05,          # 5%のドローダウン
        'current_positions': 1     # 1ポジション保有中
    }
    
    market_info = {
        'current_price': 150.25,   # USD/JPY
        'atr': 0.85,               # ATR値
        'volatility_ratio': 1.2    # やや高めのボラティリティ
    }
    
    # 取引コマンド生成
    command = engine.generate_trade_command(
        ai_prediction=ai_prediction,
        account_info=account_info,
        market_info=market_info,
        current_time=datetime.now()
    )
    
    print("\n生成された取引コマンド:")
    print(json.dumps(command, indent=2, ensure_ascii=False))