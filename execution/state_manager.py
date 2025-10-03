"""
状態管理とフォールトトレランス
チェックポインティングとイベントソーシングによる状態永続化
"""
import json
import pickle
import sqlite3
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List, Optional, Union, TypedDict
from dataclasses import dataclass, asdict
from enum import Enum
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class EventType(Enum):
    """イベントタイプの定義"""
    TRADE_SIGNAL_SENT = "TRADE_SIGNAL_SENT"
    FILL_CONFIRMED = "FILL_CONFIRMED"
    EQUITY_UPDATED = "EQUITY_UPDATED"
    STOP_LOSS_HIT = "STOP_LOSS_HIT"
    TAKE_PROFIT_HIT = "TAKE_PROFIT_HIT"
    POSITION_CLOSED = "POSITION_CLOSED"
    DRAWDOWN_UPDATED = "DRAWDOWN_UPDATED"
    MODEL_PERFORMANCE_UPDATED = "MODEL_PERFORMANCE_UPDATED"
    SYSTEM_STARTED = "SYSTEM_STARTED"
    SYSTEM_STOPPED = "SYSTEM_STOPPED"


# 型定義
class BrokerPositionDict(TypedDict):
    """ブローカーポジション情報の型定義"""
    ticket: int
    symbol: str
    direction: str
    lots: float
    entry_price: float
    stop_loss: float
    take_profit: float
    entry_time: str
    unrealized_pnl: float


class BrokerStateDict(TypedDict):
    """ブローカー状態の型定義"""
    equity: float
    balance: float
    positions: List[BrokerPositionDict]


@dataclass
class Position:
    """ポジション情報"""
    ticket: int
    symbol: str
    direction: str  # 'BUY' or 'SELL'
    lots: float
    entry_price: float
    stop_loss: float
    take_profit: float
    entry_time: str
    unrealized_pnl: float = 0.0


@dataclass
class SystemState:
    """システム状態のスナップショット"""
    timestamp: str
    current_equity: float
    current_balance: float
    current_drawdown: float
    open_positions: Dict[int, Position]
    m1_rolling_precision: List[float]
    m2_rolling_auc: List[float]
    recent_trades_count: int
    last_heartbeat_timestamp: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """辞書形式に変換"""
        result = asdict(self)
        result['open_positions'] = {
            k: asdict(v) for k, v in self.open_positions.items()
        }
        return result
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'SystemState':
        """辞書から復元"""
        data['open_positions'] = {
            int(k): Position(**v) for k, v in data['open_positions'].items()
        }
        return cls(**data)


class StateManager:
    """
    状態管理マネージャー
    チェックポインティングとイベントソーシングの両方をサポート
    """
    
    def __init__(self, 
                 checkpoint_dir: str = "data/state",
                 event_log_path: str = "data/state/event_log.jsonl",
                 use_event_sourcing: bool = False):
        """
        Args:
            checkpoint_dir: チェックポイント保存ディレクトリ
            event_log_path: イベントログファイルパス
            use_event_sourcing: イベントソーシングを使用するか
        """
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        self.event_log_path = Path(event_log_path)
        self.event_log_path.parent.mkdir(parents=True, exist_ok=True)
        
        self.use_event_sourcing = use_event_sourcing
        self.current_state: Optional[SystemState] = None
        
        logger.info(f"StateManager初期化: checkpoint_dir={checkpoint_dir}, "
                   f"event_sourcing={use_event_sourcing}")
    
    # ========== チェックポインティング ==========
    
    def save_checkpoint(self, state: SystemState, format: str = 'json') -> bool:
        """
        現在の状態をチェックポイントとして保存
        
        Args:
            state: 保存する状態
            format: 保存形式 ('json' or 'pickle')
        
        Returns:
            保存成功の場合True
        """
        try:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            
            if format == 'json':
                checkpoint_file = self.checkpoint_dir / f"checkpoint_{timestamp}.json"
                with open(checkpoint_file, 'w', encoding='utf-8') as f:
                    json.dump(state.to_dict(), f, indent=2, ensure_ascii=False)
            
            elif format == 'pickle':
                checkpoint_file = self.checkpoint_dir / f"checkpoint_{timestamp}.pkl"
                with open(checkpoint_file, 'wb') as f:
                    pickle.dump(state, f)
            
            else:
                raise ValueError(f"未対応の形式: {format}")
            
            # 最新チェックポイントへのシンボリックリンク更新
            latest_link = self.checkpoint_dir / f"latest_checkpoint.{format}"
            if latest_link.exists():
                latest_link.unlink()
            latest_link.write_text(str(checkpoint_file))
            
            self.current_state = state
            logger.info(f"✓ チェックポイント保存成功: {checkpoint_file}")
            
            # 古いチェックポイントのクリーンアップ（最新10個を保持）
            self._cleanup_old_checkpoints(keep_count=10)
            
            return True
            
        except Exception as e:
            logger.error(f"✗ チェックポイント保存失敗: {e}")
            return False
    
    def load_checkpoint(self, format: str = 'json') -> Optional[SystemState]:
        """
        最新のチェックポイントを読み込み
        
        Args:
            format: 読み込み形式 ('json' or 'pickle')
        
        Returns:
            復元された状態、失敗時None
        """
        try:
            latest_link = self.checkpoint_dir / f"latest_checkpoint.{format}"
            
            if not latest_link.exists():
                logger.warning("チェックポイントが見つかりません。")
                return None
            
            checkpoint_path = Path(latest_link.read_text())
            
            if not checkpoint_path.exists():
                logger.error(f"チェックポイントファイルが存在しません: {checkpoint_path}")
                return None
            
            if format == 'json':
                with open(checkpoint_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                state = SystemState.from_dict(data)
            
            elif format == 'pickle':
                with open(checkpoint_path, 'rb') as f:
                    state = pickle.load(f)
            
            else:
                raise ValueError(f"未対応の形式: {format}")
            
            self.current_state = state
            logger.info(f"✓ チェックポイント読み込み成功: {checkpoint_path}")
            logger.info(f"  エクイティ: {state.current_equity:.2f}, "
                       f"ドローダウン: {state.current_drawdown:.2%}, "
                       f"保有ポジション数: {len(state.open_positions)}")
            
            return state
            
        except Exception as e:
            logger.error(f"✗ チェックポイント読み込み失敗: {e}")
            return None
    
    def _cleanup_old_checkpoints(self, keep_count: int = 10) -> None:
        """古いチェックポイントを削除"""
        try:
            checkpoints = sorted(self.checkpoint_dir.glob("checkpoint_*.json"))
            checkpoints.extend(sorted(self.checkpoint_dir.glob("checkpoint_*.pkl")))
            
            if len(checkpoints) > keep_count:
                for checkpoint in checkpoints[:-keep_count]:
                    checkpoint.unlink()
                    logger.debug(f"古いチェックポイントを削除: {checkpoint}")
        
        except Exception as e:
            logger.warning(f"チェックポイントクリーンアップ失敗: {e}")
    
    # ========== イベントソーシング ==========
    
    def append_event(self, event_type: EventType, data: Dict[str, Any]) -> bool:
        """
        イベントをログに追記
        
        Args:
            event_type: イベントタイプ
            data: イベントデータ
        
        Returns:
            追記成功の場合True
        """
        if not self.use_event_sourcing:
            return True
        
        try:
            event = {
                'timestamp': datetime.now().isoformat(),
                'event_type': event_type.value,
                'data': data
            }
            
            with open(self.event_log_path, 'a', encoding='utf-8') as f:
                f.write(json.dumps(event, ensure_ascii=False) + '\n')
            
            logger.debug(f"イベント記録: {event_type.value}")
            return True
            
        except Exception as e:
            logger.error(f"✗ イベント記録失敗: {e}")
            return False
    
    def replay_events(self, 
                     initial_state: Optional[SystemState] = None) -> Optional[SystemState]:
        """
        イベントログをリプレイして状態を再構築
        
        Args:
            initial_state: 初期状態（Noneの場合はデフォルト状態から開始）
        
        Returns:
            再構築された状態
        """
        if not self.event_log_path.exists():
            logger.warning("イベントログが見つかりません。")
            return initial_state
        
        try:
            # 初期状態の設定
            if initial_state is None:
                state = SystemState(
                    timestamp=datetime.now().isoformat(),
                    current_equity=0.0,
                    current_balance=0.0,
                    current_drawdown=0.0,
                    open_positions={},
                    m1_rolling_precision=[],
                    m2_rolling_auc=[],
                    recent_trades_count=0
                )
            else:
                state = initial_state
            
            # イベントログを読み込んでリプレイ
            event_count = 0
            with open(self.event_log_path, 'r', encoding='utf-8') as f:
                for line in f:
                    event = json.loads(line)
                    state = self._apply_event(state, event)
                    event_count += 1
            
            self.current_state = state
            logger.info(f"✓ イベントリプレイ完了: {event_count}個のイベントを処理")
            
            return state
            
        except Exception as e:
            logger.error(f"✗ イベントリプレイ失敗: {e}")
            return None
    
    def _apply_event(self, state: SystemState, event: Dict[str, Any]) -> SystemState:
        """イベントを状態に適用"""
        event_type = EventType(event['event_type'])
        data = event['data']
        timestamp = event['timestamp']
        
        if event_type == EventType.FILL_CONFIRMED:
            # ポジション追加
            position = Position(**data['position'])
            state.open_positions[position.ticket] = position
            state.recent_trades_count += 1
        
        elif event_type == EventType.POSITION_CLOSED:
            # ポジション削除
            ticket = data['ticket']
            if ticket in state.open_positions:
                del state.open_positions[ticket]
        
        elif event_type == EventType.EQUITY_UPDATED:
            state.current_equity = data['equity']
            state.current_balance = data['balance']
        
        elif event_type == EventType.DRAWDOWN_UPDATED:
            state.current_drawdown = data['drawdown']
        
        elif event_type == EventType.MODEL_PERFORMANCE_UPDATED:
            if 'm1_precision' in data:
                state.m1_rolling_precision.append(data['m1_precision'])
                if len(state.m1_rolling_precision) > 20:
                    state.m1_rolling_precision.pop(0)
            
            if 'm2_auc' in data:
                state.m2_rolling_auc.append(data['m2_auc'])
                if len(state.m2_rolling_auc) > 20:
                    state.m2_rolling_auc.pop(0)
        
        state.timestamp = timestamp
        return state
    
    # ========== ブローカー整合性検証 ==========
    
    def reconcile_with_broker(self, broker_state: BrokerStateDict) -> Dict[str, Any]:
        """
        ローカル状態とブローカー状態の整合性検証
        
        Args:
            broker_state: ブローカーから取得した状態
        
        Returns:
            整合性検証結果
                {
                    'is_consistent': bool,
                    'discrepancies': List[str],
                    'reconciled_state': SystemState
                }
        """
        if self.current_state is None:
            logger.warning("ローカル状態が存在しません。ブローカー状態で初期化します。")
            return self._initialize_from_broker(broker_state)
        
        discrepancies: List[str] = []
        
        # エクイティの検証
        equity_diff = abs(self.current_state.current_equity - broker_state['equity'])
        if equity_diff > 1.0:  # 1ドル以上の差異
            discrepancies.append(
                f"エクイティ不一致: ローカル={self.current_state.current_equity:.2f}, "
                f"ブローカー={broker_state['equity']:.2f}"
            )
        
        # 残高の検証
        balance_diff = abs(self.current_state.current_balance - broker_state['balance'])
        if balance_diff > 1.0:
            discrepancies.append(
                f"残高不一致: ローカル={self.current_state.current_balance:.2f}, "
                f"ブローカー={broker_state['balance']:.2f}"
            )
        
        # ポジションの検証
        broker_tickets = {pos['ticket'] for pos in broker_state['positions']}
        local_tickets = set(self.current_state.open_positions.keys())
        
        missing_in_local = broker_tickets - local_tickets
        missing_in_broker = local_tickets - broker_tickets
        
        if missing_in_local:
            discrepancies.append(
                f"ブローカーに存在するがローカルに無いポジション: {missing_in_local}"
            )
        
        if missing_in_broker:
            discrepancies.append(
                f"ローカルに存在するがブローカーに無いポジション: {missing_in_broker}"
            )
        
        # 整合性の判定
        is_consistent = len(discrepancies) == 0
        
        if not is_consistent:
            logger.warning("状態の不整合を検出:")
            for disc in discrepancies:
                logger.warning(f"  - {disc}")
            logger.info("ブローカー状態を真実の源として、ローカル状態を更新します。")
        
        # 整合状態の構築（ブローカーが真実の源）
        reconciled_positions = {}
        for pos_data in broker_state['positions']:
            position = Position(
                ticket=pos_data['ticket'],
                symbol=pos_data['symbol'],
                direction=pos_data['direction'],
                lots=pos_data['lots'],
                entry_price=pos_data['entry_price'],
                stop_loss=pos_data['stop_loss'],
                take_profit=pos_data['take_profit'],
                entry_time=pos_data['entry_time'],
                unrealized_pnl=pos_data.get('unrealized_pnl', 0.0)
            )
            reconciled_positions[position.ticket] = position
        
        reconciled_state = SystemState(
            timestamp=datetime.now().isoformat(),
            current_equity=broker_state['equity'],
            current_balance=broker_state['balance'],
            current_drawdown=self.current_state.current_drawdown,  # ドローダウンは再計算
            open_positions=reconciled_positions,
            m1_rolling_precision=self.current_state.m1_rolling_precision,
            m2_rolling_auc=self.current_state.m2_rolling_auc,
            recent_trades_count=self.current_state.recent_trades_count,
            last_heartbeat_timestamp=datetime.now().isoformat()
        )
        
        # 整合状態を保存
        self.current_state = reconciled_state
        self.save_checkpoint(reconciled_state)
        
        if is_consistent:
            logger.info("✓ ブローカー整合性検証: 一致")
        else:
            logger.info("✓ ブローカー整合性検証: 調整完了")
        
        return {
            'is_consistent': is_consistent,
            'discrepancies': discrepancies,
            'reconciled_state': reconciled_state
        }
    
    def _initialize_from_broker(self, broker_state: Dict[str, Any]) -> Dict[str, Any]:
        """ブローカー状態からローカル状態を初期化"""
        positions = {}
        for pos_data in broker_state['positions']:
            position = Position(**pos_data)
            positions[position.ticket] = position
        
        initial_state = SystemState(
            timestamp=datetime.now().isoformat(),
            current_equity=broker_state['equity'],
            current_balance=broker_state['balance'],
            current_drawdown=0.0,
            open_positions=positions,
            m1_rolling_precision=[],
            m2_rolling_auc=[],
            recent_trades_count=0,
            last_heartbeat_timestamp=datetime.now().isoformat()
        )
        
        self.current_state = initial_state
        self.save_checkpoint(initial_state)
        
        logger.info("✓ ブローカー状態からローカル状態を初期化しました。")
        
        return {
            'is_consistent': True,
            'discrepancies': [],
            'reconciled_state': initial_state
        }


# 使用例
if __name__ == '__main__':
    # チェックポインティングのテスト
    print("=" * 60)
    print("チェックポインティングのテスト")
    print("=" * 60)
    
    manager = StateManager(use_event_sourcing=False)
    
    # サンプル状態の作成
    sample_state = SystemState(
        timestamp=datetime.now().isoformat(),
        current_equity=1050000.0,
        current_balance=1000000.0,
        current_drawdown=0.05,
        open_positions={
            12345: Position(
                ticket=12345,
                symbol='USDJPY',
                direction='BUY',
                lots=1.0,
                entry_price=150.25,
                stop_loss=149.40,
                take_profit=151.95,
                entry_time=datetime.now().isoformat(),
                unrealized_pnl=5000.0
            )
        },
        m1_rolling_precision=[0.72, 0.75, 0.73],
        m2_rolling_auc=[0.68, 0.70, 0.69],
        recent_trades_count=15
    )
    
    # チェックポイント保存
    manager.save_checkpoint(sample_state)
    
    # チェックポイント読み込み
    loaded_state = manager.load_checkpoint()
    
    if loaded_state:
        print(f"\n復元された状態:")
        print(f"  エクイティ: {loaded_state.current_equity:.2f}")
        print(f"  保有ポジション: {len(loaded_state.open_positions)}個")
    
    # イベントソーシングのテスト
    print("\n" + "=" * 60)
    print("イベントソーシングのテスト")
    print("=" * 60)
    
    es_manager = StateManager(use_event_sourcing=True)
    
    # イベント記録
    es_manager.append_event(EventType.SYSTEM_STARTED, {'version': '2.0'})
    es_manager.append_event(EventType.EQUITY_UPDATED, {'equity': 1000000.0, 'balance': 1000000.0})
    
    # イベントリプレイ
    replayed_state = es_manager.replay_events()
    
    if replayed_state:
        print(f"\nリプレイされた状態:")
        print(f"  エクイティ: {replayed_state.current_equity:.2f}")
    
    print("\n✓ すべてのテスト完了")