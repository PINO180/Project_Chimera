# /workspace/execution/state_manager.py
"""
状態管理とフォールトトレランス
チェックポインティングとイベントソーシングによる状態永続化
+ ブローカー状態整合性対応版 [V3.2]
+ ドローダウン管理フィールド追加 [V3.3]
"""

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

import blueprint as config
import json
import pickle
import sqlite3
from pathlib import Path
from datetime import datetime, timezone
from typing import Dict, Any, List, Optional, Union, TypedDict
from dataclasses import dataclass, asdict
from enum import Enum
import logging

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
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
class BrokerPositionDict(TypedDict, total=False):
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


class BrokerStateDict(TypedDict, total=False):
    """ブローカー状態の型定義 [V3.2 修正]"""

    equity: float
    balance: float
    margin: float
    free_margin: float
    positions: List[BrokerPositionDict]


@dataclass
class Trade:
    """ポジション情報"""

    ticket: int
    symbol: str
    direction: str  # 'BUY' or 'SELL'
    lots: float
    entry_price: float
    stop_loss: float
    take_profit: float
    entry_time: datetime
    unrealized_pnl: float = 0.0


@dataclass
class SystemState:
    """システム状態のスナップショット"""

    timestamp: datetime
    current_equity: float
    current_balance: float
    current_margin: float
    free_margin: float
    current_drawdown: float  # [V3.3 追加]
    trades: List[Trade]

    # [V3.3 追加] 他の統計情報も拡張しやすいように kwargs を受け取る
    m1_rolling_precision: List[float] = None
    m2_rolling_auc: List[float] = None
    recent_trades_count: int = 0

    def __post_init__(self):
        if self.m1_rolling_precision is None:
            self.m1_rolling_precision = []
        if self.m2_rolling_auc is None:
            self.m2_rolling_auc = []

    def to_dict(self) -> Dict[str, Any]:
        """辞書形式に変換"""
        return {
            "timestamp": self.timestamp.isoformat(),
            "current_equity": self.current_equity,
            "current_balance": self.current_balance,
            "current_margin": self.current_margin,
            "free_margin": self.free_margin,
            "current_drawdown": self.current_drawdown,  # [V3.3 追加]
            "recent_trades_count": self.recent_trades_count,  # [V3.3 追加]
            "trades": [
                {
                    "ticket": t.ticket,
                    "symbol": t.symbol,
                    "direction": t.direction,
                    "lots": t.lots,
                    "entry_price": t.entry_price,
                    "stop_loss": t.stop_loss,
                    "take_profit": t.take_profit,
                    "entry_time": t.entry_time.isoformat(),
                    "unrealized_pnl": t.unrealized_pnl,
                }
                for t in self.trades
            ],
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SystemState":
        """辞書から復元"""
        trades = [
            Trade(
                ticket=t["ticket"],
                symbol=t["symbol"],
                direction=t["direction"],
                lots=t["lots"],
                entry_price=t["entry_price"],
                stop_loss=t["stop_loss"],
                take_profit=t["take_profit"],
                entry_time=datetime.fromisoformat(t["entry_time"]),
                unrealized_pnl=t.get("unrealized_pnl", 0.0),
            )
            for t in data.get("trades", [])
        ]

        return cls(
            timestamp=datetime.fromisoformat(data["timestamp"]),
            current_equity=data["current_equity"],
            current_balance=data["current_balance"],
            current_margin=data.get("current_margin", 0.0),
            free_margin=data.get("free_margin", 0.0),
            current_drawdown=data.get(
                "current_drawdown", 0.0
            ),  # [V3.3 追加] 安全に取得
            trades=trades,
            recent_trades_count=data.get("recent_trades_count", 0),
        )


class StateManager:
    """
    状態管理マネージャー
    チェックポインティングとイベントソーシングの両方をサポート
    + ブローカー状態整合性対応版 [V3.2]
    + ドローダウン管理対応 [V3.3]
    """

    def __init__(
        self,
        checkpoint_dir: str = str(config.STATE_CHECKPOINT_DIR),
        event_log_path: str = str(config.STATE_EVENT_LOG),
        use_event_sourcing: bool = False,
    ):
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

        logger.info(
            f"StateManager初期化: checkpoint_dir={checkpoint_dir}, "
            f"event_sourcing={use_event_sourcing}"
        )

    # ========== チェックポインティング ==========

    def save_checkpoint(self, state: SystemState, format: str = "json") -> bool:
        """
        現在の状態をチェックポイントとして保存

        Args:
            state: 保存する状態
            format: 保存形式 ('json' or 'pickle')

        Returns:
            保存成功の場合True
        """
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

            if format == "json":
                checkpoint_file = self.checkpoint_dir / f"checkpoint_{timestamp}.json"
                with open(checkpoint_file, "w", encoding="utf-8") as f:
                    json.dump(state.to_dict(), f, indent=2, ensure_ascii=False)

            elif format == "pickle":
                checkpoint_file = self.checkpoint_dir / f"checkpoint_{timestamp}.pkl"
                with open(checkpoint_file, "wb") as f:
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

    def load_checkpoint(self, format: str = "json") -> Optional[SystemState]:
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
                logger.error(
                    f"チェックポイントファイルが存在しません: {checkpoint_path}"
                )
                return None

            if format == "json":
                with open(checkpoint_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                state = SystemState.from_dict(data)

            elif format == "pickle":
                with open(checkpoint_path, "rb") as f:
                    state = pickle.load(f)

            else:
                raise ValueError(f"未対応の形式: {format}")

            self.current_state = state
            logger.info(f"✓ チェックポイント読み込み成功: {checkpoint_path}")
            logger.info(
                f"  エクイティ: {state.current_equity:.2f}, "
                f"DD: {state.current_drawdown:.2%}, "
                f"ポジション数: {len(state.trades)}"
            )

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
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "event_type": event_type.value,
                "data": data,
            }

            with open(self.event_log_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(event, ensure_ascii=False) + "\n")

            logger.debug(f"イベント記録: {event_type.value}")
            return True

        except Exception as e:
            logger.error(f"✗ イベント記録失敗: {e}")
            return False

    def replay_events(
        self, initial_state: Optional[SystemState] = None
    ) -> Optional[SystemState]:
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
                    timestamp=datetime.now(timezone.utc),
                    current_equity=0.0,
                    current_balance=0.0,
                    current_margin=0.0,
                    free_margin=0.0,
                    current_drawdown=0.0,  # [V3.3]
                    trades=[],
                )
            else:
                state = initial_state

            # イベントログを読み込んでリプレイ
            event_count = 0
            with open(self.event_log_path, "r", encoding="utf-8") as f:
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
        event_type = EventType(event["event_type"])
        data = event["data"]
        timestamp = datetime.fromisoformat(event["timestamp"])

        if event_type == EventType.FILL_CONFIRMED:
            # ポジション追加
            trade = Trade(
                ticket=data["ticket"],
                symbol=data["symbol"],
                direction=data["direction"],
                lots=data["lots"],
                entry_price=data["entry_price"],
                stop_loss=data["stop_loss"],
                take_profit=data["take_profit"],
                entry_time=datetime.fromisoformat(data["entry_time"]),
                unrealized_pnl=data.get("unrealized_pnl", 0.0),
            )
            state.trades.append(trade)
            state.recent_trades_count += 1  # [V3.3]

        elif event_type == EventType.POSITION_CLOSED:
            # ポジション削除
            ticket = data["ticket"]
            state.trades = [t for t in state.trades if t.ticket != ticket]

        elif event_type == EventType.EQUITY_UPDATED:
            state.current_equity = data["equity"]
            state.current_balance = data["balance"]
            state.current_margin = data.get("margin", 0.0)
            state.free_margin = data.get("free_margin", 0.0)

        elif event_type == EventType.DRAWDOWN_UPDATED:  # [V3.3]
            state.current_drawdown = data.get("drawdown", 0.0)

        state.timestamp = timestamp
        return state

    # ========== ブローカー状態整合性検証 [V3.2 修正] ==========

    def reconcile_with_broker(self, broker_state: BrokerStateDict) -> bool:
        """
        ローカル状態とブローカー状態の整合性を確保

        ✨ [V3.2 修正] positions キーが存在しない場合の処理を追加
        ✨ [修正版] ブローカーにあってローカルにないポジションのインポート処理を追加

        Args:
            broker_state: ブローカーから取得した状態

        Returns:
            整合性確保成功の場合True
        """
        try:
            if self.current_state is None:
                logger.warning(
                    "ローカル状態が存在しません。ブローカー状態で初期化します。"
                )
                return self._initialize_from_broker(broker_state)

            logger.info("ブローカー状態とローカル状態を整合性検証中...")

            # 口座情報の更新
            equity_diff = abs(
                broker_state.get("equity", 0.0) - self.current_state.current_equity
            )

            # [修正] Balanceもチェックして更新する
            balance_diff = abs(
                broker_state.get("balance", 0.0) - self.current_state.current_balance
            )

            if equity_diff > 0.01 or balance_diff > 0.01:  # 差分があれば更新
                logger.warning(
                    f"口座情報の差分を検出して同期します。\n"
                    f"  Equity: {self.current_state.current_equity:.2f} -> {broker_state.get('equity', 0.0):.2f}\n"
                    f"  Balance: {self.current_state.current_balance:.2f} -> {broker_state.get('balance', 0.0):.2f}"
                )
                self.current_state.current_equity = broker_state.get("equity", 0.0)
                self.current_state.current_balance = broker_state.get(
                    "balance", 0.0
                )  # ★ここを追加

            # ✨ [V3.2 修正] positions キーが存在するかチェック
            broker_positions = broker_state.get("positions", [])

            if not isinstance(broker_positions, list):
                logger.warning(
                    f"broker_state['positions'] が正しいリスト形式ではありません"
                )
                broker_positions = []

            # ポジション整合性チェック
            broker_tickets = {pos.get("ticket") for pos in broker_positions}
            local_tickets = {trade.ticket for trade in self.current_state.trades}

            # 1. ブローカーに存在しないポジションをローカルから削除 (決済検知)
            missing_in_broker = local_tickets - broker_tickets
            if missing_in_broker:
                logger.warning(
                    f"ブローカーに存在しないポジションが検出(削除): {missing_in_broker}"
                )
                # ローカル側から削除
                self.current_state.trades = [
                    trade
                    for trade in self.current_state.trades
                    if trade.ticket not in missing_in_broker
                ]

            # 2. ブローカーにあるがローカルにないポジションを追加 (インポート)
            # ★★★ ここが今回追加される重要な修正箇所です ★★★
            missing_in_local = broker_tickets - local_tickets
            if missing_in_local:
                logger.warning(
                    f"ローカルに存在しないポジションを検出(インポート): {missing_in_local}"
                )
                for pos_data in broker_positions:
                    ticket = pos_data.get("ticket")
                    if ticket in missing_in_local:
                        try:
                            # [修正] MT5の日付形式 "YYYY.MM.DD HH:MM:SS" に対応
                            entry_time_str = pos_data.get("entry_time", "")
                            if "." in entry_time_str:
                                # "2025.11.27 03:00:07" -> datetime
                                entry_time_dt = datetime.strptime(
                                    entry_time_str, "%Y.%m.%d %H:%M:%S"
                                )
                                # タイムゾーン情報を付与 (UTCと仮定、またはJSTなら調整)
                                entry_time_dt = entry_time_dt.replace(
                                    tzinfo=timezone.utc
                                )
                            else:
                                # ISO形式の場合のフォールバック
                                entry_time_dt = datetime.fromisoformat(entry_time_str)

                            # Tradeオブジェクトを作成して追加
                            new_trade = Trade(
                                ticket=ticket,
                                symbol=pos_data.get("symbol", "UNKNOWN"),
                                direction=pos_data.get("direction", "BUY"),
                                lots=pos_data.get("lots", 0.0),
                                entry_price=pos_data.get("entry_price", 0.0),
                                stop_loss=pos_data.get("stop_loss", 0.0),
                                take_profit=pos_data.get("take_profit", 0.0),
                                entry_time=entry_time_dt,
                                unrealized_pnl=pos_data.get("unrealized_pnl", 0.0),
                            )
                            self.current_state.trades.append(new_trade)

                            # 取引回数カウントも増やしておく
                            self.current_state.recent_trades_count += 1

                            logger.info(
                                f"  -> ポジションをインポートしました: Ticket={ticket}"
                            )

                        except Exception as e:
                            logger.error(
                                f"  -> ポジションインポート失敗 (Ticket={ticket}): {e}"
                            )

            # 状態を保存
            self.save_checkpoint(self.current_state)

            logger.info("✓ 状態の整合性を確保しました。")
            return True

        except Exception as e:
            logger.error(f"整合性検証に失敗: {e}", exc_info=True)
            return False

    def _initialize_from_broker(self, broker_state: BrokerStateDict) -> bool:
        """
        ブローカー状態からシステム状態を初期化

        ✨ [V3.2 修正] positions キーが存在しない場合の処理を追加
        ✨ [V3.3 修正] current_drawdown を初期化

        Args:
            broker_state: ブローカー状態辞書

        Returns:
            初期化成功の場合True
        """
        try:
            logger.info("ブローカー状態からシステム状態を初期化中...")

            # 口座情報の取得
            equity = broker_state.get("equity", 0.0)
            balance = broker_state.get("balance", 0.0)
            margin = broker_state.get("margin", 0.0)
            free_margin = broker_state.get("free_margin", 0.0)

            logger.info(f"  Equity: {equity:.2f}")
            logger.info(f"  Balance: {balance:.2f}")
            logger.info(f"  Margin: {margin:.2f}")
            logger.info(f"  Free Margin: {free_margin:.2f}")

            # ✨ [V3.2 修正] positions キーが存在するかチェック
            positions = broker_state.get("positions", [])

            if not isinstance(positions, list):
                logger.warning(
                    f"positions が正しいリスト形式ではありません: {type(positions)}"
                )
                positions = []

            logger.info(f"  ポジション数: {len(positions)}")

            # システム状態を構築
            trades = []
            if positions:
                for pos_data in positions:
                    try:
                        trade = Trade(
                            ticket=pos_data.get("ticket", 0),
                            symbol=pos_data.get("symbol", "UNKNOWN"),
                            direction=pos_data.get("direction", "BUY"),
                            lots=pos_data.get("lots", 0.0),
                            entry_price=pos_data.get("entry_price", 0.0),
                            stop_loss=pos_data.get("stop_loss", 0.0),
                            take_profit=pos_data.get("take_profit", 0.0),
                            entry_time=datetime.fromisoformat(
                                pos_data.get(
                                    "entry_time", datetime.now(timezone.utc).isoformat()
                                )
                            ),
                            unrealized_pnl=pos_data.get("unrealized_pnl", 0.0),
                        )
                        trades.append(trade)
                        logger.debug(
                            f"    ポジション追加: {trade.symbol} {trade.direction} {trade.lots}ロット"
                        )
                    except Exception as e:
                        logger.warning(f"    ポジション解析エラー: {e}")
            else:
                logger.info("  アクティブなポジションがありません。")

            self.current_state = SystemState(
                timestamp=datetime.now(timezone.utc),
                current_equity=equity,
                current_balance=balance,
                current_margin=margin,
                free_margin=free_margin,
                current_drawdown=0.0,  # [V3.3 追加] 初期化時は0.0
                trades=trades,
            )

            # チェックポイント保存
            self.save_checkpoint(self.current_state)

            logger.info("✓ ブローカー状態から初期化完了")
            return True

        except Exception as e:
            logger.error(f"ブローカー状態からの初期化に失敗: {e}", exc_info=True)
            return False


# 使用例
if __name__ == "__main__":
    # チェックポインティングのテスト
    print("=" * 60)
    print("チェックポインティングのテスト (V3.3)")
    print("=" * 60)

    manager = StateManager(use_event_sourcing=False)

    # サンプル状態の作成
    sample_state = SystemState(
        timestamp=datetime.now(timezone.utc),
        current_equity=1050000.0,
        current_balance=1000000.0,
        current_margin=50000.0,
        free_margin=950000.0,
        current_drawdown=0.05,  # [V3.3]
        trades=[
            Trade(
                ticket=12345,
                symbol="XAUUSD",
                direction="BUY",
                lots=1.0,
                entry_price=2050.25,
                stop_loss=2045.40,
                take_profit=2055.95,
                entry_time=datetime.now(timezone.utc),
                unrealized_pnl=5000.0,
            )
        ],
    )

    # チェックポイント保存
    manager.save_checkpoint(sample_state)

    # チェックポイント読み込み
    loaded_state = manager.load_checkpoint()

    if loaded_state:
        print(f"\n復元された状態:")
        print(f"  エクイティ: {loaded_state.current_equity:.2f}")
        print(f"  Drawdown: {loaded_state.current_drawdown:.2%}")
        print(f"  保有ポジション: {len(loaded_state.trades)}個")

    print("\n✓ すべてのテスト完了")
