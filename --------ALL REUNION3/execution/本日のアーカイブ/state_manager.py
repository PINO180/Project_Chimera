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
    """ポジション情報 [V5 Observability拡張]"""

    ticket: int
    symbol: str
    direction: str  # 'BUY' or 'SELL'
    lots: float
    entry_price: float
    stop_loss: float
    take_profit: float
    entry_time: datetime
    unrealized_pnl: float = 0.0

    # --- V5 Observability ---
    active_longs_at_entry: int = 0
    active_shorts_at_entry: int = 0

    def get_duration_minutes(self, close_time: datetime) -> float:
        """
        [V5追加] エントリーからの経過時間を分単位で算出
        """
        safe_close_time = close_time
        if safe_close_time.tzinfo is None:
            safe_close_time = safe_close_time.replace(tzinfo=timezone.utc)

        safe_entry_time = self.entry_time
        if safe_entry_time.tzinfo is None:
            safe_entry_time = safe_entry_time.replace(tzinfo=timezone.utc)

        delta = safe_close_time - safe_entry_time
        return delta.total_seconds() / 60.0


@dataclass
class SystemState:
    """システム状態のスナップショット [V5 Circuit Breaker拡張・完全同期版]"""

    timestamp: datetime
    current_equity: float
    current_balance: float
    current_margin: float
    free_margin: float
    current_drawdown: float
    trades: List[Trade]  # ▼ 移動: デフォルト値を持たない引数を上に配置
    high_water_mark: float = 0.0  # デフォルト値あり

    m1_rolling_precision: List[float] = None
    m2_rolling_auc: List[float] = None
    recent_trades_count: int = 0

    # --- V5 Circuit Breaker (バックテスター互換) ---
    consecutive_sl_long: int = 0
    consecutive_sl_short: int = 0
    cooldown_until_long: Optional[datetime] = None  # 変更: 解除時刻を直接保持
    cooldown_until_short: Optional[datetime] = None  # 変更: 解除時刻を直接保持

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
            "current_drawdown": self.current_drawdown,
            "high_water_mark": self.high_water_mark,
            "recent_trades_count": self.recent_trades_count,
            # --- V5 追加分 ---
            "consecutive_sl_long": self.consecutive_sl_long,
            "consecutive_sl_short": self.consecutive_sl_short,
            "cooldown_until_long": self.cooldown_until_long.isoformat()
            if self.cooldown_until_long
            else None,
            "cooldown_until_short": self.cooldown_until_short.isoformat()
            if self.cooldown_until_short
            else None,
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
                    "active_longs_at_entry": t.active_longs_at_entry,
                    "active_shorts_at_entry": t.active_shorts_at_entry,
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
                active_longs_at_entry=t.get("active_longs_at_entry", 0),
                active_shorts_at_entry=t.get("active_shorts_at_entry", 0),
            )
            for t in data.get("trades", [])
        ]

        cd_long_str = data.get("cooldown_until_long")
        cd_short_str = data.get("cooldown_until_short")

        return cls(
            timestamp=datetime.fromisoformat(data["timestamp"]),
            current_equity=data["current_equity"],
            current_balance=data["current_balance"],
            current_margin=data.get("current_margin", 0.0),
            free_margin=data.get("free_margin", 0.0),
            current_drawdown=data.get("current_drawdown", 0.0),
            high_water_mark=data.get(
                "high_water_mark", data.get("current_equity", 0.0)
            ),
            trades=trades,
            recent_trades_count=data.get("recent_trades_count", 0),
            consecutive_sl_long=data.get("consecutive_sl_long", 0),
            consecutive_sl_short=data.get("consecutive_sl_short", 0),
            cooldown_until_long=datetime.fromisoformat(cd_long_str)
            if cd_long_str
            else None,
            cooldown_until_short=datetime.fromisoformat(cd_short_str)
            if cd_short_str
            else None,
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

        logger.debug(
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
                # [FIX-INFO] pickle は任意コード実行リスクあり。本番では json を推奨。
                logger.warning(
                    "pickle形式のチェックポイントはセキュリティリスクがあります。json形式の使用を推奨します。"
                )
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
            logger.debug(
                f"✓ チェックポイント保存成功: {checkpoint_file}"
            )  # infoをdebugに

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
            logger.info(f"✓ チェックポイント読み込み成功: {checkpoint_path.name}")
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
        from datetime import timedelta

        event_type = EventType(event["event_type"])
        data = event["data"]
        # [FIX-9] タイムゾーン一貫性: naive datetime は UTC として扱う
        raw_ts = datetime.fromisoformat(event["timestamp"])
        if raw_ts.tzinfo is None:
            raw_ts = raw_ts.replace(tzinfo=timezone.utc)
        timestamp = raw_ts

        if event_type == EventType.FILL_CONFIRMED:
            active_longs = sum(1 for t in state.trades if t.direction == "BUY")
            active_shorts = sum(1 for t in state.trades if t.direction == "SELL")

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
                active_longs_at_entry=active_longs,
                active_shorts_at_entry=active_shorts,
            )
            state.trades.append(trade)
            state.recent_trades_count += 1

        elif event_type == EventType.POSITION_CLOSED:
            ticket = data["ticket"]
            close_reason = data.get("close_reason", "UNKNOWN")  # 'SL', 'PT', 'TO' 等

            # ライブ実行側からイベントを送る際、現在適用中のリスクパラメータを含める
            max_sl = data.get("max_consecutive_sl", 2)
            cooldown_mins = data.get("cooldown_minutes_after_sl", 30)

            closed_trade = next((t for t in state.trades if t.ticket == ticket), None)

            if closed_trade:
                if close_reason == "SL":
                    if closed_trade.direction == "BUY":
                        state.consecutive_sl_long += 1
                        logger.warning(
                            f"LongポジションSL被弾。連続SL回数(Long): {state.consecutive_sl_long}"
                        )
                        if state.consecutive_sl_long >= max_sl:
                            state.cooldown_until_long = timestamp + timedelta(
                                minutes=cooldown_mins
                            )
                            state.consecutive_sl_long = 0  # カウンターリセット
                            logger.info(
                                f"Longのサーキットブレーカー発動。{state.cooldown_until_long} までロックアウトします。"
                            )

                    elif closed_trade.direction == "SELL":
                        state.consecutive_sl_short += 1
                        logger.warning(
                            f"ShortポジションSL被弾。連続SL回数(Short): {state.consecutive_sl_short}"
                        )
                        if state.consecutive_sl_short >= max_sl:
                            state.cooldown_until_short = timestamp + timedelta(
                                minutes=cooldown_mins
                            )
                            state.consecutive_sl_short = 0  # カウンターリセット
                            logger.info(
                                f"Shortのサーキットブレーカー発動。{state.cooldown_until_short} までロックアウトします。"
                            )

                elif close_reason in ["PT", "TO"]:
                    # PTまたはTOでカウンターをリセット（バックテストの仕様と一致）
                    if closed_trade.direction == "BUY":
                        state.consecutive_sl_long = 0
                    elif closed_trade.direction == "SELL":
                        state.consecutive_sl_short = 0

            # ポジション削除
            state.trades = [t for t in state.trades if t.ticket != ticket]

        elif event_type == EventType.EQUITY_UPDATED:
            # [FIX-6] EQUITY_UPDATED ハンドラ実装
            new_equity = data.get("equity")
            new_balance = data.get("balance")
            new_margin = data.get("margin")
            new_free_margin = data.get("free_margin")
            if new_equity is not None:
                state.current_equity = float(new_equity)
            if new_balance is not None:
                state.current_balance = float(new_balance)
            if new_margin is not None:
                state.current_margin = float(new_margin)
            if new_free_margin is not None:
                state.free_margin = float(new_free_margin)

        elif event_type == EventType.DRAWDOWN_UPDATED:
            # [FIX-6] DRAWDOWN_UPDATED ハンドラ実装
            dd = data.get("current_drawdown")
            if dd is not None:
                state.current_drawdown = float(dd)

        elif event_type == EventType.STOP_LOSS_HIT:
            # [FIX-6] STOP_LOSS_HIT ハンドラ実装 (POSITION_CLOSED と同等処理)
            ticket = data.get("ticket")
            closed_trade = next((t for t in state.trades if t.ticket == ticket), None)
            if closed_trade:
                from datetime import timedelta

                max_sl = data.get("max_consecutive_sl", 2)
                cooldown_mins = data.get("cooldown_minutes_after_sl", 30)
                if closed_trade.direction == "BUY":
                    state.consecutive_sl_long += 1
                    if state.consecutive_sl_long >= max_sl:
                        state.cooldown_until_long = timestamp + timedelta(
                            minutes=cooldown_mins
                        )
                        state.consecutive_sl_long = 0
                elif closed_trade.direction == "SELL":
                    state.consecutive_sl_short += 1
                    if state.consecutive_sl_short >= max_sl:
                        state.cooldown_until_short = timestamp + timedelta(
                            minutes=cooldown_mins
                        )
                        state.consecutive_sl_short = 0
            state.trades = [t for t in state.trades if t.ticket != ticket]

        elif event_type == EventType.TAKE_PROFIT_HIT:
            # [FIX-6] TAKE_PROFIT_HIT ハンドラ実装
            ticket = data.get("ticket")
            closed_trade = next((t for t in state.trades if t.ticket == ticket), None)
            if closed_trade:
                if closed_trade.direction == "BUY":
                    state.consecutive_sl_long = 0
                elif closed_trade.direction == "SELL":
                    state.consecutive_sl_short = 0
            state.trades = [t for t in state.trades if t.ticket != ticket]

        elif event_type == EventType.MODEL_PERFORMANCE_UPDATED:
            # [FIX-6] MODEL_PERFORMANCE_UPDATED ハンドラ実装
            m1_prec = data.get("m1_rolling_precision")
            m2_auc = data.get("m2_rolling_auc")
            if m1_prec is not None:
                state.m1_rolling_precision = list(m1_prec)
            if m2_auc is not None:
                state.m2_rolling_auc = list(m2_auc)

        # SYSTEM_STARTED / SYSTEM_STOPPED / TRADE_SIGNAL_SENT はタイムスタンプのみ更新

        state.timestamp = timestamp
        return state

    # ========== V5 サーキットブレーカー & Observability ==========

    def is_cooldown_active(self, direction: str) -> bool:
        """
        [V5] 現在指定した方向がクールダウン（ロックアウト）中かどうかを判定

        Args:
            direction: 'BUY' または 'SELL'

        Returns:
            クールダウン中の場合True
        """
        if self.current_state is None:
            return False

        now = datetime.now(timezone.utc)

        if direction == "BUY":
            cd_until = self.current_state.cooldown_until_long
        elif direction == "SELL":
            cd_until = self.current_state.cooldown_until_short
        else:
            return False

        if cd_until:
            # [FIX-9] naive datetime を UTC として扱いタイムゾーン比較エラーを防ぐ
            if cd_until.tzinfo is None:
                cd_until = cd_until.replace(tzinfo=timezone.utc)
            if now < cd_until:
                elapsed = (cd_until - now).total_seconds() / 60.0
                logger.info(
                    f"サーキットブレーカー作動中 ({direction}): 残り {elapsed:.1f} 分"
                )
                return True

        return False

    def apply_event_and_update(
        self, event_type: EventType, data: Dict[str, Any]
    ) -> bool:
        """
        [FIX-8] パブリックAPIとしてイベントを適用し内部状態を更新する。
        main.py が _apply_event を直接呼ぶ代わりにこちらを使用すること。
        """
        try:
            if self.current_state is None:
                logger.warning(
                    "apply_event_and_update: current_state が None です。スキップします。"
                )
                return False

            from datetime import timezone as _tz
            import json as _json

            event_dict = {
                "event_type": event_type.value,
                "data": data,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }
            self.current_state = self._apply_event(self.current_state, event_dict)
            self.append_event(event_type, data)
            return True
        except Exception as e:
            logger.error(f"apply_event_and_update 失敗: {e}", exc_info=True)
            return False

    def get_active_positions_count(self, direction: str) -> int:
        """
        [V5] 指定した方向の現在のアクティブポジション数を取得
        (prevent_simultaneous_orders の判定に使用)

        Args:
            direction: 'BUY' または 'SELL'

        Returns:
            アクティブなポジション数
        """
        if self.current_state is None:
            return 0

        return sum(1 for t in self.current_state.trades if t.direction == direction)

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

            logger.debug(
                "ブローカー状態とローカル状態を整合性検証中..."
            )  # infoをdebugに

            # 口座情報の更新 (値の更新自体は裏で常に実行する)
            new_equity = broker_state.get("equity", 0.0)
            new_balance = broker_state.get("balance", 0.0)

            equity_diff = abs(new_equity - self.current_state.current_equity)
            balance_diff = abs(new_balance - self.current_state.current_balance)

            # バランス（残高）が変わった時だけINFOでログを出す（決済検知時など）
            if balance_diff > 0.01:
                logger.info(
                    f"口座残高(Balance)の変動を同期: "
                    f"{self.current_state.current_balance:.2f} -> {new_balance:.2f} "
                    f"(Equity: {new_equity:.2f})"
                )

            # エクイティの変動は毎秒起こるためDEBUGレベルに降格（通常時は非表示）
            if equity_diff > 0.01 or balance_diff > 0.01:
                logger.debug(
                    f"口座情報を同期: Equity {self.current_state.current_equity:.2f} -> {new_equity:.2f}"
                )
                self.current_state.current_equity = new_equity
                self.current_state.current_balance = new_balance

            # HWM と Drawdown の自動更新を毎回の整合性検証時に実行
            self.current_state.high_water_mark = max(
                self.current_state.high_water_mark, self.current_state.current_equity
            )
            if self.current_state.high_water_mark > 0:
                self.current_state.current_drawdown = (
                    self.current_state.high_water_mark
                    - self.current_state.current_equity
                ) / self.current_state.high_water_mark

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
                                if entry_time_dt.tzinfo is None:
                                    entry_time_dt = entry_time_dt.replace(
                                        tzinfo=timezone.utc
                                    )

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

            logger.debug("✓ 状態の整合性を確保しました。")  # infoをdebugに
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
                high_water_mark=equity,  # [V3.3 修正] HWM初期化
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
