"""
MQL5ブリッジ 3.0 - 高信頼性通信
レイジー・パイレートパターンと双方向ハートビートによるミッションクリティカル通信
+ ZMQ履歴データ取得対応版 [V3.0]
+ ZMQ フレームクリーニング対応版 [V3.1]
"""

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

import blueprint as config
import zmq
import json
import time
import threading
from pathlib import Path
from datetime import datetime, timezone
from typing import Dict, Any, Optional, Callable, List
from dataclasses import dataclass
from enum import Enum
import logging
import polars as pl
import pandas as pd

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class MessageType(Enum):
    """メッセージタイプの定義"""

    TRADE_COMMAND = "TRADE_COMMAND"
    PING = "PING"
    PONG = "PONG"
    ACK = "ACK"
    NACK = "NACK"
    REQUEST_BROKER_STATE = "REQUEST_BROKER_STATE"
    BROKER_STATE_RESPONSE = "BROKER_STATE_RESPONSE"
    REQUEST_HISTORY = "REQUEST_HISTORY"
    HISTORY_DATA = "HISTORY_DATA"
    REQUEST_M1_BAR = "REQUEST_M1_BAR"
    M1_BAR_DATA = "M1_BAR_DATA"


@dataclass
class BridgeConfig:
    """通信設定"""

    trade_endpoint: str = "tcp://127.0.0.1:5555"
    heartbeat_endpoint: str = "tcp://127.0.0.1:5556"
    request_timeout: int = 9000  # (9秒)
    request_timeout_large: int = 600000  # 👈 10秒から10分(600,000ms)に延長
    request_retries: int = 3
    heartbeat_interval: int = 10
    heartbeat_timeout: int = 30
    log_dir: str = str(config.LOGS_ZMQ_BRIDGE)


# ✨ [V3.1 新規] ZMQ フレームクリーニング関数
def clean_zmq_json(response_json: str) -> str:
    """
    ZMQ フレームから余分なバイトを削除し、
    正しい JSON オブジェクトだけを抽出

    Args:
        response_json: ZMQから受け取った文字列

    Returns:
        クリーニング済み JSON 文字列
    """
    # JSON オブジェクトの開始と終了を探す
    json_start = response_json.find("{")
    json_end = response_json.rfind("}") + 1

    if json_start == -1 or json_end <= json_start:
        logger.warning(f"JSON構造が不正: {response_json[:100]}")
        raise ValueError("JSON structure invalid")

    return response_json[json_start:json_end]


def parse_response(response_bytes: bytes) -> dict:
    """
    ZMQ レスポンスをパースして辞書に変換

    Args:
        response_bytes: ZMQから受け取ったバイト列

    Returns:
        パース済み辞書
    """
    response_json = response_bytes.decode("utf-8")
    clean_json = clean_zmq_json(response_json)
    return json.loads(clean_json)


class MQL5BridgePublisherV2:
    """
    ZeroMQを介したPython-MQL5高信頼性通信
    レイジー・パイレートパターンに基づく送受信確認と双方向ハートビート
    + ZMQ履歴データ・M1バー取得対応 [V3.0]
    + ZMQ フレームクリーニング対応 [V3.1]
    """

    def __init__(self, config: Optional[BridgeConfig] = None):
        """
        Args:
            config: 通信設定
        """
        self.config = config or BridgeConfig()

        # ログディレクトリ
        self.log_dir = Path(self.config.log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        # ZeroMQコンテキスト
        self.context: Optional[zmq.Context] = None

        # 取引コマンド用ソケット（REQパターン）
        self.trade_socket: Optional[zmq.Socket] = None

        # ハートビート用ソケット（DEALER/ROUTERパターン）
        self.heartbeat_socket: Optional[zmq.Socket] = None

        # 接続状態
        self.is_connected = False
        self.last_heartbeat_sent: Optional[datetime] = None
        self.last_heartbeat_received: Optional[datetime] = None

        # 統計情報
        self.messages_sent = 0
        self.messages_acked = 0
        self.messages_failed = 0
        self.heartbeats_sent = 0
        self.heartbeats_received = 0

        # ハートビートスレッド
        self.heartbeat_thread: Optional[threading.Thread] = None
        self.heartbeat_running = False

        # コールバック
        self.on_connection_lost: Optional[Callable[[], None]] = None
        self.on_connection_restored: Optional[Callable[[], None]] = None

        logger.info("MQL5BridgePublisherV2 (v3.1) を初期化しました。")

    # ========== 接続管理 ==========

    def connect(self) -> bool:
        """
        ZeroMQソケットを初期化し、エンドポイントに接続

        Returns:
            接続成功の場合True
        """
        try:
            logger.info(f"ZeroMQ高信頼性通信を初期化中...")
            logger.info(f"  取引エンドポイント: {self.config.trade_endpoint}")
            logger.info(
                f"  ハートビートエンドポイント: {self.config.heartbeat_endpoint}"
            )

            # コンテキスト作成
            self.context = zmq.Context()

            # 取引コマンド用REQソケット
            self.trade_socket = self.context.socket(zmq.REQ)
            self.trade_socket.connect(self.config.trade_endpoint)

            # ハートビート用REQソケット (REQ/REPパターンに統一)
            self.heartbeat_socket = self.context.socket(zmq.REQ)
            # self.heartbeat_socket.setsockopt(zmq.IDENTITY, b"PythonCore") # 👈 REQソケットにIDは不要
            self.heartbeat_socket.connect(self.config.heartbeat_endpoint)

            self.is_connected = True

            # ハートビートスレッドを開始
            self._start_heartbeat_thread()

            logger.info("✓ ZeroMQ高信頼性通信の起動に成功しました。")

            return True

        except Exception as e:
            logger.error(f"✗ ZeroMQ通信の起動に失敗しました: {e}")
            self.is_connected = False
            return False

    def disconnect(self) -> None:
        """ZeroMQソケットとコンテキストをクリーンアップ"""
        # ハートビートスレッドを停止
        self._stop_heartbeat_thread()

        if self.trade_socket:
            self.trade_socket.close()
            logger.info("取引ソケットをクローズしました。")

        if self.heartbeat_socket:
            self.heartbeat_socket.close()
            logger.info("ハートビートソケットをクローズしました。")

        if self.context:
            self.context.term()
            logger.info("ZeroMQコンテキストを終了しました。")

        self.is_connected = False

    # ========== レイジー・パイレート: 取引コマンド送信 ==========

    def send_trade_command(self, trade_command: Dict[str, Any]) -> bool:
        """
        取引コマンドをREQソケットで送信（レイジー・パイレートパターン）

        Args:
            trade_command: 取引コマンド辞書

        Returns:
            送信成功（ACK受信）の場合True
        """
        if not self.is_connected or self.trade_socket is None:
            logger.error("ZeroMQソケットが接続されていません。")
            self.messages_failed += 1
            return False

        # タイムスタンプとメッセージIDを追加
        if "timestamp" not in trade_command:
            trade_command["timestamp"] = datetime.now().isoformat()

        trade_command["message_id"] = self.messages_sent + 1
        trade_command["message_type"] = MessageType.TRADE_COMMAND.value

        # レイジー・パイレート: リトライループ
        for attempt in range(self.config.request_retries):
            try:
                # JSON形式でメッセージ送信
                json_message = json.dumps(trade_command, ensure_ascii=False)
                self.trade_socket.send_string(json_message)

                logger.debug(
                    f"取引コマンド送信 [#{trade_command['message_id']}, 試行{attempt + 1}]: "
                    f"{trade_command['action']}"
                )

                # Pollerで応答待機（タイムアウト付き）
                poller = zmq.Poller()
                poller.register(self.trade_socket, zmq.POLLIN)

                socks = dict(poller.poll(self.config.request_timeout))

                # 応答受信
                if (
                    self.trade_socket in socks
                    and socks[self.trade_socket] == zmq.POLLIN
                ):
                    response_bytes = self.trade_socket.recv()
                    response_data = parse_response(response_bytes)

                    # ACK確認
                    if response_data.get("message_type") == MessageType.ACK.value:
                        self.messages_sent += 1
                        self.messages_acked += 1

                        self._log_message(trade_command, status="sent_acked")

                        logger.info(
                            f"✓ 取引コマンド送信成功 [#{trade_command['message_id']}]: "
                            f"{trade_command['action']} {trade_command.get('lots', 0):.2f}ロット"
                        )

                        return True

                    # NACK受信
                    elif response_data.get("message_type") == MessageType.NACK.value:
                        logger.warning(
                            f"MQL5 EAから拒否されました: {response_data.get('message')}"
                        )
                        self.messages_failed += 1
                        self._log_message(
                            trade_command,
                            status="nack",
                            error=response_data.get("message"),
                        )
                        return False

                # タイムアウト
                logger.warning(
                    f"応答タイムアウト（試行{attempt + 1}/{self.config.request_retries}）"
                )

                # ソケットを再作成（ZeroMQ REQソケットの状態リセット）
                if attempt < self.config.request_retries - 1:
                    self._recreate_trade_socket()

            except Exception as e:
                logger.error(f"取引コマンド送信エラー（試行{attempt + 1}）: {e}")

        # すべてのリトライが失敗
        self.messages_failed += 1
        self._log_message(trade_command, status="failed", error="all_retries_exhausted")
        logger.error(
            f"✗ 取引コマンド送信失敗 [#{trade_command['message_id']}]: "
            f"すべてのリトライが失敗しました。"
        )

        return False

    def _recreate_trade_socket(self) -> None:
        """取引ソケットを再作成（状態リセット）"""
        try:
            if self.trade_socket:
                self.trade_socket.close(linger=0)  # 👈 linger=0 を追加

            self.trade_socket = self.context.socket(zmq.REQ)
            self.trade_socket.connect(self.config.trade_endpoint)

            logger.debug("取引ソケットを再作成しました。")

        except Exception as e:
            logger.error(f"取引ソケット再作成失敗: {e}")

    # ========== 履歴データ取得リクエスト [V3.0 新規] ==========

    def request_historical_data(
        self, symbol: str, timeframe_name: str, lookback_bars: int
    ) -> Optional[pd.DataFrame]:
        """
        MQL5 EAから履歴データをリクエスト

        Args:
            symbol: 通貨ペア (例: "XAUUSD")
            timeframe_name: 時間足名 (例: "M1", "D1")
            lookback_bars: 取得本数

        Returns:
            Pandas DataFrame (time, open, high, low, close, volume)
        """
        if not self.is_connected or self.trade_socket is None:
            logger.error("ZeroMQソケットが接続されていません。")
            return None

        request = {
            "message_type": MessageType.REQUEST_HISTORY.value,
            "symbol": symbol,
            "timeframe_name": timeframe_name,
            "lookback_bars": lookback_bars,
            "timestamp": datetime.now().isoformat(),
        }

        try:
            # リクエスト送信
            json_request = json.dumps(request, ensure_ascii=False)
            self.trade_socket.send_string(json_request)

            logger.debug(
                f"履歴データリクエスト送信: {symbol} {timeframe_name} {lookback_bars}本"
            )

            # Pollerでレスポンス待機（大容量用タイムアウト）
            poller = zmq.Poller()
            poller.register(self.trade_socket, zmq.POLLIN)

            socks = dict(poller.poll(self.config.request_timeout_large))

            if self.trade_socket in socks and socks[self.trade_socket] == zmq.POLLIN:
                response_bytes = self.trade_socket.recv()
                response_data = parse_response(response_bytes)

                if response_data.get("message_type") == MessageType.HISTORY_DATA.value:
                    # JSONをPandas DataFrameに変換
                    bars_list = response_data.get("data", [])

                    if not bars_list:
                        logger.warning("履歴データが空です。")
                        return None

                    # Pandas DataFrameを構築
                    df = pd.DataFrame(bars_list)
                    df["time"] = pd.to_datetime(df["time"], unit="s", utc=True)
                    df = df.rename(columns={"time": "timestamp"})

                    logger.info(f"✓ 履歴データを取得: {timeframe_name} {len(df)}本")
                    return df

                elif response_data.get("message_type") == MessageType.NACK.value:
                    logger.error(
                        f"履歴データリクエストが拒否されました: {response_data.get('message')}"
                    )
                    return None

            logger.warning(
                "履歴データリクエストがタイムアウトしました。ソケットを再接続します。"
            )
            self._recreate_trade_socket()  # 👈 修正：ソケットをリセット
            return None

        except Exception as e:
            logger.error(f"履歴データリクエストエラー: {e}")
            return None

    # ========== M1バー取得リクエスト [V3.0 新規] ==========

    def request_latest_m1_bar(self, symbol: str) -> Optional[Dict[str, Any]]:
        """
        MQL5 EAから最新のM1バーをリクエスト

        Args:
            symbol: 通貨ペア (例: "XAUUSD")

        Returns:
            バー情報 (time, open, high, low, close, volume, timestamp)
        """
        if not self.is_connected or self.trade_socket is None:
            logger.error("ZeroMQソケットが接続されていません。")
            return None

        request = {
            "message_type": MessageType.REQUEST_M1_BAR.value,
            "symbol": symbol,
            "timestamp": datetime.now().isoformat(),
        }

        try:
            # リクエスト送信
            json_request = json.dumps(request, ensure_ascii=False)
            self.trade_socket.send_string(json_request)

            # Pollerでレスポンス待機
            poller = zmq.Poller()
            poller.register(self.trade_socket, zmq.POLLIN)

            socks = dict(poller.poll(self.config.request_timeout))

            if self.trade_socket in socks and socks[self.trade_socket] == zmq.POLLIN:
                response_bytes = self.trade_socket.recv()
                response_data = parse_response(response_bytes)

                if response_data.get("message_type") == MessageType.M1_BAR_DATA.value:
                    bar_data = response_data.get("data", {})

                    # timestamp を datetime オブジェクトに変換
                    bar_data["timestamp"] = datetime.fromtimestamp(
                        bar_data["time"], timezone.utc
                    )

                    logger.debug(f"✓ M1バーを取得: {bar_data['timestamp']}")
                    return bar_data

                elif response_data.get("message_type") == MessageType.NACK.value:
                    logger.debug(
                        f"M1バーリクエストが拒否されました: {response_data.get('message')}"
                    )
                    return None

            logger.debug(
                "M1バーリクエストがタイムアウト（新しいバーなし）。ソケットを再接続します。"
            )
            self._recreate_trade_socket()  # 👈 修正：ソケットをリセット
            return None

        except Exception as e:
            logger.error(f"M1バーリクエストエラー: {e}")
            return None

    # ========== 双方向ハートビート ==========

    def _start_heartbeat_thread(self) -> None:
        """ハートビートスレッドを開始"""
        self.heartbeat_running = True
        self.heartbeat_thread = threading.Thread(
            target=self._heartbeat_loop, daemon=True
        )
        self.heartbeat_thread.start()
        logger.info("ハートビートスレッドを開始しました。")

    def _stop_heartbeat_thread(self) -> None:
        """ハートビートスレッドを停止"""
        self.heartbeat_running = False
        if self.heartbeat_thread:
            self.heartbeat_thread.join(timeout=5.0)
        logger.info("ハートビートスレッドを停止しました。")

    def _heartbeat_loop(self) -> None:
        """ハートビートループ（別スレッド） - REQ/REPパターン"""

        while self.heartbeat_running:
            try:
                # --- PING送信 ---
                ping_message = {
                    "message_type": MessageType.PING.value,
                    "timestamp": datetime.now().isoformat(),
                }

                if self.heartbeat_socket is None:
                    logger.warning("ハートビートソケットがありません。")
                    time.sleep(self.config.heartbeat_interval)
                    continue

                self.heartbeat_socket.send_string(json.dumps(ping_message))
                self.last_heartbeat_sent = datetime.now()
                self.heartbeats_sent += 1
                logger.debug(f"PING送信 (累計: {self.heartbeats_sent})")

                # --- PONG受信 (タイムアウト付き) ---
                poller = zmq.Poller()
                poller.register(self.heartbeat_socket, zmq.POLLIN)
                # 10秒間隔(heartbeat_interval)より短い9秒(9000ms)をタイムアウトに設定
                socks = dict(poller.poll(9000))

                if (
                    self.heartbeat_socket in socks
                    and socks[self.heartbeat_socket] == zmq.POLLIN
                ):
                    # REQソケットは単一のペイロードを受信
                    response_bytes = self.heartbeat_socket.recv()
                    if response_bytes:
                        # (V3.1フレームクリーニングを適用)
                        message = parse_response(response_bytes)
                        if message.get("message_type") == MessageType.PONG.value:
                            self.last_heartbeat_received = datetime.now()
                            self.heartbeats_received += 1
                            logger.debug(f"PONG受信 (累計: {self.heartbeats_received})")

                else:  # タイムアウト
                    logger.warning(
                        f"ハートビートタイムアウト（PONGが受信できません）。ソケットを再接続します。"
                    )
                    # REQソケットはタイムアウト時に再作成が必要
                    self._recreate_heartbeat_socket()  # ヘルパー関数を呼ぶ
                    # continue を削除し、ループの最後にある time.sleep() に移行させる

                # インターバル
                time.sleep(self.config.heartbeat_interval)

            except Exception as e:
                logger.error(f"ハートビートループエラー: {e}")
                time.sleep(1.0)

    def _send_ping(self) -> None:
        """PINGメッセージを送信"""
        try:
            ping_message = {
                "message_type": MessageType.PING.value,
                "timestamp": datetime.now().isoformat(),
            }

            # DEALERソケットで送信
            self.heartbeat_socket.send_string(json.dumps(ping_message))

            self.last_heartbeat_sent = datetime.now()
            self.heartbeats_sent += 1

            logger.debug(f"PING送信 (累計: {self.heartbeats_sent})")

        except Exception as e:
            logger.error(f"PING送信エラー: {e}")

    def _attempt_reconnection(self) -> None:
        """再接続を試みる"""
        logger.info("再接続を試みています...")

        try:
            self.disconnect()
            time.sleep(2.0)

            if self.connect():
                logger.info("✓ 再接続成功")

                if self.on_connection_restored:
                    self.on_connection_restored()
            else:
                logger.error("✗ 再接続失敗")

        except Exception as e:
            logger.error(f"再接続エラー: {e}")

    # ========== ブローカー状態リクエスト ==========

    def request_broker_state(self) -> Optional[Dict[str, Any]]:
        """
        MQL5 EAにブローカー状態をリクエスト

        Returns:
            ブローカー状態辞書、失敗時None
        """
        if not self.is_connected or self.trade_socket is None:
            logger.error("ZeroMQソケットが接続されていません。")
            return None

        request = {
            "message_type": MessageType.REQUEST_BROKER_STATE.value,
            "timestamp": datetime.now().isoformat(),
        }

        try:
            # リクエスト送信
            self.trade_socket.send_string(json.dumps(request))

            # Pollerで応答待機
            poller = zmq.Poller()
            poller.register(self.trade_socket, zmq.POLLIN)

            socks = dict(poller.poll(self.config.request_timeout))

            if self.trade_socket in socks and socks[self.trade_socket] == zmq.POLLIN:
                response_bytes = self.trade_socket.recv()
                response_data = parse_response(response_bytes)

                if (
                    response_data.get("message_type")
                    == MessageType.BROKER_STATE_RESPONSE.value
                ):
                    logger.info("✓ ブローカー状態を取得しました。")
                    return response_data.get("data")

            logger.warning(
                "ブローカー状態リクエストがタイムアウトしました。ソケットを再接続します。"
            )
            self._recreate_trade_socket()  # 👈 修正：ソケットをリセット
            return None

        except Exception as e:
            logger.error(f"ブローカー状態リクエストエラー: {e}")
            return None

    def _recreate_heartbeat_socket(self) -> None:
        """ハートビートソケット(REQ)を再作成（状態リセット）"""
        try:
            if self.heartbeat_socket:
                self.heartbeat_socket.close()

            if self.context is None:  # context がない場合のガード
                logger.error(
                    "ZMQコンテキストがありません。ハートビートソケットを再作成できません。"
                )
                return

            self.heartbeat_socket = self.context.socket(zmq.REQ)  # REQ
            self.heartbeat_socket.connect(self.config.heartbeat_endpoint)
            logger.debug("ハートビートソケット(REQ)を再作成しました。")
        except Exception as e:
            logger.error(f"ハートビートソケット再作成失敗: {e}")

    # ========== ログ記録 ==========

    def _log_message(
        self, trade_command: Dict[str, Any], status: str, error: Optional[str] = None
    ) -> None:
        """
        送信メッセージをログファイルに記録

        Args:
            trade_command: 取引コマンド
            status: ステータス（'sent_acked', 'nack', 'failed'）
            error: エラーメッセージ（オプション）
        """
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "status": status,
            "message_id": trade_command.get("message_id"),
            "command": trade_command,
            "error": error,
        }

        # 日付ごとのログファイル
        log_file = (
            self.log_dir / f"zmq_publisher_v3_{datetime.now().strftime('%Y%m%d')}.jsonl"
        )

        try:
            with open(log_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(log_entry, ensure_ascii=False) + "\n")
        except Exception as e:
            logger.warning(f"ログファイル書き込み失敗: {e}")

    # ========== 統計情報 ==========

    def get_statistics(self) -> Dict[str, Any]:
        """
        通信統計情報を取得

        Returns:
            統計情報辞書
        """
        success_rate = (
            self.messages_acked / self.messages_sent if self.messages_sent > 0 else 0.0
        )

        return {
            "messages_sent": self.messages_sent,
            "messages_acked": self.messages_acked,
            "messages_failed": self.messages_failed,
            "success_rate": success_rate,
            "heartbeats_sent": self.heartbeats_sent,
            "heartbeats_received": self.heartbeats_received,
            "last_heartbeat_sent": (
                self.last_heartbeat_sent.isoformat()
                if self.last_heartbeat_sent
                else None
            ),
            "last_heartbeat_received": (
                self.last_heartbeat_received.isoformat()
                if self.last_heartbeat_received
                else None
            ),
            "is_connected": self.is_connected,
        }

    # ========== コンテキストマネージャー ==========

    def __enter__(self):
        """コンテキストマネージャーのエントリー"""
        self.connect()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """コンテキストマネージャーの終了"""
        self.disconnect()


# ========== フォールバックメカニズム ==========


class FileBasedBridge:
    """
    ZeroMQ通信が失敗した場合のフォールバック
    JSONファイルを介した通信
    """

    def __init__(self, file_path: str = str(config.BRIDGE_FALLBACK_FILE)):
        self.file_path = Path(file_path)
        self.file_path.parent.mkdir(parents=True, exist_ok=True)
        logger.info(f"ファイルベースBridgeを初期化: {self.file_path}")

    def write_command(self, trade_command: Dict[str, Any]) -> bool:
        """
        取引コマンドをJSONファイルに書き込み

        Args:
            trade_command: 取引コマンド辞書

        Returns:
            書き込み成功の場合True
        """
        try:
            if "timestamp" not in trade_command:
                trade_command["timestamp"] = datetime.now().isoformat()

            with open(self.file_path, "w", encoding="utf-8") as f:
                json.dump(trade_command, f, indent=2, ensure_ascii=False)

            logger.info(f"✓ ファイルベース通信: コマンドを書き込みました")
            return True

        except Exception as e:
            logger.error(f"✗ ファイルベース通信失敗: {e}")
            return False
