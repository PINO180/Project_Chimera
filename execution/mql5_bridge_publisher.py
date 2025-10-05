"""
MQL5ブリッジ 2.0 - 高信頼性通信
レイジー・パイレートパターンと双方向ハートビートによるミッションクリティカル通信
"""
import config
import zmq
import json
import time
import threading
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional, Callable
from dataclasses import dataclass
from enum import Enum
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
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


@dataclass
class BridgeConfig:
    """通信設定"""
    trade_endpoint: str = "tcp://127.0.0.1:5555"
    heartbeat_endpoint: str = "tcp://127.0.0.1:5556"
    request_timeout: int = 2500  # ミリ秒
    request_retries: int = 3
    heartbeat_interval: int = 10  # 秒
    heartbeat_timeout: int = 30  # 秒
    log_dir: str = str(config.LOGS_ZMQ_BRIDGE)


class MQL5BridgePublisherV2:
    """
    ZeroMQを介したPython-MQL5高信頼性通信
    レイジー・パイレートパターンによる送達確認と双方向ハートビート
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
        
        # コールバック（型付け改善）
        self.on_connection_lost: Optional[Callable[[], None]] = None
        self.on_connection_restored: Optional[Callable[[], None]] = None
        
        logger.info("MQL5BridgePublisherV2を初期化しました。")
    
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
            logger.info(f"  ハートビートエンドポイント: {self.config.heartbeat_endpoint}")
            
            # コンテキスト作成
            self.context = zmq.Context()
            
            # 取引コマンド用REQソケット
            self.trade_socket = self.context.socket(zmq.REQ)
            self.trade_socket.connect(self.config.trade_endpoint)
            
            # ハートビート用DEALERソケット
            self.heartbeat_socket = self.context.socket(zmq.DEALER)
            self.heartbeat_socket.setsockopt(zmq.IDENTITY, b"PythonCore")
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
        if 'timestamp' not in trade_command:
            trade_command['timestamp'] = datetime.now().isoformat()
        
        trade_command['message_id'] = self.messages_sent + 1
        trade_command['message_type'] = MessageType.TRADE_COMMAND.value
        
        # レイジー・パイレート: リトライループ
        for attempt in range(self.config.request_retries):
            try:
                # JSON形式でメッセージ送信
                json_message = json.dumps(trade_command, ensure_ascii=False)
                self.trade_socket.send_string(json_message)
                
                logger.debug(f"取引コマンド送信 [#{trade_command['message_id']}, 試行{attempt + 1}]: "
                           f"{trade_command['action']}")
                
                # Pollerで応答待機（タイムアウト付き）
                poller = zmq.Poller()
                poller.register(self.trade_socket, zmq.POLLIN)
                
                socks = dict(poller.poll(self.config.request_timeout))
                
                # 応答受信
                if self.trade_socket in socks and socks[self.trade_socket] == zmq.POLLIN:
                    response = self.trade_socket.recv_string()
                    response_data = json.loads(response)
                    
                    # ACK確認
                    if response_data.get('message_type') == MessageType.ACK.value:
                        self.messages_sent += 1
                        self.messages_acked += 1
                        
                        self._log_message(trade_command, status='sent_acked')
                        
                        logger.info(f"✓ 取引コマンド送信成功 [#{trade_command['message_id']}]: "
                                  f"{trade_command['action']} {trade_command.get('lots', 0):.2f}ロット")
                        
                        return True
                    
                    # NACK受信
                    elif response_data.get('message_type') == MessageType.NACK.value:
                        logger.warning(f"MQL5 EAから拒否されました: {response_data.get('reason')}")
                        self.messages_failed += 1
                        self._log_message(trade_command, status='nack', 
                                        error=response_data.get('reason'))
                        return False
                
                # タイムアウト
                logger.warning(f"応答タイムアウト（試行{attempt + 1}/{self.config.request_retries}）")
                
                # ソケットを再作成（ZeroMQ REQソケットの状態をリセット）
                if attempt < self.config.request_retries - 1:
                    self._recreate_trade_socket()
                
            except Exception as e:
                logger.error(f"取引コマンド送信エラー（試行{attempt + 1}）: {e}")
        
        # すべてのリトライが失敗
        self.messages_failed += 1
        self._log_message(trade_command, status='failed', error='all_retries_exhausted')
        logger.error(f"✗ 取引コマンド送信失敗 [#{trade_command['message_id']}]: "
                   f"すべてのリトライが失敗しました。")
        
        return False
    
    def _recreate_trade_socket(self) -> None:
        """取引ソケットを再作成（状態リセット）"""
        try:
            if self.trade_socket:
                self.trade_socket.close()
            
            self.trade_socket = self.context.socket(zmq.REQ)
            self.trade_socket.connect(self.config.trade_endpoint)
            
            logger.debug("取引ソケットを再作成しました。")
        
        except Exception as e:
            logger.error(f"取引ソケット再作成失敗: {e}")
    
    # ========== 双方向ハートビート ==========
    
    def _start_heartbeat_thread(self) -> None:
        """ハートビートスレッドを開始"""
        self.heartbeat_running = True
        self.heartbeat_thread = threading.Thread(target=self._heartbeat_loop, daemon=True)
        self.heartbeat_thread.start()
        logger.info("ハートビートスレッドを開始しました。")
    
    def _stop_heartbeat_thread(self) -> None:
        """ハートビートスレッドを停止"""
        self.heartbeat_running = False
        if self.heartbeat_thread:
            self.heartbeat_thread.join(timeout=5.0)
        logger.info("ハートビートスレッドを停止しました。")
    
    def _heartbeat_loop(self) -> None:
        """ハートビートループ（別スレッド）"""
        poller = zmq.Poller()
        poller.register(self.heartbeat_socket, zmq.POLLIN)
        
        while self.heartbeat_running:
            try:
                # PINGを送信
                self._send_ping()
                
                # PONGを受信（ノンブロッキング）
                socks = dict(poller.poll(1000))  # 1秒タイムアウト
                
                if self.heartbeat_socket in socks and socks[self.heartbeat_socket] == zmq.POLLIN:
                    # マルチパートメッセージ受信（DEALER/ROUTERパターン）
                    frames = self.heartbeat_socket.recv_multipart()
                    
                    if len(frames) >= 1:
                        message = json.loads(frames[-1].decode('utf-8'))
                        
                        if message.get('message_type') == MessageType.PONG.value:
                            self.last_heartbeat_received = datetime.now()
                            self.heartbeats_received += 1
                            logger.debug(f"PONGを受信 (累計: {self.heartbeats_received})")
                
                # 接続喪失チェック
                if self.last_heartbeat_received:
                    time_since_last = (datetime.now() - self.last_heartbeat_received).total_seconds()
                    
                    if time_since_last > self.config.heartbeat_timeout:
                        logger.error(f"ハートビートタイムアウト（{time_since_last:.1f}秒）")
                        
                        if self.on_connection_lost:
                            self.on_connection_lost()
                        
                        # 再接続を試みる
                        self._attempt_reconnection()
                
                # インターバル
                time.sleep(self.config.heartbeat_interval)
            
            except Exception as e:
                logger.error(f"ハートビートループエラー: {e}")
                time.sleep(1.0)
    
    def _send_ping(self) -> None:
        """PINGメッセージを送信"""
        try:
            ping_message = {
                'message_type': MessageType.PING.value,
                'timestamp': datetime.now().isoformat()
            }
            
            # DEALERソケットで送信（エンベロープなし）
            self.heartbeat_socket.send_string(json.dumps(ping_message))
            
            self.last_heartbeat_sent = datetime.now()
            self.heartbeats_sent += 1
            
            logger.debug(f"PINGを送信 (累計: {self.heartbeats_sent})")
        
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
            'message_type': MessageType.REQUEST_BROKER_STATE.value,
            'timestamp': datetime.now().isoformat()
        }
        
        try:
            # リクエスト送信
            self.trade_socket.send_string(json.dumps(request))
            
            # Pollerで応答待機
            poller = zmq.Poller()
            poller.register(self.trade_socket, zmq.POLLIN)
            
            socks = dict(poller.poll(self.config.request_timeout))
            
            if self.trade_socket in socks and socks[self.trade_socket] == zmq.POLLIN:
                response = self.trade_socket.recv_string()
                response_data = json.loads(response)
                
                if response_data.get('message_type') == MessageType.BROKER_STATE_RESPONSE.value:
                    logger.info("✓ ブローカー状態を取得しました。")
                    return response_data.get('data')
            
            logger.warning("ブローカー状態リクエストがタイムアウトしました。")
            return None
        
        except Exception as e:
            logger.error(f"ブローカー状態リクエストエラー: {e}")
            return None
    
    # ========== ログ記録 ==========
    
    def _log_message(self,
                    trade_command: Dict[str, Any],
                    status: str,
                    error: Optional[str] = None) -> None:
        """
        送信メッセージをログファイルに記録
        
        Args:
            trade_command: 取引コマンド
            status: ステータス（'sent_acked', 'nack', 'failed'）
            error: エラーメッセージ（オプション）
        """
        log_entry = {
            'timestamp': datetime.now().isoformat(),
            'status': status,
            'message_id': trade_command.get('message_id'),
            'command': trade_command,
            'error': error
        }
        
        # 日付ごとのログファイル
        log_file = self.log_dir / f"zmq_publisher_v2_{datetime.now().strftime('%Y%m%d')}.jsonl"
        
        try:
            with open(log_file, 'a', encoding='utf-8') as f:
                f.write(json.dumps(log_entry, ensure_ascii=False) + '\n')
        except Exception as e:
            logger.warning(f"ログファイル書き込み失敗: {e}")
    
    # ========== 統計情報 ==========
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        通信統計情報を取得
        
        Returns:
            統計情報辞書
        """
        success_rate = (self.messages_acked / self.messages_sent 
                       if self.messages_sent > 0 else 0.0)
        
        return {
            'messages_sent': self.messages_sent,
            'messages_acked': self.messages_acked,
            'messages_failed': self.messages_failed,
            'success_rate': success_rate,
            'heartbeats_sent': self.heartbeats_sent,
            'heartbeats_received': self.heartbeats_received,
            'last_heartbeat_sent': (self.last_heartbeat_sent.isoformat() 
                                   if self.last_heartbeat_sent else None),
            'last_heartbeat_received': (self.last_heartbeat_received.isoformat() 
                                       if self.last_heartbeat_received else None),
            'is_connected': self.is_connected
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
            if 'timestamp' not in trade_command:
                trade_command['timestamp'] = datetime.now().isoformat()
            
            with open(self.file_path, 'w', encoding='utf-8') as f:
                json.dump(trade_command, f, indent=2, ensure_ascii=False)
            
            logger.info(f"✓ ファイルベース通信: コマンドを書き込みました")
            return True
            
        except Exception as e:
            logger.error(f"✗ ファイルベース通信失敗: {e}")
            return False


# 使用例
if __name__ == '__main__':
    # サンプル取引コマンド
    sample_command = {
        'action': 'BUY',
        'lots': 1.25,
        'entry_price': 150.25,
        'stop_loss': 149.40,
        'take_profit': 151.95,
        'confidence_m2': 0.72,
        'predicted_time': 45,
        'reason': 'M2確信度72%、Kelly2.0%、BUYシグナル'
    }
    
    # コールバック関数の定義
    def on_connection_lost():
        print("\n⚠ 警告: MQL5 EAとの接続が失われました。")
    
    def on_connection_restored():
        print("\n✓ 情報: MQL5 EAとの接続が復旧しました。")
    
    # コンテキストマネージャーを使用した安全な通信
    print("=" * 60)
    print("高信頼性通信のテスト")
    print("=" * 60)
    
    config = BridgeConfig(
        trade_endpoint="tcp://127.0.0.1:5555",
        heartbeat_endpoint="tcp://127.0.0.1:5556",
        request_timeout=2500,
        request_retries=3
    )
    
    with MQL5BridgePublisherV2(config=config) as publisher:
        # コールバック設定
        publisher.on_connection_lost = on_connection_lost
        publisher.on_connection_restored = on_connection_restored
        
        # 取引コマンドを送信
        success = publisher.send_trade_command(sample_command)
        
        if success:
            print("\n✓ 取引コマンドの送信に成功しました（ACK受信）。")
        else:
            print("\n✗ 取引コマンドの送信に失敗しました。")
            print("フォールバックメカニズムを起動します...")
            
            fallback = FileBasedBridge()
            fallback.write_command(sample_command)
        
        # ブローカー状態をリクエスト
        print("\nブローカー状態をリクエスト中...")
        broker_state = publisher.request_broker_state()
        
        if broker_state:
            print("✓ ブローカー状態:")
            print(json.dumps(broker_state, indent=2, ensure_ascii=False))
        
        # 統計情報を表示
        stats = publisher.get_statistics()
        print(f"\n通信統計:")
        print(f"  - 送信: {stats['messages_sent']}")
        print(f"  - ACK受信: {stats['messages_acked']}")
        print(f"  - 失敗: {stats['messages_failed']}")
        print(f"  - 成功率: {stats['success_rate']:.2%}")
        print(f"  - ハートビート送信: {stats['heartbeats_sent']}")
        print(f"  - ハートビート受信: {stats['heartbeats_received']}")
        
        # ハートビートの動作を観察（10秒間）
        print("\nハートビートの動作を観察中（10秒間）...")
        time.sleep(10)
    
    print("\n✓ すべてのテスト完了")