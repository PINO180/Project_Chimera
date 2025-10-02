import zmq
import json
import time
import logging
from typing import Dict, Any, Optional
from pathlib import Path
from datetime import datetime

# ロギング設定
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class MQL5BridgePublisher:
    """
    ZeroMQを介してPythonからMQL5 EAに取引コマンドを送信するPublisher
    非同期・ノンブロッキング通信を実現
    """
    
    def __init__(self, 
                 endpoint: str = "tcp://127.0.0.1:5555",
                 log_dir: str = "logs/zmq_bridge"):
        """
        Args:
            endpoint: ZeroMQエンドポイント（デフォルト: tcp://127.0.0.1:5555）
            log_dir: 通信ログの保存ディレクトリ
        """
        self.endpoint = endpoint
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # ZeroMQコンテキストとソケットの初期化
        self.context: Optional[zmq.Context] = None
        self.socket: Optional[zmq.Socket] = None
        self.is_connected = False
        
        # 統計情報
        self.messages_sent = 0
        self.messages_failed = 0
        
    def connect(self) -> bool:
        """
        ZeroMQソケットを初期化し、エンドポイントにバインド
        
        Returns:
            接続成功の場合True
        """
        try:
            logger.info(f"ZeroMQ Publisherを初期化中... エンドポイント: {self.endpoint}")
            
            # コンテキスト作成
            self.context = zmq.Context()
            
            # PUBソケット作成
            self.socket = self.context.socket(zmq.PUB)
            
            # 送信タイムアウト設定（100ms）
            self.socket.setsockopt(zmq.SNDTIMEO, 100)
            
            # High Water Mark設定（キューサイズ制限）
            self.socket.setsockopt(zmq.SNDHWM, 10)
            
            # エンドポイントにバインド
            self.socket.bind(self.endpoint)
            
            # バインド直後は接続が確立されていないため、短い待機時間を設ける
            time.sleep(0.5)
            
            self.is_connected = True
            logger.info(f"✓ ZeroMQ Publisherの起動に成功しました: {self.endpoint}")
            
            return True
            
        except Exception as e:
            logger.error(f"✗ ZeroMQ Publisherの起動に失敗しました: {e}")
            self.is_connected = False
            return False
    
    def disconnect(self) -> None:
        """ZeroMQソケットとコンテキストをクリーンアップ"""
        if self.socket:
            self.socket.close()
            logger.info("ZeroMQソケットをクローズしました。")
        
        if self.context:
            self.context.term()
            logger.info("ZeroMQコンテキストを終了しました。")
        
        self.is_connected = False
    
    def send_trade_command(self, trade_command: Dict[str, Any]) -> bool:
        """
        取引コマンドをJSON形式でシリアライズし、ZeroMQ経由で送信
        
        Args:
            trade_command: 取引コマンド辞書
                {
                    'action': 'BUY' | 'SELL' | 'HOLD',
                    'lots': float,
                    'entry_price': float,
                    'stop_loss': float,
                    'take_profit': float,
                    'confidence': float,
                    'reason': str,
                    ...
                }
        
        Returns:
            送信成功の場合True
        """
        if not self.is_connected or self.socket is None:
            logger.error("ZeroMQソケットが接続されていません。connect()を先に呼び出してください。")
            self.messages_failed += 1
            return False
        
        try:
            # タイムスタンプを追加
            if 'timestamp' not in trade_command:
                trade_command['timestamp'] = datetime.now().isoformat()
            
            # メッセージIDを追加（デバッグ用）
            trade_command['message_id'] = self.messages_sent + 1
            
            # JSON形式にシリアライズ
            json_message = json.dumps(trade_command, ensure_ascii=False)
            
            # メッセージ送信（非ブロッキング）
            self.socket.send_string(json_message, flags=zmq.NOBLOCK)
            
            self.messages_sent += 1
            
            # 通信ログを保存
            self._log_message(trade_command, status='sent')
            
            logger.info(f"✓ 取引コマンド送信成功 [#{trade_command['message_id']}]: "
                       f"{trade_command['action']} {trade_command.get('lots', 0):.2f}ロット")
            
            return True
            
        except zmq.Again:
            # タイムアウト（送信バッファが満杯）
            logger.warning("ZeroMQ送信タイムアウト: 送信バッファが満杯です。")
            self.messages_failed += 1
            self._log_message(trade_command, status='timeout')
            return False
            
        except Exception as e:
            logger.error(f"✗ 取引コマンド送信失敗: {e}")
            self.messages_failed += 1
            self._log_message(trade_command, status='failed', error=str(e))
            return False
    
    def _log_message(self, 
                    trade_command: Dict[str, Any], 
                    status: str,
                    error: Optional[str] = None) -> None:
        """
        送信メッセージをログファイルに記録
        
        Args:
            trade_command: 取引コマンド
            status: ステータス（'sent', 'timeout', 'failed'）
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
        log_file = self.log_dir / f"zmq_publisher_{datetime.now().strftime('%Y%m%d')}.jsonl"
        
        try:
            with open(log_file, 'a') as f:
                f.write(json.dumps(log_entry, ensure_ascii=False) + '\n')
        except Exception as e:
            logger.warning(f"ログファイル書き込み失敗: {e}")
    
    def get_statistics(self) -> Dict[str, int]:
        """
        送信統計情報を取得
        
        Returns:
            統計情報辞書
        """
        return {
            'messages_sent': self.messages_sent,
            'messages_failed': self.messages_failed,
            'success_rate': self.messages_sent / (self.messages_sent + self.messages_failed) 
                           if (self.messages_sent + self.messages_failed) > 0 else 0.0
        }
    
    def __enter__(self):
        """コンテキストマネージャーのエントリー"""
        self.connect()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """コンテキストマネージャーの終了"""
        self.disconnect()


# フォールバックメカニズム: ファイルベース通信
class FileBasedBridge:
    """
    ZeroMQ通信が失敗した場合のフォールバック
    JSONファイルを介した通信
    """
    
    def __init__(self, file_path: str = "data/bridge/trade_command.json"):
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
            # タイムスタンプを追加
            if 'timestamp' not in trade_command:
                trade_command['timestamp'] = datetime.now().isoformat()
            
            # ファイルに書き込み
            with open(self.file_path, 'w') as f:
                json.dump(trade_command, f, indent=2, ensure_ascii=False)
            
            logger.info(f"✓ ファイルベース通信: コマンドを書き込みました: {self.file_path}")
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
        'confidence': 0.72,
        'predicted_time': 45,
        'reason': '確信度72%、リスク2.0%、BUYシグナル'
    }
    
    # コンテキストマネージャーを使用した安全な通信
    with MQL5BridgePublisher(endpoint="tcp://127.0.0.1:5555") as publisher:
        # 取引コマンドを送信
        success = publisher.send_trade_command(sample_command)
        
        if success:
            print("\n✓ 取引コマンドの送信に成功しました。")
        else:
            print("\n✗ 取引コマンドの送信に失敗しました。")
            print("フォールバックメカニズムを起動します...")
            
            # フォールバック: ファイルベース通信
            fallback = FileBasedBridge()
            fallback.write_command(sample_command)
        
        # 統計情報を表示
        stats = publisher.get_statistics()
        print(f"\n送信統計:")
        print(f"  - 成功: {stats['messages_sent']}")
        print(f"  - 失敗: {stats['messages_failed']}")
        print(f"  - 成功率: {stats['success_rate']:.2%}")
    
    # 継続的な送信例（リアルタイムシステム用）
    print("\n" + "="*50)
    print("継続的送信モードのデモ（Ctrl+Cで終了）")
    print("="*50)
    
    try:
        publisher = MQL5BridgePublisher()
        publisher.connect()
        
        counter = 0
        while True:
            counter += 1
            
            # 新しいコマンドを生成（シミュレーション）
            new_command = sample_command.copy()
            new_command['entry_price'] = 150.00 + counter * 0.10
            
            # 送信
            publisher.send_trade_command(new_command)
            
            # 待機（実際のシステムでは、AIの予測をトリガーとして実行）
            time.sleep(5)
            
    except KeyboardInterrupt:
        print("\n\n終了シグナルを受信しました。")
        publisher.disconnect()
        print("✓ クリーンアップ完了。")