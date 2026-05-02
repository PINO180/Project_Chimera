# /workspace/execution/mql5_bridge_publisher.py
"""
MQL5ブリッジ 3.0 - 高信頼性通信 (V11.0 アーキテクチャ)
ハイブリッドZMQパターン（制御/データ分離）とゼロ・シリアライズ転送の実装
"""

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

import blueprint as config
import zmq
import json
import time
import threading
import numpy as np  # [V11.0] ゼロ・デシリアライズ用
from pathlib import Path
from datetime import datetime, timezone
from typing import Dict, Any, Optional, Callable, List
from dataclasses import dataclass
from enum import Enum
import logging
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
    M1_BAR_DATA = "M1_BAR_DATA"


@dataclass
class BridgeConfig:
    """通信設定 (V11.0)"""

    # blueprint.py の ZMQ 設定に対応
    control_endpoint: str = config.ZMQ.get("control_endpoint", "tcp://127.0.0.1:5555")
    data_endpoint: str = config.ZMQ.get("data_endpoint", "tcp://127.0.0.1:5556")
    m3_notify_endpoint: str = config.ZMQ.get(
        "m3_notify_endpoint", "tcp://127.0.0.1:5557"
    )
    # [V11.02] デフォルトを5558に変更
    heartbeat_endpoint: str = config.ZMQ.get(
        "heartbeat_endpoint", "tcp://127.0.0.1:5558"
    )

    request_timeout: int = 45000  # 45秒
    heartbeat_timeout: int = config.ZMQ.get(
        "heartbeat_timeout", 3000
    )  # ハートビート用タイムアウト (EA再起動検知を3秒以内にするため9000→3000)
    request_retries: int = 3
    heartbeat_interval: int = 10
    log_dir: str = str(config.LOGS_ZMQ_BRIDGE)


def clean_zmq_json(response_json: str) -> str:
    """ZMQ フレームから余分なバイトを削除し、JSONを抽出"""
    # オブジェクト '{...}' または 配列 '[...]' を正しく抽出する
    obj_start = response_json.find("{")
    arr_start = response_json.find("[")

    # どちらが先に出現するか（存在しない場合は -1 なので除外）
    valid_starts = [i for i in (obj_start, arr_start) if i != -1]

    if not valid_starts:
        if "ACK" in response_json or "NACK" in response_json:
            return response_json
        return response_json

    start_idx = min(valid_starts)
    end_char = "}" if response_json[start_idx] == "{" else "]"
    end_idx = response_json.rfind(end_char) + 1

    if end_idx <= start_idx:
        return response_json

    return response_json[start_idx:end_idx]


def parse_response(response_bytes: bytes) -> Any:
    """ZMQ レスポンスをパース"""
    try:
        response_str = response_bytes.decode("utf-8")
        # JSON形式か単純テキストかを簡易判定 (オブジェクト '{' または 配列 '[' に対応)
        if response_str.strip().startswith("{") or response_str.strip().startswith("["):
            clean_json = clean_zmq_json(response_str)
            return json.loads(clean_json)
        return response_str
    except Exception as e:
        logger.warning(f"レスポンスパース警告: {e}")
        return None


class MQL5BridgePublisherV3:
    """
    MQL5BridgePublisher V3 (V11.0 Architecture)
    - Control Channel (REQ/REP): コマンド、ハンドシェイク
    - Data Channel (PUSH/PULL): バルクデータ転送 (ゼロ・シリアライズ)
    - Heartbeat Channel (REQ/REP): 接続監視
    """

    # MQL5 MqlRates 構造体のメモリレイアウト定義 (60 bytes, pack=1)
    # long time, double open, double high, double low, double close,
    # long tick_volume, int spread, long real_volume
    MQL_RATES_DTYPE = np.dtype(
        [
            ("time", "<i8"),  # datetime (8 bytes)
            ("open", "<f8"),  # double (8 bytes)
            ("high", "<f8"),  # double (8 bytes)
            ("low", "<f8"),  # double (8 bytes)
            ("close", "<f8"),  # double (8 bytes)
            ("tick_volume", "<i8"),  # long (8 bytes)
            ("spread", "<i4"),  # int (4 bytes)
            ("real_volume", "<i8"),  # long (8 bytes)
        ]
    )  # Total 60 bytes, Little Endian

    def __init__(self, config: Optional[BridgeConfig] = None):
        self.config = config or BridgeConfig()
        self.log_dir = Path(self.config.log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        self.context: Optional[zmq.Context] = None

        # V11.0 ソケット構成
        self.control_socket: Optional[zmq.Socket] = None  # REQ (5555)
        self.data_socket: Optional[zmq.Socket] = None  # PULL (5556)
        self.m3_notify_socket: Optional[zmq.Socket] = None  # PULL (5557) M3確定通知
        self.heartbeat_socket: Optional[zmq.Socket] = None  # REQ (5558)

        self.is_connected = False
        self.heartbeat_thread: Optional[threading.Thread] = None
        self.heartbeat_running = False

        # [STALE-GUARD] EA側のg_python_readyがfalseになったことをHeartbeatスレッドが検知し、
        # メインスレッドにnotify_python_ready()の再送を要求するためのスレッドセーフなフラグ
        self.needs_notify = threading.Event()
        # [STALE-GUARD] Python側の準備完了状態。
        # HeartbeatにPING:READY/PING:NOT_READYとして乗せてEAに毎回通知する。
        # EA再起動・瞬断後も次のHeartbeat(最大10秒)で自動的に状態が同期される。
        self.is_python_ready = False

        # 統計
        self.messages_sent = 0
        self.messages_acked = 0
        self.messages_failed = 0
        self.heartbeats_sent = 0
        self.heartbeats_received = 0
        self.last_heartbeat_sent: Optional[datetime] = None
        self.last_heartbeat_received: Optional[datetime] = None

        logger.info("MQL5BridgePublisherV3 (V11.0) を初期化しました。")

    # ========== 接続管理 ==========

    def connect(self) -> bool:
        try:
            logger.info("ZeroMQ通信 (V11.0) を初期化中...")
            self.context = zmq.Context()

            # 1. 制御チャネル (REQ)
            self.control_socket = self.context.socket(zmq.REQ)
            self.control_socket.connect(self.config.control_endpoint)
            logger.info(f"  Control Endpoint: {self.config.control_endpoint}")

            # 2. データチャネル (PULL) - 受信専用
            self.data_socket = self.context.socket(zmq.PULL)
            self.data_socket.connect(self.config.data_endpoint)
            # HWM (High Water Mark) を設定してバッファ溢れを防ぐ
            self.data_socket.set_hwm(10000)
            logger.info(f"  Data Endpoint: {self.config.data_endpoint}")

            # 3. M3確定通知チャネル (PULL) - M3確定をMT5からPushで受け取る
            self.m3_notify_socket = self.context.socket(zmq.PULL)
            self.m3_notify_socket.connect(self.config.m3_notify_endpoint)
            self.m3_notify_socket.set_hwm(100)
            logger.info(f"  M3 Notify Endpoint: {self.config.m3_notify_endpoint}")

            # ▼▼▼ 修正: ZMQソケットのマルチスレッド違反を解消 ▼▼▼
            # 4. ハートビートチャネル (REQ) のソケット作成は _heartbeat_loop (別スレッド) 内部で行うためここでは作成しない
            # ▲▲▲ ここまで修正 ▲▲▲

            self.is_connected = True
            self._start_heartbeat_thread()

            return True

        except Exception as e:
            logger.error(f"✗ ZeroMQ通信の起動に失敗: {e}")
            self.is_connected = False
            return False

    def disconnect(self) -> None:
        self._stop_heartbeat_thread()

        if self.control_socket:
            self.control_socket.close()
        if self.data_socket:
            self.data_socket.close()
        if self.m3_notify_socket:
            self.m3_notify_socket.close()
        # ▼▼▼ 修正: heartbeat_socket のクローズはスレッド側で行うため削除 ▼▼▼
        # if self.heartbeat_socket:
        #     self.heartbeat_socket.close()
        # ▲▲▲ ここまで修正 ▲▲▲
        if self.context:
            self.context.term()
        self.is_connected = False
        logger.info("ZeroMQ通信を切断しました。")

    # ========== V11.0 履歴データ取得 (ゼロ・シリアライズ) ==========

    def request_historical_data(
        self, symbol: str, timeframe_name: str, lookback_bars: int
    ) -> Optional[pd.DataFrame]:
        """
        V11.0 プロトコルによる履歴データ取得
        1. Handshake (REQ): データ転送をリクエスト
        2. Streaming (PULL): 生バイトデータを受信 (numpy.frombuffer)
        3. Completion: EoSシグナル待機と確認
        """
        if not self.is_connected:
            logger.error("接続されていません。")
            return None

        # 1. ハンドシェイク: 履歴リクエスト送信 (制御チャネル)
        # [FIX-7] lookback_bars をリクエスト文字列に含め MQL5 側に取得本数を通知する
        request_msg = f"REQ_HISTORY:{timeframe_name}:{lookback_bars}"

        try:
            # REQUEST送信
            self.control_socket.send_string(request_msg)

            # ACK受信 (メタデータを含む)
            poller = zmq.Poller()
            poller.register(self.control_socket, zmq.POLLIN)

            if not poller.poll(self.config.request_timeout):
                logger.error("履歴リクエストがタイムアウトしました (ACK未受信)")
                self._recreate_control_socket()
                return None

            ack_response = self.control_socket.recv_string()

            if not ack_response.startswith("ACK:"):
                logger.error(f"MQL5からの応答がACKではありません: {ack_response}")
                return None

            # メタデータのパース
            meta = {}
            parts = ack_response[4:].split(";")
            for part in parts:
                if "=" in part:
                    k, v = part.split("=")
                    meta[k] = v

            total_bars = int(meta.get("TOTAL_BARS", 0))
            total_chunks = int(meta.get("TOTAL_CHUNKS", 0))

            logger.info(
                f"転送開始: {total_bars} bars, {total_chunks} chunks (via PULL)"
            )

            # 2. データストリーミング受信 (データチャネル)
            all_chunks = []
            received_bars = 0
            chunk_count = 0

            # 受信ループ
            while True:
                if self.data_socket.poll(10000) == 0:
                    logger.warning("データ受信タイムアウト (ストリーム中断)")
                    break

                message = self.data_socket.recv()
                msg_len = len(message)

                # デバッグ: バイト数を出力 (最初の数回のみ)
                if chunk_count < 3:
                    logger.info(f"DEBUG: 受信メッセージサイズ: {msg_len} bytes")

                if message == b"END_OF_STREAM":
                    logger.info(
                        f"転送完了シグナル (EoS) を受信。受信済みチャンク: {chunk_count}/{total_chunks}"
                    )
                    break

                try:
                    if msg_len == 0:
                        logger.warning("サイズ0の空メッセージを受信しました。")
                        continue

                    if msg_len % 60 != 0:
                        logger.warning(
                            f"不正なサイズのチャンクを受信: {msg_len} bytes (60の倍数ではありません)"
                        )
                        continue

                    chunk_array = np.frombuffer(message, dtype=self.MQL_RATES_DTYPE)

                    # デバッグ: 生成された配列の長さを確認
                    current_bars = len(chunk_array)
                    # if chunk_count < 3:
                    #    logger.info(f"DEBUG: NumPy配列変換結果: {current_bars} bars")

                    all_chunks.append(chunk_array)

                    received_bars += current_bars
                    chunk_count += 1

                    if chunk_count % 10 == 0 or chunk_count == total_chunks:
                        logger.info(
                            f"受信中... {chunk_count}/{total_chunks} chunks ({received_bars} bars)"
                        )

                except Exception as e:
                    logger.error(f"チャンクデコードエラー: {e}")
                    break

            # 3. データ結合とDataFrame化
            if not all_chunks:
                logger.error("データを受信できませんでした (all_chunks is empty)。")
                # 完了通知を送らずに終了（再送を促すため）
                return None

            full_array = np.concatenate(all_chunks)

            # DataFrame変換
            df = pd.DataFrame(full_array)

            # [修正] エンジンが期待する 'volume' にリネーム (tick_volume を使用)
            if "tick_volume" in df.columns:
                df = df.rename(columns={"tick_volume": "volume"})

            # タイムスタンプ変換 (MQL5 time は Unix Timestamp)
            df["timestamp"] = pd.to_datetime(df["time"], unit="s", utc=True)

            logger.info(f"✓ 全データ受信完了: {len(df)} 行 (期待値: {total_bars})")

            # 4. 完了確認通知 (制御チャネル)
            self.control_socket.send_string("CONFIRM_HISTORY:OK")
            # ACK_CONFIRMED を待つ (タイムアウト付き)
            if self.control_socket.poll(5000):
                self.control_socket.recv_string()
            else:
                logger.warning("完了確認のACKを受信できませんでした")

            return df

        except Exception as e:
            logger.error(f"履歴データ取得エラー: {e}", exc_info=True)
            self._recreate_control_socket()
            return None

    # ========== その他のリクエスト (V11.0 互換実装) ==========

    def send_trade_command(self, command: Dict[str, Any]) -> bool:
        """
        取引コマンドをZMQ経由でMT5に送信する (V11.0 実装版)
        """
        try:
            if self.control_socket is None:
                logger.error("ZMQ Control Socket is not connected.")
                return False

            # 1. メッセージの構築
            # MT5 EA (ProjectForgeReceiver) が解釈できる形式にラップする
            message = {
                "type": "TRADE_COMMAND",  # EA側で識別するためのヘッダー
                "payload": command,
            }
            json_str = json.dumps(message, default=str)

            # 2. 送信 (REQソケット)
            # logger.debug(f"Sending Trade Command: {json_str}")
            self.control_socket.send_string(json_str)

            # 3. ACK待機
            # [FIX-6] タイムアウトを 3000ms → 9000ms に延長
            # XAUUSD 高ボラティリティ時や市場オープン時は MT5 の応答が遅れる場合があるため、
            # タイムアウト後はソケットを再生成してデッドロックを防ぐ。
            if self.control_socket.poll(9000) == 0:
                logger.error(
                    "Timeout waiting for trade ACK from MT5 (9000ms). Recreating socket."
                )
                self._recreate_control_socket()
                return False

            reply = self.control_socket.recv_string()

            # 4. 応答の解析
            try:
                reply_data = json.loads(reply)
                if reply_data.get("status") == "ACK":
                    ack_spread = reply_data.get("spread", None)
                    spread_str = f" [約定時Spread: {ack_spread:.1f}pips]" if ack_spread is not None else ""
                    logger.info(
                        f"✓ Trade Command Accepted by MT5: Ticket={reply_data.get('ticket', 'N/A')}{spread_str}"
                    )
                    return True
                else:
                    logger.error(
                        f"✗ Trade Command Rejected: {reply_data.get('reason', 'Unknown')}"
                    )
                    return False
            except json.JSONDecodeError:
                logger.error(f"Invalid response from MT5: {reply}")
                return False

        except Exception as e:
            logger.error(f"Exception in send_trade_command: {e}", exc_info=True)
            return False

    @staticmethod
    def _validate_bar(bar: Dict[str, Any], prev_close: Optional[float] = None) -> bool:
        """
        [SPIKE-GUARD] 受信バーの価格異常を検出して破棄するバリデーター。

        チェック項目:
        1. OHLC整合性: High >= Low, High >= Open/Close, Low <= Open/Close
        2. 価格範囲: XAU/USDの合理的範囲（500〜15000ドル）外は破棄
        3. 前バー比変化: 前Closeから10%超の変動は桁落ち等のスパイクとして破棄
        """
        try:
            o, h, l, c = bar["open"], bar["high"], bar["low"], bar["close"]

            # 1. OHLC整合性チェック
            if not (h >= l and h >= o and h >= c and l <= o and l <= c):
                logger.warning(
                    f"[SPIKE-GUARD] OHLC不整合を検出して破棄: O={o} H={h} L={l} C={c}"
                )
                return False

            # 2. 価格範囲チェック（XAU/USD合理的範囲）
            if not (500.0 <= c <= 15000.0):
                logger.warning(
                    f"[SPIKE-GUARD] 価格範囲外を検出して破棄: close={c}"
                )
                return False

            # 3. 前バー比変化チェック（10%超は桁落ち等のスパイク）
            if prev_close is not None and prev_close > 0:
                change_ratio = abs(c - prev_close) / prev_close
                if change_ratio > 0.10:
                    logger.warning(
                        f"[SPIKE-GUARD] 前バー比{change_ratio*100:.1f}%変化を検出して破棄: "
                        f"prev_close={prev_close:.3f} close={c:.3f}"
                    )
                    return False

            return True
        except (KeyError, TypeError):
            return False

    def poll_m3_bar(self, timeout_ms: int = 1000) -> Optional[Dict[str, Any]]:
        """
        M3確定通知をPULLソケットで受け取る。
        timeout_ms以内に通知が来なければNoneを返す。
        """
        if not self.is_connected or self.m3_notify_socket is None:
            return None
        try:
            if self.m3_notify_socket.poll(timeout_ms):
                msg = self.m3_notify_socket.recv_string()
                bar = json.loads(msg)
                bar["timestamp"] = datetime.fromtimestamp(bar["time"], tz=timezone.utc)
                # [SPIKE-GUARD] 受信バーのバリデーション
                if not self._validate_bar(bar):
                    return None
                return bar
            return None
        except Exception as e:
            logger.error(f"M3通知受信エラー: {e}")
            return None

    def notify_python_ready(self) -> bool:
        """
        [STALE-GUARD] Python側の準備完了をEAに通知し、M3通知の送信を解禁させる。

        needs_notifyフラグをクリアすることで、次のHeartbeatから
        PING:READYが送信されるようになりEA側のg_python_readyがtrueになる。
        EA再起動・瞬断後も自動的に状態が同期される。
        """
        self.needs_notify.clear()
        logger.info("[STALE-GUARD] needs_notifyをクリア。次のHeartbeatでPING:READYをEAに送信します。")
        return True

    def request_broker_state(self) -> Optional[Dict[str, Any]]:
        """
        ブローカー状態（残高・ポジション）を取得
        V11.02: MQL5側の手動JSON構築に対応
        """
        if not self.is_connected:
            return None

        try:
            # リクエスト送信
            self.control_socket.send_string("REQUEST_BROKER_STATE")

            # レスポンス待機
            poller = zmq.Poller()
            poller.register(self.control_socket, zmq.POLLIN)

            if poller.poll(self.config.request_timeout):
                response_bytes = self.control_socket.recv()

                # JSONパース (parse_responseヘルパーを使用)
                state = parse_response(response_bytes)

                if isinstance(state, dict):
                    logger.debug(
                        f"✓ ブローカー状態取得: Equity={state.get('equity')}, Pos={len(state.get('positions', []))}"
                    )
                    return state
                else:
                    logger.warning(f"ブローカー状態の形式が不正です: {state}")
                    return None
            else:
                logger.warning("ブローカー状態リクエストがタイムアウトしました")
                self._recreate_control_socket()
                return None

        except Exception as e:
            logger.error(f"ブローカー状態リクエスト失敗: {e}")
            self._recreate_control_socket()
            return None

    def request_recent_history(self) -> Optional[List[Dict[str, Any]]]:
        """直近決済されたポジション履歴を取得 (サイレント・クローズ対策)"""
        if not self.is_connected:
            return None

        try:
            self.control_socket.send_string("REQUEST_RECENT_HISTORY")

            poller = zmq.Poller()
            poller.register(self.control_socket, zmq.POLLIN)

            if poller.poll(self.config.request_timeout):
                response_bytes = self.control_socket.recv()
                history = parse_response(response_bytes)

                if isinstance(history, list):
                    return history
                return None
            else:
                logger.warning("履歴リクエストがタイムアウトしました")
                self._recreate_control_socket()
                return None

        except Exception as e:
            logger.error(f"履歴リクエスト失敗: {e}")
            self._recreate_control_socket()
            return None

    # ========== 内部メソッド ==========

    def _recreate_control_socket(self):
        """制御ソケットの再作成"""
        self.control_socket.close(linger=0)
        self.control_socket = self.context.socket(zmq.REQ)
        self.control_socket.connect(self.config.control_endpoint)
        logger.warning("制御ソケットを再作成しました。")

    def _start_heartbeat_thread(self):
        self.heartbeat_running = True
        self.heartbeat_thread = threading.Thread(
            target=self._heartbeat_loop, daemon=True
        )
        self.heartbeat_thread.start()

    def _stop_heartbeat_thread(self):
        self.heartbeat_running = False
        if self.heartbeat_thread:
            self.heartbeat_thread.join(timeout=2.0)

    def _heartbeat_loop(self):
        """ハートビートループ (PING:READY / PING:NOT_READY)

        [STALE-GUARD] 単純なPING/PONGからPython準備状態通知に拡張。
        - needs_notify未セット（通常時）: PING:READY を送信
          → EA側はg_python_ready=trueを維持。EA再起動後もこれを受けて即解禁。
        - needs_notify セット中: PING:NOT_READY を送信
          → EA側はg_python_ready=falseをセット（ウォームアップ中の誤発注を防ぐ）
        - Heartbeatタイムアウト: EA再起動の可能性 → needs_notifyをセット＋即再接続
          → メインスレッドがnotify_python_ready()を再送後にneeds_notifyをクリア
        """
        self.heartbeat_socket = self.context.socket(zmq.REQ)
        self.heartbeat_socket.connect(self.config.heartbeat_endpoint)
        logger.info(f"  Heartbeat Endpoint (Thread): {self.config.heartbeat_endpoint}")

        try:
            while self.heartbeat_running:
                try:
                    if self.heartbeat_socket:
                        # [STALE-GUARD] 準備状態をPINGメッセージに乗せて送信
                        # needs_notifyがセット中 = まだnotify_python_ready()再送が済んでいない
                        ping_msg = "PING:NOT_READY" if self.needs_notify.is_set() else "PING:READY"
                        self.heartbeat_socket.send_string(ping_msg)
                        self.last_heartbeat_sent = datetime.now()
                        self.heartbeats_sent += 1

                        # PONG受信待機
                        if self.heartbeat_socket.poll(self.config.heartbeat_timeout):
                            pong = self.heartbeat_socket.recv_string()
                            if "PONG" in pong:
                                self.last_heartbeat_received = datetime.now()
                                self.heartbeats_received += 1
                        else:
                            # タイムアウト = EA再起動中または切断中
                            logger.warning("Heartbeat timeout")
                            # [STALE-GUARD] EA再起動の可能性 → 復帰後にnotify_python_ready()再送が必要
                            self.needs_notify.set()
                            # ソケット再作成後即PINGを送るためsleepをスキップ
                            self.heartbeat_socket.close(linger=0)
                            self.heartbeat_socket = self.context.socket(zmq.REQ)
                            self.heartbeat_socket.connect(
                                self.config.heartbeat_endpoint
                            )
                            continue  # sleep不要・即次のPINGへ

                    time.sleep(self.config.heartbeat_interval)
                except Exception as e:
                    logger.error(f"Heartbeat loop error: {e}")
                    time.sleep(1)
        finally:
            if self.heartbeat_socket:
                self.heartbeat_socket.close(linger=0)
                self.heartbeat_socket = None

    def _log_message(
        self, trade_command: Dict[str, Any], status: str, error: Optional[str] = None
    ) -> None:
        """ログ記録"""
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "status": status,
            "message_id": trade_command.get("message_id"),
            "command": trade_command,
            "error": error,
        }
        log_file = (
            Path(self.config.log_dir)
            / f"zmq_publisher_v11_{datetime.now().strftime('%Y%m%d')}.jsonl"
        )
        try:
            with open(log_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(log_entry, ensure_ascii=False) + "\n")
        except Exception as e:
            logger.warning(f"ログファイル書き込み失敗: {e}")

    def get_statistics(self) -> Dict[str, Any]:
        """統計情報取得"""
        return {
            "messages_sent": self.messages_sent,
            "messages_acked": self.messages_acked,
            "messages_failed": self.messages_failed,
            "heartbeats_sent": self.heartbeats_sent,
            "heartbeats_received": self.heartbeats_received,
            "last_heartbeat_sent": self.last_heartbeat_sent,
            "last_heartbeat_received": self.last_heartbeat_received,
            "is_connected": self.is_connected,
        }


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
