#!/usr/bin/env python
"""
ZMQ RAW ソケット診断
MQL5 からのメッセージが実際に届いているか確認する
"""

import sys
import time
import json
from pathlib import Path
import zmq
import logging

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger("ZMQ_RAW")

project_root = Path(__file__).resolve().parents[0]
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

import blueprint as config


def test_raw_sockets():
    """RAW ソケットテスト"""

    context = zmq.Context()

    logger.info("=" * 70)
    logger.info("ZMQ RAW ソケット診断開始")
    logger.info("=" * 70)

    try:
        # === テスト1: PUSH-PULL ペアで確認 ===
        logger.info("\n[テスト1] PUSH-PULL ペアで確認")
        logger.info("-" * 70)

        # Python がサーバーになる場合
        push_socket = context.socket(zmq.PUSH)
        push_socket.bind("tcp://127.0.0.1:15555")

        pull_socket = context.socket(zmq.PULL)
        pull_socket.setsockopt(zmq.RCVTIMEO, 3000)
        pull_socket.connect("tcp://127.0.0.1:15555")

        time.sleep(0.5)

        logger.info("   テストメッセージを送信...")
        push_socket.send_string("HELLO_FROM_PYTHON")

        logger.info("   受信中...")
        try:
            msg = pull_socket.recv_string()
            logger.info(f"   ✓ 受信成功: {msg}")
        except zmq.error.Again:
            logger.error("   ✗ 受信タイムアウト")

        push_socket.close()
        pull_socket.close()

        # === テスト2: MQL5 との通信（診断モード）===
        logger.info("\n[テスト2] MQL5 との通信（診断モード）")
        logger.info("-" * 70)

        trade_endpoint = config.ZMQ["trade_endpoint"]
        logger.info(f"   エンドポイント: {trade_endpoint}")

        # ✨ PAIR ソケット（双方向通信テスト）
        pair_socket = context.socket(zmq.PAIR)
        pair_socket.setsockopt(zmq.LINGER, 0)
        pair_socket.setsockopt(zmq.RCVTIMEO, 5000)

        logger.info(f"   PAIR ソケットを接続...")
        pair_socket.connect(trade_endpoint)

        time.sleep(1)

        # MQL5 にリクエスト送信（文字列送信）
        request = json.dumps(
            {
                "message_type": "REQUEST_BROKER_STATE",
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            }
        )

        logger.info(f"   リクエスト送信: {request}")
        pair_socket.send_string(request)

        logger.info("   レスポンス受信中...")
        try:
            response_bytes = pair_socket.recv()
            logger.info(f"   ✓ 受信成功（バイト数: {len(response_bytes)}）")
            logger.info(f"   [DEBUG] Raw bytes: {response_bytes[:200]}")

            try:
                response_str = response_bytes.decode("utf-8")
                logger.info(f"   ✓ UTF-8 デコード成功")
                logger.info(f"   [DEBUG] 文字列長: {len(response_str)}")
                logger.info(f"   [DEBUG] 内容: {response_str}")

                if len(response_str) > 0:
                    response_json = json.loads(response_str)
                    logger.info(
                        f"   ✓ JSON パース成功: {response_json.get('message_type')}"
                    )
                else:
                    logger.error("   ✗ 文字列が空です")
            except Exception as e:
                logger.error(f"   ✗ デコードエラー: {e}")

        except zmq.error.Again as e:
            logger.error(f"   ✗ タイムアウト: {e}")

        pair_socket.close()

        # === テスト3: ハートビート（DEALER-ROUTER） ===
        logger.info("\n[テスト3] ハートビート（DEALER-ROUTER）")
        logger.info("-" * 70)

        hb_endpoint = config.ZMQ["heartbeat_endpoint"]
        logger.info(f"   エンドポイント: {hb_endpoint}")

        dealer_socket = context.socket(zmq.DEALER)
        dealer_socket.setsockopt(zmq.LINGER, 0)
        dealer_socket.setsockopt(zmq.RCVTIMEO, 3000)

        logger.info(f"   DEALER ソケットを接続...")
        dealer_socket.connect(hb_endpoint)

        time.sleep(1)

        ping = json.dumps(
            {"message_type": "PING", "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")}
        )

        logger.info(f"   PING送信: {ping}")
        dealer_socket.send_string(ping)

        logger.info("   PONG受信中...")
        try:
            pong_bytes = dealer_socket.recv()
            logger.info(f"   ✓ 受信成功（バイト数: {len(pong_bytes)}）")
            pong_str = pong_bytes.decode("utf-8")
            logger.info(f"   ✓ PONG: {pong_str}")
        except zmq.error.Again:
            logger.error("   ✗ タイムアウト")

        dealer_socket.close()

        logger.info("\n" + "=" * 70)
        logger.info("✓ 診断完了")
        logger.info("=" * 70)

    except Exception as e:
        logger.error(f"✗ エラー: {e}", exc_info=True)

    finally:
        context.term()


if __name__ == "__main__":
    test_raw_sockets()
