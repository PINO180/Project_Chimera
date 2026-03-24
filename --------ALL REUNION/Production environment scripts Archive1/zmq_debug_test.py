#!/usr/bin/env python
"""
ZMQ接続デバッグスクリプト
ProjectForgeReceiver.mq5 との通信を確認する
"""

import sys
import time
import json
from pathlib import Path
import zmq
import logging

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger("ZMQ_Debug")

# プロジェクトルートをパスに追加
project_root = Path(__file__).resolve().parents[0]
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

import blueprint as config


def test_zmq_connection():
    """ZMQ接続テスト"""

    trade_endpoint = config.ZMQ["trade_endpoint"]
    heartbeat_endpoint = config.ZMQ["heartbeat_endpoint"]

    logger.info("=" * 60)
    logger.info("ZMQ 接続テスト開始")
    logger.info("=" * 60)
    logger.info(f"取引エンドポイント: {trade_endpoint}")
    logger.info(f"ハートビートエンドポイント: {heartbeat_endpoint}")

    context = zmq.Context()

    try:
        # 取引ソケット (REQ)
        logger.info("\n[1] 取引ソケット (REQ) をセットアップ...")
        trade_socket = context.socket(zmq.REQ)
        trade_socket.setsockopt(zmq.LINGER, 0)
        trade_socket.setsockopt(zmq.RCVTIMEO, 5000)  # 5秒タイムアウト

        logger.info(f"   接続中: {trade_endpoint}")
        trade_socket.connect(trade_endpoint)
        logger.info("   ✓ ソケット作成完了")

        # ブローカー状態リクエスト送信
        logger.info("\n[2] ブローカー状態リクエストを送信...")
        request = {
            "message_type": "REQUEST_BROKER_STATE",
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
        request_json = json.dumps(request)
        logger.info(f"   送信内容: {request_json}")

        trade_socket.send_string(request_json)
        logger.info("   ✓ リクエスト送信完了")

        # レスポンス受信
        logger.info("\n[3] レスポンスを受信中 (最大5秒)...")
        try:
            response_json = trade_socket.recv_string()
            logger.info(f"   ✓ レスポンス受信: {response_json[:200]}")

            response = json.loads(response_json)
            logger.info(f"   メッセージタイプ: {response.get('message_type')}")

        except zmq.error.Again:
            logger.error("   ✗ タイムアウト (5秒以内にレスポンスがありません)")
            logger.error(
                "   → ProjectForgeReceiver.mq5 が起動していない可能性があります"
            )

        trade_socket.close()

        # ハートビートソケット (DEALER)
        logger.info("\n[4] ハートビートソケット (DEALER) をセットアップ...")
        hb_socket = context.socket(zmq.DEALER)
        hb_socket.setsockopt(zmq.LINGER, 0)
        hb_socket.setsockopt(zmq.RCVTIMEO, 3000)  # 3秒タイムアウト

        logger.info(f"   接続中: {heartbeat_endpoint}")
        hb_socket.connect(heartbeat_endpoint)
        logger.info("   ✓ ソケット作成完了")

        # PINGメッセージ送信
        logger.info("\n[5] PINGメッセージを送信...")
        ping = {"message_type": "PING", "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")}
        ping_json = json.dumps(ping)
        logger.info(f"   送信内容: {ping_json}")

        hb_socket.send_string(ping_json)
        logger.info("   ✓ PING送信完了")

        # PONG受信
        logger.info("\n[6] PONGを受信中 (最大3秒)...")
        try:
            pong_json = hb_socket.recv_string()
            logger.info(f"   ✓ PONG受信: {pong_json}")
        except zmq.error.Again:
            logger.error("   ✗ タイムアウト (3秒以内にPONGがありません)")

        hb_socket.close()

        logger.info("\n" + "=" * 60)
        logger.info("✓ ZMQ接続テスト完了")
        logger.info("=" * 60)

    except Exception as e:
        logger.error(f"✗ エラー発生: {e}", exc_info=True)

    finally:
        context.term()


if __name__ == "__main__":
    test_zmq_connection()
