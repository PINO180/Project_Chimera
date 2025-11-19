#!/usr/bin/env python
"""
ZMQ接続デバッグスクリプト V2 - 詳細版
ProjectForgeReceiver.mq5 との通信を確認する
"""

import sys
import time
import json
from pathlib import Path
import zmq
import logging

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger("ZMQ_Debug_V2")

# プロジェクトルートをパスに追加
project_root = Path(__file__).resolve().parents[0]
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

import blueprint as config


def test_zmq_connection():
    """ZMQ接続テスト V2"""

    trade_endpoint = config.ZMQ["trade_endpoint"]
    heartbeat_endpoint = config.ZMQ["heartbeat_endpoint"]

    logger.info("=" * 70)
    logger.info("ZMQ 接続テスト V2 開始")
    logger.info("=" * 70)
    logger.info(f"取引エンドポイント: {trade_endpoint}")
    logger.info(f"ハートビートエンドポイント: {heartbeat_endpoint}")

    context = zmq.Context()

    try:
        # 取引ソケット (REQ)
        logger.info("\n" + "=" * 70)
        logger.info("[1] 取引ソケット (REQ) をセットアップ...")
        logger.info("=" * 70)

        trade_socket = context.socket(zmq.REQ)

        # ✨ [V2 修正] ソケットオプションの最適化
        trade_socket.setsockopt(zmq.LINGER, 0)
        trade_socket.setsockopt(zmq.RCVTIMEO, 10000)  # 10秒に延長
        trade_socket.setsockopt(zmq.SNDHWM, 1000)  # 送信ハイウォーターマーク
        trade_socket.setsockopt(zmq.RCVHWM, 1000)  # 受信ハイウォーターマーク

        logger.info(f"   接続中: {trade_endpoint}")
        trade_socket.connect(trade_endpoint)
        logger.info("   ✓ ソケット作成・接続完了")

        # ブローカー状態リクエスト送信
        logger.info("\n[2] ブローカー状態リクエストを送信...")
        request = {
            "message_type": "REQUEST_BROKER_STATE",
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
        request_json = json.dumps(request)
        logger.info(f"   送信内容 (文字数: {len(request_json)}): {request_json}")

        trade_socket.send_string(request_json)
        logger.info("   ✓ リクエスト送信完了（wait フェーズ）")

        # レスポンス受信
        logger.info("\n[3] レスポンスを受信中 (最大10秒)...")
        try:
            # ✨ [V2 修正] バイナリ受信で詳細を確認
            response_bytes = trade_socket.recv()
            logger.info(f"   ✓ レスポンス受信（バイト数: {len(response_bytes)}）")
            logger.info(f"   [DEBUG] Raw bytes: {response_bytes[:100]}")

            response_json = response_bytes.decode("utf-8")
            logger.info(f"   ✓ UTF-8 デコード成功")
            logger.info(f"   [DEBUG] デコード後の文字列長: {len(response_json)}")
            logger.info(f"   [DEBUG] 文字列内容: {response_json[:200]}")

            if len(response_json) == 0:
                logger.error("   ✗ 受け取った文字列が空です！")
            else:
                # ✨ [修正] JSON オブジェクトの部分だけを抽出
                # ZMQ フレームの余分なバイトを削除
                json_start = response_json.find("{")
                json_end = response_json.rfind("}") + 1

                if json_start != -1 and json_end > json_start:
                    clean_json = response_json[json_start:json_end]
                    logger.info(f"   [DEBUG] クリーニング後: {clean_json[:200]}")
                    response = json.loads(clean_json)
                else:
                    logger.error(f"   ✗ JSON構造が不正です: {response_json}")
                    raise ValueError("JSON structure invalid")
                logger.info(f"   ✓ JSON パース成功")
                logger.info(f"   メッセージタイプ: {response.get('message_type')}")
                logger.info(f"   タイムスタンプ: {response.get('timestamp')}")

                if "data" in response:
                    data = response["data"]
                    logger.info(f"   口座残高 (Equity): {data.get('equity')}")
                    logger.info(f"   ポジション数: {len(data.get('positions', []))}")

        except zmq.error.Again as e:
            logger.error(f"   ✗ タイムアウト (10秒以内にレスポンスがありません): {e}")
            logger.error(
                "   → ProjectForgeReceiver.mq5 が起動していない可能性があります"
            )

        except UnicodeDecodeError as e:
            logger.error(f"   ✗ UTF-8 デコードエラー: {e}")
            logger.error(f"   → 受け取ったバイナリが正しくない")

        except json.JSONDecodeError as e:
            logger.error(f"   ✗ JSON パースエラー: {e}")
            logger.error(f"   → MQL5 が正しい JSON を送信していない")

        trade_socket.close()

        # ハートビートソケット (DEALER)
        logger.info("\n" + "=" * 70)
        logger.info("[4] ハートビートソケット (DEALER) をセットアップ...")
        logger.info("=" * 70)

        hb_socket = context.socket(zmq.DEALER)
        hb_socket.setsockopt(zmq.LINGER, 0)
        hb_socket.setsockopt(zmq.RCVTIMEO, 5000)

        logger.info(f"   接続中: {heartbeat_endpoint}")
        hb_socket.connect(heartbeat_endpoint)
        logger.info("   ✓ ソケット作成・接続完了")

        # PINGメッセージ送信
        logger.info("\n[5] PINGメッセージを送信...")
        ping = {"message_type": "PING", "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")}
        ping_json = json.dumps(ping)
        logger.info(f"   送信内容: {ping_json}")

        hb_socket.send_string(ping_json)
        logger.info("   ✓ PING送信完了")

        # PONG受信
        logger.info("\n[6] PONGを受信中 (最大5秒)...")
        try:
            pong_bytes = hb_socket.recv()
            pong_json = pong_bytes.decode("utf-8")
            logger.info(f"   ✓ PONG受信: {pong_json}")
        except zmq.error.Again:
            logger.error("   ✗ タイムアウト (5秒以内にPONGがありません)")

        hb_socket.close()

        logger.info("\n" + "=" * 70)
        logger.info("✓ ZMQ接続テスト完了")
        logger.info("=" * 70)

    except Exception as e:
        logger.error(f"✗ エラー発生: {e}", exc_info=True)

    finally:
        context.term()


if __name__ == "__main__":
    test_zmq_connection()
