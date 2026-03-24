# /workspace/main.py
"""
Project Forge 統合実行スクリプト (司令塔) V11.0 (ZMQ ハイブリッド/ゼロ・シリアライズ版)

設計図5章に基づき、全てのコンポーネントを統合し、
リアルタイム取引ループを実行する中央制御システム。

[V11.0 修正内容]
1. MQL5BridgePublisherV3 (V11.0) を採用
2. ZMQエンドポイント設定を Control/Data/Heartbeat の3系統に変更
3. 起動時のデータ取得フローを V11.0 (非同期データポンプ) に対応
[V11.1 修正内容]
- 静的文脈データ (context_features_v2) の廃止 (Top 50戦略への完全移行)
- H12などの上位足データ取得ロジックの最適化
"""

import sys
import time
from pathlib import Path
import logging
import joblib
import numpy as np
from datetime import datetime, timezone
from typing import Optional, Dict, Any, List
import polars as pl
import pandas as pd
import re

# --- プロジェクトのルートディレクトリをPythonの検索パスに追加 ---
project_root = Path(__file__).resolve().parents[0]
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

# --- 必要なコンポーネントをインポート ---
import blueprint as config
from execution.state_manager import StateManager, SystemState, BrokerStateDict

# [V11.0] V3 (ハイブリッドZMQ/ゼロ・シリアライズ) をインポート
from execution.mql5_bridge_publisher import MQL5BridgePublisherV3, BridgeConfig
from execution.extreme_risk_engine import ExtremeRiskEngineV2, MarketInfo
from execution.realtime_feature_engine import RealtimeFeatureEngine

# --- ログ設定 ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(config.LOGS_FORGE_SYSTEM, encoding="utf-8"),
    ],
)
logger = logging.getLogger("ProjectForge.Main")

# ==================================================================
# 🚨 B案戦略のためのグローバル変数とパラメータ 🚨
# ==================================================================

# --- 戦略パラメータ ---
STRATEGY_SYMBOL = "XAUUSDm"
# B案に基づき、M1をポーリングする
STRATEGY_LOOP_TIMEFRAME = "M1"
# アルファ純化に使用する市場プロキシの時間足
MARKET_PROXY_TIMEFRAME = "M5"
MARKET_PROXY_LOOKBACK = 5

# リアルタイムエンジンが要求する全時間足
# (final_feature_set_v2.txt に基づく)
ALL_TIMEFRAMES = {
    "M1": "M1",
    "M3": "M3",
    "M5": "M5",
    "M8": "M8",
    "M15": "M15",
    "M30": "M30",
    "H1": "H1",
    "H4": "H4",
    "H6": "H6",
    "H12": "H12",
    "D1": "D1",
    "W1": "W1",
    "MN": "MN",
}

# --- グローバルキャッシュ ---
g_last_processed_bar_time: Optional[int] = None  # M1ループの最終処理時刻
g_market_proxy: Optional[pd.DataFrame] = (
    None  # M5リターン (アルファ純化用) [Pandasに事前変換済み]
)
# [V11.1] g_context_features は廃止されました

# ==================================================================
# 🚨 リアルタイムデータ取得 (ZMQ経由) 🚨
# ==================================================================


def load_static_data() -> bool:
    """
    [V11.0 修正]
    静的データ（市場プロキシ）のみをロードする。
    旧来の 'context_features_v2.parquet' (D1文脈) は
    現在の Top 50 戦略では使用しないため廃止（ロード処理を削除）。
    """
    global g_market_proxy

    logger.info("--- 0. 静的データ (プロキシ) のロード ---")

    # 1. 市場プロキシ (M5) のロード (アルファ純化に必須)
    # (blueprintに定義がない場合は既存のパスロジックを使用)
    proxy_source_path = (
        config.S2_FEATURES_AFTER_AV
        / "feature_value_a_vast_universeA"
        / f"features_e1a_M5.parquet"
    )

    if not proxy_source_path.exists():
        logger.critical(f"市場プロキシファイルが見つかりません: {proxy_source_path}")
        return False

    try:
        logger.info(f"市場プロキシ {proxy_source_path} をロード中...")

        # Polarsで計算してPandasに変換
        g_market_proxy_pl = (
            pl.read_parquet(proxy_source_path)
            .select(["timestamp", "close"])
            .sort("timestamp")
            .with_columns(
                # (close[t] - close[t-5]) / close[t-5]
                (pl.col("close").pct_change(MARKET_PROXY_LOOKBACK)).alias(
                    "market_proxy"
                )
            )
            .select(["timestamp", "market_proxy"])
            .drop_nulls()  # 計算不能な先頭のNaNを削除
        )

        # TZ-Aware (UTC) に統一
        g_market_proxy = (
            g_market_proxy_pl.to_pandas().set_index("timestamp").tz_localize("UTC")
        )

        logger.info(
            f"✓ M5市場プロキシデータをPandas DFとしてキャッシュ ({len(g_market_proxy)}行)。"
        )
        return True

    except Exception as e:
        logger.critical(f"市場プロキシのロードに失敗: {e}", exc_info=True)
        return False


def initialize_data_buffer(
    engine: RealtimeFeatureEngine,
    bridge: MQL5BridgePublisherV3,  # [V11.0] V3を使用
    market_proxy_cache: pd.DataFrame,
) -> bool:
    """
    [V11.0 修正]
    M1データのみを可能な限り取得し、エンジンが M3～MN のすべてをリサンプリングする。
    [V7.0 修正]
    Top 50特徴量に含まれる最長の時間足（例: D1）を計算するために必要な
    M1データの量を動的に計算してリクエストする。
    """
    logger.info(
        "ZMQ経由で全時間足の履歴データを取得中 (V11.0: M1 Only / Zero-Serialization)..."
    )

    history_data_map = {}

    # 1. M1データのみをリクエストする
    tf_name = "M1"

    # 2. 必要なM1本数を計算 (動的算出)
    # エンジンの要求する最大ルックバック期間を M1 本数に換算する
    max_m1_bars_needed = 0

    # エンジンの定数定義を利用して上位足の必要期間をM1換算する
    # (例: D1でLookback 200の場合 -> 200 * 1440 = 288,000本必要)
    for tf_key, lookback_val in engine.lookbacks_by_tf.items():
        # 分数換算 (ALL_TIMEFRAMESの値を使用)
        minutes_per_bar = engine.ALL_TIMEFRAMES.get(tf_key)
        if minutes_per_bar is None:
            continue  # tickなどはスキップ

        # 必要なM1本数 = 上位足のLookback * その足の分数
        total_m1_needed = lookback_val * minutes_per_bar

        if total_m1_needed > max_m1_bars_needed:
            max_m1_bars_needed = total_m1_needed

    # 最低ライン(5万本) と エンジン要求(計算値 + マージン5000本) の大きい方を採用
    lookback = max(50000, max_m1_bars_needed + 5000)

    logger.info(
        f"  -> {tf_name} の履歴データを {lookback} 本取得中 (Engine要求: {max_m1_bars_needed} M1 bars)..."
    )

    # ✨ [V11.0] ZMQ リクエスト (PULLによるストリーミング受信)
    df_rates_m1 = bridge.request_historical_data(
        symbol=STRATEGY_SYMBOL,
        timeframe_name=tf_name,
        lookback_bars=lookback,
    )

    if df_rates_m1 is None or len(df_rates_m1) == 0:
        logger.error(
            f"ZMQ {tf_name} 履歴データの取得に失敗しました。起動を中止します。"
        )
        return False

    # 3. history_data_map には M1 だけを格納
    history_data_map[tf_name] = df_rates_m1

    # (M3, M5, M8, H6, H12, D1, W1, MN のリクエストは一切行わない)

    # エンジンに全履歴データを一括で渡す
    engine.fill_all_buffers(history_data_map, market_proxy_cache)

    # M1ループの開始時刻をセット
    global g_last_processed_bar_time
    if len(history_data_map["M1"]) > 0:
        g_last_processed_bar_time = int(
            history_data_map["M1"]["timestamp"].iloc[-1].timestamp()
        )
        logger.info(
            f"M1ループの最終処理時刻: {datetime.fromtimestamp(g_last_processed_bar_time, timezone.utc)}"
        )
    else:
        logger.warning("M1データが空のため、最終処理時刻を現在時刻に設定します。")
        g_last_processed_bar_time = int(time.time())

    return True


def get_latest_m1_bar(bridge: MQL5BridgePublisherV3) -> Optional[Dict[str, Any]]:
    """
    [V11.0 修正]
    ZMQ経由で ProjectForgeReceiver.mq5 から「M1」の新しい確定足ができたか確認する。
    """
    global g_last_processed_bar_time

    # ✨ ZMQ リクエストを送信
    bar = bridge.request_latest_m1_bar(symbol=STRATEGY_SYMBOL)

    if bar is None:
        return None

    # bar["time"] は Unix timestamp (int)、bar["timestamp"] は datetime
    if bar.get("time", 0) > g_last_processed_bar_time:
        g_last_processed_bar_time = bar["time"]  # 最終処理時刻を更新
        bar_time_dt = datetime.fromtimestamp(bar["time"], timezone.utc)
        logger.debug(f"新しいM1バーを検出: {bar_time_dt}")

        return {
            "timestamp": bar_time_dt,
            "open": bar["open"],
            "high": bar["high"],
            "low": bar["low"],
            "close": bar["close"],
            "volume": float(bar["volume"]),
        }
    else:
        return None


# [V11.1] get_correct_d1_context 関数は廃止されました

# ==================================================================
# メイン実行関数 (V11.0修正)
# ==================================================================


def main():
    logger.info("=" * 60)
    logger.info("🚀 Project Forge 統合実行システム V11.0 (ZMQハイブリッド版) 起動...")
    logger.info("=" * 60)

    state_manager: Optional[StateManager] = None
    bridge: Optional[MQL5BridgePublisherV3] = None
    risk_engine: Optional[ExtremeRiskEngineV2] = None
    feature_engine: Optional[RealtimeFeatureEngine] = None

    try:
        # --- 0. 静的データ (プロキシ) のロード ---
        logger.info("--- 0. 静的データ (プロキシ) のロード ---")
        if not load_static_data():
            raise RuntimeError("静的データのロードに失敗しました。")
        logger.info("✓ 静的データ（M5プロキシ）をキャッシュしました。")

        # --- 1. 状態管理 (StateManager) の初期化 ---
        logger.info("--- 1. 状態管理 (StateManager) の初期化 ---")
        state_manager = StateManager(
            checkpoint_dir=str(config.STATE_CHECKPOINT_DIR),
            event_log_path=str(config.STATE_EVENT_LOG),
        )
        initial_state = state_manager.load_checkpoint()
        if initial_state:
            logger.info(f"✓ 状態を復元 (Equity: {initial_state.current_equity:.2f})")
        else:
            logger.warning(
                "チェックポイントが見つかりません。ブローカー状態から初期化します。"
            )

        # --- 2. 通信 (MQL5BridgeV3) の初期化 ---
        logger.info("--- 2. 通信 (MQL5BridgeV3 - V11.0) の初期化 ---")
        # [V11.0] 3系統のエンドポイントを設定
        bridge_config = BridgeConfig(
            control_endpoint=config.ZMQ["control_endpoint"],
            data_endpoint=config.ZMQ["data_endpoint"],
            heartbeat_endpoint=config.ZMQ["heartbeat_endpoint"],
        )
        bridge = MQL5BridgePublisherV3(bridge_config)
        if not bridge.connect():
            raise RuntimeError(
                "MQL5ブリッジへの接続に失敗しました。EA(V11.0)が起動しているか確認してください。"
            )
        logger.info("✓ MQL5ブリッジ接続完了 (Control/Data/Heartbeat)。")

        # --- 3. ブローカー状態との整合性検証 ---
        logger.info("--- 3. ブローカー状態との整合性検証 ---")
        broker_state = bridge.request_broker_state()
        if broker_state:
            state_manager.reconcile_with_broker(broker_state)
            logger.info("✓ 状態の整合性を確保しました。")
        else:
            logger.warning(
                "ブローカー状態の取得に失敗しました (V11.0では起動時スキップの場合あり)。"
            )

        # --- 4. リスクエンジン (ExtremeRiskEngineV2) の初期化 ---
        logger.info("--- 4. リスクエンジン (ExtremeRiskEngineV2) の初期化 ---")
        risk_engine = ExtremeRiskEngineV2(
            config_path=str(config.CONFIG_RISK),
            state_manager=state_manager,
            m1_base_path=str(config.S7_M1_MODEL_PKL),
            m1_calib_path=str(config.S7_M1_CALIBRATED),
            m2_base_path=str(config.S7_M2_MODEL_PKL),
            m2_calib_path=str(config.S7_M2_CALIBRATED),
        )
        logger.info("✓ リスクエンジンを初期化しました（Base+Calibratorロード完了）。")

        # --- 5. AIモデルのロード ---
        # (初期化時に一括ロード済みのため、個別の呼び出しは削除)
        logger.info("--- 5. AIモデル (ロード済み) ---")
        logger.info("✓ AIモデルのロード完了。")

        # --- 6. リアルタイム特徴量エンジン (マルチバッファ) の初期化 ---
        logger.info(
            "--- 6. リアルタイム特徴量エンジンの初期化 (ゼロ・シリアライズ) ---"
        )

        # [推奨修正] Top 50 JSON を明示的に指定
        feature_engine = RealtimeFeatureEngine(
            feature_list_path=str(project_root / "models" / "TOP_50_FEATURES.json")
        )
        # ✨ V3 bridge と g_market_proxy を渡す
        if not initialize_data_buffer(feature_engine, bridge, g_market_proxy):
            raise RuntimeError("特徴量エンジンのマルチバッファ充填に失敗しました。")

        # --- 7. リアルタイム取引ループ開始 (M1ループ) ---
        logger.info("=" * 60)
        logger.info(f"🚀 リアルタイム取引ループ開始 ")
        logger.info("=" * 60)

        while True:
            try:
                # (A) M1の新しいバーの確定を待機
                new_m1_bar = get_latest_m1_bar(bridge)
                if new_m1_bar is None:
                    time.sleep(0.5)  # 【推奨】1秒間隔に変更（または 0.5）
                    continue

                # 新しい足が確定したタイミングで、ブローカーと状態を同期する
                broker_state_sync = bridge.request_broker_state()
                if broker_state_sync:
                    state_manager.reconcile_with_broker(broker_state_sync)

                # (B) M1バーをエンジンに渡し、シグナルを待つ
                # ※ (矛盾①解決のため) 純化用のM5プロキシデータも渡す
                # ※ (矛盾②解決のため) エンジンは内部で全15バッファを更新
                # ※ (B案戦略のため) エンジンは内部で全時間足のR4判定とシグナル生成
                signal_list = feature_engine.process_new_m1_bar(
                    new_m1_bar, g_market_proxy
                )

                if not signal_list:
                    continue  # シグナルなし

                # (C) シグナル処理ループ
                for signal in signal_list:
                    # `signal` は以下を含むと仮定:
                    #   - signal.features (純化済みの [1, 304] ベクトル)
                    #   - signal.timestamp (datetime)
                    #   - signal.timeframe (例: "M15")
                    #   - signal.market_info (V4 R4ルールの PT/SL/Payoff, ATR値など)

                    logger.info("-" * 30)
                    logger.info(
                        f"🔥 {signal.timeframe} R4 シグナル検知 @ {signal.market_info['current_price']:.3f} (ATR: {signal.market_info['atr_value']:.2f})"
                    )

                    # [V11.1] D1文脈結合処理は削除 (Top 50戦略では不要)
                    # signal.market_info は R4由来の情報のみを持つ

                    # (E) AIとリスクエンジンによるコマンド生成
                    command = risk_engine.generate_trade_command(
                        features=signal.features,  # (純化済み)
                        market_info=signal.market_info,  # (R4ルールのみ)
                        current_time=signal.timestamp,
                    )

                    # (F) 発注
                    if command["action"] != "HOLD":
                        logger.info(
                            f"-> 発注コマンドを送信: {command['action']} {command['lots']} lots"
                        )
                        success = bridge.send_trade_command(command)
                        if success:
                            logger.info("✓ コマンド送信成功 (ACK受信)")

                            # ポジション反映待ち (5秒待機)
                            logger.info("⏳ ポジション反映待ち (5秒待機)...")
                            time.sleep(5.0)

                            # 待機後、即座に状態を同期して「ポジション保有中」であることを認識させる
                            broker_state_after = bridge.request_broker_state()
                            if broker_state_after:
                                state_manager.reconcile_with_broker(broker_state_after)
                        else:
                            logger.error("✗ コマンド送信失敗 (NACKまたはタイムアウト)")
                    else:
                        logger.info(f"-> HOLD (エントリー見送り): {command['reason']}")

            except Exception as loop_error:
                logger.error(f"取引ループでエラーが発生: {loop_error}", exc_info=True)
                time.sleep(10)

    except KeyboardInterrupt:
        logger.info("\n" + "=" * 60)
        logger.info("... システム終了シグナル (Ctrl+C) を受信 ...")
        logger.info("=" * 60)
    except Exception as e:
        logger.critical(f"起動シーケンスで致命的なエラーが発生: {e}", exc_info=True)

    finally:
        # --- 8. グレースフルシャットダウン ---
        logger.info("--- 8. シャットダウン処理中 ---")
        if bridge:
            bridge.disconnect()
            logger.info("MQL5ブリッジを切断しました。")
        if state_manager and state_manager.current_state:
            state_manager.save_checkpoint(state_manager.current_state)
            logger.info("最終状態をチェックポイントに保存しました。")
        logger.info("MetaTrader5との接続をシャットダウンしました。")
        logger.info("=" * 60)
        logger.info("👋 Project Forge 正常終了")
        logger.info("=" * 60)


if __name__ == "__main__":
    main()
