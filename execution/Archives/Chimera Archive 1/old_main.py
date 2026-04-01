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
from execution.state_manager import (
    StateManager,
    SystemState,
    BrokerStateDict,
    EventType,
)

# [V11.0] V3 (ハイブリッドZMQ/ゼロ・シリアライズ) をインポート
from execution.mql5_bridge_publisher import MQL5BridgePublisherV3, BridgeConfig
from execution.extreme_risk_engine import ExtremeRiskEngineV5, MarketInfo
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
# ==================================================================
# 🚨 Project Cimera (V5) 戦略パラメータ 🚨
# ==================================================================
M2_PROBA_THRESHOLD = 0.55  # Two-Brain推論のシグナル発火閾値

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
    risk_engine: Optional[ExtremeRiskEngineV5] = None
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

        # --- 4. リスクエンジン (ExtremeRiskEngineV5) の初期化 ---
        logger.info("--- 4. リスクエンジン (ExtremeRiskEngineV5) の初期化 ---")
        risk_engine = ExtremeRiskEngineV5(config_path=str(config.CONFIG_RISK))
        logger.info("✓ リスクエンジンを初期化しました（Base+Calibratorロード完了）。")

        # --- 5. AIモデル (Two-Brain) と特徴量リストのロード ---
        logger.info("--- 5. AIモデルと専用特徴量リストのロード (V5仕様) ---")
        try:
            # 1. LightGBM Booster (M1/M2)
            models = {
                "long_m1": joblib.load(config.S7_M1_MODEL_LONG_PKL),
                "long_m2": joblib.load(config.S7_M2_MODEL_LONG_PKL),
                "short_m1": joblib.load(config.S7_M1_MODEL_SHORT_PKL),
                "short_m2": joblib.load(config.S7_M2_MODEL_SHORT_PKL),
            }

            # 2. 確率較正モデル (IsotonicRegression)
            calibrators = {
                "long_m2": joblib.load(config.S7_M2_CALIBRATED_LONG),
                "short_m2": joblib.load(config.S7_M2_CALIBRATED_SHORT),
            }

            # 3. 専用特徴量リストの読み込み関数
            def load_feature_list(filepath: Path) -> list:
                with open(filepath, "r") as f:
                    return [
                        line.strip()
                        for line in f
                        if line.strip() and not line.startswith("is_trigger")
                    ]

            feature_lists = {
                "long_m1": load_feature_list(
                    config.S3_SELECTED_FEATURES_DIR / "m1_long_features.txt"
                ),
                "long_m2": load_feature_list(
                    config.S3_SELECTED_FEATURES_DIR / "m2_long_features.txt"
                ),
                "short_m1": load_feature_list(
                    config.S3_SELECTED_FEATURES_DIR / "m1_short_features.txt"
                ),
                "short_m2": load_feature_list(
                    config.S3_SELECTED_FEATURES_DIR / "m2_short_features.txt"
                ),
            }

            logger.info(
                "✓ Two-Brainモデル、較正器、および専用特徴量リストのロード完了。"
            )
        except Exception as e:
            raise RuntimeError(f"モデルまたは特徴量リストのロードに失敗しました: {e}")

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
                # ==========================================================
                # 【シミュレーター完全同期 1】 タイムアウト(TO)決済の監視と実行
                # ==========================================================
                current_time = datetime.now(timezone.utc)
                if state_manager and state_manager.current_state:
                    for trade in state_manager.current_state.trades:
                        duration_mins = trade.get_duration_minutes(current_time)

                        # シミュレーター仕様: Longは15分、Shortは5分でタイムアウト(TO)
                        is_timeout = False
                        if trade.direction == "BUY" and duration_mins >= 15.0:
                            is_timeout = True
                        elif trade.direction == "SELL" and duration_mins >= 5.0:
                            is_timeout = True

                        if is_timeout:
                            logger.info(
                                f"⏱️ タイムアウト(TO)条件到達。強制決済を実行: Ticket={trade.ticket}, Direction={trade.direction}, Duration={duration_mins:.1f}分"
                            )
                            # ZMQで単一ポジションの決済コマンドを送信
                            bridge.send_trade_command(
                                {
                                    "action": "CLOSE",
                                    "ticket": trade.ticket,
                                    "magic": 77777,
                                }
                            )

                            # イベントを発火させ、ローカル状態から削除
                            event_data = {
                                "ticket": trade.ticket,
                                "close_reason": "TO",
                                "max_consecutive_sl": 2,  # バックテスト設定に準拠
                                "cooldown_minutes_after_sl": 10,
                            }
                            state_manager.append_event(
                                EventType.POSITION_CLOSED, event_data
                            )
                            event_dict = {
                                "event_type": EventType.POSITION_CLOSED.value,
                                "data": event_data,
                                "timestamp": current_time.isoformat(),
                            }
                            state_manager.current_state = state_manager._apply_event(
                                state_manager.current_state, event_dict
                            )

                # ==========================================================
                # 【シミュレーター完全同期 2】 サイレント・クローズ(SL/PT)の確実な捕捉
                # ==========================================================
                # ※ MQL5側 (ProjectForgeReceiver.mq5) に request_recent_history() を実装する前提
                if hasattr(bridge, "request_recent_history"):
                    recent_history = bridge.request_recent_history()
                    if recent_history:
                        for closed_pos in recent_history:
                            ticket = closed_pos.get("ticket")
                            # ローカルでまだActiveとして認識されているポジションが決済されていた場合
                            active_trade = next(
                                (
                                    t
                                    for t in state_manager.current_state.trades
                                    if t.ticket == ticket
                                ),
                                None,
                            )
                            if active_trade:
                                reason = closed_pos.get("close_reason", "UNKNOWN")
                                logger.warning(
                                    f"🔔 ブローカー側での決済を検知 (サイレントクローズ捕捉): Ticket={ticket}, Reason={reason}"
                                )
                                event_data = {
                                    "ticket": ticket,
                                    "close_reason": reason,  # "SL" または "PT"
                                    "max_consecutive_sl": 2,
                                    "cooldown_minutes_after_sl": 10,
                                }
                                state_manager.append_event(
                                    EventType.POSITION_CLOSED, event_data
                                )
                                event_dict = {
                                    "event_type": EventType.POSITION_CLOSED.value,
                                    "data": event_data,
                                    "timestamp": datetime.now(timezone.utc).isoformat(),
                                }
                                state_manager.current_state = (
                                    state_manager._apply_event(
                                        state_manager.current_state, event_dict
                                    )
                                )

                # 新しい足が確定したタイミングで、ブローカーと最終同期
                broker_state_sync = bridge.request_broker_state()
                if broker_state_sync:
                    state_manager.reconcile_with_broker(broker_state_sync)

                # (A) M1の新しいバーの確定を待機
                new_m1_bar = get_latest_m1_bar(bridge)
                if new_m1_bar is None:
                    time.sleep(1.0)  # CPU負荷軽減
                    continue

                # (B) M1バーをエンジンに渡し、シグナルを待つ
                signal_list = feature_engine.process_new_m1_bar(
                    new_m1_bar, g_market_proxy
                )

                if not signal_list:
                    continue  # シグナルなし

                # (C) シグナル処理ループ
                for signal in signal_list:
                    logger.info("-" * 30)
                    logger.info(
                        f"🔍 {signal.timeframe} R4 シグナル検知 @ {signal.market_info['current_price']:.3f}"
                    )

                    # 全特徴量を持つ辞書を作成 (リストの場合は feature_engine.feature_names と zip して辞書化する想定)
                    # 例: feature_dict = dict(zip(feature_engine.feature_names, signal.features))
                    feature_dict = signal.feature_dict

                    # --- ユーティリティ: 特徴量抽出 ---
                    def extract_features(f_list, f_dict):
                        return np.array([[f_dict.get(f, 0.0) for f in f_list]])

                    # --- Two-Brain 並列推論 (LightGBM Booster は predict_proba ではなく predict を使用) ---
                    # 1. Long 側推論
                    X_long_m1 = extract_features(feature_lists["long_m1"], feature_dict)
                    p_long_m1 = models["long_m1"].predict(X_long_m1)[0]

                    feature_dict["m1_pred_proba"] = p_long_m1

                    X_long_m2 = extract_features(feature_lists["long_m2"], feature_dict)
                    p_long_m2_raw = models["long_m2"].predict(X_long_m2)[0]

                    # 2. Short 側推論
                    X_short_m1 = extract_features(
                        feature_lists["short_m1"], feature_dict
                    )
                    p_short_m1 = models["short_m1"].predict(X_short_m1)[0]

                    feature_dict["m1_pred_proba"] = p_short_m1

                    X_short_m2 = extract_features(
                        feature_lists["short_m2"], feature_dict
                    )
                    p_short_m2_raw = models["short_m2"].predict(X_short_m2)[0]

                    # 3. M2確率の較正 (Isotonic Regression)
                    p_long_m2_calib = calibrators["long_m2"].predict([p_long_m2_raw])[0]
                    p_short_m2_calib = calibrators["short_m2"].predict(
                        [p_short_m2_raw]
                    )[0]

                    logger.info(
                        f"🧠 [M2 Calibrated] Long: {p_long_m2_calib:.4f} | Short: {p_short_m2_calib:.4f}"
                    )
                    THRESHOLD = 0.50
                    should_trade_long = p_long_m2_calib > THRESHOLD
                    should_trade_short = p_short_m2_calib > THRESHOLD

                    # --- 1. 同時発注禁止 & 両建て防止 (prevent_simultaneous_orders) ---
                    # ※バックテストの default_config.prevent_simultaneous_orders = True に準拠
                    prevent_simultaneous_orders = True

                    if prevent_simultaneous_orders:
                        # ステップA: ノイズ検知（両方向のシグナルが同時に出た場合は問答無用で両方キャンセル）
                        if should_trade_long and should_trade_short:
                            logger.warning(
                                f"⚠️ 【防衛線1-A】相場混乱検知: Long/Shortの同時シグナルにつき強制キャンセル。"
                            )
                            should_trade_long = False
                            should_trade_short = False
                        else:
                            # ステップB: 逆行ポジションの確認（片方向のみシグナルが出ている場合）
                            if (
                                should_trade_long
                                and state_manager.get_active_positions_count("SELL") > 0
                            ):
                                logger.info(
                                    "⚠️ 【防衛線1-B】逆方向(SELL)のポジションを保有しているため、Longシグナルを見送ります（両建て防止）。"
                                )
                                should_trade_long = False

                            if (
                                should_trade_short
                                and state_manager.get_active_positions_count("BUY") > 0
                            ):
                                logger.info(
                                    "⚠️ 【防衛線1-B】逆方向(BUY)のポジションを保有しているため、Shortシグナルを見送ります（両建て防止）。"
                                )
                                should_trade_short = False

                    # --- 2. サーキットブレーカー（連続SL）のチェック ---
                    # ステップC: クールダウン中かどうかの最終確認
                    if should_trade_long and state_manager.is_cooldown_active("BUY"):
                        logger.warning(
                            "🔒 【防衛線2】BUY方向は現在Cooldown中（連続SLロックアウト）。エントリーを破棄します。"
                        )
                        should_trade_long = False

                    if should_trade_short and state_manager.is_cooldown_active("SELL"):
                        logger.warning(
                            "🔒 【防衛線2】SELL方向は現在Cooldown中（連続SLロックアウト）。エントリーを破棄します。"
                        )
                        should_trade_short = False

                    # --- 3. 最終的な方向と確率の決定 ---
                    direction = None
                    final_proba = 0.0

                    if should_trade_long:
                        direction = "BUY"
                        final_proba = p_long_m2_calib
                    elif should_trade_short:
                        direction = "SELL"
                        final_proba = p_short_m2_calib
                    else:
                        continue  # どちらもFalseなら見送り (次のループへ)
                    logger.info(
                        f"🟢 防衛線クリア。{direction}方向のエントリー手続きを開始します。"
                    )

                    # --- リスクエンジンへの委譲 (V5 Pure Math Mode) ---
                    # ATRはリアルタイムエンジンの値(atr_value)を優先取得
                    current_atr = signal.market_info.get(
                        "atr_value", signal.market_info.get("atr", 5.0)
                    )

                    command = risk_engine.generate_trade_command(
                        action=direction,  # 'BUY' or 'SELL'
                        p_long=p_long_m2_calib,
                        p_short=p_short_m2_calib,
                        current_price=signal.market_info["current_price"],
                        atr=current_atr,
                        equity=state_manager.current_state.current_equity,
                        sl_multiplier=risk_engine.config.get(
                            "base_atr_sl_multiplier", 5.0
                        ),
                        tp_multiplier=risk_engine.config.get(
                            "base_atr_tp_multiplier", 1.0
                        ),
                    )

                    # V5エンジンは独立したため、ここでイベントログを記録する
                    if command["action"] != "HOLD" and state_manager.use_event_sourcing:
                        state_manager.append_event(
                            EventType.TRADE_SIGNAL_SENT, {"command": command}
                        )

                    # --- 発注 ---
                    if command["action"] != "HOLD":
                        logger.info(
                            f"-> 発注コマンドを送信: {command['action']} {command['lots']} lots (SL:{command.get('stop_loss')}, TP:{command.get('take_profit')})"
                        )
                        success = bridge.send_trade_command(command)
                        if success:
                            logger.info("✓ コマンド送信成功 (ACK受信)")
                            logger.info("⏳ ポジション反映待ち (5秒待機)...")
                            time.sleep(5.0)

                            broker_state_after = bridge.request_broker_state()
                            if broker_state_after:
                                state_manager.reconcile_with_broker(broker_state_after)
                        else:
                            logger.error("✗ コマンド送信失敗 (NACKまたはタイムアウト)")
                    else:
                        logger.info(
                            f"-> HOLD (リスクエンジン判断): {command['reason']}"
                        )

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
