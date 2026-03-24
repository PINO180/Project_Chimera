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
import csv  # ★これを追加

# --- プロジェクトのルートディレクトリをPythonの検索パスに追加 ---
project_root = Path(__file__).resolve().parents[0]
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

# --- 必要なコンポーネントをインポート ---
import blueprint as config
import os  # ▼追加: Configのホットリロード用ファイルの更新日時取得
from execution.state_manager import (
    StateManager,
    SystemState,
    BrokerStateDict,
    EventType,
)

# [V11.0] V3 (ハイブリッドZMQ/ゼロ・シリアライズ) をインポート
from execution.mql5_bridge_publisher import MQL5BridgePublisherV3, BridgeConfig
from execution.extreme_risk_engine import (
    ExtremeRiskEngineV5,
    MarketInfo,
)  # [FIX-2] MarketInfo は extreme_risk_engine に定義済み
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
risk_engine = ExtremeRiskEngineV5(config_path=str(config.CONFIG_RISK))
M2_PROBA_THRESHOLD = risk_engine.config.get("m2_proba_threshold", 0.50)
MAX_DRAWDOWN = risk_engine.config.get("max_drawdown", 0.50)
MAX_POSITIONS = risk_engine.config.get("max_positions", 1000)

# ==================================================================
# 🚨 リアルタイムデータ取得 (ZMQ経由) 🚨
# ==================================================================


def load_static_data() -> bool:
    """
    [V11.2 修正] 静的ファイルへの依存を完全に排除。
    初期の市場プロキシはZMQから取得したM1履歴データから動的生成するため、
    ここでは空のデータフレームをセットアップするのみとします。
    """
    global g_market_proxy

    logger.info("--- 0. 初期データ構造のセットアップ ---")

    # 空のDataFrameで初期化 (型とインデックスを定義)
    g_market_proxy = pd.DataFrame(
        columns=["market_proxy"], index=pd.DatetimeIndex([], tz="UTC", name="timestamp")
    )

    logger.info("✓ プロキシ用DataFrameの初期化完了 (実データはZMQから動的生成します)")
    return True


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

    # D1のOLS学習サンプルを十分確保するため最低ラインを200,000本に引き上げ
    # D1: 200,000 / 1440 ≈ 138本 → OLS2016サンプルには届かないが実用上許容範囲
    lookback = max(200000, max_m1_bars_needed + 5000)

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

    # ▼▼▼ 修正: M1履歴データから市場プロキシ(M5)を動的生成 ▼▼▼
    global g_market_proxy
    logger.info("  -> 取得したM1履歴データから初期の市場プロキシ(M5)を動的生成中...")
    try:
        temp_df = df_rates_m1[["timestamp", "close"]].copy()
        temp_df.set_index("timestamp", inplace=True)

        # M1をM5にリサンプリング(5分ごとの終値)し、リターンを計算
        m5_close = (
            temp_df["close"]
            .resample("5min", label="right", closed="right")
            .last()
            .dropna()
        )
        proxy_df = (
            m5_close.pct_change(MARKET_PROXY_LOOKBACK)
            .to_frame(name="market_proxy")
            .dropna()
        )

        # タイムゾーンの適応(UTC)
        if proxy_df.index.tz is None:
            g_market_proxy = proxy_df.tz_localize("UTC")
        else:
            g_market_proxy = proxy_df.tz_convert("UTC")

        logger.info(f"  ✓ 動的市場プロキシ生成完了 ({len(g_market_proxy)}行)")
    except Exception as e:
        logger.warning(
            f"  ⚠ 動的市場プロキシ生成に失敗しました（空のプロキシで続行）: {e}"
        )
    # ▲▲▲ ここまで修正 ▲▲▲

    # 3. history_data_map には M1 だけを格納
    history_data_map[tf_name] = df_rates_m1

    # (M3, M5, M8, H6, H12, D1, W1, MN のリクエストは一切行わない)

    # エンジンに全履歴データを一括で渡す
    # [修正] market_proxy_cache ではなく、たった今動的生成した g_market_proxy を渡す
    engine.fill_all_buffers(history_data_map, g_market_proxy)

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

    # ▼▼▼ 追加: 時刻が None のまま突入してきたら強制的に 0 にしてエラーを回避 ▼▼▼
    if g_last_processed_bar_time is None:
        g_last_processed_bar_time = 0
    # ▲▲▲ ここまで追加 ▲▲▲

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
            "tick_volume_mean_5": float(bar.get("tick_volume_mean_5", 0.0)),
            "spread": float(bar.get("spread", 16.0)),  # ▼追加: MT5からスプレッドを取得
        }
    else:
        return None


# [V11.1] get_correct_d1_context 関数は廃止されました

# ==================================================================
# メイン実行関数 (V11.0修正)
# ==================================================================


def main():
    global g_market_proxy

    logger.info("=" * 60)
    logger.info("🚀 Project Forge 統合実行システム V11.0 (ZMQハイブリッド版) 起動...")
    logger.info("=" * 60)

    # ▼▼▼ 追加: 推論値ログ用CSVのセットアップ ▼▼▼
    predictions_csv_path = config.LOGS_DIR / "m1_m2_predictions_log.csv"
    if not predictions_csv_path.exists():
        with open(predictions_csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            # Excelの1行目になるヘッダーを書き込む
            writer.writerow(
                [
                    "Timestamp",
                    "CurrentPrice",
                    "Long_M1_Raw",
                    "Long_M1_Calib",
                    "Short_M1_Raw",
                    "Short_M1_Calib",
                    "Long_M2_Raw",
                    "Long_M2_Calib",
                    "Short_M2_Raw",
                    "Short_M2_Calib",
                ]
            )
    # ▲▲▲ ここまで追加 ▲▲▲

    state_manager: Optional[StateManager] = None
    bridge: Optional[MQL5BridgePublisherV3] = None
    risk_engine: Optional[ExtremeRiskEngineV5] = None
    feature_engine: Optional[RealtimeFeatureEngine] = None

    try:
        # --- 0. 静的データ (プロキシ) のロード ---
        logger.info("")  # ★空白行を追加
        logger.info("--- 0. 静的データ (プロキシ) のロード ---")
        if not load_static_data():
            raise RuntimeError("静的データのロードに失敗しました。")
        logger.info("✓ 静的データ（M5プロキシ）をキャッシュしました。")

        # --- 1. 状態管理 (StateManager) の初期化 ---
        logger.info("")  # ★空白行を追加
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
        logger.info("")  # ★空白行を追加
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
        logger.info("")  # ★空白行を追加
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
        logger.info("")  # ★空白行を追加
        logger.info("--- 4. リスクエンジン (ExtremeRiskEngineV5) の初期化 ---")
        risk_engine = ExtremeRiskEngineV5(config_path=str(config.CONFIG_RISK))
        logger.info("✓ リスクエンジンを初期化しました（Base+Calibratorロード完了）。")

        # --- 5. AIモデル (Two-Brain) と特徴量リストのロード ---
        logger.info("")  # ★空白行を追加
        logger.info("--- 5. AIモデルと専用特徴量リストのロード (V5仕様) ---")
        try:
            # 1. LightGBM Booster (M1/M2)
            models = {
                "long_m1": joblib.load(config.S7_M1_MODEL_LONG_PKL),
                "long_m2": joblib.load(config.S7_M2_MODEL_LONG_PKL),
                "short_m1": joblib.load(config.S7_M1_MODEL_SHORT_PKL),
                "short_m2": joblib.load(config.S7_M2_MODEL_SHORT_PKL),
            }

            # 2. 専用特徴量リストの読み込み関数
            def load_feature_list(filepath: Path) -> list:
                with open(filepath, "r") as f:
                    return [
                        line.strip()
                        for line in f
                        if line.strip() and not line.startswith("is_trigger")
                    ]

            feature_lists = {
                "long_m1": load_feature_list(
                    config.S3_SELECTED_FEATURES_PURIFIED_DIR / "m1_long_features.txt"
                ),
                "long_m2": load_feature_list(
                    config.S3_SELECTED_FEATURES_PURIFIED_DIR / "m2_long_features.txt"
                ),
                "short_m1": load_feature_list(
                    config.S3_SELECTED_FEATURES_PURIFIED_DIR / "m1_short_features.txt"
                ),
                "short_m2": load_feature_list(
                    config.S3_SELECTED_FEATURES_PURIFIED_DIR / "m2_short_features.txt"
                ),
            }

            logger.info(
                "✓ Two-Brainモデル、較正器、および専用特徴量リストのロード完了。"
            )
        except Exception as e:
            raise RuntimeError(f"モデルまたは特徴量リストのロードに失敗しました: {e}")

        # --- 6. リアルタイム特徴量エンジン (マルチバッファ) の初期化 ---
        logger.info("")
        logger.info(
            "--- 6. リアルタイム特徴量エンジンの初期化 (ゼロ・シリアライズ) ---"
        )

        feature_engine = RealtimeFeatureEngine(
            feature_list_path=str(config.S3_FEATURES_FOR_TRAINING_V5)
        )

        # ▼▼▼ 修正: スナップショットからの爆速復帰と差分取得 ▼▼▼
        state_file = config.STATE_CHECKPOINT_DIR / "feature_engine_state.pkl"
        is_warmed_up = False

        if state_file.exists():
            try:
                # 1. とりあえずロードを試みる
                if feature_engine.load_state(str(state_file)):
                    logger.info("⚡ スナップショットからの爆速復帰に成功しました！")
                    if len(feature_engine.m1_dataframe) > 0:
                        global g_last_processed_bar_time
                        g_last_processed_bar_time = int(
                            feature_engine.m1_dataframe[-1]["timestamp"].timestamp()
                        )

                        # 差分（ギャップ）の時間を計算
                        now_ts = int(datetime.now(timezone.utc).timestamp())
                        diff_minutes = int((now_ts - g_last_processed_bar_time) / 60)

                        if diff_minutes > 0:
                            logger.info(
                                f"  -> 停止していた {diff_minutes} 分の差分データをM1履歴から取得して穴埋めします..."
                            )
                            # 余裕を持たせて diff_minutes + 10 本を取得
                            diff_df = bridge.request_historical_data(
                                symbol=STRATEGY_SYMBOL,
                                timeframe_name="M1",
                                lookback_bars=diff_minutes + 10,
                            )
                            if diff_df is not None and len(diff_df) > 0:
                                # まだ処理していない新しいバーだけを抽出してエンジンに流し込む
                                new_bars = diff_df[
                                    diff_df["timestamp"]
                                    > datetime.fromtimestamp(
                                        g_last_processed_bar_time, timezone.utc
                                    )
                                ]
                                for _, row in new_bars.iterrows():
                                    bar_dict = {
                                        "timestamp": row["timestamp"],
                                        "open": row["open"],
                                        "high": row["high"],
                                        "low": row["low"],
                                        "close": row["close"],
                                        "volume": float(row["volume"]),
                                        "spread": 16.0,  # 過去の差分なので固定値で代用
                                    }
                                    # プロキシの更新
                                    past_close = feature_engine.m1_dataframe[-25][
                                        "close"
                                    ]
                                    new_proxy_val = (
                                        bar_dict["close"] - past_close
                                    ) / past_close
                                    new_proxy_df = pd.DataFrame(
                                        {"market_proxy": [new_proxy_val]},
                                        index=pd.DatetimeIndex(
                                            [bar_dict["timestamp"]], tz="UTC"
                                        ),
                                    )
                                    # ▼▼▼ FutureWarning対策: 空の場合はconcatせずに代入 ▼▼▼
                                    if g_market_proxy.empty:
                                        g_market_proxy = new_proxy_df
                                    else:
                                        g_market_proxy = pd.concat(
                                            [g_market_proxy, new_proxy_df]
                                        )
                                    # ▲▲▲ ここまで修正 ▲▲▲

                                    feature_engine.process_new_m1_bar(
                                        bar_dict, g_market_proxy
                                    )
                                    g_last_processed_bar_time = int(
                                        row["timestamp"].timestamp()
                                    )

                                logger.info(
                                    f"✓ 差分 {len(new_bars)} 本の追いつき計算が完了しました！完全に同期しています。"
                                )

                            # ▼▼▼ 追加: 穴埋め完了直後に確実にセーブする ▼▼▼
                            feature_engine.save_state(str(state_file))
                            logger.info(
                                "💾 追いつき後の最新状態をスナップショットに保存しました。"
                            )
                            # ▲▲▲ ここまで追加 ▲▲▲

                    is_warmed_up = True

            # ▼▼▼ 追加: 破損を検知した場合は自動でPickleを削除し、再計算ルートへ流す ▼▼▼
            except Exception as e:
                logger.warning(
                    f"⚠️ スナップショットの破損を検知しました ({e})。ファイルを自動破棄してフルウォームアップを実行します。"
                )
                try:
                    state_file.unlink(missing_ok=True)  # 物理的にファイルを削除
                except Exception as del_e:
                    logger.error(f"ファイルの削除に失敗しました: {del_e}")
                is_warmed_up = (
                    False  # Falseにして下のフルウォームアップ処理へ合流させる
                )
            # ▲▲▲ ここまで追加 ▲▲▲

        # スナップショットが無い場合のみ、フルウォームアップを行う
        if not is_warmed_up:
            if not initialize_data_buffer(feature_engine, bridge, g_market_proxy):
                raise RuntimeError("特徴量エンジンのマルチバッファ充填に失敗しました。")
        # ▲▲▲ ここまで修正 ▲▲▲

        # --- 7. リアルタイム取引ループ開始 (M1ループ) ---
        logger.info("=" * 60)
        logger.info(f"🚀 リアルタイム取引ループ開始 ")
        logger.info("=" * 60)

        # ▼追加: ホットリロード用のタイムスタンプ監視
        last_config_mtime = os.path.getmtime(config.CONFIG_RISK)
        # ▼追加: 定期セーブ用のタイマー
        last_snapshot_time = time.time()

        while True:
            try:
                # ▼▼▼ 追加: 15分間隔でスナップショットを強制保存 ▼▼▼
                current_time_sec = time.time()
                if current_time_sec - last_snapshot_time > 900:  # 900秒 = 15分
                    if feature_engine:
                        state_file = (
                            config.STATE_CHECKPOINT_DIR / "feature_engine_state.pkl"
                        )
                        feature_engine.save_state(str(state_file))
                        logger.info(
                            "💾 [定期保存] 特徴量エンジンの状態をスナップショットに保存しました。"
                        )
                    last_snapshot_time = current_time_sec
                # ▲▲▲ ここまで追加 ▲▲▲

                # ▼追加: 毎ループ、設定ファイル(risk_config.json)の更新日時をチェック
                current_mtime = os.path.getmtime(config.CONFIG_RISK)
                if current_mtime > last_config_mtime:
                    logger.info(
                        "⚙️ risk_config.json の更新を検知しました。再起動なしで動的に反映します！"
                    )
                    risk_engine = ExtremeRiskEngineV5(
                        config_path=str(config.CONFIG_RISK)
                    )
                    last_config_mtime = current_mtime
                # ==========================================================
                # 【シミュレーター完全同期 1】 タイムアウト(TO)決済の監視と実行
                # ==========================================================
                current_time = datetime.now(timezone.utc)
                if state_manager and state_manager.current_state:
                    for trade in state_manager.current_state.trades:
                        duration_mins = trade.get_duration_minutes(current_time)

                        # ▼▼▼ 修正: Optunaの最強パラメータ(Mixed)に準拠 ▼▼▼
                        is_timeout = False
                        # コンフィグから個別のTO時間を取得 (デフォルト60.0)
                        td_long = risk_engine.config.get("td_minutes_long", 60.0)
                        td_short = risk_engine.config.get("td_minutes_short", 60.0)

                        if trade.direction == "BUY" and duration_mins >= td_long:
                            is_timeout = True
                        elif trade.direction == "SELL" and duration_mins >= td_short:
                            is_timeout = True

                        if is_timeout:
                            logger.info(
                                f"⏱️ タイムアウト(TO)条件到達。強制決済を実行: Ticket={trade.ticket}, Direction={trade.direction}, Duration={duration_mins:.1f}分"
                            )
                            # ZMQで単一ポジションの決済コマンドを送信
                            # ▼▼▼ 修正: 幽霊ポジション・無限ループの防止 ▼▼▼
                            success = bridge.send_trade_command(
                                {
                                    "action": "CLOSE",
                                    "ticket": trade.ticket,
                                    "magic": 77777,
                                }
                            )

                            if success:
                                # 送信成功時のみローカル状態を更新
                                event_data = {
                                    "ticket": trade.ticket,
                                    "close_reason": "TO",
                                    "max_consecutive_sl": risk_engine.config.get(
                                        "max_consecutive_sl", 2
                                    ),
                                    "cooldown_minutes_after_sl": risk_engine.config.get(
                                        "cooldown_minutes_after_sl", 10
                                    ),
                                }
                                state_manager.apply_event_and_update(
                                    EventType.POSITION_CLOSED, event_data
                                )
                            else:
                                logger.warning(
                                    f"⚠️ タイムアウト決済コマンドの送信に失敗しました。次ループで再試行します: Ticket={trade.ticket}"
                                )
                            # ▲▲▲ ここまで修正 ▲▲▲

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
                                # [FIX-8] パブリック API apply_event_and_update を使用
                                event_data = {
                                    "ticket": ticket,
                                    "close_reason": reason,  # "SL" または "PT"
                                    "max_consecutive_sl": risk_engine.config.get(
                                        "max_consecutive_sl", 2
                                    ),
                                    "cooldown_minutes_after_sl": risk_engine.config.get(
                                        "cooldown_minutes_after_sl", 30
                                    ),
                                }
                                state_manager.apply_event_and_update(
                                    EventType.POSITION_CLOSED, event_data
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

                # ▼▼▼ 修正: 市場プロキシ (g_market_proxy) の永久凍結防止 ▼▼▼
                # global g_market_proxy  # ▲ 削除（関数の先頭に移動済みのため）
                if feature_engine and len(feature_engine.m1_dataframe) >= 25:
                    current_close = new_m1_bar["close"]
                    past_close = feature_engine.m1_dataframe[-25]["close"]
                    if past_close > 0:
                        new_proxy_val = (current_close - past_close) / past_close
                        new_proxy_df = pd.DataFrame(
                            {"market_proxy": [new_proxy_val]},
                            index=pd.DatetimeIndex([new_m1_bar["timestamp"]], tz="UTC"),
                        )
                        # ▼▼▼ FutureWarning対策: 空の場合はconcatせずに代入 ▼▼▼
                        if g_market_proxy.empty:
                            g_market_proxy = new_proxy_df
                        else:
                            g_market_proxy = pd.concat([g_market_proxy, new_proxy_df])
                        # ▲▲▲ ここまで修正 ▲▲▲
                        # メモリ溢れ保護
                        if len(g_market_proxy) > 10000:
                            g_market_proxy = g_market_proxy.iloc[-5000:]
                # ▲▲▲ ここまで修正 ▲▲▲

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

                    # ▼▼▼ 追加: 現在の保有ポジション数の集計と表示 ▼▼▼
                    current_longs = sum(
                        1
                        for t in state_manager.current_state.trades
                        if t.direction == "BUY"
                    )
                    current_shorts = sum(
                        1
                        for t in state_manager.current_state.trades
                        if t.direction == "SELL"
                    )
                    logger.info(
                        f"📊 現在の保有ポジション: Long {current_longs} / Short {current_shorts} (Total {current_longs + current_shorts})"
                    )
                    # ▲▲▲ ここまで追加 ▲▲▲

                    # 全特徴量を持つ辞書を作成 (リストの場合は feature_engine.feature_names と zip して辞書化する想定)
                    # 例: feature_dict = dict(zip(feature_engine.feature_names, signal.features))
                    feature_dict = signal.feature_dict.copy()

                    # --------------------------------------------------
                    # 【Long側】 Two-Brain 推論
                    # --------------------------------------------------
                    # 1. M1モデル (生の予測値のみ)
                    X_long_m1 = np.array(
                        [[feature_dict.get(f, 0.0) for f in feature_lists["long_m1"]]]
                    )
                    p_long_m1_raw = models["long_m1"].predict(X_long_m1)[0]

                    # 2. M2モデル (M1が0.50以上の場合のみ推論、それ以外は強制0.0)
                    if p_long_m1_raw >= 0.50:
                        feature_dict_long = feature_dict.copy()
                        feature_dict_long["m1_pred_proba"] = p_long_m1_raw
                        X_long_m2 = np.array(
                            [
                                [
                                    feature_dict_long.get(f, 0.0)
                                    for f in feature_lists["long_m2"]
                                ]
                            ]
                        )
                        p_long_m2_raw = models["long_m2"].predict(X_long_m2)[0]
                    else:
                        p_long_m2_raw = 0.0

                    # --------------------------------------------------
                    # 【Short側】 Two-Brain 推論
                    # --------------------------------------------------
                    # 1. M1モデル (生の予測値のみ)
                    X_short_m1 = np.array(
                        [[feature_dict.get(f, 0.0) for f in feature_lists["short_m1"]]]
                    )
                    p_short_m1_raw = models["short_m1"].predict(X_short_m1)[0]

                    # 2. M2モデル (M1が0.50以上の場合のみ推論、それ以外は強制0.0)
                    if p_short_m1_raw >= 0.50:
                        feature_dict_short = feature_dict.copy()
                        feature_dict_short["m1_pred_proba"] = p_short_m1_raw
                        X_short_m2 = np.array(
                            [
                                [
                                    feature_dict_short.get(f, 0.0)
                                    for f in feature_lists["short_m2"]
                                ]
                            ]
                        )
                        p_short_m2_raw = models["short_m2"].predict(X_short_m2)[0]
                    else:
                        p_short_m2_raw = 0.0

                    # ▼▼▼ 生(Raw)確率のみをスッキリ1行で表示 ▼▼▼
                    logger.info(
                        f"🧠 [Raw Proba] M1(L: {p_long_m1_raw:.4f}, S: {p_short_m1_raw:.4f}) -> "
                        f"M2(L: {p_long_m2_raw:.4f}, S: {p_short_m2_raw:.4f})"
                    )

                    # ▼▼▼ Excel用CSVに1行追記 ▼▼▼
                    try:
                        with open(
                            predictions_csv_path, "a", newline="", encoding="utf-8"
                        ) as f:
                            writer = csv.writer(f)
                            writer.writerow(
                                [
                                    signal.timestamp.strftime("%Y-%m-%d %H:%M:%S"),
                                    signal.market_info["current_price"],
                                    round(p_long_m1_raw, 4),
                                    0.0,  # 廃止(M1 Calib)
                                    round(p_short_m1_raw, 4),
                                    0.0,  # 廃止(M1 Calib)
                                    round(p_long_m2_raw, 4),
                                    0.0,  # 廃止(M2 Calib)
                                    round(p_short_m2_raw, 4),
                                    0.0,  # 廃止(M2 Calib)
                                ]
                            )
                    except Exception as e:
                        logger.warning(f"CSVへの書き込みに失敗しました: {e}")
                    # ▲▲▲ ここまで追加 ▲▲▲

                    # ▼▼▼ 修正: Delta (差分) フィルター & 判定理由の明記 ▼▼▼
                    current_m2_thresh = risk_engine.config.get(
                        "m2_proba_threshold", 0.30
                    )
                    current_m2_delta = risk_engine.config.get(
                        "m2_delta_threshold", 0.50
                    )

                    p_l = p_long_m2_raw
                    p_s = p_short_m2_raw
                    delta = abs(p_l - p_s)

                    should_trade_long = False
                    should_trade_short = False

                    # 条件分岐ごとに、弾いた理由（または通した理由）をログに出力
                    if delta >= current_m2_delta:
                        if p_l > p_s and p_l > current_m2_thresh:
                            should_trade_long = True
                            logger.info(
                                f"🎯 [Delta Filter: PASS] Delta {delta:.4f} >= {current_m2_delta} 尚且つ Long({p_l:.4f}) > {current_m2_thresh}"
                            )
                        elif p_s > p_l and p_s > current_m2_thresh:
                            should_trade_short = True
                            logger.info(
                                f"🎯 [Delta Filter: PASS] Delta {delta:.4f} >= {current_m2_delta} 尚且つ Short({p_s:.4f}) > {current_m2_thresh}"
                            )
                        else:
                            logger.info(
                                f"🚧 [Delta Filter: SKIP] 確信度不足 (勝つ方の確率が閾値 {current_m2_thresh} 以下)"
                            )
                    else:
                        logger.info(
                            f"🚧 [Delta Filter: SKIP] 差分不足 (Delta {delta:.4f} < 閾値 {current_m2_delta})"
                        )
                    # ▲▲▲ ここまで修正 ▲▲▲
                    if should_trade_long or should_trade_short:
                        try:
                            dump_csv_path = (
                                config.LOGS_DIR / "triggered_features_log.csv"
                            )
                            file_exists = dump_csv_path.exists()

                            with open(
                                dump_csv_path, "a", newline="", encoding="utf-8"
                            ) as f:
                                writer = csv.writer(f)
                                feature_keys = list(feature_dict.keys())

                                if not file_exists:
                                    header = [
                                        "Timestamp",
                                        "Action",
                                        "Price",
                                        "P_Long_M2",
                                        "P_Short_M2",
                                    ] + feature_keys
                                    writer.writerow(header)

                                action_str = "BUY" if should_trade_long else "SELL"

                                # ▼▼▼ 修正: CSVに記録する確率もRawにし、m1_pred_proba が 0 にならないよう上書き ▼▼▼
                                if should_trade_long:
                                    feature_dict["m1_pred_proba"] = p_long_m1_raw
                                else:
                                    feature_dict["m1_pred_proba"] = p_short_m1_raw

                                row_data = [
                                    signal.timestamp.strftime("%Y-%m-%d %H:%M:%S"),
                                    action_str,
                                    signal.market_info["current_price"],
                                    round(p_long_m2_raw, 4),  # CalibからRawへ変更
                                    round(p_short_m2_raw, 4),  # CalibからRawへ変更
                                ]
                                row_data.extend([feature_dict[k] for k in feature_keys])
                                writer.writerow(row_data)
                                # ▲▲▲ ここまで修正 ▲▲▲

                            logger.info(
                                f"💾 エントリー時の全特徴量(110個)をCSVに記録しました: {dump_csv_path.name}"
                            )
                        except Exception as e:
                            logger.warning(f"特徴量CSVの保存に失敗: {e}")
                    # ▲▲▲ ここまで追加 ▲▲▲

                    # --- 1. 同時発注禁止 & 両建て防止 (prevent_simultaneous_orders) ---
                    # ※バックテストの default_config.prevent_simultaneous_orders = True に準拠
                    prevent_simultaneous_orders = risk_engine.config.get(
                        "prevent_simultaneous_orders", True
                    )

                    if prevent_simultaneous_orders:
                        # ステップA: ノイズ検知（両方向のシグナルが同時に出た場合は問答無用で両方キャンセル）
                        # ※バックテスト同様、別タイミングでの両建て(ヘッジ)は許可するためステップBは削除
                        if should_trade_long and should_trade_short:
                            logger.warning(
                                f"⚠️ 【防衛線1-A】相場混乱検知: Long/Shortの同時シグナルにつき強制キャンセル。"
                            )
                            should_trade_long = False
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

                    # --- 3. ドローダウン上限チェック (防衛線3) ---
                    current_max_dd = risk_engine.config.get("max_drawdown", 1.0)
                    current_dd = state_manager.current_state.current_drawdown
                    if current_dd >= current_max_dd:
                        logger.warning(
                            f"🛑 【防衛線3】最大ドローダウン超過 ({current_dd:.1%} >= {current_max_dd:.1%})。全エントリーを停止します。"
                        )
                        should_trade_long = False
                        should_trade_short = False

                    # --- 4. 最大ポジション数チェック (防衛線4) ---
                    current_max_pos = risk_engine.config.get("max_positions", 100)
                    current_pos_count = len(state_manager.current_state.trades)
                    if current_pos_count >= current_max_pos:
                        logger.warning(
                            f"🛑 【防衛線4】最大ポジション数到達 ({current_pos_count}/{current_max_pos})。新規エントリーを停止します。"
                        )
                        should_trade_long = False
                        should_trade_short = False

                    # --- 5. 最終的な方向と確率の決定 ---
                    direction = None
                    final_proba = 0.0

                    if should_trade_long:
                        direction = "BUY"
                        final_proba = p_long_m2_raw
                    elif should_trade_short:
                        direction = "SELL"
                        final_proba = p_short_m2_raw
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
                    # ▼追加: M1バーから取得したリアルタイムスプレッド
                    current_spread = new_m1_bar.get("spread", 16.0)

                    # ▼▼▼ 修正: Long/Shortで独立したSL/PT倍率をコンフィグから取得 ▼▼▼
                    if direction == "BUY":
                        current_sl_mult = risk_engine.config.get(
                            "sl_multiplier_long", 5.0
                        )
                        current_tp_mult = risk_engine.config.get(
                            "pt_multiplier_long", 1.0
                        )
                    else:
                        current_sl_mult = risk_engine.config.get(
                            "sl_multiplier_short", 5.0
                        )
                        current_tp_mult = risk_engine.config.get(
                            "pt_multiplier_short", 1.0
                        )

                    command = risk_engine.generate_trade_command(
                        action=direction,  # 'BUY' or 'SELL'
                        p_long=p_long_m2_raw,
                        p_short=p_short_m2_raw,
                        current_price=signal.market_info["current_price"],
                        atr=current_atr,
                        equity=state_manager.current_state.current_equity,
                        sl_multiplier=current_sl_mult,  # 修正反映
                        tp_multiplier=current_tp_mult,  # 修正反映
                        current_spread_pips=current_spread,
                    )

                    # V5エンジンは独立したため、ここでイベントログを記録する
                    # [FIX-8] use_event_sourcing フラグは append_event 内でチェック済みのため直接呼ぶ
                    if command["action"] != "HOLD":
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

        # ▼▼▼ 追加: 終了時に特徴量エンジンのスナップショットを必ず保存する ▼▼▼
        if feature_engine:
            state_file = config.STATE_CHECKPOINT_DIR / "feature_engine_state.pkl"
            feature_engine.save_state(str(state_file))
        # ▲▲▲ ここまで追加 ▲▲▲

        logger.info("MetaTrader5との接続をシャットダウンしました。")
        logger.info("=" * 60)
        logger.info("👋 Project Forge 正常終了")
        logger.info("=" * 60)


if __name__ == "__main__":
    main()
