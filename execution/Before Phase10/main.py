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
    # "H1": "H1",
    # "H4": "H4",
    # "H6": "H6",
    # "H12": "H12",
    # "D1": "D1",
    # "W1": "W1",
    # "MN": "MN",
}

# --- グローバルキャッシュ ---
g_last_processed_bar_time: Optional[int] = (
    None  # ループの最終処理時刻（廃止予定・後方互換のため残す）
)
g_market_proxy: Optional[pd.DataFrame] = (
    None  # M5リターン (アルファ純化用) [Pandasに事前変換済み]
)
# ==================================================================
# 🚨 Project Cimera (V5) 戦略パラメータ 🚨
# ==================================================================
risk_engine = ExtremeRiskEngineV5(config_path=str(config.CONFIG_RISK))
# 【未使用グローバル変数・参照禁止】
# 以下の値はループ内で risk_engine.config.get() により都度読み直されるため
# この変数は実際の発注判定では使われていない。
# ホットリロード（risk_config.jsonの動的反映）に対応するための設計上、
# グローバルへのキャッシュは意図的に使用しない。
M2_PROBA_THRESHOLD = risk_engine.config.get("m2_proba_threshold", 0.70)
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


def _run_gap_fill(
    feature_engine: "RealtimeFeatureEngine",
    bridge: "MQL5BridgePublisherV3",
    market_proxy: pd.DataFrame,
    last_processed_bar_time: int,
) -> int:
    """
    [GAP-FILL] バッファの欠落期間を M0.5 バーで補填する共通関数。

    フルウォームアップ後・スナップショット復帰後・EA再起動後・
    市場閉鎖後・メインループ整合性チェックからの全シナリオで呼ばれる。

    Returns:
        更新後の g_last_processed_bar_time (int, unixtime秒)
        充填バーが0本の場合は last_processed_bar_time をそのまま返す。
    """
    now_ts = int(datetime.now(timezone.utc).timestamp())
    diff_minutes = int((now_ts - last_processed_bar_time) / 60)

    if diff_minutes <= 0:
        return last_processed_bar_time

    logger.info(
        f"[GAP-FILL] ギャップ {diff_minutes} 分を M0.5 で穴埋めします..."
    )

    diff_df = bridge.request_historical_data(
        symbol=STRATEGY_SYMBOL,
        timeframe_name="M0.5",
        lookback_bars=diff_minutes * 2 + 60,
    )

    if diff_df is None or len(diff_df) == 0:
        logger.warning("[GAP-FILL] 差分M0.5データの取得に失敗。ギャップ未充填のまま続行します。")
        return last_processed_bar_time

    new_m05_bars = diff_df[
        diff_df["timestamp"]
        > datetime.fromtimestamp(last_processed_bar_time, timezone.utc)
    ]

    if new_m05_bars.empty:
        logger.info("[GAP-FILL] 穴埋め対象バーなし（既に最新）。")
        return last_processed_bar_time

    # [V=0 GUARD] 学習側 s1_1_B_build_ohlcv.py の filter(tick_count > 0) と
    # 完全整合させるため、gap-fill 経路でも V=0 ghost bar を除外する。
    # EA 側 CollectM05Bar の new-bucket 分岐に volume>0 ガードが欠落していた
    # ため、ProcessHistoryRequest 経由で V=0 stub が混入していた。
    # M0.5 buffer 起点でフィルタすることで、後続の M3 リサンプル close 汚染
    # → TP_REVERSED_BY_LAG / Execution Failed の連鎖を根本から断つ。
    if "volume" in new_m05_bars.columns:
        _n_before = len(new_m05_bars)
        new_m05_bars = new_m05_bars[new_m05_bars["volume"] > 0]
        _n_after = len(new_m05_bars)
        if _n_before != _n_after:
            logger.info(
                f"[V=0 GUARD] gap-fill から V=0 ghost {_n_before - _n_after} 本を除外 "
                f"(残: {_n_after} / 元: {_n_before} 本)"
            )

    if new_m05_bars.empty:
        logger.info("[GAP-FILL] V=0 ghost 除外後、穴埋め対象バーなし。")
        return last_processed_bar_time

    # [DISC-FLAG] disc をバッチ計算してから流し込む。
    # bar_dict.get("disc", False) のデフォルト False では
    # 市場停止・週末跨ぎを含む gap で disc が誤って False になる。
    gap_fill_indexed = new_m05_bars.set_index("timestamp").sort_index()

    if len(feature_engine.m05_dataframe) > 0:
        _last_m05_ts = feature_engine.m05_dataframe[-1]["timestamp"]
        _anchor = pd.DataFrame(
            index=pd.DatetimeIndex([_last_m05_ts], name="timestamp")
        )
        _combined = pd.concat([_anchor, gap_fill_indexed])
        _combined = feature_engine._add_disc_column(_combined, freq_seconds=30)
        gap_fill_with_disc = _combined.iloc[1:]  # アンカー除去
    else:
        gap_fill_with_disc = feature_engine._add_disc_column(
            gap_fill_indexed, freq_seconds=30
        )

    _new_last_ts = last_processed_bar_time
    for _ts, _row in gap_fill_with_disc.iterrows():
        _bar_dict = {
            "timestamp": _ts,
            "open": _row["open"],
            "high": _row["high"],
            "low": _row["low"],
            "close": _row["close"],
            "volume": float(_row["volume"]),
            "spread": 36.0,
            "disc": bool(_row["disc"]),
        }
        # market_proxy をインクリメンタルに更新
        if len(feature_engine.m05_dataframe) >= 20:
            _recent = pd.DataFrame(
                list(feature_engine.m05_dataframe)[-20:]
            ).set_index("timestamp")
            _m5 = (
                _recent["close"]
                .resample("5min", closed="left", label="left")
                .last()
                .dropna()
            )
            if len(_m5) >= 2:
                _new_proxy_val = (
                    float(_m5.iloc[-1]) - float(_m5.iloc[-2])
                ) / (float(_m5.iloc[-2]) + 1e-10)
                _new_proxy_df = pd.DataFrame(
                    {"market_proxy": [_new_proxy_val]},
                    index=pd.DatetimeIndex([_m5.index[-1]], tz="UTC"),
                )
                if market_proxy.empty:
                    market_proxy = _new_proxy_df
                else:
                    if _m5.index[-1] not in market_proxy.index:
                        market_proxy = pd.concat([market_proxy, _new_proxy_df])

        feature_engine.process_new_m05_bar(
            _bar_dict, market_proxy, warmup_only=True
        )
        _new_last_ts = int(_ts.timestamp())

    logger.info(
        f"✓ [GAP-FILL] {len(new_m05_bars)} 本 (M0.5) の穴埋め完了。"
    )
    return _new_last_ts


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
        "ZMQ経由で全時間足の履歴データを取得中 (V12.0: M0.5 起点 / Zero-Serialization)..."
    )

    history_data_map = {}

    # 1. M0.5（30秒足）データをリクエストする
    # [V12.0] M0.5が最細粒度のためここを起点とし、Python側でM1以上を全てリサンプリングする
    tf_name = "M0.5"

    # 2. 必要なM0.5本数を計算 (動的算出)
    # エンジンの要求する最大ルックバック期間を M0.5 本数に換算する
    max_m05_bars_needed = 0

    # エンジンの定数定義を利用して上位足の必要期間をM0.5換算する
    for tf_key, lookback_val in engine.lookbacks_by_tf.items():
        # 分数換算 (ALL_TIMEFRAMESの値を使用)
        minutes_per_bar = engine.ALL_TIMEFRAMES.get(tf_key)
        if minutes_per_bar is None:
            continue  # tickなどはスキップ

        # 必要なM0.5本数 = 上位足のLookback * その足の分数 * 2（M0.5はM1の2倍の本数）
        total_m05_needed = lookback_val * minutes_per_bar * 2

        if total_m05_needed > max_m05_bars_needed:
            max_m05_bars_needed = total_m05_needed

    # M0.5換算800,000本を最低ラインに設定（稼働率約71%を考慮してH1=4,132本をカバー）
    lookback = max(800000, max_m05_bars_needed + 10000)

    logger.info(
        f"  -> {tf_name} の履歴データを {lookback} 本取得中 (Engine要求: {max_m05_bars_needed} M0.5 bars)..."
    )

    # ✨ [V12.0] ZMQ リクエスト (PULLによるストリーミング受信)
    df_rates_m05 = bridge.request_historical_data(
        symbol=STRATEGY_SYMBOL,
        timeframe_name=tf_name,
        lookback_bars=lookback,
    )

    if df_rates_m05 is None or len(df_rates_m05) == 0:
        logger.error(
            f"ZMQ {tf_name} 履歴データの取得に失敗しました。起動を中止します。"
        )
        return False

    # [DIAG] 受信した生データの先頭・末尾タイムスタンプをログ出力
    # → EA の CopyTicksRange + g_m05_bars 合成結果の実態を確認するための診断ログ
    # → 末尾が現在時刻付近なら合成正常、数十分〜数時間前なら Tick 履歴欠落または合成バグ
    logger.info(
        f"[DIAG] M0.5 受信データ範囲: "
        f"{df_rates_m05['timestamp'].iloc[0]} 〜 {df_rates_m05['timestamp'].iloc[-1]} "
        f"({len(df_rates_m05)} 本)"
    )

    # ▼▼▼ M0.5履歴データから市場プロキシ(M5)を動的生成 ▼▼▼
    global g_market_proxy
    logger.info("  -> 取得したM0.5履歴データから初期の市場プロキシ(M5)を動的生成中...")
    try:
        temp_df = df_rates_m05[["timestamp", "close"]].copy()
        temp_df.set_index("timestamp", inplace=True)

        # [乖離④修正] 学習側(2_G_alpha_neutralizer)と計算方式を統一:
        #   学習側: close / close.shift(1) - 1  ← M5の1バー前比リターン
        #   旧本番: m5_close.pct_change(5)  ← 5バー前比リターン（OLSのX分布が異なっていた）
        #   新本番: m5_close.pct_change(1)  ← 1バー前比リターン（学習側と一致）
        #
        # [乖離⑥修正 V2] M5バー境界を学習側 s1_1_B_build_ohlcv.py と完全一致させる:
        #   学習側 (s1_1_B_build_ohlcv.py L152-153):
        #     bucket_size_ns = pd.to_timedelta(freq).total_seconds() * 1e9
        #     ddf["group_key"] = (ddf_int_ts // bucket_size_ns) * bucket_size_ns
        #     → 整数除算によるfloor → label="left", closed="left" と完全等価
        #     → tick 12:30:00〜12:34:59 → group 12:30
        #     → tick 12:35:00〜12:39:59 → group 12:35
        #
        #   旧本番側: closed="right", label="right"
        #     → tick 12:30:00.001〜12:35:00 → group 12:35  (12:35:00 ちょうども含む)
        #     → market_proxyのX変数値が学習側と完全に違う
        #     → OLS純化のbeta/alphaが歪み、純化済み特徴量も全部歪む
        #
        #   新本番側: closed="left", label="left" に統一
        m5_close = (
            temp_df["close"]
            .resample("5min", label="left", closed="left")
            .last()
            .dropna()
        )
        proxy_df = (
            m5_close.pct_change(1)
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

    # 3. history_data_map には M0.5 を格納（fill_all_buffersがここを起点にリサンプリング）
    history_data_map[tf_name] = df_rates_m05

    # エンジンに全履歴データを一括で渡す
    engine.fill_all_buffers(history_data_map, g_market_proxy)

    # M3イベント駆動ループの開始時刻をセット
    global g_last_processed_bar_time
    if len(history_data_map["M0.5"]) > 0:
        g_last_processed_bar_time = int(
            history_data_map["M0.5"]["timestamp"].iloc[-1].timestamp()
        )
        logger.info(
            f"M0.5バー最終処理時刻: {datetime.fromtimestamp(g_last_processed_bar_time, timezone.utc)}"
        )
    else:
        logger.warning("M0.5データが空のため、最終処理時刻を現在時刻に設定します。")
        g_last_processed_bar_time = int(time.time())

    return True


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
                    "Delta",
                    "Signal",  # "LONG" / "SHORT" / "NONE"
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
            # [発見#B対応] 学習側 model_training_metalabeling_Cx2._load_features の
            # exclude_exact と完全に同一の除外セットを適用することで、将来の特徴量
            # 名簿の編集で誤ってメタデータ列が混入した場合でも、本番側が学習側と
            # 同じ列順序で LightGBM に入力ベクトルを渡せるようにする堅牢化。
            #
            # 注: m1_pred_proba は除外対象に含まれる。Bx2 がjoin時に追加するカラムであり
            # 特徴量名簿には書かれないが、もし誤って書かれていたとしても以下の
            # 「m1_pred_proba を末尾に強制配置」ロジックで適切に末尾に再配置される。
            _FEATURE_EXCLUDE_EXACT = {
                "timestamp",
                "timeframe",
                "t1",
                "label",
                "label_long",
                "label_short",
                "uniqueness",
                "uniqueness_long",
                "uniqueness_short",
                "payoff_ratio",
                "payoff_ratio_long",
                "payoff_ratio_short",
                "pt_multiplier",
                "sl_multiplier",
                "direction",
                "exit_type",
                "first_ex_reason_int",
                "atr_value",
                "calculated_body_ratio",
                "fallback_vol",
                "open",
                "high",
                "low",
                "close",
                "meta_label",
                "m1_pred_proba",
                "is_trigger",
            }

            def load_feature_list(filepath: Path) -> list:
                """学習側 Cx2._load_features と同等のフィルタリング。

                除外対象:
                  - exclude_exact セット (timestamp/label_*/atr_value/open/high/low/close 等)
                  - is_trigger プレフィックスを持つ列 (is_trigger_on_M1 等)
                """
                with open(filepath, "r") as f:
                    raw = [line.strip() for line in f if line.strip()]
                cleaned = []
                dropped: list[str] = []
                for col in raw:
                    if col in _FEATURE_EXCLUDE_EXACT:
                        dropped.append(col)
                        continue
                    if col.startswith("is_trigger"):
                        dropped.append(col)
                        continue
                    cleaned.append(col)
                if dropped:
                    logger.warning(
                        f"⚠️ {filepath.name}: 学習側で除外される列が混入していたため除外: {dropped}"
                    )
                return cleaned

            feature_lists = {
                "long_m1": load_feature_list(
                    config.S3_SELECTED_FEATURES_ORTHOGONAL_DIR / "m1_long_features.txt"
                ),
                "long_m2": load_feature_list(
                    config.S3_SELECTED_FEATURES_ORTHOGONAL_DIR / "m2_long_features.txt"
                ),
                "short_m1": load_feature_list(
                    config.S3_SELECTED_FEATURES_ORTHOGONAL_DIR / "m1_short_features.txt"
                ),
                "short_m2": load_feature_list(
                    config.S3_SELECTED_FEATURES_ORTHOGONAL_DIR / "m2_short_features.txt"
                ),
            }

            # m1_pred_probaをM2特徴量リストの末尾に強制配置
            # Cx2・analyze_importance_orthogonal.pyと統一（ファイルの状態に依存しない堅牢な設計）
            for key in ["long_m2", "short_m2"]:
                fl = feature_lists[key]
                if "m1_pred_proba" in fl:
                    fl.remove("m1_pred_proba")
                fl.append("m1_pred_proba")

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

        # [FIX] orthogonal 4ファイルのユニーク和集合を特徴量名簿として渡す。
        # 旧: S3_FEATURES_FOR_TRAINING_V5 (final_feature_set_v5.txt / 1714件・D1/H4/W1/MN含む)
        #   → D1・H4・W1・MNバッファが誤生成され ZMQ を長時間占有するボトルネックの原因だった。
        # 新: orthogonal_v5 の 4ファイル和集合 (1465件・HF時間足のみ) を一時ファイルに書き出して渡す。
        import tempfile, itertools

        _orth_dir = config.S3_SELECTED_FEATURES_ORTHOGONAL_DIR
        _orth_files = [
            _orth_dir / "m1_long_features.txt",
            _orth_dir / "m1_short_features.txt",
            _orth_dir / "m2_long_features.txt",
            _orth_dir / "m2_short_features.txt",
        ]
        _union: list[str] = []
        _seen: set[str] = set()
        for _fp in _orth_files:
            for _line in open(_fp):
                _name = _line.strip()
                if _name and _name not in _seen:
                    _seen.add(_name)
                    _union.append(_name)
        _tmp_feature_list = tempfile.NamedTemporaryFile(
            mode="w", suffix=".txt", delete=False, encoding="utf-8"
        )
        _tmp_feature_list.write("\n".join(_union))
        _tmp_feature_list.close()
        logger.info(
            f"[FIX] 特徴量名簿を orthogonal 4ファイル和集合 ({len(_union)}件) に差し替えました。"
            f" -> {_tmp_feature_list.name}"
        )

        feature_engine = RealtimeFeatureEngine(feature_list_path=_tmp_feature_list.name)

        # ▼▼▼ 修正: スナップショットからの爆速復帰と差分取得 ▼▼▼
        state_file = config.STATE_CHECKPOINT_DIR / "feature_engine_state.pkl"
        is_warmed_up = False

        if state_file.exists():
            try:
                # 1. とりあえずロードを試みる
                if feature_engine.load_state(str(state_file)):
                    logger.info("⚡ スナップショットからの爆速復帰に成功しました！")
                    if len(feature_engine.m05_dataframe) > 0:
                        global g_last_processed_bar_time
                        g_last_processed_bar_time = int(
                            feature_engine.m05_dataframe[-1]["timestamp"].timestamp()
                        )

                        # [GAP-FILL] スナップショット復帰後の差分追いつき
                        g_last_processed_bar_time = _run_gap_fill(
                            feature_engine, bridge, g_market_proxy, g_last_processed_bar_time
                        )

                        # 穴埋め完了後に最新状態をスナップショット保存
                        feature_engine.save_state(str(state_file))
                        logger.info("💾 追いつき後の最新状態をスナップショットに保存しました。")

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

            # [GAP-FILL] フルウォームアップ後のバッファギャップ充填
            # fill_all_buffers は T0 時点のデータでバッファを初期化するが、
            # T0〜T2（ウォームアップ処理時間）の間に確定したリアルタイムバーが存在しない。
            g_last_processed_bar_time = _run_gap_fill(
                feature_engine, bridge, g_market_proxy, g_last_processed_bar_time
            )

            # フルウォームアップ完了状態をスナップショットに保存（次回起動を高速化）
            feature_engine.save_state(str(state_file))
            logger.info(
                "💾 [GAP-FILL] フルウォームアップ後の状態をスナップショットに保存しました。"
            )

        # ─────────────────────────────────────────────────────────────────
        # [Phase 9b 案 V] 診断 L2 (smoke test) を起動経路の合流地点で実行
        # ─────────────────────────────────────────────────────────────────
        # 旧: `fill_all_buffers()` 内で呼んでいた → フルウォームアップ時しか
        #     実行されず、スナップショット爆速復帰時は smoke test が走らない
        #     → 案 W の verbose ログが得られず、実機の死蔵バグ調査に支障
        # 新: 両ルート (フルウォームアップ / スナップショット復帰) の合流点で
        #     1 回だけ呼ぶ形に統一。起動経路によらず必ず実行される
        feature_engine.run_smoke_test()

        # [STALE-GUARD] 両ルート（スナップショット復帰・フルウォームアップ）共通:
        # ウォームアップ・差分追いつきが完全に完了した時点でis_python_readyフラグをセット。
        # 以降のHeartbeat送信（最大heartbeat_interval秒）でEAにPING:READYが届き、
        # EA側のg_python_readyが自動的にtrueになる。
        # EA再起動・瞬断後も次のHeartbeatで自動同期されるため一発コマンドへの依存がない。
        logger.info("[STALE-GUARD] Python準備完了フラグをセット。次のHeartbeatでEAのM3通知が解禁されます。")
        bridge.notify_python_ready()

        # --- 7. リアルタイム取引ループ開始 (M3イベント駆動ループ) ---
        logger.info("=" * 60)
        logger.info(f"🚀 リアルタイム取引ループ開始 ")
        logger.info("=" * 60)

        # ▼追加: ホットリロード用のタイムスタンプ監視
        last_config_mtime = os.path.getmtime(config.CONFIG_RISK)
        # ▼追加: 定期セーブ用のタイマー
        last_snapshot_time = time.time()
        # ▼追加: BUFFER-INTEGRITY チェック用タイマー
        last_buffer_check_time = time.time()
        last_gap_fill_time = 0.0
        _BUFFER_CHECK_INTERVAL_SEC = 30.0
        _GAP_FILL_COOLDOWN_SEC = 30.0
        # [LAG-FIX] 重い ZMQ リクエスト (broker_state, recent_history) の間引き用タイマー
        # 旧: 毎ループ (~100ms 周期) 実行 → poll_m3_bar の応答性を阻害してラグ増大
        # 新: M3 未確定中は 1 秒に 1 回だけ実行。M3 確定時は通常通り実行。
        # 1秒間隔: 口座残高/ポジション同期の鮮度を確保しつつ、EA OnTimer 50ms に対しては
        #          20回に1回の REQ なので負荷増加は無視できる。
        last_periodic_sync_time = 0.0
        _PERIODIC_SYNC_INTERVAL_SEC = 1.0
        # ▼追加: M3通知受信時刻トラッカー
        last_m3_received_time = time.time()
        # ▼追加: 起動後安定化クールダウン（防衛的シグナルスキップ）
        # バッファ充填直後・EA再起動後・市場閉鎖明けはOLS状態が不完全な場合があるため
        # 最初のN回のM3確定シグナルをスキップして安定化を待つ。
        # スキップ中もprocess_new_m05_barは正常実行（バッファ・OLS更新は継続）。
        # M3確定 = last_bar_timestamps["M3"] が変化した瞬間をカウント。
        #   起動時（フルウォームアップ/スナップショット復帰）: 1回
        #   EA再起動（needs_notify）: 2回
        #   BUFFER-INTEGRITY gap-fill（市場閉鎖・月曜再開等）: 1回
        _skip_signals_count = 1
        logger.info(f"[COOLDOWN] 起動後安定化クールダウン設定: {_skip_signals_count} M3確定スキップ")

        while True:
            try:
                # M3確定通知をポーリング（最大1秒待機）
                new_m05_bar = bridge.poll_m3_bar(timeout_ms=100)

                # 毎秒実行される定期処理
                current_time_sec = time.time()

                # [STALE-GUARD] EA再起動検知 → notify_python_ready()再送 + gap-fill
                # Heartbeatスレッドがタイムアウトを検知するとneeds_notifyをセットする。
                # メインループがここで検知してnotify_python_ready()（needs_notifyクリア）を呼ぶ。
                # これによりHeartbeatが次のサイクルでPING:READYを送信しEA側のM3通知が解禁される。
                if bridge.needs_notify.is_set():
                    logger.warning(
                        "⚠️ [STALE-GUARD] EA再起動を検知。notify_python_ready()を再送します..."
                    )
                    bridge.notify_python_ready()
                    # [GAP-FILL] EA停止中のバッファ欠落を補填
                    g_last_processed_bar_time = _run_gap_fill(
                        feature_engine, bridge, g_market_proxy, g_last_processed_bar_time
                    )
                    # [COOLDOWN] EA再起動後はOLS欠落が大きいため2M3確定スキップ
                    _skip_signals_count = max(_skip_signals_count, 2)
                    logger.info(f"[COOLDOWN] EA再起動検知: {_skip_signals_count} M3確定スキップに設定")

                # [BUFFER-INTEGRITY] メインループ整合性チェック & 自動 gap-fill
                # 「最後にM3通知を受け取った時刻」で判定する。
                # last_bar_timestamps["M0.5"]はEAが常に30〜60秒前のバーを送るため
                # 構造的にノイズが大きく、正常運転中でも閾値を超えてしまう。
                # M3通知受信時刻ベースなら「M3通知が途絶えた = EA停止/市場閉鎖」を
                # ノイズゼロで検知できる。閾値は M3 間隔 × 2（360秒）。
                if current_time_sec - last_buffer_check_time >= _BUFFER_CHECK_INTERVAL_SEC:
                    last_buffer_check_time = current_time_sec
                    _m3_silence_sec = current_time_sec - last_m3_received_time
                    if (
                        _m3_silence_sec > 360  # M3 間隔(180s) × 2
                        and current_time_sec - last_gap_fill_time >= _GAP_FILL_COOLDOWN_SEC
                    ):
                        logger.warning(
                            f"⚠️ [BUFFER-INTEGRITY] M3通知が {_m3_silence_sec:.0f}秒 途絶えています。"
                            f"自動 gap-fill を発動します..."
                        )
                        g_last_processed_bar_time = _run_gap_fill(
                            feature_engine, bridge, g_market_proxy, g_last_processed_bar_time
                        )
                        last_gap_fill_time = current_time_sec
                        # [COOLDOWN] 市場閉鎖・月曜再開等の復帰後は1M3確定スキップ
                        _skip_signals_count = max(_skip_signals_count, 1)
                        logger.info(f"[COOLDOWN] BUFFER-INTEGRITY復帰: {_skip_signals_count} M3確定スキップに設定")

                # 15分間隔でスナップショットを強制保存
                if current_time_sec - last_snapshot_time > 900:
                    if feature_engine:
                        state_file = (
                            config.STATE_CHECKPOINT_DIR / "feature_engine_state.pkl"
                        )
                        feature_engine.save_state(str(state_file))
                        logger.info(
                            "💾 [定期保存] 特徴量エンジンの状態をスナップショットに保存しました。"
                        )
                    last_snapshot_time = current_time_sec

                # 設定ファイル(risk_config.json)の更新日時をチェック
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
                # [LAG-FIX] M3 未確定中は 1 秒間隔に間引く (口座残高鮮度確保)
                # M3 確定時 (new_m05_bar is not None) は必ず実行する。
                # [LAG-FIX-4] M3 境界 (180秒) 前後 3 秒間は重い処理をスキップ。
                # 旧: 1秒間隔の sync が M3 close 直前に発火して REQ-REP × 2 で
                #     M3 通知の受信が最大 215ms 遅延する問題があった
                #     (実機ログで 11:51:00.176 通知 → 11:51:00.391 受信を確認)
                # 新: M3 close 前後 3 秒間 (= 全体の 3.3%) は sync を見送る
                #     代わりに M3 通知が来た時 (new_m05_bar is not None) に実行する
                # ==========================================================
                _now_in_m3_cycle = current_time_sec % 180.0
                _near_m3_boundary = (_now_in_m3_cycle > 177.0) or (_now_in_m3_cycle < 3.0)

                _do_periodic_sync = (
                    new_m05_bar is not None
                    or (
                        not _near_m3_boundary
                        and (current_time_sec - last_periodic_sync_time) >= _PERIODIC_SYNC_INTERVAL_SEC
                    )
                )
                if _do_periodic_sync:
                    last_periodic_sync_time = current_time_sec
                    current_time = datetime.now(timezone.utc)
                    if state_manager and state_manager.current_state:
                        for trade in state_manager.current_state.trades:
                            duration_mins = trade.get_duration_minutes(current_time)
                            is_timeout = False
                            td_long = risk_engine.config.get("td_minutes_long", 30.0)
                            td_short = risk_engine.config.get("td_minutes_short", 30.0)
                            if trade.direction == "BUY" and duration_mins >= td_long:
                                is_timeout = True
                            elif trade.direction == "SELL" and duration_mins >= td_short:
                                is_timeout = True
                            if is_timeout:
                                logger.info(
                                    f"⏱️ タイムアウト(TO)条件到達。強制決済を実行: Ticket={trade.ticket}, Direction={trade.direction}, Duration={duration_mins:.1f}分"
                                )
                                success = bridge.send_trade_command(
                                    {
                                        "action": "CLOSE",
                                        "ticket": trade.ticket,
                                        "magic": 77777,
                                    }
                                )
                                if success:
                                    event_data = {
                                        "ticket": trade.ticket,
                                        "close_reason": "TO",
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
                                else:
                                    logger.warning(
                                        f"⚠️ タイムアウト決済コマンドの送信に失敗しました。次ループで再試行します: Ticket={trade.ticket}"
                                    )

                    # ==========================================================
                    # 【シミュレーター完全同期 2】 サイレント・クローズ(SL/PT)の確実な捕捉
                    # [BUGFIX] state_manager.current_state が None のとき (起動直後等) に
                    #          .trades アクセスで AttributeError が出る問題を修正。
                    # ==========================================================
                    if (
                        hasattr(bridge, "request_recent_history")
                        and state_manager
                        and state_manager.current_state
                    ):
                        recent_history = bridge.request_recent_history()
                        if recent_history:
                            for closed_pos in recent_history:
                                ticket = closed_pos.get("ticket")
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
                                        "close_reason": reason,
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

                    # ブローカーと最終同期
                    broker_state_sync = bridge.request_broker_state()
                    if broker_state_sync:
                        state_manager.reconcile_with_broker(broker_state_sync)

                # M3未確定の場合はここで次のループへ
                if new_m05_bar is None:
                    continue

                # [STALE-GUARD] ウォームアップ中にキューに溜まった古いバーを破棄する。
                # ウォームアップ（JIT+OLS初期化）に最大30分かかる場合がある。
                # そのままシグナル処理すると同一秒に大量のオーダーが一斉発射される事故につながる。
                # M0.5(30秒)足なので、120秒より古ければ確実に「溜まりもの」と判断して読み飛ばす。
                _bar_ts = new_m05_bar.get("timestamp")
                if _bar_ts is not None:
                    _now_utc = datetime.now(timezone.utc)
                    if _bar_ts.tzinfo is None:
                        _bar_ts = _bar_ts.replace(tzinfo=timezone.utc)
                    _age_sec = (_now_utc - _bar_ts).total_seconds()
                    if _age_sec > 120:
                        logger.warning(
                            f"⚠️ [STALE-GUARD] 古いM0.5バーを破棄 "
                            f"(age={_age_sec:.0f}秒 / ts={_bar_ts})"
                        )
                        continue

                # [BUFFER-INTEGRITY] 有効なM3通知を受信した時刻を記録
                last_m3_received_time = current_time_sec

                # 市場プロキシの更新（M5の1バー前比リターン、学習側2Gと完全一致）
                # m05_dataframeをM5にリサンプリングしてM5クローズ2本を取得
                # M5が確定するのは5分に1回だが、ffillで参照するため問題なし
                # [乖離⑥修正 V2] 学習側 s1_1_B_build_ohlcv.py の整数除算と一致させる:
                #   学習側: (ts_int // bucket_size_ns) * bucket_size_ns でfloor集約
                #   学習側相当: closed="left", label="left"
                if feature_engine and len(feature_engine.m05_dataframe) >= 20:
                    _recent = pd.DataFrame(
                        list(feature_engine.m05_dataframe)[-20:]
                    ).set_index("timestamp")
                    _m5 = (
                        _recent["close"]
                        .resample("5min", closed="left", label="left")
                        .last()
                        .dropna()
                    )
                    if len(_m5) >= 2:
                        new_proxy_val = (
                            float(_m5.iloc[-1]) - float(_m5.iloc[-2])
                        ) / (float(_m5.iloc[-2]) + 1e-10)
                        new_proxy_df = pd.DataFrame(
                            {"market_proxy": [new_proxy_val]},
                            index=pd.DatetimeIndex(
                                [_m5.index[-1]], tz="UTC"
                            ),
                        )
                        if g_market_proxy.empty:
                            g_market_proxy = new_proxy_df
                        else:
                            # 同一タイムスタンプの重複を避けて追記
                            if _m5.index[-1] not in g_market_proxy.index:
                                g_market_proxy = pd.concat([g_market_proxy, new_proxy_df])
                        if len(g_market_proxy) > 10000:
                            g_market_proxy = g_market_proxy.iloc[-5000:]

                # M0.5バーをエンジンに渡し、シグナルを待つ
                _m3_ts_before = feature_engine.last_bar_timestamps.get("M3")
                signal_list = feature_engine.process_new_m05_bar(
                    new_m05_bar, g_market_proxy
                )
                _m3_ts_after = feature_engine.last_bar_timestamps.get("M3")

                # [COOLDOWN] M3確定を検知してスキップカウントをデクリメント
                # M3確定 = last_bar_timestamps["M3"] が変化した瞬間
                if _skip_signals_count > 0 and _m3_ts_after != _m3_ts_before:
                    _skip_signals_count -= 1
                    logger.info(
                        f"[COOLDOWN] 安定化クールダウン中のためシグナルをスキップ"
                        f"（残り {_skip_signals_count} M3確定待ち）"
                    )
                    continue  # バッファ・OLS更新は完了済み、シグナル処理のみスキップ

                if not signal_list:
                    continue  # シグナルなし

                # シグナル処理ループ
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
                        [[feature_dict.get(f, 0.0) or 0.0 for f in feature_lists["long_m1"]]]
                    )
                    p_long_m1_raw = models["long_m1"].predict(X_long_m1)[0]

                    # 2. M2モデル (M1が0.50以上の場合のみ推論、それ以外は強制0.0)
                    if p_long_m1_raw >= 0.50:
                        feature_dict_long = feature_dict.copy()
                        # ★ Logit変換: 学習時（Bx）と同じ変換を本番推論時にも適用
                        _p_l_clipped = np.clip(p_long_m1_raw, 1e-7, 1 - 1e-7)
                        _logit_long = float(
                            np.clip(
                                np.log(_p_l_clipped / (1 - _p_l_clipped)), -10.0, 10.0
                            )
                        )
                        feature_dict_long["m1_pred_proba"] = _logit_long
                        X_long_m2 = np.array(
                            [
                                [
                                    feature_dict_long.get(f, 0.0) or 0.0
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
                        [[feature_dict.get(f, 0.0) or 0.0 for f in feature_lists["short_m1"]]]
                    )
                    p_short_m1_raw = models["short_m1"].predict(X_short_m1)[0]

                    # 2. M2モデル (M1が0.50以上の場合のみ推論、それ以外は強制0.0)
                    if p_short_m1_raw >= 0.50:
                        feature_dict_short = feature_dict.copy()
                        # ★ Logit変換: 学習時（Bx）と同じ変換を本番推論時にも適用
                        _p_s_clipped = np.clip(p_short_m1_raw, 1e-7, 1 - 1e-7)
                        _logit_short = float(
                            np.clip(
                                np.log(_p_s_clipped / (1 - _p_s_clipped)), -10.0, 10.0
                            )
                        )
                        feature_dict_short["m1_pred_proba"] = _logit_short
                        X_short_m2 = np.array(
                            [
                                [
                                    feature_dict_short.get(f, 0.0) or 0.0
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

                    # ▼▼▼ ツール①: 特徴量スナップショット保存 ▼▼▼
                    try:
                        from feature_snapshot_tool import save_feature_snapshot
                        save_feature_snapshot(
                            feature_dict=feature_dict,
                            p_m1_long=p_long_m1_raw,
                            p_m1_short=p_short_m1_raw,
                            p_m2_long=p_long_m2_raw,
                            p_m2_short=p_short_m2_raw,
                            atr_ratio=signal.market_info.get("atr_ratio", 0.0),
                        )
                    except Exception as _snap_e:
                        logger.debug(f"スナップショット保存スキップ: {_snap_e}")

                    # ▼▼▼ 修正: Delta (差分) フィルター & 判定理由の明記 ▼▼▼
                    current_m2_thresh = risk_engine.config.get(
                        "m2_proba_threshold", 0.70
                    )  # バックテストと統一
                    current_m2_delta = risk_engine.config.get(
                        "m2_delta_threshold", 0.30
                    )  # バックテストと統一

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

                    # ▼▼▼ Delta・Signal確定後にCSV記録 ▼▼▼
                    try:
                        signal_str = (
                            "LONG"
                            if should_trade_long
                            else ("SHORT" if should_trade_short else "NONE")
                        )
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
                                    round(delta, 4),
                                    signal_str,
                                ]
                            )
                    except Exception as e:
                        logger.warning(f"CSVへの書き込みに失敗しました: {e}")
                    # ▲▲▲ ここまで追加 ▲▲▲

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
                                f"💾 エントリー時の全特徴量({len(feature_keys)}個)をCSVに記録しました: {dump_csv_path.name}"
                            )
                        except Exception as e:
                            logger.warning(f"特徴量CSVの保存に失敗: {e}")
                    # ▲▲▲ ここまで追加 ▲▲▲

                    # # --- 1. 同時発注禁止 & 両建て防止 (prevent_simultaneous_orders) ---
                    # # ※バックテストの default_config.prevent_simultaneous_orders = True に準拠
                    # prevent_simultaneous_orders = risk_engine.config.get(
                    #     "prevent_simultaneous_orders", True
                    # )

                    # if prevent_simultaneous_orders:
                    #     # ステップA: ノイズ検知（両方向のシグナルが同時に出た場合は問答無用で両方キャンセル）
                    #     # ※バックテスト同様、別タイミングでの両建て(ヘッジ)は許可するためステップBは削除
                    #     if should_trade_long and should_trade_short:
                    #         logger.warning(
                    #             f"⚠️ 【防衛線1-A】相場混乱検知: Long/Shortの同時シグナルにつき強制キャンセル。"
                    #         )
                    #         should_trade_long = False
                    #         should_trade_short = False

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
                    current_max_dd = risk_engine.config.get("max_drawdown", 20.0)
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
                        "atr_value", signal.market_info.get("atr", 0.0)
                    )
                    # ★追加確認: atr_ratioもmarket_infoから取得してrisk_engineに渡す
                    current_atr_ratio = signal.market_info.get("atr_ratio", 0.0)
                    # ▼追加: M0.5バーから取得したリアルタイムスプレッド
                    current_spread = new_m05_bar.get("spread", 999.0)

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
                        atr_ratio=current_atr_ratio,  # ★追加: ATR Ratio をリスクエンジンに渡す
                    )

                    # [DEBUG-BARRIER] 発注直前の診断ログ — バリア幅異常の原因特定用
                    # extreme_risk_engine.py が生成した sl_width/tp_width を直接読む（責務分離）。
                    # HOLDコマンドの場合はこれらキーが無いためフォールバックを用意。
                    _entry_price = signal.market_info["current_price"]
                    _sl_dollar   = float(command.get("sl_width", 0.0))
                    _tp_dollar   = float(command.get("tp_width", 0.0))
                    _buf_len     = signal.market_info.get("atr_buffer_len", -1)
                    _last_tr     = signal.market_info.get("last_tr", -1.0)
                    if command["action"] != "HOLD":
                        logger.info(
                            f"[DEBUG-BARRIER] direction={direction}"
                            f" | entry={_entry_price:.3f}"
                            f" | ATR={current_atr:.4f}"
                            f" | last_TR={_last_tr:.4f}"
                            f" | buf_len={_buf_len}"
                            f" | SL幅=${_sl_dollar:.3f}"
                            f" | TP幅=${_tp_dollar:.3f}"
                            f" | SL_mult={current_sl_mult}"
                            f" | TP_mult={current_tp_mult}"
                        )

                    # [BARRIER-GUARD] 最低ドル幅フィルター
                    # 閑散相場でATRが極端に小さくなりスプレッド負けするのを防ぐ。
                    # SLまたはTPのいずれかが閾値を下回る場合はHOLDに強制変換する。
                    _min_barrier = risk_engine.config.get("min_barrier_dollar_width", 1.5)
                    if command["action"] != "HOLD" and (
                        _sl_dollar < _min_barrier or _tp_dollar < _min_barrier
                    ):
                        logger.warning(
                            f"⚠️ [BARRIER-GUARD] バリア幅が最低閾値(${_min_barrier})未満のためHOLDに変換。"
                            f" SL幅=${_sl_dollar:.3f} / TP幅=${_tp_dollar:.3f}"
                        )
                        command["action"] = "HOLD"
                        command["reason"] = (
                            f"BARRIER-GUARD: SL=${_sl_dollar:.3f} TP=${_tp_dollar:.3f}"
                            f" < min=${_min_barrier}"
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
                            f"-> 発注コマンドを送信: {command['action']} {command['lots']} lots"
                            f" (SL:{command.get('stop_loss')}, TP:{command.get('take_profit')})"
                            f" [Spread: {current_spread:.1f}pips]"
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
