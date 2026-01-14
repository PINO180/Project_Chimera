# /workspace/main.py
"""
Project Forge 統合実行スクリプト (司令塔) V11.0 (ZMQ ハイブリッド/ゼロ・シリアライズ版)

設計図5章に基づき、全てのコンポーネントを統合し、
リアルタイム取引ループを実行する中央制御システム。

[V11.0 修正内容]
1. MQL5BridgePublisherV3 (V11.0) を採用
2. ZMQエンドポイント設定を Control/Data/Heartbeat の3系統に変更
3. 起動時のデータ取得フローを V11.0 (非同期データポンプ) に対応
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

# ★★★ [ログ抑制] StateManagerの定型ログ(保存成功など)をWARNING以上のみにする ★★★
logging.getLogger("ProjectForge.StateManager").setLevel(logging.WARNING)

# ==================================================================
# 📊 AI判断 可視化用 トップ50特徴量リスト
# ==================================================================
TOP_50_FEATURES = [
    "e1d_hv_standard_50_neutralized_H4",
    "e1e_hilbert_freq_energy_ratio_100_neutralized_H4",
    "e1c_wma_200_neutralized_H6",
    "e1a_statistical_variance_10_neutralized_H1",
    "e1e_signal_peak_to_peak_100_neutralized_H6",
    "e1b_theil_sen_slope_100_neutralized_H6",
    "e1a_fast_volume_mean_50_neutralized_H6",
    "m1_pred_proba",  # (Rank 8)
    "e1f_network_clustering_50_neutralized_H1",
    "e1c_atr_volatility_13_neutralized_H1",
    "e1a_statistical_moment_7_20_neutralized_H6",
    "e1f_tonality_48_neutralized_H12",
    "e1f_network_density_20_neutralized_H1",
    "e1f_harmony_48_neutralized_H1",
    "e1e_signal_rms_50_neutralized_H1",
    "e1c_rvi_signal_10_neutralized_H1",
    "e1b_theil_sen_slope_50_neutralized_H12",
    "e1f_golden_ratio_adherence_55_neutralized_M15",
    "e1f_semantic_flow_25_neutralized_H1",
    "e1e_hilbert_phase_stability_50_neutralized_H1",
    "e1a_statistical_moment_7_50_neutralized_H12",
    "e1e_spectral_energy_64_neutralized_H6",
    "e1b_adf_statistic_100_neutralized_H6",
    "e1c_bb_width_pct_30_2.5_neutralized_M15",
    "e1e_spectral_bandwidth_128_neutralized_H1",
    "e1c_relative_vigor_index_20_neutralized_H1",
    "e1c_relative_vigor_index_10_neutralized_H1",
    "e1e_spectral_energy_128_neutralized_H1",
    "e1e_wavelet_entropy_64_neutralized_H12",
    "e1e_hilbert_phase_var_50_neutralized_H1",
    "e1e_spectral_rolloff_128_neutralized_M15",
    "e1c_di_plus_13_neutralized_H12",
    "e1e_acoustic_power_128_neutralized_H4",
    "e1c_trend_strength_50_neutralized_H1",
    "e1e_hilbert_amp_cv_100_neutralized_H6",
    "e1e_spectral_energy_512_neutralized_H1",
    "e1e_spectral_centroid_128_neutralized_M15",
    "e1e_acoustic_frequency_256_neutralized_H1",
    "e1c_sma_deviation_200_neutralized_M8",
    "e1e_wavelet_mean_256_neutralized_H4",
    "e1c_ema_deviation_50_neutralized_H12",
    "e1e_acoustic_frequency_128_neutralized_H12",
    "log_return_neutralized_M15",
    "e1c_rvi_signal_20_neutralized_M8",
    "e1d_intraday_return_neutralized_H12",
    "e1c_rate_of_change_20_neutralized_H12",
    "e1e_wavelet_mean_128_neutralized_H12",
    "e1c_trix_14_neutralized_H12",
    "e1a_statistical_moment_8_50_neutralized_H12",
    "e1e_wavelet_std_32_neutralized_H12",
]

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
g_context_features: Optional[pl.DataFrame] = None  # D1文脈 (HMM, Hurst等)
g_market_proxy: Optional[pd.DataFrame] = (
    None  # M5リターン (アルファ純化用) [Pandasに事前変換済み]
)

# ==================================================================
# 🚨 リアルタイムデータ取得 (ZMQ経由) 🚨
# ==================================================================


def load_static_data() -> bool:
    """
    [新規]
    起動時に、D1文脈データとM5市場プロキシデータをメモリに一括ロードする。
    (矛盾①, 矛盾③ 解決のため)

    [V6.0 修正] パフォーマンス向上のため、M5プロキシをPandas DFに事前変換してキャッシュ
    [V6.6 修正] 致命的なデータ漏洩バグを修正
    """
    global g_context_features, g_market_proxy
    try:
        # 1. D1文脈データをロード (M2予測用)
        logger.info(f"S7 D1文脈特徴量 {config.S7_CONTEXT_FEATURES} をロード中...")
        g_context_features = pl.read_parquet(config.S7_CONTEXT_FEATURES).sort(
            "timestamp"
        )
        # join_asof のため 'date' キーを 'timestamp' と同じにする
        g_context_features = g_context_features.with_columns(
            pl.col("timestamp").alias("date_key_for_join")
        )
        logger.info(f"✓ D1文脈特徴量をキャッシュ ({len(g_context_features)}行)。")

        # 2. M5市場プロキシデータをロード (アルファ純化用)
        logger.info("S2 M5市場プロキシデータ (アルファ純化用) をロード中...")
        proxy_source_path = (
            config.S2_FEATURES_AFTER_AV
            / "feature_value_a_vast_universeA"
            / f"features_e1a_M5.parquet"
        )
        if not proxy_source_path.exists():
            raise FileNotFoundError(
                f"市場プロキシファイルが見つかりません: {proxy_source_path}"
            )

        # Polarsで計算
        g_market_proxy_pl = (
            pl.read_parquet(proxy_source_path)
            .select(["timestamp", "close"])
            .sort("timestamp")
            .with_columns(
                # --- [V6.6 致命的バグ修正] ---
                # 誤: (pl.col("close").shift(-MARKET_PROXY_LOOKBACK) / pl.col("close") - 1)
                #     -> 将来のリターン (データ漏洩)
                # 正: (pl.col("close").pct_change(MARKET_PROXY_LOOKBACK))
                #     -> 過去のリターン ( (close[t] - close[t-5]) / close[t-5] )
                (pl.col("close").pct_change(MARKET_PROXY_LOOKBACK)).alias(
                    "market_proxy"
                )
                # --- [修正ここまで] ---
            )
            .select(["timestamp", "market_proxy"])
            .drop_nulls()  # 計算不能な先頭のNaNを削除
        )

        # --- [パフォーマンス修正] ---
        # [修正] .tz_localize("UTC") を追加し、TZ-Aware (UTC) に統一する
        g_market_proxy = (
            g_market_proxy_pl.to_pandas().set_index("timestamp").tz_localize("UTC")
        )
        # ------------------------

        logger.info(
            f"✓ M5市場プロキシデータをPandas DFとしてキャッシュ ({len(g_market_proxy)}行)。"
        )
        return True

    except Exception as e:
        logger.critical(
            f"静的データ (文脈/プロキシ) のロードに失敗: {e}", exc_info=True
        )
        return False


def initialize_data_buffer(
    engine: RealtimeFeatureEngine,
    bridge: MQL5BridgePublisherV3,  # [V11.0] V3を使用
    market_proxy_cache: pd.DataFrame,
) -> bool:
    """
    [V11.0 修正]
    M1データのみを可能な限り（5万本）取得し、
    エンジンが M3～MN のすべてをリサンプリングする。
    ゼロ・シリアライズ（V11.0）により、5万本の転送は一瞬で完了する。
    """
    logger.info(
        "ZMQ経由で全時間足の履歴データを取得中 (V11.0: M1 Only / Zero-Serialization)..."
    )

    history_data_map = {}

    # 1. M1データのみをリクエストする
    tf_name = "M1"

    # 2. 必要なM1本数を計算
    # 5万本取得する (V11.0アーキテクチャなら300万本でも可能だが、まずは5万で起動確認)
    lookback = 50000

    logger.info(
        f"  -> {tf_name} の履歴データを {lookback} 本取得中 (全時間足の生成元)..."
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
    (V11.0段階では、履歴転送優先のためプレースホルダー実装の場合がある)
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


def get_correct_d1_context(
    signal_timestamp: datetime,
) -> Dict[str, Any]:
    """
    [新規]
    シグナル発生時刻に基づき、D1文脈キャッシュから正しい日付の文脈を取得する。
    (矛盾③ GIGOの解決)
    """
    global g_context_features
    if g_context_features is None:
        raise ValueError("D1文脈特徴量(g_context_features)がロードされていません。")

    # シグナル発生時刻 (例: 10:15) に対応する D1 の行 (当日 00:00) を探す
    context_row = {}
    try:
        # ( Polars 0.20+ )
        context_row = g_context_features.row(
            by_key=signal_timestamp, key="date_key_for_join", named=True
        )
    except Exception:
        # ( Polars < 0.20 )
        filtered_rows = g_context_features.filter(
            pl.col("date_key_for_join") <= signal_timestamp
        ).tail(1)

        if not filtered_rows.is_empty():
            context_row = filtered_rows.to_dicts()[0]

    if not context_row:
        logger.warning(
            f"シグナル時刻 {signal_timestamp} に対応するD1文脈が見つかりません。"
        )
        return {}

    # 特徴量ではないキーを削除
    context_row.pop("timestamp", None)
    context_row.pop("date_key_for_join", None)
    context_row.pop("returns", None)  # (HMM訓練用であり、M2特徴量ではない)

    return context_row


# ==================================================================
# 📊 [新規] AI判断 マトリックスログ出力 (横型CSV)
# ==================================================================
def save_ai_judgment_matrix(all_feats_dict: Dict[str, float], timestamp: datetime):
    """
    特徴量(A列固定)に対し、新しいシグナルを右列に追加していく形式で保存
    """
    csv_path = config.LOGS_DIR / "ai_judgment_matrix.csv"
    ts_str = timestamp.strftime("%Y-%m-%dT%H:%M:%S")

    try:
        # 1. 既存データの読み込み (なければ作成)
        if csv_path.exists():
            # A列(Feature_Name)をインデックスとして読む
            df = pd.read_csv(csv_path, index_col=0)
        else:
            # 初期化: A列のみ作成
            df = pd.DataFrame(index=TOP_50_FEATURES)
            df.index.name = "Feature_Name"

        # 2. 新しい列データの作成
        new_col_data = []
        for feat in df.index:
            # 辞書から値を取得 (なければNaN)
            val = all_feats_dict.get(feat, np.nan)
            new_col_data.append(val)

        # 3. 新しい列を追加 (列名は日時)
        df[ts_str] = new_col_data

        # 4. 保存
        df.to_csv(csv_path)
        # logger.info(f"AI Matrix Log Updated: {csv_path}")

    except Exception as e:
        logger.error(f"マトリックスログ保存エラー: {e}")


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
        # --- 0. 静的データ (D1文脈, M5プロキシ) の一括ロード ---
        logger.info("--- 0. 静的データ (文脈/プロキシ) のロード ---")
        if not load_static_data():
            raise RuntimeError("静的データのロードに失敗しました。")
        logger.info("✓ 静的データ（D1文脈, M5プロキシ）をキャッシュしました。")

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

        # --- 6. リアルタイム特徴量エンジン (マルチバッファ) の初期化 ---
        logger.info(
            "--- 6. リアルタイム特徴量エンジンの初期化 (ゼロ・シリアライズ) ---"
        )
        feature_engine = RealtimeFeatureEngine(
            feature_list_path=str(config.S3_FEATURES_FOR_TRAINING)
        )
        if not initialize_data_buffer(feature_engine, bridge, g_market_proxy):
            raise RuntimeError("特徴量エンジンのマルチバッファ充填に失敗しました。")

        # --- 7. リアルタイム取引ループ開始 (M1ループ) ---
        logger.info("=" * 60)
        logger.info("🚀 リアルタイム取引ループ開始 (Always Monitor ATR)")
        logger.info("=" * 60)

        # ループ内で制御するロガーを取得（名前を正確に指定）
        logger_bridge = logging.getLogger("execution.mql5_bridge_publisher")
        logger_sm = logging.getLogger("execution.state_manager")

        while True:
            try:
                # (A) 監視: M1の新しいバーの確定を待機
                new_m1_bar = get_latest_m1_bar(bridge)
                if new_m1_bar is None:
                    time.sleep(10)  # (10秒ポーリング)
                    continue

                # --- [修正] ログを抑制してブローカー同期 ---
                # 現在のログレベルを退避
                original_lvl_bridge = logger_bridge.level
                original_lvl_sm = logger_sm.level

                # 一時的にWARNING以上のみ表示（INFO定型文を消す）
                logger_bridge.setLevel(logging.WARNING)
                logger_sm.setLevel(logging.WARNING)

                try:
                    broker_state = bridge.request_broker_state()
                    if broker_state:
                        state_manager.reconcile_with_broker(broker_state)
                finally:
                    # 必ず元のレベルに戻す（重要なトレードログ等は表示させるため）
                    logger_bridge.setLevel(original_lvl_bridge)
                    logger_sm.setLevel(original_lvl_sm)
                # ----------------------------------------

                # (B) 発火判定 & ATR取得
                current_atr, signal_list = feature_engine.process_new_m1_bar(
                    new_m1_bar, g_market_proxy
                )

                threshold = feature_engine.ATR_REGIME_CUTOFF

                # Trigger OFF: 静寂時 (ATR < 5.0)
                if not signal_list:
                    # ATRステータスのみ表示して次へ
                    logger.info(
                        f"💤 Status: M1 ATR={current_atr:.2f} (Threshold: {threshold:.1f}). Waiting..."
                    )
                    continue

                # (C) 審査開始 (Trigger ON: ATR >= 5.0)
                # 詳細ログはCSVに出るので、画面はシンプルに
                logger.info(
                    f"🚀 TRIGGER: ATR={current_atr:.2f} (> {threshold}). Details recorded to ai_judgment_matrix.csv"
                )

                # (D) シグナル処理ループ
                for signal in signal_list:
                    # (D-1) 文脈結合
                    d1_context = get_correct_d1_context(signal.timestamp)

                    if d1_context:
                        h_prob = d1_context.get("hmm_prob_0", 0.0)
                        hurst = d1_context.get("e2a_mfdfa_hurst_mean_1000", 0.0)
                        comp = d1_context.get("e2a_kolmogorov_complexity_1000", 0.0)
                        d1_atr = d1_context.get("atr", 0.0)
                        logger.info(
                            f"🧠 Context: HMM(警戒)={h_prob:.1%}, Hurst={hurst:.3f}, Comp={comp:.3f}, D1_ATR={d1_atr:.2f}"
                        )

                    signal.market_info.update(d1_context)

                    # (E) AI判断
                    command = risk_engine.generate_trade_command(
                        features=signal.features,
                        market_info=signal.market_info,
                        current_time=signal.timestamp,
                    )

                    # ▼▼▼ [マトリックスCSVログ保存 (横型)] ▼▼▼
                    try:
                        all_feats_dict = dict(
                            zip(feature_engine.feature_list, signal.features[0])
                        )
                        all_feats_dict["m1_pred_proba"] = command.get(
                            "confidence_m1", 0.0
                        )
                        all_feats_dict["ai_score"] = command.get("confidence_m2", 0.0)

                        save_ai_judgment_matrix(all_feats_dict, signal.timestamp)
                    except Exception as e_log:
                        logger.error(f"ログ保存失敗: {e_log}")
                    # ▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲

                    # (F) 実行
                    if command["action"] != "HOLD":
                        logger.info(
                            f"-> ⚡ ACTION: {command['action']} {command['lots']} lots (Reason: {command.get('reason', 'AI Signal')})"
                        )
                        success = bridge.send_trade_command(command)
                        if success:
                            logger.info("✓ コマンド送信成功 (ACK受信)")
                            time.sleep(5.0)

                            # 発注直後の同期は重要なのでログ抑制しない
                            broker_state_after = bridge.request_broker_state()
                            if broker_state_after:
                                state_manager.reconcile_with_broker(broker_state_after)
                        else:
                            logger.error("✗ コマンド送信失敗 (NACKまたはタイムアウト)")
                    else:
                        logger.info(
                            f"-> 🛡️ JUDGMENT: HOLD (Reason: {command['reason']})"
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
