# Project Forge - Central Configuration Blueprint (v4)
from pathlib import Path

# =================================================================
# プロジェクト基盤設定
# =================================================================
BASE_DIR = Path("/workspace")
DATA_DIR = BASE_DIR / "data"
LOGS_DIR = BASE_DIR / "logs"
CONFIG_DIR = BASE_DIR / "config"
CORE_DIR = BASE_DIR / "core"          # core_indicators.py 等のコアライブラリ置き場

# 処理対象シンボル（マルチシンボル対応: ここを切り替えるだけ）
SYMBOL = "XAUUSD"  # 例: "USDJPY", "EURUSD" など

# シンボルルートを1箇所で定義（以下の全Stratumがここに依存）
_SYM = DATA_DIR / SYMBOL

# =================================================================
# STRATUM BLUEPRINT
# =================================================================

# --- Stratum 0: 生データ（取り込み前・ブローカー提供ファイル）---
S0_RAW = _SYM / "stratum_0_raw"
S0_RAW_CSV = S0_RAW / "tick_raw_data.csv"  # ブローカーCSV置き場

# --- Stratum 1: 基礎データ ---
S1_BASE = _SYM / "stratum_1_base"
S1_RAW_TICK_PARQUET = S1_BASE / "master_tick_raw.parquet"  # 単一parquet
S1_RAW_TICK_PARTITIONED = S1_BASE / "master_tick_partitioned"  # 年月日Hive
S1_MULTITIMEFRAME = S1_BASE / "master_multitimeframe"  # 15時間足OHLCV
S1_PROCESSED = S1_BASE / "master_processed"  # 特徴量付加・型統一済み

# --- Stratum 2: 特徴量（エンジン出力・検証後）---
S2_FEATURES = _SYM / "stratum_2_features"
S2_FEATURES_AFTER_KS = _SYM / "stratum_2_features_after_ks"  # 廃止（git history 参照）
S2_FEATURES_AFTER_AV = (
    _SYM / "stratum_2_features_after_av"
)  # 廃止（→ S2_FEATURES_VALIDATED に改名）

# --- Stratum 2: 旧定義の改名・追加 ---
# S2_FEATURES_AFTER_KS は廃止（git history 参照）
# S2_FEATURES_AFTER_AV → S2_FEATURES_VALIDATED に改名（同一パスで別名定義）
S2_FEATURES_VALIDATED = _SYM / "stratum_2_features_validated"
# S2_FEATURES_FIXED は廃止（後方互換のため残す場合はコメントアウト）

# --- Chapter 2 バケツ定義 ---
HF_TIMEFRAMES = ["M0.5", "M1", "M3", "M5", "M8", "M15", "M30", "H1"]
LF_SHORT_TIMEFRAMES = ["H4", "H6", "H12"]
LF_MID_TIMEFRAMES = ["D1"]
LF_LONG_TIMEFRAMES = ["W1", "MN"]
LF_ALL_TIMEFRAMES = LF_SHORT_TIMEFRAMES + LF_MID_TIMEFRAMES + LF_LONG_TIMEFRAMES

# --- Chapter 2 WF設定 ---
# グループ別ターゲットshift（M1バー換算）
WF_TARGET_SHIFT = {
    "lf_short": -4800,  # H4×20バー = 80時間
    "lf_mid": -14400,  # D1×10バー = 10日
    "lf_long": -40320,  # W1×4バー  = 4週
}
# グループ別WF訓練・検証月数
WF_CONFIG = {
    "lf_short": {"train_months": 18, "val_months": 3},
    "lf_mid": {"train_months": 24, "val_months": 6},
    "lf_long": {"train_months": 36, "val_months": 12},
}

# --- Chapter 2 統計フィルター閾値 ---
VARIANCE_THRESHOLD = 1e-6
NULL_RATE_THRESHOLD = 0.3  # 欠損率30%以上を除外
CORRELATION_THRESHOLD = 0.95  # 相関0.95以上を双子とみなす
CORR_SAMPLE_SIZE = 200_000  # 相関計算のサンプル数
KS_SAMPLE_SIZE = 100_000  # KS検定のサンプル数

# --- Chapter 2 純化設定 ---
# グループ別：プロキシ時間足とローリングウィンドウ
NEUTRALIZATION_CONFIG = {
    "HF": {"proxy_tf": "M5", "window": 2016},
    "LF_SHORT": {"proxy_tf": "H4", "window": 504},
    "LF_MID": {"proxy_tf": "D1", "window": 90},
    "LF_LONG": {"proxy_tf": "W1", "window": 52},
}

# --- Chapter 2 その他定数 ---
BARRIER_ATR_PERIOD = 13  # トリプルバリアのATR期間
# ※ATRの時間足はシグナル発現した時間足（M1/M3/M5等）に対応したものを使用。固定時間足ではない。
ATR_BASELINE_DAYS = 1  # ATR Ratioのベースライン期間（日数）
# 各時間足のN = timeframe_bars_per_day[tf] * ATR_BASELINE_DAYS
# 例：M1なら1440バー・H1なら24バー・D1なら1バーがベースライン
MAX_WORKERS_DIVISOR = 1  # cpu_count // MAX_WORKERS_DIVISOR でワーカー数決定（旧: 2）

# --- Stratum 3: 検証成果物 ---
S3_ARTIFACTS = _SYM / "stratum_3_artifacts"
S3_STABLE_FEATURE_LIST = S3_ARTIFACTS / "stable_feature_list.joblib"
S3_ADVERSARIAL_SCORES = S3_ARTIFACTS / "adversarial_scores.joblib"
S3_FINAL_FEATURE_TEAM = S3_ARTIFACTS / "final_feature_team.txt"
S3_SHAP_SCORES = S3_ARTIFACTS / "shap_scores.csv"
S3_CONCURRENCY_RESULTS = S3_ARTIFACTS / "concurrency_results.parquet_v2"
S3_SURVIVED_HF_FEATURES = S3_ARTIFACTS / "survived_hf_features.txt"
S3_FEATURES_FOR_ALPHA_DECAY = S3_ARTIFACTS / "final_feature_set.txt"
S3_FEATURES_FOR_TRAINING = S3_ARTIFACTS / "final_feature_set_v3.txt"
S3_FEATURES_FOR_TRAINING_V5 = S3_ARTIFACTS / "final_feature_set_v5.txt"
S3_SELECTED_FEATURES_DIR = S3_ARTIFACTS / "selected_features_v5"
S3_SELECTED_FEATURES_PURIFIED_DIR = S3_ARTIFACTS / "selected_features_purified_v5"
S3_SELECTED_FEATURES_ORTHOGONAL_DIR = S3_ARTIFACTS / "selected_features_orthogonal_v5"  # ★追加: M1/M2直交分割版
S3_FILTERED_LF_FEATURES = S3_ARTIFACTS / "filtered_lf_features.txt"
S3_FILTERED_HF_FEATURES = S3_ARTIFACTS / "filtered_hf_features.txt"
S3_LF_ENVIRONMENT_SCORES = S3_ARTIFACTS / "lf_environment_scores.parquet"

# --- Chapter 3 追加パス ---
# Optunaの最適化結果（spread別・Long/Short別）
S3_OPTUNA_RESULTS_DIR = S3_ARTIFACTS / "optuna_results"

# ラベル生成でATRを自前計算するためのS1_PROCESSEDへの参照
# （e1c_atr_13は相対値のため使用不可・OHLCVからWilder平滑化で計算する）
# S1_PROCESSEDは既に定義済みのため追記不要

# Stratum 4: マスターテーブル (廃止 — git history 参照)

# --- Stratum 5: 純化アルファ ---
S5_ALPHA = _SYM / "stratum_5_alpha"
S5_NEUTRALIZED_ALPHA_SET = S5_ALPHA / "neutralized_alpha_set_partitioned"

# --- Stratum 6: 訓練データ ---
S6_TRAINING = _SYM / "stratum_6_training"
S6_LABELED_DATASET = S6_TRAINING / "labeled_dataset_partitioned_v2"
S6_LABELED_DATASET_MONTHLY = S6_TRAINING / "labeled_dataset_monthly_v2"
S6_WEIGHTED_DATASET = S6_TRAINING / "weighted_dataset_partitioned_v2"

# --- Stratum 7: AIモデル ---
S7_MODELS = _SYM / "stratum_7_models"
# [FIX-INFO-1] 旧仕様の単方向モデルパス定義は廃止 (git history 参照)

# --- Chapter 3 追加パス ---
# バックテスト出力
S7_BACKTEST_RESULTS = S7_MODELS / "backtest_results"
S7_BACKTEST_OPTUNA_RESULTS = S7_MODELS / "backtest_optuna_results"
# キャッシュ格納ディレクトリ
S7_BACKTEST_CACHE_DIR = S7_MODELS / "backtest_preload_cache"
S7_BACKTEST_CACHE_M2  = S7_BACKTEST_CACHE_DIR / "backtest_preload_cache.pkl"    # M2モード用キャッシュ
S7_BACKTEST_CACHE_M1  = S7_BACKTEST_CACHE_DIR / "backtest_preload_cache_M1.pkl" # M1モード用キャッシュ
S7_BACKTEST_CACHE     = S7_BACKTEST_CACHE_M2  # 後方互換エイリアス

# バックテスト結果出力先
S7_BACKTEST_SIM_RESULTS = S7_MODELS / "backtest_simulator_results"

# --- パイプライン実行設定ファイル（Ax→Bx→Cx間の選択引き継ぎ） ---
S7_RUN_CONFIG = S7_MODELS / ".current_run_config.json"


# =====================================================================
# V5 Cimera System (双方向独立モデル)
# =====================================================================

# --- Script A / Ax_purified 出力: M1 OOF予測 ---
S7_M1_OOF_PREDICTIONS_LONG = S7_MODELS / "m1_oof_predictions_long.parquet"
S7_M1_OOF_PREDICTIONS_SHORT = S7_MODELS / "m1_oof_predictions_short.parquet"

# --- Script B / Bx_purified 出力: メタラベル付きOOF ---
S7_META_LABELED_OOF_LONG = S7_MODELS / "meta_labeled_oof_long"
S7_META_LABELED_OOF_SHORT = S7_MODELS / "meta_labeled_oof_short"

# --- Script C / Cx_purified 出力: M2 OOF予測 ---
S7_M2_OOF_PREDICTIONS_LONG = S7_MODELS / "m2_oof_predictions_long.parquet"
S7_M2_OOF_PREDICTIONS_SHORT = S7_MODELS / "m2_oof_predictions_short.parquet"

# --- Script C / Cx_purified 出力: 学習済みモデル ---
S7_M1_MODEL_LONG_PKL = S7_MODELS / "m1_model_v2_long.pkl"
S7_M1_MODEL_SHORT_PKL = S7_MODELS / "m1_model_v2_short.pkl"

S7_M2_MODEL_LONG_PKL = S7_MODELS / "m2_model_v2_long.pkl"
S7_M2_MODEL_SHORT_PKL = S7_MODELS / "m2_model_v2_short.pkl"

# --- Script C / Cx_purified 出力: 較正済み(Calibrated)モデル ---
S7_M1_CALIBRATED_LONG = S7_MODELS / "m1_calibrated_v2_long.pkl"
S7_M1_CALIBRATED_SHORT = S7_MODELS / "m1_calibrated_v2_short.pkl"

S7_M2_CALIBRATED_LONG = S7_MODELS / "m2_calibrated_v2_long.pkl"
S7_M2_CALIBRATED_SHORT = S7_MODELS / "m2_calibrated_v2_short.pkl"

# --- Script C / Cx_purified 出力: パフォーマンスレポート ---
S7_MODEL_PERFORMANCE_REPORT_LONG = S7_MODELS / "model_performance_report_long.json"
S7_MODEL_PERFORMANCE_REPORT_SHORT = S7_MODELS / "model_performance_report_short.json"

# --- 一時ディレクトリ ---
S7_M2_OOF_PREDICTIONS_TMP_LONG = S7_MODELS / "tmp_m2_oof_predictions_long"
S7_M2_OOF_PREDICTIONS_TMP_SHORT = S7_MODELS / "tmp_m2_oof_predictions_short"


# =================================================================
# 実行系ファイルパス
# =================================================================

# --- 状態管理 ---
STATE_DIR = DATA_DIR / "state"
STATE_CHECKPOINT_DIR = STATE_DIR / "checkpoints"
STATE_EVENT_LOG = STATE_DIR / "event_log.jsonl"

# --- ログ ---
LOGS_FORGE_SYSTEM = LOGS_DIR / "forge_system.log"
LOGS_ZMQ_BRIDGE = LOGS_DIR / "zmq_bridge_v2"

# --- 通信フォールバック ---
BRIDGE_FALLBACK_DIR = DATA_DIR / "bridge"
BRIDGE_FALLBACK_FILE = BRIDGE_FALLBACK_DIR / "trade_command.json"

# --- 設定ファイル ---
CONFIG_SYSTEM = CONFIG_DIR / "system_config.json"
CONFIG_RISK = CONFIG_DIR / "risk_config.json"
CONFIG_REGIME = CONFIG_DIR / "regime_config.json"

# =================================================================
# 実行設定 (V11.0: Hybrid ZMQ)
# =================================================================
ZMQ = {
    "control_endpoint": "tcp://host.docker.internal:5555",  # コマンド/ハンドシェイク
    "data_endpoint": "tcp://host.docker.internal:5556",  # バルクデータ転送 (PUSH/PULL)
    "heartbeat_endpoint": "tcp://host.docker.internal:5558",  # ハートビート
    "heartbeat_timeout": 9000,  # ハートビートのタイムアウト(ms)
}
