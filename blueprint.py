# /workspace/config.py
# Project Forge - Central Configuration Blueprint (v3 - Final)
from pathlib import Path

# --- プロジェクトの基本設定 ---
BASE_DIR = Path("/workspace")
DATA_DIR = BASE_DIR / "data"
LOGS_DIR = BASE_DIR / "logs"
CONFIG_DIR = BASE_DIR / "config"
SYMBOL = "XAUUSD"

# --- データリネージュ識別子 ---
# どの特徴量セットを使った分析パイプラインかを定義
FEATURE_SET_ID = "1A_2B"  # 例: Engine 1Aから2Bまでの特徴量を使用した場合

# =================================================================
# データディレクトリ構造定義 (STRATUM BLUEPRINT)
# =================================================================

# --- Stratum 1: 基礎データ ---
S1_BASE = DATA_DIR / SYMBOL / "stratum_1_base"
S1_RAW_TICK_PARQUET = S1_BASE / "master_tick_exness_raw.parquet"
S1_RAW_TICK_PARTITIONED = S1_BASE / "master_tick_partitioned"
S1_BASE_MULTITIMEFRAME = S1_BASE / "master_from_tick"

# --- Stratum 2: 特徴量 ---
S2_FEATURES = DATA_DIR / SYMBOL / "stratum_2_features"
S2_FEATURES_FIXED = DATA_DIR / SYMBOL / "stratum_2_features_fixed"

# KS検定後の安定特徴量セット
S2_FEATURES_AFTER_KS = DATA_DIR / SYMBOL / "stratum_2_features_after_ks"

# AV後の安定特徴量セット
S2_FEATURES_AFTER_AV = DATA_DIR / SYMBOL / "stratum_2_features_after_av"

# --- Stratum 3: 検証成果物 ---
S3_ARTIFACTS = DATA_DIR / SYMBOL / "stratum_3_artifacts" / FEATURE_SET_ID
S3_STABLE_FEATURE_LIST = S3_ARTIFACTS / "stable_feature_list.joblib"
S3_ADVERSARIAL_SCORES = S3_ARTIFACTS / "adversarial_scores.joblib"
S3_FINAL_FEATURE_TEAM = S3_ARTIFACTS / "final_feature_team.txt"
S3_SHAP_SCORES = S3_ARTIFACTS / "shap_scores.csv"
S3_CONCURRENCY_RESULTS = S3_ARTIFACTS / "concurrency_results.parquet_v2"

# --- Stratum 3.5: 事前選抜の成果物 ---
# 第2防衛線の前処理(Phase 0)で生成される軽量な成果物を格納
S3_PRESELECTION = DATA_DIR / SYMBOL / "stratum_3_artifacts" / "phase_0_preselection"
S3_ELITE_LF_FEATURES = S3_PRESELECTION / "elite_lf_features.txt"

# --- Stratum 3.6: 第二防衛網の最終成果物 ---
S3_RUN_ID = "train_12m_val_6m"  # M1/M2検証の実行ID
S3_SURVIVED_HF_FEATURES = S3_ARTIFACTS / S3_RUN_ID / "survived_hf_features.txt"
# [MODIFIED] Chapter2のアルファ純化スクリプトが使用する、サフィックス追加前の特徴量リスト
S3_FEATURES_FOR_ALPHA_DECAY = S3_ARTIFACTS / S3_RUN_ID / "final_feature_set.txt"
# [ADDED] Chapter3のモデル学習スクリプトが使用する、サフィックス追加後の特徴量リスト
S3_FEATURES_FOR_TRAINING = S3_ARTIFACTS / S3_RUN_ID / "final_feature_set_v3.txt"
# ▼▼▼ 新規追加: S3_SELECTED_FEATURES_PURIFIED_DIRの中の４つのファイルテキストの和集合リスト ▼▼▼
S3_FEATURES_FOR_TRAINING_V5 = S3_ARTIFACTS / S3_RUN_ID / "final_feature_set_v5.txt"
# ▼▼▼ V5 双方向ラベリング用: 各モデル専用の厳選特徴量リスト保存ディレクトリ ▼▼▼
S3_SELECTED_FEATURES_DIR = S3_ARTIFACTS / S3_RUN_ID / "selected_features_v5"
# ▼▼▼ 新規追加: 2周目（純化後）のさらに厳選された特徴量リスト保存ディレクトリ ▼▼▼
S3_SELECTED_FEATURES_PURIFIED_DIR = (
    S3_ARTIFACTS / S3_RUN_ID / "selected_features_purified_v5"
)

# Stratum 4: マスターテーブル (廃止 — git history 参照)

# --- Stratum 5: 純化アルファ ---
S5_ALPHA = DATA_DIR / SYMBOL / "stratum_5_alpha" / FEATURE_SET_ID
S5_NEUTRALIZED_ALPHA_SET = S5_ALPHA / "neutralized_alpha_set_partitioned"

# --- Stratum 6: 訓練データ ---
S6_TRAINING = DATA_DIR / SYMBOL / "stratum_6_training" / FEATURE_SET_ID
S6_LABELED_DATASET = S6_TRAINING / "labeled_dataset_partitioned_v2"
S6_LABELED_DATASET_MONTHLY = S6_TRAINING / "labeled_dataset_monthly_v2"
S6_WEIGHTED_DATASET = S6_TRAINING / "weighted_dataset_partitioned_v2"

# --- Stratum 7: AIモデル ---
S7_MODELS = DATA_DIR / SYMBOL / "stratum_7_models" / FEATURE_SET_ID
# [FIX-INFO-1] 旧仕様の単方向モデルパス定義は廃止 (git history 参照)


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
