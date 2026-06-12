#!/bin/bash
# ============================================================================
# Phase D-3 完全パイプライン (Step 1-22) — 範囲指定対応版 v2
# ============================================================================
# Step  1-6 : engine 1A-1F (QAState seed artifact 生成)
# Step  7-13: validation 2A-2G (Phase D-3 整合性検証 + 特徴量選別)
#             - Step 7  = 2_A
#             - Step 8  = 2_B
#             - Step 9  = 2_C
#             - Step 10 = 2_D  (実在しない場合は SKIP、ERROR にしない)
#             - Step 11 = 2_E
#             - Step 12 = 2_F
#             - Step 13 = 2_G
# Step 14   : Universal Brain V5 proxy labels (~16h)
# Step 15-17: 月次集約 + concurrency 計算 + uniqueness 重み付け
# Step 18-19: feature list 更新 + M1/M2 直交分割
# Step 20-22: M1 Two-Brain CV (Ax2) → メタラベル生成 (Bx2) → M2 較正 (Cx2)
#
# ログ: /workspace/logs/Layer1_complete/NN_<name>.log
#
# 使い方:
#   全実行 (Step 1-22):
#     bash run_phase_d3_pipeline.sh
#   範囲指定 (例: Step 11-22):
#     bash run_phase_d3_pipeline.sh 11 22
#   単一 step (例: Step 14 だけ):
#     bash run_phase_d3_pipeline.sh 14 14
#
# 範囲外の step は SKIP ログを出して処理スキップ。
# 範囲内でもスクリプトが存在しない step (例: 2_D が無い) は SKIP。
#
# v1 → v2 の変更:
#   - 2_A〜2_G ループを Step↔letter 固定マッピングに変更
#   - スクリプト不在時の挙動を ERROR → SKIP に変更 (パイプライン続行)
#
# 失敗時の挙動:
#   - set -e + pipefail で途中失敗時に即停止
#   - 途中までのログは保持される (再開時はその step から手動実行)
#
# 対話入力:
#   - engine_1_A/B/D/E/F: 1(新規) / 2(手動スレッド)→12 / 1(デフォルト出力) /
#     1(デフォルトメモリ) / 2(本格モード) / 2-15(M0.5〜MN) / y(確認)
#   - engine_1_C: 2(手動スレッド)→12 / 1 / 1 / 2 / 2-15 / y
#   - validation 2A-2G, ラベリング以降: 対話入力なし
# ============================================================================

set -e
set -o pipefail

# ─── 範囲指定の引数 (位置引数 2 つ、省略時は全実行) ───
FROM_STEP=${1:-1}
TO_STEP=${2:-22}

if ! [[ "$FROM_STEP" =~ ^[0-9]+$ ]] || ! [[ "$TO_STEP" =~ ^[0-9]+$ ]]; then
    echo "ERROR: FROM_STEP / TO_STEP は整数で指定してください (例: bash $0 11 22)"
    exit 1
fi
if [ "$FROM_STEP" -gt "$TO_STEP" ]; then
    echo "ERROR: FROM_STEP ($FROM_STEP) > TO_STEP ($TO_STEP)"
    exit 1
fi

LOG_DIR=/workspace/logs/Layer1_complete
mkdir -p "$LOG_DIR"

cd /workspace

# ─── ユーティリティ ───
run_with_stdin() {
    local idx=$1
    local name=$2
    local script=$3
    local stdin_data=$4

    if [ "$idx" -lt "$FROM_STEP" ] || [ "$idx" -gt "$TO_STEP" ]; then
        echo "[Step $idx: $name] SKIP (out of range $FROM_STEP-$TO_STEP)"
        return 0
    fi

    local log_file="${LOG_DIR}/$(printf '%02d' "$idx")_${name}.log"
    local ts_start
    ts_start=$(date +"%Y-%m-%d %H:%M:%S")

    echo ""
    echo "============================================================"
    echo "[$ts_start] Step $idx: $name 開始"
    echo "  script: $script"
    echo "  log:    $log_file"
    echo "============================================================"

    printf '%b' "$stdin_data" | python "$script" 2>&1 | tee "$log_file"

    local ts_end
    ts_end=$(date +"%Y-%m-%d %H:%M:%S")
    echo ""
    echo "[$ts_end] Step $idx: $name 完了"
}

run_no_stdin() {
    local idx=$1
    local name=$2
    local script=$3

    if [ "$idx" -lt "$FROM_STEP" ] || [ "$idx" -gt "$TO_STEP" ]; then
        echo "[Step $idx: $name] SKIP (out of range $FROM_STEP-$TO_STEP)"
        return 0
    fi

    local log_file="${LOG_DIR}/$(printf '%02d' "$idx")_${name}.log"
    local ts_start
    ts_start=$(date +"%Y-%m-%d %H:%M:%S")

    echo ""
    echo "============================================================"
    echo "[$ts_start] Step $idx: $name 開始"
    echo "  script: $script"
    echo "  log:    $log_file"
    echo "============================================================"

    python "$script" 2>&1 | tee "$log_file"

    local ts_end
    ts_end=$(date +"%Y-%m-%d %H:%M:%S")
    echo ""
    echo "[$ts_end] Step $idx: $name 完了"
}

# ─── validation step 専用: スクリプト不在は SKIP ───
run_validation_step() {
    local idx=$1
    local letter=$2

    if [ "$idx" -lt "$FROM_STEP" ] || [ "$idx" -gt "$TO_STEP" ]; then
        echo "[Step $idx: 2_${letter}_*] SKIP (out of range $FROM_STEP-$TO_STEP)"
        return 0
    fi

    local script
    script=$(ls /workspace/validation/2_${letter}_*.py 2>/dev/null | head -1)

    if [ -z "$script" ]; then
        # スクリプトが存在しない (例: 2_D が削除されている) 場合は SKIP
        echo "[Step $idx: 2_${letter}_*] SKIP (script not found in /workspace/validation/)"
        return 0
    fi

    local name
    name=$(basename "$script" .py)
    run_no_stdin "$idx" "$name" "$script"
}

# ─── 対話入力データ ───
ENGINE_AF_INPUT="1\n2\n12\n1\n1\n2\n2-15\ny\n"
ENGINE_C_INPUT="2\n12\n1\n1\n2\n2-15\ny\n"

# ─── パイプライン全体の開始ログ ───
PIPELINE_START=$(date +"%Y-%m-%d %H:%M:%S")
echo ""
echo "############################################################"
echo "Phase D-3 pipeline 開始: $PIPELINE_START"
echo "  range: Step $FROM_STEP 〜 Step $TO_STEP"
echo "  作業ディレクトリ: $(pwd)"
echo "  ログディレクトリ: $LOG_DIR"
echo "############################################################"

# ============================================================================
# Step 1-6: Engine 1A-1F (QAState seed artifact 生成)
# ============================================================================

run_with_stdin 1 "engine_1_A" "features/engine_1_A_a_vast_universe_of_features.py" "$ENGINE_AF_INPUT"
run_with_stdin 2 "engine_1_B" "features/engine_1_B_a_vast_universe_of_features.py" "$ENGINE_AF_INPUT"
run_with_stdin 3 "engine_1_C" "features/engine_1_C_a_vast_universe_of_features.py" "$ENGINE_C_INPUT"
run_with_stdin 4 "engine_1_D" "features/engine_1_D_a_vast_universe_of_features.py" "$ENGINE_AF_INPUT"
run_with_stdin 5 "engine_1_E" "features/engine_1_E_a_vast_universe_of_features.py" "$ENGINE_AF_INPUT"
run_with_stdin 6 "engine_1_F" "features/engine_1_F_a_vast_universe_of_features.py" "$ENGINE_AF_INPUT"

# ─── artifact 生成確認 (Step 1-6 が範囲に含まれた場合のみ) ───
if [ "$FROM_STEP" -le 6 ] && [ "$TO_STEP" -ge 1 ]; then
    echo ""
    echo "============================================================"
    echo "Phase D-3 artifact 生成確認"
    echo "============================================================"
    ARTIFACT_DIR=/workspace/data/XAUUSD/stratum_3_artifacts/qa_states_v5
    if [ -d "$ARTIFACT_DIR" ]; then
        ls -la "$ARTIFACT_DIR"
    else
        echo "WARN: $ARTIFACT_DIR が見つかりません"
    fi
fi

# ============================================================================
# Step 7-13: Validation 2_A 〜 2_G
# ─────────────────────────────────────────────────────────
# [Step 11: 2_E の説明]
# 2_E_hf_meta_model_trainer.py はスクリプト本体でデフォルト
# base_tf = H1 にハードコードされている。
#   - H1 base: entry 60 分間隔 → ラベルウィンドウ重複ゼロ
#   - M3 base: entry 3 分間隔 → ラベルウィンドウ重複 95%
#   - M0.5 base (旧 ASCII sort 既定): 340 万行で OOM
# 出力: survived_hf_features.txt
# ============================================================================

run_validation_step 7  A
run_validation_step 8  B
run_validation_step 9  C
run_validation_step 10 D
run_validation_step 11 E
run_validation_step 12 F
run_validation_step 13 G

# ============================================================================
# Step 14: Universal Brain V5 proxy labels (ラベリング) ~16h
# ============================================================================

resolve_script() {
    local script_name=$1
    for d in /workspace/models /workspace/scripts /workspace; do
        if [ -f "$d/$script_name" ]; then
            echo "$d/$script_name"
            return 0
        fi
    done
    echo "ERROR: $script_name が見つかりません" >&2
    exit 1
}

resolve_script_if_in_range() {
    local idx=$1
    local script_name=$2
    if [ "$idx" -lt "$FROM_STEP" ] || [ "$idx" -gt "$TO_STEP" ]; then
        echo ""
        return 0
    fi
    resolve_script "$script_name"
}

LABELING_SCRIPT=$(resolve_script_if_in_range 14 "create_proxy_labels_polars_patch_regime_Universal_Brain_V5.py")
run_no_stdin 14 "create_proxy_labels_universal_brain_v5" "$LABELING_SCRIPT"

# ============================================================================
# Step 15-22: 後処理 + 直交分割 + M1/M2 メタラベリング学習
# ============================================================================

AGG_SCRIPT=$(resolve_script_if_in_range 15 "aggregate_daily_to_monthly.py")
run_no_stdin 15 "aggregate_daily_to_monthly" "$AGG_SCRIPT"

CONC_SCRIPT=$(resolve_script_if_in_range 16 "sample_uniqueness_weighting_calculate.py")
run_no_stdin 16 "sample_uniqueness_weighting_calculate" "$CONC_SCRIPT"

JOIN_SCRIPT=$(resolve_script_if_in_range 17 "sample_uniqueness_weighting_join.py")
run_no_stdin 17 "sample_uniqueness_weighting_join" "$JOIN_SCRIPT"

UPDATE_SCRIPT=$(resolve_script_if_in_range 18 "update_feature_list_v5.py")
run_no_stdin 18 "update_feature_list_v5" "$UPDATE_SCRIPT"

SPLIT_SCRIPT=$(resolve_script_if_in_range 19 "split_features_first_orthogonal.py")
run_no_stdin 19 "split_features_first_orthogonal" "$SPLIT_SCRIPT"

AX2_SCRIPT=$(resolve_script_if_in_range 20 "model_training_metalabeling_Ax2.py")
run_no_stdin 20 "model_training_metalabeling_Ax2" "$AX2_SCRIPT"

BX2_SCRIPT=$(resolve_script_if_in_range 21 "model_training_metalabeling_Bx2.py")
run_no_stdin 21 "model_training_metalabeling_Bx2" "$BX2_SCRIPT"

CX2_SCRIPT=$(resolve_script_if_in_range 22 "model_training_metalabeling_Cx2.py")
run_no_stdin 22 "model_training_metalabeling_Cx2" "$CX2_SCRIPT"

# ─── 完了 ───
PIPELINE_END=$(date +"%Y-%m-%d %H:%M:%S")
echo ""
echo "############################################################"
echo "Phase D-3 pipeline 完了"
echo "  開始: $PIPELINE_START"
echo "  完了: $PIPELINE_END"
echo "  range: Step $FROM_STEP 〜 Step $TO_STEP"
echo "############################################################"
echo ""
echo "ログ一覧:"
ls -la "$LOG_DIR"
