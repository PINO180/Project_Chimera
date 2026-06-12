#!/bin/bash
# ============================================================================
# run_ols_diagnosis_pipeline.sh — OLS 真因調査用パイプライン (範囲指定対応版)
# ============================================================================
# 5/2 〜 5/26 の学習側 S1-S6 + 推論を延長計算する専用版。
# run_phase_d3_pipeline.sh と同じ書き方 (resolve_script / run_validation_step)。
#
# Step  1   : s1_1_A_ingest
# Step  2   : s1_1_B_build_ohlcv
# Step  3   : s1_1_C_enrich
# Step  4-9 : engine 1A-1F (QAState seed artifact 利用)
# Step 10   : 2_A_skip_ks_stability_filter
# Step 11   : 2_G_alpha_neutralizer (= 真因調査の核、 純化後データを生成)
# Step 12   : create_proxy_labels_polars_patch_regime_Universal_Brain_V5
# Step 13   : aggregate_daily_to_monthly
# Step 14   : sample_uniqueness_weighting_calculate
# Step 15   : sample_uniqueness_weighting_join
# Step 16   : infer_period (既存モデルで OOS 推論)
#
# ログ: /workspace/logs/OLS_diagnosis/NN_<name>.log
#
# 使い方:
#   全実行 (Step 1-16):
#     bash run_ols_diagnosis_pipeline.sh
#   範囲指定 (例: Step 4-16):
#     bash run_ols_diagnosis_pipeline.sh 4 16
#   2_G 以降のみ (Step 11-16):
#     bash run_ols_diagnosis_pipeline.sh 11 16
#
# 範囲外の step は SKIP ログを出して処理スキップ。
#
# 失敗時の挙動:
#   - set -e + pipefail で途中失敗時に即停止
#   - 途中までのログは保持される (再開時はその step から手動実行)
#
# 対話入力:
#   - engine_1_A/B/D/E/F: 1(新規) / 2(手動スレッド)→12 / 1(デフォルト出力) /
#     1(デフォルトメモリ) / 2(本格モード) / 2-15(M0.5〜MN) / y(確認)
#   - engine_1_C: 2(手動スレッド)→12 / 1 / 1 / 2 / 2-15 / y
#   - その他: 対話入力なし
# ============================================================================

set -e
set -o pipefail

# ─── 範囲指定の引数 (位置引数 2 つ、省略時は全実行) ───
FROM_STEP=${1:-1}
TO_STEP=${2:-16}

if ! [[ "$FROM_STEP" =~ ^[0-9]+$ ]] || ! [[ "$TO_STEP" =~ ^[0-9]+$ ]]; then
    echo "ERROR: FROM_STEP / TO_STEP は整数で指定してください (例: bash $0 4 16)"
    exit 1
fi
if [ "$FROM_STEP" -gt "$TO_STEP" ]; then
    echo "ERROR: FROM_STEP ($FROM_STEP) > TO_STEP ($TO_STEP)"
    exit 1
fi

LOG_DIR=/workspace/logs/OLS_diagnosis
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
        echo "[Step $idx: 2_${letter}_*] SKIP (script not found in /workspace/validation/)"
        return 0
    fi

    local name
    name=$(basename "$script" .py)
    run_no_stdin "$idx" "$name" "$script"
}

# ─── スクリプト解決 ───
resolve_script() {
    local script_name=$1
    for d in /workspace/pipeline /workspace/models /workspace/scripts /workspace; do
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

# ─── 対話入力データ ───
ENGINE_AF_INPUT="1\n2\n12\n1\n1\n2\n2-15\ny\n"
ENGINE_C_INPUT="2\n12\n1\n1\n2\n2-15\ny\n"

# ─── パイプライン全体の開始ログ ───
PIPELINE_START=$(date +"%Y-%m-%d %H:%M:%S")
echo ""
echo "############################################################"
echo "OLS Diagnosis pipeline 開始: $PIPELINE_START"
echo "  range: Step $FROM_STEP 〜 Step $TO_STEP"
echo "  作業ディレクトリ: $(pwd)"
echo "  ログディレクトリ: $LOG_DIR"
echo "############################################################"

# ============================================================================
# Step 1-3: s1_1_X (raw tick → multitimeframe → enriched)
# ============================================================================

S1A=$(resolve_script_if_in_range 1 "s1_1_A_ingest.py")
run_no_stdin 1 "s1_1_A_ingest" "$S1A"

S1B=$(resolve_script_if_in_range 2 "s1_1_B_build_ohlcv.py")
run_no_stdin 2 "s1_1_B_build_ohlcv" "$S1B"

S1C=$(resolve_script_if_in_range 3 "s1_1_C_enrich.py")
run_no_stdin 3 "s1_1_C_enrich" "$S1C"

# ============================================================================
# Step 4-9: Engine 1A-1F (QAState seed artifact 利用、 純化前特徴量を生成)
# ============================================================================

run_with_stdin 4 "engine_1_A" "features/engine_1_A_a_vast_universe_of_features.py" "$ENGINE_AF_INPUT"
run_with_stdin 5 "engine_1_B" "features/engine_1_B_a_vast_universe_of_features.py" "$ENGINE_AF_INPUT"
run_with_stdin 6 "engine_1_C" "features/engine_1_C_a_vast_universe_of_features.py" "$ENGINE_C_INPUT"
run_with_stdin 7 "engine_1_D" "features/engine_1_D_a_vast_universe_of_features.py" "$ENGINE_AF_INPUT"
run_with_stdin 8 "engine_1_E" "features/engine_1_E_a_vast_universe_of_features.py" "$ENGINE_AF_INPUT"
run_with_stdin 9 "engine_1_F" "features/engine_1_F_a_vast_universe_of_features.py" "$ENGINE_AF_INPUT"

# ─── artifact 生成確認 (Step 1-9 が範囲に含まれた場合のみ) ───
if [ "$FROM_STEP" -le 9 ] && [ "$TO_STEP" -ge 4 ]; then
    echo ""
    echo "============================================================"
    echo "Engine artifact 生成確認"
    echo "============================================================"
    ARTIFACT_DIR=/workspace/data/XAUUSD/stratum_3_artifacts/qa_states_v5
    if [ -d "$ARTIFACT_DIR" ]; then
        ls -la "$ARTIFACT_DIR"
    else
        echo "WARN: $ARTIFACT_DIR が見つかりません"
    fi
fi

# ============================================================================
# Step 10-11: Validation 2_A, 2_G (KS-skip フィルタ + OLS 純化)
# ============================================================================

run_validation_step 10 A
run_validation_step 11 G

# ============================================================================
# Step 12-16: create_proxy_labels → aggregate → uniqueness → infer_period
# ============================================================================

LBL=$(resolve_script_if_in_range 12 "create_proxy_labels_polars_patch_regime_Universal_Brain_V5.py")
run_no_stdin 12 "create_proxy_labels_universal_brain_v5" "$LBL"

AGG=$(resolve_script_if_in_range 13 "aggregate_daily_to_monthly.py")
run_no_stdin 13 "aggregate_daily_to_monthly" "$AGG"

UWC=$(resolve_script_if_in_range 14 "sample_uniqueness_weighting_calculate.py")
run_no_stdin 14 "sample_uniqueness_weighting_calculate" "$UWC"

UWJ=$(resolve_script_if_in_range 15 "sample_uniqueness_weighting_join.py")
run_no_stdin 15 "sample_uniqueness_weighting_join" "$UWJ"

INFER=$(resolve_script_if_in_range 16 "infer_period.py")
run_no_stdin 16 "infer_period" "$INFER"

# ─── 完了 ───
PIPELINE_END=$(date +"%Y-%m-%d %H:%M:%S")
echo ""
echo "############################################################"
echo "OLS Diagnosis pipeline 完了"
echo "  開始: $PIPELINE_START"
echo "  完了: $PIPELINE_END"
echo "  range: Step $FROM_STEP 〜 Step $TO_STEP"
echo "############################################################"
echo ""
echo "ログ一覧:"
ls -la "$LOG_DIR"
