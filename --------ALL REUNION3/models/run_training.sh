#!/bin/bash
# ============================================================
# Project Forge - Ax → Bx → Cx 連続実行スクリプト
# ※ analyze_importance_orthogonal.py は別途手動実行すること
#
# 使い方:
#   通常実行:  bash /workspace/models/run_training.sh
#   テスト実行: bash /workspace/models/run_training.sh --test
# ============================================================

set -e  # いずれかのステップが失敗したら即停止

MODELS_DIR="/workspace/models"
LOG_DIR="/workspace/logs"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="${LOG_DIR}/run_training_${TIMESTAMP}.log"

mkdir -p "$LOG_DIR"

# LightGBMの[Info]ログを確実に出力させる
# パイプ実行時に抑制されるのを防ぐ
export LIGHTGBM_VERBOSITY=1

# --test フラグの受け取り
TEST_FLAG=""
if [[ "$1" == "--test" ]]; then
    TEST_FLAG="--test"
    echo "⚠️  テストモードで実行します"
fi

# 全出力をターミナルとログファイルの両方に記録
exec > >(tee "$LOG_FILE") 2>&1

echo "============================================================"
echo " Project Forge 学習パイプライン開始"
echo " $(date)"
echo " ログ: $LOG_FILE"
echo "============================================================"

# ============================================================
# Ax: M1 Cross-Validation
# "2" = selected_features_orthogonal_v5 を選択
# ============================================================
echo ""
echo "### [1/3] Ax: M1 Cross-Validation 開始 ###"
echo "2" | python -u "${MODELS_DIR}/model_training_metalabeling_Ax_purified.py" $TEST_FLAG
echo "### [1/3] Ax 完了 ###"

# ============================================================
# Bx: Meta-Labeling
# "y" = Axのrun_configを引き継ぎ
# ============================================================
echo ""
echo "### [2/3] Bx: Meta-Labeling 開始 ###"
echo "y" | python -u "${MODELS_DIR}/model_training_metalabeling_Bx_purified.py" $TEST_FLAG
echo "### [2/3] Bx 完了 ###"

# ============================================================
# Cx: Final Training (M1/M2モデル学習・pkl保存)
# "y" = Bxのrun_configを引き継ぎ
# ============================================================
echo ""
echo "### [3/3] Cx: Final Training 開始 ###"
echo "y" | python -u "${MODELS_DIR}/model_training_metalabeling_Cx_purified.py" $TEST_FLAG
echo "### [3/3] Cx 完了 ###"

echo ""
echo "============================================================"
echo " Ax→Bx→Cx 完了: $(date)"
echo " ログ保存先: $LOG_FILE"
echo ""
echo " 次のステップ:"
echo "   python ${MODELS_DIR}/analyze_importance_orthogonal.py"
echo "============================================================"
