# /workspace/models/check_calibration_mapping_v5.py
import sys
from pathlib import Path
import joblib
import numpy as np

# プロジェクトルートの設定
project_root = Path(__file__).resolve().parents[1]
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

# Blueprintから1周目のキャリブレーターパスを読み込む
from blueprint import (
    S7_M1_CALIBRATED_LONG,
    S7_M1_CALIBRATED_SHORT,
    S7_M2_CALIBRATED_LONG,
    S7_M2_CALIBRATED_SHORT,
)


def print_mapping(calibrator_path: Path, model_name: str):
    print(f"\n{'=' * 40}")
    print(f" {model_name} Calibration Mapping")
    print(f"{'=' * 40}")

    if not calibrator_path.exists():
        print(f"⚠️ Calibrator not found: {calibrator_path.name}")
        return

    try:
        calibrator = joblib.load(calibrator_path)
        # 0.00 から 1.00 まで 0.05 刻みの生スコア配列
        raw_scores = np.arange(0.0, 1.05, 0.05)
        # 補正
        calibrated_scores = calibrator.predict(raw_scores)

        print(f"{'生スコア (Raw)':<15} | {'補正後 (Calibrated)':<15}")
        print("-" * 35)
        for raw, calib in zip(raw_scores, calibrated_scores):
            # 生スコアと補正後でどれくらい乖離しているかを見やすく
            diff = calib - raw
            diff_str = f"(+{diff:.3f})" if diff > 0 else f"({diff:.3f})"
            if abs(diff) < 0.001:
                diff_str = "( ±0.000)"

            print(f"{raw:<15.2f} | {calib:<15.4f} {diff_str}")

    except Exception as e:
        print(f"Error loading {model_name}: {e}")


if __name__ == "__main__":
    print("### 1周目 (C) 生スコア vs キャリブレーション値 確認 ###")
    print_mapping(S7_M1_CALIBRATED_LONG, "M1 LONG")
    print_mapping(S7_M1_CALIBRATED_SHORT, "M1 SHORT")
    print_mapping(S7_M2_CALIBRATED_LONG, "M2 LONG")
    print_mapping(S7_M2_CALIBRATED_SHORT, "M2 SHORT")
