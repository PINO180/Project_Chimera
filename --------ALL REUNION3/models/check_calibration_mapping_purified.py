# /workspace/models/check_calibration_mapping_purified.py
import sys
from pathlib import Path
import joblib
import numpy as np

project_root = Path(__file__).resolve().parents[1]
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

# Blueprintのディレクトリベースを取得
from blueprint import S7_MODELS

# ※Cx(purified)で保存されたキャリブレーターのパスを直接指定
# (Cxのコード上の実際の出力パスに合わせて調整してください)
PURIFIED_M1_LONG = S7_MODELS / "m1_calibrated_long_purified.pkl"
PURIFIED_M1_SHORT = S7_MODELS / "m1_calibrated_short_purified.pkl"
PURIFIED_M2_LONG = S7_MODELS / "m2_calibrated_long_purified.pkl"
PURIFIED_M2_SHORT = S7_MODELS / "m2_calibrated_short_purified.pkl"


def print_mapping(calibrator_path: Path, model_name: str):
    print(f"\n{'=' * 40}")
    print(f" {model_name} Calibration Mapping")
    print(f"{'=' * 40}")

    if not calibrator_path.exists():
        print(f"⚠️ Calibrator not found: {calibrator_path.name}")
        return

    try:
        calibrator = joblib.load(calibrator_path)
        raw_scores = np.arange(0.0, 1.05, 0.05)
        calibrated_scores = calibrator.predict(raw_scores)

        print(f"{'生スコア (Raw)':<15} | {'補正後 (Calibrated)':<15}")
        print("-" * 35)
        for raw, calib in zip(raw_scores, calibrated_scores):
            diff = calib - raw
            diff_str = f"(+{diff:.3f})" if diff > 0 else f"({diff:.3f})"
            if abs(diff) < 0.001:
                diff_str = "( ±0.000)"

            print(f"{raw:<15.2f} | {calib:<15.4f} {diff_str}")

    except Exception as e:
        print(f"Error loading {model_name}: {e}")


if __name__ == "__main__":
    print("### 2周目 (Cx Purified) 生スコア vs キャリブレーション値 確認 ###")
    print_mapping(PURIFIED_M1_LONG, "M1 LONG (Purified)")
    print_mapping(PURIFIED_M1_SHORT, "M1 SHORT (Purified)")
    print_mapping(PURIFIED_M2_LONG, "M2 LONG (Purified)")
    print_mapping(PURIFIED_M2_SHORT, "M2 SHORT (Purified)")
