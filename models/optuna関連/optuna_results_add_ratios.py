"""
add_ratios_to_optuna_results.py

Optunaが出力した6つのCSVファイル（Long3本・Short3本）に
PT%・SL%・TO%の割合カラムを追加して上書き保存するスクリプト。

使い方:
    python add_ratios_to_optuna_results.py
"""

import pandas as pd
from pathlib import Path

# ============================================================
# 対象ファイル一覧
# ============================================================
BASE_DIR = Path("/workspace/data/XAUUSD/stratum_3_artifacts/optuna_results")

INPUT_FILES = [
    BASE_DIR / "optuna_top100_pure_atr_results_spread_0.36.csv",
    BASE_DIR / "optuna_top100_pure_atr_results_spread_0.5.csv",
    BASE_DIR / "optuna_top100_pure_atr_results_spread_0.8.csv",
    BASE_DIR / "optuna_top100_pure_atr_results_short_spread_0.36.csv",
    BASE_DIR / "optuna_top100_pure_atr_results_short_spread_0.5.csv",
    BASE_DIR / "optuna_top100_pure_atr_results_short_spread_0.8.csv",
]

# ============================================================
# メイン処理
# ============================================================
def process_file(path: Path):
    if not path.exists():
        print(f"  [SKIP] ファイルが存在しません: {path.name}")
        return

    df = pd.read_csv(path)

    # 必要カラムの確認
    required = {"Total_Bets", "Wins", "Losses", "Timeouts"}
    if not required.issubset(df.columns):
        print(f"  [SKIP] 必要カラムが不足: {path.name} / 不足: {required - set(df.columns)}")
        return

    # 割合計算（Total_Bets=0の場合はNaN）
    df["PT%"] = (df["Wins"]    / df["Total_Bets"] * 100).round(2)
    df["SL%"] = (df["Losses"]  / df["Total_Bets"] * 100).round(2)
    df["TO%"] = (df["Timeouts"]/ df["Total_Bets"] * 100).round(2)

    df.to_csv(path, index=False)
    print(f"  [OK] {path.name}")


if __name__ == "__main__":
    print("=== Optunaファイルに割合カラムを追加 ===")
    for f in INPUT_FILES:
        process_file(f)
    print("完了")
