"""
filter_optuna_results.py

Optunaが出力した6つのCSVファイルに対して以下の条件でフィルタリングし、
新規ファイルとして保存するスクリプト。

フィルタ条件:
    PT% > 30
    SL% < 30
    TO% < 60

※ PT%/SL%/TO%カラムが存在しない場合はadd_ratios_to_optuna_results.pyを先に実行すること。

使い方:
    python filter_optuna_results.py
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
# フィルタ条件
# ============================================================
PT_MIN    = 30     # PT% > 30
SL_MAX    = 30     # SL% < 30
TO_MAX    = 60     # TO% < 60
BETS_MIN  = 10000  # Total_Bets >= 10000
LOSS_MIN  = 500    # Losses > 500

# ============================================================
# メイン処理
# ============================================================
def process_file(path: Path):
    if not path.exists():
        print(f"  [SKIP] ファイルが存在しません: {path.name}")
        return

    df = pd.read_csv(path)

    # 割合カラムの確認
    required = {"PT%", "SL%", "TO%"}
    if not required.issubset(df.columns):
        print(f"  [SKIP] PT%/SL%/TO%カラムがありません。先にadd_ratios_to_optuna_results.pyを実行してください: {path.name}")
        return

    before = len(df)

    # フィルタ適用
    df_filtered = df[
        (df["PT%"] > PT_MIN) &
        (df["SL%"] < SL_MAX) &
        (df["TO%"] < TO_MAX) &
        (df["Total_Bets"] >= BETS_MIN) &
        (df["Losses"] > LOSS_MIN)
    ].copy()

    after = len(df_filtered)

    # 出力ファイル名（元ファイル名に_filteredを追加）
    output_path = path.parent / path.name.replace("top100", "top100_filtered")
    df_filtered.to_csv(output_path, index=False)

    print(f"  [OK] {path.name}")
    print(f"       {before}件 → {after}件（削除: {before - after}件）")
    print(f"       出力: {output_path.name}")


if __name__ == "__main__":
    print(f"=== Optunaフィルタリング (PT%>{PT_MIN}, SL%<{SL_MAX}, TO%<{TO_MAX}, Bets>={BETS_MIN}, Losses>{LOSS_MIN}) ===")
    for f in INPUT_FILES:
        process_file(f)
    print("完了")
