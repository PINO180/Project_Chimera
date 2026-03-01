# /workspace/models/grid_search_optimizer.py

import sys
import subprocess
import logging
from pathlib import Path
import itertools
import polars as pl
import numpy as np
import csv

# --- プロジェクトのルートディレクトリをPythonの検索パスに追加 ---
project_root = Path(__file__).resolve().parents[1]
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

# --- Blueprintからパス設定をインポート ---
from blueprint import S6_LABELED_DATASET

# --- ロギング設定 ---
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

# ====================================================================
# --- ▼▼▼ 実験設定 (ここを編集してください) ▼▼▼ ---
# ====================================================================

# [1] 実行するラベリングスクリプト名
SCRIPT_TO_RUN = "/workspace/models/create_proxy_labels_polars_patch_for_grid.py"

# [2] 実験対象とする年月（この月のデータのみ処理）
TEST_YEAR_MONTH = "2025/9"  # 2024 -> 2025/9 に変更

# [3] レジームの定義 (atr_min <= atr_value < atr_max)
#    (check_atr.py の統計に基づき設定)
REGIMES = {
    "R1_LowVol": {"min": 0.23, "max": 1.0},
    "R2_MidVol": {"min": 1.0, "max": 2.5},
    "R3_HighVol": {"min": 2.5, "max": 5.0},
    "R4_SuperVol": {"min": 5.0, "max": None},  # ★★★ 修正: 15.0 -> None (上限なし)
}

# [4] 各レジームで試行するパラメータのグリッド
#    (★ ユーザー分析に基づき、1:1のペイオフレシオと、より長いTDを追加 ★)
PARAMETER_GRID = {
    # R1
    "R1_LowVol": {
        "pt_mult": [0.9, 2.0, 4.0, 10.0],
        "sl_mult": [0.9, 1.5, 2.0, 10.0],
        "duration": ["120m", "1200m"],
    },
    # R2
    "R2_MidVol": {
        "pt_mult": [0.2, 1.5, 4.0, 10.0],
        "sl_mult": [0.2, 1.0, 1.5, 10.0],
        "duration": ["60m", "600m"],
    },
    # R3
    "R3_HighVol": {
        "pt_mult": [0.1, 1.0, 2.5, 5.0, 10.0],
        "sl_mult": [0.1, 0.5, 1.0, 5.0, 10.0],
        "duration": ["15m", "180m"],
    },
    # R4
    "R4_SuperVol": {
        "pt_mult": [0.05, 1.0, 2.0, 5.0, 10.0],
        "sl_mult": [0.05, 0.5, 1.0, 5.0, 10.0],
        "duration": ["5m", "120m"],
    },
}

# [5] テストモード (Trueにすると、各レジーム1パターンのみ実行して動作確認)
TEST_MODE = False

# [6] 自動保存するCSVファイルパス
RESULTS_CSV_PATH = Path(
    "/workspace/data/XAUUSD/stratum_7_models/1A_2B/grid_search_results.csv"
)

# ====================================================================
# --- ▲▲▲ 実験設定はここまで ▲▲▲ ---
# ====================================================================


def run_labeling(regime: dict, pt: float, sl: float, td: str) -> Path:
    """
    改造した create_proxy_labels.py をサブプロセスとして実行する
    """
    script_path = Path(SCRIPT_TO_RUN)
    if not script_path.exists():
        logging.error(f"FATAL: Script not found at {SCRIPT_TO_RUN}")
        raise FileNotFoundError(f"Script not found at {SCRIPT_TO_RUN}")

    # S6_LABELED_DATASET のパス (blueprintから)
    output_dir = S6_LABELED_DATASET

    # --- ▼▼▼ 修正: 年月指定で実行 + atr_maxがNoneのケースに対応 ▼▼▼ ---
    try:
        year, month = TEST_YEAR_MONTH.split("/")
    except ValueError:
        logging.error(
            f"Invalid TEST_YEAR_MONTH format: {TEST_YEAR_MONTH}. Expected YYYY/M"
        )
        return None

    cmd = [
        "python",
        str(script_path),
        "--filter-mode=month",
        f"--year-month={TEST_YEAR_MONTH}",
        f"--atr-min={regime['min']}",
        # atr_max は None の場合、引数自体を追加しない
        f"--pt-mult={pt}",
        f"--sl-mult={sl}",
        f"--duration={td}",
        "--no-resume",  # 常にクリーンな状態で実行
    ]

    # atr_max が None でない場合のみ、引数を追加
    if regime["max"] is not None:
        cmd.append(f"--atr-max={regime['max']}")
    # --- ▲▲▲ 修正ここまで ▲▲▲ ---

    logging.info("=" * 60)
    logging.info(f"Running: {' '.join(cmd)}")

    try:
        # サブプロセスを実行
        subprocess.run(cmd, check=True, capture_output=True, text=True, timeout=600)
        logging.info("Labeling process finished successfully.")
        return output_dir

    except subprocess.CalledProcessError as e:
        # (修正: regime_name はこのスコープにないため、より汎用的なログに変更)
        logging.error(
            f"Labeling process FAILED for atr_min={regime['min']} (PT={pt}, SL={sl})"
        )
        logging.error(f"Return Code: {e.returncode}")
        logging.error(f"STDOUT: {e.stdout}")
        logging.error(f"STDERR: {e.stderr}")
        return None
    except subprocess.TimeoutExpired as e:
        # (修正: regime_name はこのスコープにないため、より汎用的なログに変更)
        logging.error(
            f"Labeling process TIMED OUT (10 min) for atr_min={regime['min']} (PT={pt}, SL={sl})"
        )
        logging.error(f"STDERR: {e.stderr}")
        return None


def analyze_results(output_dir: Path) -> dict:
    """
    ラベリング結果（S6）を読み込み、パフォーマンスを集計する
    """
    if output_dir is None:
        return {"wins": 0, "losses": 0, "timeouts": 0, "avg_payoff": 0, "pf": 0}

    # --- ▼▼▼ 修正: 年月指定でパスを構築 ▼▼▼ ---
    try:
        year, month = TEST_YEAR_MONTH.split("/")
        # TEST_YEAR_MONTH のパーティションを glob で検索
        partition_path = output_dir / f"year={year}/month={month}/**/*.parquet"
    except ValueError:
        logging.error(
            f"Invalid TEST_YEAR_MONTH format: {TEST_YEAR_MONTH}. Expected YYYY/M"
        )
        return {"wins": 0, "losses": 0, "timeouts": 0, "avg_payoff": 0, "pf": 0}
    # --- ▲▲▲ 修正ここまで ▲▲▲ ---

    try:
        df = pl.scan_parquet(str(partition_path)).collect(streaming=True)

        if df.is_empty():
            logging.warning("Analysis skipped: No labeled data found for this run.")
            return {"wins": 0, "losses": 0, "timeouts": 0, "avg_payoff": 0, "pf": 0}

        wins = df.filter(pl.col("label") == 1).height
        losses = df.filter(pl.col("label") == -1).height
        timeouts = df.filter(pl.col("label") == 0).height

        avg_payoff = df["payoff_ratio"].mean()

        if losses > 0:
            pf = (wins * avg_payoff) / losses
        else:
            pf = np.inf if wins > 0 else 1.0

        return {
            "wins": wins,
            "losses": losses,
            "timeouts": timeouts,
            "avg_payoff": avg_payoff,
            "pf": pf,
        }

    except pl.exceptions.ComputeError as e:
        logging.warning(
            f"Analysis skipped: No parquet files found (or other read error): {e}"
        )
        return {"wins": 0, "losses": 0, "timeouts": 0, "avg_payoff": 0, "pf": 0}
    except Exception as e:
        logging.error(f"An unexpected error occurred during analysis: {e}")
        return {"wins": 0, "losses": 0, "timeouts": 0, "avg_payoff": 0, "pf": 0}


if __name__ == "__main__":
    logging.info("### Phase 2: Grid Search Optimizer START ###")
    logging.info(f"Target Year/Month: {TEST_YEAR_MONTH}")
    logging.info(f"Output Directory (S6): {S6_LABELED_DATASET}")
    logging.info(f"Results will be auto-saved to: {RESULTS_CSV_PATH}")

    # --- [CSV自動保存] ヘッダー定義と書き込み ---
    header = [
        "Regime",
        "ATR_Min",
        "ATR_Max",
        "PT",
        "SL",
        "TD",
        "Wins",
        "Losses",
        "Timeouts",
        "Avg_Payoff",
        "Profit_Factor",
    ]

    if not RESULTS_CSV_PATH.exists():
        try:
            with open(RESULTS_CSV_PATH, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(header)
            logging.info(
                f"Created new results file and wrote header: {RESULTS_CSV_PATH}"
            )
        except IOError as e:
            logging.error(f"FATAL: Could not write header to {RESULTS_CSV_PATH}: {e}")
            sys.exit(1)

    print("\n--- Grid Search Results (also saving to CSV) ---")
    print(",".join(header))

    total_runs = 0

    # 外側ループ (レジーム)
    for regime_name, regime_values in REGIMES.items():
        # 内側ループ (パラメータ)
        grid = PARAMETER_GRID[regime_name]
        param_combinations = list(
            itertools.product(grid["pt_mult"], grid["sl_mult"], grid["duration"])
        )

        if TEST_MODE:
            param_combinations = param_combinations[:1]

        total_runs += len(param_combinations)

        for pt, sl, td in param_combinations:
            # 1. ラベリング実行
            labeled_output_dir = run_labeling(regime_values, pt, sl, td)

            # 2. 結果の集計
            stats = analyze_results(labeled_output_dir)

            # --- ▼▼▼ [CSV自動保存] データ行の作成 (Noneを 'inf' に変換) ▼▼▼ ---

            # atr_max が None の場合、CSV/コンソール出力用に 'inf' (無限大) という文字列を使う
            atr_max_str = (
                "inf" if regime_values["max"] is None else regime_values["max"]
            )

            data_row = [
                regime_name,
                regime_values["min"],
                atr_max_str,  # <- 修正
                pt,
                sl,
                td,
                stats["wins"],
                stats["losses"],
                stats["timeouts"],
                f"{stats['avg_payoff']:.4f}",
                f"{stats['pf']:.4f}",
            ]

            # 3a. CSVファイルに行を追記
            try:
                with open(RESULTS_CSV_PATH, "a", newline="") as f:
                    writer = csv.writer(f)
                    writer.writerow(data_row)
            except IOError as e:
                logging.error(f"Failed to append row to CSV: {e}")

            # 3b. コンソールにも出力 (監視用)
            print(",".join(map(str, data_row)))
            # --- ▲▲▲ CSV行追記ここまで ▲▲▲ ---

    logging.info(f"\n### Grid Search Optimizer FINISHED ###")
    logging.info(f"Total combinations tested: {total_runs}")
    logging.info(f"All results saved to: {RESULTS_CSV_PATH}")
