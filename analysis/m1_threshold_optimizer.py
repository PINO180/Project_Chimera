# /workspace/analysis/m1_threshold_optimizer.py

import sys
from pathlib import Path
import logging
import warnings

import polars as pl
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm  # --- ▼▼▼ tqdmをインポート ▼▼▼ ---

# --- プロジェクトのルートディレクトリをPythonの検索パスに追加 ---
project_root = Path(__file__).resolve().parents[1]
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

from blueprint import S7_M1_OOF_PREDICTIONS, S7_MODELS

# --- ロギングと警告の抑制設定 ---
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
warnings.filterwarnings("ignore", category=UserWarning)

# --- 定数 ---
INPUT_FILE = S7_M1_OOF_PREDICTIONS
OUTPUT_CHART_PATH = S7_MODELS / "m1_threshold_performance_chart.png"


def analyze_thresholds(df: pl.DataFrame):
    """
    最適な予測閾値を見つけるための分析を実行し、結果を可視化・報告する。
    """
    logging.info("Starting threshold analysis for M1 model...")

    logging.info("Converting data to NumPy for high-speed analysis...")
    y_true = np.where(df["true_label"] == 1, 1, 0)
    y_pred_proba = df["prediction"].to_numpy()
    weights = df["uniqueness"].to_numpy()
    del df

    thresholds = np.arange(0.01, 0.51, 0.005)
    results = []

    # --- ▼▼▼ 進捗表示を追加 ▼▼▼ ---
    logging.info(f"Analyzing {len(thresholds)} different thresholds...")
    for threshold in tqdm(thresholds, desc="Analyzing Thresholds"):
        y_pred_binary = (y_pred_proba >= threshold).astype(int)

        if np.sum(y_pred_binary) == 0:
            precision, recall, f1 = 0.0, 0.0, 0.0
        else:
            precision = precision_score(
                y_true, y_pred_binary, sample_weight=weights, zero_division=0
            )
            recall = recall_score(
                y_true, y_pred_binary, sample_weight=weights, zero_division=0
            )
            f1 = f1_score(y_true, y_pred_binary, sample_weight=weights, zero_division=0)

        results.append(
            {
                "threshold": threshold,
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "trade_count": int(np.sum(y_pred_binary)),
                "trade_ratio": np.sum(y_pred_binary) / len(y_true),
            }
        )

    if not results:
        logging.error("No results were generated. Cannot proceed.")
        return

    results_df = pl.DataFrame(results)
    best_result = results_df.sort("f1", descending=True).row(0, named=True)

    # --- 結果報告 ---
    print("\n" + "=" * 80)
    print("### M1 Model Threshold Optimization Report ###")
    print("=" * 80)
    print(f"✅ Best F1-Score achieved: {best_result['f1']:.4f}")
    print(f"Optimal Threshold: {best_result['threshold']:.3f}\n")
    print("--- Performance at Optimal Threshold ---")
    print(
        f"Precision: {best_result['precision']:.4f}  (GOサインのうち、本当に当たりだったものの割合)"
    )
    print(
        f"Recall:    {best_result['recall']:.4f}  (本当に当たりだったもののうち、どれだけを捉えられたかの割合)"
    )
    print("\n--- Trade Activity at Optimal Threshold ---")
    print(
        f"Total Trades Generated: {best_result['trade_count']} / {len(y_true)} (全期間)"
    )
    print(
        f"Trade Ratio: {best_result['trade_ratio'] * 100:.2f}% (全データポイントに対する取引の割合)"
    )
    print("=" * 80)

    # --- グラフ描画 ---
    logging.info(f"Generating performance chart and saving to {OUTPUT_CHART_PATH}...")
    plt.style.use("seaborn-v0_8-darkgrid")
    fig, ax1 = plt.subplots(figsize=(16, 8))

    ax1.plot(results_df["threshold"], results_df["precision"], "b-", label="Precision")
    ax1.plot(results_df["threshold"], results_df["recall"], "g-", label="Recall")
    ax1.plot(
        results_df["threshold"], results_df["f1"], "r-", label="F1 Score", linewidth=2.5
    )
    ax1.axvline(
        x=best_result["threshold"],
        color="red",
        linestyle="--",
        label=f"Optimal Threshold (F1 Max): {best_result['threshold']:.3f}",
    )
    ax1.set_xlabel("Prediction Threshold")
    ax1.set_ylabel("Score")
    ax1.set_title("M1 Model: Precision, Recall, F1 Score vs. Threshold", fontsize=16)
    ax1.legend(loc="upper left")
    ax1.grid(True)

    ax2 = ax1.twinx()
    ax2.plot(
        results_df["threshold"],
        results_df["trade_count"],
        "k--",
        label="Trade Count",
        alpha=0.5,
    )
    ax2.set_ylabel("Number of Trades")
    ax2.legend(loc="upper right")

    fig.tight_layout()
    plt.savefig(OUTPUT_CHART_PATH)
    logging.info("Chart saved successfully.")
    plt.close()


def main():
    if not INPUT_FILE.exists():
        logging.error(
            f"FATAL: Input file not found at {INPUT_FILE}. Please run Script A first."
        )
        return
    logging.info(f"Loading M1 OOF predictions from {INPUT_FILE}...")
    try:
        df = pl.read_parquet(INPUT_FILE)
        if df.is_empty():
            logging.error("Input file is empty. Cannot perform analysis.")
            return
        analyze_thresholds(df)
    except Exception as e:
        logging.error(f"An error occurred during processing: {e}", excinfo=True)


if __name__ == "__main__":
    main()
