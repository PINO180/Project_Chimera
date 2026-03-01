import polars as pl
from pathlib import Path
import sys

# プロジェクトルートの設定 (必要に応じて調整)
current_dir = Path(__file__).resolve().parent
project_root = current_dir.parent
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

# 設定ファイルからパスを取得（または直接指定）
from blueprint import S7_MODELS

# CSVファイルのパス
LOG_CSV_PATH = S7_MODELS / "detailed_trade_log.csv"
OUTPUT_SUMMARY_PATH = S7_MODELS / "timeframe_analysis_report.csv"


def analyze_timeframes():
    print(f"📂 Loading log file: {LOG_CSV_PATH}")

    if not LOG_CSV_PATH.exists():
        print(f"❌ Error: File not found at {LOG_CSV_PATH}")
        return

    try:
        # CSV読み込み (Timeframe列がない古いログだとエラーになるためチェック)
        df = pl.read_csv(LOG_CSV_PATH)

        if "timeframe" not in df.columns:
            print("❌ Error: 'timeframe' column not found in the CSV.")
            print("   Please rerun the backtest simulator with the latest code.")
            return

        # ラベル定義: 1=Win, -1=Loss, 0=Timeout (Entry logic dependent)
        # 集計処理
        summary = (
            df.group_by("timeframe")
            .agg(
                [
                    pl.len().alias("Total Trades"),
                    pl.col("label").filter(pl.col("label") == 1).len().alias("Wins"),
                    pl.col("label").filter(pl.col("label") == -1).len().alias("Losses"),
                    pl.col("label")
                    .filter(pl.col("label") == 0)
                    .len()
                    .alias("Timeouts"),
                    pl.col("pnl").sum().alias("Total PnL"),
                    pl.col("pnl").mean().alias("Avg PnL"),
                ]
            )
            .with_columns(
                [
                    # 勝率計算 (Win / Total)
                    (pl.col("Wins") / pl.col("Total Trades") * 100)
                    .round(2)
                    .alias("Win Rate %"),
                    # プロフィットファクター簡易計算 (Total PnLがプラスの時間足のみ)
                    # 注: ここでは簡易的に勝率と数だけで見る
                ]
            )
            .sort("timeframe")  # 名前順でソート（必要なら "Total Trades" などに変更）
        )

        # 時間足の並び順を自然にするためのマッピング（M8 -> M15 -> H1...）
        # 文字列ソートだと M15 が M8 より先に来たりするので、並び替えたい場合はカスタムソートが必要
        # ここでは簡易的に行数が多い順（取引が活発な順）に並び替えます
        summary = summary.sort("Total Trades", descending=True)

        print("\n" + "=" * 60)
        print("📊 Timeframe Performance Summary")
        print("=" * 60)

        # コンソールに見やすく表示
        print(summary)

        # CSVに保存
        summary.write_csv(OUTPUT_SUMMARY_PATH)
        print("\n" + "=" * 60)
        print(f"✅ Summary saved to: {OUTPUT_SUMMARY_PATH}")

    except Exception as e:
        print(f"❌ An error occurred: {e}")


if __name__ == "__main__":
    analyze_timeframes()
