import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.ticker as ticker
import os

# ==========================================
# 1. 設定
# ==========================================
# 読み込むCSVファイルの絶対パス
CSV_FILE = "/workspace/data/XAUUSD/stratum_7_models/backtest_simulator_results/M2_20260416_113241_Th0.7_D0.3_R2/detailed_trade_log_v5_M2.csv"

# 【期間指定】描画したい期間を設定してください (Noneにすると全期間)
# 書式: "YYYY-MM-DD" または "YYYY-MM-DD HH:MM:SS"
START_DATE = "2021-07-12"
END_DATE = "2021-09-10"

# 出力する画像ファイル名 (CSVと同じディレクトリに保存)
output_dir = os.path.dirname(CSV_FILE)
# 期間指定がある場合はファイル名に付与
suffix = (
    f"_{START_DATE}_to_{END_DATE}".replace("-", "").replace(" ", "_").replace(":", "")
    if START_DATE
    else ""
)
OUTPUT_IMAGE = os.path.join(output_dir, f"equity_curve_v5_M2_linear{suffix}.png")


# ==========================================
# 2. データの読み込みと前処理
# ==========================================
def load_and_prep_data(filepath, start_dt=None, end_dt=None):
    if not os.path.exists(filepath):
        print(f"Error: File '{filepath}' not found.")
        return None

    print(f"Loading data from {filepath}...")
    df = pd.read_csv(filepath, parse_dates=["timestamp"])

    # 時系列順にソートしてインデックス化
    df = df.sort_values("timestamp").set_index("timestamp")

    # --- 期間フィルタリング ---
    if start_dt:
        df = df[df.index >= pd.to_datetime(start_dt)]
    if end_dt:
        df = df[df.index <= pd.to_datetime(end_dt)]

    if df.empty:
        print("Error: 指定された期間にデータが存在しません。")
        return None

    print(f"Filtered Data: {len(df)} trades found in specified period.")
    return df


# ==========================================
# 3. ドローダウンの計算 (フィルタ後のデータ内で再計算)
# ==========================================
def calculate_drawdown(equity_series):
    rolling_max = equity_series.cummax()
    drawdown_pct = (equity_series - rolling_max) / rolling_max * 100
    return drawdown_pct


# ==========================================
# 4. グラフの描画
# ==========================================
def plot_linear_equity_curve(df, output_path):
    df["Drawdown_Pct"] = calculate_drawdown(df["balance"])

    plt.style.use("dark_background")
    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(14, 10), gridspec_kw={"height_ratios": [3, 1]}, sharex=True
    )

    title = (
        f"Project Forge V5 - Linear Equity Curve\n({START_DATE} to {END_DATE})"
        if START_DATE
        else "Project Forge V5 - Linear Equity Curve"
    )
    fig.suptitle(title, fontsize=18, fontweight="bold", color="white")

    # --- 上段: エクイティカーブ ---
    ax1.plot(
        df.index, df["balance"], color="#00ff00", linewidth=1.5, label="Equity (USD)"
    )
    ax1.set_ylabel("Equity Balance (USD)", fontsize=14)
    ax1.grid(True, linestyle="--", alpha=0.3)
    ax1.yaxis.set_major_formatter(
        ticker.FuncFormatter(lambda x, p: format(int(x), ","))
    )

    # --- 下段: ドローダウン ---
    ax2.fill_between(df.index, df["Drawdown_Pct"], 0, color="red", alpha=0.5)
    ax2.plot(df.index, df["Drawdown_Pct"], color="red", linewidth=1)
    ax2.set_ylabel("Drawdown (%)", fontsize=14)
    ax2.grid(True, linestyle="--", alpha=0.3)

    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    plt.xticks(rotation=45)
    plt.tight_layout()
    fig.subplots_adjust(top=0.9)

    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"Plot successfully saved to {output_path}")


if __name__ == "__main__":
    df_trades = load_and_prep_data(CSV_FILE, START_DATE, END_DATE)
    if df_trades is not None:
        plot_linear_equity_curve(df_trades, OUTPUT_IMAGE)
