# /workspace/models/analysis_m1_hard_filters.py
import sys
from pathlib import Path
import logging
import matplotlib.pyplot as plt
import polars as pl
import numpy as np

# --- プロジェクトのルートディレクトリをPythonの検索パスに追加 ---
project_root = Path(__file__).resolve().parents[1]
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

# --- blueprintから必要なパスをインポート ---
from blueprint import S6_WEIGHTED_DATASET, S7_M1_OOF_PREDICTIONS, S7_MODELS

# --- ロギング設定 ---
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


def load_and_prepare_daily_filters():
    """
    S6データセットから日足を作成し、フィルタ用の指標（SMA200, 前日暴落）を計算する
    """
    logging.info("Generating Daily Indicators from S6 dataset...")

    # S6データ読み込み (OHLCが必要)
    # create_proxy_labels.py 等で作成されたS6には通常 open, close が含まれると仮定
    # 含まれない場合は close のみで日次リターンを近似計算する実装に切り替える必要あり
    q = pl.scan_parquet(S6_WEIGHTED_DATASET / "**/*.parquet")

    # 日足にリサンプリング
    daily_df = (
        q.with_columns(pl.col("timestamp").dt.date().alias("date"))
        .group_by("date")
        .agg(
            [
                pl.col("open").first().alias("open"),
                pl.col("high").max().alias("high"),
                pl.col("low").min().alias("low"),
                pl.col("close").last().alias("close"),
            ]
        )
        .sort("date")
        .collect()
    )

    # 指標計算
    # 1. SMA 200
    daily_df = daily_df.with_columns(
        pl.col("close").rolling_mean(window_size=200).alias("sma_200")
    )

    # 2. 前日暴落判定
    # 前日の (Close - Open) / Open < -0.02
    daily_df = daily_df.with_columns(
        [
            pl.col("close").shift(1).alias("prev_close"),
            pl.col("open").shift(1).alias("prev_open"),
        ]
    ).with_columns(
        ((pl.col("prev_close") - pl.col("prev_open")) / pl.col("prev_open")).alias(
            "prev_day_return"
        )
    )

    # フィルタ条件フラグの作成 (True = エントリー不可/除外対象)
    # A. トレンドフィルタ: Close < SMA_200
    # B. クラッシュフィルタ: prev_day_return < -0.02
    daily_df = daily_df.with_columns(
        [
            (pl.col("close") < pl.col("sma_200"))
            .fill_null(False)
            .alias("filter_downtrend"),
            (pl.col("prev_day_return") < -0.02)
            .fill_null(False)
            .alias("filter_crash_aftermath"),
        ]
    )

    logging.info(f"Daily data prepared. Rows: {len(daily_df)}")
    return daily_df.select(
        ["date", "filter_downtrend", "filter_crash_aftermath", "close", "sma_200"]
    )


def run_analysis():
    # 1. 日足フィルタデータの準備
    daily_filters = load_and_prepare_daily_filters()

    # 2. M1 OOFデータの読み込み
    logging.info(f"Loading M1 OOF Predictions from {S7_M1_OOF_PREDICTIONS}...")
    m1_oof = pl.read_parquet(S7_M1_OOF_PREDICTIONS)

    # 3. データの結合
    # M1データに日付キーを追加して結合
    logging.info("Merging M1 Data with Daily Filters...")
    m1_oof = m1_oof.with_columns(pl.col("timestamp").dt.date().alias("date"))

    # join (left join to keep all trades initially)
    df = m1_oof.join(daily_filters, on="date", how="left")

    # 4. 基本的なPnLの計算 (単利, 固定1単位リスク)
    # Prediction > 0.5 でエントリーと仮定
    # 勝てば Payoff Ratio 分の利益、負ければ -1 の損失
    threshold = 0.5

    # エントリーしたかどうかのベースフラグ
    df = df.with_columns((pl.col("prediction") > threshold).alias("signal_buy"))

    # トレードごとの損益 (フィルタ適用前)
    # Label: 1(Win), 0(Timeout/Neutral), -1(Loss)
    # payoff_ratio が null の場合は 1.0 (RR 1:1) と仮定
    df = df.with_columns(
        pl.col("payoff_ratio").fill_null(1.0).alias("payoff_ratio")
    ).with_columns(
        pl.when((pl.col("signal_buy")) & (pl.col("label") == 1))
        .then(pl.col("payoff_ratio"))
        .when((pl.col("signal_buy")) & (pl.col("label") == -1))
        .then(pl.lit(-1.0))
        .otherwise(pl.lit(0.0))
        .alias("raw_pnl")
    )

    # 5. シナリオごとのPnL計算
    scenarios = {
        "Original (No Filter)": None,
        "With Trend Filter (MA200)": pl.col("filter_downtrend"),
        "With Crash Filter (Day<-2%)": pl.col("filter_crash_aftermath"),
        "Combined (Trend + Crash)": (
            pl.col("filter_downtrend") | pl.col("filter_crash_aftermath")
        ),
    }

    results = {}

    # グラフ描画用
    plt.figure(figsize=(12, 6))

    print("\n" + "=" * 80)
    print(
        f"{'Scenario':<30} | {'Total Return':<12} | {'PF':<6} | {'MaxDD':<8} | {'Trades':<6} | {'WinRate':<7}"
    )
    print("-" * 80)

    for name, filter_expr in scenarios.items():
        if filter_expr is None:
            # フィルタなし
            pnl_series = df["raw_pnl"]
        else:
            # フィルタ条件に合致する場合（True）はトレードを除外（PnL=0）
            pnl_series = df.select(
                pl.when(filter_expr).then(0.0).otherwise(pl.col("raw_pnl"))
            ).to_series()

        # 累積収益
        cumsum = pnl_series.cum_sum()

        # スタッツ計算
        total_return = pnl_series.sum()
        n_trades = (pnl_series != 0).sum()

        wins = pnl_series.filter(pnl_series > 0).sum()
        losses = pnl_series.filter(pnl_series < 0).sum().abs()
        pf = wins / losses if losses != 0 else float("inf")

        win_count = (pnl_series > 0).sum()
        win_rate = win_count / n_trades * 100 if n_trades > 0 else 0.0

        # 最大ドローダウン (簡易計算)
        # 資産曲線（初期値0スタート）から計算
        equity = cumsum.to_list()
        max_equity = -float("inf")
        max_dd = 0.0
        for eq in equity:
            if eq > max_equity:
                max_equity = eq
            dd = max_equity - eq
            if dd > max_dd:
                max_dd = dd

        # コンソール出力
        print(
            f"{name:<30} | {total_return:>12.2f} | {pf:>6.2f} | {max_dd:>8.2f} | {n_trades:>6} | {win_rate:>6.1f}%"
        )

        # グラフプロット
        # データが多すぎる場合は間引いてプロットしてもよいが、今回は全点描画
        plt.plot(df["timestamp"], cumsum, label=f"{name} (PF: {pf:.2f})")

        results[name] = cumsum

    print("=" * 80 + "\n")

    plt.title("M1 Hard Filter Comparison (Cumulative Return)")
    plt.xlabel("Date")
    plt.ylabel("Cumulative PnL (Units)")
    plt.legend()
    plt.grid(True)

    output_path = S7_MODELS / "analysis_m1_hard_filters.png"
    plt.savefig(output_path)
    logging.info(f"Comparison chart saved to {output_path}")
    # plt.show() # 必要に応じてコメントアウトを外す


if __name__ == "__main__":
    run_analysis()
