#!/usr/bin/env python3
"""
BT detailed_trade_log_v5_M2.csv から日次パフォーマンスを分析する。

【目的】
今日（2026/5/7 UTC22-02）のライブ成績（勝率66.7%、TO率27.3%、損益-$8.40）が
BT 4.7年分布のどこに位置するかを判定する。

【出力】
1. 日次P&L分布（+/-/±0日数）
2. 日次勝率分布（パーセンタイル）
3. 日次TO率分布
4. 連続マイナス日の最長streak
5. 最悪日 TOP20 詳細
6. 時間帯別パフォーマンス
7. 曜日別パフォーマンス（月曜性）
8. 方向偏りの影響

【使い方】
python analyze_bt_daily.py [path/to/detailed_trade_log_v5_M2.csv]
デフォルトパスは下記 DEFAULT_PATH 参照。
"""

import sys
from pathlib import Path
import polars as pl

DEFAULT_PATH = (
    "/workspace/data/XAUUSD/stratum_7_models/backtest_simulator_results/"
    "M2_20260504_195717_Th0.7_D0.3_R2 _Phase6_V5/detailed_trade_log_v5_M2.csv"
)

# 今日のライブ実績（5/7 UTC22-02）
TODAY = {
    "trades": 33,
    "win_rate": 66.7,
    "to_rate": 27.3,
    "sl_rate": 6.1,
    "pnl": -8.40,
}


def load_log(path: str) -> pl.DataFrame:
    df = pl.read_csv(path, try_parse_dates=True)
    print(f"\nロード完了: {len(df):,} 行 ({df.columns[:8]}...)")
    print(f"カラム数: {len(df.columns)}")
    print(f"主要カラム: {[c for c in df.columns if c in ['entry_time','exit_time','direction','exit_reason','pnl','profit','timestamp']]}")
    return df


def normalize_columns(df: pl.DataFrame) -> pl.DataFrame:
    """カラム名のゆらぎを吸収してentry_time/exit_reason/pnl_decimalに統一"""
    rename_map = {}
    for c in df.columns:
        cl = c.lower()
        if cl in ("entry_time", "entry_timestamp", "timestamp"):
            rename_map[c] = "entry_time"
        elif cl in ("exit_reason", "close_reason", "exit_type"):
            rename_map[c] = "exit_reason"
        elif cl in ("pnl_decimal", "pnl", "profit", "net_pnl"):
            rename_map[c] = "pnl"
        elif cl == "direction":
            rename_map[c] = "direction"
    df = df.rename(rename_map)

    # entry_time を datetime にし、UTC日付カラムを追加
    if df["entry_time"].dtype == pl.String:
        df = df.with_columns(pl.col("entry_time").str.to_datetime())
    df = df.with_columns([
        pl.col("entry_time").dt.date().alias("date"),
        pl.col("entry_time").dt.hour().alias("hour_utc"),
        pl.col("entry_time").dt.weekday().alias("weekday"),  # 1=Mon
    ])
    return df


def daily_summary(df: pl.DataFrame) -> pl.DataFrame:
    """日次集計"""
    # PT/TO/SL 判定:
    #   PT: pnl > 0 (利益確定) ※label=1 と等価
    #   TO: pnl <= 0 かつ TD >= 29.5 分 (タイムアウト決済)
    #   SL: pnl <= 0 かつ TD < 29.5 分  (ストップロス早期決済)
    return (
        df.group_by("date")
        .agg([
            pl.len().alias("trades"),
            pl.col("pnl").sum().alias("daily_pnl"),
            (pl.col("pnl") > 0).sum().alias("pt"),
            ((pl.col("pnl") <= 0) & (pl.col("TD") >= 29.5)).sum().alias("to"),
            ((pl.col("pnl") <= 0) & (pl.col("TD") < 29.5)).sum().alias("sl"),
            ((pl.col("direction").cast(pl.Utf8).str.to_uppercase().is_in(["BUY", "LONG", "1"]))).sum().alias("buys"),
            ((pl.col("direction").cast(pl.Utf8).str.to_uppercase().is_in(["SELL", "SHORT", "-1"]))).sum().alias("sells"),
        ])
        .with_columns([
            (pl.col("pt") / pl.col("trades") * 100).alias("win_rate"),
            (pl.col("to") / pl.col("trades") * 100).alias("to_rate"),
            (pl.col("sl") / pl.col("trades") * 100).alias("sl_rate"),
            (pl.col("buys") / pl.col("trades") * 100).alias("buy_pct"),
            pl.col("date").dt.weekday().alias("weekday"),
        ])
        .sort("date")
    )


def analyze_pnl_distribution(daily: pl.DataFrame, threshold_zero: float = 100.0):
    """要望項目: + / - / ±0 日カウント"""
    plus = daily.filter(pl.col("daily_pnl") > threshold_zero).height
    minus = daily.filter(pl.col("daily_pnl") < -threshold_zero).height
    zero = daily.filter(
        (pl.col("daily_pnl") >= -threshold_zero) & (pl.col("daily_pnl") <= threshold_zero)
    ).height
    n = daily.height

    print("\n" + "=" * 70)
    print(f"【1】日次P&L分布（±0定義: |daily_pnl| <= ${threshold_zero}）")
    print("=" * 70)
    print(f"  プラスの日: {plus:>4} 日 ({plus/n*100:5.1f}%)  日次P&L > +${threshold_zero}")
    print(f"  マイナス日: {minus:>4} 日 ({minus/n*100:5.1f}%)  日次P&L < -${threshold_zero}")
    print(f"  ゼロ近傍 : {zero:>4} 日 ({zero/n*100:5.1f}%)   |日次P&L| <= ${threshold_zero}")
    print(f"  合計     : {n:>4} 日")

    # 別の閾値でも見る
    print(f"\n  単純な符号判定（>0 / <0 / =0）:")
    p = daily.filter(pl.col("daily_pnl") > 0).height
    m = daily.filter(pl.col("daily_pnl") < 0).height
    z = daily.filter(pl.col("daily_pnl") == 0).height
    print(f"    > 0: {p} 日 ({p/n*100:.1f}%)")
    print(f"    < 0: {m} 日 ({m/n*100:.1f}%)")
    print(f"    = 0: {z} 日 ({z/n*100:.1f}%)")

    # 統計量
    pnl = daily["daily_pnl"]
    print(f"\n  日次P&L 統計:")
    print(f"    平均   : ${pnl.mean():>14,.2f}")
    print(f"    中央値 : ${pnl.median():>14,.2f}")
    print(f"    最大   : ${pnl.max():>14,.2f}")
    print(f"    最小   : ${pnl.min():>14,.2f}")
    print(f"    標準偏差: ${pnl.std():>14,.2f}")


def analyze_win_rate_distribution(daily: pl.DataFrame):
    """要望項目補強: 今日の66.7%がBT分布のどこか"""
    print("\n" + "=" * 70)
    print("【2】日次勝率分布")
    print("=" * 70)
    wr = daily["win_rate"]
    pcts = [1, 5, 10, 25, 50, 75, 90, 95, 99]
    for p in pcts:
        v = wr.quantile(p / 100)
        print(f"  P{p:>2}%ile: {v:5.2f}%")

    # 今日の位置
    today_wr = TODAY["win_rate"]
    below = daily.filter(pl.col("win_rate") < today_wr).height
    n = daily.height
    print(f"\n  今日(5/7)の勝率 {today_wr}% は:")
    print(f"    BT分布で下位 {below}/{n} 日 ({below/n*100:.1f}%ile)")
    print(f"    つまり「BT全 {n} 日のうち {below} 日は今日より悪かった」")


def analyze_to_rate_distribution(daily: pl.DataFrame):
    """TO率分布"""
    print("\n" + "=" * 70)
    print("【3】日次TO率分布")
    print("=" * 70)
    tor = daily["to_rate"]
    pcts = [50, 75, 90, 95, 99, 99.5, 99.9]
    for p in pcts:
        v = tor.quantile(p / 100)
        print(f"  P{p:>4}%ile: {v:5.2f}%")

    today_to = TODAY["to_rate"]
    above = daily.filter(pl.col("to_rate") > today_to).height
    n = daily.height
    print(f"\n  今日(5/7)のTO率 {today_to}% は:")
    print(f"    BT分布で上位 {above}/{n} 日 ({above/n*100:.2f}%ile)")
    if above < n * 0.01:
        print("    → BT 4.7年で1%以下の異常値")


def analyze_consecutive_loss_streaks(daily: pl.DataFrame):
    """連続マイナス日"""
    print("\n" + "=" * 70)
    print("【4】連続マイナス日の最長streak")
    print("=" * 70)

    # 符号配列（-1=マイナス, 1=プラス, 0=ゼロ）
    signs = daily["daily_pnl"].to_list()

    max_streak = 0
    current_streak = 0
    streak_dates = []
    current_dates = []
    dates = daily["date"].to_list()

    for i, p in enumerate(signs):
        if p < 0:
            current_streak += 1
            current_dates.append(dates[i])
            if current_streak > max_streak:
                max_streak = current_streak
                streak_dates = list(current_dates)
        else:
            current_streak = 0
            current_dates = []

    print(f"  連続マイナス日 最長: {max_streak} 日")
    if streak_dates:
        print(f"  期間: {streak_dates[0]} 〜 {streak_dates[-1]}")
        # その期間の合計損益
        loss_total = (
            daily.filter(pl.col("date").is_in(streak_dates))["daily_pnl"].sum()
        )
        print(f"  合計損益: ${loss_total:,.2f}")

    # 連続streak長さの分布
    print(f"\n  連続マイナス日 streak長さ分布:")
    streaks = []
    cur = 0
    for p in signs:
        if p < 0:
            cur += 1
        else:
            if cur > 0:
                streaks.append(cur)
            cur = 0
    if cur > 0:
        streaks.append(cur)

    if streaks:
        from collections import Counter
        ctr = Counter(streaks)
        for k in sorted(ctr.keys()):
            print(f"    {k}日連続マイナス: {ctr[k]} 回")


def analyze_worst_days(daily: pl.DataFrame, top_n: int = 20):
    """最悪日 TOP20"""
    print("\n" + "=" * 70)
    print(f"【5】最悪日 TOP{top_n}（日次P&L昇順）")
    print("=" * 70)

    worst = daily.sort("daily_pnl").head(top_n)
    print(
        f"  {'#':<4} {'date':<12} {'曜日':<5} {'P&L':>12} "
        f"{'取引数':>5} {'勝率%':>6} {'TO%':>6} {'SL%':>6} {'BUY%':>6}"
    )
    print("-" * 80)
    weekday_jp = {1: "Mon", 2: "Tue", 3: "Wed", 4: "Thu", 5: "Fri", 6: "Sat", 7: "Sun"}
    for i, row in enumerate(worst.iter_rows(named=True), 1):
        print(
            f"  {i:<4} {str(row['date']):<12} {weekday_jp.get(row['weekday'],'?'):<5} "
            f"${row['daily_pnl']:>+10,.2f}  "
            f"{row['trades']:>5} {row['win_rate']:>6.1f} "
            f"{row['to_rate']:>6.1f} {row['sl_rate']:>6.1f} {row['buy_pct']:>6.1f}"
        )


def analyze_hourly_pnl(df: pl.DataFrame):
    """時間帯別P&L（UTC）"""
    print("\n" + "=" * 70)
    print("【6】UTC時間帯別 取引数 / 勝率 / 平均PnL")
    print("=" * 70)

    by_hour = (
        df.group_by("hour_utc")
        .agg([
            pl.len().alias("trades"),
            pl.col("pnl").sum().alias("total_pnl"),
            (pl.col("pnl") > 0).sum().alias("pt"),
        ])
        .with_columns([
            (pl.col("pt") / pl.col("trades") * 100).alias("win_rate"),
            (pl.col("total_pnl") / pl.col("trades")).alias("avg_pnl"),
        ])
        .sort("hour_utc")
    )

    print(f"  {'UTC':<5} {'取引数':>7} {'勝率%':>7} {'平均PnL':>10} {'合計PnL':>12}")
    print("-" * 50)
    for r in by_hour.iter_rows(named=True):
        marker = " ★今日の問題時間帯" if r["hour_utc"] in (22, 0, 1, 2) else ""
        print(
            f"  {r['hour_utc']:<5} {r['trades']:>7,} {r['win_rate']:>7.2f} "
            f"${r['avg_pnl']:>9,.2f} ${r['total_pnl']:>11,.2f}{marker}"
        )


def analyze_weekday(daily: pl.DataFrame):
    """曜日別（月曜=gap明けが悪いか）"""
    print("\n" + "=" * 70)
    print("【7】曜日別（月曜=週末gap明け）")
    print("=" * 70)
    weekday_jp = {1: "Mon", 2: "Tue", 3: "Wed", 4: "Thu", 5: "Fri", 6: "Sat", 7: "Sun"}
    by_wd = (
        daily.group_by("weekday")
        .agg([
            pl.len().alias("days"),
            pl.col("daily_pnl").mean().alias("avg_pnl"),
            pl.col("daily_pnl").median().alias("median_pnl"),
            pl.col("win_rate").mean().alias("avg_win_rate"),
            pl.col("to_rate").mean().alias("avg_to_rate"),
        ])
        .sort("weekday")
    )
    print(f"  {'曜日':<5} {'日数':>5} {'平均PnL':>10} {'中央値':>10} {'平均勝率':>9} {'平均TO率':>9}")
    print("-" * 60)
    for r in by_wd.iter_rows(named=True):
        print(
            f"  {weekday_jp.get(r['weekday'],'?'):<5} {r['days']:>5} "
            f"${r['avg_pnl']:>9,.2f} ${r['median_pnl']:>9,.2f} "
            f"{r['avg_win_rate']:>8.2f}% {r['avg_to_rate']:>8.2f}%"
        )


def analyze_direction_bias(daily: pl.DataFrame):
    """方向偏り（BUY偏重日 vs SELL偏重日 vs バランス日）"""
    print("\n" + "=" * 70)
    print("【8】方向偏りの影響")
    print("=" * 70)

    daily_with_bias = daily.with_columns(
        pl.when(pl.col("buy_pct") > 70).then(pl.lit("BUY偏重"))
        .when(pl.col("buy_pct") < 30).then(pl.lit("SELL偏重"))
        .otherwise(pl.lit("バランス"))
        .alias("bias")
    )
    by_bias = (
        daily_with_bias.group_by("bias")
        .agg([
            pl.len().alias("days"),
            pl.col("daily_pnl").mean().alias("avg_pnl"),
            pl.col("win_rate").mean().alias("avg_win_rate"),
        ])
    )
    print(f"  {'バイアス':<10} {'日数':>5} {'平均PnL':>10} {'平均勝率':>9}")
    print("-" * 45)
    for r in by_bias.iter_rows(named=True):
        print(
            f"  {r['bias']:<10} {r['days']:>5} ${r['avg_pnl']:>9,.2f} {r['avg_win_rate']:>8.2f}%"
        )


def analyze_today_vs_bt(daily: pl.DataFrame):
    """今日(5/7) vs BT全体の総合比較"""
    print("\n" + "=" * 70)
    print(f"【総合】今日(5/7)のライブ成績 vs BT 4.7年分布")
    print("=" * 70)
    n = daily.height

    metrics = [
        ("勝率", TODAY["win_rate"], "win_rate", "low"),     # 低いほど悪い
        ("TO率", TODAY["to_rate"], "to_rate", "high"),      # 高いほど悪い
        ("取引数", TODAY["trades"], "trades", "neutral"),
    ]

    print(f"  {'指標':<8} {'今日':>10} {'BT中央値':>12} {'BTパーセンタイル':<25}")
    print("-" * 65)
    for name, today_val, col, direction in metrics:
        median = daily[col].median()
        if direction == "low":
            below = daily.filter(pl.col(col) < today_val).height
            pct = below / n * 100
            note = f"下位{pct:.1f}%ile （悪さで）"
        elif direction == "high":
            above = daily.filter(pl.col(col) > today_val).height
            pct = above / n * 100
            note = f"上位{pct:.1f}%ile （悪さで）"
        else:
            below = daily.filter(pl.col(col) < today_val).height
            note = f"下位{below/n*100:.1f}%ile"
        print(f"  {name:<8} {today_val:>10.2f} {median:>12.2f} {note}")

    print()
    print("  解釈:")
    print(f"    勝率66.7%以下のBT日数 / TO率27.3%以上のBT日数 から、今日が異常値か通常変動内か判定。")
    print(f"    異常値（≤1%）であれば 何らかの原因究明が必要。通常変動内であれば市場由来。")


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_PATH
    if not Path(path).exists():
        print(f"ERROR: ファイルが見つかりません: {path}")
        sys.exit(1)

    df = load_log(path)
    df = normalize_columns(df)
    daily = daily_summary(df)
    print(f"\n対象期間: {daily['date'].min()} 〜 {daily['date'].max()}")
    print(f"対象日数: {daily.height} 日 / 取引総数: {df.height:,}")

    analyze_pnl_distribution(daily)
    analyze_win_rate_distribution(daily)
    analyze_to_rate_distribution(daily)
    analyze_consecutive_loss_streaks(daily)
    analyze_worst_days(daily)
    analyze_hourly_pnl(df)
    analyze_weekday(daily)
    analyze_direction_bias(daily)
    analyze_today_vs_bt(daily)

    # CSV出力（必要なら）
    out = Path(path).parent / "daily_summary.csv"
    daily.write_csv(out)
    print(f"\n日次集計CSV: {out}")


if __name__ == "__main__":
    main()
