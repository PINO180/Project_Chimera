"""
analyze_brain_csv_stats.py
================================================================
脳CSV(detailed_trade_log)を「脳が撃った母集団の事実を集計する」ためだけに使う。

【物差しの原則 — 重要】
  このスクリプトは脳CSVから「脳が実際に何をどこで撃ったか」の事実だけを集計する。
  相場そのものの性質（L+180 から伸びるか、PT到達率、EV 等）は一切結論しない。
  ── 相場の性質は全グリッド系（verify_grid_excursion_unbiased / pt_target_from_L /
     pt_sl_race_grid）の担当。脳CSVを母集団にして相場を語ると §27/§28 の母集団バイアス
     （§30 で撤回）を再発する。ここは「脳の撃ち方の事実」に限定する。

【出す事実】
  (A) 基本: 件数・勝率(label)・PnL 分布・方向内訳・平均TD・spread
  (B) M2確信度の分布（撃ったトレードの m2_proba 帯別）
  (C) ATR値帯・atr_ratio帯 別の件数と勝率（脳がどのボラ帯を撃つか）
  (D) 【tick指定時】d_realized 分布 = 6.3倍濃縮の再現用
        d_realized = |price(L+180) − price(L)| / ATR
        price(L) = close_price 列（BTのS6 close = price(L)）をそのまま使用（tick不要で厳密）
        price(L+180) = tick から L+180 の mid（backward）… tick 指定時のみ
      → 脳CSVの d≥1 率が出る。全グリッド系の d≥1 率(§31,12.1%)と割れば濃縮倍率。

  ※ (A)(B)(C) は tick 不要（CSV だけ）。(D) のみ --tick-dir 指定時に計算。

使い方（CSVのみ）:
  python analyze_brain_csv_stats.py \
      --log "/workspace/.../detailed_trade_log_v5_M2.csv" --log-tz jst

使い方（6.3倍濃縮の d_realized も出す）:
  python analyze_brain_csv_stats.py \
      --log "/workspace/.../detailed_trade_log_v5_M2.csv" \
      --tick-dir /workspace/data/XAUUSD/stratum_1_base/master_tick_partitioned \
      --log-tz jst
"""

import argparse
from pathlib import Path
import numpy as np
import pandas as pd

ACTION_HORIZON_SEC = 180
US = 1_000_000
M2_BINS = [(0.70, 0.75), (0.75, 0.80), (0.80, 0.85), (0.85, 0.90), (0.90, 0.95), (0.95, 1.01)]
ATR_VALUE_BINS = [(0.0, 1.0), (1.0, 2.0), (2.0, 3.0), (3.0, 5.0), (5.0, 1e9)]
ATR_RATIO_BINS = [(0.8, 1.0), (1.0, 1.2), (1.2, 1.5), (1.5, 1e9)]
D_BINS = [(0.0, 0.5), (0.5, 1.0), (1.0, 1.5), (1.5, 2.0), (2.0, 3.0), (3.0, 1e9)]
# 全グリッド系(§31)の d>=1.0 率。濃縮倍率の分母。将来変わったらここを更新。
GRID_D_GE1_RATE = 12.1


def sep(t=""):
    print("\n" + "=" * 74)
    if t:
        print(f"  {t}")
        print("=" * 74)


def load_trade_log(path, tz):
    print(f"[load] trade_log(事実集計用): {path}")
    df = pd.read_csv(path, low_memory=False)
    if tz.lower() == "jst":
        df["timestamp"] = pd.to_datetime(df["timestamp"]).dt.tz_localize("Asia/Tokyo").dt.tz_convert("UTC")
    else:
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    for c in ["m2_proba", "direction", "label", "atr_value", "atr_ratio", "TD", "pnl", "close_price", "spread", "lot_size"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    print(f"  -> {len(df):,} トレード / 列: {list(df.columns)}")
    return df


def _rate_table(df, bins, col, title, label_col="label"):
    sep(title)
    has_label = label_col in df.columns
    print(f"  {'帯':>14} {'件数':>9} {'割合%':>7}" + ("  勝率%" if has_label else ""))
    n_all = len(df)
    for lo, hi in bins:
        m = (df[col] >= lo) & (df[col] < hi)
        n = int(m.sum())
        if n == 0:
            continue
        line = f"  {f'[{lo:.2f},{hi:.2f})':>14} {n:>9,} {n/n_all*100:>6.1f}%"
        if has_label:
            wr = df.loc[m, label_col].mean() * 100
            line += f"  {wr:5.1f}%"
        print(line)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", type=Path, required=True)
    ap.add_argument("--tick-dir", type=Path, default=None, help="指定時のみ d_realized(6.3倍濃縮)を計算")
    ap.add_argument("--log-tz", default="jst", choices=["jst", "utc"])
    ap.add_argument("--out-dir", type=Path, default=Path("/workspace/data/diagnostics/brain_csv_stats"))
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    df = load_trade_log(args.log, args.log_tz)

    # ── (A) 基本の事実 ──
    sep("(A) 基本の事実（脳が撃った母集団）")
    n = len(df)
    print(f"  総トレード: {n:,}")
    if "label" in df.columns:
        print(f"  勝率(label=1): {df['label'].mean()*100:.2f}%")
    if "direction" in df.columns:
        nl = int((df["direction"] > 0).sum()); ns = int((df["direction"] < 0).sum())
        print(f"  方向: Long {nl:,} ({nl/n*100:.1f}%) / Short {ns:,} ({ns/n*100:.1f}%)")
    if "TD" in df.columns:
        print(f"  平均保有(TD): {df['TD'].mean():.1f} 分  (中央 {df['TD'].median():.1f})")
    if "pnl" in df.columns:
        tot = df["pnl"].sum(); win = df.loc[df['pnl'] > 0, 'pnl'].sum(); loss = df.loc[df['pnl'] < 0, 'pnl'].sum()
        pf = (win / abs(loss)) if loss != 0 else float("inf")
        print(f"  PnL合計: {tot:,.2f}  / PF: {pf:.2f}  (総益 {win:,.0f} / 総損 {loss:,.0f})")
    if "spread" in df.columns:
        print(f"  平均spread控除: {df['spread'].mean():.4f}")

    # ── (B) M2確信度の分布 ──
    if "m2_proba" in df.columns:
        _rate_table(df, M2_BINS, "m2_proba", "(B) M2確信度の分布（撃ったトレードの確信度帯別・勝率）")

    # ── (C) ATR帯・atr_ratio帯 ──
    if "atr_value" in df.columns:
        _rate_table(df, ATR_VALUE_BINS, "atr_value", "(C-1) ATR値帯別（脳がどのボラを撃つか・勝率）")
    if "atr_ratio" in df.columns:
        _rate_table(df, ATR_RATIO_BINS, "atr_ratio", "(C-2) atr_ratio帯別（相対ボラ・勝率）")

    # ── (D) d_realized（6.3倍濃縮の再現）── tick 指定時のみ ──
    if args.tick_dir is not None and "close_price" in df.columns:
        import polars as pl
        print("\n[load] ticks (d_realized 計算用)...")
        t0 = (df["timestamp"].astype("int64").to_numpy() // 1000).astype(np.int64)
        atr = df["atr_value"].to_numpy().astype(np.float64)
        direction = df["direction"].to_numpy().astype(np.float64)
        price_L = df["close_price"].to_numpy().astype(np.float64)  # BT: close_price = price(L)
        t_min = df["timestamp"].min() - pd.Timedelta(hours=8)
        t_max = df["timestamp"].max() + pd.Timedelta(hours=8)
        lf = (pl.scan_parquet(str(args.tick_dir / "**/*.parquet"), hive_partitioning=True)
              .rename({"datetime": "timestamp"}).select("timestamp", "mid_price")
              .with_columns(pl.col("timestamp").cast(pl.Datetime("us", "UTC")))
              .filter(pl.col("timestamp").is_between(pl.lit(t_min), pl.lit(t_max)))
              .unique("timestamp", keep="first").sort("timestamp"))
        tdf = lf.collect()
        ts_us = tdf["timestamp"].cast(pl.Int64).to_numpy()
        px = tdf["mid_price"].cast(pl.Float64).to_numpy()
        print(f"  -> {len(px):,} ticks")
        entry_off = np.int64(ACTION_HORIZON_SEC * US)
        iE = np.clip(np.searchsorted(ts_us, t0 + entry_off, side="right") - 1, 0, len(px) - 1)
        price_E = px[iE]  # price(L+180) = L+180以前最新tick(backward, M3 close相当)
        with np.errstate(divide="ignore", invalid="ignore"):
            d_real = np.abs(price_E - price_L) / atr
        df["d_realized"] = d_real

        sep("(D) d_realized 分布（6.3倍濃縮の再現）")
        print("  ※ price(L)=close_price(BTのS6 close), price(L+180)=tick backward。")
        n_all = int(np.isfinite(d_real).sum())
        for lo, hi in D_BINS:
            m = (d_real >= lo) & (d_real < hi)
            k = int(m.sum())
            if k == 0:
                continue
            label = f"d[{lo:.1f},{hi:.1f})" if hi < 1e8 else f"d[{lo:.1f},inf)"
            print(f"  {label:>12} {k:>9,} ({k/n_all*100:5.1f}%)")
        ge1 = int((d_real >= 1.0).sum())
        ge1_rate = ge1 / n_all * 100
        print(f"\n  脳CSV の d>=1.0 率: {ge1_rate:.1f}%")
        print(f"  全グリッド(§31)の d>=1.0 率: {GRID_D_GE1_RATE}%")
        print(f"  → 濃縮倍率: {ge1_rate/GRID_D_GE1_RATE:.1f}倍  "
              f"（脳は勢いのある局面をこの倍率で選り好みして撃っている）")
        df.to_parquet(args.out_dir / "brain_csv_with_d.parquet", index=False)
        print(f"\n  保存: {args.out_dir}/brain_csv_with_d.parquet")
    else:
        print("\n  [note] --tick-dir 未指定のため (D) d_realized(6.3倍濃縮) はスキップ。")
        print("         (A)(B)(C) は CSV のみで完結。")

    sep("完了")
    print("  ※ 本スクリプトは脳の撃ち方の『事実』のみ。相場の性質は全グリッド系で測ること。")


if __name__ == "__main__":
    main()
