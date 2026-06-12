#!/usr/bin/env python3
"""
compare_demo_vs_real_tick.py — デモ口座 vs リアル口座 Tick データの差分検証

[目的]
  Train-Serve Skew の真因候補の一つとして、学習側に使った Tick (デモ口座
  からエクスポートした CSV) と本番ライブの broker が見ている Tick
  (リアル口座) が、同じ broker / 同じ口座タイプでも実際には異なる可能性
  を直接検証する。

[既に観測されている兆候]
  同期間 (2026-05-20 00:00:00 〜 2026-05-21 00:00:00) でデモ 356,011 件
  vs リアル 355,815 件 (196 件差)。これだけで既にサンプリング lattice
  自体が異なる可能性が確定。

[検証項目]
  1. 件数差の内訳 (demo-only / real-only / 共通)
  2. 共通 timestamp の bid/ask/volume が bit-identical か
  3. 系統的な数値オフセット (bid/ask が常に X だけ高い/低い 等)
  4. 時間帯別の差 (特定時間帯にだけ差が集中するか)
  5. M0.5 (30秒足) 集約後の OHLCV が一致するか ← 本番への実害判定

[使い方]
  python compare_demo_vs_real_tick.py \
      --demo /workspace/data/XAUUSD/stratum_0_raw/XAUUSDm_202605200000_202605202359_demo.csv \
      --real /workspace/data/XAUUSD/stratum_0_raw/XAUUSDm_202605200000_202605202359_real.csv \
      [--out-dir /tmp/tick_diff_report]   # オプション: 詳細 parquet 出力

[期待される MT5 CSV フォーマット]
  タブ区切り + ヘッダ。s1_1_A_ingest.py が処理しているのと同じ:
  <DATE>\t<TIME>\t<BID>\t<ASK>\t<LAST>\t<VOLUME>\t<FLAGS>
  例: 2026.05.20\t00:00:00.123\t4458.500\t4459.300\t\t\t1
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import polars as pl

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("compare_demo_vs_real")


# ════════════════════════════════════════════════════════════════
# 1. CSV 読み込み — s1_1_A_ingest.py と同一の parse 仕様
# ════════════════════════════════════════════════════════════════
def load_tick_csv(path: Path, label: str) -> pl.DataFrame:
    """MT5 export CSV を polars DataFrame として読み込む。

    s1_1_A_ingest.py L65-100 と同じパース手順:
      - タブ区切り
      - ヘッダ 1 行スキップ
      - DATE + TIME 結合 → datetime(ms 精度)
      - bid/ask は Float64 (本番側に合わせて float の精度は保持)
    """
    logger.info(f"[{label}] 読み込み: {path}")
    if not path.exists():
        raise FileNotFoundError(f"{path} not found")

    # polars で直接タブ区切り読み込み
    # MT5 export は <DATE>\t<TIME>\t<BID>\t<ASK>\t<LAST>\t<VOLUME>\t<FLAGS>
    df = pl.read_csv(
        path,
        separator="\t",
        has_header=True,
        null_values=["", "NA", "NULL"],
        infer_schema_length=10000,
        # 列名はファイルの実際のヘッダを使う (<DATE> 等の山括弧付き)
    )

    # 列名を正規化 (山括弧を取り除く)
    rename_map = {c: c.strip("<>").lower() for c in df.columns}
    df = df.rename(rename_map)

    logger.info(f"[{label}] 読み込み完了: {len(df):,} 行, 列={df.columns}")

    # date + time を結合して datetime(ms) に
    # フォーマット: "2026.05.20" + " " + "00:00:00.123"
    df = df.with_columns(
        (pl.col("date") + pl.lit(" ") + pl.col("time"))
        .str.to_datetime(format="%Y.%m.%d %H:%M:%S%.f", time_unit="ms", strict=False)
        .alias("timestamp")
    )

    # bid/ask を Float64 にキャスト
    for col in ["bid", "ask", "last", "volume", "flags"]:
        if col in df.columns:
            df = df.with_columns(pl.col(col).cast(pl.Float64, strict=False))

    # 必須カラムだけ残す
    keep = ["timestamp", "bid", "ask"]
    if "last" in df.columns:
        keep.append("last")
    if "volume" in df.columns:
        keep.append("volume")
    if "flags" in df.columns:
        keep.append("flags")
    df = df.select(keep)

    # mid_price も追加 (s1_1_A と同じ式)
    df = df.with_columns(
        ((pl.col("bid") + pl.col("ask")) / 2.0).alias("mid_price")
    )

    # null timestamp や bid<=0 を除外
    n_before = len(df)
    df = df.filter(
        pl.col("timestamp").is_not_null()
        & (pl.col("bid") > 0)
        & (pl.col("ask") > 0)
    )
    if len(df) != n_before:
        logger.warning(
            f"[{label}] null/invalid 行を除外: {n_before:,} → {len(df):,} 行"
        )

    # ソート
    df = df.sort("timestamp")

    return df


# ════════════════════════════════════════════════════════════════
# 2. 件数差の内訳 (集合演算)
# ════════════════════════════════════════════════════════════════
def compare_counts(df_demo: pl.DataFrame, df_real: pl.DataFrame) -> dict:
    """timestamp 集合の差分を計算。

    Returns:
      dict with keys:
        common_count, demo_only_count, real_only_count,
        demo_only_df, real_only_df, common_demo, common_real
    """
    logger.info("=" * 70)
    logger.info("【セクション 1】 timestamp 集合差")
    logger.info("=" * 70)

    # 重複 ms 精度の timestamp を確認 (同じ ms に複数 tick がある可能性)
    demo_dup = len(df_demo) - df_demo["timestamp"].n_unique()
    real_dup = len(df_real) - df_real["timestamp"].n_unique()
    logger.info(
        f"  Demo: 全 {len(df_demo):,} 件, "
        f"unique ts {df_demo['timestamp'].n_unique():,}, "
        f"同 ms 重複 {demo_dup:,}"
    )
    logger.info(
        f"  Real: 全 {len(df_real):,} 件, "
        f"unique ts {df_real['timestamp'].n_unique():,}, "
        f"同 ms 重複 {real_dup:,}"
    )

    # 同 ms 重複がある場合、最初の 1 件だけ残して比較する (公平な join のため)
    demo_unique = df_demo.unique(subset=["timestamp"], keep="first", maintain_order=True)
    real_unique = df_real.unique(subset=["timestamp"], keep="first", maintain_order=True)

    # full outer join で 3 グループに分類
    joined = demo_unique.join(
        real_unique,
        on="timestamp",
        how="full",
        suffix="_real",
    )

    # demo-only = bid_real が null
    # real-only = bid が null
    # common = 両方非 null
    demo_only = joined.filter(pl.col("bid_real").is_null())
    real_only = joined.filter(pl.col("bid").is_null())
    common = joined.filter(
        pl.col("bid").is_not_null() & pl.col("bid_real").is_not_null()
    )

    logger.info("")
    logger.info(f"  [集合差]")
    logger.info(f"    共通 timestamp:   {len(common):>8,} 件")
    logger.info(f"    demo にのみ存在:  {len(demo_only):>8,} 件")
    logger.info(f"    real にのみ存在:  {len(real_only):>8,} 件")
    logger.info(f"    全体 (和集合):    {len(joined):>8,} 件")

    return {
        "common_count": len(common),
        "demo_only_count": len(demo_only),
        "real_only_count": len(real_only),
        "demo_only_df": demo_only,
        "real_only_df": real_only,
        "common": common,
    }


# ════════════════════════════════════════════════════════════════
# 3. 共通 timestamp の数値差 (bid/ask)
# ════════════════════════════════════════════════════════════════
def analyze_value_diff(common: pl.DataFrame) -> None:
    """共通 timestamp での bid/ask/mid_price の数値差を分析。"""
    logger.info("")
    logger.info("=" * 70)
    logger.info("【セクション 2】 共通 timestamp の bid/ask/mid 数値差")
    logger.info("=" * 70)

    if len(common) == 0:
        logger.warning("  共通 timestamp が 0 件 — 完全に独立した tick stream")
        return

    # 差分計算
    common = common.with_columns([
        (pl.col("bid") - pl.col("bid_real")).alias("bid_diff"),
        (pl.col("ask") - pl.col("ask_real")).alias("ask_diff"),
        (pl.col("mid_price") - pl.col("mid_price_real")).alias("mid_diff"),
    ])

    # bit-identical 件数 (全て 0)
    n_identical_bid = (common["bid_diff"] == 0).sum()
    n_identical_ask = (common["ask_diff"] == 0).sum()
    n_identical_both = ((common["bid_diff"] == 0) & (common["ask_diff"] == 0)).sum()

    pct_identical_bid = n_identical_bid / len(common) * 100
    pct_identical_ask = n_identical_ask / len(common) * 100
    pct_identical_both = n_identical_both / len(common) * 100

    logger.info(f"  全 {len(common):,} 件中:")
    logger.info(
        f"    bid 完全一致:        {n_identical_bid:>8,} 件 ({pct_identical_bid:.4f}%)"
    )
    logger.info(
        f"    ask 完全一致:        {n_identical_ask:>8,} 件 ({pct_identical_ask:.4f}%)"
    )
    logger.info(
        f"    bid+ask 両方一致:    {n_identical_both:>8,} 件 ({pct_identical_both:.4f}%)"
    )

    # 数値差の統計 (差がある行のみ)
    diff_rows = common.filter(pl.col("bid_diff") != 0)
    if len(diff_rows) > 0:
        logger.info("")
        logger.info(f"  [bid 数値差統計 — 差がある {len(diff_rows):,} 件]")
        bid_diff_np = diff_rows["bid_diff"].to_numpy()
        logger.info(f"    mean:    {bid_diff_np.mean():+.6f}")
        logger.info(f"    median:  {np.median(bid_diff_np):+.6f}")
        logger.info(f"    std:     {bid_diff_np.std():.6f}")
        logger.info(f"    min:     {bid_diff_np.min():+.6f}")
        logger.info(f"    max:     {bid_diff_np.max():+.6f}")
        logger.info(
            f"    |diff|>0.01: {(np.abs(bid_diff_np) > 0.01).sum():,} 件 "
            f"({(np.abs(bid_diff_np) > 0.01).sum() / len(bid_diff_np) * 100:.2f}%)"
        )
        logger.info(
            f"    |diff|>0.10: {(np.abs(bid_diff_np) > 0.10).sum():,} 件 "
            f"({(np.abs(bid_diff_np) > 0.10).sum() / len(bid_diff_np) * 100:.2f}%)"
        )
        logger.info(
            f"    |diff|>1.00: {(np.abs(bid_diff_np) > 1.00).sum():,} 件 "
            f"({(np.abs(bid_diff_np) > 1.00).sum() / len(bid_diff_np) * 100:.2f}%)"
        )

        # mid_price の差 (これが特徴量に直接効く)
        logger.info("")
        logger.info(f"  [mid_price 差統計]")
        mid_diff_np = diff_rows["mid_diff"].to_numpy()
        logger.info(f"    mean:    {mid_diff_np.mean():+.6f}")
        logger.info(f"    median:  {np.median(mid_diff_np):+.6f}")
        logger.info(f"    std:     {mid_diff_np.std():.6f}")
        logger.info(f"    min:     {mid_diff_np.min():+.6f}")
        logger.info(f"    max:     {mid_diff_np.max():+.6f}")

        # 系統的バイアスの検出
        bias_threshold = 1e-3
        if abs(bid_diff_np.mean()) > bias_threshold:
            logger.warning(
                f"  ⚠️ 系統的バイアス検出: bid 平均差 = {bid_diff_np.mean():+.6f} "
                f"({'demo が常に高い' if bid_diff_np.mean() > 0 else 'real が常に高い'})"
            )


# ════════════════════════════════════════════════════════════════
# 4. 時間帯別の件数差
# ════════════════════════════════════════════════════════════════
def analyze_hourly_distribution(
    df_demo: pl.DataFrame, df_real: pl.DataFrame
) -> None:
    """時間帯 (1 時間ごと) の件数を比較。"""
    logger.info("")
    logger.info("=" * 70)
    logger.info("【セクション 3】 時間帯別の Tick 件数差")
    logger.info("=" * 70)

    demo_hourly = (
        df_demo.with_columns(pl.col("timestamp").dt.hour().alias("hour"))
        .group_by("hour")
        .len()
        .sort("hour")
        .rename({"len": "demo_count"})
    )
    real_hourly = (
        df_real.with_columns(pl.col("timestamp").dt.hour().alias("hour"))
        .group_by("hour")
        .len()
        .sort("hour")
        .rename({"len": "real_count"})
    )
    joined = demo_hourly.join(real_hourly, on="hour", how="full", coalesce=True).sort("hour")
    # [FIX] polars len() は UInt32 を返すので subtract で巻き戻る。Int64 にキャストしてから引く。
    joined = joined.with_columns(
        (
            pl.col("demo_count").fill_null(0).cast(pl.Int64)
            - pl.col("real_count").fill_null(0).cast(pl.Int64)
        ).alias("diff")
    )

    logger.info(
        f"  {'hour':>6} {'demo':>10} {'real':>10} {'diff':>10}  bar"
    )
    logger.info("  " + "-" * 60)
    for row in joined.iter_rows(named=True):
        h = row["hour"]
        d = row["demo_count"] or 0
        r = row["real_count"] or 0
        diff = row["diff"] or 0
        # ビジュアル: diff の絶対値を棒で表現
        max_abs = joined.select(pl.col("diff").abs().max()).item() or 1
        bar_len = int(abs(diff) / max_abs * 30) if max_abs > 0 else 0
        bar = ("+" if diff >= 0 else "-") * bar_len
        logger.info(
            f"  {h:>6} {d:>10,} {r:>10,} {diff:>+10,}  {bar}"
        )


# ════════════════════════════════════════════════════════════════
# 5. M0.5 集約後の OHLCV 比較 — これが本番への実害判定
# ════════════════════════════════════════════════════════════════
def compare_m05_aggregated(
    df_demo: pl.DataFrame, df_real: pl.DataFrame
) -> None:
    """30 秒バケット集約後の OHLCV を比較。

    s1_1_B_build_ohlcv.py / EA CollectM05Bar の集約ロジックと同一:
      bucket_key = (timestamp_ns // 30_000_000_000) * 30_000_000_000
      OHLCV: first/max/min/last of mid_price, count of ticks
    """
    logger.info("")
    logger.info("=" * 70)
    logger.info("【セクション 4】 M0.5 (30 秒足) 集約後の OHLCV 差")
    logger.info("    ← これが本番への実害判定: Tick が違っても集約後に同じなら影響小")
    logger.info("=" * 70)

    def aggregate_m05(df: pl.DataFrame) -> pl.DataFrame:
        # 30 秒 floor 集約 (s1_1_B と同じ整数除算)
        # [FIX] polars の Datetime.cast(Int64) は内部 unit に依存する。
        # 学習側 (s1_1_A) は ms 精度、ここではそれを ns 比較して 30 秒バケットに丸める。
        # ms → ns 変換: × 1_000_000、30 秒 = 30_000_000_000 ns
        # → これを ms に戻すために最終 / 1_000_000 し Datetime("ms") にキャスト
        # [FIX] group_by の first/last が timestamp 順 を保つよう、事前に sort
        return (
            df.sort("timestamp")
            .with_columns(
                # timestamp (Datetime ms) → Int64 (ms) → 30 秒 floor → ms に戻す
                (
                    (pl.col("timestamp").cast(pl.Int64) // 30_000)
                    * 30_000  # ms 単位での 30 秒 floor
                )
                .cast(pl.Datetime("ms"))
                .alias("bucket")
            )
            .group_by("bucket", maintain_order=True)
            .agg([
                pl.col("mid_price").first().alias("open"),
                pl.col("mid_price").max().alias("high"),
                pl.col("mid_price").min().alias("low"),
                pl.col("mid_price").last().alias("close"),
                pl.col("mid_price").count().alias("tick_count"),
            ])
            .sort("bucket")
        )

    demo_m05 = aggregate_m05(df_demo)
    real_m05 = aggregate_m05(df_real)

    logger.info(f"  Demo M0.5 集約: {len(demo_m05):,} bars")
    logger.info(f"  Real M0.5 集約: {len(real_m05):,} bars")

    # bucket で join
    joined_m05 = demo_m05.join(
        real_m05, on="bucket", how="full", suffix="_real", coalesce=True
    ).sort("bucket")

    # 完全 demo-only / real-only bucket
    demo_only_buckets = joined_m05.filter(pl.col("open_real").is_null())
    real_only_buckets = joined_m05.filter(pl.col("open").is_null())
    common_buckets = joined_m05.filter(
        pl.col("open").is_not_null() & pl.col("open_real").is_not_null()
    )

    logger.info("")
    logger.info(f"  [bucket 集合差]")
    logger.info(f"    共通 bucket:       {len(common_buckets):>8,}")
    logger.info(f"    demo のみ bucket:  {len(demo_only_buckets):>8,}")
    logger.info(f"    real のみ bucket:  {len(real_only_buckets):>8,}")

    if len(common_buckets) == 0:
        logger.warning("  共通 bucket が 0 件 — M0.5 段階で完全乖離")
        return

    # 共通 bucket での OHLCV 差を集計
    common_buckets = common_buckets.with_columns([
        (pl.col("open") - pl.col("open_real")).alias("open_diff"),
        (pl.col("high") - pl.col("high_real")).alias("high_diff"),
        (pl.col("low") - pl.col("low_real")).alias("low_diff"),
        (pl.col("close") - pl.col("close_real")).alias("close_diff"),
        # [FIX] tick_count は polars count() で UInt32 を返すため Int64 にキャストしてから引く
        (
            pl.col("tick_count").cast(pl.Int64)
            - pl.col("tick_count_real").cast(pl.Int64)
        ).alias("tickcount_diff"),
    ])

    for name in ["open", "high", "low", "close"]:
        diff_col = f"{name}_diff"
        diff_np = common_buckets[diff_col].to_numpy()
        n_identical = (diff_np == 0).sum()
        n_nonzero = (diff_np != 0).sum()
        logger.info("")
        logger.info(f"  [{name}]:")
        logger.info(
            f"    完全一致:  {n_identical:>8,} bars ({n_identical/len(common_buckets)*100:.4f}%)"
        )
        logger.info(
            f"    不一致:    {n_nonzero:>8,} bars ({n_nonzero/len(common_buckets)*100:.4f}%)"
        )
        if n_nonzero > 0:
            nz = diff_np[diff_np != 0]
            logger.info(
                f"    差 (mean/median/std/min/max): "
                f"{nz.mean():+.6f} / {np.median(nz):+.6f} / "
                f"{nz.std():.6f} / {nz.min():+.6f} / {nz.max():+.6f}"
            )

    # tick_count 差
    tc_diff_np = common_buckets["tickcount_diff"].to_numpy()
    n_tc_identical = (tc_diff_np == 0).sum()
    logger.info("")
    logger.info(f"  [tick_count] (= M0.5 内 Tick 数):")
    logger.info(
        f"    完全一致:  {n_tc_identical:>8,} bars ({n_tc_identical/len(common_buckets)*100:.4f}%)"
    )
    if n_tc_identical < len(common_buckets):
        logger.info(
            f"    diff (mean/median/min/max): "
            f"{tc_diff_np.mean():+.3f} / {np.median(tc_diff_np):+.3f} / "
            f"{tc_diff_np.min():+d} / {tc_diff_np.max():+d}"
        )

    return common_buckets, demo_only_buckets, real_only_buckets


# ════════════════════════════════════════════════════════════════
# 6. main
# ════════════════════════════════════════════════════════════════
def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--demo", type=Path, required=True, help="デモ口座 Tick CSV")
    p.add_argument("--real", type=Path, required=True, help="リアル口座 Tick CSV")
    p.add_argument(
        "--out-dir", type=Path, default=None,
        help="詳細 parquet 出力 dir (省略時は出力しない)"
    )
    args = p.parse_args()

    logger.info("=" * 70)
    logger.info("デモ vs リアル Tick データ差分検証")
    logger.info("=" * 70)

    # 1. 読み込み
    df_demo = load_tick_csv(args.demo, "DEMO")
    df_real = load_tick_csv(args.real, "REAL")

    logger.info("")
    logger.info(
        f"Demo 期間: {df_demo['timestamp'].min()} 〜 {df_demo['timestamp'].max()}"
    )
    logger.info(
        f"Real 期間: {df_real['timestamp'].min()} 〜 {df_real['timestamp'].max()}"
    )

    # 2. 件数差
    counts = compare_counts(df_demo, df_real)

    # 3. 共通 timestamp の数値差
    analyze_value_diff(counts["common"])

    # 4. 時間帯別
    analyze_hourly_distribution(df_demo, df_real)

    # 5. M0.5 集約後
    m05_result = compare_m05_aggregated(df_demo, df_real)

    # 6. 詳細 parquet 出力
    if args.out_dir is not None:
        args.out_dir.mkdir(parents=True, exist_ok=True)
        # demo-only / real-only / common (差分付き) の Tick リスト
        counts["demo_only_df"].select(
            ["timestamp", "bid", "ask", "mid_price"]
        ).write_parquet(args.out_dir / "tick_demo_only.parquet")
        counts["real_only_df"].select(
            ["timestamp", "bid_real", "ask_real", "mid_price_real"]
        ).rename({"bid_real": "bid", "ask_real": "ask", "mid_price_real": "mid_price"}).write_parquet(
            args.out_dir / "tick_real_only.parquet"
        )
        counts["common"].write_parquet(args.out_dir / "tick_common_with_diff.parquet")
        if m05_result is not None:
            common_m05, demo_only_m05, real_only_m05 = m05_result
            common_m05.write_parquet(args.out_dir / "m05_common_with_diff.parquet")
            demo_only_m05.write_parquet(args.out_dir / "m05_demo_only.parquet")
            real_only_m05.write_parquet(args.out_dir / "m05_real_only.parquet")
        logger.info("")
        logger.info(f"✓ 詳細 parquet を {args.out_dir} に出力しました")

    # 7. 総合判定
    logger.info("")
    logger.info("=" * 70)
    logger.info("【総合判定】")
    logger.info("=" * 70)
    n_common = counts["common_count"]
    n_demo_only = counts["demo_only_count"]
    n_real_only = counts["real_only_count"]
    total = n_common + n_demo_only + n_real_only

    if total == 0:
        logger.warning("空 — 検証不能")
        return

    pct_common = n_common / total * 100
    if pct_common > 99.5:
        verdict = "✓ ほぼ同一: デモ/リアル Tick は同じ source"
    elif pct_common > 90:
        verdict = "△ 大部分一致だが小さい差あり: 個別 Tick の有無で差"
    elif pct_common > 50:
        verdict = "⚠ 半分程度一致: 異なる sampling lattice の可能性大"
    else:
        verdict = "❌ ほぼ別物: Tick stream が独立、Train-Serve Skew の真因候補"
    logger.info(f"  共通率: {pct_common:.2f}% — {verdict}")


if __name__ == "__main__":
    main()
