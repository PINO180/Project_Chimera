#!/usr/bin/env python3
"""
compare_demo_vs_real_m1.py — デモ口座 vs リアル口座 M1 OHLCV データの差分検証

[目的]
  Tick 比較 (compare_demo_vs_real_tick.py) で「同じ日のデモ tick とリアル
  tick が全く違う」ことが判明した (共通率 46%、bid 完全一致 2.3%、systematic
  bias -1.4 pip)。

  ここで決定的に重要な検証: **MT5 サーバ側で集約された M1 OHLCV 自体が
  デモとリアルで違うのか?** これが違うなら、Tick だけでなく集約後 OHLC
  そのものがブローカー側で別 source として配信されている確証になる。

  逆にここで M1 OHLCV が完全一致するなら、「Tick 単発は違うが MT5 が
  サーバ側で集約したバーは同一」= ブローカー API のレイヤーが broker 側
  で統合されているという別の解釈が成り立つ。

[期待される MT5 CSV フォーマット (M1)]
  タブ区切り + ヘッダ:
    <DATE>\t<TIME>\t<OPEN>\t<HIGH>\t<LOW>\t<CLOSE>\t<TICKVOL>\t<VOL>\t<SPREAD>

  ※ Tick エクスポートとは異なるフォーマット。bid/ask の代わりに OHLC。

[使い方]
  python compare_demo_vs_real_m1.py \
      --demo /workspace/data/XAUUSD/stratum_0_raw/XAUUSDm_M1_202605200000_202605210000_demo_1379.csv \
      --real /workspace/data/XAUUSD/stratum_0_raw/XAUUSDm_M1_202605200000_202605210000_real_1379.csv
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
logger = logging.getLogger("compare_demo_vs_real_m1")


# ════════════════════════════════════════════════════════════════
# 1. M1 CSV 読み込み
# ════════════════════════════════════════════════════════════════
def load_m1_csv(path: Path, label: str) -> pl.DataFrame:
    """MT5 M1 export CSV を読み込む。

    フォーマット (典型的な MT5 export):
      <DATE>\t<TIME>\t<OPEN>\t<HIGH>\t<LOW>\t<CLOSE>\t<TICKVOL>\t<VOL>\t<SPREAD>
    """
    logger.info(f"[{label}] 読み込み: {path}")
    if not path.exists():
        raise FileNotFoundError(f"{path} not found")

    df = pl.read_csv(
        path,
        separator="\t",
        has_header=True,
        null_values=["", "NA", "NULL"],
        infer_schema_length=10000,
    )

    # 列名を正規化 (山括弧除去・小文字化)
    rename_map = {c: c.strip("<>").lower() for c in df.columns}
    df = df.rename(rename_map)

    logger.info(f"[{label}] 読み込み完了: {len(df):,} 行, 列={df.columns}")

    # date + time → datetime
    df = df.with_columns(
        (pl.col("date") + pl.lit(" ") + pl.col("time"))
        .str.to_datetime(format="%Y.%m.%d %H:%M:%S", time_unit="ms", strict=False)
        .alias("timestamp")
    )

    # 数値カラムをキャスト
    for col in ["open", "high", "low", "close", "tickvol", "vol", "spread"]:
        if col in df.columns:
            df = df.with_columns(pl.col(col).cast(pl.Float64, strict=False))

    # 必須カラムだけ残す
    keep = ["timestamp", "open", "high", "low", "close"]
    if "tickvol" in df.columns:
        keep.append("tickvol")
    if "vol" in df.columns:
        keep.append("vol")
    if "spread" in df.columns:
        keep.append("spread")
    df = df.select(keep)

    # invalid 除外
    n_before = len(df)
    df = df.filter(
        pl.col("timestamp").is_not_null()
        & (pl.col("open") > 0)
        & (pl.col("close") > 0)
    )
    if len(df) != n_before:
        logger.warning(
            f"[{label}] invalid 行を除外: {n_before:,} → {len(df):,} 行"
        )

    df = df.sort("timestamp")
    return df


# ════════════════════════════════════════════════════════════════
# 2. 比較本体
# ════════════════════════════════════════════════════════════════
def compare_m1(df_demo: pl.DataFrame, df_real: pl.DataFrame) -> None:
    logger.info("")
    logger.info("=" * 70)
    logger.info("M1 OHLCV 集合差")
    logger.info("=" * 70)

    # full outer join
    joined = df_demo.join(
        df_real, on="timestamp", how="full", suffix="_real", coalesce=True
    ).sort("timestamp")

    common = joined.filter(
        pl.col("open").is_not_null() & pl.col("open_real").is_not_null()
    )
    demo_only = joined.filter(pl.col("open_real").is_null())
    real_only = joined.filter(pl.col("open").is_null())

    logger.info(f"  Demo M1 件数: {len(df_demo):,}")
    logger.info(f"  Real M1 件数: {len(df_real):,}")
    logger.info(f"  共通 timestamp:    {len(common):>6,}")
    logger.info(f"  demo のみ:         {len(demo_only):>6,}")
    logger.info(f"  real のみ:         {len(real_only):>6,}")

    if len(common) == 0:
        logger.warning("  共通 timestamp が 0 件 — M1 タイムスタンプ自体ズレ")
        return

    # 各 OHLC の比較
    common = common.with_columns([
        (pl.col("open") - pl.col("open_real")).alias("open_diff"),
        (pl.col("high") - pl.col("high_real")).alias("high_diff"),
        (pl.col("low") - pl.col("low_real")).alias("low_diff"),
        (pl.col("close") - pl.col("close_real")).alias("close_diff"),
    ])

    logger.info("")
    logger.info("=" * 70)
    logger.info("共通 timestamp の OHLC 値差")
    logger.info("=" * 70)
    for name in ["open", "high", "low", "close"]:
        diff_col = f"{name}_diff"
        diff_np = common[diff_col].to_numpy()
        n_identical = int((diff_np == 0).sum())
        n_total = len(diff_np)
        pct_identical = n_identical / n_total * 100
        logger.info("")
        logger.info(f"  [{name}]: 全 {n_total:,} bars")
        logger.info(
            f"    完全一致:    {n_identical:>6,} bars ({pct_identical:.4f}%)"
        )
        n_nonzero = n_total - n_identical
        if n_nonzero > 0:
            nz = diff_np[diff_np != 0]
            logger.info(
                f"    不一致:      {n_nonzero:>6,} bars ({100-pct_identical:.4f}%)"
            )
            logger.info(
                f"    差統計 (mean/median/std/min/max): "
                f"{nz.mean():+.6f} / {float(np.median(nz)):+.6f} / "
                f"{nz.std():.6f} / {nz.min():+.6f} / {nz.max():+.6f}"
            )
            # |diff| > 閾値別件数
            for thr in [0.001, 0.01, 0.1, 1.0]:
                n_over = int((np.abs(nz) > thr).sum())
                logger.info(
                    f"    |diff|>{thr:6.3f}:  {n_over:>6,} bars "
                    f"({n_over/n_nonzero*100:.2f}% of nonzero)"
                )

    # tickvol / spread 比較 (列が存在すれば)
    if "tickvol" in common.columns and "tickvol_real" in common.columns:
        logger.info("")
        logger.info("=" * 70)
        logger.info("tickvol (Tick volume) 差")
        logger.info("=" * 70)
        tv_diff = (
            common["tickvol"].cast(pl.Int64) - common["tickvol_real"].cast(pl.Int64)
        ).to_numpy()
        n_identical = int((tv_diff == 0).sum())
        logger.info(f"  完全一致: {n_identical:,} / {len(tv_diff):,} bars "
                    f"({n_identical/len(tv_diff)*100:.4f}%)")
        if n_identical < len(tv_diff):
            logger.info(
                f"  差 (mean/median/std/min/max): "
                f"{tv_diff.mean():+.3f} / {float(np.median(tv_diff)):+.3f} / "
                f"{tv_diff.std():.3f} / {tv_diff.min():+d} / {tv_diff.max():+d}"
            )

    if "spread" in common.columns and "spread_real" in common.columns:
        logger.info("")
        logger.info("=" * 70)
        logger.info("spread 差")
        logger.info("=" * 70)
        sp_diff = (
            common["spread"].cast(pl.Int64) - common["spread_real"].cast(pl.Int64)
        ).to_numpy()
        n_identical = int((sp_diff == 0).sum())
        logger.info(f"  完全一致: {n_identical:,} / {len(sp_diff):,} bars "
                    f"({n_identical/len(sp_diff)*100:.4f}%)")
        if n_identical < len(sp_diff):
            logger.info(
                f"  差 (mean/median/std/min/max): "
                f"{sp_diff.mean():+.3f} / {float(np.median(sp_diff)):+.3f} / "
                f"{sp_diff.std():.3f} / {sp_diff.min():+d} / {sp_diff.max():+d}"
            )

    # 総合判定
    logger.info("")
    logger.info("=" * 70)
    logger.info("【総合判定】")
    logger.info("=" * 70)

    # 全 OHLC が完全一致した bar 数
    all_match = (
        (common["open_diff"] == 0)
        & (common["high_diff"] == 0)
        & (common["low_diff"] == 0)
        & (common["close_diff"] == 0)
    ).sum()
    pct_all_match = int(all_match) / len(common) * 100

    logger.info(
        f"  全 OHLC 完全一致 bars: {int(all_match):,} / {len(common):,} "
        f"({pct_all_match:.4f}%)"
    )

    if pct_all_match > 99.9:
        verdict = "✓ ほぼ完全一致: MT5 サーバ側で集約された M1 は demo/real で同一"
    elif pct_all_match > 90:
        verdict = "△ 大部分一致: 微小な差。集約後はかなり近い"
    elif pct_all_match > 50:
        verdict = "⚠ 半分程度一致: M1 集約後もそれなりの差異"
    else:
        verdict = "❌ ほぼ別物: MT5 集約後の M1 も demo/real で全く違う"
    logger.info(f"  → {verdict}")


# ════════════════════════════════════════════════════════════════
# 3. main
# ════════════════════════════════════════════════════════════════
def main():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--demo", type=Path, required=True)
    p.add_argument("--real", type=Path, required=True)
    args = p.parse_args()

    logger.info("=" * 70)
    logger.info("デモ vs リアル M1 OHLCV データ差分検証")
    logger.info("=" * 70)

    df_demo = load_m1_csv(args.demo, "DEMO")
    df_real = load_m1_csv(args.real, "REAL")

    logger.info("")
    logger.info(
        f"Demo 期間: {df_demo['timestamp'].min()} 〜 {df_demo['timestamp'].max()}"
    )
    logger.info(
        f"Real 期間: {df_real['timestamp'].min()} 〜 {df_real['timestamp'].max()}"
    )

    compare_m1(df_demo, df_real)


if __name__ == "__main__":
    main()
