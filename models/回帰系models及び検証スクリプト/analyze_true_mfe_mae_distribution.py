# /workspace/models/analyze_true_mfe_mae_distribution.py
# ============================================================================
# 【真の MFE/MAE 分布 観測ツール (専用ラベリング)】
# ----------------------------------------------------------------------------
# 目的:
#   各 M3 バーのエントリー地点 (L+180) から、"十分長い" 観測窓で、価格が
#   上に最大どれだけ (真のMFE)・下に最大どれだけ (真のMAE) 動いたか、そして
#   それぞれ何分で到達したか (到達時刻) を tick 走査で測り、分布を出す。
#
#   TP/SL・先着判定・方向ラベル・スプレッド・学習は一切無し。純粋な"観測"。
#   → 真の MFE/MAE の大きさと到達時刻の分布から、TO窓の最適値を人間が判断する材料。
#   (net_dominance / edge 等の学習ターゲット設計は、この観測の"後"の別工程)
#
# 出力:
#   1. parquet (各バーの生データ)  … 後で任意の窓・条件で再分析できるよう網羅的に保存
#   2. テキストレポート (分布)      … BT のレポートと同様。MFE/MAE値・到達時刻・
#                                     セッション別・ATR帯別・曜日別・時間帯別。
#   出力先: stratum_6_training/true_mfe_mae_analysis/
#
# ----------------------------------------------------------------------------
# 【実行コマンド例】
#   フル実行 (全期間):
#       python models/analyze_true_mfe_mae_distribution.py
#   月指定 (1ヶ月だけ):
#       python models/analyze_true_mfe_mae_distribution.py --year-month 2021/7
#   年指定 (1年だけ):
#       python models/analyze_true_mfe_mae_distribution.py --year 2021
#   ※ /workspace から実行 (models/ プレフィックス)。--year-month は YYYY/M 形式。
# ============================================================================
# 【調整パラメータ (ここを直接編集して調整する)】
# ----------------------------------------------------------------------------
#   TO_MINUTES : 観測窓(分)。エントリーから何分先まで最大順行/逆行を測るか。
#       真の MFE/MAE を見るため長めに取る。ベース=240分(4時間)。
#       一度長く測っておけば、後で分布から「30分で切ったら/60分で切ったら」を
#       判断できる。もっと長くしたければ増やす(tick走査は重くなる)。
#   ACTION_HORIZON_SEC : エントリーオフセット(秒)。本ラベリングと一致 (=180)。
#
#   *_BINS : テキストレポートの分布の帯 (下記で編集可)。
# ============================================================================
TO_MINUTES = 30
ACTION_HORIZON_SEC = 180

# MFE/MAE(ATR建て)の分布の帯 (下限値のリスト。例: 0.5未満, 0.5-1.0, ...)
MFE_MAE_ATR_BINS = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 5.0]  # 最後の帯は ">= 5.0"
# 到達時刻(分)の分布の帯
REACH_MIN_BINS = [0, 1, 5, 10, 20, 30, 45, 60, 90, 120, 180, 240, 360, 720, 1080]
# ATR ratio 帯 (セッション/ボラ別分析用)
ATR_RATIO_BINS = [0.0, 0.5, 0.8, 1.0, 1.2, 1.5, 2.0, 3.0]

import sys
import logging
import datetime as dt
from pathlib import Path
from typing import Tuple

import numpy as np
import polars as pl

try:
    from numba import njit, prange

    _NUMBA = True
except Exception:
    _NUMBA = False

    def njit(*a, **k):
        def deco(f):
            return f

        return deco if not (a and callable(a[0])) else a[0]

    def prange(*a):
        return range(*a)


# blueprint (= /workspace/blueprint.py) を import 可能にする
_project_root = Path(__file__).resolve().parents[1]
if str(_project_root) not in sys.path:
    sys.path.append(str(_project_root))

from blueprint import (
    S6_LABELED_DATASET,
    S1_RAW_TICK_PARTITIONED,
)

# 出力先: stratum_6_training/true_mfe_mae_analysis
OUT_DIR = S6_LABELED_DATASET.parent / "true_mfe_mae_analysis"
PARQUET_DIR = OUT_DIR / "per_bar"
REPORT_PATH = OUT_DIR / "true_mfe_mae_report.txt"

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


def _njit_if_available(func):
    if _NUMBA:
        return njit(func, parallel=True, fastmath=True, cache=True)
    return func


@_njit_if_available
def _numba_mfe_mae_reachtime(
    bets_t0: np.ndarray,
    bets_window_end: np.ndarray,
    ticks_ts: np.ndarray,
    ticks_price: np.ndarray,
    entry_offset_us: np.int64,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    エントリー後 (t0+offset, window_end] の窓で mid_price の最高値・最安値と、
    それぞれに"最初に到達した時刻(us)"を返す。既存 _numba_mfe_mae の到達時刻版。
    早期退出なし (窓を最後まで走査)。tick は単一点 (high=low=mid)。
    返り値: (out_max, out_min, out_max_time, out_min_time)。
      窓内 tick 無しは max/min=NaN, time=0。
    """
    n_bets = len(bets_t0)
    n_ticks = len(ticks_ts)
    out_max = np.full(n_bets, np.nan)
    out_min = np.full(n_bets, np.nan)
    out_max_t = np.zeros(n_bets, dtype=np.int64)
    out_min_t = np.zeros(n_bets, dtype=np.int64)
    if n_ticks == 0:
        return out_max, out_min, out_max_t, out_min_t

    for i in prange(n_bets):
        entry_ts = bets_t0[i] + entry_offset_us
        w_end = bets_window_end[i]
        start_idx = np.searchsorted(ticks_ts, entry_ts, side="right")
        mx = -np.inf
        mn = np.inf
        mxt = 0
        mnt = 0
        found = False
        for j in range(start_idx, n_ticks):
            ts = ticks_ts[j]
            if ts > w_end:
                break
            p = ticks_price[j]
            if p > mx:
                mx = p
                mxt = ts  # 最高値に(最初に)到達した時刻
            if p < mn:
                mn = p
                mnt = ts  # 最安値に(最初に)到達した時刻
            found = True
        if found:
            out_max[i] = mx
            out_min[i] = mn
            out_max_t[i] = mxt
            out_min_t[i] = mnt
    return out_max, out_min, out_max_t, out_min_t


# ── セッション判定 (JST時間帯基準。BTと同一定義) ──
def _hour_to_session_jst(h: int) -> str:
    if 9 <= h < 16:
        return "Tokyo"
    elif 16 <= h < 21:
        return "London"
    elif h >= 21 or h < 1:
        return "Overlap"
    elif 1 <= h < 6:
        return "NY"
    else:
        return "Oceania"


def _load_bars() -> pl.DataFrame:
    """S6 から M3 バー (timestamp, close=エントリー価格, atr_value, atr_ratio,
    session_atr_ratio があれば) を読む。"""
    logging.info("Loading M3 bars from S6 (timestamp/close/atr_value/atr_ratio)...")
    lf = pl.scan_parquet(str(S6_LABELED_DATASET / "**/*.parquet"))
    have = lf.collect_schema().names()
    cols = ["timestamp", "close", "atr_value"]
    for opt in ["atr_ratio", "session_atr_ratio"]:
        if opt in have:
            cols.append(opt)
    df = (
        lf.select(cols)
        .collect()
        .with_columns(pl.col("timestamp").cast(pl.Datetime("us", "UTC")))
        .unique("timestamp", keep="first")
        .sort("timestamp")
    )
    if "atr_ratio" not in df.columns:
        df = df.with_columns(pl.lit(None, dtype=pl.Float64).alias("atr_ratio"))
    logging.info(f"  -> {df.height} bars")
    return df


def run(filter_year: int = None, filter_month: int = None) -> None:
    logging.info("### 真の MFE/MAE 分布 観測 開始 ###")
    logging.info(
        f"params: TO_MINUTES={TO_MINUTES}, ACTION_HORIZON_SEC={ACTION_HORIZON_SEC}"
    )
    if filter_year is not None:
        logging.info(
            f"範囲フィルタ: year={filter_year}"
            + (f", month={filter_month}" if filter_month is not None else "")
        )

    bars = _load_bars()
    bars = bars.with_columns(
        pl.col("timestamp").dt.year().alias("_y"),
        pl.col("timestamp").dt.month().alias("_m"),
    )
    ym_pairs = bars.select(["_y", "_m"]).unique().sort(["_y", "_m"]).rows()
    if filter_year is not None:
        ym_pairs = [
            (y, m)
            for (y, m) in ym_pairs
            if y == filter_year and (filter_month is None or m == filter_month)
        ]
        if not ym_pairs:
            logging.warning("範囲フィルタに合致する月がありません。")
            return

    entry_offset_us = np.int64(ACTION_HORIZON_SEC * 1_000_000)
    window_us = np.int64(TO_MINUTES * 60 * 1_000_000)
    PARQUET_DIR.mkdir(parents=True, exist_ok=True)

    all_parts = []
    for y, m in ym_pairs:
        month_df = bars.filter((pl.col("_y") == y) & (pl.col("_m") == m))
        if month_df.is_empty():
            continue

        # tick 読込 (hive 枝刈り。窓が長いので当月〜数ヶ月先まで見る必要がある)
        month_start = dt.datetime(y, m, 1, tzinfo=dt.timezone.utc)
        # 窓終端の余裕 (TO_MINUTES + offset + バッファ)。長い窓ぶん翌月以降も要る。
        span_days = (
            int(np.ceil((TO_MINUTES + ACTION_HORIZON_SEC / 60.0) / (60 * 24))) + 2
        )
        last_bar = month_df["timestamp"].max()
        month_end = last_bar + dt.timedelta(
            minutes=TO_MINUTES + ACTION_HORIZON_SEC / 60.0 + 5
        )

        # 走査に必要な (year,month) の集合を列挙 (当月と、窓がまたぐ翌月群)
        ym_needed = set()
        cur = dt.datetime(y, m, 1, tzinfo=dt.timezone.utc)
        end_probe = month_end + dt.timedelta(days=1)
        while cur <= end_probe:
            ym_needed.add((cur.year, cur.month))
            # 次の月へ
            if cur.month == 12:
                cur = dt.datetime(cur.year + 1, 1, 1, tzinfo=dt.timezone.utc)
            else:
                cur = dt.datetime(cur.year, cur.month + 1, 1, tzinfo=dt.timezone.utc)

        try:
            ticks = (
                pl.scan_parquet(
                    str(S1_RAW_TICK_PARTITIONED / "**/*.parquet"),
                    hive_partitioning=True,
                )
                .filter(
                    pl.struct(["year", "month"]).is_in(
                        [{"year": yy, "month": mm} for (yy, mm) in sorted(ym_needed)]
                    )
                )
                .rename({"datetime": "timestamp"})
                .select("timestamp", "mid_price")
                .with_columns(pl.col("timestamp").cast(pl.Datetime("us", "UTC")))
                .filter(pl.col("timestamp").is_between(month_start, month_end))
                .collect()
                .unique("timestamp", keep="first")
                .sort("timestamp")
            )
        except Exception as e:
            logging.error(f"{y}-{m:02d}: tick load failed: {e}")
            continue

        if ticks.is_empty():
            logging.warning(f"{y}-{m:02d}: no ticks, skip.")
            continue

        ticks_ts = ticks["timestamp"].cast(pl.Int64).to_numpy()
        ticks_mid = ticks["mid_price"].to_numpy()

        bets_t0 = month_df["timestamp"].cast(pl.Int64).to_numpy()
        w_end = bets_t0 + entry_offset_us + window_us

        omax, omin, omaxt, omint = _numba_mfe_mae_reachtime(
            bets_t0, w_end, ticks_ts, ticks_mid, entry_offset_us
        )

        entry_ts_arr = bets_t0 + int(entry_offset_us)
        res = month_df.with_columns(
            pl.Series("_mfe_max_mid", omax),
            pl.Series("_mae_min_mid", omin),
            pl.Series("_mfe_time_us", omaxt),
            pl.Series("_mae_time_us", omint),
            pl.Series("_entry_ts", entry_ts_arr),
        )
        res = _finalize_month(res)
        _persist_month(res, y, m)
        all_parts.append(res)
        logging.info(
            f"{y}-{m:02d}: bars={res.height} "
            f"(MFE平均={res['mfe_atr'].mean():.3f} MAE平均={res['mae_atr'].mean():.3f} ATR)"
            if res.height > 0
            else f"{y}-{m:02d}: 0 bars"
        )

    if not all_parts:
        logging.warning("処理結果が空です。レポートは生成しません。")
        return

    full = pl.concat(all_parts, how="diagonal")
    _write_report(full)
    logging.info(f"### 完了。レポート: {REPORT_PATH} ###")


_OUT_COLS = [
    "timestamp",
    "entry_price",
    "mfe_usd",
    "mae_usd",
    "mfe_atr",
    "mae_atr",
    "mfe_dominance_atr",
    "mfe_time_min",
    "mae_time_min",
    "atr_value",
    "atr_ratio",
    "session_atr_ratio",
    "session",
    "hour_jst",
    "weekday",
    "n_ticks_window",
]


def _finalize_month(res: pl.DataFrame) -> pl.DataFrame:
    """生の max/min/到達時刻 から mfe/mae(usd,atr)・到達分・セッション等を算出。"""
    res = res.with_columns(
        (pl.col("_mfe_max_mid") - pl.col("close"))
        .cast(pl.Float64)
        .fill_nan(None)
        .alias("mfe_usd"),
        (pl.col("close") - pl.col("_mae_min_mid"))
        .cast(pl.Float64)
        .fill_nan(None)
        .alias("mae_usd"),
        pl.col("close").alias("entry_price"),
    )
    res = res.with_columns(
        pl.when(pl.col("atr_value") > 0)
        .then(pl.col("mfe_usd") / pl.col("atr_value"))
        .otherwise(None)
        .alias("mfe_atr"),
        pl.when(pl.col("atr_value") > 0)
        .then(pl.col("mae_usd") / pl.col("atr_value"))
        .otherwise(None)
        .alias("mae_atr"),
    )
    res = res.with_columns(
        (pl.col("mfe_atr") - pl.col("mae_atr")).alias("mfe_dominance_atr"),
        # 到達分 = (到達時刻 - エントリー時刻)/60e6。到達時刻0(tick無し)は null。
        pl.when(pl.col("_mfe_time_us") > 0)
        .then((pl.col("_mfe_time_us") - pl.col("_entry_ts")) / 60_000_000.0)
        .otherwise(None)
        .alias("mfe_time_min"),
        pl.when(pl.col("_mae_time_us") > 0)
        .then((pl.col("_mae_time_us") - pl.col("_entry_ts")) / 60_000_000.0)
        .otherwise(None)
        .alias("mae_time_min"),
    )
    # セッション/時間/曜日 (JST)
    res = res.with_columns(
        ((pl.col("timestamp").dt.hour() + 9) % 24).alias("hour_jst"),
        pl.col("timestamp").dt.weekday().alias("weekday"),
    )
    res = res.with_columns(
        pl.col("hour_jst")
        .map_elements(_hour_to_session_jst, return_dtype=pl.Utf8)
        .alias("session")
    )
    if "session_atr_ratio" not in res.columns:
        res = res.with_columns(
            pl.lit(None, dtype=pl.Float64).alias("session_atr_ratio")
        )
    res = res.with_columns(pl.lit(None, dtype=pl.Int64).alias("n_ticks_window"))
    return res.select([c for c in _OUT_COLS if c in res.columns])


def _persist_month(res: pl.DataFrame, y, m):
    pdir = PARQUET_DIR / f"year={y}/month={m}"
    pdir.mkdir(parents=True, exist_ok=True)
    res.write_parquet(pdir / "data.parquet", compression="zstd")


# ============================================================================
# テキストレポート (分布)
# ============================================================================
def _bin_counts(values, edges, last_is_ge=True):
    """edges = [e0, e1, ...] で [e0,e1),[e1,e2),... を数える。last_is_ge なら最後は >= 最終edge。"""
    vals = [v for v in values if v is not None and np.isfinite(v)]
    n = len(vals)
    rows = []
    for i in range(len(edges)):
        lo = edges[i]
        if i < len(edges) - 1:
            hi = edges[i + 1]
            c = sum(1 for v in vals if lo <= v < hi)
            label = f"{lo:g}-{hi:g}"
        else:
            if last_is_ge:
                c = sum(1 for v in vals if v >= lo)
                label = f">= {lo:g}"
            else:
                break
        rows.append((label, c, (c / n * 100 if n else 0.0)))
    return rows, n


def _fmt_dist(title, values, edges, unit=""):
    rows, n = _bin_counts(values, edges)
    out = [f"----- {title} (n={n}) -----"]
    for label, c, pct in rows:
        out.append(f"  {label:<12}{unit}: {str(c).rjust(8)} ({pct:5.1f} %)")
    return "\n".join(out) + "\n"


def _stats_line(name, values):
    vals = np.array(
        [v for v in values if v is not None and np.isfinite(v)], dtype=float
    )
    if vals.size == 0:
        return f"  {name:<16}: (データなし)"
    return (
        f"  {name:<16}: mean={vals.mean():.3f}  median={np.median(vals):.3f}  "
        f"p25={np.percentile(vals, 25):.3f}  p75={np.percentile(vals, 75):.3f}  "
        f"p90={np.percentile(vals, 90):.3f}  max={vals.max():.3f}"
    )


def _write_report(full: pl.DataFrame):
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    n = full.height
    ts_min = full["timestamp"].min()
    ts_max = full["timestamp"].max()

    lines = []
    lines.append("=" * 64)
    lines.append("    真の MFE/MAE 分布レポート (観測専用ラベリング)")
    lines.append("=" * 64)
    lines.append(f"観測窓 TO_MINUTES : {TO_MINUTES} 分")
    lines.append(f"エントリーオフセット: {ACTION_HORIZON_SEC} 秒")
    lines.append(f"対象バー数         : {n}")
    lines.append(f"期間               : {ts_min} - {ts_max}")
    lines.append("")

    # ── 全体サマリー ──
    lines.append("-" * 26 + " 全体サマリー " + "-" * 24)
    lines.append(_stats_line("MFE (ATR)", full["mfe_atr"].to_list()))
    lines.append(_stats_line("MAE (ATR)", full["mae_atr"].to_list()))
    lines.append(_stats_line("MFE到達(分)", full["mfe_time_min"].to_list()))
    lines.append(_stats_line("MAE到達(分)", full["mae_time_min"].to_list()))
    lines.append(_stats_line("MFE (USD)", full["mfe_usd"].to_list()))
    lines.append(_stats_line("MAE (USD)", full["mae_usd"].to_list()))
    lines.append("")

    # ── MFE/MAE 値の分布 ──
    lines.append(
        _fmt_dist("MFE (ATR) 値の分布", full["mfe_atr"].to_list(), MFE_MAE_ATR_BINS)
    )
    lines.append(
        _fmt_dist("MAE (ATR) 値の分布", full["mae_atr"].to_list(), MFE_MAE_ATR_BINS)
    )

    # ── 到達時刻の分布 (TO窓決定の核心) ──
    lines.append(
        _fmt_dist(
            "MFE 到達時刻の分布",
            full["mfe_time_min"].to_list(),
            REACH_MIN_BINS,
            unit="分",
        )
    )
    lines.append(
        _fmt_dist(
            "MAE 到達時刻の分布",
            full["mae_time_min"].to_list(),
            REACH_MIN_BINS,
            unit="分",
        )
    )

    # ── セッション別 (MFE/MAE平均・到達平均) ──
    lines.append("-" * 24 + " セッション別 平均 " + "-" * 22)
    lines.append(
        f"  {'Session':<10}{'件数':>8}{'MFE_atr':>10}{'MAE_atr':>10}"
        f"{'MFE到達分':>12}{'MAE到達分':>12}"
    )
    lines.append("  " + "-" * 60)
    for sess in ["Tokyo", "London", "Overlap", "NY", "Oceania"]:
        sub = full.filter(pl.col("session") == sess)
        if sub.is_empty():
            continue
        lines.append(
            f"  {sess:<10}{sub.height:>8}"
            f"{_safe_mean(sub['mfe_atr']):>10.3f}{_safe_mean(sub['mae_atr']):>10.3f}"
            f"{_safe_mean(sub['mfe_time_min']):>12.1f}{_safe_mean(sub['mae_time_min']):>12.1f}"
        )
    lines.append("")

    # ── ATR ratio 帯別 ──
    lines.append("-" * 24 + " ATR ratio 帯別 平均 " + "-" * 20)
    lines.append(
        f"  {'Band':<10}{'件数':>8}{'MFE_atr':>10}{'MAE_atr':>10}"
        f"{'MFE到達分':>12}{'MAE到達分':>12}"
    )
    lines.append("  " + "-" * 60)
    ar = full["atr_ratio"].to_list()
    for i in range(len(ATR_RATIO_BINS)):
        lo = ATR_RATIO_BINS[i]
        if i < len(ATR_RATIO_BINS) - 1:
            hi = ATR_RATIO_BINS[i + 1]
            label = f"{lo:g}-{hi:g}"
            mask = [(x is not None and np.isfinite(x) and lo <= x < hi) for x in ar]
        else:
            label = f">= {lo:g}"
            mask = [(x is not None and np.isfinite(x) and x >= lo) for x in ar]
        sub = full.filter(pl.Series(mask))
        if sub.is_empty():
            lines.append(f"  {label:<10}{0:>8}{'N/A':>10}")
            continue
        lines.append(
            f"  {label:<10}{sub.height:>8}"
            f"{_safe_mean(sub['mfe_atr']):>10.3f}{_safe_mean(sub['mae_atr']):>10.3f}"
            f"{_safe_mean(sub['mfe_time_min']):>12.1f}{_safe_mean(sub['mae_time_min']):>12.1f}"
        )
    lines.append("")

    # ── 曜日別 ──
    lines.append("-" * 26 + " 曜日別 平均 " + "-" * 25)
    wd_names = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    lines.append(f"  {'Weekday':<10}{'件数':>8}{'MFE_atr':>10}{'MAE_atr':>10}")
    lines.append("  " + "-" * 36)
    for wd in range(1, 8):
        sub = full.filter(pl.col("weekday") == wd)
        if sub.is_empty():
            continue
        lines.append(
            f"  {wd_names[wd - 1]:<10}{sub.height:>8}"
            f"{_safe_mean(sub['mfe_atr']):>10.3f}{_safe_mean(sub['mae_atr']):>10.3f}"
        )
    lines.append("")

    # ── 時間帯別 (JST) ──
    lines.append("-" * 25 + " 時間帯別(JST) 平均 " + "-" * 21)
    lines.append(f"  {'JST時':<10}{'件数':>8}{'MFE_atr':>10}{'MAE_atr':>10}")
    lines.append("  " + "-" * 36)
    for h in range(24):
        sub = full.filter(pl.col("hour_jst") == h)
        if sub.is_empty():
            continue
        lines.append(
            f"  {h:02d}:00     {sub.height:>8}"
            f"{_safe_mean(sub['mfe_atr']):>10.3f}{_safe_mean(sub['mae_atr']):>10.3f}"
        )
    lines.append("")
    lines.append("=" * 64)

    REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")


def _safe_mean(s: pl.Series) -> float:
    v = s.drop_nulls().drop_nans()
    return float(v.mean()) if v.len() > 0 else float("nan")


if __name__ == "__main__":
    import argparse
    import re as _re

    parser = argparse.ArgumentParser(
        description="真の MFE/MAE 分布 観測ツール (専用ラベリング)。"
    )
    parser.add_argument(
        "--year-month", type=str, default=None, help="YYYY/M (例 2021/7) その月だけ"
    )
    parser.add_argument("--year", type=int, default=None, help="その年だけ")
    args = parser.parse_args()

    fy = fm = None
    if args.year_month:
        mo = _re.match(r"^(\d{4})/(\d{1,2})$", args.year_month.strip())
        if not mo:
            raise SystemExit("ERROR: --year-month は YYYY/M 形式 (例 2021/7)")
        fy, fm = int(mo.group(1)), int(mo.group(2))
    elif args.year is not None:
        fy = args.year

    run(filter_year=fy, filter_month=fm)
