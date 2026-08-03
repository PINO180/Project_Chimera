# /workspace/models/analyze_breakeven_mfe_mae_distribution.py
# ============================================================================
# 【建値クロス版 真の MFE/MAE 分布 観測ツール (B2 観測)】
# ----------------------------------------------------------------------------
# 目的:
#   各 M3 バーのエントリー (L+180) から、"建値クロスで区切られた順行" を測る。
#     上方向: 価格が 建値−α を初めて割るまで の最大上振れ = MFE_be
#     下方向: 価格が 建値+α を初めて超えるまで の最大下振れ = MAE_be
#   (上下は独立に計測。α=建値固定の不感帯で、tick チカチカの偽終了を防ぐ)
#   窓内 max/min と違い、これは「順行」なので窓長に対して発散せず収束する量。
#
#   α は複数値を 1 回の tick 走査で同時計測。各 α について:
#     - MFE_be / MAE_be の分布 (ATR建て)
#     - 順行終了時刻 (エントリーから何分で建値±αに戻ったか) の分布
#     - 即時終了率 (60秒以内終了 = α が小さすぎてノイズを拾っている指標)
#     - TO打ち切り率 (TO まで一度も戻らず右打ち切り = α が大きすぎる指標)
#   → この 2 つの診断率が両方低い帯域が「妥当な α」。α 選定の実証的基準。
#
# ----------------------------------------------------------------------------
# 【実行コマンド例】 (/workspace から)
#   1ヶ月:   python models/analyze_breakeven_mfe_mae_distribution.py --year-month 2021/7
#   1年:     python models/analyze_breakeven_mfe_mae_distribution.py --year 2021
#   全期間:  python models/analyze_breakeven_mfe_mae_distribution.py
# ============================================================================
# 【調整パラメータ (ここを直接編集)】
#   TO_MINUTES : 観測の最大窓(分)。建値クロスしなければここで右打ち切り。
#   ALPHAS_ATR : 建値不感帯 α (ATR建て) のリスト。1回の走査で全て同時計測。
#       錨: ノイズ床(下限) / スプレッド≈0.75 ATR@2021 (経済錨) / TOスケール比(上限)
# ============================================================================
TO_MINUTES = 15
ALPHAS_ATR = [0.05, 0.10, 0.25, 0.50, 0.75]
ACTION_HORIZON_SEC = 180

# 分布の帯 (編集可)
MFE_MAE_ATR_BINS = [0.0, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0]  # 最後は >= 3.0
END_MIN_BINS = [0, 1, 2, 3, 5, 7, 10, 15]  # 順行終了時刻(分)。最後は >= 15
IMMEDIATE_SEC = 60  # 即時終了とみなす秒数

import sys
import logging
import datetime as dt
from pathlib import Path

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


_project_root = Path(__file__).resolve().parents[1]
if str(_project_root) not in sys.path:
    sys.path.append(str(_project_root))

from blueprint import (
    S6_LABELED_DATASET,
    S1_RAW_TICK_PARTITIONED,
)

OUT_DIR = S6_LABELED_DATASET.parent / "true_mfe_mae_analysis"
PARQUET_DIR = OUT_DIR / "per_bar_breakeven"
REPORT_PATH = OUT_DIR / "breakeven_mfe_mae_report.txt"

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


def _njit_if_available(func):
    if _NUMBA:
        return njit(func, parallel=True, fastmath=True, cache=True)
    return func


@_njit_if_available
def _numba_breakeven_runs(
    bets_t0: np.ndarray,        # int64 us (バー開始 L)
    bets_w_end: np.ndarray,     # int64 us (観測終端 = entry + TO)
    entry_prices: np.ndarray,   # float64 (エントリー価格 = close)
    alphas_price: np.ndarray,   # float64 [n_bets, n_alphas] (α を価格に換算済み)
    ticks_ts: np.ndarray,
    ticks_price: np.ndarray,
    entry_offset_us: np.int64,
):
    """建値クロス順行の同時計測 (複数α)。
    上方向: mid <= entry−α で終了。下方向: mid >= entry+α で終了。上下独立。
    各tickで running max/min を更新してから終了判定 (終了tickも順行値に含む)。
    返り値 [n_bets, n_alphas]:
      up_max   … 終了(または右打ち切り)までの最高 mid。tick無しは NaN。
      dn_min   … 同、最安 mid。
      up_end   … 上方向 順行終了時刻 us。0 = TOまで未クロス(右打ち切り)。
      dn_end   … 同、下方向。
    """
    n_bets = len(bets_t0)
    n_a = alphas_price.shape[1]
    up_max = np.full((n_bets, n_a), np.nan)
    dn_min = np.full((n_bets, n_a), np.nan)
    up_end = np.zeros((n_bets, n_a), dtype=np.int64)
    dn_end = np.zeros((n_bets, n_a), dtype=np.int64)
    n_ticks = len(ticks_ts)
    if n_ticks == 0:
        return up_max, dn_min, up_end, dn_end

    for i in prange(n_bets):
        entry_ts = bets_t0[i] + entry_offset_us
        w_end = bets_w_end[i]
        ep = entry_prices[i]
        start_idx = np.searchsorted(ticks_ts, entry_ts, side="right")

        run_max = -np.inf
        run_min = np.inf
        found = False
        # α ごとの生存フラグと凍結値
        for a in range(n_a):
            up_max[i, a] = np.nan  # 初期化 (prange 安全)
            dn_min[i, a] = np.nan
        up_alive = np.ones(n_a, dtype=np.bool_)
        dn_alive = np.ones(n_a, dtype=np.bool_)

        for j in range(start_idx, n_ticks):
            ts = ticks_ts[j]
            if ts > w_end:
                break
            p = ticks_price[j]
            if p > run_max:
                run_max = p
            if p < run_min:
                run_min = p
            found = True
            all_dead = True
            for a in range(n_a):
                al = alphas_price[i, a]
                if up_alive[a]:
                    if p <= ep - al:
                        up_alive[a] = False
                        up_end[i, a] = ts
                        up_max[i, a] = run_max
                    else:
                        all_dead = False
                if dn_alive[a]:
                    if p >= ep + al:
                        dn_alive[a] = False
                        dn_end[i, a] = ts
                        dn_min[i, a] = run_min
                    else:
                        all_dead = False
            if all_dead:
                break

        if found:
            for a in range(n_a):
                if up_alive[a]:  # 右打ち切り: 値は入れる、end=0
                    up_max[i, a] = run_max
                if dn_alive[a]:
                    dn_min[i, a] = run_min
    return up_max, dn_min, up_end, dn_end


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


def _a_tag(a: float) -> str:
    return f"a{int(round(a * 100)):03d}"  # 0.25 -> a025


def _load_bars() -> pl.DataFrame:
    logging.info("Loading M3 bars from S6 (timestamp/close/atr_value/atr_ratio)...")
    lf = pl.scan_parquet(str(S6_LABELED_DATASET / "**/*.parquet"))
    have = lf.collect_schema().names()
    cols = ["timestamp", "close", "atr_value"]
    if "atr_ratio" in have:
        cols.append("atr_ratio")
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


def run(filter_year=None, filter_month=None) -> None:
    logging.info("### 建値クロス版 MFE/MAE 分布 観測 開始 ###")
    logging.info(
        f"params: TO_MINUTES={TO_MINUTES}, ALPHAS_ATR={ALPHAS_ATR}, "
        f"offset={ACTION_HORIZON_SEC}s"
    )
    bars = _load_bars().with_columns(
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
    alphas = np.array(ALPHAS_ATR, dtype=np.float64)

    all_parts = []
    for (y, m) in ym_pairs:
        month_df = bars.filter((pl.col("_y") == y) & (pl.col("_m") == m))
        if month_df.is_empty():
            continue
        _nm = m + 1 if m < 12 else 1
        _ny = y if m < 12 else y + 1
        month_start = dt.datetime(y, m, 1, tzinfo=dt.timezone.utc)
        last_bar = month_df["timestamp"].max()
        month_end = last_bar + dt.timedelta(
            minutes=TO_MINUTES + ACTION_HORIZON_SEC / 60.0 + 5
        )
        try:
            ticks = (
                pl.scan_parquet(
                    str(S1_RAW_TICK_PARTITIONED / "**/*.parquet"),
                    hive_partitioning=True,
                )
                .filter(
                    pl.struct(["year", "month"]).is_in(
                        [{"year": y, "month": m}, {"year": _ny, "month": _nm}]
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
        entry_p = month_df["close"].cast(pl.Float64).to_numpy()
        atr_v = month_df["atr_value"].cast(pl.Float64).to_numpy()
        # α を価格に換算 (バーごとの ATR × α_atr)
        alphas_price = np.outer(atr_v, alphas)

        um, dm, ue, de = _numba_breakeven_runs(
            bets_t0, w_end, entry_p, alphas_price, ticks_ts, ticks_mid,
            entry_offset_us,
        )

        entry_ts_arr = bets_t0 + int(entry_offset_us)
        res = month_df
        for ai, a in enumerate(ALPHAS_ATR):
            tag = _a_tag(a)
            res = res.with_columns(
                pl.Series(f"_um_{tag}", um[:, ai]),
                pl.Series(f"_dm_{tag}", dm[:, ai]),
                pl.Series(f"_ue_{tag}", ue[:, ai]),
                pl.Series(f"_de_{tag}", de[:, ai]),
            )
        res = res.with_columns(pl.Series("_entry_ts", entry_ts_arr))
        res = _finalize_month(res)
        pdir = PARQUET_DIR / f"year={y}/month={m}"
        pdir.mkdir(parents=True, exist_ok=True)
        res.write_parquet(pdir / "data.parquet", compression="zstd")
        all_parts.append(res)
        logging.info(f"{y}-{m:02d}: bars={res.height} 建値クロス走査完了")

    if not all_parts:
        logging.warning("処理結果が空です。")
        return
    full = pl.concat(all_parts, how="diagonal")
    _write_report(full)
    logging.info(f"### 完了。レポート: {REPORT_PATH} ###")


def _finalize_month(res: pl.DataFrame) -> pl.DataFrame:
    exprs = []
    for a in ALPHAS_ATR:
        tag = _a_tag(a)
        exprs += [
            # MFE_be/MAE_be (ATR建て)。tick無し NaN → null。
            pl.when(pl.col("atr_value") > 0)
            .then((pl.col(f"_um_{tag}") - pl.col("close")) / pl.col("atr_value"))
            .otherwise(None)
            .fill_nan(None)
            .alias(f"mfe_be_atr_{tag}"),
            pl.when(pl.col("atr_value") > 0)
            .then((pl.col("close") - pl.col(f"_dm_{tag}")) / pl.col("atr_value"))
            .otherwise(None)
            .fill_nan(None)
            .alias(f"mae_be_atr_{tag}"),
            # 順行終了時刻(分)。end=0 は右打ち切り → null (censored 列で区別)。
            pl.when(pl.col(f"_ue_{tag}") > 0)
            .then((pl.col(f"_ue_{tag}") - pl.col("_entry_ts")) / 60_000_000.0)
            .otherwise(None)
            .alias(f"mfe_end_min_{tag}"),
            pl.when(pl.col(f"_de_{tag}") > 0)
            .then((pl.col(f"_de_{tag}") - pl.col("_entry_ts")) / 60_000_000.0)
            .otherwise(None)
            .alias(f"mae_end_min_{tag}"),
            (pl.col(f"_ue_{tag}") == 0).alias(f"mfe_censored_{tag}"),
            (pl.col(f"_de_{tag}") == 0).alias(f"mae_censored_{tag}"),
        ]
    res = res.with_columns(exprs)
    res = res.with_columns(
        ((pl.col("timestamp").dt.hour() + 9) % 24).alias("hour_jst"),
        pl.col("timestamp").dt.weekday().alias("weekday"),
    )
    res = res.with_columns(
        pl.col("hour_jst")
        .map_elements(_hour_to_session_jst, return_dtype=pl.Utf8)
        .alias("session")
    )
    keep = ["timestamp", "close", "atr_value", "atr_ratio", "session", "hour_jst",
            "weekday"]
    for a in ALPHAS_ATR:
        tag = _a_tag(a)
        keep += [
            f"mfe_be_atr_{tag}", f"mae_be_atr_{tag}",
            f"mfe_end_min_{tag}", f"mae_end_min_{tag}",
            f"mfe_censored_{tag}", f"mae_censored_{tag}",
        ]
    return res.select([c for c in keep if c in res.columns])


def _stats(vals):
    v = np.array([x for x in vals if x is not None and np.isfinite(x)], dtype=float)
    if v.size == 0:
        return None
    return dict(n=v.size, mean=v.mean(), med=float(np.median(v)),
                p25=float(np.percentile(v, 25)), p75=float(np.percentile(v, 75)),
                p90=float(np.percentile(v, 90)), mx=v.max())


def _fmt_dist(title, values, edges, unit=""):
    vals = [v for v in values if v is not None and np.isfinite(v)]
    n = len(vals)
    out = [f"----- {title} (n={n}) -----"]
    for i in range(len(edges)):
        lo = edges[i]
        if i < len(edges) - 1:
            hi = edges[i + 1]
            c = sum(1 for v in vals if lo <= v < hi)
            label = f"{lo:g}-{hi:g}"
        else:
            c = sum(1 for v in vals if v >= lo)
            label = f">= {lo:g}"
        pct = c / n * 100 if n else 0.0
        out.append(f"  {label:<10}{unit}: {str(c).rjust(8)} ({pct:5.1f} %)")
    return "\n".join(out) + "\n"


def _write_report(full: pl.DataFrame):
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    n = full.height
    lines = []
    lines.append("=" * 68)
    lines.append("    建値クロス版 MFE/MAE 分布レポート (B2 観測)")
    lines.append("=" * 68)
    lines.append(f"観測窓 TO_MINUTES : {TO_MINUTES} 分  /  α(ATR): {ALPHAS_ATR}")
    lines.append(f"対象バー数         : {n}")
    lines.append(f"期間               : {full['timestamp'].min()} - {full['timestamp'].max()}")
    lines.append("")

    # ── α 比較サマリー (α選定の実証基準) ──
    lines.append("-" * 20 + " α 比較サマリー (α選定用) " + "-" * 20)
    lines.append(
        f"  {'α':>5} | {'MFE_be中央':>9} {'MAE_be中央':>9} | "
        f"{'終了中央(分)':>10} | {'即時終了%':>8} | {'TO打切%':>7}"
    )
    lines.append("  " + "-" * 62)
    for a in ALPHAS_ATR:
        tag = _a_tag(a)
        s_mfe = _stats(full[f"mfe_be_atr_{tag}"].to_list())
        e_up = [x for x in full[f"mfe_end_min_{tag}"].to_list()
                if x is not None and np.isfinite(x)]
        s_mae = _stats(full[f"mae_be_atr_{tag}"].to_list())
        e_dn = [x for x in full[f"mae_end_min_{tag}"].to_list()
                if x is not None and np.isfinite(x)]
        ends = e_up + e_dn
        end_med = float(np.median(ends)) if ends else float("nan")
        imm = (
            sum(1 for x in ends if x <= IMMEDIATE_SEC / 60.0) / len(ends) * 100
            if ends else float("nan")
        )
        cen_up = full[f"mfe_censored_{tag}"].sum()
        cen_dn = full[f"mae_censored_{tag}"].sum()
        cen_pct = (cen_up + cen_dn) / (2 * n) * 100 if n else float("nan")
        lines.append(
            f"  {a:>5.2f} | {s_mfe['med'] if s_mfe else float('nan'):>9.3f} "
            f"{s_mae['med'] if s_mae else float('nan'):>9.3f} | "
            f"{end_med:>10.2f} | {imm:>7.1f}% | {cen_pct:>6.1f}%"
        )
    lines.append("")
    lines.append(f"  ※ 即時終了% = 順行が {IMMEDIATE_SEC}秒以内に建値±αへ戻った率 (高=αが小さすぎ)")
    lines.append("  ※ TO打切%  = TOまで一度も建値±αに戻らなかった率 (高=αが大きすぎ/窓が短い)")
    lines.append("")

    # ── α ごとの詳細分布 ──
    for a in ALPHAS_ATR:
        tag = _a_tag(a)
        lines.append("=" * 68)
        lines.append(f"◆ α = {a} ATR")
        lines.append("=" * 68)
        for name, col, unit, edges in [
            (f"MFE_be (上順行, ATR)", f"mfe_be_atr_{tag}", "", MFE_MAE_ATR_BINS),
            (f"MAE_be (下順行, ATR)", f"mae_be_atr_{tag}", "", MFE_MAE_ATR_BINS),
            (f"上順行 終了時刻", f"mfe_end_min_{tag}", "分", END_MIN_BINS),
            (f"下順行 終了時刻", f"mae_end_min_{tag}", "分", END_MIN_BINS),
        ]:
            st = _stats(full[col].to_list())
            if st:
                lines.append(
                    f"  [{name}] mean={st['mean']:.3f} med={st['med']:.3f} "
                    f"p90={st['p90']:.3f} max={st['mx']:.3f}"
                )
            lines.append(_fmt_dist(name + " 分布", full[col].to_list(), edges, unit))
        # セッション別 (中央値, 参考)
        lines.append(f"  --- セッション別 中央値 (α={a}) ---")
        lines.append(f"  {'Session':<10}{'MFE_be':>9}{'MAE_be':>9}{'終了分(上)':>10}")
        for sess in ["Tokyo", "London", "Overlap", "NY", "Oceania"]:
            sub = full.filter(pl.col("session") == sess)
            if sub.is_empty():
                continue
            sm = _stats(sub[f"mfe_be_atr_{tag}"].to_list())
            sa = _stats(sub[f"mae_be_atr_{tag}"].to_list())
            se = _stats(sub[f"mfe_end_min_{tag}"].to_list())
            lines.append(
                f"  {sess:<10}"
                f"{sm['med'] if sm else float('nan'):>9.3f}"
                f"{sa['med'] if sa else float('nan'):>9.3f}"
                f"{se['med'] if se else float('nan'):>10.2f}"
            )
        lines.append("")

    lines.append("=" * 68)
    REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    import argparse
    import re as _re

    parser = argparse.ArgumentParser(description="建値クロス版 MFE/MAE 観測 (B2)")
    parser.add_argument("--year-month", type=str, default=None, help="YYYY/M")
    parser.add_argument("--year", type=int, default=None)
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
