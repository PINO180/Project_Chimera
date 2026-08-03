# /workspace/models/compute_dynamic_exits_regression.py
# ============================================================================
# 【動的TP/SL 決済 前処理 (回帰版BTの心臓)】
# ----------------------------------------------------------------------------
# 目的:
#   M1 回帰予測 (mfe_pred / mae_pred, ATR建て) から取引ごとに動的な TP/SL 水準を作り、
#   実 tick を走査して「TP と SL のどちらに先に届いたか (先着)」を厳密に判定し、
#   決済結果 (勝/負/TO)・決済時刻・決済価格を各バーぶん出力する。
#
#   現BT (backtest_simulator_cimera.py) は決済を「ラベル (label_long) の再生」で
#   行っていたが、それは固定バリア (pt=1.0/sl=0.5) 専用。動的TP/SLでは事前ラベルが
#   使えないため、本スクリプトが「動的水準での tick 走査結果」を事前計算し、BT はそれを
#   再生する。tick 走査カーネルはラベリングと同一 (_numba_find_hits_dual、検証済み)。
#
# 入力:
#   - M1 OOF 予測: S7_M1_OOF_PREDICTIONS_LONG (mfe_pred), _SHORT (mae_pred)
#     ※本パイプラインでは long枠=mfe / short枠=mae。true_label は実測 mfe_atr/mae_atr。
#   - S6_LABELED: atr_value / close / atr_ratio / session / timestamp / timeframe
#   - 実 tick: S1_RAW_TICK_PARTITIONED (mid_price)
#
# 出力:
#   - 各パーティション (year=/month=/day=) に data.parquet:
#       timestamp, timeframe, entry_price, direction (1=Long/-1=Short/0=NoTrade),
#       tp_price, sl_price, outcome (1=TP先着/0=SL先着/-1=TO), exit_time (us), exit_price,
#       atr_value, atr_ratio, session, mfe_pred, mae_pred
#   → BT はこれを読み、valid_label→outcome / duration→(exit_time-entry) に差し替えて再生。
#
# コマンド１カ月だけ　例：python /workspace/backtest_simulator/compute_dynamic_exits_regression.py --year-month 2021/7
# ============================================================================
# 【調整パラメータ】※[道B] 確信度ゲート・方向判定は BT 側に移動。ここは tick 走査に
#   効く水準パラメータ (K_TP/K_SL) のみ。これらを変えたら前処理を回し直す必要がある。
# ============================================================================
#   K_TP / K_SL : TP/SL を予測値の何倍に置くか。1.0 = 予測そのまま。
#                 例) K_TP=0.8 = 予測の8割手前で利確、K_SL=1.2 = 予測より遠くに損切り。
#                 → TP/SL 価格 = tick 走査対象なので前処理側パラメータ。
# ---------------------------------------------------------------------------
#   ※ 確信度ゲート(ENTRY_RATIO_THRESHOLD)・atr_ratio ゲート・最小予測ゲートは
#     BT(backtest_simulator_cimera_stage*.py)の config 上部で調整する。
#     前処理は全バーの両方向決済結果を出すので、それらは前処理を回し直さず振れる。
# ============================================================================
K_TP = 1.0
K_SL = 1.0

# ─── TO 窓 (エントリーから何分先まで監視するか)。ラベリングの TD と揃える ───
TO_MINUTES = 240
# ─── エントリーオフセット (秒)。ラベリングの ACTION_HORIZON_SEC と同一 ───
ACTION_HORIZON_SEC = 180

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

        return deco if (a and callable(a[0])) is False else a[0]

    def prange(*a):
        return range(*a)


# blueprint (= /workspace/blueprint.py) を import 可能にする。
# 本スクリプトが /workspace/backtest_simulator/ にあっても親(/workspace)を解決する。
_project_root = Path(__file__).resolve().parents[1]
if str(_project_root) not in sys.path:
    sys.path.append(str(_project_root))

from blueprint import (
    S6_LABELED_DATASET,
    S1_RAW_TICK_PARTITIONED,
    S7_MODELS,
    S7_M1_OOF_PREDICTIONS_LONG,
    S7_M1_OOF_PREDICTIONS_SHORT,
)

# 出力先: 今回専用フォルダ stratum_7_models/backtest_simulator_regression/dynamic_exits
S7_DYNAMIC_EXITS = S7_MODELS / "backtest_simulator_regression" / "dynamic_exits"

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


def _njit_if_available(func):
    if _NUMBA:
        return njit(func, parallel=True, fastmath=True, cache=True)
    return func


@_njit_if_available
def _numba_find_hits_dual(
    bets_t0: np.ndarray,
    bets_t1_long: np.ndarray,
    bets_t1_short: np.ndarray,
    bets_pt_long: np.ndarray,
    bets_sl_long: np.ndarray,
    bets_pt_short: np.ndarray,
    bets_sl_short: np.ndarray,
    ticks_ts: np.ndarray,
    ticks_high: np.ndarray,
    ticks_low: np.ndarray,
    entry_offset_us: np.int64,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    ラベリングと同一の双方向 tick 走査カーネル (検証済み)。
    各 bet について (t0+offset, t1] を走査し、PT/SL の最初の到達時刻(us)を返す。0=未到達。
    LONG: high>=pt_long で PT, low<=sl_long で SL。SHORT: low<=pt_short で PT, high>=sl_short で SL。
    ※本前処理では発注方向のみ有効な水準を渡し、非発注方向は到達不能値を渡す。
    """
    n_bets = len(bets_t0)
    n_ticks = len(ticks_ts)
    out_pt_long = np.zeros(n_bets, dtype=np.int64)
    out_sl_long = np.zeros(n_bets, dtype=np.int64)
    out_pt_short = np.zeros(n_bets, dtype=np.int64)
    out_sl_short = np.zeros(n_bets, dtype=np.int64)
    if n_ticks == 0:
        return out_pt_long, out_sl_long, out_pt_short, out_sl_short

    for i in prange(n_bets):
        entry_ts = bets_t0[i] + entry_offset_us
        t1_l = bets_t1_long[i]
        t1_s = bets_t1_short[i]
        pt_l = bets_pt_long[i]
        sl_l = bets_sl_long[i]
        pt_s = bets_pt_short[i]
        sl_s = bets_sl_short[i]
        start_idx = np.searchsorted(ticks_ts, entry_ts, side="right")

        long_active = True
        short_active = True
        for j in range(start_idx, n_ticks):
            ts = ticks_ts[j]
            if ts > t1_l:
                long_active = False
            if ts > t1_s:
                short_active = False
            if not long_active and not short_active:
                break
            hi = ticks_high[j]
            lo = ticks_low[j]
            if long_active:
                if out_pt_long[i] == 0 and hi >= pt_l:
                    out_pt_long[i] = ts
                if out_sl_long[i] == 0 and lo <= sl_l:
                    out_sl_long[i] = ts
            if short_active:
                if out_pt_short[i] == 0 and lo <= pt_s:
                    out_pt_short[i] = ts
                if out_sl_short[i] == 0 and hi >= sl_s:
                    out_sl_short[i] = ts
    return out_pt_long, out_sl_long, out_pt_short, out_sl_short


def _load_predictions() -> pl.DataFrame:
    """M1 OOF (mfe/mae) を読み、(timestamp,timeframe) で結合して mfe_pred/mae_pred を得る。"""
    logging.info("Loading M1 OOF predictions (mfe=long枠 / mae=short枠)...")
    mfe = (
        pl.read_parquet(S7_M1_OOF_PREDICTIONS_LONG)
        .rename({"prediction": "mfe_pred"})
        .select(["timestamp", "timeframe", "mfe_pred"])
    )
    mae = (
        pl.read_parquet(S7_M1_OOF_PREDICTIONS_SHORT)
        .rename({"prediction": "mae_pred"})
        .select(["timestamp", "timeframe", "mae_pred"])
    )
    preds = mfe.join(mae, on=["timestamp", "timeframe"], how="inner").with_columns(
        pl.col("timestamp").cast(pl.Datetime("us", "UTC"))
    )
    logging.info(f"  -> joined predictions: {preds.height} rows")
    return preds


def _compute_dual_levels(df: pl.DataFrame) -> pl.DataFrame:
    """[道B] 全バーについて Long/Short 両方向の動的TP/SL価格を計算する(ゲート無し)。
    確信度ゲート・方向判定は BT 側で行うため、ここでは両方向の水準と決済結果を用意する。
    Long : pt = entry + mfe_pred*atr*K_TP,  sl = entry - mae_pred*atr*K_SL
    Short: pt = entry - mae_pred*atr*K_TP,  sl = entry + mfe_pred*atr*K_SL
    (K_TP/K_SL は TP/SL 価格 = tick 走査対象を変えるため前処理側パラメータ)
    """
    return df.with_columns(
        (pl.col("close") + pl.col("mfe_pred") * pl.col("atr_value") * K_TP).alias(
            "pt_long"
        ),
        (pl.col("close") - pl.col("mae_pred") * pl.col("atr_value") * K_SL).alias(
            "sl_long"
        ),
        (pl.col("close") - pl.col("mae_pred") * pl.col("atr_value") * K_TP).alias(
            "pt_short"
        ),
        (pl.col("close") + pl.col("mfe_pred") * pl.col("atr_value") * K_SL).alias(
            "sl_short"
        ),
    )


def _resolve_outcomes(df: pl.DataFrame, pt_l, sl_l, pt_s, sl_s) -> pl.DataFrame:
    """[道B] カーネルの到達時刻から、Long/Short 両方向の outcome/exit_time/exit_price を出す。
    outcome: 1=TP先着(勝) / 0=SL先着(負) / -1=TO。exit_time は先着時刻(us)、TO は t1。
    exit_price: TP/SL 先着時の動的水準。TO は null(BT 側 close_future に委ねる)。
    """
    df = df.with_columns(
        pl.Series("_pt_l", pt_l),
        pl.Series("_sl_l", sl_l),
        pl.Series("_pt_s", pt_s),
        pl.Series("_sl_s", sl_s),
    )
    df = df.with_columns(
        (
            pl.col("timestamp").cast(pl.Int64)
            + int(ACTION_HORIZON_SEC * 1_000_000)
            + int(TO_MINUTES * 60 * 1_000_000)
        ).alias("_t1_us")
    )

    def first_hit(pt_col, sl_col):
        # 1=TP先着, 0=SL先着, -1=TO
        return (
            pl.when(
                (pl.col(pt_col) > 0)
                & ((pl.col(sl_col) == 0) | (pl.col(pt_col) < pl.col(sl_col)))
            )
            .then(1)
            .when(pl.col(sl_col) > 0)
            .then(0)
            .otherwise(-1)
        )

    # 両方向の先着判定
    df = df.with_columns(
        first_hit("_pt_l", "_sl_l").cast(pl.Int8).alias("long_outcome"),
        first_hit("_pt_s", "_sl_s").cast(pl.Int8).alias("short_outcome"),
    )
    # 両方向の決済時刻(us)
    df = df.with_columns(
        pl.when(pl.col("long_outcome") == 1)
        .then(pl.col("_pt_l"))
        .when(pl.col("long_outcome") == 0)
        .then(pl.col("_sl_l"))
        .otherwise(pl.col("_t1_us"))
        .alias("long_exit_time"),
        pl.when(pl.col("short_outcome") == 1)
        .then(pl.col("_pt_s"))
        .when(pl.col("short_outcome") == 0)
        .then(pl.col("_sl_s"))
        .otherwise(pl.col("_t1_us"))
        .alias("short_exit_time"),
    )
    # 両方向の決済価格(TP/SL先着時のみ。TO は null → BT の close_future)
    df = df.with_columns(
        pl.when(pl.col("long_outcome") == 1)
        .then(pl.col("pt_long"))
        .when(pl.col("long_outcome") == 0)
        .then(pl.col("sl_long"))
        .otherwise(None)
        .alias("long_exit_price"),
        pl.when(pl.col("short_outcome") == 1)
        .then(pl.col("pt_short"))
        .when(pl.col("short_outcome") == 0)
        .then(pl.col("sl_short"))
        .otherwise(None)
        .alias("short_exit_price"),
    )

    return df.drop(["_pt_l", "_sl_l", "_pt_s", "_sl_s", "_t1_us"])


def run(filter_year: int = None, filter_month: int = None) -> None:
    logging.info("### 動的TP/SL 決済 前処理 (回帰版) 開始 ###")
    if filter_year is not None:
        logging.info(
            f"範囲フィルタ: year={filter_year}"
            + (f", month={filter_month}" if filter_month is not None else "")
        )
    logging.info(
        f"params: K_TP={K_TP}, K_SL={K_SL}, TO={TO_MINUTES}m "
        f"(確信度ゲート・方向判定は BT 側)"
    )

    preds = _load_predictions()

    # S6 から atr_value/close/atr_ratio/session を取得して予測と結合
    s6_cols = ["timestamp", "timeframe", "atr_value", "close", "atr_ratio"]
    logging.info("Scanning S6 for atr_value/close/atr_ratio (+session if present)...")
    s6_lf = pl.scan_parquet(str(S6_LABELED_DATASET / "**/*.parquet"))
    have = s6_lf.collect_schema().names()
    if "session" in have:
        s6_cols.append("session")
    s6 = (
        s6_lf.select([c for c in s6_cols if c in have])
        .collect()
        .with_columns(pl.col("timestamp").cast(pl.Datetime("us", "UTC")))
    )

    df = preds.join(s6, on=["timestamp", "timeframe"], how="inner")
    if "atr_ratio" not in df.columns:
        df = df.with_columns(pl.lit(1.0).alias("atr_ratio"))
    if "session" not in df.columns:
        df = df.with_columns(pl.lit("NA").alias("session"))

    df = _compute_dual_levels(df)
    logging.info(
        f"  -> computing dual-direction exits for ALL {df.height} bars "
        f"(確信度ゲート・方向判定は BT 側)"
    )

    # ─── 月次チャンクで tick を読み、全バーを両方向カーネル走査 ───
    S7_DYNAMIC_EXITS.mkdir(parents=True, exist_ok=True)
    df = df.with_columns(
        pl.col("timestamp").dt.year().alias("_y"),
        pl.col("timestamp").dt.month().alias("_m"),
    )
    ym_pairs = df.select(["_y", "_m"]).unique().sort(["_y", "_m"]).rows()
    if filter_year is not None:
        ym_pairs = [
            (y, m)
            for (y, m) in ym_pairs
            if y == filter_year and (filter_month is None or m == filter_month)
        ]
        if not ym_pairs:
            logging.warning("範囲フィルタに合致する月がありません。")
    entry_offset_us = np.int64(ACTION_HORIZON_SEC * 1_000_000)

    for y, m in ym_pairs:
        month_df = df.filter((pl.col("_y") == y) & (pl.col("_m") == m))
        if month_df.is_empty():
            continue

        # tick 読込 (hive パーティション枝刈り。当月+翌月)
        _nm = m + 1 if m < 12 else 1
        _ny = y if m < 12 else y + 1
        month_start = dt.datetime(y, m, 1, tzinfo=dt.timezone.utc)
        # 窓終端の余裕(翌月頭まで)
        last_day = (dt.datetime(_ny, _nm, 1, tzinfo=dt.timezone.utc)) - dt.timedelta(
            seconds=1
        )
        margin = dt.timedelta(minutes=TO_MINUTES + ACTION_HORIZON_SEC / 60.0 + 5)
        month_end = last_day + margin
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
            logging.warning(f"{y}-{m:02d}: no ticks, skipping month.")
            continue

        ticks_ts = ticks["timestamp"].cast(pl.Int64).to_numpy()
        ticks_mid = ticks["mid_price"].to_numpy()  # tick は単一点 → high=low=mid

        # [道B] 全バーを Long/Short 両方向で tick 走査 (ゲート無し)
        bets_t0 = month_df["timestamp"].cast(pl.Int64).to_numpy()
        t1 = bets_t0 + entry_offset_us + np.int64(TO_MINUTES * 60 * 1_000_000)
        pt_l = month_df["pt_long"].to_numpy()
        sl_l = month_df["sl_long"].to_numpy()
        pt_s = month_df["pt_short"].to_numpy()
        sl_s = month_df["sl_short"].to_numpy()

        opt_l, osl_l, opt_s, osl_s = _numba_find_hits_dual(
            bets_t0,
            t1,
            t1,
            pt_l,
            sl_l,
            pt_s,
            sl_s,
            ticks_ts,
            ticks_mid,
            ticks_mid,
            entry_offset_us,
        )

        resolved = _resolve_outcomes(month_df, opt_l, osl_l, opt_s, osl_s)
        _persist(_shape_out(resolved), y, m)
        logging.info(
            f"{y}-{m:02d}: bars={resolved.height} | "
            f"Long W/L/TO="
            f"{resolved.filter(pl.col('long_outcome') == 1).height}/"
            f"{resolved.filter(pl.col('long_outcome') == 0).height}/"
            f"{resolved.filter(pl.col('long_outcome') == -1).height} | "
            f"Short W/L/TO="
            f"{resolved.filter(pl.col('short_outcome') == 1).height}/"
            f"{resolved.filter(pl.col('short_outcome') == 0).height}/"
            f"{resolved.filter(pl.col('short_outcome') == -1).height}"
        )

    logging.info("### 前処理 完了 ###")


# [道B] 出力列: 方向を決めず、両方向の水準・決済結果を全バーぶん持つ。
#   BT 側で確信度ゲート → 方向決定 → 該当方向の {outcome, exit_time, exit_price} を採用。
_OUT_COLS = [
    "timestamp",
    "timeframe",
    "entry_price",
    "pt_long",
    "sl_long",
    "pt_short",
    "sl_short",
    "long_outcome",
    "long_exit_time",
    "long_exit_price",
    "short_outcome",
    "short_exit_time",
    "short_exit_price",
    "atr_value",
    "atr_ratio",
    "session",
    "mfe_pred",
    "mae_pred",
]


def _shape_out(df: pl.DataFrame) -> pl.DataFrame:
    df = df.with_columns(pl.col("close").alias("entry_price"))
    for c in _OUT_COLS:
        if c not in df.columns:
            df = df.with_columns(pl.lit(None).alias(c))
    return df.select(_OUT_COLS)


def _persist(out: pl.DataFrame, y, m):
    out = out.with_columns(
        pl.col("timestamp").dt.day().alias("_d"),
    )
    for dkey, day_df in out.partition_by("_d", as_dict=True, include_key=False).items():
        d = dkey[0] if isinstance(dkey, tuple) else dkey
        pdir = S7_DYNAMIC_EXITS / f"year={y}/month={m}/day={d}"
        pdir.mkdir(parents=True, exist_ok=True)
        day_df.write_parquet(pdir / "data.parquet", compression="zstd")


if __name__ == "__main__":
    import argparse
    import re as _re

    _parser = argparse.ArgumentParser(
        description="動的TP/SL 決済 前処理 (回帰版)。範囲指定で一部だけ処理可。"
    )
    _parser.add_argument(
        "--year-month", type=str, default=None, help="YYYY/M (例 2021/7) その月だけ処理"
    )
    _parser.add_argument("--year", type=int, default=None, help="その年だけ処理")
    _args = _parser.parse_args()

    _FILTER_Y = None
    _FILTER_M = None
    if _args.year_month:
        _mo = _re.match(r"^(\d{4})/(\d{1,2})$", _args.year_month.strip())
        if not _mo:
            raise SystemExit("ERROR: --year-month は YYYY/M 形式 (例 2021/7)")
        _FILTER_Y, _FILTER_M = int(_mo.group(1)), int(_mo.group(2))
    elif _args.year is not None:
        _FILTER_Y = _args.year

    run(filter_year=_FILTER_Y, filter_month=_FILTER_M)
