# /workspace/models/fast_grid_search_hilbert_atr.py

import sys
import logging
import gc
from pathlib import Path
import itertools
import time
from datetime import datetime, timedelta, timezone
import polars as pl
import numpy as np
import collections

try:
    from numba import njit, prange
except ImportError:
    print("Numba is required for this script. Please install it (pip install numba).")
    sys.exit(1)

# --- プロジェクトのパス設定 ---
project_root = Path(__file__).resolve().parents[1]
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

try:
    from blueprint import S2_FEATURES_FIXED, S5_NEUTRALIZED_ALPHA_SET
except ImportError:
    logging.warning("blueprint.py not found. Using fallback paths.")
    S2_FEATURES_FIXED = Path("/workspace/data/XAUUSD/stratum_2_features_fixed")
    S5_NEUTRALIZED_ALPHA_SET = Path(
        "/workspace/data/XAUUSD/stratum_5_alpha/1A_2B/neutralized_alpha_set_partitioned"
    )

# --- ロギングと定数設定 ---
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

RESULTS_CSV_PATH = Path(
    "/workspace/data/XAUUSD/stratum_7_models/1A_2B/fast_grid_search_hilbert_results.csv"
)
RESULTS_CSV_PATH.parent.mkdir(parents=True, exist_ok=True)

SPREAD_COST = 0.16
COOL_DOWN_MINUTES = 1
HILBERT_COL = "hilbert_amplitude_50"

# ★初動特化スキャルピング向けのパラメータ群
TIMEFRAMES = [
    # "M1",
    # "M3",
    "M5",
    # "M8",
    # "M15",
    # "M30"
]
TIMEFRAMES_WINDOW_MAP = {
    # "M1": "1m",
    # "M3": "3m",
    "M5": "5m",
    # "M8": "8m",
    # "M15": "15m",
    # "M30": "30m"
}

ATR_PERIODS = [
    13,
    # 21, 55
]

TRIGGER_RATIOS = [
    # 1.03,
    # 1.05,
    1.08,
    1.10,
    # 1.12,
    # 1.15,
    # 1.18,
]

ATR_FILTER_RATIOS = [
    # 0.1,
    # 0.3,
    # 0.5,
    0.8,
    # 1.0,
    # 1.5,
    # 2.0, 3.0, 5.0
]

PT_MULTS = [
    0.5,
    # 1.0,
    # 2.5,
    # 5.0
]

SL_MULTS = [
    # 0.5, 1.0, 2.5,
    5.0
]

TD_MINS = [
    # 3,
    5,
    # 10, 30, 60, 120,180,
    # 360,
    # 720,
    # 960,
    # 1080,
    # 1200,
]

MIN_TOTAL_BETS = 100
TEST_PERIOD = "all"  # "all" で全期間（月次ループ）実行、 または "2021/8" など指定可


# ====================================================================
# Numba JIT 高速トリプルバリア関数
# ====================================================================
@njit(parallel=True, fastmath=True, cache=True)
def _numba_find_hits_fast(
    bets_t0: np.ndarray,
    bets_t1_max: np.ndarray,
    bets_pt_barrier: np.ndarray,
    bets_sl_barrier: np.ndarray,
    ticks_ts: np.ndarray,
    ticks_high: np.ndarray,
    ticks_low: np.ndarray,
):
    n_bets = len(bets_t0)
    n_ticks = len(ticks_ts)
    out_first_pt_time = np.zeros(n_bets, dtype=np.int64)
    out_first_sl_time = np.zeros(n_bets, dtype=np.int64)

    for i in prange(n_bets):
        t0 = bets_t0[i]
        t1_max = bets_t1_max[i]
        pt = bets_pt_barrier[i]
        sl = bets_sl_barrier[i]

        start_idx = np.searchsorted(ticks_ts, t0, side="left")
        first_pt_found = np.int64(0)
        first_sl_found = np.int64(0)

        for j in range(start_idx, n_ticks):
            tick_time = ticks_ts[j]
            if tick_time > t1_max:
                break

            tick_high = ticks_high[j]
            tick_low = ticks_low[j]

            if first_pt_found == 0 and tick_high >= pt:
                first_pt_found = tick_time
            if first_sl_found == 0 and tick_low <= sl:
                first_sl_found = tick_time

            if first_pt_found != 0 and first_sl_found != 0:
                break

        out_first_pt_time[i] = first_pt_found
        out_first_sl_time[i] = first_sl_found

    return out_first_pt_time, out_first_sl_time


# ====================================================================
# 月次バッチ用の日付ヘルパー
# ====================================================================
def add_months(d: datetime, months: int) -> datetime:
    month = d.month - 1 + months
    year = d.year + month // 12
    month = month % 12 + 1
    return datetime(year, month, 1, tzinfo=timezone.utc)


def get_available_months(tick_dir: Path):
    if TEST_PERIOD != "all":
        y, m = map(int, TEST_PERIOD.split("/"))
        return [(y, m)]

    months = []
    for p in tick_dir.rglob("month=*"):
        try:
            y = int(p.parent.name.split("=")[1])
            m = int(p.name.split("=")[1])
            months.append((y, m))
        except Exception:
            pass
    return sorted(list(set(months)))


# ====================================================================
# コア評価ロジック
# ====================================================================
def evaluate_and_accumulate(
    tf_label: str,
    atr_p: int,
    tr: float,
    atr_f: float,
    raw_t0: np.ndarray,
    raw_close: np.ndarray,
    raw_atr: np.ndarray,
    ticks_ts: np.ndarray,
    ticks_high: np.ndarray,
    ticks_low: np.ndarray,
    results_dict: dict,
):
    """シグナル配列をトリプルバリアで評価し、集計用辞書に累積する"""
    cool_down_us = np.int64(COOL_DOWN_MINUTES * 60 * 1_000_000)
    valid_indices = []
    last_time = np.int64(-cool_down_us)

    for idx, t in enumerate(raw_t0):
        if t - last_time >= cool_down_us:
            valid_indices.append(idx)
            last_time = t

    if not valid_indices:
        return

    bets_t0 = np.ascontiguousarray(raw_t0[valid_indices])
    bets_close = np.ascontiguousarray(raw_close[valid_indices])
    bets_atr = np.ascontiguousarray(raw_atr[valid_indices])
    triggered_count = len(bets_t0)
    sum_atr = float(np.sum(bets_atr))

    for td in TD_MINS:
        td_us = np.int64(td * 60 * 1_000_000)
        bets_t1_max = bets_t0 + td_us

        for pt_mult, sl_mult in itertools.product(PT_MULTS, SL_MULTS):
            bets_pt = np.ascontiguousarray(bets_close + bets_atr * pt_mult)
            bets_sl = np.ascontiguousarray(bets_close - bets_atr * sl_mult)

            out_pt, out_sl = _numba_find_hits_fast(
                bets_t0, bets_t1_max, bets_pt, bets_sl, ticks_ts, ticks_high, ticks_low
            )

            is_win = (out_pt > 0) & ((out_sl == 0) | (out_pt < out_sl))
            is_loss = (out_sl > 0) & ((out_pt == 0) | (out_sl <= out_pt))
            is_timeout = ~(is_win | is_loss)

            wins = int(np.sum(is_win))
            losses = int(np.sum(is_loss))
            timeouts = int(np.sum(is_timeout))

            total_profit = float(np.sum(bets_atr[is_win].astype(np.float64)) * pt_mult)
            total_loss_base = float(
                np.sum(bets_atr[is_loss].astype(np.float64)) * sl_mult
            )
            total_loss_adj = total_loss_base + (timeouts * SPREAD_COST)

            agg_key = (tf_label, atr_p, tr, atr_f, pt_mult, sl_mult, td)
            if agg_key not in results_dict:
                results_dict[agg_key] = {
                    "Wins": 0,
                    "Losses": 0,
                    "Timeouts": 0,
                    "Total_Bets": 0,
                    "Total_Profit": 0.0,
                    "Total_Loss_Base": 0.0,
                    "Total_Loss_Adj": 0.0,
                    "Sum_ATR": 0.0,
                }

            agg = results_dict[agg_key]
            agg["Wins"] += wins
            agg["Losses"] += losses
            agg["Timeouts"] += timeouts
            agg["Total_Bets"] += triggered_count
            agg["Total_Profit"] += total_profit
            agg["Total_Loss_Base"] += total_loss_base
            agg["Total_Loss_Adj"] += total_loss_adj
            agg["Sum_ATR"] += sum_atr


# ====================================================================
# メインループ
# ====================================================================
def run_fast_grid_search():
    tick_dir = S2_FEATURES_FIXED / "feature_value_a_vast_universeC/features_e1c_tick"
    target_months = get_available_months(tick_dir)

    if not target_months:
        logging.error("No Tick data found. Check your directory structure.")
        return

    logging.info(
        f"Starting Monthly Batch Grid Search. Total months to process: {len(target_months)}"
    )

    results_agg = {}
    start_time = time.time()

    for year, month in target_months:
        logging.info(f"====== Processing Month: {year}/{month:02d} ======")
        gc.collect()

        start_date = datetime(year, month, 1, tzinfo=timezone.utc)
        end_date = add_months(start_date, 1)
        # タイムアウト評価用に、Tickデータのみ翌月頭から2日分余分に読み込む
        tick_end_date = end_date + timedelta(days=2)

        # --- 1. Tickデータのロード (当月分のみ) ---
        lf_tick = (
            pl.scan_parquet(str(tick_dir / "**/*.parquet"))
            .with_columns(pl.col("timestamp").cast(pl.Datetime("us", "UTC")))
            .filter(
                (pl.col("timestamp") >= start_date)
                & (pl.col("timestamp") < tick_end_date)
            )
            .sort("timestamp")
        )

        df_tick_month = lf_tick.select(["timestamp", "high", "low", "close"]).collect(
            engine="streaming"
        )
        if df_tick_month.is_empty():
            logging.warning(f"No Tick data for {year}/{month:02d}. Skipping.")
            continue

        ticks_ts = np.ascontiguousarray(
            df_tick_month.with_columns(pl.col("timestamp").dt.timestamp("us"))[
                "timestamp"
            ].to_numpy()
        )
        ticks_high = np.ascontiguousarray(df_tick_month["high"].to_numpy())
        ticks_low = np.ascontiguousarray(df_tick_month["low"].to_numpy())

        # 混合モード用: 全TFのシグナルを月ごとにプールする
        mixed_signals = collections.defaultdict(list)

        for tf in TIMEFRAMES:
            # --- 2. S2からHilbertのロード (S5の代替) ---
            lf_s5 = None
            for p in S2_FEATURES_FIXED.rglob(f"features_*_{tf}.parquet"):
                if "tick" in str(p):
                    continue
                try:
                    s2_cols = pl.scan_parquet(str(p)).collect_schema().names()
                    actual_hilbert = next(
                        (c for c in s2_cols if HILBERT_COL in c), None
                    )
                    if actual_hilbert:
                        lf_s5 = (
                            pl.scan_parquet(str(p))
                            .select(["timestamp", actual_hilbert])
                            .with_columns(
                                pl.col("timestamp").cast(pl.Datetime("us", "UTC"))
                            )
                            .sort("timestamp")
                            .with_columns(
                                pl.col(actual_hilbert)
                                .rolling_mean(window_size=50, min_samples=1)
                                .alias("hilbert_ma_50")
                            )
                            .with_columns(
                                (
                                    pl.col(actual_hilbert) / pl.col("hilbert_ma_50")
                                ).alias("trigger_ratio_val")
                            )
                            .filter(
                                (pl.col("timestamp") >= start_date)
                                & (pl.col("timestamp") < end_date)
                            )
                        )
                        break
                except Exception:
                    pass

            if lf_s5 is None:
                continue

            # ATRのロード
            atr_lf = None
            atr_cols = []
            for p in S2_FEATURES_FIXED.rglob(f"features_*_{tf}.parquet"):
                if "tick" in str(p):
                    continue
                try:
                    t_cols = pl.scan_parquet(str(p)).collect_schema().names()
                    m_cols = [
                        c for c in t_cols if any(f"atr_{a}" in c for a in ATR_PERIODS)
                    ]
                    if len(m_cols) >= len(ATR_PERIODS):
                        atr_lf = (
                            pl.scan_parquet(str(p))
                            .select(["timestamp"] + m_cols)
                            .with_columns(
                                pl.col("timestamp").cast(pl.Datetime("us", "UTC"))
                            )
                            .filter(
                                (pl.col("timestamp") >= start_date)
                                & (pl.col("timestamp") < end_date)
                            )
                            .sort("timestamp")
                        )
                        renames = {
                            c: f"atr_{a}"
                            for c in m_cols
                            for a in ATR_PERIODS
                            if f"atr_{a}" in c
                        }
                        atr_lf = atr_lf.rename(renames)
                        atr_cols = list(renames.values())
                        break
                except Exception:
                    pass

            if atr_lf is None:
                continue

            # スプレッド（ローソク足の実体幅）の計算
            spread_w = TIMEFRAMES_WINDOW_MAP[tf]
            lf_spread = (
                df_tick_month.lazy()
                .group_by_dynamic("timestamp", every=spread_w, closed="right")
                .agg(
                    [
                        pl.col("high").max().alias("bar_high"),
                        pl.col("low").min().alias("bar_low"),
                    ]
                )
                .with_columns((pl.col("bar_high") - pl.col("bar_low")).alias("spread"))
            )

            # --- 3. 結合 (S5をベースにして軽量化) ---
            lf_sig = lf_s5.join_asof(
                df_tick_month.lazy().select(["timestamp", "close"]),
                on="timestamp",
                strategy="backward",
            )
            lf_sig = lf_sig.join_asof(
                lf_spread.select(["timestamp", "spread"]),
                on="timestamp",
                strategy="backward",
            )
            lf_sig = lf_sig.join_asof(atr_lf, on="timestamp", strategy="backward")

            fill_exprs = [
                pl.col(c).fill_null(strategy="forward").fill_null(strategy="backward")
                for c in atr_cols
            ]
            lf_sig = (
                lf_sig.with_columns(fill_exprs)
                .drop_nulls(subset=["close", "trigger_ratio_val", "spread"] + atr_cols)
                .with_columns(pl.col("timestamp").dt.timestamp("us").alias("ts_int"))
            )

            df_sig = lf_sig.collect(engine="streaming")
            if df_sig.is_empty():
                continue

            sig_ts = np.ascontiguousarray(df_sig["ts_int"].to_numpy())
            sig_close = df_sig["close"].to_numpy()
            sig_trigger = df_sig["trigger_ratio_val"].to_numpy()
            sig_spread = df_sig["spread"].to_numpy()

            # --- 4. 評価 ---
            for atr_p in ATR_PERIODS:
                sig_atr = df_sig[f"atr_{atr_p}"].to_numpy()

                for tr, atr_f in itertools.product(TRIGGER_RATIOS, ATR_FILTER_RATIOS):
                    # ★AND条件: Hilbert急変化 ＆ スプレッドがATRのX倍以上
                    mask = (
                        (sig_trigger >= tr)
                        & (sig_spread > (sig_atr * atr_f))
                        & (sig_atr > 0)
                    )

                    if np.any(mask):
                        raw_t0 = sig_ts[mask]
                        raw_close = sig_close[mask]
                        raw_atr = sig_atr[mask]

                        # 個別時間足の評価
                        evaluate_and_accumulate(
                            tf,
                            atr_p,
                            tr,
                            atr_f,
                            raw_t0,
                            raw_close,
                            raw_atr,
                            ticks_ts,
                            ticks_high,
                            ticks_low,
                            results_agg,
                        )

                        # 混合モード用プールに格納
                        mixed_signals[(atr_p, tr, atr_f)].append(
                            (raw_t0, raw_close, raw_atr)
                        )

        # --- 月ごとの all_mixed (混合モード) の評価 ---
        for (atr_p, tr, atr_f), sig_list in mixed_signals.items():
            if not sig_list:
                continue
            all_t0 = np.concatenate([s[0] for s in sig_list])
            all_close = np.concatenate([s[1] for s in sig_list])
            all_atr = np.concatenate([s[2] for s in sig_list])

            sort_idx = np.argsort(all_t0)

            evaluate_and_accumulate(
                "all_mixed",
                atr_p,
                tr,
                atr_f,
                all_t0[sort_idx],
                all_close[sort_idx],
                all_atr[sort_idx],
                ticks_ts,
                ticks_high,
                ticks_low,
                results_agg,
            )

        del df_tick_month, ticks_ts, ticks_high, ticks_low
        gc.collect()

    # --- 5. 最終集計とCSV出力 ---
    logging.info("Generating Final Report...")
    final_rows = []

    for key, agg in results_agg.items():
        tf, atr_p, tr, atr_f, pt, sl, td = key
        wins, losses, timeouts, total_bets = (
            agg["Wins"],
            agg["Losses"],
            agg["Timeouts"],
            agg["Total_Bets"],
        )

        if total_bets == 0:
            continue

        win_rate = wins / total_bets
        total_profit = agg["Total_Profit"]
        total_loss_base = agg["Total_Loss_Base"]
        total_loss_adj = agg["Total_Loss_Adj"]

        pf = (
            total_profit / total_loss_base
            if total_loss_base > 0
            else (float("inf") if total_profit > 0 else 1.0)
        )
        adj_pf = (
            total_profit / total_loss_adj
            if total_loss_adj > 0
            else (float("inf") if total_profit > 0 else 1.0)
        )
        avg_payoff = pt / sl
        avg_atr = agg["Sum_ATR"] / total_bets

        final_rows.append(
            {
                "Timeframe": tf,
                "ATR_Period": atr_p,
                "Trigger_Ratio": tr,
                "ATR_Filter": atr_f,
                "Avg_ATR": round(avg_atr, 4),
                "PT": pt,
                "SL": sl,
                "TD_min": td,
                "Total_Bets": total_bets,
                "Wins": wins,
                "Losses": losses,
                "Timeouts": timeouts,
                "WinRate": round(win_rate, 4),
                "Avg_Payoff": round(avg_payoff, 4),
                "PF": round(pf, 4),
                "Adjusted_PF": round(adj_pf, 4),
            }
        )

    elapsed = time.time() - start_time
    logging.info(f"Grid Search completed in {elapsed:.2f} seconds.")

    if not final_rows:
        logging.warning("No results generated.")
        return

    res_df = pl.DataFrame(final_rows).sort("Adjusted_PF", descending=True)

    try:
        res_df.write_csv(RESULTS_CSV_PATH)
        logging.info(f"Results successfully saved to: {RESULTS_CSV_PATH}")
    except Exception as e:
        logging.error(f"Failed to save CSV: {e}")

    print("\n" + "=" * 100)
    print("🏆 Top 10 Configurations by Adjusted_PF 🏆")
    print("=" * 100)

    filtered_df = res_df.filter(pl.col("Total_Bets") >= MIN_TOTAL_BETS)

    if filtered_df.height > 0:
        print(f"(Filtered by Total_Bets >= {MIN_TOTAL_BETS})")
        print(filtered_df.head(10).to_pandas().to_string(index=False))
    else:
        print(
            f"⚠️ 指定された最低取引回数（{MIN_TOTAL_BETS}）を満たす組み合わせがありませんでした。"
        )
        print("(Showing unfiltered top 10 instead)")
        print(res_df.head(10).to_pandas().to_string(index=False))
    print("=" * 100)


if __name__ == "__main__":
    run_fast_grid_search()
