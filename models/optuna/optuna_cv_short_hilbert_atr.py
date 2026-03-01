# /workspace/models/optuna_cv_hilbert_atr.py

import sys
import logging
import gc
from pathlib import Path
import datetime
from datetime import timedelta, timezone
from typing import List, Tuple, Generator, Dict

import polars as pl
import pandas as pd
import numpy as np
import optuna
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler

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
    from blueprint import S2_FEATURES_FIXED
except ImportError:
    logging.warning("blueprint.py not found. Using fallback paths.")
    S2_FEATURES_FIXED = Path("/workspace/data/XAUUSD/stratum_2_features_fixed")

# --- ロギングと定数設定 ---
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

# 保存名をショート用に変更
RESULTS_CSV_PATH = Path(
    "/workspace/data/XAUUSD/stratum_7_models/1A_2B/optuna_top100_hilbert_atr_results_short.csv"
)
RESULTS_CSV_PATH.parent.mkdir(parents=True, exist_ok=True)

SPREAD_COST = 0.16
COOL_DOWN_MINUTES = 1
HILBERT_COL = "hilbert_amplitude_50"

# --- 探索空間 (Search Space) ---
TIMEFRAMES = [
    "M1",
    "M3",
    "M5",
    "M8",
    "M15",
    "H1",
    "H4",
    "H6",
    "H12",
    "D1",
]
TIMEFRAMES_WINDOW_MAP = {
    "M1": "1m",
    "M3": "3m",
    "M5": "5m",
    "M8": "8m",
    "M15": "15m",
    "H1": "1h",
    "H4": "4h",
    "H6": "6h",
    "H12": "12h",
    "D1": "1d",
}

ATR_PERIODS = [13]
# 探索パラメータ自体は維持し、ロジック内で逆数(1/tr)として使用する
TRIGGER_RATIOS = [1.03, 1.05, 1.08, 1.10, 1.12, 1.15]
ATR_FILTER_RATIOS = [
    0.1,
    0.3,
    0.5,
    0.8,
    1.0,
    1.5,
]
PT_MULTS = [0.5, 1.0, 2.5, 5.0]
SL_MULTS = [0.5, 1.0, 2.5, 5.0]
TD_MINS = [5, 15, 30, 60, 120, 180, 360, 720, 1080, 1200]

# --- CV・最適化設定 ---
K_FOLDS = 5
PURGE_DAYS = 3
EMBARGO_DAYS = 2
N_TRIALS = 1000
MIN_TOTAL_BETS_PER_FOLD = 100

SEARCH_TIMEFRAMES = TIMEFRAMES + ["mixed"]


# ====================================================================
# Numba JIT 高速トリプルバリア関数 (ショート用に反転)
# ====================================================================
@njit(parallel=True, fastmath=True, cache=True)
def _numba_find_hits_fast(
    bets_t0,
    bets_t1_max,
    bets_pt_barrier,
    bets_sl_barrier,
    ticks_ts,
    ticks_high,
    ticks_low,
):
    n_bets = len(bets_t0)
    n_ticks = len(ticks_ts)
    out_first_pt_time = np.zeros(n_bets, dtype=np.int64)
    out_first_sl_time = np.zeros(n_bets, dtype=np.int64)

    for i in prange(n_bets):
        t0 = bets_t0[i]
        t1_max = bets_t1_max[i]
        pt = bets_pt_barrier[i]  # ショートの利確ライン（エントリーより下）
        sl = bets_sl_barrier[i]  # ショートの損切ライン（エントリーより上）

        start_idx = np.searchsorted(ticks_ts, t0, side="left")
        first_pt_found = np.int64(0)
        first_sl_found = np.int64(0)

        for j in range(start_idx, n_ticks):
            tick_time = ticks_ts[j]
            if tick_time > t1_max:
                break
            tick_high = ticks_high[j]
            tick_low = ticks_low[j]

            # 【変更点】ショート: tick_low が pt を下回ったら利確
            if first_pt_found == 0 and tick_low <= pt:
                first_pt_found = tick_time
            # 【変更点】ショート: tick_high が sl を上回ったら損切
            if first_sl_found == 0 and tick_high >= sl:
                first_sl_found = tick_time

            # 同時足で両方到達した場合はループを抜け、呼び出し元で保守的にLoss扱いとなる
            if first_pt_found != 0 and first_sl_found != 0:
                break

        out_first_pt_time[i] = first_pt_found
        out_first_sl_time[i] = first_sl_found

    return out_first_pt_time, out_first_sl_time


# ====================================================================
# CV スプリットクラス
# ====================================================================
class PartitionPurgedKFold:
    def __init__(self, n_splits=5, purge_days=3, embargo_days=2):
        self.n_splits = n_splits
        self.purge_days = purge_days
        self.embargo_days = embargo_days

    def split(self, partitions):
        n_partitions = len(partitions)
        fold_size = n_partitions // self.n_splits
        for i in range(self.n_splits):
            start = i * fold_size
            end = start + fold_size if i < self.n_splits - 1 else n_partitions
            test_partitions = partitions[start:end]
            if not test_partitions:
                continue

            test_start_date, test_end_date = test_partitions[0], test_partitions[-1]
            purge_start = test_start_date - datetime.timedelta(days=self.purge_days)
            embargo_end = test_end_date + datetime.timedelta(days=self.embargo_days)
            train_partitions = [
                p for p in partitions if not (purge_start <= p <= embargo_end)
            ]
            yield train_partitions, test_partitions


# ====================================================================
# データローダークラス (デュアルキャッシュ対応・メモリリーク修正版)
# ====================================================================
class FoldDataLoader:
    def __init__(self, base_tick_dir: Path):
        self.base_tick_dir = base_tick_dir
        self.tick_cache = {}  # 日付ベースのキャッシュ (重いティックデータ用)
        self.signal_cache = {}  # 日付＋TFベースのキャッシュ (軽いシグナルデータ用)

    def clear_all_caches(self):
        """タイムフレーム移行時にRAMを解放するためのヘルパーメソッド"""
        self.tick_cache.clear()
        self.signal_cache.clear()
        gc.collect()

    def get_fold_data(self, test_dates, tf, atr_p):
        start_date = datetime.datetime.combine(
            test_dates[0], datetime.time.min, tzinfo=timezone.utc
        )
        end_date = datetime.datetime.combine(
            test_dates[-1] + timedelta(days=1), datetime.time.min, tzinfo=timezone.utc
        )

        date_key = f"{start_date.strftime('%Y%m%d')}_{end_date.strftime('%Y%m%d')}"
        signal_key = f"{date_key}_{tf}_{atr_p}"

        # ----------------------------------------------------------------
        # 1. ティックデータの読み込みとキャッシュ (TFに関係なくFoldごとに1回だけ実行)
        # ----------------------------------------------------------------
        if date_key not in self.tick_cache:
            logging.info(
                f"    [Tick Cache Miss] Loading TICK data for dates {date_key}..."
            )
            tick_end_date = end_date + timedelta(days=2)
            lf_tick = (
                pl.scan_parquet(str(self.base_tick_dir / "**/*.parquet"))
                .with_columns(pl.col("timestamp").cast(pl.Datetime("us", "UTC")))
                .filter(
                    (pl.col("timestamp") >= start_date)
                    & (pl.col("timestamp") < tick_end_date)
                )
                .sort("timestamp")
            )
            df_tick = lf_tick.select(["timestamp", "high", "low", "close"]).collect(
                engine="streaming"
            )
            if df_tick.is_empty():
                self.tick_cache[date_key] = None
            else:
                ticks_ts = np.ascontiguousarray(
                    df_tick.with_columns(pl.col("timestamp").dt.timestamp("us"))[
                        "timestamp"
                    ].to_numpy()
                )
                ticks_high = np.ascontiguousarray(df_tick["high"].to_numpy())
                ticks_low = np.ascontiguousarray(df_tick["low"].to_numpy())

                # シグナル生成時のスプレッド計算にもdf_tickが必要なため、一時的に保持
                self.tick_cache[date_key] = (ticks_ts, ticks_high, ticks_low, df_tick)

        tick_data = self.tick_cache.get(date_key)
        if tick_data is None:
            return None

        ticks_ts, ticks_high, ticks_low, df_tick = tick_data

        # ----------------------------------------------------------------
        # 2. シグナルデータの読み込みとキャッシュ (TFごとに実行、ただし非常に軽量)
        # ----------------------------------------------------------------
        if signal_key not in self.signal_cache:
            logging.info(
                f"    [Signal Cache Miss] Loading SIGNAL data for {tf}, ATR={atr_p}..."
            )
            hilbert_file, actual_hilbert, atr_file, actual_atr = None, None, None, None

            for p in S2_FEATURES_FIXED.rglob(f"features_*_{tf}.parquet"):
                if "tick" in str(p):
                    continue
                try:
                    cols = pl.scan_parquet(str(p)).collect_schema().names()
                    if not hilbert_file:
                        h_col = next((c for c in cols if HILBERT_COL in c), None)
                        if h_col:
                            hilbert_file, actual_hilbert = p, h_col
                    if not atr_file:
                        a_col = next((c for c in cols if f"atr_{atr_p}" in c), None)
                        if a_col:
                            atr_file, actual_atr = p, a_col
                except:
                    pass

            if not hilbert_file or not atr_file:
                self.signal_cache[signal_key] = None
                return None

            lf_hilbert = (
                pl.scan_parquet(str(hilbert_file))
                .select(["timestamp", actual_hilbert])
                .with_columns(pl.col("timestamp").cast(pl.Datetime("us", "UTC")))
                .filter(
                    (pl.col("timestamp") >= start_date)
                    & (pl.col("timestamp") < end_date)
                )
                .sort("timestamp")
                .with_columns(
                    pl.col(actual_hilbert)
                    .rolling_mean(window_size=50, min_samples=1)
                    .alias("hilbert_ma_50")
                )
                .with_columns(
                    (pl.col(actual_hilbert) / pl.col("hilbert_ma_50")).alias(
                        "trigger_ratio_val"
                    )
                )
            )

            lf_atr = (
                pl.scan_parquet(str(atr_file))
                .select(["timestamp", actual_atr])
                .with_columns(pl.col("timestamp").cast(pl.Datetime("us", "UTC")))
                .filter(
                    (pl.col("timestamp") >= start_date)
                    & (pl.col("timestamp") < end_date)
                )
                .sort("timestamp")
                .rename({actual_atr: f"atr_{atr_p}"})
            )

            spread_w = TIMEFRAMES_WINDOW_MAP[tf]
            # tick_cacheから取得したdf_tickを使ってスプレッドを計算
            lf_spread = (
                df_tick.lazy()
                .group_by_dynamic("timestamp", every=spread_w, closed="right")
                .agg(
                    [
                        pl.col("high").max().alias("bar_high"),
                        pl.col("low").min().alias("bar_low"),
                    ]
                )
                .with_columns((pl.col("bar_high") - pl.col("bar_low")).alias("spread"))
            )

            lf_sig = lf_hilbert.join_asof(
                df_tick.lazy().select(["timestamp", "close"]),
                on="timestamp",
                strategy="backward",
            )
            lf_sig = lf_sig.join_asof(
                lf_spread.select(["timestamp", "spread"]),
                on="timestamp",
                strategy="backward",
            )
            lf_sig = lf_sig.join_asof(lf_atr, on="timestamp", strategy="backward")

            lf_sig = (
                lf_sig.with_columns(
                    pl.col(f"atr_{atr_p}")
                    .fill_null(strategy="forward")
                    .fill_null(strategy="backward")
                )
                .drop_nulls(
                    subset=["close", "trigger_ratio_val", "spread", f"atr_{atr_p}"]
                )
                .with_columns(pl.col("timestamp").dt.timestamp("us").alias("ts_int"))
            )
            df_sig = lf_sig.collect(engine="streaming")

            if df_sig.is_empty():
                self.signal_cache[signal_key] = None
            else:
                sig_ts = np.ascontiguousarray(df_sig["ts_int"].to_numpy())
                sig_close = np.ascontiguousarray(df_sig["close"].to_numpy())
                sig_trigger = np.ascontiguousarray(
                    df_sig["trigger_ratio_val"].to_numpy()
                )
                sig_spread = np.ascontiguousarray(df_sig["spread"].to_numpy())
                sig_atr = np.ascontiguousarray(df_sig[f"atr_{atr_p}"].to_numpy())
                self.signal_cache[signal_key] = (
                    sig_ts,
                    sig_close,
                    sig_trigger,
                    sig_spread,
                    sig_atr,
                )

        signal_data = self.signal_cache.get(signal_key)
        if signal_data is None:
            return None

        sig_ts, sig_close, sig_trigger, sig_spread, sig_atr = signal_data

        # Numba関数（_numba_find_hits_fast）などで使えるように元の形式に結合して返す
        return (
            sig_ts,
            sig_close,
            sig_trigger,
            sig_spread,
            sig_atr,
            ticks_ts,
            ticks_high,
            ticks_low,
        )


# ====================================================================
# Optuna Objective (ショート用に反転)
# ====================================================================
def create_objective(cv_folds, data_loader, target_tf):
    def objective(trial):
        tf_choice = trial.suggest_categorical("timeframe", [target_tf])
        atr_p = trial.suggest_categorical("atr_period", ATR_PERIODS)
        tr = trial.suggest_categorical("trigger_ratio", TRIGGER_RATIOS)
        atr_f = trial.suggest_categorical("atr_filter", ATR_FILTER_RATIOS)
        pt_mult = trial.suggest_categorical("pt_mult", PT_MULTS)
        sl_mult = trial.suggest_categorical("sl_mult", SL_MULTS)
        td = trial.suggest_categorical("td_mins", TD_MINS)

        fold_scores = []
        cool_down_us = np.int64(COOL_DOWN_MINUTES * 60 * 1_000_000)
        td_us = np.int64(td * 60 * 1_000_000)

        total_bets_all = 0
        total_wins_all = 0
        total_losses_all = 0
        total_timeouts_all = 0
        total_sum_atr_all = 0.0
        total_np_all = 0.0  # ← 追加
        timeout_np_all = 0.0  # ← 追加

        for fold_idx, (_, test_dates) in enumerate(cv_folds):
            if tf_choice == "mixed":
                m_ts, m_close, m_trigger, m_spread, m_atr = [], [], [], [], []
                ticks_data = None
                for t in TIMEFRAMES:
                    fdata = data_loader.get_fold_data(test_dates, t, atr_p)
                    if fdata is not None:
                        m_ts.append(fdata[0])
                        m_close.append(fdata[1])
                        m_trigger.append(fdata[2])
                        m_spread.append(fdata[3])
                        m_atr.append(fdata[4])
                        if ticks_data is None:
                            ticks_data = fdata[5:]

                if not m_ts:
                    fold_scores.append(0.0)
                    continue

                sig_ts = np.concatenate(m_ts)
                sig_close = np.concatenate(m_close)
                sig_trigger = np.concatenate(m_trigger)
                sig_spread = np.concatenate(m_spread)
                sig_atr = np.concatenate(m_atr)

                sort_idx = np.argsort(sig_ts)
                sig_ts, sig_close, sig_trigger, sig_spread, sig_atr = (
                    sig_ts[sort_idx],
                    sig_close[sort_idx],
                    sig_trigger[sort_idx],
                    sig_spread[sort_idx],
                    sig_atr[sort_idx],
                )
                ticks_ts, ticks_high, ticks_low = ticks_data
            else:
                fold_data = data_loader.get_fold_data(test_dates, tf_choice, atr_p)
                if fold_data is None:
                    fold_scores.append(0.0)
                    continue
                (
                    sig_ts,
                    sig_close,
                    sig_trigger,
                    sig_spread,
                    sig_atr,
                    ticks_ts,
                    ticks_high,
                    ticks_low,
                ) = fold_data

            # 【変更点】ショート: trigger_ratioの逆数を閾値とし、下落方向のシグナルを抽出
            # ATRフィルタ等のボラティリティ条件は絶対値的な意味合いのため維持
            inverse_tr = 1.0 / tr
            mask = (
                (sig_trigger <= inverse_tr)
                & (sig_spread > (sig_atr * atr_f))
                & (sig_atr > 0)
            )

            if not np.any(mask):
                fold_scores.append(0.0)
                continue

            raw_t0, raw_close, raw_atr = sig_ts[mask], sig_close[mask], sig_atr[mask]

            valid_indices = []
            last_time = np.int64(-cool_down_us)
            for idx, t in enumerate(raw_t0):
                if t - last_time >= cool_down_us:
                    valid_indices.append(idx)
                    last_time = t

            if len(valid_indices) < MIN_TOTAL_BETS_PER_FOLD:
                fold_scores.append(0.0)
            else:
                bets_t0 = np.ascontiguousarray(raw_t0[valid_indices])
                bets_close = np.ascontiguousarray(raw_close[valid_indices])
                bets_atr = np.ascontiguousarray(raw_atr[valid_indices])

                bets_t1_max = bets_t0 + td_us

                # 【変更点】ショート: PTは下方向、SLは上方向に計算
                bets_pt = np.ascontiguousarray(bets_close - bets_atr * pt_mult)
                bets_sl = np.ascontiguousarray(bets_close + bets_atr * sl_mult)

                out_pt, out_sl = _numba_find_hits_fast(
                    bets_t0,
                    bets_t1_max,
                    bets_pt,
                    bets_sl,
                    ticks_ts,
                    ticks_high,
                    ticks_low,
                )

                # 勝敗判定ロジック自体は不変（PTに先に当たればWin, SLに先に当たればLoss）
                is_win = (out_pt > 0) & ((out_sl == 0) | (out_pt < out_sl))
                is_loss = (out_sl > 0) & ((out_pt == 0) | (out_sl <= out_pt))
                timeouts = int(np.sum(~(is_win | is_loss)))

                wins = int(np.sum(is_win))
                losses = int(np.sum(is_loss))
                total_profit = float(
                    np.sum(bets_atr[is_win].astype(np.float64)) * pt_mult
                )
                total_loss_base = float(
                    np.sum(bets_atr[is_loss].astype(np.float64)) * sl_mult
                )
                total_loss_adj = total_loss_base + (timeouts * SPREAD_COST)

                adj_pf = (
                    total_profit / total_loss_adj
                    if total_loss_adj > 0
                    else (float("inf") if total_profit > 0 else 1.0)
                )
                fold_scores.append(adj_pf)

                total_bets_all += len(valid_indices)
                total_wins_all += wins
                total_losses_all += losses
                total_timeouts_all += timeouts
                total_sum_atr_all += float(np.sum(bets_atr))
                total_np_all += total_profit - total_loss_adj  # ← 追加: 純利益の加算
                timeout_np_all += -(timeouts * SPREAD_COST)  # ← 追加: Timeout損失の加算

            current_mean_score = np.mean(fold_scores)
            trial.report(current_mean_score, step=fold_idx)

            if trial.should_prune():
                raise optuna.TrialPruned()

        trial.set_user_attr("Total_Bets", total_bets_all)
        trial.set_user_attr("Wins", total_wins_all)
        trial.set_user_attr("Losses", total_losses_all)
        trial.set_user_attr("Timeouts", total_timeouts_all)

        avg_atr = total_sum_atr_all / total_bets_all if total_bets_all > 0 else 0
        trial.set_user_attr("Avg_ATR", round(avg_atr, 4))
        trial.set_user_attr(
            "Avg_Payoff", round(pt_mult / sl_mult, 4) if sl_mult > 0 else 0
        )
        trial.set_user_attr(
            "Total_NP", total_np_all * 100 * 150
        )  # ← 追加: 100(Lot) * 150(JPY)
        trial.set_user_attr(
            "Timeout_NP", timeout_np_all * 100 * 150
        )  # ← 追加: 100(Lot) * 150(JPY)

        return np.mean(fold_scores)

    return objective


# ====================================================================
# メイン実行ブロック
# ====================================================================
def run_optimization():
    tick_dir = S2_FEATURES_FIXED / "feature_value_a_vast_universeC/features_e1c_tick"

    logging.info("Discovering partitions...")
    partitions = []
    for p in tick_dir.rglob("month=*"):
        try:
            y, m = int(p.parent.name.split("=")[1]), int(p.name.split("=")[1])
            for day_dir in p.glob("day=*"):
                partitions.append(datetime.date(y, m, int(day_dir.name.split("=")[1])))
        except:
            continue

    partitions = sorted(list(set(partitions)))
    if not partitions:
        return

    kfold = PartitionPurgedKFold(
        n_splits=K_FOLDS, purge_days=PURGE_DAYS, embargo_days=EMBARGO_DAYS
    )
    cv_folds = list(kfold.split(partitions))
    data_loader = FoldDataLoader(tick_dir)

    final_df_list = []
    cols_to_print = [
        "timeframe",
        "trigger_ratio",
        "atr_filter",
        "pt_mult",
        "sl_mult",
        "td_mins",
        "Total_Bets",
        "Wins",
        "Losses",
        "Timeouts",
        "Avg_ATR",
        "Avg_Payoff",
        "Total_NP",  # ← 追加
        "Timeout_NP",  # ← 追加
        "Adjusted_PF",
    ]

    # ★ ここでタイムフレームごとに独立してOptunaを回す
    for target_tf in SEARCH_TIMEFRAMES:
        # # ★ 追加：新しいタイムフレームの開始前にキャッシュを完全にクリアしてRAMを解放
        # data_loader.clear_all_caches()

        logging.info(
            f"=== Starting Optuna SHORT Optimization for {target_tf} (Trials: {N_TRIALS}) ==="
        )
        sampler = TPESampler(n_startup_trials=300, multivariate=True)
        # Study名に_SHORTを追加
        study = optuna.create_study(
            study_name=f"short_optimization_{target_tf}",
            direction="maximize",
            sampler=sampler,
            pruner=MedianPruner(
                n_startup_trials=20, n_warmup_steps=1, interval_steps=1
            ),
        )

        study.optimize(
            create_objective(cv_folds, data_loader, target_tf),
            n_trials=N_TRIALS,
            gc_after_trial=True,
        )

        df = study.trials_dataframe(attrs=("value", "params", "user_attrs", "state"))
        df = df[df["state"] == "COMPLETE"].copy()

        if len(df) == 0:
            logging.warning(f"No completed trials for {target_tf}.")
            continue

        rename_dict = {"value": "Adjusted_PF"}
        param_cols = []
        for col in df.columns:
            if col.startswith("params_"):
                cleaned_name = col.replace("params_", "")
                rename_dict[col] = cleaned_name
                param_cols.append(cleaned_name)
            elif col.startswith("user_attrs_"):
                rename_dict[col] = col.replace("user_attrs_", "")
        df = df.rename(columns=rename_dict)

        df = df.sort_values("Adjusted_PF", ascending=False)
        df = df.drop_duplicates(subset=param_cols, keep="first")

        top_100 = df.head(100)
        final_df_list.append(top_100)

        print("\n" + "=" * 90)
        print(f"🏆 {target_tf} SHORT Top 10 Configurations 🏆")
        print("=" * 90)
        print(top_100.head(10)[cols_to_print].to_string(index=False))

    if final_df_list:
        final_csv_df = pl.DataFrame(
            pd.concat(final_df_list, ignore_index=True)[cols_to_print]
        )
        final_csv_df.write_csv(RESULTS_CSV_PATH)
        print("\n" + "=" * 90)
        logging.info(
            f"Top 100 SHORT results for each timeframe successfully saved to: {RESULTS_CSV_PATH}"
        )


if __name__ == "__main__":
    run_optimization()
