# /workspace/models/optuna_cv_pure_atr.py

import sys
import logging
import gc
from pathlib import Path
import datetime
from datetime import timedelta, timezone

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
    "/workspace/data/XAUUSD/stratum_7_models/1A_2B/optuna_top100_pure_atr_results.csv"
)
RESULTS_CSV_PATH.parent.mkdir(parents=True, exist_ok=True)

SPREAD_COST = 0.16
COOL_DOWN_MINUTES = 1

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

# ピュアATR用の探索パラメータ
ATR_PERIODS = [
    13
    #    , 21, 34
]
# 低い時間足(M1~M15等)のために、小さい閾値も含めることを推奨します
ATR_THRESHOLDS = [0.5, 1.0, 2.0, 3.0, 5.0, 8.0, 10.0, 15.0]

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
# Numba JIT 高速トリプルバリア関数
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
# データローダークラス (メモリ不足対策のためキャッシュ構造を分離)
# ====================================================================
class FoldDataLoader:
    def __init__(self, base_tick_dir: Path):
        self.base_tick_dir = base_tick_dir
        self.tick_df_cache = {}  # 結合用のClose価格専用キャッシュ
        self.tick_np_cache = {}  # Numba用の巨大Tick配列キャッシュ
        self.sig_cache = {}  # S5 + ATRの軽量シグナルキャッシュ

    # ★ 追加：キャッシュをクリアするメソッド
    def clear_all_caches(self):
        self.tick_df_cache.clear()
        self.tick_np_cache.clear()
        self.sig_cache.clear()
        gc.collect()

    def get_fold_data(self, test_dates, tf, atr_p):
        start_date = datetime.datetime.combine(
            test_dates[0], datetime.time.min, tzinfo=timezone.utc
        )
        end_date = datetime.datetime.combine(
            test_dates[-1] + timedelta(days=1), datetime.time.min, tzinfo=timezone.utc
        )

        fold_key = f"{start_date.strftime('%Y%m%d')}_{end_date.strftime('%Y%m%d')}"
        sig_key = f"{fold_key}_{tf}_{atr_p}"

        # 1. 巨大なTickデータは Fold単位で1回だけロードして共有する (メモリ節約の要)
        if fold_key not in self.tick_np_cache:
            logging.info(
                f"    [Cache Miss] Loading Raw Tick Data for Fold {fold_key}..."
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
                self.tick_np_cache[fold_key] = None
                self.tick_df_cache[fold_key] = None
            else:
                self.tick_df_cache[fold_key] = df_tick.select(["timestamp", "close"])
                self.tick_np_cache[fold_key] = (
                    np.ascontiguousarray(
                        df_tick.with_columns(pl.col("timestamp").dt.timestamp("us"))[
                            "timestamp"
                        ].to_numpy()
                    ),
                    np.ascontiguousarray(df_tick["high"].to_numpy()),
                    np.ascontiguousarray(df_tick["low"].to_numpy()),
                )
            del df_tick
            gc.collect()

        if self.tick_np_cache[fold_key] is None:
            return None

        # 2. シグナルデータのロード (S5 + ATR)
        if sig_key not in self.sig_cache:
            logging.info(
                f"    [Cache Miss] Loading Signal Data for {tf}, ATR={atr_p}..."
            )

            s5_files = list(
                S5_NEUTRALIZED_ALPHA_SET.rglob(f"features_*_{tf}_neutralized.parquet")
            )
            if not s5_files:
                s5_dirs = list(
                    S5_NEUTRALIZED_ALPHA_SET.rglob(f"features_*_{tf}_neutralized")
                )
                s5_files = [d for d in s5_dirs if d.is_dir()]

            atr_file, actual_atr = None, None
            for p in S2_FEATURES_FIXED.rglob(f"features_*_{tf}.parquet"):
                if "tick" in str(p):
                    continue
                try:
                    cols = pl.scan_parquet(str(p)).collect_schema().names()
                    a_col = next((c for c in cols if f"atr_{atr_p}" in c), None)
                    if a_col:
                        atr_file, actual_atr = p, a_col
                        break
                except:
                    pass

            if not s5_files or not atr_file:
                self.sig_cache[sig_key] = None
            else:
                s5_path = s5_files[0]
                lf_s5 = (
                    (
                        pl.scan_parquet(str(s5_path / "**/*.parquet"))
                        if s5_path.is_dir()
                        else pl.scan_parquet(str(s5_path))
                    )
                    .with_columns(pl.col("timestamp").cast(pl.Datetime("us", "UTC")))
                    .filter(
                        (pl.col("timestamp") >= start_date)
                        & (pl.col("timestamp") < end_date)
                    )
                    .sort("timestamp")
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

                df_tick_close = self.tick_df_cache[fold_key]
                lf_sig = lf_s5.join_asof(
                    df_tick_close.lazy(), on="timestamp", strategy="backward"
                )
                lf_sig = lf_sig.join_asof(lf_atr, on="timestamp", strategy="backward")

                lf_sig = (
                    lf_sig.with_columns(
                        pl.col(f"atr_{atr_p}")
                        .fill_null(strategy="forward")
                        .fill_null(strategy="backward")
                    )
                    .drop_nulls(subset=["close", f"atr_{atr_p}"])
                    .with_columns(
                        pl.col("timestamp").dt.timestamp("us").alias("ts_int")
                    )
                )

                df_sig = lf_sig.collect(engine="streaming")
                if df_sig.is_empty():
                    self.sig_cache[sig_key] = None
                else:
                    self.sig_cache[sig_key] = (
                        np.ascontiguousarray(df_sig["ts_int"].to_numpy()),
                        np.ascontiguousarray(df_sig["close"].to_numpy()),
                        np.ascontiguousarray(df_sig[f"atr_{atr_p}"].to_numpy()),
                    )
                del df_sig
                gc.collect()

        if self.sig_cache[sig_key] is None:
            return None

        # シグナル配列(3つ) と Tick配列(3つ) を結合して返す
        return self.sig_cache[sig_key] + self.tick_np_cache[fold_key]


# ====================================================================
# Optuna Objective
# ====================================================================
def create_objective(cv_folds, data_loader, target_tf):
    def objective(trial):
        tf_choice = trial.suggest_categorical("timeframe", [target_tf])
        atr_p = trial.suggest_categorical("atr_period", ATR_PERIODS)
        atr_threshold = trial.suggest_categorical("atr_threshold", ATR_THRESHOLDS)
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
        total_net_profit_all = 0.0  # 追加: 全トレード純利益の合算用
        total_timeout_np_all = 0.0  # 追加: Timeouts損益の合算用

        for fold_idx, (_, test_dates) in enumerate(cv_folds):
            if tf_choice == "mixed":
                m_ts, m_close, m_atr = [], [], []
                ticks_data = None
                for t in TIMEFRAMES:
                    fdata = data_loader.get_fold_data(test_dates, t, atr_p)
                    if fdata is not None:
                        m_ts.append(fdata[0])
                        m_close.append(fdata[1])
                        m_atr.append(fdata[2])
                        if ticks_data is None:
                            ticks_data = fdata[3:]

                if not m_ts:
                    fold_scores.append(0.0)
                    continue

                sig_ts = np.concatenate(m_ts)
                sig_close = np.concatenate(m_close)
                sig_atr = np.concatenate(m_atr)

                sort_idx = np.argsort(sig_ts)
                sig_ts, sig_close, sig_atr = (
                    sig_ts[sort_idx],
                    sig_close[sort_idx],
                    sig_atr[sort_idx],
                )
                ticks_ts, ticks_high, ticks_low = ticks_data
            else:
                fold_data = data_loader.get_fold_data(test_dates, tf_choice, atr_p)
                if fold_data is None:
                    fold_scores.append(0.0)
                    continue
                (sig_ts, sig_close, sig_atr, ticks_ts, ticks_high, ticks_low) = (
                    fold_data
                )

            # フィルタリング: ATRが閾値以上かのみ判定
            mask = sig_atr >= atr_threshold
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
                bets_pt = np.ascontiguousarray(bets_close + bets_atr * pt_mult)
                bets_sl = np.ascontiguousarray(bets_close - bets_atr * sl_mult)

                out_pt, out_sl = _numba_find_hits_fast(
                    bets_t0,
                    bets_t1_max,
                    bets_pt,
                    bets_sl,
                    ticks_ts,
                    ticks_high,
                    ticks_low,
                )

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
                total_net_profit_all += (
                    total_profit - total_loss_adj
                )  # 追加: 純利益を加算
                total_timeout_np_all += -(
                    timeouts * SPREAD_COST
                )  # 追加: Timeout損失を加算

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
            "Total_NP", round(total_net_profit_all * 100 * 150, 0)
        )  # 追加: *100*150して記録
        trial.set_user_attr(
            "Timeout_NP", round(total_timeout_np_all * 100 * 150, 0)
        )  # 追加: *100*150して記録

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
        logging.warning("No partitions found. Exiting.")
        return

    kfold = PartitionPurgedKFold(
        n_splits=K_FOLDS, purge_days=PURGE_DAYS, embargo_days=EMBARGO_DAYS
    )
    cv_folds = list(kfold.split(partitions))
    data_loader = FoldDataLoader(tick_dir)

    final_df_list = []
    cols_to_print = [
        "timeframe",
        "atr_period",
        "atr_threshold",
        "pt_mult",
        "sl_mult",
        "td_mins",
        "Total_Bets",
        "Wins",
        "Losses",
        "Timeouts",
        "Avg_ATR",
        "Avg_Payoff",
        "Total_NP",  # 追加
        "Timeout_NP",  # 追加
        "Adjusted_PF",
    ]

    for target_tf in SEARCH_TIMEFRAMES:
        # # ★ 追加：タイムフレーム移行時にRAMを解放
        # data_loader.clear_all_caches()

        logging.info(
            f"=== Starting Optuna Optimization for {target_tf} (Trials: {N_TRIALS}) ==="
        )

        sampler = TPESampler(n_startup_trials=300, multivariate=True)
        study = optuna.create_study(
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
        print(f"🏆 {target_tf} Top 10 Configurations 🏆")
        print("=" * 90)
        print(top_100.head(10)[cols_to_print].to_string(index=False))

    if final_df_list:
        final_csv_df = pl.DataFrame(
            pd.concat(final_df_list, ignore_index=True)[cols_to_print]
        )
        final_csv_df.write_csv(RESULTS_CSV_PATH)
        print("\n" + "=" * 90)
        logging.info(
            f"Top 100 results for each timeframe successfully saved to: {RESULTS_CSV_PATH}"
        )


if __name__ == "__main__":
    run_optimization()
