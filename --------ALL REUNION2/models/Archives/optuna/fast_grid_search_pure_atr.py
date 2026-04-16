# /workspace/models/fast_grid_search_pure_atr.py

import sys
import logging
import gc
from pathlib import Path
import itertools
import time
import polars as pl
import numpy as np

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
    "/workspace/data/XAUUSD/stratum_7_models/1A_2B/fast_grid_search_pure_atr_results.csv"
)
RESULTS_CSV_PATH.parent.mkdir(parents=True, exist_ok=True)

SPREAD_COST = 0.16
COOL_DOWN_MINUTES = 1
MIN_TOTAL_BETS = 100

# ====================================================================
# 【ユーザー設定エリア】 グリッドサーチの設定
# ====================================================================
ALL_AVAILABLE_TFS = [
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
    "W1",
    "MN",
]

# ★ 評価対象の時間足を指定します。
# 個別評価の場合: ["M5", "M15"] のようにリストで指定（個別にCSV出力されます）
# 混合(全時間足)評価の場合: ["all"] と指定（全時間足のシグナルをガッチャンコします）
TIMEFRAMES = ["all"]

# ★ 評価対象の期間を指定します（"all" または "YYYY/M"）
# TEST_PERIOD = "all"
TEST_PERIOD = "2025/9"

ATR_PERIODS = [
    # 13,
    21,
    # 34, 55
]
ATR_THRESHOLDS = [
    # 0.5, 1.0, 1.5, 2.0, 3.0,
    5.0,
    8.0,
    10.0,
    15.0,
]

PT_MULTS = [
    1.0,
    # 2.5, 5.0, 10.0
]
SL_MULTS = [
    # 1.0, 2.5,
    5.0,
    # 10.0
]
TD_MINS = [
    # 3, 5, 10, 30, 60, 120, 180,720,
    1200
]


# ====================================================================
# Numba JIT 高速トリプルバリア関数 (維持)
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
# メイン処理ブロック
# ====================================================================
def load_and_prepare_data(target_timeframe: str):
    """単一の時間足のATRデータとTickデータを結合して返す。戻り値: (シグナル用DF, 全Tick価格DF)"""
    tick_dir = S2_FEATURES_FIXED / "feature_value_a_vast_universeC/features_e1c_tick"

    lf_tick = pl.scan_parquet(str(tick_dir / "**/*.parquet"))
    required_tick_cols = ["timestamp", "open", "high", "low", "close"]

    lf_tick = lf_tick.select(required_tick_cols).with_columns(
        pl.col("timestamp").cast(pl.Datetime("us", "UTC"))
    )

    if TEST_PERIOD != "all":
        year, month = map(int, TEST_PERIOD.split("/"))
        lf_tick = lf_tick.filter(
            (pl.col("timestamp").dt.year() == year)
            & (pl.col("timestamp").dt.month() == month)
        )

    lf_tick = lf_tick.sort("timestamp")

    def _get_atr_lf(tf):
        for p in S2_FEATURES_FIXED.rglob(f"features_*_{tf}.parquet"):
            if "tick" in str(p):
                continue
            try:
                t_cols = pl.scan_parquet(str(p)).collect_schema().names()
                m_cols = [
                    c for c in t_cols if any(f"atr_{a}" in c for a in ATR_PERIODS)
                ]
                if len(m_cols) >= len(ATR_PERIODS):
                    lf = (
                        pl.scan_parquet(str(p))
                        .select(["timestamp"] + m_cols)
                        .with_columns(
                            pl.col("timestamp").cast(pl.Datetime("us", "UTC"))
                        )
                        .sort("timestamp")
                    )
                    renames = {
                        c: f"atr_{a}_{target_timeframe}"
                        for c in m_cols
                        for a in ATR_PERIODS
                        if f"atr_{a}" in c
                    }
                    return lf.rename(renames), list(renames.values())
            except Exception:
                pass
        return None, []

    # ATRロード
    lf_atr, atr_cols = _get_atr_lf(target_timeframe)
    if not atr_cols:
        logging.warning(f"ATR columns not found for {target_timeframe}.")
        return pl.DataFrame(), pl.DataFrame()

    # S5ロード
    # S5ディレクトリ以下から、対象時間足（例: _M1_）のParquetファイルを横断検索
    s5_files = list(
        S5_NEUTRALIZED_ALPHA_SET.rglob(
            f"features_*_{target_timeframe}_neutralized.parquet"
        )
    )

    if not s5_files:
        # 拡張子なしのフォルダ形式になっているケースも念のため検索
        s5_dirs = list(
            S5_NEUTRALIZED_ALPHA_SET.rglob(f"features_*_{target_timeframe}_neutralized")
        )
        s5_files = [d for d in s5_dirs if d.is_dir()]

    if not s5_files:
        logging.warning(f"S5 alpha file not found for {target_timeframe}.")
        return pl.DataFrame(), pl.DataFrame()

    s5_path = s5_files[0]  # 最初に見つかったファイルをロード
    logging.info(f"Loaded S5 alpha file: {s5_path.name}")

    lf_s5 = (
        pl.scan_parquet(str(s5_path / "**/*.parquet"))
        if s5_path.is_dir()
        else pl.scan_parquet(str(s5_path))
    )
    lf_s5 = lf_s5.with_columns(pl.col("timestamp").cast(pl.Datetime("us", "UTC"))).sort(
        "timestamp"
    )
    # シグナル用データの作成（S5にTickとATRを紐付け）
    lf_sig = lf_s5.join_asof(lf_tick.drop("open"), on="timestamp", strategy="backward")
    lf_sig = lf_sig.join_asof(lf_atr, on="timestamp", strategy="backward")

    fill_exprs = [
        pl.col(c).fill_null(strategy="forward").fill_null(strategy="backward")
        for c in atr_cols
    ]
    lf_sig = (
        lf_sig.with_columns(fill_exprs)
        .drop_nulls(subset=["close"] + atr_cols)
        .with_columns(pl.col("timestamp").dt.timestamp("us").alias("ts_int"))
    )

    float_cols = ["high", "low", "close"] + atr_cols
    lf_sig = lf_sig.with_columns([pl.col(c).cast(pl.Float32) for c in float_cols])

    # 重要：2つの DataFrame を返す
    return lf_sig.collect(engine="streaming"), lf_tick.collect(engine="streaming")


def evaluate_grid(
    tf_label: str,
    atr_p: int,
    atr_threshold: float,
    raw_t0: np.ndarray,
    raw_close: np.ndarray,
    raw_atr: np.ndarray,
    ticks_ts: np.ndarray,
    ticks_high: np.ndarray,
    ticks_low: np.ndarray,
    results_list: list,
):
    """生シグナル配列を受け取り、重複排除からNumba処理、結果の格納までを一括して行う"""

    # 1. 重複エントリーの制御 (本番の挙動を模倣)
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

    avg_atr = float(np.mean(bets_atr)) if triggered_count > 0 else 0.0

    # 2. TD, PT, SL のグリッドサーチ
    for td in TD_MINS:
        td_us = np.int64(td * 60 * 1_000_000)
        bets_t1_max = bets_t0 + td_us

        for pt_mult, sl_mult in itertools.product(PT_MULTS, SL_MULTS):
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
            is_timeout = ~(is_win | is_loss)

            wins = int(np.sum(is_win))
            losses = int(np.sum(is_loss))
            timeouts = int(np.sum(is_timeout))
            win_rate = wins / triggered_count if triggered_count > 0 else 0.0

            total_profit = float(np.sum(bets_atr[is_win].astype(np.float64)) * pt_mult)
            total_loss_base = float(
                np.sum(bets_atr[is_loss].astype(np.float64)) * sl_mult
            )
            total_loss_adj = total_loss_base + (timeouts * SPREAD_COST)

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
            avg_payoff = pt_mult / sl_mult

            results_list.append(
                {
                    "Timeframe": tf_label,
                    "ATR_Period": atr_p,
                    "ATR_Threshold": atr_threshold,
                    "Avg_ATR": round(avg_atr, 4),
                    "PT": pt_mult,
                    "SL": sl_mult,
                    "TD_min": td,
                    "Total_Bets": triggered_count,
                    "Wins": wins,
                    "Losses": losses,
                    "Timeouts": timeouts,
                    "WinRate": round(win_rate, 4),
                    "Avg_Payoff": round(avg_payoff, 4),
                    "PF": round(pf, 4),
                    "Adjusted_PF": round(adj_pf, 4),
                }
            )


def run_fast_grid_search():
    is_mixed_mode = len(TIMEFRAMES) == 1 and TIMEFRAMES[0].lower() == "all"
    target_tfs = ALL_AVAILABLE_TFS if is_mixed_mode else TIMEFRAMES

    logging.info(
        f"Starting Pure ATR Grid Search. Mode: {'MIXED (all)' if is_mixed_mode else 'INDIVIDUAL'}"
    )

    results = []
    global_ticks = {}
    mixed_signals = {}  # 混合モード用: (atr_p, atr_threshold) をキーに、全時間足のシグナルを蓄積

    start_time = time.time()

    for tf in target_tfs:
        logging.info(f"--- Loading and extracting signals for Timeframe: {tf} ---")
        gc.collect()
        time.sleep(1)

        # 修正1: 戻り値(シグナルDFと全TickDF)を正しく受け取る
        df_sig, df_tick_full = load_and_prepare_data(tf)  # 戻り値2つを受け取る

        # 修正2: データが空（ファイル不在時など）の場合は安全にスキップ
        if df_sig.is_empty():  # df ではなく df_sig
            logging.info(f"Skipping {tf} due to missing data.")
            continue

        # 修正3: Numba評価用の価格履歴は、全解像度のデータ(df_tick_full)から一度だけ作成
        if not global_ticks:
            # 評価用の全価格推移が必要なので、シグナル用ではなく df_tick_full を使う
            df_tick_temp = df_tick_full.with_columns(
                pl.col("timestamp").dt.timestamp("us").alias("ts_int")
            )
            global_ticks["ts"] = np.ascontiguousarray(df_tick_temp["ts_int"].to_numpy())
            global_ticks["high"] = np.ascontiguousarray(df_tick_temp["high"].to_numpy())
            global_ticks["low"] = np.ascontiguousarray(df_tick_temp["low"].to_numpy())
        # 修正4: エラー原因の `df` を `df_sig` に置換
        ticks_ts = np.ascontiguousarray(df_sig["ts_int"].to_numpy())
        df_close_np = df_sig["close"].to_numpy()

        for atr_p in ATR_PERIODS:
            col_name = f"atr_{atr_p}_{tf}"
            # 静的チェック(F821)対策: df_sig を参照するように修正
            if col_name not in df_sig.columns:
                continue
            df_atr_np = df_sig[col_name].to_numpy()

            for thr in ATR_THRESHOLDS:
                # Numpy用のNaNチェックで、S5にデータがある時のみに限定
                mask = (df_atr_np >= thr) & (~np.isnan(df_close_np))

                if np.any(mask):
                    raw_t0 = ticks_ts[mask]
                    raw_close = df_close_np[mask]
                    raw_atr = df_atr_np[mask]

                    if is_mixed_mode:
                        # 混合モード: シグナルを配列のまま一旦プールする
                        mixed_signals.setdefault((atr_p, thr), []).append(
                            (raw_t0, raw_close, raw_atr)
                        )
                    else:
                        # 個別モード: 直ちに評価して CSV出力行を生成する
                        evaluate_grid(
                            tf_label=tf,
                            atr_p=atr_p,
                            atr_threshold=thr,
                            raw_t0=raw_t0,
                            raw_close=raw_close,
                            raw_atr=raw_atr,
                            ticks_ts=global_ticks["ts"],
                            ticks_high=global_ticks["high"],
                            ticks_low=global_ticks["low"],
                            results_list=results,
                        )

        # 時間足が変わる前に、メモリを食う DataFrame を確実に破棄
        # 修正：df を df_sig と df_tick_full に変更
        del df_sig, df_tick_full, ticks_ts, df_close_np
        gc.collect()

    # 混合モードの場合、プールした全時間足のシグナルを結合・ソートして評価する
    if is_mixed_mode:
        logging.info("--- Evaluating Combined Signals across all Timeframes ---")
        total_combos = len(mixed_signals)
        evaluated = 0

        for (atr_p, thr), sig_list in mixed_signals.items():
            # 複数時間足からのシグナルをガッチャンコ
            all_t0 = np.concatenate([s[0] for s in sig_list])
            all_close = np.concatenate([s[1] for s in sig_list])
            all_atr = np.concatenate([s[2] for s in sig_list])

            # ★重要: ポートフォリオ運用の重複判定を正しく行うため、時間順に並べ替える
            sort_idx = np.argsort(all_t0)

            evaluate_grid(
                tf_label="all_mixed",
                atr_p=atr_p,
                atr_threshold=thr,
                raw_t0=all_t0[sort_idx],
                raw_close=all_close[sort_idx],
                raw_atr=all_atr[sort_idx],
                ticks_ts=global_ticks["ts"],
                ticks_high=global_ticks["high"],
                ticks_low=global_ticks["low"],
                results_list=results,
            )

            evaluated += 1
            if evaluated % 10 == 0:
                logging.info(
                    f"Mixed Evaluation: {evaluated} / {total_combos} ATR combinations completed..."
                )

    elapsed = time.time() - start_time
    logging.info(f"Grid Search completed in {elapsed:.2f} seconds.")

    if not results:
        logging.warning("No results generated.")
        return

    res_df = (
        pl.DataFrame(results)
        .select(
            [
                "Timeframe",
                "ATR_Period",
                "ATR_Threshold",
                "Avg_ATR",
                "PT",
                "SL",
                "TD_min",
                "Total_Bets",
                "Wins",
                "Losses",
                "Timeouts",
                "WinRate",
                "Avg_Payoff",
                "PF",
                "Adjusted_PF",
            ]
        )
        .sort("Adjusted_PF", descending=True)
    )

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
