# /workspace/models/fast_grid_search_hilbert.py

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
    from blueprint import S2_FEATURES_FIXED
except ImportError:
    logging.warning("blueprint.py not found. Using fallback paths.")
    S2_FEATURES_FIXED = Path("/workspace/data/XAUUSD/stratum_2_features_fixed")

# --- ロギングと定数設定 ---
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

RESULTS_CSV_PATH = Path(
    "/workspace/data/XAUUSD/stratum_7_models/1A_2B/fast_grid_search_results.csv"
)
RESULTS_CSV_PATH.parent.mkdir(parents=True, exist_ok=True)

SPREAD_COST = 0.16  # スプレッドコスト（タイムアウト時のペナルティとして考慮）
COOL_DOWN_MINUTES = 1  # 連続エントリーのクールダウン時間（分）
HILBERT_COL = "e1e_hilbert_amplitude_50"
ATR_COL = "e1c_atr_21"

# ====================================================================
# グリッドサーチの設定
# ====================================================================
TRIGGER_RATIOS = [1.03, 1.05, 1.08, 1.12, 1.15, 1.20, 1.25, 1.28, 1.30, 1.35]
ATR_FILTER_RATIOS = [0.5, 0.8, 1.0, 1.2]
PT_MULTS = [
    0.01,
    0.05,
    0.1,
    0.3,
    0.5,
    1.0,
    1.5,
    2.0,
    3.0,
    5.0,
    8.0,
    10.0,
    15.0,
    20.0,
    25.0,
    30.0,
    35.0,
    40.0,
    45.0,
    50.0,
    80.0,
    100.0,
]
SL_MULTS = [
    0.01,
    0.05,
    0.1,
    0.3,
    0.5,
    1.0,
    1.5,
    2.0,
    3.0,
    5.0,
    8.0,
    10.0,
    15.0,
    20.0,
    25.0,
    30.0,
    35.0,
    40.0,
    45.0,
    50.0,
    80.0,
    100.0,
]
TD_MINS = [1, 3, 5, 10, 50, 60, 360, 720, 1200]


# ====================================================================
# Numba JIT 高速トリプルバリア関数 (第4バリア除去のシンプル版)
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
    """
    メモリ上のTick配列を直接走査し、PT/SL到達時刻を並列で返す超高速関数
    """
    n_bets = len(bets_t0)
    n_ticks = len(ticks_ts)
    out_first_pt_time = np.zeros(n_bets, dtype=np.int64)
    out_first_sl_time = np.zeros(n_bets, dtype=np.int64)

    for i in prange(n_bets):
        t0 = bets_t0[i]
        t1_max = bets_t1_max[i]
        pt = bets_pt_barrier[i]
        sl = bets_sl_barrier[i]

        # バイナリサーチで該当時刻のインデックスを特定
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

            # 両方発火したら走査終了
            if first_pt_found != 0 and first_sl_found != 0:
                break

        out_first_pt_time[i] = first_pt_found
        out_first_sl_time[i] = first_sl_found

    return out_first_pt_time, out_first_sl_time


# ====================================================================
# メイン処理
# ====================================================================
def load_and_prepare_data() -> pl.DataFrame:
    """Polarsを用いて3つのデータソース(OHLC, V5, ATR)を結合・前処理する"""
    v5_dir = S2_FEATURES_FIXED / "v5_gatekeeper_ready"
    tick_dir = S2_FEATURES_FIXED / "feature_value_a_vast_universeC/features_e1c_tick"

    logging.info("1. Loading Parquet files...")

    # --- 1. 価格データ (Tick: OHLC) ---
    lf_tick = pl.scan_parquet(str(tick_dir / "**/*.parquet"))
    tick_cols = lf_tick.collect_schema().names()
    required_tick_cols = ["timestamp", "open", "high", "low", "close"]
    for c in required_tick_cols:
        if c not in tick_cols:
            raise ValueError(
                f"Required column '{c}' not found in Tick data ({tick_dir})"
            )
    lf_tick = (
        lf_tick.select(required_tick_cols)
        .with_columns(pl.col("timestamp").cast(pl.Datetime("us", "UTC")))
        .sort("timestamp")
    )

    # --- 2. ヒルベルトデータ (V5) ---
    lf_v5 = pl.scan_parquet(str(v5_dir / "**/*.parquet"))
    v5_cols = lf_v5.collect_schema().names()
    actual_hilbert = next((c for c in v5_cols if HILBERT_COL in c), None)
    if not actual_hilbert:
        raise ValueError(
            f"Required column '{HILBERT_COL}' not found in V5 data ({v5_dir})"
        )
    lf_v5 = (
        lf_v5.select(["timestamp", actual_hilbert])
        .with_columns(pl.col("timestamp").cast(pl.Datetime("us", "UTC")))
        .sort("timestamp")
    )

    # --- 3. ATRデータ探索 ---
    actual_atr = None
    lf_atr = None
    # S2ディレクトリ内のParquetファイルを走査し、直接カラム名を確認する
    for p in S2_FEATURES_FIXED.rglob("features_*.parquet"):
        if "tick" in str(p):
            continue
        try:
            temp_lf = pl.scan_parquet(str(p))
            temp_cols = temp_lf.collect_schema().names()
            actual_atr = next((c for c in temp_cols if ATR_COL in c), None)
            if actual_atr:
                # カラムが見つかったら、その親ディレクトリをデータソースとして読み込む
                lf_atr = (
                    pl.scan_parquet(str(p.parent / "**/*.parquet"))
                    .select(["timestamp", actual_atr])
                    .with_columns(pl.col("timestamp").cast(pl.Datetime("us", "UTC")))
                    .sort("timestamp")
                )
                logging.info(f"Found ATR data in: {p.parent}")
                break
        except Exception:
            pass

    if lf_atr is None or actual_atr is None:
        raise ValueError(f"Required column '{ATR_COL}' not found in any S2 directory.")

    logging.info("2. Executing join_asof on timestamp (Combining 3 sources)...")
    # Tickデータを主軸に、V5とATRをそれぞれ直近の時刻で結合
    df = lf_tick.join_asof(lf_v5, on="timestamp", strategy="backward")
    df = df.join_asof(lf_atr, on="timestamp", strategy="backward").collect()

    logging.info("3. Calculating dynamic Hilbert Trigger (50-period rolling mean)...")
    # 欠損値補完とトリガー倍率の計算
    df = (
        df.with_columns(
            pl.col(actual_atr)
            .fill_null(strategy="forward")
            .fill_null(strategy="backward")
        )
        .drop_nulls(subset=["close", actual_hilbert, actual_atr])
        .with_columns(
            pl.col(actual_hilbert)
            .rolling_mean(window_size=50, min_periods=1)
            .alias("hilbert_ma_50")
        )
        .with_columns(
            (pl.col(actual_hilbert) / pl.col("hilbert_ma_50")).alias(
                "trigger_ratio_val"
            )
        )
        # Numba用にtimestampを整数（マイクロ秒）に変換
        .with_columns(pl.col("timestamp").dt.timestamp("us").alias("ts_int"))
    )

    # 処理用にカラム名を標準化
    df = df.rename({actual_atr: "atr_value"})
    return df


def run_fast_grid_search():
    df = load_and_prepare_data()
    if df.is_empty():
        logging.error("No data available after joining. Exiting.")
        return

    logging.info(f"Data ready. Total Rows: {df.height:,}")

    # TickデータをNumpy配列化（再利用することでメモリを極限まで節約）
    ticks_ts = np.ascontiguousarray(df["ts_int"].to_numpy())
    ticks_high = np.ascontiguousarray(df["high"].to_numpy())
    ticks_low = np.ascontiguousarray(df["low"].to_numpy())

    df_close_np = df["close"].to_numpy()
    df_atr_np = df["atr_value"].to_numpy()
    df_trigger_np = df["trigger_ratio_val"].to_numpy()

    results = []
    total_combinations = (
        len(TRIGGER_RATIOS)
        * len(ATR_FILTER_RATIOS)
        * len(TD_MINS)
        * len(PT_MULTS)
        * len(SL_MULTS)
    )
    logging.info(
        f"Starting Grid Search. Total combinations to evaluate: {total_combinations}"
    )

    start_time = time.time()
    combinations_evaluated = 0

    # 1. Trigger Ratio と ATR_Filter のループ (外側でマスクを作ることで計算量を削減)
    for tr, atr_filter in itertools.product(TRIGGER_RATIOS, ATR_FILTER_RATIOS):
        # ★変更: ヒルベルト振幅条件 AND ローソク足の実体(高値-安値)ブレイク条件
        trigger_mask = (df_trigger_np >= tr) & (
            (ticks_high - ticks_low) > (df_atr_np * atr_filter)
        )
        triggered_count = np.sum(trigger_mask)

        if triggered_count == 0:
            logging.warning(
                f"No triggers found for ratio >= {tr} and atr_filter >= {atr_filter}"
            )
            combinations_evaluated += len(TD_MINS) * len(PT_MULTS) * len(SL_MULTS)
            continue

        # 今回のトリガーに該当する起点（Bets）を一時抽出
        raw_bets_t0 = ticks_ts[trigger_mask]
        raw_bets_close = df_close_np[trigger_mask]
        raw_bets_atr = df_atr_np[trigger_mask]

        # クールダウン（Throttle）による重複エントリーの除外
        cool_down_us = np.int64(COOL_DOWN_MINUTES * 60 * 1_000_000)
        valid_indices = []
        last_time = np.int64(-cool_down_us)  # 初回が確実に発火するように初期化

        for idx, t in enumerate(raw_bets_t0):
            if t - last_time >= cool_down_us:
                valid_indices.append(idx)
                last_time = t

        if not valid_indices:
            continue

        # フィルタリングされたクリーンなインデックスのみをNumpy配列化
        bets_t0 = np.ascontiguousarray(raw_bets_t0[valid_indices])
        bets_close = np.ascontiguousarray(raw_bets_close[valid_indices])
        bets_atr = np.ascontiguousarray(raw_bets_atr[valid_indices])
        triggered_count = len(bets_t0)  # フィルタ後の有効ベット数に更新

        # 2. TD (Time Duration) のループ
        for td in TD_MINS:
            td_us = np.int64(td * 60 * 1_000_000)
            bets_t1_max = bets_t0 + td_us

            # 3. PT / SL のループ
            for pt_mult, sl_mult in itertools.product(PT_MULTS, SL_MULTS):
                bets_pt_barrier = np.ascontiguousarray(bets_close + bets_atr * pt_mult)
                bets_sl_barrier = np.ascontiguousarray(bets_close - bets_atr * sl_mult)

                # Numba関数呼び出し（コアエンジン）
                out_pt, out_sl = _numba_find_hits_fast(
                    bets_t0,
                    bets_t1_max,
                    bets_pt_barrier,
                    bets_sl_barrier,
                    ticks_ts,
                    ticks_high,
                    ticks_low,
                )

                # ベクトル演算で勝敗とタイムアウトを判定
                # (同時到達の場合はSL優先として保守的に評価)
                is_win = (out_pt > 0) & ((out_sl == 0) | (out_pt < out_sl))
                is_loss = (out_sl > 0) & ((out_pt == 0) | (out_sl <= out_pt))
                is_timeout = ~(is_win | is_loss)

                wins = np.sum(is_win)
                losses = np.sum(is_loss)
                timeouts = np.sum(is_timeout)
                win_rate = wins / triggered_count if triggered_count > 0 else 0.0

                # Adjusted_PF の計算 (取引ごとの実ATRに基づく正確な利益/損失計算)
                total_profit = np.sum(bets_atr[is_win]) * pt_mult
                total_loss_base = np.sum(bets_atr[is_loss]) * sl_mult
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

                # ペイオフレシオ (固定マルチプライヤーの比率)
                avg_payoff = pt_mult / sl_mult

                results.append(
                    {
                        "Trigger_Ratio": tr,
                        "ATR_Filter": atr_filter,
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

                combinations_evaluated += 1
                if combinations_evaluated % 100 == 0:
                    logging.info(
                        f"Progress: {combinations_evaluated} / {total_combinations} combinations evaluated..."
                    )

    elapsed = time.time() - start_time
    logging.info(f"Grid Search completed in {elapsed:.2f} seconds.")

    if not results:
        logging.warning("No results generated.")
        return

    # 結果をPolars DataFrameにして保存および出力
    res_df = pl.DataFrame(results).sort("Adjusted_PF", descending=True)

    try:
        res_df.write_csv(RESULTS_CSV_PATH)
        logging.info(f"Results successfully saved to: {RESULTS_CSV_PATH}")
    except Exception as e:
        logging.error(f"Failed to save CSV: {e}")

    print("\n" + "=" * 80)
    print("🏆 Top 10 Configurations by Adjusted_PF 🏆")
    print("=" * 80)
    # ターミナル幅に合わせて列を調整して表示
    print(res_df.head(10).to_pandas().to_string(index=False))
    print("=" * 80)


if __name__ == "__main__":
    run_fast_grid_search()
