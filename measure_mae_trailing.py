# /workspace/models/measure_mae_trailing.py
# [②較正 §41.8] トレーリング閾値の実測較正。
#   勝ちトレード(label=1)について、エントリー(L+180, tick mid)から PT 到達までの
#   (1) MAE: 最大逆行 (ATR単位) の分布
#   (2) 順行 X 到達後に建値(エントリー価格)まで戻った率 (X=0.25/0.5/0.75/1.0)
#   を tick 走査で測る。脳・ラベル・BT は一切触らない読み取り専用。
#
# 定義 (bit 一致の根拠):
#   - trade log の timestamp = ラベル L / close_price = price(L) (BT L1181,1193 で確認)
#   - エントリー時刻 = L + 180s、エントリー価格 = ts<=L+180 の最後の tick mid
#     (create_proxy_labels §11.34.16 の entry 定義と同一)
#   - PT(L起点) = price(L) + ATR*PT_MULT (long) / price(L) - ATR*PT_MULT (short)
#   - 走査は entry 直後 tick から PT 初到達 tick まで (label=1 なので必ず到達する)
#   - MAE = max(entry - min(mid)) [long] / max(max(mid) - entry) [short]
#   - 建値タッチ = 順行が X*ATR に達した「後」に mid が entry まで戻った事象
#
# 使い方:
#   python3 measure_mae_trailing.py <trade_log_csv> [PT_MULT]
#   例: python3 measure_mae_trailing.py \
#     "/workspace/data/XAUUSD/stratum_7_models/backtest_simulator_results/M2_20260701_001446_Th0.7_D0.3_R2 1.5と0.3/detailed_trade_log_v5_M2.csv" 1.5

import sys
import glob
import fnmatch
import numpy as np
import polars as pl

TICK_GLOB = "/workspace/data/XAUUSD/stratum_1_lake/raw_tick_partitioned/**/*.parquet"
ENTRY_OFFSET_US = 180 * 1_000_000
X_LEVELS = [0.25, 0.50, 0.75, 1.00]  # 順行到達の閾値 (ATR単位)
MAE_BINS = [0.0, 0.05, 0.10, 0.15, 0.20, 0.30, 0.50, 1.00, np.inf]


def main() -> None:
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)
    log_path = sys.argv[1]
    pt_mult = float(sys.argv[2]) if len(sys.argv) >= 3 else 1.5

    trades = (
        pl.read_csv(log_path, try_parse_dates=True)
        .filter(pl.col("label") == 1)
        .select(["timestamp", "direction", "close_price", "atr_value"])
        .sort("timestamp")
    )
    n_trades = trades.height
    print(f"勝ちトレード: {n_trades} 件 (PT_MULT={pt_mult})  log={log_path}")
    if n_trades == 0:
        return

    ts_l = trades["timestamp"].dt.replace_time_zone("UTC").cast(pl.Int64).to_numpy()
    dirs = trades["direction"].to_numpy().astype(np.int64)
    price_l = trades["close_price"].to_numpy().astype(np.float64)
    atrs = trades["atr_value"].to_numpy().astype(np.float64)

    months = (
        trades.with_columns(pl.col("timestamp").dt.strftime("%Y-%m").alias("ym"))["ym"]
        .unique()
        .sort()
        .to_list()
    )

    mae_atr_all = np.full(n_trades, np.nan)
    reached_x = np.zeros((len(X_LEVELS), n_trades), dtype=bool)
    touched_be = np.zeros((len(X_LEVELS), n_trades), dtype=bool)

    files = sorted(glob.glob(TICK_GLOB, recursive=True))
    if not files:
        print(f"[ERROR] tick が見つかりません: {TICK_GLOB}")
        sys.exit(1)

    trades_ym = trades.with_columns(
        pl.col("timestamp").dt.strftime("%Y-%m").alias("ym")
    )
    for ym in months:
        idx_np = np.where((trades_ym["ym"] == ym).to_numpy())[0]
        if idx_np.size == 0:
            continue
        y, m = ym.split("-")
        nxt_y = int(y) + (1 if m == "12" else 0)
        nxt_m = 1 if m == "12" else int(m) + 1
        pats = [
            f"*year={y}*month={int(m)}/*", f"*year={nxt_y}*month={nxt_m}/*",
            f"*year={y}*month={int(m)}\\*", f"*year={nxt_y}*month={nxt_m}\\*",
            f"*year={y}/month={int(m)}/*", f"*year={nxt_y}/month={nxt_m}/*",
        ]
        month_files = [f for f in files if any(fnmatch.fnmatch(f, p) for p in pats)]
        if not month_files:
            month_files = files  # partition 命名が想定外なら全読み (遅いが正確)
        tk = (
            pl.scan_parquet(month_files)
            .rename({"datetime": "timestamp"})
            .select(["timestamp", "mid_price"])
            .with_columns(pl.col("timestamp").cast(pl.Datetime("us", "UTC")))
            .sort("timestamp")
            .collect()
        )
        tts = tk["timestamp"].cast(pl.Int64).to_numpy()
        tmid = tk["mid_price"].to_numpy().astype(np.float64)

        for i in idx_np:
            t0 = ts_l[i]
            entry_idx = np.searchsorted(tts, t0 + ENTRY_OFFSET_US, side="right") - 1
            if entry_idx < 0:
                continue
            entry = tmid[entry_idx]
            atr = atrs[i]
            if atr <= 0:
                continue
            d = dirs[i]
            pt = price_l[i] + atr * pt_mult if d == 1 else price_l[i] - atr * pt_mult
            j0 = entry_idx + 1
            seg = tmid[j0:]
            if seg.size == 0:
                continue
            if d == 1:
                has_hit = bool(np.any(seg >= pt))
                hit_rel = int(np.argmax(seg >= pt)) if has_hit else -1
            else:
                has_hit = bool(np.any(seg <= pt))
                hit_rel = int(np.argmax(seg <= pt)) if has_hit else -1
            if hit_rel < 0:
                continue  # 月跨ぎで tick 窓に PT が無い (稀) → skip
            path = seg[: hit_rel + 1]
            if d == 1:
                adverse = (entry - np.minimum.accumulate(path)) / atr
                favor = (np.maximum.accumulate(path) - entry) / atr
                be_touch = path <= entry
            else:
                adverse = (np.maximum.accumulate(path) - entry) / atr
                favor = (entry - np.minimum.accumulate(path)) / atr
                be_touch = path >= entry
            mae_atr_all[i] = float(adverse.max())
            for k, x in enumerate(X_LEVELS):
                if np.any(favor >= x):
                    rk = int(np.argmax(favor >= x))
                    reached_x[k, i] = True
                    if np.any(be_touch[rk + 1:]):
                        touched_be[k, i] = True
        del tk, tts, tmid

    ok = ~np.isnan(mae_atr_all)
    if ok.sum() == 0:
        print("[ERROR] 有効走査 0 件。tick パス/期間を確認してください。")
        return
    mae = mae_atr_all[ok]
    print(f"\n走査完了: {int(ok.sum())}/{n_trades} 件 (tick 窓不足で skip: {int((~ok).sum())})")
    print("\n===== (1) 勝ちトレードの MAE (最大逆行, ATR単位) 分布 =====")
    print(f"  中央値={np.median(mae):.3f}  平均={mae.mean():.3f}  "
          f"q90={np.quantile(mae, 0.90):.3f}  q95={np.quantile(mae, 0.95):.3f}  最大={mae.max():.3f}")
    hist, _ = np.histogram(mae, bins=MAE_BINS)
    labels = ["0-0.05", "0.05-0.10", "0.10-0.15", "0.15-0.20", "0.20-0.30",
              "0.30-0.50", "0.50-1.00", ">1.00"]
    for lab, h in zip(labels, hist):
        print(f"  MAE {lab:<10} {h:>7} 件 ({100 * h / len(mae):5.1f}%)")
    print(f"  → MAE >= 0.30 (初期SL0.3で刈られたはず) の勝ち: "
          f"{int((mae >= 0.30).sum())} 件 ({100 * float((mae >= 0.30).mean()):.1f}%)")

    print("\n===== (2) 順行 X 到達後の建値タッチ率 (トレーリング閾値較正) =====")
    print(f"  {'X(ATR)':<8}{'X到達':>8}{'到達率%':>9}{'建値戻り':>9}{'戻り率%':>9}")
    for k, x in enumerate(X_LEVELS):
        r = reached_x[k] & ok
        t = touched_be[k] & ok
        nr = int(r.sum())
        nt = int(t.sum())
        rate_r = 100 * nr / int(ok.sum())
        rate_t = (100 * nt / nr) if nr else 0.0
        print(f"  {x:<8.2f}{nr:>8}{rate_r:>8.1f}%{nt:>9}{rate_t:>8.1f}%")
    print("\n  読み方: X で建値化した場合、『戻り率%』がそのまま建値タッチで")
    print("  勝ちを落とす率。低い X で戻り率が十分小さければ早期建値化が安全。")


if __name__ == "__main__":
    main()
