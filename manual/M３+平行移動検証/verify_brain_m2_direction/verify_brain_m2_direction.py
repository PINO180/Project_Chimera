"""
verify_brain_m2_direction.py   —— 検証(2)+(4)合体
================================================================
脳CSVの各トレードに、tick で測った「実挙動」を突き合わせる。
  (2) M2確信度 × 地図: M2帯別に d_realized 中央・L基準PT到達率
  (4) 方向正答率:       脳が張った dir × その時刻の実順行方向 の一致率

【読み込むファイル】
  --log     : 脳CSV detailed_trade_log（stratum_7・完成済みを指定。書き換え中の新CSVは不可）
              使う列: timestamp(=L), direction, m2_proba, atr_value, close_price(=price(L))
  --tick-dir: 生tick master_tick_partitioned（stratum_1・安全）
  ※ master_processed 等は読まない。stratum_6/7 の新CSVにも触れない。

【物差しの原則】脳CSVは「脳が撃った事実」に、tick は「その時刻の実挙動」に使う。
  相場全体の性質は結論しない（それは全グリッド系）。ここは「脳が撃った各トレードが
  実際どう動いたか」の突き合わせに限定。

【測り方】
  price(L)     = close_price 列（BT の S6 close = price(L)。tick不要で厳密）
  price(L+180) = tick の L+180 backward（M3 close 相当）
  d_realized   = |price(L+180) − price(L)| / ATR
  dir_future   = sign(L+180 から先の正味の動き)  … L+180 以降に実際に伸びた方向
                 （旧: L→L+180 の方向と比べると自己相関で100%になる誤り。修正済）
  方向正答     = (脳の direction == dir_future) … 脳が「その後伸びる方向」に張れていたか
  L基準PT到達  = L+180 から TD30分、L基準 price(L)+pt·ATR·dir(脳の方向) に到達したか
                 （dir は脳が張った方向。脳の方向で PT を測る）

使い方:
  python verify_brain_m2_direction.py \
      --log "/workspace/data/XAUUSD/stratum_7_models/backtest_simulator_results/<完成済み>/detailed_trade_log_v5_M2.csv" \
      --tick-dir /workspace/data/XAUUSD/stratum_1_base/master_tick_partitioned \
      --log-tz jst
"""

import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import polars as pl

try:
    import numba
    from numba import prange
    NUMBA = True
except ImportError:
    NUMBA = False
    def prange(x): return range(x)

ACTION_HORIZON_SEC = 180
US = 1_000_000
PT_LEVELS = [1.0, 1.5, 2.0, 3.0]
M2_BINS = [(0.70, 0.75), (0.75, 0.80), (0.80, 0.85), (0.85, 0.90), (0.90, 0.95), (0.95, 1.01)]


def sep(t=""):
    print("\n" + "=" * 76)
    if t:
        print(f"  {t}")
        print("=" * 76)


def _make_scan():
    pt_arr = np.array(PT_LEVELS, dtype=np.float64)
    n_lv = len(pt_arr)

    def _scan(L_ts_us, price_L, price_E, atr, dir_, ticks_ts, ticks_px, entry_off, t1_us):
        # (2) L 起点で 脳の方向 dir_ に対し L基準 price_L+pt·ATR·dir へ到達したか。
        # (4) L+180 から先の「正味の動き方向」dir_future を返す（終端 − 起点L+180）。
        n = len(L_ts_us)
        reach = np.zeros((n, n_lv), dtype=np.int64)
        dir_future = np.zeros(n, dtype=np.float64)  # L+180から先に実際伸びた方向(+1/-1/0)
        nt = len(ticks_ts)
        if nt == 0:
            return reach, dir_future
        for i in prange(n):
            a = atr[i]; d = dir_[i]; ep = price_L[i]; e180 = price_E[i]
            # (2) PT到達（L基準・脳の方向）
            start = np.searchsorted(ticks_ts, L_ts_us[i], side="right")
            rr = np.zeros(n_lv, dtype=np.int64)
            for j in range(start, nt):
                tt = ticks_ts[j]
                if tt > t1_us[i]:
                    break
                adv = (ticks_px[j] - ep) * d / a
                for k in range(n_lv):
                    if rr[k] == 0 and adv >= pt_arr[k]:
                        rr[k] = 1
            for k in range(n_lv):
                reach[i, k] = rr[k]
            # (4) L+180 から先の正味方向: 窓内の最終tick価格 − price(L+180) の符号
            s2 = np.searchsorted(ticks_ts, L_ts_us[i] + entry_off, side="right")
            last_px = e180
            found = 0
            for j in range(s2, nt):
                if ticks_ts[j] > t1_us[i]:
                    break
                last_px = ticks_px[j]
                found = 1
            if found == 1:
                diff = last_px - e180
                dir_future[i] = 1.0 if diff > 0 else (-1.0 if diff < 0 else 0.0)
        return reach, dir_future

    if NUMBA:
        return numba.njit(_scan, cache=True, parallel=True, fastmath=False)
    return _scan


_SCAN = _make_scan()


def load_trade_log(path, tz):
    print(f"[load] trade_log: {path}")
    df = pd.read_csv(path, low_memory=False)
    if tz.lower() == "jst":
        df["timestamp"] = pd.to_datetime(df["timestamp"]).dt.tz_localize("Asia/Tokyo").dt.tz_convert("UTC")
    else:
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    for c in ["m2_proba", "direction", "atr_value", "close_price"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=["direction", "atr_value", "close_price"]).reset_index(drop=True)
    print(f"  -> {len(df):,} トレード")
    return df


def load_ticks(tick_dir, t_min, t_max):
    print(f"[load] ticks: {tick_dir}")
    lf = (pl.scan_parquet(str(Path(tick_dir) / "**/*.parquet"), hive_partitioning=True)
          .rename({"datetime": "timestamp"}).select("timestamp", "mid_price")
          .with_columns(pl.col("timestamp").cast(pl.Datetime("us", "UTC")))
          .filter(pl.col("timestamp").is_between(pl.lit(t_min - pd.Timedelta(hours=8)),
                                                 pl.lit(t_max + pd.Timedelta(hours=8))))
          .unique("timestamp", keep="first").sort("timestamp"))
    df = lf.collect()
    print(f"  -> {len(df):,} ticks")
    return df["timestamp"].cast(pl.Int64).to_numpy(), df["mid_price"].cast(pl.Float64).to_numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", type=Path, required=True)
    ap.add_argument("--tick-dir", type=Path, required=True)
    ap.add_argument("--window-min", type=int, default=30)
    ap.add_argument("--log-tz", default="jst", choices=["jst", "utc"])
    ap.add_argument("--out-dir", type=Path, default=Path("/workspace/data/diagnostics/brain_m2_dir"))
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    df = load_trade_log(args.log, args.log_tz)
    t0 = (df["timestamp"].astype("int64").to_numpy() // 1000).astype(np.int64)
    brain_dir = df["direction"].to_numpy().astype(np.float64)
    atr = df["atr_value"].to_numpy().astype(np.float64)
    m2 = df["m2_proba"].to_numpy().astype(np.float64)
    price_L = df["close_price"].to_numpy().astype(np.float64)

    ts_us, px = load_ticks(args.tick_dir, df["timestamp"].min(), df["timestamp"].max())
    entry_off = np.int64(ACTION_HORIZON_SEC * US)
    npx = len(px)
    iE = np.clip(np.searchsorted(ts_us, t0 + entry_off, side="right") - 1, 0, npx - 1)
    price_E = px[iE]
    move = price_E - price_L
    with np.errstate(divide="ignore", invalid="ignore"):
        d_real = np.abs(move) / atr

    # L基準PT到達（脳の方向で判定）＋ L+180から先の正味方向
    window_us = np.int64(args.window_min * 60 * US)
    t1_us = t0 + entry_off + window_us
    reach, dir_future = _SCAN(t0, price_L, price_E, atr, brain_dir, ts_us, px, entry_off, t1_us)

    # 方向正答: 脳の方向が「L+180から先に実際伸びた方向」と一致したか（自己相関を排除）
    dir_correct = (brain_dir == dir_future).astype(np.int64)
    valid = np.isfinite(d_real) & (dir_future != 0)

    # ── (4) 方向正答率 ──
    sep("(4) 方向正答率 — 脳が張った方向 vs L+180 から先に実際伸びた方向（自己相関排除版）")
    va = valid
    print(f"  全体: {dir_correct[va].mean()*100:.1f}%  (n={int(va.sum()):,})")
    print("  ※ 比較相手は『L+180 から先の正味方向』。L→L+180(脳が入力で見た動き)ではない。")
    print("     50%=まぐれ当たり相当。これを超えた分が、脳の先読み方向選択力。")
    print(f"  {'M2帯':>14} {'方向正答率':>10} {'件数':>9}")
    for lo, hi in M2_BINS:
        m = va & (m2 >= lo) & (m2 < hi)
        if m.sum() == 0:
            continue
        print(f"  {f'[{lo:.2f},{hi:.2f})':>14} {dir_correct[m].mean()*100:>9.1f}% {int(m.sum()):>9,}")

    # ── (2) M2 × d_realized中央・L基準PT到達率 ──
    sep("(2) M2確信度 × 地図 — M2帯別の d中央 と L基準PT到達率（脳の方向で）")
    hdr = f"  {'M2帯':>14} {'件数':>8} {'d中央':>7} " + " ".join(f"PT{lv:>3.1f}" for lv in PT_LEVELS)
    print(hdr)
    for lo, hi in M2_BINS:
        m = va & (m2 >= lo) & (m2 < hi)
        n = int(m.sum())
        if n == 0:
            continue
        dmed = np.median(d_real[m])
        rates = [reach[m, k].mean() * 100 for k in range(len(PT_LEVELS))]
        print(f"  {f'[{lo:.2f},{hi:.2f})':>14} {n:>8,} {dmed:>7.2f} "
              + " ".join(f"{r:5.1f}%" for r in rates))
    # 全体行
    n = int(va.sum())
    print(f"  {'全体':>14} {n:>8,} {np.median(d_real[va]):>7.2f} "
          + " ".join(f"{reach[va, k].mean()*100:5.1f}%" for k in range(len(PT_LEVELS))))

    print("\n  読み: M2↑で d中央↑・PT到達率↑ なら、M2は『伸びる強い局面』を確信度で読めている。")
    print("        M2閾値0.70が新執行(L基準PT)で妥当か＝0.70帯の到達率が実用に足るかで判断。")

    # 保存
    out = df.copy()
    out["d_realized"] = d_real
    out["dir_market"] = dir_market
    out["dir_correct"] = dir_correct
    for k, lv in enumerate(PT_LEVELS):
        out[f"reach_pt{lv}"] = reach[:, k]
    out.to_parquet(args.out_dir / "brain_m2_direction.parquet", index=False)
    sep("完了")
    print(f"  出力: {args.out_dir}/brain_m2_direction.parquet")


if __name__ == "__main__":
    main()
