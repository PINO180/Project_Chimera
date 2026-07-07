"""
verify_brain_race_uplift.py   —— 検証(1) 撃った所の勝率上乗せ
================================================================
脳CSVが撃った時刻だけで PT×SL レースをかけ、WIN率が「脳ゼロの床」(§32の
全グリッドレース)からどれだけ上乗せされるかを測る。＝脳の選択力の純価値。

【読み込むファイル】
  --log     : 脳CSV detailed_trade_log（stratum_7・完成済みを指定）
              使う列: timestamp(=L), direction, atr_value, close_price(=price(L))
  --tick-dir: 生tick master_tick_partitioned（stratum_1・安全）
  ※ 全グリッド(master_processed)は読まない。脳CSVの「撃った時刻」だけが母集団。
     脳ゼロの床は §32 の全グリッドレース結果を定数として持つ（--floor で上書き可）。

【測り方】§32 の pt_sl_race と同一のレース走査を、脳CSV の時刻集合に限定して実行。
  起点 L+180、PT=price(L)+pt·ATR·dir(L基準)、SL=price(L+180)−ε·ATR·dir(建値)。
  即PT(d>=pt)は弾く。脳の方向 direction を使う（脳が張った方向でレース）。
  出力: 脳が撃った時刻での WIN率格子 − 脳ゼロの床 = 上乗せ(pt)。

使い方:
  python verify_brain_race_uplift.py \
      --log "/workspace/.../detailed_trade_log_v5_M2.csv" \
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
SL_LEVELS = [0.1, 0.2, 0.3, 0.4, 0.5]

# §32 全グリッドレースの「脳ゼロの床」WIN率(%)格子。行=PT, 列=SL。--floor で上書き可。
FLOOR_WIN = {
    (1.0, 0.1): 20.9, (1.0, 0.2): 30.2, (1.0, 0.3): 37.4, (1.0, 0.4): 43.2, (1.0, 0.5): 47.9,
    (1.5, 0.1): 13.1, (1.5, 0.2): 19.9, (1.5, 0.3): 25.7, (1.5, 0.4): 30.5, (1.5, 0.5): 34.6,
    (2.0, 0.1):  8.8, (2.0, 0.2): 13.7, (2.0, 0.3): 18.1, (2.0, 0.4): 21.8, (2.0, 0.5): 25.1,
    (3.0, 0.1):  4.6, (3.0, 0.2):  7.2, (3.0, 0.3):  9.5, (3.0, 0.4): 11.5, (3.0, 0.5): 13.2,
}


def sep(t=""):
    print("\n" + "=" * 78)
    if t:
        print(f"  {t}")
        print("=" * 78)


def _make_scan():
    pt_arr = np.array(PT_LEVELS, dtype=np.float64)
    sl_arr = np.array(SL_LEVELS, dtype=np.float64)
    n_pt = len(pt_arr); n_sl = len(sl_arr)

    def _scan(entry_ts_us, price_L, price_E, atr, dir_, ticks_ts, ticks_px, t1_us):
        n = len(entry_ts_us)
        outcome = np.zeros((n, n_pt, n_sl), dtype=np.int8)  # 1=WIN,-1=LOSS,0=TO,2=即PT弾き
        nt = len(ticks_ts)
        if nt == 0:
            return outcome
        for i in prange(n):
            a = atr[i]; d = dir_[i]; pl_ = price_L[i]; ep = price_E[i]
            start = np.searchsorted(ticks_ts, entry_ts_us[i], side="right")
            pt_done = np.zeros(n_pt, dtype=np.int64)
            sl_done = np.zeros(n_sl, dtype=np.int64)
            res = np.zeros((n_pt, n_sl), dtype=np.int8)
            n_open = n_pt * n_sl
            adv0 = (ep - pl_) * d / a
            for kp in range(n_pt):
                if adv0 >= pt_arr[kp]:
                    for ks in range(n_sl):
                        if res[kp, ks] == 0:
                            res[kp, ks] = 2; n_open -= 1
            for j in range(start, nt):
                if n_open <= 0:
                    break
                tt = ticks_ts[j]
                if tt > t1_us[i]:
                    break
                px = ticks_px[j]
                adv_from_L = (px - pl_) * d / a
                against = -(px - ep) * d / a
                for kp in range(n_pt):
                    if pt_done[kp] == 0 and adv_from_L >= pt_arr[kp]:
                        pt_done[kp] = 1
                for ks in range(n_sl):
                    if sl_done[ks] == 0 and against >= sl_arr[ks]:
                        sl_done[ks] = 1
                for kp in range(n_pt):
                    for ks in range(n_sl):
                        if res[kp, ks] != 0:
                            continue
                        if pt_done[kp] == 1:
                            res[kp, ks] = 1; n_open -= 1
                        elif sl_done[ks] == 1:
                            res[kp, ks] = -1; n_open -= 1
            for kp in range(n_pt):
                for ks in range(n_sl):
                    outcome[i, kp, ks] = res[kp, ks]
        return outcome

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
    for c in ["direction", "atr_value", "close_price"]:
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
    ap.add_argument("--out-dir", type=Path, default=Path("/workspace/data/diagnostics/brain_race_uplift"))
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    df = load_trade_log(args.log, args.log_tz)
    t0 = (df["timestamp"].astype("int64").to_numpy() // 1000).astype(np.int64)
    brain_dir = df["direction"].to_numpy().astype(np.float64)
    atr = df["atr_value"].to_numpy().astype(np.float64)
    price_L = df["close_price"].to_numpy().astype(np.float64)

    ts_us, px = load_ticks(args.tick_dir, df["timestamp"].min(), df["timestamp"].max())
    entry_off = np.int64(ACTION_HORIZON_SEC * US)
    npx = len(px)
    iE = np.clip(np.searchsorted(ts_us, t0 + entry_off, side="right") - 1, 0, npx - 1)
    price_E = px[iE]
    valid = np.isfinite(price_E) & (brain_dir != 0) & (atr > 0)

    window_us = np.int64(args.window_min * 60 * US)
    entry_ts = t0 + entry_off
    t1_us = t0 + entry_off + window_us
    outcome = _SCAN(entry_ts, price_L, price_E, atr, brain_dir, ts_us, px, t1_us)

    n_pt = len(PT_LEVELS); n_sl = len(SL_LEVELS)

    def win_rate(kp, ks):
        o = outcome[valid, kp, ks]
        elig = (o != 2)
        ne = int(elig.sum())
        if ne == 0:
            return None, 0
        return (o == 1).sum() / ne * 100, ne

    sep("脳が撃った時刻での WIN率（%）格子 — 行=PT(L基準), 列=SL建値ε")
    print("  " + f"{'PT＼ε':>8} " + " ".join(f"{lv:>6.1f}" for lv in SL_LEVELS))
    brain_win = {}
    for kp, pt in enumerate(PT_LEVELS):
        row = []
        for ks, sl in enumerate(SL_LEVELS):
            wr, ne = win_rate(kp, ks)
            brain_win[(pt, sl)] = wr
            row.append(f"{wr:6.1f}" if wr is not None else "   N/A")
        print(f"  {pt:>8.1f} " + " ".join(row))

    sep("上乗せ（脳が撃った時刻 WIN率 − 脳ゼロの床）＝脳の選択力の純価値[pt]")
    print("  " + f"{'PT＼ε':>8} " + " ".join(f"{lv:>6.1f}" for lv in SL_LEVELS))
    for kp, pt in enumerate(PT_LEVELS):
        row = []
        for ks, sl in enumerate(SL_LEVELS):
            bw = brain_win.get((pt, sl))
            fl = FLOOR_WIN.get((pt, sl))
            if bw is None or fl is None:
                row.append("   N/A")
            else:
                row.append(f"{bw-fl:+6.1f}")
        print(f"  {pt:>8.1f} " + " ".join(row))
    print("\n  読み: 上乗せが大きいほど、脳の選択力(高d濃縮＋方向)がその PT×SL 設定で勝率を押し上げている。")
    print("  ※ 床は §32 全グリッドレース値(定数)。パイプライン更新後は §32 を回し直して更新のこと。")

    out = df.copy()
    for kp, pt in enumerate(PT_LEVELS):
        for ks, sl in enumerate(SL_LEVELS):
            out[f"oc_pt{pt}_sl{sl}"] = outcome[:, kp, ks]
    out.to_parquet(args.out_dir / "brain_race_uplift.parquet", index=False)
    sep("完了")
    print(f"  出力: {args.out_dir}/brain_race_uplift.parquet")


if __name__ == "__main__":
    main()
