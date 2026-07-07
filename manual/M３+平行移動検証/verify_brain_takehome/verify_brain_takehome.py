"""
verify_brain_takehome.py   —— 検証(3) 作り直し版：手取りで測る
================================================================
【問いの立て直し — 高d は中間変数でどうでもいい】
  旧(3)は「脳は高d の中身を選別できているか」を問うたが、これは的外れだった。
  平行移動の手取り = pt − d_realized。d が大きいほど L+180 時点で既に進んでいて
  残り(取り分)が薄い。つまり「高d を選ぶこと」自体に価値はなく、むしろ薄利。
  脳は「高d を狙って」いるのでもない——学習目的は「L基準PT に届くトレード」であり、
  高d 濃縮(§32)はその副産物にすぎない。中間変数の高d はどうでもいい。

  本当に価値があるのは「手取り(pt − d)が大きい局面」＝
  「L+180 時点ではまだ動いていない(低d=残り幅たっぷり)のに、その後 2.0 まで伸びる」局面。
  → 測るべきは「脳がこの"低d なのに伸びる"を掴めているか」。

【この検証が測るもの】
  (A) 手取り分布: 脳が撃った vs 全グリッド見送り で、手取り(pt − d | PT到達時) の比較。
      脳の撃ちが見送りより手取りが厚いか（＝薄利の高d を避け、伸びしろのある局面を選ぶか）。
  (B) 低d 選別力: 低d 帯(d < --d-low, 既定0.5)に限定し、
      脳が撃った低d と 見送った低d で PT2.0 到達率を比較。
      撃った低d の到達率が見送りより高ければ「まだ動いてないが伸びる(手取り大)」を掴めている。

【読み込むファイル】
  --feature-data : 全グリッド master_processed（stratum_1・安全、timeframe=M3, ATRラベル同型計算）
  --log          : 脳CSV detailed_trade_log（stratum_7・完成済み。撃った時刻の除外に使用）
  --tick-dir     : 生tick master_tick_partitioned（stratum_1・安全）

使い方:
  python verify_brain_takehome.py \
      --feature-data /workspace/data/XAUUSD/stratum_1_base/master_processed \
      --log "/workspace/.../detailed_trade_log_v5_M2.csv" \
      --tick-dir /workspace/data/XAUUSD/stratum_1_base/master_tick_partitioned \
      --log-tz jst --pt 2.0 --sl 0.3 --d-low 0.5
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
D_BINS = [(0.0, 0.5), (0.5, 1.0), (1.0, 1.5), (1.5, 2.0), (2.0, 3.0), (3.0, 1e9)]


def sep(t=""):
    print("\n" + "=" * 80)
    if t:
        print(f"  {t}")
        print("=" * 80)


def _make_scan():
    def _scan(L_ts_us, price_L, price_E, atr, dir_, pt, sl, ticks_ts, ticks_px, t1_us):
        # 1=WIN(PT先), -1=LOSS(SL先), 0=TO, 2=即PT弾き(d>=pt)
        n = len(L_ts_us)
        outcome = np.zeros(n, dtype=np.int8)
        nt = len(ticks_ts)
        if nt == 0:
            return outcome
        eo = np.int64(ACTION_HORIZON_SEC * US)
        for i in prange(n):
            a = atr[i]; d = dir_[i]; pl_ = price_L[i]; ep = price_E[i]
            if (ep - pl_) * d / a >= pt:
                outcome[i] = 2
                continue
            start = np.searchsorted(ticks_ts, L_ts_us[i] + eo, side="right")
            res = 0
            for j in range(start, nt):
                if ticks_ts[j] > t1_us[i]:
                    break
                px = ticks_px[j]
                if (px - pl_) * d / a >= pt:
                    res = 1; break
                if -(px - ep) * d / a >= sl:
                    res = -1; break
            outcome[i] = res
        return outcome

    if NUMBA:
        return numba.njit(_scan, cache=True, parallel=True, fastmath=False)
    return _scan


_SCAN = _make_scan()


def load_grid_atr(s1_processed_path, timeframe="M3"):
    base = Path(s1_processed_path)
    tf_dir = base / f"timeframe={timeframe}"
    src = str(tf_dir / "*.parquet") if tf_dir.exists() else str(base / "**/*.parquet")
    print(f"[load] grid+ATR(ラベル同型): {src}")
    ATR_PERIOD = 13
    lf = pl.scan_parquet(src, hive_partitioning=True)
    cols = lf.collect_schema().names()
    if "timeframe" in cols:
        lf = lf.filter(pl.col("timeframe") == timeframe)
    has_disc = "disc" in cols
    sel = ["timestamp", "high", "low", "close"] + (["disc"] if has_disc else [])
    lf = lf.select(sel).with_columns(pl.col("timestamp").cast(pl.Datetime("us", "UTC"))).sort("timestamp")
    if has_disc:
        tr = (pl.when(pl.col("disc")).then(pl.col("high") - pl.col("low"))
              .otherwise(pl.max_horizontal(
                  pl.col("high") - pl.col("low"),
                  (pl.col("high") - pl.col("close").shift(1)).abs(),
                  (pl.col("low") - pl.col("close").shift(1)).abs())))
    else:
        tr = pl.max_horizontal(
            pl.col("high") - pl.col("low"),
            (pl.col("high") - pl.col("close").shift(1)).abs(),
            (pl.col("low") - pl.col("close").shift(1)).abs())
    df = (lf.with_columns(tr.ewm_mean(alpha=1.0 / ATR_PERIOD, adjust=False).alias("atr_value"))
          .select(["timestamp", "atr_value"]).drop_nulls().filter(pl.col("atr_value") > 0)
          .unique("timestamp", keep="first").sort("timestamp").collect())
    print(f"  -> {len(df):,} グリッド時刻")
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


def load_brain_times(path, tz):
    print(f"[load] 脳CSV(撃った時刻): {path}")
    df = pd.read_csv(path, low_memory=False, usecols=lambda c: c == "timestamp")
    if tz.lower() == "jst":
        t = pd.to_datetime(df["timestamp"]).dt.tz_localize("Asia/Tokyo").dt.tz_convert("UTC")
    else:
        t = pd.to_datetime(df["timestamp"], utc=True)
    us = (t.astype("int64").to_numpy() // 1000).astype(np.int64)
    print(f"  -> {len(us):,} 撃った時刻")
    return set(us.tolist())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--feature-data", type=str, required=True)
    ap.add_argument("--log", type=Path, required=True)
    ap.add_argument("--tick-dir", type=Path, required=True)
    ap.add_argument("--pt", type=float, default=2.0)
    ap.add_argument("--sl", type=float, default=0.3)
    ap.add_argument("--d-low", type=float, default=0.5, help="低d帯の上限(既定0.5)")
    ap.add_argument("--window-min", type=int, default=30)
    ap.add_argument("--log-tz", default="jst", choices=["jst", "utc"])
    ap.add_argument("--out-dir", type=Path, default=Path("/workspace/data/diagnostics/brain_takehome"))
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    grid = load_grid_atr(args.feature_data)
    L_us = grid["timestamp"].cast(pl.Int64).to_numpy().astype(np.int64)
    atr = grid["atr_value"].cast(pl.Float64).to_numpy()
    t_max = grid["timestamp"].max() + pd.Timedelta(minutes=args.window_min + 10)
    ts_us, px = load_ticks(args.tick_dir, grid["timestamp"].min(), t_max)
    entry_off = np.int64(ACTION_HORIZON_SEC * US)
    npx = len(px)

    iL = np.clip(np.searchsorted(ts_us, L_us, side="right") - 1, 0, npx - 1)
    iE = np.clip(np.searchsorted(ts_us, L_us + entry_off, side="right") - 1, 0, npx - 1)
    price_L = px[iL]; price_E = px[iE]
    move = price_E - price_L
    dir_ = np.sign(move)
    with np.errstate(divide="ignore", invalid="ignore"):
        d_real = np.abs(move) / atr

    brain_set = load_brain_times(args.log, args.log_tz)
    shot = np.array([t in brain_set for t in L_us], dtype=bool)

    window_us = np.int64(args.window_min * 60 * US)
    t1_us = L_us + entry_off + window_us
    oc = _SCAN(L_us, price_L, price_E, atr, dir_, args.pt, args.sl, ts_us, px, t1_us)

    valid = np.isfinite(d_real) & (dir_ != 0)
    win = (oc == 1)
    # 手取り(pt − d)。WIN(PT到達)時のみ意味を持つ。
    takehome = np.where(win, args.pt - d_real, np.nan)

    sep("母集団（全グリッド）")
    print(f"  全グリッド: {len(grid):,}  / 有効: {int(valid.sum()):,}")
    print(f"  脳が撃った: {int((valid & shot).sum()):,}  / 見送り: {int((valid & ~shot).sum()):,}")
    print(f"  手取り = pt({args.pt}) − d_realized。高d は中間変数で無関係、手取りがすべて。")

    # ── (A) 手取り分布: 撃った vs 見送り（WIN=PT到達 のみ）──
    sep(f"(A) 手取り(pt−d) 分布 — WIN(PT{args.pt}到達)時 の実取り幅（ATR）")
    def th_stats(mask, name):
        th = takehome[mask & valid & win]
        th = th[~np.isnan(th)]
        if len(th) == 0:
            print(f"  {name}: WINなし"); return
        q = np.quantile(th, [0.25, 0.5, 0.75])
        # PT到達率(=WIN率, 即PT弾き除外)
        elig = mask & valid & (oc != 2)
        wr = win[mask & valid & (oc != 2)].mean() * 100 if elig.sum() else 0
        print(f"  {name:<10}: 手取り中央 {q[1]:.2f} | p25 {q[0]:.2f} p75 {q[2]:.2f} "
              f"| PT到達率 {wr:.1f}% | WIN件数 {len(th):,}")
    th_stats(shot, "撃った")
    th_stats(~shot, "見送り")
    print("  → 撃った側の手取り中央が見送りより厚ければ、脳は薄利を避け伸びしろを選べている。")

    # ── (B) 低d 選別力: 低d 帯で 撃った vs 見送り の PT到達率 ──
    sep(f"(B) 低d(d<{args.d_low})での選別力 — 「まだ動いてないが伸びる(手取り大)」を掴めているか")
    lowd = valid & (d_real < args.d_low)
    def reach_rate(mask, name):
        elig = mask & (oc != 2)
        ne = int(elig.sum())
        if ne == 0:
            print(f"  {name}: 対象なし"); return
        wr = win[elig].mean() * 100
        th = takehome[elig & win]; th = th[~np.isnan(th)]
        thm = np.median(th) if len(th) else float("nan")
        print(f"  {name:<16}: 件数 {ne:,} / PT{args.pt}到達 {wr:.1f}% / 到達時手取り中央 {thm:.2f}")
    reach_rate(lowd & shot, f"低d・撃った")
    reach_rate(lowd & ~shot, f"低d・見送り")
    print("  → 低d・撃った の到達率が 低d・見送り を明確に上回れば、")
    print("     脳は『低dなのに伸びる=手取り最大』の局面を選べている＝これこそ真の選択力。")

    # ── 参考: d帯別の PT到達率（撃った vs 見送り）──
    sep(f"参考: d帯別 PT{args.pt}到達率（撃った / 見送り）と 到達時手取り中央")
    print(f"  {'d帯':>12} {'撃った到達%':>10} {'見送り到達%':>10} {'撃った手取り中':>12}")
    for lo, hi in D_BINS:
        band = valid & (d_real >= lo) & (d_real < hi)
        es = band & shot & (oc != 2); em = band & ~shot & (oc != 2)
        ws = win[es].mean()*100 if es.sum() else float("nan")
        wm = win[em].mean()*100 if em.sum() else float("nan")
        th = takehome[band & shot & win]; th = th[~np.isnan(th)]
        thm = np.median(th) if len(th) else float("nan")
        label = f"d[{lo:.1f},{hi:.1f})" if hi < 1e8 else f"d[{lo:.1f},inf)"
        print(f"  {label:>12} {ws:>9.1f}% {wm:>9.1f}% {thm:>12.2f}")

    out = pd.DataFrame({
        "timestamp": grid["timestamp"].to_numpy(),
        "atr": atr, "d_realized": d_real, "dir": dir_,
        "shot": shot, "outcome": oc, "takehome": takehome,
    })
    out.to_parquet(args.out_dir / "brain_takehome.parquet", index=False)
    sep("完了")
    print(f"  出力: {args.out_dir}/brain_takehome.parquet")


if __name__ == "__main__":
    main()
