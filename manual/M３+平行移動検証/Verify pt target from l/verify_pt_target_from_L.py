"""
verify_pt_target_from_L.py
================================================================
脳CSVを使わず全 M3 グリッド＋生tick で、L 起点（波動の起点）から相場が
各 PT 水準（L基準）にどこまで届くかを測り、PT の適正値の天井を出す。

§30 の検証器（起点 L+180・最大順行）を、起点 L・各PT水準への到達率に組み替えたもの。
  - 起点 = L（price(L)）。PT は price(L) + pt·ATR·dir（L基準＝平行移動回収）。
  - 窓 = L から TD(30分)。dir = sign(price(L+180) − price(L))（L→L+180 のモメンタム方向）。
  - 即PT(高d)も普通の順行(低d)も全て母集団に含む（捨てない）。d帯別・ATR帯別に層別。
  - 主出力: 各PT水準への「L基準到達率」と、到達率×pt（取れる幅の期待値の天井）。

【重要な留保（§30.6）】これは「相場が L から pt まで届く頻度」＝脳が完璧に撃った場合の
  上限（天井）であり、脳が実際にその時刻を撃てるかは別問題。実脳の到達率はこれ以下。
  最終 pt は PT≈2脳を作り脳のエントリー選択込みで確定する。本検証は天井の測定。

ATR はラベル同型で自前計算（disc-aware TR → Wilder ewm α=1/13）。
時刻系は全 UTC（脳CSV=JST 不使用）。
使い方:
  python verify_pt_target_from_L.py \
      --feature-data /workspace/data/XAUUSD/stratum_1_base/master_processed \
      --tick-dir /workspace/data/XAUUSD/stratum_1_base/master_tick_partitioned \
      --out-dir  /workspace/data/diagnostics/pt_target --window-min 30 --max-bets 0
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
    def prange(x):
        return range(x)

ACTION_HORIZON_SEC = 180
US = 1_000_000
# d_realized 帯（§24 三領域を内包、即PT=d>=1 を含む連続層別）。dir=sign(move) ゆえ d>=0。
D_BINS = [(0.0, 0.5), (0.5, 1.0), (1.0, 1.5), (1.5, 2.0), (2.0, 3.0), (3.0, 1e9)]
# 参考: 相場が各水準まで届いた頻度（脳成績ではなく相場の到達分布）。主役は max_adv 分布。
PT_LEVELS = [1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0]  # L基準で測るPT水準


def sep(t=""):
    print("\n" + "=" * 80)
    if t:
        print(f"  {t}")
        print("=" * 80)


# ════════════════════════════════════════════════════════════════════
# 走査エンジン: L+180 から window 分、dir 方向の最大順行を 打ち切らず 測る。
#   PT でも SL でも止めない。最大順行・最大逆行・各水準到達(頻度参考)を記録。
# ════════════════════════════════════════════════════════════════════
def _make_scan():
    pt_arr = np.array(PT_LEVELS, dtype=np.float64)
    n_lv = len(pt_arr)

    def _scan(L_ts_us, price_L, atr, dir_, ticks_ts, ticks_px, t1_us):
        # 起点 = L。L基準で最大順行(ATR)と、各PT水準への到達(L基準)を測る。
        # PT到達 = (price − price_L)·dir / atr >= pt_level。打ち切らない(全水準を一度に)。
        n = len(L_ts_us)
        max_adv = np.full(n, np.nan)      # L からの最大順行(ATR)
        max_against = np.full(n, np.nan)  # L からの最大逆行(ATR, 正=逆行)
        reach = np.zeros((n, n_lv), dtype=np.int64)
        nt = len(ticks_ts)
        if nt == 0:
            return max_adv, max_against, reach
        for i in prange(n):
            a = atr[i]
            d = dir_[i]
            ep = price_L[i]
            start = np.searchsorted(ticks_ts, L_ts_us[i], side="right")
            adv_max = -1.0e18
            agn_max = 0.0
            rr = np.zeros(n_lv, dtype=np.int64)
            for j in range(start, nt):
                tt = ticks_ts[j]
                if tt > t1_us[i]:
                    break
                adv = (ticks_px[j] - ep) * d / a   # L基準 順行(ATR)
                if adv > adv_max:
                    adv_max = adv
                if -adv > agn_max:
                    agn_max = -adv
                for k in range(n_lv):
                    if rr[k] == 0 and adv >= pt_arr[k]:
                        rr[k] = 1
            max_adv[i] = adv_max if adv_max > -1.0e17 else np.nan
            max_against[i] = agn_max
            for k in range(n_lv):
                reach[i, k] = rr[k]
        return max_adv, max_against, reach

    if NUMBA:
        return numba.njit(_scan, cache=True, parallel=True, fastmath=False)
    return _scan


_SCAN = _make_scan()


def load_grid_atr(s1_processed_path, timeframe="M3"):
    """S1_PROCESSED/timeframe=M3 の OHLC+disc から、ラベルと同型の ATR を自前計算。
       create_proxy L654-688 と同一: disc-aware TR -> ewm_mean(alpha=1/13) -> atr_ratio。
       e1c_atr_13_M3(engine, 相対値≈1.0)は使わない。"""
    base = Path(s1_processed_path)
    tf_dir = base / f"timeframe={timeframe}"
    src = str(tf_dir / "*.parquet") if tf_dir.exists() else str(base / "**/*.parquet")
    print(f"[load] grid+ATR(自前計算, ラベル同型): {src}")
    ATR_PERIOD = 13                 # blueprint BARRIER_ATR_PERIOD
    BASELINE = 480 * 1              # M3 bars_per_day(480) * ATR_BASELINE_DAYS(1)
    atr_name, ratio_name = "atr_value", "atr_ratio"
    lf = pl.scan_parquet(src, hive_partitioning=True)
    cols = lf.collect_schema().names()
    if "timeframe" in cols:
        lf = lf.filter(pl.col("timeframe") == timeframe)
    for need in ("timestamp", "high", "low", "close"):
        if need not in cols:
            raise SystemExit(f"[ERROR] 列 '{need}' なし。実列: {cols[:30]}")
    has_disc = "disc" in cols
    sel = ["timestamp", "high", "low", "close"] + (["disc"] if has_disc else [])
    lf = lf.select(sel).with_columns(pl.col("timestamp").cast(pl.Datetime("us", "UTC"))).sort("timestamp")
    if has_disc:
        tr = (pl.when(pl.col("disc"))
              .then(pl.col("high") - pl.col("low"))
              .otherwise(pl.max_horizontal(
                  pl.col("high") - pl.col("low"),
                  (pl.col("high") - pl.col("close").shift(1)).abs(),
                  (pl.col("low") - pl.col("close").shift(1)).abs())))
    else:
        print("  [WARN] disc 列なし -> 通常TR(ギャップ補正なし)。ラベルと厳密一致しない可能性")
        tr = pl.max_horizontal(
            pl.col("high") - pl.col("low"),
            (pl.col("high") - pl.col("close").shift(1)).abs(),
            (pl.col("low") - pl.col("close").shift(1)).abs())
    df = (lf.with_columns(tr.ewm_mean(alpha=1.0 / ATR_PERIOD, adjust=False).alias(atr_name))
          .with_columns((pl.col(atr_name) / (pl.col(atr_name).rolling_mean(window_size=BASELINE, min_samples=1) + 1e-10)).alias(ratio_name))
          .select(["timestamp", atr_name, ratio_name])
          .drop_nulls(subset=[atr_name]).filter(pl.col(atr_name) > 0)
          .unique("timestamp", keep="first").sort("timestamp")
          .collect())
    print(f"  -> {len(df):,} グリッド時刻 (ATR/atr_ratio をラベル同型で計算)")
    return df, True


def load_ticks(tick_dir, t_min, t_max):
    print(f"[load] ticks: {tick_dir}")
    margin = pd.Timedelta(hours=8)
    lf = (pl.scan_parquet(str(Path(tick_dir) / "**/*.parquet"), hive_partitioning=True)
          .rename({"datetime": "timestamp"})
          .select("timestamp", "mid_price")
          .with_columns(pl.col("timestamp").cast(pl.Datetime("us", "UTC")))
          .filter(pl.col("timestamp").is_between(pl.lit(t_min - margin), pl.lit(t_max + margin)))
          .unique("timestamp", keep="first")
          .sort("timestamp"))
    df = lf.collect()
    print(f"  -> {len(df):,} ticks")
    return df["timestamp"].cast(pl.Int64).to_numpy(), df["mid_price"].cast(pl.Float64).to_numpy()


def _price_backward(ts_us, px, t_us):
    """t 以前で最新の tick mid（= その時点の価格）。ラベルの join_asof(backward) と同型。"""
    idx = np.clip(np.searchsorted(ts_us, t_us, side="right") - 1, 0, len(px) - 1)
    return px[idx], idx


def _dist(arr, label):
    a = arr[~np.isnan(arr)]
    if len(a) == 0:
        return f"  {label:<14}: 該当なし"
    q = np.quantile(a, [0.1, 0.25, 0.5, 0.75, 0.9, 0.95])
    return (f"  {label:<14}: 中央 {q[2]:6.3f} | p25 {q[1]:6.3f} p75 {q[3]:6.3f} "
            f"p90 {q[4]:6.3f} p95 {q[5]:6.3f} | 平均 {a.mean():6.3f}  (n={len(a):,})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--feature-data", type=str, required=True,
                    help="S1_PROCESSED のパス(内部で timeframe=M3 を読み ATR をラベル同型で自前計算)")
    ap.add_argument("--tick-dir", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, default=Path("/workspace/data/diagnostics/grid_excursion"))
    ap.add_argument("--atr-col", default="(自前計算のため未使用)")
    ap.add_argument("--atr-ratio-col", default="(自前計算のため未使用)")
    ap.add_argument("--atr-ratio-threshold", type=float, default=0.8,
                    help="is_trigger 相当のゲート。層別表示にのみ使用（母集団は絞らない）")
    ap.add_argument("--window-min", type=int, default=30, help="L+180 から先を見る窓(分)")
    ap.add_argument("--max-bets", type=int, default=0, help="0=全件。>0 ならランダムサンプル件数")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    if not NUMBA:
        print("[WARN] numba未検出 -> 純Python(遅い)")

    feat, has_ratio = load_grid_atr(args.feature_data, timeframe="M3")

    # 母集団を間引く場合（重い時のみ）。母集団は脳と無関係な全グリッド。
    if args.max_bets and args.max_bets < len(feat):
        rng = np.random.default_rng(args.seed)
        idx = np.sort(rng.choice(len(feat), size=args.max_bets, replace=False))
        feat = feat[idx]
        print(f"  [sample] {len(feat):,} 件にランダム間引き")

    t_min = feat["timestamp"].min()
    t_max = feat["timestamp"].max() + pd.Timedelta(minutes=args.window_min + 10)
    ts_us, px = load_ticks(args.tick_dir, t_min, t_max)

    entry_offset = np.int64(ACTION_HORIZON_SEC * US)
    window_us = np.int64(args.window_min * 60 * US)

    L_us = (feat["timestamp"].cast(pl.Int64).to_numpy()).astype(np.int64)  # μs
    atr = feat["atr_value"].cast(pl.Float64).to_numpy()
    ratio = (feat["atr_ratio"].cast(pl.Float64).to_numpy()
             if has_ratio else np.full(len(feat), np.nan))

    # price(L)=backward at L, price(L+180)=backward at L+180（= M3バーclose相当）
    price_L, _ = _price_backward(ts_us, px, L_us)
    price_E, _ = _price_backward(ts_us, px, L_us + entry_offset)
    move = price_E - price_L
    dir_ = np.sign(move)
    with np.errstate(divide="ignore", invalid="ignore"):
        d_realized = np.abs(move) / atr

    valid = (dir_ != 0) & np.isfinite(d_realized) & (atr > 0)
    # 起点 = L（L直後tickから走査）。窓 = L+180+TD まで（本番のスキャン開始L+180＋TO30分をカバー）。
    t1_us = L_us + entry_offset + window_us

    # 走査: L 起点。L基準で最大順行と、各PT水準(L基準)への到達を打ち切らず測る。
    max_adv, max_against, reach = _SCAN(
        L_us, price_L, atr, dir_, ts_us, px, t1_us)
    max_adv = np.where(valid, max_adv, np.nan)
    d_realized = np.where(valid, d_realized, np.nan)

    ATR_VALUE_BINS = [(0.0, 1.0), (1.0, 2.0), (2.0, 3.0), (3.0, 5.0), (5.0, 1e9)]

    sep("母集団（脳CSV不使用・全 M3 グリッド・L 起点）")
    nval = int(np.isfinite(max_adv).sum())
    print(f"  グリッド時刻: {len(feat):,}  / 有効(方向確定・走査成功): {nval:,}")
    if has_ratio:
        ntrig = int((ratio >= args.atr_ratio_threshold).sum())
        print(f"  参考: atr_ratio>={args.atr_ratio_threshold}（is_trigger 相当）: "
              f"{ntrig:,} ({ntrig/len(feat)*100:.1f}%)")

    # ── 本体1: d帯別の『L からの最大順行(ATR)』分布 ──
    sep(f"d_realized 帯別の『L からの最大順行(ATR)』— L基準・打ち切りなし・全グリッド（窓 L+180+{args.window_min}分）")
    print("  普通の順行(低d)も即PT(高d)も全て母集団に含む。L 基準でどこまで伸びるか。")
    for lo, hi in D_BINS:
        m = valid & (d_realized >= lo) & (d_realized < hi)
        label = f"d[{lo:.1f},{hi:.1f})" if hi < 1e8 else f"d[{lo:.1f},inf)"
        print(_dist(max_adv[m], label))
    print(_dist(max_adv[valid], "全体"))

    # ── 本体2: 各PT水準(L基準)への到達率 × d帯 ──
    sep("各PT水準(L基準)への到達率 — d帯別（行=d帯, 列=PT水準。L からその水準まで届いた割合）")
    hdr = "  " + f"{'d帯':>12} " + " ".join(f"PT{lv:>4.1f}" for lv in PT_LEVELS)
    print(hdr)
    for lo, hi in D_BINS:
        m = valid & (d_realized >= lo) & (d_realized < hi)
        n = int(m.sum())
        if n == 0:
            continue
        rates = [reach[m, k].mean() * 100 for k in range(len(PT_LEVELS))]
        label = f"d[{lo:.1f},{hi:.1f})" if hi < 1e8 else f"d[{lo:.1f},inf)"
        print(f"  {label:>12} " + " ".join(f"{r:5.1f}%" for r in rates))
    # 全体
    rates_all = [reach[valid, k].mean() * 100 for k in range(len(PT_LEVELS))]
    print(f"  {'全体':>12} " + " ".join(f"{r:5.1f}%" for r in rates_all))

    # ── 本体3: 各PT水準(L基準)への到達率 × ATR値帯（BT の ATR Value Band と対応）──
    sep("各PT水準(L基準)への到達率 — ATR値帯別（BT の ATR Value Band Analysis と対応）")
    print(hdr)
    for lo, hi in ATR_VALUE_BINS:
        m = valid & (atr >= lo) & (atr < hi)
        n = int(m.sum())
        if n == 0:
            continue
        rates = [reach[m, k].mean() * 100 for k in range(len(PT_LEVELS))]
        label = f"ATR[{lo:.0f},{hi:.0f})" if hi < 1e8 else f"ATR[{lo:.0f},inf)"
        print(f"  {label:>12} " + " ".join(f"{r:5.1f}%" for r in rates))

    # ── 本体4: PT適正値の天井 = 到達率 × pt（取れる幅の期待値の上限）──
    sep("PT適正値の天井 — 到達率 × pt（全体・脳が完璧に撃った場合の上限。実脳はこれ以下）")
    print(f"  {'PT水準':>8} {'到達率':>8} {'到達率×pt':>10}  （ATR単位の期待取り幅の天井）")
    best_pt, best_val = None, -1.0
    for k, lv in enumerate(PT_LEVELS):
        rr = reach[valid, k].mean()
        ev = rr * lv
        mark = ""
        if ev > best_val:
            best_val, best_pt = ev, lv
        print(f"  {lv:>8.1f} {rr*100:>7.1f}% {ev:>10.3f}")
    print(f"\n  → 到達率×pt が最大の PT水準: {best_pt}  (天井 {best_val:.3f} ATR)")
    print("    ※ これは相場の天井。建値SL負け・脳の選択力を引く前の上限値（§30.6）。")
    print("    最終 pt は PT≈2脳を作り、脳のエントリー選択込みで確定する。")

    # 保存
    out = pd.DataFrame({
        "timestamp": feat["timestamp"].to_numpy(),
        "atr": atr, "atr_ratio": ratio,
        "price_L": price_L, "price_E180": price_E,
        "dir": dir_, "d_realized": d_realized,
        "max_adv_from_L": max_adv, "max_against_from_L": max_against,
    })
    for k, lv in enumerate(PT_LEVELS):
        out[f"reach_pt{lv}"] = reach[:, k]
    out.to_parquet(args.out_dir / "pt_target_from_L.parquet", index=False)
    sep("完了")
    print(f"  出力: {args.out_dir}/pt_target_from_L.parquet")
    print("  読み方: 本体2/3 の到達率は『相場が L から各PTまで届く頻度』。本体4 の到達率×pt が天井。")
    print("    pt を上げると1回の取り分は増えるが到達率は下がる。その積が最大の pt が相場側の適正値。")
    print("    ただし全て脳の選択力を引く前の上限。最終は PT≈2脳で脳込み確定（§30.6/§30.7）。")


if __name__ == "__main__":
    main()
