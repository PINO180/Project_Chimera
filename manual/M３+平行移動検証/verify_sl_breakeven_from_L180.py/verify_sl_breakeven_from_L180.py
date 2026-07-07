"""
verify_sl_breakeven_from_L180.py
================================================================
脳CSVを使わず全 M3 グリッド＋生tick で、L+180 起点でエントリーした後の
建値SL（起点基準）が各 SL幅でどれだけ刈られるかを測る。§29 の全グリッド版。

§29 は PT=1脳CSV（撃った時刻だけ）が母集団だった。本検証は §30/§31 と同じく
全 M3 グリッド（脳CSV不使用）を母集団にし、建値SLの素の挙動を測る。

【測るもの — 各 SL幅 ε ∈ {0.1,0.2,0.3,0.4,0.5} について】
  起点 = L+180（本番の実エントリー点＝順行速度を持った点）。
  建値SL 絶対位置 = price(L+180) − ε·ATR·dir（起点基準）。
  dir = sign(price(L+180) − price(L))（L→L+180 のモメンタム方向）。
  (a) 建値到達率: L+180 から窓内に建値SLへ touch する割合（= 本番建値SLで刈られる率）。
  (b) 建値を割った群の最大逆行深さ（建値直下を掠めるか／深く沈むか）。
  (c) 建値到達までの経過秒（エネルギー切れ＝反転確定の典型時間）。
  d帯別にも層別（高d=即PT相当が建値を割りやすいか／守られるか）。

【重要】PT は置かない。SL 単独の「逆行がどれだけ建値に届くか」を素で測る。
  §29 と同じく、起点 L+180 は順行速度を持つため建値を割りにくい（速度慣性）はず。
  本検証は脳CSVの偏りを外して、それが全グリッドでも成り立つかを確認する。
  留保（§30.6）：これは「相場の逆行頻度の上限」であり脳の選択力は別。

ATR はラベル同型自前計算（disc-aware TR → Wilder ewm α=1/13）。時刻系 全UTC。
使い方:
  python verify_sl_breakeven_from_L180.py \
      --feature-data /workspace/data/XAUUSD/stratum_1_base/master_processed \
      --tick-dir /workspace/data/XAUUSD/stratum_1_base/master_tick_partitioned \
      --out-dir  /workspace/data/diagnostics/sl_breakeven --window-min 30 --max-bets 0
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
SL_LEVELS = [0.1, 0.2, 0.3, 0.4, 0.5]  # 建値SL幅(起点L+180基準)


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
    sl_arr = np.array(SL_LEVELS, dtype=np.float64)
    n_lv = len(sl_arr)

    def _scan(entry_ts_us, entry_px, atr, dir_, ticks_ts, ticks_px, t1_us):
        # 起点 = L+180。各 SL幅 ε について、建値(entry − ε·ATR·dir)へ touch したか・
        # 何秒で・最大逆行深さ を測る。PT は置かない。打ち切らず全 ε を一度に判定。
        n = len(entry_ts_us)
        max_against = np.full(n, np.nan)            # 起点からの最大逆行(ATR, 正=逆行)
        sl_hit = np.zeros((n, n_lv), dtype=np.int64)        # 各εへ touch したか
        sl_time = np.full((n, n_lv), np.nan)                # 各εへ touch までの秒
        nt = len(ticks_ts)
        if nt == 0:
            return max_against, sl_hit, sl_time
        for i in prange(n):
            a = atr[i]
            d = dir_[i]
            ep = entry_px[i]
            ets = entry_ts_us[i]
            start = np.searchsorted(ticks_ts, ets, side="right")
            agn_max = 0.0
            hit = np.zeros(n_lv, dtype=np.int64)
            tim = np.full(n_lv, np.nan)
            for j in range(start, nt):
                tt = ticks_ts[j]
                if tt > t1_us[i]:
                    break
                against = -(ticks_px[j] - ep) * d / a   # 逆行(ATR, 正=逆行)
                if against > agn_max:
                    agn_max = against
                for k in range(n_lv):
                    if hit[k] == 0 and against >= sl_arr[k]:
                        hit[k] = 1
                        tim[k] = (tt - ets) / 1.0e6
            max_against[i] = agn_max
            for k in range(n_lv):
                sl_hit[i, k] = hit[k]
                sl_time[i, k] = tim[k]
        return max_against, sl_hit, sl_time

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
    # 起点 = L+180（本番エントリー点）。窓 = L+180 から TD(30分)。
    entry_ts = L_us + entry_offset
    t1_us = L_us + entry_offset + window_us

    # 走査: L+180 起点。各 SL幅 ε への建値到達・深さ・時間を測る（PT は置かない）。
    max_against, sl_hit, sl_time = _SCAN(
        entry_ts, price_E, atr, dir_, ts_us, px, t1_us)
    max_against = np.where(valid, max_against, np.nan)
    d_realized = np.where(valid, d_realized, np.nan)

    sep("母集団（脳CSV不使用・全 M3 グリッド・L+180 起点）")
    nval = int(np.isfinite(max_against).sum())
    print(f"  グリッド時刻: {len(feat):,}  / 有効(方向確定・走査成功): {nval:,}")
    if has_ratio:
        ntrig = int((ratio >= args.atr_ratio_threshold).sum())
        print(f"  参考: atr_ratio>={args.atr_ratio_threshold}（is_trigger 相当）: "
              f"{ntrig:,} ({ntrig/len(feat)*100:.1f}%)")

    # ── 本体1: SL幅別 建値到達率（全体）= L+180 から建値へ touch する割合 ──
    sep("各 SL幅(建値, 起点L+180基準)への到達率 — 全体（= 本番建値SLで刈られる率）")
    print(f"  {'SL幅':>8} {'建値到達率':>10}")
    for k, lv in enumerate(SL_LEVELS):
        rr = sl_hit[valid, k].mean() * 100
        print(f"  {lv:>8.1f} {rr:>9.1f}%")

    # ── 本体2: SL幅別 × d帯 の建値到達率（高dは速度慣性で守られるか）──
    sep("建値到達率 — d帯別（行=d帯, 列=SL幅。高d=即PT相当が建値を割りやすいか/守られるか）")
    hdr = "  " + f"{'d帯':>12} " + " ".join(f"SL{lv:>4.1f}" for lv in SL_LEVELS)
    print(hdr)
    for lo, hi in D_BINS:
        m = valid & (d_realized >= lo) & (d_realized < hi)
        n = int(m.sum())
        if n == 0:
            continue
        rates = [sl_hit[m, k].mean() * 100 for k in range(len(SL_LEVELS))]
        label = f"d[{lo:.1f},{hi:.1f})" if hi < 1e8 else f"d[{lo:.1f},inf)"
        print(f"  {label:>12} " + " ".join(f"{r:5.1f}%" for r in rates))
    rates_all = [sl_hit[valid, k].mean() * 100 for k in range(len(SL_LEVELS))]
    print(f"  {'全体':>12} " + " ".join(f"{r:5.1f}%" for r in rates_all))

    # ── 本体3: 建値を割った群の最大逆行深さ（建値直下を掠めるか/深く沈むか）──
    sep("建値を割った群の最大逆行深さ(ATR) — SL幅別（建値直下=速度揺らぎ / 深い=本物の反転）")
    print(f"  {'SL幅':>8} {'割った件数':>10} {'深さ中央':>9} {'p75':>7} {'p90':>7}")
    for k, lv in enumerate(SL_LEVELS):
        hit_mask = valid & (sl_hit[:, k] == 1)
        deep = max_against[hit_mask]
        deep = deep[~np.isnan(deep)]
        if len(deep) == 0:
            print(f"  {lv:>8.1f} {0:>10}")
            continue
        q = np.quantile(deep, [0.5, 0.75, 0.9])
        print(f"  {lv:>8.1f} {len(deep):>10,} {q[0]:>9.2f} {q[1]:>7.2f} {q[2]:>7.2f}")

    # ── 本体4: 建値到達までの経過時間（エネルギー切れ＝反転確定の典型時間）──
    sep("建値到達までの経過秒 — SL幅別（エネルギー切れ＝反転が確定するまでの典型時間）")
    print(f"  {'SL幅':>8} {'中央':>7} {'p25':>7} {'p75':>7} {'p90':>7}")
    for k, lv in enumerate(SL_LEVELS):
        t = sl_time[valid, k]
        t = t[~np.isnan(t)]
        if len(t) == 0:
            print(f"  {lv:>8.1f}  該当なし")
            continue
        q = np.quantile(t, [0.5, 0.25, 0.75, 0.9])
        print(f"  {lv:>8.1f} {q[0]:>7.0f} {q[1]:>7.0f} {q[2]:>7.0f} {q[3]:>7.0f}")

    # 保存
    out = pd.DataFrame({
        "timestamp": feat["timestamp"].to_numpy(),
        "atr": atr, "atr_ratio": ratio,
        "price_L": price_L, "price_E180": price_E,
        "dir": dir_, "d_realized": d_realized,
        "max_against_from_L180": max_against,
    })
    for k, lv in enumerate(SL_LEVELS):
        out[f"sl_hit_{lv}"] = sl_hit[:, k]
    out.to_parquet(args.out_dir / "sl_breakeven_from_L180.parquet", index=False)
    sep("完了")
    print(f"  出力: {args.out_dir}/sl_breakeven_from_L180.parquet")
    print("  読み方: 本体1/2 の建値到達率が小さく、本体3 の深さが建値直下に張り付くなら、")
    print("    建値割りは速度慣性の揺らぎ＝ε に余裕を持たせれば救える（深い反転ではない）。")
    print("    本体4 のSL到達時間がエネルギー切れの典型時間。全て脳の選択力を引く前の上限。")


if __name__ == "__main__":
    main()
