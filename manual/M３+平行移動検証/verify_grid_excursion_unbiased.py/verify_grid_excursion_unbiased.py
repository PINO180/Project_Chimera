"""
verify_grid_excursion_unbiased.py
================================================================
脳CSVを使わず、全 M3 グリッド＋生tick だけで「L+180 から先に相場が
実際どこまで動くか」を、PT で打ち切らずに素で測る検証器。

【なぜ脳CSVを捨てるのか — §28/§29 の母集団バイアスの清算】
  §28/§29 は PT=1脳の trade log（脳が撃ったトレード）を母集団にしていた。
  その母集団は「L+1ATR を目標に撃たれたトレード」だけ＝構造的に
  「L+1ATR あたりで決着がつく相場」に偏る。そんな母集団で
  「L+180 から先に伸びるか」を測れば、伸びない方向に答えが出るのは当たり前。
  伸びるトレードが相場に存在しても、CSV には 1ATR 目標のものしか載らず、
  最初から視界に入らない。母集団の選び方が答えを先に決めていた。
  → §28 の excess も §29 の伸びも、この偏った母集団の人工物であり、
     「L+180 以降の相場の真実」ではない。これは検証器の設計ミスだった。

【この検証器の原則】
  - 入力は脳CSVではなく、全 M3 グリッド（is_trigger 前の全時刻）＋生tick のみ。
  - 各時刻 L で L+180 から tick を走らせ、PT でも SL でも打ち切らない。
    L+180 から先、価格が実際どこまで動いたか（最大順行）を素の相場として観測する。
  - 脳が撃ったか・勝ったか・PT に届いたかは一切問わない。勝率・PT率は出さない。
  - 母集団は「相場の全時刻」、答えは「相場が L+180 から先にどう動くか」。

【測るもの】
  各グリッド時刻 L について:
    price(L)     = L 以前で最新の tick mid（backward。ラベル §9.5.1 と同型）
    price(L+180) = L+180 以前で最新の tick mid（= M3バー close 相当 = 本番 current_price）
    dir          = sign(price(L+180) - price(L))   （L→L+180 のモメンタム方向）
    d_realized   = |price(L+180) - price(L)| / ATR  （L→L+180 の実現順行・ATR単位、>=0）
    max_adv      = L+180 から先、dir 方向の最大順行 / ATR（打ち切らない）
  これを d_realized 帯別に集計する。高d帯（即PT相当）が L+180 から先に
  どれだけ追加で伸びるか／伸びないかが、脳CSVの偏りなしで出る。

【入力】
  --feature-data : atr を含む M3 特徴量データ（ラベリングが読むのと同じ
                   S6/特徴量パーティション）。timestamp と ATR 列・atr_ratio 列を持つもの。
  --tick-dir     : master_tick（mid_price）。
  時刻系は全て UTC（脳CSV=JST を使わないので tz 変換は不要）。

使い方:
  python verify_grid_excursion_unbiased.py \
      --feature-data /workspace/data/XAUUSD/.../<M3特徴量 parquet dir or file> \
      --tick-dir /workspace/data/XAUUSD/stratum_1_base/master_tick_partitioned \
      --out-dir  /workspace/data/diagnostics/grid_excursion \
      --atr-col e1c_atr_13_M3 --atr-ratio-col atr_ratio_M3 \
      --window-min 30 --max-bets 0
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
REACH_LEVELS = [1.0, 1.5, 2.0, 3.0]


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
    reach_arr = np.array(REACH_LEVELS, dtype=np.float64)
    n_lv = len(reach_arr)

    def _scan(entry_ts_us, entry_px, atr, dir_, ticks_ts, ticks_px, t1_us):
        n = len(entry_ts_us)
        max_adv = np.full(n, np.nan)      # L+180 からの最大順行(ATR)
        max_against = np.full(n, np.nan)  # L+180 からの最大逆行(ATR, 正=逆行)
        reach = np.zeros((n, n_lv), dtype=np.int64)
        nt = len(ticks_ts)
        if nt == 0:
            return max_adv, max_against, reach
        for i in prange(n):
            a = atr[i]
            d = dir_[i]
            ep = entry_px[i]
            start = np.searchsorted(ticks_ts, entry_ts_us[i], side="right")
            adv_max = -1.0e18
            agn_max = 0.0
            rr = np.zeros(n_lv, dtype=np.int64)
            for j in range(start, nt):
                tt = ticks_ts[j]
                if tt > t1_us[i]:
                    break
                adv = (ticks_px[j] - ep) * d / a   # 順行(ATR)
                if adv > adv_max:
                    adv_max = adv
                if -adv > agn_max:
                    agn_max = -adv
                for k in range(n_lv):
                    if rr[k] == 0 and adv >= reach_arr[k]:
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
    entry_ts = L_us + entry_offset
    t1_us = L_us + entry_offset + window_us

    # 走査: L+180 から先、dir 方向の最大順行を打ち切らず測る
    max_adv, max_against, reach = _SCAN(
        entry_ts, price_E, atr, dir_, ts_us, px, t1_us)
    max_adv = np.where(valid, max_adv, np.nan)
    d_realized = np.where(valid, d_realized, np.nan)

    sep("母集団（脳CSV不使用・全 M3 グリッド）")
    nval = int(np.isfinite(max_adv).sum())
    print(f"  グリッド時刻: {len(feat):,}  / 有効(方向確定・走査成功): {nval:,}")
    if has_ratio:
        ntrig = int((ratio >= args.atr_ratio_threshold).sum())
        print(f"  参考: atr_ratio>={args.atr_ratio_threshold}（is_trigger 相当）: "
              f"{ntrig:,} ({ntrig/len(feat)*100:.1f}%)")

    # ── 本体: d_realized 帯別の「L+180 から先の最大順行」分布（打ち切りなし）──
    sep(f"d_realized 帯別の『L+180 から先の最大順行(ATR)』— PT打ち切りなし・全グリッド（窓 {args.window_min}分）")
    print("  即PT(d>=1)の時刻が、L+180 から先に追加でどれだけ伸びるか/伸びないか。脳の偏りなし。")
    for lo, hi in D_BINS:
        m = valid & (d_realized >= lo) & (d_realized < hi)
        label = f"d[{lo:.1f},{hi:.1f})" if hi < 1e8 else f"d[{lo:.1f},inf)"
        print(_dist(max_adv[m], label))

    # 参考: atr_ratio>=しきい のサブ母集団（システムがエントリーを検討する局面）でも同じ層別
    if has_ratio:
        sep(f"参考: atr_ratio>={args.atr_ratio_threshold} に限定した同じ層別（システム検討局面）")
        trig = valid & (ratio >= args.atr_ratio_threshold)
        for lo, hi in D_BINS:
            m = trig & (d_realized >= lo) & (d_realized < hi)
            label = f"d[{lo:.1f},{hi:.1f})" if hi < 1e8 else f"d[{lo:.1f},inf)"
            print(_dist(max_adv[m], label))

    # 参考: 相場が各水準まで届いた頻度（脳成績ではなく相場の到達分布）
    sep("参考: L+180 から各水準まで届いた頻度（相場の到達分布・脳非依存）")
    print(f"  {'d帯':>12} " + " ".join(f">={lv}ATR" for lv in REACH_LEVELS))
    for lo, hi in D_BINS:
        m = valid & (d_realized >= lo) & (d_realized < hi)
        n = int(m.sum())
        if n == 0:
            continue
        rates = [reach[m, k].mean() * 100 for k in range(len(REACH_LEVELS))]
        label = f"d[{lo:.1f},{hi:.1f})" if hi < 1e8 else f"d[{lo:.1f},inf)"
        print(f"  {label:>12} " + " ".join(f"{r:6.1f}%" for r in rates))

    # 保存
    out = pd.DataFrame({
        "timestamp": feat["timestamp"].to_numpy(),
        "atr": atr, "atr_ratio": ratio,
        "price_L": price_L, "price_E180": price_E,
        "dir": dir_, "d_realized": d_realized,
        "max_adv_from_L180": max_adv, "max_against_from_L180": max_against,
    })
    out.to_parquet(args.out_dir / "grid_excursion.parquet", index=False)
    sep("完了")
    print(f"  出力: {args.out_dir}/grid_excursion.parquet")
    print("  読み方: d帯別の max_adv 中央が、d 上昇に対してどう動くか。")
    print("    高d帯でも max_adv が大きいなら、L+180 から先も相場は伸びる（脳CSVの偏りが結論を歪めていた）。")
    print("    高d帯で max_adv が小さく頭打ちなら、相場として本当に L+180 で出し切っている。")


if __name__ == "__main__":
    main()
