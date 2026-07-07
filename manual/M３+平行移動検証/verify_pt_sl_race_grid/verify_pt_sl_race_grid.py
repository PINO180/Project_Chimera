"""
verify_pt_sl_race_grid.py
================================================================
脳CSVを使わず全 M3 グリッド＋生tick で、PT(L起点) と SL(L+180起点の建値) を
同時に置いてレースさせ、本当の勝敗（どちらに先に届くか）を測る。

§31(PT単独) と sl_breakeven(SL単独) は片側だけの上限値だった。本検証は両者を
同時に置き、PT に先に届けば勝ち / SL に先に届けば負け / どちらも届かず時間切れ、
を全グリッドで判定する。これで勝率・決着内訳・到達率×pt の期待値が出る。

【レースの定義 — PT × SL の格子】
  起点(走査開始・建値SLの原点) = L+180（本番エントリー点）。
  PT 絶対位置 = price(L) + pt·ATR·dir     （L基準。pt ∈ {1.0,1.5,2.0,3.0}）
  SL 絶対位置 = price(L+180) − ε·ATR·dir   （建値。ε ∈ {0.1,0.2,0.3,0.4,0.5}）
  dir = sign(price(L+180) − price(L))。
  各 (pt, ε) について、L+180 から TD(30分) 走査:
    PT に先に touch → 勝ち(WIN) / SL に先に touch → 負け(LOSS) / 両方未達 → 時間切れ(TO)
  即PT(d>=pt) は L+180 時点で既に PT 達成済み＝勝ち扱い（平行移動回収）。

【出力】(pt, ε) 格子で 勝率 / WIN / LOSS / TO 内訳、および期待取り幅
  EV ≈ WIN率·pt − LOSS率·ε（ATR単位、spread・ロット前）を出す。

【重要な留保（§30.6）】全て「脳が完璧にその時刻を撃てた場合の上限」。脳の選択力は別。
  最終 pt/ε は PT≈2脳を作り脳のエントリー選択込みで確定する。本検証は相場側の天井。

ATR はラベル同型自前計算（disc-aware TR → Wilder ewm α=1/13）。時刻系 全UTC。
使い方:
  python verify_pt_sl_race_grid.py \
      --feature-data /workspace/data/XAUUSD/stratum_1_base/master_processed \
      --tick-dir /workspace/data/XAUUSD/stratum_1_base/master_tick_partitioned \
      --out-dir  /workspace/data/diagnostics/pt_sl_race --window-min 30 --max-bets 0
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
PT_LEVELS = [1.0, 1.5, 2.0, 3.0]        # L基準PT幅
SL_LEVELS = [0.1, 0.2, 0.3, 0.4, 0.5]   # 建値SL幅(起点L+180基準)


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
    sl_arr = np.array(SL_LEVELS, dtype=np.float64)
    n_pt = len(pt_arr)
    n_sl = len(sl_arr)

    def _scan(entry_ts_us, price_L, price_E, atr, dir_, ticks_ts, ticks_px, t1_us):
        # 起点 = L+180(price_E)。PT = price_L + pt·ATR·dir(L基準)、SL = price_E − ε·ATR·dir(建値)。
        # 各 (pt, ε) について先に touch した方で WIN/LOSS、両方未達で TO を判定。
        #   outcome: 1=WIN(PT先), -1=LOSS(SL先), 0=TO
        n = len(entry_ts_us)
        outcome = np.zeros((n, n_pt, n_sl), dtype=np.int8)
        nt = len(ticks_ts)
        if nt == 0:
            return outcome
        for i in prange(n):
            a = atr[i]
            d = dir_[i]
            pl_ = price_L[i]
            ep = price_E[i]
            ets = entry_ts_us[i]
            start = np.searchsorted(ticks_ts, ets, side="right")
            # 各 pt の PT 到達フラグ / 各 ε の SL 到達フラグ（まだ決着していない格子だけ更新）
            pt_done = np.zeros(n_pt, dtype=np.int64)   # その pt が PT到達済みか
            sl_done = np.zeros(n_sl, dtype=np.int64)   # その ε が SL到達済みか
            # 格子の決着状態（0=未決, それ以外は outcome 確定）
            res = np.zeros((n_pt, n_sl), dtype=np.int8)
            n_open = n_pt * n_sl
            # 即PT: L+180 時点で既に price_L+pt·ATR を通過している pt は「弾く」（本番では
            # エントリー前にPT価格を行き過ぎ＝そのpt水準では取れない）。outcome=2=対象外。
            adv0 = (ep - pl_) * d / a   # = d_realized（L→L+180 の順行, ATR）
            for kp in range(n_pt):
                if adv0 >= pt_arr[kp]:
                    for ks in range(n_sl):
                        if res[kp, ks] == 0:
                            res[kp, ks] = 2   # 即PT弾き（対象外）
                            n_open -= 1
            for j in range(start, nt):
                if n_open <= 0:
                    break
                tt = ticks_ts[j]
                if tt > t1_us[i]:
                    break
                px = ticks_px[j]
                adv = (px - ep) * d / a          # 起点(L+180)からの順行(ATR)
                adv_from_L = (px - pl_) * d / a   # L からの順行(ATR) … PT判定はL基準
                against = -adv                    # 起点からの逆行(ATR)
                # PT 到達（L基準）
                for kp in range(n_pt):
                    if pt_done[kp] == 0 and adv_from_L >= pt_arr[kp]:
                        pt_done[kp] = 1
                # SL 到達（建値, 起点基準）
                for ks in range(n_sl):
                    if sl_done[ks] == 0 and against >= sl_arr[ks]:
                        sl_done[ks] = 1
                # 未決の格子に決着を反映（この tick で PT/SL どちらが立ったか）
                for kp in range(n_pt):
                    for ks in range(n_sl):
                        if res[kp, ks] != 0:
                            continue
                        if pt_done[kp] == 1:
                            res[kp, ks] = 1   # WIN（PT先）
                            n_open -= 1
                        elif sl_done[ks] == 1:
                            res[kp, ks] = -1  # LOSS（SL先）
                            n_open -= 1
            for kp in range(n_pt):
                for ks in range(n_sl):
                    outcome[i, kp, ks] = res[kp, ks]  # 残りは 0=TO
        return outcome

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
    entry_ts = L_us + entry_offset           # L+180 起点
    t1_us = L_us + entry_offset + window_us  # L+180 から TD(30分)

    # PT(L基準) × SL(建値) レース
    outcome = _SCAN(entry_ts, price_L, price_E, atr, dir_, ts_us, px, t1_us)
    # outcome[i, kp, ks]: 1=WIN(PT先), -1=LOSS(SL先), 0=TO, 2=即PT弾き(対象外)

    dr = d_realized
    vmask = valid

    sep("母集団（脳CSV不使用・全 M3 グリッド・PT×SL レース・L+180 起点）")
    print(f"  グリッド時刻: {len(feat):,}  / 有効(方向確定): {int(vmask.sum()):,}")
    print("  即PT(d>=pt)はその pt 水準で弾く（対象外）。エントリー対象は d<pt の時刻のみ。")
    print("  WIN時手取り = pt − d_realized（L+180 からの実取り幅, ATR・spread前）。LOSS損 = ε。")

    n_pt = len(PT_LEVELS)
    n_sl = len(SL_LEVELS)

    def cell(kp, ks):
        o = outcome[vmask, kp, ks]
        d = dr[vmask]
        elig = (o != 2)                  # 即PT弾きを除外＝エントリー対象
        ne = int(elig.sum())
        if ne == 0:
            return None
        win = (o == 1)
        loss = (o == -1)
        to = (o == 0) & elig
        win_rate = win.sum() / ne * 100
        loss_rate = loss.sum() / ne * 100
        to_rate = to.sum() / ne * 100
        reject_rate = (~elig).sum() / len(o) * 100
        # 実手取り(ATR): WINは pt−d, LOSSは −ε, TOは0（建値付近で時間切れ≈0）
        gain = (PT_LEVELS[kp] - d[win]).sum() - SL_LEVELS[ks] * loss.sum()
        ev = gain / ne
        return dict(ne=ne, win=win_rate, loss=loss_rate, to=to_rate,
                    reject=reject_rate, ev=ev)

    # 勝率格子
    sep("勝率（WIN率, %）格子 — 行=PT(L基準), 列=SL建値ε  ［即PT弾き後・エントリー対象内］")
    print("  " + f"{'PT＼ε':>8} " + " ".join(f"{lv:>6.1f}" for lv in SL_LEVELS))
    for kp, pt in enumerate(PT_LEVELS):
        row = []
        for ks in range(n_sl):
            c = cell(kp, ks)
            row.append(f"{c['win']:6.1f}" if c else "   N/A")
        print(f"  {pt:>8.1f} " + " ".join(row))

    # EV 格子（実手取り ATR）
    sep("EV格子 — WIN率·(pt−d) − LOSS率·ε（ATR単位の実取り幅期待値, spread・ロット前）")
    print("  " + f"{'PT＼ε':>8} " + " ".join(f"{lv:>6.1f}" for lv in SL_LEVELS))
    best = (None, None, -1e9)
    for kp, pt in enumerate(PT_LEVELS):
        row = []
        for ks in range(n_sl):
            c = cell(kp, ks)
            if c:
                row.append(f"{c['ev']:6.3f}")
                if c['ev'] > best[2]:
                    best = (pt, SL_LEVELS[ks], c['ev'])
            else:
                row.append("   N/A")
        print(f"  {pt:>8.1f} " + " ".join(row))
    print(f"\n  → EV 最大: PT={best[0]} / ε={best[1]}  (EV {best[2]:.3f} ATR/トレード, spread前)")

    # 決着内訳（代表 ε=0.3 列）と 即PT弾き率（pt 別）
    sep("内訳（代表 ε=0.3 列）— PT別に WIN/LOSS/TO率 と 即PT弾き率・エントリー件数")
    ks03 = SL_LEVELS.index(0.3) if 0.3 in SL_LEVELS else n_sl // 2
    print(f"  {'PT':>6} {'弾き率':>7} {'件数':>9} {'WIN':>7} {'LOSS':>7} {'TO':>7}")
    for kp, pt in enumerate(PT_LEVELS):
        c = cell(kp, ks03)
        if c:
            print(f"  {pt:>6.1f} {c['reject']:>6.1f}% {c['ne']:>9,} "
                  f"{c['win']:>6.1f}% {c['loss']:>6.1f}% {c['to']:>6.1f}%")

    # 保存（代表格子をフラット展開）
    out = pd.DataFrame({
        "timestamp": feat["timestamp"].to_numpy(),
        "atr": atr, "atr_ratio": ratio, "dir": dir_, "d_realized": dr,
    })
    for kp, pt in enumerate(PT_LEVELS):
        for ks, sl in enumerate(SL_LEVELS):
            out[f"oc_pt{pt}_sl{sl}"] = outcome[:, kp, ks]
    out.to_parquet(args.out_dir / "pt_sl_race_grid.parquet", index=False)
    sep("完了")
    print(f"  出力: {args.out_dir}/pt_sl_race_grid.parquet")
    print("  読み方: 勝率格子とEV格子で、PT(L基準)×SL(建値)の最良点を見る。")
    print("    EVは pt−d の実取り幅（平行移動の手取り）。即PT(d>=pt)は弾き済み。")
    print("    全て脳の選択力を引く前の相場側の上限。最終は PT≈2脳で脳込み確定（§30.6）。")


if __name__ == "__main__":
    main()
