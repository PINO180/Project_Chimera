"""
baseline_ATR（分母=前日24h ATR平均）に基づくフィルタリング戦略の比較分析

baseline_ATR は ATR Ratio とは別の独立した次元:
  ATR Ratio   = current_ATR / baseline_ATR  (相対: 今日は昨日比どうか)
  baseline_ATR = rolling_mean(ATR, 480本)   (絶対: 昨日はどれだけ動いたか)

戦略候補:
  A. 現行: atr_ratio >= 0.8 のみ
  B. 絶対床追加: baseline_atr >= X かつ atr_ratio >= 0.8
  C. 条件付き閾値: baseline低時はatr_ratio閾値を引き上げ
  D. 複合: baseline >= X かつ atr_ratio >= Y
"""

import polars as pl
import numpy as np

LOG_PATH = "/workspace/data/XAUUSD/stratum_7_models/backtest_simulator_results/M2_20260504_195717_Th0.7_D0.3_R2 _Phase6_V5/detailed_trade_log_v5_M2.csv"

def classify_outcome(label, td):
    if label == 1: return "PT"
    elif td is not None and td < 29.9: return "SL"
    else: return "TO"

def stats(df):
    n = len(df)
    if n == 0: return {"n":0,"wr":0,"pf":0,"to_r":0,"sl_r":0}
    pt = df.filter(pl.col("outcome")=="PT")
    sl = df.filter(pl.col("outcome")=="SL")
    to = df.filter(pl.col("outcome")=="TO")
    win  = float(pt["pnl"].sum()) if len(pt)>0 else 0.0
    loss = abs(float(df.filter(pl.col("outcome")!="PT")["pnl"].sum()))
    return {
        "n":    n,
        "wr":   len(pt)/n,
        "pf":   win/loss if loss>0 else 99.0,
        "to_r": len(to)/n,
        "sl_r": len(sl)/n,
        "avg_td": float(df["TD"].mean() or 0),
    }

def prt(label, s, base_n=95392):
    excl = base_n - s["n"]
    pct  = s["n"]/base_n
    print(f"  {label:<45}  n={s['n']:6,}({pct:.1%})  除外={excl:5,}  "
          f"勝率={s['wr']:.2%}  PF={s['pf']:6.2f}  TO率={s['to_r']:.2%}  avgTD={s['avg_td']:.1f}min")

print("読み込み中...")
df = pl.read_csv(LOG_PATH, null_values=["NaN"])
df = df.with_columns([
    (pl.col("atr_value")/(pl.col("atr_ratio")+1e-10)).alias("baseline_atr"),
    pl.struct(["label","TD"]).map_elements(
        lambda r: classify_outcome(r["label"],r["TD"]),
        return_dtype=pl.String
    ).alias("outcome"),
    pl.col("pnl").cast(pl.Float64).alias("pnl"),
])

total = len(df)
print(f"総取引数: {total:,}件\n")

# baseline_ATRの分位点を確認
bases = df["baseline_atr"].to_numpy()
pcts = [10,20,25,30,40,50,60,70,75,80,90]
print("baseline_ATR 分位点:")
for p in pcts:
    print(f"  P{p:2d}: {np.percentile(bases, p):.4f}")
print()

# ============================================================
# 【前提確認】baseline_ATR と ATR Ratio は独立した次元か？
# ============================================================
print("="*100)
print("【前提確認】baseline_ATR が低い時、ATR Ratio はどう分布するか？")
print("  → 独立なら「baseline低でもATR Ratioは均等分布」")
print("  → 相関強いなら「baseline低ではATR Ratioも低い傾向」")
print("="*100)
q_b = np.quantile(bases, [0, 1/3, 2/3, 1.0])
for i, bl in enumerate(["Base低(前日静)", "Base中", "Base高(前日荒)"]):
    lo, hi = q_b[i], q_b[i+1]
    sub = df.filter((pl.col("baseline_atr")>=lo)&(pl.col("baseline_atr")<=(hi if i==2 else hi-1e-10)))
    ratios = sub["atr_ratio"].to_numpy()
    print(f"  {bl}(baseline {lo:.3f}-{hi:.3f}): "
          f"atr_ratio avg={ratios.mean():.3f}  "
          f"P25={np.percentile(ratios,25):.3f}  "
          f"P50={np.percentile(ratios,50):.3f}  "
          f"P75={np.percentile(ratios,75):.3f}  "
          f"P90={np.percentile(ratios,90):.3f}")

# ============================================================
# 【戦略A】現行
# ============================================================
print("\n" + "="*100)
print("【戦略別 成績比較】")
print("="*100)
print("\n戦略A: 現行 (atr_ratio >= 0.80 のみ)")
prt("A: ratio>=0.80", stats(df))

# ============================================================
# 【戦略B】baseline絶対床を追加
# ============================================================
print("\n戦略B: baseline絶対床追加 (baseline_atr >= X かつ ratio >= 0.80)")
for x in [0.70, 0.75, 0.80, 0.82, 0.85, 0.90, 1.00, 1.10]:
    sub = df.filter((pl.col("baseline_atr")>=x)&(pl.col("atr_ratio")>=0.80))
    prt(f"B: baseline>={x:.2f} & ratio>=0.80", stats(sub))

# ============================================================
# 【戦略C】baseline低時にATR Ratio閾値を引き上げ
# ============================================================
print("\n戦略C: 条件付き閾値 (baseline低時はratio閾値を引き上げ)")
baseline_cutoff = np.percentile(bases, 33)  # Q1/3 境界
print(f"  baseline低の定義: < {baseline_cutoff:.4f} (下位33%)")
for high_thr in [0.85, 0.90, 0.95, 1.00, 1.10]:
    # baseline低: ratio >= high_thr、baseline中高: ratio >= 0.80
    sub = df.filter(
        ((pl.col("baseline_atr") <  baseline_cutoff) & (pl.col("atr_ratio") >= high_thr)) |
        ((pl.col("baseline_atr") >= baseline_cutoff) & (pl.col("atr_ratio") >= 0.80))
    )
    prt(f"C: baseline<{baseline_cutoff:.3f}→ratio>={high_thr:.2f}, else>=0.80", stats(sub))

# ============================================================
# 【戦略D】絶対床 + ratio閾値の複合
# ============================================================
print("\n戦略D: 複合 (baseline絶対床 + ratio閾値両方)")
combos = [
    (0.80, 0.85), (0.80, 0.90),
    (0.82, 0.85), (0.82, 0.90),
    (0.85, 0.85), (0.85, 0.90),
    (0.90, 0.90),
]
for b_floor, r_thr in combos:
    sub = df.filter((pl.col("baseline_atr")>=b_floor)&(pl.col("atr_ratio")>=r_thr))
    prt(f"D: baseline>={b_floor:.2f} & ratio>={r_thr:.2f}", stats(sub))

# ============================================================
# 【除外される取引の性質確認】
# ============================================================
print("\n" + "="*100)
print("【除外される取引の性質】 baseline_atr < 0.82 の19,079件の内訳")
print("="*100)
excl = df.filter(pl.col("baseline_atr") < 0.82)
incl = df.filter(pl.col("baseline_atr") >= 0.82)
print(f"  除外対象: {len(excl):,}件 ({len(excl)/total:.1%})")
print(f"  残存:     {len(incl):,}件 ({len(incl)/total:.1%})")
se = stats(excl)
si = stats(incl)
print(f"  除外側: 勝率={se['wr']:.2%} PF={se['pf']:.2f} TO率={se['to_r']:.2%} avgTD={se['avg_td']:.1f}min")
print(f"  残存側: 勝率={si['wr']:.2%} PF={si['pf']:.2f} TO率={si['to_r']:.2%} avgTD={si['avg_td']:.1f}min")
print()
print("  除外対象のATR Ratio分布:")
excl_r = excl["atr_ratio"].to_numpy()
for lo, hi in [(0.80,0.90),(0.90,1.00),(1.00,1.20),(1.20,1.50),(1.50,99)]:
    cnt = ((excl_r>=lo)&(excl_r<hi)).sum()
    print(f"    ratio [{lo:.2f}-{hi:.2f}): {cnt:,}件 ({cnt/len(excl):.1%})")

print("\n完了。")
