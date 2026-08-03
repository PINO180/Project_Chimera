# /workspace/models/analyze_direction_by_subpopulation.py
# ============================================================================
# 【部分集団別 方向検定 — 「全市場で測れば 50% は当たり前では?」への実験的回答】
# ----------------------------------------------------------------------------
# 問題意識:
#   これまでの方向検定は全 M3 バー(全市場)で測っている。効率的市場なら無条件平均が
#   50% になるのは当然。旧トリプルバリアは atr_ratio 等でラベリング段階から絞って
#   いた = モデルの容量が「面白いバー」の識別に集中していた (ラベルノイズ希釈の回避)。
#   → 「部分集団に絞れば方向は傾くのか」を、再学習せず既存データで測る。
#
# 2つを測る (どちらも既存 OOF + S6 から、tick走査も学習も不要):
#   (A) 素の偏り [モデル非依存]  : 各セルの P(mfe_atr > mae_atr)
#         = ML を一切使わず、その部分集団で方向が傾いているか。50% からの乖離が本体。
#   (B) モデル的中率 [OOF]       : 各セルの P(sign(pred) == sign(true))
#         = 全バー学習の脳が、その部分集団では効いていたか。
#
# セル軸: atr_ratio 帯 / セッション(JST) / |d| 帯 / hour_jst / セッション×atr帯 /
#         is_trigger (S6 にあれば)
#
# 統計の注意 (レポートにも明記):
#   z_naive は独立サンプル前提。実際は TO 窓 (既定30分) 内で M3 バーが約10本重複する
#   ため実効サンプルは約 1/EFF_N_DIVISOR。z_eff = z_naive / sqrt(EFF_N_DIVISOR) を併記し、
#   判定は z_eff で行う。多重比較 (セル数が多い) にも注意。
#
# 【実行】 python models/analyze_direction_by_subpopulation.py
# ============================================================================
# 【調整パラメータ】
EFF_N_DIVISOR = 10.0      # TO窓/バー間隔 = 30分/3分 ≒ 10 本重複 → 実効n = n/10
ATR_BINS = [0.0, 0.5, 0.8, 1.0, 1.2, 1.5, 2.0, 3.0]   # 最後は >= 3.0
ABS_D_BINS = [0.0, 0.2, 0.5, 1.0, 1.5]                 # |d| 帯。最後は >= 1.5
MIN_CELL_N = 200          # これ未満のセルは表示するが判定対象外
FLAG_Z_EFF = 2.0          # |z_eff| がこれ以上のセルを注目印にする

import sys
import logging
from pathlib import Path

import numpy as np
import polars as pl

_project_root = Path(__file__).resolve().parents[1]
if str(_project_root) not in sys.path:
    sys.path.append(str(_project_root))

from blueprint import S6_LABELED_DATASET, S7_M1_OOF_PREDICTIONS_LONG

OOF_DOM = S7_M1_OOF_PREDICTIONS_LONG.parent / "m1_oof_predictions_dominance.parquet"
REPORT_PATH = (
    S6_LABELED_DATASET.parent / "true_mfe_mae_analysis" / "direction_subpopulation_report.txt"
)

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


def _hour_to_session_jst(h: int) -> str:
    if 9 <= h < 16:
        return "Tokyo"
    elif 16 <= h < 21:
        return "London"
    elif h >= 21 or h < 1:
        return "Overlap"
    elif 1 <= h < 6:
        return "NY"
    else:
        return "Oceania"


def _load() -> pl.DataFrame:
    if not OOF_DOM.exists():
        raise SystemExit(f"方向脳 OOF がありません: {OOF_DOM}")
    logging.info(f"Loading dominance OOF: {OOF_DOM}")
    oof = pl.read_parquet(OOF_DOM)
    cols = oof.columns
    logging.info(f"  -> {oof.height} rows, cols={cols}")
    sel = ["timestamp", "prediction", "true_label"]
    if "timeframe" in cols:
        sel.append("timeframe")
    oof = oof.select(sel).with_columns(
        pl.col("timestamp").cast(pl.Datetime("us", "UTC"))
    )

    logging.info("Loading S6 (atr_ratio / atr_value / close / mfe_atr / mae_atr)...")
    lf = pl.scan_parquet(str(S6_LABELED_DATASET / "**/*.parquet"))
    have = lf.collect_schema().names()
    want = ["timestamp", "close", "atr_value", "atr_ratio", "mfe_atr", "mae_atr"]
    for opt in ["is_trigger", "session_atr_ratio"]:
        if opt in have:
            want.append(opt)
    s6 = (
        lf.select([c for c in want if c in have])
        .collect()
        .with_columns(pl.col("timestamp").cast(pl.Datetime("us", "UTC")))
        .unique("timestamp", keep="first")
        .sort("timestamp")
    )
    # d = (close(L) − close(前バー)) / ATR   [L,L+180) の既発生運動。gap!=180s は除外
    s6 = s6.with_columns(
        pl.col("close").shift(1).alias("_prev_close"),
        (pl.col("timestamp") - pl.col("timestamp").shift(1))
        .dt.total_seconds()
        .alias("_gap_sec"),
    ).with_columns(
        pl.when(
            (pl.col("_gap_sec") == 180)
            & pl.col("_prev_close").is_not_null()
            & (pl.col("atr_value") > 0)
        )
        .then((pl.col("close") - pl.col("_prev_close")) / pl.col("atr_value"))
        .otherwise(None)
        .alias("d_atr")
    )

    df = oof.join(s6.drop(["_prev_close", "_gap_sec"]), on="timestamp", how="inner")
    logging.info(f"  -> joined {df.height} rows")

    df = df.with_columns(
        ((pl.col("timestamp").dt.hour() + 9) % 24).alias("hour_jst")
    ).with_columns(
        pl.col("hour_jst")
        .map_elements(_hour_to_session_jst, return_dtype=pl.Utf8)
        .alias("session")
    )
    return df


def _cell(true_v: np.ndarray, pred_v: np.ndarray) -> dict:
    """(A) 素の偏り P(true>0) と (B) 的中率 P(sign一致) を z 付きで返す。"""
    v = np.isfinite(true_v) & np.isfinite(pred_v) & (true_v != 0)
    n = int(v.sum())
    if n == 0:
        return None
    t, p = true_v[v], pred_v[v]
    base = float((t > 0).mean())
    hit = float((np.sign(p) == np.sign(t)).mean())
    se = np.sqrt(0.25 / n)
    z_base = (base - 0.5) / se
    z_hit = (hit - 0.5) / se
    k = np.sqrt(EFF_N_DIVISOR)
    return dict(
        n=n,
        base=base * 100,
        z_base=z_base / k,
        hit=hit * 100,
        z_hit=z_hit / k,
        absp=float(np.abs(p).mean()),
    )


def _fmt_rows(lines, title, cells):
    lines.append("-" * 82)
    lines.append(f"  {title}")
    lines.append("-" * 82)
    lines.append(
        f"  {'セル':<18}{'n':>8} | {'素の偏り%':>9}{'z_eff':>7} | "
        f"{'的中率%':>8}{'z_eff':>7} | {'|pred|平均':>9}"
    )
    lines.append("  " + "-" * 78)
    for label, st in cells:
        if st is None:
            continue
        flag = ""
        if st["n"] >= MIN_CELL_N:
            if abs(st["z_base"]) >= FLAG_Z_EFF:
                flag += " ★素"
            if abs(st["z_hit"]) >= FLAG_Z_EFF:
                flag += " ★的"
        small = "" if st["n"] >= MIN_CELL_N else " (n小)"
        lines.append(
            f"  {label:<18}{st['n']:>8} | {st['base']:>9.2f}{st['z_base']:>+7.2f} | "
            f"{st['hit']:>8.2f}{st['z_hit']:>+7.2f} | {st['absp']:>9.3f}{flag}{small}"
        )
    lines.append("")


def run() -> None:
    df = _load()
    t = df["true_label"].to_numpy().astype(float)
    p = df["prediction"].to_numpy().astype(float)
    ar = df["atr_ratio"].to_numpy().astype(float)
    d = df["d_atr"].to_numpy().astype(float)
    sess = df["session"].to_list()
    hj = df["hour_jst"].to_numpy()

    lines = []
    lines.append("=" * 82)
    lines.append("  部分集団別 方向検定 (再学習なし / 既存 dominance OOF + S6)")
    lines.append("=" * 82)
    lines.append(f"対象: {df.height:,} 行   期間: {df['timestamp'].min()} - {df['timestamp'].max()}")
    lines.append("")
    lines.append("  素の偏り = P(mfe_atr > mae_atr)  … ML非依存。50%からの乖離がその部分集団の方向傾斜")
    lines.append("  的中率   = P(sign(pred)==sign(true)) … 全バー学習の方向脳がそのセルで効いたか")
    lines.append(f"  z_eff    = z_naive / sqrt({EFF_N_DIVISOR:g})  … TO窓内のバー重複を補正した実効z")
    lines.append(f"  ★印      = |z_eff| >= {FLAG_Z_EFF} かつ n >= {MIN_CELL_N}。多重比較のため単発の★は要警戒")
    lines.append("")

    ov = _cell(t, p)
    lines.append("-" * 82)
    lines.append(
        f"  【全体】 n={ov['n']:,}  素の偏り={ov['base']:.2f}% (z_eff {ov['z_base']:+.2f})  "
        f"的中率={ov['hit']:.2f}% (z_eff {ov['z_hit']:+.2f})"
    )
    lines.append("")

    # ── atr_ratio 帯 ──
    cells = []
    for i in range(len(ATR_BINS)):
        lo = ATR_BINS[i]
        if i < len(ATR_BINS) - 1:
            hi = ATR_BINS[i + 1]
            m = np.isfinite(ar) & (ar >= lo) & (ar < hi)
            lab = f"atr {lo:g}-{hi:g}"
        else:
            m = np.isfinite(ar) & (ar >= lo)
            lab = f"atr >= {lo:g}"
        cells.append((lab, _cell(t[m], p[m])))
    _fmt_rows(lines, "① atr_ratio 帯別", cells)

    # ── セッション ──
    cells = []
    for s in ["Tokyo", "London", "Overlap", "NY", "Oceania"]:
        m = np.array([x == s for x in sess])
        cells.append((s, _cell(t[m], p[m])))
    _fmt_rows(lines, "② セッション別 (JST)", cells)

    # ── |d| 帯 ──
    ad = np.abs(d)
    cells = []
    for i in range(len(ABS_D_BINS)):
        lo = ABS_D_BINS[i]
        if i < len(ABS_D_BINS) - 1:
            hi = ABS_D_BINS[i + 1]
            m = np.isfinite(ad) & (ad >= lo) & (ad < hi)
            lab = f"|d| {lo:g}-{hi:g}"
        else:
            m = np.isfinite(ad) & (ad >= lo)
            lab = f"|d| >= {lo:g}"
        cells.append((lab, _cell(t[m], p[m])))
    _fmt_rows(lines, "③ |d| 帯別 (既発生運動の大きさ)", cells)

    # ── hour_jst ──
    cells = []
    for h in range(24):
        m = hj == h
        if m.sum() == 0:
            continue
        cells.append((f"{h:02d}:00 JST", _cell(t[m], p[m])))
    _fmt_rows(lines, "④ 時間帯別 (JST)", cells)

    # ── セッション × atr 帯 (多重比較が最も強い。★は慎重に) ──
    cells = []
    for s in ["Tokyo", "London", "Overlap", "NY", "Oceania"]:
        ms = np.array([x == s for x in sess])
        for i in range(len(ATR_BINS)):
            lo = ATR_BINS[i]
            if i < len(ATR_BINS) - 1:
                hi = ATR_BINS[i + 1]
                m = ms & np.isfinite(ar) & (ar >= lo) & (ar < hi)
                lab = f"{s[:4]} atr{lo:g}-{hi:g}"
            else:
                m = ms & np.isfinite(ar) & (ar >= lo)
                lab = f"{s[:4]} atr>={lo:g}"
            st = _cell(t[m], p[m])
            if st and st["n"] >= MIN_CELL_N:
                cells.append((lab, st))
    _fmt_rows(lines, "⑤ セッション × atr_ratio 帯 (n>=200 のみ・多重比較に注意)", cells)

    # ── is_trigger ──
    if "is_trigger" in df.columns:
        trig = df["is_trigger"].to_numpy()
        cells = []
        for val in np.unique(trig[~pl.Series(trig).is_null().to_numpy()]):
            m = trig == val
            cells.append((f"is_trigger={val}", _cell(t[m], p[m])))
        _fmt_rows(lines, "⑥ is_trigger 別", cells)

    lines.append("=" * 82)
    report = "\n".join(lines)
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(report, encoding="utf-8")
    print(report)
    logging.info(f"レポート保存: {REPORT_PATH}")


if __name__ == "__main__":
    run()
