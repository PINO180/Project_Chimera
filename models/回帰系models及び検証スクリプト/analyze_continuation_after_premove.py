# /workspace/models/analyze_continuation_after_premove.py
# ============================================================================
# 【継続性検定: 既発生運動 d で条件付けた建値クロス順行 (シナリオF検定)】
# ----------------------------------------------------------------------------
# 問い: 「これから上か下か」ではなく「[L, L+180) で既に起きた運動 d が、
#        エントリー後の建値クロス順行に継続するか」。
#   d はエントリー価格に織り込み済みなので、エントリー起点の MFE_be/MAE_be は
#   そのまま "d 差引後の継続分"。d のバケツ別に順方向/逆方向の非対称を測る。
#
# データ:
#   1. per_bar_breakeven parquet (analyze_breakeven_mfe_mae_distribution.py の出力)
#   2. S6 の close-to-close → d_atr = (close(L) − close(前バー))/ATR = [L,L+180) の運動
#
# 出力 (α ごと × d バケツごと):
#   n / MFE_be中央 / MAE_be中央 / 比 / 順方向優勢率 / 継続比中央 (順方向順行/|d|)
#   + 全体 Spearman(d, mfe_be−mae_be)
#
# 判定基準 (事前固定):
#   - 順方向優勢率が d の大きさに対して単調に 50% から乖離 → シナリオF生存
#   - 全バケツで ~50% 平坦 → このタイムスケールの方向は完全死亡 (量×フィルタへ撤退)
#
# 【実行】 (/workspace から。per_bar_breakeven を先に生成しておくこと)
#   python models/analyze_continuation_after_premove.py
# ============================================================================
# 【調整パラメータ】
ALPHA_TAGS = ["a005", "a010", "a025", "a050", "a075"]  # 見たい α (parquet の列タグ)
ALPHA_LABELS = {"a005": 0.05, "a010": 0.10, "a025": 0.25, "a050": 0.50, "a075": 0.75}
# d (ATR建て・符号付き) のバケツ境界
D_BINS = [-1.5, -1.0, -0.6, -0.3, -0.1, 0.1, 0.3, 0.6, 1.0, 1.5]
MIN_ABS_D_FOR_CONT = 0.10  # 継続比 (順行/|d|) を出す最小 |d| (ゼロ割・ノイズ回避)

import sys
import logging
from pathlib import Path

import numpy as np
import polars as pl

_project_root = Path(__file__).resolve().parents[1]
if str(_project_root) not in sys.path:
    sys.path.append(str(_project_root))

from blueprint import S6_LABELED_DATASET

BE_DIR = S6_LABELED_DATASET.parent / "true_mfe_mae_analysis" / "per_bar_breakeven"
REPORT_PATH = (
    S6_LABELED_DATASET.parent / "true_mfe_mae_analysis" / "continuation_report.txt"
)

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


def _load_joined() -> pl.DataFrame:
    logging.info(f"Loading breakeven per-bar: {BE_DIR}")
    be = pl.scan_parquet(str(BE_DIR / "**/*.parquet")).collect()
    logging.info(f"  -> {be.height} bars (breakeven)")

    logging.info("Loading S6 trigger bars (close/atr_value) for d...")
    # [座標系] S6 の close(L) = L+180 時点の価格 (エントリー価格)。
    #   前の M3 バーの close = L ちょうどの価格。よって
    #   d = close(L) − close(前バー) = price(L+180) − price(L) = [L, L+180) の運動そのもの。
    #   (S6 に open は無い。close-to-close の方が L の瞬間を正確に捉える)
    #   前バーとの間隔が 180 秒でない行 (週末・欠損跨ぎ) は d=null で除外。
    s6 = (
        pl.scan_parquet(str(S6_LABELED_DATASET / "**/*.parquet"))
        .select(["timestamp", "close", "atr_value"])
        .collect()
        .with_columns(pl.col("timestamp").cast(pl.Datetime("us", "UTC")))
        .unique("timestamp", keep="first")
        .sort("timestamp")
        .rename({"close": "_close_s6", "atr_value": "_atr_s6"})
    )
    s6 = s6.with_columns(
        pl.col("_close_s6").shift(1).alias("_prev_close"),
        (pl.col("timestamp") - pl.col("timestamp").shift(1))
        .dt.total_seconds()
        .alias("_gap_sec"),
    )
    df = be.join(
        s6.select(["timestamp", "_close_s6", "_atr_s6", "_prev_close", "_gap_sec"]),
        on="timestamp",
        how="inner",
    )
    logging.info(f"  -> joined {df.height} bars")

    # サニティ: 結合整合 (close が一致するか)
    mism = df.filter(
        (pl.col("close") - pl.col("_close_s6")).abs() > 1e-6
    ).height
    if mism > 0:
        logging.warning(f"  !! close 不一致 {mism} 行 (結合キー要確認)")
    else:
        logging.info("  -> close 一致 (結合サニティ OK)")

    # d_atr = (close(L) − close(前バー)) / ATR。連続 (gap=180s) の行のみ。
    n_before = df.height
    df = df.filter(
        (pl.col("_atr_s6") > 0)
        & pl.col("_prev_close").is_not_null()
        & (pl.col("_gap_sec") == 180)
    ).with_columns(
        ((pl.col("_close_s6") - pl.col("_prev_close")) / pl.col("_atr_s6")).alias(
            "d_atr"
        )
    )
    logging.info(
        f"  -> d 計算可能 {df.height} 行 (gap!=180s 等で除外 {n_before - df.height})"
    )
    return df


def _bucket_label(i: int) -> str:
    if i == 0:
        return f"< {D_BINS[0]:g}"
    if i == len(D_BINS):
        return f">= {D_BINS[-1]:g}"
    return f"{D_BINS[i-1]:g}..{D_BINS[i]:g}"


def _bucket_index(d: float) -> int:
    # 0 = 左端未満, len(D_BINS) = 右端以上
    return int(np.searchsorted(np.array(D_BINS), d, side="right"))


def _med(x):
    a = np.asarray([v for v in x if v is not None and np.isfinite(v)], dtype=float)
    return float(np.median(a)) if a.size else float("nan")


def _spearman(x, y):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    v = np.isfinite(x) & np.isfinite(y)
    if v.sum() < 3:
        return float("nan")
    rx = np.argsort(np.argsort(x[v])).astype(float)
    ry = np.argsort(np.argsort(y[v])).astype(float)
    if rx.std() == 0 or ry.std() == 0:
        return float("nan")
    return float(np.corrcoef(rx, ry)[0, 1])


def run() -> None:
    df = _load_joined()
    n_total = df.height
    d_all = df["d_atr"].to_numpy()

    lines = []
    lines.append("=" * 76)
    lines.append("  継続性検定: 既発生運動 d ([L,L+180) = close-to-close /ATR) 条件付き 建値クロス順行")
    lines.append("=" * 76)
    lines.append(f"対象バー数: {n_total}   期間: {df['timestamp'].min()} - {df['timestamp'].max()}")
    _dv = d_all[np.isfinite(d_all)]
    lines.append(
        f"d_atr 分布: mean={_dv.mean():+.3f} med={np.median(_dv):+.3f} "
        f"p10={np.percentile(_dv,10):+.3f} p90={np.percentile(_dv,90):+.3f}"
    )
    lines.append("")
    lines.append("  順方向優勢率 = d>0 なら P(MFE_be > MAE_be)、d<0 なら P(MAE_be > MFE_be)")
    lines.append(f"  継続比       = 順方向順行 / |d| の中央値 (|d| >= {MIN_ABS_D_FOR_CONT} のみ)")
    lines.append("  ※ 各バーの順行は TO15 右打ち切りを含む (打ち切り値は下限値)。バー重複による系列相関あり、n は名目。")
    lines.append("")

    for tag in ALPHA_TAGS:
        a = ALPHA_LABELS[tag]
        mfe = df[f"mfe_be_atr_{tag}"].to_numpy()
        mae = df[f"mae_be_atr_{tag}"].to_numpy()
        diff = mfe - mae

        sp = _spearman(d_all, diff)
        lines.append("=" * 76)
        lines.append(
            f"◆ α = {a} ATR   |   Spearman( d , MFE_be−MAE_be ) = {sp:+.4f}"
        )
        lines.append("=" * 76)
        lines.append(
            f"  {'dバケツ':<12}{'n':>7} | {'MFE_be中':>8}{'MAE_be中':>9}{'比':>7} | "
            f"{'順方向優勢率':>10} | {'継続比中':>8}"
        )
        lines.append("  " + "-" * 68)

        for bi in range(len(D_BINS) + 1):
            idx = np.array([_bucket_index(d) == bi if np.isfinite(d) else False
                            for d in d_all])
            nb = int(idx.sum())
            if nb == 0:
                continue
            m_mfe = _med(mfe[idx])
            m_mae = _med(mae[idx])
            ratio = m_mfe / m_mae if (np.isfinite(m_mae) and m_mae > 0) else float("nan")

            db = d_all[idx]
            fe = mfe[idx]
            ae = mae[idx]
            fin = np.isfinite(fe) & np.isfinite(ae) & np.isfinite(db)
            # 順方向優勢率 (d の符号方向の順行が逆を上回る率。d≈0 のバケツは参考)
            pos = db[fin] > 0
            win_same = np.where(pos, fe[fin] > ae[fin], ae[fin] > fe[fin])
            adv = float(win_same.mean() * 100) if fin.sum() else float("nan")
            # 継続比 = 順方向順行 / |d|
            big = fin & (np.abs(db) >= MIN_ABS_D_FOR_CONT)
            same_exc = np.where(db[big] > 0, fe[big], ae[big])
            cont = _med(same_exc / np.abs(db[big])) if big.sum() else float("nan")

            lines.append(
                f"  {_bucket_label(bi):<12}{nb:>7} | {m_mfe:>8.3f}{m_mae:>9.3f}"
                f"{ratio:>7.3f} | {adv:>9.2f}% | {cont:>8.3f}"
            )

        # 全体行 + |d| 上位10%行
        fin = np.isfinite(mfe) & np.isfinite(mae) & np.isfinite(d_all)
        pos = d_all[fin] > 0
        win_same = np.where(pos, mfe[fin] > mae[fin], mae[fin] > mfe[fin])
        adv_all = float(win_same.mean() * 100)
        q90 = np.nanquantile(np.abs(d_all), 0.9)
        top = fin & (np.abs(d_all) >= q90)
        pos_t = d_all[top] > 0
        win_t = np.where(pos_t, mfe[top] > mae[top], mae[top] > mfe[top])
        adv_top = float(win_t.mean() * 100) if top.sum() else float("nan")
        lines.append("  " + "-" * 68)
        lines.append(
            f"  {'ALL':<12}{int(fin.sum()):>7} | {'':>8}{'':>9}{'':>7} | "
            f"{adv_all:>9.2f}% |"
        )
        lines.append(
            f"  {'|d|上位10%':<12}{int(top.sum()):>7} | {'':>8}{'':>9}{'':>7} | "
            f"{adv_top:>9.2f}% |  (|d|>={q90:.3f})"
        )
        lines.append("")

    lines.append("=" * 76)
    report = "\n".join(lines)
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(report, encoding="utf-8")
    print(report)
    logging.info(f"レポート保存: {REPORT_PATH}")


if __name__ == "__main__":
    run()
