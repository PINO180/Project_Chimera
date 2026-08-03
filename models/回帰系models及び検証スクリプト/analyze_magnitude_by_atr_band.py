# /workspace/models/analyze_magnitude_by_atr_band.py
# ============================================================================
# 【量の脳(mfe/mae)の ATR帯別 診断 — 「atr_ratio 0.8 で絞るべきか」への回答】
# ----------------------------------------------------------------------------
# 問題意識:
#   旧トリプルバリアは atr_ratio (min_atr_threshold=0.8) でラベリングから絞っていたが、
#   回帰化で全バーが無条件にラベル化されている (is_trigger=1 が全行=フィルタ無効)。
#   一方、ATR建ての MFE/MAE は低ボラ帯ほど 3-4 倍大きい (ATR が分母のため)。
#   学習は L1 (絶対誤差) なので、目標値の大きいサンプルが勾配を支配する。
#   → 「脳の容量が、取引しない低ボラバーにどれだけ費やされているか」を測る。
#
# 測るもの (既存 OOF + S6、再学習も tick走査も不要):
#   ATR帯ごとに:
#     - n / 割合%              … その帯のサンプル数
#     - 真値平均 / 予測平均      … 低ボラ帯で目標値が膨らんでいるかの確認
#     - MAE (絶対誤差)          … その帯での誤差の大きさ
#     - L1負担%                … 全L1損失のうちこの帯が占める割合 ★容量の使われ方
#     - Pearson                … その帯で量の脳が効いているか
#   最後に atr>=0.8 / atr<0.8 の2群サマリー (=旧フィルタ相当の切り口)。
#
# 判定:
#   - atr<0.8 の L1負担% が サンプル割合% を大きく上回る → 容量が無駄遣いされている
#     = ラベリングを atr>=0.8 に絞る価値あり (取引帯に容量を集中できる)
#   - L1負担% ≒ サンプル割合% かつ 帯別 Pearson が平坦 → 絞っても量は改善しない
#     (GBM は atr_ratio を特徴に持つので既に条件付けできている)
#
# 【実行】 python models/analyze_magnitude_by_atr_band.py
# ============================================================================
# 【調整パラメータ】
ATR_BINS = [0.0, 0.5, 0.8, 1.0, 1.2, 1.5, 2.0, 3.0]  # 最後は >= 3.0
FILTER_THRESHOLD = 0.8  # 旧 min_atr_threshold 相当の切り口

import sys
import logging
from pathlib import Path

import numpy as np
import polars as pl

_project_root = Path(__file__).resolve().parents[1]
if str(_project_root) not in sys.path:
    sys.path.append(str(_project_root))

from blueprint import (
    S6_LABELED_DATASET,
    S7_M1_OOF_PREDICTIONS_LONG,
    S7_M1_OOF_PREDICTIONS_SHORT,
)

REPORT_PATH = (
    S6_LABELED_DATASET.parent / "true_mfe_mae_analysis" / "magnitude_atr_band_report.txt"
)

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


def _load(oof_path: Path, name: str) -> pl.DataFrame:
    logging.info(f"Loading {name} OOF: {oof_path}")
    oof = pl.read_parquet(oof_path).select(
        [c for c in ["timestamp", "prediction", "true_label"] if c in
         pl.read_parquet(oof_path).columns]
    )
    return oof.with_columns(pl.col("timestamp").cast(pl.Datetime("us", "UTC")))


def _pearson(x, y):
    v = np.isfinite(x) & np.isfinite(y)
    if v.sum() < 3:
        return float("nan")
    x, y = x[v], y[v]
    if x.std() == 0 or y.std() == 0:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def _band_masks(ar: np.ndarray):
    out = []
    for i in range(len(ATR_BINS)):
        lo = ATR_BINS[i]
        if i < len(ATR_BINS) - 1:
            hi = ATR_BINS[i + 1]
            out.append((f"atr {lo:g}-{hi:g}", np.isfinite(ar) & (ar >= lo) & (ar < hi)))
        else:
            out.append((f"atr >= {lo:g}", np.isfinite(ar) & (ar >= lo)))
    return out


def _section(lines, title, t, p, ar):
    """帯別: n/割合/真値平均/予測平均/MAE/L1負担%/Pearson"""
    fin = np.isfinite(t) & np.isfinite(p) & np.isfinite(ar)
    t, p, ar = t[fin], p[fin], ar[fin]
    abs_err = np.abs(p - t)
    total_l1 = abs_err.sum()
    n_all = len(t)

    lines.append("-" * 92)
    lines.append(f"  {title}   (n={n_all:,}  全体Pearson={_pearson(t, p):.4f})")
    lines.append("-" * 92)
    lines.append(
        f"  {'帯':<12}{'n':>8}{'割合%':>7} | {'真値平均':>8}{'予測平均':>9} | "
        f"{'MAE':>7} | {'L1負担%':>8} | {'Pearson':>8} | {'負担/割合':>8}"
    )
    lines.append("  " + "-" * 88)
    for label, m in _band_masks(ar):
        nb = int(m.sum())
        if nb == 0:
            continue
        share_n = nb / n_all * 100
        share_l1 = abs_err[m].sum() / total_l1 * 100 if total_l1 > 0 else float("nan")
        ratio = share_l1 / share_n if share_n > 0 else float("nan")
        lines.append(
            f"  {label:<12}{nb:>8,}{share_n:>7.1f} | {t[m].mean():>8.3f}{p[m].mean():>9.3f} | "
            f"{abs_err[m].mean():>7.3f} | {share_l1:>8.1f} | {_pearson(t[m], p[m]):>8.4f} | "
            f"{ratio:>8.2f}"
        )
    # 2群サマリー (旧フィルタ相当)
    lo_m = np.isfinite(ar) & (ar < FILTER_THRESHOLD)
    hi_m = np.isfinite(ar) & (ar >= FILTER_THRESHOLD)
    lines.append("  " + "-" * 88)
    for label, m in [
        (f"atr < {FILTER_THRESHOLD:g} (除外候補)", lo_m),
        (f"atr >= {FILTER_THRESHOLD:g} (取引帯)", hi_m),
    ]:
        nb = int(m.sum())
        if nb == 0:
            continue
        share_n = nb / n_all * 100
        share_l1 = abs_err[m].sum() / total_l1 * 100 if total_l1 > 0 else float("nan")
        lines.append(
            f"  {label:<22}{nb:>8,}{share_n:>7.1f}% | 真値平均={t[m].mean():>7.3f} | "
            f"L1負担={share_l1:>5.1f}% | Pearson={_pearson(t[m], p[m]):>7.4f} | "
            f"負担/割合={share_l1/share_n:>5.2f}"
        )
    lines.append("")


def run() -> None:
    logging.info("Loading S6 atr_ratio...")
    s6 = (
        pl.scan_parquet(str(S6_LABELED_DATASET / "**/*.parquet"))
        .select(["timestamp", "atr_ratio"])
        .collect()
        .with_columns(pl.col("timestamp").cast(pl.Datetime("us", "UTC")))
        .unique("timestamp", keep="first")
    )

    lines = []
    lines.append("=" * 92)
    lines.append("  量の脳 (mfe/mae) の ATR帯別 診断 — 「atr_ratio 0.8 で絞るべきか」")
    lines.append("=" * 92)
    lines.append("  L1負担% = 全L1損失のうちその帯が占める割合 (= 学習容量がそこにどれだけ食われているか)")
    lines.append("  負担/割合 = L1負担% ÷ サンプル割合%。 1.0 なら公平、>1 ならその帯が容量を余分に食っている")
    lines.append("")

    for name, path in [
        ("M1-MFE (上の伸び)", S7_M1_OOF_PREDICTIONS_LONG),
        ("M1-MAE (下の伸び)", S7_M1_OOF_PREDICTIONS_SHORT),
    ]:
        if not path.exists():
            lines.append(f"  !! {name}: OOF が見つかりません: {path}\n")
            continue
        oof = _load(path, name)
        df = oof.join(s6, on="timestamp", how="inner")
        t = df["true_label"].to_numpy().astype(float)
        p = df["prediction"].to_numpy().astype(float)
        ar = df["atr_ratio"].to_numpy().astype(float)
        _section(lines, name, t, p, ar)

    lines.append("=" * 92)
    report = "\n".join(lines)
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(report, encoding="utf-8")
    print(report)
    logging.info(f"レポート保存: {REPORT_PATH}")


if __name__ == "__main__":
    run()
