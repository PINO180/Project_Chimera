# snapshot_comparator.py
# =====================================================================
# スナップショット vs S6 自動比較ツール
#
# 【目的】
#   feature_snapshot_tool.pyが出力したスナップショットCSVの特徴量値を
#   S6データの同タイムスタンプの値と全件自動比較する。
#   「本番の特徴量生成が学習時と一致しているか」を特徴量レベルで確認する。
#
# 【使い方】
#   python snapshot_comparator.py                          # 最新スナップショットを使用
#   python snapshot_comparator.py --snapshot_file xxx.csv  # ファイル指定
#   python snapshot_comparator.py --top_n 100              # TOP100まで表示
#   python snapshot_comparator.py --include_raw            # 純化前raw特徴量も比較対象に含める
#
# 【出力】
#   /workspace/data/diagnostics/comparisons/
#     comparison_YYYYMMDD_HHMMSS.csv  # 全特徴量の差分一覧
#     summary_YYYYMMDD_HHMMSS.txt     # サマリーレポート
# =====================================================================

import sys
import logging
import argparse
import warnings
import re
import numpy as np
import pandas as pd
import polars as pl
from pathlib import Path

warnings.filterwarnings("ignore")

logging.root.setLevel(logging.ERROR)
_handler = logging.StreamHandler()
_handler.setLevel(logging.INFO)
_handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
logger = logging.getLogger("Comparator")
logger.setLevel(logging.INFO)
logger.addHandler(_handler)
logger.propagate = False

project_root = Path(__file__).resolve().parents[1]
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

from blueprint import S6_WEIGHTED_DATASET

SNAPSHOT_DIR = Path("/workspace/data/diagnostics/feature_snapshots")
OUTPUT_DIR   = Path("/workspace/data/diagnostics/comparisons")

# 乖離の分類閾値（相対乖離率ベース）
THRESH_EXACT = 0.001   # 0.1%以下 → 完全一致
THRESH_CLOSE = 0.05    # 5%以下   → 近似
THRESH_WARN  = 0.20    # 20%以下  → 要注意
                       # 20%超    → 異常


def load_snapshot(csv_path: Path) -> tuple:
    """スナップショットCSVを読み込んでタイムスタンプと特徴量辞書を返す"""
    df = pd.read_csv(csv_path, header=0, names=["feature_name", "value"])

    meta = df[df["feature_name"].str.startswith("_")].set_index("feature_name")["value"].to_dict()
    ts_str = meta.get("_timestamp_utc")
    if ts_str is None:
        raise ValueError(f"タイムスタンプが見つかりません: {csv_path}")

    snapshot_ts = pd.Timestamp(ts_str).tz_convert("UTC")

    features = df[
        ~df["feature_name"].str.startswith("_") &
        (df["feature_name"] != "---")
    ].copy()
    features["value"] = pd.to_numeric(features["value"], errors="coerce")
    feat_dict = features.set_index("feature_name")["value"].to_dict()

    return snapshot_ts, feat_dict


def load_s6_samples(
    feature_names: list,
    n_samples: int = 200,
    include_raw: bool = False,
) -> pd.DataFrame:
    """S6データからランダムにn_samples件取得する"""
    # neutralized特徴量は常に含める
    cols_neutralized = [f for f in feature_names if "_neutralized" in f]
    # raw特徴量: e1X_で始まり_neutralizedを含まないもの
    cols_raw = [
        f for f in feature_names
        if re.match(r"e1[a-f]_", f) and "_neutralized" not in f
    ] if include_raw else []

    cols_to_select = ["timestamp"] + cols_neutralized + cols_raw

    lf = pl.scan_parquet(str(S6_WEIGHTED_DATASET / "**/*.parquet"))
    # S6に存在するカラムのみ選択（rawはS6にない場合もある）
    available = lf.collect_schema().names()
    cols_to_select = [c for c in cols_to_select if c in available]

    df = lf.select(cols_to_select).collect()
    if df.is_empty():
        return None

    sample = df.sample(n=min(n_samples, len(df)), seed=42, shuffle=True)
    return sample.to_pandas()


def classify(abs_diff: float, s6_val: float) -> str:
    scale = max(abs(s6_val), 1e-6)
    rel = abs_diff / scale
    if rel <= THRESH_EXACT:
        return "✅ 一致"
    elif rel <= THRESH_CLOSE:
        return "🟡 近似"
    elif rel <= THRESH_WARN:
        return "🟠 要注意"
    else:
        return "🔴 異常"


def extract_module(feat_name: str) -> str:
    m = re.match(r"e1([a-f])_", feat_name)
    return f"1{m.group(1).upper()}" if m else "OTHER"


def extract_timeframe(feat_name: str) -> str:
    m = re.search(r"_(M[\d\.]+|H\d+)$", feat_name)
    return m.group(1) if m else "?"


def compare(
    snapshot_path: Path,
    top_n: int = 50,
    n_s6_samples: int = 200,
    include_raw: bool = False,
) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    logger.info(f"スナップショット読み込み: {snapshot_path.name}")
    snapshot_ts, feat_dict = load_snapshot(snapshot_path)
    logger.info(f"  本番時刻: {snapshot_ts}  特徴量数: {len(feat_dict)}")

    logger.info(f"S6データからランダム{n_s6_samples}件取得中...")
    feature_names = list(feat_dict.keys())
    s6_samples = load_s6_samples(feature_names, n_samples=n_s6_samples, include_raw=include_raw)

    if s6_samples is None or s6_samples.empty:
        logger.error("S6データが取得できませんでした。")
        return

    logger.info(f"  S6サンプル取得完了: {len(s6_samples)}件")

    # S6サンプルの統計量
    s6_stats = s6_samples.describe().T  # index=feature, columns=count/mean/std/min/25%/50%/75%/max

    records = []
    for feat_name, live_val in feat_dict.items():
        # neutralizedは常に対象、rawはinclude_raw=Trueかつe1X_で始まるもののみ
        is_neutralized = "_neutralized" in feat_name
        is_raw = (
            include_raw
            and re.match(r"e1[a-f]_", feat_name)
            and "_neutralized" not in feat_name
        )
        if not (is_neutralized or is_raw):
            continue
        if feat_name not in s6_stats.index:
            continue

        live_f  = float(live_val) if pd.notna(live_val) else 0.0
        s6_mean = float(s6_stats.loc[feat_name, "mean"])
        s6_std  = float(s6_stats.loc[feat_name, "std"])
        s6_min  = float(s6_stats.loc[feat_name, "min"])
        s6_max  = float(s6_stats.loc[feat_name, "max"])

        # zスコア：本番値がS6分布から何σ外れているか
        z_score = abs(live_f - s6_mean) / (s6_std + 1e-10)

        # min〜maxの範囲内か
        in_range = s6_min <= live_f <= s6_max

        # 判定（zスコアベース）
        if z_score <= 1.0:
            status = "✅ 正常"
        elif z_score <= 2.0:
            status = "🟡 1〜2σ"
        elif z_score <= 3.0:
            status = "🟠 2〜3σ"
        else:
            status = "🔴 3σ超"

        records.append({
            "feature":      feat_name,
            "feature_type": "neutralized" if "_neutralized" in feat_name else "raw",
            "module":       extract_module(feat_name),
            "timeframe":    extract_timeframe(feat_name),
            "live_val":  round(live_f, 6),
            "s6_mean":   round(s6_mean, 6),
            "s6_std":    round(s6_std, 6),
            "s6_min":    round(s6_min, 6),
            "s6_max":    round(s6_max, 6),
            "z_score":   round(z_score, 3),
            "in_range":  in_range,
            "status":    status,
        })

    if not records:
        logger.error("比較できる特徴量が0件でした。")
        return

    df = pd.DataFrame(records).sort_values("z_score", ascending=False)

    # CSV出力
    ts_str = snapshot_ts.strftime("%Y%m%d_%H%M%S")
    csv_path = OUTPUT_DIR / f"comparison_{ts_str}.csv"
    df.to_csv(csv_path, index=False)
    logger.info(f"比較CSV保存: {csv_path}")

    # 集計
    status_counts = df["status"].value_counts()
    n_total  = len(df)
    n_ok     = status_counts.get("✅ 正常",  0)
    n_1s     = status_counts.get("🟡 1〜2σ", 0)
    n_2s     = status_counts.get("🟠 2〜3σ", 0)
    n_3s     = status_counts.get("🔴 3σ超",  0)
    n_inrng  = df["in_range"].sum()

    module_summary = df.groupby("module")["z_score"].agg(["mean", "max", "count"])
    tf_summary     = df.groupby("timeframe")["z_score"].agg(["mean", "max", "count"])

    report_lines = [
        "=" * 70,
        f"  スナップショット比較レポート（zスコア方式）",
        f"  本番時刻 : {snapshot_ts}",
        f"  S6サンプル: {len(s6_samples)}件（ランダム）",
        f"  比較モード: {'neutralized + raw' if include_raw else 'neutralizedのみ'}",
        "=" * 70,
        f"  比較特徴量数: {n_total}件",
        f"  ✅ 正常 (z≤1) : {n_ok:4d}件 ({n_ok/n_total*100:.1f}%)",
        f"  🟡 1〜2σ      : {n_1s:4d}件 ({n_1s/n_total*100:.1f}%)",
        f"  🟠 2〜3σ      : {n_2s:4d}件 ({n_2s/n_total*100:.1f}%)",
        f"  🔴 3σ超       : {n_3s:4d}件 ({n_3s/n_total*100:.1f}%)",
        f"  S6範囲内(min〜max): {n_inrng}/{n_total}件 ({n_inrng/n_total*100:.1f}%)",
        "",
        "--- モジュール別 平均zスコア ---",
    ]
    for mod, row in module_summary.sort_values("mean", ascending=False).iterrows():
        report_lines.append(
            f"  {mod:6s}: mean={row['mean']:6.2f}  max={row['max']:8.2f}  ({int(row['count'])}件)"
        )

    report_lines += ["", "--- 時間足別 平均zスコア ---"]
    for tf, row in tf_summary.sort_values("mean", ascending=False).iterrows():
        report_lines.append(
            f"  {tf:6s}: mean={row['mean']:6.2f}  max={row['max']:8.2f}  ({int(row['count'])}件)"
        )

    report_lines += ["", f"--- zスコア高 TOP{top_n} ---"]
    for _, row in df.head(top_n).iterrows():
        report_lines.append(
            f"  {row['status']} [{row['module']}/{row['timeframe']}] "
            f"{row['feature'][:48]:<48} "
            f"live={row['live_val']:>10.4f}  "
            f"mean={row['s6_mean']:>10.4f}  std={row['s6_std']:>8.4f}  "
            f"z={row['z_score']:>7.2f}"
        )

    report_lines.append("=" * 70)
    report_text = "\n".join(report_lines)
    print("\n" + report_text)

    report_path = OUTPUT_DIR / f"summary_{ts_str}.txt"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report_text + "\n")
    logger.info(f"サマリー保存: {report_path}")


def main():
    parser = argparse.ArgumentParser(description="スナップショット vs S6 自動比較ツール")
    parser.add_argument("--snapshot_dir",  default=str(SNAPSHOT_DIR), type=Path)
    parser.add_argument("--snapshot_file", default=None, type=Path,
                        help="特定のCSVファイルを指定（省略時は最新）")
    parser.add_argument("--top_n",         default=50,  type=int)
    parser.add_argument("--n_s6_samples",  default=200, type=int,
                        help="S6からランダムに取得するサンプル数（多いほど統計精度向上）")
    parser.add_argument("--include_raw",   action="store_true",
                        help="純化前raw特徴量もS6と比較する（問題切り分け用）")
    args = parser.parse_args()

    if args.snapshot_file:
        target = args.snapshot_file
    else:
        snapshots = sorted(args.snapshot_dir.glob("snapshot_*.csv"))
        if not snapshots:
            logger.error(f"スナップショットが見つかりません: {args.snapshot_dir}")
            return
        target = snapshots[-1]
        logger.info(f"最新スナップショットを使用: {target.name}")

    compare(target, top_n=args.top_n, n_s6_samples=args.n_s6_samples, include_raw=args.include_raw)


if __name__ == "__main__":
    main()
