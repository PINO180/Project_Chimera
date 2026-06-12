#!/usr/bin/env python3
"""
snapshot_training_inference.py — 学習側パイプライン経由の Signal Snapshot 生成

[目的]
  Production triggered_features_log.csv との突合検証のため、 学習側パイプライン
  (S6_WEIGHTED_DATASET + 学習済モデル) を経由した M1/M2 推論結果と全特徴量を
  同一フォーマットで記録する。

[Production との 1:1 対応]
  main.py L1175-1402 と数値完全一致する推論経路:
    - dtype=np.float32 で特徴量配列構築 (DTYPE-ALIGN 段2)
    - M1 < 0.50 → M2 = 0.0 強制 (Bx2 / main.py 仕様)
    - m1_pred_proba は logit 変換 + [-10, +10] クリップ
    - シグナル判定: |delta| ≥ m2_delta AND winning_side ≥ m2_proba AND atr_ratio ≥ min_atr

[出力 parquet スキーマ]
  - timestamp (UTC datetime)
  - timeframe (str, 通常 "M3")
  - close (float, エントリー価格)
  - atr_value, atr_ratio (float)
  - p_m1_long_raw, p_m1_short_raw (float, raw)
  - p_m1_long_logit, p_m1_short_logit (float, Bx2 logit)
  - p_m2_long_raw, p_m2_short_raw (float, raw, M1<0.5 のとき 0.0)
  - delta (float)
  - passes_atr_filter, passes_delta_filter, passes_proba_filter (bool)
  - action (str: "BUY", "SELL", "HOLD")
  - [全 S6 特徴量列] (float64)

[呼び出し例]
  python snapshot_training_inference.py \\
      --start 2026-05-25 --end 2026-05-25 \\
      --start-time 21:00:00 --end-time 22:30:00 \\
      --out /workspace/data/diagnostics/training_snapshot_20260525.parquet \\
      --m2-proba 0.70 --m2-delta 0.30 --min-atr 0.80
"""

from __future__ import annotations

import argparse
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List

import joblib
import numpy as np
import polars as pl
from tqdm import tqdm

# Path setup ─ blueprint へアクセス
sys.path.append(str(Path(__file__).resolve().parents[1]))
import blueprint as config

# Logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


# ════════════════════════════════════════════════════════════════
# 1. Feature list 読み込み (infer_period.py L99-122 と一致)
# ════════════════════════════════════════════════════════════════
_FEATURE_EXCLUDE_EXACT = {
    "timestamp", "timeframe", "is_trigger",
    "label", "label_long", "label_short",
    "uniqueness", "uniqueness_long", "uniqueness_short",
    "duration_long", "duration_short",
    "concurrency_long", "concurrency_short",
    "payoff_ratio", "payoff_ratio_long", "payoff_ratio_short",
    "pt_multiplier", "sl_multiplier",
    "atr_value", "calculated_body_ratio", "fallback_vol",
    "open", "high", "low", "close",
    "m1_pred_proba", "meta_label",
    "t1", "direction", "exit_type", "first_ex_reason_int",
    "disc",
}


def load_feature_list(filepath: Path) -> List[str]:
    """Cx2._load_features / infer_period.py と同等のフィルタリング。"""
    with open(filepath, "r") as f:
        raw = [line.strip() for line in f if line.strip()]
    cleaned: List[str] = []
    for col in raw:
        if col in _FEATURE_EXCLUDE_EXACT:
            continue
        if col.startswith("is_trigger"):
            continue
        cleaned.append(col)
    return cleaned


# ════════════════════════════════════════════════════════════════
# 2. S6 partition discovery
# ════════════════════════════════════════════════════════════════
def discover_partitions(
    base_dir: Path,
    start_date: datetime,
    end_date: datetime,
) -> List[Path]:
    """S6_WEIGHTED_DATASET から指定期間の day partition を列挙。"""
    found: List[Path] = []
    for path in sorted(base_dir.glob("year=*/month=*/day=*/data.parquet")):
        try:
            y = int(path.parent.parent.parent.name.split("=")[1])
            m = int(path.parent.parent.name.split("=")[1])
            d = int(path.parent.name.split("=")[1])
        except (ValueError, IndexError):
            continue
        day_ts = datetime(y, m, d, tzinfo=timezone.utc)
        # 日付単位で含むかどうか判定 (時刻は後段で別途フィルタ)
        if start_date.date() <= day_ts.date() <= end_date.date():
            found.append(path)
    return found


# ════════════════════════════════════════════════════════════════
# 3. Feature matrix 構築 (main.py L1200-1208 と一致)
# ════════════════════════════════════════════════════════════════
def build_feature_matrix(
    df: pl.DataFrame,
    feature_list: List[str],
    name: str = "X",
) -> np.ndarray:
    """feature_list 順で float32 行列を構築。

    main.py の np.array([[feature_dict.get(f, 0.0) or 0.0 for f in feature_lists[K]]],
    dtype=np.float32) と数値完全一致するように構築する。
    """
    n_rows = len(df)
    cols_present = [c for c in feature_list if c in df.columns]
    cols_missing = [c for c in feature_list if c not in df.columns]
    if cols_missing:
        logger.warning(
            f"  [{name}] 特徴量欠損 {len(cols_missing)} 件 → 0.0 で代用 "
            f"(先頭5件: {cols_missing[:5]})"
        )

    # 必要列のみ抽出 → fill_null/fill_nan(0.0) → numpy
    if cols_present:
        df_subset = df.select(cols_present).fill_null(0.0).fill_nan(0.0)
        X_partial = df_subset.to_numpy().astype(np.float32)
    else:
        X_partial = np.zeros((n_rows, 0), dtype=np.float32)

    # feature_list 順で X を構築 (列順序の完全一致を保証)
    X = np.zeros((n_rows, len(feature_list)), dtype=np.float32)
    col_map = {c: i for i, c in enumerate(cols_present)}
    for j, feat in enumerate(feature_list):
        if feat in col_map:
            X[:, j] = X_partial[:, col_map[feat]]
        # else: 0.0 のまま
    return X


# ════════════════════════════════════════════════════════════════
# 4. main
# ════════════════════════════════════════════════════════════════
def main():
    parser = argparse.ArgumentParser(
        description="Training-side Signal Snapshot Generator"
    )
    parser.add_argument("--start", required=True, help="開始日 YYYY-MM-DD (UTC, inclusive)")
    parser.add_argument("--end",   required=True, help="終了日 YYYY-MM-DD (UTC, inclusive)")
    parser.add_argument("--start-time", default="00:00:00",
                        help="範囲開始時刻 HH:MM:SS (UTC). default: 00:00:00")
    parser.add_argument("--end-time",   default="23:59:59",
                        help="範囲終了時刻 HH:MM:SS (UTC). default: 23:59:59")
    parser.add_argument("--out", required=True, type=Path,
                        help="出力 parquet path")
    parser.add_argument("--m2-proba", type=float, default=0.70,
                        help="m2_proba_threshold (default: 0.70)")
    parser.add_argument("--m2-delta", type=float, default=0.30,
                        help="m2_delta_threshold (default: 0.30)")
    parser.add_argument("--min-atr",  type=float, default=0.80,
                        help="min_atr_threshold (default: 0.80)")
    parser.add_argument("--source-dir", type=Path,
                        default=config.S6_WEIGHTED_DATASET,
                        help=f"入力 partition dir (default: {config.S6_WEIGHTED_DATASET})")
    parser.add_argument("--all-rows", action="store_true",
                        help="is_trigger==0 の行も含める (default: 1 のみ)")
    args = parser.parse_args()

    start_date = datetime.strptime(
        f"{args.start} {args.start_time}", "%Y-%m-%d %H:%M:%S"
    ).replace(tzinfo=timezone.utc)
    end_date = datetime.strptime(
        f"{args.end} {args.end_time}", "%Y-%m-%d %H:%M:%S"
    ).replace(tzinfo=timezone.utc)

    if start_date > end_date:
        raise ValueError(f"start ({start_date}) > end ({end_date})")

    args.out.parent.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 72)
    logger.info("Training-side Signal Snapshot Generator")
    logger.info("=" * 72)
    logger.info(f"  期間:    {start_date} 〜 {end_date}")
    logger.info(f"  入力:    {args.source_dir}")
    logger.info(f"  出力:    {args.out}")
    logger.info(f"  filter:  m2_proba ≥ {args.m2_proba}, "
                f"m2_delta ≥ {args.m2_delta}, atr_ratio ≥ {args.min_atr}")
    logger.info(f"  対象行:  {'全行' if args.all_rows else 'is_trigger==1 のみ'}")

    # ─── 1. パーティション探索 ──────────────────────────────────
    partitions = discover_partitions(args.source_dir, start_date, end_date)
    if not partitions:
        raise FileNotFoundError(
            f"期間内 ({start_date.date()} 〜 {end_date.date()}) に partition なし: "
            f"{args.source_dir}"
        )
    logger.info(f"  検出 partition: {len(partitions)} 件")

    # ─── 2. データ読み込み + concat ─────────────────────────────
    logger.info("")
    logger.info("--- データ読み込み ---")
    dfs: List[pl.DataFrame] = []
    for p in tqdm(partitions, desc="reading partitions"):
        dfs.append(pl.read_parquet(p))
    df = pl.concat(dfs, how="diagonal_relaxed")
    del dfs

    # timestamp を UTC tz-aware に統一
    df = df.with_columns(
        pl.col("timestamp").cast(pl.Datetime("us", "UTC"))
    )

    # 時刻範囲フィルタ (start_time / end_time 適用)
    n_before_time = len(df)
    df = df.filter(
        (pl.col("timestamp") >= start_date) &
        (pl.col("timestamp") <= end_date)
    )
    logger.info(f"  時刻範囲フィルタ: {n_before_time:,} → {len(df):,} 行")

    # is_trigger == 1 のみ (default)
    if not args.all_rows and "is_trigger" in df.columns:
        n_before = len(df)
        df = df.filter(pl.col("is_trigger") == 1)
        logger.info(f"  is_trigger==1 でフィルタ: {n_before:,} → {len(df):,} 行")

    if len(df) == 0:
        logger.warning("対象行ゼロ。終了。")
        sys.exit(0)

    logger.info(f"  読み込み完了: {len(df):,} 行 / {len(df.columns)} cols")

    # ─── 3. モデル & 特徴量リストロード ──────────────────────────
    logger.info("")
    logger.info("--- モデル & 特徴量リストロード ---")
    models = {
        "long_m1":  joblib.load(config.S7_M1_MODEL_LONG_PKL),
        "long_m2":  joblib.load(config.S7_M2_MODEL_LONG_PKL),
        "short_m1": joblib.load(config.S7_M1_MODEL_SHORT_PKL),
        "short_m2": joblib.load(config.S7_M2_MODEL_SHORT_PKL),
    }
    feature_lists = {
        "long_m1":  load_feature_list(
            config.S3_SELECTED_FEATURES_ORTHOGONAL_DIR / "m1_long_features.txt"
        ),
        "long_m2":  load_feature_list(
            config.S3_SELECTED_FEATURES_ORTHOGONAL_DIR / "m2_long_features.txt"
        ),
        "short_m1": load_feature_list(
            config.S3_SELECTED_FEATURES_ORTHOGONAL_DIR / "m1_short_features.txt"
        ),
        "short_m2": load_feature_list(
            config.S3_SELECTED_FEATURES_ORTHOGONAL_DIR / "m2_short_features.txt"
        ),
    }
    # main.py / infer_period.py 同様、 m1_pred_proba を M2 末尾に強制配置
    for key in ["long_m2", "short_m2"]:
        fl = feature_lists[key]
        if "m1_pred_proba" in fl:
            fl.remove("m1_pred_proba")
        fl.append("m1_pred_proba")

    logger.info(
        f"  モデル 4 個ロード OK, "
        f"特徴量 M1={len(feature_lists['long_m1'])} / "
        f"M2={len(feature_lists['long_m2'])}"
    )

    # ─── 4. M1 推論 (Long, Short) ────────────────────────────────
    logger.info("")
    logger.info("--- M1 推論 ---")
    X_long_m1  = build_feature_matrix(df, feature_lists["long_m1"],  "long_m1")
    X_short_m1 = build_feature_matrix(df, feature_lists["short_m1"], "short_m1")
    p_m1_long  = models["long_m1"].predict(X_long_m1)
    p_m1_short = models["short_m1"].predict(X_short_m1)
    logger.info(
        f"  M1_long  : min={p_m1_long.min():.4f}, mean={p_m1_long.mean():.4f}, "
        f"max={p_m1_long.max():.4f}, ≥0.50: {(p_m1_long >= 0.50).sum():,}"
    )
    logger.info(
        f"  M1_short : min={p_m1_short.min():.4f}, mean={p_m1_short.mean():.4f}, "
        f"max={p_m1_short.max():.4f}, ≥0.50: {(p_m1_short >= 0.50).sum():,}"
    )

    # Bx2 logit 変換 (main.py L1215-1220 と一致)
    p_long_clip  = np.clip(p_m1_long,  1e-7, 1 - 1e-7)
    p_short_clip = np.clip(p_m1_short, 1e-7, 1 - 1e-7)
    logit_long   = np.clip(np.log(p_long_clip  / (1 - p_long_clip)),  -10.0, 10.0)
    logit_short  = np.clip(np.log(p_short_clip / (1 - p_short_clip)), -10.0, 10.0)

    # ─── 5. M2 推論 (Long, Short) ────────────────────────────────
    logger.info("")
    logger.info("--- M2 推論 ---")
    # m1_pred_proba 列を inject した DataFrame を Long/Short 別に作成
    df_for_long_m2  = df.with_columns(pl.Series("m1_pred_proba", logit_long))
    df_for_short_m2 = df.with_columns(pl.Series("m1_pred_proba", logit_short))

    X_long_m2  = build_feature_matrix(df_for_long_m2,  feature_lists["long_m2"],  "long_m2")
    X_short_m2 = build_feature_matrix(df_for_short_m2, feature_lists["short_m2"], "short_m2")
    p_m2_long  = models["long_m2"].predict(X_long_m2)
    p_m2_short = models["short_m2"].predict(X_short_m2)

    # M1 < 0.50 のとき M2 = 0.0 (Bx2 / main.py 仕様)
    n_low_long  = int((p_m1_long  < 0.50).sum())
    n_low_short = int((p_m1_short < 0.50).sum())
    p_m2_long  = np.where(p_m1_long  < 0.50, 0.0, p_m2_long)
    p_m2_short = np.where(p_m1_short < 0.50, 0.0, p_m2_short)

    logger.info(
        f"  M2_long  : min={p_m2_long.min():.4f}, mean={p_m2_long.mean():.4f}, "
        f"max={p_m2_long.max():.4f} (M1<0.5 → 0 強制: {n_low_long:,})"
    )
    logger.info(
        f"  M2_short : min={p_m2_short.min():.4f}, mean={p_m2_short.mean():.4f}, "
        f"max={p_m2_short.max():.4f} (M1<0.5 → 0 強制: {n_low_short:,})"
    )

    # ─── 6. シグナル判定 (BT 同等ロジック) ──────────────────────
    logger.info("")
    logger.info("--- シグナル判定 ---")
    delta = np.abs(p_m2_long - p_m2_short)

    # atr_ratio は S6 に格納済 (create_proxy_labels で計算)
    if "atr_ratio" in df.columns:
        atr_ratio = df["atr_ratio"].to_numpy()
    else:
        logger.warning("  atr_ratio 列が存在しない → 全て 0 とみなす (atr フィルタ全 reject)")
        atr_ratio = np.zeros(len(df))

    passes_atr   = atr_ratio >= args.min_atr
    passes_delta = delta     >= args.m2_delta
    passes_proba_long  = p_m2_long  >= args.m2_proba
    passes_proba_short = p_m2_short >= args.m2_proba

    action = np.full(len(df), "HOLD", dtype=object)
    # Long: p_long > p_short AND p_long > proba_threshold AND filter pass
    long_mask = (
        passes_atr & passes_delta
        & (p_m2_long > p_m2_short)
        & passes_proba_long
    )
    action[long_mask] = "BUY"
    short_mask = (
        passes_atr & passes_delta
        & (p_m2_short > p_m2_long)
        & passes_proba_short
    )
    action[short_mask] = "SELL"

    n_buy  = int((action == "BUY").sum())
    n_sell = int((action == "SELL").sum())
    n_hold = int((action == "HOLD").sum())
    logger.info(f"  Signal: BUY={n_buy}, SELL={n_sell}, HOLD={n_hold} (total {len(df):,})")

    # ─── 7. 出力 DataFrame 構築 ──────────────────────────────────
    # 推論結果列を追加
    output_df = df.with_columns([
        pl.Series("p_m1_long_raw",   p_m1_long),
        pl.Series("p_m1_short_raw",  p_m1_short),
        pl.Series("p_m1_long_logit", logit_long),
        pl.Series("p_m1_short_logit", logit_short),
        pl.Series("p_m2_long_raw",   p_m2_long),
        pl.Series("p_m2_short_raw",  p_m2_short),
        pl.Series("delta",           delta),
        pl.Series("passes_atr_filter",   passes_atr),
        pl.Series("passes_delta_filter", passes_delta),
        pl.Series("passes_proba_long",   passes_proba_long),
        pl.Series("passes_proba_short",  passes_proba_short),
        pl.Series("action", action.astype(str)),
    ])

    # parquet 書き出し
    output_df.write_parquet(args.out, compression="zstd")
    logger.info("")
    logger.info("=" * 72)
    logger.info(f"✅ 完了: {args.out}")
    logger.info(f"   rows={len(output_df):,}, cols={len(output_df.columns)}")
    logger.info("=" * 72)
    logger.info("")
    logger.info("次のステップ — 本番ライブログとの突合:")
    logger.info(f"  python compare_snapshots.py \\")
    logger.info(f"    --production /path/to/triggered_features_log.csv \\")
    logger.info(f"    --training {args.out} \\")
    logger.info(f"    --start {args.start} --end {args.end} \\")
    logger.info(f"    --start-time {args.start_time} --end-time {args.end_time} \\")
    logger.info(f"    --out-dir /workspace/data/diagnostics/compare_{args.start.replace('-','')}")


if __name__ == "__main__":
    main()
