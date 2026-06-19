#!/usr/bin/env python3
"""
snapshot_training_inference_with_pre_ols.py — 案 X 用 学習側拡張 snapshot

[目的]
  既存 snapshot_training_inference.py の出力 (= S6 由来の純化後特徴量 + 推論結果) に、
  S2_FEATURES_VALIDATED の純化前特徴量を timestamp で left-join して追加出力する。

  これにより 1 つの parquet 内で 純化前 vs 純化後 を timestamp 1:1 で比較可能になり、
  compare スクリプトで:
    - 本番 base_features (純化前) vs S2 純化前 → engine_1_X 計算経路の検証
    - 本番 neutralized   (純化後) vs S6 純化後 → OLS 純化経路の検証
  という二段切り分けが可能になる。

[S2 構造]
  /workspace/data/XAUUSD/stratum_2_features_validated/
    feature_value_a_vast_universe{A,B,C,D,E,F}/   ← engine 別
      features_e1{a,b,c,d,e,f}_{TF}.parquet        ← TF 別 (全期間)
  
  列名規約: e1a_statistical_mean_10  (← TF サフィックス無し、 TF はファイル名にエンコード)
  本スクリプトでは TF サフィックスを付与して S6 純化後と対称な命名にする:
    S2 由来 (純化前):  e1a_statistical_mean_10_M3
    S6 由来 (純化後):  e1a_statistical_mean_10_neutralized_M3

[呼び出し例]
  python snapshot_training_inference_with_pre_ols.py \\
      --start 2026-05-25 --end 2026-05-25 \\
      --start-time 12:00:00 --end-time 13:30:00 \\
      --out /workspace/data/diagnostics/training_snapshot_with_pre_ols_20260525.parquet \\
      --m2-proba 0.70 --m2-delta 0.30 --min-atr 0.80
"""

from __future__ import annotations

import argparse
import logging
import re  # [LOOKAHEAD-FIX §11.34.16] TF 判定用
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

import joblib
import numpy as np
import polars as pl
from tqdm import tqdm

# Path setup
sys.path.append(str(Path(__file__).resolve().parents[1]))
import blueprint as config

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


# ════════════════════════════════════════════════════════════════
# 既存 snapshot_training_inference.py の中核ロジックを import
# (重複実装を避けるため、 既存スクリプトの関数を利用)
# ════════════════════════════════════════════════════════════════
# 同じ models/ ディレクトリ内なので import 可能
from snapshot_training_inference import (
    _FEATURE_EXCLUDE_EXACT,
    load_feature_list,
    discover_partitions,
    build_feature_matrix,
)


# ════════════════════════════════════════════════════════════════
# 新規: S2 純化前データ読み込み
# ════════════════════════════════════════════════════════════════
ENGINE_TO_UNIVERSE = {
    "e1a": "A",
    "e1b": "B",
    "e1c": "C",
    "e1d": "D",
    "e1e": "E",
    "e1f": "F",
}

# Phase 11 検証対象 TF (production がリアルタイム計算する TF)
TARGET_TFS = ["M0.5", "M1", "M3", "M5", "M8", "M15"]

# S2 parquet 内の meta カラム (特徴量ではない)
S2_META_COLS = {
    "timestamp", "open", "high", "low", "close",
    "volume", "timeframe", "sample_weight",
}


# [LOOKAHEAD-FIX §11.34.16 注記]
# snapshot は A 列 (本番特徴量計算経路の検証基準) を S2 から切り出す。S2 は
# label=left (ラベル L のバー [L, L+tf)) のため、高 TF (tf>180s) をそのまま使うと
# A 列に形成中バー値が入り、閉じバー化された B/C (cpl/main 修正後) と食い違う。
# cpl と同じ index シフト (各バーの値を次ラベルへ = 「ラベル L = L 時点で閉じた
# 最新バー」) を S2 読込時にも適用し、A/B/C を同じ時刻意味論に揃える。
ACTION_HORIZON_SEC_SNAP = 180  # M3 トリガの行動猶予


def _tf_to_seconds_snap(timeframe: str):
    m = re.fullmatch(r"M(\d+(?:\.\d+)?)", timeframe)
    if m:
        return int(float(m.group(1)) * 60)
    m = re.fullmatch(r"H(\d+(?:\.\d+)?)", timeframe)
    if m:
        return int(float(m.group(1)) * 3600)
    m = re.fullmatch(r"W(\d+)", timeframe)
    if m:
        return int(m.group(1)) * 604800
    if timeframe == "MN":
        return 28 * 86400
    return None


def _shift_high_tf_snap(df: "pl.DataFrame", timeframe: str) -> "pl.DataFrame":
    """tf > 180s の TF を閉じたバー基準に index シフト (cpl と同方式)。"""
    tf_sec = _tf_to_seconds_snap(timeframe)
    if tf_sec is None or tf_sec <= ACTION_HORIZON_SEC_SNAP:
        return df  # M0.5/M1/M3 はそのまま
    return (
        df.sort("timestamp")
        .with_columns(pl.col("timestamp").shift(-1).alias("timestamp"))
        .filter(pl.col("timestamp").is_not_null())
    )


def load_s2_pre_ols(
    start: datetime,
    end: datetime,
    s2_base: Path = None,
    target_tfs: List[str] = None,
    engine_map: Dict[str, str] = None,
) -> pl.DataFrame:
    """S2_FEATURES_VALIDATED から全 engine × 全 TF の純化前データを読み込み、
    timestamp で full-outer join する。

    返り値の列名規約:
      - timestamp (UTC datetime)
      - <feature_base_name>_<TF>  (例: e1a_statistical_mean_10_M3)

    特徴量列以外の meta (open/high/low/close 等) は除外。
    """
    if s2_base is None:
        s2_base = Path(config.S2_FEATURES_VALIDATED)
    if target_tfs is None:
        target_tfs = TARGET_TFS
    if engine_map is None:
        engine_map = ENGINE_TO_UNIVERSE

    result: Optional[pl.DataFrame] = None
    pieces_loaded = 0
    pieces_missing = []
    total_feature_cols = 0

    for engine, suffix in engine_map.items():
        for tf in target_tfs:
            path = s2_base / f"feature_value_a_vast_universe{suffix}" / f"features_{engine}_{tf}.parquet"
            if not path.exists():
                pieces_missing.append(str(path))
                continue

            df = pl.read_parquet(path)
            df = df.with_columns(
                pl.col("timestamp").cast(pl.Datetime("us", "UTC"))
            )
            df = df.filter(
                (pl.col("timestamp") >= start) & (pl.col("timestamp") <= end)
            )

            if len(df) == 0:
                pieces_missing.append(f"{path.name} (期間外)")
                continue

            # [LOOKAHEAD-FIX §11.34.16] 高TF (tf>180s) を閉じたバー基準に再ラベル
            # (cpl/main と同じ時刻意味論に A 列を揃える)
            df = _shift_high_tf_snap(df, tf)

            # 特徴量列のみ抽出 → TF サフィックス付与で rename
            feat_cols = [c for c in df.columns if c not in S2_META_COLS]
            df = df.select(["timestamp"] + feat_cols)
            df = df.rename({c: f"{c}_{tf}" for c in feat_cols})

            total_feature_cols += len(feat_cols)
            pieces_loaded += 1

            if result is None:
                result = df
            else:
                # timestamp で full-outer join (= 全 timestamps を保持)
                result = result.join(df, on="timestamp", how="full", coalesce=True)

    if result is None:
        logger.error("S2 から1枚も読み込めなかった")
        return pl.DataFrame()

    logger.info(
        f"  S2 純化前: {pieces_loaded}/{len(engine_map) * len(target_tfs)} pieces, "
        f"合計 {total_feature_cols} 特徴量列, {len(result)} timestamps"
    )
    if pieces_missing:
        logger.warning(f"  S2 読み込み失敗 / 期間外: {len(pieces_missing)} 件 (先頭3: {pieces_missing[:3]})")

    return result


# ════════════════════════════════════════════════════════════════
# main (既存 snapshot_training_inference.py の処理 + S2 join 追加)
# ════════════════════════════════════════════════════════════════
def main():
    parser = argparse.ArgumentParser(
        description="Training-side snapshot with pre-OLS data (案 X 用)"
    )
    parser.add_argument("--start", required=True, help="YYYY-MM-DD (UTC, inclusive)")
    parser.add_argument("--end",   required=True, help="YYYY-MM-DD (UTC, inclusive)")
    parser.add_argument("--start-time", default="00:00:00", help="HH:MM:SS UTC")
    parser.add_argument("--end-time",   default="23:59:59", help="HH:MM:SS UTC")
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--m2-proba", type=float, default=0.70)
    parser.add_argument("--m2-delta", type=float, default=0.30)
    parser.add_argument("--min-atr",  type=float, default=0.80)
    parser.add_argument("--source-dir", type=Path,
                        default=config.S6_WEIGHTED_DATASET)
    parser.add_argument("--s2-dir", type=Path,
                        default=Path(config.S2_FEATURES_VALIDATED))
    parser.add_argument("--all-rows", action="store_true",
                        help="is_trigger==0 の行も含める (default: 1 のみ)")
    parser.add_argument("--skip-s2", action="store_true",
                        help="S2 純化前データの追加を skip (デバッグ用)")
    args = parser.parse_args()

    start_date = datetime.strptime(
        f"{args.start} {args.start_time}", "%Y-%m-%d %H:%M:%S"
    ).replace(tzinfo=timezone.utc)
    end_date = datetime.strptime(
        f"{args.end} {args.end_time}", "%Y-%m-%d %H:%M:%S"
    ).replace(tzinfo=timezone.utc)
    args.out.parent.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 72)
    logger.info("Training-side Snapshot Generator (案 X: pre-OLS + post-OLS)")
    logger.info("=" * 72)
    logger.info(f"  期間:    {start_date} 〜 {end_date}")
    logger.info(f"  S6 入力: {args.source_dir}")
    logger.info(f"  S2 入力: {args.s2_dir}")
    logger.info(f"  出力:    {args.out}")
    logger.info(f"  filter:  m2_proba ≥ {args.m2_proba}, m2_delta ≥ {args.m2_delta}, atr_ratio ≥ {args.min_atr}")
    logger.info(f"  対象行:  {'全行' if args.all_rows else 'is_trigger==1 のみ'}")
    logger.info(f"  S2 追加: {'SKIP' if args.skip_s2 else 'ON'}")

    # ─── 1. S6 partitions 読み込み (既存ロジック) ────────────────
    partitions = discover_partitions(args.source_dir, start_date, end_date)
    if not partitions:
        raise FileNotFoundError(f"S6 期間内 partition 無: {args.source_dir}")
    logger.info(f"  S6 partition: {len(partitions)} 件")

    logger.info("--- S6 データ読み込み ---")
    dfs = [pl.read_parquet(p) for p in tqdm(partitions, desc="reading S6")]
    df = pl.concat(dfs, how="diagonal_relaxed")
    del dfs

    df = df.with_columns(pl.col("timestamp").cast(pl.Datetime("us", "UTC")))
    df = df.filter(
        (pl.col("timestamp") >= start_date) & (pl.col("timestamp") <= end_date)
    )

    if not args.all_rows and "is_trigger" in df.columns:
        n_before = len(df)
        df = df.filter(pl.col("is_trigger") == 1)
        logger.info(f"  is_trigger==1 filter: {n_before:,} → {len(df):,} 行")

    if len(df) == 0:
        logger.warning("対象行ゼロ、 終了")
        sys.exit(0)

    logger.info(f"  S6 読込完了: {len(df):,} 行 / {len(df.columns)} cols")

    # ─── 2. モデル & feature_lists ロード (既存ロジック) ────────
    logger.info("")
    logger.info("--- モデル & 特徴量リストロード ---")
    models = {
        "long_m1":  joblib.load(config.S7_M1_MODEL_LONG_PKL),
        "long_m2":  joblib.load(config.S7_M2_MODEL_LONG_PKL),
        "short_m1": joblib.load(config.S7_M1_MODEL_SHORT_PKL),
        "short_m2": joblib.load(config.S7_M2_MODEL_SHORT_PKL),
    }
    feature_lists = {
        "long_m1":  load_feature_list(config.S3_SELECTED_FEATURES_ORTHOGONAL_DIR / "m1_long_features.txt"),
        "long_m2":  load_feature_list(config.S3_SELECTED_FEATURES_ORTHOGONAL_DIR / "m2_long_features.txt"),
        "short_m1": load_feature_list(config.S3_SELECTED_FEATURES_ORTHOGONAL_DIR / "m1_short_features.txt"),
        "short_m2": load_feature_list(config.S3_SELECTED_FEATURES_ORTHOGONAL_DIR / "m2_short_features.txt"),
    }
    for key in ["long_m2", "short_m2"]:
        fl = feature_lists[key]
        if "m1_pred_proba" in fl:
            fl.remove("m1_pred_proba")
        fl.append("m1_pred_proba")
    logger.info(f"  モデル/特徴量リスト OK")

    # ─── 3. M1 推論 (既存ロジック) ──────────────────────────────
    logger.info("")
    logger.info("--- M1 推論 ---")
    X_long_m1  = build_feature_matrix(df, feature_lists["long_m1"],  "long_m1")
    X_short_m1 = build_feature_matrix(df, feature_lists["short_m1"], "short_m1")
    p_m1_long  = models["long_m1"].predict(X_long_m1)
    p_m1_short = models["short_m1"].predict(X_short_m1)
    logger.info(f"  M1_long  mean/max = {p_m1_long.mean():.4f}/{p_m1_long.max():.4f}")
    logger.info(f"  M1_short mean/max = {p_m1_short.mean():.4f}/{p_m1_short.max():.4f}")

    p_long_clip  = np.clip(p_m1_long,  1e-7, 1 - 1e-7)
    p_short_clip = np.clip(p_m1_short, 1e-7, 1 - 1e-7)
    logit_long   = np.clip(np.log(p_long_clip  / (1 - p_long_clip)),  -10.0, 10.0)
    logit_short  = np.clip(np.log(p_short_clip / (1 - p_short_clip)), -10.0, 10.0)

    # ─── 4. M2 推論 (既存ロジック) ──────────────────────────────
    logger.info("")
    logger.info("--- M2 推論 ---")
    df_for_long_m2  = df.with_columns(pl.Series("m1_pred_proba", logit_long))
    df_for_short_m2 = df.with_columns(pl.Series("m1_pred_proba", logit_short))
    X_long_m2  = build_feature_matrix(df_for_long_m2,  feature_lists["long_m2"],  "long_m2")
    X_short_m2 = build_feature_matrix(df_for_short_m2, feature_lists["short_m2"], "short_m2")
    p_m2_long  = models["long_m2"].predict(X_long_m2)
    p_m2_short = models["short_m2"].predict(X_short_m2)
    p_m2_long  = np.where(p_m1_long  < 0.50, 0.0, p_m2_long)
    p_m2_short = np.where(p_m1_short < 0.50, 0.0, p_m2_short)
    logger.info(f"  M2_long  mean/max = {p_m2_long.mean():.4f}/{p_m2_long.max():.4f}")
    logger.info(f"  M2_short mean/max = {p_m2_short.mean():.4f}/{p_m2_short.max():.4f}")

    # ─── 5. シグナル判定 (既存ロジック) ─────────────────────────
    delta = np.abs(p_m2_long - p_m2_short)
    atr_ratio = df["atr_ratio"].to_numpy() if "atr_ratio" in df.columns else np.zeros(len(df))
    passes_atr   = atr_ratio >= args.min_atr
    passes_delta = delta     >= args.m2_delta
    passes_proba_long  = p_m2_long  >= args.m2_proba
    passes_proba_short = p_m2_short >= args.m2_proba
    action = np.full(len(df), "HOLD", dtype=object)
    action[passes_atr & passes_delta & (p_m2_long > p_m2_short) & passes_proba_long]  = "BUY"
    action[passes_atr & passes_delta & (p_m2_short > p_m2_long) & passes_proba_short] = "SELL"

    n_buy  = int((action == "BUY").sum())
    n_sell = int((action == "SELL").sum())
    n_hold = int((action == "HOLD").sum())
    logger.info(f"  Signal: BUY={n_buy}, SELL={n_sell}, HOLD={n_hold}")

    # ─── 6. 推論結果列を追加 ────────────────────────────────────
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

    # ─── 7. 【新規】S2 純化前データを timestamp で left-join ────
    if not args.skip_s2:
        logger.info("")
        logger.info("--- 【新規】S2 純化前データ追加 ---")
        df_s2 = load_s2_pre_ols(start_date, end_date, args.s2_dir)
        if len(df_s2) > 0:
            n_cols_before = len(output_df.columns)

            # S6 と S2 で重複する meta カラムを回避するため、 S2 側の列を先に
            # フィルタしてから join。 ただし load_s2_pre_ols で meta は既に除外済。
            # 念のため、 timestamp 以外で重複する列を S2 側から削除:
            s2_dup_cols = [c for c in df_s2.columns
                          if c != "timestamp" and c in output_df.columns]
            if s2_dup_cols:
                logger.warning(f"  S2 と S6 で重複する列を S2 から除外: {len(s2_dup_cols)} 列 (先頭3: {s2_dup_cols[:3]})")
                df_s2 = df_s2.drop(s2_dup_cols)

            output_df = output_df.join(df_s2, on="timestamp", how="left")
            n_cols_after = len(output_df.columns)
            logger.info(
                f"  S2 join 完了: {n_cols_before} → {n_cols_after} cols "
                f"(+{n_cols_after - n_cols_before} 純化前列)"
            )

            # NaN 比率確認 (S2 と S6 の timestamp 不一致による NaN を検出)
            sample_s2_cols = [c for c in df_s2.columns
                             if c != "timestamp" and "_M" in c][:5]
            for c in sample_s2_cols:
                if c in output_df.columns:
                    n_null = output_df[c].null_count()
                    logger.info(f"    {c}: null = {n_null}/{len(output_df)}")
        else:
            logger.warning("  S2 データ無、 純化前 join スキップ")
    else:
        logger.info("  --skip-s2 指定により S2 追加 SKIP")

    # ─── 8. 書き出し ────────────────────────────────────────────
    output_df.write_parquet(args.out, compression="zstd")
    logger.info("")
    logger.info("=" * 72)
    logger.info(f"✅ 完了: {args.out}")
    logger.info(f"   rows={len(output_df):,}, cols={len(output_df.columns)}")
    logger.info("=" * 72)


if __name__ == "__main__":
    main()
