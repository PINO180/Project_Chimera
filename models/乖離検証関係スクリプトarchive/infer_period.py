#!/usr/bin/env python3
"""
infer_period.py — Cimera V5 推論専用スクリプト

[目的]
  既存の学習済み AI (m1_model_v2_*.pkl, m2_model_v2_*.pkl) を再学習せずに、
  指定した期間 (例: 直近 1〜数日) の S6_WEIGHTED_DATASET を入力にして
  M1 → M2 の推論を実行し、BT 互換の OOF parquet を生成する。

[背景 — 本番ライブとの突き合わせ検証]
  本番ライブと BT の乖離原因を切り分けるため、「同じ期間を BT で再現
  する」 ためのデータを最小コストで作る。Ax2/Bx2/Cx2 は学習を伴うので
  3 日分のみのデータでは Purged K-Fold が成立しない。
  → 学習は完全にスキップし、既存モデルで predict() だけ回す経路を別途用意する。

[出力 OOF parquet スキーマ]
  BT (backtest_simulator_cimera.py L484-490) と整合:
    timestamp (UTC), timeframe, prediction (M2 raw proba),
    true_label (= label_long/label_short のコピー), uniqueness

[呼び出し例]
  python infer_period.py \
      --start 2026-05-20 --end 2026-05-21 \
      --out-dir /workspace/data/XAUUSD/stratum_7_models/infer_1day_20260520

  → 出力 dir に以下を生成:
      m1_oof_predictions_long.parquet
      m1_oof_predictions_short.parquet
      m2_oof_predictions_long.parquet
      m2_oof_predictions_short.parquet

[本番ライブとの一致仕様 — main.py L1175-1245 と完全に同じ]
  1. LightGBM .predict() (calibrator なし — BT も calibrator は使わず raw を見る)
  2. M2 入力 dtype は np.float32 (DTYPE-ALIGN 段2)
  3. M1 < 0.50 → M2=0.0 (Bx2 設計通り)
  4. m1_pred_proba は logit 変換 + [-10, +10] clip
  5. 特徴量リストは S3_SELECTED_FEATURES_ORTHOGONAL_DIR/*.txt (直交分割版)
  6. _FEATURE_EXCLUDE_EXACT で除外 (main.py L519-547 と同一セット)
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

# ─── プロジェクトルートを sys.path に追加 ──────────────────────
sys.path.append(str(Path(__file__).resolve().parents[1]))
import blueprint as config  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("infer_period")


# ════════════════════════════════════════════════════════════════
# 1. 特徴量リスト処理 (main.py L519-572 と完全に同一)
# ════════════════════════════════════════════════════════════════
_FEATURE_EXCLUDE_EXACT = {
    "timestamp",
    "timeframe",
    "t1",
    "label",
    "label_long",
    "label_short",
    "uniqueness",
    "uniqueness_long",
    "uniqueness_short",
    "payoff_ratio",
    "payoff_ratio_long",
    "payoff_ratio_short",
    "pt_multiplier",
    "sl_multiplier",
    "direction",
    "exit_type",
    "first_ex_reason_int",
    "atr_value",
    "calculated_body_ratio",
    "fallback_vol",
    "open",
    "high",
    "low",
    "close",
    "meta_label",
    "m1_pred_proba",
    "is_trigger",
}


def load_feature_list(filepath: Path) -> List[str]:
    """学習側 Cx2._load_features と同等のフィルタリング。

    除外対象:
      - _FEATURE_EXCLUDE_EXACT セット
      - is_trigger プレフィックスを持つ列 (is_trigger_on_M1 等)
    """
    with open(filepath, "r") as f:
        raw = [line.strip() for line in f if line.strip()]
    cleaned: List[str] = []
    dropped: List[str] = []
    for col in raw:
        if col in _FEATURE_EXCLUDE_EXACT:
            dropped.append(col)
            continue
        if col.startswith("is_trigger"):
            dropped.append(col)
            continue
        cleaned.append(col)
    if dropped:
        logger.warning(
            f"⚠️ {filepath.name}: 学習側で除外される列が混入していたため除外: {dropped}"
        )
    return cleaned


# ════════════════════════════════════════════════════════════════
# 2. S6 partition discovery
# ════════════════════════════════════════════════════════════════
def discover_partitions(
    base_dir: Path,
    start_date: datetime,
    end_date: datetime,
) -> List[Path]:
    """S6_WEIGHTED_DATASET から指定期間の day partition を列挙。

    期待構造:
      base_dir/year={Y}/month={M}/day={D}/data.parquet
    """
    found: List[Path] = []
    for path in sorted(base_dir.glob("year=*/month=*/day=*/data.parquet")):
        try:
            y = int(path.parent.parent.parent.name.split("=")[1])
            m = int(path.parent.parent.name.split("=")[1])
            d = int(path.parent.name.split("=")[1])
        except (ValueError, IndexError):
            continue
        day_ts = datetime(y, m, d, tzinfo=timezone.utc)
        if start_date <= day_ts <= end_date:
            found.append(path)
    return found


# ════════════════════════════════════════════════════════════════
# 3. 推論コア (main.py L1175-1245 と byte-identical な処理順序)
# ════════════════════════════════════════════════════════════════
def run_inference_for_direction(
    df: pl.DataFrame,
    direction: str,  # "long" or "short"
    models: Dict[str, object],
    feature_lists: Dict[str, List[str]],
) -> pl.DataFrame:
    """1 方向ぶんの M1→M2 推論を一括実行。

    主要設計:
      - main.py の推論経路と数値完全一致 (dtype=np.float32 で配列構築)
      - M1 < 0.50 のサンプルは M2=0.0 (Bx2 設計と同じ)
      - m1_pred_proba は logit 変換 + [-10, +10] clip

    Returns:
      DataFrame with columns:
        timestamp, timeframe, m1_pred_proba (raw), m2_pred_proba (raw),
        m1_logit (Bx2 logit 変換後)
    """
    m1_key = f"{direction}_m1"
    m2_key = f"{direction}_m2"
    m1_features = feature_lists[m1_key]
    m2_features = feature_lists[m2_key]  # 末尾は m1_pred_proba

    n_rows = len(df)
    logger.info(
        f"  [{direction.upper()}] 推論開始: rows={n_rows:,}, "
        f"m1_features={len(m1_features)}, m2_features={len(m2_features)}"
    )

    # ─── M1 入力ベクトル構築 (Float32) ─────────────────────────
    # df 内に欠ける特徴量カラムは 0.0 でフォールバック (main.py L1183 と一致)
    m1_cols_present = [c for c in m1_features if c in df.columns]
    m1_cols_missing = [c for c in m1_features if c not in df.columns]
    if m1_cols_missing:
        logger.warning(
            f"  [{direction.upper()}] M1 特徴量欠損 {len(m1_cols_missing)} 件 → 0.0 で代用: "
            f"先頭5件={m1_cols_missing[:5]}"
        )

    # df を polars で必要列のみ抽出 → numpy (Float32)
    if m1_cols_present:
        df_m1 = df.select(m1_cols_present).fill_null(0.0).fill_nan(0.0)
        X_m1_partial = df_m1.to_numpy().astype(np.float32)
    else:
        X_m1_partial = np.zeros((n_rows, 0), dtype=np.float32)

    # 欠損列を 0.0 列で埋め、最終的に m1_features の順序を再現
    if m1_cols_missing:
        # 列順序を main.py と一致させるため、index ベースで構築
        X_m1 = np.zeros((n_rows, len(m1_features)), dtype=np.float32)
        col_map = {c: i for i, c in enumerate(m1_cols_present)}
        for j, feat in enumerate(m1_features):
            if feat in col_map:
                X_m1[:, j] = X_m1_partial[:, col_map[feat]]
            # else: 0.0 のまま
    else:
        # 順序を完全に揃え直す (df の列順序が m1_features と異なる場合に備えて)
        X_m1 = np.zeros((n_rows, len(m1_features)), dtype=np.float32)
        col_map = {c: i for i, c in enumerate(m1_cols_present)}
        for j, feat in enumerate(m1_features):
            X_m1[:, j] = X_m1_partial[:, col_map[feat]]

    # ─── M1 推論 ──────────────────────────────────────────────
    p_m1_raw = models[m1_key].predict(X_m1)
    logger.info(
        f"  [{direction.upper()}] M1 推論完了: min={p_m1_raw.min():.4f}, "
        f"max={p_m1_raw.max():.4f}, mean={p_m1_raw.mean():.4f}, "
        f"≥0.50: {(p_m1_raw >= 0.50).sum():,} / {n_rows:,}"
    )

    # ─── Bx2 logit 変換 (main.py L1192-1197 と一致) ────────────
    # proba → logit。p<0.50 でもベクトル化のため全行計算 → 後で M1<0.50 は M2=0 に
    p_clipped = np.clip(p_m1_raw, 1e-7, 1 - 1e-7)
    logits = np.log(p_clipped / (1 - p_clipped))
    logits = np.clip(logits, -10.0, 10.0)

    # ─── M2 入力ベクトル構築 ──────────────────────────────────
    # m2_features の末尾は m1_pred_proba (= logits)
    # それ以外は M1 と同じく df から抽出 (Float32)
    m2_cols_no_logit = [c for c in m2_features if c != "m1_pred_proba"]
    m2_cols_present = [c for c in m2_cols_no_logit if c in df.columns]
    m2_cols_missing = [c for c in m2_cols_no_logit if c not in df.columns]
    if m2_cols_missing:
        logger.warning(
            f"  [{direction.upper()}] M2 特徴量欠損 {len(m2_cols_missing)} 件 → 0.0 で代用: "
            f"先頭5件={m2_cols_missing[:5]}"
        )

    if m2_cols_present:
        df_m2 = df.select(m2_cols_present).fill_null(0.0).fill_nan(0.0)
        X_m2_partial = df_m2.to_numpy().astype(np.float32)
    else:
        X_m2_partial = np.zeros((n_rows, 0), dtype=np.float32)

    # main.py 同様 m1_pred_proba を末尾に強制配置 (load_feature_list 後の処理)
    X_m2 = np.zeros((n_rows, len(m2_features)), dtype=np.float32)
    col_map_m2 = {c: i for i, c in enumerate(m2_cols_present)}
    for j, feat in enumerate(m2_features):
        if feat == "m1_pred_proba":
            X_m2[:, j] = logits.astype(np.float32)
        elif feat in col_map_m2:
            X_m2[:, j] = X_m2_partial[:, col_map_m2[feat]]
        # else: 0.0

    # ─── M2 推論 (全行ベクトル化) ─────────────────────────────
    p_m2_raw = models[m2_key].predict(X_m2)

    # ─── M1 < 0.50 の行は M2 = 0.0 に強制 (Bx2 / main.py 仕様) ──
    mask_low_m1 = p_m1_raw < 0.50
    p_m2_raw = np.where(mask_low_m1, 0.0, p_m2_raw)

    logger.info(
        f"  [{direction.upper()}] M2 推論完了: min={p_m2_raw.min():.4f}, "
        f"max={p_m2_raw.max():.4f}, mean={p_m2_raw.mean():.4f}, "
        f"M2=0.0 (M1<0.50 経由): {mask_low_m1.sum():,} / {n_rows:,}"
    )

    # ─── 結果 DataFrame 構築 ──────────────────────────────────
    return pl.DataFrame(
        {
            "timestamp": df["timestamp"],
            "timeframe": df["timeframe"],
            "m1_pred_proba_raw": p_m1_raw,
            "m1_logit": logits,
            "m2_pred_proba_raw": p_m2_raw,
        }
    )


# ════════════════════════════════════════════════════════════════
# 4. BT 互換 OOF parquet の書き出し
# ════════════════════════════════════════════════════════════════
def write_bt_compatible_oof(
    df_pred: pl.DataFrame,
    df_source: pl.DataFrame,
    direction: str,
    layer: str,  # "m1" or "m2"
    out_dir: Path,
) -> Path:
    """BT が読める OOF parquet 形式に変換して書き出す。

    BT 必須スキーマ (backtest_simulator_cimera.py L484-490):
        timestamp, timeframe, prediction, true_label, uniqueness

    Args:
        df_pred: run_inference_for_direction の出力
        df_source: S6_WEIGHTED_DATASET から読んだ元 DataFrame
                   (label_long/label_short/uniqueness_long/uniqueness_short を持つ)
        direction: "long" or "short"
        layer: "m1" or "m2"
        out_dir: 出力 dir
    """
    pred_col = "m1_pred_proba_raw" if layer == "m1" else "m2_pred_proba_raw"
    label_col = f"label_{direction}"
    uniq_col = f"uniqueness_{direction}"

    # df_source の label/uniqueness を timestamp+timeframe で join
    join_cols = ["timestamp", "timeframe"]
    source_subset = df_source.select(join_cols + [label_col, uniq_col])

    joined = df_pred.join(source_subset, on=join_cols, how="left")

    out_df = joined.select(
        [
            pl.col("timestamp"),
            pl.col("timeframe"),
            pl.col(pred_col).alias("prediction"),
            pl.col(label_col).alias("true_label"),
            pl.col(uniq_col).alias("uniqueness"),
        ]
    )

    out_path = out_dir / f"{layer}_oof_predictions_{direction}.parquet"
    out_df.write_parquet(out_path, compression="zstd")
    logger.info(
        f"  ✓ 書き出し: {out_path.name} "
        f"(rows={len(out_df):,}, "
        f"prediction min/mean/max = "
        f"{out_df['prediction'].min():.4f}/"
        f"{out_df['prediction'].mean():.4f}/"
        f"{out_df['prediction'].max():.4f})"
    )
    return out_path


# ════════════════════════════════════════════════════════════════
# 5. main
# ════════════════════════════════════════════════════════════════
def main():
    parser = argparse.ArgumentParser(
        description="Cimera V5 推論専用スクリプト (1〜数日範囲)"
    )
    parser.add_argument(
        "--start", required=True,
        help="開始日 (YYYY-MM-DD, UTC, inclusive)"
    )
    parser.add_argument(
        "--end", required=True,
        help="終了日 (YYYY-MM-DD, UTC, inclusive)"
    )
    parser.add_argument(
        "--out-dir", required=True, type=Path,
        help="出力 dir (4 つの parquet を生成)"
    )
    parser.add_argument(
        "--source-dir", type=Path, default=config.S6_WEIGHTED_DATASET,
        help=f"入力 partition dir (default: {config.S6_WEIGHTED_DATASET})"
    )
    args = parser.parse_args()

    start_date = datetime.strptime(args.start, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    end_date = datetime.strptime(args.end, "%Y-%m-%d").replace(tzinfo=timezone.utc)

    if start_date > end_date:
        raise ValueError(f"start ({start_date}) > end ({end_date})")

    args.out_dir.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 70)
    logger.info("Cimera V5 推論専用スクリプト")
    logger.info("=" * 70)
    logger.info(f"  期間: {start_date.date()} 〜 {end_date.date()}")
    logger.info(f"  入力: {args.source_dir}")
    logger.info(f"  出力: {args.out_dir}")

    # ─── 1. パーティション探索 ────────────────────────────────
    partitions = discover_partitions(args.source_dir, start_date, end_date)
    if not partitions:
        raise FileNotFoundError(
            f"期間内 ({start_date.date()} 〜 {end_date.date()}) に partition なし: {args.source_dir}"
        )
    logger.info(f"  検出 partition: {len(partitions)} 件")

    # ─── 2. データ読み込み + concat ───────────────────────────
    logger.info("")
    logger.info("--- データ読み込み ---")
    dfs = []
    for p in tqdm(partitions, desc="reading partitions"):
        dfs.append(pl.read_parquet(p))
    df = pl.concat(dfs, how="diagonal_relaxed")

    # timestamp を UTC tz-aware に統一 (BT との一致のため)
    df = df.with_columns(
        pl.col("timestamp").cast(pl.Datetime("us", "UTC"))
    )

    # is_trigger==1 の行だけ残す (BT もシグナル発火行のみ評価)
    if "is_trigger" in df.columns:
        n_before = len(df)
        df = df.filter(pl.col("is_trigger") == 1)
        logger.info(
            f"  is_trigger==1 でフィルタ: {n_before:,} → {len(df):,} 行"
        )

    logger.info(
        f"  読み込み完了: {len(df):,} 行 / {len(df.columns)} cols"
    )

    # ─── 3. モデル & 特徴量リストロード ───────────────────────
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

    # main.py L591-595 と同じく m1_pred_proba を M2 リストの末尾に強制配置
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

    # ─── 4. Long/Short 推論 ────────────────────────────────────
    logger.info("")
    logger.info("--- Long 推論 ---")
    df_long = run_inference_for_direction(df, "long", models, feature_lists)

    logger.info("")
    logger.info("--- Short 推論 ---")
    df_short = run_inference_for_direction(df, "short", models, feature_lists)

    # ─── 5. BT 互換 OOF 書き出し ───────────────────────────────
    logger.info("")
    logger.info("--- BT 互換 OOF parquet 書き出し ---")
    write_bt_compatible_oof(df_long,  df, "long",  "m1", args.out_dir)
    write_bt_compatible_oof(df_long,  df, "long",  "m2", args.out_dir)
    write_bt_compatible_oof(df_short, df, "short", "m1", args.out_dir)
    write_bt_compatible_oof(df_short, df, "short", "m2", args.out_dir)

    logger.info("")
    logger.info("=" * 70)
    logger.info("✅ 推論完了")
    logger.info("=" * 70)
    logger.info("")
    logger.info("次のステップ — BT で同期間を再現:")
    logger.info(f"  python backtest_simulator_cimera.py \\")
    logger.info(f"      --oof-long-path  {args.out_dir / 'm2_oof_predictions_long.parquet'} \\")
    logger.info(f"      --oof-short-path {args.out_dir / 'm2_oof_predictions_short.parquet'} \\")
    logger.info(f"      (その他 BT パラメータ)")
    logger.info("")
    logger.info("本番ライブログとの比較:")
    logger.info(f"  - 本番ライブの 'Raw Proba M1(L:X, S:Y) -> M2(L:Z, S:W)' を")
    logger.info(f"    本スクリプト出力の同 timestamp 行と突き合わせる")
    logger.info(f"  - 一致なら -> 推論レベル乖離なし (本番側の特徴量計算は学習側と一致)")
    logger.info(f"  - 不一致なら -> 特徴量レベルで既に乖離 (engine_1_A〜1_F / 純化経路を疑う)")


if __name__ == "__main__":
    main()
