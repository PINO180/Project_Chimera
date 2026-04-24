# /workspace/models/generate_layer1_training_data.py
# =====================================================================
# Layer 1 超軽量脳 学習データ生成スクリプト
#
# 【目的】
#   Pure Alphaシミュレーターが出力したトレードログ（UTC版CSV）から
#   Layer 1の学習データを生成する。
#
# 【入力】
#   detailed_trade_log_PureAlpha_UTC.csv
#   （backtest_simulator_cimera_pure_alpha.py の出力）
#
# 【特徴量（3変数のみ）】
#   atr_ratio : 相対的なボラティリティ水準
#   hour_utc  : UTC時間帯（0〜23）← timestampから計算（UTCのまま）
#   weekday   : 曜日（UTC基準・0=月〜6=日）
#
# 【ターゲット変数】
#   is_bad_trade = 1 if pnl < 0 else 0
#   （SL・タイムアウト負け両方を「悪いトレード」として統合）
#
# 【出力】
#   LAYER1_TRAINING_DATA_DIR/
#     layer1_train_long.parquet
#     layer1_train_short.parquet
#     summary.txt
#
# 【注意】
#   timestampはUTC版CSVを使用すること。
#   JST版CSVを使うとhour_utcが9時間ずれる。
#
# 【使い方】
#   # 最新のUTC版CSVを自動検出
#   python generate_layer1_training_data.py
#
#   # パスを直接指定
#   python generate_layer1_training_data.py --csv_path /path/to/UTC.csv
# =====================================================================

import sys
import argparse
import logging
from pathlib import Path

import polars as pl

project_root = Path(__file__).resolve().parents[1]
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

from blueprint import (
    S7_BACKTEST_SIM_RESULTS,
    LAYER1_TRAINING_DATA_DIR,
)

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("GenLayer1")

K_FOLDS = 5


# =====================================================================
# ユーティリティ
# =====================================================================

def find_latest_utc_csv(base_dir: Path) -> Path:
    """最新のPure Alpha UTC版トレードログCSVを自動検出する"""
    files = sorted(
        base_dir.glob("**/detailed_trade_log_PureAlpha_UTC.csv"),
        key=lambda p: p.stat().st_mtime,
    )
    if not files:
        raise FileNotFoundError(
            f"UTC版トレードログが見つかりません: {base_dir}\n"
            f"先に backtest_simulator_cimera_pure_alpha.py を実行してください。"
        )
    latest = files[-1]
    logger.info(f"UTC版トレードログ検出: {latest}")
    return latest


def assign_wf_fold(df: pl.DataFrame, n_splits: int = 5) -> pl.DataFrame:
    """
    タイムスタンプ順でWalk-Forward用のfold_indexを割り当てる。
    M3と同じ4ステップ構造:
      wf_step=1: fold0で学習 → fold1を検証
      wf_step=2: fold0+1で学習 → fold2を検証
      wf_step=3: fold0+1+2で学習 → fold3を検証
      wf_step=4: fold0+1+2+3で学習 → fold4を検証（最新期間）
    """
    n = len(df)
    fsz = n // n_splits
    indices = []
    for i in range(n_splits):
        start = i * fsz
        end = start + fsz if i < n_splits - 1 else n
        indices.extend([i] * (end - start))
    return df.with_columns(pl.Series("fold_index", indices[:n]))


# =====================================================================
# メイン処理
# =====================================================================

def generate(csv_path: Path) -> None:
    LAYER1_TRAINING_DATA_DIR.mkdir(parents=True, exist_ok=True)

    logger.info(f"トレードログ読み込み: {csv_path}")
    df = pl.read_csv(csv_path, null_values=["NaN", "nan", ""])
    logger.info(f"総トレード数: {len(df)}")

    # timestamp を datetime に変換（UTC版CSVはUTCのまま）
    if df["timestamp"].dtype == pl.Utf8:
        df = df.with_columns(
            pl.col("timestamp")
            .str.to_datetime("%Y-%m-%d %H:%M:%S", strict=False)
            .alias("timestamp")
        )

    # --- 必須カラム確認 ---
    for col in ["pnl", "atr_ratio", "direction"]:
        if col not in df.columns:
            raise ValueError(
                f"'{col}'カラムが見つかりません。UTC版CSVを確認してください。"
            )

    # --- 特徴量計算（UTC基準・本番MT5と一致）---
    df = df.with_columns([
        pl.col("timestamp").dt.hour().alias("hour_utc"),    # UTC時間帯（変換不要）
        pl.col("timestamp").dt.weekday().alias("weekday"),  # UTC基準の曜日（0=月）
    ])

    # --- ターゲット変数 ---
    df = df.with_columns(
        pl.when(pl.col("pnl") < 0)
        .then(pl.lit(1))
        .otherwise(pl.lit(0))
        .cast(pl.Int8)
        .alias("is_bad_trade")
    )

    # --- Long / Short 分割 ---
    df_long  = df.filter(pl.col("direction") == 1).sort("timestamp")
    df_short = df.filter(pl.col("direction") == -1).sort("timestamp")
    logger.info(f"Long: {len(df_long)}件 / Short: {len(df_short)}件")

    feature_cols = [
        "timestamp", "fold_index",
        "atr_ratio", "hour_utc", "weekday",
        "is_bad_trade", "pnl", "direction",
    ]

    summary_lines = [
        "=" * 60,
        "  Layer 1 学習データ生成サマリー",
        "=" * 60,
        f"  入力CSV : {csv_path.name}",
        f"  特徴量  : atr_ratio / hour_utc / weekday",
        f"  時刻基準: UTC（MT5・本番と一致）",
        f"  ターゲット: is_bad_trade = 1 if pnl < 0 else 0",
        "",
    ]

    for direction, df_dir in [("long", df_long), ("short", df_short)]:
        logger.info(f"\n--- {direction.upper()} ---")

        # WF fold割り当て
        df_dir = assign_wf_fold(df_dir, n_splits=K_FOLDS)

        # 必要カラムのみ選択
        available = [c for c in feature_cols if c in df_dir.columns]
        df_out = df_dir.select(available)

        # 統計
        n_total  = len(df_out)
        n_bad    = int(df_out["is_bad_trade"].sum())
        bad_rate = n_bad / n_total * 100

        fold_stats = df_out.group_by("fold_index").agg([
            pl.len().alias("count"),
            pl.col("is_bad_trade").mean().alias("bad_rate"),
            pl.col("atr_ratio").mean().alias("avg_atr"),
        ]).sort("fold_index")

        logger.info(f"総レコード: {n_total}件")
        logger.info(f"is_bad_trade=1: {n_bad}件 ({bad_rate:.1f}%)")

        # 保存
        out_path = LAYER1_TRAINING_DATA_DIR / f"layer1_train_{direction}.parquet"
        df_out.write_parquet(out_path, compression="zstd")
        logger.info(f"保存: {out_path}")

        # サマリー
        summary_lines += [
            f"--- {direction.upper()} ---",
            f"  総レコード数      : {n_total}件",
            f"  is_bad_trade=1率 : {bad_rate:.2f}%",
            "",
            f"  {'Fold':>5} {'件数':>8} {'負け率%':>8} {'avg_atr_ratio':>14}",
            "  " + "-" * 42,
        ]
        for row in fold_stats.iter_rows(named=True):
            summary_lines.append(
                f"  Fold{row['fold_index']:>1}  "
                f"{row['count']:>8}  "
                f"{row['bad_rate']*100:>7.2f}%  "
                f"{row['avg_atr']:>14.4f}"
            )
        summary_lines.append("")

    summary_lines += [
        "--- 次のステップ ---",
        "  train_layer1_filter.py でLayer 1モデルを学習する",
        "  WF Step1〜3で学習・WF Step4（最新期間）でテスト",
        "=" * 60,
    ]

    report = "\n".join(summary_lines)
    print("\n" + report)

    txt_path = LAYER1_TRAINING_DATA_DIR / "summary.txt"
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(report + "\n")
    logger.info(f"サマリー保存: {txt_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Layer 1 超軽量脳 学習データ生成"
    )
    parser.add_argument(
        "--csv_path", type=str, default=None,
        help="UTC版トレードログCSVのパス（未指定時は最新を自動検出）",
    )
    args = parser.parse_args()

    if args.csv_path:
        csv_path = Path(args.csv_path)
        if not csv_path.exists():
            raise FileNotFoundError(f"指定されたCSVが見つかりません: {csv_path}")
    else:
        csv_path = find_latest_utc_csv(S7_BACKTEST_SIM_RESULTS)

    generate(csv_path)


if __name__ == "__main__":
    main()
