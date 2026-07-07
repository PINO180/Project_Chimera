# /workspace/models/sample_uniqueness_weighting_join.py

import sys
import polars as pl
from pathlib import Path
import logging
import shutil
from typing import List

# プロジェクトのルートディレクトリをPythonの検索パスに追加
sys.path.append(str(Path(__file__).resolve().parents[1]))

# blueprintから一元管理された設定を読み込む
from blueprint import (
    S6_LABELED_DATASET,
    S6_WEIGHTED_DATASET,
    S3_CONCURRENCY_RESULTS,
)

# --- 定数 ---
CONCURRENCY_RESULTS_PATH = S3_CONCURRENCY_RESULTS

# --- ロギング設定 ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(processName)s - %(message)s",
)


def get_partitions(input_dir: Path) -> List[Path]:
    """日次パーティションのファイルパス一覧を取得する。"""
    logging.info("Step 1: Finding daily partitions...")
    partitions = sorted(input_dir.glob("year=*/month=*/day=*/*.parquet"))
    if not partitions:
        raise FileNotFoundError(
            f"No daily parquet files found in the expected Hive structure within {input_dir}"
        )

    logging.info(f"   -> Found {len(partitions)} daily partitions.")
    return partitions


def main():
    """メインオーケストレーション関数（日次・逐次処理版・V5仕様）"""
    logging.info("### Final Battle Stage 2: Daily Sequential Assembly (V5) ###")

    if not CONCURRENCY_RESULTS_PATH.exists():
        logging.error(
            f"CRITICAL: Concurrency results not found at {CONCURRENCY_RESULTS_PATH}"
        )
        return

    # --- 準備：出力ディレクトリのクリーンアップと作成 ---
    if S6_WEIGHTED_DATASET.exists():
        logging.warning(
            f"Output directory {S6_WEIGHTED_DATASET} exists. Removing it for a clean run."
        )
        shutil.rmtree(S6_WEIGHTED_DATASET)
    S6_WEIGHTED_DATASET.mkdir(parents=True)

    # --- ステージ1: パーティション情報の取得 ---
    partitions = get_partitions(S6_LABELED_DATASET)

    # --- ステージ2: 逐次処理ループ ---
    logging.info(
        f"Stage 2: Starting sequential processing of {len(partitions)} daily partitions."
    )

    error_count = 0
    total_processed_rows = 0

    # [TAKEHOME §36 / 案B] 手取り(pt−d, ATR単位)で正例に濃淡をつける最終重み w_final を別列で作る。
    #   uniqueness(=1/concurrency) は純粋なまま保持し、w_final = uniqueness × 手取り係数(正規化後) を新設。
    #   生係数 raw = 正例: max(EPS, 1 + GAIN*(σ(K*(np−C0)) − 0.5))、負例: 1.0。
    #   ★正規化：np分布は歪む(高dの左裾)ため固定シグモイドだけでは平均=1と濃淡が両立しない。
    #     正例の生係数の全体平均 NORM で割り「平均=1」を強制→class balance維持と濃淡を両立。
    #     NORM は前段プリパス(全labeled 1スキャン)で算出＝グローバル定数(日次38件では不安定なため)。
    TAKEHOME_EPS = 0.05
    TAKEHOME_K = 3.0  # [段階1] 勝ちバーq25/q75で厚/薄比≈2.13倍（当初想定1.5/0.3の2.11倍に一致＝設計思想準拠）。渋ければ4.0（即PT軽視が§30-32と緊張するので上限目安）
    TAKEHOME_C0 = 0.74  # [実測較正] ★勝ちバーのみのnp_realized中央値(long0.741/short0.733)に合わせる。
    # ※C0は「係数の当たる母集団=正例(勝ちバー)」の中央に置くのが正しい。当初1.5は全トリガー(勝ち+負け)
    #   中央で決めた誤りで、勝ちバー本体(0.74)から0.76上にズレていた（負けバーはd負で高np側に寄り分布が非対称）。
    #   0.74で σ傾き最大部が勝ちバー本体に乗り、即PT薄利(np≈0)も軽い減点(消しすぎない)に。
    #   正規化(NORM)が平均1を別途保証するので②は保たれる。PT変更時は「勝ちバーのみ」のnp中央で再較正。
    TAKEHOME_GAIN = 1.5  # 濃淡の強さ。段階1で弱ければ3〜4に上げる（Kと共にC0本体一致が効いてくる）。

    # --- 前段プリパス: 正例の生手取り係数の全体平均 NORM を算出（平均=1 正規化用の定数） ---
    def _raw_takehome_expr(np_col: str):
        # 生係数 raw = max(EPS, 1 + GAIN*(σ(K*(np−C0)) − 0.5))。np が非有限なら None（平均から除外）。
        return (
            pl.when(pl.col(np_col).is_finite())
            .then(
                pl.max_horizontal(
                    pl.lit(TAKEHOME_EPS),
                    1.0
                    + TAKEHOME_GAIN
                    * (
                        (1.0 / (1.0 + (-TAKEHOME_K * (pl.col(np_col) - TAKEHOME_C0)).exp()))
                        - 0.5
                    ),
                )
            )
            .otherwise(None)
        )

    _lab_all = pl.scan_parquet([str(p) for p in partitions])
    _norm_long = (
        _lab_all.filter(pl.col("label_long") == 1)
        .select(_raw_takehome_expr("np_realized_long").mean().alias("m"))
        .collect()["m"][0]
    )
    _norm_short = (
        _lab_all.filter(pl.col("label_short") == 1)
        .select(_raw_takehome_expr("np_realized_short").mean().alias("m"))
        .collect()["m"][0]
    )
    # ゼロ/None ガード（万一正例が無い等）
    TAKEHOME_NORM_LONG = _norm_long if (_norm_long and _norm_long > 0) else 1.0
    TAKEHOME_NORM_SHORT = _norm_short if (_norm_short and _norm_short > 0) else 1.0
    logging.info(
        f"[TAKEHOME] normalization constants: long={TAKEHOME_NORM_LONG:.4f}, short={TAKEHOME_NORM_SHORT:.4f} "
        f"(C0={TAKEHOME_C0}, K={TAKEHOME_K}, GAIN={TAKEHOME_GAIN})"
    )

    # concurrency_resultsを一度だけ遅延スキャンしておく
    concurrency_lf = pl.scan_parquet(CONCURRENCY_RESULTS_PATH)

    for i, path in enumerate(partitions):
        partition_name = f"{path.parent.parent.parent.name}/{path.parent.parent.name}/{path.parent.name}"
        logging.info(
            f"Processing: [{i + 1}/{len(partitions)}] - Partition {partition_name}..."
        )

        try:
            # 日次パーティションを遅延スキャン
            labeled_lf = pl.scan_parquet(path)

            # --- 修正前 ---
            # final_lf = (
            #     labeled_lf.join(concurrency_lf, on="timestamp", how="left")
            #     .with_columns(

            # --- 修正後 ---
            final_lf = (
                labeled_lf.join(
                    concurrency_lf, on=["timestamp", "timeframe"], how="left"
                )  # ★ timeframeを追加
                .with_columns(
                    [
                        # concurrency_long が存在し、かつ0より大きい場合に uniqueness_long を計算
                        pl.when(
                            pl.col("concurrency_long").is_not_null()
                            & (pl.col("concurrency_long") > 0)
                        )
                        .then(1.0 / pl.col("concurrency_long"))
                        .otherwise(0.0)
                        .alias("uniqueness_long"),
                        # concurrency_short が存在し、かつ0より大きい場合に uniqueness_short を計算
                        pl.when(
                            pl.col("concurrency_short").is_not_null()
                            & (pl.col("concurrency_short") > 0)
                        )
                        .then(1.0 / pl.col("concurrency_short"))
                        .otherwise(0.0)
                        .alias("uniqueness_short"),
                    ]
                )
                # [TAKEHOME §36 / 案B] uniqueness は純粋なまま。別列 w_final = uniqueness × 手取り係数 を新設。
                .with_columns(
                    [
                        (
                            pl.col("uniqueness_long")
                            * pl.when(
                                (pl.col("label_long") == 1)
                                & pl.col("np_realized_long").is_finite()
                            )
                            .then(
                                pl.max_horizontal(
                                    pl.lit(TAKEHOME_EPS),
                                    1.0
                                    + TAKEHOME_GAIN
                                    * (
                                        (
                                            1.0
                                            / (
                                                1.0
                                                + (
                                                    -TAKEHOME_K
                                                    * (pl.col("np_realized_long") - TAKEHOME_C0)
                                                ).exp()
                                            )
                                        )
                                        - 0.5
                                    ),
                                )
                                / TAKEHOME_NORM_LONG  # [正規化] 正例平均=1 に強制
                            )
                            .otherwise(1.0)
                        ).alias("w_final_long"),
                        (
                            pl.col("uniqueness_short")
                            * pl.when(
                                (pl.col("label_short") == 1)
                                & pl.col("np_realized_short").is_finite()
                            )
                            .then(
                                pl.max_horizontal(
                                    pl.lit(TAKEHOME_EPS),
                                    1.0
                                    + TAKEHOME_GAIN
                                    * (
                                        (
                                            1.0
                                            / (
                                                1.0
                                                + (
                                                    -TAKEHOME_K
                                                    * (pl.col("np_realized_short") - TAKEHOME_C0)
                                                ).exp()
                                            )
                                        )
                                        - 0.5
                                    ),
                                )
                                / TAKEHOME_NORM_SHORT  # [正規化] 正例平均=1 に強制
                            )
                            .otherwise(1.0)
                        ).alias("w_final_short"),
                    ]
                )
                # concurrency のみ削除。np_realized は S6_WEIGHTED に残す。
                # [XBRAIN] np_realized_long/short は Cx2 の M1 低dフィルタ(d=PT_MULT-np_realized)で使う。
                # 特徴量からは update_feature_list_v5 / split_features / 各trainerの exclude_exact で
                # 除外済み＝学習特徴量には絶対に入らない（リーク無し）。w_final には既に畳み込み済み。
                .select(
                    pl.all().exclude(
                        [
                            "concurrency_long",
                            "concurrency_short",
                        ]
                    )
                )
            )

            # 日次パーティションのパスから年/月/日を抽出
            day_part = path.parent.name
            month_part = path.parent.parent.name
            year_part = path.parent.parent.parent.name
            year = int(year_part.split("=")[1])
            month = int(month_part.split("=")[1])
            day = int(day_part.split("=")[1])

            # 結果を最終的な日次パーティション構造で書き出す
            output_dir = S6_WEIGHTED_DATASET / f"year={year}/month={month}/day={day}"
            output_dir.mkdir(parents=True, exist_ok=True)
            output_path = output_dir / "data.parquet"

            # 計算を実行し、結果をファイルに書き出す
            result_df = final_lf.collect(engine="streaming")

            if len(result_df) == 0:
                logging.warning(
                    f"Empty dataframe after processing partition: {partition_name}"
                )
                continue

            result_df.write_parquet(output_path, compression="zstd")
            total_processed_rows += len(result_df)

        except Exception as e:
            logging.error(f"Processing for {partition_name} failed: {e}", exc_info=True)
            error_count += 1
            break

    # --- 最終報告 ---
    logging.info("\n" + "=" * 60)
    if error_count == 0:
        logging.info("### MISSION ACCOMPLISHED! All Stages COMPLETED! ###")
        logging.info(
            f"Successfully processed {len(partitions)} partitions ({total_processed_rows} rows)."
        )
        logging.info(f"The final weighted dataset is ready at: {S6_WEIGHTED_DATASET}")
    else:
        logging.error("### PROCESSING FAILED ###")
        logging.error(
            f"An error occurred. {total_processed_rows} rows were processed before failure. Please check the logs."
        )
    logging.info("=" * 60)


if __name__ == "__main__":
    main()
