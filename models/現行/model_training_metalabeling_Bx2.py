# /workspace/models/model_training_metalabeling_Bx2.py
# [1周目: メタラベル生成]
# [直交分割版: S3_SELECTED_FEATURES_ORTHOGONAL_DIR から方向別特徴量を読み込む]
# [logit変換: M1予測確率をlogit空間に変換してM2へ渡す]

import sys
from pathlib import Path
import logging
import argparse
import datetime
import warnings
import shutil
from dataclasses import dataclass
from typing import List

import numpy as np
import polars as pl
from tqdm import tqdm

# --- プロジェクトのルートディレクトリをPythonの検索パスに追加 ---
project_root = Path(__file__).resolve().parents[1]
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

from blueprint import (
    S6_WEIGHTED_DATASET,
    S7_M1_OOF_PREDICTIONS_LONG,
    S7_M1_OOF_PREDICTIONS_SHORT,
    S7_META_LABELED_OOF_LONG,
    S7_META_LABELED_OOF_SHORT,
    S3_FEATURES_FOR_TRAINING_V5,
    S3_SELECTED_FEATURES_ORTHOGONAL_DIR,
)

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
warnings.filterwarnings("ignore", category=UserWarning)
try:
    from polars.exceptions import PolarsInefficientMapWarning
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    warnings.filterwarnings("ignore", category=PolarsInefficientMapWarning)
except ImportError:
    pass



# =============================================================================
# 【手動設定】M1 ゲートの向き
# =============================================================================
#
# 【背景 — 測定で判明したこと】
#   M1 は L錨ラベルで学習するため、高スコア ≒ 方向に大きな d のバーになる。
#   その前方挙動を decile 別に測ると、年によって【符号が反転】していた:
#       2021 +0.053 / 2022 +0.092 / 2023 +0.032 / 2024 +0.018 / 2025 +0.005
#       （低decile − 高decile。χ²=24.1、p=0.00008 で年による違いが有意）
#   原因も判明した。大きな |d| の後の挙動が相場のレジームで反転するためである:
#       レンジ相場(2022-23) → 大きな動きは戻る
#       トレンド相場(2024-25) → 大きな動きは続く
#   M1 はレジームの情報を持たないので、年をまたぐと打ち消し合う。
#
#   よって「高スコア側を通す」も「低スコア側を通す」も、向きを固定する根拠が無い。
#   さらに効率比ルールで選んだバーの中を M1 スコアで分けても効果は分かれなかった
#   （上−下の |t| 最大 1.24。M1 は上乗せしない）。
#
#   → ゲートは "off"（全バーを通す）が既定。母集団が最大になり、
#     弱い信号ほどデータ量が効く。レジーム判定は効率比の特徴量に任せる。
#
# 【重みについて】
#   旧構成には q_A によるサンプル重み付けは存在しない（uniqueness をそのまま使う）。
#   よって「重みを切る」操作は不要。ここではゲートのみを可変にする。
#
# 【元に戻すには】
#   M1_GATE_MODE = "high" にすれば従来どおり（proba >= 0.5 相当を通す）。
# -----------------------------------------------------------------------------
#   "off"  : ゲートなし。全バーを M2 の学習母集団にする（既定）
#   "high" : m1_pred_proba >= M1_GATE_LOGIT を通す（従来の挙動）
#   "low"  : m1_pred_proba <= M1_GATE_LOGIT を通す
M1_GATE_MODE = "off"

# ゲート閾値（logit 空間）。0.0 は proba 0.5 に相当。"off" では無視される。
M1_GATE_LOGIT = 0.0
# =============================================================================

@dataclass
class MetaLabelingConfig:
    weighted_dataset_path: Path = S6_WEIGHTED_DATASET
    m1_long_oof_path: Path = S7_M1_OOF_PREDICTIONS_LONG
    m1_short_oof_path: Path = S7_M1_OOF_PREDICTIONS_SHORT
    output_long_dir: Path = S7_META_LABELED_OOF_LONG
    output_short_dir: Path = S7_META_LABELED_OOF_SHORT
    feature_list_path: Path = S3_FEATURES_FOR_TRAINING_V5
    test: bool = False


class MetaLabelGenerator:
    def __init__(self, config: MetaLabelingConfig):
        self.config = config
        self.partitions = self._discover_partitions()

        if self.config.test:
            logging.warning(
                "--- TEST MODE: Processing only the first 5 partitions. ---"
            )
            self.partitions = self.partitions[:5]

    def _discover_partitions(self) -> List[datetime.date]:
        logging.info("Discovering all physical partitions from the weighted dataset...")
        paths = self.config.weighted_dataset_path.glob("year=*/month=*/day=*")
        dates = sorted(
            list(
                set(
                    datetime.date(
                        int(p.parent.parent.name[5:]),
                        int(p.parent.name[6:]),
                        int(p.name[4:]),
                    )
                    for p in paths
                )
            )
        )
        logging.info(f"  -> Discovered {len(dates)} daily partitions.")
        return dates

    def _load_features(self, feature_path: Path) -> List[str]:
        logging.info(f"Loading dedicated feature list from {feature_path.name}...")
        if not feature_path.exists():
            raise FileNotFoundError(f"Feature list file not found at: {feature_path}")

        with open(feature_path, "r") as f:
            raw_features = [line.strip() for line in f if line.strip()]

        # 【Phase 5 修正 (#35)】 Ax2/Bx2/Cx2 の exclude_exact を統一 (union)
        # 各ファイル間で不整合があった項目 (concurrency_long/short, duration_long/short,
        # payoff_ratio_long/short 等) を全て含めることで、3 ファイル間の挙動を一致させる。
        # disc は学習対象外メタデータ (週末跨ぎギャップ判定 bool 列) — 最終防御線。
        exclude_exact = {
            # --- 基本メタ ---
            "timestamp",
            "timeframe",  # 学習特徴量から除外（データ管理用カラムとしては保持）
            "is_trigger",
            "t1",
            "direction",
            "exit_type",
            "first_ex_reason_int",
            # --- ラベル系 (双方向) ---
            "label",
            "label_long",
            "label_short",
            # --- uniqueness 系 (双方向) ---
            "uniqueness",
            "uniqueness_long",
            "uniqueness_short",
            # --- duration 系 (双方向) ---
            "duration_long",
            "duration_short",
            # --- concurrency 系 (双方向、未来情報リーク防止) ---
            "concurrency_long",
            "concurrency_short",
            # --- payoff/multiplier 系 (双方向) ---
            "payoff_ratio",
            "payoff_ratio_long",
            "payoff_ratio_short",
            "pt_multiplier",
            "sl_multiplier",
            # --- ATR / 補助計算 ---
            "atr_value",
            "calculated_body_ratio",
            "fallback_vol",
            # --- 価格データ ---
            "open",
            "high",
            "low",
            "close",
            # --- メタラベリング系 ---
            "m1_pred_proba",
            "meta_label",
            # --- 【Phase 5 修正】学習対象外メタデータ ---
            "disc",  # 週末跨ぎギャップ判定 bool 列 (engine_1_C 経由で漏れる可能性に対する最終防御線)
        }

        features = []
        for col in raw_features:
            if col in exclude_exact:
                continue
            if col.startswith("is_trigger_on"):
                continue
            features.append(col)

        logging.info(f"  -> Loaded {len(features)} valid features.")
        return features

    def run(self) -> None:
        logging.info(
            "### Script 2/3: Meta-Label Generation (直交分割版 / Logit変換) ###"
        )
        logging.info("Using threshold-based sampling: M1 logit predictions >= 0.0 (= proba 0.50) per day.")

        for direction in ["long", "short"]:
            logging.info(
                f"\n{'=' * 60}\n=== Starting Meta-Label Generation for {direction.upper()} ===\n{'=' * 60}"
            )

            # 方向別M2特徴量リストを読み込む（直交分割版固定）
            dedicated_path = S3_SELECTED_FEATURES_ORTHOGONAL_DIR / f"m2_{direction}_features.txt"
            raw_m2_features = self._load_features(dedicated_path)

            # m1_pred_probaはS6には存在せず後段でJoinされるため、S6抽出用リストからは除外
            self.features = [f for f in raw_m2_features if f != "m1_pred_proba"]

            oof_path = (
                self.config.m1_long_oof_path
                if direction == "long"
                else self.config.m1_short_oof_path
            )
            output_dir = (
                self.config.output_long_dir
                if direction == "long"
                else self.config.output_short_dir
            )
            label_col = f"label_{direction}"
            uniqueness_col = f"uniqueness_{direction}"

            if not oof_path.exists():
                logging.error(
                    f"{direction.upper()} OOF prediction file not found at: {oof_path}. Skipping."
                )
                continue

            if output_dir.exists():
                logging.warning(
                    f"Output directory {output_dir} exists. Removing it for a clean run."
                )
                shutil.rmtree(output_dir)
            output_dir.mkdir(parents=True)

            # --- OOF予測のロード ---
            logging.info(
                f"Loading {direction.upper()} M1 OOF predictions from {oof_path}..."
            )
            try:
                m1_oof_df = pl.read_parquet(oof_path)
                m1_oof_df = m1_oof_df.with_columns(
                    pl.col("timestamp").cast(pl.Datetime("us", "UTC"))
                )

                # --- Logit変換: 確率→logit空間に変換してM2へ渡す ---
                # 学習時(Bx2)と本番(main.py)の入力分布を完全一致させるため
                # 閾値: proba 0.50 → logit 0.0
                if "prediction" in m1_oof_df.columns:
                    raw_proba = m1_oof_df["prediction"].to_numpy()
                    raw_proba_clipped = np.clip(raw_proba, 1e-7, 1 - 1e-7)
                    logits = np.log(raw_proba_clipped / (1 - raw_proba_clipped))
                    # [-10, +10]クリッピング: 1e-7クリップだと[-16.1, +16.1]になり
                    # min_data_in_leaf=100との衝突でビンが浪費されるため有効範囲に収める
                    logits = np.clip(logits, -10.0, 10.0)
                    m1_oof_df = m1_oof_df.with_columns(
                        pl.Series("prediction", logits)
                    )
                    logging.info(
                        f"  📐 Logit変換: "
                        f"proba avg {raw_proba.mean():.3f} → logit avg {logits.mean():.3f}, "
                        f"logit range [{logits.min():.2f}, {logits.max():.2f}], "
                        f"proba>=0.95: {(raw_proba >= 0.95).mean() * 100:.1f}%"
                    )
                if "prediction" in m1_oof_df.columns:
                    m1_oof_df = m1_oof_df.rename({"prediction": "m1_pred_proba"})
                m1_oof_df = m1_oof_df.with_columns(
                    pl.col("timestamp").dt.date().alias("date")
                )
            except Exception as e:
                logging.error(
                    f"Failed to load or process {direction.upper()} M1 OOF predictions: {e}",
                    exc_info=True,
                )
                continue

            # --- 特徴量カラムの選定 ---
            # timeframeはjoinキー（timestamp + timeframe）として必須のため
            # 学習特徴量リストとは独立して明示的に保持する
            columns_to_select = [
                "timestamp",
                "timeframe",
                "atr_value",
                label_col,
                uniqueness_col,
            ] + self.features

            total_records_processed = 0

            for partition_date in tqdm(
                self.partitions, desc=f"Generating {direction.capitalize()} Meta-Labels"
            ):
                s6_partition_path = (
                    self.config.weighted_dataset_path
                    / f"year={partition_date.year}/month={partition_date.month}/day={partition_date.day}/data.parquet"
                )

                try:
                    if not s6_partition_path.exists():
                        continue
                    df_chunk = pl.read_parquet(s6_partition_path)

                    # timeframeは文字列型のままunique処理（Int32変換不要）
                    if "timeframe" in df_chunk.columns:
                        df_chunk = df_chunk.unique(
                            subset=["timestamp", "timeframe"],
                            keep="last",
                            maintain_order=True,
                        )
                except Exception:
                    continue

                if df_chunk.is_empty():
                    continue

                try:
                    daily_m1_oof = m1_oof_df.filter(pl.col("date") == partition_date)
                    if daily_m1_oof.is_empty():
                        continue

                    # logit空間での閾値フィルタリング
                    # proba 0.50 → logit 0.0
                    # M1 ゲート。向きは冒頭の M1_GATE_MODE で切り替える。
                    #   "off"  = 全バー通過（既定。M1の選抜影響をゼロにする）
                    #   "high" = 高スコア側を通す（従来）
                    #   "low"  = 低スコア側を通す
                    if M1_GATE_MODE == "off":
                        _gate_expr = pl.lit(True)
                    elif M1_GATE_MODE == "low":
                        _gate_expr = pl.col("m1_pred_proba") <= M1_GATE_LOGIT
                    else:
                        _gate_expr = pl.col("m1_pred_proba") >= M1_GATE_LOGIT
                    top_n_keys = (
                        daily_m1_oof.filter(_gate_expr)
                        .select(["timestamp", "timeframe"])
                        .unique()
                    )
                    if top_n_keys.is_empty():
                        continue

                    # S6データを閾値超えシグナルでフィルタリング
                    sampled_chunk_lf = (
                        df_chunk.lazy()
                        .join(
                            top_n_keys.lazy(),
                            on=["timestamp", "timeframe"],
                            how="inner",
                        )
                        .select(
                            [
                                col
                                for col in columns_to_select
                                if col in df_chunk.columns
                            ]
                        )
                    )

                    # m1_pred_proba（logit値）を結合
                    merged_chunk_lf = sampled_chunk_lf.join(
                        daily_m1_oof.lazy().select(
                            ["timestamp", "timeframe", "m1_pred_proba"]
                        ),
                        on=["timestamp", "timeframe"],
                        how="inner",
                        coalesce=True,
                    )

                    merged_chunk_df = merged_chunk_lf.collect()
                    if merged_chunk_df.is_empty():
                        continue

                    # メタラベルを生成
                    final_chunk_df = merged_chunk_df.with_columns(
                        pl.col(label_col).alias("meta_label"),
                        pl.col(uniqueness_col).alias("uniqueness"),
                    )

                    output_partition_dir = (
                        output_dir
                        / f"year={partition_date.year}/month={partition_date.month}/day={partition_date.day}"
                    )
                    output_partition_dir.mkdir(parents=True, exist_ok=True)

                    final_chunk_df.write_parquet(
                        output_partition_dir / "data.parquet", compression="zstd"
                    )
                    total_records_processed += len(final_chunk_df)

                except Exception as e:
                    logging.error(
                        f"Error processing partition {partition_date} for {direction}: {e}",
                        exc_info=True,
                    )
                    continue

            logging.info(
                f"Finished {direction.upper()} -> Total samples generated: {total_records_processed}"
            )

        logging.info("\n" + "=" * 60)
        logging.info("### Script 2/3 FINISHED! (直交分割版 / Logit変換) ###")
        logging.info("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Script 2/3: Meta-Label Generation (直交分割版)"
    )
    parser.add_argument(
        "--test",
        action="store_true",
        help="Run in quick test mode, processing only the first 5 partitions.",
    )

    args = parser.parse_args()
    config = MetaLabelingConfig(test=args.test)
    generator = MetaLabelGenerator(config)
    generator.run()
