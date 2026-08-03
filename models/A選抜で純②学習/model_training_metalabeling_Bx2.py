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


# ============================================================================
# 【手動設定】A→B 接続 — M1 ゲートの「向き」と「重み」
# ============================================================================
#
# 【背景 — なぜ向きを切り替えられるようにしたか（測定結果、レポート§12.5）】
#   M1 は label_A（P(L)錨・下駄あり）で学習するため、高スコア ≒ 方向に大きな d。
#   そのバーは直後1〜10分でわずかに「逆行」する（短期リバーサル）。
#   前方ドリフト（M2 が知りたいもの）を decile 別に実測すると:
#
#       decile  t=1分     t=5分    t=10分   t=30分
#       D01   +0.0120  +0.0162  +0.0174  +0.0058   ← 最低スコアが最良
#       D05   -0.0014  +0.0061  +0.0042  +0.0132   ← ゲート境界(proba0.5)
#       D10   -0.0126  -0.0168  -0.0213  -0.0034   ← 最高スコアが最悪
#       D10−D01 の t統計量: t=1分 −5.36 / t=3分 −3.03 / t=5分 −3.45（有意）
#
#   さらに「ゲート境界(proba 0.5)の前後に前方ドリフトの段差が無い」。
#   → 現行ゲートは、M2 が必要とするものとは無関係な場所で母集団を半分に切っている。
#   → M1 が答えている問い（PT に先着するか）と M2 が知りたい問い（前方に動くか）は
#     別物で、しかも符号が逆、というのが測定の結論。
#
# 【実験の順序（変数分離）】
#   実験0（現行・基準）  : GATE="high", WEIGHT="high"   … これまでの結果が基準値
#   実験1（M1の影響を切る）: GATE="off",  WEIGHT="flat"   ← ★まずこれ
#       M1 の影響を完全に中立化。M2 は全 353,226 本で純②を学習する。
#       現行より良くなれば「M1 の影響は差し引きで有害だった」が確定する。
#       学習データは 約156,000 → 353,226（2.3倍）。弱信号ほどデータ量が効く。
#   実験2（積極的に逆側を選ぶ）: GATE="low", WEIGHT="low"
#       実測で前方ドリフトが良かった側（低スコア側）を能動的に選ぶ。
#       標本数は実験0とほぼ同じ（半分を別の半分に入れ替えるだけ）。
#
# 【注意】実験0→1→2 は Bx2 と Cx2 のみ再実行でよい。
#   ・ラベリング（label_A / label_B）は不変 → 再ラベリング不要
#   ・Ax2（M1 の学習と OOF）も不変 → 再実行不要
#   ・Cx2 はコード変更不要。M1 の pkl が残っていれば M1 最終学習は自動スキップされ、
#     M2 だけが学習し直される（_ensure_model_trained の model_path.exists() 判定）
#   ・比較は AUC だけでなく、measure_mu_time_profile.py --rule model --oof m2 で
#     選抜効果（現行 +0.0423 ATR @30分）が上がるかも見ること
# ----------------------------------------------------------------------------

# --- ゲートの向き（どちら側の q_A を M2 に通すか）---
#   "high" : q_A が高い側を通す（現行。下駄の世界の設計）
#   "off"  : ゲートなし。全バーを通す（M1 の選抜影響をゼロにする）
#   "low"  : q_A が低い側を通す（測定で前方ドリフトが良かった側）
M1_GATE_MODE = "off"

# ゲート閾値（logit 空間）。"high" なら >= 、"low" なら <= の境界。"off" では無視。
#   0.0 → proba 0.5。Ax2 は scale_pos_weight で proba を 0.5 中心に均衡化するため、
#   0.0 は実質「中央値カット」で強め。μ枯れのみ切るなら -0.85（≈ proba 0.3）等。
M1_GATE_LOGIT = 0.0

# --- サンプル重み h(q_A) の向き（Cx2 が消費する重み = uniqueness_B × h(q_A)）---
#   "high" : h = 2p − 1   （現行。高スコアを重く。p=0.5→0, p=1→1）
#   "flat" : h = 1        （M1 の影響なし。uniqueness_B だけで重み付け）
#   "low"  : h = 1 − 2p   （低スコアを重く。p=0→1, p=0.5→0）
M1_WEIGHT_MODE = "flat"

# h の下限。0.0 だと境界付近の標本が実質ゼロ重みで消える。
# n_eff の枯渇が心配なら 0.1〜0.2 を入れると全標本が残る。
H_FLOOR = 0.0


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
            # --- 入れ子メタラベリング: A/B 2ラベル系（接頭辞ガードと二重で防御）---
            "label_A_long",
            "label_A_short",
            "label_B_long",
            "label_B_short",
            "duration_A_long",
            "duration_A_short",
            "duration_B_long",
            "duration_B_short",
            "uniqueness_A_long",
            "uniqueness_A_short",
            "uniqueness_B_long",
            "uniqueness_B_short",
            "concurrency_A_long",
            "concurrency_A_short",
            "concurrency_B_long",
            "concurrency_B_short",
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

        # [入れ子メタラベリング] ラベル系接頭辞ガード（個別列挙の漏れに対する最終防御線）
        LEAK_PREFIXES = ("label_", "duration_", "uniqueness_", "concurrency_")

        features = []
        for col in raw_features:
            if col in exclude_exact:
                continue
            if col.startswith(LEAK_PREFIXES):
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
        _gate_txt = {
            "off": "ゲートなし（全バー通過）",
            "low": f"q_A <= logit {M1_GATE_LOGIT}（低スコア側を通す）",
            "high": f"q_A >= logit {M1_GATE_LOGIT}（高スコア側を通す・現行）",
        }.get(M1_GATE_MODE, M1_GATE_MODE)
        _w_txt = {
            "flat": "h(q_A) = 1（M1の重み影響なし）",
            "low": "h(q_A) = 1 − 2p（低スコアを重く）",
            "high": "h(q_A) = 2p − 1（高スコアを重く・現行）",
        }.get(M1_WEIGHT_MODE, M1_WEIGHT_MODE)
        logging.info("=" * 78)
        logging.info(f"  M1_GATE_MODE   = {M1_GATE_MODE:<5} : {_gate_txt}")
        logging.info(
            f"  M1_WEIGHT_MODE = {M1_WEIGHT_MODE:<5} : {_w_txt}  (H_FLOOR={H_FLOOR})"
        )
        logging.info("=" * 78)

        for direction in ["long", "short"]:
            logging.info(
                f"\n{'=' * 60}\n=== Starting Meta-Label Generation for {direction.upper()} ===\n{'=' * 60}"
            )

            # 方向別M2特徴量リストを読み込む（直交分割版固定）
            dedicated_path = (
                S3_SELECTED_FEATURES_ORTHOGONAL_DIR / f"m2_{direction}_features.txt"
            )
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
            # 入れ子メタラベリング: B は「本番発注幾何(B)」のラベルを学ぶ。
            # 重みは B幾何の uniqueness を素に、後段で h(q_A) と合成する。
            label_col = f"label_B_{direction}"
            uniqueness_col = f"uniqueness_B_{direction}"

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
                    m1_oof_df = m1_oof_df.with_columns(pl.Series("prediction", logits))
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

                    # [入れ子メタラベリング] M1ゲート。向きは上部 M1_GATE_MODE で切替。
                    #   "high" : q_A 高い側を通す（現行）
                    #   "off"  : 通さない＝全バー（M1の選抜影響をゼロに）
                    #   "low"  : q_A 低い側を通す（測定で前方ドリフトが良かった側）
                    if M1_GATE_MODE == "off":
                        _gate = pl.lit(True)
                    elif M1_GATE_MODE == "low":
                        _gate = pl.col("m1_pred_proba") <= M1_GATE_LOGIT
                    else:
                        _gate = pl.col("m1_pred_proba") >= M1_GATE_LOGIT
                    top_n_keys = (
                        daily_m1_oof.filter(_gate)
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

                    # メタラベルと合成サンプル重みを生成
                    # meta_label = B幾何ラベル（本番発注幾何の先着）
                    # uniqueness(=Cx2 が消費する重み) = uniqueness_B × h(q_A)
                    #   m1_pred_proba はこの時点で logit 値なので sigmoid で proba に戻す。
                    #   向きは上部 M1_WEIGHT_MODE で切替（high / flat / low）。
                    #   ※ n_eff 暴れが問題なら h を順位正規化に差し替える（設計上の代替）。
                    _p = 1.0 / (1.0 + (-pl.col("m1_pred_proba")).exp())  # logit → proba
                    if M1_WEIGHT_MODE == "flat":
                        h_qa = pl.lit(1.0)
                    elif M1_WEIGHT_MODE == "low":
                        h_qa = (1.0 - 2.0 * _p).clip(H_FLOOR, 1.0)
                    else:
                        h_qa = (2.0 * _p - 1.0).clip(H_FLOOR, 1.0)
                    final_chunk_df = merged_chunk_df.with_columns(
                        pl.col(label_col).alias("meta_label"),
                        (pl.col(uniqueness_col) * h_qa).alias("uniqueness"),
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
