# models/model_training_metalabeling_Bx_purified.py
# [B案改: エース投入型M2 - Top N特徴量を利用]
# [V5対応: 双方向ラベリング (Long/Short独立生成) + 外部文脈特徴量(S7_CONTEXT_FEATURES)の結合]

import sys
from pathlib import Path
import logging
import argparse
import datetime
import warnings
import shutil
from dataclasses import dataclass
from typing import List

import json
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
    # S3_FEATURES_FOR_TRAINING,  # ← 削除
    S3_FEATURES_FOR_TRAINING_V5,  # ★ 追加 (プロンプト⑫ 修正①)
    S3_SELECTED_FEATURES_DIR,     # ★ 追加: 方向別動的特徴量ロードのため
    S3_SELECTED_FEATURES_ORTHOGONAL_DIR,  # ★ 追加: 直交分割版
    S7_RUN_CONFIG,                # ★ 追加: Ax→Bx間の設定引き継ぎ
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


@dataclass
class MetaLabelingConfig:
    weighted_dataset_path: Path = S6_WEIGHTED_DATASET
    m1_long_oof_path: Path = S7_M1_OOF_PREDICTIONS_LONG
    m1_short_oof_path: Path = S7_M1_OOF_PREDICTIONS_SHORT
    output_long_dir: Path = S7_META_LABELED_OOF_LONG
    output_short_dir: Path = S7_META_LABELED_OOF_SHORT
    # feature_list_path: Path = S3_FEATURES_FOR_TRAINING  # ← 削除
    feature_list_path: Path = S3_FEATURES_FOR_TRAINING_V5  # ★ 追加 (プロンプト⑫ 修正①)
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

        # V5ラベリングエンジンが生成する全メタデータ・未来情報の完全除外
        exclude_exact = {
            "timestamp",
            # "timeframe",  # ★削除: 特徴量として次へ渡すため除外しない (Bに統一)
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
        }

        features = ["timeframe"]  # ★変更: 特徴量リストの先頭に明示的にセット (Bに統一)
        for col in raw_features:
            if col in exclude_exact or col == "timeframe":  # ★変更: 重複防止 (Bに統一)
                continue
            if col.startswith("is_trigger_on"):
                continue
            features.append(col)

        logging.info(f"  -> Loaded {len(features)} valid features.")
        return features

    def run(self) -> None:
        logging.info(
            "### Script 2/3: Meta-Label Generation (V5 Bidirectional / Context Plus) ###"
        )
        logging.info("Using threshold-based sampling: M1 predictions >= 0.50 per day.")

        # 双方向（Long/Short）でループ処理
        for direction in ["long", "short"]:
            logging.info(
                f"\n{'=' * 60}\n=== Starting Meta-Label Generation for {direction.upper()} ===\n{'=' * 60}"
            )

            # M2専用特徴量リストの動的読み込み (プロンプト⑫ 修正①・Bに統一)
            _feat_dir = getattr(self, '_selected_features_dir', S3_SELECTED_FEATURES_DIR)
            dedicated_path = _feat_dir / f"m2_{direction}_features.txt"
            raw_m2_features = self._load_features(dedicated_path)

            # m1_pred_proba はS6には存在せず後段でJoinされるため、S6抽出用リストからは除外
            self.features = [f for f in raw_m2_features if f != "m1_pred_proba"]

            # 方向ごとの設定切り替え
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

                # ★修正: replace_time_zone → cast に変更 (Bに統一)
                m1_oof_df = m1_oof_df.with_columns(
                    pl.col("timestamp").cast(pl.Datetime("us", "UTC"))
                )
                # ★追加: Temperature Scaling（AUC完全保持・過信バイアス緩和）
                _T = getattr(self, '_temperature_t', 1.0)
                if _T != 1.0 and "prediction" in m1_oof_df.columns:
                    raw_proba = m1_oof_df["prediction"].to_numpy()
                    # 対数オッズ変換 → T除算 → シグモイド
                    raw_proba = np.clip(raw_proba, 1e-7, 1 - 1e-7)
                    logits = np.log(raw_proba / (1 - raw_proba))
                    calibrated = 1.0 / (1.0 + np.exp(-logits / _T))
                    m1_oof_df = m1_oof_df.with_columns(
                        pl.Series("prediction", calibrated)
                    )
                    logging.info(
                        f"  🌡️  Temperature Scaling T={_T}: "
                        f"avg {raw_proba.mean():.3f} → {calibrated.mean():.3f}, "
                        f">=0.95: {(raw_proba>=0.95).mean()*100:.1f}% → {(calibrated>=0.95).mean()*100:.1f}%"
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
            # ★修正: "timeframe" は self.features の中に既に含まれているため外す (Bに統一)
            columns_to_select = [
                "timestamp",
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

                    # ★修正: Int32キャスト処理を完全に削除し unique 処理のみ残す (Bに統一)
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

                # --- サンプリング・結合ロジック ---
                try:
                    # 1. その日のM1 OOF予測データを取得
                    daily_m1_oof = m1_oof_df.filter(pl.col("date") == partition_date)
                    if daily_m1_oof.is_empty():
                        continue

                    # ★修正: Top 50%サンプリング(未来情報リーク)を廃止し
                    #         固定閾値フィルタに変更 (プロンプト⑫ 修正②・Bに統一)
                    # 2. 一定の確率（例：0.5以上）を超えた「自信のあるシグナル」だけをメタモデルに渡す
                    THRESHOLD = (
                        0.50  # ※ベースモデルの強さに応じて 0.55 等に調整してください
                    )
                    top_n_keys = (
                        daily_m1_oof.filter(pl.col("m1_pred_proba") >= THRESHOLD)
                        .select(["timestamp", "timeframe"])  # ★修正: timeframe追加 (プロンプト⑫ 修正②)
                        .unique()
                    )
                    if top_n_keys.is_empty():
                        continue

                    # 3. S6データを閾値超えシグナルでフィルタリング
                    sampled_chunk_lf = (
                        df_chunk.lazy()
                        .join(
                            top_n_keys.lazy(),
                            on=["timestamp", "timeframe"],  # ★修正: 複合キーに変更 (プロンプト⑫ 修正②)
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

                    # 5. M1予測確率 (m1_pred_proba) を結合
                    # ★修正: joinキーを複合キーに変更 (プロンプト⑫ 修正②)
                    merged_chunk_lf = sampled_chunk_lf.join(
                        daily_m1_oof.lazy().select(
                            [
                                "timestamp",
                                "timeframe",  # ★修正: timeframe追加 (Bに統一)
                                "m1_pred_proba",
                            ]
                        ),
                        on=["timestamp", "timeframe"],  # ★修正: 複合キーに変更 (プロンプト⑫ 修正②)
                        how="inner",
                    )

                    merged_chunk_df = merged_chunk_lf.collect()
                    if merged_chunk_df.is_empty():
                        continue

                    # 6. メタラベルを生成 (label_long / label_short と同一値)
                    final_chunk_df = merged_chunk_df.with_columns(
                        pl.col(label_col).alias("meta_label"),
                        pl.col(uniqueness_col).alias("uniqueness"),
                    )

                    # --- 出力 ---
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
        logging.info("### Script 2/3 FINISHED! (V5 Bidirectional Context Plus) ###")
        logging.info("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Script 2/3: Meta-Label Generation (V5 Bidirectional)"
    )
    parser.add_argument(
        "--test",
        action="store_true",
        help="Run in quick test mode, processing only the first 5 partitions.",
    )

    parser.add_argument(
        "--temperature",
        type=float,
        default=None,
        help="Temperature Scalingのパラメータ T (デフォルト: インタラクティブで選択)",
    )
    args = parser.parse_args()

    # =========================================================
    # 特徴量セット選択（run_configから引き継ぎ or インタラクティブ）
    # =========================================================
    FEATURE_SETS = {
        "1": {
            "label": "selected_features_v5          (オリジナル / 全部載せ)",
            "dir":   S3_SELECTED_FEATURES_DIR,
        },
        "2": {
            "label": "selected_features_orthogonal_v5 (M1/M2直交分割版)",
            "dir":   S3_SELECTED_FEATURES_ORTHOGONAL_DIR,
        },
    }

    selected_dir = S3_SELECTED_FEATURES_DIR  # デフォルト
    if S7_RUN_CONFIG.exists():
        with open(S7_RUN_CONFIG) as f:
            run_config = json.load(f)
        prev_label = run_config.get("features_label", "不明")
        prev_dir   = run_config.get("features_dir", str(S3_SELECTED_FEATURES_DIR))
        print(f"\n前回のAx実行設定: {prev_label}")
        ans = input("同じ特徴量セットを使用しますか？ [Y/n]: ").strip().lower()
        if ans in ("", "y"):
            selected_dir = Path(prev_dir)
            logging.info(f"📂 前回のAx設定を引き継ぎ: {prev_label}")
        else:
            print("\n" + "=" * 60)
            print("  📂 特徴量セットを選択してください:")
            for key, val in FEATURE_SETS.items():
                print(f"    [{key}] {val['label']}")
            print("=" * 60)
            fs_ans = input("選択 [1/2, Enterでデフォルト(1)]: ").strip() or "1"
            if fs_ans not in FEATURE_SETS:
                fs_ans = "1"
            selected_dir = FEATURE_SETS[fs_ans]["dir"]
            logging.info(f"📂 特徴量セット: {FEATURE_SETS[fs_ans]['label']}")
    else:
        print("\n⚠️  run_configが見つかりません。特徴量セットを手動選択してください。")
        print("=" * 60)
        for key, val in FEATURE_SETS.items():
            print(f"    [{key}] {val['label']}")
        print("=" * 60)
        fs_ans = input("選択 [1/2, Enterでデフォルト(1)]: ").strip() or "1"
        if fs_ans not in FEATURE_SETS:
            fs_ans = "1"
        selected_dir = FEATURE_SETS[fs_ans]["dir"]
        logging.info(f"📂 特徴量セット: {FEATURE_SETS[fs_ans]['label']}")

    # =========================================================
    # Temperature Scaling パラメータ選択
    # =========================================================
    if args.temperature is not None:
        temperature_t = args.temperature
        logging.info(f"🌡️  Temperature Scaling T={temperature_t} (引数指定)")
    else:
        print("\n" + "=" * 60)
        print("  🌡️  Temperature Scaling の設定:")
        print("    T=1.0 → 較正なし（元の確率をそのまま使用）")
        print("    T=2.0 → 推奨（0.95→約0.73に圧縮、AUC完全保持）")
        print("    T=3.0 → 強め（0.95→約0.62に圧縮）")
        print("=" * 60)
        t_input = input("Tの値を入力 [Enterでデフォルト T=2.0]: ").strip()
        try:
            temperature_t = float(t_input) if t_input else 2.0
        except ValueError:
            temperature_t = 2.0
        logging.info(f"🌡️  Temperature Scaling T={temperature_t}")

    # run_configを更新（Cxへ引き継ぎ）
    run_config_data = {
        "features_dir":   str(selected_dir),
        "features_label": str(selected_dir.name),
        "selected_by":    "Bx",
        "temperature_t":  temperature_t,
    }
    S7_RUN_CONFIG.parent.mkdir(parents=True, exist_ok=True)
    with open(S7_RUN_CONFIG, "w") as f:
        json.dump(run_config_data, f, indent=2, ensure_ascii=False)
    logging.info(f"📝 実行設定を更新: {S7_RUN_CONFIG}")

    config = MetaLabelingConfig(test=args.test)
    generator = MetaLabelGenerator(config)
    generator._selected_features_dir = selected_dir
    generator._temperature_t = temperature_t
    generator.run()
