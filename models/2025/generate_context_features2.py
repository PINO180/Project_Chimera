# /workspace/models/generate_context_features.py
# [修正版: 複数のS2カテゴリファイルから特徴量を結合するよう修正]

import sys
import logging
from pathlib import Path

import polars as pl
import numpy as np
import joblib
from hmmlearn.hmm import GaussianHMM

# --- プロジェクトのルートディレクトリをPythonの検索パスに追加 ---
project_root = Path(__file__).resolve().parents[1]
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

# --- Blueprintからパス設定をインポート ---
from blueprint import (
    S1_BASE_MULTITIMEFRAME,
    S2_FEATURES_FIXED,
    S7_REGIME_MODEL,
    S7_CONTEXT_FEATURES,
)

# --- ロギング設定 ---
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

# --- 定数設定 ---
# HMM（隠れマルコフモデル）の状態数
HMM_N_COMPONENTS = 2

# --- S1 (基礎データ) ---
S1_D1_PATH = S1_BASE_MULTITIMEFRAME / "timeframe=D1"

# --- S2 (特徴量データ) - カテゴリ別のパス定義 ---
S2_CAT_A_PATH = (
    S2_FEATURES_FIXED / "feature_value_a_vast_universeA" / "features_e1a_D1.parquet"
)
S2_CAT_C_PATH = (
    S2_FEATURES_FIXED / "feature_value_a_vast_universeC" / "features_e1c_D1.parquet"
)
# カテゴリ2Aのファイル名を 'features_e2a_D1.parquet' と仮定
S2_CAT_2A_PATH = (
    S2_FEATURES_FIXED
    / "feature_value_complexity_theory_MFDFA&kolmogorov"
    / "features_e2a_D1.parquet"
)

# --- S2から読み込む特徴量のマッピング ---
# S2 (Cat A) - 期間50の尖度を選択
FEATURES_S2_A = ["e1a_statistical_kurtosis_50"]

# S2 (Cat C) - 期間21のADXを選択 (ATRと期間を合わせる)
FEATURES_S2_C = ["e1c_adx_21"]
ATR_COL_NAME = "e1c_atr_21"  # (このカラムは存在したので変更なし)

# S2 (Cat 2A) - 期間1000のものを選択
FEATURES_S2_2A = ["e2a_mfdfa_hurst_mean_1000", "e2a_kolmogorov_complexity_1000"]


def load_and_prepare_s2_data() -> pl.LazyFrame:
    """
    カテゴリA, C, 2A から必要なD1特徴量を読み込み、
    一つのLazyFrameに結合（join）する。
    """
    logging.info("Loading S2 D1 features from multiple categories...")

    # --- 1. カテゴリA (Kurtosis) ---
    logging.info(f"Loading S2 Cat A: {S2_CAT_A_PATH}")
    if not S2_CAT_A_PATH.exists():
        raise FileNotFoundError(f"S2 Cat A data not found: {S2_CAT_A_PATH}")
    lf_s2_a = pl.scan_parquet(S2_CAT_A_PATH).select(
        pl.col("timestamp").cast(pl.Datetime("us", "UTC")),
        pl.col(FEATURES_S2_A),
    )

    # --- 2. カテゴリC (ATR, ADX) ---
    logging.info(f"Loading S2 Cat C: {S2_CAT_C_PATH}")
    if not S2_CAT_C_PATH.exists():
        raise FileNotFoundError(f"S2 Cat C data not found: {S2_CAT_C_PATH}")

    # Cat Cの全カラムをスキャンして必要なものが存在するか確認
    s2_c_cols = pl.scan_parquet(S2_CAT_C_PATH).collect_schema().names()
    required_s2_c = FEATURES_S2_C + [ATR_COL_NAME]
    missing_s2_c = [col for col in required_s2_c if col not in s2_c_cols]
    if missing_s2_c:
        raise ValueError(f"S2 Cat C data {S2_CAT_C_PATH} is missing: {missing_s2_c}")

    lf_s2_c = pl.scan_parquet(S2_CAT_C_PATH).select(
        pl.col("timestamp").cast(pl.Datetime("us", "UTC")),
        pl.col(ATR_COL_NAME).alias("atr"),  # ATRはHMMと文脈の両方で使う
        pl.col(FEATURES_S2_C),
    )

    # --- 3. カテゴリ2A (Hurst, Kolmogorov) ---
    logging.info(f"Loading S2 Cat 2A: {S2_CAT_2A_PATH}")
    if not S2_CAT_2A_PATH.exists():
        # (仮定が間違っていた場合のエラー)
        logging.error(f"S2 Cat 2A data not found: {S2_CAT_2A_PATH}")
        logging.error(
            "Assumption for Cat 2A filename (features_e2a_D1.parquet) might be wrong."
        )
        raise FileNotFoundError(f"S2 Cat 2A data not found: {S2_CAT_2A_PATH}")

    # Cat 2Aのカラムをスキャンして必要なものが存在するか確認
    s2_2a_cols = pl.scan_parquet(S2_CAT_2A_PATH).collect_schema().names()
    missing_s2_2a = [col for col in FEATURES_S2_2A if col not in s2_2a_cols]
    if missing_s2_2a:
        raise ValueError(f"S2 Cat 2A data {S2_CAT_2A_PATH} is missing: {missing_s2_2a}")

    lf_s2_2a = pl.scan_parquet(S2_CAT_2A_PATH).select(
        pl.col("timestamp").cast(pl.Datetime("us", "UTC")),
        pl.col(FEATURES_S2_2A),
    )

    # --- 4. 3つのS2 LazyFrameを結合 ---
    # Cat C (ATR/ADX) をベースに、Cat A と Cat 2A を結合
    lf_s2_combined = lf_s2_c.join(lf_s2_a, on="timestamp", how="inner").join(
        lf_s2_2a, on="timestamp", how="inner"
    )

    logging.info("S2 features successfully combined (lazily).")
    return lf_s2_combined


def load_s1_data() -> pl.LazyFrame:
    """S1から日足の終値データを読み込み、リターンを計算する"""
    logging.info(f"Loading S1 D1 data (close) from: {S1_D1_PATH}")

    # 修正: .exists() はディレクトリをチェック
    if not S1_D1_PATH.exists() or not S1_D1_PATH.is_dir():
        logging.error(f"S1 D1 partition directory not found at: {S1_D1_PATH}")
        logging.error("Expected structure: .../master_from_tick/timeframe=D1/")
        raise FileNotFoundError(f"S1 D1 partition directory not found at: {S1_D1_PATH}")

    # 修正: ディレクトリ内の全parquetファイルをスキャン
    # (画像では part.0.parquet が見えましたが、globパターン '*.parquet' で安全にすべて読み込みます)
    scan_path = str(S1_D1_PATH / "*.parquet")
    logging.info(f"Scanning S1 D1 data using glob pattern: {scan_path}")

    try:
        lf_s1 = pl.scan_parquet(scan_path).select(
            pl.col("timestamp").cast(pl.Datetime("us", "UTC")), pl.col("close")
        )
    except Exception as e:
        logging.error(f"Failed to scan S1 D1 data at {scan_path}: {e}")
        raise

    # リターン（騰落率）を計算
    lf_s1 = lf_s1.sort("timestamp").with_columns(
        pl.col("close").pct_change().alias("returns")
    )
    return lf_s1


def train_hmm(df_d1: pl.DataFrame) -> GaussianHMM:
    """HMMモデルを訓練し、ディスクに保存する"""
    logging.info("Training HMM model...")
    # HMMの入力として「リターン」と「ボラティリティ(ATR)」を使用
    hmm_input = df_d1.select(["returns", "atr"]).to_numpy()

    # HMMモデルを定義
    model = GaussianHMM(
        n_components=HMM_N_COMPONENTS,
        covariance_type="full",
        random_state=42,
        n_iter=100,
        tol=1e-3,
    )

    try:
        model.fit(hmm_input)
        logging.info("HMM training complete.")

        S7_REGIME_MODEL.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(model, S7_REGIME_MODEL)
        logging.info(f"HMM model saved to: {S7_REGIME_MODEL}")

        return model

    except Exception as e:
        logging.error(f"HMM training failed: {e}", exc_info=True)
        raise


def main():
    """フェーズ1のメイン実行関数"""
    logging.info("### Phase 1: Context Feature Generation (Corrected) ###")

    if S7_CONTEXT_FEATURES.exists() and S7_REGIME_MODEL.exists():
        logging.warning(
            "Context features and HMM model already exist. Skipping generation."
        )
        return

    try:
        # 1. S1 (終値/リターン) と S2 (文脈特徴量/ATR) をロード
        lf_s1 = load_s1_data()
        lf_s2_combined = load_and_prepare_s2_data()

        # 2. S1とS2を結合し、HMM訓練用の完全なD1データフレームを作成
        df_d1 = (
            lf_s1.join(lf_s2_combined, on="timestamp", how="inner")
            .drop_nulls()
            .collect(streaming=True)
        )

        if df_d1.is_empty():
            logging.error("No combined D1 data found after S1/S2 join. Cannot proceed.")
            return

        logging.info(f"Successfully collected {len(df_d1)} rows of combined D1 data.")

        # 3. HMMモデルを訓練
        model = train_hmm(df_d1)

        # 4. HMMのレジーム確率を予測
        logging.info("Predicting HMM regime probabilities...")
        hmm_input_predict = df_d1.select(["returns", "atr"]).to_numpy()
        probabilities = model.predict_proba(hmm_input_predict)

        prob_schema = [f"hmm_prob_{i}" for i in range(HMM_N_COMPONENTS)]
        df_probs = pl.DataFrame(probabilities, schema=prob_schema)

        # 5. 元のD1データとHMM確率を結合
        df_final_context = pl.concat([df_d1, df_probs], how="horizontal")

        # 6. 最終的な文脈特徴量ファイルを作成
        final_columns = (
            ["timestamp"]
            + prob_schema
            + ["atr"]  # 'atr' (リネーム済み)
            + FEATURES_S2_A
            + FEATURES_S2_C
            + FEATURES_S2_2A
        )

        # 念のため、最終カラムリストに重複がないか確認
        final_columns_unique = list(dict.fromkeys(final_columns))

        df_to_save = df_final_context.select(final_columns_unique)

        S7_CONTEXT_FEATURES.parent.mkdir(parents=True, exist_ok=True)
        df_to_save.write_parquet(S7_CONTEXT_FEATURES, compression="zstd")

        logging.info("\n" + "=" * 60)
        logging.info("### Phase 1 COMPLETED! ###")
        logging.info(f"HMM model saved to: {S7_REGIME_MODEL}")
        logging.info(f"Context features saved to: {S7_CONTEXT_FEATURES}")
        logging.info(f"  -> Total context records: {len(df_to_save)}")
        logging.info(f"  -> Columns: {df_to_save.columns}")
        logging.info("=" * 60)

    except FileNotFoundError as e:
        logging.error(f"Failed due to missing input files: {e}")
    except ValueError as e:
        logging.error(f"Failed due to missing columns in input files: {e}")
    except Exception as e:
        logging.error(f"An unexpected error occurred: {e}", exc_info=True)


if __name__ == "__main__":
    main()
