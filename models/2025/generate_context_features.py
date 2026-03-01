# /workspace/models/generate_context_features.py
# [修正版: ATR比率化, Trend Bias追加, 正規パスへ上書き保存]

import sys
import logging
from pathlib import Path

import polars as pl
import numpy as np
import joblib
from hmmlearn.hmm import GaussianHMM
from sklearn.preprocessing import StandardScaler

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
ATR_COL_NAME = "e1c_atr_21"

# S2 (Cat 2A) - 最適化されたウィンドウサイズを選択
# Hurst: 250日 (約1年) - 直近の市場環境（フラクタル性）
# Complexity: 60日 (約3ヶ月) - 直近のトレンド構造の綺麗さ
FEATURES_S2_2A = ["e2a_mfdfa_hurst_mean_250", "e2a_kolmogorov_complexity_60"]


def load_and_prepare_s2_data() -> pl.LazyFrame:
    """S2特徴量の読み込みと結合"""
    logging.info("Loading S2 D1 features from multiple categories...")

    if not S2_CAT_A_PATH.exists():
        raise FileNotFoundError(f"S2 Cat A data not found: {S2_CAT_A_PATH}")
    lf_s2_a = pl.scan_parquet(S2_CAT_A_PATH).select(
        pl.col("timestamp").cast(pl.Datetime("us", "UTC")),
        pl.col(FEATURES_S2_A),
    )

    if not S2_CAT_C_PATH.exists():
        raise FileNotFoundError(f"S2 Cat C data not found: {S2_CAT_C_PATH}")
    lf_s2_c = pl.scan_parquet(S2_CAT_C_PATH).select(
        pl.col("timestamp").cast(pl.Datetime("us", "UTC")),
        pl.col(ATR_COL_NAME).alias("atr"),
        pl.col(FEATURES_S2_C),
    )

    if not S2_CAT_2A_PATH.exists():
        raise FileNotFoundError(f"S2 Cat 2A data not found: {S2_CAT_2A_PATH}")
    lf_s2_2a = pl.scan_parquet(S2_CAT_2A_PATH).select(
        pl.col("timestamp").cast(pl.Datetime("us", "UTC")),
        pl.col(FEATURES_S2_2A),
    )

    lf_s2_combined = lf_s2_c.join(lf_s2_a, on="timestamp", how="inner").join(
        lf_s2_2a, on="timestamp", how="inner"
    )

    logging.info("S2 features successfully combined (lazily).")
    return lf_s2_combined


def load_s1_data() -> pl.LazyFrame:
    """S1データの読み込み"""
    logging.info(f"Loading S1 D1 data (close) from: {S1_D1_PATH}")

    scan_path = str(S1_D1_PATH / "*.parquet")
    logging.info(f"Scanning S1 D1 data using glob pattern: {scan_path}")

    lf_s1 = pl.scan_parquet(scan_path).select(
        pl.col("timestamp").cast(pl.Datetime("us", "UTC")), pl.col("close")
    )

    lf_s1 = lf_s1.sort("timestamp").with_columns(
        [
            pl.col("close").pct_change().alias("returns"),
            (
                (pl.col("close") - pl.col("close").rolling_mean(window_size=25))
                / pl.col("close").rolling_mean(window_size=25)
            ).alias("trend_bias_25"),
        ]
    )

    return lf_s1


def train_hmm(df_d1: pl.DataFrame) -> GaussianHMM:
    """HMMモデルを訓練し、ディスクに保存する"""
    logging.info("Training HMM model (v5: Dead Market Filtered)...")

    hmm_input = df_d1.select(["returns_scaled", "log_atr_scaled"]).to_numpy()

    model = GaussianHMM(
        n_components=HMM_N_COMPONENTS,
        covariance_type="full",
        random_state=42,
        n_iter=100,
        tol=1e-3,
        verbose=True,
    )

    try:
        model.fit(hmm_input)
        logging.info("HMM training complete.")

        # 平均値を確認（これでプラスとマイナス、あるいは大小に分かれるか確認）
        logging.info(f"Trained Means (State 0): {model.means_[0]}")
        logging.info(f"Trained Means (State 1): {model.means_[1]}")

        S7_REGIME_MODEL.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(model, S7_REGIME_MODEL)
        logging.info(f"HMM model saved to: {S7_REGIME_MODEL}")

        return model

    except Exception as e:
        logging.error(f"HMM training failed: {e}", exc_info=True)
        raise


def main():
    """フェーズ1のメイン実行関数 (v5)"""
    logging.info(
        "### Phase 1: Context Feature Generation (v5: Filter Dead Markets) ###"
    )

    try:
        # 1. ロード
        lf_s1 = load_s1_data()
        lf_s2_combined = load_and_prepare_s2_data()

        # 2. 結合 & 基本指標計算
        df_d1_raw = (
            lf_s1.join(lf_s2_combined, on="timestamp", how="inner")
            .with_columns((pl.col("atr") / pl.col("close") * 100).alias("atr_ratio"))
            .drop_nulls()
            .collect(streaming=True)
        )

        if df_d1_raw.is_empty():
            logging.error("No combined D1 data found. Cannot proceed.")
            return

        logging.info(f"Collected {len(df_d1_raw)} rows.")

        # --- [NEW] Dead Market Filtering ---
        # ATR比率が0.1%未満（＝価格がほとんど動いていない死んだ相場）を除外
        # これをやらないとHMMが「死んだ相場」を見つけることに全力を出してしまう
        initial_count = len(df_d1_raw)
        df_d1_filtered = df_d1_raw.filter(pl.col("atr_ratio") > 0.1)
        filtered_count = len(df_d1_filtered)

        logging.info(
            f"Filtered out {initial_count - filtered_count} 'dead market' rows (ATR < 0.1%)."
        )
        logging.info(f"Remaining rows for training: {filtered_count}")

        if filtered_count < 100:
            logging.error("Too few rows remaining after filtering. Check data quality.")
            return

        # --- 3. データ変換 ---

        # A. 対数変換
        df_d1 = df_d1_filtered.with_columns(
            (pl.col("atr_ratio") + 1e-9).log().alias("log_atr")
        )

        # B. 標準化
        scaler = StandardScaler()
        features_to_scale = df_d1.select(["returns", "log_atr"]).to_numpy()
        scaled_features = scaler.fit_transform(features_to_scale)

        # DataFrameに戻す
        df_d1 = df_d1.with_columns(
            [
                pl.Series("returns_scaled", scaled_features[:, 0]),
                pl.Series("log_atr_scaled", scaled_features[:, 1]),
            ]
        )

        # 4. HMMモデルを訓練
        model = train_hmm(df_d1)

        # 5. HMMのレジーム確率を予測
        logging.info("Predicting HMM regime probabilities...")
        hmm_input_predict = df_d1.select(
            ["returns_scaled", "log_atr_scaled"]
        ).to_numpy()
        probabilities = model.predict_proba(hmm_input_predict)

        prob_schema = [f"hmm_prob_{i}" for i in range(HMM_N_COMPONENTS)]
        df_probs = pl.DataFrame(probabilities, schema=prob_schema)

        # 6. 結合
        df_final_context = pl.concat([df_d1, df_probs], how="horizontal")

        # 7. 保存
        final_columns = (
            ["timestamp"]
            + prob_schema
            + ["atr_ratio"]
            + ["trend_bias_25"]
            + ["atr"]
            + FEATURES_S2_A
            + FEATURES_S2_C
            + FEATURES_S2_2A
        )

        final_columns_unique = list(dict.fromkeys(final_columns))
        df_to_save = df_final_context.select(final_columns_unique)

        S7_CONTEXT_FEATURES.parent.mkdir(parents=True, exist_ok=True)
        # 上書き保存
        df_to_save.write_parquet(S7_CONTEXT_FEATURES, compression="zstd")

        logging.info("\n" + "=" * 60)
        logging.info("### Phase 1 COMPLETED! (v5) ###")
        logging.info(f"HMM model saved to: {S7_REGIME_MODEL}")
        logging.info(f"Context features saved to: {S7_CONTEXT_FEATURES}")
        logging.info(f"  -> Total context records: {len(df_to_save)}")
        logging.info("=" * 60)

    except Exception as e:
        logging.error(f"An error occurred: {e}", exc_info=True)


if __name__ == "__main__":
    main()
