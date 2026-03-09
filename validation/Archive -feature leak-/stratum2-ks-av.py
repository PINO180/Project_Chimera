import sys
from pathlib import Path
import json
import pyarrow.parquet as pq
from tqdm import tqdm
import logging
import time

# --- ロギングとBlueprintの設定 ---
sys.path.append(str(Path(__file__).resolve().parents[1]))
try:
    from blueprint import S2_FEATURES_FIXED, S3_ARTIFACTS
except ImportError:
    print("WARNING: blueprint.py not found. Using hardcoded paths.")
    S2_FEATURES_FIXED = Path("/workspace/data/XAUUSD/stratum_2_features_fixed")
    S3_ARTIFACTS = Path("/workspace/data/XAUUSD/stratum_3_artifacts/1A_2B")

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def get_all_feature_names(base_path: Path) -> set:
    """
    stratum_2から全データソースをスキャンし、サフィックスを付与して
    ユニークな全特徴量名（約8214個）のセットを作成する。
    """
    all_features = set()
    non_feature_columns = {
        "timestamp",
        "open",
        "high",
        "low",
        "close",
        "volume",
        "timeframe",
    }

    all_data_sources = []
    for engine_dir in base_path.iterdir():
        if engine_dir.is_dir():
            all_data_sources.extend(
                item
                for item in engine_dir.iterdir()
                if item.is_dir() or item.suffix == ".parquet"
            )

    logger.info(
        f"Scanning {len(all_data_sources)} data sources to build full feature list..."
    )
    for path in tqdm(all_data_sources, desc="Building feature universe"):
        try:
            # タイムフレームサフィックスを決定
            if path.is_dir() and "_tick" in path.name:
                timeframe_suffix = "_tick"
            elif path.is_file():
                timeframe_suffix = f"_{path.stem.split('_')[-1]}"
            else:  # tick以外のディレクトリ
                timeframe_suffix = f"_{path.name.split('_')[-1]}"

            # スキーマ読み込み
            if path.is_dir():
                first_file = next(path.glob("**/*.parquet"), None)
                if not first_file:
                    continue
                schema = pq.read_schema(first_file)
            else:
                schema = pq.read_schema(path)

            # サフィックスを付与してセットに追加
            for col in schema.names:
                if col not in non_feature_columns:
                    all_features.add(f"{col}{timeframe_suffix}")
        except Exception as e:
            logger.warning(f"Could not process {path.name}: {e}")

    return all_features


def load_unstable_set(file_path: Path) -> set:
    """JSONファイルから不安定特徴量のセットを読み込む。"""
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return set(data.get("unstable_features", []))
    except FileNotFoundError:
        logger.error(f"File not found: {file_path}")
        return set()


def main():
    logger.info("--- Creating Final Survivor List (Post-KS & Post-AV) ---")

    # 1. 元となる全特徴量リストを作成
    all_features = get_all_feature_names(S2_FEATURES_FIXED)
    logger.info(f"Total unique features found in stratum_2: {len(all_features)}")

    # 2. 脱落者リストを読み込み
    ks_unstable = load_unstable_set(S3_ARTIFACTS / "ks_unstable_features.json")
    av_unstable = load_unstable_set(S3_ARTIFACTS / "av_unstable_features.json")
    total_unstable = ks_unstable | av_unstable  # 和集合で重複を排除

    logger.info(f"Unstable features from KS test: {len(ks_unstable)}")
    logger.info(f"Unstable features from AV test: {len(av_unstable)}")
    logger.info(f"Total unique unstable features: {len(total_unstable)}")

    # 3. 生存者を計算 (差集合)
    survivors = sorted(list(all_features - total_unstable))

    # 4. 結果をファイルに保存
    output_path = S3_ARTIFACTS / "final_survivor_feature_list.txt"
    with open(output_path, "w", encoding="utf-8") as f:
        for feature_name in survivors:
            f.write(f"{feature_name}\n")

    logger.info("-" * 50)
    logger.info("✅ Final Survivor List Created Successfully!")
    logger.info(f"   Total Survivors: {len(survivors)}")
    logger.info(f"   List saved to: {output_path}")
    logger.info("-" * 50)


if __name__ == "__main__":
    main()
