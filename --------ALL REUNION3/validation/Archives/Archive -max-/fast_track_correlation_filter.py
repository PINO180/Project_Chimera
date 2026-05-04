"""
fast_track_correlation_filter.py
第一段階（Fast Track）専用：超軽量・多重共線性フィルター

目的：
- 299個の精鋭リストから、「完全に動きが同じ（相関0.98以上）」双子特徴量だけを弾く。
- 厳しいふるい落としはせず、AIの表現力を最大限に残す。
"""

import sys
from pathlib import Path
import polars as pl
import pandas as pd
import numpy as np
import argparse
import re
from tqdm import tqdm

# プロジェクトのルートディレクトリをPythonの検索パスに追加
sys.path.append(str(Path(__file__).resolve().parents[1]))

# blueprintからパスを読み込む（環境に合わせて適宜調整してください）
from blueprint import S2_FEATURES_FIXED, S2_FEATURES_AFTER_AV, S3_ARTIFACTS

# --- 入出力ファイルの設定 ---
INPUT_FEATURE_LIST = S3_ARTIFACTS / "survived_299_features.txt"
OUTPUT_FEATURE_LIST = S3_ARTIFACTS / "fast_track_filtered_features.txt"

# サンプリング行数
SAMPLE_SIZE = 100_000


def parse_suffixed_feature_name(suffixed_name: str) -> tuple[str, str, str]:
    parts = suffixed_name.split("_")
    timeframe_pattern = re.compile(r"^(M[0-9\.]+|H[0-9]+|D[0-9]+|W[0-9]+|MN|tick)$")
    if (
        parts[0].startswith("e")
        and len(parts) > 2
        and timeframe_pattern.match(parts[-1])
    ):
        return parts[0], "_".join(parts[:-1]), parts[-1]
    if len(parts) > 1 and timeframe_pattern.match(parts[-1]):
        return "base", "_".join(parts[:-1]), parts[-1]
    return suffixed_name, suffixed_name, "unknown"


def find_feature_files(features: list[str]) -> dict[Path, list[str]]:
    print("特徴量のソースファイルを検索しています...")
    features_by_file = {}

    search_dirs = [S2_FEATURES_AFTER_AV, S2_FEATURES_FIXED]
    all_sources = []
    for d in search_dirs:
        if d.exists():
            # 修正: rglob("*") で parquetファイルだけでなく、tickのディレクトリ自体も取得
            all_sources.extend(list(d.rglob("*")))

    all_sources_with_schema = {}
    for source_path in all_sources:
        try:
            # 修正: 05スクリプトと同じ安全な判定で、ファイルとディレクトリを正確に判別
            if source_path.is_file() and source_path.name.endswith(".parquet"):
                schema = set(pl.read_parquet_schema(source_path).keys())
                all_sources_with_schema[source_path] = schema
            elif source_path.is_dir() and "tick" in source_path.name:
                first_parquet = next(source_path.rglob("*.parquet"), None)
                if first_parquet:
                    schema = set(pl.read_parquet_schema(first_parquet).keys())
                    all_sources_with_schema[source_path] = schema
        except Exception:
            continue

    for suffixed_feature in features:
        _, base_name, timeframe = parse_suffixed_feature_name(suffixed_feature)
        found_path = None
        for source_path, schema in all_sources_with_schema.items():
            path_name = source_path.name
            # 修正: 'tick' が path_name ('features_e1a_tick'など) に含まれているかで判定
            if base_name in schema and timeframe in path_name:
                found_path = source_path
                break

        if found_path:
            if found_path not in features_by_file:
                features_by_file[found_path] = []
            features_by_file[found_path].append(suffixed_feature)
        else:
            print(f"Warning: '{suffixed_feature}' のソースファイルが見つかりません。")

    return features_by_file


def main(threshold: float):
    print("=" * 60)
    print("### Fast Track: Lightweight Correlation Filter ###")
    print(f"相関閾値: {threshold} (これ以上の相関を持つ双子特徴量を間引きます)")
    print("=" * 60)

    if not INPUT_FEATURE_LIST.exists():
        raise FileNotFoundError(f"入力リストが見つかりません: {INPUT_FEATURE_LIST}")

    with open(INPUT_FEATURE_LIST, "r") as f:
        # 自動クリーンアップ
        features = [
            line.strip().replace("_neutralized", "").replace("", "")
            for line in f
            if line.strip()
        ]
    features = sorted(list(set(features)))

    print(f"読み込んだ特徴量数（クリーンアップ後）: {len(features)}")

    if len(features) == 0:
        print("特徴量がありません。終了します。")
        return

    features_by_file = find_feature_files(features)

    print(f"\nデータソースから {SAMPLE_SIZE} 行ずつサンプリングして結合します...")
    processed_dfs = []

    for file_path, suffixed_features in tqdm(
        features_by_file.items(), desc="データのロード"
    ):
        base_names = list(
            set([parse_suffixed_feature_name(f)[1] for f in suffixed_features])
        )
        cols_to_select = ["timestamp"] + base_names

        if file_path.is_dir() and "tick" in file_path.name:
            # --- 【修正】Tick (Hiveパーティション) 用の特殊高速ロード処理 ---
            files = list(file_path.rglob("*.parquet"))

            def extract_hive_date(p: Path) -> tuple:
                y, m, d = 0, 0, 0
                for part in p.parts:
                    if part.startswith("year="):
                        y = int(part.split("=")[1])
                    elif part.startswith("month="):
                        m = int(part.split("=")[1])
                    elif part.startswith("day="):
                        d = int(part.split("=")[1])
                return (y, m, d)

            files.sort(key=extract_hive_date)

            # 最新のパーティションから遡って必要行数を満たすまでロード（メモリ爆発防止）
            temp_dfs = []
            total_rows = 0
            for f in reversed(files):
                try:
                    df_part = pl.read_parquet(f, columns=cols_to_select).to_pandas()
                    temp_dfs.append(df_part)
                    total_rows += len(df_part)
                    if total_rows >= SAMPLE_SIZE:
                        break
                except Exception as e:
                    print(f"Warning: {f.name} の読み込みに失敗しました ({e})")

            if temp_dfs:
                # 遡って取得したので、時間順になるように逆順にして結合
                df = pd.concat(temp_dfs[::-1], ignore_index=True).tail(SAMPLE_SIZE)
            else:
                df = pd.DataFrame(columns=cols_to_select)
        else:
            # --- 非Tick用の通常ロード処理 ---
            lf = pl.scan_parquet(file_path).select(cols_to_select).tail(SAMPLE_SIZE)
            df = lf.collect().to_pandas()

        # サフィックス付きの名前にリネーム
        rename_dict = {}
        for sf in suffixed_features:
            _, b_name, _ = parse_suffixed_feature_name(sf)
            rename_dict[b_name] = sf

        cols_to_keep = ["timestamp"] + list(rename_dict.keys())
        df = df[cols_to_keep].rename(columns=rename_dict)
        processed_dfs.append(df)

    if not processed_dfs:
        print("有効なデータがありません。終了します。")
        return

    print("\nデータをタイムスタンプで結合中...")
    final_df = processed_dfs[0]
    for df_to_join in processed_dfs[1:]:
        final_df = pd.merge_asof(
            final_df.sort_values("timestamp"),
            df_to_join.sort_values("timestamp"),
            on="timestamp",
            direction="backward",
        )

    print("相関行列を計算中（これには少し時間がかかります）...")
    feature_cols = [c for c in final_df.columns if c != "timestamp"]

    # --- ★ここから追加: タイムフレームの短い順に並べ替える（短い方を優先して残すため） ---
    def get_tf_weight(feat_name):
        tf = feat_name.split("_")[-1]
        weight_map = {
            "tick": 0,
            "M0.5": 0.5,
            "M1": 1,
            "M3": 3,
            "M5": 5,
            "M8": 8,
            "M15": 15,
            "M30": 30,
            "H1": 60,
            "H4": 240,
            "H6": 360,
            "H12": 720,
            "D1": 1440,
            "W1": 10080,
            "MN": 43200,
        }
        return weight_map.get(tf, 999999)  # 未知のサフィックスは一番後ろに回す

    feature_cols = sorted(feature_cols, key=get_tf_weight)
    # --- ★ここまで追加 ---

    # 並べ替えた順番で相関行列を計算（これにより、後から出てくるD1などが削除対象になる）
    corr_matrix = final_df[feature_cols].fillna(0).corr().abs()
    upper_tri = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))

    to_drop = [
        column for column in upper_tri.columns if any(upper_tri[column] >= threshold)
    ]
    survivors = [f for f in feature_cols if f not in to_drop]

    print("\n" + "=" * 60)
    print(f"間引かれた特徴量数 (相関 >= {threshold}): {len(to_drop)}")
    print(f"生き残った特徴量数: {len(survivors)}")
    print("=" * 60)

    OUTPUT_FEATURE_LIST.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_FEATURE_LIST, "w") as f:
        for feature in sorted(survivors):
            f.write(f"{feature}\n")

    print(
        f"\nFast Track用のクリーンな特徴量リストを保存しました: {OUTPUT_FEATURE_LIST}"
    )
    print(
        "-> このリストを修正済みの `05_alpha_decay_analyzer.py` に直接渡してください！"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fast Track Correlation Filter")
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.95,
        help="相関係数の閾値 (デフォルト: 0.98)",
    )
    args = parser.parse_args()

    main(args.threshold)
