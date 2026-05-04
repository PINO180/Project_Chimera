"""
2_G_alpha_neutralizer.py
========================
Chapter 2 - Alpha Neutralization

旧 05_alpha_decay_analyzer.py と 05_alpha_decay_analyzer_stage_first.py を統合。
時間足スケール別プロキシを使用した純化処理を実装する。

【設計変更】
- 旧設計: 全特徴量をM5プロキシで固定
- 新設計: blueprintの NEUTRALIZATION_CONFIG に従い、時間足グループごとに
          プロキシ時間足とローリングウィンドウを切り替える

  グループ    | プロキシ | ウィンドウ
  ------------|----------|----------
  HF (H1以下) | M5       | 2016
  LF_SHORT    | H4       | 504
  LF_MID      | D1       | 90
  LF_LONG     | W1/MN    | 52

- データ型: このスクリプトのみ例外的に Float64 を維持する
  （ベータ推定の分子・分母が非常に小さくなりうるため、Float32では丸め誤差が蓄積する）
"""

import sys
import re
import shutil
import warnings
import argparse
from pathlib import Path
from typing import Optional

# ルール7: 動的パス解決（ハードコード禁止）
sys.path.append(str(Path(__file__).resolve().parents[1]))

import polars as pl
from tqdm import tqdm

import blueprint


# =============================================================================
# ヘルパー関数
# =============================================================================


def parse_suffixed_feature_name(suffixed_name: str) -> tuple[str, str, str]:
    """
    サフィックス付き特徴量名を (engine_id, base_name, timeframe) に分解する。

    例:
      "e1a_some_feature_M5" → ("e1a", "e1a_some_feature", "M5")
      "some_feature_H4"     → ("base", "some_feature", "H4")
    """
    parts = suffixed_name.split("_")
    timeframe_pattern = re.compile(
        r"^(M[0-9\.]+|H[0-9]+|D[0-9]+|W[0-9]+|MN|tick)$"
    )
    if (
        parts[0].startswith("e")
        and len(parts) > 2
        and timeframe_pattern.match(parts[-1])
    ):
        engine_id = parts[0]
        base_name = "_".join(parts[:-1])
        timeframe = parts[-1]
        return engine_id, base_name, timeframe
    if len(parts) > 1 and timeframe_pattern.match(parts[-1]):
        base_name = "_".join(parts[:-1])
        timeframe = parts[-1]
        return "base", base_name, timeframe
    return suffixed_name, suffixed_name, "unknown"


def get_group(tf: str) -> str:
    """
    時間足文字列から neutralization グループを返す。
    blueprintの HF_TIMEFRAMES / LF_*_TIMEFRAMES を参照する。
    """
    if tf in blueprint.HF_TIMEFRAMES:
        return "HF"
    elif tf in blueprint.LF_SHORT_TIMEFRAMES:
        return "LF_SHORT"
    elif tf in blueprint.LF_MID_TIMEFRAMES:
        return "LF_MID"
    elif tf in blueprint.LF_LONG_TIMEFRAMES:
        return "LF_LONG"
    else:
        raise ValueError(f"未知の時間足: {tf}")


def find_proxy_file(proxy_tf: str) -> Optional[Path]:
    """
    S2_FEATURES_VALIDATED からプロキシ時間足のファイルを動的検索する。
    ハードコードパス禁止（ルール7）に従い、glob で探す。
    tick ファイルは除外する。
    """
    for p in blueprint.S2_FEATURES_VALIDATED.rglob(f"*_{proxy_tf}.parquet"):
        if "tick" in str(p):
            continue
        return p
    return None


def build_proxy_lazyframe(proxy_tf: str) -> Optional[pl.LazyFrame]:
    """
    プロキシ時間足の close リターン系列を LazyFrame として返す。

    close は S1_PROCESSED から取得する。
    S2_FEATURES_VALIDATED の特徴量ファイルには close が存在しないファイルがあるため、
    S2_FEATURES_VALIDATED からは取得しない。

    close リターン = close / close.shift(1) - 1（1バー前比）
    shift(正の数) で過去参照のみ（未来情報リーク排除: ルール1）
    """
    price_dir = blueprint.S1_PROCESSED / f"timeframe={proxy_tf}"
    if not price_dir.exists():
        print(f"[WARN] S1_PROCESSED に {proxy_tf} が見つかりません: {price_dir}")
        return None

    col_name = f"proxy_{proxy_tf}"
    lf = (
        pl.scan_parquet(str(price_dir / "*.parquet"))
        .select(["timestamp", "close"])
        .with_columns(pl.col("timestamp").cast(pl.Datetime("us")))  # S1はns、S2はus — 統一
        .sort("timestamp")
        .with_columns(
            # 過去1バーに対するリターン（shift(1) = 1バー前 = 過去方向のみ: ルール1）
            (pl.col("close") / pl.col("close").shift(1) - 1).alias(col_name)
        )
        .select(["timestamp", col_name])
    )
    return lf


def find_features_by_file(
    final_features: list[str],
) -> dict[Path, list[str]]:
    """
    特徴量名リストを、それぞれの元ファイル（S2_FEATURES_VALIDATED）にマッピングする。
    engine_id・timeframe・base_name の3条件で精密にマッチングする。
    """
    print("Mapping features to source files (S2_FEATURES_VALIDATED)...")
    features_by_file: dict[Path, list[str]] = {}
    all_sources_with_schema: dict[Path, set[str]] = {}

    all_sources = list(blueprint.S2_FEATURES_VALIDATED.rglob("*"))
    for source_path in tqdm(all_sources, desc="Caching source schemas"):
        try:
            schema: set[str] = set()
            if source_path.is_file() and source_path.name.endswith(".parquet"):
                schema = set(pl.read_parquet_schema(source_path).keys())
            elif source_path.is_dir() and "tick" in source_path.name:
                first_parquet = next(source_path.rglob("*.parquet"), None)
                if first_parquet:
                    schema = set(pl.read_parquet_schema(first_parquet).keys())
            if schema:
                all_sources_with_schema[source_path] = schema
        except Exception:
            continue

    print(f"Searching {len(final_features)} features across {len(all_sources_with_schema)} source files...")
    for suffixed_feature in tqdm(final_features, desc="Mapping features"):
        engine_id, base_name, timeframe = parse_suffixed_feature_name(suffixed_feature)
        if timeframe == "unknown":
            print(f"[WARN] Could not determine timeframe for '{suffixed_feature}'. Skipping.")
            continue

        found_path: Optional[Path] = None
        for source_path, schema in all_sources_with_schema.items():
            path_str = source_path.name
            if base_name not in schema:
                continue
            if timeframe not in path_str:
                continue
            if engine_id != "base" and engine_id not in path_str:
                continue
            found_path = source_path
            break

        if found_path is not None:
            features_by_file.setdefault(found_path, []).append(suffixed_feature)
        else:
            print(f"[WARN] Could not find source file for '{suffixed_feature}'.")

    print(
        f"Grouped {len(final_features)} features into {len(features_by_file)} source files."
    )
    return features_by_file


def get_output_path(input_path: Path) -> Path:
    """
    入力パス（S2_FEATURES_VALIDATED 配下）から出力パス（S5_NEUTRALIZED_ALPHA_SET 配下）を生成する。
    """
    relative_path = input_path.relative_to(blueprint.S2_FEATURES_VALIDATED)
    output_name = (
        input_path.name.replace(".parquet", "_neutralized.parquet")
        if input_path.is_file()
        else f"{input_path.name}_neutralized"
    )
    return blueprint.S5_NEUTRALIZED_ALPHA_SET / relative_path.parent / output_name


# =============================================================================
# 純化式（Chapter4と完全一致）
# =============================================================================


def neutralize_lazyframe(
    lf: pl.LazyFrame,
    feature_names: list[str],
    proxy_col: str,
    window: int,
) -> pl.LazyFrame:
    """
    ローリング共分散の数学的展開を用いた純化処理。

    【数学的定義】
    mean_x  = E[X]  (rolling)
    mean_x2 = E[X²] (rolling)
    var_x   = E[X²] - E[X]²  (clip≥0 でゼロ除算を防止: ルール4)

    mean_y  = E[Y]  (rolling)
    mean_xy = E[XY] (rolling)
    cov_xy  = E[XY] - E[X]E[Y]

    rolling_beta  = cov_xy / (var_x + 1e-10)   (ルール4: ゼロ除算保護)
    rolling_alpha = mean_y - rolling_beta * mean_x
    neutralized   = Y - (rolling_beta * X + rolling_alpha)

    ★ Chapter4の _calculate_neutralized_features と同一の計算式 ★

    min_periods=30 は Chapter4 の実装と完全一致させる（値を変更しない）。

    データ型: ルール8の例外 — このスクリプトのみ Float64 を維持する。
    ベータ推定の分子・分母が非常に小さい値になりうるため、Float32（有効桁7桁）では
    丸め誤差が蓄積する可能性がある。
    """
    MIN_SAMPLES = 30  # Chapter4と完全一致させる固定値（Polars 1.21.0以降: min_samples）

    neutralization_exprs = []

    # プロキシ(X): null を 0 で埋める（欠損は中立=0リターンとして扱う）
    x = pl.col(proxy_col).fill_null(0.0)

    # X のローリング統計（未来参照なし: ルール1）
    x_mean = x.rolling_mean(window_size=window, min_samples=MIN_SAMPLES)
    # E[X²] - E[X]² によるローリング分散、負値はclipでゼロに（ルール4）
    var_x = (
        (x * x).rolling_mean(window_size=window, min_samples=MIN_SAMPLES)
        - x_mean * x_mean
    ).clip(lower_bound=0.0)

    for suffixed_feature in feature_names:
        _, base_name, _ = parse_suffixed_feature_name(suffixed_feature)

        # 特徴量(Y): F32で保存されている場合に alpha との引き算で完全相殺が起きるため
        # Float64 にキャストしてから null を 0 で埋める（ルール8: 2_G のみ Float64 維持）
        y = pl.col(base_name).cast(pl.Float64).fill_null(0.0)
        y_mean = y.rolling_mean(window_size=window, min_samples=MIN_SAMPLES)

        # ローリング共分散: E[XY] - E[X]E[Y]（ルール1: 過去方向のみ）
        cov_xy = (
            (x * y).rolling_mean(window_size=window, min_samples=MIN_SAMPLES)
            - x_mean * y_mean
        )

        # ベータ・アルファ推定
        # +1e-10 でゼロ除算を保護（ルール4）
        rolling_beta = cov_xy / (var_x + 1e-10)
        rolling_alpha = y_mean - rolling_beta * x_mean

        # 残差（純化アルファ）
        # null が伝播しないよう fill_null(0) で安全化（ルール4）
        residual_expr = (
            y - (rolling_beta.fill_null(0.0) * x + rolling_alpha.fill_null(0.0))
        ).alias(f"{base_name}_neutralized")

        neutralization_exprs.append(residual_expr)

    return lf.with_columns(neutralization_exprs)


# =============================================================================
# メイン処理
# =============================================================================


def main(test_mode: bool) -> None:
    print("=" * 70)
    print("### 2_G_alpha_neutralizer.py: Multi-Timeframe Alpha Neutralization ###")
    print("=" * 70)

    # 出力ディレクトリの準備
    blueprint.S5_ALPHA.mkdir(parents=True, exist_ok=True)
    # [Phase 6] 起動時の全削除を resume モードに変更
    # 既存の有効な出力 (size > 0) はスキップ、0KB / 不在のみ処理する
    # ★ 完全に最初からやり直したい場合は --clean フラグを付ける
    clean_flag = getattr(main, "_clean_flag", False)
    if blueprint.S5_NEUTRALIZED_ALPHA_SET.exists():
        if clean_flag:
            print(f"[INFO] --clean: Removing existing output: {blueprint.S5_NEUTRALIZED_ALPHA_SET}")
            shutil.rmtree(blueprint.S5_NEUTRALIZED_ALPHA_SET)
            blueprint.S5_NEUTRALIZED_ALPHA_SET.mkdir(parents=True, exist_ok=True)
        else:
            print(f"[INFO] Resume mode: existing output kept at {blueprint.S5_NEUTRALIZED_ALPHA_SET}")
            print(f"[INFO] 0-byte (failed) files will be removed and reprocessed")
            # 0 KB の失敗ファイルだけは削除
            for p in blueprint.S5_NEUTRALIZED_ALPHA_SET.rglob("*.parquet"):
                try:
                    if p.stat().st_size == 0:
                        print(f"  [INFO] Removing 0-byte file: {p}")
                        p.unlink()
                except Exception:
                    pass
    else:
        blueprint.S5_NEUTRALIZED_ALPHA_SET.mkdir(parents=True, exist_ok=True)

    # ----------------------------------------------------------------
    # 1. 特徴量リストの読み込み（S3_FEATURES_FOR_ALPHA_DECAY）
    # ----------------------------------------------------------------
    feature_list_path = blueprint.S3_FEATURES_FOR_ALPHA_DECAY
    if not feature_list_path.exists():
        raise FileNotFoundError(f"Feature list not found: {feature_list_path}")

    with open(feature_list_path, "r") as f:
        final_features = [line.strip() for line in f if line.strip()]
    print(f"[INFO] Loaded {len(final_features)} features from {feature_list_path}")

    # ----------------------------------------------------------------
    # 2. 特徴量 → ソースファイルのマッピング
    # ----------------------------------------------------------------
    features_by_source_file = find_features_by_file(final_features)

    # ----------------------------------------------------------------
    # 3. プロキシ LazyFrame のキャッシュ（時間足ごとに1度だけ構築）
    # ----------------------------------------------------------------
    proxy_cache: dict[str, pl.LazyFrame] = {}
    for group, cfg in blueprint.NEUTRALIZATION_CONFIG.items():
        proxy_tf = cfg["proxy_tf"]
        if proxy_tf in proxy_cache:
            continue
        print(f"[INFO] Building proxy LazyFrame: {proxy_tf} (for group={group})")
        lf = build_proxy_lazyframe(proxy_tf)
        if lf is None:
            print(f"[WARN] Proxy file for {proxy_tf} not found in S2_FEATURES_VALIDATED. "
                  f"Features in group {group} may be skipped.")
        else:
            proxy_cache[proxy_tf] = lf

    # ----------------------------------------------------------------
    # 4. 純化処理ループ
    # ----------------------------------------------------------------
    source_files_to_process = list(features_by_source_file.keys())
    if test_mode:
        print(f"\n[TEST MODE] Processing only the first 2 source files.")
        source_files_to_process = source_files_to_process[:2]

    total = len(source_files_to_process)
    print(f"\n[INFO] Starting neutralization for {total} source files...\n")

    for i, input_path in enumerate(source_files_to_process):
        features_in_file = features_by_source_file[input_path]
        print(f"--- [{i + 1}/{total}] {input_path.name} ({len(features_in_file)} features) ---")

        # [Phase 6] resume サポート: 既に有効な出力 (size > 0) があればスキップ
        skip_output_path = get_output_path(input_path)
        if (
            skip_output_path.exists()
            and not skip_output_path.is_dir()
            and skip_output_path.stat().st_size > 0
        ):
            print(f"  [SKIP] Already exists: {skip_output_path} ({skip_output_path.stat().st_size:,} bytes)")
            continue
        # tick データ (Hive ディレクトリ) の場合は中身があるか確認
        if skip_output_path.exists() and skip_output_path.is_dir():
            existing_files = list(skip_output_path.rglob("*.parquet"))
            if existing_files and all(p.stat().st_size > 0 for p in existing_files):
                print(f"  [SKIP] Already exists (Hive): {skip_output_path} ({len(existing_files)} files)")
                continue

        # 4-a. 特徴量をグループ別に分類
        groups_in_file: dict[str, list[str]] = {}
        for suffixed_feature in features_in_file:
            _, _, tf = parse_suffixed_feature_name(suffixed_feature)
            if tf == "unknown":
                print(f"  [WARN] Unknown timeframe for '{suffixed_feature}'. Skipping.")
                continue
            try:
                group = get_group(tf)
            except ValueError as e:
                print(f"  [WARN] {e}. Skipping.")
                continue
            groups_in_file.setdefault(group, []).append(suffixed_feature)

        if not groups_in_file:
            print("  [WARN] No valid features found in this file. Skipping.")
            continue

        # 4-b. ソースデータの読み込み
        base_feature_names = list(
            set(parse_suffixed_feature_name(f)[1] for f in features_in_file)
        )
        lf_base = (
            pl.scan_parquet(input_path)
            .select(["timestamp"] + base_feature_names)
            .sort("timestamp")
        )

        # 4-c. ログ出力とスキップ判定（純化計算は 4-d で一度だけ実行）
        all_neutralized_exprs = []

        for group, group_features in groups_in_file.items():
            cfg = blueprint.NEUTRALIZATION_CONFIG[group]
            proxy_tf: str = cfg["proxy_tf"]

            if proxy_tf not in proxy_cache:
                print(f"  [WARN] Proxy {proxy_tf} not available. Skipping group={group}.")
                continue

            print(f"  [INFO] Group={group}, proxy={proxy_tf}, window={cfg['window']}, "
                  f"features={len(group_features)}")

            for suffixed_feature in group_features:
                _, base_name, _ = parse_suffixed_feature_name(suffixed_feature)
                all_neutralized_exprs.append(f"{base_name}_neutralized")

        # all_neutralized_exprs が空なら次ファイルへ
        if not all_neutralized_exprs:
            print("  [WARN] No neutralized expressions generated. Skipping.")
            continue

        # 4-d. 出力列の選択
        # 最後に処理したグループの neutralized_lf に全グループの結果が含まれるわけではないため、
        # グループごとに独立して収集してから結合する方式を採用する。
        # ここでは各グループのLFを独立して収集し、timestamp でマージする。

        output_path = get_output_path(input_path)
        print(f"  [INFO] Output → {output_path}")

        # グループをまたいだ最終結果の構築
        # 全グループの純化結果を timestamp 基準で水平結合する
        group_dfs: list[pl.LazyFrame] = []

        for group, group_features in groups_in_file.items():
            cfg = blueprint.NEUTRALIZATION_CONFIG[group]
            proxy_tf = cfg["proxy_tf"]
            window = cfg["window"]
            proxy_col = f"proxy_{proxy_tf}"

            if proxy_tf not in proxy_cache:
                continue

            proxy_lf = proxy_cache[proxy_tf]
            lf_with_proxy = lf_base.join_asof(
                proxy_lf, on="timestamp", strategy="backward"
            )
            neutralized_lf = neutralize_lazyframe(
                lf_with_proxy,
                feature_names=group_features,
                proxy_col=proxy_col,
                window=window,
            )

            # timestamp + このグループの純化列のみを残す
            neutralized_cols = [
                f"{parse_suffixed_feature_name(f)[1]}_neutralized"
                for f in group_features
            ]
            group_dfs.append(neutralized_lf.select(["timestamp"] + neutralized_cols))

        if not group_dfs:
            continue

        # 複数グループを timestamp で結合（同一ファイルから派生するので timestamp は同一）
        if len(group_dfs) == 1:
            final_lf = group_dfs[0]
        else:
            final_lf = group_dfs[0]
            for gdf in group_dfs[1:]:
                final_lf = final_lf.join(gdf, on="timestamp", how="left")

        # 4-e. 出力
        if input_path.is_dir():
            # Tick データ: 手動 Hive パーティション出力
            print("  [INFO] Tick data detected. Using manual Hive partitioning.")
            df_to_save = final_lf.collect(engine="streaming")  # ルール5: streaming API
            grouped = df_to_save.group_by(
                pl.col("timestamp").dt.year().alias("year"),
                pl.col("timestamp").dt.month().alias("month"),
                pl.col("timestamp").dt.day().alias("day"),
                maintain_order=True,
            )
            output_path.mkdir(parents=True, exist_ok=True)
            partitions = list(grouped)
            for (year, month, day), data_group in tqdm(partitions, total=len(partitions), desc="  Writing partitions"):
                partition_dir = output_path / f"year={year}/month={month}/day={day}"
                partition_dir.mkdir(parents=True, exist_ok=True)
                data_group.write_parquet(
                    partition_dir / "data.parquet",
                    compression="zstd",
                    use_pyarrow=True,
                )
        else:
            # 非 Tick データ: sink_parquet で一括出力（ルール5）
            output_path.parent.mkdir(parents=True, exist_ok=True)
            final_lf.sink_parquet(
                str(output_path),
                compression="zstd",
            )

    print("\n" + "=" * 70)
    print("### 2_G_alpha_neutralizer.py: COMPLETED ###")
    print(f"Output: {blueprint.S5_NEUTRALIZED_ALPHA_SET}")
    print("=" * 70)


# =============================================================================
# エントリーポイント
# =============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="2_G_alpha_neutralizer: Multi-Timeframe Alpha Neutralization"
    )
    parser.add_argument(
        "--test-mode",
        action="store_true",
        help="Run in test mode (process only first 2 source files).",
    )
    parser.add_argument(
        "--clean",
        action="store_true",
        help="[Phase 6] Remove all existing output and start from scratch (default: resume).",
    )
    args = parser.parse_args()

    # [Phase 6] --clean フラグを main 経由で参照可能にする
    main._clean_flag = args.clean

    warnings.filterwarnings("ignore", category=UserWarning)
    try:
        from polars.exceptions import PolarsInefficientMapWarning
        warnings.filterwarnings("ignore", category=PolarsInefficientMapWarning)
    except ImportError:
        pass

    main(args.test_mode)
