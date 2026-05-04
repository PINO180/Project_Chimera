"""
2_B_statistical_filter.py
旧 01_LF_pre_selector.py + fast_track_correlation_filter.py を統合・再設計。

設計思想:
    「情報として死んでいる特徴量を弾く」だけを行う。
    予測力の判断は一切しない。
    LightGBM / SHAP / AV 関連処理は全て削除済み。

処理フロー:
    S2_FEATURES_VALIDATED の全特徴量を読み込む
        ↓
    フィルター①：分散フィルター（VARIANCE_THRESHOLD 以下を除外）
        ↓
    フィルター②：欠損率フィルター（NULL_RATE_THRESHOLD 以上を除外）
        ↓
    フィルター③：相関フィルター（全時間足横断・CORRELATION_THRESHOLD）
        ↓
    生き残り特徴量を LF_ALL_TIMEFRAMES / HF_TIMEFRAMES で分類
        ↓
    S3_FILTERED_LF_FEATURES / S3_FILTERED_HF_FEATURES に出力
"""

import sys
import json
import logging
import warnings
from pathlib import Path

# ルール7: 絶対パスハードコード禁止・動的解決
sys.path.append(str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
import polars as pl

from blueprint import (
    CORRELATION_THRESHOLD,
    CORR_SAMPLE_SIZE,
    HF_TIMEFRAMES,
    LF_ALL_TIMEFRAMES,
    NULL_RATE_THRESHOLD,
    S2_FEATURES_VALIDATED,
    S3_ARTIFACTS,
    S3_FILTERED_HF_FEATURES,
    S3_FILTERED_LF_FEATURES,
    VARIANCE_THRESHOLD,
)

warnings.filterwarnings("ignore", category=UserWarning)

# =================================================================
# ロギング設定
# =================================================================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# =================================================================
# 時間足優先度マップ（相関フィルター用）
# 小さい時間足（短周期）を優先して残す。
# ルール: ソート済みリストの先頭を常に生存させる設計。
# =================================================================
TF_PRIORITY: dict[str, int] = {
    "M0.5": 0,
    "M1": 1,
    "M3": 2,
    "M5": 3,
    "M8": 4,
    "M15": 5,
    "M30": 6,
    "H1": 7,
    "H4": 8,
    "H6": 9,
    "H12": 10,
    "D1": 11,
    "W1": 12,
    "MN": 13,
}

# =================================================================
# ユーティリティ
# =================================================================


def get_tf_from_feature_name(feature_name: str) -> str:
    """
    特徴量名末尾から時間足文字列を抽出する。
    例: e1a_statistical_mean_20_M1 → "M1"
    """
    return feature_name.split("_")[-1]


def discover_feature_files(base_path: Path) -> list[Path]:
    """S2_FEATURES_VALIDATED 以下の全 parquet ファイル / tick ディレクトリを列挙する。"""
    sources: list[Path] = []
    for item in base_path.rglob("*"):
        if item.is_file() and item.suffix == ".parquet":
            sources.append(item)
        elif item.is_dir() and "tick" in item.name:
            # tick は Hive パーティションディレクトリとして扱う
            if any(item.rglob("*.parquet")):
                sources.append(item)
    return sources


def scan_source(source: Path) -> pl.LazyFrame:
    """parquet ファイルまたは Hive パーティションディレクトリを LazyFrame として開く。"""
    if source.is_dir():
        return pl.scan_parquet(source / "**/*.parquet")
    return pl.scan_parquet(source)


# =================================================================
# 特徴量カラム収集
# =================================================================

BASE_COLS = frozenset(
    ["timestamp", "open", "high", "low", "close", "volume", "timeframe", "year", "month", "day"]
)


def collect_all_feature_columns(sources: list[Path]) -> dict[str, Path]:
    """
    全ソースファイルのスキーマを走査し、
    {サフィックス付き特徴量名: ソースパス} の辞書を返す。
    """
    feature_to_source: dict[str, Path] = {}
    for src in sources:
        try:
            if src.is_dir():
                first_file = next(src.rglob("*.parquet"), None)
                if first_file is None:
                    continue
                schema = pl.read_parquet_schema(first_file)
            else:
                schema = pl.read_parquet_schema(src)
        except Exception as e:
            logger.warning("スキーマ読み込み失敗: %s (%s)", src, e)
            continue

        # ファイル名末尾から時間足を推定してサフィックスとして付与
        tf = _infer_timeframe_from_path(src)
        for col in schema:
            if col in BASE_COLS:
                continue
            suffixed = f"{col}_{tf}" if tf else col
            if suffixed not in feature_to_source:
                feature_to_source[suffixed] = src

    return feature_to_source


def _infer_timeframe_from_path(path: Path) -> str | None:
    """パス名から時間足文字列を推定する。"""
    import re
    name = path.name
    m = re.search(r"_(M[0-9\.]+|H[0-9]+|D[0-9]+|W[0-9]+|MN|tick)(?:$|\.parquet$)", name)
    if m:
        return m.group(1)
    # Hive ディレクトリの場合
    if path.is_dir() and "tick" in name:
        return "tick"
    return None


# =================================================================
# フィルター①：分散フィルター
# =================================================================


def variance_filter(
    sources: list[Path],
    feature_to_source: dict[str, Path],
    threshold: float,
) -> list[str]:
    """
    各特徴量の全期間分散を計算し、threshold 以下を除外する。
    ルール2: Polars の rolling_std 準拠・不偏推定（ddof=1）。
    ルール8: Float32 を使用。
    """
    logger.info("フィルター①: 分散フィルター (threshold=%.2e)", threshold)

    survivors: list[str] = []
    removed: list[str] = []

    # ソースごとにまとめて処理してI/Oを削減
    src_to_features: dict[Path, list[str]] = {}
    for feat, src in feature_to_source.items():
        src_to_features.setdefault(src, []).append(feat)

    for src, feats in src_to_features.items():
        tf = _infer_timeframe_from_path(src)
        # 元カラム名（サフィックスを除いたもの）
        base_names = {f.rsplit(f"_{tf}", 1)[0] if tf and f.endswith(f"_{tf}") else f: f for f in feats}

        try:
            lf = scan_source(src).select(list(base_names.keys()))
            # ルール8: Float32
            lf = lf.with_columns(pl.col(c).cast(pl.Float32) for c in base_names)
            stats = lf.select(
                pl.col(c).var(ddof=1).alias(c) for c in base_names  # ddof=1: Polars rolling_std 準拠・不偏推定
            ).collect(engine="streaming")
        except Exception as e:
            logger.warning("分散計算失敗: %s (%s) — スキップ", src, e)
            survivors.extend(feats)
            continue

        for base_col, suffixed in base_names.items():
            try:
                var_val = stats[base_col][0]
            except Exception:
                var_val = None

            if var_val is None or np.isnan(float(var_val)):
                # 分散が計算できない場合は保守的に残す
                survivors.append(suffixed)
            elif float(var_val) <= threshold:
                removed.append(suffixed)
            else:
                survivors.append(suffixed)

    logger.info("  除外: %d 件 / 生存: %d 件", len(removed), len(survivors))
    return survivors


# =================================================================
# フィルター②：欠損率フィルター
# =================================================================


def null_rate_filter(
    feature_to_source: dict[str, Path],
    candidates: list[str],
    threshold: float,
) -> list[str]:
    """
    各特徴量の null_count / total_count を計算し、threshold 以上を除外する。
    ルール4: ゼロ除算防止（+ 1e-10）。
    """
    logger.info("フィルター②: 欠損率フィルター (threshold=%.1f%%)", threshold * 100)

    survivors: list[str] = []
    removed: list[str] = []

    src_to_feats: dict[Path, list[str]] = {}
    for feat in candidates:
        src = feature_to_source.get(feat)
        if src:
            src_to_feats.setdefault(src, []).append(feat)

    for src, feats in src_to_feats.items():
        tf = _infer_timeframe_from_path(src)
        base_names = {
            f.rsplit(f"_{tf}", 1)[0] if tf and f.endswith(f"_{tf}") else f: f
            for f in feats
        }

        try:
            lf = scan_source(src).select(list(base_names.keys()))
            collected = lf.collect(engine="streaming")
            total = len(collected) + 1e-10  # ルール4: ゼロ除算防止
        except Exception as e:
            logger.warning("欠損率計算失敗: %s (%s) — スキップ", src, e)
            survivors.extend(feats)
            continue

        for base_col, suffixed in base_names.items():
            try:
                null_rate = collected[base_col].null_count() / total
            except Exception:
                null_rate = 0.0

            if null_rate >= threshold:
                removed.append(suffixed)
            else:
                survivors.append(suffixed)

    logger.info("  除外: %d 件 / 生存: %d 件", len(removed), len(survivors))
    return survivors


# =================================================================
# フィルター③：相関フィルター（全時間足横断）
# =================================================================


def _load_sample_for_corr(
    feature_to_source: dict[str, Path],
    candidates: list[str],
    sample_size: int,
) -> pd.DataFrame:
    """
    CORR_SAMPLE_SIZE 行でサンプリングして候補特徴量の pandas DataFrame を返す。
    メモリ節約のためソースごとに tail(sample_size) で取得後に結合する。
    """
    src_to_feats: dict[Path, list[str]] = {}
    for feat in candidates:
        src = feature_to_source.get(feat)
        if src:
            src_to_feats.setdefault(src, []).append(feat)

    dfs: list[pd.DataFrame] = []

    for src, feats in src_to_feats.items():
        tf = _infer_timeframe_from_path(src)
        base_names = {
            f.rsplit(f"_{tf}", 1)[0] if tf and f.endswith(f"_{tf}") else f: f
            for f in feats
        }
        cols_to_read = ["timestamp"] + list(base_names.keys())

        try:
            # Polars lazy evaluation + tail でサンプリング（ルール1: 過去方向のみ参照）
            # NOTE: tail() は直近データに偏る。将来的には sample(n=sample_size, seed=42) への変更を検討。
            lf = scan_source(src).select(cols_to_read).tail(sample_size)
            df_part = lf.collect(engine="streaming").to_pandas()
        except Exception as e:
            logger.warning("サンプリング失敗: %s (%s) — スキップ", src, e)
            continue

        df_part = df_part.rename(columns=base_names)
        dfs.append(df_part)

    if not dfs:
        return pd.DataFrame()

    # timestamp で前方向マージ（ルール1: backward）
    result = dfs[0]
    for df_to_join in dfs[1:]:
        result = pd.merge_asof(
            result.sort_values("timestamp"),
            df_to_join.sort_values("timestamp"),
            on="timestamp",
            direction="backward",
        )

    return result.drop(columns=["timestamp"], errors="ignore")


def correlation_filter(
    feature_to_source: dict[str, Path],
    candidates: list[str],
    threshold: float,
    sample_size: int,
) -> list[str]:
    """
    全時間足横断の相関フィルター。

    ルール:
        - 特徴量リストを時間足優先度（TF_PRIORITY）で昇順ソートしてから処理する。
        - 上三角行列を走査し、threshold 以上の相関を持つペアを検出。
        - 「ソート済みリストの先頭（小さい時間足）を常に生存させる」設計。
          つまり先に現れた特徴量（小さい時間足）を残し、後から現れた方を削除する。
        - メモリが厳しい場合はチャンク処理で対応。

    ルール4: fillna(0) で NaN を処理（ゼロ除算防止は corr 内部で対応）。
    ルール8: 入力を Float32 にキャストしてから pandas に渡す。
    """
    logger.info(
        "フィルター③: 相関フィルター (threshold=%.2f, sample_size=%d)",
        threshold,
        sample_size,
    )

    # ① 時間足優先度でソート（小さい時間足が先頭 → 先頭を残すロジックで保護される）
    sorted_candidates: list[str] = sorted(
        candidates,
        key=lambda f: TF_PRIORITY.get(get_tf_from_feature_name(f), 99),
    )

    logger.info("  サンプルデータをロード中...")
    df_sample = _load_sample_for_corr(feature_to_source, sorted_candidates, sample_size)

    if df_sample.empty:
        logger.warning("  サンプルデータが空のため相関フィルターをスキップします。")
        return sorted_candidates

    # ② ソート順を DataFrame のカラム順序に反映（上三角行列の方向を確定）
    existing_cols = [c for c in sorted_candidates if c in df_sample.columns]
    df_sample = df_sample[existing_cols].astype("float32").fillna(0)

    # ③ 相関行列の計算（pandas corr）
    logger.info("  相関行列を計算中（%d 特徴量）...", len(existing_cols))

    # チャンク処理: 一度に全特徴量をメモリに乗せられない場合のフォールバック
    CHUNK_LIMIT = 2000
    if len(existing_cols) > CHUNK_LIMIT:
        logger.info("  特徴量数が %d を超えるためチャンク処理を実施します。", CHUNK_LIMIT)
        to_drop = _correlation_filter_chunked(df_sample, existing_cols, threshold, CHUNK_LIMIT)
    else:
        corr_matrix = df_sample.corr().abs()
        upper_tri = corr_matrix.where(
            np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
        )
        # 「ソート済みリストの先頭を常に生存させる」設計:
        # 上三角行列の列方向（後から現れた特徴量）が高相関なら削除対象とする。
        to_drop = {
            col for col in upper_tri.columns if any(upper_tri[col] >= threshold)
        }

    survivors = [f for f in existing_cols if f not in to_drop]
    # df_sample に存在しなかった特徴量は保守的に残す
    missing = [f for f in sorted_candidates if f not in existing_cols]
    survivors.extend(missing)

    logger.info("  除外: %d 件 / 生存: %d 件", len(to_drop), len(survivors))
    return survivors


def _correlation_filter_chunked(
    df: pd.DataFrame,
    cols: list[str],
    threshold: float,
    chunk_size: int,
) -> set[str]:
    """
    ベースチャンク × 全特徴量比較方式のチャンク処理相関フィルター。

    アルゴリズム:
        cols は TF_PRIORITY 昇順（小さい時間足が先頭）でソート済みであることを前提とする。
        i 番目の特徴量（ベース）を chunk_size 個ずつ処理し、
        それより後ろにある全特徴量（i+1 以降）との相関を numpy で計算する。
        これにより「チャンク1（M系）× チャンク2（H/D/W/MN系）」のペアも漏れなく検出できる。

    設計:
        - ベースが既に to_drop に入っていればスキップ（削除済みを軸に比較しない）。
        - ベースチャンク（chunk_size 列）× 残り全列の内積で相関係数を一括計算。
        - 「先頭（小さい時間足）を常に生存させる」ロジックを維持:
          ベース列は生存済み、後ろに現れた列が threshold 以上なら to_drop に追加。
        - numpy float32 で計算してメモリを抑制。
    """
    arr = df[cols].values.astype(np.float32)  # shape: (n_samples, n_features)
    n = len(cols)
    to_drop: set[str] = set()

    # 列ごとの標準偏差を事前計算（ルール2: ddof=1 / ルール4: ゼロ除算防止 +1e-10）
    std = arr.std(axis=0, ddof=1) + 1e-10  # shape: (n_features,)
    # 平均を引いて正規化
    arr_centered = arr - arr.mean(axis=0)  # shape: (n_samples, n_features)

    logger.info("  チャンク処理: %d 特徴量 × %d チャンクサイズ", n, chunk_size)

    for base_start in range(0, n, chunk_size):
        base_end = min(base_start + chunk_size, n)
        base_idx = [i for i in range(base_start, base_end) if cols[i] not in to_drop]
        if not base_idx:
            continue

        # ベース先頭列より後ろの全列を比較対象（チャンク内・外を問わず上三角行列の完全走査）
        rest_idx = [i for i in range(base_start + 1, n) if cols[i] not in to_drop]
        if not rest_idx:
            continue

        base_arr = arr_centered[:, base_idx]   # (n_samples, len(base_idx))
        rest_arr = arr_centered[:, rest_idx]    # (n_samples, len(rest_idx))

        # 相関行列 = (X^T Y) / (n_samples * std_x * std_y)
        n_samples = arr.shape[0]
        cov_block = (base_arr.T @ rest_arr) / n_samples  # (len(base_idx), len(rest_idx))
        std_base = std[base_idx][:, None]                 # (len(base_idx), 1)
        std_rest = std[rest_idx][None, :]                 # (1, len(rest_idx))
        corr_block = np.abs(cov_block / (std_base * std_rest))  # (len(base_idx), len(rest_idx))

        # threshold 以上の rest 列を削除対象に追加
        # axis=0 で各 rest 列についてベースチャンクのどれかと高相関かを判定
        drop_mask = np.any(corr_block >= threshold, axis=0)  # (len(rest_idx),)
        for flag, ri in zip(drop_mask, rest_idx):
            if flag:
                to_drop.add(cols[ri])

        logger.debug(
            "  チャンク [%d:%d] 処理完了 — 累計除外数: %d", base_start, base_end, len(to_drop)
        )

    return to_drop


# =================================================================
# 出力分類
# =================================================================


def classify_and_save(
    survivors: list[str],
    lf_timeframes: list[str],
    hf_timeframes: list[str],
    lf_output_path: Path,
    hf_output_path: Path,
) -> tuple[list[str], list[str]]:
    """
    生き残り特徴量を LF / HF に分類して各テキストファイルに出力する。
    各ファイルは1行1特徴量形式。
    """
    lf_set = set(lf_timeframes)
    hf_set = set(hf_timeframes)

    lf_features: list[str] = []
    hf_features: list[str] = []
    unclassified: list[str] = []

    for feat in sorted(survivors):
        tf = get_tf_from_feature_name(feat)
        if tf in lf_set:
            lf_features.append(feat)
        elif tf in hf_set:
            hf_features.append(feat)
        else:
            unclassified.append(feat)

    if unclassified:
        logger.warning("  分類不能な特徴量 %d 件（時間足不明）: %s ...", len(unclassified), unclassified[:5])

    lf_output_path.parent.mkdir(parents=True, exist_ok=True)
    hf_output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(lf_output_path, "w", encoding="utf-8") as f:
        for feat in lf_features:
            f.write(f"{feat}\n")

    with open(hf_output_path, "w", encoding="utf-8") as f:
        for feat in hf_features:
            f.write(f"{feat}\n")

    logger.info("  LF 特徴量 → %s (%d 件)", lf_output_path, len(lf_features))
    logger.info("  HF 特徴量 → %s (%d 件)", hf_output_path, len(hf_features))

    return lf_features, hf_features


# =================================================================
# メイン
# =================================================================


def main() -> None:
    logger.info("=" * 60)
    logger.info("2_B_statistical_filter.py 開始")
    logger.info("=" * 60)

    if not S2_FEATURES_VALIDATED.exists():
        raise FileNotFoundError(
            f"S2_FEATURES_VALIDATED が見つかりません: {S2_FEATURES_VALIDATED}"
        )

    # --- ソースファイル探索 ---
    logger.info("特徴量ソースファイルを探索中: %s", S2_FEATURES_VALIDATED)
    sources = discover_feature_files(S2_FEATURES_VALIDATED)
    logger.info("  発見したソース数: %d", len(sources))

    if not sources:
        raise RuntimeError("特徴量ソースファイルが 1 件も見つかりませんでした。")

    feature_to_source = collect_all_feature_columns(sources)
    all_features = list(feature_to_source.keys())
    logger.info("  発見した特徴量数（サフィックス付き）: %d", len(all_features))

    stats: dict[str, int] = {"initial": len(all_features)}

    # --- フィルター① 分散フィルター ---
    after_variance = variance_filter(
        sources,
        feature_to_source,
        threshold=VARIANCE_THRESHOLD,
    )
    stats["after_variance"] = len(after_variance)
    stats["removed_by_variance"] = stats["initial"] - len(after_variance)

    # --- フィルター② 欠損率フィルター ---
    after_null = null_rate_filter(
        feature_to_source,
        candidates=after_variance,
        threshold=NULL_RATE_THRESHOLD,
    )
    stats["after_null_rate"] = len(after_null)
    stats["removed_by_null_rate"] = len(after_variance) - len(after_null)

    # --- フィルター③ 相関フィルター ---
    after_corr = correlation_filter(
        feature_to_source,
        candidates=after_null,
        threshold=CORRELATION_THRESHOLD,
        sample_size=CORR_SAMPLE_SIZE,
    )
    stats["after_correlation"] = len(after_corr)
    stats["removed_by_correlation"] = len(after_null) - len(after_corr)

    # --- 出力分類 ---
    logger.info("出力分類を実行中...")
    lf_features, hf_features = classify_and_save(
        survivors=after_corr,
        lf_timeframes=LF_ALL_TIMEFRAMES,
        hf_timeframes=HF_TIMEFRAMES,
        lf_output_path=S3_FILTERED_LF_FEATURES,
        hf_output_path=S3_FILTERED_HF_FEATURES,
    )
    stats["lf_features"] = len(lf_features)
    stats["hf_features"] = len(hf_features)

    # --- フィルター統計ログ & JSON 出力 ---
    logger.info("=" * 60)
    logger.info("フィルター統計サマリー")
    logger.info("  初期特徴量数        : %d", stats["initial"])
    logger.info("  分散フィルター除外   : %d", stats["removed_by_variance"])
    logger.info("  欠損率フィルター除外 : %d", stats["removed_by_null_rate"])
    logger.info("  相関フィルター除外   : %d", stats["removed_by_correlation"])
    logger.info("  最終生存数           : %d", stats["after_correlation"])
    logger.info("    └ LF : %d", stats["lf_features"])
    logger.info("    └ HF : %d", stats["hf_features"])
    logger.info("=" * 60)

    json_path = S3_ARTIFACTS / "2B_filter_stats.json"
    json_path.parent.mkdir(parents=True, exist_ok=True)
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)
    logger.info("フィルター統計 JSON を保存しました: %s", json_path)

    logger.info("2_B_statistical_filter.py 完了")


if __name__ == "__main__":
    main()
