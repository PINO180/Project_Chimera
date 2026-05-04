"""
2_C_lf_wf_signal_generator.py
──────────────────────────────
LF環境スコア生成器（旧: 02_M1_walk_forward_validator.py の後継）

役割:
    LF特徴量を3グループ（lf_short / lf_mid / lf_long）に分類し、
    グループ別にPurged KFold WFでOOF予測スコアを生成する。
    出力スコアは連続値（float32）のまま 2_E（HFメタモデル）への追加特徴量として使われる。

廃止した旧処理:
    - SHAP重要度による特徴量選別
    - S3_SURVIVED_HF_FEATURESへの出力
    - HF特徴量への一切の処理
    - SHAPに関わる全コード
"""

import sys
import warnings
import argparse
import logging
from pathlib import Path
from datetime import timedelta, datetime

# ルール7: 動的パス解決
sys.path.append(str(Path(__file__).resolve().parents[1]))

import polars as pl
import pandas as pd
import numpy as np
import lightgbm as lgb

from blueprint import (
    S2_FEATURES_VALIDATED,
    S3_ARTIFACTS,
    S3_FILTERED_LF_FEATURES,
    S3_FINAL_FEATURE_TEAM,
    S3_LF_ENVIRONMENT_SCORES,
    LF_SHORT_TIMEFRAMES,
    LF_MID_TIMEFRAMES,
    LF_LONG_TIMEFRAMES,
    WF_TARGET_SHIFT,
    WF_CONFIG,
)

# [Phase 6 修正] ログ統一: 2_A/2_B と同じ logging 形式に揃える
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────
# 定数
# ─────────────────────────────────────────────
# purge_days・embargo_days はグループ別にWF_CONFIGから取得する。
# 各グループのターゲットshift距離（M1換算）に対して十分なpurge幅が必要:
#   lf_short: shift=-4800 M1バー ≒ 3.3日 → purge_days=5
#   lf_mid  : shift=-14400 M1バー ≒ 10日  → purge_days=12
#   lf_long : shift=-40320 M1バー ≒ 28日  → purge_days=30
# WF_CONFIG（blueprint.py）に purge_days・embargo_days を定義すること。

LGP_PARAMS = {
    "objective": "binary",
    "num_leaves": 8,
    "max_depth": 3,
    "n_estimators": 200,
    "learning_rate": 0.05,
    "lambda_l1": 1.0,
    "lambda_l2": 1.0,
    "min_data_in_leaf": 200,
    "colsample_bytree": 0.7,
    "subsample": 0.7,
    "verbosity": -1,
    "random_state": 42,
    "n_jobs": -1,
}


# ─────────────────────────────────────────────
# ユーティリティ
# ─────────────────────────────────────────────

def classify_features_by_group(feature_names: list[str]) -> dict[str, list[str]]:
    """
    特徴量名のサフィックス（末尾のタイムフレーム）でグループ分けする。
    例: 'ema_20_H4' → lf_short
    """
    groups: dict[str, list[str]] = {"lf_short": [], "lf_mid": [], "lf_long": []}

    short_set = set(LF_SHORT_TIMEFRAMES)
    mid_set = set(LF_MID_TIMEFRAMES)
    long_set = set(LF_LONG_TIMEFRAMES)

    for fname in feature_names:
        parts = fname.rsplit("_", 1)
        if len(parts) < 2:
            continue
        tf = parts[-1]
        if tf in short_set:
            groups["lf_short"].append(fname)
        elif tf in mid_set:
            groups["lf_mid"].append(fname)
        elif tf in long_set:
            groups["lf_long"].append(fname)
        # タイムフレーム不明の特徴量は無視

    return groups


def purged_kfold_splits(
    timestamps: pd.Series,
    train_months: int,
    val_months: int,
    purge_days: int,
    embargo_days: int,
):
    """
    日付ベースのPurged KFoldスプリットを生成するジェネレータ。

    purge_days  : 訓練末尾から前向き purge_days 日間のデータを除外
                  （ターゲット生成に使ったshiftが訓練末尾と重なるリークを防ぐ）
                  グループ別に設定すること（lf_short=5, lf_mid=12, lf_long=30）
    embargo_days: purge後さらに embargo_days 日間のデータを除外
                  （自己相関残留を遮断）
                  グループ別に設定すること（lf_short=2, lf_mid=3, lf_long=5）

    Yields:
        dict with keys: train_idx, val_idx, train_start, train_end, val_start, val_end
    """
    timestamps = pd.to_datetime(timestamps).reset_index(drop=True)
    start = timestamps.min()
    end = timestamps.max()

    current_start = start
    while True:
        train_end = current_start + pd.DateOffset(months=train_months)
        val_start_raw = train_end + timedelta(days=purge_days + embargo_days)
        val_end = val_start_raw + pd.DateOffset(months=val_months)

        if val_end > end:
            break

        # purge: 訓練末尾の purge_days 日間を除外
        purge_cutoff = train_end - timedelta(days=purge_days)

        train_mask = (timestamps >= current_start) & (timestamps < purge_cutoff)
        val_mask = (timestamps >= val_start_raw) & (timestamps < val_end)

        train_idx = timestamps[train_mask].index.tolist()
        val_idx = timestamps[val_mask].index.tolist()

        if len(train_idx) > 0 and len(val_idx) > 0:
            yield {
                "train_idx": train_idx,
                "val_idx": val_idx,
                "train_start": current_start,
                "train_end": train_end,
                "val_start": val_start_raw,
                "val_end": val_end,
            }

        current_start = current_start + pd.DateOffset(months=val_months)


def run_wf_for_group(
    df: pd.DataFrame,
    feature_cols: list[str],
    group: str,
    test_mode: bool = False,
) -> pd.Series:
    """
    グループ別にPurged KFold WFを実行し、全期間のOOFスコアを返す。

    Returns:
        pd.Series (index=df.index) with OOF predicted probabilities (float32)
    """
    wf_conf = WF_CONFIG[group]
    train_months = wf_conf["train_months"]
    val_months = wf_conf["val_months"]
    # グループ別purge_days・embargo_days（blueprint.WF_CONFIGから取得）
    # ターゲットshift距離に対して十分なpurge幅を確保し情報リークを防止する
    purge_days = wf_conf["purge_days"]
    embargo_days = wf_conf["embargo_days"]

    oof_scores = pd.Series(np.nan, index=df.index, dtype=np.float32)
    timestamps = pd.to_datetime(df["timestamp"])

    splits = list(
        purged_kfold_splits(
            timestamps, train_months, val_months,
            purge_days=purge_days,
            embargo_days=embargo_days,
        )
    )
    if test_mode:
        splits = splits[:2]

    logger.info(
        f"  [{group}] folds={len(splits)}, "
        f"train_months={train_months}, val_months={val_months}, "
        f"purge_days={purge_days}, embargo_days={embargo_days}"
    )

    for fold_i, split in enumerate(splits):
        train_idx = split["train_idx"]
        val_idx = split["val_idx"]

        X_train = df.loc[train_idx, feature_cols].fillna(0).astype(np.float32)
        y_train = df.loc[train_idx, "target"]
        X_val = df.loc[val_idx, feature_cols].fillna(0).astype(np.float32)

        # ターゲットが欠損する末尾行を除外（shiftによるNaN）
        valid_train = y_train.notna()
        if valid_train.sum() < LGP_PARAMS["min_data_in_leaf"] * 2:
            logger.info(f"  [{group}] fold {fold_i + 1}: 訓練データ不足。スキップ。")
            continue

        X_train = X_train[valid_train]
        y_train = y_train[valid_train]

        model = lgb.LGBMClassifier(**LGP_PARAMS)
        model.fit(X_train, y_train)

        if len(X_val) > 0:
            # predict_proba[:, 1] → クラス1（上昇）の確率
            preds = model.predict_proba(X_val)[:, 1].astype(np.float32)
            oof_scores.loc[val_idx] = preds

        logger.info(
            f"  [{group}] fold {fold_i + 1}/{len(splits)} done | "
            f"train={len(X_train)}, val={len(X_val)}"
        )

    return oof_scores


# ─────────────────────────────────────────────
# メイン
# ─────────────────────────────────────────────

def main(test_mode: bool) -> None:
    logger.info("=" * 60)
    logger.info("2_C_lf_wf_signal_generator.py 開始")
    logger.info("=" * 60)

    S3_ARTIFACTS.mkdir(parents=True, exist_ok=True)

    # ── 1. フィルター済みLF特徴量リストの読み込み ──────────────
    if not S3_FILTERED_LF_FEATURES.exists():
        raise FileNotFoundError(
            f"フィルター済みLF特徴量リストが見つかりません: {S3_FILTERED_LF_FEATURES}\n"
            "2_B_statistical_filter.py を先に実行してください。"
        )

    with open(S3_FILTERED_LF_FEATURES, "r") as f:
        lf_features = [line.strip() for line in f if line.strip()]

    logger.info(f"LF特徴量読み込み: {len(lf_features)} 件")

    # ── 2. LF特徴量をグループに分類 ────────────────────────────
    groups = classify_features_by_group(lf_features)
    for g, cols in groups.items():
        logger.info(f"  {g}: {len(cols)} 特徴量")

    all_lf_features = [f for cols in groups.values() for f in cols]
    if not all_lf_features:
        raise ValueError("グループに分類できるLF特徴量がゼロです。タイムフレームサフィックスを確認してください。")

    # ── 3. バリデート済みデータの読み込み ──────────────────────
    logger.info(f"\nデータ読み込み中: {S2_FEATURES_VALIDATED}")

    # M1価格データ（closeカラム）とLF特徴量を結合
    # 全特徴量はM1粒度であることを前提とする（join_asof不要）
    price_files = list(S2_FEATURES_VALIDATED.rglob("*_M1.parquet"))
    if not price_files:
        raise FileNotFoundError(
            f"M1価格ファイルが見つかりません: {S2_FEATURES_VALIDATED}/**/*_M1.parquet"
        )

    price_lf = (
        pl.scan_parquet(price_files[0])
        .select(["timestamp", "close"])
        .sort("timestamp")
    )

    # LF特徴量ファイルをスキャン（複数parquetをまとめて読み込む）
    lf_parquet_files = list(S2_FEATURES_VALIDATED.rglob("*.parquet"))
    if not lf_parquet_files:
        raise FileNotFoundError(f"parquetファイルが見つかりません: {S2_FEATURES_VALIDATED}")

    # スキーマ確認で存在するカラム名を収集
    available_cols: set[str] = set()
    for p in lf_parquet_files:
        try:
            schema = pl.read_parquet_schema(p)
            available_cols.update(schema.keys())
        except Exception as e:
            logger.info(f"Warning: スキーマ読み込み失敗 {p}: {e}")

    # ── サフィックス付き特徴量名 → parquetベース名 のマッピングを構築 ──
    # 2_Bはテキストにタイムフレームサフィックスを付与して保存しているが、
    # parquetのカラム名はサフィックスなしの元の名前のまま。
    # 例: テキスト "e1a_anderson_darling_30_D1" → parquet "e1a_anderson_darling_30"
    all_tf_set = set(LF_SHORT_TIMEFRAMES + LF_MID_TIMEFRAMES + LF_LONG_TIMEFRAMES)

    def strip_tf_suffix(name: str) -> str:
        """e1a_some_feature_20_H4 → e1a_some_feature_20"""
        parts = name.rsplit("_", 1)
        if len(parts) == 2 and parts[-1] in all_tf_set:
            return parts[0]
        return name

    # suffixed名 → base名 のマッピング
    suffixed_to_base: dict[str, str] = {f: strip_tf_suffix(f) for f in all_lf_features}

    # parquetに存在するbaseカラムに対応するsuffixed名のみを有効とする
    valid_lf_features = [f for f in all_lf_features if suffixed_to_base[f] in available_cols]
    missing = set(all_lf_features) - set(valid_lf_features)
    if missing:
        logger.info(f"Warning: {len(missing)} 特徴量がデータ内に見つかりませんでした。スキップします。")

    if not valid_lf_features:
        raise ValueError("有効なLF特徴量がゼロです。")

    # selectにはparquetの実カラム名（base名）を使用
    valid_base_cols = [suffixed_to_base[f] for f in valid_lf_features]

    # ── 特徴量データ読み込み（ファイル別にselectしてサフィックス付き名にrename）──
    # 全ファイルまとめてscan_parquetすると同一base名が複数ファイルに存在して
    # ComputeError（カラム重複）になるため、ファイルごとに:
    #   1. ファイル名末尾からタイムフレームを推測（例: features_e1a_D1.parquet → D1）
    #   2. そのタイムフレームに対応するサフィックス付き特徴量を抽出
    #   3. base名でselectしてサフィックス付き名にrename
    # これによりDataFrame上で「e1a_xxx_H4」「e1a_xxx_D1」が別カラムとして区別され、
    # LightGBMが時間足ごとの特徴量の違いを正しく学習できる。
    # ルール8: renameの後にFloat32にキャスト
    lf_parts: list[pl.LazyFrame] = []
    loaded_suffixed_cols: set[str] = set()

    for p in lf_parquet_files:
        try:
            schema = pl.read_parquet_schema(p)
        except Exception as e:
            logger.info(f"Warning: スキーマ読み込み失敗 {p}: {e}")
            continue

        # ファイル名末尾からタイムフレームを推測（例: features_e1a_D1 → D1）
        tf = p.stem.split("_")[-1]

        # このタイムフレームに対応し、まだ読み込んでいないサフィックス付き特徴量を抽出
        target_suffixed = [
            f for f in valid_lf_features
            if f.endswith(f"_{tf}") and f not in loaded_suffixed_cols
        ]
        if not target_suffixed:
            continue

        # base名が実際にスキーマに存在するもののみをrename_mapに登録
        rename_map: dict[str, str] = {}
        for suffixed in target_suffixed:
            base_name = suffixed_to_base[suffixed]
            if base_name in schema.keys():
                rename_map[base_name] = suffixed

        if not rename_map:
            continue

        base_cols = list(rename_map.keys())
        suffixed_cols = list(rename_map.values())

        # select → rename（base名→サフィックス付き名）→ Float32キャスト
        part = (
            pl.scan_parquet(p)
            .select(["timestamp"] + base_cols)
            .rename(rename_map)
            .with_columns([pl.col(c).cast(pl.Float32) for c in suffixed_cols])  # ルール8
        )
        lf_parts.append(part)
        loaded_suffixed_cols.update(suffixed_cols)

    if not lf_parts:
        raise ValueError("有効なカラムを持つparquetファイルが見つかりませんでした。")

    # timestamp基準でjoin（全特徴量がM1粒度なのでleft joinで問題なし）
    feat_lf = lf_parts[0]
    for part in lf_parts[1:]:
        feat_lf = feat_lf.join(part, on="timestamp", how="left")
    feat_lf = feat_lf.sort("timestamp")

    # priceとfeatureをtimestampで結合
    data_lf = price_lf.join(feat_lf, on="timestamp", how="left")

    logger.info("データをcollect中（streaming）...")
    # ルール5: collect(engine="streaming")
    base_df = data_lf.collect(engine="streaming").to_pandas()
    base_df["timestamp"] = pd.to_datetime(base_df["timestamp"])
    base_df = base_df.sort_values("timestamp").reset_index(drop=True)
    logger.info(f"  行数: {len(base_df):,} | カラム数: {len(base_df.columns)}")

    # ── 4. グループ別ターゲット生成 + OOFスコア生成 ───────────
    all_oof: dict[str, pd.Series] = {}

    for group, cols in groups.items():
        # 実際に読み込めたサフィックス付き特徴量のみを対象とする
        valid_suffixed = [c for c in cols if c in loaded_suffixed_cols]

        if not valid_suffixed:
            logger.info(f"\n[{group}] 有効な特徴量がないためスキップ。")
            all_oof[f"{group}_score"] = pd.Series(
                np.nan, index=base_df.index, dtype=np.float32
            )
            continue

        logger.info(f"\n{'─'*40}")
        logger.info(f"グループ処理: {group}  ({len(valid_suffixed)} 特徴量)")

        # M1粒度shiftでターゲット生成
        target_shift = WF_TARGET_SHIFT[group]
        group_df = base_df.copy()
        future_close = group_df["close"].shift(target_shift)  # 負値 → 前向きシフト
        group_df["target"] = (future_close > group_df["close"]).astype(float)
        group_df.loc[future_close.isna(), "target"] = np.nan

        oof = run_wf_for_group(
            df=group_df,
            feature_cols=valid_suffixed,  # サフィックス付きのままDataFrame上のカラム名と一致
            group=group,
            test_mode=test_mode,
        )
        all_oof[f"{group}_score"] = oof.astype(np.float32)
        covered = oof.notna().sum()
        logger.info(f"  [{group}] OOFカバレッジ: {covered:,} / {len(oof):,} 行")

    # ── 5. 結果DataFrameの構築 ─────────────────────────────────
    result_df = base_df[["timestamp"]].copy()
    for col_name, series in all_oof.items():
        result_df[col_name] = series.values

    # ── 6. S3_LF_ENVIRONMENT_SCORES への保存 ──────────────────
    # ルール8: スコアはfloat32で保存
    score_cols = [c for c in result_df.columns if c != "timestamp"]
    result_pl = pl.from_pandas(result_df).with_columns(
        [pl.col(c).cast(pl.Float32) for c in score_cols]  # ルール8
    )

    result_pl.write_parquet(S3_LF_ENVIRONMENT_SCORES)
    logger.info(f"\nLF環境スコア保存: {S3_LF_ENVIRONMENT_SCORES}")
    logger.info(f"  カラム: {result_pl.columns}")
    logger.info(f"  行数  : {len(result_pl):,}")

    # ── 7. S3_FINAL_FEATURE_TEAM への保存 ─────────────────────
    # 実際にparquetから読み込んで学習に使えたサフィックス付き特徴量のみを保存
    final_features = sorted(list(loaded_suffixed_cols))

    # [Phase 6 修正] sample_weight 系を LF 最終特徴量から除外
    # HF 側 (2_E) では EXCLUDE_SAMPLE_WEIGHT=1 採用済だが LF 側は除外漏れがあった。
    # 過去 (Phase 5 以前) は volume=0 で variance フィルタ自動除外されていたため
    # 顕在化しなかったが、Phase 6 で volume = tick_count 補完によって sample_weight も
    # 有意な分散を持ち通過するようになった。LF 側でも明示的に除外する (2_F の二重防御と整合)。
    sw_features_lf = [f for f in final_features if "sample_weight" in f]
    if sw_features_lf:
        logger.info(f"\n[Phase 6 Filter] LF から sample_weight 系 {len(sw_features_lf)} 件を除外:")
        for f in sw_features_lf:
            logger.info(f"    - {f}")
        final_features = [f for f in final_features if "sample_weight" not in f]
        logger.info(f"  -> Filtered LF features: {len(final_features)}")

    with open(S3_FINAL_FEATURE_TEAM, "w") as f:
        for feat in final_features:
            f.write(feat + "\n")

    logger.info(f"最終LF特徴量チーム保存: {S3_FINAL_FEATURE_TEAM} ({len(final_features)} 件)")
    logger.info("\n2_C_lf_wf_signal_generator.py 正常終了。")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="LF環境スコア生成器 (2_C)"
    )
    parser.add_argument(
        "--test-mode",
        action="store_true",
        help="テストモード: 各グループ最初の2フォールドのみ実行する。",
    )
    args = parser.parse_args()

    warnings.filterwarnings("ignore", category=UserWarning)
    main(args.test_mode)
