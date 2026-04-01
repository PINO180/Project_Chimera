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

    print(
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
            print(f"  [{group}] fold {fold_i + 1}: 訓練データ不足。スキップ。")
            continue

        X_train = X_train[valid_train]
        y_train = y_train[valid_train]

        model = lgb.LGBMClassifier(**LGP_PARAMS)
        model.fit(X_train, y_train)

        if len(X_val) > 0:
            # predict_proba[:, 1] → クラス1（上昇）の確率
            preds = model.predict_proba(X_val)[:, 1].astype(np.float32)
            oof_scores.loc[val_idx] = preds

        print(
            f"  [{group}] fold {fold_i + 1}/{len(splits)} done | "
            f"train={len(X_train)}, val={len(X_val)}"
        )

    return oof_scores


# ─────────────────────────────────────────────
# メイン
# ─────────────────────────────────────────────

def main(test_mode: bool) -> None:
    print("=" * 60)
    print("2_C_lf_wf_signal_generator.py 開始")
    print("=" * 60)

    S3_ARTIFACTS.mkdir(parents=True, exist_ok=True)

    # ── 1. フィルター済みLF特徴量リストの読み込み ──────────────
    if not S3_FILTERED_LF_FEATURES.exists():
        raise FileNotFoundError(
            f"フィルター済みLF特徴量リストが見つかりません: {S3_FILTERED_LF_FEATURES}\n"
            "2_B_statistical_filter.py を先に実行してください。"
        )

    with open(S3_FILTERED_LF_FEATURES, "r") as f:
        lf_features = [line.strip() for line in f if line.strip()]

    print(f"LF特徴量読み込み: {len(lf_features)} 件")

    # ── 2. LF特徴量をグループに分類 ────────────────────────────
    groups = classify_features_by_group(lf_features)
    for g, cols in groups.items():
        print(f"  {g}: {len(cols)} 特徴量")

    all_lf_features = [f for cols in groups.values() for f in cols]
    if not all_lf_features:
        raise ValueError("グループに分類できるLF特徴量がゼロです。タイムフレームサフィックスを確認してください。")

    # ── 3. バリデート済みデータの読み込み ──────────────────────
    print(f"\nデータ読み込み中: {S2_FEATURES_VALIDATED}")

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

    # スキーマ確認で存在する特徴量のみを選択
    available_cols: set[str] = set()
    for p in lf_parquet_files:
        try:
            schema = pl.read_parquet_schema(p)
            available_cols.update(schema.keys())
        except Exception as e:
            print(f"Warning: スキーマ読み込み失敗 {p}: {e}")

    valid_lf_features = [f for f in all_lf_features if f in available_cols]
    missing = set(all_lf_features) - set(valid_lf_features)
    if missing:
        print(f"Warning: {len(missing)} 特徴量がデータ内に見つかりませんでした。スキップします。")

    if not valid_lf_features:
        raise ValueError("有効なLF特徴量がゼロです。")

    # 特徴量データ読み込み（ルール8: Float32）
    feat_lf = (
        pl.scan_parquet(lf_parquet_files)
        .select(["timestamp"] + valid_lf_features)
        .sort("timestamp")
        .with_columns(
            [pl.col(c).cast(pl.Float32) for c in valid_lf_features]  # ルール8
        )
    )

    # priceとfeatureをtimestampで結合
    data_lf = price_lf.join(feat_lf, on="timestamp", how="left")

    print("データをcollect中（streaming）...")
    # ルール5: collect(engine="streaming")
    base_df = data_lf.collect(engine="streaming").to_pandas()
    base_df["timestamp"] = pd.to_datetime(base_df["timestamp"])
    base_df = base_df.sort_values("timestamp").reset_index(drop=True)
    print(f"  行数: {len(base_df):,} | カラム数: {len(base_df.columns)}")

    # ── 4. グループ別ターゲット生成 + OOFスコア生成 ───────────
    all_oof: dict[str, pd.Series] = {}

    for group, cols in groups.items():
        valid_cols = [c for c in cols if c in valid_lf_features]
        if not valid_cols:
            print(f"\n[{group}] 有効な特徴量がないためスキップ。")
            all_oof[f"{group}_score"] = pd.Series(
                np.nan, index=base_df.index, dtype=np.float32
            )
            continue

        print(f"\n{'─'*40}")
        print(f"グループ処理: {group}  ({len(valid_cols)} 特徴量)")

        # M1粒度shiftでターゲット生成
        target_shift = WF_TARGET_SHIFT[group]
        group_df = base_df.copy()
        future_close = group_df["close"].shift(target_shift)  # 負値 → 前向きシフト
        group_df["target"] = (future_close > group_df["close"]).astype(float)
        group_df.loc[future_close.isna(), "target"] = np.nan

        oof = run_wf_for_group(
            df=group_df,
            feature_cols=valid_cols,
            group=group,
            test_mode=test_mode,
        )
        all_oof[f"{group}_score"] = oof.astype(np.float32)
        covered = oof.notna().sum()
        print(f"  [{group}] OOFカバレッジ: {covered:,} / {len(oof):,} 行")

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
    print(f"\nLF環境スコア保存: {S3_LF_ENVIRONMENT_SCORES}")
    print(f"  カラム: {result_pl.columns}")
    print(f"  行数  : {len(result_pl):,}")

    # ── 7. S3_FINAL_FEATURE_TEAM への保存 ─────────────────────
    # 旧設計との互換性維持: フィルターを通過した全LF特徴量名を出力
    with open(S3_FINAL_FEATURE_TEAM, "w") as f:
        for feat in valid_lf_features:
            f.write(feat + "\n")

    print(f"最終LF特徴量チーム保存: {S3_FINAL_FEATURE_TEAM} ({len(valid_lf_features)} 件)")
    print("\n2_C_lf_wf_signal_generator.py 正常終了。")


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
