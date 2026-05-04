"""
2_E_hf_meta_model_trainer.py
==============================
Chapter 2 スクリプト E — HF メタモデルトレーナー

設計意図:
    このスクリプトは旧 03_M2_meta_model_trainer.py をリファクタリングし、
    LFスコアをHF特徴量に連続値で結合する設計に変更したものである。
    本質的に Chapter3 スクリプトC（M2モデル）と同じ構造を持つ。
    将来的に Chapter3 の設計に完全準拠すること。

フロー:
    S2_FEATURES_VALIDATED（HF特徴量）
    + S3_FILTERED_HF_FEATURES（特徴量リスト）
    + S3_LF_ENVIRONMENT_SCORES（LFスコア） を読み込む
        ↓
    LFスコアを join_asof（strategy="backward"）で HF データに結合
        ↓
    ATR を S2_FEATURES_VALIDATED から動的検索して結合
        ↓
    トリプルバリアラベルを生成（time_between 法・tick データ不要）
        ↓
    Purged KFold CV（purge_days=3・embargo_days=2）で LightGBM を学習
        ↓
    HF 生き残り特徴量リストを S3_SURVIVED_HF_FEATURES に保存

共通ルール遵守:
    - ルール1: ローリングウィンドウは過去方向のみ参照。未来情報リーク排除。
    - ルール2: ddof=1 統一（Polars の rolling_std 準拠・不偏推定）
    - ルール3: カレンダー依存処理禁止
    - ルール4: ゼロ除算・数値不安定防止（分母 +1e-10）
    - ルール5: Polars API 標準（collect(engine="streaming") 等）
    - ルール6: lambda クロージャーはデフォルト引数で束縛
    - ルール7: ハードコードパス禁止・動的解決
    - ルール8: Float32 使用（LightGBM 入力・分布比較のみ）
"""

import sys
import warnings
import argparse
import datetime
from pathlib import Path
from typing import List

import numpy as np
import polars as pl
import lightgbm as lgb
from tqdm import tqdm

# ルール7: 絶対パスハードコード禁止。動的解決。
sys.path.append(str(Path(__file__).resolve().parents[1]))

import blueprint

# =============================================================================
# blueprint から必須パスをインポート
# =============================================================================
from blueprint import (
    BARRIER_ATR_PERIOD,
    HF_TIMEFRAMES,
    S2_FEATURES_VALIDATED,
    S3_LF_ENVIRONMENT_SCORES,
    S3_FILTERED_HF_FEATURES,
    S3_SURVIVED_HF_FEATURES,
    LABEL_CONFIG,
)

# =============================================================================
# 定数 / 設定
# =============================================================================

# Chapter3 スクリプトC と統一した除外カラムセット
EXCLUDE_COLS = {
    "timestamp",
    "timeframe",
    "label",
    "label_long",
    "label_short",
    "uniqueness",
    "uniqueness_long",
    "uniqueness_short",
    "atr_value",
    "close",
    "open",
    "high",
    "low",
    "is_trigger",
    "meta_label",
    "m1_pred_proba",
    "payoff_ratio",
    "sl_multiplier",
    "pt_multiplier",
    "direction",
}

# LFスコアカラム（二値化せずそのまま特徴量として追加）
LF_SCORE_COLS = ["lf_short_score", "lf_mid_score", "lf_long_score"]

# Chapter3 スクリプトC と同一の LightGBM パラメータ
LGB_PARAMS = {
    "objective": "binary",
    "n_estimators": 2000,
    "learning_rate": 0.01,
    "num_leaves": 127,
    "min_data_in_leaf": 100,
    "lambda_l1": 0.1,
    "lambda_l2": 0.1,
    "colsample_bytree": 0.8,
    "subsample": 0.8,
    "verbosity": -1,
    "random_state": 42,
    "n_jobs": -1,
}

# Purged KFold 設定
PURGE_DAYS = 3
EMBARGO_DAYS = 2


# =============================================================================
# トリプルバリアラベル生成（time_between 法・tick データ不要）
# =============================================================================


def generate_triple_barrier_labels(
    data: pl.DataFrame,
    pt_multiplier: float,
    sl_multiplier: float,
    max_hold_minutes: int,
    min_atr_threshold: float,
    # TODO: 相対値化予定（現段階では ATR 閾値は絶対値 2.0ドル のまま使用）
    # Chapter1・2 の相対値化完了後に相対値化する
) -> pl.DataFrame:
    """
    トリプルバリアラベルを生成する（time_between 法）。

    - PT 倍率・SL 倍率・保有期間上限は risk_config.json から読み込む。
    - ATR 収縮フィルター: ATR <= min_atr_threshold のエントリーをスキップ。
    - ルール4: 分母は +1e-10 でゼロ除算防止。

    Args:
        data: timestamp / close / high / low / atr_value カラムを持つ DataFrame
        pt_multiplier: PT バリア倍率
        sl_multiplier: SL バリア倍率
        max_hold_minutes: 最大保有分数
        min_atr_threshold: ATR 収縮フィルター閾値（絶対値）

    Returns:
        label カラム（1=勝ち, 0=負け/タイムアウト）が追加された DataFrame
    """
    labels: List[int] = []

    ts_col = data["timestamp"].cast(pl.Datetime("us"))
    close_col = data["close"].cast(pl.Float64)
    high_col = data["high"].cast(pl.Float64)
    low_col = data["low"].cast(pl.Float64)
    atr_col = data["atr_value"].cast(pl.Float64)

    n = len(data)

    for i in tqdm(range(n), desc="  Triple-Barrier labeling", leave=False):
        atr = atr_col[i]

        # ATR 収縮フィルター
        # TODO: 相対値化予定（現段階では絶対値 min_atr_threshold で比較）
        if atr is None or np.isnan(atr) or atr <= min_atr_threshold:
            labels.append(-1)  # スキップ（後でフィルタリング）
            continue

        entry_price = close_col[i]
        entry_ts = ts_col[i]
        deadline = entry_ts + datetime.timedelta(
            minutes=max_hold_minutes
        )  # timedeltaを使用

        # ルール4: atr が 0 になりうる場合の保護（+1e-10）
        pt_price = entry_price + pt_multiplier * (atr + 1e-10)
        sl_price = entry_price - sl_multiplier * (atr + 1e-10)

        label = 0
        for j in range(i + 1, n):
            if ts_col[j] > deadline:
                break
            h = high_col[j]
            lo = low_col[j]
            if h is not None and h >= pt_price:
                label = 1
                break
            if lo is not None and lo <= sl_price:
                label = 0
                break

        labels.append(label)

    return data.with_columns(pl.Series("label", labels, dtype=pl.Int8))


# =============================================================================
# Purged KFold CV（Chapter3 スクリプトC と同一設計）
# =============================================================================


def purged_kfold_splits(
    timestamps: np.ndarray,
    n_splits: int = 5,
    purge_days: int = PURGE_DAYS,
    embargo_days: int = EMBARGO_DAYS,
):
    """
    Purged KFold クロスバリデーションのインデックスを生成する。
    purge_days 分のパージと embargo_days 分のエンバーゴを適用する。
    Chapter3 スクリプトC と統一した設計。

    Yields:
        (train_idx, val_idx) のタプル
    """
    n = len(timestamps)
    fold_size = n // n_splits

    purge_td = np.timedelta64(purge_days, "D")
    embargo_td = np.timedelta64(embargo_days, "D")

    for fold in range(n_splits):
        val_start = fold * fold_size
        val_end = val_start + fold_size if fold < n_splits - 1 else n

        val_ts_start = timestamps[val_start]
        val_ts_end = timestamps[val_end - 1]

        train_mask = (timestamps < val_ts_start - purge_td) | (
            timestamps > val_ts_end + embargo_td
        )
        val_mask = np.zeros(n, dtype=bool)
        val_mask[val_start:val_end] = True

        train_idx = np.where(train_mask)[0]
        val_idx = np.where(val_mask)[0]

        if len(train_idx) == 0 or len(val_idx) == 0:
            continue

        yield train_idx, val_idx


# =============================================================================
# メイン処理
# =============================================================================


def main(test_mode: bool = False) -> None:
    """
    2_E_hf_meta_model_trainer メイン処理。

    Args:
        test_mode: True の場合、最初の 500 サンプルのみ使用（動作確認用）
    """
    print("=" * 70)
    print("2_E_hf_meta_model_trainer: HF メタモデル学習開始")
    print("=" * 70)

    # -----------------------------------------------------------------------
    # 0. ラベル生成パラメータ読み込み（blueprint.LABEL_CONFIG）
    # -----------------------------------------------------------------------
    print("\n[Step 0] ラベル生成パラメータ読み込み中...")
    pt_multiplier: float = LABEL_CONFIG["pt_multiplier"]
    sl_multiplier: float = LABEL_CONFIG["sl_multiplier"]
    max_hold_minutes: int = int(LABEL_CONFIG["td_minutes"])
    min_atr_threshold: float = LABEL_CONFIG["min_atr_threshold"]
    # TODO: 相対値化予定（現段階では ATR 閾値は絶対値 2.0ドル のまま使用）

    print(f"  PT 倍率: {pt_multiplier} / SL 倍率: {sl_multiplier}")
    print(f"  最大保有分数: {max_hold_minutes} / ATR 閾値: {min_atr_threshold}")

    # -----------------------------------------------------------------------
    # 1. HF 特徴量ファイル読み込み
    #    S2_FEATURES_VALIDATED から HF_TIMEFRAMES に一致するファイルを収集
    # -----------------------------------------------------------------------
    print("\n[Step 1] HF 特徴量ファイル読み込み中...")

    if not S3_FILTERED_HF_FEATURES.exists():
        raise FileNotFoundError(
            f"HF 特徴量リストが見つかりません: {S3_FILTERED_HF_FEATURES}"
        )

    with open(S3_FILTERED_HF_FEATURES, "r", encoding="utf-8") as f:
        filtered_hf_feature_names = [line.strip() for line in f if line.strip()]
    print(f"  フィルタ済み HF 特徴量数: {len(filtered_hf_feature_names)}")

    # S2_FEATURES_VALIDATED から HF 時間足のファイルを動的収集
    hf_tf_set = set(HF_TIMEFRAMES)
    hf_parquet_paths: List[Path] = []
    for tf in hf_tf_set:
        if tf == "tick":
            continue  # tick は除外（time_between 法では不要）
        for p in S2_FEATURES_VALIDATED.rglob(f"features_*_{tf}.parquet"):
            hf_parquet_paths.append(p)

    if not hf_parquet_paths:
        raise RuntimeError(
            f"S2_FEATURES_VALIDATED に HF parquet ファイルが見つかりません: "
            f"{S2_FEATURES_VALIDATED}"
        )
    # stem のアルファベット順でソート: M0.5 < M1 < M3 … の順になり
    # 最も粒度の細かい（行数の多い）M1 系ファイルがベースになりやすくなる
    hf_parquet_paths = sorted(hf_parquet_paths, key=lambda p: p.stem)
    print(f"  発見した HF parquet ファイル数: {len(hf_parquet_paths)}")

    # 最初のファイルをベースに LazyFrame を構築
    # timeframe カラムを特徴量リストの先頭に追加（順序データとして学習）
    base_path = hf_parquet_paths[0]
    base_tf = base_path.stem.split("_")[-1]  # 例: "M1"

    print(f"  ベースファイル: {base_path} (tf={base_tf})")

    # base_lf の特徴量カラムにもサフィックスを付与し、後段 join_asof での
    # _right 衝突（DuplicateError）を防ぐ
    base_schema = pl.scan_parquet(str(base_path)).collect_schema().names()
    base_feat_cols = [
        c for c in base_schema if c not in EXCLUDE_COLS and c != "timestamp"
    ]
    base_rename_map = {c: f"{c}_{base_tf}" for c in base_feat_cols}
    base_lf = (
        pl.scan_parquet(str(base_path))
        .with_columns(pl.col("timestamp").cast(pl.Datetime("us")))
        .select(["timestamp"] + base_feat_cols)
        .rename(base_rename_map)
        .with_columns([pl.col(v).cast(pl.Float32) for v in base_rename_map.values()])
        .sort("timestamp")
    )

    # -----------------------------------------------------------------------
    # 1.5: トリプルバリアラベル生成用 OHLCV（close/high/low）を結合
    #      エンジン出力ファイルは特徴量のみでOHLCVを持たないため、
    #      S1_PROCESSED の Hive パーティションから別途取得する
    #      close/high/low は EXCLUDE_COLS に含まれるため学習特徴量には使われない
    # -----------------------------------------------------------------------
    print("\n[Step 1.5] OHLCV 結合中...")
    price_dir = blueprint.S1_PROCESSED / f"timeframe={base_tf}"
    if not price_dir.exists():
        raise FileNotFoundError(
            f"S1_PROCESSED に {base_tf} のディレクトリが見つかりません: {price_dir}"
        )
    price_lf = (
        pl.scan_parquet(str(price_dir / "*.parquet"))
        .select(["timestamp", "close", "high", "low"])
        .with_columns(pl.col("timestamp").cast(pl.Datetime("us")))
        .sort("timestamp")
    )
    base_lf = base_lf.join_asof(price_lf, on="timestamp", strategy="backward")
    print(f"  OHLCV 結合完了 (source: {price_dir})")

    # -----------------------------------------------------------------------
    # 2. LF スコアを join_asof（strategy="backward"）で結合
    #    全特徴量が M1 粒度のため join_asof での時間軸ズレは発生しない
    # -----------------------------------------------------------------------
    print("\n[Step 2] LF スコア結合中...")

    if not S3_LF_ENVIRONMENT_SCORES.exists():
        raise FileNotFoundError(
            f"LF スコアファイルが見つかりません: {S3_LF_ENVIRONMENT_SCORES}"
        )

    lf_score_lf = (
        pl.scan_parquet(str(S3_LF_ENVIRONMENT_SCORES))
        .with_columns(pl.col("timestamp").cast(pl.Datetime("us")))
        .select(["timestamp"] + LF_SCORE_COLS)
        .with_columns([pl.col(c).cast(pl.Float32) for c in LF_SCORE_COLS])
        .sort("timestamp")
    )

    base_lf = base_lf.join_asof(lf_score_lf, on="timestamp", strategy="backward")
    print(f"  LF スコア ({LF_SCORE_COLS}) を backward join で結合完了")

    # -----------------------------------------------------------------------
    # 3. ATR を S1_PROCESSED の OHLCV から自前計算（Wilderの平滑化 = RMA）
    #    Engine1C の出力は相対値化されているため絶対値（ドル値）をここで算出する。
    #    TR = max(high-low, |high-prev_close|, |low-prev_close|)
    #    ATR = ewm_mean(TR, alpha=1/period, adjust=False)  ← Wilder RMA
    #    ルール1: adjust=False で過去方向のみ参照（adjust=True は未来リーク禁止）
    # -----------------------------------------------------------------------
    print(f"\n[Step 3] ATR 算出中 (period={BARRIER_ATR_PERIOD}, Wilder RMA)...")
    base_lf = base_lf.with_columns(
        pl.max_horizontal(
            pl.col("high") - pl.col("low"),
            (pl.col("high") - pl.col("close").shift(1)).abs(),
            (pl.col("low") - pl.col("close").shift(1)).abs(),
        )
        .ewm_mean(alpha=1 / BARRIER_ATR_PERIOD, adjust=False)
        .cast(pl.Float32)
        .alias("atr_value")
    )

    # -----------------------------------------------------------------------
    # 4. 他 HF ファイルを join_asof で結合
    # -----------------------------------------------------------------------
    print("\n[Step 4] 追加 HF 特徴量を join_asof で結合中...")

    # already_joined でサフィックス付きカラム名を追跡し、
    # 同一カラムの二重 join による DuplicateError（_right 衝突）を防ぐ
    already_joined: set = set(base_lf.collect_schema().names())

    for path in hf_parquet_paths[1:]:
        tf_tag = path.stem.split("_")[-1]
        try:
            schema = pl.scan_parquet(str(path)).collect_schema().names()
        except Exception as e:
            print(f"  [WARN] スキーマ取得失敗: {path} -> {e}")
            continue

        # EXCLUDE_COLS 以外の特徴量のみ選択してサフィックスを付与
        feat_cols = [c for c in schema if c not in EXCLUDE_COLS and c != "timestamp"]
        rename_map = {c: f"{c}_{tf_tag}" for c in feat_cols}
        # 既に結合済みのサフィックス付き名はスキップ（重複 join 防止）
        new_rename = {k: v for k, v in rename_map.items() if v not in already_joined}
        if not new_rename:
            continue

        sub_lf = (
            pl.scan_parquet(str(path))
            .with_columns(pl.col("timestamp").cast(pl.Datetime("us")))
            .select(["timestamp"] + list(new_rename.keys()))
            .rename(new_rename)
            .with_columns(
                # ルール8: Float32
                [pl.col(v).cast(pl.Float32) for v in new_rename.values()]
            )
            .sort("timestamp")
        )
        base_lf = base_lf.join_asof(sub_lf, on="timestamp", strategy="backward")
        # 結合済みカラムを記録（次ループで重複チェックに使用）
        already_joined.update(new_rename.values())

    # -----------------------------------------------------------------------
    # 5. collect
    # -----------------------------------------------------------------------
    print("\n[Step 5] データを collect 中...")
    data_df = base_lf.collect(engine="streaming")

    if test_mode:
        data_df = data_df.head(500)
        print(f"  [TEST MODE] 先頭 500 サンプルのみ使用")

    print(f"  collect 完了: {data_df.shape[0]} 行 × {data_df.shape[1]} カラム")

    # close / high / low / atr_value が存在するか確認
    required_cols = {"timestamp", "close", "high", "low", "atr_value"}
    missing = required_cols - set(data_df.columns)
    if missing:
        raise RuntimeError(f"必須カラムが不足しています: {missing}")

    # -----------------------------------------------------------------------
    # 6. トリプルバリアラベル生成
    # -----------------------------------------------------------------------
    print("\n[Step 6] トリプルバリアラベル生成中...")
    data_df = generate_triple_barrier_labels(
        data=data_df,
        pt_multiplier=pt_multiplier,
        sl_multiplier=sl_multiplier,
        max_hold_minutes=max_hold_minutes,
        min_atr_threshold=min_atr_threshold,
    )

    # ATR フィルター適用（label == -1 をスキップ）
    before_n = len(data_df)
    data_df = data_df.filter(pl.col("label") >= 0)
    after_n = len(data_df)
    print(f"  ATR 収縮フィルター: {before_n - after_n} 行除外 → {after_n} 行残存")

    if after_n == 0:
        print("[ERROR] ラベル生成後にサンプルが残りませんでした。処理を終了します。")
        return

    pos_rate = data_df["label"].cast(pl.Float32).mean()
    print(f"  ポジティブレート: {pos_rate:.4f}")

    # -----------------------------------------------------------------------
    # 7. 特徴量カラム選定
    #    - filtered_hf_feature_names に含まれるもの
    #    - EXCLUDE_COLS を除外
    #    - LF スコアを追加（二値化しない）
    #    - timeframe カラムを先頭に追加（順序データとして学習）
    # -----------------------------------------------------------------------
    print("\n[Step 7] 特徴量カラム選定中...")

    all_cols = set(data_df.columns)
    candidate_features = set(filtered_hf_feature_names) | set(LF_SCORE_COLS)
    candidate_features -= EXCLUDE_COLS

    feature_cols: List[str] = []

    # timeframe カラムを先頭に追加（順序データ）
    if "timeframe" in all_cols:
        # timeframe をラベルエンコードして Float32 に変換
        data_df = data_df.with_columns(
            pl.col("timeframe")
            .cast(pl.Categorical)
            .to_physical()
            .cast(pl.Float32)
            .alias("timeframe_encoded")
        )
        feature_cols.append("timeframe_encoded")

    # 残りの特徴量を追加（データに存在するものだけ）
    for col in sorted(candidate_features):
        if col in all_cols and col not in feature_cols:
            feature_cols.append(col)

    # LF スコアを確実に追加
    for lf_col in LF_SCORE_COLS:
        if lf_col in all_cols and lf_col not in feature_cols:
            feature_cols.append(lf_col)

    print(f"  使用特徴量数: {len(feature_cols)}")

    if not feature_cols:
        print("[ERROR] 有効な特徴量が見つかりませんでした。処理を終了します。")
        return

    # Float32 に統一（ルール8）
    data_df = data_df.with_columns(
        [pl.col(c).cast(pl.Float32) for c in feature_cols if c in data_df.columns]
    )

    # -----------------------------------------------------------------------
    # 8. Purged & Embargoed KFold CV で LightGBM 学習
    #    Chapter3 スクリプトC と同一の LGB_PARAMS・Purged & Embargoed KFold 設計
    # -----------------------------------------------------------------------
    print("\n[Step 8] Purged & Embargoed KFold CV で LightGBM 学習中...")

    X_all = data_df.select(feature_cols).to_numpy().astype(np.float32)
    y_all = data_df["label"].to_numpy().astype(np.int32)
    ts_all = data_df["timestamp"].cast(pl.Datetime("us")).to_numpy()

    oof_proba = np.full(len(y_all), np.nan, dtype=np.float64)
    feature_gain_sum = np.zeros(len(feature_cols), dtype=np.float64)
    fold_count = 0

    splits = list(
        purged_kfold_splits(
            ts_all, n_splits=5, purge_days=PURGE_DAYS, embargo_days=EMBARGO_DAYS
        )
    )

    for fold_idx, (train_idx, val_idx) in enumerate(splits):
        print(
            f"  Fold {fold_idx + 1}/{len(splits)}: "
            f"train={len(train_idx)}, val={len(val_idx)}"
        )

        X_tr, y_tr = X_all[train_idx], y_all[train_idx]
        X_val, y_val = X_all[val_idx], y_all[val_idx]

        # クラス不均衡対応（scale_pos_weight）
        pos_count = y_tr.sum()
        neg_count = len(y_tr) - pos_count
        # ルール4: ゼロ除算防止
        scale_pos_weight = neg_count / (pos_count + 1e-10)

        params = dict(LGB_PARAMS)
        params["scale_pos_weight"] = scale_pos_weight

        model = lgb.LGBMClassifier(**params)
        model.fit(
            X_tr,
            y_tr,
            eval_set=[(X_val, y_val)],
            callbacks=[lgb.early_stopping(stopping_rounds=50, verbose=False)],
            feature_name=feature_cols,
        )

        oof_proba[val_idx] = model.predict_proba(X_val)[:, 1]
        feature_gain_sum += np.array(
            model.booster_.feature_importance(importance_type="gain"),
            dtype=np.float64,
        )
        fold_count += 1

    if fold_count == 0:
        print("[ERROR] 有効な fold がありませんでした。処理を終了します。")
        return

    # OOF AUC
    valid_mask = ~np.isnan(oof_proba)
    if valid_mask.sum() > 0:
        from sklearn.metrics import roc_auc_score

        oof_auc = roc_auc_score(y_all[valid_mask], oof_proba[valid_mask])
        print(f"\n  ★ OOF AUC (Purged & Embargoed KFold): {oof_auc:.4f}")
    else:
        print("  [WARN] OOF 予測が空です。")

    # -----------------------------------------------------------------------
    # 9. Gain > 0 の特徴量を生き残りとして保存
    # -----------------------------------------------------------------------
    print("\n[Step 9] Gain > 0 の特徴量を保存中...")

    mean_gain = feature_gain_sum / (fold_count + 1e-10)
    survived = [feature_cols[i] for i, g in enumerate(mean_gain) if g > 0]
    print(f"  生き残り特徴量数: {len(survived)} / {len(feature_cols)}")

    # 出力ディレクトリ作成
    S3_SURVIVED_HF_FEATURES.parent.mkdir(parents=True, exist_ok=True)

    with open(S3_SURVIVED_HF_FEATURES, "w", encoding="utf-8") as f:
        for feat in survived:
            f.write(feat + "\n")

    print(f"  保存完了: {S3_SURVIVED_HF_FEATURES}")
    print("\n2_E_hf_meta_model_trainer: 処理完了")


# =============================================================================
# エントリーポイント
# =============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="2_E_hf_meta_model_trainer — HF Meta-Model Trainer (Chapter2-E)"
    )
    parser.add_argument(
        "--test-mode",
        action="store_true",
        help="テストモード: 先頭 500 サンプルのみ使用して動作確認",
    )
    args = parser.parse_args()

    warnings.filterwarnings("ignore", category=UserWarning)
    try:
        from polars.exceptions import PolarsInefficientMapWarning

        warnings.filterwarnings("ignore", category=PolarsInefficientMapWarning)
    except ImportError:
        pass

    main(test_mode=args.test_mode)
