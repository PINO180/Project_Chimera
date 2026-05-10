#!/usr/bin/env python3
"""
OOB (Out-Of-Bin) フィルター BTテスト用スクリプト

【目的】
学習データ各特徴量の P0.5〜P99.5 を計算し、
推論時に値域外特徴量数 (OOB count) が閾値を超えるトレードをHOLDに変換した場合の
BT結果を再現して、現行 PF 19.09 と比較する。

【設計】
1. 学習特徴量データ (feature parquet) を読み込み
2. M2モデルが使う特徴量リスト (m2_long/short_features.txt) を読み込み
3. 各特徴量の P0.5 / P99.5 を全期間データから計算 (これが「学習分布の境界」)
4. BTトレードログと特徴量データを timestamp でjoin
5. 各トレードでの OOB count を計算
6. 複数の OOB threshold (10/30/50/100/200/300/500/1000) でフィルター
7. 各閾値での: トレード数, 勝率, PF, 総PnL, 除外日数 を比較

【設定】
スクリプト先頭の CONFIG セクションで全パスを指定可能。
"""

import sys
from pathlib import Path
import polars as pl
import numpy as np

# =========================================================
# CONFIG: 環境に合わせて編集
# =========================================================
BT_LOG_PATH = (
    "/workspace/data/XAUUSD/stratum_7_models/backtest_simulator_results/"
    "M2_20260504_195717_Th0.7_D0.3_R2 _Phase6_V5/detailed_trade_log_v5_M2.csv"
)

# 特徴量データ (BTシミュレーターと同じく S6_WEIGHTED_DATASET を使う)
# パーティション形式: weighted_dataset_partitioned_v2/**/*.parquet
# S6_WEIGHTED_DATASET の実体パスを下記候補から自動検出
FEATURE_PATH_CANDIDATES = [
    "/workspace/data/XAUUSD/stratum_6_training/weighted_dataset_partitioned_v2",
    "/workspace/data/XAUUSD/stratum_6/weighted_dataset_partitioned_v2",
    "/workspace/data/XAUUSD/stratum_6_weighted/weighted_dataset_partitioned_v2",
    "/workspace/data/stratum_6_training/weighted_dataset_partitioned_v2",
    # 見つからなければ広く検索する
]

# M2 特徴量リスト（611件）
M2_FEATURE_LIST_PATH = (
    "/workspace/data/XAUUSD/stratum_3_artifacts/selected_features_orthogonal_v5/m2_long_features.txt"
)

# OOB閾値スイープ
OOB_THRESHOLDS = [10, 30, 50, 100, 200, 300, 500, 1000]

# 学習分布として使う期間の比率（最初の何%を「学習データ」と見なすか）
# 0.0 = 全期間で percentile を計算（簡易）
# 0.8 = 最初の80%を学習として、percentileを計算
TRAIN_FRACTION = 0.0  # 全期間で計算（簡易、percentileは安定）

# percentile lower / upper
PERCENTILE_LOW = 0.5    # 0.5%ile
PERCENTILE_HIGH = 99.5  # 99.5%ile


def find_feature_data():
    """候補パスから feature parquet を見つける。見つからなければ広域検索。"""
    # 1. CONFIGの候補を順試行
    for cand in FEATURE_PATH_CANDIDATES:
        p = Path(cand)
        if p.exists() and p.is_dir():
            parquets = list(p.rglob("*.parquet"))
            if parquets:
                print(f"特徴量データ見つけた: {p} ({len(parquets)} parquet)")
                return parquets

    # 2. /workspace 配下で weighted_dataset_partitioned_v2 を広域検索
    print("候補パス全滅。/workspace 配下で weighted_dataset_partitioned_v2 を検索中...")
    workspace = Path("/workspace")
    if workspace.exists():
        for sub in workspace.rglob("weighted_dataset_partitioned_v2"):
            if sub.is_dir():
                parquets = list(sub.rglob("*.parquet"))
                if parquets:
                    print(f"特徴量データ見つけた: {sub} ({len(parquets)} parquet)")
                    return parquets

    # 3. 最後の手段: 大きめの parquet を探す (1MB以上)
    print("weighted_dataset_partitioned_v2 が見つからない。大きめparquetを検索...")
    if workspace.exists():
        large_parquets = []
        for pq in workspace.rglob("*.parquet"):
            try:
                if pq.stat().st_size > 1_000_000:  # 1MB以上
                    large_parquets.append(pq)
                if len(large_parquets) > 50:
                    break
            except:
                continue
        if large_parquets:
            print(f"\n候補となる大きめparquetが見つかった (上位10件):")
            for p in large_parquets[:10]:
                print(f"  {p} ({p.stat().st_size / 1e6:.1f} MB)")
            print("\nスクリプトの FEATURE_PATH_CANDIDATES を編集してください。")

    return None


def load_features(parquet_files: list, feature_cols: list) -> pl.DataFrame:
    """
    parquetを読み込み、必要な特徴量と timestamp を取得
    """
    dfs = []
    needed = ["timestamp"] + feature_cols
    
    for pf in parquet_files:
        try:
            # スキーマ確認
            schema = pl.read_parquet_schema(pf)
            available = [c for c in needed if c in schema]
            if "timestamp" not in available:
                continue
            df = pl.read_parquet(pf, columns=available)
            dfs.append(df)
        except Exception as e:
            print(f"WARN: {pf} 読み込み失敗: {e}")
            continue
    
    if not dfs:
        return None
    
    df = pl.concat(dfs, how="diagonal")
    df = df.unique(subset=["timestamp"]).sort("timestamp")
    return df


def main():
    # === 1. BTログ読み込み ===
    if not Path(BT_LOG_PATH).exists():
        print(f"ERROR: BT log が見つからない: {BT_LOG_PATH}")
        sys.exit(1)
    
    bt = pl.read_csv(BT_LOG_PATH, try_parse_dates=True)
    print(f"BTログ: {len(bt):,} 取引")
    
    # timestamp を datetime に
    if bt["timestamp"].dtype == pl.String:
        bt = bt.with_columns(pl.col("timestamp").str.to_datetime())
    bt = bt.sort("timestamp")
    print(f"  期間: {bt['timestamp'].min()} 〜 {bt['timestamp'].max()}")
    
    # === 2. M2特徴量リスト読み込み ===
    if not Path(M2_FEATURE_LIST_PATH).exists():
        print(f"ERROR: M2特徴量リストが見つからない: {M2_FEATURE_LIST_PATH}")
        sys.exit(1)
    
    with open(M2_FEATURE_LIST_PATH, "r") as f:
        m2_features = [line.strip() for line in f if line.strip()]
    print(f"M2特徴量数: {len(m2_features)}")
    
    # === 3. 特徴量データ探索 ===
    feature_files = find_feature_data()
    if feature_files is None:
        print("\nERROR: 特徴量データが見つからない。CONFIGの FEATURE_PATH_CANDIDATES を編集してください。")
        print("候補:")
        for c in FEATURE_PATH_CANDIDATES:
            print(f"  {c}")
        sys.exit(1)
    
    # === 4. 特徴量データ読み込み ===
    print(f"\n特徴量データ読み込み中... (M2の611特徴量 + timestamp)")
    feat_df = load_features(feature_files, m2_features)
    if feat_df is None:
        print("ERROR: 特徴量データ読み込み失敗")
        sys.exit(1)
    
    available_features = [c for c in m2_features if c in feat_df.columns]
    print(f"  利用可能な特徴量: {len(available_features)} / {len(m2_features)}")
    print(f"  特徴量データ: {len(feat_df):,} 行")
    
    if len(available_features) < 100:
        print(f"\nWARN: 利用可能な特徴量が少ない ({len(available_features)}件)。")
        print("特徴量データのスキーマを確認してください。")
    
    # === 5. percentile計算 ===
    print(f"\nP{PERCENTILE_LOW} / P{PERCENTILE_HIGH} 計算中...")
    
    # 学習期間の特定
    if TRAIN_FRACTION > 0:
        cutoff_idx = int(len(feat_df) * TRAIN_FRACTION)
        cutoff_ts = feat_df["timestamp"].sort()[cutoff_idx]
        train_df = feat_df.filter(pl.col("timestamp") <= cutoff_ts)
        print(f"  学習期間: 〜{cutoff_ts} ({len(train_df):,} 行)")
    else:
        train_df = feat_df
        print(f"  全期間 ({len(train_df):,} 行) で percentile 計算")
    
    p_low = {}
    p_high = {}
    for col in available_features:
        p_low[col] = train_df[col].quantile(PERCENTILE_LOW / 100)
        p_high[col] = train_df[col].quantile(PERCENTILE_HIGH / 100)
    
    print(f"  完了: {len(p_low)} 特徴量")
    
    # === 6. BTログと特徴量を join ===
    print(f"\nBTログと特徴量を timestamp で join 中...")
    
    # BTログ側の timestamp は entry_time
    bt_ts = bt.select(["timestamp", "direction", "m2_proba", "pnl", "TD"]).sort("timestamp")
    feat_ts = feat_df.select(["timestamp"] + available_features).sort("timestamp")
    
    # タイムゾーンを揃える (両方 naive [μs] に統一)
    bt_dtype = bt_ts["timestamp"].dtype
    feat_dtype = feat_ts["timestamp"].dtype
    print(f"  bt timestamp dtype:   {bt_dtype}")
    print(f"  feat timestamp dtype: {feat_dtype}")
    
    # 両方を naive datetime[μs] に揃える (replace_time_zone(None))
    if isinstance(bt_dtype, pl.Datetime) and bt_dtype.time_zone is not None:
        bt_ts = bt_ts.with_columns(pl.col("timestamp").dt.replace_time_zone(None))
    if isinstance(feat_dtype, pl.Datetime) and feat_dtype.time_zone is not None:
        feat_ts = feat_ts.with_columns(pl.col("timestamp").dt.replace_time_zone(None))
    
    # join_asof で M3 entry_time に最も近い直近の特徴量を取る
    joined = bt_ts.join_asof(
        feat_ts,
        on="timestamp",
        strategy="backward",
    )
    
    # join成功率
    null_count = joined.select(pl.col(available_features[0]).is_null().sum()).item()
    print(f"  join後: {len(joined):,} 取引、特徴量NULL: {null_count}")
    
    if null_count > len(joined) * 0.5:
        print("WARN: 半数以上のトレードで特徴量取得失敗。timestamp精度の不一致の可能性。")
    
    # === 7. OOB count 計算 ===
    print(f"\n各取引の OOB count 計算中...")
    
    feat_arr = joined.select(available_features).to_numpy()  # (n_trades, n_features)
    n_trades = feat_arr.shape[0]
    
    p_low_arr = np.array([p_low[c] for c in available_features])
    p_high_arr = np.array([p_high[c] for c in available_features])
    
    # OOB判定: 値が p_low 未満 or p_high 超過
    is_oob = (feat_arr < p_low_arr) | (feat_arr > p_high_arr)
    is_oob = is_oob & ~np.isnan(feat_arr)  # NaNはOOB扱いしない
    oob_count = is_oob.sum(axis=1)
    
    joined = joined.with_columns(pl.Series("oob_count", oob_count))
    
    print(f"  OOB count 統計:")
    print(f"    平均: {oob_count.mean():.2f}")
    print(f"    中央: {np.median(oob_count):.0f}")
    print(f"    P95 : {np.percentile(oob_count, 95):.0f}")
    print(f"    P99 : {np.percentile(oob_count, 99):.0f}")
    print(f"    最大: {oob_count.max()}")
    
    # === 8. 閾値スイープ ===
    print("\n" + "=" * 90)
    print("OOBフィルター 閾値スイープ結果")
    print("=" * 90)
    print(f"{'閾値':>6} {'取引数':>9} {'除外':>9} {'除外率':>8} {'勝率':>8} "
          f"{'TO率':>8} {'総PnL':>14} {'PF':>7} {'平均PnL':>10}")
    print("-" * 90)
    
    # ベースライン (フィルターなし)
    base_trades = joined.filter(pl.col("oob_count").is_not_null())
    base_n = len(base_trades)
    base_pt = base_trades.filter(pl.col("pnl") > 0).height
    base_to = base_trades.filter((pl.col("pnl") <= 0) & (pl.col("TD") >= 29.5)).height
    base_pnl = base_trades["pnl"].sum()
    base_win_pnl = base_trades.filter(pl.col("pnl") > 0)["pnl"].sum()
    base_loss_pnl = abs(base_trades.filter(pl.col("pnl") <= 0)["pnl"].sum())
    base_pf = base_win_pnl / base_loss_pnl if base_loss_pnl > 0 else float('inf')
    
    print(f"{'なし':>6} {base_n:>9,} {0:>9,} {0:>7.1f}% "
          f"{base_pt/base_n*100:>7.2f}% {base_to/base_n*100:>7.2f}% "
          f"${base_pnl:>13,.0f} {base_pf:>7.2f} ${base_pnl/base_n:>9,.2f}")
    
    # 閾値別
    results = []
    for thresh in OOB_THRESHOLDS:
        kept = joined.filter(
            (pl.col("oob_count").is_not_null()) & (pl.col("oob_count") <= thresh)
        )
        excluded = base_n - len(kept)
        if len(kept) == 0:
            continue
        
        n = len(kept)
        pt = kept.filter(pl.col("pnl") > 0).height
        to = kept.filter((pl.col("pnl") <= 0) & (pl.col("TD") >= 29.5)).height
        pnl_total = kept["pnl"].sum()
        win_pnl = kept.filter(pl.col("pnl") > 0)["pnl"].sum()
        loss_pnl = abs(kept.filter(pl.col("pnl") <= 0)["pnl"].sum())
        pf = win_pnl / loss_pnl if loss_pnl > 0 else float('inf')
        
        print(f"{thresh:>6} {n:>9,} {excluded:>9,} {excluded/base_n*100:>7.1f}% "
              f"{pt/n*100:>7.2f}% {to/n*100:>7.2f}% "
              f"${pnl_total:>13,.0f} {pf:>7.2f} ${pnl_total/n:>9,.2f}")
        
        results.append({
            "threshold": thresh,
            "trades": n,
            "excluded": excluded,
            "win_rate": pt / n * 100,
            "to_rate": to / n * 100,
            "total_pnl": pnl_total,
            "pf": pf,
            "avg_pnl": pnl_total / n,
        })
    
    # === 9. 解釈 ===
    print("\n" + "=" * 90)
    print("解釈")
    print("=" * 90)
    print("""
【見方】
- 「閾値」: OOB count がこの値以下のトレードのみを採用
  例: 閾値=10 → OOB特徴量が10個以下のトレードのみ実行 (厳しい)
       閾値=500 → OOB特徴量が500個以下 (大半のトレードが通過、緩い)

【判定基準】
- ベースライン (フィルターなし) の PF/総PnL が現行
- 閾値を下げる (フィルター強化) で PF が改善 → OOBフィルターは有効
- 閾値を下げると PF が悪化 → OOBフィルターは過剰除外

【現実的な判断材料】
- PF微増 + 取引数大幅減 → 単利フェーズには不利 (機会損失)
- PF大幅増 + 取引数微減 → 採用候補
- PF不変 + 取引数微減 → 効果なし
""")
    
    # 出力
    out_path = Path(BT_LOG_PATH).parent / "oob_filter_test_results.csv"
    pl.DataFrame(results).write_csv(out_path)
    print(f"結果CSV: {out_path}")


if __name__ == "__main__":
    main()
