import sys
import time
import logging
import traceback
import gc
import shutil
from pathlib import Path

import dask
import dask.dataframe as dd
import pandas as pd
import numpy as np

pd.set_option("future.no_silent_downcasting", True)

# --- パス解決と設定の読み込み ---
sys.path.append(str(Path(__file__).resolve().parents[1]))
import blueprint as config

# --- ロギング設定 ---
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


# --- 設定クラス ---
class ProcessingConfig:
    """前処理および検証のパラメータ設定クラス"""

    SCHEDULER = "single-threaded"  # 安全・確実な処理のためデフォルトはシングルスレッド

    # 計算パラメータ
    VOLATILITY_WINDOW = 20
    VOLUME_WINDOW = 50
    MOMENTUM_WINDOW = 5

    # 除外カラム (元データに存在する場合)
    COLUMNS_TO_DROP = [
        "volatility",
        "avg_volume",
        "tick_count",
        "bid",
        "ask",
        "spread",
        "price_change_pct",
        "range",
        "body_size",
        "atr",  # ← 追加：engine側で計算するため、ここでの計算結果は不要
    ]

    # 品質検証対象カラム
    FEATURES_TO_CHECK = [
        "log_return",
        "rolling_volatility",
        "rolling_avg_volume",
        "price_direction",
        "price_momentum",
        "volume_ratio",
    ]


# --- ゴールデンスキーマ定義（変更不可） ---
GOLDEN_SCHEMA = {
    "timestamp": "datetime64[ns]",
    "open": "float64",
    "high": "float64",
    "low": "float64",
    "close": "float64",
    "volume": "int64",
    "log_return": "float64",
    "rolling_volatility": "float64",
    "rolling_avg_volume": "float64",
    "price_direction": "int8",
    "price_momentum": "float64",
    "volume_ratio": "float64",
    "timeframe": "string",
}


def get_meta(base_columns: list) -> pd.DataFrame:
    """Daskの計算グラフ用メタデータ(スキーマ構造)を生成"""
    meta_dict = {}

    # 既存カラムをスキーマから抽出
    for col in base_columns:
        if col in GOLDEN_SCHEMA:
            # Pandas互換のためにstring型はobjectとして一時定義
            dtype = "object" if GOLDEN_SCHEMA[col] == "string" else GOLDEN_SCHEMA[col]
            meta_dict[col] = pd.Series(dtype=dtype)

    # 新規追加される特徴量カラムを追加
    for col in ProcessingConfig.FEATURES_TO_CHECK:
        dtype = "object" if GOLDEN_SCHEMA[col] == "string" else GOLDEN_SCHEMA[col]
        meta_dict[col] = pd.Series(dtype=dtype)

    return pd.DataFrame(meta_dict)


def calculate_enhanced_features(
    df: pd.DataFrame, global_volume_mean: float = 0.0
) -> pd.DataFrame:
    """特徴量の計算とprice_directionのゼロ値パッチを行うロジック"""

    # ※【禁止事項】
    # overnight_gapや日次リセット処理など、カレンダーに依存する処理は
    # 本スクリプト（基盤特徴量付与）では行わず、後続のengine側に一任すること。

    if len(df) == 0:
        return df

    # 時系列ソートとインデックスリセット
    df = df.sort_values("timestamp").reset_index(drop=True)

    # 1. 対数リターン
    close_shifted = df["close"].shift(1).replace(0, 1e-12)
    ratio = df["close"] / close_shifted
    df["log_return"] = np.log(ratio.fillna(1.0))

    # 2. ローリングボラティリティ
    # pandasのデフォルトはddof=1（不偏推定）。
    # 後続のnumpy等での計算においても、必ずddof=1を使用すること。
    df["rolling_volatility"] = (
        df["log_return"]
        .rolling(window=ProcessingConfig.VOLATILITY_WINDOW, min_periods=1)
        .std(ddof=1)
    )

    # 3. ローリング平均出来高
    fallback_mean = (
        global_volume_mean if global_volume_mean > 0 else df["volume"].mean()
    )
    df["rolling_avg_volume"] = (
        df["volume"]
        .rolling(window=ProcessingConfig.VOLUME_WINDOW, min_periods=1)
        .mean()
        .fillna(fallback_mean)
    )

    # 4. 価格方向性パッチ (0をpd.NAに置換 -> 前方補完 -> 先頭は-1)
    price_diff = df["close"].diff()
    df["price_direction"] = np.sign(price_diff.fillna(0.0)).astype("int8")
    df["price_direction"] = (
        df["price_direction"].replace(0, pd.NA).ffill().fillna(-1).astype("int8")
    )

    # 5. 価格モメンタム
    close_shifted_momentum = (
        df["close"].shift(ProcessingConfig.MOMENTUM_WINDOW).replace(0, 1e-12)
    )
    ratio_momentum = df["close"] / close_shifted_momentum
    df["price_momentum"] = ratio_momentum - 1

    # 6. 出来高比率
    safe_avg_vol = df["rolling_avg_volume"].replace(0, 1.0)
    df["volume_ratio"] = df["volume"] / safe_avg_vol

    # GOLDEN_SCHEMAに合わせて型キャスト（Dask側での完全キャスト前の一時処理）
    for col, dtype in GOLDEN_SCHEMA.items():
        if col in df.columns:
            if dtype == "string":
                df[col] = df[col].astype(str)
            else:
                df[col] = df[col].astype(dtype)

    return df


def verify_data() -> dict:
    """処理後のデータを読み込み、品質検証を行う（警告のみで処理は止めない）"""
    logger.info("--- 品質検証（ゼロ値・NA・infのチェック）を開始します ---")

    # 検証用にDask DataFrameを再読み込み
    ddf = dd.read_parquet(config.S1_PROCESSED, blocksize="128MB")
    all_timeframes = sorted(ddf["timeframe"].unique().compute().tolist())

    summary_counts = {}

    for tf in all_timeframes:
        logger.info(f"検証中: タイムフレーム '{tf}'")
        tf_ddf = ddf[ddf["timeframe"] == tf]

        # 行数カウント（サマリー用）
        row_count = tf_ddf.shape[0].compute()
        summary_counts[tf] = row_count

        for col in ProcessingConfig.FEATURES_TO_CHECK:
            if col not in tf_ddf.columns:
                continue

            # 1. ゼロ値チェック
            zeros_count = (tf_ddf[col] == 0).sum().compute()
            if zeros_count > 0:
                logger.warning(
                    f"[{tf}] カラム '{col}': ゼロ値が {zeros_count:,} 個あります。"
                )

            # 2. 欠損値(NA)チェック
            nas_count = tf_ddf[col].isna().sum().compute()
            if nas_count > 0:
                logger.warning(
                    f"[{tf}] カラム '{col}': 欠損値(NA)が {nas_count:,} 個あります。"
                )

            # 3. 無限大(inf)チェック
            pos_inf_count = (tf_ddf[col] == np.inf).sum().compute()
            if pos_inf_count > 0:
                logger.warning(
                    f"[{tf}] カラム '{col}': 無限大(inf)が {pos_inf_count:,} 個あります。"
                )

            # 4. 無限大(-inf)チェック
            neg_inf_count = (tf_ddf[col] == -np.inf).sum().compute()
            if neg_inf_count > 0:
                logger.warning(
                    f"[{tf}] カラム '{col}': 無限大(-inf)が {neg_inf_count:,} 個あります。"
                )

    logger.info("--- 品質検証が完了しました ---")
    return summary_counts


def main():
    start_time = time.time()

    # Daskスケジューラの設定
    dask.config.set({"scheduler": ProcessingConfig.SCHEDULER})
    logger.info("=== 第3工程: S1_1_C_ENRICH パイプライン開始 ===")

    if not config.S1_MULTITIMEFRAME.exists():
        logger.error(f"入力データが見つかりません: {config.S1_MULTITIMEFRAME}")
        sys.exit(1)

    if config.S1_PROCESSED.exists():
        logger.info(f"出力先ディレクトリを初期化します: {config.S1_PROCESSED}")
        shutil.rmtree(config.S1_PROCESSED, ignore_errors=True)

    try:
        # フォルダ構造から直接タイムフレーム一覧を取得（Daskでの全体読み込みを回避）
        all_timeframes = [
            d.name.replace("timeframe=", "")
            for d in config.S1_MULTITIMEFRAME.iterdir()
            if d.is_dir() and d.name.startswith("timeframe=")
        ]

        # meta_df用のベースカラム（読み込み元に存在する想定のカラム）
        base_columns = [
            "timeframe",
            "timestamp",
            "open",
            "high",
            "low",
            "close",
            "volume",
            "tick_count",
            "bid",
            "ask",
            "spread",
        ]
        meta_df = get_meta(base_columns)

        # 時間足ごとに計算から保存までを逐次実行
        for tf in all_timeframes:
            logger.info(f"--- 時間軸 '{tf}' の処理を開始します ---")

            # 時間足ごとのディレクトリを直接読み込む
            tf_path = config.S1_MULTITIMEFRAME / f"timeframe={tf}"

            # フォルダ名からのHive推論を無効化し、直接Parquetファイルを指定する
            tf_files = str(tf_path / "*.parquet")
            tf_ddf = dd.read_parquet(tf_files, dataset={"partitioning": None})

            # 不要カラムの削除をここで行う
            cols_to_drop = [
                col for col in ProcessingConfig.COLUMNS_TO_DROP if col in tf_ddf.columns
            ]
            if cols_to_drop:
                tf_ddf = tf_ddf.drop(columns=cols_to_drop)
                logger.info(f"不要なカラムを削除しました: {cols_to_drop}")

            # tickの処理かその他の処理かを分岐
            if tf == "tick":
                global_volume_mean = float(tf_ddf["volume"].mean().compute())
                processed = tf_ddf.map_partitions(
                    calculate_enhanced_features,
                    global_volume_mean=global_volume_mean,
                    meta=meta_df,
                )
            else:
                # すでに単一の時間足だけになっているので、groupbyは不要！
                processed = tf_ddf.map_partitions(
                    calculate_enhanced_features,
                    global_volume_mean=0.0,
                    meta=meta_df,
                )

            # GOLDEN_SCHEMAに基づいて型を強制統一
            processed = processed.astype(GOLDEN_SCHEMA)

            # Hiveパーティションディレクトリを手動で構成し、個別に即時保存
            output_path = config.S1_PROCESSED / f"timeframe={tf}"
            logger.info(f"'{tf}' を保存中: {output_path}")
            processed.to_parquet(str(output_path), write_index=False, engine="pyarrow")

            # ★重要: 次のループへ行く前にメモリを確実に解放
            logger.info(f"'{tf}' の処理と保存が完了しました。メモリを開放します。")
            del tf_ddf, processed
            gc.collect()

        logger.info("✅ 全時間軸のデータの保存が完了しました。")

        # ループ外のDask DataFrame解放
        # del ddf
        gc.collect()

        # --- 5. 品質検証とサマリー出力 ---
        summary_counts = verify_data()

        total_duration = time.time() - start_time

        # 最終サマリーの出力
        logger.info("=" * 60)
        logger.info(
            f"✅ パイプラインが正常に完了しました。総処理時間: {total_duration / 60:.1f} 分"
        )
        logger.info("--- 【時間足別 行数サマリー】 ---")
        for tf, count in summary_counts.items():
            logger.info(f"  - {tf}: {count:,} 行")
        logger.info("=" * 60)

    except Exception as e:
        logger.error(f"処理中に致命的なエラーが発生しました: {e}")
        logger.error(traceback.format_exc())
        sys.exit(1)


if __name__ == "__main__":
    main()
