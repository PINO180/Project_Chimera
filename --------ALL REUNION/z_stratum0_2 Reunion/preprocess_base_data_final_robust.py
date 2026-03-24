# preprocess_base_data_final_robust.py
# price_directionのゼロ値問題を解決し、データ型を最終仕様に合わせた最終安定版

import os
import logging
from pathlib import Path
import shutil
import dask
import dask.dataframe as dd
import pandas as pd
import time
from typing import List, Dict
import gc
import traceback
import numpy as np
import pyarrow as pa

# ロギング設定
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
)
logger = logging.getLogger(__name__)


# --- 設定パラメータ ---
class ProcessingConfig:
    """前処理パラメータの設定クラス"""
    
    # ファイルパス
    PROJECT_ROOT = Path("/workspaces/project_forge") # パスを環境に合わせて調整
    INPUT_FILE = PROJECT_ROOT / "data/1_XAUUSD_base_data/XAUUSDmulti_15timeframe_bars_CORRECT_GPU.parquet"
    OUTPUT_FILE = PROJECT_ROOT / "data/1_XAUUSD_base_data/XAUUSD_tick_master_cleaned_and_enriched.parquet"
    
    # 計算パラメータ
    VOLATILITY_WINDOW = 20
    VOLUME_WINDOW = 50
    ATR_WINDOW = 14
    MOMENTUM_WINDOW = 5
    
    # 除外カラム (元データに存在する場合)
    COLUMNS_TO_DROP = [
        'volatility', 'avg_volume', 'tick_count', 'bid', 'ask', 'spread', 
        'price_change_pct', 'range', 'body_size'
    ]
    
    # 特別扱いする巨大な時間軸
    LARGE_TIMEFRAME = 'tick'


def calculate_enhanced_features(df: pd.DataFrame, global_volume_mean: float = 0.0) -> pd.DataFrame:
    """
    特徴量を計算する関数。price_directionのゼロ値問題を解決済み。
    """
    if len(df) == 0:
        return df
    
    # タイムスタンプでソートし、インデックスをリセット
    df = df.sort_values('timestamp').reset_index(drop=True)
    
    # 1. 対数リターン
    # ゼロ除算を避けるため、shiftしたcloseが0の場合は微小な値に置き換える
    close_shifted = df['close'].shift(1)
    close_shifted = close_shifted.replace(0, 1e-12)
    ratio = df['close'] / close_shifted
    df['log_return'] = np.log(ratio.fillna(1.0))
    
    # 2. ローリングボラティリティ
    df['rolling_volatility'] = df['log_return'].rolling(
        window=ProcessingConfig.VOLATILITY_WINDOW, min_periods=1
    ).std()
    
    # 3. ローリング平均出来高
    fallback_mean = global_volume_mean if global_volume_mean > 0 else df['volume'].mean()
    df['rolling_avg_volume'] = df['volume'].rolling(
        window=ProcessingConfig.VOLUME_WINDOW, min_periods=1
    ).mean().fillna(fallback_mean)
    
    # 4. ATR
    if len(df) > 1:
        close_prev = df['close'].shift(1)
        high_low = df['high'] - df['low']
        high_prev_close = (df['high'] - close_prev).abs()
        low_prev_close = (df['low'] - close_prev).abs()
        true_range = pd.concat([high_low, high_prev_close, low_prev_close], axis=1).max(axis=1, skipna=True)
        df['atr'] = true_range.ewm(span=ProcessingConfig.ATR_WINDOW, adjust=False).mean()
    else:
        df['atr'] = 0.0
    
    # 5. 価格方向性 (0を許容しないロバストな計算)
    price_diff = df['close'].diff()
    df['price_direction'] = np.sign(price_diff.fillna(0.0)).astype('int8')
    # price_directionが0になった場合、直前の向きを維持する（前方補完 ffill）
    # データ先頭で0だった場合は-1（下落）で暫定的に埋める
    df['price_direction'] = df['price_direction'].replace(0, pd.NA).ffill().fillna(-1).astype('int8')
    
    # 6. 価格モメンタム
    close_shifted_momentum = df['close'].shift(ProcessingConfig.MOMENTUM_WINDOW)
    close_shifted_momentum = close_shifted_momentum.replace(0, 1e-12)
    ratio_momentum = df['close'] / close_shifted_momentum
    df['price_momentum'] = ratio_momentum - 1
    
    # 7. 出来高比率
    safe_avg_vol = df['rolling_avg_volume'].replace(0, 1.0) # ゼロ除算を防止
    df['volume_ratio'] = df['volume'] / safe_avg_vol
    
    return df
        
def get_final_schema_meta(base_columns: List[str]) -> pd.DataFrame:
    """Daskのmetaとして使用する、処理後の完全なスキーマ（型情報）を持つ空のDataFrameを返す"""
    
    # 元データに含まれるべき基本カラムのスキーマ
    schema = {
        'timestamp': 'datetime64[ms]',
        'open': 'float32',
        'high': 'float32',
        'low': 'float32',
        'close': 'float32',
        'volume': 'int64',
    }

    # 新しく追加される特徴量のスキーマ
    schema.update({
        'log_return': 'float32',
        'rolling_volatility': 'float32',
        'rolling_avg_volume': 'float64',
        'atr': 'float32',
        'price_direction': 'int8',
        'price_momentum': 'float32',
        'volume_ratio': 'float32'
    })
    
    # timeframeカラムもスキーマに含める
    if 'timeframe' in base_columns:
        schema['timeframe'] = 'category'
        
    # 実際に存在するカラムのみで最終的なスキーマを構築
    final_columns = [col for col in base_columns if col in schema and col not in ProcessingConfig.COLUMNS_TO_DROP]
    final_columns.extend([
        'log_return', 'rolling_volatility', 'rolling_avg_volume', 'atr', 
        'price_direction', 'price_momentum', 'volume_ratio'
    ])
    
    final_schema = {col: schema[col] for col in final_columns if col in schema}
    
    return pd.DataFrame({k: pd.Series(dtype=v) for k, v in final_schema.items()})


def main() -> bool:
    config = ProcessingConfig()
    start_time = time.time()

    logger.info("=== CPU版・堅牢なデータ前処理パイプライン開始 ===")

    if not config.INPUT_FILE.exists():
        logger.error(f"入力ファイルが見つかりません: {config.INPUT_FILE}")
        return False
    if config.OUTPUT_FILE.exists():
        logger.warning(f"出力先パスが既に存在するため、一度削除します: {config.OUTPUT_FILE}")
        shutil.rmtree(config.OUTPUT_FILE, ignore_errors=True)
    config.OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

    try:
        ddf = dd.read_parquet(config.INPUT_FILE)

        # 不要なカラムを削除
        cols_to_drop = [col for col in config.COLUMNS_TO_DROP if col in ddf.columns]
        if cols_to_drop:
            ddf = ddf.drop(columns=cols_to_drop)
            logger.info(f"不要なカラムを削除しました: {cols_to_drop}")

        # ▼▼▼【修正点1】meta_dfをここで一度だけ定義する ▼▼▼
        # Daskのmeta（処理後のデータ構造定義）を準備
        meta_df = get_final_schema_meta(ddf.columns)
        
        all_timeframes = ddf['timeframe'].unique().compute().tolist()
        
        processed_ddfs = []

        # 全ての時間足をループ処理
        for tf in all_timeframes:
            logger.info(f"--- 時間軸 '{tf}' の特徴量計算タスクを作成します ---")
            tf_ddf = ddf[ddf['timeframe'] == tf]
            
            global_volume_mean = 0.0
            if tf == config.LARGE_TIMEFRAME:
                global_volume_mean = tf_ddf['volume'].mean().compute()
                logger.info(f"'{tf}' の全体平均出来高を計算しました: {global_volume_mean}")
            
            enriched_ddf = tf_ddf.map_partitions(
                calculate_enhanced_features, global_volume_mean=float(global_volume_mean), meta=meta_df
            )
            processed_ddfs.append(enriched_ddf)
        
        logger.info("--- 全てのタスクを結合し、最終的な保存処理を実行します ---")
        
        final_ddf = dd.concat(processed_ddfs, ignore_unknown_divisions=True)
        
        # Daskの処理結果のデータ型が、定義したmetaと完全に一致するように強制する
        # これにより、ライブラリ間の解釈のズレによるエラーを完全に防ぐ
        logger.info("最終的なデータ型を強制的に揃えています...")
        final_ddf = final_ddf.astype(meta_df.dtypes.to_dict())

        final_ddf.to_parquet(
            config.OUTPUT_FILE, 
            compression='snappy', 
            write_metadata_file=True, 
            partition_on=['timeframe']
            # schema引数は不要になる
        )
        
        total_duration = time.time() - start_time
        logger.info("="*60)
        logger.info(f"✅ 全ての処理が正常に完了しました。総処理時間: {total_duration / 60:.1f} 分")
        logger.info(f"特徴量が付与された新しいデータセットが以下に出力されました:\n{config.OUTPUT_FILE}")
        logger.info("="*60)
        
        return True
        
    except Exception as e:
        logger.error(f"処理中にエラーが発生しました: {e}")
        logger.error(traceback.format_exc())
        return False


if __name__ == "__main__":
    dask.config.set({
        'scheduler': 'single-threaded'
    })
    
    success = main()

    if success:
        print("\nデータ前処理が正常に完了しました。")
    else:
        print("\nデータ前処理でエラーが発生しました。詳細はログを確認してください。")