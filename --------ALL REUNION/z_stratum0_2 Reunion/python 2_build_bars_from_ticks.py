#!/usr/bin/env python3
"""
拡張マルチタイムフレーム バー生成スクリプト
ティックデータからtick/M0.5/M1/M3/M5/M8/M15/M30/H1/H4/H6/H12/D1/W1/MNのマルチタイムフレームバーを生成
Polarsを使用した最適化実装
"""

import polars as pl
import os
import time
from datetime import datetime, timedelta

def build_bars_from_ticks():
    """
    ティックParquetからマルチタイムフレームバーを生成
    """
    # パス設定
    input_file = r"C:\clonecloneclone\data\XAUUSDtick_exness.parquet"
    output_file = r"C:\clonecloneclone\data\XAUUSDmulti_timeframe_bars_exness.parquet"
    
    print(f"開始: 拡張マルチタイムフレームバー生成")
    print(f"入力: {input_file}")
    print(f"出力: {output_file}")
    
    start_time = time.time()
    
    try:
        # ティックデータの高速読み込み
        print("ティックデータ読み込み中...")
        tick_df = pl.read_parquet(input_file, use_pyarrow=True)
        
        print(f"ティックデータ読み込み完了: {tick_df.shape[0]:,} 行")
        print(f"日時範囲: {tick_df.select(pl.col('datetime').min())} - {tick_df.select(pl.col('datetime').max())}")
        
        # 拡張タイムフレーム定義
        timeframes = {
            'tick': None,     # 生ティックデータ
            'M0.5': '30s',    # 30秒
            'M1': '1m',       # 1分
            'M3': '3m',       # 3分
            'M5': '5m',       # 5分
            'M8': '8m',       # 8分
            'M15': '15m',     # 15分
            'M30': '30m',     # 30分
            'H1': '1h',       # 1時間
            'H4': '4h',       # 4時間
            'H6': '6h',       # 6時間
            'H12': '12h',     # 12時間
            'D1': '1d',       # 日足
            'W1': '1w',       # 週足
            'MN': '1mo'       # 月足
        }
        
        # 全タイムフレームのデータを格納するリスト
        all_bars = []
        
        # 全ティックデータを追加
        print("全ティックデータ処理中...")
        tick_sample = tick_df.with_columns([
            pl.lit("tick").alias("timeframe"),
            pl.col("datetime").alias("timestamp"),
            pl.col("mid_price").alias("open"),
            pl.col("mid_price").alias("high"),
            pl.col("mid_price").alias("low"), 
            pl.col("mid_price").alias("close"),
            pl.when(pl.col("volume").is_not_null())
            .then(pl.col("volume").cast(pl.Int64))
            .otherwise(pl.lit(0, dtype=pl.Int64))
            .alias("volume"),
            pl.lit(1).cast(pl.Int64).alias("tick_count"),
            # 統一されたスキーマのために追加列を作成
            pl.lit(None, dtype=pl.Float32).alias("volatility"),
            pl.lit(None, dtype=pl.Float64).alias("avg_volume")
        ]).select([
            "timeframe", "timestamp", "open", "high", "low", "close", 
            "volume", "tick_count", "bid", "ask", "spread", "volatility", "avg_volume"
        ])
        
        all_bars.append(tick_sample)
        print(f"ティックデータ追加: {tick_sample.shape[0]:,} 行")
        
        # 各タイムフレームでバーを生成
        for tf_name, tf_period in timeframes.items():
            if tf_period is None:  # ティックはスキップ
                continue
                
            print(f"\n{tf_name}バー生成中...")
            
            # Polarsの高速グループ化集計
            bars = tick_df.sort("datetime").group_by_dynamic(
                "datetime",
                every=tf_period,
                period=tf_period,
                closed="left",
                label="left"
            ).agg([
                # OHLC計算
                pl.col("mid_price").first().alias("open"),
                pl.col("mid_price").max().alias("high"),
                pl.col("mid_price").min().alias("low"),
                pl.col("mid_price").last().alias("close"),
                
                # ボリューム・ティック数
                pl.col("volume").sum().alias("volume"),
                pl.col("volume").len().alias("tick_count"),
                
                # BID/ASK統計
                pl.col("bid").mean().alias("bid"),
                pl.col("ask").mean().alias("ask"),
                pl.col("spread").mean().alias("spread"),
                
                # 追加統計
                pl.col("mid_price").std().alias("volatility"),
                pl.col("volume").mean().alias("avg_volume")
            ]).filter(
                pl.col("tick_count") > 0  # 空のバーを除外
            ).with_columns([
                pl.lit(tf_name).alias("timeframe"),
                pl.col("datetime").alias("timestamp")
            ]).select([
                "timeframe", "timestamp", "open", "high", "low", "close",
                "volume", "tick_count", "bid", "ask", "spread", "volatility", "avg_volume"
            ])
            
            print(f"{tf_name}バー生成完了: {bars.shape[0]:,} 行")
            all_bars.append(bars)
        
        # 全タイムフレームを結合
        print(f"\n全タイムフレーム結合中...")
        final_df = pl.concat(all_bars, how="vertical_relaxed")
        
        # タイムフレーム順でソート
        timeframe_order = ["tick", "M0.5", "M1", "M3", "M5", "M8", "M15", "M30", "H1", "H4", "H6", "H12", "D1", "W1", "MN"]
        final_df = final_df.with_columns([
            pl.col("timeframe").map_elements(
                lambda x: timeframe_order.index(x) if x in timeframe_order else 999,
                return_dtype=pl.Int32
            ).alias("tf_order")
        ]).sort(["tf_order", "timestamp"]).drop("tf_order")
        
        # データ型最適化
        final_df = final_df.with_columns([
            pl.col("timestamp").cast(pl.Datetime("ms")),
            pl.col("open").cast(pl.Float32),
            pl.col("high").cast(pl.Float32),
            pl.col("low").cast(pl.Float32),
            pl.col("close").cast(pl.Float32),
            pl.col("volume").cast(pl.Int64),
            pl.col("tick_count").cast(pl.Int32),
            pl.col("bid").cast(pl.Float32),
            pl.col("ask").cast(pl.Float32),
            pl.col("spread").cast(pl.Float32)
        ])
        
        # 追加指標計算
        final_df = final_df.with_columns([
            # 価格変化率
            ((pl.col("close") - pl.col("open")) / pl.col("open") * 100).alias("price_change_pct"),
            
            # レンジ（高値-安値）
            (pl.col("high") - pl.col("low")).alias("range"),
            
            # 実体の大きさ
            (pl.col("close") - pl.col("open")).abs().alias("body_size")
        ])
        
        print(f"\n最終データ形状: {final_df.shape}")
        
        # タイムフレーム別統計
        print(f"\nタイムフレーム別統計:")
        tf_stats = final_df.group_by("timeframe").agg([
            pl.len().alias("count"),
            pl.col("timestamp").min().alias("start_time"),
            pl.col("timestamp").max().alias("end_time")
        ]).sort("timeframe")
        print(tf_stats)
        
        # 出力ディレクトリ作成
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        
        # 高圧縮Parquet出力
        final_df.write_parquet(
            output_file,
            compression="zstd",
            use_pyarrow=True,
            statistics=True,
            row_group_size=100000
        )
        
        # 処理完了情報
        end_time = time.time()
        processing_time = end_time - start_time
        file_size = os.path.getsize(output_file) / (1024**2)  # MB
        
        print(f"\n=== 処理完了 ===")
        print(f"処理時間: {processing_time/60:.1f}分 ({processing_time:.2f}秒)")
        print(f"出力ファイル: {output_file}")
        print(f"ファイルサイズ: {file_size:.2f} MB")
        print(f"総行数: {final_df.shape[0]:,}")
        print(f"タイムフレーム数: {len(timeframes)}")
        
        # データサンプル表示
        print(f"\n各タイムフレームのサンプル:")
        for tf in timeframe_order[:8]:  # 最初の8個のみ表示
            sample = final_df.filter(pl.col("timeframe") == tf).head(2)
            if sample.shape[0] > 0:
                print(f"\n{tf}:")
                print(sample.select([
                    "timestamp", "open", "high", "low", "close", "volume", "tick_count"
                ]))
        
        # 長期タイムフレームのサンプル
        print(f"\n長期タイムフレームのサンプル:")
        for tf in ["D1", "W1", "MN"]:
            sample = final_df.filter(pl.col("timeframe") == tf).head(2)
            if sample.shape[0] > 0:
                print(f"\n{tf}:")
                print(sample.select([
                    "timestamp", "open", "high", "low", "close", "tick_count"
                ]))
        
    except Exception as e:
        print(f"エラーが発生しました: {e}")
        import traceback
        traceback.print_exc()
        raise

if __name__ == "__main__":
    build_bars_from_ticks()