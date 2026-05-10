"""
[Layer 1] ReplayBridge

歴史的 M0.5 parquet (s1_1_B_build_ohlcv.py の出力) を読み込み、
warmup + test 期間に分割して順次供給する。

Production の MQL5BridgePublisherV3 とは異なり、ZMQ や EA 通信は介さず、
直接 parquet を読む。Shadow Mode の runner はこのオブジェクトから
warmup_history と test_bar generator を取得する。
"""

import logging
from pathlib import Path
from typing import Dict, Iterator, Optional, Any

import pandas as pd

logger = logging.getLogger("shadow_mode.replay_bridge")


class ReplayBridge:
    """歴史 M0.5 データの replay 供給。

    Attributes:
        m05_parquet_path: master_multitimeframe / timeframe=M0.5 のパス
                         (ディレクトリでも単一ファイルでも可)
        warmup_end_ts: warmup と test の境界 (UTC tz-aware)
        test_end_ts:   test 期間の終了時刻 (UTC tz-aware, inclusive)
    """

    def __init__(
        self,
        m05_parquet_path: Path,
        warmup_end_ts: pd.Timestamp,
        test_end_ts: pd.Timestamp,
    ):
        self.m05_parquet_path = Path(m05_parquet_path)
        self.warmup_end_ts = self._ensure_utc(warmup_end_ts)
        self.test_end_ts = self._ensure_utc(test_end_ts)
        if self.warmup_end_ts >= self.test_end_ts:
            raise ValueError(
                f"warmup_end_ts ({self.warmup_end_ts}) must precede "
                f"test_end_ts ({self.test_end_ts})"
            )
        self._df = self._load_m05_parquet()
        self._warmup_df = self._df[
            self._df["timestamp"] <= self.warmup_end_ts
        ].reset_index(drop=True)
        self._test_df = self._df[
            (self._df["timestamp"] > self.warmup_end_ts)
            & (self._df["timestamp"] <= self.test_end_ts)
        ].reset_index(drop=True)
        logger.info(
            f"ReplayBridge initialized: warmup={len(self._warmup_df)} bars, "
            f"test={len(self._test_df)} bars"
        )

    @staticmethod
    def _ensure_utc(ts: pd.Timestamp) -> pd.Timestamp:
        ts = pd.Timestamp(ts)
        if ts.tz is None:
            return ts.tz_localize("UTC")
        return ts.tz_convert("UTC")

    def _load_m05_parquet(self) -> pd.DataFrame:
        """Master multitimeframe (timeframe=M0.5) parquet を全件ロード。

        対応形式:
          - 単一ファイル: <path>.parquet
          - Hive ディレクトリ: <path>/timeframe=M0.5/*.parquet
          - その他のディレクトリ: <path>/*.parquet
        """
        if not self.m05_parquet_path.exists():
            raise FileNotFoundError(
                f"M0.5 parquet path not found: {self.m05_parquet_path}"
            )

        if self.m05_parquet_path.is_file():
            df = pd.read_parquet(self.m05_parquet_path)
        else:
            # ディレクトリ: 配下の全 parquet を結合
            files = sorted(self.m05_parquet_path.rglob("*.parquet"))
            if not files:
                raise FileNotFoundError(
                    f"No parquet files under {self.m05_parquet_path}"
                )
            logger.info(f"  Loading {len(files)} parquet file(s)")
            df = pd.concat(
                [pd.read_parquet(f) for f in files], ignore_index=True
            )

        # Schema 検証
        required = {"timestamp", "open", "high", "low", "close", "volume"}
        missing = required - set(df.columns)
        if missing:
            raise ValueError(f"M0.5 parquet missing columns: {missing}")

        # timestamp を UTC tz-aware に正規化
        if not pd.api.types.is_datetime64_any_dtype(df["timestamp"]):
            df["timestamp"] = pd.to_datetime(df["timestamp"])
        if df["timestamp"].dt.tz is None:
            df["timestamp"] = df["timestamp"].dt.tz_localize("UTC")
        else:
            df["timestamp"] = df["timestamp"].dt.tz_convert("UTC")

        # disc 列があれば破棄 (production が _compute_disc_flag で再計算するため)
        if "disc" in df.columns:
            logger.info("  Dropping disc column (production recomputes it)")
            df = df.drop(columns=["disc"])

        # ソート + 必要列のみ保持
        keep_cols = ["timestamp", "open", "high", "low", "close", "volume"]
        df = df[keep_cols].sort_values("timestamp").reset_index(drop=True)

        logger.info(
            f"  Loaded M0.5 parquet: {len(df)} bars, "
            f"range {df['timestamp'].iloc[0]} - {df['timestamp'].iloc[-1]}"
        )
        return df

    def get_warmup_history(self) -> pd.DataFrame:
        """fill_all_buffers に渡す warmup 履歴 DataFrame を返す。

        Returns:
            DataFrame with columns: timestamp, open, high, low, close, volume
        """
        return self._warmup_df.copy()

    def iter_test_bars(self) -> Iterator[Dict[str, Any]]:
        """test 期間の M0.5 バーを 1 本ずつ yield する。

        本番 EA の payload 形式に合わせた dict を返す:
            {
                "time": int (Unix timestamp),
                "timestamp": pd.Timestamp (UTC),
                "open": float, "high": float, "low": float, "close": float,
                "volume": int,
                "spread": float (合成値、shadow mode では固定 36.0),
            }
        """
        for _, row in self._test_df.iterrows():
            ts = row["timestamp"]
            yield {
                "time": int(ts.timestamp()),
                "timestamp": ts,
                "open": float(row["open"]),
                "high": float(row["high"]),
                "low": float(row["low"]),
                "close": float(row["close"]),
                "volume": int(row["volume"]),
                "spread": 36.0,
            }

    def total_test_bars(self) -> int:
        return len(self._test_df)

    def total_warmup_bars(self) -> int:
        return len(self._warmup_df)

    @property
    def test_period_summary(self) -> Dict[str, Any]:
        if len(self._test_df) == 0:
            return {"start": None, "end": None, "n_bars": 0}
        return {
            "start": self._test_df["timestamp"].iloc[0],
            "end": self._test_df["timestamp"].iloc[-1],
            "n_bars": len(self._test_df),
        }
