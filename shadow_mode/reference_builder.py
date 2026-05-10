"""
[Layer 1] ReferenceBuilder

S2_FEATURES_VALIDATED から学習側 pre-OLS 特徴量を読み出し、
captured データと同じ schema (timestamp, timeframe, feature_name, value) の
long-format DataFrame に変換する。

S2 のディレクトリ構造 (確認済):
  S2_FEATURES_VALIDATED/
    feature_value_a_vast_universeA/
      features_e1a_M0.5.parquet
      features_e1a_M1.parquet
      ...
    feature_value_a_vast_universeB/
      features_e1b_M0.5.parquet
      ...
    ... (C, D, E, F)

各 parquet の columns:
  timestamp, e1<x>_<feature_name>_<TF>, ...

例: features_e1a_M0.5.parquet:
  timestamp, e1a_fast_basic_stabilization_M0.5, e1a_fast_rolling_mean_5_M0.5, ...

長形式変換時:
  feature_name = strip(_<TF> suffix)  # e.g., e1a_fast_basic_stabilization
  → captured 側の base_features dict key と一致
"""

import logging
import re
from pathlib import Path
from typing import Iterable, List, Optional

import pandas as pd

logger = logging.getLogger("shadow_mode.reference_builder")


class ReferenceBuilder:
    """S2_FEATURES_VALIDATED 読み込み + long-format 変換。

    Args:
        s2_root: S2_FEATURES_VALIDATED のルートディレクトリ
        timeframes: 比較対象の TF (例: ["M0.5", "M1", "M3", "M5", "M8", "M15"])
        test_start_ts: 比較期間の開始 (UTC tz-aware, inclusive)
        test_end_ts:   比較期間の終了 (UTC tz-aware, inclusive)
    """

    # 学習側 engine 命名規則: features_e1<x>_<TF>.parquet
    _FILE_PATTERN = re.compile(
        r"^features_(e1[a-f])_(M0\.5|M\d+|H\d+|D\d+)\.parquet$"
    )

    def __init__(
        self,
        s2_root: Path,
        timeframes: Iterable[str],
        test_start_ts: pd.Timestamp,
        test_end_ts: pd.Timestamp,
    ):
        self.s2_root = Path(s2_root)
        self.timeframes = list(timeframes)
        self.test_start_ts = self._ensure_utc(test_start_ts)
        self.test_end_ts = self._ensure_utc(test_end_ts)

        if not self.s2_root.exists() or not self.s2_root.is_dir():
            raise FileNotFoundError(
                f"S2 root directory not found: {self.s2_root}"
            )

    @staticmethod
    def _ensure_utc(ts: pd.Timestamp) -> pd.Timestamp:
        ts = pd.Timestamp(ts)
        if ts.tz is None:
            return ts.tz_localize("UTC")
        return ts.tz_convert("UTC")

    def _discover_files(self) -> List[Path]:
        """S2 配下から features_e1*_<TF>.parquet ファイルを列挙。"""
        files = sorted(self.s2_root.rglob("features_e1*.parquet"))
        if not files:
            raise FileNotFoundError(
                f"No feature parquet files under {self.s2_root}"
            )
        logger.info(f"  Discovered {len(files)} feature parquet files")
        return files

    @classmethod
    def _parse_filename(cls, path: Path) -> Optional[tuple]:
        """ファイル名から (engine_id, timeframe) を抽出。

        Returns:
            (engine_id, timeframe) or None if not matched
        """
        m = cls._FILE_PATTERN.match(path.name)
        if not m:
            return None
        return m.group(1), m.group(2)

    def _load_one_file(self, path: Path) -> Optional[pd.DataFrame]:
        """1 ファイルを読み込み、long-format に変換して返す。

        Returns:
            DataFrame with columns: timestamp, timeframe, feature_name, value
            None if file is empty or not relevant
        """
        parsed = self._parse_filename(path)
        if parsed is None:
            logger.warning(f"  Skipping (filename pattern mismatch): {path.name}")
            return None
        engine_id, tf = parsed

        if tf not in self.timeframes:
            return None

        try:
            df = pd.read_parquet(path)
        except Exception as e:
            logger.error(f"  Failed to read {path}: {e}")
            return None

        if df.empty:
            return None

        # timestamp 正規化
        if "timestamp" not in df.columns:
            logger.warning(f"  Skipping (no timestamp column): {path.name}")
            return None
        if not pd.api.types.is_datetime64_any_dtype(df["timestamp"]):
            df["timestamp"] = pd.to_datetime(df["timestamp"])
        if df["timestamp"].dt.tz is None:
            df["timestamp"] = df["timestamp"].dt.tz_localize("UTC")
        else:
            df["timestamp"] = df["timestamp"].dt.tz_convert("UTC")

        # 期間フィルタ
        df = df[
            (df["timestamp"] >= self.test_start_ts)
            & (df["timestamp"] <= self.test_end_ts)
        ]
        if df.empty:
            return None

        # feature columns: timestamp + meta 以外、かつ engine prefix で始まる
        feature_cols = [
            c
            for c in df.columns
            if c.startswith(engine_id + "_") and c != "timestamp"
        ]
        if not feature_cols:
            logger.warning(
                f"  No feature columns found in {path.name} "
                f"(engine={engine_id})"
            )
            return None

        # melt to long
        long = df.melt(
            id_vars=["timestamp"],
            value_vars=feature_cols,
            var_name="feature_name_full",
            value_name="value",
        )

        # feature_name_full は "e1a_fast_xxx_M0.5" 形式。
        # capture 側の key は "e1a_fast_xxx" (TF suffix なし) なので strip する。
        tf_suffix = f"_{tf}"

        def _strip_tf_suffix(name: str) -> str:
            if name.endswith(tf_suffix):
                return name[: -len(tf_suffix)]
            return name

        long["feature_name"] = long["feature_name_full"].map(_strip_tf_suffix)
        long["timeframe"] = tf
        long = long[["timestamp", "timeframe", "feature_name", "value"]]
        long = long.dropna(subset=["value"])

        return long

    def build(self) -> pd.DataFrame:
        """全 S2 parquet を読み込んで結合した long-format reference を返す。

        Returns:
            DataFrame with columns: timestamp, timeframe, feature_name, value
        """
        logger.info(f"Building reference from {self.s2_root}")
        logger.info(
            f"  Period: {self.test_start_ts} → {self.test_end_ts}, "
            f"timeframes: {self.timeframes}"
        )

        files = self._discover_files()
        chunks: List[pd.DataFrame] = []
        for path in files:
            chunk = self._load_one_file(path)
            if chunk is not None and len(chunk) > 0:
                chunks.append(chunk)
                logger.debug(
                    f"  Loaded {path.name}: {len(chunk)} rows"
                )

        if not chunks:
            logger.warning("No reference data loaded — empty DataFrame returned")
            return pd.DataFrame(
                columns=["timestamp", "timeframe", "feature_name", "value"]
            )

        ref = pd.concat(chunks, ignore_index=True)
        # (timestamp, timeframe, feature_name) で重複除去 (異なる engine ファイルで
        # 同じ feature_name が存在する想定は通常ないが、安全側で keep="first")
        ref = ref.drop_duplicates(
            subset=["timestamp", "timeframe", "feature_name"], keep="first"
        ).reset_index(drop=True)

        logger.info(
            f"  Reference built: {len(ref)} rows across "
            f"{ref['timeframe'].nunique()} timeframes, "
            f"{ref['feature_name'].nunique()} features"
        )
        return ref
