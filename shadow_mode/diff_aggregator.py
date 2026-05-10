"""
[Layer 1] DiffAggregator

ShadowEngine の捕捉値と ReferenceBuilder のリファレンスを join し、
数値差分を集計する。

比較ロジック:
  - inner join on (timestamp, timeframe, feature_name)
  - is_close = np.isclose(prod, ref, rtol=1e-7, atol=1e-12)
  - abs_diff = abs(prod - ref)
  - rel_diff = abs(prod - ref) / max(abs(ref), tiny)

集計指標:
  - 全体: total / passed / failed / fail_rate
  - 特徴量別: 特徴量名で grouping した fail_count
  - 時間足別: 時間足で grouping した fail_count
  - 失敗 worst-K: 最も差分が大きい K 件 (デフォルト 50)

戻り値の構造体:
  DiffStats: 集計結果 (テーブル/数値)
  failing_df: failing rows のみの DataFrame (parquet 出力用)
  paired_df: 全行の prod/ref/diff/is_close DataFrame (デバッグ用)
"""

import logging
from dataclasses import dataclass, field
from typing import Dict, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger("shadow_mode.diff_aggregator")


@dataclass
class DiffStats:
    """差分集計結果のコンテナ。"""

    total: int
    passed: int
    failed: int
    fail_rate: float
    rtol: float
    atol: float
    # 特徴量別 fail count: feature_name → count
    fail_by_feature: Dict[str, int] = field(default_factory=dict)
    # TF 別 fail count: timeframe → count
    fail_by_timeframe: Dict[str, int] = field(default_factory=dict)
    # ペア化されなかった行数 (片側のみ存在)
    unpaired_captured: int = 0
    unpaired_reference: int = 0

    def is_pass(self) -> bool:
        return self.failed == 0


class DiffAggregator:
    """captured vs reference の差分集計。

    Args:
        rtol: 相対許容誤差 (default 1e-7、Phase 9b 検証と同じ)
        atol: 絶対許容誤差 (default 1e-12)
        worst_k: 失敗 worst-K の K (default 50)
    """

    def __init__(
        self,
        rtol: float = 1e-7,
        atol: float = 1e-12,
        worst_k: int = 50,
    ):
        self.rtol = rtol
        self.atol = atol
        self.worst_k = worst_k

    def compare(
        self,
        captured_long: pd.DataFrame,
        reference_long: pd.DataFrame,
    ) -> Dict[str, object]:
        """captured と reference を比較し、結果を dict で返す。

        Args:
            captured_long: ShadowEngine.to_long_format() の出力
            reference_long: ReferenceBuilder.build() の出力
            (両者とも columns: timestamp, timeframe, feature_name, value)

        Returns:
            dict with keys:
                "stats":      DiffStats
                "paired":     全 inner-join 結果 (DataFrame)
                "failing":    failing 行のみ (DataFrame)
                "worst":      |abs_diff| top-K rows (DataFrame)
        """
        if captured_long.empty:
            logger.warning("captured_long is empty — nothing to compare")
            return self._empty_result()
        if reference_long.empty:
            logger.warning("reference_long is empty — nothing to compare")
            return self._empty_result()

        # --- normalize timestamps to UTC tz-aware ---
        cap = captured_long.copy()
        ref = reference_long.copy()
        cap["timestamp"] = self._normalize_ts(cap["timestamp"])
        ref["timestamp"] = self._normalize_ts(ref["timestamp"])

        # --- count unpaired (片側のみ存在) before join ---
        cap_keys = cap[["timestamp", "timeframe", "feature_name"]].drop_duplicates()
        ref_keys = ref[["timestamp", "timeframe", "feature_name"]].drop_duplicates()
        cap_only = cap_keys.merge(
            ref_keys, on=["timestamp", "timeframe", "feature_name"],
            how="left", indicator=True
        )
        n_cap_only = int((cap_only["_merge"] == "left_only").sum())
        ref_only = ref_keys.merge(
            cap_keys, on=["timestamp", "timeframe", "feature_name"],
            how="left", indicator=True
        )
        n_ref_only = int((ref_only["_merge"] == "left_only").sum())

        if n_cap_only or n_ref_only:
            logger.info(
                f"  Unpaired rows: captured-only={n_cap_only}, "
                f"reference-only={n_ref_only}"
            )

        # --- inner join ---
        paired = cap.merge(
            ref,
            on=["timestamp", "timeframe", "feature_name"],
            suffixes=("_prod", "_ref"),
            how="inner",
        )
        if paired.empty:
            logger.warning("No matching rows after inner join")
            stats = DiffStats(
                total=0, passed=0, failed=0, fail_rate=0.0,
                rtol=self.rtol, atol=self.atol,
                unpaired_captured=n_cap_only,
                unpaired_reference=n_ref_only,
            )
            return {"stats": stats, "paired": paired, "failing": paired,
                    "worst": paired}

        # --- compute diffs ---
        prod = paired["value_prod"].to_numpy(dtype=np.float64)
        refv = paired["value_ref"].to_numpy(dtype=np.float64)
        abs_diff = np.abs(prod - refv)
        # rel_diff: 0 division 防止
        denom = np.maximum(np.abs(refv), 1e-300)
        rel_diff = abs_diff / denom
        is_close = np.isclose(
            prod, refv, rtol=self.rtol, atol=self.atol, equal_nan=True
        )

        paired["abs_diff"] = abs_diff
        paired["rel_diff"] = rel_diff
        paired["is_close"] = is_close

        total = len(paired)
        passed = int(is_close.sum())
        failed = total - passed
        fail_rate = (failed / total) if total > 0 else 0.0

        # --- fail breakdowns ---
        failing = paired[~paired["is_close"]].copy()
        if not failing.empty:
            fail_by_feature = (
                failing.groupby("feature_name").size().sort_values(ascending=False)
            )
            fail_by_tf = (
                failing.groupby("timeframe").size().sort_values(ascending=False)
            )
        else:
            fail_by_feature = pd.Series(dtype=int)
            fail_by_tf = pd.Series(dtype=int)

        stats = DiffStats(
            total=total, passed=passed, failed=failed, fail_rate=fail_rate,
            rtol=self.rtol, atol=self.atol,
            fail_by_feature=dict(fail_by_feature.to_dict()),
            fail_by_timeframe=dict(fail_by_tf.to_dict()),
            unpaired_captured=n_cap_only,
            unpaired_reference=n_ref_only,
        )

        # --- worst-K (largest abs_diff) ---
        worst = paired.nlargest(self.worst_k, "abs_diff")[
            [
                "timestamp", "timeframe", "feature_name",
                "value_prod", "value_ref", "abs_diff", "rel_diff", "is_close",
            ]
        ].reset_index(drop=True)

        logger.info(
            f"  Compared {total} rows: passed={passed}, failed={failed} "
            f"({fail_rate * 100:.4f}%)"
        )
        if failed > 0:
            top_features = list(fail_by_feature.head(5).index)
            logger.info(f"  Top failing features: {top_features}")

        return {
            "stats": stats,
            "paired": paired,
            "failing": failing,
            "worst": worst,
        }

    @staticmethod
    def _normalize_ts(s: pd.Series) -> pd.Series:
        if not pd.api.types.is_datetime64_any_dtype(s):
            s = pd.to_datetime(s)
        if s.dt.tz is None:
            s = s.dt.tz_localize("UTC")
        else:
            s = s.dt.tz_convert("UTC")
        return s

    def _empty_result(self) -> Dict[str, object]:
        empty_cols = [
            "timestamp", "timeframe", "feature_name",
            "value_prod", "value_ref", "abs_diff", "rel_diff", "is_close",
        ]
        empty = pd.DataFrame(columns=empty_cols)
        stats = DiffStats(
            total=0, passed=0, failed=0, fail_rate=0.0,
            rtol=self.rtol, atol=self.atol,
        )
        return {
            "stats": stats,
            "paired": empty,
            "failing": empty,
            "worst": empty,
        }
