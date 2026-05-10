"""
[Layer 1] FeatureCapturer / ShadowEngine

RealtimeFeatureEngine のサブクラスで _calculate_base_features の戻り値
(= 純化前 / pre-OLS の base_features dict) を捕捉する。

設計上の重要点:
  - 本番コードは未改変。サブクラス override で hook するのみ。
  - 親クラスの実装を super() 経由で呼ぶため、本番挙動と byte-identical。
  - capture_enabled フラグで warmup 中の不要捕捉を抑制。
  - 捕捉対象は e1a_/e1b_/.../e1f_ 始まりの特徴量のみ
    (proxy 特徴量 log_return / price_momentum / rolling_volatility /
     volume_ratio は S2 に存在しないため捕捉不要)。
"""

import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

# Production code import (sys.path に /workspace を追加してから)
_WORKSPACE = Path(__file__).resolve().parents[1]
if str(_WORKSPACE) not in sys.path:
    sys.path.insert(0, str(_WORKSPACE))

from execution.realtime_feature_engine import RealtimeFeatureEngine  # noqa: E402

logger = logging.getLogger("shadow_mode.feature_capturer")


class ShadowEngine(RealtimeFeatureEngine):
    """RealtimeFeatureEngine subclass with feature capture hook.

    Override _calculate_base_features:
        1. Call parent implementation (no behavior change)
        2. If capture_enabled, append (timestamp, tf, features) to buffer

    捕捉値の semantic:
        - timestamp = self.last_bar_timestamps[tf_name]
          (= 当該 TF の最新確定バーの START タイムスタンプ、label="left")
        - 学習側 S2 の row.timestamp と同じ慣習なので join key として使える
    """

    # 捕捉する特徴量プレフィックス (engine_1_X 出力)
    _CAPTURE_PREFIXES = ("e1a_", "e1b_", "e1c_", "e1d_", "e1e_", "e1f_")

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._capture_enabled: bool = False
        self._captured_rows: List[Dict[str, Any]] = []
        self._n_captures: int = 0

    # ---------------------------------------------------------------- API

    def enable_capture(self) -> None:
        """test 期間に入る前に呼ぶ。warmup 中の捕捉を防ぐ。"""
        self._capture_enabled = True
        logger.info("ShadowEngine capture ENABLED")

    def disable_capture(self) -> None:
        self._capture_enabled = False
        logger.info("ShadowEngine capture DISABLED")

    def get_captured_rows(self) -> List[Dict[str, Any]]:
        """捕捉された全 (timestamp, tf, features...) dict のリストを返す。"""
        return self._captured_rows

    def captured_count(self) -> int:
        return self._n_captures

    def to_long_format(self) -> pd.DataFrame:
        """捕捉値を (timestamp, timeframe, feature_name, value) の long-format
        DataFrame に変換する。reference_builder の出力と同じ schema。
        """
        if not self._captured_rows:
            return pd.DataFrame(
                columns=["timestamp", "timeframe", "feature_name", "value"]
            )

        wide = pd.DataFrame(self._captured_rows)
        # melt — feature columns are everything except timestamp/timeframe
        feature_cols = [
            c for c in wide.columns if c not in ("timestamp", "timeframe")
        ]
        long = wide.melt(
            id_vars=["timestamp", "timeframe"],
            value_vars=feature_cols,
            var_name="feature_name",
            value_name="value",
        )
        # NaN はリファレンスにも NaN がある可能性があるので保持
        # ただし「特徴量未計算」(全 NaN) の行は捨てる
        long = long.dropna(subset=["value"])

        # (timestamp, timeframe, feature_name) の重複を除去
        # (M3 close 毎に M5/M8/M15 が再計算されるが、TF の deque が
        #  変わっていなければ値は同じ。最初の 1 件だけ残す)
        long = long.drop_duplicates(
            subset=["timestamp", "timeframe", "feature_name"], keep="first"
        ).reset_index(drop=True)

        return long

    # --------------------------------------------------------------- hook

    def _calculate_base_features(
        self, data: Dict[str, np.ndarray], tf_name: str
    ) -> Dict[str, float]:
        """親実装を呼んだ後、戻り値を捕捉する。挙動変化なし。"""
        base_features = super()._calculate_base_features(data, tf_name)

        if not self._capture_enabled:
            return base_features

        # 当該 TF の最新確定バー timestamp を取得
        ts = self.last_bar_timestamps.get(tf_name)
        if ts is None:
            # 通常は warmup で last_bar_timestamps が埋まる。None ならスキップ。
            return base_features

        # 捕捉行を構築。e1*_ プレフィックスの特徴量のみ。
        row: Dict[str, Any] = {"timestamp": ts, "timeframe": tf_name}
        captured_features = 0
        for key, value in base_features.items():
            if key.startswith(self._CAPTURE_PREFIXES):
                row[key] = float(value) if value is not None else np.nan
                captured_features += 1

        if captured_features > 0:
            self._captured_rows.append(row)
            self._n_captures += 1

        return base_features
