"""
[Layer 1] StressInjector

合成ストレスシナリオの注入 (v2 拡張用 stub)。

v1 では「continuous」シナリオ (= 自然な test 期間データを修正なしで再生)
のみを実装。週末跨ぎや市場閉鎖明け等の natural gap は既にデータに含まれて
いるため、自然データだけで多様な状況をカバーできる。

v2 で追加予定:
  - synthetic_gap: 任意の M0.5 バーを drop して 60s/200s/360s ギャップを作成
  - ea_restart: warmup 後に engine state を一部 clear して再開動作を検証
  - news_spike: $0.5+ の急変を 30s で挿入し、ATR/disc 検知を検証

注: 合成シナリオは reference (S2) との直接比較が崩れるため、
    シナリオごとに reference を再生成するか、特定の特徴量だけを比較対象に
    限定する設計が必要。v2 で詳細を詰める。
"""

import logging
from typing import Iterator, Dict, Any, List

logger = logging.getLogger("shadow_mode.stress_injector")


class StressInjector:
    """合成ストレスシナリオの注入 (v1 stub)。

    Args:
        scenario: シナリオ名 (v1 では "continuous" のみ有効)
    """

    SUPPORTED_V1 = ("continuous",)
    SUPPORTED_V2 = ("synthetic_gap_60s", "synthetic_gap_200s",
                    "synthetic_gap_360s", "ea_restart", "news_spike")

    def __init__(self, scenario: str = "continuous"):
        if scenario not in self.SUPPORTED_V1:
            if scenario in self.SUPPORTED_V2:
                raise NotImplementedError(
                    f"Scenario '{scenario}' is planned for v2 but not yet "
                    f"implemented. v1 supports: {self.SUPPORTED_V1}"
                )
            raise ValueError(
                f"Unknown scenario '{scenario}'. Supported v1: "
                f"{self.SUPPORTED_V1}"
            )
        self.scenario = scenario
        logger.info(f"StressInjector initialized: scenario={scenario}")

    def transform(
        self, bars: Iterator[Dict[str, Any]]
    ) -> Iterator[Dict[str, Any]]:
        """input bar generator を変換した generator を返す。

        v1 ('continuous'): 入力をそのまま yield (passthrough)。
        """
        if self.scenario == "continuous":
            yield from bars
        else:
            # 防御的: 上の __init__ で弾いているはずだが念のため
            raise NotImplementedError(
                f"Transform not implemented for scenario={self.scenario}"
            )
