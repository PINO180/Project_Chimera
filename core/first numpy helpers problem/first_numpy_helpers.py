"""
numpy_helpers.py — Project Forge / Cimera 共通 numpy ヘルパー (SSoT)

[Phase E 後継 refactor / Plan §B.12.14.10]

Phase E (B-F) 完了時点で、`_pct_change` が rfe_1A〜1F (engine_1_C を除く 5 file)
に local 重複定義されていた。3 つの実装流派が分散し、過去には rfe_1E に
else 句欠落 bug が存在していた (Plan §B.12.14.7 警告 6、現在は解消済)。

本モジュールは、これら数値処理ヘルパーを 1 箇所に集約し、将来の divergence を
構造的に防ぐための SSoT (Single Source of Truth) として位置付ける。

設計原則:
  1. **canonical 実装の一本化**: 同一機能の複数実装を許さない。
     `pct_change_polars_compat` が `_pct_change` の唯一の真実源。
  2. **Polars semantics 完全一致**: Polars `.pct_change()` の挙動を
     bit-identical に再現する (prev=0 ケース含む)。
  3. **純 numpy vectorized**: Numba JIT 化は不採用 (Plan §B.12.14.10)。
     - numpy SIMD で本番運用上十分に高速 (3500 行で ~9.5μs/call)
     - JIT cache 不要、cache_key 設計が単純化
     - 純 Python 関数なので test/debug が容易
  4. **shadow_mode cache_key に組み込み**: 本ファイルの sha は
     `run_shadow_test.py` の cache_key に含めること。本ファイルを編集すると
     Layer 1 検証が自動で再走行される。

採用 rfe: 1A, 1B, 1D, 1E, 1F (1C は `_pct_change` を持たない)

参照: Skew_Detection_Hardening_Plan.md §B.12.14.10
"""

from __future__ import annotations

import numpy as np


def pct_change_polars_compat(arr: np.ndarray) -> np.ndarray:
    """Polars `.pct_change()` と bit-identical な pct_change 計算。

    semantics (Polars `.pct_change()` と完全一致):
      - 先頭 [0] は NaN (前の値が無い)
      - prev != 0:        (x[i] - x[i-1]) / x[i-1]
      - prev == 0, x[i] > 0:  +inf
      - prev == 0, x[i] < 0:  -inf
      - prev == 0, x[i] == 0: NaN  (0 / 0)

    実装:
      numpy SIMD vectorized + `errstate(divide='ignore', invalid='ignore')`
      で prev=0 ケースの除算警告を抑制しつつ、IEEE 754 の自然な挙動
      (±inf / NaN) をそのまま採用する。

    数値一致検証:
      Numba JIT loop 版 (rfe_1D/1E/1F の旧実装) と全ケースで bit-identical
      (Plan §B.12.14.10 worker 検証で実証済)。

    Args:
      arr: 入力配列 (1D numpy array、典型的には close[-N:] の deque snapshot)

    Returns:
      pct_change 配列 (同長、dtype=float64、先頭 NaN)
    """
    n = len(arr)
    out = np.full(n, np.nan, dtype=np.float64)
    if n < 2:
        return out
    with np.errstate(divide="ignore", invalid="ignore"):
        out[1:] = (arr[1:] - arr[:-1]) / arr[:-1]
    return out
