"""
numpy_helpers.py — Project Forge / Cimera 共通 numpy ヘルパー (SSoT)

[Phase E 後継 refactor / Plan §B.12.14.10 + §B.12.15.4 + §B.12.16.x]

Phase E (B-F) 完了時点で、`_pct_change` が rfe_1A〜1F (engine_1_C を除く 5 file)
に local 重複定義されていた。実装は B 側 vectorized (rfe_1A/1B) と F 側
Numba JIT loop (rfe_1D/1E/1F) の 2 流派に分かれていた。

本モジュールは、これらヘルパーを 1 箇所に集約し、将来の divergence を
構造的に防ぐための SSoT として位置付ける。

設計原則:
  1. **canonical 実装の一本化**: 同一機能の複数実装を許さない。
     `pct_change_polars_compat` が `_pct_change` の唯一の真実源。
  2. **Polars semantics 完全一致**: Polars `.pct_change()` の挙動を
     bit-identical に再現する (prev=0 ケース含む)。
  3. **`@nb.njit` 採用**: 旧 1D/1E/1F の F 流派 Numba JIT loop を canonical に。
     - Plan §B.12.14.10 では「numpy SIMD で十分高速、JIT 化不採用」としていたが、
       §B.12.16.x で実機 shadow_mode が 0.07% fail を示し、純 numpy 化が
       Numba module-level state を変化させて他の JIT 関数の数値結果に影響する
       可能性が判明 (症状と整合)。
     - 旧 1D/1E/1F は `@nb.njit(fastmath=False, cache=True)` 付きで、それを
       純 Python 化したことで Numba 環境の dispatcher 構成が変化。
     - canonical を `@nb.njit` 化することで、Phase E PASS 時と同じ Numba
       環境 (`@nb.njit` 関数数・dispatcher 構成) を再現する。
  4. **F 流派 loop 採用** (B 流派 vectorized ではなく):
     - 旧 1D/1E/1F が F 流派だったので、それと **完全同型** にすれば
       Phase E PASS 状態と Numba JIT 結果が一致するはず。
     - B 流派 vectorized + numpy errstate と F 流派 loop は数値結果が
       bit-identical (worker §B.12.14.10 検証で実証済)。よってどちらでも
       数値は同じ、ただし Numba 環境を維持する観点で F 流派を採用。
  5. **shadow_mode cache_key に組み込み**: 本ファイルの sha は
     `run_shadow_test.py` の cache_key に含めること。

採用 rfe: 1A, 1B, 1D, 1E, 1F (1C は `_pct_change` を持たない)

参照: Skew_Detection_Hardening_Plan.md §B.12.14.10, §B.12.15.4, §B.12.16
"""

from __future__ import annotations

import numpy as np
import numba as nb


@nb.njit(fastmath=False, cache=True)
def pct_change_polars_compat(arr: np.ndarray) -> np.ndarray:
    """Polars `.pct_change()` と bit-identical な pct_change 計算。

    semantics (Polars `.pct_change()` と完全一致):
      - 先頭 [0] は NaN (前の値が無い)
      - prev != 0:        (x[i] - x[i-1]) / x[i-1]
      - prev == 0, x[i] > 0:  +inf
      - prev == 0, x[i] < 0:  -inf
      - prev == 0, x[i] == 0: NaN  (0 / 0)

    実装:
      旧 rfe_1D/1E/1F と完全同型の Numba JIT loop。
      `@nb.njit(fastmath=False, cache=True)` で旧実装と同じ環境を維持。

    数値一致検証:
      vectorized 流派 (rfe_1A/1B 旧実装) と全ケースで bit-identical
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
    for i in range(1, n):
        prev = arr[i - 1]
        if prev != 0.0:
            out[i] = (arr[i] - prev) / prev
        else:
            cur = arr[i]
            if cur > 0.0:
                out[i] = np.inf
            elif cur < 0.0:
                out[i] = -np.inf
            else:
                out[i] = np.nan  # 0 / 0
    return out
