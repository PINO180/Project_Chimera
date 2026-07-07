"""compare_common.py — compare 系 3 本の共通ロジック (§11.34.16 改修)

今回の改修の核心は 1 つ:
  本番シグナル時刻 T = M3 バーのクローズ時刻 (rfe L1810: ラベル + 3分)。
  学習側 (S6 / snapshot / infer) の行は **ラベル L = T - 180s** のグリッド。
  従来の compare 系は prod(T) ↔ training(T) をシフトなしで突合しており、
  常に 1 本 (180s) ズレた比較をしていた (§11.34.16 B 節)。
  → 突合は prod(T) ↔ training(T - SHIFT_SEC) で行う。

加えて 3way 用に:
  - HF-NB-GATE 対応: tf >= 5 の高 TF 列は、本番ゲートが境界以外で 0 化する。
    学習側も非境界行は 0。よって「両側 0」 の行と「両側実値」 の行だけが
    正当な比較対象で、「片側 0 / 片側実値」 の行は規約のズレを比較しているだけ。
    → 高 TF 列は per-row で「両側とも非ゼロ」 の行に限って metric を計算する
       (low TF: M0.5/M1/M3 は全行対象)。
  - TF 別レポート: 列名サフィックス _M{n} で TF を判定し、TF ごとに集計。
  - 悉皆列突合: 全列の diff を出し、残差が残る列を昇順でリストアップ
    (§11.34.16 N.8 の最終残差 0.01 級の犯人特定用)。
"""

from __future__ import annotations

import re
from datetime import datetime, timezone
from typing import Optional

import numpy as np
import pandas as pd

# 本番シグナル T とラベル L の差 (M3 = 180s)。TARGET_TIMEFRAMES 変更時は要追従。
SHIFT_SEC = 180

# このTF未満は全行が本番シグナル時刻と整合する → ゲート非対象 (常に比較)
GATE_LOW_TF_MAX_MIN = 3  # M0.5/M1/M3

_TF_SUFFIX_RE = re.compile(r"_M([0-9]+(?:\.[0-9]+)?)$")


def parse_dt(s: str) -> datetime:
    """'YYYY-MM-DD HH:MM:SS' (UTC) を tz-aware datetime に。"""
    return datetime.strptime(s, "%Y-%m-%d %H:%M:%S").replace(tzinfo=timezone.utc)


def tf_minutes_of(col: str) -> Optional[float]:
    """列名サフィックス _M{n} から TF 分を取り出す。無ければ None。"""
    m = _TF_SUFFIX_RE.search(col)
    if not m:
        return None
    try:
        return float(m.group(1))
    except (TypeError, ValueError):
        return None


def is_high_tf_col(col: str) -> bool:
    """tf >= 5 分の高 TF 列か (= HF-NB-GATE 対象か)。"""
    tfm = tf_minutes_of(col)
    return tfm is not None and tfm >= 5


def shift_training_to_signal(
    df_train: pd.DataFrame, shift_sec: int = SHIFT_SEC
) -> pd.DataFrame:
    """学習側 timestamp を +shift_sec して本番シグナル時刻 T 基準に揃える。

    training は L グリッド (= T - shift_sec)。L に shift を足すと T になり、
    prod(T) と inner merge できるようになる。
    """
    out = df_train.copy()
    out["timestamp"] = pd.to_datetime(out["timestamp"], utc=True) + pd.Timedelta(
        seconds=shift_sec
    )
    return out


def merge_prod_training(
    df_train_shifted: pd.DataFrame,
    df_prod: pd.DataFrame,
    suffixes=("", "_prod"),
) -> pd.DataFrame:
    """シフト済み training と prod を timestamp inner merge。"""
    df_train_shifted = df_train_shifted.copy()
    df_prod = df_prod.copy()
    df_train_shifted["timestamp"] = pd.to_datetime(
        df_train_shifted["timestamp"], utc=True
    )
    df_prod["timestamp"] = pd.to_datetime(df_prod["timestamp"], utc=True)
    return df_train_shifted.merge(
        df_prod, on="timestamp", how="inner", suffixes=suffixes
    )


def gate_mask_for_col(
    col: str, train_vals: np.ndarray, prod_vals: np.ndarray, zero_tol: float = 1e-12
) -> np.ndarray:
    """その列の比較対象行を選ぶブール mask。

    - low TF (M0.5/M1/M3): 全行 True (両側有限なら比較)。
    - high TF (>=5分): 「両側とも非ゼロ」 の行のみ True。
      本番ゲートが境界以外で 0、学習も非境界 0 のため、
      0 vs 実値 / 実値 vs 0 の混在比較を排除する。
    """
    finite = np.isfinite(train_vals) & np.isfinite(prod_vals)
    if not is_high_tf_col(col):
        return finite
    both_nonzero = (np.abs(train_vals) > zero_tol) & (np.abs(prod_vals) > zero_tol)
    return finite & both_nonzero


def pair_metrics(b: np.ndarray, c: np.ndarray) -> dict:
    """学習側 b (=B) と本番側 c (=C) の一致メトリクス。"""
    n = len(b)
    if n == 0:
        return dict(n=0, corr=np.nan, diff_med=np.nan, diff_max=np.nan,
                    bit_rate=np.nan, rel_med=np.nan)
    diff = np.abs(b - c)
    corr = (
        float(np.corrcoef(b, c)[0, 1])
        if b.std() > 1e-12 and c.std() > 1e-12
        else np.nan
    )
    rel = diff / (np.abs(b) + 1e-10)
    return dict(
        n=int(n),
        corr=corr,
        diff_med=float(np.median(diff)),
        diff_max=float(diff.max()),
        bit_rate=float((diff < 1e-3).mean() * 100.0),  # 1e-3 未満を bit 一致扱い
        rel_med=float(np.median(rel)),
    )
