#!/usr/bin/env python3
"""
inspect_live_buffer.py — live state pickle から production 実バッファを読み、
                          gap バーの close 差をオフラインで確定する

[ここまでの確定]
  - リサンプル同一、 連続/任意 buffer 長で moment_8@gap = 7680 (= 学習)
  - production live = 115 は S1 連続のどの切り出しからも出ない
    → production のバッファの close 値そのものが S1 と違う、 が唯一の可能性
  - save_state が data_buffers (M5 close 本体) と m05_dataframe を pickle 保存

[本スクリプト (dry-run 不要)]
  feature_engine_state.pkl から:
    (1) data_buffers["M5"]["close"] に stable_moment_k → gap バー値 = 115 か 7680 か
    (2) それを S1 M5 close と位置突合 → どのバーの close が S1 と違うか (gap 周辺)
    (3) m05_dataframe (live M0.5) を S1 M0.5 と週末リオープンで close 突合
        → live 集計と offline S1 集計の差を直接確認

[使い方]
  python inspect_live_buffer.py
  python inspect_live_buffer.py --state /path/to/feature_engine_state.pkl --tf M5
"""

from __future__ import annotations

import sys
import argparse
import pickle
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl

warnings.filterwarnings("ignore", category=FutureWarning)

sys.path.insert(0, "/workspace")
import blueprint as config  # noqa: E402
sys.path.insert(0, str(config.CORE_DIR))
from stable_rolling import stable_moment_k_engine_formula  # noqa: E402


def load_s1(tf: str, col: str = "close") -> pd.DataFrame:
    tf_dir = Path(config.S1_PROCESSED) / f"timeframe={tf}"
    sel = ["timestamp", col] + (["disc"] if tf != "M0.5" else [])
    lf = pl.scan_parquet(str(tf_dir / "*.parquet"))
    have = [c for c in sel if c in lf.collect_schema().names()]
    df = (
        lf.select(have).with_columns(pl.col("timestamp").cast(pl.Datetime("us", "UTC")))
        .sort("timestamp").collect().to_pandas()
    )
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    return df


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--state", type=Path,
                    default=config.STATE_CHECKPOINT_DIR / "feature_engine_state.pkl")
    ap.add_argument("--tf", default="M5")
    ap.add_argument("--window", type=int, default=50)
    ap.add_argument("--moment", type=int, default=8)
    ap.add_argument("--gap-ts", default="2026-05-24 22:00:00")
    args = ap.parse_args()

    gap_ts = pd.Timestamp(args.gap_ts, tz="UTC")
    W, M = args.window, args.moment

    print("=" * 72)
    print("  inspect_live_buffer.py — live state からの gap close 差 (offline)")
    print("=" * 72)
    if not args.state.exists():
        print(f"  [FATAL] state pickle 無し: {args.state}"); return
    with open(args.state, "rb") as f:
        st = pickle.load(f)
    print(f"  state keys: {list(st.keys())}")

    db = st.get("data_buffers", {})
    last_ts = st.get("last_bar_timestamps", {})
    tf = args.tf
    if tf not in db or "close" not in db[tf]:
        print(f"  [FATAL] data_buffers[{tf}][close] 無し"); return

    prod_close = np.asarray(list(db[tf]["close"]), dtype=np.float64)
    prod_disc = (np.asarray(list(db[tf].get("disc", [])), dtype=bool)
                 if "disc" in db[tf] else None)
    n = len(prod_close)
    print(f"\n  [1] production 実バッファ data_buffers[{tf}][close]: {n} 本")
    print(f"      last_bar_timestamp[{tf}] = {last_ts.get(tf)}")

    # (1) production 実バッファで moment 再計算
    prod_mom = stable_moment_k_engine_formula(prod_close, W, M)
    # gap バー位置: disc=True を探す (無ければ S1 突合で特定)
    s1 = load_s1(tf)
    s1_close = s1["close"].to_numpy(np.float64)
    # 末尾を last_ts で揃え、 長さ n で位置突合
    if last_ts.get(tf) is not None:
        end = pd.Timestamp(last_ts[tf])
        if end.tzinfo is None:
            end = end.tz_localize("UTC")
        s1_until = s1[s1["timestamp"] <= end]
    else:
        s1_until = s1
    m = min(n, len(s1_until))
    prod_t = prod_close[-m:]
    s1_t = s1_until["close"].tail(m).to_numpy(np.float64)
    s1_ts_t = s1_until["timestamp"].tail(m).to_numpy()
    diff = np.abs(prod_t - s1_t)

    print(f"\n  [2] production buffer vs S1 {tf} close 位置突合 ({m} 本)")
    print(f"      bit-identical 本数: {(diff < 1e-9).sum()} / {m}")
    if (diff >= 1e-9).any():
        wi = int(np.nanargmax(diff))
        print(f"      最大 close 差: pos -{m - wi}  ts={pd.Timestamp(s1_ts_t[wi])}")
        print(f"        prod close = {prod_t[wi]:+.5f}")
        print(f"        S1   close = {s1_t[wi]:+.5f}")
        print(f"        |Δ|        = {diff[wi]:.6g}")
        # gap バー位置の close
    # gap バーの close と moment を両者で (tz 安全な pandas 比較で位置特定)
    s1_ts_series = pd.to_datetime(pd.Series(s1_ts_t), utc=True)
    gap_mask = (s1_ts_series == gap_ts).to_numpy()
    gi_s1 = np.where(gap_mask)[0]
    if len(gi_s1):
        gi = int(gi_s1[0])
        gi_full = (n - m) + gi
        print(f"\n  [3] gap バー @ {gap_ts}")
        print(f"      prod buffer close = {prod_t[gi]:+.5f}   S1 close = {s1_t[gi]:+.5f}"
              f"   Δ={prod_t[gi]-s1_t[gi]:+.6g}")
        print(f"      ── 現在の prod buffer で moment 再計算 = {prod_mom[gi_full]:+.6g}")

        # 保存された OLS deque (proxy_feature_buffers) に「積まれている」 gap の値
        feat = f"e1a_statistical_moment_{M}_{W}"
        pfb = st.get("proxy_feature_buffers", {}).get(tf, {})
        stored = None
        if feat in pfb and pfb[feat]:
            ydq = np.asarray(list(pfb[feat]), dtype=np.float64)
            # OLS deque 末尾を last_bar_ts に揃え、 gap 位置を同じ規約で特定
            nq = len(ydq)
            mm = min(nq, len(s1_until))
            ts_q = s1_until["timestamp"].tail(mm).to_numpy()
            q_mask = (pd.to_datetime(pd.Series(ts_q), utc=True) == gap_ts).to_numpy()
            qi = np.where(q_mask)[0]
            if len(qi):
                stored = float(ydq[-mm:][qi[0]])
        print(f"      ── 保存 OLS deque に積まれた gap 値 = "
              f"{stored if stored is None else f'{stored:+.6g}'}")
        print(f"      (参照: S1連続=7680.31, live dump y_deque=115.20)")

        # 判定
        recompute_is_7680 = abs(prod_mom[gi_full] - 7680.31) / 7680.31 < 1e-2
        stored_is_115 = stored is not None and abs(stored - 115.203) / 115.203 < 0.1
        if recompute_is_7680 and stored_is_115:
            print("\n      ★★ 確定: 現在 buffer 再計算=7680 だが OLS deque 凍結値=115")
            print("         → 115 は『処理時(5/24)に凍結された stale 値』。")
            print("           buffer/close は正しい。 真因は warmup/catch-up 時の")
            print("           特徴量 populate ロジック (deque へ積む瞬間の状態) にある。")
        elif not recompute_is_7680:
            print(f"\n      現在 buffer 再計算が 7680 でない ({prod_mom[gi_full]:+.6g})")
            print("         → gap 周辺 close の微差を確認 ([2] の差分位置)")

    # (3) m05_dataframe (live M0.5) vs S1 M0.5 週末リオープン突合
    m05 = st.get("m05_dataframe")
    if m05:
        m05_df = pd.DataFrame(list(m05))
        if "timestamp" in m05_df:
            m05_df["timestamp"] = pd.to_datetime(m05_df["timestamp"], utc=True)
            m05_df = m05_df.set_index("timestamp").sort_index()
            s1_m05 = load_s1("M0.5").set_index("timestamp").sort_index()
            print(f"\n  [4] live M0.5 (m05_dataframe {len(m05_df)}本) vs S1 M0.5"
                  f"  リオープン周辺")
            win = pd.date_range(gap_ts - pd.Timedelta(minutes=10),
                                gap_ts + pd.Timedelta(minutes=10), freq="30s", tz="UTC")
            print(f"      {'timestamp':<28}{'live M0.5':>14}{'S1 M0.5':>14}{'Δ':>12}")
            any_diff = False
            for ts in win:
                lv = m05_df["close"].get(ts, np.nan) if "close" in m05_df else np.nan
                s1v = s1_m05["close"].get(ts, np.nan) if "close" in s1_m05 else np.nan
                if not (np.isfinite(lv) or np.isfinite(s1v)):
                    continue
                d = (lv - s1v) if (np.isfinite(lv) and np.isfinite(s1v)) else np.nan
                if np.isfinite(d) and abs(d) > 1e-6:
                    any_diff = True
                mark = "  <== gap" if ts == gap_ts else ""
                print(f"      {str(ts):<28}"
                      f"{(f'{lv:.5f}' if np.isfinite(lv) else '(無)'):>14}"
                      f"{(f'{s1v:.5f}' if np.isfinite(s1v) else '(無)'):>14}"
                      f"{(f'{d:+.5f}' if np.isfinite(d) else '-'):>12}{mark}")
            print(f"\n      → live と S1 の M0.5 に差が{'ある' if any_diff else 'ない'}")
            if any_diff:
                print("        ★ 週末リオープンのデータ集計差が真因確定")
            else:
                print("        M0.5 同一なのに M5 buffer が違うなら、 buffer 構築経路を精査")
    else:
        print("\n  [4] m05_dataframe が state に無い/空")

    print("\n" + "=" * 72)


if __name__ == "__main__":
    main()
