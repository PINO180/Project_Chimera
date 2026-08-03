# /workspace/models/backtest_simulator_cimera_purified
# [V5改修版: Project Cimera 双方向ラベリング仕様 (Part 1)]

import sys
import pickle
from pathlib import Path
import logging
import argparse
from dataclasses import dataclass, field
from typing import Dict, Any, Tuple, List, Optional
import json
import datetime as dt
# import zoneinfo

import polars as pl
import numpy as np
import joblib
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import seaborn as sns
from tqdm import tqdm
import gc

from decimal import Decimal, getcontext, ROUND_HALF_UP

# --- Decimal の精度を設定 (5000桁) ---
getcontext().prec = 5000

# --- プロジェクトのルートディレクトリをPythonの検索パスに追加 ---
project_root = Path(__file__).resolve().parents[1]
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

# --- blueprintから必要なパスをインポート (V5仕様に合わせてパスを調整) ---
from blueprint import (
    S6_WEIGHTED_DATASET,
    S3_SELECTED_FEATURES_PURIFIED_DIR,
    S7_M1_OOF_PREDICTIONS_LONG,
    S7_M1_OOF_PREDICTIONS_SHORT,
    S7_M2_OOF_PREDICTIONS_LONG,
    S7_M2_OOF_PREDICTIONS_SHORT,
    S7_MODELS,
    S7_BACKTEST_CACHE_M1,
    S7_BACKTEST_CACHE_M2,
    S7_BACKTEST_SIM_RESULTS,
)

# --- 出力ファイルパス（実行時に動的生成）---
FINAL_REPORT_PATH = S7_MODELS / "final_backtest_report_v5.json"  # 起動時に上書き
EQUITY_CURVE_PATH = S7_MODELS / "equity_curve_v5.png"  # 起動時に上書き


# --- ロギング設定 ---
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

# XAUUSDの契約サイズ
CONTRACT_SIZE = Decimal("100")  # 1 lot = 100 oz


# ##########################################################################
# ##########################################################################
# ##
# ##   ★★★  設 定 は こ こ だ け 編 集 す れ ば O K  ★★★
# ##
# ##   CLI 引数は覚える必要はありません。ここの値がそのまま既定値になります。
# ##   （CLI で明示した場合だけ、その項目がここより優先されます）
# ##
# ##   実行:  python3 backtest_simulator_cimera.py
# ##
# ##########################################################################
# ##########################################################################

USER_PARAMS: Dict[str, Any] = {
    # ======================================================================
    # 【1】バリア幾何 — ★ラベリング側と必ず一致させること★
    # ----------------------------------------------------------------------
    #   create_proxy_labels の RULE_LONG / RULE_SHORT と同じ値にする。
    #   起動時に自動突合し、pt/sl が違えば警告（strict_geometry=True で停止）。
    #
    #   ★★ TD を絞りたいときは「ここ」だけ変える ★★
    #      下の方にある _LABEL_GEOMETRY_FALLBACK は絶対に触らないこと。
    #      あちらは「ラベリング側が何だったか」の控えであり、
    #      一緒に書き換えると突合ガードが機能しなくなる。
    #
    #   TD について:
    #     ・ラベルの TD 以下  → OK。「その時間で強制決済したら」の再現ができる
    #     ・ラベルの TD より上 → 不可。ラベルがその先を走査していないので
    #                            結果が物理的に存在しない（起動時に停止）
    # ======================================================================
    "pt_multiplier_long": 1.0,  # PT = entry + ATR × これ  (Long)
    "sl_multiplier_long": 5.0,  # SL = entry − ATR × これ  (Long)
    "pt_multiplier_short": 1.0,  # PT = entry − ATR × これ  (Short)
    "sl_multiplier_short": 5.0,  # SL = entry + ATR × これ  (Short)
    "td_minutes_long": 1200.0,  # 強制決済までの分（エントリー起点）
    "td_minutes_short": 1200.0,  # 同上。ラベル TD 以下にすること
    # --- ★ 時間決済モード（バリアを使わない） ---
    #   False にすると PT/SL を一切使わず、【必ず TD 分後に成行決済】する。
    #
    #   なぜ要るか:
    #     測定 (§13.5〜13.6) が測ったのは「t 分後の変位」であって
    #     バリア到達ではない。効果 0.25 ATR に対し ±1.4 ATR のバリアは 5.6 倍広く、
    #     実測では 94% がバリア決着 = ノイズによるコイン投げになっていた
    #     (PT 47.4% / SL 46.9% / TO 5.6%)。
    #     変位を丸ごと取るには時間で畳む必要がある。
    #
    #   使い方: use_barriers=False かつ td_minutes=15 で
    #           「エントリー → 15分後に成行決済」を再現する。
    #   注意  : 損切りが無いので 1 トレードの最大損失は無制限。
    #           エッジの有無を測るためのモードで、そのまま運用する形ではない。
    "use_barriers": False,
    # ======================================================================
    # 【2】発注ゲート — ここを振るのが本番のスイープ
    # ----------------------------------------------------------------------
    #   対称バリア (pt=sl) では p_short ≈ 1 − p_long になるため、
    #   delta ≈ |2·p_long − 1|。つまり m2_proba_threshold=0.55 は
    #   自動的に delta >= 0.10 を課す。delta をそれ未満にしても効かない。
    #   → 対称幾何では m2_proba_threshold を単独で振ること。
    # ======================================================================
    "m2_proba_threshold": 0.70,  # M2 確率の下限
    "m2_delta_threshold": 0.30,  # |p_long − p_short| の下限
    # ======================================================================
    # 【3】ボラティリティ・フィルター（0.0 = 無効）
    # ----------------------------------------------------------------------
    #   min_atr_threshold  : atr_ratio（相対）の下限。ラベル側のゲートは
    #                        撤廃済み(0.0)なので、ここも 0.0 が既定。
    #   max_atr_threshold  : atr_ratio の上限（0.0 = 無効）。
    #                        min と組み合わせて「この帯だけ撃つ」ができる。
    #                        例) 0.0〜0.8 の帯だけ → min 0.0 / max 0.8
    #                            0.5〜0.8 の帯だけ → min 0.5 / max 0.8
    #                        判定は  min <= atr_ratio < max （上限は含まない）
    #   min/max_baseline_atr: baseline_ATR = atr_value / atr_ratio の下限/上限(USD)。
    #                        スプレッドは USD 固定なので、ATR が小さいほど
    #                        バリアに対する相対コストが跳ね上がる。
    #   min_baseline_ratio : 昨日ボラ / 過去N日ボラ。価格水準に依存しない。
    #   min_sar_threshold  : 現在ATR / 過去D日の同時刻ATR平均（季節性調整）。
    #
    #   atr_ratio_bands    : レポートの ATR Ratio Band Analysis の区切り。
    #                        ここを変えると集計表の帯が変わる（発注には無関係）。
    # ======================================================================
    "min_atr_threshold": 0.0,
    "max_atr_threshold": 0.0,  # 0.0 = 上限なし
    # --- ★ atr_value（そのバーの ATR・USD）の下限/上限 ---
    #   バリア幅 = 1.4 × atr_value、スプレッドは USD 固定なので
    #       コスト比 = spread_pips × value_per_pip / (barrier_mult × atr_value × 100)
    #   例) spread_pips=36 / value_per_pip=1.0 / pt=sl=1.4 のとき
    #       ATR 0.5 → コストがバリア幅の 51%（損益分岐勝率 75.7%）
    #       ATR 1.0 → 25.7%（62.9%）
    #       ATR 2.0 → 12.8%（56.4%）
    #       ATR 5.0 →  5.1%（52.6%）
    #   §13.6 の ER ルール損益分岐 ATR 0.96 USD は spread 0.24 前提。
    #   spread 0.36 なら 1.44 USD が下限の目安。
    "min_atr_value": 0.0,  # 0.0 = 下限なし（USD）
    "max_atr_value": 0.0,  # 0.0 = 上限なし（USD）
    "min_baseline_atr": 0.0,  # 例: 1.5 / 2.0 / 2.6 / 3.0 を振る
    "max_baseline_atr": 0.0,  # 0.0 = 上限なし
    "min_baseline_ratio": 0.0,
    "baseline_ratio_lookback_days": 7,
    "min_sar_threshold": 0.0,
    "sar_lookback_days": 10,
    "atr_ratio_bands": [0.5, 0.8, 1.0, 1.2, 1.5],  # 区切り値のリスト
    # ======================================================================
    # 【4】資金管理
    # ======================================================================
    "initial_capital": 1_000_000.0,
    # --- ★ 固定ロットモード（エッジ測定用） ---
    #   True にすると常に fixed_lot_size で発注し、複利・資産減少の影響を消す。
    #   破産で期間が途中で切れないので「1トレードあたりの期待値」を
    #   全期間で観測できる。資金管理の評価には使えない（そのための機能ではない）。
    "use_fixed_lot": True,
    "fixed_lot_size": 0.01,
    "use_fixed_risk": False,  # True: 1トレードのリスクを資産の一定%に固定
    "fixed_risk_percent": 0.02,  # 0.02 = 2%。use_fixed_risk=True のとき有効
    "auto_lot_base_capital": 1000.0,  # use_fixed_risk=False のときだけ使用
    "auto_lot_size_per_base": 0.1,  # 同上
    "base_leverage": 2000.0,
    "min_lot_size": 0.01,
    "min_capital_threshold": 1.0,
    "max_positions": 100,
    # ======================================================================
    # 【5】コスト
    # ======================================================================
    "spread_pips": 16.0,  # 36 pips = 0.36 USD 相当
    "value_per_pip": 1.0,
    # ======================================================================
    # 【6】サーキットブレーカー
    # ======================================================================
    "prevent_simultaneous_orders": True,  # True: delta で勝った片方だけ発注
    "max_consecutive_sl": 2,
    "cooldown_minutes_after_sl": 30,
    "margin_call_percent": 0.0,
    "stop_out_percent": 0.0,
    # ======================================================================
    # 【7】期間・実行制御
    # ======================================================================
    "start_date": None,  # "2021-07-12" のように文字列 or None（全期間）
    "end_date": None,  # "2024-08-02" のように文字列 or None（全期間）
    "test_limit_partitions": 0,  # 0 = 全パーティション。動作確認は 20 等
    # ======================================================================
    # 【9】シグナル源 — 脳を使うか、ルールだけで撃つか
    # ----------------------------------------------------------------------
    #   "model" : M2 OOF の確率でゲート（従来どおり。m2_proba/m2_delta が効く）
    #   "rule"  : ★脳を一切使わない。効率比(ER)と符号つき1バー変位 d だけで撃つ
    #
    #   【rule の中身 — Residual_Drift_Harvester_Theory §13.5〜13.6】
    #     ER(K) = |close[t] − close[t−K]| / Σ|close[i] − close[i−1]|
    #       1に近い = 一直線に走った / 0に近い = 行って戻った
    #     d = (close − open) / ATR   符号つき1バー変位（エントリー時刻に確定）
    #
    #     ER が高い（直近60分すでに走りきった）バーの直後に そこそこ大きい
    #     1本が出ると、その後【逆行】する。行き過ぎの解消。
    #     モデル非依存・6年すべて同符号。
    #
    #   【下の既定値は測定で確定した座標そのもの】
    #     窓20本(60分) ER>=0.396 × |d| 0.6-0.9 × t=15分
    #       → −0.2508 ATR (t=−4.28)  5/5 年 同符号  χ² p=0.4487
    #       → 損益分岐 ATR 0.96 USD（現水準 4〜6 USD）
    #     ※ TD は【1】の td_minutes_long/short を 15 に合わせて使うこと
    #
    #   rule モードでも OOF ファイルは読みます（label_long/short と行集合の
    #   供給元。prediction は使いません）。m2_proba / m2_delta は無効になります。
    # ======================================================================
    "signal_source": "rule",  # "model" or "rule"
    "rule_er_column": "eff_ratio_20_M3",  # 効率比の列名（窓20本=60分）
    "rule_er_min": 0.396,  # ER の下限
    "rule_er_max": 0.0,  # ER の上限（0.0 = 上限なし）
    "rule_d_column": "d_atr_M3",  # 符号つき1バー変位の列名
    "rule_d_abs_min": 0.6,  # |d| の下限
    "rule_d_abs_max": 0.9,  # |d| の上限
    "rule_direction": "reverse",  # "reverse" = −sign(d) / "follow" = +sign(d)
    # ======================================================================
    # 【8】幾何チェックの厳格さ
    # ----------------------------------------------------------------------
    #   True  : ラベルと pt/sl が食い違ったら起動時に停止（推奨）
    #   False : 警告だけ出して続行（レポートには GEOMISMATCH が付く）
    #   ※ TD をラベルより長くした場合は、この設定に関わらず必ず停止します
    # ======================================================================
    "strict_geometry": True,
}


# ==========================================================================
# [GEOMETRY-SYNC] 学習側ラベル幾何との突合
# --------------------------------------------------------------------------
# 事故の記録:
#   ラベルを対称 1.4 / TD60 で作り直した後、BT を CLI 引数付きで起動したところ、
#   --pt-long 等を省略したために argparse の default が旧 dataclass 既定値
#   (pt=1.0 / sl=5.0 / td=30) を拾い、旧幾何のまま評価したレポートが出力された。
#   レポートは正常終了して見えるため、幾何が違うことに気づく手掛かりが
#   「Strategy: L(PT1.0/SL5.0)」の 1 行しかなかった。
#
# 対策:
#   BT 起動時に create_proxy_labels のソースを *テキストとして* 読み、
#   RULE_LONG / RULE_SHORT / ATR_RATIO_THRESHOLD / ACTION_HORIZON_SEC を
#   抽出して実効 config と突合する。import しないのは副作用 (numba JIT 登録・
#   blueprint 経由の重い依存) を持ち込まないため。
# ==========================================================================

# ──────────────────────────────────────────────────────────────────────────
# ▼▼▼ ここから下は編集不要 ▼▼▼
# ──────────────────────────────────────────────────────────────────────────
#
#  ╔════════════════════════════════════════════════════════════════════╗
#  ║  ★ 編集禁止 ★  _LABEL_GEOMETRY_FALLBACK                            ║
#  ║                                                                    ║
#  ║  これは「ラベリング側が何だったか」の控えであり、BT の動作値では    ║
#  ║  ありません。TD や pt/sl を変えたいときは USER_PARAMS【1】を編集。 ║
#  ║                                                                    ║
#  ║  通常は create_proxy_labels のソースを直接読むので、この定数は      ║
#  ║  一切参照されません。使われるのは「ラベリングのソースが見つから    ║
#  ║  なかった環境」だけで、その場合は起動時に WARNING が出ます。        ║
#  ║                                                                    ║
#  ║  ここを USER_PARAMS と一緒に書き換えると、突合ガードが「ラベルも   ║
#  ║  その値だった」と誤認し、本来止めるべき TD 延長を素通りさせます。   ║
#  ║  = 自分で嘘をつく物差しになるので、絶対に触らないこと。             ║
#  ╚════════════════════════════════════════════════════════════════════╝
#
# 値の出典: Residual_Drift_Harvester_Theory §13.11「確定した再設計」
_LABEL_GEOMETRY_FALLBACK: Dict[str, float] = {
    "pt_mult_long": 1.4,
    "sl_mult_long": 1.4,
    "pt_mult_short": 1.4,
    "sl_mult_short": 1.4,
    "td_minutes_long": 60.0,
    "td_minutes_short": 60.0,
    "atr_ratio_threshold": 0.0,
    "action_horizon_sec": 180.0,
}

LABELING_SCRIPT_NAME = "create_proxy_labels_polars_patch_regime_Universal_Brain_V5.py"

# ==========================================================================
# [TD-RESIM] ラベル側の TD と行動地平 — TD 短縮リシミュレーションの土台
# --------------------------------------------------------------------------
# S6 の duration_long/short は「t0 = L からの経過分」であり、エントリー時刻は
# L + ACTION_HORIZON_SEC (= L+3分)。したがって
#     エントリー起点の経過分 = duration − ACTION_HORIZON_MIN
#     タイムアウト行の duration = ACTION_HORIZON_MIN + TD_label   (= 63.0)
# BT の td_minutes_* は「エントリー起点の分」として扱う。
#
# TD をラベルより短くした再シミュレーションは duration から厳密に再現できる:
#     経過 <  新TD → ラベル通りの PT/SL
#     経過 >= 新TD → その時点で強制決済 (close_future)
# 逆に TD をラベルより長くすることは、ラベルが走査していない時間帯の結果を
# 要求するため原理的に不可能。
# ==========================================================================
LABEL_TD_MINUTES: float = 60.0  # __main__ で実ラベルから上書き
ACTION_HORIZON_MIN: float = 3.0  # __main__ で実ラベルから上書き

# 起動時のラベル幾何突合の結果 (レポート JSON へ埋め込む)
GEOMETRY_CHECK_RESULT: Dict[str, Any] = {"status": "not_checked"}


def _required_extra_cols(cfg: "BacktestConfig") -> List[str]:
    """signal_source に応じて S6 から追加で必要になる列。"""
    if str(cfg.signal_source).lower() == "rule":
        return [c for c in (cfg.rule_er_column, cfg.rule_d_column) if c]
    return []


def _s6_actual_columns(cfg: "BacktestConfig") -> Optional[List[str]]:
    """S6 の実スキーマを1ファイルだけ読んで返す (推測しないための確認)。"""
    try:
        base = Path(str(cfg.simulation_data_path))
        files = sorted(base.rglob("*.parquet"))
        if not files:
            return None
        return list(pl.read_parquet_schema(files[-1]).keys())
    except Exception:
        return None


def validate_preload_columns(data, cfg: "BacktestConfig", cache_path: Path) -> bool:
    """プリロード済みデータに必要列が揃っているか検証する。

    [CACHE-GUARD] backtest_preload_cache.pkl は base_cols を変更しても
    自動では作り直されない。古いキャッシュを掴んだまま走ると、
    シミュレーション途中の KeyError で初めて気づくことになる。
    ここで起動直後に検出し、原因を「キャッシュが古い」/「S6 に列が無い」の
    どちらかまで切り分ける。

    Returns:
        True  : そのまま続行してよい
        False : キャッシュを削除して再生成すべき (呼び出し側で実施)
    """
    req = _required_extra_cols(cfg)
    if not req:
        return True

    preloaded_dict, _ = data
    cols: Optional[set] = None
    for _v in preloaded_dict.values():
        if _v is not None and len(_v) > 0:
            cols = set(_v.columns)
            break
    if cols is None:
        return True

    missing = [c for c in req if c not in cols]
    if not missing:
        logging.info(f"[CACHE-GUARD] 必要列は揃っています: {req}")
        return True

    # --- 原因の切り分け: S6 の実スキーマを直接見る ---
    s6_cols = _s6_actual_columns(cfg)
    logging.warning("=" * 68)
    logging.warning(f"[CACHE-GUARD] プリロードデータに必要列がありません: {missing}")

    if s6_cols is None:
        logging.warning("  S6 のスキーマを読めませんでした。パスを確認してください:")
        logging.warning(f"    {cfg.simulation_data_path}")
        logging.warning("=" * 68)
        raise SystemExit("[CACHE-GUARD] S6 スキーマ確認不可のため中止しました。")

    s6_missing = [c for c in req if c not in s6_cols]

    if not s6_missing:
        # S6 にはある → キャッシュが古いだけ。作り直せば直る。
        logging.warning("  → S6 には存在します。キャッシュが古いだけです。")
        logging.warning(f"     {cache_path}")
        logging.warning("  → 自動で削除して再生成します。")
        logging.warning("=" * 68)
        return False

    # S6 にも無い → ラベリングがその列を出力していない
    import difflib as _dl

    logging.warning("  → S6 にも存在しません。ラベリングがこの列を出力していません。")
    for c in s6_missing:
        near = _dl.get_close_matches(c, s6_cols, n=5, cutoff=0.4)
        prefix = c.split("_")[0]
        pat = [x for x in s6_cols if x.startswith(prefix)]
        logging.warning(f"    '{c}' の候補: 類似={near} / 同接頭辞={pat}")
    logging.warning("")
    logging.warning("  S6 の全列 (実測):")
    for c in sorted(s6_cols):
        logging.warning(f"    - {c}")
    logging.warning("=" * 68)
    raise SystemExit(
        "[CACHE-GUARD] S6 に必要列がないため中止しました。"
        " USER_PARAMS【9】の rule_er_column / rule_d_column を"
        " 上の実列名に合わせるか、ラベリングを再実行してください。"
    )


def _build_band_defs(edges) -> List[Any]:
    """区切り値のリストから [(帯名, 判定関数), ...] を作る。

    例: [0.5, 0.8, 1.0] → "< 0.5" / "0.5-0.8" / "0.8-1.0" / ">= 1.0"
    判定は下限を含み上限を含まない (lo <= x < hi)。
    ルール6: lambda はデフォルト引数で束縛し late binding を避ける。
    """
    e = sorted(float(x) for x in (edges or []))
    if not e:
        return [("all", lambda x: True)]
    defs: List[Any] = [(f"< {e[0]:g}", lambda x, hi=e[0]: x < hi)]
    for a, b in zip(e[:-1], e[1:]):
        defs.append((f"{a:g}-{b:g}", lambda x, lo=a, hi=b: lo <= x < hi))
    defs.append((f">= {e[-1]:g}", lambda x, lo=e[-1]: x >= lo))
    return defs


def _find_labeling_script() -> Optional[Path]:
    """create_proxy_labels のソースを探索する。

    探索順:
      1. /workspace/models  ← 実配置 (確認済み、2026-07)
      2. /workspace/scripts, /workspace
      3. 本スクリプトの隣 / 親の models/
      4. 上記が全部外れたら /workspace 配下を再帰検索 (ファイル名パターン)
    4 を入れてあるのは、パスを当て推量しないため。移動しても見つかる。
    """
    candidates: List[Path] = []
    for d in ("/workspace/models", "/workspace/scripts", "/workspace"):
        candidates.append(Path(d) / LABELING_SCRIPT_NAME)
    here = Path(__file__).resolve().parent
    candidates.append(here / LABELING_SCRIPT_NAME)
    candidates.append(here.parent / "models" / LABELING_SCRIPT_NAME)
    for p in candidates:
        try:
            if p.is_file():
                return p
        except OSError:
            continue

    # --- 再帰フォールバック: 名前で探す ---
    for root in (Path("/workspace"), here.parent):
        try:
            if not root.is_dir():
                continue
            hits = sorted(root.rglob("create_proxy_labels_*.py"))
            if hits:
                logging.info(
                    f"[GEOMETRY-SYNC] 既定パスに無かったため再帰検索で発見: {hits[0]}"
                )
                return hits[0]
        except OSError:
            continue
    return None


def _parse_labeling_geometry(src_path: Path) -> Optional[Dict[str, float]]:
    """ラベリングスクリプトから幾何定数を正規表現で抽出する (import しない)。

    抽出対象:
        RULE_LONG  = {"pt_mult": X, "sl_mult": Y, "td": "Nm"}
        RULE_SHORT = {...}
        ATR_RATIO_THRESHOLD = Z
        ACTION_HORIZON_SEC  = W   (クラス属性なのでインデント許容)
    どれか 1 つでも読めなければ None を返す (部分一致で誤判定しないため)。
    """
    import re as _re

    try:
        src = src_path.read_text(encoding="utf-8", errors="ignore")
    except OSError:
        return None

    def _rule(name: str) -> Optional[Dict[str, float]]:
        m = _re.search(name + r"\s*=\s*\{(.*?)\}", src, _re.DOTALL)
        if not m:
            return None
        body = m.group(1)
        pt = _re.search(r'["\']pt_mult["\']\s*:\s*([0-9.]+)', body)
        sl = _re.search(r'["\']sl_mult["\']\s*:\s*([0-9.]+)', body)
        td = _re.search(r'["\']td["\']\s*:\s*["\']\s*([0-9.]+)\s*m["\']', body)
        if not (pt and sl and td):
            return None
        return {
            "pt": float(pt.group(1)),
            "sl": float(sl.group(1)),
            "td": float(td.group(1)),
        }

    rl = _rule("RULE_LONG")
    rs = _rule("RULE_SHORT")
    m_atr = _re.search(r"^ATR_RATIO_THRESHOLD\s*=\s*([0-9.]+)", src, _re.MULTILINE)
    m_ah = _re.search(r"^\s*ACTION_HORIZON_SEC\s*=\s*([0-9]+)", src, _re.MULTILINE)
    if rl is None or rs is None or m_atr is None:
        return None

    return {
        "pt_mult_long": rl["pt"],
        "sl_mult_long": rl["sl"],
        "td_minutes_long": rl["td"],
        "pt_mult_short": rs["pt"],
        "sl_mult_short": rs["sl"],
        "td_minutes_short": rs["td"],
        "atr_ratio_threshold": float(m_atr.group(1)),
        "action_horizon_sec": float(m_ah.group(1)) if m_ah else 180.0,
    }


def verify_geometry_against_labeling(
    config: "BacktestConfig", strict: bool = False
) -> Dict[str, Any]:
    """実効 BT 幾何とラベリング側幾何を突合し、結果 dict を返す。"""
    effective = {
        "pt_mult_long": float(config.pt_multiplier_long),
        "sl_mult_long": float(config.sl_multiplier_long),
        "pt_mult_short": float(config.pt_multiplier_short),
        "sl_mult_short": float(config.sl_multiplier_short),
        "td_minutes_long": float(config.td_minutes_long),
        "td_minutes_short": float(config.td_minutes_short),
        "atr_ratio_threshold": float(config.min_atr_threshold),
    }

    src_path = _find_labeling_script()
    parsed = _parse_labeling_geometry(src_path) if src_path else None

    if parsed is None:
        reference = dict(_LABEL_GEOMETRY_FALLBACK)
        origin = "_LABEL_GEOMETRY_FALLBACK (静的定数)"
        logging.warning("=" * 68)
        logging.warning(
            "[GEOMETRY-SYNC] ⚠ ラベリングのソースが見つかりませんでした: "
            f"{LABELING_SCRIPT_NAME}"
        )
        logging.warning(
            "  → 静的定数 _LABEL_GEOMETRY_FALLBACK と突合します。"
            " この定数が古いと突合そのものが無意味になります。"
        )
        logging.warning(
            "  → 探索先: /workspace/models, /workspace/scripts, /workspace,"
            " および本スクリプトの隣・親/models"
        )
        logging.warning("=" * 68)
    else:
        reference = parsed
        origin = str(src_path)

    # pt/sl : ラベルの定義そのもの。ずれたら PnL に意味が無い → critical
    # td    : ラベルより短いのは正当な再シミュレーション、長いのは不可能 → 条件付き
    # atrゲート: ラベル後の選抜フィルタ。スイープは正当 → informational
    # use_barriers=False のときは PT/SL を一切使わないので、幾何不一致は無害。
    CRITICAL_KEYS = (
        ()
        if not getattr(config, "use_barriers", True)
        else ("pt_mult_long", "sl_mult_long", "pt_mult_short", "sl_mult_short")
    )
    TD_KEYS = ("td_minutes_long", "td_minutes_short")

    diffs: List[str] = []
    notes: List[str] = []
    for key, ref_val in reference.items():
        if key == "action_horizon_sec":
            continue
        eff_val = effective.get(key)
        if eff_val is None:
            continue
        if abs(eff_val - float(ref_val)) > 1e-9:
            msg = f"{key}: BT={eff_val} vs LABEL={ref_val}"
            if key in CRITICAL_KEYS:
                diffs.append(msg)
            elif key in TD_KEYS:
                if eff_val > float(ref_val) + 1e-9:
                    diffs.append(
                        msg + "  ← TD をラベルより長くすることはできません "
                        "(ラベルが走査していない時間帯の結果は存在しない)"
                    )
                else:
                    notes.append(
                        msg + f"  ← TD 短縮リシミュレーション "
                        f"(経過 >= {eff_val}分 の PT/SL は強制決済に再分類)"
                    )
            else:
                notes.append(msg)

    status = "match" if not diffs else "mismatch"

    logging.info("=" * 68)
    logging.info("[GEOMETRY-SYNC] ラベル幾何との突合")
    logging.info(f"  参照元: {origin}")
    logging.info(
        f"  BT   : PT L/S = {effective['pt_mult_long']}/{effective['pt_mult_short']}, "
        f"SL L/S = {effective['sl_mult_long']}/{effective['sl_mult_short']}, "
        f"TD L/S = {effective['td_minutes_long']}/{effective['td_minutes_short']} min, "
        f"min_atr = {effective['atr_ratio_threshold']}"
    )
    logging.info(
        f"  LABEL: PT L/S = {reference['pt_mult_long']}/{reference['pt_mult_short']}, "
        f"SL L/S = {reference['sl_mult_long']}/{reference['sl_mult_short']}, "
        f"TD L/S = {reference['td_minutes_long']}/{reference['td_minutes_short']} min, "
        f"atr_gate = {reference['atr_ratio_threshold']}"
    )

    if notes:
        logging.info("  ℹ️  幾何以外の差分 (BT を無効にはしない):")
        for n in notes:
            logging.info(f"     - {n}")

    if status == "match":
        logging.info("  ✅ バリア幾何 (pt/sl) はラベルと一致しています。")
        logging.info("=" * 68)
    else:
        logging.warning("  " + "!" * 62)
        logging.warning("  ❌ 幾何が食い違っています。この BT 結果は無効です。")
        for d in diffs:
            logging.warning(f"     - {d}")
        logging.warning("     USER_PARAMS の【1】バリア幾何 を修正してください。")
        logging.warning("  " + "!" * 62)
        logging.info("=" * 68)
        # TD 延長は strict の有無に関わらず必ず停止 (物理的に不可能なため)
        _td_violation = any("TD をラベルより長く" in d for d in diffs)
        if strict or _td_violation:
            raise SystemExit(
                "[GEOMETRY-SYNC] 幾何不一致のため中止しました。上のログを確認してください。"
            )

    return {
        "status": status,
        "source": origin,
        "labeling": reference,
        "backtest": effective,
        "critical_diffs": diffs,
        "informational_diffs": notes,
    }


# # タイムゾーン変換用
# JST = zoneinfo.ZoneInfo("Asia/Tokyo")


# ================================================================
# フェーズ 0: 作戦司令室 (パラメータ設定 - V5仕様)
# ================================================================
@dataclass
class BacktestConfig:
    """シミュレーションの全パラメータを一元管理 (V5 Two-Brain Architecture)

    ★ 既定値はすべてファイル冒頭の USER_PARAMS から取得する。
      値を変えたいときは USER_PARAMS だけを編集すること。
      （ここを直接書き換えると USER_PARAMS と二重管理になるので触らない）
    """

    initial_capital: float = USER_PARAMS["initial_capital"]
    simulation_data_path: Path = S6_WEIGHTED_DATASET

    # 期間フィルタ (YYYY-MM-DD, UTC, inclusive)。None なら全期間
    start_date: Optional[str] = USER_PARAMS["start_date"]
    end_date: Optional[str] = USER_PARAMS["end_date"]

    # V5: Long/Short独立のOOF予測パス
    oof_long_path: Path = S7_M2_OOF_PREDICTIONS_LONG
    oof_short_path: Path = S7_M2_OOF_PREDICTIONS_SHORT

    # V5: 純化特徴量ディレクトリ (In-Sample拡張時の布石として保持)
    purified_features_dir: Path = S3_SELECTED_FEATURES_PURIFIED_DIR

    # --- 資金管理 ---
    auto_lot_base_capital: float = USER_PARAMS["auto_lot_base_capital"]
    auto_lot_size_per_base: float = USER_PARAMS["auto_lot_size_per_base"]
    use_fixed_lot: bool = USER_PARAMS["use_fixed_lot"]
    fixed_lot_size: float = USER_PARAMS["fixed_lot_size"]
    use_fixed_risk: bool = USER_PARAMS["use_fixed_risk"]
    fixed_risk_percent: float = USER_PARAMS["fixed_risk_percent"]

    # --- 発注ゲート ---
    m2_proba_threshold: float = USER_PARAMS["m2_proba_threshold"]
    m2_delta_threshold: float = USER_PARAMS["m2_delta_threshold"]

    test_limit_partitions: int = USER_PARAMS["test_limit_partitions"]
    oof_mode: bool = True
    min_capital_threshold: float = USER_PARAMS["min_capital_threshold"]
    min_lot_size: float = USER_PARAMS["min_lot_size"]

    # --- ボラティリティ・フィルター (0.0 = 無効) ---
    min_atr_threshold: float = USER_PARAMS["min_atr_threshold"]
    max_atr_threshold: float = USER_PARAMS["max_atr_threshold"]
    min_atr_value: float = USER_PARAMS["min_atr_value"]
    max_atr_value: float = USER_PARAMS["max_atr_value"]
    min_baseline_atr: float = USER_PARAMS["min_baseline_atr"]
    max_baseline_atr: float = USER_PARAMS["max_baseline_atr"]
    min_baseline_ratio: float = USER_PARAMS["min_baseline_ratio"]
    baseline_ratio_lookback_days: int = USER_PARAMS["baseline_ratio_lookback_days"]
    min_sar_threshold: float = USER_PARAMS["min_sar_threshold"]
    sar_lookback_days: int = USER_PARAMS["sar_lookback_days"]
    atr_ratio_bands: List[float] = field(
        default_factory=lambda: list(USER_PARAMS["atr_ratio_bands"])
    )

    max_positions: int = USER_PARAMS["max_positions"]

    # --- サーキットブレーカーと同時発注禁止 ---
    prevent_simultaneous_orders: bool = USER_PARAMS["prevent_simultaneous_orders"]
    max_consecutive_sl: int = USER_PARAMS["max_consecutive_sl"]
    cooldown_minutes_after_sl: int = USER_PARAMS["cooldown_minutes_after_sl"]

    base_leverage: float = USER_PARAMS["base_leverage"]
    spread_pips: float = USER_PARAMS["spread_pips"]
    value_per_pip: float = USER_PARAMS["value_per_pip"]

    # ==========================================
    # バリア幾何 — ★ラベリング側と一致必須★ (USER_PARAMS【1】)
    # ==========================================
    sl_multiplier_long: float = USER_PARAMS["sl_multiplier_long"]
    pt_multiplier_long: float = USER_PARAMS["pt_multiplier_long"]
    sl_multiplier_short: float = USER_PARAMS["sl_multiplier_short"]
    pt_multiplier_short: float = USER_PARAMS["pt_multiplier_short"]

    use_barriers: bool = USER_PARAMS["use_barriers"]
    td_minutes_long: float = USER_PARAMS["td_minutes_long"]
    td_minutes_short: float = USER_PARAMS["td_minutes_short"]

    # ==========================================
    # シグナル源 (USER_PARAMS【9】)
    # ==========================================
    signal_source: str = USER_PARAMS["signal_source"]
    rule_er_column: str = USER_PARAMS["rule_er_column"]
    rule_er_min: float = USER_PARAMS["rule_er_min"]
    rule_er_max: float = USER_PARAMS["rule_er_max"]
    rule_d_column: str = USER_PARAMS["rule_d_column"]
    rule_d_abs_min: float = USER_PARAMS["rule_d_abs_min"]
    rule_d_abs_max: float = USER_PARAMS["rule_d_abs_max"]
    rule_direction: str = USER_PARAMS["rule_direction"]

    # ==========================================
    # 証拠金維持率とロスカット設定
    # ==========================================
    margin_call_percent: float = USER_PARAMS["margin_call_percent"]
    stop_out_percent: float = USER_PARAMS["stop_out_percent"]


class BacktestSimulator:
    def __init__(self, config: BacktestConfig):
        self.config = config

        # V5仕様: OOFモード専用のため、旧In-Sample用のモデルやTop 50特徴量のロードは完全撤廃
        if not self.config.oof_mode:
            raise NotImplementedError(
                "In-Sample mode is disabled in V5. Please use --oof."
            )

        self._current_capital = Decimal(str(self.config.initial_capital))

    def _get_effective_leverage(self, equity: Decimal) -> Decimal:
        """
        有効証拠金に基づいてExnessのレバレッジ制限を適用

        [FIX-3] extreme_risk_engine._get_exness_leverage() と完全統一:
          equity < $5,000    → base_leverage そのまま
          equity < $30,000   → 2000倍上限
          equity < $100,000  → 1000倍上限
          equity >= $100,000 → 500倍上限
        """
        base_leverage_dec = Decimal(str(self.config.base_leverage))
        if equity < Decimal("5000"):
            limit_leverage = base_leverage_dec  # 上限なし (base_leverage に従う)
        elif equity < Decimal("30000"):
            limit_leverage = Decimal("2000")
        elif equity < Decimal("100000"):
            limit_leverage = Decimal("1000")
        else:
            limit_leverage = Decimal("500")
        return base_leverage_dec.min(limit_leverage)

    def preload_data(self) -> Tuple[Dict[dt.date, pl.DataFrame], pl.DataFrame]:
        """
        Optuna超高速化用: 全パーティションのデータを1回だけ読み込み、
        メモリ上の辞書(Dict)にキャッシュして返す。
        """
        logging.info("Pre-loading all data into memory (Optimized Single Scan)...")
        lf, partitions_df = self._prepare_data()

        # 1回のcollect()で全件メモリに乗せる（1382回のディスクスキャンを回避）
        logging.info("Executing single collect() pass on dataset. Please wait...")
        df_all = lf.with_columns(pl.col("timestamp").dt.date().alias("date")).collect()

        # [baseline_ratio相対フィルター用] 全データを時系列ソートしてbaseline_ratioを一括計算
        # baseline_atr  = atr_value / (atr_ratio + 1e-10)  = 過去1日(480本)ATR平均
        # baseline_7d   = baseline_atrのrolling_mean(N日分=N×480本)
        # baseline_ratio = baseline_atr / baseline_7d
        #   = 「昨日のボラ」 vs 「過去N日のボラ平均」の相対比率
        # ※ 全期間データに対して一括計算することでパーティション境界の問題を回避
        bars_per_day = 480  # M3: 1日=480本
        long_window = bars_per_day * self.config.baseline_ratio_lookback_days
        df_all = (
            df_all.sort("timestamp")
            .with_columns(
                [
                    (pl.col("atr_value") / (pl.col("atr_ratio") + 1e-10)).alias(
                        "_baseline_atr"
                    ),
                ]
            )
            .with_columns(
                [
                    pl.col("_baseline_atr")
                    .rolling_mean(window_size=long_window, min_samples=bars_per_day)
                    .alias("_baseline_long"),
                ]
            )
            .with_columns(
                [
                    (
                        pl.col("_baseline_atr") / (pl.col("_baseline_long") + 1e-10)
                    ).alias("baseline_ratio"),
                ]
            )
            .drop(["_baseline_atr", "_baseline_long"])
        )
        logging.info(
            f"baseline_ratio computed: lookback={self.config.baseline_ratio_lookback_days}days "
            f"({long_window}bars), null_count={df_all['baseline_ratio'].null_count()}"
        )

        # [SAR: 日中季節性調整済み相対ATRフィルター]
        # SAR = 現在ATR / 過去D日間の同時刻ATR平均
        # 設計原則（Gemini Deep Research推奨・案C）:
        #   - UTC時刻（時・分）でグループ化し「同時刻」のATR平均をベースラインにする
        #   - shift(1) で当日データを除外 → 重複ウィンドウ問題を完全回避
        #   - Tokyo静/London活発の日中季節性を分離評価
        #   - 前日が祝日で静くても当日London閾値は過去D日のLondon基準
        # min_sar_threshold=0.0 のとき計算をスキップ（後方互換）
        if self.config.min_sar_threshold > 0.0:
            logging.info(
                f"Computing SAR (Seasonality-Adjusted Ratio): "
                f"lookback={self.config.sar_lookback_days}days..."
            )
            df_all = (
                df_all.sort("timestamp")
                .with_columns(
                    [
                        # UTC時刻キー: (hour, minute) でグループ化
                        pl.col("timestamp").dt.hour().alias("_tod_h"),
                        pl.col("timestamp").dt.minute().alias("_tod_m"),
                    ]
                )
                .with_columns(
                    [
                        # 同時刻グループ内でshift(1)して当日を除外し、
                        # 過去D日分（sar_lookback_days本）の移動平均をベースラインに
                        pl.col("atr_value")
                        .shift(1)
                        .rolling_mean(
                            window_size=self.config.sar_lookback_days,
                            min_samples=max(1, self.config.sar_lookback_days // 2),
                        )
                        .over(["_tod_h", "_tod_m"])
                        .alias("_sar_baseline"),
                    ]
                )
                .with_columns(
                    [
                        # SAR = 現在ATR / 同時刻ベースライン
                        (pl.col("atr_value") / (pl.col("_sar_baseline") + 1e-10)).alias(
                            "sar"
                        ),
                    ]
                )
                .drop(["_tod_h", "_tod_m", "_sar_baseline"])
            )
            logging.info(
                f"SAR computed: null_count={df_all['sar'].null_count()}, "
                f"mean={df_all['sar'].drop_nulls().mean():.4f}"
            )
        else:
            # min_sar_threshold=0.0: SAR列を作らない（後方互換）
            pass

        preloaded_dict = {}
        partitions_to_process = partitions_df

        if self.config.test_limit_partitions > 0:
            partitions_to_process = partitions_df.head(
                self.config.test_limit_partitions
            )

        # メモリ上のDataFrameから日付ごとに切り出す（超高速）
        for row in tqdm(
            partitions_to_process.iter_rows(named=True),
            total=len(partitions_to_process),
            desc="Splitting to Dictionary",
        ):
            current_date = row["date"]
            df_chunk = df_all.filter(pl.col("date") == current_date)
            if not df_chunk.is_empty():
                preloaded_dict[current_date] = df_chunk

        del df_all
        gc.collect()

        logging.info(
            f"Successfully preloaded {len(preloaded_dict)} partitions into memory."
        )
        return preloaded_dict, partitions_to_process

    # ▼▼▼ def run(self): を引数付きに変更 ▼▼▼
    def run(
        self, preloaded_data: Tuple[Dict[dt.date, pl.DataFrame], pl.DataFrame] = None
    ):
        logging.info("### Project Forge V5 Backtest Simulator: START ###")
        logging.info(
            f"Strategy: Fixed Risk ({self.config.fixed_risk_percent * 100:.1f}%), "
            f"Base Leverage = {self.config.base_leverage}, "
            f"Spread = {self.config.spread_pips} pips"
        )

        # =========================================================
        # オンメモリデータの受け取り、または単独実行時の自動ロード
        # =========================================================
        if preloaded_data is not None:
            preloaded_dict, partitions_to_process = preloaded_data
            logging.info("Using PRELOADED data from memory (Ultra-fast mode).")
        else:
            # Optunaを使わず、このスクリプトを単独で実行した場合の処理
            preloaded_dict, partitions_to_process = self.preload_data()

        all_results_dfs = []
        all_trade_logs = []

        self._current_capital = Decimal(str(self.config.initial_capital))
        DECIMAL_MIN_CAPITAL = Decimal(str(self.config.min_capital_threshold))

        self.cb_simultaneous_prevented = 0
        self.cb_simultaneous_taken = 0  # 両建て成立回数
        self.rule_fire_count = 0  # [RULE-MODE] ルール発火数
        self.cb_cooldown_long = 0
        self.cb_cooldown_short = 0
        self.high_water_mark = self._current_capital
        self.min_margin_level_pct = Decimal("inf")
        self.stop_out_count = 0
        # 連続SL/Loss最大値（チャンク間で引き継ぎ）
        self.max_consec_sl_long = 0
        self.max_consec_sl_short = 0
        self.max_consec_sl_total = 0
        self.max_consec_loss_total = 0

        # tqdmのプログレスバーはOptuna側で大量に出力されると邪魔なので、
        # オンメモリ(preloaded_dataあり)の場合はバーを非表示(disable=True)にする
        disable_tqdm = preloaded_data is not None

        for row in tqdm(
            partitions_to_process.iter_rows(named=True),
            total=len(partitions_to_process),
            desc="Simulating Partitions",
            disable=disable_tqdm,  # ★追加: Optuna実行時は静かに回す
        ):
            current_date = row["date"]

            # ▼▼▼ 激重だった collect() 処理を廃止し、メモリ(辞書)から一瞬で取り出す ▼▼▼
            df_chunk = preloaded_dict.get(current_date)

            if df_chunk is None or df_chunk.is_empty():
                continue

            try:
                if self._current_capital < DECIMAL_MIN_CAPITAL:
                    # 破産した場合はそれ以降の日付をスキップ
                    break

                # 取得したメモリ上のデータを使ってシミュレーションを実行
                results_chunk_df, trade_log_chunk_df = self._run_simulation_loop(
                    df_chunk
                )

                all_results_dfs.append(results_chunk_df)
                all_trade_logs.append(trade_log_chunk_df)

                del df_chunk, results_chunk_df, trade_log_chunk_df
                gc.collect()

            except Exception as e:
                logging.error(
                    f"Error processing partition {current_date}: {e}", exc_info=True
                )
                continue

        if not all_results_dfs:
            logging.error("No simulation results were generated. Cannot create report.")
            return

        logging.info("Concatenating results from all partitions...")
        try:
            final_results_df = pl.concat(all_results_dfs).sort("timestamp")
            final_trade_log_df = (
                pl.concat([df for df in all_trade_logs if not df.is_empty()]).sort(
                    "timestamp"
                )
                if any(not df.is_empty() for df in all_trade_logs)
                else pl.DataFrame()
            )

        except Exception as e:
            logging.error(f"Error concatenating results: {e}", exc_info=True)
            return

        if final_results_df.is_empty():
            logging.error(
                "Concatenated results DataFrame is empty. Cannot generate report."
            )
            return

        # ▼▼▼ 修正前 ▼▼▼
        # self._analyze_and_report(final_results_df, final_trade_log_df)
        # logging.info("### Project Forge V5 Backtest Simulator: FINISHED ###")

        # ▼▼▼ 修正後 ▼▼▼
        report_data = self._analyze_and_report(final_results_df, final_trade_log_df)

        # report_dataに追加統計をねじ込む（ここで初めてjson.dumpする）
        report_data["min_margin_level_pct"] = (
            float(self.min_margin_level_pct)
            if self.min_margin_level_pct != Decimal("inf")
            else 9999.0
        )
        report_data["max_consec_sl_long"] = self.max_consec_sl_long
        report_data["max_consec_sl_short"] = self.max_consec_sl_short
        report_data["max_consec_sl_total"] = self.max_consec_sl_total
        report_data["max_consec_loss_total"] = self.max_consec_loss_total

        # [GEOMETRY-SYNC] 何を評価したのかをレポート自身に残す。
        #   後からレポートだけ見て「このBTは信じてよいか」を判定できるようにする。
        report_data["geometry_check"] = GEOMETRY_CHECK_RESULT
        report_data["signal_source"] = str(self.config.signal_source)
        report_data["rule_fire_count"] = int(getattr(self, "rule_fire_count", 0))
        report_data["label_td_minutes"] = float(LABEL_TD_MINUTES)
        report_data["action_horizon_min"] = float(ACTION_HORIZON_MIN)
        report_data["effective_settings"] = {
            k: getattr(self.config, k)
            for k in USER_PARAMS.keys()
            if hasattr(self.config, k)
        }

        try:
            with open(FINAL_REPORT_PATH, "w") as f:
                json.dump(report_data, f, indent=4, default=str)
            logging.info(f"Performance report saved to {FINAL_REPORT_PATH}")
        except Exception as e:
            logging.error(f"Failed to save JSON performance report: {e}")

        logging.info("### Project Forge V5 Backtest Simulator: FINISHED ###")
        return report_data

    def _prepare_data(self) -> Tuple[pl.LazyFrame, pl.DataFrame]:
        # V5仕様: timeframe を必須キーとして取得
        base_cols = [
            "timestamp",
            "timeframe",  # ★追加: 行増殖バグを防ぐための必須キー
            "close",
            "atr_value",
            "atr_ratio",  # ★追加: create_proxy_labelsで計算済み・ATR Ratio判定用 (プロンプト⑯ 修正②)
            "duration_long",
            "duration_short",
        ]
        # [RULE-MODE] 効率比 / 符号つき1バー変位。create_proxy_labels が S6 に
        #   出力している列 (split_features_first_orthogonal の M2_EXACT と同じ)。
        #   signal_source="rule" のときに必須。"model" のときも診断用に読む。
        for _c in (self.config.rule_er_column, self.config.rule_d_column):
            if _c and _c not in base_cols:
                base_cols.append(_c)

        if not self.config.oof_mode:  # In-Sample Mode
            raise NotImplementedError(
                "In-Sample mode is not supported in V5. Data preparation requires OOF files."
            )

        else:  # OOF Mode (V5 Bidirectional)
            logging.info(
                f"Preparing base data (S6) and merging Bidirectional OOF (Long/Short)..."
            )

            # 1. Base Data (S6)
            base_lf = pl.scan_parquet(
                str(self.config.simulation_data_path / "**/*.parquet")
            ).select(base_cols)

            # ★追加: timeframe を必須キーとして取得
            oof_cols = [
                "timestamp",
                "timeframe",
                "prediction",
                "true_label",
                "uniqueness",
            ]

            # 2. Long OOF (timestampをUTC awareに統一)
            long_lf = (
                pl.scan_parquet(self.config.oof_long_path)
                .select(oof_cols)
                .with_columns(pl.col("timestamp").dt.replace_time_zone("UTC"))
                .rename(
                    {
                        "prediction": "m2_proba_long",
                        "true_label": "label_long",
                        "uniqueness": "uniqueness_long",
                    }
                )
            )

            # 3. Short OOF (timestampをUTC awareに統一)
            short_lf = (
                pl.scan_parquet(self.config.oof_short_path)
                .select(oof_cols)
                .with_columns(pl.col("timestamp").dt.replace_time_zone("UTC"))
                .rename(
                    {
                        "prediction": "m2_proba_short",
                        "true_label": "label_short",
                        "uniqueness": "uniqueness_short",
                    }
                )
            )

            # 4. Merge (Two-Brain)
            # ★修正: timestamp と timeframe の両方で完全一致結合 (NxM増殖を防ぐ)
            lf = (
                base_lf.join(long_lf, on=["timestamp", "timeframe"], how="left")
                .join(short_lf, on=["timestamp", "timeframe"], how="left")
                .sort(["timestamp", "timeframe"])
            )

            # V5追加: タイムアウト決済用の未来価格を asof join (forward) で事前結合
            # ★修正: 未来価格のルックアップテーブルは timestamp で一意にする
            price_lf_long = (
                base_lf.select(
                    [
                        pl.col("timestamp").alias("ts_future"),
                        pl.col("close").alias("close_future_long"),
                    ]
                )
                .unique(subset=["ts_future"], keep="last")
                .sort("ts_future")
            )

            price_lf_short = (
                base_lf.select(
                    [
                        pl.col("timestamp").alias("ts_future"),
                        pl.col("close").alias("close_future_short"),
                    ]
                )
                .unique(subset=["ts_future"], keep="last")
                .sort("ts_future")
            )

            # ★TDのハードコード解除
            # [TD-RESIM/OFFSET-FIX] S6 の timestamp は t0 = L (ラベル時刻) であり、
            #   エントリーは L + ACTION_HORIZON (= L+3分)。強制決済 (TO) の時刻は
            #   entry + TD = L + ACTION_HORIZON + TD である。
            #   旧実装は timestamp + TD (= L + TD) を引いており、TO 決済価格が
            #   ACTION_HORIZON ぶん (M3 なら 3 分) 手前の価格になっていた。
            #   TD を短縮すると TO 比率が上がるため無視できなくなる。
            # [OFFSET-FIX2] close 列はバー X の終値 = P(X + バー長) であり、
            #   ACTION_HORIZON ぶんのオフセットが【すでに入っている】。
            #   entry = L + AH、狙う決済 = L + AH + TD、close(X) = P(X + AH)
            #     ⇒ 必要なバー X = L + TD
            #   以前 (私の修正) は X = L + AH + TD としており、AH ぶん
            #   (M3 なら 3 分) 長く保有していた。TD=15 のつもりが実質 18 分。
            #   元の `timestamp + td_minutes` が正しかった。差し戻す。
            lf = lf.with_columns(
                (
                    pl.col("timestamp")
                    + pl.duration(
                        seconds=int(round(float(self.config.td_minutes_long) * 60.0))
                    )
                ).alias("ts_plus_long")
            )
            lf = lf.join_asof(
                price_lf_long,
                left_on="ts_plus_long",
                right_on="ts_future",
                strategy="forward",
            ).drop(["ts_plus_long", "ts_future"])

            lf = lf.with_columns(
                (
                    pl.col("timestamp")
                    + pl.duration(
                        seconds=int(round(float(self.config.td_minutes_short) * 60.0))
                    )
                ).alias("ts_plus_short")
            )
            lf = lf.join_asof(
                price_lf_short,
                left_on="ts_plus_short",
                right_on="ts_future",
                strategy="forward",
            ).drop(["ts_plus_short", "ts_future"])

            # Null埋め (予測がない場合は確率0として扱う)
            lf = lf.with_columns(
                [
                    pl.col("m2_proba_long").fill_null(0.0),
                    pl.col("m2_proba_short").fill_null(0.0),
                ]
            )

        # ─── 期間フィルタ (BacktestConfig.start_date / end_date) ───
        # 注: price_lf_long / price_lf_short には適用しない (TD ルックアップで
        #     期間末の翌日の close が必要なため、base_lf 全体を保持しておく)
        if self.config.start_date is not None:
            start_dt = dt.datetime.fromisoformat(self.config.start_date).replace(
                tzinfo=dt.timezone.utc
            )
            lf = lf.filter(pl.col("timestamp") >= start_dt)
            logging.info(f"期間フィルタ適用 (start): {self.config.start_date} 以降")
        if self.config.end_date is not None:
            # end_date を inclusive にするため、その日の 23:59:59.999999 まで
            end_dt = (
                dt.datetime.fromisoformat(self.config.end_date).replace(
                    tzinfo=dt.timezone.utc
                )
                + dt.timedelta(days=1)
                - dt.timedelta(microseconds=1)
            )
            lf = lf.filter(pl.col("timestamp") <= end_dt)
            logging.info(f"期間フィルタ適用 (end):   {self.config.end_date} 以前")

        logging.info("Discovering partitions...")
        partitions_df = (
            lf.select(pl.col("timestamp").dt.date().alias("date"))
            .unique()
            .collect()
            .sort("date")
        )

        if partitions_df.is_empty():
            raise ValueError(
                "No partitions found. Check data paths and OOF/S6 alignment."
            )

        logging.info(f"Found {len(partitions_df)} partitions.")
        return lf, partitions_df

    def _run_ai_predictions(self, df_chunk: pl.DataFrame) -> pl.DataFrame:
        """
        V5仕様: OOFモードを主軸とするため、In-Sample(動的推論)モードは
        Two-Brainモデルのロードが別途必要になります。
        今回はOOFデータ(結合済み)をそのまま返すパスをデフォルトとします。
        """
        logging.debug(f"Running AI processing for chunk (size: {len(df_chunk)})...")
        if not self.config.oof_mode:
            logging.error(
                "In-Sample mode requires V5 Two-Brain model architecture. Please use OOF mode (--oof) for V5."
            )
            raise NotImplementedError(
                "In-Sample mode for V5 is not fully implemented yet."
            )
        else:
            # OOF Mode: _prepare_data で既に結合・確率マッピング済み
            return df_chunk

    def _run_simulation_loop(
        self, df_chunk: pl.DataFrame
    ) -> Tuple[pl.DataFrame, pl.DataFrame]:
        trade_log_chunk = []
        equity_values_chunk = []
        current_capital = self._current_capital

        # --- 定数の初期化 ---
        DECIMAL_ZERO = Decimal("0.0")
        DECIMAL_ONE = Decimal("1.0")
        DECIMAL_MIN_CAPITAL = Decimal(str(self.config.min_capital_threshold))
        DECIMAL_VALUE_PER_PIP = Decimal(str(self.config.value_per_pip))
        DECIMAL_MIN_LOT_SIZE = Decimal(str(self.config.min_lot_size))
        DECIMAL_CONTRACT_SIZE = CONTRACT_SIZE

        # V5 固定パラメータ
        # DECIMAL_PAYOFF_RATIO = Decimal(str(self.config.payoff_ratio))
        # ▼▼▼ 以下の2行を削除またはコメントアウト ▼▼▼
        # DECIMAL_SL_MULT = Decimal(str(self.config.sl_multiplier))
        # DECIMAL_PT_MULT = Decimal(str(self.config.pt_multiplier))
        # ▼▼▼ 修正後 ▼▼▼
        # --- サーキットブレーカー用状態管理 ---
        pending_exits = []  # [(exit_time_int, direction_int, is_sl, margin_used_decimal)] ★変更
        consecutive_sl_long = 0
        consecutive_sl_short = 0
        cooldown_until_long = 0
        cooldown_until_short = 0

        # --- 証拠金トラッキング用 ---
        total_used_margin = DECIMAL_ZERO
        active_exit_times = []
        MAX_POSITIONS = self.config.max_positions

        # --- 連続SL/Loss・証拠金維持率トラッキング ---
        consec_sl_long_cur = 0  # 現在のLong連続SL数
        consec_sl_short_cur = 0  # 現在のShort連続SL数
        consec_loss_cur = 0  # 現在の全体連続負け数（SL+TO）
        max_consec_sl_long = 0  # Long最大連続SL
        max_consec_sl_short = 0  # Short最大連続SL
        max_consec_sl_total = 0  # 全体最大連続SL
        max_consec_loss_total = 0  # 全体最大連続負け（SL+TO）

        # --- DataFrameからのデータ抽出 (高速化のためリスト/Numpy配列化) ---
        timestamps_chunk = df_chunk["timestamp"].to_list()
        close_prices_chunk = df_chunk["close"].to_numpy()
        atr_values_chunk = df_chunk["atr_value"].to_numpy()
        atr_ratios_chunk = df_chunk[
            "atr_ratio"
        ].to_numpy()  # ★追加: ATR Ratio (プロンプト⑯ 修正②)
        # [baseline_ratio相対フィルター用] preload_dataで計算済みの列を読み込む
        # nullの場合（ウォームアップ期間）はフィルタースキップ用にNaNで埋める
        if "baseline_ratio" in df_chunk.columns:
            baseline_ratios_chunk = (
                df_chunk["baseline_ratio"].fill_null(float("nan")).to_numpy()
            )
        else:
            baseline_ratios_chunk = None

        # [SAR] preload_dataで計算済みのsar列を読み込む
        # min_sar_threshold=0.0 の場合列が存在しないためNoneで安全にスキップ
        if "sar" in df_chunk.columns:
            sar_chunk = df_chunk["sar"].fill_null(float("nan")).to_numpy()
        else:
            sar_chunk = None

        # [RULE-MODE] 効率比 / 符号つき1バー変位
        _use_rule = str(self.config.signal_source).lower() == "rule"
        er_chunk = (
            df_chunk[self.config.rule_er_column].to_numpy()
            if self.config.rule_er_column in df_chunk.columns
            else None
        )
        d_chunk = (
            df_chunk[self.config.rule_d_column].to_numpy()
            if self.config.rule_d_column in df_chunk.columns
            else None
        )
        if _use_rule and (er_chunk is None or d_chunk is None):
            raise KeyError(
                f"signal_source='rule' には S6 に "
                f"'{self.config.rule_er_column}' と '{self.config.rule_d_column}' が必要です。"
                f" 実際の列: {sorted(df_chunk.columns)}"
            )

        # V5 Two-Brain の確率とラベル
        p_long_chunk = df_chunk["m2_proba_long"].to_numpy()
        p_short_chunk = df_chunk["m2_proba_short"].to_numpy()
        labels_long_chunk = df_chunk["label_long"].to_numpy()
        labels_short_chunk = df_chunk["label_short"].to_numpy()

        # V5 追加: TO計算用の経過時間と未来価格
        duration_long_chunk = df_chunk["duration_long"].to_numpy()
        duration_short_chunk = df_chunk["duration_short"].to_numpy()
        close_future_long_chunk = df_chunk["close_future_long"].to_numpy()
        close_future_short_chunk = df_chunk["close_future_short"].to_numpy()

        for i in range(len(df_chunk)):
            current_timestamp = timestamps_chunk[i]

            try:
                current_timestamp_dt = current_timestamp.replace(tzinfo=dt.timezone.utc)
                current_timestamp_int = int(
                    current_timestamp_dt.timestamp() * 1_000_000
                )
            except Exception:
                current_timestamp_int = int(current_timestamp.timestamp() * 1_000_000)

            current_price_float = close_prices_chunk[i]
            atr_value_float = atr_values_chunk[i]
            atr_ratio_float = atr_ratios_chunk[
                i
            ]  # ★追加: ATR Ratio (プロンプト⑯ 修正②)

            if (
                current_price_float is None
                or not np.isfinite(current_price_float)
                or current_price_float <= 0
            ):
                equity_values_chunk.append(current_capital)
                continue

            current_price_decimal = Decimal(str(current_price_float))

            # 破産チェック
            if current_capital < DECIMAL_MIN_CAPITAL:
                equity_values_chunk.append(DECIMAL_ZERO)
                continue

            # =========================================================
            # 完了したポジションの精算（SLカウントと証拠金の解放）
            # =========================================================
            finished_positions = [
                p for p in pending_exits if p[0] <= current_timestamp_int
            ]
            pending_exits = [p for p in pending_exits if p[0] > current_timestamp_int]

            for exit_time, direction, is_sl, margin_used, log_entry in sorted(
                finished_positions, key=lambda x: x[0]
            ):
                # 証拠金の解放
                total_used_margin -= margin_used
                if total_used_margin < DECIMAL_ZERO:
                    total_used_margin = DECIMAL_ZERO

                if direction == 1:
                    if is_sl:
                        consecutive_sl_long += 1
                        consec_sl_long_cur += 1
                        consec_sl_short_cur = 0
                        consec_loss_cur += 1
                        if consecutive_sl_long >= self.config.max_consecutive_sl:
                            cooldown_until_long = exit_time + int(
                                self.config.cooldown_minutes_after_sl * 60 * 1_000_000
                            )
                            consecutive_sl_long = 0
                            self.cb_cooldown_long += 1
                    else:  # PT（勝ち）
                        consecutive_sl_long = 0
                        consec_sl_long_cur = 0
                        consec_sl_short_cur = 0
                        consec_loss_cur = 0
                else:
                    if is_sl:
                        consecutive_sl_short += 1
                        consec_sl_short_cur += 1
                        consec_sl_long_cur = 0
                        consec_loss_cur += 1
                        if consecutive_sl_short >= self.config.max_consecutive_sl:
                            cooldown_until_short = exit_time + int(
                                self.config.cooldown_minutes_after_sl * 60 * 1_000_000
                            )
                            consecutive_sl_short = 0
                            self.cb_cooldown_short += 1
                    else:  # PT（勝ち）
                        consecutive_sl_short = 0
                        consec_sl_long_cur = 0
                        consec_sl_short_cur = 0
                        consec_loss_cur = 0

                # 最大値更新
                max_consec_sl_long = max(max_consec_sl_long, consec_sl_long_cur)
                max_consec_sl_short = max(max_consec_sl_short, consec_sl_short_cur)
                max_consec_sl_total = max(
                    max_consec_sl_total, consec_sl_long_cur + consec_sl_short_cur
                )
                max_consec_loss_total = max(max_consec_loss_total, consec_loss_cur)

                # 決済確定後の連続SL値をlog_entryに書き込んでからトレードログに追記
                log_entry["csl_L"] = consec_sl_long_cur
                log_entry["csl_S"] = consec_sl_short_cur
                log_entry["closs"] = consec_loss_cur
                trade_log_chunk.append(log_entry)

            # 決済時刻を過ぎたポジションをクリア
            active_exit_times = [
                t for t in active_exit_times if t > current_timestamp_int
            ]

            # =========================================================
            # リアルタイム証拠金維持率のチェック & 強制ロスカット(Stop Out)
            # =========================================================
            current_margin_level = Decimal("inf")
            if total_used_margin > DECIMAL_ZERO:
                current_margin_level = (current_capital / total_used_margin) * Decimal(
                    "100.0"
                )
                if current_margin_level < self.min_margin_level_pct:  # ★ self. をつける
                    self.min_margin_level_pct = current_margin_level  # ★ self. をつける

                # ストップアウト（強制ロスカット）発動
                if current_margin_level <= Decimal(str(self.config.stop_out_percent)):
                    self.stop_out_count += 1  # ★ self. をつける
                    # 簡易処理: 証拠金の大部分を失い、全ポジションを強制決済する
                    current_capital = total_used_margin * (
                        Decimal(str(self.config.stop_out_percent)) / Decimal("100.0")
                    )
                    total_used_margin = DECIMAL_ZERO
                    pending_exits.clear()
                    active_exit_times.clear()
                    continue

            # =========================================================
            # V5改修: Delta (差分) フィルター & 同時発注禁止ロジック
            # =========================================================
            p_l = p_long_chunk[i]
            p_s = p_short_chunk[i]

            should_trade_long = False
            should_trade_short = False

            # LongとShortの確率の差分（Delta）を計算
            delta = abs(p_l - p_s)

            if _use_rule:
                # ==================================================
                # [RULE-MODE] 脳を一切使わない。効率比 ER と符号つき変位 d だけ。
                #   Residual_Drift_Harvester_Theory §13.5〜13.6
                #     ER 高い(直近すでに走りきった) + そこそこ大きい 1 本
                #       → その後【逆行】する (行き過ぎの解消)
                #     モデル非依存・6年すべて同符号
                #   発火: er_min <= ER (< er_max) かつ d_abs_min <= |d| < d_abs_max
                #   方向: reverse なら −sign(d) / follow なら +sign(d)
                #   m2_proba / m2_delta はこのモードでは一切参照しない。
                #   sign(d) で方向が一意に決まるので両建ては構造的に発生しない。
                # ==================================================
                _er = er_chunk[i]
                _d = d_chunk[i]
                _ok = (
                    np.isfinite(_er)
                    and np.isfinite(_d)
                    and _er >= self.config.rule_er_min
                    and (
                        self.config.rule_er_max <= 0.0 or _er < self.config.rule_er_max
                    )
                    and self.config.rule_d_abs_min
                    <= abs(_d)
                    < self.config.rule_d_abs_max
                )
                if _ok and _d != 0.0:
                    _sgn = 1 if _d > 0 else -1
                    if str(self.config.rule_direction).lower() == "reverse":
                        _sgn = -_sgn
                    if _sgn > 0:
                        should_trade_long = True
                    else:
                        should_trade_short = True
                    self.rule_fire_count += 1

            elif self.config.prevent_simultaneous_orders:
                # ---- delta で勝った【片方だけ】を発注 (従来の挙動) ----
                # 条件1: 差分(Delta)が閾値以上開いていること
                # 条件2: 勝つ方の絶対確率自体も m2_proba_threshold を超えていること
                if delta >= self.config.m2_delta_threshold:
                    if p_l > p_s and p_l > self.config.m2_proba_threshold:
                        should_trade_long = True
                    elif p_s > p_l and p_s > self.config.m2_proba_threshold:
                        should_trade_short = True
            else:
                # ---- [FLAG-FIX] 同時発注を許可: long/short を独立に評価 ----
                #   旧実装はこの分岐が無く if/elif 直書きだったため、
                #   prevent_simultaneous_orders=False にしても
                #   --allow-simultaneous を付けても同時発注は解除されなかった。
                if p_l > self.config.m2_proba_threshold:
                    should_trade_long = True
                if p_s > self.config.m2_proba_threshold:
                    should_trade_short = True
                if should_trade_long and should_trade_short:
                    self.cb_simultaneous_taken += 1

            # 両方Falseのまま（発注なし）の回数をカウント
            if not should_trade_long and not should_trade_short:
                self.cb_simultaneous_prevented += 1

            # =========================================================
            # V5 両建て評価ロジック (Long -> Short の順で独立評価)
            # =========================================================
            directions_to_evaluate = [
                (
                    1,
                    p_long_chunk[i],
                    labels_long_chunk[i],
                    duration_long_chunk[i],
                    close_future_long_chunk[i],
                    should_trade_long,
                    Decimal(str(self.config.pt_multiplier_long)),  # ★追加
                    Decimal(str(self.config.sl_multiplier_long)),  # ★追加
                ),  # Long評価
                (
                    -1,
                    p_short_chunk[i],
                    labels_short_chunk[i],
                    duration_short_chunk[i],
                    close_future_short_chunk[i],
                    should_trade_short,
                    Decimal(str(self.config.pt_multiplier_short)),  # ★追加
                    Decimal(str(self.config.sl_multiplier_short)),  # ★追加
                ),  # Short評価
            ]

            traded_in_this_step = False

            # ▼▼▼ unpackedする変数を増やす ▼▼▼
            for (
                direction_int,
                p_float,
                actual_label,
                duration_float,
                close_future_float,
                base_should_trade,
                current_pt_mult,  # ★追加
                current_sl_mult,  # ★追加
            ) in directions_to_evaluate:
                # NoneやNaNの回避
                if p_float is None or not np.isfinite(p_float):
                    continue

                # ポジション数上限チェック (Long/Short それぞれ1枠消費)
                if len(active_exit_times) >= MAX_POSITIONS:
                    continue

                # =========================================================
                # エントリー判定とクールダウン判定
                # =========================================================
                should_trade = base_should_trade

                if direction_int == 1 and current_timestamp_int < cooldown_until_long:
                    should_trade = False
                elif (
                    direction_int == -1 and current_timestamp_int < cooldown_until_short
                ):
                    should_trade = False

                if should_trade:
                    if (
                        atr_ratio_float is None
                        or not np.isfinite(atr_ratio_float)
                        or atr_ratio_float
                        < self.config.min_atr_threshold  # ★修正: ATR Ratio と比較 (プロンプト⑯ 修正②)
                    ):
                        continue

                    # [atr_value 帯] バリア幅そのものの下限/上限 (USD)。0.0 = 無効。
                    #   スプレッドが USD 固定である以上、収益性を決めるのは
                    #   atr_ratio(相対) ではなく atr_value(絶対)。
                    if (
                        self.config.min_atr_value > 0.0
                        and atr_value_float < self.config.min_atr_value
                    ):
                        continue
                    if (
                        self.config.max_atr_value > 0.0
                        and atr_value_float >= self.config.max_atr_value
                    ):
                        continue

                    # [ATR Ratio 帯の上限] 0.0 = 無効。
                    #   min_atr_threshold と組み合わせて「この帯だけ撃つ」を作れる。
                    #   例: min=0.0 / max=0.8 → ATR Ratio Band の "<0.5" と "0.5-0.8" だけ
                    if (
                        self.config.max_atr_threshold > 0.0
                        and atr_ratio_float >= self.config.max_atr_threshold
                    ):
                        continue

                    # [baseline_ATR床フィルター] 前日24h ATR平均の絶対下限チェック
                    # baseline_atr = atr_value / atr_ratio (= 直近480本のATR平均)
                    # min_baseline_atr=0.0 のとき無効 (後方互換)
                    if (
                        self.config.min_baseline_atr > 0.0
                        or self.config.max_baseline_atr > 0.0
                    ):
                        baseline_atr_float = atr_value_float / (atr_ratio_float + 1e-10)
                        if (
                            self.config.min_baseline_atr > 0.0
                            and baseline_atr_float < self.config.min_baseline_atr
                        ):
                            continue
                        # [baseline_ATR 上限] 0.0 = 無効
                        if (
                            self.config.max_baseline_atr > 0.0
                            and baseline_atr_float >= self.config.max_baseline_atr
                        ):
                            continue

                    # [baseline_ratio相対フィルター] 昨日ボラ / 過去N日ボラ の比率チェック
                    # min_baseline_ratio=0.0 のとき無効 (後方互換)
                    if (
                        self.config.min_baseline_ratio > 0.0
                        and baseline_ratios_chunk is not None
                    ):
                        br = baseline_ratios_chunk[i]
                        if not np.isfinite(br) or br < self.config.min_baseline_ratio:
                            continue

                    # [SARフィルター] 日中季節性調整済み相対ATR
                    # SAR = 現在ATR / 過去D日の同時刻ATR平均
                    # min_sar_threshold=0.0 のとき無効 (後方互換)
                    if self.config.min_sar_threshold > 0.0 and sar_chunk is not None:
                        sar_val = sar_chunk[i]
                        if (
                            not np.isfinite(sar_val)
                            or sar_val < self.config.min_sar_threshold
                        ):
                            continue

                    # [FIX-4] Auto Lot 計算を extreme_risk_engine.calculate_auto_lot() と統一
                    # 旧: ハードコード乗数方式 (0.25倍/0.5倍) → 新: 証拠金上限数式

                    # ▼▼▼ 修正: 固定比率(Fixed Risk)と固定複利(Auto Lot)の分岐 ▼▼▼
                    if self.config.use_fixed_lot:
                        # [FIXED-LOT] エッジ測定モード。常に同じロット。
                        #   複利も資産減少も効かないので、破産で期間が
                        #   途中で切れることがなく「1トレードあたりの期待値」を
                        #   全期間で観測できる。資金管理の評価には使えない。
                        base_lot = Decimal(str(self.config.fixed_lot_size))
                    elif self.config.use_fixed_risk:
                        risk_pct_dec = Decimal(str(self.config.fixed_risk_percent))
                        max_loss_amount = current_capital * risk_pct_dec
                        # ▼▼ DECIMAL_SL_MULT を current_sl_mult に変更 ▼▼
                        sl_price_distance = (
                            Decimal(str(atr_value_float)) * current_sl_mult
                        )

                        if sl_price_distance > DECIMAL_ZERO:
                            base_lot = max_loss_amount / (
                                sl_price_distance * DECIMAL_CONTRACT_SIZE
                            )
                        else:
                            base_lot = DECIMAL_ZERO
                    else:
                        # --- 従来の固定複利（Auto Lot）---
                        base_capital_dec = Decimal(
                            str(self.config.auto_lot_base_capital)
                        )
                        size_per_base_dec = Decimal(
                            str(self.config.auto_lot_size_per_base)
                        )
                        base_lot = (
                            current_capital / base_capital_dec
                        ) * size_per_base_dec
                    # ▲▲▲ ここまで修正 ▲▲▲

                    # Step2: レバレッジに基づく証拠金上限ロット
                    effective_leverage_decimal = self._get_effective_leverage(
                        current_capital
                    )
                    max_lot_margin = (current_capital * effective_leverage_decimal) / (
                        current_price_decimal * DECIMAL_CONTRACT_SIZE
                    )

                    # Step3: 基本ロット vs 証拠金上限 の小さい方、絶対上限 200 でキャップ
                    raw_lot_size = min(base_lot, max_lot_margin, Decimal("200.0"))

                    # Step4: 0.01刻み切り捨て
                    final_lot_size_decimal = Decimal(int(raw_lot_size * 100)) / Decimal(
                        "100"
                    )

                    # ▼▼▼ 追加: ニート化防止！ 最低ロット数の保証 ▼▼▼
                    final_lot_size_decimal = max(
                        final_lot_size_decimal, DECIMAL_MIN_LOT_SIZE
                    )
                    # ▲▲▲ ここまで追加 ▲▲▲

                    if final_lot_size_decimal >= DECIMAL_MIN_LOT_SIZE:
                        margin_required_decimal = (
                            current_price_decimal
                            * final_lot_size_decimal
                            * DECIMAL_CONTRACT_SIZE
                        ) / effective_leverage_decimal
                        spread_pips_decimal = Decimal(str(self.config.spread_pips))
                        spread_cost_decimal = (
                            final_lot_size_decimal
                            * spread_pips_decimal
                            * DECIMAL_VALUE_PER_PIP
                        )

                        # 必要証拠金とスプレッドコストが現在の資金を上回る場合は安全に弾く
                        if (
                            margin_required_decimal + spread_cost_decimal
                            > current_capital
                        ):
                            continue

                        # ▼▼▼ 新規追加: マージンコール（証拠金維持率）チェック ▼▼▼
                        new_total_margin = total_used_margin + margin_required_decimal
                        new_margin_level = (
                            current_capital / new_total_margin
                        ) * Decimal("100.0")

                        if new_margin_level < Decimal(
                            str(self.config.margin_call_percent)
                        ):
                            continue  # 維持率100%を下回るような過剰なエントリーは拒否

                        total_used_margin = new_total_margin
                        # ▲▲▲ ここまで追加 ▲▲▲

                        capital_before_pnl = current_capital - spread_cost_decimal
                        pnl = DECIMAL_ZERO
                        is_sl_hit = False  # V5: SL判定フラグ
                        valid_label = (
                            actual_label
                            if (actual_label is not None and np.isfinite(actual_label))
                            else 0
                        )
                        duration_val = (
                            duration_float
                            if (
                                duration_float is not None
                                and np.isfinite(duration_float)
                            )
                            else 0.0
                        )

                        # [TD-RESIM] エントリー起点の経過分に換算してから TD 判定。
                        #   duration は t0 = L 起点なので ACTION_HORIZON_MIN を引く。
                        #   旧実装は PT 分岐 (valid_label==1) に TD 判定が無く、
                        #   TD を短縮しても「新TD より後に付いた PT」が満額の勝ちの
                        #   まま計上されていた (= TD 短縮が勝ち側だけに甘くなる主因)。
                        _ah_m = float(ACTION_HORIZON_MIN)
                        _td_l = float(self.config.td_minutes_long)
                        _td_s = float(self.config.td_minutes_short)
                        _elapsed_from_entry = duration_val - _ah_m
                        _within_td_long = _elapsed_from_entry < (_td_l - 1e-9)
                        _within_td_short = _elapsed_from_entry < (_td_s - 1e-9)

                        # [TD-RESIM/HOLD] 実際の保有時間 (エントリー起点)。
                        #   TD 打ち切りが起きた玉はここで頭打ちになる。
                        #   ★旧実装はログの "TD" 列にラベル生の duration をそのまま
                        #     入れていたため、TD を絞っても Avg TD が縮まず
                        #     「TD が効いていない」ように見えていた。実際には
                        #     PnL 側では効いており、見た目だけが生値だった。
                        # [TIME-EXIT] バリアを使わないモード。
                        #   PT/SL 判定を丸ごと無効化し、必ず TD で成行決済する。
                        #   測定が測った「t 分後の変位」をそのまま損益にする。
                        if not self.config.use_barriers:
                            _within_td_long = False
                            _within_td_short = False

                        _td_use = _td_l if direction_int == 1 else _td_s
                        if not self.config.use_barriers:
                            # [TIME-EXIT] バリアを見ないので保有は常に TD ちょうど。
                            #   旧: min(elapsed, TD) だとラベルの duration に
                            #   引きずられて Avg Hold が TD より短く表示されていた
                            #   (損益は close_future(entry+TD) で正しかった)。
                            _hold_from_entry = _td_use
                        else:
                            _hold_from_entry = min(
                                max(_elapsed_from_entry, 0.0), _td_use
                            )
                        # ポジション占有時間 (L 起点)。証拠金の解放時刻に使う。
                        _eff_duration_from_L = _ah_m + _hold_from_entry
                        _exit_kind = "TO"  # 各分岐で上書き

                        # V5 追加: Exit Price ベースの厳密な PnL 計算 (スプレッド二重取り回避)
                        exit_price_decimal = current_price_decimal
                        if direction_int == 1:  # Long
                            if valid_label == 1 and _within_td_long:
                                _exit_kind = "PT"
                                exit_price_decimal = current_price_decimal + (
                                    Decimal(str(atr_value_float)) * current_pt_mult
                                )
                            elif valid_label == 0 and _within_td_long:
                                _exit_kind = "SL"
                                exit_price_decimal = current_price_decimal - (
                                    Decimal(str(atr_value_float)) * current_sl_mult
                                )
                                is_sl_hit = True
                            else:  # タイムアウト (TD 到達 or TD 短縮による打ち切り)
                                future_p = (
                                    close_future_float
                                    if (
                                        close_future_float is not None
                                        and np.isfinite(close_future_float)
                                    )
                                    else current_price_float
                                )
                                exit_price_decimal = Decimal(str(future_p))
                            pnl = (
                                (exit_price_decimal - current_price_decimal)
                                * final_lot_size_decimal
                                * DECIMAL_CONTRACT_SIZE
                            )

                        else:  # Short
                            if valid_label == 1 and _within_td_short:
                                _exit_kind = "PT"
                                exit_price_decimal = current_price_decimal - (
                                    Decimal(str(atr_value_float)) * current_pt_mult
                                )
                            elif valid_label == 0 and _within_td_short:
                                _exit_kind = "SL"
                                exit_price_decimal = current_price_decimal + (
                                    Decimal(str(atr_value_float)) * current_sl_mult
                                )
                                is_sl_hit = True
                            else:  # タイムアウト (TD 到達 or TD 短縮による打ち切り)
                                future_p = (
                                    close_future_float
                                    if (
                                        close_future_float is not None
                                        and np.isfinite(close_future_float)
                                    )
                                    else current_price_float
                                )
                                exit_price_decimal = Decimal(str(future_p))
                            pnl = (
                                (current_price_decimal - exit_price_decimal)
                                * final_lot_size_decimal
                                * DECIMAL_CONTRACT_SIZE
                            )

                        next_capital = capital_before_pnl + pnl
                        current_capital = (
                            next_capital if next_capital.is_finite() else DECIMAL_ZERO
                        )

                        # ▼▼▼ 新規追加: HWMの更新と現在のドローダウン率計算 ▼▼▼
                        self.high_water_mark = max(
                            self.high_water_mark, current_capital
                        )
                        current_dd_pct = (
                            (
                                (current_capital - self.high_water_mark)
                                / self.high_water_mark
                                * Decimal("100.0")
                            )
                            if self.high_water_mark > DECIMAL_ZERO
                            else DECIMAL_ZERO
                        )
                        # ▲▲▲ ここまで追加 ▲▲▲

                        # ▼▼▼ 修正前 ▼▼▼
                        # if duration_float is not None and np.isfinite(duration_float):
                        #     new_exit_time = current_timestamp_int + int(duration_float * 60 * 1_000_000)
                        #     active_exit_times.append(new_exit_time)
                        #     pending_exits.append((new_exit_time, direction_int, is_sl_hit))

                        # ▼▼▼ 修正後: log_entryをpending_exitsに同梱し決済時にcsl/clossを書いてからログ追記 ▼▼▼
                        traded_in_this_step = True

                        current_active_longs = sum(
                            1 for p in pending_exits if p[1] == 1
                        )
                        current_active_shorts = sum(
                            1 for p in pending_exits if p[1] == -1
                        )

                        _mg_lv = float(
                            (current_capital / total_used_margin * Decimal("100.0"))
                            if total_used_margin > DECIMAL_ZERO
                            else Decimal("9999.0")
                        )
                        # [SPREAD-FIX] 資本更新は capital = capital − spread + pnl だが、
                        #   従来ログの "pnl" は spread 控除【前】の gross だった。
                        #   そのため PF・勝率・平均利益/損失・Exit内訳がすべて
                        #   グロス評価になり、balance と食い違っていた
                        #   (実測: Exit内訳合計 +274.67 に対し Net Profit −1000.00)。
                        #   "pnl" を net に統一し、gross は別列で保持する。
                        _pnl_net = pnl - spread_cost_decimal
                        _log_entry = {
                            "timestamp": current_timestamp,
                            "pnl": _pnl_net,
                            "pnl_gross": pnl,
                            # [DISP] 実現変位 (ATR単位)。測定 measure_mu_time_profile の
                            #   E[ΔX(t)×方向] と直接比較できる形。符号は測定と逆
                            #   (測定の −0.2508 = こちらの +0.2508)。
                            # [VERIFY] 選抜条件そのものを記録する。
                            #   測定と同じバーを拾えているかを後から直接検算できる。
                            "er": (
                                float(er_chunk[i])
                                if er_chunk is not None and np.isfinite(er_chunk[i])
                                else float("nan")
                            ),
                            "d_atr": (
                                float(d_chunk[i])
                                if d_chunk is not None and np.isfinite(d_chunk[i])
                                else float("nan")
                            ),
                            "disp_atr": (
                                float(pnl)
                                / (
                                    float(final_lot_size_decimal)
                                    * 100.0
                                    * float(atr_value_float)
                                )
                                if (
                                    final_lot_size_decimal > DECIMAL_ZERO
                                    and atr_value_float
                                    and np.isfinite(atr_value_float)
                                    and atr_value_float > 0
                                )
                                else float("nan")
                            ),
                            "balance": current_capital,
                            "m2_proba": float(p_float),
                            "direction": int(direction_int),
                            "label": int(valid_label),
                            "lot_size": float(final_lot_size_decimal),
                            "atr_value": float(atr_value_float),
                            "atr_ratio": float(atr_ratio_float),
                            "leverage": float(effective_leverage_decimal),
                            "margin": margin_required_decimal,
                            "spread": spread_cost_decimal,
                            "close_price": current_price_decimal,
                            "aL": int(current_active_longs),
                            "aS": int(current_active_shorts),
                            "TD": float(duration_val),
                            # [TD-RESIM/HOLD] 実保有時間 (エントリー起点、TD 打ち切り反映)
                            "hold": float(_hold_from_entry),
                            # 決済種別: PT / SL / TO (TD 打ち切りを含む)
                            "exit": str(_exit_kind),
                            "DD(%)": float(current_dd_pct),
                            "mg_lv%": _mg_lv,
                            "csl_L": 0,  # 決済時に上書き
                            "csl_S": 0,  # 決済時に上書き
                            "closs": 0,  # 決済時に上書き
                        }

                        # [TD-RESIM] 証拠金の解放時刻も打ち切り後の実決済時刻にする。
                        #   旧実装はラベル生 duration で解放していたため、TD を絞っても
                        #   ポジションが本来より長く枠を占有し、同時保有数・証拠金・
                        #   max_positions の判定が実態とズレていた。
                        if duration_float is not None and np.isfinite(duration_float):
                            new_exit_time = current_timestamp_int + int(
                                _eff_duration_from_L * 60 * 1_000_000
                            )
                            active_exit_times.append(new_exit_time)
                            pending_exits.append(
                                (
                                    new_exit_time,
                                    direction_int,
                                    is_sl_hit,
                                    margin_required_decimal,
                                    _log_entry,  # ← log_entryを同梱
                                )
                            )
                        else:
                            # duration不明の場合はcsl=0のままログ追記
                            trade_log_chunk.append(_log_entry)

            # 資本の時系列記録
            equity_values_chunk.append(current_capital)

        self._current_capital = current_capital
        # チャンク内の最大値をselfに反映（複数チャンクをまたいで最大値を保持）
        self.max_consec_sl_long = max(self.max_consec_sl_long, max_consec_sl_long)
        self.max_consec_sl_short = max(self.max_consec_sl_short, max_consec_sl_short)
        self.max_consec_sl_total = max(self.max_consec_sl_total, max_consec_sl_total)
        self.max_consec_loss_total = max(
            self.max_consec_loss_total, max_consec_loss_total
        )

        results_chunk_df = pl.DataFrame(
            {
                "timestamp": timestamps_chunk,
                "equity": pl.Series("equity", equity_values_chunk, dtype=pl.Object),
            }
        )

        # V5仕様のトレードログスキーマ
        trade_log_schema = {
            "timestamp": pl.Datetime,
            "pnl": pl.Object,  # スプレッド控除後 (balance と整合)
            "pnl_gross": pl.Object,  # スプレッド控除前
            "er": pl.Float64,  # 効率比 (選抜条件)
            "d_atr": pl.Float64,  # 符号つき1バー変位 (選抜条件)
            "disp_atr": pl.Float64,  # 実現変位 (ATR単位)
            "balance": pl.Object,
            "m2_proba": pl.Float64,
            "direction": pl.Int8,
            "label": pl.Int64,
            "lot_size": pl.Float64,
            "atr_value": pl.Float64,
            "atr_ratio": pl.Float64,
            "leverage": pl.Float32,
            "margin": pl.Object,
            "spread": pl.Object,
            "close_price": pl.Object,
            "aL": pl.Int32,
            "aS": pl.Int32,
            "TD": pl.Float64,
            "hold": pl.Float64,  # 実保有分(エントリー起点・TD打ち切り反映)
            "exit": pl.Utf8,  # PT / SL / TO
            "DD(%)": pl.Float64,
            "mg_lv%": pl.Float64,  # 証拠金維持率(%)
            "csl_L": pl.Int32,  # Long連続SL（決済後）
            "csl_S": pl.Int32,  # Short連続SL（決済後）
            "closs": pl.Int32,  # 全体連続負け（SL+TO）
        }

        if trade_log_chunk:
            trade_log_data = {
                key: [d.get(key) for d in trade_log_chunk]
                for key in trade_log_schema.keys()
            }
            series_dict = {
                key: pl.Series(key, trade_log_data.get(key), dtype=dtype)
                for key, dtype in trade_log_schema.items()
            }
            trade_log_chunk_df = pl.DataFrame(series_dict)
        else:
            trade_log_chunk_df = pl.DataFrame(schema=trade_log_schema)

        return results_chunk_df, trade_log_chunk_df

    def _analyze_and_report(self, results_df: pl.DataFrame, trade_log: pl.DataFrame):
        logging.info("Analyzing results and generating V5 report...")
        DECIMAL_ONE = Decimal("1.0")
        DECIMAL_ZERO = Decimal("0.0")
        DECIMAL_HUNDRED = Decimal("100.0")

        if results_df.is_empty():
            logging.error("No simulation results to analyze.")
            initial_capital = Decimal(str(self.config.initial_capital))
            final_capital = initial_capital
            total_return = DECIMAL_ZERO
            sharpe_ratio = 0.0
            sortino_ratio = 0.0
            max_drawdown = 0.0
            data_period_start = "N/A"
            data_period_end = "N/A"
            drawdown = pl.Series(dtype=pl.Float64)
        else:
            initial_capital = Decimal(str(self.config.initial_capital))
            final_capital_raw = results_df["equity"][-1]
            final_capital = (
                final_capital_raw
                if final_capital_raw is not None and final_capital_raw.is_finite()
                else DECIMAL_ZERO
            )
            total_return = (
                (final_capital / initial_capital - DECIMAL_ONE)
                if initial_capital > DECIMAL_ZERO and initial_capital.is_finite()
                else DECIMAL_ZERO
            )

            daily_equity = (
                results_df.group_by(pl.col("timestamp").dt.date().alias("date"))
                .agg(pl.last("equity"))
                .sort("date")
            )
            daily_equity_list = daily_equity["equity"].to_list()
            daily_returns_float = []

            if len(daily_equity_list) > 1:
                for i in range(1, len(daily_equity_list)):
                    prev = daily_equity_list[i - 1]
                    curr = daily_equity_list[i]
                    if (
                        prev is not None
                        and curr is not None
                        and prev.is_finite()
                        and curr.is_finite()
                        and prev > DECIMAL_ZERO
                    ):
                        daily_ret_decimal = (curr / prev) - DECIMAL_ONE
                        daily_ret_float = (
                            float(daily_ret_decimal)
                            if daily_ret_decimal.is_finite()
                            else np.nan
                        )
                        daily_returns_float.append(daily_ret_float)
                    else:
                        daily_returns_float.append(np.nan)

            daily_returns = pl.Series(
                "daily_returns", daily_returns_float, dtype=pl.Float64
            ).drop_nans()
            num_trading_days = len(daily_returns)
            sharpe_ratio = 0.0
            sortino_ratio = 0.0

            if num_trading_days > 1:
                mean_daily_return = daily_returns.mean()
                std_daily_return = daily_returns.std()
                if (
                    mean_daily_return is not None
                    and std_daily_return is not None
                    and std_daily_return > 0
                ):
                    sharpe_ratio = (mean_daily_return / std_daily_return) * np.sqrt(252)
                    negative_returns = daily_returns.filter(daily_returns < 0)
                    downside_std = negative_returns.std()
                    if downside_std is not None and downside_std > 0:
                        sortino_ratio = (mean_daily_return / downside_std) * np.sqrt(
                            252
                        )

            logging.info("  -> Calculating drawdown (Polars optimized)...")
            initial_equity_series = pl.Series(
                "equity", [initial_capital], dtype=pl.Object
            )

            equity_series_decimal = pl.concat(
                [initial_equity_series, results_df["equity"]]
            )

            equity_series_float = equity_series_decimal.map_elements(
                lambda d: float(d) if d is not None and d.is_finite() else np.nan,
                return_dtype=pl.Float64,
            ).fill_null(strategy="forward")

            if (
                equity_series_float.is_empty()
                or equity_series_float.null_count() == equity_series_float.len()
            ):
                logging.warning(
                    "  -> Drawdown calculation skipped (no valid equity data)."
                )
                drawdown = pl.Series(dtype=pl.Float64)
                max_drawdown = 0.0
            else:
                rolling_max_series = equity_series_float.cum_max().alias("rolling_max")
                drawdown_series_pct = (
                    ((equity_series_float - rolling_max_series) / rolling_max_series)
                    .fill_nan(0.0)
                    .alias("drawdown")
                )
                drawdown = drawdown_series_pct.slice(1)
                max_drawdown_raw = drawdown_series_pct.min()
                max_drawdown = (
                    max_drawdown_raw
                    if max_drawdown_raw is not None and np.isfinite(max_drawdown_raw)
                    else 0.0
                )
            logging.info("  -> Drawdown calculation complete.")

            data_period_start = str(results_df["timestamp"].min())
            data_period_end = str(results_df["timestamp"].max())

        total_trades = len(trade_log)
        win_rate = 0.0
        avg_profit = 0.0
        avg_loss = 0.0
        profit_factor = 0.0
        avg_bet_fraction = 0.0

        if total_trades > 0:
            pnl_list_decimal = trade_log["pnl"].to_list()
            pnl_list_float = [
                float(d) if d is not None and d.is_finite() else np.nan
                for d in pnl_list_decimal
            ]
            pnl_series_float = pl.Series(
                "pnl_float", pnl_list_float, dtype=pl.Float64
            ).drop_nans()

            # V5 仕様: 1=Win, 0=Lose
            # [WINRATE-FIX] 旧実装は label(=PTラベルか) で勝敗を数えていた。
            #   実損益ではないため、
            #     ・TD 打ち切りで label=1 が損失決済になるケース
            #     ・use_barriers=False (時間決済) で label が無意味になるケース
            #   のいずれでも Win Rate / Average Profit / Average Loss が壊れる。
            #   実測でも Direction 別勝率の加重平均と 64 件食い違っていた。
            #   スプレッド控除後の実損益 (pnl) で数える。
            _pnl_f = [
                float(x) if x is not None else 0.0 for x in trade_log["pnl"].to_list()
            ]
            _win_mask = pl.Series("w", [v > 0 for v in _pnl_f])
            winning_trades = trade_log.filter(_win_mask)
            losing_trades = trade_log.filter(~_win_mask)

            num_winning_trades = len(winning_trades)
            num_losing_trades = len(losing_trades)
            win_rate = num_winning_trades / total_trades if total_trades > 0 else 0.0

            winning_pnl_list = winning_trades["pnl"].to_list()
            winning_pnl_float = [
                float(d) if d is not None and d.is_finite() else np.nan
                for d in winning_pnl_list
            ]
            winning_pnl_series = pl.Series("win_pnl", winning_pnl_float).drop_nans()

            losing_pnl_list = losing_trades["pnl"].to_list()
            losing_pnl_float = [
                float(d) if d is not None and d.is_finite() else np.nan
                for d in losing_pnl_list
            ]
            losing_pnl_series = pl.Series("lose_pnl", losing_pnl_float).drop_nans()

            avg_profit = (
                winning_pnl_series.mean() if not winning_pnl_series.is_empty() else 0.0
            )
            avg_loss = (
                losing_pnl_series.mean() if not losing_pnl_series.is_empty() else 0.0
            )
            total_profit = winning_pnl_series.sum()
            total_loss = losing_pnl_series.sum()

            if total_loss is not None and total_loss != 0:
                profit_factor = (
                    abs(total_profit / total_loss) if total_profit is not None else 0.0
                )
            elif total_profit is not None and total_profit > 0:
                profit_factor = float("inf")
            else:
                profit_factor = 0.0

        # ▼▼▼ 修正: レポートのStrategy名に資金管理方式を反映 ▼▼▼
        strategy_str = "V5 Two-Brain "
        if self.config.use_fixed_risk:
            strategy_str += f"Fixed Risk ({self.config.fixed_risk_percent * 100:.1f}%)"
        else:
            strategy_str += f"Auto Lot (Base: {self.config.auto_lot_base_capital}, Size: {self.config.auto_lot_size_per_base})"
        # OOFパスからM1/M2モードを判定してStrategy文字列に反映
        _oof_label = "M1" if "m1_oof" in str(self.config.oof_long_path) else "M2"
        strategy_str += f", {_oof_label}: {self.config.m2_proba_threshold}, L(PT{self.config.pt_multiplier_long}/SL{self.config.sl_multiplier_long}), S(PT{self.config.pt_multiplier_short}/SL{self.config.sl_multiplier_short})"

        report_data = {
            "strategy": strategy_str,
            "initial_capital": float(initial_capital),
            # ▲▲▲ ここまで修正 ▲▲▲
            "final_capital": float(final_capital),
            "total_return_pct": float(total_return * DECIMAL_HUNDRED),
            "sharpe_ratio_annual": sharpe_ratio if np.isfinite(sharpe_ratio) else None,
            "sortino_ratio_annual": sortino_ratio
            if np.isfinite(sortino_ratio)
            else None,
            "max_drawdown_pct": max_drawdown * 100
            if np.isfinite(max_drawdown)
            else None,
            "total_trades": total_trades,
            "win_rate_pct": win_rate * 100,
            "average_profit": avg_profit if np.isfinite(avg_profit) else None,
            "average_loss": avg_loss if np.isfinite(avg_loss) else None,
            "profit_factor": profit_factor if np.isfinite(profit_factor) else None,
            "average_effective_bet_fraction_pct": avg_bet_fraction * 100
            if np.isfinite(avg_bet_fraction)
            else None,
            "data_period_start": data_period_start,
            "data_period_end": data_period_end,
            "cb_simultaneous_prevented": self.cb_simultaneous_prevented,  # ★追加
            "cb_cooldown_long": self.cb_cooldown_long,  # ★追加
            "cb_cooldown_short": self.cb_cooldown_short,  # ★追加
        }

        print("\n" + "=" * 50)
        print("    Project Forge V5 Backtest Performance Report")
        print("=" * 50)
        print(f" Strategy:             {report_data.get('strategy', 'N/A')}")
        print(f" Initial Capital:      {report_data.get('initial_capital', 0.0):,.2f}")
        print(f" Final Capital:        {report_data.get('final_capital', 0.0):,.2f}")
        print(
            f" Total Return:         {report_data.get('total_return_pct', 0.0):,.2f}%"
        )
        print(
            f" Sharpe Ratio (Ann.):  {report_data.get('sharpe_ratio_annual', 0.0):.2f}"
        )
        print(
            f" Sortino Ratio (Ann.): {report_data.get('sortino_ratio_annual', 0.0):.2f}"
        )
        print(
            f" Max Drawdown:         {report_data.get('max_drawdown_pct', 0.0):,.2f}%"
        )
        print("-" * 50)
        print(f" Total Trades:         {report_data.get('total_trades', 0)}")
        print(f" Win Rate:             {report_data.get('win_rate_pct', 0.0):.2f}%")
        print(f" Average Profit:       {report_data.get('average_profit', 0.0):,.3f}")
        print(f" Average Loss:         {report_data.get('average_loss', 0.0):,.3f}")
        print(f" Profit Factor:        {report_data.get('profit_factor', 0.0):.2f}")
        print(
            f" Avg. Bet Fraction:    {report_data.get('average_effective_bet_fraction_pct', 0.0):.2f}%"
        )
        print("-" * 50)
        print(
            f" Period:               {report_data.get('data_period_start', 'N/A')} to {report_data.get('data_period_end', 'N/A')}"
        )
        print("=" * 50)

        FINAL_REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)

        logging.info("Generating equity curve and drawdown chart...")
        if results_df.is_empty() or drawdown.is_empty():
            logging.warning("No data available to generate equity curve chart.")
        else:
            try:
                sns.set_style("darkgrid")
                fig, (ax1, ax2) = plt.subplots(
                    2,
                    1,
                    figsize=(15, 10),
                    sharex=True,
                    gridspec_kw={"height_ratios": [3, 1]},
                )
                timestamps_list = results_df["timestamp"].to_list()
                equity_list_float = equity_series_float.slice(1).to_list()
                drawdown_list_raw = drawdown.to_list()

                drawdown_list_float = [
                    d if np.isfinite(d) else 0.0 for d in drawdown_list_raw
                ]
                ax1.plot(
                    timestamps_list,
                    equity_list_float,
                    label="Equity Curve",
                    color="dodgerblue",
                )
                ax1.set_title(
                    f"V5 Equity Curve (Auto Lot: {self.config.auto_lot_size_per_base} per {self.config.auto_lot_base_capital}, M2 Thresh: {self.config.m2_proba_threshold})",
                    fontsize=16,
                )
                ax1.set_ylabel("Equity")
                ax1.grid(True)
                try:
                    finite_equity = [
                        e for e in equity_list_float if np.isfinite(e) and e > 0
                    ]
                    if not finite_equity:
                        ax1.ticklabel_format(style="plain", axis="y")
                    elif any(np.isinf(equity_list_float)) or (
                        max(finite_equity, default=1)
                        / max(min(finite_equity, default=1), 1)
                        > 1000
                    ):
                        ax1.set_yscale("log")
                    else:
                        ax1.ticklabel_format(style="plain", axis="y")
                except Exception as scale_err:
                    logging.warning(
                        f"Could not determine y-axis scale, using plain: {scale_err}"
                    )
                    ax1.ticklabel_format(style="plain", axis="y")
                ax2.fill_between(
                    timestamps_list, drawdown_list_float, 0, color="red", alpha=0.3
                )
                ax2.set_title("Drawdown", fontsize=16)
                ax2.set_ylabel("Drawdown (%)")
                ax2.yaxis.set_major_formatter(
                    mtick.PercentFormatter(xmax=1.0, decimals=1)
                )
                ax2.grid(True)
                plt.tight_layout()
                EQUITY_CURVE_PATH.parent.mkdir(parents=True, exist_ok=True)
                plt.savefig(EQUITY_CURVE_PATH)
                logging.info(f"Saved equity curve chart to {EQUITY_CURVE_PATH}")
                plt.close(fig)
            except Exception as e:
                logging.error(
                    f"Failed to generate equity curve chart: {e}", exc_info=True
                )

        if not trade_log.is_empty():
            _log_suffix = "M1" if "m1_oof" in str(self.config.oof_long_path) else "M2"
            trade_log_output_path = (
                FINAL_REPORT_PATH.parent / f"detailed_trade_log_v5_{_log_suffix}.csv"
            )
            logging.info(
                f"Preparing detailed trade log for CSV output ({len(trade_log)} trades)..."
            )
            try:
                temp_log_formatted = trade_log.clone()
                format_expressions = []

                format_expressions.append(
                    pl.col("timestamp")
                    .dt.strftime("%Y-%m-%d %H:%M:%S")
                    .alias("timestamp")
                )

                decimal_cols_round = {
                    "balance": 2,  # ★変更
                    "pnl": 2,
                    "pnl_gross": 2,
                    "spread": 2,  # ★変更
                    "margin": 2,  # ★変更
                    "close_price": 3,
                }
                for col_name, digits in decimal_cols_round.items():
                    if col_name in temp_log_formatted.columns:
                        format_expressions.append(
                            pl.col(col_name)
                            .map_elements(
                                lambda d: (
                                    float(d)
                                    if d is not None and d.is_finite()
                                    else None
                                ),
                                return_dtype=pl.Float64,
                            )
                            .round(digits)
                            .alias(col_name)
                        )

                float_cols_round = {
                    "m2_proba": 4,
                    "lot_size": 2,
                    "atr_value": 4,
                    "atr_ratio": 4,
                    "leverage": 0,
                    "TD": 1,
                    "hold": 1,
                    "DD(%)": 2,
                    "mg_lv%": 1,
                }
                for col_name, digits in float_cols_round.items():
                    if col_name in temp_log_formatted.columns:
                        format_expressions.append(
                            pl.col(col_name).round(digits).alias(col_name)
                        )

                if format_expressions:
                    temp_log_formatted = temp_log_formatted.with_columns(
                        format_expressions
                    )

                desired_columns_final = [
                    "timestamp",
                    "direction",
                    "label",
                    "m2_proba",
                    "pnl",
                    "pnl_gross",
                    "er",
                    "d_atr",
                    "disp_atr",
                    "balance",
                    "lot_size",
                    "aL",
                    "aS",
                    "margin",
                    "leverage",
                    "spread",
                    "close_price",
                    "atr_value",
                    "atr_ratio",
                    "TD",
                    "hold",
                    "exit",
                    "DD(%)",
                    "mg_lv%",
                    "csl_L",
                    "csl_S",
                    "closs",
                ]

                available_columns_final = [
                    col
                    for col in desired_columns_final
                    if col in temp_log_formatted.columns
                ]
                trade_log_final_csv = temp_log_formatted.select(available_columns_final)

                # timestampをUTC→JSTに変換（+9時間）して上書き
                trade_log_final_csv = trade_log_final_csv.with_columns(
                    pl.col("timestamp")
                    .str.to_datetime("%Y-%m-%d %H:%M:%S", strict=False)
                    .dt.offset_by("9h")
                    .alias("timestamp")
                )

                trade_log_final_csv.write_csv(
                    trade_log_output_path,
                    null_value="NaN",
                )
                logging.info(
                    f"Formatted detailed trade log saved to {trade_log_output_path}"
                )

            except PermissionError as pe:
                logging.error(
                    f"Permission denied saving trade log to {trade_log_output_path}: {pe}"
                )
            except Exception as e:
                logging.error(
                    f"Failed to save formatted detailed trade log: {e}", exc_info=True
                )
        else:
            logging.info("No trades were executed, skipping detailed trade log output.")

        text_report_path = FINAL_REPORT_PATH.with_suffix(".txt")  # _M1/.txt or _M2/.txt
        logging.info(f"Generating text performance report to {text_report_path}...")
        try:
            # ▼▼▼ 追加計算: 各種統計情報の取得 ▼▼▼
            if not trade_log.is_empty():
                max_active_l = trade_log["aL"].max()
                max_active_s = trade_log["aS"].max()
                max_active_tot = (trade_log["aL"] + trade_log["aS"]).max()

                l_trades = trade_log.filter(pl.col("direction") == 1)
                s_trades = trade_log.filter(pl.col("direction") == -1)
                count_l = len(l_trades)
                count_s = len(s_trades)
                # [TD-RESIM/HOLD] "TD" はラベル生 duration (L 起点)、
                #   "hold" は TD 打ち切りを反映した実保有分 (エントリー起点)。
                #   Avg TD は実保有分で出す (旧実装は生 duration で、TD を絞っても
                #   数字が縮まず「TD が効いていない」ように見えた)。
                _hold_col = "hold" if "hold" in trade_log.columns else "TD"
                avg_td_l = l_trades[_hold_col].mean() if count_l > 0 else 0.0
                avg_td_s = s_trades[_hold_col].mean() if count_s > 0 else 0.0
                avg_raw_l = l_trades["TD"].mean() if count_l > 0 else 0.0
                avg_raw_s = s_trades["TD"].mean() if count_s > 0 else 0.0

                # 方向別 勝率・PF
                def _win_rate_pf(trades):
                    if len(trades) == 0:
                        return 0.0, 0.0
                    pnls = [float(p) for p in trades["pnl"].to_list() if p is not None]
                    wins = [p for p in pnls if p > 0]
                    losses = [p for p in pnls if p < 0]
                    wr = len(wins) / len(pnls) * 100 if pnls else 0.0
                    pf = sum(wins) / abs(sum(losses)) if losses else float("inf")
                    return wr, pf

                wr_l, pf_l = _win_rate_pf(l_trades)
                wr_s, pf_s = _win_rate_pf(s_trades)

                # 連続SL最大値（selfから取得）
                max_csl_long = self.max_consec_sl_long
                max_csl_short = self.max_consec_sl_short
                max_csl_total = self.max_consec_sl_total
                max_closs_total = self.max_consec_loss_total

                # 証拠金維持率最低値（selfから取得）
                min_mg_lv = (
                    float(self.min_margin_level_pct)
                    if self.min_margin_level_pct != Decimal("inf")
                    else None
                )

                # ▼▼▼ 修正前 ▼▼▼
                # to_count = len(
                #     trade_log.filter(
                #         (pl.col("label") == 0)
                #         & (
                #             ((pl.col("direction") == 1) & (pl.col("TD") >= 119.9))
                #             | ((pl.col("direction") == -1) & (pl.col("TD") >= 59.9))
                #         )
                #     )
                # )

                # ▼▼▼ 修正後 ▼▼▼
                # [TD-RESIM] 決済種別は決済時点で確定済みなので "exit" 列を直接数える。
                #   (旧: label と TD から推定していたため、TD 打ち切りで TO 化した
                #    PT 玉が数え漏れていた)
                if "exit" in trade_log.columns:
                    to_count = len(trade_log.filter(pl.col("exit") == "TO"))
                else:
                    _ah = float(ACTION_HORIZON_MIN)
                    _to_expr = (
                        (pl.col("direction") == 1)
                        & ((pl.col("TD") - _ah) >= (self.config.td_minutes_long - 1e-9))
                    ) | (
                        (pl.col("direction") == -1)
                        & (
                            (pl.col("TD") - _ah)
                            >= (self.config.td_minutes_short - 1e-9)
                        )
                    )
                    to_count = len(trade_log.filter(_to_expr))
                m2_lst = trade_log["m2_proba"].to_list()
                m2_bins = {
                    "<= 0.50": sum(1 for x in m2_lst if x <= 0.50),
                    "0.50-0.55": sum(1 for x in m2_lst if 0.50 < x <= 0.55),
                    "0.55-0.60": sum(1 for x in m2_lst if 0.55 < x <= 0.60),
                    "0.60-0.65": sum(1 for x in m2_lst if 0.60 < x <= 0.65),
                    "0.65-0.70": sum(1 for x in m2_lst if 0.65 < x <= 0.70),
                    "0.70-0.75": sum(1 for x in m2_lst if 0.70 < x <= 0.75),
                    "0.75-0.80": sum(1 for x in m2_lst if 0.75 < x <= 0.80),
                    "0.80-0.85": sum(1 for x in m2_lst if 0.80 < x <= 0.85),
                    "0.85-0.90": sum(1 for x in m2_lst if 0.85 < x <= 0.90),
                    "0.90-0.95": sum(1 for x in m2_lst if 0.90 < x <= 0.95),
                    "0.95-1.00": sum(1 for x in m2_lst if x > 0.95),
                }
                # ATRは相対値(atr_ratio)で集計 ── min_atr_thresholdと同じ軸で評価するため
                # trade_logにatr_ratioカラムがなければatr_valueで代替（警告付き）
                if "atr_ratio" in trade_log.columns:
                    atr_rel_lst = trade_log["atr_ratio"].to_list()
                    atr_label = "ATR Ratio (Relative)"
                else:
                    atr_rel_lst = trade_log["atr_value"].to_list()
                    atr_label = "ATR Value (Absolute) ※atr_ratio列なし"
                    logging.warning(
                        "atr_ratio column not found in trade_log. Falling back to atr_value."
                    )
                # [BANDS] 帯の区切りは USER_PARAMS["atr_ratio_bands"] から
                _ratio_band_defs = _build_band_defs(self.config.atr_ratio_bands)
                atr_bins = {
                    _bn: sum(1 for x in atr_rel_lst if x is not None and _bf(x))
                    for _bn, _bf in _ratio_band_defs
                }

                # pnl・labelリストを帯別分析で共通利用するため先に取得
                pnl_lst = trade_log["pnl"].to_list()
                label_lst = trade_log["label"].to_list()

                # ATR絶対値帯別 勝率・PF分析（参考）
                atr_abs_lst = trade_log["atr_value"].to_list()
                atr_abs_band_defs = [
                    ("< 1.0", lambda x: x < 1.0),
                    ("1.0-2.0", lambda x: 1.0 <= x < 2.0),
                    ("2.0-3.0", lambda x: 2.0 <= x < 3.0),
                    ("3.0-5.0", lambda x: 3.0 <= x < 5.0),
                    (">= 5.0", lambda x: x >= 5.0),
                ]
                atr_abs_band_stats = {}
                for band_name, band_fn in atr_abs_band_defs:
                    idxs = [
                        i
                        for i, x in enumerate(atr_abs_lst)
                        if x is not None and band_fn(x)
                    ]
                    if not idxs:
                        atr_abs_band_stats[band_name] = None
                        continue
                    band_labels = [label_lst[i] for i in idxs]
                    band_pnls = [
                        float(pnl_lst[i]) if pnl_lst[i] is not None else 0.0
                        for i in idxs
                    ]
                    wins = [p for p in band_pnls if p > 0]
                    loses = [p for p in band_pnls if p < 0]
                    pf = sum(wins) / abs(sum(loses)) if loses else float("inf")
                    atr_abs_band_stats[band_name] = {
                        "count": len(idxs),
                        "win_rate": sum(1 for l in band_labels if l == 1)
                        / len(idxs)
                        * 100,
                        "pf": pf,
                        "avg_pnl": sum(band_pnls) / len(band_pnls),
                    }

                # ATR Ratio帯別 勝率・PF分析
                atr_band_stats = {}
                # [BANDS] USER_PARAMS["atr_ratio_bands"] で区切りを変えられる
                atr_band_defs = _ratio_band_defs
                for band_name, band_fn in atr_band_defs:
                    idxs = [
                        i
                        for i, x in enumerate(atr_rel_lst)
                        if x is not None and band_fn(x)
                    ]
                    if not idxs:
                        atr_band_stats[band_name] = None
                        continue
                    band_labels = [label_lst[i] for i in idxs]
                    band_pnls = [
                        float(pnl_lst[i]) if pnl_lst[i] is not None else 0.0
                        for i in idxs
                    ]
                    wins = [p for p in band_pnls if p > 0]
                    loses = [p for p in band_pnls if p < 0]
                    pf = sum(wins) / abs(sum(loses)) if loses else float("inf")
                    atr_band_stats[band_name] = {
                        "count": len(idxs),
                        # [WINRATE-FIX] 旧実装は label==1 の割合を勝率としていたが、
                        #   TD 打ち切りで label==1 が損失決済になりうるため、
                        #   全体の Win Rate (pnl>0) と定義が食い違っていた。pnl 基準に統一。
                        "win_rate": len(wins) / len(band_pnls) * 100
                        if band_pnls
                        else 0.0,
                        "pf": pf,
                        "avg_pnl": sum(band_pnls) / len(band_pnls),
                    }
            else:
                max_active_l = max_active_s = max_active_tot = count_l = count_s = (
                    to_count
                ) = 0
                avg_td_l = avg_td_s = 0.0
                wr_l = wr_s = pf_l = pf_s = 0.0
                max_csl_long = max_csl_short = max_csl_total = max_closs_total = 0
                min_mg_lv = None
                m2_bins = atr_bins = {}
                atr_band_stats = {}
                atr_abs_band_stats = {}
                atr_label = "ATR Ratio (Relative)"
            # ▲▲▲ ここまで追加 ▲▲▲

            # --- 時間帯・曜日分析の事前計算（CSVとTXT両方で使用）---
            def _session(h):
                """JST時間帯でセッション分類"""
                if 9 <= h < 16:
                    return "Tokyo"
                elif 16 <= h < 21:
                    return "London"
                elif h >= 21 or h < 1:
                    return "Overlap"
                elif 1 <= h < 6:
                    return "NY"
                else:
                    return "Oceania"  # 6-9 JST

            def _band_stats(indices, pnl_lst, label_lst):
                if not indices:
                    return None
                p = [float(pnl_lst[i]) for i in indices if pnl_lst[i] is not None]
                wins = [x for x in p if x > 0]
                losses = [x for x in p if x < 0]
                wr = len(wins) / len(p) * 100 if p else 0.0
                pf = sum(wins) / abs(sum(losses)) if losses else float("inf")
                avg = sum(p) / len(p) if p else 0.0
                tot = sum(p)
                return {
                    "count": len(p),
                    "win_rate": wr,
                    "pf": pf,
                    "avg_pnl": avg,
                    "total_pnl": tot,
                }

            hourly_stats = {}
            weekday_stats = {}
            hxatr_stats = {}

            if not trade_log.is_empty():
                ts_list2 = trade_log["timestamp"].to_list()
                pnl_lst2 = trade_log["pnl"].to_list()
                lbl_lst2 = trade_log["label"].to_list()
                atr_lst2 = (
                    trade_log["atr_ratio"].to_list()
                    if "atr_ratio" in trade_log.columns
                    else [1.0] * len(ts_list2)
                )

                # [BANDS] USER_PARAMS["atr_ratio_bands"] で区切りを変えられる
                atr_bands = _build_band_defs(self.config.atr_ratio_bands)

                hour_idx = {}
                weekday_idx = {}
                hxatr_idx = {}

                for i, ts in enumerate(ts_list2):
                    try:
                        # UTC→JST変換（+9時間）
                        h_jst = (ts.hour + 9) % 24
                        # 日またぎ考慮: UTC時刻+9が24を超えた場合は翌日
                        day_offset = 1 if (ts.hour + 9) >= 24 else 0
                        wd_jst = (ts.weekday() + day_offset) % 7
                    except Exception:
                        continue
                    hour_idx.setdefault(h_jst, []).append(i)
                    weekday_idx.setdefault(wd_jst, []).append(i)
                    ar = atr_lst2[i]
                    if ar is not None and isinstance(ar, (int, float)):
                        for band_name, band_fn in atr_bands:
                            if band_fn(float(ar)):
                                hxatr_idx.setdefault((h_jst, band_name), []).append(i)
                                break

                for h in range(24):
                    hourly_stats[h] = _band_stats(
                        hour_idx.get(h, []), pnl_lst2, lbl_lst2
                    )
                for wd in range(7):
                    weekday_stats[wd] = _band_stats(
                        weekday_idx.get(wd, []), pnl_lst2, lbl_lst2
                    )
                for h in range(24):
                    for band_name, _ in atr_bands:
                        hxatr_stats[(h, band_name)] = _band_stats(
                            hxatr_idx.get((h, band_name), []), pnl_lst2, lbl_lst2
                        )

            with open(text_report_path, "w", encoding="utf-8") as f:
                f.write("=" * 60 + "\n")
                f.write("    V5 Two-Brain Strategy Performance Report (MT5 Style)\n")
                f.write("=" * 60 + "\n\n")
                f.write(f"Strategy:\t\t{report_data.get('strategy', 'N/A')}\n")
                f.write(
                    f"Period:\t\t\t{report_data.get('data_period_start', 'N/A')} - {report_data.get('data_period_end', 'N/A')}\n\n"
                )

                # ▼▼▼ 新規追加: BacktestConfigの全パラメーターを出力 ▼▼▼
                f.write("-" * 22 + " Configuration " + "-" * 23 + "\n")
                for key, value in self.config.__dict__.items():
                    f.write(f"{key.ljust(30)}: {value}\n")
                f.write("\n")
                # ▲▲▲ ここまで追加 ▲▲▲

                # ▼▼▼ 新規追加: サーキットブレーカーのサマリーを出力 ▼▼▼
                f.write("-" * 21 + " Circuit Breakers " + "-" * 21 + "\n")
                f.write(
                    f"Simultaneous Orders Prevented:  {report_data.get('cb_simultaneous_prevented', 0)} times\n"
                )
                f.write(
                    f"Cooldown Triggered (Long)    :  {report_data.get('cb_cooldown_long', 0)} times\n"
                )
                f.write(
                    f"Cooldown Triggered (Short)   :  {report_data.get('cb_cooldown_short', 0)} times\n\n"
                )
                # ▲▲▲ ここまで追加 ▲▲▲

                f.write("-" * 30 + " Summary " + "-" * 30 + "\n")
                initial_cap = report_data.get("initial_capital", 0.0)
                final_cap = report_data.get("final_capital", 0.0)
                total_net_profit = final_cap - initial_cap
                total_ret_pct = report_data.get("total_return_pct", 0.0)
                f.write(f"Initial Deposit:\t{initial_cap:,.2f}\n")
                f.write(f"Total Net Profit:\t{total_net_profit:,.2f}\n")
                f.write(f"Final Balance:\t\t{final_cap:,.2f}\n")
                f.write(f"Total Return:\t\t{total_ret_pct:,.2f} %\n")
                profit_factor = report_data.get("profit_factor", 0.0)
                f.write(f"Profit Factor:\t\t{profit_factor:.2f}\n")
                sharpe = report_data.get("sharpe_ratio_annual", 0.0)
                f.write(f"Sharpe Ratio (Ann.):\t{sharpe:.2f}\n")
                sortino = report_data.get("sortino_ratio_annual", 0.0)
                f.write(f"Sortino Ratio (Ann.):\t{sortino:.2f}\n")
                max_dd = report_data.get("max_drawdown_pct", 0.0)
                f.write(f"Maximal Drawdown:\t{abs(max_dd):,.2f} %\n")
                f.write(
                    f"Min Margin Level:\t{min_mg_lv:,.1f} %"
                    if min_mg_lv is not None
                    else "Min Margin Level:\tN/A (no open positions)"
                )
                f.write("\n\n")
                f.write("-" * 30 + " Trades " + "-" * 30 + "\n")
                total_trades = report_data.get("total_trades", 0)
                win_pct = report_data.get("win_rate_pct", 0.0)
                loss_pct = 100.0 - win_pct if total_trades > 0 else 0.0
                num_win_trades = int(total_trades * (win_pct / 100.0))
                num_loss_trades = total_trades - num_win_trades
                f.write(f"Total Trades:\t\t{total_trades}\n")
                f.write(f"Winning Trades (%):\t{num_win_trades} ({win_pct:.2f} %)\n")
                f.write(f"Losing Trades (%):\t{num_loss_trades} ({loss_pct:.2f} %)\n")
                avg_profit = report_data.get("average_profit", 0.0)
                avg_loss = report_data.get("average_loss", 0.0)
                f.write(f"Average Profit:\t\t{avg_profit:,.3f}\n")
                f.write(f"Average Loss:\t\t{avg_loss:,.3f}\n")
                avg_bet_pct = report_data.get("average_effective_bet_fraction_pct", 0.0)
                f.write(f"Avg Bet Size (% Cap):\t{avg_bet_pct:.2f} %\n\n")

                f.write("-" * 25 + " Direction Analysis " + "-" * 25 + "\n")
                f.write(f"{'':20}{'Long':>12}{'Short':>12}\n")
                f.write(f"{'Trade Count':20}{count_l:>12}{count_s:>12}\n")
                f.write(f"{'Win Rate (%)':20}{wr_l:>11.2f}%{wr_s:>11.2f}%\n")
                pf_l_str = f"{pf_l:.2f}" if pf_l != float("inf") else "inf"
                pf_s_str = f"{pf_s:.2f}" if pf_s != float("inf") else "inf"
                f.write(f"{'Profit Factor':20}{pf_l_str:>12}{pf_s_str:>12}\n\n")

                f.write("-" * 25 + " Consecutive Losses " + "-" * 25 + "\n")
                f.write(f"Max Consec SL (Long):\t{max_csl_long}\n")
                f.write(f"Max Consec SL (Short):\t{max_csl_short}\n")
                f.write(f"Max Consec SL (Total):\t{max_csl_total}\n")
                f.write(f"Max Consec Loss (SL+TO):\t{max_closs_total}\n\n")

                # ▼▼▼ 追加出力: 詳細統計と分布 ▼▼▼
                f.write("-" * 23 + " Positions & Durations " + "-" * 14 + "\n")
                f.write(f"Max Concurrent Longs:\t{max_active_l}\n")
                f.write(f"Max Concurrent Shorts:\t{max_active_s}\n")
                f.write(f"Max Concurrent Total:\t{max_active_tot}\n")
                f.write(f"Total Long Trades:\t{count_l}\n")
                f.write(f"Total Short Trades:\t{count_s}\n")
                f.write(
                    f"Avg Hold (Long):\t{avg_td_l:.1f} mins   "
                    f"(ラベル生 duration: {avg_raw_l:.1f})\n"
                )
                f.write(
                    f"Avg Hold (Short):\t{avg_td_s:.1f} mins   "
                    f"(ラベル生 duration: {avg_raw_s:.1f})\n"
                )
                f.write(
                    f"  ※ Avg Hold = エントリー起点の実保有分 (TD={self.config.td_minutes_long:g}"
                    f"/{self.config.td_minutes_short:g} 打ち切り反映)\n"
                )
                f.write(
                    f"  ※ ラベル生 duration = L 起点・打ち切り前 "
                    f"(ラベル TD {LABEL_TD_MINUTES:g} 分 + 行動地平 {ACTION_HORIZON_MIN:g} 分)\n"
                )
                # [SPREAD-FIX] コスト内訳と照合
                if "spread" in trade_log.columns:
                    _sp = sum(
                        float(x) for x in trade_log["spread"].to_list() if x is not None
                    )
                    _gr = (
                        sum(
                            float(x)
                            for x in trade_log["pnl_gross"].to_list()
                            if x is not None
                        )
                        if "pnl_gross" in trade_log.columns
                        else float("nan")
                    )
                    _nt = sum(
                        float(x) for x in trade_log["pnl"].to_list() if x is not None
                    )
                    f.write("\n--- Cost Breakdown ---\n")
                    f.write(f"  Gross PnL (スプレッド控除前) : {_gr:>12,.2f}\n")
                    f.write(f"  Total Spread Cost            : {-_sp:>12,.2f}\n")
                    f.write(f"  Net PnL (= 以降の全指標の基準): {_nt:>12,.2f}\n")
                    if len(trade_log) > 0:
                        f.write(
                            f"  1トレード平均: gross {_gr / len(trade_log):+.3f}"
                            f" / spread {-_sp / len(trade_log):+.3f}"
                            f" / net {_nt / len(trade_log):+.3f}\n"
                        )
                    if abs(_gr) > 1e-9:
                        f.write(
                            f"  スプレッドが gross を食う割合: {_sp / abs(_gr) * 100:.1f}%\n"
                        )
                    f.write("\n")

                # [DISP] 年別 実現変位 — measure_mu_time_profile 第3段と直接突合する表
                if "disp_atr" in trade_log.columns:
                    f.write("\n--- 年別 実現変位 (ATR単位) ---\n")
                    f.write(
                        "  measure_mu_time_profile.py --rule sweep の【第3段】と"
                        "直接比較できます。\n"
                        "  測定は E[ΔX×sign(d)] で逆行が負、こちらは損益なので符号が反転します。\n"
                        "  例) 測定 -0.2508  ⇔  BT +0.2508\n\n"
                    )
                    f.write(
                        f"  {'年':>6}{'n':>8}{'効果':>10}{'SE':>9}{'t':>8}"
                        f"{'±3刈り':>11}{'t':>8}   符号\n"
                    )
                    f.write("  " + "-" * 71 + "\n")
                    _yrs = [t.year for t in trade_log["timestamp"].to_list()]
                    _ds = [
                        (float(x) if x is not None else float("nan"))
                        for x in trade_log["disp_atr"].to_list()
                    ]
                    _all: List[float] = []
                    for _y in sorted(set(_yrs)):
                        _v = [
                            d
                            for yy, d in zip(_yrs, _ds)
                            if yy == _y and d == d  # NaN 除外
                        ]
                        if len(_v) < 2:
                            continue
                        _all.extend(_v)
                        _m = float(np.mean(_v))
                        _se = float(np.std(_v, ddof=1) / np.sqrt(len(_v)))
                        _t = _m / _se if _se > 0 else 0.0
                        # [OUTLIER] ±3 ATR で刈った平均。外れ値(週末ギャップ・
                        #   板飛び等)が平均を潰していないかの判定に使う。
                        #   測定器は STALE_TICK_LIMIT / disc 対応でこの種のバーを
                        #   落としているが、BT には同等の除外が無い。
                        _w = [min(max(x, -3.0), 3.0) for x in _v]
                        _mw = float(np.mean(_w))
                        _sew = float(np.std(_w, ddof=1) / np.sqrt(len(_w)))
                        _tw = _mw / _sew if _sew > 0 else 0.0
                        _sg = "順行" if _m > 0 else "逆行"
                        _bar = "#" * min(20, int(abs(_t) * 4))
                        f.write(
                            f"  {_y:>6}{len(_v):>8}{_m:>+10.4f}{_se:>9.4f}"
                            f"{_t:>+8.2f}{_mw:>+11.4f}{_tw:>+8.2f}   {_sg} {_bar}\n"
                        )
                    if len(_all) > 1:
                        _m = float(np.mean(_all))
                        _se = float(np.std(_all, ddof=1) / np.sqrt(len(_all)))
                        f.write("  " + "-" * 52 + "\n")
                        f.write(
                            f"  {'統合':>6}{len(_all):>8}{_m:>+10.4f}{_se:>9.4f}"
                            f"{(_m / _se if _se > 0 else 0.0):>+8.2f}\n"
                        )
                        # [OUTLIER] 分布の裾を見る。測定 σ ≒ 1.5 に対し
                        #   BT σ が大きければ、測定が落としているバーを拾っている。
                        _q = np.percentile(_all, [0.1, 1, 5, 25, 50, 75, 95, 99, 99.9])
                        f.write("\n  disp_atr の分布 (外れ値診断):\n")
                        f.write(
                            "    "
                            + "".join(
                                f"{lbl:>9}"
                                for lbl in [
                                    "0.1%",
                                    "1%",
                                    "5%",
                                    "25%",
                                    "50%",
                                    "75%",
                                    "95%",
                                    "99%",
                                    "99.9%",
                                ]
                            )
                            + "\n"
                        )
                        f.write("    " + "".join(f"{v:>9.3f}" for v in _q) + "\n")
                        _sd = float(np.std(_all, ddof=1))
                        _n3 = sum(1 for x in _all if abs(x) > 3.0)
                        f.write(
                            f"    σ = {_sd:.3f}  (測定器 ≒ 1.5)"
                            f" / |disp| > 3 ATR: {_n3} 件 ({100.0 * _n3 / len(_all):.2f}%)\n\n"
                        )
                        _sp_per = (
                            0.36
                            if not len(trade_log)
                            else sum(
                                float(x)
                                for x in trade_log["spread"].to_list()
                                if x is not None
                            )
                            / len(trade_log)
                        )
                        _lot = float(self.config.fixed_lot_size)
                        if _m > 0 and _lot > 0:
                            f.write(
                                f"  損益分岐 ATR = spread/件 ÷ (効果 × lot × 100)"
                                f" = {_sp_per:.3f} ÷ ({_m:.4f} × {_lot} × 100)"
                                f" = {_sp_per / (_m * _lot * 100):.2f} USD\n"
                            )
                    f.write("\n")

                # 決済種別の内訳 (PT / SL / TO)
                if "exit" in trade_log.columns:
                    f.write("\n--- Exit Type Breakdown ---\n")
                    f.write(
                        f"  {'Type':<6}{'Count':>7}{'Ratio%':>9}"
                        f"{'WinRate%':>10}{'TotalPnL':>12}{'AvgPnL':>10}\n"
                    )
                    _tot_n = len(trade_log)
                    for _k in ("PT", "SL", "TO"):
                        _sub = trade_log.filter(pl.col("exit") == _k)
                        _n = len(_sub)
                        if _n == 0:
                            f.write(
                                f"  {_k:<6}{0:>7}{0.0:>9.1f}{'-':>10}{'-':>12}{'-':>10}\n"
                            )
                            continue
                        _pnls = [float(x) for x in _sub["pnl"].to_list()]
                        _wins = sum(1 for x in _pnls if x > 0)
                        _sum = sum(_pnls)
                        f.write(
                            f"  {_k:<6}{_n:>7}{100.0 * _n / _tot_n:>9.1f}"
                            f"{100.0 * _wins / _n:>10.2f}{_sum:>12.2f}{_sum / _n:>10.3f}\n"
                        )
                    f.write("\n")
                f.write(f"Timeout (TO) Count:\t{to_count}\n\n")

                _proba_label = (
                    "M1" if "m1_oof" in str(self.config.oof_long_path) else "M2"
                )

                # --- M1 全トリガー分布（参考値・M2モード時のみ追加表示）---
                if "m1_oof" not in str(self.config.oof_long_path):
                    try:
                        _m1_oof_long = pl.read_parquet(S7_M1_OOF_PREDICTIONS_LONG)
                        _m1_oof_short = pl.read_parquet(S7_M1_OOF_PREDICTIONS_SHORT)
                        _m1_all = pl.concat([_m1_oof_long, _m1_oof_short])[
                            "prediction"
                        ].to_list()
                        _m1_total = len(_m1_all)
                        _m1_bins = {
                            "<= 0.50": sum(1 for x in _m1_all if x <= 0.50),
                            "0.50-0.55": sum(1 for x in _m1_all if 0.50 < x <= 0.55),
                            "0.55-0.60": sum(1 for x in _m1_all if 0.55 < x <= 0.60),
                            "0.60-0.65": sum(1 for x in _m1_all if 0.60 < x <= 0.65),
                            "0.65-0.70": sum(1 for x in _m1_all if 0.65 < x <= 0.70),
                            "0.70-0.75": sum(1 for x in _m1_all if 0.70 < x <= 0.75),
                            "0.75-0.80": sum(1 for x in _m1_all if 0.75 < x <= 0.80),
                            "0.80-0.85": sum(1 for x in _m1_all if 0.80 < x <= 0.85),
                            "0.85-0.90": sum(1 for x in _m1_all if 0.85 < x <= 0.90),
                            "0.90-0.95": sum(1 for x in _m1_all if 0.90 < x <= 0.95),
                            "0.95-1.00": sum(1 for x in _m1_all if x > 0.95),
                        }
                        f.write(
                            "-" * 23
                            + " M1 Proba Distribution (参考 / OOFベース) "
                            + "-" * 3
                            + "\n"
                        )
                        f.write("  ※ M1の生の出力分布（M2への入力前）\n")
                        for k, v in _m1_bins.items():
                            pct = (v / _m1_total) * 100 if _m1_total > 0 else 0
                            f.write(
                                f"{k.ljust(15)}: {str(v).rjust(8)} ({pct:5.1f} %)\n"
                            )
                        f.write("\n")
                    except Exception as _e:
                        logging.warning(f"M1分布の計算に失敗しました: {_e}")

                # --- 全トリガー分布（OOFファイルから直接計算）---
                try:
                    _oof_long = pl.read_parquet(self.config.oof_long_path)
                    _oof_short = pl.read_parquet(self.config.oof_short_path)
                    _oof_all = pl.concat([_oof_long, _oof_short])[
                        "prediction"
                    ].to_list()
                    _total_triggers = len(_oof_all)
                    _all_bins = {
                        "<= 0.50": sum(1 for x in _oof_all if x <= 0.50),
                        "0.50-0.55": sum(1 for x in _oof_all if 0.50 < x <= 0.55),
                        "0.55-0.60": sum(1 for x in _oof_all if 0.55 < x <= 0.60),
                        "0.60-0.65": sum(1 for x in _oof_all if 0.60 < x <= 0.65),
                        "0.65-0.70": sum(1 for x in _oof_all if 0.65 < x <= 0.70),
                        "0.70-0.75": sum(1 for x in _oof_all if 0.70 < x <= 0.75),
                        "0.75-0.80": sum(1 for x in _oof_all if 0.75 < x <= 0.80),
                        "0.80-0.85": sum(1 for x in _oof_all if 0.80 < x <= 0.85),
                        "0.85-0.90": sum(1 for x in _oof_all if 0.85 < x <= 0.90),
                        "0.90-0.95": sum(1 for x in _oof_all if 0.90 < x <= 0.95),
                        "0.95-1.00": sum(1 for x in _oof_all if x > 0.95),
                    }
                    f.write(
                        "-" * 23
                        + f" {_proba_label} Proba Distribution (全トリガー / OOFベース) "
                        + "-" * 3
                        + "\n"
                    )
                    f.write("  ※ 全シグナル候補に対する生の分布（フィルター前）\n")
                    for k, v in _all_bins.items():
                        pct = (v / _total_triggers) * 100 if _total_triggers > 0 else 0
                        f.write(f"{k.ljust(15)}: {str(v).rjust(8)} ({pct:5.1f} %)\n")
                    f.write("\n")
                except Exception as _e:
                    logging.warning(f"全トリガー分布の計算に失敗しました: {_e}")

                # --- 約定トレード分布（濃縮後）---
                f.write(
                    "-" * 23
                    + f" {_proba_label} Proba Distribution (約定トレードのみ / 濃縮後) "
                    + "-" * 3
                    + "\n"
                )
                f.write("  ※ フィルター（閾値・Delta・ATR）通過後の約定トレードのみ\n")
                for k, v in m2_bins.items():
                    pct = (v / total_trades) * 100 if total_trades > 0 else 0
                    f.write(f"{k.ljust(15)}: {str(v).rjust(6)} ({pct:5.1f} %)\n")
                f.write("\n")

                f.write("-" * 23 + f" {atr_label} Distribution " + "-" * 3 + "\n")
                f.write(f"  (min_atr_threshold = {self.config.min_atr_threshold})\n")
                for k, v in atr_bins.items():
                    pct = (v / total_trades) * 100 if total_trades > 0 else 0
                    f.write(f"{k.ljust(15)}: {str(v).rjust(6)} ({pct:5.1f} %)\n")
                f.write("\n")

                f.write("-" * 23 + " ATR Ratio Band Analysis " + "-" * 12 + "\n")
                f.write(
                    f"  {'Band':<10} {'件数':>7} {'割合%':>7} {'勝率%':>7} {'PF':>7} {'平均PnL':>12}\n"
                )
                f.write("  " + "-" * 56 + "\n")
                for band_name, stats in atr_band_stats.items():
                    if stats is None:
                        f.write(f"  {band_name:<10} {'N/A':>7}\n")
                        continue
                    pct = stats["count"] / total_trades * 100 if total_trades > 0 else 0
                    f.write(
                        f"  {band_name:<10} {stats['count']:>7} {pct:>7.1f} "
                        f"{stats['win_rate']:>7.2f} {stats['pf']:>7.2f} {stats['avg_pnl']:>12.2f}\n"
                    )
                f.write("\n")

                f.write(
                    "-" * 23
                    + " ATR Value Band Analysis (参考: 絶対値) "
                    + "-" * 0
                    + "\n"
                )
                f.write(
                    f"  {'Band':<10} {'件数':>7} {'割合%':>7} {'勝率%':>7} {'PF':>7} {'平均PnL':>12}\n"
                )
                f.write("  " + "-" * 56 + "\n")
                for band_name, stats in atr_abs_band_stats.items():
                    if stats is None:
                        f.write(f"  {band_name:<10} {'N/A':>7}\n")
                        continue
                    pct = stats["count"] / total_trades * 100 if total_trades > 0 else 0
                    f.write(
                        f"  {band_name:<10} {stats['count']:>7} {pct:>7.1f} "
                        f"{stats['win_rate']:>7.2f} {stats['pf']:>7.2f} {stats['avg_pnl']:>12.2f}\n"
                    )
                f.write("\n")
                # ▲▲▲ ここまで追加 ▲▲▲

                # --- TXTに時間帯・曜日サマリーを追記 ---
                wd_names = [
                    "Monday",
                    "Tuesday",
                    "Wednesday",
                    "Thursday",
                    "Friday",
                    "Saturday",
                    "Sunday",
                ]
                session_order = ["Tokyo", "London", "Overlap", "NY", "Oceania"]

                # セッション別集計
                session_stats = {}
                for h, st in hourly_stats.items():
                    if st is None:
                        continue
                    s = _session(h)
                    if s not in session_stats:
                        session_stats[s] = {
                            "count": 0,
                            "wins": 0,
                            "pnl_wins": 0.0,
                            "pnl_losses": 0.0,
                        }
                    session_stats[s]["count"] += st["count"]
                    w = int(st["count"] * st["win_rate"] / 100)
                    session_stats[s]["wins"] += w
                    if st["pf"] != float("inf"):
                        l_cnt = st["count"] - w
                        if l_cnt > 0:
                            avg_loss = st["avg_pnl"] - st["win_rate"] / 100 * (
                                st["avg_pnl"] * st["pf"] / (1 + st["pf"])
                                if st["pf"] > 0
                                else 0
                            )
                    session_stats[s]["pnl_wins"] += (
                        st["total_pnl"] if st["total_pnl"] > 0 else 0
                    )
                    session_stats[s]["pnl_losses"] += (
                        st["total_pnl"] if st["total_pnl"] < 0 else 0
                    )

                f.write("-" * 22 + " Session Summary " + "-" * 21 + "\n")
                f.write(
                    f"  {'Session':<10}{'Trades':>8}{'WinRate%':>10}{'PF':>8}{'AvgPnL':>14}{'TotalPnL':>16}\n"
                )
                f.write("  " + "-" * 66 + "\n")
                # [SESSION-PF-FIX] 旧実装は「1時間ごとの合計PnL」を勝ち負けに
                #   振り分けて PF を作っていた (= 勝ちの時間帯の合計 / 負けの時間帯の合計)。
                #   これは Profit Factor ではなく、全部の時間帯がマイナスなら 0.00 に
                #   なる別物だった (実測: London 124件 勝率36% で PF 0.00)。
                #   トレード単位の gross profit / gross loss に直す。
                _sess_acc = {
                    _s: {"n": 0, "win": 0, "gp": 0.0, "gl": 0.0, "tot": 0.0}
                    for _s in session_order
                }
                for _ts, _pv in zip(
                    trade_log["timestamp"].to_list(), trade_log["pnl"].to_list()
                ):
                    try:
                        _h_jst = (_ts.hour + 9) % 24
                    except Exception:
                        continue
                    _sname = _session(_h_jst)
                    if _sname not in _sess_acc:
                        continue
                    _p = float(_pv) if _pv is not None else 0.0
                    _a = _sess_acc[_sname]
                    _a["n"] += 1
                    _a["tot"] += _p
                    if _p > 0:
                        _a["win"] += 1
                        _a["gp"] += _p
                    elif _p < 0:
                        _a["gl"] += -_p

                for s in session_order:
                    _a = _sess_acc.get(s)
                    if not _a or _a["n"] == 0:
                        continue
                    tc = _a["n"]
                    wr = _a["win"] / tc * 100
                    pf_s = _a["gp"] / _a["gl"] if _a["gl"] > 0 else float("inf")
                    pf_str = f"{pf_s:.2f}" if pf_s != float("inf") else "inf"
                    tot = _a["tot"]
                    avg = tot / tc
                    f.write(
                        f"  {s:<10}{tc:>8}{wr:>9.2f}%{pf_str:>8}{avg:>14,.2f}{tot:>16,.2f}\n"
                    )
                f.write("\n")

                f.write("-" * 22 + " Weekday Summary " + "-" * 21 + "\n")
                f.write(
                    f"  {'Weekday':<12}{'Trades':>8}{'WinRate%':>10}{'PF':>8}{'AvgPnL':>14}{'TotalPnL':>16}\n"
                )
                f.write("  " + "-" * 66 + "\n")
                for wd in range(7):
                    st = weekday_stats.get(wd)
                    if st is None:
                        continue
                    pf_str = f"{st['pf']:.2f}" if st["pf"] != float("inf") else "inf"
                    f.write(
                        f"  {wd_names[wd]:<12}{st['count']:>8}{st['win_rate']:>9.2f}%{pf_str:>8}{st['avg_pnl']:>14,.2f}{st['total_pnl']:>16,.2f}\n"
                    )
                f.write("\n")

                # ベスト・ワースト時間帯
                valid_hours = [
                    (h, st)
                    for h, st in hourly_stats.items()
                    if st and st["count"] >= 10
                ]
                if valid_hours:
                    best_wr = max(valid_hours, key=lambda x: x[1]["win_rate"])
                    worst_wr = min(valid_hours, key=lambda x: x[1]["win_rate"])
                    best_pf = max(
                        valid_hours,
                        key=lambda x: x[1]["pf"] if x[1]["pf"] != float("inf") else 0,
                    )
                    f.write("-" * 22 + " Hourly Highlights " + "-" * 19 + "\n")
                    f.write(
                        f"  Best  Win Rate : {best_wr[0]:02d}:00 UTC ({_session(best_wr[0])}) → {best_wr[1]['win_rate']:.2f}%\n"
                    )
                    f.write(
                        f"  Worst Win Rate : {worst_wr[0]:02d}:00 UTC ({_session(worst_wr[0])}) → {worst_wr[1]['win_rate']:.2f}%\n"
                    )
                    f.write(
                        f"  Best  PF       : {best_pf[0]:02d}:00 UTC ({_session(best_pf[0])}) → PF {best_pf[1]['pf']:.2f}\n\n"
                    )

                f.write("=" * 60 + "\n")
            logging.info(f"Text performance report saved successfully.")
        except Exception as e:
            logging.error(f"Failed to save text performance report: {e}", exc_info=True)

        # --- 月別・年別リターン CSV 出力 ---
        try:
            if not trade_log.is_empty():
                monthly_path = FINAL_REPORT_PATH.parent / (
                    FINAL_REPORT_PATH.stem + "_monthly_breakdown.csv"
                )
                # pnl/balance はObject型のため先にFloat64へ変換
                _tl = trade_log.with_columns(
                    [
                        pl.col("pnl")
                        .map_elements(
                            lambda x: float(x) if x is not None else None,
                            return_dtype=pl.Float64,
                        )
                        .alias("pnl_f"),
                        pl.col("label").cast(pl.Int32).alias("label_i"),
                        pl.col("timestamp").dt.year().alias("year"),
                        pl.col("timestamp").dt.month().alias("month"),
                    ]
                )

                monthly_rows = []
                for (yr, mo), grp in _tl.group_by(
                    ["year", "month"], maintain_order=False
                ):
                    pnls = grp["pnl_f"].to_list()
                    labels = grp["label_i"].to_list()
                    wins = [p for p in pnls if p > 0]
                    losses = [p for p in pnls if p < 0]
                    wr = len(wins) / len(pnls) * 100 if pnls else 0.0
                    pf = sum(wins) / abs(sum(losses)) if losses else float("inf")
                    tot_pnl = sum(pnls)
                    dd_vals = grp["DD(%)"].to_list()
                    max_dd = min(dd_vals) if dd_vals else 0.0
                    monthly_rows.append(
                        {
                            "year": yr,
                            "month": mo,
                            "trades": len(pnls),
                            "win_rate_%": round(wr, 2),
                            "profit_factor": round(pf, 3)
                            if pf != float("inf")
                            else None,
                            "total_pnl": round(tot_pnl, 2),
                            "max_dd_%": round(max_dd, 2),
                        }
                    )

                monthly_rows.sort(key=lambda r: (r["year"], r["month"]))

                # 年計行を挿入
                output_rows = []
                cur_year = None
                year_buf = []
                for row in monthly_rows:
                    if cur_year is not None and row["year"] != cur_year:
                        # 年計
                        yr_pnls_w = [
                            r["total_pnl"] for r in year_buf if r["total_pnl"] > 0
                        ]
                        yr_pnls_l = [
                            r["total_pnl"] for r in year_buf if r["total_pnl"] < 0
                        ]
                        yr_trades = sum(r["trades"] for r in year_buf)
                        yr_pf = (
                            sum(yr_pnls_w) / abs(sum(yr_pnls_l)) if yr_pnls_l else None
                        )
                        output_rows.append(
                            {
                                "year": cur_year,
                                "month": "TOTAL",
                                "trades": yr_trades,
                                "win_rate_%": "",
                                "profit_factor": round(yr_pf, 3) if yr_pf else None,
                                "total_pnl": round(
                                    sum(r["total_pnl"] for r in year_buf), 2
                                ),
                                "max_dd_%": round(
                                    min(r["max_dd_%"] for r in year_buf), 2
                                ),
                            }
                        )
                        output_rows.append({})  # 空行
                        year_buf = []
                    cur_year = row["year"]
                    year_buf.append(row)
                    output_rows.append(row)

                # 最終年の年計
                if year_buf:
                    yr_pnls_w = [r["total_pnl"] for r in year_buf if r["total_pnl"] > 0]
                    yr_pnls_l = [r["total_pnl"] for r in year_buf if r["total_pnl"] < 0]
                    yr_pf = sum(yr_pnls_w) / abs(sum(yr_pnls_l)) if yr_pnls_l else None
                    output_rows.append(
                        {
                            "year": cur_year,
                            "month": "TOTAL",
                            "trades": sum(r["trades"] for r in year_buf),
                            "win_rate_%": "",
                            "profit_factor": round(yr_pf, 3) if yr_pf else None,
                            "total_pnl": round(
                                sum(r["total_pnl"] for r in year_buf), 2
                            ),
                            "max_dd_%": round(min(r["max_dd_%"] for r in year_buf), 2),
                        }
                    )

                import csv as _csv

                fieldnames = [
                    "year",
                    "month",
                    "trades",
                    "win_rate_%",
                    "profit_factor",
                    "total_pnl",
                    "max_dd_%",
                ]
                with open(monthly_path, "w", newline="", encoding="utf-8") as f:
                    writer = _csv.DictWriter(f, fieldnames=fieldnames)
                    writer.writeheader()
                    for row in output_rows:
                        writer.writerow({k: row.get(k, "") for k in fieldnames})
                logging.info(f"Monthly breakdown CSV saved to {monthly_path}")
        except Exception as e:
            logging.error(f"Failed to save monthly breakdown CSV: {e}", exc_info=True)

        # --- 時間帯別分析 CSV ---
        try:
            if not trade_log.is_empty() and hourly_stats:
                import csv as _csv

                hourly_path = FINAL_REPORT_PATH.parent / (
                    FINAL_REPORT_PATH.stem + "_hourly_analysis.csv"
                )
                fields = [
                    "hour_jst",
                    "session",
                    "trades",
                    "win_rate_%",
                    "profit_factor",
                    "avg_pnl",
                    "total_pnl",
                ]
                with open(hourly_path, "w", newline="", encoding="utf-8") as f:
                    w = _csv.DictWriter(f, fieldnames=fields)
                    w.writeheader()
                    for h in range(24):
                        st = hourly_stats.get(h)
                        if st is None:
                            w.writerow(
                                {
                                    "hour_jst": h,
                                    "session": _session(h),
                                    "trades": 0,
                                    "win_rate_%": "",
                                    "profit_factor": "",
                                    "avg_pnl": "",
                                    "total_pnl": "",
                                }
                            )
                        else:
                            pf_val = (
                                round(st["pf"], 3) if st["pf"] != float("inf") else None
                            )
                            w.writerow(
                                {
                                    "hour_jst": h,
                                    "session": _session(h),
                                    "trades": st["count"],
                                    "win_rate_%": round(st["win_rate"], 2),
                                    "profit_factor": pf_val,
                                    "avg_pnl": round(st["avg_pnl"], 2),
                                    "total_pnl": round(st["total_pnl"], 2),
                                }
                            )
                logging.info(f"Hourly analysis CSV saved to {hourly_path}")
        except Exception as e:
            logging.error(f"Failed to save hourly analysis CSV: {e}", exc_info=True)

        # --- 曜日別分析 CSV ---
        try:
            if not trade_log.is_empty() and weekday_stats:
                import csv as _csv

                wd_path = FINAL_REPORT_PATH.parent / (
                    FINAL_REPORT_PATH.stem + "_weekday_analysis.csv"
                )
                wd_names_csv = [
                    "Monday",
                    "Tuesday",
                    "Wednesday",
                    "Thursday",
                    "Friday",
                    "Saturday",
                    "Sunday",
                ]
                fields = [
                    "weekday",
                    "weekday_name",
                    "trades",
                    "win_rate_%",
                    "profit_factor",
                    "avg_pnl",
                    "total_pnl",
                ]
                with open(wd_path, "w", newline="", encoding="utf-8") as f:
                    w = _csv.DictWriter(f, fieldnames=fields)
                    w.writeheader()
                    for wd in range(7):
                        st = weekday_stats.get(wd)
                        if st is None:
                            w.writerow(
                                {
                                    "weekday": wd,
                                    "weekday_name": wd_names_csv[wd],
                                    "trades": 0,
                                    "win_rate_%": "",
                                    "profit_factor": "",
                                    "avg_pnl": "",
                                    "total_pnl": "",
                                }
                            )
                        else:
                            pf_val = (
                                round(st["pf"], 3) if st["pf"] != float("inf") else None
                            )
                            w.writerow(
                                {
                                    "weekday": wd,
                                    "weekday_name": wd_names_csv[wd],
                                    "trades": st["count"],
                                    "win_rate_%": round(st["win_rate"], 2),
                                    "profit_factor": pf_val,
                                    "avg_pnl": round(st["avg_pnl"], 2),
                                    "total_pnl": round(st["total_pnl"], 2),
                                }
                            )
                logging.info(f"Weekday analysis CSV saved to {wd_path}")
        except Exception as e:
            logging.error(f"Failed to save weekday analysis CSV: {e}", exc_info=True)

        # --- 時間帯×ATR帯 分析 CSV ---
        try:
            if not trade_log.is_empty() and hxatr_stats:
                import csv as _csv

                hxatr_path = FINAL_REPORT_PATH.parent / (
                    FINAL_REPORT_PATH.stem + "_hour_x_atr_analysis.csv"
                )
                atr_band_names = [
                    "< 0.5",
                    "0.5-0.8",
                    "0.8-1.0",
                    "1.0-1.2",
                    "1.2-1.5",
                    ">= 1.5",
                ]
                fields = [
                    "hour_jst",
                    "session",
                    "atr_band",
                    "trades",
                    "win_rate_%",
                    "profit_factor",
                    "avg_pnl",
                    "total_pnl",
                ]
                with open(hxatr_path, "w", newline="", encoding="utf-8") as f:
                    w = _csv.DictWriter(f, fieldnames=fields)
                    w.writeheader()
                    for h in range(24):
                        for band_name in atr_band_names:
                            st = hxatr_stats.get((h, band_name))
                            if st is None or st["count"] == 0:
                                continue
                            pf_val = (
                                round(st["pf"], 3) if st["pf"] != float("inf") else None
                            )
                            w.writerow(
                                {
                                    "hour_jst": h,
                                    "session": _session(h),
                                    "atr_band": band_name,
                                    "trades": st["count"],
                                    "win_rate_%": round(st["win_rate"], 2),
                                    "profit_factor": pf_val,
                                    "avg_pnl": round(st["avg_pnl"], 2),
                                    "total_pnl": round(st["total_pnl"], 2),
                                }
                            )
                logging.info(f"Hour x ATR analysis CSV saved to {hxatr_path}")
        except Exception as e:
            logging.error(f"Failed to save hour x ATR analysis CSV: {e}", exc_info=True)

        return report_data


if __name__ == "__main__":
    default_config = BacktestConfig()

    parser = argparse.ArgumentParser(
        description="Project Forge V5 Backtest Simulator (Two-Brain, Auto Lot + Dynamic SL + Timeouts)"
    )

    parser.add_argument(
        "--auto-lot-base",
        type=float,
        default=default_config.auto_lot_base_capital,
        dest="auto_lot_base_capital",
        help=f"Base capital for auto lot calculation. Default: {default_config.auto_lot_base_capital}",
    )
    parser.add_argument(
        "--auto-lot-size",
        type=float,
        default=default_config.auto_lot_size_per_base,
        dest="auto_lot_size_per_base",
        help=f"Lot size per base capital. Default: {default_config.auto_lot_size_per_base}",
    )
    # ▼▼▼ 追加: 引数パーサー ▼▼▼
    parser.add_argument(
        "--use-fixed-risk",
        action="store_true",
        # [BUGFIX] argparse は help 文字列を % 書式として展開するため、
        #   素の % があると --help 実行時に ValueError で落ちる。%% にエスケープする。
        help="Use fixed risk %% position sizing instead of auto lot.",
    )
    parser.add_argument(
        "--no-fixed-risk",
        action="store_true",
        dest="no_fixed_risk",
        help="Disable fixed-risk sizing and fall back to auto-lot.",
    )
    # [追加] USER_PARAMS にあるが CLI から届いていなかったフィルター群
    parser.add_argument(
        "--max-atr",
        type=float,
        default=default_config.max_atr_threshold,
        dest="max_atr",
        help=f"Upper bound on atr_ratio (0 = off). Default: {default_config.max_atr_threshold}",
    )
    parser.add_argument(
        "--max-baseline-atr",
        type=float,
        default=default_config.max_baseline_atr,
        help=f"Upper bound on baseline ATR in USD (0 = off). Default: {default_config.max_baseline_atr}",
    )
    parser.add_argument(
        "--min-baseline-atr",
        type=float,
        default=default_config.min_baseline_atr,
        help=f"Absolute floor on baseline ATR in USD (0 = off). Default: {default_config.min_baseline_atr}",
    )
    parser.add_argument(
        "--min-baseline-ratio",
        type=float,
        default=default_config.min_baseline_ratio,
        help=f"Floor on baseline_ratio (0 = off). Default: {default_config.min_baseline_ratio}",
    )
    parser.add_argument(
        "--baseline-days",
        type=int,
        default=default_config.baseline_ratio_lookback_days,
        help=f"Lookback days for baseline_ratio denominator. Default: {default_config.baseline_ratio_lookback_days}",
    )
    parser.add_argument(
        "--min-sar",
        type=float,
        default=default_config.min_sar_threshold,
        help=f"Floor on seasonality-adjusted ATR ratio (0 = off). Default: {default_config.min_sar_threshold}",
    )
    parser.add_argument(
        "--sar-days",
        type=int,
        default=default_config.sar_lookback_days,
        help=f"Lookback days for SAR same-hour average. Default: {default_config.sar_lookback_days}",
    )
    parser.add_argument(
        "--strict-geometry",
        action="store_true",
        default=bool(USER_PARAMS.get("strict_geometry", True)),
        dest="strict_geometry",
        help="Abort if BT barrier geometry (pt/sl) does not match the labeling script.",
    )
    parser.add_argument(
        "--no-strict-geometry",
        action="store_false",
        dest="strict_geometry",
        help="Warn instead of aborting on geometry mismatch.",
    )
    parser.add_argument(
        "--fixed-risk-pct",
        type=float,
        default=default_config.fixed_risk_percent,
        dest="fixed_risk_pct",
        help=f"Risk percentage for fixed risk sizing (e.g., 0.02 for 2%%). Default: {default_config.fixed_risk_percent}",
    )
    # ▲▲▲ ここまで追加 ▲▲▲
    parser.add_argument(
        "--base-leverage",
        type=float,
        default=default_config.base_leverage,
        dest="base_leverage",
        help=f"Base leverage setting. Default: {default_config.base_leverage}",
    )

    parser.add_argument(
        "--m2-th",
        type=float,
        default=default_config.m2_proba_threshold,
        dest="m2_th",
        help=f"Min M2 prob threshold. Default: {default_config.m2_proba_threshold}",
    )
    # ▼▼▼ 追加: 差分(Delta)閾値用の引数 ▼▼▼
    parser.add_argument(
        "--m2-delta",
        type=float,
        default=default_config.m2_delta_threshold,
        dest="m2_delta",
        help=f"Min M2 probability delta (difference between L and S). Default: {default_config.m2_delta_threshold}",
    )
    # ▲▲▲ ここまで追加 ▲▲▲
    parser.add_argument(
        "--min-capital",
        type=float,
        default=default_config.min_capital_threshold,
        dest="min_capital",
        help=f"Min capital threshold. Default: {default_config.min_capital_threshold}",
    )
    # ★追加
    parser.add_argument(
        "--min-atr",
        type=float,
        default=default_config.min_atr_threshold,
        dest="min_atr",
        help=f"Minimum ATR threshold. Default: {default_config.min_atr_threshold}",
    )
    parser.add_argument(
        "--value-per-pip",
        type=float,
        default=default_config.value_per_pip,
        dest="value_per_pip",
        help=f"Value per lot per pip. Default: {default_config.value_per_pip}",
    )
    parser.add_argument(
        "--spread-pips",
        type=float,
        default=default_config.spread_pips,
        dest="spread_pips",
        help=f"Spread in pips for cost calculation. Default: {default_config.spread_pips}",
    )
    parser.add_argument(
        "--test",
        type=int,
        default=default_config.test_limit_partitions,
        metavar="N",
        dest="test_limit_partitions",
        help=f"Limit to first N partitions. Default: {default_config.test_limit_partitions} (all)",
    )
    parser.add_argument(
        "--max-positions",
        type=int,
        default=default_config.max_positions,
        dest="max_positions",
        help=f"Max concurrent positions. Default: {default_config.max_positions}",
    )

    # --- V5 新規: サーキットブレーカー引数 ---
    parser.add_argument(
        "--allow-simultaneous",
        action="store_true",
        help="Allow simultaneous Long/Short orders (default: prevented)",
    )
    parser.add_argument(
        "--max-consecutive-sl",
        type=int,
        default=default_config.max_consecutive_sl,
        dest="max_consecutive_sl",
        help=f"Max consecutive SLs before cooldown. Default: {default_config.max_consecutive_sl}",
    )
    parser.add_argument(
        "--cooldown-minutes",
        type=int,
        default=default_config.cooldown_minutes_after_sl,
        dest="cooldown_minutes",
        help=f"Cooldown minutes after max SLs. Default: {default_config.cooldown_minutes_after_sl}",
    )

    # V5 新規追加パラメータ (Long/Short独立)
    parser.add_argument(
        "--sl-long",
        type=float,
        default=default_config.sl_multiplier_long,
        dest="sl_long",
    )
    parser.add_argument(
        "--pt-long",
        type=float,
        default=default_config.pt_multiplier_long,
        dest="pt_long",
    )
    parser.add_argument(
        "--sl-short",
        type=float,
        default=default_config.sl_multiplier_short,
        dest="sl_short",
    )
    parser.add_argument(
        "--pt-short",
        type=float,
        default=default_config.pt_multiplier_short,
        dest="pt_short",
    )
    parser.add_argument(
        "--td-long", type=float, default=default_config.td_minutes_long, dest="td_long"
    )
    parser.add_argument(
        "--td-short",
        type=float,
        default=default_config.td_minutes_short,
        dest="td_short",
    )

    parser.add_argument(
        "--oof-long",
        type=str,
        default=str(default_config.oof_long_path),
        dest="oof_long_path",
        help=f"Path to Long OOF predictions.",
    )
    parser.add_argument(
        "--oof-short",
        type=str,
        default=str(default_config.oof_short_path),
        dest="oof_short_path",
        help=f"Path to Short OOF predictions.",
    )

    # ─── 期間フィルタと初期資本 ───
    parser.add_argument(
        "--initial-capital",
        type=float,
        default=default_config.initial_capital,
        dest="initial_capital",
        help=f"Initial capital. Default: {default_config.initial_capital}",
    )
    parser.add_argument(
        "--start-date",
        type=str,
        default=None,
        dest="start_date",
        help="Start date filter (YYYY-MM-DD, UTC, inclusive). Default: None (no filter)",
    )
    parser.add_argument(
        "--end-date",
        type=str,
        default=None,
        dest="end_date",
        help="End date filter (YYYY-MM-DD, UTC, inclusive). Default: None (no filter)",
    )

    args = parser.parse_args()

    config = BacktestConfig(
        initial_capital=args.initial_capital,
        start_date=args.start_date,
        end_date=args.end_date,
        auto_lot_base_capital=args.auto_lot_base_capital,
        auto_lot_size_per_base=args.auto_lot_size_per_base,
        # [BUGFIX] 旧: default_config.* を直接渡していたため --fixed-risk-pct が無効だった。
        #   ロットサイズ → 資産曲線 → 破産判定まで直撃する無音バグ。
        use_fixed_risk=(not args.no_fixed_risk) and USER_PARAMS["use_fixed_risk"],
        fixed_risk_percent=args.fixed_risk_pct,
        base_leverage=args.base_leverage,
        m2_proba_threshold=args.m2_th,
        m2_delta_threshold=args.m2_delta,  # ★これを追加！
        test_limit_partitions=args.test_limit_partitions,
        oof_mode=True,
        min_capital_threshold=args.min_capital,
        min_atr_threshold=args.min_atr,
        max_atr_threshold=args.max_atr,
        max_baseline_atr=args.max_baseline_atr,
        min_baseline_atr=args.min_baseline_atr,
        min_baseline_ratio=args.min_baseline_ratio,
        baseline_ratio_lookback_days=args.baseline_days,
        min_sar_threshold=args.min_sar,
        sar_lookback_days=args.sar_days,
        value_per_pip=args.value_per_pip,
        spread_pips=args.spread_pips,
        max_positions=args.max_positions,
        prevent_simultaneous_orders=not args.allow_simultaneous,
        max_consecutive_sl=args.max_consecutive_sl,
        cooldown_minutes_after_sl=args.cooldown_minutes,
        sl_multiplier_long=args.sl_long,
        pt_multiplier_long=args.pt_long,
        sl_multiplier_short=args.sl_short,
        pt_multiplier_short=args.pt_short,
        td_minutes_long=args.td_long,
        td_minutes_short=args.td_short,
        oof_long_path=Path(args.oof_long_path),
        oof_short_path=Path(args.oof_short_path),
    )

    if config.base_leverage < 1.0:
        parser.error("--base-leverage must be >= 1.0.")
    if config.value_per_pip <= 0:
        parser.error("--value-per-pip must be greater than 0.")
    if config.spread_pips < 0:
        parser.error("--spread-pips cannot be negative.")
    if config.sl_multiplier_long <= 0 or config.sl_multiplier_short <= 0:
        parser.error("SL multipliers must be greater than 0.")
    if config.use_fixed_risk and not (0.0 < config.fixed_risk_percent < 1.0):
        parser.error("fixed_risk_percent must be in (0, 1), e.g. 0.02 for 2%.")

    # ======================================================================
    # [GEOMETRY-SYNC] ラベル幾何との突合 — シミュレーション開始前に必ず実行
    # ======================================================================
    geometry_check = verify_geometry_against_labeling(
        config, strict=bool(args.strict_geometry)
    )
    GEOMETRY_CHECK_RESULT = geometry_check

    # [TD-RESIM] ラベル側の TD / 行動地平をグローバルへ反映
    _lab = geometry_check.get("labeling") or {}
    LABEL_TD_MINUTES = float(
        _lab.get("td_minutes_long", _LABEL_GEOMETRY_FALLBACK["td_minutes_long"])
    )
    ACTION_HORIZON_MIN = (
        float(
            _lab.get(
                "action_horizon_sec",
                _LABEL_GEOMETRY_FALLBACK["action_horizon_sec"],
            )
        )
        / 60.0
    )
    logging.info(
        f"[TD-RESIM] ラベル TD = {LABEL_TD_MINUTES:g} 分 (エントリー起点) / "
        f"行動地平 = {ACTION_HORIZON_MIN:g} 分 (L → エントリー)"
    )
    if (
        config.td_minutes_long < LABEL_TD_MINUTES - 1e-9
        or config.td_minutes_short < LABEL_TD_MINUTES - 1e-9
    ):
        logging.info(
            f"[TD-RESIM] TD 短縮モード: L={config.td_minutes_long:g} / "
            f"S={config.td_minutes_short:g} 分。経過が新 TD を超えた PT/SL は"
            " すべて close_future による強制決済に再分類します。"
        )

    # ---------------- 実効設定の明示 ----------------
    logging.info("-" * 68)
    logging.info("[EFFECTIVE SETTINGS]  (USER_PARAMS + CLI 上書き後の最終値)")
    if str(config.signal_source).lower() == "rule":
        logging.info(
            "  signal  : ★RULE (脳不使用) — "
            f"{config.rule_er_column} >= {config.rule_er_min:g}"
            + (f" かつ < {config.rule_er_max:g}" if config.rule_er_max > 0 else "")
            + f" / {config.rule_d_abs_min:g} <= |{config.rule_d_column}| < "
            f"{config.rule_d_abs_max:g} / 方向={config.rule_direction}"
            f"  (m2_th / m2_delta は無効)"
        )
    else:
        logging.info("  signal  : MODEL (M2 OOF 確率でゲート)")
    logging.info(
        f"  barrier : PT {config.pt_multiplier_long}/{config.pt_multiplier_short}"
        f" | SL {config.sl_multiplier_long}/{config.sl_multiplier_short}"
        f" | TD {config.td_minutes_long}/{config.td_minutes_short} min   (L/S)"
        + (
            ""
            if config.use_barriers
            else "   ★TIME-EXIT (PT/SL 無効・必ず TD で成行決済)"
        )
    )
    logging.info(
        f"  gates   : m2_th={config.m2_proba_threshold}"
        f" | m2_delta={config.m2_delta_threshold}"
        f" | atr_ratio=[{config.min_atr_threshold:g}, "
        f"{('∞' if config.max_atr_threshold <= 0 else format(config.max_atr_threshold, 'g'))})"
        f" | atr_value=[{config.min_atr_value:g}, "
        f"{('∞' if config.max_atr_value <= 0 else format(config.max_atr_value, 'g'))}) USD"
        f" | baseline_atr=[{config.min_baseline_atr:g}, "
        f"{('∞' if config.max_baseline_atr <= 0 else format(config.max_baseline_atr, 'g'))}) USD"
        f" | min_baseline_ratio={config.min_baseline_ratio}"
        f" | min_sar={config.min_sar_threshold}"
    )
    logging.info(
        (
            "  sizing  : ★FIXED-LOT "
            + f"{config.fixed_lot_size:g} lot (エッジ測定モード)"
            if config.use_fixed_lot
            else ""
        )
        or f"  sizing  : fixed_risk={config.use_fixed_risk}"
        f" ({config.fixed_risk_percent * 100:.2f}%)"
        f" | leverage={config.base_leverage} | max_positions={config.max_positions}"
    )
    logging.info(
        f"  cost    : spread_pips={config.spread_pips}"
        f" | value_per_pip={config.value_per_pip}"
    )
    logging.info(
        f"  capital : initial={config.initial_capital}"
        f" | period={config.start_date or 'ALL'} 〜 {config.end_date or 'ALL'}"
    )
    if abs(config.pt_multiplier_long - config.sl_multiplier_long) < 1e-9:
        logging.info(
            "  note    : 対称バリア (pt=sl) のため基底率≈50%。"
            "p_short≈1−p_long となり delta≈|2·p_long−1| なので、"
            "m2_delta は m2_th に支配されます (m2_th 単独で振ること)。"
        )
    logging.info("-" * 68)

    simulator = BacktestSimulator(config)

    # =========================================================
    # 推論モード選択: M1単独 or M2 (Two-Brain)
    # =========================================================
    print("\n" + "=" * 50)
    print("  🧠 推論モードを選択してください:")
    print("    [1] M2モード (通常: Two-Brain) [デフォルト]")
    print("    [2] M1モード (実験: M1単独)")
    print("=" * 50)
    mode_ans = input("選択 [1/2, Enterでデフォルト]: ").strip()

    if mode_ans == "2":
        config.oof_long_path = S7_M1_OOF_PREDICTIONS_LONG
        config.oof_short_path = S7_M1_OOF_PREDICTIONS_SHORT
        inference_mode = "M1"
        active_cache_path = S7_BACKTEST_CACHE_M1
        oof_ref_long = S7_M1_OOF_PREDICTIONS_LONG
        oof_ref_short = S7_M1_OOF_PREDICTIONS_SHORT
        logging.info("🔬 [M1モード] M1単独OOFを使用します。")
    else:
        inference_mode = "M2"
        active_cache_path = S7_BACKTEST_CACHE_M2
        oof_ref_long = S7_M2_OOF_PREDICTIONS_LONG
        oof_ref_short = S7_M2_OOF_PREDICTIONS_SHORT
        logging.info("🧠 [M2モード] Two-Brain OOFを使用します。")

    # モードに応じて結果フォルダを生成して出力パスを設定
    import datetime as _dt

    _now_str = _dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    _risk_pct = config.fixed_risk_percent * 100
    # [GEOMETRY-SYNC] 幾何とフィルタをフォルダ名に焼き込む。
    #   旧実装は Th/D/R しか含めず、pt/sl/td 違いのレポートが区別できなかった。
    #   幾何不一致のまま続行した場合は _GEOMISMATCH を付す。
    _geo_tag = (
        f"_PT{config.pt_multiplier_long:g}"
        f"_SL{config.sl_multiplier_long:g}"
        f"_TD{config.td_minutes_long:g}"
    )
    _flt_tag = ""
    if config.min_baseline_atr > 0:
        _flt_tag += f"_bATR{config.min_baseline_atr:g}"
    if config.min_baseline_ratio > 0:
        _flt_tag += f"_bRatio{config.min_baseline_ratio:g}"
    if config.min_sar_threshold > 0:
        _flt_tag += f"_SAR{config.min_sar_threshold:g}"
    if config.min_atr_threshold > 0 or config.max_atr_threshold > 0:
        _hi = (
            "inf" if config.max_atr_threshold <= 0 else f"{config.max_atr_threshold:g}"
        )
        _flt_tag += f"_aATR{config.min_atr_threshold:g}to{_hi}"
    if config.max_baseline_atr > 0:
        _flt_tag += f"_bATRmax{config.max_baseline_atr:g}"
    _mismatch_tag = "_GEOMISMATCH" if geometry_check.get("status") == "mismatch" else ""
    _mode_tag = (
        "RULE" if str(config.signal_source).lower() == "rule" else inference_mode
    )
    _folder_name = (
        f"{_mode_tag}_{_now_str}"
        f"{_geo_tag}"
        f"_Th{config.m2_proba_threshold}"
        f"_D{config.m2_delta_threshold}"
        f"_R{_risk_pct:g}"
        f"{_flt_tag}"
        f"{_mismatch_tag}"
    )
    _result_dir = S7_BACKTEST_SIM_RESULTS / _folder_name
    _result_dir.mkdir(parents=True, exist_ok=True)

    FINAL_REPORT_PATH = _result_dir / f"final_backtest_report_v5_{inference_mode}.json"
    EQUITY_CURVE_PATH = _result_dir / f"equity_curve_v5_{inference_mode}.png"
    logging.info(f"出力先フォルダ: {_result_dir}")

    # =========================================================
    # キャッシュ管理: モード別キャッシュファイルを使い分け
    # =========================================================
    def load_or_generate_cache() -> tuple:
        if active_cache_path.exists():
            cache_mtime = active_cache_path.stat().st_mtime
            oof_mtime = max(
                oof_ref_long.stat().st_mtime,
                oof_ref_short.stat().st_mtime,
            )
            stale = cache_mtime < oof_mtime

            print(f"\n[{inference_mode}] キャッシュが存在します: {active_cache_path}")
            if stale:
                print(
                    "  ⚠️  キャッシュがOOFより古い可能性があります。再生成を推奨します。"
                )
            print("  [y] このまま使用する")
            print("  [r] 削除して再生成する")
            ans = input("選択 [y/r]: ").strip().lower()

            if ans == "r":
                active_cache_path.unlink()
                logging.info(
                    f"[{inference_mode}] キャッシュを削除しました。再生成します..."
                )
            else:
                logging.info(f"[{inference_mode}] キャッシュを読み込んでいます...")
                with open(active_cache_path, "rb") as f:
                    data = pickle.load(f)
                logging.info(f"[{inference_mode}] キャッシュ読み込み完了。")
                return data

        logging.info(
            f"[{inference_mode}] キャッシュがありません。データを生成します..."
        )
        data = simulator.preload_data()
        logging.info(
            f"[{inference_mode}] データ生成完了。キャッシュに保存しています..."
        )
        active_cache_path.parent.mkdir(parents=True, exist_ok=True)  # ← これを追加
        with open(active_cache_path, "wb") as f:
            pickle.dump(data, f)
        logging.info(f"[{inference_mode}] キャッシュ保存完了: {active_cache_path}")
        return data

    preloaded_data = load_or_generate_cache()

    # [CACHE-GUARD] 必要列の検証。古いキャッシュなら自動で作り直す。
    if not validate_preload_columns(preloaded_data, config, active_cache_path):
        try:
            active_cache_path.unlink()
        except OSError:
            pass
        logging.info("[CACHE-GUARD] キャッシュを再生成します...")
        preloaded_data = simulator.preload_data()
        active_cache_path.parent.mkdir(parents=True, exist_ok=True)
        with open(active_cache_path, "wb") as f:
            pickle.dump(preloaded_data, f)
        logging.info(f"[CACHE-GUARD] 再生成して保存しました: {active_cache_path}")
        validate_preload_columns(preloaded_data, config, active_cache_path)

    # ─── キャッシュ load 後の期間フィルタ ───
    # キャッシュには全期間データが入っているため、--start-date / --end-date が
    # 指定された場合は preloaded_dict と partitions_to_process を切り出す
    if config.start_date is not None or config.end_date is not None:
        preloaded_dict, partitions_to_process = preloaded_data
        start_d = (
            dt.date.fromisoformat(config.start_date) if config.start_date else None
        )
        end_d = dt.date.fromisoformat(config.end_date) if config.end_date else None

        # filter preloaded_dict (Dict[date, DataFrame])
        filtered_dict = {
            d: df
            for d, df in preloaded_dict.items()
            if (start_d is None or d >= start_d) and (end_d is None or d <= end_d)
        }

        # filter partitions_to_process (pl.DataFrame with "date" col)
        filter_expr = pl.lit(True)
        if start_d is not None:
            filter_expr = filter_expr & (pl.col("date") >= start_d)
        if end_d is not None:
            filter_expr = filter_expr & (pl.col("date") <= end_d)
        filtered_partitions = partitions_to_process.filter(filter_expr)

        logging.info(
            f"[期間フィルタ post-cache] preloaded_dict: "
            f"{len(preloaded_dict)} → {len(filtered_dict)} dates"
        )
        logging.info(
            f"[期間フィルタ post-cache] partitions: "
            f"{len(partitions_to_process)} → {len(filtered_partitions)} dates"
        )

        if len(filtered_dict) == 0:
            raise ValueError(
                f"期間フィルタ後に対象データが 0 件です。"
                f"start_date={config.start_date} end_date={config.end_date} を"
                f"確認してください。キャッシュ内データの範囲: "
                f"{min(preloaded_dict.keys())} 〜 {max(preloaded_dict.keys())}"
            )

        preloaded_data = (filtered_dict, filtered_partitions)

    simulator.run(preloaded_data=preloaded_data)
